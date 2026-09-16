import os, json, pickle, queue, threading, time, sys, struct, subprocess, ctypes, fcntl, shutil, glob, optax, jax, argparse
import numpy as np, jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import linen as nn
from jax.flatten_util import ravel_pytree
from jax.extend.core import Literal
from scipy.io import wavfile
from functools import partial, reduce

from processing import raw_memmap_loader, get_full_track_data, get_cached_metadata

jax.config.update("jax_default_matmul_precision", "float32")
jax.config.update("jax_enable_x64", False)

CONCURRENT_MODELS = 3
CURR_CKPT, PREV_CKPT = "checkpoints/checkpoint_bundle.pickle", "checkpoints/checkpoint_bundle_prev.pickle"
CKPT_LOCK_PATH, GRAD_LOCK_PATH = "checkpoints/checkpoint.lock", "data/shared_gradients.lock"
META_LOCK_PATH = "checkpoints/meta.lock"
NTK_LOCK_PATH = "checkpoints/ntk.lock"
INIT_SEED_PATH = "checkpoints/init_seed.lock"
LIE_PARAMS = {"query", "key", "value"}

# -----------------------------------------------------------------------------
# 1. Transformer Architecture, Multimodal Mixing & Diffusion Core
# -----------------------------------------------------------------------------
rms_norm = lambda x, scale, eps=1e-5: x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps) * scale
apply_rope = lambda x, freq=10000.0: (lambda c, s: x.at[..., 0::2].set(x[..., 0::2]*c - x[..., 1::2]*s).at[..., 1::2].set(x[..., 0::2]*s + x[..., 1::2]*c))(jnp.cos(jnp.arange(x.shape[-2], dtype=jnp.float32)[None, :, None] / (freq ** (jnp.arange(0, x.shape[-1], 2, dtype=jnp.float32) / x.shape[-1]))), jnp.sin(jnp.arange(x.shape[-2], dtype=jnp.float32)[None, :, None] / (freq ** (jnp.arange(0, x.shape[-1], 2, dtype=jnp.float32) / x.shape[-1]))))
scaled_dot_product_attention = lambda q, k, v, mask=None: jnp.matmul(jax.nn.softmax(jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * (1.0 / jnp.sqrt(q.shape[-1])) + (mask if mask is not None else 0.0), axis=-1), v)
clamp_frobenius_norm = lambda w, steps=2: reduce(lambda mat, _: jnp.where((n := jnp.sqrt(jnp.sum(mat * mat))) > 1.0, mat / n, mat), range(steps), w)

def multimodal_mixing_layer(x, modality_type, params, target_dim=1024, comp_dim=512):
    if modality_type == "audio":
        patch_dim = 441
        B, T, L = x.shape
        num_patches = L // patch_dim
        encoded = jax.nn.gelu(x.reshape(B, T, num_patches, patch_dim) @ params['audio_encoder']) @ params['down_proj_2']
        return encoded, num_patches
    elif modality_type == "token":
        B, T, D = x.shape
        token_enc = params.get('token_encoder', params['audio_encoder'][:D, :comp_dim])
        encoded = jax.nn.gelu(x @ token_enc) @ params['down_proj_2']
        return encoded, 1
    else:
        raise ValueError(f"Unsupported modality type: {modality_type}")

def combined_audio_loss(pred, target, mask=None, n_fft=1024):
    if mask is not None:
        pred = pred * mask[:, :, None]
        target = target * mask[:, :, None]
    pf, nf = map(lambda x: jnp.pad(x.reshape(-1), (0, (-x.size) % n_fft)), (pred, target))
    xf, yf = map(lambda arr: jnp.abs(jnp.fft.rfft(arr.reshape(-1, n_fft))), (pf, nf))
    return jnp.mean(jnp.abs(xf - yf)) + jnp.mean(jnp.square(jnp.log(xf + 1e-5) - jnp.log(yf + 1e-5))) + 0.5 * jnp.mean(jnp.abs(pred - target))

def compute_empirical_ntk(params, batch_x, scales, bpms, stems, steps, sigmas, mask=None, modality_type="audio"):
    model_fn = lambda p: gpt_forward(p, jnp.cos(sigmas * jnp.pi / 2.0)[:, :, None] * batch_x + jnp.sin(sigmas * jnp.pi / 2.0)[:, :, None] * jax.random.normal(jax.random.PRNGKey(0), batch_x.shape), (scales, bpms, stems, steps, sigmas), mask, modality_type=modality_type)
    _, vjp_fun = jax.vjp(model_fn, params)
    dummy = model_fn(params)
    
    num_slices = min(128, dummy.size)
    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    
    J_rows = []
    for idx in np.linspace(0, dummy.size - 1, num_slices, dtype=int):
        vjp_out = vjp_fun(jnp.zeros(dummy.shape).at[idx].set(1.0).reshape(dummy.shape))[0]
        flat_grad = jnp.concatenate([jnp.ravel(v) for v in jax.tree_util.tree_leaves(vjp_out)])
        J_rows.append(flat_grad / jnp.sqrt(float(total_params)))
        
    J = jnp.stack(J_rows)
    ntk = J @ J.T
    ntk_stable = ntk + 1e-4 * jnp.eye(ntk.shape[0])
    
    return {
        "matrix": ntk, 
        "condition_number": float(jnp.linalg.cond(ntk_stable)), 
        "trace": float(jnp.trace(ntk) / float(num_slices))
    }

def gpt_forward(params, x, cond, mask=None, target_dim=44100, patch_dim=441, num_patches=100, n_heads=16, modality_type="audio"):
    scale, bpm, stem, step_indices, sigma_t = cond
    B, T = x.shape[0], x.shape[1]
    
    x = rms_norm(x, params['input_rms_scale'])
    
    encoded, num_p = multimodal_mixing_layer(x, modality_type, params)
    C, bt = encoded.shape[-1], B * T
    head_dim = C // n_heads
    
    p_mask = None
    if mask is not None:
        patch_mask_2d = jnp.ones((B, T, num_p, num_p), dtype=bool) & mask[:, :, None, None]
        p_mask = jnp.where(patch_mask_2d.reshape(bt, num_p, num_p)[:, None, :, :], 0.0, -1e9)

    patch_seq = encoded.reshape(bt, num_p, C)
    q_p, k_p, v_p = map(lambda k: (patch_seq @ params[k]).reshape(bt, num_p, n_heads, head_dim).transpose(0, 2, 1, 3), ['query', 'key', 'value'])
    
    attn_out_p = scaled_dot_product_attention(apply_rope(q_p), apply_rope(k_p), v_p, p_mask).transpose(0, 2, 1, 3).reshape(bt, num_p, C)
    h_p = rms_norm(attn_out_p + patch_seq, params['rms_scale_1'])
    h_p = rms_norm(h_p + jax.nn.gelu(h_p @ params['ff_1']) @ params['ff_2'], params['rms_scale_2'])
    
    h_frames = h_p.reshape(B, T, num_p, C)
    ft = jnp.mean(h_frames, axis=2) + params['t_pos_emb'][:T][None, :, :]
    q_t, k_t, v_t = map(lambda k: (ft @ params[k]).reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3), ['t_query', 't_key', 't_value'])
    
    t_mask = jnp.where(mask[:, None, :] & mask[:, :, None], 0.0, -1e9)[:, None, :, :] if mask is not None else None
    attn_out_t = scaled_dot_product_attention(apply_rope(q_t), apply_rope(k_t), v_t, t_mask).transpose(0, 2, 1, 3).reshape(B, T, C)
    h_t = rms_norm(attn_out_t + ft, params['t_rms_scale'])
    
    base_cond = (params['time_emb'][step_indices] + params['scale_emb'][jnp.clip(scale, 0, 127)][:, None, :] + (((bpm - 120.0) / 60.0)[:, None] @ params['bpm_proj'])[:, None, :] + params['stem_emb'][stem][:, None, :])[:, :, None, :]
    h_frames = h_frames + jnp.expand_dims(h_t, 2) + base_cond + sigma_t[:, :, None, None] * params['sigma_emb'][None, None, None, :]
    
    out_encoded = rms_norm(jax.nn.gelu(h_frames @ params['up_proj_1']) @ params['up_proj_2'], params['out_rms_scale']) @ params['out_proj']
    if modality_type == "audio":
        return out_encoded.reshape(B, T, num_p * patch_dim)
    return out_encoded

xavier_normal = lambda key, shape: jax.random.normal(key, shape) * jnp.sqrt(2.0 / (shape[-2] if len(shape) >= 2 else shape[0] + shape[-1] if len(shape) >= 2 else shape[0]))

def init_params(key, dim=1024, patch_dim=441, comp_dim=512, steps=50):
    keys = jax.random.split(key, 23)
    return {
        'audio_encoder': xavier_normal(keys[0], (patch_dim, comp_dim)), 'down_proj_2': xavier_normal(keys[1], (comp_dim, dim)),
        'token_encoder': xavier_normal(keys[22], (patch_dim, comp_dim)),
        'query': jax.random.orthogonal(keys[2], dim), 'key': jax.random.orthogonal(keys[3], dim), 'value': jax.random.orthogonal(keys[4], dim),
        't_query': jax.random.orthogonal(keys[5], dim), 't_key': jax.random.orthogonal(keys[6], dim), 't_value': jax.random.orthogonal(keys[7], dim),
        'ff_1': xavier_normal(keys[8], (dim, dim * 4)), 'ff_2': xavier_normal(keys[9], (dim * 4, dim)),
        'up_proj_1': xavier_normal(keys[10], (dim, comp_dim)), 'up_proj_2': xavier_normal(keys[11], (comp_dim, comp_dim)), 'out_proj': xavier_normal(keys[19], (comp_dim, patch_dim)),
        'rms_scale_1': jnp.ones((dim,)), 'rms_scale_2': jnp.ones((dim,)), 't_rms_scale': jnp.ones((dim,)), 'out_rms_scale': jnp.ones((comp_dim,)),
        'input_rms_scale': jnp.ones((1,)),
        'scale_emb': xavier_normal(keys[12], (128, dim)), 'bpm_proj': xavier_normal(keys[13], (1, dim)), 'stem_emb': xavier_normal(keys[14], (2, dim)),
        'time_emb': xavier_normal(keys[15], (steps, dim)), 't_pos_emb': xavier_normal(keys[16], (512, dim)), 'sigma_emb': xavier_normal(keys[18], (dim,))
    }

# -----------------------------------------------------------------------------
# 2. Meta-Preconditioner & Synchronized Spectral Daemon Components
# -----------------------------------------------------------------------------
align_drift = lambda w_new, w_old: jnp.linalg.norm(ravel_pytree(w_new)[0] - ravel_pytree(w_old)[0]) / (jnp.linalg.norm(ravel_pytree(w_new)[0]) + 1e-6)

class SpectralPreconditionerMLP(nn.Module):
    fixed_dim: int = 1025

    @nn.compact
    def __call__(self, x, target_dim=None):
        if x.shape[-1] != self.fixed_dim:
            x = jax.image.resize(x, (self.fixed_dim,), 'linear')
        x = nn.Dense(self.fixed_dim)(x)
        x = nn.gelu(nn.Dense(512)(x))
        x = nn.gelu(nn.Dense(512)(x))
        scales = jax.nn.sigmoid(nn.Dense(self.fixed_dim)(x)) * 2.0
        if target_dim is not None and target_dim != self.fixed_dim:
            scales = jax.image.resize(scales, (target_dim,), 'linear')
        return scales

class MetaDashboard:
    def __init__(self):
        self.enabled = True
        try:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(7, 4))
            self.losses = []
        except Exception:
            self.enabled = False

    def update(self, loss):
        if not self.enabled: return
        try:
            self.losses.append(loss)
            self.ax.clear()
            self.ax.plot(self.losses, color='#8b5cf6', label='Meta-Loss (Curvature Variance)')
            self.ax.set_title("Manifold-Aware Spectral Preconditioner")
            plt.draw(); plt.pause(0.01)
        except Exception:
            self.enabled = False

def _load_params_from_bundle(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["params"] if isinstance(obj, dict) and "params" in obj else obj

def get_meta_preconditioner(grads, loss=None):
    meta_ckpt = "checkpoints/meta_preconditioner.pickle"
    if not os.path.exists(meta_ckpt): return None
    ntk_files = sorted(glob.glob("ntk_logs/ntk_step_*.npy"))
    if not ntk_files: return None

    drift = 0.0
    if os.path.exists(CURR_CKPT) and os.path.exists(PREV_CKPT):
        try:
            w_new = _load_params_from_bundle(CURR_CKPT)
            w_old = _load_params_from_bundle(PREV_CKPT)
            drift = align_drift(w_new, w_old)
        except Exception:
            pass

    loss_val = 0.0 if loss is None else float(loss)
    raw_jac = jnp.array(np.load(ntk_files[-1])).flatten()
    raw_jac = jnp.pad(raw_jac, (0, max(0, 1024 - raw_jac.shape[0])))[:1024]

    ntk_data = jnp.concatenate([raw_jac, jnp.array([drift, loss_val])])
    
    with open(META_LOCK_PATH, "a+") as mf:
        fcntl.flock(mf, fcntl.LOCK_EX)
        try:
            with open(meta_ckpt, "rb") as f:
                meta_params = pickle.load(f)
        finally:
            fcntl.flock(mf, fcntl.LOCK_UN)

    flat_grads, treedef = ravel_pytree(grads)
    model = SpectralPreconditionerMLP()
    scales = model.apply(meta_params, ntk_data, target_dim=flat_grads.shape[0])
    return treedef(flat_grads * scales)

@jax.jit
def train_meta_step(params, opt_state, tx, inputs):
    def loss_fn(p):
        pred_scales = SpectralPreconditionerMLP().apply(p, inputs)
        jac_diag = inputs[:1024]
        effective_curvature = pred_scales[:jac_diag.shape[0]] * jac_diag
        mean_curv = jnp.mean(effective_curvature)
        return jnp.var(effective_curvature) + 0.1 * jnp.square(mean_curv - 1.0)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, new_opt_state = tx.update(grads, opt_state, params)
    return loss, optax.apply_updates(params, updates), new_opt_state

def run_meta_daemon():
    os.makedirs("ntk_logs", exist_ok=True)
    meta_ckpt = "checkpoints/meta_preconditioner.pickle"
    dashboard = MetaDashboard()
    params, opt_state, tx = None, None, optax.adam(1e-4)

    while True:
        ntk_files = sorted(glob.glob("ntk_logs/ntk_step_*.npy"))
        if len(ntk_files) > 0:
            try:
                raw_jac = jnp.array(np.load(ntk_files[-1])).flatten()
                raw_jac = jnp.pad(raw_jac, (0, max(0, 1024 - raw_jac.shape[0])))[:1024]
                ntk_data = jnp.concatenate([raw_jac, jnp.array([0.0, 0.0])])

                if params is None:
                    dummy_input = jnp.zeros(1025)
                    params = SpectralPreconditionerMLP().init(jax.random.PRNGKey(0), dummy_input)
                    opt_state = tx.init(params)

                loss, params, opt_state = train_meta_step(params, opt_state, tx, ntk_data)
                dashboard.update(float(loss))
                
                with open(META_LOCK_PATH, "a+") as mf:
                    fcntl.flock(mf, fcntl.LOCK_EX)
                    try:
                        with open(meta_ckpt, "wb") as f:
                            pickle.dump(params, f)
                    finally:
                        fcntl.flock(mf, fcntl.LOCK_UN)
            except Exception:
                pass
        time.sleep(5)

# -----------------------------------------------------------------------------
# 3. Synchronized Checkpoint & Gradient Utilities
# -----------------------------------------------------------------------------
load_checkpoint_safely = lambda: (lambda clf: (fcntl.flock(clf, fcntl.LOCK_EX), res := (pickle.load(open(CURR_CKPT, "rb")) if os.path.exists(CURR_CKPT) else None), fcntl.flock(clf, fcntl.LOCK_UN), res)[-1])(open(CKPT_LOCK_PATH, "a+"))

def get_or_create_init_seed(default_seed=42):
    with open(INIT_SEED_PATH, "a+") as sf:
        fcntl.flock(sf, fcntl.LOCK_EX)
        try:
            sf.seek(0)
            content = sf.read().strip()
            seed = int(content) if content else default_seed
            if not content:
                sf.seek(0); sf.truncate(); sf.write(str(seed)); sf.flush()
        finally:
            fcntl.flock(sf, fcntl.LOCK_UN)
        return seed

def push_and_pull_gradients(optimizer, local_grads, loss_val, global_step, expected_version, expected_step, accumulation_steps=4, single_track_params_list=None):
    with open(CKPT_LOCK_PATH, "a+") as clf, open(GRAD_LOCK_PATH, "a+b") as gf:
        fcntl.flock(clf, fcntl.LOCK_EX); fcntl.flock(gf, fcntl.LOCK_EX)
        try:
            shared = pickle.load(open(GRAD_LOCK_PATH, "rb")) if os.path.exists(GRAD_LOCK_PATH) and os.path.getsize(GRAD_LOCK_PATH) > 0 else {"accumulated_grads": None, "count": 0, "version": expected_version}
            shared["accumulated_grads"] = jax.tree_util.tree_map(lambda x, y: x + y, shared["accumulated_grads"], jax.device_get(local_grads)) if shared["accumulated_grads"] is not None else jax.device_get(local_grads)
            shared["count"] += 1
            apply_update = shared["count"] >= accumulation_steps
            
            if apply_update:
                avg_grads = jax.tree_util.tree_map(lambda x: x / shared["count"], shared["accumulated_grads"])
                shared.update({"accumulated_grads": None, "count": 0, "version": shared["version"] + 1})
            
            pickle.dump(shared, open(GRAD_LOCK_PATH, "wb"))
            fcntl.flock(gf, fcntl.LOCK_UN)
            
            if apply_update:
                if os.path.exists(CURR_CKPT):
                    shutil.copy(CURR_CKPT, PREV_CKPT)
                bundle = pickle.load(open(CURR_CKPT, "rb"))
                updates, opt_state = optimizer.update(avg_grads, bundle.get("opt_state", optimizer.init(bundle["params"])), bundle["params"])
                new_params = optax.apply_updates(bundle["params"], updates)
                
                if single_track_params_list and len(single_track_params_list) > 0:
                    blend_weight = 0.1 / len(single_track_params_list)
                    for st_params in single_track_params_list:
                        new_params = jax.tree_util.tree_map(lambda np_val, st_val: (1.0 - blend_weight) * np_val + blend_weight * st_val, new_params, st_params)

                new_bundle = {"params": new_params, "ema_params": jax.tree_util.tree_map(lambda ep, p: 0.9999 * ep + 0.0001 * p, bundle.get("ema_params", bundle["params"]), new_params), "opt_state": opt_state, "version": shared["version"], "global_step": expected_step + 1}
                pickle.dump(new_bundle, open(CURR_CKPT, "wb"))
                return new_bundle["params"], new_bundle["ema_params"], new_bundle["version"], True
        finally:
            fcntl.flock(clf, fcntl.LOCK_UN)
    bundle = load_checkpoint_safely()
    return bundle["params"], bundle.get("ema_params", bundle["params"]), bundle.get("version", expected_version), False

def run_single_track_worker(worker_idx, initial_params, quantization="fp32"):
    optimizer = optax.adam(1e-4)
    params = initial_params
    opt_state = optimizer.init(params)
    
    track_data, scale, bpm, stem, steps, sigmas, yt_id = get_full_track_data(quantization=quantization)
    step = 0
    while True:
        scales = np.array([scale], dtype=np.int32)
        bpms = np.array([bpm], dtype=np.float32)
        stems = np.array([stem], dtype=np.int32)
        cond = (scales, bpms, stems, steps, sigmas)
        
        loss_val, grads = jax.value_and_grad(lambda p: combined_audio_loss(gpt_forward(p, jnp.cos(sigmas * np.pi / 2.0)[:, :, None] * track_data + jnp.sin(sigmas * np.pi / 2.0)[:, :, None] * jax.random.normal(jax.random.PRNGKey(step), track_data.shape), cond, modality_type="audio"), jax.random.normal(jax.random.PRNGKey(step), track_data.shape)))(params)
        
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        if float(loss_val) <= 1e-6: break
        step += 1
    return params

# -----------------------------------------------------------------------------
# 4. Main Execution & Synchronized Daemon Interface
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synchronized Multimodal JAX Diffusion Transformer Daemon")
    parser.add_argument("--train", action="store_true", help="Launch the synchronized training daemon loop")
    parser.add_argument("--ckpt-mix", type=str, default="checkpoints/checkpoint_bundle.pickle", 
                        help="Comma-separated list of checkpoint bundles to mix/load")
    parser.add_argument("--quantization", type=str, default="fp32", choices=["fp32", "fp16", "int8", "int4"],
                        help="Data ingestion quantization fidelity tier")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("checkpoints/ntk", exist_ok=True)
    os.makedirs("ntk_logs", exist_ok=True)
    
    optimizer = optax.adam(1e-4)
    ckpt_paths = [p.strip() for p in args.ckpt_mix.split(",")]
    primary_ckpt = ckpt_paths[0]
    
    # Atomic initialization of master seed and checkpoint bundle across concurrent containers
    seed_key = get_or_create_init_seed(42)
    with open(CKPT_LOCK_PATH, "a+") as clf:
        fcntl.flock(clf, fcntl.LOCK_EX)
        try:
            if not os.path.exists(primary_ckpt):
                init_p = init_params(jax.random.PRNGKey(seed_key))
                pickle.dump({"params": init_p, "ema_params": init_p, "opt_state": optimizer.init(init_p), "version": 0, "global_step": 0}, open(primary_ckpt, "wb"))
        finally:
            fcntl.flock(clf, fcntl.LOCK_UN)

    bundle = load_checkpoint_safely()
    params = bundle["params"] if isinstance(bundle, dict) and "params" in bundle else bundle

    if len(ckpt_paths) > 1:
        print(f"[Daemon] Blending checkpoint mix from paths: {ckpt_paths}")
        for alt_path in ckpt_paths[1:]:
            if os.path.exists(alt_path):
                alt_p = _load_params_from_bundle(alt_path)
                params = jax.tree_util.tree_map(lambda p1, p2: 0.5 * p1 + 0.5 * p2, params, alt_p)

    ema_params = bundle.get("ema_params", params) if isinstance(bundle, dict) else params
    version = bundle.get("version", 0) if isinstance(bundle, dict) else 0
    global_step = bundle.get("global_step", 0) if isinstance(bundle, dict) else 0

    if args.train:
        single_track_params_results = [None] * CONCURRENT_MODELS
        def spawn_worker(w_idx, init_p):
            single_track_params_results[w_idx] = run_single_track_worker(w_idx, init_p, quantization=args.quantization)

        threads = [threading.Thread(target=spawn_worker, args=(i, params), daemon=True) for i in range(CONCURRENT_MODELS)]
        for t in threads: t.start()

        threading.Thread(target=run_meta_daemon, daemon=True).start()

        print(f"[Daemon] Launching synchronized multimodal training daemon at global step {global_step} | Quantization: {args.quantization.upper()} (Version {version}).")
        loader = raw_memmap_loader(batch_size=8, min_seq_len=4, max_seq_len=16, samples_per_sec=44100, num_diffusion_steps=50, quantization=args.quantization)
        try:
            for batch_data in loader:
                batch_x, scales, bpms, stems, steps, sigmas, masks, batch_ids, batch_modalities = batch_data
                cond = (scales, bpms, stems, steps, sigmas)
                modality_type = batch_modalities[0] if batch_modalities else "audio"

                loss_val, grads = jax.value_and_grad(lambda p: combined_audio_loss(gpt_forward(p, jnp.cos(sigmas * np.pi / 2.0)[:, :, None] * batch_x + jnp.sin(sigmas * np.pi / 2.0)[:, :, None] * jax.random.normal(jax.random.PRNGKey(0), batch_x.shape), cond, masks, modality_type=modality_type), jax.random.normal(jax.random.PRNGKey(0), batch_x.shape), masks))(params)
                
                print(f"\n[Daemon] Step {global_step:04d} | Modality: {modality_type} | Quantization: {args.quantization} | Batch Loss: {float(loss_val):.4f} | Version: {version}")

                preconditioned_grads = get_meta_preconditioner(grads, loss_val)
                if preconditioned_grads is not None:
                    grads = preconditioned_grads
                    print("  -> [Meta] Gradients successfully preconditioned by spectral MLP.")
                
                # Guard NTK calculation with an exclusive lock so only one container/instance computes it per milestone step
                if global_step % 10 == 0:
                    ntk_pickle_path = f"checkpoints/ntk/ntk_step_{global_step:04d}.pickle"
                    with open(NTK_LOCK_PATH, "a+") as ntf:
                        fcntl.flock(ntf, fcntl.LOCK_EX)
                        try:
                            if not os.path.exists(ntk_pickle_path):
                                ntk = compute_empirical_ntk(params, batch_x, scales, bpms, stems, steps, sigmas, masks, modality_type=modality_type)
                                print(f"  -> [NTK] Computed & Logged Trace: {ntk['trace']:.4f} | Condition Number: {ntk['condition_number']:.4f}")
                                pickle.dump(ntk, open(ntk_pickle_path, "wb"))
                                np.save(f"ntk_logs/ntk_step_{global_step:04d}.npy", ntk["matrix"].flatten())
                            else:
                                print(f"  -> [NTK] Step {global_step:04d} already computed by another instance. Skipping redundant calculation.")
                        finally:
                            fcntl.flock(ntf, fcntl.LOCK_UN)

                active_single_params = [res for res in single_track_params_results if res is not None]
                params, ema_params, version, updated = push_and_pull_gradients(optimizer, grads, loss_val, global_step, version, global_step, accumulation_steps=4, single_track_params_list=active_single_params)
                global_step += 1
        except KeyboardInterrupt:
            print("\n[Daemon] Training daemon interrupted safely.")
