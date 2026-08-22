import os, json, pickle, queue, threading, time, sys, struct, subprocess, ctypes, fcntl, shutil, optax, jax
import numpy as np, jax.numpy as jnp
from jax.extend.core import Literal
from scipy.io import wavfile
from functools import partial, reduce

jax.config.update("jax_default_matmul_precision", "float32")
jax.config.update("jax_enable_x64", False)

CURR_CKPT, PREV_CKPT = "checkpoints/checkpoint_bundle.pickle", "checkpoints/checkpoint_bundle_prev.pickle"
CKPT_LOCK_PATH, GRAD_LOCK_PATH = "checkpoints/checkpoint.lock", "data/shared_gradients.lock"
LIE_PARAMS = {"query", "key", "value"}

# -----------------------------------------------------------------------------
# 1. Transformer Architecture & Diffusion Core
# -----------------------------------------------------------------------------
rms_norm = lambda x, scale, eps=1e-5: x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps) * scale
apply_rope = lambda x, freq=10000.0: (lambda c, s: x.at[..., 0::2].set(x[..., 0::2]*c - x[..., 1::2]*s).at[..., 1::2].set(x[..., 0::2]*s + x[..., 1::2]*c))(jnp.cos(jnp.arange(x.shape[-2], dtype=jnp.float32)[None, :, None] / (freq ** (jnp.arange(0, x.shape[-1], 2, dtype=jnp.float32) / x.shape[-1]))), jnp.sin(jnp.arange(x.shape[-2], dtype=jnp.float32)[None, :, None] / (freq ** (jnp.arange(0, x.shape[-1], 2, dtype=jnp.float32) / x.shape[-1]))))
scaled_dot_product_attention = lambda q, k, v, mask=None: jnp.matmul(jax.nn.softmax(jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * (1.0 / jnp.sqrt(q.shape[-1])) + (mask if mask is not None else 0.0), axis=-1), v)
clamp_frobenius_norm = lambda w, steps=2: reduce(lambda mat, _: jnp.where((n := jnp.sqrt(jnp.sum(mat * mat))) > 1.0, mat / n, mat), range(steps), w)

def combined_audio_loss(pred, target, mask=None, n_fft=1024):
    if mask is not None:
        pred = pred * mask[:, :, None]
        target = target * mask[:, :, None]
    pf, nf = map(lambda x: jnp.pad(x.reshape(-1), (0, (-x.size) % n_fft)), (pred, target))
    xf, yf = map(lambda arr: jnp.abs(jnp.fft.rfft(arr.reshape(-1, n_fft))), (pf, nf))
    return jnp.mean(jnp.abs(xf - yf)) + jnp.mean(jnp.square(jnp.log(xf + 1e-5) - jnp.log(yf + 1e-5))) + 0.5 * jnp.mean(jnp.abs(pred - target))

def compute_empirical_ntk(params, batch_x, scales, bpms, stems, steps, sigmas, mask=None):
    model_fn = lambda p: gpt_forward(p, jnp.cos(sigmas * jnp.pi / 2.0)[:, :, None] * batch_x + jnp.sin(sigmas * jnp.pi / 2.0)[:, :, None] * jax.random.normal(jax.random.PRNGKey(0), batch_x.shape), (scales, bpms, stems, steps, sigmas), mask)
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

def gpt_forward(params, x, cond, mask=None, target_dim=44100, patch_dim=441, num_patches=100, n_heads=16):
    scale, bpm, stem, step_indices, sigma_t = cond
    B, T, L = x.shape
    encoded = jax.nn.gelu(x.reshape(B, T, num_patches, patch_dim) @ params['audio_encoder']) @ params['down_proj_2']
    C, bt = encoded.shape[-1], B * T
    head_dim = C // n_heads
    
    p_mask = None
    if mask is not None:
        patch_mask_2d = jnp.ones((B, T, num_patches, num_patches), dtype=bool) & mask[:, :, None, None]
        p_mask = jnp.where(patch_mask_2d.reshape(bt, num_patches, num_patches)[:, None, :, :], 0.0, -1e9)

    patch_seq = encoded.reshape(bt, num_patches, C)
    q_p, k_p, v_p = map(lambda k: (patch_seq @ params[k]).reshape(bt, num_patches, n_heads, head_dim).transpose(0, 2, 1, 3), ['query', 'key', 'value'])
    
    attn_out_p = scaled_dot_product_attention(apply_rope(q_p), apply_rope(k_p), v_p, p_mask).transpose(0, 2, 1, 3).reshape(bt, num_patches, C)
    h_p = rms_norm(attn_out_p + patch_seq, params['rms_scale_1'])
    h_p = rms_norm(h_p + jax.nn.gelu(h_p @ params['ff_1']) @ params['ff_2'], params['rms_scale_2'])
    
    h_frames = h_p.reshape(B, T, num_patches, C)
    ft = jnp.mean(h_frames, axis=2) + params['t_pos_emb'][:T][None, :, :]
    q_t, k_t, v_t = map(lambda k: (ft @ params[k]).reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3), ['t_query', 't_key', 't_value'])
    
    t_mask = jnp.where(mask[:, None, :] & mask[:, :, None], 0.0, -1e9)[:, None, :, :] if mask is not None else None
    attn_out_t = scaled_dot_product_attention(apply_rope(q_t), apply_rope(k_t), v_t, t_mask).transpose(0, 2, 1, 3).reshape(B, T, C)
    h_t = rms_norm(attn_out_t + ft, params['t_rms_scale'])
    
    base_cond = (params['time_emb'][step_indices] + params['scale_emb'][jnp.clip(scale, 0, 127)][:, None, :] + (((bpm - 120.0) / 60.0)[:, None] @ params['bpm_proj'])[:, None, :] + params['stem_emb'][stem][:, None, :])[:, :, None, :]
    h_frames = h_frames + jnp.expand_dims(h_t, 2) + base_cond + sigma_t[:, :, None, None] * params['sigma_emb'][None, None, None, :]
    return (rms_norm(jax.nn.gelu(h_frames @ params['up_proj_1']) @ params['up_proj_2'], params['out_rms_scale']) @ params['out_proj']).reshape(B, T, num_patches * patch_dim)

xavier_normal = lambda key, shape: jax.random.normal(key, shape) * jnp.sqrt(2.0 / (shape[-2] if len(shape) >= 2 else shape[0] + shape[-1] if len(shape) >= 2 else shape[0]))

def init_params(key, dim=1024, patch_dim=441, comp_dim=512, steps=50):
    keys = jax.random.split(key, 21)
    return {
        'audio_encoder': xavier_normal(keys[0], (patch_dim, comp_dim)), 'down_proj_2': xavier_normal(keys[1], (comp_dim, dim)),
        'query': jax.random.orthogonal(keys[2], dim), 'key': jax.random.orthogonal(keys[3], dim), 'value': jax.random.orthogonal(keys[4], dim),
        't_query': jax.random.orthogonal(keys[5], dim), 't_key': jax.random.orthogonal(keys[6], dim), 't_value': jax.random.orthogonal(keys[7], dim),
        'ff_1': xavier_normal(keys[8], (dim, dim * 4)), 'ff_2': xavier_normal(keys[9], (dim * 4, dim)),
        'up_proj_1': xavier_normal(keys[10], (dim, comp_dim)), 'up_proj_2': xavier_normal(keys[11], (comp_dim, comp_dim)), 'out_proj': xavier_normal(keys[19], (comp_dim, patch_dim)),
        'rms_scale_1': jnp.ones((dim,)), 'rms_scale_2': jnp.ones((dim,)), 't_rms_scale': jnp.ones((dim,)), 'out_rms_scale': jnp.ones((comp_dim,)),
        'scale_emb': xavier_normal(keys[12], (128, dim)), 'bpm_proj': xavier_normal(keys[13], (1, dim)), 'stem_emb': xavier_normal(keys[14], (2, dim)),
        'time_emb': xavier_normal(keys[15], (steps, dim)), 't_pos_emb': xavier_normal(keys[16], (512, dim)), 'sigma_emb': xavier_normal(keys[18], (dim,))
    }

# -----------------------------------------------------------------------------
# 2. Utilities & Data Pipeline (Variable-Length Windows)
# -----------------------------------------------------------------------------
load_checkpoint_safely = lambda: (lambda clf: (fcntl.flock(clf, fcntl.LOCK_EX), res := (pickle.load(open(CURR_CKPT, "rb")) if os.path.exists(CURR_CKPT) else None), fcntl.flock(clf, fcntl.LOCK_UN), res)[-1])(open(CKPT_LOCK_PATH, "a+"))

def push_and_pull_gradients(optimizer, local_grads, loss_val, global_step, expected_version, expected_step, worker_id="worker_0", accumulation_steps=4):
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
                bundle = pickle.load(open(CURR_CKPT, "rb"))
                updates, opt_state = optimizer.update(avg_grads, bundle.get("opt_state", optimizer.init(bundle["params"])), bundle["params"])
                new_params = optax.apply_updates(bundle["params"], updates)
                new_bundle = {"params": new_params, "ema_params": jax.tree_util.tree_map(lambda ep, p: 0.9999 * ep + 0.0001 * p, bundle.get("ema_params", bundle["params"]), new_params), "opt_state": opt_state, "version": shared["version"], "global_step": expected_step + 1}
                pickle.dump(new_bundle, open(CURR_CKPT, "wb"))
                return new_bundle["params"], new_bundle["ema_params"], new_bundle["version"], True
        finally:
            fcntl.flock(clf, fcntl.LOCK_UN)
    bundle = load_checkpoint_safely()
    return bundle["params"], bundle.get("ema_params", bundle["params"]), bundle.get("version", expected_version), False

def get_cached_metadata(meta_path):
    if not os.path.exists(meta_path): return []
    with open(meta_path, "r") as f: return [json.loads(l) for l in f if l.strip()]

def raw_memmap_loader(batch_size, min_seq_len=4, max_seq_len=16, samples_per_sec=44100, num_diffusion_steps=50):
    meta_path = "data/audio_vault.meta.jsonl"
    pool = {}
    while True:
        metadata = get_cached_metadata(meta_path)
        if not metadata:
            time.sleep(0.5)
            continue
            
        raw_samples = []
        max_T = 0
        
        while len(raw_samples) < batch_size:
            entry = metadata[np.random.randint(len(metadata))]
            shard_path = os.path.join("data", entry.get("shard", "shard_0.bin"))
            if not os.path.exists(shard_path): continue
            
            seq_len = int(np.random.randint(min_seq_len, max_seq_len + 1))
            track_duration = entry.get("duration", (os.path.getsize(shard_path) // 4 - entry.get("offset_bytes", 0) // 4) / samples_per_sec)
            if track_duration < seq_len: continue
            
            offset_frames = entry.get("offset_bytes", 0) // 4
            if shard_path not in pool: pool[shard_path] = np.memmap(shard_path, dtype=np.float32, mode='r').reshape(-1)
            mmap_arr = pool[shard_path]
            
            start_idx = int(np.random.uniform(0, track_duration - seq_len) * samples_per_sec)
            patches = [mmap_arr[offset_frames + start_idx + (i * samples_per_sec) : offset_frames + start_idx + ((i + 1) * samples_per_sec)].reshape(-1) for i in range(seq_len)]
            
            steps_arr = [int(np.random.randint(0, num_diffusion_steps)) for _ in range(seq_len)]
            sigma_vals = [float(np.sin((s + 1.0) / float(num_diffusion_steps) * np.pi / 2.0)) for s in steps_arr]
            
            raw_url = entry.get("url", "unknown_url")
            yt_id = raw_url.split("v=")[-1].split("&")[0] if "v=" in raw_url else "yt_0"
            tw_str = f"{start_idx / samples_per_sec:.0f}s-{(start_idx / samples_per_sec) + seq_len:.0f}s"
            
            max_T = max(max_T, seq_len)
            raw_samples.append({
                "patches": np.stack(patches),
                "scale": int(entry.get("scale", 0)),
                "bpm": float(entry.get("bpm", 120.0)),
                "stem": int(entry.get("stem", 0)),
                "steps": np.array(steps_arr, dtype=np.int32),
                "sigmas": np.array(sigma_vals, dtype=np.float32),
                "id": (yt_id, tw_str),
                "len": seq_len
            })
            
        batch_x, batch_scales, batch_bpms, batch_stems, batch_steps, batch_sigmas, batch_masks, batch_ids = [], [], [], [], [], [], [], []
        
        for item in raw_samples:
            L = item["len"]
            pad_len = max_T - L
            
            padded_patches = np.pad(item["patches"], ((0, pad_len), (0, 0)), 'constant') if pad_len > 0 else item["patches"]
            padded_steps = np.pad(item["steps"], (0, pad_len), 'constant') if pad_len > 0 else item["steps"]
            padded_sigmas = np.pad(item["sigmas"], (0, pad_len), 'constant') if pad_len > 0 else item["sigmas"]
            mask = np.concatenate([np.ones(L, dtype=bool), np.zeros(pad_len, dtype=bool)]) if pad_len > 0 else np.ones(L, dtype=bool)
            
            batch_x.append(padded_patches)
            batch_scales.append(item["scale"])
            batch_bpms.append(item["bpm"])
            batch_stems.append(item["stem"])
            batch_steps.append(padded_steps)
            batch_sigmas.append(padded_sigmas)
            batch_masks.append(mask)
            batch_ids.append(item["id"])
            
        yield np.stack(batch_x), np.array(batch_scales, dtype=np.int32), np.array(batch_bpms, dtype=np.float32), np.array(batch_stems, dtype=np.int32), np.stack(batch_steps), np.stack(batch_sigmas), np.stack(batch_masks), batch_ids

# -----------------------------------------------------------------------------
# 3. Main Execution Interface
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("checkpoints/ntk", exist_ok=True)
    
    optimizer = optax.adam(1e-4)
    
    if not os.path.exists(CURR_CKPT):
        init_p = init_params(jax.random.PRNGKey(42))
        pickle.dump({"params": init_p, "ema_params": init_p, "opt_state": optimizer.init(init_p), "version": 0, "global_step": 0}, open(CURR_CKPT, "wb"))

    bundle = load_checkpoint_safely()
    params, ema_params, version, global_step = bundle["params"], bundle.get("ema_params", bundle["params"]), bundle.get("version", 0), bundle.get("global_step", 0)

    if "--train" in sys.argv:
        print(f"[Trainer] Resuming variable-length training loop at global step {global_step} (Version {version}). Press Ctrl+C to halt.")
        loader = raw_memmap_loader(batch_size=8, min_seq_len=4, max_seq_len=16, samples_per_sec=44100, num_diffusion_steps=50)
        try:
            for batch_data in loader:
                batch_x, scales, bpms, stems, steps, sigmas, masks, batch_ids = batch_data
                cond = (scales, bpms, stems, steps, sigmas)
                loss_val, grads = jax.value_and_grad(lambda p: combined_audio_loss(gpt_forward(p, jnp.cos(sigmas * jnp.pi / 2.0)[:, :, None] * batch_x + jnp.sin(sigmas * jnp.pi / 2.0)[:, :, None] * jax.random.normal(jax.random.PRNGKey(0), batch_x.shape), cond, masks), jax.random.normal(jax.random.PRNGKey(0), batch_x.shape), masks))(params)
                
                print(f"\n[Trainer] Step {global_step:04d} | Batch Loss: {float(loss_val):.4f} | Real Loss (0-1): {1.0/(1.0+float(loss_val)):.4f} | Version: {version}")
                print("  -> Active Batch Windows (YouTube ID : Time Window -> Individual Real Loss):")
                
                for i, (yt_id, tw) in enumerate(batch_ids):
                    single_cond = (scales[i:i+1], bpms[i:i+1], stems[i:i+1], steps[i:i+1], sigmas[i:i+1])
                    s_mask = masks[i:i+1]
                    s_loss = float(combined_audio_loss(gpt_forward(params, jnp.cos(sigmas[i:i+1] * jnp.pi / 2.0)[:, :, None] * batch_x[i:i+1] + jnp.sin(sigmas[i:i+1] * jnp.pi / 2.0)[:, :, None] * jax.random.normal(jax.random.PRNGKey(i), batch_x[i:i+1].shape), single_cond, s_mask), jax.random.normal(jax.random.PRNGKey(i), batch_x[i:i+1].shape), s_mask))
                    print(f"     [{i}] {yt_id} : {tw}  -->  Raw Loss: {s_loss:.4f} | Real Loss: {1.0/(1.0+s_loss):.4f}")

                if global_step % 10 == 0:
                    ntk = compute_empirical_ntk(params, batch_x, scales, bpms, stems, steps, sigmas, masks)
                    print(f"  -> [NTK] Trace: {ntk['trace']:.4f} | Condition Number: {ntk['condition_number']:.4f}")
                    pickle.dump(ntk, open(f"checkpoints/ntk/ntk_step_{global_step:04d}.pickle", "wb"))

                params, ema_params, version, updated = push_and_pull_gradients(optimizer, grads, loss_val, global_step, version, global_step, accumulation_steps=4)
                global_step += 1
        except KeyboardInterrupt:
            print("\n[Trainer] Training interrupted safely.")