import os, json, pickle, queue, threading, time, sys, struct, subprocess, ctypes, fcntl, shutil, optax, jax
import numpy as np, jax.numpy as jnp
from jax.core import Literal
from functools import partial


# Target Apple Silicon / Metal performance settings
jax.config.update("jax_default_matmul_precision", "float32")
jax.config.update("jax_enable_x64", False)

CURR_CKPT = "checkpoints/checkpoint_bundle.pickle"
PREV_CKPT = "checkpoints/checkpoint_bundle_prev.pickle"
CKPT_LOCK_PATH = "checkpoints/checkpoint.lock"
GRAD_LOCK_PATH = "data/shared_gradients.lock"
LIE_PARAMS = {"query", "key", "value"}

# -----------------------------------------------------------------------------
# 1. ARM64 NEON Elementwise AOT Compiler & Runtime Backend
# -----------------------------------------------------------------------------

def float_to_hex_halves(f_val):
    uint32_val = struct.unpack('<I', struct.pack('<f', float(f_val)))[0]
    return uint32_val & 0xFFFF, (uint32_val >> 16) & 0xFFFF

def literal_key(x):
    try:
        arr = np.asarray(x)
        if arr.size > 0:
            return float(arr.reshape(-1)[0])
    except Exception:
        pass
    return float(x)

def compile_elementwise_jaxpr_to_neon(closed_jaxpr):
    jaxpr = closed_jaxpr.jaxpr
    
    for invar in jaxpr.invars:
        if hasattr(invar, 'aval') and invar.aval.dtype != jnp.float32:
            raise NotImplementedError("Only float32 arrays are supported by the ARM64 NEON vectorizer.")

    asm = [
        ".text",
        ".align 2",
        ".global _jax_elementwise_neon_kernel",
        ".global jax_elementwise_neon_kernel",
        "_jax_elementwise_neon_kernel:",
        "jax_elementwise_neon_kernel:",
        "    // AAPCS64 Prologue: Preserve callee-saved 128-bit SIMD registers q8-q15",
        "    stp x29, x30, [sp, #-144]!",
        "    mov x29, sp",
        "    stp q8,  q9,  [sp, #16]",
        "    stp q10, q11, [sp, #48]",
        "    stp q12, q13, [sp, #80]",
        "    stp q14, q15, [sp, #112]"
    ]

    if len(jaxpr.invars) > 6:
        raise NotImplementedError("Only up to 6 inputs supported under AAPCS64 register bounds.")

    literals = {}
    for eqn in jaxpr.eqns:
        for invar in eqn.invars:
            if isinstance(invar, Literal):
                val = literal_key(invar.val)
                if val not in literals:
                    literals[val] = None

    if len(literals) > 15:
        raise RuntimeError("Exceeded maximum limit of 15 vector registers for literal constants (v31 reserved for scratch).")

    ALLOCATABLE = set(range(16, 31))
    temp_free = list(ALLOCATABLE)
    literal_reg_map = {}
    
    for val in literals:
        literal_reg_map[val] = temp_free.pop(0)

    LITERAL_REGS = set(literal_reg_map.values())
    free_regs = list(ALLOCATABLE - LITERAL_REGS)
    free_regs.sort()

    last_use = {}
    for i, eqn in enumerate(jaxpr.eqns):
        for invar in eqn.invars:
            if not isinstance(invar, Literal):
                last_use[invar] = i
    for outvar in jaxpr.outvars:
        last_use[outvar] = len(jaxpr.eqns)

    reg_map = {}

    def fn_alloc_reg(var):
        if var in reg_map:
            return reg_map[var]
        if not free_regs:
            raise RuntimeError("Out of vector registers during allocation. Consider simplifying sub-expressions.")
        r = free_regs.pop(0)
        reg_map[var] = r
        return r

    for val, reg_idx in literal_reg_map.items():
        lower_16, upper_16 = float_to_hex_halves(val)
        asm.append(f"    movz w10, #{lower_16}")
        if upper_16 != 0:
            asm.append(f"    movk w10, #{upper_16}, lsl #16")
        asm.append("    fmov s31, w10")
        asm.append(f"    dup v{reg_idx}.4s, v31.s[0]")

    asm.append("    mov x9, #0")
    asm.append(".loop_start:")
    asm.append("    cmp x9, x1")
    asm.append("    b.ge .loop_end")

    for i, invar in enumerate(jaxpr.invars):
        reg_idx = fn_alloc_reg(invar)
        asm.append(f"    ldr q{reg_idx}, [x{i + 2}, x9, lsl #4]")

    primitive_map = {
        "add": ("fadd", 2),
        "sub": ("fsub", 2),
        "mul": ("fmul", 2),
        "div": ("fdiv", 2),
        "neg": ("fneg", 1),
        "negative": ("fneg", 1),
        "abs": ("fabs", 1),
        "sqrt": ("fsqrt", 1),
        "maximum": ("fmax", 2),
        "minimum": ("fmin", 2),
    }

    final_outvar = jaxpr.outvars[0] if jaxpr.outvars else None

    for step, eqn in enumerate(jaxpr.eqns):
        prim_name = eqn.primitive.name
        if prim_name not in primitive_map:
            raise NotImplementedError(f"Primitive '{prim_name}' requires XLA/Metal fallback dispatch.")
        
        op_mnemonic, arity = primitive_map[prim_name]

        input_regs = [
            literal_reg_map[literal_key(inv.val)] if isinstance(inv, Literal) else fn_alloc_reg(inv)
            for inv in eqn.invars
        ]
        out_var = eqn.outvars[0]
        out_reg = fn_alloc_reg(out_var)

        if arity == 1:
            asm.append(f"    {op_mnemonic} v{out_reg}.4s, v{input_regs[0]}.4s")
        elif arity == 2:
            asm.append(f"    {op_mnemonic} v{out_reg}.4s, v{input_regs[0]}.4s, v{input_regs[1]}.4s")

        for invar in eqn.invars:
            if not isinstance(invar, Literal) and last_use.get(invar, -1) == step:
                if invar in reg_map and invar not in eqn.outvars and invar != final_outvar:
                    free_regs.append(reg_map[invar])
                    del reg_map[invar]
        
        if last_use.get(out_var, -1) == step and out_var not in jaxpr.outvars and out_var != final_outvar:
            if out_var in reg_map:
                free_regs.append(reg_map[out_var])
                del reg_map[out_var]

        free_regs.sort()

    final_reg = reg_map.get(final_outvar, fn_alloc_reg(final_outvar)) if final_outvar else 16
    asm.append(f"    str q{final_reg}, [x0, x9, lsl #4]")
    asm.append("    add x9, x9, #1")
    asm.append("    b .loop_start")
    asm.append(".loop_end:")
    
    asm.append("    ldp q14, q15, [sp, #112]")
    asm.append("    ldp q12, q13, [sp, #80]")
    asm.append("    ldp q10, q11, [sp, #48]")
    asm.append("    ldp q8,  q9,  [sp, #16]")
    asm.append("    ldp x29, x30, [sp], #144")
    asm.append("    ret")

    return "\n".join(asm)

class AppleSiliconNeonRuntime:
    def __init__(self, arm_asm):
        self.arm_asm = arm_asm
        self.lib_arm = None

    def compile_and_load(self):
        os.makedirs("checkpoints", exist_ok=True)
        with open("kernel.s", "w") as f:
            f.write(self.arm_asm)
        try:
            subprocess.run([
                "clang", "-arch", "arm64", "-dynamiclib", "-O3",
                "-Wl,-dead_strip", "-o", "./libarm.dylib", "kernel.s"
            ], check=True)
            self.lib_arm = ctypes.CDLL("./libarm.dylib")
        except Exception as e:
            raise RuntimeError(f"Failed to compile ARM64 assembly on macOS: {e}")

    def execute(self, *inputs):
        if not inputs:
            raise ValueError("At least one input tensor is required.")

        shapes = [x.shape for x in inputs]
        if len(set(shapes)) > 1:
            raise ValueError("All input tensors must match shape.")

        shape = shapes[0]
        N = inputs[0].size

        flat_inputs = [np.require(np.asarray(x).flatten(), dtype=np.float32, requirements=['ALIGNED', 'C']) for x in inputs]
        for fi in flat_inputs:
            if fi.ctypes.data % 16 != 0:
                raise RuntimeError("Tensor memory address is not 16-byte aligned for NEON vector loading.")

        padded_N = ((N + 3) // 4) * 4
        vector_count = padded_N // 4

        padded_inputs = [
            np.require(np.pad(x, (0, padded_N - N), mode="constant", constant_values=0), requirements=['ALIGNED', 'C'])
            for x in flat_inputs
        ] if padded_N != N else flat_inputs

        for pi in padded_inputs:
            if pi.ctypes.data % 16 != 0:
                raise RuntimeError("Padded tensor memory address is not 16-byte aligned for NEON vector loading.")

        padded_out = np.empty(padded_N, dtype=np.float32)
        padded_out.fill(np.nan)  # Guard against silent failures
        if padded_out.ctypes.data % 16 != 0:
            padded_out = np.require(padded_out, requirements=['ALIGNED', 'C'])

        ptr_padded_inputs = [x.ctypes.data_as(ctypes.c_void_p) for x in padded_inputs]
        ptr_padded_out = padded_out.ctypes.data_as(ctypes.c_void_p)
        cpu_out_flat = np.zeros(N, dtype=np.float32)

        if self.lib_arm:
            kernel = getattr(self.lib_arm, "jax_elementwise_neon_kernel")
            kernel.argtypes = [ctypes.c_void_p, ctypes.c_size_t] + [ctypes.c_void_p] * len(inputs)
            kernel.restype = None
            kernel(ptr_padded_out, ctypes.c_size_t(vector_count), *ptr_padded_inputs)
            
            if np.isnan(padded_out).any():
                raise RuntimeError("NEON kernel execution failed or produced uninitialized NaN outputs.")
                
            np.copyto(cpu_out_flat, padded_out[:N])

        return cpu_out_flat.reshape(shape)

# -----------------------------------------------------------------------------
# 2. Transformer Architecture & Diffusion Core
# -----------------------------------------------------------------------------

def rms_norm(x, scale, eps=1e-5):
    return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps) * scale

def apply_rope(x, freq_scale=10000.0):
    B, T, H, D = x.shape
    pos = jnp.arange(T, dtype=jnp.float32)[:, None]
    dim = jnp.arange(0, D, 2, dtype=jnp.float32)
    theta = pos / (freq_scale ** (dim / D))
    cos = jnp.cos(theta)[None, :, None, :]
    sin = jnp.sin(theta)[None, :, None, :]
    
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    
    out = jnp.empty_like(x)
    out = out.at[..., 0::2].set(x1 * cos - x2 * sin)
    out = out.at[..., 1::2].set(x1 * sin + x2 * cos)
    return out

def scaled_dot_product_attention(query, key, value, is_causal=False):
    scale = 1.0 / jnp.sqrt(query.shape[-1])
    scores = jnp.matmul(query, jnp.swapaxes(key, -2, -1)) * scale
    if is_causal:
        T_q = query.shape[-2]
        T_k = key.shape[-2]
        mask = jnp.tril(jnp.ones((T_q, T_k), dtype=bool))
        scores = jnp.where(mask, scores, -1e9)
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.matmul(weights, value)

def clamp_frobenius_norm(w, steps=2):
    for _ in range(steps):
        norm = jnp.sqrt(jnp.sum(w * w))
        w = jnp.where(norm > 1.0, w / norm, w)
    return w

def stft_loss(pred_noise, noise, n_fft=1024):
    pred_flat = pred_noise.reshape(-1)
    noise_flat = noise.reshape(-1)
    
    pad = (-pred_flat.shape[-1]) % n_fft
    if pad > 0:
        pred_flat = jnp.pad(pred_flat, (0, pad))
        noise_flat = jnp.pad(noise_flat, (0, pad))

    xf = jnp.abs(jnp.fft.rfft(pred_flat.reshape(-1, n_fft)))
    yf = jnp.abs(jnp.fft.rfft(noise_flat.reshape(-1, n_fft)))
    return jnp.mean(jnp.abs(xf - yf)) + jnp.mean(jnp.square(jnp.log(xf + 1e-5) - jnp.log(yf + 1e-5)))

def gpt_forward(params, x, scale, bpm, stem, step_indices, sigma_t, target_dim=88200, patch_dim=882, num_patches=100, n_heads=16):
    B, T, L = x.shape
    x_patches = x.reshape(B, T, num_patches, patch_dim)

    encoded = jax.nn.gelu(x_patches @ params['audio_encoder']) @ params['down_proj_2']
    C = encoded.shape[-1]
    head_dim = C // n_heads

    bt_shape = B * T
    x_patch_seq = encoded.reshape(bt_shape, num_patches, C)
    
    qp = (x_patch_seq @ params['query']).reshape(bt_shape, num_patches, n_heads, head_dim)
    kp = (x_patch_seq @ params['key']).reshape(bt_shape, num_patches, n_heads, head_dim)
    vp = (x_patch_seq @ params['value']).reshape(bt_shape, num_patches, n_heads, head_dim)
    
    qp = apply_rope(qp)
    kp = apply_rope(kp)
    
    qp = jnp.transpose(qp, (0, 2, 1, 3))
    kp = jnp.transpose(kp, (0, 2, 1, 3))
    vp = jnp.transpose(vp, (0, 2, 1, 3))
    
    attn_p = scaled_dot_product_attention(query=qp, key=kp, value=vp, is_causal=True)
    attn_p = jnp.transpose(attn_p, (0, 2, 1, 3))
    
    h_p = attn_p.reshape(bt_shape, num_patches, C)
    h_p = rms_norm(h_p + x_patch_seq, params['rms_scale_1'])

    ff_p = jax.nn.gelu(h_p @ params['ff_1']) @ params['ff_2']
    h_p = rms_norm(h_p + ff_p, params['rms_scale_2'])
    
    h_frames = h_p.reshape(B, T, num_patches, C)
    frame_tokens = jnp.mean(h_frames, axis=2) + params['t_pos_emb'][:T][None, :, :]

    qt = (frame_tokens @ params['t_query']).reshape(B, T, n_heads, head_dim)
    kt = (frame_tokens @ params['t_key']).reshape(B, T, n_heads, head_dim)
    vt = (frame_tokens @ params['t_value']).reshape(B, T, n_heads, head_dim)
    
    qt = apply_rope(qt)
    kt = apply_rope(kt)
    
    qt = jnp.transpose(qt, (0, 2, 1, 3))
    kt = jnp.transpose(kt, (0, 2, 1, 3))
    vt = jnp.transpose(vt, (0, 2, 1, 3))
    
    attn_t = scaled_dot_product_attention(query=qt, key=kt, value=vt, is_causal=True)
    attn_t = jnp.transpose(attn_t, (0, 2, 1, 3))
    
    h_t = attn_t.reshape(B, T, C)
    h_t = rms_norm(h_t + frame_tokens, params['t_rms_scale'])

    h_frames = h_frames + jnp.expand_dims(h_t, 2)

    scale_clipped = jnp.clip(scale, 0, 127)
    bpm_norm = (bpm - 120.0) / 60.0
    
    t_emb = params['time_emb'][step_indices]
    s_emb = params['scale_emb'][scale_clipped][:, None, :]
    b_emb = (bpm_norm[:, None] @ params['bpm_proj'])[:, None, :]
    stem_cond = params['stem_emb'][stem][:, None, :]
    
    base_cond = (t_emb + s_emb + b_emb + stem_cond)[:, :, None, :]
    sigma_emb = sigma_t[:, :, None, None] * params['sigma_emb'][None, None, None, :]
    
    h_frames = h_frames + base_cond + sigma_emb

    h_out = jax.nn.gelu(h_frames @ params['up_proj_1']) @ params['up_proj_2']
    h_out = rms_norm(h_out, params['out_rms_scale'])
    out_hidden = h_out @ params['out_proj']
    
    return out_hidden.reshape(B, T, num_patches * patch_dim)

def xavier_normal(key, shape):
    fan_in = shape[-2] if len(shape) >= 2 else shape[0]
    fan_out = shape[-1] if len(shape) >= 2 else shape[0]
    return jax.random.normal(key, shape) * jnp.sqrt(2.0 / (fan_in + fan_out))

def init_params(key, dim=1024, patch_dim=882, compressed_audio_dim=512, num_diffusion_steps=50):
    keys = jax.random.split(key, 21)
    params = {
        'audio_encoder': xavier_normal(keys[0], (patch_dim, compressed_audio_dim)),
        'down_proj_2': xavier_normal(keys[1], (compressed_audio_dim, dim)),
        'query': jax.random.orthogonal(keys[2], dim),
        'key': jax.random.orthogonal(keys[3], dim),
        'value': jax.random.orthogonal(keys[4], dim),
        't_query': jax.random.orthogonal(keys[5], dim),
        't_key': jax.random.orthogonal(keys[6], dim),
        't_value': jax.random.orthogonal(keys[7], dim),
        'ff_1': xavier_normal(keys[8], (dim, dim * 4)),
        'ff_2': xavier_normal(keys[9], (dim * 4, dim)),
        'up_proj_1': xavier_normal(keys[10], (dim, compressed_audio_dim)),
        'up_proj_2': xavier_normal(keys[11], (compressed_audio_dim, compressed_audio_dim)), 
        'out_proj': xavier_normal(keys[19], (compressed_audio_dim, patch_dim)),
        'rms_scale_1': jnp.ones((dim,)),
        'rms_scale_2': jnp.ones((dim,)),
        't_rms_scale': jnp.ones((dim,)),
        'out_rms_scale': jnp.ones((compressed_audio_dim,)),
        'scale_emb': xavier_normal(keys[12], (128, dim)),  
        'bpm_proj': xavier_normal(keys[13], (1, dim)),    
        'stem_emb': xavier_normal(keys[14], (2, dim)),
        'time_emb': xavier_normal(keys[15], (num_diffusion_steps, dim)),
        't_pos_emb': xavier_normal(keys[16], (512, dim)),
        'sigma_emb': xavier_normal(keys[18], (dim,))
    }
    jax.tree_util.tree_map(lambda x: assert_float32(x), params)
    return params

def assert_float32(x):
    if x.dtype != jnp.float32:
        raise TypeError(f"Expected float32 parameter, got {x.dtype}")

# -----------------------------------------------------------------------------
# 3. JAX-Native DDIM Sampler (`jax.lax.scan`)
# -----------------------------------------------------------------------------

def generate_audio_ddim(ema_params, key, scale_val, bpm_val, stem_val, batch_size=1, seq_len=10, num_steps=50, target_dim=88200, eta=0.0, cfg_scale=3.5):
    k_noise, k_run = jax.random.split(key)
    x_t = jax.random.normal(k_noise, (batch_size, seq_len, target_dim))
    
    scale = jnp.full((batch_size,), scale_val, dtype=jnp.int32)
    bpm = jnp.full((batch_size,), bpm_val, dtype=jnp.float32)
    stem = jnp.full((batch_size,), stem_val, dtype=jnp.int32)
    
    time_steps = jnp.linspace(1.0, 0.0, num_steps + 1)
    t_curr_arr = time_steps[:-1]
    t_prev_arr = time_steps[1:]

    def scan_step(carry, t_pair):
        x_t, k_run = carry
        t_curr, t_prev = t_pair
        k_run, k_step = jax.random.split(k_run)
        
        alpha_curr = jnp.cos(t_curr * jnp.pi / 2.0)
        sigma_curr = jnp.sin(t_curr * jnp.pi / 2.0)
        alpha_prev = jnp.cos(t_prev * jnp.pi / 2.0)
        sigma_prev = jnp.sin(t_prev * jnp.pi / 2.0)
        
        step_indices = jnp.maximum(0, jnp.floor(t_curr * num_steps).astype(jnp.int32) - 1)
        step_indices = jnp.broadcast_to(step_indices, (batch_size, seq_len))
        sigma_vals = jnp.full((batch_size, seq_len), sigma_curr, dtype=jnp.float32)
        
        if cfg_scale > 1.0:
            pred_noise_cond = gpt_forward(ema_params, x_t, scale, bpm, stem, step_indices, sigma_vals, target_dim=target_dim)
            zero_stem = jnp.zeros_like(stem)
            pred_noise_uncond = gpt_forward(ema_params, x_t, scale, bpm, zero_stem, step_indices, sigma_vals, target_dim=target_dim)
            pred_noise = pred_noise_uncond + cfg_scale * (pred_noise_cond - pred_noise_uncond)
        else:
            pred_noise = gpt_forward(ema_params, x_t, scale, bpm, stem, step_indices, sigma_vals, target_dim=target_dim)
            
        x0_est = (x_t - sigma_curr * pred_noise) / jnp.maximum(alpha_curr, 1e-5)
        sigma_hat = eta * jnp.sqrt(jnp.maximum(0.0, (1.0 - jnp.square(alpha_prev)) / jnp.maximum(1.0 - jnp.square(alpha_curr), 1e-5))) * jnp.sqrt(jnp.maximum(0.0, 1.0 - jnp.square(alpha_curr / alpha_prev)))
        dir_xt = jnp.sqrt(jnp.maximum(0.0, 1.0 - jnp.square(alpha_prev) - jnp.square(sigma_hat))) * pred_noise
        noise = jax.random.normal(k_step, x_t.shape) if eta > 0.0 else 0.0
        
        next_x_t = alpha_prev * x0_est + dir_xt + sigma_hat * noise
        return (next_x_t, k_run), None

    (x_final, _), _ = jax.lax.scan(scan_step, (x_t, k_run), (t_curr_arr, t_prev_arr))
    return x_final

# -----------------------------------------------------------------------------
# 4. Optimized Data Loader with Metadata Caching & Bounded MMAP Pool
# -----------------------------------------------------------------------------

_CACHED_METADATA = {"mtime": 0.0, "data": []}

def get_cached_metadata(meta_path):
    if not os.path.exists(meta_path):
        return []
    mtime = os.path.getmtime(meta_path)
    if mtime == _CACHED_METADATA["mtime"] and _CACHED_METADATA["data"]:
        return _CACHED_METADATA["data"]
    with open(meta_path, "r") as f:
        metadata = [json.loads(l) for l in f if l.strip()]
    _CACHED_METADATA["mtime"] = mtime
    _CACHED_METADATA["data"] = metadata
    return metadata

class BoundedMMapPool:
    def __init__(self, max_size=16):
        self.pool = {}
        self.max_size = max_size

    def get(self, shard_path):
        if shard_path in self.pool:
            self.pool[shard_path] = self.pool.pop(shard_path)  # Refresh LRU position
            return self.pool[shard_path]
        if len(self.pool) >= self.max_size:
            oldest_key = next(iter(self.pool))
            del self.pool[oldest_key]
        m = np.memmap(shard_path, dtype=np.float32, mode='r').reshape(-1)
        self.pool[shard_path] = m
        return m

def raw_memmap_loader(batch_size, seq_len=10, samples_per_sec=88200, num_diffusion_steps=50):
    meta_path = "data/audio_vault.meta.jsonl"
    mmap_pool = BoundedMMapPool(max_size=16)
    
    while True:
        metadata = get_cached_metadata(meta_path)
        if not metadata:
            time.sleep(0.5)
            continue
            
        batch, batch_scales, batch_bpms, batch_stems, batch_steps, batch_sigmas = [], [], [], [], [], []
        attempts = 0
        
        while len(batch) < batch_size:
            attempts += 1
            if attempts > 10000:
                raise RuntimeError("Data loader exceeded 10,000 invalid index attempts. Check dataset paths.")
                
            idx = np.random.randint(len(metadata))
            entry = metadata[idx]
            
            if entry.get("sample_rate", samples_per_sec) != samples_per_sec:
                continue
                
            shard_path = os.path.join("data", entry["shard"])
            if not os.path.exists(shard_path):
                continue
            
            file_frames = os.path.getsize(shard_path) // 4
            offset_bytes = entry.get("offset_bytes", 0)
            offset_frames = offset_bytes // 4
            
            mmap_arr = mmap_pool.get(shard_path)
                
            file_duration = (file_frames - offset_frames) / samples_per_sec
            if file_duration < seq_len:
                continue
                
            start_idx = int(np.random.uniform(0, file_duration - seq_len) * samples_per_sec)
            
            if offset_frames + start_idx + (seq_len * samples_per_sec) > file_frames:
                continue
            
            raw_audio_patches = [
                mmap_arr[offset_frames + start_idx + (i * samples_per_sec) : offset_frames + start_idx + ((i + 1) * samples_per_sec)].reshape(-1)
                for i in range(seq_len)
            ]
            
            steps_arr = [int(np.random.randint(0, num_diffusion_steps)) for _ in range(seq_len)]
            t_vals = [(s + 1.0) / float(num_diffusion_steps) for s in steps_arr]
            sigma_vals = [float(np.sin(t * np.pi / 2.0)) for t in t_vals]
            
            batch.append(np.stack(raw_audio_patches))
            batch_scales.append(int(entry.get("scale", 0)))
            batch_bpms.append(float(entry.get("bpm", 120.0)))
            batch_stems.append(int(entry.get("stem", 0)))
            batch_steps.append(np.array(steps_arr, dtype=np.int32))
            batch_sigmas.append(np.array(sigma_vals, dtype=np.float32))
            
        yield (
            np.stack(batch), 
            np.array(batch_scales, dtype=np.int32), 
            np.array(batch_bpms, dtype=np.float32), 
            np.array(batch_stems, dtype=np.int32),
            np.stack(batch_steps),
            np.stack(batch_sigmas)
        )

class PrefetchDataLoader:
    def __init__(self, generator, queue_size=8):
        self.generator = generator
        self.queue = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        try:
            while not self.stopped:
                try:
                    item = next(self.generator)
                    self.queue.put(item, timeout=1)
                except queue.Full:
                    continue
                except Exception as e:
                    print(f"[DataLoader Error] {e}")
                    self.stopped = True
                    break
        finally:
            self.stopped = True

    def __iter__(self):
        return self

    def __next__(self):
        if self.stopped:
            raise StopIteration
        return self.queue.get()

# -----------------------------------------------------------------------------
# 5. Atomic Checkpointing & Unified Lock Hierarchy
# -----------------------------------------------------------------------------

def _load_checkpoint_unlocked():
    if not os.path.exists(CURR_CKPT):
        return None
    with open(CURR_CKPT, "rb") as pf:
        return pickle.load(pf)

def load_checkpoint_safely():
    with open(CKPT_LOCK_PATH, "a+") as clf:
        fcntl.flock(clf, fcntl.LOCK_EX)
        try:
            return _load_checkpoint_unlocked()
        finally:
            fcntl.flock(clf, fcntl.LOCK_UN)

def push_and_pull_gradients(optimizer, local_grads, loss_val, global_step, expected_version, worker_id="worker_0", accumulation_steps=100):
    grad_store_path = "data/shared_gradients.pickle"
    apply_global_update = False
    new_params, new_ema_params, new_version = None, None, expected_version

    host_grads = jax.device_get(local_grads)

    with open(CKPT_LOCK_PATH, "a+") as clf:
        fcntl.flock(clf, fcntl.LOCK_EX)
        try:
            with open(GRAD_LOCK_PATH, "a+b") as gf:
                fcntl.flock(gf, fcntl.LOCK_EX)
                try:
                    if os.path.exists(grad_store_path):
                        with open(grad_store_path, "rb") as df:
                            shared_data = pickle.load(df)
                    else:
                        shared_data = {
                            "accumulated_grads": None,
                            "count": 0,
                            "version": expected_version,
                            "workers": {}
                        }
                except Exception:
                    shared_data = {
                        "accumulated_grads": None,
                        "count": 0,
                        "version": expected_version,
                        "workers": {}
                    }
                    
                shared_data.setdefault("workers", {})[worker_id] = {
                    "step": global_step,
                    "timestamp": time.time()
                }
                    
                if shared_data.get("version", expected_version) != expected_version:
                    bundle = _load_checkpoint_unlocked()
                    if bundle is None:
                        return None, None, expected_version, False
                    return bundle["params"], bundle.get("ema_params", bundle["params"]), bundle.get("version", expected_version), False

                if shared_data["accumulated_grads"] is None:
                    shared_data["accumulated_grads"] = host_grads
                else:
                    shared_data["accumulated_grads"] = jax.tree_util.tree_map(lambda x, y: x + y, shared_data["accumulated_grads"], host_grads)
                    
                shared_data["count"] += 1
                apply_global_update = shared_data["count"] >= accumulation_steps
                
                if apply_global_update:
                    shared_grads = jax.tree_util.tree_map(lambda x: x / shared_data["count"], shared_data["accumulated_grads"])
                    shared_data["accumulated_grads"] = None
                    shared_data["count"] = 0
                    shared_data["version"] += 1
                    new_version = shared_data["version"]
                    
                tmp_grad_path = grad_store_path + ".tmp"
                with open(tmp_grad_path, "wb") as tf:
                    pickle.dump(shared_data, tf)
                    os.fsync(tf.fileno())
                os.replace(tmp_grad_path, grad_store_path)
                fcntl.flock(gf, fcntl.LOCK_UN)
                
            if apply_global_update:
                os.makedirs("checkpoints", exist_ok=True)
                bundle = _load_checkpoint_unlocked()
                if bundle is None:
                    return None, None, expected_version, False
                params = bundle["params"]
                ema_params = bundle.get("ema_params", params)
                opt_state = bundle.get("opt_state", None)

                if opt_state is None:
                    opt_state = optimizer.init(params)

                updates, opt_state = optimizer.update(shared_grads, opt_state, params)
                new_params = optax.apply_updates(params, updates)
                
                if global_step % 1000 == 0:
                    for k in LIE_PARAMS:
                        if k in new_params:
                            new_params[k] = clamp_frobenius_norm(new_params[k])

                ema_decay = 0.9999
                new_ema_params = jax.tree_util.tree_map(
                    lambda ep, p: ema_decay * ep + (1.0 - ema_decay) * p,
                    ema_params, new_params
                )
                
                new_bundle = {
                    "params": new_params,
                    "ema_params": new_ema_params,
                    "opt_state": opt_state,
                    "version": new_version
                }

                if os.path.exists(CURR_CKPT):
                    if os.path.exists(PREV_CKPT):
                        os.remove(PREV_CKPT)
                    try:
                        os.link(CURR_CKPT, PREV_CKPT)
                    except OSError:
                        shutil.copy2(CURR_CKPT, PREV_CKPT)

                tmp_ckpt = CURR_CKPT + ".tmp"
                with open(tmp_ckpt, "wb") as tf:
                    pickle.dump(new_bundle, tf)
                    os.fsync(tf.fileno())
                os.replace(tmp_ckpt, CURR_CKPT)

                return new_params, new_ema_params, new_version, True
                
        finally:
            fcntl.flock(clf, fcntl.LOCK_UN)

    bundle = _load_checkpoint_unlocked()
    if bundle is None:
        return None, None, expected_version, False
    return bundle["params"], bundle.get("ema_params", bundle["params"]), bundle.get("version", expected_version), False

# -----------------------------------------------------------------------------
# 6. Main Runtime Engine
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    os.makedirs("data", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    NUM_DIFFUSION_STEPS = 50
    global_optimizer = optax.adam(1e-4)

    if not os.path.exists(CURR_CKPT):
        initial_params = init_params(key, num_diffusion_steps=NUM_DIFFUSION_STEPS)
        initial_bundle = {
            "params": initial_params,
            "ema_params": initial_params,
            "opt_state": global_optimizer.init(initial_params),
            "version": 0
        }
        with open(CKPT_LOCK_PATH, "a+") as clf:
            fcntl.flock(clf, fcntl.LOCK_EX)
            try:
                tmp_init = CURR_CKPT + ".tmp"
                with open(tmp_init, "wb") as f:
                    pickle.dump(initial_bundle, f)
                    os.fsync(f.fileno())
                os.replace(tmp_init, CURR_CKPT)
            finally:
                fcntl.flock(clf, fcntl.LOCK_UN)

    bundle = load_checkpoint_safely()
    params = bundle["params"]
    ema_params = bundle.get("ema_params", params)
    version = bundle.get("version", 0)

    if "--sample" in sys.argv:
        print("[Sampler] Generating audio sample using EMA weights with CFG via jax.lax.scan...")
        key, subkey = jax.random.split(key)
        sampled_audio = generate_audio_ddim(
            ema_params, subkey, scale_val=0, bpm_val=120.0, stem_val=0,
            batch_size=1, seq_len=10, num_steps=NUM_DIFFUSION_STEPS, target_dim=88200, eta=0.0, cfg_scale=3.5
        )
        np.save("checkpoints/generated_sample.npy", np.array(sampled_audio))
        print(f"[Sampler] Output generated with shape: {sampled_audio.shape}. Saved to checkpoints/generated_sample.npy")
        sys.exit(0)

    if "--aot" in sys.argv:
        dummy_a = jnp.zeros((5, 5000), dtype=jnp.float32)
        dummy_b = jnp.zeros((5, 5000), dtype=jnp.float32)
        
        def elementwise_fn(a, b):
            return jnp.sqrt(jnp.abs(a * b)) + jnp.maximum(a, b)
            
        closed_jaxpr = jax.make_jaxpr(elementwise_fn)(dummy_a, dummy_b)
        runtime = AppleSiliconNeonRuntime(compile_elementwise_jaxpr_to_neon(closed_jaxpr))
        runtime.compile_and_load()
        
        np_res = runtime.execute(dummy_a, dummy_b)
        jax_res = elementwise_fn(dummy_a, dummy_b)
        np.testing.assert_allclose(np_res, jax_res, rtol=1e-5, atol=1e-6)
        print("[AOT Compiler] ARM64 NEON elementwise backend compiled successfully with strict output validation.")
