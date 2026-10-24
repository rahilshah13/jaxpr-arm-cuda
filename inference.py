"""
    python3 inference.py --compile --seconds 10 --ckpt-mix checkpoints/checkpoint_bundle.pickle --quantization fp32
    python3 inference.py --generate --seconds 10 --ckpt-mix checkpoints/checkpoint_bundle.pickle --quantization fp32
"""

import argparse
import os
import pickle
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.io import wavfile

# Import directly from the model and processing modules
from model import (
    _load_params_from_bundle,
    gpt_forward,
    init_params,
    load_checkpoint_safely,
)
from processing import apply_quantization_tier


class JaxprInferenceEngine:
    """Manages JAXPR tracing, inspection, and JIT-compiled inference execution."""

    def __init__(self, params):
        self.params = params
        # JIT-compile the forward pass using JAX XLA
        self.jit_forward = jax.jit(
            partial(gpt_forward, self.params, modality_type="audio")
        )

    def trace_jaxpr(self, dummy_x, dummy_cond):
        """Extracts and returns the closed JAXPR representation of the forward pass."""
        closed_jaxpr = jax.make_jaxpr(
            partial(gpt_forward, self.params, modality_type="audio")
        )(dummy_x, dummy_cond)
        return closed_jaxpr

    def execute(self, noise, cond):
        """Executes the model inference pass via JAX JIT."""
        return self.jit_forward(noise, cond)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multimodal JAX Inference & JAXPR Inspection Engine with Timing Metrics"
    )
    parser.add_argument(
        "--compile", action="store_true", help="Run JAXPR tracing, inspection, and timing diagnostics"
    )
    parser.add_argument(
        "--generate", action="store_true", help="Run standard production generative sampling with execution timing"
    )
    parser.add_argument("--seconds", type=int, default=10, help="Duration in seconds")
    parser.add_argument(
        "--ckpt-mix",
        type=str,
        default="checkpoints/checkpoint_bundle.pickle",
        help="Comma-separated list of checkpoint bundles to mix/load",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "int8", "int4"],
        help="Data quantization tier to simulate during generation/compilation input",
    )
    args = parser.parse_args()

    ckpt_paths = [p.strip() for p in args.ckpt_mix.split(",")]
    primary_ckpt = ckpt_paths[0]


    def load_mixed_params():
        if not os.path.exists(primary_ckpt):
            return None
        try:
            bundle = (
                _load_params_from_bundle(primary_ckpt)
                if isinstance(_load_params_from_bundle(primary_ckpt), dict)
                else load_checkpoint_safely()
            )
        except Exception:
            bundle = load_checkpoint_safely()

        if not bundle:
            return None

        params = (
            bundle["params"] if isinstance(bundle, dict) and "params" in bundle else bundle
        )
        if len(ckpt_paths) > 1:
            print(f"[Inference] Blending checkpoint mix from paths: {ckpt_paths}")
            for alt_path in ckpt_paths[1:]:
                if os.path.exists(alt_path):
                    alt_p = _load_params_from_bundle(alt_path)
                    params = jax.tree_util.tree_map(
                        lambda p1, p2: 0.5 * p1 + 0.5 * p2, params, alt_p
                    )
        return bundle, params


    if args.compile:
        print(f"[JAX Diagnostics] Initializing model parameters for compilation analysis ({args.seconds}s)...")
        res = load_mixed_params()
        if not res:
            print("[JAX Diagnostics] Error: No checkpoint bundle found. Train the model first.")
            sys.exit(1)

        bundle, params = res
        os.makedirs("output", exist_ok=True)

        samples_per_sec = 44100
        dummy_x = jnp.zeros((1, args.seconds, samples_per_sec), dtype=jnp.float32)
        dummy_cond = (
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.zeros((1,), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.zeros((1, args.seconds), dtype=jnp.int32),
            jnp.zeros((1, args.seconds), dtype=jnp.float32),
        )

        # Measure JAXPR Tracing Overhead
        print("[JAX Diagnostics] Tracing model execution jaxpr...")
        t_trace_start = time.time()
        engine = JaxprInferenceEngine(params)
        closed_jaxpr = engine.trace_jaxpr(dummy_x, dummy_cond)
        t_trace_duration = time.time() - t_trace_start
        print(f"  -> Tracing Complete | Equations: {len(closed_jaxpr.jaxpr.eqns)} | Time: {t_trace_duration:.4f}s")

        key = jax.random.PRNGKey(1337)
        noise = jnp.array(jax.random.normal(key, (1, args.seconds, samples_per_sec)))

        if args.quantization != "fp32":
            noise_np = np.array(noise)
            noise_np = apply_quantization_tier([noise_np], args.quantization)[0]
            noise = jnp.array(noise_np)

        scale = jnp.array([64], dtype=jnp.int32)
        bpm = jnp.array([120.0], dtype=jnp.float32)
        stem = jnp.array([0], dtype=jnp.int32)
        steps_arr = jnp.zeros((1, args.seconds), dtype=jnp.int32)
        sigmas_arr = jnp.ones((1, args.seconds), dtype=jnp.float32)
        cond = (scale, bpm, stem, steps_arr, sigmas_arr)

        # Measure JIT Compilation & Execution Latency
        print("[JAX Diagnostics] Executing JIT-compiled inference pass (includes XLA compilation warmup)...")
        t_exec_start = time.time()
        generated_audio = engine.execute(noise, cond)
        jax.block_until_ready(generated_audio)
        t_exec_duration = time.time() - t_exec_start

        print(f"\n[Performance Metrics: --compile mode]")
        print(f"  -> JAXPR Tracing Time    : {t_trace_duration:.4f}s")
        print(f"  -> JIT Compilation + Run : {t_exec_duration:.4f}s")
        print(f"  -> Total Diagnostic Time : {t_trace_duration + t_exec_duration:.4f}s")

        flat_audio = np.array(generated_audio.reshape(-1), dtype=np.float32)
        flat_audio = np.clip(flat_audio, -1.0, 1.0)
        output_path = f"output/traced_generated_{args.seconds}s.wav"
        wavfile.write(output_path, samples_per_sec, (flat_audio * 32767).astype(np.int16))
        print(f"[JAX Diagnostics] Audio successfully exported to {output_path}")

    elif args.generate:
        print(f"[Production Engine] Generating {args.seconds} seconds of audio (Quantization: {args.quantization.upper()})...")
        res = load_mixed_params()
        if not res:
            print("[Production Engine] Error: No checkpoint bundle found. Train the model first.")
            sys.exit(1)

        bundle, params = res
        os.makedirs("output", exist_ok=True)

        samples_per_sec = 44100
        key = jax.random.PRNGKey(1337)
        noise = jnp.array(jax.random.normal(key, (1, args.seconds, samples_per_sec)))

        if args.quantization != "fp32":
            noise_np = np.array(noise)
            noise_np = apply_quantization_tier([noise_np], args.quantization)[0]
            noise = jnp.array(noise_np)

        scale = jnp.array([64], dtype=jnp.int32)
        bpm = jnp.array([120.0], dtype=jnp.float32)
        stem = jnp.array([0], dtype=jnp.int32)
        steps_arr = jnp.zeros((1, args.seconds), dtype=jnp.int32)
        sigmas_arr = jnp.ones((1, args.seconds), dtype=jnp.float32)
        cond = (scale, bpm, stem, steps_arr, sigmas_arr)

        t_init_start = time.time()
        engine = JaxprInferenceEngine(params)
        t_init_duration = time.time() - t_init_start

        print("[Production Engine] Running pure production generation pass...")
        t_gen_start = time.time()
        generated_audio = engine.execute(noise, cond)
        jax.block_until_ready(generated_audio)
        t_gen_duration = time.time() - t_gen_start

        print(f"\n[Performance Metrics: --generate mode]")
        print(f"  -> Engine Init Time      : {t_init_duration:.4f}s (No tracing overhead)")
        print(f"  -> Pure Execution Time   : {t_gen_duration:.4f}s")
        print(f"  -> Total Production Time : {t_init_duration + t_gen_duration:.4f}s")

        flat_audio = np.array(generated_audio.reshape(-1), dtype=np.float32)
        flat_audio = np.clip(flat_audio, -1.0, 1.0)
        output_path = f"output/generated_{args.seconds}s.wav"
        wavfile.write(output_path, samples_per_sec, (flat_audio * 32767).astype(np.int16))
        print(f"[Production Engine] Audio successfully exported to {output_path}")

    else:
        print(
            "Usage: python3 inference.py [--compile --seconds <N> | --generate --seconds <N>] [--ckpt-mix <path1,path2>] [--quantization <tier>]"
        )
