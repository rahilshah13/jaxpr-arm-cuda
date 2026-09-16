"""
This module handles multimodal/audio data ingestion (YouTube + Demucs) in daemon mode
and quantization-aware memory-mapped loaders.

Usage:
    python3 processing.py --ingest-daemon
"""

import jax
import jax.numpy as jnp
import numpy as np
import os
import sys
import time
import json
import argparse
import fcntl
from scipy.io import wavfile
from yt_dlp import YoutubeDL
import demucs.api

# -----------------------------------------------------------------------------
# 1. Daemon Ingestion & Quantization-Aware Vault Pipeline
# -----------------------------------------------------------------------------
MAX_SHARD_BYTES = 500 * 1024 * 1024
DATA_DIR = "data"
URL_FILE = "data/urls.txt"
URL_LOCK_PATH = "data/urls.lock"
META_PATH = "data/audio_vault.meta.jsonl"
OUTPUT_DIR = "data/separated"

def get_current_shard_info():
    shard_idx = 0
    while True:
        bin_path = os.path.join(DATA_DIR, f"shard_{shard_idx}.bin")
        if not os.path.exists(bin_path):
            return shard_idx, bin_path, 0
        size = os.path.getsize(bin_path)
        if size < MAX_SHARD_BYTES:
            return shard_idx, bin_path, size
        shard_idx += 1

def process_single_url(url, separator):
    try:
        ydl_opts = {
            'format': 'bestaudio', 
            'outtmpl': 'data/%(id)s.%(ext)s', 
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}], 
            'quiet': True
        }
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            wav_path = os.path.join(DATA_DIR, f"{info['id']}.wav")
        
        origin, separated = separator.separate_audio_file(wav_path)
        
        track_out_dir = os.path.join(OUTPUT_DIR, info['id'])
        os.makedirs(track_out_dir, exist_ok=True)
        
        vocal_path = os.path.join(track_out_dir, "vocals.wav")
        accomp_path = os.path.join(track_out_dir, "accompaniment.wav")
        
        vocal_tensor = separated['vocals']
        accomp_tensor = sum(
            tensor for stem, tensor in separated.items() if stem != 'vocals'
        )
        
        sr = separator.samplerate
        demucs.api.save_audio(vocal_tensor, vocal_path, samplerate=sr)
        demucs.api.save_audio(accomp_tensor, accomp_path, samplerate=sr)
        
        sr_v, data_v = wavfile.read(vocal_path)
        if data_v.ndim == 1: 
            data_v = data_v[:, None].repeat(2, axis=1)
        
        sr_a, data_a = wavfile.read(accomp_path)
        if data_a.ndim == 1:
            data_a = data_a[:, None].repeat(2, axis=1)
        
        min_len = min(len(data_v), len(data_a))
        data_v = data_v[:min_len]
        data_a = data_a[:min_len]
        
        four_channel_data = np.hstack([data_v, data_a]).astype(np.float32)
        track_duration = len(four_channel_data) / sr_v
        
        shard_idx, bin_path, current_bytes = get_current_shard_info()
        
        with open(bin_path, "ab") as bf:
            bf.write(four_channel_data.tobytes())
        
        meta_entry = {
            "shard": f"shard_{shard_idx}.bin",
            "offset_bytes": current_bytes,
            "num_samples": len(four_channel_data),
            "sample_rate": sr_v,
            "duration": track_duration,
            "url": url
        }
        with open(META_PATH, "a") as mf:
            mf.write(json.dumps(meta_entry) + "\n")

        if os.path.exists(wav_path): os.remove(wav_path)
        if os.path.exists(vocal_path): os.remove(vocal_path)
        if os.path.exists(accomp_path): os.remove(accomp_path)
        try: os.rmdir(track_out_dir)
        except Exception: pass

        print(f"[Ingestion Daemon] Vaulted dual-stem (4-Channels) {info['id']} ({track_duration:.1f}s) -> Shard {shard_idx}")
        return True
    except Exception as e:
        print(f"[Ingestion Daemon] Error processing URL {url}: {e}")
        return False

def run_vault_ingestion_daemon(poll_interval=10):
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    separator = demucs.api.Separator()

    print(f"[Ingestion Daemon] Launching vault ingestion daemon polling '{URL_FILE}' every {poll_interval}s...")
    
    while True:
        if not os.path.exists(URL_FILE):
            time.sleep(poll_interval)
            continue

        with open(URL_LOCK_PATH, "a+") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                with open(URL_FILE, "r") as f:
                    lines = f.read().splitlines()
                
                updated_lines = []
                processed_any = False
                
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith("DONE: "):
                        updated_lines.append(line)
                        continue
                    
                    print(f"[Ingestion Daemon] Found new URL to process: {line}")
                    success = process_single_url(line, separator)
                    if success:
                        updated_lines.append(f"DONE: {line}")
                        processed_any = True
                    else:
                        updated_lines.append(line)
                
                if processed_any:
                    with open(URL_FILE, "w") as f:
                        f.write("\n".join(updated_lines) + ("\n" if updated_lines else ""))
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)
        
        time.sleep(poll_interval)

def get_cached_metadata(meta_path):
    if not os.path.exists(meta_path): return []
    with open(meta_path, "r") as f: return [json.loads(l) for l in f if l.strip()]

def apply_quantization_tier(patches, quantization="fp32"):
    processed_patches = []
    for p in patches:
        if quantization == "fp16":
            p_quant = p.astype(np.float16).astype(np.float32)
        elif quantization == "int8":
            p_clipped = np.clip(p, -1.0, 1.0)
            p_quant = (np.round(p_clipped * 127.0).astype(np.int8)).astype(np.float32) / 127.0
        elif quantization == "int4":
            p_clipped = np.clip(p, -1.0, 1.0)
            p_quant = (np.round(p_clipped * 7.0).astype(np.int8)).astype(np.float32) / 7.0
        else:
            p_quant = p.astype(np.float32)
        processed_patches.append(p_quant)
    return processed_patches

def get_full_track_data(samples_per_sec=44100, num_diffusion_steps=50, quantization="fp32"):
    meta_path = META_PATH
    while True:
        metadata = get_cached_metadata(meta_path)
        if not metadata: time.sleep(0.5); continue
        entry = metadata[np.random.randint(len(metadata))]
        shard_path = os.path.join(DATA_DIR, entry.get("shard", "shard_0.bin"))
        if not os.path.exists(shard_path): continue
        
        track_duration = entry.get("duration", (os.path.getsize(shard_path) // 4 - entry.get("offset_bytes", 0) // 4) / samples_per_sec)
        seq_len = int(track_duration)
        if seq_len < 2: continue
        
        offset_frames = entry.get("offset_bytes", 0) // 4
        mmap_arr = np.memmap(shard_path, dtype=np.float32, mode='r').reshape(-1)
        patches = [mmap_arr[offset_frames + (i * samples_per_sec) : offset_frames + ((i + 1) * samples_per_sec)].reshape(-1) for i in range(seq_len)]
        patches = apply_quantization_tier(patches, quantization)
        
        steps_arr = [int(np.random.randint(0, num_diffusion_steps)) for _ in range(seq_len)]
        sigma_vals = [float(np.sin((s + 1.0) / float(num_diffusion_steps) * np.pi / 2.0)) for s in steps_arr]
        
        raw_url = entry.get("url", "unknown_url")
        yt_id = raw_url.split("v=")[-1].split("&")[0] if "v=" in raw_url else "yt_0"
        return (
            np.stack(patches)[None, :, :],
            int(entry.get("scale", 0)),
            float(entry.get("bpm", 120.0)),
            int(entry.get("stem", 0)),
            np.array(steps_arr, dtype=np.int32)[None, :],
            np.array(sigma_vals, dtype=np.float32)[None, :],
            yt_id
        )

def raw_memmap_loader(batch_size, min_seq_len=4, max_seq_len=16, samples_per_sec=44100, num_diffusion_steps=50, quantization="fp32"):
    meta_path = META_PATH
    pool = {}
    while True:
        metadata = get_cached_metadata(meta_path)
        if not metadata: time.sleep(0.5); continue
            
        raw_samples = []
        max_T = 0
        while len(raw_samples) < batch_size:
            entry = metadata[np.random.randint(len(metadata))]
            shard_path = os.path.join(DATA_DIR, entry.get("shard", "shard_0.bin"))
            if not os.path.exists(shard_path): continue
            
            seq_len = int(np.random.randint(min_seq_len, max_seq_len + 1))
            track_duration = entry.get("duration", (os.path.getsize(shard_path) // 4 - entry.get("offset_bytes", 0) // 4) / samples_per_sec)
            if track_duration < seq_len: continue
            
            offset_frames = entry.get("offset_bytes", 0) // 4
            if shard_path not in pool: pool[shard_path] = np.memmap(shard_path, dtype=np.float32, mode='r').reshape(-1)
            mmap_arr = pool[shard_path]
            
            start_idx = int(np.random.uniform(0, track_duration - seq_len) * samples_per_sec)
            patches = [mmap_arr[offset_frames + start_idx + (i * samples_per_sec) : offset_frames + start_idx + ((i + 1) * samples_per_sec)].reshape(-1) for i in range(seq_len)]
            patches = apply_quantization_tier(patches, quantization)
            
            steps_arr = [int(np.random.randint(0, num_diffusion_steps)) for _ in range(seq_len)]
            sigma_vals = [float(np.sin((s + 1.0) / float(num_diffusion_steps) * np.pi / 2.0)) for s in steps_arr]
            
            raw_url = entry.get("url", "unknown_url")
            yt_id = raw_url.split("v=")[-1].split("&")[0] if "v=" in raw_url else "yt_0"
            tw_str = f"{start_idx / samples_per_sec:.0f}s-{(start_idx / samples_per_sec) + seq_len:.0f}s"
            
            max_T = max(max_T, seq_len)
            raw_samples.append({
                "patches": np.stack(patches), "scale": int(entry.get("scale", 0)),
                "bpm": float(entry.get("bpm", 120.0)), "stem": int(entry.get("stem", 0)),
                "steps": np.array(steps_arr, dtype=np.int32), "sigmas": np.array(sigma_vals, dtype=np.float32),
                "id": (yt_id, tw_str), "len": seq_len, "modality": "audio"
            })
            
        batch_x, batch_scales, batch_bpms, batch_stems, batch_steps, batch_sigmas, batch_masks, batch_ids, batch_modalities = [], [], [], [], [], [], [], [], []
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
            batch_modalities.append(item["modality"])
            
        yield np.stack(batch_x), np.array(batch_scales, dtype=np.int32), np.array(batch_bpms, dtype=np.float32), np.array(batch_stems, dtype=np.int32), np.stack(batch_steps), np.stack(batch_sigmas), np.stack(batch_masks), batch_ids, batch_modalities

if __name__ == "__main__":
    if "--ingest-daemon" in sys.argv:
        run_vault_ingestion_daemon()
    else:
        print("Usage: python3 processing.py --ingest-daemon")
