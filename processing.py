import os
import json
import numpy as np
from scipy.io import wavfile
from yt_dlp import YoutubeDL
import demucs.api

MAX_SHARD_BYTES = 500 * 1024 * 1024
DATA_DIR = "data"
URL_FILE = "data/urls.txt"
META_PATH = "data/audio_vault.meta.jsonl"
OUTPUT_DIR = "data/separated"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

separator = demucs.api.Separator()

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

if os.path.exists(URL_FILE):
    with open(URL_FILE, "r+") as f:
        urls = f.read().splitlines()
        f.seek(0)
        
        for url in urls:
            if url.startswith("DONE: "):
                f.write(f"{url}\n")
                continue
                
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
                    "duration": track_duration, # Explicit duration constraint added here
                    "url": url
                }
                with open(META_PATH, "a") as mf:
                    mf.write(json.dumps(meta_entry) + "\n")

                if os.path.exists(wav_path): os.remove(wav_path)
                if os.path.exists(vocal_path): os.remove(vocal_path)
                if os.path.exists(accomp_path): os.remove(accomp_path)
                try: os.rmdir(track_out_dir)
                except Exception: pass

                f.write(f"DONE: {url}\n")
                print(f"Vaulted dual-stem (4-Channels) {info['id']} ({track_duration:.1f}s) -> Shard {shard_idx}")

            except Exception as e:
                print(f"Error processing {url}: {e}")
                f.write(f"{url}\n")
        f.truncate()