"""
Discriminator & RLHF Daemon: Manages sample grading (0-10) and exposes 
endpoints/locks for reward aggregation into the meta-learning loop.
"""

import os
import json
import fcntl
import time
import random
import numpy as np

RLHF_DATA_PATH = "data/rlhf_feedback.jsonl"
RLHF_LOCK_PATH = "data/rlhf.lock"

os.makedirs("data", exist_ok=True)

def record_feedback(sample_id, score, prompt_context=""):
    """Records a human feedback score (0-10) for a given generated sample."""
    score = max(0.0, min(10.0, float(score)))
    entry = {
        "timestamp": time.time(),
        "sample_id": sample_id,
        "score": score,
        "normalized_reward": score / 10.0,
        "context": prompt_context
    }
    with open(RLHF_LOCK_PATH, "a+") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            with open(RLHF_DATA_PATH, "a") as f:
                f.write(json.dumps(entry) + "\n")
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
    print(f"[Discriminator Daemon] Recorded RLHF Grade for {sample_id}: {score}/10")

def get_latest_rlhf_reward_scaling():
    """Computes a running reward scalar from recent feedback to condition the meta-preconditioner."""
    if not os.path.exists(RLHF_DATA_PATH):
        return 1.0
    
    with open(RLHF_LOCK_PATH, "a+") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            with open(RLHF_DATA_PATH, "r") as f:
                lines = [json.loads(l) for l in f if l.strip()]
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
            
    if not lines:
        return 1.0
    
    recent = lines[-50:] # Look at last 50 ratings
    avg_reward = sum(r["normalized_reward"] for r in recent) / len(recent)
    # Map average reward (0 to 1) to a scaling factor [0.5, 1.5] for meta-curriculum
    return float(0.5 + avg_reward)

if __name__ == "__main__":
    print("[Discriminator Daemon] Initialized and listening for RLHF grading loops...")
    while True:
        # Background maintenance or reward telemetry logging
        time.sleep(10)
