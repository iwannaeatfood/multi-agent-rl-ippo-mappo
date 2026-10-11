"""Resumes the interrupted run: retrains ippo_fixed (clobbered by a smoke
test) and trains mappo_fixed/mappo_random, each as an isolated subprocess.
ippo_random already completed successfully in the earlier run and is left
untouched. Run: python resume_remaining.py
"""
import json
import subprocess
import sys

N_ITER = 700
REMAINING = ["ippo_fixed", "mappo_fixed", "mappo_random"]

for name in REMAINING:
    print(f"\n--- launching subprocess for {name} ---", flush=True)
    subprocess.run([sys.executable, "train_optimized.py", "--config", name, "--n_iter", str(N_ITER)], check=True)

summary = {}
for name in ["ippo_fixed", "ippo_random", "mappo_fixed", "mappo_random"]:
    with open(f"checkpoints_v2/{name}_summary.json") as f:
        summary[name] = json.load(f)

with open("checkpoints_v2/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\n=== summary ===")
print(json.dumps(summary, indent=2))
