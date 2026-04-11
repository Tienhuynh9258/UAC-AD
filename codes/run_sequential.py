"""
Sequential runner: chạy tuần tự để tránh CUDA OOM.

  Step 1: Baseline (open_trace=False)
  Step 2: Trace-v3 (open_trace=True, new architecture)

Hyperparameters:
  epoches=30/30, batch_size=256, patience=5
  alpha=0.16, open_gan_sep=True
  (window_size=50, hidden_size=32, theta=0.15 — defaults)
"""
import subprocess, sys, os

python = sys.executable
script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.py")
data   = "C:/Users/us/Desktop/UAC-AD/.claude/worktrees/data/micross_v3"

RESULT_BASE  = "C:/Users/us/Desktop/UAC-AD/.claude/worktrees/data/result_fuse_baseline"
RESULT_TRACE = "C:/Users/us/Desktop/UAC-AD/.claude/worktrees/data/result_fuse_trace"

COMMON = [
    "--data",         data,
    "--dataset",      "micross",
    "--data_type",    "fuse",
    "--epoches",      "10", "10",
    "--batch_size",   "256",
    "--patience",     "5",
    "--alpha",        "0.16",
    "--open_gan_sep", "True",
    "--run_start",    "0",
    "--run_end",      "3",
]

jobs = [
    ("Baseline  (open_trace=False)  runs 0-4",
     ["--open_trace",  "False",
      "--result_dir",  RESULT_BASE]),

    ("Trace-v3  (open_trace=True)   runs 0-4",
     ["--open_trace",  "True",
      "--num_services","4",
      "--trace_c",     "5",
      "--result_dir",  RESULT_TRACE]),
]

for label, extra in jobs:
    print(f"\n{'='*60}")
    print(f"  >>> {label}")
    print(f"{'='*60}\n", flush=True)
    cmd = [python, script] + COMMON + extra
    print("CMD:", " ".join(cmd), "\n", flush=True)
    ret = subprocess.run(cmd, check=False)
    if ret.returncode != 0:
        print(f"\n[WARN] '{label}' exited with code {ret.returncode}. Continuing...\n",
              flush=True)

print("\n=== All sequential jobs done! ===")
