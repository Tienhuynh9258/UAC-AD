"""
Evaluation script for MicroSS (GAIA-DataSet) — baseline vs trace-enhanced.

MicroSS has a single train/test split (no per-scenario fault-type files).
This script runs UAC-AD twice (baseline and trace) and reports comparison.

Usage:
  # Baseline: log + KPI only
  python codes/common/eval_micross.py \\
      --data data/micross --dataset micross --data_type fuse \\
      --open_trace False --batch_size 256 --window_size 50 \\
      --epoches 10 10 --patience 5 --alpha 0.16 --open_gan_sep True \\
      --run_start 0 --run_end 3 \\
      --result_dir data/micross/result_fuse_baseline

  # Trace: log + KPI + trace (GAT + residual gate)
  python codes/common/eval_micross.py \\
      --data data/micross --dataset micross --data_type fuse \\
      --open_trace True --batch_size 256 --window_size 50 \\
      --epoches 10 10 --patience 5 --alpha 0.16 --open_gan_sep True \\
      --gate_lambda 0.01 --run_start 0 --run_end 3 \\
      --result_dir data/micross/result_fuse_trace
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _scan_latest_results(result_dir: str):
    """
    Scan result_dir recursively for info_score.txt files.
    Each file contains a line like:
        * Test -- f1:0.2174  rc:0.1220  pc:1.0000
    Returns (best_f1, best_pc, best_rc) or None.
    """
    if not os.path.isdir(result_dir):
        return None

    best_f1 = best_pc = best_rc = -1.0
    _pat = re.compile(r"f1:([\d.]+)\s+rc:([\d.]+)\s+pc:([\d.]+)")

    for root, _, files in os.walk(result_dir):
        for fname in files:
            if fname != "info_score.txt":
                continue
            try:
                with open(os.path.join(root, fname)) as f:
                    for line in f:
                        m = _pat.search(line)
                        if m:
                            f1 = float(m.group(1))
                            rc = float(m.group(2))
                            pc = float(m.group(3))
                            if f1 > best_f1:
                                best_f1, best_pc, best_rc = f1, pc, rc
            except Exception:
                continue

    return (best_f1, best_pc, best_rc) if best_f1 >= 0 else None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Evaluation for MicroSS dataset (baseline vs trace)"
    )
    p.add_argument("--data",           required=True)
    p.add_argument("--dataset",        default="micross")
    p.add_argument("--data_type",      default="fuse", choices=["fuse", "kpi", "log"])
    p.add_argument("--open_trace",     default="False")
    p.add_argument("--num_services",   default=4,    type=int)
    p.add_argument("--trace_c",        default=6,    type=int)
    p.add_argument("--epoches",        default=[10, 10], nargs="+", type=int)
    p.add_argument("--batch_size",     default=256,  type=int)
    p.add_argument("--patience",       default=5,    type=int)
    p.add_argument("--window_size",    default=50,   type=int)
    p.add_argument("--alpha",          default=0.16, type=float)
    p.add_argument("--open_gan_sep",   default="True")
    p.add_argument("--gate_lambda",    default=0.01, type=float,
                   help="L1 regularizer on residual-gated trace gate g "
                        "(applied when open_trace=True).")
    p.add_argument("--val_percentile", default=95,   type=float,
                   help="Percentile of normal (unlabel) losses used as anomaly threshold. "
                        "No val.pkl for MicroSS — falls back to unlabel_loader (same as RE2-OB/RE3-OB).")
    p.add_argument("--run_start",      default=0,    type=int)
    p.add_argument("--run_end",        default=3,    type=int)
    p.add_argument("--result_dir",     default=None,
                   help="Result output dir. "
                        "Defaults to {data}/result_{data_type}_{trace|baseline}")
    p.add_argument("--run_py",         default=None,
                   help="Path to run.py (auto-detected if not specified)")
    args = p.parse_args()

    # Auto-detect run.py
    if args.run_py:
        run_py = args.run_py
    else:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        run_py = os.path.normpath(os.path.join(this_dir, "..", "run.py"))
    if not os.path.exists(run_py):
        raise FileNotFoundError(f"run.py not found at {run_py}. Use --run_py to specify.")

    open_trace_str = str(args.open_trace).lower()
    suffix = "trace" if open_trace_str in ("true", "1", "yes") else "baseline"
    result_dir = args.result_dir or os.path.join(
        args.data, f"result_{args.data_type}_{suffix}"
    )
    os.makedirs(result_dir, exist_ok=True)

    logging.info(f"{'='*60}")
    logging.info(f"MicroSS evaluation — open_trace={args.open_trace}")
    logging.info(f"  data       : {args.data}")
    logging.info(f"  result_dir : {result_dir}")
    logging.info(f"{'='*60}")

    cmd = [
        sys.executable, run_py,
        "--data",         args.data,
        "--dataset",      args.dataset,
        "--data_type",    args.data_type,
        "--open_trace",   args.open_trace,
        "--num_services", str(args.num_services),
        "--trace_c",      str(args.trace_c),
        "--epoches",      *[str(e) for e in args.epoches],
        "--batch_size",   str(args.batch_size),
        "--patience",     str(args.patience),
        "--window_size",  str(args.window_size),
        "--alpha",        str(args.alpha),
        "--open_gan_sep",    args.open_gan_sep,
        "--gate_lambda",     str(args.gate_lambda),
        "--val_percentile",  str(args.val_percentile),
        "--run_start",       str(args.run_start),
        "--run_end",      str(args.run_end),
        "--result_dir",   result_dir,
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            cwd=os.path.dirname(run_py),
        )
        if proc.returncode != 0:
            logging.warning(f"run.py exited with code {proc.returncode}")
    except Exception as e:
        logging.error(f"Failed to run: {e}")
        sys.exit(1)

    res = _scan_latest_results(result_dir)
    if res is None:
        logging.warning(f"No result files found in {result_dir}")
        return

    f1, pc, rc = res
    print(f"\n{'='*60}")
    print(f"  MicroSS  ({args.data_type}, open_trace={args.open_trace})")
    print(f"{'='*60}")
    print(f"  Best F1        : {f1:.4f}")
    print(f"  Best Precision : {pc:.4f}")
    print(f"  Best Recall    : {rc:.4f}")
    print(f"{'='*60}\n")

    summary = {
        "config": {
            "data_type":      args.data_type,
            "open_trace":     args.open_trace,
            "window_size":    args.window_size,
            "gate_lambda":    args.gate_lambda,
            "val_percentile": args.val_percentile,
            "run_start":      args.run_start,
            "run_end":        args.run_end,
        },
        "result": {"f1": f1, "precision": pc, "recall": rc},
    }
    summary_path = os.path.join(result_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
