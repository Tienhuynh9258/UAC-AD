"""
Per-scenario evaluation for RCAEval OnlineBoutique (RE2-OB) dataset.

Runs UAC-AD on 6 fault-type scenarios (cpu, delay, disk, loss, mem, socket)
and reports F1 / Precision / Recall per scenario plus mean±std.

Run twice to compare baseline vs trace-enhanced:

  # Baseline: log + metric only
  python codes/common/eval_per_scenario_rcaeval_re2_ob.py \\
      --data data/rcaeval_re2_ob --dataset rcaeval_re2_ob --data_type fuse \\
      --open_trace False --batch_size 128 --window_size 30 \\
      --epoches 5 5 --patience 3 \\    
      --result_dir data/rcaeval_re2_ob/result_per_scenario_fuse_baseline

  # Trace: log + metric + trace (GAT)
  python codes/common/eval_per_scenario_rcaeval_re2_ob.py \\
      --data data/rcaeval_re2_ob --dataset rcaeval_re2_ob --data_type fuse \\
      --open_trace True --batch_size 128 --window_size 30 \\
      --epoches 5 5 --patience 3 \\
      --result_dir data/rcaeval_re2_ob/result_per_scenario_fuse_trace
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

FAULT_TYPES = ["cpu", "delay", "disk", "loss", "mem", "socket"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_scenario_files(data_dir: str):
    """Return sorted list of (fault_type, test_pkl_path) for all 6 fault types."""
    entries = []
    for fault in FAULT_TYPES:
        path = os.path.join(data_dir, f"test_{fault}.pkl")
        if os.path.exists(path):
            entries.append((fault, path))
        else:
            logging.warning(f"Missing test file: {path}")
    return entries


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
        description="Per-scenario evaluation for RCAEval OnlineBoutique dataset"
    )
    # ── data / model args (passed through to run.py) ──
    p.add_argument("--data",           required=True)
    p.add_argument("--dataset",        default="rcaeval_re2_ob")
    p.add_argument("--data_type",      default="fuse", choices=["fuse", "kpi", "log"])
    p.add_argument("--open_trace",     default="False")
    p.add_argument("--num_services",   default=11,  type=int)
    p.add_argument("--trace_c",        default=6,   type=int)
    p.add_argument("--epoches",        default=[10, 10], nargs="+", type=int)
    p.add_argument("--batch_size",     default=256, type=int)
    p.add_argument("--patience",       default=5,   type=int)
    p.add_argument("--window_size",    default=60,  type=int,
                   help="Sliding window size in timesteps (1 timestep = 1 second).")
    p.add_argument("--val_percentile", default=95,  type=float,
                   help="Percentile of normal training losses as anomaly threshold.")
    p.add_argument("--alpha",          default=0.5, type=float)
    p.add_argument("--open_gan_sep",   default="True")
    p.add_argument("--run_start",      default=0,   type=int)
    p.add_argument("--run_end",        default=1,   type=int)
    p.add_argument("--result_dir",     default=None,
                   help="Base result dir. Each fault type gets its own subdir. "
                        "Defaults to {data}/result_per_scenario_{data_type}_"
                        "{'trace' if open_trace else 'baseline'}")
    p.add_argument("--gate_lambda",    default=0.01, type=float,
                   help="L1 regularizer on residual-gated trace gate g (auto-applied when open_trace=True).")
    p.add_argument("--run_py",         default=None,
                   help="Path to run.py (auto-detected if not set)")
    args = p.parse_args()

    # Convert data path to absolute so run.py subprocess (cwd=codes/) resolves it correctly
    args.data = os.path.abspath(args.data)
    if args.result_dir:
        args.result_dir = os.path.abspath(args.result_dir)

    # Auto-detect run.py
    if args.run_py:
        run_py = args.run_py
    else:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        run_py = os.path.normpath(os.path.join(this_dir, "..", "run.py"))
    if not os.path.exists(run_py):
        raise FileNotFoundError(f"run.py not found at {run_py}. Use --run_py.")

    open_trace_str = str(args.open_trace).lower()
    suffix = "trace" if open_trace_str in ("true", "1", "yes") else "baseline"
    result_base = args.result_dir or os.path.join(
        args.data, f"result_per_scenario_{args.data_type}_{suffix}"
    )

    scenario_files = _find_scenario_files(args.data)
    if not scenario_files:
        logging.error("No scenario test files found. Run preprocess_rcaeval_re2_ob.py first.")
        sys.exit(1)

    logging.info(f"Found {len(scenario_files)} scenarios.")
    logging.info(f"Results will be saved under: {result_base}/\n")

    results = []   # list of (fault_type, f1, pc, rc)

    for fault, test_pkl in scenario_files:
        sc_result_dir = os.path.join(result_base, fault)
        os.makedirs(sc_result_dir, exist_ok=True)

        logging.info(f"{'='*60}")
        logging.info(f"Scenario: {fault}")
        logging.info(f"  test_pkl : {test_pkl}")
        logging.info(f"  result   : {sc_result_dir}")

        cmd = [
            sys.executable, run_py,
            "--data",           args.data,
            "--dataset",        args.dataset,
            "--data_type",      args.data_type,
            "--open_trace",     args.open_trace,
            "--num_services",   str(args.num_services),
            "--trace_c",        str(args.trace_c),
            "--epoches",        *[str(e) for e in args.epoches],
            "--batch_size",     str(args.batch_size),
            "--patience",       str(args.patience),
            "--window_size",    str(args.window_size),
            "--val_percentile", str(args.val_percentile),
            "--alpha",          str(args.alpha),
            "--open_gan_sep",   args.open_gan_sep,
            "--run_start",      str(args.run_start),
            "--run_end",        str(args.run_end),
            "--test_pkl",       test_pkl,
            "--result_dir",     sc_result_dir,
            "--gate_lambda",    str(args.gate_lambda),
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=False,
                text=True,
                cwd=os.path.dirname(run_py),
            )
            if proc.returncode != 0:
                logging.warning(f"  run.py exited with code {proc.returncode}")
        except Exception as e:
            logging.error(f"  Failed to run scenario {fault}: {e}")
            results.append((fault, 0.0, 0.0, 0.0))
            continue

        res = _scan_latest_results(sc_result_dir)
        if res is None:
            logging.warning(f"  No result files found in {sc_result_dir}")
            results.append((fault, 0.0, 0.0, 0.0))
        else:
            f1, pc, rc = res
            results.append((fault, f1, pc, rc))
            logging.info(f"  → F1={f1:.4f}  P={pc:.4f}  R={rc:.4f}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    if not results:
        logging.error("No results collected.")
        return

    f1s = np.array([r[1] for r in results])
    pcs = np.array([r[2] for r in results])
    rcs = np.array([r[3] for r in results])

    col = 10
    sep = "-" * (col + 34)
    print(f"\n{'='*56}")
    print(f"  RE2-OB Per-Scenario Results  "
          f"({args.data_type}, open_trace={args.open_trace})")
    print(f"{'='*56}")
    print(f"  {'Fault':<{col}}  {'F1':>6}  {'Precision':>9}  {'Recall':>6}")
    print(f"  {sep}")
    for fault, f1, pc, rc in results:
        print(f"  {fault:<{col}}  {f1:>6.4f}  {pc:>9.4f}  {rc:>6.4f}")
    print(f"  {sep}")
    print(f"  {'Mean':<{col}}  {f1s.mean():>6.4f}  {pcs.mean():>9.4f}  {rcs.mean():>6.4f}")
    print(f"  {'Std':<{col}}  {f1s.std():>6.4f}  {pcs.std():>9.4f}  {rcs.std():>6.4f}")
    print(f"{'='*56}\n")

    summary = {
        "config": {
            "data_type":      args.data_type,
            "open_trace":     args.open_trace,
            "val_percentile": args.val_percentile,
            "window_size":    args.window_size,
        },
        "per_scenario": [
            {"fault": fault, "f1": f1, "precision": pc, "recall": rc}
            for fault, f1, pc, rc in results
        ],
        "aggregate": {
            "f1_mean":         float(f1s.mean()),  "f1_std":         float(f1s.std()),
            "precision_mean":  float(pcs.mean()),  "precision_std":  float(pcs.std()),
            "recall_mean":     float(rcs.mean()),  "recall_std":     float(rcs.std()),
        },
    }
    summary_path = os.path.join(result_base, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
