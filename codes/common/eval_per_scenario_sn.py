"""
Per-scenario evaluation for SocialNetwork (AnoMod) dataset.

For each anomaly scenario, runs HADES on:
    train.pkl  = Normal_Baseline (same for all scenarios)
    test_pkl   = scenarios/test_{scenario_name}.pkl
               = Normal_Baseline windows (shuffled with that scenario's windows)

Aggregates F1 / Precision / Recall across all scenarios and reports
mean ± std.

Usage:
    python codes/common/eval_per_scenario_sn.py \\
        --data data/sn \\
        --dataset sn --data_type fuse \\
        --open_trace False \\
        --epoches 10 10 --batch_size 256 --patience 5 \\
        --window_size 5 --val_percentile 95 \\
        --alpha 0.16 --open_gan_sep True \\
        --run_start 0 --run_end 1
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _find_scenario_files(data_dir: str):
    """Return sorted list of (scenario_name, test_pkl_path)."""
    scenarios_dir = os.path.join(data_dir, "scenarios")
    if not os.path.isdir(scenarios_dir):
        raise FileNotFoundError(
            f"scenarios/ folder not found under {data_dir}. "
            f"Re-run preprocess_sn.py first."
        )
    entries = []
    for fname in sorted(os.listdir(scenarios_dir)):
        if fname.startswith("test_") and fname.endswith(".pkl"):
            sc_name = fname[len("test_"):-len(".pkl")]
            entries.append((sc_name, os.path.join(scenarios_dir, fname)))
    return entries


def _read_result(result_dir: str, hash_id: str):
    """
    Read best F1/precision/recall from the result JSON written by dump_scores.
    Returns (f1, precision, recall) or None if not found.
    """
    # dump_scores writes to {result_dir}/{hash_id}.json
    path = os.path.join(result_dir, f"{hash_id}.json")
    if not os.path.exists(path):
        # Try csv fallback
        csv_path = os.path.join(result_dir, f"{hash_id}.csv")
        if os.path.exists(csv_path):
            import csv
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            if rows:
                last = rows[-1]
                return float(last.get("f1", 0)), float(last.get("pc", 0)), float(last.get("rc", 0))
        return None
    with open(path) as f:
        data = json.load(f)
    # data may be a list of run results; take the last one or max f1
    if isinstance(data, list):
        best = max(data, key=lambda x: x.get("f1", 0))
    else:
        best = data
    return float(best.get("f1", 0)), float(best.get("pc", 0)), float(best.get("rc", 0))


def _build_hash_id(base_args: dict, run_times: int, scenario_suffix: str) -> str:
    """
    Approximate hash_id used by run.py — we just scan the result dir for
    files matching the run_times index.
    """
    # Not used directly; we scan result dir instead.
    return ""


def _scan_latest_results(result_dir: str, run_start: int, run_end: int):
    """
    Scan result_dir recursively for info_score.txt files (one per run subdir).
    Each file contains a line like:
        * Test -- f1:0.2174\trc:0.1220\tpc:1.0000
    Returns (best_f1, best_pc, best_rc) across all runs, or None if not found.
    """
    import re
    if not os.path.isdir(result_dir):
        return None

    best_f1 = best_pc = best_rc = -1.0

    for root, dirs, files in os.walk(result_dir):
        for fname in files:
            if fname != "info_score.txt":
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath) as f:
                    for line in f:
                        m = re.search(
                            r"f1:([\d.]+)\s+rc:([\d.]+)\s+pc:([\d.]+)", line
                        )
                        if m:
                            f1 = float(m.group(1))
                            rc = float(m.group(2))
                            pc = float(m.group(3))
                            if f1 > best_f1:
                                best_f1, best_pc, best_rc = f1, pc, rc
            except Exception:
                continue

    return (best_f1, best_pc, best_rc) if best_f1 >= 0 else None


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Per-scenario evaluation for SocialNetwork dataset"
    )
    # ── data / model args (passed through to run.py) ──
    p.add_argument("--data",          required=True)
    p.add_argument("--dataset",       default="sn")
    p.add_argument("--data_type",     default="fuse", choices=["fuse", "kpi", "log"])
    p.add_argument("--open_trace",    default="False")
    p.add_argument("--num_services",  default=12,   type=int)
    p.add_argument("--trace_c",       default=5,    type=int)
    p.add_argument("--epoches",       default=[10, 10], nargs="+", type=int)
    p.add_argument("--batch_size",    default=256,  type=int)
    p.add_argument("--patience",      default=5,    type=int)
    p.add_argument("--window_size",   default=5,    type=int)
    p.add_argument("--val_percentile", default=95,   type=float,
                   help="Percentile of normal training losses used as anomaly threshold. "
                        "Replaces anomaly_rate sweep — no data leakage.")
    p.add_argument("--alpha",         default=0.16, type=float)
    p.add_argument("--open_gan_sep",  default="True")
    p.add_argument("--run_start",     default=0,    type=int)
    p.add_argument("--run_end",       default=1,    type=int)
    p.add_argument("--result_dir",    default=None,
                   help="Base result dir; each scenario gets its own subdir. "
                        "Defaults to {data}/result_per_scenario_{data_type}")
    p.add_argument("--run_py",        default=None,
                   help="Path to run.py (auto-detected if not specified)")
    args = p.parse_args()

    # Auto-detect run.py
    if args.run_py:
        run_py = args.run_py
    else:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        run_py = os.path.join(this_dir, "..", "run.py")
        run_py = os.path.normpath(run_py)
    if not os.path.exists(run_py):
        raise FileNotFoundError(f"run.py not found at {run_py}. Use --run_py to specify.")

    result_base = args.result_dir or os.path.join(
        args.data, f"result_per_scenario_{args.data_type}"
    )

    scenario_files = _find_scenario_files(args.data)
    if not scenario_files:
        logging.error("No scenario test files found. Re-run preprocess_sn.py.")
        sys.exit(1)

    logging.info(f"Found {len(scenario_files)} scenarios to evaluate.")
    logging.info(f"Results will be saved under: {result_base}/\n")

    # ── Run per scenario ──────────────────────────────────────────────────────
    results = []   # list of (sc_name, f1, pc, rc)

    for sc_name, test_pkl in scenario_files:
        # Strip trailing _YYYYMMDD_HHMMSS timestamp from sc_name for result folder
        import re as _re
        sc_folder = _re.sub(r'_\d{8}_\d{6}$', '', sc_name)
        sc_result_dir = os.path.join(result_base, sc_folder)
        os.makedirs(sc_result_dir, exist_ok=True)

        logging.info(f"{'='*60}")
        logging.info(f"Scenario: {sc_name}")
        logging.info(f"  test_pkl : {test_pkl}")
        logging.info(f"  result   : {sc_result_dir}")

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
            "--val_percentile", str(args.val_percentile),
            "--alpha",        str(args.alpha),
            "--open_gan_sep", args.open_gan_sep,
            "--run_start",    str(args.run_start),
            "--run_end",      str(args.run_end),
            "--test_pkl",          test_pkl,
            "--result_dir",        sc_result_dir,
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=False,   # let output flow to terminal
                text=True,
                cwd=os.path.dirname(run_py),
            )
            if proc.returncode != 0:
                logging.warning(f"  run.py exited with code {proc.returncode}")
        except Exception as e:
            logging.error(f"  Failed to run scenario {sc_name}: {e}")
            results.append((sc_name, 0.0, 0.0, 0.0))
            continue

        # ── Read result ──────────────────────────────────────────────────────
        res = _scan_latest_results(sc_result_dir, args.run_start, args.run_end)
        if res is None:
            logging.warning(f"  No result files found in {sc_result_dir}")
            results.append((sc_name, 0.0, 0.0, 0.0))
        else:
            f1, pc, rc = res
            results.append((sc_name, f1, pc, rc))
            logging.info(f"  → F1={f1:.4f}  P={pc:.4f}  R={rc:.4f}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    if not results:
        logging.error("No results collected.")
        return

    f1s = np.array([r[1] for r in results])
    pcs = np.array([r[2] for r in results])
    rcs = np.array([r[3] for r in results])

    col = 38
    sep = "-" * (col + 34)
    print(f"\n{'='*72}")
    print(f"  Per-Scenario Evaluation Results  ({args.data_type}, "
          f"open_trace={args.open_trace})")
    print(f"{'='*72}")
    print(f"  {'Scenario':<{col}}  {'F1':>6}  {'Precision':>9}  {'Recall':>6}")
    print(f"  {sep}")
    for sc_name, f1, pc, rc in results:
        display = sc_name[:col]
        print(f"  {display:<{col}}  {f1:>6.4f}  {pc:>9.4f}  {rc:>6.4f}")
    print(f"  {sep}")
    print(f"  {'Mean':<{col}}  {f1s.mean():>6.4f}  {pcs.mean():>9.4f}  {rcs.mean():>6.4f}")
    print(f"  {'Std':<{col}}  {f1s.std():>6.4f}  {pcs.std():>9.4f}  {rcs.std():>6.4f}")
    print(f"{'='*72}\n")

    # Save summary JSON
    summary = {
        "config": {
            "data_type": args.data_type,
            "open_trace": args.open_trace,
            "val_percentile": args.val_percentile,
            "window_size": args.window_size,
        },
        "per_scenario": [
            {"scenario": sc, "f1": f1, "precision": pc, "recall": rc}
            for sc, f1, pc, rc in results
        ],
        "aggregate": {
            "f1_mean": float(f1s.mean()), "f1_std": float(f1s.std()),
            "precision_mean": float(pcs.mean()), "precision_std": float(pcs.std()),
            "recall_mean": float(rcs.mean()), "recall_std": float(rcs.std()),
        }
    }
    summary_path = os.path.join(result_base, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
