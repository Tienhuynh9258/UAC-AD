"""
Per-scenario evaluation for TrainTicket (AnoMod TT) dataset.

For each anomaly scenario, runs HADES on:
    train.pkl  = Normal_case windows (same for all scenarios)
    test_pkl   = scenarios/test_{scenario_name}.pkl
               = Normal_case windows shuffled with that scenario's anomaly windows

Aggregates F1 / Precision / Recall per scenario AND per fault type:
    Lv_P → Performance  (CPU, DiskIO, NetLoss)
    Lv_S → Service      (DNSFail, HTTPAbort, KillPod)
    Lv_D → Database     (CacheLimit, ConnectionPool, TransactionTimeout)
    Lv_C → Code         (ExceptionInjection, SecurityCheck, TravelDetailFailure)

Usage:
    python codes/common/eval_per_scenario_tt.py \\
        --data data/tt \\
        --data_type fuse \\
        --open_trace False \\
        --epoches 10 10 --batch_size 256 --patience 5 \\
        --window_size 5 --val_percentile 95 \\
        --run_start 0 --run_end 1
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from collections import defaultdict

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

# Fault type mapping from scenario name prefix
_FAULT_TYPES = {
    "Lv_P": "Performance",
    "Lv_S": "Service",
    "Lv_D": "Database",
    "Lv_C": "Code",
}

# Short display names for each scenario (strip common prefix/suffix noise)
_SHORT_NAME_RE = re.compile(
    r'^(Lv_[PSDC])_(.+?)(?:_preserve)?(?:_no_order)?_\d{8}T\d{6}Z_em$',
    re.IGNORECASE,
)


def _short_name(sc_name: str) -> str:
    """Convert long scenario folder name to a concise display name.

    e.g. 'Lv_P_CPU_preserve_20251103T140939Z_em' → 'Lv_P / CPU'
         'Lv_S_DNSFAIL_preserve_no_order_...'    → 'Lv_S / DNSFAIL'
    """
    m = _SHORT_NAME_RE.match(sc_name)
    if m:
        return f"{m.group(1)} / {m.group(2).upper()}"
    # Fallback: strip trailing timestamp
    return re.sub(r'_\d{8}T\d{6}Z_em$', '', sc_name)


def _fault_type(sc_name: str) -> str:
    """Return fault type label from scenario name prefix."""
    for prefix, label in _FAULT_TYPES.items():
        if sc_name.startswith(prefix):
            return label
    return "Unknown"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _find_scenario_files(data_dir: str):
    """Return sorted list of (scenario_name, test_pkl_path)."""
    scenarios_dir = os.path.join(data_dir, "scenarios")
    if not os.path.isdir(scenarios_dir):
        raise FileNotFoundError(
            f"scenarios/ folder not found under {data_dir}. "
            f"Re-run preprocess_tt.py first."
        )
    entries = []
    for fname in sorted(os.listdir(scenarios_dir)):
        if fname.startswith("test_") and fname.endswith(".pkl"):
            sc_name = fname[len("test_"):-len(".pkl")]
            entries.append((sc_name, os.path.join(scenarios_dir, fname)))
    return entries


def _scan_latest_results(result_dir: str):
    """
    Scan result_dir recursively for info_score.txt files.
    Each file contains a line like:
        * Test -- f1:0.2174\trc:0.1220\tpc:1.0000
    Returns (best_f1, best_pc, best_rc) or None if not found.
    """
    if not os.path.isdir(result_dir):
        return None

    best_f1 = best_pc = best_rc = -1.0
    pat = re.compile(r"f1:([\d.]+)\s+rc:([\d.]+)\s+pc:([\d.]+)")

    for root, _, files in os.walk(result_dir):
        for fname in files:
            if fname != "info_score.txt":
                continue
            try:
                with open(os.path.join(root, fname)) as f:
                    for line in f:
                        m = pat.search(line)
                        if m:
                            f1 = float(m.group(1))
                            if f1 > best_f1:
                                best_f1 = f1
                                best_rc = float(m.group(2))
                                best_pc = float(m.group(3))
            except Exception:
                continue

    return (best_f1, best_pc, best_rc) if best_f1 >= 0 else None


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Per-scenario evaluation for TrainTicket (AnoMod TT) dataset"
    )
    # ── data / model args (passed through to run.py) ─────────────────────────
    p.add_argument("--data",            required=True,
                   help="Path to preprocessed TT data dir (contains train.pkl, scenarios/)")
    p.add_argument("--dataset",         default="sn",
                   help="Dataset tag passed to run.py (keep 'sn' — TT uses same format)")
    p.add_argument("--data_type",       default="fuse",   choices=["fuse", "kpi", "log"])
    p.add_argument("--open_trace",      default="False",
                   help="Enable trace branch (True/False)")
    p.add_argument("--num_services",    default=20,       type=int,
                   help="Number of services (default 20 for TT top-K)")
    p.add_argument("--trace_c",         default=5,        type=int)
    p.add_argument("--epoches",         default=[10, 10], nargs="+", type=int)
    p.add_argument("--batch_size",      default=256,      type=int)
    p.add_argument("--patience",        default=5,        type=int)
    p.add_argument("--window_size",     default=5,        type=int,
                   help="Sequence window for model input (default 5 for TT: 14 trimmed normal "
                        "windows → 11 train → 7 training sequences with window_size=5)")
    p.add_argument("--val_percentile",  default=95,       type=float,
                   help="Percentile of normal train losses used as anomaly threshold")
    p.add_argument("--alpha",           default=0.16,     type=float)
    p.add_argument("--open_gan_sep",    default="True")
    p.add_argument("--run_start",       default=0,        type=int)
    p.add_argument("--run_end",         default=1,        type=int)
    p.add_argument("--result_dir",      default=None,
                   help="Base result dir. Defaults to {data}/result_per_scenario_{data_type}")
    p.add_argument("--run_py",          default=None,
                   help="Path to run.py (auto-detected if not specified)")
    args = p.parse_args()

    # Resolve data path to absolute so run.py (which runs from codes/ dir) finds it
    args.data = os.path.abspath(args.data)

    # Auto-detect run.py (lives one level above this file: codes/run.py)
    if args.run_py:
        run_py = args.run_py
    else:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        run_py   = os.path.normpath(os.path.join(this_dir, "..", "run.py"))
    if not os.path.exists(run_py):
        raise FileNotFoundError(f"run.py not found at {run_py}. Use --run_py to specify.")

    # Append trace suffix to result dir when trace is enabled
    trace_suffix = "_trace" if args.open_trace.lower() == "true" else "_baseline"
    result_base  = args.result_dir or os.path.join(
        args.data, f"result_per_scenario_{args.data_type}{trace_suffix}"
    )

    scenario_files = _find_scenario_files(args.data)
    if not scenario_files:
        logging.error("No scenario test files found. Re-run preprocess_tt.py.")
        sys.exit(1)

    logging.info(f"Found {len(scenario_files)} scenarios to evaluate.")
    logging.info(f"Results will be saved under: {result_base}/\n")

    # ── Run per scenario ──────────────────────────────────────────────────────
    results = []   # list of (sc_name, f1, pc, rc)

    for sc_name, test_pkl in scenario_files:
        # Use short name as result subfolder (safe for filesystem)
        sc_folder     = re.sub(r'[^A-Za-z0-9_]', '_', _short_name(sc_name))
        sc_result_dir = os.path.join(result_base, sc_folder)
        os.makedirs(sc_result_dir, exist_ok=True)

        logging.info(f"{'='*60}")
        logging.info(f"Scenario : {_short_name(sc_name)}  [{_fault_type(sc_name)}]")
        logging.info(f"  test   : {os.path.basename(test_pkl)}")
        logging.info(f"  result : {sc_result_dir}")

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
            logging.error(f"  Failed to run scenario {sc_name}: {e}")
            results.append((sc_name, 0.0, 0.0, 0.0))
            continue

        res = _scan_latest_results(sc_result_dir)
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

    # ── Print per-scenario table ──────────────────────────────────────────────
    COL = 22
    SEP = "-" * (COL + 46)
    print(f"\n{'='*72}")
    print(f"  TrainTicket Per-Scenario Results  "
          f"({args.data_type}, open_trace={args.open_trace})")
    print(f"{'='*72}")
    print(f"  {'Scenario':<{COL}}  {'FaultType':<12}  {'F1':>6}  {'Precision':>9}  {'Recall':>6}")
    print(f"  {SEP}")

    # Group print by fault type
    by_type = defaultdict(list)
    for sc_name, f1, pc, rc in results:
        ft = _fault_type(sc_name)
        by_type[ft].append((sc_name, f1, pc, rc))

    for ft in ["Performance", "Service", "Database", "Code"]:
        group = by_type.get(ft, [])
        for sc_name, f1, pc, rc in group:
            display = _short_name(sc_name)[:COL]
            print(f"  {display:<{COL}}  {ft:<12}  {f1:>6.4f}  {pc:>9.4f}  {rc:>6.4f}")
        if group:
            gf1s = np.array([r[1] for r in group])
            gpcs = np.array([r[2] for r in group])
            grcs = np.array([r[3] for r in group])
            print(f"  {'  '+ft+' avg':<{COL}}  {'':12}  "
                  f"{gf1s.mean():>6.4f}  {gpcs.mean():>9.4f}  {grcs.mean():>6.4f}")
            print(f"  {'-'*(COL+46)}")

    print(f"  {'Overall Mean':<{COL}}  {'':12}  "
          f"{f1s.mean():>6.4f}  {pcs.mean():>9.4f}  {rcs.mean():>6.4f}")
    print(f"  {'Overall Std':<{COL}}  {'':12}  "
          f"{f1s.std():>6.4f}  {pcs.std():>9.4f}  {rcs.std():>6.4f}")
    print(f"{'='*72}\n")

    # ── Save summary JSON ─────────────────────────────────────────────────────
    ft_agg = {}
    for ft, group in by_type.items():
        gf1s = np.array([r[1] for r in group])
        gpcs = np.array([r[2] for r in group])
        grcs = np.array([r[3] for r in group])
        ft_agg[ft] = {
            "f1_mean":        float(gf1s.mean()),
            "f1_std":         float(gf1s.std()),
            "precision_mean": float(gpcs.mean()),
            "recall_mean":    float(grcs.mean()),
        }

    summary = {
        "config": {
            "data_type":      args.data_type,
            "open_trace":     args.open_trace,
            "val_percentile": args.val_percentile,
            "window_size":    args.window_size,
            "num_services":   args.num_services,
        },
        "per_scenario": [
            {
                "scenario":   sc,
                "short_name": _short_name(sc),
                "fault_type": _fault_type(sc),
                "f1":         f1,
                "precision":  pc,
                "recall":     rc,
            }
            for sc, f1, pc, rc in results
        ],
        "per_fault_type": ft_agg,
        "aggregate": {
            "f1_mean":        float(f1s.mean()),
            "f1_std":         float(f1s.std()),
            "precision_mean": float(pcs.mean()),
            "precision_std":  float(pcs.std()),
            "recall_mean":    float(rcs.mean()),
            "recall_std":     float(rcs.std()),
        },
    }

    os.makedirs(result_base, exist_ok=True)
    summary_path = os.path.join(result_base, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
