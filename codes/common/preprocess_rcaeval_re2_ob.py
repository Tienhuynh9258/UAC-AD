r"""
Preprocess the RCAEval OnlineBoutique (RE2-OB) dataset for use with UAC-AD.

Dataset facts (verified):
  - 30 scenarios: 5 services × 6 fault types
  - Each scenario: 3 independent runs (folders 1/, 2/, 3/)
  - Each run: 1441 rows @ 1-second intervals → ~24 minutes total
  - Fault inject at row ~720 → ~12 min pre-injection, ~12 min post-injection
  - 6 fault types: cpu, delay, disk, loss, mem, socket
  - 5 injected services: checkoutservice, currencyservice, emailservice,
                         productcatalogservice, recommendationservice
  - 11 monitored services (in simple_metrics.csv)

Input structure:
  <data_root>/
    {service}_{fault}/
      1/, 2/, 3/
        simple_metrics.csv  # time(unix_sec), 73 metric cols @ 1s
        logs.csv            # timestamp(ns), container_name, log_template, ...
        traces.csv          # serviceName, startTimeMillis, duration,
                            # statusCode, parentSpanID, spanID
        cluster_info.json   # template_id(str) → {template, container[]}
        inject_time.txt     # Unix timestamp (seconds)

Output (--output_dir):
  unlabel.pkl    ← 80% of normal (pre-injection) timestep samples
  train.pkl      ← 20% of normal (pre-injection) timestep samples
  test_cpu.pkl   ← shuffle(normal_sample + anomaly_cpu), anomaly ≤ 20%
  test_delay.pkl, test_disk.pkl, test_loss.pkl, test_mem.pkl, test_socket.pkl
  meta.pkl       ← {num_services, trace_c, metric_names, log_c, kpi_c, ...}

Sample format (each pkl entry = 1 timestep):
  {
    "label":               int,             # 0=normal, 1=anomaly
    "kpis":                np.float32[73],  # metric values at this timestep
    "logs":                list[str],       # template strings (for FeatureExtractor)
    "seqs":                list[str],       # same as logs (compatibility)
    "log_features":        np.float32[1],   # placeholder; overwritten by semantics.py
    "metric_name":         list[str],       # 73 metric column names
    "trace_node_features": np.float32[11,5],
    "trace_adj":           np.float32[11,11],
  }

Usage:
  python codes/common/preprocess_rcaeval_re2_ob.py \
      --data_root D:/RE2-OB/RE2-OB \
      --output_dir data/rcaeval_re2_ob \
      --anomaly_rate 0.20 \
      --unlabel_ratio 0.80
"""

import argparse
import json
import logging
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ── Constants ────────────────────────────────────────────────────────────────

FAULT_TYPES = ["cpu", "delay", "disk", "loss", "mem", "socket"]

# Service order matches simple_metrics.csv column grouping (11 services).
# "redis" in metrics corresponds to "redis-cart" in traces.
SERVICES = [
    "adservice", "cartservice", "checkoutservice", "currencyservice",
    "emailservice", "frontend", "paymentservice", "productcatalogservice",
    "recommendationservice", "redis", "shippingservice",
]
SERVICE2IDX = {s: i for i, s in enumerate(SERVICES)}
# Alias redis-cart → redis for trace lookup
SERVICE_ALIASES = {"redis-cart": "redis"}

NUM_SERVICES = len(SERVICES)
TRACE_C = 5  # [call_count, avg_dur_ms, max_dur_ms, error_rate, root_rate]


# ── Preprocessor ─────────────────────────────────────────────────────────────

class RCAEvalOBPreprocessor:

    def __init__(self, data_root, output_dir,
                 anomaly_rate=0.20, unlabel_ratio=0.80, random_seed=42):
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.anomaly_rate = anomaly_rate
        self.unlabel_ratio = unlabel_ratio
        self.rng = random.Random(random_seed)
        self.np_rng = np.random.default_rng(random_seed)
        self.canonical_metric_cols = None  # populated by _build_canonical_metric_cols()

    # ── Canonical metric columns ─────────────────────────────────────────────

    def _build_canonical_metric_cols(self):
        """
        Scan all simple_metrics.csv headers to build the union of metric columns.
        Different runs have 69–78 columns due to service availability.
        We take the UNION and fill missing cols with 0 during loading.
        Stored in self.canonical_metric_cols (sorted list, excludes 'time').
        """
        all_cols = set()
        for scenario_dir in sorted(self.data_root.iterdir()):
            if not scenario_dir.is_dir():
                continue
            parts = scenario_dir.name.rsplit("_", 1)
            if len(parts) != 2 or parts[1] not in FAULT_TYPES:
                continue
            for run_id in [1, 2, 3]:
                path = scenario_dir / str(run_id) / "simple_metrics.csv"
                if path.exists():
                    cols = pd.read_csv(path, nrows=0).columns.tolist()
                    all_cols.update(cols[1:])  # skip 'time'
        self.canonical_metric_cols = sorted(all_cols)
        logging.info(f"Canonical metric cols: {len(self.canonical_metric_cols)} "
                     f"(union across all runs)")

    # ── Loaders ──────────────────────────────────────────────────────────────

    def _load_inject_time(self, exp_dir: Path) -> int:
        return int(Path(exp_dir, "inject_time.txt").read_text().strip())

    def _load_kpi(self, exp_dir: Path):
        """
        Returns (timestamps [T], kpis [T, N], metric_names list[N]).
        All runs are aligned to self.canonical_metric_cols (missing cols → 0).
        """
        df = pd.read_csv(exp_dir / "simple_metrics.csv")
        timestamps = pd.to_numeric(df.iloc[:, 0], errors="coerce").fillna(0).values.astype(np.int64)
        # Align to canonical columns (reindex fills missing with NaN → 0)
        df_metrics = df.iloc[:, 1:].copy()
        df_metrics = df_metrics.reindex(columns=self.canonical_metric_cols, fill_value=0.0)
        kpis = df_metrics.values.astype(np.float32)
        return timestamps, kpis, self.canonical_metric_cols

    def _load_cluster_info(self, exp_dir: Path) -> dict:
        """Returns {template_id_str: template_text}."""
        with open(exp_dir / "cluster_info.json") as f:
            raw = json.load(f)
        return {k: v["template"] for k, v in raw.items()}

    # ── Log features ─────────────────────────────────────────────────────────

    def _build_log_features(self, exp_dir: Path, timestamps: np.ndarray):
        """
        Returns:
          logs [T] — list[list[str]]: template strings per timestep

        Two logs.csv formats exist in this dataset:
          - Full (24/90 runs):    has 'log_template' column → use directly (~145 unique templates)
          - Minimal (66/90 runs): no 'log_template' column  → fallback to 'container_name'
                                  (11 unique values, one per service)

        Using raw 'container_name + message' as fallback causes 86k+ unique strings,
        making FeatureExtractor extremely slow. container_name-only fallback keeps
        total vocab at ~156 unique templates (145 from log_template + 11 service names).
        """
        T = len(timestamps)
        logs_per_ts = [[] for _ in range(T)]

        try:
            raw = pd.read_csv(
                exp_dir / "logs.csv",
                usecols=lambda c: c in ["timestamp", "container_name", "log_template"],
                low_memory=False,
            )
            raw = raw.dropna(subset=["timestamp"])
            raw["timestamp"] = pd.to_numeric(raw["timestamp"], errors="coerce")
            raw = raw.dropna(subset=["timestamp"])

            # Use log_template when available; fallback to container_name only
            if "log_template" in raw.columns:
                templates = raw["log_template"].fillna("").astype(str)
            else:
                templates = raw.get(
                    "container_name", pd.Series(["padding"] * len(raw))
                ).fillna("padding").astype(str)

            # Convert nanoseconds → seconds, assign to timestep bucket
            ts_sec = raw["timestamp"].values.astype(np.int64) // 1_000_000_000
            wi = np.searchsorted(timestamps, ts_sec, side="right") - 1
            valid = (wi >= 0) & (wi < T)

            for idx, tmpl in zip(wi[valid], templates.values[valid]):
                t = str(tmpl).strip()
                if t:
                    logs_per_ts[idx].append(t)

        except Exception as e:
            logging.warning(f"  Logs load failed for {exp_dir}: {e}")

        for i in range(T):
            if not logs_per_ts[i]:
                logs_per_ts[i] = ["padding"]

        return logs_per_ts

    # ── Trace features ────────────────────────────────────────────────────────

    def _build_trace_features(self, exp_dir: Path, timestamps: np.ndarray):
        """
        Returns:
          node_feats [T, NUM_SERVICES, TRACE_C]
          adj        [T, NUM_SERVICES, NUM_SERVICES]  (row-normalized)
        """
        T = len(timestamps)
        node_feats = np.zeros((T, NUM_SERVICES, TRACE_C), dtype=np.float32)
        adj = np.zeros((T, NUM_SERVICES, NUM_SERVICES), dtype=np.float32)

        try:
            df = pd.read_csv(
                exp_dir / "traces.csv",
                usecols=["serviceName", "startTimeMillis", "duration",
                          "statusCode", "parentSpanID", "spanID"],
                low_memory=False,
            )
            df = df.dropna(subset=["serviceName", "startTimeMillis"])

            # Normalize service names (redis-cart → redis)
            df["serviceName"] = df["serviceName"].replace(SERVICE_ALIASES)

            # Map service → index; drop unknown services
            df["si"] = df["serviceName"].map(SERVICE2IDX)
            df = df.dropna(subset=["si"])
            df["si"] = df["si"].astype(np.int32)

            # Assign to time bucket (startTimeMillis / 1000 → seconds)
            ts_sec = df["startTimeMillis"].values / 1000.0
            wi = np.searchsorted(timestamps, ts_sec, side="right") - 1
            valid = (wi >= 0) & (wi < T)
            df = df[valid].copy()
            df["wi"] = wi[valid].astype(np.int32)

            if df.empty:
                return node_feats, adj

            # ── Node features (vectorised) ────────────────────────────────
            dur = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0).clip(lower=0)
            status = df["statusCode"].astype(str).str.strip()
            is_error = (~status.isin(["0", "0.0", "nan"])).astype(np.float64).values

            parent_col = df["parentSpanID"].astype(str).str.strip()
            is_root = parent_col.isin(["", "0", "nan", "None"]).astype(np.float64).values

            wi_arr = df["wi"].values
            si_arr = df["si"].values
            dur_arr = dur.values

            acc_count = np.zeros((T, NUM_SERVICES), dtype=np.float64)
            acc_dur_sum = np.zeros((T, NUM_SERVICES), dtype=np.float64)
            acc_max_dur = np.zeros((T, NUM_SERVICES), dtype=np.float64)
            acc_errors = np.zeros((T, NUM_SERVICES), dtype=np.float64)
            acc_roots = np.zeros((T, NUM_SERVICES), dtype=np.float64)

            np.add.at(acc_count,   (wi_arr, si_arr), 1.0)
            np.add.at(acc_dur_sum, (wi_arr, si_arr), dur_arr)
            np.maximum.at(acc_max_dur, (wi_arr, si_arr), dur_arr)
            np.add.at(acc_errors,  (wi_arr, si_arr), is_error)
            np.add.at(acc_roots,   (wi_arr, si_arr), is_root)

            safe = np.where(acc_count > 0, acc_count, 1.0)
            node_feats[:, :, 0] = acc_count
            node_feats[:, :, 1] = acc_dur_sum / safe
            node_feats[:, :, 2] = acc_max_dur
            node_feats[:, :, 3] = acc_errors / safe
            node_feats[:, :, 4] = acc_roots / safe

            # Normalize call_count, avg_dur, max_dur to [0,1]
            for col in [0, 1, 2]:
                mx = node_feats[:, :, col].max()
                if mx > 0:
                    node_feats[:, :, col] /= mx

            # ── Adjacency (parent → child edges, vectorised) ──────────────
            # Build spanID → (si, wi) lookup via merge
            span_df = df[["spanID", "si", "wi"]].copy()
            span_df = span_df.drop_duplicates(subset=["spanID"])
            span_df.columns = ["parentSpanID", "parent_si", "parent_wi"]

            edge_df = df[["parentSpanID", "si", "wi"]].copy()
            edge_df.columns = ["parentSpanID", "child_si", "child_wi"]

            # Only keep non-root spans
            non_root_mask = ~df["parentSpanID"].isin(["", "0", "nan", "None"])
            edge_df = edge_df[non_root_mask.values]

            merged = edge_df.merge(span_df, on="parentSpanID", how="inner")
            # Keep only same-window edges
            merged = merged[merged["child_wi"] == merged["parent_wi"]]

            if not merged.empty:
                np.add.at(
                    adj,
                    (merged["child_wi"].values,
                     merged["parent_si"].values,
                     merged["child_si"].values),
                    1.0,
                )
                # Row-normalize adjacency (safe: avoid div-by-zero warnings)
                row_sum = adj.sum(axis=2, keepdims=True)
                safe_denom = np.where(row_sum > 0, row_sum, 1.0)
                adj = (adj / safe_denom * (row_sum > 0)).astype(np.float32)

        except Exception as e:
            logging.warning(f"  Trace load failed for {exp_dir}: {e}")

        return node_feats, adj

    # ── Labels ───────────────────────────────────────────────────────────────

    def _build_labels(self, timestamps: np.ndarray, inject_time: int) -> np.ndarray:
        return (timestamps >= inject_time).astype(np.int32)

    # ── Build all timestep samples for one experiment ─────────────────────────

    def _process_experiment(self, exp_dir: Path, service: str, fault: str, run_id: int):
        """
        Returns:
          normal_samples  dict[id → sample]  (pre-injection, label=0)
          anomaly_samples dict[id → sample]  (post-injection, label=1)
        """
        inject_time = self._load_inject_time(exp_dir)
        timestamps, kpis, metric_names = self._load_kpi(exp_dir)
        logs_per_ts = self._build_log_features(exp_dir, timestamps)
        node_feats, adj = self._build_trace_features(exp_dir, timestamps)
        labels = self._build_labels(timestamps, inject_time)

        normal_samples, anomaly_samples = {}, {}

        for i in range(len(timestamps)):
            sid = f"{service}_{fault}_{run_id}_{i}"
            sample = {
                "label":               int(labels[i]),
                "kpis":                kpis[i].copy(),
                "logs":                logs_per_ts[i],
                "seqs":                logs_per_ts[i],
                "log_features":        np.zeros(1, dtype=np.float32),  # placeholder
                "metric_name":         metric_names,
                "trace_node_features": node_feats[i].copy(),
                "trace_adj":           adj[i].copy(),
            }
            if labels[i] == 0:
                normal_samples[sid] = sample
            else:
                anomaly_samples[sid] = sample

        return normal_samples, anomaly_samples

    # ── Save pickle ───────────────────────────────────────────────────────────

    def _save_pkl(self, data: dict, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logging.info(f"  Saved {path}  ({len(data)} samples)")

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── Pre-pass: build canonical metric column list ───────────────────────
        self._build_canonical_metric_cols()

        all_normal = {}                              # all pre-injection samples
        fault_anomaly = {f: {} for f in FAULT_TYPES}  # post-injection per fault

        # ── Pass: process all 90 experiments ─────────────────────────────────
        for scenario_dir in sorted(self.data_root.iterdir()):
            if not scenario_dir.is_dir():
                continue
            # Parse service and fault from folder name (e.g. checkoutservice_cpu)
            parts = scenario_dir.name.rsplit("_", 1)
            if len(parts) != 2 or parts[1] not in FAULT_TYPES:
                continue
            service, fault = parts

            for run_id in [1, 2, 3]:
                exp_dir = scenario_dir / str(run_id)
                if not exp_dir.exists():
                    logging.warning(f"  Missing: {exp_dir}")
                    continue

                logging.info(f"Processing {scenario_dir.name}/run{run_id} …")
                normal, anomaly = self._process_experiment(
                    exp_dir, service, fault, run_id
                )
                all_normal.update(normal)
                fault_anomaly[fault].update(anomaly)

                logging.info(
                    f"  → normal={len(normal)}  anomaly={len(anomaly)}"
                )

        # ── Split normal → unlabel / train ────────────────────────────────────
        normal_ids = list(all_normal.keys())
        self.rng.shuffle(normal_ids)
        split = int(len(normal_ids) * self.unlabel_ratio)
        unlabel_ids = set(normal_ids[:split])
        train_ids   = set(normal_ids[split:])

        unlabel_data = {k: all_normal[k] for k in unlabel_ids}
        train_data   = {k: all_normal[k] for k in train_ids}

        logging.info(
            f"Normal pool: {len(all_normal)} total  "
            f"→ unlabel={len(unlabel_data)}  train={len(train_data)}"
        )
        self._save_pkl(unlabel_data, self.output_dir / "unlabel.pkl")
        self._save_pkl(train_data,   self.output_dir / "train.pkl")

        # ── Build 6 test files ────────────────────────────────────────────────
        normal_pool = list(all_normal.keys())

        for fault in FAULT_TYPES:
            anom_ids = list(fault_anomaly[fault].keys())
            n_anom = len(anom_ids)

            if n_anom == 0:
                logging.warning(f"  No anomaly samples for fault={fault}, skipping.")
                continue

            # Sample normal so that anomaly_rate = n_anom / (n_anom + n_normal) <= target
            max_anom_rate = self.anomaly_rate
            n_normal_needed = int(n_anom * (1 - max_anom_rate) / max_anom_rate)

            available_normal = [k for k in normal_pool if k not in fault_anomaly[fault]]
            if len(available_normal) < n_normal_needed:
                logging.warning(
                    f"  fault={fault}: need {n_normal_needed} normal but only "
                    f"{len(available_normal)} available — using all."
                )
                sampled_normal_ids = available_normal
            else:
                sampled_normal_ids = self.rng.sample(available_normal, n_normal_needed)

            test_data = {}
            for k in sampled_normal_ids:
                test_data[k] = all_normal[k]
            for k in anom_ids:
                test_data[k] = fault_anomaly[fault][k]

            # Shuffle key order (Python 3.7+ dicts preserve insertion order)
            all_ids = list(test_data.keys())
            self.rng.shuffle(all_ids)
            test_data = {k: test_data[k] for k in all_ids}

            achieved_rate = n_anom / len(test_data)
            logging.info(
                f"  test_{fault}: {len(test_data)} samples  "
                f"anomaly_rate={achieved_rate:.2%}  "
                f"(normal={len(sampled_normal_ids)}  anomaly={n_anom})"
            )
            self._save_pkl(test_data, self.output_dir / f"test_{fault}.pkl")

        # ── meta.pkl ──────────────────────────────────────────────────────────
        n_kpi = len(self.canonical_metric_cols)
        meta = {
            "num_services": NUM_SERVICES,
            "service2idx":  SERVICE2IDX,
            "trace_c":      TRACE_C,
            "kpi_c":        n_kpi,
            "log_c":        1,       # placeholder; updated by semantics.py at runtime
            "metric_names": self.canonical_metric_cols,
            "fault_types":  FAULT_TYPES,
            "services":     SERVICES,
        }
        self._save_pkl(meta, self.output_dir / "meta.pkl")
        logging.info("Preprocessing complete.")
        logging.info(
            f"  Run model with:\n"
            f"    python codes/common/eval_per_scenario_rcaeval_re2_ob.py \\\n"
            f"        --data {self.output_dir} --dataset rcaeval_re2_ob \\\n"
            f"        --data_type fuse --open_trace True"
        )


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Preprocess RCAEval OnlineBoutique → UAC-AD pkl format"
    )
    p.add_argument("--data_root",     required=True,
                   help="Root dir of RE2-OB (contains checkoutservice_cpu/, ...)")
    p.add_argument("--output_dir",    default="../../data/rcaeval_re2_ob")
    p.add_argument("--anomaly_rate",  default=0.20, type=float,
                   help="Max anomaly fraction in each test_<fault>.pkl (default 0.20)")
    p.add_argument("--unlabel_ratio", default=0.80, type=float,
                   help="Fraction of normal data → unlabel.pkl (rest → train.pkl)")
    p.add_argument("--random_seed",   default=42, type=int)
    args = p.parse_args()

    RCAEvalOBPreprocessor(
        data_root    = args.data_root,
        output_dir   = args.output_dir,
        anomaly_rate = args.anomaly_rate,
        unlabel_ratio= args.unlabel_ratio,
        random_seed  = args.random_seed,
    ).run()


if __name__ == "__main__":
    main()
