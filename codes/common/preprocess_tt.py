"""
Preprocess TrainTicket (AnoMod TT) → UAC-AD pkl format.

Dataset layout expected under TT_DATA_ROOT/:
  log_data/    {scenario}/  {service_subdir}/  *.log   (Spring Boot logs)
  metric_data/ {scenario}/  {scenario}_metrics_*.csv   (Prometheus, 1 CSV/scenario)
  trace_data/  {scenario}/  {scenario}_traces_*.json   (SkyWalking format)

Output (OUTPUT_DIR/):
  train.pkl              — first 80% of Normal_case windows (normal only)
  unlabel.pkl            — same as train.pkl
  val.pkl                — last 20% of Normal_case windows (threshold tuning)
  meta.pkl               — dataset metadata
  scenarios/
    test_{name}.pkl      — per-scenario test file: ~41 normal + 8 anomaly windows

KPI features (6 system + 4 * top_k_services container = 86 by default):
  System (6):    cpu_rate, load5, mem_avail_bytes,
                 disk_read_rate, disk_write_rate, net_rx_bytes
  Container (4 per service): container_cpu, container_mem,
                              container_net_rx_err, container_net_tx_err

Trace node features per service (5-dim):
  [call_count, avg_dur_ms, max_dur_ms, error_rate, root_rate]

Static adjacency [N x N]: built from Normal_case traces.

Usage:
    python codes/common/preprocess_tt.py \\
        --tt_data_root D:/AnoMod/TT_data \\
        --output_dir data/tt \\
        --window_sec 15 --warmup_minutes 5 \\
        --max_anomaly_windows 8 --top_k_services 20 --seed 42
"""

import argparse
import glob
import hashlib
import json
import logging
import os
import pickle
import random
import re
from collections import Counter
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

# ─── Constants ────────────────────────────────────────────────────────────────

TRACE_NODE_FEAT_DIM = 5  # [call_count, avg_dur_ms, max_dur_ms, error_rate, root_rate]

# System-level metrics (aggregated across all nodes per window)
SYSTEM_METRICS = [
    "rate(node_cpu_seconds_total[5m])",
    "node_load5",
    "node_memory_MemAvailable_bytes",
    "rate(node_disk_read_bytes_total[5m])",
    "rate(node_disk_written_bytes_total[5m])",
    "node_network_receive_bytes_total",
]

# Container-level metrics (one value per service per window)
CONTAINER_METRICS = [
    "container_cpu_usage_seconds_total",
    "container_memory_usage_bytes",
    "container_network_receive_errors_total",
    "container_network_transmit_errors_total",
]

# Short names for container metrics (used as column name prefix)
_CONTAINER_SHORT = {
    "container_cpu_usage_seconds_total":      "cpu",
    "container_memory_usage_bytes":           "mem",
    "container_network_receive_errors_total": "net_rx_err",
    "container_network_transmit_errors_total":"net_tx_err",
}

# Spring Boot log: "2025-11-03 14:09:38.123  INFO ..."
_LOG_TS_RE  = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)')
_LOG_LVL_RE = re.compile(r'\b(INFO|WARN|WARNING|ERROR|DEBUG|TRACE)\b')


# ─── Preprocessor ─────────────────────────────────────────────────────────────

class TTPreprocessor:
    """Preprocess TrainTicket (AnoMod) dataset → UAC-AD pkl format."""

    def __init__(
        self,
        tt_data_root: str,
        output_dir: str,
        window_sec: int = 15,
        warmup_minutes: int = 5,
        max_anomaly_windows: int = 8,
        top_k_services: int = 20,
        seed: int = 42,
    ):
        self.tt_data_root        = tt_data_root
        self.output_dir          = output_dir
        self.window_sec          = window_sec
        self.warmup_minutes      = warmup_minutes
        self.max_anomaly_windows = max_anomaly_windows
        self.top_k_services      = top_k_services
        self.seed                = seed
        self.rng                 = random.Random(seed)

        self.log_dir    = os.path.join(tt_data_root, "log_data")
        self.metric_dir = os.path.join(tt_data_root, "metric_data")
        self.trace_dir  = os.path.join(tt_data_root, "trace_data")

        # Populated during run()
        self.services: List[str] = []
        self.num_services: int   = 0
        self.service2idx: Dict[str, int] = {}

        self._miner      = None   # Drain3 TemplateMiner fitted on Normal_case
        self._adj        = None   # Static adjacency [N, N] from Normal_case
        self._metric_names: List[str] = []

    # ── Scenario discovery ────────────────────────────────────────────────────

    def _scenario_dirs(self) -> Dict[str, Dict[str, str]]:
        """
        Map each scenario name → {log_dir, metric_dir, trace_dir}.
        In TT the scenario folder name is the same across all three modality dirs.
        """
        def _subdirs(base: str) -> set:
            return {d for d in os.listdir(base)
                    if os.path.isdir(os.path.join(base, d))}

        common = sorted(
            _subdirs(self.log_dir)
            & _subdirs(self.metric_dir)
            & _subdirs(self.trace_dir)
        )
        logging.info(f"Found {len(common)} scenarios: {common}")
        return {
            sc: {
                "log_dir":    os.path.join(self.log_dir,    sc),
                "metric_dir": os.path.join(self.metric_dir, sc),
                "trace_dir":  os.path.join(self.trace_dir,  sc),
            }
            for sc in common
        }

    # ── Step 1: Discover top-K services from Normal_case traces ───────────────

    def _discover_services(self, trace_dir: str) -> List[str]:
        """
        Count call_count per service from Normal_case trace JSON.
        Return top-K service names sorted by descending call count.
        """
        json_path = self._find_trace_json(trace_dir)
        if json_path is None:
            raise FileNotFoundError(f"No trace JSON found in {trace_dir}")

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        counter: Counter = Counter()
        for trace in data.get("traces", []):
            for span in trace.get("spans", []):
                svc = span.get("service_code", "")
                if svc:
                    counter[svc] += 1

        top = [svc for svc, _ in counter.most_common(self.top_k_services)]
        logging.info(f"  Top-{self.top_k_services} services by call count:")
        for svc, cnt in counter.most_common(self.top_k_services):
            logging.info(f"    {cnt:>5}  {svc}")
        return top

    def _pod_to_service(self, pod_name: str) -> Optional[str]:
        """
        Map a Kubernetes pod name to a canonical service name.
        Strategy: find the longest known service name that is a prefix
        of the pod name (k8s pods are named {service}-{rs_hash}-{pod_hash}).
        """
        if not pod_name or not isinstance(pod_name, str):
            return None
        for svc in self.services:          # services sorted longest-first
            if pod_name.startswith(svc):
                return svc
        return None

    # ── Step 2: Build static adjacency from Normal_case traces ───────────────

    def _build_static_adj(self, trace_dir: str) -> np.ndarray:
        """Build boolean adjacency matrix [N, N] from Normal_case trace spans."""
        N   = self.num_services
        adj = np.zeros((N, N), dtype=np.float32)

        json_path = self._find_trace_json(trace_dir)
        if json_path is None:
            logging.warning(f"  No trace JSON in {trace_dir}, returning zero adj")
            return adj

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        # node_id is globally unique per span in SkyWalking
        node2svc: Dict[str, str] = {}
        for trace in data.get("traces", []):
            for span in trace.get("spans", []):
                node_id = span.get("node_id")
                svc     = span.get("service_code", "")
                if node_id is not None:
                    node2svc[str(node_id)] = svc

        edge_count = 0
        for trace in data.get("traces", []):
            for span in trace.get("spans", []):
                child_svc  = span.get("service_code", "")
                parent_nid = span.get("parent_node_id")

                if parent_nid is None or parent_nid == -1:
                    continue
                parent_svc = node2svc.get(str(parent_nid))
                if not parent_svc or parent_svc == child_svc:
                    continue

                ci = self.service2idx.get(child_svc,  -1)
                pi = self.service2idx.get(parent_svc, -1)
                if ci >= 0 and pi >= 0 and adj[pi, ci] == 0:
                    adj[pi, ci] = 1.0   # parent → child
                    adj[ci, pi] = 1.0   # undirected
                    edge_count += 1

        np.fill_diagonal(adj, 1.0)
        logging.info(f"  Static adj: {edge_count} edges (symmetric) "
                     f"from {len(node2svc)} spans")
        return adj

    # ── Step 3: Fit Drain3 on Normal_case logs ────────────────────────────────

    def _fit_drain3(self, log_dir: str) -> None:
        """Walk all .log files under log_dir (recursive), fit Drain3."""
        try:
            from drain3 import TemplateMiner
            from drain3.template_miner_config import TemplateMinerConfig
        except ImportError:
            logging.warning("drain3 not installed — falling back to regex templates")
            self._miner = None
            return

        config = TemplateMinerConfig()
        config.drain_depth                = 4
        config.drain_sim_th               = 0.5
        config.drain_max_children         = 100
        config.parametrize_numeric_tokens = True

        miner  = TemplateMiner(config=config)
        n_msgs = 0

        for fpath in self._iter_log_files(log_dir):
            with open(fpath, encoding="utf-8", errors="replace") as f:
                for line in f:
                    content = self._extract_log_content(line)
                    if content:
                        miner.add_log_message(content)
                        n_msgs += 1

        n_tmpl = len(miner.drain.id_to_cluster)
        logging.info(f"  Drain3 fitted on {n_msgs:,} messages → {n_tmpl} templates")
        self._miner = miner

    def _extract_log_content(self, line: str) -> Optional[str]:
        """Strip Spring Boot timestamp prefix, return the rest."""
        m = _LOG_TS_RE.match(line)
        if m:
            return line[m.end():].strip() or None
        return line.strip() or None

    def _to_template(self, line: str) -> str:
        content = self._extract_log_content(line)
        if not content:
            return "padding"
        if self._miner is not None:
            result = self._miner.add_log_message(content)
            return result["template_mined"] if result else content
        # Fallback: normalise numbers and hex tokens
        return re.sub(r'[0-9a-f]{8,}', '<*>',
                      re.sub(r'\d+', '<NUM>', content))

    # ── Step 4: Build per-window KPI matrix ───────────────────────────────────

    def _load_metric_df(self, metric_dir: str) -> pd.DataFrame:
        """
        Load the single Prometheus CSV in metric_dir.
        Returns DataFrame with columns:
          metric_name, timestamp (datetime), value, kubernetes_pod_name
        """
        csv_files = glob.glob(os.path.join(metric_dir, "*.csv"))
        if not csv_files:
            logging.warning(f"  No metric CSV in {metric_dir}")
            return pd.DataFrame()

        # Determine pod column name (prefer kubernetes_pod_name, fallback to pod)
        header = pd.read_csv(csv_files[0], nrows=0)
        pod_col = ("kubernetes_pod_name"
                   if "kubernetes_pod_name" in header.columns
                   else "pod")

        usecols = ["metric_name", "timestamp", "value", pod_col]
        usecols = [c for c in usecols if c in header.columns]

        df = pd.read_csv(csv_files[0], usecols=usecols, low_memory=False)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        if pod_col != "kubernetes_pod_name":
            df = df.rename(columns={pod_col: "kubernetes_pod_name"})
        elif "kubernetes_pod_name" not in df.columns:
            df["kubernetes_pod_name"] = np.nan

        return df

    def _build_kpi_matrix(
        self,
        metric_dir: str,
        win_starts: List[datetime],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build [W, 6 + 4*N] KPI matrix for one scenario.
        Returns (matrix, metric_column_names).
        """
        W     = len(win_starts)
        delta = timedelta(seconds=self.window_sec)

        df = self._load_metric_df(metric_dir)
        if df.empty:
            return np.zeros((W, 0), dtype=np.float32), []

        # ── System metrics: mean across all nodes per timestamp ───────────────
        sys_available = [m for m in SYSTEM_METRICS if m in df["metric_name"].values]
        missing_sys   = [m for m in SYSTEM_METRICS if m not in sys_available]
        if missing_sys:
            logging.warning(f"  Missing system metrics: {missing_sys}")

        sys_df = (
            df[df["metric_name"].isin(sys_available)]
            .groupby(["timestamp", "metric_name"])["value"]
            .mean()
            .unstack("metric_name")
            .reindex(columns=sys_available)    # keep consistent order
        )
        sys_names = [f"sys__{m}" for m in sys_available]

        # ── Container metrics: mean per (timestamp, service) ─────────────────
        cont_df_raw = df[df["metric_name"].isin(CONTAINER_METRICS)].copy()
        cont_df_raw["service"] = cont_df_raw["kubernetes_pod_name"].map(
            self._pod_to_service
        )
        cont_df_raw = cont_df_raw[cont_df_raw["service"].notna()]

        # Build one column per (short_metric, service)
        cont_df_raw["col"] = (
            cont_df_raw["metric_name"].map(_CONTAINER_SHORT)
            + "__"
            + cont_df_raw["service"]
        )
        cont_pivot = (
            cont_df_raw
            .groupby(["timestamp", "col"])["value"]
            .mean()
            .unstack("col")
        )
        # Ensure all (metric, service) combos exist in a fixed order
        all_cont_cols = [
            f"{_CONTAINER_SHORT[m]}__{svc}"
            for m in CONTAINER_METRICS
            for svc in self.services
        ]
        cont_pivot = cont_pivot.reindex(columns=all_cont_cols, fill_value=np.nan)

        # ── Merge on timestamp, align ─────────────────────────────────────────
        all_df = pd.concat([sys_df, cont_pivot], axis=1).sort_index()
        all_names = sys_names + all_cont_cols

        if not hasattr(self, "_metric_names") or not self._metric_names:
            self._metric_names = all_names

        # ── Aggregate into windows ────────────────────────────────────────────
        ts_arr = all_df.index.values   # datetime64
        vals   = all_df.values.astype(np.float64)
        matrix = np.zeros((W, len(all_names)), dtype=np.float32)

        for i, t_start in enumerate(win_starts):
            t0   = np.datetime64(t_start)
            t1   = np.datetime64(t_start + delta)
            mask = (ts_arr >= t0) & (ts_arr < t1)
            if mask.any():
                with np.errstate(all="ignore"):   # suppress empty-slice warning
                    col_means = np.nanmean(vals[mask], axis=0)
                col_means = np.where(np.isnan(col_means), 0.0, col_means)
                matrix[i] = col_means.astype(np.float32)

        return matrix, all_names

    # ── Step 5: Build per-window trace node features ──────────────────────────

    def _build_trace_node_features(
        self,
        trace_dir: str,
        win_starts: List[datetime],
    ) -> np.ndarray:
        """
        Build [W, N, 5] trace node feature array.
        Features: [call_count, avg_dur_ms, max_dur_ms, error_rate, root_rate]
        """
        W      = len(win_starts)
        N      = self.num_services
        result = np.zeros((W, N, TRACE_NODE_FEAT_DIM), dtype=np.float32)
        delta  = timedelta(seconds=self.window_sec)

        json_path = self._find_trace_json(trace_dir)
        if json_path is None:
            return result

        # Flatten all spans to a list of dicts
        spans = []
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        for trace in data.get("traces", []):
            for span in trace.get("spans", []):
                svc = span.get("service_code", "")
                ts_ms = span.get("start_timestamp_ms")
                if not svc or ts_ms is None:
                    continue
                svc_idx = self.service2idx.get(svc, -1)
                if svc_idx < 0:
                    continue
                spans.append({
                    "svc_idx":     svc_idx,
                    "ts":          datetime.utcfromtimestamp(ts_ms / 1000),
                    "duration_ms": float(span.get("duration_ms") or 0),
                    "is_error":    1 if span.get("is_error", False) else 0,
                    "is_root":     1 if (span.get("parent_span_id", -1) == -1
                                         and span.get("parent_node_id", -1) == -1) else 0,
                })

        if not spans:
            return result

        spans.sort(key=lambda x: x["ts"])
        ts_arr = np.array([s["ts"] for s in spans], dtype="datetime64[us]")

        for i, t_start in enumerate(win_starts):
            t0   = np.datetime64(t_start)
            t1   = np.datetime64(t_start + delta)
            idxs = np.where((ts_arr >= t0) & (ts_arr < t1))[0]
            if idxs.size == 0:
                continue

            win_spans = [spans[j] for j in idxs]
            # Group by service
            svc_spans: Dict[int, List] = {}
            for s in win_spans:
                svc_spans.setdefault(s["svc_idx"], []).append(s)

            for svc_idx, ss in svc_spans.items():
                n    = len(ss)
                durs = np.array([s["duration_ms"] for s in ss], dtype=np.float32)
                result[i, svc_idx, 0] = float(n)
                result[i, svc_idx, 1] = float(durs.mean())
                result[i, svc_idx, 2] = float(durs.max())
                result[i, svc_idx, 3] = sum(s["is_error"] for s in ss) / n
                result[i, svc_idx, 4] = sum(s["is_root"]  for s in ss) / n

        # Normalise: call_count → log1p/10, durations → /1000 (ms→s)
        result[:, :, 0] = np.log1p(result[:, :, 0]) / 10.0
        result[:, :, 1] = result[:, :, 1] / 1000.0
        result[:, :, 2] = result[:, :, 2] / 1000.0

        return result

    # ── Step 6: Build per-window log windows & features ───────────────────────

    def _load_log_windows(
        self,
        log_dir: str,
        win_starts: List[datetime],
    ) -> List[List[str]]:
        """
        Walk all .log files under log_dir (recursive).
        Parse Spring Boot timestamps, assign templates to windows.
        Returns list of length W, each element is a list of template strings.
        """
        W     = len(win_starts)
        delta = timedelta(seconds=self.window_sec)

        records: List[Tuple[datetime, str]] = []

        for fpath in self._iter_log_files(log_dir):
            # Use immediate parent dir name as a "service hint"
            svc_hint = os.path.basename(os.path.dirname(fpath))
            with open(fpath, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.rstrip()
                    if not line:
                        continue
                    m = _LOG_TS_RE.match(line)
                    if not m:
                        continue   # skip stack trace continuations
                    try:
                        ts = datetime.strptime(
                            m.group(1)[:23],   # trim to ms precision
                            "%Y-%m-%d %H:%M:%S.%f"
                        )
                    except ValueError:
                        continue
                    tmpl = self._to_template(line)
                    records.append((ts, f"{svc_hint}|{tmpl}"))

        if not records:
            return [["padding"]] * W

        records.sort(key=lambda x: x[0])
        ts_list   = [r[0] for r in records]
        tmpl_list = [r[1] for r in records]

        win_logs = []
        for i in range(W):
            t_start = win_starts[i]
            t_end   = t_start + delta
            batch   = [
                tmpl_list[j]
                for j, ts in enumerate(ts_list)
                if t_start <= ts < t_end
            ]
            win_logs.append(batch if batch else ["padding"])

        return win_logs

    def _compute_log_features(self, msgs: List[str]) -> np.ndarray:
        """
        6-dim compact log feature vector (same as preprocess_sn.py):
        [error_rate, warn_rate, info_rate, retry_rate,
         service_diversity_norm, log_count_norm]
        """
        N_FEATS = 6
        real = [m for m in msgs if m and m != "padding"]
        if not real:
            return np.zeros(N_FEATS, dtype=np.float32)

        n = len(real)
        error_cnt = warn_cnt = info_cnt = retry_cnt = 0
        services_seen: set = set()

        for m in real:
            lv = _LOG_LVL_RE.search(m)
            if lv:
                lvl = lv.group(1).upper()
                if lvl == "ERROR":
                    error_cnt += 1
                elif lvl in ("WARN", "WARNING"):
                    warn_cnt  += 1
                elif lvl == "INFO":
                    info_cnt  += 1
            if "retry" in m.lower():
                retry_cnt += 1
            parts = m.split("|")
            if parts:
                services_seen.add(parts[0].strip())

        return np.array([
            error_cnt / n,
            warn_cnt  / n,
            info_cnt  / n,
            retry_cnt / n,
            len(services_seen) / max(self.num_services, 1),
            min(np.log1p(n) / 8.0, 1.0),
        ], dtype=np.float32)

    # ── Step 7: Process one scenario ─────────────────────────────────────────

    def _process_scenario(
        self,
        scenario_name: str,
        dirs: Dict[str, str],
        is_anomaly: bool,
    ) -> List[Tuple[str, dict]]:
        """
        Process one scenario → list of (block_id, sample_dict) per window.
        Anomaly scenarios skip the first warmup_minutes.
        """
        logging.info(f"  Processing scenario: {scenario_name}")
        metric_dir = dirs["metric_dir"]
        log_dir    = dirs["log_dir"]
        trace_dir  = dirs["trace_dir"]

        # Determine time range from metric CSV timestamp column
        csv_files = glob.glob(os.path.join(metric_dir, "*.csv"))
        if not csv_files:
            logging.warning(f"    No metric CSV found, skipping")
            return []

        ts_df  = pd.read_csv(csv_files[0], usecols=["timestamp"])
        t_min  = datetime.utcfromtimestamp(ts_df["timestamp"].min())
        t_max  = datetime.utcfromtimestamp(ts_df["timestamp"].max())
        dur_m  = (t_max - t_min).total_seconds() / 60
        logging.info(f"    Time range: {t_min.strftime('%H:%M:%S')} → "
                     f"{t_max.strftime('%H:%M:%S')}  ({dur_m:.1f} min)")

        warmup = timedelta(minutes=self.warmup_minutes) if is_anomaly else timedelta(0)
        win_starts = self._window_starts(t_min + warmup, t_max)
        W = len(win_starts)
        if W == 0:
            logging.warning(f"    No windows after warmup skip, skipping")
            return []
        logging.info(f"    Windows: {W} "
                     f"(warmup={self.warmup_minutes if is_anomaly else 0} min)")

        # Build features
        kpi_matrix, metric_names = self._build_kpi_matrix(metric_dir, win_starts)
        if not self._metric_names and metric_names:
            self._metric_names = metric_names

        node_feats   = self._build_trace_node_features(trace_dir, win_starts)
        win_log_lists = self._load_log_windows(log_dir, win_starts)

        # Assemble samples
        label   = 1 if is_anomaly else 0
        samples = []

        for i in range(W):
            t_start  = win_starts[i]
            block_id = hashlib.md5(
                f"{scenario_name}_{t_start}".encode()
            ).hexdigest()[:12]

            msgs     = win_log_lists[i]
            log_feat = self._compute_log_features(msgs)

            sample = {
                "label":               label,
                "kpi_label":           label,
                "log_label":           label,
                "kpis":                kpi_matrix[i].copy(),
                "logs":                msgs,
                "seqs":                msgs,
                "log_features":        log_feat,
                "trace_node_features": node_feats[i].copy(),   # [N, 5]
                "trace_adj":           self._adj.copy(),        # [N, N]
                "_scenario":           scenario_name,
                "_t_start":            t_start,
            }
            samples.append((block_id, sample))

        logging.info(f"    → {W} windows assembled (label={label})")
        return samples

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self):
        scenario_dirs = self._scenario_dirs()

        # Identify Normal_case and anomaly scenarios
        normal_keys  = [k for k in scenario_dirs if "Normal_case" in k]
        anomaly_keys = sorted([k for k in scenario_dirs if "Normal_case" not in k])

        if not normal_keys:
            raise ValueError("No Normal_case scenario found in data root.")
        normal_key = normal_keys[0]
        logging.info(f"Normal scenario: {normal_key}")
        logging.info(f"Anomaly scenarios ({len(anomaly_keys)}): {anomaly_keys}")

        # Step 1: Discover top-K services
        logging.info("Step 1: Discovering top-K services from Normal_case traces …")
        self.services     = self._discover_services(scenario_dirs[normal_key]["trace_dir"])
        # Sort longest-first for reliable prefix matching in _pod_to_service
        self.services     = sorted(self.services, key=len, reverse=True)
        self.num_services = len(self.services)
        self.service2idx  = {s: i for i, s in enumerate(self.services)}
        logging.info(f"  Using {self.num_services} services")

        # Step 2: Build static adjacency
        logging.info("Step 2: Building static adjacency from Normal_case traces …")
        self._adj = self._build_static_adj(scenario_dirs[normal_key]["trace_dir"])

        # Step 3: Fit Drain3 on Normal_case logs
        logging.info("Step 3: Fitting Drain3 on Normal_case logs …")
        self._fit_drain3(scenario_dirs[normal_key]["log_dir"])

        # Step 4: Process Normal_case
        logging.info("Step 4: Processing Normal_case …")
        normal_samples = self._process_scenario(
            normal_key, scenario_dirs[normal_key], is_anomaly=False
        )

        # Step 5: Process anomaly scenarios
        logging.info("Step 5: Processing anomaly scenarios …")
        os.makedirs(self.output_dir, exist_ok=True)
        scenarios_dir = os.path.join(self.output_dir, "scenarios")
        os.makedirs(scenarios_dir, exist_ok=True)

        anomaly_by_scenario: Dict[str, List] = {}
        for sc in anomaly_keys:
            samples = self._process_scenario(sc, scenario_dirs[sc], is_anomaly=True)
            anomaly_by_scenario[sc] = samples
            self._save_scenario_test(sc, samples, normal_samples, scenarios_dir)

        total_anomaly = sum(len(v) for v in anomaly_by_scenario.values())
        logging.info(f"\nNormal windows  : {len(normal_samples)}")
        logging.info(f"Anomaly windows : {total_anomaly} ({len(anomaly_by_scenario)} scenarios)")

        # Step 6: Save train / unlabel / val / meta
        logging.info("Step 6: Saving train/unlabel/val/meta …")
        self._build_and_save(normal_samples, anomaly_by_scenario)

        logging.info("Preprocessing complete.")

    # ── Save helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _to_dict(samples: List[Tuple[str, dict]]) -> dict:
        return {
            block_id: {k: v for k, v in s.items() if not k.startswith("_")}
            for block_id, s in samples
        }

    def _save_scenario_test(
        self,
        sc_name: str,
        sc_samples: List[Tuple[str, dict]],
        normal_samples: List[Tuple[str, dict]],
        scenarios_dir: str,
    ):
        """
        Save one per-scenario test pkl.
        Anomaly windows are evenly subsampled to max_anomaly_windows.
        Anomaly rate ≈ max_anomaly_windows / (len(normal_samples) + max_anomaly_windows).
        """
        if self.max_anomaly_windows and len(sc_samples) > self.max_anomaly_windows:
            n       = self.max_anomaly_windows
            indices = [int(round(i * (len(sc_samples) - 1) / (n - 1)))
                       for i in range(n)]
            sc_sub  = [sc_samples[i] for i in indices]
        else:
            sc_sub = sc_samples

        combined = normal_samples + sc_sub
        self.rng.shuffle(combined)
        data = self._to_dict(combined)

        n_anom    = len(sc_sub)
        anom_rate = n_anom / len(combined)
        safe_name = re.sub(r"[^A-Za-z0-9_]", "_", sc_name)
        path = os.path.join(scenarios_dir, f"test_{safe_name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logging.info(f"  → scenarios/test_{safe_name}.pkl  "
                     f"({len(combined)} windows, {n_anom} anomaly, "
                     f"rate={anom_rate:.2f})")

    def _build_and_save(
        self,
        normal_samples: List[Tuple[str, dict]],
        anomaly_by_scenario: Dict[str, List],
    ):
        """
        Temporal 80/20 split of Normal_case windows:
          first 80% → train.pkl + unlabel.pkl
          last  20% → val.pkl  (unseen normal, for threshold tuning)
        """
        n_val   = max(1, round(len(normal_samples) * 0.2))
        n_train = len(normal_samples) - n_val

        train_samples = normal_samples[:n_train]
        val_samples   = normal_samples[n_train:]

        train_data = self._to_dict(train_samples)
        val_data   = self._to_dict(val_samples)

        for split, data in (("train", train_data), ("unlabel", train_data)):
            path = os.path.join(self.output_dir, f"{split}.pkl")
            with open(path, "wb") as f:
                pickle.dump(data, f)
            logging.info(f"  Saved {split}.pkl: {len(data)} normal windows")

        val_path = os.path.join(self.output_dir, "val.pkl")
        with open(val_path, "wb") as f:
            pickle.dump(val_data, f)
        logging.info(f"  Saved val.pkl: {len(val_data)} windows (for threshold)")

        # Fault type per scenario (from folder name prefix)
        fault_types = {}
        for sc in anomaly_by_scenario:
            for code, name in (("Lv_P", "Performance"), ("Lv_S", "Service"),
                               ("Lv_D", "Database"),    ("Lv_C", "Code")):
                if sc.startswith(code):
                    fault_types[sc] = name
                    break
            else:
                fault_types[sc] = "Unknown"

        meta = {
            "num_services":    self.num_services,
            "service2idx":     self.service2idx,
            "services":        self.services,
            "metric_names":    self._metric_names,
            "kpi_c":           len(self._metric_names),
            "log_c":           1,
            "trace_c":         TRACE_NODE_FEAT_DIM,
            "window_sec":      self.window_sec,
            "n_log_templates": (len(self._miner.drain.id_to_cluster)
                                if self._miner else 0),
            "scenario_names":  list(anomaly_by_scenario.keys()),
            "fault_types":     fault_types,
        }
        meta_path = os.path.join(self.output_dir, "meta.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)
        logging.info(f"  Saved meta.pkl")
        logging.info(
            f"\n  Evaluate with:\n"
            f"  python codes/common/eval_per_scenario_sn.py"
            f" --data {self.output_dir} --dataset sn --data_type fuse"
            f" --window_size 10"
        )

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _window_starts(self, t_min: datetime, t_max: datetime) -> List[datetime]:
        delta   = timedelta(seconds=self.window_sec)
        current = t_min.replace(microsecond=0,
                                second=(t_min.second // self.window_sec)
                                       * self.window_sec)
        wins = []
        while current < t_max:
            wins.append(current)
            current += delta
        return wins

    @staticmethod
    def _find_trace_json(trace_dir: str) -> Optional[str]:
        """Return the first .json file in trace_dir, or None."""
        files = glob.glob(os.path.join(trace_dir, "*.json"))
        return files[0] if files else None

    @staticmethod
    def _iter_log_files(log_dir: str):
        """Yield paths of all .log files under log_dir recursively."""
        for root, _, files in os.walk(log_dir):
            for fname in sorted(files):
                if fname.endswith(".log"):
                    yield os.path.join(root, fname)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Preprocess TrainTicket (AnoMod TT) → UAC-AD pkl format"
    )
    p.add_argument("--tt_data_root",        default=r"D:\AnoMod\TT_data",
                   help="Root dir containing log_data/, metric_data/, trace_data/")
    p.add_argument("--output_dir",          default=r"D:\UAC-AD\data\tt",
                   help="Output directory for pkl files")
    p.add_argument("--window_sec",          default=15,  type=int,
                   help="Window size in seconds (default: 15s)")
    p.add_argument("--warmup_minutes",      default=5,   type=int,
                   help="Skip first N minutes of anomaly scenarios")
    p.add_argument("--max_anomaly_windows", default=8,   type=int,
                   help="Max anomaly windows per scenario test file (evenly subsampled)")
    p.add_argument("--top_k_services",      default=20,  type=int,
                   help="Top-K services by call count from Normal_case traces")
    p.add_argument("--seed",                default=42,  type=int)
    args = p.parse_args()

    TTPreprocessor(
        tt_data_root        = args.tt_data_root,
        output_dir          = args.output_dir,
        window_sec          = args.window_sec,
        warmup_minutes      = args.warmup_minutes,
        max_anomaly_windows = args.max_anomaly_windows,
        top_k_services      = args.top_k_services,
        seed                = args.seed,
    ).run()


if __name__ == "__main__":
    main()
