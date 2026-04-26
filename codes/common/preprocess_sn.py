"""
Preprocess SocialNetwork (AnoMod) → UAC-AD pkl format.

Dataset layout expected under SN_DATA_ROOT/:
  log_data/    {scenario}_logs_{timestamp}/    *.log files per service
  metric_data/ {scenario}_metrics_{timestamp}/ *.csv (system + container + jaeger)
  trace_data/  {scenario}_traces_{timestamp}/  all_traces.csv

Output (OUTPUT_DIR/):
  train.pkl              — first 80% of Normal_Baseline windows (for training)
  unlabel.pkl            — same as train.pkl
  val.pkl                — last 20% of Normal_Baseline windows (unseen normal,
                           used to compute anomaly threshold without test leakage)
  meta.pkl               — dataset metadata
  scenarios/
    test_{name}.pkl      — per-scenario test file: normal windows (40) +
                           subsampled anomaly windows (max_anomaly_windows, default 6),
                           shuffled. Anomaly rate ~13%.

KPI features (59 total):
  10 system   : cpu_usage, disk_io_time, disk_read_bytes, disk_usage_pct,
                disk_write_bytes, load1, memory_usage_pct, network_errors,
                network_receive_bytes, network_transmit_bytes
  48 container: 12 services x 4 metrics (cpu, memory, net_rx, net_tx)
   1 jaeger   : spans_rate (result="ok")

Trace node features per service (5-dim):
  [call_count, avg_dur_us, max_dur_us, error_rate, root_rate]

Static adjacency [12x12]: built from Normal_Baseline traces (bool call graph).

Usage:
    python codes/common/preprocess_sn.py \\
        --sn_data_root D:/AnoMod/SN_data \\
        --output_dir data/sn \\
        --window_sec 30 --warmup_minutes 5 \\
        --max_anomaly_windows 6 --seed 42
"""

import argparse
import hashlib
import logging
import os
import pickle
import random
import re
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

# 12 canonical services (from Jaeger available_services.json)
SN_SERVICES = [
    "compose-post-service",
    "home-timeline-service",
    "media-service",
    "nginx-web-server",
    "post-storage-service",
    "social-graph-service",
    "text-service",
    "unique-id-service",
    "url-shorten-service",
    "user-mention-service",
    "user-service",
    "user-timeline-service",
]

# Container metric label for each trace service (most are identical; nginx is different)
_CONTAINER_LABEL = {svc: svc for svc in SN_SERVICES}
_CONTAINER_LABEL["nginx-web-server"] = "nginx-thrift"

# 10 system metric files (one scalar per window)
SYSTEM_METRIC_FILES = [
    "system_cpu_usage.csv",
    "system_disk_io_time.csv",
    "system_disk_read_bytes.csv",
    "system_disk_usage_percent.csv",
    "system_disk_write_bytes.csv",
    "system_load1.csv",
    "system_memory_usage_percent.csv",
    "system_network_errors.csv",
    "system_network_receive_bytes.csv",
    "system_network_transmit_bytes.csv",
]

# 4 container metric files (one value per service per window)
CONTAINER_METRIC_FILES = [
    "socialnet_container_cpu.csv",
    "socialnet_container_memory.csv",
    "socialnet_container_network_receive.csv",
    "socialnet_container_network_transmit.csv",
]
CONTAINER_LABEL_COL = "container_label_com_docker_compose_service"

TRACE_NODE_FEAT_DIM = 6  # [call_count, avg_dur_us, max_dur_us, error_rate, root_rate, latency_dev]

# Log timestamp format: [YYYY-Mon-DD HH:MM:SS.ffffff] <LEVEL>: ...
_LOG_TS_RE  = re.compile(
    r'^\[(\d{4}-\w{3}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)\]'
)
_LOG_LVL_RE = re.compile(r'<(info|warn|warning|error|debug|trace)>', re.IGNORECASE)
_MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


def _parse_log_ts(ts_str: str) -> Optional[datetime]:
    """Parse '[YYYY-Mon-DD HH:MM:SS.ffffff]' style timestamp."""
    try:
        date_part, time_part = ts_str.strip().split()
        y, mon, d = date_part.split("-")
        month = _MONTHS.get(mon, 0)
        if month == 0:
            return None
        h, mi, rest = time_part.split(":")
        # rest may be 'SS.ffffff' or just 'SS'
        if "." in rest:
            s, us = rest.split(".")
            us = int(us[:6].ljust(6, "0"))
        else:
            s, us = rest, 0
        return datetime(int(y), month, int(d), int(h), int(mi), int(s), us)
    except Exception:
        return None


# ─── Preprocessor ─────────────────────────────────────────────────────────────

class SNPreprocessor:
    """
    Preprocess SocialNetwork (AnoMod) dataset → UAC-AD pkl format.
    """

    def __init__(
        self,
        sn_data_root: str,
        output_dir: str,
        window_sec: int = 30,
        warmup_minutes: int = 5,
        max_anomaly_windows: int = 6,
        seed: int = 42,
    ):
        self.sn_data_root        = sn_data_root
        self.output_dir          = output_dir
        self.window_sec          = window_sec
        self.warmup_minutes      = warmup_minutes
        self.max_anomaly_windows = max_anomaly_windows
        self.seed                = seed
        self.rng                 = random.Random(seed)

        self.services    = SN_SERVICES
        self.num_services = len(self.services)
        self.service2idx  = {s: i for i, s in enumerate(self.services)}

        self.log_dir    = os.path.join(sn_data_root, "log_data")
        self.metric_dir = os.path.join(sn_data_root, "metric_data")
        self.trace_dir  = os.path.join(sn_data_root, "trace_data")

        self._miner            = None   # Drain3 TemplateMiner fitted on Normal_Baseline
        self._adj              = None   # Static adjacency [12, 12] from Normal_Baseline
        self._latency_baseline = None   # Tuple(mean[N], std[N]) of avg_dur/1e6 from Normal_Baseline

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _scenario_dirs(self) -> Dict[str, Dict[str, str]]:
        """
        Map each scenario name → {log_dir, metric_dir, trace_dir}.
        Scenario name is derived from the folder name up to the first '_metrics' etc.
        Uses the folder name prefix (before _logs/_metrics/_traces).
        """
        def _strip_suffix(name: str, suffix: str) -> str:
            idx = name.find(suffix)
            return name[:idx] if idx != -1 else name

        metrics_folders = {_strip_suffix(d, "_metrics"): d
                           for d in os.listdir(self.metric_dir)
                           if os.path.isdir(os.path.join(self.metric_dir, d))}
        log_folders     = {_strip_suffix(d, "_logs"): d
                           for d in os.listdir(self.log_dir)
                           if os.path.isdir(os.path.join(self.log_dir, d))}
        trace_folders   = {_strip_suffix(d, "_traces"): d
                           for d in os.listdir(self.trace_dir)
                           if os.path.isdir(os.path.join(self.trace_dir, d))}

        common = sorted(set(metrics_folders) & set(log_folders) & set(trace_folders))
        logging.info(f"Found {len(common)} scenarios: {common}")
        return {
            sc: {
                "metric_dir": os.path.join(self.metric_dir, metrics_folders[sc]),
                "log_dir":    os.path.join(self.log_dir,    log_folders[sc]),
                "trace_dir":  os.path.join(self.trace_dir,  trace_folders[sc]),
            }
            for sc in common
        }

    def _window_starts(self, t_min: datetime, t_max: datetime) -> List[datetime]:
        delta   = timedelta(seconds=self.window_sec)
        current = t_min.replace(microsecond=0, second=(t_min.second // self.window_sec) * self.window_sec)
        wins = []
        while current < t_max:
            wins.append(current)
            current += delta
        return wins

    # ── Step 1: Build static adjacency from Normal_Baseline ──────────────────

    def _build_static_adj(self, trace_dir: str) -> np.ndarray:
        """Build boolean adjacency matrix [N, N] from trace spans."""
        adj = np.zeros((self.num_services, self.num_services), dtype=np.float32)
        path = os.path.join(trace_dir, "all_traces.csv")
        if not os.path.exists(path):
            logging.warning(f"  No all_traces.csv in {trace_dir}, returning zero adj")
            return adj

        df = pd.read_csv(path, usecols=["span_id", "parent_span_id", "service"])
        span2svc = dict(zip(df["span_id"], df["service"]))

        for _, row in df.iterrows():
            child_svc  = row["service"]
            parent_sid = row["parent_span_id"]
            if pd.isna(parent_sid):
                continue
            parent_svc = span2svc.get(parent_sid)
            if parent_svc is None or parent_svc == child_svc:
                continue
            ci = self.service2idx.get(child_svc,  -1)
            pi = self.service2idx.get(parent_svc, -1)
            if ci >= 0 and pi >= 0:
                adj[pi, ci] = 1.0  # parent → child
                adj[ci, pi] = 1.0  # undirected

        # Self-loops
        np.fill_diagonal(adj, 1.0)
        n_edges = int(adj.sum()) - self.num_services
        logging.info(f"  Static adj built: {n_edges} edges (symmetric) from {len(df)} spans")
        return adj

    # ── Step 1b: Build per-service latency baseline from Normal_Baseline ────────

    def _build_latency_baseline(self, trace_dir: str) -> None:
        """
        Compute per-service mean and std of avg span duration (in seconds) from
        Normal_Baseline all_traces.csv.  Stored in self._latency_baseline as
        (mean [N], std [N]).  Used by _build_trace_node_features to compute
        latency_dev (col 5) as a z-score for every scenario.
        """
        baseline_mean = np.zeros(self.num_services, dtype=np.float32)
        baseline_std  = np.full(self.num_services, 1e-6, dtype=np.float32)

        path = os.path.join(trace_dir, "all_traces.csv")
        if not os.path.exists(path):
            logging.warning("  No all_traces.csv — latency_dev will be zero for all windows")
            self._latency_baseline = (baseline_mean, baseline_std)
            return

        df = pd.read_csv(path, usecols=["service", "duration_us"])
        df = df.dropna(subset=["service", "duration_us"])
        df["service_idx"] = df["service"].map(self.service2idx)
        df = df[df["service_idx"].notna()].copy()
        df["dur_sec"] = pd.to_numeric(df["duration_us"], errors="coerce").fillna(0) / 1e6

        for i in range(self.num_services):
            vals = df[df["service_idx"] == i]["dur_sec"].values
            if len(vals) > 1:
                baseline_mean[i] = float(vals.mean())
                baseline_std[i]  = float(vals.std()) + 1e-6

        logging.info(
            f"  Latency baseline from {len(df):,} spans — "
            f"mean dur/svc: {baseline_mean.mean()*1000:.2f} ms"
        )
        self._latency_baseline = (baseline_mean, baseline_std)

    # ── Step 2: Fit Drain3 on Normal_Baseline logs ───────────────────────────

    def _fit_drain3(self, log_dir: str) -> None:
        """Read all *.log files in log_dir and fit a Drain3 TemplateMiner."""
        try:
            from drain3 import TemplateMiner
            from drain3.template_miner_config import TemplateMinerConfig
        except ImportError:
            logging.warning("drain3 not installed — falling back to regex templates")
            self._miner = None
            return

        config = TemplateMinerConfig()
        config.drain_depth          = 4
        config.drain_sim_th         = 0.5
        config.drain_max_children   = 100
        config.parametrize_numeric_tokens = True

        miner = TemplateMiner(config=config)
        n_msgs = 0

        for fname in sorted(os.listdir(log_dir)):
            if not fname.endswith(".log"):
                continue
            fpath = os.path.join(log_dir, fname)
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
        """Extract message content (after '] ') from a log line."""
        # Format: [YYYY-Mon-DD HH:MM:SS.ffffff] <level>: ...
        m = re.match(r'^\[[\d\w\-\s:\.]+\]\s*', line)
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
        # Fallback: basic regex normalisation
        return re.sub(r'[0-9a-f]{8,}', '<*>', re.sub(r'\d+', '<NUM>', content))

    # ── Step 3: Per-window KPI features ──────────────────────────────────────

    def _load_system_metrics(self, metric_dir: str) -> pd.DataFrame:
        """
        Load 10 system metric CSVs, aggregate to 15-s samples, return
        DataFrame with DatetimeIndex and 10 columns.
        """
        dfs = {}
        for fname in SYSTEM_METRIC_FILES:
            path = os.path.join(metric_dir, fname)
            col  = fname.replace("system_", "").replace(".csv", "")
            if not os.path.exists(path):
                logging.warning(f"    Missing system metric: {fname}")
                continue
            df = pd.read_csv(path, usecols=["timestamp", "value"],
                             parse_dates=["timestamp"])
            df = df.dropna(subset=["value"])
            df = df.groupby("timestamp")["value"].mean()  # resolve duplicate timestamps
            df.name = col
            dfs[col] = df

        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs.values(), axis=1).sort_index()

    def _load_container_metrics(self, metric_dir: str) -> pd.DataFrame:
        """
        Load 4 container metric CSVs (cpu, memory, net_rx, net_tx),
        filter to 12 SN services, pivot to 48 columns: {metric}_{service}.
        """
        all_frames = []
        metric_names_map = {
            "socialnet_container_cpu.csv":              "container_cpu",
            "socialnet_container_memory.csv":           "container_memory",
            "socialnet_container_network_receive.csv":  "container_net_rx",
            "socialnet_container_network_transmit.csv": "container_net_tx",
        }

        for fname, metric_name in metric_names_map.items():
            path = os.path.join(metric_dir, fname)
            if not os.path.exists(path):
                logging.warning(f"    Missing container metric: {fname}")
                continue
            df = pd.read_csv(path, usecols=["timestamp", "value", CONTAINER_LABEL_COL],
                             parse_dates=["timestamp"])
            df = df.rename(columns={CONTAINER_LABEL_COL: "service"})
            # Map container labels to canonical service names
            inv_map = {v: k for k, v in _CONTAINER_LABEL.items()}
            df["service"] = df["service"].map(lambda s: inv_map.get(s, s))
            # Keep only the 12 canonical services
            df = df[df["service"].isin(self.services)]
            df["col_name"] = metric_name + "__" + df["service"]
            pivot = df.pivot_table(index="timestamp", columns="col_name",
                                   values="value", aggfunc="mean")
            all_frames.append(pivot)

        if not all_frames:
            return pd.DataFrame()
        result = pd.concat(all_frames, axis=1).sort_index()
        return result

    def _load_jaeger_metric(self, metric_dir: str) -> pd.Series:
        """Load jaeger_spans_rate.csv, sum 'ok' result per timestamp."""
        path = os.path.join(metric_dir, "jaeger_spans_rate.csv")
        if not os.path.exists(path):
            logging.warning(f"    Missing jaeger_spans_rate.csv")
            return pd.Series(dtype=float, name="jaeger_spans_ok")
        df = pd.read_csv(path, usecols=["timestamp", "value", "result"],
                         parse_dates=["timestamp"])
        ok = df[df["result"] == "ok"].set_index("timestamp")["value"]
        ok.name = "jaeger_spans_ok"
        return ok

    def _build_kpi_matrix(
        self,
        metric_dir: str,
        win_starts: List[datetime],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build [W, 59] KPI matrix for a scenario's metric directory.
        Returns (matrix, metric_names).
        """
        W = len(win_starts)
        delta = timedelta(seconds=self.window_sec)

        # Load all metric dataframes
        sys_df  = self._load_system_metrics(metric_dir)
        cont_df = self._load_container_metrics(metric_dir)
        jae_sr  = self._load_jaeger_metric(metric_dir)

        # Build a unified time-indexed DataFrame
        parts = []
        if not sys_df.empty:
            parts.append(sys_df)
        if not cont_df.empty:
            parts.append(cont_df)
        if not jae_sr.empty:
            parts.append(jae_sr.to_frame())

        if not parts:
            metric_names = []
            return np.zeros((W, 0), dtype=np.float32), metric_names

        all_metrics = pd.concat(parts, axis=1).sort_index()
        metric_names = list(all_metrics.columns)

        # For each window, compute mean over [t_start, t_start+window_sec)
        matrix = np.zeros((W, len(metric_names)), dtype=np.float32)

        ts_arr = all_metrics.index.values  # datetime64
        vals   = all_metrics.values.astype(np.float32)

        for i, t_start in enumerate(win_starts):
            t_end = t_start + delta
            t0 = np.datetime64(t_start)
            t1 = np.datetime64(t_end)
            mask = (ts_arr >= t0) & (ts_arr < t1)
            if mask.any():
                window_vals = vals[mask]
                # column-wise mean, ignore NaN
                col_means = np.nanmean(window_vals, axis=0)
                # replace remaining NaN with 0
                col_means = np.where(np.isnan(col_means), 0.0, col_means)
                matrix[i] = col_means.astype(np.float32)

        return matrix, metric_names

    # ── Step 4: Per-window trace node features ────────────────────────────────

    def _build_trace_node_features(
        self,
        trace_dir: str,
        win_starts: List[datetime],
    ) -> np.ndarray:
        """
        Build [W, N, 6] trace node features.
        Features per service per window:
          [call_count, avg_dur_us, max_dur_us, error_rate, root_rate, latency_dev]
        latency_dev = z-score of avg_dur vs Normal_Baseline per service.
        """
        W = len(win_starts)
        N = self.num_services
        result = np.zeros((W, N, TRACE_NODE_FEAT_DIM), dtype=np.float32)

        path = os.path.join(trace_dir, "all_traces.csv")
        if not os.path.exists(path):
            return result

        df = pd.read_csv(
            path,
            usecols=["span_id", "parent_span_id", "service",
                     "start_time", "duration_us", "http_status_code"],
            parse_dates=["start_time"],
        )
        df = df.dropna(subset=["service"])
        df["service_idx"] = df["service"].map(self.service2idx)
        df = df[df["service_idx"].notna()].copy()
        df["service_idx"] = df["service_idx"].astype(int)

        # is_error: http_status_code >= 400 or NaN (treat as ok)
        df["is_error"] = df["http_status_code"].apply(
            lambda x: 1 if (pd.notna(x) and float(x) >= 400) else 0
        )
        # is_root: no parent_span_id
        df["is_root"] = df["parent_span_id"].isna().astype(int)

        ts_arr = df["start_time"].values  # datetime64

        delta = timedelta(seconds=self.window_sec)

        for i, t_start in enumerate(win_starts):
            t_end = t_start + delta
            t0 = np.datetime64(t_start)
            t1 = np.datetime64(t_end)
            mask = (ts_arr >= t0) & (ts_arr < t1)
            if not mask.any():
                continue
            win_df = df[mask]

            for svc_idx in range(N):
                sdf = win_df[win_df["service_idx"] == svc_idx]
                if sdf.empty:
                    continue
                n   = len(sdf)
                dur = sdf["duration_us"].fillna(0).values
                result[i, svc_idx, 0] = float(n)                           # call_count
                result[i, svc_idx, 1] = float(dur.mean())                  # avg_dur_us
                result[i, svc_idx, 2] = float(dur.max())                   # max_dur_us
                result[i, svc_idx, 3] = float(sdf["is_error"].sum()) / n   # error_rate
                result[i, svc_idx, 4] = float(sdf["is_root"].sum()) / n    # root_rate

        # Normalise call_count (log1p) and durations (/ 1e6 → seconds)
        result[:, :, 0] = np.log1p(result[:, :, 0]) / 10.0
        result[:, :, 1] = result[:, :, 1] / 1e6
        result[:, :, 2] = result[:, :, 2] / 1e6

        # col 5: latency_dev = z-score of avg_dur vs Normal_Baseline per service
        if self._latency_baseline is not None:
            bl_mean, bl_std = self._latency_baseline
            result[:, :, 5] = (result[:, :, 1] - bl_mean) / bl_std

        return result

    # ── Step 5: Per-window log features ──────────────────────────────────────

    def _load_log_windows(
        self,
        log_dir: str,
        win_starts: List[datetime],
    ) -> List[List[str]]:
        """
        Read all .log files, parse timestamps, assign templates to windows.
        Returns list of length W, each element is a list of template strings.
        """
        W = len(win_starts)
        delta = timedelta(seconds=self.window_sec)

        # Collect (timestamp, template) from all log files
        records: List[Tuple[datetime, str]] = []

        for fname in sorted(os.listdir(log_dir)):
            if not fname.endswith(".log"):
                continue
            fpath = os.path.join(log_dir, fname)
            service_name = fname.replace("_.log", "").replace(".log", "")

            with open(fpath, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    m = _LOG_TS_RE.match(line)
                    if not m:
                        continue
                    ts = _parse_log_ts(m.group(1))
                    if ts is None:
                        continue
                    tmpl = self._to_template(line)
                    records.append((ts, f"{service_name}|{tmpl}"))

        if not records:
            return [["padding"]] * W

        records.sort(key=lambda x: x[0])
        ts_list   = [r[0] for r in records]
        tmpl_list = [r[1] for r in records]

        win_logs = []
        for i in range(W):
            t_start = win_starts[i]
            t_end   = t_start + delta
            batch = [tmpl_list[j] for j, ts in enumerate(ts_list)
                     if t_start <= ts < t_end]
            win_logs.append(batch if batch else ["padding"])

        return win_logs

    # ── Step 6: Compute compact log_features vector ───────────────────────────

    _SVC_ABBREVS = {
        "ComposePostService": "compose-post-service",
        "HomeTimelineService": "home-timeline-service",
        "MediaService": "media-service",
        "NginxThrift": "nginx-web-server",
        "PostStorageService": "post-storage-service",
        "SocialGraphService": "social-graph-service",
        "TextService": "text-service",
        "UniqueIdService": "unique-id-service",
        "UrlShortenService": "url-shorten-service",
        "UserMentionService": "user-mention-service",
        "UserService": "user-service",
        "UserTimelineService": "user-timeline-service",
    }

    def _compute_log_features(self, msgs: List[str]) -> np.ndarray:
        """
        6-dim compact log feature vector:
        [error_rate, warn_rate, info_rate, retry_rate,
         service_diversity_norm, log_count_norm]
        """
        N_FEATS = 6
        real = [m for m in msgs if m and m != "padding"]
        if not real:
            return np.zeros(N_FEATS, dtype=np.float32)

        n = len(real)
        error_cnt = warn_cnt = info_cnt = retry_cnt = 0
        services_seen = set()

        for m in real:
            lv = _LOG_LVL_RE.search(m)
            if lv:
                lvl = lv.group(1).lower()
                if lvl == "error":              error_cnt += 1
                elif lvl in ("warn", "warning"): warn_cnt  += 1
                elif lvl == "info":              info_cnt  += 1
            if "retry" in m.lower():
                retry_cnt += 1
            # Service name comes from the prefix before '|'
            parts = m.split("|")
            if parts:
                svc = parts[0].strip()
                if svc:
                    services_seen.add(svc)

        feat = np.array([
            error_cnt / n,
            warn_cnt  / n,
            info_cnt  / n,
            retry_cnt / n,
            len(services_seen) / self.num_services,
            min(np.log1p(n) / 8.0, 1.0),
        ], dtype=np.float32)
        return feat

    # ── Step 7: Process one scenario → windows ────────────────────────────────

    def _process_scenario(
        self,
        scenario_name: str,
        dirs: Dict[str, str],
        is_anomaly: bool,
    ) -> List[Dict]:
        """
        Process one scenario and return a list of window samples.
        For anomaly scenarios, the first WARMUP_MINUTES are excluded.
        """
        logging.info(f"  Processing scenario: {scenario_name}")
        metric_dir = dirs["metric_dir"]
        log_dir    = dirs["log_dir"]
        trace_dir  = dirs["trace_dir"]

        # Determine time range from system CPU metric (most reliable)
        cpu_path = os.path.join(metric_dir, "system_cpu_usage.csv")
        if not os.path.exists(cpu_path):
            logging.warning(f"    No system_cpu_usage.csv, skipping")
            return []

        cpu_df = pd.read_csv(cpu_path, usecols=["timestamp"], parse_dates=["timestamp"])
        t_min  = cpu_df["timestamp"].min()
        t_max  = cpu_df["timestamp"].max()
        logging.info(f"    Time range: {t_min} → {t_max} "
                     f"({(t_max - t_min).total_seconds() / 60:.1f} min)")

        # Warmup skip for anomaly scenarios
        warmup = timedelta(minutes=self.warmup_minutes) if is_anomaly else timedelta(0)
        win_starts = self._window_starts(t_min + warmup, t_max)
        W = len(win_starts)
        if W == 0:
            logging.warning(f"    No windows after warmup skip, skipping")
            return []
        logging.info(f"    Windows: {W} (after {self.warmup_minutes}-min warmup skip)")

        # KPI matrix
        kpi_matrix, metric_names = self._build_kpi_matrix(metric_dir, win_starts)
        if not hasattr(self, "_metric_names"):
            self._metric_names = metric_names
            logging.info(f"    KPI features: {len(metric_names)}")

        # Trace node features
        node_feats = self._build_trace_node_features(trace_dir, win_starts)

        # Log windows
        win_log_lists = self._load_log_windows(log_dir, win_starts)

        # Assemble samples
        samples = []
        label   = 1 if is_anomaly else 0
        delta   = timedelta(seconds=self.window_sec)

        for i in range(W):
            t_start  = win_starts[i]
            block_id = hashlib.md5(
                f"{scenario_name}_{t_start}".encode()
            ).hexdigest()[:12]

            msgs    = win_log_lists[i]
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
                "_scenario":           scenario_name,           # metadata (not used by model)
                "_t_start":            t_start,
            }
            samples.append((block_id, sample))

        logging.info(f"    → {W} windows assembled (label={label})")
        return samples

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self):
        scenario_dirs = self._scenario_dirs()

        # Identify Normal_Baseline
        normal_keys = [k for k in scenario_dirs if "Normal_Baseline" in k]
        anomaly_keys = [k for k in scenario_dirs if "Normal_Baseline" not in k]
        if not normal_keys:
            raise ValueError("No Normal_Baseline scenario found in data root.")
        normal_key = normal_keys[0]
        logging.info(f"Normal scenario: {normal_key}")
        logging.info(f"Anomaly scenarios ({len(anomaly_keys)}): {anomaly_keys}")

        # Step 1: Build static adjacency from Normal_Baseline
        logging.info("Step 1: Building static adjacency from Normal_Baseline traces …")
        self._adj = self._build_static_adj(scenario_dirs[normal_key]["trace_dir"])

        # Step 1b: Build per-service latency baseline from Normal_Baseline traces
        logging.info("Step 1b: Building latency baseline from Normal_Baseline traces …")
        self._build_latency_baseline(scenario_dirs[normal_key]["trace_dir"])

        # Step 2: Fit Drain3 on Normal_Baseline logs
        logging.info("Step 2: Fitting Drain3 on Normal_Baseline logs …")
        self._fit_drain3(scenario_dirs[normal_key]["log_dir"])

        # Step 3: Process Normal_Baseline
        logging.info("Step 3: Processing Normal_Baseline …")
        normal_samples = self._process_scenario(
            normal_key, scenario_dirs[normal_key], is_anomaly=False
        )

        # Step 4: Process anomaly scenarios + save each test file immediately
        logging.info("Step 4: Processing anomaly scenarios …")
        os.makedirs(self.output_dir, exist_ok=True)
        scenarios_dir = os.path.join(self.output_dir, "scenarios")
        os.makedirs(scenarios_dir, exist_ok=True)

        anomaly_by_scenario: Dict[str, List[Tuple[str, Dict]]] = {}
        for sc in anomaly_keys:
            samples = self._process_scenario(sc, scenario_dirs[sc], is_anomaly=True)
            anomaly_by_scenario[sc] = samples
            # Save scenario test file immediately after processing
            self._save_scenario_test(sc, samples, normal_samples, scenarios_dir)

        anomaly_samples: List[Tuple[str, Dict]] = [
            s for sc_samples in anomaly_by_scenario.values() for s in sc_samples
        ]

        logging.info(f"\nNormal windows  : {len(normal_samples)}")
        logging.info(f"Anomaly windows : {len(anomaly_samples)} "
                     f"({len(anomaly_by_scenario)} scenarios)")

        # Step 5: Save train / unlabel / meta
        logging.info("Step 5: Saving train/unlabel/meta …")
        self._build_and_save(normal_samples, anomaly_by_scenario)

        logging.info("Preprocessing complete.")

    # ── Save ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _to_dict(samples: List[Tuple[str, Dict]]) -> Dict:
        d = {}
        for block_id, s in samples:
            d[block_id] = {k: v for k, v in s.items() if not k.startswith("_")}
        return d

    def _save_scenario_test(
        self,
        sc_name: str,
        sc_samples: List[Tuple[str, Dict]],
        normal_samples: List[Tuple[str, Dict]],
        scenarios_dir: str,
    ):
        """Save one per-scenario test file immediately after processing.

        Anomaly windows are subsampled to max_anomaly_windows (evenly spaced)
        to keep anomaly rate low (~10-15%) and minimise consecutive anomaly
        clusters after shuffle.
        """
        # Subsample anomaly windows evenly across the scenario
        if self.max_anomaly_windows and len(sc_samples) > self.max_anomaly_windows:
            n = self.max_anomaly_windows
            indices = [int(round(i * (len(sc_samples) - 1) / (n - 1))) for i in range(n)]
            sc_samples_sub = [sc_samples[i] for i in indices]
        else:
            sc_samples_sub = sc_samples

        combined = normal_samples + sc_samples_sub
        self.rng.shuffle(combined)
        sc_data = self._to_dict(combined)

        n_anom    = len(sc_samples_sub)
        anom_rate = n_anom / len(combined)
        safe_name = re.sub(r'[^A-Za-z0-9_]', '_', sc_name)
        path = os.path.join(scenarios_dir, f"test_{safe_name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(sc_data, f)
        logging.info(f"  -> Saved scenarios/test_{safe_name}.pkl  "
                     f"({len(combined)} windows, {n_anom} anomaly, rate={anom_rate:.2f})")

    def _build_and_save(
        self,
        normal_samples: List[Tuple[str, Dict]],
        anomaly_by_scenario: Dict[str, List[Tuple[str, Dict]]],
    ):
        """Save train.pkl, unlabel.pkl, val.pkl, meta.pkl.

        Temporal split: first 80% → train/unlabel, last 20% → val (unseen normal).
        Val is used to compute anomaly threshold without test data leakage.
        """
        n_val   = max(1, round(len(normal_samples) * 0.2))
        n_train = len(normal_samples) - n_val
        train_samples = normal_samples[:n_train]   # first 80% (temporal order)
        val_samples   = normal_samples[n_train:]   # last  20%

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
        logging.info(f"  Saved val.pkl: {len(val_data)} normal windows (unseen, for threshold)")

        metric_names = getattr(self, "_metric_names", [])
        meta = {
            "num_services":    self.num_services,
            "service2idx":     self.service2idx,
            "metric_names":    metric_names,
            "kpi_c":           len(metric_names),
            "log_c":           1,
            "trace_c":         TRACE_NODE_FEAT_DIM,
            "window_sec":      self.window_sec,
            "n_log_templates": 1,
            "scenario_names":  list(anomaly_by_scenario.keys()),
        }
        meta_path = os.path.join(self.output_dir, "meta.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)
        logging.info(f"  Saved meta.pkl → {self.output_dir}")
        logging.info(
            f"\n  Evaluate with:\n"
            f"  python codes/common/eval_per_scenario_sn.py"
            f" --data {self.output_dir} --dataset sn --data_type fuse"
        )


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Preprocess SocialNetwork (AnoMod) → UAC-AD pkl format"
    )
    p.add_argument(
        "--sn_data_root",
        default=r"D:\AnoMod\SN_data",
        help="Root directory containing log_data/, metric_data/, trace_data/",
    )
    p.add_argument(
        "--output_dir",
        default=r"C:\Users\us\Desktop\UAC-AD\.claude\worktrees\affectionate-bell\data\sn",
        help="Output directory for pkl files",
    )
    p.add_argument("--window_sec",           default=30,  type=int,
                   help="Window size in seconds (default: 30s; metrics sampled at 15s)")
    p.add_argument("--warmup_minutes",       default=5,   type=int)
    p.add_argument("--max_anomaly_windows",  default=6,   type=int,
                   help="Max anomaly windows per scenario test file (evenly subsampled). "
                        "Keeps anomaly rate ~13%% to reduce consecutive clusters after shuffle.")
    p.add_argument("--seed",                 default=42,  type=int)
    args = p.parse_args()

    SNPreprocessor(
        sn_data_root        = args.sn_data_root,
        output_dir          = args.output_dir,
        window_sec          = args.window_sec,
        warmup_minutes      = args.warmup_minutes,
        max_anomaly_windows = args.max_anomaly_windows,
        seed                = args.seed,
    ).run()


if __name__ == "__main__":
    main()
