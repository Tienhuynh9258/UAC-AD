"""
Preprocess the MicroSS dataset (GAIA-DataSet) for use with UAC-AD + Trace branch.

Expected input directory structure (after extracting the split archives):
  <micross_root>/
    trace/        -- trace CSV files
    metric/       -- metric CSV files  (one file per metric per node)
    business/     -- business log CSV files
    run/          -- anomaly injection records (optional)

CSV column schemas (from MicroSS system description):
  trace:    timestamp(YYYY-MM-DD HH:MM:SS), host_ip, service_name, trace_id,
            span_id, parent_id, start_time(ms), end_time(ms), url,
            status_code, message
  metric:   timestamp(13-digit Unix ms), value
  business: datetime(YYYY-MM-DD HH:MM:SS), service, message

Output (saved to --output_dir):
  train.pkl   -- normal-only samples from the first train_ratio of the time range
  unlabel.pkl -- same as train.pkl (for unsupervised training)
  test.pkl    -- all samples from the remaining period (normal + anomalous)
  meta.pkl    -- metadata: num_services, service2idx, kpi_c, trace_c, ...

Usage:
  python preprocess_micross.py \\
      --trace_dir   C:\Users\us\Desktop\UAC-AD\MicroSS\trace \\
      --metric_dir  C:\Users\us\Desktop\UAC-AD\MicroSS\metric \\
      --log_dir     C:\Users\us\Desktop\UAC-AD\MicroSS\business \\
      --run_dir     C:\Users\us\Desktop\UAC-AD\MicroSS\run \\
      --output_dir  ../../data/micross \\
      --window_sec  60 \\
      --max_services 20 \\
      --max_metrics  50

Then run the model with:
  python run.py --data ../../data/micross --dataset micross --data_type fuse \\
                --open_trace true --num_services 20 --trace_c 5
"""

import os
import pickle
import hashlib
import logging
import argparse
from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _read_csvs(directory, required_cols=None):
    """Read all CSV files in a directory and concatenate them."""
    dfs = []
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".csv"):
            continue
        path = os.path.join(directory, fname)
        try:
            df = pd.read_csv(path)
            if required_cols:
                missing = [c for c in required_cols if c not in df.columns]
                if missing:
                    logging.warning(f"  {fname}: missing columns {missing}, skipping")
                    continue
            df["_source_file"] = fname
            dfs.append(df)
        except Exception as e:
            logging.warning(f"  Could not read {fname}: {e}")
    if not dfs:
        raise FileNotFoundError(f"No valid CSV files found in {directory}")
    return pd.concat(dfs, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main preprocessor class
# ─────────────────────────────────────────────────────────────────────────────

class MicroSSPreprocessor:
    # Number of node-level features stored per service in the STG
    TRACE_NODE_FEAT_DIM = 5  # [call_count, avg_dur, max_dur, error_rate, root_rate]

    def __init__(
        self,
        trace_dir,
        metric_dir,
        log_dir,
        run_dir,
        output_dir,
        window_sec=60,
        train_ratio=0.7,
        max_services=20,
        max_metrics=50,
    ):
        self.trace_dir = trace_dir
        self.metric_dir = metric_dir
        self.log_dir = log_dir
        self.run_dir = run_dir
        self.output_dir = output_dir
        self.window_sec = window_sec
        self.train_ratio = train_ratio
        self.max_services = max_services
        self.max_metrics = max_metrics

        # Populated during run()
        self.service2idx: dict = {}
        self.num_services: int = 0
        self.metric_names: list = []
        self.anomaly_periods: list = []  # list of (pd.Timestamp, pd.Timestamp)

    # ── Data loaders ──────────────────────────────────────────────────────────

    def _load_trace(self) -> pd.DataFrame:
        logging.info("Loading trace data …")
        trace = _read_csvs(
            self.trace_dir,
            required_cols=["timestamp", "service_name", "span_id", "status_code"],
        )
        trace["timestamp"] = pd.to_datetime(trace["timestamp"], errors="coerce")
        trace["start_time"] = pd.to_numeric(trace.get("start_time", np.nan), errors="coerce")
        trace["end_time"]   = pd.to_numeric(trace.get("end_time",   np.nan), errors="coerce")
        trace["duration_ms"] = trace["end_time"] - trace["start_time"]
        trace["is_error"] = (trace["status_code"].astype(str) != "200").astype(int)
        trace = trace.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        logging.info(f"  {len(trace):,} spans | {trace['service_name'].nunique()} services")
        return trace

    def _build_service_index(self, trace: pd.DataFrame):
        """Fix a service→index mapping using the top-K most frequent services."""
        top = (
            trace["service_name"]
            .value_counts()
            .head(self.max_services)
            .index.tolist()
        )
        services = sorted(top)
        self.service2idx = {s: i for i, s in enumerate(services)}
        self.num_services = len(services)
        logging.info(f"  STG nodes ({self.num_services}): {services}")

    def _load_metrics(self) -> dict:
        """Return {metric_key: DataFrame(timestamp, value)} for all metric files."""
        logging.info("Loading metric data …")
        metrics = {}
        for fname in sorted(os.listdir(self.metric_dir)):
            if not fname.endswith(".csv"):
                continue
            path = os.path.join(self.metric_dir, fname)
            try:
                df = pd.read_csv(path, names=["timestamp", "value"])
                # 13-digit Unix ms → datetime
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
                df = df.dropna().sort_values("timestamp")
                key = os.path.splitext(fname)[0]
                metrics[key] = df
            except Exception as e:
                logging.warning(f"  Could not read metric {fname}: {e}")

        # Keep the max_metrics most data-rich files
        if len(metrics) > self.max_metrics:
            by_size = sorted(metrics.items(), key=lambda x: len(x[1]), reverse=True)
            metrics = dict(by_size[: self.max_metrics])

        self.metric_names = sorted(metrics.keys())
        logging.info(f"  {len(self.metric_names)} metric series kept")
        return metrics

    def _load_logs(self) -> pd.DataFrame:
        """Return DataFrame(datetime, service, message) for all business log files."""
        logging.info("Loading business log data …")
        try:
            logs = _read_csvs(self.log_dir, required_cols=["datetime", "message"])
        except FileNotFoundError:
            logging.warning("  No log CSV files found — log branch will use 'padding'")
            return pd.DataFrame(columns=["datetime", "service", "message"])
        logs["datetime"] = pd.to_datetime(logs["datetime"], errors="coerce")
        logs = logs.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
        logging.info(f"  {len(logs):,} log entries")
        return logs

    def _load_anomaly_periods(self):
        """Parse anomaly injection records from the run directory (best-effort)."""
        self.anomaly_periods = []
        if not self.run_dir or not os.path.isdir(self.run_dir):
            return
        for fname in sorted(os.listdir(self.run_dir)):
            if not fname.endswith(".csv"):
                continue
            try:
                df = pd.read_csv(os.path.join(self.run_dir, fname))
                # Heuristic: look for columns that contain 'start' and 'end'/'duration'
                start_col = next((c for c in df.columns if "start" in c.lower()), None)
                end_col   = next((c for c in df.columns if "end"   in c.lower()), None)
                if start_col and end_col:
                    for _, row in df.iterrows():
                        s = pd.to_datetime(row[start_col], errors="coerce")
                        e = pd.to_datetime(row[end_col],   errors="coerce")
                        if pd.notna(s) and pd.notna(e) and s < e:
                            self.anomaly_periods.append((s, e))
            except Exception as ex:
                logging.warning(f"  Could not parse anomaly file {fname}: {ex}")
        logging.info(f"  {len(self.anomaly_periods)} anomaly injection periods loaded")

    # ── Per-window feature builders ───────────────────────────────────────────

    def _build_stg(self, spans: pd.DataFrame):
        """
        Build Service Trace Graph for one time window.

        Node features (TRACE_NODE_FEAT_DIM = 5 per service):
          0: call_count   — normalised number of spans
          1: avg_dur      — normalised mean(end_time - start_time) [ms]
          2: max_dur      — normalised max(end_time - start_time) [ms]
          3: error_rate   — fraction of spans with status_code ≠ 200
          4: root_rate    — fraction of spans that are root spans (no parent)

        Adjacency:
          adj[parent_idx, child_idx] = 1  if the parent service called the child service
        """
        N = self.num_services
        feats = np.zeros((N, self.TRACE_NODE_FEAT_DIM), dtype=np.float32)
        adj   = np.zeros((N, N), dtype=np.float32)

        if len(spans) == 0:
            return feats, adj

        # span_id → service_name for parent resolution
        span2svc = dict(zip(spans["span_id"].astype(str),
                            spans["service_name"].astype(str)))

        for svc, grp in spans.groupby("service_name"):
            if svc not in self.service2idx:
                continue
            i = self.service2idx[svc]
            dur = grp["duration_ms"].dropna()
            feats[i, 0] = len(grp)                                 # call_count (raw)
            feats[i, 1] = float(dur.mean()) if len(dur) > 0 else 0 # avg_dur
            feats[i, 2] = float(dur.max())  if len(dur) > 0 else 0 # max_dur
            feats[i, 3] = float(grp["is_error"].mean())             # error_rate
            # root_rate: parent_id is NaN or empty string
            parent_is_null = grp["parent_id"].isna() | (grp["parent_id"].astype(str) == "")
            feats[i, 4] = float(parent_is_null.mean())             # root_rate

            # Build edges
            for _, span in grp.iterrows():
                pid = str(span.get("parent_id", ""))
                if pid and pid != "nan" and pid in span2svc:
                    parent_svc = span2svc[pid]
                    if parent_svc in self.service2idx:
                        j = self.service2idx[parent_svc]
                        adj[j, i] = 1.0  # j → i  (parent called child)

        # Normalise call_count, avg_dur, max_dur to [0, 1]
        for col in [0, 1, 2]:
            mx = feats[:, col].max()
            if mx > 0:
                feats[:, col] /= mx

        return feats, adj

    def _kpi_features(self, metrics: dict, t_start, t_end) -> np.ndarray:
        """Mean metric value in [t_start, t_end) for each metric series."""
        feats = []
        for name in self.metric_names:
            df = metrics[name]
            mask = (df["timestamp"] >= t_start) & (df["timestamp"] < t_end)
            vals = df.loc[mask, "value"].dropna()
            feats.append(float(vals.mean()) if len(vals) > 0 else 0.0)
        return np.array(feats, dtype=np.float32)

    def _log_messages(self, logs: pd.DataFrame, t_start, t_end) -> list:
        """Return list of message strings in [t_start, t_end)."""
        if logs.empty:
            return ["padding"]
        mask = (logs["datetime"] >= t_start) & (logs["datetime"] < t_end)
        msgs = logs.loc[mask, "message"].fillna("padding").tolist()
        return msgs if msgs else ["padding"]

    def _is_anomalous(self, spans: pd.DataFrame, t_start, t_end) -> int:
        """1 if any span has status_code ≠ 200, or if the window overlaps an
        anomaly injection period; 0 otherwise."""
        if len(spans) > 0 and spans["is_error"].any():
            return 1
        for (s, e) in self.anomaly_periods:
            if not (t_end <= s or t_start >= e):
                return 1
        return 0

    # ── Window builder ────────────────────────────────────────────────────────

    def _build_windows(self, trace, metrics, logs):
        """Segment all data into fixed-length windows and build per-window samples."""
        t_min = trace["timestamp"].min().floor("T")  # round down to minute
        t_max = trace["timestamp"].max()
        delta = timedelta(seconds=self.window_sec)

        windows = []
        current = t_min
        total = int((t_max - t_min).total_seconds() / self.window_sec) + 1
        logging.info(f"Building ~{total} windows of {self.window_sec}s each …")

        while current < t_max:
            win_end = current + delta

            mask = (trace["timestamp"] >= current) & (trace["timestamp"] < win_end)
            win_spans = trace[mask]

            node_feats, adj  = self._build_stg(win_spans)
            kpis              = self._kpi_features(metrics, current, win_end)
            log_msgs          = self._log_messages(logs, current, win_end)
            label             = self._is_anomalous(win_spans, current, win_end)

            # Stable unique ID for this window
            block_id = hashlib.md5(str(current).encode()).hexdigest()[:10]

            sample = {
                "label":     label,
                "kpi_label": label,
                "log_label": label,
                "kpis":      kpis,      # np.array [num_metrics]
                "logs":      log_msgs,  # list[str]  — used by FeatureExtractor
                "seqs":      log_msgs,  # duplicate for compatibility
                "trace_node_features": node_feats,  # np.array [num_services, 5]
                "trace_adj":           adj,         # np.array [num_services, num_services]
            }
            windows.append((current, block_id, sample))
            current = win_end

        n_anom = sum(s["label"] for _, _, s in windows)
        logging.info(f"  {len(windows)} windows built | "
                     f"anomaly rate: {n_anom}/{len(windows)} = {n_anom/max(1,len(windows)):.3f}")
        return windows

    # ── Split & save ──────────────────────────────────────────────────────────

    def _split_and_save(self, windows):
        os.makedirs(self.output_dir, exist_ok=True)
        n = len(windows)
        split_idx = int(n * self.train_ratio)

        train_data, test_data = {}, {}
        for i, (ts, block_id, sample) in enumerate(windows):
            if i < split_idx:
                # Only normal samples in train/unlabel (unsupervised setting)
                if sample["label"] == 0:
                    train_data[block_id] = sample
            else:
                test_data[block_id] = sample

        n_test_anom = sum(s["label"] for s in test_data.values())
        logging.info(f"Train/unlabel: {len(train_data)} samples (all normal)")
        logging.info(f"Test:          {len(test_data)} samples | "
                     f"anomaly rate: {n_test_anom}/{len(test_data)} = "
                     f"{n_test_anom/max(1,len(test_data)):.3f}")

        for split, data in [("train", train_data), ("unlabel", train_data), ("test", test_data)]:
            path = os.path.join(self.output_dir, f"{split}.pkl")
            with open(path, "wb") as f:
                pickle.dump(data, f)
            logging.info(f"  Saved {path}")

        # Save metadata so run.py knows the right --num_services / --trace_c values
        meta = {
            "num_services":  self.num_services,
            "service2idx":   self.service2idx,
            "metric_names":  self.metric_names,
            "kpi_c":         len(self.metric_names),
            "trace_c":       self.TRACE_NODE_FEAT_DIM,
            "window_sec":    self.window_sec,
        }
        meta_path = os.path.join(self.output_dir, "meta.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)
        logging.info(f"  Metadata saved to {meta_path}")
        logging.info(f"  ── Run model with ──────────────────────────────────────")
        logging.info(f"  python run.py --data {self.output_dir} --dataset micross")
        logging.info(f"    --data_type fuse --open_trace true")
        logging.info(f"    --num_services {self.num_services}")
        logging.info(f"    --trace_c {self.TRACE_NODE_FEAT_DIM}")
        logging.info(f"  ────────────────────────────────────────────────────────")

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self):
        trace   = self._load_trace()
        self._build_service_index(trace)
        metrics = self._load_metrics()
        logs    = self._load_logs()
        self._load_anomaly_periods()
        windows = self._build_windows(trace, metrics, logs)
        self._split_and_save(windows)
        logging.info("Preprocessing complete.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MicroSS (GAIA-DataSet) → UAC-AD pkl format"
    )
    parser.add_argument("--trace_dir",   required=True,
                        help="Directory containing extracted trace CSV files")
    parser.add_argument("--metric_dir",  required=True,
                        help="Directory containing extracted metric CSV files")
    parser.add_argument("--log_dir",     required=True,
                        help="Directory containing extracted business/log CSV files")
    parser.add_argument("--run_dir",     default="",
                        help="Directory containing anomaly injection records (optional)")
    parser.add_argument("--output_dir",  default="../../data/micross",
                        help="Output directory for pkl files (default: ../../data/micross)")
    parser.add_argument("--window_sec",  default=60, type=int,
                        help="Time window size in seconds (default: 60)")
    parser.add_argument("--train_ratio", default=0.7, type=float,
                        help="Fraction of time used as training period (default: 0.7)")
    parser.add_argument("--max_services", default=20, type=int,
                        help="Max services to include in STG (top-K by frequency)")
    parser.add_argument("--max_metrics",  default=50, type=int,
                        help="Max metric series to include (top-K by data coverage)")
    args = parser.parse_args()

    preprocessor = MicroSSPreprocessor(
        trace_dir    = args.trace_dir,
        metric_dir   = args.metric_dir,
        log_dir      = args.log_dir,
        run_dir      = args.run_dir,
        output_dir   = args.output_dir,
        window_sec   = args.window_sec,
        train_ratio  = args.train_ratio,
        max_services = args.max_services,
        max_metrics  = args.max_metrics,
    )
    preprocessor.run()


if __name__ == "__main__":
    main()
