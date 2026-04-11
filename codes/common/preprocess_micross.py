r"""
Preprocess the MicroSS dataset (GAIA-DataSet) for use with UAC-AD + Trace branch.

Expected input directory structure (after extracting the split archives):
  <micross_root>/
    trace/trace/        -- trace CSV files  (nested subdir after extraction)
    metric/metric/      -- metric CSV files (one file per metric per date range)
    business/business/  -- business log CSV files
    run/                -- contains run.zip with anomaly injection records

CSV column schemas:
  trace:    timestamp, host_ip, service_name, trace_id, span_id, parent_id,
            start_time(datetime str), end_time(datetime str), url, status_code, message
  metric:   timestamp(13-digit Unix ms), value
  business: datetime(YYYY-MM-DD HH:MM:SS), service, message
  run:      datetime, service, message  (anomaly info embedded in message text)

Usage:
  python preprocess_micross.py \
      --trace_dir   D:\GAIA-DataSet\MicroSS\trace\trace \
      --metric_dir  D:\GAIA-DataSet\MicroSS\metric\metric \
      --log_dir     D:\GAIA-DataSet\MicroSS\business\business \
      --run_dir     D:\GAIA-DataSet\MicroSS\run \
      --output_dir  ../../data/micross \
      --window_sec  60 \
      --max_services 4 \
      --max_metrics  85

Paper spec (Table I, Dataset B / MicroSS):
  Log Messages : 5,308,847
  Metric Length: 42,813   (≈ 30 days × 1,440 windows/day at 60s window)
  Anomaly Ratio: 5.63%
  Services     : 4
  Metrics      : 85  (sampled every 30s, averaged per 60s window)
  Log templates: 20  (after Drain3 parsing)
"""

import os
import re
import zipfile
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

CHUNK_SIZE  = 500_000   # rows per chunk when streaming large trace CSVs
ADJ_SAMPLE  = 50_000    # rows per file for static adjacency topology
LOG_CHUNK   = 300_000   # rows per chunk for business log CSVs (C engine)

# Regex: extract "start at YYYY-MM-DD HH:MM:SS... and lasts NNN seconds" from run messages
_ANOMALY_RE = re.compile(
    r'start at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[\d.]*) and lasts (\d+) seconds'
)
# Regex: strip date-range suffix "_YYYY-MM-DD_YYYY-MM-DD" from metric filenames
_DATE_SUFFIX_RE = re.compile(r'_\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}$')


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _csv_files(directory: str):
    """Return all .csv files under *directory* (recursive)."""
    results = []
    for root, _dirs, files in os.walk(directory):
        for f in files:
            if f.endswith(".csv"):
                results.append(os.path.join(root, f))
    return sorted(results)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessor
# ─────────────────────────────────────────────────────────────────────────────

class MicroSSPreprocessor:
    TRACE_NODE_FEAT_DIM = 5  # [call_count, avg_dur, max_dur, error_rate, root_rate]

    def __init__(self, trace_dir, metric_dir, log_dir, run_dir, output_dir,
                 window_sec=60, train_ratio=0.7, max_services=4, max_metrics=85,
                 services=None):
        self.trace_dir    = trace_dir
        self.metric_dir   = metric_dir
        self.log_dir      = log_dir
        self.run_dir      = run_dir
        self.output_dir   = output_dir
        self.window_sec   = window_sec
        self.train_ratio  = train_ratio
        self.max_services = max_services
        self.max_metrics  = max_metrics
        # Optional explicit service list (e.g. ["webservice1","dbservice1",...])
        # If None, auto-selects max_services with type-diversity heuristic.
        self.services     = services

        self.service2idx: dict   = {}
        self.num_services: int   = 0
        self.metric_names: list  = []
        self.anomaly_periods: list = []
        self._t_min = None  # populated by run() for in-range filtering
        self._t_max = None

    # ── Step 1: scan time range ───────────────────────────────────────────────

    def _scan_time_range(self):
        """Read only 'timestamp' column per file to find global t_min / t_max."""
        logging.info("Scanning time range from trace files …")
        t_min = t_max = None
        for path in _csv_files(self.trace_dir):
            for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE,
                                     usecols=["timestamp"], low_memory=False):
                ts = pd.to_datetime(chunk["timestamp"], errors="coerce").dropna()
                if ts.empty:
                    continue
                cmin, cmax = ts.min(), ts.max()
                t_min = cmin if t_min is None else min(t_min, cmin)
                t_max = cmax if t_max is None else max(t_max, cmax)
            logging.info(f"  {os.path.basename(path)}: scanned")
        if t_min is None:
            raise RuntimeError("No valid timestamps found in trace directory.")
        t_min = t_min.floor("min")
        logging.info(f"  Time range: {t_min} → {t_max}")
        return t_min, t_max

    # ── Step 2: service index from sample ────────────────────────────────────

    _SVC_DIGIT_RE = re.compile(r'\d+$')   # strip trailing digits to get service type

    def _build_service_index_from_sample(self):
        logging.info("Building service index …")

        # ── Case 1: explicit list provided (e.g. via --services) ──────────────
        if self.services:
            services = sorted(self.services)
            self.service2idx  = {s: i for i, s in enumerate(services)}
            self.num_services = len(services)
            logging.info(f"  Using explicit service list ({self.num_services}): {services}")
            return

        # ── Case 2: auto-select with type-diversity heuristic ─────────────────
        # Count calls per service from a small sample of each trace file.
        counts: dict = defaultdict(int)
        for path in _csv_files(self.trace_dir):
            try:
                s = pd.read_csv(path, nrows=10_000,
                                usecols=["service_name"], low_memory=False)
                for svc, c in s["service_name"].value_counts().items():
                    counts[str(svc)] += int(c)
            except Exception as e:
                logging.warning(f"  {os.path.basename(path)}: {e}")

        # Group by service type (strip trailing digits: "webservice1" → "webservice")
        type_groups: dict = defaultdict(list)
        for svc, cnt in counts.items():
            stype = self._SVC_DIGIT_RE.sub("", svc)
            type_groups[stype].append((svc, cnt))

        # From each service type, pick the instance with the highest call count.
        # Then rank types by their representative's call count and take top max_services.
        type_reps = []
        for stype, svcs in type_groups.items():
            best_svc, best_cnt = max(svcs, key=lambda x: x[1])
            type_reps.append((stype, best_svc, best_cnt))
        type_reps.sort(key=lambda x: -x[2])

        selected = sorted(rep for _, rep, _ in type_reps[: self.max_services])
        self.service2idx  = {s: i for i, s in enumerate(selected)}
        self.num_services = len(selected)
        logging.info(f"  {self.num_services} services (type-diverse): {selected}")

    # ── Step 3: static adjacency ──────────────────────────────────────────────

    def _build_static_adjacency(self) -> np.ndarray:
        """Sample ADJ_SAMPLE rows per file to build the fixed call-graph topology."""
        logging.info("Building static adjacency …")
        adj = np.zeros((self.num_services, self.num_services), dtype=np.float32)

        # Pass 1: span_id → service_name registry
        span_svc: dict = {}
        for path in _csv_files(self.trace_dir):
            try:
                s = pd.read_csv(path, nrows=ADJ_SAMPLE,
                                usecols=lambda c: c in ["service_name", "span_id"],
                                low_memory=False)
                s = s[s["service_name"].isin(self.service2idx)]
                for sid, svc in zip(s["span_id"].astype(str),
                                    s["service_name"].astype(str)):
                    span_svc[sid] = svc
            except Exception as e:
                logging.warning(f"  adj pass1 {os.path.basename(path)}: {e}")

        # Pass 2: parent_id lookup → build edge
        for path in _csv_files(self.trace_dir):
            try:
                s = pd.read_csv(path, nrows=ADJ_SAMPLE,
                                usecols=lambda c: c in ["service_name", "span_id",
                                                        "parent_id"],
                                low_memory=False)
                s = s[s["service_name"].isin(self.service2idx)]
                non_root = s[s["parent_id"].astype(str) != "0"]
                for _, row in non_root.iterrows():
                    child  = str(row["service_name"])
                    parent = span_svc.get(str(row["parent_id"]))
                    if parent and parent in self.service2idx and parent != child:
                        adj[self.service2idx[parent], self.service2idx[child]] = 1.0
            except Exception as e:
                logging.warning(f"  adj pass2 {os.path.basename(path)}: {e}")

        logging.info(f"  {int(adj.sum())} directed edges")
        return adj

    # ── Step 4: anomaly periods from run table ───────────────────────────────

    def _load_anomaly_periods(self):
        """
        Parse run_table CSVs (inside run.zip) to extract anomaly injection windows.
        Each row's 'message' column may contain:
          "... start at YYYY-MM-DD HH:MM:SS.ffffff and lasts NNN seconds ..."
        """
        self.anomaly_periods = []
        if not self.run_dir:
            return

        sources = []  # list of file-like objects
        if os.path.isdir(self.run_dir):
            for fname in os.listdir(self.run_dir):
                fpath = os.path.join(self.run_dir, fname)
                if fname.endswith(".zip"):
                    try:
                        zf = zipfile.ZipFile(fpath)
                        for zname in zf.namelist():
                            if zname.endswith(".csv"):
                                sources.append((zname, zf.open(zname)))
                    except Exception as e:
                        logging.warning(f"  Cannot open {fname}: {e}")
                elif fname.endswith(".csv"):
                    sources.append((fname, open(fpath, "rb")))

        for name, fobj in sources:
            try:
                df = pd.read_csv(fobj, on_bad_lines="skip", low_memory=False)
                msg_col = next((c for c in df.columns
                                if "message" in c.lower()), None)
                if msg_col is None:
                    logging.warning(f"  {name}: no 'message' column found, "
                                    f"columns={df.columns.tolist()}")
                    continue
                for msg in df[msg_col].dropna().astype(str):
                    m = _ANOMALY_RE.search(msg)
                    if m:
                        try:
                            start = pd.Timestamp(m.group(1))
                            end   = start + pd.Timedelta(seconds=int(m.group(2)))
                            self.anomaly_periods.append((start, end))
                        except Exception:
                            pass
            except Exception as e:
                logging.warning(f"  {name}: {e}")

        logging.info(f"  {len(self.anomaly_periods)} anomaly injection periods loaded")
        if self.anomaly_periods:
            logging.info(f"  First: {self.anomaly_periods[0]}")

    # ── Step 5: trace streaming ───────────────────────────────────────────────

    def _stream_trace_to_windows(self, win_starts_ns: np.ndarray) -> np.ndarray:
        """
        Stream trace CSVs in chunks. Never holds more than CHUNK_SIZE spans in RAM.
        Returns node_feats [W, S, 5].
        """
        W, S = len(win_starts_ns), self.num_services
        logging.info(f"Streaming trace → {W} windows × {S} services …")

        acc_count   = np.zeros((W, S), dtype=np.float64)
        acc_dur_sum = np.zeros((W, S), dtype=np.float64)
        acc_max_dur = np.zeros((W, S), dtype=np.float64)
        acc_errors  = np.zeros((W, S), dtype=np.float64)
        acc_roots   = np.zeros((W, S), dtype=np.float64)

        _need = {"timestamp", "service_name", "start_time", "end_time",
                 "status_code", "parent_id"}

        for path in _csv_files(self.trace_dir):
            fname = os.path.basename(path)
            n_chunks = 0
            for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE,
                                     usecols=lambda c: c in _need,
                                     low_memory=False):
                chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce")
                chunk = chunk.dropna(subset=["timestamp"])
                chunk = chunk[chunk["service_name"].isin(self.service2idx)]
                if chunk.empty:
                    continue

                # Duration: start_time / end_time are datetime strings
                dur = (
                    pd.to_datetime(chunk["end_time"],   errors="coerce") -
                    pd.to_datetime(chunk["start_time"], errors="coerce")
                ).dt.total_seconds().mul(1000).clip(lower=0).fillna(0).values

                is_error = (chunk["status_code"].astype(str) != "200").values.astype(np.float64)
                is_root  = (chunk["parent_id"].astype(str) == "0").values.astype(np.float64)
                si = chunk["service_name"].map(self.service2idx).values.astype(np.int64)

                ts_ns = chunk["timestamp"].values.astype(np.int64)
                wi    = np.searchsorted(win_starts_ns, ts_ns, side="right") - 1
                valid = (wi >= 0) & (wi < W)
                wi, si = wi[valid], si[valid]
                dur, is_error, is_root = dur[valid], is_error[valid], is_root[valid]

                np.add.at(acc_count,   (wi, si), 1.0)
                np.add.at(acc_dur_sum, (wi, si), dur)
                np.maximum.at(acc_max_dur, (wi, si), dur)
                np.add.at(acc_errors,  (wi, si), is_error)
                np.add.at(acc_roots,   (wi, si), is_root)
                n_chunks += 1

            logging.info(f"  {fname}: {n_chunks} chunks")

        safe = np.where(acc_count > 0, acc_count, 1.0)
        node_feats = np.zeros((W, S, self.TRACE_NODE_FEAT_DIM), dtype=np.float32)
        node_feats[:, :, 0] = acc_count                 # call_count
        node_feats[:, :, 1] = acc_dur_sum / safe        # avg_dur_ms
        node_feats[:, :, 2] = acc_max_dur               # max_dur_ms
        node_feats[:, :, 3] = acc_errors  / safe        # error_rate
        node_feats[:, :, 4] = acc_roots   / safe        # root_rate

        for col in [0, 1, 2]:
            mx = node_feats[:, :, col].max()
            if mx > 0:
                node_feats[:, :, col] /= mx

        logging.info("  Trace streaming complete.")
        return node_feats

    # ── Step 6: metric matrix (vectorised) ───────────────────────────────────

    def _discover_metric_groups(self) -> dict:
        """
        Group the 10,817 metric files by unique metric name
        (stripping the _YYYY-MM-DD_YYYY-MM-DD date-range suffix).
        Returns {metric_name: [file_path, ...]} for the top max_metrics metrics.
        """
        groups: dict = defaultdict(list)
        for fname in sorted(os.listdir(self.metric_dir)):
            if not fname.endswith(".csv"):
                continue
            base   = os.path.splitext(fname)[0]
            mname  = _DATE_SUFFIX_RE.sub("", base)
            groups[mname].append(os.path.join(self.metric_dir, fname))

        # NEW selection logic: ensure diversity across metric types
        _TYPE_KEYWORDS = ["cpu", "mem", "disk", "network", "io", "load", "fsstat", "system"]

        # Group by (service_prefix, metric_type)
        type_buckets: dict = defaultdict(list)
        for name, paths in groups.items():
            mtype = next((k for k in _TYPE_KEYWORDS if k in name.lower()), "other")
            type_buckets[mtype].append((name, paths))

        # Round-robin pick from each type bucket to fill max_metrics slots
        selected = {}
        per_type = max(1, self.max_metrics // max(len(type_buckets), 1))
        # First pass: take up to per_type from each type
        for mtype, items in type_buckets.items():
            # Sort within type by coverage (more splits = better)
            items_sorted = sorted(items, key=lambda x: -len(x[1]))
            for name, paths in items_sorted[:per_type]:
                if len(selected) >= self.max_metrics:
                    break
                selected[name] = paths
        # Second pass: fill remaining slots with most-covered metrics
        if len(selected) < self.max_metrics:
            remaining = [(n, p) for n, p in groups.items() if n not in selected]
            remaining_sorted = sorted(remaining, key=lambda x: -len(x[1]))
            for name, paths in remaining_sorted:
                if len(selected) >= self.max_metrics:
                    break
                selected[name] = paths
        result = selected
        logging.info(f"  Discovered {len(groups)} unique metrics, "
                     f"selected {len(result)} ({sum(len(v) for v in result.values())} files)")
        self.metric_names = sorted(result.keys())
        return result

    def _file_overlaps_range(self, path: str) -> bool:
        """
        Check if a metric file's date-range suffix overlaps [_t_min, _t_max].
        Filename pattern: ..._YYYY-MM-DD_YYYY-MM-DD.csv
        Returns True (load it) if no suffix found or dates overlap the range.
        """
        if self._t_min is None or self._t_max is None:
            return True
        base = os.path.splitext(os.path.basename(path))[0]
        m = _DATE_SUFFIX_RE.search(base)
        if not m:
            return True
        dates = re.findall(r'\d{4}-\d{2}-\d{2}', m.group())
        if len(dates) < 2:
            return True
        try:
            file_start = pd.Timestamp(dates[0])
            file_end   = pd.Timestamp(dates[1]) + pd.Timedelta(days=1)
            # Overlap: file_start < t_max  AND  file_end > t_min
            return file_start < self._t_max and file_end > self._t_min
        except Exception:
            return True

    def _build_kpi_matrix(self, metric_groups: dict,
                          win_starts_ns: np.ndarray, W: int) -> np.ndarray:
        """
        Build [W, M] KPI feature matrix fully vectorised.
        Only loads metric files whose date range overlaps [t_min, t_max]
        (skips August split files when time range is July only).
        """
        logging.info(f"Building KPI matrix ({W} × {len(self.metric_names)}) …")
        M       = len(self.metric_names)
        kpi_sum = np.zeros((W, M), dtype=np.float64)
        kpi_cnt = np.zeros((W, M), dtype=np.float64)

        for j, name in enumerate(self.metric_names):
            dfs = []
            for path in metric_groups[name]:
                # Skip files outside [t_min, t_max] (e.g. Aug splits)
                if not self._file_overlaps_range(path):
                    continue
                try:
                    # Metric CSVs have header: "timestamp,value"
                    df = pd.read_csv(path, low_memory=False)
                    # Rename if needed
                    if df.columns[0] != "timestamp":
                        df.columns = ["timestamp", "value"]
                    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
                    df["value"]     = pd.to_numeric(df["value"],     errors="coerce")
                    df = df.dropna()
                    dfs.append(df)
                except Exception as e:
                    logging.warning(f"  metric {os.path.basename(path)}: {e}")

            if not dfs:
                continue

            df_all = pd.concat(dfs, ignore_index=True)
            # Unix ms → nanoseconds for searchsorted
            ts_ns = (df_all["timestamp"].values.astype(np.int64)) * 1_000_000
            vals  = df_all["value"].values.astype(np.float32)

            wi    = np.searchsorted(win_starts_ns, ts_ns, side="right") - 1
            valid = (wi >= 0) & (wi < W)
            np.add.at(kpi_sum[:, j], wi[valid], vals[valid].astype(np.float64))
            np.add.at(kpi_cnt[:, j], wi[valid], 1.0)

            if (j + 1) % 10 == 0:
                logging.info(f"  KPI {j+1}/{M} done")

        safe_cnt   = np.where(kpi_cnt > 0, kpi_cnt, 1.0)
        kpi_matrix = np.where(kpi_cnt > 0,
                              kpi_sum / safe_cnt, 0.0).astype(np.float32)
        cov = float((kpi_cnt > 0).mean())
        logging.info(f"  KPI matrix built. Coverage: {cov:.1%}")
        return kpi_matrix

    # ── Step 7: log loading (chunked) ────────────────────────────────────────

    def _log_file_for_services(self, fname: str) -> bool:
        """
        Return True only if the log filename explicitly contains at least one
        of the selected service names.
        e.g. "business_table_webservice1_2021-07.csv" → True (contains "webservice1")
             "business_table_2021-08.csv"             → False (generic/combined file)
        This skips the large combined-month files that contain all services and
        are potentially huge (22 GB+) — only per-service extracts are loaded.
        """
        if not self.service2idx:
            return True
        fname_lower = fname.lower()
        return any(svc.lower() in fname_lower for svc in self.service2idx)

    def _load_logs_chunked(self):
        """
        Stream per-service business log CSVs (C engine, on_bad_lines='skip').
        Only files whose filename contains a selected service name are loaded
        (generic combined files like business_table_2021-08.csv are skipped).
        Only rows whose embedded timestamp falls within [t_min, t_max] are kept.

        EOF handling: if the C engine reads some chunks then hits a truncation
        error at the last row, the partial data is kept (not rolled back) and
        the file is considered successfully loaded — we lose at most one row.
        Returns (ts_ns: np.ndarray[int64], msgs: list[str]) sorted by timestamp.
        """
        logging.info(f"Loading business logs "
                     f"(range {self._t_min} → {self._t_max}) …")
        all_ts   = []
        all_msgs = []

        for path in _csv_files(self.log_dir):
            fname   = os.path.basename(path)
            size_gb = os.path.getsize(path) / 1e9

            # Skip combined/generic files — only load per-service extracts
            if not self._log_file_for_services(fname):
                logging.info(f"  {fname}: {size_gb:.1f} GB — skipped "
                             f"(not a per-service file)")
                continue

            logging.info(f"  {fname}: {size_gb:.1f} GB — loading …")
            n_rows  = 0
            success = False

            for engine in ("c", "python"):
                if success:
                    break
                ts_before   = len(all_ts)
                rows_before = len(all_msgs)
                n_chunks    = 0
                n_rows      = 0
                try:
                    read_kwargs = dict(
                        chunksize=LOG_CHUNK,
                        usecols=lambda c: c in ["datetime", "message"],
                        on_bad_lines="skip",
                        engine=engine,
                        dtype=str,
                    )
                    if engine == "c":
                        read_kwargs["low_memory"] = False

                    for chunk in pd.read_csv(path, **read_kwargs):
                        msg_col = chunk["message"].fillna("")
                        # Extract actual timestamps from the message prefix.
                        # MicroSS format: "YYYY-MM-DD HH:MM:SS,mmm | LEVEL | ..."
                        # The `datetime` column contains only the date (midnight).
                        extracted = msg_col.str.extract(
                            r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[,.](\d+)',
                            expand=True
                        )
                        has_ts = extracted[0].notna()
                        ts_str = extracted[0].str.cat(extracted[1], sep=".")
                        chunk["datetime"] = pd.NaT
                        chunk.loc[has_ts, "datetime"] = pd.to_datetime(
                            ts_str[has_ts], format="%Y-%m-%d %H:%M:%S.%f",
                            errors="coerce"
                        )
                        chunk = chunk.dropna(subset=["datetime"])
                        if chunk.empty:
                            n_chunks += 1
                            continue
                        # Keep only rows within our analysis time range
                        if self._t_min is not None and self._t_max is not None:
                            mask = ((chunk["datetime"] >= self._t_min) &
                                    (chunk["datetime"] <  self._t_max))
                            chunk = chunk[mask]
                        if chunk.empty:
                            n_chunks += 1
                            continue
                        # Force nanoseconds — pandas 3.x defaults to datetime64[us]
                        all_ts.append(
                            chunk["datetime"].dt.as_unit("ns").astype(np.int64).values
                        )
                        all_msgs.extend(msg_col[chunk.index].tolist())
                        n_rows   += len(chunk)
                        n_chunks += 1

                    logging.info(f"    {n_chunks} chunks, {n_rows:,} rows "
                                 f"in-range (engine={engine})")
                    success = True

                except Exception as e:
                    if engine == "c" and n_rows > 0:
                        # C engine read partial data before hitting EOF/truncation.
                        # Keep what we have — we lose at most the last (corrupt) row.
                        logging.warning(
                            f"    C engine partial read ({n_rows:,} rows, "
                            f"{n_chunks} chunks): {e} — keeping partial data"
                        )
                        success = True   # don't retry with python engine
                    elif engine == "c":
                        logging.warning(f"    C engine failed: {e} — "
                                        f"retrying with python engine …")
                        all_ts    = all_ts[:ts_before]
                        all_msgs  = all_msgs[:rows_before]
                    else:
                        logging.warning(
                            f"    python engine also failed: {e} — skipped"
                        )

        if not all_ts:
            logging.warning("  No log entries loaded — log branch will use padding.")
            return np.array([], dtype=np.int64), []

        ts_ns = np.concatenate(all_ts)
        order = np.argsort(ts_ns)
        ts_ns = ts_ns[order]
        msgs  = [all_msgs[i] for i in order]
        logging.info(f"  {len(ts_ns):,} log entries loaded and sorted")
        return ts_ns, msgs

    # ── Step 7b: Drain3 log parser ───────────────────────────────────────────

    # Fallback: normalise variable tokens (numbers, IPs, hex) to <*>
    _VAR_RE = re.compile(
        r'\b(?:[0-9a-f]{8,}|(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?|\d+)\b', re.IGNORECASE
    )

    def _regex_template(self, msg: str) -> str:
        """Simple regex-based fallback template normaliser."""
        parts = msg.split("|")
        # Use last field (message content) only; prepend service if available
        content = parts[-1].strip() if len(parts) >= 2 else msg
        svc     = parts[4].strip() if len(parts) >= 5 else ""
        normed  = self._VAR_RE.sub("<*>", content).strip()
        return f"{svc}|{normed}" if svc else normed

    def _parse_to_templates(self, msgs: list) -> list:
        """
        Convert raw MicroSS log messages to Drain3 template strings.

        MicroSS message format (pipe-delimited):
          YYYY-MM-DD HH:MM:SS,mmm | LEVEL | ip | ip | service | trace_id | content...

        Returns a list (same length as msgs) of template strings like
          "webservice|User <*> logged in from <*>"
        These are used as the "log event type" fed to FeatureExtractor / template_appear.
        """
        if not msgs:
            return []

        # Extract content fields for Drain3 (strip timestamp / level / IP header)
        def _content(msg: str) -> str:
            parts = msg.split("|")
            return parts[-1].strip() if len(parts) >= 2 else msg

        def _service(msg: str) -> str:
            parts = msg.split("|")
            return parts[4].strip() if len(parts) >= 5 else ""

        try:
            from drain3 import TemplateMiner
            from drain3.template_miner_config import TemplateMinerConfig

            config = TemplateMinerConfig()
            config.drain_depth          = 4
            config.drain_sim_th         = 0.5
            config.drain_max_children   = 100
            config.parametrize_numeric_tokens = True

            miner = TemplateMiner(config=config)
            templates = []
            log_interval = max(1, len(msgs) // 10)

            for i, msg in enumerate(msgs):
                content = _content(msg)
                svc     = _service(msg)
                result  = miner.add_log_message(content)
                tmpl    = result["template_mined"] if result else content
                templates.append(f"{svc}|{tmpl}" if svc else tmpl)
                if (i + 1) % log_interval == 0:
                    n_clusters = len(miner.drain.id_to_cluster)
                    logging.info(f"    Drain3: {i+1:,}/{len(msgs):,} msgs, "
                                 f"{n_clusters} templates so far")

            n_templates = len(miner.drain.id_to_cluster)
            logging.info(f"  Drain3 complete: {n_templates} unique templates extracted")
            return templates

        except ImportError:
            logging.warning(
                "drain3 not installed — install with: pip install drain3\n"
                "Falling back to regex normalisation (variable tokens → <*>)."
            )
            templates = [self._regex_template(m) for m in msgs]
            unique = len(set(templates))
            logging.info(f"  Regex normalisation complete: {unique} unique templates")
            return templates

    def _build_log_lists(self, log_ts_ns: np.ndarray, log_msgs: list,
                         win_starts_ns: np.ndarray, window_ns: int,
                         W: int) -> list:
        """
        Assign each log entry to its window using binary search.
        Returns list[list[str]] of length W.
        """
        win_logs = [None] * W
        if len(log_ts_ns) == 0:
            return [["padding"]] * W

        for i in range(W):
            lo = int(np.searchsorted(log_ts_ns, win_starts_ns[i],          side="left"))
            hi = int(np.searchsorted(log_ts_ns, win_starts_ns[i] + window_ns, side="left"))
            msgs = [m for m in log_msgs[lo:hi] if m.strip()]
            win_logs[i] = msgs if msgs else ["padding"]

        logging.info("  Log-to-window assignment complete.")
        return win_logs

    # ── Step 8: assemble and save ─────────────────────────────────────────────

    # Regex to extract log level from MicroSS message format:
    # "YYYY-MM-DD HH:MM:SS,mmm | LEVEL | ..."
    _LOG_LEVEL_RE = re.compile(r'\|\s*(INFO|ERROR|WARN|WARNING|DEBUG|TRACE)\s*\|')

    def _compute_log_features(self, msgs: list) -> np.ndarray:
        """
        Compute a compact 6-dim feature vector from raw log messages.
        Features: [error_rate, warn_rate, info_rate, retry_rate,
                   service_diversity_norm, log_count_norm]
        Returns np.float32 array of shape [6].
        """
        N_FEATS = 6
        real = [m for m in msgs if m != "padding" and m.strip()]
        if not real:
            return np.zeros(N_FEATS, dtype=np.float32)

        n = len(real)
        error_cnt = warn_cnt = info_cnt = retry_cnt = 0
        services_seen = set()

        for m in real:
            lv = self._LOG_LEVEL_RE.search(m)
            if lv:
                lvl = lv.group(1).upper()
                if lvl == "ERROR":                error_cnt += 1
                elif lvl in ("WARN", "WARNING"):  warn_cnt  += 1
                elif lvl == "INFO":               info_cnt  += 1
            if "retry" in m.lower():
                retry_cnt += 1
            # Extract service name from pipe-delimited format (field index 4)
            parts = m.split("|")
            if len(parts) >= 5:
                svc = parts[4].strip()
                if svc:
                    services_seen.add(svc)

        n_services = max(self.num_services, 1)
        feat = np.array([
            error_cnt / n,
            warn_cnt  / n,
            info_cnt  / n,
            retry_cnt / n,
            len(services_seen) / n_services,
            min(np.log1p(n) / 8.0, 1.0),   # log-scale count, capped at 1
        ], dtype=np.float32)
        return feat

    def _is_anomalous(self, t_start, t_end, win_error_rate: float = 0.0) -> int:
        for (s, e) in self.anomaly_periods:
            if not (t_end <= s or t_start >= e):
                return 1
        if not self.anomaly_periods and win_error_rate > 0.3:
            return 1
        return 0

    def _build_and_save(self, win_starts, node_feats_all, adj_global,
                        kpi_matrix, win_log_lists):
        W     = len(win_starts)
        delta = timedelta(seconds=self.window_sec)

        # Count unique log templates across all windows (excluding "padding")
        all_templates = set()
        for wl in win_log_lists:
            for t in wl:
                if t != "padding":
                    all_templates.add(t)
        n_log_templates = len(all_templates)
        logging.info(f"  Unique log templates: {n_log_templates}")

        logging.info(f"Assembling {W} windows …")

        train_data, test_data = {}, {}
        split_idx = int(W * self.train_ratio)
        n_anom    = 0

        for i in range(W):
            t_start = win_starts[i]
            t_end   = t_start + delta
            win_err = float(node_feats_all[i, :, 3].mean())
            label   = self._is_anomalous(t_start, t_end, win_err)
            n_anom += label

            block_id = hashlib.md5(str(t_start).encode()).hexdigest()[:10]
            sample = {
                "label":               label,
                "kpi_label":           label,
                "log_label":           label,
                "kpis":                kpi_matrix[i],
                # logs / seqs: Drain3 template strings (NOT raw messages).
                # FeatureExtractor.fit/transform will compute log_features
                # (template_appear vector) from these strings at training time.
                "logs":                win_log_lists[i],
                "seqs":                win_log_lists[i],
                # log_features: placeholder zero — overwritten by semantics.py
                "log_features":        np.zeros(max(n_log_templates, 1),
                                                dtype=np.float32),
                "trace_node_features": node_feats_all[i].copy(),
                "trace_adj":           adj_global.copy(),
            }

            if i < split_idx:
                if label == 0:
                    train_data[block_id] = sample
            else:
                test_data[block_id] = sample

        n_test_anom = sum(s["label"] for s in test_data.values())
        logging.info(f"  {W} windows | total anomaly rate: "
                     f"{n_anom}/{W} = {n_anom/max(1,W):.3f}")
        logging.info(f"  Train/unlabel: {len(train_data)} normal samples")
        logging.info(f"  Test:          {len(test_data)} samples | "
                     f"anomaly {n_test_anom}/{len(test_data)} = "
                     f"{n_test_anom/max(1,len(test_data)):.3f}")

        os.makedirs(self.output_dir, exist_ok=True)
        for split, data in [("train", train_data), ("unlabel", train_data),
                             ("test", test_data)]:
            path = os.path.join(self.output_dir, f"{split}.pkl")
            with open(path, "wb") as f:
                pickle.dump(data, f)
            logging.info(f"  Saved {path}")

        meta = {
            "num_services":    self.num_services,
            "service2idx":     self.service2idx,
            "metric_names":    self.metric_names,
            "kpi_c":           len(self.metric_names),
            # log_c = actual number of unique Drain3 templates
            # (used by run.py as fallback if semantics.py reports 0)
            "log_c":           n_log_templates,
            "trace_c":         self.TRACE_NODE_FEAT_DIM,
            "window_sec":      self.window_sec,
            "n_log_templates": n_log_templates,
        }
        with open(os.path.join(self.output_dir, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)
        logging.info(f"  meta.pkl saved → {self.output_dir}")
        logging.info("  ── Run model with ──────────────────────────────────────")
        logging.info(f"  python run.py --data {self.output_dir} --dataset micross")
        logging.info(f"    --data_type fuse --open_trace true")
        logging.info(f"    --num_services {self.num_services} --trace_c {self.TRACE_NODE_FEAT_DIM}")
        logging.info("  ─────────────────────────────────────────────────────────")

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self):
        # 1. Time range (streaming, no full load)
        t_min, t_max = self._scan_time_range()
        self._t_min, self._t_max = t_min, t_max   # used for log in-range filter

        # 2. Service index from 10K-row sample per file
        self._build_service_index_from_sample()

        # 3. Static adjacency from 50K-row sample per file
        adj_global = self._build_static_adjacency()

        # 4. Anomaly periods from run.zip (regex parse of message column)
        self._load_anomaly_periods()

        # 5. Build window list
        delta   = timedelta(seconds=self.window_sec)
        current = t_min
        win_starts = []
        while current < t_max:
            win_starts.append(current)
            current += delta
        W          = len(win_starts)
        win_ns     = np.array([ts.value for ts in win_starts], dtype=np.int64)
        window_ns  = int(self.window_sec * 1_000_000_000)
        logging.info(f"Total windows: {W} × {self.window_sec}s")

        # 6. Stream trace (7.5 GB) → per-window node features
        node_feats = self._stream_trace_to_windows(win_ns)

        # 7. Discover metric groups (10,817 files → 50 unique metrics × ~3 splits)
        #    then build [W, M] matrix fully vectorised
        metric_groups = self._discover_metric_groups()
        kpi_matrix    = self._build_kpi_matrix(metric_groups, win_ns, W)

        # 8. Load logs → parse to Drain3 templates → assign to windows
        log_ts_ns, log_msgs  = self._load_logs_chunked()
        log_templates        = self._parse_to_templates(log_msgs)  # raw → template str
        win_log_lists        = self._build_log_lists(log_ts_ns, log_templates,
                                                     win_ns, window_ns, W)

        # 9. Assemble and save (pure numpy/dict, no pandas per-window)
        self._build_and_save(win_starts, node_feats, adj_global,
                             kpi_matrix, win_log_lists)
        logging.info("Preprocessing complete.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Preprocess MicroSS (GAIA-DataSet) → UAC-AD pkl format"
    )
    p.add_argument("--trace_dir",    required=True,
                   help="Dir with extracted trace CSVs (e.g. .../trace/trace)")
    p.add_argument("--metric_dir",   required=True,
                   help="Dir with extracted metric CSVs (.../metric/metric)")
    p.add_argument("--log_dir",      required=True,
                   help="Dir with extracted business log CSVs (.../business/business)")
    p.add_argument("--run_dir",      default="",
                   help="Dir that contains run.zip (anomaly labels, optional)")
    p.add_argument("--output_dir",   default="../../data/micross")
    p.add_argument("--window_sec",   default=60,  type=int,
                   help="Window size in seconds. Paper (Table I) uses 60s for MicroSS.")
    p.add_argument("--train_ratio",  default=0.7, type=float)
    p.add_argument("--max_services", default=4,   type=int,
                   help="Max service types to use. Paper uses 4 for MicroSS. "
                        "Auto-selects one representative per service type by call count.")
    p.add_argument("--max_metrics",  default=85,  type=int,
                   help="Max unique metrics to use. Paper uses 85 for MicroSS.")
    p.add_argument("--services",     nargs="+",   default=None,
                   help="Explicit service names to use (overrides --max_services). "
                        "E.g.: --services webservice1 dbservice1 logservice1 redisservice1")
    args = p.parse_args()

    MicroSSPreprocessor(
        trace_dir    = args.trace_dir,
        metric_dir   = args.metric_dir,
        log_dir      = args.log_dir,
        run_dir      = args.run_dir,
        output_dir   = args.output_dir,
        window_sec   = args.window_sec,
        train_ratio  = args.train_ratio,
        max_services = args.max_services,
        max_metrics  = args.max_metrics,
        services     = args.services,
    ).run()


if __name__ == "__main__":
    main()
