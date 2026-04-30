# MicroSS Dataset Preprocessing — `preprocess_micross.py`

## 1. Input Data Structure

```
D:\GAIA-DataSet\MicroSS\
  ├── trace/
  │   └── trace/                  ← 10 CSV files (~7.5 GB total)
  │       ├── trace_table_dbservice1_2021-07.csv
  │       ├── trace_table_dbservice2_2021-07.csv
  │       ├── trace_table_redisservice1_2021-07.csv
  │       └── ...
  ├── metric/
  │   └── metric/                 ← 10,817 CSV files (metric × date-split)
  │       ├── dbservice1_..._2021-07-01_2021-07-15.csv
  │       ├── dbservice1_..._2021-07-15_2021-07-31.csv
  │       ├── dbservice1_..._2021-08-01_2021-08-31.csv  ← skipped (Aug)
  │       └── ...
  ├── business/
  │   └── business/               ← 2 CSV files (business logs)
  │       ├── business_table_2021-08.csv          (22.7 GB → skipped)
  │       └── business_table_webservice1_2021-07.csv  (1.5 GB → loaded)
  └── run/
      └── run.zip                 ← anomaly injection records
```

**CSV file schemas:**

| File     | Main columns                                                                                    |
|:---------|:------------------------------------------------------------------------------------------------|
| trace    | `timestamp`, `service_name`, `span_id`, `parent_id`, `start_time`, `end_time`, `status_code`   |
| metric   | `timestamp` (Unix ms 13 digits), `value`                                                        |
| business | `datetime` (YYYY-MM-DD HH:MM:SS), `service`, `message`                                         |
| run      | `datetime`, `service`, `message` (anomaly info embedded in text)                               |

---

## 2. Processing Pipeline Overview

```
D:\GAIA-DataSet\MicroSS\
  ├── trace/trace/
  ├── metric/metric/
  ├── business/business/
  └── run/run.zip
          │
          ▼
  MicroSSPreprocessor.run()
          │
          ├── Step 1 ──► Scan Time Range
          │                   t_min = 2021-07-01 09:57:00
          │                   t_max = 2021-08-01 00:00:00
          │
          ├── Step 2 ──► Build Service Index
          │                   10 services: dbservice1..webservice2
          │
          ├── Step 3 ──► Build Static Adjacency  [10×10]  (row-normalized)
          │                   30 directed edges (call graph)
          │
          ├── Step 4 ──► Load Anomaly Periods
          │                   1,367 injection periods from run.zip
          │
          ├── Step 5 ──► Build Window List
          │                   44,043 windows × 60s
          │
          ├── Step 6 ──► Stream Trace → node_feats  [W, 10, 5]
          │
          ├── Step 6b ─► Compute latency_dev (col 5) → node_feats  [W, 10, 6]
          │
          ├── Step 7 ──► Build KPI Matrix            [W, 50]
          │
          ├── Step 8 ──► Load Business Logs → win_logs  [W × list[str]]
          │
          └── Step 9 ──► Assemble & Save pkl files
                              train.pkl / unlabel.pkl / test.pkl / meta.pkl
```

---

## 3. Step-by-Step Details

### Step 1 — Scan Time Range

```
For each trace CSV:
  pd.read_csv(path, usecols=["timestamp"], chunksize=500K)
  → find min/max timestamp

Result:
  t_min = 2021-07-01 09:57:00  (floored to minute)
  t_max = 2021-08-01 00:00:00

→ Used as filter for metrics and logs in subsequent steps
  (only July data is kept, August is skipped)
```

---

### Step 2 — Build Service Index

```
For each trace CSV:
  pd.read_csv(path, nrows=10,000, usecols=["service_name"])
  → count occurrence frequency per service

Top 10 services → sorted alphabetically:
  service2idx = {
    "dbservice1":   0,
    "dbservice2":   1,
    "logservice1":  2,
    "logservice2":  3,
    "mobservice1":  4,
    "mobservice2":  5,
    "redisservice1":6,
    "redisservice2":7,
    "webservice1":  8,
    "webservice2":  9,
  }
```

---

### Step 3 — Build Static Adjacency

```
Pass 1 — build span_id → service map:
  Read 50K rows/file: {span_id → service_name}

Pass 2 — build call graph:
  Read 50K rows/file: for each span with parent_id != "0":
    parent_svc = span_svc[parent_id]
    child_svc  = service_name
    adj[parent_svc_idx, child_svc_idx] = 1.0

Row-normalize (consistent with RE2-OB / RE3-OB):
  row_sum = adj.sum(axis=1, keepdims=True)
  adj = adj / row_sum  (rows with no outgoing edges stay 0)

Result: adj [10×10] — row-normalized directed adjacency matrix
  30 directed edges (who calls whom in the microservice system)

Example (illustrative):
  webservice → dbservice    (web calls DB)
  webservice → redisservice (web calls cache)
  mobservice → logservice   (mobile calls log)
  ...
```

---

### Step 4 — Load Anomaly Periods

```
Open run/run.zip → read CSV inside
  Column "message" contains text of the form:
  "... start at 2021-07-01 11:44:26.882752 and lasts 600 seconds ..."

Regex:
  r'start at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[\d.]*) and lasts (\d+) seconds'

→ Parsed into: (start_timestamp, end_timestamp)
→ 1,367 anomaly periods
→ First interval: 2021-07-01 11:44:26 → 11:54:26 (10 minutes)
```

---

### Step 5 — Build Window List

```
current = t_min
while current < t_max:
    win_starts.append(current)
    current += 60s

→ 44,043 windows
→ win_starts_ns [44043] — timestamps as int64 nanoseconds
   (used for np.searchsorted in subsequent steps)
```

---

### Step 6 — Stream Trace → Node Features

```
Input:  trace CSV files (10 files, ~7.5 GB)
Output: node_feats [W=44043, S=10, C=6]

┌─────────────────────────────────────────────────────────────────┐
│  For each trace file, read chunks of 500K rows:                 │
│                                                                 │
│  Compute:                                                       │
│    duration  = end_time - start_time  (milliseconds)            │
│    is_error  = (status_code != "200")                           │
│    is_root   = (parent_id == "0")                               │
│    si        = service2idx[service_name]                        │
│                                                                 │
│  Assign to window via binary search:                            │
│    wi = searchsorted(win_starts_ns, ts_ns, side="right") - 1    │
│    valid = (wi >= 0) & (wi < W)                                 │
│                                                                 │
│  Accumulate using np.add.at / np.maximum.at:                   │
│    acc_count[wi, si]    += 1                                    │
│    acc_dur_sum[wi, si]  += duration                             │
│    acc_max_dur[wi, si]   = max(acc_max_dur, duration)           │
│    acc_errors[wi, si]   += is_error                             │
│    acc_roots[wi, si]    += is_root                              │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
node_feats [W, S, 5]:  (col 5 zero-filled at this stage)
  dim 0 — call_count   (number of calls within window, normalized)
  dim 1 — avg_dur_ms   (average response time, normalized)
  dim 2 — max_dur_ms   (maximum response time, normalized)
  dim 3 — error_rate   (fraction of spans with status != 200)
  dim 4 — root_rate    (fraction of spans that are root calls)
```

> **Why use `np.add.at` instead of a loop?**
> The dataset contains tens of millions of spans. A Python loop would take many hours.
> `np.add.at` accumulates directly into a NumPy array; this entire step completes in ~6 minutes.

---

### Step 6b — Compute latency_dev (col 5)

```
After streaming all trace data, compute latency deviation relative to the
normal training baseline (first train_ratio=70% of windows).

  split_idx  = int(W * 0.7)                       # = 30,830
  bl_mean[s] = mean(node_feats[:split_idx, s, 1]) # per-service avg_dur mean
  bl_std[s]  = std (node_feats[:split_idx, s, 1]) # per-service avg_dur std

  node_feats[:, :, 5] = (node_feats[:, :, 1] - bl_mean) / (bl_std + 1e-6)

→ node_feats [W, S, 6]:
  dim 5 — latency_dev  (z-score of avg_dur vs normal-train baseline per service)
           positive = slower than normal, negative = faster

Consistent with RE2-OB / RE3-OB (which use pre-injection window as baseline).
```

---

### Step 7 — Build KPI Matrix

```
Input:  10,817 metric CSV files
Output: kpi_matrix [W=44043, M=50]

┌─────────────────────────────────────────────────────────────────┐
│  _discover_metric_groups()                                      │
│                                                                 │
│  10,817 files with names like:                                  │
│    dbservice1_0.0.0.4_docker_cpu_core_0_2021-07-01_2021-07-15  │
│    dbservice1_0.0.0.4_docker_cpu_core_0_2021-07-15_2021-07-31  │
│    dbservice1_0.0.0.4_docker_cpu_core_0_2021-08-01_2021-08-31  │
│                                           ↑                     │
│  Strip suffix _YYYY-MM-DD_YYYY-MM-DD ────┘                      │
│  → 4,967 unique metrics                                         │
│  → Select top 50 (most date-splits)                             │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  _file_overlaps_range()  — load July only, skip August          │
│                                                                 │
│  For each file, read suffix _YYYY-MM-DD_YYYY-MM-DD:             │
│    file_start = 2021-08-01  →  file_start >= t_max  → SKIP      │
│    file_start = 2021-07-01  →  overlaps July range  → LOAD      │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Vectorised binning:                                            │
│    ts_ns = timestamp_unix_ms × 1,000,000                        │
│    wi = searchsorted(win_starts_ns, ts_ns) - 1                  │
│    np.add.at(kpi_sum[:, j], wi, value)                          │
│    np.add.at(kpi_cnt[:, j], wi, 1.0)                            │
│                                                                 │
│  kpi_matrix = kpi_sum / kpi_cnt  (0 if no data)                │
└─────────────────────────────────────────────────────────────────┘

Coverage: 83.2%
(~16.8% of windows have no metric data → filled with 0)
```

---

### Step 8 — Load Business Logs

```
Files in business/business/:
  business_table_2021-08.csv          (22.7 GB)  → SKIP (> 5 GB limit)
  business_table_webservice1_2021-07.csv (1.5 GB) → LOAD

┌─────────────────────────────────────────────────────────────────┐
│  Try C engine (chunk 300K rows):                                │
│    ❌ "EOF inside string at row 7,285,312"                       │
│       (embedded newline \n in quoted field, chunk cut           │
│        in the middle of a field → C engine cannot handle)      │
│                                                                 │
│  Fallback: Python engine:                                       │
│    ⚠️  Successfully loads most of the file                      │
│       Stops when encountering a malformed row near end of file  │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
  Filter: keep only rows with datetime ∈ [t_min, t_max]
  Sort by datetime
  → 7,051,685 log entries (ts_ns[], msgs[])

              │
              ▼  _build_log_lists() — binary search
  For each window i:
    lo = searchsorted(log_ts_ns, win_start)
    hi = searchsorted(log_ts_ns, win_start + 60s)
    win_logs[i] = msgs[lo:hi]
                  (or ["padding"] if no logs in window)
```

---

### Step 9 — Assemble & Save

```
For each window i in 44,043:
┌─────────────────────────────────────────────────────────────────┐
│  Assign anomaly label:                                          │
│    Check if [t_start, t_end) overlaps with any anomaly period   │
│    label = 1 if overlap, 0 otherwise                            │
│                                                                 │
│  Create sample dict:                                            │
│  {                                                              │
│    "label":               0 or 1,                               │
│    "kpis":                kpi_matrix[i],           shape [50]   │
│    "logs":                win_logs[i],             list[str]    │
│    "trace_node_features": node_feats[i],           shape [10,6] │
│    "trace_adj":           adj_global,              shape [10,10]│
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘

Split 70% / 30%:
  i < 30,830  →  train:  only normal samples (label=0)
  i >= 30,830 →  test:   all samples

Result:
  train.pkl   — 26,235 normal samples
  unlabel.pkl — (same as train, used for semi-supervised)
  test.pkl    — 13,213 samples (2,064 anomaly = 15.6%)
  meta.pkl    — {num_services:10, kpi_c:50, trace_c:6, window_sec:60, ...}
```

---

## 4. Output Statistics

| Parameter             | Value          |
|:----------------------|---------------:|
| Processing time       |        ~8 min  |
| Total windows         |         44,043 |
| Window size           |        60 sec  |
| Services              |             10 |
| Metrics (KPI)         |             50 |
| Log entries loaded    |      7,051,685 |
| Anomaly periods       |          1,367 |
| Anomaly rate (total)  |          15.1% |
| Anomaly rate (test)   |          15.6% |
| KPI coverage          |          83.2% |

---

## 5. Re-running Preprocessing

```bash
python preprocess_micross.py \
    --trace_dir   "D:\GAIA-DataSet\MicroSS\trace\trace" \
    --metric_dir  "D:\GAIA-DataSet\MicroSS\metric\metric" \
    --log_dir     "D:\GAIA-DataSet\MicroSS\business\business" \
    --run_dir     "D:\GAIA-DataSet\MicroSS\run" \
    --output_dir  "../../data/micross" \
    --window_sec  60 \
    --max_metrics 50
```

**After completion, run evaluation:**

```bash
# Baseline (log + KPI only)
python codes/common/eval_micross.py \
    --data data/micross --dataset micross --data_type fuse \
    --open_trace False --val_percentile 95 \
    --result_dir data/micross/result_fuse_baseline

# Trace (log + KPI + trace GAT, residual gate)
python codes/common/eval_micross.py \
    --data data/micross --dataset micross --data_type fuse \
    --open_trace True --num_services 10 --trace_c 6 \
    --gate_lambda 0.01 --val_percentile 95 \
    --result_dir data/micross/result_fuse_trace
```

---

## 6. Model Input Data Schema

```
train.pkl / test.pkl
  {
    block_id_xxxx: {
      "label":               int (0=normal, 1=anomaly)
      "kpis":                float32 [50]        ← from metric CSV
      "logs":                list[str]            ← from business log CSV
      "trace_node_features": float32 [10, 6]     ← from trace CSV
      "trace_adj":           float32 [10, 10]    ← from trace CSV (static, row-normalized)
    },
    block_id_yyyy: { ... },
    ...
  }

meta.pkl
  {
    "num_services": 10,
    "service2idx":  {name: idx, ...},
    "metric_names": [str × 50],
    "kpi_c":        50,
    "trace_c":      6,   # 6 features: call_count, avg_dur, max_dur, error_rate, root_rate, latency_dev
    "window_sec":   60,
  }
```
