# Preprocessing RCAEval OnlineBoutique (RE2-OB) Dataset — `preprocess_rcaeval_re2_ob.py`

---

## 1. Dataset Overview

**RE2-OB** (RCAEval Online Boutique, infrastructure faults) is a benchmark for multimodal anomaly detection on the OnlineBoutique (Google) microservice system.

| Parameter | Value |
|:----------|------:|
| Number of scenarios | 30 (5 services × 6 fault types) |
| Runs per scenario | 3 |
| Total experiments | 90 |
| Duration per run | ~24 minutes (1441 timesteps × 1 second) |
| Fault injection time | ~second 720 (mid-run) |
| Fault types | cpu, delay, disk, loss, mem, socket |
| Injected services | checkoutservice, currencyservice, emailservice, productcatalogservice, recommendationservice |
| Monitored services | 11 (in simple_metrics.csv) |

---

## 2. Input Data Structure

```
D:/RE2-OB/RE2-OB/
  {service}_{fault}/          # e.g. checkoutservice_cpu/
    1/, 2/, 3/                # 3 independent runs
      simple_metrics.csv      # [1441 rows, ~73 cols] — 1-second intervals
      logs.csv                # timestamp(ns), container_name, log_template
      traces.csv              # serviceName, startTimeMillis, duration,
                              # statusCode, parentSpanID, spanID
      cluster_info.json       # template_id → {template, container[]}
      inject_time.txt         # Unix timestamp (seconds) of fault injection
```

**Key file schemas:**

| File | Important Columns | Notes |
|:-----|:-----------------|:------|
| `simple_metrics.csv` | `time` (Unix sec), 73 metric cols | 1-second interval; column count varies per run (69–78) |
| `logs.csv` | `timestamp` (ns), `container_name`, `log_template` | 24/90 runs have `log_template`; 66/90 don't → fallback to `container_name` |
| `traces.csv` | `serviceName`, `startTimeMillis`, `duration`, `statusCode`, `parentSpanID`, `spanID` | millisecond timestamps |
| `cluster_info.json` | `{template_id: {template, container[]}}` | Maps template ID → template text |
| `inject_time.txt` | Unix timestamp (seconds) | Fault injection start time |

---

## 3. Processing Pipeline Overview

```
D:/RE2-OB/RE2-OB/
  {service}_{fault}/1/, 2/, 3/
          │
          ▼
  RCAEvalOBPreprocessor.run()
          │
          ├── Step 1 ──► Build canonical metric columns
          │                   Scan ALL simple_metrics.csv headers
          │                   → Union of 73–78 cols → 86 canonical cols
          │
          ├── Step 2 ──► For each experiment (90 runs):
          │    ├── load_inject_time()      → inject_time (Unix sec)
          │    ├── load_kpi()              → timestamps[T], kpis[T,86]
          │    ├── build_log_features()    → logs[T][list[str]]
          │    ├── build_trace_features()  → node_feats[T,11,6], adj[T,11,11]
          │    ├── build_labels()          → labels[T]  (0/1)
          │    └── segment → (normal_session, full_session)
          │
          ├── Step 3 ──► Merge all normal sessions
          │                   Shuffle → 80% unlabel + 20% train
          │
          └── Step 4 ──► Build 6 test files by fault type
                              normal_pool + anomaly_{fault} → shuffle
                              Ensure anomaly_rate ≤ 20%
```

---

## 4. Step-by-Step Details

### Step 1 — Build Canonical Metric Columns

```
Problem: Column count in simple_metrics.csv varies per run (69–78 cols)
  → Cannot concatenate runs directly

Solution:
  1. Scan ALL 90 × simple_metrics.csv (header only, no data loaded)
  2. Take UNION of all metric columns (excluding 'time')
  3. → 86 canonical cols (sorted alphabetically)
  4. When loading each run: reindex to canonical cols, missing cols → fill 0

Result: kpis [T, 86] for every run, uniform shape
```

### Step 2a — Load KPI

```python
df = pd.read_csv(exp_dir / "simple_metrics.csv")
timestamps = df.iloc[:, 0].values.astype(int64)   # Unix seconds
kpis = df.reindex(columns=canonical_cols).values   # [T, 86], fill 0 if missing
```

### Step 2b — Build Log Features

```
Two logs.csv formats in the dataset:
  - Full (24/90 runs): has 'log_template' column → use directly (~145 unique templates)
  - Minimal (66/90 runs): no 'log_template' column → fallback to 'container_name'
                           (11 unique values, 1 per service)

Using raw 'container_name + message': 86k+ unique strings → FeatureExtractor extremely slow
→ Use container_name-only when log_template is missing → vocab ~156 unique strings

Time-bucket alignment:
  ts_sec = log["timestamp"] // 1_000_000_000   # nanoseconds → seconds
  wi = searchsorted(timestamps, ts_sec, side="right") - 1
  logs_per_ts[wi].append(template_or_container)

Empty timesteps → fill with ["padding"]
```

### Step 2c — Build Trace Features (TRACE_C = 6)

```
Node feature layout [T, 11, 6]:
  col 0: call_count   — number of spans in time bucket, normalized by global max
  col 1: avg_dur_ms   — mean(duration), normalized
  col 2: max_dur_ms   — max(duration), normalized
  col 3: error_rate   — fraction(statusCode ∉ {"0", "0.0", "nan"}) ∈ [0,1]
  col 4: root_rate    — fraction(parentSpanID ∈ {"", "0", "nan", "None"}) ∈ [0,1]
  col 5: latency_dev  — z-score of avg_dur_ms vs pre-fault baseline per service
                        = (avg_dur - mean_pre) / (std_pre + 1e-6)
                        Positive = slower than normal, Negative = faster

Computing latency_dev:
  pre_fault_idx = searchsorted(timestamps, inject_time, side="left")
  baseline_avg  = node_feats[:pre_fault_idx, :, 1].mean(axis=0)   # [11]
  baseline_std  = node_feats[:pre_fault_idx, :, 1].std(axis=0) + 1e-6
  node_feats[:, :, 5] = (node_feats[:, :, 1] - baseline_avg) / baseline_std

Adjacency matrix [T, 11, 11]:
  adj[t, caller_idx, callee_idx] += 1  (parent span → child span)
  → Row-normalize: adj[t] /= adj[t].sum(axis=1, keepdims=True)

Service name alias: "redis-cart" → "redis"
Services not in SERVICES list → ignored
```

### Step 2d — Build Labels

```python
labels = np.where(timestamps >= inject_time, 1, 0)
# label = 1 from fault injection time to end of run
```

### Step 2e — Segment Sessions

```
Each experiment produces 2 session types (1 timestep = 1 session):
  - normal_session: rows BEFORE inject_time  → label = 0 (used for unlabel/train)
  - full_session:   ALL rows                 → label = 0/1 (used for test)
```

### Step 3 — Split Normal Sessions

```
Pool = all normal_sessions from 90 experiments (~65k timesteps)
Shuffle (random_seed=42)
→ 80% → unlabel.pkl  (52,251 samples)
→ 20% → train.pkl    (13,063 samples)
```

### Step 4 — Build Test Files (6 fault types)

```
For each fault_type in {cpu, delay, disk, loss, mem, socket}:
  anomaly_pool = full_sessions from 15 experiments with this fault_type
                 (5 services × 3 runs = 15 runs)

  normal_pool  = random sample from normal_sessions such that:
                 anomaly_timesteps / total_timesteps ≤ 20%

  test_{fault}.pkl = shuffle(normal_pool + anomaly_pool)

Actual sample counts:
  test_cpu.pkl    : 51,520 samples (10,304 anomaly ≈ 20%)
  test_delay.pkl  : 54,075 samples (10,815 anomaly ≈ 20%)
  test_disk.pkl   : 54,075 samples (10,815 anomaly ≈ 20%)
  test_loss.pkl   : 54,075 samples (10,815 anomaly ≈ 20%)
  test_mem.pkl    : 51,889 samples (10,378 anomaly ≈ 20%)
  test_socket.pkl : 54,075 samples (10,815 anomaly ≈ 20%)
```

---

## 5. Output Data

```
data/rcaeval_re2_ob/
  unlabel.pkl     ← 52,251 normal samples
  train.pkl       ← 13,063 normal samples
  test_cpu.pkl    ← 51,520 samples (≤20% anomaly)
  test_delay.pkl  ← 54,075 samples
  test_disk.pkl   ← 54,075 samples
  test_loss.pkl   ← 54,075 samples
  test_mem.pkl    ← 51,889 samples
  test_socket.pkl ← 54,075 samples
  meta.pkl        ← metadata
```

**Sample format (1 timestep per sample):**

```python
{
  "label":               int,                # 0 = normal, 1 = anomaly
  "kpis":                np.float32[86],     # 86 metric values
  "logs":                list[str],          # template strings (for FeatureExtractor)
  "seqs":                list[str],          # same as logs (compatibility)
  "log_features":        np.float32[1],      # placeholder, overwritten by semantics.py
  "metric_name":         list[str],          # 86 metric column names
  "trace_node_features": np.float32[11, 6],  # [11 services, 6 features]
  "trace_adj":           np.float32[11, 11], # row-normalized adjacency matrix
}
```

**meta.pkl:**

```python
{
  "num_services": 11,
  "trace_c":      6,
  "kpi_c":        86,
  "log_c":        1,
  "metric_names": [...],   # 86 metric names
  "service2idx":  {...},   # 11 services → index
  "fault_types":  ["cpu", "delay", "disk", "loss", "mem", "socket"],
  "services":     ["adservice", "cartservice", ..., "shippingservice"],
}
```

---

## 6. Run Commands

```bash
cd D:/UAC-AD/codes

python common/preprocess_rcaeval_re2_ob.py \
    --data_root D:/RE2-OB/RE2-OB \
    --output_dir ../data/rcaeval_re2_ob \
    --anomaly_rate 0.20 \
    --unlabel_ratio 0.80
```

**Arguments:**

| Argument | Default | Description |
|:---------|:--------|:------------|
| `--data_root` | (required) | Root directory of the RE2-OB dataset |
| `--output_dir` | (required) | Output directory |
| `--anomaly_rate` | 0.20 | Maximum anomaly fraction in test files |
| `--unlabel_ratio` | 0.80 | Fraction of normal data allocated to unlabel.pkl |

**After preprocessing, run evaluation:**

```bash
# Trace eval
python common/eval_per_scenario_rcaeval_re2_ob.py \
    --data data/rcaeval_re2_ob --dataset rcaeval_re2_ob --data_type fuse \
    --open_trace True --trace_c 6 --batch_size 128 --window_size 30 \
    --epoches 5 5 --patience 3 \
    --result_dir data/rcaeval_re2_ob/result_per_scenario_fuse_trace

# Baseline eval
python common/eval_per_scenario_rcaeval_re2_ob.py \
    --data data/rcaeval_re2_ob --dataset rcaeval_re2_ob --data_type fuse \
    --open_trace False --batch_size 128 --window_size 30 \
    --epoches 5 5 --patience 3 \
    --result_dir data/rcaeval_re2_ob/result_per_scenario_fuse_baseline
```
