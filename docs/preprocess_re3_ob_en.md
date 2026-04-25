# Preprocessing RCAEval OnlineBoutique (RE3-OB) Dataset — `preprocess_rcaeval_re3_ob.py`

---

## 1. Dataset Overview

**RE3-OB** (RCAEval Online Boutique, code-defect faults) is a benchmark for multimodal anomaly detection on the OnlineBoutique (Google) microservice system. Unlike RE2-OB (infrastructure faults), RE3-OB injects **software bugs** directly into service logic — faults that do not change latency or call-graph topology and are therefore invisible to trace-only detectors.

| Parameter | Value |
|:----------|------:|
| Number of scenarios | 10 (4 services × subset of f1–f5) |
| Runs per scenario | 3 |
| Total experiments | 30 |
| Duration per run | ~24 minutes (1441 timesteps × 1 second) |
| Fault injection time | ~second 720 (mid-run) |
| Fault types | f1, f2, f3, f4, f5 (code-defect) |
| Injected services | adservice (f3,f4,f5), cartservice (f1), currencyservice (f1), emailservice (f1–f5) |
| Monitored services | 11 (in simple_metrics.csv) |

---

## 2. Input Data Structure

```
D:/RE3-OB/RE3-OB/
  {service}_{fault}/          # e.g. adservice_f3/, emailservice_f1/
    1/, 2/, 3/                # 3 independent runs
      simple_metrics.csv      # time(Unix sec), metric cols @ 1-second intervals
      logs.csv                # time, timestamp(ns), container_name, message, pod_name, node_name
      traces.csv              # time, traceID, spanID, serviceName, ...,
                              # startTimeMillis, startTime, duration, statusCode, parentSpanID
      inject_time.txt         # Unix timestamp (seconds) of fault injection
```

**Key file schemas:**

| File | Important Columns | Notes |
|:-----|:-----------------|:------|
| `simple_metrics.csv` | `time` (Unix sec), metric cols | 1-second interval; column count varies per run → aligned to canonical union |
| `logs.csv` | `timestamp` (ns), `container_name`, `message` | No `log_template` column → Drain3 used for template extraction |
| `traces.csv` | `serviceName`, `startTimeMillis`, `duration`, `statusCode`, `parentSpanID`, `spanID` | millisecond timestamps |
| `inject_time.txt` | Unix timestamp (seconds) | Fault injection start time |

---

## 3. Processing Pipeline Overview

```
D:/RE3-OB/RE3-OB/
  {service}_{fault}/1/, 2/, 3/
          │
          ▼
  RCAEvalRE3OBPreprocessor.run()
          │
          ├── Step 1 ──► Build canonical metric columns
          │                   Scan ALL simple_metrics.csv headers (30 runs)
          │                   → Union of all cols → N canonical metric cols
          │
          ├── Step 2 ──► Drain3 two-pass log template extraction
          │    ├── First pass:  fit TemplateMiner on ALL log messages
          │    └── Second pass: assign stable templates per message
          │                   Fallback: container_name only (if drain3 not installed)
          │
          ├── Step 3 ──► For each experiment (30 runs):
          │    ├── load_inject_time()      → inject_time (Unix sec)
          │    ├── load_kpi()              → timestamps[T], kpis[T,N]
          │    ├── build_log_features()    → logs[T][list[str]]
          │    ├── build_trace_features()  → node_feats[T,11,6], adj[T,11,11]
          │    ├── build_labels()          → labels[T]  (0/1)
          │    └── segment → (normal_samples, anomaly_samples)
          │
          ├── Step 4 ──► Merge all normal samples
          │                   Shuffle → 80% unlabel + 20% train
          │
          └── Step 5 ──► Build 5 test files by fault type
                              normal_pool + anomaly_{fault} → shuffle
                              Ensure anomaly_rate ≤ 20%
```

---

## 4. Step-by-Step Details

### Step 1 — Build Canonical Metric Columns

```
Problem: Column count in simple_metrics.csv varies per run
  → Cannot concatenate runs directly

Solution:
  1. Scan ALL 30 × simple_metrics.csv (header only, no data loaded)
  2. Take UNION of all metric columns (excluding 'time')
  3. Sort alphabetically → N canonical cols
  4. When loading each run: reindex to canonical cols, missing cols → fill 0

Result: kpis [T, N] for every run, uniform shape
```

### Step 2 — Drain3 Two-Pass Log Template Extraction

```
RE3-OB logs.csv has no 'log_template' column (unlike 24/90 RE2-OB runs).
All runs must use template extraction from raw 'message' text.

Two-pass approach:
  Pass 1 (fit): feed ALL messages from ALL 30 runs through Drain3 TemplateMiner
                → stabilises cluster boundaries before any data is extracted
  Pass 2 (extract): re-process each run, map message → cluster template
                    Token format: "{container_name}|{drain3_template}"

Drain3 config:
  drain_depth = 4, drain_sim_th = 0.5, drain_max_children = 100
  parametrize_numeric_tokens = True

Fallback (drain3 not installed):
  Token = container_name only  (~11 unique values)
  Install: pip install drain3

Time-bucket alignment:
  ts_sec = log["timestamp"] // 1_000_000_000   # nanoseconds → seconds
  wi = searchsorted(timestamps, ts_sec, side="right") - 1
  logs_per_ts[wi].append(token)

Empty timesteps → fill with ["padding"]
```

### Step 3a — Load KPI

```python
df = pd.read_csv(exp_dir / "simple_metrics.csv")
timestamps = pd.to_numeric(df.iloc[:, 0], errors="coerce").values.astype(int64)
kpis = df.iloc[:, 1:].reindex(columns=canonical_cols, fill_value=0.0).values  # [T, N]
```

### Step 3b — Build Trace Features (TRACE_C = 6)

```
Node feature layout [T, 11, 6]:
  col 0: call_count   — span count per service per timestep, normalized by global max
  col 1: avg_dur_ms   — mean(duration), normalized by global max
  col 2: max_dur_ms   — max(duration), normalized by global max
  col 3: error_rate   — fraction(statusCode ∉ {"0", "0.0", "nan"}) ∈ [0,1]
  col 4: root_rate    — fraction(parentSpanID ∈ root set) ∈ [0,1]
  col 5: latency_dev  — z-score of avg_dur_ms vs pre-fault baseline per service
                        = (avg_dur - mean_pre) / (std_pre + 1e-6)
                        Positive = slower than normal, Negative = faster

Computing latency_dev:
  pre_fault_idx = searchsorted(timestamps, inject_time, side="left")
  baseline_avg  = node_feats[:pre_fault_idx, :, 1].mean(axis=0)   # [11]
  baseline_std  = node_feats[:pre_fault_idx, :, 1].std(axis=0) + 1e-6
  node_feats[:, :, 5] = (node_feats[:, :, 1] - baseline_avg) / baseline_std

Adjacency matrix [T, 11, 11]:
  adj[t, parent_si, child_si] += 1  (parent span → child span, same time bucket)
  → Row-normalize: adj[t] /= adj[t].sum(axis=1, keepdims=True)

Service name alias: "redis-cart" → "redis"
Services not in SERVICES list → ignored
```

### Step 3c — Build Labels

```python
labels = (timestamps >= inject_time).astype(int32)
# label = 1 from fault injection time to end of run
```

### Step 3d — Segment Samples

```
Each experiment (1 run) produces:
  - normal_samples:  rows BEFORE inject_time → label = 0 (used for unlabel/train)
  - anomaly_samples: rows FROM inject_time   → label = 1 (used for test)
```

### Step 4 — Split Normal Samples

```
Pool = all normal_samples from 30 experiments
Shuffle (random_seed=42)
→ 80% → unlabel.pkl
→ 20% → train.pkl
```

### Step 5 — Build Test Files (5 fault types)

```
For each fault_type in {f1, f2, f3, f4, f5}:
  anomaly_pool = anomaly_samples from experiments with this fault_type

  normal_pool  = random sample from normal_samples such that:
                 anomaly_timesteps / total_timesteps ≤ 20%

  test_{fault}.pkl = shuffle(normal_pool + anomaly_pool)

Note: not all services have all fault types.
  f1: cartservice, currencyservice, emailservice      (3 services × 3 runs = 9 experiments)
  f2: emailservice                                    (1 service  × 3 runs = 3 experiments)
  f3: adservice, emailservice                         (2 services × 3 runs = 6 experiments)
  f4: adservice, emailservice                         (2 services × 3 runs = 6 experiments)
  f5: adservice, emailservice                         (2 services × 3 runs = 6 experiments)
```

---

## 5. Output Data

```
data/rcaeval_re3_ob/
  unlabel.pkl    ← 80% of normal samples
  train.pkl      ← 20% of normal samples
  test_f1.pkl    ← shuffle(normal_pool + anomaly_f1), ≤20% anomaly
  test_f2.pkl
  test_f3.pkl
  test_f4.pkl
  test_f5.pkl
  meta.pkl       ← metadata
```

**Sample format (1 timestep per sample):**

```python
{
  "label":               int,                 # 0 = normal, 1 = anomaly
  "kpis":                np.float32[N],       # N canonical metric values
  "logs":                list[str],           # template strings (for FeatureExtractor)
  "seqs":                list[str],           # same as logs (compatibility)
  "log_features":        np.float32[1],       # placeholder, overwritten by semantics.py
  "metric_name":         list[str],           # N metric column names
  "trace_node_features": np.float32[11, 6],   # [11 services, 6 features]
  "trace_adj":           np.float32[11, 11],  # row-normalized adjacency matrix
}
```

**meta.pkl:**

```python
{
  "num_services": 11,
  "service2idx":  {...},   # 11 services → index
  "trace_c":      6,
  "kpi_c":        N,       # number of canonical metric columns
  "log_c":        1,       # placeholder; updated by semantics.py at runtime
  "metric_names": [...],   # N metric names
  "fault_types":  ["f1", "f2", "f3", "f4", "f5"],
  "services":     ["adservice", "cartservice", ..., "shippingservice"],
}
```

---

## 6. Key Differences from RE2-OB Preprocessing

| Aspect | RE2-OB | RE3-OB |
|:-------|:-------|:-------|
| Fault type | Infrastructure (cpu, delay, disk, loss, mem, socket) | Code-defect (f1–f5) |
| Scenarios | 30 (5 services × 6 faults) | 10 (4 services × subset of faults) |
| Total experiments | 90 | 30 |
| Log extraction | `log_template` col present in 24/90 runs; container_name fallback | No `log_template` → Drain3 two-pass extraction on raw `message` |
| Test files | 6 (by fault type) | 5 (f1–f5) |
| Trace signal | Strongly discriminative for network/resource faults | Non-discriminative (code bugs don't change latency/topology) |

---

## 7. Run Commands

```bash
cd D:/UAC-AD/codes

python common/preprocess_rcaeval_re3_ob.py \
    --data_root D:/RE3-OB/RE3-OB \
    --output_dir ../data/rcaeval_re3_ob \
    --anomaly_rate 0.20 \
    --unlabel_ratio 0.80
```

**Arguments:**

| Argument | Default | Description |
|:---------|:--------|:------------|
| `--data_root` | (required) | Root directory of the RE3-OB dataset |
| `--output_dir` | `../../data/rcaeval_re3_ob` | Output directory |
| `--anomaly_rate` | 0.20 | Maximum anomaly fraction in test files |
| `--unlabel_ratio` | 0.80 | Fraction of normal data allocated to unlabel.pkl |
| `--random_seed` | 42 | Random seed for reproducibility |

**After preprocessing, run evaluation:**

```bash
# Trace eval (residual-gated, auto-applied when open_trace=True)
python common/eval_per_scenario_rcaeval_re3_ob.py \
    --data data/rcaeval_re3_ob --dataset rcaeval_re3_ob --data_type fuse \
    --open_trace True --trace_c 6 --batch_size 128 --window_size 30 \
    --epoches 5 5 --patience 3 --gate_lambda 0.01 \
    --result_dir data/rcaeval_re3_ob/result_per_scenario_fuse_trace

# Baseline eval (log + metric only)
python common/eval_per_scenario_rcaeval_re3_ob.py \
    --data data/rcaeval_re3_ob --dataset rcaeval_re3_ob --data_type fuse \
    --open_trace False --batch_size 128 --window_size 30 \
    --epoches 5 5 --patience 3 \
    --result_dir data/rcaeval_re3_ob/result_per_scenario_fuse_baseline
```
