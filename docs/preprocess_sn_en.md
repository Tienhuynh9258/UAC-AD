# Preprocessing: SocialNetwork (AnoMod) Dataset

## 1. Dataset Overview

The SocialNetwork (SN) dataset is part of the **AnoMod benchmark** for cloud microservice anomaly detection. It was collected from a 12-service microservice application deployed on a real cluster. The dataset captures three modalities:

| Modality    | Source                            | Location       |
|:------------|:----------------------------------|:---------------|
| KPI metrics | System + container + Jaeger spans | `metric_data/` |
| Logs        | Per-service log files             | `log_data/`    |
| Traces      | Distributed traces (Jaeger)       | `trace_data/`  |

### Scenarios

| Type                  | Count | Description                           |
|:----------------------|------:|:--------------------------------------|
| Normal_Baseline       |     1 | ~20 min of steady-state traffic       |
| Code_Stop_*           |     3 | Service process killed via code       |
| DB_Redis_CacheLimit_* |     3 | Redis cache limit injected on service |
| Perf_CPU_Contention   |     1 | CPU stress injected at host level     |
| Perf_Disk_IO_Stress   |     1 | Disk I/O stress injected              |
| Perf_Network_Loss     |     1 | Network packet loss injected          |
| Svc_Kill_*            |     3 | Container-level service kill          |

Each scenario runs approximately **20–25 minutes** of real-time data collection.

---

## 2. Feature Engineering

### 2.1 Windowing

Raw time-series data is divided into **non-overlapping windows** of `window_sec=30` seconds. Each window becomes one data point fed to the model.

**Why 30 seconds?**
- Short enough to capture transient anomalies (service kills manifest within seconds)
- Long enough to produce stable aggregated statistics (avoid noise from individual metric reads)
- 20-minute scenarios yield ~40 windows each, which is the maximum usable for a single Normal_Baseline recording

**Warmup skip:** The first `warmup_minutes=5` (10 windows) of each scenario are discarded to avoid initialization artifacts (services stabilizing, JVM warm-up, etc.). This is applied to both normal and anomaly scenarios.

Result: Normal_Baseline → **40 windows** after warmup; most anomaly scenarios → **30 windows** (19.5 min ÷ 30s = 39 raw, minus 10 warmup = 29–30).

### 2.2 KPI Features (59 dimensions)

| Group     | Count | Features                                                                                                                                      |
|:----------|------:|:----------------------------------------------------------------------------------------------------------------------------------------------|
| System    |    10 | cpu_usage, disk_io_time, disk_read_bytes, disk_usage_pct, disk_write_bytes, load1, memory_usage_pct, network_errors, network_receive_bytes, network_transmit_bytes |
| Container |    48 | 12 services × 4 metrics: cpu, memory, net_rx, net_tx                                                                                         |
| Jaeger    |     1 | spans_rate (result="ok", normalized per window)                                                                                               |

Each KPI is aggregated per window using the mean value of all samples within that 30-second interval. Missing values (e.g., a container not yet started) are filled with column means computed from non-NaN windows.

### 2.3 Log Features

Logs are parsed using **Drain3** (streaming template miner), fitted on Normal_Baseline logs only:
- Templates learned: ~458 from 317,055 log messages
- Feature type: `template_appear` — binary presence/absence of each template in the window
- New templates encountered in anomaly scenarios are treated as unseen (mapped to a special "unknown" bucket)

**Why fit Drain3 on Normal_Baseline only?**  
To avoid contamination from anomaly log patterns when building the vocabulary. The model should learn to flag unseen templates as anomalous, which is only possible if the template vocabulary was built from normal logs.

### 2.4 Trace Features (optional, `open_trace=True`)

For each service, a 6-dimensional feature vector is computed per window (`trace_c=6`):

```
[call_count, avg_duration_us, max_duration_us, error_rate, root_rate, latency_dev]
```

- `call_count`: number of trace spans involving this service in the window
- `avg_duration_us` / `max_duration_us`: latency statistics (normalized to seconds)
- `error_rate`: fraction of spans with non-OK HTTP status codes
- `root_rate`: fraction of spans that are root spans (entry points)
- `latency_dev`: z-score of `avg_duration` vs per-service baseline from Normal_Baseline
  = `(avg_dur − mean_baseline) / (std_baseline + 1e-6)` — positive means slower than normal

`latency_dev` baseline is computed once from **Normal_Baseline** `all_traces.csv` (per-service mean and std of `duration_us / 1e6`), then applied to all scenarios uniformly.

A **static adjacency matrix** (12×12) is built from Normal_Baseline traces: edge (i, j) = 1 if service i calls service j at least once. This graph is fixed for all scenarios — we assume the call graph topology does not change between experiments.

---

## 3. Split Strategy

### 3.1 Train / Val Split (Normal_Baseline)

The 40 Normal_Baseline windows are split **temporally** (no shuffling):

```
Normal_Baseline (40 windows)
├── train.pkl   = first 32 windows (80%)  → model training
├── unlabel.pkl = same 32 windows         → unsupervised/GAN phase
└── val.pkl     = last  8 windows (20%)   → threshold calibration (unseen during training)
```

**Why temporal split (not random)?**  
- Temporal order matters: the last windows of Normal_Baseline are the most "recent" normal state before anomaly injection begins
- Shuffling would contaminate: val windows could be interpolated from surrounding train windows (data leakage)
- The val set must represent truly unseen normal behavior for threshold calibration to be valid

**Why 80/20?**  
- 32 windows × 30s = 16 minutes of training data: sufficient for a GAN to learn normal KPI/log patterns
- 8 val windows → with `window_size=5`, produces **3 overlapping sequences** (windows 1–5, 2–6, 3–7) → ~3 loss values for computing the 95th percentile threshold
- A smaller val set (e.g., 4 windows) would give only 0 sequences with window_size=5

### 3.2 Test Files (Per-Scenario)

For each of the 12 anomaly scenarios, a test file is built:

```
test_{scenario_name}.pkl = 40 normal windows (from Normal_Baseline)
                         + 6 anomaly windows (subsampled from scenario)
                         → shuffled (random order)
```

**Why 40 normal + 6 anomaly?**
- **Anomaly rate ~13%** (6/46): realistic for production systems where anomalies are rare
- Avoids the inflated anomaly rate problem of using all anomaly windows (30+ windows) which would create long consecutive anomaly clusters in the sequence model after shuffling, inflating recall via `point_adjust`

**Why subsampling to 6 anomaly windows?**  
Original anomaly scenarios have 30–41 windows. Using all of them would raise the anomaly rate to ~43%, causing long contiguous anomaly clusters after shuffling → `point_adjust` would flag whole segments with a single detection → inflated recall/F1. By capping at `max_anomaly_windows=6`, the rate stays ~13% and the test is more realistic.

**Subsampling strategy:** evenly spaced indices across the scenario's full window range:
```python
indices = [int(round(i * (len(sc_samples) - 1) / (n - 1))) for i in range(n)]
```
This preserves the temporal diversity of anomaly patterns (early, middle, late phase of the anomaly event).

**Why shuffle?**  
The model receives a mixed stream (as in production) and must score each window individually. Without shuffling, all anomaly windows would be at the end — trivially detectable by position.

---

## 4. Threshold Calibration (No Data Leakage)

The anomaly threshold is computed **after training**, from val losses only:

```python
threshold = np.percentile(val_losses, val_percentile=95)
```

This replaces the old approach of sweeping `anomaly_rate` over the test set (which leaked ground truth labels). With the val-based threshold:
- No information from the test set is used to choose the threshold
- The 95th percentile means ~5% of normal sequences may be flagged as anomalies (expected FP rate)
- Applied as a fixed cutoff: each test sequence with `loss > threshold` is predicted anomalous

---

## 5. Output Structure

```
data/sn/
├── train.pkl              # 32 normal windows (first 80% of Normal_Baseline)
├── unlabel.pkl            # 32 normal windows (same as train, for GAN unlabeled phase)
├── val.pkl                # 8 normal windows (last 20%, unseen, for threshold)
├── meta.pkl               # dataset metadata (adj matrix, feature dims, etc.)
└── scenarios/
    ├── test_Code_Stop_MediaService_20251104_024819.pkl   # 46 windows, 6 anomaly, rate=0.13
    ├── test_Code_Stop_TextService_20251104_022416.pkl
    ├── test_Code_Stop_UserService_20251104_020019.pkl
    ├── test_DB_Redis_CacheLimit_HomeTimeline_20251104_004905.pkl
    ├── test_DB_Redis_CacheLimit_SocialGraph_20251104_013615.pkl
    ├── test_DB_Redis_CacheLimit_UserTimeline_20251104_011238.pkl
    ├── test_Perf_CPU_Contention_20251103_222601.pkl
    ├── test_Perf_Disk_IO_Stress_20251103_231335.pkl
    ├── test_Perf_Network_Loss_20251103_224954.pkl
    ├── test_Svc_Kill_Media_20251104_000111.pkl
    ├── test_Svc_Kill_SocialGraph_20251104_002506.pkl
    └── test_Svc_Kill_UserTimeline_20251103_233717.pkl
```

---

## 6. Usage

```bash
python codes/common/preprocess_sn.py \
    --sn_data_root D:/AnoMod/SN_data \
    --output_dir data/sn \
    --window_sec 30 \
    --warmup_minutes 5 \
    --max_anomaly_windows 6 \
    --seed 42
```

### Key Parameters

| Parameter               | Value | Rationale                                                              |
|:------------------------|------:|:-----------------------------------------------------------------------|
| `--window_sec`          |    30 | 30-second granularity: captures transient anomalies, stable aggregates |
| `--warmup_minutes`      |     5 | Skip first 5 min (10 windows) to avoid initialization noise           |
| `--max_anomaly_windows` |     6 | Cap anomaly windows per scenario → ~13% rate, avoids point_adjust inflation |
| `--seed`                |    42 | Reproducibility: controls shuffle order of test files                  |
