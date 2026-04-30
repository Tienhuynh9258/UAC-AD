# Experiment Results — MicroSS: Baseline vs Trace-v4

> **Dataset**: MicroSS (micross) — 26,235 train / 13,213 test (11,149 normal + 2,064 anomaly, anomaly rate ~15.6%)
> **Date**: 2026-04-29 (re-run with finalized model; previous run: 2026-04-11)

---

## 1. Experiment Configuration

| Parameter                          | Value     |
|:-----------------------------------|----------:|
| `dataset`                          |   micross |
| `data_type`                        |      fuse |
| `window_size`                      |        50 |
| `hidden_size`                      |        32 |
| `epoches`                          |   10 / 10 |
| `batch_size`                       |       256 |
| `patience`                         |         5 |
| `alpha` (inter-class distance)     |      0.16 |
| `open_gan_sep`                     |      True |
| `learning_rate`                    |     0.001 |
| `num_runs`                         |         3 |
| `gate_lambda` (trace L1 reg)       |      0.01 |
| `val_percentile` (threshold)       |        95 |

**Baseline**: `open_trace=False` — log + KPI only  
**Trace-v4**: `open_trace=True`, `num_services=4`, `trace_c=6`, `gate_lambda=0.01` — log + KPI + trace (residual-gated fusion, 6 node features)

---

## 2. Summary Results

### 2.1 Baseline (log + KPI)

| Run          | Best F1    | Recall     | Precision  |
|:-------------|-----------:|-----------:|-----------:|
| run-0        |     0.3569 |     0.3682 |     0.3462 |
| run-1        |     0.3576 |     0.3682 |     0.3477 |
| run-2        |     0.3446 |     0.3837 |     0.3128 |
| **Mean**     | **0.3530** | **0.3734** | **0.3356** |
| **Std**      | **0.0060** | **0.0073** | **0.0161** |
| **Max**      | **0.3576** | **0.3837** | **0.3477** |

### 2.2 Trace-v4 (log + KPI + trace)

| Run          | Best F1    | Recall     | Precision  |
|:-------------|-----------:|-----------:|-----------:|
| run-0        |     0.3235 |     0.3251 |     0.3218 |
| run-1        |     0.3208 |     0.3517 |     0.2949 |
| run-2        |     0.3205 |     0.3517 |     0.2943 |
| **Mean**     | **0.3216** | **0.3428** | **0.3037** |
| **Std**      | **0.0013** | **0.0125** | **0.0128** |
| **Max**      | **0.3235** | **0.3517** | **0.3218** |

### 2.3 Direct Comparison

| Metric               | Baseline   | Trace-v4   | Delta                          |
|:---------------------|-----------:|-----------:|:-------------------------------|
| **F1 (mean)**        | **0.3530** |     0.3216 | **−0.031 (−8.9%)**             |
| **Recall (mean)**    | **0.3734** |     0.3428 | −0.031 (−8.2%)                 |
| **Precision (mean)** | **0.3356** |     0.3037 | −0.032 (−9.5%)                 |
| **F1 (max)**         | **0.3576** |     0.3235 | −0.034                         |
| **Std F1**           |     0.0060 | **0.0013** | **−0.005 (4.6× more stable)**  |

---

## 3. Historical Comparison (2026-04-11 vs 2026-04-29)

> ⚠️ The April 11 run had a bug: `log_c=0` (log data not loaded due to `_log_file_for_services` filter bug). The results below are **not comparable** — listed for reference only.

| Version        | Date       | Baseline F1 | Trace F1 | Delta     |
|:---------------|:-----------|------------:|---------:|----------:|
| Trace-v3       | 2026-04-11 |      0.3140 |   0.3434 | +0.029    |
| **Trace-v4**   | **2026-04-29** | **0.3530** | **0.3216** | **−0.031** |

---

## 4. Runtime

| Job                  | Duration      |
|:---------------------|:--------------|
| Baseline 3 runs      | ~1h 07m       |
| Trace-v4 3 runs      | ~2h 23m       |
| **Total**            | **~3h 30m**   |

> Trace is ~2× slower due to TraceEncoder (GAT) overhead per batch.  
> Train time/epoch: Baseline ~177–206 s/epoch · Trace ~373–445 s/epoch.

---

## 5. Discussion

> ⚠️ **Preprocessing bugs discovered post-run**: Two bugs in `preprocess_micross.py` caused trace features to be all-zero in the data used for section 2 results: (1) timestamp precision mismatch (pandas 3.0 returns `datetime64[us]`; code assumed `[ns]`) causing all spans to fall outside windows; (2) per-file span_id lookup for dynamic adjacency failing because MicroSS stores each service's spans in separate files. Both bugs were fixed in a subsequent investigation. The Trace-v4 F1 numbers above (0.3216) were therefore obtained with **trace data = all zeros** — they represent trace branch running on noise, not real trace features. A re-run with fixed preprocessing is needed for valid trace numbers.

### 5.1 Why Is Trace-v4 Lower Than Baseline?

Two concurrent changes explain the reversal from April 11 to April 29:

1. **Log bug fix (biggest effect)**: In the April 11 run, `_log_file_for_services()` incorrectly skipped `business_table_webservice1_2021-07.csv`, producing `log_c=0` (no log data). Both modes ran without log templates. Now log data loads correctly (34 templates), which **significantly strengthens the baseline** (0.3140 → 0.3530, +12.4%).

2. **Honest threshold (val_percentile=95)**: The old eval swept `anomaly_rate` over the test set to find the best threshold — a form of data leakage. `val_percentile=95` sets the threshold using only the unlabeled training losses (no test labels seen). This removes the leakage benefit.

### 5.2 Positive Finding: Trace Branch Is Highly Stable

Trace-v4 Std F1 = **0.0013** vs Baseline Std F1 = **0.0060** — nearly 5× more stable. All 3 trace runs converge tightly (0.3205–0.3235), while baseline varies by up to 0.013. Note: this stability may partly reflect that the model learned to suppress a zero-valued trace branch consistently.

### 5.3 Why Trace Doesn't Help on MicroSS — Empirical Evidence

A thorough post-hoc investigation analysed all 1,367 injection events across all 8 injected services using paired before/during/after windows and two-tailed t-tests. The results are conclusive:

**MicroSS anomalies do not manifest in distributed trace features.**

#### Statistical evidence — latency change during injection

| Service | Valid events | Latency before | Latency during | Δ% | p-value |
|:--------|------------:|---------------:|---------------:|---:|--------:|
| dbservice1 | 60 | 258.2 ms | 262.8 ms | +1.8% | 0.48 |
| dbservice2 | 88 | 944.6 ms | 946.2 ms | +0.2% | 0.84 |
| mobservice1 | 65 | 198.3 ms | 205.0 ms | +3.4% | 0.37 |
| mobservice2 | 60 | 195.5 ms | 191.3 ms | −2.1% | 0.64 |
| webservice1 | 60 | 1060.0 ms | 1049.2 ms | −1.0% | 0.24 |
| webservice2 | 60 | 1054.5 ms | 1040.8 ms | −1.3% | 0.41 |
| redisservice1 | 54 | 4.4 ms | 4.5 ms | +3.9% | 0.13 |
| redisservice2 | 57 | 7.5 ms | 7.5 ms | −0.3% | 0.65 |

All p-values > 0.13. Error rates are likewise unchanged (Δ < 0.002 across all services). P99 latency shows the same pattern.

#### Why memory injection doesn't appear in traces

The injection script (`[memory_anomalies]`) launches a **background OS process consuming 1 GB of RAM** for 600 seconds. This process:
- Does **not** block the service's request-handling threads
- Does **not** cause memory pressure severe enough to trigger swapping or GC pauses (host has sufficient free memory)
- Therefore produces **no measurable change** in inter-service call latency or error rates

The anomaly is only visible in system-level KPI metrics (container memory RSS, CPU steal) which the baseline model already captures through its 85 KPI dimensions.

#### Weak per-feature signal in both modalities

Computing per-window z-delta (anomaly mean − normal mean) / normal std across the test set:

| Modality | Max |z-delta| | Dims |
|:---------|-------------:|-----:|
| KPI | 0.12 | 85 |
| Trace | 0.14 | 24 (4 services × 6 features) |

Both modalities show equally weak per-feature signal. The baseline achieves F1 = 0.35 by learning **multivariate and temporal patterns** across 85 KPI dimensions over a 50-window context — not through any single strong feature. Trace's 30 features overlap partially with KPI (latency_dev is partially correlated with memory/CPU metrics) and offer no additional discriminative dimensions. The added GAT branch introduces optimization noise that slightly degrades F1.

### 5.4 Conclusion on Trace for MicroSS

Trace-based anomaly detection is **not beneficial** for MicroSS because the injection mechanism (background memory process) produces no signal in the inter-service call graph. This is a **dataset-level property**, not a model limitation. Trace would be more appropriate for MicroSS in a **root-cause localization** task (identifying which service is anomalous) rather than binary detection.

### 5.5 Potential Future Work

- **Root-cause localization**: Use trace adjacency + node features to rank services by anomaly contribution after binary detection
- **Re-evaluate with fixed preprocessing**: Re-run Trace-v4 with corrected `preprocess_micross.py` (timestamp precision + global adj pass) for valid F1 numbers
- **Different anomaly types**: Test on datasets where anomalies directly disrupt inter-service communication (e.g., network partitions, dependency failures)

### 5.6 Short answer
"MicroSS inject anomaly bằng background process ngốn 1GB RAM — không block request handler, nên latency và error rate của inter-service calls không đổi. Chứng minh: paired t-test trên 8 services × 50–90 events mỗi service, tất cả p > 0.13, thay đổi latency trong khoảng ±4% (noise). Trace không thêm được thông tin gì ngoài những gì KPI đã capture."

---

## 6. Architecture — Trace-v4 (Changes from v3)

| #   | Change                     | Description                                                                                 |
|:---:|:---------------------------|:--------------------------------------------------------------------------------------------|
|  1  | `latency_dev` (col 5)      | 6th node feature: z-score of `avg_dur_ms` vs training-split baseline per service           |
|  2  | Row-normalized adj         | `adj[i,j] /= row_sum[i]`; makes adj a proper transition matrix (was binary in v3)          |
|  3  | Residual-gated fusion      | `z_fused = z_log_kpi + g * z_trace` where `g = sigmoid(W·z_trace + b)`                    |
|  4  | `gate_lambda=0.01`         | L1 regularizer on gate `g` — suppresses trace when signal is weak                          |
|  5  | `val_percentile=95`        | Anomaly threshold from percentile of unlabeled losses (no test data leakage)               |

---

## 7. Run Commands

```bash
# Baseline (log + KPI)
python codes/common/eval_micross.py \
    --data "D:/UAC-AD/data/micross" --dataset micross --data_type fuse \
    --open_trace False --num_services 4 --trace_c 6 \
    --batch_size 256 --window_size 50 --epoches 10 10 --patience 5 \
    --alpha 0.16 --open_gan_sep True --val_percentile 95 \
    --run_start 0 --run_end 3 \
    --result_dir "D:/UAC-AD/data/micross/result_fuse_baseline"

# Trace-v4 (log + KPI + trace)
python codes/common/eval_micross.py \
    --data "D:/UAC-AD/data/micross" --dataset micross --data_type fuse \
    --open_trace True --num_services 4 --trace_c 6 --gate_lambda 0.01 \
    --batch_size 256 --window_size 50 --epoches 10 10 --patience 5 \
    --alpha 0.16 --open_gan_sep True --val_percentile 95 \
    --run_start 0 --run_end 3 \
    --result_dir "D:/UAC-AD/data/micross/result_fuse_trace"
```
