# Experiment Results — RE2-OB: Baseline vs Trace (add-attributes-trace)

> **Dataset**: RCAEval OnlineBoutique (RE2-OB) — 30 scenarios × 3 runs = 90 experiments  
> **Branch**: `add-attributes-trace` (checked out from `RCAEval-RE2-OB`)  
> **Date**: 2026-04-21  

---

## 1. Experiment Configuration

| Parameter | Value |
|:----------|------:|
| `dataset` | rcaeval_re2_ob |
| `data_type` | fuse |
| `window_size` | 30 |
| `batch_size` | 128 |
| `epoches` | 5 / 5 |
| `patience` | 3 |
| `trace_c` | **6** (`latency_dev` added vs v1=5) |
| `num_services` | 11 |
| `val_percentile` | 95 |
| `open_gan_sep` | True |

**Baseline**: `open_trace=False` — log + KPI only  
**Trace (add-attributes)**: `open_trace=True`, `trace_c=6` — log + KPI + trace (improved architecture)

### Improvements in `add-attributes-trace` branch

| # | Change | File |
|:-:|:-------|:-----|
| 1 | Add `latency_dev` feature (z-score of avg latency vs pre-fault baseline per service) → `TRACE_C: 5 → 6` | `preprocess_rcaeval_re2_ob.py` |
| 2 | Add attribute reconstruction loss: MSE(latency_dev) + BCE(error_rate) into `trace_dis` | `trace_model_v3.py` |
| 3 | Replace learnable `trace_alpha = nn.Parameter(-2.2)` with **variance-based alpha**: `α = var(trace_dis) / (var(log_kpi) + var(trace_dis) + ε)` | `fuse_v3.py` |
| 4 | Add **attribute discriminator**: separate MLP head — `REAL=trace_nodes[:,:,[3,5]]`, `FAKE=feats_hat[:,:,[3,5]]` → `Linear(2,H)→ReLU→Linear(H,1)` | `fuse_v3.py` |

**Motivation**:
- Learnable `trace_alpha` only converges to balance reconstruction errors on normal data — it does not reflect the actual discriminativeness of the trace signal
- `latency_dev` carries a strong signal for network-related fault types (delay, loss) by detecting which service is slower than its pre-fault baseline
- Variance-based alpha automatically drops to 0 when trace is non-discriminative, preventing noise injection into the fused anomaly score
- Attribute discriminator forces the Generator to produce realistic `error_rate` and `latency_dev` values, strengthening the training signal for anomalous node attributes

---

## 2. Per-Scenario Results

### 2.1 Trace (open_trace=True, trace_c=6)

| Fault    | F1         | Precision  | Recall     |
|:------   |---:        |----------: |-------:    |
| cpu      | 0.7153     | 0.7376     | 0.6943     |
| delay    | 0.8508     | **0.9962** | 0.7424     |
| disk     | **0.9843** | **1.0000** | 0.9691     |
| loss     | 0.7797     | 0.8811     | 0.6993     |
| mem      | 0.8288     | 0.8214     | 0.8363     |
| socket   | 0.6021     | 0.6362     | 0.5714     |
| **Mean** | **0.7935** | **0.8454** | **0.7521** |
| **Std**  | 0.1182     | 0.1316     | 0.1245     |

### 2.2 Baseline (open_trace=False)

| Fault    | F1         | Precision  | Recall |
|:------   |---:        |----------: |-------:|
| cpu      | 0.5954     | 0.5825     | 0.6089 |
| delay    | 0.7974     | 0.7870     | 0.8080 |
| disk     | 0.9843     | 1.0000     | 0.9691 |
| loss     | 0.6829     | 0.9630     | 0.5290 |
| mem      | 0.7061     | 0.7748     | 0.6485 |
| socket   | 0.4363     | 0.4316     | 0.4412 |
| **Mean** | **0.7004** | **0.7565** | **0.6675** |
| **Std**  | 0.1689     | 0.1996     | 0.1755 |

### 2.3 Head-to-Head Comparison

| Fault    | Baseline F1 | Trace F1   | **Δ F1**    | **Δ%**     |
|:------   |-----------: |---------:  |---------:   |-------:    |
| cpu      | 0.5954      | 0.7153     | **+0.1199** | +20.1%     |
| delay    | 0.7974      | 0.8508     | **+0.0534** | +6.7%      |
| disk     | 0.9843      | 0.9843     | **=0.0000** | —          |
| loss     | 0.6829      | 0.7797     | **+0.0968** | +14.2%     |
| mem      | 0.7061      | 0.8288     | **+0.1227** | +17.4%     |
| socket   | 0.4363      | 0.6021     | **+0.1658** | +38.0%     |
| **Mean** | **0.7004**  | **0.7935** | **+0.0931** | **+13.3%** |

**Trace wins on 5/6 scenarios, ties on disk.**

---

## 3. Analysis

### 3.1 Why did socket improve the most (+38.0%)?

- Socket exhaustion slows connections → `latency_dev` captures the latency signal even though the call graph adjacency is unchanged
- Attribute discriminator further strengthens this: the Generator must produce `feats_hat[:,:,[3,5]]` that is indistinguishable from ground truth → richer anomaly signal at training time
- Combined effect of attribute reconstruction loss (CHANGE 2) + attribute discriminator (CHANGE 4) on a fault type with no structural change but strong latency deviation

### 3.2 Why did cpu and mem also improve significantly?

- **cpu** (+20.1%): CPU fault causes service slowdowns → `latency_dev` z-score rises consistently; attribute discriminator provides additional gradient pressure on the trace branch
- **mem** (+17.4%): Memory fault increases latency via GC pressure and swap → both `latency_dev` and `error_rate` spike → attribute signals are strong and discriminative

### 3.3 Why did delay improve more than before (+6.7%)?

- Previously delay improved only +1.3% because baseline already captured network delay via KPI/log
- With attribute discriminator, the trace branch's Precision reaches 0.996 (near-perfect) → fewer false positives → better F1 overall
- Trace is now more confident: fewer false alarms while maintaining reasonable recall

### 3.4 Why did disk not improve?

- Disk fault creates very strong signal in KPI (disk I/O metrics) and logs (error messages)
- Both baseline and trace already achieve F1 ≈ 0.984, P = 1.000 — near-perfect performance leaves no room for improvement

### 3.5 How does variance-based alpha + attribute discriminator work together?

```
α = var(trace_dis) / (var(log_kpi_loss) + var(trace_dis) + ε)

- When trace is discriminative (socket/cpu/mem): var(trace_dis) is high → α increases → trace contributes more
- When trace is noisy: var(trace_dis) is low → α → 0 → model relies on log+KPI

Attribute discriminator (separate from α):
- REAL: trace_nodes[:,:,[3,5]].mean(N)  — ground-truth error_rate & latency_dev
- FAKE: feats_hat[:,:,[3,5]].mean(N)    — reconstructed by attribute decoder
- Forces Generator to produce realistic node attributes → stronger trace_dis gradient
```

---

## 4. Runtime

| Job | Time |
|:----|-----:|
| Preprocessing (TRACE_C=6) | ~1 minute |
| Trace eval (6 scenarios × 5 epochs) | ~3.4 hours (04:48 → 08:14) |
| Baseline eval (6 scenarios × 5 epochs) | ~2.1 hours (03:51 → 05:59) |
| **Total** | **~5.6 hours** |

> Trace is ~1.6× slower than baseline due to TraceEncoder (2-layer GAT) + attribute decoder overhead.

---

## 5. Run Commands

```bash
# Preprocessing (creates data/rcaeval_re2_ob/ with TRACE_C=6)
cd D:/UAC-AD/codes
python common/preprocess_rcaeval_re2_ob.py \
    --data_root D:/RE2-OB/RE2-OB \
    --output_dir ../data/rcaeval_re2_ob

# Trace eval (log + metric + trace, TRACE_C=6)
python common/eval_per_scenario_rcaeval_re2_ob.py \
    --data data/rcaeval_re2_ob --dataset rcaeval_re2_ob --data_type fuse \
    --open_trace True --batch_size 128 --window_size 30 \
    --epoches 5 5 --patience 3 --trace_c 6 \
    --result_dir data/rcaeval_re2_ob/result_per_scenario_fuse_trace

# Baseline eval (log + metric only)
python common/eval_per_scenario_rcaeval_re2_ob.py \
    --data data/rcaeval_re2_ob --dataset rcaeval_re2_ob --data_type fuse \
    --open_trace False --batch_size 128 --window_size 30 \
    --epoches 5 5 --patience 3 \
    --result_dir data/rcaeval_re2_ob/result_per_scenario_fuse_baseline
```
