# Experiment Results — MicroSS: Baseline vs Trace-v3

> **Dataset**: MicroSS (micross) — 26,235 train / 13,213 test (11,149 normal + 2,064 anomaly, anomaly rate ~15.6%)
> **Date**: 2026-04-11

---

## 1. Cấu hình thí nghiệm

| Tham số | Giá trị |
|---|---|
| `dataset` | micross |
| `data_type` | fuse |
| `window_size` | 50 |
| `hidden_size` | 32 |
| `epoches` | 10 / 10 |
| `batch_size` | 256 |
| `patience` | 5 |
| `alpha` (inter-class distance) | 0.16 |
| `theta` (difference threshold) | 0.15 |
| `open_gan_sep` | True |
| `learning_rate` | 0.001 |
| `num_runs` | 3 (run_times: 0, 1, 2) |

**Baseline**: `open_trace=False` — log + KPI only  
**Trace-v3**: `open_trace=True`, `num_services=4`, `trace_c=5` — log + KPI + trace (kiến trúc mới)

---

## 2. Kết quả tổng hợp

### 2.1 Baseline (log + KPI)

| Run | Best F1 | Recall | Precision | Best Epoch |
|---|---|---|---|---|
| run-0 | 0.3494 | 0.4273 | 0.2956 | 8 |
| run-1 | 0.3394 | 0.4128 | 0.2882 | 9 |
| run-2 | 0.2532 | 0.2951 | 0.2218 | 2 |
| **Mean** | **0.3140** | **0.3784** | **0.2685** | |
| **Std** | **0.0428** | **0.0578** | **0.0325** | |
| **Max** | **0.3494** | **0.4273** | **0.2956** | |

### 2.2 Trace-v3 (log + KPI + trace)

| Run | Best F1 | Recall | Precision | Best Epoch |
|---|---|---|---|---|
| run-0 | 0.3496 | 0.4273 | 0.2958 | 4 |
| run-1 | 0.3386 | 0.4113 | 0.2877 | 6 |
| run-2 | 0.3421 | 0.4167 | 0.2901 | 2 |
| **Mean** | **0.3434** | **0.4184** | **0.2912** | |
| **Std** | **0.0046** | **0.0067** | **0.0034** | |
| **Max** | **0.3496** | **0.4273** | **0.2958** | |

### 2.3 So sánh trực tiếp

| Metric | Baseline | Trace-v3 | Delta |
|---|---|---|---|
| **F1 (mean)** | 0.3140 | **0.3434** | **+0.029 (+9.4%)** |
| **Recall (mean)** | 0.3784 | **0.4184** | **+0.040 (+10.6%)** |
| **Precision (mean)** | 0.2685 | **0.2912** | **+0.023 (+8.5%)** |
| **F1 (max)** | 0.3494 | **0.3496** | +0.0002 |
| **Std F1** | 0.0428 | **0.0046** | **-0.038 (ổn định hơn 9x)** |

![alt text](image.png)
---

## 3. Diễn biến F1 theo epoch

### Baseline

| Epoch | Run-0 | Run-1 | Run-2 |
|---|---|---|---|
| 0 | 0.092 | 0.065 | 0.053 |
| 1 | 0.297 | 0.251 | 0.060 |
| 2 | 0.339 | 0.336 | **0.253** ← best |
| 3 | 0.312 | 0.314 | 0.199 |
| 4 | 0.314 | 0.318 | 0.190 |
| 5 | 0.335 | 0.314 | 0.181 |
| 6 | 0.342 | 0.338 | 0.182 |
| 7 | 0.346 | 0.332 | 0.211 ← early stop |
| 8 | **0.349** ← best | 0.339 | |
| 9 | 0.346 | **0.339** ← best | |

### Trace-v3

| Epoch | Run-0 | Run-1 | Run-2 |
|---|---|---|---|
| 0 | 0.065 | 0.308 | 0.318 |
| 1 | 0.299 | 0.276 | 0.311 |
| 2 | 0.308 | 0.306 | **0.342** ← best |
| 3 | 0.338 | 0.300 | 0.339 |
| 4 | **0.350** ← best | 0.323 | 0.339 |
| 5 | 0.346 | 0.331 | 0.331 |
| 6 | 0.281 | **0.339** ← best | 0.326 |
| 7 | 0.338 | 0.325 | 0.338 ← early stop |
| 8 | 0.349 | 0.333 | |
| 9 | 0.346 | 0.337 | |

---

## 4. Thời gian chạy

| Job | Thời gian |
|---|---|
| Baseline 3 runs | ~1h 41m (06:40 → 08:21) |
| Trace-v3 3 runs | ~3h 20m (08:25 → 11:45) |
| **Tổng** | **~5h 05m** |

> Trace chậm hơn ~2x do overhead của TraceEncoder (GAT) và TraceModel trên mỗi batch.

---

## 5. Nhận xét

### 5.1 Cải thiện chính: độ ổn định, không phải điểm số tuyệt đối

- Mean F1 tăng **+9.4%** — có ý nghĩa thực tế
- **Quan trọng hơn**: Std F1 giảm từ 0.043 xuống 0.005 (**9x ổn định hơn**)
  - Baseline có run-2 bị collapse (F1=0.253, peak rất sớm ở epoch 2 rồi giảm)
  - Trace-v3 không bị vậy — 3 runs hội tụ đều và ổn định

### 5.2 Tại sao trace không cải thiện mạnh hơn?

1. **Anomaly chủ yếu resource-based**: MicroSS inject lỗi dạng CPU spike, memory leak — biểu hiện rõ ở KPI/log nhưng **không thay đổi topology** call graph (service A vẫn gọi B như bình thường dù A đang quá tải)
2. **Binary adjacency mất thông tin**: `trace_adj[i,j] ∈ {0,1}` — mất call frequency, latency, error rate per edge. Structure AE học được topology nhưng không học được "A gọi B timeout liên tục"
3. **trace_c=5 có thể overlap KPI**: 5 node features per service có thể đã được cover bởi 85 KPI metrics
4. **Learnable alpha tự converge về thấp**: Vì trace signal yếu, gradient đẩy `trace_alpha` nhỏ lại → trace đóng góp ít vào anomaly score

### 5.3 Hướng cải thiện tiềm năng

- Dùng **weighted adjacency** (response time per edge) thay vì binary
- Thêm **call latency / error rate** vào `trace_node_features`
- Trace phù hợp hơn cho **root cause localization** hơn là binary anomaly detection

---

## 6. Kiến trúc Trace-v3 (5 thay đổi so với phiên bản cũ)

| # | Thay đổi | Mô tả |
|---|---|---|
| 1 | Self-Attention | Chỉ log+KPI → fused_modal [B,W,2H], trace tách riêng → ZV |
| 2 | Decoder input | cat([fused_modal, ZV]) → 3H (ZV inject vào decoder, lấy cảm hứng TraceDAE Eq.10) |
| 3 | adj_hat return | MultiModel trả adj_hat [B,W,N,N] để Discriminator dùng |
| 4 | Learnable alpha | `trace_alpha = nn.Parameter(-2.2)`, sigmoid(-2.2) ≈ 0.10 |
| 5 | Discriminator FAKE | FAKE pass dùng adj_hat thay vì trace_adj (fix bug gradient mâu thuẫn) |

---

## 7. Lệnh chạy

```bash
# Baseline (log + metric)
python codes/run.py \
    --data "C:/Users/us/Desktop/UAC-AD/.claude/worktrees/data/micross" \
    --dataset micross --data_type fuse \
    --open_trace False \
    --epoches 10 10 --batch_size 256 --patience 5 \
    --alpha 0.16 --open_gan_sep True --run_start 0 --run_end 3 \
    --result_dir "C:/Users/us/Desktop/UAC-AD/.claude/worktrees/data/result_fuse_baseline"

# Trace-v3 (log + metric + trace)
python codes/run.py \
    --data "C:/Users/us/Desktop/UAC-AD/.claude/worktrees/data/micross" \
    --dataset micross --data_type fuse \
    --open_trace True --num_services 4 --trace_c 5 \
    --epoches 10 10 --batch_size 256 --patience 5 \
    --alpha 0.16 --open_gan_sep True --run_start 0 --run_end 3 \
    --result_dir "C:/Users/us/Desktop/UAC-AD/.claude/worktrees/data/result_fuse_trace"
```
