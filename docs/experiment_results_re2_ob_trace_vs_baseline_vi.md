# Kết quả thí nghiệm — RE2-OB: Baseline vs Trace (add-attributes-trace)

> **Dataset**: RCAEval OnlineBoutique (RE2-OB) — 30 scenarios × 3 runs = 90 experiments  
> **Branch**: `add-attributes-trace` (checkout từ `RCAEval-RE2-OB`)  
> **Ngày**: 2026-04-21  

---

## 1. Cấu hình thí nghiệm

| Tham số | Giá trị |
|:--------|--------:|
| `dataset` | rcaeval_re2_ob |
| `data_type` | fuse |
| `window_size` | 30 |
| `batch_size` | 128 |
| `epoches` | 5 / 5 |
| `patience` | 3 |
| `trace_c` | **6** (thêm `latency_dev` so với v1=5) |
| `num_services` | 11 |
| `val_percentile` | 95 |
| `open_gan_sep` | True |

**Baseline**: `open_trace=False` — chỉ dùng log + KPI  
**Trace (add-attributes)**: `open_trace=True`, `trace_c=6` — log + KPI + trace (kiến trúc cải tiến)

### Các cải tiến trong branch `add-attributes-trace`

| # | Thay đổi | File |
|:-:|:---------|:-----|
| 1 | Thêm feature `latency_dev` (z-score độ trễ so với baseline trước fault) → `TRACE_C: 5 → 6` | `preprocess_rcaeval_re2_ob.py` |
| 2 | Thêm attribute reconstruction loss: MSE(latency_dev) + BCE(error_rate) vào `trace_dis` | `trace_model_v3.py` |
| 3 | Thay `trace_alpha = nn.Parameter(-2.2)` bằng **variance-based alpha**: `α = var(trace_dis) / (var(log_kpi) + var(trace_dis) + ε)` | `fuse_v3.py` |
| 4 | Thêm **attribute discriminator**: head MLP riêng — `REAL=trace_nodes[:,:,[3,5]]`, `FAKE=feats_hat[:,:,[3,5]]` → `Linear(2,H)→ReLU→Linear(H,1)` | `fuse_v3.py` |

**Lý do cải tiến**:
- `trace_alpha` học được (learnable) chỉ converge để cân bằng reconstruction error trên normal data — không phản ánh discriminativeness thực sự của trace signal
- `latency_dev` mang signal mạnh cho các fault type liên quan đến network (delay, loss) vì z-score phát hiện service nào chậm hơn bình thường
- Variance-based alpha tự động giảm về 0 khi trace không discriminative, tránh noise injection
- Attribute discriminator buộc Generator sinh ra `error_rate` và `latency_dev` thực tế hơn, tăng cường training signal cho node attributes bất thường

---

## 2. Kết quả per-scenario

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

### 2.3 So sánh trực tiếp

| Fault    | Baseline F1 | Trace F1   | **Δ F1**    | **Δ%**     |
|:------   |-----------: |---------:  |---------:   |-------:    |
| cpu      | 0.5954      | 0.7153     | **+0.1199** | +20.1%     |
| delay    | 0.7974      | 0.8508     | **+0.0534** | +6.7%      |
| disk     | 0.9843      | 0.9843     | **=0.0000** | —          |
| loss     | 0.6829      | 0.7797     | **+0.0968** | +14.2%     |
| mem      | 0.7061      | 0.8288     | **+0.1227** | +17.4%     |
| socket   | 0.4363      | 0.6021     | **+0.1658** | +38.0%     |
| **Mean** | **0.7004**  | **0.7935** | **+0.0931** | **+13.3%** |

**Trace thắng trên 5/6 scenarios, tie ở disk.**

---

## 3. Nhận xét

### 3.1 Tại sao socket cải thiện nhiều nhất (+38.0%)?

- Socket exhaustion làm chậm kết nối → `latency_dev` bắt được signal dù adjacency graph không thay đổi
- Attribute discriminator tăng cường thêm: Generator phải sinh `feats_hat[:,:,[3,5]]` không phân biệt được với ground truth → signal anomaly mạnh hơn khi training
- Hiệu ứng kết hợp của attribute reconstruction loss (CHANGE 2) + attribute discriminator (CHANGE 4) trên fault type không có thay đổi cấu trúc nhưng có latency deviation mạnh

### 3.2 Tại sao cpu và mem cũng cải thiện đáng kể?

- **cpu** (+20.1%): CPU fault gây chậm service → `latency_dev` z-score tăng ổn định; attribute discriminator cung cấp gradient bổ sung cho trace branch
- **mem** (+17.4%): Memory fault tăng latency do GC pressure và swap → cả `latency_dev` lẫn `error_rate` đều spike → signal attribute mạnh và discriminative

### 3.3 Tại sao delay cải thiện nhiều hơn lần trước (+6.7%)?

- Trước đây delay chỉ cải thiện +1.3% vì baseline đã capture network delay qua KPI/log
- Với attribute discriminator, Precision của trace branch đạt 0.996 (gần perfect) → ít false positive hơn → F1 tổng tốt hơn
- Trace giờ tự tin hơn: ít false alarm trong khi vẫn duy trì recall hợp lý

### 3.4 Tại sao disk không cải thiện?

- Disk fault tạo signal rất mạnh trong KPI (disk I/O metrics) và logs (error messages)
- Cả baseline và trace đều đạt F1 ≈ 0.984, P = 1.000 — đã ở mức gần perfect, không còn room để cải thiện

### 3.5 Variance-based alpha và attribute discriminator phối hợp như thế nào?

```
α = var(trace_dis) / (var(log_kpi_loss) + var(trace_dis) + ε)

- Khi trace discriminative (socket/cpu/mem): var(trace_dis) cao → α tăng → trace đóng góp nhiều
- Khi trace noise: var(trace_dis) thấp → α → 0 → model dựa vào log+KPI

Attribute discriminator (độc lập với α):
- REAL: trace_nodes[:,:,[3,5]].mean(N)  — ground-truth error_rate & latency_dev
- FAKE: feats_hat[:,:,[3,5]].mean(N)    — tái tạo bởi attribute decoder
- Buộc Generator sinh node attributes thực tế hơn → gradient trace_dis mạnh hơn
```

---

## 4. Thời gian chạy

| Job | Thời gian |
|:----|----------:|
| Preprocessing (TRACE_C=6) | ~1 phút |
| Trace eval (6 scenarios × 5 epochs) | ~3.4 giờ (04:48 → 08:14) |
| Baseline eval (6 scenarios × 5 epochs) | ~2.1 giờ (03:51 → 05:59) |
| **Tổng** | **~5.6 giờ** |

> Trace chậm hơn baseline ~1.6x do overhead TraceEncoder (2 lớp GAT) + attribute decoder.

---

## 5. Lệnh chạy

```bash
# Preprocessing (tạo data/rcaeval_re2_ob/ với TRACE_C=6)
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
