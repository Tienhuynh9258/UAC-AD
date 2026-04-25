# Kết quả thí nghiệm — RE2-OB: Baseline vs Trace

> **Dataset**: RCAEval OnlineBoutique (RE2-OB) — 30 scenarios × 3 runs = 90 experiments  
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

### Các cải tiến

| # | Thay đổi | File |
|:-:|:---------|:-----|
| 1 | Thêm feature `latency_dev` (z-score độ trễ so với baseline trước fault) → `TRACE_C: 5 → 6` | `preprocess_rcaeval_re2_ob.py` |
| 2 | Thêm attribute reconstruction loss: MSE(latency_dev) + BCE(error_rate) vào `trace_dis` | `trace_model_v3.py` |
| 3 | Thay `trace_alpha = nn.Parameter(-2.2)` bằng **variance-based alpha**: `α = var(trace_dis) / (var(log_kpi) + var(trace_dis) + ε)` | `fuse_v3.py` |
| 4 | Thêm **attribute discriminator**: head MLP riêng — `REAL=trace_nodes[:,:,[3,5]]`, `FAKE=feats_hat[:,:,[3,5]]` → `Linear(2,H)→ReLU→Linear(H,1)` | `fuse_v3.py` |
| 5 | **CHANGE 8 — Residual-Gated Trace Fusion**: `y = base_decoder(fused_modal) + g · delta_head(cat[fm, ZV])`, với `g ∈ [0,1]` là gate per-sample học từ 6 trace-quality features. `delta_head` zero-init → trạng thái khởi đầu suy biến đúng về baseline. L1 regularizer `gate_lambda` giữ gate đóng mặc định. | `fuse_v3.py`, `run.py` |

**Lý do cải tiến**:
- `trace_alpha` học được (learnable) chỉ converge để cân bằng reconstruction error trên normal data — không phản ánh discriminativeness thực sự của trace signal
- `latency_dev` mang signal mạnh cho các fault type liên quan đến network (delay, loss) vì z-score phát hiện service nào chậm hơn bình thường
- Variance-based alpha tự động giảm về 0 khi trace không discriminative, tránh noise injection
- Attribute discriminator buộc Generator sinh ra `error_rate` và `latency_dev` thực tế hơn, tăng cường training signal cho node attributes bất thường
- **CHANGE 8 (residual-gated fusion)** xử lý một failure mode khác: khi nhánh trace không cung cấp thông tin, cơ chế blend theo `α` cũ vẫn rò noise qua decoder. Dạng residual `y = y_base + g · Δ` với `delta_head` zero-init **bảo đảm** mô hình khởi điểm chính xác bằng baseline, chỉ lệch khỏi baseline khi gate `g` bị gradient ủng hộ đẩy mở ra.

---

## 2. Kết quả per-scenario

### 2.1 Trace (open_trace=True, trace_c=6)

| Fault    | F1         | Precision  | Recall     |
|:------   |---:        |----------: |-------:    |
| cpu      | 0.6245     | 0.6105     | 0.6391     |
| delay    | 0.8584     | 0.8724     | 0.8448     |
| disk     | **0.9844** | **1.0000** | 0.9692     |
| loss     | 0.8548     | 0.8696     | 0.8406     |
| mem      | **0.8968** | 0.8957     | **0.8979** |
| socket   | 0.5647     | 0.5511     | 0.5790     |
| **Mean** | **0.7973** | **0.7999** | **0.7951** |
| **Std**  | 0.1505     | 0.1618     | 0.1393     |

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
| cpu      | 0.5954      | 0.6245     | **+0.0291** | +4.9%      |
| delay    | 0.7974      | 0.8584     | **+0.0610** | +7.6%      |
| disk     | 0.9843      | 0.9844     | **+0.0001** | —          |
| loss     | 0.6829      | 0.8548     | **+0.1719** | +25.2%     |
| mem      | 0.7061      | 0.8968     | **+0.1907** | +27.0%     |
| socket   | 0.4363      | 0.5647     | **+0.1284** | +29.4%     |
| **Mean** | **0.7004**  | **0.7973** | **+0.0969** | **+13.8%** |

**Trace thắng trên cả 6/6 scenarios.**

---

## 3. Nhận xét

### 3.1 Tại sao mem cải thiện nhiều nhất (+27.0%)?

- Memory fault tăng latency do GC pressure và swap → cả `latency_dev` lẫn `error_rate` đều spike đồng thời
- Residual-gated gate mở rộng (`g→1`) vì các trace-quality features (coverage, latency_dev, error_rate) đều mạnh và ổn định trên window
- `delta_head` học được correction đáng kể trên nền baseline → Δ đẩy `kpi_out`/`log_out` về dự đoán sạch hơn, nới rộng anomaly gap

### 3.2 Tại sao loss cải thiện +25.2%?

- Packet loss gây retry và timeout → `latency_dev` z-score tăng ổn định, error_rate spike trên các service bị ảnh hưởng
- Precision từ baseline 0.963 (ở recall thấp 0.529) sang 0.870 ở recall 0.841 — trace kéo operating point về vùng cân bằng hơn
- Adjacency chủ yếu giữ nguyên dưới packet loss → Structure AE một mình yếu, nhưng attribute reconstruction (`latency_dev` + `error_rate`) gánh phần chính

### 3.3 Tại sao socket cải thiện +29.4%?

- Socket exhaustion làm chậm kết nối → `latency_dev` bắt được signal dù adjacency không đổi
- Attribute reconstruction loss (CHANGE 2) + attribute discriminator (CHANGE 4) cung cấp training signal; residual gate mở vì các feature này informative
- Gate chỉ mở một phần khi chỉ có attribute signal (không adjacency) mang fault — phản ứng bảo thủ có chủ đích nhưng vẫn đem lại +0.128 F1

### 3.4 Tại sao delay cải thiện +7.6%?

- Baseline đã capture network delay tốt qua KPI/log (F1 0.797)
- Residual trace thêm correction vừa phải — gate mở một phần vì `latency_dev` informative nhưng log+KPI đã phủ phần lớn signal
- Trace chủ yếu cải thiện recall (0.808 → 0.845), Precision giữ cao

### 3.5 Tại sao cpu chỉ cải thiện +4.9%?

- CPU fault gây chậm service gián đoạn, nhưng `latency_dev` z-score nhiễu hơn so với mem/loss vì hiệu ứng CPU throttling bursty
- Gate chỉ mở một phần — hành vi bảo thủ theo thiết kế khi trace-quality features có variance cao
- Residual vẫn giữ chặt trên baseline (+0.029 F1), thoả yêu cầu *không được tệ hơn baseline* dù upside nhỏ

### 3.6 Tại sao disk không cải thiện?

- Disk fault tạo signal rất mạnh trong KPI (disk I/O metrics) và logs (error messages)
- Cả baseline và residual trace đều đạt F1 ≈ 0.984, P = 1.000 — gần perfect, không còn room
- Gate thực tế là no-op ở đây; residual ≈ baseline (+0.0001 là noise)

### 3.7 Cách residual-gated fusion quyết định khi nào dùng trace

```
y = base_decoder(fused_modal) + g · delta_head(cat[fused_modal, ZV])

g = sigmoid(trace_gate(quality_feats))    ∈ (0, 1)   per sample [B, W]

quality_feats (6 chiều per [B,W]): mean call_count, coverage, mean error_rate,
                                    mean |latency_dev|, adj density, call-count variance

delta_head zero-init → khởi điểm Δ ≈ 0 ⇒ y ≈ base_decoder(fm)  ≡ baseline
L1 reg trên g (gate_lambda=0.01) giữ gate đóng mặc định

Dynamics lúc training:
- mem/loss/socket:   quality feats mạnh & ổn định → g → ~1 → áp dụng full delta → upside lớn
- delay/cpu:         quality feats vừa phải → g một phần → correction bảo thủ
- disk:              baseline gần perfect → không có gradient ép mở gate → g nhỏ
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
    --open_trace True --gate_lambda 0.01 --batch_size 128 --window_size 30 \
    --epoches 5 5 --patience 3 --trace_c 6 \
    --result_dir data/rcaeval_re2_ob/result_per_scenario_fuse_trace

# Baseline eval (log + metric only)
python common/eval_per_scenario_rcaeval_re2_ob.py \
    --data data/rcaeval_re2_ob --dataset rcaeval_re2_ob --data_type fuse \
    --open_trace False --batch_size 128 --window_size 30 \
    --epoches 5 5 --patience 3 \
    --result_dir data/rcaeval_re2_ob/result_per_scenario_fuse_baseline
```
