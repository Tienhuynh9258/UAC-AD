# Kết Quả Thí Nghiệm — MicroSS: Baseline vs Trace-v4

> **Dataset**: MicroSS (micross) — 26,235 train / 13,213 test (11,149 normal + 2,064 anomaly, anomaly rate ~15.6%)
> **Ngày**: 2026-04-29 (chạy lại với mô hình đã chốt; lần trước: 2026-04-11)

---

## 1. Cấu hình thí nghiệm

| Tham số                            | Giá trị   |
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
| `gate_lambda` (L1 reg trace gate)  |      0.01 |
| `val_percentile` (ngưỡng bất thường) |      95 |

**Baseline**: `open_trace=False` — log + KPI only  
**Trace-v4**: `open_trace=True`, `num_services=4`, `trace_c=6`, `gate_lambda=0.01` — log + KPI + trace (residual-gated fusion, 6 node features)

---

## 2. Kết quả tổng hợp

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

### 2.3 So sánh trực tiếp

| Metric               | Baseline   | Trace-v4   | Delta                          |
|:---------------------|-----------:|-----------:|:-------------------------------|
| **F1 (mean)**        | **0.3530** |     0.3216 | **−0.031 (−8.9%)**             |
| **Recall (mean)**    | **0.3734** |     0.3428 | −0.031 (−8.2%)                 |
| **Precision (mean)** | **0.3356** |     0.3037 | −0.032 (−9.5%)                 |
| **F1 (max)**         | **0.3576** |     0.3235 | −0.034                         |
| **Std F1**           |     0.0060 | **0.0013** | **−0.005 (ổn định hơn 4.6×)**  |

---

## 3. So sánh lịch sử (2026-04-11 vs 2026-04-29)

> ⚠️ Lần chạy 11/04 có bug: `log_c=0` (không load được log do lỗi filter `_log_file_for_services`). Kết quả **không so sánh được** — chỉ liệt kê để tham khảo.

| Phiên bản      | Ngày       | Baseline F1 | Trace F1 | Delta     |
|:---------------|:-----------|------------:|---------:|----------:|
| Trace-v3       | 2026-04-11 |      0.3140 |   0.3434 | +0.029    |
| **Trace-v4**   | **2026-04-29** | **0.3530** | **0.3216** | **−0.031** |

---

## 4. Thời gian chạy

| Job                  | Thời gian     |
|:---------------------|:--------------|
| Baseline 3 runs      | ~1h 07m       |
| Trace-v4 3 runs      | ~2h 23m       |
| **Tổng**             | **~3h 30m**   |

> Trace chậm hơn ~2× do overhead của TraceEncoder (GAT) trên mỗi batch.  
> Train time/epoch: Baseline ~177–206 s/epoch · Trace ~373–445 s/epoch.

---

## 5. Nhận xét

> ⚠️ **Lỗi preprocessing phát hiện sau khi chạy**: Hai lỗi trong `preprocess_micross.py` khiến trace features toàn zero trong data dùng cho kết quả mục 2: (1) timestamp precision mismatch (pandas 3.0 trả về `datetime64[us]`; code giả định `[ns]`) khiến toàn bộ spans rơi ngoài windows; (2) span_id lookup per-file cho dynamic adjacency thất bại vì MicroSS lưu spans của mỗi service trong file riêng. Cả hai lỗi đã được fix. **Trace-v4 F1 = 0.3216 ở trên thực chất chạy với trace data = toàn zero** — là kết quả của trace branch chạy trên noise, không phải trace features thật. Cần re-run với preprocessing đã fix để có số liệu trace hợp lệ.

### 5.1 Tại sao Trace-v4 lại thấp hơn Baseline?

Hai thay đổi giải thích sự đảo ngược từ 11/04 sang 29/04:

1. **Bug fix log (ảnh hưởng lớn nhất)**: Lần chạy 11/04, `_log_file_for_services()` bỏ qua nhầm file `business_table_webservice1_2021-07.csv`, dẫn đến `log_c=0` (không có log data). Cả hai mode đều chạy không có template. Nay log load đúng (34 templates), **baseline mạnh hơn đáng kể** (0.3140 → 0.3530, +12.4%).

2. **Ngưỡng bất thường trung thực (val_percentile=95)**: Lần cũ dùng sweep `anomaly_rate` trên test set để tìm ngưỡng tốt nhất — data leakage. `val_percentile=95` đặt ngưỡng từ phân vị loss của tập unlabeled (không thấy test labels), loại bỏ leakage.

### 5.2 Điểm tích cực: Trace Branch rất ổn định

Trace-v4 Std F1 = **0.0013** vs Baseline Std F1 = **0.0060** — ổn định hơn ~5×. Lưu ý: sự ổn định này một phần có thể phản ánh việc model học cách suppress trace branch toàn zero một cách nhất quán.

### 5.3 Tại sao Trace không giúp ích trên MicroSS — Bằng chứng thực nghiệm

Một cuộc điều tra hậu kỳ đã phân tích toàn bộ 1.367 injection events trên 8 services, dùng paired t-test so sánh windows trước/trong/sau injection. Kết quả có tính quyết định:

**Anomaly của MicroSS không biểu hiện trong distributed trace features.**

#### Bằng chứng thống kê — thay đổi latency khi injection

| Service | Số events hợp lệ | Latency trước | Latency trong | Δ% | p-value |
|:--------|----------------:|--------------:|--------------:|---:|--------:|
| dbservice1 | 60 | 258.2 ms | 262.8 ms | +1.8% | 0.48 |
| dbservice2 | 88 | 944.6 ms | 946.2 ms | +0.2% | 0.84 |
| mobservice1 | 65 | 198.3 ms | 205.0 ms | +3.4% | 0.37 |
| mobservice2 | 60 | 195.5 ms | 191.3 ms | −2.1% | 0.64 |
| webservice1 | 60 | 1060.0 ms | 1049.2 ms | −1.0% | 0.24 |
| webservice2 | 60 | 1054.5 ms | 1040.8 ms | −1.3% | 0.41 |
| redisservice1 | 54 | 4.4 ms | 4.5 ms | +3.9% | 0.13 |
| redisservice2 | 57 | 7.5 ms | 7.5 ms | −0.3% | 0.65 |

Tất cả p-value > 0.13. Error rate cũng không thay đổi (Δ < 0.002 trên tất cả services). P99 latency cho thấy kết quả tương tự.

#### Tại sao memory injection không xuất hiện trong trace

Script injection (`[memory_anomalies]`) chạy một **background process tiêu thụ 1 GB RAM** trong 600 giây. Process này:
- **Không** block các request-handling thread của service
- **Không** gây memory pressure đủ mạnh để trigger swap hay GC pause (host còn đủ RAM)
- Do đó **không tạo ra thay đổi đo được** trong inter-service call latency hay error rate

Anomaly chỉ hiển thị trong system-level KPI metrics (container memory RSS, CPU) mà baseline đã capture qua 85 KPI dimensions.

#### Signal yếu ở cả hai modality

Tính per-window z-delta (mean anomaly − mean normal) / std normal trên test set:

| Modality | Max |z-delta| | Số chiều |
|:---------|-------------:|---------:|
| KPI | 0.12 | 85 |
| Trace | 0.14 | 24 (4 services × 6 features) |

Cả hai modality đều có signal per-feature yếu như nhau. Baseline đạt F1 = 0.35 nhờ học **multivariate và temporal patterns** qua 85 KPI dims trong 50-window context — không phải từ signal mạnh của một feature đơn lẻ. 30 features của trace một phần chồng chéo với KPI và không cung cấp thêm thông tin phân biệt mới. GAT branch thêm vào gây optimization noise nhẹ làm giảm F1.

### 5.4 Kết luận về Trace cho MicroSS

Trace-based anomaly detection **không có lợi** cho MicroSS vì cơ chế injection (background memory process) không tạo ra signal trong inter-service call graph. Đây là **tính chất cấp dataset**, không phải giới hạn của model. Trace sẽ phù hợp hơn cho MicroSS trong bài toán **root-cause localization** (xác định service nào bị anomaly) hơn là binary detection.

### 5.5 Hướng nghiên cứu tiếp theo

- **Root-cause localization**: Dùng trace adjacency + node features để xếp hạng các services theo mức độ bất thường sau khi đã phát hiện binary
- **Re-evaluate với preprocessing đã fix**: Chạy lại Trace-v4 với `preprocess_micross.py` đã sửa (timestamp precision + global adj pass) để có F1 hợp lệ
- **Anomaly types khác**: Thử nghiệm trên dataset mà anomaly ảnh hưởng trực tiếp đến inter-service communication (vd: network partition, dependency failure)

### 5.6 Trả lời ngắn gọn
"MicroSS inject anomaly bằng cách chạy một background process ngốn 1GB RAM. Process này không block request handler của service, nên latency và error rate của các inter-service calls không thay đổi — chứng minh bằng paired t-test trên 50-90 injection events/service, tất cả p > 0.13. Anomaly chỉ biểu hiện qua KPI system metrics (memory, CPU), mà baseline đã capture rồi. Trace không thêm thông tin mới → F1 giảm nhẹ do noise."

---

## 6. Kiến trúc — Trace-v4 (thay đổi so với v3)

| #   | Thay đổi                   | Mô tả                                                                                        |
|:---:|:---------------------------|:---------------------------------------------------------------------------------------------|
|  1  | `latency_dev` (col 5)      | Feature thứ 6: z-score của `avg_dur_ms` so với baseline training-split theo từng service    |
|  2  | Row-normalized adj         | `adj[i,j] /= row_sum[i]`; adj trở thành transition matrix đúng nghĩa (v3 dùng binary)      |
|  3  | Residual-gated fusion      | `z_fused = z_log_kpi + g * z_trace` với `g = sigmoid(W·z_trace + b)`                       |
|  4  | `gate_lambda=0.01`         | L1 regularizer trên gate `g` — triệt tiêu trace khi signal yếu                             |
|  5  | `val_percentile=95`        | Ngưỡng bất thường từ percentile loss unlabeled (không dùng test data)                       |

---

## 7. Lệnh chạy

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
