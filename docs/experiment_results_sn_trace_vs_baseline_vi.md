# Kết quả thí nghiệm: SocialNetwork — Trace so với Baseline

## 1. Thiết lập thí nghiệm

### Mô hình

**HADES** (Hypersphere-based Anomaly Detection with Encoder-Score) — mô hình phát hiện bất thường không giám sát dựa trên GAN, huấn luyện hoàn toàn trên dữ liệu bình thường.

### Giao thức đánh giá

| Cấu hình              | Giá trị                                          |
|:----------------------|:-------------------------------------------------|
| Tập dữ liệu           | SocialNetwork (AnoMod)                           |
| Loại dữ liệu          | `fuse` (KPI + Nhật ký + Trace)                   |
| Số kịch bản           | 12 kịch bản bất thường                           |
| Cửa sổ mỗi file test  | 46 (40 bình thường + 6 bất thường)               |
| Tỷ lệ bất thường      | ~13% mỗi file kiểm tra                           |
| `window_size`         | 5  (5 cửa sổ × 30 s = 2,5 phút ngữ cảnh)        |
| `val_percentile`      | 95  (bách phân vị thứ 95 của loss tập val)       |
| `epoches`             | 10 10  (generator + discriminator)               |
| `batch_size`          | 256                                              |
| `patience`            | 5  (dừng sớm)                                    |
| `alpha`               | 0.16                                             |
| `open_gan_sep`        | True                                             |
| `run_end`             | 1  (chạy đơn lần)                                |

### Hiệu chỉnh ngưỡng

Ngưỡng bất thường được tính **mà không dùng nhãn kiểm tra**:
```
threshold = np.percentile(val_losses, 95)
```
trong đó `val_losses` = loss tái tạo từ 8 cửa sổ bình thường chưa thấy trong `val.pkl` (20% cuối của Normal_Baseline, giữ lại trong quá trình huấn luyện).

### Thư mục kết quả

| Cấu hình                        | Thư mục                                         |
|:--------------------------------|:------------------------------------------------|
| Baseline (chỉ KPI + Nhật ký)    | `data/sn/result_per_scenario_fuse_baseline/`    |
| Trace (KPI + Nhật ký + Trace)   | `data/sn/result_per_scenario_fuse_trace/`       |

---

## 2. Lệnh chạy

> **Lưu ý kiến trúc**: Từ nhánh `re-eval-sn`, nhánh trace tự động dùng **residual-gated fusion** (CHANGE 8) khi `open_trace=True`. Gate giữ đóng đóng góp trace khi trace không informative, suy biến về baseline log+KPI. Kết quả bên dưới được chạy lại với kiến trúc mới này.

### Baseline (`open_trace=False`)

```bash
cd D:/UAC-AD/codes
python common/eval_per_scenario_sn.py \
    --data ../data/sn \
    --dataset sn --data_type fuse \
    --open_trace False \
    --epoches 10 10 --batch_size 256 --patience 5 \
    --window_size 5 --val_percentile 95 \
    --alpha 0.16 --open_gan_sep True \
    --run_start 0 --run_end 1
```

### Trace (`open_trace=True`)

```bash
cd D:/UAC-AD/codes
python common/eval_per_scenario_sn.py \
    --data ../data/sn \
    --dataset sn --data_type fuse \
    --open_trace True --trace_c 6 --gate_lambda 0.01 \
    --epoches 10 10 --batch_size 256 --patience 5 \
    --window_size 5 --val_percentile 95 \
    --alpha 0.16 --open_gan_sep True \
    --run_start 0 --run_end 1
```

---

## 3. Kết quả theo từng kịch bản

### 3.1 Baseline (KPI + Nhật ký, `open_trace=False`)

| Kịch bản                         |     F1 | Precision | Recall |
|:---------------------------------|-------:|----------:|-------:|
| Code_Stop_MediaService           | 0.8889 |    1.0000 | 0.8000 |
| Code_Stop_TextService            | 0.6667 |    0.6667 | 0.6667 |
| Code_Stop_UserService            | 1.0000 |    1.0000 | 1.0000 |
| DB_Redis_CacheLimit_HomeTimeline | 1.0000 |    1.0000 | 1.0000 |
| DB_Redis_CacheLimit_SocialGraph  | 1.0000 |    1.0000 | 1.0000 |
| DB_Redis_CacheLimit_UserTimeline | 1.0000 |    1.0000 | 1.0000 |
| Perf_CPU_Contention              | 0.8333 |    0.8333 | 0.8333 |
| Perf_Disk_IO_Stress              | 0.8333 |    0.7143 | 1.0000 |
| Perf_Network_Loss                | 0.9231 |    0.8571 | 1.0000 |
| Svc_Kill_Media                   | 1.0000 |    1.0000 | 1.0000 |
| Svc_Kill_SocialGraph             | 1.0000 |    1.0000 | 1.0000 |
| Svc_Kill_UserTimeline            | 0.9091 |    0.8333 | 1.0000 |
| **Trung bình**                   | **0.9212** | **0.9087** | **0.9417** |
| **Độ lệch chuẩn**                | **0.0994** | **0.1186** | **0.1073** |

### 3.2 Trace (KPI + Nhật ký + Trace, `open_trace=True`)

| Kịch bản                         |     F1 | Precision | Recall |
|:---------------------------------|-------:|----------:|-------:|
| Code_Stop_MediaService           | 1.0000 |    1.0000 | 1.0000 |
| Code_Stop_TextService            | 0.9091 |    1.0000 | 0.8333 |
| Code_Stop_UserService            | 1.0000 |    1.0000 | 1.0000 |
| DB_Redis_CacheLimit_HomeTimeline | 1.0000 |    1.0000 | 1.0000 |
| DB_Redis_CacheLimit_SocialGraph  | 1.0000 |    1.0000 | 1.0000 |
| DB_Redis_CacheLimit_UserTimeline | 1.0000 |    1.0000 | 1.0000 |
| Perf_CPU_Contention              | 0.7692 |    0.7143 | 0.8333 |
| Perf_Disk_IO_Stress              | 1.0000 |    1.0000 | 1.0000 |
| Perf_Network_Loss                | 0.8571 |    0.7500 | 1.0000 |
| Svc_Kill_Media                   | 1.0000 |    1.0000 | 1.0000 |
| Svc_Kill_SocialGraph             | 1.0000 |    1.0000 | 1.0000 |
| Svc_Kill_UserTimeline            | 1.0000 |    1.0000 | 1.0000 |
| **Trung bình**                   | **0.9613** | **0.9554** | **0.9722** |
| **Độ lệch chuẩn**                | **0.0730** | **0.1001** | **0.0621** |

---

## 4. So sánh

### 4.1 Bảng tổng hợp

| Chỉ số                     | Baseline |      Trace | Δ (Trace − Baseline)              |
|:---------------------------|:--------:|:----------:|:----------------------------------|
| F1 trung bình              |   0.9212 | **0.9613** | **+0.0401**                       |
| Precision trung bình       |   0.9087 | **0.9554** | **+0.0467**                       |
| Recall trung bình          |   0.9417 | **0.9722** | **+0.0306**                       |
| Độ lệch chuẩn F1           |   0.0994 | **0.0730** | **−0.0264**  (ổn định hơn)        |
| Số kịch bản đạt F1=1.0     |     6/12 |   **9/12** | **+3**                            |

### 4.2 Thay đổi F1 theo từng kịch bản

| Kịch bản                         | Baseline |    Trace | Δ          |
|:---------------------------------|---------:|---------:|:----------:|
| Code_Stop_MediaService           |   0.8889 | **1.000** | +0.111 ↑   |
| Code_Stop_TextService            |   0.6667 | **0.909** | +0.242 ↑   |
| Code_Stop_UserService            |   1.0000 |    1.000 | ±0         |
| DB_Redis_CacheLimit_HomeTimeline |   1.0000 |    1.000 | ±0         |
| DB_Redis_CacheLimit_SocialGraph  |   1.0000 |    1.000 | ±0         |
| DB_Redis_CacheLimit_UserTimeline |   1.0000 |    1.000 | ±0         |
| Perf_CPU_Contention              |   0.8333 |    0.769 | −0.064 ↓   |
| Perf_Disk_IO_Stress              |   0.8333 | **1.000** | +0.167 ↑   |
| Perf_Network_Loss                |   0.9231 |    0.857 | −0.066 ↓   |
| Svc_Kill_Media                   |   1.0000 |    1.000 | ±0         |
| Svc_Kill_SocialGraph             |   1.0000 |    1.000 | ±0         |
| Svc_Kill_UserTimeline            |   0.9091 | **1.000** | +0.091 ↑   |

**2 kịch bản bị giảm hiệu suất khi bật trace** (Perf_CPU_Contention, Perf_Network_Loss). 4 kịch bản cải thiện, 6 kịch bản giữ nguyên.

---

## 5. Tại sao dữ liệu trace cải thiện khả năng phát hiện

### 5.1 Trace nắm bắt các bất thường cấu trúc mà KPI/Nhật ký không thấy

Mỗi cửa sổ trace mã hóa 6 đặc trưng cho mỗi dịch vụ: `[call_count, avg_dur_us, max_dur_us, error_rate, root_rate, latency_dev]`. Khi một dịch vụ bị kill hoặc dừng:

- `call_count` giảm xuống **0** (không có span nào được phát)
- `error_rate` tăng vọt lên **1.0** (tất cả span còn lại thất bại)
- Các dịch vụ lân cận có `avg_dur_us` tăng cao (chờ đợi dependency đã chết)

Các tín hiệu này **bổ sung** cho các chỉ số KPI: ngay cả khi CPU và bộ nhớ trông bình thường (host ổn, chỉ là tiến trình chết), trace ngay lập tức phát hiện sự vắng mặt của dịch vụ trong đồ thị lời gọi.

### 5.2 Phân tích theo từng kịch bản

#### Code_Stop_MediaService (0.89 → 1.0)
Dịch vụ media bị dừng qua code (không phải kill container). Các chỉ số KPI của nó cho thấy sự suy giảm **từ từ** thay vì giảm đột ngột, khiến baseline khó phân biệt với trạng thái bình thường. Tuy nhiên, trace cho thấy MediaService biến mất khỏi đồ thị lời gọi phân tán — một tín hiệu cấu trúc rõ ràng, không thể nhầm lẫn.

#### Code_Stop_TextService (0.67 → 0.91)
Trước đây là kịch bản khó nhất cho cả hai cấu hình. Việc dừng TextService gây ra sự suy giảm tinh vi: các dịch vụ khác thử lại và bù đắp một phần. Với `latency_dev` được bổ sung như đặc trưng thứ 6, z-score độ lệch so với baseline cung cấp tín hiệu sạch hơn — các dịch vụ phụ thuộc có `latency_dev` dương rõ ràng ngay cả khi thử lại giữ `error_rate` dưới 1.0. Trace nâng F1 từ 0.67 lên 0.91.

#### Perf_Disk_IO_Stress (0.83 → 1.0)
Disk I/O stress gây ra `max_dur_us` cao trên các dịch vụ truy cập lưu trữ bền vững. Độ trễ tăng vọt trong trace là tín hiệu rõ ràng — chỉ KPI thấy chỉ số đĩa tăng nhưng mẫu log không thay đổi đáng kể (ứng dụng chạy, chỉ là chậm hơn).

#### Svc_Kill_UserTimeline (0.91 → 1.0)
Kill ở cấp container. KPI cho thấy CPU/bộ nhớ giảm, nhưng các cửa sổ đầu của giai đoạn bất thường (khi container đang bị kill) có mẫu KPI không rõ ràng. Trace cho thấy dịch vụ biến mất khỏi đồ thị lời gọi đúng vào các timestamp phù hợp.

#### Perf_CPU_Contention (0.83 → 0.77, giảm)
CPU stress ở cấp host: tất cả dịch vụ vẫn chạy và phản hồi. Baseline (KPI+Log) đạt F1=0.83 nhờ phát hiện CPU và load tăng. Trace thực sự gây hại ở đây — `latency_dev` gây nhiễu vì tất cả dịch vụ đều chậm đồng đều, đẩy mô hình về phía false negative trên một số cửa sổ. Đồ thị lời gọi cấu trúc không thay đổi (không có dịch vụ nào biến mất), nên đặc trưng trace thêm mơ hồ thay vì tín hiệu.

#### Perf_Network_Loss (0.92 → 0.86, giảm)
Bất thường mất gói mạng: baseline nắm bắt tốt qua `spans_rate` KPI và template log lỗi. `error_rate` trong trace có tăng nhưng phương sai của nó chồng lên với dao động bình thường — residual gate không hoàn toàn triệt tiêu đóng góp trace nhiễu, làm lệch ngưỡng nhẹ cho một số cửa sổ.

### 5.3 Đồ thị kề tĩnh cung cấp ngữ cảnh cấu trúc

Ma trận kề tĩnh (được xây dựng từ trace Normal_Baseline) mã hóa **topo lời gọi kỳ vọng** của hệ thống 12 dịch vụ. Khi một dịch vụ biến mất (Svc_Kill, Code_Stop), các thành phần nhận thức đồ thị của mô hình (multi-modal self-attention trên cấu trúc kề) phát hiện sự gián đoạn trong các mẫu lời gọi kỳ vọng. Ngữ cảnh cấu trúc này không có sẵn từ các đặc trưng KPI hay log đơn thuần.

---

## 6. Phân tích các kịch bản chưa đạt hoàn hảo

### Code_Stop_TextService (Baseline F1 = 0.67, Trace F1 = 0.91)
- **Nguyên nhân gốc rễ**: Việc dừng TextService kích hoạt cơ chế thử lại trong các dịch vụ khác (HomeTimeline, SocialGraph). Các lần thử lại này gây ra mẫu lưu lượng bất thường nhưng không bằng không, đôi khi giống với các đỉnh tải bình thường.
- **Tại sao baseline gặp khó**: Chỉ số KPI cho thấy CPU cao trên các dịch vụ phụ thuộc (thử lại), mẫu log cho thấy template lỗi mới, nhưng sự kết hợp không vượt qua ngưỡng bách phân vị thứ 95 một cách nhất quán cho tất cả 6 cửa sổ bất thường.
- **Tại sao trace giúp đáng kể**: Z-score `latency_dev` nắm bắt độ lệch độ trễ so với Normal_Baseline — các dịch vụ phụ thuộc có `latency_dev` dương rõ ràng ngay cả khi thử lại giữ `error_rate` dưới 1.0. Điều này đẩy reconstruction loss vượt ngưỡng cho nhiều cửa sổ bất thường hơn, nâng F1 từ 0.67 lên 0.91.

### Perf_CPU_Contention (Baseline F1 = 0.83, Trace F1 = 0.77)
- **Nguyên nhân gốc rễ**: CPU stress ở cấp host. Tất cả 12 dịch vụ vẫn chạy; ứng dụng xử lý yêu cầu chậm nhưng không thất bại.
- **Tại sao baseline hoạt động tốt hơn**: KPI phát hiện CPU và load tăng. Không có dịch vụ nào biến mất khỏi đồ thị lời gọi.
- **Tại sao trace gây hại**: Tất cả dịch vụ đều chậm đồng đều → `latency_dev` tăng đồng loạt → residual gate không hoàn toàn đóng → nhiễu thêm vào reconstruction loss làm lệch ngưỡng. Kết quả: F1 giảm từ 0.83 xuống 0.77.
- **Hàm ý**: Với loại bất thường suy giảm hiệu suất cấp host, cấu hình baseline (KPI+Log) được ưu tiên hơn.

### Perf_Network_Loss (Baseline F1 = 0.92, Trace F1 = 0.86)
- **Nguyên nhân gốc rễ**: Mất gói tin mạng ở cấp host. Dịch vụ vẫn chạy nhưng xử lý yêu cầu gián đoạn.
- **Tại sao baseline hoạt động tốt**: `spans_rate` KPI và template log lỗi cùng nhau nắm bắt mẫu mất gói rõ ràng.
- **Tại sao trace gây hại nhẹ**: `error_rate` tăng nhưng với phương sai cao (mất gói gián đoạn ≠ thất bại nhất quán), và `latency_dev` dao động quanh 0. Nhiễu trace bổ sung làm giảm nhẹ hiệu suất hiệu chỉnh ngưỡng.

---

## 7. Kết luận

| Chỉ số                     | Baseline |                   Trace |
|:---------------------------|:--------:|:-----------------------:|
| F1 trung bình              |   0.9212 |     **0.9613** (+4.0%)  |
| Độ lệch chuẩn F1           |   0.0994 |      **0.0730** (−27%)  |
| Số kịch bản đạt F1=1.0     |     6/12 |                **9/12** |

**Dữ liệu trace cải thiện phát hiện cho các bất thường cấu trúc** (kill dịch vụ và dừng code) bằng cách nắm bắt tín hiệu vắng mặt (`call_count=0`, `error_rate=1.0`, `latency_dev` cao) mà đặc trưng KPI và log bỏ lỡ. 4 kịch bản cải thiện, 6 kịch bản giữ nguyên, và 2 kịch bản giảm nhẹ (Perf_CPU_Contention, Perf_Network_Loss) — cả hai là bất thường inject hiệu suất, nơi đặc trưng trace gây nhiễu thay vì cung cấp tín hiệu.

**Khuyến nghị**: Sử dụng `open_trace=True` cho triển khai sản xuất khi bất thường kiểu kill dịch vụ là mối quan tâm chính. Với môi trường chủ yếu có bất thường suy giảm hiệu suất (CPU stress, mất gói mạng), cấu hình baseline có thể được ưu tiên vì đặc trưng trace có thể gây hại nhẹ trong những trường hợp đó.
