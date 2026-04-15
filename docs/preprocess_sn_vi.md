# Tiền xử lý: Tập dữ liệu SocialNetwork (AnoMod)

## 1. Tổng quan tập dữ liệu

Tập dữ liệu SocialNetwork (SN) là một phần của **benchmark AnoMod** dành cho phát hiện bất thường trong kiến trúc microservice đám mây. Dữ liệu được thu thập từ ứng dụng microservice gồm 12 dịch vụ, triển khai trên cụm thực tế. Tập dữ liệu ghi lại ba phương thức:

| Phương thức  | Nguồn                                      | Vị trí         |
|:-------------|:-------------------------------------------|:---------------|
| Chỉ số KPI   | Hệ thống + container + Jaeger spans        | `metric_data/` |
| Nhật ký      | File log theo từng dịch vụ                 | `log_data/`    |
| Trace        | Distributed traces (Jaeger)                | `trace_data/`  |

### Các kịch bản

| Loại                  | Số lượng | Mô tả                                              |
|:----------------------|---------:|:---------------------------------------------------|
| Normal_Baseline       |        1 | ~20 phút lưu lượng ổn định                         |
| Code_Stop_*           |        3 | Tiến trình dịch vụ bị kill thông qua code          |
| DB_Redis_CacheLimit_* |        3 | Giới hạn cache Redis bị inject vào dịch vụ         |
| Perf_CPU_Contention   |        1 | CPU stress inject ở cấp host                       |
| Perf_Disk_IO_Stress   |        1 | Disk I/O stress inject                             |
| Perf_Network_Loss     |        1 | Mất gói tin mạng inject                            |
| Svc_Kill_*            |        3 | Kill dịch vụ ở cấp container                       |

Mỗi kịch bản thu thập dữ liệu thực tế trong khoảng **20–25 phút**.

---

## 2. Kỹ thuật đặc trưng

### 2.1 Phân cửa sổ

Dữ liệu chuỗi thời gian thô được chia thành các **cửa sổ không chồng lấp** dài `window_sec=30` giây. Mỗi cửa sổ trở thành một điểm dữ liệu đầu vào cho mô hình.

**Tại sao chọn 30 giây?**
- Đủ ngắn để nắm bắt các bất thường thoáng qua (kill dịch vụ xuất hiện trong vài giây)
- Đủ dài để tạo ra các thống kê tổng hợp ổn định (tránh nhiễu từ các lần đọc chỉ số riêng lẻ)
- Kịch bản 20 phút tạo ra ~40 cửa sổ mỗi kịch bản, là tối đa khả dụng cho một lần ghi Normal_Baseline

**Bỏ qua giai đoạn khởi động:** `warmup_minutes=5` đầu tiên (10 cửa sổ) của mỗi kịch bản bị loại bỏ để tránh các artifact khởi tạo (dịch vụ ổn định, JVM warm-up, v.v.). Áp dụng cho cả kịch bản bình thường lẫn bất thường.

Kết quả: Normal_Baseline → **40 cửa sổ** sau khi bỏ warmup; hầu hết kịch bản bất thường → **30 cửa sổ** (19,5 phút ÷ 30s = 39 thô, trừ 10 warmup = 29–30).

### 2.2 Đặc trưng KPI (59 chiều)

| Nhóm      | Số lượng | Đặc trưng                                                                                                                                                    |
|:----------|---------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Hệ thống  |       10 | cpu_usage, disk_io_time, disk_read_bytes, disk_usage_pct, disk_write_bytes, load1, memory_usage_pct, network_errors, network_receive_bytes, network_transmit_bytes |
| Container |       48 | 12 dịch vụ × 4 chỉ số: cpu, memory, net_rx, net_tx                                                                                                          |
| Jaeger    |        1 | spans_rate (result="ok", chuẩn hóa theo cửa sổ)                                                                                                             |

Mỗi KPI được tổng hợp theo cửa sổ bằng giá trị trung bình của tất cả mẫu trong khoảng 30 giây đó. Giá trị thiếu (ví dụ: container chưa khởi động) được điền bằng giá trị trung bình cột tính từ các cửa sổ không có NaN.

### 2.3 Đặc trưng nhật ký

Nhật ký được phân tích bằng **Drain3** (streaming template miner), khớp chỉ trên nhật ký Normal_Baseline:
- Template học được: ~458 từ 317.055 tin nhắn log
- Loại đặc trưng: `template_appear` — sự hiện diện/vắng mặt nhị phân của mỗi template trong cửa sổ
- Template mới gặp trong kịch bản bất thường được coi là chưa thấy (ánh xạ vào bucket "unknown" đặc biệt)

**Tại sao khớp Drain3 chỉ trên Normal_Baseline?**  
Để tránh nhiễm bởi các mẫu log bất thường khi xây dựng từ điển. Mô hình cần học cách đánh dấu các template chưa thấy là bất thường, điều này chỉ khả thi nếu từ điển template được xây dựng từ log bình thường.

### 2.4 Đặc trưng trace (tùy chọn, `open_trace=True`)

Với mỗi dịch vụ, một vector đặc trưng 5 chiều được tính toán theo cửa sổ:

```
[call_count, avg_duration_us, max_duration_us, error_rate, root_rate]
```

- `call_count`: số lượng trace span liên quan đến dịch vụ này trong cửa sổ
- `avg_duration_us` / `max_duration_us`: thống kê độ trễ
- `error_rate`: tỷ lệ span có mã trạng thái không phải OK
- `root_rate`: tỷ lệ span là root span (điểm vào)

Một **ma trận kề tĩnh** (12×12) được xây dựng từ trace Normal_Baseline: cạnh (i, j) = 1 nếu dịch vụ i gọi dịch vụ j ít nhất một lần. Đồ thị này cố định cho tất cả các kịch bản — chúng ta giả sử topo đồ thị lời gọi không thay đổi giữa các thí nghiệm.

---

## 3. Chiến lược phân chia

### 3.1 Phân chia Train / Val (Normal_Baseline)

40 cửa sổ Normal_Baseline được phân chia **theo thứ tự thời gian** (không xáo trộn):

```
Normal_Baseline (40 cửa sổ)
├── train.pkl   = 32 cửa sổ đầu (80%)  → huấn luyện mô hình
├── unlabel.pkl = 32 cửa sổ tương tự   → giai đoạn unsupervised/GAN
└── val.pkl     = 8 cửa sổ cuối (20%)  → hiệu chỉnh ngưỡng (chưa thấy trong quá trình huấn luyện)
```

**Tại sao phân chia theo thời gian (không ngẫu nhiên)?**  
- Thứ tự thời gian quan trọng: các cửa sổ cuối của Normal_Baseline là trạng thái bình thường "gần nhất" trước khi inject bất thường
- Xáo trộn sẽ gây nhiễm: các cửa sổ val có thể được nội suy từ các cửa sổ train xung quanh (data leakage)
- Tập val phải đại diện cho hành vi bình thường thực sự chưa thấy để hiệu chỉnh ngưỡng có giá trị

**Tại sao chọn tỷ lệ 80/20?**  
- 32 cửa sổ × 30s = 16 phút dữ liệu huấn luyện: đủ để GAN học các mẫu KPI/log bình thường
- 8 cửa sổ val → với `window_size=5`, tạo ra **3 chuỗi chồng lấp** (cửa sổ 1–5, 2–6, 3–7) → ~3 giá trị loss để tính ngưỡng bách phân vị thứ 95
- Tập val nhỏ hơn (ví dụ: 4 cửa sổ) sẽ không tạo được chuỗi nào với window_size=5

### 3.2 File kiểm tra (theo từng kịch bản)

Với mỗi trong 12 kịch bản bất thường, một file kiểm tra được tạo:

```
test_{tên_kịch_bản}.pkl = 40 cửa sổ bình thường (từ Normal_Baseline)
                        + 6 cửa sổ bất thường (lấy mẫu con từ kịch bản)
                        → xáo trộn (thứ tự ngẫu nhiên)
```

**Tại sao dùng 40 bình thường + 6 bất thường?**
- **Tỷ lệ bất thường ~13%** (6/46): thực tế cho hệ thống sản xuất nơi bất thường hiếm gặp
- Tránh vấn đề tỷ lệ bất thường phồng khi dùng tất cả cửa sổ bất thường (30+ cửa sổ), điều này sẽ tạo ra các cụm bất thường liên tiếp dài trong mô hình chuỗi sau khi xáo trộn, làm phồng recall thông qua `point_adjust`

**Tại sao lấy mẫu con xuống còn 6 cửa sổ bất thường?**  
Các kịch bản bất thường gốc có 30–41 cửa sổ. Dùng tất cả sẽ nâng tỷ lệ bất thường lên ~43%, gây ra các cụm bất thường liên tiếp dài sau khi xáo trộn → `point_adjust` sẽ đánh dấu toàn bộ đoạn chỉ với một lần phát hiện → recall/F1 bị phồng. Bằng cách giới hạn ở `max_anomaly_windows=6`, tỷ lệ giữ ở mức ~13% và bài kiểm tra thực tế hơn.

**Chiến lược lấy mẫu con:** các chỉ số cách đều nhau trên toàn bộ dải cửa sổ của kịch bản:
```python
indices = [int(round(i * (len(sc_samples) - 1) / (n - 1))) for i in range(n)]
```
Điều này bảo toàn tính đa dạng thời gian của các mẫu bất thường (giai đoạn đầu, giữa, cuối của sự kiện bất thường).

**Tại sao xáo trộn?**  
Mô hình nhận một luồng hỗn hợp (như trong sản xuất) và phải chấm điểm từng cửa sổ riêng lẻ. Không xáo trộn, tất cả cửa sổ bất thường sẽ ở cuối — dễ phát hiện theo vị trí.

---

## 4. Hiệu chỉnh ngưỡng (không rò rỉ dữ liệu)

Ngưỡng bất thường được tính **sau khi huấn luyện**, chỉ từ các loss của tập val:

```python
threshold = np.percentile(val_losses, val_percentile=95)
```

Cách này thay thế phương pháp cũ là dò quét `anomaly_rate` trên tập kiểm tra (vốn rò rỉ nhãn thực tế). Với ngưỡng dựa trên tập val:
- Không có thông tin từ tập kiểm tra được dùng để chọn ngưỡng
- Bách phân vị thứ 95 nghĩa là ~5% chuỗi bình thường có thể bị đánh dấu là bất thường (tỷ lệ FP kỳ vọng)
- Áp dụng như một ngưỡng cắt cố định: mỗi chuỗi kiểm tra với `loss > threshold` được dự đoán là bất thường

---

## 5. Cấu trúc đầu ra

```
data/sn/
├── train.pkl              # 32 cửa sổ bình thường (80% đầu của Normal_Baseline)
├── unlabel.pkl            # 32 cửa sổ bình thường (giống train, cho giai đoạn unlabeled của GAN)
├── val.pkl                # 8 cửa sổ bình thường (20% cuối, chưa thấy, để xác định ngưỡng)
├── meta.pkl               # Metadata tập dữ liệu (ma trận kề, chiều đặc trưng, v.v.)
└── scenarios/
    ├── test_Code_Stop_MediaService_20251104_024819.pkl   # 46 cửa sổ, 6 bất thường, tỷ lệ=0.13
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

## 6. Cách sử dụng

```bash
python codes/common/preprocess_sn.py \
    --sn_data_root D:/AnoMod/SN_data \
    --output_dir data/sn \
    --window_sec 30 \
    --warmup_minutes 5 \
    --max_anomaly_windows 6 \
    --seed 42
```

### Các tham số chính

| Tham số                 | Giá trị | Lý do                                                                        |
|:------------------------|--------:|:-----------------------------------------------------------------------------|
| `--window_sec`          |      30 | Độ chi tiết 30 giây: nắm bắt bất thường thoáng qua, tổng hợp ổn định        |
| `--warmup_minutes`      |       5 | Bỏ qua 5 phút đầu (10 cửa sổ) để tránh nhiễu khởi tạo                       |
| `--max_anomaly_windows` |       6 | Giới hạn cửa sổ bất thường mỗi kịch bản → tỷ lệ ~13%, tránh phồng point_adjust |
| `--seed`                |      42 | Tái hiện kết quả: kiểm soát thứ tự xáo trộn của file kiểm tra               |
