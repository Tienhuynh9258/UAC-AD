# Xử lý Dataset MicroSS — `preprocess_micross.py`

## 1. Cấu trúc dữ liệu đầu vào

```
D:\GAIA-DataSet\MicroSS\
  ├── trace/
  │   └── trace/                  ← 10 file CSV (~7.5 GB tổng)
  │       ├── trace_table_dbservice1_2021-07.csv
  │       ├── trace_table_dbservice2_2021-07.csv
  │       ├── trace_table_redisservice1_2021-07.csv
  │       └── ...
  ├── metric/
  │   └── metric/                 ← 10,817 file CSV (metric × date-split)
  │       ├── dbservice1_..._2021-07-01_2021-07-15.csv
  │       ├── dbservice1_..._2021-07-15_2021-07-31.csv
  │       ├── dbservice1_..._2021-08-01_2021-08-31.csv  ← bị skip (Aug)
  │       └── ...
  ├── business/
  │   └── business/               ← 2 file CSV (business logs)
  │       ├── business_table_2021-08.csv          (22.7 GB → bị skip)
  │       └── business_table_webservice1_2021-07.csv  (1.5 GB → load)
  └── run/
      └── run.zip                 ← anomaly injection records
```

**Schema các file CSV:**

| File     | Các cột chính                                                                              |
|:---------|:-------------------------------------------------------------------------------------------|
| trace    | `timestamp`, `service_name`, `span_id`, `parent_id`, `start_time`, `end_time`, `status_code` |
| metric   | `timestamp` (Unix ms 13 chữ số), `value`                                                   |
| business | `datetime` (YYYY-MM-DD HH:MM:SS), `service`, `message`                                     |
| run      | `datetime`, `service`, `message` (anomaly info nhúng trong text)                           |

---

## 2. Sơ đồ tổng quan luồng xử lý

```
D:\GAIA-DataSet\MicroSS\
  ├── trace/trace/
  ├── metric/metric/
  ├── business/business/
  └── run/run.zip
          │
          ▼
  MicroSSPreprocessor.run()
          │
          ├── Step 1 ──► Scan Time Range
          │                   t_min = 2021-07-01 09:57:00
          │                   t_max = 2021-08-01 00:00:00
          │
          ├── Step 2 ──► Build Service Index
          │                   10 services: dbservice1..webservice2
          │
          ├── Step 3 ──► Build Static Adjacency  [10×10]  (row-normalized)
          │                   30 directed edges (call graph)
          │
          ├── Step 4 ──► Load Anomaly Periods
          │                   1,367 injection periods from run.zip
          │
          ├── Step 5 ──► Build Window List
          │                   44,043 windows × 60s
          │
          ├── Step 6 ──► Stream Trace → node_feats  [W, 10, 5]
          │
          ├── Step 6b ─► Tính latency_dev (col 5) → node_feats  [W, 10, 6]
          │
          ├── Step 7 ──► Build KPI Matrix            [W, 50]
          │
          ├── Step 8 ──► Load Business Logs → win_logs  [W × list[str]]
          │
          └── Step 9 ──► Assemble & Save pkl files
                              train.pkl / unlabel.pkl / test.pkl / meta.pkl
```

---

## 3. Chi tiết từng bước

### Step 1 — Scan Time Range

```
Với mỗi trace CSV:
  pd.read_csv(path, usecols=["timestamp"], chunksize=500K)
  → tìm min/max timestamp

Kết quả:
  t_min = 2021-07-01 09:57:00  (floor theo phút)
  t_max = 2021-08-01 00:00:00

→ Dùng làm filter cho metric và log ở các bước sau
  (chỉ lấy dữ liệu tháng 7, bỏ qua tháng 8)
```

---

### Step 2 — Build Service Index

```
Với mỗi trace CSV:
  pd.read_csv(path, nrows=10,000, usecols=["service_name"])
  → đếm tần suất xuất hiện mỗi service

Top 10 services → sắp xếp alphabetically:
  service2idx = {
    "dbservice1":   0,
    "dbservice2":   1,
    "logservice1":  2,
    "logservice2":  3,
    "mobservice1":  4,
    "mobservice2":  5,
    "redisservice1":6,
    "redisservice2":7,
    "webservice1":  8,
    "webservice2":  9,
  }
```

---

### Step 3 — Build Static Adjacency

```
Pass 1 — xây dựng span_id → service map:
  Đọc 50K rows/file: {span_id → service_name}

Pass 2 — xây dựng call graph:
  Đọc 50K rows/file: với mỗi span có parent_id != "0":
    parent_svc = span_svc[parent_id]
    child_svc  = service_name
    adj[parent_svc_idx, child_svc_idx] = 1.0

Row-normalize (đồng bộ với RE2-OB / RE3-OB):
  row_sum = adj.sum(axis=1, keepdims=True)
  adj = adj / row_sum  (hàng không có cạnh đi ra giữ nguyên 0)

Kết quả: adj [10×10] — ma trận liền kề có hướng, row-normalized
  30 directed edges (ai gọi ai trong hệ thống microservice)

Ví dụ (minh họa):
  webservice → dbservice    (web gọi DB)
  webservice → redisservice (web gọi cache)
  mobservice → logservice   (mobile gọi log)
  ...
```

---

### Step 4 — Load Anomaly Periods

```
Mở run/run.zip → đọc CSV bên trong
  Cột "message" chứa text dạng:
  "... start at 2021-07-01 11:44:26.882752 and lasts 600 seconds ..."

Regex:
  r'start at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[\d.]*) and lasts (\d+) seconds'

→ Parse thành: (start_timestamp, end_timestamp)
→ 1,367 anomaly periods
→ Khoảng đầu tiên: 2021-07-01 11:44:26 → 11:54:26 (10 phút)
```

---

### Step 5 — Build Window List

```
current = t_min
while current < t_max:
    win_starts.append(current)
    current += 60s

→ 44,043 windows
→ win_starts_ns [44043] — timestamp dạng int64 nanoseconds
   (dùng cho np.searchsorted ở các bước sau)
```

---

### Step 6 — Stream Trace → Node Features

```
Input:  trace CSV files (10 files, ~7.5 GB)
Output: node_feats [W=44043, S=10, C=6]

┌─────────────────────────────────────────────────────────────────┐
│  Với mỗi file trace, đọc chunk 500K rows:                       │
│                                                                 │
│  Tính toán:                                                     │
│    duration  = end_time - start_time  (milliseconds)            │
│    is_error  = (status_code != "200")                           │
│    is_root   = (parent_id == "0")                               │
│    si        = service2idx[service_name]                        │
│                                                                 │
│  Gán vào window bằng binary search:                             │
│    wi = searchsorted(win_starts_ns, ts_ns, side="right") - 1    │
│    valid = (wi >= 0) & (wi < W)                                 │
│                                                                 │
│  Tích lũy bằng np.add.at / np.maximum.at:                      │
│    acc_count[wi, si]    += 1                                    │
│    acc_dur_sum[wi, si]  += duration                             │
│    acc_max_dur[wi, si]   = max(acc_max_dur, duration)           │
│    acc_errors[wi, si]   += is_error                             │
│    acc_roots[wi, si]    += is_root                              │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
node_feats [W, S, 5]:  (col 5 được điền 0 ở bước này)
  dim 0 — call_count   (số lần gọi trong window, normalized)
  dim 1 — avg_dur_ms   (thời gian phản hồi trung bình, normalized)
  dim 2 — max_dur_ms   (thời gian phản hồi tối đa, normalized)
  dim 3 — error_rate   (tỉ lệ lỗi status != 200)
  dim 4 — root_rate    (tỉ lệ span là root call)
```

> **Tại sao dùng `np.add.at` thay vì vòng lặp?**
> Dataset có hàng chục triệu spans. Nếu dùng Python loop sẽ mất nhiều giờ.
> `np.add.at` tích lũy trực tiếp vào mảng NumPy, toàn bộ bước này hoàn thành trong ~6 phút.

---

### Step 6b — Tính latency_dev (col 5)

```
Sau khi stream toàn bộ trace, tính độ lệch latency so với baseline bình thường
(70% window đầu tiên = train split).

  split_idx  = int(W * 0.7)                       # = 30,830
  bl_mean[s] = mean(node_feats[:split_idx, s, 1]) # avg_dur trung bình/service
  bl_std[s]  = std (node_feats[:split_idx, s, 1]) # độ lệch chuẩn/service

  node_feats[:, :, 5] = (node_feats[:, :, 1] - bl_mean) / (bl_std + 1e-6)

→ node_feats [W, S, 6]:
  dim 5 — latency_dev  (z-score của avg_dur so với baseline train per service)
           dương = chậm hơn bình thường, âm = nhanh hơn bình thường

Nhất quán với RE2-OB / RE3-OB (dùng window trước khi inject fault làm baseline).
```

---

### Step 7 — Build KPI Matrix

```
Input:  10,817 metric CSV files
Output: kpi_matrix [W=44043, M=50]

┌─────────────────────────────────────────────────────────────────┐
│  _discover_metric_groups()                                      │
│                                                                 │
│  10,817 files có tên dạng:                                      │
│    dbservice1_0.0.0.4_docker_cpu_core_0_2021-07-01_2021-07-15  │
│    dbservice1_0.0.0.4_docker_cpu_core_0_2021-07-15_2021-07-31  │
│    dbservice1_0.0.0.4_docker_cpu_core_0_2021-08-01_2021-08-31  │
│                                           ↑                     │
│  Strip suffix _YYYY-MM-DD_YYYY-MM-DD ────┘                      │
│  → 4,967 unique metrics                                         │
│  → Chọn top 50 (nhiều date-split nhất)                          │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  _file_overlaps_range()  — chỉ load July, skip August           │
│                                                                 │
│  Với mỗi file, đọc suffix _YYYY-MM-DD_YYYY-MM-DD:               │
│    file_start = 2021-08-01  →  file_start >= t_max  → SKIP      │
│    file_start = 2021-07-01  →  overlaps July range  → LOAD      │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Vectorised binning:                                            │
│    ts_ns = timestamp_unix_ms × 1,000,000                        │
│    wi = searchsorted(win_starts_ns, ts_ns) - 1                  │
│    np.add.at(kpi_sum[:, j], wi, value)                          │
│    np.add.at(kpi_cnt[:, j], wi, 1.0)                            │
│                                                                 │
│  kpi_matrix = kpi_sum / kpi_cnt  (0 nếu không có data)         │
└─────────────────────────────────────────────────────────────────┘

Coverage: 83.2%
(~16.8% windows không có metric data → fill 0)
```

---

### Step 8 — Load Business Logs

```
Danh sách file trong business/business/:
  business_table_2021-08.csv          (22.7 GB)  → SKIP (> 5 GB limit)
  business_table_webservice1_2021-07.csv (1.5 GB) → LOAD

┌─────────────────────────────────────────────────────────────────┐
│  Try C engine (chunk 300K rows):                                │
│    ❌ "EOF inside string at row 7,285,312"                       │
│       (embedded newline \n trong quoted field, chunk bị cắt     │
│        đúng giữa field → C engine không xử lý được)            │
│                                                                 │
│  Fallback: Python engine:                                       │
│    ⚠️  Load thành công phần lớn file                            │
│       Dừng khi gặp row bị lỗi gần cuối file                     │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
  Filter: chỉ giữ rows có datetime ∈ [t_min, t_max]
  Sort by datetime
  → 7,051,685 log entries (ts_ns[], msgs[])

              │
              ▼  _build_log_lists() — binary search
  Với mỗi window i:
    lo = searchsorted(log_ts_ns, win_start)
    hi = searchsorted(log_ts_ns, win_start + 60s)
    win_logs[i] = msgs[lo:hi]
                  (hoặc ["padding"] nếu không có log trong window)
```

---

### Step 9 — Assemble & Save

```
Với mỗi window i trong 44,043:
┌─────────────────────────────────────────────────────────────────┐
│  Gán nhãn anomaly:                                              │
│    Kiểm tra [t_start, t_end) có overlap với anomaly period?     │
│    label = 1 nếu có, 0 nếu không                                │
│                                                                 │
│  Tạo sample dict:                                               │
│  {                                                              │
│    "label":               0 or 1,                               │
│    "kpis":                kpi_matrix[i],           shape [50]   │
│    "logs":                win_logs[i],             list[str]    │
│    "trace_node_features": node_feats[i],           shape [10,6] │
│    "trace_adj":           adj_global,              shape [10,10]│
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘

Split 70% / 30%:
  i < 30,830  →  train:  chỉ lấy normal samples (label=0)
  i >= 30,830 →  test:   lấy tất cả

Kết quả:
  train.pkl   — 26,235 normal samples
  unlabel.pkl — (giống train, dùng cho semi-supervised)
  test.pkl    — 13,213 samples (2,064 anomaly = 15.6%)
  meta.pkl    — {num_services:10, kpi_c:50, trace_c:6, window_sec:60, ...}
```

---

## 4. Thống kê kết quả

| Thông số              | Giá trị        |
|:----------------------|---------------:|
| Thời gian xử lý       |        ~8 phút |
| Tổng số windows       |         44,043 |
| Window size           |        60 giây |
| Services              |             10 |
| Metrics (KPI)         |             50 |
| Log entries loaded    |      7,051,685 |
| Anomaly periods       |          1,367 |
| Anomaly rate (tổng)   |          15.1% |
| Anomaly rate (test)   |          15.6% |
| KPI coverage          |          83.2% |

---

## 5. Chạy lại preprocessing

```bash
python preprocess_micross.py \
    --trace_dir   "D:\GAIA-DataSet\MicroSS\trace\trace" \
    --metric_dir  "D:\GAIA-DataSet\MicroSS\metric\metric" \
    --log_dir     "D:\GAIA-DataSet\MicroSS\business\business" \
    --run_dir     "D:\GAIA-DataSet\MicroSS\run" \
    --output_dir  "../../data/micross" \
    --window_sec  60 \
    --max_metrics 50
```

**Sau khi xong, chạy eval:**

```bash
# Baseline (chỉ log + KPI)
python codes/common/eval_micross.py \
    --data data/micross --dataset micross --data_type fuse \
    --open_trace False --val_percentile 95 \
    --result_dir data/micross/result_fuse_baseline

# Trace (log + KPI + trace GAT, residual gate)
python codes/common/eval_micross.py \
    --data data/micross --dataset micross --data_type fuse \
    --open_trace True --num_services 10 --trace_c 6 \
    --gate_lambda 0.01 --val_percentile 95 \
    --result_dir data/micross/result_fuse_trace
```

---

## 6. Sơ đồ dữ liệu đầu ra cho model

```
train.pkl / test.pkl
  {
    block_id_xxxx: {
      "label":               int (0=normal, 1=anomaly)
      "kpis":                float32 [50]        ← từ metric CSV
      "logs":                list[str]            ← từ business log CSV
      "trace_node_features": float32 [10, 6]     ← từ trace CSV
      "trace_adj":           float32 [10, 10]    ← từ trace CSV (static, row-normalized)
    },
    block_id_yyyy: { ... },
    ...
  }

meta.pkl
  {
    "num_services": 10,
    "service2idx":  {name: idx, ...},
    "metric_names": [str × 50],
    "kpi_c":        50,
    "trace_c":      6,   # 6 features: call_count, avg_dur, max_dur, error_rate, root_rate, latency_dev
    "window_sec":   60,
  }
```


