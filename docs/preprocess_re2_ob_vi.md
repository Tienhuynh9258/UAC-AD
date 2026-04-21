# Xử lý Dataset RCAEval OnlineBoutique (RE2-OB) — `preprocess_rcaeval_re2_ob.py`

---

## 1. Tổng quan dataset

**RE2-OB** (RCAEval Online Boutique, infrastructure faults) là benchmark đánh giá phát hiện bất thường đa phương thức trên hệ thống microservice OnlineBoutique (Google).

| Thông số | Giá trị |
|:---------|--------:|
| Số scenarios | 30 (5 services × 6 fault types) |
| Số runs mỗi scenario | 3 |
| Tổng experiments | 90 |
| Thời gian mỗi run | ~24 phút (1441 timesteps × 1 giây) |
| Thời điểm inject fault | ~giây 720 (giữa run) |
| Fault types | cpu, delay, disk, loss, mem, socket |
| Services bị inject | checkoutservice, currencyservice, emailservice, productcatalogservice, recommendationservice |
| Services được monitor | 11 (trong simple_metrics.csv) |

---

## 2. Cấu trúc dữ liệu đầu vào

```
D:/RE2-OB/RE2-OB/
  {service}_{fault}/          # VD: checkoutservice_cpu/
    1/, 2/, 3/                # 3 independent runs
      simple_metrics.csv      # [1441 rows, ~73 cols] — 1-second intervals
      logs.csv                # timestamp(ns), container_name, log_template
      traces.csv              # serviceName, startTimeMillis, duration,
                              # statusCode, parentSpanID, spanID
      cluster_info.json       # template_id → {template, container[]}
      inject_time.txt         # Unix timestamp (seconds) của fault injection
```

**Schema các file chính:**

| File | Cột quan trọng | Ghi chú |
|:-----|:--------------|:--------|
| `simple_metrics.csv` | `time` (Unix sec), 73 metric cols | 1-second interval; số cột thay đổi theo run (69–78) |
| `logs.csv` | `timestamp` (ns), `container_name`, `log_template` | 24/90 runs có cột `log_template`; 66/90 không có → fallback `container_name` |
| `traces.csv` | `serviceName`, `startTimeMillis`, `duration`, `statusCode`, `parentSpanID`, `spanID` | millisecond timestamp |
| `cluster_info.json` | `{template_id: {template, container[]}}` | Map template ID → template text |
| `inject_time.txt` | Unix timestamp (seconds) | Thời điểm bắt đầu fault |

---

## 3. Luồng xử lý tổng quan

```
D:/RE2-OB/RE2-OB/
  {service}_{fault}/1/, 2/, 3/
          │
          ▼
  RCAEvalOBPreprocessor.run()
          │
          ├── Bước 1 ──► Build canonical metric columns
          │                   Scan TẤT CẢ simple_metrics.csv headers
          │                   → Union 73–78 cols → 86 canonical cols
          │
          ├── Bước 2 ──► Với mỗi experiment (90 runs):
          │    ├── load_inject_time()      → inject_time (Unix sec)
          │    ├── load_kpi()              → timestamps[T], kpis[T,86]
          │    ├── build_log_features()    → logs[T][list[str]]
          │    ├── build_trace_features()  → node_feats[T,11,6], adj[T,11,11]
          │    ├── build_labels()          → labels[T]  (0/1)
          │    └── segment → (normal_session, full_session)
          │
          ├── Bước 3 ──► Gộp tất cả normal sessions
          │                   Shuffle → 80% unlabel + 20% train
          │
          └── Bước 4 ──► Tạo 6 test files theo fault type
                              normal_pool + anomaly_{fault} → shuffle
                              Đảm bảo anomaly_rate ≤ 20%
```

---

## 4. Chi tiết từng bước

### Bước 1 — Build Canonical Metric Columns

```
Vấn đề: Số cột trong simple_metrics.csv thay đổi theo run (69–78 cols)
  → Không thể concatenate trực tiếp

Giải pháp:
  1. Scan TẤT CẢ 90 × simple_metrics.csv (chỉ đọc header, không load data)
  2. Lấy UNION của tất cả metric columns (bỏ cột 'time')
  3. → 86 canonical cols (sorted alphabetically)
  4. Khi load mỗi run: reindex về canonical cols, missing cols → fill 0

Kết quả: kpis [T, 86] cho mọi run, shape đồng nhất
```

### Bước 2a — Load KPI

```python
df = pd.read_csv(exp_dir / "simple_metrics.csv")
timestamps = df.iloc[:, 0].values.astype(int64)   # Unix seconds
kpis = df.reindex(columns=canonical_cols).values   # [T, 86], fill 0 nếu thiếu
```

### Bước 2b — Build Log Features

```
Hai format logs.csv trong dataset:
  - Full (24/90 runs): có cột 'log_template' → dùng trực tiếp (~145 unique templates)
  - Minimal (66/90 runs): không có 'log_template' → fallback sang 'container_name'
                           (11 unique values, 1 per service)

Nếu dùng raw 'container_name + message': 86k+ unique strings → FeatureExtractor cực chậm
→ Dùng container_name-only khi không có log_template → vocab ~156 unique strings

Alignment theo time bucket:
  ts_sec = log["timestamp"] // 1_000_000_000   # nanoseconds → seconds
  wi = searchsorted(timestamps, ts_sec, side="right") - 1
  logs_per_ts[wi].append(template_or_container)

Timestep trống → điền ["padding"]
```

### Bước 2c — Build Trace Features (TRACE_C = 6)

```
Node feature layout [T, 11, 6]:
  col 0: call_count   — số spans trong time bucket, normalized bởi global max
  col 1: avg_dur_ms   — mean(duration), normalized
  col 2: max_dur_ms   — max(duration), normalized
  col 3: error_rate   — fraction(statusCode ∉ {"0", "0.0", "nan"}) ∈ [0,1]
  col 4: root_rate    — fraction(parentSpanID ∈ {"", "0", "nan", "None"}) ∈ [0,1]
  col 5: latency_dev  — z-score của avg_dur_ms so với baseline pre-fault per service
                        = (avg_dur - mean_pre) / (std_pre + 1e-6)
                        Dương = chậm hơn bình thường, âm = nhanh hơn

Cách tính latency_dev:
  pre_fault_idx = searchsorted(timestamps, inject_time, side="left")
  baseline_avg  = node_feats[:pre_fault_idx, :, 1].mean(axis=0)   # [11]
  baseline_std  = node_feats[:pre_fault_idx, :, 1].std(axis=0) + 1e-6
  node_feats[:, :, 5] = (node_feats[:, :, 1] - baseline_avg) / baseline_std

Adjacency matrix [T, 11, 11]:
  adj[t, caller_idx, callee_idx] += 1  (parent span → child span)
  → Row-normalize: adj[t] /= adj[t].sum(axis=1, keepdims=True)

Service name alias: "redis-cart" → "redis"
Services không có trong SERVICES list → bỏ qua
```

### Bước 2d — Build Labels

```python
labels = np.where(timestamps >= inject_time, 1, 0)
# label = 1 từ thời điểm inject fault đến hết run
```

### Bước 2e — Segment Sessions

```
Mỗi experiment tạo ra 2 loại session (1 timestep = 1 session):
  - normal_session: rows TRƯỚC inject_time  → label = 0 (dùng cho unlabel/train)
  - full_session:   TẤT CẢ rows             → label = 0/1 (dùng cho test)
```

### Bước 3 — Split Normal Sessions

```
Pool = tất cả normal_sessions từ 90 experiments (~65k timesteps)
Shuffle (random_seed=42)
→ 80% → unlabel.pkl  (52,251 samples)
→ 20% → train.pkl    (13,063 samples)
```

### Bước 4 — Build Test Files (6 fault types)

```
Với mỗi fault_type trong {cpu, delay, disk, loss, mem, socket}:
  anomaly_pool = full_sessions từ 15 experiments có fault_type này
                 (5 services × 3 runs = 15 runs)

  normal_pool  = random sample từ normal_sessions sao cho:
                 anomaly_timesteps / total_timesteps ≤ 20%

  test_{fault}.pkl = shuffle(normal_pool + anomaly_pool)

Số lượng thực tế:
  test_cpu.pkl    : 51,520 samples (10,304 anomaly ≈ 20%)
  test_delay.pkl  : 54,075 samples (10,815 anomaly ≈ 20%)
  test_disk.pkl   : 54,075 samples (10,815 anomaly ≈ 20%)
  test_loss.pkl   : 54,075 samples (10,815 anomaly ≈ 20%)
  test_mem.pkl    : 51,889 samples (10,378 anomaly ≈ 20%)
  test_socket.pkl : 54,075 samples (10,815 anomaly ≈ 20%)
```

---

## 5. Dữ liệu đầu ra

```
data/rcaeval_re2_ob/
  unlabel.pkl     ← 52,251 normal samples
  train.pkl       ← 13,063 normal samples
  test_cpu.pkl    ← 51,520 samples (≤20% anomaly)
  test_delay.pkl  ← 54,075 samples
  test_disk.pkl   ← 54,075 samples
  test_loss.pkl   ← 54,075 samples
  test_mem.pkl    ← 51,889 samples
  test_socket.pkl ← 54,075 samples
  meta.pkl        ← metadata
```

**Format mỗi sample (1 timestep):**

```python
{
  "label":               int,                # 0 = normal, 1 = anomaly
  "kpis":                np.float32[86],     # 86 metric values
  "logs":                list[str],          # template strings (cho FeatureExtractor)
  "seqs":                list[str],          # giống logs (compatibility)
  "log_features":        np.float32[1],      # placeholder, overwrite bởi semantics.py
  "metric_name":         list[str],          # 86 tên metric
  "trace_node_features": np.float32[11, 6],  # [11 services, 6 features]
  "trace_adj":           np.float32[11, 11], # row-normalized adjacency
}
```

**meta.pkl:**

```python
{
  "num_services": 11,
  "trace_c":      6,
  "kpi_c":        86,
  "log_c":        1,
  "metric_names": [...],   # 86 tên metric
  "service2idx":  {...},   # 11 services → index
  "fault_types":  ["cpu", "delay", "disk", "loss", "mem", "socket"],
  "services":     ["adservice", "cartservice", ..., "shippingservice"],
}
```

---

## 6. Lệnh chạy

```bash
cd D:/UAC-AD/codes

python common/preprocess_rcaeval_re2_ob.py \
    --data_root D:/RE2-OB/RE2-OB \
    --output_dir ../data/rcaeval_re2_ob \
    --anomaly_rate 0.20 \
    --unlabel_ratio 0.80
```

**Tham số:**

| Tham số | Mặc định | Mô tả |
|:--------|:---------|:------|
| `--data_root` | (bắt buộc) | Thư mục gốc chứa RE2-OB dataset |
| `--output_dir` | (bắt buộc) | Thư mục output |
| `--anomaly_rate` | 0.20 | Tỉ lệ anomaly tối đa trong test files |
| `--unlabel_ratio` | 0.80 | Tỉ lệ normal data dành cho unlabel.pkl |

**Sau khi preprocessing, chạy evaluation:**

```bash
# Trace eval
python common/eval_per_scenario_rcaeval_re2_ob.py \
    --data data/rcaeval_re2_ob --dataset rcaeval_re2_ob --data_type fuse \
    --open_trace True --trace_c 6 --batch_size 128 --window_size 30 \
    --epoches 5 5 --patience 3 \
    --result_dir data/rcaeval_re2_ob/result_per_scenario_fuse_trace

# Baseline eval
python common/eval_per_scenario_rcaeval_re2_ob.py \
    --data data/rcaeval_re2_ob --dataset rcaeval_re2_ob --data_type fuse \
    --open_trace False --batch_size 128 --window_size 30 \
    --epoches 5 5 --patience 3 \
    --result_dir data/rcaeval_re2_ob/result_per_scenario_fuse_baseline
```
