# Tiền xử lý Dataset RCAEval OnlineBoutique (RE3-OB) — `preprocess_rcaeval_re3_ob.py`

---

## 1. Tổng quan Dataset

**RE3-OB** (RCAEval Online Boutique, fault code-defect) là benchmark phát hiện bất thường đa phương thức trên hệ thống microservice OnlineBoutique (Google). Khác với RE2-OB (fault hạ tầng), RE3-OB inject **lỗi phần mềm** trực tiếp vào logic service — các lỗi này không làm thay đổi latency hay topology call-graph, nên trace-only detector không thể phát hiện được.

| Tham số | Giá trị |
|:--------|--------:|
| Số scenarios | 10 (4 service × tập con f1–f5) |
| Runs per scenario | 3 |
| Tổng experiments | 30 |
| Thời lượng mỗi run | ~24 phút (1441 timestep × 1 giây) |
| Thời điểm inject fault | ~giây 720 (giữa run) |
| Fault types | f1, f2, f3, f4, f5 (code-defect) |
| Services bị inject | adservice (f3,f4,f5), cartservice (f1), currencyservice (f1), emailservice (f1–f5) |
| Services được monitor | 11 (trong simple_metrics.csv) |

---

## 2. Cấu trúc dữ liệu đầu vào

```
D:/RE3-OB/RE3-OB/
  {service}_{fault}/          # vd. adservice_f3/, emailservice_f1/
    1/, 2/, 3/                # 3 run độc lập
      simple_metrics.csv      # time(Unix sec), các cột metric @ 1 giây/timestep
      logs.csv                # time, timestamp(ns), container_name, message, pod_name, node_name
      traces.csv              # time, traceID, spanID, serviceName, ...,
                              # startTimeMillis, startTime, duration, statusCode, parentSpanID
      inject_time.txt         # Unix timestamp (giây) thời điểm inject fault
```

**Schema các file chính:**

| File | Cột quan trọng | Ghi chú |
|:-----|:--------------|:--------|
| `simple_metrics.csv` | `time` (Unix sec), các cột metric | 1 giây/timestep; số cột thay đổi giữa các run → align về union canonical |
| `logs.csv` | `timestamp` (ns), `container_name`, `message` | Không có cột `log_template` → dùng Drain3 để extract template |
| `traces.csv` | `serviceName`, `startTimeMillis`, `duration`, `statusCode`, `parentSpanID`, `spanID` | Timestamp tính bằng millisecond |
| `inject_time.txt` | Unix timestamp (giây) | Thời điểm bắt đầu inject fault |

---

## 3. Tổng quan Pipeline Xử lý

```
D:/RE3-OB/RE3-OB/
  {service}_{fault}/1/, 2/, 3/
          │
          ▼
  RCAEvalRE3OBPreprocessor.run()
          │
          ├── Bước 1 ──► Xây canonical metric columns
          │                   Scan header ALL simple_metrics.csv (30 runs)
          │                   → Union → N canonical cols
          │
          ├── Bước 2 ──► Drain3 two-pass log template extraction
          │    ├── Pass 1 (fit):     feed ALL messages từ 30 runs qua Drain3
          │    └── Pass 2 (extract): map message → template đã stabilise
          │                         Fallback: container_name (nếu chưa cài drain3)
          │
          ├── Bước 3 ──► Với mỗi experiment (30 runs):
          │    ├── load_inject_time()      → inject_time (Unix sec)
          │    ├── load_kpi()              → timestamps[T], kpis[T,N]
          │    ├── build_log_features()    → logs[T][list[str]]
          │    ├── build_trace_features()  → node_feats[T,11,6], adj[T,11,11]
          │    ├── build_labels()          → labels[T]  (0/1)
          │    └── phân tách → (normal_samples, anomaly_samples)
          │
          ├── Bước 4 ──► Gộp tất cả normal samples
          │                   Shuffle → 80% unlabel + 20% train
          │
          └── Bước 5 ──► Build 5 test file theo fault type
                              normal_pool + anomaly_{fault} → shuffle
                              Đảm bảo anomaly_rate ≤ 20%
```

---

## 4. Chi tiết từng bước

### Bước 1 — Xây Canonical Metric Columns

```
Vấn đề: Số cột trong simple_metrics.csv thay đổi giữa các run
  → Không thể concat trực tiếp

Giải pháp:
  1. Scan header của cả 30 × simple_metrics.csv (chỉ đọc header, không load data)
  2. Lấy UNION các cột metric (bỏ cột 'time')
  3. Sắp xếp theo alphabet → N canonical cols
  4. Khi load mỗi run: reindex về canonical cols, cột thiếu → fill 0

Kết quả: kpis [T, N] đồng nhất cho mọi run
```

### Bước 2 — Drain3 Two-Pass Log Template Extraction

```
logs.csv của RE3-OB không có cột 'log_template' (khác với 24/90 run trong RE2-OB).
Tất cả run phải extract template từ raw 'message'.

Quy trình hai pass:
  Pass 1 (fit): đưa TẤT CẢ message từ 30 run qua Drain3 TemplateMiner
                → ổn định cluster boundary trước khi extract
  Pass 2 (extract): xử lý lại từng run, map message → template
                    Format token: "{container_name}|{drain3_template}"

Cấu hình Drain3:
  drain_depth = 4, drain_sim_th = 0.5, drain_max_children = 100
  parametrize_numeric_tokens = True

Fallback (chưa cài drain3):
  Token = container_name only  (~11 giá trị unique)
  Cài: pip install drain3

Gán vào time bucket:
  ts_sec = log["timestamp"] // 1_000_000_000   # nanoseconds → seconds
  wi = searchsorted(timestamps, ts_sec, side="right") - 1
  logs_per_ts[wi].append(token)

Timestep rỗng → fill ["padding"]
```

### Bước 3a — Load KPI

```python
df = pd.read_csv(exp_dir / "simple_metrics.csv")
timestamps = pd.to_numeric(df.iloc[:, 0], errors="coerce").values.astype(int64)
kpis = df.iloc[:, 1:].reindex(columns=canonical_cols, fill_value=0.0).values  # [T, N]
```

### Bước 3b — Xây Trace Features (TRACE_C = 6)

```
Layout node feature [T, 11, 6]:
  col 0: call_count   — số span per service per timestep, normalize theo global max
  col 1: avg_dur_ms   — mean(duration), normalize theo global max
  col 2: max_dur_ms   — max(duration), normalize theo global max
  col 3: error_rate   — fraction(statusCode ∉ {"0", "0.0", "nan"}) ∈ [0,1]
  col 4: root_rate    — fraction(parentSpanID ∈ tập root) ∈ [0,1]
  col 5: latency_dev  — z-score của avg_dur_ms so với baseline trước fault per service
                        = (avg_dur - mean_pre) / (std_pre + 1e-6)
                        Dương = chậm hơn bình thường, Âm = nhanh hơn

Tính latency_dev:
  pre_fault_idx = searchsorted(timestamps, inject_time, side="left")
  baseline_avg  = node_feats[:pre_fault_idx, :, 1].mean(axis=0)   # [11]
  baseline_std  = node_feats[:pre_fault_idx, :, 1].std(axis=0) + 1e-6
  node_feats[:, :, 5] = (node_feats[:, :, 1] - baseline_avg) / baseline_std

Adjacency matrix [T, 11, 11]:
  adj[t, parent_si, child_si] += 1  (parent span → child span, cùng time bucket)
  → Row-normalize: adj[t] /= adj[t].sum(axis=1, keepdims=True)

Alias tên service: "redis-cart" → "redis"
Service không có trong danh sách SERVICES → bỏ qua
```

### Bước 3c — Xây Labels

```python
labels = (timestamps >= inject_time).astype(int32)
# label = 1 từ thời điểm inject fault đến cuối run
```

### Bước 3d — Phân tách Samples

```
Mỗi experiment (1 run) tạo ra:
  - normal_samples:  các row TRƯỚC inject_time → label = 0 (dùng cho unlabel/train)
  - anomaly_samples: các row TỪ inject_time   → label = 1 (dùng cho test)
```

### Bước 4 — Chia Normal Samples

```
Pool = toàn bộ normal_samples từ 30 experiments
Shuffle (random_seed=42)
→ 80% → unlabel.pkl
→ 20% → train.pkl
```

### Bước 5 — Build Test Files (5 fault types)

```
Với mỗi fault_type trong {f1, f2, f3, f4, f5}:
  anomaly_pool = anomaly_samples từ experiments có fault_type này

  normal_pool  = random sample từ normal_samples sao cho:
                 anomaly_timesteps / total_timesteps ≤ 20%

  test_{fault}.pkl = shuffle(normal_pool + anomaly_pool)

Lưu ý: không phải service nào cũng có đủ 5 fault type.
  f1: cartservice, currencyservice, emailservice      (3 service × 3 run = 9 experiments)
  f2: emailservice                                    (1 service × 3 run = 3 experiments)
  f3: adservice, emailservice                         (2 service × 3 run = 6 experiments)
  f4: adservice, emailservice                         (2 service × 3 run = 6 experiments)
  f5: adservice, emailservice                         (2 service × 3 run = 6 experiments)
```

---

## 5. Dữ liệu đầu ra

```
data/rcaeval_re3_ob/
  unlabel.pkl    ← 80% normal samples
  train.pkl      ← 20% normal samples
  test_f1.pkl    ← shuffle(normal_pool + anomaly_f1), ≤20% anomaly
  test_f2.pkl
  test_f3.pkl
  test_f4.pkl
  test_f5.pkl
  meta.pkl       ← metadata
```

**Format mỗi sample (1 timestep = 1 sample):**

```python
{
  "label":               int,                 # 0 = normal, 1 = anomaly
  "kpis":                np.float32[N],       # N giá trị metric canonical
  "logs":                list[str],           # template strings (cho FeatureExtractor)
  "seqs":                list[str],           # như logs (compatibility)
  "log_features":        np.float32[1],       # placeholder, overwrite bởi semantics.py
  "metric_name":         list[str],           # N tên cột metric
  "trace_node_features": np.float32[11, 6],   # [11 service, 6 features]
  "trace_adj":           np.float32[11, 11],  # adjacency matrix row-normalize
}
```

**meta.pkl:**

```python
{
  "num_services": 11,
  "service2idx":  {...},   # 11 service → index
  "trace_c":      6,
  "kpi_c":        N,       # số canonical metric cols
  "log_c":        1,       # placeholder; được cập nhật bởi semantics.py lúc runtime
  "metric_names": [...],   # N tên metric
  "fault_types":  ["f1", "f2", "f3", "f4", "f5"],
  "services":     ["adservice", "cartservice", ..., "shippingservice"],
}
```

---

## 6. Điểm khác biệt so với RE2-OB

| Khía cạnh | RE2-OB | RE3-OB |
|:----------|:-------|:-------|
| Loại fault | Hạ tầng (cpu, delay, disk, loss, mem, socket) | Code-defect (f1–f5) |
| Số scenarios | 30 (5 service × 6 fault) | 10 (4 service × tập con fault) |
| Tổng experiments | 90 | 30 |
| Extract log | Có cột `log_template` trong 24/90 run; fallback container_name | Không có `log_template` → Drain3 two-pass trên raw `message` |
| Test files | 6 (theo fault type) | 5 (f1–f5) |
| Tín hiệu trace | Discriminative mạnh với network/resource fault | Không discriminative (lỗi code không đổi latency/topology) |

---

## 7. Lệnh chạy

```bash
cd D:/UAC-AD/codes

python common/preprocess_rcaeval_re3_ob.py \
    --data_root D:/RE3-OB/RE3-OB \
    --output_dir ../data/rcaeval_re3_ob \
    --anomaly_rate 0.20 \
    --unlabel_ratio 0.80
```

**Các tham số:**

| Tham số | Mặc định | Mô tả |
|:--------|:---------|:------|
| `--data_root` | (bắt buộc) | Thư mục gốc của dataset RE3-OB |
| `--output_dir` | `../../data/rcaeval_re3_ob` | Thư mục đầu ra |
| `--anomaly_rate` | 0.20 | Tỉ lệ anomaly tối đa trong mỗi test file |
| `--unlabel_ratio` | 0.80 | Tỉ lệ normal data dành cho unlabel.pkl |
| `--random_seed` | 42 | Seed ngẫu nhiên để tái tạo kết quả |

**Sau khi preprocess, chạy eval:**

```bash
# Trace eval (residual-gated, tự động bật khi open_trace=True)
python common/eval_per_scenario_rcaeval_re3_ob.py \
    --data data/rcaeval_re3_ob --dataset rcaeval_re3_ob --data_type fuse \
    --open_trace True --trace_c 6 --batch_size 128 --window_size 30 \
    --epoches 5 5 --patience 3 --gate_lambda 0.01 \
    --result_dir data/rcaeval_re3_ob/result_per_scenario_fuse_trace

# Baseline eval (chỉ log + metric)
python common/eval_per_scenario_rcaeval_re3_ob.py \
    --data data/rcaeval_re3_ob --dataset rcaeval_re3_ob --data_type fuse \
    --open_trace False --batch_size 128 --window_size 30 \
    --epoches 5 5 --patience 3 \
    --result_dir data/rcaeval_re3_ob/result_per_scenario_fuse_baseline
```
