# HADES + Trace — Luồng mô hình (Model Architecture Flow)

> Tài liệu này mô tả kiến trúc và luồng dữ liệu của mô hình HADES được mở rộng với nhánh Trace (GAT Structure Autoencoder).
> **Phiên bản**: `fuse_v3.py` với 5 thay đổi kiến trúc (dựa trên TraceDAE).

---

## 1. Tổng quan kiến trúc dự án

```
UAC-AD/
├── codes/
│   ├── run.py                          ← Điểm vào chính
│   ├── run_sequential.py               ← Chạy tuần tự (tránh CUDA OOM)
│   ├── common/
│   │   ├── data_loads.py               ← Load & window data → DataLoader
│   │   ├── semantics.py                ← Trích xuất log features (Word2Vec/template)
│   │   ├── utils.py                    ← Tiện ích chung (seed, dump results...)
│   │   └── preprocess_micross.py       ← Build pkl từ raw MicroSS CSV
│   └── models/
│       ├── basev3.py                   ← Vòng lặp train/eval (BaseModel)
│       ├── fuse_v3.py                  ← Model đa phương thức (log+metric+trace)
│       ├── log_model_v3.py             ← Log encoder (Transformer)
│       ├── kpi_model_v3.py             ← Metric encoder (Transformer)
│       ├── trace_model_v3.py           ← Trace encoder (GAT) + TraceModel
│       └── utils.py                    ← Các module dùng chung (Attention, ...)
└── data/
    └── micross/
        ├── train.pkl / unlabel.pkl / test.pkl
        └── meta.pkl
```

---

## 2. Luồng dữ liệu đầu vào

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  INPUT  [B, W, *]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  log_features        [B, W, log_c]          ← log template features
  kpi_features        [B, W, kpi_c]          ← KPI metrics
  trace_node_features [B, W, N, trace_c]     ← MicroSS: service node features
  trace_adj           [B, W, N, N]           ← MicroSS: service call graph (STG)
  unmatched_kpi       [B, W, kpi_c]          ← shuffled KPI từ windows khác

  B = batch size | W = window size (50) | N = num_services (4)
  H = hidden_size (32)
```

---

## 3. Generator — `MultiModel.forward()`

### 3.1 MultiEncoder — [CHANGE 1] Trace tách khỏi Self-Attention

> **Thay đổi**: Trước đây trace được đưa vào Self-Attention cùng log+KPI → `[B,W,3H]`.
> Giờ Self-Attention **chỉ** áp dụng trên log+KPI. Trace encoder chạy **riêng** → `ZV`.
> Lý do: self-attention trên 3 modalities pha loãng quan hệ log↔KPI; đúng với thiết kế TraceDAE
> (structural info nên guide **decoder**, không phải encoder attention).

```
  ┌──────────────────────────── MultiEncoder ─────────────────────────────┐
  │                                                                        │
  │  log_features ──► LogEncoder (4-layer Transformer) ──► log_re [B,W,H] │
  │                                                                        │
  │  kpi_features ──► KpiEncoder (Transformer)          ──► kpi_re [B,W,H] │
  │                                                                        │
  │  cat([kpi_re ‖ log_re]) ──► Self-Attention (2H, n_heads=2)            │
  │                                  └──► fused_modal [B, W, 2H]          │
  │                                                                        │
  │  trace_nodes ──► TraceEncoder (2-layer GAT)                            │
  │  trace_adj        reshape [B*W, N, C] → z [B*W, N, H]                │
  │                   mean_pool over N → ZV [B, W, H]     (riêng biệt)   │
  └────────────────────────────────────────────────────────────────────────┘

  Returns: (fused_kpi, fused_log, fused_modal [B,W,2H], ZV [B,W,H])
```

> **Không có trace** (`open_trace=False`): ZV = None, fused_modal vẫn là `[B,W,2H]`.

---

### 3.2 Reconstruction — [CHANGE 2] ZV inject vào Decoder

> **Thay đổi**: Decoder nhận `cat([fused_modal, ZV])` thay vì chỉ `fused_modal`.
> Lý do: Terinspirasi từ TraceDAE (Eq.10) — structural embedding ZV nên guide **reconstruction**,
> không phải fusion attention. Giống `X_hat = ZV · ZA^T` trong TraceDAE.

```
  fused_modal [B,W,2H] ──┐
                          ├──► cat → [B,W,3H] ──► fuse_decoder (Linear 3H→kpi_c+log_c)
  ZV [B,W,H] ────────────┘                                │
                                               fused_out [B, W, kpi_c+log_c]
                                              ┌────────────┴────────────┐
                                         kpi_out [B,W,kpi_c]    log_out [B,W,log_c]

  kpi_dis = L1(kpi_out, kpi_features).mean(dim=-1)    [B, W]
  log_dis = L1(log_out, log_features).mean(dim=-1)    [B, W]
```

> **Không có trace** (ZV=None): fuse_decoder = Linear(2H → kpi_c+log_c) — decoder input = 2H.

---

### 3.3 Trace Structure Autoencoder & adj_hat — [CHANGE 3]

> **Thay đổi**: `adj_hat` (adjacency matrix tái tạo) được return ra ngoài `MultiModel`
> để Discriminator có thể dùng làm "fake trace adjacency".

```
  trace_nodes ──► TraceEncoder (GAT) ──► ZV [B*W, N, H]
                                              │
                         A_hat = sigmoid(ZV · ZVᵀ)  [B*W, N, N]
                                              │
              trace_dis = BCE(A_hat, trace_adj).mean([N,N])  [B, W]

  adj_hat_4d = A_hat.reshape(B, W, N, N)   ← CHANGE 3: trả ra MultiModel output
```

---

### 3.4 Fusion Loss — [CHANGE 4] Learnable trace_alpha

> **Thay đổi**: `trace_weight` (float cố định, cần tuning tay) → `trace_alpha` (learnable parameter).
> Khởi tạo: `nn.Parameter(tensor(-2.2))` → `sigmoid(-2.2) ≈ 0.10` (khớp với α=0.1 trong TraceDAE).
> α được học từ dữ liệu, tự cân bằng giữa (log+KPI) reconstruction và trace structural loss.

```
  log_d   = log_dis  × expand_anomaly_gap(log_dis)
  kpi_d   = kpi_dis  × expand_anomaly_gap(kpi_dis)
  trace_d = trace_dis × expand_anomaly_gap(trace_dis)

  α = sigmoid(trace_alpha)   ← học được, khởi tạo ≈ 0.10

  log_kpi_loss = log_d + kpi_d + narrow_modal_gap(|log_d − kpi_d|)

  fusion_loss = (1 − α) × log_kpi_loss + α × trace_d    [B, W]
                └─────────────────────────────────────────────────────┘
                         Anomaly Score (dùng để đánh giá)
```

---

### 3.5 Contrastive Loss (Unmatched pairs)

```
  contrastive = max(0,  L1(kpi_features, kpi_out)
                      + unmatch_k
                      − L1(unmatched_kpi, kpi_out_unmatched))
```

---

## 4. Discriminator — `MultiDiscriminator.get_loss()` — [CHANGE 5]

> **Thay đổi (bug fix quan trọng)**: Trước đây cả REAL và FAKE pass đều dùng `trace_adj` thật
> → `trace_re == trace_re_fake` → CE loss yêu cầu cùng 1 vector phải là 1 VÀ 0 cùng lúc → gradient mâu thuẫn.
>
> Bây giờ FAKE pass dùng `adj_hat` (adjacency tái tạo từ Structure AE) → `trace_re_fake ≠ trace_re` → loss có ý nghĩa.
> Tương tự như log_out/kpi_out là "fake" cho log/KPI, thì `adj_hat` là "fake" cho trace.

```
  MultiEncoder_low (lightweight, 1-layer GAT):
  ┌──────────────────────────────────────────────────────────────────────┐
  │  REAL:  encoder_low(log_x,   kpi_x,   trace_nodes, trace_adj)        │
  │         → log_re, kpi_re, trace_re              [B, W, H]            │
  │                                                                       │
  │  FAKE:  encoder_low(log_out, kpi_out, trace_nodes, adj_hat)  ← CHANGE│
  │         → log_re_fake, kpi_re_fake, trace_re_fake                    │
  │         (adj_hat từ Structure AE → trace_re_fake ≠ trace_re  ✅)     │
  └──────────────────────────────────────────────────────────────────────┘

  Discriminator loss (cập nhật Discriminator weights):
  disc_loss = CE(pred_kpi,       real=1) + CE(pred_kpi_fake,   fake=0)
            + CE(pred_log,       real=1) + CE(pred_log_fake,   fake=0)
            + CE(pred_trace,     real=1) + CE(pred_trace_fake, fake=0)

  Deceive loss (cập nhật Generator weights):
  deceive_loss = MSE(kpi_re, kpi_re_fake)
               + MSE(log_re, log_re_fake)
               + MSE(trace_re, trace_re_fake)
```

---

## 5. Training Loop (mỗi epoch)

```
  for batch in unlabel_loader:

    ① Generator step:
        res = model(batch)
        loss_G = res["loss"]
                + λ₁ × discriminator.get_loss(batch, res)["deceive_loss"]
        loss_G.backward() → optimizer_G.step()

    ② Discriminator step:
        loss_D = discriminator.get_loss(batch, res)["loss"]
        loss_D.backward() → optimizer_D.step()

    ③ Evaluation (end of epoch):
        score = fusion_loss [B, W]
        → point_adjustment(pred, gt)  ← nếu phát hiện bất kỳ điểm nào
                                         trong segment → cả segment = detected
        → F1 / Recall / Precision
```

---

## 6. Routing theo `data_type`

```
  data_type = "fuse"  →  MultiModel   (log + KPI + trace nếu open_trace=True)
  data_type = "log"   →  LogModel     (log only)
  data_type = "kpi"   →  KpiModel     (KPI only, bỏ qua open_trace)
```

> ⚠️ Chỉ `data_type=fuse` mới kích hoạt nhánh trace.

---

## 7. So sánh kiến trúc: Trước vs Sau (fuse_v3.py)

| # | Thành phần | Trước (cũ) | Sau (mới) |
|---|-----------|------------|-----------|
| 1 | **Self-Attention** | cat([log‖kpi‖trace]) → 3H | cat([log‖kpi]) → 2H, trace **tách riêng** |
| 2 | **Decoder input** | fused_modal [B,W,2H hoặc 3H] | cat([fused_modal, ZV]) → [B,W,3H] |
| 3 | **adj_hat** | không dùng | return ra ngoài MultiModel để Discriminator dùng |
| 4 | **trace_weight** | float cố định (hyperparameter) | `trace_alpha = nn.Parameter(−2.2)`, learned |
| 5 | **Discriminator FAKE** | dùng `trace_adj` (thật) → contradiction | dùng `adj_hat` (tái tạo) → valid loss |

---

## 8. Sơ đồ tổng thể End-to-End (sau khi cải tiến)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  INPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  log_x [B,W,log_c]  ──┐
  kpi_x [B,W,kpi_c]  ──┼──► MultiEncoder ──► fused_modal [B,W,2H]
                        │    (Self-Attn      │
  trace [B,W,N,C]    ──┘     log+KPI only)  └──► ZV [B,W,H]  (GAT)
  adj   [B,W,N,N]    ──────────────────────────────│
                                                    │
                        cat([fused_modal, ZV]) [B,W,3H]
                                    │
                             fuse_decoder (Linear)
                                    │
                        kpi_out [B,W,kpi_c] + log_out [B,W,log_c]
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         │                          │                          │
    kpi_dis [B,W]            log_dis [B,W]            trace_dis [B,W]
    L1(kpi_out, kpi_x)       L1(log_out, log_x)       BCE(A_hat, adj)
         │                          │                          │
         └──────────────────────────┴──────────────────────────┘
                                    │
                    α = sigmoid(trace_alpha)  ← learned
                    fusion_loss = (1-α)×(log+kpi) + α×trace    [B,W]
                             = Anomaly Score
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  OUTPUT: F1 / Recall / Precision (với point-adjustment)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
