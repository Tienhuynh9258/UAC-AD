# HADES + Trace — Luồng mô hình (Model Architecture Flow)

> Tài liệu này mô tả kiến trúc và luồng dữ liệu của mô hình HADES được mở rộng với nhánh Trace (GAT Structure Autoencoder).
> **Phiên bản**: `fuse_v3.py` với 7 thay đổi kiến trúc (dựa trên TraceDAE).

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
│   │   ├── preprocess_XX.py            ← Build pkl từ raw dataset XX
|   |   └── eval_per_scenario_XX.py     ← Eval cho dataset XX có nhiều fault type/scenario
│   └── models/
│       ├── basev3.py                   ← Vòng lặp train/eval (BaseModel)
│       ├── fuse_v3.py                  ← Model đa phương thức (log+metric+trace)
│       ├── log_model_v3.py             ← Log encoder (Transformer)
│       ├── kpi_model_v3.py             ← Metric encoder (Transformer)
│       ├── trace_model_v3.py           ← Trace encoder (GAT) + TraceModel
│       └── utils.py                    ← Các module dùng chung (Attention, ...)
└── data/
    └── XX/
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
  trace_node_features [B, W, N, trace_c]     ← service node features (trace_c=6)
  trace_adj           [B, W, N, N]           ← service call graph (STG)
  unmatched_kpi       [B, W, kpi_c]          ← shuffled KPI từ windows khác

  B = batch size | W = window size (tùy dataset) | N = num_services (tùy dataset)
  H = hidden_size (32) | trace_c = 6

  Node feature layout (trace_c=6):
    col 0: call_count   — số spans, normalized
    col 1: avg_dur_ms   — mean duration, normalized
    col 2: max_dur_ms   — max duration, normalized
    col 3: error_rate   — fraction lỗi ∈ [0,1]
    col 4: root_rate    — fraction root spans ∈ [0,1]
    col 5: latency_dev  — z-score(avg_dur vs pre-fault baseline)
```

---

## 3. Generator — `MultiModel.forward()`

### 3.1 MultiEncoder — [CHANGE 1] Trace tách khỏi Self-Attention

> **Thay đổi**:
> Giờ Self-Attention **chỉ** áp dụng trên log+KPI. Trace encoder chạy **riêng** → `ZV`.
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

### 3.3 Trace Autoencoder — [CHANGE 3] adj_hat + [CHANGE 6] Attribute Reconstruction Loss

> **CHANGE 3**: `adj_hat` (adjacency matrix tái tạo) được return ra ngoài `MultiModel`
> để Discriminator có thể dùng làm "fake trace adjacency".
>
> **CHANGE 6**: TraceModel thêm **attribute decoder** và hai reconstruction loss bổ sung:
> - `loss_latency`: MSE trên `latency_dev` (col 5) — phát hiện service chậm hơn baseline
> - `loss_error`: BCE trên `error_rate` (col 3) — phát hiện service tăng lỗi
>
> Mục đích: buộc node embedding ZV encode cả thông tin topology lẫn attribute anomaly,
> làm `trace_dis` discriminative hơn cho delay/loss/mem/socket fault.

```
  trace_nodes ──► TraceEncoder (2-layer GAT) ──► ZV [B*W, N, H]
                                                       │
                   ┌───────────────────────────────────┼─────────────────────────┐
                   │                                   │                         │
           Structural decoder                  Attribute decoder          adj_hat return
           A_hat = sigmoid(ZV·ZVᵀ)            X_hat = Linear(ZV)
           loss_struct = BCE(A_hat, adj)       loss_lat = MSE(X_hat[:,5], x[:,5])
           .mean(dim=[-2,-1])  [B]             loss_err = BCE(σ(X_hat[:,3]),      [B]
                                                              x[:,3].clamp(0,1))  [B]
                   │
  trace_dis = loss_struct + λ_lat × loss_lat + λ_err × loss_err    [B]
              (λ_lat = λ_err = 0.5 mặc định)

  adj_hat_4d   = A_hat.reshape(B, W, N, N)          ← CHANGE 3: trả ra MultiModel output
  feats_hat_4d = feats_hat[:,:,[3,5]].reshape(B, W, N, 2)  ← CHANGE 7: trả ra cho attribute discriminator
```

---

### 3.4 Fusion Loss — [CHANGE 4] Variance-based Alpha (thay thế Learnable trace_alpha)

> **Thay đổi**: `trace_alpha = nn.Parameter(-2.2)` (learnable theo thiết kế TraceDAE) → Learnable alpha bị thay bằng **variance-based alpha** — không có parameter.
>
> **Vấn đề với learnable alpha**: Gradient đẩy `trace_alpha` converge để cân bằng *reconstruction error magnitude*
> trên normal data — không phản ánh discriminativeness thực sự. Trên dataset trace yếu (e.g. RE3-OB),
> `trace_dis` gần constant → alpha tăng lên để bù → noise injection.
>
> **Giải pháp variance-based**: α phản ánh trực tiếp discriminativeness của trace signal so với log+KPI.
> Khi trace noise → var(trace_dis) thấp → α → 0 tự động.

```
  log_d   = log_dis  × expand_anomaly_gap(log_dis)
  kpi_d   = kpi_dis  × expand_anomaly_gap(kpi_dis)
  trace_d = trace_dis × expand_anomaly_gap(trace_dis)

  log_kpi_loss = log_d + kpi_d + narrow_modal_gap(|log_d − kpi_d|)

  var_lk    = var(log_kpi_loss.detach())      ← không có gradient
  var_trace = var(trace_dis.detach())
  α = var_trace / (var_lk + var_trace + ε)    ← ∈ [0, 1], không học được

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

### 3.6 Residual-Gated Trace Fusion — [CHANGE 8] (luôn bật khi `open_trace=True`)

> **Vấn đề**: Decoder cũ luôn dùng `cat([fused_modal, ZV])` → khi trace không có thông tin
> (vd RE3-OB code-defect), ZV nhiễu rò vào `kpi_out` / `log_out`, kéo F1 *xuống dưới baseline*.
> Variance-based α ở §3.4 chỉ re-weight **loss**, không chữa được output của decoder.
>
> **Giải pháp**: dạng residual với gate per-sample có thể đóng hoàn toàn đóng góp của trace.
>
> - `base_decoder`: `Linear(2H → H) → ReLU → Linear(H → kpi_c+log_c)` — đường baseline chỉ log+KPI
> - `delta_head`:   `Linear(3H → 2H) → ReLU → Linear(2H → kpi_c+log_c)` — **zero-init lớp cuối**
> - `trace_gate`:   `Linear(6 → 16) → ReLU → Linear(16 → 1) → sigmoid` — bias init `−2.0` → g₀ ≈ 0.12
> - **Trace-quality features** (per B,W): mean call count, coverage (span khác 0), mean error_rate,
>   mean |latency_dev|, adjacency density, call-count variance. Log1p-normalize.

```
  Trace-quality features [B, W, 6]
         │
   trace_gate (MLP, bias=-2.0) ──► g ∈ (0, 1)   [B, W, 1]
                                          │
  fused_modal [B,W,2H] ──► base_decoder ──► y_base [B,W,kpi_c+log_c]
                                          │
  cat([fm, ZV]) [B,W,3H] ─► delta_head ──► Δ [B,W,kpi_c+log_c]   (zero-init → Δ≈0 lúc đầu)
                                          │
                            fused_out = y_base + g · Δ    ← g=0 ⇒ đúng bằng baseline

  fusion_loss = log_kpi_loss + g · trace_d + gate_lambda · g.mean()
                                            └─── L1 reg giữ gate đóng mặc định
```

> **Bảo đảm**: khởi điểm `Δ ≈ 0` và `g ≈ 0.12`; gradient chỉ mở gate nếu
> `Δ` giảm loss log+KPI hơn `gate_lambda` (mặc định 0.01).
> Trên dataset mà trace nhiễu, gate đóng và mô hình *đúng bằng* baseline.
>
> **CLI**: `--gate_lambda 0.01` (tự động áp dụng khi `--open_trace True`).
>
> **Kiểm chứng**: Trên RE3-OB (5 fault type code-defect), residual-gated recover baseline F1
> trong phạm vi 0.015 trên mọi fault type.
> Xem `experiment_results_re3_ob_trace_vs_baseline_vi.md`.

---

## 4. Discriminator — `MultiDiscriminator.get_loss()`

### 4.1 Structural Trace Discrimination — [CHANGE 5] (không thay đổi)

> **Thay đổi**: FAKE pass dùng `adj_hat` (adjacency tái tạo từ Structure AE) →
> `trace_re_fake ≠ trace_re` → loss có ý nghĩa.
> Tương tự như log_out/kpi_out là "fake" cho log/KPI, `adj_hat` là "fake" cho trace structure.
>
> **Lưu ý về `trace_nodes`**: `trace_nodes` giống nhau trong cả REAL và FAKE → không đóng góp
> discriminative signal. Nó chỉ đóng vai trò là input node feature cho GAT attention trong `encoder_low`.
> Toàn bộ signal structural discrimination đến từ sự khác biệt giữa `adj` và `adj_hat`.

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

### 4.2 Attribute Trace Discrimination — [CHANGE 7] (mới, hoàn toàn độc lập)

> **Động lực**: Structural head (CHANGE 5) chỉ discriminate qua topology (adj vs adj_hat).
> Node attributes giống nhau trong cả REAL và FAKE → attributes không đóng góp gì cho discrimination.
> Thêm một attribute head riêng biệt so sánh ground-truth vs reconstructed node attributes
> buộc Generator phải sinh ra `error_rate` và `latency_dev` thực tế hơn.
>
> **Tại sao chỉ dùng col 3 & col 5?**
> - `error_rate` (col 3) và `latency_dev` (col 5) có supervision rõ ràng trong CHANGE 6
>   → chất lượng của `feats_hat[:,:,[3,5]]` đáng tin cậy.
> - 4 col còn lại (call_count, avg_dur_ms, max_dur_ms, root_rate) không có explicit loss
>   → chất lượng reconstruction kém → dùng chúng sẽ inject noise.
>
> **Code structural cũ không cần sửa** — attribute head là hoàn toàn additive.

```
  Attribute head (độc lập với structural, không qua encoder_low):

  REAL:  trace_nodes[:, :, [3, 5]]            [B, W, N, 2]
         → mean over N → attr_real            [B*W, 2]

  FAKE:  feats_hat[:, :, [3, 5]]              [B, W, N, 2]   ← từ TraceModel (CHANGE 7)
         → mean over N → attr_fake            [B*W, 2]

  attr_classifier: Linear(2, H) → ReLU → Linear(H, 1) → sigmoid
                                                          [B*W, 1]

  attr_disc_loss = BCE(attr_classifier(attr_real), real=1)
                 + BCE(attr_classifier(attr_fake), fake=0)

  attr_deceive_loss = MSE(attr_classifier(attr_real),
                          attr_classifier(attr_fake))

  Tổng hợp loss (cộng thêm vào):
  disc_loss    += attr_disc_loss
  deceive_loss += attr_deceive_loss
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

| #   | Thành phần                   | Trước (cũ)                                         | Sau (mới)                                                                             |
|:---:|:-----------------------------|:---------------------------------------------------|:--------------------------------------------------------------------------------------|
|  1  | **Self-Attention**           | cat([log‖kpi]) → 2H                               | cat([log‖kpi]) → 2H, trace **tách riêng**                                             |
|  2  | **Decoder input**            | fused_modal [B,W,2H hoặc 3H]                       | cat([fused_modal, ZV]) → [B,W,3H]                                                     |
|  3  | **adj_hat**                  | không dùng                                         | return ra ngoài MultiModel để Discriminator dùng                                      |
|  4  | **trace_alpha**              | `nn.Parameter(−2.2)`, learned ≈ 0.10              | **variance-based**: `α = var(trace) / (var(lk) + var(trace) + ε)`                    |
|  5  | **Discriminator FAKE**       | dùng `trace_adj` (thật) → contradiction            | dùng `adj_hat` (tái tạo) → valid loss                                                 |
|  6  | **trace_dis**                | BCE(A_hat, adj) — chỉ structural loss              | + λ_lat×MSE(latency_dev) + λ_err×BCE(error_rate)                                     |
|  7  | **Attribute Discriminator**  | không có — node attributes không được discriminate | head riêng: REAL=trace_nodes[:,:,[3,5]], FAKE=feats_hat[:,:,[3,5]] → Linear(2,H)→ReLU→Linear(H,1) |
|  8  | **Decoder fusion mode**      | luôn `cat([fm, ZV])` → noise rò vào kpi_out / log_out khi trace không informative | **Residual-gated**: `y_base(fm) + g · delta_head(cat[fm,ZV])`, `g∈[0,1]` per-sample từ 6 trace-quality features, `delta_head` zero-init, L1 reg trên g (`gate_lambda`) |

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
                        ┌─── base_decoder(fused_modal) ─────► y_base ─┐
                        │                                                │
                        │  cat([fused_modal, ZV]) [B,W,3H]              │
                        │           │                                    │
                        │    delta_head (zero-init) ─► Δ [B,W,out] ───►  +  ◄── g·Δ
                        │                                    ▲           │
                        │   trace-quality feats [B,W,6] ─► trace_gate → g│
                        └──────────── (g→0 khi trace không informative → y ≡ baseline) ┘
                                                         │
                                                 fused_out = y_base + g·Δ
                                                         │
                        kpi_out [B,W,kpi_c] + log_out [B,W,log_c]
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         │                          │                          │
    kpi_dis [B,W]            log_dis [B,W]            trace_dis [B,W]
    L1(kpi_out, kpi_x)       L1(log_out, log_x)       BCE(A_hat, adj)
                                                      + λ_lat×MSE(X_hat[:,5], x[:,5])
                                                      + λ_err×BCE(σ(X_hat[:,3]), x[:,3])
         │                          │                          │
         └──────────────────────────┴──────────────────────────┘
                                    │
                    α = var(trace_dis) / (var(log_kpi) + var(trace_dis) + ε)  ← variance-based
                    fusion_loss = (1-α)×(log+kpi) + α×trace    [B,W]
                             = Anomaly Score

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DISCRIMINATOR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Structural head (CHANGE 5):
    REAL: encoder_low(log_x,   kpi_x,   trace_nodes, adj)     → trace_re
    FAKE: encoder_low(log_out, kpi_out, trace_nodes, adj_hat) → trace_re_fake
    disc_loss    += CE(trace_re, real=1) + CE(trace_re_fake, fake=0)
    deceive_loss += MSE(trace_re, trace_re_fake)

  Attribute head (CHANGE 7 — mới, hoàn toàn độc lập):
    REAL: trace_nodes[:,:,[3,5]].mean(N) → attr_real  [B*W, 2]
    FAKE: feats_hat[:,:,[3,5]].mean(N)  → attr_fake  [B*W, 2]
    attr_classifier: Linear(2,H) → ReLU → Linear(H,1)
    disc_loss    += BCE(attr_classifier(attr_real), 1) + BCE(attr_classifier(attr_fake), 0)
    deceive_loss += MSE(attr_classifier(attr_real), attr_classifier(attr_fake))
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  OUTPUT: F1 / Recall / Precision (với point-adjustment)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 9. Khả năng đóng góp của Trace theo loại Fault

> **Khi nào trace giúp ích?** Chỉ khi fault thay đổi **hành vi observable ở tầng service-to-service**
> — tức là topology call graph, latency giữa các service, hoặc error_rate qua mạng.
> Khi fault **nội bộ trong service** và không ảnh hưởng đến các pattern này, trace không discriminative
> và CHANGE 8 (hard gate) tự động vô hiệu hoá nó hoàn toàn.

| Loại fault                                        | Trace có ích? | Lý do |
| **Network** (delay, packet loss, bandwidth limit) | ✅ Rõ ràng    | Latency spike trực tiếp + error_rate thay đổi giữa các service |
| **Resource** (CPU, memory, disk overload)         | ✅ Vừa phải   | Overload → service chậm → latency_dev tăng |
| **Service crash / OOM**                           | ✅ Rõ ràng    | Error rate spike, call count giảm |
| **Code Logic / Defect**                           | ❌ Không      | Bug chạy nội bộ — call graph và latency không đổi |
| **Configuration error**                           | ❌ Không      | Sai config nhưng service vẫn respond bình thường |
| **Database internal** (slow query, deadlock)      | ⚠️ Một phần   | Chỉ thấy nếu DB trong service mesh; latency service gọi DB có thể tăng |
| **Business logic** (tính sai, output sai)         | ❌ Không      | Output sai nhưng performance characteristics không đổi |
| **Security** (auth bypass, injection)             | ❌ Không      | Call pattern và latency không thay đổi |
