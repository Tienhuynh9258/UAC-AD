# HADES + Trace — Model Architecture Flow

> This document describes the architecture and data flow of the HADES model extended with the Trace branch (GAT Structure Autoencoder).
> **Version**: `fuse_v3.py` with 5 architectural changes (based on TraceDAE).

---

## 1. Project Architecture Overview

```
UAC-AD/
├── codes/
│   ├── run.py                          ← Main entry point
│   ├── run_sequential.py               ← Sequential run (avoids CUDA OOM)
│   ├── common/
│   │   ├── data_loads.py               ← Load & window data → DataLoader
│   │   ├── semantics.py                ← Extract log features (Word2Vec/template)
│   │   ├── utils.py                    ← General utilities (seed, dump results...)
│   │   └── preprocess_micross.py       ← Build pkl from raw MicroSS CSV
│   └── models/
│       ├── basev3.py                   ← Train/eval loop (BaseModel)
│       ├── fuse_v3.py                  ← Multi-modal model (log+metric+trace)
│       ├── log_model_v3.py             ← Log encoder (Transformer)
│       ├── kpi_model_v3.py             ← Metric encoder (Transformer)
│       ├── trace_model_v3.py           ← Trace encoder (GAT) + TraceModel
│       └── utils.py                    ← Shared modules (Attention, ...)
└── data/
    └── micross/
        ├── train.pkl / unlabel.pkl / test.pkl
        └── meta.pkl
```

---

## 2. Input Data Flow

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  INPUT  [B, W, *]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  log_features        [B, W, log_c]          ← log template features
  kpi_features        [B, W, kpi_c]          ← KPI metrics
  trace_node_features [B, W, N, trace_c]     ← MicroSS: service node features
  trace_adj           [B, W, N, N]           ← MicroSS: service call graph (STG)
  unmatched_kpi       [B, W, kpi_c]          ← shuffled KPI from other windows

  B = batch size | W = window size (50) | N = num_services (4)
  H = hidden_size (32)
```

---

## 3. Generator — `MultiModel.forward()`

### 3.1 MultiEncoder — [CHANGE 1] Trace Separated from Self-Attention

> **Change**: Previously trace was fed into Self-Attention together with log+KPI → `[B,W,3H]`.
> Now Self-Attention is applied **only** to log+KPI. Trace encoder runs **separately** → `ZV`.
> Rationale: self-attention over 3 modalities dilutes the log↔KPI relationship; consistent with TraceDAE design
> (structural info should guide the **decoder**, not encoder attention).

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
  │                   mean_pool over N → ZV [B, W, H]     (separate)      │
  └────────────────────────────────────────────────────────────────────────┘

  Returns: (fused_kpi, fused_log, fused_modal [B,W,2H], ZV [B,W,H])
```

> **Without trace** (`open_trace=False`): ZV = None, fused_modal remains `[B,W,2H]`.

---

### 3.2 Reconstruction — [CHANGE 2] ZV Injected into Decoder

> **Change**: Decoder receives `cat([fused_modal, ZV])` instead of only `fused_modal`.
> Rationale: Inspired by TraceDAE (Eq.10) — structural embedding ZV should guide **reconstruction**,
> not fusion attention. Analogous to `X_hat = ZV · ZA^T` in TraceDAE.

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

> **Without trace** (ZV=None): fuse_decoder = Linear(2H → kpi_c+log_c) — decoder input = 2H.

---

### 3.3 Trace Structure Autoencoder & adj_hat — [CHANGE 3]

> **Change**: `adj_hat` (reconstructed adjacency matrix) is returned from `MultiModel`
> so the Discriminator can use it as "fake trace adjacency".

```
  trace_nodes ──► TraceEncoder (GAT) ──► ZV [B*W, N, H]
                                              │
                         A_hat = sigmoid(ZV · ZVᵀ)  [B*W, N, N]
                                              │
              trace_dis = BCE(A_hat, trace_adj).mean([N,N])  [B, W]

  adj_hat_4d = A_hat.reshape(B, W, N, N)   ← CHANGE 3: returned from MultiModel output
```

---

### 3.4 Fusion Loss — [CHANGE 4] Learnable trace_alpha

> **Change**: `trace_weight` (fixed float, requires manual tuning) → `trace_alpha` (learnable parameter).
> Initialization: `nn.Parameter(tensor(-2.2))` → `sigmoid(-2.2) ≈ 0.10` (matches α=0.1 in TraceDAE).
> α is learned from data, automatically balancing between (log+KPI) reconstruction and trace structural loss.

```
  log_d   = log_dis  × expand_anomaly_gap(log_dis)
  kpi_d   = kpi_dis  × expand_anomaly_gap(kpi_dis)
  trace_d = trace_dis × expand_anomaly_gap(trace_dis)

  α = sigmoid(trace_alpha)   ← learned, initialized ≈ 0.10

  log_kpi_loss = log_d + kpi_d + narrow_modal_gap(|log_d − kpi_d|)

  fusion_loss = (1 − α) × log_kpi_loss + α × trace_d    [B, W]
                └─────────────────────────────────────────────────────┘
                         Anomaly Score (used for evaluation)
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

> **Change (important bug fix)**: Previously both REAL and FAKE passes used the real `trace_adj`
> → `trace_re == trace_re_fake` → CE loss required the same vector to be both 1 AND 0 simultaneously → contradictory gradients.
>
> Now the FAKE pass uses `adj_hat` (adjacency reconstructed by the Structure AE) → `trace_re_fake ≠ trace_re` → loss is meaningful.
> Analogous to how log_out/kpi_out are "fake" for log/KPI, `adj_hat` is "fake" for trace.

```
  MultiEncoder_low (lightweight, 1-layer GAT):
  ┌──────────────────────────────────────────────────────────────────────┐
  │  REAL:  encoder_low(log_x,   kpi_x,   trace_nodes, trace_adj)        │
  │         → log_re, kpi_re, trace_re              [B, W, H]            │
  │                                                                       │
  │  FAKE:  encoder_low(log_out, kpi_out, trace_nodes, adj_hat)  ← CHANGE│
  │         → log_re_fake, kpi_re_fake, trace_re_fake                    │
  │         (adj_hat from Structure AE → trace_re_fake ≠ trace_re  ✅)   │
  └──────────────────────────────────────────────────────────────────────┘

  Discriminator loss (update Discriminator weights):
  disc_loss = CE(pred_kpi,       real=1) + CE(pred_kpi_fake,   fake=0)
            + CE(pred_log,       real=1) + CE(pred_log_fake,   fake=0)
            + CE(pred_trace,     real=1) + CE(pred_trace_fake, fake=0)

  Deceive loss (update Generator weights):
  deceive_loss = MSE(kpi_re, kpi_re_fake)
               + MSE(log_re, log_re_fake)
               + MSE(trace_re, trace_re_fake)
```

---

## 5. Training Loop (per epoch)

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
        → point_adjustment(pred, gt)  ← if any point in a segment is detected
                                         → entire segment = detected
        → F1 / Recall / Precision
```

---

## 6. Routing by `data_type`

```
  data_type = "fuse"  →  MultiModel   (log + KPI + trace if open_trace=True)
  data_type = "log"   →  LogModel     (log only)
  data_type = "kpi"   →  KpiModel     (KPI only, open_trace ignored)
```

> ⚠️ Only `data_type=fuse` activates the trace branch.

---

## 7. Architecture Comparison: Before vs After (fuse_v3.py)

| #   | Component              | Before (old)                                       | After (new)                                                        |
|:---:|:-----------------------|:---------------------------------------------------|:-------------------------------------------------------------------|
|  1  | **Self-Attention**     | cat([log‖kpi‖trace]) → 3H                          | cat([log‖kpi]) → 2H, trace **separated**                           |
|  2  | **Decoder input**      | fused_modal [B,W,2H or 3H]                         | cat([fused_modal, ZV]) → [B,W,3H]                                  |
|  3  | **adj_hat**            | not used                                           | returned from MultiModel for use by Discriminator                  |
|  4  | **trace_weight**       | fixed float (hyperparameter)                       | `trace_alpha = nn.Parameter(−2.2)`, learned                        |
|  5  | **Discriminator FAKE** | uses `trace_adj` (real) → contradiction            | uses `adj_hat` (reconstructed) → valid loss                        |

---

## 8. End-to-End Overview Diagram (after improvements)

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
  OUTPUT: F1 / Recall / Precision (with point-adjustment)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
