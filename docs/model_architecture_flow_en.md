# HADES + Trace — Model Architecture Flow

> This document describes the architecture and data flow of the HADES model extended with the Trace branch (GAT Structure Autoencoder).
> **Version**: `fuse_v3.py` with 7 architectural changes (based on TraceDAE).

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
│   │   ├── preprocess_XX.py            ← Build pkl from raw dataset XX
|   |   └── eval_per_scenario_XX.py     ← Eval for dataset XX have many inject fault types/ scenarios
│   └── models/
│       ├── basev3.py                   ← Train/eval loop (BaseModel)
│       ├── fuse_v3.py                  ← Multi-modal model (log+metric+trace)
│       ├── log_model_v3.py             ← Log encoder (Transformer)
│       ├── kpi_model_v3.py             ← Metric encoder (Transformer)
│       ├── trace_model_v3.py           ← Trace encoder (GAT) + TraceModel
│       └── utils.py                    ← Shared modules (Attention, ...)
└── data/
    └── XX/
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
  trace_node_features [B, W, N, trace_c]     ← service node features (trace_c=6)
  trace_adj           [B, W, N, N]           ← service call graph (STG)
  unmatched_kpi       [B, W, kpi_c]          ← shuffled KPI from other windows

  B = batch size | W = window size (numbers based on dataset) | N = num_services (numbers based on dataset)
  H = hidden_size (32) | trace_c = 6

  Node feature layout (trace_c=6):
    col 0: call_count   — span count, normalized
    col 1: avg_dur_ms   — mean duration, normalized
    col 2: max_dur_ms   — max duration, normalized
    col 3: error_rate   — error fraction ∈ [0,1]
    col 4: root_rate    — root span fraction ∈ [0,1]
    col 5: latency_dev  — z-score(avg_dur vs pre-fault baseline)
```

---

## 3. Generator — `MultiModel.forward()`

### 3.1 MultiEncoder — [CHANGE 1] Trace Separated from Self-Attention

> **Change**:
> Now Self-Attention is applied **only** to log+KPI. Trace encoder runs **separately** → `ZV`.
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

### 3.3 Trace Autoencoder — [CHANGE 3] adj_hat + [CHANGE 6] Attribute Reconstruction Loss

> **CHANGE 3**: `adj_hat` (reconstructed adjacency matrix) is returned from `MultiModel`
> so the Discriminator can use it as "fake trace adjacency".
>
> **CHANGE 6** : TraceModel adds an **attribute decoder** with two additional reconstruction losses:
> - `loss_latency`: MSE on `latency_dev` (col 5) — detects services slower than their pre-fault baseline
> - `loss_error`: BCE on `error_rate` (col 3) — detects services with elevated error rates
>
> Purpose: forces node embeddings ZV to encode both topology and attribute anomaly information,
> making `trace_dis` more discriminative for delay/loss/mem/socket fault types.

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
              (λ_lat = λ_err = 0.5 by default)

  adj_hat_4d  = A_hat.reshape(B, W, N, N)      ← CHANGE 3: returned from MultiModel output
  feats_hat_4d = feats_hat[:,:,[3,5]].reshape(B, W, N, 2)  ← CHANGE 7: returned for attr discriminator
```

---

### 3.4 Fusion Loss — [CHANGE 4] Variance-based Alpha (replaces Learnable trace_alpha)

> **Change**: `trace_alpha = nn.Parameter(-2.2)` (learnable as TraceDAE design) → Learnable alpha replaced by **variance-based alpha** — no parameters.
>
> **Problem with learnable alpha**: Gradients push `trace_alpha` to converge so as to balance
> *reconstruction error magnitudes* on normal data — not actual discriminativeness. On datasets
> with weak trace signal (e.g. RE3-OB), `trace_dis` is nearly constant → alpha increases to
> compensate → noise is injected into the anomaly score.
>
> **Variance-based solution**: α directly reflects the relative discriminativeness of the trace signal
> compared to log+KPI. When trace is noisy → var(trace_dis) is low → α → 0 automatically.

```
  log_d   = log_dis  × expand_anomaly_gap(log_dis)
  kpi_d   = kpi_dis  × expand_anomaly_gap(kpi_dis)
  trace_d = trace_dis × expand_anomaly_gap(trace_dis)

  log_kpi_loss = log_d + kpi_d + narrow_modal_gap(|log_d − kpi_d|)

  var_lk    = var(log_kpi_loss.detach())      ← no gradient
  var_trace = var(trace_dis.detach())
  α = var_trace / (var_lk + var_trace + ε)    ← ∈ [0, 1], no learned parameters

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

## 4. Discriminator — `MultiDiscriminator.get_loss()`

### 4.1 Structural Trace Discrimination — [CHANGE 5] (unchanged)

> **Change**: FAKE pass uses `adj_hat` (adjacency reconstructed by the Structure AE) →
> `trace_re_fake ≠ trace_re` → loss is meaningful.
> Analogous to how log_out/kpi_out are "fake" for log/KPI, `adj_hat` is "fake" for trace structure.
>
> **Note on `trace_nodes`**: `trace_nodes` is identical in both REAL and FAKE passes — it contributes
> zero discriminative signal on its own. It serves only as node feature input for GAT attention
> computation in `encoder_low`. The entire structural discrimination signal comes from `adj` vs `adj_hat`.

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

### 4.2 Attribute Trace Discrimination — [CHANGE 7] (new, additive)

> **Motivation**: The structural head (CHANGE 5) only discriminates via topology (adj vs adj_hat).
> Node attributes are identical in REAL and FAKE → attributes contribute nothing to discrimination.
> Adding a separate attribute head that compares ground-truth vs reconstructed node attributes
> forces the Generator to produce realistic `error_rate` and `latency_dev` values.
>
> **Why only col 3 & col 5?**
> - `error_rate` (col 3) and `latency_dev` (col 5) have explicit supervision in CHANGE 6
>   → `feats_hat[:,:,[3,5]]` quality is reliable.
> - Other 4 cols (call_count, avg_dur_ms, max_dur_ms, root_rate) have no explicit loss
>   → reconstruction quality is poor → using them would inject noise.
>
> **Existing structural code is unchanged** — the attribute head is purely additive.

```
  Attribute head (separate from structural, no encoder_low involved):

  REAL:  trace_nodes[:, :, [3, 5]]            [B, W, N, 2]
         → mean over N → attr_real            [B*W, 2]

  FAKE:  feats_hat[:, :, [3, 5]]              [B, W, N, 2]   ← from TraceModel (CHANGE 7)
         → mean over N → attr_fake            [B*W, 2]

  attr_classifier: Linear(2, H) → ReLU → Linear(H, 1) → sigmoid
                                                          [B*W, 1]

  attr_disc_loss = BCE(attr_classifier(attr_real), real=1)
                 + BCE(attr_classifier(attr_fake), fake=0)

  attr_deceive_loss = MSE(attr_classifier(attr_real),
                          attr_classifier(attr_fake))

  Total losses (combined):
  disc_loss    += attr_disc_loss
  deceive_loss += attr_deceive_loss
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

| #   | Component                   | Before (old)                                       | After (new)                                                                    |
|:---:|:----------------------------|:---------------------------------------------------|:-------------------------------------------------------------------------------|
|  1  | **Self-Attention**          | cat([log‖kpi]) → 2H                               | cat([log‖kpi]) → 2H, trace **separated**                                       |
|  2  | **Decoder input**           | fused_modal [B,W,2H or 3H]                         | cat([fused_modal, ZV]) → [B,W,3H]                                              |
|  3  | **adj_hat**                 | not used                                           | returned from MultiModel for use by Discriminator                              |
|  4  | **trace_alpha**             | `nn.Parameter(−2.2)`, learned ≈ 0.10              | **variance-based**: `α = var(trace) / (var(lk) + var(trace) + ε)`             |
|  5  | **Discriminator FAKE**      | uses `trace_adj` (real) → contradiction            | uses `adj_hat` (reconstructed) → valid loss                                    |
|  6  | **trace_dis**               | BCE(A_hat, adj) — structural loss only             | + λ_lat×MSE(latency_dev) + λ_err×BCE(error_rate)                              |
|  7  | **Attribute Discriminator** | not present — node attributes not discriminated    | separate head: REAL=trace_nodes[:,:,[3,5]], FAKE=feats_hat[:,:,[3,5]] → Linear(2,H)→ReLU→Linear(H,1) |

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

  Attribute head (CHANGE 7 — new, separate):
    REAL: trace_nodes[:,:,[3,5]].mean(N) → attr_real  [B*W, 2]
    FAKE: feats_hat[:,:,[3,5]].mean(N)  → attr_fake  [B*W, 2]
    attr_classifier: Linear(2,H) → ReLU → Linear(H,1)
    disc_loss    += BCE(attr_classifier(attr_real), 1) + BCE(attr_classifier(attr_fake), 0)
    deceive_loss += MSE(attr_classifier(attr_real), attr_classifier(attr_fake))
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  OUTPUT: F1 / Recall / Precision (with point-adjustment)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
