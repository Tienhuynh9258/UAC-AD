import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """Single Graph Attention Network layer.

    Computes attention-weighted aggregation of neighbor features following
    the TraceDAE Structure Autoencoder formulation (equations 3-6).
    """

    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super(GATLayer, self).__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        """
        Args:
            x:   [B, N, in_features]  node feature matrix
            adj: [B, N, N]            adjacency matrix (binary or weighted)
        Returns:
            h_out: [B, N, out_features]
        """
        h = self.W(x)                          # [B, N, out_features]
        B, N, D = h.shape

        # Build pairwise concatenation for attention scoring
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)   # [B, N, N, D]
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)   # [B, N, N, D]
        e = self.leakyrelu(
            self.a(torch.cat([h_i, h_j], dim=-1)).squeeze(-1)
        )  # [B, N, N]

        # Add self-loops, then mask non-neighbors to -inf
        adj_self = (adj + torch.eye(N, device=adj.device).unsqueeze(0)) > 0
        e = e.masked_fill(~adj_self, float('-inf'))

        alpha = F.softmax(e, dim=-1)           # [B, N, N]
        alpha = torch.nan_to_num(alpha, nan=0.0)
        alpha = self.dropout(alpha)

        h_out = torch.bmm(alpha, h)            # [B, N, out_features]
        return F.elu(h_out)


class TraceEncoder(nn.Module):
    """GAT-based Structure Autoencoder encoder (TraceDAE, Section IV-C-1).

    Encodes a Service Trace Graph (STG) into node embeddings ZV via:
      1. Linear projection: Z'V = ReLU(X * Wx + bx)
      2. Two GAT layers for structural neighborhood aggregation
    """

    def __init__(self, device, **kwargs):
        super(TraceEncoder, self).__init__()
        self.trace_c = kwargs["trace_c"]
        self.hidden_size = kwargs["hidden_size"]
        dropout = kwargs.get("trace_dropout", 0.1)

        self.input_proj = nn.Linear(self.trace_c, self.hidden_size)
        self.gat1 = GATLayer(self.hidden_size, self.hidden_size, dropout=dropout)
        self.gat2 = GATLayer(self.hidden_size, self.hidden_size, dropout=dropout)

    def forward(self, x, adj):
        """
        Args:
            x:   [B, N, trace_c]   node feature matrix
            adj: [B, N, N]         adjacency matrix
        Returns:
            z: [B, N, hidden_size]  node embeddings ZV
        """
        z = F.relu(self.input_proj(x))  # Z'V
        z = self.gat1(z, adj)
        z = self.gat2(z, adj)
        return z


class TraceEncoder_low(nn.Module):
    """Lightweight GAT encoder used in the Discriminator."""

    def __init__(self, device, **kwargs):
        super(TraceEncoder_low, self).__init__()
        dropout = kwargs.get("trace_dropout", 0.1)
        self.input_proj = nn.Linear(kwargs["trace_c"], kwargs["hidden_size"])
        self.gat = GATLayer(kwargs["hidden_size"], kwargs["hidden_size"], dropout=dropout)

    def forward(self, x, adj):
        """
        Args:
            x:   [B, N, trace_c]
            adj: [B, N, N]
        Returns:
            z: [B, N, hidden_size]
        """
        z = F.relu(self.input_proj(x))
        z = self.gat(z, adj)
        return z


class TraceModel(nn.Module):
    """Structure + Attribute Autoencoder for trace data.

    Encoder: two-layer GAT → node embeddings ZV.
    Structural decoder:  A_hat = sigmoid(ZV * ZV^T)  (equation 7 in TraceDAE).
    Attribute decoder:   X_hat = Linear(ZV) → reconstruct latency_dev and error_rate.

    Combined anomaly score (trace_dis):
        loss_struct  = BCE(A_hat, A)              — topology reconstruction
        loss_latency = MSE(X_hat[:,5], X[:,5])    — latency_dev (z-score) reconstruction
        loss_error   = BCE(X_hat[:,3], X[:,3])    — error_rate reconstruction
        trace_dis    = loss_struct
                     + lambda_lat * loss_latency
                     + lambda_err * loss_error

    Feature layout expected in x (col index):
        0  call_count
        1  avg_dur_ms
        2  max_dur_ms
        3  error_rate   ∈ [0,1]  ← attribute loss (BCE)
        4  root_rate
        5  latency_dev           ← attribute loss (MSE)
    """

    # Feature column indices (must match preprocessing TRACE_C layout)
    COL_ERROR_RATE  = 3
    COL_LATENCY_DEV = 5

    def __init__(self, device, **kwargs):
        super(TraceModel, self).__init__()
        self.encoder    = TraceEncoder(device, **kwargs)
        self.trace_c    = kwargs["trace_c"]
        hidden_size     = kwargs["hidden_size"]

        # Attribute decoder: Z [B,N,H] → X_hat [B,N,trace_c]
        self.feat_decoder = nn.Linear(hidden_size, self.trace_c)

        # Loss weights for attribute reconstruction terms
        self.lambda_lat = kwargs.get("lambda_lat", 0.5)
        self.lambda_err = kwargs.get("lambda_err", 0.5)

    def forward(self, x, adj):
        """
        Args:
            x:   [B, N, trace_c]
            adj: [B, N, N]         binary adjacency (ground-truth structure)
        Returns:
            z:       [B, N, hidden_size]  node embeddings
            adj_hat: [B, N, N]            reconstructed adjacency
            loss:    [B]                  per-sample combined reconstruction loss
        """
        z = self.encoder(x, adj)                                    # [B, N, H]

        # ── Structural decoder ────────────────────────────────────────────────
        adj_hat = torch.sigmoid(torch.bmm(z, z.transpose(1, 2)))   # [B, N, N]
        loss_struct = F.binary_cross_entropy(adj_hat, adj, reduction='none')
        loss_struct = loss_struct.mean(dim=[-2, -1])                # [B]

        # ── Attribute decoder ─────────────────────────────────────────────────
        feats_hat = self.feat_decoder(z)                            # [B, N, trace_c]

        # latency_dev (col 5) — MSE: z-score can be negative, no sigmoid needed
        loss_latency = F.mse_loss(
            feats_hat[:, :, self.COL_LATENCY_DEV],
            x[:, :, self.COL_LATENCY_DEV],
            reduction='none',
        ).mean(dim=-1)                                              # [B]

        # error_rate (col 3) — BCE: both in [0,1]
        err_hat = torch.sigmoid(feats_hat[:, :, self.COL_ERROR_RATE])
        loss_error = F.binary_cross_entropy(
            err_hat,
            x[:, :, self.COL_ERROR_RATE].clamp(0.0, 1.0),
            reduction='none',
        ).mean(dim=-1)                                              # [B]

        # ── Combined trace_dis ────────────────────────────────────────────────
        loss = (loss_struct
                + self.lambda_lat * loss_latency
                + self.lambda_err * loss_error)

        # CHANGE 7: return feats_hat[:,:,[3,5]] for attribute discriminator
        # Only cols with explicit supervision (error_rate=3, latency_dev=5)
        feats_hat_slice = feats_hat[:, :, [self.COL_ERROR_RATE, self.COL_LATENCY_DEV]]  # [B, N, 2]

        return z, adj_hat, loss, feats_hat_slice
