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
    """Structure Autoencoder for trace data.

    Encoder: two-layer GAT → node embeddings ZV.
    Decoder: A_hat = sigmoid(ZV * ZV^T)  (equation 7 in TraceDAE).
    Loss:    BCE(A_hat, A) — reconstruction of adjacency matrix.
    """

    def __init__(self, device, **kwargs):
        super(TraceModel, self).__init__()
        self.encoder = TraceEncoder(device, **kwargs)

    def forward(self, x, adj):
        """
        Args:
            x:   [B, N, trace_c]
            adj: [B, N, N]         binary adjacency (ground-truth structure)
        Returns:
            z:       [B, N, hidden_size]  node embeddings
            adj_hat: [B, N, N]            reconstructed adjacency
            loss:    [B]                  per-sample BCE reconstruction loss
        """
        z = self.encoder(x, adj)                                    # [B, N, H]
        adj_hat = torch.sigmoid(torch.bmm(z, z.transpose(1, 2)))   # [B, N, N]
        loss = F.binary_cross_entropy(adj_hat, adj, reduction='none')
        loss = loss.mean(dim=[-2, -1])                              # [B]
        return z, adj_hat, loss
