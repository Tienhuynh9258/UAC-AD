import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softmax as sf
from models.utils import *
from models.kpi_model_v3 import KpiEncoder, KpiEncoder_low
from models.log_model_v3 import LogEncoder, LogEncoder_low
from models.utils import MultiHeadAttention
from models.trace_model_v3 import TraceEncoder, TraceEncoder_low, TraceModel

class AddAttention(nn.Module): #k=V
    def __init__(self, dimensions,windows_lens=100):
        super(AddAttention, self).__init__()
        self.linear_in = nn.Linear(dimensions, dimensions, bias=True)
        self.linear_in2 = nn.Linear(dimensions, dimensions, bias=True)
        self.linear_out = nn.Linear(dimensions, windows_lens, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.fc=nn.Linear(2*dimensions, dimensions, bias=True)

    def forward(self, query, context): #[batch_size, length, dim]
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)
        query_vec = self.linear_in(query.reshape(batch_size * output_len, dimensions))
        context_vec = self.linear_in2(context.reshape(batch_size * output_len, dimensions))
        alpha=self.linear_out (self.tanh(query_vec+context_vec))
        alpha=alpha.reshape(batch_size,output_len,output_len)
        alpha=self.softmax(alpha)
        output= torch.bmm(alpha,context)
        output = torch.cat((output, context), dim=-1)
        output=self.fc(output)
        return output, alpha

class DotAttention(nn.Module): #k=V
    def __init__(self, dimensions):
        super(DotAttention, self).__init__()
        self.linear_in = nn.Linear(dimensions, dimensions, bias=False)
        self.linear_in2 = nn.Linear(dimensions, dimensions, bias=False)
        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context): #[batch_size, length, dim]
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)
        query = query.reshape(batch_size * output_len, dimensions)
        query = self.linear_in(query)
        query = query.reshape(batch_size, output_len, dimensions)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)
        mix = torch.bmm(attention_weights, context)
        return mix, attention_weights


# ═══════════════════════════════════════════════════════════════════════════════
# CHANGE 1: MultiEncoder — trace tách khỏi Self-Attention
#   - Self-Attention chỉ áp dụng trên log+KPI → fused_modal [B,W,2H]
#   - TraceEncoder chạy riêng → ZV [B,W,H]  (4th return value)
#   - fused_modal luôn là 2H, không phụ thuộc open_trace
# ═══════════════════════════════════════════════════════════════════════════════
class MultiEncoder(nn.Module):
    def __init__(self, var_nums, device, vocab_size=300, fuse_type="cross_attn", **kwargs):
        super(MultiEncoder, self).__init__()
        self.log_encoder = LogEncoder(device, **kwargs)
        self.kpi_encoder = KpiEncoder(device, **kwargs)
        self.hidden_size = kwargs["hidden_size"]
        self.window_size = 100
        self.feature_type = kwargs["feature_type"]
        self.fuse_type = fuse_type
        self.open_trace = kwargs.get("open_trace", False)
        if self.open_trace:
            self.trace_encoder = TraceEncoder(device, **kwargs)
        if self.fuse_type == "cross_attn" or self.fuse_type == "sep_attn":
            if kwargs["attn_type"] == "add":
                self.attn_alpha = AddAttention(self.hidden_size, kwargs["window_size"])
                self.attn_beta  = AddAttention(self.hidden_size, kwargs["window_size"])
            elif kwargs["attn_type"] == "qkv":
                self.attn_alpha = MultiHeadAttention(self.hidden_size, 8, device=device)
                self.attn_beta  = MultiHeadAttention(self.hidden_size, 8, device=device)
            elif kwargs["attn_type"] == "dot":
                self.attn_alpha = DotAttention(self.hidden_size)
                self.attn_beta  = DotAttention(self.hidden_size)
        elif self.fuse_type == "multi_modal_self_attn":
            # CHANGE 1: Self-Attention chỉ trên log+KPI → luôn 2H, bất kể open_trace
            self.self_attention = MultiHeadAttention(2 * self.hidden_size, 2, device=device)

    def forward(self, log_x, kpi_x, trace_nodes=None, trace_adj=None):
        kpi_re = self.kpi_encoder(kpi_x)  # [B, W, H]
        log_re = self.log_encoder(log_x)  # [B, W, H]

        # CHANGE 1: Trace encoder chạy riêng → ZV [B,W,H], KHÔNG đưa vào attention
        ZV = None
        if self.open_trace and trace_nodes is not None and trace_adj is not None:
            B, W, N, C = trace_nodes.shape
            trace_z = self.trace_encoder(
                trace_nodes.reshape(B * W, N, C),
                trace_adj.reshape(B * W, N, N)
            )  # [B*W, N, H]
            ZV = trace_z.mean(dim=1).reshape(B, W, self.hidden_size)  # [B, W, H]

        # Fusion: chỉ log + KPI
        fused_modal = None
        if self.fuse_type == "cross_attn":
            fused_kpi, _ = self.attn_alpha(query=log_re, context=kpi_re)
            fused_log, _ = self.attn_beta(query=kpi_re, context=log_re)
            fused_modal = torch.cat((fused_kpi, fused_log), dim=-1)
        elif self.fuse_type == "sep_attn":
            fused_kpi, _ = self.attn_alpha(query=kpi_re, context=kpi_re)
            fused_log, _ = self.attn_beta(query=log_re, context=log_re)
            fused_modal = torch.cat((fused_kpi, fused_log), dim=-1)
        elif self.fuse_type == "concat":
            fused_kpi  = kpi_re
            fused_log  = log_re
            fused_modal = torch.cat((kpi_re, log_re), dim=-1)   # [B,W,2H]
        elif self.fuse_type == "multi_modal_self_attn":
            fused_kpi  = kpi_re
            fused_log  = log_re
            # CHANGE 1: Self-Attention chỉ log+KPI, không có trace
            fused_modal = torch.cat((kpi_re, log_re), dim=-1)   # [B,W,2H]
            fused_modal = self.self_attention(fused_modal, fused_modal)[0]

        # 4th return: ZV thay cho trace_re cũ
        return fused_kpi, fused_log, fused_modal, ZV


class ReturnSelf(nn.Module):
    def __init__(self):
        super(ReturnSelf, self).__init__()
    def forward(self, x):
        return x

class ReturnTopXWeight(nn.Module):
    def __init__(self):
        super(ReturnTopXWeight, self).__init__()
        self.relu = nn.ReLU()
    def forward(self, x):
        m = torch.quantile(x, 0.8, dim=-1, keepdim=True)
        w = x - m
        return self.relu(w) + 1.0

class Return0(nn.Module):
    def __init__(self):
        super(Return0, self).__init__()
    def forward(self, x):
        return 0

class Return1(nn.Module):
    def __init__(self):
        super(Return1, self).__init__()
    def forward(self, x):
        return 1

class Loss_fuse_model(nn.Module):
    def __init__(self, **keywds):
        super(Loss_fuse_model, self).__init__()
        self.weight = nn.Linear(keywds["window_size"]*2, keywds["window_size"], bias=False)
        if keywds["sigma_matrix"]:
            self.sigmas_dota = nn.Parameter(
                nn.init.uniform_(torch.empty(keywds["window_size"]), a=0.2, b=1.0),
                requires_grad=True)
        else:
            self.sigmas_dota = nn.Parameter(
                nn.init.uniform_(torch.empty(1), a=0.2, b=1.0),
                requires_grad=True)
        self.weight2_0 = nn.Linear(keywds["hidden_size"]*2, 1, bias=False)
        self.s  = ReturnSelf()       if keywds["open_narrowing_modal_gap"] else Return0()
        self.f3 = ReturnTopXWeight() if keywds["open_expand_anomaly_gap"]  else Return1()

    def forward(self, loss_set):
        w = self.sigmas_dota
        log_d = loss_set[0] * self.f3(loss_set[0])
        kpi_d = loss_set[1] * self.f3(loss_set[1])
        loss_part = w*log_d + (1-w)*kpi_d + self.s(torch.abs(log_d-kpi_d))
        return {"loss": loss_part, "w": w}

    def get_loss(self, res_set):
        log_x = res_set["features"][0]
        kpi_x = res_set["features"][1]
        log_d = res_set["distance"][0] * self.f3(res_set["distance"][0])
        kpi_d = res_set["distance"][1] * self.f3(res_set["distance"][1])
        w = self.weight2_0(torch.concatenate([kpi_x, log_x], dim=-1)).squeeze()
        loss_part = w*log_d + (1-w)*kpi_d + self.s(
            torch.abs(res_set["distance"][0] - res_set["distance"][1]))
        return {"loss": loss_part, "w": w}


class MultiEncoder_low(nn.Module):
    """Lightweight encoder used by the Discriminator (concat fusion, no self-attn)."""
    def __init__(self, var_nums, device, vocab_size=300, fuse_type="cross_attn", **kwargs):
        super(MultiEncoder_low, self).__init__()
        self.log_encoder = LogEncoder_low(device, **kwargs).to(device)
        self.kpi_encoder = KpiEncoder_low(device, **kwargs).to(device)
        self.hidden_size = kwargs["hidden_size"]
        self.window_size = 100
        self.feature_type = kwargs["feature_type"]
        self.fuse_type = "concat"
        self.open_trace = kwargs.get("open_trace", False)
        if self.open_trace:
            self.trace_encoder = TraceEncoder_low(device, **kwargs).to(device)

    def forward(self, log_x, kpi_x, trace_nodes=None, trace_adj=None):
        kpi_re = self.kpi_encoder(kpi_x)  # [B, W, H]
        log_re = self.log_encoder(log_x)  # [B, W, H]

        trace_re = None
        if self.open_trace and trace_nodes is not None and trace_adj is not None:
            B, W, N, C = trace_nodes.shape
            trace_z = self.trace_encoder(
                trace_nodes.reshape(B * W, N, C),
                trace_adj.reshape(B * W, N, N)
            )  # [B*W, N, H]
            trace_re = trace_z.mean(dim=1).reshape(B, W, self.hidden_size)  # [B, W, H]

        fused_kpi = kpi_re
        fused_log = log_re
        if trace_re is not None:
            fused = torch.cat((kpi_re, log_re, trace_re), dim=-1)  # [B,W,3H]
        else:
            fused = torch.cat((kpi_re, log_re), dim=-1)             # [B,W,2H]

        return fused_kpi, fused_log, fused, trace_re


# ═══════════════════════════════════════════════════════════════════════════════
# CHANGE 5: MultiDiscriminator — FAKE pass dùng adj_hat thay vì trace_adj
#   - REAL: encode(log_x,   kpi_x,   trace_nodes, trace_adj)
#   - FAKE: encode(log_out, kpi_out, trace_nodes, adj_hat)  ← adj_hat từ Structure AE
#   - trace_re ≠ trace_re_fake → loss terms có ý nghĩa
# ═══════════════════════════════════════════════════════════════════════════════
class MultiDiscriminator(nn.Module):
    def __init__(self, var_nums, device, fuse_type="cross_attn", **kwargs):
        super(MultiDiscriminator, self).__init__()
        self.fuse_type  = fuse_type
        self.open_trace = kwargs.get("open_trace", False)
        self.encoder    = MultiEncoder_low(var_nums=var_nums, device=device,
                                           fuse_type=fuse_type, **kwargs)
        _n_modal = 3 if self.open_trace else 2
        _W, _H   = kwargs["window_size"], kwargs["hidden_size"]
        self.fc           = nn.Linear(_W * _H * _n_modal, _H)
        self.decoder_fuse = nn.Linear(_H, 2)
        self.decoder      = nn.Linear(_W * _H * _n_modal, 2)
        self.decoder2     = nn.Linear(_W * _H * _n_modal, 2)
        self.decoder3     = nn.Linear(_W * _H, 2)   # log classifier
        self.decoder4     = nn.Linear(_W * _H, 2)   # kpi classifier
        if self.open_trace:
            self.decoder5 = nn.Linear(_W * _H, 2)   # structural trace classifier
            # CHANGE 7: attribute head — discriminate on error_rate & latency_dev
            # Input: mean over N services of feats_hat[:,:,[3,5]] → [B*W, 2]
            self.attr_classifier = nn.Sequential(
                nn.Linear(2, _H),
                nn.ReLU(),
                nn.Linear(_H, 1),
            )
        self.criterion  = nn.CrossEntropyLoss()
        self.criterion2 = nn.MSELoss()

    def _get_real_trace_inputs(self, input_dict):
        """Return (trace_nodes, trace_adj) for the REAL pass."""
        if self.open_trace and "trace_node_features" in input_dict:
            return input_dict["trace_node_features"], input_dict["trace_adj"]
        return None, None

    def _get_fake_trace_adj(self, input_dict, res_set):
        """Return adj_hat [B,W,N,N] for the FAKE pass (from Structure AE output)."""
        if self.open_trace and "adj_hat" in res_set:
            return res_set["adj_hat"]   # [B, W, N, N]
        return None

    def get_loss_old(self, input_dict, res_set, flag=False):
        log_x          = input_dict["log_features"]
        kpi_x          = input_dict["kpi_features"]
        unmatched_kpi_x = input_dict["unmatched_kpi_features"]
        log_x_fake     = res_set["output"][0]
        kpi_x_fake     = res_set["output"][1]
        trace_nodes, trace_adj = self._get_real_trace_inputs(input_dict)
        b, _, _ = kpi_x.shape
        kpi_re, log_re, concate_feature, _ = self.encoder(log_x, kpi_x, trace_nodes, trace_adj)
        pred = self.decoder(concate_feature.reshape(b, -1))
        kpi_re_fake, log_re_fake, concate_feature_fake, _ = self.encoder(
            log_x_fake, kpi_x_fake, trace_nodes, trace_adj)
        pred_fake = self.decoder(concate_feature_fake.reshape(b, -1))
        y1 = torch.ones_like(pred).mean(dim=-1).type(torch.LongTensor).to(pred.device)
        y2 = torch.zeros_like(pred).mean(dim=-1).type(torch.LongTensor).to(pred.device)
        loss = self.criterion(pred, y1) + self.criterion(pred_fake, y2)
        deceive_loss = self.criterion2(concate_feature, concate_feature_fake)
        return {"loss": loss, "deceive_loss": deceive_loss}

    def get_loss_sep(self, input_dict, res_set, flag=False):
        log_x          = input_dict["log_features"]
        kpi_x          = input_dict["kpi_features"]
        log_x_fake     = res_set["output"][0]
        kpi_x_fake     = res_set["output"][1]
        trace_nodes, trace_adj = self._get_real_trace_inputs(input_dict)
        b, _, _ = kpi_x.shape
        kpi_re, log_re, concate_feature_, _ = self.encoder(log_x, kpi_x, trace_nodes, trace_adj)
        concate_feature = self.fc(concate_feature_.reshape(b, -1))
        pred = self.decoder_fuse(concate_feature)
        kpi_re_fake, log_re_fake, concate_feature_fake_, _ = self.encoder(
            log_x_fake, kpi_x_fake, trace_nodes, trace_adj)
        concate_feature_fake = self.fc(concate_feature_fake_.reshape(b, -1))
        pred_fake = self.decoder_fuse(concate_feature_fake)
        y1 = torch.ones_like(pred).mean(dim=-1).type(torch.LongTensor).to(pred.device)
        y2 = torch.zeros_like(pred).mean(dim=-1).type(torch.LongTensor).to(pred.device)
        loss = self.criterion(pred, y1) + self.criterion(pred_fake, y2)
        deceive_loss = self.criterion2(concate_feature, concate_feature_fake)
        return {"loss": loss, "deceive_loss": deceive_loss}

    def get_loss(self, input_dict, res_set, flag=False):
        log_x      = input_dict["log_features"]
        kpi_x      = input_dict["kpi_features"]
        log_x_fake = res_set["output"][0]
        kpi_x_fake = res_set["output"][1]
        b, _, _    = kpi_x.shape

        # REAL: dùng trace_adj thật
        trace_nodes, trace_adj = self._get_real_trace_inputs(input_dict)
        kpi_re, log_re, concate_feature, trace_re = self.encoder(
            log_x, kpi_x, trace_nodes, trace_adj)
        pred_log = self.decoder3(log_re.reshape(b, -1))
        pred_kpi = self.decoder4(kpi_re.reshape(b, -1))

        # CHANGE 5: FAKE dùng adj_hat từ Structure AE thay vì trace_adj
        adj_hat_4d = self._get_fake_trace_adj(input_dict, res_set)
        kpi_re_fake, log_re_fake, concate_feature_fake, trace_re_fake = self.encoder(
            log_x_fake, kpi_x_fake, trace_nodes, adj_hat_4d)
        pred_log_fake = self.decoder3(log_re_fake.reshape(b, -1))
        pred_kpi_fake = self.decoder4(kpi_re_fake.reshape(b, -1))

        y1 = torch.ones_like(pred_kpi).mean(dim=-1).type(torch.LongTensor).to(pred_kpi.device)
        y2 = torch.zeros_like(pred_kpi).mean(dim=-1).type(torch.LongTensor).to(pred_kpi.device)

        loss = (self.criterion(pred_kpi,      y1) + self.criterion(pred_kpi_fake, y2) +
                self.criterion(pred_log,      y1) + self.criterion(pred_log_fake, y2))

        # Trace discrimination: giờ trace_re ≠ trace_re_fake (adj khác nhau)
        if trace_re is not None and trace_re_fake is not None:
            pred_trace      = self.decoder5(trace_re.reshape(b, -1))
            pred_trace_fake = self.decoder5(trace_re_fake.reshape(b, -1))
            loss = loss + self.criterion(pred_trace, y1) + self.criterion(pred_trace_fake, y2)

        deceive_loss = self.criterion2(kpi_re, kpi_re_fake) + self.criterion2(log_re, log_re_fake)
        if trace_re is not None and trace_re_fake is not None:
            deceive_loss = deceive_loss + self.criterion2(trace_re, trace_re_fake)

        # CHANGE 7: Attribute trace discrimination — error_rate (col 3) & latency_dev (col 5)
        # REAL: ground-truth node attributes | FAKE: reconstructed by attribute decoder
        if (self.open_trace
                and trace_nodes is not None
                and "feats_hat" in res_set
                and res_set["feats_hat"] is not None):
            feats_hat_4d = res_set["feats_hat"]          # [B, W, N, 2]
            W = feats_hat_4d.shape[1]

            # mean over N services → [B, W, 2] → [B*W, 2]
            attr_real = (trace_nodes[:, :, :, [3, 5]]    # [B, W, N, 2]
                         .mean(dim=2)                    # [B, W, 2]
                         .reshape(b * W, 2))             # [B*W, 2]
            attr_fake = (feats_hat_4d
                         .mean(dim=2)                    # [B, W, 2]
                         .reshape(b * W, 2))             # [B*W, 2]

            # logits → BCEWithLogitsLoss (numerically stable, no manual sigmoid)
            logits_real = self.attr_classifier(attr_real)   # [B*W, 1]
            logits_fake = self.attr_classifier(attr_fake)   # [B*W, 1]

            attr_disc_loss = (
                F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real)) +
                F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))
            )
            attr_deceive_loss = self.criterion2(logits_real, logits_fake)

            loss         = loss + attr_disc_loss
            deceive_loss = deceive_loss + attr_deceive_loss

        return {"loss": loss, "deceive_loss": deceive_loss}


# ═══════════════════════════════════════════════════════════════════════════════
# CHANGE 2, 3, 4: MultiModel
#   CHANGE 2: Decoder nhận cat([fused_modal, ZV]) thay vì concat_feature từ encoder
#   CHANGE 3: Return thêm adj_hat [B,W,N,N] để Discriminator dùng
#   CHANGE 4: trace_weight (fixed) → variance-based alpha (dynamic, no nn.Parameter)
# ═══════════════════════════════════════════════════════════════════════════════
class MultiModel(nn.Module):
    def __init__(self, var_nums, device, fuse_type="cross_attn", **kwargs):
        super(MultiModel, self).__init__()
        self.fuse_type   = fuse_type
        self.hidden_size = kwargs["hidden_size"]
        self.encoder     = MultiEncoder(var_nums=var_nums, device=device,
                                        fuse_type=fuse_type, **kwargs)
        self.log_c    = kwargs["log_c"]
        self.kpi_c    = kwargs["kpi_c"]
        self.unmatch_k = kwargs["unmatch_k"] * 0.01

        self.open_trace = kwargs.get("open_trace", False)

        # Base decoder: log+KPI only (fused_modal 2H → H → kpi_c+log_c)
        # Khi open_trace=True, nhánh trace được cộng qua residual-gated head (CHANGE 8).
        H = kwargs["hidden_size"]
        self.fuse_decoder = nn.Sequential(
            nn.Linear(2 * H, H),
            nn.ReLU(),
            nn.Linear(H, self.kpi_c + self.log_c),
        )

        if kwargs["criterion"] == "l1":
            self.criterion1 = nn.L1Loss()
            self.criterion2 = nn.L1Loss(reduction='none')
        else:
            self.criterion1 = nn.MSELoss()
            self.criterion2 = nn.MSELoss(reduction='none')
        self.criterion3 = nn.MSELoss()
        self.criterion4 = nn.L1Loss()

        self.narrow_modal_gap  = ReturnSelf()       if kwargs["open_narrowing_modal_gap"] else Return0()
        self.expand_anomaly_gap = ReturnTopXWeight() if kwargs["open_expand_anomaly_gap"]  else Return1()

        if self.open_trace:
            self.trace_model = TraceModel(device, **kwargs)

        # ── Residual-gated trace fusion (CHANGE 8) ──────────────────────────────
        # y = fuse_decoder(fm) + g * Δ_trace, g∈[0,1] per-sample từ trace-quality.
        # Δ-head zero-init + L1 trên g → khởi điểm chính xác bằng baseline log+KPI;
        # gate chỉ mở khi có gradient ủng hộ (trace thực sự giảm reconstruction loss).
        self.gate_lambda = float(kwargs.get("gate_lambda", 0.01))
        if self.open_trace:
            # Δ-head: nhận cat([fm, ZV]) = 3H, zero-init lớp cuối để khởi điểm Δ≈0
            self.delta_head = nn.Sequential(
                nn.Linear(3 * H, 2 * H), nn.ReLU(),
                nn.Linear(2 * H, self.kpi_c + self.log_c),
            )
            nn.init.zeros_(self.delta_head[-1].weight)
            nn.init.zeros_(self.delta_head[-1].bias)
            # Gate MLP: 6 trace-quality features → g ∈ [0,1]
            self.trace_gate = nn.Sequential(
                nn.Linear(6, 16), nn.ReLU(),
                nn.Linear(16, 1),
            )
            # Bias âm để g khởi điểm ≈ 0.12 (gate đóng) — chỉ mở khi có gradient ủng hộ
            nn.init.zeros_(self.trace_gate[-1].weight)
            nn.init.constant_(self.trace_gate[-1].bias, -2.0)

    @staticmethod
    def _trace_quality_feats(trace_nodes, trace_adj):
        """Compute 6 per-(B,W) scalar features describing trace quality / informativeness.
        trace_nodes: [B,W,N,C>=6] — cols: [0]call_count, [3]error_rate, [5]latency_dev
        trace_adj:   [B,W,N,N]
        Returns: [B, W, 6]
        """
        B, W, N, C = trace_nodes.shape
        call_count   = trace_nodes[..., 0]                          # [B,W,N]
        mean_calls   = call_count.mean(dim=-1)                      # [B,W]
        coverage     = (call_count > 0).float().mean(dim=-1)        # [B,W]
        if C > 3:
            err_rate = trace_nodes[..., 3].mean(dim=-1)             # [B,W]
        else:
            err_rate = torch.zeros_like(mean_calls)
        if C > 5:
            lat_dev  = trace_nodes[..., 5].abs().mean(dim=-1)       # [B,W]
        else:
            lat_dev  = torch.zeros_like(mean_calls)
        adj_density  = trace_adj.float().mean(dim=(-1, -2))         # [B,W]
        call_var     = call_count.var(dim=-1, unbiased=False)       # [B,W]
        feats = torch.stack([mean_calls, coverage, err_rate, lat_dev,
                             adj_density, call_var], dim=-1)        # [B,W,6]
        # robust normalize: log1p cho các đại lượng count-like
        feats = torch.stack([
            torch.log1p(feats[..., 0]),
            feats[..., 1],
            feats[..., 2],
            torch.log1p(feats[..., 3]),
            feats[..., 4],
            torch.log1p(feats[..., 5]),
        ], dim=-1)
        return feats

    def forward(self, input_dict, flag=False):
        trace_nodes = input_dict.get("trace_node_features", None)
        trace_adj   = input_dict.get("trace_adj", None)

        # ── Encoder ────────────────────────────────────────────────────────────
        # fused_modal: [B,W,2H]  (log+KPI self-attention, không có trace)
        # ZV:          [B,W,H]   (trace GAT output) hoặc None
        fused_kpi, fused_log, fused_modal, ZV = self.encoder(
            input_dict["log_features"], input_dict["kpi_features"], trace_nodes, trace_adj)

        fused_kpi_unmatched, fused_log_unmatched, fused_modal_unmatched, ZV_unmatched = self.encoder(
            input_dict["log_features"], input_dict["unmatched_kpi_features"], trace_nodes, trace_adj)

        # CHANGE 8: Residual-gated decoder — luôn bật khi open_trace=True.
        #   y = fuse_decoder(fm) + g * delta_head(cat[fm, zv])
        # Δ-head zero-init + bias gate âm → khởi điểm y ≡ baseline log+KPI.
        gate_g = None
        if self.open_trace and ZV is not None and trace_nodes is not None:
            q_feats = self._trace_quality_feats(trace_nodes, trace_adj)   # [B,W,6]
            gate_g  = torch.sigmoid(self.trace_gate(q_feats))             # [B,W,1] ∈ (0,1)

        def _decode(fm, zv):
            y_base = self.fuse_decoder(fm)                                # [B,W,kpi_c+log_c]
            if zv is not None and gate_g is not None:
                delta = self.delta_head(torch.cat([fm, zv], dim=-1))      # [B,W,kpi_c+log_c]
                return y_base + gate_g * delta
            return y_base

        fused_out           = _decode(fused_modal, ZV)
        fused_out_unmatched = _decode(fused_modal_unmatched, ZV_unmatched)

        kpi_out           = fused_out[:, :, :self.kpi_c]
        log_out           = fused_out[:, :, self.kpi_c:]
        kpi_out_unmatched = fused_out_unmatched[:, :, :self.kpi_c]
        log_out_unmatched = fused_out_unmatched[:, :, self.kpi_c:]

        # Fake pass (cycle re-encode)
        fused_kpi_fake, fused_log_fake, fused_modal_fake, ZV_fake = self.encoder(
            log_out, kpi_out, trace_nodes, trace_adj)
        fused_out_fake = _decode(fused_modal_fake, ZV_fake)

        # ── Reconstruction losses ───────────────────────────────────────────────
        kpi_dis = self.criterion2(kpi_out, input_dict["kpi_features"]).mean(dim=-1)  # [B,W]
        log_dis = self.criterion2(log_out, input_dict["log_features"]).mean(dim=-1)  # [B,W]

        log_d = log_dis * self.expand_anomaly_gap(log_dis)
        kpi_d = kpi_dis * self.expand_anomaly_gap(kpi_dis)
        log_kpi_loss = log_d + kpi_d + self.narrow_modal_gap(torch.abs(log_d - kpi_d))

        loss = log_dis.mean() + kpi_dis.mean()

        # ── Trace Structure Autoencoder ─────────────────────────────────────────
        trace_dis    = None
        adj_hat_4d   = None   # [B,W,N,N] — để Discriminator dùng làm "fake trace_adj"
        feats_hat_4d = None   # [B,W,N,2] — error_rate & latency_dev (CHANGE 7)

        if self.open_trace and trace_nodes is not None and trace_adj is not None:
            B, W, N, _ = trace_nodes.shape
            _, adj_hat, trace_dis_flat, feats_hat_slice = self.trace_model(
                trace_nodes.reshape(B * W, N, -1),
                trace_adj.reshape(B * W, N, N)
            )  # adj_hat: [B*W,N,N], trace_dis_flat: [B*W], feats_hat_slice: [B*W,N,2]

            trace_dis    = trace_dis_flat.reshape(B, W)           # [B, W]
            adj_hat_4d   = adj_hat.reshape(B, W, N, N)            # [B, W, N, N]  (CHANGE 3)
            feats_hat_4d = feats_hat_slice.reshape(B, W, N, 2)    # [B, W, N, 2]  (CHANGE 7)

            trace_d = trace_dis * self.expand_anomaly_gap(trace_dis)

            # Residual form: log_kpi_loss đã PHẢN ÁNH ảnh hưởng của gate (decoder output dùng g*Δ).
            # Thêm phần gated của trace_d + L1 regularizer để gate mặc định đóng.
            g2d = gate_g.squeeze(-1)                         # [B,W]
            fusion_loss = log_kpi_loss + g2d * trace_d + self.gate_lambda * g2d
            loss        = loss + (g2d * trace_dis).mean() + self.gate_lambda * g2d.mean()
        else:
            fusion_loss = log_kpi_loss

        # ── Contrastive loss ────────────────────────────────────────────────────
        if flag:
            loss += max(0, self.criterion4(input_dict["kpi_features"], kpi_out)
                           + self.unmatch_k
                           - self.criterion4(input_dict["unmatched_kpi_features"], kpi_out_unmatched))

        # ── Decoder input used as "concat_feature" for feature tracking ─────────
        concat_feature = (torch.cat([fused_modal, ZV], dim=-1)
                          if ZV is not None else fused_modal)

        dis_tuple = (log_dis, kpi_dis) if trace_dis is None else (log_dis, kpi_dis, trace_dis)

        return {
            "fusion_loss": fusion_loss,
            "loss":        loss,
            "dis":         dis_tuple,
            "features":    (fused_log, fused_kpi, concat_feature),
            "output":      (log_out, kpi_out),
            "adj_hat":     adj_hat_4d,    # CHANGE 3: [B,W,N,N] hoặc None
            "feats_hat":   feats_hat_4d,  # CHANGE 7: [B,W,N,2] hoặc None
        }
