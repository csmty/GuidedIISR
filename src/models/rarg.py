import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroLinear(nn.Linear):
    def reset_parameters(self):
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class SummaryTokenPool(nn.Module):
    def __init__(self, channels, num_summary_tokens=4, num_heads=4):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}")
        self.summary_tokens = nn.Parameter(torch.randn(num_summary_tokens, channels) * 0.02)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        tokens, spatial_shape = _flatten_feature(x)
        b = tokens.shape[0]
        query = self.summary_tokens.unsqueeze(0).expand(b, -1, -1)
        summary, _ = self.attn(query, tokens, tokens, need_weights=False)
        return self.norm(summary), spatial_shape


class RARG(nn.Module):
    """Spatially variant cross-modal adaptive information control gate.

    The module compares thermal self consistency with thermal-visible cross
    consistency and returns a gate for RARG reference attention residuals.
    Inputs can be image-like features [B, C, H, W] or token features [B, N, C].
    """

    def __init__(
        self,
        channels,
        num_summary_tokens=4,
        num_heads=4,
        mlp_hidden_channels=None,
        gate_channels=1,
        eps=1e-6,
    ):
        super().__init__()
        self.channels = channels
        self.gate_channels = gate_channels
        self.eps = eps
        hidden = mlp_hidden_channels or max(channels // 2, 32)
        self.thermal_pool = SummaryTokenPool(channels, num_summary_tokens, num_heads)
        self.visible_pool = SummaryTokenPool(channels, num_summary_tokens, num_heads)
        self.local_mlp = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, gate_channels),
        )
        self.global_mlp = nn.Sequential(
            nn.LayerNorm(8),
            nn.Linear(8, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, gate_channels),
        )

    def forward(self, thermal_feature, visible_feature):
        thermal_tokens, spatial_shape = _flatten_feature(thermal_feature)
        visible_tokens, _ = _flatten_feature(visible_feature)
        if thermal_tokens.shape[-1] != self.channels or visible_tokens.shape[-1] != self.channels:
            raise ValueError(
                f"RARG expects channel dim {self.channels}, "
                f"got thermal={thermal_tokens.shape[-1]}, visible={visible_tokens.shape[-1]}"
            )

        s_t, _ = self.thermal_pool(thermal_tokens)
        s_v, _ = self.visible_pool(visible_tokens)
        q_t = F.normalize(thermal_tokens, dim=-1, eps=self.eps)
        s_t = F.normalize(s_t, dim=-1, eps=self.eps)
        s_v = F.normalize(s_v, dim=-1, eps=self.eps)

        c_cross = torch.einsum("bnc,bsc->bns", q_t, s_v)
        c_self = torch.einsum("bnc,bsc->bns", q_t, s_t)
        cross_mean = c_cross.mean(dim=-1)
        cross_max = c_cross.max(dim=-1).values
        self_mean = c_self.mean(dim=-1)
        diff = cross_mean - self_mean
        local_stats = torch.stack([cross_mean, cross_max, self_mean, diff], dim=-1)
        local_gate = torch.sigmoid(self.local_mlp(local_stats))

        global_stats = torch.stack(
            [
                c_cross.mean(dim=(1, 2)),
                c_cross.max(dim=2).values.mean(dim=1),
                c_cross.std(dim=(1, 2), unbiased=False),
                c_self.mean(dim=(1, 2)),
                c_self.max(dim=2).values.mean(dim=1),
                c_self.std(dim=(1, 2), unbiased=False),
                diff.mean(dim=1),
                diff.std(dim=1, unbiased=False),
            ],
            dim=-1,
        )
        global_gate = torch.sigmoid(self.global_mlp(global_stats)).unsqueeze(1)
        gate = local_gate * global_gate
        return _restore_gate(gate, thermal_feature, spatial_shape)


class GatedReferenceAttention(nn.Module):
    """Reference attention with RARG and zero-initialized output projection."""

    def __init__(self, channels, num_heads=8, num_summary_tokens=4, gate_channels=1):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}")
        self.norm_thermal = nn.LayerNorm(channels)
        self.norm_visible = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.zero_out = ZeroLinear(channels, channels)
        self.gate = RARG(
            channels=channels,
            num_summary_tokens=num_summary_tokens,
            num_heads=num_heads,
            gate_channels=gate_channels,
        )

    def forward(self, thermal_feature, visible_feature):
        thermal_tokens, spatial_shape = _flatten_feature(thermal_feature)
        visible_tokens, _ = _flatten_feature(visible_feature)
        q = self.norm_thermal(thermal_tokens)
        kv = self.norm_visible(visible_tokens)
        delta, _ = self.attn(q, kv, kv, need_weights=False)
        delta = self.zero_out(delta)
        delta = _restore_tokens(delta, thermal_feature, spatial_shape)
        gate = self.gate(thermal_feature, visible_feature)
        return thermal_feature + gate * delta


def _flatten_feature(x):
    if x.dim() == 4:
        b, c, h, w = x.shape
        return x.flatten(2).transpose(1, 2), (h, w)
    if x.dim() == 3:
        return x, None
    raise ValueError(f"Expected feature shape [B,C,H,W] or [B,N,C], got {tuple(x.shape)}")


def _restore_tokens(tokens, reference, spatial_shape):
    if reference.dim() == 4:
        h, w = spatial_shape
        return tokens.transpose(1, 2).reshape(reference.shape[0], reference.shape[1], h, w)
    return tokens


def _restore_gate(gate, reference, spatial_shape):
    if reference.dim() == 4:
        h, w = spatial_shape
        return gate.transpose(1, 2).reshape(reference.shape[0], gate.shape[-1], h, w)
    return gate
