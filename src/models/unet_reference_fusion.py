import copy
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroLinear(nn.Linear):
    def reset_parameters(self):
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class PerHeadSVCMAICG(nn.Module):
    def __init__(self, heads: int, dim_head: int, num_summary_tokens: int = 16, mlp_hidden: int = 32):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.num_summary_tokens = num_summary_tokens
        self.rgb_summary_tokens = nn.Parameter(torch.randn(heads, num_summary_tokens, dim_head) * 0.02)
        self.thermal_summary_tokens = nn.Parameter(torch.randn(heads, num_summary_tokens, dim_head) * 0.02)
        self.local_mlp = nn.Sequential(
            nn.LayerNorm(9),
            nn.Linear(9, mlp_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(mlp_hidden, 1),
        )
        self.global_mlp = nn.Sequential(
            nn.LayerNorm(9),
            nn.Linear(9, mlp_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(mlp_hidden, 1),
        )

    def _summarize(self, keys, summary_tokens):
        # keys: [B, H, N, D], summary tokens: [H, S, D]
        tokens = summary_tokens.unsqueeze(0).to(dtype=keys.dtype, device=keys.device)
        logits = torch.einsum("bhsd,bhnd->bhsn", tokens, keys) / math.sqrt(self.dim_head)
        weights = torch.softmax(logits, dim=-1)
        return torch.einsum("bhsn,bhnd->bhsd", weights, keys)

    def forward(self, q_t, k_rgb, k_thermal):
        # q_t/k_*: [B, H, N, D]
        s_rgb = self._summarize(k_rgb, self.rgb_summary_tokens)
        s_thermal = self._summarize(k_thermal, self.thermal_summary_tokens)

        q = F.normalize(q_t, dim=-1)
        s_rgb = F.normalize(s_rgb, dim=-1)
        s_thermal = F.normalize(s_thermal, dim=-1)

        c_cross = torch.einsum("bhnd,bhsd->bhns", q, s_rgb)
        c_self = torch.einsum("bhnd,bhsd->bhns", q, s_thermal)
        c_diff = c_cross - c_self

        local_stats = torch.cat([_local_stats(c_cross), _local_stats(c_self), _local_stats(c_diff)], dim=-1)
        g_local = torch.sigmoid(self.local_mlp(local_stats))

        global_stats = torch.cat([_global_stats(c_cross), _global_stats(c_self), _global_stats(c_diff)], dim=-1)
        g_global = torch.sigmoid(self.global_mlp(global_stats)).view(q.shape[0], 1, 1, 1)
        return g_local * g_global


class SVReferenceAttention(nn.Module):
    def __init__(self, attn, num_summary_tokens: int = 16):
        super().__init__()
        self.heads = attn.heads
        self.inner_dim = attn.inner_dim
        self.dim_head = self.inner_dim // self.heads
        self.to_q = copy.deepcopy(attn.to_q)
        self.to_k_t = copy.deepcopy(attn.to_k)
        self.to_k_rgb = copy.deepcopy(attn.to_k)
        self.to_v_rgb = copy.deepcopy(attn.to_v)
        self.to_out = copy.deepcopy(attn.to_out)
        self.gate = PerHeadSVCMAICG(self.heads, self.dim_head, num_summary_tokens=num_summary_tokens)
        self.zero_linear = ZeroLinear(self.inner_dim, self.inner_dim)

    def forward(self, thermal_hidden, rgb_hidden):
        q = self.to_q(thermal_hidden)
        k_thermal = self.to_k_t(thermal_hidden)
        k_rgb = self.to_k_rgb(rgb_hidden)
        v_rgb = self.to_v_rgb(rgb_hidden)

        q = self._split_heads(q)
        k_rgb = self._split_heads(k_rgb)
        v_rgb = self._split_heads(v_rgb)
        k_thermal = self._split_heads(k_thermal)

        rgb = F.scaled_dot_product_attention(q, k_rgb, v_rgb, dropout_p=0.0, is_causal=False)
        rgb = rgb * self.gate(q, k_rgb, k_thermal)
        rgb = self._merge_heads(rgb)

        for layer in self.to_out:
            rgb = layer(rgb)
        return self.zero_linear(rgb)

    def _split_heads(self, x):
        b, n, _ = x.shape
        return x.view(b, n, self.heads, self.dim_head).transpose(1, 2)

    def _merge_heads(self, x):
        b, h, n, d = x.shape
        return x.transpose(1, 2).reshape(b, n, h * d)


class ReferenceWriteBlock(nn.Module):
    def __init__(self, block, bank: List[torch.Tensor]):
        super().__init__()
        self.block = block
        self.bank = bank

    def forward(self, hidden_states, *args, **kwargs):
        self.bank.append(hidden_states.detach())
        return self.block(hidden_states, *args, **kwargs)


class ReferenceReadBlock(nn.Module):
    def __init__(self, block, bank: List[torch.Tensor], index: int, num_summary_tokens: int = 16):
        super().__init__()
        self.block = block
        self.bank = bank
        self.index = index
        self.ref_attn = SVReferenceAttention(block.attn1, num_summary_tokens=num_summary_tokens)

    def forward(self, hidden_states, *args, **kwargs):
        if self.index < len(self.bank):
            rgb_hidden = self.bank[self.index].to(device=hidden_states.device, dtype=hidden_states.dtype)
            if rgb_hidden.shape == hidden_states.shape:
                hidden_states = hidden_states + self.ref_attn(hidden_states, rgb_hidden)
        return self.block(hidden_states, *args, **kwargs)


def install_unet_reference_fusion(unet, ref_unet, bank: List[torch.Tensor], num_summary_tokens: int = 16):
    main_paths = _transformer_block_paths(unet)
    ref_paths = _transformer_block_paths(ref_unet)
    if len(main_paths) != len(ref_paths):
        raise ValueError(f"Main/ref UNet transformer counts differ: {len(main_paths)} vs {len(ref_paths)}")
    wrappers = []
    for index, (main_path, ref_path) in enumerate(zip(main_paths, ref_paths)):
        main_block = _get_module(unet, main_path)
        ref_block = _get_module(ref_unet, ref_path)
        read = ReferenceReadBlock(main_block, bank, index, num_summary_tokens=num_summary_tokens)
        write = ReferenceWriteBlock(ref_block, bank)
        _set_module(unet, main_path, read)
        _set_module(ref_unet, ref_path, write)
        wrappers.append(read)
    return wrappers, main_paths


def _transformer_block_paths(unet) -> List[str]:
    paths = []
    for name, module in unet.named_modules():
        if module.__class__.__name__ == "BasicTransformerBlock":
            paths.append(name)
    return sorted(paths, key=_execution_order_key)


def _execution_order_key(path: str):
    parts = path.split(".")
    if parts[0] == "down_blocks":
        return (0, int(parts[1]), int(parts[3]), path)
    if parts[0] == "mid_block":
        return (1, 0, int(parts[2]), path)
    if parts[0] == "up_blocks":
        return (2, int(parts[1]), int(parts[3]), path)
    return (3, path)


def _get_module(root, path: str):
    module = root
    for part in path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def _set_module(root, path: str, module):
    parts = path.split(".")
    parent = _get_module(root, ".".join(parts[:-1])) if len(parts) > 1 else root
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = module
    else:
        setattr(parent, last, module)


def _local_stats(x):
    return torch.stack([x.mean(dim=-1), x.max(dim=-1).values, x.std(dim=-1, unbiased=False)], dim=-1)


def _global_stats(x):
    return torch.stack([x.mean(dim=(1, 2, 3)), x.amax(dim=(1, 2, 3)), x.std(dim=(1, 2, 3), unbiased=False)], dim=-1)
