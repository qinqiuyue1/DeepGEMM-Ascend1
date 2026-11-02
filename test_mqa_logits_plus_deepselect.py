"""
Helper functions for the QLI mqa_logits + deep_select end-to-end test.

This module is imported by QLI.py. It provides:
- quantize_mqa_tensor: quantize a bf16 tensor to FP8 e4m3 or FP4 + UE8M0 SF.
- ref_mqa_logits: golden bf16/fp8 reference for the MQA logits kernel.
- ref_topk_per_row: golden topk-per-row reference using torch.topk, respecting
  per-row [ks, ke) ranges.
"""

import torch


# ---------------------------------------------------------------------------
# Quantization helpers (ported from DeepGEMM-Ascend tests/test_attention.py)
# ---------------------------------------------------------------------------

def ceil_to_ue8m0_with_byte(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Round x's magnitude up to UE8M0 (power of two) and also return the byte exponent."""
    bits = x.abs().float().view(torch.int)
    exp = (((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()).clamp(1, 254)
    return (exp << 23).view(torch.float), exp.to(torch.uint8)


def quantize_to_fp4_raw(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize to FP4 (E2M1) packed as int8 (two elements per byte) and the dequantized f32."""
    ax = x.abs()
    code = torch.zeros_like(x, dtype=torch.uint8)
    for boundary in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        code += (ax > boundary).to(torch.uint8)
    sign = (x < 0) & (code != 0)
    values = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), device=x.device, dtype=torch.float32)
    repr_f32 = values[code.long()]
    repr_f32 = torch.where(sign, -repr_f32, repr_f32)
    code = code | (sign.to(torch.uint8) * 8)
    packed = (code[..., ::2] & 0x0f) | ((code[..., 1::2] & 0x0f) << 4)
    return packed.contiguous().view(torch.int8), repr_f32


def quantize_mqa_tensor(x: torch.Tensor, use_fp4: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize a bf16 tensor to FP8 e4m3 (default) or FP4 (E2M1).

    Returns:
        x_quant:  quantized tensor (uint8 for fp8, int8-packed for fp4)
        x_sf:     scaling factors packed as int16 (UE8M0 pairs), MN-major
        x_ref:    dequantized f32 reference of the *quantized* tensor
    """
    sf_div = 6.0 if use_fp4 else 448.0
    sf_shape = (*x.shape[:-1], x.shape[-1] // 32, 32)
    sf, sf_byte = ceil_to_ue8m0_with_byte(x.float().reshape(sf_shape).abs().amax(dim=-1).clamp(1e-4) / sf_div)
    x_scaled = (x.float().reshape(sf_shape) / sf[..., None]).reshape_as(x)
    if use_fp4:
        x_quant, x_repr = quantize_to_fp4_raw(x_scaled)
    else:
        # FP8 e4m3 quantization: keep the float8_e4m3fn dtype (do NOT view as uint8)
        x_quant = x_scaled.to(torch.float8_e4m3fn)
        x_repr = x_quant.float()
    x_ref = (x_repr.reshape(sf_shape) * sf[..., None]).reshape_as(x)
    # Pair-adjacent SF bytes into int16 (UE8M0), as DeepGEMM-Ascend expects.
    x_sf = (sf_byte[..., 0::2].to(torch.int16) |
            (sf_byte[..., 1::2].to(torch.int16) << 8)).contiguous()
    return x_quant, x_sf, x_ref


# ---------------------------------------------------------------------------
# Reference MQA logits (golden)
# ---------------------------------------------------------------------------

def ref_mqa_logits(q: torch.Tensor, kv: torch.Tensor, weights: torch.Tensor,
                   cu_seq_len_k_start: torch.Tensor, cu_seq_len_k_end: torch.Tensor) -> torch.Tensor:
    """
    Reference mqa_logits in float32.

    Args:
        q:           [S, H, D] bf16/f32
        kv:          [Skv, D] bf16/f32 (single head, shared across H)
        weights:     [S, H] bf16/f32
        cu_seq_len_k_start: [S] int32
        cu_seq_len_k_end:   [S] int32

    Returns:
        logits: [S, Skv] bf16 (rows padded with -inf outside [ks, ke))
    """
    seq_len, num_heads, head_dim = q.shape
    seq_len_kv = kv.shape[0]
    device = q.device

    q_f32 = q.float()
    kv_f32 = kv.float()
    w_f32 = weights.float()

    # scores[s, h, k] = <q[s, h], kv[k]> ; then weighted-sum over h -> logits[s, k]
    # logits[s, k] = sum_h w[s, h] * <q[s, h], kv[k]>
    qk = torch.einsum('shd,kd->shk', q_f32, kv_f32)        # [S, H, Skv]
    weighted = (w_f32.unsqueeze(-1) * qk).sum(dim=1)       # [S, Skv]

    # Mask out positions outside [ks, ke) for each row.
    ks = cu_seq_len_k_start.long()
    ke = cu_seq_len_k_end.long()
    arange_kv = torch.arange(seq_len_kv, device=device).unsqueeze(0)   # [1, Skv]
    mask = (arange_kv >= ks.unsqueeze(1)) & (arange_kv < ke.unsqueeze(1))  # [S, Skv]
    logits = weighted.masked_fill(~mask, float('-inf'))
    return logits.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Reference per-row topk (golden) for deep_select comparison
# ---------------------------------------------------------------------------

def ref_topk_per_row(x: torch.Tensor,
                     cu_seq_len_k_start: torch.Tensor,
                     cu_seq_len_k_end: torch.Tensor,
                     topk: int):
    """
    Reference top-k per row, respecting per-row [ks, ke) valid ranges.

    For each row i, only positions [ks_i, ke_i) are considered. We replace
    out-of-range entries with -inf so torch.topk ignores them.

    Args:
        x: [S, N] bfloat16
        cu_seq_len_k_start, cu_seq_len_k_end: [S] int32
        topk: int

    Returns:
        values:  [S, topk] float32 (sorted descending within each row)
        indices: [S, topk] int64  (positions in [0, N))
    """
    s, n = x.shape
    device = x.device
    ks = cu_seq_len_k_start.long()
    ke = cu_seq_len_k_end.long()

    arange_n = torch.arange(n, device=device).unsqueeze(0)   # [1, N]
    mask = (arange_n >= ks.unsqueeze(1)) & (arange_n < ke.unsqueeze(1))  # [S, N]
    masked = x.float().masked_fill(~mask, float('-inf'))

    k = min(topk, n)
    values, indices = torch.topk(masked, k=k, dim=-1, sorted=True)
    return values, indices
