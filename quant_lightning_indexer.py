#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
QuantLightningIndexer: fused mqa_logits + topk operator.

This module wraps the fused kernel `deep_gemm.fp8_fp4_mqa_logits_topk` into a
single-call API. Internally, the fused kernel runs mqa_logits (Stage 1) and
topk (Stage 2) in a single kernel launch, producing (values, indices) that
are bit-identical to the original two-kernel pipeline:

    mqa_out = deep_gemm.fp8_fp4_mqa_logits(...)
    values, indices = deep_select.topk(mqa_out, ...)

Golden reference reuses `test_mqa_logits_plus_deepselect.py` helpers.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_mqa_logits_plus_deepselect import (
    quantize_mqa_tensor,
    ref_mqa_logits,
    ref_topk_per_row,
)

try:
    import deep_gemm
except ImportError as e:
    raise ImportError("deep_gemm is required. Install DeepGEMM-Ascend first.") from e


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    """Cosine-similarity based diff."""
    x, y = x.float(), y.float()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


# QLI test-shape table (same as QLI.py)
S2_S1_MAP = {
    512:    [128, 256, 512],
    1024:   [128, 256, 512, 1024],
    2048:   [128, 256, 512, 1024, 2048],
    4096:   [128, 256, 512, 1024, 2048, 4096],
    8192:   [128, 256, 512, 1024, 2048, 4096, 8192],
    16384:  [128, 256, 512, 1024, 2048, 4096, 8192],
    32768:  [128, 256, 512, 1024, 2048, 4096, 8192],
    65536:  [128, 256, 512, 1024, 2048, 4096, 8192],
}

NUM_HEADS = 32
HEAD_DIM = 128
TOPK = 512


class QuantLightningIndexer:
    """Fused mqa_logits + topk operator (single kernel launch).

    Produces (values, indices) bit-identical to the original two-kernel pipeline:
        mqa_out = deep_gemm.fp8_fp4_mqa_logits((q, q_sf), (kv, kv_sf), weights,
                                                cu_seq_len_k_start, cu_seq_len_k_end, max_seqlen_k)
        values, indices = deep_select.topk(mqa_out, topk, sorted=False, end=valid_len,
                                           indices_type=torch.int32, sorted_index=False,
                                           return_value=True, abort_when_nan_found=False)
    """

    def __init__(self,
                 num_heads: int = 32,
                 head_dim: int = 128,
                 topk: int = 512,
                 use_fp4: bool = False,
                 abort_when_nan_found: bool = False,
                 sorted_index: bool = False):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.topk = topk
        self.use_fp4 = use_fp4
        self.abort_when_nan_found = abort_when_nan_found
        self.sorted_index = sorted_index

    def forward(self,
                q_bf16: torch.Tensor,           # [S1, H, D]
                kv_bf16: torch.Tensor,          # [S2, D]
                weights: torch.Tensor,          # [S1, H]
                cu_seq_len_k_start: torch.Tensor,  # [S1] int32
                cu_seq_len_k_end: torch.Tensor    # [S1] int32
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            values:  [S1, topk] bf16
            indices: [S1, topk] int32
        """
        S1 = q_bf16.shape[0]
        max_seqlen_k = int((cu_seq_len_k_end - cu_seq_len_k_start).max().item())

        # Quantize (reuse existing helper)
        q_quant, q_sf, _ = quantize_mqa_tensor(q_bf16, self.use_fp4)
        kv_quant, kv_sf, _ = quantize_mqa_tensor(kv_bf16, self.use_fp4)

        # Single fused kernel call (returns values, indices, logits)
        result = deep_gemm.fp8_fp4_mqa_logits_topk(
            (q_quant, q_sf), (kv_quant, kv_sf), weights,
            cu_seq_len_k_start, cu_seq_len_k_end,
            max_seqlen_k, self.topk,
            return_values=True,
            abort_when_nan_found=self.abort_when_nan_found,
            sorted_index=self.sorted_index,
        )
        # result is (values, indices, logits)
        values, indices = result[0], result[1]
        logits = result[2] if len(result) > 2 else None
        return values, indices, logits

    __call__ = forward


def quant_lightning_indexer(q_bf16, kv_bf16, weights,
                             cu_seq_len_k_start, cu_seq_len_k_end,
                             num_heads=32, head_dim=128, topk=512, use_fp4=False):
    """Convenience function wrapper."""
    return QuantLightningIndexer(num_heads, head_dim, topk, use_fp4).forward(
        q_bf16, kv_bf16, weights, cu_seq_len_k_start, cu_seq_len_k_end)
