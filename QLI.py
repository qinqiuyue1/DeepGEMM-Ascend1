#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Benchmark execution script: mqa_logits (DeepGEMM-Ascend) + deep_select (TopK),
driven by quant_lightning_indexer (QLI) test shapes.

This script ONLY constructs inputs and runs the pipeline. It does NOT do its
own timing / msprof -- wrap it externally, e.g.:

    msprof python bench_mqa_logits_plus_deepselect.py --s2 4096 --s1 1024

Kernels to match in msprof:
    - mqa_logits:   'mqa_logits_impl'      (DeepGEMM-Ascend)
    - deep_select:  'topk_select_kernel'   (DeepSelect, kernel.asc:923)

QLI shape mapping (path A: full-range scoring, no candidate-block semantics):
    B=1, NQ=32, D=128, TOPK=512, MASK_MODE=3 (rightDownCausal), CMP_RATIO=2
    -> num_heads=32, head_dim=128, S=S1, Skv=S2 (compressed KV length)
    -> kv: [S2, 128] (k_head=1 shared)
    -> ks=0, ke[i] = clamp(i + (S2 - S1), max=S2)   (rightDownCausal)
    -> deep_select.topk(logits, 512, end=ke-ks)

Run outside the DeepGEMM-Ascend source tree so `deep_gemm` resolves to the
installed site-packages build.
"""

import argparse
import sys
import os

import torch

# Reuse quantization / golden / check helpers from the existing e2e test.
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

try:
    import deep_select
except ImportError as e:
    raise ImportError("deep_select is required. Install DeepSelect (Ascend build) first.") from e


# QLI test-shape table: S2 (compressed KV len) -> list of S1 (query len).
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


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    """Cosine-similarity based diff, consistent with DeepGEMM (tests/utils.py:36)."""
    x, y = x.float(), y.float()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


def run_one(s1: int, s2: int, use_fp4: bool, iters: int, do_check: bool,
            seed: int, verbose: bool) -> bool:
    """Run the mqa_logits -> deep_select pipeline once (or `iters` times).

    Returns True if (when do_check) all checks passed, else False.
    """
    torch.manual_seed(seed)
    device = 'npu'

    # ------------------------------------------------------------------
    # 1. Inputs: q [S1, 32, 128], kv [S2, 128] (MQA shared), w [S1, 32].
    # ------------------------------------------------------------------
    q_bf16 = torch.randn(s1, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    kv_bf16 = torch.randn(s2, HEAD_DIM, device=device, dtype=torch.bfloat16)
    weights = torch.randn(s1, NUM_HEADS, device=device, dtype=torch.bfloat16)

    # rightDownCausal: row i attends to [0, i + (S2 - S1) + 1), clamped to S2.
    cu_seq_len_k_start = torch.zeros(s1, device=device, dtype=torch.int32)
    cu_seq_len_k_end = (torch.arange(s1, device=device, dtype=torch.int32)
                        + (s2 - s1) + 1).clamp(max=s2)
    valid_len = cu_seq_len_k_end - cu_seq_len_k_start   # [S1] int32
    max_seqlen_k = int(valid_len.max().item())

    # ------------------------------------------------------------------
    # 2. MX quantization (FP8 e4m3 by default, or FP4).
    # ------------------------------------------------------------------
    q_quant, q_sf, q_ref = quantize_mqa_tensor(q_bf16, use_fp4)
    kv_quant, kv_sf, kv_ref = quantize_mqa_tensor(kv_bf16, use_fp4)

    # ------------------------------------------------------------------
    # 3. Stage 1: fp8_fp4_mqa_logits -> [S1, max_seqlen_k] compressed rows.
    # ------------------------------------------------------------------
    def run_pipeline():
        mqa_out = deep_gemm.fp8_fp4_mqa_logits(
            (q_quant, q_sf), (kv_quant, kv_sf), weights,
            cu_seq_len_k_start, cu_seq_len_k_end, max_seqlen_k)
        # Each row's valid length is valid_len[i]; deep_select handles per-row
        # variable length via `end`.
        _, indices = deep_select.topk(
            mqa_out, TOPK,
            sorted=False,
            end=valid_len.to(torch.int32),
            indices_type=torch.int32,
            sorted_index=False,
            return_value=True,
            abort_when_nan_found=False,
        )
        return mqa_out, indices

    mqa_out, indices = run_pipeline()
    for _ in range(1, iters):
        mqa_out, indices = run_pipeline()

    if verbose:
        print(f"[shape] S1={s1}, S2={s2}, H={NUM_HEADS}, D={HEAD_DIM}, "
              f"topk={TOPK}, fp4={use_fp4}")
        print(f"[shape] mqa_out={tuple(mqa_out.shape)}, "
              f"indices={tuple(indices.shape)} {indices.dtype}")

    # ------------------------------------------------------------------
    # 4. Optional correctness check (small shapes only in practice).
    # ------------------------------------------------------------------
    if not do_check:
        return True

    # Expand compressed mqa rows to full width for comparison.
    mqa_full = torch.full((s1, s2), float('-inf'), device=device, dtype=torch.bfloat16)
    ks_list = cu_seq_len_k_start.cpu().tolist()
    ke_list = cu_seq_len_k_end.cpu().tolist()
    for i in range(s1):
        ks, ke = ks_list[i], ke_list[i]
        if ke > ks:
            mqa_full[i, ks:ke] = mqa_out[i, :ke - ks]

    # L1: cosine diff vs ideal bf16 reference.
    ref_full_ideal = ref_mqa_logits(q_bf16, kv_bf16, weights,
                                    cu_seq_len_k_start, cu_seq_len_k_end)
    ref_full_sim = ref_mqa_logits(q_ref, kv_ref, weights,
                                  cu_seq_len_k_start, cu_seq_len_k_end)
    neginf = torch.isinf(ref_full_ideal.float()) & (ref_full_ideal.float() < 0)
    orig_diff = calc_diff(mqa_full.masked_fill(neginf, 0),
                          ref_full_ideal.masked_fill(neginf, 0))
    sim_diff = calc_diff(mqa_full.masked_fill(neginf, 0),
                         ref_full_sim.masked_fill(neginf, 0))
    ORIG_TOL = 2e-2 if use_fp4 else 1e-3
    SIM_TOL = 5e-4
    l1_ok = (orig_diff < ORIG_TOL) or (sim_diff < SIM_TOL)

    # L2: deep_select index sets vs ref topk on the same mqa_full (100% expected).
    ref_vals_on_mqa, ref_idx_on_mqa = ref_topk_per_row(
        mqa_full, cu_seq_len_k_start, cu_seq_len_k_end, TOPK)
    ds_set_ok = True
    for i in range(s1):
        n_valid = min(TOPK, ke_list[i] - ks_list[i])
        if n_valid <= 0:
            continue
        ds_row = indices[i, :n_valid].cpu().tolist()
        ref_row = ref_idx_on_mqa[i, :n_valid].cpu().tolist()
        if set(ds_row) != set(ref_row):
            ds_set_ok = False
            break
    l2_ok = ds_set_ok

    ok = l1_ok and l2_ok
    print(f"[check] S1={s1}, S2={s2}: L1 orig={orig_diff:.4e}(tol {ORIG_TOL}) "
          f"sim={sim_diff:.4e}(tol {SIM_TOL}) -> {'OK' if l1_ok else 'FAIL'}, "
          f"L2 index-set={'OK' if l2_ok else 'FAIL'} -> "
          f"{'OK' if ok else 'MISMATCH'}")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description='QLI-shape mqa_logits + deep_select execution script '
                    '(timing is left to an external msprof wrapper)')
    parser.add_argument('--s1', type=int, default=None, help='query seq length')
    parser.add_argument('--s2', type=int, default=None, help='kv seq length (compressed)')
    parser.add_argument('--s2-all', action='store_true',
                        help='iterate over all (S2, S1) pairs in S2_S1_MAP')
    parser.add_argument('--fp4', action='store_true', help='use FP4 (default FP8)')
    parser.add_argument('--iters', type=int, default=1,
                        help='pipeline iterations per shape (default 1)')
    parser.add_argument('--check', action='store_true',
                        help='run correctness check after the pipeline')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    if args.s2_all:
        pairs = [(s2, s1) for s2, s1s in S2_S1_MAP.items() for s1 in s1s]
    elif args.s2 is not None and args.s1 is not None:
        pairs = [(args.s2, args.s1)]
    elif args.s2 is not None:
        pairs = [(args.s2, s1) for s1 in S2_S1_MAP[args.s2]]
    elif args.s1 is not None:
        pairs = [(s2, args.s1) for s2 in S2_S1_MAP]
    else:
        parser.error('specify --s1/--s2, or --s2-all')

    all_ok = True
    for s2, s1 in pairs:
        ok = run_one(s1, s2, args.fp4, args.iters, args.check, args.seed,
                     verbose=not args.quiet)
        all_ok = all_ok and ok

    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()