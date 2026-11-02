#pragma once

// mqa_logits_topk: fused mqa_logits + topk kernel.
//
// Stage 1 (AIC + AIV): runs the original mqa_logits_core_impl verbatim,
//   producing compressed logits rows in GM (same as standalone mqa_logits).
// Stage 2 (AIV only): after draining all MTE3 stores from Stage 1, each AIV
//   iterates over the query rows owned by this core and calls
//   qli_topk::topk_select_run on the GM logits row, writing (values, indices)
//   to the output GM buffers.
//
// Because Stage 1 is byte-identical to the standalone mqa_logits kernel and
// Stage 2 calls the same topk_select_run (ported verbatim from DeepSelect)
// on the same GM bytes with the same valid_len, the fused kernel produces
// bit-identical output to the original two-kernel pipeline.

#include <deep_gemm/ascend.hpp>
#include <deep_gemm/common.hpp>
#include <deep_gemm/mqa_logits.hpp>
#include <deep_gemm/QuantLightningIndexer.hpp>
#include <deep_gemm/scheduler/mqa_logits.hpp>

namespace deep_gemm {

template <
    bool kIsFP4,
    uint32_t kNumHeads, uint32_t kHeadDim,
    uint32_t kSplitKV, uint32_t kMadM, uint32_t kNumCDStages,
    uint32_t kNumQStages, uint32_t kNumKVStages, uint32_t kNumAccumStages,
    uint32_t kNumLogitsStages,
    uint32_t kNumCores,
    uint32_t kTopkTier,              // 512, 1024, or 4096
    bool kAbortWhenNanFound,
    bool kSortedIndex
>
__global__ __mix__(1, 2) void mqa_logits_topk_impl(
    uint32_t num_q_tokens, uint32_t num_kv_tokens,
    gm_ptr<uint8_t, Major::K> q, gm_ptr<uint8_t, Major::K> kv,
    __gm__ int16_t* sf_q, __gm__ int16_t* sf_kv,
    gm_ptr<bfloat16_t, Major::K> weights,
    gm_ptr<bfloat16_t, Major::K> logits,
    gm_ptr<bfloat16_t, Major::K> output_values,
    gm_ptr<int32_t, Major::K> output_indices,
    __gm__ uint32_t* cu_seq_len_k_start, __gm__ uint32_t* cu_seq_len_k_end,
    uint32_t logits_stride,           // in bf16 elements (row stride)
    uint32_t output_values_stride,    // in bf16 elements
    uint32_t output_indices_stride,   // in int32 elements
    uint32_t topk
) {
    asc_init();
    constexpr uint32_t kNumQTokensPerBlock = 8 * kMadM / kNumHeads;

    const auto make_scheduler = [&](uint32_t core_id) __aicore__ {
        return MQALogitsScheduler<kNumQTokensPerBlock, kSplitKV, kNumCores>(
            core_id, num_q_tokens, num_kv_tokens,
            cu_seq_len_k_start, cu_seq_len_k_end, kNumHeads);
    };

    // Stage 1: original mqa_logits_core_impl (AIC MMAD + AIV reduce/store).
    // Writes compressed logits to GM: row i has valid data at [0, ke_i - ks_i).
    mqa_logits_core_impl<
        kIsFP4, kNumHeads, kHeadDim, kSplitKV, kMadM,
        kNumCDStages, kNumQStages, kNumKVStages, kNumAccumStages,
        kNumLogitsStages, 0u, false, decltype(make_scheduler)>(
        q, kv, sf_q, sf_kv, weights, logits, make_scheduler);

    // Stage 2: per-row topk on the GM logits (AIV only).
    if ASCEND_IS_AIV {
        // Drain MTE3 stores: asc_sync_mte3 ensures MTE3 stores complete and
        // are visible to MTE2 and V. asc_sync drains remaining pipes (S, etc).
        asc_sync_mte3(6);
        asc_sync();

        // Select the TopkSelectPolicyConfig based on kTopkTier at compile time.
        using PolicyConfig = std::conditional_t<
            kTopkTier <= 512, qli_topk::Topk512Config,
            std::conditional_t<
                kTopkTier <= 1024, qli_topk::Topk1024Config,
                qli_topk::Topk4096Config>>;

        const uint32_t sub_id = asc_get_sub_block_id();
        const uint32_t core_id = block_idx;

        const uint32_t num_q_blocks = ceil_div(num_q_tokens, kNumQTokensPerBlock);

        // UB scratch for ks/ke staging, placed after topk's kUbEnd (safe zone).
        constexpr uintptr_t kSeqUBOffset = 253440;  // 248KB - 512B, 32B aligned
        __ubuf__ uint32_t* ub_ks = reinterpret_cast<__ubuf__ uint32_t*>(kSeqUBOffset);
        __ubuf__ uint32_t* ub_ke = ub_ks + kNumQTokensPerBlock;

        for (uint32_t q_block_idx = core_id; q_block_idx < num_q_blocks;
             q_block_idx += kNumCores) {
            const uint32_t q_base = q_block_idx * kNumQTokensPerBlock;
            const uint32_t n_rows = min(kNumQTokensPerBlock, num_q_tokens - q_base);

            // Read ks/ke via MTE2 (not ld_dev) to avoid DCache stale data.
            constexpr uint32_t kSeqBytes = 64;
            copy_gm_to_ub(reinterpret_cast<__ubuf__ uint8_t*>(ub_ks),
                          reinterpret_cast<__gm__ uint8_t*>(cu_seq_len_k_start + q_base), 1,
                          kSeqBytes, kSeqBytes,
                          asc_load_l2_cache_mode::NOTALLOC_CLEAN);
            copy_gm_to_ub(reinterpret_cast<__ubuf__ uint8_t*>(ub_ke),
                          reinterpret_cast<__gm__ uint8_t*>(cu_seq_len_k_end + q_base), 1,
                          kSeqBytes, kSeqBytes,
                          asc_load_l2_cache_mode::NOTALLOC_CLEAN);
            set_flag(PIPE_MTE2, PIPE_S, static_cast<event_t>(7));
            wait_flag(PIPE_MTE2, PIPE_S, static_cast<event_t>(7));

            for (uint32_t r = sub_id; r < n_rows; r += 2) {
                const uint32_t row = q_base + r;
                const uint32_t ks = ub_ks[r];
                const uint32_t ke = ub_ke[r];
                const uint32_t valid_len = ke > ks ? ke - ks : 0;

                if (valid_len == 0)
                    continue;

                auto* row_in  = logits.index(row, 0).ptr();
                auto* row_val = output_values.index(row, 0).ptr();
                auto* row_idx = output_indices.index(row, 0).ptr();

                // Dispatch lower/upper config based on valid_len vs pivot,
                // mirroring Kernel::run (kernel.asc:909-919).
                const bool use_lower = valid_len <= PolicyConfig::pivot;
                if (use_lower) {
                    qli_topk::topk_select_run<
                        typename PolicyConfig::LowerConfig, kSortedIndex,
                        kAbortWhenNanFound, bfloat16_t, int32_t>(
                        valid_len, topk, row_in, row_val, row_idx,
                        /*return_value=*/true,
                        /*row_output_idx_offset=*/reinterpret_cast<__gm__ int32_t*>(cu_seq_len_k_start + row),
                        /*idx_oob_fill=*/static_cast<int32_t>(2147483647),
                        /*value_oob_fill_bits=*/static_cast<uint16_t>(0xFF80));
                } else {
                    qli_topk::topk_select_run<
                        typename PolicyConfig::UpperConfig, kSortedIndex,
                        kAbortWhenNanFound, bfloat16_t, int32_t>(
                        valid_len, topk, row_in, row_val, row_idx,
                        /*return_value=*/true,
                        /*row_output_idx_offset=*/reinterpret_cast<__gm__ int32_t*>(cu_seq_len_k_start + row),
                        /*idx_oob_fill=*/static_cast<int32_t>(2147483647),
                        /*value_oob_fill_bits=*/static_cast<uint16_t>(0xFF80));
                }
                // Drain topk's MTE3 stores (output_values/indices writes) before
                // the next row's topk reads GM logits via MTE2. Without this,
                // the MTE3 stores may pollute L2 cache for subsequent MTE2 loads.
                asc_sync_mte3(6);
            }
        }
    }
}

} // namespace deep_gemm