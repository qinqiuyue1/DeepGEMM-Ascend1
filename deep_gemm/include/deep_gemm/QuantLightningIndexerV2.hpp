#pragma once

// QuantLightningIndexerV2: tile-level pipelined fused kernel (方案 B).
//
// vs 方案 A (mqa_logits_topk.hpp, serial):
//   方案 A: mqa_logits (all KV splits) → asc_sync → topk (full radix select)
//   方案 B: each KV split's reduce → incremental histogram update → store GM
//           after all KV splits → topk finalize (threshold known, lightweight select)
//
// Key change: topk_build_high_histogram is called inside the reduce loop,
// after each tile's reduce produces ub_logits[stage]. This overlaps the
// histogram V-pipe work with AIC's next-tile MMAD.
//
// After all KV splits, topk finalize uses the pre-accumulated high-byte
// histogram to find the threshold immediately (no warmup scan), then does
// a single low-byte radix pass on GM logits to extract the exact top-K.
//
// UB layout (non-paged, H=32, D=128):
//   [0, 128KB)         ub_accum[2] (64KB each)
//   [128KB, 132KB)     ub_weights[2] (2KB each)
//   [132KB, 138KB)     ub_logits[3] (2KB each) + ub_logits_head (512B)
//   [138KB, 139KB)     ub_seq_start/end (128B each)
//   [139KB, 143KB)     ub_histogram[8] (per-row, 256bin×2B = 512B each, 8 rows)
//   [143KB, 144KB)     ub_threshold[8] + ub_nan_count[8] + ub_state[8]
//   [144KB, 248KB)     topk workspace (candidates, input staging, etc.)
//   [248KB, 248KB+64B) ub_ks/ub_ke (ks/ke staging)

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
    uint32_t kTopkTier,
    bool kAbortWhenNanFound,
    bool kSortedIndex
>
__global__ __mix__(1, 2) void mqa_logits_topk_pipelined_impl(
    uint32_t num_q_tokens, uint32_t num_kv_tokens,
    gm_ptr<uint8_t, Major::K> q, gm_ptr<uint8_t, Major::K> kv,
    __gm__ int16_t* sf_q, __gm__ int16_t* sf_kv,
    gm_ptr<bfloat16_t, Major::K> weights,
    gm_ptr<bfloat16_t, Major::K> logits,
    gm_ptr<bfloat16_t, Major::K> output_values,
    gm_ptr<int32_t, Major::K> output_indices,
    __gm__ uint32_t* cu_seq_len_k_start, __gm__ uint32_t* cu_seq_len_k_end,
    uint32_t logits_stride,
    uint32_t output_values_stride,
    uint32_t output_indices_stride,
    uint32_t topk
) {
    // ========================================================================
    // Constants (same as mqa_logits_core_impl)
    // ========================================================================
    using qk_dtype_t = std::conditional_t<kIsFP4, float4_e2m1x2_t, float8_e4m3_t>;
    constexpr bool kIsPaged = false;
    constexpr uint32_t kNumVecElems = asc_get_vf_len() / sizeof(bfloat16_t);
    constexpr uint32_t MAD_M = kMadM;
    constexpr uint32_t MAD_M_PADDED = aligned(MAD_M, FRAC_MN);
    constexpr uint32_t MAD_N = kSplitKV;
    constexpr uint32_t MAD_K = kHeadDim;
    constexpr uint32_t kNumQHTilesPerBlock = 8;
    constexpr uint32_t kNumQHRowsPerBlock = kNumQHTilesPerBlock * MAD_M;
    constexpr uint32_t kNumPaddedQHRowsPerBlock = kNumQHTilesPerBlock * MAD_M_PADDED;
    constexpr uint32_t kNumQTokensPerBlock = kNumQHRowsPerBlock / kNumHeads;
    constexpr uint32_t kNumQTokensPerTile = MAD_M / kNumHeads;
    constexpr uint32_t kSFPairs = kHeadDim / MX_SF_DIVISOR;

    // L1/L0 layout (same as mqa_logits_core_impl)
    constexpr l1_ptr<qk_dtype_t, Major::K> l1q(kNumPaddedQHRowsPerBlock, kHeadDim, 0);
    constexpr l1_ptr<qk_dtype_t, Major::K> l1kv(kSplitKV, kHeadDim, l1q.offset(kNumQStages));
    constexpr l1_ptr<int16_t> l1_sf_q(kNumPaddedQHRowsPerBlock, kSFPairs, l1kv.offset(kNumKVStages));
    constexpr l1_ptr<int16_t> l1_sf_kv(kSplitKV, kSFPairs, l1_sf_q.offset(kNumQStages));
    constexpr l1_ptr<int16_t> l1_sf_q_tiles(MAD_M_PADDED, kSFPairs, l1_sf_q.addr);
    constexpr l0a_ptr<qk_dtype_t> l0a(MAD_M_PADDED, MAD_K);
    constexpr l0b_ptr<qk_dtype_t> l0b(MAD_N, MAD_K);
    constexpr l0c_ptr<float> l0c(MAD_M_PADDED, MAD_N);

    // UB layout (same as mqa_logits_core_impl for reduce part)
    constexpr uint32_t kNumWeightStages = 2;
    constexpr ub_ptr<bfloat16_t> ub_accum(MAD_M_PADDED, MAD_N, 0);
    constexpr ub_ptr<bfloat16_t> ub_weights(kNumQTokensPerBlock, kNumHeads, ub_accum.offset(kNumAccumStages));
    constexpr ub_ptr<bfloat16_t> ub_logits(kNumQTokensPerTile, MAD_N, ub_weights.offset(kNumWeightStages));
    constexpr ub_ptr<bfloat16_t> ub_logits_head(1, asc_get_vf_len() / sizeof(bfloat16_t),
                                                    ub_logits.offset(kNumLogitsStages));
    constexpr ub_ptr<uint32_t> ub_seq_start(1, aligned(kNumQTokensPerBlock, 8), ub_logits_head.offset(1));
    constexpr ub_ptr<uint32_t> ub_seq_end(1, aligned(kNumQTokensPerBlock, 8), ub_seq_start.offset(1));

    // UB layout for topk (NEW: per-row high-byte histogram)
    // Place after ub_seq_end. 8 rows per AIV (kNumQTokensPerBlock/2).
    // Each row: 256 bins × 2 bytes = 512 bytes.
    constexpr uint32_t kRowsPerAIV = kNumQTokensPerBlock / 2;  // 8 for H=32
    constexpr uintptr_t kHistogramUBOffset = ub_seq_end.offset(1);
    constexpr uintptr_t kHistogramSize = 256 * sizeof(uint16_t);  // 512 bytes per row
    // Per-row: histogram[256] + nan_count[1] + threshold[1] + initialized[1]
    constexpr uintptr_t kPerRowTopkSize = kHistogramSize + 32;  // 512 + 32 = 544 bytes
    constexpr uintptr_t kTopkStateEnd = kHistogramUBOffset + kRowsPerAIV * kPerRowTopkSize;

    // Scheduler
    auto scheduler = MQALogitsScheduler<kNumQTokensPerBlock, kSplitKV, kNumCores>(
        block_idx, num_q_tokens, num_kv_tokens,
        cu_seq_len_k_start, cu_seq_len_k_end, kNumHeads);
    const gm_ptr<int16_t, Major::MN> sf_q_t(scheduler.get_q_sf_stride(), reinterpret_cast<uintptr_t>(sf_q));
    const gm_ptr<qk_dtype_t, Major::K> q_t(q.stride_outer, q.addr, q.stride_batch);
    const gm_ptr<qk_dtype_t, Major::K> kv_t(kv.stride_outer, kv.addr, kv.stride_batch);

    uint32_t q_block_idx, kv_token_base, num_kv_splits;

    // ========================================================================
    // AIC branch: identical to mqa_logits_core_impl AIC
    // ========================================================================
    if ASCEND_IS_AIC {
        asc_set_mmad_direction_n();
        constexpr uint32_t kQEventBase = 0;
        constexpr uint32_t kKVEventBase = kNumQStages;
        set_flags<PIPE_M, PIPE_MTE1, 2>();
        set_flags<PIPE_MTE1, PIPE_MTE2, kNumQStages, kQEventBase>();
        set_flags<PIPE_MTE1, PIPE_MTE2, kNumKVStages, kKVEventBase>();

        uint32_t q_stage_idx = 0;
        uint32_t kv_stage_idx = 0;
        uint32_t cd_stage_idx = 0;
        while (scheduler.next_q_block(q_block_idx, kv_token_base, num_kv_splits)) {
            const uint32_t q_base = scheduler.get_q_base(q_block_idx);
            const uint32_t num_q_tokens_in_block = scheduler.get_num_q_tokens_in_block(q_block_idx);
            const uint32_t num_qh_rows_in_block = num_q_tokens_in_block * kNumHeads;
            const uint32_t num_qh_tiles_in_block = ceil_div(num_qh_rows_in_block, MAD_M);

            asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(kQEventBase + q_stage_idx));
            if constexpr (MAD_M == MAD_M_PADDED) {
                fill_mn_tail_block(l1q[q_stage_idx], num_qh_rows_in_block, aligned(num_qh_rows_in_block, FRAC_MN));
                copy_gm_to_l1(l1q[q_stage_idx], q_t.index(q_base * kNumHeads, 0), num_qh_rows_in_block, kHeadDim,
                              asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                copy_gm_to_l1_mx(l1_sf_q[q_stage_idx], sf_q_t.index(q_base * kNumHeads, 0), num_qh_rows_in_block,
                                 kSFPairs, asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
            } else {
                const l1_ptr<qk_dtype_t> l1q_m_tiles(MAD_M_PADDED, get_frac_k<qk_dtype_t>(), l1q[q_stage_idx].addr);
                for (uint32_t qh_tile_idx = 0; qh_tile_idx < num_qh_tiles_in_block; ++qh_tile_idx) {
                    const uint32_t qh_base = qh_tile_idx * MAD_M;
                    const uint32_t num_m_rows = min(MAD_M, num_qh_rows_in_block - qh_base);
                    const uint32_t sf_q_tile_idx = q_stage_idx * kNumQHTilesPerBlock + qh_tile_idx;
                    const auto q_tile = l1q_m_tiles[qh_tile_idx].as_mad_aligned(kNumPaddedQHRowsPerBlock, kHeadDim);
                    const auto sf_q_tile = l1_sf_q_tiles[sf_q_tile_idx];
                    fill_mn_tail_block(q_tile, num_m_rows, MAD_M_PADDED);
                    copy_gm_to_l1(q_tile, q_t.index(q_base * kNumHeads + qh_base, 0), num_m_rows, kHeadDim,
                                  asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                    copy_gm_to_l1_mx(sf_q_tile, sf_q_t.index(q_base * kNumHeads + qh_base, 0), num_m_rows, kSFPairs,
                                     asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                }
            }
            asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(kQEventBase + q_stage_idx));

            asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(kQEventBase + q_stage_idx));
            asc_sync_wait(PIPE_M, PIPE_MTE1, EVENT_ID1);
            for (uint32_t qh_tile_idx = 0; qh_tile_idx < num_qh_tiles_in_block; ++qh_tile_idx) {
                const uint32_t qh_base = qh_tile_idx * MAD_M;
                const uint32_t qh_padded_base = qh_tile_idx * MAD_M_PADDED;
                const uint32_t num_m_rows = min(MAD_M, num_qh_rows_in_block - qh_base);
                const auto l0a_tile = l0a[qh_tile_idx].as_mad_aligned(num_m_rows, MAD_K);
                copy_l1_to_l0a(l0a_tile, l1q[q_stage_idx], qh_padded_base, 0);
                copy_l1_to_l0a_mx(l0a_tile, l1_sf_q[q_stage_idx], qh_padded_base, 0, MAD_K);
            }
            asc_sync_notify(PIPE_MTE1, PIPE_M, EVENT_ID1);

            const auto is_tile_active = [&](uint32_t q_tile_base, uint32_t kv_idx, uint32_t kv_end) __aicore__ -> bool {
                for (uint32_t q_idx = q_tile_base; q_idx < min(q_tile_base + kNumQTokensPerTile, num_q_tokens_in_block); ++q_idx) {
                    const uint32_t row = q_base + q_idx;
                    if (kv_idx < scheduler.cu_seq_len_k_end[row] &&
                        scheduler.cu_seq_len_k_start[row] < kv_end) {
                        return true;
                    }
                }
                return false;
            };

            asc_sync_wait(PIPE_MTE1, PIPE_M, EVENT_ID1);
            for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++kv_split_idx) {
                const uint32_t kv_idx = scheduler.get_kv_token_offset(kv_token_base, kv_split_idx);
                const uint32_t actual_kv = scheduler.get_num_kv_block_tokens(kv_token_base, kv_split_idx);
                const uint32_t issued_n = aligned(actual_kv, kNumVecElems);
                asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(kKVEventBase + kv_stage_idx));
                fill_mn_tail_block(l1kv[kv_stage_idx], actual_kv, issued_n);
                const gm_ptr<int16_t, Major::MN> sf_kv_t(scheduler.get_kv_sf_stride(), reinterpret_cast<uintptr_t>(sf_kv));
                copy_gm_to_l1(l1kv[kv_stage_idx], kv_t.index(kv_idx, 0), actual_kv, kHeadDim,
                              asc_load_l2_cache_mode::NOTALLOC_KEEP);
                copy_gm_to_l1_mx(l1_sf_kv[kv_stage_idx], sf_kv_t.index(kv_idx, 0), actual_kv, kSFPairs,
                                 asc_load_l2_cache_mode::NOTALLOC_KEEP);
                asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(kKVEventBase + kv_stage_idx));
                asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(kKVEventBase + kv_stage_idx));

                asc_sync_wait(PIPE_M, PIPE_MTE1, EVENT_ID0);
                const auto l0b_tile = l0b.as_mad_aligned(issued_n, MAD_K);
                copy_l1_to_l0b(l0b_tile, l1kv[kv_stage_idx], 0, 0);
                copy_l1_to_l0b_mx(l0b_tile, l1_sf_kv[kv_stage_idx], 0, 0, MAD_K);
                asc_sync_notify(PIPE_MTE1, PIPE_M, EVENT_ID0);
                asc_sync_wait(PIPE_MTE1, PIPE_M, EVENT_ID0);

                const uint32_t owner_offset = num_qh_tiles_in_block & kv_split_idx & 1;
                for (uint32_t qh_tile_idx = 0; qh_tile_idx < num_qh_tiles_in_block; ++qh_tile_idx) {
                    const uint32_t qh_base = qh_tile_idx * MAD_M;
                    const uint32_t q_tile_base = qh_base / kNumHeads;
                    if (!is_tile_active(q_tile_base, kv_idx, kv_idx + actual_kv))
                        continue;
                    const uint32_t num_m_rows = min(MAD_M, num_qh_rows_in_block - qh_base);
                    const auto l0c_tile = l0c[cd_stage_idx].as_mad_aligned(num_m_rows, issued_n);
                    const uint32_t accum_owner = (qh_tile_idx + owner_offset) & 1u;
                    const uint32_t accum_stage_idx = ((kv_split_idx * num_qh_tiles_in_block + qh_tile_idx) / 2) % kNumAccumStages;
                    mad(l0c_tile, l0a[qh_tile_idx], l0b_tile, num_m_rows, issued_n, MAD_K,
                        asc_unit_flag_mode::ENABLE_UPDATE, true);
                    wait_intra_block(PIPE_FIX, static_cast<uint8_t>(accum_owner),
                                     static_cast<uint8_t>(accum_stage_idx));
                    const auto accum_tile = ub_accum[accum_stage_idx].as_mad_aligned(num_m_rows, issued_n);
                    copy_l0c_to_ub(accum_tile, l0c_tile, num_m_rows, issued_n, asc_dual_dst_mode::DUAL_DST_DISABLE,
                                   accum_owner, infer_quant_mode<bfloat16_t, float>(), true, false,
                                   asc_unit_flag_mode::ENABLE_UPDATE);
                    set_intra_block(PIPE_FIX, accum_owner, accum_stage_idx);
                    cd_stage_idx = (cd_stage_idx + 1) % kNumCDStages;
                }
                asc_sync_notify(PIPE_M, PIPE_MTE1, EVENT_ID0);
                asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(kKVEventBase + kv_stage_idx));
                kv_stage_idx = (kv_stage_idx + 1) % kNumKVStages;
            }
            asc_sync_notify(PIPE_M, PIPE_MTE1, EVENT_ID1);
            asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(kQEventBase + q_stage_idx));
            q_stage_idx = (q_stage_idx + 1) % kNumQStages;
        }
    }
    // ========================================================================
    // AIV branch: reduce + store + INCREMENTAL HISTOGRAM + topk finalize
    // ========================================================================
    else {
        const uint32_t sub_id = asc_get_sub_block_id();

        // Same event setup as mqa_logits_core_impl
        set_intra_blocks<PIPE_V, kNumAccumStages>(static_cast<uint8_t>(sub_id));
        set_flags<PIPE_MTE3, PIPE_V, kNumLogitsStages>();
        asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(kNumLogitsStages));
        set_flags<PIPE_V, PIPE_MTE2, kNumWeightStages>();

        uint32_t logits_stage_idx = 0;
        uint32_t weight_stage_idx = 0;

        // Per-row topk state pointers
        // Each AIV handles kRowsPerAIV rows per Q-block (alternating sub_id).
        // Row index within AIV: (q_in_block - sub_id) / 2, range [0, kRowsPerAIV).
        auto get_row_hist = [&](uint32_t q_in_block) __aicore__ -> __ubuf__ uint16_t* {
            uint32_t local_row = (q_in_block - sub_id) / 2;
            return reinterpret_cast<__ubuf__ uint16_t*>(kHistogramUBOffset + local_row * kPerRowTopkSize);
        };
        auto get_row_nan_count = [&](uint32_t q_in_block) __aicore__ -> __ubuf__ uint32_t* {
            uint32_t local_row = (q_in_block - sub_id) / 2;
            return reinterpret_cast<__ubuf__ uint32_t*>(kHistogramUBOffset + local_row * kPerRowTopkSize + kHistogramSize);
        };
        auto get_row_threshold = [&](uint32_t q_in_block) __aicore__ -> __ubuf__ uint16_t* {
            uint32_t local_row = (q_in_block - sub_id) / 2;
            return reinterpret_cast<__ubuf__ uint16_t*>(kHistogramUBOffset + local_row * kPerRowTopkSize + kHistogramSize + 4);
        };
        auto get_row_initialized = [&](uint32_t q_in_block) __aicore__ -> __ubuf__ uint32_t* {
            uint32_t local_row = (q_in_block - sub_id) / 2;
            return reinterpret_cast<__ubuf__ uint32_t*>(kHistogramUBOffset + local_row * kPerRowTopkSize + kHistogramSize + 8);
        };

        while (scheduler.next_q_block(q_block_idx, kv_token_base, num_kv_splits)) {
            const uint32_t q_base = scheduler.get_q_base(q_block_idx);
            const uint32_t num_q_tokens_in_block = scheduler.get_num_q_tokens_in_block(q_block_idx);
            const uint32_t num_qh_rows_in_block = num_q_tokens_in_block * kNumHeads;
            const uint32_t num_qh_tiles_in_block = ceil_div(num_qh_rows_in_block, MAD_M);

            // Load ks/ke (same as mqa_logits_core_impl)
            copy_gm_to_ub(ub_seq_start.ptr(), scheduler.cu_seq_len_k_start + q_base, 1, num_q_tokens_in_block,
                          num_q_tokens_in_block * sizeof(uint32_t), asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
            copy_gm_to_ub(ub_seq_end.ptr(), scheduler.cu_seq_len_k_end + q_base, 1, num_q_tokens_in_block,
                          num_q_tokens_in_block * sizeof(uint32_t), asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
            asc_sync_notify(PIPE_MTE2, PIPE_S, EVENT_ID0);

            const auto seq_ks = [&](uint32_t q_in_block) __aicore__ -> uint32_t {
                return ub_seq_start.ptr()[q_in_block];
            };
            const auto seq_ke = [&](uint32_t q_in_block) __aicore__ -> uint32_t {
                return ub_seq_end.ptr()[q_in_block];
            };
            const auto is_tile_active = [&](uint32_t q_tile_base, uint32_t kv_idx, uint32_t kv_end) __aicore__ -> bool {
                for (uint32_t q_idx = q_tile_base; q_idx < min(q_tile_base + kNumQTokensPerTile, num_q_tokens_in_block); ++q_idx) {
                    if (kv_idx >= seq_ke(q_idx))
                        continue;
                    if (seq_ks(q_idx) < kv_end)
                        return true;
                }
                return false;
            };

            // Initialize per-row topk state for this Q-block
            for (uint32_t r = sub_id; r < kNumQTokensPerBlock; r += 2) {
                if (r >= num_q_tokens_in_block) break;
                get_row_initialized(r)[0] = 0;
            }

            asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(weight_stage_idx));
            copy_gm_to_ub(ub_weights[weight_stage_idx].ptr(), weights.index(q_base, 0).ptr(), num_q_tokens_in_block,
                          kNumHeads, weights.stride_outer * sizeof(bfloat16_t),
                          asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
            asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(weight_stage_idx));
            asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(weight_stage_idx));
            asc_sync_wait(PIPE_MTE2, PIPE_S, EVENT_ID0);

            // ================================================================
            // KV split loop: reduce + store + INCREMENTAL HISTOGRAM
            // ================================================================
            uint32_t kv_stage_idx = 0;
            for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++kv_split_idx) {
                const uint32_t kv_idx = scheduler.get_kv_token_offset(kv_token_base, kv_split_idx);
                const uint32_t actual_kv = scheduler.get_num_kv_block_tokens(kv_token_base, kv_split_idx);
                const uint32_t issued_n = aligned(actual_kv, kNumVecElems);

                const uint32_t owner_offset = num_qh_tiles_in_block & kv_split_idx & 1;
                for (uint32_t qh_tile_idx = sub_id ^ owner_offset; qh_tile_idx < num_qh_tiles_in_block; qh_tile_idx += 2) {
                    const uint32_t qh_base = qh_tile_idx * MAD_M;
                    const uint32_t q_tile_base = qh_base / kNumHeads;
                    if (!is_tile_active(q_tile_base, kv_idx, kv_idx + actual_kv))
                        continue;
                    const uint32_t accum_stage_idx = ((kv_split_idx * num_qh_tiles_in_block + qh_tile_idx) / 2) % kNumAccumStages;
                    wait_intra_block(PIPE_V, static_cast<uint8_t>(sub_id), static_cast<uint8_t>(accum_stage_idx));
                    asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(logits_stage_idx));
                    const auto ub_logits_stage = ub_logits[logits_stage_idx];
                    auto score = ub_accum[accum_stage_idx].ptr();
                    auto weight = ub_weights[weight_stage_idx].ptr();

                    // --- REDUCE: ub_accum → ub_logits[stage] ---
                    for (uint32_t q_local = 0; q_local < kNumQTokensPerTile;) {
                        const uint32_t q_idx = q_tile_base + q_local;
                        if (q_idx >= num_q_tokens_in_block)
                            break;
                        auto line = ub_logits_stage.ptr() + q_local * kSplitKV;
                        q_local += mqa_logits_reduce_queries<0, kNumQTokensPerTile, kNumHeads, kSplitKV>(
                            score, weight, line, q_local, q_idx,
                            num_q_tokens_in_block, issued_n, 0);
                    }
                    set_intra_block(PIPE_V, static_cast<uint8_t>(sub_id), static_cast<uint8_t>(accum_stage_idx));
                    asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(logits_stage_idx));

                    // --- STORE: ub_logits[stage] → GM logits ---
                    asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(logits_stage_idx));
                    for (uint32_t q_store = 0; q_store < kNumQTokensPerTile; ++q_store) {
                        const uint32_t q_in_block = q_tile_base + q_store;
                        if (q_in_block >= num_q_tokens_in_block)
                            break;
                        const uint32_t row = q_base + q_in_block;
                        const uint32_t ks = seq_ks(q_in_block);
                        const uint32_t ke = seq_ke(q_in_block);
                        const uint32_t store_begin = max(kv_idx, ks);
                        const uint32_t store_end = min(kv_idx + actual_kv, ke);
                        if (store_begin >= store_end)
                            continue;
                        const uint32_t dst_col = store_begin - ks;
                        const auto line = ub_logits_stage.ptr() + q_store * kSplitKV;
                        mqa_logits_store<kNumLogitsStages>(logits.index(row, dst_col).ptr(),
                                                           line + store_begin - kv_idx, ub_logits_head.ptr(),
                                                           store_end - store_begin);

                        // --- INCREMENTAL HISTOGRAM (NEW in 方案 B) ---
                        // Only for rows owned by this AIV (sub_id matches)
                        if ((q_in_block & 1) == sub_id) {
                            const uint32_t valid_count = store_end - store_begin;
                            __ubuf__ uint16_t* hist = get_row_hist(q_in_block);
                            __ubuf__ uint32_t* nan_cnt = get_row_nan_count(q_in_block);
                            __ubuf__ uint32_t* initd = get_row_initialized(q_in_block);
                            bool do_init = (initd[0] == 0);
                            initd[0] = 1;
                            // Call topk_build_high_histogram (from QuantLightningIndexer.hpp)
                            qli_topk::topk_build_high_histogram(
                                line + store_begin - kv_idx, valid_count,
                                hist, do_init, topk, nan_cnt);
                        }
                    }
                    asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(logits_stage_idx));
                    logits_stage_idx = (logits_stage_idx + 1) % kNumLogitsStages;
                }
                kv_stage_idx = (kv_stage_idx + 1) % kNumKVStages;
            }
            asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(weight_stage_idx));
            weight_stage_idx = (weight_stage_idx + 1) % kNumWeightStages;

            // ================================================================
            // TOPK FINALIZE (方案 B Phase 2: use pre-accumulated histogram)
            // ================================================================
            asc_sync_mte3(6);
            asc_sync();

            using PolicyConfig = std::conditional_t<
                kTopkTier <= 512, qli_topk::Topk512Config,
                std::conditional_t<
                    kTopkTier <= 1024, qli_topk::Topk1024Config,
                    qli_topk::Topk4096Config>>;

            constexpr uintptr_t kSeqUBOffset = 253440;
            __ubuf__ uint32_t* ub_ks2 = reinterpret_cast<__ubuf__ uint32_t*>(kSeqUBOffset);
            __ubuf__ uint32_t* ub_ke2 = ub_ks2 + kNumQTokensPerBlock;

            constexpr uint32_t kSeqBytes = 64;
            copy_gm_to_ub(reinterpret_cast<__ubuf__ uint8_t*>(ub_ks2),
                          reinterpret_cast<__gm__ uint8_t*>(cu_seq_len_k_start + q_base), 1,
                          kSeqBytes, kSeqBytes,
                          asc_load_l2_cache_mode::NOTALLOC_CLEAN);
            copy_gm_to_ub(reinterpret_cast<__ubuf__ uint8_t*>(ub_ke2),
                          reinterpret_cast<__gm__ uint8_t*>(cu_seq_len_k_end + q_base), 1,
                          kSeqBytes, kSeqBytes,
                          asc_load_l2_cache_mode::NOTALLOC_CLEAN);
            set_flag(PIPE_MTE2, PIPE_S, static_cast<event_t>(7));
            wait_flag(PIPE_MTE2, PIPE_S, static_cast<event_t>(7));

            for (uint32_t r = sub_id; r < kNumQTokensPerBlock; r += 2) {
                if (r >= num_q_tokens_in_block) break;
                const uint32_t row = q_base + r;
                const uint32_t ks = ub_ks2[r];
                const uint32_t ke = ub_ke2[r];
                const uint32_t valid_len = ke > ks ? ke - ks : 0;
                if (valid_len == 0) continue;

                auto* row_in  = logits.index(row, 0).ptr();
                auto* row_val = output_values.index(row, 0).ptr();
                auto* row_idx = output_indices.index(row, 0).ptr();

                // Get pre-accumulated histogram and NaN count for this row
                __ubuf__ uint16_t* pre_hist = get_row_hist(r);
                __ubuf__ uint32_t* pre_nan = get_row_nan_count(r);

                const bool use_lower = valid_len <= PolicyConfig::pivot;
                if (use_lower) {
                    qli_topk::topk_finalize_run<
                        typename PolicyConfig::LowerConfig, kSortedIndex,
                        kAbortWhenNanFound, bfloat16_t, int32_t>(
                        valid_len, topk, row_in, row_val, row_idx,
                        true,
                        reinterpret_cast<__gm__ int32_t*>(cu_seq_len_k_start + row),
                        static_cast<int32_t>(2147483647),
                        static_cast<uint16_t>(0xFF80),
                        pre_hist, pre_nan);
                } else {
                    qli_topk::topk_finalize_run<
                        typename PolicyConfig::UpperConfig, kSortedIndex,
                        kAbortWhenNanFound, bfloat16_t, int32_t>(
                        valid_len, topk, row_in, row_val, row_idx,
                        true,
                        reinterpret_cast<__gm__ int32_t*>(cu_seq_len_k_start + row),
                        static_cast<int32_t>(2147483647),
                        static_cast<uint16_t>(0xFF80),
                        pre_hist, pre_nan);
                }
                asc_sync_mte3(6);
            }
        }
    }
}

} // namespace deep_gemm
