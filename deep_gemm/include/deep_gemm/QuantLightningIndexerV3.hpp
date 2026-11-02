#pragma once

// QuantLightningIndexerV3: 方案 A + 块间 topk 流水.
//
// vs V1 (mqa_logits_topk.hpp, 两阶段串行):
//   V1: 全部 Q-block 的 logits 写完 -> 全局屏障 -> 全部 topk
//   V3: 每个 Q-block 的 logits 写完后立刻做该块的 topk, AIC 同时继续下一个
//       Q-block 的 MMAD (设计意图是 topk(block k) 被 cube(block k+1) 掩盖;
//       实测未发生, 原因见下方"实测结论").
//
// 插入点: mqa_logits_core_impl 的 AIV q-block 循环体末尾 (OnBlockDone 回调).
//
// 实测结论 (s1=1024, s2=65536, 2026-11-02): 正确性全绿 (值与索引, 多 seed),
// 但 topk 的 ~122us 几乎 100% 暴露, 相对两 kernel 串行约 0~-3%.
// 原因不是 barrier (实测 barrier 次数 x5 几乎不改变耗时), 而是 Stage1 的
// AIC<->AIV 紧耦合: AIC 每 tile 都要等 AIV 放 ub_accum, 而 L0C 只有 2 个 CD
// stage (kSplitKV==512 时 kNumCDStages 被 static_assert 钉死为 2), 每个 tile
// 仅 ~0.36us, 故 AIC 超前量 ~0.72us, 盖不住每块 ~61us 的 topk.
// 所以"topk(block k) 被 cube(block k+1) 掩盖"在实测中并未发生;
// 要拿收益必须减小 topk 工作量或把它移出 AIV 主循环, 加深 accum 无效.
//
// UB 布局: Stage 1 保持 [0, Stage1End). topk 的 candidate 集群上移到 Stage1End,
// input 集群保持 >= 128KB 的原位. 这里取 kBlockSize=8192 / kNumStages=4:
// 在能装进 248KB 的配置里 staging 复用次数 (barrier 次数) 最少. s2=65536 的
// valid_len 不再受编码位宽限制 (见 QuantLightningIndexer.hpp 的槽复用修复).

#include <deep_gemm/ascend.hpp>
#include <deep_gemm/common.hpp>
#include <deep_gemm/mqa_logits.hpp>
#include <deep_gemm/QuantLightningIndexer.hpp>
#include <deep_gemm/scheduler/mqa_logits.hpp>

namespace deep_gemm {

// Stage 1 UB footprint, mirroring the ub_ptr chain in mqa_logits_core_impl
// (non-paged: ub_sf_kv is empty).
template <uint32_t kNumHeads, uint32_t kHeadDim, uint32_t kMadM,
          uint32_t kNumAccumStages, uint32_t kNumLogitsStages>
__aicore__ constexpr uintptr_t v3_stage1_ub_end() {
    constexpr uint32_t MAD_N = 512;  // kSplitKV
    constexpr uint32_t MAD_M_PADDED = aligned(kMadM, FRAC_MN);
    constexpr uint32_t kNumQTokensPerBlock = 8 * kMadM / kNumHeads;
    constexpr uint32_t kNumQTokensPerTile = kMadM / kNumHeads;
    constexpr uint32_t kNumVecElems = asc_get_vf_len() / sizeof(bfloat16_t);
    constexpr uintptr_t accum_end =
        kNumAccumStages * MAD_M_PADDED * MAD_N * sizeof(bfloat16_t);
    constexpr uintptr_t weights_end =
        accum_end + 2 * kNumQTokensPerBlock * kNumHeads * sizeof(bfloat16_t);
    constexpr uintptr_t logits_end =
        weights_end + kNumLogitsStages * kNumQTokensPerTile * MAD_N * sizeof(bfloat16_t);
    constexpr uintptr_t head_end = logits_end + kNumVecElems * sizeof(bfloat16_t);
    return head_end + 2 * aligned(kNumQTokensPerBlock, 8u) * sizeof(uint32_t);
}

// V3 topk config. A 8192-element block with 4 staging slots is the largest block that
// still fits the topk workspace above Stage 1 in UB, and it minimises how often the
// staging ring wraps (each wrap needs a VF drain before the slot can be reused).
template <uint32_t kTopk, uint32_t kFiltered, uint32_t kWarmup>
using V3AlgoConfig = qli_topk::TopkSelectConfig<kTopk, kFiltered, kWarmup,
                                                /*kNumStages=*/4,
                                                /*kBlockSize=*/8192,
                                                /*kShufflePageSize=*/2048,
                                                /*kBlockSizeBits=*/13>;

template <
    bool kIsFP4,
    uint32_t kNumHeads, uint32_t kHeadDim,
    uint32_t kSplitKV, uint32_t kMadM, uint32_t kNumCDStages,
    uint32_t kNumQStages, uint32_t kNumKVStages, uint32_t kNumAccumStages,
    uint32_t kNumLogitsStages,
    uint32_t kNumCores,
    uint32_t kTopkTier,              // 512, 1024, or 4096
    bool kAbortWhenNanFound,
    bool kSortedIndex,
    uint32_t kAccumStagesForV3 = 1   // block=8192/stages=4 needs ACC=1 to stay under 248KB
>
__global__ __mix__(1, 2) void mqa_logits_topk_v3_impl(
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
    asc_init();
    constexpr uint32_t kNumQTokensPerBlock = 8 * kMadM / kNumHeads;

    using PolicyConfig = std::conditional_t<
        kTopkTier <= 512, qli_topk::Topk512Config,
        std::conditional_t<kTopkTier <= 1024, qli_topk::Topk1024Config,
                           qli_topk::Topk4096Config>>;
    constexpr uint32_t kPivot = PolicyConfig::pivot;
    using LowerAlgo = V3AlgoConfig<
        PolicyConfig::LowerConfig::topk_capacity,
        PolicyConfig::LowerConfig::filtered_capacity,
        PolicyConfig::LowerConfig::warmup_pages>;
    using UpperAlgo = V3AlgoConfig<
        PolicyConfig::UpperConfig::topk_capacity,
        PolicyConfig::UpperConfig::filtered_capacity,
        PolicyConfig::UpperConfig::warmup_pages>;

    // Stage 1 accum stages (see kAccumStagesForV3). V3 relocates the topk workspace
    // above Stage 1, so a second accum stage only raises kStage1End; the layout still
    // fits inside TOPK_UB_LIMIT_BYTES (guarded by the static_asserts below).
    constexpr uint32_t kAccumStages = kAccumStagesForV3;

    // UB relocation. Stage 1 owns [0, stage1_end). The topk workspace is split
    // into two clusters so each one dodges Stage 1 without wasting 128KB:
    //   * candidate/histogram/state cluster -> starts just above Stage 1
    //     (it must NOT overlap ub_accum, which lives at 0)
    //   * input staging / sort cluster -> keeps the original >= 128KB placement
    //     required by the uint16 gather indices
    constexpr uintptr_t kStage1End = v3_stage1_ub_end<
        kNumHeads, kHeadDim, kMadM, kAccumStages, kNumLogitsStages>();
    constexpr uintptr_t kCandBias = kStage1End;

    using LowerPlan = qli_topk::UnifiedBufferMemoryPlan<LowerAlgo, kSortedIndex, bfloat16_t>;
    using UpperPlan = qli_topk::UnifiedBufferMemoryPlan<UpperAlgo, kSortedIndex, bfloat16_t>;
    // The candidate/histogram/state/metadata cluster runs up to kMetadataEnd: kReuseBytes
    // only covers the candidate buffers, while the histogram/state/shuffle-page-id tail
    // sits above them. The input/sort cluster ends at kUbEnd. Each is relocated separately.
    constexpr uintptr_t kCandEnd =
        kCandBias + (LowerPlan::kMetadataEnd > UpperPlan::kMetadataEnd
                         ? LowerPlan::kMetadataEnd
                         : UpperPlan::kMetadataEnd);
    // The shifted candidate cluster can run past the default 128KB input
    // placement, so push the input cluster (and everything above it) just clear
    // of it. The shift is a multiple of 32B, preserving alignment.
    // Align to the input staging block size: ub_input + stage*kBlockSize must stay
    // block-aligned, and the base must keep the hardware's 128KB offset rule.
    constexpr uintptr_t kInputStrideBytes =
        LowerPlan::kBlockSize * sizeof(bfloat16_t);
    constexpr uintptr_t kInputBias =
        kCandEnd > LowerPlan::kInputOffset
            ? aligned(kCandEnd - LowerPlan::kInputOffset, kInputStrideBytes)
            : 0u;
    constexpr uintptr_t kInputEnd =
        kInputBias + (LowerPlan::kUbEnd > UpperPlan::kUbEnd ? LowerPlan::kUbEnd
                                                            : UpperPlan::kUbEnd);
    constexpr uintptr_t kV3UbEnd = kCandEnd > kInputEnd ? kCandEnd : kInputEnd;
    static_assert(kV3UbEnd <= qli_topk::TOPK_UB_LIMIT_BYTES,
                  "V3 topk workspace does not fit alongside Stage 1 in UB");
    // The candidate cluster must clear Stage 1's ub_accum, and the input cluster
    // must keep the hardware's >= 128KB placement.
    static_assert(kCandBias >= kStage1End, "candidate cluster overlaps Stage 1");

    const auto make_scheduler = [&](uint32_t core_id) __aicore__ {
        return MQALogitsScheduler<kNumQTokensPerBlock, kSplitKV, kNumCores>(
            core_id, num_q_tokens, num_kv_tokens,
            cu_seq_len_k_start, cu_seq_len_k_end, kNumHeads);
    };

    const uint32_t sub_id = asc_get_sub_block_id();

    // Per-Q-block topk, run at the tail of this block's Stage 1.
    // Each AIV runs the rows it wrote (even/odd split), so no cross-AIV data
    // dependency is needed for the logits themselves.
    const auto on_block_done = [&](uint32_t, uint32_t q_base,
                                   uint32_t num_q_tokens_in_block) __aicore__ {
        if ASCEND_IS_AIC
            return;

        // This block's MTE3 stores must be visible to the topk MTE2 loads.
        asc_sync_mte3(6);
        asc_sync();

        for (uint32_t r = sub_id; r < num_q_tokens_in_block; r += 2) {
            const uint32_t row = q_base + r;
            const uint32_t ks = cu_seq_len_k_start[row];
            const uint32_t ke = cu_seq_len_k_end[row];
            const uint32_t valid_len = ke > ks ? ke - ks : 0;
            if (valid_len == 0)
                continue;

            auto* row_in = logits.index(row, 0).ptr();
            auto* row_val = output_values.index(row, 0).ptr();
            auto* row_idx = output_indices.index(row, 0).ptr();
            auto* row_ks = reinterpret_cast<__gm__ int32_t*>(cu_seq_len_k_start + row);

            const bool use_lower = valid_len <= kPivot;
            if (use_lower)
                qli_topk::topk_select_run<
                    LowerAlgo, kSortedIndex, kAbortWhenNanFound, bfloat16_t, int32_t, kCandBias, kInputBias>(
                    valid_len, topk, row_in, row_val, row_idx,
                    /*return_value=*/true, row_ks,
                    /*idx_oob_fill=*/static_cast<int32_t>(2147483647),
                    /*value_oob_fill_bits=*/static_cast<uint16_t>(0xFF80));
            else
                qli_topk::topk_select_run<
                    UpperAlgo, kSortedIndex, kAbortWhenNanFound, bfloat16_t, int32_t, kCandBias, kInputBias>(
                    valid_len, topk, row_in, row_val, row_idx,
                    /*return_value=*/true, row_ks,
                    /*idx_oob_fill=*/static_cast<int32_t>(2147483647),
                    /*value_oob_fill_bits=*/static_cast<uint16_t>(0xFF80));

            // Drain this row's output stores before the next row's MTE2 reads.
            asc_sync_mte3(6);
        }
    };

    mqa_logits_core_impl<
        kIsFP4, kNumHeads, kHeadDim, kSplitKV, kMadM,
        kNumCDStages, kNumQStages, kNumKVStages, kAccumStages,
        kNumLogitsStages, 0u, false, decltype(make_scheduler), decltype(on_block_done)>(
        q, kv, sf_q, sf_kv, weights, logits, make_scheduler, on_block_done);
}

} // namespace deep_gemm