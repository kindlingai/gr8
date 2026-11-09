// gr8 DeepSeek-V4.1-Flash read-once prefill MoE: the production config of the expert loop. One place for the adapter
// (ds41_moe_ops.cuh experts_ro) and the GPU test (tests/ds_prefill_ro_test.cu static_asserts its production variant against these). No CUDA dependency.
//
// On four real route sets (layers 0, 2, 20, 39 of a 1200-token prompt) fc1 KS8 CW2 NTG4 + fc2 W8 D3 NTG4 is the fastest of all 9 x 6 bench variants on
// every set (fc1 + fc2 1.23x to 1.32x faster than KS2 CW4 NTG4 + W8 D2 NTG4); on uniform n <= 16 it is within 1.6% of the best. It loses on uniform
// n = 32..128 (1.1x to 1.5x), a shape real routes do not have.
// Numerics: fc1 results depend on KS only (the k split and its fixed reduce order). KS = 8 is bitwise equal to the one-token path (0 of W tokens differ,
// 18 of 18 cases); KS = 2 is not (79 values of the W1024 output within 2.45e-3 of the row max). fc2 results do not depend on W, D, NTG.
// Override at compile time with -DDS41_RO_KS=2 etc. (the GPU test static_asserts that its table row matches whatever is set here).
#pragma once
#ifndef DS41_RO_KS
#define DS41_RO_KS 8                                  // fc1: k parts per chunk (a warp each)
#endif
#ifndef DS41_RO_CW
#define DS41_RO_CW 2                                  // fc1: chunks per CTA
#endif
#ifndef DS41_RO_NTG1
#define DS41_RO_NTG1 4                                // fc1: tiles per group
#endif
#ifndef DS41_RO_W2
#define DS41_RO_W2 8                                  // fc2: warps (output tiles) per CTA
#endif
#ifndef DS41_RO_D2
#define DS41_RO_D2 3                                  // fc2: record prefetch depth
#endif
#ifndef DS41_RO_NTG2
#define DS41_RO_NTG2 4                                // fc2: tiles per group (dynamic smem NTG2 * 9216 B)
#endif
// GR8_DS_MOE_TC=1 (ds_moe_tc.cuh): the tensor-core expert loop's shape, measured in tests/ds_moe_tc_bench.cu on one GB10 at 1024 rows x 60..140 distinct
// experts: fc1 CW 4 (9 CTAs of 4 warps per expert, 2 CTAs per SM at 44 KB smem), NTG 8 (64-token groups), a 4-deep record ring, GY 4 (an expert's token
// groups on 4 neighbouring CTAs: with GY 1 the second group of an expert read its weights from DRAM again, 1.8 x the unique bytes); fc2 4 warps x 2 token
// tiles (64-token groups, the rows in registers), the expert's records once per group in stages of 8 output tiles through an 8-deep ring (2 CTAs per SM),
// the output tiles over 16 CTAs (OS), GY 4, and the combine fused into fc2 (DS41_TC_COMBINE: the CTA that completes a token's routes in a slice sums
// them in route order; os is the slowest block index so a slice's partials stay in L2 and are discarded after the sum). fc1 990..1380 us (59..73 TFLOP/s)
// against 3.1..4.1 ms for the fp16 loop; fc2 + combine 915..1160 us against 1240..1550 for fc2 + k_combine_mt and 2.1..2.3 ms for the fp16 loop's.
#ifndef DS41_TC_CW
#define DS41_TC_CW 4
#endif
#ifndef DS41_TC_NTG1
#define DS41_TC_NTG1 8
#endif
#ifndef DS41_TC_ST1
#define DS41_TC_ST1 4
#endif
#ifndef DS41_TC_GY1
#define DS41_TC_GY1 4
#endif
#ifndef DS41_TC_W2
#define DS41_TC_W2 4                                  // fc2: warps per CTA
#endif
#ifndef DS41_TC_WT
#define DS41_TC_WT 2                                  // fc2: token tiles per warp
#endif
#ifndef DS41_TC_TPC
#define DS41_TC_TPC 8                                 // fc2: output tiles per ring stage
#endif
#ifndef DS41_TC_ST2
#define DS41_TC_ST2 8
#endif
#ifndef DS41_TC_GY2
#define DS41_TC_GY2 4
#endif
#ifndef DS41_TC_OS2
#define DS41_TC_OS2 16                                // fc2: CTAs an expert's output tiles split over (blocks of 8 tiles dealt as evenly as 40 / OS2 allows)
#endif
#ifndef DS41_TC_COMBINE
#define DS41_TC_COMBINE 1                             // 1: fc2's last-route CTA sums a token's slice (no combine launch, no ypart round trip); 0: dsh::k_combine_mt
#endif
// Shared expert (ds_shared_gl.cuh): DS41_SH_GL = 1 runs it as two launches per layer (k_sh_fc1_gl / k_sh_fc2_gl, grid.y = row groups) instead of 2 x W/32 launches of the
// 32-row chunk kernels. Bit-identical by construction (same per-chunk bodies); the gate is `ds_prefill_ro_test check`, which matches bitwise in the final output and in the
// shared fp32 sout / fp16 sact. -DDS41_SH_GL=0 is the chunk loop.
// DS41_SH_NT tiles of 8 rows per chunk (4 = the 32-row chunk of the chunk loop, 8 = 64 rows), DS41_SH_RG = the most row groups (CTA rows of the grid; chunks are dealt round robin).
#ifndef DS41_SH_GL
#define DS41_SH_GL 1
#endif
#ifndef DS41_SH_NT
#define DS41_SH_NT 8
#endif
#ifndef DS41_SH_RG
#define DS41_SH_RG 4
#endif
