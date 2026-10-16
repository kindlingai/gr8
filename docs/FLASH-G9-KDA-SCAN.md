# G9 — KDA scan + conv prefill kernel: design (CPU, step 1)

G9 lane, 2026-10-16. Branch `flash-G9` off base `a5edf0a`. CPU desk work only: no Spark, no

Rev 2 (after KA review of 0126253): OF path in out16, test readers of OG, K2 tail zfill, plane layout, T0 over all
functions, K2 residency, exp count, small number fixes.
Rev 3 (after KA review of e4d6d67): register cap 200 and smem gate 49 KB, min-blocks by VS, F1 saving 13.50 E, tail
loads predicated in conv16/gz16, bench arm calls kda::g9 directly, nits.

Labels: **[MEAS]** measured file (path given). **[SRC]** read in someone else's source or doc. **[EST]** my arithmetic
(script given). **[PLAN]** an assumption I chose, not a measurement.

## 1. Goal and the gap

- Item #9 of the gx10 ledger: KDA scan + conv + gate, 32K, TP2, per rank. Ours **3.040 s** pf36 kernels [MEAS],
  **3.396 s** "now" [EST]; gx10 **0.912 s** [EST] (TP4 0.456 s × 2) — `flash/engine/prefill/GX10-LEDGER.md`
  (sha256 a18d4315) §3/§4. Gap ~2.5 s.
- Our stage split (pf36) [MEAS]: intra3 1.070, state2 0.813, conv 0.481, gz 0.406, out 0.270 s.
- gx10 [MEAS TP4 rank 0]: FlashKDA prepare 0.134 + recurrence 0.130, causal_conv1d 0.138, layer_norm_gated 0.054 s.
  Their f_b/g_b GEMMs sit in their bf16 GEMM row, not in this row; our gz (0.406 s) is in this row.
- Rule for this round: SPEED FIRST; quality is a report line, not a gate. OFF stays
  byte-identical to a5edf0a.

## 2. Prior work this design builds on

| item | result | what G9 takes from it |
|---|---|---|
| KDA scan v2, flash-PF `d551181` | row-parallel conv, tiled gz, intra3, state2: 75 → 25 ms/layer at 9216 rows | the OFF path; conv op order (kept for the ON conv) |
| KA floor table `10deb51` | current scan 481.4 GB logical → **2.102 s** byte floor; actual 3.064 s; 148.3 G pair exps | the byte convention (one-pass logical bytes, 229 GB/s roof) used in §5 |
| KA chunk-parallel note (`chunk_parallel_design.md`) | affine chunk regrouping fails exact FP32 order (1-ulp CPU counterexample) | no affine/tree scan; chunks stay serial in the state |
| KA tensor-core scan note (`evidence/tc_scan/DESIGN.md`) | no exact tensor-core route; non-exact B2 (CH64, bf16 MMA, compact tape) planning −0.635 s | ON cannot be byte-exact; G9 is a precision arm. G9 goes further than B2: CH16 removes the pairwise exps entirely |

## 3. What gx10 does (FlashKDA 17a037d, read as reference only, no code copied)

Read from `github.com/vllm-project/FlashKDA` at 17a037d (local copy `/tmp/g9-ref/flashkda`) [SRC]:

- **Chunk 16**, not 64 (`csrc/smxx/fwd_launch.cu:43`; `docs/20260420-flashkda-v1-deep-dive.md` §1). With the gate
  floor −5, the cumulative log-decay inside 16 rows stays in [−80, 0], so exp(±gc) fits fp32/bf16 range and the
  pairwise decay factorizes: `exp(gc_i − gc_j) = exp(gc_i) · exp(−gc_j)`. The 16×16 L and Mqk become two bf16 MMAs.
- **Two kernels** (deep-dive §2): K1 prepare, grid (16-row tile, head), does gate act + l2norm + decay + L/Mqk +
  16×16 inverse and writes a bf16 workspace (k·e^gc, q·e^gc, k·e^(gt−gc), g_total fp32, INV, Mqk). K2 recurrence,
  grid (head × V-half) when it fits one wave (`fwd_launch.cu:44,190,205`, VD = 64), walks the 16-row chunks in order.
  They report the split beat a single fused kernel by ≥15 % on their GPUs.
- **State fp32** across chunks in registers; a bf16 copy feeds each MMA (`fwd_kernel2.cuh:521-523`). (The older
  b59532f rounded state to bf16 every 16 rows; gx10 README.md:428 moved to 17a037d because long prefills broke.)
- Inputs bf16: q, k, v from `causal_conv1d` written as three dense tensors (gx10 `experimental/fixes/kda.py:600-617`),
  raw f_b logit g bf16, raw beta bf16; gate `lower_bound · sigmoid(exp(A_log)·(g + dt_bias))` in-kernel.
- Approximations: sigmoid via `tanh.approx`, `ex2.approx.ftz` (`fwd_kernel1.cuh:244,305,453`).

## 4. G9 design

### 4.1 Math (exact in real arithmetic; same algebra as prefill.h with CH = 16)

Per head, per 16-row chunk (rows of the engine call, from row 0), with gate floor GATE_LO = −5:

    g  = GATE_LO · sigm(exp(A_log[h]) · (G + dt_bias))    ∈ [−5, 0]       gc = cumsum16(g), gt = gc[last]  ∈ [−80, 0]
    p  = gc[7]  (per-channel pivot, row 7 of the chunk)
    kdp = kn·e^(gc−p),  qdp = qn·QSCALE·e^(gc−p),  kip = kn·e^(p−gc)            exponents in [−40, 40]
    L[i][j]   = beta_i · Σ_d kdp_i kip_j    (j < i)        Mqk[i][j] = Σ_d qdp_i kip_j   (j ≤ i)
    INVb = (I + L)^−1 · diag(beta)       (fp32 forward substitution, 16 steps)
    kd = kn·e^gc,  qd = qn·QSCALE·e^gc,  kr = kn·e^(gt−gc)
    per chunk, in order:   u = INVb (v − kd S);   O = qd S + Mqk u;   S = diag(e^gt) S + kr^T u

This is the current prefill.h algebra (W = Ai·diag(β)·kd, U = Ai·diag(β)·v, VN = U − W S = u, AQ = Mqk,
KD = kr) regrouped for CH = 16. Range: g ∈ [−5, 0] because sigm ∈ [0, 1] in fp32, so |gc| ≤ 80 + rounding;
e^80 = 5.5e34 < 3.39e38 (bf16/fp32 max). The **pivot p is my addition** (FlashKDA does not use one): without it,
kn·e^gc goes subnormal for |kn| < 6.5e-4 at gc ≈ −80 and is flushed, while its partner kn·e^(−gc) is ~e^80, so an
O(|kn_i kn_j|) term of L is lost. With p = gc[7] the exponents stay in [−40, 40] and the flush needs |kn| < 3e-21.
It costs up to one extra exp-class op per element in K1 (compute, no bytes; count in §5). kd/qd/kr for the state products keep the unpivoted
form: their underflow only drops terms that are already below 1e-38 · |S|.

### 4.2 Kernels (ON path, flag value 1 = "split")

All new code goes in a new header `engine/kda/kda_g9.cuh`, plain CUDA with `mma.sync.m16n8k16` bf16→fp32 and
`ldmatrix` (sm_80+ PTX; no CUTLASS, no TMA). Dims fixed: DK = DV = 128, CH16 = 16.

1. **conv16** — CTA (8-row block, head), 384 threads = 3 streams × 128 channels. Each thread walks its 8 rows with
   a 4-tap sliding window in registers (P q|k|v fp32 read once + 3 halo rows; rows < 3 take the slot's history,
   as k_kda_conv_pf). silu(conv) uses **the same op order as k_kda_conv_pf**. Whether the fp32 pre-norm values then
   equal today's C bit for bit depends on FMA contraction in the compiled code; T2 checks it, it is not assumed. Then fp32 sum of squares per row for q and k (warp shuffle + 4-warp smem combine),
   `qn = q / sqrtf(ss + L2_EPS)`, `kn` likewise, and one round each to bf16: dense `Qn, Kn, V` [rows][dl].
   `k_kda_conv_end_pf` (history update) is reused unchanged, launched after conv16.
2. **gz16** — tensor-core GEMM [M × 128] · [128 × dl] for f_b and g_b. A = f_a / g_a columns of P (fp32 → bf16 on
   load), B = f_b / g_b bf16 weights ([n][k] row-major → ldmatrix non-trans). CTA tile 64 rows × 128 cols (one head),
   K = 128 in smem. Output raw `G16` (f_b logit, gate applied in K1) and `Z16`, bf16.
3. **K1 prepare** — CTA (16-row tile, head), 128 threads, thread = channel d. Loads Qn/Kn/G16 tile rows, beta
   (sigm of P's b column, fp32). Thread d: gate, cumsum down 16 rows in fp32 (registers), pivot, the exp-class
   ops (≤ 5 per element excluding the gate's reciprocal, §5), writes kd, qd, kr bf16 to global; kdp, qdp, kip bf16 to smem. Warps 0-1: L = kdp·kip^T and
   Mqk = qdp·kip^T (16 MMAs each, fp32 acc); mask; β-scale L. 16 threads: fp32 forward substitution of (I + L)^−1 in
   smem, column-parallel; fold diag(β); write INVb, Mqk bf16 (16×16 each) and gt fp32 [128]. Tail tile: rows ≥ n
   (n = valid rows of the last tile, `M − 16·tile`) get kn = qn = 0, β = 0, g = 0 *as register/smem values* (Qn/Kn/G16
   loads for rows ≥ M are predicated, zero-filled, never dereferenced). Then those rows add nothing, INVb rows and
   columns ≥ n are exactly 0 and gt = gc of the last real row. K1 stores its kd/qd/kr rows only for rows < M.
4. **K2 recurrence** — CTA (head, v-slice of 128/VS columns), VS ∈ {1, 2, 4} (template; env `G9_KDA_VS`, default 2 →
   64 CTAs at TP2 on 48 SMs). Math warps = 8/VS; one producer warp streams chunk tiles (kd, qd, kr, v-slice, INVb,
   Mqk, gt) with `cp.async` into a 3-stage smem ring (~16 KB/stage at VS = 2).
   **Tail rows (K2).** K1 never stores rows ≥ M, so a plane row ≥ M holds stale bytes (aliased fp32 scratch); ~1/256
   of bf16 patterns are NaN/Inf and 0 × NaN = NaN through r^T·INVb^T and S += u^T·kr. The producer therefore loads
   kd, qd, kr, **and V** rows ≥ M of the last tile with `cp.async` src-size 0 (zfill), predicate `row < M`, and never
   reads past plane row M. INVb/Mqk/gt are always full 16×16 tiles written by K1 (zeros for pad rows), so they need
   no predicate. With u rows ≥ n exactly 0 (INVb rows ≥ n = 0) and kr rows ≥ n = 0, the state update is exact for
   the real rows. K2 stores O rows < M only.
   **Residency.** `__launch_bounds__(32·(8/VS + 1), VS == 1 ? 1 : 2)`: at VS = 2 and 4, 2 CTAs/SM are co-resident, so VS = 2 (64 CTAs) fits one wave on
   48 SMs (up to 96 slots; 16 SMs hold 2 CTAs, 32 hold 1 — the per-SM streaming rate must carry two CTAs; the byte
   floor is global, the SM imbalance is a risk). Checked at build, VS = 2: ptxas ≤ **200** regs/thread (registers
   are allocated per warp in units of 256: 208·32·10 warps = 66560 > 65536, 200·32·10 = 64000 fits; standard occupancy
   rule, not measured on GB10), 0 spills; static + dynamic smem ≤ **49 KB**/CTA (GB10 `shared_per_sm` = 102400,
   `engine/results/flash-kpool-stream-screen.log:1`; two CTAs plus ~1 KB reserve each). The VS = 2 ring is 3 × 15872 =
   47616 B, so barriers and the u exchange must fit in the remaining ~1.4-2.5 KB. **VS = 1 uses min-blocks 1** (288
   threads × 2 would cap regs at ~112 and spill the 64 fp32 S^T accumulators; VS = 1 is only 32 CTAs, one per SM, and
   is the F1/F2 default and the F3 requirement). If the VS = 2 check fails, VS = 4 (128 CTAs, 96 threads) is the fallback; a second
   wave would double K2. VS = 1 gives 32 CTAs = 32 of 48 SMs busy (F3 needs VS = 1, so its K2 cannot use the other
   16 SMs; per-SM bandwidth may then cap it — F3 stays only if measured faster).
   **VS > 1 costs.** Each of the VS v-slice CTAs of a head re-reads the whole K1 tape (kd, qd, kr, INVb, Mqk, gt;
   L2 hits expected, since sibling CTAs walk the same chunk at nearly the same time; §5 counts DRAM bytes once; not
   measured). In F1/F2 the K1 work (and in F2 the conv) is **duplicated per v-slice CTA**, VS× the compute; so F1/F2
   default to VS = 1 and VS > 1 there is a measured option only.
   **State kept transposed, S^T [v][d], as fp32 MMA accumulators**: a warp owns 16 v rows × 128 d = 64 fp32
   registers/thread. Its bf16 repack *is* the m16n8k16 A-fragment layout (accumulator → A operand, no transpose,
   no movmatrix, no smem round trip). Per chunk, per warp:
   - X = bf16(S^T)·kd^T and Y = bf16(S^T)·qd^T (K = 128: 8 k-steps × 2 n-tiles each; kd/qd `[row][d]` = B via ldmatrix);
   - r^T = v^T − X (fp32) → bf16 A-fragment; U^T = r^T · INVb^T (1 k-step); U = u of §4.1;
   - O^T = Y + bf16(U^T) · Mqk^T → bf16, staged in smem, 16-byte stores to O [row][h·128 + v];
   - S^T = S^T ∘ e^gt[d] (fp32, per accumulator column) + bf16(U^T) · kr (16 n-tiles, kr `[row][d]` via ldmatrix.trans).
   416 MMAs per (chunk, head) in total. S loads/stores fp32 [d][v] from the slot once per engine call (layout of
   `st_cell`, unchanged for decode).
5. **out16** — warp per (row, head): lanes hold 4 v each; O and Z bf16 in; fp32 sum of squares by shuffle;
   `y = o · r · onorm · sigm(z)`. It writes **exactly the pointers out_2 (math.h:353-359) writes, minus OG**:
   `if (a.OF) a.OF[frag_u16(m, h·DK + t, dl)] = f2bf_rn(y)` and `if (a.OH) a.OH[i] = f2h_rn(y)` — same index
   functions, same roundings. `make_args` (kda.cuh:351-352) sets OF (OH = null) when `w.ot.w` is null (BF16W
   o_proj), and `pf_kda_block` (flash_prefill.cuh:470-476) then reads `ksc.OF`; out16 covers that path, so a BF16W
   KDA o_proj is not a hole. Rows ≥ M are never stored (OF/OH pad rows are never written; their content is don't-care because pad outputs are discarded). OG fp32 is not
   written: no runtime reader in prefill, decode or MTP (the engine reads `ksc.OH`/`OF`; decode uses `m.kda_sc`).
   The *test* readers of OG are handled in §6.

**Scratch: no new allocation; plane layout.** E = MP·dl elements (MP = allocated rows, dl = nh·128). Every plane is
row-major `[row][dl]`, row stride dl elements, MP·dl·2 B per plane. All bases are `cudaMalloc` (256 B aligned) plus
multiples of MP·dl·2 B (dl multiple of 128, so 16 B aligned). Writers store rows < M only; the tile kernels (conv16
8-row blocks, gz16 64-row tiles, K1 16-row tiles) predicate their stores **and their loads** on `row < M` (zero-fill
as `k_kda_gz_pf` does, `kda_prefill.cuh:78`). The engine's P has MP % 64 == 0 rows, but `kda_pf_bench` allocates P with
exactly M rows (`kda_pf_bench.cu:68`) and T2 uses M = 17 and 8191, so an unpredicated tile load would read past the
allocation.

| array (alloc) | B per E | ON content |
|---|---:|---|
| `ksc.C` (MP × 3·dlmax fp32) | 12 | six bf16 planes at byte offsets k·MP·dl·2, k = 0..5: Qn, Kn, V, G16, Z16, O16 (exact fit when dl = dlmax, otherwise smaller) |
| `kps.QE`, `kps.KD`, `kps.W` (MP × dl fp32 each) | 4 each | kd, qd, kr: one bf16 plane at offset 0 each (half used) |
| `kps.U` (MP × nhmax·128 fp32 = MP·nhmax·512 B) | 4 | block (head h, tile t) at byte `(h·NT + t)·1536`, NT = ceil(M/16): INVb bf16 [16][16] at +0, Mqk bf16 [16][16] at +512, gt fp32 [128] at +1024 (1536 B = 3 × 512 B, 16 B aligned; head-contiguous so one K2 CTA streams one run) |

Assert at dispatch: `ceil(M/16)·3 ≤ MP` (U capacity NT·1536 B ≤ MP·512 B per head; this also demands MP ≥ 3, which the
engine's MP % 64 == 0 satisfies, `flash_prefill.cuh:230`), `M ≤ MP`, `dl ≤ ksc.dlmax`. Outside the
scan only `ksc.P` (inputs) and `ksc.OH` / `ksc.OF` (output) are read (`flash_prefill.cuh:352-476`, including :443 and
:470-476). `ksc.G/Z/O/OG` and `kps.GC/AQ` are not touched under ON; `ksc.C` and `kps.*` hold the aliased bf16 data.
`kda_g9.cuh` adds no `__device__` globals, no macros, and changes nothing in `math.h`, `prefill.h` or any constant
(CH = 64 etc.); everything is in `namespace kda::g9` as `inline` / template functions and new `__global__` kernels.

### 4.3 Fused increments (later flag values; each reuses the step-before code as device functions)

| flag | change | why |
|---|---|---|
| 2 = F1 | K1 runs in producer warps of the K2 CTA, one chunk ahead; kd/qd/kr/INVb/Mqk stay in smem | removes the K1 workspace write + read (13.50 E of 51.97 E bytes; script: (237.154 − 175.548) GB / 34 layers / E) |
| 3 = F2 | conv also moves into the fused CTA (rows walk in order, so halo rows stay in registers) | removes the bf16 q/k/v write + read; conv16 is the largest split stage (0.36 s floor) |
| 4 = F3 | out norm + gate in the K2 epilogue; needs VS = 1 (CTA owns a whole head row) | removes O write + read |

KA's P3 lesson applies: F1-F3 put token-parallel work on the head-serial path. Per-chunk prep at CH16 is ~2 K
elements of gate/exp + 32 MMAs + a 16-step solve, not P3's 266 K pair exps per CH64 chunk. Each F step is kept only
if its standalone time beats the step before it (§6, T4).

### 4.4 Flag and dispatch

- Env `G9_KDA_SCAN` read once (`static const`, as the existing `G6_KDA_PF_*` flags) in `launch_core_prefill`.
  Unset or 0 → the existing launches, untouched. 1/2/3/4 → `kda::g9::launch(...)`.
- **Interaction with the old flags.** `G6_KDA_PF_P1/IN1/ST1` select kernels inside the OFF path only; under ON they
  are ignored (the ON branch returns before them), with one log line if any is set. `G9_KDA_VS` ∈ {1, 2, 4} is read
  only under ON (default 2; flag 4 forces 1 and logs it; F1/F2 default to 1, §4.2). Any other value is a `die`.
- No struct, allocation, default or other call-site change. `kda_g9.cuh` is included by `kda_prefill.cuh`, so it is
  compiled into every TU that includes it (`flash_prefill.cuh:22`, `kda_test`, `kda_pf_bench`): T0 therefore diffs all
  pre-existing functions of those cubins, not only KDA kernels (§6).

## 5. Cost bound per 32K prompt per TP2 rank (34 KDA layers)

Script: `engine/results/g9-kda-scan-budget.py` → `g9-kda-scan-budget.json`. Same convention as 10deb51: one-pass
logical bytes (each array once per stage boundary), 229 GB/s roof. E = rows × dl = one [M][4096] array.
NOT measured DRAM traffic; not a runtime prediction.

| stage | bytes/layer (× E) | 34-layer byte floor [EST] |
|---|---:|---:|
| conv16 (P fp32 in, bf16 Qn/Kn/V out) | 18.0 | 0.359 s |
| gz16 (bf16 G/Z out) | 4.31 | 0.086 s |
| K1 (Qn, Kn, G16 in; kd, qd, kr, gt, INVb, Mqk out) | 12.78 | 0.255 s |
| K2 (K1 tape + V in; O bf16 out; fp32 state per call; no beta read, folded into INVb) | 10.88 | 0.217 s |
| out16 (O, Z in; OH out) | 6.0 | 0.120 s |
| **G9 split (flag 1)** | **51.97** | **1.036 s** |
| F1 (flag 2) | | 0.767 s |
| F2 (flag 3) | | 0.527 s |
| F3 (flag 4) | | 0.448 s |
| gx10 path, same convention (bf16 conv in) | 41.5 | 0.827 s (their EST 0.912 s) |

- **Compute is not the limit**: 6.43 TFLOP of m16n8k16 work over 34 layers (gz 16.8 M, K1 2.1 M, K2 27.3 M MMAs per
  layer) = 0.052 s at the 122.6 TF/s roof. Exps: ≤ 22.9 G (5 exp-class ops
  per element, not counting the gate's reciprocal (math.h:221): gate exp, e^gc, e^(gc−p), e^(p−gc), e^(gt−gc); no sharing by multiplication identities, so the real
  count is lower) vs 148.3 G pair exps today. Compute only, not a byte term.
- Conv-history bytes (3 rows × 3 streams per call) are ~0.01 E and not in the table.
- **K2 serial latency**: at the byte floor, each CTA has **3.1 µs per 16-row chunk** (2048 chunks/layer). Planning
  chain [PLAN: assumed 32-cycle dependent MMA latency + 300 cycles of loads/conversion/barriers, 1989 MHz] ≈
  **0.33 µs**. So K2 should be bandwidth-bound, not latency-bound, with ~10× slack. Not measured.
- **Planning range** [EST] = floor × (measured / floor) ratio, low = gx10's own 1.10, high = our current 1.45:
  split **1.14-1.50 s**, F1 0.85-1.11 s, F2 0.58-0.76 s, F3 0.49-0.65 s. Versus 3.040 s, split saves ~1.5-1.9 s
  and F2/F3 ~2.3-2.5 s. These are planning numbers only; §6 T4 is the measurement.
- Note: if a projection arm (KA) later writes q/k/v as bf16, conv16 drops from 18 E to 12 E (−0.12 s). Not in scope.

## 6. Exactness plan

**ON is not byte-exact and cannot be** (KA tc_scan §a: bf16 operand casts and unspecified MMA sum order; also CH16
regrouping). Per the brief, ON is a **precision arm with a quality line**. OFF must be byte-identical.

Rounding points of ON (every other value is fp32):
1. Qn, Kn, V: one bf16 round after fp32 conv/silu/l2norm (gx10 rounds q/k/v three times: proj, conv, norm).
2. G16, Z16: f_a/g_a rounded to bf16 as MMA operands; outputs rounded to bf16.
3. K1: kdp/qdp/kip/kd/qd/kr rounded to bf16; L/Mqk fp32-accumulated; INVb solved in fp32, stored bf16; Mqk bf16.
4. K2: bf16 copy of S^T per chunk for the MMA; r^T, U^T rounded to bf16 as MMA operands; S accumulates fp32 always;
   O rounded to bf16.
5. out16: fp32 math; OH fp16 (same final rounding as today).


- **T0 OFF identity, CPU.** (a) Diff review: OFF executes the same kernels with the same launch arguments.
  (b) `cuobjdump -sass` of **every pre-existing function** (not only KDA kernels; `kda_g9.cuh` compiles into every TU
  that includes `kda_prefill.cuh`) in sm_121a cubins of `kda_pf_bench.cu`, `kda_test.cu` and the engine TU(s) that
  include `flash_prefill.cuh`, base a5edf0a vs flash-G9, identical flags: per-function byte-equal SASS required; new
  functions (`kda::g9::*`) are excluded from the diff and listed. `cuobjdump -res-usage` is equal for all
  pre-existing functions. (c) existing CPU tests (`kda_cpu_test`) give identical output.
- **T1 ON algorithm, CPU.** `engine/kda/tests/kda_g9_cpu.h`: the exact ON algorithm on the CPU with bf16 rounding
  at the same points (round-to-nearest-even emulation; MMA sums in a fixed order). Compare against the current CPU
  model (prefill.h through kda_cpu.h) and an fp64 reference, on `golden/kda-L00-4096.safetensors` and synthetic
  stress: rows {1, 15, 16, 17, 63, 64, 65, 4096}, heads {1, 3, 32}, nonzero state and history, weak/strong decay
  (A_log, dt_bias pushing g to 0 and to −5), small/large |k|. Report max-abs and rel-L2 of O, OH and final S, and
  the error growth over 4096 rows. This produces the **predicted quality line before any GPU run**.
  **Depth requirement (accepted):** T1 also runs the full engine depth, **32768 rows of one slot (4 calls
  of 8192, state carried between calls)**, not only 2K-4K. It reports the KDA state S error (max-abs, rel-L2 vs the
  fp32 reference = the current CPU model, and vs fp64) at checkpoints 2K, 4K, 8K, 16K, 32K, plus O/OH error at
  the same depths, for synthetic stress inputs (weak/strong decay) and for the golden L00-4096 tiled to depth where
  real data ends. The 32K S error is the number that goes on the card beside the speed row. Scan error compounds
  along the sequence, so the 2K number alone is not accepted.
  (a bug detector, not a quality gate: only MMA sum order differs). Conv history and untouched state slots and
  padding **byte-equal** to OFF. No NaN/Inf. Each kernel (conv16, gz16, K1, K2, out16) is checked alone first.
  Compared on **OH** (and **OF** in a BF16W-config case, `w.ot.w == null`); OG is not written under ON. conv16
  pre-norm fp32 values vs today's C are compared bitwise (depends on FMA contraction; a mismatch is reported, not
  assumed away). **Tail/poison case:** M ∈ {1, 15, 17, 8191} with the aliased bf16 planes pre-filled with 0xFFFF (NaN)
  beyond row M: outputs must be finite and equal to the clean-fill run (tests the K2 zfill and K1 predicates).
  **Existing test readers:** `kda_pf_bench.cu:115-135` (PASS gated on OG) and `kda_test.cu:243-265` (reads
  sc.C/G/O/OG) would see aliased or zero data under ON. Under ON the bench gates on OH (rel-L2 vs the OFF run's OH,
  plus a finite check); `kda_test` chunk mode `die`s when `G9_KDA_SCAN != 0`.
  **Bench arm.** `kda_pf_bench` launches its kernels directly (`:96-109`), so `G9_KDA_SCAN` never reaches it. The ON arm
  calls `kda::g9::launch` itself, and the reference (OFF) run must allocate OH and set `a.OH` (the bench leaves
  `a.OH`/`a.OF` null, `:57-72`) so the two arms are compared on OH.
  logged; per kernel times; ptxas registers/smem/spills. VS ∈ {1, 2, 4}. Then 45L 32K TP2 ON vs OFF, every number
  with branch + commit.
  KL(p_OFF‖p_ON) mean/p95/max and ΔNLL, plus long-prefix error growth. Reported with every ON speed row.

## 7. Build plan (step 2, after KA PASS)


1. `kda_g9_cpu.h` + `kda_g9_cpu_test.cpp` (T1) → first quality prediction.
2. `kda_g9.cuh`: conv16 + out16 (simplest) behind flag; compile-only on CPU (nvcc sm_121a), T0 SASS diff.
3. gz16, K1, K2 (VS template); `kda_g9_test.cu` (T2), bench arm in `kda_pf_bench.cu`.
4. Flag 1 wired in `launch_core_prefill`; T0 again.
5. After GPU T2-T5 on flag 1: F1, then F2, then F3, each kept only if faster.

## 8. Risks

- K2 at 64 CTAs may not reach the roof if one SM cannot stream ~16 KB per 3 µs; VS = 4 (128 CTAs) is the fallback.
- gz16 rounding f_a to bf16 changes the gate logit by ~2^-9 relative; if T5 shows it matters, keep gz in fp32 inputs
  (TF32 or scalar) with bf16 outputs (+0.05 s floor).
- INVb in bf16: (I + L)^−1 can have entries > 1 when L is ill-conditioned; T1 stress cases measure it. Fallback:
  keep the u·INVb product in fp32 FMA (16×16, small) instead of an MMA.
- Fused increments may lose to the split (FlashKDA saw that on GB200/H100). The split alone is a complete ON arm.

## 9. Not done in this step

No kernel code, no CUDA build, no GPU or Spark use, no quality number. All times above are [EST]/[PLAN].

## 10. T1 result (CPU, flash-G9 a67bcf6; log `engine/results/g9-t1-cpu-32k.log`)

`kda_g9_cpu_test` (CPU only): ON emulation of §4 (bf16 RNE at the §6 rounding points, fp32 state, fixed fp32 sum order,
ftz on exp results) vs the OFF CPU model (prefill.h via kda_cpu.h, fp32) vs an fp64 token-by-token gated delta rule.
32768 rows of one slot in calls of 2048 rows, state carried (chunk-aligned, so bit-identical to 4 calls of 8192; checked
for OFF: PASS). 8 synthetic stress cases (weak / strong / mid / mixed decay, high beta, no decay dt = −12, correlated
keys) and golden L00 heads {0, 17, 33, 60} tiled 8× to 32K. Not modeled: GPU MMA sum order, ex2.approx/tanh.approx
accuracy, conv16 pre-norm bits, f_a operand rounding in the golden cases (modeled in the synthetic ones).

State S error of ON vs the OFF fp32 model (rel-L2 over the [128][128] state), by depth:

| case | 2K | 4K | 8K | 16K | 32K | max-abs at 32K |
|---|---:|---:|---:|---:|---:|---:|
| weak | 3.97e-3 | 4.15e-3 | 4.25e-3 | 3.93e-3 | 4.16e-3 | 9.5e-3 |
| strong | 3.10e-3 | 2.97e-3 | 3.40e-3 | 2.87e-3 | 2.66e-3 | 2.0e-3 |
| mixed | 3.10e-3 | 3.21e-3 | 3.30e-3 | 3.65e-3 | 4.19e-3 | 2.3e-2 |
| no decay | 4.57e-3 | 4.77e-3 | 4.57e-3 | 4.66e-3 | 4.68e-3 | 2.3e-2 |
| corr-nodecay (worst) | 7.77e-3 | 7.78e-3 | 8.30e-3 | 8.41e-3 | 8.83e-3 | 6.5e-2 |
| golden L00 h0 / h17 / h33 / h60 | 2.9e-3 / 1.6e-3 / 5.3e-3 / 4.3e-3 | | | | 2.6e-3 / 1.4e-3 / 3.6e-3 / 2.7e-3 | 1.7e-5 / 1.1e-4 / 3.5e-5 / 1.5e-5 |

- **No growth with depth.** The ON state error sits at 0.15-0.9 % rel-L2 (bf16 operand rounding is ~0.4 %) from 2K to 32K in
  every case; the worst, correlated keys with no decay, drifts 7.8e-3 → 8.8e-3 over 32K. The OFF model is 1e-7 to 1e-5
  from fp64; ON vs fp64 equals ON vs OFF to the digits shown.
- Output over the same windows (rel-L2 vs OFF): O 2.1e-3 to 8.9e-3, gated y 2.2e-3 to 9.1e-3, OH (fp16) the same
  within 2e-4 (fp16 rounding of OFF itself is 2.1e-4). No growth with depth.
- Golden L00 state vs the python fp32 S at 4096: ON 1.5e-3 to 3.7e-3, OFF 1.6e-4 to 3.2e-4. The golden layer decays fast, so
  tiling to 32K repeats the same state; only the synthetic no-decay cases test long accumulation.
- Small calls from a nonzero state, rows {1, 15, 16, 17, 63, 64, 65, 4096}: ON S 0.9e-3 to 3.4e-3, O 3.5e-3 to 4.1e-3
  vs fp64; OFF ≤ 4e-7. The 10 PASS lines of the log are bug detectors (ON ≤ 5e-2 vs fp64), not a quality gate.
  doc's rounding choices; the INVb-in-bf16 fallback of §8 is not needed at these errors.

## 11. Implementation status (kda_g9.cuh, compile only; nothing run on a GPU)

ptxas 13.1 sm_121a, 0 spills everywhere: conv16 40 regs; out16 33; gz16 50 regs, 34816 B smem; K1 96 regs, 16896 B smem;
**K2 VS=2 154 regs (cap 200), 47616 B dynamic smem, no static smem (margin 2560 B kept; barriers are named, not in smem)**;
K2 VS=4 154 regs; K2 VS=1 168 regs (53760 B, needs the max-dynamic-smem attribute, set in launch_k2).
Differences from the text above (all inside kda_g9.cuh): gz16 tile is 64 rows x 64 channels (8 warps) so smem stays
under 48 KB static; the K1 "gt" slot holds **e^gt** (fp32, already in a register as exp(gc[15])), so K2 does no exp;
K2 kd/qd/kr stages are unpadded with the 16 B chunk XOR (row & 7) (padding would not fit the 3 x 15872 ring);
O^T is stored from the accumulators with 2-byte stores (no staging smem; a movmatrix transpose is a later option);
producer/consumer sync uses named barriers full = 1 + s, empty = 4 + s (empty arrives only when the producer will wait).
`kda::g9::launch(a, ps, VS, st)` runs conv16 -> gz16 -> K1 -> K2 -> out16. Not yet done: flag wiring, bench arm,
