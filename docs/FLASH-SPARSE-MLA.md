# Sparse MLA prefill: opt-in high/low probability path

For up to 32 local heads, one 256-thread CTA processes one query. Selected latent tiles serve QK and PV without a global probability buffer. FP16 high and low probability components feed separate FP32 accumulators. The low component is scaled by 2048 before FP16 conversion and divided by 2048 in the final combination. Online normalization and output layout remain unchanged.

Only `G6_PF_SPARSE_MLA=1` selects this path. An unset variable or explicit `0` selects the prior QK, softmax and PV kernels. More than 32 local heads retain that prior path. Decode is unchanged. Dynamic shared memory is 77056 bytes. The pool-owner receive hook runs after query absorption and before either sparse-attention consumer.

## Validation status and contract

This is a numerical change, not an exact replacement. Default-ON is not approved. An exit 0 from a report wrapper does not prove numerical acceptance.

The test mode `G6_TEST_SPARSE_MLA=1` compares explicit OFF against explicit ON. Check the final binary's unset mode against explicit OFF and the baseline on both ranks. Preserve actual child environment receipts and all logit, residual, KDA-state, and step/hash outputs. The earlier default-ON test binary does not establish the final opt-in binary's default behavior.

The current 8-layer path requires independent OFF/ON reference reports and the four numerical gates. Report linear and quadratic projections separately from measurements. The current 45-layer P1 capture needs its own independent reference: KL(ref || ON) <= 3.13e-3, top-1 >= 254/256, and greedy tokens equal OFF. Historical hi/lo captures do not substitute for that current capture. Report maximum KL and the reference floor without hiding raw failed gates.

Raw attention, BF16/W8d block, host, full-topology, identity, and applicable runtime regression gates remain required before release. No result is implied by this document.

Unset or explicit `0` is the rollback. This change does not alter pool selection, communications, KV precision, or decode kernels. The default-ON decision remains separate.

## Experimental DSA512 opt-in

Set both `G6_PF_SPARSE_MLA=1` and `G6_PF_SPARSE_MLA_512=1` to select
the 512-thread variant. The second flag alone has no effect. Both remain
default OFF. Unset or zero for the second flag retains the 256-thread P1
kernel. More than 32 local heads retain the original split QK/softmax/PV
path regardless of the second flag. Decode and the force-P1 small-row
fallback are unchanged. This is an experimental engine environment flag,
not a new supported launcher/configuration option.

The variant redistributes the same PV output fragments from eight to four
per warp, with sixteen warps instead of eight. The QK warp mapping,
per-output reduction order, high/low probability arithmetic and shared
memory layout remain the same. The new kernel body matches the isolated
microbenchmark source byte-for-byte.

The isolated synthetic checks at 32,768 and 300,000 context matched every
FP16 output byte against P1 before and after timing. This establishes
exactness only relative to P1 on those fixtures. **P1 remains a precision
item, not an exact replacement for the OFF path.** No full-model
equivalence follows from the microbenchmark.

Pending release gates:
1. Eight-layer byte equality, DSA512 ON versus P1 ON, at 72 and 9,224 tokens.
2. Clean 45-layer 32K timing with five paired runs under the flip rule.
   flips P1. DSA512 must never select the OFF path.

CPU checks (no CUDA execution):
`python3 -m unittest discover -s engine/attn/tests -p test_flash_dsa512.py -v`.
These check the actual host dispatch expressions across 216 flag/head
combinations, the exact tested kernel body, output mapping and full
reconstruction of the pinned P1 source after removing this additive change.
They do not satisfy either pending GPU gate.

## V2 dispatch (#10, `sparse_mla_pf_v2`)

V2 now runs by default only when `G6_PF_SPARSE_MLA=1`, `G6_PF_SPARSE_MLA_512=1`, and the local head count is at most 32. `G6_PF_SPARSE_MLA_V2=0` restores `_512`. Explicit `_V2=1` keeps the prior opt-in behavior when `_512` is off, but still needs sparse MLA and at most 32 local heads. The OFF path is unchanged.
V2 is a restructure of `sparse_mla_pf_512` with byte-identical output: same QK / PV HMMA k-chains, same softmax and epilogue operations. It keeps P_lo.
It adds an index-to-row table in shared memory (84,432 B total), a register prefetch of the next KV tile, raw `mma.sync` with `ldmatrix`, and 4 barriers per live tile.
Gate: `engine/attn/tests/flash_sparse_mla_v2_test.cu` (memcmp against `_512`, BF16 and `-DG6_FLASH_KV_FP8=1` builds), then the 8L byte gate and the 45L timing gate.
