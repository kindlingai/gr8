# Pair-owner QO prefill at TP4 (`G6_PF_OWN=4`)

Design by adapt (branch `tp4-engine-fusedrx`, `engine-tp4/DESIGN-QO-PAIR.md`), ported to `main` in #39. This page
describes what `main` implements. Code: `engine/runtime/qp.cuh`, `engine/runtime/qo_weights.cuh` (pair weights and the
rolling pack), `engine/runtime/qp_wire.h` (the RDMA protocol), `tools/convert/pairpack.py` (resident pair packs).

## Why

`G6_PF_OWN=3` (QO attention) is g6's biggest prefill lever: the owner of a row runs the whole attention block for all
64 heads of that row. It needs the full-head QO pack (13.1 GB) on every rank. At TP4 a rank pack is 90.3 GB, and the
two do not fit together on a GB10 (#2). So TP4 ran `G6_PF_OWN=1`.

## The mode

TP4 only (`G6_NR=4`). Needs `G6_PF_RS=1` and `--pfv2`, like OWN=3.

- The four ranks form two head pairs, {0,1} and {2,3}. In layer l, rank m holds head group (m - l) mod 4 (16 heads).
  Both members of a pair together hold 32 heads.
- Rank r has a counterpart c = r ^ 2 (same position in the other pair). Rank r runs the attention block (q_b, absorb,
  QO core, UV, o_proj) for its pair's 32 heads on W = the owner rows of c, then its own owner rows, in one QO call.
- The o_proj result is a BF16 partial over 32 heads. Rank r sends the partial of c's rows to c only (collective 0 as a
  point-to-point write, `Fabric::Rs::p2p`) and receives c's partial of its own rows.
- The owner adds the two partials in a fixed order per layer (the pair that holds head group 0 of the layer first), adds
  the residual, norms the row and publishes collective 1, as OWN=3 does.
- Everything else is OWN=2: the owner computes the next layer's projections and the index top-2048 of its rows, and
  collectives 3 and 4 gather those to every rank. So rank r already has c's rows' q_a and index IDs.
- `G6_PF_QP_LAYERS` (default all of 0-77) picks the layers that run pair-owner attention; the others run OWN=2. The MTP
  layer keeps its own path.

Per rank, a QO call covers 2 x (M / 4) rows x 32 heads, the same core work as OWN=3's M / 4 rows x 64 heads.

## Where the pair weights come from

A pair's weights per layer: q_b [8192][2048] MXFP8, o_proj [6144][8192] MXFP8, UK / UV for 32 heads. Byte-identical on
both pair members (the lower head group first).

- **Rolling pack (`G6_PF_QP_ROLL=S`, the TP4 default in `launch_tp4.sh`, S = 2).** S slots roll through the layers
  during a prefill. A slot for layer l is filled from the rank's own head slice (already in its rank pack) and the
  partner's slice, which the partner writes over RDMA into a small pinned ring (`qp_wire.h`; one bulk fabric on
  port + 56, 84 MB). Then the slot's QO state converts UK / UV to FP16. Builds run on their own stream, one layer ahead,
  after both prefill lanes are done with the slot. Cost per rank at the launcher defaults (`G6_PF_QP_CTX` = 393,216,
  4,096-row lanes): about 1.3-1.4 GB of device memory plus 84 MB of pinned host memory for the ring. The device part is
  the shared QO workspace (key table 453 MB, RoPE table 101 MB, q / ctx buffers 281 MB), two slots (168 MB), two FP16
  states (29 MB) and the per-lane scratch (about 128 MB x 2). Wire traffic: 40 MiB per layer per pass to the partner.
  `G6_PF_QP_CTX` sets the QO key-table size; a request whose prompt + generation is longer runs OWN=2 instead. About
  1.4 KB per position scales with it, so it is the memory lever if a node runs short.
- **Early send (`G6_PF_QP_EARLY=1`, the TP4 launcher default).** The rank quantizes and projects the counterpart's rows
  first and sends that partial before it runs o_proj on its own rows, so the send overlaps that GEMM. It measured
  4-5% faster prefill than 0 at 8K and 32K; 2 (also splitting the QO call in two) was slower than 1.
- **Resident pair pack (`--qp-manifest` / `--qp-pack`).** `tools/convert/pairpack.py` builds `qp.rank<r>.pack` from the
  four rank packs (5.97 GB per rank). This does not fit next to a TP4 rank pack at the 393,216-token pool, so it is
  only for smaller pools or tests.
- Dense trellis q_b / o_proj (the exlr8 release form, #116). The q_b half is the member's FP8 tile copy
  (`engine/runtime/dtr_find.h`, exact: the codebook is E4M3) with every scale 127, so the slots and the wire carry it
  as they carry MXFP8. The o_proj half travels and sits in the slot as the member's raw trellis tiles (12.6 MB, inside
  the MXFP8 half's code range), and the pair's o_proj GEMM decodes them (`k_gemm_tr`). The half image adds the member's q_b svh and o_proj suh slices (8 KiB each). The shared
  q_b suh rotates the pair's q_b input and the pair's svh its output. The pair's suh rotates the o_proj input and the
  shared svh its output (`Xq::suh`, `Gemm::osvh`). A layer's form is part of the wire's ready tag, so a
  pair that disagrees fails closed. The resident pair pack has no trellis form.

## Numerics

Projections, caches and index IDs are byte-equal to OWN=2. From the attention output on, results differ as OWN=3's do
(the QO core, then o_proj as two BF16 partials of K = 8192 summed in FP32 on the owner). Outputs on the five fixed TP4
prompts stay coherent and diverge from OWN=1 after a few dozen tokens, as two attention paths do.

## Builds

All OWN=4 kernels and host paths compile only with `G6_NR=4`. A TP6 build has no OWN=4 code, and `G6_PF_OWN=4` exits
with a message there.
