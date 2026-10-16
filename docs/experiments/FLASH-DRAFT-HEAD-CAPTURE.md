# D2 draft-head capture, current offline source

Base 9da3ab1 (rebased from 19c431a). This diagnostic is not a small-head implementation.
The diagnostic patch was ported unchanged from the parked 9b563d8 tree.
ARM CUDA13.0 baseline/candidate engine and simulator builds pass.
No GPU OFF/ON output gate or actual hidden capture exists yet.

Opt-in G6_DRAFT_HEAD_DUMP=/path/stem records per-rank files after the
existing offline draft stream fence. Capture the actual normalized
FP32 D.d.xn, before the head's FP16 fragment conversion, plus every
first/chain call's row IDs, sample decisions, log probabilities and
sampling keys. Do not substitute MAIN teacher hidden states.

The OFF writer performs no CUDA call or allocation. This source property
does not prove compiled device-code equality or GPU output equivalence.
Paired local CUDA13.1/sm121a object builds on the prior 19c431a base
passed. Full SASS and resource dumps match byte-for-byte without metadata
normalization. This does not prove ARM linked-binary or GPU equivalence.
Paired local builds on 9da3ab1 also pass with byte-identical full SASS
and resource dumps. Gate complete OFF/ON outputs
before collecting training/evaluation captures.

CPU checks on this base:
- reader 18 tests PASS;
- C++ writer OFF null pointers and two-rank roundtrip PASS;
- git diff --check PASS.
Evidence lives in evidence/d2-current (19c431a) and evidence/d2-pf35
(9da3ab1) under the separate verification worktree.

ARM SASS/resources match after mapping one differing anonymous identity_pages
symbol; that unchanged source file has the same SHA on both sides. Raw
binaries and dumps are not byte-identical. Evidence: d2-pf35/arm.

Next: allocated TP2 OFF/ON output checks, then
capture on held-out prompts, fixed K3/K7, both ranks with complete footers.
Pair C currently belongs to MT, not D2. Do not start a GPU capture.

A frozen-hidden INT4-head replay tests only a proxy. It cannot establish
new chain acceptance because changed samples alter later hidden states.
Keep draft-only weights isolated; do not change the MAIN head.
