# ctune_off_key4

Key-block while-loop unroll 4. Private copy of OUR `ds41_attn.cuh:941-1131` at 1ed4501c;
no external code and no arithmetic/data-layout/work-order changes.

Independent off comparison is provided by the shared sweep harness.
All three off specializations are tuned (window, CQ and floor).
No pre-pass, allocation, or additional output. Dynamic smem is unchanged.

Local compile resource screen: regs 145,173,173,
STACK=0, spill stores=0, spill loads=0 (sm_120a).
Validation/evidence: legion report `reports/sweep-ctune.md` and raw sibling directory.
5090 timing is a hint only; this is not a served-engine change.
