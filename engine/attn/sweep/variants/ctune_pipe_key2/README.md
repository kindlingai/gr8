# ctune_pipe_key2

Key-block while-loop unroll 2. Private copy of OUR `ds41_attn_pipe.cuh:52-266` at 1ed4501c;
no external code and no arithmetic/data-layout/work-order changes.

Independent off comparison is provided by the shared sweep harness.
Window-only and floor>0 cases dispatch the original off kernels, as pipe does.
No pre-pass, allocation, or additional output. Dynamic smem is unchanged.

Local compile resource screen: regs 253,
STACK=0, spill stores=0, spill loads=0 (sm_120a).
Validation/evidence: legion report `reports/sweep-ctune.md` and raw sibling directory.
5090 timing is a hint only; this is not a served-engine change.
