// Compile-only gate for the DSpark device code (B12): instantiates every kernel and launcher of engine/attn/ds41_attn_dsp.cuh and
// engine/runtime/ds41_dspark.cuh so a plain nvcc run proves they build for sm_121a. It is not run (no main); the on-box device
// check comes with the booked box.
//   /usr/local/cuda/bin/nvcc -std=c++17 -gencode=arch=compute_121a,code=sm_121a -DG6_MODEL_DS41=1 -I. -Iengine -Xcompiler -Wall
//       -c engine/tests/ds41_dspark_compile.cu -o /tmp/ds41_dspark_compile.o
#include "attn/ds41_attn_dsp.cuh"
#include "runtime/ds41_dspark.cuh"

void ds41_dspark_compile_all(const ds41a::Weights& W, const ds41dsp::Ring& R, const ds41a::Scratch& S, const ds41dsp::bf16* y, const int* pos, const int* qpos,
                             ds41dsp::bf16* out, const float* streams, ds41dsp::bf16* tap, const ds41dsp::MainW& MW, const ds41dsp::MainS& MS,
                             const int* i1, const int* i2, const int* i3, int* o1, int* o2, int* o3, int* o4, int* o5, const ds41dsp::bf16* membed,
                             const ds41dsp::bf16* mhead, float* hlog, float* conf, cudaStream_t st)
{
    ds41dsp::attn_proj_blk(W, R, S, y, 40, 5, pos, 16, st);
    ds41dsp::attn_core_blk(W, R, S, 40, 5, qpos, 16, st);
    ds41dsp::attn_out_blk(W, S, 40, pos, 16, out, st);
    ds41dsp::ds_tap(streams, tap, 0, 6, st);
    ds41dsp::ds_main(MW, MS, tap, 6, 6, pos, st);
    ds41dsp::ds_draft_in(i1, i2, i3, 6, 1, o1, o2, o3, o4, o5, st);
    ds41dsp::markov_add(membed, mhead, o2, o4, 1, 1, hlog, 32384, 32384, st);
    ds41dsp::conf_head(tap, membed, mhead, o2, o4, 1, conf, o5, st);
    ds41dsp::ds_verify_in(o2, o5, o1, 5, 1, o3, o4, st);
}
// the per-Model bundle is plain data: it must stay constructible without rank.cuh (B9 includes the header at global scope)
ds41dsp::Dsp ds41_dspark_compile_dsp() { ds41dsp::Dsp d; d.nsess = 1; return d; }
