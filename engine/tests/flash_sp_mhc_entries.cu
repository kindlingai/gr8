// Compile-only TU: instantiates every mhc.cuh entry point the Flash SP path adds (k_mhc_*_sp, prepare_range) and
// the OFF wrappers that now share row bodies with them (including their default-argument call shapes), so one
// nvcc -c catches template-argument, signature or launch-configuration breakage on any CUDA box.
// Build: nvcc -std=c++17 -arch=sm_90 -c engine/tests/flash_sp_mhc_entries.cu -o /tmp/flash_sp_mhc_entries.o
// (dev box: no GPU run, compile only.)
#include "../mhc/mhc.cuh"

void instantiate_sp_and_off(flash_mhc::Buffers b, flash_mhc::Weights w, flash_mhc::Fp32Block blk,
                            const uint4* fnp, const __nv_bfloat16* norm, cudaStream_t st)
{
    __half* out = nullptr;
    float* normalized = nullptr;
    float* mean = nullptr;
    unsigned* f16 = nullptr;
    unsigned* bf16 = nullptr;
    // OFF entries: exact pre-SP call shapes, defaults included
    flash_mhc::k_mhc_out_inv<<<4, flash_mhc::THREADS, 0, st>>>(b, blk, 4096, 1e-6f);
    flash_mhc::k_mhc_out_inv_prepare_pk<flash_mhc::Fp32Block, 2><<<2, flash_mhc::THREADS, 0, st>>>(b, blk, w, fnp, 4, 1e-6f, out, normalized);
    flash_mhc::k_mhc_out_inv_prepare_pk<flash_mhc::Fp32Block, 2><<<2, flash_mhc::THREADS, 0, st>>>(b, blk, w, fnp, 4, 1e-6f, out);   // default normalized
    flash_mhc::k_mhc_out_inv_prepare_rr<flash_mhc::Fp32Block, 2, 1><<<2, flash_mhc::THREADS, 0, st>>>(b, blk, w, fnp, 4, 1e-6f, out, normalized);   // default f16/bf16
    flash_mhc::k_mhc_out_inv_prepare_rr<flash_mhc::Fp32Block, 2, 1><<<2, flash_mhc::THREADS, 0, st>>>(b, blk, w, fnp, 4, 1e-6f, out, normalized, f16, bf16);
    flash_mhc::k_mhc_out_inv_final<<<4, flash_mhc::THREADS, 0, st>>>(b, blk, norm, 4096, 1e-6f, out, normalized, mean);   // default f16/bf16
    flash_mhc::k_mhc_out_inv_final<<<4, flash_mhc::THREADS, 0, st>>>(b, blk, norm, 4096, 1e-6f, out, normalized, mean, f16, bf16);
    flash_mhc::prepare(b, w, 4, 4096, 1e-6f, out, st, normalized, f16, bf16);
    // SP owner-range entries: one call shape each
    flash_mhc::k_mhc_out_inv_sp<<<4, flash_mhc::THREADS, 0, st>>>(b, blk, 1, 4096, 1e-6f);
    flash_mhc::k_mhc_out_inv_prepare_pk_sp<flash_mhc::Fp32Block, 2><<<2, flash_mhc::THREADS, 0, st>>>(b, blk, w, fnp, 1, 3, 1e-6f, out, normalized);
    flash_mhc::k_mhc_out_inv_prepare_rr_sp<flash_mhc::Fp32Block, 2, 1><<<2, flash_mhc::THREADS, 0, st>>>(b, blk, w, fnp, 1, 3, 4, 1e-6f, out, normalized, f16, bf16);
    flash_mhc::k_mhc_out_inv_final_sp<<<4, flash_mhc::THREADS, 0, st>>>(b, blk, norm, 1, 4096, 1e-6f, out, normalized, mean);
    flash_mhc::prepare_range(b, w, 1, 3, 4, 4096, 1e-6f, out, st, normalized);
    flash_mhc::prepare_range(b, w, 1, 1, 4, 4096, 1e-6f, out, st);   // zero-row owner: default normalized, no launch
}
