// One-GPU fixture for the production cluster projector used by decode and every prefill normalize variant.
// It covers all-off, all-on and alternating rows. Disabled rows must remain byte-identical.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "../../kernels/glue.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::printf("CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

__global__ void __cluster_dims__(glue::RN_CL, 1, 1) __launch_bounds__(glue::RN_T)
k_project(float* rows, const float* dir, const float* scale, float alpha)
{
    const int m = blockIdx.y;
    const int col0 = blockIdx.x * glue::RN_COLS + threadIdx.x * 8;
    float x[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) x[k] = rows[(size_t)m * HID + col0 + k];
    glue::cluster_project(x, glue::SteerArgs{dir, scale, 0.f, alpha}, m, col0);
    #pragma unroll
    for (int k = 0; k < 8; ++k) rows[(size_t)m * HID + col0 + k] = x[k];
}

// Exercise the production decode fast path that folds projection into the existing residual-add/RMSNorm launch.
// The launcher only selects this body for enabled rows; exact-zero rows retain the stock body tested above.
__global__ void __cluster_dims__(glue::RN_CL, 1, 1) __launch_bounds__(glue::RN_T)
k_project_add_norm(float* writer, float* residual, const float* dir, const float* scale,
                   const __nv_bfloat16* weight, __half* out, float alpha, float eps)
{
    const int m = blockIdx.y;
    const int col0 = blockIdx.x * glue::RN_COLS + threadIdx.x * 8;
    float y[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) y[k] = writer[(size_t)m * HID + col0 + k];
    float4* r = reinterpret_cast<float4*>(residual + (size_t)m * HID + col0);
    glue::cluster_project_add_rms_store(y, glue::SteerArgs{dir, scale, 0.f, alpha}, m, col0, r,
                                        weight, eps, out + (size_t)m * HID);
}

// An almost-collinear writer must normalize its small surviving component, not subtract large squared norms.
static bool cancellation_case()
{
    std::vector<float> writer(HID, .01f), residual(HID, 0.f), direction(HID, 0.f);
    std::vector<__half> out(HID);
    std::vector<unsigned short> weight(HID, 0x3f80u);
    for(int c=0;c<4096;++c) { writer[c]=1000.f; direction[c]=1.f/64.f; }
    float *dy, *dr, *dd, *ds; __half* dout; unsigned short* dw;
    CK(cudaMalloc(&dy,HID*sizeof(float))); CK(cudaMalloc(&dr,HID*sizeof(float)));
    CK(cudaMalloc(&dd,HID*sizeof(float))); CK(cudaMalloc(&ds,sizeof(float)));
    CK(cudaMalloc(&dout,HID*sizeof(__half))); CK(cudaMalloc(&dw,HID*sizeof(unsigned short)));
    const float one=1.f;
    CK(cudaMemcpy(dy,writer.data(),HID*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dr,residual.data(),HID*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dd,direction.data(),HID*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ds,&one,sizeof one,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dw,weight.data(),HID*sizeof(unsigned short),cudaMemcpyHostToDevice));
    k_project_add_norm<<<dim3(glue::RN_CL,1),glue::RN_T>>>(dy,dr,dd,ds,
        reinterpret_cast<const __nv_bfloat16*>(dw),dout,1.f,1e-5f);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(out.data(),dout,HID*sizeof(__half),cudaMemcpyDeviceToHost));
    const double tail=double(writer.back());
    const float expected=float(tail/std::sqrt((HID-4096)*tail*tail/HID+1e-5));
    const float got=__half2float(out.back());
    const bool ok=std::isfinite(got) && std::fabs(got-expected)<2e-3f;
    std::printf("%s cancellation normalized tail got=%g expected=%g\n",ok?"PASS":"FAIL",got,expected);
    cudaFree(dy);cudaFree(dr);cudaFree(dd);cudaFree(ds);cudaFree(dout);cudaFree(dw);
    return ok;
}

int main()
{
    if (!cancellation_case()) return 1;
    constexpr int M = 4;
    std::vector<float> h((size_t)M * HID), before, dir(HID), want;
    const float rv = 1.f / std::sqrt(float(HID));
    for (int c = 0; c < HID; ++c) {
        dir[c] = (c & 1) ? -rv : rv;
        for (int m = 0; m < M; ++m) h[(size_t)m * HID + c] = float((c % (31 + 4*m)) - (15 + m)) / float(29 + 3*m);
    }
    before = h;

    float *drow = nullptr, *ddir = nullptr, *dscale = nullptr;
    CK(cudaMalloc(&drow, h.size() * sizeof(float)));
    CK(cudaMalloc(&ddir, dir.size() * sizeof(float)));
    CK(cudaMalloc(&dscale, M * sizeof(float)));
    auto run = [&](const float (&scale)[M]) -> float {
        h = before; want = before;
        for (int m = 0; m < M; ++m) if (scale[m] != 0.f) {
            double dot = 0.0;
            for (int c = 0; c < HID; ++c) dot += double(want[(size_t)m * HID + c]) * double(dir[c]);
            const double kappa = double(scale[m]) * 1.25 * dot;
            for (int c = 0; c < HID; ++c)
                want[(size_t)m * HID + c] = float(double(want[(size_t)m * HID + c]) - kappa * double(dir[c]));
        }
        CK(cudaMemcpy(drow, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dscale, scale, sizeof scale, cudaMemcpyHostToDevice));
        k_project<<<dim3(glue::RN_CL, M), glue::RN_T>>>(drow, ddir, dscale, 1.25f);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(h.data(), drow, h.size() * sizeof(float), cudaMemcpyDeviceToHost));
        float max_err = 0.f;
        for (int m = 0; m < M; ++m) for (int c = 0; c < HID; ++c)
            if (!std::isfinite(h[(size_t)m * HID + c])) return INFINITY;
            else max_err = std::max(max_err, std::fabs(h[(size_t)m * HID + c] - want[(size_t)m * HID + c]));
        return max_err;
    };
    CK(cudaMemcpy(drow, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ddir, dir.data(), dir.size() * sizeof(float), cudaMemcpyHostToDevice));
    const float alternating[M] = {0.f, .75f, 0.f, 1.f};
    const float alt_err = run(alternating);
    const bool zero_exact = !std::memcmp(h.data(), before.data(), HID * sizeof(float)) &&
                            !std::memcmp(h.data() + 2 * HID, before.data() + 2 * HID, HID * sizeof(float));
    const float all_off[M] = {0.f, 0.f, 0.f, 0.f};
    const float off_err = run(all_off);
    const bool all_off_exact = !std::memcmp(h.data(), before.data(), h.size() * sizeof(float));
    const float all_on[M] = {.25f, .5f, .75f, 1.f};
    const float on_err = run(all_on);

    // Fused projection + residual add + RMSNorm. Use deterministic, non-collinear writer/residual rows and compare
    // both the FP32 residual and the stored FP16 normalized output with a direct host implementation.
    constexpr int FM = 2;
    constexpr float alpha = .875f, eps = 1e-5f;
    const float fused_scale[FM] = {.625f, 1.125f};
    std::vector<float> writer((size_t)FM * HID), residual((size_t)FM * HID), want_resid;
    std::vector<unsigned short> weight(HID, 0x3f80u);  // BF16 1.0
    std::vector<__half> got_out((size_t)FM * HID);
    std::vector<float> want_out((size_t)FM * HID);
    for (int m = 0; m < FM; ++m) for (int c = 0; c < HID; ++c) {
        writer[(size_t)m * HID + c] = float((c % (37 + m)) - (17 + 2*m)) / float(31 + m);
        residual[(size_t)m * HID + c] = float((c % (23 + 2*m)) - (9 + m)) / float(41 + m);
    }
    want_resid = residual;
    for (int m = 0; m < FM; ++m) {
        double dot = 0.0;
        for (int c = 0; c < HID; ++c) dot += double(writer[(size_t)m * HID + c]) * double(dir[c]);
        const double kappa = double(fused_scale[m]) * double(alpha) * dot;
        double ss = 0.0;
        for (int c = 0; c < HID; ++c) {
            const size_t i = (size_t)m * HID + c;
            want_resid[i] += writer[i] - float(kappa * double(dir[c]));
            ss += double(want_resid[i]) * double(want_resid[i]);
        }
        const double inv = 1.0 / std::sqrt(ss / double(HID) + double(eps));
        for (int c = 0; c < HID; ++c) {
            const size_t i = (size_t)m * HID + c;
            want_out[i] = float(double(want_resid[i]) * inv);
        }
    }
    float *dwriter = nullptr, *dresid = nullptr;
    unsigned short* dweight = nullptr;
    __half* dout = nullptr;
    CK(cudaMalloc(&dwriter, writer.size() * sizeof(float)));
    CK(cudaMalloc(&dresid, residual.size() * sizeof(float)));
    CK(cudaMalloc(&dweight, weight.size() * sizeof(unsigned short)));
    CK(cudaMalloc(&dout, got_out.size() * sizeof(__half)));
    CK(cudaMemcpy(dwriter, writer.data(), writer.size() * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dresid, residual.data(), residual.size() * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dweight, weight.data(), weight.size() * sizeof(unsigned short), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dscale, fused_scale, sizeof fused_scale, cudaMemcpyHostToDevice));
    k_project_add_norm<<<dim3(glue::RN_CL, FM), glue::RN_T>>>(
        dwriter, dresid, ddir, dscale, reinterpret_cast<const __nv_bfloat16*>(dweight), dout, alpha, eps);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(residual.data(), dresid, residual.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(got_out.data(), dout, got_out.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    float fused_resid_err = 0.f, fused_norm_err = 0.f;
    for (size_t i = 0; i < residual.size(); ++i) {
        if (!std::isfinite(residual[i]) || !std::isfinite(__half2float(got_out[i]))) {
            std::printf("steering_cuda_test: FAIL non-finite output\n"); return 1;
        }
        fused_resid_err = std::max(fused_resid_err, std::fabs(residual[i] - want_resid[i]));
        fused_norm_err = std::max(fused_norm_err, std::fabs(__half2float(got_out[i]) - want_out[i]));
    }
    cudaFree(dwriter); cudaFree(dresid); cudaFree(dweight); cudaFree(dout);
    cudaFree(drow); cudaFree(ddir); cudaFree(dscale);
    if (!zero_exact || !all_off_exact || !(alt_err < 2e-5f) || off_err != 0.f || !(on_err < 2e-5f)) {
        std::printf("steering_cuda_test: FAIL zero_exact=%d all_off_exact=%d errors=%g/%g/%g\n",
                    int(zero_exact), int(all_off_exact), alt_err, off_err, on_err); return 1;
    }
    if (!(fused_resid_err < 4e-5f) || !(fused_norm_err < 2e-3f)) {
        std::printf("steering_cuda_test: FAIL fused residual/norm errors=%g/%g\n", fused_resid_err, fused_norm_err);
        return 1;
    }
    std::printf("steering_cuda_test: PASS all-off/all-on/alternating, zero byte identity, projection and fused "
                "residual/RMSNorm max_err=%g/%g/%g\n", std::max(alt_err, on_err), fused_resid_err, fused_norm_err);
    return 0;
}
