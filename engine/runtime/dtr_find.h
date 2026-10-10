// Dense trellis matrices (#116, kernels/dtr.cuh) on the prefill path. The prefill binder (bindp::layer_tensors) and the
// pair-owner weights (qo_weights.cuh qp_half) read q_b, o_proj and the dense MLP as MXFP8 codes + scales under
// L%03d.NAME.mx.codes / .mx.scales. For a matrix in the release's trellis form, dtr_overlay serves its FP8 tile copy
// (dtr::make_fp8: exact, still rotated, every scale 127) under those names. The binders find the matrix's suh / svh
// themselves and wrap the GEMMs in the rotations.
#pragma once
#include <map>
#include <memory>
#include "rank.cuh"

namespace g6rt
{

// base with the FP8 copies of every trellis matrix it has (layers 0..78) laid over it. The copies are made here, once:
// call it once per rank and keep the result. prefill = false: no copies (decode reads the trellis tiles).
inline Find dtr_overlay(const Find& base, bool prefill, cudaStream_t st, int rank)
{
    if (!prefill) return base;
    auto over = std::make_shared<std::map<std::string, TensorRef>>();
    size_t bytes = 0;
    int n = 0;
    char b[64];
    for (int l = 0; l <= 78; ++l)
        for (int i = 0; i < dtr::NMAT; ++i)
        {
            snprintf(b, sizeof b, "L%03d.%s.", l, dtr::MAT_NAMES[i]);
            const std::string pre = b;
            const TensorRef t = base(pre + "trellis");
            if (!t.p) continue;
            dtr::Mat m;
            if (const char* e = dtr::make_mat(t, base(pre + "suh"), base(pre + "svh"), dtr::MAT_SIDE[i], m)) die(pre + "trellis: " + e);
            dtr::Fp8 F;
            if (const char* e = dtr::make_fp8(m, st, F)) die(pre + "trellis: " + e);
            TensorRef c;
            c.p = F.codes; c.nd = 2; c.shape[0] = F.N; c.shape[1] = F.K;
            TensorRef s = c;
            s.p = F.scales; s.shape[1] = F.K / 32;
            (*over)[pre + "mx.codes"] = c;
            (*over)[pre + "mx.scales"] = s;
            bytes += (size_t)F.N * F.K / 32 * 33;
            ++n;
        }
    if (!n) return base;
    fprintf(stderr, "# rank %d: dense trellis FP8 copies for prefill: %d matrices, %.3f GB\n", rank, n, bytes / 1e9);
    return [base, over](const std::string& name) {
        const auto it = over->find(name);
        return it != over->end() ? it->second : base(name);
    };
}

}   // namespace g6rt
