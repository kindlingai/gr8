// The glue between the GPU test (kda_g9_test.cu) and the CPU ON-emulation (kda_g9_cpu.h): the multi-head slicing of the engine layouts into
// the emulation's one-head inputs. Shared by the GPU tests (kda_g9_test.cu, kda_g9_det.cu).
//   Cm   [M][3][dl]      fp32 conv+silu output (Args::C layout of the OFF path)
//   pG, pZ  [m * dl + h * DK + d]  bf16 G16 / Z16 planes (row stride dl)
//   P    [M][ld]         projections; b_proj of head h at column bcol + h
//   hS   whole state, head h at a.s_off(0, h) (only a.s_off is used from `a`)
// Output: eS (same layout as hS), eH [m * dl + h * DK + d] = fp16 OH of the emulation as float.
#pragma once
#include <algorithm>
#include <vector>
#include "kda_g9_cpu.h"
#include "../math.h"

namespace g9glue
{
inline void emu_on_heads(const kda::Args& a, int M, int nh, const std::vector<float>& Cm, const std::vector<uint16_t>& pG,
                         const std::vector<uint16_t>& pZ, const float* P, int ld, int bcol, const std::vector<float>& hAlog,
                         const std::vector<float>& hdt, const std::vector<uint16_t>& hon, const std::vector<float>& hS,
                         std::vector<float>& eS, std::vector<float>& eH)
{
    using kda::DK; using kda::DV;
    const int dl = nh * DK;
    std::vector<float> onf(DK);
    for (int d = 0; d < DK; ++d) onf[d] = kda::bf2f(hon[d]);
    eS.assign(hS.size(), 0.f);
    eH.assign(static_cast<size_t>(M) * dl, 0.f);
    for (int h = 0; h < nh; ++h)
    {
        std::vector<float> Ch(static_cast<size_t>(M) * 3 * DK), G16h(static_cast<size_t>(M) * DK), Z16h(G16h.size()), br(M), Sh(DK * DV), o16(G16h.size()), yy(G16h.size());
        std::vector<uint16_t> ohe(G16h.size());
        for (int m = 0; m < M; ++m)
        {
            for (int s3 = 0; s3 < 3; ++s3)
                for (int d = 0; d < DK; ++d) Ch[(static_cast<size_t>(m) * 3 + s3) * DK + d] = Cm[static_cast<size_t>(m) * 3 * dl + s3 * dl + h * DK + d];
            for (int d = 0; d < DK; ++d)
            {
                G16h[static_cast<size_t>(m) * DK + d] = kda::bf2f(pG[static_cast<size_t>(m) * dl + h * DK + d]);
                Z16h[static_cast<size_t>(m) * DK + d] = kda::bf2f(pZ[static_cast<size_t>(m) * dl + h * DK + d]);
            }
            br[m] = P[static_cast<size_t>(m) * ld + bcol + a.h0 + h];
        }
        std::copy(hS.begin() + a.s_off(0, h), hS.begin() + a.s_off(0, h) + DK * DV, Sh.begin());
        g9cpu::OnIn in; in.M = M; in.C = Ch.data(); in.G16 = G16h.data(); in.Z16 = Z16h.data(); in.braw = br.data(); in.A_log = hAlog[h];
        in.dt_bias = &hdt[h * DK]; in.onorm = onf.data();
        g9cpu::OnOut out; out.O16 = o16.data(); out.y = yy.data(); out.OH = ohe.data();
        g9cpu::on_call(in, Sh.data(), out);
        std::copy(Sh.begin(), Sh.end(), eS.begin() + a.s_off(0, h));
        for (int m = 0; m < M; ++m)
            for (int d = 0; d < DK; ++d) eH[static_cast<size_t>(m) * dl + h * DK + d] = g9cpu::h2f(ohe[static_cast<size_t>(m) * DK + d]);
    }
}
}
