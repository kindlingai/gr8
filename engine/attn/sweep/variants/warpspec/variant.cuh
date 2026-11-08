// Private warp-specialised form of our k_attn_cpa_t, baseline 4fb7f0bf.
// 64 producers fill one K tile; 192 consumers own eight logical baseline
// warps (roles 6 and 7 are second roles on consumer warps 0 and 1).
// Arithmetic for each role is unchanged. Named barriers: 1 producers,
// 2 tile-ready (all), 3 tile-consumed (all), 4 consumers only.
#pragma once
namespace warpspec_detail {
    using namespace ds41a;
    constexpr size_t SMEM = AT_SMEM_CPA + 16;
    __device__ __forceinline__ void producers() { asm volatile("bar.sync 1, 64;" ::: "memory"); }
    __device__ __forceinline__ void ready() { asm volatile("bar.sync 2, 256;" ::: "memory"); }
    __device__ __forceinline__ void consumed() { asm volatile("bar.sync 3, 256;" ::: "memory"); }
    __device__ __forceinline__ void consumers() { asm volatile("bar.sync 4, 192;" ::: "memory"); }

    template <bool FL, bool CQ>
    __global__ void __launch_bounds__(256, 1) k_warpspec(AttnP p) {
        extern __shared__ __align__(16) uint8_t smem[];
        bf16* Qs = (bf16*)smem;
        bf16* Ks = Qs + 16 * AT_QLD;
        float* Ss = (float*)(Ks + 64 * AT_KLD);
        bf16* Ps = (bf16*)(Ss + 16 * AT_SLD);
        const bf16** rowp = (const bf16**)(Ps + 16 * AT_PLD);
        float* sc_s = (float*)(rowp + 128);
        float* fin_s = sc_s + 16;
        int* live = (int*)(fin_s + 16); // two warp votes for each of two slot lists
        const int t = blockIdx.x, hg = blockIdx.y, tid = threadIdx.x;
        const int lane = tid & 31, physical_warp = tid >> 5;
        const int L = p.WL + p.TK, nblk = (L + 63) >> 6;
        if (tid < 64) {
            const int start = start_of(p.start, p.dstart);
            const int R = p.ring_rows ? p.ring_rows : WIN;
            const int tpos = pos_of(start, p.dstart, p.dsess, t);
            const bf16 *kvr = p.kvw, *ckr = p.ck;
            if (p.dsess) {
                const int k = p.dsess[t];
                kvr += (size_t)k * R * HD;
                if (ckr) ckr = cq_slab<CQ>(ckr, p.ck_ss, k);
            }
            auto build = [&](int blk, int stage) {
                const int i = blk * 64 + tid;
                const bf16* rp = nullptr;
                if (i < p.WL) {
                    if (!p.decode) {
                        const int id = max(t - (WIN - 1), 0) + i;
                        if (id <= t) rp = p.kvw + (size_t)id * HD;
                    } else {
                        const int pos = tpos - (WIN - 1) + i;
                        if (pos >= (FL ? p.floor : 0)) rp = kvr + (size_t)(pos % R) * HD;
                    }
                } else if (i < L) {
                    const int j = p.topk[(size_t)t * p.ldtk + (i - p.WL)];
                    if (j >= 0) rp = cq_slot<CQ>(ckr, j);
                }
                rowp[stage * 64 + tid] = rp;
                const int has = __any_sync(0xffffffffu, rp != nullptr);
                if (lane == 0) live[stage * 2 + physical_warp] = has;
                producers();
            };
            const uint4* src = (const uint4*)(p.q + ((size_t)t * p.heads + (size_t)hg * 16) * HD);
            for (int c = tid; c < 16 * HD / 8; c += 64)
            cp_async16(Qs + (c >> 6) * AT_QLD + (c & 63) * 8, src + c, 16);
            cp_async_commit();
            cp_async_wait0();
            if (nblk) build(0, 0);
            for (int blk = 0; blk < nblk; ++blk) {
                const int stage = blk & 1;
                const bf16** cl = rowp + stage * 64;
                if (live[stage * 2] | live[stage * 2 + 1]) {
                    // Two producer warps cover all 64 rows, no consumer loads K.
                    for (int c = tid; c < 64 * HD / 8; c += 64) {
                        const bf16* rp = cl[c >> 6];
                        if (CQ && cq_tag(rp)) continue;
                        cp_async16(Ks + (c >> 6) * AT_KLD + (c & 63) * 8,
                        rp ? (const void*)((const uint4*)rp + (c & 63)) : (const void*)p.q, rp ? 16 : 0);
                    }
                    cp_async_commit();
                    if (CQ) {
                        for (int r = physical_warp; r < 64; r += 2) {
                            const bf16* rp = cl[r];
                            if (!cq_tag(rp)) continue;
                            const uint8_t* packed = cq_row(rp);
                            const uint2 cd = __ldg((const uint2*)packed + lane);
                            const uint8_t sb = __ldg(packed + CQ_CODES + lane);
                            const __nv_bfloat162 s2 = __float2bfloat162_rn(f_e4m3(sb));
                            uint32_t o[8];
                            cq_dq4(cd.x, s2, o[0], o[1]);
                            cq_dq4(cd.x >> 16, s2, o[2], o[3]);
                            cq_dq4(cd.y, s2, o[4], o[5]);
                            cq_dq4(cd.y >> 16, s2, o[6], o[7]);
                            uint4* dst = (uint4*)(Ks + r * AT_KLD + lane * 16);
                            dst[0] = make_uint4(o[0], o[1], o[2], o[3]);
                            dst[1] = make_uint4(o[4], o[5], o[6], o[7]);
                        }
                    }
                    cp_async_wait0();
                }
                ready();
                // Independent producer work overlaps the consumers' QK/softmax/PV.
                if (blk + 1 < nblk) {
                    const int next = stage ^ 1;
                    build(blk + 1, next);
                    const bf16* rp = rowp[next * 64 + tid];
                    if (CQ && cq_tag(rp)) {
                        prefetch_l2(cq_row(rp));
                        prefetch_l2(cq_row(rp) + 128);
                        prefetch_l2(cq_row(rp) + CQ_ROW - 1);
                    } else if (rp) {
                        for (int off = 0; off < HD * 2; off += 128)
                        prefetch_l2((const char*)rp + off);
                    }
                }
                consumed();
            }
            return;
        }
        const int cw = physical_warp - 2;
        float m_run[2] = {-1e30f, -1e30f}, sum_run[2] = {0.0f, 0.0f};
        float acc[2][8][4];
#pragma unroll
        for (int role = 0; role < 2; ++role)
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[role][i][j] = 0.0f;
        for (int blk = 0; blk < nblk; ++blk) {
            const int stage = blk & 1;
            const bf16** cl = rowp + stage * 64;
            ready();
            if (live[stage * 2] | live[stage * 2 + 1]) {
#pragma unroll
                for (int role = 0; role < 2; ++role) {
                    const int w = cw + role * 6;
                    if (w < 8) {
                        const int sh = w * 2 + (lane >> 4), sl = lane & 15;
                        {   // S[16 heads][8 slots of this warp] = Q . K^T over 512
                            float cc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                            const bf16* krow = Ks + (w * 8 + (lane >> 2)) * AT_KLD + 2 * (lane & 3);
#pragma unroll 4
                            for (int ks = 0; ks < HD / 16; ks++)
                            {
                                uint32_t a[4];
                                ldsm_x4(a, Qs + (lane & 15) * AT_QLD + ks * 16 + (lane >> 4) * 8);
                                const uint32_t b0 = *(const uint32_t*)(krow + ks * 16);
                                const uint32_t b1 = *(const uint32_t*)(krow + ks * 16 + 8);
                                mma16816(cc, a, b0, b1);
                            }
                            const int r0 = lane >> 2, col = w * 8 + 2 * (lane & 3);
                            const bool v0 = cl[col] != nullptr, v1 = cl[col + 1] != nullptr;
                            Ss[r0 * AT_SLD + col] = v0 ? cc[0] * p.scale : -INFINITY;
                            Ss[r0 * AT_SLD + col + 1] = v1 ? cc[1] * p.scale : -INFINITY;
                            Ss[(r0 + 8) * AT_SLD + col] = v0 ? cc[2] * p.scale : -INFINITY;
                            Ss[(r0 + 8) * AT_SLD + col + 1] = v1 ? cc[3] * p.scale : -INFINITY;
                        }
                    }
                }
                consumers();
#pragma unroll
                for (int role = 0; role < 2; ++role) {
                    const int w = cw + role * 6;
                    if (w < 8) {
                        const int sh = w * 2 + (lane >> 4), sl = lane & 15;
                        {   // online softmax, head sh, 16 threads x 4 slots
                            const float4 s4 = *(const float4*)(Ss + sh * AT_SLD + sl * 4);
                            float mb = fmaxf(fmaxf(s4.x, s4.y), fmaxf(s4.z, s4.w));
                            for (int o = 8; o; o >>= 1) mb = fmaxf(mb, __shfl_xor_sync(0xFFFFFFFFu, mb, o));
                            const float m_new = fmaxf(m_run[role], mb);
                            const float scl = expf(m_run[role] - m_new);
                            const float p0 = expf(s4.x - m_new), p1 = expf(s4.y - m_new), p2 = expf(s4.z - m_new), p3 = expf(s4.w - m_new);
                            float ps = (p0 + p1) + (p2 + p3);
                            for (int o = 8; o; o >>= 1) ps += __shfl_xor_sync(0xFFFFFFFFu, ps, o);
                            sum_run[role] = sum_run[role] * scl + ps;
                            m_run[role] = m_new;
                            bf16* pr = Ps + sh * AT_PLD + sl * 4;
                            pr[0] = __float2bfloat16_rn(p0);
                            pr[1] = __float2bfloat16_rn(p1);
                            pr[2] = __float2bfloat16_rn(p2);
                            pr[3] = __float2bfloat16_rn(p3);
                            if (sl == 0) sc_s[sh] = scl;
                        }
                    }
                }
                consumers();
#pragma unroll
                for (int role = 0; role < 2; ++role) {
                    const int w = cw + role * 6;
                    if (w < 8) {
                        const int sh = w * 2 + (lane >> 4), sl = lane & 15;
                        {   // acc[16 heads][64 dims of this warp] = acc * scale + P . V   (V = the same K rows)
                            const float sc0 = sc_s[lane >> 2], sc1 = sc_s[(lane >> 2) + 8];
#pragma unroll
                            for (int nt = 0; nt < 8; nt++)
                            {
                                acc[role][nt][0] *= sc0;
                                acc[role][nt][1] *= sc0;
                                acc[role][nt][2] *= sc1;
                                acc[role][nt][3] *= sc1;
                            }
#pragma unroll
                            for (int kj = 0; kj < 4; kj++)
                            {
                                uint32_t a[4];
                                ldsm_x4(a, Ps + (lane & 15) * AT_PLD + kj * 16 + (lane >> 4) * 8);
#pragma unroll
                                for (int np = 0; np < 4; np++)
                                {
                                    uint32_t b[4];
                                    const int mi = lane >> 3, r = lane & 7;
                                    ldsm_x4_t(b, Ks + (kj * 16 + (mi & 1) * 8 + r) * AT_KLD + w * 64 + (np * 2 + (mi >> 1)) * 8);
                                    mma16816(acc[role][np * 2], a, b[0], b[1]);
                                    mma16816(acc[role][np * 2 + 1], a, b[2], b[3]);
                                }
                            }
                        }
                    }
                }
            }
            consumed();
        }
#pragma unroll
        for (int role = 0; role < 2; ++role) {
            const int w = cw + role * 6;
            if (w < 8) {
                const int sh = w * 2 + (lane >> 4), sl = lane & 15;
                if (sl == 0) fin_s[sh] = sum_run[role] + expf(p.sink[hg * 16 + sh] - m_run[role]);
            }
        }
        consumers();
#pragma unroll
        for (int role = 0; role < 2; ++role) {
            const int w = cw + role * 6;
            if (w < 8) {
                const int sh = w * 2 + (lane >> 4), sl = lane & 15;
                {
                    const int r0 = lane >> 2;
                    const float d0 = fin_s[r0], d1 = fin_s[r0 + 8];
                    bf16* o0 = p.out + ((size_t)t * p.heads + hg * 16 + r0) * HD;
                    bf16* o1 = p.out + ((size_t)t * p.heads + hg * 16 + r0 + 8) * HD;
#pragma unroll
                    for (int nt = 0; nt < 8; nt++)
                    {
                        const int d = w * 64 + nt * 8 + 2 * (lane & 3);
                        *(__nv_bfloat162*)(o0 + d) = __floats2bfloat162_rn(acc[role][nt][0] / d0, acc[role][nt][1] / d0);
                        *(__nv_bfloat162*)(o1 + d) = __floats2bfloat162_rn(acc[role][nt][2] / d1, acc[role][nt][3] / d1);
                    }
                }
            }
        }
    }
} // namespace warpspec_detail
namespace sweep_variant {
    inline const void* kernel(const ds41a::AttnP& p) {
        if (p.floor) return (const void*)warpspec_detail::k_warpspec<true,true>;
        return p.TK ? (const void*)warpspec_detail::k_warpspec<false,true> : (const void*)warpspec_detail::k_warpspec<false,false>;
    }
    inline const char* ptxas_key(const ds41a::AttnP& p) {
        return p.floor ? "k_warpspecILb1ELb1EE" : p.TK ? "k_warpspecILb0ELb1EE" : "k_warpspecILb0ELb0EE";
    }
    inline size_t dynamic_smem(const ds41a::AttnP&) { return warpspec_detail::SMEM; }
    inline cudaError_t init(const ds41a::AttnP& p) {
        return cudaFuncSetAttribute(kernel(p), cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_smem(p));
    }
    inline void launch(const ds41a::AttnP& p, cudaStream_t st) {
        dim3 g(p.n, p.heads / 16);
        if (p.floor) warpspec_detail::k_warpspec<true,true><<<g,256,warpspec_detail::SMEM,st>>>(p);
        else if (p.TK) warpspec_detail::k_warpspec<false,true><<<g,256,warpspec_detail::SMEM,st>>>(p);
        else warpspec_detail::k_warpspec<false,false><<<g,256,warpspec_detail::SMEM,st>>>(p);
    }
}
