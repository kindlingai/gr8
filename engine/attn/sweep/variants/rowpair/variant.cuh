// Two adjacent query rows share one widened K tile. Each row retains the
// baseline's slot positions, 64-key blocks, MMA order and softmax reduction.
// Original implementation derived only from our ds41_attn.cuh CPA core.
#pragma once
namespace rowpair_detail {
using namespace ds41a;

// The second row only replaces K slots whose pointers differ. Null entries
// are compared too: a matching null keeps the zero filled by the first row.
template <bool REUSE>
__device__ __forceinline__ bool tile(
    AttnP p, int t, int hg, int blk, const uint4 (&qcache)[4],
    float& m_run, float& sum_run, float (&acc)[8][4], bool previous_live,
    bf16* Qs, bf16* Ks, float* Ss, bf16* Ps, const bf16** rowp, float* sc_s)
{
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int sh = tid >> 4, sl = tid & 15;
    const bf16** cl = rowp + (REUSE ? 64 : 0);
    const int R = p.ring_rows ? p.ring_rows : WIN;
    const int tpos = pos_of(start_of(p.start, p.dstart), p.dstart, p.dsess, t);
    const bf16 *kvr = p.kvw, *ckr = p.ck;
    if (p.dsess) {
        const int sess = p.dsess[t];
        kvr += (size_t)sess * R * HD;
        ckr = cq_slab<true>(ckr, p.ck_ss, sess);
    }
    int has = 0;
    if (tid < 64) {
        const int i = blk * 64 + tid;
        const bf16* rp = nullptr;
        if (i < p.WL) {
            if (!p.decode) {
                const int id = max(t - (WIN - 1), 0) + i;
                if (id <= t) rp = p.kvw + (size_t)id * HD;
            } else {
                const int pos = tpos - (WIN - 1) + i;
                if (pos >= p.floor) rp = kvr + (size_t)(pos % R) * HD;
            }
        } else if (i < p.WL + p.TK) {
            const int j = p.topk[(size_t)t * p.ldtk + i - p.WL];
            if (j >= 0) rp = cq_slot<true>(ckr, j);
        }
        cl[tid] = rp;
        has = rp != nullptr;
    }
    if (!__syncthreads_or(has)) return false;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c = tid + i * AT_THREADS;
        *(uint4*)(Qs + (c >> 6) * AT_QLD + (c & 63) * 8) = qcache[i];
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int c = tid + i * AT_THREADS, k = c >> 6;
        const bf16* rp = cl[k];
        if ((REUSE && previous_live && rp == rowp[k]) || cq_tag(rp)) continue;
        cp_async16(Ks + k * AT_KLD + (c & 63) * 8,
                   rp ? (const void*)((const uint4*)rp + (c & 63)) : (const void*)p.q,
                   rp ? 16 : 0);
    }
    cp_async_commit();
    // Keep each packed row's baseline widen bit operations; no arithmetic
    // approximation or reordering is involved in omitting identical stores.
    uint2 cd[8];
    uint32_t sb[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int slot = w + 8 * k;
        const bf16* rp = cl[slot];
        if (!cq_tag(rp) || (REUSE && previous_live && rp == rowp[slot])) continue;
        const uint8_t* r = cq_row(rp);
        cd[k] = __ldg((const uint2*)r + lane);
        sb[k] = __ldg(r + CQ_CODES + lane);
    }
    cp_async_wait0();
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int slot = w + 8 * k;
        const bf16* rp = cl[slot];
        if (!cq_tag(rp) || (REUSE && previous_live && rp == rowp[slot])) continue;
        const __nv_bfloat162 s2 = __float2bfloat162_rn(f_e4m3((uint8_t)sb[k]));
        uint32_t o[8];
        cq_dq4(cd[k].x, s2, o[0], o[1]);
        cq_dq4(cd[k].x >> 16, s2, o[2], o[3]);
        cq_dq4(cd[k].y, s2, o[4], o[5]);
        cq_dq4(cd[k].y >> 16, s2, o[6], o[7]);
        uint4* dst = (uint4*)(Ks + slot * AT_KLD + lane * 16);
        dst[0] = make_uint4(o[0], o[1], o[2], o[3]);
        dst[1] = make_uint4(o[4], o[5], o[6], o[7]);
    }
    __syncthreads();
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
        __syncthreads();
        {   // online softmax, head sh, 16 threads x 4 slots
            const float4 s4 = *(const float4*)(Ss + sh * AT_SLD + sl * 4);
            float mb = fmaxf(fmaxf(s4.x, s4.y), fmaxf(s4.z, s4.w));
            for (int o = 8; o; o >>= 1) mb = fmaxf(mb, __shfl_xor_sync(0xFFFFFFFFu, mb, o));
            const float m_new = fmaxf(m_run, mb);
            const float scl = expf(m_run - m_new);
            const float p0 = expf(s4.x - m_new), p1 = expf(s4.y - m_new), p2 = expf(s4.z - m_new), p3 = expf(s4.w - m_new);
            float ps = (p0 + p1) + (p2 + p3);
            for (int o = 8; o; o >>= 1) ps += __shfl_xor_sync(0xFFFFFFFFu, ps, o);
            sum_run = sum_run * scl + ps;
            m_run = m_new;
            bf16* pr = Ps + sh * AT_PLD + sl * 4;
            pr[0] = __float2bfloat16_rn(p0);
            pr[1] = __float2bfloat16_rn(p1);
            pr[2] = __float2bfloat16_rn(p2);
            pr[3] = __float2bfloat16_rn(p3);
            if (sl == 0) sc_s[sh] = scl;
        }
        __syncthreads();
        {   // acc[16 heads][64 dims of this warp] = acc * scale + P . V   (V = the same K rows)
            const float sc0 = sc_s[lane >> 2], sc1 = sc_s[(lane >> 2) + 8];
#pragma unroll
            for (int nt = 0; nt < 8; nt++)
            {
                acc[nt][0] *= sc0;
                acc[nt][1] *= sc0;
                acc[nt][2] *= sc1;
                acc[nt][3] *= sc1;
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
                    mma16816(acc[np * 2], a, b[0], b[1]);
                    mma16816(acc[np * 2 + 1], a, b[2], b[3]);
                }
            }
        }
        __syncthreads();

    return true;
}
__device__ __forceinline__ void finish(
    AttnP p, int t, int hg, float m_run, float sum_run,
    const float (&acc)[8][4], float* fin_s)
{
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int sh = tid >> 4, sl = tid & 15;
    if (sl == 0) fin_s[sh] = sum_run + expf(p.sink[hg * 16 + sh] - m_run);
    __syncthreads();
    {
        const int r0 = lane >> 2;
        const float d0 = fin_s[r0], d1 = fin_s[r0 + 8];
        bf16* o0 = p.out + ((size_t)t * p.heads + hg * 16 + r0) * HD;
        bf16* o1 = p.out + ((size_t)t * p.heads + hg * 16 + r0 + 8) * HD;
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            const int d = w * 64 + nt * 8 + 2 * (lane & 3);
            *(__nv_bfloat162*)(o0 + d) = __floats2bfloat162_rn(acc[nt][0] / d0, acc[nt][1] / d0);
            *(__nv_bfloat162*)(o1 + d) = __floats2bfloat162_rn(acc[nt][2] / d1, acc[nt][3] / d1);
        }
    }
    __syncthreads();
}

__global__ void __launch_bounds__(AT_THREADS, 1) k_rowpair(AttnP p)
{
    extern __shared__ __align__(16) uint8_t smem[];
    bf16* Qs = (bf16*)smem;
    bf16* Ks = Qs + 16 * AT_QLD;
    float* Ss = (float*)(Ks + 64 * AT_KLD);
    bf16* Ps = (bf16*)(Ss + 16 * AT_SLD);
    const bf16** rowp = (const bf16**)(Ps + 16 * AT_PLD);
    float* sc_s = (float*)(rowp + 128);
    float* fin_s = sc_s + 16;
    const int t = blockIdx.x * 2, hg = blockIdx.y, tid = threadIdx.x;
    const bool paired = t + 1 < p.n;
    uint4 q0[4], q1[4];
    const uint4* src0 = (const uint4*)(p.q + ((size_t)t * p.heads + hg * 16) * HD);
    const uint4* src1 = (const uint4*)(p.q + ((size_t)(paired ? t + 1 : t) * p.heads + hg * 16) * HD);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        q0[i] = src0[tid + i * AT_THREADS];
        q1[i] = src1[tid + i * AT_THREADS];
    }
    float acc0[8][4], acc1[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc0[i][j] = acc1[i][j] = 0.0f;
    float m0 = -1e30f, m1 = -1e30f, sum0 = 0.0f, sum1 = 0.0f;
    const int nblk = (p.WL + p.TK + 63) >> 6;
    for (int blk = 0; blk < nblk; ++blk) {
        const bool live0 = tile<false>(p,t,hg,blk,q0,m0,sum0,acc0,false,Qs,Ks,Ss,Ps,rowp,sc_s);
        if (paired) tile<true>(p,t+1,hg,blk,q1,m1,sum1,acc1,live0,Qs,Ks,Ss,Ps,rowp,sc_s);
    }
    finish(p,t,hg,m0,sum0,acc0,fin_s);
    if (paired) finish(p,t+1,hg,m1,sum1,acc1,fin_s);
}
} // namespace rowpair_detail

namespace sweep_variant {
inline bool paired(const ds41a::AttnP& p) { return p.TK && p.ckq && p.n > 1; }
inline const void* kernel(const ds41a::AttnP& p) {
    if (paired(p)) return (const void*)rowpair_detail::k_rowpair;
    if (p.floor) return (const void*)ds41a::k_attn_cpa_t<true,true>;
    return p.TK ? (const void*)ds41a::k_attn_cpa_t<false,true> : (const void*)ds41a::k_attn_cpa_t<false,false>;
}
inline const char* ptxas_key(const ds41a::AttnP& p) {
    if (paired(p)) return "k_rowpair";
    return p.floor ? "k_attn_cpa_tILb1ELb1EE" : p.TK ? "k_attn_cpa_tILb0ELb1EE" : "k_attn_cpa_tILb0ELb0EE";
}
inline size_t dynamic_smem(const ds41a::AttnP&) { return ds41a::AT_SMEM_CPA; }
inline cudaError_t init(const ds41a::AttnP& p) {
    return cudaFuncSetAttribute(kernel(p),cudaFuncAttributeMaxDynamicSharedMemorySize,dynamic_smem(p));
}
inline void launch(const ds41a::AttnP& p,cudaStream_t st) {
    if (paired(p)) rowpair_detail::k_rowpair<<<dim3((p.n+1)/2,p.heads/16),ds41a::AT_THREADS,ds41a::AT_SMEM_CPA,st>>>(p);
    else if (p.floor) ds41a::k_attn_cpa_t<true,true><<<dim3(p.n,p.heads/16),ds41a::AT_THREADS,ds41a::AT_SMEM_CPA,st>>>(p);
    else if (p.TK) ds41a::k_attn_cpa_t<false,true><<<dim3(p.n,p.heads/16),ds41a::AT_THREADS,ds41a::AT_SMEM_CPA,st>>>(p);
    else ds41a::k_attn_cpa_t<false,false><<<dim3(p.n,p.heads/16),ds41a::AT_THREADS,ds41a::AT_SMEM_CPA,st>>>(p);
}
}
