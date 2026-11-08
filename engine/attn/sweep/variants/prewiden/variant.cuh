#include "../../scratch_guard.cuh"
#define SWEEP_VARIANT_HAS_SETUP 1
// One timed global widening pass; every attention CTA then loads BF16 keys.
// All arithmetic helpers and the attention kernel are our original ds41a code.
namespace sweep_variant {
using ds41a::bf16;

// One warp per packed row, one lane per 16-value scale group. Match cq_store's
// operation order and BF16 multiplication exactly, including signed zero.
__global__ void k_prewiden_keys(const uint8_t* packed, bf16* wide, size_t rows) {
    const size_t row = size_t(blockIdx.x) * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= rows) return;
    const uint8_t* src = packed + row * ds41a::CQ_ROW;
    const uint2 codes = __ldg(reinterpret_cast<const uint2*>(src) + lane);
    const __nv_bfloat162 scale = __float2bfloat162_rn(ds41a::f_e4m3(__ldg(src + ds41a::CQ_CODES + lane)));
    uint32_t v[8];
    ds41a::cq_dq4(codes.x, scale, v[0], v[1]);
    ds41a::cq_dq4(codes.x >> 16, scale, v[2], v[3]);
    ds41a::cq_dq4(codes.y, scale, v[4], v[5]);
    ds41a::cq_dq4(codes.y >> 16, scale, v[6], v[7]);
    uint4* dst = reinterpret_cast<uint4*>(wide + row * ds41a::HD + lane * 16);
    dst[0] = make_uint4(v[0], v[1], v[2], v[3]);
    dst[1] = make_uint4(v[4], v[5], v[6], v[7]);
}

struct Scratch {
    bf16* data = nullptr;
    ~Scratch() { if (data) sweep_memory::scratch_free(data); }
};
inline Scratch scratch;

// The sweep describes a single session's slab extent through ck_ss. Unknown
// extents or multiple sessions use off, never a guessed allocation or stride.
inline bool supported(const ds41a::AttnP& p) {
    return p.TK && p.ckq && p.ck && !p.dsess && p.ck_ss && p.ck_ss % ds41a::CQ_ROW == 0;
}
inline size_t key_rows(const ds41a::AttnP& p) { return p.ck_ss / ds41a::CQ_ROW; }
inline size_t scratch_bytes(const ds41a::AttnP& p) {
    return supported(p) ? key_rows(p) * ds41a::HD * sizeof(bf16) : 0;
}
inline const void* kernel(const ds41a::AttnP& p) {
    if (supported(p)) return p.floor ? (const void*)ds41a::k_attn_cpa_t<true,false>
                                    : (const void*)ds41a::k_attn_cpa_t<false,false>;
    if (p.floor) return (const void*)ds41a::k_attn_cpa_t<true,true>;
    return p.TK ? (const void*)ds41a::k_attn_cpa_t<false,true>
                : (const void*)ds41a::k_attn_cpa_t<false,false>;
}
inline const char* ptxas_key(const ds41a::AttnP& p) {
    if (supported(p)) return p.floor ? "k_attn_cpa_tILb1ELb0EE" : "k_attn_cpa_tILb0ELb0EE";
    return p.floor ? "k_attn_cpa_tILb1ELb1EE" : p.TK ? "k_attn_cpa_tILb0ELb1EE" : "k_attn_cpa_tILb0ELb0EE";
}
inline size_t dynamic_smem(const ds41a::AttnP&) { return ds41a::AT_SMEM_CPA; }
inline cudaError_t init(const ds41a::AttnP& p) {
    if (scratch.data) {
        const cudaError_t e = sweep_memory::scratch_free(scratch.data);
        if (e != cudaSuccess) return e;
        scratch.data = nullptr;
    }
    cudaError_t e = cudaFuncSetAttribute(kernel(p), cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_smem(p));
    if (e != cudaSuccess) return e;
    if (supported(p)) return sweep_memory::scratch_alloc(reinterpret_cast<void**>(&scratch.data), scratch_bytes(p));
    return cudaSuccess;
}
inline void prepass(const ds41a::AttnP& p, cudaStream_t stream) {
    k_prewiden_keys<<<(key_rows(p) + 7) / 8,256,0,stream>>>(
        reinterpret_cast<const uint8_t*>(p.ck), scratch.data, key_rows(p));
}
inline void setup_chunk(const ds41a::AttnP& p, cudaStream_t stream) { if(supported(p))prepass(p,stream); }
inline void launch(const ds41a::AttnP& p, cudaStream_t stream) {
    const dim3 grid(p.n, p.heads / 16);
    if (supported(p)) {
        auto q = p;
        q.ck = scratch.data;
        q.ckq = 0;
        q.ck_ss = key_rows(p) * ds41a::HD;
        if (p.floor) ds41a::k_attn_cpa_t<true,false><<<grid,ds41a::AT_THREADS,ds41a::AT_SMEM_CPA,stream>>>(q);
        else ds41a::k_attn_cpa_t<false,false><<<grid,ds41a::AT_THREADS,ds41a::AT_SMEM_CPA,stream>>>(q);
    } else if (p.floor) ds41a::k_attn_cpa_t<true,true><<<grid,ds41a::AT_THREADS,ds41a::AT_SMEM_CPA,stream>>>(p);
    else if (p.TK) ds41a::k_attn_cpa_t<false,true><<<grid,ds41a::AT_THREADS,ds41a::AT_SMEM_CPA,stream>>>(p);
    else ds41a::k_attn_cpa_t<false,false><<<grid,ds41a::AT_THREADS,ds41a::AT_SMEM_CPA,stream>>>(p);
}
}
