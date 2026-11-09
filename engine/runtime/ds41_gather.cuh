// DS41 prefill gather (GR8_DS_PF_GATHER=1). Needs kernels/glue.cuh (glue::peer_slot, bf16x8_to_f) and g6nr.h before it; no includes of its own,
// because ds41_step.cuh includes it inside namespace g6rt.
#pragma once
// GR8_DS_PF_GATHER=1 (default off; prefill consumes of more than MX rows on the main Model only): k_ds41_gather reads the NR slots of collective
// c with 16-byte loads (one uint4 = 8 BF16 per thread and slot, every slot load in flight before the adds), sums them in rank order p = 0..NR-1 in
// FP32 from 0.f -- the same adds in the same order as Ds41Block::operator() -- rounds to BF16 and stores the row to DEVICE memory (Model::ds41_gsum).
// The consume kernel then reads that buffer through Ds41DevBlock. The consumers only ever use rn16(block(k)) (ds_mhc.cuh), and rn16 of a BF16 value is
// that value, so every result is bit-equal to the slot reads. The peers' tails are awaited first by glue::k_wait_peers (one CTA, as GR8_DS_WAIT_K=1).
// grid (HID / 8 / 128, rows), block 128.
constexpr int DS41_GATHER_T=128;
__global__ void __launch_bounds__(DS41_GATHER_T) k_ds41_gather(g6::FabricDev f,const u64* stepp,int c,int H,__nv_bfloat16* out) {
    const u64 step=*stepp;
    const int m=blockIdx.y, col0=(blockIdx.x*DS41_GATHER_T+threadIdx.x)*8;
    if(col0>=H) return;
    const size_t off=((size_t)m*H+col0)*2;
    uint4 v[g6nr::NR];
    #pragma unroll
    for(int p=0;p<g6nr::NR;++p) v[p]=__ldcv(reinterpret_cast<const uint4*>(glue::peer_slot(f,step,c,p)+off));
    float x[8]={0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    #pragma unroll
    for(int p=0;p<g6nr::NR;++p) {
        float b[8];
        glue::bf16x8_to_f(v[p],b);
        #pragma unroll
        for(int k=0;k<8;++k) x[k]+=b[k];
    }
    uint4 o;
    __nv_bfloat16* ob=reinterpret_cast<__nv_bfloat16*>(&o);
    #pragma unroll
    for(int k=0;k<8;++k) ob[k]=__float2bfloat16_rn(x[k]);
    *reinterpret_cast<uint4*>(out+(size_t)m*H+col0)=o;
}
// the consume block of a gathered collective: device memory, already reduced and BF16; no wait (k_wait_peers + k_ds41_gather ran first, in stream order)
struct Ds41DevBlock {
    const __nv_bfloat16* sum;
    const u64* stepp;
    __device__ __forceinline__ u64 init() { return *stepp; }
    __device__ __forceinline__ void wait(u64) const {}
    __device__ __forceinline__ float operator()(size_t k) const { return __bfloat162float(sum[k]); }
};
