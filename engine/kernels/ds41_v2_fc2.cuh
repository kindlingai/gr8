// gr8 v2 experimental resident-weight down projection. One CTA loads one
// expert's 128-column weight slab ONCE, then walks every bucketed row group.
// Reuses gr8 record/MMA/quantized activation formats and weighted-half epilogue.
#pragma once
#include "ds_moe_tc.cuh"

namespace ds41v2moe {
template<int WARPS,int TPC,int I>
constexpr size_t resident_smem() {
    return size_t(TPC)*(I/mx4::KG)*mx4::GR + size_t(WARPS)*8*mxtc::YP*sizeof(__half);
}
// Optional heavy-first order without moving any bucket data. Every key is
// unique because equal tile counts break ties by original slot id.
template<int NE,int TK,int MM>
__global__ void k_order_desc(const dsr::PlanMT<TK,MM>* p,int* order) {
    const int s=threadIdx.x,ns=p->n_slots;
    if(s>=ns)return;
    const int n=p->slot_nt[s];int pos=0;
    for(int j=0;j<ns;++j)pos+=p->slot_nt[j]>n||(p->slot_nt[j]==n&&j<s);
    order[pos]=s;
}
template<int I,int NOUT,int WARPS=4,int TW=2,int TPC=8,int GROUPS=4>
__global__ void __launch_bounds__(WARPS*32,1) k_fc2_resident(mxtc::Args a,const int* order=nullptr) {
    constexpr int NG=I/mx4::KG,OT=NOUT/16,NB=OT/TPC,GB=NB/GROUPS;
    constexpr int NTH=WARPS*32,NTG=WARPS*TW,NSW=(NG+1)/2;
    constexpr size_t WB=size_t(TPC)*NG*mx4::GR;
    static_assert(I%64==0 && OT%TPC==0 && NB%GROUPS==0 && TPC*16<=128,"resident fc2 tiles");
    static_assert(NSW*4<=mxtc::ASP && WB%16==0,"resident fc2 storage");
    const int pos=blockIdx.x/GB,ob=a.col_group*GB+blockIdx.x%GB;
    if(a.n_slots_dev&&pos>=*a.n_slots_dev)return;
    const int slot=order?order[pos]:pos;
    const int nt=a.slot_nt[slot],tile0=a.slot_tile0[slot];
    if(!nt)return;
    const int tid=threadIdx.x,warp=tid>>5,lane=tid&31,g=lane>>2,c=lane&3;
    extern __shared__ __align__(16) uint8_t sm[];
    __half* ys=reinterpret_cast<__half*>(sm+WB)+warp*8*mxtc::YP;
    const uint8_t* weights=a.fc2+dsr::ro_fc2_rec(a.slot_expert[slot],ob*TPC,0,OT,NG)*mx4::GR;
    // This is the ONLY global weight load site. It is outside the row loop.
    for(size_t i=size_t(tid)*16;i<WB;i+=NTH*16)mxtc::cp16(sm+i,weights+i);
    mxtc::cp_commit();mxtc::cp_wait<0>();__syncthreads();
    for(int first=0;first<nt;first+=NTG) {
        const int nlive=min(max(nt-first-warp*TW,0),TW);
        uint4 bq[TW][NG];uint32_t sw[TW][NSW];float w0[TW],w1[TW];int mask[TW];
        #pragma unroll
        for(int tw=0;tw<TW;++tw) {
            const int tile=tile0+first+warp*TW+tw;
            const size_t row=size_t(tile)*8+g;
            const bool live=tw<nlive;
            #pragma unroll
            for(int r=0;r<NG;++r)bq[tw][r]=live?__ldg(reinterpret_cast<const uint4*>(a.aq+row*I+r*mxtc::XREC+c*16)):make_uint4(0,0,0,0);
            #pragma unroll
            for(int k=0;k<NSW;++k)sw[tw][k]=live?__ldg(reinterpret_cast<const uint32_t*>(a.as+row*mxtc::ASP+k*4)):0;
            w0[tw]=live?a.tile_w[tile*8+2*c]:0.f;w1[tw]=live?a.tile_w[tile*8+2*c+1]:0.f;
            mask[tw]=live?a.tile_mask[tile]:0;
        }
        float acc[TPC][TW][4]={};
        mxtc::static_for<NG>([&](auto rc) {
            constexpr int r=decltype(rc)::value;
            #pragma unroll
            for(int j=0;j<TPC;++j) {
                mxtc::Rec rec;rec.load(sm+(j*NG+r)*mx4::GR,lane);
                const uint32_t a0=mxtc::x4lo(rec.c[0]),a1=mxtc::x4hi(rec.c[0]),a2=mxtc::x4lo(rec.c[1]),a3=mxtc::x4hi(rec.c[1]);
                const uint32_t a4=mxtc::x4lo(rec.c[2]),a5=mxtc::x4hi(rec.c[2]),a6=mxtc::x4lo(rec.c[3]),a7=mxtc::x4hi(rec.c[3]);
                #pragma unroll
                for(int tw=0;tw<TW;++tw)if(tw<nlive) {
                    mxtc::mma_mx<0,(2*r)&3>(acc[j][tw],a0,a1,a2,a3,bq[tw][r].x,bq[tw][r].y,rec.s,sw[tw][r/2]);
                    mxtc::mma_mx<2,(2*r+1)&3>(acc[j][tw],a4,a5,a6,a7,bq[tw][r].z,bq[tw][r].w,rec.s,sw[tw][r/2]);
                }
            }
        });
        #pragma unroll
        for(int tw=0;tw<TW;++tw) {
            if(tw>=nlive)break;
            #pragma unroll
            for(int j=0;j<TPC;++j) {
                __half* dst=ys+(2*c)*mxtc::YP+j*16+g;
                dst[0]=__float2half_rn(w0[tw]*acc[j][tw][0]);dst[8]=__float2half_rn(w0[tw]*acc[j][tw][2]);
                dst[mxtc::YP]=__float2half_rn(w1[tw]*acc[j][tw][1]);dst[mxtc::YP+8]=__float2half_rn(w1[tw]*acc[j][tw][3]);
            }
            __syncwarp();
            __half* dst=a.ypart+(size_t(tile0+first+warp*TW+tw)*8)*NOUT+ob*TPC*16;
            constexpr int PR=TPC*2;
            for(int i=lane;i<8*PR;i+=32) {
                const int row=i/PR,pc=i%PR;
                if(mask[tw]>>row&1)*reinterpret_cast<uint4*>(dst+size_t(row)*NOUT+pc*8)=*reinterpret_cast<const uint4*>(ys+row*mxtc::YP+pc*8);
            }
            __syncwarp();
        }
        // Every warp has its own output stage; shared weights are never changed.
    }
}
} // namespace ds41v2moe
