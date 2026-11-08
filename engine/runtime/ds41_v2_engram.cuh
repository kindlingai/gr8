// v2 owned-cell wire. Include globally after engram_dev.cuh.
#pragma once
namespace ds41::v2::engram_wire {
// Fixed worst-case per TOKEN row, not fixed column ownership. Header carries
// device count + mask; only cells owned by this rank's sidecar (range or columns)
// are materialized. Padding is sent too: this is NOT a variable-wire claim.
constexpr int CELLS=24, CODES=256, SCALES=8, HEADER=16;
constexpr size_t ROW_BYTES=HEADER+CELLS*(CODES+SCALES);
static_assert(ROW_BYTES==6352);
__global__ void pack(const uint32_t* ids,const uint8_t* stage,const uint8_t* scales,
                     int64_t first,int64_t valid,int rows,uint8_t* out) {
    const int r=blockIdx.x,t=threadIdx.x;
    uint8_t* dst=out+size_t(r)*ROW_BYTES;
    if(t==0) {
        unsigned mask=0;
        for(int c=0;c<CELLS;++c) {
            const int64_t loc=int64_t(ids[size_t(r)*CELLS+c])-first;
            if(loc>=0 && loc<valid)mask|=1u<<c;
        }
        reinterpret_cast<uint4*>(dst)[0]=make_uint4(__popc(mask),mask,0,0);
    }
    for(int i=t;i<CELLS*(CODES+SCALES);i+=blockDim.x) {
        const int c=i/(CODES+SCALES),b=i%(CODES+SCALES);
        const int64_t loc=int64_t(ids[size_t(r)*CELLS+c])-first;
        dst[HEADER+i]=(loc>=0 && loc<valid) ? (b<CODES?stage[(size_t(r)*CELLS+c)*CODES+b]:scales[size_t(loc)*SCALES+b-CODES]):0;
    }
}
// Column-sharded counterpart; wire layout and the rank-order unpack stay unchanged.
__global__ void pack_col(const uint32_t* ids,const uint8_t* stage,const uint8_t* scales,
                     ::ds41::engram::ColShard columns,int rows,uint8_t* out) {
    const int r=blockIdx.x,t=threadIdx.x;
    uint8_t* dst=out+size_t(r)*ROW_BYTES;
    if(t==0) {
        unsigned mask=0;
        for(int c=0;c<CELLS;++c) {
            const int64_t loc=columns.at(c,ids[size_t(r)*CELLS+c]);
            if(loc>=0)mask|=1u<<c;
        }
        reinterpret_cast<uint4*>(dst)[0]=make_uint4(__popc(mask),mask,0,0);
    }
    for(int i=t;i<CELLS*(CODES+SCALES);i+=blockDim.x) {
        const int c=i/(CODES+SCALES),b=i%(CODES+SCALES);
        const int64_t loc=columns.at(c,ids[size_t(r)*CELLS+c]);
        dst[HEADER+i]=(loc>=0) ? (b<CODES?stage[(size_t(r)*CELLS+c)*CODES+b]:scales[size_t(loc)*SCALES+b-CODES]):0;
    }
}
__global__ void unpack(const uint8_t* in,int Q,int owned,int nr,__nv_bfloat16* emb) {
    const int r=blockIdx.x,c=blockIdx.y,b=threadIdx.x;
    float x=0.f;
    for(int p=0;p<nr;++p) {
        const uint8_t* src=in+(size_t(p)*Q+r)*ROW_BYTES;
        if(reinterpret_cast<const unsigned*>(src)[1]&(1u<<c)) {
            src+=HEADER+c*(CODES+SCALES);
            x+=__bfloat162float(__ushort_as_bfloat16(::ds41::engram::dequant1(src[b],src[CODES+b/32])));
        }
    }
    emb[size_t(r)*CELLS*CODES+c*CODES+b]=__float2bfloat16_rn(x);
}
} // namespace ds41::v2::engram_wire
