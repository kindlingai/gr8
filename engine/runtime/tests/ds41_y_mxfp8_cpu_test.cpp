// CPU oracle for the optional lossy MXFP8 wire form. This does not execute a GPU kernel.
#include "../../mhc/ds_cpu.h"
#include "../ds41_rowshard.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
namespace {
constexpr int M=4096,H=128,NR=4,Q=M/NR;
float fp8(uint8_t b) {
    const int e=(b>>3)&15,m=b&7;
    const float a=e ? std::ldexp(1.f+m/8.f,e-7) : std::ldexp(float(m),-9);
    return (b&128) ? -a : a;
}
uint8_t code(float v) {
    const bool neg=std::signbit(v); float a=std::fabs(v);
    int c;
    if (a>=448.f) c=126;
    else if (a<std::ldexp(1.f,-6)) c=std::clamp(int(std::nearbyint(a*512.f)),0,7);
    else {
        const int e=int(std::floor(std::log2(a)));
        int mant=int(std::nearbyint((std::ldexp(a,-e)-1.f)*8.f));
        int exp=e+7;
        if (mant==8) { ++exp; mant=0; }
        c=std::min(126,(exp<<3)+mant);
    }
    return uint8_t(c|(neg?128:0));
}
void quant(const float* x,int rows,uint8_t* codes,uint8_t* scales) {
    for(int r=0;r<rows;++r) for(int b=0;b<H/32;++b) {
        float a=1e-4f;
        for(int j=0;j<32;++j) a=std::max(a,std::fabs(x[(size_t)r*H+b*32+j]));
        int exp; const float z=a*0.002232142857142857f;
        const float f=std::frexp(z,&exp);
        const int e=exp-(f==0.5f ? 1:0);
        scales[(size_t)r*(H/32)+b]=uint8_t(e+127);
        for(int j=0;j<32;++j) codes[(size_t)r*H+b*32+j]=code(std::ldexp(x[(size_t)r*H+b*32+j],-e));
    }
}
void run(int sessions) {
    double maxabs=0,meanabs=0; size_t n=0;
    for(int s=0;s<sessions;++s) {
        std::vector<float> x((size_t)M*H);
        for(size_t i=0;i<x.size();++i) x[i]=ds_mhc::bf16r((int((i*37+s*19)%4093)-2046)*0.0012f);
        std::vector<uint8_t> refc(x.size()),refs(x.size()/32),localc(x.size()),locals(refs.size());
        quant(x.data(),M,refc.data(),refs.data());
        for(int p=0;p<NR;++p) quant(x.data()+(size_t)p*Q*H,Q,localc.data()+(size_t)p*Q*H,locals.data()+(size_t)p*Q*H/32);
        assert(std::memcmp(refc.data(),localc.data(),refc.size())==0);
        assert(std::memcmp(refs.data(),locals.data(),refs.size())==0);
        std::array<std::vector<uint8_t>,NR> packets;
        for(int owner=0;owner<NR;++owner) {
            packets[owner].resize(ds41::rowshard::y_mxfp8_bytes(Q,H));
            std::memcpy(packets[owner].data(),localc.data()+(size_t)owner*Q*H,(size_t)Q*H);
            std::memcpy(packets[owner].data()+(size_t)Q*H,locals.data()+(size_t)owner*Q*H/32,(size_t)Q*H/32);
        }
        for(int rank=0;rank<NR;++rank) {
            std::vector<uint8_t> gotc(x.size()),gots(refs.size());
            for(int owner=0;owner<NR;++owner) {
                std::memcpy(gotc.data()+(size_t)owner*Q*H,packets[owner].data(),(size_t)Q*H);
                std::memcpy(gots.data()+(size_t)owner*Q*H/32,packets[owner].data()+(size_t)Q*H,(size_t)Q*H/32);
            }
            assert(gotc==refc && gots==refs);
        }
        for(size_t i=0;i<x.size();++i) {
            const float y=ds_mhc::bf16r(fp8(localc[i])*std::ldexp(1.f,int(locals[i/32])-127));
            const double e=std::fabs(double(y)-x[i]); maxabs=std::max(maxabs,e); meanabs+=e; ++n;
        }
    }
    std::printf("Y_MXFP8_CPU_CODE_SCALE_EXACT sessions=%d rows=%d H=%d Q=%d all_ranks=1 dequant_maxabs=%.9g dequant_meanabs=%.9g (no pass threshold)\n",
                sessions,M,H,Q,maxabs,meanabs/n);
}
}
int main() { assert(ds41::rowshard::y_mxfp8_bytes(Q,5120)==5406720); run(8); run(32); }
