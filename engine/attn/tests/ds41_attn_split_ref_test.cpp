// CPU-only arithmetic/reference test for the default-off two-CTA sparse attention merge.
// Serial run uses the source k_attn_cpa_t online-block order; split uses the shared GPU merge helper.
// This is not a GPU kernel numeric or speed result.
#include "../ds41_attn_split_math.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct Key { float score, v[4]; bool live; };
struct State { float m=-1e30f, sum=0.0f, acc[4]={}; };
static float bf16(float x) {
    uint32_t u; std::memcpy(&u,&x,4);
    u += 0x7fffu + ((u >> 16) & 1u); u &= 0xffff0000u;
    std::memcpy(&x,&u,4); return x;
}
static State online(const std::vector<Key>& keys, int lo, int hi) {
    State a;
    for (int blk=lo; blk<hi; ++blk) {
        float mb=-INFINITY; bool live=false;
        for (int i=blk*64;i<(blk+1)*64;++i) if (keys[i].live) {mb=std::fmax(mb,keys[i].score); live=true;}
        if (!live) continue;
        const float m=std::fmax(a.m,mb), scl=std::exp(a.m-m);
        float ps=0.0f, pa[4]={};
        for (int i=blk*64;i<(blk+1)*64;++i) if (keys[i].live) {
            float p=std::exp(keys[i].score-m); ps+=p; p=bf16(p);
            for (int j=0;j<4;++j) pa[j]+=p*keys[i].v[j];
        }
        a.sum=a.sum*scl+ps;
        for (int j=0;j<4;++j) a.acc[j]=a.acc[j]*scl+pa[j];
        a.m=m;
    }
    return a;
}
static std::vector<Key> make_keys(int session,int row,int head) {
    std::vector<Key> v(640);
    for (int i=0;i<640;++i) {
        bool live=(i<128 ? i<=row+session*3 : ((i+session+head)%7!=0 && i<128+512-(session%3)*23));
        if (session%4==0 && i<320) live=false; // empty first partition
        if (session%11==0 && row==1) live=false; // no keys; sink alone
        v[i].live=live;
        v[i].score=std::sin(float(i*17+session*7+row*11+head*13)*0.017f)*2.0f;
        for (int j=0;j<4;++j) v[i].v[j]=std::cos(float(i*3+j*19+head*5)*0.033f);
    }
    return v;
}
static int sweep(int ns,const std::string& mutant) {
    float maxerr=0.0f, sq=0.0f; int n=0;
    for (int session=0;session<ns;++session) for (int row=0;row<3;++row) for (int head=0;head<3;++head) {
        const float sink=5.0f+0.17f*head-0.03f*session;
        const auto keys=make_keys(session,row,head);
        const State serial=online(keys,0,10);
        std::vector<Key> cand=keys;
        if (mutant=="mask") for (int i=0;i<640;++i) if (!cand[i].live && i%5==0) {cand[i].live=true; cand[i].score=6.0f;}
        State a=online(cand,mutant=="partition" ? 1 : 0,5), b=online(cand,5,10);
        auto w=ds41a::split_merge_weights(a.m,a.sum,b.m,b.sum,sink);
        if (mutant=="sink") w.denom+=std::exp(sink-std::fmax(a.m,b.m));
        if (mutant=="scale") w.a=0.0f;
        const float denom=serial.sum+std::exp(sink-serial.m);
        for (int j=0;j<4;++j) {
            const float expect=serial.acc[j]/denom;
            const float got=ds41a::split_merge_value(a.acc[j],b.acc[j],w);
            if (!std::isfinite(got)) {std::puts("CPU_REF_RED nonfinite"); return 1;}
            const float e=std::fabs(got-expect);
            maxerr=std::fmax(maxerr,e); sq+=e*e; ++n;
        }
    }
    const float rms=std::sqrt(sq/n);
    const bool pass=maxerr<=0.01f && rms<=0.003f;
    std::printf("CPU_REF_%s sessions=%d mutant=%s cases=%d max_abs=%.9g rms=%.9g\n",pass?"PASS":"RED",ns,mutant.c_str(),n,maxerr,rms);
    return pass?0:1;
}
int main(int argc,char** argv) {
    const std::string mutant=argc>1?argv[1]:"none";
    if (mutant!="none" && mutant!="mask" && mutant!="partition" && mutant!="sink" && mutant!="scale") return 2;
    for (int ns : {8,32}) {
        // g6m4.cu:577,643,1731: MTP5 is limited to <=8 sessions; at 32, MTP is off.
        const int decode_rows=ns<=8 ? ns*6 : ns;
        if (ds41a::split_eligible(decode_rows,128,512) || !ds41a::split_eligible(4096,128,512) ||
            ds41a::split_eligible(128,128,512)) return 3;
        std::printf("SESSION_SELECT_PASS sessions=%d decode_rows=%d prefill_rows=4096\n",ns,decode_rows);
    }
    int rc=0; rc|=sweep(8,mutant); rc|=sweep(32,mutant); return rc;
}
