// Host-only checks for the INT4 query permutation, stable compaction and knob contract.
#include "../knobs-rs/gr8_knobs.h"
#include "../attn/kv4_dispatch.h"
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

static const char* names[]={"GR8_KV4_CTX","GR8_KV4_FUSE_ROT","GR8_KV4_QT","GR8_KV4_SC","GR8_KV4_SC_TR","GR8_KV4_FOLD","GR8_KV4_TC"};
static void knobs(const std::array<int,7>& values,int changed=-1,const char* invalid=nullptr){
    const pid_t pid=fork();assert(pid>=0);
    if(!pid){
        for(int k=0;k<7;k++)unsetenv(names[k]);
        if(changed>=0){const std::string v=std::to_string(values[changed]);setenv(names[changed],invalid?invalid:v.c_str(),1);}
        const int errors=knob::init();
        if(invalid)_exit(errors>0?0:1);
        const std::array<int,7> got={int(knob::kv4_ctx()),int(knob::kv4_fuse_rot()),int(knob::kv4_qt()),int(knob::kv4_sc()),int(knob::kv4_sc_tr()),int(knob::kv4_fold()),int(knob::kv4_tc())};
        _exit(!errors&&got==values?0:1);
    }
    int status=0;assert(waitpid(pid,&status,0)==pid);assert(WIFEXITED(status)&&WEXITSTATUS(status)==0);
}
static int padded(int d){return d+(d>>5);}
static void permutation(){
    std::array<uint32_t,512> orig{},trans{},roundtrip{};std::array<uint32_t,528> sm{};
    for(int d=0;d<512;d++){orig[d]=0x80000000u+unsigned(d)*7919;assert(padded(d)>=0&&padded(d)<528);}
    for(int k=0;k<16;k++){
        uint32_t banks=0;
        for(int lane=0;lane<32;lane++){const int d=lane+32*k;const uint32_t bit=1u<<(padded(d)%32);assert(!(banks&bit));banks|=bit;sm[padded(d)]=orig[d];}
        assert(banks==0xffffffffu);
    }
    for(int c=0;c<4;c++)for(int e=0;e<4;e++){
        uint32_t banks=0;
        for(int lane=0;lane<32;lane++){
            const int d=16*lane+4*c+e,index=33*(lane>>1)+16*(lane&1)+4*c+e;
            assert(index==padded(d));const uint32_t bit=1u<<(index%32);assert(!(banks&bit));banks|=bit;
            trans[(c*32+lane)*4+e]=sm[index];roundtrip[d]=trans[(c*32+lane)*4+e];
            assert(trans[(c*32+lane)*4+e]==orig[d]);
        }
        assert(banks==0xffffffffu);
    }
    assert(orig==roundtrip);
}
static std::vector<int> compact(const std::array<bool,256>& mask,int warps){
    std::vector<int> out(256,-1);int live=0;
    const int threads=warps*32;
    for(int pass=0;pass<256/threads;pass++){
        uint32_t ballots[8]={};int bases[8]={};
        for(int w=0;w<warps;w++)for(int lane=0;lane<32;lane++)if(mask[pass*threads+32*w+lane])ballots[w]|=1u<<lane;
        for(int w=0;w<warps;w++){bases[w]=live;live+=__builtin_popcount(ballots[w]);}
        for(int t=0;t<threads;t++)if(mask[pass*threads+t]){
            const int w=t>>5,lane=t&31,pos=bases[w]+__builtin_popcount(ballots[w]&((1u<<lane)-1u));
            assert(out[pos]==-1);out[pos]=pass*threads+t;
        }
    }
    out.resize(live);return out;
}
static bool same(float a,float b){return !std::memcmp(&a,&b,sizeof a);}
static void compaction(){
    for(int warps:{4,8})for(int count:{0,1,17,64,256})for(int pattern:{0,1}){
        std::array<bool,256> mask{};
        for(int k=0;k<count;k++)mask[pattern?(k*73)%256:k]=true;
        const auto slots=compact(mask,warps);assert(int(slots.size())==count);
        int last=-1;for(int j:slots){assert(j>last&&mask[j]);last=j;}
        for(int hg:{2,4,8})for(int nh=1;nh<=hg;nh++)for(int dim=0;dim<512;dim++){
            float p[256][8]={},x[256];
            for(int j=0;j<256;j++){
                // Head zero keeps each live row. Head one skips a NaN scale with signed zero.
                x[j]=(j%19==0)?std::numeric_limits<float>::quiet_NaN():float((j*13+dim)%31-15)*0.03125f;
                for(int h=0;h<nh;h++)p[j][h]=!mask[j]||((j+h)%3==0&&h>0)||((j%19==0)&&h>0)?((j&1)?-0.f:0.f):float((j+h)%7+1)*0.00390625f;
            }
            float ref[8]={},got[8]={};
            for(int j=0;j<256;j++)if(mask[j])for(int h=0;h<hg;h++)if(p[j][h]!=0.f)ref[h]=std::fma(p[j][h],x[j],ref[h]);
            // The double-buffered loop consumes batches in increasing order, including a short tail.
            const int nb=(count+15)/16;
            auto use=[&](int b){for(int u=0;u<16;u++){
                const int idx=b*16+u;if(idx>=count)break;const int j=slots[idx];
                for(int h=0;h<hg;h++){const float f=std::fma(p[j][h],x[j],got[h]);got[h]=p[j][h]!=0.f?f:got[h];}
            }};
            for(int b=0;b<nb;b+=2){use(b);if(b+1<nb)use(b+1);}
            for(int h=0;h<hg;h++)assert(same(ref[h],got[h]));
        }
    }
    float zero=-0.f;const float f=std::fma(0.f,std::numeric_limits<float>::quiet_NaN(),zero);
    const float selected=0.f!=0.f?f:zero;assert(same(selected,zero));
}
static void dispatch(){
    for(bool tc:{false,true})for(bool fold:{false,true})for(bool qt:{false,true})
    for(bool heads:{false,true})for(int sc:{0,1,2,3,4,8}){
        const bool expected=!tc&&!fold&&qt&&heads&&(sc==1||sc==2||sc==4);
        assert(g6kv4_dispatch::qt_enable(tc,fold,qt,heads,sc)==expected);
        if(tc||fold)assert(!g6kv4_dispatch::qt_enable(tc,fold,qt,heads,sc));
    }
    for(bool fold:{false,true})for(int rot:{-1,0,1,2}){
        assert(g6kv4_dispatch::fuse_rot_enable(fold,rot)==(!fold&&rot!=0));
        if(fold)assert(!g6kv4_dispatch::fuse_rot_enable(fold,rot));
    }
}
int main(){
    const std::array<int,7> defaults={0,0,0,0,2,0,0};knobs(defaults);
    const std::vector<int> accepted[]={{0,1,2},{0,1},{0,1},{0,1,2,4},{0,1,2},{0,1},{0,1}};
    for(int k=0;k<7;k++)for(int v:accepted[k]){auto values=defaults;values[k]=v;knobs(values,k);}
    knobs(defaults,0,"3");knobs(defaults,1,"2");knobs(defaults,2,"2");knobs(defaults,3,"3");knobs(defaults,3,"8");knobs(defaults,4,"3");
    knobs(defaults,5,"2");knobs(defaults,6,"2");
    dispatch();permutation();compaction();
    std::puts("PASS test_s6_kv4_host: knobs, exhaustive dispatch, 512-dim bank/permutation map, stable 4/8-warp compaction and float bits");
}
