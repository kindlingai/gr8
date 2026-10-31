// CPU-only structural and arithmetic model for the default-off indexer levers.
// This is not a GPU parity test: k_logits_pipe_qr still needs a run on the GPU.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

static uint32_t bits(float x) { uint32_t u; std::memcpy(&u,&x,4); return u; }
static float mul(float a,float b) { volatile float x=a*b; return x; }
static float add(float a,float b) { volatile float x=a+b; return x; }
static std::array<float,2> rotate(float a,float b,float c,float s,bool odd) {
    const float ra=add(mul(a,c),-mul(b,s));
    const float rb=add(mul(a,s),mul(b,c));
    return {odd?rb:ra,odd?ra:rb};
}
static float score(int sess,int r,int col,int key) {
    // Deterministic stand-in for the unchanged MMA+epilogue arithmetic.
    float z=0.0f;
    for(int h=0;h<32;++h) {
        const float q=float(((r+3*h+sess)%17)-8)*0.03125f;
        const float k=float(((key+5*h)%23)-11)*0.015625f;
        const float w=float((h%7)+1)*0.0625f;
        const float v=mul(q,k);
        z=add(z,mul(std::max(0.0f,v),w));
    }
    return add(z,float(col%5)*0.0009765625f);
}
static bool check(int ns,const std::string& mutant) {
    constexpr int ROWS=32, COLS=385, STAGES=3;
    std::array<float,32> inv{};
    for(int i=0;i<32;++i) inv[i]=std::pow(0.87f,float(i));
    size_t mismatches=0, cases=0, skipped=0, old_q_stage=0, new_q_stage=0;
    for(int sess=0;sess<ns;++sess) {
        for(int r=0;r<ROWS;++r) {
            const int pos=977+sess*41+r;
            std::array<std::array<float,2>,32> cached{};
            for(int i=0;i<32;++i) {
                const float ang=mul(float(pos+(mutant=="q_angle")),inv[i]);
                cached[i]={float(std::cos(double(ang))),float(std::sin(double(ang)))};
            }
            for(int h=0;h<32;++h) for(int i=0;i<32;++i) for(int odd=0;odd<2;++odd) {
                const float ang=mul(float(pos),inv[i]);
                const float c=float(std::cos(double(ang))),s=float(std::sin(double(ang)));
                const float a=float(((r+h+i)%13)-6)*0.125f;
                const float b=float(((r+2*h+i)%11)-5)*0.0625f;
                const auto ref=rotate(a,b,c,s,odd);
                auto got=rotate(a,b,cached[i][0],cached[i][1],odd);
                if(mutant=="q_pair" && odd) got[0]=ref[1];
                if(bits(ref[0])!=bits(got[0])) ++mismatches;
                ++cases;
            }
        }
        std::array<int,ROWS> klen{};
        for(int r=0;r<ROWS;++r) klen[r]=std::min(COLS,1+((r*37+sess*29)%COLS));
        std::vector<float> ref(ROWS*COLS),got(ROWS*COLS,std::numeric_limits<float>::quiet_NaN());
        const float ninf=-std::numeric_limits<float>::infinity();
        const int tiles=(COLS+127)/128;
        for(int g=0;g<ROWS;g+=4) for(int tile=0;tile<tiles;++tile) {
            ++old_q_stage;
            for(int r=g;r<std::min(g+4,ROWS);++r) for(int col=tile*128;col<std::min(COLS,(tile+1)*128);++col)
                ref[r*COLS+col]=col<klen[r]?score(sess,r,col,col):ninf;
        }
        for(int g=0;g<ROWS;g+=16) {
            ++new_q_stage;
            int maxlen=0; for(int r=g;r<std::min(g+16,ROWS);++r) maxlen=std::max(maxlen,klen[r]);
            if(mutant=="groupmax") maxlen=klen[g];
            const int useful=(maxlen+127)/128;
            std::array<int,STAGES> ring{};
            for(int tile=0;tile<std::min(STAGES,useful);++tile) ring[tile]=tile;
            for(int tile=0;tile<useful;++tile) {
                const int stage=tile%STAGES;
                const int key=ring[(stage+(mutant=="ring"))%STAGES]*128;
                for(int r=g;r<std::min(g+16,ROWS);++r) for(int col=tile*128;col<std::min(COLS,(tile+1)*128);++col)
                    got[r*COLS+col]=col<klen[r]?score(sess,r,col,key+col%128):ninf;
                if(tile+STAGES<useful) ring[stage]=tile+STAGES;
            }
            for(int r=g;r<std::min(g+16,ROWS);++r) for(int col=useful*128;col<COLS;++col) {
                got[r*COLS+col]=(mutant=="mask")?0.0f:ninf; ++skipped;
            }
        }
        for(size_t i=0;i<ref.size();++i) { if(bits(ref[i])!=bits(got[i])) ++mismatches; ++cases; }
    }
    // Source dispatch: decode has chunk_end=0; solo W-prefill may have 4096
    // rows, while ragged pack has <=8 segments of <=64 rows, never 32 in one pack.
    const int decode_rows=ns==8?48:32, solo_prefill_rows=4096, pack_segment_rows=64;
    const bool decode_qr=false, solo_qr=solo_prefill_rows>=16, pack_qr=pack_segment_rows>=16;
    if(decode_qr || !solo_qr || !pack_qr || decode_rows>48 || new_q_stage>=old_q_stage) ++mismatches;
    std::printf("CPU_%s sessions=%d mutant=%s cases=%zu mismatch=%zu skipped=%zu q_stages=%zu/%zu decode_rows=%d solo_W=%d pack_segment=%d\n",
        mismatches?"RED":"PASS",ns,mutant.c_str(),cases,mismatches,skipped,new_q_stage,old_q_stage,decode_rows,solo_prefill_rows,pack_segment_rows);
    return mismatches==0;
}
int main(int argc,char**argv) {
    const std::string mutant=argc>1?argv[1]:"none";
    const bool pass8=check(8,mutant), pass32=check(32,mutant);
    const bool pass=pass8&&pass32;
    return pass?0:1;
}
