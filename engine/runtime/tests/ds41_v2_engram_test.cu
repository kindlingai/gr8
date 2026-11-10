// Synthetic v2 Engram seam test. No model, NVMe or physical fabric is used.
// Actual stock dequant + rank-order sum, activation quantizer, WKV TC GEMM and
// gate kernels are the references. D2D copies model the source-major A2A.
// --old-owned-dispatch reproduces the removed per-owner TC/GEMV switch (fails
// the synthetic tolerance at M33). Default requires bit-exact dispatch results.
// nvcc -O2 -std=c++17 -gencode=arch=compute_120a,code=sm_120a -DG6_MODEL_DS41=1 \
//   engine/runtime/tests/ds41_v2_engram_test.cu engine/attn/ds41_attn.cu \
//   engine/knobs-rs/target/release/libgr8knobs.a -lpthread -ldl -lm -o ds41_v2_engram_test
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include "../../engram/engram_dev.cuh"
#include "../ds41_v2.h"
#include "../ds41_v2_engram.cuh"
#include "../../attn/ds41_attn_gv.cuh"
extern "C" void ds41_attn_set_gv(int,int,int,ds41a::XfragFn,ds41a::GemvFn);
extern "C" int ds41_attn_project(const ds41a::Fp8Mat*,const void*,int,void*,int,uint8_t*,uint8_t*,uint32_t*,void*);
namespace eg=ds41::engram;
namespace ew=ds41::v2::engram_wire;
using bf16=__nv_bfloat16;
#define CK(x) do { auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s: %s\n",__FILE__,__LINE__,#x,cudaGetErrorString(e));exit(2);} } while(0)
#define CHECK(x) do { if(!(x)){fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);exit(1);} } while(0)
constexpr int NR=4, PART=67, E=eg::kEmbDim, K=eg::kKvDim, T=eg::kStreams*eg::kHid;
constexpr int64_t FIRST=0x80000010LL;
__host__ __device__ uint32_t hash32(uint32_t x){x^=x>>16;x*=0x7feb352d;x^=x>>15;x*=0x846ca68b;return x^(x>>16);}
__host__ __device__ int valid(int p,int mode){return p==3?(mode==2?0:PART-7):PART;}
__host__ __device__ uint8_t code(int row,int cell,int col,int p){const auto h=hash32(row*11939u+cell*256+col+p*173);return uint8_t(h%0x70|(h&0x80));}
__host__ __device__ uint8_t scale(int p,int loc,int b){return uint8_t(119+hash32(p*977u+loc*8+b)%12);}
__host__ __device__ uint32_t id(int r,int c,int mode){
    if(mode==1)return uint32_t(FIRST+2*PART+hash32(r*24u+c)%PART); // every cell on one source
    const int x=(r*24+c)%12;
    if(x<4)return uint32_t(FIRST+x*PART); // exact lower boundaries
    if(x<8)return uint32_t(FIRST+(x-3)*PART-1); // exact upper boundaries
    if(x==8)return uint32_t(FIRST-1); // before all partitions
    if(x==9)return uint32_t(FIRST+4*PART); // after all partitions
    return uint32_t(FIRST+hash32(r*24u+c)%(4*PART));
}
__global__ void init_ids(uint32_t* out,int M,int mode){for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<M*24;i+=gridDim.x*blockDim.x)out[i]=id(i/24,i%24,mode);}
__global__ void init_stage(uint8_t* out,int M,int p){for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size_t(M)*E;i+=size_t(gridDim.x)*blockDim.x)out[i]=code(i/E,(i%E)/256,i%256,p);}
__global__ void init_scales(uint8_t* out,int p){int i=threadIdx.x+blockIdx.x*blockDim.x;if(i<PART*8)out[i]=scale(p,i/8,i%8);}
__global__ void verify_wire(const uint8_t* out,int M,int p,int mode,unsigned* bad){
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size_t(M)*ew::ROW_BYTES;i+=size_t(gridDim.x)*blockDim.x){
        const int r=i/ew::ROW_BYTES,b=i%ew::ROW_BYTES;
        uint8_t want=0;
        if(b<ew::HEADER){unsigned mask=0;for(int c=0;c<24;++c){int64_t loc=int64_t(id(r,c,mode))-(FIRST+p*PART);if(loc>=0&&loc<valid(p,mode))mask|=1u<<c;}
            const uint32_t h=b<4?__popc(mask):b<8?mask:0;want=uint8_t(h>>((b%4)*8));
        }else{const int cell=(b-ew::HEADER)/264,j=(b-ew::HEADER)%264;const int64_t loc=int64_t(id(r,cell,mode))-(FIRST+p*PART);
            if(loc>=0&&loc<valid(p,mode))want=j<256?code(r,cell,j,p):scale(p,loc,j-256);
        }
        if(out[i]!=want)atomicAdd(bad,1u);
    }
}
__global__ void rank_sum(const bf16* partial,bf16* out,int M){for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size_t(M)*E;i+=size_t(gridDim.x)*blockDim.x){float s=0;for(int p=0;p<NR;++p)s+=__bfloat162float(partial[size_t(p)*M*E+i]);out[i]=__float2bfloat16_rn(s);}}
__global__ void compare(const bf16* a,const bf16* b,size_t n,unsigned* bad){for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)if(__bfloat16_as_ushort(a[i])!=__bfloat16_as_ushort(b[i]))atomicAdd(bad,1u);}
__global__ void guards(const uint8_t* p,size_t n,unsigned* bad){int i=threadIdx.x;if(i<256&&(p[i]!=0xa5||p[256+n+i]!=0xa5))atomicAdd(bad,1u);}
__global__ void padding(const bf16* p,size_t n,unsigned* bad){for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)if(__bfloat16_as_ushort(p[i])!=0xa5a5)atomicAdd(bad,1u);}
template<class U>struct Dev{
    U* p;uint8_t* base;size_t n;
    explicit Dev(size_t nn):n(nn){CK(cudaMalloc(&base,512+n*sizeof(U)));p=reinterpret_cast<U*>(base+256);poison();}
    Dev(const Dev&)=delete;Dev& operator=(const Dev&)=delete;
    ~Dev(){cudaFree(base);}
    void poison(){CK(cudaMemset(base,0xa5,512+n*sizeof(U)));}
    void check(unsigned* bad){guards<<<1,256>>>(base,n*sizeof(U),bad);}
};
void clean(unsigned* bad){CK(cudaMemset(bad,0,sizeof(unsigned)));}
void good(unsigned* bad){CK(cudaGetLastError());unsigned n=0;CK(cudaMemcpy(&n,bad,4,cudaMemcpyDeviceToHost));if(n){fprintf(stderr,"mismatches=%u\n",n);exit(1);}}
struct WireCase{
    int M,Q,mode;Dev<uint32_t> ids;Dev<uint8_t> stage,scales,send,recv;Dev<bf16> partial,dense,own;Dev<unsigned> bad;
    WireCase(int m,int md):M(m),Q((m+3)/4),mode(md),ids(size_t(m)*24),stage(size_t(m)*E),scales(PART*8),send(size_t(NR)*m*ew::ROW_BYTES),recv(size_t(NR)*Q*ew::ROW_BYTES),partial(size_t(NR)*m*E),dense(size_t(m)*E),own(size_t(Q)*E),bad(1){
        clean(bad.p);init_ids<<<64,256>>>(ids.p,M,mode);
        for(int p=0;p<NR;++p){init_stage<<<256,256>>>(stage.p,M,p);init_scales<<<3,256>>>(scales.p,p);
            ew::pack<<<M,256>>>(ids.p,stage.p,scales.p,FIRST+p*PART,valid(p,mode),M,send.p+size_t(p)*M*ew::ROW_BYTES);
            verify_wire<<<256,256>>>(send.p+size_t(p)*M*ew::ROW_BYTES,M,p,mode,bad.p);
            eg::k_dequant<<<dim3(M,3),256>>>(ids.p,stage.p,scales.p,FIRST+p*PART,valid(p,mode),M,partial.p+size_t(p)*M*E);
        }
        rank_sum<<<256,256>>>(partial.p,dense.p,M);
        for(int dst=0;dst<NR;++dst)assemble(dst);
        ids.check(bad.p);stage.check(bad.p);scales.check(bad.p);send.check(bad.p);recv.check(bad.p);partial.check(bad.p);dense.check(bad.p);own.check(bad.p);good(bad.p);
        printf("PASS wire M=%d mode=%d: header/count/mask, FP8/scales, foreign-zero, source-major A2A model, stock rank-sum exact, guards\n",M,mode);
    }
    void assemble(int dst){auto g=ds41::v2::geometry(M,NR,dst);recv.poison();own.poison();
        for(int p=0;p<NR;++p)if(g.owned)CK(cudaMemcpyAsync(recv.p+size_t(p)*Q*ew::ROW_BYTES,send.p+(size_t(p)*M+g.row0)*ew::ROW_BYTES,size_t(g.owned)*ew::ROW_BYTES,cudaMemcpyDeviceToDevice));
        if(g.owned){ew::unpack<<<dim3(g.owned,24),256>>>(recv.p,Q,g.owned,NR,own.p);compare<<<256,256>>>(own.p,dense.p+size_t(g.row0)*E,size_t(g.owned)*E,bad.p);}
        padding<<<64,256>>>(own.p+size_t(g.owned)*E,size_t(Q-g.owned)*E,bad.p);
    }
};
__global__ void init_weights(uint8_t* w,uint8_t* s){for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size_t(K)*E;i+=size_t(gridDim.x)*blockDim.x){uint32_t h=hash32(i+113);w[i]=uint8_t(h%0x60|(h&0x80));}for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<K*(E/32);i+=gridDim.x*blockDim.x)s[i]=uint8_t(116+hash32(i)%4);}
__global__ void init_trunk(bf16* bf,float* f,int M,float* wg,uint8_t* dead){for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size_t(M)*T;i+=size_t(gridDim.x)*blockDim.x){bf16 x=__float2bfloat16_rn(float(int(hash32(i+229)%1023)-511)/256);bf[i]=x;f[i]=__bfloat162float(x);}for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<T;i+=gridDim.x*blockDim.x)wg[i]=float(int(hash32(i+773)%255)-127)/128;for(int r=blockIdx.x*blockDim.x+threadIdx.x;r<M;r+=gridDim.x*blockDim.x)dead[r]=(r%7==1||r==M-1);}
__global__ void compare_gate(const bf16* got,const float* gotf,const bf16* ref,const float* reff,const bf16* input,size_t n,int row0,const uint8_t* dead,unsigned* bad){
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x){const size_t j=size_t(row0)*T+i;const float v=__bfloat162float(got[i]);
        if(__bfloat16_as_ushort(got[i])!=__bfloat16_as_ushort(ref[j])||gotf[i]!=reff[j]||v!=gotf[i])atomicAdd(bad,1u);
        if(dead&&dead[j/T]&&__bfloat16_as_ushort(got[i])!=__bfloat16_as_ushort(input[j]))atomicAdd(bad,1u);
    }
}
void project(const bf16* x,int M,const uint8_t* w,const uint8_t* sw,uint8_t* a,uint8_t* sa,bf16* out){
    eg::k_act_quant_rows<<<(M*(E/32)+7)/8,256>>>(x,M,a,sa);
    CHECK(ds41_gemm_mx_tc(a,E,sa,E/32,w,sw,out,K,M,K,E,nullptr)==0);
}
void projection_case(int M,const Dev<uint8_t>& w,const Dev<uint8_t>& sw){
    WireCase c(M,0);Dev<uint8_t> aq(size_t(M)*E),as(size_t(M)*(E/32)),dead(M);Dev<bf16> kv(size_t(M)*K),ownkv(size_t(c.Q)*K),input(size_t(M)*T),trunk(size_t(M)*T),owntrunk(size_t(c.Q)*T);Dev<float> wg(T),inputf(size_t(M)*T),trunkf(size_t(M)*T),ownf(size_t(c.Q)*T);
    project(c.dense.p,M,w.p,sw.p,aq.p,as.p,kv.p);
    init_trunk<<<256,256>>>(input.p,inputf.p,M,wg.p,dead.p);
    for(bool images:{false,true}){
        CK(cudaMemcpyAsync(trunk.p,input.p,size_t(M)*T*2,cudaMemcpyDeviceToDevice));CK(cudaMemcpyAsync(trunkf.p,inputf.p,size_t(M)*T*4,cudaMemcpyDeviceToDevice));
        if(images){eg::k_gate_dead<<<dim3(M,4),512>>>(trunk.p,kv.p,wg.p,M,dead.p);eg::k_gate_dead<<<dim3(M,4),512>>>(trunkf.p,kv.p,wg.p,M,dead.p);}
        else {eg::k_gate<<<dim3(M,4),512>>>(trunk.p,kv.p,wg.p,M);eg::k_gate<<<dim3(M,4),512>>>(trunkf.p,kv.p,wg.p,M);}
        for(int dst=0;dst<NR;++dst){auto g=ds41::v2::geometry(M,NR,dst);c.assemble(dst);if(!g.owned)continue;
            project(c.own.p,g.owned,w.p,sw.p,aq.p,as.p,ownkv.p);compare<<<256,256>>>(ownkv.p,kv.p+size_t(g.row0)*K,size_t(g.owned)*K,c.bad.p);
            CK(cudaMemcpyAsync(owntrunk.p,input.p+size_t(g.row0)*T,size_t(g.owned)*T*2,cudaMemcpyDeviceToDevice));CK(cudaMemcpyAsync(ownf.p,inputf.p+size_t(g.row0)*T,size_t(g.owned)*T*4,cudaMemcpyDeviceToDevice));
            if(images){eg::k_gate_dead<<<dim3(g.owned,4),512>>>(owntrunk.p,ownkv.p,wg.p,g.owned,dead.p+g.row0);eg::k_gate_dead<<<dim3(g.owned,4),512>>>(ownf.p,ownkv.p,wg.p,g.owned,dead.p+g.row0);}
            else {eg::k_gate<<<dim3(g.owned,4),512>>>(owntrunk.p,ownkv.p,wg.p,g.owned);eg::k_gate<<<dim3(g.owned,4),512>>>(ownf.p,ownkv.p,wg.p,g.owned);}
            compare_gate<<<256,256>>>(owntrunk.p,ownf.p,trunk.p,trunkf.p,input.p,size_t(g.owned)*T,g.row0,images?dead.p:nullptr,c.bad.p);
        }
    }
    aq.check(c.bad.p);as.check(c.bad.p);kv.check(c.bad.p);ownkv.check(c.bad.p);input.check(c.bad.p);trunk.check(c.bad.p);owntrunk.check(c.bad.p);inputf.check(c.bad.p);trunkf.check(c.bad.p);ownf.check(c.bad.p);wg.check(c.bad.p);dead.check(c.bad.p);good(c.bad.p);
    printf("PASS projection M=%d: actual WKV [25600][6144] TC full-vs-owned exact; stock gate float/bf16 text/image exact (not scalar hook dispatch)\n",M);
}
// Same MXFP8 record layout as stock tests/ds41_attn_test.cu, from gr8.
__global__ void make_gv_records(const uint8_t* codes,const uint8_t* scales,uint8_t* out){
    const int ng=E/32,t=blockIdx.x/ng,g=blockIdx.x%ng,i=threadIdx.x;
    uint8_t* rec=out+size_t(blockIdx.x)*528;
    const int l=i>>3,e=i&7,r=l>>2,c=l&3;
    const int row=r+((e&2)?8:0),col=2*c+(e&1)+((e&4)?8:0);
    for(int h=0;h<2;++h)rec[h*256+i]=codes[size_t(16*t+row)*E+32*g+16*h+col];
    if(i<8){rec[512+2*i]=scales[size_t(16*t+i)*ng+g];rec[513+2*i]=scales[size_t(16*t+i+8)*ng+g];}
}
// Device comparison statistic: changed values, max absolute error. A local
// 0.008*(1+abs(ref)) synthetic seam bound is NOT a model quality criterion.
__global__ void compare_tolerant(const bf16* got,const bf16* ref,size_t n,unsigned* changed,float* maxerr,unsigned* bad){
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x){float a=__bfloat162float(got[i]),b=__bfloat162float(ref[i]),d=fabsf(a-b);
        if(d)atomicAdd(changed,1u);atomicMax(reinterpret_cast<int*>(maxerr),__float_as_int(d));
        if(!isfinite(a)||!isfinite(b)||d>.008f*(1+fabsf(b))){unsigned nbad=atomicAdd(bad,1u);if(nbad<8)printf("TOL_FAIL i=%llu got=%g ref=%g diff=%g\n",(unsigned long long)i,a,b,d);}
    }
}
void dispatch_case(int M,const Dev<uint8_t>& w,const Dev<uint8_t>& sw,const Dev<uint8_t>& gv,bool old_dispatch){
    WireCase c(M,0);Dev<uint8_t> aq(size_t(M)*E),as(size_t(M)*(E/32)),dead(M);Dev<uint32_t> xf(E*4);Dev<bf16> kv(size_t(M)*K),ownkv(size_t(c.Q)*K),input(size_t(M)*T),trunk(size_t(M)*T),owntrunk(size_t(c.Q)*T);Dev<float> wg(T),inputf(size_t(M)*T),maxerr(2);Dev<unsigned> changed(2);
    CK(cudaMemset(changed.p,0,8));CK(cudaMemset(maxerr.p,0,8));
    const ds41a::Fp8Mat W{w.p,sw.p,K,E,gv.p};
    auto dispatch=[&](const bf16* x,int rows,bf16* out){if((old_dispatch?rows:M)>eg::kDecodeRows)project(x,rows,w.p,sw.p,aq.p,as.p,out);else CHECK(ds41_attn_project(&W,x,rows,out,K,aq.p,as.p,xf.p,nullptr)==0);};
    dispatch(c.dense.p,M,kv.p);init_trunk<<<256,256>>>(input.p,inputf.p,M,wg.p,dead.p);
    CK(cudaMemcpyAsync(trunk.p,input.p,size_t(M)*T*2,cudaMemcpyDeviceToDevice));eg::k_gate_dead<<<dim3(M,4),512>>>(trunk.p,kv.p,wg.p,M,dead.p);
    for(int dst=0;dst<NR;++dst){auto g=ds41::v2::geometry(M,NR,dst);c.assemble(dst);if(!g.owned)continue;
        dispatch(c.own.p,g.owned,ownkv.p);if(!old_dispatch)compare<<<256,256>>>(ownkv.p,kv.p+size_t(g.row0)*K,size_t(g.owned)*K,c.bad.p);compare_tolerant<<<256,256>>>(ownkv.p,kv.p+size_t(g.row0)*K,size_t(g.owned)*K,changed.p,maxerr.p,c.bad.p);
        CK(cudaMemcpyAsync(owntrunk.p,input.p+size_t(g.row0)*T,size_t(g.owned)*T*2,cudaMemcpyDeviceToDevice));eg::k_gate_dead<<<dim3(g.owned,4),512>>>(owntrunk.p,ownkv.p,wg.p,g.owned,dead.p+g.row0);
        if(!old_dispatch)compare<<<256,256>>>(owntrunk.p,trunk.p+size_t(g.row0)*T,size_t(g.owned)*T,c.bad.p);
        compare_tolerant<<<256,256>>>(owntrunk.p,trunk.p+size_t(g.row0)*T,size_t(g.owned)*T,changed.p+1,maxerr.p+1,c.bad.p);
    }
    unsigned hc[2];float he[2];CK(cudaMemcpy(hc,changed.p,8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(he,maxerr.p,8,cudaMemcpyDeviceToHost));
    printf("CHECK dispatch M=%d old_owned_threshold=%d: actual TC/small-row-GEMV; kv changed=%u max_abs=%g; gate changed=%u max_abs=%g (%s, not model EVAL)\n",M,int(old_dispatch),hc[0],he[0],hc[1],he[1],old_dispatch?"old-dispatch synthetic tolerance":"bit-exact required");fflush(stdout);good(c.bad.p);
}
// Compile both op_owned instantiations and verify the uninitialized error path.
// A live Engram object needs production-compatible hash sidecars; not mocked.
void uninitialized(){eg::Engram e;eg::EngramHooks h;h.project=[](void*,int,int,void*,const bf16*,bf16*){return 0;};e.set_hooks(h);bool called=false;auto exchange=[&](const uint32_t*,const uint8_t*,const uint8_t*,int64_t,int64_t,bf16*){called=true;return 0;};CHECK(e.op_owned(nullptr,0,1,0,1,(float*)nullptr,nullptr,exchange)==cudaErrorInvalidValue);CHECK(e.op_owned(nullptr,0,1,0,1,(bf16*)nullptr,nullptr,exchange)==cudaErrorInvalidValue);CHECK(!called);}
int main(int argc,char** argv){
    const bool old_dispatch=argc==2&&std::strcmp(argv[1],"--old-owned-dispatch")==0;
    const bool quick=old_dispatch||(argc==2&&std::strcmp(argv[1],"--quick")==0);uninitialized();
    for(int m:{1,2,5,31,32,33,71,127,128,129,256})for(int mode=0;mode<3;++mode){WireCase c(m,mode);}
    if(!quick)for(int m:{4095,4096,8191,8192})for(int mode=0;mode<3;++mode){WireCase c(m,mode);}
    Dev<uint8_t> w(size_t(K)*E),sw(size_t(K)*(E/32));init_weights<<<256,256>>>(w.p,sw.p);
    for(int m:{1,5,33,71,128,129,257})projection_case(m,w,sw);
    Dev<uint8_t> gv(size_t(K/16)*(E/32)*528);make_gv_records<<<(K/16)*(E/32),256>>>(w.p,sw.p,gv.p);
    ds41_attn_set_gv(1,0,0,ds41a::gv_xfrag_launch,ds41a::gv_gemv_launch);
    for(int m:{1,5,33,71,128,129,257})dispatch_case(m,w,sw,gv,old_dispatch);
    CK(cudaDeviceSynchronize());puts("DS41_V2_ENGRAM_PASS (synthetic kernels/seams; no NVMe/fabric/model/EVAL/speed claim)");
}
