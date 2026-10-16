// Standalone GPU per-op gate, driven by test_mhc.py. Build on any CUDA 13 host:
// nvcc -O2 -std=c++17 -arch=sm_121a -DG6_MODEL_FLASH=1 engine/mhc/test_mhc.cu -o /tmp/mhc_test
#include "mhc.cuh"
#include "cpu.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>
using namespace flash_mhc;
#define CK(x) do { auto e=(x); if(e!=cudaSuccess) { fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1); } } while(0)
template<class T> T* alloc(size_t n) { T* p; CK(cudaMalloc(&p,n*sizeof(T))); return p; }
std::vector<float> read(std::ifstream& f,size_t n) {
    std::vector<float> v(n); f.read((char*)v.data(),n*sizeof(float));
    if(!f) { fprintf(stderr,"short fixture\n"); exit(2); } return v;
}
float* upload(const std::vector<float>& v) {
    float* p=alloc<float>(v.size()); CK(cudaMemcpy(p,v.data(),v.size()*sizeof(float),cudaMemcpyHostToDevice)); return p;
}
void save(std::ofstream& f,const float* p,size_t n) {
    std::vector<float> v(n); CK(cudaMemcpy(v.data(),p,n*sizeof(float),cudaMemcpyDeviceToHost));
    f.write((const char*)v.data(),n*sizeof(float)); if(!f) exit(2);
}
void check(const char* what, const float* device, const std::vector<float>& ref, float tol=1e-4f) {
    std::vector<float> got(ref.size()); CK(cudaMemcpy(got.data(),device,got.size()*4,cudaMemcpyDeviceToHost));
    double err=0,den=0;
    for(size_t i=0;i<ref.size();++i) {
        if(!std::isfinite(got[i])) { fprintf(stderr,"nonfinite %s\n",what); exit(3); }
        err+=double(got[i]-ref[i])*(got[i]-ref[i]); den+=double(ref[i])*ref[i];
    }
    const double rel=sqrt(err/fmax(den,1e-30));
    if(rel>tol) { fprintf(stderr,"%s relative error %.9g\n",what,rel); exit(3); }
}
void auxiliary(Buffers b,Weights w,const std::vector<float>& r,const std::vector<float>& y,
               int M,int H,float eps,const std::vector<__nv_bfloat16>& norm) {
    const size_t N=(size_t)M*H;
    // Rank order and BF16 partial path, deliberately cancellation-sensitive.
    const float factors[4]={256.f,.125f,-256.f,1.f};
    std::vector<float> reduced(N,0.f);
    std::vector<__nv_bfloat16*> peers(4);
    for(int p=0;p<4;++p) {
        std::vector<__nv_bfloat16> v(N);
        for(size_t i=0;i<N;++i) { v[i]=__float2bfloat16(y[i]*factors[p]); reduced[i]+=__bfloat162float(v[i]); }
        peers[p]=alloc<__nv_bfloat16>(N); CK(cudaMemcpy(peers[p],v.data(),N*2,cudaMemcpyHostToDevice));
    }
    auto dp=alloc<const __nv_bfloat16*>(4); CK(cudaMemcpy(dp,peers.data(),4*sizeof(void*),cudaMemcpyHostToDevice));
    std::vector<float> post(M*4),comb(M*16),expected=r;
    CK(cudaMemcpy(post.data(),b.post,post.size()*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(comb.data(),b.comb,comb.size()*4,cudaMemcpyDeviceToHost));
    output_cpu(expected.data(),reduced.data(),post.data(),comb.data(),M,H);
    CK(cudaMemcpy(b.streams,r.data(),r.size()*4,cudaMemcpyHostToDevice));
    k_mhc_out_inv<<<M,THREADS>>>(b,Bf16Partials{dp,4},H,eps);
    check("rank-ordered BF16 consume",b.streams,expected);
    std::vector<float> inv(M);
    CK(cudaMemcpy(inv.data(),b.inv,M*4,cudaMemcpyDeviceToHost));
    k_mhc_flat_inv<<<M,THREADS>>>(b,H,eps);
    check("consume flat inverse",b.inv,inv);
    // Embedding replication and flat norm start state.
    int* tok=alloc<int>(M); CK(cudaMemset(tok,0,M*4));
    k_mhc_embed<<<M,THREADS>>>(b,tok,w.norm,H,eps);
    for(int m=0;m<M;++m) for(int j=0;j<4;++j) for(int h=0;h<H;++h)
        expected[((size_t)m*4+j)*H+h]=__bfloat162float(norm[h]);
    check("embedding",b.streams,expected,0.f);
    CK(cudaMemcpy(inv.data(),b.inv,M*4,cudaMemcpyDeviceToHost));
    k_mhc_flat_inv<<<M,THREADS>>>(b,H,eps);
    check("embedding flat inverse",b.inv,inv);
    // Direct FP32 -> both fragment types, including padded rows. Use values
    // that expose double-rounding if the BF16 path starts from FP16.
    std::vector<float> x(N);
    for(size_t i=0;i<N;++i) x[i]=1.00390625f+(int(i%7)-3)*.0001f;
    float* dx=upload(x);
    const int padded=(M+7)/8*8; const size_t words=(size_t)padded*H/2;
    auto f16=alloc<unsigned>(words),bf16=alloc<unsigned>(words);
    fragments(dx,M,H,f16,bf16,nullptr);
    std::vector<unsigned> a(words),b16(words);
    CK(cudaMemcpy(a.data(),f16,words*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b16.data(),bf16,words*4,cudaMemcpyDeviceToHost));
    for(int m=0;m<padded;++m) for(int h=0;h<H;h+=2) {
        const float u=m<M?x[(size_t)m*H+h]:0.f,v=m<M?x[(size_t)m*H+h+1]:0.f;
        const size_t o=(size_t)(m/8)*(H/16)*64+((h/16)*32+(m%8)*4+(h%8)/2)*2+(h/8)%2;
        const unsigned ea=__half_as_ushort(__float2half(u))|((unsigned)__half_as_ushort(__float2half(v))<<16);
        const unsigned eb=__bfloat16_as_ushort(__float2bfloat16(u))|((unsigned)__bfloat16_as_ushort(__float2bfloat16(v))<<16);
        if(a[o]!=ea || b16[o]!=eb) { fprintf(stderr,"fragment mismatch\n"); exit(3); }
    }
    // Fused prepare (mix + coefficients_warp + collapse_norm + fragments) bit-identity across two calls.
    b.done=alloc<unsigned>(M); CK(cudaMemset(b.done,0,M*4));
    CK(cudaMemcpy(b.streams,r.data(),r.size()*4,cudaMemcpyHostToDevice));
    k_mhc_flat_inv<<<M,THREADS>>>(b,H,eps);
    auto norm_ref=alloc<float>(N),norm_fus=alloc<float>(N);
    auto f16_ref=alloc<unsigned>(words),bf16_ref=alloc<unsigned>(words);
    k_mhc_mix<<<dim3(MIX,M),THREADS>>>(b,w,H);
    k_mhc_coefficients<<<M,32>>>(b,w);
    k_mhc_collapse_norm<<<M,THREADS,H*sizeof(float)>>>(b,w,H,eps,nullptr,nullptr,norm_ref);
    fragments(norm_ref,M,H,f16_ref,bf16_ref,nullptr);
    std::vector<float> pre_ref(4*M),post_ref(4*M),comb_ref(16*M),n_ref(N);
    std::vector<unsigned> a_ref(words),b_ref(words);
    CK(cudaMemcpy(pre_ref.data(),b.pre,4*M*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(post_ref.data(),b.post,4*M*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(comb_ref.data(),b.comb,16*M*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(n_ref.data(),norm_ref,N*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(a_ref.data(),f16_ref,words*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b_ref.data(),bf16_ref,words*4,cudaMemcpyDeviceToHost));
    for(int pass=0;pass<2;++pass) {
        CK(cudaMemset(b.pre,0,4*M*4)); CK(cudaMemset(b.post,0,4*M*4)); CK(cudaMemset(b.comb,0,16*M*4));
        CK(cudaMemset(f16,0xff,words*4)); CK(cudaMemset(bf16,0xff,words*4));
        prepare(b,w,M,H,eps,nullptr,nullptr,norm_fus,f16,bf16);
        check("fused pre",b.pre,pre_ref,0.f);
        check("fused post",b.post,post_ref,0.f);
        check("fused comb",b.comb,comb_ref,0.f);
        check("fused normalized",norm_fus,n_ref,0.f);
        CK(cudaMemcpy(a.data(),f16,words*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b16.data(),bf16,words*4,cudaMemcpyDeviceToHost));
        if(a!=a_ref || b16!=b_ref) { fprintf(stderr,"fused fragment mismatch pass %d\n",pass); exit(3); }
    }
    if (H == 4096) {
        CK(cudaMemcpy(b.streams,r.data(),r.size()*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(b.post,post.data(),4*M*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(b.comb,comb.data(),16*M*4,cudaMemcpyHostToDevice));
        k_mhc_out_inv<<<M,THREADS>>>(b,Bf16Partials{dp,4},H,eps);
        prepare(b,w,M,H,eps,nullptr,nullptr,norm_ref,f16_ref,bf16_ref);
        std::vector<float> s_ref(4*N),inv_ref(M),mix_ref(24*M);
        CK(cudaMemcpy(s_ref.data(),b.streams,4*N*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(inv_ref.data(),b.inv,M*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(mix_ref.data(),b.mix,24*M*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(pre_ref.data(),b.pre,4*M*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(post_ref.data(),b.post,4*M*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(comb_ref.data(),b.comb,16*M*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(n_ref.data(),norm_ref,N*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(a_ref.data(),f16_ref,words*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b_ref.data(),bf16_ref,words*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.streams,r.data(),r.size()*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(b.post,post.data(),4*M*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(b.comb,comb.data(),16*M*4,cudaMemcpyHostToDevice));
        CK(cudaMemset(b.inv,0,M*4)); CK(cudaMemset(b.mix,0,24*M*4)); CK(cudaMemset(b.pre,0,4*M*4));
        CK(cudaMemset(f16,0xff,words*4)); CK(cudaMemset(bf16,0xff,words*4));
        k_mhc_out_inv_prepare<<<M,THREADS>>>(b,Bf16Partials{dp,4},w,H,eps,nullptr,norm_fus,f16,bf16);
        check("out_inv_prepare streams",b.streams,s_ref,0.f);
        check("out_inv_prepare inv",b.inv,inv_ref,0.f);
        check("out_inv_prepare mix",b.mix,mix_ref,0.f);
        check("out_inv_prepare pre",b.pre,pre_ref,0.f);
        check("out_inv_prepare post",b.post,post_ref,0.f);
        check("out_inv_prepare comb",b.comb,comb_ref,0.f);
        check("out_inv_prepare normalized",norm_fus,n_ref,0.f);
        CK(cudaMemcpy(a.data(),f16,words*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b16.data(),bf16,words*4,cudaMemcpyDeviceToHost));
        if(a!=a_ref || b16!=b_ref) { fprintf(stderr,"out_inv_prepare fragment mismatch\n"); exit(3); }
    }
    CK(cudaFree(b.done)); b.done=nullptr;
    CK(cudaFree(norm_ref)); CK(cudaFree(norm_fus)); CK(cudaFree(f16_ref)); CK(cudaFree(bf16_ref));
    CK(cudaFree(dx)); CK(cudaFree(f16)); CK(cudaFree(bf16)); CK(cudaFree(tok)); CK(cudaFree(dp));
    for(auto p:peers) CK(cudaFree(p));
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
}
int main(int argc,char** argv) {
    if(argc!=3) return 2;
    std::ifstream f(argv[1],std::ios::binary); int M,H; float eps;
    f.read((char*)&M,4); f.read((char*)&H,4); f.read((char*)&eps,4);
    if(!f || M<1 || M>32 || H<16 || H%16 || H>4096 || !(eps>0)) return 2;
    const size_t N=(size_t)M*H;
    auto r=read(f,4*N),fn=read(f,24*4*H),base=read(f,24),scale=read(f,3),norm=read(f,H),y=read(f,N);
    if(f.peek()!=std::char_traits<char>::eof()) return 2;
    Buffers b{upload(r),alloc<float>(M),alloc<float>(24*M),alloc<float>(4*M),alloc<float>(4*M),alloc<float>(16*M)};
    std::vector<__nv_bfloat16> wn(H); for(int h=0;h<H;++h) wn[h]=__float2bfloat16(norm[h]);
    auto dn=alloc<__nv_bfloat16>(H); CK(cudaMemcpy(dn,wn.data(),H*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice));
    std::vector<__nv_bfloat16> wf(fn.size());
    for(size_t i=0;i<fn.size();++i) wf[i]=__float2bfloat16(fn[i]);
    auto df=alloc<__nv_bfloat16>(fn.size());
    CK(cudaMemcpy(df,wf.data(),wf.size()*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice));
    Weights w{df,upload(base),upload(scale),dn};
    float *raw=alloc<float>(N),*normalized=alloc<float>(N),*final=alloc<float>(N),*dy=upload(y);
    k_mhc_flat_inv<<<M,THREADS>>>(b,H,eps);
    k_mhc_mix<<<dim3(MIX,M),THREADS>>>(b,w,H);
    k_mhc_coefficients<<<M,32>>>(b,w);
    k_mhc_collapse_norm<<<M,THREADS,H*sizeof(float)>>>(b,w,H,eps,nullptr,raw,normalized);
    k_mhc_out_inv<<<M,THREADS>>>(b,Fp32Block{dy},H,eps);
    k_mhc_final<<<M,THREADS,H*sizeof(float)>>>(b,dn,H,eps,nullptr,final);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::ofstream out(argv[2],std::ios::binary);
    save(out,b.pre,4*M); save(out,b.post,4*M); save(out,b.comb,16*M);
    save(out,raw,N); save(out,normalized,N); save(out,b.streams,4*N); save(out,final,N);
    auxiliary(b,w,r,y,M,H,eps,wn);
    CK(cudaFree(b.streams)); CK(cudaFree(b.inv)); CK(cudaFree(b.mix)); CK(cudaFree(b.pre)); CK(cudaFree(b.post)); CK(cudaFree(b.comb));
    CK(cudaFree((void*)w.fn)); CK(cudaFree((void*)w.base)); CK(cudaFree((void*)w.scale)); CK(cudaFree(dn));
    CK(cudaFree(raw)); CK(cudaFree(normalized)); CK(cudaFree(final)); CK(cudaFree(dy));
    puts("PASS: GPU mHC fixture executed");
}
