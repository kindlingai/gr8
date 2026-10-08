// Head-group unit test for the sparse prefill attention kernels gp::sparse_qk and gp::sparse_pv (prefill.cuh).
// The grid z dimension = ceil(heads/16); block z handles heads [16z, 16z+16). TP3 ranks hold 22 or 21 heads,
// so heads 16..21 live in z = 1 (before 811f6e6 only the first 16 heads were computed). One GPU, random data,
// with invalid indices (-1, > position, >= context) to exercise masking. Every head is compared against a CPU
// reference (fp16 inputs, fp32/double accumulate); outputs start as NaN so an unwritten head fails.
//   build (from engine/attn/tests): nvcc -O2 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -DG6_NR=3 $FLAGS \
//          -o headgroup_test headgroup_test.cu ../bind.cu -lcublas
//        ($FLAGS = the K1H2_FLAGS of scripts/build.sh; bind.cu supplies g6_attn_bind_prepare for prefill.cuh)
//   run: headgroup_test            prints max error per head count (and for heads 16..21), PASS/FAIL; exit 1 on FAIL
#include "../attn.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#define CK(x) do{cudaError_t e_=(x);if(e_!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_));exit(1);}}while(0)

static bool valid(int id,int pos,int ctx){return id>=0&&id<=pos&&id<ctx;}

// One case: heads H, n rows starting at row `start` of a batch with rows = start+n, tile R >= n.
static bool run_case(int H,int n,int R,int start,int ctx,std::mt19937& rng){
    const int TOP=ga::TOP,rows=start+n;
    std::uniform_real_distribution<float> U(-1.f,1.f),P(0.f,1.f);
    // inputs
    std::vector<int> pos(rows),ind((size_t)rows*TOP);
    for(int r=0;r<rows;r++)pos[r]=ctx-rows+r-7;           // a few keys beyond every position (masked by id > position)
    int ninv=0;
    for(int r=0;r<rows;r++)for(int j=0;j<TOP;j++){unsigned u=rng()%100;int v;
        if(u<8)v=-1;
        else if(u<13)v=pos[r]+1+(int)(rng()%(unsigned)(ctx-pos[r]-1));   // > position, < context
        else if(u<15)v=ctx+(int)(rng()%50);                               // >= context
        else v=(int)(rng()%(unsigned)(pos[r]+1));
        if(!valid(v,pos[r],ctx))ninv++;ind[(size_t)r*TOP+j]=v;}
    std::vector<__half> cache((size_t)ctx*576),q((size_t)H*R*576);
    for(auto& v:cache)v=__float2half(U(rng));
    for(auto& v:q)v=__float2half(U(rng));
    std::vector<float> prob((size_t)R*H*TOP);                 // sparse_pv input [(row*H+h)*TOP+j]; finite even where masked
    for(auto& v:prob)v=P(rng)*(1.f/64);
    // device
    int *dpos,*dind;__half *dcache,*dq;float *dprob,*dscore,*dctx;
    CK(cudaMalloc(&dpos,rows*4));CK(cudaMalloc(&dind,ind.size()*4));
    CK(cudaMalloc(&dcache,cache.size()*2));CK(cudaMalloc(&dq,q.size()*2));
    CK(cudaMalloc(&dprob,prob.size()*4));CK(cudaMalloc(&dscore,(size_t)R*H*TOP*4));CK(cudaMalloc(&dctx,(size_t)H*R*512*4));
    CK(cudaMemcpy(dpos,pos.data(),rows*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dind,ind.data(),ind.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dcache,cache.data(),cache.size()*2,cudaMemcpyHostToDevice));CK(cudaMemcpy(dq,q.data(),q.size()*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dprob,prob.data(),prob.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemset(dscore,0xff,(size_t)R*H*TOP*4));CK(cudaMemset(dctx,0xff,(size_t)H*R*512*4));   // NaN sentinel
    G6Attn a;memset(&a,0,sizeof a);
    a.rows=rows;a.heads=H;a.context=ctx;a.query_tile=R;a.positions=dpos;a.indices=dind;
    // same launch shapes as the engine (prefill.cuh trace loop)
    gp::sparse_qk<<<dim3(ga::TOP/64,n,(a.heads+15)/16),128>>>(a,start,R,dq,dcache,dscore);CK(cudaGetLastError());
    gp::sparse_pv<<<dim3(8,n,(a.heads+15)/16),128>>>(a,start,R,dprob,dcache,dctx);CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<float> score((size_t)R*H*TOP),ctxo((size_t)H*R*512);
    CK(cudaMemcpy(score.data(),dscore,score.size()*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(ctxo.data(),dctx,ctxo.size()*4,cudaMemcpyDeviceToHost));
    // reference + compare
    std::vector<double> qkerr(H,0),pverr(H,0);std::vector<long> qkbad(H,0),pvbad(H,0);
    std::vector<float> cf(cache.size()),qf(q.size());
    for(size_t i=0;i<cache.size();i++)cf[i]=__half2float(cache[i]);for(size_t i=0;i<q.size();i++)qf[i]=__half2float(q[i]);
    for(int row=0;row<n;row++){int r=start+row;const int* id=&ind[(size_t)r*TOP];
        for(int h=0;h<H;h++){const float* qq=&qf[((size_t)h*R+row)*576];
            // sparse_qk
            for(int j=0;j<TOP;j++){float got=score[((size_t)row*H+h)*TOP+j];
                if(!valid(id[j],pos[r],ctx)){if(!(std::isinf(got)&&got<0))qkbad[h]++;continue;}
                const float* k=&cf[(size_t)id[j]*576];double s=0,m=0;for(int d=0;d<576;d++){double t=(double)qq[d]*k[d];s+=t;m+=fabs(t);}
                double ref=s/16,mag=m/16,err=fabs(got-ref);                 // got NaN -> comparison fails below
                if(!(err<=1e-3*mag+1e-4))qkbad[h]++;
                if(std::isfinite(got))qkerr[h]=std::max(qkerr[h],err/(mag+1e-6));else qkerr[h]=INFINITY;}
            // sparse_pv
            std::vector<double> acc(512,0),mag(512,0);const float* pp=&prob[((size_t)row*H+h)*TOP];
            for(int j=0;j<TOP;j++){if(!valid(id[j],pos[r],ctx))continue;float ph=__half2float(__float2half_rn(pp[j]));const float* v=&cf[(size_t)id[j]*576];
                for(int d=0;d<512;d++){double t=(double)ph*v[d];acc[d]+=t;mag[d]+=fabs(t);}}
            for(int d=0;d<512;d++){float got=ctxo[((size_t)h*R+row)*512+d];double err=fabs(got-acc[d]);
                if(!(err<=1e-3*mag[d]+1e-4))pvbad[h]++;
                if(std::isfinite(got))pverr[h]=std::max(pverr[h],err/(mag[d]+1e-6));else pverr[h]=INFINITY;}
        }}
    double mq=0,mp=0,mq16=0,mp16=0;long bq=0,bp=0;bool ok=true;
    for(int h=0;h<H;h++){mq=std::max(mq,qkerr[h]);mp=std::max(mp,pverr[h]);bq+=qkbad[h];bp+=pvbad[h];
        if(h>=16){mq16=std::max(mq16,qkerr[h]);mp16=std::max(mp16,pverr[h]);}
        if(qkbad[h]||pvbad[h]){ok=false;printf("  FAIL head %2d: sparse_qk bad %ld (max rel err %.3g), sparse_pv bad %ld (max rel err %.3g)\n",h,qkbad[h],qkerr[h],pvbad[h],pverr[h]);}}
    printf("heads=%2d n=%d R=%d start=%d ctx=%d invalid=%d/%d  sparse_qk max rel err %.3g  sparse_pv max rel err %.3g",
        H,n,R,start,ctx,ninv,rows*TOP,mq,mp);
    if(H>16)printf("  [heads 16..%d: qk %.3g pv %.3g]",H-1,mq16,mp16);
    printf("  %s\n",ok?"PASS":"FAIL");
    cudaFree(dpos);cudaFree(dind);cudaFree(dcache);cudaFree(dq);cudaFree(dprob);cudaFree(dscore);cudaFree(dctx);
    return ok;
}

int main(){
    std::mt19937 rng(811);
    bool ok=true;
    const int heads[]={22,21,16,8};
    for(int H:heads){ok&=run_case(H,3,4,2,700,rng);ok&=run_case(H,1,1,0,300,rng);}
    printf("%s\n",ok?"ALL PASS":"FAILED");
    return ok?0:1;
}
