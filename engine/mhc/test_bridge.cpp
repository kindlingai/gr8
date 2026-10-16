#include "cpu.h"
extern "C" {
void mhc_prepare(const float* r,const float* fn,const float* base,const float* scale,const float* norm,
                 int M,int H,float eps,float* pre,float* post,float* comb,float* x,float* normalized) {
    flash_mhc::prepare_cpu(r,fn,base,scale,norm,M,H,eps,pre,post,comb,x,normalized);
}
void mhc_output(float* r,const float* block,const float* post,const float* comb,int M,int H) {
    flash_mhc::output_cpu(r,block,post,comb,M,H);
}
void mhc_final(const float* r,const float* norm,int M,int H,float eps,float* out) {
    flash_mhc::final_cpu(r,norm,M,H,eps,out);
}
}
