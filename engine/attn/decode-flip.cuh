// G6_ATTN_FLIP (review tp3-attn-flip.md): head-inner ("flip") variants of scores_direct / context_parts_cpb.
// One KV record load serves HG heads; per head the FMA chain, slot order and p==0 skip are unchanged, so every
// output byte equals scores_direct / mla_scores and context_parts_cpb / context_parts (proved by idxbench FLIPCHK).
// Head-inner ("flip") variants of small_decode::scores_direct / context_parts_cpb (ATTENTION.md #1/#2).
// Each warp/thread loads its KV record ONCE and runs the identical per-head FMA chain for HG heads.
namespace small_decode {
template<class Input,int HG>
__global__ void scores_flip(Input input,int start,const float* q,const float* rope,float* out){
 auto a=ga::resolve(input);if(!a.active)return;constexpr int WARPS=8;int row=blockIdx.z,r=start+row,h0=blockIdx.y*HG,j=blockIdx.x*WARPS+threadIdx.x/32,lane=threadIdx.x&31;FOR_ROW(a,input,r);
 int id=a.indices[(int64_t)r*ga::TOP+j];const uint8_t* v=(id>=0&&id<=a.positions[r]&&CP_MINE(a,id))?ga::kv_at(a,id):nullptr;
 float scale[4]={0,0,0,0},lat[16];float rp[2];
 if(v){
 #pragma unroll
 for(int g=0;g<4;g++)scale[g]=((const float*)(v+512))[g];
 #pragma unroll
 for(int k=0;k<16;k++)lat[k]=ga::e4(v[lane+k*32])*scale[k/4];
 #pragma unroll
 for(int k=0;k<2;k++)rp[k]=__bfloat162float(((const bf16*)(v+528))[lane+k*32]);}
 for(int hh=0;hh<HG;hh++){int h=h0+hh;if(h>=a.heads)break;float sum=0;
  if(v){
  #pragma unroll
  for(int k=0;k<16;k++){int d=lane+k*32;sum=fmaf(__ldg(q+((int64_t)row*a.heads+h)*512+d),lat[k],sum);}
  #pragma unroll
  for(int k=0;k<2;k++){int d=lane+k*32;sum=fmaf(__ldg(rope+((int64_t)row*a.heads+h)*64+d),rp[k],sum);}}
  sum=ga::warp_sum(sum);if(lane==0)out[((int64_t)row*a.heads+h)*ga::TOP+j]=v?sum*(1.f/16):-INFINITY;}
}
// GENERIC_E4 (design review): true = generic context_parts' scalar e4() per byte; false = context_parts_cpb's fp8_pair.
// They differ only for NaN codes (0x7f/0xff); each request keeps the conversion of its release reference.
template<class Input,int HG,int D,int CB,bool GENERIC_E4=false,bool IDOK=false>
__global__ void context_flip(Input input,int start,const float* probs,float* parts){
 auto a=ga::resolve(input);if(!a.active)return;constexpr int THREADS=D/2,GROUPS=(D+127)/128;
 int row=blockIdx.z,r=start+row,h0=blockIdx.y*HG,part=blockIdx.x/(512/D),d0=(blockIdx.x%(512/D))*D,t=threadIdx.x;FOR_ROW(a,input,r);
 __shared__ uintptr_t rows[256];__shared__ float probability[HG][256],scale[256*GROUPS];bool active=false;
 for(int j=t;j<256;j+=THREADS){int slot=part*256+j,id=a.indices[(int64_t)r*ga::TOP+slot];bool anyp=false;
  const bool idok=id>=0&&id<=a.positions[r];   // D256 bench: IDOK=true skips prob reads for dead slots (rows[j]=null, never read)
  if(!IDOK||idok)for(int hh=0;hh<HG;hh++){int h=h0+hh;float p=h<a.heads?probs[((int64_t)row*a.heads+h)*ga::TOP+slot]:0.f;probability[hh][j]=p;anyp|=p!=0;}
  const uint8_t* src=nullptr;if(idok&&anyp)src=ga::kv_at(a,id);
  rows[j]=(uintptr_t)src;active|=src!=nullptr;if(src)for(int g=0;g<GROUPS;g++)scale[j*GROUPS+g]=((const float*)(src+512))[d0/128+g];}
 bool any=__syncthreads_or(active);float lo[HG],hi[HG];int d=d0+2*t;
 #pragma unroll
 for(int hh=0;hh<HG;hh++){lo[hh]=0;hi[hh]=0;}
 if(any){
 for(int j0=0;j0<256;j0+=CB){uint16_t raw[CB];
  #pragma unroll
  for(int u=0;u<CB;u++){const uint8_t* src=(const uint8_t*)rows[j0+u];raw[u]=src?*(const uint16_t*)(src+d):(uint16_t)0;}
  #pragma unroll
  for(int u=0;u<CB;u++){const int j=j0+u;if(!rows[j])continue;float2 x;if(GENERIC_E4){x.x=ga::e4((uint8_t)raw[u]);x.y=ga::e4((uint8_t)(raw[u]>>8));}else x=fp8_pair(raw[u]);float s=scale[j*GROUPS+(2*t)/128];
   #pragma unroll
   for(int hh=0;hh<HG;hh++){float p=probability[hh][j];if(p==0)continue;lo[hh]=fmaf(p,x.x*s,lo[hh]);hi[hh]=fmaf(p,x.y*s,hi[hh]);}}}}
 #pragma unroll
 for(int hh=0;hh<HG;hh++){int h=h0+hh;if(h>=a.heads)break;float* dst=parts+(((int64_t)row*a.heads+h)*ga::PART+part)*512;dst[d]=lo[hh];dst[d+1]=hi[hh];}
}
}
