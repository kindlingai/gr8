// G6_KV4 decode kernels: the head-batched scores and context of decode-small.cuh over the INT4 latent row (kvfmt.h).
// Included inside namespace ga, before decode-small.cuh; only G6_KV4 builds dispatch them. The query and the context
// are in the rotated basis (fwht_rows in attn.cu), so the latent is read as stored.
#if G6_KV4
namespace kv4 {
using ga::half;using ga::bf16;
// warp per (slot, row), heads [y*HG, y*HG + HG): lane l reads dims 16l..16l+15 (8 bytes, one scale) and the rope
// elements l and l + 32; lane hh stores head y*HG + hh
template<int HG,class Input>
__global__ void __launch_bounds__(256) scores_heads(Input input,int start,const float* q,const float* rope,float* out){
 auto a=ga::resolve(input);if(!a.active)return;constexpr int WARPS=8;int row=blockIdx.z,r=start+row,h0=blockIdx.y*HG,j=blockIdx.x*WARPS+threadIdx.x/32,lane=threadIdx.x&31;FOR_ROW(a,input,r);
 const int nh=min(HG,a.heads-h0);
 int id=a.indices[(int64_t)r*ga::TOP+j];const uint8_t* v=(id>=0&&id<=a.positions[r]&&CP_MINE(a,id))?ga::kv_at(a,id):nullptr;
 float mine=-INFINITY;
 if(v){float kv[16],rv[2];
  const uint2 raw=*reinterpret_cast<const uint2*>(v+8*lane);
  const float s=__half2float(reinterpret_cast<const half*>(v+g6kv::SCALE_OFF)[lane>>1]);
  #pragma unroll
  for(int i=0;i<8;i++){const uint32_t w=i<4?raw.x:raw.y;const uint32_t b=(w>>(8*(i&3)))&0xffu;
   kv[2*i]=(float)((int)(b&15u)-8)*s;kv[2*i+1]=(float)((int)(b>>4)-8)*s;}
  #pragma unroll
  for(int k=0;k<2;k++)rv[k]=__bfloat162float(((const bf16*)(v+g6kv::ROPE_OFF))[lane+k*32]);
  for(int hh=0;hh<nh;hh++){const int h=h0+hh;float sum=0;
   const float4* qh=reinterpret_cast<const float4*>(q+((int64_t)row*a.heads+h)*512+16*lane);
   #pragma unroll
   for(int k=0;k<4;k++){const float4 x=__ldg(qh+k);sum=fmaf(x.x,kv[4*k],sum);sum=fmaf(x.y,kv[4*k+1],sum);sum=fmaf(x.z,kv[4*k+2],sum);sum=fmaf(x.w,kv[4*k+3],sum);}
   #pragma unroll
   for(int k=0;k<2;k++)sum=fmaf(__ldg(rope+((int64_t)row*a.heads+h)*64+lane+k*32),rv[k],sum);
   sum=ga::warp_sum(sum);if(lane==hh)mine=sum*(1.f/16);}}
 if(lane<nh)out[((int64_t)row*a.heads+h0+lane)*ga::TOP+j]=mine;
}
// block per (part, half of the 512 dims, head group, row); thread t owns dims d, d + 1 (one byte) for HG heads
template<int HG,class Input>
__global__ void __launch_bounds__(128) context_parts_heads(Input input,int start,const float* probs,float* parts){
 auto a=ga::resolve(input);if(!a.active)return;constexpr int D=256,THREADS=D/2,GROUPS=D/32,PB=16;
 int row=blockIdx.z,r=start+row,h0=blockIdx.y*HG,part=blockIdx.x/(512/D),d0=(blockIdx.x%(512/D))*D,t=threadIdx.x;FOR_ROW(a,input,r);
 const int nh=min(HG,a.heads-h0);
 __shared__ uintptr_t rows[256];__shared__ float probability[HG][256],scale[256*GROUPS];bool active=false;
 for(int j=t;j<256;j+=THREADS){int slot=part*256+j,id=a.indices[(int64_t)r*ga::TOP+slot];const uint8_t* src=nullptr;
  const bool idok=id>=0&&id<=a.positions[r];   // dead slots read no probabilities
  bool live=false;for(int hh=0;hh<HG;hh++){float p=idok&&hh<nh?probs[((int64_t)row*a.heads+h0+hh)*ga::TOP+slot]:0.f;probability[hh][j]=p;live|=p!=0;}
  if(live)src=ga::kv_at(a,id);
  rows[j]=(uintptr_t)src;
  if(src){const half* sp=reinterpret_cast<const half*>(src+g6kv::SCALE_OFF)+d0/32;
   #pragma unroll
   for(int g=0;g<GROUPS;g++)scale[j*GROUPS+g]=__half2float(sp[g]);}
  active|=src!=nullptr;}
 const bool any=__syncthreads_or(active);float lo[HG],hi[HG];
 #pragma unroll
 for(int hh=0;hh<HG;hh++){lo[hh]=0;hi[hh]=0;}
 const int d=d0+2*t;
 if(any)for(int j0=0;j0<256;j0+=PB){uint8_t raw[PB];
  #pragma unroll
  for(int u=0;u<PB;u++){const uint8_t* src=(const uint8_t*)rows[j0+u];raw[u]=src?src[d>>1]:(uint8_t)0x88;}
  #pragma unroll
  for(int u=0;u<PB;u++){const int j=j0+u;if(!rows[j])continue;const float s=scale[j*GROUPS+(2*t)/32];
   const float x0=(float)((int)(raw[u]&15)-8)*s,x1=(float)((int)(raw[u]>>4)-8)*s;
   #pragma unroll
   for(int hh=0;hh<HG;hh++){float p=probability[hh][j];if(p==0)continue;lo[hh]=fmaf(p,x0,lo[hh]);hi[hh]=fmaf(p,x1,hi[hh]);}}}
 #pragma unroll
 for(int hh=0;hh<HG;hh++)if(hh<nh){float* dst=parts+(((int64_t)row*a.heads+h0+hh)*ga::PART+part)*512;dst[d]=lo[hh];dst[d+1]=hi[hh];}
}
}
#endif
