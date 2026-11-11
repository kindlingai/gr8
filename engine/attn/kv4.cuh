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
// Four warps share the launch; each warp reuses its query across S key slots.
template<int S,int TR,bool QT,class Input,bool PH=false>
__device__ __forceinline__ void scores_sc_body(Input input,int start,const float* q,const float* rope,float* out,int tid){
 static_assert(S==1||S==2||S==4,"S in {1,2,4}");
 static_assert(TR>=0&&TR<=2,"TR in {0,1,2}");
 auto a=ga::resolve(input);if(!a.active)return;
 constexpr int WARPS=4,HG=16;
 int row=blockIdx.z,r=start+row,h0=blockIdx.y*HG,j0=(blockIdx.x*WARPS+tid/32)*S,lane=tid&31;FOR_ROW(a,input,r);
 const int nh=PH?min(HG,a.heads-h0):HG;
 const uint8_t* v[S];bool any=false;float mine[S];
 #pragma unroll
 for(int s=0;s<S;s++){
  const int id=a.indices[(int64_t)r*ga::TOP+j0+s];
  v[s]=(id>=0&&id<=a.positions[r]&&CP_MINE(a,id))?ga::kv_at(a,id):nullptr;
  any|=v[s]!=nullptr;mine[s]=-INFINITY;
 }
 if(any){
  float kv[S][16],rv[S][2];
  #pragma unroll
  for(int s=0;s<S;s++){
   if(v[s]){
    const uint2 raw=*reinterpret_cast<const uint2*>(v[s]+8*lane);
    const float sc=__half2float(reinterpret_cast<const half*>(v[s]+g6kv::SCALE_OFF)[lane>>1]);
    #pragma unroll
    for(int i=0;i<8;i++){
     const uint32_t word=i<4?raw.x:raw.y,byte=(word>>(8*(i&3)))&0xffu;
     kv[s][2*i]=(float)((int)(byte&15u)-8)*sc;kv[s][2*i+1]=(float)((int)(byte>>4)-8)*sc;
    }
    #pragma unroll
    for(int k=0;k<2;k++)rv[s][k]=__bfloat162float(reinterpret_cast<const bf16*>(v[s]+g6kv::ROPE_OFF)[lane+32*k]);
   }else{
    #pragma unroll
    for(int k=0;k<16;k++)kv[s][k]=0.f;
    rv[s][0]=rv[s][1]=0.f;
   }
  }
  auto loadq=[&](int hh,float4 (&qx)[4],float (&pv)[2]){
   const float* qb=q+((int64_t)row*a.heads+h0+hh)*512;
   #pragma unroll
   for(int c=0;c<4;c++){
    const int offset=QT?(c*32+lane)*4:16*lane+4*c;
    if((((uintptr_t)qb)&15u)==0u)qx[c]=__ldg(reinterpret_cast<const float4*>(qb+offset));
    else{const float* p4=qb+offset;qx[c]=make_float4(__ldg(p4),__ldg(p4+1),__ldg(p4+2),__ldg(p4+3));}
   }
   const float* rb=rope+((int64_t)row*a.heads+h0+hh)*64+lane;
   pv[0]=__ldg(rb);pv[1]=__ldg(rb+32);
  };
  auto dot=[&](const float4 (&qx)[4],const float (&pv)[2],int s){
   float sum=0.f;
   #pragma unroll
   for(int c=0;c<4;c++){
    sum=fmaf(qx[c].x,kv[s][4*c],sum);sum=fmaf(qx[c].y,kv[s][4*c+1],sum);
    sum=fmaf(qx[c].z,kv[s][4*c+2],sum);sum=fmaf(qx[c].w,kv[s][4*c+3],sum);
   }
   sum=fmaf(pv[0],rv[s][0],sum);sum=fmaf(pv[1],rv[s][1],sum);return sum;
  };
  // Even-lane head slot map (src = 2 * want) after four head/lane transpose stages.
  const int want=lane&15;
  const int src=((want&8)<<1)|((want&4)<<1)|((want&2)<<1)|((want&1)<<1);
  if constexpr(TR==2){
   auto head=[&](int hh,float (&out_head)[S]){
    if(PH&&hh>=nh){
     #pragma unroll
     for(int s=0;s<S;s++)out_head[s]=0.f;
     return;
    }
    float4 qx[4];float pv[2];loadq(hh,qx,pv);
    #pragma unroll
    for(int s=0;s<S;s++)out_head[s]=dot(qx,pv,s);
   };
   auto comb=[&](float x,float y,int o){
    const bool hi=(lane&o)!=0;const float keep=hi?y:x,send=hi?x:y;
    return keep+__shfl_xor_sync(0xffffffff,send,o);
   };
   float t2[2][S];
   #pragma unroll
   for(int a2=0;a2<2;a2++){
    float t3[2][S];
    #pragma unroll
    for(int a3=0;a3<2;a3++){
     float t4[2][S];
     #pragma unroll
     for(int a4=0;a4<2;a4++){
      const int hh=a2+2*a3+4*a4;float x[S],y[S];head(hh,x);head(hh+8,y);
      #pragma unroll
      for(int s=0;s<S;s++)t4[a4][s]=comb(x[s],y[s],16);
     }
     #pragma unroll
     for(int s=0;s<S;s++)t3[a3][s]=comb(t4[0][s],t4[1][s],8);
    }
    #pragma unroll
    for(int s=0;s<S;s++)t2[a2][s]=comb(t3[0][s],t3[1][s],4);
   }
   #pragma unroll
   for(int s=0;s<S;s++){
    float f=comb(t2[0][s],t2[1][s],2);f+=__shfl_xor_sync(0xffffffff,f,1);
    const float got=__shfl_sync(0xffffffff,f,src);mine[s]=got*(1.f/16);
   }
  }else{
   float acc[TR?S:1][TR?HG:1];
   #pragma unroll
   for(int hh=0;hh<HG;hh++){
    if(PH&&hh>=nh){
     if constexpr(TR){
      #pragma unroll
      for(int s=0;s<S;s++)acc[s][hh]=0.f;
     }
     continue;
    }
    float4 qx[4];float pv[2];loadq(hh,qx,pv);
    #pragma unroll
    for(int s=0;s<S;s++){
     float sum=dot(qx,pv,s);
     if constexpr(TR)acc[s][hh]=sum;
     else{sum=ga::warp_sum(sum);if(lane==hh)mine[s]=sum*(1.f/16);}
    }
   }
   if constexpr(TR==1){
    #pragma unroll
    for(int s=0;s<S;s++){
     #pragma unroll
     for(int n=HG,o=16;o>=1;o>>=1){
      if(n>1){
       const int hf=n/2;const bool hi=(lane&o)!=0;
       #pragma unroll
       for(int i=0;i<HG/2;i++)if(i<hf){
        const float keep=hi?acc[s][i+hf]:acc[s][i],send=hi?acc[s][i]:acc[s][i+hf];
        acc[s][i]=keep+__shfl_xor_sync(0xffffffff,send,o);
       }
       n=hf;
      }else acc[s][0]+=__shfl_xor_sync(0xffffffff,acc[s][0],o);
     }
     const float got=__shfl_sync(0xffffffff,acc[s][0],src);mine[s]=got*(1.f/16);
    }
   }
  }
  #pragma unroll
  for(int s=0;s<S;s++)if(!v[s])mine[s]=-INFINITY;
 }
 if(lane<nh){
  float* o=out+((int64_t)row*a.heads+h0+lane)*ga::TOP+j0;
  if constexpr(S==4){if((((uintptr_t)o)&15u)==0u){*reinterpret_cast<float4*>(o)=make_float4(mine[0],mine[1],mine[2],mine[3]);return;}}
  if constexpr(S==2){if((((uintptr_t)o)&7u)==0u){*reinterpret_cast<float2*>(o)=make_float2(mine[0],mine[1]);return;}}
  #pragma unroll
  for(int s=0;s<S;s++)o[s]=mine[s];
 }
}
template<int S,int TR,class Input,bool PH=false>
__global__ void __launch_bounds__(128) scores_sc(Input input,int start,const float* q,const float* rope,float* out){
 scores_sc_body<S,TR,false,Input,PH>(input,start,q,rope,out,threadIdx.x);
}
template<int S,int TR,class Input,bool PH=false>
__global__ void __launch_bounds__(128) scores_sc_qt(Input input,int start,const float* q,const float* rope,float* out){
 scores_sc_body<S,TR,true,Input,PH>(input,start,q,rope,out,threadIdx.x);
}

// Compact in ascending slot order before the original per-dimension FMA chain.
template<int HG,int PB,int D,class Input>
__global__ void __launch_bounds__(D/2) context_pk_compact(Input input,int start,const float* probs,float* parts){
 static_assert((D==256||D==512)&&PB==16&&(HG==2||HG==4||HG==8),"kv4 context_pk_compact shape");
 constexpr int THREADS=D/2,GROUPS=D/32,NWARPS=THREADS/32,PASSES=256/THREADS,SL=512/D;
 __shared__ const uint8_t* c_row[256];
 __shared__ __align__(16) float pm[256][HG];
 __shared__ __align__(16) half scale_h[256][GROUPS];
 __shared__ int wbase[NWARPS],n_live_sh;
 auto a=ga::resolve(input);if(!a.active)return;
 int row=blockIdx.z,r=start+row,h0=blockIdx.y*HG,part=blockIdx.x/SL,d0=(blockIdx.x%SL)*D,t=threadIdx.x,w=t>>5,lane=t&31;FOR_ROW(a,input,r);
 const int nh=min(HG,a.heads-h0);
 if(t==0)n_live_sh=0;
 __syncthreads();
 #pragma unroll
 for(int pass=0;pass<PASSES;pass++){
  const int j=pass*THREADS+t,slot=part*256+j,id=a.indices[(int64_t)r*ga::TOP+slot];
  const bool idok=id>=0&&id<=a.positions[r];
  float pv[HG];bool live=false;
  #pragma unroll
  for(int hh=0;hh<HG;hh++){
   const float p=(idok&&hh<nh)?probs[((int64_t)row*a.heads+h0+hh)*ga::TOP+slot]:0.f;
   pv[hh]=p;live|=p!=0.f;
  }
  const uint8_t* src=live?ga::kv_at(a,id):nullptr;
  const bool rd=src!=nullptr;const uint32_t mask=__ballot_sync(0xffffffff,rd);
  if(lane==0)wbase[w]=__popc(mask);
  __syncthreads();
  if(t==0){int s=n_live_sh;for(int k=0;k<NWARPS;k++){const int c=wbase[k];wbase[k]=s;s+=c;}n_live_sh=s;}
  __syncthreads();
  if(rd){
   const int pos=wbase[w]+__popc(mask&((1u<<lane)-1u));c_row[pos]=src;
   #pragma unroll
   for(int hh=0;hh<HG;hh++)pm[pos][hh]=pv[hh];
   const uint2* sp2=reinterpret_cast<const uint2*>(src+g6kv::SCALE_OFF+(d0>>4));
   uint2* dst2=reinterpret_cast<uint2*>(&scale_h[pos][0]);
   #pragma unroll
   for(int g4=0;g4<GROUPS/4;g4++)dst2[g4]=__ldg(sp2+g4);
  }
  __syncthreads();
 }
 const int n_live=n_live_sh,d=d0+2*t,byte_off=d>>1,grp=t>>4;
 float lo[HG],hi[HG];
 #pragma unroll
 for(int hh=0;hh<HG;hh++){lo[hh]=0.f;hi[hh]=0.f;}
 if(n_live>0){
  const int nb=(n_live+PB-1)/PB;
  auto probs_of=[&](int idx,float (&p)[HG]){
   if constexpr(HG%4==0){
    #pragma unroll
    for(int i=0;i<HG;i+=4){const float4 v=*reinterpret_cast<const float4*>(&pm[idx][i]);p[i]=v.x;p[i+1]=v.y;p[i+2]=v.z;p[i+3]=v.w;}
   }else{const float2 v=*reinterpret_cast<const float2*>(&pm[idx][0]);p[0]=v.x;p[1]=v.y;}
  };
  auto load_batch=[&](uint8_t (&raw)[PB],int b){
   const int base=b*PB;
   #pragma unroll
   for(int u=0;u<PB;u++){const int idx=base+u;raw[u]=idx<n_live?__ldg(c_row[idx]+byte_off):(uint8_t)0x88;}
  };
  auto use_batch=[&](const uint8_t (&raw)[PB],int b){
   const int base=b*PB;
   #pragma unroll
   for(int u=0;u<PB;u++){
    const int idx=base+u;if(idx>=n_live)break;
    const float s=__half2float(scale_h[idx][grp]);
    const float x0=(float)((int)(raw[u]&15u)-8)*s,x1=(float)((int)(raw[u]>>4)-8)*s;
    float p[HG];probs_of(idx,p);
    #pragma unroll
    for(int hh=0;hh<HG;hh++){
     const float fl=fmaf(p[hh],x0,lo[hh]),fh=fmaf(p[hh],x1,hi[hh]);
     lo[hh]=p[hh]!=0.f?fl:lo[hh];hi[hh]=p[hh]!=0.f?fh:hi[hh];
    }
   }
  };
  uint8_t ra[PB],rb[PB];load_batch(ra,0);
  for(int b=0;b<nb;b+=2){if(b+1<nb)load_batch(rb,b+1);use_batch(ra,b);if(b+2<nb)load_batch(ra,b+2);if(b+1<nb)use_batch(rb,b+1);}
 }
 #pragma unroll
 for(int hh=0;hh<HG;hh++)if(hh<nh){float* dst=parts+(((int64_t)row*a.heads+h0+hh)*ga::PART+part)*512;dst[d]=lo[hh];dst[d+1]=hi[hh];}
}

// The tile is independent of the total head count, including gathered CP heads.
__device__ __forceinline__ void tc_store8(float* dst,const float* x){
 *reinterpret_cast<float4*>(dst)=make_float4(x[0],x[1],x[2],x[3]);
 *reinterpret_cast<float4*>(dst+4)=make_float4(x[4],x[5],x[6],x[7]);
}
// Each packed byte holds the even dimension in its low nibble.
__device__ __forceinline__ __half2 tc_pair(unsigned byte,half scale){
 return __hmul2(__floats2half2_rn(float(int(byte&15)-8),float(int((byte>>4)&15)-8)),__halves2half2(scale,scale));
}
template<class Input>
__global__ void __launch_bounds__(128,4) scores_tc16(Input input,int start,const float* __restrict__ q,const float* __restrict__ rope,float* __restrict__ out){
 using namespace nvcuda;
 auto a=ga::resolve(input);if(!a.active)return;
 int row=blockIdx.z,r=start+row,h0=blockIdx.y*16,j0=blockIdx.x*64,w=threadIdx.x>>5,lane=threadIdx.x&31;FOR_ROW(a,input,r);
 const int nh=min(16,a.heads-h0);
 __shared__ __align__(32) half sq[16][584];
 __shared__ float qscale[16];
 __shared__ const uint8_t* sv[4][16];
 __shared__ __align__(32) half sk[4][16][40];
 for(int hh=4*w;hh<4*w+4;hh++){
  float x[18],am=0;
  #pragma unroll
  for(int u=0;u<18;u++){
   int d=lane+32*u;float v=0;
   if(hh<nh)v=d<512?q[((int64_t)row*a.heads+h0+hh)*512+d]:rope[((int64_t)row*a.heads+h0+hh)*64+d-512];
   x[u]=v;am=fmaxf(am,fabsf(v));
  }
  am=ga::warp_max(am);int e=0;frexpf(am,&e);
  const float qs=am>0?ldexpf(1.f,e-8):0.f,inv=am>0?ldexpf(1.f,8-e):0.f;
  if(!lane)qscale[hh]=qs*(1.f/16.f);
  #pragma unroll
  for(int u=0;u<18;u++)sq[hh][lane+32*u]=__float2half_rn(x[u]*inv);
  if(lane<8)sq[hh][576+lane]=__float2half_rn(0.f);
 }
 __syncthreads();
 if(lane<16){int id=a.indices[(int64_t)r*ga::TOP+j0+w*16+lane];sv[w][lane]=(id>=0&&id<=a.positions[r]&&CP_MINE(a,id))?ga::kv_at(a,id):nullptr;}
 __syncwarp();
 const unsigned mask=__ballot_sync(0xffffffffu,lane<16&&sv[w][lane]!=nullptr);
 const int hh=lane>>1,hs0=(lane&1)*8;
 if(!mask){if(hh<nh){float x[8];for(int i=0;i<8;i++)x[i]=-INFINITY;tc_store8(out+((int64_t)row*a.heads+h0+hh)*ga::TOP+j0+w*16+hs0,x);}return;}
 wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major>A;
 wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major>B;
 wmma::fragment<wmma::accumulator,16,16,16,float>C;wmma::fill_fragment(C,0.f);
 const int slot=lane>>1,hl=lane&1;
 const uint8_t* vp=sv[w][slot];
 for(int g=0;g<16;g++){
  __align__(16) half vals[16];
  uint2 raw=vp?*reinterpret_cast<const uint2*>(vp+g*16+hl*8):make_uint2(0x88888888u,0x88888888u);
  half scale=vp?reinterpret_cast<const half*>(vp+g6kv::SCALE_OFF)[g]:__float2half_rn(0.f);
  #pragma unroll
  for(int i=0;i<8;i++){unsigned word=i<4?raw.x:raw.y;reinterpret_cast<__half2*>(vals)[i]=tc_pair(word>>(8*(i&3)),scale);}
  reinterpret_cast<uint4*>(&sk[w][slot][hl*16])[0]=reinterpret_cast<uint4*>(vals)[0];
  reinterpret_cast<uint4*>(&sk[w][slot][hl*16])[1]=reinterpret_cast<uint4*>(vals)[1];
  __syncwarp();
  #pragma unroll
  for(int k=0;k<32;k+=16){wmma::load_matrix_sync(A,&sq[0][g*32+k],584);wmma::load_matrix_sync(B,&sk[w][0][k],40);wmma::mma_sync(C,A,B,C);}
  __syncwarp();
 }
 for(int g=0;g<2;g++){
  __align__(16) half vals[16];
  #pragma unroll
  for(int i=0;i<16;i++)vals[i]=__float2half_rn(vp?__bfloat162float(reinterpret_cast<const bf16*>(vp+g6kv::ROPE_OFF)[g*32+hl*16+i]):0.f);
  reinterpret_cast<uint4*>(&sk[w][slot][hl*16])[0]=reinterpret_cast<uint4*>(vals)[0];
  reinterpret_cast<uint4*>(&sk[w][slot][hl*16])[1]=reinterpret_cast<uint4*>(vals)[1];
  __syncwarp();
  #pragma unroll
  for(int k=0;k<32;k+=16){wmma::load_matrix_sync(A,&sq[0][512+g*32+k],584);wmma::load_matrix_sync(B,&sk[w][0][k],40);wmma::mma_sync(C,A,B,C);}
  __syncwarp();
 }
 float* ws=reinterpret_cast<float*>(&sk[w][0][0]);wmma::store_matrix_sync(ws,C,16,wmma::mem_row_major);__syncwarp();
 if(hh<nh){float x[8];for(int i=0;i<8;i++)x[i]=sv[w][hs0+i]?ws[hh*16+hs0+i]*qscale[hh]:-INFINITY;tc_store8(out+((int64_t)row*a.heads+h0+hh)*ga::TOP+j0+w*16+hs0,x);}
}

template<class Input>
__global__ void __launch_bounds__(128,4) context_parts_tc16(Input input,int start,const float* __restrict__ probs,float* __restrict__ parts){
 using namespace nvcuda;
 auto a=ga::resolve(input);if(!a.active)return;
 int row=blockIdx.z,r=start+row,h0=blockIdx.y*16,part=blockIdx.x>>1,d0=(blockIdx.x&1)*256,t=threadIdx.x,w=t>>5,lane=t&31;FOR_ROW(a,input,r);
 const int nh=min(16,a.heads-h0),dw=d0+w*64,g0=dw>>5;
 __shared__ const uint8_t* srows[256];
 __shared__ __align__(32) half sp[16][264];
 __shared__ float pscale[16];
 __shared__ __align__(32) half sv[4][16][72];
 bool active=false;
 for(int j=t;j<256;j+=128){
  int slot=part*256+j,id=a.indices[(int64_t)r*ga::TOP+slot];bool live=false;
  if(id>=0&&id<=a.positions[r])for(int hh=0;hh<nh;hh++)live|=probs[((int64_t)row*a.heads+h0+hh)*ga::TOP+slot]!=0.f;
  const uint8_t* vp=live?ga::kv_at(a,id):nullptr;srows[j]=vp;active|=vp!=nullptr;
 }
 if(!__syncthreads_or(active)){
  for(int i=t;i<nh*64;i+=128){int hh=i/64,d=(i%64)*4;*reinterpret_cast<float4*>(parts+(((int64_t)row*a.heads+h0+hh)*ga::PART+part)*512+d0+d)=make_float4(0,0,0,0);}return;
 }
 for(int hh=4*w;hh<4*w+4;hh++){
  float x[8],am=0;
  #pragma unroll
  for(int u=0;u<8;u++){int j=lane+32*u;float v=hh<nh&&srows[j]?probs[((int64_t)row*a.heads+h0+hh)*ga::TOP+part*256+j]:0.f;x[u]=v;am=fmaxf(am,fabsf(v));}
  am=ga::warp_max(am);int e=0;frexpf(am,&e);float ps=am>0?ldexpf(1.f,e-8):0.f,inv=am>0?ldexpf(1.f,8-e):0.f;
  if(!lane)pscale[hh]=ps;
  #pragma unroll
  for(int u=0;u<8;u++)sp[hh][lane+32*u]=__float2half_rn(x[u]*inv);
  if(lane<8)sp[hh][256+lane]=__float2half_rn(0.f);
 }
 __syncthreads();
 wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major>A;
 wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major>B;
 wmma::fragment<wmma::accumulator,16,16,16,float>C[4];
 #pragma unroll
 for(int c=0;c<4;c++)wmma::fill_fragment(C[c],0.f);
 for(int j0=0;j0<256;j0+=16){
  if(!__ballot_sync(0xffffffffu,lane<16&&srows[j0+lane]!=nullptr))continue;
  int u=lane>>1,grp=lane&1;const uint8_t* vp=srows[j0+u];
  __align__(16) half vals[32];
  uint4 raw=vp?*reinterpret_cast<const uint4*>(vp+(g0+grp)*16):make_uint4(0x88888888u,0x88888888u,0x88888888u,0x88888888u);
  half scale=vp?reinterpret_cast<const half*>(vp+g6kv::SCALE_OFF)[g0+grp]:__float2half_rn(0.f);
  #pragma unroll
  for(int i=0;i<16;i++){unsigned word=i<4?raw.x:i<8?raw.y:i<12?raw.z:raw.w;reinterpret_cast<__half2*>(vals)[i]=tc_pair(word>>(8*(i&3)),scale);}
  #pragma unroll
  for(int i=0;i<4;i++)reinterpret_cast<uint4*>(&sv[w][u][grp*32])[i]=reinterpret_cast<uint4*>(vals)[i];
  __syncwarp();wmma::load_matrix_sync(A,&sp[0][j0],264);
  #pragma unroll
  for(int c=0;c<4;c++){wmma::load_matrix_sync(B,&sv[w][0][c*16],72);wmma::mma_sync(C[c],A,B,C[c]);}
  __syncwarp();
 }
 float* ws=reinterpret_cast<float*>(&sv[w][0][0]);int hh=lane>>1,hd0=(lane&1)*8;
 #pragma unroll
 for(int c=0;c<4;c++){
  wmma::store_matrix_sync(ws,C[c],16,wmma::mem_row_major);__syncwarp();
  if(hh<nh){float x[8];for(int i=0;i<8;i++)x[i]=ws[hh*16+hd0+i]*pscale[hh];tc_store8(parts+(((int64_t)row*a.heads+h0+hh)*ga::PART+part)*512+dw+c*16+hd0,x);}
  __syncwarp();
 }
}

}
#endif
