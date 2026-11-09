// Exact small-decode kernels; included inside ga after Work is defined.
// No cache, arithmetic boundary, selection-order or scratch-layout changes.
#define G6_SMALL_DECODE_KERNELS 1
namespace small_decode {
using ga::half;using ga::bf16;
__device__ __forceinline__ float2 fp8_pair(uint16_t u){__half2_raw raw=__nv_cvt_fp8x2_to_halfraw2(u,__NV_E4M3);return __half22float2(__half2(raw));}
// No CTA-level query staging: cached read-only Q loads and fully unrolled
// lane-strided accumulation retain the old 16 latent +2 RoPE FMA order.
template<class Input>
__global__ void scores_direct(Input input,int start,const float* q,const float* rope,float* out){
 auto a=ga::resolve(input);if(!a.active)return;constexpr int WARPS=8;int row=blockIdx.z,r=start+row,h=blockIdx.y,j=blockIdx.x*WARPS+threadIdx.x/32,lane=threadIdx.x&31;FOR_ROW(a,input,r);
 int id=a.indices[(int64_t)r*ga::TOP+j];const uint8_t* v=(id>=0&&id<=a.positions[r]&&CP_MINE(a,id))?ga::kv_at(a,id):nullptr;float sum=0;
 if(v){float scale[4];
 #pragma unroll
 for(int g=0;g<4;g++)scale[g]=((const float*)(v+512))[g];
 #pragma unroll
 for(int k=0;k<16;k++){int d=lane+k*32;sum=fmaf(__ldg(q+((int64_t)row*a.heads+h)*512+d),ga::e4(v[d])*scale[k/4],sum);}
 #pragma unroll
 for(int k=0;k<2;k++){int d=lane+k*32;sum=fmaf(__ldg(rope+((int64_t)row*a.heads+h)*64+d),__bfloat162float(((const bf16*)(v+528))[d]),sum);}}
 sum=ga::warp_sum(sum);if(lane==0)out[((int64_t)row*a.heads+h)*ga::TOP+j]=v?sum*(1.f/16):-INFINITY;
}
template<class Input>
__global__ void context_parts(Input input,int start,const float* probs,float* parts){
 auto a=ga::resolve(input);if(!a.active)return;constexpr int D=256,THREADS=D/2,GROUPS=(D+127)/128;
 int row=blockIdx.z,r=start+row,h=blockIdx.y,part=blockIdx.x/(512/D),d0=(blockIdx.x%(512/D))*D,t=threadIdx.x;FOR_ROW(a,input,r);
 __shared__ uintptr_t rows[256];__shared__ float probability[256],scale[256*GROUPS];bool active=false;
 for(int j=t;j<256;j+=THREADS){int slot=part*256+j,id=a.indices[(int64_t)r*ga::TOP+slot];float p=probs[((int64_t)row*a.heads+h)*ga::TOP+slot];const uint8_t* src=nullptr;if(id>=0&&id<=a.positions[r]&&p!=0)src=ga::kv_at(a,id);
  rows[j]=(uintptr_t)src;probability[j]=p;active|=src!=nullptr;if(src)for(int g=0;g<GROUPS;g++)scale[j*GROUPS+g]=((const float*)(src+512))[d0/128+g];}
 bool any=__syncthreads_or(active);float lo=0,hi=0;int d=d0+2*t;
 if(any){
 #pragma unroll 4
 for(int j=0;j<256;j++){const uint8_t* src=(const uint8_t*)rows[j];if(!src)continue;float2 x=fp8_pair(*(const uint16_t*)(src+d));float s=scale[j*GROUPS+(2*t)/128],p=probability[j];lo=fmaf(p,x.x*s,lo);hi=fmaf(p,x.y*s,hi);}}
 float* dst=parts+(((int64_t)row*a.heads+h)*ga::PART+part)*512;dst[d]=lo;dst[d+1]=hi;
}
// DEC_CPB: context_parts with the fp8 pairs of the next CPB slots loaded before their FMAs, so
// the gathered loads overlap (ncu: context_parts is long_scoreboard-bound). Same FMAs in the same slot order, same skip
// of dead / p == 0 slots, so every output byte matches context_parts.
#ifndef DEC_CPB2   // DEC_CPB2: batch 8 -> 64 slots; default 8 = DEC_CPB as before
#define DEC_CPB2 8
#endif
constexpr int CPB=DEC_CPB2;
template<class Input>
__global__ void context_parts_cpb(Input input,int start,const float* probs,float* parts){
 auto a=ga::resolve(input);if(!a.active)return;constexpr int D=256,THREADS=D/2,GROUPS=(D+127)/128;
 int row=blockIdx.z,r=start+row,h=blockIdx.y,part=blockIdx.x/(512/D),d0=(blockIdx.x%(512/D))*D,t=threadIdx.x;FOR_ROW(a,input,r);
 __shared__ uintptr_t rows[256];__shared__ float probability[256],scale[256*GROUPS];bool active=false;
 for(int j=t;j<256;j+=THREADS){int slot=part*256+j,id=a.indices[(int64_t)r*ga::TOP+slot];float p=probs[((int64_t)row*a.heads+h)*ga::TOP+slot];const uint8_t* src=nullptr;if(id>=0&&id<=a.positions[r]&&p!=0)src=ga::kv_at(a,id);
  rows[j]=(uintptr_t)src;probability[j]=p;active|=src!=nullptr;if(src)for(int g=0;g<GROUPS;g++)scale[j*GROUPS+g]=((const float*)(src+512))[d0/128+g];}
 bool any=__syncthreads_or(active);float lo=0,hi=0;int d=d0+2*t;
 if(any){
 for(int j0=0;j0<256;j0+=CPB){uint16_t raw[CPB];
  #pragma unroll
  for(int u=0;u<CPB;u++){const uint8_t* src=(const uint8_t*)rows[j0+u];raw[u]=src?*(const uint16_t*)(src+d):(uint16_t)0;}
  #pragma unroll
  for(int u=0;u<CPB;u++){const int j=j0+u;if(!rows[j])continue;float2 x=fp8_pair(raw[u]);float s=scale[j*GROUPS+(2*t)/128],p=probability[j];lo=fmaf(p,x.x*s,lo);hi=fmaf(p,x.y*s,hi);}}}
 float* dst=parts+(((int64_t)row*a.heads+h)*ga::PART+part)*512;dst[d]=lo;dst[d+1]=hi;
}
// Head-batched forms of scores_direct and context_parts_cpb: one key's bytes are loaded once for HG heads
// instead of once per head. Every (row, head, slot) output keeps the per-head kernels' arithmetic: the same products, the
// same fmaf order, the same warp_sum, the same skip of dead slots and of p == 0, so the outputs are bit-identical.
// scores_heads: warp per (slot, row), heads [y*HG, y*HG + HG); lane l stores head y*HG + l.
template<int HG,class Input>
__global__ void __launch_bounds__(256) scores_heads(Input input,int start,const float* q,const float* rope,float* out){
 auto a=ga::resolve(input);if(!a.active)return;constexpr int WARPS=8;int row=blockIdx.z,r=start+row,h0=blockIdx.y*HG,j=blockIdx.x*WARPS+threadIdx.x/32,lane=threadIdx.x&31;FOR_ROW(a,input,r);
 const int nh=min(HG,a.heads-h0);
 int id=a.indices[(int64_t)r*ga::TOP+j];const uint8_t* v=(id>=0&&id<=a.positions[r]&&CP_MINE(a,id))?ga::kv_at(a,id):nullptr;
 float mine=-INFINITY;
 if(v){float scale[4],kv[16],rv[2];
  #pragma unroll
  for(int g=0;g<4;g++)scale[g]=((const float*)(v+512))[g];
  #pragma unroll
  for(int k=0;k<16;k++)kv[k]=ga::e4(v[lane+k*32])*scale[k/4];
  #pragma unroll
  for(int k=0;k<2;k++)rv[k]=__bfloat162float(((const bf16*)(v+528))[lane+k*32]);
  for(int hh=0;hh<nh;hh++){const int h=h0+hh;float sum=0;
   #pragma unroll
   for(int k=0;k<16;k++)sum=fmaf(__ldg(q+((int64_t)row*a.heads+h)*512+lane+k*32),kv[k],sum);
   #pragma unroll
   for(int k=0;k<2;k++)sum=fmaf(__ldg(rope+((int64_t)row*a.heads+h)*64+lane+k*32),rv[k],sum);
   sum=ga::warp_sum(sum);if(lane==hh)mine=sum*(1.f/16);}}
 if(lane<nh)out[((int64_t)row*a.heads+h0+lane)*ga::TOP+j]=mine;
}
// context_parts_heads: block per (part, half of the 512 dims, head group, row); thread t owns dims d, d+1 for HG heads.
// PB gathered pairs are loaded per batch (16; GR8_ATTN_CTX2=1 / 2: 32 / 64, the same FMAs in the same order).
template<int HG,int PB=16,class Input>
__global__ void __launch_bounds__(128) context_parts_heads(Input input,int start,const float* probs,float* parts){
 auto a=ga::resolve(input);if(!a.active)return;constexpr int D=256,THREADS=D/2,GROUPS=(D+127)/128;
 int row=blockIdx.z,r=start+row,h0=blockIdx.y*HG,part=blockIdx.x/(512/D),d0=(blockIdx.x%(512/D))*D,t=threadIdx.x;FOR_ROW(a,input,r);
 const int nh=min(HG,a.heads-h0);
 __shared__ uintptr_t rows[256];__shared__ float probability[HG][256],scale[256*GROUPS];bool active=false;
 for(int j=t;j<256;j+=THREADS){int slot=part*256+j,id=a.indices[(int64_t)r*ga::TOP+slot];const uint8_t* src=nullptr;
  const bool idok=id>=0&&id<=a.positions[r];   // dead slots read no probabilities
  bool live=false;for(int hh=0;hh<HG;hh++){float p=idok&&hh<nh?probs[((int64_t)row*a.heads+h0+hh)*ga::TOP+slot]:0.f;probability[hh][j]=p;live|=p!=0;}
  if(live)src=ga::kv_at(a,id);
  rows[j]=(uintptr_t)src;if(src)for(int g=0;g<GROUPS;g++)scale[j*GROUPS+g]=((const float*)(src+512))[d0/128+g];active|=src!=nullptr;}
 const bool any=__syncthreads_or(active);float lo[HG],hi[HG];
 #pragma unroll
 for(int hh=0;hh<HG;hh++){lo[hh]=0;hi[hh]=0;}
 int d=d0+2*t;
 if(any)for(int j0=0;j0<256;j0+=PB){uint16_t raw[PB];
  #pragma unroll
  for(int u=0;u<PB;u++){const uint8_t* src=(const uint8_t*)rows[j0+u];raw[u]=src?*(const uint16_t*)(src+d):(uint16_t)0;}
  #pragma unroll
  for(int u=0;u<PB;u++){const int j=j0+u;if(!rows[j])continue;float2 x=fp8_pair(raw[u]);float s=scale[j*GROUPS+(2*t)/128];
   #pragma unroll
   for(int hh=0;hh<HG;hh++){float p=probability[hh][j];if(p==0)continue;lo[hh]=fmaf(p,x.x*s,lo[hh]);hi[hh]=fmaf(p,x.y*s,hi[hh]);}}}
 #pragma unroll
 for(int hh=0;hh<HG;hh++)if(hh<nh){float* dst=parts+(((int64_t)row*a.heads+h0+hh)*ga::PART+part)*512;dst[d]=lo[hh];dst[d+1]=hi[hh];}
}
// GR8_ATTN_SC2=<S> (attn_sc2): scores_heads<16> with S consecutive slots per warp. Each head's 16 latent + 2 RoPE query
// values are loaded once per warp and used for all S slots (Q traffic / S, S independent fmaf chains per head), head loop
// unrolled. Per (row, head, slot): the same products in the same fmaf order, then the same warp_sum (TR = false) or
// scores_tr's transpose reduction (TR = true, GR8_ATTN_SC2_TR=1: per head the same xor tree, a + b commutes bitwise), the
// same 1/16 scale, -inf for a dead slot, the same output address -> bit-identical to scores_heads<16>. Validity is per slot
// and warp-uniform (one id per slot), so every shuffle stays full-warp; a dead slot in a live warp runs on zero operands and
// stores -inf. Full groups (heads % 16 == 0) or, with PH below, a partial last group. grid dim3(TOP/(4*S), ceil(heads/16), n),
// block 128: warp w of block x owns slots
// (4x + w)*S .. +S-1, lane l < 16 stores head h0 + l of them (one S-wide vector store).
// TR = 2 (GR8_ATTN_SC2_TR=2): the same transpose reduction, eager. Every combine of TR = 1 pairs entries i and i + n/2 of the
// stage, so the head entering it can be computed right before: heads in bit-reversed order (0 8 4 12 2 10 6 14 1 ...), each
// pair combined at offset 16 at once, two such results at 8, then 4, 2, and the last stage's a += shfl(a, 1). The same
// combines of the same values (keep / send by the same lane bit) -> TR = 1's bytes, with at most a few partials per slot
// live instead of 16 (registers). S = 8 (kv alone is 128 registers) is built with TR = 0 only (GR8_ATTN_SC2_TR ignored).
// QT (GR8_ATTN_QT, attn.cu use_qt): q / rope are lane-transposed (absorb_m2<R,true>): head hh's 16 latent q of a lane are four
// float4 (c*32 + lane, c = 0..3: qv[4c + e]) and its 2 rope q one float2 (lane), 5 loads instead of 18; a slot's 4 KV
// scales are one float4 (row = 41 x 16 B, mla_cache 16 B aligned). The same values in qv / pv / scale -> the same bytes.
// scores_sc_qt is that kernel, registers capped at 4 blocks / SM (S = 4 is 128 registers without QT).
// PH (partial heads, GR8_ATTN_SC2_PART, attn.cu attn_sc2_part; TP=6 and TP=3: 10 / 11 / 21 / 22 heads a rank): the last
// head group holds nh = heads - h0 < 16 heads. Heads nh..15 of it load nothing and enter the reductions as 0 (TR = 0:
// skipped); a real head's partials meet only its own partials in every combine (keep / send pick the same head index on
// both lanes), so each real head gets the full kernel's bytes, = scores_heads<16>'s with its nh. Only lanes < nh store.
// grid y = ceil(heads / 16). PH = false is the full-group kernel as before (heads % 16 == 0).
template<int S,int TR,bool QT,class Input,bool PH=false>
__device__ __forceinline__ void scores_sc_body(Input input,int start,const float* q,const float* rope,float* out,int tid){
 static_assert(S==1||S==2||S==4||S==8,"S in {1,2,4,8}");
 static_assert(TR>=0&&TR<=2&&!(S==8&&TR),"TR in {0,1,2}; S = 8 with TR 0");
 auto a=ga::resolve(input);if(!a.active)return;constexpr int HG=16,WARPS=4;
 int row=blockIdx.z,r=start+row,h0=blockIdx.y*HG,j0=(blockIdx.x*WARPS+tid/32)*S,lane=tid&31;FOR_ROW(a,input,r);
 const int nh=PH?min(HG,a.heads-h0):HG;   // PH: heads of this group (warp-uniform); else the full group
 const uint8_t* v[S];bool any=false;
 #pragma unroll
 for(int s=0;s<S;s++){int id=a.indices[(int64_t)r*ga::TOP+j0+s];v[s]=(id>=0&&id<=a.positions[r]&&CP_MINE(a,id))?ga::kv_at(a,id):nullptr;any|=v[s]!=nullptr;}
 float mine[S];
 #pragma unroll
 for(int s=0;s<S;s++)mine[s]=-INFINITY;
 if(any){float kv[S][16],rv[S][2];
  #pragma unroll
  for(int s=0;s<S;s++){
   if(v[s]){float scale[4];
    if constexpr(QT){const float4 s4=*(const float4*)(v[s]+512);scale[0]=s4.x;scale[1]=s4.y;scale[2]=s4.z;scale[3]=s4.w;}
    else{
    #pragma unroll
    for(int g=0;g<4;g++)scale[g]=((const float*)(v[s]+512))[g];}
    #pragma unroll
    for(int k=0;k<16;k++)kv[s][k]=ga::e4(v[s][lane+k*32])*scale[k/4];
    #pragma unroll
    for(int k=0;k<2;k++)rv[s][k]=__bfloat162float(((const bf16*)(v[s]+528))[lane+k*32]);}
   else{
    #pragma unroll
    for(int k=0;k<16;k++)kv[s][k]=0;
    rv[s][0]=rv[s][1]=0;}}
  const float* qh=q+((int64_t)row*a.heads+h0)*512+lane;const float* ph=rope+((int64_t)row*a.heads+h0)*64+lane;
  // head hh's query of this lane: d = lane + 32k (qv[k]), j = lane + 32k (pv[k])
  auto loadq=[&](int hh,float (&qv)[16],float (&pv)[2]){
   if constexpr(QT){const float4* q4=(const float4*)(qh-lane+hh*512)+lane;
    #pragma unroll
    for(int c=0;c<4;c++){const float4 x=__ldg(q4+c*32);qv[4*c]=x.x;qv[4*c+1]=x.y;qv[4*c+2]=x.z;qv[4*c+3]=x.w;}
    const float2 y=__ldg((const float2*)(ph-lane+hh*64)+lane);pv[0]=y.x;pv[1]=y.y;}
   else{
    #pragma unroll
    for(int k=0;k<16;k++)qv[k]=__ldg(qh+hh*512+k*32);
    #pragma unroll
    for(int k=0;k<2;k++)pv[k]=__ldg(ph+hh*64+k*32);}};
  if constexpr(TR==2){
   // head hh's S dot products (the TR = 0 / 1 chains)
   auto head=[&](int hh,float (&o)[S]){
    if(PH&&hh>=nh){
     #pragma unroll
     for(int s=0;s<S;s++)o[s]=0.f;
     return;}
    float qv[16],pv[2];loadq(hh,qv,pv);
    #pragma unroll
    for(int s=0;s<S;s++){float sum=0;
     #pragma unroll
     for(int k=0;k<16;k++)sum=fmaf(qv[k],kv[s][k],sum);
     #pragma unroll
     for(int k=0;k<2;k++)sum=fmaf(pv[k],rv[s][k],sum);
     o[s]=sum;}};
   // TR = 1's combine of entries x = acc[i] and y = acc[i + n/2] at offset o
   auto comb=[&](float x,float y,int o){const bool hi=(lane&o)!=0;const float keep=hi?y:x,send=hi?x:y;return keep+__shfl_xor_sync(0xffffffff,send,o);};
   int src=0;{const int want=lane%HG;int n=HG,o=16,hd=0;while(n>1){const int hf=n/2;if(want>=hd+hf){src|=o;hd+=hf;}n=hf;o>>=1;}}
   float t2[2][S];   // T2(a2) = comb(T3(a2), T3(a2+2), 4), T3(i) = comb(T4(i), T4(i+4), 8), T4(i) = comb(H(i), H(i+8), 16)
   #pragma unroll
   for(int a2=0;a2<2;a2++){float t3[2][S];
    #pragma unroll
    for(int a3=0;a3<2;a3++){float t4[2][S];
     #pragma unroll
     for(int a4=0;a4<2;a4++){const int i=a2+2*a3+4*a4;float x0[S],x1[S];head(i,x0);head(i+8,x1);
      #pragma unroll
      for(int s=0;s<S;s++)t4[a4][s]=comb(x0[s],x1[s],16);}
     #pragma unroll
     for(int s=0;s<S;s++)t3[a3][s]=comb(t4[0][s],t4[1][s],8);}
    #pragma unroll
    for(int s=0;s<S;s++)t2[a2][s]=comb(t3[0][s],t3[1][s],4);}
   #pragma unroll
   for(int s=0;s<S;s++){float f=comb(t2[0][s],t2[1][s],2);f+=__shfl_xor_sync(0xffffffff,f,1);
    const float got=__shfl_sync(0xffffffff,f,src);
    if(lane<HG)mine[s]=got*(1.f/16);}}
  else{
  float acc[TR?S:1][TR?HG:1];
  #pragma unroll
  for(int hh=0;hh<HG;hh++){
   if(PH&&hh>=nh){
    if constexpr(TR){
     #pragma unroll
     for(int s=0;s<S;s++)acc[s][hh]=0.f;}
    continue;}
   float qv[16],pv[2];loadq(hh,qv,pv);
   #pragma unroll
   for(int s=0;s<S;s++){float sum=0;
    #pragma unroll
    for(int k=0;k<16;k++)sum=fmaf(qv[k],kv[s][k],sum);
    #pragma unroll
    for(int k=0;k<2;k++)sum=fmaf(pv[k],rv[s][k],sum);
    if constexpr(TR)acc[s][hh]=sum;
    else{sum=ga::warp_sum(sum);if(lane==hh)mine[s]=sum*(1.f/16);}}}
  if constexpr(TR){
   int src=0;{const int want=lane%HG;int n=HG,o=16,hd=0;while(n>1){const int hf=n/2;if(want>=hd+hf){src|=o;hd+=hf;}n=hf;o>>=1;}}
   #pragma unroll
   for(int s=0;s<S;s++){   // scores_tr's reduction, per slot
    #pragma unroll
    for(int n=HG,o=16;o>=1;o>>=1){
     if(n>1){const bool hi=(lane&o)!=0;const int hf=n/2;
      #pragma unroll
      for(int i=0;i<HG/2;i++)if(i<hf){float keep=hi?acc[s][i+hf]:acc[s][i],send=hi?acc[s][i]:acc[s][i+hf];acc[s][i]=keep+__shfl_xor_sync(0xffffffff,send,o);}
      n=hf;}
     else acc[s][0]+=__shfl_xor_sync(0xffffffff,acc[s][0],o);}
    const float got=__shfl_sync(0xffffffff,acc[s][0],src);
    if(lane<HG)mine[s]=got*(1.f/16);}}}
  #pragma unroll
  for(int s=0;s<S;s++)if(!v[s])mine[s]=-INFINITY;}
 if(lane<nh){float* o=out+((int64_t)row*a.heads+h0+lane)*ga::TOP+j0;
  if constexpr(S==8){((float4*)o)[0]=make_float4(mine[0],mine[1],mine[2],mine[3]);((float4*)o)[1]=make_float4(mine[4],mine[5],mine[6],mine[7]);}
  else if constexpr(S==4)*(float4*)o=make_float4(mine[0],mine[1],mine[2],mine[3]);
  else if constexpr(S==2)*(float2*)o=make_float2(mine[0],mine[1]);
  else o[0]=mine[0];}
}
template<int S,int TR,class Input,bool PH=false>
__global__ void __launch_bounds__(128) scores_sc(Input input,int start,const float* q,const float* rope,float* out){
 scores_sc_body<S,TR,false,Input,PH>(input,start,q,rope,out,threadIdx.x);}
template<int S,int TR,class Input,bool PH=false>
__global__ void __launch_bounds__(128,4) scores_sc_qt(Input input,int start,const float* q,const float* rope,float* out){
 scores_sc_body<S,TR,true,Input,PH>(input,start,q,rope,out,threadIdx.x);}
// GR8_ATTN_CTX2=3 / 4 (attn_ctx2): context_parts_heads<HG> with the per-slot work packed and branch-free. Prologue as
// context_parts_heads, but slot j's HG probabilities sit together (pm[j], 16 B smem loads), a slot whose row pointer is null
// gets p = 0 for every head, and the skip is a select: lo = p != 0 ? fmaf(p, x*s, lo) : lo, which is `if (!rows[j]) continue`
// + `if (p == 0) continue` (-0 and NaN p included). The gathered pairs of the next PB slots are loaded before the current PB
// are consumed (two register batches in flight). SP = slots per partition: 256 (P = PART partitions) gives every (row, head,
// dim) chain the same fmaf operands in the same slot order -> parts bit-identical to context_parts_heads.
// grid dim3(P*2, heads/HG, n), block 128.
template<int HG,int PB,int SP,class Input>
__global__ void __launch_bounds__(128) context_pk(Input input,int start,const float* probs,float* parts){
 static_assert(SP%(2*PB)==0&&HG%4==0,"SP multiple of 2*PB, HG multiple of 4");
 auto a=ga::resolve(input);if(!a.active)return;constexpr int D=256,THREADS=D/2,GROUPS=(D+127)/128,P=ga::TOP/SP;
 int row=blockIdx.z,r=start+row,h0=blockIdx.y*HG,part=blockIdx.x/(512/D),d0=(blockIdx.x%(512/D))*D,t=threadIdx.x;FOR_ROW(a,input,r);
 const int nh=min(HG,a.heads-h0);
 __shared__ uintptr_t rows[SP];__shared__ __align__(16) float pm[SP][HG];__shared__ float scale[SP*GROUPS];bool active=false;
 for(int j=t;j<SP;j+=THREADS){int slot=part*SP+j,id=a.indices[(int64_t)r*ga::TOP+slot];const uint8_t* src=nullptr;
  const bool idok=id>=0&&id<=a.positions[r];   // dead slots read no probabilities
  float p[HG];bool live=false;
  #pragma unroll
  for(int hh=0;hh<HG;hh++){p[hh]=idok&&hh<nh?probs[((int64_t)row*a.heads+h0+hh)*ga::TOP+slot]:0.f;live|=p[hh]!=0;}
  if(live)src=ga::kv_at(a,id);
  #pragma unroll
  for(int hh=0;hh<HG;hh++)pm[j][hh]=src?p[hh]:0.f;
  rows[j]=(uintptr_t)src;if(src)for(int g=0;g<GROUPS;g++)scale[j*GROUPS+g]=((const float*)(src+512))[d0/128+g];active|=src!=nullptr;}
 const bool any=__syncthreads_or(active);float lo[HG],hi[HG];
 #pragma unroll
 for(int hh=0;hh<HG;hh++){lo[hh]=0;hi[hh]=0;}
 const int d=d0+2*t,gs=(2*t)/128;
 if(any){uint16_t ra[PB],rb[PB];
  auto load=[&](uint16_t (&raw)[PB],int j0){
   #pragma unroll
   for(int u=0;u<PB;u++){const uint8_t* src=(const uint8_t*)rows[j0+u];raw[u]=src?*(const uint16_t*)(src+d):(uint16_t)0;}};
  auto use=[&](const uint16_t (&raw)[PB],int j0){
   #pragma unroll
   for(int u=0;u<PB;u++){const int j=j0+u;const float2 x=fp8_pair(raw[u]);const float sc=scale[j*GROUPS+gs];float p[HG];
    #pragma unroll
    for(int i=0;i<HG;i+=4){const float4 q4=*(const float4*)&pm[j][i];p[i]=q4.x;p[i+1]=q4.y;p[i+2]=q4.z;p[i+3]=q4.w;}
    #pragma unroll
    for(int hh=0;hh<HG;hh++){const float fl=fmaf(p[hh],x.x*sc,lo[hh]),fh=fmaf(p[hh],x.y*sc,hi[hh]);lo[hh]=p[hh]!=0?fl:lo[hh];hi[hh]=p[hh]!=0?fh:hi[hh];}}};
  load(ra,0);
  for(int j0=0;j0<SP;j0+=2*PB){load(rb,j0+PB);use(ra,j0);if(j0+2*PB<SP)load(ra,j0+2*PB);use(rb,j0+PB);}}
 #pragma unroll
 for(int hh=0;hh<HG;hh++)if(hh<nh){float* dst=parts+(((int64_t)row*a.heads+h0+hh)*P+part)*512;dst[d]=lo[hh];dst[d+1]=hi[hh];}
}
// GR8_ATTN_CTX3=1..4 (attn_ctx3): context_pk with less work per slot. Prologue as context_pk (pm[j] = slot j's HG
// probabilities, 0 for a slot whose KV row is not read), plus the slot's KV row index (kv_row; a slot that is not read points
// at row 0, which is loaded and never used), its scales (scl[g][j], g = the CTA's 128-dim groups) and, per batch of PB
// slots, one state from two warp ballots: 0 = no slot read (skipped), 1 = every slot read with all HG probabilities != 0
// (plain fmaf: no test), 2 = mixed (context_pk's select lo = p != 0 ? fmaf(p, x*s, lo) : lo; an unread slot has p = 0 for
// every head). Batches run in slot order up to the last one with a slot read, the next batch's pairs (32-bit registers, one
// read-only 16-bit load from the row index, no pointer test) loaded before the current batch is used. Per (row, head, dim)
// chain: the same fmaf operands in the same slot order as context_parts_heads, the same skips of unread slots and of
// p == 0 (-0 and NaN p as there) -> parts bit-identical. D = dims per CTA: 256 (two CTAs per partition, as
// context_parts_heads) or 512 (one, the prologue once per partition). grid dim3(P*512/D, ceil(heads/HG), n), block D/2.
// Rows are 32-bit: the dispatch (use_ctx3) needs page_count * page_size <= INT_MAX.
// SP = slots per partition, P = TOP/SP partitions: 256 (P = PART, the bytes above); 64 / 128 (GR8_ATTN_CTX3_SP, run_body
// only, E1): every (row, head, dim) chain runs over SP slots in slot order and merge_parts<P> sums the P partials in
// partition order, so the 2048-slot sum is regrouped (deterministic, not the same bits).
template<int HG,int PB,int D,int SP=256,class Input>
__global__ void __launch_bounds__(D/2) context_pk3(Input input,int start,const float* probs,float* parts){
 static_assert((D==256||D==512)&&(PB==16||PB==32)&&(HG==2||HG%4==0),"D 256 / 512, PB 16 / 32, HG 2 or a multiple of 4");
 static_assert(SP==64||SP==128||SP==256,"SP 64 / 128 / 256");
 auto a=ga::resolve(input);if(!a.active)return;constexpr int THREADS=D/2,GROUPS=D/128,NB=SP/PB,P=ga::TOP/SP;
 const int row=blockIdx.z,r=start+row,h0=blockIdx.y*HG,part=blockIdx.x/(512/D),d0=(blockIdx.x%(512/D))*D,t=threadIdx.x,lane=t&31;FOR_ROW(a,input,r);
 const int nh=min(HG,a.heads-h0);
 __shared__ __align__(16) float pm[SP][HG];__shared__ float scl[GROUPS][SP];__shared__ int kv_row[SP];__shared__ uint8_t state[NB];
 for(int j=t;j<SP;j+=THREADS){const int slot=part*SP+j,id=a.indices[(int64_t)r*ga::TOP+slot];   // j - lane: a multiple of 32
  const bool idok=id>=0&&id<=a.positions[r];   // dead slots read no probabilities
  float p[HG];bool live=false,full=true;
  #pragma unroll
  for(int hh=0;hh<HG;hh++){p[hh]=idok&&hh<nh?probs[((int64_t)row*a.heads+h0+hh)*ga::TOP+slot]:0.f;live|=p[hh]!=0;full&=hh>=nh||p[hh]!=0;}
  int kr=-1;if(live){const int pg=ga::page(a,id);if(pg>=0)kr=(int)((int64_t)pg*a.page_size+id%a.page_size);}   // kv_at's row (< 2^31: dispatch)
  const bool rd=kr>=0;full&=rd;
  #pragma unroll
  for(int hh=0;hh<HG;hh++)pm[j][hh]=rd?p[hh]:0.f;
  kv_row[j]=rd?kr:0;
  #pragma unroll
  for(int g=0;g<GROUPS;g++)scl[g][j]=rd?((const float*)((const uint8_t*)a.mla_cache+(int64_t)kr*ga::KV_BYTES+512))[d0/128+g]:0.f;
  const unsigned br=__ballot_sync(0xffffffffu,rd),bf=__ballot_sync(0xffffffffu,full);
  if(lane==0){
   #pragma unroll
   for(int h2=0;h2<32/PB;h2++){const unsigned m=PB==32?0xffffffffu:0xffffu<<(16*h2);state[j/PB+h2]=(bf&m)==m?1:(br&m)?2:0;}}}
 __syncthreads();
 int nb=0;
 #pragma unroll
 for(int b=0;b<NB;b++)if(state[b])nb=b+1;
 float lo[HG],hi[HG];
 #pragma unroll
 for(int hh=0;hh<HG;hh++){lo[hh]=0;hi[hh]=0;}
 const int d=d0+2*t,gs=(2*t)/128;
 if(nb){const uint8_t* kvd=(const uint8_t*)a.mla_cache+d;const float* sg=scl[gs];uint32_t ra[PB],rb[PB];
  auto load=[&](uint32_t (&raw)[PB],int b){
   #pragma unroll
   for(int u=0;u<PB;u++)raw[u]=__ldg((const unsigned short*)(kvd+(int64_t)kv_row[b*PB+u]*ga::KV_BYTES));};
  auto probs_of=[&](int j,float (&p)[HG]){
   if constexpr(HG==2){const float2 q2=*(const float2*)&pm[j][0];p[0]=q2.x;p[1]=q2.y;}
   else{
    #pragma unroll
    for(int i=0;i<HG;i+=4){const float4 q4=*(const float4*)&pm[j][i];p[i]=q4.x;p[i+1]=q4.y;p[i+2]=q4.z;p[i+3]=q4.w;}}};
  auto use=[&](const uint32_t (&raw)[PB],int b){
   const int st=state[b];
   if(st==1){
    #pragma unroll
    for(int u=0;u<PB;u++){const int j=b*PB+u;const float2 x=fp8_pair((uint16_t)raw[u]);const float sc=sg[j];float p[HG];probs_of(j,p);
     #pragma unroll
     for(int hh=0;hh<HG;hh++){lo[hh]=fmaf(p[hh],x.x*sc,lo[hh]);hi[hh]=fmaf(p[hh],x.y*sc,hi[hh]);}}}
   else if(st==2){
    #pragma unroll
    for(int u=0;u<PB;u++){const int j=b*PB+u;const float2 x=fp8_pair((uint16_t)raw[u]);const float sc=sg[j];float p[HG];probs_of(j,p);
     #pragma unroll
     for(int hh=0;hh<HG;hh++){const float fl=fmaf(p[hh],x.x*sc,lo[hh]),fh=fmaf(p[hh],x.y*sc,hi[hh]);lo[hh]=p[hh]!=0?fl:lo[hh];hi[hh]=p[hh]!=0?fh:hi[hh];}}}};
  load(ra,0);
  for(int b=0;b<nb;b+=2){if(b+1<nb)load(rb,b+1);use(ra,b);if(b+2<nb)load(ra,b+2);if(b+1<nb)use(rb,b+1);}}
 #pragma unroll
 for(int hh=0;hh<HG;hh++)if(hh<nh){float* dst=parts+(((int64_t)row*a.heads+h0+hh)*P+part)*512;dst[d]=lo[hh];dst[d+1]=hi[hh];}
}
// scores_tr = scores_heads with all HG head dot products held per lane (ILP across
// heads), then ONE transpose reduction over xor offsets 16,8,4,2,1: each stage keeps half of the remaining heads, so every
// head still gets the warp_sum tree (same operands, same add order per stage; a+b commutes bitwise) -> 16+ shuffles instead
// of 5*HG. Same products, same fmaf order, same dead-slot skip as scores_heads -> bit-identical.
// Needs nh == HG (dispatch guards a.heads % HG == 0). Lane hh stores head h0+hh.
template<int HG,class Input>
__global__ void __launch_bounds__(256) scores_tr(Input input,int start,const float* q,const float* rope,float* out){
 static_assert(HG==16||HG==8||HG==4||HG==2,"HG power of two <= 16");
 auto a=ga::resolve(input);if(!a.active)return;constexpr int WARPS=8;int row=blockIdx.z,r=start+row,h0=blockIdx.y*HG,j=blockIdx.x*WARPS+threadIdx.x/32,lane=threadIdx.x&31;FOR_ROW(a,input,r);
 int id=a.indices[(int64_t)r*ga::TOP+j];const uint8_t* v=(id>=0&&id<=a.positions[r]&&CP_MINE(a,id))?ga::kv_at(a,id):nullptr;
 float mine=-INFINITY;
 if(v){float scale[4],kv[16],rv[2],s[HG];   // v is warp-uniform (one slot per warp), so the shuffles below are full-warp
  #pragma unroll
  for(int g=0;g<4;g++)scale[g]=((const float*)(v+512))[g];
  #pragma unroll
  for(int k=0;k<16;k++)kv[k]=ga::e4(v[lane+k*32])*scale[k/4];
  #pragma unroll
  for(int k=0;k<2;k++)rv[k]=__bfloat162float(((const bf16*)(v+528))[lane+k*32]);
  #pragma unroll
  for(int hh=0;hh<HG;hh++){const int h=h0+hh;float sum=0;
   #pragma unroll
   for(int k=0;k<16;k++)sum=fmaf(__ldg(q+((int64_t)row*a.heads+h)*512+lane+k*32),kv[k],sum);
   #pragma unroll
   for(int k=0;k<2;k++)sum=fmaf(__ldg(rope+((int64_t)row*a.heads+h)*64+lane+k*32),rv[k],sum);
   s[hh]=sum;}
  // transpose reduction: stage o keeps the upper half of the remaining heads if lane bit o is set, else the lower half
  #pragma unroll
  for(int n=HG,o=16;o>=1;o>>=1){
   if(n>1){const bool hi=(lane&o)!=0;const int hf=n/2;
    #pragma unroll
    for(int i=0;i<HG/2;i++)if(i<hf){float keep=hi?s[i+hf]:s[i],send=hi?s[i]:s[i+hf];s[i]=keep+__shfl_xor_sync(0xffffffff,send,o);}
    n=hf;}
   else s[0]+=__shfl_xor_sync(0xffffffff,s[0],o);}
  // owner lane of head (lane % HG): the lane whose bit choices select that head (low bits 0)
  int src=0;{const int want=lane%HG;int n=HG,o=16,hd=0;while(n>1){const int hf=n/2;if(want>=hd+hf){src|=o;hd+=hf;}n=hf;o>>=1;}}
  const float got=__shfl_sync(0xffffffff,s[0],src);
  if(lane<HG)mine=got*(1.f/16);}
 if(lane<HG)out[((int64_t)row*a.heads+h0+lane)*ga::TOP+j]=mine;
}
// context_parts_tile<HG,D> = context_parts_heads with D dims per CTA (D/2 threads) and HG heads per CTA.
// grid dim3(PART*(512/D), ceil(heads/HG), n), block D/2. Same slot order, same fmaf order, same p == 0 / dead-slot skips,
// same part layout -> parts (and merge_parts output) bit-identical.
template<int HG,int D,class Input>
__global__ void __launch_bounds__(128) context_parts_tile(Input input,int start,const float* probs,float* parts){
 static_assert(D==64||D==128||D==256,"D in {64,128,256}");
 auto a=ga::resolve(input);if(!a.active)return;constexpr int THREADS=D/2,GROUPS=(D+127)/128,PB=16,SL=512/D;
 int row=blockIdx.z,r=start+row,h0=blockIdx.y*HG,part=blockIdx.x/SL,d0=(blockIdx.x%SL)*D,t=threadIdx.x;FOR_ROW(a,input,r);
 const int nh=min(HG,a.heads-h0);
 __shared__ uintptr_t rows[256];__shared__ float probability[HG][256],scale[256*GROUPS];bool active=false;
 for(int j=t;j<256;j+=THREADS){int slot=part*256+j,id=a.indices[(int64_t)r*ga::TOP+slot];const uint8_t* src=nullptr;
  const bool idok=id>=0&&id<=a.positions[r];   // dead slots read no probabilities
  bool live=false;for(int hh=0;hh<HG;hh++){float p=idok&&hh<nh?probs[((int64_t)row*a.heads+h0+hh)*ga::TOP+slot]:0.f;probability[hh][j]=p;live|=p!=0;}
  if(live)src=ga::kv_at(a,id);
  rows[j]=(uintptr_t)src;if(src)for(int g=0;g<GROUPS;g++)scale[j*GROUPS+g]=((const float*)(src+512))[d0/128+g];active|=src!=nullptr;}
 const bool any=__syncthreads_or(active);float lo[HG],hi[HG];
 #pragma unroll
 for(int hh=0;hh<HG;hh++){lo[hh]=0;hi[hh]=0;}
 int d=d0+2*t;
 if(any)for(int j0=0;j0<256;j0+=PB){uint16_t raw[PB];
  #pragma unroll
  for(int u=0;u<PB;u++){const uint8_t* src=(const uint8_t*)rows[j0+u];raw[u]=src?*(const uint16_t*)(src+d):(uint16_t)0;}
  #pragma unroll
  for(int u=0;u<PB;u++){const int j=j0+u;if(!rows[j])continue;float2 x=fp8_pair(raw[u]);float s=scale[j*GROUPS+(2*t)/128];
   #pragma unroll
   for(int hh=0;hh<HG;hh++){float p=probability[hh][j];if(p==0)continue;lo[hh]=fmaf(p,x.x*s,lo[hh]);hi[hh]=fmaf(p,x.y*s,hi[hh]);}}}
 #pragma unroll
 for(int hh=0;hh<HG;hh++)if(hh<nh){float* dst=parts+(((int64_t)row*a.heads+h0+hh)*ga::PART+part)*512;dst[d]=lo[hh];dst[d+1]=hi[hh];}
}
template<class Input>
__global__ void index_scores(Input input,int start,const half* q,const float* weights,float* scores,int* ids){
 auto a=ga::resolve(input);if(!a.active)return;using namespace nvcuda;int row=blockIdx.y,r=start+row,t0=blockIdx.x*32,t=threadIdx.x,warp=t/32;FOR_ROW(a,input,r);
 __shared__ __align__(32) half key[32*128];__shared__ __align__(32) float products[32*32];__shared__ float scales[32];__shared__ uintptr_t pointers[32];
 if(t0>a.positions[r]){if(t<32&&t0+t<a.context){scores[(int64_t)row*a.context+t0+t]=-INFINITY;ids[(int64_t)row*a.context+t0+t]=t0+t;}return;}
 if(t<32){float scale=0;const uint8_t* p=t0+t<a.live_context&&CP_MINE(a,t0+t)?ga::ik_at(a,t0+t,scale):nullptr;pointers[t]=(uintptr_t)p;scales[t]=scale;}__syncthreads();
 for(int i=t;i<32*128/2;i+=128){int col=i/64,d=(i%64)*2;const uint8_t* p=(const uint8_t*)pointers[col];uint16_t code=p?*(const uint16_t*)(p+d):0;((__half2*)key)[i]=__half2(__nv_cvt_fp8x2_to_halfraw2(code,__NV_E4M3));}
 __syncthreads();
 wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major>A;wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major>B;wmma::fragment<wmma::accumulator,16,16,16,float>C;wmma::fill_fragment(C,0.f);int hs=(warp/2)*16,ts=(warp%2)*16;
 for(int k=0;k<128;k+=16){wmma::load_matrix_sync(A,q+((int64_t)row*32+hs)*128+k,128);wmma::load_matrix_sync(B,key+ts*128+k,128);wmma::mma_sync(C,A,B,C);}
 wmma::store_matrix_sync(products+hs*32+ts,C,32,wmma::mem_row_major);__syncthreads();
 if(t<32&&t0+t<a.context){float score=0;for(int h=0;h<32;h++)score+=fmaxf(products[h*32+t],0.f)*weights[row*32+h];score*=scales[t];if(t0+t>a.positions[r]||!CP_MINE(a,t0+t))score=-INFINITY;else if(!isfinite(score)){ga::error(a,4);score=-INFINITY;}scores[(int64_t)row*a.context+t0+t]=score;ids[(int64_t)row*a.context+t0+t]=t0+t;}
}
}
#include "decode-flip.cuh"
static bool use_small_decode(const G6Attn& a){return a.rows<=3&&a.context<=2048;}
// Sparse MLA consumes exactly TOP selected logical IDs, not a.context keys.
// Keep fresh-index dispatch unchanged; widen only decode/verify reuse rows.
static bool use_fixed_k_decode(const G6Attn& a){
 #ifdef DEC_S2
 if(a.s2_batched&&a.rows<=G6_MX)return true;   // batched S2 step: rows of several sessions, up to the row budget
 #endif
 return use_small_decode(a)||(a.reuse_indices&&a.rows<=8);
}
// GR8_ATTN_FLIP=1 (default off; design-review delta). Attention-only dispatch, launch_index_scores is untouched.
// Eligible = n 1..8, heads 1..22, and either the release fixed-K request (small decode or reuse rows <= 8, or a batched
// S2 step) or a genuine fresh decode request (a.rows <= 3). Eligible at context >= GR8_ATTN_FLIP_MIN (default 2048, the
// GB10 sweep: byte-identical and <= 0.9x cold+hot for M=1..3 from 2048 up): scores_flip<11>, context_flip M=1 HG=4 D=128
// CB=32 / M>=2 HG=11 D=128 CB=16. The flip is chosen before the head-batched kernels (attn_heads), which stay the default
// and serve every call the flip does not take. Under GR8_ATTN_HEADS=0, a fresh decode request below the minimum runs the
// fixed-K HG=1 forms, and requests whose reference is the generic context_parts keep scalar e4() (GENERIC_E4); with the
// head-batched kernels as the reference every request uses fp8_pair. Scores need no mode: latent(v,d) is literally
// e4(v[d])*scale[d/128]. Off = the dispatch below, unchanged.
// Host-side, read once (set before graph capture); attn_flip_min() < 0 = off. Tests may assign it.
static int& attn_flip_min(){static int v=knob::attn_flip()?int(knob::attn_flip_min()):-1;return v;}
// GR8_ATTN_D256=1 (default off; only with GR8_ATTN_FLIP): context M=1 HG4 D256 CB16, M>=2 HG8 D256 CB16 + idok dead-slot skip.
// Host-side, read once (set before graph capture). Tests may assign it.
static int& attn_flip_d256(){static int v=knob::attn_d256();return v;}
static bool attn_flip_eligible(const G6Attn& a,int n){
 return attn_flip_min()>=0&&n>=1&&n<=8&&a.heads>=1&&a.heads<=22&&(use_fixed_k_decode(a)||(a.rows>0&&a.rows<=3));}
static bool use_attn_flip(const G6Attn& a,int n){return attn_flip_eligible(a,n)&&a.context>=attn_flip_min();}
static bool use_long_fixed_k(const G6Attn& a,int n){return attn_flip_eligible(a,n)&&!use_fixed_k_decode(a);}
// GR8_IDX_VEC=1 (default 0 = off): the per-row index_scores loads its key tile with 16 B
// vector loads (page_size >= 32). Byte-exact; -7.65 ms/step index_scores @S4 L32K. Host-side, read once (set
// before graph capture). Short decode (use_small_decode) and the DEC_IDX31 shared-key kernel are not affected.
static bool idx_vec(){return knob::idx_vec();}
template<class Input>
static void launch_index_scores(const G6Attn& a,Input input,Work& w,int start,int n,cudaStream_t st){
 if(use_small_decode(a))small_decode::index_scores<<<dim3((a.context+31)/32,n),128,0,st>>>(input,start,w.iq,w.weights,w.si,w.ii);
 else if(idx_vec())index_scores<1><<<dim3((a.context+31)/32,n),128,0,st>>>(input,start,w.iq,w.weights,w.si,w.ii);
 else index_scores<<<dim3((a.context+31)/32,n),128,0,st>>>(input,start,w.iq,w.weights,w.si,w.ii);
}
// GR8_ATTN_HEADS=0 (read at launch / capture): the per-head scores_direct / context_parts kernels (A/B and tests)
static bool attn_heads(){return knob::attn_heads();}
// GR8_ATTN_SC2=<S> (default 0 = off): scores_sc<S> (S = 1, 2, 4, 8 slots per warp) for the scores_heads<16> calls;
// GR8_ATTN_SC2_TR=1 adds scores_tr's transpose reduction, =2 the same reduction eager (S = 8: TR 0 only).
// Bit-identical. Host-side, read once (before graph capture). Tests may assign them.
static int& attn_sc2(){static int v=int(knob::attn_sc2());return v;}
static int& attn_sc2_tr(){static int v=int(knob::attn_sc2_tr());return v;}
// GR8_ATTN_SC2_PART (default 1; 0 = off): scores_sc (and QT) also serve a head count that is not a multiple of 16 (TP=6:
// 10 / 11 heads a rank, TP=3: 21 / 22) with the partial-head forms (scores_sc_body PH), bit-identical to scores_heads<16>.
// Off: those calls take scores_heads<16> as before. Inert at a multiple of 16 (TP=4). Host-side, read once. Tests may assign it.
static int& attn_sc2_part(){static int v=knob::attn_sc2_part();return v;}
static bool sc2_heads_ok(const G6Attn& a){return a.heads%16==0||(attn_sc2_part()&&a.heads>0);}
// qt (run_body, attn.cu use_qt: S = 2 / 4 only): scores_sc_qt on the lane-transposed q / rope
// PH: the partial-head instantiations (a.heads % 16 != 0, GR8_ATTN_SC2_PART), grid y = ceil(heads / 16)
template<int S,bool PH,class Input>
static void launch_scores_qt(const G6Attn& a,Input input,Work& w,int start,int n,cudaStream_t st){
 const dim3 g(TOP/(4*S),(a.heads+15)/16,n);const int tr=attn_sc2_tr();
 if(tr==2)small_decode::scores_sc_qt<S,2,Input,PH><<<g,128,0,st>>>(input,start,w.q,w.rope,w.prob);
 else if(tr)small_decode::scores_sc_qt<S,1,Input,PH><<<g,128,0,st>>>(input,start,w.q,w.rope,w.prob);
 else small_decode::scores_sc_qt<S,0,Input,PH><<<g,128,0,st>>>(input,start,w.q,w.rope,w.prob);
}
template<bool PH,class Input>
static void launch_scores_sc_ph(const G6Attn& a,Input input,Work& w,int start,int n,cudaStream_t st,bool qt){
 if(qt){if(attn_sc2()==2)launch_scores_qt<2,PH>(a,input,w,start,n,st);else launch_scores_qt<4,PH>(a,input,w,start,n,st);return;}
 const int y=(a.heads+15)/16;
 const dim3 b(128),g1(TOP/4,y,n),g2(TOP/8,y,n),g4(TOP/16,y,n),g8(TOP/32,y,n);
 const int tr=attn_sc2_tr();
 switch(attn_sc2()){
  case 1:if(tr==2)small_decode::scores_sc<1,2,Input,PH><<<g1,b,0,st>>>(input,start,w.q,w.rope,w.prob);
   else if(tr)small_decode::scores_sc<1,1,Input,PH><<<g1,b,0,st>>>(input,start,w.q,w.rope,w.prob);
   else small_decode::scores_sc<1,0,Input,PH><<<g1,b,0,st>>>(input,start,w.q,w.rope,w.prob);break;
  case 2:if(tr==2)small_decode::scores_sc<2,2,Input,PH><<<g2,b,0,st>>>(input,start,w.q,w.rope,w.prob);
   else if(tr)small_decode::scores_sc<2,1,Input,PH><<<g2,b,0,st>>>(input,start,w.q,w.rope,w.prob);
   else small_decode::scores_sc<2,0,Input,PH><<<g2,b,0,st>>>(input,start,w.q,w.rope,w.prob);break;
  case 8:small_decode::scores_sc<8,0,Input,PH><<<g8,b,0,st>>>(input,start,w.q,w.rope,w.prob);break;
  default:if(tr==2)small_decode::scores_sc<4,2,Input,PH><<<g4,b,0,st>>>(input,start,w.q,w.rope,w.prob);
   else if(tr)small_decode::scores_sc<4,1,Input,PH><<<g4,b,0,st>>>(input,start,w.q,w.rope,w.prob);
   else small_decode::scores_sc<4,0,Input,PH><<<g4,b,0,st>>>(input,start,w.q,w.rope,w.prob);break;}
}
// heads % 16 == 0 (TP=4, ...): the full-group kernels, as before; otherwise the partial-head forms (sc2_heads_ok)
template<class Input>
static void launch_scores_sc(const G6Attn& a,Input input,Work& w,int start,int n,cudaStream_t st,bool qt=false){
 if(a.heads%16==0)launch_scores_sc_ph<false>(a,input,w,start,n,st,qt);
 else launch_scores_sc_ph<true>(a,input,w,start,n,st,qt);
}
// GR8_ATTN_CTX2 (attn.cu attn_ctx2): the context_parts_heads calls (HG 4 up to 4 rows, then 8) under the same head groups
template<int HG,class Input>
static void launch_ctx2(const G6Attn& a,Input input,Work& w,int start,int n,cudaStream_t st){
 const dim3 g(PART*2,(a.heads+HG-1)/HG,n);
 switch(attn_ctx2()){
  case 1:small_decode::context_parts_heads<HG,32><<<g,128,0,st>>>(input,start,w.prob,w.parts);break;
  case 2:small_decode::context_parts_heads<HG,64><<<g,128,0,st>>>(input,start,w.prob,w.parts);break;
  case 3:small_decode::context_pk<HG,16,256><<<g,128,0,st>>>(input,start,w.prob,w.parts);break;
  case 4:small_decode::context_pk<HG,32,256><<<g,128,0,st>>>(input,start,w.prob,w.parts);break;}
}
// GR8_ATTN_CTX3 (attn.cu attn_ctx3): context_pk3 on the context_parts_heads calls; head group GR8_ATTN_CTX3_HG, else as there.
// Only while every KV row index fits in 32 bits (page() bounds a page by the fixed page_count).
// Not with remote rows: their slots sit outside mla_cache.
static bool use_ctx3(const G6Attn& a){return attn_ctx3()&&!a.rt_page&&(int64_t)a.page_count*a.page_size<=INT_MAX;}
// SP slots per partition (sp, ctx_sp below): P = TOP/SP partitions; SP = 256 is PART, the launches as before
template<int HG,int SP,class Input>
static void launch_ctx3_sp(const G6Attn& a,Input input,Work& w,int start,int n,cudaStream_t st){
 const int y=(a.heads+HG-1)/HG;constexpr int P=TOP/SP;
 switch(attn_ctx3()){
  case 1:small_decode::context_pk3<HG,16,256,SP><<<dim3(P*2,y,n),128,0,st>>>(input,start,w.prob,w.parts);break;
  case 2:small_decode::context_pk3<HG,32,256,SP><<<dim3(P*2,y,n),128,0,st>>>(input,start,w.prob,w.parts);break;
  case 3:small_decode::context_pk3<HG,16,512,SP><<<dim3(P,y,n),256,0,st>>>(input,start,w.prob,w.parts);break;
  default:small_decode::context_pk3<HG,32,512,SP><<<dim3(P,y,n),256,0,st>>>(input,start,w.prob,w.parts);break;}
}
template<int HG,class Input>
static void launch_ctx3_hg(const G6Attn& a,Input input,Work& w,int start,int n,cudaStream_t st,int sp){
 if(sp==64)launch_ctx3_sp<HG,64>(a,input,w,start,n,st);
 else if(sp==128)launch_ctx3_sp<HG,128>(a,input,w,start,n,st);
 else launch_ctx3_sp<HG,256>(a,input,w,start,n,st);
}
template<class Input>
static void launch_ctx3(const G6Attn& a,Input input,Work& w,int start,int n,cudaStream_t st,int sp=256){
 switch(attn_ctx3_hg()?attn_ctx3_hg():n>=5?8:4){
  case 2:launch_ctx3_hg<2>(a,input,w,start,n,st,sp);break;
  case 8:launch_ctx3_hg<8>(a,input,w,start,n,st,sp);break;
  default:launch_ctx3_hg<4>(a,input,w,start,n,st,sp);break;}
}
// GR8_ATTN_P11 decode tile retune. Host-side, read once (set before graph
// capture). GR8_ATTN_P11=0 -> the old head-batched path (scores_heads<16>, context_parts_heads HG4/HG8 x D256), unchanged.
// Default OFF (GR8_ATTN_P11=1 turns it on). Compile switch: -DATTN_P11_OLD=1 forces the old path regardless of the env. Only the attn_heads() branch
// changes; GR8_ATTN_FLIP still takes precedence, GR8_ATTN_HEADS=0 still selects the per-head kernels.
static bool attn_p11(){
 #if defined(ATTN_P11_OLD) && ATTN_P11_OLD
 return false;
 #else
 return knob::attn_p11();
 #endif
}
// GR8_ATTN_CTX3_SP (attn.cu attn_ctx3_sp, E1): the slots per partition of a run_body call's context, 64 / 128 exactly when
// launch_context_parts reaches launch_ctx3 (no FLIP, the head-batched kernels, no P11, use_ctx3); 256 otherwise. run_body
// passes it to launch_context_parts and merges TOP/sp partials; the CP calls keep 256 (their merges assume PART).
static int ctx_sp(const G6Attn& a,int n){
 #if G6_KV4
 (void)a;(void)n;return 256;
 #else
 return attn_ctx3_sp()&&!use_attn_flip(a,n)&&attn_heads()&&!attn_p11()&&use_ctx3(a)?attn_ctx3_sp():256;
 #endif
}
// Context length used to pick the tile. a.context at capture is the cell window, not the live length (live_context is
// device-side), so graphs captured per context bucket get the right row; GR8_ATTN_TILE_L=<L> overrides (A/B, tests).
static int attn_tile_l(const G6Attn& a){static const int f=int(knob::attn_tile_l());return f>0?f:a.context;}
// Shape -> context tile (HG heads x D dims per CTA), from the measured best-unfused rows (GB10, 2000 MHz, 16 heads
// per rank, n = S x 3 batched S2 rows). Rows: n >= 10 (S4), 5..9 (S2), <= 4 (S1). Columns: L <= 512 | L <= 1024 | L > 1024.
// Measured: S4 L512 4x128, L1024 8x128, L2048 16x256, L32768 16x256; S2 L512 8x128, L2048 8x256; S1 L512 4x128, L2048 4x256.
// Not measured (nearest measured row used): S2/S1 at L 513..1024 (= their L512 tile), S2/S1 above 2048 (= their L2048 tile).
// GR8_ATTN_CTX_TILE=<HG>x<D> (e.g. 16x256) forces one tile for every shape (A/B); the table lists the tiles launch_context_tile has.
struct CtxTile{int hg,d;};
static CtxTile attn_ctx_tile(const G6Attn& a,int n){
 static const int force=[]{const auto& t=knob::attn_ctx_tile();if(!t)return 0;   // "<HG>x<D>", one of the table's values
  const char* e=t->c_str();const int h=atoi(e);const char* x=e;while(*x!='x')x++;return h*1000+atoi(x+1);}();
 if(force)return {force/1000,force%1000};
 static const CtxTile T[3][3]={
  /* S1 n<=4  */ {{4,128},{4,128},{4,256}},
  /* S2 n5..9 */ {{8,128},{8,128},{8,256}},
  /* S4 n>=10 */ {{4,128},{8,128},{16,256}}};
 const int L=attn_tile_l(a),ri=n>=10?2:n>=5?1:0,ci=L<=512?0:L<=1024?1:2;return T[ri][ci];
}
template<int HG,int D,class Input>
static void launch_ctx_tile1(const G6Attn& a,Input input,Work& w,int start,int n,cudaStream_t st){
 small_decode::context_parts_tile<HG,D><<<dim3(PART*(512/D),(a.heads+HG-1)/HG,n),D/2,0,st>>>(input,start,w.prob,w.parts);
}
template<class Input>
static void launch_context_tile(const G6Attn& a,Input input,Work& w,int start,int n,cudaStream_t st){
 const CtxTile c=attn_ctx_tile(a,n);
 switch(c.hg*1000+c.d){
  case 2128:launch_ctx_tile1<2,128>(a,input,w,start,n,st);break;
  case 4128:launch_ctx_tile1<4,128>(a,input,w,start,n,st);break;
  case 8128:launch_ctx_tile1<8,128>(a,input,w,start,n,st);break;
  case 16128:launch_ctx_tile1<16,128>(a,input,w,start,n,st);break;
  case 4256:launch_ctx_tile1<4,256>(a,input,w,start,n,st);break;
  case 16256:launch_ctx_tile1<16,256>(a,input,w,start,n,st);break;
  case 4064:launch_ctx_tile1<4,64>(a,input,w,start,n,st);break;
  case 8064:launch_ctx_tile1<8,64>(a,input,w,start,n,st);break;
  default:launch_ctx_tile1<8,256>(a,input,w,start,n,st);break;}   // 8x256 = the old n >= 5 tile
}
// The head-batched kernels do the arithmetic of both the fixed-K and the generic kernels (mla_scores, context_parts:
// the same products, FMA order and skips per (row, head, slot)), so they serve every decode call: verify rows of a fresh
// index and contexts past 2048 included.
template<class Input>   // qt: run_body only (attn.cu use_qt, which implies the scores_sc branch below)
static void launch_mla_scores(const G6Attn& a,Input input,Work& w,int start,int n,cudaStream_t st,bool qt=false){
 #if G6_KV4   // the INT4 row has the head-batched kernel and the generic one; the FP8-only variants below are not used
 if(attn_heads()){constexpr int HG=16;kv4::scores_heads<HG><<<dim3(TOP/8,(a.heads+HG-1)/HG,n),256,0,st>>>(input,start,w.q,w.rope,w.prob);}
 else mla_scores<<<dim3(TOP/8,a.heads,n),256,0,st>>>(input,start,w.q,w.rope,w.prob);
 return;
 #endif
 if(use_attn_flip(a,n))small_decode::scores_flip<Input,11><<<dim3(TOP/8,(a.heads+10)/11,n),256,0,st>>>(input,start,w.q,w.rope,w.prob);
 else if(attn_heads()&&attn_p11()&&a.heads%16==0)small_decode::scores_tr<16><<<dim3(TOP/8,a.heads/16,n),256,0,st>>>(input,start,w.q,w.rope,w.prob);
 else if(attn_heads()&&attn_sc2()&&sc2_heads_ok(a))launch_scores_sc(a,input,w,start,n,st,qt);
 else if(attn_heads()){constexpr int HG=16;small_decode::scores_heads<HG><<<dim3(TOP/8,(a.heads+HG-1)/HG,n),256,0,st>>>(input,start,w.q,w.rope,w.prob);}
 else if(use_long_fixed_k(a,n))small_decode::scores_flip<Input,1><<<dim3(TOP/8,a.heads,n),256,0,st>>>(input,start,w.q,w.rope,w.prob);
 else if(use_fixed_k_decode(a))small_decode::scores_direct<<<dim3(TOP/8,a.heads,n),256,0,st>>>(input,start,w.q,w.rope,w.prob);
 else mla_scores<<<dim3(TOP/8,a.heads,n),256,0,st>>>(input,start,w.q,w.rope,w.prob);
}
// sp (run_body, ctx_sp): context_pk3's slots per partition; 256 = PART partitions (every other caller)
template<class Input>
static void launch_context_parts(const G6Attn& a,Input input,Work& w,int start,int n,cudaStream_t st,int sp=256){
 #if G6_KV4
 if(!attn_heads())context_parts<<<dim3(PART,a.heads,n),256,0,st>>>(input,start,w.prob,w.parts);
 else if(n>=5){constexpr int HG=8;kv4::context_parts_heads<HG><<<dim3(PART*2,(a.heads+HG-1)/HG,n),128,0,st>>>(input,start,w.prob,w.parts);}
 else{constexpr int HG=4;kv4::context_parts_heads<HG><<<dim3(PART*2,(a.heads+HG-1)/HG,n),128,0,st>>>(input,start,w.prob,w.parts);}
 return;
 #endif
 if(use_attn_flip(a,n)){const bool fk=use_fixed_k_decode(a),ge4=!fk&&!attn_heads();   // ge4: the reference is the generic context_parts
  if(attn_flip_d256()){
   if(n==1){if(!ge4)small_decode::context_flip<Input,4,256,16,false,false><<<dim3(PART*2,(a.heads+3)/4,n),128,0,st>>>(input,start,w.prob,w.parts);
    else small_decode::context_flip<Input,4,256,16,true,false><<<dim3(PART*2,(a.heads+3)/4,n),128,0,st>>>(input,start,w.prob,w.parts);}
   else{if(!ge4)small_decode::context_flip<Input,8,256,16,false,true><<<dim3(PART*2,(a.heads+7)/8,n),128,0,st>>>(input,start,w.prob,w.parts);
    else small_decode::context_flip<Input,8,256,16,true,true><<<dim3(PART*2,(a.heads+7)/8,n),128,0,st>>>(input,start,w.prob,w.parts);}
  }else if(!ge4){
   if(n==1)small_decode::context_flip<Input,4,128,32,false><<<dim3(PART*4,(a.heads+3)/4,n),64,0,st>>>(input,start,w.prob,w.parts);
   else small_decode::context_flip<Input,11,128,16,false><<<dim3(PART*4,(a.heads+10)/11,n),64,0,st>>>(input,start,w.prob,w.parts);
  }else{
   if(n==1)small_decode::context_flip<Input,4,128,32,true><<<dim3(PART*4,(a.heads+3)/4,n),64,0,st>>>(input,start,w.prob,w.parts);
   else small_decode::context_flip<Input,11,128,16,true><<<dim3(PART*4,(a.heads+10)/11,n),64,0,st>>>(input,start,w.prob,w.parts);
  }
  return;}
 if(attn_heads()&&attn_p11()){launch_context_tile(a,input,w,start,n,st);return;}
 if(attn_heads()){
  // head group by rows (measured on GB10, contexts 500 and 2048): 4 heads per block up to 4 rows, then 8
  if(use_ctx3(a)){launch_ctx3(a,input,w,start,n,st,sp);return;}
  if(attn_ctx2()){if(n>=5)launch_ctx2<8>(a,input,w,start,n,st);else launch_ctx2<4>(a,input,w,start,n,st);return;}
  if(n>=5){constexpr int HG=8;small_decode::context_parts_heads<HG><<<dim3(PART*2,(a.heads+HG-1)/HG,n),128,0,st>>>(input,start,w.prob,w.parts);}
  else{constexpr int HG=4;small_decode::context_parts_heads<HG><<<dim3(PART*2,(a.heads+HG-1)/HG,n),128,0,st>>>(input,start,w.prob,w.parts);}
  return;}
 if(use_long_fixed_k(a,n)){small_decode::context_flip<Input,1,256,8,true><<<dim3(PART*2,a.heads,n),128,0,st>>>(input,start,w.prob,w.parts);return;}
 #if defined(DEC_CPB) && DEC_CPB
 if(use_fixed_k_decode(a))small_decode::context_parts_cpb<<<dim3(PART*2,a.heads,n),128,0,st>>>(input,start,w.prob,w.parts);
 #else
 if(use_fixed_k_decode(a))small_decode::context_parts<<<dim3(PART*2,a.heads,n),128,0,st>>>(input,start,w.prob,w.parts);
 #endif
 else context_parts<<<dim3(PART,a.heads,n),256,0,st>>>(input,start,w.prob,w.parts);
}
template<class Input>   // tests: the DEC_CPB fixed-K launch regardless of the build flag
static void launch_context_parts_cpb(const G6Attn& a,Input input,Work& w,int start,int n,cudaStream_t st){
 small_decode::context_parts_cpb<<<dim3(PART*2,a.heads,n),128,0,st>>>(input,start,w.prob,w.parts);
}
