// Self-contained --col-shard mode for test_engram_gpu: no model files or cmap.
#include "runtime/ds41_v2_engram.cuh"
namespace col_shard_test {
namespace eg = ds41::engram;
namespace wire = ds41::v2::engram_wire;
using B = __nv_bfloat16;
#define CS_CUDA(call) do { auto rc=(call); if(rc!=cudaSuccess){fprintf(stderr,"col_shard CUDA: %s: %s\n",#call,cudaGetErrorString(rc));exit(2);} } while(0)
#define CS_CHECK(x) do { if(!(x)){fprintf(stderr,"col_shard failed line %d: %s\n",__LINE__,#x);exit(1);} } while(0)
template<class T> struct Buffer {
    T* p=nullptr;
    explicit Buffer(size_t n){CS_CUDA(cudaMalloc(&p,n*sizeof(T)));}
    ~Buffer(){cudaFree(p);}
    void put(const std::vector<T>& v){CS_CUDA(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> get(size_t n){std::vector<T> v(n);CS_CUDA(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
__global__ void sum(const B* p,B* out,size_t n) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x){
        float x=0.f;for(int rank=0;rank<4;++rank)x=__fadd_rn(x,__bfloat162float(p[size_t(rank)*n+i]));
        out[i]=__float2bfloat16_rn(x);
    }
}
inline int run() {
    size_t free=0,total=0;CS_CUDA(cudaMemGetInfo(&free,&total));
    printf("col_shard GPU free_bytes=%zu total_bytes=%zu (test allocations < 64 MiB)\n",free,total);
    if(free<256ull*1024*1024){fprintf(stderr,"insufficient free GPU memory\n");return 77;}
    uint64_t compared=0,negative_zeros=0;
    eg::HashParams hp{};
    for(int e=0;e<2;++e)for(int j=0;j<24;++j){hp.prime[e][j]=257+j*3+e;hp.offset[e][j]=j?hp.offset[e][j-1]+hp.prime[e][j-1]:0;}
    for(int e=0;e<2;++e)for(int rows : {1,5,31,33,129,256,257}) {
        const size_t n=size_t(rows)*eg::kEmbDim,wire_n=size_t(rows)*wire::ROW_BYTES;
        const int table_rows=hp.offset[e][23]+hp.prime[e][23],part=(table_rows+3)/4,Q=(rows+3)/4;
        std::vector<uint32_t> ids(size_t(rows)*24);
        std::vector<uint8_t> stage(n),all_scales(size_t(table_rows)*8);
        for(int j=0;j<24;++j)for(uint32_t i=0;i<hp.prime[e][j];++i)for(int k=0;k<8;++k)all_scales[size_t(hp.offset[e][j]+i)*8+k]=uint8_t(i);
        for(int r=0;r<rows;++r)for(int j=0;j<24;++j){
            uint32_t loc=rows==256?uint32_t(r):r==0?0:r==1?hp.prime[e][j]-1:(r*131u+j*17u)%hp.prime[e][j];
            ids[size_t(r)*24+j]=hp.offset[e][j]+loc;
            for(int b=0;b<256;++b)stage[(size_t(r)*24+j)*256+b]=uint8_t(b); // all 256 FP8 codes, including -0 and NaNs
        }
        Buffer<uint32_t> di(ids.size());di.put(ids);
        Buffer<uint8_t> ds(n);ds.put(stage);
        Buffer<B> old_partial(4*n),new_partial(4*n),old_sum(n),new_sum(n),assembled(size_t(Q)*eg::kEmbDim);
        Buffer<uint8_t> old_wire(4*wire_n),new_wire(4*wire_n),recv(size_t(4)*Q*wire::ROW_BYTES);
        for(int rank=0;rank<4;++rank){
            const auto c=eg::ColShard::make(hp,e,rank,4);
            const int first=rank*part,valid=std::min(part,table_rows-first);
            std::vector<uint8_t> compact(size_t(c.rows)*8),contiguous(all_scales.begin()+first*8,all_scales.begin()+(first+valid)*8);
            for(int j=rank;j<24;j+=4)memcpy(compact.data()+size_t(c.local[j])*8,all_scales.data()+size_t(c.first[j])*8,size_t(c.count[j])*8);
            Buffer<uint8_t> cs(compact.size()),os(contiguous.size());cs.put(compact);os.put(contiguous);
            eg::k_dequant<<<dim3(rows,3),256>>>(di.p,ds.p,os.p,first,valid,rows,old_partial.p+rank*n);
            eg::k_dequant_col<<<dim3(rows,3),256>>>(di.p,ds.p,cs.p,c,rows,new_partial.p+rank*n);
            wire::pack<<<rows,256>>>(di.p,ds.p,os.p,first,valid,rows,old_wire.p+rank*wire_n);
            wire::pack_col<<<rows,256>>>(di.p,ds.p,cs.p,c,rows,new_wire.p+rank*wire_n);
            CS_CUDA(cudaDeviceSynchronize());
        }
        auto oldp=old_partial.get(4*n),newp=new_partial.get(4*n);
        auto wireh=new_wire.get(4*wire_n);
        for(int rank=0;rank<4;++rank)for(int r=0;r<rows;++r){
            const uint8_t* w=wireh.data()+rank*wire_n+size_t(r)*wire::ROW_BYTES;
            uint32_t hdr[4];memcpy(hdr,w,16);unsigned mask=0;for(int j=rank;j<24;j+=4)mask|=1u<<j;
            CS_CHECK(hdr[0]==6 && hdr[1]==mask && hdr[2]==0 && hdr[3]==0);
            for(int j=0;j<24;++j){
                const size_t at=(size_t(r)*24+j)*256;
                const int owner=ids[size_t(r)*24+j]/part;
                for(int b=0;b<256;++b){
                    uint16_t got,ref;memcpy(&got,&newp[rank*n+at+b],2);memcpy(&ref,&oldp[owner*n+at+b],2);
                    CS_CHECK(got==(j%4==rank?ref:0));
                    if(j%4==rank && got==0x8000)++negative_zeros;
                }
                const uint8_t* cell=w+16+j*264;
                for(int b=0;b<264;++b)CS_CHECK(cell[b]==(j%4==rank?(b<256?stage[at+b]:all_scales[size_t(ids[size_t(r)*24+j])*8+b-256]):0));
            }
        }
        sum<<<128,256>>>(old_partial.p,old_sum.p,n);sum<<<128,256>>>(new_partial.p,new_sum.p,n);
        auto ref=old_sum.get(n),got=new_sum.get(n);CS_CHECK(memcmp(ref.data(),got.data(),n*2)==0);compared+=n;
        // Model the actual source-major all-to-all, including ragged/empty row owners.
        for(int dst=0;dst<4;++dst){
            int row0=dst*Q,owned=std::max(0,std::min(Q,rows-row0));if(!owned)continue;
            for(auto src : {old_wire.p,new_wire.p}){
                CS_CUDA(cudaMemset(recv.p,0xa5,size_t(4)*Q*wire::ROW_BYTES));
                for(int rank=0;rank<4;++rank)CS_CUDA(cudaMemcpy(recv.p+size_t(rank)*Q*wire::ROW_BYTES,src+rank*wire_n+size_t(row0)*wire::ROW_BYTES,size_t(owned)*wire::ROW_BYTES,cudaMemcpyDeviceToDevice));
                wire::unpack<<<dim3(owned,24),256>>>(recv.p,Q,owned,4,assembled.p);
                auto actual=assembled.get(size_t(owned)*eg::kEmbDim);
                CS_CHECK(memcmp(actual.data(),ref.data()+size_t(row0)*eg::kEmbDim,actual.size()*2)==0);compared+=actual.size();
            }
        }
    }
    CS_CUDA(cudaGetLastError());CS_CHECK(negative_zeros>0);
    printf("PASS col_shard GPU: both layers, all four ranks, M=1/5/31/33/129/256/257, raw dequant and wire cells exact; %llu summed/unpacked bf16 elements byte-equal; %llu negative-zero raw cells; all 256 FP8 x 256 E8M0 combinations (NaN/Inf/subnormal included)\n",(unsigned long long)compared,(unsigned long long)negative_zeros);
    return 0;
}
#undef CS_CHECK
#undef CS_CUDA
} // namespace col_shard_test
