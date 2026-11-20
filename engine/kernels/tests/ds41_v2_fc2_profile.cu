// Profile preparation only: no weights allocated or GEMM launched. Device
// attributes/occupancy are queried on THIS GPU, never presented as GB10 data.
// Compile/run on the target to get that target's occupancy before its trace.
// nvcc -O2 -std=c++17 -gencode=arch=compute_120a,code=sm_120a \
//   engine/kernels/tests/ds41_v2_fc2_profile.cu -o ds41_v2_fc2_profile
#include <cstdio>
#include <cstdlib>
#include "../ds41_v2_fc2.cuh"
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(2);}}while(0)
int main() {
    constexpr int I=576,H=5120,NE=384,TOPK=6;
    constexpr size_t RS=ds41v2moe::resident_smem<4,8,I>(),SS=mxtc::fc2_smem<4,8,8>();
    auto resident=ds41v2moe::k_fc2_resident<I,H>;auto stock=mxtc::k_fc2_tc<I,H,4,2,8,8,4,ds41v2moe::Args>;
    CK(cudaFuncSetAttribute(resident,cudaFuncAttributeMaxDynamicSharedMemorySize,RS));
    CK(cudaFuncSetAttribute(stock,cudaFuncAttributeMaxDynamicSharedMemorySize,SS));
    cudaDeviceProp dev{};int device;CK(cudaGetDevice(&device));CK(cudaGetDeviceProperties(&dev,device));
    cudaFuncAttributes ra{},sa{};CK(cudaFuncGetAttributes(&ra,resident));CK(cudaFuncGetAttributes(&sa,stock));
    int rb=0,sb=0;CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&rb,resident,128,RS));CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&sb,stock,128,SS));
    printf("device=%s sm=%d.%d SMs=%d (measured device query; no GEMM timing)\n",dev.name,dev.major,dev.minor,dev.multiProcessorCount);
    printf("kernel,registers_per_thread,local_bytes_per_thread,dynamic_smem_bytes,active_CTAs_per_SM,occupancy_pct\n");
    printf("resident,%d,%zu,%zu,%d,%.3f\n",ra.numRegs,ra.localSizeBytes,RS,rb,100.0*rb*128/dev.maxThreadsPerMultiProcessor);
    printf("grouped_stock,%d,%zu,%zu,%d,%.3f\n",sa.numRegs,sa.localSizeBytes,SS,sb,100.0*sb*128/dev.maxThreadsPerMultiProcessor);
    puts("M,D_assumed,launch_CTAs_per_group,live_CTAs_per_group,route_rows_avg,64row_groups_avg,full_group_waves,unique_fc2_weight_bytes");
    for(int M:{4096,8192})for(int D:{64,128,384}) {
        const int live=D*10,launch=NE*10,rows=(M*TOPK+D-1)/D,groups=(rows+63)/64;
        const int waves=(live+dev.multiProcessorCount*rb-1)/(dev.multiProcessorCount*rb);
        printf("%d,%d,%d,%d,%d,%d,%d,%zu\n",M,D,launch,live,rows,groups,waves,size_t(D)*(H/16)*(I/64)*mx4::GR);
    }
    puts("DERIVED grid rows assume D used experts and uniform routes; actual skew comes from the route capture.");
    puts("Per live CTA: 39168 weight bytes loaded once, then ceil(slot_nt/8) row groups; four column launches per layer.");
    puts("Profile kernel ds41v2moe::k_fc2_resident; compare GR8_DS_PF_V2=1 GR8_DS_PF_V2_FC2=0/1 on the same target boot.");
    return 0;
}
