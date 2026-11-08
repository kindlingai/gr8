// Private pre-pass correctness/resource/time probe. No external implementation.
#include "../../../ds41_attn.cuh"
#include "variant.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>

#define CHECK(expr) do { cudaError_t e = (expr); if (e != cudaSuccess) { \
    fprintf(stderr, "%s: %s\n", #expr, cudaGetErrorString(e)); return 2; } } while(0)

// Compare against the actual original cq_load/cq_store widening path. Every
// possible scale byte and code is exercised, including signed zero and NaN.
__global__ void k_original_widen(const uint8_t* pk, ds41a::bf16* output, size_t rows) {
    __shared__ ds41a::bf16 tile[64 * ds41a::HD];
    __shared__ const ds41a::bf16* rowp[64];
    const int tid = threadIdx.x;
    const size_t base = size_t(blockIdx.x) * 64;
    if (tid < 64) rowp[tid] = base + tid < rows
        ? reinterpret_cast<const ds41a::bf16*>(pk + (base + tid) * ds41a::CQ_ROW + 1) : nullptr;
    __syncthreads();
    uint2 codes[8]; uint32_t scales[8];
    ds41a::cq_load(rowp, tid >> 5, tid & 31, codes, scales);
    ds41a::cq_store(tile, ds41a::HD, rowp, tid >> 5, tid & 31, codes, scales);
    __syncthreads();
    for (int i = tid; i < 64 * ds41a::HD / 8; i += 256)
        if (base + i / 64 < rows) reinterpret_cast<uint4*>(output + base * ds41a::HD)[i] = reinterpret_cast<uint4*>(tile)[i];
}

int main(int argc, char** argv) {
    const size_t rows = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1025;
    if (!rows || rows > 1048576) return 2;
    const size_t packed_bytes = rows * ds41a::CQ_ROW, wide_bytes = rows * ds41a::HD * 2;
    std::vector<uint8_t> packed(packed_bytes), actual(wide_bytes), expect(wide_bytes);
    uint8_t* device_pk = nullptr; ds41a::bf16* device_ref = nullptr;
    CHECK(cudaMalloc(&device_pk, packed_bytes)); CHECK(cudaMalloc(&device_ref, wide_bytes));
    ds41a::AttnP p{}; p.TK = 512; p.ckq = 1; p.ck = reinterpret_cast<ds41a::bf16*>(device_pk); p.ck_ss = packed_bytes;
    CHECK(sweep_variant::init(p));
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t r = 0; r < rows; ++r) {
            for (int c = 0; c < ds41a::CQ_CODES; ++c) packed[r * ds41a::CQ_ROW + c] = (c + r + pass * 37) & 255;
            for (int g = 0; g < 32; ++g) packed[r * ds41a::CQ_ROW + ds41a::CQ_CODES + g] = (r + g + pass * 53) & 255;
        }
        CHECK(cudaMemcpy(device_pk, packed.data(), packed_bytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemset(sweep_variant::scratch.data, 0xA5, wide_bytes)); CHECK(cudaMemset(device_ref, 0x5A, wide_bytes));
        k_original_widen<<<(rows+63)/64,256>>>(device_pk, device_ref, rows);
        sweep_variant::prepass(p, 0); CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(actual.data(), sweep_variant::scratch.data, wide_bytes, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(expect.data(), device_ref, wide_bytes, cudaMemcpyDeviceToHost));
        size_t bad = 0; for(size_t i=0;i<wide_bytes;++i) bad += actual[i] != expect[i];
        printf("pass=%d rows=%zu mismatch_bytes=%zu\n",pass,rows,bad);
        if(bad) return 1;
    }
    cudaFuncAttributes attr{}; CHECK(cudaFuncGetAttributes(&attr,sweep_variant::k_prewiden_keys));
    cudaEvent_t begin,end; CHECK(cudaEventCreate(&begin)); CHECK(cudaEventCreate(&end));
    for(int i=0;i<3;++i) sweep_variant::prepass(p,0);
    double mean=0;
    for(int r=0;r<5;++r) {
        CHECK(cudaEventRecord(begin));
        for(int i=0;i<20;++i) sweep_variant::prepass(p,0);
        CHECK(cudaEventRecord(end)); CHECK(cudaEventSynchronize(end));
        float ms=0; CHECK(cudaEventElapsedTime(&ms,begin,end)); mean += ms * 1000.0 / 20 / 5;
    }
    printf("PREPASS rows=%zu read_bytes=%zu write_bytes=%zu scratch_bytes=%zu mean_us=%.6f regs=%d local_bytes=%zu static_smem=%zu dynamic_smem=0\n",
        rows,packed_bytes,wide_bytes,sweep_variant::scratch_bytes(p),mean,attr.numRegs,attr.localSizeBytes,attr.sharedSizeBytes);
    CHECK(cudaEventDestroy(begin)); CHECK(cudaEventDestroy(end)); CHECK(cudaFree(device_pk)); CHECK(cudaFree(device_ref));
    return 0;
}
