// Unit test of the trellis codebook decode. The decode path's exlr8-e4m3 decode (Tr3Lane<BITS, true>) is checked
// against the prefill path's E4M3 rounding (bindp::tr3_b8) and a host reference. The TR3 decode (Tr3Lane<BITS, false>)
// is checked against the same reference.
//
//   nvcc -O2 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -o tr3_e4m3_test tr3_e4m3_test.cu -lcublas
//   ./tr3_e4m3_test [K3_TILES_FILE K4_TILES_FILE]
//
// Without files the test decodes 65,536 random tiles per bit width, K2 to K4 (any bit pattern is a valid tail-biting
// stream).
// A tiles file holds raw trellis bytes, 32*K bytes per tile: one trellis tensor of a layer file, written with
// ST(path).get(name).tofile(out) (tools/convert/st.py).
// Host reference per weight t of a tile: state = the 16 stream bits ending at bit (t+1)*K (MSB first over LE u32 words,
// tail-biting). value = fp16(lo) + fp16(hi) of the 3INST hash. E4M3 = __nv_cvt_halfraw_to_fp8(value, SATFINITE, E4M3).
// The exlr8 encoder rounded with that same call.
// Checks: (1) TR3 decode == value. (2) E4 decode == value rounded to E4M3 and back. (3) Prefill bytes == E4M3(value).
// (2) and (3) together show that both paths decode the same E4M3 values. Exit 0 when all three hold.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_fp8.h>
#include "../bind_prefill.cuh"
#include "../../h2d.h"

// per (tile pair, lane): the decode fragments of tiles p and q, then the prefill E4M3 bytes of both
struct Out { u32 tr3[2][4], e4[2][4], pf[4]; };

template <int BITS>
__global__ void k_decode(const uint8_t* tiles, int npairs, Out* out)
{
    const int lane = threadIdx.x & 31;
    const int pair = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (pair >= npairs) return;
    constexpr int TB = 32 * BITS;
    const uint8_t* p = tiles + static_cast<size_t>(2 * pair) * TB;
    Tr3Lane<BITS, false> a;
    Tr3Lane<BITS, true> b;
    a.init(lane);
    b.init(lane);
    Out o;
    u32 w[2][2];
    for (int s = 0; s < 2; ++s)
    {
        a.load(p + s * TB, w[s][0], w[s][1]);
        u32 f0[2], f1[2], g0[2], g1[2];
        a.decode(w[s][0], w[s][1], f0, f1);
        b.decode(w[s][0], w[s][1], g0, g1);
        o.tr3[s][0] = f0[0]; o.tr3[s][1] = f0[1]; o.tr3[s][2] = f1[0]; o.tr3[s][3] = f1[1];
        o.e4[s][0] = g0[0]; o.e4[s][1] = g0[1]; o.e4[s][2] = g1[0]; o.e4[s][3] = g1[1];
    }
    bindp::tr3_b8<BITS>(a, w[0], w[1], o.pf);
    out[static_cast<size_t>(pair) * 32 + lane] = o;
}

static uint16_t ref_state(const uint8_t* tile, int K, int t)
{
    const int nb = 256 * K;
    uint32_t s = 0;
    for (int i = 0; i < 16; ++i)
    {
        const int bit = (((t + 1) * K - 16 + i) % nb + nb) % nb;     // stream bit, MSB first over LE u32 words
        uint32_t word;
        memcpy(&word, tile + (bit / 32) * 4, 4);
        s = (s << 1) | ((word >> (31 - bit % 32)) & 1u);
    }
    return static_cast<uint16_t>(s);
}

static __half_raw ref_value(uint16_t state)
{
    uint32_t x = static_cast<uint32_t>(state) * 0xCBAC1FEDu;
    x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
    __half_raw lo, hi;
    lo.x = static_cast<unsigned short>(x & 0xffff);
    hi.x = static_cast<unsigned short>(x >> 16);
    return __half_raw(__hadd(__half(lo), __half(hi)));
}

static uint16_t bits_of(u32 word, int hi) { return static_cast<uint16_t>(hi ? word >> 16 : word & 0xffff); }

template <int BITS>
static int run(const std::vector<uint8_t>& tiles, const char* what)
{
    constexpr int TB = 32 * BITS;
    const int ntiles = static_cast<int>(tiles.size() / TB) & ~1, npairs = ntiles / 2;
    if (npairs == 0) { fprintf(stderr, "%s: no tiles\n", what); return 1; }
    uint8_t* dt;
    Out* dout;
    cudaMalloc(&dt, static_cast<size_t>(ntiles) * TB);
    cudaMalloc(&dout, static_cast<size_t>(npairs) * 32 * sizeof(Out));
    g6h2d::copy(dt, tiles.data(), static_cast<size_t>(ntiles) * TB);     // pageable source: pinned staging
    k_decode<BITS><<<(npairs + 7) / 8, 256>>>(dt, npairs, dout);
    std::vector<Out> out(static_cast<size_t>(npairs) * 32);
    cudaMemcpy(out.data(), dout, out.size() * sizeof(Out), cudaMemcpyDeviceToHost);
    if (cudaGetLastError() != cudaSuccess) { fprintf(stderr, "%s: CUDA error\n", what); return 1; }
    cudaFree(dt);
    cudaFree(dout);
    long bad_tr3 = 0, bad_e4 = 0, bad_pf = 0, changed = 0, n = 0;
    for (int pair = 0; pair < npairs; ++pair)
        for (int lane = 0; lane < 32; ++lane)
        {
            const Out& o = out[static_cast<size_t>(pair) * 32 + lane];
            for (int s = 0; s < 2; ++s)
                for (int j = 0; j < 8; ++j)
                {
                    const uint8_t* tile = tiles.data() + static_cast<size_t>(2 * pair + s) * TB;
                    const __half_raw v = ref_value(ref_state(tile, BITS, lane * 8 + j));
                    const __nv_fp8_storage_t q = __nv_cvt_halfraw_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
                    const __half_raw r = __nv_cvt_fp8_to_halfraw(q, __NV_E4M3);
                    const int reg = j / 2, hi = j & 1;
                    bad_tr3 += bits_of(o.tr3[s][reg], hi) != v.x;
                    bad_e4 += bits_of(o.e4[s][reg], hi) != r.x;
                    const uint8_t pb = static_cast<uint8_t>(o.pf[reg] >> (16 * s + 8 * hi));   // b[reg] = p's two bytes | q's << 16
                    bad_pf += pb != q;
                    changed += r.x != v.x;
                    ++n;
                }
        }
    printf("%s: %ld weights (%d tiles, K%d): TR3 decode vs reference %ld bad; E4 decode vs E4M3(reference) %ld bad; "
           "prefill E4M3 bytes vs E4M3(reference) %ld bad; rounding changed %.2f%% of values\n",
           what, n, ntiles, BITS, bad_tr3, bad_e4, bad_pf, 100.0 * changed / n);
    return bad_tr3 || bad_e4 || bad_pf;
}

static std::vector<uint8_t> random_tiles(int ntiles, int K, uint64_t seed)
{
    std::vector<uint8_t> v(static_cast<size_t>(ntiles) * 32 * K);
    for (auto& b : v)
    {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        b = static_cast<uint8_t>(seed >> 24);
    }
    return v;
}

static std::vector<uint8_t> read_file(const char* path)
{
    std::vector<uint8_t> v;
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    uint8_t buf[1 << 16];
    size_t k;
    while ((k = fread(buf, 1, sizeof buf, f)) > 0) v.insert(v.end(), buf, buf + k);
    fclose(f);
    return v;
}

int main(int argc, char** argv)
{
    int rc = 0;
    rc |= run<2>(random_tiles(65536, 2, 0x2545F4914F6CDD1Dull), "random K2");     // exlr8 v2 K2
    rc |= run<3>(random_tiles(65536, 3, 0x9E3779B97F4A7C15ull), "random K3");
    rc |= run<4>(random_tiles(65536, 4, 0xD1B54A32D192ED03ull), "random K4");
    if (argc >= 3)
    {
        rc |= run<3>(read_file(argv[1]), argv[1]);
        rc |= run<4>(read_file(argv[2]), argv[2]);
    }
    printf(rc ? "FAIL\n" : "PASS\n");
    return rc;
}
