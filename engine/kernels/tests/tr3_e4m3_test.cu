// Unit test of the trellis codebook decode (tr3.cuh), for each codebook: mcg, mul1 and pmcgx.
//
//   nvcc -O2 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -o tr3_e4m3_test tr3_e4m3_test.cu -lcublas
//   ./tr3_e4m3_test                                         every state, then random tiles
//   ./tr3_e4m3_test --cb mul1 --k 3 --tiles T.bin --values V.bin [--cb ... --k ... --tiles ... --values ...]
//
// Every state: the GPU codebook functions (tr3_cb2, tr3_mul1x2, tr3_pmcgx), with and without the E4M3 rounding, for
// all 65,536 16-bit states (mcg, mul1) and all 1,048,576 20-bit states (pmcgx), against a host reference written here
// from the format. The tables' FNV-1a 64 hashes must equal the constants below, which the Rust loader's codebook
// (engine/loader-rs/src/exlr8.rs tests) and the Python reference (engine/ref/g6ref.py codebook_hashes) check too, so the
// three implementations agree on every state.
//
// Random tiles: 65,536 tiles per width K2 to K4 (any bit pattern is a valid tail-biting stream), decoded by every lane
// path the kernels use: Tr3Lane load + decode (TR3 and E4), loadw + decodew, the wide order's (chunk[l-1] << 8 K) |
// chunk[l] (tr3_wide.cuh), the prefill path's E4M3 bytes (bindp::tr3_b8) and the per-weight tr3_weight. Each against
// the host reference of every weight: its state (the 16 stream bits ending at bit (t+1)*K, or for pmcgx the 20 ending at
// bit ((t|1)+1)*K; MSB first over LE u32 words, tail-biting) and its codebook value, rounded with
// __nv_cvt_halfraw_to_fp8(value, SATFINITE, E4M3) for the E4 paths.
//
// --tiles / --values: raw packed tiles (32 K bytes each) and the encoder's own decoded values of them (fp32, 256 per tile
// in stream order). The E4 paths must reproduce those values exactly.
// Exit 0 when every check holds.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_fp8.h>
#include "../bind_prefill.cuh"
#include "../../h2d.h"

// FNV-1a 64 of the codebook tables (little-endian u16 fp16 bits; pmcgx: (weight 2p, weight 2p + 1) per state)
struct Hashes { const char* name; uint64_t f16, e4; };
static const Hashes WANT[TR3_NCB] = {
    {"mcg", 0x7bb36b05a3b885a7ull, 0xd6995a3ae102700aull},
    {"mul1", 0x55e1c748ed9f0cd7ull, 0xf4f1a8cc7cda90bfull},
    {"pmcgx", 0x6f7539ea76219e77ull, 0xc8d37146b1cb8465ull},
};

static uint64_t fnv(const std::vector<uint16_t>& v)
{
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint16_t x : v)
        for (int b = 0; b < 2; ++b) { h ^= (x >> (8 * b)) & 0xff; h *= 0x100000001b3ull; }
    return h;
}

// ------------------------------------------------------------------------------------------ host reference
static __half_raw hr(uint16_t b) { __half_raw r; r.x = b; return r; }

static uint16_t ref_e4(uint16_t v)
{
    const __nv_fp8_storage_t q = __nv_cvt_halfraw_to_fp8(hr(v), __NV_SATFINITE, __NV_E4M3);
    return __nv_cvt_fp8_to_halfraw(q, __NV_E4M3).x;
}

static uint32_t mask3(uint32_t x) { return (x & 0x8fff8fffu) ^ 0x3b603b60u; }
static uint16_t hadd(uint16_t a, uint16_t b) { return __half_raw(__hadd(__half(hr(a)), __half(hr(b)))).x; }

// the fp16 value(s) of a state: v[0] (and v[1] for pmcgx, weight 2p + 1)
static void ref_values(int cb, uint32_t s, uint16_t v[2])
{
    if (cb == TR3_MCG)
    {
        const uint32_t x = mask3(s * 0xCBAC1FEDu);
        v[0] = hadd(x & 0xffff, x >> 16);
    }
    else if (cb == TR3_MUL1)
    {
        const uint32_t x = s * 0x83DCD12Du;
        const int sum = (x & 0xff) + ((x >> 8) & 0xff) + ((x >> 16) & 0xff) + (x >> 24);
        // exact in double (11-bit operands), then one rounding to fp16
        v[0] = __half_raw(__double2half((1024.0 + sum) * 0.00676727294921875 - 10.3828125)).x;
    }
    else
    {
        const uint32_t m1 = mask3(s * 0xCBAC1FEDu), m2 = mask3(s * 0xBCA14C51u);
        v[0] = hadd(m1 & 0xffff, m2 >> 16);
        v[1] = hadd(m1 >> 16, m2 & 0xffff);
    }
}

static uint32_t ref_state(const uint8_t* tile, int K, int end, int L)
{
    const int nb = 256 * K;
    uint32_t s = 0;
    for (int i = 0; i < L; ++i)
    {
        const int bit = ((end - L + i) % nb + nb) % nb;     // stream bit, MSB first over LE u32 words
        uint32_t word;
        memcpy(&word, tile + (bit / 32) * 4, 4);
        s = (s << 1) | ((word >> (31 - bit % 32)) & 1u);
    }
    return s;
}

// fp16 value of stream weight t of a tile
static uint16_t ref_weight(int cb, const uint8_t* tile, int K, int t)
{
    uint16_t v[2];
    if (cb == TR3_PMCGX)
    {
        ref_values(cb, ref_state(tile, K, ((t | 1) + 1) * K, 20), v);
        return v[t & 1];
    }
    ref_values(cb, ref_state(tile, K, (t + 1) * K, 16), v);
    return v[0];
}

// ------------------------------------------------------------------------------------------ every state
template <int CB>
__global__ void k_states(int n, uint16_t* f16, uint16_t* e4)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n) return;
    if constexpr (CB == TR3_PMCGX)
    {
        const u32 a = tr3_pmcgx<false>(s), b = tr3_pmcgx<true>(s);
        f16[2 * s] = a & 0xffff; f16[2 * s + 1] = a >> 16;
        e4[2 * s] = b & 0xffff; e4[2 * s + 1] = b >> 16;
    }
    else
    {
        // the state goes in either half: both must give its value
        const u32 a = CB == TR3_MUL1 ? tr3_mul1x2<false>(s, s ^ 0x5a5au) : tr3_cb2<false>(s, s ^ 0x5a5au);
        const u32 b = CB == TR3_MUL1 ? tr3_mul1x2<true>(s ^ 0x5a5au, s) : tr3_cb2<true>(s ^ 0x5a5au, s);
        f16[s] = a & 0xffff;
        e4[s] = b >> 16;
    }
}

template <int CB>
static int check_states()
{
    const int n = CB == TR3_PMCGX ? 1 << 20 : 1 << 16, per = CB == TR3_PMCGX ? 2 : 1;
    uint16_t *df, *de;
    cudaMalloc(&df, static_cast<size_t>(n) * per * 2);
    cudaMalloc(&de, static_cast<size_t>(n) * per * 2);
    k_states<CB><<<(n + 255) / 256, 256>>>(n, df, de);
    std::vector<uint16_t> f(static_cast<size_t>(n) * per), e(f.size());
    cudaMemcpy(f.data(), df, f.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(e.data(), de, e.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(df);
    cudaFree(de);
    if (cudaGetLastError() != cudaSuccess) { fprintf(stderr, "%s states: CUDA error\n", WANT[CB].name); return 1; }
    long bad = 0, changed = 0;
    for (int s = 0; s < n; ++s)
    {
        uint16_t v[2];
        ref_values(CB, s, v);
        for (int i = 0; i < per; ++i)
        {
            bad += f[s * per + i] != v[i];
            bad += e[s * per + i] != ref_e4(v[i]);
            changed += ref_e4(v[i]) != v[i];
        }
    }
    const uint64_t hf = fnv(f), he = fnv(e);
    const bool hok = hf == WANT[CB].f16 && he == WANT[CB].e4;
    printf("%s: %d states x %d values: GPU vs host reference %ld bad; E4M3 changed %.2f%%; fnv64 f16 %016llx e4 %016llx (%s)\n",
           WANT[CB].name, n, per, bad, 100.0 * changed / (static_cast<double>(n) * per), static_cast<unsigned long long>(hf),
           static_cast<unsigned long long>(he), hok ? "as expected" : "NOT the expected hashes");
    return bad || !hok;
}

// ------------------------------------------------------------------------------------------ tiles
// per (tile pair, lane): fp16 bits of the lane's 8 weights of tiles p and q by each path, and the prefill bytes
enum { P_TR3, P_E4, P_E4W, P_WIDE, P_WEIGHT, NP };
static const char* const PATH[NP] = {"load+decode (fp16)", "load+decode (E4)", "loadw+decodew (E4)", "wide order (E4)", "tr3_weight (E4)"};
struct Out { uint16_t v[NP][2][8]; u32 pf[4]; };

template <int BITS, int CB>
__global__ void k_decode(const uint8_t* tiles, int npairs, Out* out)
{
    const int lane = threadIdx.x & 31;
    const int pair = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (pair >= npairs) return;      // whole warps
    constexpr int TB = 32 * BITS;
    const uint8_t* p = tiles + static_cast<size_t>(2 * pair) * TB;
    Tr3Lane<BITS, false, CB> a;
    Tr3Lane<BITS, true, CB> b;
    a.init(lane);
    b.init(lane);
    Out o;
    u32 w[2][2];
    auto put = [&](int path, int s, const u32 (&f0)[2], const u32 (&f1)[2]) {
        const u32 r[4] = {f0[0], f0[1], f1[0], f1[1]};
        for (int j = 0; j < 8; ++j) o.v[path][s][j] = static_cast<uint16_t>(r[j / 2] >> (16 * (j & 1)));
    };
    for (int s = 0; s < 2; ++s)
    {
        const uint8_t* t = p + s * TB;
        a.load(t, w[s][0], w[s][1]);
        u32 f0[2], f1[2];
        a.decode(w[s][0], w[s][1], f0, f1);
        put(P_TR3, s, f0, f1);
        b.decode(w[s][0], w[s][1], f0, f1);
        put(P_E4, s, f0, f1);
        b.decodew(b.loadw(t, lane), f0, f1);
        put(P_E4W, s, f0, f1);
        const u32 c = trw_tile_chunk<BITS>(t, lane);
        const u32 pc = __shfl_sync(0xffffffffu, c, (lane + 31) & 31);
        Tr3Lane<BITS, true, CB>::decode_m((static_cast<u64>(pc) << (8 * BITS)) | c, 7 * BITS, f0, f1);
        put(P_WIDE, s, f0, f1);
        for (int j = 0; j < 8; ++j) o.v[P_WEIGHT][s][j] = static_cast<uint16_t>(tr3_weight<CB, true>(t, lane * 8 + j, BITS));
    }
    bindp::tr3_b8(a, w[0], w[1], o.pf);
    out[static_cast<size_t>(pair) * 32 + lane] = o;
}

// values: the encoder's decoded values (fp32, stream order), or empty for the host reference
template <int BITS, int CB>
static int run(const std::vector<uint8_t>& tiles, const std::vector<float>& values, const char* what)
{
    constexpr int TB = 32 * BITS;
    const int ntiles = static_cast<int>(tiles.size() / TB) & ~1, npairs = ntiles / 2;
    if (npairs == 0) { fprintf(stderr, "%s: no tiles\n", what); return 1; }
    if (!values.empty() && values.size() < static_cast<size_t>(ntiles) * 256) { fprintf(stderr, "%s: too few values\n", what); return 1; }
    uint8_t* dt;
    Out* dout;
    cudaMalloc(&dt, static_cast<size_t>(ntiles) * TB);
    cudaMalloc(&dout, static_cast<size_t>(npairs) * 32 * sizeof(Out));
    g6h2d::copy(dt, tiles.data(), static_cast<size_t>(ntiles) * TB);     // pageable source: pinned staging
    k_decode<BITS, CB><<<(npairs + 7) / 8, 256>>>(dt, npairs, dout);
    std::vector<Out> out(static_cast<size_t>(npairs) * 32);
    cudaMemcpy(out.data(), dout, out.size() * sizeof(Out), cudaMemcpyDeviceToHost);
    if (cudaGetLastError() != cudaSuccess) { fprintf(stderr, "%s: CUDA error\n", what); return 1; }
    cudaFree(dt);
    cudaFree(dout);
    long bad[NP] = {}, bad_pf = 0, bad_ref = 0, changed = 0, n = 0;
    for (int pair = 0; pair < npairs; ++pair)
        for (int lane = 0; lane < 32; ++lane)
        {
            const Out& o = out[static_cast<size_t>(pair) * 32 + lane];
            for (int s = 0; s < 2; ++s)
                for (int j = 0; j < 8; ++j)
                {
                    const int tile = 2 * pair + s, t = lane * 8 + j;
                    const uint16_t v = ref_weight(CB, tiles.data() + static_cast<size_t>(tile) * TB, BITS, t);
                    uint16_t r = ref_e4(v);
                    if (!values.empty())
                    {
                        const uint16_t enc = __half_raw(__float2half_rn(values[static_cast<size_t>(tile) * 256 + t])).x;
                        bad_ref += enc != r;             // the host reference vs the encoder
                        r = enc;
                    }
                    bad[P_TR3] += o.v[P_TR3][s][j] != v;
                    for (int p = P_E4; p < NP; ++p) bad[p] += o.v[p][s][j] != r;
                    const __nv_fp8_storage_t q = __nv_cvt_halfraw_to_fp8(hr(r), __NV_SATFINITE, __NV_E4M3);
                    const uint8_t pb = static_cast<uint8_t>(o.pf[j / 2] >> (16 * s + 8 * (j & 1)));   // b[reg] = p's two bytes | q's << 16
                    bad_pf += pb != q;
                    changed += r != v;
                    ++n;
                }
        }
    long total = bad_pf + bad_ref;
    printf("%s (%s, K%d, %d tiles, %ld weights): ", what, WANT[CB].name, BITS, ntiles, n);
    for (int p = 0; p < NP; ++p) { printf("%s %ld bad; ", PATH[p], bad[p]); total += bad[p]; }
    printf("prefill E4M3 bytes %ld bad; ", bad_pf);
    if (!values.empty()) printf("host reference vs encoder %ld bad; ", bad_ref);
    printf("E4M3 changed %.2f%%\n", 100.0 * changed / n);
    return total != 0;
}

template <int CB>
static int run_k(int k, const std::vector<uint8_t>& tiles, const std::vector<float>& values, const char* what)
{
    if (k == 2) return run<2, CB>(tiles, values, what);
    if (k == 3) return run<3, CB>(tiles, values, what);
    if (k == 4) return run<4, CB>(tiles, values, what);
    fprintf(stderr, "K%d: not a width\n", k);
    return 1;
}

static int run_any(int cb, int k, const std::vector<uint8_t>& tiles, const std::vector<float>& values, const char* what)
{
    if (cb == TR3_MUL1) return run_k<TR3_MUL1>(k, tiles, values, what);
    if (cb == TR3_PMCGX) return run_k<TR3_PMCGX>(k, tiles, values, what);
    return run_k<TR3_MCG>(k, tiles, values, what);
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

static std::vector<uint8_t> read_file(const std::string& path)
{
    std::vector<uint8_t> v;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(2); }
    uint8_t buf[1 << 16];
    size_t k;
    while ((k = fread(buf, 1, sizeof buf, f)) > 0) v.insert(v.end(), buf, buf + k);
    fclose(f);
    return v;
}

int main(int argc, char** argv)
{
    int rc = 0;
    if (argc > 1)
    {
        int cb = -1, k = 0;
        std::string tiles;
        for (int i = 1; i + 1 < argc; i += 2)
        {
            const std::string a = argv[i], v = argv[i + 1];
            if (a == "--cb") { cb = -1; for (int c = 0; c < TR3_NCB; ++c) if (v == WANT[c].name) cb = c; }
            else if (a == "--k") k = atoi(v.c_str());
            else if (a == "--tiles") tiles = v;
            else if (a == "--values")
            {
                if (cb < 0 || tiles.empty()) { fprintf(stderr, "--values needs --cb (mcg, mul1 or pmcgx) and --tiles first\n"); return 2; }
                const std::vector<uint8_t> raw = read_file(v);
                std::vector<float> vals(raw.size() / 4);
                memcpy(vals.data(), raw.data(), vals.size() * 4);
                rc |= run_any(cb, k, read_file(tiles), vals, tiles.c_str());
            }
            else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
        }
        printf(rc ? "FAIL\n" : "PASS\n");
        return rc;
    }
    rc |= check_states<TR3_MCG>();
    rc |= check_states<TR3_MUL1>();
    rc |= check_states<TR3_PMCGX>();
    const uint64_t seeds[3] = {0x2545F4914F6CDD1Dull, 0x9E3779B97F4A7C15ull, 0xD1B54A32D192ED03ull};
    for (int cb = 0; cb < TR3_NCB; ++cb)
        for (int k = 2; k <= 4; ++k)
            rc |= run_any(cb, k, random_tiles(65536, k, seeds[k - 2] + cb), {}, "random");
    printf(rc ? "FAIL\n" : "PASS\n");
    return rc;
}
