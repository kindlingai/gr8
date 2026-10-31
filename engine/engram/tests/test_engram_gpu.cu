// GPU test of the Engram device path (engram_dev.cuh) with a fake rank-1-of-4 sidecar on local disk and fake hooks.
//   test_engram_gpu <PREFIX.engram.cmap> <vectors.bin> <scratch dir> [rank]     (cmap + vectors: engine/engram/gen_engram.py; rank of 16, default 1)
// What it proves, on a real GPU:
//   1. k_hash: the ids the device writes are bit-identical to the official vectors for every position of every vector sequence
//      (batches of up to 32 rows, one begin_step each).
//   2. the NVMe path: fake E4M3 rows / E8M0 scales are pwritten to the sparse w8/s8 files for the rows rank 1 owns; the device
//      embedding that reaches the reduce hook equals the host dequant of those rows bit for bit, and is zero for foreign rows.
//   3. k_gate: T after each op equals the host emulation of the same fp32 operations (device reduction order) within 1 bf16 ulp,
//      for kv = the fake projection (double-precision GEMV, host-checked on sampled outputs, fully for 1 row).
//   4. a repeat of the same windows costs 0 cache misses; rows = 3, 1, 32 reuse the Jobs; the async chain (no syncs between ops)
//      is bit-identical to the stepwise run.
//   5. begin_step + op + op captured into a CUDA graph replays correctly twice (different windows each replay).
//   6. fail-closed: no hooks -> 801, a window token outside the vocabulary -> status() == -1.
//   7. W-row chunks (prefill; not yet run on a GPU): a prompt of 3079 positions in chunks of 1024, of 2048 and of odd sizes around
//      the ring size (61..65, 127..129, 1000, ...): Engram::window_wide (windows == the true sequence, the ring == the 32-row path's) + begin_step (ids == host
//      hash of the true windows) + op x 2 with a cheap fake projection (embedding == host dequant, gate within 1 ulp of the host emulation); k_act_quant_rows
//      == the host act_quant.h lines; positions that are not consecutive -> status() == -2; rows > max_rows refused.
//   7d. the tensor-core wkv seam (ds41_gemm_mx_tc linked from engine/attn/ds41_attn.cu; not yet run on a GPU): a synthetic full-size wkv pack for both layers
//      (codes [25600][6144], E8M0 scales [25600][192]), W1024 then W2048 chunks through Engram::op = k_dequant -> reduce hook -> k_act_quant_rows -> ds41_gemm_mx_tc -> k_gate;
//      sampled kv outputs vs a host double-precision reference (host act_quant.h on the embedding, host dequant of the pack); the project hook must not run
//      (hook calls 0 = the tensor-core path was taken, not the scalar fallback). A number, not a gate: op time with the default and the M-tile-fastest order.
//   7e. the BF16 trunk (the engine under -DDS_MHC_TRUNK_BF16 passes a __nv_bfloat16 trunk to Engram::op): k_gate(__nv_bfloat16*) vs k_gate(float*) on the same BF16-valued trunk,
//      bit for bit (this test calls both overloads whatever the build flag is). Not yet run on a GPU.
// Host code is built with -ffp-contract=off (the reference rounds every fp32 op separately).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include "engram/engram_dev.cuh"
using namespace ds41::engram;

static int fails = 0;
static long long g_gate_cancel_total = 0;   // section 7 gate/write elements beyond 1 bf16 ulp that are fp32 cancellation cases (all plans); the PASS line must say so
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); ++fails; } } while (0)
#define CU(x) do { const cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(2); } } while (0)

typedef __nv_bfloat16 bf16;
constexpr int kMaxRows = 32, kNR = 16;
static int kRank = 1;   // argv[4]; rank 15 owns the highest ids of both tables (ids above 360 M, file offsets above 4 GiB)   // rank 1 of 16 keeps the device footprint under 0.5 GB (2 x 192 MB of scales); the real TP4 rank holds 2 x 768 MB

// ---- fake table data: a pure function of (table, id, column), so the host never has to read the files back -------------------------
static uint64_t mix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
// E4M3 byte with exponent field 5..10 (values 2^-2..2^3 times 1..1.875): never the NaN code 0x7F / 0xFF
static uint8_t fake_w(int e, uint32_t id, int c)
{
    const uint64_t h = mix64(((uint64_t)e << 50) ^ ((uint64_t)id << 10) ^ (uint64_t)c);
    return (uint8_t)(((h >> 40) & 1) << 7 | (5 + (h >> 41) % 6) << 3 | ((h >> 50) & 7));
}
static uint8_t fake_s(int e, uint32_t id, int g) { return (uint8_t)(120 + mix64(((uint64_t)e << 50) ^ ((uint64_t)id << 10) ^ 0x3FF ^ ((uint64_t)g << 20)) % 8); }

// the fake wkv: W(e, o, k) = (hash % 17 - 8) / 512, exactly representable; accumulation in double in k order, so host == device
ENGRAM_HD static int wproj(int e, int o, int k)
{
    uint32_t h = (uint32_t)(o * kEmbDim + k) * 2654435761u + (uint32_t)e * 40503u;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    return (int)(h % 17u) - 8;
}
__global__ void k_fake_project(int e, int rows, const bf16* emb, bf16* kv)
{
    const int o = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (o >= kKvDim || r >= rows) return;
    const bf16* x = emb + (size_t)r * kEmbDim;
    double acc = 0.0;
    for (int k = 0; k < kEmbDim; ++k) acc += (double)__bfloat162float(x[k]) * ((double)wproj(e, o, k) * (1.0 / 512.0));
    kv[(size_t)r * kKvDim + o] = __float2bfloat16_rn((float)acc);
}
// W-row cases: kv is a cheap exact function of the embedding (the double-precision fake above costs hours at 2048 rows)
__global__ void k_cheap_project(int e, int rows, const bf16* emb, bf16* kv)
{
    const int o = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (o >= kKvDim || r >= rows) return;
    kv[(size_t)r * kKvDim + o] = emb[(size_t)r * kEmbDim + (size_t)((o * 37 + e) % kEmbDim)];
}
// ---- 7d: the synthetic wkv pack, one function for the host reference and the device fill ------------------------------------------------
__host__ __device__ static inline uint32_t pk_hash(uint32_t x, uint32_t salt)
{
    x = x * 2654435761u + salt * 0x9E3779B1u;
    x ^= x >> 15; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u; x ^= x >> 16;
    return x;
}
__host__ __device__ static inline uint8_t pk_code(int e, uint32_t n, uint32_t k)
{
    uint8_t c = (uint8_t)pk_hash(n * 6144u + k, (uint32_t)e + 1u);
    if ((c & 0x7F) == 0x7F) c = (uint8_t)((c & 0x80) | 0x38);   // the e4m3 NaN code -> +-1.0
    return c;
}
__host__ __device__ static inline uint8_t pk_scale(int e, uint32_t n, uint32_t g) { return (uint8_t)(124u + pk_hash(n * 192u + g, 77u + (uint32_t)e) % 7u); }   // 2^-3 .. 2^3
__global__ void k_fill_pack(int e, uint8_t* codes, uint8_t* scales)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t nc = (size_t)kKvDim * kEmbDim, ns = (size_t)kKvDim * (kEmbDim / 32);
    if (i < nc) codes[i] = pk_code(e, (uint32_t)(i / kEmbDim), (uint32_t)(i % kEmbDim));
    if (i < ns) scales[i] = pk_scale(e, (uint32_t)(i / (kEmbDim / 32)), (uint32_t)(i % (kEmbDim / 32)));
}
static double e4m3_val(uint8_t c)
{
    const int sg = c >> 7, ex = (c >> 3) & 15, m = c & 7;
    const double v = ex == 0 ? (m / 8.0) * std::ldexp(1.0, -6) : (1.0 + m / 8.0) * std::ldexp(1.0, ex - 7);
    return sg ? -v : v;
}
extern "C" void ds41_attn_set_tc_mfast(int on) __attribute__((weak));   // the attention GEMM's launch-order switch (GR8_DS_ATTN_TC_MF); absent from builds without it

static float host_project1(int e, const uint16_t* emb_row, int o)
{
    double acc = 0.0;
    for (int k = 0; k < kEmbDim; ++k) acc += (double)bf16_to_f32(emb_row[k]) * ((double)wproj(e, o, k) * (1.0 / 512.0));
    return bf16_to_f32(f32_to_bf16((float)acc));
}

// ---- hooks ----------------------------------------------------------------------------------------------------------------------
struct Ctx
{
    bool capturing = false, spy = false;
    uint16_t* emb_spy[kNE] = {};   // pinned host copies of the embedding at the reduce hook and of kv after the project hook
    uint16_t* kv_spy[kNE] = {};
    int reduce_calls = 0, project_calls = 0;
} g;

static int hook_reduce(void* model, int e, int rows, void* stream, bf16* emb)
{
    Ctx& c = *(Ctx*)model;
    ++c.reduce_calls;   // one process stands for the 4 ranks: the sum over ranks of "mine + zeros" is the identity
    if (c.spy && !c.capturing)
        if (cudaMemcpyAsync(c.emb_spy[e], emb, (size_t)rows * kEmbDim * 2, cudaMemcpyDeviceToHost, (cudaStream_t)stream) != cudaSuccess) return 1;
    return 0;
}
static int hook_project(void* model, int e, int rows, void* stream, const bf16* emb, bf16* kv)
{
    Ctx& c = *(Ctx*)model;
    ++c.project_calls;
    k_fake_project<<<dim3((kKvDim + 127) / 128, rows), 128, 0, (cudaStream_t)stream>>>(e, rows, emb, kv);
    if (cudaGetLastError() != cudaSuccess) return 2;
    if (c.spy && !c.capturing)
        if (cudaMemcpyAsync(c.kv_spy[e], kv, (size_t)rows * kKvDim * 2, cudaMemcpyDeviceToHost, (cudaStream_t)stream) != cudaSuccess) return 3;
    return 0;
}

static int hook_project_cheap(void* model, int e, int rows, void* stream, const bf16* emb, bf16* kv)
{
    Ctx& c = *(Ctx*)model;
    ++c.project_calls;
    k_cheap_project<<<dim3((kKvDim + 127) / 128, rows), 128, 0, (cudaStream_t)stream>>>(e, rows, emb, kv);
    if (cudaGetLastError() != cudaSuccess) return 2;
    if (c.spy)
        if (cudaMemcpyAsync(c.kv_spy[e], kv, (size_t)rows * kKvDim * 2, cudaMemcpyDeviceToHost, (cudaStream_t)stream) != cudaSuccess) return 3;
    return 0;
}

// ---- host helpers -----------------------------------------------------------------------------------------------------------------
struct HostMath
{
    static float rsqrt(float x) { return 1.f / std::sqrt(x); }
    static float sqrt(float x) { return std::sqrt(x); }
    static float exp(float x) { return std::exp(x); }
};
static int ordered(uint16_t b) { return (b & 0x8000) ? -(int)(b & 0x7FFF) : (int)b; }

// the three sums of k_gate for one (row, stream), in the kernel's order: 512 threads x 10 strided terms, a 32-lane xor butterfly per warp,
// then thread 0 adds the 16 warp results in order
static void gate_sums_dev_order(const float* h, const uint16_t* key, const float* w, float* out_h, float* out_k, float* out_hk)
{
    float th[kGateThreads], tk[kGateThreads], thk[kGateThreads];
    for (int t = 0; t < kGateThreads; ++t)
    {
        float a = 0.f, b = 0.f, c = 0.f;
        for (int q = 0; q < kGatePer; ++q)
        {
            const int i = t + q * kGateThreads;
            const float hv = h[i], kk = bf16_to_f32(key[i]);
            const float p = hv * hv; a += p;
            const float p2 = kk * kk; b += p2;
            const float p3 = hv * w[i]; const float p4 = p3 * kk; c += p4;
        }
        th[t] = a; tk[t] = b; thk[t] = c;
    }
    float red[kGateThreads / 32][3];
    for (int wv = 0; wv < kGateThreads / 32; ++wv)
    {
        float x[3][32];
        for (int l = 0; l < 32; ++l) { x[0][l] = th[wv * 32 + l]; x[1][l] = tk[wv * 32 + l]; x[2][l] = thk[wv * 32 + l]; }
        for (int o = 16; o > 0; o >>= 1)
            for (int s = 0; s < 3; ++s)
            {
                float nx[32];
                for (int l = 0; l < 32; ++l) nx[l] = x[s][l] + x[s][l ^ o];
                memcpy(x[s], nx, sizeof nx);
            }
        red[wv][0] = x[0][0]; red[wv][1] = x[1][0]; red[wv][2] = x[2][0];
    }
    float a = 0.f, b = 0.f, c = 0.f;
    for (int k = 0; k < kGateThreads / 32; ++k) { a += red[k][0]; b += red[k][1]; c += red[k][2]; }
    *out_h = a; *out_k = b; *out_hk = c;
}
// T <- bf16(T + gate * value) for all rows and streams; kv bf16 [rows][25600]
static void host_gate_op(const std::vector<float>& Tin, const uint16_t* kv, const std::vector<float>& wgt, int rows, std::vector<float>* Tout, std::vector<float>* gates = nullptr)
{
    Tout->assign(Tin.size(), 0.f);
    if (gates) gates->assign((size_t)rows * kStreams, 0.f);
    for (int r = 0; r < rows; ++r)
        for (int j = 0; j < kStreams; ++j)
        {
            const float* h = &Tin[((size_t)r * kStreams + j) * kHid];
            const uint16_t* kvr = kv + (size_t)r * kKvDim;
            float sh, sk, shk;
            gate_sums_dev_order(h, kvr + (size_t)j * kHid, &wgt[(size_t)j * kHid], &sh, &sk, &shk);
            const float gate = gate_from_sums<HostMath>(sh, sk, shk, 1e-20f);
            if (gates) (*gates)[(size_t)r * kStreams + j] = gate;
            for (int i = 0; i < kHid; ++i)
            {
                const float val = bf16_to_f32(kvr[(size_t)kStreams * kHid + i]);
                const float p = gate * val;
                (*Tout)[((size_t)r * kStreams + j) * kHid + i] = round_bf16(h[i] + p);
            }
        }
}

static std::string cmap_path, vec_path, dir;
static std::vector<uint32_t> cmap;

static void write_file(const std::string& p, const std::string& s)
{
    FILE* f = fopen(p.c_str(), "wb");
    if (!f) { printf("cannot write %s\n", p.c_str()); exit(2); }
    fwrite(s.data(), 1, s.size(), f);
    fclose(f);
}

static void host_ids(const int32_t* win, uint32_t ids[kNE][kIds]) { hash_row(kHashParams, cmap.data(), kVocab, win, ids); }

// random window generators; a window is [pos, pos-1, pos-2, pos-3] raw token ids, -1 before the start
static std::vector<int32_t> synth_windows(int rows, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::vector<int32_t> w((size_t)rows * 4);
    for (auto& t : w) t = (int32_t)(rng() % kVocab);
    auto setw = [&](int r, int a, int b, int c, int d) { w[r * 4] = a; w[r * 4 + 1] = b; w[r * 4 + 2] = c; w[r * 4 + 3] = d; };
    if (rows >= 2) memcpy(&w[4], &w[0], 16);                    // a repeated n-gram: the same ids twice in one fetch
    if (rows >= 3) setw(2, w[8], -1, -1, -1);                   // first token of a sequence
    if (rows >= 4) setw(3, w[12], w[13], -1, -1);               // second
    if (rows >= 5) setw(4, w[16], w[17], w[18], -1);            // third
    if (rows >= 6) setw(5, 2, 2, 2, 2);                         // the pad token itself
    if (rows >= 7) setw(6, 0, 1, 129279, 5);                    // edge ids
    return w;
}

static void h2d(void* d, const void* h, size_t n) { CU(cudaMemcpy(d, h, n, cudaMemcpyHostToDevice)); CU(cudaDeviceSynchronize()); }

static float rnd_bf16_val(std::mt19937& rng, float scale)
{
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    return round_bf16(u(rng) * scale);
}

int main(int argc, char** argv)
{
    if (argc != 4 && argc != 5) { printf("usage: %s PREFIX.engram.cmap vectors.bin scratchdir [rank 0..15, default 1]\n", argv[0]); return 2; }
    cmap_path = argv[1]; vec_path = argv[2]; dir = argv[3];
    if (argc == 5) kRank = atoi(argv[4]);
    if (kRank < 0 || kRank >= kNR) { printf("rank must be 0..%d\n", kNR - 1); return 2; }
    int dev = 0; cudaDeviceProp prop;
    CU(cudaGetDevice(&dev)); CU(cudaGetDeviceProperties(&prop, dev));
    printf("device %d: %s sm_%d%d, %.1f GiB\n", dev, prop.name, prop.major, prop.minor, prop.totalGlobalMem / 1073741824.0);

    // ---- the cmap (host copy for the expected ids) and the official vectors
    std::string cm, err;
    if (!slurp(cmap_path, &cm, &err)) { printf("%s\n", err.c_str()); return 2; }
    cmap.resize(kVocab);
    memcpy(cmap.data(), cm.data() + 24, 4ull * kVocab);
    std::string vb;
    if (!slurp(vec_path, &vb, &err)) { printf("%s\n", err.c_str()); return 2; }
    struct Seq { std::vector<int32_t> tok; std::vector<uint32_t> ids; };
    std::vector<Seq> seqs;
    {
        size_t o = 0;
        auto rd = [&](void* d, size_t n) { if (o + n > vb.size()) { printf("short vectors\n"); exit(2); } memcpy(d, vb.data() + o, n); o += n; };
        char magic[8]; rd(magic, 8);
        if (memcmp(magic, "G6EVEC01", 8)) { printf("vectors magic\n"); return 2; }
        uint32_t nseq; rd(&nseq, 4);
        for (uint32_t s = 0; s < nseq; ++s)
        {
            uint32_t n; rd(&n, 4);
            Seq q; q.tok.resize(n); q.ids.resize((size_t)n * kNE * kIds);
            rd(q.tok.data(), 4ull * n); rd(q.ids.data(), 4ull * q.ids.size());
            seqs.push_back(std::move(q));
        }
    }
    auto seq_win = [&](const Seq& q, uint32_t m, int32_t* w) { for (int k = 0; k < 4; ++k) w[k] = (int32_t)m - k >= 0 ? q.tok[m - k] : -1; };

    // ---- the fake sidecar of rank 1 of 16 (sparse files: 6.2 GB logical each, a few MB of real rows)
    const std::string prefix = dir + "/gpu_r" + std::to_string(kRank);
    int64_t first[kNE], valid[kNE];
    int wfd[kNE], sfd[kNE];
    {
        std::string man = "{\n \"format\": \"g6engram-v1\",\n \"rank\": " + std::to_string(kRank) + ",\n \"tp\": " + std::to_string(kNR) + ",\n \"tables\": [";
        for (int e = 0; e < kNE; ++e)
        {
            const int64_t part = (kRows[e] + kNR - 1) / kNR;
            first[e] = std::min<int64_t>(kRank * part, kRows[e]); valid[e] = std::min<int64_t>(part, kRows[e] - first[e]);
            man += std::string(e ? "," : "") + "\n  {\"layer\": " + std::to_string(kLayerIds[e]) + ", \"rows\": " + std::to_string(kRows[e]) + ", \"part\": " + std::to_string(part) +
                   ", \"first\": " + std::to_string(first[e]) + ", \"valid\": " + std::to_string(valid[e]) + ", \"dim\": 256, \"sdim\": 8, \"w_bytes\": " + std::to_string(part * 256) +
                   ", \"s_bytes\": " + std::to_string(part * 8) + ", \"w_sha256\": \"00\", \"s_sha256\": \"00\"}";
            wfd[e] = open(table_path(prefix, kLayerIds[e], "w8").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            sfd[e] = open(table_path(prefix, kLayerIds[e], "s8").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (wfd[e] < 0 || sfd[e] < 0 || ftruncate(wfd[e], (off_t)(part * 256)) || ftruncate(sfd[e], (off_t)(part * 8))) { printf("cannot create the sidecar files in %s\n", dir.c_str()); return 2; }
        }
        man += "\n ]\n}\n";
        write_file(prefix + ".engram.json", man);
        write_file(prefix + ".engram.cmap", cm);
    }
    // rows to materialize: every window of every vector position, plus the synthetic windows used below
    std::vector<std::vector<int32_t>> synth;
    for (int rows : {8, 8, 3, 32, 1, 8})
    {
        std::vector<int32_t> w = synth_windows(rows, 1000u + (unsigned)synth.size());
        if (rows == 1)   // the one-row case checks every projection output: pick a window whose ids this rank owns in both layers (at least 2 each)
            for (unsigned sd = 5000; sd < 6000; ++sd)
            {
                w = synth_windows(1, sd);
                uint32_t ids[kNE][kIds]; host_ids(w.data(), ids);
                int own[kNE] = {0, 0};
                for (int e = 0; e < kNE; ++e) for (int j = 0; j < kIds; ++j) own[e] += (int64_t)ids[e][j] >= first[e] && (int64_t)ids[e][j] < first[e] + valid[e];
                if (own[0] >= 2 && own[1] >= 2) break;
            }
        synth.push_back(w);
    }
    // the W-row prompt of section 7: 3 * 1024 + 7 random tokens of one sequence from position 0, with a few repeated n-grams
    std::vector<int32_t> wide_prompt(3 * 1024 + 7);
    {
        std::mt19937 wr(4242);
        for (auto& t : wide_prompt) t = (int32_t)(wr() % kVocab);
        for (size_t i = 100; i + 5 < wide_prompt.size(); i += 389) wide_prompt[i + 4] = wide_prompt[i + 1];
    }
    size_t nrows_written = 0;
    {
        std::set<uint64_t> done;
        auto put = [&](const int32_t* win) {
            uint32_t ids[kNE][kIds];
            host_ids(win, ids);
            for (int e = 0; e < kNE; ++e)
                for (int j = 0; j < kIds; ++j)
                {
                    const uint32_t id = ids[e][j];
                    if ((int64_t)id < first[e] || (int64_t)id >= first[e] + valid[e] || !done.insert((uint64_t)e << 32 | id).second) continue;
                    uint8_t w[256], s[8];
                    for (int c = 0; c < 256; ++c) w[c] = fake_w(e, id, c);
                    for (int g = 0; g < 8; ++g) s[g] = fake_s(e, id, g);
                    if (pwrite(wfd[e], w, 256, (off_t)(id - first[e]) * 256) != 256 || pwrite(sfd[e], s, 8, (off_t)(id - first[e]) * 8) != 8) { printf("pwrite failed\n"); exit(2); }
                    ++nrows_written;
                }
        };
        for (const Seq& q : seqs) for (uint32_t m = 0; m < q.tok.size(); ++m) { int32_t w[4]; seq_win(q, m, w); put(w); }
        for (const auto& sw : synth) for (size_t r = 0; r < sw.size() / 4; ++r) put(&sw[r * 4]);
        for (size_t m = 0; m < wide_prompt.size(); ++m) { int32_t w[4]; for (int k = 0; k < 4; ++k) w[k] = (int)m - k >= 0 ? wide_prompt[m - k] : -1; put(w); }
        for (int e = 0; e < kNE; ++e) { fsync(wfd[e]); fsync(sfd[e]); close(wfd[e]); close(sfd[e]); }
    }
    printf("sidecar: rank %d of %d under %s, %zu fake rows written (of %lld + %lld owned)\n", kRank, kNR, prefix.c_str(), nrows_written, (long long)valid[0], (long long)valid[1]);

    // ---- device buffers: q/k weights, trunk, windows
    std::mt19937 rng(7);
    std::vector<uint16_t> qh(kNE * kStreams * kHid), kh(kNE * kStreams * kHid);
    for (auto& x : qh) x = f32_to_bf16(rnd_bf16_val(rng, 1.f));
    for (auto& x : kh) x = f32_to_bf16(rnd_bf16_val(rng, 1.f));
    bf16 *q_d, *k_d;
    CU(cudaMalloc(&q_d, qh.size() * 2)); CU(cudaMalloc(&k_d, kh.size() * 2));
    h2d(q_d, qh.data(), qh.size() * 2); h2d(k_d, kh.data(), kh.size() * 2);
    std::vector<float> wgt[kNE];
    for (int e = 0; e < kNE; ++e)
    {
        wgt[e].resize(kStreams * kHid);
        for (int i = 0; i < kStreams * kHid; ++i) wgt[e][i] = bf16_to_f32(qh[e * kStreams * kHid + i]) * bf16_to_f32(kh[e * kStreams * kHid + i]);
    }
    float* T_d; int32_t* win_d;
    CU(cudaMalloc(&T_d, (size_t)kMaxRows * kStreams * kHid * 4)); CU(cudaMalloc(&win_d, (size_t)kMaxRows * 4 * 4));
    for (int e = 0; e < kNE; ++e)
    {
        CU(cudaHostAlloc(&g.emb_spy[e], (size_t)kMaxRows * kEmbDim * 2, cudaHostAllocDefault));
        CU(cudaHostAlloc(&g.kv_spy[e], (size_t)kMaxRows * kKvDim * 2, cudaHostAllocDefault));
    }
    cudaStream_t cs;
    CU(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));

    // ---- boot
    Engram eng;
    {
        Config c;
        c.prefix = prefix; c.rank = kRank; c.nr = kNR; c.max_rows = kMaxRows;
        c.knobs.on = true; c.knobs.store.cache_bytes = 64ull << 20; c.knobs.store.ways = 16; c.knobs.store.io_threads = 16; c.knobs.verbose = false;
        for (int e = 0; e < kNE; ++e) { c.q_w[e] = q_d + e * kStreams * kHid; c.k_w[e] = k_d + e * kStreams * kHid; }
        c.hooks.reduce = &hook_reduce; c.hooks.project = &hook_project;
        const auto t0 = std::chrono::steady_clock::now();
        if (!eng.init(c, &err)) { printf("FAIL init: %s\n", err.c_str()); return 1; }
        printf("init ok in %.2f s (s8 to the device, cmap, q*k, store open; O_DIRECT: L1 %s, L14 %s)\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
               eng.store().direct(0) ? "yes" : "NO (buffered fallback)", eng.store().direct(1) ? "yes" : "NO (buffered fallback)");
    }

    // ---- 1. ids vs the official vectors, batches of up to 32 rows
    {
        long long npos = 0, nid = 0, bad = 0;
        std::vector<int32_t> w((size_t)kMaxRows * 4);
        std::vector<uint32_t> got((size_t)kNE * kMaxRows * kIds);
        for (const Seq& q : seqs)
            for (uint32_t m0 = 0; m0 < q.tok.size(); m0 += kMaxRows)
            {
                const int rows = (int)std::min<uint32_t>(kMaxRows, (uint32_t)q.tok.size() - m0);
                for (int r = 0; r < rows; ++r) seq_win(q, m0 + r, &w[r * 4]);
                h2d(win_d, w.data(), (size_t)rows * 16);
                if (int x = eng.begin_step(rows, win_d, cs)) { printf("FAIL begin_step rc %d\n", x); return 1; }
                CU(cudaDeviceSynchronize());
                CU(cudaMemcpy(got.data(), eng.ids_device(), (size_t)kNE * rows * kIds * 4, cudaMemcpyDeviceToHost));
                for (int r = 0; r < rows; ++r)
                    for (int e = 0; e < kNE; ++e)
                        for (int j = 0; j < kIds; ++j)
                        {
                            const uint32_t want = q.ids[(((size_t)(m0 + r)) * kNE + e) * kIds + j];
                            ++nid;
                            if (got[((size_t)e * rows + r) * kIds + j] != want && bad++ < 10) printf("  id mismatch pos %u layer %d id %d: got %u want %u\n", m0 + r, e, j, got[((size_t)e * rows + r) * kIds + j], want);
                        }
                npos += rows;
            }
        CHECK(bad == 0 && eng.status() == 0, "%lld of %lld device ids differ from the official vectors (status %d)", bad, nid, eng.status());
        printf("1. ids: %lld positions, %lld ids from k_hash bit-identical to the official vectors (status %d)\n", npos, nid, eng.status());
    }

    // ---- 2-4. the full op chain on a window set, stepwise (strict) and async (determinism), host-checked
    auto run_case = [&](const char* name, const std::vector<int32_t>& win, bool full_proj_check) {
        const int rows = (int)win.size() / 4;
        std::vector<float> T0((size_t)rows * kStreams * kHid);
        for (auto& t : T0) t = rnd_bf16_val(rng, 4.f);
        h2d(win_d, win.data(), win.size() * 4); h2d(T_d, T0.data(), T0.size() * 4);
        g.spy = true;
        const FetchStats s0 = eng.store().totals();
        if (int x = eng.begin_step(rows, win_d, cs)) { printf("FAIL begin_step rc %d\n", x); exit(1); }
        uint32_t hid[kMaxRows][kNE][kIds];
        for (int r = 0; r < rows; ++r) host_ids(&win[r * 4], hid[r]);
        std::vector<float> Tin = T0, Tdev(T0.size()), Thost;
        long long bits_diff = 0, over = 0, nelem = 0; int max_ulp = 0;
        long long emb_bad = 0, kv_bad = 0, kv_checked = 0, owned = 0, asked = 0;
        for (int e = 0; e < kNE; ++e)
        {
            if (int x = eng.op(&g, e, rows, T_d, cs)) { printf("FAIL op rc %d\n", x); exit(1); }
            CU(cudaStreamSynchronize(cs));
            CU(cudaMemcpy(Tdev.data(), T_d, Tdev.size() * 4, cudaMemcpyDeviceToHost));
            // (2) the embedding at the reduce hook
            for (int r = 0; r < rows; ++r)
                for (int gi = 0; gi < kIds; ++gi)
                {
                    const uint32_t id = hid[r][e][gi];
                    const bool mine = (int64_t)id >= first[e] && (int64_t)id < first[e] + valid[e];
                    ++asked; owned += mine;
                    for (int c = 0; c < kRowBytes; ++c)
                    {
                        const uint16_t want = mine ? dequant1(fake_w(e, id, c), fake_s(e, id, c >> 5)) : 0;
                        const uint16_t got = g.emb_spy[e][(size_t)r * kEmbDim + (size_t)gi * kRowBytes + c];
                        if (got != want && emb_bad++ < 5) printf("  emb mismatch row %d layer %d id#%d col %d: got %04x want %04x (%s)\n", r, e, gi, c, got, want, mine ? "owned" : "foreign");
                    }
                }
            // (3a) the fake projection: sampled outputs (all 25600 for the full check), double accumulate on the host
            std::vector<int> os;
            if (full_proj_check) for (int o = 0; o < kKvDim; ++o) os.push_back(o);
            else { os = {0, 1, 2, 255, 5119, 5120, 5121, 10239, 15360, 20479, 20480, 25598, 25599}; for (int i = 0; i < 40; ++i) os.push_back((int)(rng() % kKvDim)); }
            for (int r = 0; r < rows; ++r)
                for (int o : os)
                {
                    ++kv_checked;
                    const float want = host_project1(e, &g.emb_spy[e][(size_t)r * kEmbDim], o);
                    const float got = bf16_to_f32(g.kv_spy[e][(size_t)r * kKvDim + o]);
                    if (f32_bits(got) != f32_bits(want) && kv_bad++ < 5) printf("  kv mismatch row %d layer %d out %d: got %.9g want %.9g\n", r, e, o, got, want);
                }
            // (3b) the gate + write: host emulation from the device's own inputs (T before the op, kv after the project hook)
            host_gate_op(Tin, g.kv_spy[e], wgt[e], rows, &Thost);
            for (size_t i = 0; i < Tdev.size(); ++i)
            {
                const uint16_t a = f32_to_bf16(Tdev[i]), b = f32_to_bf16(Thost[i]);
                ++nelem;
                if (a != b) ++bits_diff;
                const int d = std::abs(ordered(a) - ordered(b));
                if (d > max_ulp) max_ulp = d;
                if (d > 1) ++over;
                if (Tdev[i] != round_bf16(Tdev[i])) ++over;   // the trunk stays BF16-valued
            }
            Tin = Tdev;
        }
        // ids of this step on the device
        std::vector<uint32_t> gid((size_t)kNE * rows * kIds);
        CU(cudaMemcpy(gid.data(), eng.ids_device(), gid.size() * 4, cudaMemcpyDeviceToHost));
        long long id_bad = 0;
        for (int r = 0; r < rows; ++r) for (int e = 0; e < kNE; ++e) for (int j = 0; j < kIds; ++j) id_bad += gid[((size_t)e * rows + r) * kIds + j] != hid[r][e][j];
        const FetchStats s1 = eng.store().totals();
        CHECK(id_bad == 0, "%s: %lld ids differ", name, id_bad);
        CHECK(emb_bad == 0, "%s: %lld embedding elements differ from the host dequant", name, emb_bad);
        CHECK(kv_bad == 0, "%s: %lld of %lld sampled projection outputs differ", name, kv_bad, kv_checked);
        CHECK(over == 0 && max_ulp <= 1, "%s: gate/write: %lld elements beyond 1 bf16 ulp or not BF16-valued (max %d ulp)", name, over, max_ulp);
        CHECK(eng.status() == 0, "%s: status %d", name, eng.status());
        // the async chain on a fresh copy of T0: no syncs between begin_step, op, op -> bit-identical to the stepwise result
        g.spy = false;
        h2d(T_d, T0.data(), T0.size() * 4);
        const auto t0 = std::chrono::steady_clock::now();
        if (int x = eng.begin_step(rows, win_d, cs)) { printf("FAIL begin_step rc %d\n", x); exit(1); }
        for (int e = 0; e < kNE; ++e) if (int x = eng.op(&g, e, rows, T_d, cs)) { printf("FAIL op rc %d\n", x); exit(1); }
        CU(cudaStreamSynchronize(cs));
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::vector<float> Tasync(T0.size());
        CU(cudaMemcpy(Tasync.data(), T_d, Tasync.size() * 4, cudaMemcpyDeviceToHost));
        CHECK(memcmp(Tasync.data(), Tdev.data(), Tdev.size() * 4) == 0, "%s: async chain differs from the stepwise run", name);
        const FetchStats s2 = eng.store().totals();
        CHECK(s2.miss_req == s1.miss_req && s2.req > s1.req, "%s: the repeat missed %u times (asked %u)", name, s2.miss_req - s1.miss_req, s2.req - s1.req);
        printf("%s rows=%d: ids ok, emb %lld/%lld ids owned bit-exact, proj %lld outputs bit-exact, gate/write %lld elements (%lld not bit-identical, max %d ulp), async == stepwise; first pass req %u miss %u foreign %u, repeat miss %u; async chain %.2f ms (fake double-precision projection, not representative)\n",
               name, rows, owned, asked, kv_checked, nelem, bits_diff, max_ulp, s1.req - s0.req, s1.miss_req - s0.miss_req, s1.foreign - s0.foreign, s2.miss_req - s1.miss_req, ms);
        return Tdev;
    };

    // a real-text window set: 8 consecutive positions of the first vector sequence
    std::vector<int32_t> w_real;
    {
        const Seq& q = seqs[0];
        const uint32_t m0 = q.tok.size() > 40 ? 20 : 0;
        w_real.resize(32);
        for (int r = 0; r < 8; ++r) seq_win(q, m0 + r, &w_real[r * 4]);
    }
    run_case("real8   ", w_real, false);
    const std::vector<float> ref_synth8 = run_case("synth8  ", synth[0], false);
    run_case("synth8b ", synth[1], false);
    run_case("synth3  ", synth[2], false);      // a different rows value reuses the Jobs machinery
    run_case("synth32 ", synth[3], false);
    run_case("synth1  ", synth[4], true);       // every one of the 25600 outputs of the fake projection, both layers

    // ---- 5. CUDA graph: begin_step + op + op captured once, replayed with new windows
    {
        const int rows = 8;
        const std::vector<int32_t>& wa = synth[0];
        const std::vector<int32_t>& wb = synth[5];
        std::vector<float> T0((size_t)rows * kStreams * kHid);
        for (auto& t : T0) t = rnd_bf16_val(rng, 4.f);
        // eager references (same code path as the stepwise run, no spies)
        g.spy = false;
        auto eager = [&](const std::vector<int32_t>& w) {
            h2d(win_d, w.data(), w.size() * 4); h2d(T_d, T0.data(), T0.size() * 4);
            eng.begin_step(rows, win_d, cs);
            for (int e = 0; e < kNE; ++e) eng.op(&g, e, rows, T_d, cs);
            CU(cudaStreamSynchronize(cs));
            std::vector<float> o(T0.size()); CU(cudaMemcpy(o.data(), T_d, o.size() * 4, cudaMemcpyDeviceToHost));
            return o;
        };
        const std::vector<float> ra = eager(wa), rb = eager(wb);
        CHECK(memcmp(ra.data(), rb.data(), ra.size() * 4) != 0, "graph test: the two window sets must give different results");
        g.capturing = true;
        cudaGraph_t graph; cudaGraphExec_t exec;
        CU(cudaStreamBeginCapture(cs, cudaStreamCaptureModeGlobal));
        int rc = eng.begin_step(rows, win_d, cs);
        for (int e = 0; e < kNE && !rc; ++e) rc = eng.op(&g, e, rows, T_d, cs);
        CHECK(rc == 0, "graph capture: begin_step/op rc %d", rc);
        CU(cudaStreamEndCapture(cs, &graph));
        CU(cudaGraphInstantiate(&exec, graph, 0));
        g.capturing = false;
        size_t nnodes = 0; CU(cudaGraphGetNodes(graph, nullptr, &nnodes));
        std::vector<float> o(T0.size());
        for (int rep = 0; rep < 2; ++rep)
        {
            const std::vector<int32_t>& w = rep == 0 ? wa : wb;
            h2d(win_d, w.data(), w.size() * 4); h2d(T_d, T0.data(), T0.size() * 4);
            const FetchStats a = eng.store().totals();
            const auto t0 = std::chrono::steady_clock::now();
            CU(cudaGraphLaunch(exec, cs));
            CU(cudaStreamSynchronize(cs));
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            CU(cudaMemcpy(o.data(), T_d, o.size() * 4, cudaMemcpyDeviceToHost));
            const std::vector<float>& want = rep == 0 ? ra : rb;
            CHECK(memcmp(o.data(), want.data(), o.size() * 4) == 0, "graph replay %d differs from the eager run", rep);
            CHECK(eng.status() == 0, "graph replay %d: status %d", rep, eng.status());
            printf("5. graph replay %d (%zu graph nodes, windows %c): %s the eager run, %u rows asked, %.2f ms\n", rep, nnodes, rep == 0 ? 'A' : 'B',
                   memcmp(o.data(), want.data(), o.size() * 4) == 0 ? "bit-identical to" : "DIFFERS from", eng.store().totals().req - a.req, ms);
        }
        CU(cudaGraphExecDestroy(exec)); CU(cudaGraphDestroy(graph));
    }

    // ---- 6. fail-closed
    {
        EngramHooks none;
        eng.set_hooks(none);
        CHECK(eng.op(&g, 0, 8, T_d, cs) == kNotSup, "op without hooks must return 801");
        EngramHooks h; h.reduce = &hook_reduce; h.project = &hook_project;
        eng.set_hooks(h);
        CHECK(eng.begin_step(0, win_d, cs) != 0 && eng.begin_step(kMaxRows + 1, win_d, cs) != 0, "rows outside 1..max_rows must be refused");
        CHECK(eng.op(&g, 2, 8, T_d, cs) != 0, "layer slot 2 must be refused");
        std::vector<int32_t> bad = {5, 6, 7, 8, (int32_t)kVocab, 1, 2, 3};
        h2d(win_d, bad.data(), bad.size() * 4);
        CHECK(eng.status() == 0, "status before the bad token");
        eng.begin_step(2, win_d, cs);
        CU(cudaDeviceSynchronize());
        CHECK(eng.status() == -1, "a token outside the vocabulary must make status() -1 (got %d)", eng.status());
        printf("6. fail-closed: no hooks -> %d, rows 0 / %d refused, layer slot 2 refused, token %u -> status %d\n", kNotSup, kMaxRows + 1, kVocab, eng.status());
    }

    // ---- 7. W-row chunks (prefill): window_wide + begin_step + op at W = 1024, 2048 and odd sizes; k_act_quant_rows; status -2
    {
        constexpr int WMAX = 2048;
        Ctx gw;
        gw.spy = true;
        for (int e = 0; e < kNE; ++e)
        {
            CU(cudaHostAlloc(&gw.emb_spy[e], (size_t)WMAX * kEmbDim * 2, cudaHostAllocDefault));
            CU(cudaHostAlloc(&gw.kv_spy[e], (size_t)WMAX * kKvDim * 2, cudaHostAllocDefault));
        }
        Engram engw;
        {
            Config c;
            c.prefix = prefix; c.rank = kRank; c.nr = kNR; c.max_rows = WMAX;
            c.knobs.on = true; c.knobs.store.cache_bytes = 64ull << 20; c.knobs.store.ways = 16; c.knobs.store.io_threads = 16; c.knobs.verbose = false;
            for (int e = 0; e < kNE; ++e) { c.q_w[e] = q_d + e * kStreams * kHid; c.k_w[e] = k_d + e * kStreams * kHid; }
            c.hooks.reduce = &hook_reduce; c.hooks.project = &hook_project_cheap;   // no wkv tensors: the hook runs (the tensor-core GEMM is not used here)
            if (!engw.init(c, &err)) { printf("FAIL init (W-row engine, max_rows %d): %s\n", WMAX, err.c_str()); return 1; }
        }
        int *tok_d, *pos_d, *ring_d; int32_t* wwin_d; float* Tw_d;
        CU(cudaMalloc(&tok_d, WMAX * 4)); CU(cudaMalloc(&pos_d, WMAX * 4)); CU(cudaMalloc(&ring_d, kRing * 4));
        CU(cudaMalloc(&wwin_d, (size_t)WMAX * 16)); CU(cudaMalloc(&Tw_d, (size_t)WMAX * kStreams * kHid * 4));
        const int N = (int)wide_prompt.size();
        auto true_win = [&](int p, int k) { return p - k >= 0 ? wide_prompt[p - k] : -1; };
        auto run_plan = [&](const char* name, const std::vector<int>& plan, bool ops) {
            std::vector<int32_t> ring0(kRing, -777);   // garbage: a row before the start of the sequence must never read it
            h2d(ring_d, ring0.data(), kRing * 4);
            std::vector<int> hring(kRing, -777);
            long long win_bad = 0, ring_bad = 0, id_bad = 0, emb_bad = 0, kv_bad = 0, gate_bad = 0, gate_over1 = 0, gate_cancel = 0, chunks = 0, nrows = 0, nops = 0;
            int p0 = 0; size_t ci = 0;
            while (p0 < N)
            {
                const int rows = std::min(ci < plan.size() ? plan[ci] : N - p0, N - p0);
                ++ci;
                std::vector<int32_t> tok(rows), pos(rows);
                for (int r = 0; r < rows; ++r) { tok[r] = wide_prompt[p0 + r]; pos[r] = p0 + r; }
                h2d(tok_d, tok.data(), rows * 4); h2d(pos_d, pos.data(), rows * 4);
                if (int x = engw.window_wide(tok_d, pos_d, rows, ring_d, wwin_d, cs)) { printf("FAIL window_wide rc %d (rows %d)\n", x, rows); exit(1); }
                CU(cudaStreamSynchronize(cs));
                std::vector<int32_t> win((size_t)rows * 4);
                CU(cudaMemcpy(win.data(), wwin_d, win.size() * 4, cudaMemcpyDeviceToHost));
                for (int r = 0; r < rows; ++r) for (int k = 0; k < 4; ++k) win_bad += win[(size_t)r * 4 + k] != true_win(p0 + r, k);
                for (int r = 0; r < rows; ++r) ring_put(hring.data(), p0 + r, tok[r]);   // the 32-row path's ring: every row written in position order
                std::vector<int32_t> ringd(kRing);
                CU(cudaMemcpy(ringd.data(), ring_d, kRing * 4, cudaMemcpyDeviceToHost));
                for (int i = 0; i < kRing; ++i) ring_bad += ringd[i] != hring[i];
                if (int x = engw.begin_step(rows, wwin_d, cs)) { printf("FAIL begin_step rc %d (rows %d)\n", x, rows); exit(1); }
                CU(cudaDeviceSynchronize());   // the hash runs on the engine's side stream; syncing only `cs` read stale ids on chunks without ops (first GPU run)
                std::vector<uint32_t> gid((size_t)kNE * rows * kIds);
                CU(cudaMemcpy(gid.data(), engw.ids_device(), gid.size() * 4, cudaMemcpyDeviceToHost));
                std::vector<uint32_t> hid((size_t)rows * kNE * kIds);
                for (int r = 0; r < rows; ++r)
                {
                    int32_t w4[4]; for (int k = 0; k < 4; ++k) w4[k] = true_win(p0 + r, k);
                    uint32_t ids[kNE][kIds]; host_ids(w4, ids);
                    for (int e = 0; e < kNE; ++e) for (int j = 0; j < kIds; ++j)
                    {
                        hid[((size_t)r * kNE + e) * kIds + j] = ids[e][j];
                        id_bad += gid[((size_t)e * rows + r) * kIds + j] != ids[e][j];
                    }
                }
                if (ops && rows > 32)
                {
                    std::vector<float> T0((size_t)rows * kStreams * kHid);
                    for (auto& t : T0) t = rnd_bf16_val(rng, 4.f);
                    h2d(Tw_d, T0.data(), T0.size() * 4);
                    std::vector<float> Tin = T0, Tdev(T0.size()), Thost;
                    for (int e = 0; e < kNE; ++e)
                    {
                        if (int x = engw.op(&gw, e, rows, Tw_d, cs)) { printf("FAIL op rc %d (rows %d)\n", x, rows); exit(1); }
                        CU(cudaStreamSynchronize(cs));
                        CU(cudaMemcpy(Tdev.data(), Tw_d, Tdev.size() * 4, cudaMemcpyDeviceToHost));
                        for (int r = 0; r < rows; ++r)
                            for (int gi = 0; gi < kIds; ++gi)
                            {
                                const uint32_t id = hid[((size_t)r * kNE + e) * kIds + gi];
                                const bool mine = (int64_t)id >= first[e] && (int64_t)id < first[e] + valid[e];
                                for (int c = 0; c < kRowBytes; ++c)
                                {
                                    const uint16_t want = mine ? dequant1(fake_w(e, id, c), fake_s(e, id, c >> 5)) : 0;
                                    emb_bad += gw.emb_spy[e][(size_t)r * kEmbDim + (size_t)gi * kRowBytes + c] != want;
                                }
                            }
                        for (int r = 0; r < rows; ++r)
                            for (int o = 0; o < kKvDim; ++o)
                                kv_bad += gw.kv_spy[e][(size_t)r * kKvDim + o] != gw.emb_spy[e][(size_t)r * kEmbDim + (size_t)((o * 37 + e) % kEmbDim)];
                        std::vector<float> gates;
                        host_gate_op(Tin, gw.kv_spy[e], wgt[e], rows, &Thost, &gates);
                        for (size_t i = 0; i < Tdev.size(); ++i)
                        {
                            const bool nonbf = Tdev[i] != round_bf16(Tdev[i]);   // the trunk stays BF16-valued (a NaN counts)
                            gate_bad += nonbf;
                            if (std::abs(ordered(f32_to_bf16(Tdev[i])) - ordered(f32_to_bf16(Thost[i]))) <= 1 && !nonbf) continue;
                            if (!nonbf)
                            {
                                // more than 1 bf16 ulp from the host emulation. The host gate (std::exp / 1/sqrt) and the device gate (expf / rsqrtf) can differ by a few fp32 ulp;
                                // when T and gate*value nearly cancel, that moves the tiny result by many bf16 ulp of the RESULT but not of the operands. Count it as a
                                // cancellation case only if the difference stays within 2^-12 of the larger operand; anything bigger is a real mismatch.
                                ++gate_over1;
                                const size_t rj = i / kHid, r = rj / kStreams, ci = i % kHid;
                                const float val = bf16_to_f32(gw.kv_spy[e][r * kKvDim + (size_t)kStreams * kHid + ci]);
                                const double gv = (double)gates[rj] * (double)val, tin = Tin[i];
                                const double scale = std::max(std::abs(tin), std::abs(gv));
                                const bool cancel = std::abs((double)Tdev[i] - (double)Thost[i]) <= scale * 0.000244140625;
                                if (cancel) ++gate_cancel; else ++gate_bad;
                                if (gate_over1 <= 6)
                                    printf("   gate/write element beyond 1 ulp (%s): %s layer slot %d chunk row %zu stream %zu col %zu: T in %.9g, host gate %.9g, value %.9g, device %.9g, host %.9g, exact(double) %.9g\n",
                                           cancel ? "cancellation, within 2^-12 of the operands" : "MISMATCH", name, e, r, rj % kStreams, ci, tin, (double)gates[rj], (double)val, (double)Tdev[i], (double)Thost[i], tin + gv);
                            }
                            else if (gate_bad <= 6) printf("   not BF16-valued: %s layer slot %d element %zu = %.9g\n", name, e, i, (double)Tdev[i]);
                        }
                        Tin = Tdev;
                    }
                    ++nops;
                }
                p0 += rows; nrows += rows; ++chunks;
            }
            CHECK(win_bad == 0, "%s: %lld window entries differ from the true sequence", name, win_bad);
            CHECK(ring_bad == 0, "%s: %lld ring slots differ from the 32-row path's ring", name, ring_bad);
            CHECK(id_bad == 0, "%s: %lld device ids differ from the host hash of the true windows", name, id_bad);
            CHECK(emb_bad == 0 && kv_bad == 0, "%s: %lld embedding elements / %lld cheap-projection outputs differ", name, emb_bad, kv_bad);
            CHECK(gate_bad == 0, "%s: %lld gate/write elements beyond 1 bf16 ulp (and not a cancellation case) or not BF16-valued", name, gate_bad);
            g_gate_cancel_total += gate_cancel;
            CHECK(engw.status() == 0, "%s: status %d", name, engw.status());
            printf("7. %s: %lld chunks, %lld rows, window/ring/ids exact; %lld op chains (emb, kv, gate/write) checked (%lld gate/write elements beyond 1 ulp of the host emulation, %lld of them cancellation cases); status %d\n", name, chunks, nrows, nops, gate_over1, gate_cancel, engw.status());
        };
        run_plan("W1024     ", {1024, 1024, 1024, 7}, true);
        run_plan("W2048     ", {2048, 1031}, true);
        run_plan("odd sizes ", {61, 62, 63, 64, 65, 127, 128, 129, 33, 1, 2, 3, 500, 1000}, false);   // around kRing and 2 x kRing; the last chunk takes the rest
        // k_act_quant_rows against the host lines of act_quant.h (the host lines are checked against an independent official-formula reference by tests/test_act_quant.cpp)
        {
            constexpr int R = 64;
            std::vector<uint16_t> x((size_t)R * kEmbDim);
            std::mt19937 ar(99);
            for (int r = 0; r < R; ++r)
                for (int i = 0; i < kEmbDim; ++i)
                {
                    float v = rnd_bf16_val(ar, std::ldexp(1.f, (int)(ar() % 24) - 12));
                    if (r % 8 == 0) v = 0.f;                          // all-zero blocks: the 1e-4 floor
                    else if (r % 8 == 1) v = rnd_bf16_val(ar, 1e-6f);   // below the floor
                    else if (r % 8 == 2 && i % 97 == 0) v *= 3000.f;  // outliers: saturation
                    x[(size_t)r * kEmbDim + i] = f32_to_bf16(v);
                }
            bf16* x_d; uint8_t *c_d, *s_d;
            CU(cudaMalloc(&x_d, x.size() * 2)); CU(cudaMalloc(&c_d, (size_t)R * kEmbDim)); CU(cudaMalloc(&s_d, (size_t)R * (kEmbDim / 32)));
            h2d(x_d, x.data(), x.size() * 2);
            k_act_quant_rows<<<(unsigned)(((long)R * (kEmbDim / 32) + 7) / 8), 256>>>(x_d, R, c_d, s_d);
            CU(cudaGetLastError()); CU(cudaDeviceSynchronize());
            std::vector<uint8_t> cc((size_t)R * kEmbDim), ss((size_t)R * (kEmbDim / 32));
            CU(cudaMemcpy(cc.data(), c_d, cc.size(), cudaMemcpyDeviceToHost)); CU(cudaMemcpy(ss.data(), s_d, ss.size(), cudaMemcpyDeviceToHost));
            long long aq_bad = 0;
            for (int r = 0; r < R; ++r)
                for (int b = 0; b < kEmbDim / 32; ++b)
                {
                    float amax = 0.f;
                    for (int l = 0; l < 32; ++l) amax = std::fmax(amax, std::fabs(bf16_to_f32(x[(size_t)r * kEmbDim + b * 32 + l])));
                    const int e = aq_exp(amax);
                    aq_bad += ss[(size_t)r * (kEmbDim / 32) + b] != aq_scale_byte(e);
                    for (int l = 0; l < 32; ++l) aq_bad += cc[(size_t)r * kEmbDim + b * 32 + l] != aq_code(bf16_to_f32(x[(size_t)r * kEmbDim + b * 32 + l]), e);
                }
            CHECK(aq_bad == 0, "k_act_quant_rows: %lld codes / scale bytes differ from the host act_quant.h lines", aq_bad);
            printf("7. k_act_quant_rows: %d rows x %d, codes + scales == the host lines: %s\n", R, kEmbDim, aq_bad ? "NO" : "yes");
            cudaFree(x_d); cudaFree(c_d); cudaFree(s_d);
        }
        // rows above max_rows are refused; rows that are not consecutive positions fail closed (status -2); last, because the flag is sticky
        {
            CHECK(engw.window_wide(tok_d, pos_d, WMAX + 1, ring_d, wwin_d, cs) != 0 && engw.window_wide(tok_d, pos_d, 0, ring_d, wwin_d, cs) != 0, "window_wide rows outside 1..max_rows must be refused");
            std::vector<int32_t> tok(40), pos(40);
            for (int r = 0; r < 40; ++r) { tok[r] = 11 + r; pos[r] = 500 + r; }
            pos[17] = 999;
            h2d(tok_d, tok.data(), 40 * 4); h2d(pos_d, pos.data(), 40 * 4);
            CHECK(engw.status() == 0, "status before the bad positions");
            CHECK(engw.window_wide(tok_d, pos_d, 40, ring_d, wwin_d, cs) == 0, "window_wide launch");
            CU(cudaDeviceSynchronize());
            CHECK(engw.status() == -2, "rows that are not consecutive positions must make status() -2 (got %d)", engw.status());
            printf("7. fail-closed: rows > max_rows refused, non-consecutive positions -> status %d\n", engw.status());
        }
        printf("W-row engine: %s\n", engw.stats_line().c_str());
    }

    // ---- 7d. the tensor-core wkv seam: act-quant -> ds41_gemm_mx_tc with a synthetic full-size wkv pack, W1024 then W2048
    {
        constexpr int WMAX = 2048;
        CHECK(ds41_gemm_mx_tc != nullptr, "7d: ds41_gemm_mx_tc is not linked (the test must be built with engine/attn/ds41_attn.cu)");
        uint8_t *pc[kNE], *ps[kNE];
        for (int e = 0; e < kNE; ++e)
        {
            CU(cudaMalloc(&pc[e], (size_t)kKvDim * kEmbDim)); CU(cudaMalloc(&ps[e], (size_t)kKvDim * (kEmbDim / 32)));
            k_fill_pack<<<(unsigned)(((size_t)kKvDim * kEmbDim + 255) / 256), 256>>>(e, pc[e], ps[e]);
            CU(cudaGetLastError());
        }
        CU(cudaDeviceSynchronize());
        Ctx gt; gt.spy = true;
        for (int e = 0; e < kNE; ++e) CU(cudaHostAlloc(&gt.emb_spy[e], (size_t)WMAX * kEmbDim * 2, cudaHostAllocDefault));
        Ctx gn;   // no spy: the timing runs
        Engram engt;
        {
            Config c;
            c.prefix = prefix; c.rank = kRank; c.nr = kNR; c.max_rows = WMAX;
            c.knobs.on = true; c.knobs.store.cache_bytes = 64ull << 20; c.knobs.store.ways = 16; c.knobs.store.io_threads = 16; c.knobs.verbose = false;
            for (int e = 0; e < kNE; ++e) { c.q_w[e] = q_d + e * kStreams * kHid; c.k_w[e] = k_d + e * kStreams * kHid; c.wkv_codes[e] = pc[e]; c.wkv_scales[e] = ps[e]; }
            c.hooks.reduce = &hook_reduce; c.hooks.project = &hook_project_cheap;   // must not run for these chunks (rows > 32, tensor cores wired)
            if (!engt.init(c, &err)) { printf("FAIL init (7d engine): %s\n", err.c_str()); return 1; }
        }
        int *tok_d, *pos_d, *ring_d; int32_t* wwin_d; float* Tw_d;
        CU(cudaMalloc(&tok_d, WMAX * 4)); CU(cudaMalloc(&pos_d, WMAX * 4)); CU(cudaMalloc(&ring_d, kRing * 4));
        CU(cudaMalloc(&wwin_d, (size_t)WMAX * 16)); CU(cudaMalloc(&Tw_d, (size_t)WMAX * kStreams * kHid * 4));
        std::vector<int32_t> ring0(kRing, -777);
        h2d(ring_d, ring0.data(), kRing * 4);
        double e4[256]; for (int c = 0; c < 256; ++c) e4[c] = e4m3_val((uint8_t)c);
        std::vector<uint16_t> kvh((size_t)WMAX * kKvDim);
        std::mt19937 tr(777);
        long long tc_bad = 0, tc_checked = 0;
        int p0 = 0;
        for (const int rows : {1024, 2048})
        {
            std::vector<int32_t> tok(rows), pos(rows);
            for (int r = 0; r < rows; ++r) { tok[r] = wide_prompt[p0 + r]; pos[r] = p0 + r; }
            h2d(tok_d, tok.data(), rows * 4); h2d(pos_d, pos.data(), rows * 4);
            if (int x = engt.window_wide(tok_d, pos_d, rows, ring_d, wwin_d, cs)) { printf("FAIL 7d window_wide rc %d\n", x); return 1; }
            if (int x = engt.begin_step(rows, wwin_d, cs)) { printf("FAIL 7d begin_step rc %d\n", x); return 1; }
            CU(cudaStreamSynchronize(cs));
            std::vector<float> T0((size_t)rows * kStreams * kHid);
            for (auto& t : T0) t = rnd_bf16_val(tr, 4.f);
            h2d(Tw_d, T0.data(), T0.size() * 4);
            // sampled rows (edges of the 128-row M tiles, of the 32-row slabs, the last rows) and columns (edges of the 64/128-column tiles, of the 4 keys + 1 value streams)
            std::vector<int> srow, scol;
            for (int r : {0, 1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 511, 512, 513, 1023, 1024, 1025, 2046, 2047, rows / 2, rows - 2, rows - 1}) if (r >= 0 && r < rows) srow.push_back(r);
            for (int i = 0; i < 6; ++i) srow.push_back((int)(tr() % rows));
            for (int n : {0, 1, 7, 8, 63, 64, 65, 127, 128, 129, 5119, 5120, 5121, 10239, 10240, 15359, 15360, 20479, 20480, 25535, 25598, 25599}) scol.push_back(n);
            for (int i = 0; i < 26; ++i) scol.push_back((int)(tr() % kKvDim));
            std::sort(srow.begin(), srow.end()); srow.erase(std::unique(srow.begin(), srow.end()), srow.end());
            for (int e = 0; e < kNE; ++e)
            {
                const int calls0 = gt.project_calls;
                if (int x = engt.op(&gt, e, rows, Tw_d, cs)) { printf("FAIL 7d op rc %d (rows %d)\n", x, rows); return 1; }
                CU(cudaStreamSynchronize(cs));
                CHECK(gt.project_calls == calls0, "7d: the project hook ran at rows %d, layer slot %d (the tensor-core path was refused or not wired)", rows, e);
                CU(cudaMemcpy(kvh.data(), engt.kv_device(), (size_t)rows * kKvDim * 2, cudaMemcpyDeviceToHost));
                double se = 0, sr = 0, maxrel = 0; long long bad = 0, cnt = 0;
                std::vector<double> ref(srow.size() * scol.size());
                double rms = 0;
                std::vector<uint8_t> ac(kEmbDim); std::vector<int> ea(kEmbDim / 32);
                for (size_t ri = 0; ri < srow.size(); ++ri)
                {
                    const uint16_t* x = gt.emb_spy[e] + (size_t)srow[ri] * kEmbDim;
                    for (int b = 0; b < kEmbDim / 32; ++b)
                    {
                        float amax = 0.f;
                        for (int l = 0; l < 32; ++l) amax = std::fmax(amax, std::fabs(bf16_to_f32(x[b * 32 + l])));
                        const int ex = aq_exp(amax);
                        ea[b] = ex;
                        for (int l = 0; l < 32; ++l) ac[b * 32 + l] = aq_code(bf16_to_f32(x[b * 32 + l]), ex);
                    }
                    for (size_t ci = 0; ci < scol.size(); ++ci)
                    {
                        const uint32_t n = (uint32_t)scol[ci];
                        double acc = 0;
                        for (int b = 0; b < kEmbDim / 32; ++b)
                        {
                            double blk = 0;
                            for (int l = 0; l < 32; ++l) blk += e4[ac[b * 32 + l]] * e4[pk_code(e, n, (uint32_t)(b * 32 + l))];
                            acc += blk * std::ldexp(1.0, ea[b]) * std::ldexp(1.0, (int)pk_scale(e, n, (uint32_t)b) - 127);
                        }
                        ref[ri * scol.size() + ci] = acc; rms += acc * acc;
                    }
                }
                rms = std::sqrt(rms / (double)ref.size());
                for (size_t ri = 0; ri < srow.size(); ++ri)
                    for (size_t ci = 0; ci < scol.size(); ++ci)
                    {
                        const double want = ref[ri * scol.size() + ci];
                        const double got = bf16_to_f32(kvh[(size_t)srow[ri] * kKvDim + scol[ci]]);
                        const double d = std::fabs(got - want);
                        se += d * d; sr += want * want; ++cnt;
                        maxrel = std::max(maxrel, d / (std::fabs(want) + 1e-3 * rms));
                        bad += !(d <= std::ldexp(1.0, -7) * std::fabs(want) + std::ldexp(1.0, -12) * rms);   // a NaN fails too
                    }
                const double rl2 = std::sqrt(se / (sr > 0 ? sr : 1));
                CHECK(bad == 0 && rl2 < 5e-3, "7d W%d layer slot %d: %lld of %lld sampled kv outputs outside 2^-7 relative (+ 2^-12 rms), rel-L2 %.3e", rows, e, bad, cnt, rl2);
                printf("7d. W%d layer slot %d: %lld sampled kv outputs (%zu rows x %zu cols) vs the host double reference: %lld outside tolerance, rel-L2 %.3e, max |err|/(|ref|+1e-3 rms) %.3e\n",
                       rows, e, cnt, srow.size(), scol.size(), bad, rl2, maxrel);
                tc_bad += bad; tc_checked += cnt;
            }
            CHECK(engt.status() == 0, "7d: status %d", engt.status());
            // timing, a number and not a gate: layer slot 0 op (dequant + reduce no-op + act-quant + wkv + gate) in the default order and the M-tile-fastest order
            if (ds41_attn_set_tc_mfast)
            {
                cudaEvent_t t0, t1; CU(cudaEventCreate(&t0)); CU(cudaEventCreate(&t1));
                double ms[2] = {0, 0};
                for (int mode = 0; mode < 2; ++mode)
                {
                    ds41_attn_set_tc_mfast(mode);
                    std::vector<float> v;
                    for (int rep = 0; rep < 7; ++rep)
                    {
                        CU(cudaEventRecord(t0, cs));
                        if (int x = engt.op(&gn, 0, rows, Tw_d, cs)) { printf("FAIL 7d timing op rc %d\n", x); return 1; }
                        CU(cudaEventRecord(t1, cs)); CU(cudaEventSynchronize(t1));
                        float m; CU(cudaEventElapsedTime(&m, t0, t1)); if (rep) v.push_back(m);   // rep 0 warms
                    }
                    std::sort(v.begin(), v.end()); ms[mode] = v[v.size() / 2];
                }
                ds41_attn_set_tc_mfast(0);
                printf("7d. timing W%d op(layer slot 0), median of 6: default order %.3f ms, M-tile-fastest order %.3f ms (this is the whole op, not only the wkv GEMM)\n", rows, ms[0], ms[1]);
                CU(cudaEventDestroy(t0)); CU(cudaEventDestroy(t1));
            }
            else printf("7d. timing skipped: ds41_attn_set_tc_mfast is not in this build\n");
            p0 += rows;
        }
        printf("7d. tensor-core wkv seam: %lld sampled outputs, %lld outside tolerance; engine: %s\n", tc_checked, tc_bad, engt.stats_line().c_str());
    }
    // ---- 7e. the BF16 trunk (-DDS_MHC_TRUNK_BF16): k_gate(__nv_bfloat16*) against k_gate(float*) on the same BF16-valued trunk, bit for bit
    {
        constexpr int R = 96;   // rows of 4 streams x 5120; the real q*k weights of slot 0, random BF16 key/value rows
        const size_t nT = (size_t)R * kStreams * kHid;
        std::vector<float> th(nT);
        std::vector<uint16_t> kvh((size_t)R * kKvDim);
        for (auto& x : th) x = rnd_bf16_val(rng, 4.f);
        for (auto& x : kvh) x = f32_to_bf16(rnd_bf16_val(rng, 1.f));
        std::vector<uint16_t> tb(nT);
        for (size_t i = 0; i < nT; ++i) tb[i] = f32_to_bf16(th[i]);   // exact: th is BF16-valued
        float *Tf_d; uint16_t *Tb_d, *kv7_d; float* w7_d;
        CU(cudaMalloc(&Tf_d, nT * 4)); CU(cudaMalloc(&Tb_d, nT * 2)); CU(cudaMalloc(&kv7_d, kvh.size() * 2)); CU(cudaMalloc(&w7_d, wgt[0].size() * 4));
        h2d(Tf_d, th.data(), nT * 4); h2d(Tb_d, tb.data(), nT * 2); h2d(kv7_d, kvh.data(), kvh.size() * 2); h2d(w7_d, wgt[0].data(), wgt[0].size() * 4);
        k_gate<<<dim3(R, kStreams), kGateThreads, 0, cs>>>(Tf_d, (const bf16*)kv7_d, w7_d, R);
        k_gate<<<dim3(R, kStreams), kGateThreads, 0, cs>>>((bf16*)Tb_d, (const bf16*)kv7_d, w7_d, R);
        CU(cudaGetLastError()); CU(cudaStreamSynchronize(cs));
        std::vector<float> rf(nT); std::vector<uint16_t> rb(nT);
        CU(cudaMemcpy(rf.data(), Tf_d, nT * 4, cudaMemcpyDeviceToHost)); CU(cudaMemcpy(rb.data(), Tb_d, nT * 2, cudaMemcpyDeviceToHost));
        long long diff = 0, moved = 0;
        for (size_t i = 0; i < nT; ++i)
        {
            if (f32_to_bf16(rf[i]) != rb[i] || rf[i] != bf16_to_f32(rb[i])) ++diff;
            if (rb[i] != tb[i]) ++moved;
        }
        printf("7e. BF16 trunk gate: %zu elements, %lld differ from the FP32-trunk k_gate (bit compare), %lld changed by the gate\n", nT, diff, moved);
        CHECK(diff == 0, "7e: k_gate on the BF16 trunk is not bit-identical to k_gate on the FP32 trunk (%lld of %zu elements)", diff, nT);
        CHECK(moved > (long long)(nT / 2), "7e: the gate changed only %lld of %zu trunk elements; the test is not exercising k_gate", moved, nT);
        CU(cudaFree(Tf_d)); CU(cudaFree(Tb_d)); CU(cudaFree(kv7_d)); CU(cudaFree(w7_d));
    }
    printf("%s\n", eng.stats_line().c_str());
    printf("hook calls: reduce %d, project %d\n", g.reduce_calls, g.project_calls);
    if (fails) { printf("FAILED %d checks\n", fails); return 1; }
    char gate_note[256];
    if (g_gate_cancel_total == 0) snprintf(gate_note, sizeof gate_note, "gate/write within 1 bf16 ulp of the host emulation (0 exceptions)");
    else snprintf(gate_note, sizeof gate_note, "gate/write within 1 bf16 ulp of the host emulation EXCEPT %lld fp32-cancellation element(s) (|device - host| <= 2^-12 of the larger operand, printed above; not 1-ulp everywhere)", g_gate_cancel_total);
    printf("PASS engram_gpu: BF16-trunk k_gate == FP32-trunk k_gate bit for bit (7e), tensor-core wkv seam (7d), W-row chunks (window, ring, ids, ops, act-quant, status -2), ids == official vectors, NVMe rows -> embedding bit-exact, %s, repeat = 0 misses, async == stepwise, CUDA graph replays x2, fail-closed\n", gate_note);
    return 0;
}
