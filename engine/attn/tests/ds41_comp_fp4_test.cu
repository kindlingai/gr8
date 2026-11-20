// GR8_DS_COMP_FP4 exactness test: packed fp4 compressed rows (288 B) against the bf16 rows (1024 B) they replace. No fixture, random data.
//   0. the gather's dequant (cq_dq4), exhaustive over code words and scale bytes.
//   1. writers: k_comp_pool / k_comp_pool_dev (ratio-2 sources 2, 8, 14) with ckq vs ck, and k_ca_row_r1 (layer 20) ca vs its bf16 shadow:
//      every packed row, decoded on the host, is bit-equal to the bf16 row.
//   2. attention core: launch_attn on bf16 rows vs the same rows packed (p.ckq = 1), output bytes compared, for the plain, cp.async and split
//      kernels and the floor (bounded replay) form; decode (1 row, 5 sessions on slot slabs), a prefill chunk from 0 and a 4096-row chunk at a
//      long position; ratio 2 (layers 2..19) and ratio 1 (layers 20..39) key counts.
//   3. --time ITERS: kernel time of the served cp.async kernel, bf16 vs packed rows (decode at 8K and 128K keys, a 4096-row chunk).
//
//   build: nvcc -std=c++17 -O3 -gencode=arch=compute_121a,code=sm_121a -DG6_MODEL_DS41=1 -I engine -o ds41_comp_fp4_test engine/attn/tests/ds41_comp_fp4_test.cu engine/knobs-rs/target/release/libgr8knobs.a -lpthread -ldl -lm
//   run:   ds41_comp_fp4_test [--heads 16] [--time 50] [--long 131072]
//   v2 pipeline: ds41_comp_fp4_test --pipe-only [--time 3] (bounded allocations, heads=16)
// Exit code 1 if any comparison differs.
#include "../ds41_attn.cuh"
#include "../ds41_idx.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <algorithm>

using ds41a::bf16;
using ds41a::HD;
using ds41a::CQ_ROW;
using ds41a::CQ_CODES;

#define CK(x)                                                                                                       \
    do                                                                                                              \
    {                                                                                                               \
        cudaError_t e_ = (x);                                                                                       \
        if (e_ != cudaSuccess)                                                                                      \
        {                                                                                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);                \
            exit(2);                                                                                                \
        }                                                                                                           \
    } while (0)

static int g_fail = 0;
static std::mt19937_64 rng(12345);

template <class T> static T* dalloc(size_t n)
{
    T* p;
    CK(cudaMalloc(&p, n * sizeof(T) + 16));
    CK(cudaMemset(p, 0, n * sizeof(T) + 16));
    return p;
}
template <class T> static T* dup(const std::vector<T>& h)
{
    T* p = dalloc<T>(h.size());
    CK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
template <class T> static std::vector<T> back(const T* d, size_t n)
{
    std::vector<T> h(n);
    CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}
static uint16_t f2bf(float f)   // round to nearest even
{
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1)) >> 16);
}
static float bf2f(uint16_t b)
{
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
// host decoders, written from the formats (not from the device helpers)
static float h_e4m3(uint8_t b)
{
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    const float v = e == 0 ? ldexpf((float)m, -9) : ldexpf(1.0f + m / 8.0f, e - 7);
    return s ? -v : v;
}
static float h_e2m1(uint32_t c)
{
    static const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return (c & 8) ? -mag[c & 7] : mag[c & 7];
}
// packed row -> 512 bf16 bit patterns (the product is exact in bf16: checked)
static void h_decode(const uint8_t* row, uint16_t* out)
{
    for (int c = 0; c < HD; c++)
    {
        const uint32_t code = (row[c >> 1] >> ((c & 1) * 4)) & 15u;
        const float v = h_e2m1(code) * h_e4m3(row[CQ_CODES + (c >> 4)]);
        const uint16_t b = f2bf(v);
        if (bf2f(b) != v) { fprintf(stderr, "h_decode: product not exact in bf16\n"); exit(2); }
        out[c] = b;
    }
}
// fills every SM's shared memory with 0xFF (bf16 NaN) before a packed launch: without it, a slot the packed path fails to
// write still holds the key the bf16 launch just before left there (same CTAs on the same SMs), and the outputs agree
__global__ void k_smem_poison(int words)
{
    extern __shared__ uint32_t sm[];
    for (int i = threadIdx.x; i < words; i += blockDim.x) sm[i] = 0xFFFFFFFFu;
    __syncthreads();
}
static void poison_smem()
{
    static int sms = 0;
    const int bytes = (int)ds41a::AT_SMEM_PIPE;
    if (!sms)
    {
        CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
        CK(cudaFuncSetAttribute(k_smem_poison, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    }
    k_smem_poison<<<sms * 2, 256, bytes>>>(bytes / 4);
    CK(cudaGetLastError());
}
static void report(const char* what, long diff, long total, long negzero = -1)
{
    printf("  %-64s %s  (%ld of %ld differ", what, diff ? "DIFF " : "EXACT", diff, total);
    if (negzero >= 0) printf(", %ld -0 values", negzero);
    printf(")\n");
    if (diff) g_fail = 1;
}
// compare packed rows (decoded) with bf16 rows; rows = row count
static void cmp_rows(const char* what, const std::vector<uint8_t>& pk, const std::vector<uint16_t>& bf, size_t rows)
{
    long diff = 0, nz = 0;
    std::vector<uint16_t> d(HD);
    for (size_t r = 0; r < rows; r++)
    {
        h_decode(&pk[r * CQ_ROW], d.data());
        for (int c = 0; c < HD; c++)
        {
            diff += d[c] != bf[r * HD + c];
            nz += bf[r * HD + c] == 0x8000u;
        }
    }
    report(what, diff, (long)(rows * HD), nz);
}
static std::vector<float> randf(size_t n, float sd)
{
    std::normal_distribution<float> N(0.0f, sd);
    std::vector<float> v(n);
    for (auto& x : v) x = N(rng);
    return v;
}
static std::vector<uint16_t> randbf(size_t n, float sd)
{
    std::normal_distribution<float> N(0.0f, sd);
    std::vector<uint16_t> v(n);
    for (auto& x : v) x = f2bf(N(rng));
    return v;
}

// ------------------------------------------------------------------------------------------------ 1. writers
// 0. the gather's dequant (cq_dq4) against bf16(float(e2m1) * float(E4M3)) for every 16-bit code word (4 codes) and scale byte
__global__ void k_dq_all(unsigned long long* bad)
{
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;   // 0 .. 65535
    unsigned long long nb = 0;
    for (int sb = 0; sb < 256; sb++)
    {
        const float sc = ds41a::f_e4m3((uint8_t)sb);
        if (sc != sc) continue;   // NaN scales (0x7F, 0xFF): a writer never stores one
        uint32_t o[2];
        ds41a::cq_dq4(c, __float2bfloat162_rn(sc), o[0], o[1]);
        for (int e = 0; e < 4; e++)
        {
            const float r = ds41a::cq_e2m1((c >> (4 * e)) & 15) * sc;
            const uint16_t want = __bfloat16_as_ushort(__float2bfloat16_rn(r));
            nb += (uint16_t)(o[e >> 1] >> (16 * (e & 1))) != want;
        }
    }
    if (nb) atomicAdd(bad, nb);
}
static void test_dequant()
{
    printf("gather dequant\n");
    unsigned long long* d = dalloc<unsigned long long>(1);
    CK(cudaMemset(d, 0, 8));
    k_dq_all<<<256, 256>>>(d);
    CK(cudaGetLastError());
    unsigned long long h = 0;
    CK(cudaMemcpy(&h, d, 8, cudaMemcpyDeviceToHost));
    CK(cudaFree(d));
    report("cq_dq4, all 65536 code words x 254 scales == bf16(e2m1 * scale)", (long)h, 65536L * 254 * 4);
}

static void test_writers()
{
    printf("writers\n");
    const int T = 64, R = 1024;   // tokens per call, rows per slab
    const auto kv = randf((size_t)T * HD, 1.0f), sc = randf((size_t)T * HD, 2.0f);
    const auto nw = randbf(HD, 0.3f);
    std::vector<float> inv(32);
    for (int i = 0; i < 32; i++) inv[i] = powf(10000.0f, -(float)i / 32.0f);
    float *d_kv = dup(kv), *d_sc = dup(sc), *d_inv = dup(inv);
    bf16* d_nw = (bf16*)dup(nw);
    // (a) prefill from 0, T tokens -> T/2 rows; T-1 tokens (odd) parks the last token in the state
    for (int n : {T, T - 1})
    {
        bf16* ck = dalloc<bf16>((size_t)R * HD);
        uint8_t* ckq = dalloc<uint8_t>((size_t)R * CQ_ROW);
        float *kvs = dalloc<float>(2 * HD), *scs = dalloc<float>(2 * HD);
        ds41a::k_comp_pool<<<(n + 1) / 2, 512>>>(d_kv, d_sc, n, 0, d_nw, d_inv, ck, kvs, scs, nullptr, nullptr);
        ds41a::k_comp_pool<<<(n + 1) / 2, 512>>>(d_kv, d_sc, n, 0, d_nw, d_inv, nullptr, kvs, scs, nullptr, nullptr, ckq);
        CK(cudaDeviceSynchronize());
        char what[96];
        snprintf(what, sizeof what, "k_comp_pool prefill n=%d: packed == bf16", n);
        cmp_rows(what, back(ckq, (size_t)R * CQ_ROW), back((const uint16_t*)ck, (size_t)R * HD), R);
        // decode form: token at an odd position completes row pos / 2 from the parked state
        CK(cudaMemcpy(kvs, d_kv, HD * 4, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(scs, d_sc, HD * 4, cudaMemcpyDeviceToDevice));
        ds41a::k_comp_pool<<<1, 512>>>(d_kv + HD, d_sc + HD, 1, 777, d_nw, d_inv, ck, kvs, scs, nullptr, nullptr);
        ds41a::k_comp_pool<<<1, 512>>>(d_kv + HD, d_sc + HD, 1, 777, d_nw, d_inv, nullptr, kvs, scs, nullptr, nullptr, ckq);
        CK(cudaDeviceSynchronize());
        snprintf(what, sizeof what, "k_comp_pool decode pos=777 (n=%d state): packed == bf16", n);
        cmp_rows(what, back(ckq, (size_t)R * CQ_ROW), back((const uint16_t*)ck, (size_t)R * HD), R);
    }
    // (b) engine-step form, 2 sessions on slot slabs (slot 1, slot 0), odd and even starts
    {
        const int ns = 2, rps = T / ns;
        bf16* ck = dalloc<bf16>((size_t)ns * R * HD);
        uint8_t* ckq = dalloc<uint8_t>((size_t)ns * R * CQ_ROW);
        float* kvs = dup(randf((size_t)ns * 2 * HD, 1.0f));
        float* scs = dup(randf((size_t)ns * 2 * HD, 2.0f));
        std::vector<int> pos(T), sess(T);
        for (int t = 0; t < T; t++) { pos[t] = (t < rps ? 1001 : 500) + t % rps; sess[t] = t < rps ? 1 : 0; }
        int *d_pos = dup(pos), *d_sess = dup(sess);
        ds41a::k_comp_pool_dev<<<dim3((rps + 1) / 2, ns), 512>>>(d_kv, d_sc, T, d_pos, d_nw, d_inv, ck, kvs, scs, nullptr, d_sess, (size_t)R * HD);
        ds41a::k_comp_pool_dev<<<dim3((rps + 1) / 2, ns), 512>>>(d_kv, d_sc, T, d_pos, d_nw, d_inv, nullptr, kvs, scs, nullptr, d_sess,
                                                                 (size_t)R * CQ_ROW, ckq);
        CK(cudaDeviceSynchronize());
        cmp_rows("k_comp_pool_dev 2 sessions on slabs: packed == bf16", back(ckq, (size_t)ns * R * CQ_ROW),
                 back((const uint16_t*)ck, (size_t)ns * R * HD), (size_t)ns * R);
    }
    // (c) layer 20: k_ca_row_r1 writes the ca row and its bf16 shadow; 2 slots, rows by position
    {
        const int rpp = 512, slots = 2;
        uint8_t* ca = dalloc<uint8_t>((size_t)slots * rpp * CQ_ROW);
        bf16* shadow = dalloc<bf16>((size_t)slots * rpp * HD);
        const auto y = randbf((size_t)T * HD, 1.0f);
        const auto nw20 = randbf(HD, 0.3f);
        std::vector<int> pos(T), pages(T);
        for (int t = 0; t < T; t++) { pos[t] = t < T / 2 ? 100 + t : 300 + t; pages[t] = t < T / 2 ? 1 : 0; }
        int *d_pos = dup(pos), *d_pages = dup(pages);
        for (int half = 0; half < 2; half++)   // a call's page table is per session: one launch per half
        {
            ds41i::CaR1P p{};
            p.y = (const bf16*)dup(y) + (size_t)half * (T / 2) * HD;
            p.norm_w = (const bf16*)dup(nw20);
            p.inv = ds41i::rope_inv_ds41();
            p.pos0 = 0;
            p.ca = ds41i::PagedRows{ca, d_pages + half * (T / 2), rpp, CQ_ROW, (size_t)rpp * CQ_ROW};
            p.dpos = d_pos + half * (T / 2);
            p.shadow = shadow;
            ds41i::k_ca_row_r1<<<T / 2, 512>>>(p);
        }
        CK(cudaDeviceSynchronize());
        cmp_rows("k_ca_row_r1 (layer 20) 2 slots: ca == shadow", back(ca, (size_t)slots * rpp * CQ_ROW),
                 back((const uint16_t*)shadow, (size_t)slots * rpp * HD), (size_t)slots * rpp);
    }
}

// ------------------------------------------------------------------------------------------------ 2. attention core
struct Keys
{
    size_t rows = 0, slots = 1;
    bf16* bf = nullptr;      // [slots][rows][512]
    uint8_t* pk = nullptr;   // [slots][rows][288]
};
// random packed rows (every code, scale bytes 0x01 .. 0x47 incl. subnormals, some all-zero groups) and their bf16 values
static Keys make_keys(size_t rows, size_t slots)
{
    Keys k;
    k.rows = rows; k.slots = slots;
    const size_t n = rows * slots;
    std::vector<uint8_t> pk(n * CQ_ROW);
    std::uniform_int_distribution<int> byte(0, 255), scl(1, 0x47), z(0, 99);
    for (size_t r = 0; r < n; r++)
    {
        uint8_t* row = &pk[r * CQ_ROW];
        for (int i = 0; i < CQ_CODES; i++) row[i] = (uint8_t)byte(rng);
        for (int g = 0; g < 32; g++)
        {
            row[CQ_CODES + g] = (uint8_t)scl(rng);
            if (z(rng) == 0) for (int i = 0; i < 8; i++) row[g * 8 + i] = (i & 1) ? 0x88 : 0x00;   // +0 / -0 group
        }
    }
    std::vector<uint16_t> bf(n * HD);
    for (size_t r = 0; r < n; r++) h_decode(&pk[r * CQ_ROW], &bf[r * HD]);
    k.pk = dup(pk);
    k.bf = (bf16*)dup(bf);
    return k;
}
static void free_keys(Keys& k)
{
    CK(cudaFree(k.bf));
    CK(cudaFree(k.pk));
}

struct Case
{
    const char* name;
    int ratio;      // 2: layers 2..19, 1: layers 20..39
    int n;          // rows
    int ns;         // sessions (decode form with dsess when > 1)
    int pos;        // first position (decode form); -1 = prefill from 0
    int floor;      // window floor (bounded replay), 0 = none
    int wl = 0;     // window list length override (decode form only; 0 = the engine's: WIN, or min(n, WIN) for prefill)
};

// top-k list of a row at position p: up to 512 random visible rows ((p + 1) / ratio), ascending, -1 tail
static void fill_topk(int32_t* out, int p, int ratio, size_t rows)
{
    const int vis = std::min<long>((p + 1) / ratio, (long)rows);
    int m = 0;
    if (vis <= 512) for (; m < vis; m++) out[m] = m;
    else
    {
        std::uniform_int_distribution<int> U(0, vis - 1);
        for (; m < 512; m++) out[m] = U(rng);
        std::sort(out, out + 512);
    }
    for (; m < 512; m++) out[m] = -1;
}

struct Built
{
    ds41a::AttnP p{};
    int *d_pos = nullptr, *d_sess = nullptr, *d_topk = nullptr;
    bf16 *q = nullptr, *kvw = nullptr, *out_a = nullptr, *out_b = nullptr;
    float *sink = nullptr, *split = nullptr;
    size_t out_n = 0;
};
static Built build(const Case& c, int heads, const Keys& K, bool split = true)
{
    Built b;
    ds41a::AttnP& p = b.p;
    const int n = c.n;
    const bool decode = c.pos >= 0;
    b.q = (bf16*)dup(randbf((size_t)n * heads * HD, 0.5f));
    b.sink = dup(randf(heads, 1.0f));
    b.out_n = (size_t)n * heads * HD;
    b.out_a = dalloc<bf16>(b.out_n);
    b.out_b = dalloc<bf16>(b.out_n);
    if (split) b.split = dalloc<float>((size_t)n * heads * 2 * (HD + 2));
    std::vector<int32_t> tk((size_t)n * 512);
    std::vector<int> pos(n), sess(n);
    const int rps = n / c.ns;
    for (int t = 0; t < n; t++)
    {
        const int j = t / rps;
        pos[t] = decode ? c.pos + j * 37 + t % rps : t;   // sessions at nearby positions
        sess[t] = c.ns - 1 - j;                          // slot order reversed: slab strides matter
        fill_topk(&tk[(size_t)t * 512], pos[t], c.ratio, K.rows);
    }
    b.d_topk = dup(tk);
    const int ring_rows = decode ? ds41a::WIN + rps - 1 : 0;
    if (decode)
    {
        b.kvw = (bf16*)dup(randbf((size_t)c.ns * ring_rows * HD, 0.3f));
        b.d_pos = dup(pos);
        if (c.ns > 1) b.d_sess = dup(sess);
    }
    else b.kvw = (bf16*)dup(randbf((size_t)n * HD, 0.3f));
    p.q = b.q; p.kvw = b.kvw; p.topk = b.d_topk; p.sink = b.sink;
    p.ldtk = 512; p.n = n; p.heads = heads; p.start = 0;
    p.dstart = decode ? b.d_pos : nullptr;
    p.decode = decode;
    p.WL = decode ? ds41a::WIN : std::min(n, ds41a::WIN);
    if (c.wl) p.WL = c.wl;
    p.ring_rows = ring_rows;
    p.TK = 512;
    p.scale = 0.04419417382415922f;   // 512^-0.5
    p.floor = c.floor;
    p.dsess = c.ns > 1 ? b.d_sess : nullptr;
    p.split = b.split;
    return b;
}
static void set_rows(ds41a::AttnP& p, const Keys& K, bool packed, bf16* out)
{
    p.ckq = packed;
    p.ck = packed ? (const bf16*)K.pk : K.bf;
    p.ck_ss = packed ? K.rows * CQ_ROW : K.rows * HD;
    p.out = out;
}
static void free_built(Built& b)
{
    for (void* x : {(void*)b.d_pos, (void*)b.d_sess, (void*)b.d_topk, (void*)b.q, (void*)b.kvw, (void*)b.out_a, (void*)b.out_b, (void*)b.sink,
                    (void*)b.split})
        if (x) CK(cudaFree(x));
}
static void set_kernel(int kind)   // 0 = k_attn, 1 = k_attn_cpa (served), 2 = split
{
    ds41a::g_attn_cpa = kind >= 1;
    ds41a::g_attn_split = kind == 2;
}
static const char* kname[3] = {"k_attn", "k_attn_cpa", "k_attn_cpa_split"};

static void test_attention(int heads, int long_keys)
{
    printf("attention core (heads %d)\n", heads);
    const std::vector<Case> cases = {
        {"decode 1 row, 8K keys", 2, 1, 1, 8191, 0},
        {"decode 1 row, 8K keys", 1, 1, 1, 8191, 0},
        {"decode 1 row, long", 2, 1, 1, long_keys - 1, 0},
        {"decode 1 row, long", 1, 1, 1, long_keys - 1, 0},
        {"decode 5 sessions on slabs, 64K keys", 2, 5, 5, 65000, 0},
        {"decode 5 sessions on slabs, 64K keys", 1, 5, 5, 65000, 0},
        {"decode 1 row, floor (bounded replay)", 2, 1, 1, 20000, 19950},
        {"decode 1 row, floor (bounded replay)", 1, 1, 1, 20000, 19950},
        {"prefill chunk from 0, 1024 rows", 2, 1024, 1, -1, 0},
        {"prefill chunk from 0, 1024 rows", 1, 1024, 1, -1, 0},
        {"4096-row chunk at a long position", 2, 4096, 1, long_keys - 4096, 0},
        {"4096-row chunk at a long position", 1, 4096, 1, long_keys - 4096, 0},
        // WL not a multiple of 64: one 64-row block holds window rows and packed compressed rows
        {"prefill chunk from 0, 37 rows (WL 37)", 2, 37, 1, -1, 0},
        {"prefill chunk from 0, 37 rows (WL 37)", 1, 37, 1, -1, 0},
        {"prefill chunk from 0, 100 rows (WL 100)", 2, 100, 1, -1, 0},
        {"prefill chunk from 0, 100 rows (WL 100)", 1, 100, 1, -1, 0},
        {"decode 1 row, 8K keys, WL 37 (override)", 2, 1, 1, 8191, 0, 37},
        {"decode 1 row, 8K keys, WL 37 (override)", 1, 1, 1, 8191, 0, 37},
        {"decode 1 row, 8K keys, WL 100 (override)", 2, 1, 1, 8191, 0, 100},
        {"decode 1 row, 8K keys, WL 100 (override)", 1, 1, 1, 8191, 0, 100},
        {"decode 5 sessions on slabs, WL 100 (override)", 2, 5, 5, 65000, 0, 100},
        {"decode 5 sessions on slabs, WL 100 (override)", 1, 5, 5, 65000, 0, 100},
    };
    for (const Case& c : cases)
    {
        const int maxpos = c.pos >= 0 ? c.pos + 37 * (c.ns - 1) + c.n / c.ns : c.n;
        Keys K = make_keys((size_t)(maxpos / c.ratio + 2), (size_t)c.ns);
        Built b = build(c, heads, K);
        for (int kind = 0; kind < 3; kind++)
        {
            if (kind == 2 && (c.floor || !ds41a::split_eligible(b.p.n, b.p.WL, b.p.TK))) continue;
            set_kernel(kind);
            set_rows(b.p, K, false, b.out_a);
            ds41a::launch_attn(b.p, 0);
            set_rows(b.p, K, true, b.out_b);
            poison_smem();
            ds41a::launch_attn(b.p, 0);
            CK(cudaDeviceSynchronize());
            const auto oa = back((const uint16_t*)b.out_a, b.out_n), ob = back((const uint16_t*)b.out_b, b.out_n);
            long diff = 0;
            for (size_t i = 0; i < b.out_n; i++) diff += oa[i] != ob[i];
            long bad = 0, nonzero = 0;   // the output is finite and not trivially zero
            for (size_t i = 0; i < b.out_n; i++)
            {
                bad += ((oa[i] >> 7) & 0xFF) == 0xFF;
                nonzero += (oa[i] & 0x7FFF) != 0;
            }
            char what[160];
            snprintf(what, sizeof what, "%s ratio %d %s: packed == bf16", c.name, c.ratio, kname[kind]);
            report(what, diff, (long)b.out_n);
            if (bad || nonzero < (long)b.out_n / 2)
            {
                printf("  ^ output check: %ld inf/nan, %ld of %zu nonzero\n", bad, nonzero, b.out_n);
                g_fail = 1;
            }
        }
        if (&c == &cases[0])   // negative control: one scale byte of the first row's first listed key changed -> the packed output must differ
        {
            int32_t j0;
            CK(cudaMemcpy(&j0, b.d_topk, 4, cudaMemcpyDeviceToHost));
            uint8_t sb;
            CK(cudaMemcpy(&sb, K.pk + (size_t)j0 * CQ_ROW + CQ_CODES, 1, cudaMemcpyDeviceToHost));
            sb ^= 0x08;
            CK(cudaMemcpy(K.pk + (size_t)j0 * CQ_ROW + CQ_CODES, &sb, 1, cudaMemcpyHostToDevice));
            set_kernel(1);
            set_rows(b.p, K, false, b.out_a);
            ds41a::launch_attn(b.p, 0);
            set_rows(b.p, K, true, b.out_b);
            poison_smem();
            ds41a::launch_attn(b.p, 0);
            CK(cudaDeviceSynchronize());
            const auto oa = back((const uint16_t*)b.out_a, b.out_n), ob = back((const uint16_t*)b.out_b, b.out_n);
            long diff = 0;
            for (size_t i = 0; i < b.out_n; i++) diff += oa[i] != ob[i];
            printf("  %-64s %s  (%ld of %zu differ)\n", "negative control: one packed scale byte changed (must differ)", diff ? "DIFF ok" : "NO DIFF", diff,
                   b.out_n);
            if (!diff) g_fail = 1;
        }
        free_built(b);
        free_keys(K);
    }
}

// ------------------------------------------------------------------------------------------------ 3. timing (served kernel: cp.async)
static void time_attention(int heads, int iters)
{
    printf("kernel time, k_attn_cpa (heads %d, %d iterations, median of 5 runs of the mean)\n", heads, iters);
    const std::vector<Case> cases = {
        {"decode 1 row, 8K keys", 2, 1, 1, 8191, 0},
        {"decode 1 row, 8K keys", 1, 1, 1, 8191, 0},
        {"decode 1 row, 128K keys", 2, 1, 1, 131071, 0},
        {"decode 1 row, 128K keys", 1, 1, 1, 131071, 0},
        {"decode 5 sessions, 128K keys", 2, 5, 5, 131000, 0},
        {"decode 5 sessions, 128K keys", 1, 5, 5, 131000, 0},
        {"4096-row chunk at 128K", 2, 4096, 1, 131072 - 4096, 0},
        {"4096-row chunk at 128K", 1, 4096, 1, 131072 - 4096, 0},
    };
    set_kernel(1);
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    for (const Case& c : cases)
    {
        const int maxpos = c.pos + 37 * (c.ns - 1) + c.n / c.ns;
        Keys K = make_keys((size_t)(maxpos / c.ratio + 2), (size_t)c.ns);
        Built b = build(c, heads, K);
        float us[2];
        for (int packed = 0; packed < 2; packed++)
        {
            set_rows(b.p, K, packed, packed ? b.out_b : b.out_a);
            for (int i = 0; i < 3; i++) ds41a::launch_attn(b.p, 0);
            std::vector<float> runs;
            for (int r = 0; r < 5; r++)
            {
                CK(cudaEventRecord(e0));
                for (int i = 0; i < iters; i++) ds41a::launch_attn(b.p, 0);
                CK(cudaEventRecord(e1));
                CK(cudaEventSynchronize(e1));
                float ms;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                runs.push_back(ms * 1000.0f / iters);
            }
            std::sort(runs.begin(), runs.end());
            us[packed] = runs[2];
        }
        printf("  %-34s ratio %d: bf16 %9.2f us   packed %9.2f us   (%+.1f%%)\n", c.name, c.ratio, us[0], us[1], 100.0f * (us[1] / us[0] - 1.0f));
        free_built(b);
        free_keys(K);
    }
}


// Pipeline operand proof: same packed bytes -> original cq_store bf16 tile ->
// the new packed-stage pipe_store, comparing every expanded bf16 byte.
__global__ void k_pipe_operands(const uint8_t* keys, unsigned* bad)
{
    extern __shared__ __align__(16) uint8_t operand_sm[];
    bf16* wide = (bf16*)operand_sm;
    uint8_t* tile = operand_sm + 64 * ds41a::AT_KLD * sizeof(bf16);
    __shared__ const bf16* rows[64];
    const int tid = threadIdx.x, w = tid / 32, lane = tid % 32;
    if (tid < 64) rows[tid] = (const bf16*)(keys + tid * CQ_ROW + 1);
    for (int i = tid; i < 64 * CQ_ROW; i += 256) tile[i] = keys[i];
    __syncthreads();
    uint2 cd[8]; uint32_t sb[8];
    ds41a::cq_load(rows, w, lane, cd, sb);
    ds41a::cq_store(wide, ds41a::AT_KLD, rows, w, lane, cd, sb);
    __syncthreads();
    uint32_t reference[64];
#pragma unroll
    for (int k = 0; k < 64; k++) {
        const int i = tid + k * 256;
        reference[k] = *(const uint32_t*)(wide + (i / (HD / 2)) * ds41a::AT_KLD + (i % (HD / 2)) * 2);
    }
    __syncthreads();
    // A missing store must not pass by retaining cq_store's expected bytes.
    for (int i = tid; i < 64 * ds41a::AT_KLD; i += 256) wide[i] = __ushort_as_bfloat16(0xFFFFu);
    __syncthreads();
    ds41a::pipe_store(wide, tile, w, lane);
    __syncthreads();
    unsigned n = 0;
#pragma unroll
    for (int k = 0; k < 64; k++) {
        const int i = tid + k * 256;
        n += reference[k] != *(const uint32_t*)(wide + (i / (HD / 2)) * ds41a::AT_KLD + (i % (HD / 2)) * 2);
    }
    if (n) atomicAdd(bad, n);
}
static void test_pipe_operands()
{
    // Every code pair and every finite scale, including signed zero and negative
    // scales. 64 rows * 32 groups * 32 batches covers all 256 x 254 pairs.
    std::vector<uint8_t> bytes(64 * CQ_ROW);
    auto* pk = dalloc<uint8_t>(bytes.size());
    auto* bad = dalloc<unsigned>(1);
    const int smem = 64 * ds41a::AT_KLD * sizeof(bf16) + 64 * CQ_ROW;
    CK(cudaFuncSetAttribute(k_pipe_operands, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    for (int batch = 0; batch < 32; batch++) {
        for (int row = 0; row < 64; row++) for (int g = 0; g < 32; g++) {
            const int pair = row * 32 + g + batch * 2048;
            int scale = (pair / 256) % 254;
            if (scale >= 127) scale++;
            bytes[row * CQ_ROW + CQ_CODES + g] = (uint8_t)scale;
            for (int b = 0; b < 8; b++) bytes[row * CQ_ROW + g * 8 + b] = uint8_t(pair % 256);
        }
        CK(cudaMemcpy(pk, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
        k_pipe_operands<<<1, 256, smem>>>(pk, bad);
    }
    CK(cudaDeviceSynchronize());
    const unsigned errors = back(bad, 1)[0];
    printf("pipe operands: cq_store vs packed-stage pipe_store bf16 bits: %u mismatches\n", errors);
    if (errors) g_fail = 1;
    CK(cudaFree(pk)); CK(cudaFree(bad));
}
static void pipe_launch(ds41a::AttnP p, bool pipe)
{
    const dim3 grid(p.n, p.heads / 16);
    if (pipe) ds41a::k_attn_cpa_pipe_q<<<grid, ds41a::AT_THREADS, ds41a::AT_SMEM_PIPE>>>(p);
    else ds41a::k_attn_cpa_t<false, true><<<grid, ds41a::AT_THREADS, ds41a::AT_SMEM_CPA>>>(p);
    CK(cudaGetLastError());
}
static void test_pipe(int heads, int iters)
{
    CK(cudaFuncSetAttribute(ds41a::k_attn_cpa_pipe_q, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ds41a::AT_SMEM_PIPE));
    CK(cudaFuncSetAttribute(ds41a::k_attn_cpa_t<false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ds41a::AT_SMEM_CPA));
    cudaFuncAttributes a{}, b{};
    CK(cudaFuncGetAttributes(&a, ds41a::k_attn_cpa_t<false, true>));
    CK(cudaFuncGetAttributes(&b, ds41a::k_attn_cpa_pipe_q));
    printf("pipe resource: dynamic_smem=%zu baseline_regs=%d pipe_regs=%d baseline_local=%zu pipe_local=%zu\n", ds41a::AT_SMEM_PIPE, a.numRegs, b.numRegs, a.localSizeBytes, b.localSizeBytes);
    test_pipe_operands();
    for (int n : {1, 7, 64, 2047, 2048, 8192}) for (int mode = 0; mode < 3; mode++) {
        Case c{"pipe", 2, n, 1, mode ? 16384 : -1, 0};
        if (mode == 2) c.wl = 100; // mixed block, then packed-stage reuse
        Keys K = make_keys(16384, 1);
        Built x = build(c, heads, K, false); // no unused split scratch: < 450 MiB at 8192x16
        if (mode == 2) {
            auto tk = back(x.d_topk, (size_t)n * 512);
            const int lengths[] = {0, 1, 63, 64, 65, 129, 511, 512};
            for (int t = 0; t < n; t++) {
                const int len = lengths[t % 8];
                for (int j = len; j < 512; j++) tk[(size_t)t * 512 + j] = -1;
                // Include an empty interior block followed by live blocks.
                if (t % 8 == 7) for (int j = 64; j < 192; j++) tk[(size_t)t * 512 + j] = -1;
            }
            CK(cudaMemcpy(x.d_topk, tk.data(), tk.size() * 4, cudaMemcpyHostToDevice));
        }
        set_rows(x.p, K, true, x.out_a); pipe_launch(x.p, false);
        set_rows(x.p, K, true, x.out_b); poison_smem(); pipe_launch(x.p, true);
        CK(cudaDeviceSynchronize());
        const auto ref = back((const uint16_t*)x.out_a, x.out_n), got = back((const uint16_t*)x.out_b, x.out_n);
        const bool equal = memcmp(ref.data(), got.data(), x.out_n * 2) == 0;
        printf("pipe memcmp rows=%d mode=%d WL=%d bytes=%zu %s\n", n, mode, x.p.WL, x.out_n * 2, equal ? "BYTE-EQUAL" : "DIFF");
        if (!equal) { g_fail = 1; for (size_t i = 0; i < x.out_n; i++) if (ref[i] != got[i]) { printf(" first diff %zu: %04x != %04x\n", i, ref[i], got[i]); break; } }
        // Compare through the actual launch selector too (and guard split/floor exclusions separately in the original suite).
        ds41a::g_attn_cpa = 1; ds41a::g_attn_split = 0; ds41a::g_attn_pipe = true;
        ds41a::launch_attn(x.p, 0); ds41a::g_attn_pipe = false;
        CK(cudaDeviceSynchronize());
        const auto selected = back((const uint16_t*)x.out_b, x.out_n);
        const bool selected_equal = memcmp(ref.data(), selected.data(), x.out_n * 2) == 0;
        printf("pipe selector rows=%d mode=%d %s\n", n, mode, selected_equal ? "BYTE-EQUAL" : "DIFF");
        if (!selected_equal) g_fail = 1;
        if (iters > 0 && mode == 1) {
            float us[2];
            cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
            for (int arm = 0; arm < 2; arm++) {
                std::vector<float> runs;
                pipe_launch(x.p, arm);
                for (int r = 0; r < 3; r++) {
                    CK(cudaEventRecord(e0));
                    for (int i = 0; i < iters; i++) pipe_launch(x.p, arm);
                    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
                    float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); runs.push_back(ms * 1000 / iters);
                }
                std::sort(runs.begin(), runs.end()); us[arm] = runs[1];
            }
            printf("pipe time rows=%d baseline_us=%.3f pipe_us=%.3f (median 3 x %d iterations)\n", n, us[0], us[1], iters);
            CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
        }
        free_built(x); free_keys(K);
    }
}

int main(int argc, char** argv)
{
    int heads = 16, iters = 0, long_keys = 131072;
    bool pipe_only = false;
    for (int i = 1; i < argc; i++)
    {
        const std::string a = argv[i];
        if (a == "--pipe-only") pipe_only = true;
        else if (a == "--heads" && i + 1 < argc) heads = atoi(argv[++i]);
        else if (a == "--time" && i + 1 < argc) iters = atoi(argv[++i]);
        else if (a == "--long" && i + 1 < argc) long_keys = atoi(argv[++i]);
        else { fprintf(stderr, "usage: %s [--heads 16] [--time ITERS] [--long KEYS] [--pipe-only]\n", argv[0]); return 2; }
    }
    if (heads % 16) { fprintf(stderr, "--heads must be a multiple of 16\n"); return 2; }
    if (pipe_only) {
        if (heads != 16) { fprintf(stderr, "--pipe-only uses heads=16 for bounded memory\n"); return 2; }
        size_t free, total; CK(cudaMemGetInfo(&free, &total));
        printf("GPU free_bytes=%zu total_bytes=%zu; test allocations below 450 MiB\n", free, total);
        if (free < (size_t)1536 * 1024 * 1024) { fprintf(stderr, "SKIP: insufficient free memory\n"); return 77; }
        test_pipe(heads, iters);
        printf("%s\n", g_fail ? "FAIL" : "PASS"); return g_fail;
    }
    test_dequant();
    test_writers();
    test_attention(heads, long_keys);
    if (iters > 0) time_attention(heads, iters);
    printf("%s\n", g_fail ? "FAIL" : "PASS");
    return g_fail;
}
