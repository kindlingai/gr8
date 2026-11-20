// DS41 MoE at 33..48 rows a step (G6_MX 40 / 48): one GB10, no fabric, no pack, synthetic weights.
// The ops under test are the engine's adapters, called as ds41_step.cuh calls them: ds41moe::bind_moe (through a synthetic
// Model::ds41_find), prepare_moe, then op_router -> op_shared -> op_experts on Model::hc_y -> Model::sc.part[1]. At 33..48 rows
// they take router_pw / shared_pw (or shared_pw_1pass) / experts_pw; at <= 32 rows the decode paths (router / router_mt ...).
// Claim: each row of a 33..48-row call has the same bits as that row run through the <= 32-row path, with the rows split
// into chunks of 32 (32 + rest) and into chunks of 8. Weights and inputs are hash-filled records of the pack formats, so the
// claim is bit equality between the paths, not agreement with a model.
//   main layer  384 experts, top 6; rows 33 40 41 47 48; decode step (Model::ds_moe_decode_step = true: the top-k cut
//               GR8_DS_MOE_TOPK_DECODE, 4 here) and non-decode (cut GR8_DS_MOE_TOPK, 6)
//   draft layer 128 experts, top 3 (geom::DS_NEXP / DS_TOPK); rows 35 40
//   G           the 33..48-row call captured in a CUDA graph (as a decode step is) and replayed: same bits as the eager call
//   also        rows < M written and finite (outputs are poisoned 0xFF before every call), rows >= M not written
// Controls (each must FAIL the row compare in the expected rows, or the compare could not fail):
//   CTRL-IN     one input row (37 of 48) changed: that row differs from the reference, the other 47 stay identical
//   CTRL-DROP   the 40-row call run as a 32-row call (a dispatch that stops at 32 rows, as nt >= 4 -> the NT = 4 body did in
//               g6mx.h before G6_MX 48): the checker flags rows 32..39 and only them
//   CTRL-CUT    the 48-row call with the decode flag off (cut 6) against the decode reference (cut 4): rows differ, so
//               the decode cut reaches the 33..48-row path
// Trellis layers (a layer with L<l>.tr3.table) are not built: the finder has no tr3.* tensor, as the launcher's default
// GR8_DS_MANIFEST (unset = native experts) serves. bind_moe's own lookup decides it; the test prints "trellis 0".
//   ds41_moe_pw_test          the three arms below, each in its own process (the knobs are read once per process):
//                             GR8_DS_MOE_SH_1PASS=0 PDL=1, SH_1PASS=1 PDL=1, SH_1PASS=0 PDL=0 (GR8_DS_MOE_TOPK_DECODE=4)
//   ds41_moe_pw_test arm      one arm with the environment as it is
// Device memory: about 1.9 GB (the 384 experts' fc1 + fc2 records are 1.80 GB; the draft layer reuses the first 128); the
// peak is printed. Exit 0 = last line "DS41_MOE_PW_ALL_PASS"; 1 any FAIL; 2 CUDA / setup error.
// Build (needs a G6_MX 40 or 48 build; from the repo root, the DS41 engine flags of scripts/build.sh MODEL=ds41 DS41_MX=48):
//   nvcc -O2 -std=c++17 -gencode=arch=compute_121a,code=sm_121a $K1H2_FLAGS(-DPF_FC2GL=4) -DG6_MODEL_DS41=1 -DG6_NR=4
//        -DDEC_S2 -DG6_MX=48 -I engine/runtime -o ds41_moe_pw_test engine/runtime/tests/ds41_moe_pw_test.cu
//        engine/knobs-rs/target/release/libgr8knobs.a -lcublas -lpthread -ldl -lm
#include <cassert>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <algorithm>
#include <array>
#include <ctime>
#include <type_traits>
#include <sys/wait.h>
#include <unistd.h>
#include "../../knobs-rs/gr8_knobs.h"
#include "../rank.cuh"

using namespace g6rt;
static_assert(MX > 32, "ds41_moe_pw_test needs a G6_MX 40 or 48 build (-DG6_MX=48)");

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

constexpr int H = dsr::HIDDEN, IW = ds41moe::IW, NE = dsr::NEXP, NED = geom::DS_NEXP, RMAX = 48;
constexpr int L_MAIN = 10, L_DRAFT = 43;                                       // checkpoint layer numbers of the two synthetic layers
static_assert(dsr::TOPK == 6 && geom::DS_TOPK == 3 && NED == 128 && IW == 576 && H == 5120, "the shapes this test names");
constexpr size_t FC1_B = (size_t)(IW / 16) * (H / 64) * 1088, FC2_B = (size_t)(H / 16) * (IW / 64) * 544;   // one expert's records
constexpr size_t SH13_B = (size_t)(IW / 16) * (H / 32) * 528, SH2_B = (size_t)(H / 16) * (IW / 32) * 528;
constexpr int CTRL_ROW = 37;

static int g_pass = 0, g_fail = 0;
static void report(bool ok, const char* fmt, ...)
{
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    printf("%s %s\n", ok ? "PASS" : "FAIL", buf);
    fflush(stdout);
    (ok ? g_pass : g_fail)++;
}

__device__ __forceinline__ uint32_t hash32(uint32_t i, uint32_t seed)
{
    uint32_t h = i * 2654435761u ^ seed;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
    return h;
}
__global__ void k_fill_mx4(uint8_t* p, size_t n, uint32_t seed)       // mx4g records (GR 544): 512 random fp4 bytes + 32 scales 2^-10..2^-3
{
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += 256ull * gridDim.x)
    {
        const uint32_t h = hash32(static_cast<uint32_t>(i) ^ static_cast<uint32_t>(i >> 32) * 0x9e3779b1u, seed);
        p[i] = static_cast<int>(i % 544) < 512 ? static_cast<uint8_t>(h) : static_cast<uint8_t>(117 + (h >> 24 & 7));
    }
}
__global__ void k_fill_mx8(uint8_t* p, size_t n, uint32_t seed)       // MXFP8 gv records (GR 528): 512 E4M3 codes |x| < 2 (no NaN) + 16 scales
{
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += 256ull * gridDim.x)
    {
        const uint32_t h = hash32(static_cast<uint32_t>(i) ^ static_cast<uint32_t>(i >> 32) * 0x9e3779b1u, seed);
        p[i] = static_cast<int>(i % 528) < 512 ? static_cast<uint8_t>((h & 0x80) | (h & 0x3f)) : static_cast<uint8_t>(117 + (h >> 24 & 7));
    }
}
__global__ void k_fill_bf16(__nv_bfloat16* p, size_t n, uint32_t seed, float amp)   // finite bf16 in [-amp, amp)
{
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += 256ull * gridDim.x)
        p[i] = __float2bfloat16_rn((static_cast<float>(hash32(static_cast<uint32_t>(i), seed) & 0xffffu) / 65535.f - 0.5f) * 2.f * amp);
}
__global__ void k_fill_f32(float* p, size_t n, uint32_t seed, float amp)
{
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += 256ull * gridDim.x)
        p[i] = (static_cast<float>(hash32(static_cast<uint32_t>(i), seed) & 0xffffffu) / 16777215.f - 0.5f) * 2.f * amp;
}

static size_t g_free0 = 0, g_free_min = ~(size_t)0;
static void mem_mark()
{
    size_t f = 0, t = 0;
    CK(cudaMemGetInfo(&f, &t));
    g_free_min = std::min(g_free_min, f);
}

// one call of the three MoE ops, as the step runs them, on rows x[0..rows) -> out[0..rows)
static void call(Model& m, int li, const __nv_bfloat16* x, int rows, uint16_t* out, cudaStream_t st)
{
    m.hc_y = const_cast<__nv_bfloat16*>(x);
    m.sc.part[1] = out;
    int r = ds41moe::op_router(&m, li, rows, st);
    if (!r) r = ds41moe::op_shared(&m, li, rows, st);
    if (!r) r = ds41moe::op_experts(&m, li, rows, st);
    if (r) { printf("op error %d (layer %d, rows %d)\n", r, li, rows); exit(2); }
}
// the rows in chunks of `chunk` (<= 32: the decode paths)
static void call_chunked(Model& m, int li, const __nv_bfloat16* x, int rows, int chunk, uint16_t* out, cudaStream_t st)
{
    for (int r0 = 0; r0 < rows; r0 += chunk) call(m, li, x + (size_t)r0 * H, std::min(chunk, rows - r0), out + (size_t)r0 * H, st);
}
static void poison(uint16_t* out, cudaStream_t st) { CK(cudaMemsetAsync(out, 0xff, (size_t)RMAX * H * 2, st)); }
static std::vector<uint16_t> fetch(const uint16_t* d, cudaStream_t st)
{
    std::vector<uint16_t> h((size_t)RMAX * H);
    CK(cudaStreamSynchronize(st));
    CK(cudaMemcpy(h.data(), d, h.size() * 2, cudaMemcpyDeviceToHost));
    mem_mark();
    return h;
}
// rows (< n) whose bits differ between a and b
static std::vector<int> diff_rows(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, int n)
{
    std::vector<int> d;
    for (int r = 0; r < n; ++r) if (memcmp(&a[(size_t)r * H], &b[(size_t)r * H], (size_t)H * 2)) d.push_back(r);
    return d;
}
static std::string rows_s(const std::vector<int>& v)
{
    std::string s;
    for (size_t i = 0; i < v.size() && i < 12; ++i) s += (i ? "," : "") + std::to_string(v[i]);
    if (v.size() > 12) s += ",...";
    return s.empty() ? "-" : s;
}
// rows < n: no poison word, finite, not all zero; rows n..RMAX-1: all poison
static bool written_ok(const std::vector<uint16_t>& a, int n, std::string& why)
{
    size_t nonzero = 0;
    for (int r = 0; r < RMAX; ++r)
        for (int c = 0; c < H; ++c)
        {
            const uint16_t v = a[(size_t)r * H + c];
            if (r >= n) { if (v != 0xffff) { why = "row " + std::to_string(r) + " >= M written"; return false; } continue; }
            if ((v & 0x7f80) == 0x7f80) { why = "row " + std::to_string(r) + " not finite / not written"; return false; }
            nonzero += (v & 0x7fff) != 0;
        }
    if (nonzero < (size_t)n * H / 2) { why = "mostly zero"; return false; }
    return true;
}

struct Case { int li; const char* name; int rows; bool dec; };

static int run_arm()
{
    const bool sh1 = ds41moe::sh_1pass_on(), pdl = ds41moe::pdl_on();
    const int kc_main = ds41moe::topk_cut(), kc_dec = ds41moe::topk_decode_cut();
    printf("# arm: GR8_DS_MOE_SH_1PASS=%d GR8_DS_MOE_PDL=%d cut %d decode cut %d G6_MX=%d MTMAX=%d RO_FROM=%d SH1_MAX=%d\n", sh1, pdl, kc_main, kc_dec,
           MX, ds41moe::MTMAX, ds41moe::RO_FROM, ds41moe::SH1_MAX);
    {
        size_t t = 0;
        CK(cudaMemGetInfo(&g_free0, &t));
        g_free_min = g_free0;
    }
    // weights: the main layer's 384 experts; the draft layer's finder hands out the first 128 of the same records
    uint8_t *fc1, *fc2, *sh1w, *sh3w, *sh2w, *dsh1, *dsh3, *dsh2;
    __nv_bfloat16 *gate, *dgate, *x, *xc;
    float *bias, *dbias;
    uint16_t *o_big, *o_ref, *o_g;
    CK(cudaMalloc(&fc1, FC1_B * NE)); CK(cudaMalloc(&fc2, FC2_B * NE));
    CK(cudaMalloc(&sh1w, SH13_B)); CK(cudaMalloc(&sh3w, SH13_B)); CK(cudaMalloc(&sh2w, SH2_B));
    CK(cudaMalloc(&dsh1, SH13_B)); CK(cudaMalloc(&dsh3, SH13_B)); CK(cudaMalloc(&dsh2, SH2_B));
    CK(cudaMalloc(&gate, (size_t)NE * H * 2)); CK(cudaMalloc(&dgate, (size_t)NED * H * 2));
    CK(cudaMalloc(&bias, NE * 4)); CK(cudaMalloc(&dbias, NED * 4));
    CK(cudaMalloc(&x, (size_t)RMAX * H * 2)); CK(cudaMalloc(&xc, (size_t)RMAX * H * 2));
    CK(cudaMalloc(&o_big, (size_t)RMAX * H * 2)); CK(cudaMalloc(&o_ref, (size_t)RMAX * H * 2)); CK(cudaMalloc(&o_g, (size_t)RMAX * H * 2));
    k_fill_mx4<<<2048, 256>>>(fc1, FC1_B * NE, 11); k_fill_mx4<<<2048, 256>>>(fc2, FC2_B * NE, 12);
    k_fill_mx8<<<512, 256>>>(sh1w, SH13_B, 13); k_fill_mx8<<<512, 256>>>(sh3w, SH13_B, 14); k_fill_mx8<<<512, 256>>>(sh2w, SH2_B, 15);
    k_fill_mx8<<<512, 256>>>(dsh1, SH13_B, 23); k_fill_mx8<<<512, 256>>>(dsh3, SH13_B, 24); k_fill_mx8<<<512, 256>>>(dsh2, SH2_B, 25);
    k_fill_bf16<<<512, 256>>>(gate, (size_t)NE * H, 16, 0.05f); k_fill_bf16<<<512, 256>>>(dgate, (size_t)NED * H, 26, 0.05f);
    k_fill_f32<<<4, 256>>>(bias, NE, 17, 0.5f); k_fill_f32<<<4, 256>>>(dbias, NED, 27, 0.5f);
    k_fill_bf16<<<512, 256>>>(x, (size_t)RMAX * H, 18, 4.f);
    CK(cudaMemcpy(xc, x, (size_t)RMAX * H * 2, cudaMemcpyDeviceToDevice));
    k_fill_bf16<<<64, 256>>>(xc + (size_t)CTRL_ROW * H, H, 99, 4.f);               // CTRL-IN: one changed input row
    CK(cudaDeviceSynchronize());

    std::map<std::string, TensorRef> T;
    auto put = [&](int l, const char* n, const void* p, std::initializer_list<long long> sh, int dt) {
        TensorRef t;
        t.p = p; t.nd = (int)sh.size();
        int k = 0;
        for (long long s : sh) t.shape[k++] = s;
        t.bf16 = dt == 1; t.fp32 = dt == 2;
        T[lname(l, n)] = t;
    };
    for (int d = 0; d < 2; ++d)
    {
        const int l = d ? L_DRAFT : L_MAIN, ne = d ? NED : NE;
        put(l, "router.w", d ? dgate : gate, {ne, H}, 1);
        put(l, "router.bias", d ? dbias : bias, {ne}, 2);
        put(l, "sh.w1.mx.gv", d ? dsh1 : sh1w, {IW / 16, H / 32, 528}, 0);
        put(l, "sh.w3.mx.gv", d ? dsh3 : sh3w, {IW / 16, H / 32, 528}, 0);
        put(l, "sh.w2.mx.gv", d ? dsh2 : sh2w, {H / 16, IW / 32, 528}, 0);
        put(l, "x.fc1", fc1, {ne, IW / 16, H / 64, 1088}, 0);
        put(l, "x.fc2", fc2, {ne, H / 16, IW / 64, 544}, 0);
    }
    Model& m = *new Model();
    m.rank = 0;
    m.hc_rows = RMAX;
    m.ds41_find = [&T](const std::string& n) { auto it = T.find(n); return it == T.end() ? TensorRef{} : it->second; };
    m.layers.resize(2);
    m.layers[0].l = L_MAIN; m.layers[0].moe = true;
    m.layers[1].l = L_DRAFT; m.layers[1].moe = true;
    for (int i = 0; i < 2; ++i) if (ds41moe::bind_moe(&m, i, 0, nullptr)) { printf("bind_moe %d failed\n", i); return 2; }
    for (int i = 0; i < 2; ++i)
    {
        const auto* s = (const ds41moe::LayerState*)m.layers[i].ds41_moe;
        printf("# layer %d: ne %d tk %d trellis %d\n", m.layers[i].l, s->ne, s->tk, s->tr ? 1 : 0);
        if (s->tr || s->ne != (i ? NED : NE)) { printf("unexpected layer state\n"); return 2; }
    }
    const int rows_main[] = {33, 40, 41, 47, 48}, rows_draft[] = {35, 40};
    for (int i = 0; i < 2; ++i)                                                    // host init: every row count the test calls
        for (int M = 1; M <= RMAX; ++M)
            if (int r = ds41moe::prepare_moe(&m, i, M, nullptr)) { printf("prepare_moe(%d, %d) = %d\n", i, M, r); return 2; }
    cudaStream_t st;
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    mem_mark();

    std::vector<Case> cases;
    for (int M : rows_main) cases.push_back({0, "main", M, true});
    for (int M : rows_main) cases.push_back({0, "main", M, false});
    for (int M : rows_draft) cases.push_back({1, "draft", M, true});
    for (const Case& c : cases)
    {
        m.ds_moe_decode_step = c.dec;
        const int kc = c.li ? geom::DS_TOPK : c.dec ? kc_dec : kc_main;
        char tag[96];
        snprintf(tag, sizeof tag, "%s ne %d k %d rows %d%s", c.name, c.li ? NED : NE, kc, c.rows, c.dec ? " decode" : "");
        poison(o_big, st); call(m, c.li, x, c.rows, o_big, st);
        const auto big = fetch(o_big, st);
        std::string why;
        report(written_ok(big, c.rows, why), "%s: rows < M written + finite, rows >= M untouched %s", tag, why.c_str());
        for (int chunk : {32, 8})
        {
            poison(o_ref, st); call_chunked(m, c.li, x, c.rows, chunk, o_ref, st);
            const auto d = diff_rows(big, fetch(o_ref, st), c.rows);
            report(d.empty(), "%s: == %d-row chunks, %d/%d rows bit-identical (differ: %s)", tag, chunk, c.rows - (int)d.size(), c.rows, rows_s(d).c_str());
        }
        // G: the same call captured in a graph and replayed (a decode step at these rows is a graph)
        cudaGraph_t g;
        cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
        call(m, c.li, x, c.rows, o_g, st);
        CK(cudaStreamEndCapture(st, &g));
        CK(cudaGraphInstantiate(&ge, g, 0));
        poison(o_g, st);
        CK(cudaGraphLaunch(ge, st));
        const auto d = diff_rows(big, fetch(o_g, st), c.rows);
        report(d.empty(), "%s: graph replay == eager, %d/%d rows (differ: %s)", tag, c.rows - (int)d.size(), c.rows, rows_s(d).c_str());
        CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));
    }

    // controls: each runs the row compare above on a broken input or a broken call, and must see exactly the expected rows differ
    m.ds_moe_decode_step = true;
    {   // CTRL-IN
        poison(o_ref, st); call_chunked(m, 0, x, RMAX, 32, o_ref, st);
        const auto ref = fetch(o_ref, st);
        poison(o_big, st); call(m, 0, xc, RMAX, o_big, st);
        const auto d = diff_rows(fetch(o_big, st), ref, RMAX);
        report(d.size() == 1 && d[0] == CTRL_ROW, "CTRL-IN main rows 48 decode, input row %d changed: rows that differ %s (want %d only)", CTRL_ROW, rows_s(d).c_str(), CTRL_ROW);
    }
    {   // CTRL-DROP: a dispatch that stops at 32 rows
        poison(o_ref, st); call_chunked(m, 0, x, 40, 32, o_ref, st);
        const auto ref = fetch(o_ref, st);
        poison(o_big, st); call(m, 0, x, 32, o_big, st);
        const auto d = diff_rows(fetch(o_big, st), ref, 40);
        bool want = d.size() == 8;
        for (size_t i = 0; want && i < d.size(); ++i) want = d[i] == 32 + (int)i;
        report(want, "CTRL-DROP main rows 40 run as 32: rows that differ %s (want 32..39)", rows_s(d).c_str());
    }
    if (kc_dec != kc_main)
    {   // CTRL-CUT: the decode flag off on the big call only
        poison(o_ref, st); call_chunked(m, 0, x, RMAX, 32, o_ref, st);
        const auto ref = fetch(o_ref, st);
        m.ds_moe_decode_step = false;
        poison(o_big, st); call(m, 0, x, RMAX, o_big, st);
        m.ds_moe_decode_step = true;
        const auto d = diff_rows(fetch(o_big, st), ref, RMAX);
        report(d.size() == (size_t)RMAX, "CTRL-CUT main rows 48: cut %d against the decode cut %d reference, %d/48 rows differ (want 48)", kc_main, kc_dec, (int)d.size());
    }
    else printf("# CTRL-CUT skipped: GR8_DS_MOE_TOPK_DECODE == GR8_DS_MOE_TOPK (%d)\n", kc_main);
    CK(cudaDeviceSynchronize());
    mem_mark();
    printf("# peak device memory used by this arm: %.2f GB (free %.2f -> min %.2f GB)\n", (g_free0 - g_free_min) / 1e9, g_free0 / 1e9, g_free_min / 1e9);
    printf("%d PASS, %d FAIL\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}

int main(int argc, char** argv)
{
    if (argc > 1 && !strcmp(argv[1], "arm")) return run_arm();
    struct Arm { const char* sh1; const char* pdl; };
    const Arm arms[] = {{"0", "1"}, {"1", "1"}, {"0", "0"}};
    int worst = 0;
    for (const Arm& a : arms)
    {
        printf("=== arm GR8_DS_MOE_SH_1PASS=%s GR8_DS_MOE_PDL=%s GR8_DS_MOE_TOPK_DECODE=4\n", a.sh1, a.pdl);
        fflush(stdout);
        const pid_t pid = fork();
        if (pid < 0) { perror("fork"); return 2; }
        if (pid == 0)
        {
            setenv("GR8_DS_MOE_SH_1PASS", a.sh1, 1);
            setenv("GR8_DS_MOE_PDL", a.pdl, 1);
            setenv("GR8_DS_MOE_TOPK_DECODE", "4", 1);
            execl("/proc/self/exe", argv[0], "arm", (char*)nullptr);
            perror("exec");
            _exit(2);
        }
        int ws = 0;
        waitpid(pid, &ws, 0);
        const int rc = WIFEXITED(ws) ? WEXITSTATUS(ws) : 2;
        printf("=== arm exit %d\n", rc);
        worst = std::max(worst, rc);
    }
    if (!worst) printf("DS41_MOE_PW_ALL_PASS\n");
    else printf("DS41_MOE_PW FAIL (worst arm exit %d)\n", worst);
    return worst;
}
