// GPU check of the KDA decode block (kda.cuh) against the W0 fp32 goldens. TP4 ranks run one after another on one
// GPU, each bound through kda::bind_kda on loader-format tensors (replicated BF16 .gv records, o_proj shard of its
// heads), state and scratch from build_state / build_scratch; partials summed in fp32 in rank order.
//   core: projections from the golden (`mixed`, f_a/g_a/b in double) into the scratch, then prep/step/out + o_proj:
//         gate 1e-4 on the fp32 ops, 1e-2 on `out` (BF16 o_gated).
//   full: kda::launch from the BF16 fragment table of the golden block input: gate 1e-2 on every op (BF16 x).
//   chunk: the same through the chunked prefill (kda_prefill.cuh: launch_core_prefill / launch_prefill), calls of
//          up to PF_ROWS rows of one slot.
//
//   nvcc -O3 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -DG6_MODEL_FLASH=1 -o kda_test kda_test.cu
//   kda_test --dense DIR [--prompt F] [--step F] [--long F] [--layers 0,1,2]
#include <cstdio>
#include <functional>
#include <map>
#include "../kda_prefill.cuh"
#include "common.h"

using namespace kda;
using namespace kda_test;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(3); } } while (0)

struct TRef
{
    const void* p = nullptr;
    int nd = 0;
    long long shape[4] = {0, 0, 0, 0};
};

template <class T> static T* up(const T* h, size_t n)
{
    T* d;
    CK(cudaMalloc(&d, n * sizeof(T)));
    CK(cudaMemcpy(d, h, n * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <class T> static std::vector<T> down(const T* d, size_t n)
{
    std::vector<T> h(n);
    CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

// loader-rs records::bf16_record: [n/16][k/16][512], lane l = (r, c): (r,2c) (r,2c+1) (r+8,2c) (r+8,2c+1) (r,2c+8) ...
static std::vector<uint16_t> gv_record(const uint16_t* w, int n, int k, int ld, int col0)
{
    std::vector<uint16_t> out(static_cast<size_t>(n) * k);
    const int s = k / 16;
    for (int t = 0; t < n / 16; ++t)
        for (int j = 0; j < s; ++j)
        {
            uint16_t* rec = &out[(static_cast<size_t>(t) * s + j) * 256];
            for (int l = 0; l < 32; ++l)
            {
                const int r = l / 4, c = l % 4;
                const int e[8][2] = {{r, 2 * c}, {r, 2 * c + 1}, {r + 8, 2 * c}, {r + 8, 2 * c + 1},
                                     {r, 2 * c + 8}, {r, 2 * c + 9}, {r + 8, 2 * c + 8}, {r + 8, 2 * c + 9}};
                for (int i = 0; i < 8; ++i) rec[l * 8 + i] = w[static_cast<size_t>(16 * t + e[i][0]) * ld + col0 + 16 * j + e[i][1]];
            }
        }
    return out;
}

static std::vector<uint16_t> x_frag(const float* x, int M)
{
    std::vector<uint16_t> f(static_cast<size_t>((M + 7) / 8) * HID_ * 8, 0);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < HID_; ++k) f[frag_u16(m, k, HID_)] = f2bf_rn(x[static_cast<size_t>(m) * HID_ + k]);
    return f;
}

constexpr int NR = 4, ROWS = 8, SLOTS = 2, PF_ROWS = 1024;

struct Layer   // one layer, NR ranks
{
    int l = 0;
    std::map<std::string, TRef> img[NR];
    KdaLayer w[NR];
    KdaState s[NR];
    KdaScratch sc[NR], psc[NR];         // decode scratch (ROWS), prefill scratch (PF_ROWS)
    KdaPrefillScratch pps[NR];
    std::vector<void*> mem;
    void build(const W& hw, int l_)
    {
        l = l_;
        char pre[16];
        snprintf(pre, sizeof pre, "L%03d.", l);
        const std::string P = pre, A = P + "self_attn.";
        auto put = [&](std::map<std::string, TRef>& m, const std::string& n, const void* d, std::initializer_list<long long> sh) {
            TRef t; t.p = d; t.nd = 0;
            for (long long v : sh) t.shape[t.nd++] = v;
            m[n] = t;
        };
        auto rec = [&](const uint16_t* w, int n, int k, int ld, int c0) {
            auto r = gv_record(w, n, k, ld, c0);
            void* d = up(r.data(), r.size());
            mem.push_back(d);
            return d;
        };
        auto raw = [&](const void* h, size_t bytes) {
            void* d = up(static_cast<const uint8_t*>(h), bytes);
            mem.push_back(d);
            return d;
        };
        const st::File& f = hw.f;
        void* q = rec(f["self_attn.q_proj.bf16"].u16(), D, HID_, HID_, 0);
        void* k = rec(f["self_attn.k_proj.bf16"].u16(), D, HID_, HID_, 0);
        void* v = rec(f["self_attn.v_proj.bf16"].u16(), D, HID_, HID_, 0);
        void* fa = rec(hw.fa, DK, HID_, HID_, 0);
        void* ga = rec(hw.ga, DK, HID_, HID_, 0);
        void* b = rec(hw.b, HEADS, HID_, HID_, 0);
        void* cv[3];
        for (int i = 0; i < 3; ++i) cv[i] = raw(hw.conv[i], static_cast<size_t>(D) * CONV * 2);
        void* fb = raw(hw.fb, static_cast<size_t>(D) * DK * 2);
        void* gb = raw(hw.gb, static_cast<size_t>(D) * DK * 2);
        void* al = raw(hw.A_log, HEADS * 4);
        void* dt = raw(hw.dt_bias, D * 4);
        void* on = raw(hw.onorm, DK * 2);
        for (int r = 0; r < NR; ++r)
        {
            auto& m = img[r];
            const long long G = HID_ / 16;
            put(m, A + "q_proj.weight.gv", q, {D / 16, G, 512});
            put(m, A + "k_proj.weight.gv", k, {D / 16, G, 512});
            put(m, A + "v_proj.weight.gv", v, {D / 16, G, 512});
            put(m, A + "f_a_proj.weight.gv", fa, {DK / 16, G, 512});
            put(m, A + "g_a_proj.weight.gv", ga, {DK / 16, G, 512});
            put(m, A + "b_proj.weight.gv", b, {HEADS / 16, G, 512});
            put(m, A + "q_conv1d.weight", cv[0], {D, 1, CONV});
            put(m, A + "k_conv1d.weight", cv[1], {D, 1, CONV});
            put(m, A + "v_conv1d.weight", cv[2], {D, 1, CONV});
            put(m, A + "f_b_proj.weight", fb, {D, DK});
            put(m, A + "g_b_proj.weight", gb, {D, DK});
            put(m, A + "A_log", al, {HEADS});
            put(m, A + "dt_bias", dt, {D});
            put(m, A + "o_norm.weight", on, {DK});
            int h0, nh;
            head_rows(l, NR, r, h0, nh);
            put(m, P + "o_proj.gv", rec(hw.o, HID_, nh * DK, D, h0 * DK), {G, nh * DK / 16, 512});
            auto find = [&](const std::string& n) { auto it = m.find(n); return it == m.end() ? TRef() : it->second; };
            if (const char* e = bind_kda(w[r], l, find, r, NR)) { fprintf(stderr, "bind_kda: %s\n", e); exit(3); }
            if (w[r].h0 != h0 || w[r].nh != nh) { fprintf(stderr, "bind_kda heads\n"); exit(3); }
            if (const char* e = build_state(s[r], SLOTS, NR)) { fprintf(stderr, "%s\n", e); exit(3); }
            if (const char* e = build_scratch(sc[r], ROWS, NR)) { fprintf(stderr, "%s\n", e); exit(3); }
            if (const char* e = build_scratch(psc[r], PF_ROWS, NR)) { fprintf(stderr, "%s\n", e); exit(3); }
            if (const char* e = build_prefill_scratch(pps[r], PF_ROWS, NR)) { fprintf(stderr, "%s\n", e); exit(3); }
        }
    }
    void free_all()
    {
        for (void* p : mem) cudaFree(p);
        for (int r = 0; r < NR; ++r)
        {
            cudaFree(s[r].S); cudaFree(s[r].conv);
            for (const KdaScratch* x : {&sc[r], &psc[r]})
            {
                for (float* p : {x->P, x->C, x->G, x->Z, x->O, x->OG}) cudaFree(p);
                cudaFree(x->OF); cudaFree(x->OH);
            }
            for (float* p : {pps[r].QE, pps[r].KD, pps[r].W, pps[r].U, pps[r].GC, pps[r].AQ}) cudaFree(p);
        }
    }
    void reset_all()
    {
        for (int r = 0; r < NR; ++r)
            for (int sl = 0; sl < SLOTS; ++sl) reset(s[r], sl, 0);
        CK(cudaDeviceSynchronize());
    }
    void set_state(int slot, const float* gS, const float* gconv)
    {
        for (int r = 0; r < NR; ++r)
        {
            const KdaState& S = s[r];
            const int li = kda_slot(l), h0 = w[r].h0, nh = w[r].nh;
            float* base = S.S + li * S.s_layer() + static_cast<size_t>(slot) * S.nhmax * DK * DV;
            CK(cudaMemcpy(base, gS + static_cast<size_t>(h0) * DK * DV, static_cast<size_t>(nh) * DK * DV * 4, cudaMemcpyHostToDevice));
            Args a; a.nhmax = S.nhmax;
            float* cb = S.conv + li * S.c_layer();
            for (int j = 0; j < NHIST; ++j)
                for (int st_ = 0; st_ < 3; ++st_)
                    CK(cudaMemcpy(cb + a.c_off(slot, j, st_, 0), gconv + static_cast<size_t>(j) * 3 * D + st_ * D + h0 * DK, nh * DK * 4, cudaMemcpyHostToDevice));
        }
    }
    void get_state(int slot, std::vector<float>& S_, std::vector<float>& conv)
    {
        S_.assign(static_cast<size_t>(HEADS) * DK * DV, 0);
        conv.assign(static_cast<size_t>(NHIST) * 3 * D, 0);
        for (int r = 0; r < NR; ++r)
        {
            const KdaState& S = s[r];
            const int li = kda_slot(l), h0 = w[r].h0, nh = w[r].nh;
            CK(cudaMemcpy(&S_[static_cast<size_t>(h0) * DK * DV], S.S + li * S.s_layer() + static_cast<size_t>(slot) * S.nhmax * DK * DV,
                          static_cast<size_t>(nh) * DK * DV * 4, cudaMemcpyDeviceToHost));
            Args a; a.nhmax = S.nhmax;
            const float* cb = S.conv + li * S.c_layer();
            for (int j = 0; j < NHIST; ++j)
                for (int st_ = 0; st_ < 3; ++st_)
                    CK(cudaMemcpy(&conv[static_cast<size_t>(j) * 3 * D + st_ * D + h0 * DK], cb + a.c_off(slot, j, st_, 0), nh * DK * 4, cudaMemcpyDeviceToHost));
        }
    }
};

struct Run
{
    std::vector<float> g, q, k, v, o, og, out;
    void init(int T)
    {
        const size_t n = static_cast<size_t>(T) * D;
        g.assign(n, 0); q = g; k = g; v = g; o = g; og = g;
        out.assign(static_cast<size_t>(T) * HID_, 0);
    }
};

// rows t0 .. t0 + M - 1 of a sequence through every rank; core = projections from the golden, else kda::launch from x;
// chunked = the prefill path
static void call(Layer& L, bool core, const float* x, const float* mixed, const float* sp, const int* slots, int M, int t0, Run& out,
                 int* d_slots, void* d_part, uint16_t* d_xf, bool chunked = false)
{
    CK(cudaMemcpy(d_slots, slots, M * 4, cudaMemcpyHostToDevice));
    if (!core)
    {
        auto xf = x_frag(x, M);
        CK(cudaMemcpy(d_xf, xf.data(), xf.size() * 2, cudaMemcpyHostToDevice));
    }
    std::vector<float> acc(static_cast<size_t>(M) * HID_, 0.f);
    for (int r = 0; r < NR; ++r)
    {
        const KdaLayer& w = L.w[r];
        const KdaScratch& sc = chunked ? L.psc[r] : L.sc[r];
        const int dl = w.nh * DK;
        if (core)
        {
            Args a = make_args(w, L.s[r], sc, d_slots, M);
            const PLayout pl = a.pl();
            std::vector<float> P(static_cast<size_t>(M) * pl.ld());
            for (int m = 0; m < M; ++m)
            {
                for (int s = 0; s < 3; ++s)
                    for (int c = 0; c < dl; ++c) P[static_cast<size_t>(m) * pl.ld() + s * dl + c] = mixed[static_cast<size_t>(m) * 3 * D + s * D + w.h0 * DK + c];
                for (int i = 0; i < 320; ++i) P[static_cast<size_t>(m) * pl.ld() + pl.fa() + i] = sp[static_cast<size_t>(m) * 320 + i];
            }
            CK(cudaMemcpy(sc.P, P.data(), P.size() * 4, cudaMemcpyHostToDevice));
            if (chunked) launch_core_prefill(make_pf_args(a, L.pps[r]), 0);
            else launch_core(a, 0);
            launch_oproj_rows(w, sc, M, d_part, gv::O_BF16, 0);
        }
        else if (chunked)
            launch_prefill(w, L.s[r], sc, L.pps[r], reinterpret_cast<const u32*>(d_xf), d_slots, M, d_part, 0);
        else
            launch(w, L.s[r], sc, reinterpret_cast<const u32*>(d_xf), d_slots, M, d_part, 0);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        auto part = down(static_cast<const uint16_t*>(d_part), static_cast<size_t>(M) * HID_);
        for (size_t i = 0; i < part.size(); ++i) acc[i] += bf2f(part[i]);
        auto C = down(sc.C, static_cast<size_t>(M) * 3 * dl), G = down(sc.G, static_cast<size_t>(M) * dl);
        auto O = down(sc.O, static_cast<size_t>(M) * dl), OG = down(sc.OG, static_cast<size_t>(M) * dl);
        for (int m = 0; m < M; ++m)
        {
            const size_t t = static_cast<size_t>(t0 + m);
            for (int c = 0; c < dl; ++c)
            {
                const size_t gi = t * D + w.h0 * DK + c, li = static_cast<size_t>(m) * dl + c;
                out.g[gi] = G[li]; out.o[gi] = O[li]; out.og[gi] = OG[li];
                out.v[gi] = C[static_cast<size_t>(m) * 3 * dl + 2 * dl + c];
            }
            for (int h = 0; h < w.nh; ++h)
                for (int s = 0; s < 2; ++s)
                {
                    const float* xx = &C[static_cast<size_t>(m) * 3 * dl + s * dl + h * DK];
                    double ss = 0;
                    for (int i = 0; i < DK; ++i) ss += static_cast<double>(xx[i]) * xx[i];
                    const double dn = std::sqrt(ss + L2_EPS);
                    float* dst = (s ? out.k : out.q).data() + t * D + (w.h0 + h) * DK;
                    for (int i = 0; i < DK; ++i) dst[i] = static_cast<float>(xx[i] / dn * (s ? 1.0 : QSCALE));
                }
        }
    }
    memcpy(&out.out[static_cast<size_t>(t0) * HID_], acc.data(), acc.size() * 4);
}

static void compare(const st::File& g, const std::string& L, const Run& r, int t0, int T, const std::string& tag, double gf)
{
    const size_t off = static_cast<size_t>(t0) * D, n = static_cast<size_t>(T) * D;
    for (const char* op : {"g", "q", "k", "v", "o", "o_gated"})
    {
        const std::vector<float>& a = !strcmp(op, "g") ? r.g : !strcmp(op, "q") ? r.q : !strcmp(op, "k") ? r.k : !strcmp(op, "v") ? r.v
                                    : !strcmp(op, "o") ? r.o : r.og;
        gate(tag + " " + op, rel(a.data(), g[L + "kda." + op].f32() + off, n), gf);
    }
    gate(tag + " out", rel(r.out.data(), g[L + "kda.out"].f32() + static_cast<size_t>(t0) * HID_, static_cast<size_t>(T) * HID_), 1e-2);
}

int main(int argc, char** argv)
{
    std::string dense, prompt, step, lng, layers = "0,1,2";
    for (int i = 1; i + 1 < argc; i += 2)
    {
        const std::string k = argv[i];
        if (k == "--dense") dense = argv[i + 1];
        else if (k == "--prompt") prompt = argv[i + 1];
        else if (k == "--step") step = argv[i + 1];
        else if (k == "--long") lng = argv[i + 1];
        else if (k == "--layers") layers = argv[i + 1];
    }
    if (dense.empty()) { fprintf(stderr, "usage: see header\n"); return 2; }
    std::vector<int> ls;
    for (const char* p = layers.c_str(); *p;) { ls.push_back(strtol(p, const_cast<char**>(&p), 10)); if (*p == ',') ++p; }
    int* d_slots;
    void* d_part;
    uint16_t* d_xf;
    CK(cudaMalloc(&d_slots, PF_ROWS * 4));
    CK(cudaMalloc(&d_part, static_cast<size_t>(PF_ROWS) * HID_ * 2));
    CK(cudaMalloc(&d_xf, static_cast<size_t>(PF_ROWS) * HID_ * 2));
    static const std::vector<int> zero(PF_ROWS, 0);

    for (int l : ls)
    {
        W hw;
        hw.load(dense, l);
        Layer Ly;
        Ly.build(hw, l);
        char Lb[8];
        snprintf(Lb, sizeof Lb, "L%02d.", l);
        const std::string L = Lb;
        // one sequence from the given state (zero if S_in null): rows of M per call; chunked: the prefill path with
        // call sizes `sizes` (cyclic; empty = M)
        auto seq = [&](const st::File& g, bool core, int T, int M, const float* S_in, const float* conv_in, Run& r,
                       const std::vector<int>& marks, std::function<void(int)> snap, bool chunked = false, std::vector<int> sizes = {}) {
            Ly.reset_all();
            if (S_in) Ly.set_state(0, S_in, conv_in);
            r.init(T);
            if (sizes.empty()) sizes = {M};
            const auto sp = small_proj(hw, g[L + "attn_in"].f32(), T);
            for (int t0 = 0, n = 0; t0 < T; t0 += sizes[n++ % sizes.size()])
            {
                const int m = std::min(sizes[n % sizes.size()], T - t0);
                if (m > (chunked ? PF_ROWS : ROWS)) { fprintf(stderr, "call of %d rows\n", m); exit(2); }
                call(Ly, core, g[L + "attn_in"].f32() + static_cast<size_t>(t0) * HID_, g[L + "kda.mixed"].f32() + static_cast<size_t>(t0) * 3 * D,
                     &sp[static_cast<size_t>(t0) * 320], zero.data(), m, t0, r, d_slots, d_part, d_xf, chunked);
                for (int mk : marks)
                    if (mk == t0 + m) snap(mk);
            }
        };
        std::vector<float> S, conv;
        if (!prompt.empty())
        {
            st::File g;
            g.open(prompt);
            const int T = static_cast<int>(g[L + "attn_in"].shape[0]);
            printf("prompt %s: %d tokens from zero state, TP%d\n", L.c_str(), T, NR);
            Run c1, c8, f1;
            seq(g, true, T, 1, nullptr, nullptr, c1, {T}, [&](int) { Ly.get_state(0, S, conv); });
            compare(g, L, c1, 0, T, "core M1", 1e-4);
            gate("core M1 S after " + std::to_string(T), rel(S.data(), g[L + "kda.S"].f32(), S.size()), 1e-4);
            seq(g, true, T, 8, nullptr, nullptr, c8, {}, [](int) {});
            same("core M8 (one slot) o_gated == M1", c8.og, c1.og);
            same("core M8 (one slot) out == M1", c8.out, c1.out);
            seq(g, false, T, 1, nullptr, nullptr, f1, {T}, [&](int) { Ly.get_state(0, S, conv); });
            compare(g, L, f1, 0, T, "full M1", 1e-2);
            gate("full M1 S after " + std::to_string(T), rel(S.data(), g[L + "kda.S"].f32(), S.size()), 1e-2);
            // two slots interleaved, same sequence in both: rows [slot 0 t, slot 1 t]
            {
                Ly.reset_all();
                const auto sp = small_proj(hw, g[L + "attn_in"].f32(), T);
                Run a, b;
                a.init(T); b.init(T);
                const int sl[2] = {0, 1};
                for (int t = 0; t < T; ++t)
                {
                    const size_t xo = static_cast<size_t>(t) * HID_, mo = static_cast<size_t>(t) * 3 * D;
                    std::vector<float> x2(2 * HID_), m2(2 * 3 * D), s2(640);
                    for (int i = 0; i < 2; ++i)
                    {
                        memcpy(&x2[i * HID_], g[L + "attn_in"].f32() + xo, HID_ * 4);
                        memcpy(&m2[i * 3 * D], g[L + "kda.mixed"].f32() + mo, 3 * D * 4);
                        memcpy(&s2[i * 320], &sp[static_cast<size_t>(t) * 320], 320 * 4);
                    }
                    Run two;
                    two.init(2);
                    call(Ly, true, x2.data(), m2.data(), s2.data(), sl, 2, 0, two, d_slots, d_part, d_xf);
                    memcpy(&a.og[static_cast<size_t>(t) * D], &two.og[0], D * 4);
                    memcpy(&b.og[static_cast<size_t>(t) * D], &two.og[D], D * 4);
                }
                same("core 2 slots interleaved: slot 0 == M1", a.og, c1.og);
                same("core 2 slots interleaved: slot 1 == M1", b.og, c1.og);
            }
            {
                Run k1, k2, kf;
                seq(g, true, T, T, nullptr, nullptr, k1, {T}, [&](int) { Ly.get_state(0, S, conv); }, true);
                compare(g, L, k1, 0, T, "chunk core M" + std::to_string(T), 1e-4);
                gate("chunk core S after " + std::to_string(T), rel(S.data(), g[L + "kda.S"].f32(), S.size()), 1e-4);
                gate("chunk core o == recurrent core o", rel(k1.o.data(), c1.o.data(), k1.o.size()), 1e-4);
                seq(g, true, T, T, nullptr, nullptr, k2, {T}, [&](int) { Ly.get_state(0, S, conv); }, true, {5, 59});
                compare(g, L, k2, 0, T, "chunk core calls 5,59", 1e-4);
                gate("chunk core calls 5,59 S after " + std::to_string(T), rel(S.data(), g[L + "kda.S"].f32(), S.size()), 1e-4);
                seq(g, false, T, T, nullptr, nullptr, kf, {T}, [&](int) { Ly.get_state(0, S, conv); }, true);
                compare(g, L, kf, 0, T, "chunk full M" + std::to_string(T), 1e-2);
                gate("chunk full S after " + std::to_string(T), rel(S.data(), g[L + "kda.S"].f32(), S.size()), 1e-2);
            }
        }
        if (!step.empty())
        {
            st::File g;
            g.open(step);
            if (g.has(L + "state_in.S"))
            {
                printf("step %s: one token from the golden state_in, TP%d\n", L.c_str(), NR);
                for (bool core : {true, false})
                {
                    Run r;
                    seq(g, core, 1, 1, g[L + "state_in.S"].f32(), g[L + "state_in.conv"].f32(), r, {1}, [&](int) { Ly.get_state(0, S, conv); });
                    const std::string tag = core ? "core step" : "full step";
                    compare(g, L, r, 0, 1, tag, core ? 1e-4 : 1e-2);
                    gate(tag + " S", rel(S.data(), g[L + "kda.S"].f32(), S.size()), core ? 1e-4 : 1e-2);
                    Run rc;
                    seq(g, core, 1, 1, g[L + "state_in.S"].f32(), g[L + "state_in.conv"].f32(), rc, {1}, [&](int) { Ly.get_state(0, S, conv); }, true);
                    compare(g, L, rc, 0, 1, "chunk " + tag, core ? 1e-4 : 1e-2);
                    gate("chunk " + tag + " S", rel(S.data(), g[L + "kda.S"].f32(), S.size()), core ? 1e-4 : 1e-2);
                }
            }
        }
        if (!lng.empty())
        {
            st::File g;
            g.open(lng);
            if (g.has(L + "attn_in"))
            {
                const int T = static_cast<int>(g[L + "attn_in"].shape[0]);
                printf("long %s: %d tokens from zero state, TP%d, M1\n", L.c_str(), T, NR);
                std::vector<float> rec_o;
                for (bool core : {true, false})
                {
                    Run r;
                    std::vector<int> marks = {1, 64, T};
                    std::vector<std::pair<int, std::pair<std::vector<float>, std::vector<float>>>> snaps;
                    seq(g, core, T, 1, nullptr, nullptr, r, marks, [&](int t) { Ly.get_state(0, S, conv); snaps.push_back({t, {S, conv}}); });
                    const std::string tag = core ? "long core" : "long full";
                    const double gf = core ? 1e-4 : 1e-2;
                    compare(g, L, r, 0, T, tag, gf);
                    for (auto& s : snaps)
                    {
                        const std::string t = std::to_string(s.first);
                        gate(tag + " S after " + t, rel(s.second.first.data(), g[L + "S." + t].f32(), s.second.first.size()), gf);
                        gate(tag + " conv after " + t, rel(s.second.second.data(), g[L + "conv." + t].f32(), s.second.second.size()), gf);
                    }
                    if (core) rec_o = r.o;
                }
                // chunked prefill: calls 1, 62, 1, 1024 x 3, T - 3137, 1 (state marks at call ends); full: calls of 1024
                for (bool core : {true, false})
                {
                    Run r;
                    std::vector<int> marks = {1, 63, 64, T - 1, T};
                    std::vector<int> sizes = core ? std::vector<int>{1, 62, 1, 1024, 1024, 1024, T - 3137, 1} : std::vector<int>{PF_ROWS};
                    std::vector<std::pair<int, std::pair<std::vector<float>, std::vector<float>>>> snaps;
                    seq(g, core, T, PF_ROWS, nullptr, nullptr, r, core ? marks : std::vector<int>{T},
                        [&](int t) { Ly.get_state(0, S, conv); snaps.push_back({t, {S, conv}}); }, true, sizes);
                    const std::string tag = core ? "long chunk core" : "long chunk full";
                    const double gf = core ? 1e-4 : 1e-2;
                    compare(g, L, r, 0, T, tag, gf);
                    for (auto& s : snaps)
                    {
                        const std::string t = std::to_string(s.first);
                        gate(tag + " S after " + t, rel(s.second.first.data(), g[L + "S." + t].f32(), s.second.first.size()), gf);
                        gate(tag + " conv after " + t, rel(s.second.second.data(), g[L + "conv." + t].f32(), s.second.second.size()), gf);
                    }
                    if (core)
                    {
                        gate(tag + " o == recurrent core o", rel(r.o.data(), rec_o.data(), r.o.size()), 1e-4);
                        gate(tag + " o vs ref kda_chunk o", rel(r.o.data(), g[L + "chunk.o"].f32(), r.o.size()), 1e-4);
                        gate(tag + " S vs ref kda_chunk S", rel(snaps.back().second.first.data(), g[L + "chunk.S"].f32(), snaps.back().second.first.size()), 1e-4);
                        for (int t : {64, 2048, T})
                        {
                            const size_t off = static_cast<size_t>(t - 64) * D, n = static_cast<size_t>(64) * D;
                            gate(tag + " o, tokens " + std::to_string(t - 64) + ".." + std::to_string(t - 1), rel(r.o.data() + off, g[L + "kda.o"].f32() + off, n), 1e-4);
                        }
                    }
                }
            }
        }
        Ly.free_all();
    }
    printf("%s: %d gate failures\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
