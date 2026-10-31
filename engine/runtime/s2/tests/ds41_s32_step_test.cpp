// SESS=32 MAIN steps of more than 8 sessions: the host S2In (s2_in.h) holds them, native_inputs fills it, the DS41 layout
// check (ds41_step_layout.h, the code ds41_native.cuh runs) gives each row its slot, and the strided S2In copy of the S2
// kernels covers every word at 128 and 256 threads. An S2N of 8 refuses a 9-session step. No CUDA.
#include "s2_in.h"
#include "s2/dispatch.h"
#include "s2/ds41_step_layout.h"
#include <cstdio>
#include <vector>

using namespace g6s2;
static int fails = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x); ++fails; } } while (0)
template<class F> static bool throws(F f) { try { f(); } catch (const std::exception&) { return true; } return false; }

constexpr int MXT = 48;   // the DS41 build's G6_MX
// the fields native_inputs and ds41_session_rows use
struct Input {
    uint64_t step; int M,context,tok[MXT],pos[MXT],tgt[MXT]; float steering_scale[MXT]; int steering_slot[MXT];
    struct Sample { float inv_temp; uint32_t seed, unused; int pos,nodraft; float top_p; int topk; } srow[MXT];
    int sess[MXT];
};
struct Layout8 { int ns,cls,sess[8],row0[8],rows[8],ctx[8],src0[8]; };   // an S2N = 8 layout, for contrast

// ns sessions of rps rows each, on slots base, base + stride, ...
static Launch step(int ns, int rps, int base = 0, int stride = 1) {
    Launch l; l.step.kind = s2::Step::MAIN; l.step.cls = 1;
    for (int j = 0; j < ns; ++j) {
        const int q = base + j * stride;
        l.step.sess.push_back(q); l.step.row0.push_back(j * rps); l.step.rows.push_back(rps); l.step.ctx.push_back(100 + j);
        for (int r = 0; r < rps; ++r) l.step.r.push_back({q, 1000 + j, 99 + j - rps + 1 + r, 99 + j - rps + 1 + r, -1});
        l.source_rows.push_back(0);
    }
    l.samples.assign(size_t(ns * rps), Sampling{});
    l.steering_scales.assign(size_t(ns * rps), 0.f);
    return l;
}

static void sessions(int ns, int rps, int base, int stride) {
    const Launch l = step(ns, rps, base, stride);
    Input in{}; g6rt::S2In h{};
    CHECK(!throws([&] { native_inputs(l, in, h); }));
    CHECK(h.ns == ns && in.M == ns * rps);
    for (int j = 0; j < ns; ++j) CHECK(h.sess[j] == base + j * stride && h.row0[j] == j * rps && h.rows[j] == rps && h.ctx[j] == 100 + j);
    CHECK(ds41_session_rows(h, 32, 5, rps > 1, in) == nullptr);
    for (int r = 0; r < ns * rps; ++r) CHECK(in.sess[r] == base + (r / rps) * stride);
    // an 8-session layout refuses it ("native S2In session capacity too small")
    Input in8{}; Layout8 h8{};
    CHECK(ns <= 8 || throws([&] { native_inputs(l, in8, h8); }));
}

static void copy_cover(int nthreads) {
    std::vector<uint32_t> src(g6rt::S2IN_WORDS), dst(g6rt::S2IN_WORDS, 0xdeadbeefu), old(g6rt::S2IN_WORDS, 0xdeadbeefu);
    for (int i = 0; i < g6rt::S2IN_WORDS; ++i) src[size_t(i)] = 0x1000u + uint32_t(i);
    for (int t = 0; t < nthreads; ++t) {
        g6rt::s2in_copy(dst.data(), src.data(), t, nthreads);
        if (t < g6rt::S2IN_WORDS) old[size_t(t)] = src[size_t(t)];   // a plain copy: one word per thread
    }
    int miss = 0, old_miss = 0;
    for (int i = 0; i < g6rt::S2IN_WORDS; ++i) { miss += dst[size_t(i)] != src[size_t(i)]; old_miss += old[size_t(i)] != src[size_t(i)]; }
    CHECK(miss == 0);
    CHECK(old_miss == (nthreads < g6rt::S2IN_WORDS ? g6rt::S2IN_WORDS - nthreads : 0));
    std::printf("copy %d threads: words %d missed %d (old one-word-per-thread copy: %d)\n", nthreads, g6rt::S2IN_WORDS, miss, old_miss);
}

int main() {
    CHECK(g6rt::S2N == 32 && g6rt::S2N == int(MAX_SESSIONS) && g6rt::S2IN_WORDS == 162);
    sessions(9, 1, 0, 1);      // 9 resident sessions, the smallest count past 8
    sessions(9, 1, 3, 3);      // 9 sessions on scattered slots 3..27
    sessions(32, 1, 0, 1);     // every SESS=32 slot in one MAIN step (rps 1: MTP is off past 8)
    sessions(8, 6, 0, 1);      // the MTP verify of 8 sessions x (5 + 1) rows still fits
    sessions(1, 1, 31, 1);     // the last slot alone
    {   // 33 sessions: more than S2In holds -> refused
        const Launch l = step(33, 1);
        Input in{}; g6rt::S2In h{};
        CHECK(throws([&] { native_inputs(l, in, h); }));
    }
    {   // a session slot past ds_slots, and a non-uniform layout -> refused by the DS41 check
        Input in{}; g6rt::S2In h{};
        native_inputs(step(9, 1, 0, 4), in, h);   // slots 0..32
        CHECK(ds41_session_rows(h, 32, 5, false, in) != nullptr);
        Launch l = step(9, 1); l.step.row0[4] = 5;
        Input in2{}; g6rt::S2In h2{};
        native_inputs(l, in2, h2);
        CHECK(ds41_session_rows(h2, 32, 5, false, in2) != nullptr);
    }
    copy_cover(128);           // k_s2_cells
    copy_cover(256);           // k_s2_draft_hid, k_s2_draft_keep
    copy_cover(32);
    std::printf("ds41_s32_step_test: %s (%d failure(s))\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
