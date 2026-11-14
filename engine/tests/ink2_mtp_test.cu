// The Inkling 2 Small MTP drafter on the device, through the whole path a server takes: a miniature checkpoint
// directory (ink2_mini_ckpt.h: 6 trunk layers with NVFP4 and MXFP8 experts, 3 drafter layers, the rnn_hidden Markov
// head) is written to /tmp, planned by the loader and run by ink2rt::Engine on 1, 4 and 6 simulated ranks.
//
//   1. One draft block of 7 after a 70-token prefill (past the first 64-token page and both windows).
//      The drafter's device state on 1 rank (context K / V rows of every drafter layer, the taps of the last row) is
//      given to the CPU reference (ink2ref::Model::mtp_set_state), which then drafts the same block teacher-forced
//      with the device's tokens, at Config::tp 1, 4 and 6. A device token must be the reference's argmax; a reference
//      logit gap of at most max(|top|, 1) / 32 is counted as a near tie and printed (the device and the reference
//      round BF16 at different places), anything larger fails. The same state, cut to each rank's KV heads, is then
//      given to the engines of 4 and 6 ranks: their drafts must be the tokens of 1 rank under the same rule.
//   2. On each of 1, 4 and 6 ranks: 24 tokens, greedy and sampled (temperature 0.7, top-p 0.9, seed 42), without
//      the drafter, with 3 and with 7 drafts a round, one session alone and 3 sessions in one batch (different
//      prompts, so each session has its own position and its own accepted run; the runs cross position 64). Every
//      run must give the tokens of the plain single-session run: exact.
//   3. On each of 1, 4 and 6 ranks, 7 drafts a round: a session is stopped at position 70, its state (trunk and
//      drafter) captured through KvStateDesc / StashPool / KvStateTiers into a memory store and into a file store,
//      restored into another slot, and both slots continue: the same drafts and the same tokens, exact.
//
// Build (from engine/runtime, after scripts/build.sh has built the Rust libraries and bin/obj/g6pack.o):
//   nvcc -O3 -std=c++17 -Xcompiler -ffp-contract=off -gencode=arch=compute_121a,code=sm_121a -DG6_MODEL_INK2=1 -DDEC_S2 \
//     -DG6_RS_LOADER -I . -I ../kernels -I ../attn -I ../ref/ink2 -o ../../bin/ink2_mtp_test ../tests/ink2_mtp_test.cu \
//     ../../bin/obj/g6pack.o ../loader-rs/target/release/libg6load.a ../knobs-rs/target/release/libgr8knobs.a \
//     ../target/release/libg6host.a -lcublas -lpthread -ldl -lm
// Run: bin/ink2_mtp_test   (one GPU; prints PASS and returns 0)
#include "../runtime/s2/kv_state_desc.h"
#include "../runtime/s2/kv_state_mover.cuh"
#include "../runtime/ink2_step.cuh"
#include "ink2_mini_ckpt.h"
#include <filesystem>
#include <memory>

using namespace ink2mini;
using ink2rt::Engine;
using ink2rt::Options;
using ink2rt::RowOut;
using ink2rt::RowSample;

#define CK(x)                                                                                              \
    do {                                                                                                   \
        const cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess) { fprintf(stderr, "FATAL %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } \
    } while (0)

static int fails = 0;
static void check(bool ok, const std::string& what)
{
    if (ok) return;
    ++fails;
    if (fails <= 40) fprintf(stderr, "FAIL %s\n", what.c_str());
}

constexpr int SLOTS = 3, CAP = 192, CHUNK = 16, GEN = 24, PAGE = 64;

struct Sampling
{
    float inv_temp = 0, top_p = 0;
    uint32_t seed = 0;
    RowSample row(int pos) const { return RowSample{inv_temp, top_p, seed, static_cast<uint32_t>(pos + 1)}; }   // the row at position pos
    RowSample draft() const { return RowSample{inv_temp, top_p, seed, 0}; }
};

struct Sess
{
    int slot = 0, pos = 0, next = 0;   // pos: committed positions; next: the token sampled at pos - 1
    std::vector<int> fed;              // the tokens at positions 0 .. pos - 1
    std::vector<int> said;             // the generated tokens, `next` included
    std::vector<int> drafts;           // every draft of every round, in order
};

static std::unique_ptr<Engine> engine(const std::string& dir, int T, int mtp)
{
    Options o;
    o.nr = T;
    o.rank0 = 0;
    o.nloc = T;
    o.slots = SLOTS;
    o.mtp = mtp;
    o.max_rows = SLOTS * 8;
    o.cap = CAP;
    o.nc = 64;
    std::unique_ptr<Engine> e(new Engine);
    e->init(dir, o);
    return e;
}

static Sess prefill(Engine& e, int slot, const std::vector<int>& prompt, const Sampling& sp)
{
    Sess s;
    s.slot = slot;
    e.reset_slot(slot);
    std::vector<RowOut> out;
    const int P = static_cast<int>(prompt.size());
    for (int p = 0; p < P; p += CHUNK)
    {
        const int n = std::min(CHUNK, P - p);
        ink2::RowsHost rh;
        rh.add(slot, p, n);
        std::vector<RowSample> sm;
        for (int i = 0; i < n; ++i) sm.push_back(sp.row(p + i));
        e.forward(rh, &prompt[static_cast<size_t>(p)], sm.data(), out);
        e.commit();
        s.next = out[static_cast<size_t>(n) - 1].tok;
    }
    s.pos = P;
    s.fed = prompt;
    s.said.push_back(s.next);
    return s;
}

// One plain step of every session: its last token goes in, the next comes out.
static void plain_step(Engine& e, std::vector<Sess*>& ss, const Sampling& sp)
{
    ink2::RowsHost rh;
    std::vector<int> tk;
    std::vector<RowSample> sm;
    std::vector<RowOut> out;
    for (Sess* s : ss)
    {
        rh.add(s->slot, s->pos, 1);
        tk.push_back(s->next);
        sm.push_back(sp.row(s->pos));
    }
    e.forward(rh, tk.data(), sm.data(), out);
    e.commit();
    for (size_t i = 0; i < ss.size(); ++i)
    {
        ss[i]->fed.push_back(ss[i]->next);
        ss[i]->next = out[i].tok;
        ss[i]->said.push_back(out[i].tok);
        ss[i]->pos += 1;
    }
}

// One speculative round of every session with K drafts: draft, verify in one step, commit each session's accepted run.
static void spec_round(Engine& e, std::vector<Sess*>& ss, int K, const Sampling& sp)
{
    const size_t S = ss.size();
    const int n = K + 1;
    std::vector<int> slots, qp, golden, d, tk;
    std::vector<RowSample> dsm(S, sp.draft()), sm;
    for (Sess* s : ss)
    {
        slots.push_back(s->slot);
        qp.push_back(s->pos - 1);
        golden.push_back(s->next);
    }
    e.mtp_draft(slots, qp, golden, dsm, K, d);
    ink2::RowsHost rh;
    for (size_t i = 0; i < S; ++i)
    {
        rh.add(ss[i]->slot, ss[i]->pos, n);
        tk.push_back(ss[i]->next);
        for (int j = 0; j < K; ++j)
        {
            tk.push_back(d[i * K + j]);
            ss[i]->drafts.push_back(d[i * K + j]);
        }
        for (int j = 0; j < n; ++j) sm.push_back(sp.row(ss[i]->pos + j));
    }
    std::vector<RowOut> out;
    e.forward(rh, tk.data(), sm.data(), out);
    std::vector<uint8_t> acc(S * n, 0);
    for (size_t i = 0; i < S; ++i)
    {
        int a = 0;
        while (a < K && d[i * K + a] == out[i * n + a].tok) ++a;
        for (int j = 0; j <= a; ++j)
        {
            acc[i * n + j] = 1;
            ss[i]->fed.push_back(tk[i * n + j]);
            ss[i]->said.push_back(out[i * n + j].tok);
        }
        ss[i]->next = out[i * n + a].tok;
        ss[i]->pos += a + 1;
    }
    e.commit(&acc);
}

// `gen` tokens of every prompt (slot = index), K drafts a round (0 = plain steps). Each session's tokens, cut to gen.
static std::vector<std::vector<int>> generate(Engine& e, const std::vector<std::vector<int>>& prompts, int K, const Sampling& sp, int gen,
                                              std::vector<std::vector<int>>* drafts = nullptr)
{
    std::vector<Sess> s;
    for (size_t i = 0; i < prompts.size(); ++i) s.push_back(prefill(e, static_cast<int>(i), prompts[i], sp));
    std::vector<Sess*> ss;
    for (Sess& x : s) ss.push_back(&x);
    const auto short_one = [&]() {
        for (const Sess& x : s)
            if (static_cast<int>(x.said.size()) < gen) return true;
        return false;
    };
    while (short_one())
    {
        if (K > 0) spec_round(e, ss, K, sp);
        else plain_step(e, ss, sp);
    }
    std::vector<std::vector<int>> r;
    for (Sess& x : s)
    {
        x.said.resize(static_cast<size_t>(gen));
        r.push_back(x.said);
        if (drafts) drafts->push_back(x.drafts);
    }
    return r;
}

static std::vector<uint16_t> pull(const uint8_t* dev, size_t cells)
{
    std::vector<uint16_t> v(cells);
    if (cells) CK(cudaMemcpy(v.data(), dev, cells * 2, cudaMemcpyDeviceToHost));
    return v;
}
static void push(uint8_t* dev, const std::vector<uint16_t>& v)
{
    if (!v.empty()) CK(cudaMemcpy(dev, v.data(), v.size() * 2, cudaMemcpyHostToDevice));
}

// The drafter's state of slot 0 on a 1-rank engine, on the host.
struct DraftState
{
    std::vector<int> global;                  // per drafter layer
    std::vector<std::vector<uint16_t>> kv;    // local: [2][MW][nk]; global: [max_pps][2][64][nk]
    std::vector<uint16_t> taps;               // [taps][H]
    int max_pps = 0, nk = 0;
};

static DraftState take_state(Engine& e)
{
    DraftState st;
    st.max_pps = e.max_pages_per_slot();
    st.nk = e.rank_nk(0);
    for (const Engine::KvPtrs& p : e.mtp_kv_ptrs(0))
    {
        st.global.push_back(p.global ? 1 : 0);
        st.kv.push_back(p.global ? pull(p.pages, static_cast<size_t>(st.max_pps) * 2 * PAGE * st.nk) : pull(p.ring, static_cast<size_t>(2) * e.c.mtp_window * st.nk));
    }
    st.taps = pull(e.mtp_last_taps(), static_cast<size_t>(e.mtp_taps()) * e.c.H);
    return st;
}

// K (plane 0) or V (plane 1) row of position pos of drafter layer l
static const uint16_t* state_row(const DraftState& st, int l, int plane, int pos, int mw)
{
    const size_t nk = static_cast<size_t>(st.nk);
    if (st.global[static_cast<size_t>(l)]) return &st.kv[static_cast<size_t>(l)][((static_cast<size_t>(pos / PAGE) * 2 + plane) * PAGE + pos % PAGE) * nk];
    return &st.kv[static_cast<size_t>(l)][(static_cast<size_t>(plane) * mw + pos % mw) * nk];
}

// The state of slot 0 of a 1-rank engine into slot 0 of every rank of e: each rank keeps the columns of its KV heads.
static void give_state(Engine& e, const DraftState& st)
{
    e.reset_slot(0);
    const int T = e.o.nr, nh_q = NH / T, gqa = NH / NKV;
    for (int i = 0; i < e.o.nloc; ++i)
    {
        const int q0 = (e.o.rank0 + i) * nh_q, kv0 = q0 / gqa, nk = e.rank_nk(i);
        const std::vector<Engine::KvPtrs> ptrs = e.mtp_kv_ptrs(i);
        for (size_t l = 0; l < ptrs.size(); ++l)
        {
            const size_t rows = st.kv[l].size() / static_cast<size_t>(st.nk);
            std::vector<uint16_t> cut(rows * nk);
            for (size_t r = 0; r < rows; ++r) memcpy(&cut[r * nk], &st.kv[l][r * st.nk + static_cast<size_t>(kv0) * HD], static_cast<size_t>(nk) * 2);
            push(ptrs[l].global ? ptrs[l].pages : ptrs[l].ring, cut);
        }
    }
    push(e.mtp_last_taps(), st.taps);
}

static std::string list(const std::vector<int>& v)
{
    std::string s;
    for (int x : v) s += " " + std::to_string(x);
    return s;
}

// ---------------------------------------------------------------------------------------------- check 1
static void check_draft_block(const Mini& mm, Engine& e1, Engine& e4, Engine& e6, const std::vector<int>& prompt)
{
    const Sampling greedy;
    const int P = static_cast<int>(prompt.size());
    Sess s = prefill(e1, 0, prompt, greedy);
    std::vector<int> d1;
    e1.mtp_draft({0}, {P - 1}, {s.next}, {greedy.draft()}, MK, d1);
    const DraftState st = take_state(e1);

    // the reference, from the device's state, teacher-forced with the device's tokens
    std::vector<float> taps(st.taps.size());
    for (size_t i = 0; i < taps.size(); ++i) taps[i] = ink2ref::bf16_to_f32(st.taps[i]);
    std::vector<float> lg1;   // the 1-rank reference logits [MK][VOCAB]
    int ties = 0;
    for (int T : {1, 4, 6})
    {
        Config c = mm.c;
        c.tp = T;
        ink2ref::Model ref(c, mm.w);
        ref.mtp_set_state(P, taps.data());
        for (int l = 0; l < ML; ++l)
        {
            const int lo = c.mtp_global[static_cast<size_t>(l)] ? 0 : std::max(0, P - MW);
            for (int p = lo; p < P; ++p)
            {
                const uint16_t *k = state_row(st, l, 0, p, MW), *v = state_row(st, l, 1, p, MW);
                for (int j = 0; j < st.nk; ++j)
                {
                    ref.mtp_ctx_k(l)[static_cast<size_t>(p) * st.nk + j] = ink2ref::bf16_to_f32(k[j]);
                    ref.mtp_ctx_v(l)[static_cast<size_t>(p) * st.nk + j] = ink2ref::bf16_to_f32(v[j]);
                }
            }
        }
        std::vector<float> lg;
        const std::function<int(const float*, int)> pick = [&](const float*, int step) { return d1[static_cast<size_t>(step)]; };
        ref.mtp_draft(s.next, MK, &lg, nullptr, &pick);
        for (int k = 0; k < MK; ++k)
        {
            const float* row = &lg[static_cast<size_t>(k) * VOCAB];
            int best = 0;
            for (int t = 1; t < VALID; ++t)
                if (row[t] > row[best]) best = t;
            const int dv = d1[static_cast<size_t>(k)];
            if (dv == best) continue;
            const float gap = row[best] - row[dv], tol = std::max(std::fabs(row[best]), 1.0f) / 32.0f;
            if (dv >= 0 && dv < VALID && gap <= tol)
            {
                ++ties;
                printf("  draft block: reference tp %d step %d: near tie, device %d (%.5f) reference %d (%.5f)\n", T, k, dv, row[dv], best, row[best]);
            }
            else check(false, "draft block: reference tp " + std::to_string(T) + " step " + std::to_string(k) + ": device " + std::to_string(dv) + ", reference " +
                                  std::to_string(best) + " (gap " + std::to_string(gap) + ")");
        }
        if (T == 1) lg1 = lg;
    }
    printf("  draft block on 1 rank:%s (%d near ties against the reference at tp 1, 4, 6)\n", list(d1).c_str(), ties);

    // the same state on 4 and 6 ranks
    for (Engine* e : {&e4, &e6})
    {
        give_state(*e, st);
        std::vector<int> d;
        e->mtp_draft({0}, {P - 1}, {s.next}, {greedy.draft()}, MK, d);
        printf("  draft block on %d ranks:%s\n", e->o.nr, list(d).c_str());
        for (int k = 0; k < MK; ++k)
        {
            const int a = d1[static_cast<size_t>(k)], b = d[static_cast<size_t>(k)];
            if (a == b) continue;
            // the steps after a different token run on a different input: only this step can be judged
            const float* row = &lg1[static_cast<size_t>(k) * VOCAB];
            const float gap = b >= 0 && b < VALID ? std::fabs(row[a] - row[b]) : 1e30f, tol = std::max(std::fabs(row[a]), 1.0f) / 32.0f;
            if (gap <= tol) printf("  draft block: %d ranks step %d: near tie, %d against %d on 1 rank\n", e->o.nr, k, b, a);
            else check(false, "draft block: " + std::to_string(e->o.nr) + " ranks step " + std::to_string(k) + ": " + std::to_string(b) + " against " + std::to_string(a) + " on 1 rank");
            break;
        }
    }
}

// ---------------------------------------------------------------------------------------------- check 2
static void check_generation(Engine& plain, Engine& spec, const std::vector<std::vector<int>>& prompts)
{
    const int T = plain.o.nr;
    for (int mode = 0; mode < 2; ++mode)
    {
        Sampling sp;
        if (mode) sp = Sampling{1.0f / 0.7f, 0.9f, 42};
        const std::string tag = std::to_string(T) + " ranks, " + (mode ? "sampled" : "greedy");
        // the tokens every run must give: each prompt alone, plain steps, the engine without a drafter
        std::vector<std::vector<int>> want;
        for (const std::vector<int>& p : prompts) want.push_back(generate(plain, {p}, 0, sp, GEN)[0]);
        check(generate(plain, prompts, 0, sp, GEN) == want, tag + ": plain steps, 3 sessions in a batch");
        for (size_t i = 0; i < prompts.size(); ++i) check(generate(spec, {prompts[i]}, 0, sp, GEN)[0] == want[i], tag + ": plain steps on the drafter's engine, prompt " + std::to_string(i));
        for (int K : {3, MK})
        {
            for (size_t i = 0; i < prompts.size(); ++i)
            {
                const std::vector<int> got = generate(spec, {prompts[i]}, K, sp, GEN)[0];
                check(got == want[i], tag + ": " + std::to_string(K) + " drafts, prompt " + std::to_string(i) + " alone:" + list(got) + " expected" + list(want[i]));
            }
            const std::vector<std::vector<int>> got = generate(spec, prompts, K, sp, GEN);
            for (size_t i = 0; i < prompts.size(); ++i)
                check(got[i] == want[i], tag + ": " + std::to_string(K) + " drafts, 3 sessions, prompt " + std::to_string(i) + ":" + list(got[i]) + " expected" + list(want[i]));
        }
        printf("  %s: 24 tokens with 0, 3 and 7 drafts, 1 and 3 sessions:%s ...\n", tag.c_str(), list(std::vector<int>(want[0].begin(), want[0].begin() + 8)).c_str());
    }
}

// ---------------------------------------------------------------------------------------------- check 3
// The stores of one rank of the engine (a server has one engine per rank; here every simulated rank gets its own).
struct RankStore
{
    std::unique_ptr<g6s2::KvStateTiers> tiers;
    std::unique_ptr<g6s2::StashPool> stash;
    uint32_t max_pps = 0;
    void drain()
    {
        const auto io = stash->io();
        tiers->run(io, 0);
        stash->io_done();
    }
    std::vector<uint32_t> pages(unsigned q, unsigned pos) const
    {
        std::vector<uint32_t> v(pos / 64u);
        for (unsigned i = 0; i < v.size(); ++i) v[i] = g6s2::pack_page_handle(q, i, max_pps);
        return v;
    }
};

static std::unique_ptr<RankStore> rank_store(Engine& e, int i, const std::string& spec, const std::string& name)
{
    using g6s2::Ink2KvLayer;
    std::vector<Ink2KvLayer> kl, ml;
    for (const auto& p : e.kv_ptrs(i)) kl.push_back(Ink2KvLayer{p.global, p.pages, p.staging, p.ring, p.hist_k, p.hist_v, p.hist_a, p.hist_m});
    for (const auto& p : e.mtp_kv_ptrs(i)) ml.push_back(Ink2KvLayer{p.global, p.pages, p.staging, p.ring, nullptr, nullptr, nullptr, nullptr});
    std::unique_ptr<RankStore> rs(new RankStore);
    rs->max_pps = static_cast<uint32_t>(e.max_pages_per_slot());
    g6s2::KvStateDesc desc = g6s2::ink2_kv_state(kl, static_cast<uint32_t>(e.o.slots), rs->max_pps, static_cast<uint32_t>(e.rank_nk(i)), static_cast<uint32_t>(e.c.H),
                                                 static_cast<uint32_t>(e.c.window), static_cast<uint32_t>(e.max_rank_nk()), ml, static_cast<uint32_t>(e.c.mtp_window),
                                                 e.mtp_last_taps(), static_cast<uint32_t>(e.mtp_taps()));
    rs->tiers.reset(new g6s2::KvStateTiers(desc));
    rs->stash.reset(new g6s2::StashPool(2, 64, desc.state_blocks(), false));
    rs->stash->set_rules(false, true, 0);
    const uint64_t blocks = rs->tiers->add(spec, name, 64ull << 20);
    const std::string compat = "ink2 mtp test rank " + std::to_string(i) + " format " + std::to_string(desc.format_hash());
    rs->stash->add_tier(blocks, 2, nullptr, compat.c_str());
    rs->stash->set_policy(0, 0, 0, desc.slot_bytes());
    rs->stash->tiers_loaded();
    return rs;
}

static void check_stash(Engine& e, const std::vector<int>& prompt, bool ssd, const std::string& tmp)
{
    const int T = e.o.nr, STOP = 70, Q2 = 2;
    const std::string tag = std::to_string(T) + " ranks, " + (ssd ? "file store" : "memory store");
    const Sampling greedy;
    Sess a = prefill(e, 0, prompt, greedy);
    std::vector<Sess*> one{&a};
    while (a.pos < STOP)
    {
        const int k = std::min(MK, STOP - a.pos - 1);
        if (k > 0) spec_round(e, one, k, greedy);
        else plain_step(e, one, greedy);
    }
    check(a.pos == STOP && static_cast<int>(a.fed.size()) == STOP, tag + ": the session did not stop at position 70");

    // capture slot 0 at position 70, restore into slot Q2: every rank's state through its own store
    e.reset_slot(Q2);
    std::vector<std::unique_ptr<RankStore>> stores;
    for (int i = 0; i < e.o.nloc; ++i)
    {
        std::string spec = "mem:";
        if (ssd)
        {
            const std::string dir = tmp + "/store-tp" + std::to_string(T) + "-r" + std::to_string(i);
            std::filesystem::create_directories(dir);
            spec = "ssd:" + dir + "/blocks";
        }
        stores.push_back(rank_store(e, i, spec, "ink2-mtp-test-" + std::to_string(getpid()) + "-tp" + std::to_string(T) + "-r" + std::to_string(i) + (ssd ? "-f" : "-m")));
        RankStore& rs = *stores.back();
        const uint32_t p = static_cast<uint32_t>(STOP);
        e.stage_partial_page(0, p, true, i);
        rs.stash->earn(p / 64u + rs.tiers->desc().state_blocks());
        const uint64_t id = rs.stash->capture(a.fed, static_cast<int32_t>(p), static_cast<int32_t>(p), static_cast<int32_t>(p), rs.pages(0, p), g6s2::pack_state_handle(0, p), 1,
                                              g6rt::CacheNamespace{});
        rs.drain();
        check(id != 0, tag + ": rank " + std::to_string(i) + ": the capture was refused");
        if (!id) return;
        const auto r = rs.stash->restore(rs.stash->info(id).tier, id, rs.pages(Q2, p), g6s2::pack_state_handle(Q2, p));
        rs.drain();
        check(r.valid == static_cast<int32_t>(p), tag + ": rank " + std::to_string(i) + ": restored position " + std::to_string(r.valid));
        e.stage_partial_page(Q2, p, false, i);
    }

    Sess b = a;   // the restored session: the same position and last token, the other slot
    b.slot = Q2;
    const size_t said0 = a.said.size();
    a.drafts.clear();
    b.drafts.clear();
    std::vector<Sess*> two{&b};
    while (static_cast<int>(a.said.size() - said0) < GEN) spec_round(e, one, MK, greedy);
    while (static_cast<int>(b.said.size() - said0) < GEN) spec_round(e, two, MK, greedy);
    check(a.drafts == b.drafts, tag + ": drafts after the restore:" + list(b.drafts) + " uninterrupted" + list(a.drafts));
    check(a.said == b.said, tag + ": tokens after the restore:" + list(b.said) + " uninterrupted" + list(a.said));
    check(a.pos == b.pos, tag + ": position after the restore");
    printf("  %s: restore at position 70, %zu tokens and %zu drafts after it the same\n", tag.c_str(), a.said.size() - said0, a.drafts.size());
}

int main()
{
    knob::init();
    Mini mm;
    build_mini(mm);
    char tmpl[] = "/tmp/ink2-mtp-test.XXXXXX";
    if (!mkdtemp(tmpl)) { perror("mkdtemp"); return 2; }
    const std::string tmp = tmpl, dir = tmp + "/ckpt";
    write_ckpt(mm, dir);

    // prompts: valid token ids; the 3 sessions of check 2 start below position 64 and end past it
    Rng r{0x70726f6d};
    const auto prompt = [&](int n) {
        std::vector<int> v;
        for (int i = 0; i < n; ++i) v.push_back(static_cast<int>(r.next() % VALID));
        return v;
    };
    const std::vector<int> p70 = prompt(70), p50 = prompt(50);
    const std::vector<std::vector<int>> three{prompt(58), prompt(61), prompt(45)};

    std::unique_ptr<Engine> plain[3], spec[3];
    const int tps[3] = {1, 4, 6};
    for (int i = 0; i < 3; ++i)
    {
        plain[i] = engine(dir, tps[i], 0);
        spec[i] = engine(dir, tps[i], MK);
    }
    printf("1. one draft block against the CPU reference and across ranks\n");
    check_draft_block(mm, *spec[0], *spec[1], *spec[2], p70);
    printf("2. generation with and without the drafter\n");
    for (int i = 0; i < 3; ++i) check_generation(*plain[i], *spec[i], three);
    printf("3. state round trip through the KV store\n");
    for (int i = 0; i < 3; ++i)
    {
        check_stash(*spec[i], p50, false, tmp);
        check_stash(*spec[i], p50, true, tmp);
    }
    for (int i = 0; i < 3; ++i)
    {
        plain[i].reset();
        spec[i].reset();
    }
    std::error_code ec;
    std::filesystem::remove_all(std::filesystem::path(tmpl), ec);
    if (fails)
    {
        printf("ink2_mtp_test: %d FAILED\n", fails);
        return 1;
    }
    printf("ink2_mtp_test: PASS\n");
    return 0;
}
