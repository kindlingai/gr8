// CPU four-rank SCHEDULE + DEPENDENCY gate for the Flash SP prefill (G6_PF_RS=2), TP4-SP-MHC-RS-AG-BUILD.md "CPU
// four-rank gate". Answers the design review's caveats on test_flash_sp_layout.cpp (which proves static bytes only):
//  (1) real producer/consumer order: every rank runs the exact SP op sequence of pf_step_flash / pf_step2_flash +
//      pf_consume_flash_sp (engine/runtime/flash_prefill.cuh) on ONE in-order stream, and every rank/lane has a FIFO
//      proxy with fabric.cpp's semantics (job per layer-step, collectives c0 -> c4 in order, spin until doorbell[c] ==
//      coll_seq(step, c, 5) exactly, Rs slice p -> peer's recv[c][me] offset 0 with the tail right after the payload,
//      AG payload + tail from send offset 0, peers in rank order). Each block's output is a function of ALL M input
//      rows (a digest), so any stale / poisoned / not-yet-gathered row read by any consumer changes every later value
//      and is caught against a full-broadcast reference (the OFF semantics: every rank sums all four partials).
//  (2) parity hazards are checked directly, not by timestamps: shadow state per slot flags a send slot repacked
//      before its proxy wired every peer, a receive slot rewritten before its consumer read it, a doorbell rung again
//      before the proxy saw the previous value (fabric.cpp's `*db != seq` spin would hang), and a proxy wiring a slot
//      that holds another step. Schedules: random, one delayed rank, GPU-first (max producer run-ahead), proxy-first.
//  (3) negatives that MUST be detected: dropped AG / RS / c4 wires (exact ERR_AR | c << 8 | p timeout code), c4
//      enqueued on an early layer (proxy doorbell timeout 0x70000004), an AG packed before the RS wait + owner mHC, a
//      skipped AG wait, and a rogue early write into a receive slot.
// No CUDA, no NIC: layout/schedule/dependency only. Arithmetic is test_flash_sp_arith.cpp; real fabric bytes are
// engine/comms/tests/sp4_test.cpp (lead's slot).
// Build: g++ -std=c++17 -O2 -Wall -ffp-contract=off -I engine/runtime engine/tests/test_flash_sp_sched.cpp -o sp_sched_test
#include "pf_layout.h"

#include <cstdio>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

using namespace g6rt;

namespace
{
using u64 = unsigned long long;
constexpr int NR = 4;
constexpr int H = 8;                 // tiny row width, identical layout math
constexpr size_t RB = H * 2, RB4 = H * 4;
constexpr int NC = sp_ncoll;
constexpr unsigned ERR_AR = 0x61000000u, ERR_DB = 0x70000000u;

u64 mix64(u64 x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
float hf(u64 x) { return (float)((int)(mix64(x) % 4001) - 2000) / 1024.f; }
unsigned short to_bf16(float f) { unsigned u; memcpy(&u, &f, 4); return (unsigned short)(u >> 16); }
float bf16f(unsigned short v) { const unsigned u = (unsigned)v << 16; float f; memcpy(&f, &u, 4); return f; }
void put_u64(unsigned char* p, u64 v) { memcpy(p, &v, 8); }
u64 get_u64(const unsigned char* p) { u64 v; memcpy(&v, p, 8); return v; }
u64 digest(const float* x, int M)
{
    u64 h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < (size_t)M * H; ++i) { unsigned u; memcpy(&u, &x[i], 4); h = (h ^ u) * 0x100000001b3ull; }
    return h;
}
// a block (attention b = 0 / MLP b = 1) of rank r, layer i: BF16 partial rows from ALL M input rows
void block(int r, int i, int b, u64 pos0, const float* x, int M, unsigned short* out)
{
    const u64 D = digest(x, M);
    for (int row = 0; row < M; ++row)
        for (int h = 0; h < H; ++h)
            out[(size_t)row * H + h] = to_bf16(x[(size_t)row * H + h] * 0.25f +
                                               hf(D ^ ((u64)r << 60) ^ ((u64)i << 44) ^ ((u64)b << 40) ^ ((pos0 + row) << 8) ^ (u64)h));
}
float kf(int i, int b, int h) { return 0.5f + (float)((i * 7 + b * 3 + h) % 5) / 8.f; }
// owner mHC of one row: rank-ordered FP32 sum of the four BF16 partials (MhcRsBlock), stream update, next input
void mhc_row(const unsigned short* const s[NR], float* state, float* out, int i, int b, bool last, float* resid)
{
    for (int h = 0; h < H; ++h)
    {
        float sum = 0.f;
        for (int p = 0; p < NR; ++p) sum += bf16f(s[p][h]);
        state[h] = state[h] * 0.5f + sum;
        out[h] = state[h] * (last ? 0.75f : kf(i, b, h));
        if (last) resid[h] = state[h];
    }
}
void embed(u64 pos0, int M, float* x)
{
    for (int row = 0; row < M; ++row)
        for (int h = 0; h < H; ++h) x[(size_t)row * H + h] = hf(((pos0 + row) << 8) ^ (u64)h ^ 0xE3ull << 56);
}

// the full-broadcast reference (OFF semantics) of one lane sub-batch
struct Ref
{
    std::vector<std::vector<float>> xn2, xn;    // [layer][M*H]: after the attention / MLP boundary
    std::vector<float> resid;
};
Ref reference(int M, u64 pos0, int nl)
{
    Ref R;
    std::vector<float> x((size_t)M * H), state, xn2((size_t)M * H);
    embed(pos0, M, x.data());
    state = x;
    R.resid.assign((size_t)M * H, 0.f);
    std::vector<std::vector<unsigned short>> part(NR, std::vector<unsigned short>((size_t)M * H));
    for (int i = 0; i < nl; ++i)
        for (int b = 0; b < 2; ++b)
        {
            const bool last = b == 1 && i == nl - 1;
            for (int r = 0; r < NR; ++r) block(r, i, b, pos0, b == 0 ? x.data() : xn2.data(), M, part[r].data());
            std::vector<float>& out = b == 0 ? xn2 : x;
            for (int row = 0; row < M; ++row)
            {
                const unsigned short* s[NR];
                for (int p = 0; p < NR; ++p) s[p] = &part[p][(size_t)row * H];
                mhc_row(s, &state[(size_t)row * H], &out[(size_t)row * H], i, b, last, &R.resid[(size_t)row * H]);
            }
            (b == 0 ? R.xn2 : R.xn).push_back(out);
        }
    return R;
}

// ---------------------------------------------------------------- mutations (negatives) and run configuration
struct Mut
{
    int drop_rank = -1, drop_lane = 0, drop_layer = 0, drop_c = -1;      // proxy never wires (step, c)
    bool c4_early = false;                                              // enqueue c4 on the first layer-step too
    int prepack_rank = -1;                                              // AG packed before RS wait + owner mHC
    int skip_rank = -1, skip_layer = 0, skip_b = 0;                     // no AG wait / copy on that boundary
    int rogue_rank = -1;                                                // early foreign write into a recv slot
};
struct Cfg
{
    int M[2] = {1, 1};
    int lanes = 1, subs = 2, nl = 3;
    int mode = 0;          // 0 random, 1 delayed rank, 2 GPU-first, 3 proxy-first
    int delayed = -1;
    u64 seed = 1;
    Mut mut;
};
struct Res
{
    int hazards = 0, mismatches = 0;
    bool deadlock = false;
    std::vector<unsigned> errors;
    std::string first;
};

struct Job { u64 step; std::vector<uint32_t> sizes; std::vector<uint32_t> rs_bytes; size_t ss; };
struct SendSh { long long step = -1; unsigned wired = 0; };
struct RecvSh { long long step = -1; bool consumed = true; };
struct Fab   // one rank's prefill fabric of one lane + its proxy
{
    int rank = 0;
    size_t stride = 0, ss = 0;
    std::vector<unsigned char> send, recv;
    u64 db[NC] = {}, rung[NC] = {}, passed[NC] = {};
    std::deque<Job> q;
    bool busy = false, seen = false;
    Job cur;
    int c = 0, p = 0;
    SendSh ssh[2][NC];
    RecvSh rsh[2][NC][NR];
    unsigned char* send_slot(u64 step, int cc) { return send.data() + ((step & 1) * NC + cc) * stride; }
    unsigned char* recv_slot(u64 step, int cc, int pp) { return recv.data() + (((step & 1) * NC + cc) * NR + pp) * stride; }
};

enum OpK { ENQ, EMBED, ATTN, MLP, PUT_RS, WAIT_RS, MHC, PUT_AG, WAIT_AG, COPY_AG, CHECK_OUT };
struct Op { OpK k; int lane = 0, layer = 0, b = 0, c = 0, sub = 0; };

struct Lane   // one rank's activations of one lane (the FlashPrefill buffers this gate needs)
{
    int M = 0;
    u64 pos0 = 0, step = 0, ls0 = 1;
    int sub = 0;
    std::vector<float> xn, xn2, state, resid;
    std::vector<unsigned short> part[2];
};

struct World
{
    Cfg cfg;
    Res res;
    size_t MP = 64;
    Fab fab[2][NR];
    Lane L[2][NR];
    std::vector<Ref> ref[2];   // [lane][sub]
    std::vector<Op> ops[NR];
    size_t pc[NR] = {};
    std::vector<std::pair<int, int>> attn_order[NR];   // (layer, lane) in stream order: A before B per layer
    u64 rng = 0;

    void hazard(const std::string& s)
    {
        if (!res.hazards && res.first.empty()) res.first = s;
        ++res.hazards;
    }
    int rows(int lane, int p) const { return pf_rows(cfg.M[lane], p, NR); }
    int first(int lane, int p) const { return pf_r0(cfg.M[lane], p, NR); }
    static u64 pos0_of(int sub, int lane) { return 100000ull * (sub + 1) + 50000ull * lane; }

    // ---------------------------------------------------------------- setup
    void init()
    {
        MP = 64;
        for (int l = 0; l < cfg.lanes; ++l) MP = std::max(MP, (size_t)(cfg.M[l] + 63) / 64 * 64);
        const size_t slot = pf_slot_bytes_rb((int)MP, true, NR, RB);
        CHECK_STRIDE(slot);
        for (int l = 0; l < cfg.lanes; ++l)
            for (int r = 0; r < NR; ++r)
            {
                Fab& F = fab[l][r];
                F.rank = r;
                F.stride = (slot + 8 + 255) & ~size_t(255);
                F.ss = (F.stride / NR) & ~size_t(255);
                F.send.assign(2 * NC * F.stride, 0xA5);
                F.recv.assign(2 * NC * NR * F.stride, 0xA5);
            }
        for (int l = 0; l < cfg.lanes; ++l)
            for (int s = 0; s < cfg.subs; ++s) ref[l].push_back(reference(cfg.M[l], pos0_of(s, l), cfg.nl));
        rng = mix64(cfg.seed * 0x9E37ull + 17);
        for (int r = 0; r < NR; ++r) build_ops(r);
    }
    void CHECK_STRIDE(size_t slot)
    {
        if (slot < (size_t)(MP + NR - 1) / NR * RB4 + 8) hazard("slot smaller than an owner's FP32 AG rows");
    }
    // the per-rank stream: g6m4 enqueues both lanes' jobs, then pf_step_flash / pf_step2_flash, then the full-row
    // consumers (hash / head / draft read P.xn and P.resid) after the stream sync, sub-batch after sub-batch
    void consume(std::vector<Op>& o, int r, int lane, int i, int b)
    {
        const bool last = b == 1 && i == cfg.nl - 1;
        const int cr = 2 * b, ca = cr + 1;
        const Mut& m = cfg.mut;
        const bool prepack = m.prepack_rank == r;
        if (prepack) o.push_back({PUT_AG, lane, i, b, ca});
        o.push_back({WAIT_RS, lane, i, b, cr});
        o.push_back({MHC, lane, i, b, cr});
        if (!prepack) o.push_back({PUT_AG, lane, i, b, ca});
        if (last) o.push_back({PUT_AG, lane, i, b, sp_c_resid_ag});
        const bool skip = m.skip_rank == r && m.skip_layer == i && m.skip_b == b && lane == 0;
        if (!skip)
        {
            o.push_back({WAIT_AG, lane, i, b, ca});
            o.push_back({COPY_AG, lane, i, b, ca});
        }
        if (last)
        {
            o.push_back({WAIT_AG, lane, i, b, sp_c_resid_ag});
            o.push_back({COPY_AG, lane, i, b, sp_c_resid_ag});
        }
    }
    void build_ops(int r)
    {
        std::vector<Op>& o = ops[r];
        const int nl = cfg.nl;
        for (int s = 0; s < cfg.subs; ++s)
        {
            o.push_back({ENQ, 0, 0, 0, 0, s});
            for (int l = 0; l < cfg.lanes; ++l) o.push_back({EMBED, l, 0, 0, 0, s});
            if (cfg.lanes == 1)
                for (int i = 0; i < nl; ++i)
                {
                    o.push_back({ATTN, 0, i}); o.push_back({PUT_RS, 0, i, 0, 0}); consume(o, r, 0, i, 0);
                    o.push_back({MLP, 0, i}); o.push_back({PUT_RS, 0, i, 1, 2}); consume(o, r, 0, i, 1);
                }
            else   // pf_step2_flash
            {
                for (int i = 0; i < nl; ++i)
                {
                    o.push_back({ATTN, 0, i}); o.push_back({PUT_RS, 0, i, 0, 0});
                    if (i > 0) consume(o, r, 1, i - 1, 1);
                    o.push_back({ATTN, 1, i}); o.push_back({PUT_RS, 1, i, 0, 0});
                    consume(o, r, 0, i, 0);
                    o.push_back({MLP, 0, i}); o.push_back({PUT_RS, 0, i, 1, 2});
                    consume(o, r, 1, i, 0);
                    o.push_back({MLP, 1, i}); o.push_back({PUT_RS, 1, i, 1, 2});
                    consume(o, r, 0, i, 1);
                }
                consume(o, r, 1, nl - 1, 1);
            }
            for (int l = 0; l < cfg.lanes; ++l) o.push_back({CHECK_OUT, l, 0, 0, 0, s});
        }
    }

    // ---------------------------------------------------------------- GPU ops
    bool ready(int r)
    {
        if (pc[r] >= ops[r].size()) return false;
        const Op& op = ops[r][pc[r]];
        if (op.k == WAIT_RS) return missing_rs(r, op) < 0;
        if (op.k == WAIT_AG) return missing_ag(r, op) < 0;
        return true;
    }
    int missing_rs(int r, const Op& op)   // k_wait_rs: the first peer whose slice tail for MY rows has not landed
    {
        Fab& F = fab[op.lane][r];
        const u64 step = L[op.lane][r].step, seq = g6::coll_seq(step, op.c, NC);
        for (int p = 0; p < NR; ++p)
            if (p != r && get_u64(F.recv_slot(step, op.c, p) + (size_t)rows(op.lane, r) * RB) != seq) return p;
        return -1;
    }
    int missing_ag(int r, const Op& op)   // k_wait_ag: tail after pf_rows(M, p) FP32 rows of each peer
    {
        Fab& F = fab[op.lane][r];
        const u64 step = L[op.lane][r].step, seq = g6::coll_seq(step, op.c, NC);
        for (int p = 0; p < NR; ++p)
            if (p != r && get_u64(F.recv_slot(step, op.c, p) + (size_t)rows(op.lane, p) * RB4) != seq) return p;
        return -1;
    }
    void ring(Fab& F, int c, u64 seq)   // glue::publish: doorbell[c] = seq (one cell per collective, no parity)
    {
        if (F.rung[c] && F.passed[c] < F.rung[c]) hazard("doorbell " + std::to_string(c) + " rung again before the proxy saw it");
        F.db[c] = seq; F.rung[c] = seq;
    }
    void pack_check(Fab& F, u64 step, int c)
    {
        SendSh& sh = F.ssh[step & 1][c];
        const unsigned all = 0xFu & ~(1u << F.rank);
        if (sh.step >= 0 && sh.wired != all)
            hazard("send slot (parity " + std::to_string(step & 1) + ", c" + std::to_string(c) + ") repacked for step " +
                   std::to_string(step) + " before step " + std::to_string(sh.step) + " was wired to every peer");
        sh.step = (long long)step; sh.wired = 0;
    }
    void enqueue(int r, int sub)
    {
        for (int l = 0; l < cfg.lanes; ++l)
        {
            Lane& A = L[l][r];
            A.M = cfg.M[l];
            A.pos0 = pos0_of(sub, l);
            A.sub = sub;
            if (sub == 0) A.ls0 = 1; else A.ls0 += cfg.nl;
            Fab& F = fab[l][r];
            const int M = A.M;
            for (int i = 0; i < cfg.nl; ++i)   // pf_enqueue_flash_sp
            {
                Job j;
                j.step = A.ls0 + i;
                j.sizes = sp_sizes(M, r, NR, H, i == cfg.nl - 1);
                if (cfg.mut.c4_early && i == 0 && j.sizes.size() == 4) j.sizes.push_back(sp_ag_bytes(M, r, NR, H));
                j.ss = F.ss;
                for (int p = 0; p < NR; ++p) j.rs_bytes.push_back(sp_rs_bytes(M, p, NR, H));
                F.q.push_back(j);
            }
        }
    }
    void step_gpu(int r)
    {
        const Op op = ops[r][pc[r]++];
        if (op.k == ENQ) { enqueue(r, op.sub); return; }
        Lane& A = L[op.lane][r];
        Fab& F = fab[op.lane][r];
        const int M = A.M, me_first = first(op.lane, r), me_rows = rows(op.lane, r);
        switch (op.k)
        {
        case EMBED:
        {
            A.xn.assign(MP * H, 0.f); A.xn2.assign(MP * H, 0.f); A.resid.assign(MP * H, 0.f);
            A.part[0].assign(MP * H, 0); A.part[1].assign(MP * H, 0);
            embed(A.pos0, M, A.xn.data());
            A.state = A.xn;
            const unsigned nan = 0x7FC0DEADu;   // non-owner stream rows are never read on SP: poison them
            for (int row = 0; row < M; ++row)
                if (row < me_first || row >= me_first + me_rows)
                    for (int h = 0; h < H; ++h) memcpy(&A.state[(size_t)row * H + h], &nan, 4);
            break;
        }
        case ATTN:
        case MLP:
        {
            if (op.k == ATTN) { A.step = A.ls0 + op.layer; attn_order[r].push_back({op.layer, op.lane}); }   // k_set_step
            const int b = op.k == ATTN ? 0 : 1;
            block(r, op.layer, b, A.pos0, b == 0 ? A.xn.data() : A.xn2.data(), M, A.part[b].data());
            break;
        }
        case PUT_RS:   // k_put_rs: row of owner p -> slice p at (row - pf_r0(M, p)) * RB, then the doorbell
        {
            pack_check(F, A.step, op.c);
            unsigned char* slot = F.send_slot(A.step, op.c);
            for (int row = 0; row < M; ++row)
            {
                const int p = pf_owner_of(M, row, NR);
                memcpy(slot + (size_t)p * F.ss + (size_t)(row - pf_r0(M, p, NR)) * RB, &A.part[op.b][(size_t)row * H], RB);
            }
            ring(F, op.c, g6::coll_seq(A.step, op.c, NC));
            break;
        }
        case MHC:   // owner rows only: self from the device partial, peers from their slices (MhcRsBlock)
        {
            const bool last = op.b == 1 && op.layer == cfg.nl - 1;
            std::vector<float>& out = op.b == 0 ? A.xn2 : A.xn;
            const unsigned char* self_slice = F.send_slot(A.step, op.c) + (size_t)r * F.ss;
            for (int j = 0; j < me_rows; ++j)
            {
                const int row = me_first + j;
                if (memcmp(self_slice + (size_t)j * RB, &A.part[op.b][(size_t)row * H], RB))
                    hazard("self RS slice differs from the device partial");
                const unsigned short* s[NR];
                for (int p = 0; p < NR; ++p)
                    s[p] = p == r ? &A.part[op.b][(size_t)row * H]
                                  : reinterpret_cast<const unsigned short*>(F.recv_slot(A.step, op.c, p) + (size_t)j * RB);
                mhc_row(s, &A.state[(size_t)row * H], &out[(size_t)row * H], op.layer, op.b, last, &A.resid[(size_t)row * H]);
            }
            for (int p = 0; p < NR; ++p)
                if (p != r) consumed(F, A.step, op.c, p);
            break;
        }
        case PUT_AG:   // owner FP32 rows -> send offset 0, then the doorbell (also with zero rows)
        {
            const std::vector<float>& src = op.c == sp_c_resid_ag ? A.resid : op.c == sp_c_attn_ag ? A.xn2 : A.xn;
            pack_check(F, A.step, op.c);
            if (me_rows) memcpy(F.send_slot(A.step, op.c), &src[(size_t)me_first * H], (size_t)me_rows * RB4);
            ring(F, op.c, g6::coll_seq(A.step, op.c, NC));
            break;
        }
        case WAIT_RS:
        case WAIT_AG:
            break;   // ready() held it until every tail landed
        case COPY_AG:   // k_ag_copy: peer p's rows -> global rows pf_r0(M, p), then the full-row check
        {
            std::vector<float>& dst = op.c == sp_c_resid_ag ? A.resid : op.c == sp_c_attn_ag ? A.xn2 : A.xn;
            for (int p = 0; p < NR; ++p)
            {
                if (p == r) continue;
                if (rows(op.lane, p)) memcpy(&dst[(size_t)first(op.lane, p) * H], F.recv_slot(A.step, op.c, p), (size_t)rows(op.lane, p) * RB4);
                consumed(F, A.step, op.c, p);
            }
            if (op.c != sp_c_resid_ag)
            {
                const Ref& R = ref[op.lane][A.sub];
                const std::vector<float>& want = op.c == sp_c_attn_ag ? R.xn2[op.layer] : R.xn[op.layer];
                if (memcmp(dst.data(), want.data(), (size_t)M * RB4)) mismatch("gathered rows", r, op);
            }
            break;
        }
        case CHECK_OUT:   // pf_hash_flash / pf_head_flash / pf_draft_flash read every row of P.xn and P.resid
            if (memcmp(A.xn.data(), ref[op.lane][A.sub].xn.back().data(), (size_t)M * RB4)) mismatch("final norm rows", r, op);
            if (memcmp(A.resid.data(), ref[op.lane][A.sub].resid.data(), (size_t)M * RB4)) mismatch("final residual rows", r, op);
            break;
        default: break;
        }
    }
    void mismatch(const char* what, int r, const Op& op)
    {
        if (!res.mismatches && res.first.empty())
            res.first = std::string(what) + " differ from the reference: rank " + std::to_string(r) + " lane " +
                        std::to_string(op.lane) + " layer " + std::to_string(op.layer) + " c" + std::to_string(op.c);
        ++res.mismatches;
    }
    void consumed(Fab& F, u64 step, int c, int p)
    {
        RecvSh& sh = F.rsh[step & 1][c][p];
        if (sh.step != (long long)step) hazard("consumer read a receive slot holding another step");
        sh.consumed = true;
    }

    // ---------------------------------------------------------------- proxies (fabric.cpp, Rs form, uniform slots)
    bool proxy_ready(int l, int r)
    {
        Fab& F = fab[l][r];
        if (!F.busy) return !F.q.empty();
        if (F.c >= (int)F.cur.sizes.size()) return true;
        if (!F.seen) return F.db[F.c] == g6::coll_seq(F.cur.step, F.c, NC);
        return true;
    }
    void step_proxy(int l, int r)
    {
        Fab& F = fab[l][r];
        if (!F.busy) { F.cur = F.q.front(); F.q.pop_front(); F.busy = true; F.c = 0; F.p = 0; F.seen = false; return; }
        Job& j = F.cur;
        if (F.c >= (int)j.sizes.size()) { F.busy = false; return; }
        const u64 seq = g6::coll_seq(j.step, F.c, NC);
        if (!F.seen) { F.seen = true; F.passed[F.c] = seq; return; }
        while (F.p < NR && F.p == r) ++F.p;
        if (F.p >= NR) { ++F.c; F.p = 0; F.seen = false; return; }
        const int p = F.p++;
        const Mut& m = cfg.mut;
        if (m.drop_rank == r && m.drop_lane == l && m.drop_c == F.c && j.step == L[l][r].ls0 + m.drop_layer) return;
        const bool rs = (sp_rs_mask >> F.c) & 1u;
        const uint32_t nb = rs ? j.rs_bytes[p] : j.sizes[F.c];
        unsigned char* src = F.send_slot(j.step, F.c) + (rs ? (size_t)p * j.ss : 0);
        SendSh& sh = F.ssh[j.step & 1][F.c];
        if (sh.step != (long long)j.step) hazard("proxy wired a send slot holding step " + std::to_string(sh.step) + " as step " + std::to_string(j.step));
        put_u64(src + nb, seq);
        Fab& P = fab[l][p];
        RecvSh& rh = P.rsh[j.step & 1][F.c][r];
        if (!rh.consumed) hazard("receive slot (parity " + std::to_string(j.step & 1) + ", c" + std::to_string(F.c) + ") of rank " +
                                 std::to_string(p) + " rewritten before its consumer read step " + std::to_string(rh.step));
        if (nb + 8 > P.stride) hazard("payload + tail exceed the slot");
        memcpy(P.recv_slot(j.step, F.c, r), src, nb + 8);
        rh.step = (long long)j.step; rh.consumed = false;
        sh.wired |= 1u << p;
    }
    // a rogue early write (negative only): a peer's step + 2 data lands in a slot before its consumer read it
    void rogue(int r)
    {
        Fab& F = fab[0][r];
        const int p = (r + 1) % NR, c = sp_c_attn_ag;
        RecvSh& rh = F.rsh[L[0][r].step & 1][c][p];
        if (rh.step >= 0 && !rh.consumed)
        {
            hazard("rogue: receive slot rewritten before its consumer read it");
            memset(F.recv_slot((u64)rh.step, c, p), 0x3C, RB4);
            cfg.mut.rogue_rank = -1;
        }
    }

    // ---------------------------------------------------------------- scheduler
    u64 rnd() { rng = mix64(rng); return rng; }
    void run()
    {
        init();
        struct Act { int kind, lane, rank; };
        for (long iter = 0; iter < 50000000; ++iter)
        {
            if (cfg.mut.rogue_rank >= 0) rogue(cfg.mut.rogue_rank);
            std::vector<Act> act;
            for (int r = 0; r < NR; ++r)
                if (ready(r)) act.push_back({0, 0, r});
            for (int l = 0; l < cfg.lanes; ++l)
                for (int r = 0; r < NR; ++r)
                    if (proxy_ready(l, r)) act.push_back({1, l, r});
            if (act.empty())
            {
                bool done = true;
                for (int r = 0; r < NR; ++r) done &= pc[r] >= ops[r].size();
                for (int l = 0; l < cfg.lanes; ++l)
                    for (int r = 0; r < NR; ++r) done &= !fab[l][r].busy && fab[l][r].q.empty();
                if (!done) timeouts();
                break;
            }
            std::vector<Act> pick;
            if (cfg.mode == 1)
            {
                for (const Act& a : act) if (a.rank != cfg.delayed) pick.push_back(a);
                if (pick.empty()) pick = act;
            }
            else if (cfg.mode == 2 || cfg.mode == 3)
            {
                for (const Act& a : act) if (a.kind == (cfg.mode == 2 ? 0 : 1)) pick.push_back(a);
                if (pick.empty()) pick = act;
            }
            else pick = act;
            const Act a = pick[rnd() % pick.size()];
            if (a.kind == 0) step_gpu(a.rank); else step_proxy(a.lane, a.rank);
        }
        for (int r = 0; r < NR; ++r)   // pf_step2_flash cache order: A's attention before B's in every layer
            for (size_t k = 1; k < attn_order[r].size(); ++k)
                if (attn_order[r][k].first == attn_order[r][k - 1].first && attn_order[r][k].second < attn_order[r][k - 1].second)
                    hazard("lane B attention before lane A in one layer");
    }
    // a stall: what each spinning party reports (k_wait_rs / k_wait_ag: ERR_AR | c << 8 | p; proxy: 0x70000000 | c)
    void timeouts()
    {
        res.deadlock = true;
        for (int r = 0; r < NR; ++r)
        {
            if (pc[r] >= ops[r].size()) continue;
            const Op& op = ops[r][pc[r]];
            const int p = op.k == WAIT_RS ? missing_rs(r, op) : op.k == WAIT_AG ? missing_ag(r, op) : -1;
            if (p >= 0) res.errors.push_back(ERR_AR | ((unsigned)op.c << 8) | (unsigned)p);
        }
        for (int l = 0; l < cfg.lanes; ++l)
            for (int r = 0; r < NR; ++r)
            {
                const Fab& F = fab[l][r];
                if (F.busy && F.c < (int)F.cur.sizes.size() && !F.seen) res.errors.push_back(ERR_DB | (unsigned)F.c);
            }
    }
};

int fails = 0;
Res run(const Cfg& c)
{
    World* w = new World();
    w->cfg = c;
    w->run();
    Res r = w->res;
    delete w;
    return r;
}
bool has(const Res& r, unsigned code)
{
    for (unsigned e : r.errors) if (e == code) return true;
    return false;
}
void expect_clean(const Cfg& c, const char* tag)
{
    const Res r = run(c);
    if (r.hazards || r.mismatches || r.deadlock)
    {
        fprintf(stderr, "FAIL %s M %d/%d lanes %d mode %d delayed %d seed %llu: hazards %d mismatches %d deadlock %d (%s)\n", tag,
                c.M[0], c.M[1], c.lanes, c.mode, c.delayed, c.seed, r.hazards, r.mismatches, (int)r.deadlock, r.first.c_str());
        ++fails;
    }
}
}   // namespace

int main()
{
    int runs = 0;
    // positives: every M of the build spec, one and two lanes (uneven B), two sub-batches, four schedule modes
    const int Ms[] = {1, 2, 3, 4, 5, 63, 64, 8191, 8192};
    for (int M : Ms)
        for (int lanes = 1; lanes <= 2; ++lanes)
            for (int mode = 0; mode < 4; ++mode)
            {
                const int seeds = M >= 8191 ? 1 : 4;
                for (int s = 0; s < seeds; ++s)
                {
                    Cfg c;
                    c.M[0] = M; c.M[1] = M > 5 ? M - 3 : (M % 3) + 1;
                    c.lanes = lanes; c.mode = mode; c.delayed = (s + mode) % NR; c.seed = 1000 * M + 10 * mode + s + 100000 * lanes;
                    expect_clean(c, "positive");
                    ++runs;
                }
            }
    // negatives: each must be DETECTED (a timeout with the exact code, a hazard, or a row mismatch)
    auto neg = [&](const char* tag, Cfg c, auto pred) {
        const Res r = run(c);
        if (!pred(r))
        {
            fprintf(stderr, "FAIL negative %s not detected: hazards %d mismatches %d deadlock %d errors", tag, r.hazards, r.mismatches, (int)r.deadlock);
            for (unsigned e : r.errors) fprintf(stderr, " %08x", e);
            fprintf(stderr, "\n");
            ++fails;
        }
        else printf("negative %-34s detected: hazards %d mismatches %d deadlock %d first-error %08x%s%s\n", tag, r.hazards, r.mismatches,
                    (int)r.deadlock, r.errors.empty() ? 0u : r.errors[0], r.first.empty() ? "" : " | ", r.first.c_str());
        ++runs;
    };
    for (int M : {3, 64, 8191})
        for (int lanes = 1; lanes <= 2; ++lanes)
        {
            Cfg b;
            b.M[0] = M; b.M[1] = M; b.lanes = lanes; b.seed = 77 + M;
            Cfg c = b; c.mut.drop_rank = 2; c.mut.drop_layer = 1; c.mut.drop_c = sp_c_attn_ag;   // AG c1 of rank 2 lost
            neg("drop AG c1 wire of rank 2", c, [](const Res& r) { return r.deadlock && has(r, ERR_AR | (1u << 8) | 2u); });
            c = b; c.mut.drop_rank = 1; c.mut.drop_layer = 2; c.mut.drop_c = sp_c_mlp_ag;          // AG c3
            neg("drop AG c3 wire of rank 1", c, [](const Res& r) { return r.deadlock && has(r, ERR_AR | (3u << 8) | 1u); });
            c = b; c.mut.drop_rank = 3; c.mut.drop_layer = 0; c.mut.drop_c = sp_c_mlp_rs;          // RS c2
            neg("drop RS c2 wire of rank 3", c, [](const Res& r) { return r.deadlock && has(r, ERR_AR | (2u << 8) | 3u); });
            c = b; c.mut.drop_rank = 0; c.mut.drop_layer = 2; c.mut.drop_c = sp_c_resid_ag;        // final c4
            neg("drop final c4 wire of rank 0", c, [](const Res& r) { return r.deadlock && has(r, ERR_AR | (4u << 8) | 0u); });
            c = b; c.mut.c4_early = true;                                                          // c4 on layer 0
            neg("c4 enqueued on a non-last layer", c, [](const Res& r) { return r.deadlock && has(r, ERR_DB | 4u); });
            c = b; c.mut.prepack_rank = 1;                                                         // caveat (1)
            neg("AG packed before RS wait + owner mHC", c, [](const Res& r) { return r.mismatches > 0; });
            c = b; c.mut.skip_rank = 0; c.mut.skip_layer = 1; c.mut.skip_b = 0; c.mode = 2;        // consumer skips AG
            neg("AG wait + copy skipped on rank 0", c, [](const Res& r) { return r.mismatches > 0 || r.hazards > 0 || r.deadlock; });
            c = b; c.mut.rogue_rank = 2;                                                           // caveat (2)
            neg("rogue early receive-slot write", c, [](const Res& r) { return r.hazards > 0; });
        }
    if (fails) { fprintf(stderr, "%d failures in %d runs\n", fails, runs); return 1; }
    printf("sp_sched_test: OK (%d runs)\n", runs);
    return 0;
}
