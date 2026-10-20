// Remote attention rows (G6_RT_STORE): a rank's latent rows past L (default the CP map's R) live in a rows store
// (kvman-rs src/rows.rs: this process's memory, or a lease on a KV node), and each indexer group keeps a pool of
// resident slots that CP decode reads them from (engine/attn/rt.cuh, attn.h G6RtGroup).
//
// A whole page past L that this rank holds goes to the store once every row of it is committed (spill, between steps),
// and its physical page is flagged. Attention reads a flagged page's rows through the slots; the local page stays
// allocated and valid, so a row the pool had no room for reads it, and dropping the flag (drop) is always safe. The
// page tables are never changed: S2's table rewrites (mode switches, binds, growth) and the index keys are as without
// the tier. Only the main model's CP bindings carry the flags: the MTP layer, the DFlash2 drafter and the staged
// prefill views never read the slots.
//
// Release (local > 0, G6_RT_LOCAL): each session's tail pages past L on this rank come from its window of `local`
// physical pages, taken as rows reach them (ensure) and given back once spilled (release). In g6m4 the S2 pool grants
// the window with the run (kvman-rs Pool::window) and the drafter's tail apart from it (s2_cp_map). A released logical page's table entry
// becomes page_count + v, where v (0..V-1) names its records in the store; its rows read only from the slots (attention
// error 0x40000 when one has none), its index keys move to the group's arena (G6Attn::rt_ix), and a prefill gathers it
// from the store (rt_pf_stage). The pool then holds the pages between R and L, plus the tail pages not yet whole: a
// context past L needs `local` >= the pages a prefill chunk or step spans per rank, plus one.
//
// Store layout, one store per rank: the layout header (G6KV_ROWS_HEADER bytes), then group after group in layer order,
// one record per row (page * 64 + pos % 64: the physical page, or under release the released id v) of this rank's
// attention cache. A record is the row of every layer of the group, in layer order, g6kv::ROW bytes each. A group is a
// full-index layer and the shared-index layers that reuse its list.
//
// G6_RT_FMT=int4 (int4 here): a record holds each row in the INT4 layout (kvfmt.h int4row, 416 B for 656), so a lease
// holds 0.63x the bytes and a fetch moves 0.63x; the pages and the slots stay FP8, so no attention kernel changes. The
// spill packs each page's rows on the GPU (attn.cu g6_rt_pack_int4); the helper reads records into a stage at the
// slot's place, and after the wait rt_unpack (attn/rt.cuh) requantizes them into the slot as append_kv's FP8 path
// would. The rope key goes through exactly. check (G6_RT_CHECK=1): a round trip of random rows at build, and of every
// spilled row, against the FP8 rows (latent SQNR, rope bytes), printed by rt_check_print. A released page's records
// are packed the same way, and a prefill turns them back into FP8 rows on the GPU (rt_pf_stage).
#pragma once
#include "rank.cuh"
#include "../kvman-rs/g6kvman.h"
#include <cuda_fp8.h>
#include <cfloat>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace g6rt {

struct RtTier
{
    struct Group
    {
        std::vector<int> layers;      // positions in Model::layers, the full-index layer first
        G6RtGroup h{};                // host copy of the device struct (its addresses)
        G6RtGroup* dev = nullptr;
        uint8_t* slots = nullptr;     // mapped pinned [slots][layers][g6kv::ROW]
        uint8_t* stage = nullptr;     // int4: mapped pinned [slots][rec], what the helper reads into
        void* req = nullptr;          // mapped pinned ring
        uint64_t* done = nullptr;     // mapped pinned
        uint8_t* ix = nullptr;        // release: [V] pages of the full-index layer's index keys (device)
    };
    int L = 0, S = 0, nr = 0, rank = 0, pages = 0;
    int V = 0, local = 0;             // release: ids for released pages, window pages per session (0 = no release)
    bool poison = false, int4 = false, check = false;
    uint8_t *pack = nullptr, *back = nullptr;   // int4: device [64][rec] for the spill; check: [64][rec of FP8] back
    struct Check { double sig = 0, err = 0; uint64_t rows = 0, rope_bad = 0; } spill_ck;
    g6kv_rows* store = nullptr;
    std::vector<Group> g;
    std::vector<std::pair<int, int>> at;   // [layer] its group and place in it
    uint8_t* d_flag = nullptr;        // [pages + V] device: the page is in the store
    std::vector<int> owner;           // [pages + V] the session slot whose spill flagged it, -1 = not flagged
    int* d_list = nullptr;            // [pages + V] device scratch: the pages a drop clears
    uint64_t spilled = 0, dropped = 0;
    // release only, per session slot: its run's list (to find it from its CP map), its tail entries (logical page past
    // L this rank holds -> its local page, page_count + id, or -1), its window's free pages, local pages in use now and
    // at most, pages released. `write` puts slot q's tables, with its tail entries, where attention reads them (the
    // driver: the layers' tables, the S2 slot table, or both).
    struct Seq
    {
        bool on = false;
        std::vector<int> list, tail, pfree;
        int window = 0, in_use = 0, peak = 0;
        uint64_t released = 0;
    };
    Seq seq[8];
    std::function<void(int q)> write;
    int tsize = 0;                    // the table entries (logical pages) of a binding
    std::vector<int> vfree, vlp, vslot;   // free ids; each id's logical page and slot (-1 = free)
    int peak = 0;                     // the most local tail pages of one session at once, over all sessions
    uint64_t released = 0;
    bool hold = false;                // release with G6_RT_CHECK=1: rt_check after every change (one holder per record)
    uint64_t readback = 0, readback_bad = 0;   // with hold: released pages' records read back after the write, and how many differed
    // with hold: each released id's FP8 rows of every main layer as they were ([layer][64][g6kv::ROW]), and how the rows a
    // prefill staged from the store compare with them (fp8 records: exact; int4: about 20 dB, the rope exact)
    std::map<int, std::vector<uint8_t>> orig;
    Check stage_ck;
    uint64_t stage_bad = 0;           // staged pages under 10 dB or with a rope byte off
    uint8_t* stage = nullptr;         // pinned: one layer's rows of the released pages a prefill gathers
    size_t stage_bytes = 0;
    ~RtTier() { if (store) g6kv_rows_free(store); }
};

// G6_RT_FMT: whether the records hold INT4 rows (unset or fp8: the cache's own row)
inline bool rt_fmt_int4()
{
    const char* f = getenv("G6_RT_FMT");
    if (f && strcmp(f, "fp8") && strcmp(f, "int4")) die("G6_RT_FMT must be fp8 or int4");
    const bool int4 = f && !strcmp(f, "int4");
    if (int4 && G6_KV4) die("G6_RT_FMT=int4: this engine's rows are INT4 already (G6_KV4)");
    return int4;
}
// what the lease name and the layout header add for INT4 records; fp8 records keep the names they had before
inline std::string rt_fmt_name(bool int4) { return int4 ? "-int4" : ""; }
inline std::string rt_fmt_layout(bool int4) { return int4 ? " fmt=int4" : ""; }

// FP8 rows a against b (the same rows after a round trip): latent signal and error energy, rope bytes that differ
inline void rt_cmp(const uint8_t* a, const uint8_t* b, size_t n, RtTier::Check& c)
{
    namespace f8 = g6kv::fp8row;
    auto dec = [](const uint8_t* r, int d) { __nv_fp8_e4m3 x; x.__x = r[d]; return (double)(float)x * ((const float*)(r + f8::SCALE_OFF))[d / 128]; };
    for (size_t i = 0; i < n; ++i, a += f8::ROW, b += f8::ROW)
    {
        for (int d = 0; d < 512; ++d) { const double x = dec(a, d), y = dec(b, d); c.sig += x * x; c.err += (x - y) * (x - y); }
        c.rope_bad += memcmp(a + f8::ROPE_OFF, b + f8::ROPE_OFF, f8::ROW - f8::ROPE_OFF) != 0;
        ++c.rows;
    }
}
inline void rt_check_print(const char* what, const RtTier::Check& c)
{
    fprintf(stderr, "# G6_RT_CHECK %s: %llu rows FP8 -> INT4 -> FP8, latent SQNR %.2f dB, rope differs in %llu\n", what,
            (unsigned long long)c.rows, 10 * log10(c.sig / std::max(c.err, 1e-300)), (unsigned long long)c.rope_bad);
}
// check: n random FP8 rows (Gaussian latent, a spread of channel scales as the real latent has, FP8 as append_kv
// quantizes; random rope bits) through the round trip
inline void rt_selftest(int n, cudaStream_t st)
{
    namespace f8 = g6kv::fp8row;
    std::vector<uint8_t> h((size_t)n * f8::ROW), r(h.size());
    uint64_t z = 0x9e3779b97f4a7c15ull;
    auto u = [&] { z ^= z << 13; z ^= z >> 7; z ^= z << 17; return (double)(z >> 11) / 9007199254740992.0; };
    std::vector<float> ch(512), v(512);
    for (float& c : ch) c = (float)exp(3 * (u() - 0.5));
    for (int i = 0; i < n; ++i)
    {
        uint8_t* row = h.data() + (size_t)i * f8::ROW;
        for (int d = 0; d < 512; ++d) v[d] = ch[d] * (float)(sqrt(-2 * log(u() + 1e-300)) * cos(6.283185307179586 * u()));
        for (int g = 0; g < 4; ++g)
        {
            float mx = 0;
            for (int d = g * 128; d < (g + 1) * 128; ++d) mx = std::max(mx, fabsf(v[d]));
            const float sc = std::max(mx / 448.f, FLT_MIN);
            ((float*)(row + f8::SCALE_OFF))[g] = sc;
            for (int d = g * 128; d < (g + 1) * 128; ++d) row[d] = __nv_fp8_e4m3(v[d] / sc).__x;
        }
        for (int b = f8::ROPE_OFF; b < f8::ROW; ++b) row[b] = (uint8_t)(u() * 256);
    }
    uint8_t *d8 = dmalloc<uint8_t>(h.size()), *d4 = dmalloc<uint8_t>((size_t)n * g6kv::int4row::ROW);
    G6RT_CK(cudaMemcpy(d8, h.data(), h.size(), cudaMemcpyHostToDevice));
    G6RT_CK((cudaError_t)g6_rt_pack_int4(d8, f8::ROW, d4, g6kv::int4row::ROW, n, st));
    G6RT_CK(cudaMemsetAsync(d8, 0, h.size(), st));
    G6RT_CK((cudaError_t)g6_rt_unpack_int4(d4, g6kv::int4row::ROW, d8, f8::ROW, n, st));
    G6RT_CK(cudaMemcpyAsync(r.data(), d8, r.size(), cudaMemcpyDeviceToHost, st));
    G6RT_CK(cudaStreamSynchronize(st));
    RtTier::Check c;
    rt_cmp(h.data(), r.data(), n, c);
    rt_check_print("random rows", c);
    G6RT_CK(cudaFree(d8));
    G6RT_CK(cudaFree(d4));
}

// grid (pages), block 64: page list[x]'s rows leave their slots (the slot is free, the row absent)
__global__ void __launch_bounds__(64) k_rt_drop(const int* list, int* pos2slot, int* slot2pos)
{
    const int k = list[blockIdx.x] * 64 + threadIdx.x, s = pos2slot[k];
    if (s >= 0) slot2pos[s] = -1;
    pos2slot[k] = -1;
}

// the flags of `list` off and their rows out of the slots. Between steps (synchronizes st).
inline void rt_drop(RtTier& t, const std::vector<int>& list, cudaStream_t st)
{
    std::vector<int> v;
    for (int p : list)
        if (p >= 0 && p < t.pages + t.V && t.owner[p] >= 0) v.push_back(p);
    if (v.empty()) return;
    G6RT_CK(cudaStreamSynchronize(st));
    G6RT_CK(cudaMemcpy(t.d_list, v.data(), v.size() * sizeof(int), cudaMemcpyHostToDevice));
    for (RtTier::Group& G : t.g) k_rt_drop<<<(unsigned)v.size(), 64, 0, st>>>(t.d_list, G.h.pos2slot, G.h.slot2pos);
    G6RT_CK(cudaGetLastError());
    for (int p : v) { G6RT_CK(cudaMemsetAsync(t.d_flag + p, 0, 1, st)); t.owner[p] = -1; }
    G6RT_CK(cudaStreamSynchronize(st));
    t.dropped += v.size();
}
// every page session slot q flagged
inline void rt_drop_slot(RtTier& t, int q, cudaStream_t st)
{
    std::vector<int> v;
    for (int p = 0; p < t.pages + t.V; ++p) if (t.owner[p] == q) v.push_back(p);
    rt_drop(t, v, st);
}

// what one spill call saw of the pages past L below `committed`
struct RtSpillSeen
{
    int pages = 0;                    // logical pages past L the session has committed
    int other = 0;                    // another rank's (or no) physical page
    int range = 0;                    // a physical page past this rank's pool
    int owned = 0;                    // already in the store
    int spilled = 0;
};
// release: a logical page past L that this rank holds (the CP map's rule)
inline bool rt_mine(const RtTier& t, int lp) { return lp >= t.L / 64 && lp < t.tsize && lp % t.nr == t.rank; }
// release, G6_RT_CHECK=1: every record has exactly one holder (its slot's tail entry for its logical page), every tail
// entry past the physical pages names a record its slot holds, and the free ids are the rest, each once
inline void rt_check(const RtTier& t, const char* where)
{
    if (!t.hold) return;
    auto bad = [&](const std::string& why) { die(std::string("G6_RT_CHECK after ") + where + ": " + why); };
    std::vector<int> seen(t.V, 0);
    for (int v : t.vfree)
    {
        if (v < 0 || v >= t.V || seen[v]++) bad("id " + std::to_string(v) + " free twice or out of range");
        if (t.vlp[v] >= 0) bad("free id " + std::to_string(v) + " still names a page");
    }
    for (int q = 0; q < 8; ++q)
    {
        const RtTier::Seq& s = t.seq[q];
        for (int lp = 0; lp < (int)s.tail.size(); ++lp)
        {
            const int e = s.tail[lp];
            if (e < t.pages) continue;
            const int v = e - t.pages;
            if (!s.on || v >= t.V || t.vlp[v] != lp || t.vslot[v] != q || seen[v]++)
                bad("slot " + std::to_string(q) + " page " + std::to_string(lp) + " -> id " + std::to_string(v) + " held twice or not its own");
        }
    }
    for (int v = 0; v < t.V; ++v)
        if (!seen[v]) bad("id " + std::to_string(v) + " has no holder and is not free");
}
// release: slot q's records back to the free ids, their rows out of the slots. Between steps.
inline void rt_end(RtTier& t, int q, cudaStream_t st)
{
    RtTier::Seq& s = t.seq[q];
    if (!s.on) return;
    std::vector<int> ids;
    for (int v = 0; v < t.V; ++v)
        if (t.vslot[v] == q) ids.push_back(t.pages + v);
    rt_drop(t, ids, st);
    for (int k : ids) { const int v = k - t.pages; t.vlp[v] = t.vslot[v] = -1; t.vfree.push_back(v); t.orig.erase(v); }
    s = RtTier::Seq{};
    rt_check(t, "end");
}
// release: slot q starts a run (list: its pages, to find it by; window: the local pages its tail cycles through), its
// tail pages past L unmapped (-1). The caller then writes its tables. Between steps.
inline void rt_begin(RtTier& t, int q, const std::vector<int>& window, const std::vector<int>& list, cudaStream_t st)
{
    rt_end(t, q, st);
    if ((int)window.size() < t.local) die("G6_RT_LOCAL: slot " + std::to_string(q) + " has " + std::to_string(window.size()) + " window pages, wants " + std::to_string(t.local));
    RtTier::Seq& s = t.seq[q];
    s.on = true;
    s.list = list;
    s.window = (int)window.size();
    s.tail.assign(t.tsize, -1);
    s.pfree.assign(window.rbegin(), window.rend());   // the first window page is taken first (pop_back)
    for (int p : window) if (p < 0 || p >= t.pages) die("G6_RT_LOCAL: window page " + std::to_string(p) + " is not a local page");
}
// release: the tail entries of the slot whose run has this list (S2Model::rt_tail), or null
inline const std::vector<int>* rt_tail_of(const RtTier& t, const std::vector<int>& list)
{
    for (const RtTier::Seq& s : t.seq)
        if (s.on && s.list == list) return &s.tail;
    return nullptr;
}
// release: a window page for every page of slot q's past L with rows below `end` that has none yet, cleared as reset()
// clears a new sequence's. Between steps, before the step or prefill chunk that writes those rows.
inline void rt_ensure(Model& m, RtTier& t, int q, int end, cudaStream_t st)
{
    if (!t.local) return;
    RtTier::Seq& s = t.seq[q];
    if (!s.on) return;
    bool any = false;
    for (int lp = t.L / 64; lp < std::min((end + 63) / 64, t.tsize); ++lp)
    {
        if (!rt_mine(t, lp) || s.tail[lp] != -1) continue;
        if (s.pfree.empty())
            die("G6_RT_LOCAL: rank " + std::to_string(t.rank) + " slot " + std::to_string(q) + " needs a page for logical page " + std::to_string(lp) +
                " and all " + std::to_string(s.window) + " window pages hold rows not yet whole (raise G6_RT_LOCAL)");
        if (!any) G6RT_CK(cudaStreamSynchronize(st));
        any = true;
        const int p = s.pfree.back();
        s.pfree.pop_back();
        for (const Layer& y : m.layers)
        {
            const G6Attn& a = y.ab.args;
            G6RT_CK(cudaMemset((uint8_t*)a.mla_cache + (size_t)p * 64 * g6kv::ROW, 0, 64 * g6kv::ROW));
            if (a.index_cache) G6RT_CK(cudaMemset((uint8_t*)a.index_cache + (size_t)p * 64 * g6kv::IDX_ROW, 0, 64 * g6kv::IDX_ROW));
        }
        s.tail[lp] = p;
        s.peak = std::max(s.peak, ++s.in_use);
        t.peak = std::max(t.peak, s.peak);
    }
    if (!any) return;
    if (t.write) t.write(q);
    rt_check(t, "ensure");
}
// group G's records of physical page p into buf (64 records; G6_RT_FMT=int4: packed on the GPU, and under check
// compared with the FP8 rows after a round trip)
inline void rt_page_record(Model& m, RtTier& t, const RtTier::Group& G, int p, std::vector<uint8_t>& buf, cudaStream_t st)
{
    const size_t rec = G.h.rec, srec = G.layers.size() * g6kv::ROW;
    buf.resize(64 * rec);
    if (t.int4)
    {
        for (size_t j = 0; j < G.layers.size(); ++j)
        {
            const G6Attn& a = m.layers[G.layers[j]].ab.args;
            G6RT_CK((cudaError_t)g6_rt_pack_int4((const uint8_t*)a.mla_cache + (size_t)p * 64 * g6kv::ROW, g6kv::ROW,
                                                 t.pack + j * g6kv::int4row::ROW, rec, 64, st));
        }
        G6RT_CK(cudaMemcpyAsync(buf.data(), t.pack, 64 * rec, cudaMemcpyDeviceToHost, st));
        if (t.check)
        {
            G6RT_CK((cudaError_t)g6_rt_unpack_int4(t.pack, g6kv::int4row::ROW, t.back, g6kv::ROW, 64 * (int)G.layers.size(), st));
            std::vector<uint8_t> was(64 * srec), back(64 * srec);
            for (size_t j = 0; j < G.layers.size(); ++j)   // [pos][layer], as the packed records
            {
                const G6Attn& a = m.layers[G.layers[j]].ab.args;
                G6RT_CK(cudaMemcpy2DAsync(was.data() + j * g6kv::ROW, srec, (const uint8_t*)a.mla_cache + (size_t)p * 64 * g6kv::ROW,
                                          g6kv::ROW, g6kv::ROW, 64, cudaMemcpyDeviceToHost, st));
            }
            G6RT_CK(cudaMemcpyAsync(back.data(), t.back, back.size(), cudaMemcpyDeviceToHost, st));
            G6RT_CK(cudaStreamSynchronize(st));
            rt_cmp(was.data(), back.data(), 64 * G.layers.size(), t.spill_ck);
        }
        G6RT_CK(cudaStreamSynchronize(st));
    }
    else
        for (size_t j = 0; j < G.layers.size(); ++j)
        {
            const G6Attn& a = m.layers[G.layers[j]].ab.args;
            G6RT_CK(cudaMemcpy2D(buf.data() + j * g6kv::ROW, rec, (const uint8_t*)a.mla_cache + (size_t)p * 64 * g6kv::ROW,
                                 g6kv::ROW, g6kv::ROW, 64, cudaMemcpyDeviceToHost));
        }
}
// slot q's whole pages past L below `committed` (the first row its next step writes), through `map` (logical page ->
// this rank's physical page, < 0 = another rank's or none: the table attention reads): into the store, then flagged.
// Release: the pages through slot q's tail instead; each goes to the store under a free id, its index keys to the
// arena, its table entry to the id, and its window page back once the write has returned. Between steps.
inline RtSpillSeen rt_release(Model& m, RtTier& t, int q, int committed, cudaStream_t st);
inline RtSpillSeen rt_spill(Model& m, RtTier& t, const std::vector<int>& map, int committed, int q, cudaStream_t st)
{
    if (t.local) return rt_release(m, t, q, committed, st);
    RtSpillSeen seen;
    const int end = std::min(committed / 64, (int)map.size());
    std::vector<int> v;
    for (int lp = t.L / 64; lp < end; ++lp)
    {
        const int p = map[lp];
        ++seen.pages;
        if (p < 0) ++seen.other;
        else if (p >= t.pages) ++seen.range;
        else if (t.owner[p] >= 0) ++seen.owned;
        else v.push_back(p);
    }
    seen.spilled = (int)v.size();
    if (v.empty()) return seen;
    G6RT_CK(cudaStreamSynchronize(st));
    std::vector<uint8_t> buf;
    for (int p : v)
    {
        for (RtTier::Group& G : t.g)
        {
            const size_t rec = G.h.rec;
            rt_page_record(m, t, G, p, buf, st);
            if (g6kv_rows_write(t.store, G.h.base + (uint64_t)p * 64 * rec, buf.data(), buf.size()))
                die(std::string("G6_RT_STORE write: ") + g6kv_tier_error());
        }
        G6RT_CK(cudaMemset(t.d_flag + p, 1, 1));
        t.owner[p] = q;
        if (t.poison)
            for (const Layer& y : m.layers)
                G6RT_CK(cudaMemset((uint8_t*)y.ab.args.mla_cache + (size_t)p * 64 * g6kv::ROW, 0x7f, 64 * g6kv::ROW));
        t.spilled += 64;
    }
    return seen;
}

// one released page's records: page p of every layer of each group -> the records of id v
inline void rt_write_page(Model& m, RtTier& t, int p, int v, std::vector<uint8_t>& buf, cudaStream_t st)
{
    if (t.hold)   // G6_RT_CHECK: the rows as they were, to hold a staged prefill's rows against
    {
        std::vector<uint8_t>& o = t.orig[v];
        o.resize((size_t)m.nl * 64 * g6kv::ROW);
        for (int i = 0; i < m.nl; ++i)
            G6RT_CK(cudaMemcpy(o.data() + (size_t)i * 64 * g6kv::ROW, (const uint8_t*)m.layers[i].ab.args.mla_cache + (size_t)p * 64 * g6kv::ROW,
                               64 * g6kv::ROW, cudaMemcpyDeviceToHost));
    }
    for (RtTier::Group& G : t.g)
    {
        const size_t rec = G.h.rec;
        rt_page_record(m, t, G, p, buf, st);
        if (g6kv_rows_write(t.store, G.h.base + (uint64_t)v * 64 * rec, buf.data(), buf.size()))
            die(std::string("G6_RT_STORE write: ") + g6kv_tier_error());
        if (t.hold)   // G6_RT_CHECK: the records read back as written, on the path a staged prefill reads them by
        {
            std::vector<uint8_t> back(buf.size());
            const uint64_t off = G.h.base + (uint64_t)v * 64 * rec;
            if (g6kv_rows_gather(t.store, &off, 1, back.size(), back.data())) die(std::string("G6_RT_CHECK read-back: ") + g6kv_tier_error());
            ++t.readback;
            if (back != buf)
            {
                size_t k = 0;
                while (back[k] == buf[k]) ++k;
                fprintf(stderr, "# rank %d: G6_RT_CHECK read-back of id %d group %zu differs from byte %zu (record %zu) on\n", t.rank, v,
                        (size_t)(&G - t.g.data()), k, k / rec);
                ++t.readback_bad;
            }
        }
        const G6Attn& a = m.layers[G.layers[0]].ab.args;
        if (G.ix && a.index_cache)
            G6RT_CK(cudaMemcpy(G.ix + (size_t)v * 64 * g6kv::IDX_ROW, (const uint8_t*)a.index_cache + (size_t)p * 64 * g6kv::IDX_ROW,
                               64 * g6kv::IDX_ROW, cudaMemcpyDeviceToDevice));
    }
}
inline RtSpillSeen rt_release(Model& m, RtTier& t, int q, int committed, cudaStream_t st)
{
    RtSpillSeen seen;
    RtTier::Seq& s = t.seq[q];
    if (!s.on) return seen;
    const int end = std::min(committed / 64, t.tsize);
    std::vector<uint8_t> buf;
    std::vector<int> freed;
    for (int lp = t.L / 64; lp < end; ++lp)
    {
        if (!rt_mine(t, lp)) continue;
        ++seen.pages;
        const int p = s.tail[lp];
        if (p >= t.pages) { ++seen.owned; continue; }
        if (p < 0) die("G6_RT_LOCAL: slot " + std::to_string(q) + " page " + std::to_string(lp) + " was committed without a local page");
        if (t.vfree.empty()) die("G6_RT_LOCAL: rank " + std::to_string(t.rank) + ": every id of the rows store is in use");
        if (freed.empty()) G6RT_CK(cudaStreamSynchronize(st));
        const int v = t.vfree.back();
        t.vfree.pop_back();
        rt_write_page(m, t, p, v, buf, st);
        G6RT_CK(cudaMemset(t.d_flag + t.pages + v, 1, 1));
        t.owner[t.pages + v] = q;
        t.vlp[v] = lp; t.vslot[v] = q;
        s.tail[lp] = t.pages + v;
        s.pfree.push_back(p);
        freed.push_back(p);
        --s.in_use;
        ++seen.spilled; ++s.released; ++t.released;
        t.spilled += 64;
    }
    if (freed.empty()) return seen;
    if (t.write) t.write(q);
    if (t.poison)   // the freed pages as the next row would find them: no read may reach them
        for (int p : freed)
            for (const Layer& y : m.layers)
                G6RT_CK(cudaMemset((uint8_t*)y.ab.args.mla_cache + (size_t)p * 64 * g6kv::ROW, 0x7f, 64 * g6kv::ROW));
    rt_check(t, "release");
    return seen;
}
// release: layer i's rows and index keys of every page slot q released below `start` into the staged prefill pools
// (each {pool, its index offset or 0}, logical page lp at lp), which k_cp_pf_gather / k_cp_pf_fill leave to this.
// Plain launches on st (synchronizes it first, only when there is something to stage: the pinned stage is reused).
inline void rt_pf_stage(RtTier& t, int q, int i, int start, const std::vector<std::pair<char*, size_t>>& pools, cudaStream_t st)
{
    if (!t.local) return;
    const int np = (start + 63) / 64;
    std::vector<int> ids;
    for (int v = 0; v < t.V; ++v) if (t.vslot[v] == q && t.vlp[v] < np) ids.push_back(v);
    if (ids.empty()) return;
    const RtTier::Group& G = t.g[t.at[i].first];
    const int j = t.at[i].second;
    const size_t row = t.int4 ? g6kv::int4row::ROW : g6kv::ROW, need = ids.size() * 64 * row;   // a record's bytes per layer
    G6RT_CK(cudaStreamSynchronize(st));
    if (t.stage_bytes < need)
    {
        if (t.stage) G6RT_CK(cudaFreeHost(t.stage));
        G6RT_CK(cudaHostAlloc(reinterpret_cast<void**>(&t.stage), need, cudaHostAllocMapped));   // int4: the unpack reads it
        t.stage_bytes = need;
    }
    std::vector<uint64_t> offs;
    for (int v : ids)
        for (int k = 0; k < 64; ++k) offs.push_back(G.h.base + ((uint64_t)v * 64 + k) * G.h.rec + (uint64_t)j * row);
    if (g6kv_rows_gather(t.store, offs.data(), offs.size(), row, t.stage))
        die(std::string("G6_RT_STORE prefill read: ") + g6kv_tier_error());
    void* dstage = nullptr;
    if (t.int4) G6RT_CK(cudaHostGetDevicePointer(&dstage, t.stage, 0));
    for (const auto& [pool, ixoff] : pools)
        for (size_t n = 0; n < ids.size(); ++n)
        {
            const int lp = t.vlp[ids[n]];
            char* dst = pool + (size_t)lp * 64 * g6kv::ROW;
            if (t.int4)   // back into FP8 rows, as a fetch's rt_unpack does
                G6RT_CK((cudaError_t)g6_rt_unpack_int4(static_cast<uint8_t*>(dstage) + n * 64 * row, row, dst, g6kv::ROW, 64, st));
            else
                G6RT_CK(cudaMemcpyAsync(dst, t.stage + n * 64 * row, 64 * row, cudaMemcpyHostToDevice, st));
            if (ixoff && j == 0)
            {
                if (!G.ix) die("G6_RT_LOCAL: a staged layer's index keys without an arena");
                G6RT_CK(cudaMemcpyAsync(pool + ixoff + (size_t)lp * 64 * g6kv::IDX_ROW, G.ix + (size_t)ids[n] * 64 * g6kv::IDX_ROW,
                                        64 * g6kv::IDX_ROW, cudaMemcpyDeviceToDevice, st));
            }
        }
    if (!t.hold) return;
    // G6_RT_CHECK: every staged page of this layer against the rows it was released with
    G6RT_CK(cudaStreamSynchronize(st));
    std::vector<uint8_t> got(64 * g6kv::ROW);
    for (const auto& [pool, ixoff] : pools)
        for (int v : ids)
        {
            const auto it = t.orig.find(v);
            if (it == t.orig.end()) continue;
            G6RT_CK(cudaMemcpy(got.data(), pool + (size_t)t.vlp[v] * 64 * g6kv::ROW, got.size(), cudaMemcpyDeviceToHost));
            RtTier::Check c;
            rt_cmp(it->second.data() + (size_t)i * 64 * g6kv::ROW, got.data(), 64, c);
            t.stage_ck.sig += c.sig; t.stage_ck.err += c.err; t.stage_ck.rows += c.rows; t.stage_ck.rope_bad += c.rope_bad;
            const double db = 10 * log10(c.sig / std::max(c.err, 1e-300));
            if ((db < 10 || c.rope_bad) && t.stage_bad++ < 8)
                fprintf(stderr, "# rank %d: G6_RT_CHECK staged layer %d page %d (id %d, start %d): latent %.2f dB, rope differs in %llu of 64 rows\n",
                        t.rank, i, t.vlp[v], v, start, db, (unsigned long long)c.rope_bad);
        }
}

// L: tokens (multiple of 64, >= the CP map's R); S: slots per group; layout: the store's header, so records are only
// read under the layout that wrote them; lease: the store's name without the rank; local > 0: release, each session
// with that many window pages past L, the store holding released ids only: ids of them (0 = one per page past L this
// rank could hold in one table); step_rows: the most rows one step carries (release: S must hold every row each of them
// selects); int4: INT4 records (G6_RT_FMT); check: G6_RT_CHECK. Before s2_init and any prepare: the views and replay
// entries copy the bindings' arguments.
inline std::unique_ptr<RtTier> rt_build(Model& m, const std::string& spec, const std::string& lease, const std::string& layout,
                                        int L, int S, int local, int ids, int step_rows, bool poison, bool int4, bool check,
                                        cudaStream_t st)
{
#if defined(G6_CP) && G6_CP
    if (!m.cp.on) die("G6_RT_STORE needs G6_CP on (the rows past R)");
    if (L < m.cp.R || L % 64) die("G6_RT_L must be a multiple of 64 and at least G6_CP_R");
    if (L != m.cp.R && std::find(m.bk, m.bk + m.nbk, L) == m.bk + m.nbk) die("G6_RT_L must be a context bucket boundary (G6_DEC_BK)");
    if (S < 2048) die("G6_RT_POOL must be at least 2048 slots (one row's selection)");
    // a released row has no local copy: every row one step selects must find a slot
    if (step_rows < 1 || step_rows > MX) die("rt_build: step rows " + std::to_string(step_rows) + " outside 1..MX");
    if (local && S < step_rows * 2048)
        die("G6_RT_LOCAL needs G6_RT_POOL >= " + std::to_string(step_rows * 2048) + " (2048 selected rows for each of a step's " +
            std::to_string(step_rows) + " rows)");
    auto t = std::make_unique<RtTier>();
    t->L = L; t->S = S; t->nr = g6nr::NR; t->rank = m.rank; t->poison = poison; t->local = local;
    t->int4 = int4; t->check = check && int4; t->hold = check && local;
    t->pages = m.layers[0].ab.args.page_count;
    t->tsize = m.layers[0].ab.args.table_size;
    if (local && ids > 0) t->V = ids;
    else if (local)
        for (int lp = L / 64; lp < t->tsize; ++lp) t->V += lp % t->nr == m.rank;
    for (int i = 0; i < m.nl; ++i)
    {
        const G6Attn& a = m.layers[i].ab.args;
        if (!a.cp_nr || a.page_count != t->pages) die("G6_RT_STORE: layer " + std::to_string(m.layers[i].l) + " is not a CP binding of the pool's pages");
        if (m.layers[i].indexer) t->g.push_back({{i}});
        else
            for (RtTier::Group& G : t->g)
                if (G.layers[0] == m.layers[i].index_src) G.layers.push_back(i);
    }
    t->at.assign(m.nl, {-1, -1});
    for (size_t k = 0; k < t->g.size(); ++k)
        for (size_t j = 0; j < t->g[k].layers.size(); ++j) t->at[t->g[k].layers[j]] = {(int)k, (int)j};
    for (int i = 0; i < m.nl; ++i)
        if (local && t->at[i].first < 0) die("G6_RT_LOCAL: layer " + std::to_string(m.layers[i].l) + " has no full-index layer before it to read released rows through");
    const size_t row = int4 ? g6kv::int4row::ROW : g6kv::ROW;   // a record's bytes per layer
    size_t most = 0, page_bytes = 0;   // the most layers of a group; one page's records over every group
    for (RtTier::Group& G : t->g) { page_bytes += 64 * G.layers.size() * row; most = std::max(most, G.layers.size()); }
    // G6_RT_SIZE (the launcher's KV_ROWS size=): this rank's lease budget in bytes. Release sizes its ids from it (the pool
    // then admits a tail only when its records fit, kvman-rs Pool::window); the layout keyed by physical page must fit it
    if (const char* e = getenv("G6_RT_SIZE"))
    {
        const uint64_t budget = strtoull(e, nullptr, 10), keyed = (uint64_t)t->pages * page_bytes;
        if (budget <= G6KV_ROWS_HEADER) die("G6_RT_SIZE " + std::string(e) + ": not even the store's header");
        if (local) t->V = std::min<uint64_t>(t->V, (budget - G6KV_ROWS_HEADER) / page_bytes);
        else if (budget < G6KV_ROWS_HEADER + keyed)
            die("G6_RT_SIZE " + std::string(e) + ": the store keyed by local page needs " + std::to_string(G6KV_ROWS_HEADER + keyed) +
                " B a rank; release (G6_RT_LOCAL) sizes it by released pages instead");
        if (local && t->V < 1) die("G6_RT_SIZE " + std::string(e) + ": less than one page of records (" + std::to_string(page_bytes) + " B)");
    }
    const int rows = (t->pages + t->V) * 64, recs = local ? t->V * 64 : rows;   // pos2slot rows, store records
    uint64_t bytes = G6KV_ROWS_HEADER;
    for (RtTier::Group& G : t->g) bytes += (uint64_t)recs * G.layers.size() * row;
    const std::string name = lease + "-r" + std::to_string(m.rank);
    if (g6kv_rows_open(spec.c_str(), name.c_str(), layout.c_str(), bytes, &t->store))
        die("G6_RT_STORE " + spec + " (" + name + "): " + g6kv_tier_error());
    t->d_flag = dmalloc<uint8_t>(t->pages + t->V);
    t->d_list = dmalloc<int>(t->pages + t->V);
    t->owner.assign(t->pages + t->V, -1);
    t->vlp.assign(t->V, -1);
    t->vslot.assign(t->V, -1);
    for (int v = t->V - 1; v >= 0; --v) t->vfree.push_back(v);
    if (int4) t->pack = dmalloc<uint8_t>(64 * most * row);
    if (t->check) t->back = dmalloc<uint8_t>(64 * most * g6kv::ROW);
    const int cap = MX * 2048;
    uint64_t base = G6KV_ROWS_HEADER;
    for (RtTier::Group& G : t->g)
    {
        const size_t rec = G.layers.size() * row, srec = G.layers.size() * g6kv::ROW;   // a record, a slot
        G6RtGroup& h = G.h;
        h.pos2slot = dmalloc<int>(rows);
        h.slot2pos = dmalloc<int>(S);
        h.last = dmalloc<unsigned long long>(S);
        h.miss = dmalloc<int>(cap);
        G6RT_CK(cudaMemset(h.pos2slot, 0xff, (size_t)rows * sizeof(int)));
        G6RT_CK(cudaMemset(h.slot2pos, 0xff, S * sizeof(int)));
        G6RT_CK(cudaMemset(h.last, 0xff, S * sizeof(unsigned long long)));
        h.step = reinterpret_cast<const unsigned long long*>(m.stepp);
        h.base = base; h.rec = rec; h.slots = S; h.rows = rows; h.cap = cap; h.kbase = local ? t->pages * 64 : 0;
        base += (uint64_t)recs * rec;
        if (local && m.layers[G.layers[0]].ab.args.index_cache) G.ix = dmalloc<uint8_t>((size_t)t->V * 64 * g6kv::IDX_ROW);
        const size_t ring = sizeof(g6kv_rows_req) + (size_t)cap * sizeof(g6kv_rows_ent);
        G6RT_CK(cudaHostAlloc(&G.req, ring, cudaHostAllocMapped));
        memset(G.req, 0, ring);
        G6RT_CK(cudaHostAlloc(reinterpret_cast<void**>(&G.done), sizeof(uint64_t), cudaHostAllocMapped));
        *G.done = 0;
        G6RT_CK(cudaHostAlloc(reinterpret_cast<void**>(&G.slots), (size_t)S * srec, cudaHostAllocMapped));
        memset(G.slots, 0, (size_t)S * srec);
        void *rq = nullptr, *dn = nullptr, *sl = nullptr, *sg = nullptr;
        G6RT_CK(cudaHostGetDevicePointer(&rq, G.req, 0));
        G6RT_CK(cudaHostGetDevicePointer(&dn, G.done, 0));
        G6RT_CK(cudaHostGetDevicePointer(&sl, G.slots, 0));
        h.req = rq; h.done = static_cast<const unsigned long long*>(dn);
        h.nl = (int)G.layers.size();
        if (int4)
        {
            G6RT_CK(cudaHostAlloc(reinterpret_cast<void**>(&G.stage), (size_t)S * rec, cudaHostAllocMapped));
            memset(G.stage, 0, (size_t)S * rec);
            G6RT_CK(cudaHostGetDevicePointer(&sg, G.stage, 0));
            h.stage = static_cast<const unsigned char*>(sg);
            h.out = static_cast<unsigned char*>(sl);
        }
        uint8_t* into = int4 ? G.stage : G.slots;   // what the helper reads records into
        G.dev = dmalloc<G6RtGroup>(1);
        G6RT_CK(cudaMemcpy(G.dev, &h, sizeof h, cudaMemcpyHostToDevice));
        if (g6kv_rows_register(t->store, into, (size_t)S * rec)) die(std::string("G6_RT_STORE register: ") + g6kv_tier_error());
        if (g6kv_rows_group(t->store, static_cast<g6kv_rows_req*>(G.req), G.done, into, rec, (uint32_t)S, (uint32_t)cap) < 0)
            die(std::string("G6_RT_STORE group: ") + g6kv_tier_error());
        for (size_t j = 0; j < G.layers.size(); ++j)
        {
            G6Attn& a = m.layers[G.layers[j]].ab.args;
            a.rt_page = t->d_flag;
            a.rt_table = h.pos2slot;
            a.rt_rows = static_cast<const unsigned char*>(sl) + j * g6kv::ROW;
            a.rt_stride = (int)srec;
            a.rt_group = j == 0 ? G.dev : nullptr;
            a.rt_int4 = j == 0 && int4;
            a.rt_vpages = t->V;
            a.rt_ix = j == 0 ? G.ix : nullptr;
        }
    }
    G6RT_CK(cudaDeviceSynchronize());
    g6kv_rows_start(t->store);
    char d[256];
    g6kv_rows_describe(t->store, d, sizeof d);
    // fp8 keeps its line as it was
    const double rec_all = (double)(bytes - G6KV_ROWS_HEADER) / recs;   // every group's record of one row
    if (int4)
        fprintf(stderr, "# rank %d: G6_RT_STORE %s (%s): int4 records (%.0f B a token), rows past %d, %zu groups, %d slots each (%.3f GB "
                "pinned with the stage), %.3f GB of records\n",
                m.rank, spec.c_str(), d, rec_all, L, t->g.size(), S, (double)S * (rec_all / row * g6kv::ROW + rec_all) / 1e9,
                (bytes - G6KV_ROWS_HEADER) / 1e9);
    else
        fprintf(stderr, "# rank %d: G6_RT_STORE %s (%s): rows past %d, %zu groups, %d slots each (%.3f GB pinned), %.3f GB of records\n",
                m.rank, spec.c_str(), d, L, t->g.size(), S, (double)S * (bytes - G6KV_ROWS_HEADER) / recs / 1e9, (bytes - G6KV_ROWS_HEADER) / 1e9);
    if (local)
        fprintf(stderr, "# rank %d: G6_RT_LOCAL %d local pages past L, %d released ids (%.3f GB of index keys kept)\n", m.rank, local, t->V,
                (double)t->V * 64 * g6kv::IDX_ROW * t->g.size() / 1e9);
    if (t->check && m.rank == 0) rt_selftest(4096, st);
    return t;
#else
    (void)m; (void)spec; (void)lease; (void)layout; (void)L; (void)S; (void)local; (void)ids; (void)step_rows; (void)poison; (void)int4;
    (void)check; (void)st;
    die("G6_RT_STORE needs a G6_CP build");
#endif
}

inline uint64_t rt_fetched(const RtTier& t) { return g6kv_rows_fetched(t.store); }
// rows the pools had no room for (read locally), summed over the groups
inline uint64_t rt_local(const RtTier& t)
{
    uint64_t n = 0;
    for (const RtTier::Group& G : t.g)
    {
        G6RtGroup h;
        G6RT_CK(cudaMemcpy(&h, G.dev, sizeof h, cudaMemcpyDeviceToHost));
        n += h.local;
    }
    return n;
}
// a lease name part: letters, digits, '.', '_' and '-' only
inline std::string rt_name_part(const std::string& s)
{
    std::string o;
    for (char c : s) o += (isalnum((unsigned char)c) || c == '.' || c == '_' || c == '-') ? c : '_';
    return o.empty() ? "none" : o;
}

} // namespace g6rt
