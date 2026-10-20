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
// Store layout, one store per rank: the layout header (G6KV_ROWS_HEADER bytes), then group after group in layer order,
// one record per physical row (page * 64 + pos % 64) of this rank's attention cache. A record is the row of every layer
// of the group, in layer order, g6kv::ROW bytes each. A group is a full-index layer and the shared-index layers that
// reuse its list.
//
// G6_RT_FMT=int4 (int4 here): a record holds each row in the INT4 layout (kvfmt.h int4row, 416 B for 656), so a lease
// holds 0.63x the bytes and a fetch moves 0.63x; the pages and the slots stay FP8, so no attention kernel changes. The
// spill packs each page's rows on the GPU (attn.cu g6_rt_pack_int4); the helper reads records into a stage at the
// slot's place, and after the wait rt_unpack (attn/rt.cuh) requantizes them into the slot as append_kv's FP8 path
// would. The rope key goes through exactly. check (G6_RT_CHECK=1): a round trip of random rows at build, and of every
// spilled row, against the FP8 rows (latent SQNR, rope bytes), printed by rt_check_print.
#pragma once
#include "rank.cuh"
#include "../kvman-rs/g6kvman.h"
#include <cuda_fp8.h>
#include <cfloat>
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
    };
    int L = 0, S = 0, nr = 0, rank = 0, pages = 0;
    bool poison = false, int4 = false, check = false;
    uint8_t *pack = nullptr, *back = nullptr;   // int4: device [64][rec] for the spill; check: [64][rec of FP8] back
    struct Check { double sig = 0, err = 0; uint64_t rows = 0, rope_bad = 0; } spill_ck;
    g6kv_rows* store = nullptr;
    std::vector<Group> g;
    uint8_t* d_flag = nullptr;        // [pages] device: the page is in the store
    std::vector<int> owner;           // [pages] the session slot whose spill flagged it, -1 = not flagged
    int* d_list = nullptr;            // [pages] device scratch: the pages a drop clears
    uint64_t spilled = 0, dropped = 0;
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
        if (p >= 0 && p < t.pages && t.owner[p] >= 0) v.push_back(p);
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
    for (int p = 0; p < t.pages; ++p) if (t.owner[p] == q) v.push_back(p);
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
// slot q's whole pages past L below `committed` (the first row its next step writes), through `map` (logical page ->
// this rank's physical page, < 0 = another rank's or none: the table attention reads): into the store, then flagged.
// Between steps.
inline RtSpillSeen rt_spill(Model& m, RtTier& t, const std::vector<int>& map, int committed, int q, cudaStream_t st)
{
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

// L: tokens (multiple of 64, >= the CP map's R); S: slots per group; layout: the store's header, so records are only
// read under the layout that wrote them; lease: the store's name without the rank; int4: INT4 records (G6_RT_FMT);
// check: G6_RT_CHECK. Before s2_init and any prepare:
// the views and replay entries copy the bindings' arguments.
inline std::unique_ptr<RtTier> rt_build(Model& m, const std::string& spec, const std::string& lease, const std::string& layout,
                                        int L, int S, bool poison, bool int4, bool check, cudaStream_t st)
{
#if defined(G6_CP) && G6_CP
    if (!m.cp.on) die("G6_RT_STORE needs G6_CP on (the rows past R)");
    if (L < m.cp.R || L % 64) die("G6_RT_L must be a multiple of 64 and at least G6_CP_R");
    if (L != m.cp.R && std::find(m.bk, m.bk + m.nbk, L) == m.bk + m.nbk) die("G6_RT_L must be a context bucket boundary (G6_DEC_BK)");
    if (S < 2048) die("G6_RT_POOL must be at least 2048 slots (one row's selection)");
    auto t = std::make_unique<RtTier>();
    t->L = L; t->S = S; t->nr = g6nr::NR; t->rank = m.rank; t->poison = poison; t->int4 = int4; t->check = check && int4;
    t->pages = m.layers[0].ab.args.page_count;
    for (int i = 0; i < m.nl; ++i)
    {
        const G6Attn& a = m.layers[i].ab.args;
        if (!a.cp_nr || a.page_count != t->pages) die("G6_RT_STORE: layer " + std::to_string(m.layers[i].l) + " is not a CP binding of the pool's pages");
        if (m.layers[i].indexer) t->g.push_back({{i}});
        else
            for (RtTier::Group& G : t->g)
                if (G.layers[0] == m.layers[i].index_src) G.layers.push_back(i);
    }
    const int rows = t->pages * 64;
    const size_t row = int4 ? g6kv::int4row::ROW : g6kv::ROW;   // a record's bytes per layer
    uint64_t bytes = G6KV_ROWS_HEADER;
    size_t most = 0;
    for (RtTier::Group& G : t->g) { bytes += (uint64_t)rows * G.layers.size() * row; most = std::max(most, G.layers.size()); }
    const std::string name = lease + "-r" + std::to_string(m.rank);
    if (g6kv_rows_open(spec.c_str(), name.c_str(), layout.c_str(), bytes, &t->store))
        die("G6_RT_STORE " + spec + " (" + name + "): " + g6kv_tier_error());
    t->d_flag = dmalloc<uint8_t>(t->pages);
    t->d_list = dmalloc<int>(t->pages);
    t->owner.assign(t->pages, -1);
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
        h.base = base; h.rec = rec; h.slots = S; h.rows = rows; h.cap = cap;
        base += (uint64_t)rows * rec;
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
        }
    }
    G6RT_CK(cudaDeviceSynchronize());
    g6kv_rows_start(t->store);
    char d[256];
    g6kv_rows_describe(t->store, d, sizeof d);
    // fp8 keeps its line as it was
    const double rec_all = (double)(bytes - G6KV_ROWS_HEADER) / rows;   // every group's record of one row
    if (int4)
        fprintf(stderr, "# rank %d: G6_RT_STORE %s (%s): int4 records (%.0f B a token), rows past %d, %zu groups, %d slots each (%.3f GB "
                "pinned with the stage), %.3f GB of records\n",
                m.rank, spec.c_str(), d, rec_all, L, t->g.size(), S, (double)S * (rec_all / row * g6kv::ROW + rec_all) / 1e9,
                (bytes - G6KV_ROWS_HEADER) / 1e9);
    else
        fprintf(stderr, "# rank %d: G6_RT_STORE %s (%s): rows past %d, %zu groups, %d slots each (%.3f GB pinned), %.3f GB of records\n",
                m.rank, spec.c_str(), d, L, t->g.size(), S, (double)S * (bytes - G6KV_ROWS_HEADER) / rows / 1e9, (bytes - G6KV_ROWS_HEADER) / 1e9);
    if (t->check && m.rank == 0) rt_selftest(4096, st);
    return t;
#else
    (void)m; (void)spec; (void)lease; (void)layout; (void)L; (void)S; (void)poison; (void)int4; (void)check; (void)st;
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
