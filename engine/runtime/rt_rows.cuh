// Remote attention rows (G6_RT_STORE): a rank's latent rows past L (default the CP map's R) live in a rows store
// (kvman-rs src/rows.rs: this process's memory, or a lease on a KV node), and each indexer group keeps a pool of
// resident slots that decode reads them from (engine/attn/rt.cuh, attn.h G6RtGroup).
//
// Rows past L are written to the store after they are computed (spill, between steps), and their pages are marked
// G6_RT_REMOTE in every layer's block table. The local pages stay allocated; nothing reads them after the mark (spill
// with poison = 0x7f proves that). Index keys stay local: every binding reads them through its own copy of the table as
// built (index_table).
//
// Store layout, one store per rank: the layout header (G6KV_ROWS_HEADER bytes), then group after group in layer
// order, `rows` records each. Record k of a group is the row index k (attn.h: this rank's k-th row past L, every
// nr-th page under CP) of every layer of the group, in layer order, g6kv::ROW bytes each. A group is a full-index
// layer and the shared-index layers that reuse its list.
//
// M1 scope: the CP decode path (G6_CP on, context > R); L >= R. Not with the MTP draft model, S2, prefix reuse or a
// prefill after rows were spilled.
#pragma once
#include "rank.cuh"
#include "../kvman-rs/g6kvman.h"
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
        uint8_t* slots = nullptr;     // mapped pinned [slots][rec]
        void* req = nullptr;          // mapped pinned ring
        uint64_t* done = nullptr;     // mapped pinned
    };
    int L = 0, S = 0, nr = 0, rank = 0, pages = 0, rows = 0, next_lp = 0;
    bool poison = false;
    g6kv_rows* store = nullptr;
    std::vector<Group> g;
    std::vector<int> table;           // the CP map as built (logical page -> physical page or G6_CP_REMOTE)
    std::vector<int> marked;          // logical pages marked G6_RT_REMOTE
    uint64_t spilled = 0;
    ~RtTier() { if (store) g6kv_rows_free(store); }
};

inline void rt_reset(Model& m, RtTier& t, cudaStream_t st);
inline bool rt_owned(const RtTier& t, int lp) { return lp >= t.L / 64 && t.table[lp] >= 0; }
inline int rt_index(const RtTier& t, int lp) { return (lp - t.L / 64) / t.nr * 64; }   // attn.cu rt_index, a page's first row

// L: tokens (multiple of 64, >= the CP map's R); S: slots per group; layout: the store's header and part of its lease
// name, so records are only read under the layout that wrote them
inline std::unique_ptr<RtTier> rt_build(Model& m, const std::string& spec, const std::string& name, const std::string& layout,
                                        int L, int S, bool poison, cudaStream_t st)
{
#if defined(G6_CP) && G6_CP
    if (!m.cp.on) die("G6_RT_STORE needs G6_CP on (the rows past R)");
    if (L < m.cp.R || L % 64) die("G6_RT_L must be a multiple of 64 and at least G6_CP_R");
    if (L != m.cp.R && std::find(m.bk, m.bk + m.nbk, L) == m.bk + m.nbk) die("G6_RT_L must be a context bucket boundary (G6_DEC_BK)");
    if (S < 2048) die("G6_RT_POOL must be at least 2048 slots (one row's selection)");
    auto t = std::make_unique<RtTier>();
    t->L = L; t->S = S; t->nr = g6nr::NR; t->rank = m.rank; t->poison = poison;
    const G6Attn& a0 = m.layers[0].ab.args;
    t->pages = a0.table_size;
    t->table.resize(t->pages);
    G6RT_CK(cudaMemcpy(t->table.data(), a0.block_table, t->pages * sizeof(int), cudaMemcpyDeviceToHost));
    t->rows = t->pages > L / 64 ? (t->pages - L / 64 + t->nr - 1) / t->nr * 64 : 0;
    for (int i = 0; i < m.nl; ++i)
    {
        if (!m.layers[i].ab.args.cp_nr) die("G6_RT_STORE: layer " + std::to_string(m.layers[i].l) + " has no CP binding");
        if (m.layers[i].indexer) t->g.push_back({{i}});
        else
            for (RtTier::Group& G : t->g)
                if (G.layers[0] == m.layers[i].index_src) G.layers.push_back(i);
    }
    uint64_t bytes = G6KV_ROWS_HEADER;
    for (RtTier::Group& G : t->g) bytes += (uint64_t)t->rows * G.layers.size() * g6kv::ROW;
    const std::string lease = name + "-r" + std::to_string(m.rank);
    if (g6kv_rows_open(spec.c_str(), lease.c_str(), layout.c_str(), bytes, &t->store))
        die("G6_RT_STORE " + spec + ": " + g6kv_tier_error());
    const int cap = MX * 2048;
    uint64_t base = G6KV_ROWS_HEADER;
    for (RtTier::Group& G : t->g)
    {
        const size_t rec = G.layers.size() * g6kv::ROW;
        G6RtGroup& h = G.h;
        h.pos2slot = dmalloc<int>(std::max(t->rows, 1));
        h.slot2pos = dmalloc<int>(S);
        h.last = dmalloc<unsigned long long>(S);
        h.miss = dmalloc<int>(cap);
        h.step = reinterpret_cast<const unsigned long long*>(m.stepp);
        h.base = base; h.rec = rec; h.slots = S; h.rows = t->rows; h.cap = cap;
        base += (uint64_t)t->rows * rec;
        const size_t ring = sizeof(g6kv_rows_req) + (size_t)cap * sizeof(g6kv_rows_ent);
        G6RT_CK(cudaHostAlloc(&G.req, ring, cudaHostAllocMapped));
        memset(G.req, 0, ring);
        G6RT_CK(cudaHostAlloc(reinterpret_cast<void**>(&G.done), sizeof(uint64_t), cudaHostAllocMapped));
        *G.done = 0;
        G6RT_CK(cudaHostAlloc(reinterpret_cast<void**>(&G.slots), (size_t)S * rec, cudaHostAllocMapped));
        memset(G.slots, 0, (size_t)S * rec);
        void *rq = nullptr, *dn = nullptr, *sl = nullptr;
        G6RT_CK(cudaHostGetDevicePointer(&rq, G.req, 0));
        G6RT_CK(cudaHostGetDevicePointer(&dn, G.done, 0));
        G6RT_CK(cudaHostGetDevicePointer(&sl, G.slots, 0));
        h.req = rq; h.done = static_cast<const unsigned long long*>(dn);
        G.dev = dmalloc<G6RtGroup>(1);
        G6RT_CK(cudaMemcpy(G.dev, &h, sizeof h, cudaMemcpyHostToDevice));
        if (g6kv_rows_register(t->store, G.slots, (size_t)S * rec)) die(std::string("G6_RT_STORE register: ") + g6kv_tier_error());
        if (g6kv_rows_group(t->store, static_cast<g6kv_rows_req*>(G.req), G.done, G.slots, rec, (uint32_t)cap) < 0)
            die(std::string("G6_RT_STORE group: ") + g6kv_tier_error());
        for (size_t j = 0; j < G.layers.size(); ++j)
        {
            G6Attn& a = m.layers[G.layers[j]].ab.args;
            int* it = dmalloc<int>(t->pages);   // the index keys keep the table as built
            G6RT_CK(cudaMemcpy(it, a.block_table, t->pages * sizeof(int), cudaMemcpyDeviceToDevice));
            a.index_table = it;
            a.rt_table = h.pos2slot;
            a.rt_rows = static_cast<const unsigned char*>(sl) + j * g6kv::ROW;
            a.rt_stride = (int)rec;
            a.rt_L = L;
            a.rt_group = j == 0 ? G.dev : nullptr;
        }
    }
    rt_reset(m, *t, st);
    g6kv_rows_start(t->store);
    if (m.rank == 0)
    {
        char d[256];
        g6kv_rows_describe(t->store, d, sizeof d);
        fprintf(stderr, "# rank 0: G6_RT_STORE %s (%s): rows past %d, %zu groups, %d slots each (%.3f GB pinned), %.3f GB of records\n",
                spec.c_str(), d, L, t->g.size(), S, (double)S * (bytes - G6KV_ROWS_HEADER) / std::max(t->rows, 1) / 1e9,
                (bytes - G6KV_ROWS_HEADER) / 1e9);
    }
    return t;
#else
    (void)m; (void)spec; (void)name; (void)layout; (void)L; (void)S; (void)poison; (void)st;
    die("G6_RT_STORE needs a G6_CP build");
    return nullptr;
#endif
}

// a new sequence: the pages as built, every slot free (the ring's seq keeps counting)
inline void rt_reset(Model& m, RtTier& t, cudaStream_t st)
{
    G6RT_CK(cudaStreamSynchronize(st));
    for (int lp : t.marked)
        for (Layer& y : m.layers)
            G6RT_CK(cudaMemcpy(const_cast<int*>(y.ab.args.block_table) + lp, &t.table[lp], sizeof(int), cudaMemcpyHostToDevice));
    t.marked.clear();
    t.next_lp = 0;
    for (RtTier::Group& G : t.g)
    {
        G6RT_CK(cudaMemset(G.h.pos2slot, 0xff, std::max(t.rows, 1) * sizeof(int)));
        G6RT_CK(cudaMemset(G.h.slot2pos, 0xff, t.S * sizeof(int)));
        G6RT_CK(cudaMemset(G.h.last, 0xff, t.S * sizeof(unsigned long long)));
        const int zero[2] = {0, 0};   // hand, n
        G6RT_CK(cudaMemcpy(&G.dev->hand, zero, sizeof zero, cudaMemcpyHostToDevice));
    }
}

// rows below `upto` (the next step's first position): this rank's whole pages past L that are not spilled yet go to
// the store and are marked remote. Between steps (synchronizes st).
inline void rt_spill(Model& m, RtTier& t, int upto, cudaStream_t st)
{
    G6RT_CK(cudaStreamSynchronize(st));
    const int end = std::min(upto / 64, t.pages);
    std::vector<uint8_t> buf;
    for (int lp = std::max(t.next_lp, t.L / 64); lp < end; ++lp)
    {
        if (!rt_owned(t, lp)) continue;
        const int phys = t.table[lp];
        for (RtTier::Group& G : t.g)
        {
            const size_t rec = G.h.rec;
            buf.resize(64 * rec);
            for (size_t j = 0; j < G.layers.size(); ++j)
            {
                const G6Attn& a = m.layers[G.layers[j]].ab.args;
                G6RT_CK(cudaMemcpy2D(buf.data() + j * g6kv::ROW, rec, (const uint8_t*)a.mla_cache + (size_t)phys * 64 * g6kv::ROW,
                                     g6kv::ROW, g6kv::ROW, 64, cudaMemcpyDeviceToHost));
            }
            if (g6kv_rows_write(t.store, G.h.base + (uint64_t)rt_index(t, lp) * rec, buf.data(), buf.size()))
                die(std::string("G6_RT_STORE write: ") + g6kv_tier_error());
        }
        const int mark = G6_RT_REMOTE;
        for (Layer& y : m.layers)
        {
            G6RT_CK(cudaMemcpy(const_cast<int*>(y.ab.args.block_table) + lp, &mark, sizeof mark, cudaMemcpyHostToDevice));
            if (t.poison) G6RT_CK(cudaMemset((uint8_t*)y.ab.args.mla_cache + (size_t)phys * 64 * g6kv::ROW, 0x7f, 64 * g6kv::ROW));
        }
        t.marked.push_back(lp);
        t.spilled += 64;
    }
    t.next_lp = std::max(t.next_lp, end);
}

inline uint64_t rt_fetched(const RtTier& t) { return g6kv_rows_fetched(t.store); }

} // namespace g6rt
