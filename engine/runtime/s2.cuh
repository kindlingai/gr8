// S2 batched decode, Stage 1 (component 6; internal design notes,
// not in this repository). Included by rank.cuh under -DDEC_S2; every hook there is inert unless
// Model::s2on. Host code plus four new kernels (k_s2_cells in s2_cells.cuh, k_s2_draft_hid, k_s2_draft_keep); the attention
// kernels read per-row bounds (attn.cu, DEC_S2 only).
//
// A step carries the rows of up to S2N sessions, session-major, each session's rows contiguous in position order
// (s2_batch.h Step). Everything but attention is row-invariant (Stage 0) and runs over all M rows as today, with the
// MoE k1 split of the step's row class (A1: S2In::cls, bind::make_layer split_m). Attention runs once per layer over all
// rows of the step (batched, below; G6_S2_ATTN_PERSESSION=1: once per session over the session's rows of q_b / kv_a / index
// inputs, then a device copy of its output rows) into the concatenated attention rows (cat) that o_proj's fragment table reads.
// KV (S2 design decision: v1 = CONTIGUOUS per-session runs, base + full reservation, not paged).
// Stage 1 stub of TP4's pool: per layer the pool is the layer's OWN binding (y.ab, created by build at s2_pool_slots x the
// per-session reservation, the --cap pages; identity block table); session slot q owns the run of pages
// [base, base + run_pages), base = q * run_pages until s2_set_base. Pool-absolute pages (joint TP6 + TP4 ruling, TP4
// option (a), provisional): slot q's view = the pool binding's args with its OWN block table [run_pages] = base + i,
// table_size = run_pages (logical bound), page_count = the pool's pages (fixed physical bound of the args), and its own
// index list / output / error words. The step cell bounds the live run: page_count = base + pages(ctx) (k_s2_cells from
// S2In::base; today's path: k_step_in from StepIn::base_page), table_size = pages(ctx). Base 0 = today's values.
// A base change rewrites table CONTENTS only (fixed addresses), so no replay entry or graph is re-made.
// S=1 (fresh or 2 -> 1): today's path and graphs on the layers' own bindings; s2_solo_base writes their table to B + i and
// the driver sets StepIn::base_page = B. Fallback (only if the page_count audit kills option (a)): s2_relocate.
// Per-session cells [slot][layer] are written by k_s2_cells from the device S2In of the step (written by the driver before
// the launch, like StepIn); the per-session calls run without graphs and without replay entries.
// Batched attention (default; G6_S2_ATTN_PERSESSION=1 keeps the per-session calls above): ONE attention call per layer for
// all rows of the step. Per layer one batch view (its own index list and error word, output = cat) and one replay entry keyed
// (bucket of the longest session, layer, M): independent of which sessions and how many rows each has. The per-row bounds and
// block table live in device data (G6AttnRow per row, s2_cells.cuh) that k_s2_cells writes from S2In before the launch, so one
// captured graph serves every session mix of the same M and bucket; the outputs land in the concatenated rows (cat) directly,
// and a shared-index layer reads its source layer's index row by step row.
// Draft model (MTP layer): the eh_proj hidden rows are gathered into hcat by a kernel before k_mtp_in (first form:
// the main model's x rows of the session, chain: the session's kept row), and a kernel keeps each session's last row;
// both read S2In (k_s2_draft_hid, k_s2_draft_keep).
#pragma once
#include <map>
#include <memory>
#include <tuple>
#include "s2_cells.cuh"

namespace g6rt
{

// Graph key of an S2 step = (kind, M, bucket) and nothing else: the session layout (sess / row0 / rows / ctx / base / src0)
// is data in S2In, read on the device by k_s2_cells, k_s2_draft_hid and k_s2_draft_keep.
// A graph is valid for any layout of its M rows over the sessions (the host checks the layout against the key at launch).
// The step's bucket (S2Model::bucket) is the largest bucket of its sessions' contexts.
//
// The attention reads its per-row contexts and run bases from the per-row cells (s2_cells.cuh, one call per layer), so the
// promise holds in the batched form (default). The per-session form (G6_S2_ATTN_PERSESSION=1) bakes the layout into its
// replay entries: it is for runs without graphs only (s2_init refuses it with graphs).
constexpr bool S2_ATTN_LAYOUT_FREE = true;

struct S2Model
{
    bool graph = false;
    int nslots = 0;
    S2In h{};                         // host copy of the current step (set by the driver before prepare / launch)
    const S2In* in = nullptr;         // device (or pinned host) copy, read by k_s2_cells
    int bk[S2N] = {};                 // graph form: per slot, the bucket of its context in the current step
    int bucket = 0;                   // graph form: the bucket of the current step (the largest of its sessions' contexts)
    int pool_pages = 0;               // pages of the pool (the layers' own bindings)
    int run_pages = 0;                // the most pages one session's run can hold (each view's table allocation)
    int base[S2N] = {};               // per slot, the first page of its run
    int pages[S2N] = {};              // per slot, the pages of its run (0 = unbound: table all -1); s2_set_run
    int solo_pages = 0;               // the run of the S=1 session on the layers' own tables (s2_solo_run; default run_pages)
    G6AttnStepCell* cells = nullptr;  // [S2N][nl] device
    bool batch = true;                // one attention call per layer for the whole step (false: per session)
    G6AttnStepCell* bcells = nullptr; // [nl] device: the batched cells, rows = rowcells + l * MX
    G6AttnRow* rowcells = nullptr;    // [nl][MX] device: per step row, its session's bounds and block table
    const int** tabs = nullptr;       // [S2N][nl] device: the slots' block tables (fixed addresses)
    std::vector<G6AttnBinding*> bv;   // [nl] the batch views
    std::map<std::tuple<int, int, int>, std::unique_ptr<G6AttnReplay>> brp;           // (bucket of the longest session, layer, M)
    unsigned** errs = nullptr;        // [S2N][nl] device; unused slots -> zero
    unsigned* zero = nullptr;
    __half* cat = nullptr;            // [MX][heads * 256] the step's attention rows, session-major
    std::vector<std::vector<G6AttnBinding*>> sab;            // [nl][S2N] the slots' views
    std::vector<G6AttnBinding*> pool;                        // [nl] the layers' own bindings (the pool)
    std::vector<std::vector<int>> tab0;                      // [nl] S2 below R: the pool tables as built (base-0 solo = today)
    std::vector<std::unique_ptr<G6AttnBinding>> views;       // shallow views of a pool binding (never released / reset)
    std::map<std::tuple<int, int, int>, std::unique_ptr<bind::Layer>> L;              // (layer, M, cls)
    __half *hcat = nullptr, *hprev_b = nullptr;              // draft model: [MX][HID], [S2N][HID]
};

// once per model, after build, before any S2 step: nslots session slots (views on the pool; the layers' own bindings
// stay allocated but unused while s2on)
// slot q's tables (every layer) = base + i
inline void s2_write_table(Model& m, int q)
{
    S2Model& s = *m.s2;
    std::vector<int> t(s.run_pages);
    for (int i = 0; i < s.run_pages; ++i) t[i] = i < s.pages[q] ? s.base[q] + i : -1;   // -1 past the run: attention error bit 2
    for (int l = 0; l < m.nl; ++l) G6RT_CK(g6h2d::copy(const_cast<int*>(s.sab[l][q]->args.block_table), t.data(), t.size() * 4));
}
inline void s2_init(Model& m, int nslots, cudaStream_t st)
{
    if (nslots < 1 || nslots > S2N) die("s2: 1..8 session slots");
    S2Model* s = new S2Model();
    s->graph = m.graph;
    s->nslots = nslots;
    s->cells = dmalloc<G6AttnStepCell>((size_t)S2N * m.nl);
    {
        const char* e = getenv("G6_S2_ATTN_PERSESSION");
        s->batch = !(e && atoi(e));
        if (!s->batch && s->graph) die("s2: G6_S2_ATTN_PERSESSION=1 bakes the session layout into replay entries; use it without graphs");
    }
    s->errs = dmalloc<unsigned*>((size_t)S2N * m.nl);
    s->zero = dmalloc<unsigned>(1);
    int heads = 0;
    for (const Layer& y : m.layers) heads = std::max(heads, y.ab.args.heads);
    s->cat = dmalloc<__half>((size_t)MX * heads * 256);
    s->hcat = dmalloc<__half>((size_t)MX * HID);
    s->hprev_b = dmalloc<__half>((size_t)S2N * HID);
    s->sab.assign(m.nl, std::vector<G6AttnBinding*>(S2N, nullptr));
    s->bv.assign(m.nl, nullptr);
    std::vector<const int*> tabs((size_t)S2N * m.nl, nullptr);
    std::vector<unsigned*> er((size_t)S2N * m.nl, s->zero);
    if (m.s2_below)
    {
        // S2 below R (S2-STATE.md, CP builds): the pool is each layer's own binding as built. A run holds at most
        // s2_below rows, all on the replicated path (pages < R/64 of the CP map are replicated; the views clear cp_nr), so
        // runs may sit on any physical page. Pool pages = the FEWEST physical pages of any rank (the CP tail split
        // differs by one page across ranks), so every rank grants the same runs.
        if (m.s2_below % 64 || m.s2_below / 64 > m.s2_run_pages) die("s2: s2_below must be a multiple of 64 and <= the capacity");
        s->run_pages = s->solo_pages = m.s2_below / 64;
        const G6Attn& a0 = m.layers[0].ab.args;
        s->pool_pages = a0.page_count;
#if defined(G6_CP) && G6_CP
        if (a0.cp_nr)
        {
            if (m.s2_below > m.cp.R) die("s2: s2_below > G6_CP_R (the runs must stay on the replicated path)");
            const int rp = m.cp.R / 64;
            s->pool_pages = rp + (a0.table_size - rp) / a0.cp_nr;
        }
#endif
    }
    else
    {
        if (m.s2_pool_slots < nslots) die("s2: set Model::s2_pool_slots >= the session slots before build");
        s->run_pages = s->solo_pages = m.s2_run_pages;
        s->pool_pages = m.layers[0].ab.args.page_count;
    }
    // default layout (the m4sim stub): slot q at q x run_pages, full runs, when slots x run fits the pool; else every slot
    // unbound (table all -1) until s2_set_run
    const bool fits = (int64_t)nslots * s->run_pages <= s->pool_pages;
    for (int q = 0; q < nslots; ++q) { s->base[q] = fits ? q * s->run_pages : 0; s->pages[q] = fits ? s->run_pages : 0; }
    size_t bytes = 0, vbytes = 0;
    for (int i = 0; i < m.nl; ++i)
    {
        Layer& y = m.layers[i];
        if ((m.s2_below ? y.ab.args.page_count < s->pool_pages : y.ab.args.page_count != s->pool_pages) || s->pool_pages < s->run_pages)
            die("s2: pool bindings differ or are smaller than one run");
        s->pool.push_back(&y.ab);
        if (m.s2_below)             // the layer's table as built (CP map / identity): s2_solo_run at base 0 restores it
        {
            s->tab0.emplace_back(y.ab.args.table_size);
            G6RT_CK(cudaMemcpy(s->tab0.back().data(), y.ab.args.block_table, s->tab0.back().size() * 4, cudaMemcpyDeviceToHost));
        }
        G6AttnBinding* p = &y.ab;
        bytes += p->storage_bytes;
        for (int q = 0; q < nslots; ++q)
        {
            s->views.emplace_back(new G6AttnBinding());
            G6AttnBinding* v = s->views.back().get();
            *v = *p;
            v->capacity = y.ab.capacity;
            v->storage_bytes = 0;
            // a view only runs S2 decode steps (StepIn holds <= MX rows); the layer binding is sized for the prefill
            // attention chunk (pf_attn_rows, 1024 served) -> MX rows here, not y.ab.max_rows (78 x 8 views x 14 MB = 8.9 GB/rank)
            v->max_rows = MX;
            G6Attn& a = v->args;
            a.rows = MX;
            a.context = y.ab.capacity;
            a.table_size = s->run_pages;
            a.page_count = p->args.page_count;
            a.block_table = dmalloc<int>(s->run_pages);
            a.indices = dmalloc<int>((size_t)MX * 2048);
            a.output = dmalloc<__half>((size_t)MX * a.heads * 256);
            a.error = dmalloc<unsigned>(1);
            vbytes += (size_t)s->run_pages * 4 + (size_t)MX * 2048 * 4 + (size_t)MX * a.heads * 256 * 2 + 4;
#if defined(G6_CP) && G6_CP
            a.cp_nr = 0; a.cp_rank = 0;   // S2 below R: replicated path (a no-op on non-CP bindings)
#endif
            G6RT_CK(cudaMemset(a.indices, 0xff, (size_t)MX * 2048 * 4));
            s->sab[i][q] = v;
            tabs[(size_t)q * m.nl + i] = a.block_table;
            if (!s->batch) er[(size_t)q * m.nl + i] = a.error;
        }
        if (s->batch)   // the layer's batch view: all rows of a step, output straight into cat, error word of slot 0's entry
        {
            s->views.emplace_back(new G6AttnBinding());
            G6AttnBinding* v = s->views.back().get();
            *v = *p;
            v->capacity = y.ab.capacity;
            v->storage_bytes = 0;
            v->max_rows = MX;
            G6Attn& a = v->args;
            a.rows = MX;
            a.context = y.ab.capacity;
            a.table_size = s->run_pages;
            a.page_count = p->args.page_count;
            a.block_table = s->sab[i][0]->args.block_table;   // never read: every row has its own table (G6AttnRow)
            a.indices = dmalloc<int>((size_t)MX * 2048);
            a.output = s->cat;
            a.error = dmalloc<unsigned>(1);
            vbytes += (size_t)MX * 2048 * 4 + 4;
#if defined(G6_CP) && G6_CP
            a.cp_nr = 0; a.cp_rank = 0;
#endif
            G6RT_CK(cudaMemset(a.indices, 0xff, (size_t)MX * 2048 * 4));
            s->bv[i] = v;
            er[i] = a.error;
        }
    }
    if (s->batch)
    {
        s->rowcells = dmalloc<G6AttnRow>((size_t)m.nl * MX);
        s->bcells = dmalloc<G6AttnStepCell>(m.nl);
        s->tabs = dmalloc<const int*>(tabs.size());
        std::vector<G6AttnStepCell> bc(m.nl);
        memset(bc.data(), 0, bc.size() * sizeof(G6AttnStepCell));
        for (int l = 0; l < m.nl; ++l) bc[l].rows = s->rowcells + (size_t)l * MX;
        G6RT_CK(g6h2d::copy(s->bcells, bc.data(), bc.size() * sizeof(G6AttnStepCell)));
        G6RT_CK(g6h2d::copy(s->tabs, tabs.data(), tabs.size() * sizeof(const int*)));
    }
    m.s2 = s;
    for (int q = 0; q < nslots; ++q) s2_write_table(m, q);
    G6RT_CK(g6h2d::copy(s->errs, er.data(), er.size() * sizeof(unsigned*)));
    G6RT_CK(cudaStreamSynchronize(st));
    if (m.rank == 0)
        fprintf(stderr, "# s2: %d session slots, runs of at most %d pages in a pool of %d pages (%s), %d layers: %.3f GB pool storage, %.3f GB new slot views (%d rows), attention %s\n",
                nslots, s->run_pages, s->pool_pages, fits ? "contiguous runs, base = slot x run" : "slots unbound until s2_set_run", m.nl,
                bytes / 1e9, vbytes / 1e9, MX, s->batch ? "batched (one call per layer)" : "per session");
    m.s2on = true;
}
// new sequences: every view's index list / output / error word (the pool caches: reset(), the layers' own bindings)
inline void s2_reset(Model& m, cudaStream_t st)
{
    for (auto& v : m.s2->views)
    {
        const G6Attn& a = v->args;
        G6RT_CK(cudaMemsetAsync(a.indices, 0xff, (size_t)v->max_rows * 2048 * 4, st));
        G6RT_CK(cudaMemsetAsync(a.output, 0, (size_t)v->max_rows * a.heads * 256 * 2, st));
        G6RT_CK(cudaMemsetAsync(a.error, 0, 4, st));
    }
}

// a new session's run (base_page, pages) on the layers' own bindings: today's reset() limited to that run (other sessions'
// runs stay), plus the bindings' per-step words. The whole pool at base 0 (a long session, S2 below R) = today's reset().
// Host, at an admission fence, on every rank, main AND draft model, before the session's prefill.
inline void s2_reset_run(Model& m, int base_page, int pages, cudaStream_t st)
{
    const S2Model& s = *m.s2;
    if (base_page == 0 && pages >= s.pool_pages) { reset(m, st); return; }
    if (pages < 1 || base_page < 0 || base_page > s.pool_pages - pages) die("s2_reset_run: run outside the pool");
    for (G6AttnBinding* p : s.pool)
    {
        const G6Attn& a = p->args;
        G6RT_CK(cudaMemsetAsync((char*)a.mla_cache + (size_t)base_page * 64 * 656, 0, (size_t)pages * 64 * 656, st));
        if (a.index_cache) G6RT_CK(cudaMemsetAsync((char*)a.index_cache + (size_t)base_page * 64 * 132, 0, (size_t)pages * 64 * 132, st));
        G6RT_CK(cudaMemsetAsync(a.indices, 0xff, (size_t)p->max_rows * 2048 * 4, st));
        G6RT_CK(cudaMemsetAsync(a.output, 0, (size_t)p->max_rows * a.heads * 256 * 2, st));
        G6RT_CK(cudaMemsetAsync(a.error, 0, 4, st));
    }
}

// Variable runs: a session's run is (base_page, pages), 1 <= pages <= run_pages, inside the
// pool. Host, at a round boundary, outside any capture, no step in flight; on every rank, main AND draft model.
// Table contents only (fixed addresses): replay entries and graphs stay valid. The driver fills S2In::base from
// S2Model::base. The pages are not cleared (the scheduler's prefill / decode writes every row the session reads).
// Entries past the run are -1 (inside the table allocation), so a step reading a page in [pages, run_pages) fails with
// attention error bit 2; past run_pages the existing bounds fire instead (replay validation error 32, page() error 1).
// The CALLER keeps a session's context <= its granted pages x 64 before the launch. Overlap between runs is not checked
// (the allocator's).
inline void s2_check_run(const S2Model& s, int base_page, int pages, const char* who)
{
    if (pages < 1 || pages > s.run_pages || base_page < 0 || base_page > s.pool_pages - pages)
        die(std::string(who) + ": run (base " + std::to_string(base_page) + ", " + std::to_string(pages) + " pages) outside the pool or over run_pages");
}
inline void s2_set_run(Model& m, int q, int base_page, int pages)
{
    S2Model& s = *m.s2;
    if (q < 0 || q >= s.nslots) die("s2_set_run: slot out of range");
    s2_check_run(s, base_page, pages, "s2_set_run");
    s.base[q] = base_page; s.pages[q] = pages;
    s2_write_table(m, q);
}
// re-base slot q's run, keeping its pages (s2_relocate)
inline void s2_set_base(Model& m, int q, int base_page)
{
    const S2Model& s = *m.s2;
    if (q < 0 || q >= s.nslots || !s.pages[q]) die("s2_set_base: slot out of range or unbound (use s2_set_run)");
    s2_set_run(m, q, base_page, s.pages[q]);
}
// S=1 on today's path with its run (base_page, pages) (fresh, or the 2 -> 1 survivor's run): the layers' own tables =
// base + i over the run, -1 over the rest of the pool. Same rules as s2_set_run; the driver then sets
// StepIn::base_page = base_page on every step of it.
inline void s2_solo_run(Model& m, int base_page, int pages)
{
    S2Model& s = *m.s2;
    s2_check_run(s, base_page, pages, "s2_solo_run");
    s.solo_pages = pages;
    if (m.s2_below)                 // S2 below R: base 0 = the table as built (today's path, CP map included); else base + i
    {                               // over the run, -1 over the rest of the table (all table_size entries: the CP map is longer)
        for (size_t l = 0; l < s.pool.size(); ++l)
        {
            std::vector<int> t = s.tab0[l];
            if (base_page) for (size_t i = 0; i < t.size(); ++i) t[i] = (int)i < pages ? base_page + (int)i : -1;
            G6RT_CK(g6h2d::copy(const_cast<int*>(s.pool[l]->args.block_table), t.data(), t.size() * 4));
        }
        return;
    }
    std::vector<int> t(s.pool_pages);
    for (int i = 0; i < s.pool_pages; ++i) t[i] = i < pages ? base_page + i : -1;
    for (G6AttnBinding* p : s.pool) G6RT_CK(g6h2d::copy(const_cast<int*>(p->args.block_table), t.data(), t.size() * 4));
}
// compatibility: the last s2_solo_run's pages (default run_pages, the full reservation)
inline void s2_solo_base(Model& m, int base_page) { s2_solo_run(m, base_page, m.s2->solo_pages); }

// pages [B, B + pages) -> [0, pages) of a page-major cache (page_bytes per page), overlap-safe: chunks of at most B pages
// (dst < src, a chunk's source never overlaps its destination), ascending (a chunk never reads a page an earlier chunk wrote)
inline void s2_reloc_copy(char* cache, size_t page_bytes, int B, int pages, cudaStream_t st)
{
    for (int p0 = 0; p0 < pages; p0 += B)
    {
        const int n = std::min(B, pages - p0);
        G6RT_CK(cudaMemcpyAsync(cache + (size_t)p0 * page_bytes, cache + (size_t)(B + p0) * page_bytes, (size_t)n * page_bytes,
                                cudaMemcpyDeviceToDevice, st));
    }
}
// the 2->1 transition (design decision): the survivor in slot q (base B != 0) moves to base 0, its first `pages` pages of
// every layer's MLA cache (656 B per row) and index cache (132 B per row, full-index layers), then s2_set_base(q, 0).
// Host, at a round boundary, outside capture, before any new admission; call for the main and the draft model of every rank.
inline void s2_relocate(Model& m, int q, int pages, cudaStream_t st)
{
    S2Model& s = *m.s2;
    const int B = s.base[q];
    if (B == 0) return;
    if (pages < 0 || pages > s.pages[q]) die("s2_relocate: pages out of the run");
    for (G6AttnBinding* p : s.pool)
    {
        s2_reloc_copy(static_cast<char*>(p->args.mla_cache), 64 * 656, B, pages, st);
        if (p->args.index_cache) s2_reloc_copy(static_cast<char*>(p->args.index_cache), 64 * 132, B, pages, st);
    }
    s2_set_base(m, q, 0);
}
// the draft kept row across the S=1 <-> S2 transitions: S2 keeps it per slot (hprev_b), today's path in Draft::hprev
inline void s2_hprev_to_solo(Draft& D, int q, cudaStream_t st)
{
    G6RT_CK(cudaMemcpyAsync(D.hprev, D.d.s2->hprev_b + (size_t)q * HID, HID * 2, cudaMemcpyDeviceToDevice, st));
}
inline void s2_hprev_from_solo(Draft& D, int q, cudaStream_t st)
{
    G6RT_CK(cudaMemcpyAsync(D.d.s2->hprev_b + (size_t)q * HID, D.hprev, HID * 2, cudaMemcpyDeviceToDevice, st));
}

// the bound layer of the step's row class (A1): today's arguments except the k1 split (split_m = cls) and o_proj's
// fragment source (the concatenated attention rows)
inline const bind::Layer& s2_layer(Model& m, int i, int M)
{
    S2Model& s = *m.s2;
    const auto key = std::make_tuple(i, M, s.h.cls);
    auto it = s.L.find(key);
    if (it != s.L.end()) return *it->second;
    const bind::Layer& base = layer_base(m, i, M);      // today's layer of M rows (DEC_SLOT slots, DEC_RSPLIT shares)
    Layer& y = m.layers[i];
    bind::Io io;
    io.x = m.x; io.q_a = m.q_a; io.q_a_n = m.q_a_n; io.kv_a = m.kv_a; io.q_b = m.q_b;
    io.idx_q = m.idx_q; io.idx_k = m.idx_k; io.idx_w = m.idx_w;
    io.attn = s.cat;
    io.attn_part = m.sc.part[0];
    io.x2 = m.x2; io.logits = m.logits; io.plan = bind::plan_ptrs(m.plan);
    io.mlp_part = m.sc.part[1];
    static const int fuse = getenv("G6_FUSE") ? atoi(getenv("G6_FUSE")) : 1;
    std::unique_ptr<bind::Layer> L(new bind::Layer());
    if (const char* e = bind::make_layer(y.T, M, m.sc, io, *L, fuse, s.h.cls))
        die("s2: rank " + std::to_string(m.rank) + " layer " + std::to_string(y.l) + ": " + e);
    if (L->heads != y.ab.args.heads) die("kernel binder and attention binder disagree on the head count");
#ifdef DEC_SLOT
    L->o_proj.slot = base.o_proj.slot;
    L->k1.a.slot = base.k1.a.slot;
#else
    (void)base;
#endif
    return *(s.L[key] = std::move(L));
}

// attention step of the j-th session of the step on layer i (its rows of the step's inputs, its slot's binding)
inline G6AttnStep s2_step(Model& m, int i, int j, int context)
{
    const S2Model& s = *m.s2;
    const Layer& y = m.layers[i];
    const int q = s.h.sess[j], r0 = s.h.row0[j], n = s.h.rows[j];
    G6AttnStep a{};
    a.rows = n; a.context = context;
    a.q_b = m.q_b + (size_t)r0 * y.ab.args.heads * 256;
    a.kv_a = m.kv_a + (size_t)r0 * KVA;
    a.positions = m.pos + r0;
    if (y.indexer) { a.index_q = m.idx_q + (size_t)r0 * IDXQ; a.index_k = m.idx_k + (size_t)r0 * IDXK; a.index_weight = m.idx_w + (size_t)r0 * IDXW; }
    else { a.reuse_indices = s.sab[y.index_src][q]->args.indices; a.reuse_rows = n; }
    return a;
}

// the batched attention step of layer i: every row of the step (q_b / kv_a / index inputs and positions as the step has
// them), the batch view; context is only the placeholder of the replay entry (the rows' contexts are device data)
inline G6AttnStep s2_bstep(Model& m, int i, int M)
{
    const S2Model& s = *m.s2;
    const Layer& y = m.layers[i];
    G6AttnStep a{};
    a.rows = M; a.context = M;
    a.s2_batched = 1;                 // replayed with the batched cell: the fixed-K decode kernels (decode-small.cuh)
    a.q_b = m.q_b; a.kv_a = m.kv_a; a.positions = m.pos;
    if (y.indexer) { a.index_q = m.idx_q; a.index_k = m.idx_k; a.index_weight = m.idx_w; }
    else { a.reuse_indices = s.bv[y.index_src]->args.indices; a.reuse_rows = M; }
    return a;
}
// the step's bucket: the largest of its sessions' (the entry scans that capacity for every row). The native driver sets
// S2Model::bucket; drivers that only set bk (m4sim) leave it 0
inline int s2_bucket(const S2Model& s)
{
    if (s.bucket > 0) return s.bucket;
    int b = 0;
    for (int j = 0; j < s.h.ns; ++j) b = std::max(b, s.bk[s.h.sess[j]]);
    return b;
}
inline int s2_step_rows(const S2Model& s) { return s.h.ns <= 0 ? 0 : s.h.row0[s.h.ns - 1] + s.h.rows[s.h.ns - 1]; }

inline void s2_attention(Model& m, int i, int M, cudaStream_t st)
{
    S2Model& s = *m.s2;
    if (s.batch)
    {
        if (s2_step_rows(s) != M) die("s2: the sessions' rows do not add up to the step's rows");
        auto it = s.brp.find(std::make_tuple(s2_bucket(s), i, M));
        if (it == s.brp.end()) die("s2_prepare was not called before a step");
        G6RT_CK(g6_attn_replay_run(it->second.get(), st));
        return;
    }
    const int W = m.layers[i].ab.args.heads * 256;
    int n = 0;
    for (int j = 0; j < s.h.ns; ++j)
    {
        G6AttnBinding* b = s.sab[i][s.h.sess[j]];
        const G6AttnStep a = s2_step(m, i, j, s.h.ctx[j]);
        G6RT_CK(g6_attn_bind_run(b, &a, st));
        G6RT_CK(cudaMemcpyAsync(s.cat + (size_t)s.h.row0[j] * W, b->args.output, (size_t)s.h.rows[j] * W * 2, cudaMemcpyDeviceToDevice, st));
        n += s.h.rows[j];
    }
    if (n != M) die("s2: the sessions' rows do not add up to the step's rows");
}

// once per step layout, outside any capture (the driver sets s2->h and s2->bk first): bound layers of the class and the
// replay entries (batched: one per layer; per session: none, the calls are plain)
inline void s2_prepare(Model& m, int M, cudaStream_t st)
{
    S2Model& s = *m.s2;
    head_for(m, M);
    for (int i = 0; i < m.nl; ++i)
    {
        s2_layer(m, i, M);
        if (s.batch)
        {
            const int bk = s2_bucket(s);
            const auto key = std::make_tuple(bk, i, M);
            if (s.brp.count(key)) continue;
            std::unique_ptr<G6AttnReplay> r(new G6AttnReplay());
            memset(r.get(), 0, sizeof *r);
            const G6AttnStep a = s2_bstep(m, i, M);
            G6RT_CK(g6_attn_bind_replay_init(s.bv[i], &a, s.bcells + i, m.bk[bk], r.get(), st));
            s.brp[key] = std::move(r);
        }
    }
}

inline void s2_launch_cells(Model& m, cudaStream_t st)
{
    const S2Model& s = *m.s2;
    k_s2_cells<<<1, 128, 0, st>>>(S2Cells{reinterpret_cast<const uint32_t*>(s.in), s.cells, m.shared, m.nl, s.bcells, s.rowcells,
                                          s.tabs, MX});
}
inline StepEnd s2_step_end(Model& m, StepEnd e)
{
    e.errs = m.s2->errs;
    e.nl = m.nl * S2N;
    return e;
}

// sessions in the current S2 step (rank.cuh launch_draft_step: a chain step has one row per session)
inline int s2_sessions(const Model& m) { return m.s2->h.ns; }
// draft model: the eh_proj hidden rows of the step, gathered into hcat. Device-driven (S2In), so a captured graph holds
// for any layout: output row b of hcat = the main model's x row src0[j] + (b - row0[j]) (first form) or session j's kept
// row (chain: one row per session). grid MX, one block per row, rows past the step's last row return.
struct S2Hid
{
    const uint32_t* in;
    __half* hcat;
    const __half* hprev_b;
    const __half* main_x;
    int chain;
};
__global__ void __launch_bounds__(256) k_s2_draft_hid(S2Hid a)
{
    __shared__ S2In sh;
    const int t = threadIdx.x;
    if (t < (int)(sizeof(S2In) / 4)) reinterpret_cast<uint32_t*>(&sh)[t] = __ldcv(a.in + t);
    __syncthreads();
    const int b = blockIdx.x;
    int j = 0;
    while (j < sh.ns && b >= sh.row0[j] + sh.rows[j]) ++j;
    if (j >= sh.ns) return;
    const __half* src = a.chain ? a.hprev_b + (size_t)sh.sess[j] * HID : a.main_x + (size_t)(sh.src0[j] + b - sh.row0[j]) * HID;
    const uint4* s4 = reinterpret_cast<const uint4*>(src);
    uint4* d4 = reinterpret_cast<uint4*>(a.hcat + (size_t)b * HID);
    for (int i = t; i < HID / 8; i += 256) d4[i] = s4[i];
}
inline const __half* s2_draft_hid(Draft& D, bool chain, cudaStream_t st)
{
    S2Model& s = *D.d.s2;
    k_s2_draft_hid<<<MX, 256, 0, st>>>(S2Hid{reinterpret_cast<const uint32_t*>(s.in), s.hcat, s.hprev_b, D.main_x, chain ? 1 : 0});
    return s.hcat;
}
// draft model: every session keeps its last row (the next chain step's hidden). grid S2N, block j = the j-th session
struct S2Keep
{
    const uint32_t* in;
    const __half* x;
    __half* hprev_b;
};
__global__ void __launch_bounds__(256) k_s2_draft_keep(S2Keep a)
{
    __shared__ S2In sh;
    const int t = threadIdx.x;
    if (t < (int)(sizeof(S2In) / 4)) reinterpret_cast<uint32_t*>(&sh)[t] = __ldcv(a.in + t);
    __syncthreads();
    const int j = blockIdx.x;
    if (j >= sh.ns) return;
    const uint4* s4 = reinterpret_cast<const uint4*>(a.x + (size_t)(sh.row0[j] + sh.rows[j] - 1) * HID);
    uint4* d4 = reinterpret_cast<uint4*>(a.hprev_b + (size_t)sh.sess[j] * HID);
    for (int i = t; i < HID / 8; i += 256) d4[i] = s4[i];
}
inline void s2_draft_keep(Draft& D, cudaStream_t st)
{
    S2Model& s = *D.d.s2;
    k_s2_draft_keep<<<S2N, 256, 0, st>>>(S2Keep{reinterpret_cast<const uint32_t*>(s.in), D.d.x, s.hprev_b});
}

// after the step's stream is synchronized: step_line over the rows r0..r0+n-1 of a step (tokens, logprobs, target
// logprobs, residual hash of those rows), so a session's line in a batched step compares with its solo line
inline int s2_step_line(Model& m, const StepOut& o, int s, int r0, int n, std::string& line)
{
    std::vector<float> r((size_t)n * HID);
    G6RT_CK(cudaMemcpy(r.data(), m.resid + (size_t)r0 * HID, r.size() * 4, cudaMemcpyDeviceToHost));
    char b[96];
    snprintf(b, sizeof b, "step %d M %d tok", s, n);
    line = b;
    for (int i = r0; i < r0 + n; ++i) { snprintf(b, sizeof b, " %d", o.tok[i]); line += b; }
    line += " lp";
    for (int i = r0; i < r0 + n; ++i)
    {
        uint32_t u;
        memcpy(&u, &o.lp[i], 4);
        snprintf(b, sizeof b, " %.5f/%08x", o.lp[i], u);
        line += b;
    }
    line += " lpt";
    for (int i = r0; i < r0 + n; ++i)
    {
        uint32_t u;
        memcpy(&u, &o.lpt[i], 4);
        snprintf(b, sizeof b, " %.5f/%08x", o.lpt[i], u);
        line += b;
    }
    snprintf(b, sizeof b, " resid %016llx", (unsigned long long)fnv(0xcbf29ce484222325ull, r.data(), r.size() * 4));
    line += b;
    return o.tok[r0 + n - 1];
}

}   // namespace g6rt
