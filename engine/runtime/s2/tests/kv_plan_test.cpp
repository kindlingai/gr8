// kvt::next_batch (s2/kv_plan.h), the batches KvTiers::run cuts the pool's copies into: every batch takes a copy (a
// batch of 1 included), a block spilled twice in one call ends with the second page, two moves into one block end with
// the second, and random copy sequences run in batches give what they give one copy at a time. No CUDA: a batch runs
// here as KvTiers::run does it (spills: device copies, then the store writes; moves: reads, then writes; restores:
// reads, then device copies), on cells: a block's cell 0 is a page or a shard, cell 1 + k its k-th replicated part.
#include "../kv_plan.h"
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <utility>
#include <vector>

using g6s2::kvt::Batch;
using g6s2::kvt::next_batch;

static void require(bool ok, const char* what) {
    if (!ok) { fprintf(stderr, "FAIL: %s\n", what); exit(1); }
}

using Cells = std::vector<long>;   // -1: not written
struct Sim {
    uint32_t per;
    std::map<std::pair<uint32_t, uint64_t>, Cells> store;
    std::vector<long> dev;
    int batches = 0;
    explicit Sim(uint32_t per_slot, int pages) : per(per_slot), dev(pages) { for (int p = 0; p < pages; ++p) dev[p] = 1000 + p; }
    size_t cell(const g6kv_io& x) const { return x.part == 2 ? 1 + x.sub : 0; }
    Cells& block(uint32_t t, uint64_t s) { auto& b = store[{t, s}]; if (b.empty()) b.assign(per + 1, -1); return b; }
    // one copy at a time: a spill writes its block whole (what a batch's host slot held), so a block's copies are one
    // write; the caller passes them together
    void one(const std::vector<g6kv_io>& io) {
        for (size_t i = 0; i < io.size();) {
            const g6kv_io& x = io[i];
            if (x.kind == 0) {
                Cells c(per + 1, -1);
                size_t j = i;
                for (; j < io.size() && io[j].kind == 0 && io[j].tier == x.tier && io[j].slot == x.slot && (j == i || (io[j].part == 2 && io[j].sub > 0)); ++j)
                    c[cell(io[j])] = dev[io[j].page];
                block(x.tier, x.slot) = c; i = j;
            } else if (x.kind == 1) { block(x.to, x.to_slot) = block(x.tier, x.slot); ++i; }
            else { dev[x.page] = block(x.tier, x.slot)[cell(x)]; ++i; }
        }
    }
    void batched(const std::vector<g6kv_io>& io, int batch, int items_cap) {
        for (size_t i = 0; i < io.size();) {
            const size_t i0 = i;
            const Batch b = next_batch(io, i, batch, per, items_cap);
            require(i > i0, "a batch takes a copy");
            require((int)b.slots.size() <= batch && (int)b.items.size() <= items_cap, "a batch within its bounds");
            ++batches;
            std::vector<Cells> host(b.slots.size(), Cells(per + 1, -1));
            std::map<std::pair<int, size_t>, int> written;   // a cell two copies of one batch write: no order
            if (b.kind == 0) {
                for (const auto& it : b.items) {
                    const size_t c = it.part == 2 ? 1 + it.sub : 0;
                    require(++written[{it.hslot, c}] == 1, "two copies of a spill batch into one host cell");
                    host[it.hslot][c] = dev[it.page];
                }
                for (size_t j = 0; j < b.slots.size(); ++j) {
                    for (size_t k = 0; k < j; ++k) require(!(b.from[k] == b.from[j] && b.slots[k] == b.slots[j]), "one block written twice by a batch");
                    block(b.from[j], b.slots[j]) = host[j];
                }
            } else if (b.kind == 1) {
                for (size_t j = 0; j < b.slots.size(); ++j) host[j] = block(b.from[j], b.slots[j]);
                for (size_t j = 0; j < b.slots.size(); ++j) {
                    for (size_t k = 0; k < j; ++k) require(!(b.to[k] == b.to[j] && b.to_slots[k] == b.to_slots[j]), "one block written twice by a batch");
                    block(b.to[j], b.to_slots[j]) = host[j];
                }
            } else {
                for (size_t j = 0; j < b.slots.size(); ++j) host[j] = block(b.from[j], b.slots[j]);
                for (const auto& it : b.items) {
                    require(++written[{it.page, 0}] == 1, "two copies of a restore batch into one page");
                    dev[it.page] = host[it.hslot][it.part == 2 ? 1 + it.sub : 0];
                }
            }
        }
    }
};

static g6kv_io spill(uint32_t page, uint64_t slot, uint32_t part = 0, uint32_t sub = 0) { return {0, page, 0, 0, slot, 0, part, sub}; }
static g6kv_io restore(uint32_t page, uint64_t slot, uint32_t part = 0, uint32_t sub = 0) { return {2, page, 0, 0, slot, 0, part, sub}; }
static g6kv_io move(uint64_t slot, uint32_t to, uint64_t to_slot) { return {1, 0, 0, to, slot, to_slot, 0, 0}; }

int main() {
    // a batch of 1 with replicated parts (per_slot 3, items 1 x 3): every block a batch of its own
    {
        Sim s(3, 8);
        const std::vector<g6kv_io> io = {spill(0, 0), spill(1, 1, 1), spill(2, 2, 2, 0), spill(3, 2, 2, 1), spill(4, 2, 2, 2),
                                         restore(5, 0), restore(6, 1, 1), restore(7, 2, 2, 2)};
        s.batched(io, 1, 3);
        require(s.batches == 6, "batch 1: a batch per block");
        require(s.dev[5] == 1000 && s.dev[6] == 1001 && s.dev[7] == 1004, "batch 1: restores");
    }
    // one slot spilled twice in one call (freed and taken again): the block holds the second page (B-E1)
    for (uint32_t per : {0u, 2u}) {
        Sim s(per, 4);
        s.batched({spill(0, 0), spill(1, 0), restore(2, 0)}, 32, 32 * int(per ? per : 1));
        require(s.dev[2] == 1001, "a slot spilled twice holds the second page");
        require(s.batches == 3, "the second spill of a slot starts a batch");
    }
    // a tail's replicated parts of one block share a host slot; the next block's sub 0 starts its own
    {
        Sim s(2, 8);
        s.batched({spill(0, 4, 2, 0), spill(1, 4, 2, 1), spill(2, 5, 2, 0), restore(5, 4, 2, 1), restore(6, 5, 2, 0)}, 32, 64);
        require(s.dev[5] == 1001 && s.dev[6] == 1002, "replicated parts");
        require(s.batches == 2, "replicated parts in one spill batch");
    }
    // two moves into one block: the second wins
    {
        Sim s(0, 4);
        s.batched({spill(0, 0), spill(1, 1), move(0, 0, 9), move(1, 0, 9), restore(2, 9)}, 32, 32);
        require(s.dev[2] == 1001, "two moves into one block: the second");
    }
    // one block restored into two pages: one read
    {
        Sim s(0, 4);
        s.batched({spill(0, 3), restore(1, 3), restore(2, 3)}, 32, 32);
        require(s.dev[1] == 1000 && s.dev[2] == 1000 && s.batches == 2, "one block restored twice");
    }
    // random calls: batched = one copy at a time, for batch sizes 1..5 and per_slot 0..3
    std::mt19937 rng(7);
    long calls = 0;
    for (int round = 0; round < 4000; ++round) {
        const uint32_t per = rng() % 4;
        const int batch = 1 + rng() % 5, pages = 24;
        const int cap = batch * int(per ? per : 1);
        Sim a(per, pages), b(per, pages);
        std::vector<g6kv_io> io;
        for (int run = 0, n = 1 + rng() % 4; run < n; ++run) {
            const uint32_t kind = rng() % 3;
            for (int k = 0, m = 1 + rng() % 6; k < m; ++k) {
                const uint64_t slot = rng() % 4;
                if (kind == 0) {
                    if (per && rng() % 2) for (uint32_t sub = 0, r = 1 + rng() % per; sub < r; ++sub) io.push_back(spill(rng() % pages, slot, 2, sub));
                    else io.push_back(spill(rng() % pages, slot, rng() % 2));
                } else if (kind == 1) {
                    io.push_back(move(slot, 0, rng() % 4));
                } else {
                    const uint32_t part = per ? rng() % 3 : rng() % 2;
                    io.push_back(restore(rng() % pages, slot, part, part == 2 ? rng() % per : 0));
                }
            }
        }
        // a restore batch writes a page once: drop a restore into a page an earlier restore of its run writes
        std::vector<g6kv_io> ok;
        std::vector<int> seen(pages, -1);
        for (size_t i = 0, run = 0; i < io.size(); ++i) {
            if (i && io[i].kind != io[i - 1].kind) ++run;
            if (io[i].kind == 2 && seen[io[i].page] == (int)run) continue;
            if (io[i].kind == 2) seen[io[i].page] = (int)run;
            ok.push_back(io[i]);
        }
        io.swap(ok);
        a.one(io);
        b.batched(io, batch, cap);
        require(a.store == b.store && a.dev == b.dev, "batched copies = one at a time");
        ++calls;
    }
    printf("kv_plan_test: ok (%ld random calls)\n", calls);
    return 0;
}
