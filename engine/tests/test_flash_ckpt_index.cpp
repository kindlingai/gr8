// G6_FLASH_CACHE's host half (s2/flash_checkpoint.h): the aligned point, the match rule (tokens through p, the limit,
// the namespace, the longest and then the newest entry), first-fit placement with least-recently-used drops, the digest
// two ranks compare, and the manifest checks that refuse a torn or reused range. No CUDA, no store.
#include "../runtime/s2/flash_checkpoint.h"
#include <cassert>
#include <cstdio>
#include <numeric>
using namespace g6s2;

static std::vector<int> seq(int n, int salt = 0) {
    std::vector<int> v(n);
    for (int i = 0; i < n; ++i) v[i] = (i * 7 + salt) % 1000 + 1;
    return v;
}
static FlashEntry entry(FlashIndex& ix, uint32_t p, const std::vector<int>& prompt, uint64_t blocks,
                        const g6rt::CacheNamespace& ns = {}) {
    FlashEntry x;
    uint64_t at = 0;
    assert(ix.reserve(blocks, at));
    x.generation = ix.next_generation(); x.base = at; x.blocks = blocks; x.position = p; x.ns = ns;
    x.tokens.assign(prompt.begin(), prompt.begin() + p + 1);
    x.manifest = flash_manifest(x.generation, 7, p, x.tokens, ns, p / 64, 100, 50);
    x.manifest.seal();
    return x;
}
int main() {
    // the aligned point: the last full-quantum end at or before prompt_tokens - 1 rows
    // the modes
    assert(flash_cache_mode(nullptr) == FlashCache::Off && flash_cache_mode("off") == FlashCache::Off);
    assert(flash_cache_mode("aligned") == FlashCache::Aligned && flash_cache_mode("terminal") == FlashCache::Terminal);
    assert(std::string(flash_cache_name(FlashCache::Terminal)) == "terminal");
    try { flash_cache_mode("sessions"); assert(false); } catch (const std::runtime_error&) {}
    assert(aligned_point(8193, 8192) == 8192);   // 8192 prefill rows + the prompt tail
    assert(aligned_point(8192, 8192) == 0);      // 8191 rows: one partial quantum
    assert(aligned_point(20000, 8192) == 16384);
    assert(aligned_point(100, 0) == 0 && aligned_point(0, 64) == 0);
    for (uint32_t n = 1; n < 40000; n += 97) {
        const uint32_t a = aligned_point(n, 8192);
        assert(a % 8192 == 0 && a <= n - 1 && n - 1 - a < 8192);
    }

    FlashIndex ix(1000);
    const auto A = seq(30000), B = seq(30000, 3);
    ix.add(entry(ix, 8192, A, 200));
    ix.add(entry(ix, 16384, A, 300));
    // the longest entry that the prompt continues through token p, p <= limit
    FlashHit h = ix.match(A, 29999, {});
    assert(h.hit && h.position == 16384 && h.generation == 2);
    h = ix.match(A, 16383, {});   // the limit cuts the longer entry: the shorter one serves
    assert(h.hit && h.position == 8192 && h.generation == 1);
    h = ix.match(A, 8191, {});
    assert(!h.hit && std::string(h.why) == "no-match");
    // token p must match too (the MTP layer's row p - 1 reads it): a prompt that differs exactly at p misses
    auto A2 = A; A2[16384] += 1;
    h = ix.match(A2, 29999, {});
    assert(h.hit && h.position == 8192);
    auto A3 = A; A3[8192] += 1;
    h = ix.match(A3, 29999, {});
    assert(!h.hit && h.lcp == 8192);
    // a prompt that ends at p: the tokens are not all there
    h = ix.match(std::vector<int>(A.begin(), A.begin() + 8192), 8191, {});
    assert(!h.hit);
    h = ix.match(B, 29999, {});
    assert(!h.hit);
    // the namespace must be the same
    g6rt::CacheNamespace ns{}; ns[0] = 1;
    h = ix.match(A, 29999, ns);
    assert(!h.hit);
    ix.add(entry(ix, 8192, A, 200, ns));
    h = ix.match(A, 29999, ns);
    assert(h.hit && h.generation == 3 && h.position == 8192);
    // same(): a prefix kept again is the old entry
    assert(ix.same(std::vector<int>(A.begin(), A.begin() + 8193), {})->generation == 1);
    assert(!ix.same(std::vector<int>(A.begin(), A.begin() + 8193), g6rt::CacheNamespace{2}));

    // placement: 1 at [0,200), 2 at [200,500), 3 at [500,700), [700,1000) free. 1 is touched, so 2 and then 3 are the
    // least recently used: 400 blocks drop 2 (two gaps of 300, neither fits), then 3 ([200,1000) fits at 200)
    ix.touch(1);
    std::vector<uint64_t> gone;
    uint64_t at = 0;
    assert(ix.reserve(400, at, &gone));
    assert(gone.size() == 2 && gone[0] == 2 && gone[1] == 3 && at == 200 && ix.size() == 1);
    assert(!ix.find(2) && !ix.find(3) && ix.find(1));
    // a reserve that fits drops nothing
    gone.clear();
    assert(ix.reserve(100, at, &gone) && gone.empty() && at == 200);
    // the store smaller than the entry: nothing dropped
    gone.clear();
    const size_t before = ix.size();
    assert(!ix.reserve(1001, at, &gone) && gone.empty() && ix.size() == before);

    // two ranks that make the same calls agree; a different call changes the digest
    FlashIndex r0(5000), r1(5000);
    for (FlashIndex* r : {&r0, &r1}) {
        r->add(entry(*r, 8192, A, 200));
        r->add(entry(*r, 8192, B, 200));
        r->touch(1);
    }
    assert(r0.digest() == r1.digest());
    r1.touch(2);
    assert(r0.digest() != r1.digest());

    // the manifest: the entry's own, sealed, is accepted; torn, reused or edited is not
    FlashManifest m = flash_manifest(5, 9, 8192, std::vector<int>(A.begin(), A.begin() + 8193), {}, 128, 1277952, 76316672);
    FlashManifest want = m; want.seal();
    FlashManifest got = want;
    assert(!got.differs(want));
    FlashManifest torn = m;   // never sealed: the data may be there, the manifest is not
    assert(torn.differs(want));
    FlashManifest zero{};
    assert(std::string(zero.differs(want)) == "no manifest");
    FlashManifest other = m; other.generation = 4; other.seal();
    assert(std::string(other.differs(want)) == "another entry");
    FlashManifest flipped = want; flipped.page_sum ^= 1;   // a bit changed after sealing
    assert(std::string(flipped.differs(want)) == "manifest checksum");
    FlashManifest compat = m; compat.compat = 10; compat.seal();
    assert(std::string(compat.differs(want)) == "another engine or layout");

    // a group lease (D-331 M8 (c)): the counters in the pool tier's form, the ages a give costs, grow, and a shrink that
    // drops the least recently used and packs the rest below the new end (moves down, in base order)
    {
        uint64_t t_ms = 1000;
        FlashIndex g(100);
        g.now_ms = [&] { return t_ms; };
        const auto P = seq(20000, 9);
        std::vector<uint64_t> gens;
        for (int i = 0; i < 4; ++i) {   // bases 0, 20, 40, 60, used at 1000, 1010, 1020, 1030 ms
            auto Pi = P; Pi[0] = 5000 + i;
            FlashEntry x = entry(g, 64, Pi, 20);
            gens.push_back(x.generation);
            g.add(std::move(x));
            t_ms += 10;
        }
        g6kv_tier_totals t = g.totals();
        assert(t.slots == 100 && t.free_slots == 20 && t.entries == 4 && t.spilled_pages == 80 && t.full == 0 && t.dropped_pages == 0);
        t_ms = 2000;
        g.touch(gens[0]);   // gens[1] is now the least recently used
        // segments of 10 blocks: 20 free cover 2, then gens[1] (age 990), gens[2] (980), gens[3] (970), gens[0] (0)
        const auto ages = g.coldest(10, 8);
        const std::vector<uint64_t> want_ages = {UINT64_MAX, UINT64_MAX, 990, 990, 980, 980, 970, 970};
        assert(ages == want_ages);
        uint64_t at = 0;
        assert(!g.reserve(150, at) && g.oversize == 1 && g.oversize_last == 150);
        {   // 30 contiguous blocks: gens[1] (20..40) and then gens[2] (40..60) go; one capture that had to drop
            FlashIndex f = g;
            assert(f.reserve(30, at) && at == 20 && f.full == 1 && f.dropped_blocks == 40 && f.totals().dropped_pages == 40);
            assert(!f.find(gens[1]) && !f.find(gens[2]) && f.totals().full == 1);
        }
        // shrink to 50 blocks (80 held): gens[1] and gens[2] go (least recently used first), gens[0] at 0 stays and
        // gens[3] moves from 60 to 20
        FlashIndex h = g;
        std::vector<uint64_t> gone;
        const auto moves = g.shrink(50, &gone);
        assert((gone == std::vector<uint64_t>{gens[1], gens[2]}) && g.blocks() == 50 && g.used_blocks() == 40);
        assert(moves.size() == 1 && moves[0].from == 60 && moves[0].to == 20 && moves[0].blocks == 20);
        assert(g.find(gens[0])->base == 0 && g.find(gens[3])->base == 20);
        // the same shrink on another rank's copy: the same drops, moves and digest
        std::vector<uint64_t> gone2;
        assert(h.shrink(50, &gone2).size() == 1 && gone2 == gone && h.digest() == g.digest());
        // a shrink that every entry fits below needs no moves; a grow keeps the entries and changes the digest
        assert(g.shrink(40).empty() && g.blocks() == 40 && g.size() == 2);
        const uint64_t before = g.digest();
        g.grow(80);
        assert(g.blocks() == 80 && g.size() == 2 && g.digest() != before);
        assert(g.reserve(40, at) && at == 40);   // the grown blocks are used
        try { g.grow(10); assert(false); } catch (const std::runtime_error&) {}
    }

    // the checksum moves with the word's place
    uint64_t w[4] = {1, 2, 3, 4}, v[4] = {2, 1, 3, 4};
    assert(ckpt_sum_host(w, 32) != ckpt_sum_host(v, 32));
    assert(ckpt_sum_host(w, 16) + ckpt_sum_host(w + 2, 16, 2) == ckpt_sum_host(w, 32));
    puts("flash_ckpt_index: PASS (aligned point, match rule, namespace, placement and drops, digest, manifest checks, "
         "group lease totals, ages, grow, shrink with drops and packing)");
}
