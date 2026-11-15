// CPU-only lifecycle regressions for deferred prompt-end replacement. Run each knob mode in a fresh process.
// State copies use host-memory CUDA substitutes; these are bookkeeping tests, not GPU numerical or speed claims.
#include "ds41_prefix_cache_host.h"
using g6s2::Ds41Snap;
using g6s2::Ds41SnapSlots;

static std::vector<int> tokens(int salt = 0) {
    std::vector<int> v(2048);
    for (size_t i = 0; i < v.size(); ++i) v[i] = 1 + int(i) + salt * 10000;
    return v;
}
static size_t prompts(const Ds41PrefixCache& c) {
    return std::count_if(c.entries().begin(), c.entries().end(), [](const auto& e) { return e.kind == Ds41Snap::PromptEnd; });
}
static void accounting(const Ds41PrefixCache& c) {
    size_t bytes = 0;
    for (const auto& e : c.entries()) {
        bytes += e.bytes;
        CHECK(mem.count(e.dev_buf) == 1);
        if (e.sibling && !(e.sibling >> 63))
            if (const auto* other = c.find(e.sibling)) CHECK(other->sibling == e.generation);
    }
    CHECK(bytes == c.used_bytes() && bytes <= c.budget_bytes());
}
static void canonical_then_divergent(bool fp4, bool park) {
    Fixture f(fp4); Ds41PrefixCache c(f.m, nullptr, 64); Ds41SnapSlots s(2);
    const bool defer = c.defer_replace_on(), replace = knob::ds_cache_replace();
    auto t = tokens();
    uint64_t pe = 0, re = 0;
    uint32_t pp = 0, rp = 0;
    for (int turn = 0; turn < 12; ++turn) {
        if (turn) {
            const auto before = c.siblings();
            if (park) s.resume(c, 0, rp, t, ns(1));
            else { s.reset(0); CHECK(s.restore(c, re, 0, 0) == Ds41Snap::RequestEnd); }
            if (replace) {
                CHECK(!c.find(pe));
                CHECK(c.siblings() == before + uint64_t(defer || turn == 1));
                CHECK(prompts(c) == 0);
            }
        }
        pp = 100 + turn * 140; rp = pp + 70; // past any short-reply dedup window
        pe = s.capture_prompt_end(c, 0, pp, t, ns(1), 0);
        CHECK(pe);
        if (park) s.park(c, 0, rp, t, ns(1));
        else re = s.capture_request_end(c, 0, rp, t, ns(1), 0);
        CHECK(bool(c.find(pe)) == (defer || !replace || turn == 0));
        if (replace) CHECK(prompts(c) <= 1);
        accounting(c);
    }
    auto next = t; next[pp + 10] = -7; // BPE divergence inside the reply, not the prompt
    const auto hit = c.match(next, int(rp + 10), ns(1));
    if (defer || !replace) {
        CHECK(hit.hit && hit.generation == pe && hit.position == pp);
        const auto captures = c.captures();
        if (park) {
            const auto gen = s.evict_parked(c, 0, rp, t, ns(1), 0, pe);
            if (replace) CHECK(!gen && c.captures() == captures);
        }
        CHECK(s.restore(c, pe, 1, 0) == Ds41Snap::PromptEnd);
        if (replace && !park) CHECK(!c.find(re));
    } else {
        CHECK(!c.find(pe) && hit.position < pp);
        CHECK(c.replaced() == 11);
        if (park) s.evict_parked(c, 0, rp, t, ns(1), 0, 0);
    }
    accounting(c);
    printf("canonical/divergent: 12 turns fp4=%d park=%d defer=%d replace=%d PASS\n", fp4, park, defer, replace);
}

static void thinking_drop(bool park) {
    Fixture f(false); Ds41PrefixCache c(f.m, nullptr, 64); Ds41SnapSlots s(2);
    auto t = tokens(), previous = t;
    uint64_t old_pe = 0, old_re = 0; uint32_t old_rp = 0;
    for (int turn = 0; turn < 3; ++turn) {
        if (turn) {
            t[100 + (turn - 1) * 140 + 10] = -turn; // reasoning omitted in next prompt
            const auto hit = c.match(t, 100 + turn * 140, ns(1));
            CHECK(hit.hit && hit.generation == old_pe);
            if (park) {
                const auto gen = s.evict_parked(c, 0, old_rp, previous, ns(1), 0, old_pe);
                if (knob::ds_cache_replace()) CHECK(gen == 0);
            } else s.reset(0);
            CHECK(s.restore(c, old_pe, 0, 0) == Ds41Snap::PromptEnd);
            if (!park && knob::ds_cache_replace()) CHECK(!c.find(old_re));
        }
        const uint32_t pp = 100 + turn * 140; old_rp = pp + 70;
        old_pe = s.capture_prompt_end(c, 0, pp, t, ns(1), 0);
        if (park) s.park(c, 0, old_rp, t, ns(1));
        else old_re = s.capture_request_end(c, 0, old_rp, t, ns(1), 0);
        previous = t;
        CHECK(old_pe);
    }
    if (park) s.evict_parked(c, 0, old_rp, t, ns(1), 0, 0);
    accounting(c);
    printf("thinking-drop: three turns park=%d PASS\n", park);
}

// A prior request-end resume sets restored_end; then another slot takes the parked state.
// Both possible sibling restores must free exactly once, including the already-restored PE path.
static void eviction_order(bool prompt, bool restore_first) {
    Fixture f(false); Ds41PrefixCache c(f.m, nullptr, 64); Ds41SnapSlots s(2); auto t = tokens();
    s.capture_prompt_end(c, 0, 100, t, ns(1), 0); s.park(c, 0, 200, t, ns(1));
    s.resume(c, 0, 200, t, ns(1));
    const auto pe = s.capture_prompt_end(c, 0, 300, t, ns(1), 0);
    s.park(c, 0, 400, t, ns(1));
    CHECK(c.find(pe) && c.find(pe)->sibling >> 63);
    if (restore_first) {
        CHECK(s.restore(c, pe, 1, 0) == Ds41Snap::PromptEnd);
        const auto captures = c.captures();
        CHECK(!s.evict_parked(c, 0, 400, t, ns(1), 0, 0) && captures == c.captures());
    } else {
        const auto re = s.evict_parked(c, 0, 400, t, ns(1), 0, 0);
        CHECK(re && c.find(pe)->sibling == re && c.find(re)->sibling == pe);
        auto next = t; if (prompt) next[310] = -1;
        const auto hit = c.match(next, 450, ns(1));
        CHECK(hit.generation == (prompt ? pe : re));
        const auto sibs = c.siblings();
        CHECK(s.restore(c, hit.generation, 1, 0) == (prompt ? Ds41Snap::PromptEnd : Ds41Snap::RequestEnd));
        CHECK(!c.find(prompt ? re : pe) && c.siblings() == sibs + 1);
        s.restore(c, hit.generation, 1, 0);
        CHECK(c.siblings() == sibs + 1);
    }
    for (const auto& e : c.entries()) CHECK(!(e.sibling >> 63));
    accounting(c);
    printf("park/evict/restore: prompt=%d restore_first=%d PASS\n", prompt, restore_first);
}

// Equal stored request ends, short-reply dedup, absent/evicted/zero-budget prompt entries and a competing sibling.
static void park_edges() {
    auto t = tokens();
    for (bool park : {false, true}) for (bool prompt : {false, true}) {
        Fixture f(false); Ds41PrefixCache c(f.m, nullptr, 64); Ds41SnapSlots s(2);
        const auto old = c.capture(0, 50, t, ns(1), 0, Ds41Snap::RequestEnd);
        s.restore(c, old, 0, 0);
        const auto pe = s.capture_prompt_end(c, 0, 150, t, ns(1), 0);
        const auto re = c.capture(1, 300, t, ns(1), 0, Ds41Snap::RequestEnd);
        if (park) s.park(c, 0, 300, t, ns(1)); else CHECK(s.capture_request_end(c, 0, 300, t, ns(1), 0) == re);
        if (park) CHECK((c.find(pe)->sibling >> 63) && !c.find(re)->sibling);
        else CHECK(c.find(pe)->sibling == re && c.find(re)->sibling == pe);
        if (park) {
            if (prompt) {
                // Choosing PE suppresses the park before restore as well as after it.
                auto next = t; next[160] = -1;
                CHECK(c.match(next, 350, ns(1)).generation == pe);
                const auto captures = c.captures();
                CHECK(!s.evict_parked(c, 0, 300, t, ns(1), 0, pe));
                CHECK(c.captures() == captures);
                s.restore(c, pe, 1, 0);
                CHECK(c.find(pe) && c.find(re)); // the shared pre-existing RE is independently stored
            } else { s.resume(c, 0, 300, t, ns(1)); CHECK(!c.find(pe) && c.find(re)); }
        } else {
            s.restore(c, prompt ? pe : re, 1, 0);
            CHECK(!c.find(prompt ? re : pe));
        }
        accounting(c);
    }
    // Equal RE appears after park: resume must free the original placeholder's PE, not just detach it.
    {
        Fixture f(false); Ds41PrefixCache c(f.m, nullptr, 64); Ds41SnapSlots s(2);
        auto pe = s.capture_prompt_end(c, 0, 100, t, ns(1), 0); s.park(c, 0, 200, t, ns(1));
        c.capture(1, 200, t, ns(1), 0, Ds41Snap::RequestEnd);
        s.resume(c, 0, 200, t, ns(1));
        CHECK(!c.find(pe) && prompts(c) == 0);
    }
    // An already-paired RE must keep the other request's pair; our unpaired prompt is released on resume.
    {
        Fixture f(false); Ds41PrefixCache c(f.m, nullptr, 64); Ds41SnapSlots s(2);
        auto p0 = s.capture_prompt_end(c, 0, 100, t, ns(1), 0);
        auto re = s.capture_request_end(c, 0, 300, t, ns(1), 0);
        auto p1 = s.capture_prompt_end(c, 1, 200, t, ns(1), 0);
        s.park(c, 1, 300, t, ns(1));
        CHECK(c.find(re)->sibling == p0 && (c.find(p1)->sibling >> 63));
        s.resume(c, 1, 300, t, ns(1));
        CHECK(!c.find(p0) && !c.find(p1) && c.find(re));
    }
    // Dedup folds a short RE into the PE itself: resume retains it, and treats it as PromptEnd.
    if (knob::ds_cache_dedup()) {
        Fixture f(false); Ds41PrefixCache c(f.m, nullptr, 64); Ds41SnapSlots s(2);
        auto pe = s.capture_prompt_end(c, 0, 100, t, ns(1), 0); s.park(c, 0, 101, t, ns(1));
        s.resume(c, 0, 101, t, ns(1)); CHECK(c.find(pe) && c.siblings() == 0);
    }
    for (int mode : {0, 1, 2, 3, 4}) {
        Fixture f(false); Ds41PrefixCache c(f.m, nullptr, mode == 2 ? 0 : 1); Ds41SnapSlots s(2);
        if (mode == 1) {
            auto pe = s.capture_prompt_end(c, 0, 100, t, ns(1), 0);
            for (int j = 2; j < 8; ++j) c.capture(1, 100, tokens(j), ns(1), 0);
            CHECK(!c.find(pe));
        }
        const uint32_t pos = mode == 3 ? 0 : 200;
        const auto hist = mode == 4 ? std::vector<int>{1} : t;
        s.park(c, 0, pos, hist, ns(1));
        s.evict_parked(c, 0, pos, hist, ns(1), 0, 0);
        accounting(c);
        for (const auto& e : c.entries()) CHECK(!(e.sibling >> 63));
    }
    puts("park edges: equal-entry, concurrent equality, paired entry, dedup, absent/evicted/invalid/over-budget PASS");
}
static std::vector<uint64_t> churn() {
    Fixture f(false, 32);
    // Scale 75 MiB prompt snapshots / 1 GiB budget by 1/1024. Exactly 13 snapshots fit; no giant allocations.
    f.m.layers[0].C.ring_rows = 75; f.m.layers[0].C.comp = nullptr;
    Ds41PrefixCache c(f.m, nullptr, 1); Ds41SnapSlots s(32);
    CHECK(c.entry_bytes(100) == 75 * 1024 && 32 * c.entry_bytes(100) > c.budget_bytes());
    CHECK(c.budget_bytes() / c.entry_bytes(100) == 13);
    std::array<std::vector<int>, 32> hist;
    std::array<uint32_t, 32> rp{};
    std::array<uint64_t, 32> pe{};
    size_t fallbacks = 0, cold = 0, resumes = 0;
    std::vector<uint64_t> digests;
    for (int round = 0; round < 5; ++round) for (int i = 0; i < 32; ++i) {
        int q = round % 2 ? 31 - i : i;
        auto next = round ? hist[q] : tokens(q);
        if (round) {
            if (q % 3) { s.resume(c, q, rp[q], hist[q], ns(1)); ++resumes; CHECK(!c.find(pe[q])); }
            else {
                next[rp[q] - 40] = -round;
                const auto hit = c.match(next, int(rp[q] + 10), ns(1));
                if (hit.hit) { CHECK(hit.generation == pe[q]); ++fallbacks; } else ++cold;
                const auto captures = c.captures();
                const auto re = s.evict_parked(c, q, rp[q], hist[q], ns(1), 0, hit.generation);
                if (hit.hit) {
                    CHECK(!re && c.captures() == captures);
                    CHECK(s.restore(c, hit.generation, q, 0) == Ds41Snap::PromptEnd);
                }
            }
        }
        const uint32_t pp = 100 + round * 140; rp[q] = pp + 70;
        pe[q] = s.capture_prompt_end(c, q, pp, next, ns(1), 0);
        hist[q] = next; s.park(c, q, rp[q], hist[q], ns(1));
        accounting(c);
        CHECK(prompts(c) <= 13);
        digests.push_back(c.digest() ^ c.park_digest() ^ s.digest());
    }
    for (int q = 0; q < 32; ++q) s.evict_parked(c, q, rp[q], hist[q], ns(1), 0, 0);
    CHECK(fallbacks && cold && resumes && c.evictions() > 0);
    for (const auto& e : c.entries()) CHECK(!(e.sibling >> 63));
    accounting(c);
    printf("32-slot/13-snapshot churn: resumes=%zu fallbacks=%zu cold=%zu evictions=%llu PASS\n",
           resumes, fallbacks, cold, (unsigned long long)c.evictions());
    return digests;
}
int main() {
    for (bool fp4 : {false, true}) for (bool park : {false, true}) canonical_then_divergent(fp4, park);
    thinking_drop(false); thinking_drop(true);
    if (knob::ds_cache_replace() && knob::ds_cache_defer_replace()) {
        eviction_order(false, false); eviction_order(true, false); eviction_order(true, true);
        park_edges();
        const auto a = churn(), b = churn();
        CHECK(a == b && a.size() == 160);
    }
    CHECK(mem.empty() && live_bytes == 0);
    printf("test_ds41_prefix_cache: %s (%d failures, live allocations %zu)\n", fails ? "FAIL" : "PASS", fails, mem.size());
    return fails ? 1 : 0;
}
