// PoolOps (g6sched.h g6s2_pool_ops), the C++ side: a module's policy over its rank's RunPool, as ModuleOps is for its
// other calls (module.h). The trampolines a module fills its table with (pol::match<C, &C::on_match>), PoolPolicy (the
// caller's side, which rethrows what the module threw) and FnPool (a table over the service's std::function pool hooks).
// A module's pool context derives from ModuleCtx, which holds the exception.
#pragma once
#include "module.h"
#include "pool.h"
#include <array>
#include <optional>
#include <string>
#include <vector>

namespace g6s2 {
static_assert(sizeof(PoolHit) == sizeof(g6s2_pool_hit) && offsetof(PoolHit, why) == offsetof(g6s2_pool_hit, why) &&
              offsetof(PoolHit, age) == offsetof(g6s2_pool_hit, age) && offsetof(PoolHit, stored) == offsetof(g6s2_pool_hit, stored) &&
              offsetof(PoolHit, tier) == offsetof(g6s2_pool_hit, tier) && offsetof(PoolHit, slot) == offsetof(g6s2_pool_hit, slot),
              "PoolHit layout");

namespace pol {
using mod::as;
inline std::vector<int> ints(const int32_t* p, size_t n) { return std::vector<int>(p, p + n); }
/// A table for context c and the rank's pool, every call null.
template <class C> g6s2_pool_ops table(C& c, RunPool& pool) {
    g6s2_pool_ops o{}; o.ctx = static_cast<ModuleCtx*>(&c); o.pool = pool.handle(); return o;
}
template <class C, PoolHit (C::*F)(const std::vector<int>&, uint64_t, uint64_t, int, const g6rt::CacheNamespace&, uint32_t)>
int32_t match(void* x, const int32_t* p, size_t n, uint64_t rows, uint64_t worst, int32_t limit, const uint8_t* ns, uint32_t r,
              g6s2_pool_hit* out) noexcept {
    return mod::guard<C>(x, [&](C& c) { *as<PoolHit>(out) = (c.*F)(ints(p, n), rows, worst, limit, mod::ns(ns), r); });
}
template <class C, Run (C::*F)(const PoolHit&, uint64_t, uint64_t, uint32_t)>
int32_t take(void* x, const g6s2_pool_hit* h, uint64_t rows, uint64_t worst, uint32_t r, g6sch_run* out) noexcept {
    return mod::guard<C>(x, [&](C& c) { *as<Run>(out) = (c.*F)(*as<const PoolHit>(h), rows, worst, r); });
}
template <class C, uint64_t (C::*F)()> int32_t digest(void* x, uint64_t* out) noexcept {
    return mod::guard<C>(x, [&](C& c) { *out = (c.*F)(); });
}
template <class C, std::vector<g6kv_tier_totals> (C::*F)()> int32_t tiers(void* x, g6s2_tiers* out) noexcept {
    return mod::guard<C>(x, [&](C& c) { *as<std::vector<g6kv_tier_totals>>(out) = (c.*F)(); });
}
template <class C, std::string (C::*F)(uint64_t, uint32_t)> int32_t refusal(void* x, uint64_t worst, uint32_t r, g6s2_str* out) noexcept {
    return mod::guard<C>(x, [&](C& c) { *as<std::string>(out) = (c.*F)(worst, r); });
}
template <class C, uint32_t (C::*F)(const Request&, uint64_t, uint64_t, const std::vector<int>*)>
int32_t r_rows(void* x, const g6sch_request* q, uint64_t rows, uint64_t worst, const int32_t* p, size_t n, bool has_prompt,
               uint32_t* out) noexcept {
    return mod::guard<C>(x, [&](C& c) {
        const std::optional<std::vector<int>> prompt = has_prompt ? std::optional<std::vector<int>>(ints(p, n)) : std::nullopt;
        *out = (c.*F)(*as<const Request>(q), rows, worst, prompt ? &*prompt : nullptr);
    });
}
template <class C, std::optional<Shrink> (C::*F)(const std::array<Slot, MAX_SESSIONS>&, const Request&, uint64_t, uint64_t)>
int32_t make_room(void* x, const g6sch_slot* s, const g6sch_request* q, uint64_t rows, uint64_t worst, bool* made,
                  g6sch_shrink* out) noexcept {
    return mod::guard<C>(x, [&](C& c) {
        const std::optional<Shrink> k = (c.*F)(*as<const std::array<Slot, MAX_SESSIONS>>(s), *as<const Request>(q), rows, worst);
        *made = k.has_value();
        if (k) *as<Shrink>(out) = *k;
    });
}
} // namespace pol

/// The caller's side of a pool table: each call through its pointer, and what the module threw rethrown.
class PoolPolicy {
    template <class Fn, class... A> void call(Fn f, A... a) const { mod::call(o.ctx, f, a...); }
public:
    g6s2_pool_ops o{};
    PoolPolicy() = default;
    explicit PoolPolicy(const g6s2_pool_ops& t) : o(t) {}
    PoolHit match(const std::vector<int>& prompt, uint64_t rows, uint64_t worst, int limit, const g6rt::CacheNamespace& ns,
                  uint32_t r) const {
        PoolHit h;
        call(o.match, prompt.data(), prompt.size(), rows, worst, int32_t(limit), ns.data(), r, mod::as<g6s2_pool_hit>(&h));
        return h;
    }
    Run take(const PoolHit& h, uint64_t rows, uint64_t worst, uint32_t r) const {
        Run run;
        call(o.take, mod::as<const g6s2_pool_hit>(&h), rows, worst, r, mod::as<g6sch_run>(&run));
        return run;
    }
    uint64_t digest() const { uint64_t d = 0; call(o.digest, &d); return d; }
    std::vector<g6kv_tier_totals> tiers() const {
        std::vector<g6kv_tier_totals> v;
        call(o.tiers, mod::as<g6s2_tiers>(&v));
        return v;
    }
    std::string refusal(uint64_t worst, uint32_t r) const { std::string s; call(o.refusal, worst, r, mod::as<g6s2_str>(&s)); return s; }
    uint32_t r_rows(const Request& q, uint64_t rows, uint64_t worst, const std::vector<int>* prompt) const {
        uint32_t out = 0;
        call(o.r_rows, mod::as<const g6sch_request>(&q), rows, worst, prompt ? prompt->data() : nullptr, prompt ? prompt->size() : size_t(0),
             prompt != nullptr, &out);
        return out;
    }
    std::optional<Shrink> make_room(const std::array<Slot, MAX_SESSIONS>& slots, const Request& q, uint64_t rows, uint64_t worst) const {
        bool made = false; Shrink k{};
        call(o.make_room, mod::as<const g6sch_slot>(slots.data()), mod::as<const g6sch_request>(&q), rows, worst, &made,
             mod::as<g6sch_shrink>(&k));
        return made ? std::optional<Shrink>(k) : std::nullopt;
    }
};

/// A table over the service's std::function pool hooks (H is PoolHooks): each hook that is set makes its call non-null,
/// and the data is copied. h outlives the FnPool. The table points at the FnPool, so it stays where it is.
template <class H> class FnPool : public ModuleCtx {
    const H& h_;
    PoolHit on_match(const std::vector<int>& p, uint64_t rows, uint64_t worst, int limit, const g6rt::CacheNamespace& ns, uint32_t r) {
        return h_.match(p, rows, worst, limit, ns, r);
    }
    Run on_take(const PoolHit& x, uint64_t rows, uint64_t worst, uint32_t r) { return h_.take(x, rows, worst, r); }
    uint64_t on_digest() { return h_.digest(); }
    std::vector<g6kv_tier_totals> on_tiers() { return h_.tiers(); }
    std::string on_refusal(uint64_t worst, uint32_t r) { return h_.refusal(worst, r); }
    uint32_t on_r_rows(const Request& q, uint64_t rows, uint64_t worst, const std::vector<int>* p) { return h_.r_rows(q, rows, worst, p); }
    std::optional<Shrink> on_make_room(const std::array<Slot, MAX_SESSIONS>& s, const Request& q, uint64_t rows, uint64_t worst) {
        return h_.make_room(s, q, rows, worst);
    }
public:
    explicit FnPool(const H& h) : h_(h) {}
    FnPool(const FnPool&) = delete;
    FnPool& operator=(const FnPool&) = delete;
    g6s2_pool_ops ops() {
        using F = FnPool;
        g6s2_pool_ops o{};
        o.ctx = static_cast<ModuleCtx*>(this);
        o.pool = h_.pool ? h_.pool->handle() : nullptr;
        if (h_.match) o.match = pol::match<F, &F::on_match>;
        if (h_.take) o.take = pol::take<F, &F::on_take>;
        if (h_.digest) o.digest = pol::digest<F, &F::on_digest>;
        if (h_.tiers) o.tiers = pol::tiers<F, &F::on_tiers>;
        if (h_.refusal) o.refusal = pol::refusal<F, &F::on_refusal>;
        if (h_.r_rows) o.r_rows = pol::r_rows<F, &F::on_r_rows>;
        if (h_.make_room) o.make_room = pol::make_room<F, &F::on_make_room>;
        o.max_run_pages = h_.max_run_pages; o.grow = h_.grow; o.count = h_.count; o.status = h_.status;
        return o;
    }
};
} // namespace g6s2
