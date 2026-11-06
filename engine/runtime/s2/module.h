// ModuleOps (g6sched.h g6s2_module_ops), the C++ side: the trampolines a module fills its table with, Module (the
// caller's side, which rethrows what the module threw) and FnModule (a table over std::function hooks).
// An adapter's context derives from ModuleCtx. Its hooks are member functions, and each table member is the trampoline
// over one, as mod::bind<C, &C::on_bind>. A trampoline catches what its hook throws and returns 1. Module rethrows it.
#pragma once
#include "sessions.h"
#include "../prefix_reuse.h"
#include <exception>
#include <functional>
#include <type_traits>

namespace g6s2 {
struct Launch;
struct DispatchHooks;
static_assert(sizeof(Sampling) == sizeof(g6s2_sampling) && offsetof(Sampling, seed) == offsetof(g6s2_sampling, seed) &&
              offsetof(Sampling, plp) == offsetof(g6s2_sampling, plp), "Sampling layout");
static_assert(sizeof(g6rt::LpRow) == sizeof(g6s2_lp_row) && offsetof(g6rt::LpRow, id) == offsetof(g6s2_lp_row, id) &&
              offsetof(g6rt::LpRow, tlp) == offsetof(g6s2_lp_row, tlp), "LpRow layout");
static_assert(sizeof(Resize) == sizeof(g6sch_resize) && offsetof(Resize, segs) == offsetof(g6sch_resize, segs) &&
              offsetof(Resize, txn) == offsetof(g6sch_resize, txn), "Resize layout");
static_assert(sizeof(s2::DflReq) == sizeof(g6s2_dfl_req) && offsetof(s2::DflReq, phase) == offsetof(g6s2_dfl_req, phase) &&
              sizeof(s2::DflDrafts) == sizeof(g6s2_dfl_drafts), "DflReq layout");
static_assert(sizeof(g6rt::CacheNamespace) == G6KV_NAMESPACE_BYTES, "cache namespace size");

/// The base of every module context: ModuleOps::ctx points at it, and it holds the exception a call threw. Module calls
/// never overlap: the head's control thread calls the module only inside the event-boundary pause. A call moved off the
/// pause needs its own error slot.
struct ModuleCtx { std::exception_ptr err; };

namespace mod {
// the C types of the table's arguments, and back
template <class C, class T> C* as(T* p) { return reinterpret_cast<C*>(p); }
inline g6rt::CacheNamespace ns(const uint8_t* p) { g6rt::CacheNamespace n; std::memcpy(n.data(), p, n.size()); return n; }
/// A table for context c, every call null.
template <class C> g6s2_module_ops table(C& c) { g6s2_module_ops o{}; o.ctx = static_cast<ModuleCtx*>(&c); return o; }
template <class C, class F> int32_t guard(void* x, F&& f) noexcept {
    ModuleCtx* m = static_cast<ModuleCtx*>(x);
    try { f(*static_cast<C*>(m)); return 0; } catch (...) { m->err = std::current_exception(); return 1; }
}
// one trampoline per call, over context C's member function F
template <class C, void (C::*F)(const Grant&)> int32_t bind(void* x, const g6sch_grant* g) noexcept {
    return guard<C>(x, [&](C& c) { (c.*F)(*as<const Grant>(g)); });
}
template <class C, void (C::*F)(unsigned, const g6rt::ServeReq&)> int32_t bind_request(void* x, uint32_t slot, const g6s2_serve_req* r) noexcept {
    return guard<C>(x, [&](C& c) { (c.*F)(slot, *as<const g6rt::ServeReq>(r)); });
}
template <class C, void (C::*F)(unsigned, const Run&)> int32_t grown(void* x, uint32_t slot, const g6sch_run* r) noexcept {
    return guard<C>(x, [&](C& c) { (c.*F)(slot, *as<const Run>(r)); });
}
template <class C, void (C::*F)(const Shrink&, const Run&, uint64_t)>
int32_t shrink(void* x, const g6sch_shrink* k, const g6sch_run* was, uint64_t length) noexcept {
    return guard<C>(x, [&](C& c) { (c.*F)(*as<const Shrink>(k), *as<const Run>(was), length); });
}
template <class C, void (C::*F)(const Slot&)> int32_t retire(void* x, const g6sch_slot* s) noexcept {
    return guard<C>(x, [&](C& c) { (c.*F)(*as<const Slot>(s)); });
}
template <class C, void (C::*F)(const Run&, const g6rt::PrefixReuse&, const g6rt::CacheNamespace&)>
int32_t retain(void* x, const g6sch_run* r, const g6s2_reuse* st, const uint8_t* n) noexcept {
    return guard<C>(x, [&](C& c) { (c.*F)(*as<const Run>(r), *as<const g6rt::PrefixReuse>(st), ns(n)); });
}
template <class C, void (C::*F)(unsigned, const Run&, const g6rt::PrefixReuse&, const g6rt::CacheNamespace&)>
int32_t park(void* x, uint32_t slot, const g6sch_run* r, const g6s2_reuse* st, const uint8_t* n) noexcept {
    return guard<C>(x, [&](C& c) { (c.*F)(slot, *as<const Run>(r), *as<const g6rt::PrefixReuse>(st), ns(n)); });
}
template <class C, void (C::*F)()> int32_t call(void* x) noexcept { return guard<C>(x, [&](C& c) { (c.*F)(); }); }
template <class C, bool (C::*F)()> int32_t drained(void* x, bool* out) noexcept {
    return guard<C>(x, [&](C& c) { *out = (c.*F)(); });
}
template <class C, void (C::*F)(const Resize&)> int32_t resize(void* x, const g6sch_resize* r) noexcept {
    return guard<C>(x, [&](C& c) { (c.*F)(*as<const Resize>(r)); });
}
template <class C, uint32_t (C::*F)()> int32_t flash_room(void* x, uint32_t* out) noexcept {
    return guard<C>(x, [&](C& c) { *out = (c.*F)(); });
}
template <class C, int (C::*F)(bool, int)> int32_t bucket(void* x, bool draft, int32_t context, int32_t* out) noexcept {
    return guard<C>(x, [&](C& c) { *out = (c.*F)(draft, context); });
}
template <class C, void (C::*F)(int, int)> int32_t mode(void* x, int32_t old_solo, int32_t new_solo) noexcept {
    return guard<C>(x, [&](C& c) { (c.*F)(old_solo, new_solo); });
}
template <class C, std::vector<int> (C::*F)(const Launch&)> int32_t launch(void* x, const g6s2_launch* l, g6s2_ids* out) noexcept {
    return guard<C>(x, [&](C& c) { *as<std::vector<int>>(out) = (c.*F)(*as<const Launch>(l)); });
}
template <class C, void (C::*F)(const Slot&, const Payload&, uint32_t, uint32_t)>
int32_t prefill(void* x, const g6sch_slot* s, const g6s2_payload* p, uint32_t first, uint32_t rows) noexcept {
    return guard<C>(x, [&](C& c) { (c.*F)(*as<const Slot>(s), *as<const Payload>(p), first, rows); });
}
template <class C, void (C::*F)(const Slot&, const Payload&, uint32_t, uint32_t, uint64_t)>
int32_t prefill_seq(void* x, const g6sch_slot* s, const g6s2_payload* p, uint32_t first, uint32_t rows, uint64_t seq) noexcept {
    return guard<C>(x, [&](C& c) { (c.*F)(*as<const Slot>(s), *as<const Payload>(p), first, rows, seq); });
}
template <class C, void (C::*F)(const std::vector<PrefillPart>&, uint64_t)>
int32_t prefill_pack(void* x, const g6s2_parts* parts, uint64_t seq) noexcept {
    return guard<C>(x, [&](C& c) { (c.*F)(*as<const std::vector<PrefillPart>>(parts), seq); });
}
template <class C, g6rt::LpRow (C::*F)(int, bool, int)>
int32_t logprob(void* x, int32_t row, bool target, int32_t k, g6s2_lp_row* out) noexcept {
    return guard<C>(x, [&](C& c) { *as<g6rt::LpRow>(out) = (c.*F)(row, target, k); });
}
template <class C, float (C::*F)(int)> int32_t draft_lp(void* x, int32_t row, float* out) noexcept {
    return guard<C>(x, [&](C& c) { *out = (c.*F)(row); });
}
template <class C, bool (C::*F)(int)> int32_t dfl_ok(void* x, int32_t slot, bool* out) noexcept {
    return guard<C>(x, [&](C& c) { *out = (c.*F)(slot); });
}
template <class C, int (C::*F)(int)> int32_t dfl_start(void* x, int32_t slot, int32_t* out) noexcept {
    return guard<C>(x, [&](C& c) { *out = (c.*F)(slot); });
}
template <class C, s2::DflDrafts (C::*F)(const s2::DflReq&, const Sampling&)>
int32_t dfl(void* x, const g6s2_dfl_req* r, const g6s2_sampling* s, g6s2_dfl_drafts* out) noexcept {
    return guard<C>(x, [&](C& c) { *as<s2::DflDrafts>(out) = (c.*F)(*as<const s2::DflReq>(r), *as<const Sampling>(s)); });
}
} // namespace mod
// An adapter's context holds a reference to each local its hooks used, under the local's own name, so the hook bodies
// read as they did. CAPTURES(X) lists the names: CAPTURE_TYPE before the struct (a member's type may not name the member
// it declares), CAPTURE_REF in it, CAPTURE_ARG in its initializer after the ModuleCtx.
#define G6S2_CAPTURE_TYPE(x) using x##_capture_t = decltype(x);
#define G6S2_CAPTURE_REF(x) x##_capture_t& x;
#define G6S2_CAPTURE_ARG(x) x,

/// The caller's side of a table: each call through its pointer, and what the module threw rethrown.
class Module {
    // a missing call throws what calling an empty std::function hook did
    template <class Fn, class... A> void call(Fn f, A... a) const {
        if (!f) throw std::bad_function_call();
        if (!f(o.ctx, a...)) return;
        std::exception_ptr& err = static_cast<ModuleCtx*>(o.ctx)->err;
        if (!err) throw std::runtime_error("s2: a module call failed without an exception");
        std::exception_ptr e = err; err = nullptr; std::rethrow_exception(e);
    }
public:
    g6s2_module_ops o{};
    Module() = default;
    explicit Module(const g6s2_module_ops& t) : o(t) {}
    void bind(const Grant& g) const { call(o.bind, mod::as<const g6sch_grant>(&g)); }
    void bind_request(unsigned slot, const g6rt::ServeReq& r) const { call(o.bind_request, slot, mod::as<const g6s2_serve_req>(&r)); }
    void grown(unsigned slot, const Run& r) const { call(o.grown, slot, mod::as<const g6sch_run>(&r)); }
    void shrink(const Shrink& k, const Run& was, uint64_t length) const {
        call(o.shrink, mod::as<const g6sch_shrink>(&k), mod::as<const g6sch_run>(&was), length);
    }
    void retire(const Slot& s) const { call(o.retire, mod::as<const g6sch_slot>(&s)); }
    void retain(const Run& r, const g6rt::PrefixReuse& st, const g6rt::CacheNamespace& ns) const {
        call(o.retain, mod::as<const g6sch_run>(&r), mod::as<const g6s2_reuse>(&st), ns.data());
    }
    void park(unsigned slot, const Run& r, const g6rt::PrefixReuse& st, const g6rt::CacheNamespace& ns) const {
        call(o.park, slot, mod::as<const g6sch_run>(&r), mod::as<const g6s2_reuse>(&st), ns.data());
    }
    void fence() const { call(o.fence); }
    void before_work() const { call(o.before_work); }
    bool drained() const { bool d = false; call(o.drained, &d); return d; }
    void resize(const Resize& r) const { call(o.resize, mod::as<const g6sch_resize>(&r)); }
    uint32_t flash_room() const { uint32_t n = 0; call(o.flash_room, &n); return n; }
    int bucket(bool draft, int context) const { int32_t b = 0; call(o.bucket, draft, context, &b); return b; }
    void mode(int old_solo, int new_solo) const { call(o.mode, old_solo, new_solo); }
    std::vector<int> launch(const Launch& l) const {
        std::vector<int> ids;
        call(o.launch, mod::as<const g6s2_launch>(&l), mod::as<g6s2_ids>(&ids));
        return ids;
    }
    void prefill(const Slot& s, const Payload& p, uint32_t first, uint32_t rows) const {
        call(o.prefill, mod::as<const g6sch_slot>(&s), mod::as<const g6s2_payload>(&p), first, rows);
    }
    void prefill_seq(const Slot& s, const Payload& p, uint32_t first, uint32_t rows, uint64_t seq) const {
        call(o.prefill_seq, mod::as<const g6sch_slot>(&s), mod::as<const g6s2_payload>(&p), first, rows, seq);
    }
    void prefill_pack(const std::vector<PrefillPart>& parts, uint64_t seq) const { call(o.prefill_pack, mod::as<const g6s2_parts>(&parts), seq); }
    g6rt::LpRow logprob(int row, bool target, int k) const { g6rt::LpRow r; call(o.logprob, row, target, k, mod::as<g6s2_lp_row>(&r)); return r; }
    float draft_lp(int row) const { float v = 0; call(o.draft_lp, row, &v); return v; }
    bool dfl_ok(int slot) const { bool ok = false; call(o.dfl_ok, slot, &ok); return ok; }
    int dfl_start(int slot) const { int32_t lo = 0; call(o.dfl_start, slot, &lo); return lo; }
    s2::DflDrafts dfl(const s2::DflReq& r, const Sampling& s) const {
        s2::DflDrafts d;
        call(o.dfl, mod::as<const g6s2_dfl_req>(&r), mod::as<const g6s2_sampling>(&s), mod::as<g6s2_dfl_drafts>(&d));
        return d;
    }
};

/// A table over std::function hooks: H is DispatchHooks (the step calls only) or the service's NativeHooks (also its
/// module members and pool.retain and pool.park). Each hook that is set makes its call non-null. h outlives the
/// FnModule. The table points at the FnModule, so it stays where it is.
template <class H> class FnModule : public ModuleCtx {
    const H& h_;
    const auto& d() const {   // deduced: the compiler looks up H's members once H is complete
        if constexpr (std::is_same_v<H, DispatchHooks>) return h_; else return h_.dispatch;
    }
    void on_bind(const Grant& g) { h_.bind(g); }
    void on_bind_request(unsigned slot, const g6rt::ServeReq& r) { h_.bind_request(slot, r); }
    void on_grown(unsigned slot, const Run& r) { h_.grown(slot, r); }
    void on_shrink(const Shrink& k, const Run& was, uint64_t length) { h_.shrink(k, was, length); }
    void on_retire(const Slot& s) { h_.retire(s); }
    void on_retain(const Run& r, const g6rt::PrefixReuse& st, const g6rt::CacheNamespace& ns) { h_.pool.retain(r, st, ns); }
    void on_park(unsigned slot, const Run& r, const g6rt::PrefixReuse& st, const g6rt::CacheNamespace& ns) { h_.pool.park(slot, r, st, ns); }
    void on_fence() { h_.fence(); }
    void on_before_work() { h_.before_work(); }
    bool on_drained() { return h_.drained(); }
    void on_resize(const Resize& r) { h_.resize(r); }
    uint32_t on_flash_room() { return h_.flash_room(); }
    int on_bucket(bool draft, int context) { return d().bucket(draft, context); }
    void on_mode(int old_solo, int new_solo) { d().mode(old_solo, new_solo); }
    std::vector<int> on_launch(const Launch& l) { return d().launch(l); }
    void on_prefill(const Slot& s, const Payload& p, uint32_t first, uint32_t rows) { d().prefill(s, p, first, rows); }
    void on_prefill_seq(const Slot& s, const Payload& p, uint32_t first, uint32_t rows, uint64_t seq) { d().prefill_seq(s, p, first, rows, seq); }
    void on_prefill_pack(const std::vector<PrefillPart>& parts, uint64_t seq) { d().prefill_pack_seq(parts, seq); }
    g6rt::LpRow on_logprob(int row, bool target, int k) { return d().logprob(row, target, k); }
    float on_draft_lp(int row) { return d().draft_lp(row); }
    bool on_dfl_ok(int slot) { return d().dfl_ok(slot); }
    int on_dfl_start(int slot) { return d().dfl_start(slot); }
    s2::DflDrafts on_dfl(const s2::DflReq& r, const Sampling& s) { return d().dfl(r, s); }
public:
    explicit FnModule(const H& h) : h_(h) {}
    FnModule(const FnModule&) = delete;
    FnModule& operator=(const FnModule&) = delete;
    g6s2_module_ops ops() {
        using F = FnModule;
        g6s2_module_ops o = mod::table(*this);
        if constexpr (!std::is_same_v<H, DispatchHooks>) {
            if (h_.bind) o.bind = mod::bind<F, &F::on_bind>;
            if (h_.bind_request) o.bind_request = mod::bind_request<F, &F::on_bind_request>;
            if (h_.grown) o.grown = mod::grown<F, &F::on_grown>;
            if (h_.shrink) o.shrink = mod::shrink<F, &F::on_shrink>;
            if (h_.retire) o.retire = mod::retire<F, &F::on_retire>;
            if (h_.pool.retain) o.retain = mod::retain<F, &F::on_retain>;
            if (h_.pool.park) o.park = mod::park<F, &F::on_park>;
            if (h_.fence) o.fence = mod::call<F, &F::on_fence>;
            if (h_.before_work) o.before_work = mod::call<F, &F::on_before_work>;
            if (h_.drained) o.drained = mod::drained<F, &F::on_drained>;
            if (h_.resize) o.resize = mod::resize<F, &F::on_resize>;
            if (h_.flash_room) o.flash_room = mod::flash_room<F, &F::on_flash_room>;
        }
        const auto& x = d();
        if (x.bucket) o.bucket = mod::bucket<F, &F::on_bucket>;
        if (x.mode) o.mode = mod::mode<F, &F::on_mode>;
        if (x.launch) o.launch = mod::launch<F, &F::on_launch>;
        if (x.prefill) o.prefill = mod::prefill<F, &F::on_prefill>;
        if (x.prefill_seq) o.prefill_seq = mod::prefill_seq<F, &F::on_prefill_seq>;
        if (x.prefill_pack_seq) o.prefill_pack = mod::prefill_pack<F, &F::on_prefill_pack>;
        if (x.logprob) o.logprob = mod::logprob<F, &F::on_logprob>;
        if (x.draft_lp) o.draft_lp = mod::draft_lp<F, &F::on_draft_lp>;
        if (x.dfl_ok) o.dfl_ok = mod::dfl_ok<F, &F::on_dfl_ok>;
        if (x.dfl_start) o.dfl_start = mod::dfl_start<F, &F::on_dfl_start>;
        if (x.dfl) o.dfl = mod::dfl<F, &F::on_dfl>;
        o.gm_on = x.gm_on; o.gm_bits = x.gm_bits; o.gm_words = x.gm_words;
        o.pen_rows = x.pen_rows; o.pen_sess = x.pen_sess;
        return o;
    }
};
} // namespace g6s2
