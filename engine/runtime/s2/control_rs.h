// The C++ side of the control links in Rust (engine/scheduler-rs src/control, g6ctl_* in g6sched.h): a call's result,
// the buffers the Rust appends to, and the commit as the control link carries it.
#pragma once
#include "scheduler.h"
#include <exception>

namespace g6s2::ctl {
// 1: a callback threw, rethrown here from *held; 2: refused, with this thread's message
inline void check(int32_t rc, std::exception_ptr* held = nullptr) {
    if (rc == 1) {
        if (held && *held) { std::exception_ptr e = *held; *held = nullptr; std::rethrow_exception(e); }
        throw std::runtime_error("s2: a control callback failed without an exception");
    }
    if (rc) throw std::runtime_error(g6ctl_error());
}
inline uint8_t* grow_vector(void* ctx, size_t n) {
    auto& v = *static_cast<std::vector<uint8_t>*>(ctx);
    v.resize(v.size() + n);
    return v.data() + v.size() - n;
}
inline uint8_t* grow_string(void* ctx, size_t n) {
    auto& s = *static_cast<std::string*>(ctx);
    s.resize(s.size() + n);
    return reinterpret_cast<uint8_t*>(&s[0]) + s.size() - n;
}
inline const uint8_t* bytes(const std::string& s) { return reinterpret_cast<const uint8_t*>(s.data()); }
inline g6ctl_commit to_ctl(const Commit& e) {
    g6ctl_commit x{};
    x.c = sch::to_rs(e);
    x.hold = e.hold;
    if ((x.has_resize = bool(e.resize))) x.resize = {e.resize->tier, e.resize->segs, e.resize->slots, e.resize->txn};
    return x;
}
inline Commit from_ctl(const g6ctl_commit& x) {
    Commit c;
    c.epoch = x.c.epoch; c.event = x.c.event; c.boundary = x.c.boundary; c.before_hash = x.c.before_hash;
    if (x.c.has_cancel) c.cancel = Key{x.c.cancel.epoch, x.c.cancel.slot, x.c.cancel.generation};
    if (x.c.has_admit) { Grant g; std::memcpy(static_cast<void*>(&g), &x.c.admit, sizeof g); c.admit = g; }
    c.hold = x.hold;
    if (x.has_resize) c.resize = Resize{x.resize.tier, x.resize.segs, x.resize.slots, x.resize.txn};
    if (x.c.has_shrink) { Shrink k; std::memcpy(static_cast<void*>(&k), &x.c.shrink, sizeof k); c.shrink = k; }
    return c;
}
} // namespace g6s2::ctl
