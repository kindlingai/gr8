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
    if ((x.has_export = bool(e.export_op))) {
        x.export_op = {{e.export_op->key.epoch, e.export_op->key.slot, e.export_op->key.generation},
                       e.export_op->export_id, e.export_op->first_row, e.export_op->rows};
    }
    x.has_import = e.import_session;
    if ((x.has_switch = bool(e.switch_op))) {
        x.switch_op = {e.switch_op->group_id, e.switch_op->target_nr, e.switch_op->flags};
    }
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
    if (x.has_export) {
        c.export_op = Export{Key{x.export_op.key.epoch, x.export_op.key.slot, x.export_op.key.generation},
                             x.export_op.export_id, x.export_op.first_row, x.export_op.rows};
    }
    c.import_session = x.has_import;
    if (x.has_switch) {
        c.switch_op = SwitchOp{x.switch_op.group_id, x.switch_op.target_nr, x.switch_op.flags};
    }
    return c;
}
} // namespace g6s2::ctl
