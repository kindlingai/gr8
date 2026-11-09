// The S2 head's admission decisions as functions of their inputs. The rules are engine/scheduler-rs
// src/service/admit.rs, which the control thread (control.rs) calls; these are facades, so that
// tests/admit_vectors_test.cpp checks the Rust against transcripts recorded from the C++ they replaced.
#pragma once
#include "scheduler.h"
#include <optional>

namespace g6s2::admit {
// The boundary a notice names. A held boundary names itself: every rank is stopped there with no work in flight.
// Otherwise the horizon (the first boundary whose work after the one in flight covers NOTICE_ROWS rows), clamped to
// boundary + NOTICE_MIN .. boundary + notice_max, so a torn read of the pair can only widen the target.
inline uint64_t notice_target(uint64_t boundary, uint64_t horizon, std::optional<uint64_t> held, uint32_t notice_max) {
    return g6svc_notice_target(boundary, horizon, held.has_value(), held.value_or(0), notice_max);
}
// A stalled head waits for a new revision rather than being checked again on every 1 ms pass: AdmissionQueue::check
// refuses a repeat at one revision.
inline bool retry(uint64_t head, uint64_t attempted_head, uint64_t revision, uint64_t attempted_revision) {
    return g6svc_admit_retry(head, attempted_head, revision, attempted_revision);
}
// The revision an attempt expects next: the commit that carries it moves the scheduler one revision on.
inline uint64_t attempted_revision(uint64_t revision) { return g6svc_attempted_revision(revision); }
inline bool shrink(bool fits, bool can_make_room, unsigned live, unsigned limit) {
    return g6svc_admit_shrink(fits, can_make_room, live, limit);
}
// short_of: the chunks a recall of lent memory must bring back first (borrow-idle)
inline bool grant(Stall why, uint32_t short_of) { return g6svc_admit_grant(uint32_t(why), short_of); }
// GR8_S2_ADMIT_HOLD after a grant that leaves a slot: queue the ADMITs already written, then hold the boundary when a
// request is queued behind the grant.
inline bool drain(bool admit_hold, unsigned live, unsigned limit) { return g6svc_admit_drain(admit_hold, live, limit); }
inline bool hold(bool admit_hold, bool queued, unsigned live, unsigned limit) {
    return g6svc_admit_hold(admit_hold, queued, live, limit);
}
// The once-per-request lines: `last` is the request the line last named.
inline bool wait_line(Stall why, uint64_t last, uint64_t head) { return g6svc_wait_line(uint32_t(why), last, head); }
inline bool refusal_line(Stall why, uint64_t last, uint64_t head) { return g6svc_refusal_line(uint32_t(why), last, head); }
// Borrow-idle: Begin when the head starts waiting for a recall, Done when the head that waited is granted.
enum class Recall { None, Begin, Done };
inline Recall recall_note(Stall why, uint32_t short_of, uint64_t waiting, uint64_t head) {
    return Recall(g6svc_recall_note(uint32_t(why), short_of, waiting, head));
}
// The idle tests of the control loop's resizes. The KV share rebalances once nothing is resident or queued, and
// borrow-idle lends chunks only when no session is retiring either.
inline bool share_idle(unsigned resident, size_t queued) { return g6svc_share_idle(resident, queued); }
inline bool lend_idle(unsigned resident, size_t queued, unsigned retiring) { return g6svc_lend_idle(resident, queued, retiring); }
} // namespace g6s2::admit
