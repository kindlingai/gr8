// The S2 head's admission decisions as functions of their inputs. service.h's control loop and its event commit call
// them. When those loops move to Rust (HOST-CORE.md R5b2), tests/admit_vectors_test.cpp checks the Rust against
// transcripts recorded from these.
#pragma once
#include "scheduler.h"
#include <algorithm>
#include <optional>

namespace g6s2::admit {
// The boundary a notice names. A held boundary names itself: every rank is stopped there with no work in flight.
// Otherwise the horizon (the first boundary whose work after the one in flight covers NOTICE_ROWS rows), clamped to
// boundary + NOTICE_MIN .. boundary + notice_max, so a torn read of the pair can only widen the target.
inline uint64_t notice_target(uint64_t boundary, uint64_t horizon, std::optional<uint64_t> held, uint32_t notice_max) {
    return held ? *held : std::clamp(horizon, boundary + NOTICE_MIN, boundary + notice_max);
}
// A stalled head waits for a new revision rather than being checked again on every 1 ms pass: AdmissionQueue::check
// refuses a repeat at one revision.
inline bool retry(uint64_t head, uint64_t attempted_head, uint64_t revision, uint64_t attempted_revision) {
    return head != attempted_head || revision != attempted_revision;
}
// The revision an attempt expects next: the commit that carries it moves the scheduler one revision on.
inline uint64_t attempted_revision(uint64_t revision) { return revision + 1; }
inline bool shrink(bool fits, bool can_make_room, unsigned live, unsigned limit) {
    return !fits && can_make_room && live < limit;
}
// short_of: the chunks a recall of lent memory must bring back first (borrow-idle)
inline bool grant(Stall why, uint32_t short_of) { return why == Stall::None && !short_of; }
// GR8_S2_ADMIT_HOLD after a grant that leaves a slot: queue the ADMITs already written, then hold the boundary when a
// request is queued behind the grant.
inline bool drain(bool admit_hold, unsigned live, unsigned limit) { return admit_hold && live + 1 < limit; }
inline bool hold(bool admit_hold, bool queued, unsigned live, unsigned limit) {
    return drain(admit_hold, live, limit) && queued;
}
// The once-per-request lines: `last` is the request the line last named.
inline bool wait_line(Stall why, uint64_t last, uint64_t head) { return why != Stall::None && last != head; }
inline bool refusal_line(Stall why, uint64_t last, uint64_t head) { return why == Stall::Capacity && last != head; }
// Borrow-idle: Begin when the head starts waiting for a recall, Done when the head that waited is granted.
enum class Recall { None, Begin, Done };
inline Recall recall_note(Stall why, uint32_t short_of, uint64_t waiting, uint64_t head) {
    if (short_of) return waiting != head ? Recall::Begin : Recall::None;
    return why == Stall::None && waiting == head ? Recall::Done : Recall::None;
}
// The idle tests of the control loop's resizes. The KV share rebalances once nothing is resident or queued, and
// borrow-idle lends chunks only when no session is retiring either.
inline bool share_idle(unsigned resident, size_t queued) { return resident == 0 && queued == 0; }
inline bool lend_idle(unsigned resident, size_t queued, unsigned retiring) {
    return share_idle(resident, queued) && retiring == 0;
}
} // namespace g6s2::admit
