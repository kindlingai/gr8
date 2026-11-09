// The S2 head's admission decisions (s2/admit.h) against transcripts recorded from the C++ before R5b2 moves the
// service's loops to Rust. "notice" and "gates" run each decision over seeded inputs. "sequence" runs the control loop's
// admission path as service.h runs it: the command stream, the FIFO (the Rust AdmissionQueue) and its check, the retry
// gate, the shrink, the recall, the grant and the ADMIT_HOLD drain and chain, over seeded pool answers. Each section's
// transcript hash must equal the recorded one. Rerecord only for a change meant to alter a decision, and say so in the
// commit.
//   --dump SECTION   print that section's transcript (to diff two builds line by line)
// The generator is mt19937_64 with modulo ranges. No call or + takes two draws as operands: their order is unspecified,
// and GCC on x86_64 takes them right to left. Host only.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <optional>
#include <random>
#include <string>
#include <vector>
#include "../admit.h"

using namespace g6s2;
namespace {
struct Rng {
    std::mt19937_64 g;
    explicit Rng(uint64_t seed) : g(seed) {}
    uint64_t U(uint64_t lo, uint64_t hi) { return hi - lo == UINT64_MAX ? g() : lo + g() % (hi - lo + 1); }
    bool coin(unsigned pct) { return U(0, 99) < pct; }
};
struct Tx {
    uint64_t h = 14695981039346656037ull;
    std::vector<std::string>* lines = nullptr;
    void operator()(const std::string& s) {
        for (unsigned char c : s) h = (h ^ c) * 1099511628211ull;
        h = (h ^ '\n') * 1099511628211ull;
        if (lines) lines->push_back(s);
    }
};
std::string u(uint64_t x) { return std::to_string(x); }
const char* why(Stall w) { return w == Stall::None ? "none" : w == Stall::Slots ? "slots" : "capacity"; }
const char* note(admit::Recall r) { return r == admit::Recall::None ? "none" : r == admit::Recall::Begin ? "begin" : "done"; }
// an id near a marker: equal, one off, or far
uint64_t near(Rng& r, uint64_t x) {
    const uint64_t k = r.U(0, 3);
    return k == 0 ? x : k == 1 ? x + 1 : k == 2 ? x - 1 : r.U(0, UINT64_MAX);
}

void notice(Tx& tx) {
    Rng r(0x6e6f74696365ull);
    for (int i = 0; i < 20000; ++i) {
        const uint64_t edge = r.U(0, 2);
        const uint64_t boundary = edge == 0 ? r.U(0, 64) : edge == 1 ? r.U(0, uint64_t(INT64_MAX) - NOTICE_MAX) : uint64_t(INT64_MAX) - NOTICE_MAX;
        const uint64_t shift = r.U(0, 12);
        const uint64_t horizon = r.coin(10) ? r.U(0, UINT64_MAX) : boundary + shift;
        const uint32_t notice_max = uint32_t(r.U(NOTICE_MIN, NOTICE_MAX));
        std::optional<uint64_t> held;
        if (r.coin(20)) held = boundary;
        tx(u(boundary) + " " + u(horizon) + " " + (held ? u(*held) : "-") + " " + u(notice_max) + " -> " +
           u(admit::notice_target(boundary, horizon, held, notice_max)));
    }
}

void gates(Tx& tx) {
    Rng r(0x6761746573ull);
    for (int i = 0; i < 20000; ++i) {
        const uint64_t head = r.U(0, 6);
        const uint64_t attempted = near(r, head);
        const uint64_t revision = r.U(0, 6);
        const uint64_t expected = near(r, revision);
        const bool fits = r.coin(50), can = r.coin(50), admit_hold = r.coin(50), queued = r.coin(50);
        const unsigned limit = unsigned(r.U(1, 8));
        const unsigned live = unsigned(r.U(0, limit));
        const Stall w = Stall(r.U(0, 2));
        const uint32_t short_of = r.coin(70) ? 0 : uint32_t(r.U(1, 16));
        const uint64_t last = near(r, head);
        const uint64_t waiting = near(r, head);
        const unsigned resident = unsigned(r.U(0, 2)), retiring = unsigned(r.U(0, 2));
        const size_t qn = size_t(r.U(0, 2));
        tx(u(head) + " " + u(attempted) + " " + u(revision) + " " + u(expected) + " " + u(fits) + u(can) + u(admit_hold) +
           u(queued) + " " + u(live) + "/" + u(limit) + " " + why(w) + " " + u(short_of) + " " + u(last) + " " + u(waiting) +
           " " + u(resident) + u(qn) + u(retiring) + " -> retry " + u(admit::retry(head, attempted, revision, expected)) +
           " next " + u(admit::attempted_revision(revision)) + " shrink " + u(admit::shrink(fits, can, live, limit)) +
           " grant " + u(admit::grant(w, short_of)) + " drain " + u(admit::drain(admit_hold, live, limit)) + " hold " +
           u(admit::hold(admit_hold, queued, live, limit)) + " wait " + u(admit::wait_line(w, last, head)) + " refusal " +
           u(admit::refusal_line(w, last, head)) + " recall " + note(admit::recall_note(w, short_of, waiting, head)) + " share " +
           u(admit::share_idle(resident, qn)) + " lend " + u(admit::lend_idle(resident, qn, retiring)));
    }
}

// control.rs's controller and event over a scripted pool: every pool answer is a draw, so the transcript is the
// decision sequence alone
struct Loop {
    Rng& r; Tx& tx;
    Config c;
    AdmissionQueue q;
    bool admit_hold, flash, trace;
    bool resizing = false;   // a resize announced and not yet applied
    struct Cmd { bool admit; uint64_t id; };
    std::deque<Cmd> stream;
    uint64_t next_id = 1, revision = 0, boundary = 0;
    uint64_t attempted_head = UINT64_MAX, attempted_rev = UINT64_MAX;
    uint64_t waited = 0, refused = UINT64_MAX, recall_for = UINT64_MAX;
    uint32_t recall_due = 0;
    std::vector<uint64_t> live_ids;
    Loop(Rng& rr, Tx& t, bool hold, bool fl, bool tr) : r(rr), tx(t), c(), q(c), admit_hold(hold), flash(fl), trace(tr) {}
    unsigned live() const { return unsigned(live_ids.size()); }
    void intake(uint64_t id) {
        Request req;
        req.id = id;
        req.prompt_tokens = uint32_t(r.U(1, 4096));
        req.max_tokens = uint32_t(r.U(1, 2048));
        q.push(req);
        tx("intake " + u(id) + " queued " + u(q.size()));
    }
    // event(): one admission attempt, and its held chain
    void attempt(std::optional<uint64_t> held) {
        const uint64_t horizon_shift = r.U(0, 12);
        const uint64_t target = admit::notice_target(boundary, boundary + horizon_shift, held, c.notice_max);
        const Request* front = q.front();
        if (!front) { tx("attempt with no head"); return; }
        const uint64_t id = front->id;
        const unsigned lv = live();
        bool fits = r.coin(55);
        const bool can = r.coin(60);
        bool shrunk = false;
        if (admit::shrink(fits, can, lv, c.limit())) {
            shrunk = r.coin(50);
            if (shrunk) fits = r.coin(60);
        }
        const uint32_t room = uint32_t(r.U(0, 4096));
        const auto fit = q.check(revision, lv < c.limit(), fits, room);
        std::string line = "attempt " + u(id) + " at " + u(target) + " live " + u(lv) + " fits " + u(fits) + " shrunk " + u(shrunk);
        if (!fit) { fprintf(stderr, "admit_vectors_test: a repeated check at revision %llu\n", (unsigned long long)revision); std::abort(); }
        line += std::string(" why ") + why(fit->why) + " requested " + u(fit->requested) + " room " + u(fit->room);
        if (trace && admit::wait_line(fit->why, waited, id)) { waited = id; line += " wait-line"; }
        if (admit::refusal_line(fit->why, refused, id)) { refused = id; line += " refusal-line"; }
        const uint32_t short_of = fit->why == Stall::None && flash && r.coin(40) ? uint32_t(r.U(1, 8)) : 0;
        const auto n = admit::recall_note(fit->why, short_of, recall_for, id);
        if (n == admit::Recall::Begin) recall_for = id;
        else if (n == admit::Recall::Done) recall_for = UINT64_MAX;
        if (short_of) recall_due = short_of;
        line += std::string(" recall ") + note(n) + " short " + u(short_of);
        bool hold = false;
        if (admit::grant(fit->why, short_of) && r.coin(97)) {
            q.admitted(id);
            live_ids.push_back(id);
            line += " granted";
            // commands that land while the event chain waits for its ACKs
            const uint64_t landed = r.U(0, 2);
            for (uint64_t i = 0; i < landed; ++i) {
                if (r.coin(80)) stream.push_back({true, next_id++});
                else stream.push_back({false, r.U(1, next_id)});
            }
            if (admit::drain(admit_hold, lv, c.limit()))
                while (!stream.empty() && stream.front().admit) { const auto cm = stream.front(); stream.pop_front(); intake(cm.id); }
            hold = admit::hold(admit_hold, q.front() != nullptr, lv, c.limit());
        }
        attempted_head = id; attempted_rev = admit::attempted_revision(revision);
        ++revision;   // the commit
        tx(line + " hold " + u(hold) + " revision " + u(revision));
        if (hold) attempt(target);
    }
    // a cancel that the FIFO did not hold: an event, whose commit ends the session when it is live
    void cancel_event(uint64_t id) {
        ++revision;
        for (size_t k = 0; k < live_ids.size(); ++k)
            if (live_ids[k] == id) {
                live_ids.erase(live_ids.begin() + std::ptrdiff_t(k));
                tx("cancel event " + u(id) + " ended revision " + u(revision));
                return;
            }
        tx("cancel event " + u(id) + " revision " + u(revision));
    }
    void finish_one() {
        if (live_ids.empty()) return;
        const size_t k = size_t(r.U(0, live_ids.size() - 1));
        tx("finished " + u(live_ids[k]));
        live_ids.erase(live_ids.begin() + std::ptrdiff_t(k));
        ++revision;
    }
    void pass() {
        if (resizing && r.coin(50)) { resizing = false; tx("resize applied"); }
        // the KV share and borrow-idle resizes: announced only while no other resize is in flight
        if (!resizing && r.coin(3)) {
            const unsigned resident = live();
            const size_t queued = q.size();
            const bool idle = r.coin(50) ? admit::share_idle(resident, queued) : admit::lend_idle(resident, queued, 0);
            if (idle) { resizing = true; ++revision; tx("resize revision " + u(revision)); }
        }
        const uint64_t arrivals = r.coin(10) ? r.U(1, 3) : 0;
        for (uint64_t i = 0; i < arrivals; ++i) {
            if (r.coin(85)) stream.push_back({true, next_id++});
            else stream.push_back({false, r.U(1, next_id)});
        }
        while (!stream.empty()) {
            const Cmd cm = stream.front(); stream.pop_front();
            if (cm.admit) intake(cm.id);
            else if (q.cancel(cm.id)) tx("abort " + u(cm.id) + " queued " + u(q.size()));
            else cancel_event(cm.id);
        }
        if (const Request* f = q.front())
            if (admit::retry(f->id, attempted_head, revision, attempted_rev)) attempt({});
        if (recall_due && !resizing) { tx("recall " + u(recall_due)); recall_due = 0; resizing = true; ++revision; }
        const uint64_t steps = r.U(0, 3);
        boundary += steps;
        if (r.coin(25)) finish_one();
        const unsigned resident = live();
        const size_t queued = q.size();
        tx("idle share " + u(admit::share_idle(resident, queued)) + " lend " + u(admit::lend_idle(resident, queued, 0)));
    }
};

void sequence(Tx& tx) {
    for (uint64_t run = 0; run < 24; ++run) {
        Rng r(0x73657175656e6365ull + run);
        const bool hold = run % 2 == 1, flash = run % 4 >= 2, trace = run % 3 != 0;
        tx("run " + u(run) + " hold " + u(hold) + " flash " + u(flash) + " trace " + u(trace));
        Loop l(r, tx, hold, flash, trace);
        for (int i = 0; i < 1500; ++i) l.pass();
    }
}

struct Section { const char* name; std::function<void(Tx&)> run; uint64_t want; };
const Section SECTIONS[] = {
    {"notice", notice, 0xa833da1088f68211ull},
    {"gates", gates, 0x9c2616923025ddf0ull},
    {"sequence", sequence, 0x0b85c5b59959d603ull},
};
} // namespace

int main(int argc, char** argv) {
    if (argc == 3 && !std::strcmp(argv[1], "--dump")) {
        for (const auto& s : SECTIONS)
            if (!std::strcmp(s.name, argv[2])) {
                std::vector<std::string> lines; Tx tx; tx.lines = &lines;
                s.run(tx);
                for (const auto& l : lines) puts(l.c_str());
                return 0;
            }
        fprintf(stderr, "no section %s\n", argv[2]);
        return 2;
    }
    int bad = 0;
    for (const auto& s : SECTIONS) {
        Tx tx; s.run(tx);
        const bool ok = tx.h == s.want;
        if (!ok) ++bad;
        printf("%-9s %016llx %s\n", s.name, (unsigned long long)tx.h, ok ? "ok" : "MISMATCH");
    }
    printf(bad ? "admit_vectors_test: %d MISMATCH\n" : "admit_vectors_test: OK\n", bad);
    return bad ? 1 : 0;
}
