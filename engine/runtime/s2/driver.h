// Component 5 bridge to component 6's persistent Batch. No CUDA or allocator.
// All calls run on the compute thread at safe boundaries. apply() requires an
// EventGate all-rank fence; callbacks must fail the whole epoch on any error.
#pragma once
#include "scheduler.h"
#include "../s2_batch.h"

namespace g6s2 {
// decode_prompt: the whole prompt runs as decode steps from position 0, no tensor prefill (prompt logprobs)
struct Payload { std::vector<int> prompt; int chunk = 4; bool decode_prompt = false; };
struct Emission { uint32_t count = 0; bool finished = false; };
struct DriverHooks {
    // Real TP4 pool/binding adapter: canonical base, never a per-rank first-fit.
    std::function<void(const Grant&)> bind;
    std::function<void(const Slot&)> retire;
    // Full prefill quantum: both lanes, main AND draft consumption, synchronized.
    std::function<void(const Slot&, const Payload&, uint32_t, uint32_t)> prefill;
    std::function<std::vector<int>(const s2::Step&)> run;
    // Exact native Emitter policy (stop/min/max) on EVERY rank, not just rank0.
    std::function<Emission(Key, const int*, int, int)> emit;   // (key, ids, count, row of ids[0] in the last main step)
    // optional: after each prompt main step of a session, (key, rows, position of row 0)
    std::function<void(Key, int, int)> prompt;
    std::function<void(Key, long, int, int, int)> round;
    // optional, DFlash2: may a session alone at its prompt tail use the drafter (it owns the prefill's taps); the
    // drafter's prompt context (returns the lowest position it holds); one drafter step
    std::function<bool(Key)> dfl_ok;
    std::function<int(Key)> dfl_start;
    std::function<s2::DflDrafts(Key, const s2::DflReq&)> dfl;
    // One completion notification; physical KV release waits for the event fence.
    std::function<void(Key, const s2::Stats&)> terminal;
};
class Driver {
    Scheduler scheduler_;
    DriverHooks hooks_;
    std::array<Payload, MAX_SESSIONS> payload_;
    std::array<uint32_t, MAX_SESSIONS> emitted_{};
    std::array<bool, MAX_SESSIONS> initialized_{};
    s2::Batch batch_;
    s2::Hooks batch_hooks() {
        return {
            [this](const s2::Step& s) {
                auto ids = hooks_.run(s);
                require(ids.size() == s.r.size(), "engine returned wrong row count");
                return ids;
            },
            [this](int q, long round, int base, int accepted, int K) {
                if (hooks_.round) hooks_.round(scheduler_.slots()[q].key, round, base, accepted, K);
            },
            [this](int q, int M, int pos0) {
                if (hooks_.prompt) hooks_.prompt(scheduler_.slots()[q].key, M, pos0);
            },
            [this](int q, const int* ids, int n, int row) {
                const auto& s = scheduler_.slots()[q];
                auto e = hooks_.emit(s.key, ids, n, row);
                require(e.count <= unsigned(n) && e.count <= s.req.max_tokens - emitted_[q], "bad emitter count");
                require(e.finished || e.count == unsigned(n), "partial emission without stop");
                emitted_[q] += e.count;
                return e.finished || emitted_[q] == s.req.max_tokens;
            },
            hooks_.dfl ? std::function<s2::DflDrafts(const s2::DflReq&)>([this](const s2::DflReq& r) {
                return hooks_.dfl(scheduler_.slots()[r.sess].key, r);
            }) : nullptr,
            hooks_.dfl_start ? std::function<int(int)>([this](int q) { return hooks_.dfl_start(scheduler_.slots()[q].key); }) : nullptr
        };
    }
    // sessions in Prefill or Decode (Dispatch::synchronize_mode's count: 1 = the solo path)
    unsigned live() const {
        unsigned n = 0;
        for (const auto& s : scheduler_.slots()) n += s.phase == Phase::Prefill || s.phase == Phase::Decode;
        return n;
    }
    Completion completion(Key k) const {
        const auto& s = scheduler_.slots()[k.slot];
        return {k, emitted_[k.slot] - s.emitted, batch_.finished(int(k.slot))};
    }
public:
    Driver(Config c, DriverHooks h) : scheduler_(c), hooks_(std::move(h)),
        batch_(int(c.mtp), int(c.max_rows), batch_hooks()) {
        batch_.adapt(s2::adapt_env());
        require(bool(hooks_.bind) && bool(hooks_.retire) && bool(hooks_.prefill) && bool(hooks_.run) && bool(hooks_.emit), "missing driver callback");
    }
    const Scheduler& state() const { return scheduler_; }
    const s2::Stats& stats(Key k) const {
        require(k.slot < MAX_SESSIONS && scheduler_.slots()[k.slot].key == k && initialized_[k.slot], "stats for stale/uninitialized slot");
        return batch_.stats(int(k.slot));
    }
    const s2::Stats* stats_if_initialized(Key k) const {
        require(k.slot < MAX_SESSIONS && scheduler_.slots()[k.slot].key == k, "stats for stale slot");
        return initialized_[k.slot] ? &batch_.stats(int(k.slot)) : nullptr;
    }
    // a live or retiring slot's prompt and the tokens it emitted (prefix reuse history)
    const std::vector<int>& prompt(unsigned slot) const { require(slot < MAX_SESSIONS, "invalid slot"); return payload_[slot].prompt; }
    std::vector<int> emitted(unsigned slot) const {
        require(slot < MAX_SESSIONS, "invalid slot");
        if (!initialized_[slot]) return {};
        auto t = batch_.result(int(slot));
        t.resize(std::min<size_t>(t.size(), emitted_[slot]));
        return t;
    }
    // DFlash2 sessions verify up to kv drafts with the --spec-adaptive policy sl (s2_batch.h Batch::dflash)
    void dflash(int kv, const g6rt::SpecLen& sl) { batch_.dflash(kv, sl); }
    void fail() { scheduler_.fail(); }
    void idle_boundary(uint64_t b) { scheduler_.idle_boundary(b); }
    Changes apply(const Commit& c, const Payload* p = nullptr) {
        if (c.admit) {
            require(p && p->prompt.size() == c.admit->req.prompt_tokens && p->chunk > 0, "admit payload mismatch");
            require(c.admit->req.max_tokens <= INT32_MAX && p->prompt.size() <= INT32_MAX, "Batch integer range");
        } else require(!p, "payload without grant");
        auto before = scheduler_.slots();
        auto changes = scheduler_.apply(c); // validates the entire transaction first
        try {
            for (const auto& old : before) {
                const auto& now = scheduler_.slots()[old.key.slot];
                if (old.phase == Phase::Free || (now.phase != Phase::Free && old.key == now.key)) continue;
                hooks_.retire(old); // individual run only; NEVER reset the whole cache. Reads the session's tokens (prefix reuse)
                if (initialized_[old.key.slot]) batch_.release(int(old.key.slot));
                initialized_[old.key.slot] = false;
                payload_[old.key.slot] = Payload{}; emitted_[old.key.slot] = 0;
            }
            if (c.admit) {
                payload_[c.admit->key.slot] = *p;
                hooks_.bind(*c.admit);
                if (p->decode_prompt) scheduler_.skip_prefill(c.admit->key);
            }
        } catch (...) { scheduler_.fail(); throw; }
        return changes;
    }
    Quantum step() {
        auto q = scheduler_.next();
        try {
            if (q.kind == Kind::Prefill) {
                auto k = q.keys[0];
                hooks_.prefill(scheduler_.slots()[k.slot], payload_[k.slot], q.first, q.rows);
                scheduler_.complete_prefill();
            } else if (q.kind == Kind::PromptTail) {
                auto k = q.keys[0]; const auto& s = scheduler_.slots()[k.slot];
                const auto& p = payload_[k.slot];
                require(!initialized_[k.slot], "duplicate prompt tail");
                const bool dfl = hooks_.dfl_ok && live() == 1 && !p.decode_prompt && hooks_.dfl_ok(k);
                batch_.admit({int(k.slot)}, {{p.prompt, p.decode_prompt ? 0 : int(q.first), p.chunk, int(s.req.max_tokens), dfl}});
                initialized_[k.slot] = true;
                scheduler_.complete_tail(completion(k));
            } else if (q.kind == Kind::Decode) {
                std::vector<int> order;
                for (unsigned i = 0; i < q.count; ++i) {
                    require(initialized_[q.keys[i].slot] && batch_.active(int(q.keys[i].slot)), "scheduler/Batch active mismatch");
                    order.push_back(int(q.keys[i].slot));
                }
                if (live() > 1) batch_.demote();   // another session is resident (perhaps still prefilling): the shared path
                require(batch_.round(order), "empty scheduled decode round");
                std::vector<Completion> results;
                for (unsigned i = 0; i < q.count; ++i) results.push_back(completion(q.keys[i]));
                scheduler_.complete_decode(results);
            }
            if (q.kind == Kind::PromptTail || q.kind == Kind::Decode)
                for (unsigned i = 0; i < q.count; ++i) {
                    auto k = q.keys[i];
                    if (scheduler_.slots()[k.slot].phase == Phase::Retiring && hooks_.terminal)
                        hooks_.terminal(k, batch_.stats(int(k.slot)));
                }
        } catch (...) { scheduler_.fail(); throw; }
        return q;
    }
};
} // namespace g6s2
