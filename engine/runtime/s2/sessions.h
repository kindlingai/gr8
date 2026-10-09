// Per-session request/sampling/Emitter state. Exact legacy stop/min/max policy;
// no duplicate MTP acceptance. Every rank runs its Emitter, only rank0 writes.
#pragma once
#include "driver.h"
#include "ingress.h"
#include "../serve.h"
#include <cmath>
#include <functional>

namespace g6s2 {
inline std::vector<int> request_ids(const std::string& s) {
    std::vector<int> out; size_t a = 0;
    do {
        auto b = s.find(',', a); auto n = number(s.substr(a, b == std::string::npos ? b : b - a));
        require(n <= INT32_MAX, "request token integer overflow"); out.push_back(int(n));
        if (b == std::string::npos) break;
        a = b + 1;
    } while (true);
    return out;
}
inline g6rt::ServeReq parse_request(const std::string& bytes) {
    const auto line = bytes.find('\n'); require(line != std::string::npos, "request missing prompt line");
    std::istringstream in(bytes.substr(0, line));
    g6rt::ServeReq r; std::string k, gen, seed, min, stop, extra;
    require(bool(in >> k) && k == "n_gen" && bool(in >> gen >> k) && k == "temp" &&
            bool(in >> r.temp >> k) && k == "seed" && bool(in >> seed >> k) && k == "min" &&
            bool(in >> min >> k) && k == "stop" && bool(in >> stop), "invalid request header");
    // optional, in this order: top_p <float>, lp <k>. plp (prompt logprobs) is refused.
    auto next = [&] { if (!(in >> k)) k.clear(); };
    next();
    if (k == "top_p") { require(bool(in >> r.top_p), "invalid top_p header"); next(); }
    if (k == "lp") { require(bool(in >> r.lp) && r.lp >= 0 && r.lp <= g6rt::LP_MAX, "invalid lp header"); next(); }
    require(k.empty(), "invalid request header tail (S2 accepts top_p and lp)");
    auto g = number(gen), sd = number(seed), mn = number(min);
    require(g > 0 && g <= INT32_MAX && sd <= UINT32_MAX && mn <= g &&
            std::isfinite(r.temp) && r.temp >= 0 && std::isfinite(r.top_p) && r.top_p >= 0 && r.top_p <= 1,
            "invalid request sampling/length");
    r.n_gen = int(g); r.seed = uint32_t(sd); r.min_tokens = int(mn);
    if (stop != "-") r.stop = request_ids(stop);
    std::istringstream ids(bytes.substr(line + 1)); std::string prompt;
    require(bool(ids >> prompt) && !(ids >> extra), "invalid prompt line");
    r.prompt = request_ids(prompt); return r;
}
struct Sampling { float inv_temp = 0, top_p = 0; uint32_t seed = 1234; int lp = -1; };   // lp: g6rt::ServeReq::lp
struct Session {
    Key key; uint64_t request = 0;
    Run run; Payload payload; Sampling sampling; g6rt::Emitter emitter;
    bool bound = false, ended = false;
};
class Sessions {
    Config config_; std::string directory_; bool head_; int chunk_;
    std::array<Session, MAX_SESSIONS> sessions_{};
    Session& get(Key k) {
        require(k.slot < MAX_SESSIONS && sessions_[k.slot].bound && sessions_[k.slot].key == k, "stale emitter session");
        return sessions_[k.slot];
    }
    static void output_ok(const g6rt::Emitter& e) {
        require(!e.f || !ferror(e.f), "S2 response output failed");
    }
public:
    Sessions(Config c, std::string directory, bool head, int chunk) : config_(c), directory_(std::move(directory)), head_(head), chunk_(chunk) {
        c.validate(); require(chunk > 0 && unsigned(chunk) <= c.max_rows, "invalid session chunk");
    }
    ~Sessions() { for (auto& s : sessions_) if (s.emitter.f) fclose(s.emitter.f); }
    Sessions(const Sessions&) = delete; Sessions& operator=(const Sessions&) = delete;
    static Request request(uint64_t id, const g6rt::ServeReq& r) { return {id, uint32_t(r.prompt.size()), uint32_t(r.n_gen)}; }
    Payload payload(const g6rt::ServeReq& r) const { return {r.prompt, chunk_}; }
    void bind(const Grant& g, const g6rt::ServeReq& r) {
        require(g.key.epoch == config_.epoch && g.key.slot < MAX_SESSIONS, "bad session key");
        auto& s = sessions_[g.key.slot]; require(!s.bound, "session already bound");
        auto req = request(g.req.id, r);
        require(req.prompt_tokens == g.req.prompt_tokens && req.max_tokens == g.req.max_tokens &&
                reserve_pages(config_, req) == g.run.pages, "session payload/grant mismatch");
        s = Session{}; s.key = g.key; s.request = g.req.id; s.run = g.run;
        s.payload = payload(r); s.sampling = {r.temp > 0 ? 1.f / r.temp : 0.f, r.top_p, r.seed, r.lp};
        s.emitter.open(directory_, g.req.id, head_, r); s.bound = true;
        require(!head_ || s.emitter.f, "cannot open S2 response");
        if (s.emitter.f) {
            fprintf(s.emitter.f, "a %llu %u %u %u %u\n", (unsigned long long)g.key.epoch, g.key.slot,
                    g.key.generation, g.run.base_page, g.run.pages);
            require(!fflush(s.emitter.f), "S2 admission output failed"); output_ok(s.emitter);
        }
    }
    const Session& at(unsigned slot) const {
        require(slot < MAX_SESSIONS && sessions_[slot].bound, "unbound session"); return sessions_[slot];
    }
    // row: the row of ids[0] in the last main step; logprob(row, k) reads that step's logprobs (sessions with lp only)
    Emission emit(Key k, const int* ids, int count, int row, const std::function<g6rt::LpRow(int, int)>& logprob) {
        auto& s = get(k); require(!s.ended && count > 0, "emit after terminal/empty emission");
        std::vector<g6rt::LpRow> lps;
        if (s.sampling.lp >= 0) {
            require(bool(logprob), "logprobs requested without a logprob source");
            for (int i = 0; i < count; ++i) lps.push_back(logprob(row + i, s.sampling.lp));
        }
        const int before = s.emitter.emitted;
        const bool done = s.emitter.push(ids, count, lps.empty() ? nullptr : lps.data()); output_ok(s.emitter);
        return {uint32_t(s.emitter.emitted - before), done};
    }
    void terminal(Key k, const s2::Stats& stats, const char* reason = nullptr) {
        auto& s = get(k); if (s.ended) return;
        // Emitter::end uses per-request counters here, never process-wide totals.
        require(stats.acc.empty() || stats.acc.size() == config_.mtp + 1, "bad per-request acceptance histogram");
        std::array<long, 8> zero{};
        auto& em = s.emitter;
        if (em.f) {
            const long* acc = stats.acc.empty() ? zero.data() : stats.acc.data();
            fprintf(em.f, "m %ld %ld %ld", stats.rounds, stats.rounds * config_.mtp, stats.tokens - stats.rounds);
            for (unsigned i = 0; i < config_.mtp; ++i) {
                long accepted = 0; for (unsigned a = i + 1; a <= config_.mtp; ++a) accepted += acc[a];
                fprintf(em.f, " %ld", accepted);
            }
            fprintf(em.f, "\nend %s %d %d\n", reason ? reason : em.stop_id >= 0 ? "stop" : "length", em.stop_id, em.emitted);
            const bool ok = !ferror(em.f); const int rc = fclose(em.f); em.f = nullptr;
            require(ok && !rc, "S2 terminal output failed");
        }
        em.on = false;
        s.ended = true;
    }
    void retire(const Slot& old) {
        auto& s = get(old.key);
        // CANCEL can remove a prefilling slot that has no Batch stats yet.
        if (!s.ended) terminal(old.key, {}, "abort");
        s = Session{};
    }
};
} // namespace g6s2
