// Per-session request/sampling/Emitter state. Exact legacy stop/min/max policy;
// no duplicate MTP acceptance. Every rank runs its Emitter, only rank0 writes.
#pragma once
#include "driver.h"
#include "ingress.h"
#include "../serve.h"
#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include "prompt_rule.h"

namespace g6s2 {
// The request file's parser is engine/scheduler-rs/src/request.rs. A request is the header line
// `n_gen <n> temp <f> seed <n> min <n> stop <ids|->`, then optionally and in this order `top_p <f>`, `lp <k>`,
// `plp <k>`, `sys <n>`, `ins <rule>`, `gram <grammar>` and `steer <profile> <canonical-fp32-hex>`, then any number of
// `img <field>` (DS41 images, ds41_image.h: parsed here through parse::image, checked after the prompt), then the
// prompt ids.
namespace parse {
using Handle = std::unique_ptr<g6sch_parsed, void (*)(g6sch_parsed*)>;
inline std::vector<int> ints(const g6sch_ints& v) { return std::vector<int>(v.p, v.p + v.n); }
// the steering fields to the request's identity (BootConfig::resolve), its refusal kept for the parser to return;
// images: the request's `img` fields (parse::image), in order
struct Resolve { const g6rt::steering::BootConfig* boot; std::string error; std::vector<g6rt::ImageIn> images; };
inline const char* resolve(void* ctx, bool has_profile, const uint8_t* profile, size_t n, uint32_t bits, g6sch_identity* out) {
    auto& r = *static_cast<Resolve*>(ctx);
    try {
        g6rt::steering::RequestFields f;
        if (has_profile) {
            f.has_profile = f.has_scale = true;
            f.profile.assign(reinterpret_cast<const char*>(profile), n);
            f.scale = g6rt::steering::bits_float(bits);
        }
        const g6rt::steering::Identity id = r.boot->resolve(f);
        std::memcpy(out, &id, sizeof *out);
        return nullptr;
    } catch (const std::exception& e) {
        r.error = e.what();
        return r.error.c_str();
    } catch (...) {   // nothing may unwind into the Rust caller
        r.error = "steering resolve failed";
        return r.error.c_str();
    }
}
// one `img` field (ds41_image.h parse_field), kept in the context; its refusal kept for the parser to return
inline const char* image(void* ctx, const uint8_t* field, size_t n) {
    auto& r = *static_cast<Resolve*>(ctx);
    try {
        r.images.push_back(g6rt::img::parse_field(std::string(reinterpret_cast<const char*>(field), n)));
        return nullptr;
    } catch (const std::exception& e) {
        r.error = e.what();
        return r.error.c_str();
    } catch (...) {   // nothing may unwind into the Rust caller
        r.error = "img field failed";
        return r.error.c_str();
    }
}
} // namespace parse
// `gram <kind>:<strict>:<arm or -1>:<hex>` (grammar.h): kind 1 = JSON, 2 = the JSON schema whose UTF-8 bytes are <hex>
inline g6rt::GrammarSpec parse_grammar(const std::string& s) {
    uint32_t kind = 0; bool strict = false; int32_t arm = -1; const uint8_t* schema = nullptr; size_t n = 0;
    g6sch_parsed* h = nullptr;
    check(g6sch_parse_grammar(reinterpret_cast<const uint8_t*>(s.data()), s.size(), &kind, &strict, &arm, &schema, &n, &h));
    const parse::Handle own(h, g6sch_parsed_free);
    g6rt::GrammarSpec g;
    g.kind = g6rt::GrammarSpec::Kind(kind); g.strict = strict; g.arm = arm;
    g.schema.assign(reinterpret_cast<const char*>(schema), n);
    return g;
}
inline g6rt::ServeReq parse_request(const std::string& bytes, const g6rt::steering::BootConfig& steering = {}) {
    static_assert(sizeof(g6rt::steering::Identity) == sizeof(g6sch_identity), "scheduler-rs Identity layout");
    sch::check_abi();
    parse::Resolve ctx{&steering, {}, {}};
    g6sch_parsed_req p{};
    g6sch_parsed* h = nullptr;
    check(g6sch_parse_request(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size(), g6rt::grammar_ready(),
                              parse::resolve, parse::image, &ctx, &p, &h));
    const parse::Handle own(h, g6sch_parsed_free);
    g6rt::ServeReq r;
    r.prompt = parse::ints(p.prompt); r.stop = parse::ints(p.stop);
    r.n_gen = p.n_gen; r.min_tokens = p.min_tokens; r.temp = p.temp; r.top_p = p.top_p; r.seed = p.seed;
    r.lp = p.lp; r.plp = p.plp; r.sys = p.sys;
    std::memcpy(static_cast<void*>(&r.steering), &p.steering, sizeof r.steering);
    r.ins.after = p.ins_after; r.ins.unless = p.ins_unless; r.ins.tokens = parse::ints(p.ins_tokens);
    r.gram.kind = g6rt::GrammarSpec::Kind(p.gram_kind); r.gram.strict = p.gram_strict; r.gram.arm = p.gram_arm;
    r.gram.schema.assign(reinterpret_cast<const char*>(p.schema), p.schema_n);
    // the C++ parser's last step, after the prompt line: the images against the prompt, and their cache identity
    r.images = std::move(ctx.images);
    if (!r.images.empty()) {
        try { g6rt::img::check(r.prompt, r.images); } catch (const std::exception& e) { require(false, e.what()); }
        r.steering.images = true; r.steering.image_digest = g6rt::img::digest(r.images);
    }
    return r;
}
struct Sampling { float inv_temp = 0, top_p = 0; uint32_t seed = 1234; int lp = -1, plp = -1; };   // lp, plp: g6rt::ServeReq
struct Session {
    Key key; uint64_t request = 0;
    Run run; Payload payload; Sampling sampling; g6rt::Emitter emitter;
    g6rt::GrammarRef grammar;         // the reply's grammar (Payload::gram), on every rank; null = none
    bool bound = false, ended = false;
};
class Sessions {
    Config config_; std::string directory_; bool head_; int chunk_; [[maybe_unused]] uint32_t prefill_min_;
    std::array<Session, MAX_SESSIONS> sessions_{};
    Session& get(Key k) {
        require(k.slot < MAX_SESSIONS && sessions_[k.slot].bound && sessions_[k.slot].key == k, "stale emitter session");
        return sessions_[k.slot];
    }
    static void output_ok(const g6rt::Emitter& e) {
        require(!e.f || !ferror(e.f), "S2 response output failed");
    }
public:
    // prefill_min: DS41 only (ds41_prefill.h); 0 = every prompt runs as decode steps
    Sessions(Config c, std::string directory, bool head, int chunk, uint32_t prefill_min = 0)
        : config_(c), directory_(std::move(directory)), head_(head), chunk_(chunk), prefill_min_(prefill_min) {
        c.validate(); require(chunk > 0 && unsigned(chunk) <= c.max_rows, "invalid session chunk");
    }
    ~Sessions() { for (auto& s : sessions_) if (s.emitter.f) fclose(s.emitter.f); }
    Sessions(const Sessions&) = delete; Sessions& operator=(const Sessions&) = delete;
    static Request request(uint64_t id, const g6rt::ServeReq& r) {
        return {id, uint32_t(r.prompt.size()), uint32_t(r.n_gen), r.steering};
    }
    // G6_CAP_PROMPT_DECODE (DS41): no tensor prefill by default, every prompt runs as decode steps from position 0
    // (Driver decode_prompt) unless prefill_min admits it (ds41_prefill.h); otherwise only prompt logprobs keep a prompt
    // on the decode path
    Payload payload(const g6rt::ServeReq& r) const {
        const bool decode = g6mod::has(G6_CAP_PROMPT_DECODE) ? ds41_decode_prompt(prefill_min_, r.prompt.size(), r.plp, r.steering.enabled)
                                                             : r.plp >= 0;
        // r.images is empty outside DS41: Service::images_ok refuses an image request there before it is queued
        return {r.prompt, chunk_, decode, r.ins, r.steering, r.gram, r.images};
    }
    // (row, target, k) -> a row's logprob entry in the last main step (DispatchHooks::logprob)
    using Logprob = std::function<g6rt::LpRow(int, bool, int)>;
    void bind(const Grant& g, const g6rt::ServeReq& r) {
        require(g.key.epoch == config_.epoch && g.key.slot < MAX_SESSIONS, "bad session key");
        auto& s = sessions_[g.key.slot]; require(!s.bound, "session already bound");
        auto req = request(g.req.id, r);
        require(req.prompt_tokens == g.req.prompt_tokens && req.max_tokens == g.req.max_tokens &&
                req.steering == g.req.steering &&
                grant_pages_ok(config_, req, g.run.pages), "session payload/grant mismatch");
        // the kernels index the profile tables by this slot
        require(!req.steering.enabled || (req.steering.slot < config_.steering.profiles.size() &&
                req.steering == config_.steering.selected(req.steering.scale_bits, req.steering.slot)),
                "session steering is not a loaded profile");
        s = Session{}; s.key = g.key; s.request = g.req.id; s.run = g.run;
        s.payload = payload(r); s.sampling = {r.temp > 0 ? 1.f / r.temp : 0.f, r.top_p, r.seed, r.lp, r.plp};
        // the head compiled this grammar before queueing the request (Service::controller), so a failure here means
        // the ranks disagree, and the epoch ends
        if (r.gram.on()) s.grammar = g6rt::GrammarRef(g6rt::grammar_open(r.gram).release(), g6rt::GrammarDeleter{});
        s.emitter.open(directory_, g.req.id, head_, r); s.bound = true;
        require(!head_ || s.emitter.f, "cannot open S2 response");
        if (s.emitter.f) {
            fprintf(s.emitter.f, "a %llu %u %u %u %u %u\n", (unsigned long long)g.key.epoch, g.key.slot,
                    g.key.generation, g.run.first_page, g.run.pages, g.reuse);
            require(!fflush(s.emitter.f), "S2 admission output failed"); output_ok(s.emitter);
        }
    }
    // the session's run grew (Driver::step): Dispatch checks each step's rows against it
    void grow(unsigned slot, const Run& r) {
        require(slot < MAX_SESSIONS && sessions_[slot].bound && sessions_[slot].run.table_version == r.table_version, "growth of an unbound session");
        sessions_[slot].run = r;
    }
    const Session& at(unsigned slot) const {
        require(slot < MAX_SESSIONS && sessions_[slot].bound, "unbound session"); return sessions_[slot];
    }
    // the "p" lines of a prompt main step: rows 0..M-1 at positions pos0.., each scored against the next prompt token
    void prompt_rows(Key k, int M, int pos0, const Logprob& logprob) {
        auto& s = get(k);
        if (s.sampling.plp < 0) return;
        require(bool(logprob), "prompt logprobs requested without a logprob source");
        for (int r = 0; r < M && pos0 + r + 1 < int(s.payload.prompt.size()); ++r)
            s.emitter.push_prompt(pos0 + r + 1, logprob(r, true, s.sampling.plp));
        output_ok(s.emitter);
    }
    // after a prefill quantum: "f <rows>", the prompt rows prefilled so far (the front end takes it as life, not text)
    void progress(Key k, uint32_t rows) {
        auto& s = get(k);
        if (!s.emitter.f) return;
        fprintf(s.emitter.f, "f %u\n", rows);
        require(!fflush(s.emitter.f), "S2 progress output failed"); output_ok(s.emitter);
    }
    // row: the row of ids[0] in the last main step
    Emission emit(Key k, const int* ids, int count, int row, const Logprob& logprob) {
        auto& s = get(k); require(!s.ended && count > 0, "emit after terminal/empty emission");
        std::vector<g6rt::LpRow> lps;
        if (s.sampling.lp >= 0) {
            require(bool(logprob), "logprobs requested without a logprob source");
            for (int i = 0; i < count; ++i) lps.push_back(logprob(row + i, false, s.sampling.lp));
        }
        const int before = s.emitter.emitted;
        if (s.grammar) g6rt::grammar_commit(*s.grammar, ids, count);
        const bool done = s.emitter.push(ids, count, lps.empty() ? nullptr : lps.data()); output_ok(s.emitter);
        return {uint32_t(s.emitter.emitted - before), done};
    }
    void terminal(Key k, const s2::Stats& stats, const char* reason = nullptr) {
        auto& s = get(k); if (s.ended) return;
        // Emitter::end uses per-request counters here, never process-wide totals.
        // K + 1 bins: K = the MTP draft count, or 7 for a session that started on DFlash2 (s2_batch.h)
        require(stats.acc.empty() || (stats.acc.size() >= config_.mtp + 1 && stats.acc.size() <= 16), "bad per-request acceptance histogram");
        std::array<long, 8> zero{};
        auto& em = s.emitter;
        if (em.f) {
            const long* acc = stats.acc.empty() ? zero.data() : stats.acc.data();
            const unsigned K = stats.acc.empty() ? config_.mtp : unsigned(stats.acc.size() - 1);
            fprintf(em.f, "m %ld %ld %ld", stats.rounds, stats.drafted, stats.tokens - stats.rounds);
            for (unsigned i = 0; i < K; ++i) {
                long accepted = 0; for (unsigned a = i + 1; a <= K; ++a) accepted += acc[a];
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
