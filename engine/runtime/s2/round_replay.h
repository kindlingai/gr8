// round_replay.h: rerun the planner (s2_batch.h Batch) from a round log (GR8_S2_REPLAY_LOG, s2_log.h) and check that it
// asks for the same steps. Host code only.
//
// The log holds the planner's settings, each driver call, and each hook's request and answer. The replay builds a Batch
// from the settings and makes the driver calls in order. Each hook call takes the next hook record: the request the
// planner makes now must equal the logged one, and the logged answer is returned. After each call the state the driver
// reads back (Batch::state: which sessions are active or finished, their lengths and committed tokens) must equal the
// logged state. So a replay passes when this build of the planner makes every decision the logged one made, over every
// round of every request in the log.
#pragma once
#include <cstdio>
#include <string>
#include <vector>
#include "../s2_batch.h"

namespace s2 {

struct ReplayResult
{
    bool ok = false;
    long calls = 0, steps = 0, hooks = 0;
    size_t line = 0;                  // the log line where the replay stopped (1-based; 0: none)
    int version = 0;                  // the log's format
    std::string libc;                 // version 3: the C library the logging planner ran on (libc_version())
    std::string learn_text;           // dump_learn: the replaying planner's learned state at that log line (Batch::learn_text)
    std::string error;
};

namespace replay_detail {
inline bool starts(const std::string& l, const std::string& p)
{
    return l.compare(0, p.size(), p) == 0 && (l.size() == p.size() || l[p.size()] == ' ');
}
inline std::string tag(const std::string& l) { return l.substr(0, l.find(' ')); }
}   // namespace replay_detail

// replays the log's lines. table: the n-gram table when the log's planner had one (its digest must match)
// learn = false (round_replay --steps-only): skip the learn records, so a build that rounds the estimates differently
// (contracted multiply-adds) still shows whether any step, hook or driver state changes
// dump_learn: the 1-based log line of a learn record whose state to write out (learn_text), 0 = none
inline ReplayResult replay(const std::vector<std::string>& lines, const g6rt::NgramTable* table = nullptr, bool learn = true,
                           size_t dump_learn = 0)
{
    using namespace replay_detail;
    ReplayResult res;
    size_t i = 0;
    auto fail = [&](const std::string& w) { throw std::runtime_error(w); };
    try
    {
        if (lines.empty()) fail("empty log");
        s2log::In h(lines[0].c_str());
        if (h.word() != "g6s2log") fail("not a round log");
        int ver = 0, K = 0, max_rows = 0, slots = 0, flash_tp = 0;
        bool par = false, fresh = false, flash_cost = false, has_dfl = false, has_dfl_start = false, has_lp = false;
        bool has_round = false, has_prompt = false;
        h(ver);
        if (ver != 2 && ver != 3) fail("log version " + std::to_string(ver) + ", want 2 or 3");
        res.version = ver;
        h(K)(max_rows)(slots)(par)(fresh)(flash_cost)(flash_tp)(has_dfl)(has_dfl_start)(has_lp);
        // version 3: the round and prompt hooks are logged, the step costs and C library are in the header, and every
        // state record is followed by a learn record
        const bool v3 = ver >= 3;
        if (v3)
        {
            S2Cost cost;
            uint32_t model = 0;
            h(has_round)(has_prompt)(cost)(res.libc)(model);
            if (model != g6mod::INFO.model)
                fail("the log is model " + std::to_string(model) + "'s, this build model " + std::to_string(g6mod::INFO.model) +
                     "'s (module_info.h): replay with the same model's build");
            if (s2log::line("", cost) != s2log::line("", S2Cost{}))
                fail("the log's step costs (S2Cost) differ from this build's: replay with the build that logged it");
        }
        if (slots != S2N_SLOTS)
            fail("the log's planner has " + std::to_string(slots) + " slots, this build " + std::to_string(S2N_SLOTS) +
                 ": replay with the same model's build");
#if G6_MODEL_FLASH
        if (flash_cost != knob::glm53f_s2_cost()) fail("GR8_GLM53F_S2_COST differs from the log's: set it as the log has it");
        if (flash_cost && flash_tp != S2Cost::flash_tp)
            fail("the log's Flash cost table is TP" + std::to_string(flash_tp) + "'s: build with -DG6_NR=" + std::to_string(flash_tp));
#else
        if (flash_cost) fail("the log is a Flash planner's: replay with the Flash build");
#endif
        i = 1;
        // the next hook record: its request must be `want`; returns the reader positioned at the answer
        auto next = [&](const std::string& want) -> s2log::In {
            if (i >= lines.size()) fail("the planner made a call the log does not have: " + want.substr(0, 200));
            const std::string& l = lines[i];
            if (!starts(l, want))
                fail("the planner asked for\n  " + want.substr(0, 400) + "\nthe log has\n  " + l.substr(0, 400));
            ++i;
            ++res.hooks;
            return s2log::In(l.c_str() + want.size());
        };
        Hooks hk;
        hk.run = [&](const Step& s) {
            s2log::In in = next(s2log::line("run", s));
            std::vector<int> t;
            in(t);
            ++res.steps;
            return t;
        };
        hk.emit = [&](int q, const int* t, int k, int row) {
            s2log::In in = next(s2log::line("emit", q, std::vector<int>(t, t + k), row));
            bool done = false;
            in(done);
            return done;
        };
        if (has_dfl)
            hk.dfl = [&](const DflReq& r) {
                s2log::In in = next(s2log::line("dfl", r));
                DflDrafts o{};
                in(o);
                return o;
            };
        if (has_dfl_start)
            hk.dfl_start = [&](int q) {
                s2log::In in = next(s2log::line("dfl_start", q));
                int lo = 0;
                in(lo);
                return lo;
            };
        if (has_lp)
            hk.lp = [&](int r) {
                s2log::In in = next(s2log::line("lp", r));
                float v = 0;
                in(v);
                return v;
            };
        if (has_round)
            hk.round = [&](int q, long n, int base, int a, int k) { next(s2log::line("on_round", q, n, base, a, k)); };
        if (has_prompt) hk.prompt = [&](int q, int M, int pos0) { next(s2log::line("on_prompt", q, M, pos0)); };
        Batch b(K, max_rows, hk);
        b.draft_par(par);
        b.fresh_spec(fresh);
        while (i < lines.size())
        {
            const std::string& l = lines[i];
            const std::string t = tag(l);
            s2log::In in(l.c_str());
            in.word();
            const size_t at = i++;
            ++res.calls;
            res.line = at + 1;
            if (t == "round") { std::vector<int> o; in(o); b.round(o); }
            else if (t == "admit" || t == "admit_packed_tail")
            {
                std::vector<int> qs;
                std::vector<Session> ss;
                in(qs)(ss);
                if (t == "admit") b.admit(qs, ss); else b.admit_packed_tail(qs, ss);
            }
            else if (t == "cancel") { int q; in(q); b.cancel(q); }
            else if (t == "promote") { int q; in(q); b.promote(q); }
            else if (t == "release") { int q; in(q); b.release(q); }
            else if (t == "demote") b.demote();
            else if (t == "mtp_sessions") { int n; in(n); b.set_mtp_sessions(n); }
            else if (t == "sess_drafts") { bool v; in(v); b.set_sess_drafts(v); }
            else if (t == "adapt") { bool v; in(v); b.adapt(v); }
            else if (t == "draft_par") { bool v; in(v); b.draft_par(v); }
            else if (t == "fresh_spec") { bool v; in(v); b.fresh_spec(v); }
            else if (t == "dflash") { int kv; g6rt::SpecLen sl; in(kv)(sl); b.dflash(kv, sl); }
            else if (t == "dsel") { DSel d; in(d); b.dsel(d); }
            else if (t == "kstop") { KStop k; in(k); b.kstop(k); }
            else if (t == "dfl_reuse") { DflReuse r; in(r); b.dfl_reuse(r); }
            else if (t == "ngram")
            {
                NgramSrc n;
                unsigned long long digest = 0;
                in(n)(digest);
                if (digest)
                {
                    if (!table || table->digest != digest) fail("the log's planner had an n-gram table: pass the same table");
                    n.t = table;
                }
                b.ngram(n);
            }
            else fail("a hook record the planner did not ask for, or an unknown call: " + l.substr(0, 200));
            const std::string st = b.state();
            if (i >= lines.size() || lines[i] != st)
                fail("after " + t + " the planner's state is\n  " + st.substr(0, 400) + "\nthe log has\n  " +
                     (i < lines.size() ? lines[i].substr(0, 400) : std::string("(end of log)")));
            ++i;
            if (!v3) continue;
            if (dump_learn == i + 1) res.learn_text = b.learn_text();
            if (!learn) { if (i < lines.size() && starts(lines[i], "learn")) ++i; continue; }
            const std::string lr = s2log::line("learn", b.learn());
            if (i >= lines.size() || lines[i] != lr)
                fail("after " + t + " the planner's learned state differs: " + lr + ", the log has " +
                     (i < lines.size() ? lines[i].substr(0, 100) : std::string("(end of log)")));
            ++i;
        }
        res.line = 0;
        res.ok = true;
    }
    catch (const std::exception& e)
    {
        res.line = i + 1;
        res.error = e.what();
    }
    return res;
}

inline std::vector<std::string> read_lines(std::FILE* f)
{
    std::vector<std::string> out;
    std::string cur;
    int c;
    while ((c = std::fgetc(f)) != EOF)
    {
        if (c == '\n') { out.push_back(cur); cur.clear(); }
        else cur += (char)c;
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

}   // namespace s2
