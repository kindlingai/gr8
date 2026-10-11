// S2 v1 step 1: host-only row/transaction reference. NOT wired to Driver, Batch,
// Service or CUDA. TP4 still owns allocation; callbacks still own real KV/hprev.
#pragma once
#include "scheduler.h"
#include "../../g6mx.h"
#include <set>
#include <tuple>

namespace g6s2::v1 {
constexpr unsigned MX = G6_MX, RESIDENT = 4;
inline unsigned request_k(std::optional<int> k, unsigned ceiling) {
    require(ceiling >= 1 && ceiling <= 7, "bad loaded k ceiling");
    const int v = k.value_or(int(ceiling));
    require(v >= 1 && unsigned(v) <= ceiling, "request k outside loaded ceiling");
    return unsigned(v); // never clamp; omission resets on every request
}
inline uint32_t pages(uint64_t prompt, uint64_t output, unsigned k, uint32_t pool) {
    require(k >= 1 && k <= 7 && prompt && output, "bad reservation input");
    require(prompt <= pool && output <= pool, "reservation overflow/pool limit");
    const uint64_t rows = prompt + output + 2 * k; // both terms bounded by uint32 pool
    require(rows <= pool, "full reservation exceeds pool");
    return uint32_t((rows + PAGE - 1) / PAGE);
}
struct Runnable { uint64_t ordinal; unsigned slot, k; };
struct Selection { std::vector<unsigned> slots; uint64_t first; unsigned rows; };
inline Selection select(std::vector<Runnable> live, std::optional<uint64_t> cursor = {}) {
    require(!live.empty() && live.size() <= RESIDENT, "bad runnable count");
    std::sort(live.begin(), live.end(), [](auto a, auto b) { return a.ordinal < b.ordinal; });
    std::set<unsigned> slots;
    for (size_t i=0; i<live.size(); ++i) {
        auto r=live[i];
        require(r.slot < MAX_SESSIONS && slots.insert(r.slot).second && r.k>=1 && r.k<=7, "bad runnable");
        require(!i || live[i-1].ordinal != r.ordinal, "duplicate ordinal");
    }
    size_t first=0;
    if (cursor) {
        while (first<live.size() && live[first].ordinal<=*cursor) ++first;
        if (first==live.size()) first=0;
    }
    Selection out{{}, live[first].ordinal, 0};
    for (size_t j=0; j<live.size(); ++j) {
        auto r=live[(first+j)%live.size()];
        if (out.rows + r.k + 1 <= MX) { out.slots.push_back(r.slot); out.rows += r.k + 1; }
    }
    return out;
}
struct Ready {
    Key key; Run run; uint64_t ordinal;
    unsigned k, max_output, prompt_size;
    std::vector<int> history; // prompt + ACTUALLY emitted IDs, never rejected candidates
    uint32_t main_end, draft_end; // exclusive committed/materialized valid ends
    uint64_t rounds=0;
    bool terminal=false;
};
struct Span { Key key; Run run; unsigned row0, rows, base, context, cls; };
struct Plan { uint64_t revision; Selection selection; std::vector<Span> spans; bool true_solo; };
struct Verify { Key key; std::vector<int> draft, target; unsigned emitted; bool finished; };
struct CommitResult { Key key; unsigned accepted, emitted; uint32_t main_end, draft_end; bool terminal; };

class DecodeState {
    uint64_t epoch_, revision_=0, last_ordinal_=0;
    unsigned ceiling_, pool_;
    bool have_ordinal_=false, busy_=false, failed_=false;
    std::optional<uint64_t> cursor_;
    std::array<std::optional<Ready>, MAX_SESSIONS> live_{};
    std::array<uint32_t, MAX_SESSIONS> generations_{};
    std::optional<Plan> pending_;
    void fence() const { require(!busy_ && !failed_, "state requires quiescent live fence"); }
public:
    DecodeState(uint64_t epoch, unsigned ceiling, unsigned pool=614400)
        : epoch_(epoch), ceiling_(ceiling), pool_(pool) {
        require(epoch && pool>=614400 && pool%PAGE==0, "bad v1 epoch/pool");
        request_k({}, ceiling);
    }
    const auto& slots() const { return live_; }
    bool failed() const { return failed_; }
    unsigned resident() const { unsigned n=0; for (const auto& x:live_) n+=bool(x); return n; }
    // Prepared prompt/tail only: real cold prefill/OWN2 resume is NOT implemented here.
    // Events call this only after the existing all-rank event fence; no allocator here.
    void admit_ready(const Ready& s) {
        fence();
        require(resident()<RESIDENT && s.key.epoch==epoch_ && s.key.slot<MAX_SESSIONS, "admission limit/key");
        const unsigned q=s.key.slot;
        require(!live_[q] && generations_[q]!=UINT32_MAX && s.key.generation==generations_[q]+1, "stale/reused slot");
        require(!have_ordinal_ || s.ordinal>last_ordinal_, "ordinal must increase");
        require(request_k(int(s.k),ceiling_)==s.k, "invalid selected k");
        require(s.prompt_size && s.history.size()>s.prompt_size && s.history.size()-s.prompt_size<s.max_output && !s.terminal, "not decode-ready");
        for (int id:s.history) require(id>=0, "bad history token");
        require(s.main_end<=s.history.size() && s.draft_end<=s.history.size(), "poisoned valid ends");
        require(s.main_end>=s.history.size()-1, "unprepared target prefix before verify");
        require(s.run.table_version && s.run.pages>=pages(s.prompt_size,s.max_output,s.k,pool_) && uint64_t(s.run.first_page)+s.run.pages<=pool_/PAGE, "bad full run reservation");
        for (const auto& x:live_) if (x)
            require(uint64_t(s.run.first_page)+s.run.pages<=x->run.first_page || uint64_t(x->run.first_page)+x->run.pages<=s.run.first_page, "overlapping runs");
        live_[q]=s; generations_[q]=s.key.generation; last_ordinal_=s.ordinal; have_ordinal_=true; ++revision_;
    }
    Run retire(Key key) {
        fence();
        require(key.slot<MAX_SESSIONS && live_[key.slot] && live_[key.slot]->key==key, "stale retire");
        Run run=live_[key.slot]->run; live_[key.slot].reset(); ++revision_; return run;
    }
    Plan next() {
        fence();
        std::vector<Runnable> ready;
        for (unsigned q=0; q<MAX_SESSIONS; ++q) if (live_[q] && !live_[q]->terminal)
            ready.push_back({live_[q]->ordinal,q,live_[q]->k});
        auto selected=select(ready,cursor_);
        Plan p{revision_,selected,{},resident()==1};
        unsigned row=0;
        for (unsigned q:selected.slots) {
            const auto& x=*live_[q]; unsigned base=unsigned(x.history.size()-1), n=x.k+1;
            require(uint64_t(base)+n<=uint64_t(x.run.pages)*PAGE, "verify outside granted run");
            p.spans.push_back({x.key,x.run,row,n,base,base+n,n==1?1u:n<=3?3u:5u}); row+=n;
        }
        busy_=true; pending_=p; return p;
    }
    // Independent per-session numerical commits. Validate the WHOLE result before
    // publishing any session state. No transport or per-round allocator callback.
    std::vector<CommitResult> complete(uint64_t revision, const std::vector<Verify>& results) {
        try {
            require(!failed_ && busy_ && pending_ && revision==revision_, "stale round completion");
            require(results.size()==pending_->spans.size(), "partial completion");
            auto after=live_; std::vector<CommitResult> commits;
            for (size_t j=0; j<results.size(); ++j) {
                const auto& v=results[j]; const auto& span=pending_->spans[j];
                require(v.key==span.key, "completion key/order mismatch");
                auto& x=*after[v.key.slot];
                require(v.draft.size()==x.k && v.target.size()==x.k+1, "bad ragged result width");
                unsigned a=0; while (a<x.k && v.draft[a]==v.target[a]) ++a;
                require(v.emitted<=a+1 && (v.finished || v.emitted==a+1), "bad actual emission prefix");
                require(x.history.size()-x.prompt_size+v.emitted<=x.max_output, "emission exceeds request");
                for (int id:v.target) require(id>=0, "bad token id");
                for (int id:v.draft) require(id>=0, "bad draft id");
                x.history.insert(x.history.end(),v.target.begin(),v.target.begin()+v.emitted);
                x.main_end=uint32_t(std::min<size_t>(span.base+a+1,x.history.size()));
                // Verify does not materialize any new first-form draft KV. Chain
                // suffixes are provisional and MUST NOT advance this valid end.
                x.draft_end=uint32_t(std::min<size_t>(x.draft_end,x.history.size()));
                x.terminal=v.finished || x.history.size()-x.prompt_size==x.max_output;
                ++x.rounds;
                commits.push_back({x.key,a,v.emitted,x.main_end,x.draft_end,x.terminal});
            }
            live_=std::move(after); cursor_=pending_->selection.first; busy_=false; pending_.reset(); ++revision_;
            return commits;
        } catch (...) { failed_=true; throw; } // partial/stale result kills epoch
    }
};

// Shapes only: current native Step has one cls, so mixed-class verify must not
// be sent through that adapter until per-span class binding is implemented.
struct Wave { bool chain; unsigned depth; std::vector<std::pair<unsigned,unsigned>> rows; };
inline std::vector<Wave> draft_waves(const Plan& p, const std::vector<CommitResult>& commits) {
    require(p.spans.size()==commits.size(), "partial wave input");
    for (size_t i=0; i<commits.size(); ++i)
        require(commits[i].key==p.spans[i].key && commits[i].accepted<p.spans[i].rows, "bad wave commit");
    std::vector<Wave> out;
    for (unsigned cls:{1u,3u,5u}) {
        Wave w{false,0,{}};
        for (size_t i=0; i<commits.size(); ++i) {
            if (commits[i].terminal) continue;
            const unsigned n=commits[i].accepted+1;
            if ((n==1?1u:n<=3?3u:5u)==cls) w.rows.push_back({commits[i].key.slot,n});
        }
        if (!w.rows.empty()) out.push_back(w);
    }
    for (unsigned depth=1; depth<7; ++depth) {
        Wave w{true,depth,{}};
        for (size_t i=0; i<p.spans.size(); ++i)
            if (!commits[i].terminal && depth<p.spans[i].rows-1) w.rows.push_back({p.spans[i].key.slot,1});
        if (!w.rows.empty()) out.push_back(w);
    }
    return out;
}

struct History {
    Key key; Run run; uint64_t last_request, layout_cookie;
    std::vector<int> tokens; uint32_t main_end, draft_end;
    bool busy=false; std::string hint;
};
// Match first, then test busy/fit: a second request for the best busy history goes
// COLD, not into a shorter free history. Hint is tertiary after length+recency.
inline std::optional<History> reuse(const std::vector<History>& all, const std::vector<int>& prompt,
    uint64_t epoch, uint64_t layout, uint32_t required_pages, const std::string& hint="") {
    require(required_pages>0, "empty reuse reservation");
    const History* best=nullptr;
    auto score=[&](const History& h) { return std::make_tuple(h.tokens.size(),h.last_request,!hint.empty() && h.hint==hint); };
    for (const auto& h:all) {
        if (h.key.epoch!=epoch || !h.key.generation || h.key.slot>=MAX_SESSIONS || h.layout_cookie!=layout ||
            h.tokens.empty() || h.tokens.size()>=prompt.size() || !h.run.table_version ||
            h.main_end>h.tokens.size() || h.draft_end>h.tokens.size() || h.tokens.size()>uint64_t(h.run.pages)*PAGE) continue;
        if (!std::equal(h.tokens.begin(),h.tokens.end(),prompt.begin())) continue;
        if (!best || score(h)>score(*best) || (score(h)==score(*best) && h.key.slot<best->key.slot)) best=&h;
    }
    if (!best || best->busy || required_pages>best->run.pages) return {};
    return *best; // caller owns event-fenced lifecycle/epoch registration, never copies KV
}
inline std::vector<Key> eviction_order(std::vector<History> all) {
    std::sort(all.begin(),all.end(),[](const auto& a,const auto& b) {
        return std::tie(a.last_request,a.key.slot)<std::tie(b.last_request,b.key.slot);
    });
    std::vector<Key> out; for (const auto& h:all) if (!h.busy) out.push_back(h.key); return out;
}
} // namespace g6s2::v1
