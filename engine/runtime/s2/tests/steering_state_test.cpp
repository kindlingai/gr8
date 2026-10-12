#include "s2/control_protocol.h"
#include "s2/pool.h"
#include <cstdio>

using namespace g6s2;
namespace st = g6rt::steering;
static int fails = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x); ++fails; } } while (0)

static st::BootConfig boot() {
    st::BootConfig b; b.loaded = true; b.profile_name = "example"; b.scope = st::Scope::Full; b.active_layers=2;
    for (unsigned i=0;i<32;++i) { b.profile_digest[i]=uint8_t(i); b.schedule_digest[i]=uint8_t(63-i); }
    return b;
}

static g6rt::PrefixReuse finished(const std::vector<int>& p) {
    g6rt::PrefixReuse r; r.prefill_rows(0,int(p.size())-1,true); r.main_rows(int(p.size())-1,2);
    r.draft_rows(int(p.size())-1,2); r.finish(p,{7},1,true); return r;
}

int main() {
    auto b = boot();
    st::RequestFields none, named; named.has_profile=true; named.profile=b.profile_name;
    st::RequestFields zero; zero.has_scale=true; zero.scale=-0.0;
    const auto off=b.resolve(none), on=b.resolve(named);
    CHECK(!off.enabled && on.enabled && st::bits_float(on.scale_bits)==1.0f && !b.resolve(zero).enabled);
    bool threw=false; try { st::RequestFields bad; bad.has_scale=true; bad.scale=1; (void)b.resolve(bad); } catch(...) { threw=true; } CHECK(threw);
    st::RequestFields named2=named; named2.has_scale=true; named2.scale=2.0;
    const auto request_two=b.resolve(named2);
    b.forced=true; b.forced_scale_bits=st::canonical_scale_bits(2.0);
    CHECK(st::bits_float(b.resolve(none).scale_bits)==2.0f && b.resolve(none)==request_two);
    threw=false; try { (void)b.resolve(named); } catch(...) { threw=true; } CHECK(threw);

    Config cfg; cfg.epoch=9; cfg.pool_rows=4096; cfg.max_rows=8; cfg.max_sessions=2; cfg.mtp=2; cfg.prefill_rows=16;
    Scheduler plain(cfg), steered(cfg);
    Grant gp{plain.next_key(),Request{1,8,4,off},Run{0,1,1},0};
    Commit cp{9,1,0,plain.digest(),{},gp}; plain.apply(cp);
    Grant gs{steered.next_key(),Request{1,8,4,on},Run{0,1,1},0};
    Commit cs{9,1,0,steered.digest(),{},gs}; steered.apply(cs);
    CHECK(plain.digest()!=steered.digest());

    Notice n{9,1,0,1,77,EventKind::Admit}; EventCommit ec{n,cs};
    auto bytes=encode(ec); const auto round=decode_commit(bytes);
    CHECK(round.commit.admit.has_value());
    CHECK(round.commit.admit->req.steering==on);
    CHECK(round.commit.admit->req.steering.scale_bits==st::canonical_scale_bits(1.0));
    auto old=bytes; old[7]=0x31; threw=false; try { (void)decode_commit(old); } catch(...) { threw=true; } CHECK(threw);

    g6rt::PrefixReuse rules; RunPool pool(64,64,2048); pool.keep(rules);
    std::vector<int> prompt(100); for(int i=0;i<100;++i)prompt[i]=i;
    auto run=pool.reserve(200); pool.retain(run,finished(prompt),st::cache_namespace(on));
    auto next=prompt; next.push_back(7); next.push_back(8);
    auto h=pool.match(next,200,int(next.size())-1,st::cache_namespace(off)); CHECK(h.skip==0 && std::string(h.why)=="namespace");
    auto different=on; different.profile_digest[0]^=1;
    h=pool.match(next,200,int(next.size())-1,st::cache_namespace(different)); CHECK(h.skip==0 && std::string(h.why)=="namespace");
    different=on; different.scope=st::Scope::Shared;
    h=pool.match(next,200,int(next.size())-1,st::cache_namespace(different)); CHECK(h.skip==0 && std::string(h.why)=="namespace");
    different=on; different.schedule_digest[0]^=1;
    h=pool.match(next,200,int(next.size())-1,st::cache_namespace(different)); CHECK(h.skip==0 && std::string(h.why)=="namespace");
    different=on; different.scale_bits=st::canonical_scale_bits(2.0);
    h=pool.match(next,200,int(next.size())-1,st::cache_namespace(different)); CHECK(h.skip==0 && std::string(h.why)=="namespace");
    h=pool.match(next,200,int(next.size())-1,st::cache_namespace(on)); CHECK(h.skip>0 && std::string(h.why)=="hit");

    // A retired slot is cleared before its generation is reused; the next admission owns its complete identity.
    Scheduler reused(cfg);
    Grant first{reused.next_key(),Request{1,1,1,on},Run{0,1,1},0};
    reused.apply(Commit{9,1,0,reused.digest(),{},first});
    auto work=reused.next(); CHECK(work.kind==Kind::PromptTail);
    reused.complete_tail(Completion{first.key,1,true});
    Grant second{reused.next_key(),Request{2,1,1,off},Run{0,1,2},0};
    reused.apply(Commit{9,2,reused.boundary(),reused.digest(),{},second});
    CHECK(reused.slots()[second.key.slot].key==second.key);
    CHECK(!reused.slots()[second.key.slot].req.steering.enabled);

    std::printf(fails ? "steering_state_test: %d FAIL\n" : "steering_state_test: PASS\n",fails);
    return fails!=0;
}
