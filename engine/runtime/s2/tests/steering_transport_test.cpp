// Synthetic loaded profiles exercise the inactive request/row transport. No CUDA or model files.
#include "s2/control_protocol.h"
#include "s2/dispatch.h"
#include "s2/ingress.h"
#include <cstdio>
#include <iostream>
#include <iterator>

using namespace g6s2;
namespace st = g6rt::steering;
static int fails = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x); ++fails; } } while (0)
template<class F> static void rejects(F f) { bool threw=false; try { f(); } catch (const std::exception&) { threw=true; } CHECK(threw); }
static st::BootConfig boot() {
    st::BootConfig b; b.loaded=true; b.profile_name="example"; b.active_layers=2;
    for (unsigned i=0;i<32;++i) { b.profile_digest[i]=uint8_t(i); b.schedule_digest[i]=uint8_t(63-i); }
    return b;
}
static const std::string header="n_gen 20 temp 0 seed 1 min 0 stop - top_p 1";
static g6rt::ServeReq request(const std::string& suffix, const st::BootConfig& b={}) {
    return parse_request(header+suffix+"\n1,2,3,4,5,6,7,8\n",b);
}
static void parsing() {
    const auto b=boot();
    CHECK(!request("").steering.enabled);
    rejects([&]{request(" steer example 00000000");}); // Even zero must name a loaded profile.
    for (const char* tail : {"steer", "steer example", "steer other 3f800000", "steer _bad 3f800000",
            "steer example 7f800000", "steer example 7fc00000", "steer example bf800000",
            "steer example 40800001", "steer example 3F800000", "steer example 3f80000",
            "steer example 3f800000 junk", "steer example 3f800000 steer example 3f800000"})
        rejects([&]{request(std::string(" ")+tail,b);});
    CHECK(!request(" steer example 80000000",b).steering.enabled);
    CHECK(request(" steer example 3f8ccccd",b).steering.scale_bits==0x3f8ccccd);
    CHECK(request(" lp 2 plp 1 sys 3 steer example 3f8ccccd",b).lp==2);
    CHECK(request(" ins 2:-1:9,10 steer example 3f8ccccd",b).ins.tokens.size()==2);
    auto forced=b; forced.forced=true; forced.forced_scale_bits=0x40000000;
    CHECK(request("",forced).steering.scale_bits==0x40000000);
    CHECK(request(" steer example 40000000",forced).steering==request("",forced).steering);
    rejects([&]{request(" steer example 3f800000",forced);});
    rejects([&]{request(" steer example 00000000",forced);});
    Config c; CHECK(capability(c,4)=="G6S2 3 1 4 614400 2 4 24 20\n");
    c.steering=b; CHECK(capability(c,4).find("G6S2 4 ")==0);
    for(uint32_t bits : {0u,0x80000000u,0x7fc00000u,0x7f800000u,0x40800001u}) {
        auto bad=c; bad.steering.forced=true; bad.steering.forced_scale_bits=bits;
        rejects([&]{capability(bad,4);});
    }
    auto bad=c; bad.steering.active_layers=0; rejects([&]{capability(bad,4);});
}
static void codec() {
    const auto b=boot();
    for (uint32_t bits : {0x00000001u,0x3f800000u,0x3f8ccccdu,0x40800000u}) {
        EventCommit e; e.notice={1,1,0,7,77,EventKind::Admit};
        e.commit={1,1,0,99,{},Grant{{1,0,1},{7,8,20,b.selected(bits)},{0,1,1},0}};
        const auto bytes=encode(e);
        const auto d=decode_commit(bytes);
        CHECK(d.commit.admit->req.steering==e.commit.admit->req.steering);
        CHECK(d.commit.admit->req.steering.scale_bits==bits);
        for(size_t n=0;n<bytes.size();++n) { auto cut=bytes; cut.resize(n); rejects([&]{decode_commit(cut);}); }
        auto trailing=bytes; trailing.push_back(0); rejects([&]{decode_commit(trailing);});
        auto old=bytes; old[7]='4'; rejects([&]{decode_commit(old);});
    }
    for (uint64_t flag : {2ull,~0ull}) {
        Encoder e(Tag::Commit); e.u64(flag); Decoder d(e.b); rejects([&]{d.steering();});
    }
    for (uint64_t scope : {0ull,3ull,~0ull}) {
        Encoder e(Tag::Commit); e.u64(1); for(int i=0;i<4;++i)e.u64(0); e.u64(scope);
        Decoder d(e.b); rejects([&]{d.steering();});
    }
    for (uint64_t scale : {0ull,0x80000000ull,0x7fc00000ull,0x7f800000ull,0xbf800000ull,0x40800001ull,0x100000000ull}) {
        Encoder e(Tag::Commit); e.u64(1); for(int i=0;i<4;++i)e.u64(0); e.u64(2);
        for(int i=0;i<4;++i)e.u64(0);
        e.u64(scale); Decoder d(e.b); rejects([&]{d.steering();});
    }
}
// Match the fields native_inputs consumes, including padding larger than the live row count.
struct Input {
    uint64_t step; int M,context,tok[24],pos[24],tgt[24]; float steering_scale[24];
    struct Sample { float inv_temp; uint32_t seed, unused; int pos,nodraft; float top_p; int topk; } srow[24];
};
struct Layout { int ns,cls,sess[8],row0[8],rows[8],ctx[8],src0[8]; };
static s2::Step rows(s2::Step::Kind kind) {
    s2::Step s; s.kind=kind; s.sess={0,1}; s.row0={0,2}; s.rows={2,1}; s.ctx={6,1};
    s.r={{0,4,4,5,5},{0,5,5,6,-1},{1,1,0,1,-1}}; return s;
}
static void dispatch() {
    Config c; c.epoch=9; c.max_rows=24; c.steering=boot();
    Sessions sessions(c,"",false,4);
    auto on=request(" steer example 3f8ccccd",c.steering), off=request("");
    Grant g0{{9,0,1},Sessions::request(1,on),{0,1,1},0};
    Grant g1{{9,1,1},Sessions::request(2,off),{1,1,2},0};
    sessions.bind(g0,on); sessions.bind(g1,off);
    auto wrong=g0; wrong.key.slot=2; wrong.req.steering={};
    rejects([&]{sessions.bind(wrong,on);});
    unsigned calls=0,prefills=0; std::string main_key;
    DispatchHooks h; h.bucket=[](bool,int){return 0;}; h.mode=[](int,int){};
    h.launch=[&](const Launch& l) {
        Input in{}; Layout layout{};
        for(float& scale:in.steering_scale)scale=9.f; // Prove the previous step cannot leak into padding.
        native_inputs(l,in,layout);
        CHECK(in.M==3 && layout.ns==2 && in.srow[1].nodraft && in.srow[2].nodraft);
        const bool target=l.step.kind==s2::Step::MAIN;
        for(int r=0;r<24;++r) {
            const auto want=target && r<2 && calls<3 ? 0x3f8ccccdu : 0u;
            CHECK(st::float_bits(in.steering_scale[r])==want);
        }
        if(target) { if(main_key.empty())main_key=l.graph_key; else CHECK(l.graph_key==main_key); }
        ++calls; return std::vector<int>(3,1);
    };
    h.prefill=[&](const Slot&,const Payload& p,uint32_t,uint32_t){CHECK(p.steering==on.steering);++prefills;};
    Dispatch d(sessions,24,h); d.track_rows();
    d.begin(0,0,st::cache_namespace(on.steering)); d.begin(1,0);
    Slot slot; slot.key=g0.key; slot.run=g0.run;
    d.prefill(slot,sessions.payload(on),0,4); CHECK(prefills==1);
    d.run(rows(s2::Step::MAIN)); d.run(rows(s2::Step::DRAFT)); d.run(rows(s2::Step::CHAIN));
    CHECK(d.rows(0).cache_namespace==st::cache_namespace(on.steering));
    // Cancel and reuse the same slot as stock. Its scale and row/cache state must be reset.
    sessions.retire(slot); d.release(0); CHECK(d.rows(0).cache_namespace==g6rt::CacheNamespace{});
    g0.key.generation++; g0.req=Sessions::request(3,off); sessions.bind(g0,off); d.begin(0,4);
    d.run(rows(s2::Step::MAIN)); CHECK(calls==4);
}
static void payload_agreement() {
    Config c; unsigned binds=0;
    DriverHooks h; h.bind=[&](const Grant&){++binds;}; h.retire=[](const Slot&){};
    h.prefill=[](const Slot&,const Payload&,uint32_t,uint32_t){};
    h.run=[](const s2::Step& s){return std::vector<int>(s.M(),1);};
    h.emit=[](Key,const int*,int n,int){return Emission{uint32_t(n),false};};
    Driver d(c,h); const auto before=d.state().digest();
    auto r=request(" steer example 3f8ccccd",boot());
    Grant g{d.state().next_key(),Sessions::request(1,r),{0,1,1},0};
    Commit cm{c.epoch,1,0,before,{},g};
    Payload p{r.prompt,4,false,{}, {}};
    rejects([&]{d.apply(cm,&p);}); CHECK(d.state().digest()==before && binds==0);
    p.steering=r.steering; d.apply(cm,&p); CHECK(binds==1);
    const auto old=g.key;
    g.key=d.state().next_key(old); g.req=Sessions::request(2,request(""));
    cm={c.epoch,2,0,d.state().digest(),old,g}; p.steering={};
    d.apply(cm,&p); CHECK(!d.state().slots()[g.key.slot].req.steering.enabled && binds==2);
}
int main(int argc, char** argv) {
    if(argc==2 && std::string(argv[1])=="--request") {
        // The Python fixture supplies the real shim's bytes; round-trip through the replicated grant.
        const std::string bytes{std::istreambuf_iterator<char>(std::cin),{}};
        const auto r=parse_request(bytes,boot());
        EventCommit e; e.notice={1,1,0,7,77,EventKind::Admit};
        e.commit={1,1,0,99,{},Grant{{1,0,1},Sessions::request(7,r),{0,1,1},0}};
        const auto s=decode_commit(encode(e)).commit.admit->req.steering;
        CHECK(s==r.steering);
        std::printf("%08x\n",s.scale_bits); return fails!=0;
    }
    parsing(); codec(); dispatch(); payload_agreement();
    std::printf("steering_transport_test: %s\n",fails?"FAIL":"PASS"); return fails!=0;
}
