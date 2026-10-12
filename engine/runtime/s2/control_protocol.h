// Versioned big-endian event metadata, and the head's request payloads; control frames never contain model tensors.
#pragma once
#include "event_gate.h"
#include <string>
namespace g6s2 {
enum class Tag : uint64_t { Notice=1, Ack=2, Commit=3, Request=4, Quit=5, Sync=6, Go=7 };
struct Encoder {
    std::vector<uint8_t> b;
    void u64(uint64_t x){for(int i=7;i>=0;--i)b.push_back(uint8_t(x>>(i*8)));}
    explicit Encoder(Tag t){u64(0x473653324535ull);u64(uint64_t(t));} // G6S2E5
    void notice(const Notice& n){u64(n.epoch);u64(n.event);u64(n.boundary);u64(n.request);u64(n.payload_hash);u64(uint64_t(n.kind));}
    void key(Key k){u64(k.epoch);u64(k.slot);u64(k.generation);}
    void steering(const g6rt::steering::Identity& s){
        u64(s.enabled);if(!s.enabled)return;
        for(unsigned i=0;i<32;i+=8){uint64_t x=0;for(unsigned k=0;k<8;++k)x=x*256+s.profile_digest[i+k];u64(x);}
        u64(uint64_t(s.scope));
        for(unsigned i=0;i<32;i+=8){uint64_t x=0;for(unsigned k=0;k<8;++k)x=x*256+s.schedule_digest[i+k];u64(x);}
        u64(s.scale_bits);
    }
};
struct Decoder {
    const std::vector<uint8_t>& b;size_t i=0;Tag tag;
    uint64_t u64(){require(b.size()-i>=8,"truncated control metadata");uint64_t x=0;for(unsigned n=0;n<8;++n)x=x*256+b[i++];return x;}
    uint32_t u32(){auto x=u64();require(x<=UINT32_MAX,"control integer overflow");return uint32_t(x);}
    explicit Decoder(const std::vector<uint8_t>& bytes):b(bytes){require(u64()==0x473653324535ull,"control version mismatch");auto t=u64();require(t>=1&&t<=7,"unknown control tag");tag=Tag(t);}
    Notice notice(){Notice n;n.epoch=u64();n.event=u64();n.boundary=u64();n.request=u64();n.payload_hash=u64();auto k=u64();require(k==1||k==2,"unknown event kind");n.kind=EventKind(k);return n;}
    Key key(){Key k;k.epoch=u64();k.slot=u32();k.generation=u32();return k;}
    g6rt::steering::Identity steering(){
        g6rt::steering::Identity s;auto on=u64();require(on<=1,"invalid steering enabled bit");s.enabled=on;
        if(!s.enabled)return s;
        for(unsigned i=0;i<32;i+=8){auto x=u64();for(int k=7;k>=0;--k){s.profile_digest[i+k]=uint8_t(x);x>>=8;}}
        auto scope=u64();require(scope==uint64_t(g6rt::steering::Scope::Shared)||scope==uint64_t(g6rt::steering::Scope::Full),"invalid steering scope");s.scope=g6rt::steering::Scope(scope);
        for(unsigned i=0;i<32;i+=8){auto x=u64();for(int k=7;k>=0;--k){s.schedule_digest[i+k]=uint8_t(x);x>>=8;}}
        s.scale_bits=u32();const float scale=g6rt::steering::bits_float(s.scale_bits);
        require(s.scale_bits&&std::isfinite(scale)&&scale>0&&scale<=g6rt::steering::SCALE_MAX,"invalid steering scale bits");return s;
    }
    void end(){require(i==b.size(),"trailing control metadata");}
};
inline std::vector<uint8_t> encode(const Notice& n){Encoder e(Tag::Notice);e.notice(n);return e.b;}
inline std::vector<uint8_t> encode(const Ack& a){Encoder e(Tag::Ack);e.notice(a.notice);e.u64(a.rank);e.u64(a.state_hash);return e.b;}
struct EventCommit { Notice notice; Commit commit; };
inline void match(const EventCommit& e){
    require(e.notice.epoch==e.commit.epoch && e.notice.event==e.commit.event && e.notice.boundary==e.commit.boundary,"commit/notice mismatch");
    if(e.commit.admit) require(e.notice.kind==EventKind::Admit && e.commit.admit->req.id==e.notice.request,"grant does not match announced request");
}
inline std::vector<uint8_t> encode(const EventCommit& x){
    match(x);Encoder e(Tag::Commit);e.notice(x.notice);const auto& c=x.commit;
    e.u64(c.before_hash);e.u64(c.cancel.has_value());if(c.cancel)e.key(*c.cancel);
    e.u64(c.admit.has_value());if(c.admit){const auto&g=*c.admit;e.key(g.key);e.u64(g.req.id);e.u64(g.req.prompt_tokens);e.u64(g.req.max_tokens);e.steering(g.req.steering);e.u64(g.run.first_page);e.u64(g.run.pages);e.u64(g.run.table_version);e.u64(g.run.list_hash);e.u64(g.reuse);e.u64(g.restored);e.u64(g.counted);}
    return e.b;
}
inline Notice decode_notice(const std::vector<uint8_t>& b){Decoder d(b);require(d.tag==Tag::Notice,"expected notice");auto n=d.notice();d.end();return n;}
inline Ack decode_ack(const std::vector<uint8_t>& b){Decoder d(b);require(d.tag==Tag::Ack,"expected ACK");Ack a;a.notice=d.notice();a.rank=d.u32();a.state_hash=d.u64();d.end();return a;}
inline EventCommit decode_commit(const std::vector<uint8_t>& b){
    Decoder d(b);require(d.tag==Tag::Commit,"expected commit");EventCommit e;e.notice=d.notice();auto&c=e.commit;
    c.epoch=e.notice.epoch;c.event=e.notice.event;c.boundary=e.notice.boundary;c.before_hash=d.u64();
    auto has=d.u64();require(has<=1,"invalid optional cancel");if(has)c.cancel=d.key();
    has=d.u64();require(has<=1,"invalid optional grant");if(has){Grant g;g.key=d.key();g.req.id=d.u64();g.req.prompt_tokens=d.u32();g.req.max_tokens=d.u32();g.req.steering=d.steering();g.run.first_page=d.u32();g.run.pages=d.u32();g.run.table_version=d.u32();g.run.list_hash=d.u64();g.reuse=d.u32();g.restored=d.u32();g.counted=d.u32();require(!g.restored||g.reuse,"invalid restored grant");c.admit=g;}
    d.end();return e;
}
// The head sends a request's bytes (the shim's req.<id>) to every worker just before each ADMIT notice for it, on
// the same link, so the workers need no copy of the head's files.
struct RequestBytes { uint64_t epoch=0, request=0; std::string bytes; };
inline std::vector<uint8_t> encode(const RequestBytes& r){
    Encoder e(Tag::Request);e.u64(r.epoch);e.u64(r.request);e.u64(r.bytes.size());e.b.insert(e.b.end(),r.bytes.begin(),r.bytes.end());return e.b;
}
inline RequestBytes decode_request(const std::vector<uint8_t>& b){
    Decoder d(b);require(d.tag==Tag::Request,"expected request");RequestBytes r;r.epoch=d.u64();r.request=d.u64();
    const auto n=d.u64();require(n==b.size()-d.i,"request length mismatch");r.bytes.assign(b.begin()+long(d.i),b.end());d.i=b.size();d.end();return r;
}
// The head's orderly quit (its DIR/quit), passed on so every worker quits the same way
inline std::vector<uint8_t> encode_quit(uint64_t epoch){Encoder e(Tag::Quit);e.u64(epoch);return e.b;}
inline uint64_t decode_quit(const std::vector<uint8_t>& b){Decoder d(b);require(d.tag==Tag::Quit,"expected quit");auto e=d.u64();d.end();return e;}
// A barrier between rounds (EventEndpoint::sync): each worker sends Sync with the barrier's number, the head answers Go
// once every worker has. No tensors, no event: it only lines the ranks up in time.
struct SyncFrame { uint64_t epoch = 0; unsigned rank = 0; uint64_t seq = 0; };
inline std::vector<uint8_t> encode(const SyncFrame& s){Encoder e(Tag::Sync);e.u64(s.epoch);e.u64(s.rank);e.u64(s.seq);return e.b;}
inline SyncFrame decode_sync(const std::vector<uint8_t>& b){Decoder d(b);require(d.tag==Tag::Sync,"expected sync");SyncFrame s;s.epoch=d.u64();s.rank=d.u32();s.seq=d.u64();d.end();return s;}
inline std::vector<uint8_t> encode_go(uint64_t epoch,uint64_t seq){Encoder e(Tag::Go);e.u64(epoch);e.u64(seq);return e.b;}
inline std::pair<uint64_t,uint64_t> decode_go(const std::vector<uint8_t>& b){Decoder d(b);require(d.tag==Tag::Go,"expected go");auto e=d.u64();auto q=d.u64();d.end();return {e,q};}
} // namespace g6s2
