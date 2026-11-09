// Versioned big-endian event metadata, and the head's request payloads; control frames never contain model tensors.
// The codec is Rust (engine/scheduler-rs src/control/wire.rs); these are facades with the same API.
#pragma once
#include "event_gate.h"
#include <string>
namespace g6s2 {
enum class Tag : uint64_t { Notice=1, Ack=2, Commit=3, Request=4, Quit=5, Sync=6, Go=7 };
struct Encoder {
    std::vector<uint8_t> b;
    explicit Encoder(Tag t) { g6ctl_put_header(uint32_t(t), ctl::grow_vector, &b); }   // numeric base G6S2E0 + CONTROL_VERSION
    void u64(uint64_t x) { g6ctl_put_u64(x, ctl::grow_vector, &b); }
    void notice(const Notice& n) { g6ctl_put_notice(sch::c<g6ctl_notice>(&n), ctl::grow_vector, &b); }
    void key(Key k) { g6ctl_put_key(sch::c<g6sch_key>(&k), ctl::grow_vector, &b); }
    void steering(const g6rt::steering::Identity& s) { g6ctl_put_steering(sch::c<g6sch_identity>(&s), ctl::grow_vector, &b); }
};
struct Decoder {
    const std::vector<uint8_t>& b; size_t i = 0; Tag tag;
    explicit Decoder(const std::vector<uint8_t>& bytes) : b(bytes) {
        uint32_t t = 0; ctl::check(g6ctl_get_header(b.data(), b.size(), &i, &t)); tag = Tag(t);
    }
    uint64_t u64() { uint64_t x = 0; ctl::check(g6ctl_get_u64(b.data(), b.size(), &i, &x)); return x; }
    uint32_t u32() { uint32_t x = 0; ctl::check(g6ctl_get_u32(b.data(), b.size(), &i, &x)); return x; }
    Notice notice() { Notice n; ctl::check(g6ctl_get_notice(b.data(), b.size(), &i, sch::m<g6ctl_notice>(&n))); return n; }
    Key key() { Key k; ctl::check(g6ctl_get_key(b.data(), b.size(), &i, sch::m<g6sch_key>(&k))); return k; }
    g6rt::steering::Identity steering() {
        g6rt::steering::Identity s; ctl::check(g6ctl_get_steering(b.data(), b.size(), &i, sch::m<g6sch_identity>(&s))); return s;
    }
    void end() { ctl::check(g6ctl_get_end(b.data(), b.size(), i)); }
};
namespace ctl {
inline std::vector<uint8_t> encode(const g6ctl_frame& f) { std::vector<uint8_t> b; check(g6ctl_encode(&f, grow_vector, &b)); return b; }
inline g6ctl_frame decode(const std::vector<uint8_t>& b, Tag t) { g6ctl_frame f{}; check(g6ctl_decode(b.data(), b.size(), uint32_t(t), &f)); return f; }
inline g6ctl_frame frame(Tag t) { g6ctl_frame f{}; f.tag = uint32_t(t); return f; }
inline Notice notice(const g6ctl_notice& n) { Notice x; std::memcpy(static_cast<void*>(&x), &n, sizeof x); return x; }
} // namespace ctl
inline std::vector<uint8_t> encode(const Notice& n){auto f=ctl::frame(Tag::Notice);std::memcpy(&f.notice,&n,sizeof n);return ctl::encode(f);}
inline std::vector<uint8_t> encode(const Ack& a){auto f=ctl::frame(Tag::Ack);std::memcpy(&f.ack,&a,sizeof a);return ctl::encode(f);}
struct EventCommit { Notice notice; Commit commit; };
inline void match(const EventCommit& e){const auto c=ctl::to_ctl(e.commit);ctl::check(g6ctl_match(sch::c<g6ctl_notice>(&e.notice),&c));}
inline std::vector<uint8_t> encode(const EventCommit& x){
    auto f=ctl::frame(Tag::Commit);std::memcpy(&f.notice,&x.notice,sizeof x.notice);f.commit=ctl::to_ctl(x.commit);return ctl::encode(f);
}
inline Notice decode_notice(const std::vector<uint8_t>& b){return ctl::notice(ctl::decode(b,Tag::Notice).notice);}
inline Ack decode_ack(const std::vector<uint8_t>& b){const auto f=ctl::decode(b,Tag::Ack);Ack a;std::memcpy(static_cast<void*>(&a),&f.ack,sizeof a);return a;}
inline EventCommit decode_commit(const std::vector<uint8_t>& b){const auto f=ctl::decode(b,Tag::Commit);return {ctl::notice(f.notice),ctl::from_ctl(f.commit)};}
// The head sends a request's bytes (the shim's req.<id>) to every worker just before each ADMIT notice for it, on
// the same link, so the workers need no copy of the head's files.
struct RequestBytes { uint64_t epoch=0, request=0; std::string bytes; };
inline std::vector<uint8_t> encode(const RequestBytes& r){
    auto f=ctl::frame(Tag::Request);f.epoch=r.epoch;f.request=r.request;f.bytes=ctl::bytes(r.bytes);f.n=r.bytes.size();return ctl::encode(f);
}
inline RequestBytes decode_request(const std::vector<uint8_t>& b){
    const auto f=ctl::decode(b,Tag::Request);RequestBytes r;r.epoch=f.epoch;r.request=f.request;
    r.bytes.assign(reinterpret_cast<const char*>(f.bytes),f.n);return r;
}
// The head's orderly quit (its DIR/quit), passed on so every worker quits the same way
inline std::vector<uint8_t> encode_quit(uint64_t epoch){auto f=ctl::frame(Tag::Quit);f.epoch=epoch;return ctl::encode(f);}
inline uint64_t decode_quit(const std::vector<uint8_t>& b){return ctl::decode(b,Tag::Quit).epoch;}
// A barrier between rounds (EventEndpoint::sync): each worker sends Sync with the barrier's number, the head answers Go
// once every worker has. No tensors, no event: it only lines the ranks up in time.
struct SyncFrame { uint64_t epoch = 0; unsigned rank = 0; uint64_t seq = 0; };
inline std::vector<uint8_t> encode(const SyncFrame& s){auto f=ctl::frame(Tag::Sync);f.epoch=s.epoch;f.rank=s.rank;f.seq=s.seq;return ctl::encode(f);}
inline SyncFrame decode_sync(const std::vector<uint8_t>& b){const auto f=ctl::decode(b,Tag::Sync);return {f.epoch,f.rank,f.seq};}
inline std::vector<uint8_t> encode_go(uint64_t epoch,uint64_t seq){auto f=ctl::frame(Tag::Go);f.epoch=epoch;f.seq=seq;return ctl::encode(f);}
inline std::pair<uint64_t,uint64_t> decode_go(const std::vector<uint8_t>& b){const auto f=ctl::decode(b,Tag::Go);return {f.epoch,f.seq};}
} // namespace g6s2
