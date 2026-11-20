// Native display-memory wire client. No CUDA dependency.
#pragma once
#include "knobs-rs/gr8_knobs.h"
#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <fcntl.h>
#include <poll.h>
#define bind g6dispram_posix_bind_undeclared
#include <sys/socket.h>
#include <sys/un.h>
#undef bind
#include <sys/syscall.h>
#include <unistd.h>

namespace g6dispram {
constexpr size_t GRANULE = 2ull << 20;
inline bool enabled() { return knob::dispram() && !knob::dispram_disable(); }
inline const std::string& socket_path() { return knob::dispram_socket(); }
struct Info { uint64_t base = 0, size = 0, free_bytes = 0, largest = 0; };
struct Lease { int fd = -1; uint64_t base = 0, size = 0; };
struct SpanPlan { size_t total_bytes = 0, carve_bytes = 0, tail_bytes = 0; };
inline size_t lcm_gran(size_t a, size_t b) { return !a || !b ? 0 : a / std::gcd(a,b) * b; }
inline SpanPlan plan_span(size_t bytes, size_t gran, uint64_t avail) {
    if (!bytes || !gran) return {};
    size_t total = ((bytes + gran - 1) / gran) * gran;
    uint64_t usable = (avail / gran) * gran;
    size_t carve = usable >= total ? total : static_cast<size_t>(usable);
    return {total, carve, total - carve};
}
#ifdef __APPLE__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
inline int posix_bind(int fd, const sockaddr* a, socklen_t n) { return int(::syscall(SYS_bind, fd, a, n)); }
#ifdef __APPLE__
#pragma clang diagnostic pop
#endif
inline void set_cloexec(int fd) {
    if (fd >= 0) { int fl = ::fcntl(fd, F_GETFD); if (fl >= 0) (void)::fcntl(fd, F_SETFD, fl | FD_CLOEXEC); }
}
inline bool ws(char c) { return c==' ' || c=='\t' || c=='\n' || c=='\r'; }
// Scan quoted tokens, skipping escaped characters so keys embedded in values cannot match.
inline size_t json_value(const std::string& s, const char* key) {
    if (s.size() > 4096) return std::string::npos;
    size_t begin = s.find_first_not_of(" \t\r\n"), end = s.find_last_not_of(" \t\r\n");
    if (begin == std::string::npos || s[begin]!='{' || s[end]!='}') return std::string::npos;
    for (size_t i=begin+1; i<end; ++i) {
        if (s[i]!='"') continue;
        size_t a=++i; bool escaped=false;
        while (i<end && s[i]!='"') { if (s[i]=='\\') { escaped=true; ++i; } ++i; }
        if (i>=end) return std::string::npos;
        size_t next=i+1; while (next<end && ws(s[next])) ++next;
        if (!escaped && s.compare(a,i-a,key)==0 && next<end && s[next]==':') {
            ++next; while (next<end && ws(s[next])) ++next; return next;
        }
    }
    return std::string::npos;
}
inline bool json_has_key(const std::string& s,const char* key) { return json_value(s,key)!=std::string::npos; }
inline bool json_end(const std::string& s,size_t i) {
    while (i<s.size() && ws(s[i])) ++i;
    return i<s.size() && (s[i]==',' || s[i]=='}');
}
inline bool json_bool(const std::string& s,const char* key,bool* out) {
    size_t i=json_value(s,key); if (i==std::string::npos || !out) return false;
    if (s.compare(i,4,"true")==0 && json_end(s,i+4)) { *out=true; return true; }
    if (s.compare(i,5,"false")==0 && json_end(s,i+5)) { *out=false; return true; }
    return false;
}
inline bool json_u64(const std::string& s,const char* key,uint64_t* out) {
    size_t i=json_value(s,key); if (i==std::string::npos || !out || s[i]<'0' || s[i]>'9') return false;
    uint64_t n=0; size_t a=i;
    while (i<s.size() && s[i]>='0' && s[i]<='9') {
        unsigned d=s[i++]-'0'; if (n>(UINT64_MAX-d)/10) return false; n=n*10+d;
    }
    if ((i-a>1 && s[a]=='0') || !json_end(s,i)) return false;
    *out=n; return true;
}
class Client {
    int fd_=-1;
    bool broken_=false;
    bool exchange(const std::string& request,std::string& reply,int& got,int timeout) {
        got=-1; if (fd_<0 || broken_) return false;
        int flags=0;
#ifdef MSG_NOSIGNAL
        flags |= MSG_NOSIGNAL;
#endif
        if (::send(fd_,request.data(),request.size(),flags)!=static_cast<ssize_t>(request.size())) { broken_=true; return false; }
        pollfd p{fd_,POLLIN,0};
        if (::poll(&p,1,timeout)<=0) { broken_=true; return false; }
        char buf[4096]; alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int)*4)]{};
        iovec io{buf,sizeof(buf)}; msghdr msg{};
        msg.msg_iov=&io; msg.msg_iovlen=1; msg.msg_control=control; msg.msg_controllen=sizeof(control);
        int rflags=0;
#ifdef MSG_CMSG_CLOEXEC
        rflags |= MSG_CMSG_CLOEXEC;
#endif
        ssize_t n=::recvmsg(fd_,&msg,rflags);
        for (cmsghdr* c=CMSG_FIRSTHDR(&msg); c; c=CMSG_NXTHDR(&msg,c)) {
            if (c->cmsg_level!=SOL_SOCKET || c->cmsg_type!=SCM_RIGHTS || c->cmsg_len<CMSG_LEN(0)) continue;
            size_t count=(c->cmsg_len-CMSG_LEN(0))/sizeof(int);
            for (size_t i=0;i<count;++i) {
                int f=-1; std::memcpy(&f,CMSG_DATA(c)+i*sizeof(int),sizeof(f)); set_cloexec(f);
                if (got<0) got=f; else ::close(f);
            }
        }
        if (n<=0 || (msg.msg_flags & (MSG_CTRUNC|MSG_TRUNC))) {
            if (got>=0) ::close(got);
            got=-1; broken_=true; return false;
        }
        reply.assign(buf,static_cast<size_t>(n)); return true;
    }
public:
    Client()=default;
    Client(const Client&)=delete;
    Client& operator=(const Client&)=delete;
    ~Client() { close_conn(); }
    void close_conn() { if (fd_>=0) ::close(fd_); fd_=-1; broken_=false; }
    bool is_broken() const { return broken_; }
    bool is_open() const { return fd_>=0; }
    bool open_conn(const std::string& path,int timeout_ms=2000) {
        close_conn();
#ifdef SOCK_SEQPACKET
        sockaddr_un addr{}; addr.sun_family=AF_UNIX;
        if (path.empty() || path.size()>=sizeof(addr.sun_path)) return false;
        std::memcpy(addr.sun_path,path.c_str(),path.size()+1);
        int stype=SOCK_SEQPACKET;
#ifdef SOCK_CLOEXEC
        stype |= SOCK_CLOEXEC;
#endif
        int s=::socket(AF_UNIX,stype,0);
#ifdef SOCK_CLOEXEC
        if (s<0 && errno==EINVAL) s=::socket(AF_UNIX,SOCK_SEQPACKET,0);
#endif
        if (s<0) return false;
        set_cloexec(s);
#ifdef SO_NOSIGPIPE
        int one=1; (void)::setsockopt(s,SOL_SOCKET,SO_NOSIGPIPE,&one,sizeof(one));
#endif
        int fl=::fcntl(s,F_GETFL);
        if (fl<0 || ::fcntl(s,F_SETFL,fl|O_NONBLOCK)<0) { ::close(s); return false; }
        int rc=::connect(s,reinterpret_cast<sockaddr*>(&addr),sizeof(addr));
        if (rc<0 && errno!=EINPROGRESS && errno!=EAGAIN) { ::close(s); return false; }
        pollfd p{s,POLLOUT,0}; int err=0; socklen_t len=sizeof(err);
        if (::poll(&p,1,timeout_ms)<=0 || ::getsockopt(s,SOL_SOCKET,SO_ERROR,&err,&len)<0 || err ||
            ::fcntl(s,F_SETFL,fl)<0) { ::close(s); return false; }
        fd_=s; return true;
#else
        (void)path; (void)timeout_ms; return false;
#endif
    }
    bool query_info(Info* out,int timeout_ms=2000) {
        if (!out) return false;
        std::string reply; int got=-1;
        bool sent=exchange("{\"key\":\"kindlingai_1\",\"op\":\"info\"}",reply,got,timeout_ms);
        if (got>=0) ::close(got);
        Info info;
        if (!sent || json_has_key(reply,"error") || !json_u64(reply,"free",&info.free_bytes) || !json_u64(reply,"size",&info.size)) { broken_=true; return false; }
        json_u64(reply,"base",&info.base); json_u64(reply,"largest",&info.largest); *out=info; return true;
    }
    bool lease(size_t bytes,Lease* out,int timeout_ms=2000) {
        if (fd_<0 || broken_ || !out || bytes<GRANULE || bytes%GRANULE) return false;
        std::string reply; int got=-1; bool ok=false; uint64_t sz=0,base=0;
        if (!exchange("{\"key\":\"kindlingai_1\",\"op\":\"alloc\",\"size\":"+std::to_string(bytes)+"}",reply,got,timeout_ms)) return false;
        if (!json_bool(reply,"ok",&ok)) { if (got>=0) ::close(got); broken_=true; return false; }
        if (!ok) { if (got>=0) ::close(got); return false; }
        if (got<0 || !json_u64(reply,"size",&sz) || sz!=bytes) { if (got>=0) ::close(got); broken_=true; return false; }
        json_u64(reply,"base",&base); *out={got,base,sz}; return true;
    }
};

struct Span { size_t total=0,carve=0,gran=0; };
// The session owns the connection lifetime; a wire failure never closes it ahead of live mappings.
template<class Backend> class Session {
    Backend& backend;
    std::mutex mu;
    Client client;
    bool probed=false,unavailable=false;
    uint64_t rem_bytes=0;
    std::map<uintptr_t,Span> spans;
    std::atomic<size_t> claimed_bytes_total{0},mapped_bytes_total{0},claimed_spans_total{0};
    int fail_after=-1,timeout_ms=2000;
    void reset_connection() { client.close_conn(); probed=false; unavailable=false; rem_bytes=0; }
    void release_locked() {
        if (!spans.empty()) {
            backend.sync();
            for (const auto& e:spans) backend.unmap_span(e.first,e.second);
            spans.clear();
        }
        claimed_bytes_total.store(0,std::memory_order_relaxed);
        mapped_bytes_total.store(0,std::memory_order_relaxed);
        claimed_spans_total.store(0,std::memory_order_relaxed);
        reset_connection();
    }
    [[noreturn]] void fail_locked(const char* detail,size_t min_req) {
        const size_t have=claimed_bytes();
        release_locked();
        if (detail) std::fprintf(stderr,"g6: dispram shortfall: %s (live mapped %zu bytes, required %zu bytes)\n",detail,have,min_req);
        else std::fprintf(stderr,"g6: dispram shortfall: live mapped %zu bytes < required %zu bytes\n",have,min_req);
        std::fflush(stderr); std::_Exit(75);
    }
public:
    using Err=typename Backend::Err;
    explicit Session(Backend& b):backend(b) {}
    ~Session() { release_all(); }
    size_t claimed_bytes() const { return claimed_bytes_total.load(std::memory_order_relaxed); }
    size_t mapped_bytes() const { return mapped_bytes_total.load(std::memory_order_relaxed); }
    size_t claimed_spans() const { return claimed_spans_total.load(std::memory_order_relaxed); }
    // Test hooks are called on the allocating thread, including from backend callbacks under mu.
    bool has_open_conn_for_test() const { return client.is_open(); }
    bool is_broken_for_test() const { return client.is_broken(); }
    void set_fail_vmm_after_spans_for_test(int n) { std::lock_guard<std::mutex> lk(mu); fail_after=n; }
    void set_timeout_ms_for_test(int ms) { std::lock_guard<std::mutex> lk(mu); timeout_ms=ms; }
    Err alloc_kv(void** out_ptr,size_t bytes) {
        if (!out_ptr) return Backend::invalid_err();
        *out_ptr=nullptr;
        if (!bytes) return backend.fallback_alloc(out_ptr,0);
        const auto minimum=knob::dispram_min_bytes();
        const size_t min_req=minimum>0 ? static_cast<size_t>(minimum):0;
        std::unique_lock<std::mutex> lk(mu);
        if (!enabled()) {
            if (!spans.empty()) fail_locked("disabled after live mapping",min_req);
            if (min_req && claimed_bytes()<min_req) fail_locked(nullptr,min_req);
            lk.unlock(); return backend.fallback_alloc(out_ptr,bytes);
        }
        const bool need_carve=min_req>0 && claimed_bytes()<min_req;
        size_t gmin=0;
        if (!backend.granularity(&gmin) || !gmin) {
            if (need_carve) fail_locked(nullptr,min_req);
            unavailable=true; if (spans.empty()) client.close_conn();
            lk.unlock(); return backend.fallback_alloc(out_ptr,bytes);
        }
        const size_t gran=lcm_gran(GRANULE,gmin);
        auto handle_broken=[&]() {
            if (!client.is_broken()) return false;
            if (!spans.empty()) fail_locked("transport error after live mapping",min_req);
            if (need_carve) fail_locked("transport error before minimum claim",min_req);
            client.close_conn(); unavailable=true; return true;
        };
        if (!unavailable && !probed) {
            probed=true; Info info;
            if (!client.open_conn(socket_path(),timeout_ms) || !client.query_info(&info,timeout_ms)) {
                handle_broken();
                if (need_carve) fail_locked(nullptr,min_req);
                if (spans.empty()) client.close_conn();
                unavailable=true;
            } else {
                rem_bytes=info.free_bytes/gran*gran;
                if (rem_bytes<gran) { if (spans.empty()) client.close_conn(); unavailable=true; }
            }
        }
        if (unavailable || rem_bytes<gran) {
            if (need_carve) fail_locked(nullptr,min_req);
            if (spans.empty()) client.close_conn();
            lk.unlock(); return backend.fallback_alloc(out_ptr,bytes);
        }
        SpanPlan plan=plan_span(bytes,gran,rem_bytes);
        auto insufficient=[&] { return need_carve && (plan.carve_bytes<gran ||
            (rem_bytes-plan.carve_bytes<gran && claimed_bytes()+plan.carve_bytes<min_req)); };
        if (insufficient()) fail_locked(nullptr,min_req);
        Lease lease; bool got=client.lease(plan.carve_bytes,&lease,timeout_ms);
        if (handle_broken()) { lk.unlock(); return backend.fallback_alloc(out_ptr,bytes); }
        if (!got && client.is_open() && !client.is_broken()) {
            Info info;
            if (!client.query_info(&info,timeout_ms)) {
                if (handle_broken()) { lk.unlock(); return backend.fallback_alloc(out_ptr,bytes); }
            } else {
                rem_bytes=info.free_bytes/gran*gran; plan=plan_span(bytes,gran,rem_bytes);
                if (insufficient()) fail_locked(nullptr,min_req);
                if (plan.carve_bytes>=gran) {
                    got=client.lease(plan.carve_bytes,&lease,timeout_ms);
                    if (handle_broken()) { lk.unlock(); return backend.fallback_alloc(out_ptr,bytes); }
                }
            }
        }
        if (!got || lease.fd<0) {
            if (need_carve) fail_locked(nullptr,min_req);
            unavailable=true; if (spans.empty()) client.close_conn();
            lk.unlock(); return backend.fallback_alloc(out_ptr,bytes);
        }
        rem_bytes-=plan.carve_bytes;
        uintptr_t va=0;
        const bool mapped=backend.map_span(lease.fd,plan,gran,fail_after,spans.size(),&va); lease.fd=-1;
        if (mapped) {
            spans[va]={plan.total_bytes,plan.carve_bytes,gran};
            claimed_bytes_total.fetch_add(plan.carve_bytes,std::memory_order_relaxed);
            mapped_bytes_total.fetch_add(plan.total_bytes,std::memory_order_relaxed);
            claimed_spans_total.fetch_add(1,std::memory_order_relaxed);
            *out_ptr=reinterpret_cast<void*>(va); return Backend::ok_err();
        }
        unavailable=true;
        if (need_carve) fail_locked(nullptr,min_req);
        if (spans.empty()) client.close_conn();
        lk.unlock(); return backend.fallback_alloc(out_ptr,bytes);
    }
    Err free_kv(void* ptr) {
        std::unique_lock<std::mutex> lk(mu);
        auto it=spans.find(reinterpret_cast<uintptr_t>(ptr));
        if (it==spans.end()) { lk.unlock(); return backend.fallback_free(ptr); }
        const Span sp=it->second;
        backend.sync(); backend.unmap_span(it->first,sp); spans.erase(it);
        claimed_bytes_total.fetch_sub(sp.carve,std::memory_order_relaxed);
        mapped_bytes_total.fetch_sub(sp.total,std::memory_order_relaxed);
        claimed_spans_total.fetch_sub(1,std::memory_order_relaxed);
        if (spans.empty()) reset_connection();
        return Backend::ok_err();
    }
    void release_all() { std::lock_guard<std::mutex> lk(mu); release_locked(); }
};
}
