// Host control transport primitive. Owns an ALREADY CONNECTED stream socket
// (TCP in a native adapter, socketpair in CPU tests). Only event/control threads use it.
// Persistent fd; no per-request or per-round connect. No CUDA dependency.
// The link is Rust (engine/scheduler-rs src/control/io.rs): ControlIO is a facade that owns its handle.
#pragma once
#include "control_rs.h"
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <memory>
#include "posix_socket.h"
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/time.h>
#include <unistd.h>

namespace g6s2 {
namespace ctl {
// the Rust declares the socket calls' constants and layouts itself: they must be this system's
inline void check_sys() {
    static const bool ok = [] {
#ifdef TCP_USER_TIMEOUT
        constexpr int64_t user_timeout = TCP_USER_TIMEOUT;
#else
        constexpr int64_t user_timeout = -1;
#endif
#ifdef SOCK_CLOEXEC
        constexpr int64_t sock_cloexec = SOCK_CLOEXEC, sock_nonblock = SOCK_NONBLOCK;
#else
        constexpr int64_t sock_cloexec = 0, sock_nonblock = 0;
#endif
#ifdef SO_NOSIGPIPE
        constexpr int64_t nosigpipe = SO_NOSIGPIPE;
#else
        constexpr int64_t nosigpipe = -1;
#endif
        const int64_t want[] = {
            AF_INET, SOCK_STREAM, SOL_SOCKET, SO_REUSEADDR, SO_ERROR, SO_RCVTIMEO, SO_SNDTIMEO, SO_KEEPALIVE,
            IPPROTO_TCP, TCP_NODELAY, TCP_KEEPALIVE_IDLE, TCP_KEEPINTVL, TCP_KEEPCNT, user_timeout, SEND_FLAGS,
            sock_cloexec, sock_nonblock, nosigpipe, O_NONBLOCK, F_GETFL, F_SETFL, F_SETFD, FD_CLOEXEC, EINTR, EAGAIN,
            EWOULDBLOCK, EINPROGRESS, ECONNREFUSED, POLLIN, POLLOUT, POLLNVAL, SHUT_RDWR, sizeof(sockaddr_in),
            offsetof(sockaddr_in, sin_family), offsetof(sockaddr_in, sin_port), offsetof(sockaddr_in, sin_addr),
            sizeof(timeval), offsetof(timeval, tv_usec), sizeof(pollfd)};
        static_assert(sizeof want / sizeof want[0] == G6CTL_SYS_N, "one value per G6CTL_SYS_*");
        for (uint32_t i = 0; i < G6CTL_SYS_N; ++i) if (g6ctl_sys(i) != want[i]) return false;
        return true;
    }();
    require(ok, "scheduler-rs: socket constants differ from the system headers");
}
} // namespace ctl
class ControlIO {
    struct Free { void operator()(g6ctl_io* p) const { g6ctl_io_free(p); } };
    std::unique_ptr<g6ctl_io, Free> h_;
public:
    struct Adopt {};   // a link the Rust bootstrap made
    explicit ControlIO(int fd) : h_((ctl::check_sys(), g6ctl_io_new(fd))) { require(bool(h_), "bad control fd"); }
    ControlIO(Adopt, g6ctl_io* io) : h_(io) { require(bool(h_), "bad control fd"); }
    const g6ctl_io* handle() const { return h_.get(); }
    uint64_t sent() const { uint64_t s, r; g6ctl_io_counts(h_.get(), &s, &r); return s; }
    uint64_t received() const { uint64_t s, r; g6ctl_io_counts(h_.get(), &s, &r); return r; }
    void shutdown() { g6ctl_io_shutdown(h_.get()); } // wake both control directions on epoch failure
    // Adapter configures runtime timeout on the socket. Failure is epoch-fatal;
    // an adapter must call EventGate::fail() to wake a boundary waiter.
    void timeout_seconds(unsigned seconds) { ctl::check(g6ctl_io_timeout(h_.get(), seconds)); }
    // TCP peer liveness for the idle event wait: keepalive probes after idle_s of silence, every intvl_s, cnt times,
    // and TCP_USER_TIMEOUT for unacknowledged sends. A dead peer then fails recv/send. TCP sockets only.
    void keepalive(int idle_s, int intvl_s, int cnt) { ctl::check(g6ctl_io_keepalive(h_.get(), idle_s, intvl_s, cnt)); }
    void nodelay() { ctl::check(g6ctl_io_nodelay(h_.get())); }   // TCP sockets only
    void send(const std::vector<uint8_t>& body) { ctl::check(g6ctl_io_send(h_.get(), body.data(), body.size())); }
    // idle = true: wait for the next frame header with no timeout (event readers only)
    std::vector<uint8_t> receive(bool idle = false) {
        std::vector<uint8_t> b;
        ctl::check(g6ctl_io_receive(h_.get(), idle, ctl::grow_vector, &b));
        return b;
    }
};
} // namespace g6s2
