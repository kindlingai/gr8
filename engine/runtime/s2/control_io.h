// Host control transport primitive. Owns an ALREADY CONNECTED stream socket
// (TCP in a native adapter, socketpair in CPU tests). Only event/control threads use it.
// Persistent fd; no per-request ssh or per-round connect. No CUDA dependency.
#pragma once
#include "scheduler.h"
#include <cerrno>
#include <cstdio>
#include <cstring>
#include "posix_socket.h"
#include <netinet/tcp.h>
#include <unistd.h>

namespace g6s2 {
class ControlIO {
    int fd_;
    static constexpr uint32_t MAX_BYTES = 1u << 20; // control/event metadata, never model tensors
    // disconnect logging (root-cause follow-up: r1 died with only 'control disconnected/truncated' across 5 crashes):
    // before the epoch-fatal require, print ONE greppable reason line to stderr (rN.err) so the
    // next FABRIC capture shows WHY the control link dropped: recv/send return code, errno text,
    // bytes still missing, and the frame counters. Diagnostics only: the throw stays identical.
    [[gnu::cold]] void log_disconnect(const char* dir, ssize_t k, size_t remaining, size_t total) const {
        const int e = errno; // captured before anything else can touch it
        std::fprintf(stderr, "g6ctl: control %s failed: ret=%zd errno=%d (%s) remaining=%zu of %zu bytes, frames sent=%llu recv=%llu\n",
                     dir, k, e, std::strerror(e), remaining, total,
                     (unsigned long long)sent, (unsigned long long)received);
    }
    void send_all(const uint8_t* p, size_t n) {
        const size_t total = n;
        while (n) {
            ssize_t k = ::send(fd_, p, n, MSG_NOSIGNAL);
            if (k < 0 && errno == EINTR) continue;
            if (k <= 0) log_disconnect("send", k, n, total);
            require(k > 0, "control send failed"); p += k; n -= size_t(k);
        }
    }
    // idle: the link may stay silent indefinitely before the first byte (a worker's event reader between
    // admit/cancel events, #26), so a receive timeout (EAGAIN) with nothing read yet is retried. Once a frame has
    // started, the timeout applies again. A peer close or shutdown() still ends the wait (recv returns 0).
    void read_all(uint8_t* p, size_t n, bool idle = false) {
        const size_t total = n;
        while (n) {
            ssize_t k = ::recv(fd_, p, n, 0);
            if (k < 0 && errno == EINTR) continue;
            if (k < 0 && idle && n == total && (errno == EAGAIN || errno == EWOULDBLOCK)) continue;
            if (k <= 0) log_disconnect("recv", k, n, total);
            require(k > 0, "control disconnected/truncated"); p += k; n -= size_t(k);
        }
    }
public:
    uint64_t sent = 0, received = 0;
    explicit ControlIO(int fd) : fd_(fd) { require(fd >= 0, "bad control fd"); }
    ~ControlIO() { if (fd_ >= 0) ::close(fd_); }
    ControlIO(const ControlIO&) = delete; ControlIO& operator=(const ControlIO&) = delete;
    void shutdown() { ::shutdown(fd_, SHUT_RDWR); } // wake both control directions on epoch failure
    // Adapter configures runtime timeout on the socket. Failure is epoch-fatal;
    // an adapter must call EventGate::fail() to wake a boundary waiter.
    void timeout_seconds(unsigned seconds) {
        require(seconds > 0, "control timeout must be positive");
        timeval tv{time_t(seconds), 0};
        require(!setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv) &&
                !setsockopt(fd_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv), "control timeout setup failed");
    }
    // TCP peer liveness for the idle event wait (#26): the idle read has no timeout, so a peer that is lost without a
    // FIN/RST (power loss, cable pull) must be detected by the kernel. Keepalive probes after idle_s of silence, every
    // intvl_s, cnt times; TCP_USER_TIMEOUT bounds unacknowledged sends the same way. A dead peer then fails recv/send
    // with ETIMEDOUT, which ends the idle wait. TCP sockets only.
    void keepalive(int idle_s, int intvl_s, int cnt) {
        require(idle_s > 0 && intvl_s > 0 && cnt > 0, "invalid keepalive");
        const int on = 1; const unsigned user_ms = unsigned(idle_s + intvl_s * cnt) * 1000u;
        require(!setsockopt(fd_, SOL_SOCKET, SO_KEEPALIVE, &on, sizeof on) &&
                !setsockopt(fd_, IPPROTO_TCP, TCP_KEEPIDLE, &idle_s, sizeof idle_s) &&
                !setsockopt(fd_, IPPROTO_TCP, TCP_KEEPINTVL, &intvl_s, sizeof intvl_s) &&
                !setsockopt(fd_, IPPROTO_TCP, TCP_KEEPCNT, &cnt, sizeof cnt) &&
                !setsockopt(fd_, IPPROTO_TCP, TCP_USER_TIMEOUT, &user_ms, sizeof user_ms), "control keepalive setup failed");
    }
    void send(const std::vector<uint8_t>& body) {
        require(!body.empty() && body.size() <= MAX_BYTES, "control frame size invalid");
        const uint32_t n = uint32_t(body.size());
        const uint8_t head[4] = {uint8_t(n >> 24), uint8_t(n >> 16), uint8_t(n >> 8), uint8_t(n)};
        send_all(head, 4); send_all(body.data(), body.size()); ++sent;
    }
    // idle = true: wait for the next frame header with no timeout (event readers only; see read_all)
    std::vector<uint8_t> receive(bool idle = false) {
        uint8_t h[4]; read_all(h, 4, idle);
        const uint32_t n = uint32_t(h[0]) << 24 | uint32_t(h[1]) << 16 | uint32_t(h[2]) << 8 | h[3];
        require(n && n <= MAX_BYTES, "control frame size invalid");
        std::vector<uint8_t> body(n); read_all(body.data(), n); ++received; return body;
    }
};
} // namespace g6s2
