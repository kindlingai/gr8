// Persistent host-control sockets for the opt-in S2 server. No CUDA/fabric I/O.
// Call AFTER local model load. A failed handshake never publishes ready.s2.
#pragma once
#include "control_io.h"
#include "ingress.h"
#include <chrono>
#include <memory>
#include <thread>
#include <poll.h>
#include <fcntl.h>

namespace g6s2 {
class Socket {
    int fd_ = -1;
public:
    explicit Socket(int fd = -1) : fd_(fd) {}
    ~Socket() { if (fd_ >= 0) ::close(fd_); }
    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;
    int get() const { return fd_; }
    int release() { int fd = fd_; fd_ = -1; return fd; }
};
using Deadline = std::chrono::steady_clock::time_point;
inline Deadline deadline(unsigned seconds) {
    require(seconds && seconds <= 86400, "invalid control timeout");
    return std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
}
inline int remaining_ms(Deadline end) {
    auto n = std::chrono::duration_cast<std::chrono::milliseconds>(end - std::chrono::steady_clock::now()).count();
    require(n > 0, "control bootstrap timed out");
    return int(std::min<int64_t>(n, INT32_MAX));
}
inline void socket_wait(int fd, short events, Deadline end) {
    for (;;) {
        pollfd p{fd, events, 0};
        int n = ::poll(&p, 1, remaining_ms(end));
        if (n < 0 && errno == EINTR) continue;
        require(n > 0 && !(p.revents & POLLNVAL), "control socket wait failed");
        return; // caller's accept / SO_ERROR / recv distinguishes hangup and readiness
    }
}
inline sockaddr_in address(const std::string& ip, uint16_t port) {
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(port);
    require(port && inet_pton(AF_INET, ip.c_str(), &a.sin_addr) == 1, "control requires IPv4 and nonzero port");
    return a;
}
inline int listen_control(const std::string& ip, uint16_t port) {
    Socket fd(::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0));
    require(fd.get() >= 0, "control socket failed");
    int one = 1;
    require(!setsockopt(fd.get(), SOL_SOCKET, SO_REUSEADDR, &one, sizeof one), "control reuseaddr failed");
    const auto a = address(ip, port);
    require(!posix_bind(fd.get(), reinterpret_cast<const sockaddr*>(&a), sizeof a) && !::listen(fd.get(), 8), "control listen failed");
    return fd.release();
}
inline std::unique_ptr<ControlIO> accept_control(int listener, unsigned timeout) {
    auto end = deadline(timeout);
    for (;;) {
        socket_wait(listener, POLLIN, end);
        Socket fd(::accept4(listener, nullptr, nullptr, SOCK_CLOEXEC));
        if (fd.get() < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
        require(fd.get() >= 0, "control accept failed");
        auto io = std::make_unique<ControlIO>(fd.release()); io->timeout_seconds(timeout); io->keepalive(10, 5, 4); return io; // dead peer seen in <= 30 s (#26)
    }
}
inline std::unique_ptr<ControlIO> connect_control(const std::string& ip, uint16_t port, unsigned timeout) {
    const auto a = address(ip, port); const auto end = deadline(timeout);
    for (;;) {
        Socket fd(::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0));
        require(fd.get() >= 0, "control socket failed");
        int rc = ::connect(fd.get(), reinterpret_cast<const sockaddr*>(&a), sizeof a), error = 0;
        if (rc && errno == EINPROGRESS) {
            socket_wait(fd.get(), POLLOUT, end); socklen_t n = sizeof error;
            require(!getsockopt(fd.get(), SOL_SOCKET, SO_ERROR, &error, &n), "control connect status failed");
        } else if (rc) error = errno;
        if (!error) {
            require(!fcntl(fd.get(), F_SETFL, 0), "control blocking mode failed");
            auto io = std::make_unique<ControlIO>(fd.release()); io->timeout_seconds(timeout); io->keepalive(10, 5, 4); return io; // dead peer seen in <= 30 s (#26)
        }
        require(error == ECONNREFUSED || error == EINTR, "control connect failed");
        // Worker may finish loading before rank0. Bootstrap only, never decode.
        std::this_thread::sleep_for(std::chrono::milliseconds(std::min(20, remaining_ms(end))));
    }
}
inline std::vector<uint8_t> frame(const std::string& s) { return {s.begin(), s.end()}; }
struct BootstrapConfig {
    Config scheduler;
    unsigned rank = 0, ranks = 6, timeout = 120;
    std::string head_ip, directory, runtime_signature;
    uint16_t port = 0;
    void validate() const {
        scheduler.validate();
        require((ranks == 4 || ranks == 6) && rank < ranks, "invalid control rank count");
        (void)address(head_ip, port); (void)deadline(timeout);
        require(!directory.empty() && !runtime_signature.empty(), "missing bootstrap directory/runtime signature");
    }
    std::string agreement() const {
        return capability(scheduler, ranks) + std::to_string(scheduler.max_sessions) + " " +
            std::to_string(scheduler.prefill_rows) + " " + std::to_string(scheduler.two_lanes) + "\n" + runtime_signature;
    }
};
inline void atomic_file(const std::string& path, const std::string& bytes) {
    const auto temp = path + ".tmp";
    { std::ofstream f(temp, std::ios::binary | std::ios::trunc); require(bool(f), "cannot create S2 file");
      f << bytes; f.close(); require(bool(f), "S2 file write failed"); }
    require(!::rename(temp.c_str(), path.c_str()), "S2 file publication failed");
}
struct ControlLinks {
    std::vector<std::unique_ptr<ControlIO>> links; // head: rank1..NR-1; worker: only head
    std::vector<ControlIO*> peers() const {
        std::vector<ControlIO*> p; for (const auto& x : links) p.push_back(x.get()); return p;
    }
    void shutdown() { for (auto& p : links) if (p) p->shutdown(); }
};
inline ControlLinks bootstrap(const BootstrapConfig& c) {
    c.validate();
    // Fresh per-rank directory is part of the protocol. Check before any socket.
    for (const char* f : {"ready", "ready.s2", "abort.s2", "quit", "cmd.0", "req.0"})
        require(!exists(c.directory + "/" + f), "S2 directory is not fresh");
    const auto agree = c.agreement();
    const auto release = frame("G6S2 START\n" + agree);
    const auto ack = frame("G6S2 STARTED\n" + agree);
    ControlLinks out;
    try {
        if (!c.rank) {
            Socket listener(listen_control(c.head_ip, c.port));
            out.links.resize(c.ranks - 1);
            for (unsigned n = 1; n < c.ranks; ++n) {
                auto io = accept_control(listener.get(), c.timeout); const auto b = io->receive();
                std::string hello(b.begin(), b.end()); const auto eol = hello.find('\n');
                require(eol != std::string::npos && hello.substr(0, 5) == "G6S2 ", "bad bootstrap hello");
                unsigned rank = 0; const auto nr = number(hello.substr(5, eol - 5));
                require(nr > 0 && nr < c.ranks, "bad bootstrap worker rank"); rank = unsigned(nr);
                require(!out.links[rank - 1] && hello.substr(eol + 1) == agree, "duplicate rank or bootstrap config mismatch");
                out.links[rank - 1] = std::move(io);
            }
            for (auto& io : out.links) io->send(release);
            for (auto& io : out.links) require(io->receive() == ack, "worker bootstrap ACK mismatch");
            // Return with all rank identities/configs acknowledged. Caller starts
            // its control/compute service before publishing capability below.
        } else {
            auto io = connect_control(c.head_ip, c.port, c.timeout);
            io->send(frame("G6S2 " + std::to_string(c.rank) + "\n" + agree));
            require(io->receive() == release, "head bootstrap config mismatch");
            io->send(ack); out.links.push_back(std::move(io));
        }
        return out;
    } catch (...) { out.shutdown(); throw; }
}
inline void publish_ready(const BootstrapConfig& c, const ControlLinks& links) {
    require(c.rank == 0 && links.links.size() == c.ranks - 1, "ready before all-rank bootstrap");
    for (auto& p : links.links) require(bool(p), "ready with missing rank");
    require(!exists(c.directory + "/abort.s2"), "aborted before ready");
    atomic_file(c.directory + "/ready.s2", capability(c.scheduler, c.ranks));
}
} // namespace g6s2
