// Socket headers for the S2 control path. g6m4 has a global namespace `bind` (kernels/bind.cuh), so POSIX bind() may not
// be declared in its translation unit: the headers are included with bind renamed, and bind goes through the syscall.
#pragma once
#define bind g6s2_posix_bind_undeclared
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#undef bind
#include <sys/syscall.h>
#include <fcntl.h>
#include <netinet/tcp.h>
#include <unistd.h>

namespace g6s2 {
#ifdef __APPLE__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"   // macOS still serves SYS_bind through syscall(2)
#endif
inline int posix_bind(int fd, const sockaddr* a, socklen_t n) { return int(::syscall(SYS_bind, fd, a, n)); }
#ifdef __APPLE__
#pragma clang diagnostic pop
#endif

// Linux sets close-on-exec and non-blocking in socket() and accept4(), and MSG_NOSIGNAL stops SIGPIPE per send. macOS
// (the front end's tests build these headers) has none of these, so it sets them with fcntl and SO_NOSIGPIPE. An
// accepted socket inherits O_NONBLOCK from its listener on macOS, so it is cleared to match Linux.
#ifdef SOCK_CLOEXEC
constexpr int SEND_FLAGS = MSG_NOSIGNAL;
inline int cloexec_socket(int domain, int type, bool nonblock) {
    return ::socket(domain, type | SOCK_CLOEXEC | (nonblock ? SOCK_NONBLOCK : 0), 0);
}
inline int cloexec_accept(int listener) { return ::accept4(listener, nullptr, nullptr, SOCK_CLOEXEC); }
#else
constexpr int SEND_FLAGS = 0;
inline int set_socket_flags(int fd, bool nonblock) {
    const int on = 1, fl = ::fcntl(fd, F_GETFL);
    if (fl < 0 || ::fcntl(fd, F_SETFD, FD_CLOEXEC) || ::fcntl(fd, F_SETFL, nonblock ? fl | O_NONBLOCK : fl & ~O_NONBLOCK) ||
        ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &on, sizeof on)) {
        ::close(fd);
        return -1;
    }
    return fd;
}
inline int cloexec_socket(int domain, int type, bool nonblock) {
    const int fd = ::socket(domain, type, 0);
    return fd < 0 ? fd : set_socket_flags(fd, nonblock);
}
inline int cloexec_accept(int listener) {
    const int fd = ::accept(listener, nullptr, nullptr);
    return fd < 0 ? fd : set_socket_flags(fd, false);
}
#endif
// TCP keepalive's idle time is TCP_KEEPIDLE on Linux and TCP_KEEPALIVE on macOS.
#ifdef TCP_KEEPIDLE
constexpr int TCP_KEEPALIVE_IDLE = TCP_KEEPIDLE;
#else
constexpr int TCP_KEEPALIVE_IDLE = TCP_KEEPALIVE;
#endif
// macOS has no TCP_USER_TIMEOUT: there only the keepalive probes detect a dead peer.
inline int set_user_timeout(int fd, unsigned ms) {
#ifdef TCP_USER_TIMEOUT
    return ::setsockopt(fd, IPPROTO_TCP, TCP_USER_TIMEOUT, &ms, sizeof ms);
#else
    (void)fd, (void)ms;
    return 0;
#endif
}
} // namespace g6s2
