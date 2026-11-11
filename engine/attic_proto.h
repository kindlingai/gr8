// The carveout lender's socket protocol: KindlingAI's dispramd protocol (dispram, Kindling Spark OS), as atticd
// (github.com/bertholomus/spark-attic) speaks it; engine/attic.h uses it. The lender gives the display carveout as whole 2 MiB
// slices over an AF_UNIX SOCK_SEQPACKET socket, one flat JSON object a message, a slice's descriptor beside the alloc
// reply (SCM_RIGHTS). The requests are byte for byte those of the other atticd clients (spark-attic's own):
// python/spark_attic/_proto.py, the TensorFold client patches). No CUDA here: engine/tests/attic_test.cpp tests it on the
// host.
#pragma once
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
// g6m4 has a global namespace `bind` (kernels/bind.cuh), so POSIX bind() may not be declared in its translation unit
// (as in runtime/s2/posix_socket.h): the socket headers come in with bind renamed. A client never binds.
#define bind g6attic_posix_bind_undeclared
#include <sys/socket.h>
#include <sys/un.h>
#undef bind
#include <sys/time.h>
#include <unistd.h>
#include "knobs-rs/gr8_knobs.h"

namespace g6attic
{

constexpr const char* DEFAULT_KEY = "spark-attic-1";   // atticd's own default key
constexpr size_t SLICE = 2u << 20;                       // atticd lends whole 2 MiB slices
constexpr size_t KEY_MAX = 127;                          // a key goes into a request as it is
constexpr size_t REPLY_MAX = 4096;
constexpr size_t MAX_FDS = 12;                           // room for descriptors: a reply carrying more is refused
constexpr int TIMEOUT_MS = 5000;                         // a daemon that does not answer in this long is none

// 1 to 127 printable ASCII bytes, none a space, '"' or '\\': a key we can send without escaping
inline bool key_ok(const std::string& k)
{
    if (k.empty() || k.size() > KEY_MAX) return false;
    for (unsigned char c : k)
        if (c < 0x21 || c > 0x7e || c == '"' || c == '\\') return false;
    return true;
}

// {"key":K,"op":OP} or {"key":K,"op":"alloc","size":N}: the same bytes as _proto.py's json.dumps(separators=(",", ":"))
inline std::string request(const std::string& key, const char* op, size_t size = 0)
{
    std::string r = "{\"key\":\"" + key + "\",\"op\":\"" + op + "\"";
    if (size) r += ",\"size\":" + std::to_string(size);
    return r + "}";
}

// The members of a flat JSON object (atticd's info and alloc replies), each value as its text: a string without its quotes
// (escapes kept as written), a number, true, false or null as written. false for anything else (nested values included).
inline bool members(const std::string& j, std::vector<std::pair<std::string, std::string>>& out)
{
    size_t i = 0;
    auto ws = [&] { while (i < j.size() && (j[i] == ' ' || j[i] == '\t' || j[i] == '\n' || j[i] == '\r')) ++i; };
    auto str = [&](std::string& s) {
        if (i >= j.size() || j[i] != '"') return false;
        const size_t a = ++i;
        while (i < j.size() && j[i] != '"') i += j[i] == '\\' ? 2 : 1;
        if (i >= j.size()) return false;
        s = j.substr(a, i++ - a);
        return true;
    };
    ws();
    if (i >= j.size() || j[i++] != '{') return false;
    ws();
    if (i < j.size() && j[i] == '}')
    {
        ++i;
        ws();
        return i == j.size();
    }
    for (;;)
    {
        std::string k, v;
        ws();
        if (!str(k)) return false;
        ws();
        if (i >= j.size() || j[i++] != ':') return false;
        ws();
        if (i < j.size() && j[i] == '"') { if (!str(v)) return false; }
        else
        {
            const size_t a = i;
            while (i < j.size() && j[i] != ',' && j[i] != '}' && j[i] != ' ' && j[i] != '\t' && j[i] != '\n' && j[i] != '\r') ++i;
            v = j.substr(a, i - a);
            if (v.empty() || v[0] == '{' || v[0] == '[') return false;
        }
        out.emplace_back(k, v);
        ws();
        if (i >= j.size()) return false;
        if (j[i] == '}') { ++i; ws(); return i == j.size(); }
        if (j[i++] != ',') return false;
    }
}

// A reply that says "ok":true, with the numbers asked for. why: atticd's own reason when it refused.
inline bool ok_reply(const std::string& reply, const std::vector<const char*>& want, std::vector<uint64_t>& got, std::string& why)
{
    std::vector<std::pair<std::string, std::string>> m;
    if (!members(reply, m)) { why = "atticd's reply is not a flat JSON object"; return false; }
    auto find = [&](const char* k) -> const std::string* {
        for (const auto& kv : m) if (kv.first == k) return &kv.second;
        return nullptr;
    };
    const std::string* ok = find("ok");
    if (!ok || *ok != "true")
    {
        const std::string* e = find("error");
        why = "atticd refused: " + (e ? *e : std::string("no reason given"));
        return false;
    }
    got.clear();
    for (const char* k : want)
    {
        const std::string* v = find(k);
        if (!v || v->empty() || v->find_first_not_of("0123456789") != std::string::npos || v->size() > 19)
        {
            why = std::string("atticd's reply has no ") + k;
            return false;
        }
        got.push_back(std::stoull(*v));
    }
    return true;
}

// One connection to atticd. The slices it lends stay lent to it until close().
struct Client
{
    int fd = -1;

    bool connect(const std::string& path, std::string& why)
    {
        sockaddr_un a = {};
        a.sun_family = AF_UNIX;
        if (path.empty() || path.size() >= sizeof a.sun_path) { why = "no atticd at " + path; return false; }
        memcpy(a.sun_path, path.data(), path.size());
        fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
        if (fd < 0) { why = std::string("socket: ") + strerror(errno); return false; }
        const timeval tv = {TIMEOUT_MS / 1000, (TIMEOUT_MS % 1000) * 1000};
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
        int r;
        do r = ::connect(fd, reinterpret_cast<const sockaddr*>(&a), sizeof a); while (r < 0 && errno == EINTR);
        if (r < 0)
        {
            why = "no atticd at " + path + " (" + strerror(errno) + ")";
            close();
            return false;
        }
        return true;
    }

    // one request, one reply; fdout: the descriptor an alloc reply carries, -1 with none (a refusal). A reply carrying
    // more than it may is refused and every descriptor it carried closed.
    bool ask(const std::string& req, std::string& reply, int* fdout, std::string& why)
    {
        ssize_t n;
        do n = send(fd, req.data(), req.size(), MSG_NOSIGNAL); while (n < 0 && errno == EINTR);
        if (n != (ssize_t)req.size()) { why = std::string("send to atticd: ") + (n < 0 ? strerror(errno) : "short"); return false; }
        char buf[REPLY_MAX];
        alignas(cmsghdr) char ctl[CMSG_SPACE(MAX_FDS * sizeof(int))];
        iovec io = {buf, sizeof buf};
        msghdr m = {};
        m.msg_iov = &io;
        m.msg_iovlen = 1;
        m.msg_control = ctl;
        m.msg_controllen = sizeof ctl;
        do n = recvmsg(fd, &m, MSG_CMSG_CLOEXEC); while (n < 0 && errno == EINTR);
        std::vector<int> fds;
        if (n > 0)
            for (cmsghdr* c = CMSG_FIRSTHDR(&m); c; c = CMSG_NXTHDR(&m, c))
                if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS)
                    for (size_t k = 0; k < (c->cmsg_len - CMSG_LEN(0)) / sizeof(int); ++k)
                    {
                        int x;
                        memcpy(&x, CMSG_DATA(c) + k * sizeof(int), sizeof x);
                        fds.push_back(x);
                    }
        auto refuse = [&](const std::string& w) {
            for (int x : fds) ::close(x);
            why = w;
            return false;
        };
        if (n < 0) return refuse(std::string("no reply from atticd: ") + strerror(errno));
        if (n == 0) return refuse("atticd closed the connection");
        if (m.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) return refuse("atticd's reply did not fit");
        if (fds.size() > (fdout ? 1u : 0u)) return refuse("atticd sent " + std::to_string(fds.size()) + " descriptors with its reply");
        reply.assign(buf, (size_t)n);
        if (fdout) *fdout = fds.empty() ? -1 : fds[0];
        return true;
    }

    void close()
    {
        if (fd >= 0) ::close(fd);
        fd = -1;
    }
};

// The key the requests carry: GR8_ATTIC_KEY_FILE's text with its whitespace trimmed, else atticd's default
inline bool key_of(std::string& key, std::string& why)
{
    static const auto file = knob::attic_key_file();
    if (!file) { key = DEFAULT_KEY; return true; }
    std::ifstream in(*file);
    std::stringstream s;
    if (in) s << in.rdbuf();
    if (!in) { why = "GR8_ATTIC_KEY_FILE " + *file + ": " + strerror(errno); return false; }
    key = s.str();
    const size_t a = key.find_first_not_of(" \t\r\n"), b = key.find_last_not_of(" \t\r\n");
    key = a == std::string::npos ? std::string() : key.substr(a, b - a + 1);
    if (!key_ok(key)) { why = "GR8_ATTIC_KEY_FILE does not hold a key we can send (1 to 127 printable bytes, no space, quote or backslash)"; return false; }
    return true;
}

}   // namespace g6attic
