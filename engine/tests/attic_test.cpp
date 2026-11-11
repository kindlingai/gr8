// engine/attic_proto.h on the host: the request bytes (the same as _proto.py and the TensorFold client send), the key
// rule, the reply parser, and a client against a fake atticd on a real SOCK_SEQPACKET socket that sends a descriptor
// beside its alloc reply (SCM_RIGHTS), refuses, sends too many descriptors and hangs up.
//   g++ -std=c++17 -O2 -Wall -Wextra -pthread -I engine engine/tests/attic_test.cpp -o /tmp/attic_test && /tmp/attic_test
#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>   // the fake daemon binds: the real bind() before attic_proto.h's renamed include
#include <sys/stat.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include "attic_proto.h"

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

// the fake daemon's side of one message: a reply and `n` descriptors of fresh memory
static void reply(int s, const std::string& body, int n)
{
    std::vector<int> fds;
    for (int i = 0; i < n; ++i)
    {
        const int f = memfd_create("attic-test", MFD_CLOEXEC);
        CHECK(f >= 0 && ftruncate(f, 4 << 20) == 0);
        fds.push_back(f);
    }
    iovec io = {const_cast<char*>(body.data()), body.size()};
    msghdr m = {};
    m.msg_iov = &io;
    m.msg_iovlen = 1;
    alignas(cmsghdr) char ctl[CMSG_SPACE(4 * sizeof(int))];
    if (n)
    {
        m.msg_control = ctl;
        m.msg_controllen = CMSG_SPACE(n * sizeof(int));
        cmsghdr* c = CMSG_FIRSTHDR(&m);
        c->cmsg_level = SOL_SOCKET;
        c->cmsg_type = SCM_RIGHTS;
        c->cmsg_len = CMSG_LEN(n * sizeof(int));
        memcpy(CMSG_DATA(c), fds.data(), n * sizeof(int));
    }
    CHECK(sendmsg(s, &m, 0) == (ssize_t)body.size());
    for (int f : fds) close(f);
}

static std::string request_of(int s)
{
    char b[1024];
    const ssize_t n = recv(s, b, sizeof b, 0);
    return n > 0 ? std::string(b, (size_t)n) : std::string();
}

static bool open_fd(int f) { return f >= 0 && fcntl(f, F_GETFD) >= 0; }

// this process's open descriptors (the directory's own is counted the same way every time)
static int open_count()
{
    int n = 0;
    if (DIR* d = opendir("/proc/self/fd"))
    {
        while (readdir(d)) ++n;
        closedir(d);
    }
    return n;
}

int main()
{
    using namespace g6attic;
    const int open_at_start = open_count();
    // the requests, byte for byte
    CHECK(request("spark-attic-1", "info") == "{\"key\":\"spark-attic-1\",\"op\":\"info\"}");
    CHECK(request("attic_0123456789abcdef01234567", "alloc", 1904214016) ==
          "{\"key\":\"attic_0123456789abcdef01234567\",\"op\":\"alloc\",\"size\":1904214016}");
    CHECK(request("k", "status") == "{\"key\":\"k\",\"op\":\"status\"}");
    // the key rule: 1 to 127 printable bytes, no space, quote or backslash
    CHECK(key_ok("spark-attic-1") && key_ok(std::string(127, 'k')));
    for (const char* bad : {"", "a b", "a\"b", "a\\b", "a\tb", "a\x7f" "b", "caf\xc3\xa9"}) CHECK(!key_ok(bad));
    CHECK(!key_ok(std::string(128, 'k')));
    // replies
    std::vector<uint64_t> got;
    std::string why;
    CHECK(ok_reply("{\"ok\":true,\"size\":2143289344,\"free\":2143289344,\"largest\":2143289344}", {"largest", "size"}, got, why));
    CHECK(got.size() == 2 && got[0] == 2143289344ull && got[1] == 2143289344ull);
    CHECK(ok_reply("{\"ok\": true, \"size\": 4194304}", {"size"}, got, why) && got[0] == 4194304);   // the spacing of a fake
    CHECK(!ok_reply("{\"ok\":false,\"error\":\"bad key\"}", {}, got, why) && why == "atticd refused: bad key");
    CHECK(!ok_reply("{\"ok\":true}", {"size"}, got, why) && why == "atticd's reply has no size");
    CHECK(!ok_reply("{\"ok\":true,\"size\":-2}", {"size"}, got, why));
    CHECK(!ok_reply("{\"ok\":true,\"size\":\"4\"x}", {"size"}, got, why));
    CHECK(!ok_reply("{\"ok\":true,\"regions\":[1,2]}", {}, got, why));      // nested: not a flat object
    CHECK(!ok_reply("not json", {}, got, why) && !ok_reply("{\"ok\":true", {}, got, why) && !ok_reply("", {}, got, why));
    CHECK(ok_reply("{}", {}, got, why) == false);                             // no "ok"
    std::vector<std::pair<std::string, std::string>> m;
    CHECK(members("{\"error\":\"say \\\"no\\\"\",\"ok\":false}", m) && m.size() == 2 && m[1].second == "false");

    // no daemon
    Client none;
    CHECK(!none.connect("/nonexistent/attic.sock", why) && why.find("no atticd at /nonexistent/attic.sock") == 0);
    CHECK(!none.connect(std::string(200, 'x'), why) && none.fd < 0);

    // a fake atticd
    const std::string path = "/tmp/attic-test-" + std::to_string(getpid()) + ".sock";
    unlink(path.c_str());
    const int srv = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    sockaddr_un a = {};
    a.sun_family = AF_UNIX;
    memcpy(a.sun_path, path.data(), path.size());
    CHECK(srv >= 0 && bind(srv, reinterpret_cast<sockaddr*>(&a), sizeof a) == 0 && listen(srv, 1) == 0);
    std::vector<std::string> seen;
    std::thread daemon([&] {
        const int s = accept4(srv, nullptr, nullptr, SOCK_CLOEXEC);
        CHECK(s >= 0);
        seen.push_back(request_of(s));
        reply(s, "{\"ok\":true,\"size\":2143289344,\"free\":2143289344,\"largest\":2143289344}", 0);
        seen.push_back(request_of(s));
        reply(s, "{\"ok\":true,\"size\":4194304}", 1);                 // a lent slice
        seen.push_back(request_of(s));
        reply(s, "{\"ok\":false,\"error\":\"not enough free\"}", 0);    // a refusal carries none
        seen.push_back(request_of(s));
        reply(s, "{\"ok\":true,\"size\":4194304}", 2);                  // one too many
        seen.push_back(request_of(s));
        close(s);                                                       // hang up
    });
    Client c;
    CHECK(c.connect(path, why));
    std::string rep;
    int fd = 7;
    CHECK(c.ask(request("spark-attic-1", "info"), rep, nullptr, why) && ok_reply(rep, {"largest"}, got, why) && got[0] == 2143289344ull);
    CHECK(c.ask(request("spark-attic-1", "alloc", 4 << 20), rep, &fd, why) && ok_reply(rep, {"size"}, got, why));
    CHECK(open_fd(fd));
    struct stat st;
    CHECK(fstat(fd, &st) == 0 && st.st_size == 4 << 20);
    CHECK((fcntl(fd, F_GETFD) & FD_CLOEXEC) != 0);                          // MSG_CMSG_CLOEXEC
    close(fd);
    CHECK(c.ask(request("spark-attic-1", "alloc", 4 << 20), rep, &fd, why) && fd == -1);
    CHECK(!ok_reply(rep, {"size"}, got, why) && why == "atticd refused: not enough free");
    CHECK(!c.ask(request("spark-attic-1", "alloc", 4 << 20), rep, &fd, why) && why == "atticd sent 2 descriptors with its reply");
    CHECK(!c.ask(request("spark-attic-1", "info"), rep, nullptr, why) && why == "atticd closed the connection");
    daemon.join();
    c.close();
    close(srv);
    unlink(path.c_str());
    CHECK(seen.size() == 5);
    if (seen.size() == 5)
    {
        CHECK(seen[0] == "{\"key\":\"spark-attic-1\",\"op\":\"info\"}");
        CHECK(seen[1] == "{\"key\":\"spark-attic-1\",\"op\":\"alloc\",\"size\":4194304}");
    }
    // every descriptor is closed again, the two of the refused reply included
    CHECK(open_count() == open_at_start);

    if (fails) { fprintf(stderr, "attic_test: %d failed\n", fails); return 1; }
    printf("attic_test: all passed\n");
    return 0;
}
