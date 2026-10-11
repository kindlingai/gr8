// The commit frame round trip (control_protocol.h): a grant's Run, list hash included, and a frame of another version
// refused. Host only.
#include "control_protocol.h"
#include <cstdio>
using namespace g6s2;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main()
{
    EventCommit e;
    e.notice = Notice{7, 12, 3400, 99, 0x1234, EventKind::Admit};
    e.commit.epoch = 7; e.commit.event = 12; e.commit.boundary = 3400; e.commit.before_hash = 0xfedcba9876543210ull;
    e.commit.cancel = Key{7, 2, 5};
    e.commit.admit = Grant{Key{7, 3, 9}, Request{99, 5000, 300}, Run{17, 84, 41, 0x8000000000000001ull}, 4096};
    auto b = encode(e);
    const EventCommit d = decode_commit(b);
    CHECK(d.commit.before_hash == e.commit.before_hash && d.commit.cancel && *d.commit.cancel == *e.commit.cancel);
    CHECK(d.commit.admit && d.commit.admit->run == e.commit.admit->run && d.commit.admit->reuse == 4096);
    CHECK(d.commit.admit->key == e.commit.admit->key && d.commit.admit->req.id == 99 && d.commit.admit->req.prompt_tokens == 5000);
    b[7] ^= 1;                                               // another frame version
    bool refused = false;
    try { decode_commit(b); } catch (const std::exception&) { refused = true; }
    CHECK(refused);
    std::printf("control_protocol_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
