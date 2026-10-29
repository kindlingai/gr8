// The commit frame round trip (control_protocol.h): a grant's Run, list hash included, its reused and restored rows, the prefixes it counted, its
// R, and a frame of another version refused. Host only.
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
    e.commit.admit = Grant{Key{7, 3, 9}, Request{99, 5000, 300}, Run{17, 84, 41, 0x8000000000000001ull}, 4096, 2, 3};
    e.commit.admit->req.r_rows = 16384;
    auto b = encode(e);
    const EventCommit d = decode_commit(b);
    CHECK(d.commit.before_hash == e.commit.before_hash && d.commit.cancel && *d.commit.cancel == *e.commit.cancel);
    CHECK(d.commit.admit && d.commit.admit->run == e.commit.admit->run && d.commit.admit->reuse == 4096 && d.commit.admit->restored == 2 &&
          d.commit.admit->counted == 3);
    CHECK(d.commit.admit->key == e.commit.admit->key && d.commit.admit->req.id == 99 && d.commit.admit->req.prompt_tokens == 5000);
    CHECK(d.commit.admit->req.r_rows == 16384);
    // an R off the page grid is refused
    EventCommit o = e;
    o.commit.admit->req.r_rows = 100;
    bool off_grid = false;
    try { (void)decode_commit(encode(o)); } catch (const std::exception&) { off_grid = true; }
    CHECK(off_grid);
    // a Resize event: its commit carries the change and nothing else, and needs a RESIZE notice
    EventCommit r;
    r.notice = Notice{7, 13, 3500, 0, 0, EventKind::Resize};
    r.commit.epoch = 7; r.commit.event = 13; r.commit.boundary = 3500; r.commit.before_hash = 42;
    r.commit.resize = Resize{1, 12, 12 * 38, 5};
    const EventCommit rd = decode_commit(encode(r));
    CHECK(rd.notice.kind == EventKind::Resize && rd.commit.resize && *rd.commit.resize == *r.commit.resize && !rd.commit.admit &&
          !rd.commit.cancel && !rd.commit.hold);
    bool mixed = false;
    r.commit.cancel = Key{7, 2, 5};
    try { (void)encode(r); } catch (const std::exception&) { mixed = true; }
    CHECK(mixed);
    bool unannounced = false;
    r.commit.cancel.reset(); r.notice.kind = EventKind::Cancel;
    try { (void)encode(r); } catch (const std::exception&) { unannounced = true; }
    CHECK(unannounced);
    b[7] ^= 1;                                               // another frame version
    bool refused = false;
    try { decode_commit(b); } catch (const std::exception&) { refused = true; }
    CHECK(refused);
    std::printf("control_protocol_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
