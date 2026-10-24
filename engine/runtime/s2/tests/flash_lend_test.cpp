// FLASH_CAP_POLICY=borrow-idle (s2/flash_lend.h) and the ACK's room (Ack::room, control_channel.h): the lend comes once
// per idle stretch and only after idle_ms; a request is admitted only when the least room over the ranks covers its
// worst case, else it waits for a recall to the chunks it needs; the room crosses the wire and the head sees the least
// of the ranks' (a rank whose recall the KV node did not release in time holds the request back). Host only.
#include "control_channel.h"
#include "flash_lend.h"
#include <sys/socket.h>
#include <atomic>
#include <cstdio>
#include <thread>
using namespace g6s2;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main()
{
    using ms = std::chrono::milliseconds;
    const auto t0 = FlashLend::Clock::time_point{} + std::chrono::hours(1);
    // Flash CAP 262,208 rows = 4,097 pages; 256-page chunks of 16,384 rows: 17 chunks, the last one page
    FlashLend f;
    f.idle_ms = 1000; f.keep = 1; f.chunk_rows = 256 * 64; f.chunks = 17;
    CHECK(f.need(1) == 1 && f.need(16384) == 1 && f.need(16385) == 2 && f.need(262208) == 17 && f.need(1u << 30) == 17);

    // the lend: not before idle_ms of idleness, once per idle stretch, again after a busy spell
    CHECK(!f.poll(true, t0));
    CHECK(!f.poll(true, t0 + ms(999)));
    CHECK(f.poll(true, t0 + ms(1000)) == std::optional<uint32_t>(1));
    CHECK(!f.poll(true, t0 + ms(5000)));
    CHECK(!f.poll(false, t0 + ms(5001)));   // a request: the stretch ends
    CHECK(!f.poll(true, t0 + ms(5002)));
    CHECK(!f.poll(true, t0 + ms(6001)));
    CHECK(f.poll(true, t0 + ms(6002)) == std::optional<uint32_t>(1));
    // busy before the stretch ends: no lend
    CHECK(!f.poll(false, t0 + ms(7000)));
    CHECK(!f.poll(true, t0 + ms(7001)));
    CHECK(!f.poll(false, t0 + ms(7500)));
    CHECK(!f.poll(true, t0 + ms(7600)));
    CHECK(!f.poll(true, t0 + ms(8500)));
    CHECK(f.poll(true, t0 + ms(8600)));
    // protected (idle_ms 0) never lends
    FlashLend p = f; p.idle_ms = 0;
    CHECK(!p.poll(true, t0) && !p.poll(true, t0 + std::chrono::hours(9)));

    // admission: room covers the worst case = admit (0), else recall to what it needs
    CHECK(f.recall(10000, 1) == 0);           // a short request runs on the chunk kept online
    CHECK(f.recall(100000, 1) == 7);          // 100,000 rows need 7 chunks: recall first
    CHECK(f.recall(100000, 7) == 0);
    CHECK(f.recall(262208, 16) == 17);        // one rank still short after a recall the node did not release
    CHECK(f.recall(262208, UINT32_MAX) == 0); // no rank has a chunk map
    FlashLend off;                            // no chunk map: no gate
    CHECK(off.need(262208) == 0 && off.recall(262208, 0) == 0);

    // the wire: Ack::room round trips, and the head's agreement keeps the least room
    Notice n{1, 1, 20, 7, 0, EventKind::Admit};
    Ack a{n, 1, 42, 5};
    const Ack b = decode_ack(encode(a));
    CHECK(b.room == 5 && b.state_hash == 42 && b.rank == 1);
    CHECK(decode_ack(encode(Ack{n, 1, 42})).room == UINT32_MAX);
    Agreement ag(2, n);
    ag.ack(Ack{n, 0, 42, 17});
    ag.ack(Ack{n, 1, 42, 3});
    CHECK(ag.ready() && ag.room() == 3);

    // TP2 over a socket pair: the worker reports 3 chunks, the head 17; inside make_commit the head sees 3, and a later
    // event after the worker's recall sees 17
    int fd[2];
    CHECK(!socketpair(AF_UNIX, SOCK_STREAM, 0, fd));
    ControlIO hi(fd[0]), wi(fd[1]);
    LeaderChannel head(1, {&hi});
    WorkerChannel worker(1, 1, wi);
    std::atomic<uint32_t> head_room{17}, worker_room{3};
    head.room = [&] { return head_room.load(); };
    worker.room = [&] { return worker_room.load(); };
    std::thread reader([&] { try { worker.receive_events(); } catch (...) {} });
    std::atomic<int> applied{0};
    auto compute = [&](EventEndpoint& e, int events) {
        uint64_t b = 1;
        for (int k = 0; k < events; ++k)
        {
            e.wait_notice();
            for (;; ++b)
                if (auto c = e.boundary(b, [] { return uint64_t(42); })) { e.applied(); ++applied; ++b; break; }
        }
    };
    std::thread h([&] { compute(head, 2); }), w([&] { compute(worker, 2); });
    uint32_t seen[2] = {0, 0};
    Notice n1{1, 1, 20, 7, 0, EventKind::Cancel};
    head.event(n1, [&](uint64_t hash) { seen[0] = head.agreed_room(); return Commit{n1.epoch, n1.event, n1.boundary, hash, {}, {}}; });
    worker_room = 17;   // the worker's recall went through
    Notice n2{1, 2, 40, 8, 0, EventKind::Cancel};
    head.event(n2, [&](uint64_t hash) { seen[1] = head.agreed_room(); return Commit{n2.epoch, n2.event, n2.boundary, hash, {}, {}}; });
    h.join(); w.join();
    CHECK(applied == 4 && seen[0] == 3 && seen[1] == 17);
    head.fail();
    reader.join();

    std::printf("flash_lend_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
