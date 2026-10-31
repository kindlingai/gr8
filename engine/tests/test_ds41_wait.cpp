// Host test of F1 (GR8_DS_WAIT_K), engine/runtime/ds41_wait.h: no CUDA. Build:
//   g++ -std=c++17 -O2 -Wall -Wextra -DG6_MODEL_DS41=1 -I engine engine/tests/test_ds41_wait.cpp -o /tmp/test_ds41_wait && /tmp/test_ds41_wait
// What it pins: (1) off = the exact argument the consume kernel had before F1 (rows * HID * 2, no wait kernel); (2) on = one wait kernel with that
// same payload and wait_payload 0 in the block; (3) in both modes exactly ONE party waits, for exactly the bytes put_publish writes
// (rows * (HID * 2 / 16) uint4); (4) a stream model of the 80 consumes of a step: every consume(c) has its wait for c ahead of it (own or
// the kernel's), in both modes; (5) K=2 (relay): wait_mode() truth table (2 only on the main Model without GR8_DRAFT_PAR, else 1), consume_wait_mode()
// (no wait kernel, the block's payload is the published bytes, relay set; modes 0 / 1 equal the old bool overload, relay clear); (6) a std::thread model of the
// relay protocol (NIC thread writes payload then the tail with release, CTA (0,0) acquires the tails and release-stores the relay word, the other CTAs acquire the
// relay word and then read the payload; 400 collectives, one "kernel" each): every reader sees the payload of its collective. Build with -pthread
// (and -fsanitize=thread to let TSAN judge the happens-before chain). It checks the logic of the chain under the C++ memory model, not the PTX one.
#include <atomic>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "runtime/ds41_ops.h"
#include "runtime/ds41_wait.h"

#define CHECK(x) do { if (!(x)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #x); std::exit(1); } } while (0)

// what put_publish writes per slot: k_put_publish gets n16 = M * (HID * 2 / 16) uint4
static size_t published_bytes(int rows, int hid) { return (size_t)rows * (hid * 2 / 16) * 16; }

int main()
{
    const int HID = geom::HID;
    CHECK(HID == 5120);
    int n = 0;
    for (int rows = 1; rows <= 32; ++rows)
    {
        const ds41::ConsumeWait off = ds41::consume_wait(false, rows, HID), on = ds41::consume_wait(true, rows, HID);
        // (1) off: the pre-F1 argument, no wait kernel
        CHECK(!off.kernel && off.kernel_payload == 0 && off.block_payload == (size_t)rows * HID * 2);
        // (2) on: one wait kernel with the same payload, the block skips
        CHECK(on.kernel && on.kernel_payload == (size_t)rows * HID * 2 && on.block_payload == 0);
        // (3) one waiter, the published bytes
        CHECK((off.block_payload != 0) != off.kernel && (on.block_payload != 0) != on.kernel);
        CHECK((off.kernel ? off.kernel_payload : off.block_payload) == published_bytes(rows, HID));
        CHECK((on.kernel ? on.kernel_payload : on.block_payload) == published_bytes(rows, HID));
        ++n;
    }
    // (4) stream model: ds41_consume_wait runs right before the consume launch; Ds41Block::wait = `if (wait_payload) wait`
    for (int mode = 0; mode < 2; ++mode)
    {
        std::vector<std::string> stream;
        for (int c = 0; c < 80; ++c)
        {
            const ds41::ConsumeWait w = ds41::consume_wait(mode == 1, 5, HID);
            if (w.kernel) stream.push_back("wait" + std::to_string(c));
            stream.push_back("consume" + std::to_string(c) + (w.block_payload ? "+ownwait" : ""));
        }
        int ownwaits = 0, kwaits = 0;
        for (size_t i = 0; i < stream.size(); ++i)
        {
            const std::string& s = stream[i];
            if (s.rfind("consume", 0) != 0) { ++kwaits; continue; }
            const std::string c = s.substr(7, s.find('+') == std::string::npos ? std::string::npos : s.find('+') - 7);
            const bool own = s.find("+ownwait") != std::string::npos;
            ownwaits += own;
            CHECK(own || (i > 0 && stream[i - 1] == "wait" + c));   // the wait for c sits directly ahead, or inside
        }
        CHECK(mode == 0 ? (ownwaits == 80 && kwaits == 0) : (ownwaits == 0 && kwaits == 80));
    }
    // (5) K=2 rules
    {
        CHECK(ds41::wait_mode(0, false, false) == 0 && ds41::wait_mode(0, true, true) == 0);
        CHECK(ds41::wait_mode(1, false, false) == 1 && ds41::wait_mode(1, true, false) == 1 && ds41::wait_mode(1, false, true) == 1);
        CHECK(ds41::wait_mode(2, false, false) == 2);
        CHECK(ds41::wait_mode(2, true, false) == 1 && ds41::wait_mode(2, false, true) == 1 && ds41::wait_mode(2, true, true) == 1);
        CHECK(ds41::wait_mode(3, false, false) == 1 && ds41::wait_mode(-1, false, false) == 1);
        for (int rows = 1; rows <= 32; ++rows)
        {
            const ds41::ConsumeWait m0 = ds41::consume_wait_mode(0, rows, HID), m1 = ds41::consume_wait_mode(1, rows, HID), m2 = ds41::consume_wait_mode(2, rows, HID);
            const ds41::ConsumeWait o = ds41::consume_wait(false, rows, HID), k = ds41::consume_wait(true, rows, HID);
            CHECK(m0.kernel == o.kernel && m0.kernel_payload == o.kernel_payload && m0.block_payload == o.block_payload && !m0.relay);
            CHECK(m1.kernel == k.kernel && m1.kernel_payload == k.kernel_payload && m1.block_payload == k.block_payload && !m1.relay);
            CHECK(!m2.kernel && m2.kernel_payload == 0 && m2.relay && m2.block_payload == published_bytes(rows, HID));
        }
        ++n;
    }
    // (6) the relay chain, modelled with threads
    {
        const int NCTA = 160, NPEER = 3, NCOLL = 400, WORDS = 64;
        std::vector<std::vector<unsigned long long>> payload(NPEER, std::vector<unsigned long long>(WORDS, 0));   // plain stores, like the RDMA data
        std::atomic<unsigned long long> tail[NPEER], relay{0};
        for (auto& t : tail) t.store(0);
        std::atomic<int> bad{0};
        for (int coll = 0; coll < NCOLL; ++coll)
        {
            const unsigned long long seq = 7 + (unsigned long long)coll;   // unique and increasing, like coll_seq
            std::vector<std::thread> th;
            for (int p = 0; p < NPEER; ++p)       // the NIC / proxy: data, then the tail behind it
                th.emplace_back([&, p] {
                    if (coll % 3 == 0) std::this_thread::yield();
                    for (int w = 0; w < WORDS; ++w) payload[p][w] = seq * 1000 + p * 100 + w;
                    tail[p].store(seq, std::memory_order_release);
                });
            for (int cta = 0; cta < NCTA; ++cta)  // the consume kernel
                th.emplace_back([&, cta] {
                    if (cta == 0)
                    {
                        for (int p = 0; p < NPEER; ++p) while (tail[p].load(std::memory_order_acquire) != seq) {}
                        relay.store(seq, std::memory_order_release);
                    }
                    else while (relay.load(std::memory_order_acquire) != seq) {}
                    for (int p = 0; p < NPEER; ++p)
                        for (int w = 0; w < WORDS; w += 7)
                            if (payload[p][w] != seq * 1000 + p * 100 + w) bad.fetch_add(1);
                });
            for (auto& t : th) t.join();      // kernel end: the next collective's kernel starts after it
        }
        CHECK(bad.load() == 0);
        ++n;
    }
    std::printf("PASS test_ds41_wait: %d row counts x off/on, 80-consume stream order in both modes, K=2 mode rules, relay chain model\n", n);
    return 0;
}
