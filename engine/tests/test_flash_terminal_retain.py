"""Compile the actual G6_FLASH_CACHE=terminal retain hook (s2/flash_native.cuh) against CPU mocks: the rows it keeps,
the KDA ring slot it keeps them from, the tokens (through the pending token p), and when it keeps nothing."""
from pathlib import Path
import subprocess
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    source = (root / 'runtime/s2/flash_native.cuh').read_text()
    start = source.index('        h.pool.retain = [&]')
    hook = source[start:source.index('\n        };', start) + len('\n        };')]
    program = r'''
#include <algorithm>
#include <cassert>
#include <functional>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include "s2/flash_ring.h"
using namespace g6s2;
namespace g6rt {
struct CacheNamespace { int id = 0; };
struct PrefixReuse { std::vector<int> history; int main_end = 0; };
}
struct Run { int id = 0; };
struct Capture { uint32_t p; int slot; std::vector<int> tokens; bool terminal; };
struct Index { size_t n = 0; size_t size() const { return n; } };
struct Ckpt {
    std::vector<Capture> got;
    bool full = false;
    struct Last { uint64_t blocks = 0; double ms = 0; } last;
    Index ix;
    Index& index() { return ix; }
    uint64_t blocks() const { return 300; }
    uint64_t capture(uint32_t p, int slot, const std::vector<int>& t, const g6rt::CacheNamespace&, int, bool terminal) {
        if (full) return 0;
        got.push_back({p, slot, t, terminal}); ++ix.n;
        return got.size();
    }
};
struct Pool { std::vector<int> released; void release(const Run& r) { released.push_back(r.id); } };
int main() {
    struct { bool terminal = true; int stream = 0; } c;
    Ckpt real; Ckpt* ck = &real;
    Pool pool; FlashRing ring; std::mutex ck_mu;
    struct { struct { std::function<void(const Run&, const g6rt::PrefixReuse&, const g6rt::CacheNamespace&)> retain; } pool; } h;
    int flushed = 0;   // the pending aligned capture, taken before the terminal one (a cancel at its boundary)
    auto take_capture = [&] { ++flushed; };
''' + hook + r'''
    std::vector<int> seq(100);
    for (int i = 0; i < 100; ++i) seq[i] = 1000 + i;
    auto st = [&](int n, int main_end) { g6rt::PrefixReuse s; s.history.assign(seq.begin(), seq.begin() + n); s.main_end = main_end; return s; };
    int cases = 0;
    // a verify step at 40 of 2 rows (MTP=1), both accepted and the bonus token: 43 tokens, 42 consumed, ring slot 1
    ring.record(40, 2, true);
    h.pool.retain(Run{1}, st(43, 42), {});
    assert(real.got.size() == 1 && real.got[0].p == 42 && real.got[0].slot == 1 && real.got[0].terminal);
    assert(real.got[0].tokens.size() == 43 && real.got[0].tokens.back() == seq[42]);
    assert(pool.released.back() == 1); ++cases;
    // the same step, the draft rejected: 42 tokens, 41 consumed, slot 0
    h.pool.retain(Run{2}, st(42, 42), {});
    assert(real.got.size() == 2 && real.got[1].p == 41 && real.got[1].slot == 0 && real.got[1].tokens.size() == 42); ++cases;
    // a tail step (not verify) at 50 of 1 row: the state is in slot 0
    ring.record(50, 1, false);
    h.pool.retain(Run{3}, st(52, 51), {});
    assert(real.got.size() == 3 && real.got[2].p == 51 && real.got[2].slot == 0); ++cases;
    // main_end below the consumed rows (a cancel mid-prefill: 30 prompt rows prefilled, no step yet): p = 30, slot 0
    ring.reset();
    h.pool.retain(Run{4}, st(60, 30), {});
    assert(real.got.size() == 4 && real.got[3].p == 30 && real.got[3].slot == 0 && real.got[3].tokens.size() == 31); ++cases;
    // the ring does not hold the state for p (p before the last step): nothing kept, the run still released
    ring.record(40, 2, true);
    h.pool.retain(Run{5}, st(41, 39), {});
    assert(real.got.size() == 4 && pool.released.back() == 5); ++cases;
    // one token, or a full store: nothing kept, the run released
    h.pool.retain(Run{6}, st(1, 0), {});
    real.full = true;
    h.pool.retain(Run{7}, st(43, 42), {});
    assert(real.got.size() == 4 && pool.released.size() == 7); ++cases;
    // aligned mode: the hook only releases
    c.terminal = false; real.full = false;
    h.pool.retain(Run{8}, st(43, 42), {});
    assert(real.got.size() == 4 && pool.released.back() == 8); ++cases;
    assert(flushed == 8);   // every retain flushed the pending aligned capture first
    printf("flash_terminal_retain: PASS %d CPU mock cases (actual retain hook)\n", cases);
}
'''
    with tempfile.TemporaryDirectory() as d:
        src, exe = Path(d) / 't.cpp', Path(d) / 't'
        src.write_text(program)
        subprocess.run(['g++', '-std=c++17', '-O1', '-Wall', '-I', str(root / 'runtime'), str(src), '-o', str(exe)],
                       check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == '__main__':
    main()
