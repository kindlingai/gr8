#pragma once
// Diagnostic host writer. No CUDA calls; instantiated inert unless open() is called.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace draft_head_dump {
struct Row {
    int32_t pos, input, output, predict_pos;
    uint32_t seed_lo, seed_hi;
    int32_t draw;
    float inv_temp, top_p, lp;
};
static_assert(sizeof(Row) == 40 && sizeof(float) == 4, "dump requires 32-bit scalar fields");
class Writer {
    FILE* f = nullptr;
    uint64_t calls = 0, rows = 0;
    int hid = 0, mx = 0;
    [[noreturn]] static void fail(const char* why) {
        fprintf(stderr, "G6_DRAFT_HEAD_DUMP: %s\n", why); std::exit(1);
    }
    void put(const void* p, size_t n) {
        if (fwrite(p, 1, n, f) != n) fail("write failed");
    }
public:
    Writer() = default;
    Writer(const Writer&) = delete;
    Writer& operator=(const Writer&) = delete;
    ~Writer() { if (f) fclose(f); } // no success footer on an unfinished run
    bool on() const { return f != nullptr; }
    void open(const char* path, int nr, int rank, int hidden, int vocab, int max_rows, int k, bool adaptive) {
        const uint32_t endian = 1;
        if (*(const unsigned char*)&endian != 1) fail("little-endian host required");
        if (f || !path || !*path || nr < 1 || rank < 0 || rank >= nr || hidden < 1 ||
            vocab < 1 || max_rows < 1 || k < 1 || k >= max_rows) fail("invalid header");
        const std::string name = std::string(path) + ".rank" + std::to_string(rank);
        f = fopen(name.c_str(), "wbx"); // refuse to silently replace an earlier capture
        if (!f) fail("cannot create rank file (parent must exist; file must not exist)");
        hid = hidden; mx = max_rows;
        put("G6DHD001", 8);
        const uint32_t h[] = {uint32_t(nr), uint32_t(rank), uint32_t(hid), uint32_t(vocab),
                              uint32_t(mx), uint32_t(k), uint32_t(adaptive), 1}; // 1 = FP32 xn
        put(h, sizeof h);
    }
    void call(uint64_t seq, uint64_t step, int s, int m, bool chain, int context,
              const Row* row, const float* xn) {
        if (!f) return;
        if (m < 1 || m > mx || (chain && m != 1) || !row || !xn) fail("invalid call shape");
        put("CALL", 4);
        const uint64_t ids[] = {calls, seq, step}; put(ids, sizeof ids);
        const int32_t meta[] = {s, m, chain ? 1 : 0, context}; put(meta, sizeof meta);
        put(row, size_t(m) * sizeof(Row));
        put(xn, size_t(m) * hid * sizeof(float));
        ++calls; rows += m;
        if (fflush(f)) fail("flush failed");
    }
    void finish() {
        if (!f) return;
        put("DONE", 4);
        const uint64_t n[] = {calls, rows}; put(n, sizeof n);
        FILE* out = f; f = nullptr;
        if (fclose(out)) fail("close failed");
    }
};
} // namespace draft_head_dump
