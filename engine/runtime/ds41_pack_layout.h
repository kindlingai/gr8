// DS41 F12: host description of one ragged short-prompt W pass. This is never a decode
// graph shape; decode keeps its existing uniform ds_ns/rps representation.
#pragma once
#include <array>
#include <algorithm>
#include <cstdint>

namespace ds41::pack {
constexpr int MAX_SEGMENTS = 8, MAX_ROWS = 512, MAX_SEGMENT_ROWS = 64;
// S2 session slots a segment may name (g6s2::MAX_SESSIONS): SESS=32 has slot ids 0..31, a pack at most 8 of them.
constexpr int MAX_SLOTS = 32;
struct Segment {
    int row0 = 0, rows = 0;       // packed activation/fabric slice [row0,row0+rows)
    int slot = 0;                 // independent S2 KV/index/Engram/DSpark state slab
    int first = 0, context = 0;   // positions in this session, not in the packed tensor
    int prompt_rows = 0, run_pages = 0;
    uint64_t request = 0;
};
struct Layout {
    int count = 0, rows = 0;
    std::array<Segment, MAX_SEGMENTS> seg{};
};
// Return the first invalid condition; callers fail closed before any device launch. Apart
// from the request id, all values are bounded so row offsets cannot wrap an int.
inline const char* error(const Layout& p, int width, int slots) {
    if (width < 1 || slots < 2 || slots > MAX_SLOTS || p.count < 2 || p.count > std::min(slots, MAX_SEGMENTS) ||
        p.rows < 2 || p.rows > std::min(width, MAX_ROWS)) return "ragged pack shape/cap";
    std::array<bool, MAX_SLOTS> seen{};
    int end = 0; uint64_t last = 0; bool have_last = false;
    for (int i = 0; i < p.count; ++i) {
        const Segment& s = p.seg[i];
        if (s.row0 != end || s.rows < 1 || s.rows > MAX_SEGMENT_ROWS || s.rows > p.rows - end)
            return "ragged pack row range";
        if (s.slot < 0 || s.slot >= slots || seen[s.slot]) return "ragged pack slot ownership";
        if (s.first != 0 || s.context != s.rows || s.prompt_rows != s.rows + 1 ||
            s.run_pages < 1 || int64_t(s.rows) > int64_t(s.run_pages) * 64) return "ragged pack prompt/run bound";
        if (have_last && s.request <= last) return "ragged pack request order"; // ordinal 0 is a valid first request
        seen[s.slot] = true; last = s.request; have_last = true; end += s.rows;
    }
    return end == p.rows ? nullptr : "ragged pack total mismatch";
}
inline const char* attn_error(const Layout& p, int rows, int rowcap, int slots) {
    if (const char* e = error(p, rowcap, slots)) return e;
    return p.rows == rows ? nullptr : "ragged attention row count";
}
inline const char* index_error(const Layout& p, int rows, int rowcap, int sess_cap, int slots) {
    if (const char* e = error(p, rowcap, slots)) return e;
    if (p.rows != rows) return "ragged index row count";
    return sess_cap >= MAX_SEGMENT_ROWS ? nullptr : "ragged index session scratch under 64 rows";
}
inline const char* dspark_error(const Layout& p, int rows, int tapcap, int slots, bool has_sess, int lo, int hi) {
    if (const char* e = error(p, tapcap, slots)) return e;
    return p.rows == rows && has_sess && lo == 0 && (hi < 0 || hi == rows) ? nullptr : "ragged DSpark tap/key layout";
}
}
