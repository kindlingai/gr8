// GLM-5.3-Flash, DFlash2 under --serve: which session's prefill taps Dfl::pf_feat holds, from which position, and
// whether a session may start on the drafter (g6m4.cu, the Flash serve's dfl_install). No CUDA.
//
// pf_feat holds one request's taps: the last WIN prompt positions [lo, hi) of the session whose prefill started last
// (owner; -1 = none). A session checkpoint hit (GR8_GLM53F_CACHE, flash_native.cuh bind_ckpt) restores rows [0, p) of
// the target but none of the drafter's context, and its prefill starts at p, so the taps cover [max(lo, p), hi) only.
// A hit at or below lo leaves the whole window to the prefill, as a cold prompt. A hit inside the window:
// - GR8_DFL_REUSE=0: no drafter for the session (owner -1, MTP for its whole life), as GLM-5.3 without the reuse;
// - GR8_DFL_REUSE >= 1 (Flash's default with the cache, g6m4.cu): the session owns the taps from `from` = p on, and
//   s2_batch.h Batch::reuse_start extends its context down into the window with the drafter's ring rows of the last
//   DFlash2 session when that session held the same tokens (the previous turn of the conversation), or runs MTP rounds
//   that feed the drafter until it holds GR8_DFL_REUSE_MIN rows.
// A hit that holds every prompt row but the last has no prefill at all (whole): with the reuse it starts on the drafter
// from its prompt step, as a 1-token prompt does. The same rules as the GLM-5.3 serve's prefix-reuse hits (g6m4.cu).
#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace g6s2 {
struct FlashDflWindow {
    int win = 2048;                 // dfl::WIN
    int reuse = 0;                  // GR8_DFL_REUSE (0: a hit into the window keeps no drafter)
    int owner = -1, from = 0;       // the session the taps are for, the first position they hold for it
    int lo = 0, hi = 0;             // the window of the request whose prefill started last (Dfl::pf_lo, pf_hi)
    uint64_t req = UINT64_MAX;      // that request
    std::vector<uint32_t> prompt;   // each slot's prompt tokens
    std::vector<bool> whole;        // each slot: a hit with no prefill (reuse only)

    FlashDflWindow(int sessions, int window, int reuse_mode)
        : win(window), reuse(reuse_mode), prompt(size_t(sessions), 0), whole(size_t(sessions), false) {
        if (sessions < 1 || window < 1 || reuse_mode < 0 || reuse_mode > 2)
            throw std::runtime_error("Flash DFlash2 window: sessions >= 1, window >= 1, GR8_DFL_REUSE 0..2");
    }
    // a grant bound to slot q: `reused` prompt rows came from a checkpoint (0: none)
    void bind(int q, uint32_t prompt_tokens, uint32_t reused) {
        check(q);
        prompt[size_t(q)] = prompt_tokens;
        whole[size_t(q)] = reuse && reused && reused + 1 == prompt_tokens;
        if (owner == q) owner = -1;   // the slot's previous session owned the taps
    }
    // slot q's prefill quantum, the slot at row `prefilled` (Slot::prefilled: the hit's rows at the first quantum).
    // true: the first quantum of request `id`, a new window [lo, hi) the caller clears before the prefill taps it
    bool quantum(int q, uint64_t id, uint32_t prompt_tokens, uint32_t prefilled) {
        check(q);
        if (id == req) return false;
        req = id;
        hi = int(prompt_tokens) - 1; lo = std::max(0, hi - win);
        owner = int(prefilled) <= lo || reuse ? q : -1;
        from = std::max(lo, int(prefilled));
        return true;
    }
    // slot q may start on the drafter at its prompt tail
    bool ok(int q) const {
        check(q);
        return owner == q || prompt[size_t(q)] == 1 || whole[size_t(q)];
    }
    // slot q's start fills the drafter from the taps, rows [from, hi): it owns a non-empty window. Else no fill, and
    // the drafter's context starts at the last prompt token (start)
    bool fills(int q) const { check(q); return owner == q && hi > lo; }
    int start(int q) const { return fills(q) ? from : int(prompt[size_t(q)]) - 1; }

private:
    void check(int q) const {
        if (q < 0 || q >= int(prompt.size())) throw std::runtime_error("Flash DFlash2 window: slot out of range");
    }
};
}
