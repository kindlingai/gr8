// DeepSeek-V4.1-Flash S2 prefill admission, host-only (no CUDA, no sessions.h): which prompts prefill, what one W-row chunk is, and the
// fabric collectives a chunk without a head rings. The W-row device work is the callee's (g6m4.cu, Ds41Context::prefill); this file is
// the contract it is built and tested against.
//
// Admission: by default every prompt runs as MAIN decode steps (Payload::decode_prompt = true). With GR8_DS_PREFILL_W set, a prompt of
// at least prefill_min rows prefills its first n - 1 rows in chunks of at most W rows; its last token is a MAIN row (PromptTail), so
// no chunk needs the head. prefill_min = 0 (the default) keeps decode_prompt true for every prompt.
//
// One fabric counter: prefill and decode rings share the main fabric, whose slot is step & 1 and whose doorbell is
// coll_seq(step, c, ncoll). A chunk takes one id from Dispatch's main sequence (Dispatch::prefill, prefill_seq); the next MAIN step has
// sequence + 1. Every rank runs the same Driver, so the ids agree.
#pragma once
#include "scheduler.h"
#include "../ds41_coll_order.h"
#include "../ds41_pack_layout.h"
#include <algorithm>
#include <array>
#include <functional>
#include <vector>

namespace g6s2 {
// DS41 headless W prefill is one native lane. Keep the shared Config's decode-only default
// untouched when prefill is disabled; the served DS41 pf_min branch calls this helper.
inline void ds41_prefill_single_lane(Config& sc, uint32_t rows) {
    require(rows > 0, "DS41 prefill width must be positive");
    sc.prefill_rows = rows;
    sc.two_lanes = false;
}
// does this prompt run as MAIN decode steps (true) or prefill (false)? Prompt logprobs (plp) score each prompt row against the next prompt
// token, which only a decode step produces; steering scales target MAIN rows. Both stay on the decode path.
inline bool ds41_decode_prompt(uint32_t prefill_min, size_t prompt_tokens, int plp, bool steering) {
    return prefill_min == 0 || prompt_tokens < prefill_min || plp >= 0 || steering;
}
// One W-row chunk of one session, as Dispatch hands it to the callee. The callee ENQUEUES and LAUNCHES (no sync); the caller fences.
struct Ds41PrefillChunk {
    const int* tok = nullptr;      // host: prompt[first .. first + rows - 1]; valid for the call
    uint32_t first = 0, rows = 0;  // row r sits at position first + r; rows <= W; first + rows < prompt size (the last token is a MAIN row)
    int context = 0;               // first + rows: the last row's position + 1 (k_step_in's StepIn::context)
    uint64_t sequence = 0;         // the fabric step id = the m.stepp value; from Dispatch's main sequence, never 0
    uint64_t request = 0;          // Slot::req.id
    unsigned slot = 0;             // Slot::key.slot (0 with one session)
    uint32_t first_page = 0, pages = 0;   // Slot::run
    uint32_t prompt_rows = 0;      // the prompt size N (GR8_DS_PF_LATE_FINAL: rows after this chunk = N - 1 - context)
};
using Ds41PrefillFn = std::function<void(const Ds41PrefillChunk&)>;
// One indivisible ragged W launch. Host token spans remain separately owned by the
// synchronous native call; layout is the shape every device op validates.
struct Ds41PrefillPack {
    std::array<Ds41PrefillChunk, MAX_SESSIONS> chunk{};
    ds41::pack::Layout layout{};
    uint64_t sequence = 0;
};
using Ds41PrefillPackFn = std::function<void(const Ds41PrefillPack&)>;
static_assert(ds41::pack::MAX_SLOTS == int(MAX_SESSIONS) && ds41::pack::MAX_SEGMENTS == int(PACK_MAX_MEMBERS),
              "DS41 pack slot/member caps must match the scheduler");
inline Ds41PrefillPack ds41_make_pack(const std::vector<Ds41PrefillChunk>& chunks, uint32_t width, unsigned slots) {
    require(chunks.size() >= 2 && chunks.size() <= size_t(ds41::pack::MAX_SEGMENTS) && width >= 1, "DS41 ragged pack member count/width");
    Ds41PrefillPack p; p.layout.count = int(chunks.size()); p.sequence = chunks[0].sequence;
    require(p.sequence != 0, "DS41 ragged pack sequence missing");
    for (int i = 0; i < p.layout.count; ++i) {
        const auto& c = chunks[i];
        require(c.tok && c.sequence == p.sequence && c.rows >= 1 && c.rows <= PAGE &&
                c.pages >= 1 && c.pages <= INT32_MAX / int(PAGE) && c.prompt_rows <= INT32_MAX &&
                uint64_t(p.layout.rows) + c.rows <= 512, "DS41 ragged pack token/sequence/size invalid");
        p.chunk[i] = c;
        p.layout.seg[i] = {p.layout.rows, int(c.rows), int(c.slot), int(c.first), c.context, int(c.prompt_rows), int(c.pages), c.request};
        p.layout.rows += int(c.rows);
    }
    require(ds41::pack::error(p.layout, int(width), int(slots)) == nullptr, "DS41 ragged pack layout invalid");
    return p;
}

inline Ds41PrefillChunk ds41_make_chunk(const std::vector<int>& prompt, uint32_t first, uint32_t rows, uint32_t width,
                                        uint64_t sequence, uint64_t request, unsigned slot, const Run& run) {
    require(rows >= 1 && rows <= width, "DS41 prefill chunk rows outside 1..W");
    require(uint64_t(first) + rows < prompt.size(), "DS41 prefill chunk reaches the prompt tail (the last token is a MAIN row)");
    require(uint64_t(first) + rows <= uint64_t(run.pages) * PAGE, "DS41 prefill chunk outside the session's run");
    require(sequence != 0, "DS41 prefill chunk without a fabric sequence");
    require(slot < MAX_SESSIONS, "DS41 prefill chunk slot");
    Ds41PrefillChunk c;
    c.tok = prompt.data() + first; c.first = first; c.rows = rows; c.context = int(first + rows);
    c.sequence = sequence; c.request = request; c.slot = slot; c.first_page = run.first_page; c.pages = run.pages;
    c.prompt_rows = uint32_t(prompt.size());
    return c;
}
// the attention step cell of one layer for the chunk: exactly what k_step_in (rank.cuh) writes for a step of `rows` rows at this context.
// DS41 serve never runs s2_init (Model::s2on false, StepIn::page_bound 0), so page_count is the context's pages.
struct Ds41PrefillCell { int context, page_count, table_size, reuse_rows; };
inline Ds41PrefillCell ds41_chunk_cell(const Ds41PrefillChunk& c, bool shared_index_layer) {
    const int pages = (c.context + int(PAGE) - 1) / int(PAGE);
    return {c.context, pages, pages, shared_index_layer ? int(c.rows) : 0};
}
// the fabric step of a chunk that has no head: `sizes` for Fabric::enqueue_step and `order` for Fabric::Rs::order.
//   Engram off: sizes 2*nl entries (c = 0..2nl-1; the head c = 2nl is not sized, or the proxy waits for a doorbell nobody rings), order
//               empty (the plain enqueue_step(step, sizes) call).
//   Engram on : sizes 2*nl + 3 entries, sizes[2nl] = 0 (the head, not in the order), sizes[2nl+1+e] = rows * engram_row_bytes;
//               order = ds41::coll_exec_order(ids, 2, false): the Engram reduces run mid-layers, so the proxy must wait for them there.
// row_bytes = HID * 2 (a layer exchange row); engram_row_bytes = 6144 * 2. Every size is a multiple of 8 and fits 32 bits.
// min_stride(): the least fabric slot stride that holds every size (Fabric::enqueue_step refuses sizes[c] + 8 > stride: payload + the u64 tail).
// The Engram reduces are wider than the layer ones (12,288 vs 10,240 B/row), so the slot is sized by the widest collective.
struct Ds41ChunkColl {
    std::vector<uint32_t> sizes; std::vector<int> order;
    uint64_t min_stride() const { uint32_t m = 0; for (uint32_t s : sizes) m = std::max(m, s); return uint64_t(m) + 8; }
};
// Bounded replay (ds41_pf_split.h): i_late >= 0 is the first layer position of range B; its layer exchanges c >= 2 * i_late carry late_rows
// rows (the attention publish of layer i_late is c = 2 * i_late). The Engram reduces (layers 1, 14 < i_late) keep `rows`. i_late = -1: every
// layer exchange carries `rows` (the unsplit chunk).
inline Ds41ChunkColl ds41_chunk_collectives(const std::vector<int>& ids, int ecoll, uint32_t rows, size_t row_bytes, size_t engram_row_bytes,
                                            int i_late = -1, uint32_t late_rows = 0, bool late_skip = false) {
    require(!ids.empty() && rows >= 1 && (ecoll == 0 || ecoll == 2), "DS41 chunk collectives: layers / rows / Engram collectives");
    require(row_bytes && engram_row_bytes && row_bytes % 8 == 0 && engram_row_bytes % 8 == 0, "DS41 chunk collectives: row bytes");
    require(i_late < 0 || (i_late >= 1 && size_t(i_late) < ids.size() && late_rows >= 1 && late_rows <= rows), "DS41 chunk collectives: late layers / rows");
    const uint64_t layer = uint64_t(rows) * row_bytes, engram = uint64_t(rows) * engram_row_bytes;
    require(layer <= UINT32_MAX && engram <= UINT32_MAX, "DS41 chunk collectives: size past 32 bits");
    const size_t nl = ids.size();
    if (i_late >= 0 && ecoll)
        for (int i = i_late; i < int(nl); ++i) require(!geom::is_engram(ids[size_t(i)]), "DS41 chunk collectives: an Engram layer past the bounded-replay boundary");
    const size_t c_late = i_late < 0 ? 2 * nl : 2 * size_t(i_late);
    Ds41ChunkColl o;
    o.sizes.assign(2 * nl + (ecoll ? size_t(1 + ecoll) : 0), 0);
    for (size_t c = 0; c < 2 * nl; ++c) o.sizes[c] = uint32_t(c < c_late ? layer : uint64_t(late_rows) * row_bytes);
    for (int e = 0; e < ecoll; ++e) o.sizes[2 * nl + 1 + size_t(e)] = uint32_t(engram);
    o.order = ds41::coll_exec_order(ids, ecoll, false);
    // GR8_DS_PF_LATE_FINAL: a chunk that skips range B rings no layer exchange c >= c_late. Engram off: the proxy handles 0..sizes.size()-1,
    // so sizes stops at c_late. Engram on: those sizes are 0 and the order leaves them out (as it leaves out the head).
    if (late_skip) {
        require(i_late >= 1, "DS41 chunk collectives: late skip without a bounded-replay split");
        if (!ecoll) o.sizes.resize(c_late);
        else {
            for (size_t c = c_late; c < 2 * nl; ++c) o.sizes[c] = 0;
            std::vector<int> keep;
            for (int c : o.order) if (c >= int(2 * nl) || size_t(c) < c_late) keep.push_back(c);
            o.order = keep;
        }
    }
    return o;
}
}
