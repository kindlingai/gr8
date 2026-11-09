// attnproj knob link: the C knob getters (engine/knobs-rs/gr8_knobs.h) that the engine's attention TUs read, with their STOCK values.
// The payload has no libgr8knobs.a (cargo is not available offline), so this file answers the four ids the three engine TUs
// (attn/ds41_attn.cu, attn/ds41_bind.cu, attn/ds41_idx.cu) read through knob::*, with the default of knobs/ds41-attn.yaml
// (stock sets none of them: work-decode/mb/ds/stock-effective.txt). Any other id exits 2 (fail closed): a tree whose TUs read
// another knob must be re-checked, not silently defaulted.
//   id 152 ds_wob_persist  GR8_DS_WOB_PERSIST   flag, default 0   (ds41_attn.cuh:2232)
//   id  80 ds_c2_trace     GR8_DS_C2_TRACE      flag, default 0   (ds41_attn.cuh:2237)
//   id  57 ds41_idx_logits GR8_DS41_IDX_LOGITS  text, default "mma"   (ds41_bind.cu:17)
//   id  65 ds41_idx_topk   GR8_DS41_IDX_TOPK    text, default "mc"    (ds41_bind.cu:22)
// The replay itself never calls knob:: getters. It hard-codes the stock values it sets (see replay.cu bind_stock()). The new proj-fuse knob (id 73, GR8_DS_ATTN_PROJ_FUSE) is read only by ds41_attn_ops.cuh (the rank TU), which this replay does not compile.
// A set GR8_* environment variable is ignored on purpose: the replay measures the stock configuration only.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

static void unknown(const char* kind, uint32_t id)
{
    std::fprintf(stderr, "attnproj knob_stub: %s knob id %u is not one of 152/80/57/65: the engine TUs read a knob this stub does not know\n", kind, id);
    std::exit(2);
}
extern "C" int gr8_knob_flag(uint32_t id, uint32_t, int* out)
{
    if (id == 152u || id == 80u) { *out = 0; return 1; }
    unknown("flag", id);
    return 0;
}
extern "C" int gr8_knob_int(uint32_t id, uint32_t, int64_t*) { unknown("int", id); return 0; }
extern "C" int gr8_knob_float(uint32_t id, uint32_t, double*) { unknown("float", id); return 0; }
extern "C" const char* gr8_knob_text(uint32_t id, uint32_t)
{
    if (id == 57u) return "mma";
    if (id == 65u) return "mc";
    unknown("text", id);
    return nullptr;
}
extern "C" ptrdiff_t gr8_knob_ints(uint32_t id, uint32_t, const int64_t**) { unknown("ints", id); return -1; }
extern "C" ptrdiff_t gr8_knob_floats(uint32_t id, uint32_t, const double**) { unknown("floats", id); return -1; }
