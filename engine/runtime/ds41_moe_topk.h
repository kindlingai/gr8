// Host-only selection rule for the DS41 main-model route cut.
#pragma once
#include <string>
#include "../knobs-rs/gr8_knobs.h"

namespace ds41_moe_topk {
// GR8_DS_MOE_TOPK_DECODE: the decode step's routed experts, at most the main step's; unset = the main step's
inline int decode_env(int main_cut, int max_cut, std::string* err) {
    const int v = (int)knob::ds_moe_topk_decode().value_or(main_cut);
    if (v < 1 || v > max_cut || v > main_cut) {
        *err = "GR8_DS_MOE_TOPK_DECODE must be 1.." + std::to_string(main_cut < max_cut ? main_cut : max_cut);
        return -1;
    }
    return v;
}
inline int for_step(bool scheduler_decode, int main_cut, int decode_cut) {
    return scheduler_decode ? decode_cut : main_cut;
}
inline int graph_kind(bool scheduler_decode, int main_cut, int decode_cut) {
    return scheduler_decode && decode_cut != main_cut ? 1 : 0;
}
}
