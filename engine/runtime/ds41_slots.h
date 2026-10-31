#pragma once
// SESS (S2 session slots) range checks of build_ds41 and ds41dsp_ops::build, host-only (no CUDA) so engine/tests/test_ds41_slots.cpp runs them.
// nullptr = ok, else the reason. The draft Model has one ring slab per S2 slot like the main Model: same range for both.
namespace ds41slots {
constexpr int MAX_SLOTS = 32;
inline const char* slots_error(int slots)
{
    return slots < 1 || slots > MAX_SLOTS ? "DS41 build: ds_slots (S2 sessions) must be 1..32" : nullptr;
}
// DSpark: nsess = min(main_slots, maxs) (<= maxs, the markov step's sessions; above maxs S2 dynamically turns MTP off); k = MTP depth 1..block;
// a round of up to nsess sessions must fit one step: nsess x (k + 1) verify rows and nsess x block draft rows <= mx
inline const char* dsp_nsess_error(int nsess, int main_slots, int draft_slots, int k, int mx, int maxs, int block)
{
    const int want_nsess = main_slots < maxs ? main_slots : maxs;
    if (nsess < 1 || nsess > maxs || nsess != want_nsess || draft_slots != main_slots) return "DSpark: nsess must be the S2 slots of both Models";
    if (k < 1 || k > block) return "DSpark: k must be 1..5";
    if (nsess * (k + 1) > mx || nsess * block > mx) return "DSpark: the sessions need more rows a step than MX";
    return nullptr;
}
}  // namespace ds41slots
