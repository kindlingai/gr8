// DS41 SESS > 1 main step layout check, host-only (no CUDA) so s2/tests/ds41_s32_step_test.cpp runs the code ds41_native.cuh runs.
#pragma once

namespace g6s2 {
// h: the step's S2In (native_inputs), slots: Model::ds_slots, mtp: GR8_DS_MTP, round: DSpark on. ns sessions of rps uniform
// rows each; writes each row's slot to si.sess. nullptr = ok, else the reason (ds41_native.cuh requires it).
template<class S2In, class StepIn>
inline const char* ds41_session_rows(const S2In& h, int slots, int mtp, bool round, StepIn& si) {
    const int rps = h.rows[0];
    for (int j = 0; j < h.ns; ++j) {
        if (!(h.sess[j] >= 0 && h.sess[j] < slots && h.rows[j] == rps && h.row0[j] == j * rps)) return "DS41 SESS step layout is not uniform";
        for (int r = 0; r < rps; ++r) si.sess[j * rps + r] = h.sess[j];
    }
    if (!(h.ns == 1 || rps == 1 || (round && rps <= mtp + 1)))
        return "DS41 SESS step of several sessions with more rows each than a verify (needs GR8_DS_PREFILL_MIN >= 2)";
    return nullptr;
}
}  // namespace g6s2
