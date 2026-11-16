// The slot sizes of the prefill fabrics: prefill.cuh and flash_prefill.cuh read them, and so does the fabric plan
// (fabric_plan.h), which is host code. Host only.
#pragma once
#include <algorithm>
#include <cstddef>
#include "../g6nr.h"
#include "../model_geom.h"

namespace g6rt
{
inline size_t pf_ss_min(int Mmax, int nr) { return ((size_t)(Mmax + nr - 1) / nr * geom::HID * 2 + 8 + 255) & ~size_t(255); }
// FabricConfig::slot_bytes of a prefill region
inline size_t pf_slot_bytes(int Mmax, bool rs, int nr = g6nr::NR)
{
    const size_t ar = (size_t)Mmax * geom::HID * 2;
    return rs ? std::max(ar, nr * pf_ss_min(Mmax, nr)) : ar;
}
// GLM's reduce-scatter form (GR8_PF_RS=1; prefill.cuh pf_enqueue): c0 and c2 are reduce-scatters, so their send slot holds
// nr slices (pf_slot_bytes). c1 gathers each rank's normed rows; c3 its normed rows or GR8_PF_OWN records; c4 the last
// layer's FP32 residual rows or GR8_PF_OWN=2's owner IDs. A rank writes its own rows at the start of its slot, so these
// three hold one rank's rows: pf_ag_rows of pf_ag_row_bytes (c3, c4) or HID * 2 bytes (c1).
inline size_t pf_ag_rows(int Mmax, int nr) { return (size_t)(Mmax + nr - 1) / nr; }
inline size_t pf_ag_row_bytes()
{
    const size_t own = (size_t)(geom::QA + geom::KVA + geom::IDX_DIM + geom::IDX_HEADS) * 2;   // prefill.cuh OWN_REC
    const size_t ids = (size_t)2048 * 4;                                                        // prefill.cuh IX_ROW
    return std::max({(size_t)geom::HID * 4, own, ids});
}
// Flash with GR8_GLM53F_PF_SP (flash_prefill.cuh)
inline size_t pf_sp_slot_bytes(int MP, int nr)
{
    return std::max(pf_slot_bytes(MP, true, nr), ((size_t)(MP + nr - 1) / nr * geom::HID * 4 + 8 + 255) & ~size_t(255));
}
}   // namespace g6rt
