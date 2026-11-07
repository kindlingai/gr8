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
// Flash with GR8_GLM53F_PF_SP (flash_prefill.cuh)
inline size_t pf_sp_slot_bytes(int MP, int nr)
{
    return std::max(pf_slot_bytes(MP, true, nr), ((size_t)(MP + nr - 1) / nr * geom::HID * 4 + 8 + 255) & ~size_t(255));
}
}   // namespace g6rt
