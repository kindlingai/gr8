// The slot sizes of the prefill fabrics: prefill.cuh and flash_prefill.cuh read them, and so does the fabric plan. The
// rules are engine/scheduler-rs src/plan_sizes.rs. Host only.
#pragma once
#include <cstddef>
#include "../g6nr.h"
#include "../model_geom.h"
#include "../scheduler-rs/g6plan.h"

namespace g6rt
{
inline size_t pf_ss_min(int Mmax, int nr) { return g6plan_pf_ss_min(Mmax, nr, geom::HID); }
// FabricConfig::slot_bytes of a prefill region
inline size_t pf_slot_bytes(int Mmax, bool rs, int nr = g6nr::NR) { return g6plan_pf_slot_bytes(Mmax, rs, nr, geom::HID); }
// Flash with GR8_GLM53F_PF_SP (flash_prefill.cuh)
inline size_t pf_sp_slot_bytes(int MP, int nr) { return g6plan_pf_sp_slot_bytes(MP, nr, geom::HID); }
}   // namespace g6rt
