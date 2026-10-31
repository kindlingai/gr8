// MoE kernel headers for the DS41 step, at global scope. rank.cuh includes this next to ds41_ops.h / ds_mhc.cuh (before
// namespace g6rt opens); ds41_moe_ops.cuh (the adapters, included inside g6rt by ds41_step.cuh) refuses to compile without it.
#pragma once
#include "ds_shared.cuh"      // mx4:: (routed MXFP4 GEMV), dsr:: (router, plan), dsh:: (shared MXFP8 expert, combine)
#include "ds_prefill_ro.cuh"   // read-once expert loop + multi-CTA planner
#include "ds_shared_gl.cuh"    // shared expert group loop (2 launches per layer; the adapter uses it only with DS41_SH_GL = 1)
#define G6_DS41_MOE_KERNELS 1
#include "ds41_l2pf.h"        // GR8_L2PF targets of a MoE layer (the gate, the shared expert); host only
