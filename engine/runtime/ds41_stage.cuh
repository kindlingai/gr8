// F7 k16 DS41 staged slot read (GR8_DS_DEC_STAGE=1). Needs kernels/glue.cuh and g6nr.h before it; no includes of its own, because ds41_step.cuh includes it inside
// namespace g6rt. Test: engine/runtime/tests/ds41_stage_test.cu (host emulation of the tile maps + bit-equal against the Ds41Block slot reads on a spark).
#pragma once
// F7 k16 (GR8_DS_DEC_STAGE=1, default off): the same blocks with the 16-byte slot read the staged mHC kernels use (ds_mhc.cuh: has_stage_v, k_step / k_consume / k_finish).
// Same members, init() and wait() as B, so the layout and the wait are unchanged; only the kernels' read of the slots differs. The kernels sum the 4 slots in rank order
// from 0.f as Ds41Block::operator() does, so every result is bit-equal. Decode consumes only (rows <= MX; ds41_stage_on).
template<class B> struct Ds41Staged : B {
    static constexpr bool STAGE=true;
    static constexpr int NSLOT=g6nr::NR;
    __device__ __forceinline__ uint4 ld16(int p,size_t elem) const { return __ldcv(reinterpret_cast<const uint4*>(this->base[p]+elem)); }   // elem: a multiple of 8 BF16
};
