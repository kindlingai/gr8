// Deliberately wrong: launches column groups 0..2 only; the last 1280 columns stay poison. Must report MISMATCH.
// This must NEVER be scored as a candidate.
namespace sweep_variant {
inline const void* kernel(const sweep_adapter::Params& p) { return sweep_reference::kernel(p); }
inline const char* ptxas_key(const sweep_adapter::Params& p) { return sweep_reference::ptxas_key(p); }
inline size_t dynamic_smem(const sweep_adapter::Params& p) { return sweep_reference::dynamic_smem(p); }
inline cudaError_t init(const sweep_adapter::Params&) { return cudaSuccess; }
inline void launch(const sweep_adapter::Params& p, cudaStream_t st) {
    for (int c = 0; c < sweep_adapter::kColGroups - 1; ++c) {
        auto a = p.a; a.col_group = c;
        mxtc::k_fc2_tc<576, 5120, 4, 2, 8, 8, 4><<<a.n_slots * a.gy2 * a.os2, 128, mxtc::fc2_smem<4, 8, 8>(), st>>>(a);
    }
}
}
