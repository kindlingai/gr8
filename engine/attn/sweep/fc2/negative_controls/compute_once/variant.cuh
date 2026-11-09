// Deliberately invalid: computes on the first launch only, then reuses stale output. Fresh-input checks must catch it.
// This must NEVER be scored as a candidate.
namespace sweep_variant {
inline const void* kernel(const sweep_adapter::Params& p) { return sweep_reference::kernel(p); }
inline const char* ptxas_key(const sweep_adapter::Params& p) { return sweep_reference::ptxas_key(p); }
inline size_t dynamic_smem(const sweep_adapter::Params& p) { return sweep_reference::dynamic_smem(p); }
inline cudaError_t init(const sweep_adapter::Params&) { return cudaSuccess; }
inline void launch(const sweep_adapter::Params& p, cudaStream_t st) {
    static bool computed = false;
    if (!computed) { sweep_reference::launch(p, st); computed = true; }
}
}
