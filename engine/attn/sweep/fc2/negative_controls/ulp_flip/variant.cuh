// Deliberately wrong: the stock launch, then ONE bit (the LSB of the first live fp16 output element, 1 ulp) flipped.
// The bench must report MISMATCH with exactly 1 differing byte. This must NEVER be scored as a candidate.
namespace sweep_variant {
static __global__ void flip_lsb(unsigned short* y) { y[0] ^= 1; }   // row 0 column 0 is live in every plan
inline const void* kernel(const sweep_adapter::Params& p) { return sweep_reference::kernel(p); }
inline const char* ptxas_key(const sweep_adapter::Params& p) { return sweep_reference::ptxas_key(p); }
inline size_t dynamic_smem(const sweep_adapter::Params& p) { return sweep_reference::dynamic_smem(p); }
inline cudaError_t init(const sweep_adapter::Params&) { return cudaSuccess; }
inline void launch(const sweep_adapter::Params& p, cudaStream_t st) {
    sweep_reference::launch(p, st);
    flip_lsb<<<1, 1, 0, st>>>(reinterpret_cast<unsigned short*>(p.a.ypart));
}
}
