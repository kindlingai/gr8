// Deliberately wrong: the stock launch, then one write into a row the kernel must never write (the first dead column of
// the first partly-live tile, else the first unused tile). A masked row is not "don't care": it must stay pristine.
// The bench must report MISMATCH. This must NEVER be scored as a candidate.
namespace sweep_variant {
static __global__ void poke_dead_row(const int* tile_mask, const int* n_tiles, unsigned short* y) {
    const int nt = *n_tiles; size_t row = size_t(nt) * 8;
    for (int t = 0; t < nt; ++t) if (tile_mask[t] != 255) { row = size_t(t) * 8 + (__ffs(~tile_mask[t]) - 1); break; }
    y[row * 5120] = 0x3c00;   // fp16 1.0
}
inline const void* kernel(const sweep_adapter::Params& p) { return sweep_reference::kernel(p); }
inline const char* ptxas_key(const sweep_adapter::Params& p) { return sweep_reference::ptxas_key(p); }
inline size_t dynamic_smem(const sweep_adapter::Params& p) { return sweep_reference::dynamic_smem(p); }
inline cudaError_t init(const sweep_adapter::Params&) { return cudaSuccess; }
inline void launch(const sweep_adapter::Params& p, cudaStream_t st) {
    sweep_reference::launch(p, st);
    poke_dead_row<<<1, 1, 0, st>>>(p.a.tile_mask, p.a.n_tiles_dev, reinterpret_cast<unsigned short*>(p.a.ypart));
}
}
