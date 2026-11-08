#pragma once
namespace sweep_variant {
inline const void* kernel(const sweep_adapter::Params&){return (const void*)sweep_adapter::toy;}
inline const char* ptxas_key(const sweep_adapter::Params&){return "toy";}
inline size_t dynamic_smem(const sweep_adapter::Params&){return 0;}
inline cudaError_t init(const sweep_adapter::Params&){return cudaSuccess;}
inline void launch(const sweep_adapter::Params& p,cudaStream_t st){sweep_adapter::toy<<<(p.n+255)/256,256,0,st>>>(p);}
}
