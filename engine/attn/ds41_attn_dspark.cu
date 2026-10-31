// The DeepSeek-V4.1 attention TU with the DSpark device code in the same translation unit: ds41_attn.cu (kernels in ds41_attn.cuh, C entry
// points) plus ds41_attn_dsp.cuh / ds41_dspark.cuh, which reuse those kernels. One TU, so the ds41a kernels exist once in the link (engine/runtime/build_m4.sh
// builds this file as ATTN_CU for -DG6_MODEL_DS41=1). The step driver sees only runtime/ds41_dspark_api.h.
#include "ds41_attn.cu"
#include "../runtime/ds41_dspark.cuh"
