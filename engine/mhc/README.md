# Flash mHC

Four-stream FP32 residuals, a BF16 24-by-16384 projection with FP32
accumulation, FP32 gates and 20 Sinkhorn passes. `comb[source][destination]`
is deliberate. The first pass is row softmax plus epsilon, then column
normalization; only the remaining 19 passes do row and column division.

`mhc.cuh` provides embedding replication, coefficient computation, collapse
and RMSNorm, rank-ordered BF16 partial consumption, residual update, and final
stream mean plus RMSNorm. Consume also computes the next flat-RMS inverse.
No kernel narrows the residual or the coefficient state. The block boundary
has FP32 normalized output plus an optional FP16 copy. Fragment tables round
FP32 directly to FP16 or BF16, not FP32 -> FP16 -> BF16.

The host allocates `Buffers` once. After `k_mhc_embed` or
`k_mhc_flat_inv`, call `prepare`, execute the block, then
`k_mhc_out_inv`. Repeat for the next block. All calls accept a CUDA stream;
there is no allocation, host transfer, or synchronization inside a launch.
`k_mhc_final` applies the final head norm to the stream mean.

## Gates

```sh
python engine/mhc/test_mhc.py
nvcc -O2 -std=c++17 -arch=sm_121a -DG6_MODEL_FLASH=1 \
  engine/mhc/test_mhc.cu -o /tmp/mhc_test
python engine/mhc/test_mhc.py --gpu /tmp/mhc_test
python engine/mhc/test_mhc.py --dense DENSE_EXPORT \
  --golden GOLDENS/prompt-bf16.safetensors \
  --golden GOLDENS/step-bf16.safetensors --gpu /tmp/mhc_test
```

Python needs PyTorch, safetensors and the W0 transformers environment.
Without `--gpu`, the test compiles `cpu.h` into a temporary shared library.
It checks tiny and H4096 inputs, zero streams, saturated gates, a non-symmetric
combination matrix, and layers 0-3. Real golden tests load only the needed
mHC tensors from lossless dense safetensors. M=1 and M=8 use the same inputs
and weights as W0. Per-op relative-L2 gate: 1e-4. The tiny CPU chain gate is
2e-2; attention and MLP use W0 in that test. It is not a CUDA layer-chain gate.

## Measured correctness

The sm_121a test passed 196 GPU per-op comparisons, including real H4096
prompt/decode goldens for layers 0-3 with M=1 and M=8. Maximum relative-L2:
3.42e-6 (gate 1e-4). The same executable also checked embedding replication,
rank-ordered BF16 partial consumption, the next flat-RMS inverse, direct
FP32-to-FP16/BF16 fragment conversion, and zero padding.

`--replay-only` runs the eight consecutive mHC blocks in layers 0-3 with W0
attention/MLP outputs replayed between them. It carries the computed streams
into the next block. This is a residual-composition test, not an integrated
W3-W5 layer-chain test. The latter still needs those blocks and their state.

The CPU host oracle passed 181 checks (random, real-weight and tiny chain),
maximum per-op relative-L2 5.1e-5. No performance result is claimed: the GPU
was shared with another workload during these correctness tests.
