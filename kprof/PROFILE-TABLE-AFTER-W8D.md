# GLM-5.3-Flash TP2 Decode Stage Profile (`G6_DEC_EVPROF`, `--graph`, `M=1`)

- **Source run (rank 0)**: `128` steps, 8-layer `stepA` total = **7.859 ms/step**
- **Source run (rank 1)**: `128` steps, 8-layer `stepA` total = **7.860 ms/step**
- **Scaled 45-layer full-model total**: **38.395 ms/step** (vs `fs128` 45-layer wall p50 `54.94 ms/step`)

## Summary by Functional Group

| Group | Lane | 8L `stepA` (ms/step) | 8L Share | 45L Scaled (ms/step) | 45L Share |
|---|---|---:|---:|---:|---:|
| **MoE** | KA | 1.492 | 19.0% | **12.533** | **32.6%** |
| **KDA** | KB | 1.512 | 19.2% | **8.568** | **22.3%** |
| **mHC** | KB | 1.056 | 13.4% | **5.940** | **15.5%** |
| **DSA** | KB | 0.985 | 12.5% | **5.418** | **14.1%** |
| **Fabric** | KB | 0.405 | 5.2% | **2.278** | **5.9%** |
| **Head** | KB | 1.585 | 20.2% | **1.585** | **4.1%** |
| **MoE-Router** | KA/KB | 0.152 | 1.9% | **1.277** | **3.3%** |
| **DenseMLP** | KB | 0.645 | 8.2% | **0.645** | **1.7%** |
| **Other** | KB | 0.027 | 0.3% | **0.152** | **0.4%** |
| **TOTAL** | **ALL** | **7.859** | **100.0%** | **38.395** | **100.0%** |

## Per-Stage Breakdown

| Stage | Lane | Group | 8L Count | 45L Count | 8L r0 (ms) | 8L r1 (ms) | Per-Layer (us) | 45L Scaled (ms) | 45L % | Description |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `k1 fc1 (fused)` | KA | MoE | 5 | 42 | 0.897 | 0.905 | 179.4 | **7.535** | 19.6% | MoE K1 gate+up (routed + shared trellis) |
| `kda proj` | KB | KDA | 6 | 34 | 0.865 | 0.871 | 144.2 | **4.902** | 12.8% | KDA q/k/v + f_a/g_a/b projections |
| `k1 fc2` | KA | MoE | 5 | 42 | 0.486 | 0.490 | 97.2 | **4.082** | 10.6% | MoE K1 down (routed + shared trellis) |
| `reduce+norm(attn)` | KB | mHC | 8 | 45 | 0.380 | 0.381 | 47.5 | **2.138** | 5.6% | mHC FFN input mix + coeffs + collapse_norm + fragments |
| `reduce+norm(mlp)` | KB | mHC | 8 | 45 | 0.349 | 0.354 | 43.6 | **1.963** | 5.1% | mHC next-attn input mix + coeffs + collapse_norm + fragments (or final norm) |
| `kda oproj` | KB | KDA | 6 | 34 | 0.340 | 0.301 | 56.7 | **1.927** | 5.0% | KDA o_proj shard GEMV |
| `mhc out_inv` | KB | mHC | 8 | 45 | 0.327 | 0.330 | 40.9 | **1.839** | 4.8% | mHC peer reduce + post/comb stream update + flat RMS inv (2x/layer) |
| `head local` | KB | Head | 1 | 1 | 1.420 | 1.499 | 1420.0 | **1.420** | 3.7% | Final norm fragments + lm_head GEMV + local top-1/max |
| `dsa oproj` | KB | DSA | 2 | 11 | 0.189 | 0.177 | 94.5 | **1.040** | 2.7% | DSA xf_attn + o_proj shard GEMV |
| `kda prep` | KB | KDA | 6 | 34 | 0.163 | 0.162 | 27.2 | **0.924** | 2.4% | KDA conv1d + silu + f_b/g_b gates |
| `wait(mlp)` | KB | Fabric | 8 | 45 | 0.144 | 0.128 | 18.0 | **0.810** | 2.1% | Wait for peer MLP partials (2i+1) |
| `wait(attn)` | KB | Fabric | 8 | 45 | 0.140 | 0.187 | 17.5 | **0.788** | 2.1% | Wait for peer attention partials (2i) |
| `q_b+idx_q` | KB | DSA | 2 | 11 | 0.143 | 0.142 | 71.5 | **0.786** | 2.0% | DSA xf_qa + q_b + idx.wq_b |
| `moe router` | KA/KB | MoE-Router | 5 | 42 | 0.088 | 0.090 | 17.6 | **0.739** | 1.9% | MoE router GEMV (FP32 x @ BF16 router.w) |
| `attn absorb` | KB | DSA | 2 | 11 | 0.127 | 0.124 | 63.5 | **0.699** | 1.8% | DSA W_UK absorb into q_nope |
| `q_a+kv_a` | KB | DSA | 2 | 11 | 0.126 | 0.131 | 63.0 | **0.693** | 1.8% | DSA q_a + kv_a + idx_k + idx_w + idx_g + q_a_norm |
| `attn value project` | KB | DSA | 2 | 11 | 0.124 | 0.113 | 62.0 | **0.682** | 1.8% | DSA W_UV value projection |
| `kda step` | KB | KDA | 6 | 34 | 0.111 | 0.111 | 18.5 | **0.629** | 1.6% | KDA FP32 state recurrence |
| `moe plan` | KA/KB | MoE-Router | 5 | 42 | 0.064 | 0.066 | 12.8 | **0.538** | 1.4% | MoE k_router_plan top-8 + slot packing |
| `attn context` | KB | DSA | 2 | 11 | 0.093 | 0.095 | 46.5 | **0.511** | 1.3% | DSA MLA context accumulate |
| `k1 fin` | KA | MoE | 5 | 42 | 0.054 | 0.057 | 10.8 | **0.454** | 1.2% | MoE K1 finalize weighted sum -> BF16 partial |
| `dense gate` | KB | DenseMLP | 3 | 3 | 0.412 | 0.412 | 137.3 | **0.412** | 1.1% | Dense MLP L0-2 gate + up + SwiGLU |
| `o_proj+put` | KB | Fabric | 8 | 45 | 0.061 | 0.059 | 7.6 | **0.343** | 0.9% | Publish attention partial to fabric slot (2i) |
| `put(mlp)` | KB | Fabric | 8 | 45 | 0.060 | 0.060 | 7.5 | **0.337** | 0.9% | Publish MLP partial to fabric slot (2i+1) |
| `attn index scores` | KB | DSA | 2 | 11 | 0.060 | 0.058 | 30.0 | **0.330** | 0.9% | DSA indexer kpool + top-k scores |
| `attn mla scores` | KB | DSA | 2 | 11 | 0.058 | 0.062 | 29.0 | **0.319** | 0.8% | DSA MLA attention scores |
| `dense down` | KB | DenseMLP | 3 | 3 | 0.233 | 0.226 | 77.7 | **0.233** | 0.6% | Dense MLP L0-2 down projection |
| `moe xfrag` | KA | MoE | 5 | 42 | 0.023 | 0.023 | 4.6 | **0.193** | 0.5% | MoE k_xfrag B-operand fragment table |
| `kda out` | KB | KDA | 6 | 34 | 0.033 | 0.033 | 5.5 | **0.187** | 0.5% | KDA o_norm + sigmoid gate + frag/fp16 |
| `moe xhalf` | KA | MoE | 5 | 42 | 0.022 | 0.024 | 4.4 | **0.185** | 0.5% | MoE k_to_half (FP32 -> FP16) |
| `other` | KB | Other | 8 | 45 | 0.027 | 0.027 | 3.4 | **0.152** | 0.4% | Inter-stage launch / mark overhead |
| `attn append` | KB | DSA | 2 | 11 | 0.026 | 0.027 | 13.0 | **0.143** | 0.4% | DSA KV cache + kpool append |
| `head merge` | KB | Head | 1 | 1 | 0.095 | 0.015 | 95.0 | **0.095** | 0.2% | TP head merge across ranks |
| `attn normalize` | KB | DSA | 2 | 11 | 0.017 | 0.018 | 8.5 | **0.093** | 0.2% | DSA softmax normalize |
| `experts` | KA | MoE | 5 | 42 | 0.010 | 0.009 | 2.0 | **0.084** | 0.2% | MoE block tail |
| `step in/out+embed` | KB | Head | 1 | 1 | 0.070 | 0.071 | 70.0 | **0.070** | 0.2% | k_step_in + k_mhc_embed + L0 input mHC + k_step_out |
| `attn merge` | KB | DSA | 2 | 11 | 0.011 | 0.012 | 5.5 | **0.060** | 0.2% | DSA split-K merge |
| `attention` | KB | DSA | 2 | 11 | 0.009 | 0.008 | 4.5 | **0.049** | 0.1% | DSA attention tail |
| `attn select/reuse` | KB | DSA | 2 | 11 | 0.002 | 0.002 | 1.0 | **0.011** | 0.0% | DSA index page top-k select |
