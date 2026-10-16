# GLM-5.3-Flash TP2 Decode Stage Profile (`G6_DEC_EVPROF`, `--graph`, `M=1`)

- **Source run (rank 0)**: `128` steps, 8-layer `stepA` total = **13.346 ms/step**
- **Source run (rank 1)**: `128` steps, 8-layer `stepA` total = **13.343 ms/step**
- **Scaled 45-layer full-model total**: **56.963 ms/step** (vs `fs128` 45-layer wall p50 `54.94 ms/step`)

## Summary by Functional Group

| Group | Lane | 8L `stepA` (ms/step) | 8L Share | 45L Scaled (ms/step) | 45L Share |
|---|---|---:|---:|---:|---:|
| **KDA** | KB | 3.729 | 27.9% | **21.131** | **37.1%** |
| **MoE** | KA | 1.440 | 10.8% | **12.096** | **21.2%** |
| **DSA** | KB | 1.548 | 11.6% | **8.514** | **14.9%** |
| **mHC** | KB | 1.045 | 7.8% | **5.878** | **10.3%** |
| **Fabric** | KB | 0.558 | 4.2% | **3.139** | **5.5%** |
| **Head** | KB | 2.770 | 20.8% | **2.770** | **4.9%** |
| **DenseMLP** | KB | 2.088 | 15.6% | **2.088** | **3.7%** |
| **MoE-Router** | KA/KB | 0.145 | 1.1% | **1.218** | **2.1%** |
| **Other** | KB | 0.023 | 0.2% | **0.129** | **0.2%** |
| **TOTAL** | **ALL** | **13.346** | **100.0%** | **56.963** | **100.0%** |

## Per-Stage Breakdown

| Stage | Lane | Group | 8L Count | 45L Count | 8L r0 (ms) | 8L r1 (ms) | Per-Layer (us) | 45L Scaled (ms) | 45L % | Description |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `kda proj` | KB | KDA | 6 | 34 | 2.568 | 2.666 | 428.0 | **14.552** | 25.5% | KDA q/k/v + f_a/g_a/b projections |
| `k1 fc1 (fused)` | KA | MoE | 5 | 42 | 0.863 | 0.892 | 172.6 | **7.249** | 12.7% | MoE K1 gate+up (routed + shared trellis) |
| `kda oproj` | KB | KDA | 6 | 34 | 0.867 | 0.900 | 144.5 | **4.913** | 8.6% | KDA o_proj shard GEMV |
| `k1 fc2` | KA | MoE | 5 | 42 | 0.475 | 0.475 | 95.0 | **3.990** | 7.0% | MoE K1 down (routed + shared trellis) |
| `dsa oproj` | KB | DSA | 2 | 11 | 0.568 | 0.595 | 284.0 | **3.124** | 5.5% | DSA xf_attn + o_proj shard GEMV |
| `head local` | KB | Head | 1 | 1 | 2.456 | 2.696 | 2456.0 | **2.456** | 4.3% | Final norm fragments + lm_head GEMV + local top-1/max |
| `reduce+norm(attn)` | KB | mHC | 8 | 45 | 0.377 | 0.382 | 47.1 | **2.121** | 3.7% | mHC FFN input mix + coeffs + collapse_norm + fragments |
| `reduce+norm(mlp)` | KB | mHC | 8 | 45 | 0.343 | 0.349 | 42.9 | **1.929** | 3.4% | mHC next-attn input mix + coeffs + collapse_norm + fragments (or final norm) |
| `mhc out_inv` | KB | mHC | 8 | 45 | 0.325 | 0.330 | 40.6 | **1.828** | 3.2% | mHC peer reduce + post/comb stream update + flat RMS inv (2x/layer) |
| `q_b+idx_q` | KB | DSA | 2 | 11 | 0.311 | 0.327 | 155.5 | **1.710** | 3.0% | DSA xf_qa + q_b + idx.wq_b |
| `wait(attn)` | KB | Fabric | 8 | 45 | 0.284 | 0.069 | 35.5 | **1.597** | 2.8% | Wait for peer attention partials (2i) |
| `dense gate` | KB | DenseMLP | 3 | 3 | 1.426 | 1.430 | 475.3 | **1.426** | 2.5% | Dense MLP L0-2 gate + up + SwiGLU |
| `q_a+kv_a` | KB | DSA | 2 | 11 | 0.180 | 0.192 | 90.0 | **0.990** | 1.7% | DSA q_a + kv_a + idx_k + idx_w + idx_g + q_a_norm |
| `wait(mlp)` | KB | Fabric | 8 | 45 | 0.158 | 0.086 | 19.8 | **0.889** | 1.6% | Wait for peer MLP partials (2i+1) |
| `kda prep` | KB | KDA | 6 | 34 | 0.151 | 0.155 | 25.2 | **0.856** | 1.5% | KDA conv1d + silu + f_b/g_b gates |
| `moe router` | KA/KB | MoE-Router | 5 | 42 | 0.083 | 0.086 | 16.6 | **0.697** | 1.2% | MoE router GEMV (FP32 x @ BF16 router.w) |
| `dense down` | KB | DenseMLP | 3 | 3 | 0.650 | 0.678 | 216.7 | **0.650** | 1.1% | Dense MLP L0-2 down projection |
| `attn absorb` | KB | DSA | 2 | 11 | 0.113 | 0.111 | 56.5 | **0.622** | 1.1% | DSA W_UK absorb into q_nope |
| `kda step` | KB | KDA | 6 | 34 | 0.106 | 0.107 | 17.7 | **0.601** | 1.1% | KDA FP32 state recurrence |
| `attn value project` | KB | DSA | 2 | 11 | 0.099 | 0.103 | 49.5 | **0.544** | 1.0% | DSA W_UV value projection |
| `moe plan` | KA/KB | MoE-Router | 5 | 42 | 0.062 | 0.062 | 12.4 | **0.521** | 0.9% | MoE k_router_plan top-8 + slot packing |
| `attn context` | KB | DSA | 2 | 11 | 0.093 | 0.092 | 46.5 | **0.511** | 0.9% | DSA MLA context accumulate |
| `k1 fin` | KA | MoE | 5 | 42 | 0.052 | 0.053 | 10.4 | **0.437** | 0.8% | MoE K1 finalize weighted sum -> BF16 partial |
| `put(mlp)` | KB | Fabric | 8 | 45 | 0.059 | 0.059 | 7.4 | **0.332** | 0.6% | Publish MLP partial to fabric slot (2i+1) |
| `attn index scores` | KB | DSA | 2 | 11 | 0.059 | 0.059 | 29.5 | **0.325** | 0.6% | DSA indexer kpool + top-k scores |
| `o_proj+put` | KB | Fabric | 8 | 45 | 0.057 | 0.058 | 7.1 | **0.321** | 0.6% | Publish attention partial to fabric slot (2i) |
| `attn mla scores` | KB | DSA | 2 | 11 | 0.058 | 0.058 | 29.0 | **0.319** | 0.6% | DSA MLA attention scores |
| `head merge` | KB | Head | 1 | 1 | 0.247 | 0.008 | 247.0 | **0.247** | 0.4% | TP head merge across ranks |
| `kda out` | KB | KDA | 6 | 34 | 0.037 | 0.038 | 6.2 | **0.210** | 0.4% | KDA o_norm + sigmoid gate + frag/fp16 |
| `moe xhalf` | KA | MoE | 5 | 42 | 0.021 | 0.021 | 4.2 | **0.176** | 0.3% | MoE k_to_half (FP32 -> FP16) |
| `moe xfrag` | KA | MoE | 5 | 42 | 0.021 | 0.022 | 4.2 | **0.176** | 0.3% | MoE k_xfrag B-operand fragment table |
| `attn append` | KB | DSA | 2 | 11 | 0.028 | 0.029 | 14.0 | **0.154** | 0.3% | DSA KV cache + kpool append |
| `other` | KB | Other | 8 | 45 | 0.023 | 0.024 | 2.9 | **0.129** | 0.2% | Inter-stage launch / mark overhead |
| `attn normalize` | KB | DSA | 2 | 11 | 0.017 | 0.017 | 8.5 | **0.093** | 0.2% | DSA softmax normalize |
| `experts` | KA | MoE | 5 | 42 | 0.008 | 0.008 | 1.6 | **0.067** | 0.1% | MoE block tail |
| `step in/out+embed` | KB | Head | 1 | 1 | 0.067 | 0.070 | 67.0 | **0.067** | 0.1% | k_step_in + k_mhc_embed + L0 input mHC + k_step_out |
| `attn merge` | KB | DSA | 2 | 11 | 0.012 | 0.011 | 6.0 | **0.066** | 0.1% | DSA split-K merge |
| `attention` | KB | DSA | 2 | 11 | 0.008 | 0.008 | 4.0 | **0.044** | 0.1% | DSA attention tail |
| `dense out` | KB | DenseMLP | 3 | 3 | 0.012 | 0.012 | 4.0 | **0.012** | 0.0% | Dense MLP L0-2 BF16 cast |
| `attn select/reuse` | KB | DSA | 2 | 11 | 0.002 | 0.002 | 1.0 | **0.011** | 0.0% | DSA index page top-k select |
