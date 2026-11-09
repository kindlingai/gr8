// Default-off Q-stationary 16-row dense indexer. The MMA and epilogue below
// are copied from k_logits_pipe<false>; only task geometry and K staging change.
// One CTA holds Q for all key tiles, skips K tiles beyond group max klen, and
// uses a three-stage cp.async ring. Skipped logits are explicitly -inf.
constexpr int PIPE_QR_RB = 16, PIPE_QR_STAGES = 3;
// Dynamic shared storage is mandatory above CUDA's 48 KiB static limit.
// Every offset is 16-byte aligned so the uint4 and cp.async destinations are aligned.
constexpr size_t PIPE_QR_A_OFF = 0;
constexpr size_t PIPE_QR_SF_OFF = PIPE_QR_A_OFF + PIPE_QR_RB * 4 * 32 * sizeof(uint4);
constexpr size_t PIPE_QR_W_OFF = PIPE_QR_SF_OFF + PIPE_QR_RB * 4 * 32 * sizeof(uint32_t);
constexpr size_t PIPE_QR_KC_OFF = PIPE_QR_W_OFF + PIPE_QR_RB * IH * sizeof(float);
constexpr size_t PIPE_QR_KE_OFF = PIPE_QR_KC_OFF + PIPE_QR_STAGES * 128 * 16 * sizeof(uint32_t);
constexpr size_t PIPE_QR_SMEM_BYTES = PIPE_QR_KE_OFF + PIPE_QR_STAGES * 128 * sizeof(uint32_t);
static_assert(sizeof(uint4) == 16 && alignof(uint4) >= 16);
static_assert((PIPE_QR_A_OFF | PIPE_QR_SF_OFF | PIPE_QR_W_OFF | PIPE_QR_KC_OFF | PIPE_QR_KE_OFF | PIPE_QR_SMEM_BYTES) % 16 == 0);
static_assert(PIPE_QR_SMEM_BYTES == 69120);
__device__ __forceinline__ void idx_cp16(void* dst, const void* src, int valid)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::
        "r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src), "r"(valid) : "memory");
}
__device__ __forceinline__ void idx_cp4(void* dst, const void* src, int valid)
{
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::
        "r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src), "r"(valid) : "memory");
}
__device__ __forceinline__ void idx_cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void idx_cp_wait(int pending)
{
    if (pending >= 3) asm volatile("cp.async.wait_group 2;\n" ::: "memory");
    else if (pending == 2) asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    else asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}
__global__ void __launch_bounds__(256) k_logits_pipe_qr(LogitsP p)
{
    extern __shared__ uint4 sraw[];
    unsigned char* sm = reinterpret_cast<unsigned char*>(sraw);
    uint4* sA = reinterpret_cast<uint4*>(sm + PIPE_QR_A_OFF);
    uint32_t* sSF = reinterpret_cast<uint32_t*>(sm + PIPE_QR_SF_OFF);
    float (*sw)[IH] = reinterpret_cast<float (*)[IH]>(sm + PIPE_QR_W_OFF);
    uint32_t (*sKC)[128][16] = reinterpret_cast<uint32_t (*)[128][16]>(sm + PIPE_QR_KC_OFF);
    uint32_t (*sKE)[128] = reinterpret_cast<uint32_t (*)[128]>(sm + PIPE_QR_KE_OFF);
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const bool math = wid < 4;
    const int warp = wid;
    const bf16 ninf = __float2bfloat16_rn(-INFINITY);
    const int groups = (p.R + PIPE_QR_RB - 1) / PIPE_QR_RB;
    for (int group = blockIdx.x; group < groups; group += gridDim.x)
    {
        const int r0 = group * PIPE_QR_RB, nr = min(PIPE_QR_RB, p.R - r0);
        // Q and weights are staged once for this 16-row group, not per K tile.
        for (int i = tid; i < nr * 4 * 32; i += 256)
        {
            const int l = i & 31, kh = (i >> 5) & 1, rt = (i >> 6) & 1, rr = i >> 7;
            const int t = l & 3, m = l >> 2;
            const uint8_t* qa = p.q_pack + ((size_t)(r0 + rr) * IH + rt * 16 + m) * IK_CODES;
            const uint8_t* qb = qa + 8 * IK_CODES;
            uint4 a;
            a.x = *reinterpret_cast<const uint32_t*>(qa + 32 * kh + 4 * t);
            a.y = *reinterpret_cast<const uint32_t*>(qb + 32 * kh + 4 * t);
            a.z = *reinterpret_cast<const uint32_t*>(qa + 32 * kh + 16 + 4 * t);
            a.w = *reinterpret_cast<const uint32_t*>(qb + 32 * kh + 16 + 4 * t);
            sA[i] = a;
            const int hs = rt * 16 + (l >> 2) + 8 * (l & 1);
            sSF[i] = *reinterpret_cast<const uint16_t*>(p.q_exp + ((size_t)(r0 + rr) * IH + hs) * IK_SF + 2 * kh);
        }
        for (int i = tid; i < nr * IH; i += 256) sw[i / IH][i % IH] = __bfloat162float(p.w[(size_t)r0 * IH + i]);
        __syncthreads();
        int kmax = 0;
        for (int rr = 0; rr < nr; rr++) kmax = max(kmax, p.klen[r0 + rr]);
        kmax = min(kmax, p.ncols);
        const int useful = (kmax + 127) / 128;
        const int prefill = min(PIPE_QR_STAGES, useful);
        const int ix = (wid - 4) * 32 + lane;
        // Prime three independent cp.async groups, each into its own ring slot.
        if (!math) for (int tile = 0; tile < prefill; tile++)
        {
            const int tok = tile * 128 + ix;
            const uint8_t* cr = tok < p.ncols ? p.k.codes.row(tok) : p.q_pack;
            const uint8_t* sf = tok < p.ncols ? p.k.sf.row(tok) : p.q_exp;
#pragma unroll
            for (int g = 0; g < 4; g++) idx_cp16(&sKC[tile][ix][4*g], cr + g * 16, tok < p.ncols ? 16 : 0);
            idx_cp4(&sKE[tile][ix], sf, tok < p.ncols ? 4 : 0);
            idx_cp_commit();
        }
        int pending = prefill;
        for (int tile = 0; tile < useful; tile++)
        {
            const int stage = tile % PIPE_QR_STAGES;
            const int base = tile * 128;
            if (!math) idx_cp_wait(pending);
            __syncthreads();
        if (math)
        {
            const int t = lane & 3, n = lane >> 2;
            const int c0 = base + warp * 32;
            uint32_t bw[4][4], sfb[4][2];
#pragma unroll
            for (int tl = 0; tl < 4; tl++)
            {
                const int tok = c0 + tl * 8 + n;
                if (tok < kmax)
                {
                    const int ix = warp * 32 + tl * 8 + n;
#pragma unroll
                    for (int g = 0; g < 4; g++) bw[tl][g] = sKC[stage][ix][4*g+t];
                    const uint32_t se = sKE[stage][ix];
                    sfb[tl][0] = se & 0xFFFFu; sfb[tl][1] = se >> 16;
                }
                else { bw[tl][0] = bw[tl][1] = bw[tl][2] = bw[tl][3] = 0u; sfb[tl][0] = sfb[tl][1] = 0x7F7Fu; }
            }
        {
#pragma unroll 1
        for (int rr = 0; rr < nr; rr++)
        {
            float ps[4][2];
#pragma unroll
            for (int tl = 0; tl < 4; tl++) ps[tl][0] = ps[tl][1] = 0.0f;
#pragma unroll
            for (int rt = 0; rt < 2; rt++)
            {
                const uint4 a0 = sA[((rr * 2 + rt) * 2 + 0) * 32 + lane], a1 = sA[((rr * 2 + rt) * 2 + 1) * 32 + lane];
                const uint32_t f0 = sSF[((rr * 2 + rt) * 2 + 0) * 32 + lane], f1 = sSF[((rr * 2 + rt) * 2 + 1) * 32 + lane];
                const float w_lo = sw[rr][rt * 16 + n], w_hi = sw[rr][rt * 16 + n + 8];
#pragma unroll
                for (int tl = 0; tl < 4; tl++)
                {
                    float acc[4] = {0.f, 0.f, 0.f, 0.f}, d1[4], d2[4];
                    mma_mxf4(d1, a0, bw[tl][0], bw[tl][1], acc, f0, sfb[tl][0]);
                    mma_mxf4(d2, a1, bw[tl][2], bw[tl][3], d1, f1, sfb[tl][1]);
#pragma unroll
                    for (int j = 0; j < 2; j++)
                    {
                        const float x0 = bf16r(d2[j]), x1 = bf16r(d2[2 + j]);
                        ps[tl][j] += bf16r((x0 > 0.0f ? x0 : 0.0f) * w_lo);
                        ps[tl][j] += bf16r((x1 > 0.0f ? x1 : 0.0f) * w_hi);
                    }
                }
            }
            const int kl = p.klen[r0 + rr];
#pragma unroll
            for (int tl = 0; tl < 4; tl++)
#pragma unroll
                for (int j = 0; j < 2; j++)
                {
                    float v = ps[tl][j];
                    v += __shfl_xor_sync(0xFFFFFFFFu, v, 4);
                    v += __shfl_xor_sync(0xFFFFFFFFu, v, 8);
                    v += __shfl_xor_sync(0xFFFFFFFFu, v, 16);
                    const int col = c0 + tl * 8 + 2 * t + j;
                    if (n == 0 && col < p.ncols) p.p[(size_t)(r0 + rr) * p.ldp + col] = col < kl ? __float2bfloat16_rn(v) : ninf;
                }
        }
        }
        }
            __syncthreads(); // all consumers released this stage before the producer reuses it
            pending--;
            const int next = tile + PIPE_QR_STAGES;
            if (!math && next < useful)
            {
                const int tok = next * 128 + ix;
                const uint8_t* cr = tok < p.ncols ? p.k.codes.row(tok) : p.q_pack;
                const uint8_t* sf = tok < p.ncols ? p.k.sf.row(tok) : p.q_exp;
#pragma unroll
                for (int g = 0; g < 4; g++) idx_cp16(&sKC[stage][ix][4*g], cr + g * 16, tok < p.ncols ? 16 : 0);
                idx_cp4(&sKE[stage][ix], sf, tok < p.ncols ? 4 : 0);
                idx_cp_commit();
            }
            if (next < useful) pending++;
        }
        // Causal tile skip must preserve the dense contract: every col < ncols
        // has a defined value, and all skipped cols are -inf for top-k.
        for (int j = tid; j < nr * (p.ncols - useful * 128); j += blockDim.x)
        {
            const int tail = p.ncols - useful * 128;
            p.p[(size_t)(r0 + j / tail) * p.ldp + useful * 128 + j % tail] = ninf;
        }
        __syncthreads();
    }
}
inline void launch_logits_pipe_qr(const LogitsP& p, cudaStream_t s)
{
    const int groups = (p.R + PIPE_QR_RB - 1) / PIPE_QR_RB;
    k_logits_pipe_qr<<<max(1, min(144, groups)), 256, PIPE_QR_SMEM_BYTES, s>>>(p);
}
