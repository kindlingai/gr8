// ink2 head: one rank's share of a row's head, packed on the device.
//
// The wire row is the one ink2rt::head_pack writes on the host:
//   [max f32][count i32][sum f64][NC x {f32 logit, i32 id}]
// Only finite logits are candidates. count = min(finite, NC). The candidates are ordered by logit, largest
// first, and by id among equal logits; the slots past count are zero. max and the candidates are bit-equal to the
// host's; sum is a double sum of exp(logit - max) in a fixed order of its own, so it can differ from the host's in
// the last place.
//
// One CTA of 256 threads per row: max and count, the sum, a radix select (4 passes of 8 bits) for the key of the
// NC-th largest logit, a compaction that keeps id order among the logits equal to it, and a rank sort of the
// at most INK2_HEAD_MAX_NC survivors in shared memory.
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace ink2
{
constexpr int INK2_HEAD_MAX_NC = 512;

// An order-preserving key of a finite float (0 for inf / NaN; both zeros share one key).
__device__ __forceinline__ uint32_t head_key(float x)
{
    const uint32_t u = __float_as_uint(x);
    if ((u & 0x7f800000u) == 0x7f800000u) return 0u;
    if ((u << 1) == 0u) return 0x80000000u;
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__global__ void k_ink2_head_pack(const float* __restrict__ lg, int vr, int v0, int NC, uint8_t* __restrict__ out, size_t rb)
{
    __shared__ uint32_t hist[256];
    __shared__ double dsum[256];
    __shared__ float fmx[256];
    __shared__ int cg[256], ce[256];
    __shared__ float cl[INK2_HEAD_MAX_NC];
    __shared__ int cid[INK2_HEAD_MAX_NC];
    __shared__ uint32_t s_prefix;
    __shared__ int s_rem;
    const int t = threadIdx.x;
    const float* x = lg + static_cast<size_t>(blockIdx.x) * vr;
    uint8_t* o = out + static_cast<size_t>(blockIdx.x) * rb;

    // max and count of the finite logits
    {
        float m = -INFINITY;
        int cnt = 0;
        for (int j = t; j < vr; j += 256)
        {
            const float v = x[j];
            if (head_key(v))
            {
                ++cnt;
                if (v > m) m = v;
            }
        }
        fmx[t] = m;
        cg[t] = cnt;
    }
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1)
    {
        if (t < s)
        {
            if (fmx[t + s] > fmx[t]) fmx[t] = fmx[t + s];
            cg[t] += cg[t + s];
        }
        __syncthreads();
    }
    const float mx = fmx[0];
    const int finite = cg[0];
    const int n = finite < NC ? finite : NC;

    // sum of exp(logit - max), in doubles
    {
        double a = 0.0;
        for (int j = t; j < vr; j += 256)
        {
            const float v = x[j];
            if (head_key(v)) a += exp(static_cast<double>(v - mx));
        }
        dsum[t] = a;
    }
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1)
    {
        if (t < s) dsum[t] += dsum[t + s];
        __syncthreads();
    }
    const double sum = dsum[0];

    if (n > 0)
    {
        // the key of the n-th largest logit, and how many logits of that key are in
        uint32_t prefix = 0, mask = 0;
        int rem = n;
        for (int sh = 24; sh >= 0; sh -= 8)
        {
            hist[t] = 0;
            __syncthreads();
            for (int j = t; j < vr; j += 256)
            {
                const uint32_t k = head_key(x[j]);
                if (k && (k & mask) == prefix) atomicAdd(&hist[(k >> sh) & 255u], 1u);
            }
            __syncthreads();
            if (t == 0)
            {
                int acc = 0, d = 255;
                for (; d > 0; --d)
                {
                    const int c = static_cast<int>(hist[d]);
                    if (acc + c >= rem) break;
                    acc += c;
                }
                s_prefix = prefix | (static_cast<uint32_t>(d) << sh);
                s_rem = rem - acc;
            }
            __syncthreads();
            prefix = s_prefix;
            rem = s_rem;
            mask |= 255u << sh;
            __syncthreads();
        }
        const uint32_t thr = prefix;
        const int take = rem, g = n - rem;

        // compaction: each thread owns a run of ids, so the logits equal to the threshold keep id order
        const int ch = (vr + 255) / 256;
        const int j0 = t * ch < vr ? t * ch : vr, j1 = j0 + ch < vr ? j0 + ch : vr;
        {
            int a = 0, b = 0;
            for (int j = j0; j < j1; ++j)
            {
                const uint32_t k = head_key(x[j]);
                a += k > thr;
                b += k == thr;
            }
            cg[t] = a;
            ce[t] = b;
        }
        __syncthreads();
        if (t == 0)
        {
            int a = 0, b = 0;
            for (int i = 0; i < 256; ++i)
            {
                const int ca = cg[i], cb = ce[i];
                cg[i] = a;
                ce[i] = b;
                a += ca;
                b += cb;
            }
        }
        __syncthreads();
        {
            int pg = cg[t], pe = ce[t];
            for (int j = j0; j < j1; ++j)
            {
                const float v = x[j];
                const uint32_t k = head_key(v);
                if (k > thr)
                {
                    cl[pg] = v;
                    cid[pg] = v0 + j;
                    ++pg;
                }
                else if (k == thr)
                {
                    if (pe < take)
                    {
                        cl[g + pe] = v;
                        cid[g + pe] = v0 + j;
                    }
                    ++pe;
                }
            }
        }
        __syncthreads();
        // rank sort (ids are distinct, so the ranks are)
        for (int i = t; i < n; i += 256)
        {
            const float li = cl[i];
            const int ii = cid[i];
            int r = 0;
            for (int j = 0; j < n; ++j) r += (cl[j] > li) || (cl[j] == li && cid[j] < ii);
            *reinterpret_cast<float*>(o + 16 + static_cast<size_t>(r) * 8) = li;
            *reinterpret_cast<int*>(o + 20 + static_cast<size_t>(r) * 8) = ii;
        }
    }
    for (int i = n + t; i < NC; i += 256) *reinterpret_cast<uint64_t*>(o + 16 + static_cast<size_t>(i) * 8) = 0ull;
    if (t == 0)
    {
        *reinterpret_cast<float*>(o) = mx;
        *reinterpret_cast<int*>(o + 4) = n;
        *reinterpret_cast<double*>(o + 8) = sum;
    }
}

// rows of logits [rows][vr] -> wire rows [rows][rb] on the device. NC <= INK2_HEAD_MAX_NC, rb = 16 + 8 * NC.
inline void head_pack_dev(const float* lg, int rows, int vr, int v0, int NC, uint8_t* out, size_t rb, cudaStream_t st)
{
    if (rows <= 0) return;
    k_ink2_head_pack<<<rows, 256, 0, st>>>(lg, vr, v0, NC, out, rb);
}
}   // namespace ink2
