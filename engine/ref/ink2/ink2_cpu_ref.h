// CPU golden reference of the Inkling-2-Small forward pass. Our own code, C++17, no CUDA. It is written for clarity and
// a fixed fp32 order, not speed: build with -ffp-contract=off and compare a device against it on a small model.
//
// Conventions
//   * A "BF16" activation is held as a float whose value is BF16 representable (rbf() = round to nearest even, once).
//   * Weights are plain float vectors, row major; a BF16 weight is its exact float value.
//   * A dot product is one fp32 sum in index order. Where a sum is split over TP ranks (wo over the heads of a rank,
//     fc2 over the intermediate slice of a rank) the partial sums are added in rank order; Config::tp = 1 is one sum.
//   * State is kept as the whole history (the inputs of every short convolution and the K / V of every position), so
//     the device's rings and caches are the last rows of it and rollback(pos) is a truncation.
//   * forward() runs layer by layer over the M new tokens. Any split of a sequence into forward() calls gives the same
//     bits (engine/tests/test_ink2_cpu_ref.cpp).
//
// The layer, for token position p and layer l (B = Config::block, the layers per block of the depth attention):
//   resid = h_in
//   attention input: l == 0 -> h_in; else AttnRes(states, resid, attn_query[l]) (1 query on a local layer; 3 on a
//                    global layer, one each for q, k and v); then attn_norm, or q / k / v_input_norm on a global layer
//   a     = attn_sconv(attention(q_in, k_in, v_in))
//   l % B == 0: states += resid, res_mid = a;   else res_mid = bf16(resid + a)
//   m     = mlp_sconv(moe(mlp_norm(AttnRes(states, res_mid, mlp_query[l]))))
//   h_out = bf16(res_mid + m)
// and logits = softcap30(lm_head . final_norm(AttnRes(states, h_last, output_query))).
#pragma once
#include "../../kernels/nvfp4_pack.h"
#include <cfloat>
#include <map>

namespace ink2ref
{
constexpr float EPS = 1e-6f;

inline float bf16_to_f32(uint16_t h)
{
    const uint32_t u = static_cast<uint32_t>(h) << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

// fp32 -> BF16 bits, round to nearest even; inf and NaN keep their top 16 bits.
inline uint16_t f32_to_bf16(float f)
{
    uint32_t u;
    memcpy(&u, &f, 4);
    if ((u & 0x7f800000u) != 0x7f800000u) u += 0x7fffu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}

inline float rbf(float f) { return bf16_to_f32(f32_to_bf16(f)); }
inline float sigmoidf(float z) { return 1.0f / (1.0f + std::exp(-z)); }
inline float rsqrtf(float x) { return 1.0f / std::sqrt(x); }

inline float dot(const float* a, const float* b, int n)
{
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

inline float mean_sq(const float* x, int n) { return dot(x, x, n) / static_cast<float>(n); }
inline float l2_norm(const float* x, int n) { return std::sqrt(dot(x, x, n)); }

// ---- MXFP8: E4M3 codes, one E8M0 scale per 32, and the 528-byte decode record (two 16x16 tiles in lane order, then
// the 8 scale pairs {s[i], s[i + 8]}); the twin of mxfp8_quantize / mxfp8_fc1 / mxfp8_fc2 in engine/loader-rs.
namespace mx8
{
constexpr int GR = 528;

inline float scale_f32(uint8_t s) { return std::ldexp(1.0f, static_cast<int>(s) - 127); }

// nblk blocks of 32: s = clamp(ceil(log2(fl32(amax / 448))) + 127, 0, 254), code = e4m3(x * 2^(127 - s)), RNE.
inline void quantize(const float* x, size_t nblk, uint8_t* codes, uint8_t* scales)
{
    for (size_t b = 0; b < nblk; ++b)
    {
        float amax = FLT_MIN;
        for (int j = 0; j < 32; ++j) amax = std::fmax(amax, std::fabs(x[b * 32 + j]));
        int e;
        const double fr = std::frexp(static_cast<double>(amax / 448.0f), &e);   // ratio = fr * 2^e, fr in [.5, 1)
        int s = (fr == 0.5 ? e - 1 : e) + 127;
        s = s < 0 ? 0 : s > 254 ? 254 : s;
        scales[b] = static_cast<uint8_t>(s);
        const double mul = std::ldexp(1.0, 127 - s);
        for (int j = 0; j < 32; ++j)
            codes[b * 32 + j] = nv4pack::f32_to_e4m3_rne(static_cast<float>(static_cast<double>(x[b * 32 + j]) * mul));
    }
}

inline void dequantize(const uint8_t* codes, const uint8_t* scales, size_t nblk, float* out)
{
    for (size_t b = 0; b < nblk; ++b)
        for (int j = 0; j < 32; ++j) out[b * 32 + j] = nv4pack::e4m3_to_f32(codes[b * 32 + j]) * scale_f32(scales[b]);
}

// Element i (0 .. 256) of a 16x16 tile in record order: lane = i / 8 = (r, c), r = lane / 4, c = lane % 4, and its
// eight elements (r, 2c), (r, 2c+1), (r+8, 2c), (r+8, 2c+1), (r, 2c+8), (r, 2c+9), (r+8, 2c+8), (r+8, 2c+9).
inline void lane_rc(int i, int& row, int& col)
{
    const int lane = i >> 3, j = i & 7, r = lane >> 2, c = lane & 3;
    row = r + ((j >> 1) & 1) * 8;
    col = 2 * c + (j & 1) + (j >> 2) * 8;
}

// One record: rows r0 .. r0 + 16, codes c0 .. c0 + 32, scale column sg. cs / ss = code / scale bytes per source row.
inline void pack_group(uint8_t* rec, const uint8_t* codes, const uint8_t* scales, size_t cs, size_t ss, int r0, int c0, int sg)
{
    for (int h = 0; h < 2; ++h)
        for (int i = 0; i < 256; ++i)
        {
            int r, c;
            lane_rc(i, r, c);
            rec[h * 256 + i] = codes[static_cast<size_t>(r0 + r) * cs + c0 + 16 * h + c];
        }
    for (int i = 0; i < 8; ++i)
    {
        rec[512 + 2 * i] = scales[static_cast<size_t>(r0 + i) * ss + sg];
        rec[512 + 2 * i + 1] = scales[static_cast<size_t>(r0 + i + 8) * ss + sg];
    }
}

// gate and up [R][K] -> [R / 16][K / 32][gate GR | up GR]
inline std::vector<uint8_t> pack_fc1(const uint8_t* gc, const uint8_t* gs, const uint8_t* uc, const uint8_t* us, int R, int K)
{
    if (R % 16 || K % 32) throw std::runtime_error("mx8: R % 16 or K % 32");
    const int ng = K / 32;
    std::vector<uint8_t> out(static_cast<size_t>(R / 16) * ng * 2 * GR);
    for (int ch = 0; ch < R / 16; ++ch)
        for (int g = 0; g < ng; ++g)
        {
            uint8_t* d = out.data() + (static_cast<size_t>(ch) * ng + g) * 2 * GR;
            pack_group(d, gc, gs, K, ng, ch * 16, g * 32, g);
            pack_group(d + GR, uc, us, K, ng, ch * 16, g * 32, g);
        }
    return out;
}

// Columns [c0, c0 + Kn) of a [R][K] matrix -> [R / 16][Kn / 32][GR].
inline std::vector<uint8_t> pack_fc2(const uint8_t* c, const uint8_t* s, int R, int K, int c0, int Kn)
{
    if (R % 16 || K % 32 || c0 % 32 || Kn % 32 || c0 + Kn > K) throw std::runtime_error("mx8: R % 16, or the column slice");
    const int ng = Kn / 32;
    std::vector<uint8_t> out(static_cast<size_t>(R / 16) * ng * GR);
    for (int ot = 0; ot < R / 16; ++ot)
        for (int g = 0; g < ng; ++g)
            pack_group(out.data() + (static_cast<size_t>(ot) * ng + g) * GR, c, s, K, K / 32, ot * 16, c0 + g * 32, c0 / 32 + g);
    return out;
}

inline void decode_group(const uint8_t* rec, float (&out)[16][32])
{
    static const std::vector<float> e4m3 = [] {
        std::vector<float> t(256);
        for (int i = 0; i < 256; ++i) t[i] = nv4pack::e4m3_to_f32(static_cast<uint8_t>(i));
        return t;
    }();
    for (int h = 0; h < 2; ++h)
        for (int i = 0; i < 256; ++i)
        {
            int r, c;
            lane_rc(i, r, c);
            const uint8_t s = r < 8 ? rec[512 + 2 * r] : rec[512 + 2 * (r - 8) + 1];
            out[r][16 * h + c] = e4m3[rec[h * 256 + i]] * scale_f32(s);
        }
}
}   // namespace mx8

// ---- the model
struct Config
{
    int hidden = 0, n_layers = 0;
    int block = 6;                 // layers per block of the depth attention (n_layers_per_attn_res_block)
    int n_heads = 0, n_kv_heads = 0, head_dim = 0;
    int rel_dim = 16, rel_win = 0;
    int sliding_window = 0;        // local layers attend to q_pos - k_pos < sliding_window
    int log_scale_base = 128000;   // global layers: tau(pos) = 1 + 0.1 ln(max(1, (pos + 1) / base))
    int inter = 0, n_exp = 0, top_k = 0;
    float route_scale = 8.0f;
    int vocab = 0, valid_vocab = 0, image_token_id = -1;
    int tp = 1;                    // ranks the sharded sums are split over
    std::vector<uint8_t> global;   // [n_layers] 1 = full attention with the dynamic K / V convolution
    std::vector<uint8_t> mxfp8;    // [n_layers] 1 = MXFP8 experts, 0 = NVFP4
};

struct Layer
{
    std::vector<float> attn_query;                           // [Q][H], Q = 3 on a global layer, else 1; unused at l = 0
    std::vector<float> mlp_query;                            // [H]
    std::vector<float> attn_norm;                            // [H]  local layers, and layer 0
    std::vector<float> q_norm, k_norm, v_norm;               // [H]  global layers
    std::vector<float> wq, gate;                             // [nh * hd][H]
    std::vector<float> wr;                                   // [nh * rel_dim][H]
    std::vector<float> sink;                                 // [nh][H]
    std::vector<float> wk, wv;                               // [nkv * hd][H]
    std::vector<float> wo;                                   // [H][nh * hd]
    std::vector<float> q_clip;                               // [nh][hd]
    std::vector<float> rel_clip;                             // [nh][rel_dim]
    std::vector<float> rel_proj;                             // [rel_dim][rel_win]
    std::vector<float> k_sconv, v_sconv;                     // local: [nkv * hd][4], tap 3 = the current token
    std::vector<float> k_static, v_static;                   // global: [4][nkv * hd], row = lag
    std::vector<float> k_wproj, v_wproj;                     // global: [nkv * 4][H], row = head * 4 + lag
    std::vector<float> attn_sconv, mlp_sconv;                // [H][4]
    std::vector<float> mlp_norm;                             // [H]
    std::vector<float> router_w, router_bias;                // [E][H], [E]
    // experts: NVFP4 fc1 [E][I/16][H/64][gate 576 | up 576], fc2 [E][H/16][I/64][576], outer scales s1, s2 [E];
    //          MXFP8 fc1 [E][I/16][H/32][gate 528 | up 528], fc2 [E][H/16][I/32][528]
    std::vector<uint8_t> fc1, fc2;
    std::vector<float> s1, s2;
};

struct Weights
{
    std::vector<float> embed, embed_norm, final_norm, output_query, lm_head;   // [V][H], [H], [H], [H], [V][H]
    std::vector<Layer> layers;
};

class Model
{
public:
    Model(const Config& c, const Weights& w) : C(c), W(w), st(static_cast<size_t>(c.n_layers))
    {
        const auto bad = [](const char* m) { throw std::runtime_error(std::string("ink2ref: ") + m); };
        if (C.n_layers < 1 || static_cast<int>(W.layers.size()) != C.n_layers) bad("layer count");
        if (static_cast<int>(C.global.size()) != C.n_layers || static_cast<int>(C.mxfp8.size()) != C.n_layers) bad("per-layer lists");
        if (C.global[0]) bad("layer 0 must be a local layer");
        if (C.n_heads % C.n_kv_heads || C.tp < 1 || C.n_heads % C.tp || C.inter % (64 * C.tp) || C.hidden % 64) bad("geometry");
        if (C.top_k < 1 || C.top_k > C.n_exp || C.block < 1) bad("top_k or block");
    }

    int pos() const { return pos_; }

    // Drop every position >= p: the convolution inputs, the K / V rows and the position counter.
    void rollback(int p)
    {
        if (p < 0 || p > pos_) throw std::runtime_error("ink2ref: rollback position");
        const size_t H = static_cast<size_t>(C.hidden), nk = static_cast<size_t>(C.n_kv_heads) * C.head_dim;
        for (State& s : st)
        {
            s.hist_a.resize(p * H);
            s.hist_m.resize(p * H);
            s.hist_k.resize(p * nk);
            s.hist_v.resize(p * nk);
            s.K.resize(p * nk);
            s.V.resize(p * nk);
        }
        pos_ = p;
    }

    // M tokens at positions pos() .. pos() + M. vis = [M][H] BF16 feature rows, read where toks[m] == image_token_id
    // (the row replaces the embedding after embed_norm). Returns the logits [M][vocab].
    std::vector<float> forward(const int* toks, int M, const float* vis = nullptr)
    {
        const int H = C.hidden;
        std::vector<float> h(static_cast<size_t>(M) * H);
        for (int m = 0; m < M; ++m)
        {
            if (toks[m] == C.image_token_id && C.image_token_id >= 0)
            {
                if (!vis) throw std::runtime_error("ink2ref: an image token without its feature row");
                for (int c = 0; c < H; ++c) h[static_cast<size_t>(m) * H + c] = rbf(vis[static_cast<size_t>(m) * H + c]);
            }
            else
            {
                if (toks[m] < 0 || toks[m] >= C.vocab) throw std::runtime_error("ink2ref: token id");
                wnorm(&W.embed[static_cast<size_t>(toks[m]) * H], W.embed_norm, &h[static_cast<size_t>(m) * H]);
            }
        }
        std::vector<std::vector<float>> states;   // [blocks done][M][H]
        for (int l = 0; l < C.n_layers; ++l)
        {
            const Layer& w = W.layers[l];
            const bool glob = C.global[l] != 0;
            std::vector<float> qi(h.size()), ki, vi;
            if (l == 0)
                for (int m = 0; m < M; ++m) wnorm(&h[static_cast<size_t>(m) * H], w.attn_norm, &qi[static_cast<size_t>(m) * H]);
            else if (!glob)
            {
                const std::vector<float> x = attn_res(states, h, w.attn_query, 1, M);
                for (int m = 0; m < M; ++m) wnorm(&x[static_cast<size_t>(m) * H], w.attn_norm, &qi[static_cast<size_t>(m) * H]);
            }
            else
            {
                const std::vector<float> x = attn_res(states, h, w.attn_query, 3, M);
                ki.resize(h.size());
                vi.resize(h.size());
                for (int m = 0; m < M; ++m)
                {
                    const float* xm = &x[static_cast<size_t>(m) * 3 * H];
                    wnorm(xm, w.q_norm, &qi[static_cast<size_t>(m) * H]);
                    wnorm(xm + H, w.k_norm, &ki[static_cast<size_t>(m) * H]);
                    wnorm(xm + 2 * H, w.v_norm, &vi[static_cast<size_t>(m) * H]);
                }
            }
            const std::vector<float> att = attention(l, qi, glob ? ki : qi, glob ? vi : qi, M);
            const std::vector<float> a = sconv_static(w.attn_sconv, st[l].hist_a, att, M, H);
            std::vector<float> mid(h.size());
            if (l % C.block == 0)
            {
                states.push_back(h);
                mid = a;
            }
            else
                for (size_t i = 0; i < h.size(); ++i) mid[i] = rbf(h[i] + a[i]);
            const std::vector<float> xm = attn_res(states, mid, w.mlp_query, 1, M);
            std::vector<float> mo(h.size()), xn(static_cast<size_t>(H));
            for (int m = 0; m < M; ++m)
            {
                wnorm(&xm[static_cast<size_t>(m) * H], w.mlp_norm, xn.data());
                moe(l, xn.data(), &mo[static_cast<size_t>(m) * H]);
            }
            const std::vector<float> ms = sconv_static(w.mlp_sconv, st[l].hist_m, mo, M, H);
            for (size_t i = 0; i < h.size(); ++i) h[i] = rbf(mid[i] + ms[i]);
        }
        const std::vector<float> xf = attn_res(states, h, W.output_query, 1, M);
        std::vector<float> logits(static_cast<size_t>(M) * C.vocab), hf(static_cast<size_t>(H));
        for (int m = 0; m < M; ++m)
        {
            wnorm(&xf[static_cast<size_t>(m) * H], W.final_norm, hf.data());
            for (int v = 0; v < C.vocab; ++v)
            {
                const float z = dot(hf.data(), &W.lm_head[static_cast<size_t>(v) * H], H);
                logits[static_cast<size_t>(m) * C.vocab + v] = v < C.valid_vocab ? 30.0f * std::tanh(z / 30.0f) : -std::numeric_limits<float>::infinity();
            }
        }
        pos_ += M;
        return logits;
    }

    // ---- the pieces, public so a test can call one alone

    // Weighted RMSNorm: bf16(w * (x * rsqrt(mean(x^2) + eps))), the product with w in fp32.
    void wnorm(const float* x, const std::vector<float>& w, float* out) const
    {
        const float r = rsqrtf(mean_sq(x, C.hidden) + EPS);
        for (int c = 0; c < C.hidden; ++c) out[c] = rbf(w[c] * (x[c] * r));
    }

    // Depth attention over the finished blocks and the partial residual, per token: rows = states..., partial, each
    // RMS normalised (no weight); logit = query . row / sqrt(H); out = bf16(sum softmax * row). Returns [M][Q][H].
    std::vector<float> attn_res(const std::vector<std::vector<float>>& states, const std::vector<float>& partial,
                                const std::vector<float>& query, int Q, int M) const
    {
        const int H = C.hidden, D = static_cast<int>(states.size()) + 1;
        std::vector<float> out(static_cast<size_t>(M) * Q * H), U(static_cast<size_t>(D) * H), lg(static_cast<size_t>(D));
        for (int m = 0; m < M; ++m)
        {
            for (int d = 0; d < D; ++d)
            {
                const float* s = (d + 1 < D ? states[d].data() : partial.data()) + static_cast<size_t>(m) * H;
                const float r = rsqrtf(mean_sq(s, H) + EPS);
                for (int c = 0; c < H; ++c) U[static_cast<size_t>(d) * H + c] = s[c] * r;
            }
            for (int q = 0; q < Q; ++q)
            {
                float mx = -std::numeric_limits<float>::infinity(), den = 0.0f;
                for (int d = 0; d < D; ++d)
                {
                    lg[d] = dot(&query[static_cast<size_t>(q) * H], &U[static_cast<size_t>(d) * H], H) / std::sqrt(static_cast<float>(H));
                    mx = std::fmax(mx, lg[d]);
                }
                for (int d = 0; d < D; ++d)
                {
                    lg[d] = std::exp(lg[d] - mx);
                    den += lg[d];
                }
                float* o = &out[(static_cast<size_t>(m) * Q + q) * H];
                for (int c = 0; c < H; ++c)
                {
                    float s = 0.0f;
                    for (int d = 0; d < D; ++d) s += (lg[d] / den) * U[static_cast<size_t>(d) * H + c];
                    o[c] = rbf(s);
                }
            }
        }
        return out;
    }

    // Static causal convolution over 4 tokens with its residual: w [C][4], w[c][3 - lag] on token t - lag;
    // y = bf16((x[t] + sum_lag w * x[t - lag]) / sqrt(5)). hist [positions][C] gets the M new rows.
    static std::vector<float> sconv_static(const std::vector<float>& w, std::vector<float>& hist, const std::vector<float>& x, int M, int Cn)
    {
        const size_t t0 = hist.size() / Cn;
        hist.insert(hist.end(), x.begin(), x.end());
        std::vector<float> y(x.size());
        for (int m = 0; m < M; ++m)
            for (int c = 0; c < Cn; ++c)
            {
                const size_t t = t0 + m;
                float s = 0.0f;
                for (size_t lag = 0; lag < 4 && lag <= t; ++lag) s += w[static_cast<size_t>(c) * 4 + 3 - lag] * hist[(t - lag) * Cn + c];
                y[static_cast<size_t>(m) * Cn + c] = rbf((hist[t * Cn + c] + s) / std::sqrt(5.0f));
            }
        return y;
    }

    // The K / V convolution of a global layer: tap(lag) = static_w[lag][c] + (weight_proj . norm input)[head][lag],
    // lag 0 = the current token; y = bf16((x[t] + sum_lag tap * x[t - lag]) / 3).
    std::vector<float> sconv_dynamic(const std::vector<float>& stat, const std::vector<float>& wproj, std::vector<float>& hist,
                                     const std::vector<float>& x, const std::vector<float>& in, int M) const
    {
        const int H = C.hidden, hd = C.head_dim, nk = C.n_kv_heads * hd;
        const size_t t0 = hist.size() / nk;
        hist.insert(hist.end(), x.begin(), x.end());
        std::vector<float> y(x.size());
        for (int m = 0; m < M; ++m)
            for (int h = 0; h < C.n_kv_heads; ++h)
            {
                float dyn[4];
                for (int lag = 0; lag < 4; ++lag) dyn[lag] = dot(&in[static_cast<size_t>(m) * H], &wproj[static_cast<size_t>(h * 4 + lag) * H], H);
                for (int d = 0; d < hd; ++d)
                {
                    const size_t t = t0 + m, c = static_cast<size_t>(h) * hd + d;
                    float s = 0.0f;
                    for (size_t lag = 0; lag < 4 && lag <= t; ++lag) s += (stat[lag * nk + c] + dyn[lag]) * hist[(t - lag) * nk + c];
                    y[static_cast<size_t>(m) * nk + c] = rbf((hist[t * nk + c] + s) / 3.0f);
                }
            }
        return y;
    }

    float tau(int p) const { return 1.0f + 0.1f * std::log(std::fmax(1.0f, static_cast<float>(p + 1) / static_cast<float>(C.log_scale_base))); }

    // qi, ki, vi [M][H] -> wo(gated attention output) [M][H], BF16; appends the M tokens' K / V rows.
    std::vector<float> attention(int l, const std::vector<float>& qi, const std::vector<float>& ki, const std::vector<float>& vi, int M)
    {
        const Layer& w = W.layers[l];
        State& s = st[l];
        const bool glob = C.global[l] != 0;
        const int H = C.hidden, nh = C.n_heads, nkv = C.n_kv_heads, hd = C.head_dim, rd = C.rel_dim, nq = nh * hd, nk = nkv * hd, grp = nh / nkv;
        const float clip = 2.0f * std::sqrt(static_cast<float>(hd)), rclip = 2.0f * std::sqrt(static_cast<float>(rd));
        // K and V: projection, convolution, head norms, into the cache
        std::vector<float> kp(static_cast<size_t>(M) * nk), vp(kp.size());
        for (int m = 0; m < M; ++m)
            for (int c = 0; c < nk; ++c)
            {
                kp[static_cast<size_t>(m) * nk + c] = rbf(dot(&w.wk[static_cast<size_t>(c) * H], &ki[static_cast<size_t>(m) * H], H));
                vp[static_cast<size_t>(m) * nk + c] = rbf(dot(&w.wv[static_cast<size_t>(c) * H], &vi[static_cast<size_t>(m) * H], H));
            }
        const std::vector<float> kc = glob ? sconv_dynamic(w.k_static, w.k_wproj, s.hist_k, kp, ki, M) : sconv_static(w.k_sconv, s.hist_k, kp, M, nk);
        const std::vector<float> vc = glob ? sconv_dynamic(w.v_static, w.v_wproj, s.hist_v, vp, vi, M) : sconv_static(w.v_sconv, s.hist_v, vp, M, nk);
        for (int m = 0; m < M; ++m)
            for (int h = 0; h < nkv; ++h)
            {
                const float* k = &kc[static_cast<size_t>(m) * nk + h * hd];
                const float* v = &vc[static_cast<size_t>(m) * nk + h * hd];
                const float kf = std::fmin(1.0f, clip / (l2_norm(k, hd) + EPS)), vr = rsqrtf(mean_sq(v, hd) + EPS);
                for (int d = 0; d < hd; ++d)
                {
                    s.K.push_back(rbf(k[d] * kf));
                    s.V.push_back(rbf(v[d] * vr));
                }
            }
        std::vector<float> out(static_cast<size_t>(M) * H), q(static_cast<size_t>(nq)), r(static_cast<size_t>(nh) * rd), og(static_cast<size_t>(nq)),
            prof(static_cast<size_t>(C.rel_win)), sc, tmp(static_cast<size_t>(hd > rd ? hd : rd));
        for (int m = 0; m < M; ++m)
        {
            const int p = pos_ + m;
            const float* x = &qi[static_cast<size_t>(m) * H];
            const float tq = glob ? tau(p) : 1.0f;
            for (int h = 0; h < nh; ++h)
            {
                // q and r of the head: scale, L2 clip, then the log scaling of a global layer
                float* qh = &q[static_cast<size_t>(h) * hd];
                for (int d = 0; d < hd; ++d) tmp[d] = rbf(dot(&w.wq[static_cast<size_t>(h * hd + d) * H], x, H)) * w.q_clip[static_cast<size_t>(h) * hd + d];
                const float qf = std::fmin(1.0f, clip / (l2_norm(tmp.data(), hd) + EPS));
                for (int d = 0; d < hd; ++d) qh[d] = rbf(tmp[d] * qf);
                float* rh = &r[static_cast<size_t>(h) * rd];
                for (int d = 0; d < rd; ++d) tmp[d] = rbf(dot(&w.wr[static_cast<size_t>(h * rd + d) * H], x, H)) * w.rel_clip[static_cast<size_t>(h) * rd + d];
                const float rf = std::fmin(1.0f, rclip / (l2_norm(tmp.data(), rd) + EPS));
                for (int d = 0; d < rd; ++d) rh[d] = rbf(tmp[d] * rf);
                if (glob)
                {
                    for (int d = 0; d < hd; ++d) qh[d] = rbf(qh[d] * tq);
                    for (int d = 0; d < rd; ++d) rh[d] = rbf(rh[d] * tq);
                }
                for (int j = 0; j < C.rel_win; ++j)
                {
                    float a = 0.0f;
                    for (int d = 0; d < rd; ++d) a += rh[d] * w.rel_proj[static_cast<size_t>(d) * C.rel_win + j];
                    prof[j] = a;
                }
                // scores over the visible keys, the sink in the denominator only
                const int kvh = h / grp, lo = glob ? 0 : (p - (C.sliding_window - 1) > 0 ? p - (C.sliding_window - 1) : 0);
                const float sink = rbf(dot(&w.sink[static_cast<size_t>(h) * H], x, H));
                sc.assign(static_cast<size_t>(p - lo + 1), 0.0f);
                float mx = sink;
                for (int kpos = lo; kpos <= p; ++kpos)
                {
                    const int dist = p - kpos;
                    float z = dot(qh, &s.K[static_cast<size_t>(kpos) * nk + kvh * hd], hd) / std::sqrt(static_cast<float>(hd));
                    if (dist < C.rel_win) z += prof[dist];
                    sc[kpos - lo] = z;
                    mx = std::fmax(mx, z);
                }
                float den = std::exp(sink - mx);
                for (float& z : sc)
                {
                    z = std::exp(z - mx);
                    den += z;
                }
                for (int d = 0; d < hd; ++d)
                {
                    float o = 0.0f;
                    for (int kpos = lo; kpos <= p; ++kpos) o += (sc[kpos - lo] / den) * s.V[static_cast<size_t>(kpos) * nk + kvh * hd + d];
                    const float g = rbf(dot(&w.gate[static_cast<size_t>(h * hd + d) * H], x, H));
                    og[static_cast<size_t>(h) * hd + d] = rbf(rbf(o) * sigmoidf(g));
                }
            }
            // wo: one partial sum per rank (its heads), added in rank order
            const int per = nq / C.tp;
            for (int c = 0; c < H; ++c)
            {
                float a = 0.0f;
                for (int rk = 0; rk < C.tp; ++rk)
                {
                    const float part = dot(&w.wo[static_cast<size_t>(c) * nq + rk * per], &og[static_cast<size_t>(rk) * per], per);
                    a = rk ? a + part : part;
                }
                out[static_cast<size_t>(m) * H + c] = rbf(a);
            }
        }
        return out;
    }

    // The router of one token: the top_k experts by sigmoid(logit) + bias (ties to the lower index, in pick order) and
    // their weights route_scale * score / (sum of the picked scores + 1e-20).
    void route(int l, const float* x, std::vector<int>& idx, std::vector<float>& wt) const
    {
        const Layer& w = W.layers[l];
        std::vector<float> score(static_cast<size_t>(C.n_exp));
        std::vector<uint8_t> used(static_cast<size_t>(C.n_exp), 0);
        for (int e = 0; e < C.n_exp; ++e) score[e] = sigmoidf(dot(x, &w.router_w[static_cast<size_t>(e) * C.hidden], C.hidden));
        idx.clear();
        wt.clear();
        float sum = 0.0f;
        for (int k = 0; k < C.top_k; ++k)
        {
            int best = -1;
            for (int e = 0; e < C.n_exp; ++e)
                if (!used[e] && (best < 0 || score[e] + w.router_bias[e] > score[best] + w.router_bias[best])) best = e;
            used[best] = 1;
            idx.push_back(best);
            sum += score[best];
        }
        for (int k = 0; k < C.top_k; ++k) wt.push_back(C.route_scale * score[idx[k]] / (sum + 1e-20f));
    }

    // Softcapped SwiGLU; the sigmoid sees the raw gate.
    static float act(float gate, float up) { return rbf((4.0f * std::tanh(gate / 4.0f)) * sigmoidf(gate) * (25.0f * std::tanh(up / 25.0f))); }

    // Quantize then dequantize n values in the layer's activation format: NVFP4 with its own outer scale (the rule
    // four over six), or MXFP8.
    void requantize(int l, const float* x, int n, float* out) const
    {
        std::vector<uint8_t> codes(static_cast<size_t>(n)), scales(static_cast<size_t>(n) / 16);
        if (C.mxfp8[l])
        {
            mx8::quantize(x, static_cast<size_t>(n) / 32, codes.data(), scales.data());
            mx8::dequantize(codes.data(), scales.data(), static_cast<size_t>(n) / 32, out);
        }
        else
        {
            const float outer = nv4pack::quantize_row(x, n, nv4pack::ScaleRule::FourOverSixMae, codes.data(), scales.data());
            nv4pack::dequantize(codes.data(), scales.data(), static_cast<size_t>(n) / 16, outer, out);
        }
    }

    // x = mlp_norm output [H] -> the routed experts' sum [H], BF16. Each rank quantizes its own slice of the
    // activation; the fc2 partial sums are added in rank order, then the experts in pick order.
    void moe(int l, const float* x, float* out) const
    {
        const int H = C.hidden, I = C.inter, iw = I / C.tp;
        std::vector<int> idx;
        std::vector<float> wt, xq(static_cast<size_t>(H)), a(static_cast<size_t>(I)), aq(static_cast<size_t>(I)), acc(static_cast<size_t>(H), 0.0f);
        route(l, x, idx, wt);
        requantize(l, x, H, xq.data());
        for (int k = 0; k < C.top_k; ++k)
        {
            const Dense& d = expert(l, idx[k]);
            for (int i = 0; i < I; ++i)
                a[i] = act(dot(&d.gate[static_cast<size_t>(i) * H], xq.data(), H), dot(&d.up[static_cast<size_t>(i) * H], xq.data(), H));
            for (int rk = 0; rk < C.tp; ++rk) requantize(l, &a[static_cast<size_t>(rk) * iw], iw, &aq[static_cast<size_t>(rk) * iw]);
            for (int c = 0; c < H; ++c)
            {
                float y = 0.0f;
                for (int rk = 0; rk < C.tp; ++rk)
                {
                    const float part = dot(&d.down[static_cast<size_t>(c) * I + rk * iw], &aq[static_cast<size_t>(rk) * iw], iw);
                    y = rk ? y + part : part;
                }
                acc[c] += wt[k] * y;
            }
        }
        for (int c = 0; c < H; ++c) out[c] = rbf(acc[c]);
    }

private:
    struct State
    {
        std::vector<float> hist_a, hist_m;   // [pos][H]         inputs of attn_sconv and mlp_sconv
        std::vector<float> hist_k, hist_v;   // [pos][nkv * hd]  inputs of k_sconv and v_sconv
        std::vector<float> K, V;             // [pos][nkv * hd]  after the head norms
    };
    struct Dense
    {
        std::vector<float> gate, up, down;   // [I][H], [I][H], [H][I], dequantized
    };

    // The records of expert e, decoded once.
    const Dense& expert(int l, int e) const
    {
        const auto it = cache.find(l * C.n_exp + e);
        if (it != cache.end()) return it->second;
        const Layer& w = W.layers[l];
        const int H = C.hidden, I = C.inter;
        Dense d;
        d.gate.resize(static_cast<size_t>(I) * H);
        d.up.resize(d.gate.size());
        d.down.resize(d.gate.size());
        if (C.mxfp8[l])
        {
            const int n1 = H / 32, n2 = I / 32;
            const size_t per1 = static_cast<size_t>(I / 16) * n1 * 2 * mx8::GR, per2 = static_cast<size_t>(H / 16) * n2 * mx8::GR;
            if (w.fc1.size() != per1 * C.n_exp || w.fc2.size() != per2 * C.n_exp) throw std::runtime_error("ink2ref: MXFP8 expert size");
            float t[16][32];
            for (int ch = 0; ch < I / 16; ++ch)
                for (int g = 0; g < n1; ++g)
                    for (int half = 0; half < 2; ++half)
                    {
                        mx8::decode_group(&w.fc1[e * per1 + ((static_cast<size_t>(ch) * n1 + g) * 2 + half) * mx8::GR], t);
                        std::vector<float>& dst = half ? d.up : d.gate;
                        for (int r = 0; r < 16; ++r) memcpy(&dst[static_cast<size_t>(ch * 16 + r) * H + g * 32], t[r], sizeof t[r]);
                    }
            for (int ot = 0; ot < H / 16; ++ot)
                for (int g = 0; g < n2; ++g)
                {
                    mx8::decode_group(&w.fc2[e * per2 + (static_cast<size_t>(ot) * n2 + g) * mx8::GR], t);
                    for (int r = 0; r < 16; ++r) memcpy(&d.down[static_cast<size_t>(ot * 16 + r) * I + g * 32], t[r], sizeof t[r]);
                }
        }
        else
        {
            const int n1 = H / 64, n2 = I / 64;
            const size_t per1 = static_cast<size_t>(I / 16) * n1 * 2 * nv4pack::GR, per2 = static_cast<size_t>(H / 16) * n2 * nv4pack::GR;
            if (w.fc1.size() != per1 * C.n_exp || w.fc2.size() != per2 * C.n_exp) throw std::runtime_error("ink2ref: NVFP4 expert size");
            float t[16][64];
            for (int ch = 0; ch < I / 16; ++ch)
                for (int g = 0; g < n1; ++g)
                    for (int half = 0; half < 2; ++half)
                    {
                        nv4pack::decode_group(&w.fc1[e * per1 + ((static_cast<size_t>(ch) * n1 + g) * 2 + half) * nv4pack::GR], w.s1[e], t);
                        std::vector<float>& dst = half ? d.up : d.gate;
                        for (int r = 0; r < 16; ++r) memcpy(&dst[static_cast<size_t>(ch * 16 + r) * H + g * 64], t[r], sizeof t[r]);
                    }
            for (int ot = 0; ot < H / 16; ++ot)
                for (int g = 0; g < n2; ++g)
                {
                    nv4pack::decode_group(&w.fc2[e * per2 + (static_cast<size_t>(ot) * n2 + g) * nv4pack::GR], w.s2[e], t);
                    for (int r = 0; r < 16; ++r) memcpy(&d.down[static_cast<size_t>(ot * 16 + r) * I + g * 64], t[r], sizeof t[r]);
                }
        }
        return cache.emplace(l * C.n_exp + e, std::move(d)).first->second;
    }

    Config C;
    const Weights& W;
    std::vector<State> st;
    mutable std::map<int, Dense> cache;
    int pos_ = 0;
};
}   // namespace ink2ref
