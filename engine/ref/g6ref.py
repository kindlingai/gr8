"""gr8 M1 NumPy reference: codecs, kernel record layouts, reference forwards.

Codecs
  TR3   : EXL3 trellis (K bits/weight, 16x16 tiles, mcg 3INST codebook), Hadamard-128 rotations.
          Written from the format (exllamav3 read for understanding only).
  NVFP4 : FP4 E2M1 codes, FP8 E4M3 scale per 16 inputs, FP32 alpha per matrix.
  INT4  : GPTQ int4 symmetric (code - 8), FP16 scale per 32 inputs ("g32").

Numerics contract (what the CUDA kernels are checked against) is in the *_forward functions:
fp32 accumulation, fp16 rounding at the points marked f16().
"""
import numpy as np

H = 6144
MCG_MUL = 0xCBAC1FED
SQ128 = np.float32(0.088388347648)  # 1/sqrt(128)


def f16(a):
    return np.asarray(a, dtype=np.float32).astype(np.float16)


def silu(x):
    x = x.astype(np.float32)
    return x / (np.float32(1.0) + np.exp(-x, dtype=np.float32))


# ----------------------------------------------------------------------------- TR3
def tr3_cb(state):
    """16-bit trellis state -> fp16 weight (3INST, mcg multiplier)."""
    x = (state.astype(np.uint64) * MCG_MUL) & 0xFFFFFFFF
    x = (x & 0x8FFF8FFF) ^ 0x3B603B60
    lo = (x & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float32)
    hi = (x >> 16).astype(np.uint16).view(np.float16).astype(np.float32)
    return (lo + hi).astype(np.float16)


def tr3_tile_pos():
    """t (0..255) -> (k, n) inside a 16x16 tile (k = input dim, n = output dim)."""
    t = np.arange(256)
    lane, j = t // 8, t % 8
    k = 2 * (lane % 4) + (j & 1) + 8 * ((j >> 1) & 1)
    n = lane // 4 + 8 * (j >> 2)
    return k, n


def tr3_states(tiles):
    """tiles int16 [..., 16K] -> states uint32 [..., 256].
    The tile is a tail-biting bit stream, MSB-first over little-endian uint32 words;
    weight t's state is the 16 bits ending at bit (t+1)*K."""
    K = tiles.shape[-1] // 16
    u32 = np.ascontiguousarray(tiles).view(np.uint16).astype(np.uint32)
    w = u32[..., 0::2] | (u32[..., 1::2] << 16)
    be = w.astype(">u4").view(np.uint8).reshape(*w.shape[:-1], -1)
    bits = np.unpackbits(be, axis=-1)  # [..., 256K]
    nb = 256 * K
    t = np.arange(256)[:, None]
    idx = ((t + 1) * K - 16 + np.arange(16)[None, :]) % nb  # [256,16]
    sel = bits[..., idx].astype(np.uint32)  # [...,256,16]
    return (sel << (15 - np.arange(16, dtype=np.uint32))).sum(-1).astype(np.uint32)


def tr3_states_lane(tiles):
    """Same result computed the way the CUDA lane decode does it (2 words + shifts).
    Used by the unit test to validate the kernel's index/shift formulas on CPU."""
    K = tiles.shape[-1] // 16
    nw = 8 * K
    u32 = np.ascontiguousarray(tiles).view(np.uint16).astype(np.uint64)
    w = u32[..., 0::2] | (u32[..., 1::2] << 16)
    out = np.zeros(tiles.shape[:-1] + (256,), np.uint32)
    for lane in range(32):
        b0 = (lane * 8 + 1) * K - 16 + 256 * K
        i0 = (b0 // 32) % nw
        off = b0 % 32
        assert off + 16 + 7 * K <= 64
        m = (w[..., i0] << 32) | w[..., (i0 + 1) % nw]
        for j in range(8):
            out[..., lane * 8 + j] = ((m >> np.uint64(48 - off - K * j)) & 0xFFFF).astype(np.uint32)
    return out


def tr3_unpack(trellis):
    """trellis int16 [kt, nt, 16K] -> W_hat fp16 [kt*16, nt*16]."""
    kt, nt, _ = trellis.shape
    vals = tr3_cb(tr3_states(trellis))  # [kt, nt, 256]
    k, n = tr3_tile_pos()
    tile = np.zeros((kt, nt, 16, 16), np.float16)
    tile[:, :, k, n] = vals
    return tile.transpose(0, 2, 1, 3).reshape(kt * 16, nt * 16)


_H128 = None


def h128():
    global _H128
    if _H128 is None:
        h = np.array([[1.0]], np.float32)
        for _ in range(7):
            h = np.block([[h, h], [h, -h]])
        _H128 = h
    return _H128


def had128(v):
    """Blockwise 128-point Hadamard (Sylvester order) along the last dim, scaled 1/sqrt(128)."""
    s = v.shape
    r = v.astype(np.float32).reshape(*s[:-1], s[-1] // 128, 128) @ h128()
    return r.reshape(s) * SQ128


def tr3_linear(x, w_hat, suh, svh):
    """EXL3 linear: f16(had(x*suh)) @ W_hat -> f16 -> had * svh (fp32, caller rounds)."""
    xr = f16(had128(x.astype(np.float32) * suh.astype(np.float32)))
    z = f16(xr.astype(np.float32) @ w_hat.astype(np.float32))
    return had128(z) * svh.astype(np.float32)


def tr3_expert_forward(x, e):
    """x fp16 [rows, 6144]; e: dict gate/up/down trellis + suh_gu, svh_g, svh_u, suh_d, svh_d.
    Returns y fp32 [rows, 6144] (before routing weight)."""
    wg, wu, wd = tr3_unpack(e["gate"]), tr3_unpack(e["up"]), tr3_unpack(e["down"])
    xr = f16(had128(x.astype(np.float32) * e["suh_gu"].astype(np.float32))).astype(np.float32)
    g = f16(had128(f16(xr @ wg.astype(np.float32))) * e["svh_g"].astype(np.float32))
    u = f16(had128(f16(xr @ wu.astype(np.float32))) * e["svh_u"].astype(np.float32))
    act = f16(silu(g) * u.astype(np.float32))
    h2 = f16(had128(act.astype(np.float32) * e["suh_d"].astype(np.float32)))
    z = f16(h2.astype(np.float32) @ wd.astype(np.float32))
    return had128(z) * e["svh_d"].astype(np.float32)


# record layout <-> canonical (tile order only; the bits are untouched)
def tr3_record_fc1(gate, up):
    """gate/up trellis [384, 8B, 16K] -> fc1 stream [B][384][16][16K] (j<8 gate, j>=8 up)."""
    B = gate.shape[1] // 8
    g = gate.reshape(384, B, 8, -1)
    u = up.reshape(384, B, 8, -1)
    return np.ascontiguousarray(np.concatenate([g, u], axis=2).transpose(1, 0, 2, 3))


def tr3_unrecord_fc1(rec):
    B = rec.shape[0]
    r = rec.transpose(1, 0, 2, 3)  # [384][B][16][W]
    gate = r[:, :, :8].reshape(384, B * 8, -1)
    up = r[:, :, 8:].reshape(384, B * 8, -1)
    return np.ascontiguousarray(gate), np.ascontiguousarray(up)


def tr3_record_fc2(down):
    """down trellis [8B, 384, 16K] -> fc2 stream [384][8B][16K]."""
    return np.ascontiguousarray(down.transpose(1, 0, 2))


def tr3_unrecord_fc2(rec):
    return np.ascontiguousarray(rec.transpose(1, 0, 2))


# ----------------------------------------------------------------------------- NVFP4 / INT4
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)


def e4m3(b):
    b = np.asarray(b, np.uint8).astype(np.int32)
    s = np.where(b & 0x80, -1.0, 1.0)
    e = (b >> 3) & 0xF
    m = b & 7
    v = np.where(e == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * 2.0 ** (e - 7.0))
    return (s * v).astype(np.float32)


def nib(codes):
    """u8 [rows, cols/2] -> nibble [rows, cols], low nibble first."""
    out = np.empty(codes.shape[:-1] + (codes.shape[-1] * 2,), np.uint8)
    out[..., 0::2] = codes & 0xF
    out[..., 1::2] = codes >> 4
    return out


def nvfp4_codes(codes):
    return E2M1[nib(codes)]


def int4_codes(codes):
    return nib(codes).astype(np.float32) - 8.0


class RT:
    """row-tile codec descriptor: G k16-steps per scale group, SB scale bytes."""

    def __init__(self, name):
        self.name = name
        self.G = 1 if name == "nvfp4" else 2
        self.SB = 1 if name == "nvfp4" else 2
        self.GR = self.G * 128 + 16 * self.SB
        self.gk = 16 * self.G

    def code_vals(self, codes):
        return nvfp4_codes(codes) if self.name == "nvfp4" else int4_codes(codes)

    def scale_vals(self, scales):
        if self.name == "nvfp4":
            return e4m3(scales)
        return scales.view(np.float16).astype(np.float32) if scales.dtype != np.float16 else scales.astype(np.float32)

    def rand(self, rng, rows, cols):
        codes = rng.integers(0, 256, (rows, cols // 2), dtype=np.uint8)
        if self.name == "nvfp4":
            scales = rng.integers(0x30, 0x48, (rows, cols // 16), dtype=np.uint8)
        else:
            scales = (rng.random((rows, cols // 32), dtype=np.float32) * 0.02 + 0.005).astype(np.float16)
        return codes, scales


def rt_matmul(rt, x, codes, scales, alpha=1.0):
    """y[rows_x, out] = alpha * sum_groups scale[out,g] * (x[:, g] @ code[out, g]) in fp32.
    x fp16 [M, K]; codes u8 [out, K/2]; scales [out, K/gk]."""
    c = rt.code_vals(codes)  # [out, K]
    s = rt.scale_vals(scales)  # [out, K/gk]
    out_, K = c.shape
    ng = K // rt.gk
    xg = x.astype(np.float32).reshape(x.shape[0], ng, rt.gk)
    cg = c.reshape(out_, ng, rt.gk)
    part = np.einsum("mgk,ogk->mog", xg, cg, dtype=np.float32)
    return np.float32(alpha) * np.einsum("mog,og->mo", part, s, dtype=np.float32)


def rt_tiles(codes):
    """u8 [16*T, 8*S] -> mma-A-native tiles uint32 [T, S, 32] (see README: word per lane)."""
    T, S = codes.shape[0] // 16, codes.shape[1] // 8
    b = codes.reshape(T, 16, S, 8).transpose(0, 2, 1, 3).astype(np.uint32)  # [T,S,row,byte]
    lane = np.arange(32)
    r, c = lane // 4, lane % 4
    return b[..., r, c] | (b[..., r + 8, c] << 8) | (b[..., r, c + 4] << 16) | (b[..., r + 8, c + 4] << 24)


def rt_untiles(w):
    T, S, _ = w.shape
    b = np.zeros((T, S, 16, 8), np.uint8)
    lane = np.arange(32)
    r, c = lane // 4, lane % 4
    b[..., r, c] = w & 0xFF
    b[..., r + 8, c] = (w >> 8) & 0xFF
    b[..., r, c + 4] = (w >> 16) & 0xFF
    b[..., r + 8, c + 4] = (w >> 24) & 0xFF
    return b.transpose(0, 2, 1, 3).reshape(T * 16, S * 8)


def rt_record(rt, codes, scales):
    """canonical [16T rows, K] matrix -> stream bytes [T][NG][GR]:
    G tiles of 128 B, then 8 pairs {s[i], s[i+8]} of SB bytes each."""
    T = codes.shape[0] // 16
    S = codes.shape[1] // 8
    ng = S // rt.G
    tiles = np.ascontiguousarray(rt_tiles(codes).astype("<u4")).view(np.uint8).reshape(T, ng, rt.G * 128)
    sc = np.ascontiguousarray(scales).view(np.uint8).reshape(T, 16, ng, rt.SB)
    pairs = np.stack([sc[:, :8], sc[:, 8:]], axis=2)  # [T, 8, 2, ng, SB]
    pairs = pairs.transpose(0, 3, 1, 2, 4).reshape(T, ng, 16 * rt.SB)
    return np.ascontiguousarray(np.concatenate([tiles, pairs], axis=2))


def rt_unrecord(rt, rec):
    T, ng, _ = rec.shape
    tiles = np.ascontiguousarray(rec[:, :, : rt.G * 128]).view("<u4").reshape(T, ng * rt.G, 32)
    codes = rt_untiles(tiles)
    pairs = rec[:, :, rt.G * 128 :].reshape(T, ng, 8, 2, rt.SB).transpose(0, 3, 2, 1, 4)  # [T,2,8,ng,SB]
    sc = np.ascontiguousarray(pairs.reshape(T * 16, ng, rt.SB))
    scales = sc.reshape(T * 16, ng) if rt.SB == 1 else sc.view(np.float16).reshape(T * 16, ng)
    return codes, scales


def rt_expert_record(rt, gate, gs, up, us, down, ds):
    """fc1 stream [chunk][NG][gate GR | up GR], fc2 stream [out-tile][NG2][GR]."""
    g = rt_record(rt, gate, gs)
    u = rt_record(rt, up, us)
    return np.ascontiguousarray(np.concatenate([g, u], axis=2)), rt_record(rt, down, ds)


def rt_expert_unrecord(rt, fc1, fc2):
    g, gs = rt_unrecord(rt, fc1[:, :, : rt.GR])
    u, us = rt_unrecord(rt, fc1[:, :, rt.GR :])
    d, ds = rt_unrecord(rt, fc2)
    return g, gs, u, us, d, ds


def rt_expert_forward(rt, x, g, gs, u, us, d, ds, alpha=(1.0, 1.0, 1.0)):
    """Returns (act fp16 [M, I], y fp32 [M, 6144]) before routing weight."""
    gg = rt_matmul(rt, x, g, gs, alpha[0])
    uu = rt_matmul(rt, x, u, us, alpha[1])
    act = f16(silu(gg) * uu)
    return act, rt_matmul(rt, act, d, ds, alpha[2])


# ----------------------------------------------------------------------------- dense GEMV records (densev.cuh)
def bf16_round(v):
    """f32 -> nearest bf16 (round to nearest even), returned as the u16 bit pattern."""
    u = np.ascontiguousarray(v, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_vals(bits):
    return (np.asarray(bits, np.uint16).astype(np.uint32) << 16).view(np.float32)


def e8m0(code):
    return np.ldexp(np.float32(1.0), np.asarray(code, np.int32) - 127).astype(np.float32)


def _gv_lane():
    lane = np.arange(32)
    return lane // 4, lane % 4


def mxfp8_record(codes, scales):
    """codes u8 [16T, K] (E4M3FN), scales u8 [16T, K/32] (E8M0 code = e + 127) -> stream u8 [T][K/32][528]:
    2 tiles of [32 lanes][8 B] = {row r k 2c,2c+1 | row r+8 same k | row r k 2c+8,2c+9 | row r+8 same k}, r = lane/4, c = lane%4,
    then 8 pairs {s[i], s[i+8]}."""
    T, S = codes.shape[0] // 16, codes.shape[1] // 16
    b = codes.reshape(T, 16, S, 16).transpose(0, 2, 1, 3)  # [T,S,row,k]
    r, c = _gv_lane()
    t = np.stack([b[..., r, 2 * c], b[..., r, 2 * c + 1], b[..., r + 8, 2 * c], b[..., r + 8, 2 * c + 1],
                  b[..., r, 2 * c + 8], b[..., r, 2 * c + 9], b[..., r + 8, 2 * c + 8], b[..., r + 8, 2 * c + 9]], axis=-1)  # [T,S,32,8]
    t = t.reshape(T, S // 2, 512)
    sc = scales.reshape(T, 16, S // 2)
    pairs = np.stack([sc[:, :8], sc[:, 8:]], axis=2).transpose(0, 3, 1, 2).reshape(T, S // 2, 16)
    return np.ascontiguousarray(np.concatenate([t, pairs], axis=2))


def mxfp8_unrecord(rec):
    T, ng, _ = rec.shape
    t = rec[:, :, :512].reshape(T, ng * 2, 32, 8)
    b = np.zeros((T, ng * 2, 16, 16), np.uint8)
    r, c = _gv_lane()
    for i, (rr, kk) in enumerate([(r, 2 * c), (r, 2 * c + 1), (r + 8, 2 * c), (r + 8, 2 * c + 1),
                                  (r, 2 * c + 8), (r, 2 * c + 9), (r + 8, 2 * c + 8), (r + 8, 2 * c + 9)]):
        b[:, :, rr, kk] = t[..., i]
    codes = b.transpose(0, 2, 1, 3).reshape(T * 16, ng * 32)
    scales = rec[:, :, 512:].reshape(T, ng, 8, 2).transpose(0, 3, 2, 1).reshape(T * 16, ng)
    return codes, scales


def mxfp8_matmul(x, codes, scales):
    """y = sum_g 2^(s-127) * (x_g @ e4m3(code_g)) in fp32. x fp16 [M, K]."""
    c = e4m3(codes)
    O, K = c.shape
    part = np.einsum("mgk,ogk->mog", x.astype(np.float32).reshape(-1, K // 32, 32), c.reshape(O, K // 32, 32), dtype=np.float32)
    return np.einsum("mog,og->mo", part, e8m0(scales), dtype=np.float32)


def bf16_record(w):
    """w u16 [16T, K] (bf16 bits) -> stream u8 [T][K/16][512]: lane l holds 4 u32 = (row r, k 2c..2c+1), (row r+8, same k),
    (row r, k 2c+8..), (row r+8, k 2c+8..)."""
    T, S = w.shape[0] // 16, w.shape[1] // 16
    b = w.reshape(T, 16, S, 16).transpose(0, 2, 1, 3)
    r, c = _gv_lane()
    t = np.stack([b[..., r, 2 * c], b[..., r, 2 * c + 1], b[..., r + 8, 2 * c], b[..., r + 8, 2 * c + 1],
                  b[..., r, 2 * c + 8], b[..., r, 2 * c + 9], b[..., r + 8, 2 * c + 8], b[..., r + 8, 2 * c + 9]], axis=-1)
    return np.ascontiguousarray(t.astype("<u2")).view(np.uint8).reshape(T, S, 512)


def bf16_unrecord(rec):
    T, S, _ = rec.shape
    t = np.ascontiguousarray(rec).view("<u2").reshape(T, S, 32, 8)
    b = np.zeros((T, S, 16, 16), np.uint16)
    r, c = _gv_lane()
    for i, (rr, kk) in enumerate([(r, 2 * c), (r, 2 * c + 1), (r + 8, 2 * c), (r + 8, 2 * c + 1),
                                  (r, 2 * c + 8), (r, 2 * c + 9), (r + 8, 2 * c + 8), (r + 8, 2 * c + 9)]):
        b[:, :, rr, kk] = t[..., i]
    return b.transpose(0, 2, 1, 3).reshape(T * 16, S * 16)


def bf16_matmul(x, w):
    """y = bf16(x) @ W^T in fp32 (the bf16 mma takes both operands as bf16). x fp16 [M, K], w bf16 bits [O, K]."""
    return bf16_vals(bf16_round(x.astype(np.float32))).astype(np.float32) @ bf16_vals(w).T


def xfrag(x):
    """x fp16 [M<=8, K] -> mma-B-native fragments uint32 [K/16, 32, 2]."""
    M, K = x.shape
    xp = np.zeros((8, K), np.float16)
    xp[:M] = x
    u = xp.view(np.uint16).astype(np.uint32).reshape(8, K // 16, 16)
    lane = np.arange(32)
    t, c = lane // 4, lane % 4
    out = np.empty((K // 16, 32, 2), np.uint32)
    out[:, :, 0] = (u[t, :, 2 * c] | (u[t, :, 2 * c + 1] << 16)).T
    out[:, :, 1] = (u[t, :, 8 + 2 * c] | (u[t, :, 9 + 2 * c] << 16)).T
    return out


# ----------------------------------------------------------------------------- routing
def make_routes(M, d, topk=8):
    """Synthetic route plan: token m uses slots (m*topk + j) % d, j < min(topk, d).
    Returns tok_slot int32 [8, 8] (-1 = none), slot_mask int32 [d]."""
    tok_slot = -np.ones((8, 8), np.int32)
    mask = np.zeros(d, np.int32)
    for m in range(M):
        for j in range(min(topk, d)):
            s = (m * topk + j) % d
            tok_slot[m, j] = s
            mask[s] |= 1 << m
    return tok_slot, mask


def rel_err(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    d = np.abs(a - b).max()
    return float(d), float(d / max(np.abs(b).max(), 1e-30))
