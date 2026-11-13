//! NVFP4: E2M1 4-bit codes, one E4M3FN scale per 16 elements along K, one fp32 outer scale per matrix. Our own code.
//!
//! Checkpoint layout of one [N][K] matrix (K a multiple of 16):
//!   codes  [N][K / 2]  bytes, byte j of a row = element 2j in the low nibble, 2j + 1 in the high nibble
//!   scales [N][K / 16] E4M3FN bytes
//!   outer  fp32, stored directly (`weight_scale_2`) or as `weight_amax / weight_outer_scale_denominator`
//!   value(n, k) = e2m1(code) * (outer * e4m3(scale[n][k / 16]))
//!
//! Two scale-selection rules quantize an fp32 vector (activations at run time, weights offline):
//!   Static6        block scale = e4m3((block amax / 6) / outer), outer = amax / (6 * 448) unless fixed
//!   FourOverSixMae per block, the better (L1 error) of block amax / 6 and block amax / 4; outer = amax / (6 * 256)
//!                  unless fixed, which leaves the amax / 4 candidate (<= 384) inside E4M3. Ties keep the / 6 one.
//!
//! The device record "nv4g" covers 16 rows x 64 k in 576 bytes (nine 64-byte bursts): 512 code bytes in the lane / tile
//! order of the MXFP4 record "mx4g", then the 64 block scales (pack_nv4_scales_64). engine/kernels/nvfp4_pack.h is the
//! C++ twin, byte for byte (tests/nvfp4.rs).

use crate::exlr8::{e4m3_round, e4m3_to_f32};
use crate::json::Json;
use crate::records::e4m3_byte;

/// K elements of one record.
pub const NV4_KG: usize = 64;
/// Byte offset of the 64 block scales inside a record.
pub const NV4_SOFF: usize = 512;
/// Bytes of one record (16 rows x 64 k).
pub const NV4_GR: usize = 576;
/// Largest E2M1 magnitude.
pub const E2M1_MAX: f32 = 6.0;

const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// E2M1 code (bit 3 = sign, bits 2..0 = magnitude index) -> value. Code 8 is -0.0.
#[inline(always)]
pub fn e2m1_to_f32(code: u8) -> f32 {
    let m = E2M1[(code & 7) as usize];
    if code & 8 != 0 { -m } else { m }
}

/// fp32 -> E2M1 code, round to nearest even on the grid, saturating at 6. A result of magnitude 0 is code 0.
#[inline(always)]
pub fn f32_to_e2m1_rne(v: f32) -> u8 {
    let a = v.abs().min(E2M1_MAX);
    let idx = if a < 2.0 {
        (a * 2.0).round_ties_even() as u8
    } else if a < 4.0 {
        4 + (a - 2.0).round_ties_even() as u8
    } else {
        6 + ((a - 4.0) * 0.5).round_ties_even() as u8
    };
    if v.is_sign_negative() && idx != 0 { 8 | idx } else { idx }
}

/// fp32 -> E4M3FN byte, round to nearest even, saturating at 448. NaN gives 0x7f; a result of 0 is byte 0x00.
#[inline(always)]
pub fn f32_to_e4m3_rne(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7f;
    }
    let r = e4m3_round(v);
    if r == 0.0 { 0 } else { e4m3_byte(r) }
}

/// The fp32 outer scale of each matrix (expert) of a tensor, in either checkpoint convention.
#[derive(Clone, Debug, PartialEq)]
pub enum Nvfp4OuterScale {
    /// `weight_scale_2` stored directly: one scalar for all, or one per matrix.
    Direct(Vec<f32>),
    /// `weight_amax` (one, or one per matrix) over the scalar `weight_outer_scale_denominator`.
    AmaxDenominator { amax: Vec<f32>, denominator: f32 },
}

impl Nvfp4OuterScale {
    /// Outer scale of matrix `e` (a one-element vector applies to every matrix).
    pub fn at(&self, e: usize) -> f32 {
        match self {
            Nvfp4OuterScale::Direct(v) => if v.len() == 1 { v[0] } else { v[e] },
            Nvfp4OuterScale::AmaxDenominator { amax, denominator } => {
                (if amax.len() == 1 { amax[0] } else { amax[e] }) / denominator
            }
        }
    }

    /// Matrices described (1 = shared).
    pub fn len(&self) -> usize {
        match self {
            Nvfp4OuterScale::Direct(v) => v.len(),
            Nvfp4OuterScale::AmaxDenominator { amax, .. } => amax.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// How the block scales (and the default outer scale) of a quantized vector are chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nvfp4ScaleRule {
    /// Block amax maps to 6; outer = amax / 2688.
    Static6,
    /// Per block the lower L1 error of amax -> 6 and amax -> 4; outer = amax / 1536.
    FourOverSixMae,
}

impl Nvfp4ScaleRule {
    /// amax / this = the default outer scale.
    pub fn outer_denominator(self) -> f32 {
        match self {
            Nvfp4ScaleRule::Static6 => 2688.0,
            Nvfp4ScaleRule::FourOverSixMae => 1536.0,
        }
    }
}

/// The NVFP4 rule named by checkpoint metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Nvfp4RuleConfig {
    pub rule: Nvfp4ScaleRule,
}

impl Nvfp4RuleConfig {
    /// From a `scale_selection_rule` string; absent means Static6.
    pub fn from_rule_name(name: Option<&str>) -> Result<Self, String> {
        let rule = match name {
            None | Some("static_6") => Nvfp4ScaleRule::Static6,
            Some("fouroversix_mae") => Nvfp4ScaleRule::FourOverSixMae,
            Some(o) => return Err(format!("nvfp4: unknown scale_selection_rule {o:?}")),
        };
        Ok(Nvfp4RuleConfig { rule })
    }

    /// From either a parameter group of an `hf_quant_config.json` (its `scale_selection_rule`, directly or under
    /// `required_activation_quantization`) or a standard `quantization_config` object (`quant_method` modelopt or
    /// compressed-tensors with an NVFP4 `format` / `quant_algo`), which is always Static6.
    pub fn parse(j: &Json) -> Result<Self, String> {
        let direct = j.get("scale_selection_rule").and_then(Json::str);
        let nested = j.get("required_activation_quantization").and_then(|a| a.get("scale_selection_rule")).and_then(Json::str);
        if let Some(name) = direct.or(nested) {
            return Self::from_rule_name(Some(name));
        }
        if let Some(method) = j.get("quant_method").and_then(Json::str) {
            if method != "modelopt" && method != "compressed-tensors" {
                return Err(format!("nvfp4: quant_method {method:?} is not an NVFP4 container"));
            }
            let named = |k: &str| j.get(k).and_then(Json::str).map(|s| s.to_ascii_lowercase().contains("nvfp4"));
            return match named("format").or(named("quant_algo")) {
                Some(false) => Err("nvfp4: quantization_config names a format that is not NVFP4".to_string()),
                _ => Ok(Nvfp4RuleConfig { rule: Nvfp4ScaleRule::Static6 }),
            };
        }
        Self::from_rule_name(None)
    }
}

/// One candidate of a 16-element block: scale byte, 16 codes, L1 error. `target` is the E2M1 magnitude the block amax
/// maps to (6 or 4).
#[inline(always)]
fn block_candidate(x: &[f32; 16], b_max: f32, outer: f32, target: f32) -> (u8, [u8; 16], f32) {
    let s = if outer == 0.0 { 0 } else { f32_to_e4m3_rne((b_max / target) / outer) };
    let sf = outer * e4m3_to_f32(s);
    let mut q = [0u8; 16];
    let mut err = 0f32;
    for j in 0..16 {
        if sf != 0.0 {
            q[j] = f32_to_e2m1_rne(x[j] / sf);
        }
        err += (e2m1_to_f32(q[j]) * sf - x[j]).abs();
    }
    (s, q, err)
}

/// Quantize a row-major [n][k] fp32 slice (k a multiple of 16) under one outer scale: `fixed_outer`, or the amax of
/// the whole slice over the rule's denominator. Returns (codes [n][k / 2], scales [n][k / 16], outer scale used).
/// Quantize one row at a time for a per-row (per-token) outer scale.
pub fn nvfp4_quantize(x: &[f32], k: usize, rule: Nvfp4ScaleRule, fixed_outer: Option<f32>) -> (Vec<u8>, Vec<u8>, f32) {
    assert!(k.is_multiple_of(16) && x.len().is_multiple_of(k.max(1)));
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let outer = fixed_outer.unwrap_or(amax / rule.outer_denominator());
    let mut codes = vec![0u8; x.len() / 2];
    let mut scales = vec![0u8; x.len() / 16];
    for (b, blk) in x.as_chunks::<16>().0.iter().enumerate() {
        let b_max = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let (mut s, mut q, e6) = block_candidate(blk, b_max, outer, 6.0);
        if rule == Nvfp4ScaleRule::FourOverSixMae {
            let (s4, q4, e4) = block_candidate(blk, b_max, outer, 4.0);
            if e4 < e6 {
                (s, q) = (s4, q4);
            }
        }
        scales[b] = s;
        for j in 0..8 {
            codes[b * 8 + j] = q[2 * j] | (q[2 * j + 1] << 4);
        }
    }
    (codes, scales, outer)
}

/// Row-major dequantization, the definition of the format: codes [n][k / 2], scales [n][k / 16] -> fp32 [n][k].
pub fn nvfp4_dequantize(codes: &[u8], scales: &[u8], outer: f32) -> Vec<f32> {
    assert_eq!(codes.len(), scales.len() * 8);
    let mut out = vec![0f32; codes.len() * 2];
    for (b, &s) in scales.iter().enumerate() {
        let sf = outer * e4m3_to_f32(s);
        for j in 0..8 {
            let c = codes[b * 8 + j];
            out[b * 16 + 2 * j] = e2m1_to_f32(c & 15) * sf;
            out[b * 16 + 2 * j + 1] = e2m1_to_f32(c >> 4) * sf;
        }
    }
    out
}

/// The 64 scale bytes of a record: for r in 0..8, bytes r * 8 .. r * 8 + 4 = the 4 block scales of row r0 + r, bytes
/// r * 8 + 4 .. r * 8 + 8 = those of row r0 + r + 8 (each an aligned u32). `stride` = scale bytes per source row,
/// `sb` = first block-16 scale column.
#[inline(always)]
pub fn pack_nv4_scales_64(dst: &mut [u8], scales: &[u8], stride: usize, r0: usize, sb: usize) {
    for r in 0..8 {
        dst[r * 8..r * 8 + 4].copy_from_slice(&scales[(r0 + r) * stride + sb..][..4]);
        dst[r * 8 + 4..r * 8 + 8].copy_from_slice(&scales[(r0 + r + 8) * stride + sb..][..4]);
    }
}

/// Inverse of `pack_nv4_scales_64`: [row 0..16][block 0..4].
#[inline(always)]
pub fn unpack_nv4_scales_64(src: &[u8]) -> [[u8; 4]; 16] {
    let mut o = [[0u8; 4]; 16];
    for r in 0..8 {
        o[r].copy_from_slice(&src[r * 8..r * 8 + 4]);
        o[r + 8].copy_from_slice(&src[r * 8 + 4..r * 8 + 8]);
    }
    o
}

/// One record: rows r0 .. r0 + 16, code bytes pc .. pc + 32 and block scales sc .. sc + 4 of each row. `ps` / `ss` =
/// code / scale bytes per source row.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn nv4_group(dst: &mut [u8], p: &[u8], s: &[u8], ps: usize, ss: usize, r0: usize, pc: usize, sc: usize) {
    for lane in 0..32 {
        let (r, c) = (lane >> 2, lane & 3);
        for t in 0..4 {
            let j = pc + t * 8 + c;
            let o = lane * 16 + t * 4;
            dst[o] = p[(r0 + r) * ps + j];
            dst[o + 1] = p[(r0 + r + 8) * ps + j];
            dst[o + 2] = p[(r0 + r) * ps + j + 4];
            dst[o + 3] = p[(r0 + r + 8) * ps + j + 4];
        }
    }
    pack_nv4_scales_64(&mut dst[NV4_SOFF..NV4_GR], s, ss, r0, sc);
}

/// Any [N][K] matrix (N a multiple of 16, K of 64): codes [N][K / 2], scales [N][K / 16] -> [N / 16][K / 64][576].
pub fn nvfp4_record(codes: &[u8], scales_e4m3: &[u8], k: usize) -> Vec<u8> {
    assert!(k > 0 && k.is_multiple_of(NV4_KG));
    let n = codes.len() / (k / 2);
    nvfp4_fc2(codes, scales_e4m3, n, k, 0, k)
}

/// Gate and up matrices [r][k] -> [r / 16][k / 64][gate 576 | up 576].
pub fn nvfp4_fc1(gp: &[u8], gs: &[u8], up: &[u8], us: &[u8], r: usize, k: usize) -> Vec<u8> {
    assert!(r.is_multiple_of(16) && k > 0 && k.is_multiple_of(NV4_KG));
    assert!(gp.len() == r * k / 2 && up.len() == r * k / 2 && gs.len() == r * k / 16 && us.len() == r * k / 16);
    let ng = k / NV4_KG;
    let mut out = vec![0u8; r / 16 * ng * 2 * NV4_GR];
    for (i, d) in out.as_chunks_mut::<{ 2 * NV4_GR }>().0.iter_mut().enumerate() {
        let (ch, g) = (i / ng, i % ng);
        nv4_group(&mut d[..NV4_GR], gp, gs, k / 2, k / 16, ch * 16, g * 32, g * 4);
        nv4_group(&mut d[NV4_GR..], up, us, k / 2, k / 16, ch * 16, g * 32, g * 4);
    }
    out
}

/// Columns c0 .. c0 + kn (both multiples of 64) of a [rows][kfull] matrix -> [rows / 16][kn / 64][576].
pub fn nvfp4_fc2(p: &[u8], s: &[u8], rows: usize, kfull: usize, c0: usize, kn: usize) -> Vec<u8> {
    assert!(rows.is_multiple_of(16) && kfull.is_multiple_of(16) && c0.is_multiple_of(NV4_KG) && kn.is_multiple_of(NV4_KG));
    assert!(c0 + kn <= kfull && p.len() == rows * kfull / 2 && s.len() == rows * kfull / 16);
    let ng = kn / NV4_KG;
    let mut out = vec![0u8; rows / 16 * ng * NV4_GR];
    for (i, d) in out.as_chunks_mut::<NV4_GR>().0.iter_mut().enumerate() {
        let (ot, g) = (i / ng, i % ng);
        nv4_group(d, p, s, kfull / 2, kfull / 16, ot * 16, c0 / 2 + g * 32, c0 / 16 + g * 4);
    }
    out
}

/// Host oracle of the record layout: the 16 x 64 dequantized weights of one record, read back from the lane / tile /
/// byte positions the device reads.
pub fn nvfp4_decode_group(rec: &[u8; NV4_GR], outer_scale: f32) -> [[f32; 64]; 16] {
    let sc = unpack_nv4_scales_64(&rec[NV4_SOFF..]);
    let mut out = [[0f32; 64]; 16];
    for lane in 0..32 {
        let (r, c) = (lane >> 2, lane & 3);
        for t in 0..4 {
            let w = &rec[lane * 16 + t * 4..][..4];
            let (sl, sh) = (outer_scale * e4m3_to_f32(sc[r][t]), outer_scale * e4m3_to_f32(sc[r + 8][t]));
            let kb = t * 16 + 2 * c;
            out[r][kb] = e2m1_to_f32(w[0] & 15) * sl;
            out[r][kb + 1] = e2m1_to_f32(w[0] >> 4) * sl;
            out[r + 8][kb] = e2m1_to_f32(w[1] & 15) * sh;
            out[r + 8][kb + 1] = e2m1_to_f32(w[1] >> 4) * sh;
            out[r][kb + 8] = e2m1_to_f32(w[2] & 15) * sl;
            out[r][kb + 9] = e2m1_to_f32(w[2] >> 4) * sl;
            out[r + 8][kb + 8] = e2m1_to_f32(w[3] & 15) * sh;
            out[r + 8][kb + 9] = e2m1_to_f32(w[3] >> 4) * sh;
        }
    }
    out
}
