//! gr8's weight forms, built at load from decoded values. The decode GEMV records (engine/kernels/densev.cuh and
//! rowtile.cuh) follow the byte layouts of engine/ref/g6ref.py mxfp8_record, bf16_record, rt_record and
//! rt_expert_record. Two round-to-nearest quantizers make MXFP8 and int4 copies of matrices the release stores otherwise:
//!   MXFP8: the D2 equation of tools/convert/mxfp8.py, one E8M0 scale per 32 inputs, on fp32 inputs;
//!   int4 g32: symmetric, scale = fp16(amax / 7) per 32 inputs, code = clamp(rint(x / scale), -8, 7) + 8.

use crate::exlr8::e4m3_round;

/// f32 -> f16 bits, round to nearest even (subnormals kept, overflow to inf).
pub fn f16_bits(x: f32) -> u16 {
    let s = ((x.to_bits() >> 16) & 0x8000) as u16;
    let a = x.abs() as f64;
    if a.is_nan() {
        return s | 0x7e00;
    }
    if a < f64::powi(2.0, -14) {
        return s | (a * f64::powi(2.0, 24)).round_ties_even() as u16;   // 0x400 rounds up into the first normal
    }
    let e = ((a.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let m = (a / f64::powi(2.0, e) * 1024.0).round_ties_even() as u32;   // 1024..=2048
    let (e, m) = if m == 2048 { (e + 1, 1024) } else { (e, m) };
    if e > 15 {
        return s | 0x7c00;
    }
    s | (((e + 15) as u16) << 10) | (m - 1024) as u16
}

/// An E4M3FN value (already on the grid, |v| <= 448) -> its byte.
pub fn e4m3_byte(v: f32) -> u8 {
    let s = if v.is_sign_negative() { 0x80u8 } else { 0 };
    let a = v.abs() as f64;
    if a < f64::powi(2.0, -6) {
        return s | (a * 512.0) as u8;
    }
    let e = ((a.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let m = (a / f64::powi(2.0, e) * 8.0) as u8 - 8;
    s | (((e + 7) as u8) << 3) | m
}

/// MXFP8 codes and E8M0 scales of a row-major [n][k] fp32 matrix (k a multiple of 32): amax per 32 (at least the
/// smallest normal f32), s = clamp(ceil(log2(fl32(amax / 448))) + 127, 0, 254), code = e4m3(x * 2^(127 - s)), RNE.
pub fn mxfp8_quantize(w: &[f32], k: usize) -> (Vec<u8>, Vec<u8>) {
    assert_eq!(k % 32, 0);
    let mut codes = vec![0u8; w.len()];
    let mut scales = vec![0u8; w.len() / 32];
    for (g, (x, c)) in w.as_chunks::<32>().0.iter().zip(codes.as_chunks_mut::<32>().0.iter_mut()).enumerate() {
        let amax = x.iter().fold(0f32, |m, v| m.max(v.abs())).max(f32::MIN_POSITIVE);
        let ratio = (amax / 448.0) as f64;      // the f32 division, then exact in f64 (normal there even when f32 is not)
        // ceil(log2(ratio)), exact: the exponent, plus one unless ratio is a power of two
        let b = ratio.to_bits();
        let (e, frac) = (((b >> 52) & 0x7ff) as i32 - 1023, b & 0xf_ffff_ffff_ffff);
        let ceil = if frac == 0 { e } else { e + 1 };
        let s = (ceil + 127).clamp(0, 254);
        scales[g] = s as u8;
        let mul = f64::powi(2.0, 127 - s);
        for (o, &v) in c.iter_mut().zip(x) {
            *o = e4m3_byte(e4m3_round(((v as f64) * mul) as f32));
        }
    }
    (codes, scales)
}

/// int4 g32 codes ([n][k / 2], input 2j in the low nibble) and fp16 scale bits ([n][k / 32]) of a row-major [n][k]
/// fp32 matrix, round to nearest: scale = fp16(amax / 7), code = clamp(rint(x / scale), -8, 7) + 8 (8 where scale = 0).
pub fn int4_quantize(w: &[f32], k: usize) -> (Vec<u8>, Vec<u16>) {
    assert_eq!(k % 32, 0);
    let mut q = vec![0x88u8; w.len() / 2];
    let mut sc = vec![0u16; w.len() / 32];
    for (g, x) in w.as_chunks::<32>().0.iter().enumerate() {
        let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
        let s = f16_bits(amax / 7.0);
        sc[g] = s;
        let sf = crate::exlr8::f16_to_f32(s);
        if sf == 0.0 {
            continue;
        }
        for j in 0..32 {
            let n = ((x[j] / sf).round_ties_even().clamp(-8.0, 7.0) as i32 + 8) as u8;
            let b = &mut q[g * 16 + j / 2];
            *b = if j.is_multiple_of(2) { (*b & 0xf0) | n } else { (*b & 0x0f) | (n << 4) };
        }
    }
    (q, sc)
}

/// The 32 lanes' (row, k) pairs of a 16x16 tile in the GEMV record order: lane l = (r, c), r = l / 4, c = l % 4,
/// eight elements (r, 2c), (r, 2c+1), (r+8, 2c), (r+8, 2c+1), (r, 2c+8), (r, 2c+9), (r+8, 2c+8), (r+8, 2c+9).
fn gv_order() -> [(usize, usize); 256] {
    let mut o = [(0, 0); 256];
    for l in 0..32 {
        let (r, c) = (l / 4, l % 4);
        let e = [(r, 2 * c), (r, 2 * c + 1), (r + 8, 2 * c), (r + 8, 2 * c + 1),
                 (r, 2 * c + 8), (r, 2 * c + 9), (r + 8, 2 * c + 8), (r + 8, 2 * c + 9)];
        for (i, p) in e.iter().enumerate() {
            o[l * 8 + i] = *p;
        }
    }
    o
}

/// MXFP8 decode record: codes [16T][K], scales [16T][K/32] -> [T][K/32][528] (two 16x16 tiles in lane order, then
/// the 8 pairs {s[i], s[i+8]}).
pub fn mxfp8_record(codes: &[u8], scales: &[u8], k: usize) -> Vec<u8> {
    let n = codes.len() / k;
    assert!(n.is_multiple_of(16) && k.is_multiple_of(32));
    let ord = gv_order();
    let ng = k / 32;
    let mut out = vec![0u8; n / 16 * ng * 528];
    for t in 0..n / 16 {
        for g in 0..ng {
            let rec = &mut out[(t * ng + g) * 528..][..528];
            for h in 0..2 {
                for (i, &(r, c)) in ord.iter().enumerate() {
                    rec[h * 256 + i] = codes[(16 * t + r) * k + 32 * g + 16 * h + c];
                }
            }
            for i in 0..8 {
                rec[512 + 2 * i] = scales[(16 * t + i) * ng + g];
                rec[512 + 2 * i + 1] = scales[(16 * t + i + 8) * ng + g];
            }
        }
    }
    out
}

/// BF16 decode record: w [16T][K] BF16 bits -> [T][K/16][512] (one tile per group, lane order).
pub fn bf16_record(w: &[u16], k: usize) -> Vec<u8> {
    let n = w.len() / k;
    assert!(n.is_multiple_of(16) && k.is_multiple_of(16));
    let ord = gv_order();
    let s = k / 16;
    let mut out = vec![0u8; n / 16 * s * 512];
    for t in 0..n / 16 {
        for j in 0..s {
            let rec = &mut out[(t * s + j) * 512..][..512];
            for (i, &(r, c)) in ord.iter().enumerate() {
                rec[2 * i..2 * i + 2].copy_from_slice(&w[(16 * t + r) * k + 16 * j + c].to_le_bytes());
            }
        }
    }
    out
}

/// int4 g32 row-tile record: codes [16T][K/2], fp16 scale bits [16T][K/32] -> [T][K/32][288] (two 128-byte tiles,
/// lane l's word = bytes (r, c), (r+8, c), (r, c+4), (r+8, c+4) of its 16x8-byte tile, then the 8 pairs {s[i], s[i+8]}).
pub fn rt_record(q: &[u8], s: &[u16], k: usize) -> Vec<u8> {
    let hb = k / 2;
    let n = q.len() / hb;
    assert!(n.is_multiple_of(16) && k.is_multiple_of(32));
    let ng = k / 32;
    let mut out = vec![0u8; n / 16 * ng * 288];
    for t in 0..n / 16 {
        for g in 0..ng {
            let rec = &mut out[(t * ng + g) * 288..][..288];
            for h in 0..2 {
                let col = (2 * g + h) * 8;          // byte column of this 8-byte-wide tile
                for l in 0..32 {
                    let (r, c) = (l / 4, l % 4);
                    let b = |rr: usize, cc: usize| q[(16 * t + rr) * hb + col + cc];
                    let o = h * 128 + l * 4;
                    rec[o..o + 4].copy_from_slice(&[b(r, c), b(r + 8, c), b(r, c + 4), b(r + 8, c + 4)]);
                }
            }
            for i in 0..8 {
                rec[256 + 4 * i..256 + 4 * i + 2].copy_from_slice(&s[(16 * t + i) * ng + g].to_le_bytes());
                rec[256 + 4 * i + 2..256 + 4 * i + 4].copy_from_slice(&s[(16 * t + i + 8) * ng + g].to_le_bytes());
            }
        }
    }
    out
}

/// The dense MLP's int4 records (the shared/dense expert layout): fc1 [I/16][192][gate 288 | up 288] from gate and up
/// ([I][6144] codes as [I][3072] + scales [I][192]), fc2 = rt_record(down) [384][I/32][288] (down [6144][I]).
pub fn rt_expert_record(gate: (&[u8], &[u16]), up: (&[u8], &[u16]), down: (&[u8], &[u16]), hidden: usize, inter: usize)
                        -> (Vec<u8>, Vec<u8>) {
    let g = rt_record(gate.0, gate.1, hidden);
    let u = rt_record(up.0, up.1, hidden);
    let ng = hidden / 32;
    let mut fc1 = vec![0u8; g.len() * 2];
    for t in 0..inter / 16 {
        for j in 0..ng {
            let i = t * ng + j;
            fc1[i * 576..i * 576 + 288].copy_from_slice(&g[i * 288..][..288]);
            fc1[i * 576 + 288..i * 576 + 576].copy_from_slice(&u[i * 288..][..288]);
        }
    }
    (fc1, rt_record(down.0, down.1, inter))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exlr8::{e4m3_to_f32, f16_to_f32};

    #[test]
    fn f16_round_trip_and_rounding() {
        for h in [0u16, 1, 0x3ff, 0x400, 0x3c00, 0x7bff, 0x8001, 0xbc00, 0x3555] {
            assert_eq!(f16_bits(f16_to_f32(h)), h, "{h:#x}");
        }
        assert_eq!(f16_bits(1.0 + 1.0 / 2048.0), 0x3c00);          // tie to even
        assert_eq!(f16_bits(1.0 + 3.0 / 2048.0), 0x3c02);
        assert_eq!(f16_bits(70000.0), 0x7c00);
        assert_eq!(f16_bits(f32::powi(2.0, -25)), 0);               // half the smallest subnormal ties to 0
    }

    #[test]
    fn e4m3_bytes_round_trip() {
        for b in 0..=255u8 {
            let v = e4m3_to_f32(b);
            if !v.is_nan() && !(b == 0x80) {
                assert_eq!(e4m3_byte(v), b, "{b:#x}");
            }
        }
    }

    #[test]
    fn int4_quantize_inverts_on_the_grid() {
        // a row whose values sit on the grid of its own scale comes back exactly
        let s = f16_to_f32(f16_bits(0.7 / 7.0));
        let w: Vec<f32> = (0..32).map(|j| ((j % 15) as f32 - 7.0) * s).collect();
        let (q, sc) = int4_quantize(&w, 32);
        assert_eq!(f16_to_f32(sc[0]), s);
        for j in 0..32usize {
            let n = if j.is_multiple_of(2) { q[j / 2] & 15 } else { q[j / 2] >> 4 } as f32 - 8.0;
            assert_eq!(n * s, w[j]);
        }
    }
}
