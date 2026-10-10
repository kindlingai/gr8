//! Decodes the exlr8 v2 release's dense and vocabulary matrices (`dense/exlr8-dense-LLL.safetensors`,
//! `dense/exlr8-vocab.safetensors`). The values equal the output of `loader/exlr8.py` (kindlingai/exlr8) bit for bit.
//!
//! A matrix M has `M.record`, int64 [8] = (format, out, in, K, side, unit, shared input rotation, solve method), and
//! data tensors by format (the model card's Loader section). `Matrix::decode` returns any block of the HF weight
//! (out, in) as fp32. For a trellis matrix it decodes only the 128-channel entries that cover the block on the sharded
//! side.
//!
//! Trellis reconstruction is W (in, out) = diag(suh) Hb T Hb diag(svh), Hb the block-diagonal 128-block Hadamard. The
//! reference computes the two Hadamard products as fp32 matmuls (torch on NVPL or cuBLAS). Both sum each output in
//! input order with one fused multiply-add per term, starting from zero, so `had_left` and `had_right` do the same,
//! and multiply by suh and svh as separate roundings. Any other order changes the last bit of some results.
//!
//! Written from the card's format description and checked against the reference (tests/exlr8_fixture.rs in CI,
//! tests/exlr8_compare.py on the release's files). The trellis codebook, the bit packing, the tile order and the
//! 128x128 Hadamard are exllamav3's primitives (MIT, Turboderp), as the card says.

use crate::json::{self, Json};
use std::fs::File;
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::sync::OnceLock;

pub const TRELLIS: i64 = 1;
pub const INT4: i64 = 2;
pub const MXFP8: i64 = 3;
pub const FP8_BLOCK: i64 = 4;
pub const FP8_ROW: i64 = 5;
pub const BF16: i64 = 6;
pub const FP32: i64 = 7;
/// record field 4: the sharded side
pub const SIDE_REPLICATED: i64 = 0;
pub const SIDE_OUT: i64 = 1;
pub const SIDE_IN: i64 = 2;

const MCG: u32 = 0xCBAC1FED;

/// The data tensors of each format, in the release's checksum order.
pub fn data_tensors(fmt: i64) -> &'static [&'static str] {
    match fmt {
        TRELLIS => &["trellis", "suh", "svh"],
        INT4 => &["q4", "s4"],
        MXFP8 => &["fp8", "e8m0"],
        FP8_BLOCK | FP8_ROW => &["fp8", "scale"],
        BF16 => &["bf16"],
        FP32 => &["fp32"],
        _ => &[],
    }
}

// ------------------------------------------------------------------------------------------------ files

/// The dense file of layer `l`, or the vocabulary file (`l` = None), under `dir/dense/` or `dir/`.
pub fn dense_path(dir: &str, l: Option<usize>) -> String {
    let name = match l {
        Some(l) => format!("exlr8-dense-{l:03}.safetensors"),
        None => "exlr8-vocab.safetensors".to_string(),
    };
    let nested = format!("{dir}/dense/{name}");
    if std::path::Path::new(&nested).exists() { nested } else { format!("{dir}/{name}") }
}

/// A tensor's header entry.
#[derive(Clone, Debug)]
pub struct Info {
    pub dtype: String,
    pub shape: Vec<usize>,
    /// absolute byte range in the file
    pub start: u64,
    pub end: u64,
}

/// A safetensors file: header parsed at open, tensor bytes read on demand.
pub struct StFile {
    pub path: String,
    f: File,
    names: Vec<String>,
    infos: std::collections::HashMap<String, Info>,
}

impl StFile {
    pub fn open(path: &str) -> Result<StFile, String> {
        let f = File::open(path).map_err(|e| format!("cannot open {path}: {e}"))?;
        let mut n = [0u8; 8];
        f.read_exact_at(&mut n, 0).map_err(|e| format!("{path}: {e}"))?;
        let n = u64::from_le_bytes(n);
        let mut h = vec![0u8; n as usize];
        f.read_exact_at(&mut h, 8).map_err(|e| format!("{path}: {e}"))?;
        let head = json::parse(std::str::from_utf8(&h).map_err(|_| format!("{path}: bad header"))?)?;
        let obj = head.obj().ok_or(format!("{path}: header is not an object"))?;
        let mut names = Vec::new();
        let mut infos = std::collections::HashMap::new();
        for (k, v) in obj {
            if k == "__metadata__" || k.starts_with("__pad.") {
                continue;
            }
            let bad = || format!("{path}: bad header entry {k}");
            let off = v.get("data_offsets").and_then(Json::arr).ok_or_else(bad)?;
            let (a, b) = (off[0].int().ok_or_else(bad)? as u64, off[1].int().ok_or_else(bad)? as u64);
            let shape = v.get("shape").and_then(Json::arr).ok_or_else(bad)?.iter()
                .map(|d| d.int().map(|x| x as usize).ok_or_else(bad)).collect::<Result<Vec<_>, _>>()?;
            let dtype = v.get("dtype").and_then(Json::str).ok_or_else(bad)?.to_string();
            names.push(k.clone());
            infos.insert(k.clone(), Info { dtype, shape, start: 8 + n + a, end: 8 + n + b });
        }
        Ok(StFile { path: path.to_string(), f, names, infos })
    }

    /// Tensor names, sorted, without `__pad.*`.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    pub fn info(&self, name: &str) -> Result<&Info, String> {
        self.infos.get(name).ok_or_else(|| format!("{}: no tensor {name}", self.path))
    }

    /// `len` bytes of tensor `name` from byte `off` within it.
    pub fn read(&self, name: &str, off: u64, len: u64) -> Result<Vec<u8>, String> {
        let i = self.info(name)?;
        if off + len > i.end - i.start {
            return Err(format!("{}: read past the end of {name}", self.path));
        }
        let mut v = vec![0u8; len as usize];
        self.f.read_exact_at(&mut v, i.start + off).map_err(|e| format!("{}: {name}: {e}", self.path))?;
        Ok(v)
    }

    pub fn read_all(&self, name: &str) -> Result<Vec<u8>, String> {
        let i = self.info(name)?;
        self.read(name, 0, i.end - i.start)
    }
}

// ------------------------------------------------------------------------------------------------ number formats

pub fn f16_to_f32(h: u16) -> f32 {
    let s = ((h as u32) & 0x8000) << 16;
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as u32;
    let v = if e == 0 {
        m as f32 * (1.0 / 16777216.0)          // subnormal: m * 2^-24, exact
    } else if e == 31 {
        return f32::from_bits(s | 0x7f80_0000 | (m << 13));
    } else {
        return f32::from_bits(s | (((e + 112) as u32) << 23) | (m << 13));
    };
    f32::from_bits(s | v.to_bits())
}

/// The unbiased exponent of a finite f32 (f32 subnormals give -127 and lower values never matter here).
fn exponent(x: f32) -> i32 {
    ((x.to_bits() >> 23) & 0xff) as i32 - 127
}

/// |x| rounded to nearest even on a grid with `mant` fraction bits and exponents from `emin` up, sign kept.
fn round_to(x: f32, mant: i32, emin: i32) -> f32 {
    if x == 0.0 {
        return x;
    }
    let step = f64::powi(2.0, exponent(x).max(emin) - mant);
    ((x.abs() as f64 / step).round_ties_even() * step).copysign(x as f64) as f32
}

/// f32 -> f16, round to nearest even, as f32. Finite values inside the f16 range only (the codebook sums).
fn f32_to_f16_value(x: f32) -> f32 {
    round_to(x, 10, -14)
}

/// f32 -> the nearest E4M3FN value (round to nearest even, saturating at 448), as f32.
pub fn e4m3_round(x: f32) -> f32 {
    round_to(x.clamp(-448.0, 448.0), 3, -6).clamp(-448.0, 448.0)
}

/// E4M3FN byte -> f32 (exact).
pub fn e4m3_to_f32(b: u8) -> f32 {
    let e = ((b >> 3) & 15) as i32;
    let m = (b & 7) as f32;
    let v = if e == 0 { m * (1.0 / 512.0) } else if e == 15 && (b & 7) == 7 { f32::NAN } else { (1.0 + m / 8.0) * f32::powi(2.0, e - 7) };
    if b & 0x80 != 0 { -v } else { v }
}

/// 2^(code - 127) as f32, the E8M0 scale. Code 0 gives an f32 subnormal, exactly.
fn e8m0_to_f32(code: u8) -> f32 {
    let e = code as i32 - 127;
    if e >= -126 { f32::from_bits(((e + 127) as u32) << 23) } else { f32::from_bits(1u32 << (e + 149)) }
}

/// f32 -> BF16 bits, round to nearest even (torch's `.to(torch.bfloat16)`). Finite inputs.
pub fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

fn u16s(b: &[u8]) -> Vec<u16> {
    b.as_chunks::<2>().0.iter().map(|c| u16::from_le_bytes(*c)).collect()
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

// ------------------------------------------------------------------------------------------------ trellis codec

/// The mcg codebook for every 16-bit state, rounded to E4M3 (the rounding is part of the format).
fn codebook() -> &'static [f32] {
    static CB: OnceLock<Vec<f32>> = OnceLock::new();
    CB.get_or_init(|| {
        (0u32..65536).map(|s| {
            let x = (s.wrapping_mul(MCG) & 0x8FFF_8FFF) ^ 0x3B60_3B60;
            // two fp16 values in [0.125, 2): their f32 sum is exact, so rounding it to f16 is the fp16 add
            let sum = f16_to_f32(x as u16) + f16_to_f32((x >> 16) as u16);
            e4m3_round(f32_to_f16_value(sum))
        }).collect()
    })
}

/// perm[j] = row * 16 + col of the tile element at stream position j (exllamav3's tensor-core order); rows are the
/// input side.
fn tile_perm() -> &'static [usize; 256] {
    static P: OnceLock<[usize; 256]> = OnceLock::new();
    P.get_or_init(|| {
        let mut p = [0usize; 256];
        for t in 0..32 {
            let (r0, c0) = (2 * (t % 4), t / 4);
            let rc = [(r0, c0), (r0 + 1, c0), (r0 + 8, c0), (r0 + 9, c0),
                      (r0, c0 + 8), (r0 + 1, c0 + 8), (r0 + 8, c0 + 8), (r0 + 9, c0 + 8)];
            for (n, (r, c)) in rc.iter().enumerate() {
                p[8 * t + n] = 16 * r + c;
            }
        }
        p
    })
}

/// One packed 16x16 tile at width k (32k bytes) -> 256 values, row-major (input, output).
pub fn decode_tile(raw: &[u8], k: usize, out: &mut [f32; 256]) {
    debug_assert_eq!(raw.len(), 32 * k);
    let cb = codebook();
    let perm = tile_perm();
    let nw = 16 * k;
    // the words in stream order: each adjacent pair swapped
    let word = |i: usize| -> u32 {
        let w = i ^ 1;
        u16::from_le_bytes([raw[2 * w], raw[2 * w + 1]]) as u32
    };
    let mut vals = [0u32; 256];
    let mut acc: u64 = 0;
    let mut have = 0usize;
    let mut wi = 0usize;
    for v in vals.iter_mut() {
        while have < k {
            acc = (acc << 16) | word(wi) as u64;
            wi += 1;
            have += 16;
        }
        *v = ((acc >> (have - k)) & ((1u64 << k) - 1)) as u32;
        have -= k;
    }
    debug_assert_eq!(wi, nw);
    let reach = 16usize.div_ceil(k);
    for i in 0..256 {
        let mut s = 0u32;
        for j in 0..reach {
            s |= vals[(i + 256 - j) % 256] << (j * k);
        }
        out[perm[i]] = cb[(s & 0xffff) as usize];
    }
}

/// Tiles stored [k16][c = 0..7], as in gate/up halves and dense entries on the output side -> T (rows, 128), row-major.
pub fn tiles_out_major(raw: &[u8], k: usize, rows: usize) -> Vec<f32> {
    let tb = 32 * k;
    let mut t = vec![0f32; rows * 128];
    let mut tile = [0f32; 256];
    for k16 in 0..rows / 16 {
        for c in 0..8 {
            decode_tile(&raw[(k16 * 8 + c) * tb..][..tb], k, &mut tile);
            for r in 0..16 {
                t[(16 * k16 + r) * 128 + 16 * c..][..16].copy_from_slice(&tile[16 * r..16 * r + 16]);
            }
        }
    }
    t
}

/// Tiles stored [b][r = 0..7][c = 0..7], as in down halves and dense entries on the input side -> T (128, cols),
/// row-major.
pub fn tiles_in_major(raw: &[u8], k: usize, cols: usize) -> Vec<f32> {
    let tb = 32 * k;
    let mut t = vec![0f32; 128 * cols];
    let mut tile = [0f32; 256];
    for b in 0..cols / 128 {
        for r in 0..8 {
            for c in 0..8 {
                decode_tile(&raw[(64 * b + 8 * r + c) * tb..][..tb], k, &mut tile);
                for i in 0..16 {
                    t[(16 * r + i) * cols + 128 * b + 16 * c..][..16].copy_from_slice(&tile[16 * i..16 * i + 16]);
                }
            }
        }
    }
    t
}

/// Sylvester Hadamard matrix of order 128 scaled by 1/sqrt(128) in f32, as the reference builds it.
fn hadamard() -> &'static [f32] {
    static H: OnceLock<Vec<f32>> = OnceLock::new();
    H.get_or_init(|| {
        let c = (1.0f64 / 128f64.sqrt()) as f32;
        (0..128 * 128u32).map(|i| if ((i / 128) & (i % 128)).count_ones() % 2 == 1 { -c } else { c }).collect()
    })
}

/// X = (Hb T) * suh[row]: T (rows, cols) row-major, rows a multiple of 128; suh one value per row.
fn had_left(t: &[f32], rows: usize, cols: usize, suh: &[f32]) -> Vec<f32> {
    let h = hadamard();
    let mut x = vec![0f32; rows * cols];
    let mut acc = vec![0f32; cols];
    for b in 0..rows / 128 {
        for r in 0..128 {
            acc.fill(0.0);
            for q in 0..128 {
                let a = h[r * 128 + q];
                let tr = &t[(b * 128 + q) * cols..][..cols];
                for (y, &v) in acc.iter_mut().zip(tr) {
                    *y = a.mul_add(v, *y);
                }
            }
            let s = suh[b * 128 + r];
            for (o, &y) in x[(b * 128 + r) * cols..][..cols].iter_mut().zip(&acc) {
                *o = y * s;
            }
        }
    }
    x
}

/// W = (X Hb) * svh[col]: X (rows, cols) row-major, cols a multiple of 128; svh one value per column.
fn had_right(x: &mut [f32], rows: usize, cols: usize, svh: &[f32]) {
    let h = hadamard();
    let mut acc = [0f32; 128];
    for r in 0..rows {
        let row = &mut x[r * cols..][..cols];
        for b in 0..cols / 128 {
            acc.fill(0.0);
            for q in 0..128 {
                let a = row[b * 128 + q];
                for (y, &v) in acc.iter_mut().zip(&h[q * 128..][..128]) {
                    *y = a.mul_add(v, *y);
                }
            }
            for (c, &y) in acc.iter().enumerate() {
                row[b * 128 + c] = y * svh[b * 128 + c];
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ matrices

/// The 8 fields of `M.record`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Record {
    pub fmt: i64,
    pub out: usize,
    pub inp: usize,
    pub k: usize,
    pub side: i64,
    /// channels per shard on the sharded side
    pub unit: usize,
    pub rotation: i64,
    pub solve: i64,
}

/// One dense matrix of a dense or vocabulary file.
pub struct Matrix<'a> {
    pub f: &'a StFile,
    pub name: String,
    pub rec: Record,
}

impl<'a> Matrix<'a> {
    pub fn open(f: &'a StFile, name: &str) -> Result<Matrix<'a>, String> {
        let r = f.read_all(&format!("{name}.record"))?;
        if r.len() != 64 {
            return Err(format!("{}: {name}.record is not int64 [8]", f.path));
        }
        let v: Vec<i64> = r.as_chunks::<8>().0.iter().map(|c| i64::from_le_bytes(*c)).collect();
        let rec = Record { fmt: v[0], out: v[1] as usize, inp: v[2] as usize, k: v[3] as usize, side: v[4],
                           unit: v[5] as usize, rotation: v[6], solve: v[7] };
        if !(TRELLIS..=FP32).contains(&rec.fmt) {
            return Err(format!("{}: {name}: unknown format {}", f.path, rec.fmt));
        }
        Ok(Matrix { f, name: name.to_string(), rec })
    }

    /// The names of a file's matrices (every name with a `.record`), in sorted order.
    pub fn names(f: &StFile) -> Vec<String> {
        f.names().iter().filter_map(|n| n.strip_suffix(".record").map(str::to_string)).collect()
    }

    fn t(&self, s: &str) -> String {
        format!("{}.{s}", self.name)
    }

    fn check_shape(&self, s: &str, shape: &[usize]) -> Result<&Info, String> {
        let i = self.f.info(&self.t(s))?;
        if i.shape != shape {
            return Err(format!("{}: {}.{s} has shape {:?}, expected {shape:?}", self.f.path, self.name, i.shape));
        }
        Ok(i)
    }

    /// Rows `rows` of a row-major [out, w] tensor (w elements of `esz` bytes per row).
    fn rows_of(&self, s: &str, rows: &Range<usize>, w: usize, esz: usize) -> Result<Vec<u8>, String> {
        let rb = (w * esz) as u64;
        self.f.read(&self.t(s), rows.start as u64 * rb, rows.len() as u64 * rb)
    }

    /// The block `rows` x `cols` of the HF weight (out, in) as fp32, row-major. Trellis and side-2 int4 matrices need
    /// the block's range on their sharded side on 128-channel boundaries.
    pub fn decode(&self, rows: Range<usize>, cols: Range<usize>) -> Result<Vec<f32>, String> {
        let r = self.rec;
        if rows.end > r.out || cols.end > r.inp || rows.is_empty() || cols.is_empty() {
            return Err(format!("{}: {} block {rows:?} x {cols:?} is outside ({}, {})", self.f.path, self.name, r.out, r.inp));
        }
        let (nr, nc) = (rows.len(), cols.len());
        let pick = |row_major: &dyn Fn(usize, usize) -> f32| -> Vec<f32> {
            let mut o = Vec::with_capacity(nr * nc);
            for i in 0..nr {
                for j in cols.clone() {
                    o.push(row_major(i, j));
                }
            }
            o
        };
        match r.fmt {
            TRELLIS => self.decode_trellis(rows, cols),
            INT4 => {
                let (q4, s4) = self.int4(&rows, &cols)?;
                let (hw, gw) = (nc / 2, nc / 32);
                let mut o = vec![0f32; nr * nc];
                for i in 0..nr {
                    for j in 0..nc {
                        let b = q4[i * hw + j / 2];
                        let n = if j % 2 == 0 { b & 15 } else { b >> 4 } as i32 - 8;
                        o[i * nc + j] = n as f32 * f16_to_f32(s4[i * gw + j / 32]);
                    }
                }
                Ok(o)
            }
            MXFP8 => {
                self.check_shape("fp8", &[r.out, r.inp])?;
                self.check_shape("e8m0", &[r.out, r.inp / 32])?;
                let w = self.rows_of("fp8", &rows, r.inp, 1)?;
                let e = self.rows_of("e8m0", &rows, r.inp / 32, 1)?;
                Ok(pick(&|i, j| e4m3_to_f32(w[i * r.inp + j]) * e8m0_to_f32(e[i * (r.inp / 32) + j / 32])))
            }
            FP8_BLOCK => {
                let (bo, bi) = (r.out.div_ceil(128), r.inp.div_ceil(128));
                self.check_shape("fp8", &[r.out, r.inp])?;
                self.check_shape("scale", &[bo, bi])?;
                let w = self.rows_of("fp8", &rows, r.inp, 1)?;
                let s = f32s(&self.f.read_all(&self.t("scale"))?);
                Ok(pick(&|i, j| e4m3_to_f32(w[i * r.inp + j]) * s[(rows.start + i) / 128 * bi + j / 128]))
            }
            FP8_ROW => {
                self.check_shape("fp8", &[r.out, r.inp])?;
                self.check_shape("scale", &[r.out])?;
                let w = self.rows_of("fp8", &rows, r.inp, 1)?;
                let s = f32s(&self.rows_of("scale", &rows, 1, 4)?);
                Ok(pick(&|i, j| e4m3_to_f32(w[i * r.inp + j]) * s[i]))
            }
            BF16 => {
                self.check_shape("bf16", &[r.out, r.inp])?;
                let w = u16s(&self.rows_of("bf16", &rows, r.inp, 2)?);
                Ok(pick(&|i, j| bf16_to_f32(w[i * r.inp + j])))
            }
            _ => {
                self.check_shape("fp32", &[r.out, r.inp])?;
                let w = f32s(&self.rows_of("fp32", &rows, r.inp, 4)?);
                Ok(pick(&|i, j| w[i * r.inp + j]))
            }
        }
    }

    /// int4 codes and fp16 scale bits of the block, as [rows][cols / 2] and [rows][cols / 32] in the HF orientation.
    /// Input 2j sits in the low nibble of byte j. Side-2 storage comes back transposed to this layout. The columns
    /// must fall on 32-input groups, or 128-input blocks on side 2.
    pub fn int4(&self, rows: &Range<usize>, cols: &Range<usize>) -> Result<(Vec<u8>, Vec<u16>), String> {
        let r = self.rec;
        if r.fmt != INT4 {
            return Err(format!("{}: {} is not int4", self.f.path, self.name));
        }
        let (nr, nc) = (rows.len(), cols.len());
        if !cols.start.is_multiple_of(32) || !nc.is_multiple_of(32) {
            return Err(format!("{}: {} int4 columns {cols:?} are not on 32-input groups", self.f.path, self.name));
        }
        let mut q = vec![0u8; nr * nc / 2];
        let mut s = vec![0u16; nr * nc / 32];
        if r.side == SIDE_IN {
            if !cols.start.is_multiple_of(128) || !nc.is_multiple_of(128) {
                return Err(format!("{}: {} side-2 int4 columns {cols:?} are not on 128-input blocks", self.f.path, self.name));
            }
            self.check_shape("q4", &[r.inp / 128, r.out, 64])?;
            self.check_shape("s4", &[r.inp / 128, r.out, 4])?;
            for (bi, b) in (cols.start / 128..cols.end / 128).enumerate() {
                let qb = self.f.read(&self.t("q4"), (b * r.out * 64) as u64, (r.out * 64) as u64)?;
                let sb = u16s(&self.f.read(&self.t("s4"), (b * r.out * 8) as u64, (r.out * 8) as u64)?);
                for i in 0..nr {
                    let o = rows.start + i;
                    q[i * nc / 2 + bi * 64..][..64].copy_from_slice(&qb[o * 64..o * 64 + 64]);
                    s[i * nc / 32 + bi * 4..][..4].copy_from_slice(&sb[o * 4..o * 4 + 4]);
                }
            }
        } else {
            self.check_shape("q4", &[r.out, r.inp / 2])?;
            self.check_shape("s4", &[r.out, r.inp / 32])?;
            let qa = self.rows_of("q4", rows, r.inp / 2, 1)?;
            let sa = u16s(&self.rows_of("s4", rows, r.inp / 32, 2)?);
            for i in 0..nr {
                q[i * nc / 2..][..nc / 2].copy_from_slice(&qa[i * r.inp / 2 + cols.start / 2..][..nc / 2]);
                s[i * nc / 32..][..nc / 32].copy_from_slice(&sa[i * r.inp / 32 + cols.start / 32..][..nc / 32]);
            }
        }
        Ok((q, s))
    }

    /// MXFP8 codes and E8M0 scales of rows `rows`, as stored ([rows][in], [rows][in / 32]).
    pub fn mxfp8(&self, rows: &Range<usize>) -> Result<(Vec<u8>, Vec<u8>), String> {
        let r = self.rec;
        if r.fmt != MXFP8 {
            return Err(format!("{}: {} is not MXFP8", self.f.path, self.name));
        }
        self.check_shape("fp8", &[r.out, r.inp])?;
        self.check_shape("e8m0", &[r.out, r.inp / 32])?;
        Ok((self.rows_of("fp8", rows, r.inp, 1)?, self.rows_of("e8m0", rows, r.inp / 32, 1)?))
    }

    /// Byte range (in the file) of trellis entries `entries` and the entry size: raw tiles, no re-layout.
    pub fn trellis_entries(&self, entries: &Range<usize>) -> Result<(u64, u64), String> {
        let i = self.f.info(&self.t("trellis"))?;
        let eb = *i.shape.get(1).ok_or(format!("{}: {}.trellis is not 2-D", self.f.path, self.name))? as u64;
        if entries.end > i.shape[0] {
            return Err(format!("{}: {} has {} trellis entries, not {}", self.f.path, self.name, i.shape[0], entries.end));
        }
        Ok((i.start + entries.start as u64 * eb, eb))
    }

    fn decode_trellis(&self, rows: Range<usize>, cols: Range<usize>) -> Result<Vec<f32>, String> {
        let r = self.rec;
        let (inp, out, k) = (r.inp, r.out, r.k);
        if !(1..=8).contains(&k) {
            return Err(format!("{}: {} trellis width {k}", self.f.path, self.name));
        }
        let tb = 32 * k;
        let side_in = r.side == SIDE_IN;
        let (n_ent, ent_bytes) = if side_in { (inp / 128, out * 128 / 256 * tb) } else { (out / 128, inp * 128 / 256 * tb) };
        self.check_shape("trellis", &[n_ent, ent_bytes])?;
        self.check_shape("suh", &[inp])?;
        self.check_shape("svh", &[out])?;
        let suh: Vec<f32> = u16s(&self.f.read_all(&self.t("suh"))?).into_iter().map(f16_to_f32).collect();
        let svh: Vec<f32> = u16s(&self.f.read_all(&self.t("svh"))?).into_iter().map(f16_to_f32).collect();
        // the block's entries: HF columns on side 2 (input), HF rows otherwise (output)
        let sharded = if side_in { &cols } else { &rows };
        if !sharded.start.is_multiple_of(128) || !sharded.end.is_multiple_of(128) {
            return Err(format!("{}: {} trellis block {rows:?} x {cols:?} is not on 128-channel entries", self.f.path, self.name));
        }
        let ents = sharded.start / 128..sharded.end / 128;
        let (nr, nc) = (rows.len(), cols.len());
        let mut o = vec![0f32; nr * nc];
        let (off, eb) = self.trellis_entries(&ents)?;
        let i = self.f.info(&self.t("trellis"))?;
        let raw = self.f.read(&self.t("trellis"), off - i.start, ents.len() as u64 * eb)?;
        for (n, g) in ents.clone().enumerate() {
            let e = &raw[n * eb as usize..][..eb as usize];
            if side_in {
                // entry g: W rows (inputs) 128g.., all outputs; HF column c = W row
                let t = tiles_in_major(e, k, out);
                let mut x = had_left(&t, 128, out, &suh[128 * g..128 * g + 128]);
                had_right(&mut x, 128, out, &svh);
                for (ri, row) in rows.clone().enumerate() {
                    for c in 0..128 {
                        o[ri * nc + (128 * g + c - cols.start)] = x[c * out + row];
                    }
                }
            } else {
                // entry g: W columns (outputs) 128g.., all inputs; HF row = W column
                let t = tiles_out_major(e, k, inp);
                let mut x = had_left(&t, inp, 128, &suh);
                had_right(&mut x, inp, 128, &svh[128 * g..128 * g + 128]);
                for c in 0..128 {
                    let ri = 128 * g + c - rows.start;
                    for (ci, col) in cols.clone().enumerate() {
                        o[ri * nc + ci] = x[col * 128 + c];
                    }
                }
            }
        }
        Ok(o)
    }
}

/// `decode` split over `threads` threads along the sharded side (128-channel entries), for whole-matrix decodes.
pub fn decode_par(m: &Matrix, rows: Range<usize>, cols: Range<usize>, threads: usize) -> Result<Vec<f32>, String> {
    let along_cols = m.rec.side == SIDE_IN;
    let span = if along_cols { cols.clone() } else { rows.clone() };
    let step = if m.rec.fmt == TRELLIS || (m.rec.fmt == INT4 && along_cols) { 128 } else { 1 };
    let units = span.len() / step;
    let threads = threads.max(1).min(units.max(1));
    if threads == 1 || !span.len().is_multiple_of(step) {
        return m.decode(rows, cols);
    }
    let per = units.div_ceil(threads) * step;
    let parts: Vec<Range<usize>> = (span.start..span.end).step_by(per).map(|a| a..(a + per).min(span.end)).collect();
    let out: Vec<Result<Vec<f32>, String>> = std::thread::scope(|s| {
        let hs: Vec<_> = parts.iter().map(|p| {
            let (r, c) = if along_cols { (rows.clone(), p.clone()) } else { (p.clone(), cols.clone()) };
            s.spawn(move || m.decode(r, c))
        }).collect();
        hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("a decode thread panicked".into()))).collect()
    });
    let (nr, nc) = (rows.len(), cols.len());
    let mut o = vec![0f32; nr * nc];
    for (p, part) in parts.iter().zip(out) {
        let part = part?;
        if along_cols {
            for i in 0..nr {
                o[i * nc + (p.start - cols.start)..][..p.len()].copy_from_slice(&part[i * p.len()..][..p.len()]);
            }
        } else {
            o[(p.start - rows.start) * nc..][..p.len() * nc].copy_from_slice(&part);
        }
    }
    Ok(o)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn number_formats() {
        for b in 0..=255u8 {
            let v = e4m3_to_f32(b);
            if !v.is_nan() {
                assert_eq!(e4m3_round(v).to_bits(), v.to_bits(), "code {b:#x}");
            }
        }
        assert_eq!(e4m3_round(500.0), 448.0);
        assert_eq!(e4m3_round(-0.0).to_bits(), (-0.0f32).to_bits());
        assert_eq!(e4m3_round(1.0625), 1.0);          // tie to even
        assert_eq!(e4m3_round(1.1875), 1.25);         // tie to even
        assert_eq!(e4m3_round(0.0009765625), 0.0);    // 2^-10: half of the smallest subnormal, ties to 0
        assert_eq!(e8m0_to_f32(127), 1.0);
        assert_eq!(e8m0_to_f32(0), f32::from_bits(1 << 22));
        assert_eq!(bf16_bits(1.0 + 1.0 / 256.0), 0x3f80);    // tie to even
        assert_eq!(bf16_bits(1.0 + 3.0 / 256.0), 0x3f82);
        for h in [0u16, 0x3c00, 0xbc00, 0x0001, 0x03ff, 0x7bff, 0x3555] {
            assert_eq!(f32_to_f16_value(f16_to_f32(h)).to_bits(), f16_to_f32(h).to_bits());
        }
    }

    #[test]
    fn perm_is_a_permutation() {
        let mut p = tile_perm().to_vec();
        p.sort_unstable();
        assert_eq!(p, (0..256).collect::<Vec<_>>());
    }
}
