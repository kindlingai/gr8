//! A rank's whole image from the exlr8 v2 release alone, planned at boot: no gr8 rank pack. Every non-expert tensor
//! comes from the release's dense and vocabulary files (src/exlr8.rs), the routed and shared experts from the width
//! files at the index slices' offsets (src/v2plan.rs, as `--v2` plans them). The tensors and their names are the ones
//! the engine binds from a converted pack, in the forms it reads today:
//!
//!   exact (the release's bytes, or a re-layout of them):
//!     norms, router weight and bias, indexer weights_proj and k_norm: as stored (+ BF16 decode records)
//!     q_a, kv_a, indexer wq_b, MTP eh_proj: MXFP8 codes and scales as stored + MXFP8 decode records
//!     indexer wk: MXFP8 -> BF16 (exact: E4M3 times a power of two) + BF16 decode record
//!     q_b and o_proj stored as int4 g32 (layers 0, 2, 3): int4 decode records from the release's codes and scales
//!   BF16 of the release's value (what loader/exlr8.py --dtype bf16 gives):
//!     kv_b (FP8 128x128 blocks) -> UK / UV (GR8_KVB=fp8: the FP8 bytes and block scales as stored, #50); the embedding
//!     (FP8 per row)
//!   the trellis matrices (q_b, o_proj, the dense MLP of layers 0-2), by Opts::trellis:
//!     trellis: the rank's entries as stored + suh / svh, for the dense trellis kernels (engine/kernels/dtr.cuh) - exact
//!     rounded: MXFP8 (q_b, and every prefill copy) and int4 g32 decode records (o_proj, dense MLP) of the decoded weight
//!   rounded, either way: lm_head (MXFP8 decode record); the prefill copies of int4 q_b / o_proj (MXFP8)
//! The rounding is round to nearest (src/records.rs). The planner returns a table of which matrix took which form.
//! Sharding follows a layout (src/layout.rs), by default the formulas of tools/convert/plan.py: heads and dense
//! channels rotate by layer, vocab rows do not. A rank's heads and dense blocks need not be contiguous: each tensor
//! holds them in ascending order.

use crate::exlr8::{self, bf16_bits, Matrix, StFile, FP8_BLOCK, INT4, MXFP8, TRELLIS};
use crate::json::Json;
use crate::layout::{Group, Layout, Model};
use crate::plan::{Gen, Op, Plan};
use crate::records;
use crate::v2plan::{self, Builder, Slices, V2Image};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::Arc;

#[path = "flash.rs"]
mod flash;

const H: usize = 6144;
const L_MTP: usize = 78;
const VOCAB: usize = 154880;

/// Runs of consecutive pieces as channel ranges, `unit` channels a piece: [0, 1, 2, 9] x 256 is 0..768, 2304..2560.
fn runs(pieces: &[usize], unit: usize) -> Vec<Range<usize>> {
    let mut out: Vec<Range<usize>> = Vec::new();
    for &p in pieces {
        match out.last_mut() {
            Some(r) if r.end == p * unit => r.end += unit,
            _ => out.push(p * unit..(p + 1) * unit),
        }
    }
    out
}

fn total(runs: &[Range<usize>]) -> usize {
    runs.iter().map(Range::len).sum()
}

/// Blocks of the same rows, side by side: each block's rows are its length / `rows` wide.
fn cat_cols<T: Copy>(mut parts: Vec<Vec<T>>, rows: usize) -> Vec<T> {
    if parts.len() == 1 {
        return parts.pop().unwrap();
    }
    let mut o = Vec::with_capacity(parts.iter().map(Vec::len).sum());
    for i in 0..rows {
        for p in &parts {
            let w = p.len() / rows;
            o.extend_from_slice(&p[i * w..(i + 1) * w]);
        }
    }
    o
}

fn cat_rows<T: Copy>(mut parts: Vec<Vec<T>>) -> Vec<T> {
    if parts.len() == 1 { parts.pop().unwrap() } else { parts.concat() }
}

/// decode_par of the rows `rows` x the columns `cols`, each given as runs: row runs one after another, column runs
/// side by side.
fn decode_runs(m: &Matrix, rows: &[Range<usize>], cols: &[Range<usize>], t: usize) -> Result<Vec<f32>, String> {
    let mut out = Vec::new();
    for r in rows {
        let parts = cols.iter().map(|c| exlr8::decode_par(m, r.clone(), c.clone(), t)).collect::<Result<Vec<_>, _>>()?;
        out.push(cat_cols(parts, r.len()));
    }
    Ok(cat_rows(out))
}

/// Matrix::int4 over runs of rows and columns, as decode_runs.
fn int4_runs(m: &Matrix, rows: &[Range<usize>], cols: &[Range<usize>]) -> Result<(Vec<u8>, Vec<u16>), String> {
    let (mut q, mut s) = (Vec::new(), Vec::new());
    for r in rows {
        let (qs, ss): (Vec<_>, Vec<_>) = cols.iter().map(|c| m.int4(r, c)).collect::<Result<Vec<_>, _>>()?.into_iter().unzip();
        q.push(cat_cols(qs, r.len()));
        s.push(cat_cols(ss, r.len()));
    }
    Ok((cat_rows(q), cat_rows(s)))
}

fn le16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bf16v(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&x| bf16_bits(x)).collect()
}

/// One row of the exactness table: the matrix, its release format, its form in the image, and how the two relate.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Form {
    what: &'static str,
    src: String,
    form: &'static str,
    exact: &'static str,
}

const EXACT: &str = "exact";
const BF16_OF: &str = "BF16 of exact";
const ROUNDED: &str = "rounded";

fn fmt_name(m: &Matrix) -> String {
    match m.rec.fmt {
        TRELLIS => format!("trellis K{}", m.rec.k),
        INT4 => "int4 g32".into(),
        MXFP8 => "MXFP8".into(),
        FP8_BLOCK => "FP8 block".into(),
        exlr8::FP8_ROW => "FP8 row".into(),
        exlr8::BF16 => "BF16".into(),
        _ => "FP32".into(),
    }
}

struct Planner<'a> {
    b: Builder,
    layout: &'a Layout,
    rank: usize,
    threads: usize,
    forms: BTreeMap<Form, Vec<usize>>,
    /// add every layer's full-head q_b / o_proj / UK / UV (GR8_PF_OWN=3)
    qo: bool,
    /// trellis matrices as the release's entries (the dense trellis kernels); false: rounded to MXFP8 / int4
    trellis: bool,
    /// kv_b (UK / UV) as the release's block FP8 bytes and scales (#50); false: BF16
    kvb_fp8: bool,
}

/// A tensor of a planned group: name, manifest dtype, shape, bytes.
type Spec = (String, &'static str, Vec<i64>);

fn nbytes(dtype: &str, shape: &[i64]) -> u64 {
    let e = match dtype { "bf16" | "f16" => 2, "f32" => 4, "i64" => 8, _ => 1 };
    shape.iter().product::<i64>() as u64 * e
}

impl Planner<'_> {
    fn note(&mut self, l: usize, f: Form) {
        self.forms.entry(f).or_default().push(l);
    }

    /// The pieces of `g` this rank holds as (first, count), for GLM-5.3-Flash's planner (flash.rs), which reads one
    /// contiguous chunk a rank as the formulas give it.
    fn chunk(&self, g: Group) -> Result<(usize, usize), String> {
        match runs(&self.layout.held(g, self.rank)?, 1).as_slice() {
            [r] => Ok((r.start, r.len())),
            _ => Err(format!("{g}: GLM-5.3-Flash plans only layouts whose pieces are contiguous, as the formulas' are")),
        }
    }

    /// A tensor copied from a file range.
    fn copy(&mut self, name: String, dtype: &'static str, shape: Vec<i64>, f: &StFile, t: &str, off: u64) -> Result<(), String> {
        let n = nbytes(dtype, &shape);
        self.copy_runs(name, dtype, shape, f, t, &[(off, n)])
    }

    /// A tensor copied from ranges of file tensor `t`, one after another: (offset in t, bytes).
    fn copy_runs(&mut self, name: String, dtype: &'static str, shape: Vec<i64>, f: &StFile, t: &str, runs: &[(u64, u64)])
                 -> Result<(), String> {
        let i = f.info(t)?;
        let n = nbytes(dtype, &shape);
        if runs.iter().map(|r| r.1).sum::<u64>() != n || runs.iter().any(|&(o, k)| i.start + o + k > i.end) {
            return Err(format!("{}: {t} is smaller than {name}", f.path));
        }
        let mut o = self.b.begin(name, dtype, shape, n, None);
        let fi = self.b.file(f.path.clone());
        for &(off, k) in runs {
            self.b.seg(o, k, Op::Copy, fi, i.start + off);
            o += k;
        }
        Ok(())
    }

    /// Consecutive tensors computed together at load: `make` returns each tensor's bytes, in order.
    fn gen(&mut self, what: String, specs: Vec<Spec>, make: impl Fn() -> Result<Vec<Vec<u8>>, String> + Send + Sync + 'static) {
        let mut offs = Vec::new();
        for (name, dtype, shape) in &specs {
            let n = nbytes(dtype, shape);
            offs.push((self.b.begin(name.clone(), dtype, shape.clone(), n, None), n));
        }
        let (a, end) = (offs[0].0, offs.last().map(|&(o, n)| o + n).unwrap());
        let gi = self.b.gens.len();
        let span = end - a;
        self.b.gens.push(Gen::new(what, span, Box::new(move || {
            let parts = make()?;
            let mut v = vec![0u8; span as usize];
            for ((o, n), p) in offs.iter().zip(parts) {
                if p.len() as u64 != *n {
                    return Err(format!("a part of {} bytes, {n} planned", p.len()));
                }
                v[(o - a) as usize..(o - a + n) as usize].copy_from_slice(&p);
            }
            Ok(v)
        })));
        self.b.segs.push(crate::plan::Seg { pack_off: a, nbytes: span, op: Op::Gen, file_off: 0, file: gi });
    }

    /// A single manifest tensor split into contiguous `Op::Gen` segments so loader worker threads compute stripes in parallel.
    fn gen_striped(
        &mut self,
        what: &str,
        name: String,
        dtype: &'static str,
        shape: Vec<i64>,
        stripes: Vec<(u64, Box<dyn Fn() -> Result<Vec<u8>, String> + Send + Sync>)>,
    ) {
        let total = nbytes(dtype, &shape);
        let base = self.b.begin(name, dtype, shape, total, None);
        let mut off = 0u64;
        for (span, make) in stripes {
            let gi = self.b.gens.len();
            self.b.gens.push(Gen::new(what.to_string(), span, make));
            self.b.segs.push(crate::plan::Seg { pack_off: base + off, nbytes: span, op: Op::Gen, file_off: 0, file: gi });
            off += span;
        }
        assert_eq!(off, total);
    }

    /// A vector (norm, bias) as stored, under its gr8 name.
    fn vector(&mut self, name: String, f: &StFile, t: &str) -> Result<(), String> {
        let i = f.info(t)?.clone();
        let dtype = match i.dtype.as_str() { "BF16" => "bf16", "F32" => "f32", "F16" => "f16", d => return Err(format!("{t}: dtype {d}")) };
        self.copy(name, dtype, i.shape.iter().map(|&d| d as i64).collect(), f, t, 0)
    }

    /// An MXFP8 matrix as stored (all rows): codes + scales copied (unless `decode_only`), and the decode record.
    fn mxfp8(&mut self, l: usize, what: &'static str, p: &str, f: &Arc<StFile>, m: &str, decode_only: bool) -> Result<(), String> {
        let mx = Matrix::open(f, m)?;
        if mx.rec.fmt != MXFP8 {
            return Err(format!("{}: {m} is {}, gr8 reads it as MXFP8", f.path, fmt_name(&mx)));
        }
        let (o, k) = (mx.rec.out, mx.rec.inp);
        if !decode_only {
            self.copy(format!("{p}.mx.codes"), "u8", vec![o as i64, k as i64], f, &format!("{m}.fp8"), 0)?;
            self.copy(format!("{p}.mx.scales"), "u8", vec![o as i64, (k / 32) as i64], f, &format!("{m}.e8m0"), 0)?;
        }
        let (f2, m2) = (f.clone(), m.to_string());
        self.gen(format!("{p}.mx.gv"), vec![(format!("{p}.mx.gv"), "u8", vec![(o / 16) as i64, (k / 32) as i64, 528])], move || {
            let mx = Matrix::open(&f2, &m2)?;
            let (c, s) = mx.mxfp8(&(0..o))?;
            Ok(vec![records::mxfp8_record(&c, &s, k)])
        });
        self.note(l, Form { what, src: "MXFP8".into(), form: if decode_only { "MXFP8 decode record" } else { "MXFP8 + decode record" }, exact: EXACT });
        Ok(())
    }

    /// A BF16 matrix as stored + its BF16 decode record.
    fn bf16(&mut self, l: usize, what: &'static str, p: &str, f: &Arc<StFile>, m: &str) -> Result<(), String> {
        let mx = Matrix::open(f, m)?;
        let (o, k) = (mx.rec.out, mx.rec.inp);
        match mx.rec.fmt {
            exlr8::BF16 => {
                self.copy(p.to_string(), "bf16", vec![o as i64, k as i64], f, &format!("{m}.bf16"), 0)?;
                self.note(l, Form { what, src: "BF16".into(), form: "BF16 + decode record", exact: EXACT });
            }
            MXFP8 => {
                let (f2, m2) = (f.clone(), m.to_string());
                let t = self.threads;
                self.gen(p.to_string(), vec![(p.to_string(), "bf16", vec![o as i64, k as i64])], move || {
                    let w = exlr8::decode_par(&Matrix::open(&f2, &m2)?, 0..o, 0..k, t)?;
                    Ok(vec![le16(&bf16v(&w))])
                });
                self.note(l, Form { what, src: "MXFP8".into(), form: "BF16 + decode record", exact: EXACT });
            }
            _ => return Err(format!("{}: {m} is {}, gr8 reads it as BF16", f.path, fmt_name(&mx))),
        }
        let (f2, m2) = (f.clone(), m.to_string());
        let t = self.threads;
        self.gen(format!("{p}.gv"), vec![(format!("{p}.gv"), "u8", vec![(o / 16) as i64, (k / 16) as i64, 512])], move || {
            let w = exlr8::decode_par(&Matrix::open(&f2, &m2)?, 0..o, 0..k, t)?;
            Ok(vec![records::bf16_record(&bf16v(&w), k)])
        });
        Ok(())
    }

    fn layer(&mut self, l: usize, dir: &str) -> Result<(), String> {
        let f = Arc::new(StFile::open(&exlr8::dense_path(dir, Some(l)))?);
        let p = format!("L{l:03}");
        let mtp = l == L_MTP;
        let heads = self.layout.held(Group::Heads(l), self.rank)?;
        let nh = heads.len();
        let t = self.threads;
        self.vector(format!("{p}.attn_norm"), &f, "input_layernorm.weight")?;
        // q_a has only its MXFP8 record. Decode reads it as the MTP layer always did
        self.mxfp8(l, "q_a", &format!("{p}.q_a"), &f, "self_attn.q_a_proj", false)?;
        self.vector(format!("{p}.q_a_norm"), &f, "self_attn.q_a_layernorm.weight")?;
        self.mxfp8(l, "kv_a", &format!("{p}.kv_a"), &f, "self_attn.kv_a_proj_with_mqa", false)?;
        self.vector(format!("{p}.kv_a_norm"), &f, "self_attn.kv_a_layernorm.weight")?;

        // q_b: this rank's heads, output rows h*256..
        let qb = Matrix::open(&f, "self_attn.q_b_proj")?;
        let rows = runs(&heads, 256);
        if qb.rec.fmt == TRELLIS && self.trellis {
            self.trellis(l, "q_b", &p, &qb, &rows)?;
        } else {
            self.q_b_rounded(l, &p, &f, &qb, rows.clone());
        }

        // kv_b: heads' rows of [64 x 448][512]; per head the first 192 rows are UK, the other 256 UV
        let kvb = Matrix::open(&f, "self_attn.kv_b_proj")?;
        if self.kvb_fp8 {
            // the FP8 form takes a contiguous head range (kv_b_fp8): a planned layout's heads may not be
            let h0 = heads[0];
            if heads.iter().enumerate().any(|(i, &h)| h != h0 + i) {
                return Err(format!("L{l:03}: GR8_KVB=fp8 needs this rank's heads in one run; the layout gives {heads:?}"));
            }
            self.kv_b_fp8(l, &p, &f, &kvb, h0, nh)?;
        } else {
            self.note(l, Form { what: "kv_b", src: fmt_name(&kvb), form: "BF16 (UK, UV)",
                                exact: if kvb.rec.fmt == FP8_BLOCK { BF16_OF } else { ROUNDED } });
            let (f2, kv_rows) = (f.clone(), runs(&heads, 448));
            self.gen(format!("{p}.kv_b"), vec![
                (format!("{p}.kv_b.uk"), "bf16", vec![nh as i64, 192, 512]),
                (format!("{p}.kv_b.uv"), "bf16", vec![nh as i64, 256, 512])], move || {
                let w = bf16v(&decode_runs(&Matrix::open(&f2, "self_attn.kv_b_proj")?, &kv_rows, &[0..512], t)?);
                let (mut uk, mut uv) = (Vec::new(), Vec::new());
                for h in 0..nh {
                    uk.extend_from_slice(&w[h * 448 * 512..][..192 * 512]);
                    uv.extend_from_slice(&w[(h * 448 + 192) * 512..][..256 * 512]);
                }
                Ok(vec![le16(&uk), le16(&uv)])
            });
        }

        // o_proj: this rank's heads' input columns
        let op = Matrix::open(&f, "self_attn.o_proj")?;
        let cols = rows;
        if op.rec.fmt == TRELLIS && self.trellis {
            self.trellis(l, "o_proj", &p, &op, &cols)?;
        } else {
            self.o_proj_rounded(l, &p, &f, &op, cols);
        }

        if f.info("self_attn.indexer.wq_b.record").is_ok() {
            self.mxfp8(l, "indexer wq_b", &format!("{p}.idx.wq_b"), &f, "self_attn.indexer.wq_b", false)?;
            self.bf16(l, "indexer wk", &format!("{p}.idx.wk"), &f, "self_attn.indexer.wk")?;
            self.vector(format!("{p}.idx.k_norm.w"), &f, "self_attn.indexer.k_norm.weight")?;
            self.vector(format!("{p}.idx.k_norm.b"), &f, "self_attn.indexer.k_norm.bias")?;
            self.bf16(l, "indexer weights_proj", &format!("{p}.idx.weights_proj"), &f, "self_attn.indexer.weights_proj")?;
        }
        self.vector(format!("{p}.mlp_norm"), &f, "post_attention_layernorm.weight")?;
        if l >= 3 {
            self.bf16(l, "router", &format!("{p}.router.w"), &f, "mlp.gate")?;
            self.vector(format!("{p}.router.bias"), &f, "mlp.gate.e_score_correction_bias")?;
        } else if self.trellis && ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"].iter()
            .all(|m| Matrix::open(&f, m).is_ok_and(|x| x.rec.fmt == TRELLIS)) {
            let ch = runs(&self.layout.held(Group::Dense(l), self.rank)?, 128);
            for (n, m) in [("dense.gate", "mlp.gate_proj"), ("dense.up", "mlp.up_proj"), ("dense.down", "mlp.down_proj")] {
                self.trellis(l, n, &p, &Matrix::open(&f, m)?, &ch)?;
            }
        } else {
            self.dense_mlp(l, &p, &f)?;
        }
        if self.qo && !mtp {
            self.full_heads(l, &p, &f);
        }
        if mtp {
            self.mxfp8(l, "MTP eh_proj", &format!("{p}.eh"), &f, "eh_proj", false)?;   // MTP prefill reads the codes too
            self.vector(format!("{p}.enorm"), &f, "enorm.weight")?;
            self.vector(format!("{p}.hnorm"), &f, "hnorm.weight")?;
            self.vector(format!("{p}.shared_head_norm"), &f, "shared_head.norm.weight")?;
        }
        Ok(())
    }

    /// q_b as MXFP8 (prefill copy + decode record) of the decoded weight. An int4 q_b also gets its exact int4 record.
    fn q_b_rounded(&mut self, l: usize, p: &str, f: &Arc<StFile>, qb: &Matrix, rows: Vec<Range<usize>>) {
        let int4 = qb.rec.fmt == INT4;
        self.note(l, Form { what: "q_b", src: fmt_name(qb), form: "MXFP8 + decode record", exact: ROUNDED });
        if int4 {
            self.note(l, Form { what: "q_b", src: fmt_name(qb), form: "int4 g32 decode record", exact: EXACT });
        }
        let (n, t, f2) = (total(&rows), self.threads, f.clone());
        let mut specs = vec![
            (format!("{p}.q_b.mx.codes"), "u8", vec![n as i64, 2048]),
            (format!("{p}.q_b.mx.scales"), "u8", vec![n as i64, 64]),
            (format!("{p}.q_b.mx.gv"), "u8", vec![(n / 16) as i64, 64, 528])];
        if int4 {
            specs.push((format!("{p}.q_b.gptq"), "u8", vec![(n / 16) as i64, 64, 288]));
        }
        self.gen(format!("{p}.q_b"), specs, move || {
            let m = Matrix::open(&f2, "self_attn.q_b_proj")?;
            let w = decode_runs(&m, &rows, &[0..2048], t)?;
            let (c, s) = records::mxfp8_quantize(&w, 2048);
            let gv = records::mxfp8_record(&c, &s, 2048);
            let mut out = vec![c, s, gv];
            if int4 {
                let (q, qs) = int4_runs(&m, &rows, &[0..2048])?;
                out.push(records::rt_record(&q, &qs, 2048));
            }
            Ok(out)
        });
    }

    /// o_proj for today's kernels: decode reads an int4 record (MXFP8 in the MTP layer), prefill an MXFP8 copy. The
    /// int4 record is the release's own where o_proj is stored as int4, else rounded like the MXFP8 copy.
    fn o_proj_rounded(&mut self, l: usize, p: &str, f: &Arc<StFile>, op: &Matrix, cols: Vec<Range<usize>>) {
        let (mtp, int4, k, t) = (l == L_MTP, op.rec.fmt == INT4, total(&cols), self.threads);
            if !mtp {
                self.note(l, Form { what: "o_proj decode", src: fmt_name(op), form: "int4 g32 decode record", exact: if int4 { EXACT } else { ROUNDED } });
            }
            self.note(l, Form { what: "o_proj prefill", src: fmt_name(op), form: if mtp { "MXFP8 + decode record" } else { "MXFP8" }, exact: ROUNDED });
            let mut specs = Vec::new();
            if !mtp {
                specs.push((format!("{p}.o_proj.gptq"), "u8", vec![384, (k / 32) as i64, 288]));
            }
            specs.push((format!("{p}.o_proj.mx.codes"), "u8", vec![H as i64, k as i64]));
            specs.push((format!("{p}.o_proj.mx.scales"), "u8", vec![H as i64, (k / 32) as i64]));
            if mtp {
                specs.push((format!("{p}.o_proj.mx.gv"), "u8", vec![384, (k / 32) as i64, 528]));
            }
            let f2 = f.clone();
            self.gen(format!("{p}.o_proj"), specs, move || {
                let m = Matrix::open(&f2, "self_attn.o_proj")?;
                let w = decode_runs(&m, &[0..H], &cols, t)?;
                let (c, s) = records::mxfp8_quantize(&w, k);
                let mut out = Vec::new();
                if !mtp {
                    let (q, qs) = if int4 { int4_runs(&m, &[0..H], &cols)? } else { records::int4_quantize(&w, k) };
                    out.push(records::rt_record(&q, &qs, k));
                }
                let gv = if mtp { Some(records::mxfp8_record(&c, &s, k)) } else { None };
                out.push(c);
                out.push(s);
                out.extend(gv);
                Ok(out)
            });
    }

    /// A trellis matrix as the dense trellis kernels read it (engine/kernels/dtr.cuh layer_mats): the rank's entries of
    /// M.trellis as stored, `ch` on the sharded side (runs of 128-channel entries), and suh / svh sliced on that side.
    fn trellis(&mut self, l: usize, what: &str, p: &str, m: &Matrix, ch: &[Range<usize>]) -> Result<(), String> {
        let r = m.rec;
        self.note(l, Form { what: match what { "q_b" => "q_b", "o_proj" => "o_proj", w if w.starts_with("dense.") => "dense MLP", _ => "KDA projection" }, src: fmt_name(m),
                            form: "trellis entries (dense trellis kernels)", exact: EXACT });
        let t = format!("{}.trellis", m.name);
        let start = m.f.info(&t)?.start;
        let (mut ents, mut eb, mut n) = (Vec::new(), 0, 0);
        for c in ch {
            let e = c.start / 128..c.end / 128;
            let (off, b) = m.trellis_entries(&e)?;
            ents.push((off - start, e.len() as u64 * b));
            (eb, n) = (b, n + e.len());
        }
        self.copy_runs(format!("{p}.{what}.trellis"), "u8", vec![n as i64, eb as i64], m.f, &t, &ents)?;
        // f16 vectors: the sharded side's runs, the other side whole
        let side = |sharded: bool, all: usize| -> Vec<(u64, u64)> {
            if sharded { ch.iter().map(|c| (2 * c.start as u64, 2 * c.len() as u64)).collect() } else { vec![(0, 2 * all as u64)] }
        };
        let (suh, svh) = (side(r.side == exlr8::SIDE_IN, r.inp), side(r.side != exlr8::SIDE_IN, r.out));
        let len = |v: &[(u64, u64)]| v.iter().map(|x| x.1 as i64 / 2).sum::<i64>();
        self.copy_runs(format!("{p}.{what}.suh"), "f16", vec![len(&suh)], m.f, &format!("{}.suh", m.name), &suh)?;
        self.copy_runs(format!("{p}.{what}.svh"), "f16", vec![len(&svh)], m.f, &format!("{}.svh", m.name), &svh)
    }

    /// kv_b of heads h0 .. h0 + nh as the release's block FP8 (#50, Opts::kvb_fp8): L.kv_b.uk / .uv hold the E4M3 bytes of
    /// the UK / UV rows ([nh][192][512] / [nh][256][512], the BF16 tensors' layout) and L.kv_b.uk.scale / .uv.scale each
    /// row's scales ([nh][192][4] / [nh][256][4] f32: its 128-row block's scale for each 128-input block). The engine reads
    /// weight (i, j) as BF16(e4m3 x scale), the value of the BF16 form, so the two forms give the same outputs.
    fn kv_b_fp8(&mut self, l: usize, p: &str, f: &Arc<StFile>, kvb: &Matrix, h0: usize, nh: usize) -> Result<(), String> {
        if kvb.rec.fmt != FP8_BLOCK || kvb.rec.out != 64 * 448 || kvb.rec.inp != 512 {
            return Err(format!("{}: kv_b is {} [{} x {}]: GR8_KVB=fp8 needs the release's block FP8 [28672 x 512]",
                               f.path, fmt_name(kvb), kvb.rec.out, kvb.rec.inp));
        }
        self.note(l, Form { what: "kv_b", src: fmt_name(kvb), form: "FP8 + block scales (UK, UV; GR8_KVB=fp8)", exact: EXACT });
        let f2 = f.clone();
        self.gen(format!("{p}.kv_b"), vec![
            (format!("{p}.kv_b.uk"), "u8", vec![nh as i64, 192, 512]),
            (format!("{p}.kv_b.uv"), "u8", vec![nh as i64, 256, 512]),
            (format!("{p}.kv_b.uk.scale"), "f32", vec![nh as i64, 192, 4]),
            (format!("{p}.kv_b.uv.scale"), "f32", vec![nh as i64, 256, 4])], move || {
            let (w, s) = Matrix::open(&f2, "self_attn.kv_b_proj")?.fp8_rows(&(h0 * 448..(h0 + nh) * 448))?;
            let (mut uk, mut uv, mut uks, mut uvs) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for h in 0..nh {
                uk.extend_from_slice(&w[h * 448 * 512..][..192 * 512]);
                uv.extend_from_slice(&w[(h * 448 + 192) * 512..][..256 * 512]);
                uks.extend_from_slice(&s[h * 448 * 4..][..192 * 4]);
                uvs.extend_from_slice(&s[(h * 448 + 192) * 4..][..256 * 4]);
            }
            let le32 = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
            Ok(vec![uk, uv, le32(&uks), le32(&uvs)])
        });
        Ok(())
    }

    /// GR8_PF_OWN=3's full-head weights of layer l, the full-head QO tensors:
    /// q_b and o_proj as MXFP8 for all 64 heads, UK and UV in BF16. MXFP8 scales cover 32 inputs of one row, so the
    /// whole matrix rounds to the concatenation of the rank slices.
    fn full_heads(&mut self, l: usize, p: &str, f: &Arc<StFile>) {
        self.note(l, Form { what: "q_b, o_proj all heads (GR8_PF_OWN=3)", src: "as above".into(), form: "MXFP8", exact: ROUNDED });
        let (f2, t) = (f.clone(), self.threads);
        let cache = qo_cache_path(p);
        self.gen(format!("{p}.qo"), vec![
            (format!("{p}.qo.q_b.mx.codes"), "u8", vec![16384, 2048]),
            (format!("{p}.qo.q_b.mx.scales"), "u8", vec![16384, 64]),
            (format!("{p}.qo.o_proj.mx.codes"), "u8", vec![H as i64, 16384]),
            (format!("{p}.qo.o_proj.mx.scales"), "u8", vec![H as i64, 512]),
            (format!("{p}.qo.kv_b.uk"), "bf16", vec![64, 192, 512]),
            (format!("{p}.qo.kv_b.uv"), "bf16", vec![64, 256, 512])], move || {
            let key = qo_cache_key(&f2.path);
            if let (Some(c), Some(k)) = (&cache, &key) {
                if let Some(parts) = qo_cache_read(c, k, 6) {
                    return Ok(parts);
                }
            }
            let parts = full_heads_make(&f2, t)?;
            if let (Some(c), Some(k)) = (&cache, &key) {
                qo_cache_write(c, k, &parts);
            }
            Ok(parts)
        });
    }
}

/// GR8_QO_CACHE=DIR: the full-head GR8_PF_OWN=3 tensors of a layer are the same bytes on every rank and every boot
/// (decoded from the release and rounded to MXFP8 by the same code), so the first boot stores them under DIR and later
/// boots read them back instead of decoding the whole q_b / o_proj / kv_b again. Key = source file path, size and
/// mtime; a missing, short or mismatched file is ignored (the tensors are made and the file rewritten).
fn qo_cache_path(p: &str) -> Option<String> {
    let d = gr8knobs::QO_CACHE.get()?;
    Some(format!("{d}/{}.qo.bin", p.replace('/', "_")))
}

fn qo_cache_key(src: &str) -> Option<String> {
    let m = std::fs::metadata(src).ok()?;
    let t = m.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?;
    Some(format!("{src}|{}|{}.{:09}", m.len(), t.as_secs(), t.subsec_nanos()))
}

const QO_MAGIC: &[u8; 8] = b"G6QOC001";

fn qo_cache_read(path: &str, key: &str, nparts: usize) -> Option<Vec<Vec<u8>>> {
    let b = std::fs::read(path).ok()?;
    let u64at = |o: usize| -> Option<u64> { Some(u64::from_le_bytes(b.get(o..o + 8)?.try_into().ok()?)) };
    if b.get(..8)? != QO_MAGIC {
        return None;
    }
    let kl = u64at(8)? as usize;
    if b.get(16..16 + kl)? != key.as_bytes() {
        return None;
    }
    let mut o = 16 + kl;
    let mut lens = Vec::with_capacity(nparts);
    for _ in 0..nparts {
        lens.push(u64at(o)? as usize);
        o += 8;
    }
    let mut parts = Vec::with_capacity(nparts);
    for n in lens {
        parts.push(b.get(o..o + n)?.to_vec());
        o += n;
    }
    if o != b.len() {
        return None;
    }
    Some(parts)
}

fn qo_cache_write(path: &str, key: &str, parts: &[Vec<u8>]) {
    let mut b = Vec::with_capacity(16 + key.len() + parts.iter().map(|p| 8 + p.len()).sum::<usize>());
    b.extend_from_slice(QO_MAGIC);
    b.extend_from_slice(&(key.len() as u64).to_le_bytes());
    b.extend_from_slice(key.as_bytes());
    for p in parts {
        b.extend_from_slice(&(p.len() as u64).to_le_bytes());
    }
    for p in parts {
        b.extend_from_slice(p);
    }
    let tmp = format!("{path}.tmp{}", std::process::id());
    if std::fs::write(&tmp, &b).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    } else {
        let _ = std::fs::remove_file(&tmp);
    }
}

/// GR8_PF_OWN=3's full-head tensors of one layer: [q_b codes, q_b scales, o_proj codes, o_proj scales, UK, UV].
fn full_heads_make(f2: &StFile, t: usize) -> Result<Vec<Vec<u8>>, String> {
    {
        {
            let q = exlr8::decode_par(&Matrix::open(f2, "self_attn.q_b_proj")?, 0..16384, 0..2048, t)?;
            let (qc, qs) = records::mxfp8_quantize(&q, 2048);
            drop(q);
            let o = exlr8::decode_par(&Matrix::open(f2, "self_attn.o_proj")?, 0..H, 0..16384, t)?;
            let (oc, os) = records::mxfp8_quantize(&o, 16384);
            drop(o);
            let w = bf16v(&exlr8::decode_par(&Matrix::open(f2, "self_attn.kv_b_proj")?, 0..64 * 448, 0..512, t)?);
            let (mut uk, mut uv) = (Vec::new(), Vec::new());
            for h in 0..64 {
                uk.extend_from_slice(&w[h * 448 * 512..][..192 * 512]);
                uv.extend_from_slice(&w[(h * 448 + 192) * 512..][..256 * 512]);
            }
            Ok(vec![qc, qs, oc, os, le16(&uk), le16(&uv)])
        }
    }
}

impl Planner<'_> {

    /// Layers 0-2: this rank's channels of gate/up (rows) and down (columns), int4 records for decode and MXFP8
    /// copies for prefill, both rounded from the decoded weights.
    fn dense_mlp(&mut self, l: usize, p: &str, f: &Arc<StFile>) -> Result<(), String> {
        let ch = runs(&self.layout.held(Group::Dense(l), self.rank)?, 128);
        let i = total(&ch);
        for m in ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"] {
            let mx = Matrix::open(f, m)?;
            self.note(l, Form { what: "dense MLP", src: fmt_name(&mx), form: "int4 g32 decode records + MXFP8", exact: ROUNDED });
        }
        let (f2, t) = (f.clone(), self.threads);
        self.gen(format!("{p}.dense"), vec![
            (format!("{p}.dense.gptq.fc1"), "u8", vec![(i / 16) as i64, 192, 576]),
            (format!("{p}.dense.gptq.fc2"), "u8", vec![384, (i / 32) as i64, 288]),
            (format!("{p}.dense.gate.mx.codes"), "u8", vec![i as i64, H as i64]),
            (format!("{p}.dense.gate.mx.scales"), "u8", vec![i as i64, (H / 32) as i64]),
            (format!("{p}.dense.up.mx.codes"), "u8", vec![i as i64, H as i64]),
            (format!("{p}.dense.up.mx.scales"), "u8", vec![i as i64, (H / 32) as i64]),
            (format!("{p}.dense.down.mx.codes"), "u8", vec![H as i64, i as i64]),
            (format!("{p}.dense.down.mx.scales"), "u8", vec![H as i64, (i / 32) as i64])], move || {
            let g = decode_runs(&Matrix::open(&f2, "mlp.gate_proj")?, &ch, &[0..H], t)?;
            let u = decode_runs(&Matrix::open(&f2, "mlp.up_proj")?, &ch, &[0..H], t)?;
            let d = decode_runs(&Matrix::open(&f2, "mlp.down_proj")?, &[0..H], &ch, t)?;
            let (gq, gs) = records::int4_quantize(&g, H);
            let (uq, us) = records::int4_quantize(&u, H);
            let (dq, ds) = records::int4_quantize(&d, i);
            let (fc1, fc2) = records::rt_expert_record((&gq, &gs), (&uq, &us), (&dq, &ds), H, i);
            let (gc, gsc) = records::mxfp8_quantize(&g, H);
            let (uc, usc) = records::mxfp8_quantize(&u, H);
            let (dc, dsc) = records::mxfp8_quantize(&d, i);
            Ok(vec![fc1, fc2, gc, gsc, uc, usc, dc, dsc])
        });
        Ok(())
    }

    fn vocab(&mut self, dir: &str) -> Result<(), String> {
        let f = Arc::new(StFile::open(&exlr8::dense_path(dir, None))?);
        let t = self.threads;
        let em = Matrix::open(&f, "embed_tokens")?;
        self.note(L_MTP + 1, Form { what: "embedding", src: fmt_name(&em), form: "BF16",
                                     exact: if em.rec.fmt == exlr8::BF16 { EXACT } else { BF16_OF } });
        let f2 = f.clone();
        self.gen("G.embed".into(), vec![("G.embed".into(), "bf16", vec![VOCAB as i64, H as i64])], move || {
            let m = Matrix::open(&f2, "embed_tokens")?;
            let mut out = Vec::with_capacity(VOCAB * H * 2);
            for a in (0..VOCAB).step_by(8192) {
                let w = exlr8::decode_par(&m, a..(a + 8192).min(VOCAB), 0..H, t)?;
                out.extend(w.iter().flat_map(|&x| bf16_bits(x).to_le_bytes()));
            }
            Ok(vec![out])
        });
        self.vector("G.final_norm".into(), &f, "norm.weight")?;
        let lm = Matrix::open(&f, "lm_head")?;
        self.note(L_MTP + 1, Form { what: "lm_head", src: fmt_name(&lm), form: "MXFP8 decode record", exact: ROUNDED });
        // contiguous: Layout::check holds vocab tiles to the formula
        let tiles = self.layout.held(Group::Vocab, self.rank)?;
        let rows = tiles[0] * 128..(tiles[tiles.len() - 1] + 1) * 128;
        let n = rows.len();
        let f2 = f.clone();
        self.gen("G.lm_head".into(), vec![("G.lm_head.mx.gv".into(), "u8", vec![(n / 16) as i64, (H / 32) as i64, 528])], move || {
            let w = exlr8::decode_par(&Matrix::open(&f2, "lm_head")?, rows.clone(), 0..H, t)?;
            let (c, s) = records::mxfp8_quantize(&w, H);
            Ok(vec![records::mxfp8_record(&c, &s, H)])
        });
        Ok(())
    }

    /// The exactness table: one line per (matrix, release format, gr8 form) with its layers.
    fn table(&self) -> String {
        let mut s = String::from("| matrix | release | gr8 form | weights | layers |\n|---|---|---|---|---|\n");
        for (f, ls) in &self.forms {
            let layers = if ls == &[L_MTP + 1] { "vocab".to_string() } else { ranges(ls) };
            s += &format!("| {} | {} | {} | {} | {} |\n", f.what, f.src, f.form, f.exact, layers);
        }
        s
    }
}

fn ranges(ls: &[usize]) -> String {
    let mut v = ls.to_vec();
    v.sort_unstable();
    v.dedup();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < v.len() {
        let mut j = i;
        while j + 1 < v.len() && v[j + 1] == v[j] + 1 {
            j += 1;
        }
        out.push(if j > i { format!("{}-{}", v[i], v[j]) } else { v[i].to_string() });
        i = j + 1;
    }
    out.join(",")
}

/// What the plan holds besides the defaults.
#[derive(Clone, Copy, Default)]
pub struct Opts {
    /// every layer's full-head q_b / o_proj / UK / UV (GR8_PF_OWN=3)
    pub qo: bool,
    /// trellis matrices as the release's entries for the dense trellis kernels. false rounds the matrices to MXFP8 / int4
    pub trellis: bool,
    /// kv_b (UK / UV) as the release's block FP8 bytes and scales (#50, GR8_KVB=fp8); false: BF16 of them
    pub kvb_fp8: bool,
}

/// Plans rank `rank`'s image at TP `nr` from the release: dense and vocabulary files under `dense_dir` (or its
/// dense/), experts from the width files under `v2_dir` (or its experts/) at the index slices' offsets, widths from
/// the width manifest. The rank holds the pieces `layout` gives it (None: the formulas). It reads the index slices
/// at TP `nr` of every rank whose slice lists one of its halves. Returns the image and the exactness table (markdown).
#[allow(clippy::too_many_arguments)]
pub fn plan(dense_dir: &str, v2_dir: &str, slices_dir: &str, widths_path: &str, nr: usize, rank: usize,
            layout: Option<&Layout>, opts: Opts) -> Result<(V2Image, String), String> {
    let man: Json = v2plan::width_manifest(widths_path)?;
    let pieces = Model::release(&man)?;
    let formula;
    let layout = match layout {
        Some(l) => l,
        None => {
            formula = Layout::formula(&pieces, (0..nr).map(|r| r.to_string()).collect())?;
            &formula
        }
    };
    if layout.nr() != nr || rank >= nr {
        return Err(format!("rank {rank} of TP {nr} with a layout of {} members", layout.nr()));
    }
    layout.check(&pieces)?;
    let mut need = BTreeSet::new();
    for (g, _) in &pieces.groups {
        if let Group::Halves(l, e) = *g {
            need.extend(layout.held(*g, rank)?.iter().map(|h| (16 * e + h + l) % nr));
        }
    }
    let slices = Slices::open(slices_dir, nr, &need.into_iter().collect::<Vec<_>>())?;
    let wfile = |l: usize, k: u8| v2plan::width_file(v2_dir, l, k);
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    let workers = gr8knobs::LOAD_THREADS.get() as usize;
    let mut p = Planner { b: Builder::new(), layout, rank, threads: (cores / workers).max(1), forms: BTreeMap::new(),
                         qo: opts.qo, trellis: opts.trellis, kvb_fp8: opts.kvb_fp8 };
    // the release's files name the model (their metadata); the width manifest has one format for every model
    let model = flash::model_meta(dense_dir)?;
    let is_flash = model.as_ref().and_then(|m| m.get("hidden_size")).and_then(Json::str) == Some("4096");
    if model.is_some() && !is_flash { return Err(format!("{dense_dir}: model metadata of an unsupported model")); }
    let model = model.unwrap_or(Json::Null);
    if is_flash && opts.qo { return Err("Flash full-head QO prefill copies are not implemented".into()); }
    if is_flash && opts.kvb_fp8 { return Err("GR8_KVB=fp8: Flash kv_b loads as BF16 only".into()); }
    let last = if is_flash { 45 } else { L_MTP };
    if is_flash { flash::globals(&mut p, dense_dir, &model)?; } else { p.vocab(dense_dir)?; }
    let mut moe = 0;
    for l in 0..=last {
        if is_flash { flash::layer(&mut p, l, dense_dir, &model)?; } else { p.layer(l, dense_dir)?; }
        if let Some(w) = v2plan::widths(&man, l)? {
            v2plan::v2_layer(&mut p.b, l, &w, &slices, &wfile, &|e| layout.held(Group::Halves(l, e), rank))?;
            moe += 1;
        } else if l >= 3 {
            return Err(format!("{widths_path} has no widths for MoE layer {l}"));
        }
    }
    let listed = man.get("layers").and_then(Json::obj).map(|m| m.len()).unwrap_or(0);
    if moe != listed {
        return Err(format!("{widths_path} lists {listed} layers, the model has {moe} MoE layers"));
    }
    let bytes = p.b.cur.div_ceil(4096) * 4096;
    let table = p.table();
    let b = p.b;
    let plan = Plan::from_segs(b.files, b.segs, b.mems, b.gens, bytes)?;
    Ok((V2Image { rank, tensors: b.tensors, bytes, plan }, table))
}

#[cfg(test)]
mod tests {
    #[test]
    fn qo_cache_roundtrip_and_rejects() {
        let d = std::env::temp_dir().join(format!("qoc-test-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("L003.qo.bin").to_string_lossy().into_owned();
        let parts = vec![vec![1u8, 2, 3], vec![], vec![9u8; 1000], vec![4u8], vec![5u8; 7], vec![6u8; 2]];
        super::qo_cache_write(&p, "src|1|2.0", &parts);
        assert_eq!(super::qo_cache_read(&p, "src|1|2.0", 6), Some(parts.clone()));
        assert_eq!(super::qo_cache_read(&p, "src|1|3.0", 6), None); // other source mtime
        let mut b = std::fs::read(&p).unwrap();
        b.pop();
        std::fs::write(&p, &b).unwrap();
        assert_eq!(super::qo_cache_read(&p, "src|1|2.0", 6), None); // short file
        assert_eq!(super::qo_cache_read(&d.join("none").to_string_lossy(), "src|1|2.0", 6), None);
        std::fs::remove_dir_all(&d).unwrap();
    }

    use super::*;

    #[test]
    fn runs_join_consecutive_pieces() {
        assert_eq!(runs(&[0, 1, 2, 9], 256), vec![0..768, 2304..2560]);
        assert_eq!(runs(&[16, 17], 448), vec![7168..8064]);
    }

    #[test]
    fn blocks_of_runs_are_the_blocks_of_the_whole() {
        // a 4 x 6 matrix, rows 0..1 and 2..4, columns 0..2 and 4..6: the block of rows 0, 2, 3 and columns 0, 1, 4, 5
        let w: Vec<u32> = (0..24).collect();
        let get = |r: &Range<usize>, c: &Range<usize>| -> Vec<u32> {
            r.clone().flat_map(|i| w[i * 6 + c.start..i * 6 + c.end].to_vec()).collect()
        };
        let (rows, cols) = ([0..1, 2..4], [0..2, 4..6]);
        let out = cat_rows(rows.iter().map(|r| cat_cols(cols.iter().map(|c| get(r, c)).collect(), r.len())).collect());
        assert_eq!(out, vec![0, 1, 4, 5, 12, 13, 16, 17, 18, 19, 22, 23]);
    }

    #[test]
    fn layer_ranges_print_compactly() {
        assert_eq!(ranges(&[0, 2, 3, 4, 3, 78]), "0,2-4,78");
    }
}
