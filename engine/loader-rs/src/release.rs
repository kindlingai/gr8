//! A rank's whole image from the exlr8 v2 release alone, planned at boot: no g6 rank pack. Every non-expert tensor
//! comes from the release's dense and vocabulary files (src/exlr8.rs), the routed and shared experts from the width
//! files at the index slices' offsets (src/v2plan.rs, as `--v2` plans them). The tensors and their names are the ones
//! the engine binds from a converted pack (tools/convert/convert.py), in the forms it reads today:
//!
//!   exact (the release's bytes, or a re-layout of them):
//!     norms, router weight and bias, indexer weights_proj and k_norm: as stored (+ BF16 decode records)
//!     q_a, kv_a, indexer wq_b, MTP eh_proj: MXFP8 codes and scales as stored + MXFP8 decode records
//!     indexer wk: MXFP8 -> BF16 (exact: E4M3 times a power of two) + BF16 decode record
//!     q_b and o_proj stored as int4 g32 (layers 0, 2, 3): int4 decode records from the release's codes and scales
//!   BF16 of the release's value (what loader/exlr8.py --dtype bf16 gives):
//!     kv_b (FP8 128x128 blocks) -> UK / UV; the embedding (FP8 per row)
//!   the trellis matrices (q_b, o_proj, the dense MLP of layers 0-2), by Opts::trellis:
//!     trellis: the rank's entries as stored + suh / svh, for the dense trellis kernels (engine/kernels/dtr.cuh) - exact
//!     rounded: MXFP8 (q_b, and every prefill copy) and int4 g32 decode records (o_proj, dense MLP) of the decoded weight
//!   rounded, either way: lm_head (MXFP8 decode record); the prefill copies of int4 q_b / o_proj (MXFP8)
//! The rounding is round to nearest (src/records.rs). The planner returns a table of which matrix took which form.
//! Sharding follows tools/convert/plan.py: heads and dense channels rotate by layer, vocab rows do not.

use crate::exlr8::{self, bf16_bits, Matrix, StFile, FP8_BLOCK, INT4, MXFP8, TRELLIS};
use crate::json::Json;
use crate::plan::{Gen, Op, Plan};
use crate::records;
use crate::v2plan::{self, Builder, V2Image};
use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;

const H: usize = 6144;
const L_MTP: usize = 78;
const VOCAB: usize = 154880;

/// Units over nr ranks, the larger chunks first (engine/g6nr.h, tools/convert/plan.py widths).
fn widths(n: usize, nr: usize) -> Vec<usize> {
    (0..nr).map(|j| n / nr + usize::from(j < n % nr)).collect()
}

/// (start, count) of rank r's chunk when rank order starts at `off` (plan.py _chunks).
fn chunk(w: &[usize], off: usize, r: usize) -> (usize, usize) {
    let nr = w.len();
    let j = (r + nr - off % nr) % nr;
    (w[..j].iter().sum(), w[j])
}

/// Rank r's ownership at TP nr (plan.py heads, dense_groups, vocab_rows).
pub struct Shard {
    pub nr: usize,
    pub rank: usize,
}

impl Shard {
    pub fn new(nr: usize, rank: usize) -> Result<Shard, String> {
        if !(nr == 3 || nr == 6 || (nr > 0 && 64 % nr == 0)) || rank >= nr {
            return Err(format!("TP {nr} rank {rank}: g6 builds TP 3, 6 or a divisor of 64"));
        }
        if !(384 / nr).is_multiple_of(4) {
            return Err(format!("TP {nr}: the dense MLP's rank slices do not fall on the release's 128-channel blocks"));
        }
        Ok(Shard { nr, rank })
    }
    /// (first head, heads) of layer l
    pub fn heads(&self, l: usize) -> (usize, usize) {
        chunk(&widths(64, self.nr), l, self.rank)
    }
    /// dense MLP channels of layer l (g32 groups x 32)
    pub fn dense(&self, l: usize) -> Range<usize> {
        let (a, n) = chunk(&widths(384, self.nr), l, self.rank);
        a * 32..(a + n) * 32
    }
    pub fn vocab(&self) -> Range<usize> {
        let (a, n) = chunk(&widths(1210, self.nr), 0, self.rank);
        a * 128..(a + n) * 128
    }
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

struct Planner {
    b: Builder,
    shard: Shard,
    threads: usize,
    forms: BTreeMap<Form, Vec<usize>>,
    /// add every layer's full-head q_b / o_proj / UK / UV (G6_PF_OWN=3)
    qo: bool,
    /// trellis matrices as the release's entries (the dense trellis kernels); false: rounded to MXFP8 / int4
    trellis: bool,
}

/// A tensor of a planned group: name, manifest dtype, shape, bytes.
type Spec = (String, &'static str, Vec<i64>);

fn nbytes(dtype: &str, shape: &[i64]) -> u64 {
    let e = match dtype { "bf16" | "f16" => 2, "f32" => 4, "i64" => 8, _ => 1 };
    shape.iter().product::<i64>() as u64 * e
}

impl Planner {
    fn note(&mut self, l: usize, f: Form) {
        self.forms.entry(f).or_default().push(l);
    }

    /// A tensor copied from a file range.
    fn copy(&mut self, name: String, dtype: &'static str, shape: Vec<i64>, f: &StFile, t: &str, off: u64) -> Result<(), String> {
        let i = f.info(t)?;
        let n = nbytes(dtype, &shape);
        if i.start + off + n > i.end {
            return Err(format!("{}: {t} is smaller than {name}", f.path));
        }
        let o = self.b.begin(name, dtype, shape, n, None);
        let fi = self.b.file(f.path.clone());
        self.b.seg(o, n, Op::Copy, fi, i.start + off);
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

    /// A vector (norm, bias) as stored, under its g6 name.
    fn vector(&mut self, name: String, f: &StFile, t: &str) -> Result<(), String> {
        let i = f.info(t)?.clone();
        let dtype = match i.dtype.as_str() { "BF16" => "bf16", "F32" => "f32", "F16" => "f16", d => return Err(format!("{t}: dtype {d}")) };
        self.copy(name, dtype, i.shape.iter().map(|&d| d as i64).collect(), f, t, 0)
    }

    /// An MXFP8 matrix as stored (all rows): codes + scales copied (unless `decode_only`), and the decode record.
    fn mxfp8(&mut self, l: usize, what: &'static str, p: &str, f: &Arc<StFile>, m: &str, decode_only: bool) -> Result<(), String> {
        let mx = Matrix::open(f, m)?;
        if mx.rec.fmt != MXFP8 {
            return Err(format!("{}: {m} is {}, g6 reads it as MXFP8", f.path, fmt_name(&mx)));
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
            _ => return Err(format!("{}: {m} is {}, g6 reads it as BF16", f.path, fmt_name(&mx))),
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
        let (h0, nh) = self.shard.heads(l);
        let t = self.threads;
        self.vector(format!("{p}.attn_norm"), &f, "input_layernorm.weight")?;
        // q_a has only its MXFP8 record. Decode reads it as the MTP layer always did
        self.mxfp8(l, "q_a", &format!("{p}.q_a"), &f, "self_attn.q_a_proj", false)?;
        self.vector(format!("{p}.q_a_norm"), &f, "self_attn.q_a_layernorm.weight")?;
        self.mxfp8(l, "kv_a", &format!("{p}.kv_a"), &f, "self_attn.kv_a_proj_with_mqa", false)?;
        self.vector(format!("{p}.kv_a_norm"), &f, "self_attn.kv_a_layernorm.weight")?;

        // q_b: this rank's heads, output rows h0*256..
        let qb = Matrix::open(&f, "self_attn.q_b_proj")?;
        let rows = h0 * 256..(h0 + nh) * 256;
        if qb.rec.fmt == TRELLIS && self.trellis {
            self.trellis(l, "q_b", &p, &qb, rows.clone())?;
        } else {
            self.q_b_rounded(l, &p, &f, &qb, rows.clone());
        }

        // kv_b: heads' rows of [64 x 448][512]; per head the first 192 rows are UK, the other 256 UV
        let kvb = Matrix::open(&f, "self_attn.kv_b_proj")?;
        self.note(l, Form { what: "kv_b", src: fmt_name(&kvb), form: "BF16 (UK, UV)",
                            exact: if kvb.rec.fmt == FP8_BLOCK { BF16_OF } else { ROUNDED } });
        let f2 = f.clone();
        self.gen(format!("{p}.kv_b"), vec![
            (format!("{p}.kv_b.uk"), "bf16", vec![nh as i64, 192, 512]),
            (format!("{p}.kv_b.uv"), "bf16", vec![nh as i64, 256, 512])], move || {
            let w = bf16v(&exlr8::decode_par(&Matrix::open(&f2, "self_attn.kv_b_proj")?, h0 * 448..(h0 + nh) * 448, 0..512, t)?);
            let (mut uk, mut uv) = (Vec::new(), Vec::new());
            for h in 0..nh {
                uk.extend_from_slice(&w[h * 448 * 512..][..192 * 512]);
                uv.extend_from_slice(&w[(h * 448 + 192) * 512..][..256 * 512]);
            }
            Ok(vec![le16(&uk), le16(&uv)])
        });

        // o_proj: this rank's heads' input columns
        let op = Matrix::open(&f, "self_attn.o_proj")?;
        let cols = h0 * 256..(h0 + nh) * 256;
        if op.rec.fmt == TRELLIS && self.trellis {
            self.trellis(l, "o_proj", &p, &op, cols)?;
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
            let ch = self.shard.dense(l);
            for (n, m) in [("dense.gate", "mlp.gate_proj"), ("dense.up", "mlp.up_proj"), ("dense.down", "mlp.down_proj")] {
                self.trellis(l, n, &p, &Matrix::open(&f, m)?, ch.clone())?;
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
    fn q_b_rounded(&mut self, l: usize, p: &str, f: &Arc<StFile>, qb: &Matrix, rows: Range<usize>) {
        let int4 = qb.rec.fmt == INT4;
        self.note(l, Form { what: "q_b", src: fmt_name(qb), form: "MXFP8 + decode record", exact: ROUNDED });
        if int4 {
            self.note(l, Form { what: "q_b", src: fmt_name(qb), form: "int4 g32 decode record", exact: EXACT });
        }
        let (n, t, f2) = (rows.len(), self.threads, f.clone());
        let mut specs = vec![
            (format!("{p}.q_b.mx.codes"), "u8", vec![n as i64, 2048]),
            (format!("{p}.q_b.mx.scales"), "u8", vec![n as i64, 64]),
            (format!("{p}.q_b.mx.gv"), "u8", vec![(n / 16) as i64, 64, 528])];
        if int4 {
            specs.push((format!("{p}.q_b.gptq"), "u8", vec![(n / 16) as i64, 64, 288]));
        }
        self.gen(format!("{p}.q_b"), specs, move || {
            let m = Matrix::open(&f2, "self_attn.q_b_proj")?;
            let w = exlr8::decode_par(&m, rows.clone(), 0..2048, t)?;
            let (c, s) = records::mxfp8_quantize(&w, 2048);
            let gv = records::mxfp8_record(&c, &s, 2048);
            let mut out = vec![c, s, gv];
            if int4 {
                let (q, qs) = m.int4(&rows, &(0..2048))?;
                out.push(records::rt_record(&q, &qs, 2048));
            }
            Ok(out)
        });
    }

    /// o_proj for today's kernels: decode reads an int4 record (MXFP8 in the MTP layer), prefill an MXFP8 copy. The
    /// int4 record is the release's own where o_proj is stored as int4, else rounded like the MXFP8 copy.
    fn o_proj_rounded(&mut self, l: usize, p: &str, f: &Arc<StFile>, op: &Matrix, cols: Range<usize>) {
        let (mtp, int4, k, t) = (l == L_MTP, op.rec.fmt == INT4, cols.len(), self.threads);
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
                let w = exlr8::decode_par(&m, 0..H, cols.clone(), t)?;
                let (c, s) = records::mxfp8_quantize(&w, k);
                let mut out = Vec::new();
                if !mtp {
                    let (q, qs) = if int4 { m.int4(&(0..H), &cols)? } else { records::int4_quantize(&w, k) };
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
    /// M.trellis as stored, `ch` on the sharded side (128-channel entries), and suh / svh sliced on that side.
    fn trellis(&mut self, l: usize, what: &'static str, p: &str, m: &Matrix, ch: Range<usize>) -> Result<(), String> {
        let r = m.rec;
        self.note(l, Form { what: if what.starts_with("dense.") { "dense MLP" } else { what }, src: fmt_name(m),
                            form: "trellis entries (dense trellis kernels)", exact: EXACT });
        let ents = ch.start / 128..ch.end / 128;
        let (off, eb) = m.trellis_entries(&ents)?;
        let t = format!("{}.trellis", m.name);
        let rel = off - m.f.info(&t)?.start;
        self.copy(format!("{p}.{what}.trellis"), "u8", vec![ents.len() as i64, eb as i64], m.f, &t, rel)?;
        let (suh, svh) = if r.side == exlr8::SIDE_IN { (ch.clone(), 0..r.out) } else { (0..r.inp, ch.clone()) };
        self.copy(format!("{p}.{what}.suh"), "f16", vec![suh.len() as i64], m.f, &format!("{}.suh", m.name), 2 * suh.start as u64)?;
        self.copy(format!("{p}.{what}.svh"), "f16", vec![svh.len() as i64], m.f, &format!("{}.svh", m.name), 2 * svh.start as u64)
    }

    /// G6_PF_OWN=3's full-head weights of layer l, the tensors tools/convert/qopack.py assembles from the rank packs:
    /// q_b and o_proj as MXFP8 for all 64 heads, UK and UV in BF16. MXFP8 scales cover 32 inputs of one row, so the
    /// whole matrix rounds to the concatenation of the rank slices.
    fn full_heads(&mut self, l: usize, p: &str, f: &Arc<StFile>) {
        self.note(l, Form { what: "q_b, o_proj all heads (G6_PF_OWN=3)", src: "as above".into(), form: "MXFP8", exact: ROUNDED });
        let (f2, t) = (f.clone(), self.threads);
        self.gen(format!("{p}.qo"), vec![
            (format!("{p}.qo.q_b.mx.codes"), "u8", vec![16384, 2048]),
            (format!("{p}.qo.q_b.mx.scales"), "u8", vec![16384, 64]),
            (format!("{p}.qo.o_proj.mx.codes"), "u8", vec![H as i64, 16384]),
            (format!("{p}.qo.o_proj.mx.scales"), "u8", vec![H as i64, 512]),
            (format!("{p}.qo.kv_b.uk"), "bf16", vec![64, 192, 512]),
            (format!("{p}.qo.kv_b.uv"), "bf16", vec![64, 256, 512])], move || {
            let q = exlr8::decode_par(&Matrix::open(&f2, "self_attn.q_b_proj")?, 0..16384, 0..2048, t)?;
            let (qc, qs) = records::mxfp8_quantize(&q, 2048);
            drop(q);
            let o = exlr8::decode_par(&Matrix::open(&f2, "self_attn.o_proj")?, 0..H, 0..16384, t)?;
            let (oc, os) = records::mxfp8_quantize(&o, 16384);
            drop(o);
            let w = bf16v(&exlr8::decode_par(&Matrix::open(&f2, "self_attn.kv_b_proj")?, 0..64 * 448, 0..512, t)?);
            let (mut uk, mut uv) = (Vec::new(), Vec::new());
            for h in 0..64 {
                uk.extend_from_slice(&w[h * 448 * 512..][..192 * 512]);
                uv.extend_from_slice(&w[(h * 448 + 192) * 512..][..256 * 512]);
            }
            Ok(vec![qc, qs, oc, os, le16(&uk), le16(&uv)])
        });
    }

    /// Layers 0-2: this rank's channels of gate/up (rows) and down (columns), int4 records for decode and MXFP8
    /// copies for prefill, both rounded from the decoded weights.
    fn dense_mlp(&mut self, l: usize, p: &str, f: &Arc<StFile>) -> Result<(), String> {
        let ch = self.shard.dense(l);
        let i = ch.len();
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
            let g = exlr8::decode_par(&Matrix::open(&f2, "mlp.gate_proj")?, ch.clone(), 0..H, t)?;
            let u = exlr8::decode_par(&Matrix::open(&f2, "mlp.up_proj")?, ch.clone(), 0..H, t)?;
            let d = exlr8::decode_par(&Matrix::open(&f2, "mlp.down_proj")?, 0..H, ch.clone(), t)?;
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
        let rows = self.shard.vocab();
        let n = rows.len();
        let f2 = f.clone();
        self.gen("G.lm_head".into(), vec![("G.lm_head.mx.gv".into(), "u8", vec![(n / 16) as i64, (H / 32) as i64, 528])], move || {
            let w = exlr8::decode_par(&Matrix::open(&f2, "lm_head")?, rows.clone(), 0..H, t)?;
            let (c, s) = records::mxfp8_quantize(&w, H);
            Ok(vec![records::mxfp8_record(&c, &s, H)])
        });
        Ok(())
    }

    /// The exactness table: one line per (matrix, release format, g6 form) with its layers.
    fn table(&self) -> String {
        let mut s = String::from("| matrix | release | g6 form | weights | layers |\n|---|---|---|---|---|\n");
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
    /// every layer's full-head q_b / o_proj / UK / UV (G6_PF_OWN=3)
    pub qo: bool,
    /// trellis matrices as the release's entries for the dense trellis kernels. false rounds the matrices to MXFP8 / int4
    pub trellis: bool,
}

/// Plans rank `rank`'s image at TP `nr` from the release: dense and vocabulary files under `dense_dir` (or its
/// dense/), experts from the width files under `v2_dir` (or its experts/) at the index slices' offsets, widths from
/// the width manifest. Returns the image and the exactness table (markdown).
pub fn plan(dense_dir: &str, v2_dir: &str, slices_dir: &str, widths_path: &str, nr: usize, rank: usize, opts: Opts)
            -> Result<(V2Image, String), String> {
    let shard = Shard::new(nr, rank)?;
    let man: Json = v2plan::width_manifest(widths_path)?;
    let slices = v2plan::open_slices(slices_dir, nr, rank)?;
    let wfile = |l: usize, k: u8| v2plan::width_file(v2_dir, l, k);
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    let workers = std::env::var("G6_LOAD_THREADS").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&v| v > 0).unwrap_or(8);
    let mut p = Planner { b: Builder::new(), shard, threads: (cores / workers).max(1), forms: BTreeMap::new(),
                         qo: opts.qo, trellis: opts.trellis };
    p.vocab(dense_dir)?;
    let mut moe = 0;
    for l in 0..=L_MTP {
        p.layer(l, dense_dir)?;
        if let Some(w) = v2plan::widths(&man, l)? {
            v2plan::v2_layer(&mut p.b, l, &w, &slices, &wfile, nr, rank)?;
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
    use super::*;

    #[test]
    fn ownership_matches_plan_py() {
        // tools/convert/plan.py at TP4: heads 16 per rank rotating with the layer, dense 3072 channels, vocab tiles
        // 303, 303, 302, 302
        let s = Shard::new(4, 1).unwrap();
        assert_eq!(s.heads(0), (16, 16));
        assert_eq!(s.heads(1), (0, 16));
        assert_eq!(s.heads(2), (48, 16));
        assert_eq!(s.dense(0), 3072..6144);
        assert_eq!(s.vocab(), 303 * 128..606 * 128);
        // TP3: 22, 21, 21 heads. Layer 4's rank 0 holds the last group (the TP3 pack's L004 heads [43, 64])
        let s = Shard::new(3, 0).unwrap();
        assert_eq!(s.heads(0), (0, 22));
        assert_eq!(s.heads(4), (43, 21));
        assert_eq!(s.dense(0), 0..4096);
        assert_eq!(s.vocab(), 0..51712);
        // TP6: 11, 11, 11, 11, 10, 10
        assert_eq!(Shard::new(6, 5).unwrap().heads(0), (54, 10));
        assert!(Shard::new(5, 0).is_err());
    }

    #[test]
    fn layer_ranges_print_compactly() {
        assert_eq!(ranges(&[0, 2, 3, 4, 3, 78]), "0,2-4,78");
    }
}
