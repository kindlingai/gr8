//! Inkling-2-Small: a rank's device image planned straight from the Hugging Face checkpoint directory (config.json,
//! optional hf_quant_config.json, model.safetensors.index.json + shards or a single model.safetensors, optional
//! mtp.safetensors). No offline conversion: tensors are copied from shard byte ranges or computed at load (Op::Gen).
//!
//! Tensor-parallel split over `nr` ranks (rank r):
//!   attention heads   q heads r * H/nr .. (r + 1) * H/nr; the contiguous KV heads those q heads read (head_ranges)
//!   intermediate dim  inter_slice of every expert: row pairs (gate, up) of w13, columns of w2
//!   vocabulary        padded to a multiple of 128 * nr, split in equal row ranges (rows past the vocabulary are zero)
//!   everything else   replicated
//! Routed experts become decode record streams: NVFP4 layers `nv4g` (src/nvfp4.rs), MXFP8 layers `mxfp8`
//! (src/records.rs), in stripes of EXP_GROUP experts. Every dimension comes from config.json and the tensor shapes,
//! so a miniature checkpoint plans through the same code.

use crate::ds41::{inter_slice, Reads};
use crate::exlr8::StFile;
use crate::json::{self, Json};
use crate::nvfp4::{self, Nvfp4OuterScale, Nvfp4RuleConfig, Nvfp4ScaleRule};
use crate::plan::{Gen, Op, Plan, Seg};
use crate::records;
use crate::v2plan::{self, V2Image};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{c_char, c_void};
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

/// Experts computed per `Op::Gen` stripe.
const EXP_GROUP: usize = 4;
/// Column slices up to this many bytes are read at plan time; larger ones are computed at load.
const SMALL: u64 = 64 << 10;
/// Input bytes of one decode-record or column-slice stripe.
const STRIPE: usize = 16 << 20;

type Make = Box<dyn Fn() -> Result<Vec<u8>, String> + Send + Sync>;

/// (q heads, KV heads) rank `rank` of `nr` owns: q heads are an equal contiguous split; the KV heads are the
/// contiguous range those q heads attend through. Local q head lq uses local KV head
/// (q.start + lq) / (n_heads / n_kv_heads) - kv.start.
pub fn head_ranges(n_heads: usize, n_kv_heads: usize, nr: usize, rank: usize) -> Result<(Range<usize>, Range<usize>), String> {
    if nr == 0 || rank >= nr {
        return Err(format!("rank {rank} of {nr} ranks"));
    }
    if n_heads == 0 || n_kv_heads == 0 || !n_heads.is_multiple_of(n_kv_heads) {
        return Err(format!("{n_heads} attention heads over {n_kv_heads} KV heads"));
    }
    if !n_heads.is_multiple_of(nr) {
        return Err(format!("{n_heads} attention heads do not split over {nr} ranks"));
    }
    let (gqa, nh_q) = (n_heads / n_kv_heads, n_heads / nr);
    let q = rank * nh_q..(rank + 1) * nh_q;
    let kv = q.start / gqa..(q.end - 1) / gqa + 1;
    Ok((q, kv))
}

/// (padded vocabulary, rows per rank): the vocabulary rounded up to a multiple of 128 * nr, split equally.
pub fn vocab_split(vocab: usize, nr: usize) -> (usize, usize) {
    let padded = vocab.next_multiple_of(128 * nr);
    (padded, padded / nr)
}

/// The dimensions the planner takes from config.json (`text_config`, or the top level without one).
#[derive(Clone, Debug)]
pub struct Config {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub inter: usize,
    pub n_exp: usize,
    pub top_k: usize,
    pub vocab: usize,
    /// per layer: full (global) attention; false = sliding-window (local) attention
    pub global: Vec<bool>,
}

impl Config {
    pub fn parse(text: &str) -> Result<Config, String> {
        let j = json::parse(text)?;
        let t = j.get("text_config").unwrap_or(&j);
        let int = |k: &str| t.get(k).and_then(Json::int).filter(|&v| v > 0).map(|v| v as usize);
        let need = |k: &str| int(k).ok_or(format!("config.json: no {k}"));
        let layers = need("num_hidden_layers")?;
        let global: Vec<bool> = if let Some(lt) = t.get("layer_types").and_then(Json::arr) {
            lt.iter().map(|x| x.str() == Some("full_attention")).collect()
        } else if let Some(ids) = t.get("local_layer_ids").and_then(Json::arr) {
            let local: BTreeSet<i64> = ids.iter().filter_map(Json::int).collect();
            (0..layers).map(|l| !local.contains(&(l as i64))).collect()
        } else {
            return Err("config.json: neither layer_types nor local_layer_ids".into());
        };
        if global.len() != layers {
            return Err(format!("config.json: {} layer types for {layers} layers", global.len()));
        }
        Ok(Config {
            hidden: need("hidden_size")?,
            layers,
            heads: need("num_attention_heads")?,
            kv_heads: need("num_key_value_heads")?,
            head_dim: need("head_dim")?,
            inter: int("moe_intermediate_size").map_or_else(|| need("intermediate_size"), Ok)?,
            n_exp: need("n_routed_experts")?,
            top_k: need("num_experts_per_tok")?,
            vocab: need("vocab_size")?,
            global,
        })
    }
}

/// How a layer's routed experts are stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertFormat {
    /// E2M1 codes + E4M3 block-16 scales + an fp32 outer scale per expert; the activation rule the checkpoint names
    Nvfp4(Nvfp4ScaleRule),
    /// E4M3 codes + E8M0 block-32 scales
    Mxfp8,
}

/// Options of a plan. The default plans everything the checkpoint holds.
#[derive(Clone, Debug)]
pub struct Opts {
    /// add the decode records `X.gv` of the BF16 matrices
    pub bf16_gv: bool,
    /// plan mtp.safetensors (when the file exists) as `MTP.*`
    pub with_mtp: bool,
    /// plan `model.visual.*` (when present) as `V.*`
    pub with_vision: bool,
}

impl Default for Opts {
    fn default() -> Self {
        Opts { bf16_gv: true, with_mtp: true, with_vision: true }
    }
}

/// A tensor of the checkpoint: its file and absolute byte range.
#[derive(Clone)]
struct Src {
    f: Arc<StFile>,
    start: u64,
    len: u64,
    dtype: String,
    shape: Vec<usize>,
}

impl Src {
    fn read(&self, off: u64, len: u64) -> Result<Vec<u8>, String> {
        if off + len > self.len {
            return Err(format!("{}: read past the end of a tensor", self.f.path));
        }
        self.f.read_at(self.start + off, len)
    }
}

/// The checkpoint's tensors by name. Tensors of mtp.safetensors are keyed `mtp.<name>` whether or not the file
/// prefixes them itself.
struct Ckpt {
    dir: String,
    t: BTreeMap<String, Src>,
    /// tensor name -> (compute_dtype, activation rule) from hf_quant_config.json; empty without the file
    quant: BTreeMap<String, (String, Nvfp4ScaleRule)>,
}

impl Ckpt {
    fn add(&mut self, f: &Arc<StFile>, name: &str, key: String) -> Result<(), String> {
        let i = f.info(name)?;
        let s = Src { f: f.clone(), start: i.start, len: i.end - i.start, dtype: i.dtype.clone(), shape: i.shape.clone() };
        if self.t.insert(key, s).is_some() {
            return Err(format!("{}: tensor {name} is stored twice", f.path));
        }
        Ok(())
    }

    fn open(dir: &str, with_mtp: bool) -> Result<Ckpt, String> {
        let mut ck = Ckpt { dir: dir.trim_end_matches('/').to_string(), t: BTreeMap::new(), quant: BTreeMap::new() };
        let dir = ck.dir.clone();
        let index = format!("{dir}/model.safetensors.index.json");
        if std::path::Path::new(&index).exists() {
            let text = std::fs::read_to_string(&index).map_err(|e| format!("{index}: {e}"))?;
            let j = json::parse(&text).map_err(|e| format!("{index}: {e}"))?;
            let map = j.get("weight_map").and_then(Json::obj).ok_or(format!("{index}: no weight_map"))?;
            let mut files: BTreeMap<&str, Arc<StFile>> = BTreeMap::new();
            for (name, file) in map {
                let file = file.str().ok_or(format!("{index}: {name} has no file"))?;
                if !files.contains_key(file) {
                    files.insert(file, Arc::new(StFile::open(&format!("{dir}/{file}"))?));
                }
                let f = files[file].clone();
                ck.add(&f, name, name.clone())?;
            }
            // a shard tensor the index does not list would be silently dropped: refuse
            for f in files.values() {
                if let Some(n) = f.names().iter().find(|n| !map.contains_key(*n)) {
                    return Err(format!("{}: tensor {n} is not in the index", f.path));
                }
            }
        } else {
            let f = Arc::new(StFile::open(&format!("{dir}/model.safetensors"))?);
            for name in f.names().to_vec() {
                ck.add(&f, &name, name.clone())?;
            }
        }
        let mtp = format!("{dir}/mtp.safetensors");
        if with_mtp && std::path::Path::new(&mtp).exists() {
            let f = Arc::new(StFile::open(&mtp)?);
            for name in f.names().to_vec() {
                ck.add(&f, &name, format!("mtp.{}", name.strip_prefix("mtp.").unwrap_or(&name)))?;
            }
        }
        let qpath = format!("{dir}/hf_quant_config.json");
        if std::path::Path::new(&qpath).exists() {
            let text = std::fs::read_to_string(&qpath).map_err(|e| format!("{qpath}: {e}"))?;
            let j = json::parse(&text).map_err(|e| format!("{qpath}: {e}"))?;
            let groups = j.get("model").and_then(|m| m.get("parameter_groups")).and_then(Json::arr)
                .ok_or(format!("{qpath}: no model.parameter_groups"))?;
            for g in groups {
                let dt = g.get("compute_dtype").and_then(Json::str).ok_or(format!("{qpath}: a group has no compute_dtype"))?;
                let rule = if dt == "nvfp4" { Nvfp4RuleConfig::parse(g).map_err(|e| format!("{qpath}: {e}"))?.rule } else { Nvfp4ScaleRule::Static6 };
                for n in g.get("tensors").and_then(Json::arr).unwrap_or(&[]) {
                    if let Some(n) = n.str() {
                        ck.quant.insert(n.to_string(), (dt.to_string(), rule));
                    }
                }
            }
        }
        Ok(ck)
    }
}

fn label(dtype: &str) -> Result<&'static str, String> {
    match dtype {
        "BF16" => Ok("bf16"),
        "F16" => Ok("f16"),
        "F32" => Ok("f32"),
        "U8" | "I8" | "F8_E4M3" => Ok("u8"),
        d => Err(format!("dtype {d}")),
    }
}

fn nbytes(dtype: &str, shape: &[i64]) -> u64 {
    let e = match dtype { "bf16" | "f16" => 2, "f32" => 4, _ => 1 };
    shape.iter().product::<i64>() as u64 * e
}

fn dims(shape: &[usize]) -> Vec<i64> {
    shape.iter().map(|&d| d as i64).collect()
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

/// Rows r0 .. r0 + m of a BF16 [.][k] matrix at `s`, columns `cols`, as raw bytes [m][cols.len()].
fn rows_cols(s: &Src, k: usize, r0: usize, m: usize, cols: &Range<usize>) -> Result<Vec<u8>, String> {
    let raw = s.read((r0 * k * 2) as u64, (m * k * 2) as u64)?;
    if cols.len() == k {
        return Ok(raw);
    }
    let mut out = Vec::with_capacity(m * cols.len() * 2);
    for row in raw.chunks_exact(k * 2) {
        out.extend_from_slice(&row[cols.start * 2..cols.end * 2]);
    }
    Ok(out)
}

struct Planner<'a> {
    ck: &'a Ckpt,
    cfg: &'a Config,
    b: v2plan::Builder,
    used: BTreeSet<String>,
    reads: Reads,
    nr: usize,
    rank: usize,
    opts: Opts,
    q: Range<usize>,
    kv: Range<usize>,
    /// rows of the padded vocabulary this rank owns
    v: Range<usize>,
    experts: Vec<ExpertFormat>,
}

impl Planner<'_> {
    fn has(&self, native: &str) -> bool {
        self.ck.t.contains_key(native)
    }

    /// The checkpoint tensor `native`, marked consumed.
    fn take(&mut self, native: &str) -> Result<Src, String> {
        let s = self.ck.t.get(native).ok_or_else(|| format!("{}: no tensor {native}", self.ck.dir))?.clone();
        self.used.insert(native.to_string());
        Ok(s)
    }

    /// Notes ranges of the file of `s` this rank reads (the fetch list). Offsets are relative to the tensor.
    fn note(&mut self, s: &Src, runs: impl IntoIterator<Item = (u64, u64)>) {
        let rel = s.f.path.strip_prefix(&format!("{}/", self.ck.dir)).unwrap_or(&s.f.path).to_string();
        let start = s.start;
        self.reads.entry(rel).or_default().extend(runs.into_iter().map(|(a, n)| (start + a, n)));
    }

    fn copy_seg(&mut self, pack_off: u64, s: &Src, off: u64, n: u64) {
        let fi = self.b.file(s.f.path.clone());
        self.b.seg(pack_off, n, Op::Copy, fi, s.start + off);
        self.note(s, [(off, n)]);
    }

    fn mem_seg(&mut self, pack_off: u64, bytes: Vec<u8>) {
        self.b.segs.push(Seg { pack_off, nbytes: bytes.len() as u64, op: Op::Mem, file_off: 0, file: self.b.mems.len() });
        self.b.mems.push(bytes);
    }

    fn gen_segs(&mut self, what: &str, base: u64, total: u64, stripes: Vec<(u64, Make)>) {
        let mut off = 0u64;
        for (span, make) in stripes {
            let gi = self.b.gens.len();
            self.b.gens.push(Gen::new(what.to_string(), span, make));
            self.b.segs.push(Seg { pack_off: base + off, nbytes: span, op: Op::Gen, file_off: 0, file: gi });
            off += span;
        }
        assert_eq!(off, total, "{what}: stripes do not cover the tensor");
    }

    /// A replicated tensor copied whole as `name` with `shape` (the stored element count), checked against `dtype`
    /// when one is expected.
    fn whole(&mut self, name: String, native: &str, dtype: Option<&'static str>, shape: Option<&[usize]>) -> Result<(), String> {
        let s = self.take(native)?;
        let dt = label(&s.dtype).map_err(|e| format!("{native}: {e}"))?;
        if dtype.is_some_and(|d| d != dt) {
            return Err(format!("{native}: stored as {}, expected {}", s.dtype, dtype.unwrap()));
        }
        let shape = dims(shape.unwrap_or(&s.shape));
        let n = nbytes(dt, &shape);
        if n != s.len {
            return Err(format!("{native}: shape {:?} does not hold {shape:?}", s.shape));
        }
        let o = self.b.begin(name, dt, shape, n, None);
        self.copy_seg(o, &s, 0, n);
        Ok(())
    }

    /// A BF16 matrix `native` [N][K] (K = the product of the trailing dimensions): rows `rows` and columns `cols` of it
    /// as `name` [out_rows][cols], zero rows after the stored ones when `out_rows` is larger, plus the decode record
    /// `name.gv` when `gv` and both output dimensions are multiples of 16. `n_full` is the row count the config implies.
    #[allow(clippy::too_many_arguments)]
    fn mat(&mut self, name: &str, native: &str, n_full: usize, rows: Option<Range<usize>>, cols: Option<Range<usize>>,
           out_rows: Option<usize>, gv: bool) -> Result<(), String> {
        let s = self.take(native)?;
        if s.dtype != "BF16" || s.shape.is_empty() {
            return Err(format!("{native}: stored as {} {:?}, expected a BF16 matrix", s.dtype, s.shape));
        }
        let (nn, k) = (s.shape[0], s.shape[1..].iter().product::<usize>());
        if nn != n_full {
            return Err(format!("{native}: {nn} rows, the config implies {n_full}"));
        }
        let (rows, cols) = (rows.unwrap_or(0..nn), cols.unwrap_or(0..k));
        if rows.end > nn || cols.end > k || cols.is_empty() || rows.start > rows.end {
            return Err(format!("{native}: rows {rows:?} columns {cols:?} of [{nn}, {k}]"));
        }
        let (n, kw) = (rows.len(), cols.len());
        let out_rows = out_rows.unwrap_or(n);
        if out_rows < n || out_rows == 0 {
            return Err(format!("{native}: {n} rows into {out_rows}"));
        }
        let total = (out_rows * kw * 2) as u64;
        let stored = (n * kw * 2) as u64;
        let base = self.b.begin(name.to_string(), "bf16", vec![out_rows as i64, kw as i64], total, None);
        if n > 0 && kw == k {
            self.copy_seg(base, &s, (rows.start * k * 2) as u64, stored);
        } else if n > 0 && stored <= SMALL {
            self.note(&s, rows.clone().map(|r| ((r * k + cols.start) as u64 * 2, kw as u64 * 2)));
            self.mem_seg(base, rows_cols(&s, k, rows.start, n, &cols)?);
        } else if n > 0 {
            // whole rows are read and the columns cut at load
            self.note(&s, [((rows.start * k * 2) as u64, (n * k * 2) as u64)]);
            let per = (STRIPE / (k * 2)).max(1);
            let mut stripes: Vec<(u64, Make)> = Vec::new();
            let mut a = 0;
            while a < n {
                let m = per.min(n - a);
                let (s2, r0, c) = (s.clone(), rows.start + a, cols.clone());
                stripes.push(((m * kw * 2) as u64, Box::new(move || rows_cols(&s2, k, r0, m, &c))));
                a += m;
            }
            self.gen_segs(name, base, stored, stripes);
        }
        if out_rows > n {
            self.mem_seg(base + stored, vec![0u8; (total - stored) as usize]);
        }
        if gv && out_rows.is_multiple_of(16) && kw.is_multiple_of(16) {
            self.note(&s, [((rows.start * k * 2) as u64, (n * k * 2) as u64)]);
            let per = (STRIPE / (k * 2) / 16).max(1) * 16;
            let mut stripes: Vec<(u64, Make)> = Vec::new();
            let mut a = 0;
            while a < out_rows {
                let m = per.min(out_rows - a);
                let have = m.min(n.saturating_sub(a));
                let (s2, r0, c) = (s.clone(), rows.start + a, cols.clone());
                stripes.push(((m / 16 * (kw / 16) * 512) as u64, Box::new(move || {
                    let mut raw = if have > 0 { rows_cols(&s2, k, r0, have, &c)? } else { Vec::new() };
                    raw.resize(m * kw * 2, 0);
                    let w: Vec<u16> = raw.as_chunks::<2>().0.iter().map(|c| u16::from_le_bytes(*c)).collect();
                    Ok(records::bf16_record(&w, kw))
                })));
                a += m;
            }
            let shape = vec![(out_rows / 16) as i64, (kw / 16) as i64, 512];
            let gtotal = nbytes("u8", &shape);
            let gbase = self.b.begin(format!("{name}.gv"), "u8", shape, gtotal, None);
            self.gen_segs(&format!("{name}.gv"), gbase, gtotal, stripes);
        }
        Ok(())
    }

    fn globals(&mut self) -> Result<(), String> {
        let (c, gv) = (self.cfg, self.opts.bf16_gv);
        let h = [c.hidden];
        self.whole("G.embed".into(), "model.llm.embed.weight", Some("bf16"), Some(&[c.vocab, c.hidden]))?;
        self.whole("G.embed_norm".into(), "model.llm.embed_norm.weight", Some("bf16"), Some(&h))?;
        self.whole("G.final_norm".into(), "model.llm.norm.weight", Some("bf16"), Some(&h))?;
        self.whole("G.attn_res.output_query".into(), "model.llm.attn_res.output_query", Some("f32"), Some(&h))?;
        let valid = self.v.start.min(c.vocab)..self.v.end.min(c.vocab);
        self.mat("G.lm_head", "model.llm.unembed.weight", c.vocab, Some(valid), None, Some(self.v.len()), gv)
    }

    /// The routed experts of a layer as record streams of this rank's intermediate slice, in stripes of EXP_GROUP
    /// experts: `P.x.fc1` (gate | up records per 16-row chunk and k group) and `P.x.fc2`, and for NVFP4 the outer
    /// scales `P.x.s1` / `P.x.s2`.
    fn routed(&mut self, p: &str, pre: &str) -> Result<(), String> {
        let (e_n, i_full, h) = (self.cfg.n_exp, self.cfg.inter, self.cfg.hidden);
        let names = [format!("{pre}w13_weight"), format!("{pre}w13_weight_scale"), format!("{pre}w2_weight"), format!("{pre}w2_weight_scale")];
        let [w13, w13s, w2, w2s] = [self.take(&names[0])?, self.take(&names[1])?, self.take(&names[2])?, self.take(&names[3])?];
        // (code bytes, scale bytes) per `k` elements along K
        type Per = fn(usize) -> usize;
        let (nv, cb, sb): (bool, Per, Per) = match (w13.dtype.as_str(), w13s.dtype.as_str()) {
            ("U8", "F8_E4M3") => (true, |k| k / 2, |k| k / 16),
            ("F8_E4M3", "U8") => (false, |k| k, |k| k / 32),
            (a, b) => return Err(format!("{}: weight {a} with scale {b} is neither NVFP4 nor MXFP8", names[0])),
        };
        if (w2.dtype.as_str(), w2s.dtype.as_str()) != (w13.dtype.as_str(), w13s.dtype.as_str()) {
            return Err(format!("{}: stored as {} / {}, unlike w13", names[2], w2.dtype, w2s.dtype));
        }
        let kg = if nv { nvfp4::NV4_KG } else { 32 };
        if !h.is_multiple_of(kg) || !h.is_multiple_of(16) {
            return Err(format!("{pre}: hidden size {h} is not a multiple of {kg}"));
        }
        for (s, n, want) in [(&w13, &names[0], vec![e_n, 2 * i_full, cb(h)]), (&w13s, &names[1], vec![e_n, 2 * i_full, sb(h)]),
                             (&w2, &names[2], vec![e_n, h, cb(i_full)]), (&w2s, &names[3], vec![e_n, h, sb(i_full)])] {
            if s.shape != want {
                return Err(format!("{n}: shape {:?}, expected {want:?}", s.shape));
            }
        }
        let rule = match self.ck.quant.get(&names[0]) {
            _ if self.ck.quant.is_empty() => Nvfp4ScaleRule::Static6,
            Some((dt, rule)) => {
                let (d2, stored) = (self.ck.quant.get(&names[2]).map(|q| q.0.as_str()), if nv { "nvfp4" } else { "mxfp8" });
                if dt != stored || d2 != Some(stored) {
                    return Err(format!("{}: stored as {stored}, hf_quant_config.json says {dt} / {d2:?}", names[0]));
                }
                *rule
            }
            None => return Err(format!("{}: not listed in hf_quant_config.json", names[0])),
        };
        self.experts.push(if nv { ExpertFormat::Nvfp4(rule) } else { ExpertFormat::Mxfp8 });

        let isl = inter_slice(i_full, self.nr, self.rank)?;
        let iw = isl.len();
        let (rec, xf) = if nv { (nvfp4::NV4_GR, "nv4g") } else { (528, "mxfp8") };
        let (per1, per2) = (iw / 16 * (h / kg) * 2 * rec, h / 16 * (iw / kg) * rec);
        // fc1: interleaved row pairs (gate, up) for rows isl of each expert's [2 inter][hidden]
        let (crow, srow) = (cb(h), sb(h));
        let i0 = isl.start;
        let at = move |e: usize, row_bytes: usize| ((e * 2 * i_full + 2 * i0) * row_bytes) as u64;
        for e in 0..e_n {
            self.note(&w13, [(at(e, crow), (2 * iw * crow) as u64)]);
            self.note(&w13s, [(at(e, srow), (2 * iw * srow) as u64)]);
        }
        let deinterleave = move |pair_rows: &[u8], row_bytes: usize| -> (Vec<u8>, Vec<u8>) {
            let (mut g, mut u) = (Vec::with_capacity(iw * row_bytes), Vec::with_capacity(iw * row_bytes));
            for (k, row) in pair_rows.chunks_exact(row_bytes).enumerate() {
                if k % 2 == 0 {
                    g.extend_from_slice(row);
                } else {
                    u.extend_from_slice(row);
                }
            }
            (g, u)
        };
        let mut stripes: Vec<(u64, Make)> = Vec::new();
        for e0 in (0..e_n).step_by(EXP_GROUP) {
            let ne = EXP_GROUP.min(e_n - e0);
            let (c, s) = (w13.clone(), w13s.clone());
            stripes.push(((ne * per1) as u64, Box::new(move || {
                let mut out = Vec::with_capacity(ne * per1);
                for e in e0..e0 + ne {
                    let (gc, uc) = deinterleave(&c.read(at(e, crow), (2 * iw * crow) as u64)?, crow);
                    let (gs, us) = deinterleave(&s.read(at(e, srow), (2 * iw * srow) as u64)?, srow);
                    out.extend_from_slice(&if nv { nvfp4::nvfp4_fc1(&gc, &gs, &uc, &us, iw, h) } else { records::mxfp8_fc1(&gc, &gs, &uc, &us, iw, h) });
                }
                Ok(out)
            })));
        }
        let shape = vec![e_n as i64, (iw / 16) as i64, (h / kg) as i64, (2 * rec) as i64];
        let total = nbytes("u8", &shape);
        let base = self.b.begin(format!("{p}.x.fc1"), "u8", shape, total, Some(xf));
        self.gen_segs(&format!("{p}.x.fc1"), base, total, stripes);

        // fc2: columns isl of each expert's [hidden][inter]; whole experts are read and the columns cut at load
        let (crow, srow) = (cb(i_full), sb(i_full));
        self.note(&w2, [(0, w2.len)]);
        self.note(&w2s, [(0, w2s.len)]);
        let mut stripes: Vec<(u64, Make)> = Vec::new();
        for e0 in (0..e_n).step_by(EXP_GROUP) {
            let ne = EXP_GROUP.min(e_n - e0);
            let (c, s, c0) = (w2.clone(), w2s.clone(), isl.start);
            stripes.push(((ne * per2) as u64, Box::new(move || {
                let mut out = Vec::with_capacity(ne * per2);
                for e in e0..e0 + ne {
                    let (pc, ps) = (c.read((e * h * crow) as u64, (h * crow) as u64)?, s.read((e * h * srow) as u64, (h * srow) as u64)?);
                    out.extend_from_slice(&if nv { nvfp4::nvfp4_fc2(&pc, &ps, h, i_full, c0, iw) } else { records::mxfp8_fc2(&pc, &ps, h, i_full, c0, iw) });
                }
                Ok(out)
            })));
        }
        let shape = vec![e_n as i64, (h / 16) as i64, (iw / kg) as i64, rec as i64];
        let total = nbytes("u8", &shape);
        let base = self.b.begin(format!("{p}.x.fc2"), "u8", shape, total, Some(xf));
        self.gen_segs(&format!("{p}.x.fc2"), base, total, stripes);

        if nv {
            for (w, out) in [("w13", "s1"), ("w2", "s2")] {
                let scale = self.outer_scale(&format!("{pre}{w}_weight"), e_n)?;
                let bytes: Vec<u8> = (0..e_n).flat_map(|e| scale.at(e).to_le_bytes()).collect();
                let o = self.b.begin(format!("{p}.x.{out}"), "f32", vec![e_n as i64], bytes.len() as u64, None);
                self.mem_seg(o, bytes);
            }
        }
        Ok(())
    }

    /// The outer scales of NVFP4 tensor `w`, read at plan time: `w_amax` over `w_outer_scale_denominator`, or `w_scale_2`.
    fn outer_scale(&mut self, w: &str, e_n: usize) -> Result<Nvfp4OuterScale, String> {
        let get = |me: &mut Self, name: String| -> Result<Vec<f32>, String> {
            let s = me.take(&name)?;
            if s.dtype != "F32" {
                return Err(format!("{name}: stored as {}, expected F32", s.dtype));
            }
            me.note(&s, [(0, s.len)]);
            Ok(f32s(&s.read(0, s.len)?))
        };
        let scale = if self.has(&format!("{w}_amax")) {
            let amax = get(self, format!("{w}_amax"))?;
            let den = get(self, format!("{w}_outer_scale_denominator"))?;
            if den.len() != 1 || den[0] == 0.0 {
                return Err(format!("{w}_outer_scale_denominator: expected one nonzero value"));
            }
            Nvfp4OuterScale::AmaxDenominator { amax, denominator: den[0] }
        } else if self.has(&format!("{w}_scale_2")) {
            Nvfp4OuterScale::Direct(get(self, format!("{w}_scale_2"))?)
        } else {
            return Err(format!("{w}: NVFP4 without {w}_amax or {w}_scale_2"));
        };
        if scale.len() != 1 && scale.len() != e_n {
            return Err(format!("{w}: {} outer scales for {e_n} experts", scale.len()));
        }
        Ok(scale)
    }

    fn layer(&mut self, l: usize) -> Result<(), String> {
        let c = self.cfg;
        let (h, hd, gv, global) = (c.hidden, c.head_dim, self.opts.bf16_gv, c.global[l]);
        let p = format!("L{l:03}");
        let t = |s: &str| format!("model.llm.layers.{l}.{s}");
        let o = |s: &str| format!("{p}.{s}");
        let (q, kv) = (self.q.clone(), self.kv.clone());
        let scaled = |r: &Range<usize>, m: usize| Some(r.start * m..r.end * m);

        if l >= 1 {
            let shape: &[usize] = if global { &[3, h] } else { &[h] };
            self.whole(o("attn_res.attn_query"), &format!("model.llm.attn_res.attn_queries.{}", l - 1), Some("f32"), Some(shape))?;
        }
        self.whole(o("attn_res.mlp_query"), &format!("model.llm.attn_res.mlp_queries.{l}"), Some("f32"), Some(&[h]))?;
        if global {
            for n in ["q_input_norm", "k_input_norm", "v_input_norm"] {
                self.whole(o(n), &t(&format!("{n}.weight")), Some("bf16"), Some(&[h]))?;
            }
        } else {
            self.whole(o("attn_norm"), &t("attn_norm.weight"), Some("bf16"), Some(&[h]))?;
        }
        let wr = self.ck.t.get(&t("attn.wr_du.weight")).ok_or_else(|| format!("no tensor {}", t("attn.wr_du.weight")))?;
        if wr.shape.is_empty() || !wr.shape[0].is_multiple_of(c.heads) {
            return Err(format!("{}: shape {:?} over {} heads", t("attn.wr_du.weight"), wr.shape, c.heads));
        }
        let rel_dim = wr.shape[0] / c.heads;
        self.mat(&o("wq"), &t("attn.wq_du.weight"), c.heads * hd, scaled(&q, hd), None, None, gv)?;
        self.mat(&o("wr"), &t("attn.wr_du.weight"), c.heads * rel_dim, scaled(&q, rel_dim), None, None, gv)?;
        self.mat(&o("gate_attn"), &t("attn.gate_attn.weight"), c.heads * hd, scaled(&q, hd), None, None, gv)?;
        self.mat(&o("attn_sink_proj"), &t("attn.attn_sink_proj.weight"), c.heads, Some(q.clone()), None, None, false)?;
        self.mat(&o("wk"), &t("attn.wk_dv.weight"), c.kv_heads * hd, scaled(&kv, hd), None, None, gv)?;
        self.mat(&o("wv"), &t("attn.wv_dv.weight"), c.kv_heads * hd, scaled(&kv, hd), None, None, gv)?;
        self.mat(&o("wo"), &t("attn.wo_ud.weight"), h, None, scaled(&q, hd), None, gv)?;
        self.mat(&o("q_l2_clip"), &t("attn.q_l2_clip.scale"), c.heads, Some(q.clone()), None, None, false)?;
        self.mat(&o("rel_l2_clip"), &t("attn.rel_l2_clip.scale"), c.heads, Some(q.clone()), None, None, false)?;
        self.mat(&o("rel_logits_proj"), &t("attn.rel_logits_proj.proj"), rel_dim, None, None, None, false)?;
        for w in ["k_sconv", "v_sconv"] {
            if global {
                self.mat(&o(&format!("{w}.static_w")), &t(&format!("attn.{w}.static_w")), 4, None, scaled(&kv, hd), None, false)?;
                self.mat(&o(&format!("{w}.weight_proj")), &t(&format!("attn.{w}.weight_proj")), c.kv_heads * 4, scaled(&kv, 4), None, None, false)?;
            } else {
                self.mat(&o(w), &t(&format!("attn.{w}.weight")), c.kv_heads * hd, scaled(&kv, hd), None, None, false)?;
            }
        }
        self.mat(&o("attn_sconv"), &t("attn_sconv.weight"), h, None, None, None, false)?;
        self.mat(&o("mlp_sconv"), &t("mlp_sconv.weight"), h, None, None, None, false)?;
        self.whole(o("mlp_norm"), &t("mlp_norm.weight"), Some("bf16"), Some(&[h]))?;
        self.mat(&o("router.w"), &t("mlp.gate.weight"), c.n_exp, None, None, None, gv)?;
        self.whole(o("router.bias"), &t("mlp.gate.bias"), Some("f32"), Some(&[c.n_exp]))?;
        self.routed(&p, &t("mlp.experts."))
    }

    /// Every `model.visual.*` tensor, replicated as stored under `V.`; left out (and accounted for) without vision.
    fn vision(&mut self) -> Result<(), String> {
        let names: Vec<String> = self.ck.t.range("model.visual.".to_string()..).map(|(k, _)| k.clone())
            .take_while(|k| k.starts_with("model.visual.")).collect();
        for n in names {
            if self.opts.with_vision {
                self.whole(format!("V.{}", &n["model.visual.".len()..]), &n, None, None)?;
            } else {
                self.used.insert(n);
            }
        }
        Ok(())
    }

    /// The tensors of mtp.safetensors under `MTP.`: head and intermediate slices like the trunk's, the rest (the
    /// Markov token embedding too) replicated.
    fn mtp(&mut self) -> Result<(), String> {
        if !self.ck.t.keys().any(|k| k.starts_with("mtp.")) {
            return Ok(());
        }
        let c = self.cfg;
        for n in ["fuse_weight", "mask_embed", "norm.weight", "markov_head.dflare_norm.weight", "markov_head.rnn_proj.weight",
                  "markov_head.rnn_proj.bias", "markov_head.hidden_out.weight", "markov_head.hidden_out.bias"] {
            self.whole(format!("MTP.{n}"), &format!("mtp.{n}"), None, None)?;
        }
        // an input embedding: every rank looks up the row of any previous token, so it is replicated like G.embed
        self.whole("MTP.markov_head.markov_w1".into(), "mtp.markov_head.markov_w1.weight", Some("bf16"), None)?;
        let (q, kv) = (self.q.clone(), self.kv.clone());
        let scaled = |r: &Range<usize>, m: usize| Some(r.start * m..r.end * m);
        let rows_of = |me: &Self, n: &str| me.ck.t.get(n).and_then(|s| s.shape.first().copied()).ok_or(format!("no tensor {n}"));
        let mut m = 0;
        while self.has(&format!("mtp.layers.{m}.input_layernorm.weight")) {
            let t = |s: &str| format!("mtp.layers.{m}.{s}");
            let o = |s: &str| format!("MTP.L{m:02}.{s}");
            for n in ["dflare_norm.weight", "input_layernorm.weight", "post_attention_layernorm.weight",
                      "self_attn.q_norm.weight", "self_attn.k_norm.weight", "self_attn.k_norm_ctx.weight"] {
                self.whole(o(n), &t(n), None, None)?;
            }
            let (nq, nk) = (rows_of(self, &t("self_attn.q_proj.weight"))?, rows_of(self, &t("self_attn.k_proj.weight"))?);
            if !nq.is_multiple_of(c.heads) || !nk.is_multiple_of(c.kv_heads) {
                return Err(format!("mtp layer {m}: {nq} q rows / {nk} k rows over {} / {} heads", c.heads, c.kv_heads));
            }
            let (hq, hk) = (nq / c.heads, nk / c.kv_heads);
            self.mat(&o("self_attn.q_proj.weight"), &t("self_attn.q_proj.weight"), nq, scaled(&q, hq), None, None, false)?;
            for n in ["k_proj", "k_proj_ctx", "v_proj", "v_proj_ctx"] {
                let n = format!("self_attn.{n}.weight");
                self.mat(&o(&n), &t(&n), nk, scaled(&kv, hk), None, None, false)?;
            }
            let no = rows_of(self, &t("self_attn.o_proj.weight"))?;
            self.mat(&o("self_attn.o_proj.weight"), &t("self_attn.o_proj.weight"), no, None, scaled(&q, hq), None, false)?;
            let inter = rows_of(self, &t("mlp.gate_proj.weight"))?;
            let isl = inter_slice(inter, self.nr, self.rank)?;
            self.mat(&o("mlp.gate_proj.weight"), &t("mlp.gate_proj.weight"), inter, Some(isl.clone()), None, None, false)?;
            self.mat(&o("mlp.up_proj.weight"), &t("mlp.up_proj.weight"), inter, Some(isl.clone()), None, None, false)?;
            let nd = rows_of(self, &t("mlp.down_proj.weight"))?;
            self.mat(&o("mlp.down_proj.weight"), &t("mlp.down_proj.weight"), nd, None, Some(isl), None, false)?;
            m += 1;
        }
        Ok(())
    }
}

/// A planned rank image and what the plan knows about it.
pub struct Ink2Plan {
    pub img: V2Image,
    /// checkpoint file (relative to the directory) -> the byte ranges this rank reads, coalesced
    pub reads: Reads,
    pub cfg: Config,
    pub q_range: Range<usize>,
    pub kv_range: Range<usize>,
    pub vocab_padded: usize,
    /// padded vocabulary rows this rank owns
    pub v_range: Range<usize>,
    /// per layer, how its routed experts are stored
    pub experts: Vec<ExpertFormat>,
}

impl Ink2Plan {
    pub fn reads_json(&self) -> String {
        crate::ds41::reads_json(&self.reads)
    }

    /// Bytes this rank reads from the checkpoint.
    pub fn read_bytes(&self) -> u64 {
        self.reads.values().flatten().map(|r| r.1).sum()
    }

    /// The image manifest (g6pack-v1) with the given per-tensor sha256 strings.
    pub fn manifest_json(&self, shas: &[String], nr: usize) -> String {
        manifest_json(&self.img, shas, nr)
    }
}

fn manifest_json(img: &V2Image, shas: &[String], nr: usize) -> String {
    let q = |s: &str| format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""));
    let mut out = format!("{{\n \"align\": 4096,\n \"format\": \"g6pack-v1\",\n \"model\": \"inkling-2-small\",\n \"pack_bytes\": {},\n \"rank\": {},\n \"tp\": {},\n \"tensors\": [", img.bytes, img.rank, nr);
    for (i, t) in img.tensors.iter().enumerate() {
        let shape: Vec<String> = t.shape.iter().map(i64::to_string).collect();
        let xf = t.expert_format.as_deref().map_or(String::new(), |x| format!(", \"expert_format\": {}", q(x)));
        out += &format!("{}\n  {{\"dtype\": {}, \"name\": {}, \"nbytes\": {}, \"offset\": {}, \"sha256\": \"{}\", \"shape\": [{}]{xf}}}",
                        if i > 0 { "," } else { "" }, q(&t.dtype), q(&t.name), t.nbytes, t.offset, shas[i], shape.join(", "));
    }
    out + "\n ]\n}\n"
}

/// Plans rank `rank`'s image at TP `nr` from the checkpoint directory `dir`.
pub fn plan(dir: &str, nr: usize, rank: usize, opts: &Opts) -> Result<Ink2Plan, String> {
    let dir = dir.trim_end_matches('/');
    let cpath = format!("{dir}/config.json");
    let cfg = Config::parse(&std::fs::read_to_string(&cpath).map_err(|e| format!("{cpath}: {e}"))?)?;
    let (q, kv) = head_ranges(cfg.heads, cfg.kv_heads, nr, rank)?;
    inter_slice(cfg.inter, nr, rank)?;
    let (vocab_padded, per) = vocab_split(cfg.vocab, nr);
    let v = rank * per..(rank + 1) * per;
    let ck = Ckpt::open(dir, opts.with_mtp)?;
    let mut p = Planner { ck: &ck, cfg: &cfg, b: v2plan::Builder::new(), used: BTreeSet::new(), reads: Reads::new(), nr, rank,
                          opts: opts.clone(), q: q.clone(), kv: kv.clone(), v: v.clone(), experts: Vec::new() };
    p.globals()?;
    for l in 0..cfg.layers {
        p.layer(l)?;
    }
    p.vision()?;
    p.mtp()?;
    let missing: Vec<&String> = ck.t.keys().filter(|n| !p.used.contains(*n)).collect();
    if !missing.is_empty() {
        return Err(format!("{} checkpoint tensors are not planned, e.g. {:?}", missing.len(), &missing[..missing.len().min(8)]));
    }
    // every file the plan opens: its header too (the reads above are tensor data)
    let headers: Vec<(String, u64)> = p.reads.keys().map(|k| {
        let n = std::fs::File::open(format!("{dir}/{k}")).ok().and_then(|f| {
            let mut b = [0u8; 8];
            std::os::unix::fs::FileExt::read_exact_at(&f, &mut b, 0).ok().map(|_| 8 + u64::from_le_bytes(b))
        });
        (k.clone(), n.unwrap_or(0))
    }).collect();
    for (k, n) in headers {
        p.reads.get_mut(&k).unwrap().push((0, n));
    }
    for v in p.reads.values_mut() {
        v.sort();
        let mut m: Vec<(u64, u64)> = Vec::with_capacity(v.len());
        for &(a, n) in v.iter() {
            match m.last_mut() {
                Some(l) if a <= l.0 + l.1 => l.1 = l.1.max(a + n - l.0),
                _ => m.push((a, n)),
            }
        }
        *v = m;
    }
    let (b, reads, experts) = (p.b, p.reads, p.experts);
    let bytes = b.cur.div_ceil(4096) * 4096;
    let plan = Plan::from_segs(b.files, b.segs, b.mems, b.gens, bytes)?;
    Ok(Ink2Plan { img: V2Image { rank, tensors: b.tensors, bytes, plan, cache: None }, reads, cfg, q_range: q, kv_range: kv,
                  vocab_padded, v_range: v, experts })
}

// ---------------------------------------------------------------------------------------------------- C interface

/// Plans rank `rank`'s image at TP `nr` from the checkpoint directory `model_dir` and writes the image's manifest
/// (sha256 fields zero) to `out_manifest`. Returns a handle for g6l_plan_load, g6l_plan_read and g6l_plan_free, or NULL
/// with a message in `err`.
///
/// # Safety
/// The strings must be NUL-terminated, `err` must hold `errlen` bytes.
#[no_mangle]
pub unsafe extern "C" fn g6l_ink2_plan(model_dir: *const c_char, nr: u32, rank: u32, out_manifest: *const c_char,
                                       err: *mut c_char, errlen: usize) -> *mut c_void {
    let t0 = Instant::now();
    let arg = |p, what| crate::cstr(p).ok_or(format!("g6l_ink2_plan: no {what}"));
    let r = (|| {
        let dir = arg(model_dir, "model dir")?;
        let d = dir.trim_end_matches('/');
        let (mpath, ppath) = (format!("{d}/tp{nr}.rank{rank}.manifest.json"), format!("{d}/tp{nr}.rank{rank}.pack"));
        if std::path::Path::new(&mpath).is_file() && std::path::Path::new(&ppath).is_file() {
            let manifest = std::fs::read_to_string(&mpath).map_err(|e| format!("{mpath}: {e}"))?;
            let j = json::parse(&manifest).map_err(|e| format!("{mpath}: {e}"))?;
            let bytes = j.get("pack_bytes").and_then(Json::int).filter(|&v| v > 0)
                .ok_or_else(|| format!("{mpath}: no pack_bytes"))? as u64;
            let flen = std::fs::metadata(&ppath).map_err(|e| format!("{ppath}: {e}"))?.len();
            if flen != bytes {
                return Err(format!("{ppath}: file is {flen} bytes, manifest says {bytes}"));
            }
            let out = arg(out_manifest, "output manifest")?;
            std::fs::write(&out, &manifest).map_err(|e| format!("{out}: {e}"))?;
            let segs = vec![Seg { pack_off: 0, nbytes: bytes, op: Op::Copy, file: 0, file_off: 0 }];
            let plan = Plan::from_segs(vec![ppath], segs, Vec::new(), Vec::new(), bytes)?;
            return Ok::<_, String>(V2Image { rank: rank as usize, tensors: Vec::new(), bytes, plan, cache: None });
        }
        let p = plan(&dir, nr as usize, rank as usize, &Opts::default())?;
        let out = arg(out_manifest, "output manifest")?;
        let zeros = vec!["0".repeat(64); p.img.tensors.len()];
        let manifest = manifest_json(&p.img, &zeros, nr as usize);
        std::fs::write(&out, &manifest).map_err(|e| format!("{out}: {e}"))?;
        // GR8_GEN_CACHE (gencache.rs): the key binds the engine binary, the planned manifest and the checkpoint directory.
        let mut img = p.img;
        let key_text = format!("ink2 dir={dir} nr={nr} rank={rank}\n{manifest}\n{}", crate::gencache::source_identity(&[&dir]));
        img.cache = crate::gencache::GenCache::from_env(&key_text);
        Ok::<_, String>(img)
    })();
    match r {
        Ok(img) => Box::into_raw(Box::new((img, t0))) as *mut c_void,
        Err(e) => {
            crate::write_err(err, errlen, &e);
            std::ptr::null_mut()
        }
    }
}
