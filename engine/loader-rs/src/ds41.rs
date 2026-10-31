//! DeepSeek-V4.1-Flash-1017's tensor map onto the shared loader: a rank's whole image planned at boot from the
//! release kindlingai/deepseek-v4.1-flash-exlr8-native (the official weights byte for byte, in the exlr8 v2 file
//! layout: dense records per layer, expert stores of 128-channel units, Engram tables in split files). Its README and
//! exlr8 FORMAT.md describe the files; docs/DEEPSEEK-WEIGHTS.md lists every tensor name, form and ownership rule.
//!
//!   FP8 matrices (32 x 32 block scales) -> MXFP8: codes as stored, the block scale repeated over its 32 rows, and the
//!                                          decode record (records::mxfp8_record), `P.mx.{codes,scales,gv}`
//!   BF16 matrices                       -> as stored + the BF16 decode record `P.gv` (rows % 16 == 0)
//!   routed experts (MXFP4, EP = 1)      -> the "mx4g" records of engine/kernels/mxfp4.cuh: `x.fc1`, `x.fc2`. Every
//!                                          rank holds the intermediate slice [r Iw, (r + 1) Iw) of every expert,
//!                                          read from the units that cover it (not the release's unit owners)
//!   the n-gram tables (Engram)          -> not in the image: `EngramSlice` says which rows of the split files a rank
//!                                          serves
//!
//! The engine loads the image through g6l_ds41_plan and the usual g6l_plan_load / g6l_plan_read / g6l_plan_free.
//! The planner reads headers and index files only; the bytes are read by the load. `Ds41Plan::reads` lists the file
//! ranges the load reads, which is what a box fetches.
use crate::exlr8::{self, StFile};
use crate::json::Json;
use crate::plan::{Gen, Op, Plan};
use crate::records;
use crate::sha256::Sha256;
use crate::v2plan::{self, Builder, Tensor, V2Image};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{c_char, c_void};
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::time::Instant;

/// The fp8 and fp4 scale group: one scale byte per 32 values along K (fp8: also 32 rows).
const BLK: usize = 32;
/// lm_head rows come in tiles of 128 rows.
const VTILE: usize = 128;
/// Attention is split by output group; the checkpoint has 8 (config o_groups).
const O_GROUPS: usize = 8;
/// mx4g record: 16 rows x 64 k = 544 bytes (engine/kernels/mxfp4.cuh).
const KG: usize = 64;
const GR: usize = 544;
const SOFF: usize = 512;
/// Experts per generated stripe: a stripe is one load-time work item.
const EXP_GROUP: usize = 4;
/// E8M0 byte for 1.0, the pad scale of the Engram table's tail rows.
pub const PAD_SCALE: u8 = 127;

// ---------------------------------------------------------------------------------------------------- ownership

/// `n` units over `nr` ranks, the larger chunks first (tools/convert/plan.py widths, engine/g6nr.h).
pub fn widths(n: usize, nr: usize) -> Vec<usize> {
    (0..nr).map(|j| n / nr + usize::from(j < n % nr)).collect()
}

fn chunk(w: &[usize], j: usize) -> Range<usize> {
    let s: usize = w[..j].iter().sum();
    s..s + w[j]
}

/// The O-groups (of 8; 8 heads each) layer `l` gives rank `rank` at `nr` ranks: chunk j of widths(8, nr) goes to rank
/// (j + l) % nr (tools/convert/plan.py `_chunks`). TP4: 2 groups = 16 heads a rank.
pub fn o_groups(l: usize, nr: usize, rank: usize) -> Range<usize> {
    chunk(&widths(O_GROUPS, nr), (rank + nr - l % nr) % nr)
}

/// The 128-row lm_head tiles rank `rank` holds, not rotated: widths(n_tiles, nr) chunk `rank`.
pub fn vocab_tiles(n_tiles: usize, nr: usize, rank: usize) -> Range<usize> {
    chunk(&widths(n_tiles, nr), rank)
}

/// The slice of an intermediate dimension of `i` rank `rank` holds in every expert (EP=1). The slice must be a
/// multiple of 64 (the mx4g record's k group, which is also two 32-value scale groups): TP 1, 2, 3, 4, 6, 9, 12 for 2304.
pub fn inter_slice(i: usize, nr: usize, rank: usize) -> Result<Range<usize>, String> {
    if nr == 0 || i % nr != 0 || (i / nr) % KG != 0 {
        return Err(format!("intermediate size {i} over {nr} ranks: each slice must be a multiple of {KG}"));
    }
    Ok(rank * (i / nr)..(rank + 1) * (i / nr))
}

// ---------------------------------------------------------------------------------------------------- release

/// Layer of the first DSpark (MTP) layer in the release: native `mtp.{m}.` is release layer MTP0 + m.
const MTP0: usize = 40;

/// A tensor's byte range in one release file.
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

/// Where the codes and scales of a block-scaled matrix [out][in] (HF orientation) sit. `sr` rows share a scale row.
#[derive(Clone)]
enum Place {
    /// codes [out][in] and scales [out / sr][in / 32], both row-major
    Rows { codes: u64, scales: u64 },
    /// unit u holds rows [u unit, (u + 1) unit): codes [unit][in] at .0, scales [unit / sr][in / 32] at .1
    OutUnits { unit: usize, offs: Vec<(u64, u64)> },
    /// unit u holds columns [u unit, (u + 1) unit): codes [out][unit] at .0, scales [out / sr][unit / 32] at .1
    InUnits { unit: usize, offs: Vec<(u64, u64)> },
}

/// A block-scaled matrix of the release: FP8 E4M3 with E8M0 per 32 x 32 (sr = 32), or MXFP4 with E8M0 per 32 inputs
/// of each row (nib, sr = 1).
#[derive(Clone)]
struct QMat {
    f: Arc<StFile>,
    out: usize,
    inp: usize,
    nib: bool,
    sr: usize,
    place: Place,
    /// bytes of one whole unit (codes and scales) of an expert matrix; 0 for a dense record
    ubytes: u64,
}

impl QMat {
    /// code bytes of k columns
    fn cb(&self, k: usize) -> usize {
        if self.nib { k / 2 } else { k }
    }

    /// The file ranges that hold `rows` x `cols` of the codes, and of the scales of those rows, as (offset, bytes).
    /// Each range is one contiguous run; `codes` and `scales` read them in this order.
    fn ranges(&self, rows: &Range<usize>, cols: &Range<usize>) -> (Vec<(u64, u64)>, Vec<(u64, u64)>) {
        let (cb, sc) = (self.cb(self.inp), self.inp / BLK);
        let (br, bc) = (rows.start / self.sr..rows.end.div_ceil(self.sr), cols.start / BLK..cols.end / BLK);
        let (mut c, mut s) = (Vec::new(), Vec::new());
        match &self.place {
            Place::Rows { codes, scales } => {
                for r in rows.clone() {
                    c.push((codes + (r * cb + self.cb(cols.start)) as u64, self.cb(cols.len()) as u64));
                }
                for b in br {
                    s.push((scales + (b * sc + bc.start) as u64, bc.len() as u64));
                }
            }
            Place::OutUnits { unit, offs } => {
                for r in rows.clone() {
                    let (u, ru) = (r / unit, r % unit);
                    c.push((offs[u].0 + (ru * cb + self.cb(cols.start)) as u64, self.cb(cols.len()) as u64));
                }
                for b in br {
                    let (u, bu) = (b * self.sr / unit, b % (unit / self.sr));
                    s.push((offs[u].1 + (bu * sc + bc.start) as u64, bc.len() as u64));
                }
            }
            Place::InUnits { unit, offs } => {
                let (ucb, usc) = (self.cb(*unit), unit / BLK);
                for r in rows.clone() {
                    for u in cols.start / unit..cols.end.div_ceil(*unit) {
                        let (a, e) = (cols.start.max(u * unit) - u * unit, cols.end.min((u + 1) * unit) - u * unit);
                        c.push((offs[u].0 + (r * ucb + self.cb(a)) as u64, self.cb(e - a) as u64));
                    }
                }
                for b in br {
                    for u in cols.start / unit..cols.end.div_ceil(*unit) {
                        let (a, e) = (cols.start.max(u * unit) - u * unit, cols.end.min((u + 1) * unit) - u * unit);
                        s.push((offs[u].1 + (b * usc + a / BLK) as u64, ((e - a) / BLK) as u64));
                    }
                }
            }
        }
        (c, s)
    }

    fn check(&self, rows: &Range<usize>, cols: &Range<usize>) -> Result<(), String> {
        if rows.is_empty() || cols.is_empty() || rows.end > self.out || cols.end > self.inp || cols.start % BLK != 0 || cols.end % BLK != 0 {
            return Err(format!("{}: rows {rows:?} cols {cols:?} of [{},{}] are not 32-column aligned", self.f.path, self.out, self.inp));
        }
        Ok(())
    }

    fn gather(&self, runs: &[(u64, u64)]) -> Result<Vec<u8>, String> {
        let mut v = Vec::with_capacity(runs.iter().map(|r| r.1 as usize).sum());
        let mut i = 0;
        while i < runs.len() {
            // adjacent runs read as one
            let (a, mut n) = runs[i];
            let mut j = i + 1;
            while j < runs.len() && runs[j].0 == a + n {
                n += runs[j].1;
                j += 1;
            }
            v.extend_from_slice(&self.f.read_at(a, n)?);
            i = j;
        }
        Ok(v)
    }

    /// codes of `rows` x `cols`, row-major [rows][cb(cols)]
    fn codes(&self, rows: &Range<usize>, cols: &Range<usize>) -> Result<Vec<u8>, String> {
        self.check(rows, cols)?;
        self.gather(&self.ranges(rows, cols).0)
    }

    /// scales of the scale rows `rows` covers, row-major [scale rows][cols / 32]
    fn scales(&self, rows: &Range<usize>, cols: &Range<usize>) -> Result<Vec<u8>, String> {
        self.check(rows, cols)?;
        self.gather(&self.ranges(rows, cols).1)
    }
}

/// One layer's expert stores: the routed experts (MXFP4) and the shared expert (FP8 32 x 32), 128-channel units at
/// the offsets the layer's index file lists ([expert][proj][unit] = (offset, bytes)).
struct Experts {
    mx4: Arc<StFile>,
    fp8: Arc<StFile>,
    mx4_index: Vec<i64>,
    fp8_index: Vec<i64>,
    n_exp: usize,
    units: usize,
}

/// The release under `dir` (kindlingai/deepseek-v4.1-flash-exlr8-native, exlr8 v2 layout, FORMAT.md): native tensor
/// names resolve to its dense records, plain tensors and expert units. Headers and index files only.
pub struct Release {
    dir: String,
    dense: BTreeMap<usize, Arc<StFile>>,
    vocab: Arc<StFile>,
    experts: BTreeMap<usize, Experts>,
    engram: BTreeMap<usize, Vec<Arc<StFile>>>,
    hid: usize,
    inter: usize,
}

const UNIT: usize = 128;

fn meta_usize(f: &StFile, k: &str) -> Result<usize, String> {
    f.meta.get(k).and_then(Json::str).and_then(|v| v.parse().ok()).ok_or_else(|| format!("{}: no metadata {k}", f.path))
}

fn i64s(f: &StFile, name: &str) -> Result<Vec<i64>, String> {
    let i = f.info(name)?;
    if i.dtype != "I64" {
        return Err(format!("{}: {name} is {}, not I64", f.path, i.dtype));
    }
    Ok(f.read_all(name)?.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap())).collect())
}

impl Release {
    pub fn open(dir: &str) -> Result<Release, String> {
        let vocab = Arc::new(StFile::open(&exlr8::dense_path(dir, None))?);
        let (hid, inter) = (meta_usize(&vocab, "hidden_size")?, meta_usize(&vocab, "moe_intermediate_size")?);
        let (mut dense, mut experts, mut engram) = (BTreeMap::new(), BTreeMap::new(), BTreeMap::new());
        // main layers from 0, the DSpark layers from MTP0
        for l in (0..MTP0).chain(MTP0..2 * MTP0) {
            let p = exlr8::dense_path(dir, Some(l));
            if !std::path::Path::new(&p).exists() {
                continue;
            }
            let f = Arc::new(StFile::open(&p)?);
            if meta_usize(&f, "layer")? != l {
                return Err(format!("{p}: metadata layer is not {l}"));
            }
            dense.insert(l, f);
            let e = |s: &str| format!("{dir}/experts/exlr8-layer-{l:03}.{s}.safetensors");
            let index = StFile::open(&e("index"))?;
            let mx4_index = i64s(&index, "mxfp4.index")?;
            let fp8_index = i64s(&index, "shared.fp8b32.index")?;
            let shape = index.info("mxfp4.index")?.shape.clone();
            let (n_exp, units) = match shape[..] { [n, 3, u, 2] => (n, u), _ => return Err(format!("{}: mxfp4.index {shape:?}", index.path)) };
            if units * UNIT != inter || fp8_index.len() != 3 * units * 2 {
                return Err(format!("{}: {units} units of {UNIT}, intermediate {inter}", index.path));
            }
            experts.insert(l, Experts { mx4: Arc::new(StFile::open(&e("mxfp4"))?), fp8: Arc::new(StFile::open(&e("fp8b32"))?),
                                        mx4_index, fp8_index, n_exp, units });
            let mut parts = Vec::new();
            for pp in 0.. {
                let s = format!("{dir}/dense/exlr8-split-{l:03}.engram.embed.p{pp:03}.safetensors");
                if !std::path::Path::new(&s).exists() {
                    break;
                }
                parts.push(Arc::new(StFile::open(&s)?));
            }
            if !parts.is_empty() {
                engram.insert(l, parts);
            }
        }
        if dense.is_empty() {
            return Err(format!("{dir}: no dense/exlr8-dense-000.safetensors"));
        }
        Ok(Release { dir: dir.to_string(), dense, vocab, experts, engram, hid, inter })
    }

    /// (file, local name) of a native name: `layers.{l}.X` and `mtp.{m}.X` are layer files, the rest the vocabulary.
    fn locate(&self, native: &str) -> Result<(&Arc<StFile>, String), String> {
        let split = |s: &str| s.split_once('.').and_then(|(n, x)| n.parse::<usize>().ok().map(|n| (n, x.to_string())));
        let (l, x) = if let Some(r) = native.strip_prefix("layers.").and_then(split) {
            r
        } else if let Some((m, x)) = native.strip_prefix("mtp.").and_then(split) {
            (MTP0 + m, x)
        } else {
            let x = match native {
                "embed" => "embed_tokens",
                "head" => "lm_head",
                n => n,
            };
            return Ok((&self.vocab, x.to_string()));
        };
        Ok((self.dense.get(&l).ok_or_else(|| format!("the release has no layer {l} for {native}"))?, x))
    }

    fn record(&self, native: &str) -> Result<Option<(&Arc<StFile>, String, Vec<i64>)>, String> {
        let (f, x) = self.locate(native)?;
        if f.info(&format!("{x}.record")).is_err() {
            return Ok(None);
        }
        Ok(Some((f, x.clone(), i64s(f, &format!("{x}.record"))?)))
    }

    /// Is `native` a matrix record (`X` for `X.weight`) or a plain tensor of the release?
    fn has(&self, native: &str) -> bool {
        let m = native.strip_suffix(".weight").unwrap_or(native);
        if self.expert(native).is_some() {
            return true;
        }
        matches!(self.record(m), Ok(Some(_))) || self.locate(native).is_ok_and(|(f, x)| f.info(&x).is_ok())
    }

    /// A plain tensor (vectors, norms, mHC parameters, sinks).
    fn plain(&self, native: &str) -> Result<Src, String> {
        let (f, x) = self.locate(native)?;
        let i = f.info(&x)?;
        Ok(Src { f: f.clone(), start: i.start, len: i.end - i.start, dtype: i.dtype.clone(), shape: i.shape.clone() })
    }

    /// A BF16 matrix (a record, or a plain 2-D tensor): its data and (out, in).
    fn bf16(&self, native: &str) -> Result<(Src, usize, usize), String> {
        let Some((f, x, r)) = self.record(native)? else {
            let s = self.plain(native).map_err(|_| format!("the release has no matrix {native}"))?;
            return match s.shape[..] {
                [out, inp] if s.dtype == "BF16" => Ok((s, out, inp)),
                _ => Err(format!("{native}: {} {:?}, gr8 reads a BF16 matrix", s.dtype, s.shape)),
            };
        };
        if r[0] != exlr8::BF16 || r[4] == exlr8::SIDE_IN {
            return Err(format!("{native}: format {} side {}, gr8 reads BF16 stored by rows", r[0], r[4]));
        }
        let i = f.info(&format!("{x}.bf16"))?;
        let (out, inp) = (r[1] as usize, r[2] as usize);
        if i.end - i.start != (out * inp * 2) as u64 {
            return Err(format!("{native}: {} bytes for [{out},{inp}]", i.end - i.start));
        }
        Ok((Src { f: f.clone(), start: i.start, len: i.end - i.start, dtype: "BF16".into(), shape: vec![out, inp] }, out, inp))
    }

    /// (layer, expert or None for the shared one, projection 0 gate / 1 up / 2 down) of an expert matrix name.
    fn expert(&self, native: &str) -> Option<(usize, Option<usize>, usize)> {
        let (f, x) = self.locate(native).ok()?;
        let l = meta_usize(f, "layer").ok()?;
        let x = x.strip_prefix("ffn.")?;
        let x = x.strip_suffix(".weight").unwrap_or(x);
        let (e, w) = if let Some(w) = x.strip_prefix("shared_experts.") {
            (None, w)
        } else {
            let (e, w) = x.strip_prefix("experts.")?.split_once('.')?;
            (Some(e.parse().ok()?), w)
        };
        let proj = match w { "w1" => 0, "w3" => 1, "w2" => 2, _ => return None };
        Some((l, e, proj))
    }

    /// A block-scaled matrix: a dense FP8 32 x 32 record, or an expert projection from its units.
    fn qmat(&self, native: &str) -> Result<QMat, String> {
        if let Some((l, e, proj)) = self.expert(native) {
            let x = self.experts.get(&l).ok_or_else(|| format!("the release has no experts for layer {l}"))?;
            // routed: MXFP4, a scale per 32 inputs of each row; shared: FP8, a scale per 32 x 32 block
            let (f, index, e, nib, sr) = match e {
                Some(e) if e < x.n_exp => (&x.mx4, &x.mx4_index, e, true, 1),
                None => (&x.fp8, &x.fp8_index, 0, false, BLK),
                Some(e) => return Err(format!("{native}: expert {e} of {}", x.n_exp)),
            };
            let (out, inp) = if proj == 2 { (self.hid, self.inter) } else { (self.inter, self.hid) };
            let cb = |k: usize| if nib { k / 2 } else { k };
            // a unit: the codes of its 128 rows (gate, up) or columns (down), then their scales
            let (ucodes, uscales) = if proj == 2 { (out * cb(UNIT), out / sr * (UNIT / BLK)) } else { (UNIT * cb(inp), UNIT / sr * (inp / BLK)) };
            let mut offs = Vec::with_capacity(x.units);
            for u in 0..x.units {
                let i = ((e * 3 + proj) * x.units + u) * 2;
                let (at, n) = (index[i] as u64, index[i + 1] as u64);
                if n != (ucodes + uscales) as u64 {
                    return Err(format!("{native}: unit {u} has {n} bytes, not {}", ucodes + uscales));
                }
                offs.push((at, at + ucodes as u64));
            }
            let place = if proj == 2 { Place::InUnits { unit: UNIT, offs } } else { Place::OutUnits { unit: UNIT, offs } };
            return Ok(QMat { f: f.clone(), out, inp, nib, sr, place, ubytes: (ucodes + uscales) as u64 });
        }
        let (f, x, r) = self.record(native)?.ok_or_else(|| format!("the release has no matrix {native}"))?;
        if r[0] != exlr8::FP8_B32 {
            return Err(format!("{native}: record format {}, gr8 reads FP8 with E8M0 per 32 x 32 (format 8) here", r[0]));
        }
        let (out, inp) = (r[1] as usize, r[2] as usize);
        let (ci, si) = (f.info(&format!("{x}.fp8"))?, f.info(&format!("{x}.e8m0"))?);
        let (c, s) = (ci.start, si.start);
        let place = if r[4] == exlr8::SIDE_IN {
            // input side: [in / b][out][b] codes and [in / b][out / 32][b / 32] scales, b from the tensor's shape (the
            // record's unit field is the sharding unit, a multiple of b)
            let unit = match ci.shape[..] { [nb, o, b] if o == out && nb * b == inp && b % BLK == 0 => b, _ => 0 };
            if unit == 0 || si.shape != [inp / unit, out / BLK, unit / BLK] {
                return Err(format!("{native}: input-side codes {:?} / scales {:?} for [{out},{inp}]", ci.shape, si.shape));
            }
            let offs = (0..inp / unit).map(|u| (c + (u * out * unit) as u64, s + (u * (out / BLK) * (unit / BLK)) as u64)).collect();
            Place::InUnits { unit, offs }
        } else {
            Place::Rows { codes: c, scales: s }
        };
        Ok(QMat { f: f.clone(), out, inp, nib: false, sr: BLK, place, ubytes: 0 })
    }

    /// (out, in) of a matrix (`X` or `X.weight`), or the shape of a plain tensor.
    fn dims(&self, native: &str) -> Result<Vec<usize>, String> {
        let m = native.strip_suffix(".weight").unwrap_or(native);
        if self.expert(native).is_some() {
            let q = self.qmat(m)?;
            return Ok(vec![q.out, q.inp]);
        }
        if let Some((_, _, r)) = self.record(m)? {
            return Ok(vec![r[1] as usize, r[2] as usize]);
        }
        Ok(self.plain(native)?.shape)
    }

    /// Every record and plain tensor of the layer and vocabulary files, as native names (the coverage check).
    fn names(&self) -> Vec<String> {
        let mut out = Vec::new();
        let mut add = |pre: &str, f: &StFile| {
            for n in f.names() {
                let (base, suffix) = n.rsplit_once('.').unwrap_or((n, ""));
                if ["checksum", "fp8", "e8m0", "bf16", "fp32"].contains(&suffix) {
                    continue;
                }
                out.push(if suffix == "record" { format!("{pre}{base}") } else { format!("{pre}{n}") });
            }
        };
        for (&l, f) in &self.dense {
            add(&if l >= MTP0 { format!("mtp.{}.", l - MTP0) } else { format!("layers.{l}.") }, f);
        }
        add("", &self.vocab);
        out.into_iter().map(|n| match n.as_str() { "embed_tokens" => "embed".into(), "lm_head" => "head".into(), _ => n }).collect()
    }
}

// ---------------------------------------------------------------------------------------------------- planner

/// Release tensors that are not part of the text model: skipped on purpose.
fn skipped(name: &str) -> bool {
    name.ends_with(".bias_vl")
}

fn label(dtype: &str) -> Result<&'static str, String> {
    match dtype {
        "BF16" => Ok("bf16"),
        "F32" => Ok("f32"),
        "U8" | "I8" => Ok("u8"),
        d => Err(format!("dtype {d}")),
    }
}

fn nbytes(dtype: &str, shape: &[i64]) -> u64 {
    let e = match dtype { "bf16" | "f16" => 2, "f32" => 4, "i64" => 8, _ => 1 };
    shape.iter().product::<i64>() as u64 * e
}

/// What a rank serves of one Engram table: global rows [first, first + valid) of the layer's `engram.embed` (E4M3
/// rows of `dim`, E8M0 scales of `sdim` per row), in the release's split files. `part` = rows / nr rounded up; the
/// rows past `valid` are padding (weight 0, scale PAD_SCALE).
#[derive(Clone, Debug)]
pub struct EngramSlice {
    pub layer: usize,
    pub rows: usize,
    pub part: usize,
    pub first: usize,
    pub valid: usize,
    pub dim: usize,
    pub sdim: usize,
    pub segs: Vec<EngramSeg>,
}

/// Rows [row0, row0 + rows) of a table, in one split file: the weights at `w_off`, the scales at `s_off` (absolute).
#[derive(Clone, Debug)]
pub struct EngramSeg {
    pub path: String,
    pub row0: usize,
    pub rows: usize,
    pub w_off: u64,
    pub s_off: u64,
}

impl EngramSlice {
    pub fn bytes(&self) -> u64 {
        (self.part * (self.dim + self.sdim)) as u64
    }
}

/// Options of a plan. The defaults plan everything and check that the release is fully accounted for.
#[derive(Clone, Default)]
pub struct Opts {
    /// plan only these layers (main layers 0..39, then the DSpark layers 40..42); None = all
    pub layers: Option<Vec<usize>>,
    /// plan only the first n routed experts of each layer (a verification sample; shapes change)
    pub experts: Option<usize>,
    /// leave out the G.* tensors
    pub no_globals: bool,
    /// plan from manifests/exlr8-manifest-NAME.json: its layers with numeric widths load exlr8 trellis experts
    /// (GR8_DS_MANIFEST); None = every layer native
    pub manifest: Option<String>,
}

/// Blocks of 128 in an expert (2304 wide): the exlr8 v2 halves of a DS41 layer.
const NH_DS41: usize = 18;

/// The trellis layers of a release manifest: those whose first gate entry is a width (a number), with their widths.
fn trellis_layers(dir: &str, name: &str) -> Result<BTreeMap<usize, v2plan::Widths>, String> {
    let path = format!("{dir}/manifests/exlr8-manifest-{name}.json");
    let man = v2plan::width_manifest(&path)?;
    let mut out = BTreeMap::new();
    for (k, e) in man.get("layers").and_then(Json::obj).ok_or(format!("{path}: no layers"))? {
        let l: usize = k.parse().map_err(|_| format!("{path}: layer {k}"))?;
        let first = e.get("gate").and_then(Json::arr).and_then(|g| g.first()).and_then(Json::arr).and_then(|r| r.first());
        if first.and_then(Json::int).is_none() {
            continue;
        }
        out.insert(l, v2plan::widths_nh(&man, l, NH_DS41)?.ok_or(format!("{path}: layer {l}"))?);
    }
    Ok(out)
}

/// The byte ranges of the release a rank reads: (path relative to the release, offset, bytes).
pub type Reads = BTreeMap<String, Vec<(u64, u64)>>;

struct Planner<'a> {
    b: Builder,
    rel: &'a Release,
    used: BTreeSet<String>,
    reads: Reads,
    nr: usize,
    rank: usize,
    opts: Opts,
    trellis: BTreeMap<usize, v2plan::Widths>,
    slices: Option<v2plan::Slices>,
}

impl Planner<'_> {
    /// Notes ranges of `f` this rank reads (the fetch list), and the file's header.
    fn note(&mut self, f: &StFile, runs: impl IntoIterator<Item = (u64, u64)>) {
        let rel = f.path.strip_prefix(&format!("{}/", self.rel.dir)).unwrap_or(&f.path).to_string();
        let v = self.reads.entry(rel).or_default();
        v.extend(runs);
    }

    /// A tensor copied from a file range.
    fn copy(&mut self, name: String, dtype: &'static str, shape: Vec<i64>, s: &Src, off: u64) -> Result<(), String> {
        let n = nbytes(dtype, &shape);
        if off + n > s.len {
            return Err(format!("{}: a tensor is smaller than {name}", s.f.path));
        }
        let o = self.b.begin(name, dtype, shape, n, None);
        let fi = self.b.file(s.f.path.clone());
        self.b.seg(o, n, Op::Copy, fi, s.start + off);
        self.note(&s.f.clone(), [(s.start + off, n)]);
        Ok(())
    }

    /// A vector or small tensor as stored, under its gr8 name.
    fn vector(&mut self, name: String, native: &str) -> Result<(), String> {
        let s = self.rel.plain(native)?;
        self.used.insert(native.to_string());
        let dt = label(&s.dtype).map_err(|e| format!("{native}: {e}"))?;
        let shape = s.shape.iter().map(|&d| d as i64).collect();
        self.copy(name, dt, shape, &s, 0)
    }

    /// Consecutive tensors computed together at load: `make` returns each tensor's bytes, in order.
    fn gen(&mut self, what: String, specs: Vec<(String, &'static str, Vec<i64>)>,
           make: impl Fn() -> Result<Vec<Vec<u8>>, String> + Send + Sync + 'static) {
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
            if parts.len() != offs.len() {
                return Err(format!("{} parts, {} planned", parts.len(), offs.len()));
            }
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

    /// One manifest tensor made of contiguous `Op::Gen` stripes, so load workers compute them in parallel.
    fn gen_striped(&mut self, what: &str, name: String, dtype: &'static str, shape: Vec<i64>,
                   stripes: Vec<(u64, Box<dyn Fn() -> Result<Vec<u8>, String> + Send + Sync>)>) {
        let total = nbytes(dtype, &shape);
        let base = self.b.begin(name, dtype, shape, total, None);
        let mut off = 0u64;
        for (span, make) in stripes {
            let gi = self.b.gens.len();
            self.b.gens.push(Gen::new(what.to_string(), span, make));
            self.b.segs.push(crate::plan::Seg { pack_off: base + off, nbytes: span, op: Op::Gen, file_off: 0, file: gi });
            off += span;
        }
        assert_eq!(off, total, "{what}: stripes do not cover the tensor");
    }

    /// An FP8 32 x 32 matrix `native` (a dense record, or the shared expert's units), rows `rows` and columns `cols` of
    /// it, as `P.mx.codes` u8 [n,k], `P.mx.scales` u8 [n,k/32] (each block-row scale repeated over its 32 rows: exact)
    /// and `P.mx.gv`, the decode record.
    fn mx(&mut self, p: &str, native: &str, rows: Option<Range<usize>>, cols: Option<Range<usize>>) -> Result<(), String> {
        let q = self.rel.qmat(native)?;
        self.used.insert(native.to_string());
        if q.nib || q.sr != BLK {
            return Err(format!("{native}: not an FP8 32 x 32 matrix"));
        }
        let rows = rows.unwrap_or(0..q.out);
        let cols = cols.unwrap_or(0..q.inp);
        q.check(&rows, &cols)?;
        if rows.len() % 16 != 0 {
            return Err(format!("{native}: rows {rows:?} are not 16-row aligned"));
        }
        let (c, s) = q.ranges(&rows, &cols);
        self.note(&q.f.clone(), c.into_iter().chain(s));
        let (n, k) = (rows.len(), cols.len());
        let name = |s: &str| format!("{p}.mx.{s}");
        let specs = vec![(name("codes"), "u8", vec![n as i64, k as i64]),
                         (name("scales"), "u8", vec![n as i64, (k / BLK) as i64]),
                         (name("gv"), "u8", vec![(n / 16) as i64, (k / BLK) as i64, 528])];
        self.gen(p.to_string(), specs, move || {
            let codes = q.codes(&rows, &cols)?;
            let raw = q.scales(&rows, &cols)?;
            let (b0, kc) = (rows.start / BLK, k / BLK);
            let mut scales = Vec::with_capacity(n * kc);
            for r in rows.clone() {
                let br = (r / BLK - b0) * kc;
                scales.extend_from_slice(&raw[br..br + kc]);
            }
            let gv = records::mxfp8_record(&codes, &scales, k);
            Ok(vec![codes, scales, gv])
        });
        Ok(())
    }

    /// A bf16 matrix `native` [N,K], rows `rows` of it, as stored, plus its decode record `P.gv` when `gv` and
    /// rows % 16 == 0 and K % 16 == 0.
    fn bf16_mat(&mut self, p: &str, native: &str, rows: Option<Range<usize>>, gv: bool) -> Result<(), String> {
        let native = native.strip_suffix(".weight").unwrap_or(native);
        let (s, nn, k) = self.rel.bf16(native)?;
        self.used.insert(native.to_string());
        let rows = rows.unwrap_or(0..nn);
        if rows.is_empty() || rows.end > nn {
            return Err(format!("{native}: rows {rows:?} of {nn}"));
        }
        let n = rows.len();
        self.copy(p.to_string(), "bf16", vec![n as i64, k as i64], &s, (rows.start * k * 2) as u64)?;
        if gv && n % 16 == 0 && k % 16 == 0 {
            // stripes of whole 16-row tiles, about 16 MB of input each
            let per = ((16usize << 20) / (k * 2) / 16).max(1) * 16;
            let mut stripes: Vec<(u64, Box<dyn Fn() -> Result<Vec<u8>, String> + Send + Sync>)> = Vec::new();
            let mut a = 0;
            while a < n {
                let b = (a + per).min(n);
                let (s2, r0, m) = (s.clone(), rows.start + a, b - a);
                stripes.push(((m / 16 * (k / 16) * 512) as u64, Box::new(move || {
                    let raw = s2.read((r0 * k * 2) as u64, (m * k * 2) as u64)?;
                    let w: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                    Ok(records::bf16_record(&w, k))
                })));
                a = b;
            }
            self.gen_striped(p, format!("{p}.gv"), "u8", vec![(n / 16) as i64, (k / 16) as i64, 512], stripes);
        }
        Ok(())
    }

    /// A layer's routed experts as the mx4g record streams of this rank's intermediate slice:
    /// `P.x.fc1` u8 [E, Iw/16, K1/64, 1088] (gate = w1 rows, up = w3 rows, per 16-row chunk and 64-k group) and
    /// `P.x.fc2` u8 [E, H/16, Iw/64, 544] (w2 with its column slice).
    fn routed(&mut self, p: &str, pre: &str, n_exp: usize) -> Result<(), String> {
        let mut exps = Vec::with_capacity(n_exp);
        for e in 0..n_exp {
            let m = |w: &str| self.rel.qmat(&format!("{pre}ffn.experts.{e}.{w}"));
            exps.push([m("w1")?, m("w3")?, m("w2")?]);
        }
        let (i_full, k1, h) = (exps[0][0].out, exps[0][0].inp, exps[0][2].out);
        if k1 % KG != 0 || h % 16 != 0 || exps[0][2].inp != i_full || !exps[0][0].nib {
            return Err(format!("{pre}experts: K {k1} / H {h} are not multiples of {KG} / 16, or not MXFP4"));
        }
        let isl = inter_slice(i_full, self.nr, self.rank)?;
        let iw = isl.len();
        // the units this rank reads, whole: their codes and scales are one range each
        for x in &exps {
            for q in x {
                let (Place::OutUnits { unit, offs } | Place::InUnits { unit, offs }) = &q.place else { unreachable!() };
                let runs: Vec<(u64, u64)> = (isl.start / unit..isl.end.div_ceil(*unit)).map(|u| (offs[u].0, q.ubytes)).collect();
                self.note(&q.f.clone(), runs);
            }
        }
        let (per1, per2) = (iw / 16 * (k1 / KG) * 2 * GR, h / 16 * (iw / KG) * GR);
        let e_i = n_exp as i64;
        for (which, per) in [("fc1", per1), ("fc2", per2)] {
            let mut stripes: Vec<(u64, Box<dyn Fn() -> Result<Vec<u8>, String> + Send + Sync>)> = Vec::new();
            for group in exps.chunks(EXP_GROUP) {
                let xs: Vec<[QMat; 3]> = group.to_vec();
                let span = (xs.len() * per) as u64;
                let isl = isl.clone();
                if which == "fc1" {
                    stripes.push((span, Box::new(move || {
                        let mut out = Vec::with_capacity(xs.len() * per1);
                        for [g, u, _] in &xs {
                            let all = 0..k1;
                            let (gc, gs, uc, us) = (g.codes(&isl, &all)?, g.scales(&isl, &all)?, u.codes(&isl, &all)?, u.scales(&isl, &all)?);
                            out.extend_from_slice(&mx4g_fc1(&gc, &gs, &uc, &us, iw, k1));
                        }
                        Ok(out)
                    })));
                } else {
                    stripes.push((span, Box::new(move || {
                        let mut out = Vec::with_capacity(xs.len() * per2);
                        for [_, _, d] in &xs {
                            let rows = 0..h;
                            out.extend_from_slice(&mx4g_fc2(&d.codes(&rows, &isl)?, &d.scales(&rows, &isl)?, h, iw, 0, iw));
                        }
                        Ok(out)
                    })));
                }
            }
            let shape = if which == "fc1" { vec![e_i, (iw / 16) as i64, (k1 / KG) as i64, (2 * GR) as i64] }
                        else { vec![e_i, (h / 16) as i64, (iw / KG) as i64, GR as i64] };
            self.gen_striped(&format!("{p}.x.{which}"), format!("{p}.x.{which}"), "u8", shape, stripes);
        }
        Ok(())
    }

    fn layer(&mut self, l: usize, pre: &str) -> Result<(), String> {
        let p = format!("L{l:03}");
        let t = |s: &str| format!("{pre}{s}");
        self.vector(format!("{p}.attn_norm"), &t("attn_norm.weight"))?;
        self.vector(format!("{p}.ffn_norm"), &t("ffn_norm.weight"))?;
        for hc in ["attn", "ffn"] {
            for part in ["fn", "base", "scale"] {
                self.vector(format!("{p}.hc_{hc}.{part}"), &t(&format!("hc_{hc}_{part}")))?;
            }
        }
        // attention: this rank's O-groups. Their heads are the q_b rows, the o_a rows and the o_b columns it holds.
        let sink = self.rel.plain(&t("attn.attn_sink"))?;
        self.used.insert(t("attn.attn_sink"));
        let heads = sink.shape[0];
        let g = o_groups(l, self.nr, self.rank);
        if heads % O_GROUPS != 0 || g.is_empty() {
            return Err(format!("{p}: {heads} heads over {O_GROUPS} groups, rank {} of {} holds {g:?}", self.rank, self.nr));
        }
        let hpg = heads / O_GROUPS;
        let qb = self.rel.dims(&t("attn.wq_b"))?;
        let hd = qb[0] / heads;
        let oa = self.rel.dims(&t("attn.wo_a"))?;
        let ob = self.rel.dims(&t("attn.wo_b"))?;
        self.mx(&format!("{p}.q_a"), &t("attn.wq_a"), None, None)?;
        self.vector(format!("{p}.q_a_norm"), &t("attn.q_norm.weight"))?;
        self.mx(&format!("{p}.q_b"), &t("attn.wq_b"), Some(g.start * hpg * hd..g.end * hpg * hd), None)?;
        self.mx(&format!("{p}.kv"), &t("attn.wkv"), None, None)?;
        self.vector(format!("{p}.kv_norm"), &t("attn.kv_norm.weight"))?;
        self.copy(format!("{p}.attn_sink"), "f32", vec![(g.len() * hpg) as i64], &sink, (g.start * hpg * 4) as u64)?;
        let rpg = oa[0] / O_GROUPS;
        self.mx(&format!("{p}.o_a"), &t("attn.wo_a"), Some(g.start * rpg..g.end * rpg), None)?;
        let cpg = ob[1] / O_GROUPS;
        self.mx(&format!("{p}.o_b"), &t("attn.wo_b"), None, Some(g.start * cpg..g.end * cpg))?;
        if self.rel.has(&t("attn.compressor.wkv")) {
            self.bf16_mat(&format!("{p}.cmp.wkv"), &t("attn.compressor.wkv"), None, true)?;
        }
        if self.rel.has(&t("attn.compressor.wgate")) {
            self.bf16_mat(&format!("{p}.cmp.wgate"), &t("attn.compressor.wgate"), None, true)?;
        }
        if self.rel.has(&t("attn.compressor.norm.weight")) {
            self.vector(format!("{p}.cmp.norm"), &t("attn.compressor.norm.weight"))?;
        }
        if self.rel.has(&t("attn.indexer.wq_b")) {
            self.mx(&format!("{p}.idx.wq_b"), &t("attn.indexer.wq_b"), None, None)?;
        }
        if self.rel.has(&t("attn.indexer.weights_proj")) {
            self.bf16_mat(&format!("{p}.idx.weights_proj"), &t("attn.indexer.weights_proj"), None, true)?;
        }
        if self.rel.has(&t("attn.indexer.wk")) {
            self.bf16_mat(&format!("{p}.idx.wk"), &t("attn.indexer.wk"), None, true)?;
        }
        if self.rel.has(&t("attn.indexer.k_norm.weight")) {
            self.vector(format!("{p}.idx.k_norm"), &t("attn.indexer.k_norm.weight"))?;
        }
        // the MoE block
        self.bf16_mat(&format!("{p}.router.w"), &t("ffn.gate"), None, true)?;
        self.vector(format!("{p}.router.bias"), &t("ffn.gate.bias"))?;
        let n_exp = self.rel.dims(&t("ffn.gate"))?[0];
        if self.trellis.contains_key(&l) {
            self.trellis_experts(l)?;
            return self.layer_tail(pre, &p);
        }
        let i_sh = self.rel.dims(&t("ffn.shared_experts.w1"))?[0];
        let sl = inter_slice(i_sh, self.nr, self.rank)?;
        if sl.len() % BLK != 0 {
            return Err(format!("{p}: shared expert slice {sl:?} is not 32-aligned"));
        }
        self.mx(&format!("{p}.sh.w1"), &t("ffn.shared_experts.w1"), Some(sl.clone()), None)?;
        self.mx(&format!("{p}.sh.w3"), &t("ffn.shared_experts.w3"), Some(sl.clone()), None)?;
        self.mx(&format!("{p}.sh.w2"), &t("ffn.shared_experts.w2"), None, Some(sl))?;
        let n_exp = self.opts.experts.map_or(n_exp, |m| m.min(n_exp));
        self.routed(&p, pre, n_exp)?;
        self.layer_tail(pre, &p)
    }

    /// Layer l's experts as exlr8 trellis (the manifest gave it numeric widths): the tr3.* tensors of the 128-blocks
    /// this rank owns, routed and shared, built from the width files and the index slices (src/v2plan.rs).
    fn trellis_experts(&mut self, l: usize) -> Result<(), String> {
        let (nr, rank, dir) = (self.nr, self.rank, self.rel.dir.clone());
        let slices = self.slices.as_ref().ok_or("trellis layers without index slices")?;
        let first = self.b.segs.len();
        v2plan::v2_layer(&mut self.b, l, &self.trellis[&l], slices, &|l, k| v2plan::width_file(&dir, l, k),
                         &|e| Ok(v2plan::formula_halves_nh(e, l, nr, rank, NH_DS41)))?;
        // the fetch list: the width-file ranges those tensors copy
        let runs: Vec<(String, u64, u64)> = self.b.segs[first..].iter().filter(|g| g.op == Op::Copy)
            .map(|g| (self.b.files[g.file].clone(), g.file_off, g.nbytes)).collect();
        for (path, off, n) in runs {
            let rel = path.strip_prefix(&format!("{dir}/")).unwrap_or(&path).to_string();
            self.reads.entry(rel).or_default().push((off, n));
        }
        Ok(())
    }

    /// The rest of a layer after its experts: the DSpark heads and the Engram weights a layer has.
    fn layer_tail(&mut self, pre: &str, p: &str) -> Result<(), String> {
        let t = |s: &str| format!("{pre}{s}");
        // DSpark layers: the first one joins the trunk, the last one carries the output heads
        if self.rel.has(&t("main_norm.weight")) {
            self.vector(format!("{p}.main_norm"), &t("main_norm.weight"))?;
        }
        if self.rel.has(&t("main_proj")) {
            self.mx(&format!("{p}.main_proj"), &t("main_proj"), None, None)?;
        }
        if self.rel.has(&t("norm.weight")) {
            self.vector(format!("{p}.norm"), &t("norm.weight"))?;
        }
        if self.rel.has(&t("confidence_head.proj")) {
            self.bf16_mat(&format!("{p}.confidence_head.proj"), &t("confidence_head.proj"), None, true)?;
        }
        for h in ["embed", "head"] {
            if self.rel.has(&t(&format!("markov_head.{h}"))) {
                self.bf16_mat(&format!("{p}.markov_head.{h}"), &t(&format!("markov_head.{h}")), None, true)?;
            }
        }
        // Engram: the projection and the two key/query weights live in the image, the table does not
        if self.rel.has(&t("engram.wkv")) {
            self.mx(&format!("{p}.engram.wkv"), &t("engram.wkv"), None, None)?;
            self.bf16_mat(&format!("{p}.engram.q_weight"), &t("engram.q_weight"), None, true)?;
            self.bf16_mat(&format!("{p}.engram.k_weight"), &t("engram.k_weight"), None, true)?;
        }
        Ok(())
    }

    fn globals(&mut self) -> Result<(), String> {
        self.bf16_mat("G.embed", "embed", None, false)?;
        self.vector("G.final_norm".into(), "norm.weight")?;
        let dims = self.rel.dims("head")?;
        if dims[0] % VTILE != 0 {
            return Err(format!("head has {} rows, not a multiple of {VTILE}", dims[0]));
        }
        let tiles = vocab_tiles(dims[0] / VTILE, self.nr, self.rank);
        self.bf16_mat("G.lm_head", "head", Some(tiles.start * VTILE..tiles.end * VTILE), true)
    }

    /// Engram tables a rank serves: global rows [rank * part, ...) of every table, with part = ceil(n / nr) (the
    /// reference's sharding, ParallelEngramEmbedding), as runs of the release's split files.
    fn engram(&mut self) -> Result<Vec<EngramSlice>, String> {
        let mut out = Vec::new();
        let tables: Vec<(usize, Vec<Arc<StFile>>)> = self.rel.engram.iter().map(|(&l, v)| (l, v.clone())).collect();
        for (layer, parts) in tables {
            let mut files = Vec::new();
            let (mut rows, mut dim, mut sdim) = (0, 0, 0);
            for f in &parts {
                let r = i64s(f, "engram.embed.record")?;
                let (w, s) = (f.info("engram.embed.fp8")?, f.info("engram.embed.e8m0")?);
                if r[0] != exlr8::MXFP8 || w.shape.len() != 2 || s.shape.len() != 2 || s.shape[0] != w.shape[0] || w.shape[1] != s.shape[1] * BLK {
                    return Err(format!("{}: not an MXFP8 table [n,dim] with E8M0 [n,dim/32]", f.path));
                }
                if dim != 0 && (w.shape[1], s.shape[1]) != (dim, sdim) {
                    return Err(format!("{}: row width differs from the table's other files", f.path));
                }
                (dim, sdim) = (w.shape[1], s.shape[1]);
                files.push((f.clone(), rows, w.shape[0], w.start, s.start));
                rows += w.shape[0];
            }
            let part = rows.div_ceil(self.nr);
            let first = (self.rank * part).min(rows);
            let valid = part.min(rows - first);
            let mut segs = Vec::new();
            for (f, row0, n, w0, s0) in files {
                let (a, b) = (first.max(row0), (first + valid).min(row0 + n));
                if a >= b {
                    continue;
                }
                let (w_off, s_off) = (w0 + ((a - row0) * dim) as u64, s0 + ((a - row0) * sdim) as u64);
                self.note(&f, [(w_off, ((b - a) * dim) as u64), (s_off, ((b - a) * sdim) as u64)]);
                segs.push(EngramSeg { path: f.path.clone(), row0: a, rows: b - a, w_off, s_off });
            }
            out.push(EngramSlice { layer, rows, part, first, valid, dim, sdim, segs });
        }
        Ok(out)
    }
}

/// A rank's plan: the image (G.* tensors, then L000..), the Engram tables it serves, and the release ranges it reads.
pub struct Ds41Plan {
    pub img: V2Image,
    pub engram: Vec<EngramSlice>,
    pub reads: Reads,
}

/// Plans rank `rank` of `nr` from the release under `dir`. Reads headers and index files only. With default options
/// the plan also checks that every tensor of the release's layer and vocabulary files is either in the plan or
/// skipped on purpose (the vision routing bias).
pub fn plan(dir: &str, nr: usize, rank: usize, opts: &Opts) -> Result<Ds41Plan, String> {
    plan_from(&Release::open(dir)?, nr, rank, opts)
}

pub fn plan_from(rel: &Release, nr: usize, rank: usize, opts: &Opts) -> Result<Ds41Plan, String> {
    if nr == 0 || nr > O_GROUPS || rank >= nr {
        return Err(format!("rank {rank} of {nr}: attention splits {O_GROUPS} O-groups, so 1 <= nr <= {O_GROUPS}"));
    }
    let trellis = match &opts.manifest {
        Some(m) if m != "native" => trellis_layers(&rel.dir, m)?,
        _ => BTreeMap::new(),
    };
    let slices = if trellis.is_empty() { None } else { Some(v2plan::Slices::open(&rel.dir, nr, &[rank])?) };
    let mut p = Planner { b: Builder::new(), rel, used: BTreeSet::new(), reads: BTreeMap::new(), nr, rank, opts: opts.clone(),
                          trellis, slices };
    let n_main = rel.dense.keys().filter(|&&l| l < MTP0).count();
    if rel.dense.keys().copied().take(n_main).ne(0..n_main) {
        return Err(format!("{}: the main layers are not 0..{n_main}", rel.dir));
    }
    if !opts.no_globals {
        p.globals()?;
    }
    for l in rel.dense.keys().copied().collect::<Vec<_>>() {
        if opts.layers.as_ref().is_some_and(|r| !r.contains(&l)) {
            continue;
        }
        let pre = if l < MTP0 { format!("layers.{l}.") } else { format!("mtp.{}.", l - MTP0) };
        p.layer(l, &pre)?;
    }
    let engram = p.engram()?;
    if opts.layers.is_none() && opts.experts.is_none() && !opts.no_globals {
        let mut missing: Vec<String> = rel.names().into_iter().filter(|n| !p.used.contains(n) && !skipped(n)).collect();
        if !missing.is_empty() {
            missing.sort();
            return Err(format!("{} release tensors are neither planned nor skipped, e.g. {:?}", missing.len(), &missing[..missing.len().min(8)]));
        }
    }
    // the records the planner reads to plan, in every layer, vocabulary and Engram file
    let files: Vec<Arc<StFile>> = rel.dense.values().chain([&rel.vocab]).chain(rel.engram.values().flatten()).cloned().collect();
    for f in files {
        let runs: Vec<(u64, u64)> = f.names().iter().filter(|n| n.ends_with(".record")).filter_map(|n| f.info(n).ok())
            .map(|i| (i.start, i.end - i.start)).collect();
        p.note(&f, runs);
    }
    // every file the plan opens: its header too (the reads above are tensor data)
    let headers: Vec<(String, u64)> = p.reads.keys().map(|k| {
        let path = format!("{}/{k}", rel.dir);
        let n = std::fs::File::open(&path).ok().and_then(|f| {
            let mut b = [0u8; 8];
            std::os::unix::fs::FileExt::read_exact_at(&f, &mut b, 0).ok().map(|_| 8 + u64::from_le_bytes(b))
        });
        (k.clone(), n.unwrap_or(0))
    }).collect();
    for (k, n) in headers {
        p.reads.get_mut(&k).unwrap().push((0, n));
    }
    for (_, v) in p.reads.iter_mut() {
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
    let (b, reads) = (p.b, p.reads);
    let bytes = b.cur.div_ceil(4096) * 4096;
    let plan = Plan::from_segs(b.files, b.segs, b.mems, b.gens, bytes)?;
    Ok(Ds41Plan { img: V2Image { rank, tensors: b.tensors, bytes, plan }, engram, reads })
}

// ---------------------------------------------------------------------------------------------------- mx4g records

/// One record: rows r0 .. r0 + 15, k k0 .. k0 + 63 (k0 % 64 == 0), of a matrix whose rows are `ps` bytes of packed
/// nibbles (the slice starts `pc` bytes in) and `es` bytes of E8M0 scales (starting `ec` in). Layout:
/// engine/kernels/mxfp4.cuh. Lane l = 4r + c reads {row r byte c, row r+8 byte c, row r byte c+4, row r+8 byte c+4}
/// of each 8-byte k16 step; the code bytes are permuted, never re-nibbled.
#[allow(clippy::too_many_arguments)]
fn mx4g_group(dst: &mut [u8], p: &[u8], ps: usize, pc: usize, e: &[u8], es: usize, ec: usize, r0: usize, k0: usize) {
    for lane in 0..32 {
        let (r, c) = (lane >> 2, lane & 3);
        for t in 0..4 {
            let j = pc + k0 / 2 + t * 8 + c;
            let o = lane * 16 + t * 4;
            dst[o] = p[(r0 + r) * ps + j];
            dst[o + 1] = p[(r0 + r + 8) * ps + j];
            dst[o + 2] = p[(r0 + r) * ps + j + 4];
            dst[o + 3] = p[(r0 + r + 8) * ps + j + 4];
        }
    }
    let g = ec + k0 / BLK;
    for r in 0..8 {
        let s = SOFF + r * 4;
        dst[s] = e[(r0 + r) * es + g];
        dst[s + 1] = e[(r0 + r + 8) * es + g];
        dst[s + 2] = e[(r0 + r) * es + g + 1];
        dst[s + 3] = e[(r0 + r + 8) * es + g + 1];
    }
}

/// fc1: gate and up [r][k] (packed bytes [r][k/2], scales [r][k/32]) -> [r/16][k/64][gate record | up record].
pub fn mx4g_fc1(gp: &[u8], ge: &[u8], up: &[u8], ue: &[u8], r: usize, k: usize) -> Vec<u8> {
    assert!(r % 16 == 0 && k % KG == 0 && gp.len() == r * k / 2 && up.len() == gp.len() && ge.len() == r * k / BLK && ue.len() == ge.len());
    let mut out = vec![0u8; r / 16 * (k / KG) * 2 * GR];
    for ch in 0..r / 16 {
        for g in 0..k / KG {
            let d = &mut out[(ch * (k / KG) + g) * 2 * GR..][..2 * GR];
            mx4g_group(&mut d[..GR], gp, k / 2, 0, ge, k / BLK, 0, ch * 16, g * KG);
            mx4g_group(&mut d[GR..], up, k / 2, 0, ue, k / BLK, 0, ch * 16, g * KG);
        }
    }
    out
}

/// fc2: the columns [c0, c0 + kn) of a matrix of `rows` rows and `kfull` columns (packed [rows][kfull/2], scales
/// [rows][kfull/32]) -> [rows/16][kn/64][record]. c0 and kn are multiples of 64.
pub fn mx4g_fc2(p: &[u8], e: &[u8], rows: usize, kfull: usize, c0: usize, kn: usize) -> Vec<u8> {
    assert!(rows % 16 == 0 && kn % KG == 0 && c0 % KG == 0 && c0 + kn <= kfull && p.len() == rows * kfull / 2 && e.len() == rows * kfull / BLK);
    let mut out = vec![0u8; rows / 16 * (kn / KG) * GR];
    for ot in 0..rows / 16 {
        for g in 0..kn / KG {
            let d = &mut out[(ot * (kn / KG) + g) * GR..][..GR];
            mx4g_group(d, p, kfull / 2, c0 / 2, e, kfull / BLK, c0 / BLK, ot * 16, g * KG);
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------------- sizes

/// The tensor's kind: its name without the `L###.` prefix.
fn kind(name: &str) -> String {
    let b = name.as_bytes();
    if b.len() > 5 && b[0] == b'L' && b[1..4].iter().all(u8::is_ascii_digit) && b[4] == b'.' { name[5..].to_string() } else { name.to_string() }
}

/// (tensors, bytes) by kind, in name order. Padding between tensors is not counted.
pub fn kinds(img: &V2Image) -> BTreeMap<String, (usize, u64)> {
    let mut m: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    for t in &img.tensors {
        let e = m.entry(kind(&t.name)).or_default();
        e.0 += 1;
        e.1 += t.nbytes;
    }
    m
}

/// The rank's sizes: tensor bytes, image bytes (4096-aligned), the Engram rows it serves.
pub fn size_report(img: &V2Image, engram: &[EngramSlice]) -> String {
    let mut out = String::new();
    let data: u64 = img.tensors.iter().map(|t| t.nbytes).sum();
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    out += &format!("rank {}: {} tensors, {:.3} GiB of data, image {:.3} GiB ({} bytes)\n", img.rank, img.tensors.len(), gib(data), gib(img.bytes), img.bytes);
    for (k, (n, b)) in kinds(img) {
        out += &format!("  {n:6} {:10.3} MiB  {k}\n", b as f64 / (1u64 << 20) as f64);
    }
    let dsp: u64 = img.tensors.iter().filter(|t| kind(&t.name) != t.name && t.name[1..4].parse::<usize>().is_ok_and(|l| l >= MTP0)).map(|t| t.nbytes).sum();
    out += &format!("  DSpark layers (L040+): {:.3} GiB of the data\n", gib(dsp));
    for e in engram {
        out += &format!("  engram L{:03}: rows {}..{} of {} (part {}, pad {}) in {} split files, {:.3} GiB\n", e.layer, e.first,
                        e.first + e.valid, e.rows, e.part, e.part - e.valid, e.segs.len(), gib(e.bytes()));
    }
    let eg: u64 = engram.iter().map(EngramSlice::bytes).sum();
    out += &format!("  engram total {:.3} GiB; weights image + engram {:.3} GiB\n", gib(eg), gib(img.bytes + eg));
    out
}

// ---------------------------------------------------------------------------------------------------- streaming, hash

/// Reads the image once, in order, with `workers` threads computing the pieces ahead: `sink(offset, bytes)` gets
/// consecutive pieces in increasing offset order (gaps between planned ranges are zeros the sink does not see).
pub fn stream(img: &V2Image, workers: usize, mut sink: impl FnMut(u64, &[u8]) -> Result<(), String>) -> Result<(), String> {
    const CHUNK: u64 = 64 << 20;
    const BATCH: u64 = 16 << 20;
    let mut items: Vec<(u64, u64)> = Vec::new();
    let mut cur: Option<(u64, u64)> = None;
    for s in img.plan.sources(0, img.bytes) {
        let (a, e) = (s.pack_off, s.pack_off + s.nbytes);
        if s.op == Op::Copy && s.nbytes > CHUNK {
            items.extend(cur.take());
            let mut x = a;
            while x < e {
                items.push((x, (x + CHUNK).min(e)));
                x += CHUNK;
            }
            continue;
        }
        let c = cur.get_or_insert((a, a));
        c.1 = e;
        if c.1 - c.0 >= BATCH {
            items.extend(cur.take());
        }
    }
    items.extend(cur.take());
    let workers = workers.max(1);
    let window = 2 * workers;
    let next = AtomicUsize::new(0);
    let abort = AtomicBool::new(false);
    let written = Mutex::new(0usize);
    let cv = Condvar::new();
    let (tx, rx) = mpsc::channel::<(usize, Result<Vec<u8>, String>)>();
    std::thread::scope(|sc| {
        for _ in 0..workers {
            let tx = tx.clone();
            let (items, next, abort, written, cv) = (&items, &next, &abort, &written, &cv);
            sc.spawn(move || loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                if i >= items.len() {
                    break;
                }
                let g = cv.wait_while(written.lock().unwrap(), |w| i >= *w + window && !abort.load(Ordering::SeqCst)).unwrap();
                drop(g);
                if abort.load(Ordering::SeqCst) {
                    break;
                }
                let (a, e) = items[i];
                let mut buf = vec![0u8; (e - a) as usize];
                let r = crate::read_range(&img.plan, a, &mut buf).map(|_| buf);
                if tx.send((i, r)).is_err() {
                    break;
                }
            });
        }
        drop(tx);
        let mut pending: BTreeMap<usize, Vec<u8>> = BTreeMap::new();
        let mut want = 0usize;
        let mut result = Ok(());
        while want < items.len() {
            let Ok((i, r)) = rx.recv() else { break };
            match r {
                Ok(b) => {
                    pending.insert(i, b);
                }
                Err(e) => {
                    result = Err(e);
                    break;
                }
            }
            while let Some(b) = pending.remove(&want) {
                if let Err(e) = sink(items[want].0, &b) {
                    result = Err(e);
                    break;
                }
                want += 1;
                *written.lock().unwrap() = want;
                cv.notify_all();
            }
            if result.is_err() {
                break;
            }
        }
        if result.is_ok() && want < items.len() {
            result = Err("a worker stopped before the image was complete".into());
        }
        abort.store(true, Ordering::SeqCst);
        cv.notify_all();
        result
    })
}

/// Tracks the sha256 of every tensor while the image streams past in offset order.
struct Hashes<'a> {
    tensors: &'a [Tensor],
    h: Vec<Sha256>,
}

impl<'a> Hashes<'a> {
    fn new(tensors: &'a [Tensor]) -> Hashes<'a> {
        Hashes { tensors, h: tensors.iter().map(|_| Sha256::new()).collect() }
    }

    fn feed(&mut self, off: u64, buf: &[u8]) {
        let end = off + buf.len() as u64;
        let first = self.tensors.partition_point(|t| t.offset + t.nbytes <= off);
        for (j, t) in self.tensors.iter().enumerate().skip(first) {
            if t.offset >= end {
                break;
            }
            let (a, b) = (t.offset.max(off), (t.offset + t.nbytes).min(end));
            if a < b {
                self.h[j].update(&buf[(a - off) as usize..(b - off) as usize]);
            }
        }
    }

    fn finish(self) -> Vec<String> {
        self.h.into_iter().map(Sha256::hex).collect()
    }
}

/// sha256 of every tensor of the planned image, straight from the plan (no pack file).
pub fn hash_image(img: &V2Image, workers: usize) -> Result<Vec<String>, String> {
    let mut h = Hashes::new(&img.tensors);
    stream(img, workers, |off, b| {
        h.feed(off, b);
        Ok(())
    })?;
    Ok(h.finish())
}

fn manifest_json(img: &V2Image, shas: &[String], nr: usize) -> String {
    let q = |s: &str| format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""));
    let mut out = format!("{{\n \"align\": 4096,\n \"format\": \"g6pack-v1\",\n \"model\": \"deepseek-v4.1-flash-1017\",\n \"pack_bytes\": {},\n \"rank\": {},\n \"tp\": {},\n \"tensors\": [", img.bytes, img.rank, nr);
    for (i, t) in img.tensors.iter().enumerate() {
        let shape: Vec<String> = t.shape.iter().map(i64::to_string).collect();
        out += &format!("{}\n  {{\"dtype\": {}, \"name\": {}, \"nbytes\": {}, \"offset\": {}, \"sha256\": \"{}\", \"shape\": [{}]}}",
                        if i > 0 { "," } else { "" }, q(&t.dtype), q(&t.name), t.nbytes, t.offset, shas[i], shape.join(", "));
    }
    out + "\n ]\n}\n"
}


/// One tensor of the image, read through the plan.
pub fn read_tensor(img: &V2Image, name: &str) -> Result<Vec<u8>, String> {
    let t = img.tensors.iter().find(|t| t.name == name).ok_or_else(|| format!("the image has no tensor {name}"))?;
    let mut buf = vec![0u8; t.nbytes as usize];
    crate::read_range(&img.plan, t.offset, &mut buf)?;
    Ok(buf)
}


fn q(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

/// The release ranges a rank reads, as JSON: {"files": {"<path in the release>": [[offset, bytes], ...]}}.
pub fn reads_json(reads: &Reads) -> String {
    let files: Vec<String> = reads.iter().map(|(k, v)| {
        let r: Vec<String> = v.iter().map(|(a, n)| format!("[{a}, {n}]")).collect();
        format!("  {}: [{}]", q(k), r.join(", "))
    }).collect();
    format!("{{\"files\": {{\n{}\n}}}}\n", files.join(",\n"))
}

/// The Engram tables a rank serves, as JSON (g6engram-v2): per table its rows, part, first, valid, dim, sdim and the
/// split-file runs (path, row0, rows, w_off, s_off: absolute byte offsets of the run's first row).
pub fn engram_json(engram: &[EngramSlice], nr: usize, rank: usize) -> String {
    let tables: Vec<String> = engram.iter().map(|e| {
        let segs: Vec<String> = e.segs.iter().map(|s| format!("{{\"path\": {}, \"row0\": {}, \"rows\": {}, \"w_off\": {}, \"s_off\": {}}}",
                                                               q(&s.path), s.row0, s.rows, s.w_off, s.s_off)).collect();
        format!("  {{\"layer\": {}, \"rows\": {}, \"part\": {}, \"first\": {}, \"valid\": {}, \"dim\": {}, \"sdim\": {}, \"segs\": [{}]}}",
                e.layer, e.rows, e.part, e.first, e.valid, e.dim, e.sdim, segs.join(", "))
    }).collect();
    format!("{{\"format\": \"g6engram-v2\", \"tp\": {nr}, \"rank\": {rank}, \"tables\": [\n{}\n]}}\n", tables.join(",\n"))
}

// ---------------------------------------------------------------------------------------------------- C interface

/// Plans rank `rank`'s image at TP `nr` from the release under `release_dir` (src/ds41.rs). Writes the image's
/// manifest (sha256 fields zero) to `out_manifest` and the Engram tables the rank serves (g6engram-v2) to
/// `out_engram`. Returns a handle for g6l_plan_load, g6l_plan_read and g6l_plan_free, or NULL with a message in `err`.
///
/// # Safety
/// The strings must be NUL-terminated, `err` must hold `errlen` bytes.
#[no_mangle]
pub unsafe extern "C" fn g6l_ds41_plan(release_dir: *const c_char, nr: u32, rank: u32, out_manifest: *const c_char,
                                       out_engram: *const c_char, err: *mut c_char, errlen: usize) -> *mut c_void {
    let t0 = Instant::now();
    let arg = |p, what| crate::cstr(p).ok_or(format!("g6l_ds41_plan: no {what}"));
    let r = (|| {
        let opts = Opts { manifest: gr8knobs::DS_MANIFEST.get().map(str::to_string), ..Opts::default() };
        let p = plan(&arg(release_dir, "release dir")?, nr as usize, rank as usize, &opts)?;
        let out = arg(out_manifest, "output manifest")?;
        let zeros = vec!["0".repeat(64); p.img.tensors.len()];
        std::fs::write(&out, manifest_json(&p.img, &zeros, nr as usize)).map_err(|e| format!("{out}: {e}"))?;
        let out = arg(out_engram, "Engram output")?;
        std::fs::write(&out, engram_json(&p.engram, nr as usize, rank as usize)).map_err(|e| format!("{out}: {e}"))?;
        Ok::<_, String>(p.img)
    })();
    match r {
        Ok(img) => Box::into_raw(Box::new((img, t0))) as *mut c_void,
        Err(e) => {
            crate::write_err(err, errlen, &e);
            std::ptr::null_mut()
        }
    }
}
