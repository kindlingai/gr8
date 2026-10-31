//! A rank's image straight from an existing pack and the exlr8 v2 files, planned at boot: every base tensor is
//! read from the base pack, and each v2 layer's tr3.* tensors are built from the published per-rank index slices
//! (tools/convert/v2slices.py) and the raw width files, at the widths a manifest picks. The tensors, their order and
//! their bytes are those tools/convert/v2pack.py writes (and tools/convert/v2.py convert_experts_v2 for one rank).

use crate::codebook::{self, Codebook, TrellisRanges};
use crate::exlr8::StFile;
use crate::json::{self, Json};
use crate::plan::{Gen, Op, Plan, Seg};
use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt;

const ALIGN: u64 = 4096;
pub(crate) const NH: usize = 16;
/// The most 128-blocks an expert of any model has: DeepSeek-V4.1-Flash, 2304 wide.
pub(crate) const NH_MAX: usize = 18;

/// A manifest entry (tools/convert/pack.py): dtype is the manifest label (u8, f16, i64, ...).
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<i64>,
    pub offset: u64,
    pub nbytes: u64,
    pub expert_format: Option<String>,
}

pub struct V2Image {
    pub rank: usize,
    pub tensors: Vec<Tensor>,
    pub bytes: u64,
    pub plan: Plan,
}

impl V2Image {
    /// The image's g6pack-v1 manifest. The sha256 fields are zero: the Rust loader does not check them, and the C++
    /// loader cannot load this image.
    pub fn manifest(&self) -> String {
        let q = |s: &str| format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""));
        let mut out = format!("{{\"align\": {ALIGN}, \"format\": \"g6pack-v1\", \"pack_bytes\": {}, \"rank\": {}, \"tensors\": [",
                              self.bytes, self.rank);
        for (i, t) in self.tensors.iter().enumerate() {
            let shape: Vec<String> = t.shape.iter().map(i64::to_string).collect();
            out += &format!("{}\n{{\"dtype\": {}, \"name\": {}, \"nbytes\": {}, \"offset\": {}, \"sha256\": \"{}\", \"shape\": [{}]",
                            if i > 0 { "," } else { "" }, q(&t.dtype), q(&t.name), t.nbytes, t.offset, "0".repeat(64), shape.join(", "));
            if let Some(x) = &t.expert_format {
                out += &format!(", \"expert_format\": {}", q(x));
            }
            out += "}";
        }
        out + "\n]}\n"
    }
}

/// One per-rank index slice file at one width (tools/convert/v2slices.py).
pub(crate) struct Slice {
    f: File,
    base: u64,
    head: Json,
}

impl Slice {
    pub(crate) fn open(path: &str) -> Result<Slice, String> {
        let f = File::open(path).map_err(|e| format!("cannot open {path}: {e}"))?;
        let mut n = [0u8; 8];
        f.read_exact_at(&mut n, 0).map_err(|e| format!("{path}: {e}"))?;
        let n = u64::from_le_bytes(n);
        let mut h = vec![0u8; n as usize];
        f.read_exact_at(&mut h, 8).map_err(|e| format!("{path}: {e}"))?;
        let head = json::parse(std::str::from_utf8(&h).map_err(|_| format!("{path}: bad header"))?)?;
        Ok(Slice { f, base: 8 + n, head })
    }

    pub(crate) fn has(&self, name: &str) -> bool {
        self.head.get(name).is_some()
    }

    /// (raw bytes, shape) of a tensor.
    pub(crate) fn get(&self, name: &str) -> Result<(Vec<u8>, Vec<usize>), String> {
        let t = self.head.get(name).ok_or_else(|| format!("index slice has no {name}"))?;
        let off = t.get("data_offsets").and_then(Json::arr).ok_or("bad data_offsets")?;
        let (a, b) = (off[0].int().unwrap_or(0) as u64, off[1].int().unwrap_or(0) as u64);
        let shape = t.get("shape").and_then(Json::arr).ok_or("bad shape")?.iter().map(|d| d.int().unwrap_or(0) as usize).collect();
        let mut v = vec![0u8; (b - a) as usize];
        self.f.read_exact_at(&mut v, self.base + a).map_err(|e| format!("{name}: {e}"))?;
        Ok((v, shape))
    }
}

/// The slice data of one layer at one width: rows by (expert, half), and the per-expert / per-layer vectors.
struct LayerSlice {
    row: HashMap<(usize, usize), usize>,
    offset: Vec<i64>,   // [M][3]
    chan: Vec<u8>,      // [M][3][128] f16
    block: Vec<u8>,     // [E][3][48] f16
    signs: Vec<u8>,     // [2][hidden] i8
    hidden: usize,
}

fn layer_slice(s: &Slice, l: usize) -> Result<Option<LayerSlice>, String> {
    let p = format!("L{l:03}.");
    if !s.has(&format!("{p}units")) {
        return Ok(None);
    }
    let (units, _) = s.get(&format!("{p}units"))?;
    let (off, _) = s.get(&format!("{p}offset"))?;
    let row = units.chunks(4).enumerate()
        .map(|(j, u)| ((i16::from_le_bytes([u[0], u[1]]) as usize, i16::from_le_bytes([u[2], u[3]]) as usize), j))
        .collect();
    Ok(Some(LayerSlice {
        row,
        offset: off.chunks(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect(),
        chan: s.get(&format!("{p}chan"))?.0,
        block: s.get(&format!("{p}block"))?.0,
        signs: s.get(&format!("{p}signs"))?.0,
        hidden: s.get(&format!("{p}signs"))?.1.get(1).copied().ok_or("bad signs shape")?,
    }))
}

/// Widths of one layer from a width manifest: gate[e][h], up[e][h], down[e], and the shared expert's (down 0: none).
/// Rows hold `nh` blocks (16 for GLM, 18 for DeepSeek-V4.1-Flash); entries from nh on are 0.
pub(crate) struct Widths {
    pub(crate) gate: Vec<[u8; NH_MAX]>,
    pub(crate) up: Vec<[u8; NH_MAX]>,
    pub(crate) down: Vec<u8>,
    pub(crate) sgate: [u8; NH_MAX],
    pub(crate) sup: [u8; NH_MAX],
    pub(crate) sdown: u8,
    pub(crate) nh: usize,
}

pub(crate) fn widths(m: &Json, l: usize) -> Result<Option<Widths>, String> {
    widths_nh(m, l, NH)
}

/// widths() for a model whose experts have `nh` 128-blocks.
pub(crate) fn widths_nh(m: &Json, l: usize, nh: usize) -> Result<Option<Widths>, String> {
    let Some(e) = m.get("layers").and_then(|x| x.get(&l.to_string())) else { return Ok(None) };
    let row16 = |v: &Json| -> Result<[u8; NH_MAX], String> {
        let a = v.arr().ok_or("width row is not a list")?;
        if a.len() != nh {
            return Err(format!("width row is not {nh} halves"));
        }
        let mut r = [0u8; NH_MAX];
        for (i, x) in a.iter().enumerate() {
            r[i] = x.int().ok_or("width is not an integer")? as u8;
        }
        Ok(r)
    };
    let full = |w: u8| { let mut r = [0u8; NH_MAX]; r[..nh].fill(w); r };
    let list = |k: &str| e.get(k).and_then(Json::arr).ok_or(format!("layer {l}: no {k} widths"));
    let gate = list("gate")?.iter().map(row16).collect::<Result<Vec<_>, _>>()?;
    let up = list("up")?.iter().map(row16).collect::<Result<Vec<_>, _>>()?;
    let down = list("down")?.iter().map(|x| x.int().map(|v| v as u8).ok_or("bad down width")).collect::<Result<Vec<_>, _>>()?;
    let (sgate, sup, sdown) = match e.get("shared") {
        Some(Json::Obj(_)) => {
            let s = e.get("shared").unwrap();
            let b = |k| -> Result<[u8; NH_MAX], String> {
                match s.get(k) {
                    Some(Json::Arr(_)) => row16(s.get(k).unwrap()),
                    Some(v) => Ok(full(v.int().ok_or("bad shared width")? as u8)),
                    None => Err(format!("layer {l}: shared has no {k}")),
                }
            };
            (b("gate")?, b("up")?, s.get("down").and_then(Json::int).ok_or("bad shared down")? as u8)
        }
        Some(v) => {
            let w = v.int().unwrap_or(0) as u8;
            (full(w), full(w), w)
        }
        None => ([0; NH_MAX], [0; NH_MAX], 0),
    };
    Ok(Some(Widths { gate, up, down, sgate, sup, sdown, nh }))
}

pub(crate) struct Builder {
    pub tensors: Vec<Tensor>,
    pub segs: Vec<Seg>,
    pub mems: Vec<Vec<u8>>,
    pub gens: Vec<Gen>,
    pub files: Vec<String>,
    file_ix: HashMap<String, usize>,
    pub cur: u64,
}

impl Builder {
    pub fn new() -> Builder {
        Builder { tensors: Vec::new(), segs: Vec::new(), mems: Vec::new(), gens: Vec::new(), files: Vec::new(),
                  file_ix: HashMap::new(), cur: 0 }
    }
    pub fn begin(&mut self, name: String, dtype: &str, shape: Vec<i64>, nbytes: u64, xf: Option<&str>) -> u64 {
        let off = self.cur.div_ceil(ALIGN) * ALIGN;
        self.tensors.push(Tensor { name, dtype: dtype.into(), shape, offset: off, nbytes, expert_format: xf.map(str::to_string) });
        self.cur = off + nbytes;
        off
    }
    pub fn file(&mut self, path: String) -> usize {
        let n = self.files.len();
        *self.file_ix.entry(path.clone()).or_insert_with(|| {
            self.files.push(path);
            n
        })
    }
    pub fn seg(&mut self, pack_off: u64, nbytes: u64, op: Op, file: usize, file_off: u64) {
        if let Some(last) = self.segs.last_mut() {
            if op == Op::Copy && last.op == Op::Copy && last.file == file && last.pack_off + last.nbytes == pack_off
                && last.file_off + last.nbytes == file_off {
                last.nbytes += nbytes;
                return;
            }
        }
        self.segs.push(Seg { pack_off, nbytes, op, file_off, file });
    }
    fn mem(&mut self, name: String, dtype: &str, shape: Vec<i64>, bytes: Vec<u8>, xf: &str) {
        let off = self.begin(name, dtype, shape, bytes.len() as u64, Some(xf));
        if !bytes.is_empty() {
            self.segs.push(Seg { pack_off: off, nbytes: bytes.len() as u64, op: Op::Mem, file_off: 0, file: self.mems.len() });
            self.mems.push(bytes);
        }
    }
}

fn kmix_word(gw: &[u8; NH_MAX], uw: &[u8; NH_MAX], dk: u8, hs: &[usize]) -> i64 {
    if hs.iter().all(|&h| gw[h] == dk && uw[h] == dk) {
        return 0;
    }
    hs.iter().enumerate().fold(0i64, |v, (b, &h)| v | (gw[h] as i64) << (8 * b) | (uw[h] as i64) << (8 * b + 4))
}

fn f16_signed(scale: [u8; 2], sign: i8) -> [u8; 2] {
    if sign < 0 { [scale[0], scale[1] ^ 0x80] } else { scale }
}

/// Plans the image of a base pack's rank with the v2 experts of every layer the width manifest lists: reads the base
/// manifest, the width manifest and the index slices of the rank (tp<nr>/k<K>/rank<r>.safetensors under `slices_dir`,
/// nr from the base manifest). The raw width files are read later, by the load.
pub fn plan(base_manifest: &str, base_pack: &str, v2_dir: &str, slices_dir: &str, widths_path: &str)
            -> Result<V2Image, String> {
    let read = |p: &str| std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"));
    let bm = json::parse(&read(base_manifest)?)?;
    let rank = bm.get("rank").and_then(Json::int).ok_or(format!("{base_manifest}: no rank"))? as usize;
    let nr = bm.get("nr").and_then(Json::int).unwrap_or(6) as usize;
    let mut base = bm.get("tensors").and_then(Json::arr).ok_or(format!("{base_manifest}: no tensors"))?
        .iter().map(base_tensor).collect::<Result<Vec<_>, _>>()?;
    base.sort_by_key(|t| t.offset);
    let man = width_manifest(widths_path)?;
    let slices = Slices::open(slices_dir, nr, &[rank])?;
    let wfile = |l: usize, k: u8| width_file(v2_dir, l, k);
    let mut b = Builder::new();
    let base_file = b.file(base_pack.to_string());
    let mut done: Vec<usize> = Vec::new();
    for t in &base {
        // the base layer's routed experts and shared-expert records give way to the v2 tensors, which start where
        // the first of them did
        let layer = t.name.strip_prefix('L').and_then(|r| r.get(..3)).and_then(|d| d.parse::<usize>().ok());
        let rest = t.name.get(5..).unwrap_or("");
        if let Some(l) = layer.filter(|_| rest.starts_with("tr3.") || rest.starts_with("shared.")) {
            if let Some(w) = widths(&man, l)? {
                if !done.contains(&l) {
                    v2_layer(&mut b, l, &w, &slices, &wfile, &|e| Ok(formula_halves(e, l, nr, rank)))?;
                    done.push(l);
                }
                continue;
            }
        }
        let off = b.begin(t.name.clone(), &t.dtype, t.shape.clone(), t.nbytes, t.expert_format.as_deref());
        if t.nbytes > 0 {
            b.seg(off, t.nbytes, Op::Copy, base_file, t.offset);
        }
    }
    let listed = man.get("layers").and_then(Json::obj).map(|m| m.len()).unwrap_or(0);
    if done.is_empty() || done.len() != listed {
        return Err(format!("the base pack has routed experts for {} of the {listed} layers {widths_path} lists", done.len()));
    }
    let bytes = b.cur.div_ceil(ALIGN) * ALIGN;
    let plan = Plan::from_segs(b.files, b.segs, b.mems, b.gens, bytes)?;
    Ok(V2Image { rank, tensors: b.tensors, bytes, plan })
}

/// A width manifest (exlr8-v2-manifest).
pub(crate) fn width_manifest(path: &str) -> Result<Json, String> {
    let man = json::parse(&std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?)?;
    if man.get("format").and_then(Json::str) != Some("exlr8-v2-manifest") {
        return Err(format!("{path}: not an exlr8-v2-manifest"));
    }
    Ok(man)
}

/// The halves of expert e in layer l that rank `rank` of TP `nr` owns under the formula, which the index slices of
/// TP `nr` follow too (tools/convert/v2.py owner).
pub(crate) fn formula_halves(e: usize, l: usize, nr: usize, rank: usize) -> Vec<usize> {
    formula_halves_nh(e, l, nr, rank, NH)
}

/// formula_halves() for experts of `nh` 128-blocks: block h of expert e is rank (nh e + h + l) % nr's.
pub(crate) fn formula_halves_nh(e: usize, l: usize, nr: usize, rank: usize, nh: usize) -> Vec<usize> {
    (0..nh).filter(|&h| (nh * e + h + l) % nr == rank).collect()
}

/// Index slices at TP `nr` of some of its ranks, by (width, rank): unit (e, h) of layer l is in rank
/// (16 e + h + l) % nr's slice.
pub(crate) struct Slices {
    nr: usize,
    by: HashMap<(u8, usize), Slice>,
}

impl Slices {
    /// The slices of `ranks` at TP `nr`, one per width present: SLICES/tp<N>/k<K>/rank<r>.safetensors, SLICES being
    /// the index-slices dir itself or a release dir that holds index-slices/.
    pub(crate) fn open(slices_dir: &str, nr: usize, ranks: &[usize]) -> Result<Slices, String> {
        let mut by: HashMap<(u8, usize), Slice> = HashMap::new();
        for &rank in ranks {
            for root in [slices_dir.to_string(), format!("{slices_dir}/index-slices")] {
                for k in [2u8, 3, 4] {
                    let p = format!("{root}/tp{nr}/k{k}/rank{rank}.safetensors");
                    if !by.contains_key(&(k, rank)) && std::path::Path::new(&p).exists() {
                        by.insert((k, rank), Slice::open(&p)?);
                    }
                }
            }
            if !by.keys().any(|&(_, r)| r == rank) {
                // a layout's halves are listed in other ranks' slices, which a box's cache does not hold
                let why = if ranks.len() > 1 { " (a layout reads every rank's slices at its TP)" } else { "" };
                return Err(format!("no index slices for TP {nr} rank {rank} under {slices_dir}{why}"));
            }
        }
        Ok(Slices { nr, by })
    }
}

/// The width file of layer l at width k: DIR/exlr8-layer-LLL.kK.safetensors, or under DIR/experts/.
pub(crate) fn width_file(dir: &str, l: usize, k: u8) -> String {
    let flat = format!("{dir}/exlr8-layer-{l:03}.k{k}.safetensors");
    if std::path::Path::new(&flat).exists() { flat } else { format!("{dir}/experts/exlr8-layer-{l:03}.k{k}.safetensors") }
}

fn base_tensor(t: &Json) -> Result<Tensor, String> {
    let bad = || format!("bad base manifest entry {:?}", t.get("name").and_then(Json::str));
    Ok(Tensor {
        name: t.get("name").and_then(Json::str).ok_or_else(bad)?.to_string(),
        dtype: t.get("dtype").and_then(Json::str).ok_or_else(bad)?.to_string(),
        shape: t.get("shape").and_then(Json::arr).ok_or_else(bad)?.iter().map(|d| d.int().ok_or_else(bad)).collect::<Result<_, _>>()?,
        offset: t.get("offset").and_then(Json::int).ok_or_else(bad)? as u64,
        nbytes: t.get("nbytes").and_then(Json::int).ok_or_else(bad)? as u64,
        expert_format: t.get("expert_format").and_then(Json::str).map(str::to_string),
    })
}

/// The codebook of layer l's width files at the widths `ks` (None when one of them is not there), and each file's
/// trellis tensor ranges by width. Every width file of a layer must use the same codebook: the kernels take it per
/// layer (bind::LayerTensors::cb).
fn layer_codebook(l: usize, ks: &[u8], wfile: &dyn Fn(usize, u8) -> String)
                  -> Result<(Option<Codebook>, HashMap<u8, TrellisRanges>), String> {
    let mut cb: Option<(Codebook, u8)> = None;
    let mut ranges = HashMap::new();
    let mut missing = false;
    for &k in ks {
        let path = wfile(l, k);
        if !std::path::Path::new(&path).exists() {
            missing = true;
            continue;
        }
        let f = StFile::open(&path).map_err(|e| format!("layer {l}: the K{k} width file's header: {e}"))?;
        let (c, r) = codebook::width_file_codebook(&f).map_err(|e| format!("layer {l}: {e}"))?;
        if let Some((c0, k0)) = cb.filter(|&(c0, _)| c0 != c) {
            return Err(format!("layer {l}: the K{k0} width file is in codebook {} and the K{k} one in {}; gr8 takes one \
                                codebook per layer", c0.name(), c.name()));
        }
        cb = Some((c, k));
        ranges.insert(k, r);
    }
    Ok((if missing { None } else { Some(cb.map_or(Codebook::Mcg, |c| c.0)) }, ranges))
}

/// Layer l's v2 expert tensors for the halves `own` gives for each expert (256: the shared expert), in ascending order.
/// Their expert_format names the layer's codebook (src/codebook.rs), read from its width files' headers.
pub(crate) fn v2_layer(b: &mut Builder, l: usize, w: &Widths, slices: &Slices, wfile: &dyn Fn(usize, u8) -> String,
            own: &dyn Fn(usize) -> Result<Vec<usize>, String>) -> Result<(), String> {
    let p = format!("L{l:03}");
    let mut ls: HashMap<(u8, usize), LayerSlice> = HashMap::new();
    for (&k, s) in &slices.by {
        if let Some(x) = layer_slice(s, l)? {
            ls.insert(k, x);
        }
    }
    let nroute = w.down.len();
    let shared = nroute;
    let hidden = ls.values().next().ok_or("no layer slices")?.hidden;
    let geometry = match (hidden, w.nh) {
        (6144, NH) => true,
        (4096, NH) => nroute == 288,
        (5120, 18) => nroute == 384,
        _ => false,
    };
    if !geometry || w.gate.len() != nroute || w.up.len() != nroute {
        return Err(format!("layer {l}: invalid geometry or widths"));
    }
    let blocks = hidden / 128;
    for s in ls.values() {
        let rows = s.row.len();
        if s.hidden != hidden || s.signs.len() != 2 * hidden
            || s.block.len() < nroute * 3 * blocks * 2
            || s.offset.len() != rows * 3 || s.chan.len() != rows * 3 * 256 {
            return Err(format!("layer {l}: inconsistent slice shapes"));
        }
    }
    let nh = w.nh;
    let rows = w.gate.iter().chain(&w.up).flat_map(|r| &r[..nh]);
    if w.down.iter().chain(rows).any(|k| !(2..=4).contains(k))
        || (w.sdown > 0 && (!(2..=4).contains(&w.sdown) || w.sgate[..nh].iter().chain(&w.sup[..nh]).any(|k| !(2..=4).contains(k)))) {
        return Err(format!("layer {l}: invalid width"));
    }
    let mut experts: Vec<usize> = (0..nroute).collect();
    if w.sdown > 0 {
        experts.push(shared);
    }
    let of = |e: usize| -> ([u8; NH_MAX], [u8; NH_MAX], u8) {
        if e == shared { (w.sgate, w.sup, w.sdown) } else { (w.gate[e], w.up[e], w.down[e]) }
    };
    let own: Vec<Vec<usize>> = experts.iter().map(|&e| own(e)).collect::<Result<_, _>>()?;
    let nblk: usize = own.iter().map(Vec::len).sum();
    // the widths this rank reads, their files' codebook, and the trellis tensor each half must lie in
    let mut ks: Vec<u8> = experts.iter().zip(&own)
        .flat_map(|(&e, hs)| { let (g, u, d) = of(e); hs.iter().flat_map(move |&h| [g[h], u[h], d]).collect::<Vec<_>>() })
        .collect();
    ks.sort_unstable();
    ks.dedup();
    let (cb, tranges) = layer_codebook(l, &ks, wfile)?;
    let xf = cb.map_or(codebook::UNCHECKED, Codebook::expert_format);
    let inside = |k: u8, e: usize, proj: usize, off: u64| -> Result<(), String> {
        match tranges.get(&k) {
            None => return Ok(()),         // a missing file: the layer is UNCHECKED, and its bytes cannot load
            Some(r) if r.holds(e == shared, proj, off, hidden as u64 * 16 * k as u64) => return Ok(()),
            Some(_) => {}
        }
        Err(format!("layer {l}: expert {e} {} half at offset {off} of the K{k} width file is not inside its trellis \
                     tensor (the index slices and the width file disagree)", ["gate", "up", "down"][proj]))
    };
    let bs = w.sdown > 0;
    // the slice that holds unit (e, h) at width k
    let get = |k: u8, e: usize, h: usize| ls.get(&(k, (w.nh * e + h + l) % slices.nr)).ok_or(format!("layer {l}: no index slice at K{k}"));
    let row_of = |s: &LayerSlice, e: usize, h: usize| s.row.get(&(e, h)).copied().ok_or(format!("layer {l}: expert {e} half {h} not in the slice"));
    // block row of expert e: routed e, the shared expert after the routed ones
    let brow = |e: usize| if e == shared { nroute } else { e };
    let hb = |k: u8| (hidden as u64 * 16) * k as u64;
    let ne = experts.len();
    let mut table = vec![[0i64; 6]; ne];
    let mut kmix = vec![0i64; ne];
    let nb = if bs { blocks } else { hidden };
    let mut v1 = vec![0u8; nblk * 2 * nb * 2];
    let mut v2 = vec![0u8; nblk * 3 * 128 * 2];
    let mut v3 = vec![0u8; nblk * nb * 2];
    let put_hidden = |s: &LayerSlice, e: usize, proj: usize, dst: &mut [u8]| {
        let blk = &s.block[(brow(e) * 3 + proj) * blocks * 2..(brow(e) * 3 + proj + 1) * blocks * 2];
        if bs {
            dst.copy_from_slice(blk);
        } else {
            let sign = &s.signs[if proj == 2 { hidden } else { 0 }..][..hidden];
            for c in 0..hidden {
                let v = f16_signed([blk[c / 128 * 2], blk[c / 128 * 2 + 1]], sign[c] as i8);
                dst[c * 2..c * 2 + 2].copy_from_slice(&v);
            }
        }
    };
    // pass 1: fc1 (gate, up halves), v1, v2[0:2]
    let fc1_bytes: u64 = experts.iter().zip(&own).map(|(&e, hs)| hs.iter().map(|&h| hb(of(e).0[h]) + hb(of(e).1[h])).sum::<u64>()).sum();
    let fc1 = b.begin(format!("{p}.tr3.fc1"), "u8", vec![fc1_bytes as i64], fc1_bytes, Some(xf));
    let (mut off, mut row) = (0u64, 0usize);
    for (i, &e) in experts.iter().enumerate() {
        let (gw, uw, dk) = of(e);
        let hs = &own[i];
        table[i] = [off as i64, 0, row as i64, hs.len() as i64, dk as i64, *hs.first().unwrap_or(&0) as i64];
        kmix[i] = kmix_word(&gw, &uw, dk, hs);
        for &h in hs {
            for (proj, k) in [(0usize, gw[h]), (1, uw[h])] {
                let s = get(k, e, h)?;
                let j = row_of(s, e, h)?;
                inside(k, e, proj, s.offset[j * 3 + proj] as u64)?;
                let f = b.file(wfile(l, k));
                b.seg(fc1 + off, hb(k), Op::Copy, f, s.offset[j * 3 + proj] as u64);
                off += hb(k);
                put_hidden(s, e, proj, &mut v1[(row * 2 + proj) * nb * 2..(row * 2 + proj + 1) * nb * 2]);
                v2[(row * 3 + proj) * 256..(row * 3 + proj + 1) * 256].copy_from_slice(&s.chan[(j * 3 + proj) * 256..(j * 3 + proj + 1) * 256]);
            }
            row += 1;
        }
    }
    // pass 2: fc2 (down halves, item128 -> gr8 order), v2[2], v3
    let fc2_bytes: u64 = experts.iter().zip(&own).map(|(&e, hs)| hs.len() as u64 * hb(of(e).2)).sum();
    let fc2 = b.begin(format!("{p}.tr3.fc2"), "u8", vec![fc2_bytes as i64], fc2_bytes, Some(xf));
    let (mut off, mut row) = (0u64, 0usize);
    for (i, &e) in experts.iter().enumerate() {
        let dk = of(e).2;
        table[i][1] = off as i64;
        for &h in &own[i] {
            let s = get(dk, e, h)?;
            let j = row_of(s, e, h)?;
            inside(dk, e, 2, s.offset[j * 3 + 2] as u64)?;
            let f = b.file(wfile(l, dk));
            b.seg(fc2 + off, hb(dk), Op::Item128 { hidden }, f, s.offset[j * 3 + 2] as u64);
            off += hb(dk);
            v2[(row * 3 + 2) * 256..(row * 3 + 3) * 256].copy_from_slice(&s.chan[(j * 3 + 2) * 256..(j * 3 + 3) * 256]);
            put_hidden(s, e, 2, &mut v3[row * nb * 2..(row + 1) * nb * 2]);
            row += 1;
        }
    }
    let n = nblk as i64;
    if bs {
        let s = ls.values().next().ok_or(format!("layer {l}: no index slice"))?;
        let sg: Vec<u8> = s.signs.iter().flat_map(|&x| if (x as i8) < 0 { [0x00, 0xBC] } else { [0x00, 0x3C] }).collect();
        b.mem(format!("{p}.tr3.sg"), "f16", vec![2, hidden as i64], sg, xf);
        b.mem(format!("{p}.tr3.v1b"), "f16", vec![n, 2, blocks as i64], v1, xf);
    } else {
        b.mem(format!("{p}.tr3.v1"), "f16", vec![n, 2, hidden as i64], v1, xf);
    }
    b.mem(format!("{p}.tr3.v2"), "f16", vec![n, 3, 128], v2, xf);
    if bs {
        b.mem(format!("{p}.tr3.v3b"), "f16", vec![n, blocks as i64], v3, xf);
    } else {
        b.mem(format!("{p}.tr3.v3"), "f16", vec![n, hidden as i64], v3, xf);
    }
    b.mem(format!("{p}.tr3.table"), "i64", vec![ne as i64, 6], table.iter().flatten().flat_map(|v| v.to_le_bytes()).collect(), xf);
    b.mem(format!("{p}.tr3.kmix"), "i64", vec![ne as i64], kmix.iter().flat_map(|v| v.to_le_bytes()).collect(), xf);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const D: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/v2fetch");
    const NE: u64 = 2;     // the fixture's routed experts (tests/make_v2fetch_fixture.py)

    /// A width file header of the fixture's layout (gate, up, down tensors of NE x 16 halves from file offset 65536):
    /// trellis tensors named with `suffix`, metadata codebook `meta`, the tensors `short` bytes shorter than the halves.
    fn width_file(dir: &std::path::Path, l: usize, k: u8, suffix: &str, meta: Option<&str>, short: u64) {
        let hb = 98304 * k as u64;
        let n = NE * 16 * hb;
        let mut h = String::from("{");
        let mut m = String::from("\"format\":\"exlr8-v2\",\"contents\":\"trellis\"");
        if let Some(c) = meta {
            m += &format!(",\"codebook\":\"{c}\"");
        }
        h += &format!("\"__metadata__\":{{{m}}}");
        for (p, proj) in ["gate", "up", "down"].iter().enumerate() {
            let a = p as u64 * n;
            h += &format!(",\"trellis.{proj}{suffix}\":{{\"dtype\":\"U8\",\"shape\":[{}],\"data_offsets\":[{a},{}]}}", n - short, a + n - short);
        }
        h += "}";
        let mut b = h.into_bytes();
        b.resize(65536 - 8, b' ');
        let mut out = (b.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&b);
        let path = dir.join(format!("exlr8-layer-{l:03}.k{k}.safetensors"));
        std::fs::write(&path, &out).unwrap();
        std::fs::OpenOptions::new().write(true).open(&path).unwrap().set_len(65536 + 3 * n).unwrap();
    }

    /// Plans the fixture's layers at TP4 rank 1 against width files in `dir`: each layer's expert_format.
    fn plan_layers(dir: &std::path::Path) -> Result<Vec<String>, String> {
        let man = width_manifest(&format!("{D}/manifest.json"))?;
        let slices = Slices::open(&format!("{D}/slices"), 4, &[1])?;
        let d = dir.to_str().unwrap().to_string();
        let wfile = move |l: usize, k: u8| width_file_path(&d, l, k);
        let mut b = Builder::new();
        let mut out = Vec::new();
        for l in [3, 4] {
            let w = widths(&man, l)?.unwrap();
            v2_layer(&mut b, l, &w, &slices, &wfile, &|e| Ok(formula_halves(e, l, 4, 1)))?;
            let xf: Vec<&str> = b.tensors.iter().filter(|t| t.name.starts_with(&format!("L{l:03}.")))
                .map(|t| t.expert_format.as_deref().unwrap_or("")).collect();
            assert!(xf.iter().all(|x| *x == xf[0]), "layer {l}: tensors with different expert formats {xf:?}");
            out.push(xf[0].to_string());
        }
        Ok(out)
    }

    fn width_file_path(dir: &str, l: usize, k: u8) -> String {
        super::width_file(dir, l, k)
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("v2plan-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// (suffix, metadata) of each codebook as the exlr8 writer names them
    fn named(cb: &str) -> (String, Option<String>) {
        match cb {
            "mcg" => (String::new(), None),
            "mul1" => (".mul1".into(), Some("mul1 0x83DCD12D, every value rounded to E4M3".into())),
            _ => (".pmcgx".into(), Some("pmcgx V2 L20 0xCBAC1FED/0xBCA14C51, every value rounded to E4M3".into())),
        }
    }

    fn files(dir: &std::path::Path, l: usize, cb: &str) {
        let (s, m) = named(cb);
        for k in [2, 3, 4] {
            width_file(dir, l, k, &s, m.as_deref(), 0);
        }
    }

    #[test]
    fn each_layer_takes_its_files_codebook() {
        let d = tmp("cb");
        // mcg as published (no metadata), and the mcg metadata string the index files carry
        files(&d, 3, "mcg");
        width_file(&d, 4, 2, "", Some("mcg 0xCBAC1FED, every value rounded to E4M3"), 0);
        width_file(&d, 4, 3, "", None, 0);
        width_file(&d, 4, 4, "", None, 0);
        assert_eq!(plan_layers(&d).unwrap(), ["exlr8-e4m3", "exlr8-e4m3"]);
        // a model whose layers differ: each layer's tensors carry its own codebook
        files(&d, 3, "mul1");
        files(&d, 4, "pmcgx");
        assert_eq!(plan_layers(&d).unwrap(), ["exlr8-e4m3-mul1", "exlr8-e4m3-pmcgx"]);
        files(&d, 3, "pmcgx");
        files(&d, 4, "mcg");
        assert_eq!(plan_layers(&d).unwrap(), ["exlr8-e4m3-pmcgx", "exlr8-e4m3"]);
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn a_wrong_or_unknown_codebook_stops_the_plan() {
        let d = tmp("bad");
        files(&d, 4, "mcg");
        let refused = |what: &str, want: &[&str]| {
            let e = plan_layers(&d).err().unwrap_or_else(|| panic!("{what}: planned"));
            for w in want {
                assert!(e.contains(w), "{what}: {e}");
            }
        };
        // one layer's widths in two codebooks
        files(&d, 3, "mcg");
        width_file(&d, 3, 3, ".mul1", Some("mul1 0x83DCD12D, E4M3"), 0);
        refused("mixed widths", &["layer 3", "mcg", "mul1"]);
        // a codebook gr8 does not know, by name or by metadata
        files(&d, 3, "mcg");
        width_file(&d, 3, 2, ".gauss", Some("gauss 0x12345678, E4M3"), 0);
        refused("unknown suffix", &["layer 3", "gauss"]);
        files(&d, 3, "mcg");
        width_file(&d, 3, 4, "", Some("gauss 0x12345678, E4M3"), 0);
        refused("unknown metadata", &["layer 3", "gauss"]);
        // name and metadata disagree, or a suffixed name has no metadata
        files(&d, 3, "mcg");
        width_file(&d, 3, 4, "", Some("pmcgx V2 L20 0xCBAC1FED/0xBCA14C51, E4M3"), 0);
        refused("pmcgx metadata on mcg names", &["layer 3", "pmcgx"]);
        files(&d, 3, "mcg");
        width_file(&d, 3, 4, ".mul1", Some("mcg 0xCBAC1FED, E4M3"), 0);
        refused("mcg metadata on mul1 names", &["layer 3", "mul1"]);
        files(&d, 3, "mcg");
        width_file(&d, 3, 4, ".mul1", None, 0);
        refused("mul1 names without metadata", &["layer 3", "mul1"]);
        // a pair trellis of another shape, and a multiplier that contradicts the name
        files(&d, 3, "mcg");
        width_file(&d, 3, 4, ".pmcgx", Some("pmcgx V2 L22 0xCBAC1FED/0xBCA14C51, E4M3"), 0);
        refused("pmcgx L22", &["layer 3", "L20"]);
        files(&d, 3, "mcg");
        width_file(&d, 3, 4, ".mul1", Some("mul1 0xCBAC1FED, E4M3"), 0);
        refused("mul1 with mcg's multiplier", &["layer 3", "0x83DCD12D"]);
        // halves the slices place outside the trellis tensors
        files(&d, 3, "mcg");
        width_file(&d, 3, 4, "", None, 16 * 98304 * 4);       // expert 1 is past the end
        refused("short tensors", &["layer 3", "not inside"]);
        // no header at all (a sparse cache file without its header range)
        files(&d, 3, "mcg");
        std::fs::write(d.join("exlr8-layer-003.k2.safetensors"), [0u8; 16]).unwrap();
        refused("no header", &["layer 3", "K2"]);
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn a_layer_without_its_width_files_is_unchecked() {
        let d = tmp("missing");
        files(&d, 3, "pmcgx");
        // layer 4's K3 file is not there (a partial copy of the release): its codebook is unknown, never mcg
        files(&d, 4, "mul1");
        std::fs::remove_file(d.join("exlr8-layer-004.k3.safetensors")).unwrap();
        assert_eq!(plan_layers(&d).unwrap(), ["exlr8-e4m3-pmcgx", codebook::UNCHECKED]);
        std::fs::remove_dir_all(&d).unwrap();
    }
}
