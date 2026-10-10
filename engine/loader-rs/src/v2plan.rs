//! A rank's image straight from an existing pack and the exlr8 v2 files, planned at boot (#65): every base tensor is
//! read from the base pack, and each v2 layer's tr3.* tensors are built from the published per-rank index slices
//! (tools/convert/v2slices.py) and the raw width files, at the widths a manifest picks. The tensors, their order and
//! their bytes are those tools/convert/v2pack.py writes (and tools/convert/v2.py convert_experts_v2 for one rank).

use crate::json::{self, Json};
use crate::plan::{Gen, Op, Plan, Seg};
use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt;

const ALIGN: u64 = 4096;
pub(crate) const HALF_PER_BIT: u64 = 98304;
pub(crate) const NH: usize = 16;
const SHARED: usize = 256;
const XF_EXLR8: &str = "exlr8-e4m3";

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
    signs: Vec<u8>,     // [2][6144] i8
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
    }))
}

/// Widths of one layer from a width manifest: gate[e][h], up[e][h], down[e], and the shared expert's (down 0: none).
pub(crate) struct Widths {
    pub(crate) gate: Vec<[u8; NH]>,
    pub(crate) up: Vec<[u8; NH]>,
    pub(crate) down: Vec<u8>,
    pub(crate) sgate: [u8; NH],
    pub(crate) sup: [u8; NH],
    pub(crate) sdown: u8,
}

pub(crate) fn widths(m: &Json, l: usize) -> Result<Option<Widths>, String> {
    let Some(e) = m.get("layers").and_then(|x| x.get(&l.to_string())) else { return Ok(None) };
    let row16 = |v: &Json| -> Result<[u8; NH], String> {
        let a = v.arr().ok_or("width row is not a list")?;
        if a.len() != NH {
            return Err("width row is not 16 halves".into());
        }
        let mut r = [0u8; NH];
        for (i, x) in a.iter().enumerate() {
            r[i] = x.int().ok_or("width is not an integer")? as u8;
        }
        Ok(r)
    };
    let list = |k: &str| e.get(k).and_then(Json::arr).ok_or(format!("layer {l}: no {k} widths"));
    let gate = list("gate")?.iter().map(row16).collect::<Result<Vec<_>, _>>()?;
    let up = list("up")?.iter().map(row16).collect::<Result<Vec<_>, _>>()?;
    let down = list("down")?.iter().map(|x| x.int().map(|v| v as u8).ok_or("bad down width")).collect::<Result<Vec<_>, _>>()?;
    let (sgate, sup, sdown) = match e.get("shared") {
        Some(Json::Obj(_)) => {
            let s = e.get("shared").unwrap();
            let b = |k| -> Result<[u8; NH], String> {
                match s.get(k) {
                    Some(Json::Arr(_)) => row16(s.get(k).unwrap()),
                    Some(v) => Ok([v.int().ok_or("bad shared width")? as u8; NH]),
                    None => Err(format!("layer {l}: shared has no {k}")),
                }
            };
            (b("gate")?, b("up")?, s.get("down").and_then(Json::int).ok_or("bad shared down")? as u8)
        }
        Some(v) => {
            let w = v.int().unwrap_or(0) as u8;
            ([w; NH], [w; NH], w)
        }
        None => ([0; NH], [0; NH], 0),
    };
    Ok(Some(Widths { gate, up, down, sgate, sup, sdown }))
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
    fn mem(&mut self, name: String, dtype: &str, shape: Vec<i64>, bytes: Vec<u8>) {
        let off = self.begin(name, dtype, shape, bytes.len() as u64, Some(XF_EXLR8));
        if !bytes.is_empty() {
            self.segs.push(Seg { pack_off: off, nbytes: bytes.len() as u64, op: Op::Mem, file_off: 0, file: self.mems.len() });
            self.mems.push(bytes);
        }
    }
}

fn kmix_word(gw: &[u8; NH], uw: &[u8; NH], dk: u8, hs: &[usize]) -> i64 {
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
    let slices = open_slices(slices_dir, nr, rank)?;
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
                    v2_layer(&mut b, l, &w, &slices, &wfile, nr, rank)?;
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

/// Rank `rank`'s index slices at TP `nr`, one per width present: SLICES/tp<N>/k<K>/rank<r>.safetensors, SLICES being
/// the index-slices dir itself or a release dir that holds index-slices/.
pub(crate) fn open_slices(slices_dir: &str, nr: usize, rank: usize) -> Result<HashMap<u8, Slice>, String> {
    let mut slices: HashMap<u8, Slice> = HashMap::new();
    for root in [slices_dir.to_string(), format!("{slices_dir}/index-slices")] {
        for k in [2u8, 3, 4] {
            let p = format!("{root}/tp{nr}/k{k}/rank{rank}.safetensors");
            if !slices.contains_key(&k) && std::path::Path::new(&p).exists() {
                slices.insert(k, Slice::open(&p)?);
            }
        }
    }
    if slices.is_empty() {
        return Err(format!("no index slices for TP {nr} rank {rank} under {slices_dir}"));
    }
    Ok(slices)
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

pub(crate) fn v2_layer(b: &mut Builder, l: usize, w: &Widths, slices: &HashMap<u8, Slice>, wfile: &dyn Fn(usize, u8) -> String,
            nr: usize, rank: usize) -> Result<(), String> {
    let p = format!("L{l:03}");
    let mut ls: HashMap<u8, LayerSlice> = HashMap::new();
    for (&k, s) in slices {
        if let Some(x) = layer_slice(s, l)? {
            ls.insert(k, x);
        }
    }
    let nroute = w.down.len();
    let mut experts: Vec<usize> = (0..nroute).collect();
    if w.sdown > 0 {
        experts.push(SHARED);
    }
    let of = |e: usize| -> ([u8; NH], [u8; NH], u8) {
        if e == SHARED { (w.sgate, w.sup, w.sdown) } else { (w.gate[e], w.up[e], w.down[e]) }
    };
    let own: Vec<Vec<usize>> = experts.iter().map(|&e| (0..NH).filter(|&h| (16 * e + h + l) % nr == rank).collect()).collect();
    let nblk: usize = own.iter().map(Vec::len).sum();
    let bs = w.sdown > 0;
    let get = |k: u8| ls.get(&k).ok_or(format!("layer {l}: no index slice at K{k}"));
    let row_of = |s: &LayerSlice, e: usize, h: usize| s.row.get(&(e, h)).copied().ok_or(format!("layer {l}: expert {e} half {h} not in the slice"));
    // block row of expert e: routed e, the shared expert after the routed ones
    let brow = |e: usize| if e == SHARED { nroute } else { e };
    let hb = |k: u8| HALF_PER_BIT * k as u64;
    let ne = experts.len();
    let mut table = vec![[0i64; 6]; ne];
    let mut kmix = vec![0i64; ne];
    let nb = if bs { 48 } else { 6144 };
    let mut v1 = vec![0u8; nblk * 2 * nb * 2];
    let mut v2 = vec![0u8; nblk * 3 * 128 * 2];
    let mut v3 = vec![0u8; nblk * nb * 2];
    let hidden = |s: &LayerSlice, e: usize, proj: usize, dst: &mut [u8]| {
        let blk = &s.block[(brow(e) * 3 + proj) * 96..(brow(e) * 3 + proj + 1) * 96];
        if bs {
            dst.copy_from_slice(blk);
        } else {
            let sign = &s.signs[if proj == 2 { 6144 } else { 0 }..][..6144];
            for c in 0..6144 {
                let v = f16_signed([blk[c / 128 * 2], blk[c / 128 * 2 + 1]], sign[c] as i8);
                dst[c * 2..c * 2 + 2].copy_from_slice(&v);
            }
        }
    };
    // pass 1: fc1 (gate, up halves), v1, v2[0:2]
    let fc1_bytes: u64 = experts.iter().zip(&own).map(|(&e, hs)| hs.iter().map(|&h| hb(of(e).0[h]) + hb(of(e).1[h])).sum::<u64>()).sum();
    let fc1 = b.begin(format!("{p}.tr3.fc1"), "u8", vec![fc1_bytes as i64], fc1_bytes, Some(XF_EXLR8));
    let (mut off, mut row) = (0u64, 0usize);
    for (i, &e) in experts.iter().enumerate() {
        let (gw, uw, dk) = of(e);
        let hs = &own[i];
        table[i] = [off as i64, 0, row as i64, hs.len() as i64, dk as i64, *hs.first().unwrap_or(&0) as i64];
        kmix[i] = kmix_word(&gw, &uw, dk, hs);
        for &h in hs {
            for (proj, k) in [(0usize, gw[h]), (1, uw[h])] {
                let s = get(k)?;
                let j = row_of(s, e, h)?;
                let f = b.file(wfile(l, k));
                b.seg(fc1 + off, hb(k), Op::Copy, f, s.offset[j * 3 + proj] as u64);
                off += hb(k);
                hidden(s, e, proj, &mut v1[(row * 2 + proj) * nb * 2..(row * 2 + proj + 1) * nb * 2]);
                v2[(row * 3 + proj) * 256..(row * 3 + proj + 1) * 256].copy_from_slice(&s.chan[(j * 3 + proj) * 256..(j * 3 + proj + 1) * 256]);
            }
            row += 1;
        }
    }
    // pass 2: fc2 (down halves, item128 -> g6 order), v2[2], v3
    let fc2_bytes: u64 = experts.iter().zip(&own).map(|(&e, hs)| hs.len() as u64 * hb(of(e).2)).sum();
    let fc2 = b.begin(format!("{p}.tr3.fc2"), "u8", vec![fc2_bytes as i64], fc2_bytes, Some(XF_EXLR8));
    let (mut off, mut row) = (0u64, 0usize);
    for (i, &e) in experts.iter().enumerate() {
        let dk = of(e).2;
        table[i][1] = off as i64;
        for &h in &own[i] {
            let s = get(dk)?;
            let j = row_of(s, e, h)?;
            let f = b.file(wfile(l, dk));
            b.seg(fc2 + off, hb(dk), Op::Item128, f, s.offset[j * 3 + 2] as u64);
            off += hb(dk);
            v2[(row * 3 + 2) * 256..(row * 3 + 3) * 256].copy_from_slice(&s.chan[(j * 3 + 2) * 256..(j * 3 + 3) * 256]);
            hidden(s, e, 2, &mut v3[row * nb * 2..(row + 1) * nb * 2]);
            row += 1;
        }
    }
    let n = nblk as i64;
    if bs {
        let s = ls.values().next().ok_or(format!("layer {l}: no index slice"))?;
        let sg: Vec<u8> = s.signs.iter().flat_map(|&x| if (x as i8) < 0 { [0x00, 0xBC] } else { [0x00, 0x3C] }).collect();
        b.mem(format!("{p}.tr3.sg"), "f16", vec![2, 6144], sg);
        b.mem(format!("{p}.tr3.v1b"), "f16", vec![n, 2, 48], v1);
    } else {
        b.mem(format!("{p}.tr3.v1"), "f16", vec![n, 2, 6144], v1);
    }
    b.mem(format!("{p}.tr3.v2"), "f16", vec![n, 3, 128], v2);
    if bs {
        b.mem(format!("{p}.tr3.v3b"), "f16", vec![n, 48], v3);
    } else {
        b.mem(format!("{p}.tr3.v3"), "f16", vec![n, 6144], v3);
    }
    b.mem(format!("{p}.tr3.table"), "i64", vec![ne as i64, 6], table.iter().flatten().flat_map(|v| v.to_le_bytes()).collect());
    b.mem(format!("{p}.tr3.kmix"), "i64", vec![ne as i64], kmix.iter().flat_map(|v| v.to_le_bytes()).collect());
    Ok(())
}
