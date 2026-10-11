//! The DFlash2 drafter (config.json + one BF16 model.safetensors) -> one gr8 pack per rank: the same bytes and the same
//! manifest tools/convert/dflash2.py writes, without numpy. tools/convert/dflash2.py documents the tensors, the split
//! across ranks and the record layouts. tests/dflash2.rs checks both agree.

use crate::json::{self, Json};
use crate::sha256::Sha256;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;
use std::os::unix::fs::FileExt;

pub const VERSION: &str = "g6dflash2-v1";
pub const CLASSES: [&str; 5] = ["fc", "attn", "mlp", "conv", "sel"];
const ALIGN: u64 = 4096;
const META: [&str; 17] = ["mask_id", "nr", "block", "layers", "window", "tl0", "tl1", "tl2", "tl3", "tl4", "tl5", "theta",
                          "eps_bits", "sel_rank", "sel_topk", "conv_group", "conv_kernel"];

/// The model shape the converter is written for. The tests use a small one.
#[derive(Clone, Copy)]
pub struct Dims {
    pub hid: usize,
    pub hd: usize,
    pub nh: usize,
    pub nkv: usize,
    pub inter: usize,
    pub vocab: i64,
}

pub const GLM53: Dims = Dims { hid: 6144, hd: 128, nh: 64, nkv: 8, inter: 12288, vocab: 154880 };

/// A safetensors file read a tensor at a time (BF16 kept as its bits).
pub struct St {
    path: String,
    f: File,
    base: u64,
    h: BTreeMap<String, Json>,
}

impl St {
    pub fn open(path: &str) -> Result<St, String> {
        let f = File::open(path).map_err(|e| format!("{path}: {e}"))?;
        let mut n = [0u8; 8];
        f.read_exact_at(&mut n, 0).map_err(|e| format!("{path}: {e}"))?;
        let n = u64::from_le_bytes(n);
        let mut h = vec![0u8; n as usize];
        f.read_exact_at(&mut h, 8).map_err(|e| format!("{path}: {e}"))?;
        let h = json::parse(std::str::from_utf8(&h).map_err(|_| format!("{path}: header is not UTF-8"))?)?;
        let h = h.obj().ok_or(format!("{path}: header is not an object"))?.clone();
        Ok(St { path: path.into(), f, base: 8 + n, h })
    }

    pub fn has(&self, name: &str) -> bool {
        self.h.contains_key(name)
    }

    /// The BF16 tensor `name` (rows a..b of dim 0 when given) and its shape.
    pub fn get(&self, name: &str, rows: Option<(usize, usize)>) -> Result<(Vec<u16>, Vec<usize>), String> {
        let t = self.h.get(name).ok_or(format!("{}: no tensor {name}", self.path))?;
        if t.get("dtype").and_then(|d| d.str()) != Some("BF16") {
            return Err(format!("{}: {name} is not BF16", self.path));
        }
        let mut shape: Vec<usize> = t.get("shape").and_then(|s| s.arr()).ok_or(format!("{name}: no shape"))?
            .iter().map(|d| d.int().map(|d| d as usize).ok_or(format!("{name}: bad shape"))).collect::<Result<_, _>>()?;
        let off = t.get("data_offsets").and_then(|o| o.arr()).ok_or(format!("{name}: no data_offsets"))?;
        let (mut lo, mut hi) = (off[0].int().unwrap_or(0) as u64, off[1].int().unwrap_or(0) as u64);
        if let Some((a, b)) = rows {
            if shape.is_empty() || b > shape[0] || a > b {
                return Err(format!("{name}: rows {a}..{b} outside {shape:?}"));
            }
            let row = (hi - lo) / shape[0] as u64;
            (lo, hi) = (lo + a as u64 * row, lo + b as u64 * row);
            shape[0] = b - a;
        }
        let mut raw = vec![0u8; (hi - lo) as usize];
        self.f.read_exact_at(&mut raw, self.base + lo).map_err(|e| format!("{}: {name}: {e}", self.path))?;
        let v = raw.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
        Ok((v, shape))
    }
}

/// Rows [O][K] -> columns a..b of every row.
fn cols(w: &[u16], k: usize, a: usize, b: usize) -> Vec<u16> {
    w.chunks_exact(k).flat_map(|r| r[a..b].iter().copied()).collect()
}

fn pow2(n: i32) -> f64 {
    f64::from_bits(((n + 1023) as u64) << 52)
}

/// numpy.frexp's exponent of a normal or zero float64: x = m * 2**e with m in [0.5, 1). Also whether m == 0.5.
fn frexp(x: f64) -> (i32, bool) {
    if x == 0.0 {
        return (0, false);
    }
    let b = x.to_bits();
    (((b >> 52) & 0x7ff) as i32 - 1022, b & ((1 << 52) - 1) == 0)
}

/// tools/convert/mxfp8.e4m3fn_encode for one float32 (|x| <= 448): round to nearest even, -0.0 keeps its sign.
fn e4m3(x: f32) -> Result<u8, String> {
    let a = x.abs() as f64;
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let mut e = (frexp(a).0 - 1).max(-6);
    let mut q = (a * pow2(3 - e)).round_ties_even();
    if q >= 16.0 {
        e += 1;
        q /= 2.0;
    }
    let code = if q >= 8.0 { ((e + 7) << 3) as i64 | (q as i64 - 8) } else { q as i64 };
    if code > 126 {
        return Err("E4M3FN overflow (|x| > 448 after rounding)".into());
    }
    Ok(code as u8 | sign)
}

/// tools/convert/mxfp8.quantize for one group of 32 BF16 values: the E8M0 scale and the 32 codes.
fn mx_group(w: &[u16], codes: &mut [u8]) -> Result<u8, String> {
    let x: Vec<f32> = w.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs())).max(f32::MIN_POSITIVE);
    let (ex, half) = frexp((amax / 448.0f32) as f64);
    let s = ((if half { ex - 1 } else { ex }) + 127).clamp(0, 254);
    let k = pow2(127 - s);
    for (c, v) in codes.iter_mut().zip(&x) {
        *c = e4m3((*v as f64 * k) as f32)?;
    }
    Ok(s as u8)
}

/// The lane order of a 16 x 16 tile in a decode GEMV record (engine/ref/g6ref.py _gv_lane): lane l = 4r + c holds
/// (row, k) = (r, 2c), (r, 2c+1), (r+8, 2c), (r+8, 2c+1), (r, 2c+8), (r, 2c+9), (r+8, 2c+8), (r+8, 2c+9).
fn lane_order() -> [(usize, usize); 256] {
    let mut o = [(0, 0); 256];
    for l in 0..32 {
        let (r, c) = (l / 4, l % 4);
        let p = [(r, 2 * c), (r, 2 * c + 1), (r + 8, 2 * c), (r + 8, 2 * c + 1), (r, 2 * c + 8), (r, 2 * c + 9),
                 (r + 8, 2 * c + 8), (r + 8, 2 * c + 9)];
        o[8 * l..8 * l + 8].copy_from_slice(&p);
    }
    o
}

/// g6ref.mxfp8_record(mxfp8.quantize(w)): [O, K] BF16 -> [O/16][K/32][528]. Tiles of 16 rows run on all cores.
fn mxfp8_record(w: &[u16], o: usize, k: usize) -> Result<Vec<u8>, String> {
    let (t, g) = (o / 16, k / 32);
    let mut out = vec![0u8; t * g * 528];
    let lanes = lane_order();
    let work = |ti: usize, dst: &mut [u8]| -> Result<(), String> {
        let mut codes = vec![0u8; 16 * k];
        let mut scales = vec![0u8; 16 * g];
        for r in 0..16 {
            let row = &w[(ti * 16 + r) * k..(ti * 16 + r + 1) * k];
            for gi in 0..g {
                scales[r * g + gi] = mx_group(&row[gi * 32..gi * 32 + 32], &mut codes[r * k + gi * 32..r * k + gi * 32 + 32])?;
            }
        }
        for gi in 0..g {
            let rec = &mut dst[gi * 528..(gi + 1) * 528];
            for half in 0..2 {
                let s = 2 * gi + half;
                for (i, &(r, c)) in lanes.iter().enumerate() {
                    rec[256 * half + i] = codes[r * k + s * 16 + c];
                }
            }
            for i in 0..8 {
                rec[512 + 2 * i] = scales[i * g + gi];
                rec[512 + 2 * i + 1] = scales[(i + 8) * g + gi];
            }
        }
        Ok(())
    };
    par_tiles(&mut out, t, g * 528, work)?;
    Ok(out)
}

/// g6ref.bf16_record(w): [O, K] BF16 -> [O/16][K/16][512], each value little-endian.
fn bf16_record(w: &[u16], o: usize, k: usize) -> Result<Vec<u8>, String> {
    let (t, s) = (o / 16, k / 16);
    let mut out = vec![0u8; t * s * 512];
    let lanes = lane_order();
    let work = |ti: usize, dst: &mut [u8]| -> Result<(), String> {
        for si in 0..s {
            for (i, &(r, c)) in lanes.iter().enumerate() {
                let v = w[(ti * 16 + r) * k + si * 16 + c];
                dst[si * 512 + 2 * i..si * 512 + 2 * i + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
        Ok(())
    };
    par_tiles(&mut out, t, s * 512, work)?;
    Ok(out)
}

/// Runs work(tile, its bytes of out) for tiles 0..n, split across the cores.
fn par_tiles<F>(out: &mut [u8], n: usize, per: usize, work: F) -> Result<(), String>
where F: Fn(usize, &mut [u8]) -> Result<(), String> + Sync {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(n.max(1));
    let per_thread = n.div_ceil(threads);
    std::thread::scope(|sc| {
        let hs: Vec<_> = out.chunks_mut(per_thread * per).enumerate().map(|(i, chunk)| {
            let work = &work;
            sc.spawn(move || {
                for (j, tile) in chunk.chunks_mut(per).enumerate() {
                    work(i * per_thread + j, tile)?;
                }
                Ok::<_, String>(())
            })
        }).collect();
        hs.into_iter().try_for_each(|h| h.join().unwrap())
    })
}

/// tools/convert/pack.PackWriter: tensors on 4096-byte boundaries, a sha256 each, the manifest written last.
struct Pack {
    path: String,
    f: std::io::BufWriter<File>,
    off: u64,
    rank: usize,
    entries: Vec<Json>,
}

impl Pack {
    fn new(path: String, rank: usize) -> Result<Pack, String> {
        let f = File::create(&path).map_err(|e| format!("{path}: {e}"))?;
        Ok(Pack { path, f: std::io::BufWriter::with_capacity(8 << 20, f), off: 0, rank, entries: Vec::new() })
    }

    fn pad(&mut self) -> Result<(), String> {
        let n = (ALIGN - self.off % ALIGN) % ALIGN;
        self.f.write_all(&vec![0u8; n as usize]).map_err(|e| format!("{}: {e}", self.path))?;
        self.off += n;
        Ok(())
    }

    fn put(&mut self, name: &str, data: &[u8], dtype: &str, shape: &[usize], info: Vec<(&str, Json)>) -> Result<(), String> {
        self.pad()?;
        let mut h = Sha256::new();
        h.update(data);
        let mut e: BTreeMap<String, Json> = info.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        e.insert("name".into(), Json::Str(name.into()));
        e.insert("dtype".into(), Json::Str(dtype.into()));
        e.insert("shape".into(), Json::Arr(shape.iter().map(|&d| Json::Int(d as i64)).collect()));
        e.insert("offset".into(), Json::Int(self.off as i64));
        e.insert("nbytes".into(), Json::Int(data.len() as i64));
        e.insert("sha256".into(), Json::Str(h.hex()));
        self.entries.push(Json::Obj(e));
        self.f.write_all(data).map_err(|e| format!("{}: {e}", self.path))?;
        self.off += data.len() as u64;
        Ok(())
    }

    fn put_bf16(&mut self, name: &str, (w, shape): (Vec<u16>, Vec<usize>)) -> Result<(), String> {
        let b: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.put(name, &b, "bf16", &shape, vec![])
    }

    /// Writes the padding, syncs the pack, and writes <stem>.manifest.json. Returns the manifest.
    fn close(mut self, meta: BTreeMap<String, Json>) -> Result<Json, String> {
        self.pad()?;
        let f = self.f.into_inner().map_err(|e| format!("{}: {e}", self.path))?;
        f.set_len(self.off).and_then(|_| f.sync_all()).map_err(|e| format!("{}: {e}", self.path))?;
        let mut m = meta;
        m.insert("format".into(), Json::Str("g6pack-v1".into()));
        m.insert("rank".into(), Json::Int(self.rank as i64));
        m.insert("align".into(), Json::Int(ALIGN as i64));
        m.insert("pack_bytes".into(), Json::Int(self.off as i64));
        m.insert("tensors".into(), Json::Arr(self.entries));
        let man = Json::Obj(m);
        let mp = format!("{}.manifest.json", self.path.strip_suffix(".pack").unwrap_or(&self.path));
        std::fs::write(format!("{mp}.tmp"), json::dump_py(&man) + "\n")
            .and_then(|_| std::fs::rename(format!("{mp}.tmp"), &mp)).map_err(|e| format!("{mp}: {e}"))?;
        Ok(man)
    }
}

fn key<'a>(j: &'a Json, k: &str) -> Result<&'a Json, String> {
    j.get(k).ok_or(format!("config has no {k}"))
}

fn num(j: &Json, k: &str) -> Result<f64, String> {
    key(j, k)?.num().ok_or(format!("config {k} is not a number"))
}

/// Python's dict.get(k) or a default.
fn get_or<'a>(j: &'a Json, k: &str, default: &'a Json) -> &'a Json {
    j.get(k).unwrap_or(default)
}

fn int(j: &Json, k: &str) -> Result<i64, String> {
    key(j, k)?.int().ok_or(format!("config {k} is not an integer"))
}

/// tools/convert/dflash2.check_config.
pub fn check_config(cfg: &Json, dims: &Dims) -> Result<(), String> {
    let d = key(cfg, "dflash_config")?;
    for (k, v) in [("hidden_size", dims.hid), ("head_dim", dims.hd), ("num_attention_heads", dims.nh),
                   ("num_key_value_heads", dims.nkv), ("intermediate_size", dims.inter)] {
        if num(cfg, k)? != v as f64 {
            return Err(format!("config {k} = {}, this converter is written for {v}", json::dump_py(key(cfg, k)?)));
        }
    }
    if num(d, "conv_kernel_size")? != 2.0 || dims.hid as i64 % int(d, "conv_group_size")? != 0 {
        return Err("conv_kernel_size must be 2 and conv_group_size must divide hidden_size".into());
    }
    if get_or(cfg, "is_causal", &Json::Bool(false)).truthy() {
        return Err("a causal drafter is not supported (the block is attended to as a whole)".into());
    }
    if get_or(cfg, "attention_bias", &Json::Bool(false)).truthy() {
        return Err("attention bias is not supported".into());
    }
    let one = Json::Num(1.0);
    for k in ["input_embedding_scale", "output_multiplier"] {
        if get_or(d, k, &one).num() != Some(1.0) {
            return Err(format!("dflash_config {k} is not supported"));
        }
    }
    if !matches!(get_or(d, "attention_value_scale", &Json::Null), Json::Null) {
        return Err("dflash_config attention_value_scale is not supported".into());
    }
    for k in ["final_logit_softcapping", "attention_sink_bias", "sample_from_anchor"] {
        if get_or(d, k, &Json::Null).truthy() {
            return Err(format!("dflash_config {k} is not supported"));
        }
    }
    if num(cfg, "vocab_size")? != dims.vocab as f64 {
        return Err(format!("vocab must be GLM 5.3 ({})", dims.vocab));
    }
    if key(d, "target_layer_ids")?.arr().map(|a| a.len()) != Some(6) {
        return Err("the engine taps six target layers".into());
    }
    Ok(())
}

/// cfg.get('sliding_window') if cfg.get('use_sliding_window', True) else None
fn window(cfg: &Json) -> Json {
    if get_or(cfg, "use_sliding_window", &Json::Bool(true)).truthy() {
        get_or(cfg, "sliding_window", &Json::Null).clone()
    } else {
        Json::Null
    }
}

/// D.meta: int32 [17] in META order.
fn meta_values(cfg: &Json, nr: usize) -> Result<Vec<i32>, String> {
    let d = key(cfg, "dflash_config")?;
    let as_i32 = |v: i64, what: &str| i32::try_from(v).map_err(|_| format!("config {what} = {v} does not fit int32"));
    let opt = |k: &str| -> Result<i64, String> {
        let v = get_or(d, k, &Json::Null);
        if v.truthy() { v.int().ok_or(format!("dflash_config {k} is not an integer")) } else { Ok(0) }
    };
    let win = match window(cfg) {
        Json::Null if get_or(cfg, "use_sliding_window", &Json::Bool(true)).truthy() => {
            return Err("config has use_sliding_window but no sliding_window".into())
        }
        Json::Null => 0,
        w => w.int().ok_or("config sliding_window is not an integer")?,
    };
    let mut v = vec![int(d, "mask_token_id")?, nr as i64, int(d, "block_size")?, int(cfg, "num_hidden_layers")?, win];
    for t in key(d, "target_layer_ids")?.arr().unwrap_or(&[]) {
        v.push(t.int().ok_or("target_layer_ids must be integers")?);
    }
    v.push(num(key(cfg, "rope_parameters")?, "rope_theta")?.trunc() as i64);
    v.push((num(cfg, "rms_norm_eps")? as f32).to_bits() as i32 as i64);
    v.extend([opt("selector_rank")?, opt("selector_top_k")?, int(d, "conv_group_size")?, int(d, "conv_kernel_size")?]);
    v.into_iter().zip(META).map(|(x, k)| as_i32(x, k)).collect()
}

/// tools/convert/dflash2.convert_rank: rank r of nr into <out>/dflash.r<r>.{pack,manifest.json}. bf16 lists the record
/// classes kept in BF16. Returns the manifest.
pub fn convert_rank(st: &St, cfg: &Json, dims: &Dims, nr: usize, r: usize, out: &str, bf16: &[String])
                    -> Result<Json, String> {
    let d = key(cfg, "dflash_config")?;
    let Dims { hid, hd, .. } = *dims;
    let (nh, nkv, ni, nf) = (dims.nh / nr, dims.nkv / nr, dims.inter / nr, hid / nr);
    let mut classes: Vec<String> = bf16.to_vec();
    classes.sort();
    classes.dedup();
    let mut meta = BTreeMap::new();
    let copy = |m: &mut BTreeMap<String, Json>, name: &str, v: &Json| m.insert(name.to_string(), v.clone());
    meta.insert("format_dflash".into(), Json::Str(VERSION.into()));
    meta.insert("nr".into(), Json::Int(nr as i64));
    copy(&mut meta, "block_size", key(d, "block_size")?);
    copy(&mut meta, "mask_token_id", key(d, "mask_token_id")?);
    copy(&mut meta, "target_layer_ids", key(d, "target_layer_ids")?);
    copy(&mut meta, "num_layers", key(cfg, "num_hidden_layers")?);
    meta.insert("sliding_window".into(), window(cfg));
    copy(&mut meta, "rope_theta", key(key(cfg, "rope_parameters")?, "rope_theta")?);
    copy(&mut meta, "rms_norm_eps", key(cfg, "rms_norm_eps")?);
    copy(&mut meta, "conv_group_size", key(d, "conv_group_size")?);
    copy(&mut meta, "conv_kernel_size", key(d, "conv_kernel_size")?);
    copy(&mut meta, "selector_rank", get_or(d, "selector_rank", &Json::Null));
    copy(&mut meta, "selector_top_k", get_or(d, "selector_top_k", &Json::Null));
    meta.insert("bf16_classes".into(), Json::Arr(classes.iter().map(|c| Json::Str(c.clone())).collect()));

    let mut w = Pack::new(format!("{out}/dflash.r{r}.pack"), r)?;
    let put_rec = |w: &mut Pack, name: &str, (a, shape): (Vec<u16>, Vec<usize>), cls: &str| -> Result<(), String> {
        if shape.len() != 2 || shape[0] % 16 != 0 || shape[1] % 32 != 0 {
            return Err(format!("{name}: shape {shape:?} is not [16 n, 32 m]"));
        }
        let (o, k) = (shape[0], shape[1]);
        let (rec, layout, rshape) = if classes.iter().any(|c| c == cls) {
            (bf16_record(&a, o, k)?, "bf16_gv_record[O/16][K/16][512]", vec![o / 16, k / 16, 512])
        } else {
            (mxfp8_record(&a, o, k)?, "mxfp8_gv_record[O/16][K/32][528]", vec![o / 16, k / 32, 528])
        };
        w.put(name, &rec, "u8", &rshape, vec![("layout", Json::Str(layout.into())), ("rows", Json::Int(o as i64)),
                                              ("cols", Json::Int(k as i64))])
    };
    let cat = |a: (Vec<u16>, Vec<usize>), b: (Vec<u16>, Vec<usize>)| {
        let mut v = a.0;
        v.extend(b.0);
        (v, vec![a.1[0] + b.1[0], a.1[1]])
    };
    let colslice = |(m, shape): (Vec<u16>, Vec<usize>), a: usize, b: usize| (cols(&m, shape[1], a, b), vec![shape[0], b - a]);

    let mv = meta_values(cfg, nr)?;
    let mb: Vec<u8> = mv.iter().flat_map(|v| v.to_le_bytes()).collect();
    w.put("D.meta", &mb, "i32", &[mv.len()], vec![("fields", Json::Arr(META.iter().map(|s| Json::Str(s.to_string())).collect()))])?;
    put_rec(&mut w, "D.fc.gv", st.get("fc.weight", Some((r * nf, (r + 1) * nf)))?, "fc")?;
    w.put_bf16("D.hidden_norm", st.get("hidden_norm.weight", None)?)?;
    w.put_bf16("D.norm", st.get("norm.weight", None)?)?;
    for j in 0..int(cfg, "num_hidden_layers")? {
        let p = format!("layers.{j}.");
        let g = |n: &str, rows| st.get(&format!("{p}{n}"), rows);
        w.put_bf16(&format!("D{j}.in_norm"), g("input_layernorm.weight", None)?)?;
        w.put_bf16(&format!("D{j}.post_norm"), g("post_attention_layernorm.weight", None)?)?;
        w.put_bf16(&format!("D{j}.q_norm"), g("self_attn.q_norm.weight", None)?)?;
        w.put_bf16(&format!("D{j}.k_norm"), g("self_attn.k_norm.weight", None)?)?;
        put_rec(&mut w, &format!("D{j}.q.gv"), g("self_attn.q_proj.weight", Some((r * nh * hd, (r + 1) * nh * hd)))?, "attn")?;
        let kvr = Some((r * nkv * hd, (r + 1) * nkv * hd));
        put_rec(&mut w, &format!("D{j}.kv.gv"), cat(g("self_attn.k_proj.weight", kvr)?, g("self_attn.v_proj.weight", kvr)?), "attn")?;
        put_rec(&mut w, &format!("D{j}.o.gv"), colslice(g("self_attn.o_proj.weight", None)?, r * nh * hd, (r + 1) * nh * hd), "attn")?;
        let mr = Some((r * ni, (r + 1) * ni));
        put_rec(&mut w, &format!("D{j}.gu.gv"), cat(g("mlp.gate_proj.weight", mr)?, g("mlp.up_proj.weight", mr)?), "mlp")?;
        put_rec(&mut w, &format!("D{j}.down.gv"), colslice(g("mlp.down_proj.weight", None)?, r * ni, (r + 1) * ni), "mlp")?;
        for (c, s) in [("aconv", "attention_conv"), ("mconv", "mlp_conv")] {
            put_rec(&mut w, &format!("D{j}.{c}.gv"), g(&format!("{s}.kernel_projection.weight"), None)?, "conv")?;
            w.put_bf16(&format!("D{j}.{c}.base"), g(&format!("{s}.base_kernel"), None)?)?;
        }
    }
    if st.has("candidate_selector.hidden_projection.weight") {
        put_rec(&mut w, "D.sel.hproj.gv", st.get("candidate_selector.hidden_projection.weight", None)?, "sel")?;
        w.put_bf16("D.sel.pred", st.get("candidate_selector.predecessor_codebook", None)?)?;
        w.put_bf16("D.sel.succ", st.get("candidate_selector.successor_codebook", None)?)?;
    }
    w.close(meta)
}

/// g6load dflash2: --src DIR (config.json + model.safetensors) --out DIR [--nr 4] [--ranks 0,1] [--bf16 conv,sel].
/// Prints one JSON line per rank, as tools/convert/dflash2.py does.
pub fn main(src: &str, out: &str, nr: usize, ranks: Option<&str>, bf16: &str, dims: &Dims) -> Result<(), String> {
    if nr < 1 || dims.nkv % nr != 0 {
        return Err(format!("--nr must divide the {} KV heads", dims.nkv));
    }
    let ranks: Vec<usize> = match ranks {
        Some(s) => s.split(',').map(|x| x.parse().map_err(|_| format!("--ranks {s}"))).collect::<Result<_, _>>()?,
        None => (0..nr).collect(),
    };
    if let Some(r) = ranks.iter().find(|&&r| r >= nr) {
        return Err(format!("rank {r} is not below --nr {nr}"));
    }
    let bf16: Vec<String> = bf16.split(',').filter(|x| !x.is_empty()).map(String::from).collect();
    if let Some(c) = bf16.iter().find(|c| !CLASSES.contains(&c.as_str())) {
        return Err(format!("--bf16: unknown class {c} (classes: {})", CLASSES.join(",")));
    }
    let cp = format!("{src}/config.json");
    let cfg = json::parse(&std::fs::read_to_string(&cp).map_err(|e| format!("{cp}: {e}"))?)?;
    check_config(&cfg, dims)?;
    let st = St::open(&format!("{src}/model.safetensors"))?;
    std::fs::create_dir_all(out).map_err(|e| format!("{out}: {e}"))?;
    for r in ranks {
        let t0 = std::time::Instant::now();
        let m = convert_rank(&st, &cfg, dims, nr, r, out, &bf16)?;
        println!("{{\"rank\": {r}, \"tensors\": {}, \"bytes\": {}, \"seconds\": {:.1}}}",
                 m.get("tensors").and_then(|t| t.arr()).map_or(0, |t| t.len()),
                 m.get("pack_bytes").and_then(|b| b.int()).unwrap_or(0), t0.elapsed().as_secs_f64());
    }
    Ok(())
}
