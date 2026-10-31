//! Tests of src/ds41.rs on a small synthetic model: native tensors with the 1017 layout (F8_E4M3 + E8M0 block scales,
//! nibble-packed MXFP4 experts, an Engram table, DSpark layers), written out as an exlr8 native release (Mini::write).
//! Each oracle below reads the native tensors and is written from the layout rules in docs/DEEPSEEK-WEIGHTS.md and
//! engine/kernels/mxfp4.cuh, not from src/ds41.rs. The cross-check against the C++ packer (mxfp4_pack.h) runs when
//! GR8_TEST_DS41_KERNELS_DIR names engine/kernels.
use g6load::ds41::{self, Opts};
use g6load::records;
use std::collections::BTreeMap;
use std::ffi::CString;
use std::ops::Range;

const HID: usize = 128;
const V: usize = 2048;
const HEADS: usize = 16;
const HD: usize = 32;
const QA: usize = 64;
const ISH: usize = 256;
const E: usize = 16;
const NMAIN: usize = 3;
const NMTP: usize = 2;
const ENG_ROWS: usize = 1002;
const ENG_DIM: usize = 64;
const ENG_SDIM: usize = 2;

struct Mini {
    dir: String,
    t: BTreeMap<String, (&'static str, Vec<usize>, Vec<u8>)>,
    seed: u64,
}

impl Mini {
    fn rnd(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| {
            self.seed ^= self.seed << 13;
            self.seed ^= self.seed >> 7;
            self.seed ^= self.seed << 17;
            (self.seed >> 24) as u8
        }).collect()
    }
    fn add(&mut self, name: &str, dtype: &'static str, shape: &[usize], data: Vec<u8>) {
        let e = match dtype { "BF16" => 2, "F32" => 4, _ => 1 };
        assert_eq!(data.len(), shape.iter().product::<usize>() * e, "{name}");
        assert!(self.t.insert(name.to_string(), (dtype, shape.to_vec(), data)).is_none(), "{name} twice");
    }
    fn bf16(&mut self, name: &str, shape: &[usize]) {
        let d = self.rnd(shape.iter().product::<usize>() * 2);
        self.add(name, "BF16", shape, d);
    }
    fn f32(&mut self, name: &str, shape: &[usize]) {
        let d = self.rnd(shape.iter().product::<usize>() * 4);
        self.add(name, "F32", shape, d);
    }
    fn i8(&mut self, name: &str, shape: &[usize]) {
        let d = self.rnd(shape.iter().product());
        self.add(name, "I8", shape, d);
    }
    fn e8(&mut self, name: &str, shape: &[usize]) {
        let d: Vec<u8> = self.rnd(shape.iter().product()).iter().map(|b| 118 + b % 16).collect();
        self.add(name, "F8_E8M0", shape, d);
    }
    /// F8_E4M3 `name`.weight [n,k] + `name`.scale [n/32,k/32]
    fn f8(&mut self, name: &str, n: usize, k: usize) {
        let d: Vec<u8> = self.rnd(n * k).iter().map(|&b| if b & 0x7f == 0x7f { b & 0xfe } else { b }).collect();
        self.add(&format!("{name}.weight"), "F8_E4M3", &[n, k], d);
        self.e8(&format!("{name}.scale"), &[n / 32, k / 32]);
    }
    fn get(&self, name: &str) -> &Vec<u8> {
        &self.t.get(name).unwrap_or_else(|| panic!("{name}")).2
    }

    fn layer(&mut self, pre: &str, l: usize, dsp: usize) {
        let t = |s: &str| format!("{pre}{s}");
        self.bf16(&t("attn_norm.weight"), &[HID]);
        self.bf16(&t("ffn_norm.weight"), &[HID]);
        for hc in ["attn", "ffn"] {
            self.f32(&t(&format!("hc_{hc}_fn")), &[24, 4 * HID]);
            self.f32(&t(&format!("hc_{hc}_base")), &[24]);
            self.f32(&t(&format!("hc_{hc}_scale")), &[3]);
        }
        self.f8(&t("attn.wq_a"), QA, HID);
        self.f8(&t("attn.wq_b"), HEADS * HD, QA);
        self.f8(&t("attn.wkv"), HD, HID);
        self.f8(&t("attn.wo_a"), 8 * 32, HEADS * HD / 8);
        self.f8(&t("attn.wo_b"), HID, 8 * 32);
        self.bf16(&t("attn.q_norm.weight"), &[QA]);
        self.bf16(&t("attn.kv_norm.weight"), &[HD]);
        self.f32(&t("attn.attn_sink"), &[HEADS]);
        if l == 2 {
            self.bf16(&t("attn.compressor.wkv.weight"), &[HD, HID]);
            self.bf16(&t("attn.compressor.wgate.weight"), &[HD, HID]);
            self.bf16(&t("attn.compressor.norm.weight"), &[HD]);
            self.f8(&t("attn.indexer.wq_b"), 64, 64);
            self.bf16(&t("attn.indexer.weights_proj.weight"), &[16, HID]);
            self.bf16(&t("attn.indexer.wk.weight"), &[32, 32]);
            self.bf16(&t("attn.indexer.k_norm.weight"), &[32]);
        }
        self.bf16(&t("ffn.gate.weight"), &[E, HID]);
        self.f32(&t("ffn.gate.bias"), &[E]);
        self.f32(&t("ffn.gate.bias_vl"), &[E]);
        self.f8(&t("ffn.shared_experts.w1"), ISH, HID);
        self.f8(&t("ffn.shared_experts.w3"), ISH, HID);
        self.f8(&t("ffn.shared_experts.w2"), HID, ISH);
        for e in 0..E {
            let x = |w: &str, s: &str| t(&format!("ffn.experts.{e}.{w}.{s}"));
            self.i8(&x("w1", "weight"), &[ISH, HID / 2]);
            self.e8(&x("w1", "scale"), &[ISH, HID / 32]);
            self.i8(&x("w3", "weight"), &[ISH, HID / 2]);
            self.e8(&x("w3", "scale"), &[ISH, HID / 32]);
            self.i8(&x("w2", "weight"), &[HID, ISH / 2]);
            self.e8(&x("w2", "scale"), &[HID, ISH / 32]);
        }
        if l == 1 {
            // the table: E4M3 rows [n, dim] and one E8M0 scale per 32 values of a row, [n, dim/32]
            let d: Vec<u8> = self.rnd(ENG_ROWS * ENG_DIM).iter().map(|&b| if b & 0x7f == 0x7f { b & 0xfe } else { b }).collect();
            self.add(&t("engram.embed.weight"), "F8_E4M3", &[ENG_ROWS, ENG_DIM], d);
            self.e8(&t("engram.embed.scale"), &[ENG_ROWS, ENG_SDIM]);
            self.f8(&t("engram.wkv"), 64, 128);
            self.bf16(&t("engram.q_weight"), &[4, HID]);
            self.bf16(&t("engram.k_weight"), &[4, HID]);
        }
        if dsp == 1 {
            self.bf16(&t("main_norm.weight"), &[HID]);
            self.f8(&t("main_proj"), HID, 3 * HID);
        }
        if dsp == 2 {
            self.bf16(&t("norm.weight"), &[HID]);
            self.bf16(&t("confidence_head.proj.weight"), &[1, 144]);
            self.bf16(&t("markov_head.embed.weight"), &[V, 16]);
            self.bf16(&t("markov_head.head.weight"), &[V, 16]);
        }
    }
}

fn mini(name: &str) -> Mini {
    let dir = format!("{}/ds41-test-{}-{name}", std::env::temp_dir().display(), std::process::id());
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut m = Mini { dir, t: BTreeMap::new(), seed: 0x9e3779b97f4a7c15 };
    m.bf16("embed.weight", &[V, HID]);
    m.bf16("norm.weight", &[HID]);
    m.bf16("head.weight", &[V, HID]);
    for l in 0..NMAIN {
        m.layer(&format!("layers.{l}."), l, 0);
    }
    for j in 0..NMTP {
        m.layer(&format!("mtp.{j}."), 0, j + 1);
    }
    m.bf16("vision.patch.weight", &[16, 16]);
    m.bf16("aligner.proj.weight", &[16, 16]);
    m.bf16("image_start_token", &[16]);
    m
}

/// Writes a safetensors file: `items` are (name, safetensors dtype, shape, bytes), `meta` string pairs. Tensors start on
/// 64-byte boundaries. Returns each tensor's absolute byte offset in the file.
fn st_write(path: &str, items: &[(String, &str, Vec<usize>, Vec<u8>)], meta: &[(&str, String)]) -> BTreeMap<String, u64> {
    let q = |s: &str| format!("\"{}\"", s.replace('"', "\\\""));
    let (mut parts, mut off, mut at) = (Vec::new(), 0usize, BTreeMap::new());
    let m: Vec<String> = meta.iter().map(|(k, v)| format!("{}: {}", q(k), q(v))).collect();
    parts.push(format!("\"__metadata__\": {{{}}}", m.join(", ")));
    let mut offs = Vec::new();
    for (n, d, sh, b) in items {
        let pad = (64 - off % 64) % 64;
        if pad > 0 {
            parts.push(format!("\"__pad.{off}\": {{\"dtype\": \"U8\", \"shape\": [{pad}], \"data_offsets\": [{off}, {}]}}", off + pad));
            off += pad;
        }
        let shape: Vec<String> = sh.iter().map(usize::to_string).collect();
        parts.push(format!("{}: {{\"dtype\": \"{d}\", \"shape\": [{}], \"data_offsets\": [{off}, {}]}}", q(n), shape.join(", "), off + b.len()));
        offs.push((n.clone(), off));
        off += b.len();
    }
    let mut head = format!("{{{}}}", parts.join(", ")).into_bytes();
    while (8 + head.len()) % 64 != 0 {
        head.push(b' ');
    }
    let base = 8 + head.len() as u64;
    let mut out = (head.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&head);
    let mut cur = 0usize;
    for ((_, _, _, b), (n, o)) in items.iter().zip(&offs) {
        out.resize(out.len() + (o - cur), 0);
        out.extend_from_slice(b);
        cur = o + b.len();
        at.insert(n.clone(), base + *o as u64);
    }
    std::fs::write(path, out).unwrap();
    at
}

fn rec(fmt: i64, out: usize, inp: usize, side: i64, unit: usize) -> Vec<u8> {
    [fmt, out as i64, inp as i64, 0, side, unit as i64, 0, 0].iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The release layer of test layer `l`: main layers as they are, the DSpark layers from 40.
fn rl(l: usize) -> usize {
    if l < NMAIN { l } else { 40 + l - NMAIN }
}

/// Engram split files hold this many rows (the release uses 2^24): small, so a rank's rows cross files.
const ENG_PART: usize = 256;

impl Mini {
    /// The release of this checkpoint, laid out as kindlingai/deepseek-v4.1-flash-exlr8-native: dense records per layer
    /// (wo_b on its input side in 128-column units), the vocabulary file, expert stores of 128-channel units with
    /// their index, and the Engram table in split files.
    fn write(&self) {
        let dir = &self.dir;
        std::fs::create_dir_all(format!("{dir}/dense")).unwrap();
        std::fs::create_dir_all(format!("{dir}/experts")).unwrap();
        let meta = |l: Option<usize>| {
            let mut m = vec![("hidden_size", HID.to_string()), ("moe_intermediate_size", ISH.to_string()), ("tensor_prefix", "layers.".to_string())];
            if let Some(l) = l {
                m.push(("layer", l.to_string()));
            }
            m
        };
        let item = |n: &str, d: &'static str, sh: Vec<usize>, b: Vec<u8>| (n.to_string(), d, sh, b);
        // a native tensor as a release item under `local`: matrices become records, the rest stays plain
        let mut layers: BTreeMap<usize, Vec<(String, &str, Vec<usize>, Vec<u8>)>> = BTreeMap::new();
        let mut vocab = Vec::new();
        for (name, (d, sh, b)) in &self.t {
            let (l, local) = if let Some(x) = name.strip_prefix("layers.") {
                let (n, x) = x.split_once('.').unwrap();
                (Some(n.parse::<usize>().unwrap()), x.to_string())
            } else if let Some(x) = name.strip_prefix("mtp.") {
                let (n, x) = x.split_once('.').unwrap();
                (Some(rl(NMAIN + n.parse::<usize>().unwrap())), x.to_string())
            } else {
                (None, name.clone())
            };
            if name.starts_with("vision.") || name.starts_with("aligner.") || name.starts_with("image_") {
                continue;
            }
            if local.contains("experts.") || local.starts_with("engram.embed.") || local.ends_with(".scale") {
                continue;
            }
            let v = match l { Some(l) => layers.entry(l).or_default(), None => &mut vocab };
            let local = match local.as_str() { "embed.weight" => "embed_tokens.weight".to_string(), "head.weight" => "lm_head.weight".to_string(), _ => local };
            let base = local.strip_suffix(".weight");
            if *d == "F8_E4M3" {
                let base = base.unwrap();
                let sc = self.get(&name.replace(".weight", ".scale"));
                let (n, k) = (sh[0], sh[1]);
                if base == "attn.wo_b" {
                    // input side: [k/128][n][128] codes, [k/128][n/32][4] scales
                    let (mut c, mut s) = (Vec::new(), Vec::new());
                    for u in 0..k / 128 {
                        for r in 0..n {
                            c.extend_from_slice(&b[r * k + u * 128..r * k + u * 128 + 128]);
                        }
                        for br in 0..n / 32 {
                            s.extend_from_slice(&sc[br * (k / 32) + u * 4..br * (k / 32) + u * 4 + 4]);
                        }
                    }
                    v.push(item(&format!("{base}.record"), "I64", vec![8], rec(8, n, k, 2, 128)));
                    v.push(item(&format!("{base}.fp8"), "U8", vec![k / 128, n, 128], c));
                    v.push(item(&format!("{base}.e8m0"), "U8", vec![k / 128, n / 32, 4], s));
                } else {
                    v.push(item(&format!("{base}.record"), "I64", vec![8], rec(8, n, k, 1, 128)));
                    v.push(item(&format!("{base}.fp8"), "U8", vec![n, k], b.clone()));
                    v.push(item(&format!("{base}.e8m0"), "U8", vec![n / 32, k / 32], sc.clone()));
                }
            } else if *d == "BF16" && sh.len() == 2 && base.is_some() && !local.starts_with("engram.") {
                let base = base.unwrap();
                v.push(item(&format!("{base}.record"), "I64", vec![8], rec(6, sh[0], sh[1], 0, 0)));
                v.push(item(&format!("{base}.bf16"), "BF16", sh.clone(), b.clone()));
            } else {
                let d = match *d { "F8_E4M3" | "F8_E8M0" | "I8" => "U8", d => d };
                v.push(item(&local, d, sh.clone(), b.clone()));
            }
        }
        st_write(&format!("{dir}/dense/exlr8-vocab.safetensors"), &vocab, &meta(None));
        for (&l, v) in &layers {
            st_write(&format!("{dir}/dense/exlr8-dense-{l:03}.safetensors"), v, &meta(Some(l)));
        }
        // experts: units of 128 intermediate channels; gate / up units are rows, down units columns
        for l in 0..NMAIN + NMTP {
            let pre = layer_prefix(l);
            let units = ISH / 128;
            let unit = |w: &[u8], s: &[u8], rows_side: bool, k: usize, cb: usize, sr: usize, u: usize| -> Vec<u8> {
                // rows_side: rows [128u, 128u + 128) of [ISH][k] (codes cb bytes a row, scales k/32 a scale row of sr rows)
                // else: columns [128u, ..) of [HID][ISH] (codes cb bytes a row, scales ISH/32 a scale row)
                let mut out = Vec::new();
                if rows_side {
                    out.extend_from_slice(&w[u * 128 * cb..(u + 1) * 128 * cb]);
                    out.extend_from_slice(&s[u * 128 / sr * (k / 32)..(u + 1) * 128 / sr * (k / 32)]);
                } else {
                    let ucb = cb * 128 / ISH;
                    for r in 0..HID {
                        out.extend_from_slice(&w[r * cb + u * ucb..r * cb + (u + 1) * ucb]);
                    }
                    for r in 0..HID / sr {
                        out.extend_from_slice(&s[r * (ISH / 32) + u * 4..r * (ISH / 32) + u * 4 + 4]);
                    }
                }
                out
            };
            let (mut mx4, mut fp8) = (Vec::new(), Vec::new());
            for e in 0..E {
                for (j, w) in ["w1", "w3", "w2"].iter().enumerate() {
                    let (wb, sb) = (self.get(&format!("{pre}ffn.experts.{e}.{w}.weight")), self.get(&format!("{pre}ffn.experts.{e}.{w}.scale")));
                    for u in 0..units {
                        mx4.push(((e, j, u), unit(wb, sb, j < 2, HID, if j < 2 { HID / 2 } else { ISH / 2 }, 1, u)));
                    }
                }
            }
            for (j, w) in ["w1", "w3", "w2"].iter().enumerate() {
                let (wb, sb) = (self.get(&format!("{pre}ffn.shared_experts.{w}.weight")), self.get(&format!("{pre}ffn.shared_experts.{w}.scale")));
                for u in 0..units {
                    fp8.push(((0, j, u), unit(wb, sb, j < 2, HID, if j < 2 { HID } else { ISH }, 32, u)));
                }
            }
            let store = |path: String, us: &[((usize, usize, usize), Vec<u8>)], n_exp: usize| -> Vec<u8> {
                // units in reverse order, so offsets are not the natural order
                let items: Vec<(String, &str, Vec<usize>, Vec<u8>)> = us.iter().rev().enumerate()
                    .map(|(i, (_, b))| (format!("u.{i:06}"), "U8", vec![b.len()], b.clone())).collect();
                let at = st_write(&path, &items, &meta(Some(rl(l))));
                let mut index = vec![0i64; n_exp * 3 * units * 2];
                for (i, ((e, j, u), b)) in us.iter().rev().enumerate() {
                    let k = ((e * 3 + j) * units + u) * 2;
                    index[k] = at[&format!("u.{i:06}")] as i64;
                    index[k + 1] = b.len() as i64;
                }
                index.iter().flat_map(|v| v.to_le_bytes()).collect()
            };
            let e = |s: &str| format!("{dir}/experts/exlr8-layer-{:03}.{s}.safetensors", rl(l));
            let mi = store(e("mxfp4"), &mx4, E);
            let fi = store(e("fp8b32"), &fp8, 1);
            st_write(&e("index"), &[item("mxfp4.index", "I64", vec![E, 3, units, 2], mi), item("shared.fp8b32.index", "I64", vec![1, 3, units, 2], fi)], &meta(Some(rl(l))));
        }
        // the Engram table in split files of ENG_PART rows (MXFP8 records)
        let (w, s) = (self.get("layers.1.engram.embed.weight"), self.get("layers.1.engram.embed.scale"));
        for (pp, r0) in (0..ENG_ROWS).step_by(ENG_PART).enumerate() {
            let n = ENG_PART.min(ENG_ROWS - r0);
            st_write(&format!("{dir}/dense/exlr8-split-001.engram.embed.p{pp:03}.safetensors"), &[
                item("engram.embed.record", "I64", vec![8], rec(3, n, ENG_DIM, 0, 0)),
                item("engram.embed.fp8", "U8", vec![n, ENG_DIM], w[r0 * ENG_DIM..(r0 + n) * ENG_DIM].to_vec()),
                item("engram.embed.e8m0", "U8", vec![n, ENG_SDIM], s[r0 * ENG_SDIM..(r0 + n) * ENG_SDIM].to_vec()),
            ], &meta(Some(1)));
        }
    }
}

impl Drop for Mini {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn layer_prefix(l: usize) -> String {
    if l < NMAIN { format!("layers.{l}.") } else { format!("mtp.{}.", l - NMAIN) }
}

// ---------------------------------------------------------------------------------------------------- oracles

/// The O-groups of rank `rank` in layer `l`, from the rule in the doc: widths(8, nr) larger chunks first, chunk j to
/// rank (j + l) % nr.
fn groups_of(l: usize, nr: usize, rank: usize) -> Range<usize> {
    let (q, r) = (8 / nr, 8 % nr);
    let mut start = 0;
    for j in 0..nr {
        let w = q + usize::from(j < r);
        if (j + l) % nr == rank {
            return start..start + w;
        }
        start += w;
    }
    unreachable!()
}

/// rows and columns of `native` (.weight + .scale): the codes and the scales, each block row's scale repeated per row.
fn ref_mx(m: &Mini, native: &str, rows: Range<usize>, cols: Range<usize>) -> (Vec<u8>, Vec<u8>) {
    let (w, sc) = (m.get(&format!("{native}.weight")), m.get(&format!("{native}.scale")));
    let k = m.t[&format!("{native}.weight")].1[1];
    let (mut codes, mut scales) = (Vec::new(), Vec::new());
    for r in rows {
        codes.extend_from_slice(&w[r * k + cols.start..r * k + cols.end]);
        scales.extend_from_slice(&sc[(r / 32) * (k / 32) + cols.start / 32..(r / 32) * (k / 32) + cols.end / 32]);
    }
    (codes, scales)
}

fn e2m1(code: u8) -> f64 {
    let v = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][(code & 7) as usize];
    if code & 8 != 0 { -v } else { v }
}

fn e8m0(b: u8) -> f64 {
    2f64.powi(b as i32 - 127)
}

/// Value of weight (row, k) of a nibble-packed matrix [rows][kfull/2] with E8M0 scales [rows][kfull/32].
fn fp4_value(p: &[u8], s: &[u8], kfull: usize, row: usize, k: usize) -> f64 {
    let b = p[row * kfull / 2 + k / 2];
    e2m1(if k % 2 == 0 { b & 15 } else { b >> 4 }) * e8m0(s[row * kfull / 32 + k / 32])
}

/// What the device computes for one 544-byte mx4g record (engine/kernels/mxfp4.cuh): the 16 x 64 dequantized weights.
/// Lane l = 4r + c holds, for k16 step t, the bytes {row r, row r+8, row r, row r+8} at k 2c.. and k 2c+8..; the scale
/// bytes at 512 + 4r are {row r, row r+8} of the two k32 groups.
fn decode_record(g: &[u8]) -> [[f64; 64]; 16] {
    assert_eq!(g.len(), 544);
    let mut out = [[0f64; 64]; 16];
    for lane in 0..32 {
        let (r, c) = (lane / 4, lane % 4);
        for t in 0..4 {
            let w = &g[lane * 16 + t * 4..][..4];
            let h = t / 2;
            let (s_lo, s_hi) = (e8m0(g[512 + r * 4 + 2 * h]), e8m0(g[512 + r * 4 + 2 * h + 1]));
            let kb = t * 16 + 2 * c;
            for (byte, row, dk, s) in [(w[0], r, 0, s_lo), (w[1], r + 8, 0, s_hi), (w[2], r, 8, s_lo), (w[3], r + 8, 8, s_hi)] {
                out[row][kb + dk] = e2m1(byte & 15) * s;
                out[row][kb + dk + 1] = e2m1(byte >> 4) * s;
            }
        }
    }
    out
}

fn tensor(img: &g6load::v2plan::V2Image, name: &str) -> Vec<u8> {
    ds41::read_tensor(img, name).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn plan_err(dir: &str, nr: usize, rank: usize) -> String {
    match ds41::plan(dir, nr, rank, &Opts::default()) {
        Err(e) => e,
        Ok(_) => panic!("plan nr {nr} rank {rank} should fail"),
    }
}

fn plan(m: &Mini, nr: usize, rank: usize) -> (g6load::v2plan::V2Image, Vec<ds41::EngramSlice>) {
    let p = ds41::plan(&m.dir, nr, rank, &Opts::default()).unwrap_or_else(|e| panic!("plan nr {nr} rank {rank}: {e}"));
    (p.img, p.engram)
}

// ---------------------------------------------------------------------------------------------------- tests

#[test]
fn ownership_rules() {
    for nr in 1..=8 {
        for l in 0..43 {
            let mut seen = vec![0; 8];
            for rank in 0..nr {
                let g = ds41::o_groups(l, nr, rank);
                assert_eq!(g, groups_of(l, nr, rank), "l {l} nr {nr} rank {rank}");
                for x in g {
                    seen[x] += 1;
                }
            }
            assert!(seen.iter().all(|&c| c == 1), "O-groups do not partition at nr {nr} layer {l}: {seen:?}");
        }
        let mut tiles = vec![0; 1010];
        for rank in 0..nr {
            for t in ds41::vocab_tiles(1010, nr, rank) {
                tiles[t] += 1;
            }
        }
        assert!(tiles.iter().all(|&c| c == 1), "lm_head tiles at nr {nr}");
    }
    // TP4: 2 groups = 16 heads a rank; every rank gets 2 groups in every layer
    for l in 0..43 {
        for rank in 0..4 {
            assert_eq!(ds41::o_groups(l, 4, rank).len(), 2);
        }
    }
    // the intermediate slice of 2304: nr must divide 36
    for nr in [1, 2, 3, 4, 6, 9, 12] {
        let mut x = 0;
        for rank in 0..nr {
            let s = ds41::inter_slice(2304, nr, rank).unwrap();
            assert_eq!(s.start, x);
            assert_eq!(s.len() % 64, 0);
            x = s.end;
        }
        assert_eq!(x, 2304);
    }
    for nr in [5, 7, 8, 16] {
        assert!(ds41::inter_slice(2304, nr, 0).is_err(), "nr {nr}");
    }
}

#[test]
fn coverage_and_errors() {
    let mut m = mini("coverage");
    m.write();
    for nr in [1, 2, 4] {
        for rank in 0..nr {
            let (img, _) = plan(&m, nr, rank);
            // no two tensors overlap, all offsets 4096-aligned, the image holds them all
            let mut end = 0;
            for t in &img.tensors {
                assert!(t.offset % 4096 == 0 && t.offset >= end, "{}", t.name);
                end = t.offset + t.nbytes;
            }
            assert!(img.bytes >= end && img.bytes % 4096 == 0);
        }
    }
    // TP8 and TP3 are refused with a reason
    assert!(plan_err(&m.dir, 8, 0).contains("multiple of 64"));
    assert!(ds41::plan(&m.dir, 3, 0, &Opts::default()).is_err());
    assert!(ds41::plan(&m.dir, 4, 4, &Opts::default()).is_err());
    // a tensor nothing plans: the coverage check names it
    m.bf16("layers.1.surprise.weight", &[16]);
    m.write();
    let e = plan_err(&m.dir, 2, 0);
    assert!(e.contains("layers.1.surprise.weight"), "{e}");
    // a sample plan skips the coverage check
    let o = Opts { layers: Some(vec![0]), experts: Some(4), no_globals: true };
    let img = ds41::plan(&m.dir, 2, 0, &o).unwrap().img;
    assert!(img.tensors.iter().all(|t| t.name.starts_with("L000.")));
    assert_eq!(img.tensors.iter().find(|t| t.name == "L000.x.fc1").unwrap().shape[0], 4);
    // a missing tensor
    m.t.remove("layers.1.surprise.weight");
    m.t.remove("layers.2.attn.wkv.weight");
    m.t.remove("layers.2.attn.wkv.scale");
    m.write();
    assert!(plan_err(&m.dir, 2, 0).contains("layers.2.attn.wkv"));
}

#[test]
fn slices_match_the_checkpoint() {
    let m = mini("slices");
    m.write();
    let dsp0 = NMAIN;
    for nr in [1, 2, 4] {
        for rank in 0..nr {
            let (img, _) = plan(&m, nr, rank);
            let names: Vec<&str> = img.tensors.iter().map(|t| t.name.as_str()).collect();
            // globals: embed whole, final_norm, lm_head = this rank's 128-row tiles
            assert_eq!(tensor(&img, "G.embed"), *m.get("embed.weight"));
            assert_eq!(tensor(&img, "G.final_norm"), *m.get("norm.weight"));
            let tiles = ds41::vocab_tiles(V / 128, nr, rank);
            let (r0, r1) = (tiles.start * 128, tiles.end * 128);
            assert_eq!(tensor(&img, "G.lm_head"), m.get("head.weight")[r0 * HID * 2..r1 * HID * 2]);
            let words: Vec<u16> = m.get("head.weight")[r0 * HID * 2..r1 * HID * 2].chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            assert_eq!(tensor(&img, "G.lm_head.gv"), records::bf16_record(&words, HID));
            assert!(!names.contains(&"G.embed.gv"));
            for l in 0..NMAIN + NMTP {
                let (pre, p) = (layer_prefix(l), format!("L{:03}", rl(l)));
                let g = groups_of(rl(l), nr, rank);
                let hpg = HEADS / 8;
                // vectors are whole
                assert_eq!(tensor(&img, &format!("{p}.attn_norm")), *m.get(&format!("{pre}attn_norm.weight")));
                assert_eq!(tensor(&img, &format!("{p}.hc_attn.fn")), *m.get(&format!("{pre}hc_attn_fn")));
                assert_eq!(tensor(&img, &format!("{p}.hc_ffn.scale")), *m.get(&format!("{pre}hc_ffn_scale")));
                assert_eq!(tensor(&img, &format!("{p}.router.bias")), *m.get(&format!("{pre}ffn.gate.bias")));
                assert!(!names.iter().any(|n| n.contains("bias_vl")));
                // attention: q_b rows, attn_sink, o_a rows, o_b columns of this rank's groups
                let sink = m.get(&format!("{pre}attn.attn_sink"));
                assert_eq!(tensor(&img, &format!("{p}.attn_sink")), sink[g.start * hpg * 4..g.end * hpg * 4]);
                let mxcheck = |p2: &str, native: &str, rows: Range<usize>, cols: Range<usize>| {
                    let (codes, scales) = ref_mx(&m, &format!("{pre}{native}"), rows, cols.clone());
                    assert_eq!(tensor(&img, &format!("{p}.{p2}.mx.codes")), codes, "{p2} codes");
                    assert_eq!(tensor(&img, &format!("{p}.{p2}.mx.scales")), scales, "{p2} scales");
                    assert_eq!(tensor(&img, &format!("{p}.{p2}.mx.gv")), records::mxfp8_record(&codes, &scales, cols.len()), "{p2} gv");
                };
                mxcheck("q_a", "attn.wq_a", 0..QA, 0..HID);
                mxcheck("q_b", "attn.wq_b", g.start * hpg * HD..g.end * hpg * HD, 0..QA);
                mxcheck("kv", "attn.wkv", 0..HD, 0..HID);
                mxcheck("o_a", "attn.wo_a", g.start * 32..g.end * 32, 0..HEADS * HD / 8);
                mxcheck("o_b", "attn.wo_b", 0..HID, g.start * 32..g.end * 32);
                // shared expert: the intermediate slice
                let sl = ds41::inter_slice(ISH, nr, rank).unwrap();
                mxcheck("sh.w1", "ffn.shared_experts.w1", sl.clone(), 0..HID);
                mxcheck("sh.w3", "ffn.shared_experts.w3", sl.clone(), 0..HID);
                mxcheck("sh.w2", "ffn.shared_experts.w2", 0..HID, sl);
                if l == 2 {
                    for (p2, native, rows) in [("cmp.wkv", "attn.compressor.wkv.weight", HD), ("cmp.wgate", "attn.compressor.wgate.weight", HD),
                                               ("idx.weights_proj", "attn.indexer.weights_proj.weight", 16), ("idx.wk", "attn.indexer.wk.weight", 32)] {
                        let w = m.get(&format!("{pre}{native}"));
                        let k = m.t[&format!("{pre}{native}")].1[1];
                        assert_eq!(tensor(&img, &format!("{p}.{p2}")), *w);
                        let words: Vec<u16> = w.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                        assert_eq!(rows * k, words.len());
                        assert_eq!(tensor(&img, &format!("{p}.{p2}.gv")), records::bf16_record(&words, k), "{p2}");
                    }
                    mxcheck("idx.wq_b", "attn.indexer.wq_b", 0..64, 0..64);
                    assert_eq!(tensor(&img, &format!("{p}.cmp.norm")), *m.get(&format!("{pre}attn.compressor.norm.weight")));
                    assert_eq!(tensor(&img, &format!("{p}.idx.k_norm")), *m.get(&format!("{pre}attn.indexer.k_norm.weight")));
                } else {
                    assert!(!names.iter().any(|n| n.starts_with(&format!("{p}.cmp.")) || n.starts_with(&format!("{p}.idx."))), "{p}");
                }
                if l == 1 {
                    mxcheck("engram.wkv", "engram.wkv", 0..64, 0..128);
                    assert_eq!(tensor(&img, &format!("{p}.engram.q_weight")), *m.get(&format!("{pre}engram.q_weight")));
                    assert!(!names.iter().any(|n| n.contains("engram.q_weight.gv")), "[4,k] has no decode record");
                }
                if l == dsp0 {
                    assert_eq!(tensor(&img, &format!("{p}.main_norm")), *m.get(&format!("{pre}main_norm.weight")));
                    mxcheck("main_proj", "main_proj", 0..HID, 0..3 * HID);
                }
                if l == dsp0 + 1 {
                    assert_eq!(tensor(&img, &format!("{p}.norm")), *m.get(&format!("{pre}norm.weight")));
                    assert_eq!(tensor(&img, &format!("{p}.markov_head.head")), *m.get(&format!("{pre}markov_head.head.weight")));
                    assert_eq!(tensor(&img, &format!("{p}.confidence_head.proj")), *m.get(&format!("{pre}confidence_head.proj.weight")));
                }
            }
        }
    }
}

#[test]
fn mx4g_experts_decode_to_the_checkpoint_values() {
    let m = mini("mx4g");
    m.write();
    let k1 = HID;
    for nr in [1, 2, 4] {
        for rank in 0..nr {
            let (img, _) = plan(&m, nr, rank);
            let isl = ds41::inter_slice(ISH, nr, rank).unwrap();
            let (i0, iw) = (isl.start, isl.len());
            for l in [0, 2, NMAIN] {
                let (pre, p) = (layer_prefix(l), format!("L{:03}", rl(l)));
                let (fc1, fc2) = (tensor(&img, &format!("{p}.x.fc1")), tensor(&img, &format!("{p}.x.fc2")));
                let t1 = img.tensors.iter().find(|t| t.name == format!("{p}.x.fc1")).unwrap();
                let t2 = img.tensors.iter().find(|t| t.name == format!("{p}.x.fc2")).unwrap();
                assert_eq!(t1.shape, [E as i64, (iw / 16) as i64, (k1 / 64) as i64, 1088]);
                assert_eq!(t2.shape, [E as i64, (HID / 16) as i64, (iw / 64) as i64, 544]);
                let (per1, per2) = (iw / 16 * (k1 / 64) * 1088, HID / 16 * (iw / 64) * 544);
                assert_eq!((fc1.len(), fc2.len()), (E * per1, E * per2));
                for e in [0, 1, E / 2, E - 1] {
                    let x = |w: &str, s: &str| m.get(&format!("{pre}ffn.experts.{e}.{w}.{s}"));
                    // fc1: gate record then up record per (16-row chunk, 64-k group)
                    for ch in 0..iw / 16 {
                        for g in 0..k1 / 64 {
                            let at = e * per1 + (ch * (k1 / 64) + g) * 1088;
                            for (half, w) in [(0, "w1"), (1, "w3")] {
                                let d = decode_record(&fc1[at + half * 544..at + half * 544 + 544]);
                                for r in 0..16 {
                                    for kk in 0..64 {
                                        let want = fp4_value(x(w, "weight"), x(w, "scale"), k1, i0 + ch * 16 + r, g * 64 + kk);
                                        assert_eq!(d[r][kk], want, "nr {nr} rank {rank} L{l} e{e} {w} chunk {ch} g {g} r {r} k {kk}");
                                    }
                                }
                            }
                        }
                    }
                    // fc2: w2 rows by 16, columns i0 + 64 g + kk
                    for ot in 0..HID / 16 {
                        for g in 0..iw / 64 {
                            let at = e * per2 + (ot * (iw / 64) + g) * 544;
                            let d = decode_record(&fc2[at..at + 544]);
                            for r in 0..16 {
                                for kk in 0..64 {
                                    let want = fp4_value(x("w2", "weight"), x("w2", "scale"), ISH, ot * 16 + r, i0 + g * 64 + kk);
                                    assert_eq!(d[r][kk], want, "nr {nr} rank {rank} L{l} e{e} w2 ot {ot} g {g} r {r} k {kk}");
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn engram_slices_rebuild_the_table() {
    let m = mini("engram");
    m.write();
    let (w_all, s_all) = (m.get("layers.1.engram.embed.weight").clone(), m.get("layers.1.engram.embed.scale").clone());
    let mut crossed = false;
    for nr in [1, 2, 4] {
        let (mut w_cat, mut s_cat) = (Vec::new(), Vec::new());
        for rank in 0..nr {
            let (_, eg) = plan(&m, nr, rank);
            assert_eq!(eg.len(), 1);
            let e = &eg[0];
            assert_eq!((e.layer, e.rows, e.dim, e.sdim), (1, ENG_ROWS, ENG_DIM, ENG_SDIM));
            assert_eq!(e.part, ENG_ROWS.div_ceil(nr));
            // the rank's rows, gathered from its runs of the split files
            let (mut w, mut s) = (Vec::new(), Vec::new());
            let mut next = e.first;
            for seg in &e.segs {
                assert_eq!(seg.row0, next, "runs are consecutive");
                let f = std::fs::read(&seg.path).unwrap();
                w.extend_from_slice(&f[seg.w_off as usize..seg.w_off as usize + seg.rows * ENG_DIM]);
                s.extend_from_slice(&f[seg.s_off as usize..seg.s_off as usize + seg.rows * ENG_SDIM]);
                next += seg.rows;
            }
            assert_eq!(next, e.first + e.valid);
            crossed |= e.segs.len() > 1;
            w_cat.extend_from_slice(&w);
            s_cat.extend_from_slice(&s);
            assert_eq!(e.bytes(), (e.part * (ENG_DIM + ENG_SDIM)) as u64);
        }
        assert_eq!(w_cat, w_all, "nr {nr}");
        assert_eq!(s_cat, s_all, "nr {nr}");
    }
    assert!(crossed, "some rank's rows cross split files");
}

#[test]
fn sizes_and_ffi() {
    let m = mini("ffi");
    m.write();
    let (img, eg) = plan(&m, 4, 2);
    let rep = ds41::size_report(&img, &eg);
    assert!(rep.contains("x.fc1") && rep.contains("engram total") && rep.contains("DSpark layers"), "{rep}");
    // routed experts at TP4: E * (iw/16 * K/64 * 1088 + H/16 * iw/64 * 544) per main layer
    let kinds = ds41::kinds(&img);
    let iw = ISH / 4;
    assert_eq!(kinds["x.fc1"].1, ((NMAIN + NMTP) * E * (iw / 16) * (HID / 64) * 1088) as u64);
    assert_eq!(kinds["x.fc2"].1, ((NMAIN + NMTP) * E * (HID / 16) * (iw / 64) * 544) as u64);
    // g6l_ds41_plan hands out a plan the usual g6l_plan_read / g6l_plan_free serve
    let dir = CString::new(m.dir.clone()).unwrap();
    let out = CString::new(format!("{}/ffi.manifest.json", m.dir)).unwrap();
    let mut err = vec![0 as std::ffi::c_char; 512];
    let eout = CString::new(format!("{}/ffi.engram.json", m.dir)).unwrap();
    let h = unsafe { ds41::g6l_ds41_plan(dir.as_ptr(), 4, 2, out.as_ptr(), eout.as_ptr(), err.as_mut_ptr(), err.len()) };
    assert!(!h.is_null());
    let t = img.tensors.iter().find(|t| t.name == "L002.x.fc2").unwrap();
    let mut buf = vec![0u8; t.nbytes as usize];
    let rc = unsafe { g6load::g6l_plan_read(h, t.offset, t.nbytes, buf.as_mut_ptr() as *mut _, err.as_mut_ptr(), err.len()) };
    assert_eq!(rc, 0);
    assert_eq!(buf, tensor(&img, "L002.x.fc2"));
    unsafe { g6load::g6l_plan_free(h) };
    let man = g6load::json::parse(&std::fs::read_to_string(out.to_str().unwrap()).unwrap()).unwrap();
    assert_eq!(man.get("tensors").and_then(|v| v.arr()).unwrap().len(), img.tensors.len());
    // a bad rank gives NULL and a message
    let h = unsafe { ds41::g6l_ds41_plan(dir.as_ptr(), 4, 9, out.as_ptr(), eout.as_ptr(), err.as_mut_ptr(), err.len()) };
    assert!(h.is_null());
    let msg = unsafe { std::ffi::CStr::from_ptr(err.as_ptr()) }.to_string_lossy().into_owned();
    assert!(msg.contains("rank 9"), "{msg}");
}

/// The engine's load path: g6l_ds41_plan, then g6l_plan_load into a sink (what packfind.h rs_load_release does). Every
/// tensor of the loaded image must equal what the pack writer would store, and the padding between them must be zero.
#[test]
fn engine_load_path_fills_the_image() {
    use std::os::raw::c_void;
    use std::sync::Mutex;
    struct Ctx {
        img: Mutex<Vec<u8>>,
        staging: Mutex<Vec<Vec<u8>>>,
    }
    extern "C" fn alloc(ctx: *mut c_void, n: usize) -> *mut c_void {
        let c = unsafe { &*(ctx as *const Ctx) };
        let mut v = vec![0u8; n];
        let p = v.as_mut_ptr();
        c.staging.lock().unwrap().push(v);
        p as *mut c_void
    }
    extern "C" fn free(_ctx: *mut c_void, _p: *mut c_void) {}
    extern "C" fn copy(ctx: *mut c_void, _w: i32, dst: u64, src: *const c_void, n: usize) -> i32 {
        let c = unsafe { &*(ctx as *const Ctx) };
        let mut img = c.img.lock().unwrap();
        let (a, b) = (dst as usize, dst as usize + n);
        if b > img.len() {
            return 1;
        }
        img[a..b].copy_from_slice(unsafe { std::slice::from_raw_parts(src as *const u8, n) });
        0
    }
    let m = mini("load");
    m.write();
    for (nr, rank) in [(4usize, 1usize), (2, 0)] {
        let (img, _) = plan(&m, nr, rank);
        let ctx = Ctx { img: Mutex::new(vec![0xAAu8; img.bytes as usize]), staging: Mutex::new(Vec::new()) };
        let sink = g6load::G6lSink { ctx: &ctx as *const Ctx as *mut c_void, alloc_staging: alloc, free_staging: free, copy };
        let dir = CString::new(m.dir.clone()).unwrap();
        let out = CString::new(format!("{}/load.manifest.json", m.dir)).unwrap();
        let mut err = vec![0 as std::ffi::c_char; 512];
        let eout = CString::new(format!("{}/load.engram.json", m.dir)).unwrap();
        let h = unsafe { ds41::g6l_ds41_plan(dir.as_ptr(), nr as u32, rank as u32, out.as_ptr(), eout.as_ptr(), err.as_mut_ptr(), err.len()) };
        assert!(!h.is_null());
        let mut st = g6load::G6lStats::default();
        let rc = unsafe { g6load::g6l_plan_load(h, &sink, &mut st, err.as_mut_ptr(), err.len()) };
        let msg = unsafe { std::ffi::CStr::from_ptr(err.as_ptr()) }.to_string_lossy().into_owned();
        assert_eq!(rc, 0, "nr {nr} rank {rank}: {msg}");
        unsafe { g6load::g6l_plan_free(h) };
        assert_eq!(st.image_bytes, img.bytes);
        let loaded = ctx.img.lock().unwrap();
        let mut covered = vec![false; loaded.len()];
        for t in &img.tensors {
            let (a, b) = (t.offset as usize, (t.offset + t.nbytes) as usize);
            assert_eq!(&loaded[a..b], &tensor(&img, &t.name)[..], "nr {nr} rank {rank}: {}", t.name);
            covered[a..b].iter_mut().for_each(|c| *c = true);
        }
        let stray = loaded.iter().zip(&covered).filter(|(&v, &c)| !c && v != 0).count();
        assert_eq!(stray, 0, "nr {nr} rank {rank}: {stray} nonzero padding bytes");
    }
}

/// The Rust mx4g packer against builder B6's C++ one (engine/kernels/mxfp4_pack.h), byte for byte, on random data
/// with a column slice. Runs when DS41_B6_KERNELS is set.
#[test]
fn mx4g_matches_the_b6_cpp_packer() {
    let Some(kdir) = gr8knobs::TEST_DS41_KERNELS_DIR.get() else {
        eprintln!("GR8_TEST_DS41_KERNELS_DIR not set: C++ cross-check skipped");
        return;
    };
    let mut m = Mini { dir: format!("{}/ds41-test-{}-cpp", std::env::temp_dir().display(), std::process::id()), t: BTreeMap::new(), seed: 12345 };
    std::fs::create_dir_all(&m.dir).unwrap();
    let (r, k, kfull, c0) = (96usize, 192usize, 512usize, 128usize);
    let (gp, ge, up, ue) = (m.rnd(r * k / 2), m.rnd(r * k / 32), m.rnd(r * k / 2), m.rnd(r * k / 32));
    let (w2p, w2e) = (m.rnd(r * kfull / 2), m.rnd(r * kfull / 32));
    let mut input = Vec::new();
    for v in [r, k, kfull, c0] {
        input.extend_from_slice(&(v as i32).to_le_bytes());
    }
    for b in [&gp, &ge, &up, &ue, &w2p, &w2e] {
        input.extend_from_slice(b);
    }
    let (inp, outp, exe) = (format!("{}/in.bin", m.dir), format!("{}/out.bin", m.dir), format!("{}/cpp", m.dir));
    std::fs::write(&inp, input).unwrap();
    let src = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/ds41_mx4g_cpp.cpp");
    let st = std::process::Command::new("g++").args(["-O1", "-std=c++17", &format!("-I{kdir}"), src, "-o", &exe]).status().unwrap();
    assert!(st.success(), "g++");
    assert!(std::process::Command::new(&exe).args([&inp, &outp]).status().unwrap().success());
    let cpp = std::fs::read(&outp).unwrap();
    let fc1 = ds41::mx4g_fc1(&gp, &ge, &up, &ue, r, k);
    let fc2 = ds41::mx4g_fc2(&w2p, &w2e, r, kfull, c0, k);
    assert_eq!(cpp.len(), fc1.len() + fc2.len());
    assert!(cpp[..fc1.len()] == fc1[..], "fc1 differs from the C++ packer");
    assert!(cpp[fc1.len()..] == fc2[..], "fc2 differs from the C++ packer");
    eprintln!("mx4g: fc1 {} B and fc2 {} B identical to mxfp4_pack.h", fc1.len(), fc2.len());
}

