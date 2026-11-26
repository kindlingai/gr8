//! Multi-group exlr8 dense release discovery, metadata resolution, XXH64 checksum verification, and bit-exact decode
//! parity against `exlr8/loader/exlr8.py` fixtures for both `embedding_gemma2` and `qwen3_vl` (including 8B non-128
//! `FP8_ROW` `side=OUT` ViT MLP matrices, copied 2D `BF16` `pos_embed`, and 5D `patch_embed.proj.weight`).

use g6load::exlr8::{
    self, bf16_bits, e4m3_to_f32, group_dense_path, parse_dense_filename, DenseFileEntry, DenseProfile, DenseRelease,
    Matrix, StFile, BF16, FP32, FP8_BLOCK, FP8_ROW, INT4, MXFP8, SIDE_IN, SIDE_OUT, SIDE_REPLICATED, TRELLIS,
};
use g6load::xxh64::Xxh64;
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

fn ref_fixture() -> StFile {
    StFile::open(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/exlr8-fixture.safetensors")).unwrap()
}

fn fnv_f32(v: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for x in v {
        for b in x.to_le_bytes() {
            h = (h ^ b as u64).wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn fnv_bytes_as_f32(b: &[u8]) -> u64 {
    let v: Vec<f32> = b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
    fnv_f32(&v)
}

fn temp_dir(tag: &str) -> PathBuf {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!("g6load-mg-{tag}-{}-{n}", std::process::id()));
    let _ = fs::remove_dir_all(&p);
    fs::create_dir_all(&p).unwrap();
    p
}

struct TensorEntry {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn escape_json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

fn write_safetensors(path: &Path, tensors: &[TensorEntry], meta: &[(&str, &str)]) {
    let mut head = String::from("{");
    let mut off = 0usize;
    for t in tensors {
        let shp: Vec<String> = t.shape.iter().map(usize::to_string).collect();
        head += &format!(
            "\"{}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}},",
            escape_json_str(&t.name),
            t.dtype,
            shp.join(","),
            off,
            off + t.bytes.len()
        );
        off += t.bytes.len();
    }
    head += "\"__metadata__\":{";
    for (i, (k, v)) in meta.iter().enumerate() {
        if i > 0 {
            head.push(',');
        }
        head += &format!("\"{}\":\"{}\"", escape_json_str(k), escape_json_str(v));
    }
    head += "}}";
    while head.len() % 8 != 0 {
        head.push(' ');
    }
    let mut f = fs::File::create(path).unwrap();
    f.write_all(&(head.len() as u64).to_le_bytes()).unwrap();
    f.write_all(head.as_bytes()).unwrap();
    for t in tensors {
        f.write_all(&t.bytes).unwrap();
    }
    f.flush().unwrap();
}

/// Computes `M.checksum` (XXH64 seed 0 per 128-channel block) matching `dense_format.checksums()`.
fn compute_checksums(rec: &[i64; 8], data: &BTreeMap<&str, (&[usize], &[u8])>) -> Vec<u8> {
    let (fmt, out, inp, _k, side) = (rec[0], rec[1] as usize, rec[2] as usize, rec[3] as usize, rec[4]);
    let n_blocks = (if side == SIDE_IN { inp } else { out }).div_ceil(128);
    let mut out_bytes = Vec::with_capacity(n_blocks * 8);
    for b in 0..n_blocks {
        let mut h = Xxh64::new(0);
        let lo = b * 128;
        match fmt {
            TRELLIS => {
                let (t_shp, t_bytes) = data["trellis"];
                let eb = t_shp[1];
                h.update(&t_bytes[b * eb..(b + 1) * eb]);
                if side == SIDE_IN {
                    let hi = (lo + 128).min(inp);
                    h.update(&data["suh"].1[lo * 2..hi * 2]);
                    h.update(data["svh"].1);
                } else {
                    let hi = (lo + 128).min(out);
                    h.update(&data["svh"].1[lo * 2..hi * 2]);
                    h.update(data["suh"].1);
                }
            }
            INT4 if side == SIDE_IN => {
                let q_step = out * 64;
                let s_step = out * 8;
                h.update(&data["q4"].1[b * q_step..(b + 1) * q_step]);
                h.update(&data["s4"].1[b * s_step..(b + 1) * s_step]);
            }
            INT4 => {
                let hi = (lo + 128).min(out);
                let qr = inp / 2;
                let sr = inp / 32 * 2;
                h.update(&data["q4"].1[lo * qr..hi * qr]);
                h.update(&data["s4"].1[lo * sr..hi * sr]);
            }
            MXFP8 => {
                let hi = (lo + 128).min(out);
                h.update(&data["fp8"].1[lo * inp..hi * inp]);
                h.update(&data["e8m0"].1[lo * (inp / 32)..hi * (inp / 32)]);
            }
            FP8_BLOCK => {
                let hi = (lo + 128).min(out);
                let bi = inp.div_ceil(128);
                h.update(&data["fp8"].1[lo * inp..hi * inp]);
                h.update(&data["scale"].1[b * bi * 4..(b + 1) * bi * 4]);
            }
            FP8_ROW => {
                let hi = (lo + 128).min(out);
                h.update(&data["fp8"].1[lo * inp..hi * inp]);
                h.update(&data["scale"].1[lo * 4..hi * 4]);
            }
            BF16 => {
                let hi = (lo + 128).min(out);
                h.update(&data["bf16"].1[lo * inp * 2..hi * inp * 2]);
            }
            FP32 => {
                let hi = (lo + 128).min(out);
                h.update(&data["fp32"].1[lo * inp * 4..hi * inp * 4]);
            }
            _ => panic!("unsupported fmt {fmt} in test helper"),
        }
        out_bytes.extend_from_slice(&h.digest().to_le_bytes());
    }
    out_bytes
}

/// Copies a reference matrix from `tests/data/exlr8-fixture.safetensors` under a new matrix name and adds valid
/// `.checksum` tensor. Returns its expected FP32 FNV-1a hash from the Python reference loader.
fn copy_ref_matrix(dst: &mut Vec<TensorEntry>, fixture: &StFile, src_name: &str, dst_name: &str) -> u64 {
    let rec_bytes = fixture.read_all(&format!("{src_name}.record")).unwrap();
    let rec_vec: Vec<i64> = rec_bytes.as_chunks::<8>().0.iter().map(|c| i64::from_le_bytes(*c)).collect();
    let rec: [i64; 8] = rec_vec.try_into().unwrap();
    let fmt = rec[0];

    let mut owned_data: Vec<(&'static str, String, Vec<usize>, Vec<u8>)> = Vec::new();
    for &suffix in exlr8::data_tensors(fmt) {
        let info = fixture.info(&format!("{src_name}.{suffix}")).unwrap();
        let bytes = fixture.read_all(&format!("{src_name}.{suffix}")).unwrap();
        owned_data.push((suffix, info.dtype.clone(), info.shape.clone(), bytes));
    }
    let data_map: BTreeMap<&str, (&[usize], &[u8])> =
        owned_data.iter().map(|(s, _, shp, b)| (*s, (shp.as_slice(), b.as_slice()))).collect();
    let checksum = compute_checksums(&rec, &data_map);
    let n_blocks = checksum.len() / 8;

    dst.push(TensorEntry { name: format!("{dst_name}.record"), dtype: "I64".into(), shape: vec![8], bytes: rec_bytes });
    dst.push(TensorEntry {
        name: format!("{dst_name}.checksum"),
        dtype: "U64".into(),
        shape: vec![n_blocks],
        bytes: checksum,
    });
    for (suffix, dtype, shape, bytes) in owned_data {
        dst.push(TensorEntry { name: format!("{dst_name}.{suffix}"), dtype, shape, bytes });
    }
    u64::from_le_bytes(fixture.read_all(&format!("{src_name}.want_fnv")).unwrap().try_into().unwrap())
}

fn rand_fp8(n: usize, seed: u32) -> Vec<u8> {
    let mut fp8 = Vec::with_capacity(n);
    let mut s = seed.wrapping_add(0x9e37_79b9);
    for _ in 0..n {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        let mut b = (s >> 24) as u8;
        if (b & 0x7f) == 0x7f {
            b ^= 1;
        }
        fp8.push(b);
    }
    fp8
}

/// Builds a synthetic `FP8_ROW` matrix with `side=OUT` (e.g., 8B ViT non-128 `mlp.linear_fc1` / `linear_fc2`), returning
/// its exact reference FP32 vector.
fn add_fp8_row(dst: &mut Vec<TensorEntry>, name: &str, out: usize, inp: usize, seed: u32) -> Vec<f32> {
    let rec: [i64; 8] = [FP8_ROW, out as i64, inp as i64, 0, SIDE_OUT, 128, 0, 0];
    let fp8 = rand_fp8(out * inp, seed);
    let mut scale_f32 = Vec::with_capacity(out);
    let mut scale_bytes = Vec::with_capacity(out * 4);
    for r in 0..out {
        let sc = 0.001f32 * ((r % 17) as f32 + 1.0);
        scale_f32.push(sc);
        scale_bytes.extend_from_slice(&sc.to_le_bytes());
    }
    let mut data_map: BTreeMap<&str, (&[usize], &[u8])> = BTreeMap::new();
    let fp8_shp = [out, inp];
    let sc_shp = [out];
    data_map.insert("fp8", (&fp8_shp, &fp8));
    data_map.insert("scale", (&sc_shp, &scale_bytes));
    let checksum = compute_checksums(&rec, &data_map);
    let n_blocks = checksum.len() / 8;

    let rec_bytes: Vec<u8> = rec.iter().flat_map(|x| x.to_le_bytes()).collect();
    dst.push(TensorEntry { name: format!("{name}.record"), dtype: "I64".into(), shape: vec![8], bytes: rec_bytes });
    dst.push(TensorEntry {
        name: format!("{name}.checksum"),
        dtype: "U64".into(),
        shape: vec![n_blocks],
        bytes: checksum,
    });
    dst.push(TensorEntry { name: format!("{name}.fp8"), dtype: "U8".into(), shape: vec![out, inp], bytes: fp8.clone() });
    dst.push(TensorEntry {
        name: format!("{name}.scale"),
        dtype: "F32".into(),
        shape: vec![out],
        bytes: scale_bytes,
    });
    let mut want = Vec::with_capacity(out * inp);
    for r in 0..out {
        for c in 0..inp {
            want.push(e4m3_to_f32(fp8[r * inp + c]) * scale_f32[r]);
        }
    }
    want
}

fn add_mxfp8(dst: &mut Vec<TensorEntry>, name: &str, out: usize, inp: usize, seed: u32) {
    let rec: [i64; 8] = [MXFP8, out as i64, inp as i64, 0, SIDE_OUT, 128, 0, 0];
    let fp8 = rand_fp8(out * inp, seed);
    let e8m0: Vec<u8> = (0..out * (inp / 32)).map(|i| 120 + (i % 15) as u8).collect();
    let mut data_map: BTreeMap<&str, (&[usize], &[u8])> = BTreeMap::new();
    let fp8_shp = [out, inp];
    let e_shp = [out, inp / 32];
    data_map.insert("fp8", (&fp8_shp, &fp8));
    data_map.insert("e8m0", (&e_shp, &e8m0));
    let checksum = compute_checksums(&rec, &data_map);
    let n_blocks = checksum.len() / 8;

    let rec_bytes: Vec<u8> = rec.iter().flat_map(|x| x.to_le_bytes()).collect();
    dst.push(TensorEntry { name: format!("{name}.record"), dtype: "I64".into(), shape: vec![8], bytes: rec_bytes });
    dst.push(TensorEntry {
        name: format!("{name}.checksum"),
        dtype: "U64".into(),
        shape: vec![n_blocks],
        bytes: checksum,
    });
    dst.push(TensorEntry { name: format!("{name}.fp8"), dtype: "U8".into(), shape: vec![out, inp], bytes: fp8 });
    dst.push(TensorEntry {
        name: format!("{name}.e8m0"),
        dtype: "U8".into(),
        shape: vec![out, inp / 32],
        bytes: e8m0,
    });
}

fn add_fp8_block(dst: &mut Vec<TensorEntry>, name: &str, out: usize, inp: usize, seed: u32) {
    let rec: [i64; 8] = [FP8_BLOCK, out as i64, inp as i64, 0, SIDE_OUT, 128, 0, 0];
    let (bo, bi) = (out.div_ceil(128), inp.div_ceil(128));
    let fp8 = rand_fp8(out * inp, seed);
    let scale_bytes: Vec<u8> = (0..bo * bi).flat_map(|i| (0.01f32 * (i as f32 + 1.0)).to_le_bytes()).collect();
    let mut data_map: BTreeMap<&str, (&[usize], &[u8])> = BTreeMap::new();
    let fp8_shp = [out, inp];
    let sc_shp = [bo, bi];
    data_map.insert("fp8", (&fp8_shp, &fp8));
    data_map.insert("scale", (&sc_shp, &scale_bytes));
    let checksum = compute_checksums(&rec, &data_map);
    let n_blocks = checksum.len() / 8;

    let rec_bytes: Vec<u8> = rec.iter().flat_map(|x| x.to_le_bytes()).collect();
    dst.push(TensorEntry { name: format!("{name}.record"), dtype: "I64".into(), shape: vec![8], bytes: rec_bytes });
    dst.push(TensorEntry {
        name: format!("{name}.checksum"),
        dtype: "U64".into(),
        shape: vec![n_blocks],
        bytes: checksum,
    });
    dst.push(TensorEntry { name: format!("{name}.fp8"), dtype: "U8".into(), shape: vec![out, inp], bytes: fp8 });
    dst.push(TensorEntry {
        name: format!("{name}.scale"),
        dtype: "F32".into(),
        shape: vec![bo, bi],
        bytes: scale_bytes,
    });
}

/// Builds a copied 2D `BF16` matrix (`side=REPLICATED, unit=0`, e.g. `model.visual.pos_embed`), returning its raw BF16
/// bytes and exact FP32 values.
fn add_bf16_matrix(dst: &mut Vec<TensorEntry>, name: &str, out: usize, inp: usize) -> (Vec<u8>, Vec<f32>) {
    let rec: [i64; 8] = [BF16, out as i64, inp as i64, 0, SIDE_REPLICATED, 0, 0, 0];
    let mut raw = Vec::with_capacity(out * inp * 2);
    let mut want = Vec::with_capacity(out * inp);
    for i in 0..out * inp {
        let v = ((i % 251) as f32 - 125.0) * 0.015625;
        let bits = bf16_bits(v);
        raw.extend_from_slice(&bits.to_le_bytes());
        want.push(exlr8::bf16_to_f32(bits));
    }
    let mut data_map: BTreeMap<&str, (&[usize], &[u8])> = BTreeMap::new();
    let shp = [out, inp];
    data_map.insert("bf16", (&shp, &raw));
    let checksum = compute_checksums(&rec, &data_map);
    let n_blocks = checksum.len() / 8;

    let rec_bytes: Vec<u8> = rec.iter().flat_map(|x| x.to_le_bytes()).collect();
    dst.push(TensorEntry { name: format!("{name}.record"), dtype: "I64".into(), shape: vec![8], bytes: rec_bytes });
    dst.push(TensorEntry {
        name: format!("{name}.checksum"),
        dtype: "U64".into(),
        shape: vec![n_blocks],
        bytes: checksum,
    });
    dst.push(TensorEntry { name: format!("{name}.bf16"), dtype: "BF16".into(), shape: vec![out, inp], bytes: raw.clone() });
    (raw, want)
}

fn add_fp32_matrix(dst: &mut Vec<TensorEntry>, name: &str, out: usize, inp: usize) -> Vec<f32> {
    let rec: [i64; 8] = [FP32, out as i64, inp as i64, 0, SIDE_REPLICATED, 0, 0, 0];
    let mut want = Vec::with_capacity(out * inp);
    let mut raw = Vec::with_capacity(out * inp * 4);
    for i in 0..out * inp {
        let v = ((i % 199) as f32 - 99.0) * 0.03125;
        want.push(v);
        raw.extend_from_slice(&v.to_le_bytes());
    }
    let mut data_map: BTreeMap<&str, (&[usize], &[u8])> = BTreeMap::new();
    let shp = [out, inp];
    data_map.insert("fp32", (&shp, &raw));
    let checksum = compute_checksums(&rec, &data_map);
    let n_blocks = checksum.len() / 8;

    let rec_bytes: Vec<u8> = rec.iter().flat_map(|x| x.to_le_bytes()).collect();
    dst.push(TensorEntry { name: format!("{name}.record"), dtype: "I64".into(), shape: vec![8], bytes: rec_bytes });
    dst.push(TensorEntry {
        name: format!("{name}.checksum"),
        dtype: "U64".into(),
        shape: vec![n_blocks],
        bytes: checksum,
    });
    dst.push(TensorEntry { name: format!("{name}.fp32"), dtype: "F32".into(), shape: vec![out, inp], bytes: raw });
    want
}

fn bf16_vec(len: usize, base: f32) -> Vec<u8> {
    (0..len).flat_map(|i| bf16_bits(base + (i % 13) as f32 * 0.125).to_le_bytes()).collect()
}

#[test]
fn filename_parser_and_group_path_validation() {
    assert_eq!(parse_dense_filename("exlr8-vocab.safetensors"), Some((None, None, true)));
    assert_eq!(parse_dense_filename("exlr8-dense-007.safetensors"), Some((None, Some(7), false)));
    assert_eq!(parse_dense_filename("exlr8-dense-text-027.safetensors"), Some((Some("text".into()), Some(27), false)));
    assert_eq!(
        parse_dense_filename("exlr8-dense-vision-000.safetensors"),
        Some((Some("vision".into()), Some(0), false))
    );
    assert_eq!(parse_dense_filename("exlr8-dense-global-000.safetensors"), Some((Some("global".into()), Some(0), false)));
    assert_eq!(parse_dense_filename("exlr8-layer-000.k4.safetensors"), None);
    assert_eq!(parse_dense_filename("model.safetensors"), None);
    // Overflowing layer numbers must return None rather than Some((..., None, false)).
    assert_eq!(parse_dense_filename("exlr8-dense-99999999999999999999999999.safetensors"), None);
    assert_eq!(parse_dense_filename("exlr8-dense-text-99999999999999999999999999.safetensors"), None);

    assert!(group_dense_path("/tmp", Some("text"), Some(0)).is_ok());
    assert!(group_dense_path("/tmp", Some("../evil"), Some(0)).is_err());
    assert!(group_dense_path("/tmp", Some("a/b"), Some(0)).is_err());
    assert!(group_dense_path("/tmp", Some(""), Some(0)).is_err());
}

#[test]
fn qwen3_vl_multigroup_release_discovers_verifies_and_matches_reference_bit_for_bit() {
    let fixture = ref_fixture();
    let dir = temp_dir("qwen3vl");

    // 1. text layer 0: trellis OUT + trellis IN + int4 IN + 1D norm
    let mut text_t = Vec::new();
    let fnv_q = copy_ref_matrix(&mut text_t, &fixture, "t_out_k4", "self_attn.q_proj");
    let fnv_o = copy_ref_matrix(&mut text_t, &fixture, "t_in_k4", "self_attn.o_proj");
    let fnv_down = copy_ref_matrix(&mut text_t, &fixture, "i_in", "mlp.down_proj");
    let norm_bytes = bf16_vec(64, 1.0);
    text_t.push(TensorEntry {
        name: "input_layernorm.weight".into(),
        dtype: "BF16".into(),
        shape: vec![64],
        bytes: norm_bytes.clone(),
    });
    // Stripped metadata (matches strip_meta.py KEEP keys)
    write_safetensors(
        &dir.join("exlr8-dense-text-000.safetensors"),
        &text_t,
        &[("format", "exlr8-v2"), ("contents", "dense"), ("model_profile", "qwen3_vl"), ("group", "text"), ("layer", "0")],
    );

    // 2. vision layer 0: trellis OUT + 8B-style non-128 FP8_ROW (side=OUT) linear_fc1 [144, 128] & linear_fc2 [128, 144] + bias
    let mut vis_t = Vec::new();
    let fnv_qkv = copy_ref_matrix(&mut vis_t, &fixture, "t_out_k3", "attn.qkv");
    let want_fc1 = add_fp8_row(&mut vis_t, "mlp.linear_fc1", 144, 128, 11);
    let want_fc2 = add_fp8_row(&mut vis_t, "mlp.linear_fc2", 128, 144, 29);
    let qkv_bias = bf16_vec(128, -0.25);
    vis_t.push(TensorEntry {
        name: "attn.qkv.bias".into(),
        dtype: "BF16".into(),
        shape: vec![128],
        bytes: qkv_bias.clone(),
    });
    write_safetensors(
        &dir.join("exlr8-dense-vision-000.safetensors"),
        &vis_t,
        &[("format", "exlr8-v2"), ("contents", "dense"), ("model_profile", "qwen3_vl"), ("group", "vision"), ("layer", "0")],
    );

    // 3. global layer 0: copied 2D BF16 pos_embed + trellis merger + DeepStack merger + 5D Conv3d patch_embed.proj.weight
    let mut glob_t = Vec::new();
    let (pos_bf16_raw, want_pos_f32) = add_bf16_matrix(&mut glob_t, "model.visual.pos_embed", 160, 64);
    let fnv_m1 = copy_ref_matrix(&mut glob_t, &fixture, "t_out_k4", "model.visual.merger.linear_fc1");
    let fnv_ds0 = copy_ref_matrix(&mut glob_t, &fixture, "t_in_k2", "model.visual.deepstack_merger_list.0.linear_fc2");
    let conv_bytes = bf16_vec(4 * 3 * 2 * 4 * 4, 0.5);
    glob_t.push(TensorEntry {
        name: "model.visual.patch_embed.proj.weight".into(),
        dtype: "BF16".into(),
        shape: vec![4, 3, 2, 4, 4],
        bytes: conv_bytes.clone(),
    });
    write_safetensors(
        &dir.join("exlr8-dense-global-000.safetensors"),
        &glob_t,
        &[("format", "exlr8-v2"), ("contents", "dense"), ("model_profile", "qwen3_vl"), ("group", "global"), ("layer", "0")],
    );

    // 4. vocab: FP8_ROW embed_tokens + BF16 norm.weight (no lm_head)
    let mut voc_t = Vec::new();
    let fnv_emb = copy_ref_matrix(&mut voc_t, &fixture, "fr", "embed_tokens");
    let final_norm = bf16_vec(64, 0.75);
    voc_t.push(TensorEntry {
        name: "norm.weight".into(),
        dtype: "BF16".into(),
        shape: vec![64],
        bytes: final_norm.clone(),
    });
    write_safetensors(
        &dir.join("exlr8-vocab.safetensors"),
        &voc_t,
        &[("format", "exlr8-v2"), ("contents", "vocab"), ("model_profile", "qwen3_vl")],
    );

    let rel = DenseRelease::discover(dir.to_str().unwrap()).unwrap();
    assert_eq!(rel.profile, DenseProfile::Qwen3Vl);
    assert_eq!(rel.files.len(), 4);
    assert_eq!(rel.files[0].group.as_deref(), Some("text"));
    assert_eq!(rel.files[1].group.as_deref(), Some("vision"));
    assert_eq!(rel.files[2].group.as_deref(), Some("global"));
    assert!(rel.files[3].is_vocab);

    // file(None, Some(0)) must NOT silently return an arbitrary multi-group file.
    assert!(rel.file(None, Some(0)).is_none());
    assert_eq!(rel.file(Some("vision"), Some(0)).unwrap().group.as_deref(), Some("vision"));
    assert!(rel.file(None, None).unwrap().is_vocab);

    let f32_map = rel.decode_all(false, true, 2).unwrap();
    assert_eq!(f32_map.len(), 14);

    assert_eq!(fnv_bytes_as_f32(&f32_map["model.language_model.layers.0.self_attn.q_proj.weight"].bytes), fnv_q);
    assert_eq!(fnv_bytes_as_f32(&f32_map["model.language_model.layers.0.self_attn.o_proj.weight"].bytes), fnv_o);
    assert_eq!(fnv_bytes_as_f32(&f32_map["model.language_model.layers.0.mlp.down_proj.weight"].bytes), fnv_down);
    assert_eq!(f32_map["model.language_model.layers.0.input_layernorm.weight"].bytes, norm_bytes);

    assert_eq!(fnv_bytes_as_f32(&f32_map["model.visual.blocks.0.attn.qkv.weight"].bytes), fnv_qkv);
    assert_eq!(fnv_bytes_as_f32(&f32_map["model.visual.blocks.0.mlp.linear_fc1.weight"].bytes), fnv_f32(&want_fc1));
    assert_eq!(f32_map["model.visual.blocks.0.mlp.linear_fc1.weight"].shape, vec![144, 128]);
    assert_eq!(fnv_bytes_as_f32(&f32_map["model.visual.blocks.0.mlp.linear_fc2.weight"].bytes), fnv_f32(&want_fc2));
    assert_eq!(f32_map["model.visual.blocks.0.mlp.linear_fc2.weight"].shape, vec![128, 144]);
    assert_eq!(f32_map["model.visual.blocks.0.attn.qkv.bias"].bytes, qkv_bias);

    assert_eq!(fnv_bytes_as_f32(&f32_map["model.visual.pos_embed.weight"].bytes), fnv_f32(&want_pos_f32));
    assert_eq!(fnv_bytes_as_f32(&f32_map["model.visual.merger.linear_fc1.weight"].bytes), fnv_m1);
    assert_eq!(
        fnv_bytes_as_f32(&f32_map["model.visual.deepstack_merger_list.0.linear_fc2.weight"].bytes),
        fnv_ds0
    );
    assert_eq!(f32_map["model.visual.patch_embed.proj.weight"].shape, vec![4, 3, 2, 4, 4]);
    assert_eq!(f32_map["model.visual.patch_embed.proj.weight"].dtype, "BF16");
    assert_eq!(f32_map["model.visual.patch_embed.proj.weight"].bytes, conv_bytes);

    assert_eq!(fnv_bytes_as_f32(&f32_map["model.language_model.embed_tokens.weight"].bytes), fnv_emb);
    assert_eq!(f32_map["model.language_model.norm.weight"].bytes, final_norm);
    assert!(!f32_map.contains_key("lm_head.weight"));

    // BF16 decode preserves copied pos_embed bits identically
    let bf16_map = rel.decode_all(true, true, 2).unwrap();
    assert_eq!(bf16_map["model.visual.pos_embed.weight"].dtype, "BF16");
    assert_eq!(bf16_map["model.visual.pos_embed.weight"].bytes, pos_bf16_raw);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn embedding_gemma2_multigroup_release_in_nested_dense_dir_and_split_layout() {
    let fixture = ref_fixture();
    let dir = temp_dir("eg2");
    let dense = dir.join("dense");
    fs::create_dir_all(&dense).unwrap();

    let mut text_t = Vec::new();
    let fnv_q = copy_ref_matrix(&mut text_t, &fixture, "t_out_k4", "self_attn.q_proj");
    write_safetensors(
        &dense.join("exlr8-dense-text-000.safetensors"),
        &text_t,
        &[("format", "exlr8-v2"), ("contents", "dense"), ("model_profile", "embedding_gemma2"), ("group", "text"), ("layer", "0")],
    );

    let mut vis_t = Vec::new();
    let fnv_vq = copy_ref_matrix(&mut vis_t, &fixture, "t_out_k3", "self_attn.q_proj.linear");
    write_safetensors(
        &dense.join("exlr8-dense-vision-000.safetensors"),
        &vis_t,
        &[("format", "exlr8-v2"), ("contents", "dense"), ("model_profile", "embedding_gemma2"), ("group", "vision"), ("layer", "0")],
    );

    let mut aud_t = Vec::new();
    let fnv_aq = copy_ref_matrix(&mut aud_t, &fixture, "i_out", "self_attn.post.linear");
    write_safetensors(
        &dense.join("exlr8-dense-audio-000.safetensors"),
        &aud_t,
        &[("format", "exlr8-v2"), ("contents", "dense"), ("model_profile", "embedding_gemma2"), ("group", "audio"), ("layer", "0")],
    );

    let mut glob_t = Vec::new();
    let fnv_g = copy_ref_matrix(&mut glob_t, &fixture, "t_out_k4", "language_model.embedding_projection");
    write_safetensors(
        &dense.join("exlr8-dense-global-000.safetensors"),
        &glob_t,
        &[("format", "exlr8-v2"), ("contents", "dense"), ("model_profile", "embedding_gemma2"), ("group", "global"), ("layer", "0")],
    );

    // Put exlr8-vocab.safetensors at root (split layout: dense shards in dir/dense/, vocab in dir/).
    let mut voc_t = Vec::new();
    let fnv_e = copy_ref_matrix(&mut voc_t, &fixture, "fr", "embed_tokens");
    write_safetensors(
        &dir.join("exlr8-vocab.safetensors"),
        &voc_t,
        &[("format", "exlr8-v2"), ("contents", "vocab"), ("model_profile", "embedding_gemma2")],
    );

    let rel = DenseRelease::discover(dir.to_str().unwrap()).unwrap();
    assert_eq!(rel.profile, DenseProfile::EmbeddingGemma2);
    assert_eq!(rel.files.len(), 5);
    assert_eq!(
        rel.files.iter().map(|e| e.group.as_deref()).collect::<Vec<_>>(),
        vec![Some("text"), Some("vision"), Some("audio"), Some("global"), None]
    );

    let out = rel.decode_all(false, true, 2).unwrap();
    assert_eq!(fnv_bytes_as_f32(&out["language_model.layers.0.self_attn.q_proj.weight"].bytes), fnv_q);
    assert_eq!(fnv_bytes_as_f32(&out["vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight"].bytes), fnv_vq);
    assert_eq!(fnv_bytes_as_f32(&out["audio_tower.layers.0.self_attn.post.linear.weight"].bytes), fnv_aq);
    assert_eq!(fnv_bytes_as_f32(&out["language_model.embedding_projection.weight"].bytes), fnv_g);
    assert_eq!(fnv_bytes_as_f32(&out["language_model.embed_tokens.weight"].bytes), fnv_e);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn metadata_overrides_hf_prefix_custom_group_tensor_prefix_and_hf_names_replacement() {
    let dir = temp_dir("overrides");

    // 1. Custom group with explicit hf_prefix succeeds in discover() and decode_all().
    let mut custom_t = Vec::new();
    let _ = add_fp8_row(&mut custom_t, "proj", 64, 64, 1);
    write_safetensors(
        &dir.join("exlr8-dense-custom-000.safetensors"),
        &custom_t,
        &[
            ("format", "exlr8-v2"),
            ("contents", "dense"),
            ("model_profile", "qwen3_vl"),
            ("group", "custom"),
            ("layer", "0"),
            ("hf_prefix", "model.custom_encoder.blocks.0."),
        ],
    );

    // 2. Partial hf_names on vocab replaces profile table completely: unmapped norm.weight stays literal.
    let mut voc_t = Vec::new();
    let _ = add_fp8_row(&mut voc_t, "embed_tokens", 64, 64, 2);
    voc_t.push(TensorEntry { name: "norm.weight".into(), dtype: "BF16".into(), shape: vec![64], bytes: bf16_vec(64, 1.0) });
    write_safetensors(
        &dir.join("exlr8-vocab.safetensors"),
        &voc_t,
        &[
            ("format", "exlr8-v2"),
            ("contents", "vocab"),
            ("model_profile", "qwen3_vl"),
            ("hf_names", r#"{"embed_tokens":"custom.embed"}"#),
        ],
    );

    let rel = DenseRelease::discover(dir.to_str().unwrap()).unwrap();
    let map = rel.decode_all(false, true, 2).unwrap();
    assert!(map.contains_key("model.custom_encoder.blocks.0.proj.weight"));
    assert!(map.contains_key("custom.embed.weight"));
    assert!(map.contains_key("norm.weight"), "unmapped key with hf_names must stay literal");
    assert!(!map.contains_key("model.language_model.norm.weight"));
    let _ = fs::remove_dir_all(&dir);

    // 3. Legacy GLM single-group release: file(None, Some(0)) matches and tensor_prefix override works.
    let glm_dir = temp_dir("glm-prefix");
    let mut glm_t = Vec::new();
    let _ = add_fp8_row(&mut glm_t, "self_attn.q_proj", 64, 64, 3);
    write_safetensors(
        &glm_dir.join("exlr8-dense-000.safetensors"),
        &glm_t,
        &[("format", "exlr8-v2"), ("contents", "dense"), ("tensor_prefix", "backbone.blocks.")],
    );
    let glm_rel = DenseRelease::discover(glm_dir.to_str().unwrap()).unwrap();
    assert_eq!(glm_rel.profile, DenseProfile::Glm);
    assert!(glm_rel.file(None, Some(0)).is_some());
    let glm_map = glm_rel.decode_all(false, true, 1).unwrap();
    assert!(glm_map.contains_key("backbone.blocks.0.self_attn.q_proj.weight"));
    let _ = fs::remove_dir_all(&glm_dir);
}

#[test]
fn checksums_cover_mxfp8_fp8_block_fp32_and_reject_side_in_bad_dims_and_bad_dtypes() {
    let dir = temp_dir("checksums");

    // 1. Valid MXFP8, FP8_BLOCK, FP32 checksums pass, and tampering fails.
    let mut t = Vec::new();
    add_mxfp8(&mut t, "mx", 144, 160, 10);
    add_fp8_block(&mut t, "fb", 144, 160, 20);
    let want_fp32 = add_fp32_matrix(&mut t, "f32", 144, 64);
    let ok_path = dir.join("ok.safetensors");
    write_safetensors(&ok_path, &t, &[("format", "exlr8-v2")]);
    let f = StFile::open(ok_path.to_str().unwrap()).unwrap();
    for name in ["mx", "fb", "f32"] {
        Matrix::open(&f, name).unwrap().verify_checksums().unwrap();
    }
    let got_f32 = Matrix::open(&f, "f32").unwrap().decode(0..144, 0..64).unwrap();
    assert_eq!(fnv_f32(&got_f32), fnv_f32(&want_fp32));

    for (mat, tensor) in [("mx", "mx.e8m0"), ("fb", "fb.scale"), ("f32", "f32.fp32")] {
        let mut bad = Vec::new();
        add_mxfp8(&mut bad, "mx", 144, 160, 10);
        add_fp8_block(&mut bad, "fb", 144, 160, 20);
        let _ = add_fp32_matrix(&mut bad, "f32", 144, 64);
        bad.iter_mut().find(|e| e.name == tensor).unwrap().bytes[0] ^= 0x01;
        let p = dir.join(format!("tamper_{mat}.safetensors"));
        write_safetensors(&p, &bad, &[("format", "exlr8-v2")]);
        let sf = StFile::open(p.to_str().unwrap()).unwrap();
        assert!(Matrix::open(&sf, mat).unwrap().verify_checksums().unwrap_err().contains("checksum"));
    }

    // 2. SIDE_IN rejected cleanly for ALL 5 row-major formats even when inp > out (no panic / no OOB).
    let row_major_cases: Vec<(&str, Vec<TensorEntry>)> = vec![
        ("mx", { let mut v = Vec::new(); add_mxfp8(&mut v, "m", 64, 256, 1); v }),
        ("fb", { let mut v = Vec::new(); add_fp8_block(&mut v, "m", 64, 256, 2); v }),
        ("fr", { let mut v = Vec::new(); let _ = add_fp8_row(&mut v, "m", 64, 256, 3); v }),
        ("bf", { let mut v = Vec::new(); let _ = add_bf16_matrix(&mut v, "m", 64, 256); v }),
        ("f3", { let mut v = Vec::new(); let _ = add_fp32_matrix(&mut v, "m", 64, 256); v }),
    ];
    for (tag, mut entries) in row_major_cases {
        let rec = entries.iter_mut().find(|e| e.name == "m.record").unwrap();
        rec.bytes[4 * 8..5 * 8].copy_from_slice(&SIDE_IN.to_le_bytes());
        let p = dir.join(format!("side_in_{tag}.safetensors"));
        write_safetensors(&p, &entries, &[("format", "exlr8-v2")]);
        let sf = StFile::open(p.to_str().unwrap()).unwrap();
        let err = Matrix::open(&sf, "m").unwrap().verify_checksums().unwrap_err();
        assert!(err.contains("side=OUT"), "{tag}: unexpected err {err}");
    }

    // 3. Non-128 TRELLIS dims and bad dtype on suh must fail cleanly (no panic).
    let rec_bad_trellis: [i64; 8] = [TRELLIS, 130, 128, 4, SIDE_OUT, 128, 1, 1];
    let bad_trellis = vec![
        TensorEntry {
            name: "t.record".into(),
            dtype: "I64".into(),
            shape: vec![8],
            bytes: rec_bad_trellis.iter().flat_map(|x| x.to_le_bytes()).collect(),
        },
        TensorEntry { name: "t.checksum".into(), dtype: "U64".into(), shape: vec![2], bytes: vec![0u8; 16] },
        TensorEntry { name: "t.trellis".into(), dtype: "U8".into(), shape: vec![1, 8192], bytes: vec![0u8; 8192] },
        TensorEntry { name: "t.suh".into(), dtype: "F16".into(), shape: vec![128], bytes: vec![0u8; 256] },
        TensorEntry { name: "t.svh".into(), dtype: "F16".into(), shape: vec![130], bytes: vec![0u8; 260] },
    ];
    let p_bt = dir.join("bad_trellis_dim.safetensors");
    write_safetensors(&p_bt, &bad_trellis, &[("format", "exlr8-v2")]);
    let sf_bt = StFile::open(p_bt.to_str().unwrap()).unwrap();
    assert!(Matrix::open(&sf_bt, "t").unwrap().verify_checksums().unwrap_err().contains("divisible by 128"));

    let rec_ok_trellis: [i64; 8] = [TRELLIS, 128, 128, 4, SIDE_OUT, 128, 1, 1];
    let bad_suh_dtype = vec![
        TensorEntry {
            name: "t.record".into(),
            dtype: "I64".into(),
            shape: vec![8],
            bytes: rec_ok_trellis.iter().flat_map(|x| x.to_le_bytes()).collect(),
        },
        TensorEntry { name: "t.checksum".into(), dtype: "U64".into(), shape: vec![1], bytes: vec![0u8; 8] },
        TensorEntry { name: "t.trellis".into(), dtype: "U8".into(), shape: vec![1, 8192], bytes: vec![0u8; 8192] },
        TensorEntry { name: "t.suh".into(), dtype: "U8".into(), shape: vec![128], bytes: vec![0u8; 128] },
        TensorEntry { name: "t.svh".into(), dtype: "F16".into(), shape: vec![128], bytes: vec![0u8; 256] },
    ];
    let p_dt = dir.join("bad_suh_dtype.safetensors");
    write_safetensors(&p_dt, &bad_suh_dtype, &[("format", "exlr8-v2")]);
    let sf_dt = StFile::open(p_dt.to_str().unwrap()).unwrap();
    assert!(Matrix::open(&sf_dt, "t").unwrap().verify_checksums().unwrap_err().contains("dtype"));

    // 4. Non-128 INT4 side=IN and non-32 INT4 side=OUT rejected cleanly.
    let rec_i4_in: [i64; 8] = [INT4, 64, 130, 4, SIDE_IN, 128, 0, 0];
    let bad_i4_in = vec![
        TensorEntry {
            name: "i.record".into(),
            dtype: "I64".into(),
            shape: vec![8],
            bytes: rec_i4_in.iter().flat_map(|x| x.to_le_bytes()).collect(),
        },
        TensorEntry { name: "i.checksum".into(), dtype: "U64".into(), shape: vec![2], bytes: vec![0u8; 16] },
        TensorEntry { name: "i.q4".into(), dtype: "U8".into(), shape: vec![1, 64, 64], bytes: vec![0u8; 4096] },
        TensorEntry { name: "i.s4".into(), dtype: "F16".into(), shape: vec![1, 64, 4], bytes: vec![0u8; 512] },
    ];
    let p_i4 = dir.join("bad_i4_in.safetensors");
    write_safetensors(&p_i4, &bad_i4_in, &[("format", "exlr8-v2")]);
    let sf_i4 = StFile::open(p_i4.to_str().unwrap()).unwrap();
    assert!(Matrix::open(&sf_i4, "i").unwrap().verify_checksums().unwrap_err().contains("divisible by 128"));

    // 5. BF16 fast-path rejects wrong dtype even when verify=false.
    let mut bad_bf = Vec::new();
    let _ = add_bf16_matrix(&mut bad_bf, "pos", 16, 16);
    let bf_entry = bad_bf.iter_mut().find(|e| e.name == "pos.bf16").unwrap();
    bf_entry.dtype = "U8".into();
    bf_entry.bytes.truncate(16 * 16);
    let p_bf = dir.join("exlr8-dense-global-000.safetensors");
    write_safetensors(
        &p_bf,
        &bad_bf,
        &[("format", "exlr8-v2"), ("contents", "dense"), ("model_profile", "qwen3_vl"), ("group", "global"), ("layer", "0")],
    );
    let sf_bf = StFile::open(p_bf.to_str().unwrap()).unwrap();
    let entry_bf = DenseFileEntry {
        path: p_bf.to_string_lossy().into_owned(),
        profile: DenseProfile::Qwen3Vl,
        group: Some("global".into()),
        layer: Some(0),
        is_vocab: false,
    };
    assert!(exlr8::decode_dense_file(&sf_bf, &entry_bf, true, false, 1).unwrap_err().contains("dtype"));

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn v4_files_mixed_profiles_metadata_conflicts_and_duplicate_hf_names_fail_closed() {
    // 1. exlr8-v4 format rejected
    let d_v4 = temp_dir("v4-fmt");
    let mut t = Vec::new();
    let _ = add_fp8_row(&mut t, "proj", 64, 64, 1);
    write_safetensors(
        &d_v4.join("exlr8-dense-text-000.safetensors"),
        &t,
        &[("format", "exlr8-v4"), ("contents", "dense"), ("model_profile", "qwen3_vl"), ("group", "text"), ("layer", "0")],
    );
    assert!(DenseRelease::discover(d_v4.to_str().unwrap()).unwrap_err().contains("exlr8-v4"));
    let _ = fs::remove_dir_all(&d_v4);

    // 2. __metadata__.matrices rejected even if format says exlr8-v2
    let d_mat = temp_dir("v4-matrices");
    write_safetensors(
        &d_mat.join("exlr8-dense-text-000.safetensors"),
        &t,
        &[
            ("format", "exlr8-v2"),
            ("contents", "dense"),
            ("model_profile", "qwen3_vl"),
            ("group", "text"),
            ("layer", "0"),
            ("matrices", "{}"),
        ],
    );
    assert!(DenseRelease::discover(d_mat.to_str().unwrap()).unwrap_err().contains("matrices"));
    let _ = fs::remove_dir_all(&d_mat);

    // 3. Multi-width M.k4.record rejected
    let d_mw = temp_dir("v4-multiwidth");
    let mut mw_t = Vec::new();
    let _ = add_fp8_row(&mut mw_t, "proj.k4", 64, 64, 1);
    write_safetensors(
        &d_mw.join("exlr8-dense-text-000.safetensors"),
        &mw_t,
        &[("format", "exlr8-v2"), ("contents", "dense"), ("model_profile", "qwen3_vl"), ("group", "text"), ("layer", "0")],
    );
    assert!(DenseRelease::discover(d_mw.to_str().unwrap()).unwrap_err().contains("multi-width"));
    let _ = fs::remove_dir_all(&d_mw);

    // 4. Mixed model_profile across files rejected
    let d_mix = temp_dir("mixed-prof");
    write_safetensors(
        &d_mix.join("exlr8-dense-text-000.safetensors"),
        &t,
        &[("format", "exlr8-v2"), ("contents", "dense"), ("model_profile", "qwen3_vl"), ("group", "text"), ("layer", "0")],
    );
    write_safetensors(
        &d_mix.join("exlr8-dense-vision-000.safetensors"),
        &t,
        &[
            ("format", "exlr8-v2"),
            ("contents", "dense"),
            ("model_profile", "embedding_gemma2"),
            ("group", "vision"),
            ("layer", "0"),
        ],
    );
    assert!(DenseRelease::discover(d_mix.to_str().unwrap()).unwrap_err().contains("conflicts"));
    let _ = fs::remove_dir_all(&d_mix);

    // 5. Conflicting filename vs metadata group rejected
    let d_grp = temp_dir("grp-conflict");
    write_safetensors(
        &d_grp.join("exlr8-dense-vision-000.safetensors"),
        &t,
        &[("format", "exlr8-v2"), ("contents", "dense"), ("model_profile", "qwen3_vl"), ("group", "text"), ("layer", "0")],
    );
    assert!(DenseRelease::discover(d_grp.to_str().unwrap()).unwrap_err().contains("disagrees"));
    let _ = fs::remove_dir_all(&d_grp);

    // 6. Duplicate HF tensor name across files rejected
    let d_dup = temp_dir("dup-hf");
    write_safetensors(
        &d_dup.join("exlr8-dense-text-000.safetensors"),
        &t,
        &[
            ("format", "exlr8-v2"),
            ("contents", "dense"),
            ("model_profile", "qwen3_vl"),
            ("group", "text"),
            ("layer", "0"),
            ("hf_prefix", "dup."),
        ],
    );
    write_safetensors(
        &d_dup.join("exlr8-dense-vision-000.safetensors"),
        &t,
        &[
            ("format", "exlr8-v2"),
            ("contents", "dense"),
            ("model_profile", "qwen3_vl"),
            ("group", "vision"),
            ("layer", "0"),
            ("hf_prefix", "dup."),
        ],
    );
    let rel_dup = DenseRelease::discover(d_dup.to_str().unwrap()).unwrap();
    assert!(rel_dup.decode_all(false, true, 1).unwrap_err().contains("duplicate HF tensor"));
    let _ = fs::remove_dir_all(&d_dup);
}

#[test]
fn exlr8dump_cli_multigroup_verify_shard_and_error_guards() {
    let fixture = ref_fixture();
    let dir = temp_dir("cli");
    let bin = env!("CARGO_BIN_EXE_exlr8dump");

    // Build text:0 with a sharded trellis matrix (unit=128, out=256) + a 1D norm.
    let mut text_t = Vec::new();
    let want_fc1 = add_fp8_row(&mut text_t, "mlp.up", 256, 128, 77);
    let fnv_q = copy_ref_matrix(&mut text_t, &fixture, "t_out_k4", "self_attn.q_proj");
    let norm = bf16_vec(64, 1.0);
    text_t.push(TensorEntry { name: "input_layernorm.weight".into(), dtype: "BF16".into(), shape: vec![64], bytes: norm.clone() });
    write_safetensors(
        &dir.join("exlr8-dense-text-000.safetensors"),
        &text_t,
        &[("format", "exlr8-v2"), ("contents", "dense"), ("model_profile", "qwen3_vl"), ("group", "text"), ("layer", "0")],
    );

    // 1. GROUP:LAYER with --verify --dtype f32
    let out_f32 = dir.join("out_f32.safetensors");
    let st = Command::new(bin)
        .args([dir.to_str().unwrap(), "text:0", out_f32.to_str().unwrap(), "--dtype", "f32", "--verify"])
        .output()
        .unwrap();
    assert!(st.status.success(), "stderr: {}", String::from_utf8_lossy(&st.stderr));
    let sf = StFile::open(out_f32.to_str().unwrap()).unwrap();
    assert_eq!(fnv_bytes_as_f32(&sf.read_all("model.language_model.layers.0.self_attn.q_proj.weight").unwrap()), fnv_q);
    assert_eq!(fnv_bytes_as_f32(&sf.read_all("model.language_model.layers.0.mlp.up.weight").unwrap()), fnv_f32(&want_fc1));
    assert_eq!(sf.read_all("model.language_model.layers.0.input_layernorm.weight").unwrap(), norm);

    // 2. --shard 2 0 and --shard 2 1 concatenate to full mlp.up.weight
    let s0 = dir.join("s0.safetensors");
    let s1 = dir.join("s1.safetensors");
    for (rank, path) in [(0, &s0), (1, &s1)] {
        let st = Command::new(bin)
            .args([
                dir.to_str().unwrap(),
                "text:0",
                path.to_str().unwrap(),
                "--dtype",
                "f32",
                "--shard",
                "2",
                &rank.to_string(),
                "--verify",
            ])
            .output()
            .unwrap();
        assert!(st.status.success(), "rank {rank} stderr: {}", String::from_utf8_lossy(&st.stderr));
    }
    let sf0 = StFile::open(s0.to_str().unwrap()).unwrap();
    let sf1 = StFile::open(s1.to_str().unwrap()).unwrap();
    let mut joined = sf0.read_all("model.language_model.layers.0.mlp.up.weight").unwrap();
    joined.extend_from_slice(&sf1.read_all("model.language_model.layers.0.mlp.up.weight").unwrap());
    assert_eq!(fnv_bytes_as_f32(&joined), fnv_f32(&want_fc1));

    // 3. Path traversal group rejected
    let bad_out = dir.join("bad.safetensors");
    let st_trav = Command::new(bin)
        .args([dir.to_str().unwrap(), "../evil:0", bad_out.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!st_trav.status.success());
    assert!(String::from_utf8_lossy(&st_trav.stderr).contains("invalid"));

    // 4. --shard with unit=0 on a non-replicated matrix returns a clean error instead of divide-by-zero panic
    let mut zero_unit = Vec::new();
    let _ = add_fp8_row(&mut zero_unit, "broken", 128, 128, 5);
    let rec_entry = zero_unit.iter_mut().find(|e| e.name == "broken.record").unwrap();
    rec_entry.bytes[5 * 8..6 * 8].copy_from_slice(&0i64.to_le_bytes()); // unit = 0, side = SIDE_OUT
    write_safetensors(
        &dir.join("exlr8-dense-vision-000.safetensors"),
        &zero_unit,
        &[("format", "exlr8-v2"), ("contents", "dense"), ("model_profile", "qwen3_vl"), ("group", "vision"), ("layer", "0")],
    );
    let st_u0 = Command::new(bin)
        .args([dir.to_str().unwrap(), "vision:0", bad_out.to_str().unwrap(), "--shard", "2", "0"])
        .output()
        .unwrap();
    assert!(!st_u0.status.success());
    assert!(String::from_utf8_lossy(&st_u0.stderr).contains("unit 0"));

    // 5. CLI GROUP:LAYER disagrees with metadata group -> fails closed
    write_safetensors(
        &dir.join("exlr8-dense-global-000.safetensors"),
        &zero_unit,
        &[("format", "exlr8-v2"), ("contents", "dense"), ("model_profile", "qwen3_vl"), ("group", "text"), ("layer", "0")],
    );
    let st_mm = Command::new(bin)
        .args([dir.to_str().unwrap(), "global:0", bad_out.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!st_mm.status.success());
    assert!(String::from_utf8_lossy(&st_mm.stderr).contains("disagrees"));

    let _ = fs::remove_dir_all(&dir);
}
