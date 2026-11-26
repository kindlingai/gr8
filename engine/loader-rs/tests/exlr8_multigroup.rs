//! Multi-group exlr8 dense release discovery, metadata resolution, XXH64 checksum verification, and bit-exact decode
//! parity against `exlr8/loader/exlr8.py` fixtures for both `embedding_gemma2` and `qwen3_vl` (including 8B non-128
//! `FP8_ROW` `side=OUT` ViT MLP matrices, copied 2D `BF16` `pos_embed`, and 5D `patch_embed.proj.weight`).

use g6load::exlr8::{
    self, bf16_bits, e4m3_to_f32, parse_dense_filename, DenseProfile, DenseRelease, Matrix, StFile, BF16, FP8_ROW,
    INT4, SIDE_IN, SIDE_OUT, SIDE_REPLICATED, TRELLIS,
};
use g6load::xxh64::Xxh64;
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

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

fn write_safetensors(path: &Path, tensors: &[TensorEntry], meta: &[(&str, &str)]) {
    let mut head = String::from("{");
    let mut off = 0usize;
    for t in tensors {
        let shp: Vec<String> = t.shape.iter().map(usize::to_string).collect();
        head += &format!(
            "\"{}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}},",
            t.name,
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
        head += &format!("\"{k}\":\"{v}\"");
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
            FP8_ROW => {
                let hi = (lo + 128).min(out);
                h.update(&data["fp8"].1[lo * inp..hi * inp]);
                h.update(&data["scale"].1[lo * 4..hi * 4]);
            }
            BF16 => {
                let hi = (lo + 128).min(out);
                h.update(&data["bf16"].1[lo * inp * 2..hi * inp * 2]);
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

/// Builds a synthetic `FP8_ROW` matrix with `side=OUT` (e.g., 8B ViT non-128 `mlp.linear_fc1` / `linear_fc2`), returning
/// its exact reference FP32 vector.
fn add_fp8_row(dst: &mut Vec<TensorEntry>, name: &str, out: usize, inp: usize, seed: u32) -> Vec<f32> {
    let rec: [i64; 8] = [FP8_ROW, out as i64, inp as i64, 0, SIDE_OUT, 128, 0, 0];
    let mut fp8 = Vec::with_capacity(out * inp);
    let mut s = seed.wrapping_add(0x9e37_79b9);
    for _ in 0..out * inp {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        let mut b = (s >> 24) as u8;
        if (b & 0x7f) == 0x7f {
            b ^= 1; // avoid E4M3 NaN
        }
        fp8.push(b);
    }
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

fn bf16_vec(len: usize, base: f32) -> Vec<u8> {
    (0..len).flat_map(|i| bf16_bits(base + (i % 13) as f32 * 0.125).to_le_bytes()).collect()
}

#[test]
fn filename_parser_handles_legacy_multigroup_and_vocab() {
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
fn embedding_gemma2_multigroup_release_in_nested_dense_dir() {
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

    let mut voc_t = Vec::new();
    let fnv_e = copy_ref_matrix(&mut voc_t, &fixture, "fr", "embed_tokens");
    write_safetensors(
        &dense.join("exlr8-vocab.safetensors"),
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
fn checksum_tamper_and_invalid_fp8_row_side_and_metadata_conflicts_fail_closed() {
    let dir = temp_dir("errors");

    // 1. Tampered FP8_ROW fails verify_checksums
    let mut t = Vec::new();
    let _ = add_fp8_row(&mut t, "mlp.linear_fc2", 128, 144, 7);
    let fp8_entry = t.iter_mut().find(|e| e.name == "mlp.linear_fc2.fp8").unwrap();
    fp8_entry.bytes[0] ^= 0x01;
    let bad_path = dir.join("exlr8-dense-vision-000.safetensors");
    write_safetensors(
        &bad_path,
        &t,
        &[("format", "exlr8-v2"), ("contents", "dense"), ("model_profile", "qwen3_vl"), ("group", "vision"), ("layer", "0")],
    );
    let f = StFile::open(bad_path.to_str().unwrap()).unwrap();
    let m = Matrix::open(&f, "mlp.linear_fc2").unwrap();
    assert!(m.verify_checksums().unwrap_err().contains("checksum"));

    // 2. FP8_ROW with side=IN is rejected explicitly
    let mut t_in = Vec::new();
    let _ = add_fp8_row(&mut t_in, "bad_side", 128, 144, 9);
    let rec_entry = t_in.iter_mut().find(|e| e.name == "bad_side.record").unwrap();
    rec_entry.bytes[4 * 8..5 * 8].copy_from_slice(&SIDE_IN.to_le_bytes());
    let side_path = dir.join("bad_side.safetensors");
    write_safetensors(&side_path, &t_in, &[("format", "exlr8-v2")]);
    let f2 = StFile::open(side_path.to_str().unwrap()).unwrap();
    let m2 = Matrix::open(&f2, "bad_side").unwrap();
    assert!(m2.verify_checksums().unwrap_err().contains("side=OUT"));

    // 3. Conflicting filename vs metadata group fails discovery
    let mut t_ok = Vec::new();
    let _ = add_fp8_row(&mut t_ok, "mlp.linear_fc1", 144, 128, 3);
    write_safetensors(
        &bad_path,
        &t_ok,
        &[("format", "exlr8-v2"), ("contents", "dense"), ("model_profile", "qwen3_vl"), ("group", "text"), ("layer", "0")],
    );
    assert!(DenseRelease::discover(dir.to_str().unwrap()).unwrap_err().contains("disagrees"));

    let _ = fs::remove_dir_all(&dir);
}
