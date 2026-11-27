//! Unit tests for the `embed_plan` multi-group dense embedding release C-ABI (`g6l_embed_*`).

use g6load::embed_plan::EmbedRelease;
use g6load::exlr8::{bf16_bits, BF16, FP8_ROW, SIDE_OUT, SIDE_REPLICATED};
use g6load::xxh64::Xxh64;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

fn temp_dir(tag: &str) -> PathBuf {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!("g6load-embed-capi-{tag}-{}-{n}", std::process::id()));
    let _ = fs::remove_dir_all(&p);
    fs::create_dir_all(&p).unwrap();
    p
}

struct TEnt {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn write_st(path: &Path, tensors: &[TEnt], meta: &[(&str, &str)]) {
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
}

fn add_fp8_row(dst: &mut Vec<TEnt>, name: &str, out: usize, inp: usize) -> (Vec<u8>, Vec<u8>) {
    let rec: [i64; 8] = [FP8_ROW, out as i64, inp as i64, 0, SIDE_OUT, 128, 0, 0];
    let fp8: Vec<u8> = (0..out * inp).map(|i| ((i * 37 + 13) % 120) as u8).collect();
    let scale: Vec<u8> = (0..out).flat_map(|r| (0.01f32 * (r as f32 + 1.0)).to_le_bytes()).collect();
    let n_blocks = out.div_ceil(128);
    let mut csum = Vec::with_capacity(n_blocks * 8);
    for b in 0..n_blocks {
        let lo = b * 128;
        let hi = (lo + 128).min(out);
        let mut h = Xxh64::new(0);
        h.update(&fp8[lo * inp..hi * inp]);
        h.update(&scale[lo * 4..hi * 4]);
        csum.extend_from_slice(&h.digest().to_le_bytes());
    }
    dst.push(TEnt {
        name: format!("{name}.record"),
        dtype: "I64".into(),
        shape: vec![8],
        bytes: rec.iter().flat_map(|x| x.to_le_bytes()).collect(),
    });
    dst.push(TEnt { name: format!("{name}.checksum"), dtype: "U64".into(), shape: vec![n_blocks], bytes: csum });
    dst.push(TEnt { name: format!("{name}.fp8"), dtype: "U8".into(), shape: vec![out, inp], bytes: fp8.clone() });
    dst.push(TEnt { name: format!("{name}.scale"), dtype: "F32".into(), shape: vec![out], bytes: scale.clone() });
    (fp8, scale)
}

fn add_bf16(dst: &mut Vec<TEnt>, name: &str, out: usize, inp: usize) -> Vec<u8> {
    let rec: [i64; 8] = [BF16, out as i64, inp as i64, 0, SIDE_REPLICATED, 0, 0, 0];
    let raw: Vec<u8> = (0..out * inp).flat_map(|i| bf16_bits((i % 29) as f32 * 0.25).to_le_bytes()).collect();
    let n_blocks = out.div_ceil(128);
    let mut csum = Vec::with_capacity(n_blocks * 8);
    for b in 0..n_blocks {
        let lo = b * 128;
        let hi = (lo + 128).min(out);
        let mut h = Xxh64::new(0);
        h.update(&raw[lo * inp * 2..hi * inp * 2]);
        csum.extend_from_slice(&h.digest().to_le_bytes());
    }
    dst.push(TEnt {
        name: format!("{name}.record"),
        dtype: "I64".into(),
        shape: vec![8],
        bytes: rec.iter().flat_map(|x| x.to_le_bytes()).collect(),
    });
    dst.push(TEnt { name: format!("{name}.checksum"), dtype: "U64".into(), shape: vec![n_blocks], bytes: csum });
    dst.push(TEnt { name: format!("{name}.bf16"), dtype: "BF16".into(), shape: vec![out, inp], bytes: raw.clone() });
    raw
}

#[test]
fn embed_release_opens_verifies_reads_components_and_decodes_bf16() {
    let dir = temp_dir("qwen3vl");
    let mut vis = Vec::new();
    let (fp8_fc1, sc_fc1) = add_fp8_row(&mut vis, "mlp.linear_fc1", 144, 64);
    write_st(
        &dir.join("exlr8-dense-vision-000.safetensors"),
        &vis,
        &[("format", "exlr8-v2"), ("contents", "dense"), ("model_profile", "qwen3_vl"), ("group", "vision"), ("layer", "0")],
    );

    let mut glob = Vec::new();
    let pos_raw = add_bf16(&mut glob, "model.visual.pos_embed", 16, 64);
    let conv_raw: Vec<u8> = (0..4 * 3 * 2 * 4 * 4).flat_map(|i| bf16_bits(i as f32 * 0.01).to_le_bytes()).collect();
    glob.push(TEnt {
        name: "model.visual.patch_embed.proj.weight".into(),
        dtype: "BF16".into(),
        shape: vec![4, 3, 2, 4, 4],
        bytes: conv_raw.clone(),
    });
    write_st(
        &dir.join("exlr8-dense-global-000.safetensors"),
        &glob,
        &[("format", "exlr8-v2"), ("contents", "dense"), ("model_profile", "qwen3_vl"), ("group", "global"), ("layer", "0")],
    );

    let rel = EmbedRelease::open(dir.to_str().unwrap(), true).unwrap();
    assert_eq!(rel.profile.to_str().unwrap(), "qwen3_vl");
    assert_eq!(rel.items.len(), 3);

    // Verify component reads match stored bytes exactly.
    let fc1_idx = rel
        .items
        .iter()
        .position(|x| x.hf_name.to_str().unwrap() == "model.visual.blocks.0.mlp.linear_fc1.weight")
        .unwrap();
    assert_eq!(rel.items[fc1_idx].fmt, FP8_ROW as i32);
    assert_eq!(rel.items[fc1_idx].components.len(), 2);
    let mut got_fp8 = vec![0u8; fp8_fc1.len()];
    let mut got_sc = vec![0u8; sc_fc1.len()];
    rel.read_component(fc1_idx, 0, &mut got_fp8).unwrap();
    rel.read_component(fc1_idx, 1, &mut got_sc).unwrap();
    assert_eq!(got_fp8, fp8_fc1);
    assert_eq!(got_sc, sc_fc1);

    let mut decoded_bf16 = vec![0u8; 144 * 64 * 2];
    rel.decode_bf16(fc1_idx, &mut decoded_bf16, 2).unwrap();
    assert_eq!(decoded_bf16.len(), 144 * 64 * 2);

    let pos_idx = rel
        .items
        .iter()
        .position(|x| x.hf_name.to_str().unwrap() == "model.visual.pos_embed.weight")
        .unwrap();
    let mut got_pos = vec![0u8; pos_raw.len()];
    rel.decode_bf16(pos_idx, &mut got_pos, 1).unwrap();
    assert_eq!(got_pos, pos_raw);

    let conv_idx = rel
        .items
        .iter()
        .position(|x| x.hf_name.to_str().unwrap() == "model.visual.patch_embed.proj.weight")
        .unwrap();
    assert!(!rel.items[conv_idx].is_matrix);
    assert_eq!(rel.items[conv_idx].ndim, 5);
    let mut got_conv = vec![0u8; conv_raw.len()];
    rel.read_component(conv_idx, 0, &mut got_conv).unwrap();
    assert_eq!(got_conv, conv_raw);

    // Tamper one byte -> open with verify=true must fail closed.
    let mut bad_vis = vis;
    bad_vis.iter_mut().find(|e| e.name == "mlp.linear_fc1.fp8").unwrap().bytes[0] ^= 1;
    write_st(
        &dir.join("exlr8-dense-vision-000.safetensors"),
        &bad_vis,
        &[("format", "exlr8-v2"), ("contents", "dense"), ("model_profile", "qwen3_vl"), ("group", "vision"), ("layer", "0")],
    );
    assert!(EmbedRelease::open(dir.to_str().unwrap(), true).unwrap_err().contains("checksum"));

    let _ = fs::remove_dir_all(&dir);
}
