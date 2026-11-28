//! Integration tests for the Drama 3 (`fish_omni`) TTS release loader (`tts_plan.rs` & `g6l_tts_*` C ABI).
//!
//! Tests both:
//! 1. Store 0 (`dev-ref` layout): 5 dense groups (`gdn`, `attn`, `mtp_shared`, `mtp`, `global`),
//!    `exlr8-vocab.safetensors`, and `exlr8-dense-experts-LLL.safetensors` with `FP8_BLOCK` routed experts
//!    and `BF16` shared experts. Verifies dense discovery, checksum enforcement, raw FP8 read, and FP16 decode
//!    against an exact hand computation.
//! 2. Store 1 (trellis layout): `experts/exlr8-layer-LLL.index.safetensors`, `exlr8-layer-LLL.k4.safetensors`
//!    (`mcg` and `pmcgx`), and width manifest `manifests/exlr8-manifest-bit400.json`. Verifies trellis expert
//!    decoding (gate, up, down, and shared) via `g6l_tts_expert_decode_f16` against an independent computation
//!    using `exlr8::decode_tile_cb` + Hadamard side-vector reconstruction.

use g6load::codebook::Codebook;
use g6load::exlr8::{
    bf16_bits, e4m3_to_f32, f16_bits, f16_to_f32, tiles_in_major_cb, tiles_out_major_cb, BF16,
    FP8_BLOCK, FP8_ROW, SIDE_OUT, SIDE_REPLICATED,
};
use g6load::tts_plan::TtsRelease;
use g6load::xxh64::Xxh64;
use std::ffi::{c_char, c_void, CStr, CString};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

fn temp_dir(tag: &str) -> PathBuf {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!("g6load-tts-capi-{tag}-{}-{n}", std::process::id()));
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
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
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

fn add_bf16(dst: &mut Vec<TEnt>, name: &str, out: usize, inp: usize, seed: usize) -> Vec<f32> {
    let rec: [i64; 8] = [BF16, out as i64, inp as i64, 0, SIDE_REPLICATED, 0, 0, 0];
    let mut vals = Vec::with_capacity(out * inp);
    let mut raw = Vec::with_capacity(out * inp * 2);
    for i in 0..out * inp {
        let v = (((i * 7 + seed * 13) % 41) as f32 - 20.0) * 0.125;
        let bits = bf16_bits(v);
        vals.push(f32::from_bits((bits as u32) << 16));
        raw.extend_from_slice(&bits.to_le_bytes());
    }
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
    dst.push(TEnt { name: format!("{name}.bf16"), dtype: "BF16".into(), shape: vec![out, inp], bytes: raw });
    vals
}

fn add_fp8_row(dst: &mut Vec<TEnt>, name: &str, out: usize, inp: usize) {
    let rec: [i64; 8] = [FP8_ROW, out as i64, inp as i64, 0, SIDE_OUT, 128, 0, 0];
    let fp8: Vec<u8> = (0..out * inp).map(|i| ((i * 29 + 7) % 112) as u8).collect();
    let scale: Vec<u8> = (0..out).flat_map(|r| (0.02f32 * (r as f32 + 1.0)).to_le_bytes()).collect();
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
    dst.push(TEnt { name: format!("{name}.fp8"), dtype: "U8".into(), shape: vec![out, inp], bytes: fp8 });
    dst.push(TEnt { name: format!("{name}.scale"), dtype: "F32".into(), shape: vec![out], bytes: scale });
}

fn add_fp8_block(dst: &mut Vec<TEnt>, name: &str, out: usize, inp: usize, seed: usize) -> (Vec<u8>, Vec<f32>) {
    let rec: [i64; 8] = [FP8_BLOCK, out as i64, inp as i64, 0, SIDE_OUT, 128, 0, 0];
    let (bo, bi) = (out.div_ceil(128), inp.div_ceil(128));
    // Avoid E4M3 NaN (0x7f, 0xff)
    let fp8: Vec<u8> = (0..out * inp)
        .map(|i| {
            let v = ((i * 31 + seed * 17) & 0x7e) as u8;
            if (i + seed) % 3 == 0 { v | 0x80 } else { v }
        })
        .collect();
    let scales: Vec<f32> = (0..bo * bi).map(|i| 0.05f32 * ((i % 7) as f32 + 1.0)).collect();
    let scale_bytes: Vec<u8> = scales.iter().flat_map(|s| s.to_le_bytes()).collect();
    let mut csum = Vec::with_capacity(bo * 8);
    for b in 0..bo {
        let lo = b * 128;
        let hi = (lo + 128).min(out);
        let mut h = Xxh64::new(0);
        h.update(&fp8[lo * inp..hi * inp]);
        h.update(&scale_bytes[b * bi * 4..(b + 1) * bi * 4]);
        csum.extend_from_slice(&h.digest().to_le_bytes());
    }
    dst.push(TEnt {
        name: format!("{name}.record"),
        dtype: "I64".into(),
        shape: vec![8],
        bytes: rec.iter().flat_map(|x| x.to_le_bytes()).collect(),
    });
    dst.push(TEnt { name: format!("{name}.checksum"), dtype: "U64".into(), shape: vec![bo], bytes: csum });
    dst.push(TEnt { name: format!("{name}.fp8"), dtype: "U8".into(), shape: vec![out, inp], bytes: fp8.clone() });
    dst.push(TEnt { name: format!("{name}.scale"), dtype: "F32".into(), shape: vec![bo, bi], bytes: scale_bytes });
    (fp8, scales)
}

fn hadamard128() -> Vec<f32> {
    let c = (1.0f64 / 128.0f64.sqrt()) as f32;
    (0..128 * 128u32)
        .map(|i| if ((i / 128) & (i % 128)).count_ones() % 2 == 1 { -c } else { c })
        .collect()
}

fn ref_reconstruct(t: &[f32], rows: usize, cols: usize, suh: &[f32], svh: &[f32]) -> Vec<f32> {
    let h = hadamard128();
    let mut x = vec![0f32; rows * cols];
    for b in 0..rows / 128 {
        for r in 0..128 {
            for c in 0..cols {
                let mut acc = 0.0f32;
                for q in 0..128 {
                    acc = h[r * 128 + q].mul_add(t[(b * 128 + q) * cols + c], acc);
                }
                x[(b * 128 + r) * cols + c] = acc * suh[b * 128 + r];
            }
        }
    }
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for b in 0..cols / 128 {
            for c in 0..128 {
                let mut acc = 0.0f32;
                for q in 0..128 {
                    acc = x[r * cols + b * 128 + q].mul_add(h[q * 128 + c], acc);
                }
                out[r * cols + b * 128 + c] = acc * svh[b * 128 + c];
            }
        }
    }
    out
}

#[test]
fn tts_dev_ref_groups_fp8_experts_and_capi() {
    let dir = temp_dir("devref");
    let h = 256usize;
    let i_exp = 128usize;

    // 1. Five dense groups + vocab + experts dense group
    let mut gdn = Vec::new();
    add_bf16(&mut gdn, "linear_attn.in_proj_qkv", 128, h, 1);
    gdn.push(TEnt {
        name: "input_layernorm.weight".into(),
        dtype: "BF16".into(),
        shape: vec![h],
        bytes: (0..h).flat_map(|_| bf16_bits(0.1).to_le_bytes()).collect(),
    });
    write_st(
        &dir.join("exlr8-dense-gdn-000.safetensors"),
        &gdn,
        &[("format", "exlr8-v2"), ("contents", "dense"), ("model_profile", "fish_omni"), ("group", "gdn"), ("layer", "0")],
    );

    let mut attn = Vec::new();
    add_bf16(&mut attn, "self_attn.q_proj", 128, h, 2);
    write_st(
        &dir.join("exlr8-dense-attn-003.safetensors"),
        &attn,
        &[("format", "exlr8-v2"), ("contents", "dense"), ("model_profile", "fish_omni"), ("group", "attn"), ("layer", "3")],
    );

    let mut ms = Vec::new();
    add_bf16(&mut ms, "eh_proj", h, 2 * h, 3);
    write_st(
        &dir.join("exlr8-dense-mtp_shared-000.safetensors"),
        &ms,
        &[("format", "exlr8-v2"), ("contents", "dense"), ("model_profile", "fish_omni"), ("group", "mtp_shared"), ("layer", "0")],
    );

    let mut mtp = Vec::new();
    add_bf16(&mut mtp, "eh_proj", h, 2 * h, 4);
    write_st(
        &dir.join("exlr8-dense-mtp-000.safetensors"),
        &mtp,
        &[("format", "exlr8-v2"), ("contents", "dense"), ("model_profile", "fish_omni"), ("group", "mtp"), ("layer", "0")],
    );

    let mut glob = Vec::new();
    add_fp8_row(&mut glob, "audio_decoder_mtp.heads.0", 128, h);
    write_st(
        &dir.join("exlr8-dense-global-000.safetensors"),
        &glob,
        &[("format", "exlr8-v2"), ("contents", "dense"), ("model_profile", "fish_omni"), ("group", "global"), ("layer", "0")],
    );

    let mut vocab = Vec::new();
    add_fp8_row(&mut vocab, "embed_tokens", 128, h);
    add_fp8_row(&mut vocab, "lm_head", 128, h);
    vocab.push(TEnt {
        name: "norm.weight".into(),
        dtype: "BF16".into(),
        shape: vec![h],
        bytes: (0..h).flat_map(|_| bf16_bits(0.0).to_le_bytes()).collect(),
    });
    write_st(
        &dir.join("exlr8-vocab.safetensors"),
        &vocab,
        &[("format", "exlr8-v2"), ("contents", "vocab"), ("model_profile", "fish_omni")],
    );

    // Layer 0 experts in dev-ref FP8_BLOCK + BF16 shared_expert
    let mut exp0 = Vec::new();
    let (e0_gate_fp8, e0_gate_sc) = add_fp8_block(&mut exp0, "mlp.experts.0.gate_proj", i_exp, h, 10);
    let _ = add_fp8_block(&mut exp0, "mlp.experts.0.up_proj", i_exp, h, 11);
    let (e0_down_fp8, e0_down_sc) = add_fp8_block(&mut exp0, "mlp.experts.0.down_proj", h, i_exp, 12);
    let _ = add_fp8_block(&mut exp0, "mlp.experts.1.gate_proj", i_exp, h, 13);
    let _ = add_fp8_block(&mut exp0, "mlp.experts.1.up_proj", i_exp, h, 14);
    let _ = add_fp8_block(&mut exp0, "mlp.experts.1.down_proj", h, i_exp, 15);
    let sh_gate_vals = add_bf16(&mut exp0, "mlp.shared_expert.gate_proj", i_exp, h, 16);
    let _ = add_bf16(&mut exp0, "mlp.shared_expert.up_proj", i_exp, h, 17);
    let _ = add_bf16(&mut exp0, "mlp.shared_expert.down_proj", h, i_exp, 18);
    write_st(
        &dir.join("exlr8-dense-experts-000.safetensors"),
        &exp0,
        &[("format", "exlr8-v2"), ("contents", "dense"), ("model_profile", "fish_omni"), ("group", "experts"), ("layer", "0")],
    );

    // Open through C ABI
    let c_dir = CString::new(dir.to_str().unwrap()).unwrap();
    let mut err = [0 as c_char; 256];
    let handle = unsafe { g6load::g6l_tts_open(c_dir.as_ptr(), std::ptr::null(), 1, err.as_mut_ptr(), err.len()) };
    assert!(!handle.is_null(), "g6l_tts_open failed: {:?}", unsafe { CStr::from_ptr(err.as_ptr()) });

    unsafe {
        let prof = CStr::from_ptr(g6load::g6l_tts_profile(handle)).to_str().unwrap();
        assert_eq!(prof, "fish_omni");
        assert_eq!(g6load::g6l_tts_expert_store(handle, 0), 0);
        assert_eq!(g6load::g6l_tts_expert_store(handle, 3), -1);

        // Verify all expected HF names were discovered
        let count = g6load::g6l_tts_dense_item_count(handle);
        let mut names = Vec::new();
        for idx in 0..count {
            let mut it = std::mem::zeroed::<g6load::G6lEmbedItem>();
            assert_eq!(g6load::g6l_tts_dense_item_info(handle, idx, &mut it), 0);
            names.push(CStr::from_ptr(it.hf_name).to_str().unwrap().to_string());
        }
        for want in [
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            "model.language_model.layers.0.input_layernorm.weight",
            "model.language_model.layers.3.self_attn.q_proj.weight",
            "audio_decoder_mtp_shared.layers.0.eh_proj.weight",
            "audio_decoder_mtp.layers.0.eh_proj.weight",
            "audio_decoder_mtp.heads.0.weight",
            "model.language_model.embed_tokens.weight",
            "lm_head.weight",
            "model.language_model.norm.weight",
        ] {
            assert!(names.iter().any(|n| n == want), "missing expected HF tensor {want}");
        }

        // Verify g6l_tts_dense_decode_f32 on FP8_ROW matrix (`audio_decoder_mtp.heads.0.weight`)
        let head_idx = names
            .iter()
            .position(|n| n == "audio_decoder_mtp.heads.0.weight")
            .expect("missing head item");
        let mut head_f32 = vec![0u8; 128 * h * 4];
        assert_eq!(
            g6load::g6l_tts_dense_decode_f32(
                handle,
                head_idx,
                head_f32.as_mut_ptr() as *mut c_void,
                head_f32.len(),
                2,
                err.as_mut_ptr(),
                err.len(),
            ),
            0
        );
        for r in 0..128 {
            let sc = 0.02f32 * (r as f32 + 1.0);
            for c in 0..h {
                let idx = r * h + c;
                let want = e4m3_to_f32(((idx * 29 + 7) % 112) as u8) * sc;
                let got = f32::from_le_bytes(head_f32[4 * idx..4 * idx + 4].try_into().unwrap());
                assert_eq!(got.to_bits(), want.to_bits(), "dense_decode_f32 mismatch at ({r},{c})");
            }
        }

        // Verify raw FP8_BLOCK read for expert 0 gate_proj
        let mut got_fp8 = vec![0u8; i_exp * h];
        let mut got_sc = vec![0u8; (i_exp / 128) * (h / 128) * 4];
        assert_eq!(
            g6load::g6l_tts_expert_read_fp8(
                handle,
                0,
                0,
                0,
                got_fp8.as_mut_ptr() as *mut c_void,
                got_fp8.len(),
                got_sc.as_mut_ptr() as *mut c_void,
                got_sc.len(),
                err.as_mut_ptr(),
                err.len(),
            ),
            0
        );
        assert_eq!(got_fp8, e0_gate_fp8);

        // Verify FP16 decode for expert 0 gate_proj and down_proj
        let mut gate_f16 = vec![0u8; i_exp * h * 2];
        assert_eq!(
            g6load::g6l_tts_expert_decode_f16(
                handle,
                0,
                0,
                0,
                gate_f16.as_mut_ptr() as *mut c_void,
                gate_f16.len(),
                err.as_mut_ptr(),
                err.len(),
            ),
            0
        );
        let bi = h / 128;
        for r in 0..i_exp {
            for c in 0..h {
                let want = f16_bits(e4m3_to_f32(e0_gate_fp8[r * h + c]) * e0_gate_sc[(r / 128) * bi + (c / 128)]);
                let got = u16::from_le_bytes([gate_f16[2 * (r * h + c)], gate_f16[2 * (r * h + c) + 1]]);
                assert_eq!(got, want, "gate_proj mismatch at ({r},{c})");
            }
        }

        let mut down_f16 = vec![0u8; h * i_exp * 2];
        assert_eq!(
            g6load::g6l_tts_expert_decode_f16(
                handle,
                0,
                0,
                2,
                down_f16.as_mut_ptr() as *mut c_void,
                down_f16.len(),
                err.as_mut_ptr(),
                err.len(),
            ),
            0
        );
        let bi_d = i_exp / 128;
        for r in 0..h {
            for c in 0..i_exp {
                let want = f16_bits(e4m3_to_f32(e0_down_fp8[r * i_exp + c]) * e0_down_sc[(r / 128) * bi_d + (c / 128)]);
                let got = u16::from_le_bytes([down_f16[2 * (r * i_exp + c)], down_f16[2 * (r * i_exp + c) + 1]]);
                assert_eq!(got, want, "down_proj mismatch at ({r},{c})");
            }
        }

        // Verify shared expert (expert == n_routed == 2) gate_proj decode
        let mut sh_f16 = vec![0u8; i_exp * h * 2];
        assert_eq!(
            g6load::g6l_tts_expert_decode_f16(
                handle,
                0,
                2,
                0,
                sh_f16.as_mut_ptr() as *mut c_void,
                sh_f16.len(),
                err.as_mut_ptr(),
                err.len(),
            ),
            0
        );
        for (idx, &v) in sh_gate_vals.iter().enumerate() {
            let got = u16::from_le_bytes([sh_f16[2 * idx], sh_f16[2 * idx + 1]]);
            assert_eq!(got, f16_bits(v), "shared gate mismatch at {idx}");
        }

        g6load::g6l_tts_close(handle);
    }

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn tts_trellis_expert_decode_matches_hadamard_reference() {
    let dir = temp_dir("trellis");
    let h = 256usize;
    let u_blocks = 1usize; // intermediate = 128
    let i_exp = u_blocks * 128;
    let n_routed = 2usize;
    let k = 4usize;
    let hb = 16 * k * h; // 16384 bytes per half

    // Minimal dense file so EmbedRelease discovers profile=fish_omni
    let mut gdn = Vec::new();
    add_bf16(&mut gdn, "linear_attn.in_proj_qkv", 128, h, 7);
    write_st(
        &dir.join("exlr8-dense-gdn-000.safetensors"),
        &gdn,
        &[("format", "exlr8-v2"), ("contents", "dense"), ("model_profile", "fish_omni"), ("group", "gdn"), ("layer", "0")],
    );

    // Build synthetic width file `experts/exlr8-layer-000.k4.safetensors` with pmcgx codebook
    let mk_raw = |seed: usize| -> Vec<u8> { (0..hb).map(|i| ((i * 53 + seed * 97 + 11) & 0xff) as u8).collect() };
    let mut routed_gate = Vec::new();
    let mut routed_up = Vec::new();
    let mut routed_down = Vec::new();
    for e in 0..n_routed {
        routed_gate.extend_from_slice(&mk_raw(e * 10 + 1));
        routed_up.extend_from_slice(&mk_raw(e * 10 + 2));
        routed_down.extend_from_slice(&mk_raw(e * 10 + 3));
    }
    let sh_gate = mk_raw(91);
    let sh_up = mk_raw(92);
    let sh_down = mk_raw(93);

    let wf_path = dir.join("experts/exlr8-layer-000.k4.safetensors");
    write_st(
        &wf_path,
        &[
            TEnt { name: "trellis.gate.pmcgx".into(), dtype: "U8".into(), shape: vec![routed_gate.len()], bytes: routed_gate.clone() },
            TEnt { name: "trellis.up.pmcgx".into(), dtype: "U8".into(), shape: vec![routed_up.len()], bytes: routed_up.clone() },
            TEnt { name: "trellis.down.pmcgx".into(), dtype: "U8".into(), shape: vec![routed_down.len()], bytes: routed_down.clone() },
            TEnt { name: "shared.trellis.gate.pmcgx".into(), dtype: "U8".into(), shape: vec![sh_gate.len()], bytes: sh_gate.clone() },
            TEnt { name: "shared.trellis.up.pmcgx".into(), dtype: "U8".into(), shape: vec![sh_up.len()], bytes: sh_up.clone() },
            TEnt { name: "shared.trellis.down.pmcgx".into(), dtype: "U8".into(), shape: vec![sh_down.len()], bytes: sh_down.clone() },
        ],
        &[
            ("format", "exlr8-v2"),
            ("contents", "trellis"),
            ("codebook", "pmcgx V2 L20 0xCBAC1FED/0xBCA14C51, every value rounded to E4M3"),
        ],
    );

    // Read back exact absolute offsets from the written width file
    let wf = g6load::exlr8::StFile::open(wf_path.to_str().unwrap()).unwrap();
    let off_rg = wf.info("trellis.gate.pmcgx").unwrap().start;
    let off_ru = wf.info("trellis.up.pmcgx").unwrap().start;
    let off_rd = wf.info("trellis.down.pmcgx").unwrap().start;
    let off_sg = wf.info("shared.trellis.gate.pmcgx").unwrap().start;
    let off_su = wf.info("shared.trellis.up.pmcgx").unwrap().start;
    let off_sd = wf.info("shared.trellis.down.pmcgx").unwrap().start;

    let signs_in: Vec<i8> = (0..h).map(|i| if i % 3 == 0 { -1 } else { 1 }).collect();
    let signs_out: Vec<i8> = (0..h).map(|i| if i % 5 == 0 { -1 } else { 1 }).collect();
    let nblk = h / 128;

    let mk_f16_vec = |len: usize, base: f32| -> Vec<u16> {
        (0..len).map(|i| f16_bits(base + (i % 13) as f32 * 0.03125)).collect()
    };
    let to_bytes = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };

    let g_svh = mk_f16_vec(n_routed * i_exp, 0.5);
    let g_su = mk_f16_vec(n_routed * nblk, 0.75);
    let u_svh = mk_f16_vec(n_routed * i_exp, 0.4);
    let u_su = mk_f16_vec(n_routed * nblk, 0.6);
    let d_suh = mk_f16_vec(n_routed * i_exp, 0.3);
    let d_sv = mk_f16_vec(n_routed * nblk, 0.8);

    let sg_svh = mk_f16_vec(i_exp, 0.25);
    let sg_su = mk_f16_vec(nblk, 0.9);
    let su_svh = mk_f16_vec(i_exp, 0.35);
    let su_su = mk_f16_vec(nblk, 0.85);
    let sd_suh = mk_f16_vec(i_exp, 0.45);
    let sd_sv = mk_f16_vec(nblk, 0.95);

    let mut r_index = Vec::<i64>::new();
    let mut r_csum = Vec::<u64>::new();
    let bases = [off_rg, off_ru, off_rd];
    let raws = [&routed_gate, &routed_up, &routed_down];
    let chans = [&g_svh, &u_svh, &d_suh];
    let blks = [&g_su, &u_su, &d_sv];
    for e in 0..n_routed {
        for p in 0..3 {
            let off = bases[p] + (e * hb) as u64;
            r_index.push(off as i64);
            r_index.push(hb as i64);
            let mut xh = Xxh64::new(0);
            xh.update(&raws[p][e * hb..(e + 1) * hb]);
            xh.update(&to_bytes(&chans[p][e * i_exp..(e + 1) * i_exp]));
            xh.update(&to_bytes(&blks[p][e * nblk..(e + 1) * nblk]));
            r_csum.push(xh.digest());
        }
    }

    let mut s_index = Vec::<i64>::new();
    let mut s_csum = Vec::<u64>::new();
    let sbases = [off_sg, off_su, off_sd];
    let sraws = [&sh_gate, &sh_up, &sh_down];
    let schans = [&sg_svh, &su_svh, &sd_suh];
    let sblks = [&sg_su, &su_su, &sd_sv];
    for p in 0..3 {
        s_index.push(sbases[p] as i64);
        s_index.push(hb as i64);
        let mut xh = Xxh64::new(0);
        xh.update(sraws[p]);
        xh.update(&to_bytes(schans[p]));
        xh.update(&to_bytes(sblks[p]));
        s_csum.push(xh.digest());
    }

    write_st(
        &dir.join("experts/exlr8-layer-000.index.safetensors"),
        &[
            TEnt { name: "signs.in".into(), dtype: "I8".into(), shape: vec![h], bytes: signs_in.iter().map(|&x| x as u8).collect() },
            TEnt { name: "signs.out".into(), dtype: "I8".into(), shape: vec![h], bytes: signs_out.iter().map(|&x| x as u8).collect() },
            TEnt { name: "k4.index".into(), dtype: "I64".into(), shape: vec![n_routed, 3, u_blocks, 2], bytes: r_index.iter().flat_map(|x| x.to_le_bytes()).collect() },
            TEnt { name: "k4.checksum".into(), dtype: "U64".into(), shape: vec![n_routed, 3, u_blocks], bytes: r_csum.iter().flat_map(|x| x.to_le_bytes()).collect() },
            TEnt { name: "k4.gate.svh".into(), dtype: "F16".into(), shape: vec![n_routed, i_exp], bytes: to_bytes(&g_svh) },
            TEnt { name: "k4.gate.su_block".into(), dtype: "F16".into(), shape: vec![n_routed, nblk], bytes: to_bytes(&g_su) },
            TEnt { name: "k4.up.svh".into(), dtype: "F16".into(), shape: vec![n_routed, i_exp], bytes: to_bytes(&u_svh) },
            TEnt { name: "k4.up.su_block".into(), dtype: "F16".into(), shape: vec![n_routed, nblk], bytes: to_bytes(&u_su) },
            TEnt { name: "k4.down.suh".into(), dtype: "F16".into(), shape: vec![n_routed, i_exp], bytes: to_bytes(&d_suh) },
            TEnt { name: "k4.down.sv_block".into(), dtype: "F16".into(), shape: vec![n_routed, nblk], bytes: to_bytes(&d_sv) },
            TEnt { name: "shared.k4.index".into(), dtype: "I64".into(), shape: vec![1, 3, u_blocks, 2], bytes: s_index.iter().flat_map(|x| x.to_le_bytes()).collect() },
            TEnt { name: "shared.k4.checksum".into(), dtype: "U64".into(), shape: vec![1, 3, u_blocks], bytes: s_csum.iter().flat_map(|x| x.to_le_bytes()).collect() },
            TEnt { name: "shared.k4.gate.svh".into(), dtype: "F16".into(), shape: vec![1, i_exp], bytes: to_bytes(&sg_svh) },
            TEnt { name: "shared.k4.gate.su_block".into(), dtype: "F16".into(), shape: vec![1, nblk], bytes: to_bytes(&sg_su) },
            TEnt { name: "shared.k4.up.svh".into(), dtype: "F16".into(), shape: vec![1, i_exp], bytes: to_bytes(&su_svh) },
            TEnt { name: "shared.k4.up.su_block".into(), dtype: "F16".into(), shape: vec![1, nblk], bytes: to_bytes(&su_su) },
            TEnt { name: "shared.k4.down.suh".into(), dtype: "F16".into(), shape: vec![1, i_exp], bytes: to_bytes(&sd_suh) },
            TEnt { name: "shared.k4.down.sv_block".into(), dtype: "F16".into(), shape: vec![1, nblk], bytes: to_bytes(&sd_sv) },
        ],
        &[
            ("format", "exlr8-v2"),
            ("hidden_size", "256"),
            ("moe_intermediate_size", "128"),
            ("n_routed_experts", "2"),
            ("widths", "4"),
            ("shared_widths", "4"),
        ],
    );

    fs::create_dir_all(dir.join("manifests")).unwrap();
    fs::write(
        dir.join("manifests/exlr8-manifest-bit400.json"),
        r#"{"format":"exlr8-v2-manifest","layers":{"0":{"gate":[[4],[4]],"up":[[4],[4]],"down":[4,4],"shared":{"gate":[4],"up":[4],"down":4}}}}"#,
    )
    .unwrap();

    let rel = TtsRelease::open(dir.to_str().unwrap(), Some("bit400"), true).unwrap();
    assert_eq!(rel.expert_store(0), 1);

    // Check routed expert 1 gate_proj against independent Hadamard reconstruction
    let mut got_gate = vec![0u8; i_exp * h * 2];
    rel.expert_decode_f16(0, 1, 0, &mut got_gate).unwrap();
    let t_g = tiles_out_major_cb(&routed_gate[hb..2 * hb], k, h, Codebook::Pmcgx);
    let suh_g: Vec<f32> = (0..h).map(|c| signs_in[c] as f32 * f16_to_f32(g_su[nblk + c / 128])).collect();
    let svh_g: Vec<f32> = g_svh[i_exp..2 * i_exp].iter().map(|&x| f16_to_f32(x)).collect();
    let w_g = ref_reconstruct(&t_g, h, 128, &suh_g, &svh_g);
    for r in 0..i_exp {
        for c in 0..h {
            let want = f16_bits(w_g[c * 128 + r]);
            let got = u16::from_le_bytes([got_gate[2 * (r * h + c)], got_gate[2 * (r * h + c) + 1]]);
            assert_eq!(got, want, "trellis gate mismatch at ({r},{c})");
        }
    }

    // Check routed expert 0 down_proj against independent Hadamard reconstruction
    let mut got_down = vec![0u8; h * i_exp * 2];
    rel.expert_decode_f16(0, 0, 2, &mut got_down).unwrap();
    let t_d = tiles_in_major_cb(&routed_down[0..hb], k, h, Codebook::Pmcgx);
    let suh_d: Vec<f32> = d_suh[0..i_exp].iter().map(|&x| f16_to_f32(x)).collect();
    let svh_d: Vec<f32> = (0..h).map(|c| signs_out[c] as f32 * f16_to_f32(d_sv[c / 128])).collect();
    let w_d = ref_reconstruct(&t_d, 128, h, &suh_d, &svh_d);
    for r in 0..h {
        for c in 0..i_exp {
            let want = f16_bits(w_d[c * h + r]);
            let got = u16::from_le_bytes([got_down[2 * (r * i_exp + c)], got_down[2 * (r * i_exp + c) + 1]]);
            assert_eq!(got, want, "trellis down mismatch at ({r},{c})");
        }
    }

    // Check shared expert (expert == 2) down_proj
    let mut got_sd = vec![0u8; h * i_exp * 2];
    rel.expert_decode_f16(0, 2, 2, &mut got_sd).unwrap();
    let t_sd = tiles_in_major_cb(&sh_down, k, h, Codebook::Pmcgx);
    let suh_sd: Vec<f32> = sd_suh.iter().map(|&x| f16_to_f32(x)).collect();
    let svh_sd: Vec<f32> = (0..h).map(|c| signs_out[c] as f32 * f16_to_f32(sd_sv[c / 128])).collect();
    let w_sd = ref_reconstruct(&t_sd, 128, h, &suh_sd, &svh_sd);
    for r in 0..h {
        for c in 0..i_exp {
            let want = f16_bits(w_sd[c * h + r]);
            let got = u16::from_le_bytes([got_sd[2 * (r * i_exp + c)], got_sd[2 * (r * i_exp + c) + 1]]);
            assert_eq!(got, want, "trellis shared down mismatch at ({r},{c})");
        }
    }

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn tts_null_args_fail_safely() {
    let mut err = [0 as c_char; 128];
    unsafe {
        let h = g6load::g6l_tts_open(std::ptr::null(), std::ptr::null(), 1, err.as_mut_ptr(), err.len());
        assert!(h.is_null());
        assert!(g6load::g6l_tts_profile(std::ptr::null()).is_null());
        assert_eq!(g6load::g6l_tts_dense_item_count(std::ptr::null()), 0);
        assert_eq!(g6load::g6l_tts_expert_store(std::ptr::null(), 0), -1);
        assert_eq!(
            g6load::g6l_tts_expert_decode_f16(
                std::ptr::null(),
                0,
                0,
                0,
                std::ptr::null_mut(),
                0,
                err.as_mut_ptr() as *mut c_char,
                err.len()
            ),
            1
        );
        g6load::g6l_tts_close(std::ptr::null_mut());
    }
}
