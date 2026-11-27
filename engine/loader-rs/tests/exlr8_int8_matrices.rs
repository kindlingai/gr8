//! Specification parity and refusal tests for exlr8 v4 `matrices` metadata and `int8 gG` dense groups
//! (kindlingai/exlr8 FORMAT.md PRs #13 & #14).

use g6load::exlr8::{self, Matrix, StFile, BF16, FP32, FP8_B32, FP8_BLOCK, FP8_ROW, INT4, INT8, MXFP8, ROWS, SIDE_IN, SIDE_OUT, SIDE_REPLICATED, TRELLIS};
use g6load::records;
use g6load::xxh64::xxh64;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

struct TensorSpec {
    dtype: &'static str,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn temp_st_path(tag: &str) -> PathBuf {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let id = SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("gr8-exlr8-int8-{tag}-{}-{id}.safetensors", std::process::id()))
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn write_st(path: &PathBuf, tensors: &BTreeMap<String, TensorSpec>, meta: &BTreeMap<String, String>) {
    let mut head = String::from("{");
    let mut first = true;
    if !meta.is_empty() {
        head.push_str("\"__metadata__\":{");
        for (i, (k, v)) in meta.iter().enumerate() {
            if i > 0 { head.push(','); }
            head.push_str(&format!("{}:{}", json_escape(k), json_escape(v)));
        }
        head.push('}');
        first = false;
    }
    let mut off = 0usize;
    let mut blobs: Vec<&[u8]> = Vec::new();
    for (name, spec) in tensors {
        if !first { head.push(','); }
        first = false;
        let shape: Vec<String> = spec.shape.iter().map(usize::to_string).collect();
        let end = off + spec.bytes.len();
        head.push_str(&format!(
            "{}:{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            json_escape(name), spec.dtype, shape.join(","), off, end
        ));
        off = end;
        blobs.push(&spec.bytes);
    }
    head.push('}');
    while head.len() % 8 != 0 {
        head.push(' ');
    }
    let mut data = Vec::with_capacity(8 + head.len() + off);
    data.extend_from_slice(&(head.len() as u64).to_le_bytes());
    data.extend_from_slice(head.as_bytes());
    for b in blobs {
        data.extend_from_slice(b);
    }
    std::fs::write(path, data).unwrap();
}

fn i64_tensor(vals: &[i64], shape: Vec<usize>) -> TensorSpec {
    let bytes = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    TensorSpec { dtype: "I64", shape, bytes }
}

fn u64_tensor(vals: &[u64], shape: Vec<usize>) -> TensorSpec {
    let bytes = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    TensorSpec { dtype: "U64", shape, bytes }
}

fn f16_tensor(vals: &[u16], shape: Vec<usize>) -> TensorSpec {
    let bytes = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    TensorSpec { dtype: "F16", shape, bytes }
}

fn i8_tensor(vals: &[i8], shape: Vec<usize>) -> TensorSpec {
    let bytes = vals.iter().map(|&v| v as u8).collect();
    TensorSpec { dtype: "I8", shape, bytes }
}

fn make_int8_tensors(
    base: &str,
    side: i64,
    g: usize,
    out: usize,
    inp: usize,
    seed: u64,
) -> (BTreeMap<String, TensorSpec>, Vec<i8>, Vec<u16>, Vec<f32>) {
    let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut next = || -> u64 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        rng >> 32
    };
    let mut q_hf = vec![0i8; out * inp];
    for q in &mut q_hf {
        *q = ((next() % 255) as i32 - 127) as i8; // [-127, 127], never -128
    }
    let ng = inp / g;
    let mut s_hf = vec![0u16; out * ng];
    for s in &mut s_hf {
        let v = ((next() % 1000) as f32 + 1.0) / 1000.0;
        *s = records::f16_bits(v);
    }
    s_hf[0] = 0; // zero scale decodes to zeros regardless of q8

    let mut want = vec![0f32; out * inp];
    for r in 0..out {
        for c in 0..inp {
            want[r * inp + c] = q_hf[r * inp + c] as f32 * exlr8::f16_to_f32(s_hf[r * ng + c / g]);
        }
    }

    let (q8_spec, s8_spec, sums) = if side == SIDE_IN {
        let nb = inp / 128;
        let spg = 128 / g;
        let mut q8_in = vec![0i8; nb * out * 128];
        let mut s8_in = vec![0u16; nb * out * spg];
        for b in 0..nb {
            for r in 0..out {
                for c in 0..128 {
                    q8_in[(b * out + r) * 128 + c] = q_hf[r * inp + b * 128 + c];
                }
                for sg in 0..spg {
                    s8_in[(b * out + r) * spg + sg] = s_hf[r * ng + b * spg + sg];
                }
            }
        }
        let q_spec = i8_tensor(&q8_in, vec![nb, out, 128]);
        let s_spec = f16_tensor(&s8_in, vec![nb, out, spg]);
        let mut sums = Vec::with_capacity(nb);
        for b in 0..nb {
            let qb = &q_spec.bytes[b * out * 128..(b + 1) * out * 128];
            let sb = &s_spec.bytes[b * out * spg * 2..(b + 1) * out * spg * 2];
            sums.push(xxh64(&[qb, sb], 0));
        }
        (q_spec, s_spec, sums)
    } else {
        let q_spec = i8_tensor(&q_hf, vec![out, inp]);
        let s_spec = f16_tensor(&s_hf, vec![out, ng]);
        let nb = out.div_ceil(128);
        let mut sums = Vec::with_capacity(nb);
        for b in 0..nb {
            let r0 = 128 * b;
            let r1 = (r0 + 128).min(out);
            let qb = &q_spec.bytes[r0 * inp..r1 * inp];
            let sb = &s_spec.bytes[r0 * ng * 2..r1 * ng * 2];
            sums.push(xxh64(&[qb, sb], 0));
        }
        (q_spec, s_spec, sums)
    };

    let mut t = BTreeMap::new();
    let unit = if side == SIDE_REPLICATED { 0 } else { 128 };
    t.insert(format!("{base}.record"), i64_tensor(&[0, out as i64, inp as i64, 0, side, unit, 0, 1], vec![8]));
    t.insert(format!("{base}.checksum"), u64_tensor(&sums, vec![sums.len()]));
    t.insert(format!("{base}.q8"), q8_spec);
    t.insert(format!("{base}.s8"), s8_spec);
    (t, q_hf, s_hf, want)
}

fn int8_desc_json(side: i64, g: usize, solve: &str) -> String {
    match side {
        SIDE_REPLICATED => format!(r#"{{"format":"int8 g{g}","side":"replicated","solve":"{solve}"}}"#),
        SIDE_OUT => format!(r#"{{"format":"int8 g{g}","side":"out","unit":128,"solve":"{solve}"}}"#),
        SIDE_IN => format!(r#"{{"format":"int8 g{g}","side":"in","unit":128,"solve":"{solve}"}}"#),
        _ => unreachable!(),
    }
}

#[test]
fn parse_format_valid_and_refused() {
    let ok = [
        ("trellis k4", (TRELLIS, 4)),
        ("trellis k4p5", (TRELLIS, -9)),
        ("row-trellis k2", (ROWS, 2)),
        ("row-trellis k4p5", (ROWS, -9)),
        ("int8 g16", (INT8, 16)),
        ("int8 g32", (INT8, 32)),
        ("int8 g64", (INT8, 64)),
        ("int8 g128", (INT8, 128)),
        ("int4 g32", (INT4, 0)),
        ("mxfp8", (MXFP8, 0)),
        ("fp8 b128", (FP8_BLOCK, 0)),
        ("fp8 b32", (FP8_B32, 0)),
        ("fp8 row", (FP8_ROW, 0)),
        ("bf16", (BF16, 0)),
        ("fp32", (FP32, 0)),
    ];
    for (s, want) in ok {
        assert_eq!(exlr8::parse_format(s).unwrap(), want, "format {s}");
    }
    for bad in ["int8 g48", "int8 g256", "int8 g8", "trellis", "trellis k", "MXFP8", "fp8 block", "int4 g64", ""] {
        assert!(exlr8::parse_format(bad).is_err(), "expected refusal for {bad:?}");
    }
}

#[test]
fn int8_all_groups_and_sides_decode_checksum_and_slices() {
    for side in [SIDE_REPLICATED, SIDE_OUT, SIDE_IN] {
        for g in [16usize, 32, 64, 128] {
            let (out, inp) = (256, 384);
            let (t, q_want, s_want, want) = make_int8_tensors("o_proj", side, g, out, inp, (g as u64) * 17 + side as u64);
            let mut meta = BTreeMap::new();
            meta.insert("format".into(), "exlr8-v4".into());
            meta.insert("layer".into(), "0".into());
            meta.insert("matrices".into(), format!(r#"{{"o_proj":{}}}"#, int8_desc_json(side, g, "hessian")));
            let path = temp_st_path(&format!("ok-s{side}-g{g}"));
            write_st(&path, &t, &meta);

            let f = StFile::open(path.to_str().unwrap()).unwrap();
            let m = Matrix::open(&f, "o_proj").unwrap();
            assert_eq!(m.rec.fmt, INT8);
            assert_eq!(m.rec.k, g);
            assert_eq!(m.rec.side, side);
            m.verify_checksums().unwrap();

            let (q_got, s_got) = m.int8(&(0..out), &(0..inp)).unwrap();
            assert_eq!(q_got, q_want);
            assert_eq!(s_got, s_want);

            let full = m.decode(0..out, 0..inp).unwrap();
            assert!(full.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "side={side} g={g} decode mismatch");
            assert!(full[..g].iter().all(|&v| v == 0.0), "zero scale must decode to zeros");

            let par = exlr8::decode_par(&m, 0..out, 0..inp, 4).unwrap();
            assert!(par.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "side={side} g={g} decode_par mismatch");

            // Sub-block decode parity
            let (rb, cb) = if side == SIDE_IN { (7..55, 128..256) } else { (7..55, 5..141) };
            let sub = m.decode(rb.clone(), cb.clone()).unwrap();
            let mut sub_want = Vec::new();
            for r in rb.clone() {
                sub_want.extend_from_slice(&want[r * inp + cb.start..r * inp + cb.end]);
            }
            assert!(sub.iter().zip(&sub_want).all(|(a, b)| a.to_bits() == b.to_bits()), "side={side} g={g} sub-block mismatch");

            let _ = std::fs::remove_file(&path);
        }
    }
}

#[test]
fn int8_refusals_code_scale_nan_inf_and_checksum() {
    for case in ["code_m128", "scale_neg", "scale_nan", "scale_inf", "bad_checksum"] {
        let (mut t, _, _, _) = make_int8_tensors("m", SIDE_OUT, 64, 128, 256, 42);
        match case {
            "code_m128" => { t.get_mut("m.q8").unwrap().bytes[13] = 0x80; } // -128i8
            "scale_neg" => {
                let neg = 0xBC00u16.to_le_bytes(); // -1.0 fp16
                t.get_mut("m.s8").unwrap().bytes[2..4].copy_from_slice(&neg);
            }
            "scale_nan" => {
                let nan = 0x7E00u16.to_le_bytes(); // NaN fp16
                t.get_mut("m.s8").unwrap().bytes[2..4].copy_from_slice(&nan);
            }
            "scale_inf" => {
                let inf = 0x7C00u16.to_le_bytes(); // +Inf fp16
                t.get_mut("m.s8").unwrap().bytes[2..4].copy_from_slice(&inf);
            }
            "bad_checksum" => {
                t.get_mut("m.checksum").unwrap().bytes[0] ^= 1;
            }
            _ => unreachable!(),
        }
        let mut meta = BTreeMap::new();
        meta.insert("format".into(), "exlr8-v4".into());
        meta.insert("matrices".into(), format!(r#"{{"m":{}}}"#, int8_desc_json(SIDE_OUT, 64, "hessian")));
        let path = temp_st_path(case);
        write_st(&path, &t, &meta);
        let f = StFile::open(path.to_str().unwrap()).unwrap();
        let m = Matrix::open(&f, "m").unwrap();
        if case == "bad_checksum" {
            assert!(m.verify_checksums().is_err(), "expected checksum failure");
        } else {
            assert!(m.int8(&(0..128), &(0..256)).is_err(), "{case}: int8() must refuse");
            assert!(m.decode(0..128, 0..256).is_err(), "{case}: decode() must refuse");
        }
        let _ = std::fs::remove_file(&path);
    }
}

#[test]
fn matrices_metadata_refusal_rules() {
    let cases = [
        "missing", "unknown_fmt", "extra", "legacy_code", "legacy_side", "legacy_unit",
        "legacy_solve", "v2_format", "no_matrices", "unknown_key", "bad_side", "bad_solve",
        "no_unit", "replicated_unit", "bool_unit", "zero_unit", "rotation_on_int8",
    ];
    for case in cases {
        let (mut t, _, _, _) = make_int8_tensors("o_proj", SIDE_OUT, 64, 128, 256, 7);
        let mut meta = BTreeMap::new();
        meta.insert("format".into(), "exlr8-v4".into());
        let mut matrices = Some(format!(r#"{{"o_proj":{}}}"#, int8_desc_json(SIDE_OUT, 64, "hessian")));
        match case {
            "missing" => matrices = Some("{}".into()),
            "unknown_fmt" => matrices = Some(r#"{"o_proj":{"format":"int8 g48","side":"out","unit":128,"solve":"hessian"}}"#.into()),
            "extra" => matrices = Some(r#"{"o_proj":{"format":"int8 g64","side":"out","unit":128,"solve":"hessian"},"q_proj":{"format":"bf16","side":"replicated","solve":"copy"}}"#.into()),
            "legacy_code" => { t.insert("o_proj.record".into(), i64_tensor(&[INT4, 128, 256, 0, 1, 128, 0, 1], vec![8])); }
            "legacy_side" => { t.insert("o_proj.record".into(), i64_tensor(&[0, 128, 256, 0, 2, 128, 0, 1], vec![8])); }
            "legacy_unit" => { t.insert("o_proj.record".into(), i64_tensor(&[0, 128, 256, 0, 1, 64, 0, 1], vec![8])); }
            "legacy_solve" => matrices = Some(r#"{"o_proj":{"format":"int8 g64","side":"out","unit":128,"solve":"rtn"}}"#.into()),
            "v2_format" => { meta.insert("format".into(), "exlr8-v2".into()); }
            "no_matrices" => { matrices = None; }
            "unknown_key" => matrices = Some(r#"{"o_proj":{"format":"int8 g64","side":"out","unit":128,"solve":"hessian","group":64}}"#.into()),
            "bad_side" => matrices = Some(r#"{"o_proj":{"format":"int8 g64","side":"output","unit":128,"solve":"hessian"}}"#.into()),
            "bad_solve" => matrices = Some(r#"{"o_proj":{"format":"int8 g64","side":"out","unit":128,"solve":"gptq"}}"#.into()),
            "no_unit" => matrices = Some(r#"{"o_proj":{"format":"int8 g64","side":"out","solve":"hessian"}}"#.into()),
            "replicated_unit" => {
                t.insert("o_proj.record".into(), i64_tensor(&[0, 128, 256, 0, 0, 0, 0, 1], vec![8]));
                matrices = Some(r#"{"o_proj":{"format":"int8 g64","side":"replicated","unit":128,"solve":"hessian"}}"#.into());
            }
            "bool_unit" => matrices = Some(r#"{"o_proj":{"format":"int8 g64","side":"out","unit":true,"solve":"hessian"}}"#.into()),
            "zero_unit" => matrices = Some(r#"{"o_proj":{"format":"int8 g64","side":"out","unit":0,"solve":"hessian"}}"#.into()),
            "rotation_on_int8" => matrices = Some(r#"{"o_proj":{"format":"int8 g64","side":"out","unit":128,"rotation":"attn_in","solve":"hessian"}}"#.into()),
            _ => unreachable!(),
        }
        if let Some(m) = matrices {
            meta.insert("matrices".into(), m);
        }
        let path = temp_st_path(case);
        write_st(&path, &t, &meta);
        let f = StFile::open(path.to_str().unwrap()).unwrap();
        assert!(Matrix::open(&f, "o_proj").is_err(), "case {case} must be refused");
        let _ = std::fs::remove_file(&path);
    }
}

#[test]
fn matrices_rotation_names_and_zero_legacy_fields() {
    // Trellis matrix with legacy field 6 = 2 matches default "q_resid" or custom rotations "x, y" -> "y"
    let (out, inp, k) = (128usize, 128usize, 2usize);
    let mut base_t = BTreeMap::new();
    base_t.insert("q_proj.record".into(), i64_tensor(&[TRELLIS, out as i64, inp as i64, k as i64, 1, 128, 2, 1], vec![8]));
    base_t.insert("q_proj.trellis".into(), TensorSpec { dtype: "U8", shape: vec![1, 16 * k * inp], bytes: vec![7u8; 16 * k * inp] });
    base_t.insert("q_proj.suh".into(), f16_tensor(&vec![0x3c00u16; inp], vec![inp]));
    base_t.insert("q_proj.svh".into(), f16_tensor(&vec![0x3c00u16; out], vec![out]));

    for (rotations, name, ok) in [
        (None, "q_resid", true),
        (None, "attn_in", false),
        (Some("x, y"), "y", true),
        (Some("x, y"), "q_resid", false),
    ] {
        let mut meta = BTreeMap::new();
        meta.insert("format".into(), "exlr8-v4".into());
        if let Some(r) = rotations {
            meta.insert("rotations".into(), r.into());
        }
        meta.insert(
            "matrices".into(),
            format!(r#"{{"q_proj":{{"format":"trellis k2","side":"out","unit":128,"rotation":"{name}","solve":"hessian"}}}}"#),
        );
        let path = temp_st_path(&format!("rot-{name}-{}", rotations.is_some()));
        write_st(&path, &base_t, &meta);
        let f = StFile::open(path.to_str().unwrap()).unwrap();
        let res = Matrix::open(&f, "q_proj");
        assert_eq!(res.is_ok(), ok, "rotations={rotations:?} name={name}");
        let _ = std::fs::remove_file(&path);
    }

    // Zero legacy fields with `matrices` decodes identically to legacy-only file
    let p_leg = temp_st_path("legacy-eq");
    let mut meta_leg = BTreeMap::new();
    meta_leg.insert("format".into(), "exlr8-v4".into());
    write_st(&p_leg, &base_t, &meta_leg);
    let f_leg = StFile::open(p_leg.to_str().unwrap()).unwrap();
    let want = Matrix::open(&f_leg, "q_proj").unwrap().decode(0..out, 0..inp).unwrap();

    let mut zero_t = base_t;
    zero_t.insert("q_proj.record".into(), i64_tensor(&[0, out as i64, inp as i64, 0, 0, 0, 0, 0], vec![8]));
    let mut meta_v4 = BTreeMap::new();
    meta_v4.insert("format".into(), "exlr8-v4".into());
    meta_v4.insert(
        "matrices".into(),
        r#"{"q_proj":{"format":"trellis k2","side":"out","unit":128,"rotation":"q_resid","solve":"hessian"}}"#.into(),
    );
    let p_v4 = temp_st_path("v4-eq");
    write_st(&p_v4, &zero_t, &meta_v4);
    let f_v4 = StFile::open(p_v4.to_str().unwrap()).unwrap();
    let m_v4 = Matrix::open(&f_v4, "q_proj").unwrap();
    assert_eq!(m_v4.rec.rotation, 2);
    let got = m_v4.decode(0..out, 0..inp).unwrap();
    assert!(got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()));

    let _ = std::fs::remove_file(&p_leg);
    let _ = std::fs::remove_file(&p_v4);
}
