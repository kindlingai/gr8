//! NVFP4 primitives (src/nvfp4.rs): grids, the two scale rules, the nv4g record, and byte equality with the C++ twin
//! engine/kernels/nvfp4_pack.h.

use g6load::exlr8::e4m3_to_f32;
use g6load::json;
use g6load::nvfp4::*;
use g6load::records;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
    /// Roughly bell-shaped values in about (-3, 3) with rare large outliers.
    fn floats(&mut self, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| {
                let u = |r: &mut Rng| (r.next() & 0xffff) as f32 / 65536.0;
                let v = (u(self) + u(self) + u(self) + u(self) - 2.0) * 1.5;
                if self.next().is_multiple_of(97) { v * 9.0 } else { v }
            })
            .collect()
    }
}

#[test]
fn e2m1_grid_ties_and_saturation() {
    let grid = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    for code in 0u8..16 {
        let v = e2m1_to_f32(code);
        assert_eq!(v.abs(), grid[(code & 7) as usize]);
        assert_eq!(v.is_sign_negative(), code & 8 != 0);
        // every grid value maps back to its code; both zeros map to code 0
        assert_eq!(f32_to_e2m1_rne(v), if code == 8 { 0 } else { code });
    }
    // ties go to the even grid index
    for (x, want) in [(0.25f32, 0.0f32), (0.75, 1.0), (1.25, 1.0), (1.75, 2.0), (2.5, 2.0), (3.5, 4.0), (5.0, 4.0)] {
        assert_eq!(e2m1_to_f32(f32_to_e2m1_rne(x)), want, "{x}");
        assert_eq!(e2m1_to_f32(f32_to_e2m1_rne(-x)), -want, "-{x}");
    }
    for (x, want) in [(0.26f32, 0.5f32), (2.51, 3.0), (5.01, 6.0), (4.99, 4.0), (7.0, 6.0), (1e30, 6.0), (f32::INFINITY, 6.0)] {
        assert_eq!(e2m1_to_f32(f32_to_e2m1_rne(x)), want, "{x}");
        assert_eq!(f32_to_e2m1_rne(-x), 8 | f32_to_e2m1_rne(x), "-{x}");
    }
    assert_eq!(f32_to_e2m1_rne(-0.0), 0);
    assert_eq!(f32_to_e2m1_rne(-0.1), 0);
}

#[test]
fn e4m3_round_trip_ties_and_saturation() {
    let mut finite = 0;
    for b in 0u8..=255 {
        let v = e4m3_to_f32(b);
        if v.is_nan() {
            assert!(b == 0x7f || b == 0xff);
            continue;
        }
        finite += 1;
        assert_eq!(f32_to_e4m3_rne(v), if b == 0x80 { 0 } else { b }, "byte {b:#x} value {v}");
    }
    assert_eq!(finite, 254);
    assert_eq!(f32_to_e4m3_rne(f32::NAN), 0x7f);
    assert_eq!(e4m3_to_f32(f32_to_e4m3_rne(1e9)), 448.0);
    assert_eq!(e4m3_to_f32(f32_to_e4m3_rne(-1e9)), -448.0);
    assert_eq!(e4m3_to_f32(f32_to_e4m3_rne(f32::INFINITY)), 448.0);
    assert_eq!(f32_to_e4m3_rne(-0.0), 0);
    assert_eq!(f32_to_e4m3_rne(-1e-9), 0);
    // halfway between neighbours: the even mantissa wins
    for b in 0u8..0x7d {
        let (lo, hi) = (e4m3_to_f32(b), e4m3_to_f32(b + 1));
        let mid = ((lo as f64 + hi as f64) / 2.0) as f32;
        let want = if b & 1 == 0 { b } else { b + 1 };
        assert_eq!(f32_to_e4m3_rne(mid), want, "tie above byte {b:#x}");
    }
}

#[test]
fn outer_scale_conventions_and_rule_config() {
    let d = Nvfp4OuterScale::Direct(vec![0.5, 0.25]);
    assert_eq!((d.at(0), d.at(1)), (0.5, 0.25));
    assert_eq!(Nvfp4OuterScale::Direct(vec![0.5]).at(7), 0.5);
    let a = Nvfp4OuterScale::AmaxDenominator { amax: vec![3.0, 6.0], denominator: 1536.0 };
    assert_eq!((a.at(0), a.at(1)), (3.0 / 1536.0, 6.0 / 1536.0));
    assert_eq!(Nvfp4OuterScale::AmaxDenominator { amax: vec![2688.0], denominator: 2688.0 }.at(5), 1.0);

    let rule = |t: &str| Nvfp4RuleConfig::parse(&json::parse(t).unwrap()).map(|c| c.rule);
    assert_eq!(rule(r#"{"compute_dtype": "nvfp4", "required_activation_quantization": {"scale_selection_rule": "fouroversix_mae"}}"#), Ok(Nvfp4ScaleRule::FourOverSixMae));
    assert_eq!(rule(r#"{"scale_selection_rule": "static_6"}"#), Ok(Nvfp4ScaleRule::Static6));
    assert_eq!(rule(r#"{"compute_dtype": "nvfp4"}"#), Ok(Nvfp4ScaleRule::Static6));
    assert_eq!(rule(r#"{"quant_method": "modelopt", "quant_algo": "NVFP4"}"#), Ok(Nvfp4ScaleRule::Static6));
    assert_eq!(rule(r#"{"quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized"}"#), Ok(Nvfp4ScaleRule::Static6));
    assert!(rule(r#"{"quant_method": "compressed-tensors", "format": "int-quantized"}"#).is_err());
    assert!(rule(r#"{"quant_method": "other"}"#).is_err());
    assert!(rule(r#"{"scale_selection_rule": "something_else"}"#).is_err());
    assert_eq!(Nvfp4ScaleRule::Static6.outer_denominator(), 6.0 * 448.0);
    assert_eq!(Nvfp4ScaleRule::FourOverSixMae.outer_denominator(), 6.0 * 256.0);
}

fn l1(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs() as f64).sum()
}

#[test]
fn four_over_six_beats_static_six_on_spiked_blocks() {
    // Block 0: one spike at 6 among mid-range values. With amax -> 6 the step near 2.5 is 1; with amax -> 4 the spike
    // is still exact (4 * 1.5) and the mid-range values land closer. Block 1: on-grid for amax -> 6. Block 2: zero.
    let mut x = vec![0f32; 64];
    x[..16].fill(2.5);
    x[3] = -6.0;
    x[16..32].copy_from_slice(&[6.0, 3.0, -2.0, 1.5, 1.0, 0.5, 0.0, -4.0, 6.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.0, 4.0]);
    x[48..].fill(0.001);
    let fixed = Some(1.0 / 64.0);
    let (c6, s6, o6) = nvfp4_quantize(&x, 64, Nvfp4ScaleRule::Static6, fixed);
    let (c4, s4, o4) = nvfp4_quantize(&x, 64, Nvfp4ScaleRule::FourOverSixMae, fixed);
    assert_eq!((o6, o4), (1.0 / 64.0, 1.0 / 64.0));
    let (d6, d4) = (nvfp4_dequantize(&c6, &s6, o6), nvfp4_dequantize(&c4, &s4, o4));
    assert_eq!(e4m3_to_f32(s6[0]), 64.0);
    assert_eq!(e4m3_to_f32(s4[0]), 96.0, "the spiked block takes the amax / 4 scale");
    assert!(l1(&d4[..16], &x[..16]) < l1(&d6[..16], &x[..16]));
    assert_eq!(d4[3], -6.0);
    assert_eq!(s4[1], s6[1], "an exact amax / 6 block keeps it (ties never move to amax / 4)");
    assert_eq!(&d4[16..32], &x[16..32]);
    assert_eq!((s4[2], s6[2]), (0, 0));
    assert!(c4[16..24].iter().all(|&b| b == 0) && d4[32..48].iter().all(|&v| v == 0.0));

    // Random rows with their own outer scale: per block the chosen candidate is never worse, and no scale overflows.
    let mut rng = Rng(7);
    let (n, k) = (48usize, 256usize);
    let w = rng.floats(n * k);
    let (mut picked4, mut total) = (0usize, 0usize);
    for row in w.chunks(k) {
        let (c, s, o) = nvfp4_quantize(row, k, Nvfp4ScaleRule::FourOverSixMae, None);
        let amax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert_eq!(o, amax / 1536.0);
        let (cs, ss, os) = nvfp4_quantize(row, k, Nvfp4ScaleRule::Static6, Some(o));
        assert_eq!(os, o);
        let (d, ds) = (nvfp4_dequantize(&c, &s, o), nvfp4_dequantize(&cs, &ss, os));
        for b in 0..k / 16 {
            let r = b * 16..b * 16 + 16;
            assert!(l1(&d[r.clone()], &row[r.clone()]) <= l1(&ds[r.clone()], &row[r.clone()]));
            assert!(e4m3_to_f32(s[b]) <= 384.0 && e4m3_to_f32(ss[b]) <= 256.0, "scale inside E4M3");
            picked4 += (s[b] != ss[b]) as usize;
            total += 1;
        }
        // the default Static6 outer scale puts the largest block scale at 448 exactly
        let (_, s0, o0) = nvfp4_quantize(row, k, Nvfp4ScaleRule::Static6, None);
        assert_eq!(o0, amax / 2688.0);
        assert_eq!(s0.iter().map(|&b| e4m3_to_f32(b)).fold(0f32, f32::max), 448.0);
    }
    assert!(picked4 > total / 20 && picked4 < total, "amax / 4 chosen on {picked4} of {total} blocks");

    // all-zero input: outer 0, every byte 0
    let (c, s, o) = nvfp4_quantize(&[0f32; 32], 16, Nvfp4ScaleRule::FourOverSixMae, None);
    assert!(o == 0.0 && c.iter().all(|&b| b == 0) && s.iter().all(|&b| b == 0));
    let (c, s, _) = nvfp4_quantize(&[0f32; 32], 16, Nvfp4ScaleRule::Static6, Some(0.0));
    assert!(c.iter().all(|&b| b == 0) && s.iter().all(|&b| b == 0));
}

fn same_bits(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits()
}

/// Every (row, k) of the records of [n][kn] (columns c0.. of the row-major reference `deq` [n][kfull]).
fn check_records(rec: &[u8], stride: usize, at: usize, deq: &[f32], n: usize, kfull: usize, c0: usize, kn: usize, outer: f32) {
    let ng = kn / NV4_KG;
    assert_eq!(rec.len(), n / 16 * ng * stride);
    for t in 0..n / 16 {
        for g in 0..ng {
            let r: &[u8; NV4_GR] = rec[(t * ng + g) * stride + at..][..NV4_GR].try_into().unwrap();
            let d = nvfp4_decode_group(r, outer);
            for (i, row) in d.iter().enumerate() {
                for (j, &v) in row.iter().enumerate() {
                    let want = deq[(t * 16 + i) * kfull + c0 + g * 64 + j];
                    assert!(same_bits(v, want), "row {} k {}: {v} != {want}", t * 16 + i, c0 + g * 64 + j);
                }
            }
        }
    }
}

#[test]
fn records_round_trip_against_row_major_dequantization() {
    let mut rng = Rng(99);
    for (n, k, rule) in [
        (16usize, 64usize, Nvfp4ScaleRule::Static6),
        (64, 128, Nvfp4ScaleRule::FourOverSixMae),
        (384, 512, Nvfp4ScaleRule::Static6),
        (384, 512, Nvfp4ScaleRule::FourOverSixMae),
    ] {
        let (g, u) = (rng.floats(n * k), rng.floats(n * k));
        let (gc, gs, go) = nvfp4_quantize(&g, k, rule, None);
        let (uc, us, uo) = nvfp4_quantize(&u, k, rule, None);
        let (gd, ud) = (nvfp4_dequantize(&gc, &gs, go), nvfp4_dequantize(&uc, &us, uo));
        assert!(l1(&gd, &g) / (g.iter().map(|v| v.abs() as f64).sum::<f64>()) < 0.2, "quantization is close");
        let rec = nvfp4_record(&gc, &gs, k);
        check_records(&rec, NV4_GR, 0, &gd, n, k, 0, k, go);
        assert_eq!(rec, records::nvfp4_record(&gc, &gs, k));
        let fc1 = nvfp4_fc1(&gc, &gs, &uc, &us, n, k);
        check_records(&fc1, 2 * NV4_GR, 0, &gd, n, k, 0, k, go);
        check_records(&fc1, 2 * NV4_GR, NV4_GR, &ud, n, k, 0, k, uo);
        if k >= 128 {
            let (c0, kn) = (64, k - 64);
            check_records(&nvfp4_fc2(&gc, &gs, n, k, c0, kn), NV4_GR, 0, &gd, n, k, c0, kn, go);
        }
    }
    // the full-width expert shape: a 384-wide column slice of [384][4608]
    let (n, k, c0, kn) = (384usize, 4608usize, 1152usize, 384usize);
    let w = rng.floats(n * k);
    let (c, s, o) = nvfp4_quantize(&w, k, Nvfp4ScaleRule::FourOverSixMae, None);
    let d = nvfp4_dequantize(&c, &s, o);
    check_records(&nvfp4_fc2(&c, &s, n, k, c0, kn), NV4_GR, 0, &d, n, k, c0, kn, o);
    check_records(&nvfp4_fc2(&c, &s, n, k, k - 64, 64), NV4_GR, 0, &d, n, k, k - 64, 64, o);

    // the isolated scale helper inverts
    let sc = rng.bytes(32 * 12);
    let mut packed = [0u8; 64];
    pack_nv4_scales_64(&mut packed, &sc, 12, 16, 4);
    let un = unpack_nv4_scales_64(&packed);
    for (r, row) in un.iter().enumerate() {
        assert_eq!(row[..], sc[(16 + r) * 12 + 4..][..4]);
    }
}

#[test]
fn mxfp8_fc1_fc2_share_the_record() {
    let mut rng = Rng(5);
    let (r, k, c0, kn) = (48usize, 192usize, 64usize, 96usize);
    let (gc, gs, uc, us) = (rng.bytes(r * k), rng.bytes(r * k / 32), rng.bytes(r * k), rng.bytes(r * k / 32));
    let fc1 = records::mxfp8_fc1(&gc, &gs, &uc, &us, r, k);
    let (g, u) = (records::mxfp8_record(&gc, &gs, k), records::mxfp8_record(&uc, &us, k));
    assert_eq!(fc1.len(), g.len() + u.len());
    for (i, pair) in fc1.chunks(1056).enumerate() {
        assert_eq!(pair[..528], g[i * 528..][..528]);
        assert_eq!(pair[528..], u[i * 528..][..528]);
    }
    // a column slice equals the record of the copied slice
    let (mut sc, mut ss) = (Vec::new(), Vec::new());
    for row in 0..r {
        sc.extend_from_slice(&gc[row * k + c0..][..kn]);
        ss.extend_from_slice(&gs[row * (k / 32) + c0 / 32..][..kn / 32]);
    }
    assert_eq!(records::mxfp8_fc2(&gc, &gs, r, k, c0, kn), records::mxfp8_record(&sc, &ss, kn));
    assert_eq!(records::mxfp8_fc2(&gc, &gs, r, k, 0, k), g);
}

/// The Rust code against the C++ twin (engine/kernels/nvfp4_pack.h), byte for byte: fc1, an fc2 column slice, and both
/// quantization rules with per-row and fixed outer scales.
#[test]
fn matches_the_cpp_twin() {
    let dir = format!("{}/nvfp4-test-{}", std::env::temp_dir().display(), std::process::id());
    std::fs::create_dir_all(&dir).unwrap();
    let mut rng = Rng(12345);
    let (r, k, kfull, c0, nq, kq) = (96usize, 192usize, 512usize, 128usize, 40usize, 208usize);
    let (gp, gs, up, us) = (rng.bytes(r * k / 2), rng.bytes(r * k / 16), rng.bytes(r * k / 2), rng.bytes(r * k / 16));
    let (w2p, w2s) = (rng.bytes(r * kfull / 2), rng.bytes(r * kfull / 16));
    let mut x = rng.floats(nq * kq);
    x[kq..2 * kq].fill(0.0);                              // an all-zero row
    x[2 * kq..2 * kq + 16].fill(0.0);                     // an all-zero block
    for (i, v) in x[3 * kq..4 * kq].iter_mut().enumerate() {
        *v = (i as f32 - 97.0) * 0.125;                   // many exact ties
    }
    x[4 * kq] = 3.0e37;
    x[4 * kq + 1] = 1.0e-41;
    let fixed = 0.0123f32;
    let mut input = Vec::new();
    for v in [r, k, kfull, c0, nq, kq] {
        input.extend_from_slice(&(v as i32).to_le_bytes());
    }
    for b in [&gp, &gs, &up, &us, &w2p, &w2s] {
        input.extend_from_slice(b);
    }
    for v in x.iter().chain([&fixed]) {
        input.extend_from_slice(&v.to_le_bytes());
    }
    let (inp, outp, exe) = (format!("{dir}/in.bin"), format!("{dir}/out.bin"), format!("{dir}/cpp"));
    std::fs::write(&inp, input).unwrap();
    let src = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/nvfp4_cpp.cpp");
    let inc = concat!("-I", env!("CARGO_MANIFEST_DIR"), "/../kernels");
    let st = std::process::Command::new("g++")
        .args(["-std=c++17", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror", inc, src, "-o", &exe])
        .status()
        .unwrap();
    assert!(st.success(), "g++");
    assert!(std::process::Command::new(&exe).args([&inp, &outp]).status().unwrap().success());
    let cpp = std::fs::read(&outp).unwrap();

    let mut want = nvfp4_fc1(&gp, &gs, &up, &us, r, k);
    let n_fc1 = want.len();
    want.extend_from_slice(&nvfp4_fc2(&w2p, &w2s, r, kfull, c0, k));
    let n_pack = want.len();
    for rule in [Nvfp4ScaleRule::Static6, Nvfp4ScaleRule::FourOverSixMae] {
        for row in x.chunks(kq) {
            let (c, s, o) = nvfp4_quantize(row, kq, rule, None);
            want.extend_from_slice(&c);
            want.extend_from_slice(&s);
            want.extend_from_slice(&o.to_le_bytes());
        }
        let (c, s, o) = nvfp4_quantize(&x, kq, rule, Some(fixed));
        want.extend_from_slice(&c);
        want.extend_from_slice(&s);
        want.extend_from_slice(&o.to_le_bytes());
    }
    assert_eq!(cpp.len(), want.len());
    assert!(cpp[..n_fc1] == want[..n_fc1], "fc1 differs from the C++ packer");
    assert!(cpp[n_fc1..n_pack] == want[n_fc1..n_pack], "fc2 differs from the C++ packer");
    let half = (want.len() - n_pack) / 2;
    assert!(cpp[n_pack..n_pack + half] == want[n_pack..n_pack + half], "Static6 quantization differs from C++");
    assert!(cpp[n_pack + half..] == want[n_pack + half..], "FourOverSixMae quantization differs from C++");
    eprintln!("nvfp4: {} packed bytes and {} quantizer bytes identical to nvfp4_pack.h", n_pack, 2 * half);
    let _ = std::fs::remove_dir_all(&dir);
}
