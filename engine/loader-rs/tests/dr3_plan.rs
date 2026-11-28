//! Integration tests for Drama 3 TP=1 trellis expert planner (`dr3::plan` & C ABI `g6l_dr3_plan`).

use g6load::json::{self, Json};
use g6load::{g6l_dr3_plan, g6l_plan_free, g6l_plan_read};
use std::ffi::CString;
use std::fs;
use std::path::{Path, PathBuf};

fn item128_ref(src: &[u8], hidden: usize) -> Vec<u8> {
    let groups = hidden / 128;
    let tb = src.len() / (groups * 64);
    let mut dst = vec![0u8; src.len()];
    for a in 0..groups {
        for b in 0..8 {
            for c in 0..8 {
                let o = ((a * 8 + c) * 8 + b) * tb;
                let i = ((a * 8 + b) * 8 + c) * tb;
                dst[o..o + tb].copy_from_slice(&src[i..i + tb]);
            }
        }
    }
    dst
}

const H: usize = 2048;
const BLOCKS: usize = H / 128; // 16
const NH: usize = 4;
const NEXP: usize = 256;
const TOTAL_E: usize = NEXP + 1; // 256 routed + shared at row 256

fn write_st(path: &Path, meta: &[(&str, &str)], tensors: &[(&str, &str, &[usize], &[u8])]) {
    let mut header = String::from("{");
    if !meta.is_empty() {
        header.push_str("\"__metadata__\":{");
        for (i, (k, v)) in meta.iter().enumerate() {
            if i > 0 {
                header.push(',');
            }
            header.push_str(&format!("\"{k}\":\"{v}\""));
        }
        header.push('}');
    }
    let mut off = 0usize;
    for (i, (name, dtype, shape, data)) in tensors.iter().enumerate() {
        if !meta.is_empty() || i > 0 {
            header.push(',');
        }
        let sh = shape.iter().map(usize::to_string).collect::<Vec<_>>().join(",");
        let end = off + data.len();
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{sh}],\"data_offsets\":[{off},{end}]}}"
        ));
        off = end;
    }
    header.push('}');
    let mut hb = header.into_bytes();
    while (8 + hb.len()) % 64 != 0 {
        hb.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + hb.len() + off);
    out.extend_from_slice(&(hb.len() as u64).to_le_bytes());
    out.extend_from_slice(&hb);
    for (_, _, _, data) in tensors {
        out.extend_from_slice(data);
    }
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, out).unwrap();
}

/// Builds a synthetic Drama 3 TP=1 trellis release for `layers` at widths `ks` with codebook `cb`.
fn build_synthetic_dr3_release(dir: &Path, layers: &[usize], ks: &[u8], cb: &str) {
    let (suffix, cb_meta) = match cb {
        "mcg" => ("", "mcg 0xCBAC1FED, every value rounded to E4M3"),
        "mul1" => (".mul1", "mul1 0x83DCD12D, every value rounded to E4M3"),
        _ => (".pmcgx", "pmcgx V2 L20 0xCBAC1FED/0xBCA14C51, every value rounded to E4M3"),
    };

    for &k in ks {
        let hb = H * 16 * (k as usize);
        let routed_bytes = NEXP * NH * hb;
        let shared_bytes = NH * hb;

        // Per-layer width files + one combined TP=1 rank-0 index slice file per width
        let mut slice_tensors: Vec<(String, String, Vec<usize>, Vec<u8>)> = Vec::new();
        for &l in layers {
            // Deterministic distinct byte patterns per (proj, expert, half)
            let mk_buf = |len: usize, seed: u8| -> Vec<u8> {
                let mut v = vec![0u8; len];
                for (i, b) in v.iter_mut().enumerate() {
                    *b = ((i.wrapping_mul(31) ^ (seed as usize).wrapping_mul(17) ^ (l * 7) ^ (k as usize)) & 0xff) as u8;
                }
                v
            };
            let rg = mk_buf(routed_bytes, 1);
            let ru = mk_buf(routed_bytes, 2);
            let rd = mk_buf(routed_bytes, 3);
            let sg = mk_buf(shared_bytes, 4);
            let su = mk_buf(shared_bytes, 5);
            let sd = mk_buf(shared_bytes, 6);

            let wpath = dir.join(format!("experts/exlr8-layer-{l:03}.k{k}.safetensors"));
            let names = [
                format!("trellis.gate{suffix}"),
                format!("trellis.up{suffix}"),
                format!("trellis.down{suffix}"),
                format!("shared.trellis.gate{suffix}"),
                format!("shared.trellis.up{suffix}"),
                format!("shared.trellis.down{suffix}"),
            ];
            write_st(
                &wpath,
                &[
                    ("format", "exlr8-v2"),
                    ("contents", "trellis"),
                    ("codebook", cb_meta),
                ],
                &[
                    (&names[0], "U8", &[routed_bytes], &rg),
                    (&names[1], "U8", &[routed_bytes], &ru),
                    (&names[2], "U8", &[routed_bytes], &rd),
                    (&names[3], "U8", &[shared_bytes], &sg),
                    (&names[4], "U8", &[shared_bytes], &su),
                    (&names[5], "U8", &[shared_bytes], &sd),
                ],
            );

            // Compute data_base of the width file we just wrote to get exact file offsets for index slices
            let raw_hdr_len = u64::from_le_bytes(fs::read(&wpath).unwrap()[..8].try_into().unwrap());
            let data_base = 8 + raw_hdr_len;
            let base_rg = data_base;
            let base_ru = base_rg + routed_bytes as u64;
            let base_rd = base_ru + routed_bytes as u64;
            let base_sg = base_rd + routed_bytes as u64;
            let base_su = base_sg + shared_bytes as u64;
            let base_sd = base_su + shared_bytes as u64;

            let m = TOTAL_E * NH;
            let mut units = Vec::with_capacity(m * 4);
            let mut offsets = Vec::with_capacity(m * 3 * 8);
            let mut chan = Vec::with_capacity(m * 3 * 128 * 2);
            for e in 0..TOTAL_E {
                for h in 0..NH {
                    units.extend_from_slice(&(e as i16).to_le_bytes());
                    units.extend_from_slice(&(h as i16).to_le_bytes());
                    let (og, ou, od) = if e < NEXP {
                        let d = ((e * NH + h) * hb) as u64;
                        (base_rg + d, base_ru + d, base_rd + d)
                    } else {
                        let d = (h * hb) as u64;
                        (base_sg + d, base_su + d, base_sd + d)
                    };
                    offsets.extend_from_slice(&(og as i64).to_le_bytes());
                    offsets.extend_from_slice(&(ou as i64).to_le_bytes());
                    offsets.extend_from_slice(&(od as i64).to_le_bytes());
                    for proj in 0..3usize {
                        for c in 0..128usize {
                            // fp16 1.0 (0x3C00) for even, 0.5 (0x3800) for odd
                            let bits: u16 = if (e + h + proj + c) % 2 == 0 { 0x3C00 } else { 0x3800 };
                            chan.extend_from_slice(&bits.to_le_bytes());
                        }
                    }
                }
            }
            let mut block = Vec::with_capacity(TOTAL_E * 3 * BLOCKS * 2);
            for e in 0..TOTAL_E {
                for proj in 0..3usize {
                    for b in 0..BLOCKS {
                        let bits: u16 = if (e + proj + b) % 2 == 0 { 0x3C00 } else { 0x3A00 };
                        block.extend_from_slice(&bits.to_le_bytes());
                    }
                }
            }
            let mut signs = vec![0u8; 2 * H];
            for i in 0..(2 * H) {
                signs[i] = if i % 3 == 0 { (-1i8) as u8 } else { 1u8 };
            }

            let p = format!("L{l:03}.");
            slice_tensors.push((format!("{p}units"), "I16".into(), vec![m, 2], units));
            slice_tensors.push((format!("{p}offset"), "I64".into(), vec![m, 3], offsets));
            slice_tensors.push((format!("{p}chan"), "F16".into(), vec![m, 3, 128], chan));
            slice_tensors.push((format!("{p}block"), "F16".into(), vec![TOTAL_E, 3, BLOCKS], block));
            slice_tensors.push((format!("{p}signs"), "I8".into(), vec![2, H], signs));
        }

        let refs: Vec<(&str, &str, &[usize], &[u8])> = slice_tensors
            .iter()
            .map(|(n, dt, sh, b)| (n.as_str(), dt.as_str(), sh.as_slice(), b.as_slice()))
            .collect();
        let spath = dir.join(format!("index-slices/tp1/k{k}/rank0.safetensors"));
        write_st(&spath, &[("format", "exlr8-v2-slice")], &refs);
    }
}

fn write_manifest(dir: &Path, name: &str, layers: &[usize], k_route: u8, k_shared: u8) {
    let mut ls = String::new();
    for (idx, &l) in layers.iter().enumerate() {
        if idx > 0 {
            ls.push(',');
        }
        let g_row = format!("[{k_route},{k_route},{k_route},{k_route}]");
        let g_all = vec![g_row.as_str(); NEXP].join(",");
        let d_all = vec![k_route.to_string(); NEXP].join(",");
        let s_row = format!("[{k_shared},{k_shared},{k_shared},{k_shared}]");
        ls.push_str(&format!(
            "\"{l}\":{{\"gate\":[{g_all}],\"up\":[{g_all}],\"down\":[{d_all}],\"shared\":{{\"gate\":{s_row},\"up\":{s_row},\"down\":{k_shared}}}}}"
        ));
    }
    let text = format!(
        "{{\"format\":\"exlr8-v2-manifest\",\"model\":\"dr3\",\"name\":\"{name}\",\"hidden_size\":{H},\"nh\":{NH},\"num_experts\":{NEXP},\"layers\":{{{ls}}}}}"
    );
    let mpath = dir.join(format!("manifests/exlr8-manifest-{name}.json"));
    fs::create_dir_all(mpath.parent().unwrap()).unwrap();
    fs::write(mpath, text).unwrap();
}

fn tmp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("g6load-dr3-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn dr3_plan_capi_builds_partial_and_multiwidth_expert_image() {
    let d = tmp_dir("capi");
    // Partial release: layers 0 and 3 present, routed K=3, shared K=4, pmcgx codebook
    build_synthetic_dr3_release(&d, &[0, 3], &[3, 4], "pmcgx");
    write_manifest(&d, "bit300", &[0, 3], 3, 4);

    let out_manifest = d.join("planned_manifest.json");
    let c_dir = CString::new(d.to_str().unwrap()).unwrap();
    let c_man = CString::new("bit300").unwrap();
    let c_out = CString::new(out_manifest.to_str().unwrap()).unwrap();
    let mut err = vec![0u8; 512];

    let h = unsafe {
        g6l_dr3_plan(
            c_dir.as_ptr(),
            c_man.as_ptr(),
            std::ptr::null(),
            c_out.as_ptr(),
            err.as_mut_ptr() as *mut _,
            err.len(),
        )
    };
    let err_str = std::str::from_utf8(&err).unwrap().trim_end_matches('\0');
    assert!(!h.is_null(), "g6l_dr3_plan failed: {err_str}");

    let manifest_txt = fs::read_to_string(&out_manifest).unwrap();
    let mj = json::parse(&manifest_txt).unwrap();
    let tensors = mj.get("tensors").and_then(Json::arr).unwrap();
    // 2 layers * 8 tr3 tensors = 16 tensors (fc1, fc2, sg, v1b, v2, v3b, table, kmix)
    assert_eq!(tensors.len(), 16, "expected 16 tensors for 2 layers");

    let find = |name: &str| -> (u64, u64, String, Vec<i64>) {
        let t = tensors
            .iter()
            .find(|x| x.get("name").and_then(Json::str) == Some(name))
            .unwrap_or_else(|| panic!("missing tensor {name}"));
        let off = t.get("offset").and_then(Json::int).unwrap() as u64;
        let nb = t.get("nbytes").and_then(Json::int).unwrap() as u64;
        let xf = t.get("expert_format").and_then(Json::str).unwrap().to_string();
        let sh = t.get("shape").and_then(Json::arr).unwrap().iter().map(|v| v.int().unwrap()).collect();
        (off, nb, xf, sh)
    };

    for l in [0usize, 3usize] {
        let p = format!("L{l:03}.tr3");
        let (fc1_off, fc1_nb, xf, _) = find(&format!("{p}.fc1"));
        assert_eq!(xf, "exlr8-e4m3-pmcgx");
        // Routed: 256 * 4 * 2 * (2048*16*3), Shared: 1 * 4 * 2 * (2048*16*4)
        let exp_fc1 = (NEXP * NH * 2 * H * 16 * 3 + NH * 2 * H * 16 * 4) as u64;
        assert_eq!(fc1_nb, exp_fc1);

        let (fc2_off, fc2_nb, _, _) = find(&format!("{p}.fc2"));
        let exp_fc2 = (NEXP * NH * H * 16 * 3 + NH * H * 16 * 4) as u64;
        assert_eq!(fc2_nb, exp_fc2);

        let (sg_off, sg_nb, _, sg_sh) = find(&format!("{p}.sg"));
        assert_eq!(sg_sh, vec![2, H as i64]);
        assert_eq!(sg_nb, (2 * H * 2) as u64);

        let (_, _, _, v1b_sh) = find(&format!("{p}.v1b"));
        assert_eq!(v1b_sh, vec![(TOTAL_E * NH) as i64, 2, BLOCKS as i64]);

        let (_, _, _, v2_sh) = find(&format!("{p}.v2"));
        assert_eq!(v2_sh, vec![(TOTAL_E * NH) as i64, 3, 128]);

        let (_, _, _, v3b_sh) = find(&format!("{p}.v3b"));
        assert_eq!(v3b_sh, vec![(TOTAL_E * NH) as i64, BLOCKS as i64]);

        let (tab_off, tab_nb, _, tab_sh) = find(&format!("{p}.table"));
        assert_eq!(tab_sh, vec![TOTAL_E as i64, 6]);
        let mut tab_raw = vec![0u8; tab_nb as usize];
        let rc = unsafe { g6l_plan_read(h, tab_off, tab_nb, tab_raw.as_mut_ptr() as *mut _, err.as_mut_ptr() as *mut _, err.len()) };
        assert_eq!(rc, 0);
        let tab: Vec<i64> = tab_raw.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
        // Shared expert sits at row 256 with width 4 and 4 blocks
        assert_eq!(tab[NEXP * 6 + 2], (NEXP * NH) as i64);
        assert_eq!(tab[NEXP * 6 + 3], NH as i64);
        assert_eq!(tab[NEXP * 6 + 4], 4);

        // Verify sg signs conversion (+1 -> 0x3C00, -1 -> 0xBC00)
        let mut sg_raw = vec![0u8; sg_nb as usize];
        let rc = unsafe { g6l_plan_read(h, sg_off, sg_nb, sg_raw.as_mut_ptr() as *mut _, err.as_mut_ptr() as *mut _, err.len()) };
        assert_eq!(rc, 0);
        assert_eq!(u16::from_le_bytes([sg_raw[0], sg_raw[1]]), 0xBC00); // i=0: -1
        assert_eq!(u16::from_le_bytes([sg_raw[2], sg_raw[3]]), 0x3C00); // i=1: +1

        // Verify fc1 expert 0 block 0 gate reads directly and fc2 expert 0 block 0 is Item128-permuted
        let hb3 = H * 16 * 3;
        let mut fc1_e0g = vec![0u8; hb3];
        let rc = unsafe { g6l_plan_read(h, fc1_off, hb3 as u64, fc1_e0g.as_mut_ptr() as *mut _, err.as_mut_ptr() as *mut _, err.len()) };
        assert_eq!(rc, 0);
        let mut fc2_e0 = vec![0u8; hb3];
        let rc = unsafe { g6l_plan_read(h, fc2_off, hb3 as u64, fc2_e0.as_mut_ptr() as *mut _, err.as_mut_ptr() as *mut _, err.len()) };
        assert_eq!(rc, 0);
        // Reconstruct expected Item128 output for expert 0 block 0 down (seed=3)
        let mut raw_d0 = vec![0u8; hb3];
        for (i, b) in raw_d0.iter_mut().enumerate() {
            *b = ((i.wrapping_mul(31) ^ (3usize).wrapping_mul(17) ^ (l * 7) ^ 3usize) & 0xff) as u8;
        }
        let want_d0 = item128_ref(&raw_d0, H);
        assert_eq!(fc2_e0, want_d0, "L{l:03} fc2 Item128 permutation mismatch");
    }

    unsafe { g6l_plan_free(h) };
    let _ = fs::remove_dir_all(&d);
}

#[test]
fn dr3_plan_emits_block_scales_even_without_shared_expert() {
    let d = tmp_dir("no-shared");
    build_synthetic_dr3_release(&d, &[0], &[4], "mcg");
    let g_row = "[4,4,4,4]";
    let g_all = vec![g_row; NEXP].join(",");
    let d_all = vec!["4"; NEXP].join(",");
    let text = format!(
        "{{\"format\":\"exlr8-v2-manifest\",\"model\":\"dr3\",\"name\":\"nosh\",\"hidden_size\":{H},\"nh\":{NH},\"num_experts\":{NEXP},\"layers\":{{\"0\":{{\"gate\":[{g_all}],\"up\":[{g_all}],\"down\":[{d_all}]}}}}}}"
    );
    let mpath = d.join("manifests/exlr8-manifest-nosh.json");
    fs::create_dir_all(mpath.parent().unwrap()).unwrap();
    fs::write(&mpath, text).unwrap();

    let img = g6load::dr3::plan(d.to_str().unwrap(), "nosh", None).unwrap();
    let names: Vec<&str> = img.tensors.iter().map(|t| t.name.as_str()).collect();
    assert!(names.contains(&"L000.tr3.sg"), "missing sg: {names:?}");
    assert!(names.contains(&"L000.tr3.v1b"), "missing v1b: {names:?}");
    assert!(names.contains(&"L000.tr3.v3b"), "missing v3b: {names:?}");
    let tab = img.tensors.iter().find(|t| t.name == "L000.tr3.table").unwrap();
    assert_eq!(tab.shape, vec![NEXP as i64, 6]);
    let _ = fs::remove_dir_all(&d);
}

#[test]
fn dr3_plan_rejects_bad_args_and_empty_manifest() {
    let mut err = vec![0u8; 256];
    let h = unsafe {
        g6l_dr3_plan(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            err.as_mut_ptr() as *mut _,
            err.len(),
        )
    };
    assert!(h.is_null());

    let d = tmp_dir("empty");
    build_synthetic_dr3_release(&d, &[0], &[4], "mcg");
    let mpath = d.join("manifests/exlr8-manifest-empty.json");
    fs::create_dir_all(mpath.parent().unwrap()).unwrap();
    fs::write(&mpath, "{\"format\":\"exlr8-v2-manifest\",\"model\":\"dr3\",\"layers\":{}}").unwrap();
    assert!(g6load::dr3::plan(d.to_str().unwrap(), "empty", None).is_err());
    let _ = fs::remove_dir_all(&d);
}
