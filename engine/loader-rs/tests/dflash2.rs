//! src/dflash2.rs against tests/data/dflash2: a small synthetic drafter converted at TP=4, and at TP=6 in the replicated
//! layout, gives the checked-in manifests and packs byte for byte. flash/: one of GLM 5.3 Flash's shape (five taps,
//! four q heads per KV head, rope_theta 1e4), at TP=4 and TP=2; its packs come from the same Python converter, with
//! D.meta's fields from the tap count. At TP=3 its replicated K-slices are checked against TP=4's records.

use g6load::dflash2::{self, Dims, St};
use g6load::json::{self, Json};
use g6load::sha256::Sha256;

const D: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/dflash2");
const SMALL: Dims = Dims { hid: 64, hd: 16, nh: 8, nkv: 4, inter: 128, vocab: 32 };
const SMALL_FLASH: Dims = Dims { nh: 16, ..SMALL };

fn config_of(src: &str) -> Json {
    json::parse(&std::fs::read_to_string(format!("{D}/{src}/config.json")).unwrap()).unwrap()
}

fn config() -> Json {
    config_of("src")
}

fn same_as_python(sub: &str, nr: usize, ranks: &[usize], bf16: &[&str]) {
    converts_as_python("src", &SMALL, sub, nr, ranks, bf16)
}

fn converts_as_python(src: &str, dims: &Dims, sub: &str, nr: usize, ranks: &[usize], bf16: &[&str]) {
    let out = std::env::temp_dir().join(format!("g6load-dflash2-{}-{}", sub.replace('/', "-"), std::process::id()));
    std::fs::create_dir_all(&out).unwrap();
    let out_s = out.to_str().unwrap();
    let cfg = config_of(src);
    dflash2::check_config(&cfg, dims).unwrap();
    let st = St::open(&format!("{D}/{src}/model.safetensors")).unwrap();
    let want = json::parse(&std::fs::read_to_string(format!("{D}/{sub}/packs.json")).unwrap()).unwrap();
    let bf16: Vec<String> = bf16.iter().map(|s| s.to_string()).collect();
    for &r in ranks {
        dflash2::convert_rank(&st, &cfg, dims, nr, r, out_s, &bf16).unwrap();
        let name = format!("dflash.r{r}.manifest.json");
        assert_eq!(std::fs::read_to_string(out.join(&name)).unwrap(),
                   std::fs::read_to_string(format!("{D}/{sub}/{name}")).unwrap(), "{sub} {name}");
        let mut h = Sha256::new();
        h.update(&std::fs::read(out.join(format!("dflash.r{r}.pack"))).unwrap());
        assert_eq!(Some(h.hex().as_str()), want.get(&format!("dflash.r{r}.pack")).and_then(|s| s.str()), "{sub} rank {r} pack");
    }
    std::fs::remove_dir_all(&out).unwrap();
}

#[test]
fn every_rank_matches_the_python_converter() {
    same_as_python("mxfp8", 4, &[0, 1, 2, 3], &[]);
}

#[test]
fn replicated_tp6_matches_the_python_converter() {
    same_as_python("mxfp8-tp6", 6, &[0, 5], &[]);
}

#[test]
fn bf16_classes_match_the_python_converter() {
    same_as_python("bf16-attn-sel", 4, &[2], &["sel", "attn"]);
}

#[test]
fn flash_every_rank_matches_the_python_converter() {
    converts_as_python("flash/src", &SMALL_FLASH, "flash/mxfp8", 4, &[0, 1, 2, 3], &[]);
}

#[test]
fn flash_tp2_matches_the_python_converter() {
    converts_as_python("flash/src", &SMALL_FLASH, "flash/mxfp8-tp2", 2, &[0, 1], &[]);
}

/// TP=3 replicates the body, and its o_proj / down_proj K-slices are the records of TP=4's ranks, so the drafts are TP=4's.
#[test]
fn flash_tp3_replicates_tp4s_k_slices() {
    let out = std::env::temp_dir().join(format!("g6load-dflash2-flash-tp3-{}", std::process::id()));
    std::fs::create_dir_all(&out).unwrap();
    let cfg = config_of("flash/src");
    let st = St::open(&format!("{D}/flash/src/model.safetensors")).unwrap();
    let sha = |m: &Json, name: &str| {
        let t = m.get("tensors").and_then(|t| t.arr()).unwrap().iter().find(|t| t.get("name").and_then(|n| n.str()) == Some(name));
        t.and_then(|t| t.get("sha256")).and_then(|s| s.str()).unwrap_or_else(|| panic!("no {name}")).to_string()
    };
    for r in 0..3 {
        let rep = dflash2::convert_rank(&st, &cfg, &SMALL_FLASH, 3, r, out.to_str().unwrap(), &[]).unwrap();
        assert_eq!(rep.get("replicated_ks").and_then(|k| k.int()), Some(4));
        for v in 0..4 {
            let tp4 = json::parse(&std::fs::read_to_string(format!("{D}/flash/mxfp8/dflash.r{v}.manifest.json")).unwrap()).unwrap();
            for j in 0..2 {
                assert_eq!(sha(&rep, &format!("D{j}.o{v}.gv")), sha(&tp4, &format!("D{j}.o.gv")));
                assert_eq!(sha(&rep, &format!("D{j}.down{v}.gv")), sha(&tp4, &format!("D{j}.down.gv")));
            }
        }
    }
    std::fs::remove_dir_all(&out).unwrap();
}

#[test]
fn the_hidden_size_picks_the_model() {
    let with_hidden = |h: i64| {
        let mut c = config();
        if let Json::Obj(m) = &mut c {
            m.insert("hidden_size".into(), Json::Int(h));
        }
        dflash2::dims_of(&c).map(|d| (d.hid, d.nh))
    };
    assert_eq!(with_hidden(6144), Ok((6144, 64)));
    assert_eq!(with_hidden(4096), Ok((4096, 32)));
    assert!(with_hidden(64).unwrap_err().contains("hidden_size = 64, this converter is written for 6144"));
}

#[test]
fn refusals_name_the_cause() {
    let edit = |k: &str, v: Json| {
        let mut c = config();
        if let Json::Obj(m) = &mut c {
            m.insert(k.into(), v);
        }
        dflash2::check_config(&c, &SMALL).err().unwrap()
    };
    assert!(edit("is_causal", Json::Bool(true)).contains("causal"));
    assert!(edit("hidden_size", Json::Int(128)).contains("hidden_size = 128"));
    assert!(edit("vocab_size", Json::Int(7)).contains("vocab"));
    let mut d = config().get("dflash_config").unwrap().clone();
    if let Json::Obj(m) = &mut d {
        m.insert("target_layer_ids".into(), Json::Arr(vec![]));
    }
    assert!(edit("dflash_config", d).contains("target_layer_ids"));
    assert!(dflash2::check_config(&config(), &dflash2::GLM53).err().unwrap().contains("hidden_size = 64"));
}

#[test]
fn int4_body_and_codebooks_tp2() {
    let out = std::env::temp_dir().join(format!("g6load-dflash2-int4-tp2-{}", std::process::id()));
    std::fs::create_dir_all(&out).unwrap();
    let cfg = config();
    let st = St::open(&format!("{D}/src/model.safetensors")).unwrap();
    let i4: Vec<String> = ["fc", "attn", "mlp", "conv"].into_iter().map(String::from).collect();
    for r in 0..2 {
        let m = dflash2::convert_rank_ex(&st, &cfg, &SMALL, 2, r, out.to_str().unwrap(), &[], &i4, false).unwrap();
        assert_eq!(m.get("nr").and_then(|v| v.int()), Some(2));
        let tensors = m.get("tensors").and_then(|t| t.arr()).unwrap();
        let find = |n: &str| tensors.iter().find(|t| t.get("name").and_then(|s| s.str()) == Some(n));
        let fc = find("D.fc.gv").unwrap();
        assert_eq!(fc.get("shape").and_then(|s| s.arr()).unwrap()[2].int(), Some(288));
        assert_eq!(fc.get("layout").and_then(|s| s.str()), Some("int4_gv_record[O/16][K/32][288]"));
        let hproj = find("D.sel.hproj.gv").unwrap();
        assert_eq!(hproj.get("shape").and_then(|s| s.arr()).unwrap()[2].int(), Some(528));
    }
    // Real GLM-5.3 selector codebooks have SELR = 256 (8 x g32): verify INT4 codebook shapes & dequant accuracy
    let (v, c) = (16usize, 256usize);
    let src: Vec<u16> = (0..v * c).map(|i| {
        let f = (((i * 37) % 101) as f32 - 50.0) * 0.02;
        (f.to_bits() >> 16) as u16
    }).collect();
    let (q4, s4, qs, ss) = dflash2::int4_codebook(&src, &[v, c]).unwrap();
    assert_eq!(qs, vec![v, c / 2]);
    assert_eq!(ss, vec![v, c / 32]);
    assert_eq!(q4.len(), v * c / 2);
    assert_eq!(s4.len(), v * (c / 32) * 2);
    // Verify round-trip max error matches INT4 g32 quantization bound (<= scale / 2 + f16 eps)
    for row in 0..v {
        for i in 0..c {
            let orig = f32::from_bits((src[row * c + i] as u32) << 16);
            let b = q4[row * (c / 2) + (i >> 1)];
            let nib = if (i & 1) != 0 { b >> 4 } else { b & 15 } as i32 - 8;
            let g = i >> 5;
            let sb = u16::from_le_bytes([s4[(row * (c / 32) + g) * 2], s4[(row * (c / 32) + g) * 2 + 1]]);
            let sc = g6load::exlr8::f16_to_f32(sb);
            let recon = nib as f32 * sc;
            assert!((orig - recon).abs() <= sc * 0.55 + 1e-3, "row {row} col {i}: orig {orig} recon {recon} sc {sc}");
        }
    }
    std::fs::remove_dir_all(&out).unwrap();
}
