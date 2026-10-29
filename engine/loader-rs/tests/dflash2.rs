//! src/dflash2.rs against tests/data/dflash2: a small synthetic drafter converted at TP=4, and at TP=6 in the replicated
//! layout, gives the checked-in manifests and packs byte for byte. flash/: one of GLM 5.3 Flash's shape (five taps,
//! four q heads per KV head, rope_theta 1e4), at TP=4 and TP=2; its packs come from the same Python converter, with
//! D.meta's fields from the tap count.

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
