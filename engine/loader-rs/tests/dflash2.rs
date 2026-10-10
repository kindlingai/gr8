//! src/dflash2.rs against tests/data/dflash2 (tests/make_dflash2_fixture.py): a small synthetic drafter converted at
//! TP=4 gives the manifests and packs tools/convert/dflash2.py wrote, byte for byte.

use g6load::dflash2::{self, Dims, St};
use g6load::json::{self, Json};
use g6load::sha256::Sha256;

const D: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/dflash2");
const SMALL: Dims = Dims { hid: 64, hd: 16, nh: 8, nkv: 4, inter: 128, vocab: 32 };

fn config() -> Json {
    json::parse(&std::fs::read_to_string(format!("{D}/src/config.json")).unwrap()).unwrap()
}

fn same_as_python(sub: &str, ranks: &[usize], bf16: &[&str]) {
    let out = std::env::temp_dir().join(format!("g6load-dflash2-{sub}-{}", std::process::id()));
    std::fs::create_dir_all(&out).unwrap();
    let out_s = out.to_str().unwrap();
    let cfg = config();
    dflash2::check_config(&cfg, &SMALL).unwrap();
    let st = St::open(&format!("{D}/src/model.safetensors")).unwrap();
    let want = json::parse(&std::fs::read_to_string(format!("{D}/{sub}/packs.json")).unwrap()).unwrap();
    let bf16: Vec<String> = bf16.iter().map(|s| s.to_string()).collect();
    for &r in ranks {
        dflash2::convert_rank(&st, &cfg, &SMALL, 4, r, out_s, &bf16).unwrap();
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
    same_as_python("mxfp8", &[0, 1, 2, 3], &[]);
}

#[test]
fn bf16_classes_match_the_python_converter() {
    same_as_python("bf16-attn-sel", &[2], &["sel", "attn"]);
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
    assert!(dflash2::check_config(&config(), &dflash2::GLM53).err().unwrap().contains("hidden_size = 64"));
}
