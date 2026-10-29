use std::ffi::OsString;

use super::*;

fn env(pairs: &[(&str, &str)]) -> Env {
    Env::read(pairs.iter().map(|(k, v)| (OsString::from(k), OsString::from(v))))
}

fn find(name: &str) -> usize {
    TABLE.iter().position(|s| s.name == name).unwrap()
}

#[test]
fn table_is_sorted_and_whole() {
    let names: Vec<_> = TABLE.iter().map(|s| s.name).collect();
    let mut sorted = names.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(names, sorted, "the table is not sorted, or lists a name twice");
    for (s, d) in TABLE.iter().zip(defaults()) {
        assert!(!s.doc.trim().is_empty(), "{}: no description", s.name);
        assert!(s.default.is_some() != !s.unset.is_empty(), "{}: a default and an unset text", s.name);
        assert!(!s.family || s.default.is_none(), "{}: a family with a default", s.name);
        assert!(d.is_some() == s.default.is_some());
    }
}

#[test]
fn values() {
    let k = &TABLE[find("ATTN_CTX2")]; // int, 0..4
    assert_eq!(parse(k, "3"), Ok(Value::Int(3)));
    assert!(parse(k, "5").unwrap_err().contains("outside 0..4"));
    assert!(parse(k, "x").is_err());
    let k = &TABLE[find("ATTN_CTX3_HG")]; // 0|2|4|8
    assert_eq!(parse(k, "4"), Ok(Value::Int(4)));
    assert!(parse(k, "3").unwrap_err().contains("not one of"));
    let k = &TABLE[find("L2PF_T")]; // hex
    assert_eq!(parse(k, "0x3ff"), Ok(Value::Int(0x3ff)));
    let k = &TABLE[find("ATTN_D256")]; // flag
    assert_eq!(parse(k, "1"), Ok(Value::Flag(true)));
    assert!(parse(k, "2").is_err() && parse(k, "yes").is_err());
    let k = &TABLE[find("MTP_KSTOPB_TAU")]; // floats
    assert_eq!(parse(k, "0.5, 0.9"), Ok(Value::Floats(vec![0.5, 0.9])));
    assert!(parse(k, "0.5,,0.9").is_err());
    let k = &TABLE[find("KVB")]; // bf16|fp8
    assert!(parse(k, "fp8").is_ok() && parse(k, "fp4").is_err());
}

#[test]
fn names() {
    let e = env(&[("GR8_ATTN_CTX2", "2"), ("G6_ATTN_CTX3", "1"), ("GR8_NO_SUCH", "1"), ("PATH", "/bin")]);
    assert_eq!(e.get(find("ATTN_CTX2"), 0), Some(&Value::Int(2)));
    assert_eq!(e.get(find("ATTN_CTX3"), 0), Some(&Value::Int(1)));
    let p: Vec<_> = e.problems().iter().map(|p| (p.error, p.text.as_str())).collect();
    assert_eq!(p.len(), 2, "{p:?}");
    assert!(p[0].0 && p[0].1.starts_with("GR8_NO_SUCH is not an engine variable"));
    assert!(!p[1].0 && p[1].1.starts_with("G6_ATTN_CTX3 is the old name of GR8_ATTN_CTX3"));
    // the same value under both names warns, different values are an error, and the new name wins
    let e = env(&[("GR8_ATTN_CTX2", "2"), ("G6_ATTN_CTX2", "2")]);
    assert!(e.problems().iter().all(|p| !p.error));
    let e = env(&[("GR8_ATTN_CTX2", "2"), ("G6_ATTN_CTX2", "3")]);
    assert!(e.problems()[0].error && e.get(find("ATTN_CTX2"), 0) == Some(&Value::Int(2)));
    // an empty value is unset, a family takes any number, an exact name has no G6_ form
    let e = env(&[("GR8_ATTN_CTX2", ""), ("GR8_MTP_KSTOPB_TAU_S3", "0.5"), ("TMPDIR", "/x"), ("G6_TMPDIR", "/y")]);
    assert_eq!(e.get(find("ATTN_CTX2"), 0), None);
    assert_eq!(e.get(find("MTP_KSTOPB_TAU_S"), 3), Some(&Value::Floats(vec![0.5])));
    assert_eq!(e.get(find("MTP_KSTOPB_TAU_S"), 4), None);
    assert_eq!(e.get(find("TMPDIR"), 0).map(Value::canon).as_deref(), Some("/x"));
    assert!(e.problems().iter().any(|p| p.text.starts_with("G6_TMPDIR is not")));
    // a test knob's old names: its own, and the GR8_ one it had before GR8_TEST_
    let e = env(&[("ATTNK_SESS", "2"), ("GR8_S2_RUNS", "0:4")]);
    assert_eq!(e.get(find("TEST_ATTNK_SESS"), 0), Some(&Value::Int(2)));
    assert_eq!(e.get(find("TEST_S2_RUNS"), 0).map(Value::canon).as_deref(), Some("0:4"));
    assert_eq!(e.tests_set(), ["GR8_TEST_ATTNK_SESS", "GR8_TEST_S2_RUNS"]);
}

#[test]
fn bad_value_is_an_error() {
    let e = env(&[("GR8_ATTN_CTX2", "9")]);
    assert_eq!(e.problems().len(), 1);
    assert!(e.problems()[0].error && e.problems()[0].text.starts_with("GR8_ATTN_CTX2=9 is outside 0..4"));
}

#[test]
fn signature() {
    assert_eq!(env(&[]).signature(), "");
    // agree-scope only, in table order, defaults left out, the old name signs as the new one
    let e = env(&[("GR8_ATTN_CTX3", "1"), ("G6_ATTN_CTX2", "2"), ("GR8_ATTN_FRONT", "0"), ("GR8_L2PF", "1")]);
    assert_eq!(TABLE[find("L2PF")].scope, Scope::Local);
    assert_eq!(e.signature(), " env ATTN_CTX2=2 ATTN_CTX3=1");
    let e = env(&[("GR8_MTP_KSTOPB_TAU_S2", "0.50,0.9"), ("GR8_S2_DSEL_TABLE", "a\nb")]);
    assert_eq!(e.signature(), " env MTP_KSTOPB_TAU_S2=0.5,0.9 S2_DSEL_TABLE=a\\nb");
    // the launcher resolves remote:auto per rank (odd ranks take the KV node's second rail): no part of the signature
    let rank = |addr: &str| env(&[("GR8_RT_STORE", addr), ("GR8_RT_SIZE", "1024")]).signature();
    assert_eq!(rank("remote:10.0.0.1:7350"), rank("remote:10.1.0.1:7350"));
}

#[test]
fn generated_files_are_current() {
    let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for (path, text) in gen::files() {
        let on_disk = std::fs::read_to_string(repo.join(path)).unwrap_or_default();
        assert!(on_disk == text, "{path} is out of date: cargo run --bin knobs -- gen");
    }
}

#[allow(dead_code)] // the tests read only some fields
#[path = "yaml.rs"]
mod yaml;

#[test]
fn yaml_file_rules() {
    let refused = |src: &str| {
        let mut out = std::collections::BTreeMap::new();
        yaml::add(&mut out, &mut vec![], "t.yaml", src).unwrap_err()
    };
    let one = "prefix: GR8_TEST_\nscope: test\nknobs:\n  GR8_TEST_X:\n    kind: flag\n    default: \"0\"\n";
    let mut out = std::collections::BTreeMap::new();
    yaml::add(&mut out, &mut vec![], "t.yaml", one).unwrap();
    assert_eq!(out["TEST_X"].legacy, ["G6_TEST_X"]);
    assert!(refused(&one.replace("  GR8_TEST_X:", "  GR8_X:")).contains("start with \"GR8_TEST_\""));
    let local = "prefix: GR8_\nknobs:\n  GR8_X:\n    scope: test\n    kind: flag\n    default: \"0\"\n";
    assert!(refused(local).contains("a test knob belongs in a file whose scope is test"));
    assert!(refused(&one.replace("    default: \"0\"\n", "")).contains("exactly one of default and unset"));
    assert!(refused(&format!("{one}    colour: red\n")).contains("unknown field"));
    let err = yaml::add(&mut out, &mut vec![], "u.yaml", one).unwrap_err();
    assert!(err.contains("also in t.yaml"), "{err}");
    // an old name need not carry the file's prefix
    let old = one.replace("    kind: flag", "    kind: flag\n    legacy: [\"G6_X\", \"ATTNK_X\"]");
    let mut out = std::collections::BTreeMap::new();
    yaml::add(&mut out, &mut vec![], "t.yaml", &old).unwrap();
    assert_eq!(out["TEST_X"].legacy, ["G6_X", "ATTNK_X"]);
}
