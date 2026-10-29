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
    // a legacy name other than G6_NAME
    let e = env(&[("ATTNK_SESS", "2")]);
    assert_eq!(e.get(find("ATTNK_SESS"), 0), Some(&Value::Int(2)));
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
    let e = env(&[("GR8_MTP_KSTOPB_TAU_S2", "0.50,0.9"), ("GR8_RT_STORE", "a\nb")]);
    assert_eq!(e.signature(), " env MTP_KSTOPB_TAU_S2=0.5,0.9 RT_STORE=a\\nb");
}

#[test]
fn generated_files_are_current() {
    let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for (path, text) in gen::files() {
        let on_disk = std::fs::read_to_string(repo.join(path)).unwrap_or_default();
        assert!(on_disk == text, "{path} is out of date: cargo run --bin knobs -- gen");
    }
}
