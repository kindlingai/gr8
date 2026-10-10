//! The load-time weight forms (src/records.rs) against the Python converter's (engine/ref/g6ref.py, tools/convert/
//! mxfp8.py) on the inputs of tests/data/records-fixture.safetensors (tests/make_records_fixture.py).

use g6load::exlr8::{bf16_to_f32, StFile};
use g6load::records;

fn fx() -> StFile {
    StFile::open(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/records-fixture.safetensors")).unwrap()
}

fn u16s(b: Vec<u8>) -> Vec<u16> {
    b.as_chunks::<2>().0.iter().map(|c| u16::from_le_bytes(*c)).collect()
}

#[test]
fn decode_records_match_g6ref() {
    let f = fx();
    let r = |n: &str| f.read_all(n).unwrap();
    assert_eq!(records::mxfp8_record(&r("mx.codes"), &r("mx.scales"), 64), r("mx.record"));
    assert_eq!(records::bf16_record(&u16s(r("bf.w")), 48), r("bf.record"));
    assert_eq!(records::rt_record(&r("rt.q"), &u16s(r("rt.s")), 64), r("rt.record"));
    let (gs, us, ds) = (u16s(r("ex.gs")), u16s(r("ex.us")), u16s(r("ex.ds")));
    let (fc1, fc2) = records::rt_expert_record((&r("ex.gq"), &gs), (&r("ex.uq"), &us), (&r("ex.dq"), &ds), 64, 32);
    assert_eq!(fc1, r("ex.fc1"));
    assert_eq!(fc2, r("ex.fc2"));
}

#[test]
fn mxfp8_quantize_matches_the_converter() {
    let f = fx();
    let w: Vec<f32> = u16s(f.read_all("q.w").unwrap()).into_iter().map(bf16_to_f32).collect();
    let (c, s) = records::mxfp8_quantize(&w, 96);
    assert_eq!(s, f.read_all("q.scales").unwrap());
    assert_eq!(c, f.read_all("q.codes").unwrap());
}
