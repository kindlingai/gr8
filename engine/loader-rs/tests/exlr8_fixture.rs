//! gr8's exlr8 dense decoder against the reference loader's output on synthetic matrices of every format
//! (tests/data/exlr8-fixture.safetensors, written by tests/make_exlr8_fixture.py). Bit equality in fp32.

use g6load::exlr8::{self, Matrix, StFile, FP8_BLOCK, INT4, SIDE_IN, TRELLIS};
use std::ops::Range;

fn fixture() -> StFile {
    StFile::open(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/exlr8-fixture.safetensors")).unwrap()
}

fn fnv(v: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for x in v {
        for b in x.to_le_bytes() {
            h = (h ^ b as u64).wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn block(full: &[f32], cols_full: usize, rows: &Range<usize>, cols: &Range<usize>) -> Vec<f32> {
    rows.clone().flat_map(|r| full[r * cols_full + cols.start..r * cols_full + cols.end].to_vec()).collect()
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[test]
fn every_format_matches_the_reference() {
    let f = fixture();
    let names = Matrix::names(&f);
    assert_eq!(names.len(), 11, "{names:?}");
    let mut checked = 0;
    for n in &names {
        let m = Matrix::open(&f, n).unwrap();
        let (o, i) = (m.rec.out, m.rec.inp);
        let got = m.decode(0..o, 0..i).unwrap();
        let want = u64::from_le_bytes(f.read_all(&format!("{n}.want_fnv")).unwrap().try_into().unwrap());
        assert_eq!(fnv(&got), want, "{n}: decode differs from the reference");
        if f.info(&format!("{n}.want")).is_ok() {
            let raw = f.read_all(&format!("{n}.want")).unwrap();
            let w: Vec<f32> = raw.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
            assert!(same_bits(&got, &w), "{n}: full tensor differs");
        }
        assert!(same_bits(&exlr8::decode_par(&m, 0..o, 0..i, 3).unwrap(), &got), "{n}: decode_par differs");
        checked += 1;
    }
    assert_eq!(checked, 11);
}

#[test]
fn blocks_equal_the_same_block_of_the_whole_matrix() {
    let f = fixture();
    for n in Matrix::names(&f) {
        let m = Matrix::open(&f, &n).unwrap();
        let (o, i) = (m.rec.out, m.rec.inp);
        let full = m.decode(0..o, 0..i).unwrap();
        let entries = m.rec.fmt == TRELLIS || (m.rec.fmt == INT4 && m.rec.side == SIDE_IN);
        let blocks: Vec<(Range<usize>, Range<usize>)> = if !entries {
            let c = if m.rec.fmt == INT4 { 32..96 } else { 5..i.min(37) };
            vec![(3..o.min(17), c), (o - 1..o, 0..i)]
        } else if m.rec.side == SIDE_IN {
            let last = i - 128..i;
            vec![(0..o, last.clone()), (7..o.min(40), last), (0..o, 0..128)]
        } else {
            let last = o - 128..o;
            vec![(last.clone(), 0..i), (last, 16..i.min(48)), (0..128, 0..i)]
        };
        for (r, c) in blocks {
            let got = m.decode(r.clone(), c.clone()).unwrap();
            assert!(same_bits(&got, &block(&full, i, &r, &c)), "{n}: block {r:?} x {c:?} differs");
        }
        if m.rec.fmt == FP8_BLOCK {
            // rows that start inside a 128-row scale block
            let got = m.decode(100..200, 0..i).unwrap();
            assert!(same_bits(&got, &block(&full, i, &(100..200), &(0..i))), "{n}: rows 100..200 differ");
        }
    }
}

#[test]
fn trellis_blocks_must_sit_on_entries() {
    let f = fixture();
    let m = Matrix::open(&f, "t_out_k4").unwrap();
    assert!(m.decode(64..192, 0..256).is_err());
    let m = Matrix::open(&f, "t_in_k4").unwrap();
    assert!(m.decode(0..256, 0..64).is_err());
}

#[test]
fn int4_codes_come_back_in_the_hf_orientation() {
    let f = fixture();
    // side 2 stores [in/128][out][64] and [in/128][out][4]: int4() returns [out][in/2] and [out][in/32]
    let m = Matrix::open(&f, "i_in").unwrap();
    let (q, s) = m.int4(&(0..64), &(0..256)).unwrap();
    let q4 = f.read_all("i_in.q4").unwrap();
    let s4 = f.read_all("i_in.s4").unwrap();
    for (o, b) in [(0usize, 0usize), (63, 1), (17, 1)] {
        assert_eq!(&q[o * 128 + b * 64..][..64], &q4[(b * 64 + o) * 64..][..64]);
        let sb: Vec<u16> = s4[(b * 64 + o) * 8..][..8].as_chunks::<2>().0.iter().map(|c| u16::from_le_bytes(*c)).collect();
        assert_eq!(&s[o * 8 + b * 4..][..4], &sb[..]);
    }
}
