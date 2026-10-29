//! The trellis codebooks, which turn a trellis state into a weight value, and which one a width file uses.
//!
//! Three exist (engine/kernels/tr3.cuh TR3_*, the same definitions):
//! - mcg: the 16-bit state of a weight times 0xCBAC1FED, masked with 0x8fff8fff and xored with 0x3b603b60, the two
//!   fp16 halves added.
//! - mul1: the 16-bit state times 0x83DCD12D, the four bytes of the product summed (s), then (1024 + s) * c1 + c0
//!   rounded once to fp16, c1 = 0.00676727294921875, c0 = -10.3828125.
//! - pmcgx: a pair trellis. Weights 2p and 2p + 1 share one 20-bit state, the 20 stream bits ending where weight
//!   2p + 1's 16-bit state would end. m1 = mask(w * 0xCBAC1FED) and m2 = mask(w * 0xBCA14C51) with mcg's mask; weight
//!   2p is m1.lo + m2.hi and weight 2p + 1 is m1.hi + m2.lo, fp16 adds.
//!
//! exlr8 rounds every value to FP8 E4M3 (round to nearest even, saturating) inside the encoder's search.
//!
//! Decoding one codebook's codes with another's hash gives plausible wrong weights and no error, so the files say which
//! they use and gr8 checks it. An exlr8 width file names its trellis tensors `trellis.gate`, `trellis.up`,
//! `trellis.down` (and `shared.trellis.*`) for mcg, and suffixes the codebook for any other: `trellis.gate.mul1`. Its
//! metadata `codebook` names it too (the first word, then the multipliers); mcg files written before the other
//! codebooks have none. gr8 reads expert bytes by offset (the index slices), which a name cannot protect, so the
//! planner reads each width file's header (`width_file_codebook`) and checks every half it plans lies inside the
//! trellis tensor of its projection (`TrellisRanges`). A layer whose width files are not there at all (a partial copy of
//! the release, booted with --layers) gets `UNCHECKED`: the engine refuses to bind it, and a load of it fails anyway.

use crate::exlr8::{e4m3_round, f16_to_f32, f32_to_f16_value, StFile};
use crate::json::Json;
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codebook {
    Mcg,
    Mul1,
    Pmcgx,
}

pub const ALL: [Codebook; 3] = [Codebook::Mcg, Codebook::Mul1, Codebook::Pmcgx];

/// The expert_format of a layer whose width files were missing when the plan read them (engine/loader/g6pack.cpp).
pub const UNCHECKED: &str = "exlr8-e4m3-unchecked";

impl Codebook {
    pub fn name(self) -> &'static str {
        match self {
            Codebook::Mcg => "mcg",
            Codebook::Mul1 => "mul1",
            Codebook::Pmcgx => "pmcgx",
        }
    }

    pub fn from_name(s: &str) -> Option<Codebook> {
        ALL.into_iter().find(|c| c.name() == s)
    }

    /// The multipliers its metadata string must name.
    fn multipliers(self) -> &'static [&'static str] {
        match self {
            Codebook::Mcg => &["0xCBAC1FED"],
            Codebook::Mul1 => &["0x83DCD12D"],
            Codebook::Pmcgx => &["0xCBAC1FED", "0xBCA14C51"],
        }
    }

    /// The manifest's expert_format of an exlr8-e4m3 layer in this codebook (engine/loader/g6pack.cpp). mcg keeps the
    /// name every pack before the other codebooks carries, so those manifests are unchanged.
    pub fn expert_format(self) -> &'static str {
        match self {
            Codebook::Mcg => "exlr8-e4m3",
            Codebook::Mul1 => "exlr8-e4m3-mul1",
            Codebook::Pmcgx => "exlr8-e4m3-pmcgx",
        }
    }

    /// Stream bits in one state: 16, or 20 for the pmcgx pair state.
    pub fn state_bits(self) -> usize {
        if self == Codebook::Pmcgx { 20 } else { 16 }
    }
}

fn mask3(x: u32) -> u32 {
    (x & 0x8FFF_8FFF) ^ 0x3B60_3B60
}

/// f32 -> f16 bits, for values an f16 holds exactly (zero and normals: the codebook values and their E4M3 rounding)
fn f16_bits_exact(x: f32) -> u16 {
    if x == 0.0 {
        return if x.is_sign_negative() { 0x8000 } else { 0 };
    }
    let b = x.to_bits();
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    assert!((1..31).contains(&e) && b & 0x1fff == 0, "{x} is not a normal f16 value");
    (((b >> 16) & 0x8000) | ((e as u32) << 10) | ((b >> 13) & 0x3ff)) as u16
}

/// The fp16 value(s) of a state before E4M3 rounding: [v, v], or for pmcgx [weight 2p, weight 2p + 1].
pub fn values_f16(cb: Codebook, s: u32) -> [u16; 2] {
    // fp16 halves of a masked product lie in [0.125, 2), so their f32 sum is exact and rounding it is the fp16 add
    let add = |a: u32, b: u32| f16_bits_exact(f32_to_f16_value(f16_to_f32(a as u16) + f16_to_f32(b as u16)));
    match cb {
        Codebook::Mcg => {
            let x = mask3(s.wrapping_mul(0xCBAC_1FED));
            let v = add(x & 0xffff, x >> 16);
            [v, v]
        }
        Codebook::Mul1 => {
            let x = s.wrapping_mul(0x83DC_D12D);
            let sum = (x & 0xff) + ((x >> 8) & 0xff) + ((x >> 16) & 0xff) + (x >> 24);
            // exact in f32 (11-bit operands, the result a multiple of 2^-18 below 2^4), then the fma's one rounding
            let v = f16_bits_exact(f32_to_f16_value((1024.0 + sum as f32) * 0.00676727294921875 - 10.3828125));
            [v, v]
        }
        Codebook::Pmcgx => {
            let (m1, m2) = (mask3(s.wrapping_mul(0xCBAC_1FED)), mask3(s.wrapping_mul(0xBCA1_4C51)));
            [add(m1 & 0xffff, m2 >> 16), add(m1 >> 16, m2 & 0xffff)]
        }
    }
}

/// An fp16 value rounded to E4M3 and back.
pub fn e4m3_f16(h: u16) -> u16 {
    f16_bits_exact(e4m3_round(f16_to_f32(h)))
}

/// Every state's E4M3-rounded value(s) as f32: index s for mcg and mul1, 2 s + i for weight 2p + i of pmcgx.
pub fn table(cb: Codebook) -> &'static [f32] {
    static T: [OnceLock<Vec<f32>>; 3] = [OnceLock::new(), OnceLock::new(), OnceLock::new()];
    T[cb as usize].get_or_init(|| {
        let n = 1u32 << cb.state_bits();
        let per = if cb == Codebook::Pmcgx { 2 } else { 1 };
        (0..n).flat_map(|s| {
            let v = values_f16(cb, s);
            (0..per).map(move |i| f16_to_f32(e4m3_f16(v[i])))
        }).collect()
    })
}

// ------------------------------------------------------------------------------------------------ width files

/// The codebook a metadata string names, if it is one gr8 decodes: its first word, the multipliers it must carry,
/// and for pmcgx the pair shape (V = 2 weights per 20-bit state).
pub fn parse_metadata(s: &str) -> Result<Codebook, String> {
    let first = s.split([' ', ',']).next().unwrap_or("");
    let cb = Codebook::from_name(first).ok_or(format!("codebook {s:?} is not one gr8 decodes (mcg, mul1, pmcgx)"))?;
    let lower = s.to_ascii_lowercase();
    if let Some(m) = cb.multipliers().iter().find(|m| !lower.contains(&m.to_ascii_lowercase())) {
        return Err(format!("codebook {s:?} names {first} without its multiplier {m}"));
    }
    if cb == Codebook::Pmcgx && !(lower.contains("v2") && lower.contains("l20")) {
        return Err(format!("codebook {s:?}: gr8 decodes the pmcgx pair trellis V2 L20 only"));
    }
    Ok(cb)
}

/// The byte ranges in a width file of the trellis tensor of each projection (gate, up, down), routed and shared.
pub struct TrellisRanges {
    pub routed: [Option<(u64, u64)>; 3],
    pub shared: [Option<(u64, u64)>; 3],
}

impl TrellisRanges {
    /// Whether [off, off + n) lies inside projection p's trellis tensor (shared: the shared expert's).
    pub fn holds(&self, shared: bool, p: usize, off: u64, n: u64) -> bool {
        let r = if shared { self.shared[p] } else { self.routed[p] };
        r.is_some_and(|(a, b)| a <= off && off + n <= b)
    }
}

const PROJS: [&str; 3] = ["gate", "up", "down"];

/// The codebook of an exlr8 width file, from its tensor names and its metadata, and its trellis tensors' ranges.
/// Refuses a file whose trellis tensors carry a suffix gr8 does not know, carry different ones, or disagree with the
/// metadata; and a metadata codebook gr8 does not know. A file without metadata codebook and without a suffix is mcg
/// (every file before the other codebooks).
pub fn width_file_codebook(f: &StFile) -> Result<(Codebook, TrellisRanges), String> {
    let mut by_name: Option<Codebook> = None;
    let mut ranges = TrellisRanges { routed: [None; 3], shared: [None; 3] };
    for n in f.names() {
        let (shared, rest) = match n.strip_prefix("shared.") {
            Some(r) => (true, r),
            None => (false, n.as_str()),
        };
        let Some(rest) = rest.strip_prefix("trellis.") else { continue };
        let (proj, suffix) = rest.split_once('.').unwrap_or((rest, ""));
        let p = PROJS.iter().position(|&x| x == proj).ok_or(format!("{}: unknown trellis tensor {n}", f.path))?;
        let cb = if suffix.is_empty() { Codebook::Mcg } else {
            Codebook::from_name(suffix).filter(|&c| c != Codebook::Mcg)
                .ok_or(format!("{}: trellis tensor {n} is in codebook {suffix:?}, which gr8 does not decode", f.path))?
        };
        if by_name.is_some_and(|b| b != cb) {
            return Err(format!("{}: its trellis tensors name different codebooks", f.path));
        }
        by_name = Some(cb);
        let i = f.info(n)?;
        let slot = if shared { &mut ranges.shared[p] } else { &mut ranges.routed[p] };
        *slot = Some((i.start, i.end));
    }
    let by_name = by_name.ok_or(format!("{}: no trellis tensors", f.path))?;
    match f.meta.get("codebook") {
        None if by_name == Codebook::Mcg => {}
        None => return Err(format!("{}: trellis tensors in {} but no metadata codebook", f.path, by_name.name())),
        Some(m) => {
            let s = m.str().ok_or(format!("{}: metadata codebook is not a string", f.path))?;
            let by_meta = parse_metadata(s).map_err(|e| format!("{}: {e}", f.path))?;
            if by_meta != by_name {
                return Err(format!("{}: metadata codebook {s:?} but trellis tensors in {}", f.path, by_name.name()));
            }
        }
    }
    Ok((by_name, ranges))
}

/// A dense or vocabulary file's codebook check: gr8 decodes dense trellis matrices in mcg only (the dense files name
/// no other codebook yet), so a file whose metadata names another codebook, or that has a suffixed trellis tensor
/// (`M.trellis.<codebook>`), is refused.
pub fn check_dense_mcg(f: &StFile) -> Result<(), String> {
    if let Some(s) = f.meta.get("codebook").and_then(Json::str) {
        let cb = parse_metadata(s).map_err(|e| format!("{}: {e}", f.path))?;
        if cb != Codebook::Mcg {
            return Err(format!("{}: dense trellis matrices in codebook {}; gr8 decodes them in mcg only", f.path, cb.name()));
        }
    }
    if let Some(n) = f.names().iter().find(|n| n.contains(".trellis.")) {
        return Err(format!("{}: {n}: dense trellis matrices in another codebook; gr8 decodes them in mcg only", f.path));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FNV-1a 64 of every state's value(s) as LE u16 fp16 bits, before and after the E4M3 rounding. The GPU test
    /// (engine/kernels/tests/tr3_e4m3_test.cu) and the Python reference (engine/ref/g6ref.py) check the same constants.
    pub(crate) const HASHES: [(Codebook, u64, u64); 3] = [
        (Codebook::Mcg, 0x7bb36b05a3b885a7, 0xd6995a3ae102700a),
        (Codebook::Mul1, 0x55e1c748ed9f0cd7, 0xf4f1a8cc7cda90bf),
        (Codebook::Pmcgx, 0x6f7539ea76219e77, 0xc8d37146b1cb8465),
    ];

    fn fnv(h: &mut u64, x: u16) {
        for b in x.to_le_bytes() {
            *h ^= b as u64;
            *h = h.wrapping_mul(0x100000001b3);
        }
    }

    #[test]
    fn every_state_matches_the_gpu_hashes() {
        for (cb, want_f16, want_e4) in HASHES {
            let (mut hf, mut he) = (0xcbf29ce484222325u64, 0xcbf29ce484222325u64);
            let per = if cb == Codebook::Pmcgx { 2 } else { 1 };
            for s in 0..1u32 << cb.state_bits() {
                let v = values_f16(cb, s);
                for &x in &v[..per] {
                    fnv(&mut hf, x);
                    fnv(&mut he, e4m3_f16(x));
                }
            }
            assert_eq!((hf, he), (want_f16, want_e4), "{}", cb.name());
            // the table is the E4M3 values
            let t = table(cb);
            assert_eq!(t.len(), per << cb.state_bits());
            assert_eq!(t[per * 12345].to_bits(), f16_to_f32(e4m3_f16(values_f16(cb, 12345)[0])).to_bits());
        }
    }

    #[test]
    fn known_values() {
        // state 0: mcg x = 0x3b603b60, 0.9297 + 0.9297; mul1 s = 0, 1024 c1 + c0 = -3.453125 -> E4M3 -3.5
        assert_eq!(values_f16(Codebook::Mcg, 0)[0], 0x3f60);
        assert_eq!(values_f16(Codebook::Mul1, 0)[0], 0xc2e8);
        assert_eq!(f16_to_f32(e4m3_f16(0xc2e8)), -3.5);
        let p = values_f16(Codebook::Pmcgx, 0);
        assert_eq!(p, [0x3f60, 0x3f60]);
    }

    #[test]
    fn metadata_strings() {
        assert_eq!(parse_metadata("mcg 0xCBAC1FED, E4M3").unwrap(), Codebook::Mcg);
        assert_eq!(parse_metadata("mul1 0x83DCD12D, E4M3").unwrap(), Codebook::Mul1);
        assert_eq!(parse_metadata("pmcgx V2 L20 0xCBAC1FED/0xBCA14C51, every value rounded to E4M3").unwrap(), Codebook::Pmcgx);
        for bad in ["gauss 0x12345678, E4M3", "mul1 0xCBAC1FED, E4M3", "pmcgx V2 L22 0xCBAC1FED/0xBCA14C51", "pmcgx V2 L20 0xCBAC1FED", ""] {
            assert!(parse_metadata(bad).is_err(), "{bad}");
        }
    }
}
