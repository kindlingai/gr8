//! The fetch side of the exlr8 v2 release: which byte ranges of the width files one rank loads, and the check
//! of each half once it has landed. Both read only the rank's index slices (tools/convert/v2slices.py: per owned unit
//! its offsets, checksums and per-channel values, and every expert's block scales) and the width manifest, the same
//! inputs src/v2plan.rs builds the rank's image from.

use crate::json::{self, Json};
use crate::v2plan::{widths, Slice, HALF_PER_BIT};
use crate::xxh64::Xxh64;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::os::unix::fs::FileExt;

const SHARED: usize = 256;

/// One half of one projection (0 gate, 1 up, 2 down) at width k, as its width file holds it.
pub struct Unit {
    pub layer: usize,
    pub k: u8,
    pub expert: usize,
    pub half: usize,
    pub proj: usize,
    pub offset: u64,
    pub nbytes: u64,
    pub checksum: u64,
    side: Vec<u8>, // the half's 128 per-channel f16 values, then the projection's 48 f16 block scales
}

/// Which widths a rank fetches: a width manifest's choice per unit, or every unit at the listed widths.
pub enum Choice {
    Manifest(Json),
    Widths(Vec<u8>),
}

impl Choice {
    pub fn manifest(path: &str) -> Result<Choice, String> {
        let m = json::parse(&std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?)?;
        if m.get("format").and_then(Json::str) != Some("exlr8-v2-manifest") {
            return Err(format!("{path}: not an exlr8-v2-manifest"));
        }
        Ok(Choice::Manifest(m))
    }
}

pub fn width_file(l: usize, k: u8) -> String {
    format!("exlr8-layer-{l:03}.k{k}.safetensors")
}

/// One layer's slice tensors at one width.
struct Rows {
    row: HashMap<(usize, usize), usize>,
    offset: Vec<u64>,
    checksum: Vec<u64>,
    chan: Vec<u8>,
    block: Vec<u8>,
}

fn rows(s: &Slice, l: usize) -> Result<Option<Rows>, String> {
    let p = format!("L{l:03}.");
    if !s.has(&format!("{p}units")) {
        return Ok(None);
    }
    let (units, _) = s.get(&format!("{p}units"))?;
    let u64s = |b: Vec<u8>| b.chunks(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect::<Vec<_>>();
    Ok(Some(Rows {
        row: units.chunks(4).enumerate()
            .map(|(j, u)| ((i16::from_le_bytes([u[0], u[1]]) as usize, i16::from_le_bytes([u[2], u[3]]) as usize), j))
            .collect(),
        offset: u64s(s.get(&format!("{p}offset"))?.0),
        checksum: u64s(s.get(&format!("{p}checksum"))?.0),
        chan: s.get(&format!("{p}chan"))?.0,
        block: s.get(&format!("{p}block"))?.0,
    }))
}

fn unit(r: &Rows, l: usize, k: u8, e: usize, h: usize, p: usize) -> Option<Unit> {
    let j = *r.row.get(&(e, h))?;
    let i = j * 3 + p;
    let mut side = r.chan[i * 256..(i + 1) * 256].to_vec();
    side.extend_from_slice(&r.block[(e * 3 + p) * 96..(e * 3 + p + 1) * 96]);
    Some(Unit { layer: l, k, expert: e, half: h, proj: p, offset: r.offset[i], nbytes: HALF_PER_BIT * k as u64,
                checksum: r.checksum[i], side })
}

/// The units rank `rank` of `tp` loads, layer by layer. layers: all the manifest lists (or, with Widths, all the
/// slices hold) when None. A unit the manifest wants at a width with no slice row is an error.
pub fn units(slices_dir: &str, tp: usize, rank: usize, choice: &Choice, layers: Option<&[usize]>)
             -> Result<Vec<Unit>, String> {
    let mut slices: BTreeMap<u8, Slice> = BTreeMap::new();
    for k in [2u8, 3, 4] {
        let p = format!("{slices_dir}/tp{tp}/k{k}/rank{rank}.safetensors");
        if std::path::Path::new(&p).exists() {
            slices.insert(k, Slice::open(&p)?);
        }
    }
    if slices.is_empty() {
        return Err(format!("no index slices for TP {tp} rank {rank} under {slices_dir}"));
    }
    let all: Vec<usize> = match (layers, choice) {
        (Some(ls), _) => ls.to_vec(),
        (None, Choice::Manifest(m)) => {
            let mut v: Vec<usize> = m.get("layers").and_then(Json::obj).ok_or("the manifest lists no layers")?
                .keys().filter_map(|k| k.parse().ok()).collect();
            v.sort();
            v
        }
        (None, Choice::Widths(_)) => (0..256).filter(|l| slices.values().any(|s| s.has(&format!("L{l:03}.units")))).collect(),
    };
    let mut out = Vec::new();
    for l in all {
        let mut at: BTreeMap<u8, Rows> = BTreeMap::new();
        for (&k, s) in &slices {
            if let Some(r) = rows(s, l)? {
                at.insert(k, r);
            }
        }
        if at.is_empty() {
            return Err(format!("layer {l}: no index slice holds it"));
        }
        match choice {
            Choice::Widths(ks) => {
                for (&k, r) in &at {
                    if !ks.contains(&k) {
                        continue;
                    }
                    let owned: BTreeSet<(usize, usize)> = r.row.keys().copied().collect();
                    for &(e, h) in &owned {
                        for p in 0..3 {
                            out.extend(unit(r, l, k, e, h, p));
                        }
                    }
                }
            }
            Choice::Manifest(m) => {
                let w = widths(m, l)?.ok_or(format!("layer {l}: the manifest has no widths for it"))?;
                let owned: BTreeSet<(usize, usize)> = at.values().flat_map(|r| r.row.keys().copied()).collect();
                for &(e, h) in &owned {
                    if e == SHARED && w.sdown == 0 {
                        continue; // the manifest leaves the shared expert to the base pack
                    }
                    let ks = if e == SHARED {
                        [w.sgate[h], w.sup[h], w.sdown]
                    } else {
                        [w.gate.get(e).ok_or(format!("layer {l}: no widths for expert {e}"))?[h], w.up[e][h], w.down[e]]
                    };
                    for (p, &k) in ks.iter().enumerate() {
                        let r = at.get(&k).ok_or(format!("layer {l}: width {k} wanted, no index slice at k{k}"))?;
                        out.push(unit(r, l, k, e, h, p)
                            .ok_or(format!("layer {l}: expert {e} half {h} has no row in the k{k} slice"))?);
                    }
                }
            }
        }
    }
    Ok(out)
}

/// The indices of the units whose bytes in `v2_dir`'s width files do not match their xxhash64.
pub fn verify(v2_dir: &str, units: &[Unit]) -> Result<Vec<usize>, String> {
    let mut files: HashMap<(usize, u8), File> = HashMap::new();
    let mut bad = Vec::new();
    let mut buf = Vec::new();
    for (i, u) in units.iter().enumerate() {
        let path = format!("{v2_dir}/{}", width_file(u.layer, u.k));
        if !files.contains_key(&(u.layer, u.k)) {
            files.insert((u.layer, u.k), File::open(&path).map_err(|e| format!("{path}: {e}"))?);
        }
        buf.resize(u.nbytes as usize, 0);
        if files[&(u.layer, u.k)].read_exact_at(&mut buf, u.offset).is_err() {
            bad.push(i);
            continue;
        }
        let mut h = Xxh64::new(0);
        h.update(&buf);
        h.update(&u.side);
        if h.digest() != u.checksum {
            bad.push(i);
        }
    }
    Ok(bad)
}
