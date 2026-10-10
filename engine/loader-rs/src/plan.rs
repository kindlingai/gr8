//! Where each image byte comes from: pack-file extents (manifest tensors not covered by a source range) and the
//! source ranges of rank<r>.sources (engine/loader/g6pack.h, class Sources, documents the format).

use crate::G6lExtent;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Bytes of one v2 down half per bit of width: 48 x 8 x 8 tiles of 32 bytes.
const HALF_PER_BIT: u64 = 98304;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Op {
    Copy,
    Item128,
    /// bytes from Plan::mems[file]; file_off is unused
    Mem,
    /// bytes Plan::gens[file] computes at load. file_off is unused
    Gen,
}

/// Image bytes computed at load (decoded or re-laid-out weights). `make` returns the whole range. Workers copy the range
/// out in chunks, and the range is dropped after its last byte is copied. Each image byte is copied once, so a range is
/// computed once.
pub struct Gen {
    pub what: String,
    nbytes: u64,
    make: Box<dyn Fn() -> Result<Vec<u8>, String> + Send + Sync>,
    state: Mutex<Option<Arc<Vec<u8>>>>,
    left: AtomicU64,
}

impl Gen {
    pub fn new(what: String, nbytes: u64, make: Box<dyn Fn() -> Result<Vec<u8>, String> + Send + Sync>) -> Gen {
        Gen { what, nbytes, make, state: Mutex::new(None), left: AtomicU64::new(nbytes) }
    }

    /// The bytes, computing them on first use.
    pub fn get(&self) -> Result<Arc<Vec<u8>>, String> {
        let mut s = self.state.lock().unwrap();
        if let Some(a) = &*s {
            return Ok(a.clone());
        }
        if self.left.load(Ordering::Relaxed) < self.nbytes {
            return Err(format!("{}: computed twice", self.what));
        }
        let v = (self.make)().map_err(|e| format!("{}: {e}", self.what))?;
        if v.len() as u64 != self.nbytes {
            return Err(format!("{}: {} bytes computed, {} planned", self.what, v.len(), self.nbytes));
        }
        let a = Arc::new(v);
        *s = Some(a.clone());
        Ok(a)
    }

    /// The bytes, computed afresh and dropped by the caller. Single-tensor reads call this. They leave the rest of the
    /// range unread, so the count in `used` would never reach zero.
    pub fn compute(&self) -> Result<Vec<u8>, String> {
        (self.make)().map_err(|e| format!("{}: {e}", self.what))
    }

    /// `n` bytes were copied out. After the last of them the bytes are dropped.
    pub fn used(&self, n: u64) {
        if self.left.fetch_sub(n, Ordering::AcqRel) == n {
            *self.state.lock().unwrap() = None;
        }
    }
}

#[derive(Debug)]
pub struct Seg {
    pub pack_off: u64,
    pub nbytes: u64,
    pub op: Op,
    pub file_off: u64,
    pub file: usize,
}

pub struct Plan {
    /// files[0] is the pack itself; the rest are source files in order of first use
    pub files: Vec<String>,
    pub mems: Vec<Vec<u8>>,
    pub gens: Vec<Gen>,
    data: Vec<(u64, u64)>,
    segs: Vec<Seg>,
}

impl Plan {
    pub fn new(pack: &str, pack_bytes: u64, tensors: &[G6lExtent], sources: Option<&str>, root: Option<&str>)
               -> Result<Plan, String> {
        let mut files = vec![pack.to_string()];
        let segs = match sources {
            Some(p) => parse_sources(p, root, pack_bytes, &mut files)?,
            None => Vec::new(),
        };
        let mut ext: Vec<(u64, u64)> =
            tensors.iter().filter(|t| t.nbytes > 0).map(|t| (t.offset, t.offset + t.nbytes)).collect();
        ext.sort_unstable();
        if ext.last().is_some_and(|e| e.1 > pack_bytes) {
            return Err("a tensor runs past pack_bytes".into());
        }
        // tensors minus the source ranges: the bytes the pack file itself holds
        let mut data = Vec::new();
        let mut j = 0;
        for (mut x, y) in ext {
            while j < segs.len() && segs[j].pack_off + segs[j].nbytes <= x {
                j += 1;
            }
            let mut k = j;
            while x < y {
                if k < segs.len() && segs[k].pack_off < y {
                    let (s, e) = (segs[k].pack_off, segs[k].pack_off + segs[k].nbytes);
                    if s > x {
                        data.push((x, s));
                    }
                    x = x.max(e);
                    k += 1;
                } else {
                    data.push((x, y));
                    break;
                }
            }
        }
        Ok(Plan { files, mems: Vec::new(), gens: Vec::new(), data, segs })
    }

    pub fn empty() -> Plan {
        Plan { files: Vec::new(), mems: Vec::new(), gens: Vec::new(), data: Vec::new(), segs: Vec::new() }
    }

    /// A plan of segments only (files[0] is read through Copy segments like any other file). `segs` must be sorted
    /// and must not overlap.
    pub fn from_segs(files: Vec<String>, segs: Vec<Seg>, mems: Vec<Vec<u8>>, gens: Vec<Gen>, bytes: u64)
                     -> Result<Plan, String> {
        let mut end = 0;
        for s in &segs {
            if s.pack_off < end {
                return Err(format!("planned range at {} overlaps the one before", s.pack_off));
            }
            end = s.pack_off + s.nbytes;
        }
        if end > bytes {
            return Err("planned ranges run past the image".into());
        }
        Ok(Plan { files, mems, gens, data: Vec::new(), segs })
    }

    /// Pack-file extents that overlap image bytes a..b.
    pub fn pack_data(&self, a: u64, b: u64) -> &[(u64, u64)] {
        let i = self.data.partition_point(|e| e.1 <= a);
        let j = self.data.partition_point(|e| e.0 < b);
        &self.data[i..j.max(i)]
    }

    /// Source ranges that overlap image bytes a..b.
    pub fn sources(&self, a: u64, b: u64) -> &[Seg] {
        let i = self.segs.partition_point(|s| s.pack_off + s.nbytes <= a);
        let j = self.segs.partition_point(|s| s.pack_off < b);
        &self.segs[i..j.max(i)]
    }

    pub fn ranges(&self) -> usize {
        self.segs.len()
    }

    pub fn max_item128(&self) -> u64 {
        self.segs.iter().filter(|s| s.op == Op::Item128).map(|s| s.nbytes).max().unwrap_or(0)
    }
}

fn parse_sources(path: &str, root: Option<&str>, pack_bytes: u64, files: &mut Vec<String>) -> Result<Vec<Seg>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read sources {path}: {e}"))?;
    let mut lines = text.lines();
    let head: Vec<&str> = lines.next().unwrap_or("").splitn(2, ' ').collect();
    if head[0] != "g6sources-v1" {
        return Err(format!("{path}: not a g6sources-v1 file"));
    }
    let root = root.map(str::to_string).unwrap_or_else(|| head.get(1).unwrap_or(&"").trim().to_string());
    let mut index = std::collections::HashMap::new();
    let mut segs = Vec::new();
    let mut end = 0u64;
    for (n, line) in lines.enumerate().filter(|(_, l)| !l.is_empty()) {
        let bad = || format!("{path}:{}: bad line", n + 2);
        let f: Vec<&str> = line.split(' ').collect();
        if f.len() != 6 {
            return Err(bad());
        }
        let num = |i: usize| f[i].parse::<u64>().map_err(|_| bad());
        let (pack_off, nbytes, file_off, file_nbytes) = (num(0)?, num(1)?, num(3)?, num(4)?);
        let op = match f[2] {
            "copy" => Op::Copy,
            "item128" => Op::Item128,
            o => return Err(format!("{path}:{}: unknown op {o}", n + 2)),
        };
        if nbytes == 0 || nbytes != file_nbytes || (op == Op::Item128 && nbytes % HALF_PER_BIT != 0) {
            return Err(format!("{path}:{}: bad range size", n + 2));
        }
        if pack_off < end || pack_off + nbytes > pack_bytes {
            return Err(format!("{path}:{}: range overlaps or runs past pack_bytes", n + 2));
        }
        end = pack_off + nbytes;
        let file = *index.entry(f[5].to_string()).or_insert_with(|| {
            files.push(format!("{root}/{}", f[5]));
            files.len() - 1
        });
        segs.push(Seg { pack_off, nbytes, op, file_off, file });
    }
    Ok(segs)
}

/// v2 item128 down half -> g6 fc2 order: out[a][c][b] = in[a][b][c] for a < 48 and b, c < 8, tiles of n / 3072 bytes
/// (tools/convert/v2.py down_to_g6).
pub fn item128_to_g6(src: &[u8], dst: &mut [u8]) {
    let tb = src.len() / 3072;
    for a in 0..48 {
        for b in 0..8 {
            for c in 0..8 {
                let (o, i) = (((a * 8 + c) * 8 + b) * tb, ((a * 8 + b) * 8 + c) * tb);
                dst[o..o + tb].copy_from_slice(&src[i..i + tb]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn item128_matches_the_reference_permutation() {
        let n = 3072 * 32 * 2;
        let src: Vec<u8> = (0..n).map(|i| (i * 7 + i / 64) as u8).collect();
        let mut dst = vec![0u8; n];
        item128_to_g6(&src, &mut dst);
        let tb = n / 3072;
        // reshape(48, 8, 8, tb).transpose(0, 2, 1, 3): dst[a][c][b][t] = src[a][b][c][t]
        for (a, b, c, t) in [(0, 0, 1, 0), (47, 7, 2, tb - 1), (5, 3, 6, 11)] {
            assert_eq!(dst[((a * 8 + c) * 8 + b) * tb + t], src[((a * 8 + b) * 8 + c) * tb + t]);
        }
    }

    #[test]
    fn pack_data_excludes_source_ranges() {
        let d = std::env::temp_dir().join("g6load-plan-test.sources");
        std::fs::write(&d, "g6sources-v1 /r\n4096 4096 copy 0 4096 a\n12288 4096 copy 4096 4096 a\n").unwrap();
        let t = [G6lExtent { offset: 0, nbytes: 16384 }, G6lExtent { offset: 20480, nbytes: 100 }];
        let p = Plan::new("pack", 24576, &t, Some(d.to_str().unwrap()), None).unwrap();
        assert_eq!(p.data, vec![(0, 4096), (8192, 12288), (20480, 20580)]);
        assert_eq!(p.files, vec!["pack".to_string(), "/r/a".to_string()]);
        assert_eq!(p.sources(5000, 13000).len(), 2);
        assert_eq!(p.pack_data(4096, 8192).len(), 0);
    }
}
