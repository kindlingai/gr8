//! The n-gram table's lookups (ngram_lookup.h `g6rt::NgramTable`): the C++ parses the file and hands the entries over
//! (`g6s2_batch_ngram`); the planner chains proposals from it here. A proposed token extends the context for the next
//! lookup, so one hit can propose a run.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// calibration buckets (edges as SpecLen::bucket)
pub const NB: usize = 8;
/// longest context length a table holds
pub const NMAX: usize = 32;

/// The keys are already 64-bit FNV hashes of (n, tokens): one multiply spreads them over the map's buckets.
#[derive(Default)]
pub struct KeyHasher(u64);
impl Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, b: &[u8]) {
        for &x in b {
            self.0 = (self.0 ^ x as u64).wrapping_mul(0x100000001b3);
        }
    }
    fn write_u64(&mut self, v: u64) {
        self.0 = (v ^ (v >> 29)).wrapping_mul(0xBF58476D1CE4E5B9);
    }
}

#[derive(Default)]
pub struct NgramTable {
    /// context lengths, longest first
    pub ns: Vec<i32>,
    pub has_cal: bool,
    pub cal: [f64; NB],
    pub t: HashMap<u64, (i32, f64), BuildHasherDefault<KeyHasher>>,
}

impl NgramTable {
    pub fn bucket(p: f64) -> usize {
        const EDGE: [f64; NB - 1] = [0.2, 0.4, 0.6, 0.75, 0.88, 0.95, 0.99];
        let mut b = 0;
        while b < NB - 1 && p >= EDGE[b] {
            b += 1;
        }
        b
    }

    /// the C++'s key: FNV-1a over n and each token's four bytes, low byte first
    pub fn key(ctx: &[i32]) -> u64 {
        let mut h = 1469598103934665603u64 ^ ctx.len() as u64;
        for &t in ctx {
            let v = t as u32;
            for b in 0..4 {
                h ^= ((v >> (8 * b)) & 0xff) as u64;
                h = h.wrapping_mul(1099511628211);
            }
        }
        h
    }

    /// the calibrated chance that a proposed token at stored probability p is right, given its prefix was
    pub fn conf(&self, p: f64) -> f64 {
        if self.has_cal {
            self.cal[Self::bucket(p)]
        } else {
            p
        }
    }

    pub fn nmax(&self) -> usize {
        self.ns.first().map_or(0, |&n| n as usize)
    }

    /// the next token after ctx at the longest context length with an entry
    pub fn next(&self, ctx: &[i32]) -> Option<(i32, f64)> {
        for &n in &self.ns {
            let n = n as usize;
            if ctx.len() < n {
                continue;
            }
            if let Some(&e) = self.t.get(&Self::key(&ctx[ctx.len() - n..])) {
                return Some(e);
            }
        }
        None
    }

    /// a chain of at most kmax tokens after tokens: out[1..=k], their calibrated conditionals c[1..=k]; returns k
    pub fn chain(&self, tokens: &[i32], kmax: usize, out: &mut [i32], c: &mut [f64]) -> usize {
        let nm = self.nmax();
        if nm == 0 || kmax < 1 {
            return 0;
        }
        let mut w = [0i32; NMAX + 16];
        let start = tokens.len().saturating_sub(nm);
        let mut len = tokens.len() - start;
        w[..len].copy_from_slice(&tokens[start..]);
        let mut k = 0;
        while k < kmax && len < w.len() {
            let Some((tok, p)) = self.next(&w[..len]) else { break };
            k += 1;
            out[k] = tok;
            c[k] = self.conf(p);
            w[len] = tok;
            len += 1;
        }
        k
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chains_with_backoff() {
        let mut t = NgramTable { ns: vec![3, 1], has_cal: false, ..Default::default() };
        t.t.insert(NgramTable::key(&[1, 2, 3]), (4, 0.9));
        t.t.insert(NgramTable::key(&[4]), (5, 0.5));
        let (mut out, mut c) = ([0; 8], [0.0; 8]);
        let k = t.chain(&[9, 1, 2, 3], 7, &mut out, &mut c);
        assert_eq!(k, 2);
        assert_eq!(&out[1..3], &[4, 5]);
        assert_eq!(&c[1..3], &[0.9, 0.5]);
        t.has_cal = true;
        t.cal = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        assert_eq!(t.conf(0.9), 0.6);
    }
}
