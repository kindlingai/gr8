//! XXH64, the checksum the exlr8 v2 index files carry for every expert half (SPEC.md 6.4). The reference algorithm,
//! streaming, so one hash can cover the half's trellis bytes, its per-channel values and the block scales in turn.

const P1: u64 = 11400714785074694791;
const P2: u64 = 14029467366897019727;
const P3: u64 = 1609587929392839161;
const P4: u64 = 9650029242287828579;
const P5: u64 = 2870177450012600261;

fn round(acc: u64, lane: u64) -> u64 {
    acc.wrapping_add(lane.wrapping_mul(P2)).rotate_left(31).wrapping_mul(P1)
}

fn merge(h: u64, v: u64) -> u64 {
    (h ^ round(0, v)).wrapping_mul(P1).wrapping_add(P4)
}

fn lane(b: &[u8]) -> u64 {
    u64::from_le_bytes(b[..8].try_into().unwrap())
}

pub struct Xxh64 {
    seed: u64,
    v: [u64; 4],
    buf: [u8; 32],
    nbuf: usize,
    total: u64,
}

impl Xxh64 {
    pub fn new(seed: u64) -> Xxh64 {
        Xxh64 { seed, v: [seed.wrapping_add(P1).wrapping_add(P2), seed.wrapping_add(P2), seed, seed.wrapping_sub(P1)],
                buf: [0; 32], nbuf: 0, total: 0 }
    }

    fn stripe(&mut self, s: &[u8]) {
        for (i, v) in self.v.iter_mut().enumerate() {
            *v = round(*v, lane(&s[i * 8..]));
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.total += data.len() as u64;
        if self.nbuf > 0 {
            let take = (32 - self.nbuf).min(data.len());
            self.buf[self.nbuf..self.nbuf + take].copy_from_slice(&data[..take]);
            self.nbuf += take;
            data = &data[take..];
            if self.nbuf < 32 {
                return;
            }
            let b = self.buf;
            self.stripe(&b);
            self.nbuf = 0;
        }
        while data.len() >= 32 {
            self.stripe(&data[..32]);
            data = &data[32..];
        }
        self.buf[..data.len()].copy_from_slice(data);
        self.nbuf = data.len();
    }

    pub fn digest(&self) -> u64 {
        let mut h = if self.total >= 32 {
            let [a, b, c, d] = self.v;
            let mut h = a.rotate_left(1).wrapping_add(b.rotate_left(7)).wrapping_add(c.rotate_left(12))
                .wrapping_add(d.rotate_left(18));
            for x in self.v {
                h = merge(h, x);
            }
            h
        } else {
            self.seed.wrapping_add(P5)
        };
        h = h.wrapping_add(self.total);
        let mut t = &self.buf[..self.nbuf];
        while t.len() >= 8 {
            h = (h ^ round(0, lane(t))).rotate_left(27).wrapping_mul(P1).wrapping_add(P4);
            t = &t[8..];
        }
        if t.len() >= 4 {
            h = (h ^ (u32::from_le_bytes(t[..4].try_into().unwrap()) as u64).wrapping_mul(P1)).rotate_left(23)
                .wrapping_mul(P2).wrapping_add(P3);
            t = &t[4..];
        }
        for &b in t {
            h = (h ^ (b as u64).wrapping_mul(P5)).rotate_left(11).wrapping_mul(P1);
        }
        h ^= h >> 33;
        h = h.wrapping_mul(P2);
        h ^= h >> 29;
        h = h.wrapping_mul(P3);
        h ^ (h >> 32)
    }
}

/// XXH64 of the parts as one stream.
pub fn xxh64(parts: &[&[u8]], seed: u64) -> u64 {
    let mut h = Xxh64::new(seed);
    for p in parts {
        h.update(p);
    }
    h.digest()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vectors() {
        // from the xxHash reference (xxhsum -H64)
        assert_eq!(xxh64(&[b""], 0), 0xEF46DB3751D8E999);
        assert_eq!(xxh64(&[b"a"], 0), 0xD24EC4F1A98C6E5B);
        assert_eq!(xxh64(&[b"abc"], 0), 0x44BC2CF5AD770999);
        assert_eq!(xxh64(&[b"Nobody inspects the spammish repetition"], 0), 0xFBCEA83C8A378BF1);
        assert_eq!(xxh64(&[b""], 1), 0xD5AFBA1336A3BE4B);
    }

    #[test]
    fn streaming_equals_one_shot() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 7 + i / 13) as u8).collect();
        for cut in [0, 1, 7, 31, 32, 33, 64, 500, 999, 1000] {
            for n in [0, 5, 31, 32, 100, 1000] {
                let d = &data[..n];
                let c = cut.min(n);
                assert_eq!(xxh64(&[&d[..c], &d[c..]], 3), xxh64(&[d], 3), "n {n} cut {cut}");
            }
        }
    }
}
