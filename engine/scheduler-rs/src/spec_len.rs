//! The draft length of each speculative round (spec_len.h `g6rt::SpecLen`, which the plain build's `--spec-adaptive`
//! keeps using): the planner's acceptance estimates and the choice of how many drafts a round verifies.
//!
//! A round with k drafts costs verify(k + 1 rows) plus the drafting, and is expected to commit
//! E(k) = 1 + sum over j = 1..k of c_1 * ... * c_j tokens, where c_j is the chance that draft j is accepted given that
//! drafts 1..j-1 were. Every rank makes the same choice from bit-identical inputs (the drafts' log-probabilities, the
//! accept counts and the fixed cost table), so the arithmetic here keeps the C++'s operation order.

/// most drafts in a round
pub const KX: usize = 7;
/// confidence buckets
pub const NB: usize = 8;
/// the most draft counts a policy may list. set_levels leaves at most KX, and the C++ keeps a vector
pub const LEVELS_MAX: usize = 16;

/// std::max on doubles: `a` unless `a < b` (a NaN `a` stays)
pub fn dmax(a: f64, b: f64) -> f64 {
    if a < b {
        b
    } else {
        a
    }
}

/// std::min on doubles: `a` unless `b < a`
pub fn dmin(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpecLen {
    pub on: bool,
    /// allowed draft counts, ascending: levels[..n_levels]
    pub levels: [i32; LEVELS_MAX],
    pub n_levels: usize,
    /// drafting cost paid once per round, ms
    pub draft_round: f64,
    /// drafting cost per draft after the first, ms
    pub draft_each: f64,
    /// verify ms by rows 1..8
    pub table: [f64; KX + 2],
    pub seed: [f64; KX + 1],
    /// per position: decayed rounds offered / accepted
    pub pn: [f64; KX + 1],
    pub pk: [f64; KX + 1],
    /// per (position, bucket)
    pub bn: [[f64; NB]; KX + 1],
    pub bk: [[f64; NB]; KX + 1],
    pub pos_decay: f64,
    pub bucket_decay: f64,
    /// pseudo-rounds of the fallback in each estimate
    pub prior: f64,
    pub rounds: i64,
    pub chosen: [i64; KX + 1],
    /// sums of the predicted and actual tokens per round
    pub pred_sum: f64,
    pub got_sum: f64,
}

impl Default for SpecLen {
    fn default() -> Self {
        SpecLen {
            on: false,
            levels: [0; LEVELS_MAX],
            n_levels: 0,
            draft_round: 0.0,
            draft_each: 0.0,
            table: [0.0, 43.0, 52.4, 60.4, 68.4, 76.1, 83.7, 91.1, 97.4],
            seed: [0.0, 0.75, 0.72, 0.70, 0.70, 0.70, 0.68, 0.68],
            pn: [0.0; KX + 1],
            pk: [0.0; KX + 1],
            bn: [[0.0; NB]; KX + 1],
            bk: [[0.0; NB]; KX + 1],
            pos_decay: 0.9,
            bucket_decay: 0.998,
            prior: 4.0,
            rounds: 0,
            chosen: [0; KX + 1],
            pred_sum: 0.0,
            got_sum: 0.0,
        }
    }
}

/// the log-probability of draft j (NaN when the drafts have none)
fn at(lp: Option<&[f32]>, j: usize) -> f32 {
    lp.map_or(f32::NAN, |l| l[j])
}

impl SpecLen {
    pub fn levels(&self) -> &[i32] {
        &self.levels[..self.n_levels]
    }

    pub fn bucket(lp: f32) -> usize {
        if !lp.is_finite() {
            return 0;
        }
        const EDGE: [f64; NB - 1] = [0.2, 0.4, 0.6, 0.75, 0.88, 0.95, 0.99];
        let p = (lp as f64).exp();
        let mut b = 0;
        while b < NB - 1 && p >= EDGE[b] {
            b += 1;
        }
        b
    }

    pub fn rate(&self, j: usize) -> f64 {
        (self.pk[j] + self.prior * self.seed[j]) / (self.pn[j] + self.prior)
    }

    /// c_j for a draft with log-probability lp (NaN: unknown, the position's rate)
    pub fn cond(&self, j: usize, lp: f32) -> f64 {
        let r = self.rate(j);
        if lp.is_nan() {
            return r;
        }
        let b = Self::bucket(lp);
        let (mut n, mut k) = (0.0, 0.0);
        for i in 1..=KX {
            n += self.bn[i][b];
            k += self.bk[i][b];
        }
        let pool = (k + self.prior * r) / (n + self.prior);
        (self.bk[j][b] + self.prior * pool) / (self.bn[j][b] + self.prior)
    }

    pub fn verify_ms(&self, rows: i32) -> f64 {
        self.table[rows.clamp(1, KX as i32 + 1) as usize]
    }

    pub fn draft_ms(&self, drafted: i32) -> f64 {
        self.draft_round + self.draft_each * (drafted - 1).max(0) as f64
    }

    /// E(k) / ms with lp[1..k] (lp may be None); drafted = drafts already paid for. e_out: E(k)
    pub fn ratio(&self, k: i32, lp: Option<&[f32]>, drafted: i32, e_out: Option<&mut f64>) -> f64 {
        let (mut e, mut s) = (1.0, 1.0);
        for j in 1..=k.max(0) as usize {
            s *= self.cond(j, at(lp, j));
            e += s;
        }
        if let Some(o) = e_out {
            *o = e;
        }
        e / (self.verify_ms(k + 1) + self.draft_ms(drafted.max(k)))
    }

    /// the allowed level <= n with the most expected tokens per ms (ties: fewer drafts); aborts with none, as the C++
    pub fn choose(&self, lp: Option<&[f32]>, n: i32, drafted: i32, e_out: Option<&mut f64>) -> i32 {
        let (mut best, mut br, mut be) = (0, -1.0, 1.0);
        for &k in self.levels() {
            if k > n {
                break;
            }
            let mut e = 0.0;
            let r = self.ratio(k, lp, drafted, Some(&mut e));
            if r > br * (1.0 + 1e-9) {
                br = r;
                best = k;
                be = e;
            }
        }
        if best == 0 {
            eprintln!("g6: spec_len: {n} drafts, below every level");
            std::process::abort();
        }
        if let Some(o) = e_out {
            *o = be;
        }
        best
    }

    /// MTP: after drafting n drafts (lp[1..n] known), is one more chain step worth it? It is when drafting up to the
    /// next level above n (the drafts after n at their positions' rates) gives a better ratio than the best level <= n
    #[allow(clippy::needless_range_loop)] // j indexes lp only up to n, and the rates past it
    pub fn more(&self, lp: &[f32], n: i32) -> bool {
        let Some(&next) = self.levels().iter().find(|&&k| k > n) else { return false };
        let mut now = 0.0;
        for &k in self.levels() {
            if k <= n {
                now = dmax(now, self.ratio(k, Some(lp), n, None));
            }
        }
        let (mut e, mut s) = (1.0, 1.0);
        for j in 1..=next.max(0) as usize {
            s *= if j as i32 <= n { self.cond(j, lp[j]) } else { self.rate(j) };
            e += s;
        }
        e / (self.verify_ms(next + 1) + self.draft_ms(next)) > now
    }

    /// FNV-1a over the bytes of lp[1..n] and k (the log lines' decision hash)
    pub fn hash(lp: Option<&[f32]>, n: i32, k: i32) -> u32 {
        let mut h: u32 = 2166136261;
        let mut add = |b: &[u8]| {
            for &x in b {
                h = (h ^ x as u32).wrapping_mul(16777619);
            }
        };
        if let Some(l) = lp {
            for x in &l[1..1 + n.max(0) as usize] {
                add(&x.to_ne_bytes());
            }
        }
        add(&k.to_ne_bytes());
        h
    }

    /// a round verified k drafts and accepted a of them. lp[1..k]: the drafts' log-probabilities (None: none)
    pub fn observe(&mut self, k: i32, a: i32, lp: Option<&[f32]>, predicted: f64) {
        for j in 1..=KX {
            self.pn[j] *= self.pos_decay;
            self.pk[j] *= self.pos_decay;
            for b in 0..NB {
                self.bn[j][b] *= self.bucket_decay;
                self.bk[j][b] *= self.bucket_decay;
            }
        }
        let mut j = 1;
        while j <= k && j <= a + 1 {
            let acc = if j <= a { 1.0 } else { 0.0 };
            let ju = j as usize;
            self.pn[ju] += 1.0;
            self.pk[ju] += acc;
            if let Some(l) = lp {
                let b = Self::bucket(l[ju]);
                self.bn[ju][b] += 1.0;
                self.bk[ju][b] += acc;
            }
            j += 1;
        }
        self.rounds += 1;
        self.chosen[k as usize] += 1;
        self.pred_sum += predicted;
        self.got_sum += (a + 1) as f64;
    }

    /// the levels in 1..KX, sorted and unique. false when none is
    pub fn set_levels(&mut self, v: &[i32]) -> bool {
        let mut l: Vec<i32> = v.iter().copied().filter(|&k| (1..=KX as i32).contains(&k)).collect();
        l.sort_unstable();
        l.dedup();
        self.n_levels = l.len();
        self.levels = [0; LEVELS_MAX];
        self.levels[..l.len()].copy_from_slice(&l);
        !l.is_empty()
    }

    /// a new estimate with this one's settings (policy, costs, seeds, decays, prior) and no rounds
    pub fn fresh(&self) -> SpecLen {
        SpecLen {
            on: self.on,
            levels: self.levels,
            n_levels: self.n_levels,
            draft_round: self.draft_round,
            draft_each: self.draft_each,
            table: self.table,
            seed: self.seed,
            pos_decay: self.pos_decay,
            bucket_decay: self.bucket_decay,
            prior: self.prior,
            ..SpecLen::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_and_rates() {
        assert_eq!(SpecLen::bucket(f32::NAN), 0);
        assert_eq!(SpecLen::bucket(f32::NEG_INFINITY), 0);
        assert_eq!(SpecLen::bucket(0.0), NB - 1);
        assert_eq!(SpecLen::bucket((0.5f64).ln() as f32), 2);
        let s = SpecLen::default();
        assert_eq!(s.rate(1), 0.75);
        assert_eq!(s.cond(1, f32::NAN), 0.75);
    }

    #[test]
    fn learns_and_chooses() {
        let mut s = SpecLen::default();
        assert!(s.set_levels(&[7, 2, 2, 0, 9, 4]));
        assert_eq!(s.levels(), &[2, 4, 7]);
        let lp = [0.0f32; 8];
        for _ in 0..50 {
            s.observe(7, 7, Some(&lp), 0.0);
        }
        assert_eq!(s.choose(Some(&lp), 7, 7, None), 7);
        let f = s.fresh();
        assert_eq!(f.rounds, 0);
        assert_eq!(f.levels(), s.levels());
        assert_eq!(f.pn, [0.0; KX + 1]);
    }

    #[test]
    fn hash_is_fnv() {
        assert_eq!(SpecLen::hash(None, 0, 0), {
            let mut h: u32 = 2166136261;
            for _ in 0..4 {
                h = h.wrapping_mul(16777619);
            }
            h
        });
    }
}
