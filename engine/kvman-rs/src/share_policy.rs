//! The value-weighted share policy (D-331 M6 step 2): which group on a KV node gives segments to a group that is short,
//! priced in request-seconds instead of by the age of the coldest blocks alone. Pure: the node (remote/server.rs) feeds
//! it what the heads REPORTed, and in KV_SHARE_VALUE=cost-shadow only logs what it would do beside the age rule; in
//! KV_SHARE_VALUE=cost the node acts on it (M6 step 3).
//!
//! For a give of `j` segments a rank by donor d to requester r, short `short` segments in all:
//!   benefit = p_low(r) x GiB r gets x seconds_per_GiB(r) x weight(r)
//!   loss    = p_high(d) x GiB d's quote says the give loses x seconds_per_GiB(d) x weight(d)
//!   net     = benefit - loss (transfer cost: 0 until it is measured)
//! - p comes from the head's ghost ledger (stash.rs): entries the tiers dropped, and how many came back within the
//!   horizon. p_low = returned / (dropped + 1) prices what a requester gains, p_high = (returned + pending + 1) /
//!   (dropped + 1) what a donor loses: with no history a requester gains nothing and a donor loses all, so uncertainty
//!   never pays for a destructive give.
//! - A requester that writes and never reads back (no returns) gains nothing, so it causes no give that loses rows.
//! - A give buys something only if it covers the shortfall: a large incoming run is priced as a bundle.
//! - seconds_per_GiB unknown (0): such a group gains nothing as requester and never gives destructively.
//! - A destructive give (loss > 0) needs benefit >= margin x loss and two favourable reports in a row; a free give
//!   (quote 0) needs neither. No give back the other way within `reverse_ms` of a handoff; at most `max_bytes_min`
//!   bytes moved a minute.

use std::collections::{HashMap, VecDeque};

const GIB: f64 = (1u64 << 30) as f64;

#[derive(Clone, Debug)]
pub struct Cfg {
    /// a destructive give needs benefit >= margin_milli / 1000 x loss (KV_SHARE_MARGIN, 1.25)
    pub margin_milli: u64,
    /// favourable reports in a row before a destructive give
    pub favourable: u32,
    /// no give from r to d this long after d gave to r
    pub reverse_ms: u64,
    /// bytes moved between groups a minute at most
    pub max_bytes_min: u64,
}

impl Default for Cfg {
    fn default() -> Cfg {
        Cfg { margin_milli: 1250, favourable: 2, reverse_ms: 60_000, max_bytes_min: 8 << 30 }
    }
}

/// What the node knows of one group at a report.
#[derive(Clone, Debug, Default)]
pub struct View {
    pub name: String,
    pub nr: u64,
    pub block_bytes: u64,
    pub seg_bytes: u64,
    pub segs: u64,
    pub floor: u64,
    /// 1000 = 1x
    pub weight: u64,
    /// recompute ms a GiB of this group's KV costs (KV_SHARE_SPG_*; 0 = unknown)
    pub spg_ms: u64,
    /// lost blocks a rank for a give of 1, 2, ... segments a rank (Pool::quote_points; u64::MAX = cannot give that)
    pub quotes: Vec<u64>,
    /// the head's ghost ledger: entries dropped, of them returned, and still within the horizon
    pub dropped: u64,
    pub returned: u64,
    pub pending: u64,
    /// may be asked now (complete, no transaction, not stalled, past its cooldown)
    pub eligible: bool,
}

impl View {
    fn p_low(&self) -> f64 {
        self.returned as f64 / (self.dropped + 1) as f64
    }
    fn p_high(&self) -> f64 {
        ((self.returned + self.pending + 1) as f64 / (self.dropped + 1) as f64).min(1.0)
    }
    fn w(&self) -> f64 {
        self.weight.max(1) as f64 / 1000.0
    }
}

/// One candidate give, in ms of request time.
#[derive(Clone, Debug, PartialEq)]
pub struct Score {
    pub give: u64,
    pub freed: u64,
    pub benefit_ms: f64,
    pub loss_ms: f64,
    pub net_ms: f64,
}

/// Prices donor `d` giving `give` segments a rank to `r`, which is `short` segments short in all. None: d cannot give
/// that many (floor, no quote).
pub fn score_transfer(r: &View, d: &View, give: u64, short: u64) -> Option<Score> {
    if give == 0 || give > d.segs.saturating_sub(d.floor) {
        return None;
    }
    let lost = *d.quotes.get(give as usize - 1)?;
    if lost == u64::MAX {
        return None;
    }
    let freed = give * d.nr;
    let benefit_ms = if freed >= short && r.spg_ms > 0 {
        r.p_low() * (short * r.seg_bytes) as f64 / GIB * r.spg_ms as f64 * r.w()
    } else {
        0.0
    };
    let loss_ms = if lost == 0 {
        0.0
    } else if d.spg_ms == 0 {
        f64::INFINITY
    } else {
        d.p_high() * (lost * d.block_bytes * d.nr) as f64 / GIB * d.spg_ms as f64 * d.w()
    };
    Some(Score { give, freed, benefit_ms, loss_ms, net_ms: benefit_ms - loss_ms })
}

/// The best give for `r` among `others`: free gives first (no loss), then the highest net that passes the margin. Ties:
/// fewer segments freed, then the group name.
pub fn select_transfer(r: &View, others: &[View], short: u64, cfg: &Cfg) -> Option<(String, Score)> {
    let mut best: Option<(String, Score)> = None;
    for d in others.iter().filter(|d| d.name != r.name && d.eligible) {
        for give in 1..=d.quotes.len() as u64 {
            let Some(s) = score_transfer(r, d, give, short) else { continue };
            if s.freed < short {
                continue;
            }
            let ok =
                s.loss_ms == 0.0 || (s.net_ms > 0.0 && s.benefit_ms * 1000.0 >= s.loss_ms * cfg.margin_milli as f64);
            if !ok {
                continue;
            }
            let better = match &best {
                None => true,
                Some((bn, b)) => {
                    let key = |s: &Score| (s.loss_ms == 0.0, s.net_ms);
                    let (a, c) = (key(&s), key(b));
                    a.0 > c.0
                        || (a.0 == c.0 && a.1 > c.1)
                        || (a == c && (s.freed < b.freed || (s.freed == b.freed && d.name < *bn)))
                }
            };
            if better {
                best = Some((d.name.clone(), s));
            }
            break; // a larger give from the same donor only loses more: its first give that covers is its best
        }
    }
    best
}

/// What the policy decided at one report.
#[derive(Clone, Debug, PartialEq)]
pub enum Decision {
    /// donor gives `Score::give` segments a rank
    Give(String, Score),
    /// a destructive give that passes, waiting for its next favourable report
    Wait(String, Score),
    /// a give that passes but a control holds it: "reverse" (within reverse_ms of the other way) or "rate"
    Held(String, Score, &'static str),
    /// no donor passes
    None,
}

/// The policy's memory between reports: favourable streaks, handoffs, bytes moved. In shadow it remembers what it
/// would have done, so its controls act on its own decisions.
#[derive(Debug, Default)]
pub struct Shadow {
    pub cfg: Cfg,
    streak: HashMap<(String, String), u32>,
    // (donor, requester, at ms)
    handoffs: Vec<(String, String, u64)>,
    moved: VecDeque<(u64, u64)>,
}

impl Shadow {
    pub fn new(cfg: Cfg) -> Shadow {
        Shadow { cfg, ..Default::default() }
    }

    /// `r` is `short` segments short at `now_ms`.
    pub fn decide(&mut self, now_ms: u64, r: &View, others: &[View], short: u64) -> Decision {
        let Some((d, s)) = select_transfer(r, others, short, &self.cfg) else {
            self.streak.retain(|k, _| k.1 != r.name);
            return Decision::None;
        };
        self.streak.retain(|k, _| k.1 != r.name || k.0 == d);
        let n = self.streak.entry((d.clone(), r.name.clone())).or_insert(0);
        *n += 1;
        if s.loss_ms > 0.0 && *n < self.cfg.favourable {
            return Decision::Wait(d, s);
        }
        let rw = self.cfg.reverse_ms;
        self.handoffs.retain(|h| now_ms.saturating_sub(h.2) < rw);
        if self.handoffs.iter().any(|h| h.0 == r.name && h.1 == d) {
            return Decision::Held(d, s, "reverse");
        }
        while self.moved.front().is_some_and(|m| now_ms.saturating_sub(m.0) >= 60_000) {
            self.moved.pop_front();
        }
        let bytes = s.freed * others.iter().find(|o| o.name == d).map_or(0, |o| o.seg_bytes);
        if self.moved.iter().map(|m| m.1).sum::<u64>() + bytes > self.cfg.max_bytes_min {
            return Decision::Held(d, s, "rate");
        }
        self.streak.remove(&(d.clone(), r.name.clone()));
        self.handoffs.push((d.clone(), r.name.clone(), now_ms));
        self.moved.push_back((now_ms, bytes));
        Decision::Give(d, s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEG: u64 = 1 << 30;

    fn g(name: &str, segs: u64, quotes: &[u64], dropped: u64, returned: u64) -> View {
        View {
            name: name.into(),
            nr: 2,
            block_bytes: 1 << 20,
            seg_bytes: SEG,
            segs,
            floor: 1,
            weight: 1000,
            spg_ms: 15_600,
            quotes: quotes.to_vec(),
            dropped,
            returned,
            pending: 0,
            eligible: true,
        }
    }

    #[test]
    fn a_free_give_wins_over_a_destructive_one_whatever_the_ages() {
        // w4: its first segments are empty; w3 would lose chat rows. The age rule picks by the coldest age; this
        // picks the give that loses nothing
        let r = g("full", 4, &[], 10, 8);
        let w3 = g("w3", 6, &[200, 400, 600], 10, 5);
        let w4 = g("w4", 6, &[0, 0, 300], 10, 5);
        let (d, s) = select_transfer(&r, &[w3, w4], 2, &Cfg::default()).unwrap();
        assert_eq!((d.as_str(), s.give, s.loss_ms), ("w4", 1, 0.0));
    }

    #[test]
    fn a_requester_that_never_reads_back_takes_no_rows_from_anyone() {
        let r = g("writer", 4, &[], 50, 0);
        let busy = g("chat", 6, &[300, 600], 10, 9);
        assert_eq!(select_transfer(&r, std::slice::from_ref(&busy), 2, &Cfg::default()), None);
        // a free give it still gets
        let idle = g("idle", 6, &[0, 0], 10, 9);
        assert_eq!(select_transfer(&r, &[busy, idle], 2, &Cfg::default()).unwrap().0, "idle");
    }

    #[test]
    fn a_give_that_does_not_cover_the_shortfall_buys_nothing() {
        let r = g("full", 4, &[], 10, 9);
        // short 6 in all: the donor's 2 ranks must give 3 segments each; it can give only 2 (floor 1)
        let d = g("d", 3, &[1, 2], 10, 0);
        assert_eq!(select_transfer(&r, std::slice::from_ref(&d), 6, &Cfg::default()), None);
        let d = g("d", 4, &[1, 2, 3], 10, 0);
        assert_eq!(select_transfer(&r, &[d], 6, &Cfg::default()).unwrap().1.give, 3);
    }

    #[test]
    fn unknown_or_new_groups_cause_no_destructive_give() {
        let d = g("chat", 6, &[300, 600], 10, 1);
        // late unknown group: no seconds_per_GiB
        let r = View { spg_ms: 0, ..g("flash", 1, &[], 10, 9) };
        assert_eq!(select_transfer(&r, std::slice::from_ref(&d), 2, &Cfg::default()), None);
        // late known group with no history yet: gains nothing
        let r = g("full2", 1, &[], 0, 0);
        assert_eq!(select_transfer(&r, std::slice::from_ref(&d), 2, &Cfg::default()), None);
        // with history that comes back, it takes rows a donor that rarely returns can spare
        let r = g("full2", 1, &[], 10, 9);
        assert_eq!(select_transfer(&r, &[d], 2, &Cfg::default()).unwrap().0, "chat");
        // a donor of unknown cost never gives destructively
        let d = View { spg_ms: 0, ..g("flash", 6, &[300], 10, 0) };
        assert_eq!(select_transfer(&r, &[d], 2, &Cfg::default()), None);
    }

    #[test]
    fn the_margin_holds_back_a_give_that_barely_pays() {
        // benefit 0.9 x 2 GiB x 15.6 s = 28.08 s; loss = p_high 1/11 x lost GiB x 15.6 s
        let r = g("full", 4, &[], 9, 9);
        let lost = |gib: f64| (gib * GIB / (2.0 * (1u64 << 20) as f64)) as u64;
        let pays = g("d", 6, &[lost(14.0)], 10, 0); // loss 19.9 s: 28.08 >= 1.25 x 19.9
        assert!(select_transfer(&r, &[pays], 2, &Cfg::default()).is_some());
        let barely = g("d", 6, &[lost(20.0)], 10, 0); // loss 28.4 s
        assert_eq!(select_transfer(&r, &[barely], 2, &Cfg::default()), None);
    }

    #[test]
    fn controls_two_reports_no_reversal_and_a_rate_cap() {
        let mut sh = Shadow::new(Cfg::default());
        let a = g("a", 4, &[50, 100], 10, 9);
        let b = g("b", 6, &[50, 100], 10, 1);
        // destructive: waits for the second favourable report
        assert!(matches!(sh.decide(0, &a, std::slice::from_ref(&b), 2), Decision::Wait(..)));
        assert!(matches!(sh.decide(1_000, &a, std::slice::from_ref(&b), 2), Decision::Give(ref d, _) if d == "b"));
        // the burst swaps: b is now short and a cold, but a just took from b: held for 60 s
        let a2 = g("a", 5, &[50, 100], 10, 1);
        let b2 = g("b", 5, &[50, 100], 10, 9);
        assert!(matches!(sh.decide(2_000, &b2, std::slice::from_ref(&a2), 2), Decision::Wait(..)));
        assert!(matches!(sh.decide(3_000, &b2, std::slice::from_ref(&a2), 2), Decision::Held(_, _, "reverse")));
        assert!(matches!(sh.decide(62_000, &b2, std::slice::from_ref(&a2), 2), Decision::Give(ref d, _) if d == "a"));
        // free gives skip the streak, but the rate cap holds: 2 GiB moved at 62 s, so 8 GiB more waits to 122 s
        let c = g("c", 10, &[0, 0, 0, 0], 10, 1);
        let big = g("big", 4, &[], 10, 9);
        assert!(matches!(sh.decide(63_000, &big, std::slice::from_ref(&c), 8), Decision::Held(_, _, "rate")));
        assert!(matches!(sh.decide(122_000, &big, std::slice::from_ref(&c), 8), Decision::Give(..)));
    }
}
