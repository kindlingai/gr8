//! GR8_CP_STEP_ROWS and GR8_CP_STEP_SEG (was engine/runtime/s2/cp_rows.h): which sessions a round past R takes, so a
//! step past R carries at most n rows. The CP fabric and the CP graphs past R are sized for n rows; the round a past-R
//! session takes holds at most n / (mtp + 1) sessions, and the others sit that round out and go in the next one, in
//! cycles in which every decoding session gets exactly one round. A round without a past-R session keeps the full
//! batch. With SEG a past-R round never takes a session below R. Every input is replicated state, so every rank forms
//! the same rounds without exchanging them.

/// The contexts that run the CP block: ctx > cp_after(...) exactly when bk[bucket_of(ctx)] > R, bucket_of = the
/// smallest bucket >= ctx, clamped to the last. `bk` ascending, not empty. R < 0: no CP (never past). Returns the
/// largest bucket <= R (0 when every bucket is past R), or u64::MAX when no bucket is past R.
pub fn cp_after(bk: &[i32], r: i32) -> u64 {
    match bk.last() {
        Some(&last) if r >= 0 && last > r => bk.iter().take_while(|&&b| b <= r).last().map_or(0, |&b| b as u64),
        _ => u64::MAX,
    }
}

/// The boot value of v = GR8_CP_STEP_ROWS (0 = off), `solo` = the rows a CP step takes without S2 batching, `full` =
/// the widest S2 step. None: below solo, refused. Above full: clamped to full.
pub fn cp_step_rows(v: i64, solo: i32, full: i32) -> Option<i32> {
    if v == 0 {
        return Some(0);
    }
    if v < i64::from(solo) {
        return None;
    }
    Some(v.min(i64::from(full.max(solo))) as i32)
}

/// A decoding session: its immutable id, and whether its next round may run past R.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cand {
    pub id: u64,
    pub past: bool,
}

/// A round's choice: per candidate whether it is in this round; capped = a cycle round (the others sit it out), false
/// = everyone and the cycle resets; fresh = a capped round that starts a new cycle.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Pick {
    pub in_round: Vec<bool>,
    pub capped: bool,
    pub fresh: bool,
}

/// The round's sessions among `cand` (the decoding sessions that are not stalled, in request-id order), `served` the
/// ids served in the current cycle. rows = n (0 = off: everyone), per = the most rows one session's verify takes
/// (mtp + 1), seg = GR8_CP_STEP_SEG.
pub fn pick(cand: &[Cand], served: &[u64], rows: u32, per: u32, seg: bool) -> Pick {
    let mut p = Pick { in_round: vec![true; cand.len()], ..Default::default() };
    if rows == 0 || cand.is_empty() {
        return p;
    }
    let past = cand.iter().filter(|c| c.past).count();
    if past == 0 {
        return p;
    }
    let mixed = past < cand.len();
    if cand.len() as u64 * u64::from(per) <= u64::from(rows) && !(seg && mixed) {
        return p; // everyone fits in one step past R
    }
    p.capped = true;
    let mut open: Vec<usize> = (0..cand.len()).filter(|&i| !served.contains(&cand[i].id)).collect();
    if open.is_empty() {
        p.fresh = true;
        open = (0..cand.len()).collect();
    }
    p.in_round = vec![false; cand.len()];
    if !open.iter().any(|&i| cand[i].past) {
        for &i in &open {
            p.in_round[i] = true; // a step below R: every open session
        }
        return p;
    }
    let most = ((rows / per.max(1)) as usize).max(1);
    let mut n = 0;
    for &i in &open {
        if cand[i].past && n < most {
            p.in_round[i] = true;
            n += 1;
        }
    }
    if !seg {
        for &i in &open {
            if !cand[i].past && n < most {
                p.in_round[i] = true;
                n += 1;
            }
        }
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn after() {
        assert_eq!(cp_after(&[1024, 4096, 16384], 4096), 4096);
        assert_eq!(cp_after(&[1024, 4096, 16384], 512), 0);
        assert_eq!(cp_after(&[1024, 4096], 4096), u64::MAX);
        assert_eq!(cp_after(&[1024, 4096], -1), u64::MAX);
        assert_eq!(cp_after(&[], 5), u64::MAX);
    }

    #[test]
    fn step_rows() {
        assert_eq!(cp_step_rows(0, 4, 24), Some(0));
        assert_eq!(cp_step_rows(3, 4, 24), None);
        assert_eq!(cp_step_rows(9, 4, 24), Some(9));
        assert_eq!(cp_step_rows(99, 4, 24), Some(24));
    }

    #[test]
    fn cycles_take_past_sessions_first() {
        let c = |id, past| Cand { id, past };
        let cand = [c(1, true), c(2, false), c(3, true), c(4, false)];
        let p = pick(&cand, &[], 6, 3, false);
        assert!(p.capped && !p.fresh);
        assert_eq!(p.in_round, vec![true, false, true, false]);
        let p = pick(&cand, &[1, 3], 6, 3, false); // the past ones served: a step below R takes every open one
        assert_eq!(p.in_round, vec![false, true, false, true]);
        let p = pick(&cand, &[1, 2, 3, 4], 6, 3, true);
        assert!(p.fresh);
        assert_eq!(pick(&cand, &[], 12, 3, false).capped, false); // everyone fits
        assert_eq!(pick(&cand, &[], 12, 3, true).capped, true); // seg and mixed
    }
}
