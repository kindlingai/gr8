//! Flash's KDA state slots per session and their rollback (was engine/runtime/s2/flash_ring.h). Session q owns slots
//! `[q * per, (q + 1) * per)`, per = the ring (K + 1, or 1 without MTP), with a ring of its own. In a main step a verify
//! session's row r goes to slot base + r, other rows to base; the session's first row loads base + its ring's source,
//! so sessions sharing a step never touch each other's state.

/// One session's ring (s2/flash_ring.h FlashRing, the same layout): the last main step's first position, rows, and
/// whether it verified.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ring {
    pub first: i32,
    pub rows: i32,
    pub ring: bool,
}

impl Default for Ring {
    fn default() -> Self {
        Ring { first: -1, rows: 0, ring: false }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    OutsidePrevious, // the next main position is outside the previous step
    VerifyTooWide,   // a verify step wider than the ring (or no rows)
}

impl Ring {
    /// The ring slot (relative to the session's base) the step at position `next` loads its state from.
    pub fn source(&self, next: i32) -> Result<i32, Error> {
        if self.first < 0 {
            return Ok(0);
        }
        let accepted = next - self.first;
        if accepted < 1 || accepted > self.rows {
            return Err(Error::OutsidePrevious);
        }
        Ok(if self.ring { accepted - 1 } else { 0 })
    }
    pub fn record(&mut self, pos: i32, n: i32, verify: bool) {
        *self = Ring { first: pos, rows: n, ring: verify };
    }
}

/// Session q's n rows from position pos in a main step, its ring `r` and `per` slots a session: each row's slot into
/// `kslot`, the slot the first row loads from returned, the step recorded for the next one.
pub fn rows(r: &mut Ring, per: i32, q: i32, pos: i32, n: i32, verify: bool, kslot: &mut [i32]) -> Result<i32, Error> {
    if n < 1 || (verify && n > per) {
        return Err(Error::VerifyTooWide);
    }
    let b = q * per;
    let src = b + r.source(pos)?;
    for (i, k) in kslot.iter_mut().take(n as usize).enumerate() {
        *k = b + if verify { i as i32 } else { 0 };
    }
    r.record(pos, n, verify);
    Ok(src)
}

/// The step shapes S2 can launch with several Flash sessions, each captured as a graph at every context bucket before
/// serving: main = prompt tails 1..chunk and rounds S x (k + 1), S = 1..sessions, k = 1..K with GR8_S2_ADAPT (else K);
/// draft = 1..chunk and 1..sessions x (K + 1); chain = 1..sessions (K >= 2 only). The flags are indexed by row count,
/// 0..=mx.
pub struct Shapes {
    pub mains: Vec<bool>,
    pub drafts: Vec<bool>,
    pub chains: Vec<bool>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapesError {
    Invalid,   // invalid sessions, mtp or chunk
    ExceedsMx, // sessions x (mtp + 1) rows exceed mx
}

pub fn shapes(sessions: i32, mtp: i32, chunk: i32, adapt: bool, mx: i32) -> Result<Shapes, ShapesError> {
    if sessions < 1 || mtp < 1 || chunk < 1 || chunk > mx {
        return Err(ShapesError::Invalid);
    }
    let top = (mx.max(sessions * (mtp + 1)) + 1) as usize;
    let mut f = Shapes { mains: vec![false; top], drafts: vec![false; top], chains: vec![false; top] };
    for m in 1..=chunk as usize {
        f.mains[m] = true;
        f.drafts[m] = true;
    }
    for s in 1..=sessions {
        for k in (if adapt { 1 } else { mtp })..=mtp {
            f.mains[(s * (k + 1)) as usize] = true;
        }
    }
    for m in 1..=(sessions * (mtp + 1)) as usize {
        f.drafts[m] = true;
    }
    if mtp >= 2 {
        for m in 1..=sessions as usize {
            f.chains[m] = true;
        }
    }
    let largest = |v: &[bool]| v.iter().rposition(|&b| b).unwrap_or(0) as i32;
    if largest(&f.mains) > mx || largest(&f.drafts) > mx {
        return Err(ShapesError::ExceedsMx);
    }
    for v in [&mut f.mains, &mut f.drafts, &mut f.chains] {
        v.truncate(mx as usize + 1);
    }
    Ok(f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_rolls_back_to_the_accepted_row() {
        let mut r = Ring::default();
        let mut k = [0; 4];
        assert_eq!(rows(&mut r, 4, 2, 100, 4, true, &mut k), Ok(8)); // first step: the base
        assert_eq!(k, [8, 9, 10, 11]);
        assert_eq!(rows(&mut r, 4, 2, 103, 1, false, &mut k), Ok(10)); // 3 accepted: slot base + 2
        assert_eq!(k[0], 8);
        assert_eq!(rows(&mut r, 4, 2, 104, 4, true, &mut k), Ok(8)); // after a non-verify step: the base
        assert_eq!(rows(&mut r, 4, 2, 200, 1, false, &mut k), Err(Error::OutsidePrevious));
        assert_eq!(rows(&mut r, 4, 2, 105, 5, true, &mut k), Err(Error::VerifyTooWide));
    }

    #[test]
    fn capture_set() {
        let f = shapes(2, 3, 4, false, 32).unwrap();
        let on = |v: &[bool]| (0..v.len()).filter(|&i| v[i]).collect::<Vec<_>>();
        assert_eq!(on(&f.mains), vec![1, 2, 3, 4, 8]);
        assert_eq!(on(&f.drafts), (1..=8).collect::<Vec<_>>());
        assert_eq!(on(&f.chains), vec![1, 2]);
        assert_eq!(shapes(8, 3, 4, false, 16).err(), Some(ShapesError::ExceedsMx));
        assert_eq!(shapes(1, 0, 4, false, 16).err(), Some(ShapesError::Invalid));
    }
}
