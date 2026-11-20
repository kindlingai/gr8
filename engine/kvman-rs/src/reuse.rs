//! The part of g6rt::PrefixReuse (engine/runtime/prefix_reuse.h) that match() runs on a retained run: begin()'s
//! decision. A session's row tracking stays in C++, and retain() takes its finished state. Both copies of the rule must
//! agree.

use std::ffi::CStr;

/// The rules every retained run follows (PrefixReuse strict / noend / grid, set once by g6m4).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Rules {
    pub strict: bool,
    pub noend: bool,
    pub grid: i32,
}

/// A retired session's finished PrefixReuse state: the fields begin() reads.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Finished {
    pub history: Vec<i32>,
    pub valid: i32,
    pub written_end: i32,
    pub pf_end: i32,
}

/// Why a match() found what it found: PrefixReuse::why, plus the pool's own "none" and "room".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Why {
    #[default]
    None,
    Room,
    First,
    Strict,
    NoLcp,
    Short,
    Grid,
    Namespace,
    Hit,
    /// a hit on a stored run (stash.rs)
    Stored,
}

impl Why {
    pub fn name(self) -> &'static CStr {
        match self {
            Why::None => c"none",
            Why::Room => c"room",
            Why::First => c"first",
            Why::Strict => c"strict",
            Why::NoLcp => c"no-lcp",
            Why::Short => c"short",
            Why::Grid => c"grid",
            Why::Namespace => c"namespace",
            Why::Hit => c"hit",
            Why::Stored => c"stored",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Begin {
    pub skip: i32,
    pub lcp: i32,
    pub why: Why,
}

/// PrefixReuse::begin on a copy of `st`: the rows of `prompt` a session continuing it may skip (limit: rows its
/// prefill may skip), and the longest common prefix.
pub fn begin(rules: &Rules, st: &Finished, prompt: &[i32], limit: i32, mtp: bool) -> Begin {
    begin_on(rules, &st.history, st.valid, st.written_end, st.pf_end, prompt, limit, mtp, true)
}

/// begin() for a hit in the pool's own pages (GR8_KV_SHARE_PAGES, kvman-rs Pool::share_pages): the grid rule is off,
/// so the skip may end inside a page (skip = min(skip, pf_end) instead of its grid floor). The CUDA prefill starts at
/// any row; a tier keeps the grid (stash.rs storable), so stored hits still call begin().
pub fn begin_pool(rules: &Rules, st: &Finished, prompt: &[i32], limit: i32, mtp: bool) -> Begin {
    begin_on(rules, &st.history, st.valid, st.written_end, st.pf_end, prompt, limit, mtp, false)
}

/// begin() on the fields of a state: a live run's prompt and prefilled rows (Pool::prefilled) as well as a Finished.
/// grid = false: begin_pool's rule.
#[allow(clippy::too_many_arguments)]
pub fn begin_on(
    rules: &Rules,
    h: &[i32],
    valid: i32,
    written_end: i32,
    pf_end: i32,
    prompt: &[i32],
    limit: i32,
    mtp: bool,
    grid: bool,
) -> Begin {
    let lcp = h.iter().zip(prompt).take_while(|(a, b)| a == b).count() as i64;
    let mut skip: i64 = 0;
    let mut why;
    if h.is_empty() {
        why = Why::First;
    } else if rules.strict && !(prompt.len() > h.len() && lcp == h.len() as i64) {
        why = Why::Strict;
    } else {
        skip = (valid as i64).min(if mtp { lcp - 1 } else { lcp }).min(limit as i64);
        if skip <= 0 {
            skip = 0;
            why = Why::NoLcp;
        } else if !rules.noend && (prompt.len() as i64) < written_end as i64 {
            skip = 0;
            why = Why::Short;
        } else {
            why = Why::Hit;
        }
    }
    if skip != 0 && rules.grid > 0 {
        let (pf, g) = (pf_end as i64, rules.grid as i64);
        skip = if !grid || (skip >= pf && limit as i64 == pf) { skip.min(pf) } else { skip.min(pf) / g * g };
        if skip == 0 {
            why = Why::Grid;
        }
    }
    Begin { skip: skip as i32, lcp: lcp as i32, why }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_hit_vectors() {
        // begin_pool: the grid row of grid64_vectors without the floor (skip = min(skip, pf_end)), and the same
        // misses. (grid, mtp, pf_end, valid, written_end, hist_len, prompt_len, lcp, limit, want_skip, want_why)
        let cases = [
            (64, false, 15086, 15150, 15151, 15151, 19668, 15151, 19667, 15086, Why::Hit),
            (64, true, 15086, 15150, 15151, 15151, 19668, 15151, 19667, 15086, Why::Hit),
            (64, false, 999, 1030, 1031, 1031, 2192, 191, 2191, 191, Why::Hit),
            (64, true, 999, 1030, 1031, 1031, 2192, 191, 2191, 190, Why::Hit),
            (64, false, 999, 1030, 1031, 1031, 2000, 63, 1999, 63, Why::Hit),
            (64, true, 999, 1030, 1031, 1031, 2000, 64, 1999, 63, Why::Hit),
            (64, false, 30000, 30000, 30000, 30000, 30128, 30000, 30127, 30000, Why::Hit),
            (64, true, 30000, 30000, 30000, 30000, 30128, 30000, 30127, 29999, Why::Hit),
            (64, true, 30000, 30000, 30000, 30001, 30128, 30001, 30127, 30000, Why::Hit),
            (64, false, 0, 0, 0, 30000, 30128, 30000, 30127, 0, Why::NoLcp),
            (0, false, 999, 1030, 1031, 1031, 2192, 191, 2191, 191, Why::Hit),
        ];
        for (grid, mtp, pf_end, valid, written_end, hist_len, prompt_len, lcp, limit, want_skip, want_why) in cases {
            let rules = Rules { grid, strict: false, noend: false };
            let st = Finished { history: vec![1; hist_len], valid, written_end, pf_end };
            let mut prompt = vec![2; prompt_len];
            prompt[..lcp].fill(1);
            let b = begin_pool(&rules, &st, &prompt, limit, mtp);
            assert_eq!(b, Begin { skip: want_skip, lcp: lcp as i32, why: want_why }, "{grid} {mtp} {pf_end} {lcp}");
            // the grid rule never gives more than the pool rule
            assert!(begin(&rules, &st, &prompt, limit, mtp).skip <= b.skip);
        }
    }

    #[test]
    fn grid64_vectors() {
        // Shared vector table: keep fields and rows identical to tests/test_prefix_reuse.cpp.
        // grid, strict, noend, mtp, pf_end, valid, written_end, hist_len, prompt_len, lcp, limit,
        // want_skip, want_why
        let cases = [
            (64, false, false, false, 15086, 15150, 15151, 15151, 19668, 15151, 19667, 15040, Why::Hit),
            (64, false, false, true, 15086, 15150, 15151, 15151, 19668, 15151, 19667, 15040, Why::Hit),
            (64, false, false, false, 999, 1030, 1031, 1031, 2192, 191, 2191, 128, Why::Hit),
            (64, false, false, false, 999, 1030, 1031, 1031, 2192, 192, 2191, 192, Why::Hit),
            (64, false, false, false, 999, 1030, 1031, 1031, 2192, 193, 2191, 192, Why::Hit),
            (64, false, false, false, 999, 1030, 1031, 1031, 2192, 194, 2191, 192, Why::Hit),
            (64, false, false, true, 999, 1030, 1031, 1031, 2192, 191, 2191, 128, Why::Hit),
            (64, false, false, true, 999, 1030, 1031, 1031, 2192, 192, 2191, 128, Why::Hit),
            (64, false, false, true, 999, 1030, 1031, 1031, 2192, 193, 2191, 192, Why::Hit),
            (64, false, false, true, 999, 1030, 1031, 1031, 2192, 194, 2191, 192, Why::Hit),
            (64, false, false, false, 999, 1030, 1031, 1031, 2000, 63, 1999, 0, Why::Grid),
            (64, false, false, true, 999, 1030, 1031, 1031, 2000, 64, 1999, 0, Why::Grid),
            (64, false, false, false, 999, 1030, 1031, 1031, 2000, 64, 1999, 64, Why::Hit),
            (64, false, false, true, 999, 1030, 1031, 1031, 2000, 65, 1999, 64, Why::Hit),
            (64, false, true, false, 8099, 8162, 8163, 8163, 8100, 8100, 8099, 8099, Why::Hit),
            (64, false, true, true, 8099, 8162, 8163, 8163, 8100, 8100, 8099, 8099, Why::Hit),
            (64, false, true, false, 8099, 8162, 8163, 8163, 8090, 8090, 8089, 8064, Why::Hit),
            (64, false, true, true, 8099, 8162, 8163, 8163, 8090, 8090, 8089, 8064, Why::Hit),
        ];
        for (grid, strict, noend, mtp, pf_end, valid, written_end, hist_len, prompt_len, lcp, limit, want_skip, want_why) in cases {
            let rules = Rules { grid, strict, noend };
            let st = Finished { history: vec![1; hist_len], valid, written_end, pf_end };
            let mut prompt = vec![2; prompt_len];
            prompt[..lcp].fill(1);
            assert_eq!(begin(&rules, &st, &prompt, limit, mtp), Begin { skip: want_skip, lcp: lcp as i32, why: want_why });
        }
    }
}
