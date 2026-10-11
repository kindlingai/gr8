//! The part of g6rt::PrefixReuse (engine/runtime/prefix_reuse.h) that match() runs on a retained run: begin()'s
//! decision. A session's row tracking stays in C++, and retain() takes its finished state. Both copies of the rule must
//! agree: the differential test (engine/runtime/s2/tests/pool_diff_test.cpp) runs the C++ begin() beside this one.

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
    let h = &st.history;
    let lcp = h.iter().zip(prompt).take_while(|(a, b)| a == b).count() as i64;
    let mut skip: i64 = 0;
    let mut why;
    if h.is_empty() {
        why = Why::First;
    } else if rules.strict && !(prompt.len() > h.len() && lcp == h.len() as i64) {
        why = Why::Strict;
    } else {
        skip = (st.valid as i64).min(if mtp { lcp - 1 } else { lcp }).min(limit as i64);
        if skip <= 0 {
            skip = 0;
            why = Why::NoLcp;
        } else if !rules.noend && (prompt.len() as i64) < st.written_end as i64 {
            skip = 0;
            why = Why::Short;
        } else {
            why = Why::Hit;
        }
    }
    if skip != 0 && rules.grid > 0 {
        let (pf, grid) = (st.pf_end as i64, rules.grid as i64);
        skip = if skip >= pf && limit as i64 == pf { pf } else { skip.min(pf) / grid * grid };
        if skip == 0 {
            why = Why::Grid;
        }
    }
    Begin { skip: skip as i32, lcp: lcp as i32, why }
}
