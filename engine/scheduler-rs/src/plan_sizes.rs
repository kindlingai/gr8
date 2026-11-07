//! The sizing rules the fabric plan reads (fabric_plan.rs), which step-time C++ reads too: the prefill fabrics' slots
//! (pf_slots.h), DS41's W-row prefill fabric (ds41_prefill_fabric.h, ds41_prefill_capacity.h, ds41_pf_split.h's tail
//! and split rows, ds41_rowshard.h, ds41_rs.h main_ncoll, ds41_lanes.h's knob), the QP wire's bulk area (qp_wire.h) and
//! the CP fabric's collectives (cplayout.h CpColls). The headers are facades over the C entries at the end, so every
//! reader shares this one copy. The build's constants come in as arguments from those headers.
//!
//! The arithmetic is the C++'s: size_t products wrap, and an int converted to size_t sign-extends first.

use std::ffi::{c_char, CStr, CString};

use crate::cstd::{c_strtol10, c_strtol10_end};

/// An option or message: bytes, whatever their encoding.
pub type Str = Vec<u8>;

/// (size_t)v for an int
pub fn sz(v: i32) -> usize {
    v as isize as usize
}

// ------------------------------------------------------------------------------------------------- prefill fabrics

pub fn pf_ss_min(mmax: i32, nr: i32, hid: i32) -> usize {
    (sz(mmax.wrapping_add(nr).wrapping_sub(1)) / sz(nr)).wrapping_mul(sz(hid)).wrapping_mul(2).wrapping_add(8 + 255)
        & !255usize
}

/// FabricConfig::slot_bytes of a prefill region
pub fn pf_slot_bytes(mmax: i32, rs: bool, nr: i32, hid: i32) -> usize {
    let ar = sz(mmax).wrapping_mul(sz(hid)).wrapping_mul(2);
    if rs {
        ar.max(sz(nr).wrapping_mul(pf_ss_min(mmax, nr, hid)))
    } else {
        ar
    }
}

/// Flash with GR8_GLM53F_PF_SP
pub fn pf_sp_slot_bytes(mp: i32, nr: i32, hid: i32) -> usize {
    let sp =
        (sz(mp.wrapping_add(nr).wrapping_sub(1)) / sz(nr)).wrapping_mul(sz(hid)).wrapping_mul(4).wrapping_add(8 + 255)
            & !255usize;
    pf_slot_bytes(mp, true, nr, hid).max(sp)
}

// ---------------------------------------------------------------------------------------- DS41 W-row prefill fabric

/// GR8_DS_PREFILL_W's largest value, and GR8_DS_PF_TAIL's
pub const DS41_TAIL_MAX: i32 = 4096;
pub const DS41_LANES_ERROR: &CStr = c"GR8_DS_PF_LANES must be 1 (one lane, the default) or 2 (two-lane W prefill)";

/// GR8_DS_PREFILL_W: the rows of a prefill chunk; unset = decode_rows, 0 = a bad value
pub fn ds41_prefill_rowcap(value: Option<&[u8]>, decode_rows: i32) -> i32 {
    let Some(v) = value else { return decode_rows };
    if v.is_empty() {
        return 0;
    }
    // strtol's ERANGE values fall outside the range below
    let (n, used, at_end) = c_strtol10_end(v);
    if used == 0 || !at_end || n < i64::from(decode_rows) || n > i64::from(DS41_TAIL_MAX) {
        return 0;
    }
    n as i32
}

/// the shared decode/prefill fabric's largest payload (Fabric::init adds the tail and the alignment)
pub fn ds41_prefill_payload(rows: i32, hidden: i32, engram_dim: i32) -> usize {
    sz(rows).wrapping_mul(sz(if engram_dim != 0 { engram_dim } else { hidden })).wrapping_mul(2)
}

/// GR8_DS_PF_TAIL: -1 = not a valid value
pub fn ds41_parse_tail(v: Option<&[u8]>, tail_rows: i32, window: i32) -> i32 {
    let Some(v) = v.filter(|v| !v.is_empty()) else { return tail_rows };
    let (t, at_end) = c_strtol10(v);
    let valid = t == 0 || (t >= i64::from(window) && t <= i64::from(DS41_TAIL_MAX));
    if !at_end || t < 0 || t > i64::from(DS41_TAIL_MAX) || !valid {
        return -1;
    }
    t as i32
}

pub fn ds41_tail_error(v: Option<&[u8]>, window: i32, late_layer: i32) -> Str {
    let mut s = b"GR8_DS_PF_TAIL=".to_vec();
    s.extend_from_slice(v.unwrap_or_default());
    s.extend_from_slice(
        format!(
            ": must be 0 (bounded replay off) or {window}..{DS41_TAIL_MAX} (rows of each prefill chunk that run the layers >= {late_layer})"
        )
        .as_bytes(),
    );
    s
}

/// rows of range B for a chunk of `rows` rows
pub fn ds41_late_rows(rows: i32, tail: i32) -> i32 {
    if tail == 0 || tail >= rows {
        rows
    } else {
        tail
    }
}

/// the first position with an id >= late_layer, -1 when there is none or it is position 0
pub fn ds41_late_start(ids: &[i32], late_layer: i32) -> i32 {
    match ids.iter().position(|&l| l >= late_layer) {
        Some(i) if i > 0 => i as i32,
        _ => -1,
    }
}

/// GR8_DS_PF_LANES: 1 or 2, 0 = a bad value
pub fn ds41_lanes(v: Option<&[u8]>) -> i32 {
    match v {
        None | Some(b"") | Some(b"1") => 1,
        Some(b"2") => 2,
        _ => 0,
    }
}

// ds41_rs.h's quarter and slice_stride stay C++ for the kernels; these are their copies (plan_twins_test checks them)
pub fn ds41_rs_quarter(rows: i32, nr: i32) -> i32 {
    (rows + nr - 1) / nr
}
pub fn ds41_rs_slice_stride(q: i32, row_bytes: usize) -> usize {
    sz(q).wrapping_mul(row_bytes).wrapping_add(8 + 255) & !255usize
}

/// the main fabric's collectives: decode + the RS all-gathers + the row-shard state collective
pub fn ds41_main_ncoll(nc: i32, k: i32, rowshard: bool) -> i32 {
    nc + k + i32::from(rowshard)
}

pub fn ds41_rowshard_ok(width: i32, tail: i32, nr: i32, rs: bool, fuse: bool) -> bool {
    width == 4096 && tail == 128 && nr == 4 && rs && fuse
}
pub fn ds41_state_row_bytes(hid: i32, trunk_bytes: usize) -> usize {
    4usize
        .wrapping_mul(sz(hid))
        .wrapping_mul(trunk_bytes)
        .wrapping_add((24 + 4 + 4 + 16) * 4)
        .wrapping_add(sz(hid).wrapping_mul(2))
}
pub fn ds41_state_bytes(late: i32, hid: i32, trunk_bytes: usize) -> usize {
    sz(late).wrapping_mul(ds41_state_row_bytes(hid, trunk_bytes))
}

/// g6plan_runs: ds41_prefill_fabric.h FabricRuns, the FabricConfig fields it sets
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Runs {
    pub run1: i32,
    pub run2: i32,
    pub run3: i32,
    pub ncoll: i32,
    pub slot_bytes: usize,
    pub slot_bytes1: usize,
    pub slot_bytes2: usize,
    pub slot_bytes3: usize,
    pub rs_k: i32,
    pub quarter: i32,
    pub slice_stride: usize,
}

/// The main fabric's slot runs for W-row prefill chunks
#[allow(clippy::too_many_arguments)]
pub fn ds41_fabric_runs(
    pf_rows: i32,
    tail: i32,
    i_late: i32,
    nl: i32,
    nx: i32,
    mx: i32,
    hid: i32,
    nr: i32,
    head_payload: usize,
    rs_on: bool,
    c8_pack: bool,
    rowshard: bool,
    pack_rows: i32,
) -> Runs {
    let mut o = Runs::default();
    let nc = 2 * nl + 1 + nx;
    let dec = sz(mx).wrapping_mul(sz(if nx != 0 { 6144 } else { hid })).wrapping_mul(2);
    let w_hid = sz(pf_rows).wrapping_mul(sz(hid)).wrapping_mul(2);
    let late_hid = sz(ds41_late_rows(pf_rows, tail)).wrapping_mul(sz(hid)).wrapping_mul(2);
    o.run1 = if i_late >= 0 { 2 * i_late } else { 2 * nl };
    o.run2 = 2 * nl + 1;
    o.slot_bytes = dec.max(w_hid);
    // a ragged pack carries up to pack_rows rows through the late layers
    let pack_late = if c8_pack && i_late >= 0 { sz(pack_rows).wrapping_mul(sz(hid)).wrapping_mul(2) } else { 0 };
    o.slot_bytes1 = dec.max(if i_late >= 0 { late_hid } else { 0 }).max(head_payload).max(pack_late);
    o.slot_bytes2 = if nx != 0 { dec.max(sz(pf_rows).wrapping_mul(6144 * 2)) } else { 0 };
    o.ncoll = nc;
    if rs_on {
        o.rs_k = 2 * i_late;
        o.quarter = ds41_rs_quarter(pf_rows, nr);
        o.slice_stride = ds41_rs_slice_stride(o.quarter, sz(hid).wrapping_mul(2));
        o.slot_bytes = o.slot_bytes.max(sz(nr).wrapping_mul(o.slice_stride).wrapping_sub(8));
        o.run3 = nc;
        o.slot_bytes3 = sz(o.quarter).wrapping_mul(sz(hid)).wrapping_mul(2);
        o.ncoll = ds41_main_ncoll(nc, o.rs_k, rowshard);
    }
    o
}

// ------------------------------------------------------------------------------------------------------- QP wire

pub fn qw_bulk_bytes(e: i32, half: usize) -> usize {
    sz(e).wrapping_mul(half.wrapping_mul(2))
}
pub fn qw_bulk_flags(e: i32) -> i32 {
    2 * e
}
pub fn qw_partner(r: i32) -> i32 {
    r ^ 1
}

// ---------------------------------------------------------------------------------------------- CP collectives

/// cplayout.h CpColls: per CP layer the q all-gather, the candidates all-gather (full-index layers), and nr - 1
/// single-destination partials, in publish order
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CpColls {
    pub nr: i32,
    pub nl: i32,
    pub mmax: i32,
    pub q_row: u32,
    pub cand_row: u32,
    pub part_row: u32,
    pub q: Vec<i32>,
    pub cand: Vec<i32>,
    pub part: Vec<i32>,
    pub cap: Vec<u32>,
    pub dst_k: Vec<u8>,
}

/// heads, kv_rank, qk_dim, topk: CpColls's HEADS, KV_RANK, QK_DIM and TOPK
pub fn cp_colls(full: &[bool], nr: i32, mmax: i32, heads: i32, kv_rank: i32, qk_dim: i32, topk: i32) -> CpColls {
    let nl = full.len() as i32;
    // rows sized for the widest head group, rounded to 16 B
    let hpr = ((heads + nr - 1) / nr) as u32;
    let mut c = CpColls {
        nr,
        nl,
        mmax,
        q_row: hpr.wrapping_mul(qk_dim as u32).wrapping_mul(4),
        cand_row: (topk as u32).wrapping_mul(8),
        part_row: hpr.wrapping_mul((kv_rank + 2) as u32).wrapping_mul(4).wrapping_add(15) & !15u32,
        q: vec![-1; full.len()],
        cand: vec![-1; full.len()],
        part: vec![-1; full.len() * (nr - 1).max(0) as usize],
        ..Default::default()
    };
    fn add(c: &mut CpColls, row: u32, k: i32) -> i32 {
        c.cap.push(row.wrapping_mul(c.mmax as u32));
        c.dst_k.push(k as u8);
        c.cap.len() as i32 - 1
    }
    let (q_row, cand_row, part_row) = (c.q_row, c.cand_row, c.part_row);
    for (i, &f) in full.iter().enumerate() {
        let at = add(&mut c, q_row, 0);
        c.q[i] = at;
        if f {
            let at = add(&mut c, cand_row, 0);
            c.cand[i] = at;
        }
        for k in 1..nr {
            let at = add(&mut c, part_row, k);
            c.part[i * (nr - 1) as usize + k as usize - 1] = at;
        }
    }
    c
}

// ------------------------------------------------------------------------------------------------------ the C ABI

/// # Safety
/// `p` is null or a NUL-terminated string.
pub unsafe fn text<'a>(p: *const c_char) -> Option<&'a [u8]> {
    (!p.is_null()).then(|| CStr::from_ptr(p).to_bytes())
}

unsafe fn slice<'a, T>(p: *const T, n: usize) -> &'a [T] {
    if n == 0 || p.is_null() {
        &[]
    } else {
        std::slice::from_raw_parts(p, n)
    }
}

/// A message for the C++, freed with g6plan_str_free.
pub fn c_owned(s: &[u8]) -> *mut c_char {
    crate::cstd::cs(s).into_raw()
}

#[no_mangle]
pub extern "C" fn g6plan_pf_ss_min(mmax: i32, nr: i32, hid: i32) -> usize {
    pf_ss_min(mmax, nr, hid)
}
#[no_mangle]
pub extern "C" fn g6plan_pf_slot_bytes(mmax: i32, rs: bool, nr: i32, hid: i32) -> usize {
    pf_slot_bytes(mmax, rs, nr, hid)
}
#[no_mangle]
pub extern "C" fn g6plan_pf_sp_slot_bytes(mp: i32, nr: i32, hid: i32) -> usize {
    pf_sp_slot_bytes(mp, nr, hid)
}

/// # Safety
/// `out` is valid.
#[no_mangle]
pub unsafe extern "C" fn g6plan_ds41_runs(
    pf_rows: i32,
    tail: i32,
    i_late: i32,
    nl: i32,
    nx: i32,
    mx: i32,
    hid: i32,
    nr: i32,
    head_payload: usize,
    rs_on: bool,
    c8_pack: bool,
    rowshard: bool,
    pack_rows: i32,
    out: *mut Runs,
) {
    *out =
        ds41_fabric_runs(pf_rows, tail, i_late, nl, nx, mx, hid, nr, head_payload, rs_on, c8_pack, rowshard, pack_rows);
}
#[no_mangle]
pub extern "C" fn g6plan_ds41_late_rows(rows: i32, tail: i32) -> i32 {
    ds41_late_rows(rows, tail)
}
/// # Safety
/// `ids` holds n values.
#[no_mangle]
pub unsafe extern "C" fn g6plan_ds41_late_start(ids: *const i32, n: usize, late_layer: i32) -> i32 {
    ds41_late_start(slice(ids, n), late_layer)
}
/// # Safety
/// `v` is null or a C string.
#[no_mangle]
pub unsafe extern "C" fn g6plan_ds41_parse_tail(v: *const c_char, tail_rows: i32, window: i32) -> i32 {
    ds41_parse_tail(text(v), tail_rows, window)
}
/// # Safety
/// As above. Free the result with g6plan_str_free.
#[no_mangle]
pub unsafe extern "C" fn g6plan_ds41_tail_error(v: *const c_char, window: i32, late_layer: i32) -> *mut c_char {
    c_owned(&ds41_tail_error(text(v), window, late_layer))
}
/// # Safety
/// `s` comes from this library, once.
#[no_mangle]
pub unsafe extern "C" fn g6plan_str_free(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}
/// # Safety
/// `v` is null or a C string.
#[no_mangle]
pub unsafe extern "C" fn g6plan_ds41_lanes(v: *const c_char) -> i32 {
    ds41_lanes(text(v))
}
#[no_mangle]
pub extern "C" fn g6plan_ds41_lanes_error() -> *const c_char {
    DS41_LANES_ERROR.as_ptr()
}
/// # Safety
/// `v` is null or a C string.
#[no_mangle]
pub unsafe extern "C" fn g6plan_ds41_prefill_rowcap(v: *const c_char, decode_rows: i32) -> i32 {
    ds41_prefill_rowcap(text(v), decode_rows)
}
#[no_mangle]
pub extern "C" fn g6plan_ds41_prefill_payload(rows: i32, hidden: i32, engram_dim: i32) -> usize {
    ds41_prefill_payload(rows, hidden, engram_dim)
}
#[no_mangle]
pub extern "C" fn g6plan_ds41_rowshard_ok(width: i32, tail: i32, nr: i32, rs: bool, fuse: bool) -> bool {
    ds41_rowshard_ok(width, tail, nr, rs, fuse)
}
#[no_mangle]
pub extern "C" fn g6plan_ds41_state_row_bytes(hid: i32, trunk_bytes: usize) -> usize {
    ds41_state_row_bytes(hid, trunk_bytes)
}
#[no_mangle]
pub extern "C" fn g6plan_ds41_state_bytes(late: i32, hid: i32, trunk_bytes: usize) -> usize {
    ds41_state_bytes(late, hid, trunk_bytes)
}
#[no_mangle]
pub extern "C" fn g6plan_ds41_main_ncoll(nc: i32, k: i32, rowshard: bool) -> i32 {
    ds41_main_ncoll(nc, k, rowshard)
}
#[no_mangle]
pub extern "C" fn g6plan_ds41_rs_quarter(rows: i32, nr: i32) -> i32 {
    ds41_rs_quarter(rows, nr)
}
#[no_mangle]
pub extern "C" fn g6plan_ds41_rs_slice_stride(q: i32, row_bytes: usize) -> usize {
    ds41_rs_slice_stride(q, row_bytes)
}
#[no_mangle]
pub extern "C" fn g6plan_ds41_tail_max() -> i32 {
    DS41_TAIL_MAX
}
#[no_mangle]
pub extern "C" fn g6plan_qw_bulk_bytes(e: i32, half: usize) -> usize {
    qw_bulk_bytes(e, half)
}
#[no_mangle]
pub extern "C" fn g6plan_qw_bulk_flags(e: i32) -> i32 {
    qw_bulk_flags(e)
}
#[no_mangle]
pub extern "C" fn g6plan_qw_partner(r: i32) -> i32 {
    qw_partner(r)
}

/// g6plan_cp_dims: CpColls's constants
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CpDims {
    pub heads: i32,
    pub kv_rank: i32,
    pub qk_dim: i32,
    pub topk: i32,
}

/// CpColls::make into the caller's buffers: rows[3] = q, cand and part row bytes; q and cand hold nl entries, part
/// nl (nr - 1), cap and dst_k nl (nr + 1). Returns the collectives.
///
/// # Safety
/// The buffers hold those counts; `full` holds nl.
#[no_mangle]
pub unsafe extern "C" fn g6plan_cp_colls(
    full: *const bool,
    nl: usize,
    nr: i32,
    mmax: i32,
    d: CpDims,
    rows: *mut u32,
    q: *mut i32,
    cand: *mut i32,
    part: *mut i32,
    cap: *mut u32,
    dst_k: *mut u8,
) -> usize {
    let c = cp_colls(slice(full, nl), nr, mmax, d.heads, d.kv_rank, d.qk_dim, d.topk);
    *rows = c.q_row;
    *rows.add(1) = c.cand_row;
    *rows.add(2) = c.part_row;
    unsafe fn put<T: Copy>(src: &[T], dst: *mut T) {
        if !src.is_empty() {
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
        }
    }
    put(&c.q, q);
    put(&c.cand, cand);
    put(&c.part, part);
    put(&c.cap, cap);
    put(&c.dst_k, dst_k);
    c.cap.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rules() {
        assert_eq!(ds41_prefill_rowcap(None, 48), 48);
        assert_eq!(ds41_prefill_rowcap(Some(b""), 48), 0);
        assert_eq!(ds41_prefill_rowcap(Some(b"4096"), 48), 4096);
        assert_eq!(ds41_prefill_rowcap(Some(b"4097"), 48), 0);
        assert_eq!(ds41_prefill_rowcap(Some(b"16"), 48), 0);
        assert_eq!(ds41_prefill_rowcap(Some(b"99999999999999999999"), 48), 0);
        assert_eq!(ds41_parse_tail(None, 128, 128), 128);
        assert_eq!(ds41_parse_tail(Some(b"0"), 128, 128), 0);
        assert_eq!(ds41_parse_tail(Some(b"5"), 128, 128), -1);
        assert_eq!(ds41_parse_tail(Some(b"256x"), 128, 128), -1);
        assert_eq!(ds41_lanes(Some(b"2")), 2);
        assert_eq!(ds41_lanes(Some(b"3")), 0);
        assert_eq!(ds41_late_start(&[0, 1, 21, 22], 21), 2);
        assert_eq!(ds41_late_start(&[21, 22], 21), -1);
        assert_eq!(pf_slot_bytes(4096, false, 4, 6144), 4096 * 6144 * 2);
        let c = cp_colls(&[true, false], 4, 24, 64, 512, 576, 2048);
        assert_eq!((c.q, c.cand, c.part.len(), c.cap.len()), (vec![0, 5], vec![1, -1], 6, 9));
    }
}
