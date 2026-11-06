//! The layout of every `#[repr(C)]` type the C ABI shares, as one digest: per type its size and alignment, per field
//! its offset, size, kind (unsigned, signed, float, bool, array of a kind, pointer to a kind, struct) and, for a
//! pointer, the size of what it points to. g6sched.h computes the same digest over its declarations
//! (g6sch_layout_cpp) and the C++ compares the two once at startup, so a field that moved, changed size or changed
//! signedness on one side fails there. Both lists follow the Rust field order; a C field with another name fails to
//! compile, and a Rust field missing from its list fails to compile here.
//!
//! It cannot see: which struct a nested struct field is (only its size), pointer constness, and the function
//! signatures in g6sched.h. kvman-rs's own header for its tier counters is outside it.

use std::mem::{align_of, offset_of, size_of, MaybeUninit};

trait Kind {
    const K: u64;
    /// a pointer's pointee size; 0 for everything else
    const PSIZE: u64 = 0;
}
macro_rules! kinds {
    ($k:expr => $($t:ty),*) => { $(impl Kind for $t { const K: u64 = $k; })* };
}
kinds!(1 => u8, u16, u32, u64, usize);
kinds!(2 => i8, i16, i32, i64);
kinds!(3 => f32, f64);
kinds!(4 => bool);
impl<T: Kind, const N: usize> Kind for [T; N] {
    const K: u64 = 0x50 | T::K;
}
impl<T: Kind> Kind for *const T {
    const K: u64 = 0x70 | T::K;
    const PSIZE: u64 = size_of::<T>() as u64;
}
impl<T: Kind> Kind for *mut T {
    const K: u64 = 0x70 | T::K;
    const PSIZE: u64 = size_of::<T>() as u64;
}

fn mix(h: &mut u64, v: u64) {
    for i in 0..8 {
        *h = (*h ^ ((v >> (8 * i)) & 0xff)).wrapping_mul(1099511628211);
    }
}
fn size<T>(_: *const T) -> u64 {
    size_of::<T>() as u64
}
fn kind<T: Kind>(_: *const T) -> u64 {
    T::K
}
fn psize<T: Kind>(_: *const T) -> u64 {
    T::PSIZE
}

macro_rules! layout {
    ($h:ident, $t:path { $($f:ident),* }) => {{
        // every field is listed: a missing one fails to compile here
        #[allow(dead_code)]
        fn all(x: $t) {
            let $t { $($f: _),* } = x;
        }
        let u = MaybeUninit::<$t>::uninit();
        let p = u.as_ptr();
        mix(&mut $h, size_of::<$t>() as u64);
        mix(&mut $h, align_of::<$t>() as u64);
        $(
            // SAFETY: only the field's address is formed, never read
            let f = unsafe { std::ptr::addr_of!((*p).$f) };
            mix(&mut $h, offset_of!($t, $f) as u64);
            mix(&mut $h, size(f));
            mix(&mut $h, kind(f));
            mix(&mut $h, psize(f));
        )*
    }};
}

impl Kind for crate::admission::Status {
    const K: u64 = 6;
}
impl Kind for crate::cp_rows::Cand {
    const K: u64 = 6;
}
impl Kind for crate::ffi::Batch {
    const K: u64 = 6;
}
impl Kind for crate::ffi::Ints {
    const K: u64 = 6;
}
impl Kind for crate::ffi::ParsedReq {
    const K: u64 = 6;
}
impl Kind for crate::flash_lend::Lend {
    const K: u64 = 6;
}
impl Kind for crate::flash_ring::Ring {
    const K: u64 = 6;
}
impl Kind for crate::kv_plan::Io {
    const K: u64 = 6;
}
impl Kind for crate::kv_plan::Item {
    const K: u64 = 6;
}
impl Kind for crate::kv_share::Totals {
    const K: u64 = 6;
}
impl Kind for crate::kv_share::Resize {
    const K: u64 = 6;
}
impl Kind for crate::scheduler::Key {
    const K: u64 = 6;
}
impl Kind for crate::scheduler::Identity {
    const K: u64 = 6;
}
impl Kind for crate::scheduler::Request {
    const K: u64 = 6;
}
impl Kind for crate::scheduler::Run {
    const K: u64 = 6;
}
impl Kind for crate::scheduler::Slot {
    const K: u64 = 6;
}
impl Kind for crate::scheduler::Config {
    const K: u64 = 6;
}
impl Kind for crate::scheduler::Quantum {
    const K: u64 = 6;
}
impl Kind for crate::scheduler::State {
    const K: u64 = 6;
}
impl Kind for crate::scheduler::Grant {
    const K: u64 = 6;
}
impl Kind for crate::scheduler::Shrink {
    const K: u64 = 6;
}
impl Kind for crate::scheduler::Commit {
    const K: u64 = 6;
}
impl Kind for crate::scheduler::Completion {
    const K: u64 = 6;
}
impl Kind for crate::planner::Row {
    const K: u64 = 6;
}
impl Kind for crate::planner_ffi::CStep {
    const K: u64 = 6;
}
impl Kind for crate::planner::DflReq {
    const K: u64 = 6;
}
impl Kind for crate::planner::DflDrafts {
    const K: u64 = 6;
}
impl Kind for crate::planner_ffi::CSpecLen {
    const K: u64 = 6;
}
impl Kind for crate::planner_ffi::CDSelTable {
    const K: u64 = 6;
}
impl Kind for crate::planner_ffi::CDSel {
    const K: u64 = 6;
}
impl Kind for crate::planner_ffi::CKStop {
    const K: u64 = 6;
}
impl Kind for crate::planner_ffi::CNgram {
    const K: u64 = 6;
}
impl Kind for crate::planner_ffi::CSession {
    const K: u64 = 6;
}
impl Kind for crate::planner_ffi::CCost {
    const K: u64 = 6;
}
impl Kind for crate::planner_ffi::CStats {
    const K: u64 = 6;
}
impl Kind for crate::planner_ffi::CSlot {
    const K: u64 = 6;
}
impl Kind for crate::planner_ffi::CLearn {
    const K: u64 = 6;
}
impl Kind for crate::planner_ffi::CHooks {
    const K: u64 = 6;
}
// g6sched.h counts void as 1 byte, as Rust's c_void
impl Kind for std::ffi::c_void {
    const K: u64 = 6;
}
// a function pointer: a pointer to a struct-like kind with no size
macro_rules! fn_kinds {
    ($($t:ty),*) => { $(impl Kind for Option<$t> { const K: u64 = 0x76; })* };
}
fn_kinds!(
    crate::planner_ffi::RunFn,
    crate::planner_ffi::RoundFn,
    crate::planner_ffi::PromptFn,
    crate::planner_ffi::EmitFn,
    crate::planner_ffi::DflFn,
    crate::planner_ffi::DflStartFn,
    crate::planner_ffi::LpFn
);

/// The digest of every shared type's layout.
pub fn digest() -> u64 {
    let mut h: u64 = 14695981039346656037;
    layout!(h, crate::admission::Status { why, requested, room });
    layout!(h, crate::cp_rows::Cand { id, past });
    layout!(h, crate::ffi::Batch { kind, items, items_cap, n_items, from, to, slots, to_slots, slots_cap, n_slots });
    layout!(h, crate::ffi::Ints { p, n });
    layout!(
        h,
        crate::ffi::ParsedReq {
            prompt,
            stop,
            n_gen,
            min_tokens,
            temp,
            top_p,
            seed,
            lp,
            plp,
            sys,
            steering,
            ins_after,
            ins_unless,
            ins_tokens,
            gram_kind,
            gram_strict,
            gram_arm,
            schema,
            schema_n
        }
    );
    layout!(h, crate::flash_lend::Lend { idle_ms, keep, chunk_rows, chunks, idle, lent, since_ns });
    layout!(h, crate::flash_ring::Ring { first, rows, ring });
    layout!(h, crate::kv_plan::Io { kind, page, tier, to, slot, to_slot, part, sub });
    layout!(h, crate::kv_plan::Item { page, part, sub, hslot });
    layout!(
        h,
        crate::kv_share::Totals {
            slots,
            free_slots,
            entries,
            shared,
            spilled_pages,
            moved_in_pages,
            restored_pages,
            hits,
            dropped,
            dropped_pages,
            moved_out,
            moved_out_pages,
            refused,
            refused_pages,
            full,
            block_bytes,
            credit,
            oversize,
            oversize_last
        }
    );
    layout!(h, crate::kv_share::Resize { tier, segs, slots, txn });
    layout!(h, crate::scheduler::Key { epoch, slot, generation });
    layout!(h, crate::scheduler::Identity { enabled, profile_digest, scope, schedule_digest, scale_bits, slot });
    layout!(h, crate::scheduler::Request { id, prompt_tokens, max_tokens, steering, r_rows });
    layout!(h, crate::scheduler::Run { first_page, pages, table_version, list_hash });
    layout!(
        h,
        crate::scheduler::Slot {
            key,
            req,
            run,
            phase,
            prefilled,
            emitted,
            inserted,
            deferred,
            packable,
            pack_tail,
            parked
        }
    );
    layout!(
        h,
        crate::scheduler::Config {
            epoch,
            pool_rows,
            max_rows,
            max_sessions,
            mtp,
            guard_tokens,
            prefill_rows,
            two_lanes,
            decode_prefill_rows,
            decode_rounds,
            window_pages,
            cp_step_rows,
            cp_after,
            cp_seg,
            r2_rows,
            tail_first,
            notice_max,
            c8_pack,
            mtp_sessions,
            sess_drafts,
            park
        }
    );
    layout!(h, crate::scheduler::Quantum { kind, boundary, keys, count, first, rows, firsts, row_counts });
    layout!(
        h,
        crate::scheduler::State {
            boundary,
            event,
            revision,
            last_id,
            pool,
            have_id,
            want_decode,
            busy,
            failed,
            park_seq,
            decode_run,
            work
        }
    );
    layout!(h, crate::scheduler::Grant { key, req, run, reuse, restored, counted });
    layout!(h, crate::scheduler::Shrink { key, r_rows, run });
    layout!(
        h,
        crate::scheduler::Commit {
            epoch,
            event,
            boundary,
            before_hash,
            has_cancel,
            cancel,
            has_admit,
            admit,
            has_shrink,
            shrink
        }
    );
    layout!(h, crate::scheduler::Completion { key, emitted, finished, inserted });
    layout!(h, crate::planner::Row { sess, tok, pos, spos, tgt });
    layout!(
        h,
        crate::planner_ffi::CStep {
            kind,
            event,
            cls,
            round_active,
            round,
            n_sess,
            n_rows,
            sess,
            row0,
            rows,
            ctx,
            ev,
            r
        }
    );
    layout!(h, crate::planner::DflReq { sess, rows, pos0, tok0, lo, query, phase });
    layout!(h, crate::planner::DflDrafts { tok, lp });
    layout!(
        h,
        crate::planner_ffi::CSpecLen {
            on,
            n_levels,
            levels,
            draft_round,
            draft_each,
            table,
            seed,
            pn,
            pk,
            bn,
            bk,
            pos_decay,
            bucket_decay,
            prior,
            rounds,
            chosen,
            pred_sum,
            got_sum
        }
    );
    layout!(h, crate::planner_ffi::CDSelTable { v, d, m, e, p, g, k, n, w, r });
    layout!(
        h,
        crate::planner_ffi::CDSel { on, dwell, margin, probe, model, d, m, tab, rn, rc, an, ag, prior, dn, dt, dc }
    );
    layout!(
        h,
        crate::planner_ffi::CKStop {
            mode,
            kmax,
            tau,
            seed3,
            log,
            solo,
            b_mode,
            b_kmin,
            b_explore,
            b_tau_rows,
            b_tau,
            b_widths,
            b_n_widths
        }
    );
    layout!(
        h,
        crate::planner_ffi::CNgram { table, n_ns, ns, has_cal, cal, n, keys, toks, ps, kmax, kmin, margin, log }
    );
    layout!(
        h,
        crate::planner_ffi::CSession {
            prompt,
            n_prompt,
            first,
            chunk,
            n_gen,
            dflash,
            ins_after,
            ins_unless,
            ins_tokens,
            n_ins
        }
    );
    layout!(
        h,
        crate::planner_ffi::CCost {
            main0,
            main_row,
            draft0,
            draft_sess,
            chain0,
            chain_sess,
            flash,
            flash_table,
            flash_draft_round,
            flash_draft_each
        }
    );
    layout!(
        h,
        crate::planner_ffi::CStats {
            rounds,
            tokens,
            drafted,
            mtp_rounds,
            switches,
            inserted,
            ngram_rounds,
            ngram_drafted,
            ngram_acc,
            dfl_rounds,
            dfl_from,
            dfl_lo,
            dfl_warm,
            promoted,
            acc,
            n_acc
        }
    );
    layout!(
        h,
        crate::planner_ffi::CSlot {
            first,
            chunk,
            n_gen,
            dflash,
            ins_after,
            ins_unless,
            n_ins,
            ins_tokens,
            n_tokens,
            tokens,
            n_d,
            d,
            n_dlm,
            dlm,
            n,
            on,
            done,
            rounds,
            k,
            dfl,
            mtp,
            pb,
            pd,
            pl,
            since,
            mn,
            lo,
            warm,
            armed,
            force,
            dd,
            dl
        }
    );
    layout!(
        h,
        crate::planner_ffi::CLearn {
            round_no,
            cur_round,
            cur_active,
            ring_q,
            ring_lo,
            ring_end,
            ring_top,
            n_ring_tok,
            ring_tok
        }
    );
    layout!(
        h,
        crate::planner_ffi::CHooks {
            ctx,
            run,
            round,
            prompt,
            emit,
            dfl,
            dfl_start,
            lp,
            has_round,
            has_prompt,
            has_emit,
            has_dfl,
            has_dfl_start,
            has_lp
        }
    );
    h
}

#[no_mangle]
pub extern "C" fn g6sch_layout() -> u64 {
    digest()
}
