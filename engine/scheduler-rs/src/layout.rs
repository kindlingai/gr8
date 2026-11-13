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
impl Kind for crate::options::Strs {
    const K: u64 = 6;
}
impl Kind for crate::options::CPlan {
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
impl Kind for crate::driver::CView {
    const K: u64 = 6;
}
impl Kind for crate::driver::CPayload {
    const K: u64 = 6;
}
impl Kind for crate::driver::CHooks {
    const K: u64 = 6;
}
impl Kind for crate::driver::CPlanner {
    const K: u64 = 6;
}
impl Kind for crate::driver::CPool {
    const K: u64 = 6;
}
impl Kind for crate::module::LpRow {
    const K: u64 = 6;
}
impl Kind for crate::module::ModuleOps {
    const K: u64 = 6;
}
impl Kind for crate::pool_ops::PoolHit {
    const K: u64 = 6;
}
impl Kind for crate::pool_ops::PoolOps {
    const K: u64 = 6;
}
impl Kind for crate::control::wire::Notice {
    const K: u64 = 6;
}
impl Kind for crate::control::wire::Ack {
    const K: u64 = 6;
}
impl Kind for crate::control::wire::Commit {
    const K: u64 = 6;
}
impl Kind for crate::control::ffi::Frame {
    const K: u64 = 6;
}
impl Kind for crate::control::ffi::CBoot {
    const K: u64 = 6;
}
impl Kind for crate::service::files::Command {
    const K: u64 = 6;
}
impl Kind for crate::service::status::RunUse {
    const K: u64 = 6;
}
impl Kind for crate::service::ffi::CProfile {
    const K: u64 = 6;
}
impl Kind for crate::service::ffi::CSteering {
    const K: u64 = 6;
}
impl Kind for crate::service::ffi::CBoot {
    const K: u64 = 6;
}
impl Kind for crate::service::ffi::CStatus {
    const K: u64 = 6;
}
impl Kind for crate::service::control::Checked {
    const K: u64 = 6;
}
impl Kind for crate::service::control::PoolCall {
    const K: u64 = 6;
}
impl Kind for crate::service::control::Host {
    const K: u64 = 6;
}
impl Kind for crate::service::control::CControl {
    const K: u64 = 6;
}
impl Kind for crate::service::control::Staged {
    const K: u64 = 6;
}
impl Kind for crate::service::compute::CCompute {
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
    crate::options::FlagFn,
    crate::options::IntFn,
    crate::options::FloatFn,
    crate::options::TextFn,
    crate::options::IntsFn,
    crate::options::FloatsFn,
    crate::options::StrFn,
    crate::options::EngramFn,
    crate::planner_ffi::RunFn,
    crate::planner_ffi::RoundFn,
    crate::planner_ffi::PromptFn,
    crate::planner_ffi::EmitFn,
    crate::planner_ffi::DflFn,
    crate::planner_ffi::DflStartFn, // also driver::QIntFn, the same type
    crate::planner_ffi::LpFn,
    crate::driver::BindFn,
    crate::driver::SlotFn,
    crate::driver::PrefillFn,
    crate::driver::PrefillPackFn,
    crate::driver::KeyBoolFn,
    crate::driver::KeyFn,
    crate::driver::GrowFn,
    crate::driver::VoidFn,
    crate::driver::ShrinkFn,
    crate::driver::PayloadFn,
    crate::driver::GrownFn,
    crate::driver::AdmitFn,
    crate::driver::AdmitPackFn,
    crate::driver::RoundFn,
    crate::driver::QFn,
    crate::driver::QBoolFn,
    crate::module::BindRequestFn,
    crate::module::SlotRunFn,
    crate::module::RetainFn,
    crate::module::ParkFn,
    crate::module::BoolOutFn,
    crate::module::ResizeFn,
    crate::module::U32OutFn,
    crate::module::BucketFn,
    crate::module::ModeFn,
    crate::module::LaunchFn,
    crate::module::ModPrefillFn,
    crate::module::PrefillSeqFn,
    crate::module::PackSeqFn,
    crate::module::LogprobFn,
    crate::module::ModDflFn,
    crate::pool_ops::MatchFn,
    crate::pool_ops::TakeFn,
    crate::pool_ops::U64OutFn,
    crate::pool_ops::TiersFn,
    crate::pool_ops::RefusalFn,
    crate::pool_ops::RRowsFn,
    crate::pool_ops::MakeRoomFn,
    crate::service::control::CheckFn,
    crate::service::control::ShareFn,
    crate::service::control::ReasonFn,
    crate::service::control::TotalsFn,
    crate::service::control::BoundaryFn,
    crate::service::control::StageFn,
    crate::service::control::DigestFn,
    crate::service::control::ApplyFn,
    crate::service::control::StepFn,
    crate::service::control::CountersFn
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
            presence,
            frequency,
            tpresence,
            tfrequency,
            think_end,
            boost_start,
            boost_per1k,
            boost_max,
            seq_mult,
            seq_base,
            seq_min,
            seq_cap,
            avoid,
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
    layout!(
        h,
        crate::scheduler::Identity {
            enabled,
            profile_digest,
            scope,
            schedule_digest,
            scale_bits,
            slot,
            images,
            image_digest
        }
    );
    layout!(h, crate::scheduler::Request { id, prompt_tokens, max_tokens, steering, r_rows, ckpt_at });
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
    layout!(h, crate::driver::CView { s, n, st });
    layout!(h, crate::driver::CPayload { prompt_len, chunk, decode_prompt, steering_ok });
    layout!(h, crate::driver::CPool { grow, shrink, release, head });
    layout!(
        h,
        crate::driver::CHooks {
            ctx,
            bind,
            retire,
            prefill,
            prefill_pack,
            dfl_ok,
            terminal,
            grow,
            finished,
            resize,
            shrink,
            settle,
            payload,
            grown,
            has_prefill_pack,
            has_dfl_ok,
            has_terminal,
            has_grow,
            has_finished,
            has_resize,
            has_shrink,
            has_settle,
            has_grown
        }
    );
    layout!(
        h,
        crate::driver::CPlanner {
            ctx,
            admit,
            admit_pack,
            round,
            demote,
            promote,
            release,
            length,
            finished,
            active,
            inserted
        }
    );
    layout!(h, crate::module::LpRow { lp, rank, n, id, tlp });
    layout!(
        h,
        crate::module::ModuleOps {
            ctx,
            bind,
            bind_request,
            grown,
            shrink,
            retire,
            retain,
            park,
            fence,
            before_work,
            drained,
            resize,
            flash_room,
            bucket,
            mode,
            launch,
            prefill,
            prefill_seq,
            prefill_pack,
            logprob,
            draft_lp,
            dfl_ok,
            dfl_start,
            dfl,
            gm_on,
            gm_bits,
            gm_words,
            pen_rows,
            pen_sess
        }
    );
    layout!(h, crate::pool_ops::PoolHit { skip, lcp, why, age, stored, tier, slot });
    layout!(
        h,
        crate::pool_ops::PoolOps {
            ctx,
            r#match,
            take,
            digest,
            tiers,
            refusal,
            r_rows,
            make_room,
            pool,
            max_run_pages,
            grow,
            count,
            status
        }
    );
    layout!(h, crate::control::wire::Notice { epoch, event, boundary, request, payload_hash, kind });
    layout!(h, crate::control::wire::Ack { notice, rank, state_hash, room });
    layout!(h, crate::control::wire::Commit { c, hold, has_resize, resize });
    layout!(h, crate::control::ffi::Frame { tag, notice, ack, commit, epoch, request, seq, rank, bytes, n });
    layout!(
        h,
        crate::control::ffi::CBoot {
            rank,
            ranks,
            timeout,
            port,
            head_ip,
            head_ip_n,
            directory,
            directory_n,
            agreement,
            agreement_n
        }
    );
    layout!(h, crate::service::files::Command { epoch, request, hash, admit });
    layout!(h, crate::service::status::RunUse { table_version, held, worst });
    layout!(h, crate::service::ffi::CProfile { name, name_n, digest, active_layers });
    layout!(h, crate::service::ffi::CSteering { profiles, n, scope, forced_scale_bits, loaded, forced });
    layout!(
        h,
        crate::service::ffi::CBoot {
            scheduler,
            steering,
            max_sessions,
            rank,
            ranks,
            prefill_min,
            control_version,
            lp_max,
            directory,
            directory_n,
            signature,
            signature_n,
            mtp0,
            pack,
            park
        }
    );
    layout!(
        h,
        crate::service::ffi::CStatus {
            epoch,
            queued,
            stalls,
            grows,
            stalled,
            tiers,
            n_tiers,
            runs,
            n_runs,
            pool_pages,
            resident,
            retiring,
            free,
            room,
            held_pages,
            worst_pages,
            has_growth,
            has_tiers,
            has_runs
        }
    );
    layout!(h, crate::service::control::Checked { prompt, n, sys, plp, n_gen, steering, ok, ckpt_at });
    layout!(h, crate::service::control::PoolCall { kind, rows, worst, r, out, run, prompt, n, span, ns });
    layout!(h, crate::service::control::Staged { req, prompt, n, sys, plp });
    layout!(
        h,
        crate::service::control::Host {
            ctx,
            check,
            retire_run,
            share,
            refusal,
            tiers,
            idle_boundary,
            stage,
            unstage,
            digest,
            abort,
            apply,
            mode,
            step,
            counters,
            begin,
            bound,
            release
        }
    );
    layout!(
        h,
        crate::service::control::CControl {
            scheduler,
            directory,
            directory_n,
            endpoint,
            ingress,
            queue,
            trace,
            slots,
            n_slots,
            state,
            pool,
            host,
            lend,
            share_ms,
            rank,
            head,
            admit_hold,
            elog
        }
    );
    layout!(
        h,
        crate::service::compute::CCompute {
            control,
            endpoint,
            ingress,
            trace,
            slots,
            n_slots,
            state,
            module,
            host,
            rank,
            elog,
            dfl_reuse,
            ngram
        }
    );
    layout!(
        h,
        crate::signature::SigInputs {
            model,
            cap,
            layers_s,
            chunk,
            nc,
            set_id,
            dfl,
            dfl_k,
            s2sess,
            knobs,
            mp,
            mtp,
            dfl_reuse,
            dfl_reuse_min,
            lanes,
            gsplit,
            ckpt,
            ckpt_blocks,
            ckpt_persist,
            s2_below,
            two,
            pf_own,
            pf_qp,
            qp_ctx,
            qp_min,
            grammar,
            ngram,
            kstop_mode,
            kstop_kmax,
            kstop_tau,
            kstop_seed3,
            draft_par,
            qs_mode,
            iq_split,
            mtp_split,
            dhead_on,
            dhead_tpc,
            dhead_ks,
            mtp_last,
            dhead2,
            dfl_i4,
            dfl_body_i4,
            dfl_alpha,
            dsel_model,
            dsel_tab,
            kstopb,
            cp_step,
            cp_seg,
            ctx_sp,
            posidx,
            steer_shared,
            ds_mtp,
            sh_tc,
            l2hint,
            fc2stg,
            fc1bstg,
            c8_pack,
            sess_drafts,
            pf_lanes,
            tk_main,
            tk_decode,
            idx_pipe_b,
            idx_q_trig,
            idx_pipe_qr,
            reuse,
            vis_on,
            vis_id
        }
    );
    layout!(
        h,
        crate::module_info::ModuleInfo {
            abi,
            model,
            name,
            nr,
            mx,
            s2n,
            max_sessions,
            hidden,
            layers,
            mtp_layers,
            vocab,
            experts,
            topk,
            kv_row_bytes,
            kv_page,
            kv_format,
            caps,
            dfl_block,
            pack_members,
            dfl_drafts,
            dfl_rep,
            tp_sort,
            kv4,
            xgrammar,
            ds_markov_maxs,
            ds_block
        }
    );
    layout!(
        h,
        crate::options::KnobApi {
            table_hash,
            flag,
            int_,
            float_,
            text,
            ints,
            floats,
            tests_set
        }
    );
    layout!(
        h,
        crate::options::Host {
            ctx,
            engram_colls,
            dsk_mtp,
            dsk_relaxed,
            dsk_relax_topp,
            dsk_err
        }
    );
    layout!(
        h,
        crate::options::Strs {
            p,
            n
        }
    );
    layout!(
        h,
        crate::options::CPlan {
            prompt,
            first,
            chunk,
            n_gen,
            inv_temp,
            seed,
            top_p,
            lp,
            plp,
            rs
        }
    );
    layout!(
        h,
        crate::options::Options {
            rank,
            ips,
            ips2,
            ring,
            ring_devs,
            proxy_cpu,
            pf_ips2,
            manifest,
            packfile,
            v2_dir,
            v2_slices,
            v2_widths,
            v2_dense,
            v2_layout,
            v2_trellis,
            qo_man,
            qo_file,
            layers_s,
            levels_s,
            prompts,
            co_cap,
            co_mp,
            co_top,
            co_pfrows,
            co_mtp,
            co_lanes,
            co_nc,
            co_temp,
            co_graph,
            co_wide,
            co_pfv2,
            co_adapt,
            has_n_gen,
            n_gen,
            has_chunk,
            chunk,
            has_seed,
            seed,
            has_top_p,
            top_p,
            has_rs,
            rs,
            qp_man,
            qp_file,
            dfl_man,
            dfl_file,
            dfl_sel,
            dfl_k,
            spec_reset_per_request,
            spec_fixed,
            plan,
            image,
            hash,
            reuse_prefix,
            check_args,
            lanes_explicit,
            s2dir,
            grammar_dir,
            kv_tiers,
            kv_keep,
            kv_system_after,
            kv_write_limit,
            kv_format,
            kv_old,
            kv_old_all,
            set_id,
            steering_dir,
            steering_force_s,
            steering_model_revision,
            steering_tokenizer_revision,
            steering_template_digest,
            steering_weight_set,
            steering_template_file,
            s2port,
            s2sess,
            s2epoch,
            cap,
            top,
            pfrows,
            nc,
            mtp,
            mp,
            lanes,
            temp,
            graph,
            wide,
            pfv2,
            adapt,
            release,
            s2mode,
            dfl,
            k,
            layers,
            progress_isolate,
            route_split_mode,
            c8_pack_env,
            spec_on,
            spec_levels,
            spec_draft_round,
            spec_draft_each,
            spec_table,
            rt_win,
            kstop_mode,
            kstop_kmax,
            kstop_tau,
            kstop_seed3,
            kstop_log,
            kstop_solo,
            kstopb_mode,
            kstopb_kmin,
            kstopb_tau,
            kstopb_tau_n,
            kstopb_widths,
            kstopb_explore,
            flash_cache,
            flash_budget_mib,
            ds_moe_main_topk,
            ds_moe_decode_topk,
            ds_idx_pipe_b,
            ds_idx_q_trig,
            ds_idx_pipe_qr,
            ds_admit_keep_scratch,
            ds_sess_drafts,
            ds_draft_cap,
            plans,
            n_plans,
            row_counts,
            draft_rows,
            nl,
            nx,
            ncoll,
            r,
            own4,
            qp_roll_s,
            qp_roll_ser,
            qp_wire_e
        }
    );
    h
}

#[no_mangle]
pub extern "C" fn g6sch_layout() -> u64 {
    digest()
}
