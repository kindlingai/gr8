//! The S2 scheduler (was the Scheduler class in engine/runtime/s2/scheduler.h): every rank's copy of which session
//! holds which slot and pages, what the next quantum of work is, and how completions advance the sessions. Every rank
//! runs the same calls on the same replicated state; the event digest confirms they agree.
//!
//! The state stays in C++ (`Slot` array, `State`): these functions act on it through `#[repr(C)]` mirrors. Integer
//! arithmetic wraps where the C++ unsigned arithmetic did, so every result matches it bit for bit. A refusal returns the
//! C++ message; the functions that mutated state before refusing (complete_decode) leave the same partial state the C++
//! did.

pub const PAGE: u32 = 64;
pub const MAX_DECODE_ROUNDS: u32 = 64;
pub const PREFILL_DEFER_MAX: u32 = 3;
pub const NOTICE_MIN: u64 = 2;
pub const NOTICE_MAX: u64 = 8;
pub const NOTICE_ROWS: u32 = 2048;
pub const PACK_MAX_MEMBERS: usize = 8;
/// The most session slots any build has (g6s2::MAX_SESSIONS is 8 or 32): the size of Quantum's arrays.
pub const QMAX: usize = 32;

pub type Err = &'static str;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Key {
    pub epoch: u64,
    pub slot: u32,
    pub generation: u32,
}

/// steering.h Identity
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Identity {
    pub enabled: bool,
    pub profile_digest: [u8; 32],
    pub scope: u8,
    pub schedule_digest: [u8; 32],
    pub scale_bits: u32,
    pub slot: u32,
    /// DS41 image prompts: the prefix-cache identity of the prompt's images (steering.h)
    pub images: bool,
    pub image_digest: [u8; 32],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Request {
    pub id: u64,
    pub prompt_tokens: u32,
    pub max_tokens: u32,
    pub steering: Identity,
    pub r_rows: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Run {
    pub first_page: u32,
    pub pages: u32,
    pub table_version: u32,
    pub list_hash: u64,
}

// Phase
pub const FREE: u32 = 0;
pub const PREFILL: u32 = 1;
pub const DECODE: u32 = 2;
pub const RETIRING: u32 = 3;
pub const PARKED: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot {
    pub key: Key,
    pub req: Request,
    pub run: Run,
    pub phase: u32,
    pub prefilled: u32,
    pub emitted: u32,
    pub inserted: u32,
    pub deferred: u32,
    pub packable: bool,
    pub pack_tail: bool,
    pub parked: u64,
}

/// Config's scalar settings (the steering receipt stays in C++).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    pub epoch: u64,
    pub pool_rows: u32,
    pub max_rows: u32,
    pub max_sessions: u32,
    pub mtp: u32,
    pub guard_tokens: u32,
    pub prefill_rows: u32,
    pub two_lanes: bool,
    pub decode_prefill_rows: u32,
    pub decode_rounds: u32,
    pub window_pages: u32,
    pub cp_step_rows: u32,
    pub cp_after: u64,
    pub cp_seg: bool,
    pub r2_rows: u32,
    pub tail_first: bool,
    pub notice_max: u32,
    pub c8_pack: bool,
    pub mtp_sessions: u32,
    pub sess_drafts: bool,
    pub park: bool,
}

impl Config {
    pub fn guard(&self) -> u32 {
        if self.guard_tokens == u32::MAX {
            self.mtp.wrapping_mul(2)
        } else {
            self.guard_tokens
        }
    }
    pub fn limit(&self) -> u32 {
        let step = if self.mtp_sessions != 0 { 1 } else { self.mtp.wrapping_add(1) };
        self.max_sessions.min(self.max_rows / step)
    }
}

// Kind
pub const IDLE: u32 = 0;
pub const K_PREFILL: u32 = 1;
pub const PROMPT_TAIL: u32 = 2;
pub const K_DECODE: u32 = 3;
pub const PREFILL_PACK: u32 = 4;
pub const PROMPT_TAIL_PACK: u32 = 5;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Quantum {
    pub kind: u32,
    pub boundary: u64,
    pub keys: [Key; QMAX],
    pub count: u32,
    pub first: u32,
    pub rows: u32,
    pub firsts: [u32; QMAX],
    pub row_counts: [u32; QMAX],
}

impl Default for Quantum {
    fn default() -> Self {
        Quantum {
            kind: IDLE,
            boundary: 0,
            keys: [Key::default(); QMAX],
            count: 0,
            first: 0,
            rows: 0,
            firsts: [0; QMAX],
            row_counts: [0; QMAX],
        }
    }
}

/// The scheduler's counters and the quantum in flight.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct State {
    pub boundary: u64,
    pub event: u64,
    pub revision: u64,
    pub last_id: u64,
    pub pool: u64,
    pub have_id: bool,
    pub want_decode: bool,
    pub busy: bool,
    pub failed: bool,
    pub park_seq: u64,
    pub decode_run: u32,
    pub work: Quantum,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grant {
    pub key: Key,
    pub req: Request,
    pub run: Run,
    pub reuse: u32,
    pub restored: u32,
    pub counted: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shrink {
    pub key: Key,
    pub r_rows: u32,
    pub run: Run,
}

/// The parts of a Commit apply reads (hold and resize it never reads).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Commit {
    pub epoch: u64,
    pub event: u64,
    pub boundary: u64,
    pub before_hash: u64,
    pub has_cancel: bool,
    pub cancel: Key,
    pub has_admit: bool,
    pub admit: Grant,
    pub has_shrink: bool,
    pub shrink: Shrink,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Completion {
    pub key: Key,
    pub emitted: u32,
    pub finished: bool,
    pub inserted: u32,
}

fn require(ok: bool, what: Err) -> Result<(), Err> {
    if ok {
        Ok(())
    } else {
        Err(what)
    }
}

// ---- the free functions ------------------------------------------------------------------------------------------

pub fn reserve_rows(c: &Config, r: &Request) -> Result<u64, Err> {
    require(r.prompt_tokens != 0 && r.max_tokens != 0, "empty request")?;
    Ok(u64::from(r.prompt_tokens) + u64::from(r.max_tokens) + u64::from(c.guard()))
}
pub fn reserve_pages(c: &Config, r: &Request) -> Result<u32, Err> {
    let rows = reserve_rows(c, r)?;
    require(rows <= u64::from(c.pool_rows), "request exceeds pool including guard")?;
    Ok(rows.wrapping_add(u64::from(PAGE) - 1).wrapping_div(u64::from(PAGE)) as u32)
}
pub fn round_rows(c: &Config, length: u64) -> u64 {
    length.wrapping_add(u64::from(c.guard()))
}
pub fn grant_rows(c: &Config, r: &Request) -> Result<u64, Err> {
    Ok(reserve_rows(c, r)?.min(round_rows(c, u64::from(r.prompt_tokens) + 1)))
}
pub fn grant_pages(c: &Config, r: &Request) -> Result<u32, Err> {
    Ok((grant_rows(c, r)? + u64::from(PAGE) - 1).wrapping_div(u64::from(PAGE)) as u32)
}
pub fn grant_pages_ok(c: &Config, r: &Request, pages: u32) -> Result<bool, Err> {
    if pages == grant_pages(c, r)? {
        return Ok(true);
    }
    if pages == reserve_pages(c, r)? {
        return Ok(true);
    }
    Ok(c.window_pages != 0 && pages == reserve_pages(c, r)?.wrapping_add(c.window_pages))
}
pub fn prefill_left(s: &Slot) -> u32 {
    s.req.prompt_tokens.wrapping_sub(1).wrapping_sub(s.prefilled)
}
pub fn prefill_first(a: &Slot, b: &Slot) -> bool {
    let (owed_a, owed_b) = (a.deferred >= PREFILL_DEFER_MAX, b.deferred >= PREFILL_DEFER_MAX);
    if owed_a != owed_b {
        return owed_a;
    }
    if owed_a && a.deferred != b.deferred {
        return a.deferred > b.deferred;
    }
    let (la, lb) = (prefill_left(a), prefill_left(b));
    if la != lb {
        return la < lb;
    }
    a.req.id < b.req.id
}

/// Config::validate's scalar rules, in order (the steering receipt's rules follow in C++). mtp0, pack, park: the
/// model's capabilities (module_info.h).
pub fn validate(c: &Config, max_sessions: u32, mtp0: bool, pack: bool, park: bool) -> Result<(), Err> {
    require(c.epoch != 0 && c.pool_rows >= PAGE && c.pool_rows.is_multiple_of(PAGE), "bad epoch/pool")?;
    require(
        (c.mtp > 0 || mtp0) && c.mtp <= 7 && c.max_sessions != 0 && c.max_sessions <= max_sessions && c.limit() != 0,
        "bad session/row limit",
    )?;
    require(!c.c8_pack || pack, "c8 prefill pack is DS41 only")?;
    require(!c.park || park, "parked slots are DS41 only")?;
    require(c.guard() >= c.mtp.wrapping_mul(2) && c.prefill_rows != 0, "insufficient guard/prefill rows")?;
    require(
        c.decode_prefill_rows == 0
            || (c.decode_prefill_rows.is_multiple_of(PAGE) && c.decode_prefill_rows <= c.prefill_rows),
        "decode_prefill_rows must be 0 or a multiple of 64 no larger than prefill_rows",
    )?;
    require(c.decode_rounds <= MAX_DECODE_ROUNDS, "decode_rounds must be at most 64")?;
    require(u64::from(c.notice_max) >= NOTICE_MIN && u64::from(c.notice_max) <= NOTICE_MAX, "notice_max must be 2..8")?;
    require(
        c.cp_step_rows == 0 || c.cp_step_rows >= c.mtp.wrapping_add(1),
        "cp_step_rows must be 0 or at least one session's verify (mtp + 1)",
    )?;
    require(!c.cp_seg || c.cp_step_rows != 0, "cp_seg needs cp_step_rows")?;
    require(
        c.r2_rows.is_multiple_of(PAGE) && (c.r2_rows == 0 || c.window_pages != 0),
        "r2_rows must be 0, or a multiple of 64 with window_pages",
    )?;
    Ok(())
}

// ---- the scheduler -----------------------------------------------------------------------------------------------

pub fn init(c: &Config, s: &mut [Slot]) {
    for (i, x) in s.iter_mut().enumerate() {
        x.key = Key { epoch: c.epoch, slot: i as u32, generation: 0 };
    }
}

fn slot_index(s: &[Slot], k: &Key) -> Result<usize, Err> {
    let i = k.slot as usize;
    require(i < s.len() && s[i].key == *k && s[i].phase != FREE, "stale session key")?;
    Ok(i)
}

fn finish_slot(s: &mut Slot, r: &Completion, revision: &mut u64) -> Result<(), Err> {
    require(s.key == r.key && r.emitted <= s.req.max_tokens.wrapping_sub(s.emitted), "invalid completion")?;
    s.emitted = s.emitted.wrapping_add(r.emitted);
    require(r.inserted >= s.inserted, "insert count went back")?;
    s.inserted = r.inserted;
    if r.finished || s.emitted == s.req.max_tokens {
        s.phase = RETIRING;
        *revision = revision.wrapping_add(1);
    } else {
        s.phase = DECODE;
    }
    Ok(())
}

fn is_prefill_kind(k: u32) -> bool {
    k == K_PREFILL || k == PROMPT_TAIL || k == PREFILL_PACK || k == PROMPT_TAIL_PACK
}

fn completed(c: &Config, s: &mut [Slot], st: &mut State) {
    st.boundary = st.work.boundary;
    st.busy = false;
    alternate(c, st.work.kind, &mut st.want_decode, &mut st.decode_run);
    let w = &st.work;
    if !c.c8_pack && (w.kind == K_PREFILL || w.kind == PROMPT_TAIL) {
        for x in s.iter_mut() {
            if x.phase == PREFILL {
                x.deferred = if x.key == w.keys[0] { 0 } else { x.deferred.wrapping_add(1) };
            }
        }
    } else if is_prefill_kind(w.kind) {
        for x in s.iter_mut() {
            if x.phase == PREFILL {
                let served = w.keys[..w.count as usize].contains(&x.key);
                x.deferred = if served { 0 } else { x.deferred.wrapping_add(1) };
            }
        }
    }
}

pub fn idle_boundary(s: &[Slot], st: &mut State, target: u64) -> Result<(), Err> {
    require(!st.failed && !st.busy && resident(s) == 0 && target >= st.boundary, "cannot advance a live boundary")?;
    st.boundary = target;
    Ok(())
}

pub fn resident(s: &[Slot]) -> u32 {
    s.iter().filter(|x| x.phase != FREE && x.phase != RETIRING && x.phase != PARKED).count() as u32
}
pub fn retiring(s: &[Slot]) -> u32 {
    s.iter().filter(|x| x.phase == RETIRING).count() as u32
}
pub fn parked(s: &[Slot]) -> u32 {
    s.iter().filter(|x| x.phase == PARKED).count() as u32
}
pub fn find(s: &[Slot], id: u64) -> Option<Key> {
    s.iter().find(|x| x.phase != FREE && x.phase != PARKED && x.req.id == id).map(|x| x.key)
}

pub fn next_key(c: &Config, s: &[Slot], cancel: Option<Key>, park: Option<u32>) -> Result<Key, Err> {
    let key = |x: &Slot| -> Result<Key, Err> {
        require(x.key.generation != u32::MAX, "session generation exhausted")?;
        Ok(Key { epoch: c.epoch, slot: x.key.slot, generation: x.key.generation + 1 })
    };
    if let Some(p) = park {
        let p = p as usize;
        require(
            c.park && p < s.len() && (s[p].phase == PARKED || s[p].phase == RETIRING),
            "no parked session on that slot",
        )?;
        return key(&s[p]);
    }
    let is_cancel = |x: &Slot| cancel.is_some_and(|k| x.key == k);
    if !c.park {
        for x in s {
            if x.phase == FREE || x.phase == RETIRING || is_cancel(x) {
                return key(x);
            }
        }
        return Err("no session slot");
    }
    let lim = (c.limit() as usize).min(s.len());
    for x in &s[..lim] {
        if x.phase == FREE || (is_cancel(x) && x.phase != PARKED) {
            return key(x);
        }
    }
    let order = |x: &Slot| if x.phase == PARKED { x.parked } else { u64::MAX };
    let mut old: Option<&Slot> = None;
    for x in &s[..lim] {
        if (x.phase == PARKED || x.phase == RETIRING) && old.is_none_or(|o| order(x) < order(o)) {
            old = Some(x);
        }
    }
    match old {
        Some(o) => key(o),
        None => Err("no session slot"),
    }
}

pub fn reclaimable(c: &Config, s: &[Slot], cancel: Option<Key>) -> Vec<Run> {
    let is_cancel = |x: &Slot| cancel.is_some_and(|k| x.key == k);
    s.iter()
        .filter(|x| {
            if c.park {
                x.phase != RETIRING && x.phase != PARKED && cancel.is_some() && x.phase != FREE && is_cancel(x)
            } else {
                x.phase == RETIRING || (cancel.is_some() && x.phase != FREE && is_cancel(x))
            }
        })
        .map(|x| x.run)
        .collect()
}

fn fnv(h: &mut u64, v: u64) {
    for i in 0..8 {
        *h ^= u64::from((v >> (8 * i)) as u8);
        *h = h.wrapping_mul(1099511628211);
    }
}
fn fnv_byte(h: &mut u64, v: u8) {
    *h = (*h ^ u64::from(v)).wrapping_mul(1099511628211);
}

/// The digest's settings and counters, up to and including the pool's digest; the steering receipt's hash follows
/// (steering.h hash_boot, in C++), then digest_slots.
pub fn digest_head(c: &Config, st: &State) -> u64 {
    let mut h: u64 = 14695981039346656037;
    let add = |h: &mut u64, v: u64| fnv(h, v);
    add(&mut h, c.epoch);
    add(&mut h, u64::from(c.pool_rows));
    add(&mut h, u64::from(c.max_rows));
    add(&mut h, u64::from(c.max_sessions));
    add(&mut h, u64::from(c.mtp));
    add(&mut h, u64::from(c.guard()));
    add(&mut h, u64::from(c.prefill_rows));
    add(&mut h, u64::from(c.two_lanes));
    if c.decode_prefill_rows != 0 {
        add(&mut h, u64::from(c.decode_prefill_rows));
    }
    add(&mut h, u64::from(st.have_id));
    add(&mut h, st.last_id);
    add(&mut h, st.boundary);
    add(&mut h, st.event);
    add(&mut h, u64::from(st.want_decode));
    add(&mut h, u64::from(st.busy));
    add(&mut h, u64::from(st.failed));
    if c.decode_rounds > 1 {
        add(&mut h, u64::from(c.decode_rounds));
        add(&mut h, u64::from(st.decode_run));
    }
    if c.window_pages != 0 {
        add(&mut h, u64::from(c.window_pages));
    }
    if c.tail_first {
        add(&mut h, 0x7A11F1257);
    }
    if c.c8_pack {
        add(&mut h, 0xC8A11F1257);
    }
    if u64::from(c.notice_max) != NOTICE_MAX {
        add(&mut h, u64::from(c.notice_max));
    }
    if c.cp_step_rows != 0 {
        add(&mut h, u64::from(c.cp_step_rows));
        add(&mut h, c.cp_after);
        add(&mut h, u64::from(c.cp_seg));
    }
    if c.r2_rows != 0 {
        add(&mut h, u64::from(c.r2_rows));
    }
    if c.park {
        add(&mut h, 0x9A12CED);
        add(&mut h, st.park_seq);
    }
    add(&mut h, st.pool);
    h
}

fn hash_identity(h: &mut u64, s: &Identity) {
    fnv_byte(h, u8::from(s.enabled));
    if s.images {
        fnv_byte(h, 2);
        for &v in &s.image_digest {
            fnv_byte(h, v);
        }
    }
    if !s.enabled {
        return;
    }
    for &v in &s.profile_digest {
        fnv_byte(h, v);
    }
    fnv_byte(h, s.scope);
    for &v in &s.schedule_digest {
        fnv_byte(h, v);
    }
    for i in 0..4 {
        fnv_byte(h, (s.scale_bits >> (8 * i)) as u8);
    }
}

/// The digest's slots, after the steering receipt's hash.
pub fn digest_slots(c: &Config, s: &[Slot], mut h: u64) -> u64 {
    for x in s {
        fnv(&mut h, u64::from(x.key.slot));
        fnv(&mut h, u64::from(x.key.generation));
        fnv(&mut h, u64::from(x.phase));
        if x.phase == FREE {
            continue;
        }
        fnv(&mut h, x.req.id);
        fnv(&mut h, u64::from(x.req.prompt_tokens));
        fnv(&mut h, u64::from(x.req.max_tokens));
        hash_identity(&mut h, &x.req.steering);
        if x.req.r_rows != 0 {
            fnv(&mut h, u64::from(x.req.r_rows));
        }
        fnv(&mut h, u64::from(x.run.first_page));
        fnv(&mut h, u64::from(x.run.pages));
        fnv(&mut h, u64::from(x.run.table_version));
        fnv(&mut h, x.run.list_hash);
        fnv(&mut h, u64::from(x.prefilled));
        fnv(&mut h, u64::from(x.emitted));
        fnv(&mut h, u64::from(x.inserted));
        fnv(&mut h, u64::from(x.deferred));
        if c.c8_pack {
            fnv(&mut h, u64::from(x.packable));
            fnv(&mut h, u64::from(x.pack_tail));
        }
        if c.park {
            fnv(&mut h, x.parked);
        }
    }
    h
}

/// apply's result: the runs freed (in slot order) and whether the grant was admitted.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Changes {
    pub freed: Vec<Run>,
    pub admitted: bool,
}

/// `digest`: the digest of the state before the commit (Scheduler::digest), which the commit's before_hash must equal.
pub fn apply(c: &Config, s: &mut [Slot], st: &mut State, e: &Commit, digest: u64) -> Result<Changes, Err> {
    require(!st.failed && !st.busy, "commit requires quiescent live engine")?;
    require(
        e.epoch == c.epoch
            && e.event == st.event.wrapping_add(1)
            && e.boundary == st.boundary
            && e.before_hash == digest,
        "event/state mismatch",
    )?;
    let mut after: Vec<Slot> = s.to_vec();
    let mut d = Changes::default();
    let mut park_seq = st.park_seq;
    for x in after.iter_mut() {
        if c.park && x.phase == RETIRING {
            x.phase = PARKED;
            park_seq = park_seq.wrapping_add(1);
            x.parked = park_seq;
            continue;
        }
        if x.phase == RETIRING || (e.has_cancel && x.phase != FREE && x.phase != PARKED && x.key == e.cancel) {
            d.freed.push(x.run);
            x.phase = FREE;
        }
    }
    if e.has_cancel {
        require(e.cancel.epoch == c.epoch, "cancel epoch mismatch")?;
    }
    if e.has_shrink {
        let k = &e.shrink;
        require(k.key.epoch == c.epoch && (k.key.slot as usize) < s.len(), "bad shrink key")?;
        let t = &mut after[k.key.slot as usize];
        require(
            t.phase == DECODE
                && t.key == k.key
                && c.r2_rows != 0
                && k.r_rows == c.r2_rows
                && t.req.r_rows == 0
                && k.run.table_version == t.run.table_version
                && k.run.pages == t.run.pages,
            "bad shrink",
        )?;
        t.req.r_rows = k.r_rows;
        t.run = k.run;
    }
    if e.has_admit {
        let g = &e.admit;
        require(!st.have_id || g.req.id > st.last_id, "native request ordinals must increase")?;
        require((g.key.slot as usize) < s.len() && g.key.epoch == c.epoch, "bad grant key")?;
        let ti = g.key.slot as usize;
        {
            let target = &after[ti];
            require(
                (target.phase == FREE || (c.park && target.phase == PARKED))
                    && target.key.generation != u32::MAX
                    && g.key.generation == target.key.generation.wrapping_add(1),
                "grant slot generation mismatch",
            )?;
        }
        require(
            grant_pages_ok(c, &g.req, g.run.pages)?
                && g.run.first_page < c.pool_rows / PAGE
                && g.run.table_version != 0,
            "bad reservation",
        )?;
        let mut live = 0u32;
        for x in after.iter() {
            if x.phase != FREE && x.phase != PARKED {
                live += 1;
                require(x.req.id != g.req.id, "duplicate native request id")?;
            }
        }
        require(live < c.limit(), "active session/row limit")?;
        require(g.reuse < g.req.prompt_tokens, "reused rows past the prompt")?;
        require(g.req.r_rows == 0 || g.req.r_rows == c.r2_rows, "grant at an R the engine does not have")?;
        after[ti] = Slot {
            key: g.key,
            req: g.req,
            run: g.run,
            phase: PREFILL,
            prefilled: g.reuse,
            emitted: 0,
            inserted: 0,
            deferred: 0,
            packable: c.c8_pack && g.reuse == 0 && g.restored == 0 && !g.req.steering.enabled,
            pack_tail: false,
            parked: 0,
        };
        d.admitted = true;
    }
    if e.has_admit {
        st.last_id = e.admit.req.id;
        st.have_id = true;
    }
    s.copy_from_slice(&after);
    st.event = e.event;
    st.revision = st.revision.wrapping_add(1);
    st.park_seq = park_seq;
    Ok(d)
}

pub fn grow(c: &Config, s: &mut [Slot], st: &State, k: &Key, r: &Run) -> Result<(), Err> {
    let i = slot_index(s, k)?;
    let x = &s[i];
    let ok = !st.busy
        && x.phase == DECODE
        && r.table_version == x.run.table_version
        && r.first_page == x.run.first_page
        && r.pages > x.run.pages
        && r.pages <= reserve_pages(c, &x.req)?;
    require(ok, "bad run growth")?;
    s[i].run = *r;
    Ok(())
}

pub fn decoding(s: &[Slot]) -> Vec<Key> {
    let mut v: Vec<&Slot> = s.iter().filter(|x| x.phase == DECODE).collect();
    v.sort_by_key(|x| x.req.id);
    v.iter().map(|x| x.key).collect()
}

pub fn alternate(c: &Config, k: u32, want: &mut bool, run: &mut u32) {
    if c.tail_first && k == PROMPT_TAIL {
        return;
    }
    if k != K_DECODE {
        *want = true;
        *run = 0;
        return;
    }
    if *run < MAX_DECODE_ROUNDS {
        *run += 1;
    }
    *want = c.decode_rounds > 1 && *run < c.decode_rounds;
}

/// The next quantum's shape from slot state `s` and the alternation flag; `stalled[i]`: decoding session i sits out.
pub fn plan(c: &Config, s: &[Slot], want_decode: bool, stalled: &[bool]) -> Quantum {
    let mut q = Quantum::default();
    let is_stalled = |i: usize| stalled.get(i).copied().unwrap_or(false);
    let mut ordered: Vec<&Slot> = Vec::with_capacity(s.len());
    for (i, x) in s.iter().enumerate() {
        if x.phase == PREFILL || (x.phase == DECODE && !is_stalled(i)) {
            ordered.push(x);
        }
    }
    for i in 1..ordered.len() {
        let x = ordered[i];
        let mut j = i;
        while j > 0 && ordered[j - 1].req.id > x.req.id {
            ordered[j] = ordered[j - 1];
            j -= 1;
        }
        ordered[j] = x;
    }
    let mut pf: Option<&Slot> = None;
    let decoding_any = s.iter().any(|x| x.phase == DECODE);
    for &x in &ordered {
        if x.phase == DECODE {
            q.keys[q.count as usize] = x.key;
            q.count += 1;
        } else if let Some(p) = pf {
            if c.tail_first && (prefill_left(x) == 0) != (prefill_left(p) == 0) {
                if prefill_left(x) == 0 {
                    pf = Some(x);
                }
            } else if prefill_first(x, p) {
                pf = Some(x);
            }
        } else {
            pf = Some(x);
        }
    }
    if c.c8_pack {
        let mut held = 0usize;
        for &x in &ordered {
            if x.phase == PREFILL && x.pack_tail {
                q.keys[held] = x.key;
                q.firsts[held] = x.prefilled;
                q.row_counts[held] = 1;
                held += 1;
            }
        }
        if held > 0 {
            q.count = held as u32;
            q.kind = if held > 1 { PROMPT_TAIL_PACK } else { PROMPT_TAIL };
            q.first = q.firsts[0];
            q.rows = held as u32;
            return q;
        }
    }
    if q.count != 0 && (want_decode || pf.is_none()) {
        q.kind = K_DECODE;
    } else if let Some(p) = pf {
        if c.c8_pack && p.packable && p.prefilled == 0 && prefill_left(p) >= 1 && prefill_left(p) <= 64 {
            let cap = if decoding_any && c.decode_prefill_rows != 0 {
                512u32.min(c.prefill_rows).min(c.decode_prefill_rows)
            } else {
                512u32.min(c.prefill_rows)
            };
            let mut take: Vec<&Slot> = vec![p];
            let mut total = prefill_left(p);
            for &x in &ordered {
                if take.len() >= PACK_MAX_MEMBERS {
                    break;
                }
                if std::ptr::eq(x, p) || x.phase != PREFILL || !x.packable || x.prefilled != 0 {
                    continue;
                }
                let left = prefill_left(x);
                if left == 0 || left > 64 || total.wrapping_add(left) > cap {
                    continue;
                }
                take.push(x);
                total = total.wrapping_add(left);
            }
            if take.len() > 1 && total <= cap {
                for i in 1..take.len() {
                    let x = take[i];
                    let mut j = i;
                    while j > 0 && take[j - 1].req.id > x.req.id {
                        take[j] = take[j - 1];
                        j -= 1;
                    }
                    take[j] = x;
                }
                q.kind = PREFILL_PACK;
                q.count = take.len() as u32;
                q.rows = total;
                for (i, x) in take.iter().enumerate() {
                    q.keys[i] = x.key;
                    q.firsts[i] = x.prefilled;
                    q.row_counts[i] = prefill_left(x);
                }
                return q;
            }
        }
        q.count = 1;
        q.keys[0] = p.key;
        q.first = p.prefilled;
        let left = p.req.prompt_tokens.wrapping_sub(1).wrapping_sub(p.prefilled);
        q.kind = if left != 0 { K_PREFILL } else { PROMPT_TAIL };
        let cap: u64 = if decoding_any && c.decode_prefill_rows != 0 {
            u64::from(c.decode_prefill_rows)
        } else {
            u64::from(c.prefill_rows) * if c.two_lanes && left > c.prefill_rows { 2 } else { 1 }
        };
        q.rows = if left != 0 { u64::from(left).min(cap) as u32 } else { 1 };
    }
    q
}

pub fn next(c: &Config, s: &[Slot], st: &mut State, stalled: &[bool]) -> Result<Quantum, Err> {
    require(!st.failed && !st.busy, "scheduler failed or work in flight")?;
    let mut q = plan(c, s, st.want_decode, stalled);
    q.boundary = st.boundary.wrapping_add(1);
    if q.kind != IDLE {
        st.work = q;
        st.busy = true;
    }
    Ok(q)
}

pub fn advance(sim: &mut [Slot], q: &Quantum) {
    match q.kind {
        K_PREFILL => {
            let t = &mut sim[q.keys[0].slot as usize];
            t.prefilled = t.prefilled.wrapping_add(q.rows);
        }
        PROMPT_TAIL => sim[q.keys[0].slot as usize].phase = DECODE,
        PREFILL_PACK => {
            for i in 0..q.count as usize {
                let t = &mut sim[q.keys[i].slot as usize];
                t.prefilled = t.prefilled.wrapping_add(q.row_counts[i]);
                t.pack_tail = true;
            }
        }
        PROMPT_TAIL_PACK => {
            for i in 0..q.count as usize {
                let t = &mut sim[q.keys[i].slot as usize];
                t.phase = DECODE;
                t.pack_tail = false;
            }
        }
        _ => {}
    }
    if is_prefill_kind(q.kind) {
        for t in sim.iter_mut() {
            if t.phase == PREFILL {
                let served = q.keys[..q.count as usize].contains(&t.key);
                t.deferred = if served { 0 } else { t.deferred.wrapping_add(1) };
            }
        }
    }
}

pub fn horizon(c: &Config, s: &[Slot], st: &State) -> u64 {
    let mut sim: Vec<Slot> = s.to_vec();
    let mut want = st.want_decode;
    let mut run = st.decode_run;
    let inflight = plan(c, &sim, want, &[]);
    if inflight.kind == IDLE {
        return st.boundary.wrapping_add(NOTICE_MIN);
    }
    advance(&mut sim, &inflight);
    alternate(c, inflight.kind, &mut want, &mut run);
    let mut covered: u32 = 0;
    let mut k = NOTICE_MIN;
    while k < u64::from(c.notice_max) {
        let q = plan(c, &sim, want, &[]);
        if q.kind == IDLE {
            return st.boundary.wrapping_add(k);
        }
        advance(&mut sim, &q);
        alternate(c, q.kind, &mut want, &mut run);
        covered = covered.wrapping_add(if is_prefill_kind(q.kind) { q.rows } else { 0 });
        if covered >= NOTICE_ROWS {
            return st.boundary.wrapping_add(k);
        }
        k += 1;
    }
    st.boundary.wrapping_add(u64::from(c.notice_max))
}

pub fn skip_prefill(s: &mut [Slot], st: &State, k: &Key) -> Result<(), Err> {
    let i = slot_index(s, k)?;
    require(!st.busy && s[i].phase == PREFILL && s[i].prefilled == 0, "skip_prefill outside admission")?;
    s[i].prefilled = s[i].req.prompt_tokens.wrapping_sub(1);
    Ok(())
}

pub fn complete_prefill(c: &Config, s: &mut [Slot], st: &mut State) -> Result<(), Err> {
    require(st.busy && st.work.kind == K_PREFILL, "not a prefill quantum")?;
    let i = slot_index(s, &st.work.keys[0])?;
    s[i].prefilled = s[i].prefilled.wrapping_add(st.work.rows);
    completed(c, s, st);
    Ok(())
}

pub fn complete_prefill_pack(c: &Config, s: &mut [Slot], st: &mut State) -> Result<(), Err> {
    require(st.busy && st.work.kind == PREFILL_PACK && st.work.count >= 2, "not a prefill pack quantum")?;
    let w = st.work;
    for i in 0..w.count as usize {
        let x = &s[slot_index(s, &w.keys[i])?];
        require(
            x.phase == PREFILL
                && !x.pack_tail
                && x.prefilled == w.firsts[i]
                && w.row_counts[i] != 0
                && w.row_counts[i] == prefill_left(x),
            "prefill pack membership changed",
        )?;
    }
    for i in 0..w.count as usize {
        let j = slot_index(s, &w.keys[i])?;
        s[j].prefilled = s[j].prefilled.wrapping_add(w.row_counts[i]);
        s[j].pack_tail = true;
    }
    completed(c, s, st);
    Ok(())
}

pub fn complete_tail(c: &Config, s: &mut [Slot], st: &mut State, r: &Completion) -> Result<(), Err> {
    require(st.busy && st.work.kind == PROMPT_TAIL && st.work.keys[0] == r.key, "not this prompt tail")?;
    let i = slot_index(s, &r.key)?;
    let mut copy = s[i];
    finish_slot(&mut copy, r, &mut st.revision)?;
    copy.pack_tail = false;
    let i = slot_index(s, &r.key)?;
    s[i] = copy;
    completed(c, s, st);
    Ok(())
}

pub fn complete_tail_pack(c: &Config, s: &mut [Slot], st: &mut State, r: &[Completion]) -> Result<(), Err> {
    require(
        st.busy && st.work.kind == PROMPT_TAIL_PACK && r.len() == st.work.count as usize,
        "not this prompt tail pack",
    )?;
    let mut copy: Vec<Slot> = s.to_vec();
    for (i, x) in r.iter().enumerate() {
        let Some(t) = copy.get(x.key.slot as usize) else { return Err("tail pack completion invalid") };
        require(
            st.work.keys[i] == x.key
                && t.pack_tail
                && x.emitted <= t.req.max_tokens.wrapping_sub(t.emitted)
                && x.inserted >= t.inserted,
            "tail pack completion invalid",
        )?;
    }
    for x in r {
        // the revision counts as the C++ did: a later refusal keeps the earlier sessions' increments, not their slots
        let t = &mut copy[x.key.slot as usize];
        finish_slot(t, x, &mut st.revision)?;
        t.pack_tail = false;
    }
    s.copy_from_slice(&copy);
    completed(c, s, st);
    Ok(())
}

pub fn complete_decode(c: &Config, s: &mut [Slot], st: &mut State, r: &[Completion]) -> Result<(), Err> {
    require(st.busy && st.work.kind == K_DECODE && r.len() == st.work.count as usize, "not this decode round")?;
    for (i, x) in r.iter().enumerate() {
        require(x.key == st.work.keys[i], "decode completion order differs")?;
        let t = &s[slot_index(s, &x.key)?];
        require(x.emitted <= t.req.max_tokens.wrapping_sub(t.emitted), "too many output tokens")?;
    }
    for x in r {
        let i = slot_index(s, &x.key)?;
        finish_slot(&mut s[i], x, &mut st.revision)?; // a refusal here leaves the earlier sessions finished, as the C++ did
    }
    completed(c, s, st);
    Ok(())
}
