//! The KV node: locked memory cut into leases, served over TCP and, where the connection arrived on an address with a
//! RoCE v2 GID, to one-sided RDMA from the client.
//!
//! A lease is named by its client. Only one live lease holds a name, so two ranks (or a test and a live engine) that
//! name themselves apart can never write into each other's blocks: the TCP path checks every offset against the lease,
//! and for RDMA each lease has its own protection domain holding one memory region per segment of the lease, with its
//! clients' QPs. The NIC refuses a QP of one lease that presents another lease's rkey, or an address outside its
//! regions. A lease's memory is zeroed when it is granted and freed when its last connection closes.
//!
//! Two kinds of lease. OPEN takes one contiguous range of a fixed size (one segment the size of the lease). GOPEN
//! joins a GROUP: the leases of one cluster's ranks (same block size, same rank count). A group lease is a list of
//! fixed-size segments, each holding a whole number of blocks; it starts at the group's floor and grows (GROW) or
//! shrinks (SHRINK) by segments between floor and cap. The group's head REPORTs its demand and how long ago its coldest
//! blocks were used. The node answers with a grant (segments held for every rank of the group, which each rank then
//! GROWs into) or with a request to give segments back. When the node is short, it asks the group whose coldest blocks
//! are the oldest (by more than the margin, weighted) to give: the coldest chat across all the models on the node. The
//! node never takes a segment back on its own: a cluster's ranks hold their blocks in lock step, so only the cluster
//! can drop them, at one event on every rank (stash.rs resize_tier). REMOTE-TIER.md, "Sharing a node".
//!
//! Every grant and every give is a transaction (Txn) of the whole group: an id, one absolute target in segments per
//! rank, and each rank's own part. A rank's GROW and SHRINK name the transaction and its target; one against another
//! transaction is refused (E_STALE), and one a rank already applied is answered as done. A grant is held per rank:
//! once any rank grew into it, the others' parts never expire. Segments a rank gives back stay the group's (retired)
//! until every rank of the group has given, so no other group can take them while the group is between sizes.

use super::proto::{self, Header};
use super::sys::{self, Arena};
use super::verbs::{self, Cq, Device, Mr, Pd, Qp, QpEnd};
use std::collections::{BTreeMap, HashMap};
use std::io::{self, Read, Write};
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock, Weak};
use std::time::{Duration, Instant};

/// Leases start and end on 2 MiB boundaries of the arena.
pub const LEASE_ALIGN: usize = 2 << 20;
const GATHER_CHUNK: usize = 1 << 20;
/// Ages a REPORT carries at most.
pub const MAX_AGES: u64 = 64;

/// How group leases share the node.
#[derive(Clone, Debug)]
pub struct Share {
    /// Bytes of one segment, the unit a group lease grows and shrinks by (rounded up to LEASE_ALIGN).
    pub seg_bytes: usize,
    /// A group is asked to give only when its coldest blocks are this much older than the asking group's coldest.
    pub margin: Duration,
    /// A grant no rank grew into goes back to free after this.
    pub grant_ttl: Duration,
    /// A group that has not given what it was asked within this is marked stalled, and not asked again until it
    /// shrinks.
    pub stall: Duration,
    /// A group that gave is not asked again for this long.
    pub cooldown: Duration,
    /// Floors reserved from the start (kvman-remote --group-plan): group name -> (ranks, floor segments a rank). A
    /// planned group can start (or restart) whatever the other groups hold; its floor is kept for it while it is away.
    pub plan: BTreeMap<String, (u32, u64)>,
}

impl Default for Share {
    fn default() -> Share {
        Share {
            seg_bytes: 1 << 30,
            margin: Duration::from_secs(60),
            grant_ttl: Duration::from_secs(30),
            stall: Duration::from_secs(10),
            cooldown: Duration::from_secs(10),
            plan: BTreeMap::new(),
        }
    }
}

pub struct Config {
    pub bytes: usize,
    /// mlock the arena (always, outside tests).
    pub lock: bool,
    /// Offer RDMA where the connection's local address has a RoCE v2 GID.
    pub rdma: bool,
    /// The ack timeout exponent for the node's QPs (verbs::Qp::connect).
    pub qp_timeout: u8,
    pub share: Share,
}

struct Member {
    lease: u64,
    segs: u64,
}

/// A grant (grow) or give (shrink) of a whole group: every rank to `target` segments.
struct Txn {
    id: u64,
    grow: bool,
    target: u64,
    at: Instant,
    // the head was told (REPORT); a give made by another group's ask is told at the group's next report
    told: bool,
    // a rank applied it: from then on it neither expires nor stalls
    started: bool,
    // grow: each rank's part not yet taken
    held: BTreeMap<u32, u64>,
}

struct Group {
    nr: u32,
    block_bytes: usize,
    floor: u64,
    cap: u64,
    weight: u64,
    members: BTreeMap<u32, Member>,
    // segments held for ranks that have not opened yet (their floors)
    boot: u64,
    // the grant or give in progress, and the last transaction id
    txn: Option<Txn>,
    last_txn: u64,
    // segments ranks gave back in the open give, held until every rank gave (then free)
    retired: Vec<usize>,
    // segments a rank more than it asked that a give made for it left over (taken with its next grant, so the give's
    // rounding to the giver's rank count strands nothing)
    bonus: u64,
    stalled: bool,
    calm_until: Option<Instant>,
    ages: Vec<u64>,
    demand: u64,
    reports: u64,
    granted: u64,
    gave: u64,
    asked: u64,
}

impl Group {
    // segments per rank: the most any rank holds
    fn segs(&self) -> u64 {
        self.members.values().map(|m| m.segs).max().unwrap_or(self.floor)
    }
    fn reserved(&self) -> u64 {
        self.boot + self.txn.as_ref().map_or(0, |t| t.held.values().sum::<u64>())
    }
    fn complete(&self) -> bool {
        self.members.len() == self.nr as usize
    }
    // the age of its coldest blocks as the node compares them: weight 1000 is 1x, 2000 counts them half as old
    fn cold(&self) -> u64 {
        self.ages.first().map_or(0, |&a| a.saturating_mul(1000) / self.weight.max(1))
    }
}

struct State {
    free: BTreeMap<usize, usize>,
    leases: BTreeMap<u64, Weak<Lease>>,
    names: BTreeMap<String, u64>,
    groups: BTreeMap<String, Group>,
    next_id: u64,
    // Share::plan: the floors of planned groups, reserved while the group is away
    plan: BTreeMap<String, (u32, u64)>,
    // segments earmarked for a rank waiting in GOPEN while other groups give
    waiting: BTreeMap<String, u64>,
}

impl State {
    fn free_segs(&self, seg: usize) -> u64 {
        self.free.values().map(|&l| (l / seg) as u64).sum()
    }
    fn planned(&self, name: &str) -> u64 {
        match self.plan.get(name) {
            Some(&(nr, floor)) if !self.groups.contains_key(name) => nr as u64 * floor,
            _ => 0,
        }
    }
    fn reserved(&self) -> u64 {
        self.groups.values().map(Group::reserved).sum::<u64>()
            + self.plan.keys().map(|n| self.planned(n)).sum::<u64>()
            + self.waiting.values().sum::<u64>()
    }
    // segments the gives in progress will free
    fn coming(&self) -> u64 {
        self.groups
            .values()
            .filter_map(|g| g.txn.as_ref().filter(|t| !t.grow).map(|t| (g.segs().saturating_sub(t.target)) * g.nr as u64))
            .sum()
    }
    // segments nobody holds or has been promised
    fn unclaimed(&self, seg: usize) -> u64 {
        self.free_segs(seg).saturating_sub(self.reserved())
    }
    // a grant no rank took within grant_ttl goes back; a give no rank started within stall is dropped and the group
    // marked stalled. Once a rank applied either, it stays until every rank did (or left)
    fn tick(&mut self, now: Instant, sh: &Share) {
        for g in self.groups.values_mut() {
            let Some(t) = &g.txn else { continue };
            if t.started {
                continue;
            }
            if t.grow && now.duration_since(t.at) > sh.grant_ttl {
                g.txn = None;
            } else if !t.grow && now.duration_since(t.at) > sh.stall {
                g.txn = None;
                g.stalled = true;
            }
        }
    }
    // ends group `name`'s transaction once every rank it has applied it: a give's retired segments go free then
    fn settle(&mut self, name: &str, now: Instant, sh: &Share, seg: usize) {
        let Some(g) = self.groups.get_mut(name) else { return };
        let Some(t) = &g.txn else { return };
        let done = if t.grow {
            g.members.values().all(|m| m.segs >= t.target)
        } else {
            g.members.values().all(|m| m.segs <= t.target)
        };
        if !done {
            return;
        }
        if !t.grow {
            g.calm_until = Some(now + sh.cooldown);
        }
        g.txn = None;
        for o in std::mem::take(&mut g.retired) {
            release(&mut self.free, o, seg);
        }
    }
    fn open_txn(g: &mut Group, grow: bool, target: u64, per_rank: u64, now: Instant, told: bool) -> u64 {
        g.last_txn += 1;
        let held = if grow { g.members.keys().map(|&r| (r, per_rank)).collect() } else { BTreeMap::new() };
        g.txn = Some(Txn { id: g.last_txn, grow, target, at: now, told, started: false, held });
        g.last_txn
    }
    // the group `req` is `short` segments short: ask the group with the coldest blocks to give them, if any is colder
    // than `req`'s own coldest by the margin. floor: `req` asks for its floor (a group starting): the margin and the
    // cooldown do not apply, only the giver's own floor. The segments the give frees in all (k a rank x its ranks)
    fn ask(&mut self, req: &str, short: u64, now: Instant, sh: &Share, floor: bool) -> u64 {
        let mine = self.groups.get(req).map_or(0, Group::cold);
        let margin = sh.margin.as_millis() as u64;
        let pick = self
            .groups
            .iter()
            .filter(|(n, v)| {
                *n != req
                    && v.complete()
                    && v.txn.is_none()
                    && !v.stalled
                    && v.segs() > v.floor
                    && (floor
                        || (v.calm_until.is_none_or(|t| now >= t)
                            && !v.ages.is_empty()
                            && v.cold() > mine.saturating_add(margin)))
            })
            .max_by_key(|(_, v)| v.cold())
            .map(|(n, _)| n.clone());
        let Some(n) = pick else { return 0 };
        let v = self.groups.get_mut(&n).expect("picked");
        let k = short.div_ceil(v.nr as u64).min(v.segs() - v.floor);
        let target = v.segs() - k;
        State::open_txn(v, false, target, 0, now, false);
        v.asked += 1;
        k * v.nr as u64
    }
}

pub struct Shared {
    arena: Arena,
    state: Mutex<State>,
    devices: Mutex<HashMap<Ipv4Addr, Arc<Device>>>,
    cfg_rdma: bool,
    qp_timeout: u8,
    share: Share,
    started: Instant,
    opened: AtomicU64,
}

/// A lease's protection domain and one memory region per segment, on one device.
type Region = (Arc<Pd>, Vec<Arc<Mr>>);

pub struct Lease {
    pub id: u64,
    pub name: String,
    pub block_bytes: usize,
    // lease bytes per segment: the whole lease (OPEN), or a whole number of blocks (GOPEN)
    span: usize,
    // arena bytes per segment
    seg_len: usize,
    // each segment's offset in the arena, in lease order. Lock order: regions, then segs, then the node's state
    segs: RwLock<Vec<usize>>,
    group: Option<(String, u32)>,
    token: [u8; 16],
    regions: Mutex<HashMap<String, Region>>,
    conns: AtomicU64,
    read: AtomicU64,
    written: AtomicU64,
    since: Instant,
    shared: Arc<Shared>,
}

impl Lease {
    /// Bytes of the lease now.
    pub fn len(&self) -> usize {
        self.segs.read().unwrap().len() * self.span
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    // the memory of lease bytes off..off + len, which must lie in one segment
    fn addr(&self, segs: &[usize], off: u64, len: u64) -> Option<*mut u8> {
        let span = self.span as u64;
        let (i, at) = ((off / span) as usize, off % span);
        if i >= segs.len() || len > span - at {
            return None;
        }
        Some(unsafe { self.shared.arena.base().add(segs[i] + at as usize) })
    }
    /// This lease's region on `dev`, registered on first use; a segment added since gets its region now.
    fn region(&self, dev: &Arc<Device>) -> io::Result<Region> {
        let mut r = self.regions.lock().unwrap();
        let segs = self.segs.read().unwrap();
        let (pd, mrs) = match r.get_mut(&dev.name) {
            Some(x) => x,
            None => {
                let pd = Pd::new(dev)?;
                r.entry(dev.name.clone()).or_insert((pd, Vec::new()))
            }
        };
        while mrs.len() < segs.len() {
            let p = unsafe { self.shared.arena.base().add(segs[mrs.len()]) };
            mrs.push(Arc::new(Mr::new(
                pd,
                p,
                self.span,
                verbs::ACCESS_LOCAL_WRITE | verbs::ACCESS_REMOTE_READ | verbs::ACCESS_REMOTE_WRITE,
            )?));
        }
        Ok((pd.clone(), mrs.clone()))
    }
    fn zero(&self, offs: &[usize]) {
        for &o in offs {
            unsafe { std::ptr::write_bytes(self.shared.arena.base().add(o), 0, self.seg_len) };
        }
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        // no QP of this lease is left (each connection holds the lease until its QPs are gone); deregister before the
        // ranges can be granted again
        self.regions.lock().unwrap().clear();
        let segs = std::mem::take(self.segs.get_mut().unwrap());
        let mut st = self.shared.state.lock().unwrap();
        st.leases.remove(&self.id);
        if st.names.get(&self.name) == Some(&self.id) {
            st.names.remove(&self.name);
        }
        for o in segs {
            release(&mut st.free, o, self.seg_len);
        }
        if let Some((g, rank)) = &self.group {
            let gone = st.groups.get_mut(g).is_some_and(|x| {
                if x.members.get(rank).is_some_and(|m| m.lease == self.id) {
                    x.members.remove(rank);
                    if let Some(t) = &mut x.txn {
                        t.held.remove(rank);
                    }
                }
                x.members.is_empty()
            });
            // the last rank of a group took the group with it, and what it held for the others; a rank that left ends
            // a transaction the others have all applied
            if gone {
                if let Some(x) = st.groups.remove(g) {
                    for o in x.retired {
                        release(&mut st.free, o, self.seg_len);
                    }
                }
            } else {
                st.settle(g, Instant::now(), &self.shared.share, self.seg_len);
            }
        }
    }
}

fn release(free: &mut BTreeMap<usize, usize>, mut off: usize, mut len: usize) {
    // a range freed twice would be granted twice: two leases on the same bytes
    let prev_end = free.range(..=off).next_back().map_or(0, |(&po, &pl)| po + pl);
    let next = free.range(off..).next().map_or(usize::MAX, |(&no, _)| no);
    assert!(prev_end <= off && off + len <= next, "range {off}+{len} freed while part of it is free");
    if let Some((&po, &pl)) = free.range(..off).next_back() {
        if po + pl == off {
            free.remove(&po);
            off = po;
            len += pl;
        }
    }
    if let Some(&nl) = free.get(&(off + len)) {
        free.remove(&(off + len));
        len += nl;
    }
    free.insert(off, len);
}

/// First fit: the lowest free range that holds `len`.
fn alloc(free: &mut BTreeMap<usize, usize>, len: usize) -> Option<usize> {
    let (&off, &have) = free.iter().find(|(_, &l)| l >= len)?;
    free.remove(&off);
    if have > len {
        free.insert(off + len, have - len);
    }
    Some(off)
}

// n segments of `seg` bytes, or none (nothing taken)
fn alloc_segs(free: &mut BTreeMap<usize, usize>, seg: usize, n: u64) -> Option<Vec<usize>> {
    let mut got = Vec::new();
    for _ in 0..n {
        match alloc(free, seg) {
            Some(o) => got.push(o),
            None => {
                for o in got {
                    release(free, o, seg);
                }
                return None;
            }
        }
    }
    Some(got)
}

pub struct Node {
    pub shared: Arc<Shared>,
    listener: TcpListener,
}

impl Node {
    pub fn new(cfg: &Config, listen: SocketAddr) -> io::Result<Node> {
        let bytes = cfg.bytes / LEASE_ALIGN * LEASE_ALIGN;
        if bytes == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "the node needs at least 2 MiB"));
        }
        let mut share = cfg.share.clone();
        share.seg_bytes = share.seg_bytes.max(1).div_ceil(LEASE_ALIGN) * LEASE_ALIGN;
        let listener = TcpListener::bind(listen)?;
        let arena = Arena::new(bytes, cfg.lock)?;
        let mut free = BTreeMap::new();
        free.insert(0, bytes);
        let shared = Arc::new(Shared {
            arena,
            state: Mutex::new(State {
                free,
                leases: BTreeMap::new(),
                names: BTreeMap::new(),
                groups: BTreeMap::new(),
                next_id: 1,
                plan: share.plan.clone(),
                waiting: BTreeMap::new(),
            }),
            devices: Mutex::new(HashMap::new()),
            cfg_rdma: cfg.rdma,
            qp_timeout: cfg.qp_timeout,
            share,
            started: Instant::now(),
            opened: AtomicU64::new(0),
        });
        Ok(Node { shared, listener })
    }

    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.listener.local_addr()
    }

    /// Accepts connections until the listener fails, a thread each.
    pub fn serve(&self) -> io::Result<()> {
        loop {
            let (s, _) = match self.listener.accept() {
                Ok(x) => x,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            };
            let sh = self.shared.clone();
            std::thread::spawn(move || {
                let peer = s.peer_addr().ok();
                if let Err(e) = conn(&sh, s) {
                    if e.kind() != io::ErrorKind::UnexpectedEof && e.kind() != io::ErrorKind::ConnectionReset {
                        eprintln!("kvman-remote: {}: {e}", peer.map_or("?".into(), |p| p.to_string()));
                    }
                }
            });
        }
    }

    /// Serves on a background thread (tests, the bench's local mode).
    pub fn spawn(self) -> SocketAddr {
        let a = self.local_addr().expect("bound");
        std::thread::spawn(move || self.serve());
        a
    }
}

impl Shared {
    pub fn bytes(&self) -> usize {
        self.arena.len()
    }
    pub fn locked(&self) -> bool {
        self.arena.locked
    }
    pub fn seg_bytes(&self) -> usize {
        self.share.seg_bytes
    }
    fn device(&self, ip: Ipv4Addr) -> io::Result<Arc<Device>> {
        let mut d = self.devices.lock().unwrap();
        if let Some(x) = d.get(&ip) {
            return Ok(x.clone());
        }
        let dev = Device::open(ip)?;
        d.insert(ip, dev.clone());
        Ok(dev)
    }

    /// The node's state as JSON: its size, what is free, each group and each lease.
    pub fn stats_json(&self) -> String {
        // a lease upgraded here may be the last reference, and its drop takes the state lock: collect, then unlock
        let seg = self.share.seg_bytes;
        let (free, largest, free_segs, unclaimed, groups, leases) = {
            let mut st = self.state.lock().unwrap();
            st.tick(Instant::now(), &self.share);
            let leases: Vec<Arc<Lease>> = st.leases.values().filter_map(Weak::upgrade).collect();
            let groups: Vec<String> = st
                .groups
                .iter()
                .map(|(n, g)| {
                    let segs: Vec<String> = g.members.iter().map(|(r, m)| format!("\"{r}\":{}", m.segs)).collect();
                    let ages: Vec<String> = g.ages.iter().map(u64::to_string).collect();
                    format!(
                        "{{\"name\":\"{}\",\"nr\":{},\"block\":{},\"floor\":{},\"cap\":{},\"weight\":{},\"segs\":{{{}}},\
                         \"boot\":{},\"txn\":{},\"grant\":{},\"give\":{},\"retired\":{},\"bonus\":{},\"stalled\":{},\
                         \"demand\":{},\"ages_ms\":[{}],\"reports\":{},\"granted\":{},\"gave\":{},\"asked\":{}}}",
                        json_escape(n),
                        g.nr,
                        g.block_bytes,
                        g.floor,
                        g.cap,
                        g.weight,
                        segs.join(","),
                        g.boot,
                        g.txn.as_ref().map_or(0, |t| t.id),
                        g.txn.as_ref().map_or(0, |t| t.held.values().sum::<u64>()),
                        g.txn.as_ref().filter(|t| !t.grow).map_or(-1, |t| t.target as i64),
                        g.retired.len(),
                        g.bonus,
                        g.stalled,
                        g.demand,
                        ages.join(","),
                        g.reports,
                        g.granted,
                        g.gave,
                        g.asked
                    )
                })
                .collect();
            (
                st.free.values().sum::<usize>(),
                st.free.values().copied().max().unwrap_or(0),
                st.free_segs(seg),
                st.unclaimed(seg),
                groups,
                leases,
            )
        };
        let mut s = format!(
            "{{\"bytes\":{},\"free\":{},\"largest_free\":{},\"seg_bytes\":{},\"free_segs\":{},\"unclaimed_segs\":{},\
             \"locked\":{},\"uptime_s\":{},\"opened\":{},\"groups\":[{}],\"leases\":[",
            self.arena.len(),
            free,
            largest,
            seg,
            free_segs,
            unclaimed,
            self.arena.locked,
            self.started.elapsed().as_secs(),
            self.opened.load(Ordering::Relaxed),
            groups.join(",")
        );
        let mut first = true;
        for l in &leases {
            if !first {
                s.push(',');
            }
            first = false;
            let devs: Vec<String> =
                l.regions.lock().unwrap().keys().map(|k| format!("\"{}\"", json_escape(k))).collect();
            let group = l
                .group
                .as_ref()
                .map_or(String::new(), |(g, r)| format!(",\"group\":\"{}\",\"rank\":{r}", json_escape(g)));
            s += &format!(
                "{{\"id\":{},\"name\":\"{}\",\"bytes\":{},\"segs\":{},\"block\":{},\"conns\":{},\"read\":{},\
                 \"written\":{},\"rdma\":[{}],\"age_s\":{},\"at\":[{}]{group}}}",
                l.id,
                json_escape(&l.name),
                l.len(),
                l.segs.read().unwrap().len(),
                l.block_bytes,
                l.conns.load(Ordering::Relaxed),
                l.read.load(Ordering::Relaxed),
                l.written.load(Ordering::Relaxed),
                devs.join(","),
                l.since.elapsed().as_secs(),
                // each segment's place in the arena, in segments of seg_len
                l.segs.read().unwrap().iter().map(|o| (o / l.seg_len).to_string()).collect::<Vec<_>>().join(",")
            );
        }
        s + "]}"
    }
}

pub fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn fail(s: &mut TcpStream, code: u32, msg: &str) -> io::Result<()> {
    Header::new(code, 0, 0, msg.len() as u64).write(s)?;
    s.write_all(msg.as_bytes())
}

fn read_vec(s: &mut TcpStream, n: usize) -> io::Result<Vec<u8>> {
    let mut v = vec![0u8; n];
    s.read_exact(&mut v)?;
    Ok(v)
}

fn u64_at(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

struct Attached(Arc<Lease>);

impl Drop for Attached {
    fn drop(&mut self) {
        self.0.conns.fetch_sub(1, Ordering::Relaxed);
    }
}

fn conn(sh: &Arc<Shared>, mut s: TcpStream) -> io::Result<()> {
    s.set_nodelay(true)?;
    let fd = s.as_raw_fd();
    // declared before the lease, so this connection's QPs go first
    let mut lease: Option<Attached> = None;
    let mut qps: Vec<Qp> = Vec::new();
    let mut dev: Option<Arc<Device>> = None;
    loop {
        let h = match Header::read(&mut s) {
            Ok(h) => h,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        };
        match h.op {
            proto::STATS => {
                let j = sh.stats_json();
                Header::new(proto::OK, j.len() as u64, 0, 0).write(&mut s)?;
                s.write_all(j.as_bytes())?;
            }
            proto::OPEN | proto::ATTACH | proto::GOPEN if lease.is_some() => {
                fail(&mut s, proto::E_PROTO, "this connection already holds a lease")?;
                break;
            }
            proto::OPEN => {
                if h.c as usize > proto::NAME_MAX {
                    fail(&mut s, proto::E_PROTO, "lease name too long")?;
                    break;
                }
                let name = read_vec(&mut s, h.c as usize)?;
                if !proto::valid_name(&name) {
                    fail(&mut s, proto::E_PROTO, "a lease name is 1-128 printable ASCII characters, no spaces")?;
                    continue;
                }
                let name = String::from_utf8(name).expect("ASCII");
                match open(sh, &name, h.a, h.b) {
                    Ok(l) => {
                        Header::new(proto::OK, l.id, l.len() as u64, 0).write(&mut s)?;
                        s.write_all(&l.token)?;
                        l.conns.fetch_add(1, Ordering::Relaxed);
                        lease = Some(Attached(l));
                    }
                    Err((code, msg)) => fail(&mut s, code, &msg)?,
                }
            }
            proto::GOPEN => {
                if h.a < proto::GOPEN_FIXED as u64 || h.a as usize > proto::GOPEN_FIXED + 2 * proto::NAME_MAX + 1 {
                    fail(&mut s, proto::E_PROTO, "bad GOPEN length")?;
                    break;
                }
                let b = read_vec(&mut s, h.a as usize)?;
                let names = &b[proto::GOPEN_FIXED..];
                let Some(sp) = names.iter().position(|&c| c == b' ') else {
                    fail(&mut s, proto::E_PROTO, "GOPEN names: \"<group> <lease>\"")?;
                    continue;
                };
                let (group, name) = (&names[..sp], &names[sp + 1..]);
                if !proto::valid_name(group) || !proto::valid_name(name) {
                    fail(&mut s, proto::E_PROTO, "a group or lease name is 1-128 printable ASCII characters")?;
                    continue;
                }
                let want = Want {
                    group: String::from_utf8(group.to_vec()).expect("ASCII"),
                    name: String::from_utf8(name.to_vec()).expect("ASCII"),
                    block_bytes: u64_at(&b, 0),
                    floor_bytes: u64_at(&b, 8),
                    cap_bytes: u64_at(&b, 16),
                    nr: u32_at(&b, 24),
                    rank: u32_at(&b, 28),
                    weight: u32_at(&b, 32),
                };
                match gopen(sh, &want) {
                    Ok((l, bps)) => {
                        let n = l.segs.read().unwrap().len() as u64;
                        Header::new(proto::OK, l.id, n, bps).write(&mut s)?;
                        s.write_all(&l.token)?;
                        l.conns.fetch_add(1, Ordering::Relaxed);
                        lease = Some(Attached(l));
                    }
                    Err((code, msg)) => fail(&mut s, code, &msg)?,
                }
            }
            proto::ATTACH => {
                let token = read_vec(&mut s, 16)?;
                let l = sh.state.lock().unwrap().leases.get(&h.a).and_then(Weak::upgrade);
                match l {
                    Some(l) if l.token[..] == token[..] => {
                        Header::new(proto::OK, l.id, l.len() as u64, 0).write(&mut s)?;
                        l.conns.fetch_add(1, Ordering::Relaxed);
                        lease = Some(Attached(l));
                    }
                    Some(_) => fail(&mut s, proto::E_TOKEN, "wrong token for that lease")?,
                    None => fail(&mut s, proto::E_NOLEASE, "no such lease")?,
                }
            }
            _ if lease.is_none() => {
                fail(&mut s, proto::E_NOLEASE, "open or attach a lease first")?;
                break;
            }
            proto::RDMA => {
                let l = lease.as_ref().unwrap().0.clone();
                if h.a == 0 || h.a > proto::MAX_QPS || !qps.is_empty() {
                    fail(&mut s, proto::E_PROTO, "bad QP count, or this connection has QPs already")?;
                    break;
                }
                let gid: [u8; 16] = read_vec(&mut s, 16)?.try_into().unwrap();
                let ends: Vec<QpEnd> = read_vec(&mut s, h.a as usize * 8)?
                    .chunks(8)
                    .map(|c| QpEnd {
                        qpn: u32::from_le_bytes(c[0..4].try_into().unwrap()),
                        psn: u32::from_le_bytes(c[4..8].try_into().unwrap()),
                    })
                    .collect();
                match rdma(sh, &l, &s, &gid, &ends, h.b as u32, h.c as u32) {
                    Ok((d, mr, mine)) => {
                        let c = d.mtu as u64 | (d.max_rd_atom as u64) << 32;
                        Header::new(proto::OK, mr.addr(), mr.rkey() as u64, c).write(&mut s)?;
                        let mut b = d.gid.to_vec();
                        for q in &mine {
                            let e = q.end();
                            b.extend_from_slice(&e.qpn.to_le_bytes());
                            b.extend_from_slice(&e.psn.to_le_bytes());
                        }
                        s.write_all(&b)?;
                        qps = mine;
                        dev = Some(d);
                    }
                    Err(e) => fail(&mut s, proto::E_RDMA, &e.to_string())?,
                }
            }
            proto::SEGS => {
                let l = lease.as_ref().unwrap().0.clone();
                let Some(d) = &dev else {
                    fail(&mut s, proto::E_PROTO, "SEGS needs RDMA set up on this connection")?;
                    continue;
                };
                match l.region(d) {
                    Ok((_, mrs)) => {
                        Header::new(proto::OK, mrs.len() as u64, l.span as u64, 0).write(&mut s)?;
                        let mut b = Vec::with_capacity(mrs.len() * 16);
                        for m in &mrs {
                            b.extend_from_slice(&m.addr().to_le_bytes());
                            b.extend_from_slice(&(m.rkey() as u64).to_le_bytes());
                        }
                        s.write_all(&b)?;
                    }
                    Err(e) => fail(&mut s, proto::E_RDMA, &e.to_string())?,
                }
            }
            proto::REPORT => {
                let l = lease.as_ref().unwrap().0.clone();
                if h.b > MAX_AGES {
                    fail(&mut s, proto::E_PROTO, "too many ages")?;
                    break;
                }
                let b = read_vec(&mut s, h.b as usize * 8)?;
                let ages: Vec<u64> = b.chunks(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
                match report(sh, &l, h.a, ages) {
                    Ok((grant, give, unclaimed, txn)) => {
                        let mut b = Header::new(proto::OK, grant, give, unclaimed).bytes().to_vec();
                        b.extend_from_slice(&txn.to_le_bytes());
                        s.write_all(&b)?
                    }
                    Err((code, msg)) => fail(&mut s, code, &msg)?,
                }
            }
            proto::GROW => {
                let l = lease.as_ref().unwrap().0.clone();
                match grow(sh, &l, h.a, h.b) {
                    Ok(n) => Header::new(proto::OK, n, 0, 0).write(&mut s)?,
                    Err((code, msg)) => fail(&mut s, code, &msg)?,
                }
            }
            proto::SHRINK => {
                let l = lease.as_ref().unwrap().0.clone();
                match shrink(sh, &l, h.a, h.b) {
                    Ok(n) => Header::new(proto::OK, n, 0, 0).write(&mut s)?,
                    Err((code, msg)) => fail(&mut s, code, &msg)?,
                }
            }
            proto::READ => {
                let l = &lease.as_ref().unwrap().0;
                let segs = l.segs.read().unwrap();
                let Some(p) = l.addr(&segs, h.a, h.b) else {
                    let n = segs.len() * l.span;
                    drop(segs);
                    fail(&mut s, proto::E_RANGE, &format!("read of {} at {} past the lease ({n})", h.b, h.a))?;
                    continue;
                };
                Header::new(proto::OK, h.b, 0, 0).write(&mut s)?;
                unsafe { sys::send_all(fd, p, h.b as usize)? };
                l.read.fetch_add(h.b, Ordering::Relaxed);
            }
            proto::WRITE => {
                let l = &lease.as_ref().unwrap().0;
                let segs = l.segs.read().unwrap();
                let Some(p) = l.addr(&segs, h.a, h.b) else {
                    // the payload is on its way: the stream is out of step, so this connection ends
                    let n = segs.len() * l.span;
                    drop(segs);
                    fail(&mut s, proto::E_RANGE, &format!("write of {} at {} past the lease ({n})", h.b, h.a))?;
                    break;
                };
                unsafe { sys::recv_exact(fd, p, h.b as usize)? };
                drop(segs);
                l.written.fetch_add(h.b, Ordering::Relaxed);
                Header::new(proto::OK, 0, 0, 0).write(&mut s)?;
            }
            proto::GATHER => {
                let l = &lease.as_ref().unwrap().0;
                if h.a > proto::MAX_RANGES {
                    fail(&mut s, proto::E_PROTO, "too many ranges")?;
                    break;
                }
                let r = read_vec(&mut s, h.a as usize * 16)?;
                let ranges: Vec<(u64, u64)> = r.chunks(16).map(|c| (u64_at(c, 0), u64_at(c, 8))).collect();
                let segs = l.segs.read().unwrap();
                let ptrs: Vec<Option<*mut u8>> = ranges.iter().map(|&(o, n)| l.addr(&segs, o, n)).collect();
                if let Some(i) = ptrs.iter().position(Option::is_none) {
                    let (o, n) = ranges[i];
                    let len = segs.len() * l.span;
                    drop(segs);
                    fail(&mut s, proto::E_RANGE, &format!("gather of {n} at {o} past the lease ({len})"))?;
                    continue;
                }
                let total: u64 = ranges.iter().map(|r| r.1).sum();
                Header::new(proto::OK, total, 0, 0).write(&mut s)?;
                // rows are small: one send per row costs more than copying them together
                let mut buf = Vec::with_capacity(GATHER_CHUNK);
                for (&(_, n), p) in ranges.iter().zip(ptrs) {
                    let p = p.expect("checked");
                    if buf.len() + n as usize > GATHER_CHUNK && !buf.is_empty() {
                        s.write_all(&buf)?;
                        buf.clear();
                    }
                    if n as usize >= GATHER_CHUNK {
                        unsafe { sys::send_all(fd, p, n as usize)? };
                    } else {
                        buf.extend_from_slice(unsafe { std::slice::from_raw_parts(p, n as usize) });
                    }
                }
                s.write_all(&buf)?;
                drop(segs);
                l.read.fetch_add(total, Ordering::Relaxed);
            }
            op => {
                fail(&mut s, proto::E_PROTO, &format!("unknown request {op}"))?;
                break;
            }
        }
    }
    drop(qps);
    drop(lease);
    Ok(())
}

type Refusal = (u32, String);

fn new_lease(
    sh: &Arc<Shared>,
    st: &mut State,
    name: &str,
    block_bytes: usize,
    span: usize,
    seg_len: usize,
    segs: Vec<usize>,
    group: Option<(String, u32)>,
) -> Result<Arc<Lease>, Refusal> {
    let id = st.next_id;
    st.next_id += 1;
    let token = sys::random16().map_err(|e| (proto::E_PROTO, e.to_string()))?;
    let l = Arc::new(Lease {
        id,
        name: name.to_string(),
        block_bytes,
        span,
        seg_len,
        segs: RwLock::new(segs),
        group,
        token,
        regions: Mutex::new(HashMap::new()),
        conns: AtomicU64::new(0),
        read: AtomicU64::new(0),
        written: AtomicU64::new(0),
        since: Instant::now(),
        shared: sh.clone(),
    });
    st.leases.insert(id, Arc::downgrade(&l));
    st.names.insert(name.to_string(), id);
    Ok(l)
}

fn open(sh: &Arc<Shared>, name: &str, block_bytes: u64, blocks: u64) -> Result<Arc<Lease>, Refusal> {
    let len = block_bytes
        .checked_mul(blocks)
        .filter(|&n| block_bytes > 0 && n > 0 && n <= sh.arena.len() as u64)
        .ok_or((proto::E_NOROOM, format!("{blocks} blocks of {block_bytes} bytes do not fit the node")))?
        as usize;
    let len_al = len.div_ceil(LEASE_ALIGN) * LEASE_ALIGN;
    let l = {
        let mut st = sh.state.lock().unwrap();
        st.tick(Instant::now(), &sh.share);
        if let Some(id) = st.names.get(name) {
            return Err((proto::E_BUSY, format!("lease {name} is held (id {id}); a name has one holder at a time")));
        }
        let no_room = |st: &State| {
            let free: usize = st.free.values().sum();
            let largest = st.free.values().copied().max().unwrap_or(0);
            (proto::E_NOROOM, format!("no room for {len} bytes: {free} free, the largest range {largest}"))
        };
        let Some(off) = alloc(&mut st.free, len_al) else { return Err(no_room(&st)) };
        // what groups were promised stays theirs
        if st.free_segs(sh.share.seg_bytes) < st.reserved() {
            release(&mut st.free, off, len_al);
            return Err(no_room(&st));
        }
        new_lease(sh, &mut st, name, block_bytes as usize, len_al, len_al, vec![off], None)?
    };
    // what the last holder of these bytes left is not this client's
    l.zero(&[l.segs.read().unwrap()[0]]);
    sh.opened.fetch_add(1, Ordering::Relaxed);
    Ok(l)
}

/// What a rank asks for when it joins a group (GOPEN).
pub struct Want {
    pub group: String,
    pub name: String,
    pub block_bytes: u64,
    pub floor_bytes: u64,
    pub cap_bytes: u64,
    pub nr: u32,
    pub rank: u32,
    /// 1000 = 1x. A group of weight 2000 is asked to give as if its blocks were half as old.
    pub weight: u32,
}

// the lease and its blocks per segment. A group's first rank needs room for every rank's floor, a later rank its own
// floor beyond what the first held for it. Short of room, the rank asks the other groups to give down to their
// floors (the margin does not apply: a floor is the group's right) and waits up to Share::stall for them, the room
// earmarked for it meanwhile; a planned group (Share::plan) finds its floors reserved
fn gopen(sh: &Arc<Shared>, w: &Want) -> Result<(Arc<Lease>, u64), Refusal> {
    let seg = sh.share.seg_bytes;
    if w.block_bytes == 0 || w.block_bytes > seg as u64 {
        return Err((proto::E_NOROOM, format!("a block of {} bytes does not fit a segment of {seg}", w.block_bytes)));
    }
    if w.nr == 0 || w.rank >= w.nr {
        return Err((proto::E_PROTO, format!("rank {} of {}", w.rank, w.nr)));
    }
    let bps = seg as u64 / w.block_bytes;
    let span = bps * w.block_bytes;
    let floor = w.floor_bytes.div_ceil(span).max(1);
    let cap = w.cap_bytes.div_ceil(span).max(floor);
    let weight = if w.weight == 0 { 1000 } else { w.weight as u64 };
    let deadline = Instant::now() + sh.share.stall;
    let l = loop {
        let mut st = sh.state.lock().unwrap();
        let now = Instant::now();
        st.tick(now, &sh.share);
        if let Some(id) = st.names.get(&w.name) {
            return Err((proto::E_BUSY, format!("lease {} is held (id {id}); a name has one holder", w.name)));
        }
        // what this rank may take: what nobody holds or was promised, what is earmarked for it, its planned floors
        st.waiting.remove(&w.group);
        let room = st.unclaimed(seg) + st.planned(&w.group);
        let need = match st.groups.get(&w.group) {
            Some(g) => {
                if (g.nr, g.block_bytes as u64, g.floor, g.cap) != (w.nr, w.block_bytes, floor, cap) {
                    return Err((
                        proto::E_PROTO,
                        format!(
                            "group {} has {} ranks, blocks of {}, floor {} and cap {} segments; this rank asks {}, {}, \
                             {} and {}",
                            w.group, g.nr, g.block_bytes, g.floor, g.cap, w.nr, w.block_bytes, floor, cap
                        ),
                    ));
                }
                if g.members.contains_key(&w.rank) {
                    return Err((proto::E_BUSY, format!("rank {} of group {} is held", w.rank, w.group)));
                }
                // a rank's floor comes out of what the group's first rank held for it
                floor - floor.min(g.boot)
            }
            None => floor * w.nr as u64,
        };
        if need > room {
            let short = need - room;
            let coming = st.coming();
            if short > coming {
                st.ask(&w.group, short - coming, now, &sh.share, true);
            }
            if now >= deadline {
                return Err((
                    proto::E_NOROOM,
                    format!(
                        "no room for group {} rank {}: {need} segments of {seg} bytes, {room} unclaimed, {} more \
                         being given back",
                        w.group,
                        w.rank,
                        st.coming()
                    ),
                ));
            }
            // earmark what there is, so a grant to another group does not take it while the gives land
            st.waiting.insert(w.group.clone(), room.min(need));
            drop(st);
            std::thread::sleep(Duration::from_millis(20));
            continue;
        }
        match st.groups.get_mut(&w.group) {
            Some(g) => g.boot -= floor.min(g.boot),
            None => {
                st.groups.insert(
                    w.group.clone(),
                    Group {
                        nr: w.nr,
                        block_bytes: w.block_bytes as usize,
                        floor,
                        cap,
                        weight,
                        members: BTreeMap::new(),
                        boot: floor * (w.nr as u64 - 1),
                        txn: None,
                        last_txn: 0,
                        retired: Vec::new(),
                        bonus: 0,
                        stalled: false,
                        calm_until: None,
                        ages: Vec::new(),
                        demand: 0,
                        reports: 0,
                        granted: 0,
                        gave: 0,
                        asked: 0,
                    },
                );
            }
        }
        let segs = alloc_segs(&mut st.free, seg, floor).ok_or((proto::E_NOROOM, "no room".to_string()))?;
        let l = new_lease(
            sh,
            &mut st,
            &w.name,
            w.block_bytes as usize,
            span as usize,
            seg,
            segs,
            Some((w.group.clone(), w.rank)),
        )?;
        st.groups.get_mut(&w.group).expect("made").members.insert(w.rank, Member { lease: l.id, segs: floor });
        break l;
    };
    l.zero(&l.segs.read().unwrap());
    sh.opened.fetch_add(1, Ordering::Relaxed);
    Ok((l, bps))
}

fn group_of(l: &Lease) -> Result<&(String, u32), Refusal> {
    l.group.as_ref().ok_or((proto::E_PROTO, "this lease is not in a group (GOPEN)".to_string()))
}

// (segments a rank grows by now, segments a rank gives back, unclaimed segments, the transaction to name in GROW or
// SHRINK; 0 = nothing to do). A transaction in progress is told once, to the first report after it opened
fn report(sh: &Arc<Shared>, l: &Lease, demand: u64, ages: Vec<u64>) -> Result<(u64, u64, u64, u64), Refusal> {
    let (name, rank) = group_of(l)?;
    let seg = sh.share.seg_bytes;
    let mut st = sh.state.lock().unwrap();
    let now = Instant::now();
    st.tick(now, &sh.share);
    let unclaimed = st.unclaimed(seg);
    let g = st.groups.get_mut(name).ok_or((proto::E_NOLEASE, format!("no group {name}")))?;
    g.ages = ages;
    g.demand = demand;
    g.reports += 1;
    let mine = g.members.get(rank).map_or(g.floor, |m| m.segs);
    let nr = g.nr as u64;
    let (mut grant, mut give, mut txn) = (0, 0, 0);
    // (segments short, of segments needed, with these unclaimed left)
    let mut short = None;
    if let Some(t) = &mut g.txn {
        if !t.told {
            t.told = true;
            txn = t.id;
            if t.grow {
                grant = t.target.saturating_sub(mine);
            } else {
                give = mine.saturating_sub(t.target);
            }
        }
    } else if demand > 0 && g.complete() {
        let room = g.cap.saturating_sub(g.segs());
        let want = demand.min(room);
        grant = (unclaimed / nr).min((want + g.bonus).min(room));
        if grant > 0 {
            g.bonus = 0;
            g.granted += grant;
            let target = g.segs() + grant;
            txn = State::open_txn(g, true, target, grant, now, true);
        }
        if want > grant {
            let need = (want - grant) * nr;
            let left = unclaimed - grant * nr;
            if need > left {
                short = Some((need - left, need, left));
            }
        }
    }
    if let Some((k, need, left)) = short {
        // the giver gives whole segments on each of its ranks: what that frees past this group's need, a whole
        // segment on each of its ranks, it takes with its next grant
        let freed = st.ask(name, k, now, &sh.share, false);
        if freed > 0 {
            st.groups.get_mut(name).expect("reported").bonus = (left + freed).saturating_sub(need) / nr;
        }
    }
    Ok((grant, give, st.unclaimed(seg), txn))
}

fn stale(g: &Group, txn: u64, grow: bool) -> Refusal {
    let open = g.txn.as_ref().map_or("none".to_string(), |t| format!("{} ({})", t.id, if t.grow { "grow" } else { "give" }));
    (
        proto::E_STALE,
        format!("{} under transaction {txn}: the group's open transaction is {open}", if grow { "GROW" } else { "SHRINK" }),
    )
}

// the lease's segments after growing to `target` under transaction `txn` (REPORT's). Done already: the segments as
// they are
fn grow(sh: &Arc<Shared>, l: &Lease, target: u64, txn: u64) -> Result<u64, Refusal> {
    let (name, rank) = group_of(l)?;
    let seg = sh.share.seg_bytes;
    let new = {
        let mut st = sh.state.lock().unwrap();
        let now = Instant::now();
        st.tick(now, &sh.share);
        let spare = st.free_segs(seg).saturating_sub(st.reserved());
        let g = st.groups.get(name).ok_or((proto::E_NOLEASE, format!("no group {name}")))?;
        let have = g.members.get(rank).ok_or((proto::E_NOLEASE, "not a member".to_string()))?.segs;
        let Some(t) = g.txn.as_ref().filter(|t| t.id == txn && t.grow) else {
            if have >= target && txn != 0 && txn <= g.last_txn {
                return Ok(have);
            }
            return Err(stale(g, txn, true));
        };
        if target != t.target {
            return Err((proto::E_PROTO, format!("transaction {txn} grows to {} segments, not {target}", t.target)));
        }
        if have >= target {
            return Ok(have);
        }
        let n = target - have;
        // from this rank's part of the grant, or else from what nobody was promised
        let from_held = n.min(t.held.get(rank).copied().unwrap_or(0));
        if n - from_held > spare {
            return Err((proto::E_NOROOM, format!("no room for {n} segments: {from_held} granted, {spare} spare")));
        }
        // the rank's part leaves the reservation as its segments leave the free list
        let g = st.groups.get_mut(name).expect("checked");
        let t = g.txn.as_mut().expect("checked");
        if let Some(h) = t.held.get_mut(rank) {
            *h -= from_held;
        }
        t.started = true;
        let new = match alloc_segs(&mut st.free, seg, n) {
            Some(v) => v,
            None => {
                let t = st.groups.get_mut(name).expect("checked").txn.as_mut().expect("checked");
                if let Some(h) = t.held.get_mut(rank) {
                    *h += from_held;
                }
                return Err((proto::E_NOROOM, "no room".to_string()));
            }
        };
        st.groups.get_mut(name).expect("checked").members.get_mut(rank).expect("member").segs = target;
        st.settle(name, now, &sh.share, seg);
        new
    };
    l.zero(&new);
    let mut segs = l.segs.write().unwrap();
    segs.extend(new);
    Ok(segs.len() as u64)
}

// the lease's segments after shrinking to `to` under the group's give `txn`. The client sends nothing to the
// segments that go after it asks. They stay the group's until every rank has given (then free)
fn shrink(sh: &Arc<Shared>, l: &Lease, to: u64, txn: u64) -> Result<u64, Refusal> {
    let (name, rank) = group_of(l)?;
    {
        let mut st = sh.state.lock().unwrap();
        let g = st.groups.get_mut(name).ok_or((proto::E_NOLEASE, format!("no group {name}")))?;
        let have = g.members.get(rank).map_or(0, |m| m.segs);
        let Some(t) = g.txn.as_mut().filter(|t| t.id == txn && !t.grow) else {
            if have <= to && txn != 0 && txn <= g.last_txn {
                return Ok(have);
            }
            return Err(stale(g, txn, false));
        };
        if to != t.target {
            return Err((proto::E_PROTO, format!("transaction {txn} gives down to {} segments, not {to}", t.target)));
        }
        if to < g.floor {
            return Err((proto::E_RANGE, format!("{to} segments is below the group's floor {}", g.floor)));
        }
        // from here the give completes: it never stalls or expires
        t.started = true;
    }
    let gone = {
        let mut r = l.regions.lock().unwrap();
        let mut segs = l.segs.write().unwrap();
        if to as usize >= segs.len() {
            Vec::new()
        } else {
            for (_, mrs) in r.values_mut() {
                mrs.truncate(to as usize);
            }
            segs.split_off(to as usize)
        }
    };
    let mut st = sh.state.lock().unwrap();
    let now = Instant::now();
    match st.groups.get_mut(name) {
        Some(g) => {
            if let Some(m) = g.members.get_mut(rank) {
                m.segs = to;
            }
            g.stalled = false;
            if rank == &0 {
                g.gave += gone.len() as u64;
            }
            g.retired.extend(gone);
            st.settle(name, now, &sh.share, l.seg_len);
        }
        None => {
            for o in gone {
                release(&mut st.free, o, l.seg_len);
            }
        }
    }
    Ok(to)
}

type RdmaSetup = (Arc<Device>, Arc<Mr>, Vec<Qp>);

fn rdma(
    sh: &Shared,
    l: &Lease,
    s: &TcpStream,
    gid: &[u8; 16],
    ends: &[QpEnd],
    mtu: u32,
    rd_atom: u32,
) -> io::Result<RdmaSetup> {
    if !sh.cfg_rdma {
        return Err(io::Error::new(io::ErrorKind::Unsupported, "RDMA is off on this node"));
    }
    let IpAddr::V4(ip) = s.local_addr()?.ip() else {
        return Err(io::Error::new(io::ErrorKind::Unsupported, "RDMA needs an IPv4 connection"));
    };
    let dev = sh.device(ip)?;
    let (pd, mrs) = l.region(&dev)?;
    let mr = mrs.first().cloned().ok_or_else(|| io::Error::other("a lease without memory"))?;
    let cq = Cq::new(&dev, ends.len() as u32 * 2 + 4)?;
    let mtu = mtu.min(dev.mtu);
    let mut qps = Vec::new();
    for e in ends {
        let q = Qp::new(&pd, &cq, 1, true, verbs::psn())?;
        q.connect(*e, gid, mtu, 0, dev.max_rd_atom.min(rd_atom.max(1)).max(1), sh.qp_timeout)?;
        qps.push(q);
    }
    Ok((dev, mr, qps))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_fit_and_merge() {
        let mut f = BTreeMap::new();
        f.insert(0, 100);
        assert_eq!(alloc(&mut f, 30), Some(0));
        assert_eq!(alloc(&mut f, 30), Some(30));
        assert_eq!(alloc(&mut f, 50), None);
        release(&mut f, 0, 30);
        assert_eq!(alloc(&mut f, 40), Some(60));
        release(&mut f, 30, 30);
        release(&mut f, 60, 40);
        assert_eq!(f.into_iter().collect::<Vec<_>>(), vec![(0, 100)]);
    }

    #[test]
    #[should_panic(expected = "freed while part of it is free")]
    fn a_range_freed_twice_stops_the_node() {
        let mut f = BTreeMap::new();
        f.insert(0, 100);
        assert_eq!(alloc(&mut f, 30), Some(0));
        release(&mut f, 0, 30);
        release(&mut f, 10, 10);
    }

    #[test]
    fn segments_are_all_or_none() {
        let mut f = BTreeMap::new();
        f.insert(0, 100);
        f.insert(200, 40);
        assert_eq!(alloc_segs(&mut f, 25, 6), None);
        assert_eq!(f.values().sum::<usize>(), 140, "a failed allocation takes nothing");
        assert_eq!(alloc_segs(&mut f, 25, 5), Some(vec![0, 25, 50, 75, 200]));
    }

    fn group(nr: u32, floor: u64, segs: u64, age: Option<u64>, weight: u64) -> Group {
        let now = Instant::now();
        Group {
            nr,
            block_bytes: 1,
            floor,
            cap: 100,
            weight,
            members: (0..nr).map(|r| (r, Member { lease: r as u64, segs })).collect(),
            boot: 0,
            grant: 0,
            grant_at: now,
            give: None,
            stalled: false,
            calm_until: None,
            ages: age.into_iter().collect(),
            demand: 0,
            reports: 0,
            granted: 0,
            gave: 0,
            asked: 0,
        }
    }

    fn state(groups: Vec<(&str, Group)>) -> State {
        State {
            free: BTreeMap::new(),
            leases: BTreeMap::new(),
            names: BTreeMap::new(),
            groups: groups.into_iter().map(|(n, g)| (n.to_string(), g)).collect(),
            next_id: 1,
        }
    }

    fn sh() -> Share {
        Share { margin: Duration::from_millis(60_000), ..Share::default() }
    }

    #[test]
    fn the_coldest_group_gives_by_more_than_the_margin_and_never_below_its_floor() {
        let now = Instant::now();
        // b's coldest is 10 min old, c's 5 min; a asks with its own coldest 1 min old
        let mut st = state(vec![
            ("a", group(2, 1, 1, Some(60_000), 1000)),
            ("b", group(4, 2, 5, Some(600_000), 1000)),
            ("c", group(4, 2, 5, Some(300_000), 1000)),
        ]);
        st.ask("a", 6, now, &sh());
        assert_eq!(st.groups["b"].give.map(|g| g.0), Some(3), "6 segments from 4 ranks: 2 each, from 5 to 3");
        assert!(st.groups["c"].give.is_none());
        // b is already giving: the next ask goes to c
        st.ask("a", 100, now, &sh());
        assert_eq!(st.groups["c"].give.map(|g| g.0), Some(2), "never below the floor of 2");
        // nobody is colder than a by the margin: nobody is asked
        let mut st = state(vec![("a", group(2, 1, 1, Some(60_000), 1000)), ("b", group(2, 1, 5, Some(100_000), 1000))]);
        st.ask("a", 2, now, &sh());
        assert!(st.groups["b"].give.is_none(), "40 s colder is inside the 60 s margin");
        // weight: b's blocks count half as old
        let mut st = state(vec![("a", group(2, 1, 1, Some(0), 1000)), ("b", group(2, 1, 5, Some(100_000), 2000))]);
        st.ask("a", 2, now, &sh());
        assert!(st.groups["b"].give.is_none(), "weight 2: 100 s counts as 50 s");
        st.groups.get_mut("b").unwrap().weight = 1000;
        st.ask("a", 2, now, &sh());
        assert_eq!(st.groups["b"].give.map(|g| g.0), Some(4));
    }

    #[test]
    fn stalled_calm_incomplete_and_empty_groups_are_not_asked() {
        let now = Instant::now();
        let mut st = state(vec![("a", group(1, 1, 1, None, 1000)), ("b", group(2, 1, 5, Some(600_000), 1000))]);
        st.groups.get_mut("b").unwrap().stalled = true;
        st.ask("a", 1, now, &sh());
        assert!(st.groups["b"].give.is_none());
        let b = st.groups.get_mut("b").unwrap();
        b.stalled = false;
        b.calm_until = Some(now + Duration::from_secs(5));
        st.ask("a", 1, now, &sh());
        assert!(st.groups["b"].give.is_none());
        let b = st.groups.get_mut("b").unwrap();
        b.calm_until = None;
        b.members.remove(&1);
        st.ask("a", 1, now, &sh());
        assert!(st.groups["b"].give.is_none(), "a group whose ranks have not all joined");
        let b = st.groups.get_mut("b").unwrap();
        b.members.insert(1, Member { lease: 9, segs: 5 });
        b.ages.clear();
        st.ask("a", 1, now, &sh());
        assert!(st.groups["b"].give.is_none(), "a group that reported no blocks");
    }

    #[test]
    fn an_unclaimed_grant_expires_and_an_unanswered_ask_stalls() {
        let t0 = Instant::now();
        let mut st = state(vec![("a", group(2, 1, 1, None, 1000)), ("b", group(2, 1, 3, Some(1), 1000))]);
        let s = Share { grant_ttl: Duration::from_secs(30), stall: Duration::from_secs(10), ..sh() };
        st.groups.get_mut("a").unwrap().grant = 4;
        st.groups.get_mut("b").unwrap().give = Some((2, t0));
        st.tick(t0 + Duration::from_secs(11), &s);
        assert_eq!(st.groups["a"].grant, 4);
        assert!(st.groups["b"].give.is_none() && st.groups["b"].stalled);
        st.tick(t0 + Duration::from_secs(31), &s);
        assert_eq!(st.groups["a"].grant, 0);
    }
}
