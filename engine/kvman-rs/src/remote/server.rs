//! The KV node: locked memory cut into leases, served over TCP and, where the connection arrived on an address with a
//! RoCE v2 GID, to one-sided RDMA from the client.
//!
//! A lease is a contiguous range of the arena, named by its client. Only one live lease holds a name, so two ranks (or a
//! test and a live engine) that name themselves apart can never write into each other's blocks: the TCP path checks
//! every offset against the lease, and for RDMA each lease has its own protection domain holding one memory region over
//! exactly its range, with its clients' QPs. The NIC refuses a QP of one lease that presents another lease's rkey, or an
//! address outside its region. A lease is zeroed when it is granted and freed when its last connection closes.

use super::proto::{self, Header};
use super::sys::{self, Arena};
use super::verbs::{self, Cq, Device, Mr, Pd, Qp, QpEnd};
use std::collections::{BTreeMap, HashMap};
use std::io::{self, Read, Write};
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Instant;

/// Leases start and end on 2 MiB boundaries of the arena.
pub const LEASE_ALIGN: usize = 2 << 20;
const GATHER_CHUNK: usize = 1 << 20;

pub struct Config {
    pub bytes: usize,
    /// mlock the arena (always, outside tests).
    pub lock: bool,
    /// Offer RDMA where the connection's local address has a RoCE v2 GID.
    pub rdma: bool,
    /// The ack timeout exponent for the node's QPs (verbs::Qp::connect).
    pub qp_timeout: u8,
}

struct State {
    free: BTreeMap<usize, usize>,
    leases: BTreeMap<u64, Weak<Lease>>,
    names: BTreeMap<String, u64>,
    next_id: u64,
}

pub struct Shared {
    arena: Arena,
    state: Mutex<State>,
    devices: Mutex<HashMap<Ipv4Addr, Arc<Device>>>,
    cfg_rdma: bool,
    qp_timeout: u8,
    started: Instant,
    opened: AtomicU64,
}

/// A lease's protection domain and memory region on one device.
type Region = (Arc<Pd>, Arc<Mr>);

pub struct Lease {
    pub id: u64,
    pub name: String,
    off: usize,
    pub len: usize,
    pub block_bytes: usize,
    token: [u8; 16],
    regions: Mutex<HashMap<String, Region>>,
    conns: AtomicU64,
    read: AtomicU64,
    written: AtomicU64,
    since: Instant,
    shared: Arc<Shared>,
}

impl Lease {
    fn ptr(&self, off: u64) -> *mut u8 {
        unsafe { self.shared.arena.base().add(self.off + off as usize) }
    }
    fn check(&self, off: u64, len: u64) -> bool {
        off.checked_add(len).is_some_and(|end| end <= self.len as u64)
    }
    /// This lease's region on `dev`, registered on first use.
    fn region(&self, dev: &Arc<Device>) -> io::Result<Region> {
        let mut r = self.regions.lock().unwrap();
        if let Some(x) = r.get(&dev.name) {
            return Ok(x.clone());
        }
        let pd = Pd::new(dev)?;
        let mr = Arc::new(Mr::new(
            &pd,
            self.ptr(0),
            self.len,
            verbs::ACCESS_LOCAL_WRITE | verbs::ACCESS_REMOTE_READ | verbs::ACCESS_REMOTE_WRITE,
        )?);
        r.insert(dev.name.clone(), (pd.clone(), mr.clone()));
        Ok((pd, mr))
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        // no QP of this lease is left (each connection holds the lease until its QPs are gone); deregister before the
        // range can be granted again
        self.regions.lock().unwrap().clear();
        let mut st = self.shared.state.lock().unwrap();
        st.leases.remove(&self.id);
        if st.names.get(&self.name) == Some(&self.id) {
            st.names.remove(&self.name);
        }
        release(&mut st.free, self.off, self.len);
    }
}

fn release(free: &mut BTreeMap<usize, usize>, mut off: usize, mut len: usize) {
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
        let listener = TcpListener::bind(listen)?;
        let arena = Arena::new(bytes, cfg.lock)?;
        let mut free = BTreeMap::new();
        free.insert(0, bytes);
        let shared = Arc::new(Shared {
            arena,
            state: Mutex::new(State { free, leases: BTreeMap::new(), names: BTreeMap::new(), next_id: 1 }),
            devices: Mutex::new(HashMap::new()),
            cfg_rdma: cfg.rdma,
            qp_timeout: cfg.qp_timeout,
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
    fn device(&self, ip: Ipv4Addr) -> io::Result<Arc<Device>> {
        let mut d = self.devices.lock().unwrap();
        if let Some(x) = d.get(&ip) {
            return Ok(x.clone());
        }
        let dev = Device::open(ip)?;
        d.insert(ip, dev.clone());
        Ok(dev)
    }

    /// The node's state as JSON: its size, what is free, and each lease.
    pub fn stats_json(&self) -> String {
        // a lease upgraded here may be the last reference, and its drop takes the state lock: collect, then unlock
        let (free, largest, leases) = {
            let st = self.state.lock().unwrap();
            let leases: Vec<Arc<Lease>> = st.leases.values().filter_map(Weak::upgrade).collect();
            (st.free.values().sum::<usize>(), st.free.values().copied().max().unwrap_or(0), leases)
        };
        let mut s = format!(
            "{{\"bytes\":{},\"free\":{},\"largest_free\":{},\"locked\":{},\"uptime_s\":{},\"opened\":{},\"leases\":[",
            self.arena.len(),
            free,
            largest,
            self.arena.locked,
            self.started.elapsed().as_secs(),
            self.opened.load(Ordering::Relaxed)
        );
        let mut first = true;
        for l in &leases {
            if !first {
                s.push(',');
            }
            first = false;
            let devs: Vec<String> =
                l.regions.lock().unwrap().keys().map(|k| format!("\"{}\"", json_escape(k))).collect();
            s += &format!(
                "{{\"id\":{},\"name\":\"{}\",\"bytes\":{},\"block\":{},\"conns\":{},\"read\":{},\"written\":{},\
                 \"rdma\":[{}],\"age_s\":{}}}",
                l.id,
                json_escape(&l.name),
                l.len,
                l.block_bytes,
                l.conns.load(Ordering::Relaxed),
                l.read.load(Ordering::Relaxed),
                l.written.load(Ordering::Relaxed),
                devs.join(","),
                l.since.elapsed().as_secs()
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
            proto::OPEN | proto::ATTACH if lease.is_some() => {
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
                        Header::new(proto::OK, l.id, l.len as u64, 0).write(&mut s)?;
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
                        Header::new(proto::OK, l.id, l.len as u64, 0).write(&mut s)?;
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
                    Ok((dev, mr, mine)) => {
                        let c = dev.mtu as u64 | (dev.max_rd_atom as u64) << 32;
                        Header::new(proto::OK, mr.addr(), mr.rkey() as u64, c).write(&mut s)?;
                        let mut b = dev.gid.to_vec();
                        for q in &mine {
                            let e = q.end();
                            b.extend_from_slice(&e.qpn.to_le_bytes());
                            b.extend_from_slice(&e.psn.to_le_bytes());
                        }
                        s.write_all(&b)?;
                        qps = mine;
                    }
                    Err(e) => fail(&mut s, proto::E_RDMA, &e.to_string())?,
                }
            }
            proto::READ => {
                let l = &lease.as_ref().unwrap().0;
                if !l.check(h.a, h.b) {
                    fail(&mut s, proto::E_RANGE, &format!("read of {} at {} past the lease ({})", h.b, h.a, l.len))?;
                    continue;
                }
                Header::new(proto::OK, h.b, 0, 0).write(&mut s)?;
                unsafe { sys::send_all(fd, l.ptr(h.a), h.b as usize)? };
                l.read.fetch_add(h.b, Ordering::Relaxed);
            }
            proto::WRITE => {
                let l = &lease.as_ref().unwrap().0;
                if !l.check(h.a, h.b) {
                    // the payload is on its way: the stream is out of step, so this connection ends
                    fail(&mut s, proto::E_RANGE, &format!("write of {} at {} past the lease ({})", h.b, h.a, l.len))?;
                    break;
                }
                unsafe { sys::recv_exact(fd, l.ptr(h.a), h.b as usize)? };
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
                let ranges: Vec<(u64, u64)> = r
                    .chunks(16)
                    .map(|c| {
                        (
                            u64::from_le_bytes(c[0..8].try_into().unwrap()),
                            u64::from_le_bytes(c[8..16].try_into().unwrap()),
                        )
                    })
                    .collect();
                if let Some(&(o, n)) = ranges.iter().find(|&&(o, n)| !l.check(o, n)) {
                    fail(&mut s, proto::E_RANGE, &format!("gather of {n} at {o} past the lease ({})", l.len))?;
                    continue;
                }
                let total: u64 = ranges.iter().map(|r| r.1).sum();
                Header::new(proto::OK, total, 0, 0).write(&mut s)?;
                // rows are small: one send per row costs more than copying them together
                let mut buf = Vec::with_capacity(GATHER_CHUNK);
                for &(o, n) in &ranges {
                    if buf.len() + n as usize > GATHER_CHUNK && !buf.is_empty() {
                        s.write_all(&buf)?;
                        buf.clear();
                    }
                    if n as usize >= GATHER_CHUNK {
                        unsafe { sys::send_all(fd, l.ptr(o), n as usize)? };
                    } else {
                        buf.extend_from_slice(unsafe { std::slice::from_raw_parts(l.ptr(o), n as usize) });
                    }
                }
                s.write_all(&buf)?;
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

fn open(sh: &Arc<Shared>, name: &str, block_bytes: u64, blocks: u64) -> Result<Arc<Lease>, (u32, String)> {
    let len = block_bytes
        .checked_mul(blocks)
        .filter(|&n| block_bytes > 0 && n > 0 && n <= sh.arena.len() as u64)
        .ok_or((proto::E_NOROOM, format!("{blocks} blocks of {block_bytes} bytes do not fit the node")))?
        as usize;
    let len_al = len.div_ceil(LEASE_ALIGN) * LEASE_ALIGN;
    let l = {
        let mut st = sh.state.lock().unwrap();
        if let Some(id) = st.names.get(name) {
            return Err((proto::E_BUSY, format!("lease {name} is held (id {id}); a name has one holder at a time")));
        }
        let Some(off) = alloc(&mut st.free, len_al) else {
            let free: usize = st.free.values().sum();
            let largest = st.free.values().copied().max().unwrap_or(0);
            return Err((
                proto::E_NOROOM,
                format!("no room for {len} bytes: {free} free, the largest range {largest}"),
            ));
        };
        let id = st.next_id;
        st.next_id += 1;
        let token = sys::random16().map_err(|e| (proto::E_PROTO, e.to_string()))?;
        let l = Arc::new(Lease {
            id,
            name: name.to_string(),
            off,
            len: len_al,
            block_bytes: block_bytes as usize,
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
        l
    };
    // what the last holder of these bytes left is not this client's
    unsafe { std::ptr::write_bytes(l.ptr(0), 0, l.len) };
    sh.opened.fetch_add(1, Ordering::Relaxed);
    Ok(l)
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
    let (pd, mr) = l.region(&dev)?;
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
}
