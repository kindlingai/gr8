//! A lease on a KV node as a block store (tier::BlockStore), beside the SSD's SlotFile.
//!
//! Concurrent callers each take a lane. An RDMA lane is one RC QP with its own completion queue, a TCP lane is one
//! connection. A call takes a free lane. Over RDMA the client reads and writes the node's memory itself, and the node's
//! CPU does nothing per request. A buffer inside a region the caller registered (register) goes to the wire as it is;
//! anything else goes through the lane's registered staging buffer, one memcpy.
//!
//! A store opened with a group (Opts::group) is one rank's lease in a group on the node (server.rs): it has a number of
//! segments, each a whole number of blocks, and grows and shrinks by segments. Over RDMA each segment has its own
//! address and rkey.

use super::proto::{self, Header};
use super::verbs::{self, Cq, Device, Mr, Pd, Qp, QpEnd, SendWr, Sge, Wc};
use crate::tier::{AlignedBuf, BlockStore};
use std::io::{self, Read, Write};
use std::net::{IpAddr, SocketAddr, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, RwLock};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Want {
    /// RDMA when this end and the node both have a RoCE v2 GID on the connection's addresses, else TCP.
    Auto,
    Tcp,
    Rdma,
}

impl std::str::FromStr for Want {
    type Err = String;
    fn from_str(s: &str) -> Result<Want, String> {
        match s {
            "auto" => Ok(Want::Auto),
            "tcp" => Ok(Want::Tcp),
            "rdma" => Ok(Want::Rdma),
            _ => Err(format!("transport {s}: auto, tcp or rdma")),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transport {
    Tcp,
    Rdma,
}

#[derive(Clone, Debug)]
pub struct Opts {
    /// The lease's name on the node: unique per client (a box and rank, say).
    pub name: String,
    pub block_bytes: usize,
    pub blocks: u64,
    /// How many callers can move data at once.
    pub lanes: usize,
    pub want: Want,
    /// Ack timeout exponent for RDMA (verbs::Qp::connect).
    pub qp_timeout: u8,
    /// Work requests in flight per RDMA lane.
    pub depth: u32,
    /// Join a group on the node (a cluster's ranks sharing it with other clusters) instead of a fixed lease.
    pub group: Option<GroupOpts>,
}

/// One rank's place in a group (server.rs GOPEN). Every rank of a cluster gives the same group, ranks, sizes and
/// weight.
#[derive(Clone, Debug)]
pub struct GroupOpts {
    pub group: String,
    pub nr: u32,
    pub rank: u32,
    /// The lease never shrinks below this, nor grows past cap (rounded up to whole segments).
    pub floor_bytes: u64,
    pub cap_bytes: u64,
    /// 1000 = 1x; 2000 is asked to give as if its blocks were half as old.
    pub weight: u32,
}

/// The node's answer to a REPORT, in segments per rank.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Report {
    /// held for every rank of the group now: each rank grows by this many
    pub grant: u64,
    /// asked back: each rank shrinks by this many
    pub give: u64,
    /// segments nobody holds or was promised
    pub unclaimed: u64,
}

impl Opts {
    pub fn new(name: &str, block_bytes: usize, blocks: u64) -> Opts {
        Opts {
            name: name.into(),
            block_bytes,
            blocks,
            lanes: 4,
            want: Want::Auto,
            qp_timeout: 14,
            depth: 256,
            group: None,
        }
    }
}

struct Rdma {
    dev: Arc<Device>,
    pd: Arc<Pd>,
    // lease bytes a segment, and each segment's (remote address, rkey). A fixed lease is one segment
    span: u64,
    segs: RwLock<Vec<(u64, u32)>>,
    regions: RwLock<Vec<(usize, usize, Arc<Mr>)>>,
}

impl Rdma {
    // the remote address and rkey of lease bytes off..off + len, which lie in one segment
    fn target(&self, off: u64, len: usize) -> io::Result<(u64, u32)> {
        let (i, at) = ((off / self.span) as usize, off % self.span);
        match self.segs.read().unwrap().get(i) {
            Some(&(a, k)) if len as u64 <= self.span - at => Ok((a + at, k)),
            _ => Err(io::Error::new(io::ErrorKind::InvalidInput, "past the lease, or across a segment")),
        }
    }
}

struct RdmaLane {
    qp: Qp,
    staging: AlignedBuf,
    staging_mr: Mr,
    broken: bool,
}

enum Lane {
    Tcp(TcpStream),
    Rdma(RdmaLane),
}

pub struct RemoteStore {
    ctl: Mutex<TcpStream>,
    lanes: Vec<Mutex<Lane>>,
    next: AtomicUsize,
    rdma: Option<Rdma>,
    pub transport: Transport,
    pub block_bytes: usize,
    blocks: AtomicU64,
    /// Blocks a segment (a group lease), 0 for a fixed lease.
    pub per_seg: u64,
    pub lease_id: u64,
    pub lease_bytes: u64,
    pub local: SocketAddr,
    pub server: SocketAddr,
    /// Why Want::Auto took TCP.
    pub fallback: Option<String>,
    /// The RDMA device and GID index this end uses.
    pub device: Option<(String, u32)>,
}

fn connect(addr: &str) -> io::Result<TcpStream> {
    let a = addr
        .to_socket_addrs()?
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("{addr} resolves to nothing")))?;
    let s = TcpStream::connect_timeout(&a, Duration::from_secs(10))?;
    s.set_nodelay(true)?;
    Ok(s)
}

/// The node's STATS answer (JSON), without a lease.
pub fn stats(addr: &str) -> io::Result<String> {
    let mut s = connect(addr)?;
    Header::new(proto::STATS, 0, 0, 0).write(&mut s)?;
    let h = proto::reply(&mut s)?;
    let mut j = vec![0u8; h.a as usize];
    s.read_exact(&mut j)?;
    Ok(String::from_utf8_lossy(&j).into_owned())
}

impl RemoteStore {
    /// Opens a lease of `o.blocks` blocks on the node at `addr` (host:port).
    pub fn open(addr: &str, o: &Opts) -> io::Result<RemoteStore> {
        let mut ctl = connect(addr)?;
        let (local, server) = (ctl.local_addr()?, ctl.peer_addr()?);
        let (lease_id, lease_bytes, blocks, per_seg) = match &o.group {
            None => {
                Header::new(proto::OPEN, o.block_bytes as u64, o.blocks, o.name.len() as u64).write(&mut ctl)?;
                ctl.write_all(o.name.as_bytes())?;
                let h = proto::reply(&mut ctl)?;
                (h.a, h.b, o.blocks, 0)
            }
            Some(g) => {
                let mut b = Vec::with_capacity(proto::GOPEN_FIXED + g.group.len() + 1 + o.name.len());
                for x in [o.block_bytes as u64, g.floor_bytes, g.cap_bytes] {
                    b.extend_from_slice(&x.to_le_bytes());
                }
                for x in [g.nr, g.rank, g.weight, 0] {
                    b.extend_from_slice(&x.to_le_bytes());
                }
                b.extend_from_slice(format!("{} {}", g.group, o.name).as_bytes());
                Header::new(proto::GOPEN, b.len() as u64, 0, 0).write(&mut ctl)?;
                ctl.write_all(&b)?;
                let h = proto::reply(&mut ctl)?;
                (h.a, h.b * h.c * o.block_bytes as u64, h.b * h.c, h.c)
            }
        };
        let mut token = [0u8; 16];
        ctl.read_exact(&mut token)?;
        let n = o.lanes.max(1);
        let mut fallback = None;
        let rdma = match o.want {
            Want::Tcp => None,
            Want::Rdma => Some(Self::rdma_setup(&mut ctl, o, n, local)?),
            Want::Auto => Self::rdma_setup(&mut ctl, o, n, local).map_err(|e| fallback = Some(e.to_string())).ok(),
        };
        let (rdma, lanes) = match rdma {
            Some((mut r, ls)) => {
                if per_seg > 0 {
                    let (span, segs) = Self::segs_of(&mut ctl)?;
                    r.span = span;
                    r.segs = RwLock::new(segs);
                }
                (Some(r), ls.into_iter().map(|l| Mutex::new(Lane::Rdma(l))).collect())
            }
            None => {
                let mut ls = Vec::new();
                for _ in 0..n {
                    let mut s = connect(&server.to_string())?;
                    Header::new(proto::ATTACH, lease_id, 0, 0).write(&mut s)?;
                    s.write_all(&token)?;
                    proto::reply(&mut s)?;
                    ls.push(Mutex::new(Lane::Tcp(s)));
                }
                (None, ls)
            }
        };
        Ok(RemoteStore {
            ctl: Mutex::new(ctl),
            lanes,
            next: AtomicUsize::new(0),
            transport: if rdma.is_some() { Transport::Rdma } else { Transport::Tcp },
            device: rdma.as_ref().map(|r| (r.dev.name.clone(), r.dev.gid_index)),
            rdma,
            block_bytes: o.block_bytes,
            blocks: AtomicU64::new(blocks),
            per_seg,
            lease_id,
            lease_bytes,
            local,
            server,
            fallback,
        })
    }

    fn rdma_setup(ctl: &mut TcpStream, o: &Opts, lanes: usize, local: SocketAddr) -> io::Result<(Rdma, Vec<RdmaLane>)> {
        let IpAddr::V4(ip) = local.ip() else {
            return Err(io::Error::new(io::ErrorKind::Unsupported, "RDMA needs an IPv4 connection"));
        };
        let dev = Device::open(ip)?;
        let pd = Pd::new(&dev)?;
        let mut ls = Vec::new();
        for _ in 0..lanes {
            let cq = Cq::new(&dev, o.depth + 16)?;
            let qp = Qp::new(&pd, &cq, o.depth, false, verbs::psn())?;
            let staging = AlignedBuf::new(o.block_bytes);
            let staging_mr = Mr::new(
                &pd,
                staging.as_slice().as_ptr() as *mut u8,
                staging.as_slice().len(),
                verbs::ACCESS_LOCAL_WRITE,
            )?;
            ls.push(RdmaLane { qp, staging, staging_mr, broken: false });
        }
        Header::new(proto::RDMA, lanes as u64, dev.mtu as u64, dev.max_init_rd_atom as u64).write(ctl)?;
        let mut b = dev.gid.to_vec();
        for l in &ls {
            let e = l.qp.end();
            b.extend_from_slice(&e.qpn.to_le_bytes());
            b.extend_from_slice(&e.psn.to_le_bytes());
        }
        ctl.write_all(&b)?;
        let h = proto::reply(ctl)?;
        let (mtu, rd) = (h.c as u32, (h.c >> 32) as u32);
        let mut b = vec![0u8; 16 + lanes * 8];
        ctl.read_exact(&mut b)?;
        let gid: [u8; 16] = b[..16].try_into().unwrap();
        for (i, l) in ls.iter().enumerate() {
            let c = &b[16 + i * 8..24 + i * 8];
            let end = QpEnd {
                qpn: u32::from_le_bytes(c[0..4].try_into().unwrap()),
                psn: u32::from_le_bytes(c[4..8].try_into().unwrap()),
            };
            l.qp.connect(end, &gid, mtu.min(dev.mtu), dev.max_init_rd_atom.min(rd).max(1), 0, o.qp_timeout)?;
        }
        let segs = RwLock::new(vec![(h.a, h.b as u32)]);
        Ok((Rdma { dev, pd, span: u64::MAX, segs, regions: RwLock::new(Vec::new()) }, ls))
    }

    // each segment's (remote address, rkey), registered on the node for this connection's device
    fn segs_of(ctl: &mut TcpStream) -> io::Result<(u64, Vec<(u64, u32)>)> {
        Header::new(proto::SEGS, 0, 0, 0).write(ctl)?;
        let h = proto::reply(ctl)?;
        let mut b = vec![0u8; h.a as usize * 16];
        ctl.read_exact(&mut b)?;
        let segs = b
            .chunks(16)
            .map(|c| {
                (
                    u64::from_le_bytes(c[0..8].try_into().unwrap()),
                    u64::from_le_bytes(c[8..16].try_into().unwrap()) as u32,
                )
            })
            .collect();
        Ok((h.b, segs))
    }

    /// Blocks in the store now.
    pub fn blocks(&self) -> u64 {
        self.blocks.load(Ordering::Acquire)
    }

    /// Segments in the store now (a group lease).
    pub fn segs(&self) -> u64 {
        self.blocks() / self.per_seg.max(1)
    }

    fn group_only(&self) -> io::Result<()> {
        if self.per_seg == 0 {
            return Err(io::Error::new(io::ErrorKind::Unsupported, "a fixed lease does not grow or shrink"));
        }
        Ok(())
    }

    /// The group's head: what this cluster wants (segments a rank) and how long ago its coldest blocks were used (ms,
    /// one age a segment's worth, coldest first: coldest_ages). The node answers with a grant or a request to give.
    pub fn report(&self, want: u64, ages_ms: &[u64]) -> io::Result<Report> {
        self.group_only()?;
        let mut ctl = self.ctl.lock().unwrap();
        let n = ages_ms.len().min(super::server::MAX_AGES as usize);
        let mut b = Header::new(proto::REPORT, want, n as u64, 0).bytes().to_vec();
        for a in &ages_ms[..n] {
            b.extend_from_slice(&a.to_le_bytes());
        }
        ctl.write_all(&b)?;
        let h = proto::reply(&mut *ctl)?;
        Ok(Report { grant: h.a, give: h.b, unclaimed: h.c })
    }

    /// Adds `n` segments at the end (each rank, after its head was granted them). Returns the blocks now.
    pub fn grow(&self, n: u64) -> io::Result<u64> {
        self.group_only()?;
        let mut ctl = self.ctl.lock().unwrap();
        Header::new(proto::GROW, n, 0, 0).write(&mut *ctl)?;
        let segs = proto::reply(&mut *ctl)?.a;
        if let Some(r) = &self.rdma {
            *r.segs.write().unwrap() = Self::segs_of(&mut ctl)?.1;
        }
        self.blocks.store(segs * self.per_seg, Ordering::Release);
        Ok(segs * self.per_seg)
    }

    /// Keeps the first `segs` segments and gives the rest back. Nothing may read or write the blocks past them once
    /// this is called (the tier moved what it kept below them first: stash.rs resize_tier). Returns the blocks now.
    pub fn shrink(&self, segs: u64) -> io::Result<u64> {
        self.group_only()?;
        let mut ctl = self.ctl.lock().unwrap();
        if segs >= self.segs() {
            return Ok(self.blocks());
        }
        // the node checks every TCP offset and deregisters the segments' regions, so a stray copy fails rather than
        // reaching memory the node granted to someone else
        Header::new(proto::SHRINK, segs, 0, 0).write(&mut *ctl)?;
        proto::reply(&mut *ctl)?;
        self.blocks.store(segs * self.per_seg, Ordering::Release);
        if let Some(r) = &self.rdma {
            r.segs.write().unwrap().truncate(segs as usize);
        }
        Ok(self.blocks())
    }

    /// Registers `len` bytes at `ptr` with this store's RDMA device, so buffers inside it skip the staging copy. The
    /// memory must outlive the store. Without RDMA it does nothing.
    pub fn register(&self, ptr: *mut u8, len: usize) -> io::Result<()> {
        if let Some(r) = &self.rdma {
            let mr = Arc::new(Mr::new(&r.pd, ptr, len, verbs::ACCESS_LOCAL_WRITE)?);
            r.regions.write().unwrap().push((ptr as usize, len, mr));
        }
        Ok(())
    }

    fn lane(&self) -> MutexGuard<'_, Lane> {
        let n = self.lanes.len();
        let start = self.next.fetch_add(1, Ordering::Relaxed) % n;
        for i in 0..n {
            if let Ok(g) = self.lanes[(start + i) % n].try_lock() {
                return g;
            }
        }
        self.lanes[start].lock().unwrap()
    }

    fn lkey(&self, p: usize, len: usize) -> Option<u32> {
        let r = self.rdma.as_ref()?;
        let regions = r.regions.read().unwrap();
        regions.iter().find(|(a, n, _)| p >= *a && p + len <= a + n).map(|(_, _, m)| m.lkey())
    }

    fn check(&self, block: u64, off: usize, len: usize) -> io::Result<u64> {
        if block >= self.blocks() || off + len > self.block_bytes {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "block out of range or too long"));
        }
        Ok(block * self.block_bytes as u64 + off as u64)
    }

    /// Writes `buf` at byte `off` of the lease.
    pub fn write_at(&self, off: u64, buf: &[u8]) -> io::Result<()> {
        let mut g = self.lane();
        match &mut *g {
            Lane::Tcp(s) => {
                Header::new(proto::WRITE, off, buf.len() as u64, 0).write(s)?;
                s.write_all(buf)?;
                proto::reply(s).map(|_| ())
            }
            Lane::Rdma(l) => {
                let r = self.rdma.as_ref().expect("an RDMA lane has RDMA");
                if let Some(k) = self.lkey(buf.as_ptr() as usize, buf.len()) {
                    let (ra, rk) = r.target(off, buf.len())?;
                    return post_wait(l, verbs::WR_RDMA_WRITE, &[(ra, buf.as_ptr() as u64, buf.len() as u32, k, rk)]);
                }
                let k = l.staging_mr.lkey();
                let mut done = 0;
                while done < buf.len() {
                    let n = (buf.len() - done).min(l.staging.as_slice().len());
                    l.staging.as_mut_slice()[..n].copy_from_slice(&buf[done..done + n]);
                    let p = l.staging.as_slice().as_ptr() as u64;
                    let (ra, rk) = r.target(off + done as u64, n)?;
                    post_wait(l, verbs::WR_RDMA_WRITE, &[(ra, p, n as u32, k, rk)])?;
                    done += n;
                }
                Ok(())
            }
        }
    }

    /// Reads `buf.len()` bytes at byte `off` of the lease.
    pub fn read_at_off(&self, off: u64, buf: &mut [u8]) -> io::Result<()> {
        self.gather(&[(off, buf.len())], buf)
    }

    /// Reads each (lease offset, bytes) range into `out`, one after another. out.len() must be their total. Over TCP one
    /// round trip, over RDMA one chain of reads on one lane.
    pub fn gather(&self, ranges: &[(u64, usize)], out: &mut [u8]) -> io::Result<()> {
        let total: usize = ranges.iter().map(|r| r.1).sum();
        if total != out.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "gather: out is not the ranges' total"));
        }
        let mut g = self.lane();
        match &mut *g {
            Lane::Tcp(s) => {
                let mut b = Header::new(proto::GATHER, ranges.len() as u64, 0, 0).bytes().to_vec();
                for &(o, n) in ranges {
                    b.extend_from_slice(&o.to_le_bytes());
                    b.extend_from_slice(&(n as u64).to_le_bytes());
                }
                s.write_all(&b)?;
                proto::reply(s)?;
                s.read_exact(out)
            }
            Lane::Rdma(l) => {
                let r = self.rdma.as_ref().expect("an RDMA lane has RDMA");
                if let Some(k) = self.lkey(out.as_ptr() as usize, out.len()) {
                    let base = out.as_mut_ptr() as u64;
                    let mut at = 0u64;
                    let ops = ranges
                        .iter()
                        .map(|&(o, n)| {
                            let (ra, rk) = r.target(o, n)?;
                            let op = (ra, base + at, n as u32, k, rk);
                            at += n as u64;
                            Ok(op)
                        })
                        .collect::<io::Result<Vec<_>>>()?;
                    return post_wait(l, verbs::WR_RDMA_READ, &ops);
                }
                // in staging-sized batches, each copied out
                let (cap, k) = (l.staging.as_slice().len(), l.staging_mr.lkey());
                let p = l.staging.as_slice().as_ptr() as u64;
                let (mut i, mut put) = (0, 0);
                let mut parts: Vec<(u64, usize)> = Vec::new();
                for &(o, n) in ranges {
                    let mut d = 0;
                    while d < n {
                        let c = (n - d).min(cap);
                        parts.push((o + d as u64, c));
                        d += c;
                    }
                }
                while i < parts.len() {
                    let (mut ops, mut fill) = (Vec::new(), 0usize);
                    while i < parts.len() && fill + parts[i].1 <= cap {
                        let (ra, rk) = r.target(parts[i].0, parts[i].1)?;
                        ops.push((ra, p + fill as u64, parts[i].1 as u32, k, rk));
                        fill += parts[i].1;
                        i += 1;
                    }
                    post_wait(l, verbs::WR_RDMA_READ, &ops)?;
                    out[put..put + fill].copy_from_slice(&l.staging.as_slice()[..fill]);
                    put += fill;
                }
                Ok(())
            }
        }
    }

    /// Reads each (lease offset, destination, bytes) into its own destination. Over TCP one GATHER round trip, over
    /// RDMA one chain of reads on one lane when every destination is inside a registered region, else one gather per
    /// destination through the staging buffer.
    ///
    /// # Safety
    /// Each destination must be valid for its bytes and no other thread may touch it during the call.
    pub unsafe fn read_scatter(&self, ops: &[(u64, *mut u8, usize)]) -> io::Result<()> {
        if ops.is_empty() {
            return Ok(());
        }
        let keys: Option<Vec<u32>> = ops.iter().map(|&(_, p, n)| self.lkey(p as usize, n)).collect();
        if let (Some(keys), Some(r)) = (keys, self.rdma.as_ref()) {
            let mut g = self.lane();
            if let Lane::Rdma(l) = &mut *g {
                let mut v = Vec::with_capacity(ops.len());
                for (&(o, p, n), &k) in ops.iter().zip(&keys) {
                    let (ra, rk) = r.target(o, n)?;
                    v.push((ra, p as u64, n as u32, k, rk));
                }
                return post_wait(l, verbs::WR_RDMA_READ, &v);
            }
        }
        if self.rdma.is_none() {
            let mut g = self.lane();
            if let Lane::Tcp(s) = &mut *g {
                let mut b = Header::new(proto::GATHER, ops.len() as u64, 0, 0).bytes().to_vec();
                for &(o, _, n) in ops {
                    b.extend_from_slice(&o.to_le_bytes());
                    b.extend_from_slice(&(n as u64).to_le_bytes());
                }
                s.write_all(&b)?;
                proto::reply(s)?;
                for &(_, p, n) in ops {
                    s.read_exact(std::slice::from_raw_parts_mut(p, n))?;
                }
                return Ok(());
            }
        }
        for &(o, p, n) in ops {
            self.read_at_off(o, std::slice::from_raw_parts_mut(p, n))?;
        }
        Ok(())
    }

    /// RDMA only: one read at any remote address and rkey, with no check here. The isolation check uses it to show
    /// that the NIC refuses another lease's region. A refused read breaks its lane.
    pub fn raw_read(&self, raddr: u64, rkey: u32, len: usize) -> io::Result<()> {
        let mut g = self.lane();
        let Lane::Rdma(l) = &mut *g else { return Err(io::Error::new(io::ErrorKind::Unsupported, "not RDMA")) };
        let k = l.staging_mr.lkey();
        let p = l.staging.as_slice().as_ptr() as u64;
        post_wait(l, verbs::WR_RDMA_READ, &[(raddr, p, len.min(l.staging.as_slice().len()) as u32, k, rkey)])
    }

    /// (remote address, rkey) of this lease's (first segment's) region on the node.
    pub fn rdma_target(&self) -> Option<(u64, u32)> {
        self.rdma.as_ref().map(|r| r.segs.read().unwrap()[0])
    }

    /// `buf.len()` bytes at byte `off` of block `block`: one row of a page block.
    pub fn read_at(&self, block: u64, off: usize, buf: &mut [u8]) -> io::Result<()> {
        let o = self.check(block, off, buf.len())?;
        self.read_at_off(o, buf)
    }

    pub fn lanes(&self) -> usize {
        self.lanes.len()
    }
}

/// Posts reads or writes of (remote address, local address, bytes, lkey, rkey) on one lane, chained, at most the QP's
/// depth in flight, and waits for all of them. Each chain signals its last request alone: RC completes a send queue in
/// order.
fn post_wait(l: &mut RdmaLane, opcode: u32, ops: &[(u64, u64, u32, u32, u32)]) -> io::Result<()> {
    if l.broken {
        return Err(io::Error::other("this RDMA lane failed earlier"));
    }
    const CHAIN: usize = 32;
    let depth = l.qp.depth as usize;
    let (mut next, mut done, mut inflight) = (0usize, 0usize, 0usize);
    let mut wrs = [SendWr::default(); CHAIN];
    let mut sges = [Sge::default(); CHAIN];
    let mut wc = [Wc::default(); 16];
    let mut idle = Instant::now();
    while done < ops.len() {
        while next < ops.len() && inflight < depth {
            let n = CHAIN.min(depth - inflight).min(ops.len() - next);
            let (wp, sp) = (wrs.as_mut_ptr(), sges.as_mut_ptr());
            for j in 0..n {
                let (ra, la, len, lkey, rkey) = ops[next + j];
                let last = j == n - 1;
                // The driver follows `next` and `sg_list` through these pointers. Writes go through the same pointers.
                unsafe {
                    *sp.add(j) = Sge { addr: la, length: len, lkey };
                    *wp.add(j) = SendWr {
                        wr_id: if last { n as u64 } else { 0 },
                        next: if last { std::ptr::null_mut() } else { wp.add(j + 1) },
                        sg_list: sp.add(j),
                        num_sge: 1,
                        opcode,
                        send_flags: if last { verbs::SEND_SIGNALED } else { 0 },
                        remote_addr: ra,
                        rkey,
                        ..SendWr::default()
                    };
                }
            }
            if let Err(e) = unsafe { l.qp.post(wp) } {
                l.broken = true;
                return Err(e);
            }
            next += n;
            inflight += n;
        }
        let got = l.qp.cq.poll(&mut wc)?;
        if got == 0 {
            if idle.elapsed() > Duration::from_secs(10) {
                l.broken = true;
                return Err(io::Error::new(io::ErrorKind::TimedOut, "no RDMA completion for 10 s"));
            }
            std::hint::spin_loop();
            continue;
        }
        idle = Instant::now();
        for w in &wc[..got] {
            if w.status != 0 {
                // the QP is in the error state now and flushes the rest: this lane is done
                l.broken = true;
                return Err(io::Error::other(format!(
                    "RDMA {} failed: {} (vendor 0x{:x})",
                    if opcode == verbs::WR_RDMA_READ { "read" } else { "write" },
                    verbs::wc_status(w.status),
                    w.vendor_err
                )));
            }
            inflight -= w.wr_id as usize;
            done += w.wr_id as usize;
        }
    }
    Ok(())
}

impl BlockStore for RemoteStore {
    fn block_bytes(&self) -> usize {
        self.block_bytes
    }
    fn blocks(&self) -> u64 {
        RemoteStore::blocks(self)
    }
    fn write_block(&self, block: u64, buf: &[u8]) -> io::Result<()> {
        let off = self.check(block, 0, buf.len())?;
        self.write_at(off, buf)
    }
    fn read_block(&self, block: u64, buf: &mut [u8]) -> io::Result<()> {
        let off = self.check(block, 0, buf.len())?;
        self.read_at_off(off, buf)
    }
}

/// The ages a head REPORTs: `order` is its tier's entries as (id, blocks) in the order a shrink evicts them (stash.rs
/// give_order), and `age_ms(id)` how long ago an entry was last used. ages[i] is what giving i + 1 segments costs: the
/// age of the most recently used entry among those up to the one holding the (i x per_seg)-th block, so the ages never
/// rise (the head's clock need not agree with the eviction order). At most `most`.
pub fn coldest_ages(order: &[(u64, usize)], age_ms: impl Fn(u64) -> u64, per_seg: u64, most: usize) -> Vec<u64> {
    let mut out = Vec::new();
    let mut seen = 0u64;
    let mut youngest = u64::MAX;
    for &(id, n) in order {
        let end = seen + n as u64;
        youngest = youngest.min(age_ms(id));
        // every segment boundary i x per_seg in [seen, end) falls in this entry
        while out.len() < most && (out.len() as u64) * per_seg.max(1) < end {
            out.push(youngest);
        }
        seen = end;
    }
    out
}
