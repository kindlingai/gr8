//! A lease on a KV node as a block store (tier::BlockStore), beside the SSD's SlotFile.
//!
//! Concurrent callers each take a lane. An RDMA lane is one RC QP with its own completion queue, a TCP lane is one
//! connection. A call takes a free lane. Over RDMA the client reads and writes the node's memory itself, and the node's
//! CPU does nothing per request. A buffer inside a region the caller registered (register) goes to the wire as it is;
//! anything else goes through the lane's registered staging buffer, one memcpy.

use super::proto::{self, Header};
use super::verbs::{self, Cq, Device, Mr, Pd, Qp, QpEnd, SendWr, Sge, Wc};
use crate::tier::{AlignedBuf, BlockStore};
use std::io::{self, Read, Write};
use std::net::{IpAddr, SocketAddr, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicUsize, Ordering};
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
}

impl Opts {
    pub fn new(name: &str, block_bytes: usize, blocks: u64) -> Opts {
        Opts { name: name.into(), block_bytes, blocks, lanes: 4, want: Want::Auto, qp_timeout: 14, depth: 256 }
    }
}

struct Rdma {
    dev: Arc<Device>,
    pd: Arc<Pd>,
    raddr: u64,
    rkey: u32,
    regions: RwLock<Vec<(usize, usize, Arc<Mr>)>>,
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
    _ctl: Mutex<TcpStream>,
    lanes: Vec<Mutex<Lane>>,
    next: AtomicUsize,
    rdma: Option<Rdma>,
    pub transport: Transport,
    pub block_bytes: usize,
    pub blocks: u64,
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
        Header::new(proto::OPEN, o.block_bytes as u64, o.blocks, o.name.len() as u64).write(&mut ctl)?;
        ctl.write_all(o.name.as_bytes())?;
        let h = proto::reply(&mut ctl)?;
        let (lease_id, lease_bytes) = (h.a, h.b);
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
            Some((r, ls)) => (Some(r), ls.into_iter().map(|l| Mutex::new(Lane::Rdma(l))).collect()),
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
            _ctl: Mutex::new(ctl),
            lanes,
            next: AtomicUsize::new(0),
            transport: if rdma.is_some() { Transport::Rdma } else { Transport::Tcp },
            device: rdma.as_ref().map(|r| (r.dev.name.clone(), r.dev.gid_index)),
            rdma,
            block_bytes: o.block_bytes,
            blocks: o.blocks,
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
        Ok((Rdma { dev, pd, raddr: h.a, rkey: h.b as u32, regions: RwLock::new(Vec::new()) }, ls))
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
        if block >= self.blocks || off + len > self.block_bytes {
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
                    return post_wait(l, r, verbs::WR_RDMA_WRITE, &[(off, buf.as_ptr() as u64, buf.len() as u32, k)]);
                }
                let k = l.staging_mr.lkey();
                let mut done = 0;
                while done < buf.len() {
                    let n = (buf.len() - done).min(l.staging.as_slice().len());
                    l.staging.as_mut_slice()[..n].copy_from_slice(&buf[done..done + n]);
                    let p = l.staging.as_slice().as_ptr() as u64;
                    post_wait(l, r, verbs::WR_RDMA_WRITE, &[(off + done as u64, p, n as u32, k)])?;
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
                    let ops: Vec<_> = ranges
                        .iter()
                        .map(|&(o, n)| {
                            let op = (o, base + at, n as u32, k);
                            at += n as u64;
                            op
                        })
                        .collect();
                    return post_wait(l, r, verbs::WR_RDMA_READ, &ops);
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
                        ops.push((parts[i].0, p + fill as u64, parts[i].1 as u32, k));
                        fill += parts[i].1;
                        i += 1;
                    }
                    post_wait(l, r, verbs::WR_RDMA_READ, &ops)?;
                    out[put..put + fill].copy_from_slice(&l.staging.as_slice()[..fill]);
                    put += fill;
                }
                Ok(())
            }
        }
    }

    /// RDMA only: one read at any remote address and rkey, with no check here. The isolation check uses it to show
    /// that the NIC refuses another lease's region. A refused read breaks its lane.
    pub fn raw_read(&self, raddr: u64, rkey: u32, len: usize) -> io::Result<()> {
        let mut g = self.lane();
        let Lane::Rdma(l) = &mut *g else { return Err(io::Error::new(io::ErrorKind::Unsupported, "not RDMA")) };
        let r = self.rdma.as_ref().expect("RDMA");
        let k = l.staging_mr.lkey();
        let p = l.staging.as_slice().as_ptr() as u64;
        let fake = Rdma { dev: r.dev.clone(), pd: r.pd.clone(), raddr: 0, rkey, regions: RwLock::new(Vec::new()) };
        post_wait(l, &fake, verbs::WR_RDMA_READ, &[(raddr, p, len.min(l.staging.as_slice().len()) as u32, k)])
    }

    /// (remote address, rkey) of this lease's region on the node.
    pub fn rdma_target(&self) -> Option<(u64, u32)> {
        self.rdma.as_ref().map(|r| (r.raddr, r.rkey))
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

/// Posts reads or writes of (remote offset, local address, bytes, lkey) on one lane, chained, at most the QP's depth
/// in flight, and waits for all of them. Each chain signals its last request alone: RC completes a send queue in order.
fn post_wait(l: &mut RdmaLane, r: &Rdma, opcode: u32, ops: &[(u64, u64, u32, u32)]) -> io::Result<()> {
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
                let (ro, la, len, lkey) = ops[next + j];
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
                        remote_addr: r.raddr + ro,
                        rkey: r.rkey,
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
        self.blocks
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
