//! libibverbs, loaded with dlopen at first use. The crate builds without rdma-core, and a box without it serves and
//! reaches stores over TCP. The structs are rdma-core's (infiniband/verbs.h), and rdma-core keeps their layout fixed.
//! The tests pin every offset used here to the output of a C program built against rdma-core 50. ibv_post_send and
//! ibv_poll_cq are inline functions in verbs.h that call through the context's ops table, so they do the same here.

use super::sys;
use std::ffi::{c_char, c_int, c_void, CStr};
use std::io;
use std::net::Ipv4Addr;
use std::sync::{Arc, OnceLock};

pub const PORT: u8 = 1;
pub const ACCESS_LOCAL_WRITE: c_int = 1;
pub const ACCESS_REMOTE_WRITE: c_int = 2;
pub const ACCESS_REMOTE_READ: c_int = 4;
pub const WR_RDMA_WRITE: u32 = 0;
pub const WR_RDMA_READ: u32 = 4;
pub const SEND_SIGNALED: u32 = 2;
const QPT_RC: u32 = 2;
const QPS_INIT: u32 = 1;
const QPS_RTR: u32 = 2;
const QPS_RTS: u32 = 3;
const QP_STATE: c_int = 1;
const QP_ACCESS_FLAGS: c_int = 8;
const QP_PKEY_INDEX: c_int = 16;
const QP_PORT: c_int = 32;
const QP_AV: c_int = 128;
const QP_PATH_MTU: c_int = 256;
const QP_TIMEOUT: c_int = 512;
const QP_RETRY_CNT: c_int = 1024;
const QP_RNR_RETRY: c_int = 2048;
const QP_RQ_PSN: c_int = 4096;
const QP_MAX_QP_RD_ATOMIC: c_int = 8192;
const QP_MIN_RNR_TIMER: c_int = 32768;
const QP_SQ_PSN: c_int = 65536;
const QP_MAX_DEST_RD_ATOMIC: c_int = 131072;
const QP_DEST_QPN: c_int = 1048576;
// struct ibv_context: ops.poll_cq and ops.post_send
const CTX_POLL_CQ: usize = 96;
const CTX_POST_SEND: usize = 208;
// struct ibv_device_attr and struct ibv_port_attr fields read from raw buffers
const DEV_ATTR_BYTES: usize = 232;
const DEV_MAX_QP_WR: usize = 112;
const DEV_MAX_CQE: usize = 132;
const DEV_MAX_QP_RD_ATOM: usize = 144;
const DEV_MAX_QP_INIT_RD_ATOM: usize = 156;
const PORT_ATTR_BYTES: usize = 56;
const PORT_STATE: usize = 0;
const PORT_ACTIVE_MTU: usize = 8;
const PORT_ACTIVE: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Sge {
    pub addr: u64,
    pub length: u32,
    pub lkey: u32,
}

/// struct ibv_send_wr with the rdma member of its wr union. The other fields are padding.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SendWr {
    pub wr_id: u64,
    pub next: *mut SendWr,
    pub sg_list: *mut Sge,
    pub num_sge: i32,
    pub opcode: u32,
    pub send_flags: u32,
    pub imm: u32,
    pub remote_addr: u64,
    pub rkey: u32,
    pub _pad: u32,
    pub _wr_rest: [u64; 2],
    pub _qp_type: u32,
    pub _pad2: u32,
    pub _bind: [u64; 6],
}

impl Default for SendWr {
    fn default() -> SendWr {
        // all-zero is a valid ibv_send_wr: null pointers, no flags
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Wc {
    pub wr_id: u64,
    pub status: u32,
    pub opcode: u32,
    pub vendor_err: u32,
    pub byte_len: u32,
    pub imm: u32,
    pub qp_num: u32,
    pub src_qp: u32,
    pub wc_flags: u32,
    pub pkey_index: u16,
    pub slid: u16,
    pub sl: u8,
    pub dlid_path_bits: u8,
}

#[repr(C)]
struct QpCap {
    max_send_wr: u32,
    max_recv_wr: u32,
    max_send_sge: u32,
    max_recv_sge: u32,
    max_inline_data: u32,
}

#[repr(C)]
struct QpInitAttr {
    qp_context: *mut c_void,
    send_cq: *mut c_void,
    recv_cq: *mut c_void,
    srq: *mut c_void,
    cap: QpCap,
    qp_type: u32,
    sq_sig_all: i32,
}

#[repr(C, align(8))]
#[derive(Default)]
struct GlobalRoute {
    dgid: [u8; 16],
    flow_label: u32,
    sgid_index: u8,
    hop_limit: u8,
    traffic_class: u8,
}

#[repr(C)]
#[derive(Default)]
struct AhAttr {
    grh: GlobalRoute,
    dlid: u16,
    sl: u8,
    src_path_bits: u8,
    static_rate: u8,
    is_global: u8,
    port_num: u8,
}

#[repr(C)]
#[derive(Default)]
struct QpAttr {
    qp_state: u32,
    cur_qp_state: u32,
    path_mtu: u32,
    path_mig_state: u32,
    qkey: u32,
    rq_psn: u32,
    sq_psn: u32,
    dest_qp_num: u32,
    qp_access_flags: u32,
    cap: [u32; 5],
    ah_attr: AhAttr,
    alt_ah_attr: AhAttr,
    pkey_index: u16,
    alt_pkey_index: u16,
    en_sqd_async_notify: u8,
    sq_draining: u8,
    max_rd_atomic: u8,
    max_dest_rd_atomic: u8,
    min_rnr_timer: u8,
    port_num: u8,
    timeout: u8,
    retry_cnt: u8,
    rnr_retry: u8,
    alt_port_num: u8,
    alt_timeout: u8,
    rate_limit: u32,
}

#[repr(C)]
struct IbvMr {
    context: *mut c_void,
    pd: *mut c_void,
    addr: *mut c_void,
    length: usize,
    handle: u32,
    lkey: u32,
    rkey: u32,
}

#[repr(C)]
struct IbvQp {
    context: *mut c_void,
    qp_context: *mut c_void,
    pd: *mut c_void,
    send_cq: *mut c_void,
    recv_cq: *mut c_void,
    srq: *mut c_void,
    handle: u32,
    qp_num: u32,
}

type PostSend = unsafe extern "C" fn(*mut c_void, *mut SendWr, *mut *mut SendWr) -> c_int;
type PollCq = unsafe extern "C" fn(*mut c_void, c_int, *mut Wc) -> c_int;

struct Lib {
    get_device_list: unsafe extern "C" fn(*mut c_int) -> *mut *mut c_void,
    free_device_list: unsafe extern "C" fn(*mut *mut c_void),
    get_device_name: unsafe extern "C" fn(*mut c_void) -> *const c_char,
    open_device: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    close_device: unsafe extern "C" fn(*mut c_void) -> c_int,
    query_device: unsafe extern "C" fn(*mut c_void, *mut u8) -> c_int,
    query_port: unsafe extern "C" fn(*mut c_void, u8, *mut u8) -> c_int,
    query_gid: unsafe extern "C" fn(*mut c_void, u8, c_int, *mut [u8; 16]) -> c_int,
    alloc_pd: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    dealloc_pd: unsafe extern "C" fn(*mut c_void) -> c_int,
    reg_mr: unsafe extern "C" fn(*mut c_void, *mut c_void, usize, c_int) -> *mut IbvMr,
    dereg_mr: unsafe extern "C" fn(*mut IbvMr) -> c_int,
    create_cq: unsafe extern "C" fn(*mut c_void, c_int, *mut c_void, *mut c_void, c_int) -> *mut c_void,
    destroy_cq: unsafe extern "C" fn(*mut c_void) -> c_int,
    create_qp: unsafe extern "C" fn(*mut c_void, *mut QpInitAttr) -> *mut IbvQp,
    modify_qp: unsafe extern "C" fn(*mut IbvQp, *mut QpAttr, c_int) -> c_int,
    destroy_qp: unsafe extern "C" fn(*mut IbvQp) -> c_int,
    wc_status_str: unsafe extern "C" fn(c_int) -> *const c_char,
}

unsafe impl Send for Lib {}
unsafe impl Sync for Lib {}

#[allow(clippy::missing_transmute_annotations)] // each target type is the Lib field the symbol fills
fn load() -> Result<Lib, String> {
    unsafe {
        let h = sys::dlopen(c"libibverbs.so.1".as_ptr(), sys::RTLD_NOW);
        if h.is_null() {
            let e = sys::dlerror();
            return Err(if e.is_null() {
                "libibverbs.so.1 not found".into()
            } else {
                CStr::from_ptr(e).to_string_lossy().into_owned()
            });
        }
        macro_rules! sym {
            ($name:literal) => {{
                let p = sys::dlsym(h, concat!($name, "\0").as_ptr() as *const c_char);
                if p.is_null() {
                    return Err(format!("libibverbs.so.1 has no {}", $name));
                }
                std::mem::transmute::<*mut c_void, _>(p)
            }};
        }
        Ok(Lib {
            get_device_list: sym!("ibv_get_device_list"),
            free_device_list: sym!("ibv_free_device_list"),
            get_device_name: sym!("ibv_get_device_name"),
            open_device: sym!("ibv_open_device"),
            close_device: sym!("ibv_close_device"),
            query_device: sym!("ibv_query_device"),
            query_port: sym!("ibv_query_port"),
            query_gid: sym!("ibv_query_gid"),
            alloc_pd: sym!("ibv_alloc_pd"),
            dealloc_pd: sym!("ibv_dealloc_pd"),
            reg_mr: sym!("ibv_reg_mr"),
            dereg_mr: sym!("ibv_dereg_mr"),
            create_cq: sym!("ibv_create_cq"),
            destroy_cq: sym!("ibv_destroy_cq"),
            create_qp: sym!("ibv_create_qp"),
            modify_qp: sym!("ibv_modify_qp"),
            destroy_qp: sym!("ibv_destroy_qp"),
            wc_status_str: sym!("ibv_wc_status_str"),
        })
    }
}

fn lib() -> io::Result<&'static Lib> {
    static LIB: OnceLock<Result<Lib, String>> = OnceLock::new();
    LIB.get_or_init(load).as_ref().map_err(|e| io::Error::new(io::ErrorKind::Unsupported, e.clone()))
}

/// Whether libibverbs loads on this box.
pub fn available() -> bool {
    lib().is_ok()
}

pub fn wc_status(status: u32) -> String {
    match lib() {
        Ok(l) => unsafe { CStr::from_ptr((l.wc_status_str)(status as c_int)) }.to_string_lossy().into_owned(),
        Err(_) => format!("status {status}"),
    }
}

/// The IPv4 address a GID maps (::ffff:a.b.c.d), if any.
pub fn gid_ipv4(gid: &str) -> Option<Ipv4Addr> {
    let p: Vec<&str> = gid.trim().split(':').collect();
    if p.len() != 8 || p[..5].iter().any(|x| *x != "0000") || p[5] != "ffff" {
        return None;
    }
    let hi = u16::from_str_radix(p[6], 16).ok()?;
    let lo = u16::from_str_radix(p[7], 16).ok()?;
    Some(Ipv4Addr::new((hi >> 8) as u8, hi as u8, (lo >> 8) as u8, lo as u8))
}

/// Every RoCE v2 GID that maps an IPv4 address on an ACTIVE port: (address, device, GID index), from sysfs, the way
/// engine/comms/fabric.cpp and tools/launch/fabric.py find them. The index differs per box, so it is never assumed.
pub fn roce_v2_addrs(root: &str) -> Vec<(Ipv4Addr, String, u32)> {
    let mut out = Vec::new();
    let Ok(devs) = std::fs::read_dir(root) else { return out };
    let mut devs: Vec<String> = devs.filter_map(|d| d.ok()?.file_name().into_string().ok()).collect();
    devs.sort();
    for dev in devs {
        let port = format!("{root}/{dev}/ports/{PORT}");
        let state = std::fs::read_to_string(format!("{port}/state")).unwrap_or_default();
        if !state.contains("ACTIVE") {
            continue;
        }
        for i in 0..256u32 {
            let Ok(gid) = std::fs::read_to_string(format!("{port}/gids/{i}")) else { break };
            let ty = std::fs::read_to_string(format!("{port}/gid_attrs/types/{i}")).unwrap_or_default();
            if ty.trim() != "RoCE v2" {
                continue;
            }
            if let Some(ip) = gid_ipv4(&gid) {
                out.push((ip, dev.clone(), i));
            }
        }
    }
    out
}

pub const SYSFS: &str = "/sys/class/infiniband";

/// One opened device, with the RoCE v2 GID of the address it was opened for.
pub struct Device {
    ctx: *mut c_void,
    pub name: String,
    pub ip: Ipv4Addr,
    pub gid_index: u32,
    pub gid: [u8; 16],
    pub mtu: u32,
    pub max_qp_wr: u32,
    pub max_cqe: u32,
    pub max_rd_atom: u32,
    pub max_init_rd_atom: u32,
}

unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Device {
    /// Opens the device whose RoCE v2 GID holds `ip`.
    pub fn open(ip: Ipv4Addr) -> io::Result<Arc<Device>> {
        let l = lib()?;
        let Some((_, name, gid_index)) = roce_v2_addrs(SYSFS).into_iter().find(|(a, _, _)| *a == ip) else {
            return Err(io::Error::new(io::ErrorKind::NotFound, format!("no active RoCE v2 GID holds {ip}")));
        };
        unsafe {
            let mut n: c_int = 0;
            let list = (l.get_device_list)(&mut n);
            if list.is_null() {
                return Err(io::Error::other("ibv_get_device_list failed"));
            }
            let mut ctx = std::ptr::null_mut();
            for i in 0..n as usize {
                let d = *list.add(i);
                if CStr::from_ptr((l.get_device_name)(d)).to_bytes() == name.as_bytes() {
                    ctx = (l.open_device)(d);
                }
            }
            (l.free_device_list)(list);
            if ctx.is_null() {
                return Err(io::Error::other(format!("ibv_open_device {name} failed")));
            }
            let mut dev = Device {
                ctx,
                name,
                ip,
                gid_index,
                gid: [0; 16],
                mtu: 0,
                max_qp_wr: 0,
                max_cqe: 0,
                max_rd_atom: 0,
                max_init_rd_atom: 0,
            };
            let mut da = [0u8; DEV_ATTR_BYTES + 256];
            let mut pa = [0u8; PORT_ATTR_BYTES + 256];
            if (l.query_device)(ctx, da.as_mut_ptr()) != 0
                || (l.query_port)(ctx, PORT, pa.as_mut_ptr()) != 0
                || (l.query_gid)(ctx, PORT, gid_index as c_int, &mut dev.gid) != 0
            {
                return Err(io::Error::other(format!("querying {} failed", dev.name)));
            }
            let u = |b: &[u8], o: usize| u32::from_ne_bytes(b[o..o + 4].try_into().unwrap());
            if u(&pa, PORT_STATE) != PORT_ACTIVE {
                return Err(io::Error::other(format!("{} port {PORT} is not active", dev.name)));
            }
            dev.mtu = u(&pa, PORT_ACTIVE_MTU);
            dev.max_qp_wr = u(&da, DEV_MAX_QP_WR);
            dev.max_cqe = u(&da, DEV_MAX_CQE);
            dev.max_rd_atom = u(&da, DEV_MAX_QP_RD_ATOM);
            dev.max_init_rd_atom = u(&da, DEV_MAX_QP_INIT_RD_ATOM);
            Ok(Arc::new(dev))
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        if let Ok(l) = lib() {
            unsafe { (l.close_device)(self.ctx) };
        }
    }
}

/// A protection domain. Each lease on the node has its own, with the lease's one memory region and its clients' QPs.
/// The NIC refuses these QPs any rkey from another lease.
pub struct Pd {
    pd: *mut c_void,
    pub dev: Arc<Device>,
}

unsafe impl Send for Pd {}
unsafe impl Sync for Pd {}

impl Pd {
    pub fn new(dev: &Arc<Device>) -> io::Result<Arc<Pd>> {
        let pd = unsafe { (lib()?.alloc_pd)(dev.ctx) };
        if pd.is_null() {
            return Err(io::Error::other(format!("ibv_alloc_pd on {} failed", dev.name)));
        }
        Ok(Arc::new(Pd { pd, dev: dev.clone() }))
    }
}

impl Drop for Pd {
    fn drop(&mut self) {
        unsafe { (lib().expect("loaded").dealloc_pd)(self.pd) };
    }
}

pub struct Mr {
    mr: *mut IbvMr,
    _pd: Arc<Pd>,
}

unsafe impl Send for Mr {}
unsafe impl Sync for Mr {}

impl Mr {
    /// Registers `len` bytes at `addr`. The memory must outlive the region.
    pub fn new(pd: &Arc<Pd>, addr: *mut u8, len: usize, access: c_int) -> io::Result<Mr> {
        let mr = unsafe { (lib()?.reg_mr)(pd.pd, addr as *mut c_void, len, access) };
        if mr.is_null() {
            return Err(io::Error::other(format!(
                "ibv_reg_mr of {len} bytes on {} failed: {}",
                pd.dev.name,
                io::Error::last_os_error()
            )));
        }
        Ok(Mr { mr, _pd: pd.clone() })
    }
    pub fn lkey(&self) -> u32 {
        unsafe { (*self.mr).lkey }
    }
    pub fn rkey(&self) -> u32 {
        unsafe { (*self.mr).rkey }
    }
    pub fn addr(&self) -> u64 {
        unsafe { (*self.mr).addr as u64 }
    }
}

impl Drop for Mr {
    fn drop(&mut self) {
        unsafe { (lib().expect("loaded").dereg_mr)(self.mr) };
    }
}

pub struct Cq {
    cq: *mut c_void,
    poll: PollCq,
    dev: Arc<Device>,
}

unsafe impl Send for Cq {}
unsafe impl Sync for Cq {}

impl Cq {
    pub fn new(dev: &Arc<Device>, entries: u32) -> io::Result<Arc<Cq>> {
        let n = entries.min(dev.max_cqe).max(1) as c_int;
        let cq = unsafe { (lib()?.create_cq)(dev.ctx, n, std::ptr::null_mut(), std::ptr::null_mut(), 0) };
        if cq.is_null() {
            return Err(io::Error::other(format!("ibv_create_cq on {} failed", dev.name)));
        }
        let poll = unsafe { *((dev.ctx as *const u8).add(CTX_POLL_CQ) as *const PollCq) };
        Ok(Arc::new(Cq { cq, poll, dev: dev.clone() }))
    }
    /// Completions now, at most wc.len().
    pub fn poll(&self, wc: &mut [Wc]) -> io::Result<usize> {
        let n = unsafe { (self.poll)(self.cq, wc.len() as c_int, wc.as_mut_ptr()) };
        if n < 0 {
            return Err(io::Error::other(format!("ibv_poll_cq on {} failed", self.dev.name)));
        }
        Ok(n as usize)
    }
}

impl Drop for Cq {
    fn drop(&mut self) {
        unsafe { (lib().expect("loaded").destroy_cq)(self.cq) };
    }
}

/// What one end of an RC connection tells the other.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QpEnd {
    pub qpn: u32,
    pub psn: u32,
}

/// One RC queue pair, with only a send queue in use: the node's QPs answer one-sided reads and writes and post
/// nothing.
pub struct Qp {
    qp: *mut IbvQp,
    post: PostSend,
    pub pd: Arc<Pd>,
    pub cq: Arc<Cq>,
    pub depth: u32,
    pub psn: u32,
}

unsafe impl Send for Qp {}
unsafe impl Sync for Qp {}

impl Qp {
    /// A QP in INIT. `remote_access`: let the peer read and write this PD's regions (the node's side).
    pub fn new(pd: &Arc<Pd>, cq: &Arc<Cq>, depth: u32, remote_access: bool, psn: u32) -> io::Result<Qp> {
        let l = lib()?;
        let depth = depth.min(pd.dev.max_qp_wr).max(1);
        let mut ia = QpInitAttr {
            qp_context: std::ptr::null_mut(),
            send_cq: cq.cq,
            recv_cq: cq.cq,
            srq: std::ptr::null_mut(),
            cap: QpCap { max_send_wr: depth, max_recv_wr: 1, max_send_sge: 1, max_recv_sge: 1, max_inline_data: 0 },
            qp_type: QPT_RC,
            sq_sig_all: 0,
        };
        let qp = unsafe { (l.create_qp)(pd.pd, &mut ia) };
        if qp.is_null() {
            return Err(io::Error::other(format!(
                "ibv_create_qp on {} failed: {}",
                pd.dev.name,
                io::Error::last_os_error()
            )));
        }
        let post = unsafe { *((pd.dev.ctx as *const u8).add(CTX_POST_SEND) as *const PostSend) };
        let q = Qp { qp, post, pd: pd.clone(), cq: cq.clone(), depth, psn: psn & 0xff_ffff };
        let mut a = QpAttr {
            qp_state: QPS_INIT,
            port_num: PORT,
            qp_access_flags: if remote_access {
                (ACCESS_LOCAL_WRITE | ACCESS_REMOTE_READ | ACCESS_REMOTE_WRITE) as u32
            } else {
                ACCESS_LOCAL_WRITE as u32
            },
            ..Default::default()
        };
        q.modify(&mut a, QP_STATE | QP_PKEY_INDEX | QP_PORT | QP_ACCESS_FLAGS, "INIT")?;
        Ok(q)
    }

    pub fn end(&self) -> QpEnd {
        QpEnd { qpn: unsafe { (*self.qp).qp_num }, psn: self.psn }
    }

    fn modify(&self, a: &mut QpAttr, mask: c_int, what: &str) -> io::Result<()> {
        let rc = unsafe { (lib()?.modify_qp)(self.qp, a, mask) };
        if rc != 0 {
            return Err(io::Error::other(format!(
                "QP on {} to {what} failed: {}",
                self.pd.dev.name,
                io::Error::from_raw_os_error(rc)
            )));
        }
        Ok(())
    }

    /// INIT -> RTR -> RTS against the peer's QP. `mtu` and the read depths are the lower of both ends'.
    /// `timeout`: the ack timeout exponent, 4.096 us * 2^t. On a fabric without PFC a dropped packet costs one.
    pub fn connect(
        &self,
        peer: QpEnd,
        peer_gid: &[u8; 16],
        mtu: u32,
        rd_atom: u32,
        dest_rd_atom: u32,
        timeout: u8,
    ) -> io::Result<()> {
        let mut a = QpAttr {
            qp_state: QPS_RTR,
            path_mtu: mtu,
            dest_qp_num: peer.qpn,
            rq_psn: peer.psn,
            max_dest_rd_atomic: dest_rd_atom.min(255) as u8,
            min_rnr_timer: 12,
            ah_attr: AhAttr {
                grh: GlobalRoute {
                    dgid: *peer_gid,
                    sgid_index: self.pd.dev.gid_index as u8,
                    hop_limit: 64,
                    ..Default::default()
                },
                is_global: 1,
                port_num: PORT,
                ..Default::default()
            },
            ..Default::default()
        };
        self.modify(
            &mut a,
            QP_STATE | QP_AV | QP_PATH_MTU | QP_DEST_QPN | QP_RQ_PSN | QP_MAX_DEST_RD_ATOMIC | QP_MIN_RNR_TIMER,
            "RTR",
        )?;
        let mut a = QpAttr {
            qp_state: QPS_RTS,
            timeout,
            retry_cnt: 7,
            rnr_retry: 7,
            sq_psn: self.psn,
            max_rd_atomic: rd_atom.min(255) as u8,
            ..Default::default()
        };
        self.modify(
            &mut a,
            QP_STATE | QP_TIMEOUT | QP_RETRY_CNT | QP_RNR_RETRY | QP_SQ_PSN | QP_MAX_QP_RD_ATOMIC,
            "RTS",
        )
    }

    /// Posts a chain of work requests (linked through `next`).
    ///
    /// # Safety
    /// `wr` must head a valid chain whose scatter lists name registered memory that outlives the requests.
    pub unsafe fn post(&self, wr: *mut SendWr) -> io::Result<()> {
        let mut bad = std::ptr::null_mut();
        let rc = (self.post)(self.qp as *mut c_void, wr, &mut bad);
        if rc != 0 {
            return Err(io::Error::other(format!("ibv_post_send: {}", io::Error::from_raw_os_error(rc))));
        }
        Ok(())
    }
}

impl Drop for Qp {
    fn drop(&mut self) {
        unsafe { (lib().expect("loaded").destroy_qp)(self.qp) };
    }
}

/// A starting packet sequence number: any 24-bit value works, a varied one keeps stale packets from an earlier QP
/// with the same number from matching.
pub fn psn() -> u32 {
    let b = sys::random16().unwrap_or([7; 16]);
    u32::from_le_bytes([b[0], b[1], b[2], 0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    // the values a C program built against rdma-core 50's verbs.h prints (offsetof / sizeof)
    #[test]
    fn layouts_match_rdma_core() {
        assert_eq!(size_of::<QpAttr>(), 144);
        assert_eq!(offset_of!(QpAttr, path_mtu), 8);
        assert_eq!(offset_of!(QpAttr, rq_psn), 20);
        assert_eq!(offset_of!(QpAttr, sq_psn), 24);
        assert_eq!(offset_of!(QpAttr, dest_qp_num), 28);
        assert_eq!(offset_of!(QpAttr, qp_access_flags), 32);
        assert_eq!(offset_of!(QpAttr, ah_attr), 56);
        assert_eq!(offset_of!(QpAttr, pkey_index), 120);
        assert_eq!(offset_of!(QpAttr, max_rd_atomic), 126);
        assert_eq!(offset_of!(QpAttr, max_dest_rd_atomic), 127);
        assert_eq!(offset_of!(QpAttr, min_rnr_timer), 128);
        assert_eq!(offset_of!(QpAttr, port_num), 129);
        assert_eq!(offset_of!(QpAttr, timeout), 130);
        assert_eq!(offset_of!(QpAttr, retry_cnt), 131);
        assert_eq!(offset_of!(QpAttr, rnr_retry), 132);
        assert_eq!(offset_of!(QpAttr, rate_limit), 136);
        assert_eq!(size_of::<AhAttr>(), 32);
        assert_eq!(offset_of!(AhAttr, dlid), 24);
        assert_eq!(offset_of!(AhAttr, is_global), 29);
        assert_eq!(offset_of!(AhAttr, port_num), 30);
        assert_eq!(size_of::<GlobalRoute>(), 24);
        assert_eq!(offset_of!(GlobalRoute, sgid_index), 20);
        assert_eq!(offset_of!(GlobalRoute, hop_limit), 21);
        assert_eq!(size_of::<SendWr>(), 128);
        assert_eq!(offset_of!(SendWr, num_sge), 24);
        assert_eq!(offset_of!(SendWr, opcode), 28);
        assert_eq!(offset_of!(SendWr, send_flags), 32);
        assert_eq!(offset_of!(SendWr, remote_addr), 40);
        assert_eq!(offset_of!(SendWr, rkey), 48);
        assert_eq!(offset_of!(SendWr, _qp_type), 72);
        assert_eq!(offset_of!(SendWr, _bind), 80);
        assert_eq!(size_of::<Sge>(), 16);
        assert_eq!(size_of::<Wc>(), 48);
        assert_eq!(offset_of!(Wc, status), 8);
        assert_eq!(offset_of!(Wc, vendor_err), 16);
        assert_eq!(offset_of!(Wc, qp_num), 28);
        assert_eq!(size_of::<QpInitAttr>(), 64);
        assert_eq!(offset_of!(QpInitAttr, send_cq), 8);
        assert_eq!(offset_of!(QpInitAttr, cap), 32);
        assert_eq!(offset_of!(QpInitAttr, qp_type), 52);
        assert_eq!(offset_of!(QpInitAttr, sq_sig_all), 56);
        assert_eq!(offset_of!(IbvQp, qp_num), 52);
        assert_eq!(offset_of!(IbvMr, addr), 16);
        assert_eq!(offset_of!(IbvMr, length), 24);
        assert_eq!(offset_of!(IbvMr, lkey), 36);
        assert_eq!(offset_of!(IbvMr, rkey), 40);
    }

    #[test]
    fn gids_map_ipv4() {
        assert_eq!(gid_ipv4("0000:0000:0000:0000:0000:ffff:c000:0207\n"), Some(Ipv4Addr::new(192, 0, 2, 7)));
        assert_eq!(gid_ipv4("fe80:0000:0000:0000:4ebb:47ff:fe7c:f201"), None);
        assert_eq!(gid_ipv4("0000:0000:0000:0000:0000:0000:0000:0000"), None);
    }

    #[test]
    fn sysfs_scan_takes_roce_v2_on_active_ports() {
        let root = std::env::temp_dir().join(format!("g6kv-sysfs-{}", std::process::id()));
        let mk = |dev: &str, state: &str, gids: &[(&str, &str)]| {
            let p = root.join(dev).join("ports/1");
            std::fs::create_dir_all(p.join("gids")).unwrap();
            std::fs::create_dir_all(p.join("gid_attrs/types")).unwrap();
            std::fs::write(p.join("state"), state).unwrap();
            for (i, (g, t)) in gids.iter().enumerate() {
                std::fs::write(p.join(format!("gids/{i}")), g).unwrap();
                std::fs::write(p.join(format!("gid_attrs/types/{i}")), t).unwrap();
            }
        };
        let v4 = "0000:0000:0000:0000:0000:ffff:c000:0209";
        mk(
            "rocep1s0f0",
            "4: ACTIVE\n",
            &[("fe80:0000:0000:0000:0000:0000:0000:0001", "RoCE v2\n"), (v4, "IB/RoCE v1\n"), (v4, "RoCE v2\n")],
        );
        mk("rocep1s0f1", "1: DOWN\n", &[("0000:0000:0000:0000:0000:ffff:c000:020a", "RoCE v2\n")]);
        let got = roce_v2_addrs(root.to_str().unwrap());
        assert_eq!(got, vec![(Ipv4Addr::new(192, 0, 2, 9), "rocep1s0f0".to_string(), 2)]);
        std::fs::remove_dir_all(&root).unwrap();
    }
}
