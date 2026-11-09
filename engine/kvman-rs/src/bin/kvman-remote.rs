//! kvman-remote: a KV node. Locks --gb GiB of this box's memory and serves it as leases of page blocks (src/remote,
//! REMOTE-TIER.md), over TCP and, on addresses with a RoCE v2 GID, RDMA.
//!   kvman-remote --gb N [--port 7350] [--bind 0.0.0.0] [--min-free-gb 10] [--no-rdma] [--qp-timeout 14] [--ready FILE]
//!                [--lazy-lock] [--guard-every-s 1] [--ledger FILE]
//!                [--seg-mb 1024] [--margin-s 60] [--stall-s 10] [--cooldown-s 10] [--group-plan FILE]
//! The --seg-mb .. --group-plan flags set how group leases share the node (server.rs Share, REMOTE-TIER.md).
//! --group-plan reserves each planned group's floors from the start (lines `NAME nr=RANKS floor=SEGS`); the floors
//! must fit the node.
//! It refuses to start when locking N GiB would leave less than --min-free-gb of MemAvailable, and when RLIMIT_MEMLOCK
//! is below N GiB. --lazy-lock (for a node on a box that also serves): nothing is locked at start, and a lease's range
//! is locked when it is granted and given back to the kernel when it ends. A grant that would leave less than
//! --min-free-gb is refused, and a guard reads MemAvailable every --guard-every-s and refuses grants while the box
//! has less (server.rs); the start check is then MemAvailable >= --min-free-gb alone (run it under `systemd-run -p LimitMEMLOCK=infinity`, or as root). Once it serves, it prints one JSON
//! line (and writes it to --ready): the port, the bytes locked, and each address a client can reach it on over RDMA.
//! SIGTERM stops it. The memory goes back with the process.

use g6kvman::remote::server::json_escape;
use g6kvman::remote::{parse_plan, sys, verbs, Config, Node, Share};

fn usage() -> ! {
    eprintln!(
        "usage: kvman-remote --gb N [--port 7350] [--bind 0.0.0.0] [--min-free-gb 10] [--no-rdma] [--qp-timeout 14] \
         [--ready FILE] [--lazy-lock] [--guard-every-s 1] [--ledger FILE] [--seg-mb 1024] [--margin-s 60] \
         [--stall-s 10] [--cooldown-s 10] [--group-plan FILE]"
    );
    std::process::exit(2)
}

fn die(msg: &str) -> ! {
    eprintln!("kvman-remote: {msg}");
    std::process::exit(1)
}

const GIB: f64 = (1u64 << 30) as f64;

fn main() {
    let (mut gb, mut port, mut bind, mut min_free, mut rdma, mut qp_timeout, mut ready, mut lock) =
        (0.0f64, 7350u16, "0.0.0.0".to_string(), 10.0f64, true, 14u8, None::<String>, true);
    let (mut lazy, mut guard_every, mut ledger) = (false, 1.0f64, None);
    let mut share = Share::default();
    let secs = |v: String| std::time::Duration::from_secs_f64(v.parse().unwrap_or_else(|_| usage()));
    let mut a = std::env::args().skip(1);
    while let Some(k) = a.next() {
        let mut v = || a.next().unwrap_or_else(|| usage());
        match k.as_str() {
            "--gb" => gb = v().parse().unwrap_or_else(|_| usage()),
            "--port" => port = v().parse().unwrap_or_else(|_| usage()),
            "--bind" => bind = v(),
            "--min-free-gb" => min_free = v().parse().unwrap_or_else(|_| usage()),
            "--no-rdma" => rdma = false,
            "--qp-timeout" => qp_timeout = v().parse().unwrap_or_else(|_| usage()),
            "--ready" => ready = Some(v()),
            "--lazy-lock" => lazy = true,
            "--guard-every-s" => guard_every = v().parse().unwrap_or_else(|_| usage()),
            // with --lazy-lock: the host ledger of a box shared with a Flash rank (src/ledger.rs, kvman-ledger). The
            // bytes held never pass what it credits to the node
            "--ledger" => ledger = Some(std::path::PathBuf::from(v())),
            "--seg-mb" => share.seg_bytes = v().parse::<usize>().unwrap_or_else(|_| usage()) << 20,
            "--margin-s" => share.margin = secs(v()),
            "--stall-s" => share.stall = secs(v()),
            "--cooldown-s" => share.cooldown = secs(v()),
            "--group-plan" => {
                let p = v();
                let text = std::fs::read_to_string(&p).unwrap_or_else(|e| die(&format!("{p}: {e}")));
                share.plan = parse_plan(&text).unwrap_or_else(|e| die(&format!("{p}: {e}")));
            }
            // tests only: RLIMIT_MEMLOCK in a container is usually 8 MiB
            "--no-lock" => lock = false,
            _ => usage(),
        }
    }
    if gb <= 0.0 || !(min_free >= 0.0) || !(0.0..=3600.0).contains(&guard_every) {
        usage();
    }
    let bytes = (gb * GIB) as usize;
    if let Some(avail) = sys::mem_available() {
        // lazy: nothing is locked yet; each grant checks its own bytes
        let left = avail as f64 - if lazy { 0.0 } else { bytes as f64 };
        if left < min_free * GIB {
            die(&format!(
                "MemAvailable is {:.1} GiB: {} would leave {:.1} GiB, below --min-free-gb {min_free}",
                avail as f64 / GIB,
                if lazy { "starting".to_string() } else { format!("locking {gb} GiB") },
                left / GIB
            ));
        }
    }
    let listen = format!("{bind}:{port}").parse().unwrap_or_else(|_| die(&format!("bad address {bind}:{port}")));
    let t0 = std::time::Instant::now();
    let cfg = Config {
        bytes,
        lock,
        rdma,
        qp_timeout,
        lazy,
        min_free: (min_free * GIB) as u64,
        guard_every: std::time::Duration::from_secs_f64(guard_every),
        ledger,
        share,
    };
    let node = Node::new(&cfg, listen).unwrap_or_else(|e| die(&e.to_string()));
    let addrs = if rdma { verbs::roce_v2_addrs(verbs::SYSFS) } else { Vec::new() };
    let lib = rdma && verbs::available();
    let rd: Vec<String> = if lib {
        addrs
            .iter()
            .map(|(ip, dev, gid)| format!("{{\"ip\":\"{ip}\",\"dev\":\"{}\",\"gid\":{gid}}}", json_escape(dev)))
            .collect()
    } else {
        Vec::new()
    };
    let line = format!(
        "{{\"port\":{},\"bytes\":{},\"seg_bytes\":{},\"locked\":{},\"lazy\":{lazy},\"rdma\":[{}],\"libibverbs\":{},\"pid\":{},\"setup_s\":{:.2}}}",
        node.local_addr().map(|a| a.port()).unwrap_or(port),
        node.shared.bytes(),
        node.shared.seg_bytes(),
        node.shared.locked(),
        rd.join(","),
        lib,
        std::process::id(),
        t0.elapsed().as_secs_f64()
    );
    println!("{line}");
    if let Some(p) = ready {
        let tmp = format!("{p}.tmp");
        if std::fs::write(&tmp, format!("{line}\n")).and_then(|_| std::fs::rename(&tmp, &p)).is_err() {
            die(&format!("cannot write {p}"));
        }
    }
    if let Err(e) = node.serve() {
        die(&format!("accept: {e}"));
    }
}
