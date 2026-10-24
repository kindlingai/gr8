//! kvman-remote: a KV node. Locks --gb GiB of this box's memory and serves it as leases of page blocks (src/remote,
//! REMOTE-TIER.md), over TCP and, on addresses with a RoCE v2 GID, RDMA.
//!   kvman-remote --gb N [--port 7350] [--bind 0.0.0.0] [--min-free-gb 10] [--no-rdma] [--qp-timeout 14] [--ready FILE]
//!                [--seg-mb 1024] [--margin-s 60] [--stall-s 10] [--cooldown-s 10] [--group-plan FILE]
//!                [--holdover-s 0] [--value age|cost-shadow] [--value-spg GROUP=SECONDS,...] [--value-margin 1.25]
//!
//! --value cost-shadow (KV_SHARE_VALUE): the node still picks the group to give by its ages, and logs and counts in
//! STATS ("shadow") what the value-weighted policy would pick (share_policy.rs, D-331 M6 step 2). --value-spg: the
//! recompute seconds a GiB of each group's KV costs (KV_SHARE_SPG_FULL=15.6 for GLM 5.3 Full's group); a group not
//! listed is of unknown cost, and the policy neither takes rows from it nor gives it another's.
//! The --seg-mb .. --group-plan flags set how group leases share the node (server.rs Share, REMOTE-TIER.md).
//! --group-plan reserves each planned group's floors from the start (lines `NAME nr=RANKS floor=SEGS`); the floors
//! must fit the node. --holdover-s parks a lease opened with persist (a store's persist=lease) that long after its last
//! connection closes, so a restarted engine of the same lease name finds its blocks as it left them.
//! It refuses to start when locking N GiB would leave less than --min-free-gb of MemAvailable, and when RLIMIT_MEMLOCK
//! is below N GiB (run it under `systemd-run -p LimitMEMLOCK=infinity`, or as root). Once it serves, it prints one JSON
//! line (and writes it to --ready): the port, the bytes locked, and each address a client can reach it on over RDMA.
//! SIGTERM stops it. The memory goes back with the process.

use g6kvman::remote::server::json_escape;
use g6kvman::remote::server::Value;
use g6kvman::remote::{parse_plan, sys, verbs, Config, Node, Share};

fn usage() -> ! {
    eprintln!(
        "usage: kvman-remote --gb N [--port 7350] [--bind 0.0.0.0] [--min-free-gb 10] [--no-rdma] [--qp-timeout 14] \
         [--ready FILE] [--seg-mb 1024] [--margin-s 60] [--stall-s 10] [--cooldown-s 10] [--group-plan FILE] \
         [--holdover-s 0] [--value age|cost-shadow] [--value-spg GROUP=SECONDS,...] [--value-margin 1.25]"
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
            "--seg-mb" => share.seg_bytes = v().parse::<usize>().unwrap_or_else(|_| usage()) << 20,
            "--margin-s" => share.margin = secs(v()),
            "--stall-s" => share.stall = secs(v()),
            "--cooldown-s" => share.cooldown = secs(v()),
            "--holdover-s" => share.holdover = secs(v()),
            "--value" => {
                share.value = match v().as_str() {
                    "age" => Value::Age,
                    "cost-shadow" => Value::CostShadow,
                    _ => usage(),
                }
            }
            "--value-spg" => {
                for kv in v().split(',').filter(|x| !x.is_empty()) {
                    let (g, sec) = kv.split_once('=').unwrap_or_else(|| usage());
                    let sec: f64 = sec.parse().unwrap_or_else(|_| usage());
                    share.spg_ms.insert(g.to_string(), (sec * 1000.0).round() as u64);
                }
            }
            "--value-margin" => {
                let m: f64 = v().parse().unwrap_or_else(|_| usage());
                share.policy.margin_milli = (m * 1000.0).round() as u64;
            }
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
    if gb <= 0.0 {
        usage();
    }
    let bytes = (gb * GIB) as usize;
    if let Some(avail) = sys::mem_available() {
        let left = avail as f64 - bytes as f64;
        if left < min_free * GIB {
            die(&format!(
                "MemAvailable is {:.1} GiB: locking {gb} GiB would leave {:.1} GiB, below --min-free-gb {min_free}",
                avail as f64 / GIB,
                left / GIB
            ));
        }
    }
    let listen = format!("{bind}:{port}").parse().unwrap_or_else(|_| die(&format!("bad address {bind}:{port}")));
    let t0 = std::time::Instant::now();
    let node =
        Node::new(&Config { bytes, lock, rdma, qp_timeout, share }, listen).unwrap_or_else(|e| die(&e.to_string()));
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
        "{{\"port\":{},\"bytes\":{},\"seg_bytes\":{},\"locked\":{},\"rdma\":[{}],\"libibverbs\":{},\"pid\":{},\"setup_s\":{:.2}}}",
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
