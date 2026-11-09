//! kvman-ledger: the host ledger of a box shared by a Flash rank and a KV node (src/ledger.rs). Byte counts are integers.
//!   kvman-ledger FILE init --total B [--gpu B] [--node B] [--floor B]
//!   kvman-ledger FILE show                          one JSON line
//!   kvman-ledger FILE begin FROM TO B               FROM, TO: gpu, node or free. Prints the transfer's generation
//!   kvman-ledger FILE released GEN B
//!   kvman-ledger FILE commit GEN B
//!   kvman-ledger FILE abort GEN
//!   kvman-ledger FILE recover [--stale-s 30]        aborts a transfer pending that long; prints its generation or none
//! Exit 0 on success, 1 when the ledger refuses (the reason on stderr), 2 on a usage error.

use g6kvman::ledger::{now_s, Book, Ledger, Side};

fn usage() -> ! {
    eprintln!(
        "usage: kvman-ledger FILE init --total B [--gpu B] [--node B] [--floor B] | show | begin FROM TO B | \
         released GEN B | commit GEN B | abort GEN | recover [--stale-s S]"
    );
    std::process::exit(2)
}

fn refuse(e: impl std::fmt::Display) -> ! {
    eprintln!("kvman-ledger: {e}");
    std::process::exit(1)
}

fn num(v: Option<String>) -> u64 {
    v.and_then(|v| v.parse().ok()).unwrap_or_else(|| usage())
}

fn main() {
    let mut a = std::env::args().skip(1);
    let (file, cmd) = (a.next().unwrap_or_else(|| usage()), a.next().unwrap_or_else(|| usage()));
    let l = Ledger::at(&file);
    match cmd.as_str() {
        "init" => {
            let (mut total, mut gpu, mut node, mut floor) = (None, 0, 0, 0);
            while let Some(k) = a.next() {
                match k.as_str() {
                    "--total" => total = Some(num(a.next())),
                    "--gpu" => gpu = num(a.next()),
                    "--node" => node = num(a.next()),
                    "--floor" => floor = num(a.next()),
                    _ => usage(),
                }
            }
            let b = Book::new(total.unwrap_or_else(|| usage()), gpu, node, floor).unwrap_or_else(|e| refuse(e));
            l.init(&b).unwrap_or_else(|e| refuse(e));
            println!("{}", l.show().unwrap_or_else(|e| refuse(e)).to_json());
        }
        "show" => println!("{}", l.show().unwrap_or_else(|e| refuse(e)).to_json()),
        "begin" => {
            let side = |v: Option<String>| v.as_deref().and_then(Side::parse).unwrap_or_else(|| usage());
            let (from, to, n) = (side(a.next()), side(a.next()), num(a.next()));
            println!("{}", l.apply(|b| b.begin(from, to, n, now_s())).unwrap_or_else(|e| refuse(e)));
        }
        "released" => {
            let (g, n) = (num(a.next()), num(a.next()));
            l.apply(|b| b.released(g, n)).unwrap_or_else(|e| refuse(e));
        }
        "commit" => {
            let (g, n) = (num(a.next()), num(a.next()));
            l.apply(|b| b.commit(g, n)).unwrap_or_else(|e| refuse(e));
        }
        "abort" => {
            let g = num(a.next());
            l.apply(|b| b.abort(g)).unwrap_or_else(|e| refuse(e));
        }
        "recover" => {
            let stale = match (a.next().as_deref(), a.next()) {
                (None, _) => 30,
                (Some("--stale-s"), v) => num(v),
                _ => usage(),
            };
            match l.apply(|b| Ok(b.recover(stale, now_s()))).unwrap_or_else(|e| refuse(e)) {
                Some(g) => println!("{g}"),
                None => println!("none"),
            }
        }
        _ => usage(),
    }
}
