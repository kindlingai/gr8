//! kvman-peer: serves one folder's files read-only to the boxes of a scale (src/peer.rs), so a rank of the new width
//! reads the KV store blocks another box's old rank wrote.
//!   kvman-peer --dir DIR --listen IP:PORT --token-file FILE
//! A client must send the token in FILE (hex). Once it listens it prints `kvman-peer: serving DIR on IP:PORT` (the
//! port it got for PORT 0) on stdout. SIGTERM stops it.

use g6kvman::peer;

fn usage() -> ! {
    eprintln!("usage: kvman-peer --dir DIR --listen IP:PORT --token-file FILE");
    std::process::exit(2)
}

fn main() {
    let (mut dir, mut listen, mut token) = (None, None, None);
    let mut a = std::env::args().skip(1);
    while let Some(k) = a.next() {
        let v = a.next().unwrap_or_else(|| usage());
        match k.as_str() {
            "--dir" => dir = Some(std::path::PathBuf::from(v)),
            "--listen" => listen = Some(v),
            "--token-file" => token = Some(v),
            _ => usage(),
        }
    }
    let (Some(dir), Some(listen), Some(token)) = (dir, listen, token) else { usage() };
    let token = peer::read_token(token.as_ref()).unwrap_or_else(|e| {
        eprintln!("kvman-peer: {e}");
        std::process::exit(1)
    });
    let l = std::net::TcpListener::bind(&listen).unwrap_or_else(|e| {
        eprintln!("kvman-peer: {listen}: {e}");
        std::process::exit(1)
    });
    let at = l.local_addr().map_or(listen, |a| a.to_string());
    println!("kvman-peer: serving {} on {at}", dir.display());
    peer::serve(dir, l, token);
}
