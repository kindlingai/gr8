//! The node's wire format, over one TCP connection per client lane. Every message starts with a 32-byte header,
//! little-endian: magic, op (a reply puts its status there), then three u64 arguments.
//!
//! | op | request | reply |
//! |---|---|---|
//! | OPEN | a = block bytes, b = blocks, c = name length, then the name | a = lease id, b = lease bytes, then a 16-byte token for ATTACH |
//! | ATTACH | a = lease id, then the token: another lane of the same lease | status |
//! | RDMA | a = QPs, b = this end's active MTU, c = its initiator read depth, then its GID (16 bytes) and each QP's (qpn, psn) as two u32 | a = the lease's remote address, b = rkey, c = MTU \| read depth << 32, then the node's GID and its QPs' (qpn, psn). The QPs live as long as the connection |
//! | READ | a = offset in the lease, b = bytes | a = bytes, then the data |
//! | WRITE | a = offset, b = bytes, then the data | status |
//! | GATHER | a = ranges, then (offset, bytes) per range as two u64 | a = total bytes, then each range's data in order |
//! | STATS | (any connection, with or without a lease) | a = bytes of JSON, then the JSON |
//!
//! A failed request replies with a nonzero status and c = the length of a message that follows. A lease ends when the
//! last connection that opened or attached it closes.

use std::io::{self, Read, Write};

pub const MAGIC: u32 = 0x5653_3647; // "G6SV"
pub const VERSION: u32 = 1;
pub const OPEN: u32 = 1;
pub const ATTACH: u32 = 2;
pub const RDMA: u32 = 3;
pub const READ: u32 = 4;
pub const WRITE: u32 = 5;
pub const GATHER: u32 = 6;
pub const STATS: u32 = 7;
pub const HEADER: usize = 32;
pub const NAME_MAX: usize = 128;
pub const MAX_QPS: u64 = 256;
pub const MAX_RANGES: u64 = 1 << 16;
pub const DEFAULT_PORT: u16 = 7350;

/// Reply statuses.
pub const OK: u32 = 0;
pub const E_PROTO: u32 = 1;
pub const E_NOLEASE: u32 = 2;
pub const E_RANGE: u32 = 3;
pub const E_NOROOM: u32 = 4;
pub const E_BUSY: u32 = 5;
pub const E_TOKEN: u32 = 6;
pub const E_RDMA: u32 = 7;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Header {
    pub op: u32,
    pub a: u64,
    pub b: u64,
    pub c: u64,
}

impl Header {
    pub fn new(op: u32, a: u64, b: u64, c: u64) -> Header {
        Header { op, a, b, c }
    }
    pub fn bytes(&self) -> [u8; HEADER] {
        let mut h = [0u8; HEADER];
        h[0..4].copy_from_slice(&(MAGIC ^ VERSION).to_le_bytes());
        h[4..8].copy_from_slice(&self.op.to_le_bytes());
        h[8..16].copy_from_slice(&self.a.to_le_bytes());
        h[16..24].copy_from_slice(&self.b.to_le_bytes());
        h[24..32].copy_from_slice(&self.c.to_le_bytes());
        h
    }
    pub fn parse(h: &[u8; HEADER]) -> io::Result<Header> {
        let u32_at = |o: usize| u32::from_le_bytes(h[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(h[o..o + 8].try_into().unwrap());
        if u32_at(0) != MAGIC ^ VERSION {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "not a gr8 KV node, or another protocol version"));
        }
        Ok(Header { op: u32_at(4), a: u64_at(8), b: u64_at(16), c: u64_at(24) })
    }
    pub fn read(r: &mut impl Read) -> io::Result<Header> {
        let mut h = [0u8; HEADER];
        r.read_exact(&mut h)?;
        Header::parse(&h)
    }
    pub fn write(&self, w: &mut impl Write) -> io::Result<()> {
        w.write_all(&self.bytes())
    }
}

/// A reply, turned into an error with the node's message when its status is not OK.
pub fn reply(r: &mut impl Read) -> io::Result<Header> {
    let h = Header::read(r)?;
    if h.op != OK {
        let mut m = vec![0u8; h.c.min(4096) as usize];
        r.read_exact(&mut m)?;
        let kind = match h.op {
            E_RANGE => io::ErrorKind::InvalidInput,
            E_NOROOM => io::ErrorKind::OutOfMemory,
            E_BUSY => io::ErrorKind::AddrInUse,
            E_TOKEN | E_NOLEASE => io::ErrorKind::PermissionDenied,
            E_RDMA => io::ErrorKind::Unsupported,
            _ => io::ErrorKind::InvalidData,
        };
        return Err(io::Error::new(kind, format!("KV node: {}", String::from_utf8_lossy(&m))));
    }
    Ok(h)
}

/// A lease name is the client's namespace (a box and rank, say): 1-128 printable ASCII characters, no spaces.
pub fn valid_name(n: &[u8]) -> bool {
    !n.is_empty() && n.len() <= NAME_MAX && n.iter().all(|&c| c.is_ascii_graphic())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_round_trip() {
        let h = Header::new(GATHER, 1, u64::MAX, 3);
        assert_eq!(Header::parse(&h.bytes()).unwrap(), h);
        let mut bad = h.bytes();
        bad[0] ^= 1;
        assert!(Header::parse(&bad).is_err());
    }

    #[test]
    fn error_replies_carry_the_message() {
        let mut buf = Header::new(E_BUSY, 0, 0, 5).bytes().to_vec();
        buf.extend_from_slice(b"taken");
        let e = reply(&mut &buf[..]).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::AddrInUse);
        assert!(e.to_string().contains("taken"));
    }
}
