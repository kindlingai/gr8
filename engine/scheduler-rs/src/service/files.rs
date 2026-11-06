//! The S2 files: the shim writes `cmd.<k>` (one command each, read in order) and `req.<id>` (the request an ADMIT
//! names, checked against the ADMIT's FNV-1a hash), and `abort.s2` to end the epoch. The head publishes `ready.s2`
//! (the capability line) and every file it writes through `atomic_file`, a temporary then a rename, so a reader never
//! sees half a file.

use crate::control::{refused, require, R};
use crate::scheduler::{self as sch, Config, NOTICE_MAX};
use std::ffi::OsStr;
use std::fs;
use std::io::{Read, Write};
use std::os::unix::ffi::OsStrExt;
use std::sync::atomic::{AtomicU64, Ordering};

/// The steering profiles one boot may load (steering.h PROFILES_MAX).
pub const PROFILES_MAX: usize = 8;
/// The largest forced steering scale (steering.h SCALE_MAX).
pub const SCALE_MAX: f32 = 4.0;

/// FNV-1a over the bytes: the hash an ADMIT names for its request file.
pub fn payload_hash(b: &[u8]) -> u64 {
    let mut h: u64 = 14695981039346656037;
    for &c in b {
        h ^= u64::from(c);
        h = h.wrapping_mul(1099511628211);
    }
    h
}

fn path(p: &[u8]) -> &OsStr {
    OsStr::from_bytes(p)
}
fn join(dir: &[u8], name: &[u8]) -> Vec<u8> {
    let mut p = Vec::with_capacity(dir.len() + 1 + name.len());
    p.extend_from_slice(dir);
    p.push(b'/');
    p.extend_from_slice(name);
    p
}
fn join_n(dir: &[u8], name: &str, n: u64) -> Vec<u8> {
    join(dir, format!("{name}{n}").as_bytes())
}

/// Whether `p` names something (stat follows links).
pub fn exists(p: &[u8]) -> bool {
    fs::metadata(path(p)).is_ok()
}

/// The whole file.
pub fn read_file(p: &[u8]) -> R<Vec<u8>> {
    let mut f = fs::File::open(path(p)).map_err(|_| refused("cannot read S2 file"))?;
    let mut b = Vec::new();
    f.read_to_end(&mut b).map_err(|_| refused("S2 read failed"))?;
    Ok(b)
}

/// Writes `p.tmp` and renames it to `p`.
pub fn atomic_file(p: &[u8], bytes: &[u8]) -> R<()> {
    let mut temp = p.to_vec();
    temp.extend_from_slice(b".tmp");
    {
        let mut f = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path(&temp))
            .map_err(|_| refused("cannot create S2 file"))?;
        f.write_all(bytes).map_err(|_| refused("S2 file write failed"))?;
    }
    fs::rename(path(&temp), path(p)).map_err(|_| refused("S2 file publication failed"))
}

/// A decimal field of a command: digits only, no sign, within u64.
pub fn number(s: &[u8]) -> R<u64> {
    require(!s.is_empty(), "empty number")?;
    let mut n: u64 = 0;
    for &c in s {
        require(c.is_ascii_digit() && n <= (u64::MAX - u64::from(c - b'0')) / 10, "bad/overflow number")?;
        n = n * 10 + u64::from(c - b'0');
    }
    Ok(n)
}

/// One command file: `ADMIT <epoch> <request> <hash>` or `CANCEL <epoch> <request> 0`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Command {
    pub epoch: u64,
    pub request: u64,
    pub hash: u64,
    pub admit: bool,
}

// isspace in the C locale, as istream's >> splits words
fn space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

/// Parses one command file for `epoch`.
pub fn parse_command(text: &[u8], epoch: u64) -> R<Command> {
    let mut words = text.split(|&c| space(c)).filter(|w| !w.is_empty());
    let (Some(kind), Some(e), Some(r), Some(h), None) =
        (words.next(), words.next(), words.next(), words.next(), words.next())
    else {
        return Err(refused("invalid ingress command"));
    };
    require(kind == b"ADMIT" || kind == b"CANCEL", "unknown ingress command")?;
    let c = Command { admit: kind == b"ADMIT", epoch: number(e)?, request: number(r)?, hash: number(h)? };
    require(c.epoch == epoch && (c.admit || c.hash == 0), "ingress epoch/cancel mismatch")?;
    Ok(c)
}

/// The head's reader of the shim's files. The control thread takes the commands in order; the compute thread may read
/// a request file at the same time.
pub struct Ingress {
    dir: Vec<u8>,
    epoch: u64,
    next: AtomicU64,
}

impl Ingress {
    /// A reader of `dir` for `epoch`, from cmd.0.
    pub fn new(dir: &[u8], epoch: u64) -> R<Ingress> {
        require(epoch != 0, "invalid epoch")?;
        Ok(Ingress { dir: dir.to_vec(), epoch, next: AtomicU64::new(0) })
    }

    /// The next command, consumed, or none yet.
    pub fn next(&self) -> R<Option<Command>> {
        let c = self.peek()?;
        if c.is_some() {
            let n = self.next.load(Ordering::Relaxed);
            require(n != u64::MAX, "command sequence exhausted")?;
            self.next.store(n + 1, Ordering::Relaxed);
        }
        Ok(c)
    }

    /// The next command without consuming it.
    pub fn peek(&self) -> R<Option<Command>> {
        require(!exists(&join(&self.dir, b"abort.s2")), "shim aborted epoch")?;
        let p = join_n(&self.dir, "cmd.", self.next.load(Ordering::Relaxed));
        if !exists(&p) {
            return Ok(None);
        }
        parse_command(&read_file(&p)?, self.epoch).map(Some)
    }

    /// The request file an ADMIT of this epoch names, checked against its hash.
    pub fn request(&self, c: &Command) -> R<Vec<u8>> {
        require(c.admit && c.epoch == self.epoch, "not this epoch's admission")?;
        let b = read_file(&join_n(&self.dir, "req.", c.request))?;
        require(payload_hash(&b) == c.hash, "published request payload mismatch")?;
        Ok(b)
    }
}

/// One loaded steering profile, as the capability names it.
#[derive(Clone, Copy, Debug)]
pub struct Profile<'a> {
    pub name: &'a [u8],
    pub digest: [u8; 32],
    pub active_layers: u32,
}

/// The steering receipt of a boot (steering.h BootConfig): the profiles in load order, one scope.
#[derive(Clone, Copy, Debug)]
pub struct Steering<'a> {
    pub loaded: bool,
    pub forced: bool,
    /// 1 shared, 2 full
    pub scope: u32,
    pub forced_scale_bits: u32,
    pub profiles: &'a [Profile<'a>],
}

fn valid_profile_name(s: &[u8]) -> bool {
    let alnum = |c: u8| c.is_ascii_alphanumeric();
    !s.is_empty() && s.len() <= 128 && alnum(s[0]) && s.iter().all(|&c| alnum(c) || c == b'.' || c == b'_' || c == b'-')
}

/// The receipt's own rules (Config::validate's steering half).
pub fn steering_ok(s: &Steering) -> R<()> {
    require(
        s.loaded == !s.profiles.is_empty() && s.profiles.len() <= PROFILES_MAX && (s.scope == 1 || s.scope == 2),
        "invalid loaded steering receipt",
    )?;
    for p in s.profiles {
        require(valid_profile_name(p.name) && p.active_layers != 0, "invalid loaded steering receipt")?;
    }
    require(!s.forced || s.profiles.len() == 1, "forced steering needs exactly one loaded profile")?;
    if s.forced {
        let scale = f32::from_bits(s.forced_scale_bits);
        require(scale.is_finite() && scale > 0.0 && scale <= SCALE_MAX, "forced steering needs a finite scale in (0,4]")
    } else {
        require(s.forced_scale_bits == 0, "unforced steering has a forced scale")
    }
}

/// The module's limits a config is checked against (module_info.h).
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub max_sessions: u32,
    pub mtp0: bool,
    pub pack: bool,
    pub park: bool,
}

/// Config::validate: the scheduler's rules, then the receipt's.
pub fn validate(c: &Config, s: &Steering, shape: &Shape) -> R<()> {
    sch::validate(c, shape.max_sessions, shape.mtp0, shape.pack, shape.park).map_err(refused)?;
    steering_ok(s)
}

fn hex(d: &[u8; 32]) -> String {
    const H: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(64);
    for &b in d {
        s.push(char::from(H[usize::from(b >> 4)]));
        s.push(char::from(H[usize::from(b & 15)]));
    }
    s
}
fn scope_name(s: u32) -> &'static str {
    match s {
        1 => "shared",
        2 => "full",
        _ => "invalid",
    }
}

/// ready.s2's line: `G6S2 <version> <epoch> <ranks> <pool rows> <mtp> <guard> <max rows> <lp max>`, then the
/// steering receipt (version 4: one profile, 5: several).
pub fn capability(c: &Config, s: &Steering, shape: &Shape, ranks: u32, lp_max: u32) -> R<Vec<u8>> {
    validate(c, s, shape)?;
    require((1..=6).contains(&ranks), "invalid rank count")?;
    let base = format!("{} {} {} {} {} {} {}", c.epoch, ranks, c.pool_rows, c.mtp, c.guard(), c.max_rows, lp_max);
    let mut out: Vec<u8> = Vec::new();
    if !s.loaded {
        out.extend_from_slice(format!("G6S2 3 {base}\n").as_bytes());
        return Ok(out);
    }
    if let [p] = s.profiles {
        out.extend_from_slice(format!("G6S2 4 {base} ").as_bytes());
        out.extend_from_slice(p.name);
        out.extend_from_slice(
            format!(
                " {} {} {} {} {}\n",
                hex(&p.digest),
                scope_name(s.scope),
                p.active_layers,
                u8::from(s.forced),
                s.forced_scale_bits
            )
            .as_bytes(),
        );
        return Ok(out);
    }
    out.extend_from_slice(format!("G6S2 5 {base} {} {}", scope_name(s.scope), s.profiles.len()).as_bytes());
    for p in s.profiles {
        out.push(b' ');
        out.extend_from_slice(p.name);
        out.extend_from_slice(format!(" {} {}", hex(&p.digest), p.active_layers).as_bytes());
    }
    out.push(b'\n');
    Ok(out)
}

/// What a rank's bootstrap agreement and ready.s2 read (BootstrapConfig).
#[derive(Clone, Copy, Debug)]
pub struct Boot<'a> {
    pub scheduler: &'a Config,
    pub steering: Steering<'a>,
    pub shape: Shape,
    pub rank: u32,
    pub ranks: u32,
    pub prefill_min: u32,
    pub control_version: u32,
    pub lp_max: u32,
    pub directory: &'a [u8],
    pub signature: &'a [u8],
}

/// The line every rank must match at the bootstrap: the capability, the scheduler settings that must agree, the
/// control version, then the runtime signature.
pub fn agreement(b: &Boot) -> R<Vec<u8>> {
    let c = b.scheduler;
    let mut out = capability(c, &b.steering, &b.shape, b.ranks, b.lp_max)?;
    let mut s = format!("{} {} {}", c.max_sessions, c.prefill_rows, u8::from(c.two_lanes));
    if c.decode_prefill_rows != 0 {
        s += &format!(" dpf {}", c.decode_prefill_rows);
    }
    if c.decode_rounds > 1 {
        s += &format!(" dr {}", c.decode_rounds);
    }
    if c.cp_step_rows != 0 {
        s += &format!(" cpsr {}{} after {}", c.cp_step_rows, if c.cp_seg { " seg" } else { "" }, c.cp_after);
    }
    if c.r2_rows != 0 {
        s += &format!(" r2 {}", c.r2_rows);
    }
    if c.tail_first {
        s += " tf";
    }
    if c.c8_pack {
        s += " c8pack";
    }
    if c.park {
        s += " park";
    }
    if u64::from(c.notice_max) != NOTICE_MAX {
        s += &format!(" nmax {}", c.notice_max);
    }
    if b.prefill_min != 0 {
        s += &format!(" pfmin {}", b.prefill_min);
    }
    s += &format!(" ctl {}\n", b.control_version);
    out.extend_from_slice(s.as_bytes());
    out.extend_from_slice(b.signature);
    Ok(out)
}

/// The head's ready.s2, once every rank's link is up (`links` of them, `missing` when one is absent).
pub fn publish_ready(b: &Boot, links: usize, missing: bool) -> R<()> {
    require(b.rank == 0 && links as u64 == u64::from(b.ranks.wrapping_sub(1)), "ready before all-rank bootstrap")?;
    require(!missing, "ready with missing rank")?;
    require(!exists(&join(b.directory, b"abort.s2")), "aborted before ready")?;
    let cap = capability(b.scheduler, &b.steering, &b.shape, b.ranks, b.lp_max)?;
    atomic_file(&join(b.directory, b"ready.s2"), &cap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commands() {
        assert_eq!(parse_command(b"ADMIT 1 2 3\n", 1), Ok(Command { admit: true, epoch: 1, request: 2, hash: 3 }));
        assert_eq!(parse_command(b"CANCEL\x0b1 2 0", 1).map(|c| c.admit), Ok(false));
        assert_eq!(parse_command(b"CANCEL 1 2 3", 1), Err(refused("ingress epoch/cancel mismatch")));
        assert_eq!(parse_command(b"STOP 1 2 3", 1), Err(refused("unknown ingress command")));
        assert_eq!(parse_command(b"ADMIT 1 2", 1), Err(refused("invalid ingress command")));
        assert_eq!(parse_command(b"ADMIT 1 2 3 4", 1), Err(refused("invalid ingress command")));
        assert_eq!(number(b"18446744073709551615"), Ok(u64::MAX));
        assert_eq!(number(b"18446744073709551616"), Err(refused("bad/overflow number")));
        assert_eq!(number(b""), Err(refused("empty number")));
    }
}
