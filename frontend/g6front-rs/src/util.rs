//! Small shared pieces: request ids, a stderr logger, wall-clock seconds.

use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

/// `uuid.uuid4().hex`: 32 lowercase hex digits with the version-4 / RFC 4122 variant bits set.
pub fn hex_id() -> String {
    let mut b = [0u8; 16];
    getrandom::getrandom(&mut b).expect("getrandom");
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    let mut s = String::with_capacity(32);
    for x in b {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        s.push(HEX[(x >> 4) as usize] as char);
        s.push(HEX[(x & 15) as usize] as char);
    }
    s
}

pub fn new_id(prefix: &str) -> String {
    format!("{prefix}-{}", hex_id())
}

/// Return freed allocator pages to the OS (mimalloc keeps them for reuse otherwise). `force` also collects pages
/// other threads abandoned; a few ms, so callers run it off the request path.
pub fn release_memory(force: bool) {
    unsafe { libmimalloc_sys::mi_collect(force) };
}

pub fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

struct Logger {
    level: log::LevelFilter,
}

impl log::Log for Logger {
    fn enabled(&self, m: &log::Metadata) -> bool {
        m.level() <= self.level
    }
    fn log(&self, r: &log::Record) {
        if !self.enabled(r.metadata()) {
            return;
        }
        let d = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
        let (s, ms) = (d.as_secs(), d.subsec_millis());
        let (h, m, sec) = ((s / 3600) % 24, (s / 60) % 60, s % 60);
        let _ = writeln!(
            std::io::stderr().lock(),
            "{h:02}:{m:02}:{sec:02}.{ms:03}Z {} {}: {}",
            r.level(),
            r.target(),
            r.args()
        );
    }
    fn flush(&self) {}
}

pub fn init_logging(level: &str) {
    let level = match level.to_ascii_uppercase().as_str() {
        "DEBUG" => log::LevelFilter::Debug,
        "WARNING" | "WARN" => log::LevelFilter::Warn,
        "ERROR" | "CRITICAL" => log::LevelFilter::Error,
        "TRACE" => log::LevelFilter::Trace,
        _ => log::LevelFilter::Info,
    };
    let _ = log::set_boxed_logger(Box::new(Logger { level }));
    log::set_max_level(level);
}
