//! GR8_S2_TRACE=PATH: a Chrome trace of one rank's S2 loop in PATH.r<rank>.json, for chrome://tracing or
//! ui.perfetto.dev. Timestamps are wall-clock microseconds, so the ranks' files and a client's request times
//! (tools/trace/merge.py) line up on one timeline. Each event is written whole as it happens, and the JSON array is
//! never closed, which both viewers accept, so a trace survives a crash. Host clock only.

use std::fs::File;
use std::io::Write;
use std::os::unix::ffi::OsStrExt;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

/// The steps thread (the boundary loop).
pub const STEPS: i32 = 0;
/// The head's control thread (arrivals, admissions, notices).
pub const CONTROL: i32 = 1;

/// Wall-clock microseconds.
pub fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_micros() as u64)
}

/// One rank's trace file. The steps and control threads write to it at once.
pub struct Trace {
    out: Mutex<(File, Vec<u8>)>,
    pid: u32,
}

impl Trace {
    /// Starts PATH.r<rank>.json; none for an empty path, or (with a line on stderr) one that cannot be written.
    pub fn open(path: &[u8], rank: u32) -> Option<Trace> {
        if path.is_empty() {
            return None;
        }
        let mut file = path.to_vec();
        file.extend_from_slice(format!(".r{rank}.json").as_bytes());
        let Ok(f) = File::create(std::ffi::OsStr::from_bytes(&file)) else {
            let mut e = std::io::stderr().lock();
            let _ = e.write_all(b"# s2 trace: cannot write ");
            let _ = e.write_all(&file);
            let _ = e.write_all(b", tracing nothing\n");
            return None;
        };
        let t = Trace { out: Mutex::new((f, Vec::with_capacity(256))), pid: rank };
        t.write(|b, p| {
            let _ = write!(
                b,
                "[\n{{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":{p},\"args\":{{\"name\":\"rank {p}\"}}}},\n"
            );
        });
        t.write(|b, p| {
            let _ = writeln!(
                b,
                "{{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":{p},\"tid\":{STEPS},\"args\":{{\"name\":\"steps\"}}}},"
            );
        });
        t.write(|b, p| {
            let _ = writeln!(
                b,
                "{{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":{p},\"tid\":{CONTROL},\"args\":{{\"name\":\"control\"}}}},"
            );
        });
        Some(t)
    }

    // one event, formatted into the reused buffer and written in one call; write errors are ignored, as fprintf's were
    fn write(&self, f: impl FnOnce(&mut Vec<u8>, u32)) {
        let mut g = self.out.lock().unwrap_or_else(|e| e.into_inner());
        let (file, buf) = &mut *g;
        buf.clear();
        f(buf, self.pid);
        let _ = file.write_all(buf);
    }

    /// A span from t0 to t1. args: the inside of a JSON object, empty for none.
    pub fn span(&self, name: &[u8], tid: i32, t0: u64, t1: u64, args: &[u8]) {
        self.write(|b, p| {
            b.extend_from_slice(b"{\"name\":\"");
            b.extend_from_slice(name);
            let _ = write!(
                b,
                "\",\"ph\":\"X\",\"pid\":{p},\"tid\":{tid},\"ts\":{t0},\"dur\":{},\"args\":{{",
                t1.saturating_sub(t0)
            );
            b.extend_from_slice(args);
            b.extend_from_slice(b"}},\n");
        });
    }

    /// A point in time on one thread.
    pub fn instant(&self, name: &[u8], tid: i32, args: &[u8]) {
        let ts = now();
        self.write(|b, p| {
            b.extend_from_slice(b"{\"name\":\"");
            b.extend_from_slice(name);
            let _ = write!(b, "\",\"ph\":\"i\",\"s\":\"t\",\"pid\":{p},\"tid\":{tid},\"ts\":{ts},\"args\":{{");
            b.extend_from_slice(args);
            b.extend_from_slice(b"}},\n");
        });
    }
}
