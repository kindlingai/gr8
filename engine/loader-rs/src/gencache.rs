//! GR8_GEN_CACHE: disk cache of Op::Gen outputs (engine/loader-rs). The generators are pure functions of the
//! release files, TP width, rank and options, so their outputs can be computed once and read afterwards: the
//! OWN=3 full-head QO tensors alone cost ~600 s of thread-time per rank on every boot (cu-fastboot, 11-06).
//! The cache key binds the engine binary (sha256 of /proc/self/exe) and the planned manifest (xxh64), so any
//! rebuild or release/plan change is an automatic miss. Every read is verified (key, size, xxh64); any miss
//! or error falls back to computing, and the boot proceeds unchanged. GR8_GEN_CACHE unset = old behavior.

use crate::sha256::Sha256;
use crate::xxh64::xxh64;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct GenCache {
    dir: PathBuf,
    key: String,
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub hit_bytes: AtomicU64,
    pub written: AtomicU64,
    pub written_bytes: AtomicU64,
}

/// The running binary's sha256 (first 16 hex chars): any engine rebuild changes it, so a rebuilt generator
/// never reads a stale cache entry. Any read error yields "?" (a key that matches nothing).
fn engine_id() -> &'static str {
    static ID: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    ID.get_or_init(|| {
        use std::io::Read;
        let mut h = Sha256::new();
        let Ok(mut f) = std::fs::File::open("/proc/self/exe") else { return "?".into() };
        let mut buf = vec![0u8; 4 << 20];
        loop {
            match f.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => h.update(&buf[..n]),
                Err(_) => return "?".into(),
            }
        }
        h.hex().chars().take(16).collect()
    })
}

/// A key dir name: <16 hex engine id>-<16 hex manifest xxh64>. prune_old removes only these, so a GR8_GEN_CACHE set to
/// a wrong directory (e.g. /root) never loses other directories.
fn is_key_name(n: &str) -> bool {
    let b = n.as_bytes();
    b.len() == 33 && b[16] == b'-' && b.iter().enumerate().all(|(i, c)| i == 16 || c.is_ascii_hexdigit())
}

/// The identity of the source files a plan reads: one line per file (path, size, mtime in ns, inode), directories
/// walked recursively in name order. Part of the key text, so a release replaced at the same path (new files: a new
/// mtime or inode) is a miss. Not a content hash (the releases are hundreds of GB): a rewrite in place that keeps
/// size and mtime is not seen. An unreadable path gives a line that says so (a key that changes when it heals).
pub fn source_identity(paths: &[&str]) -> String {
    use std::os::unix::fs::MetadataExt;
    fn walk(p: &std::path::Path, out: &mut String) {
        match std::fs::metadata(p) {
            Ok(m) if m.is_dir() => {
                let mut es: Vec<PathBuf> = match std::fs::read_dir(p) {
                    Ok(rd) => rd.flatten().map(|e| e.path()).collect(),
                    Err(e) => {
                        out.push_str(&format!("{} unreadable {e}\n", p.display()));
                        return;
                    }
                };
                es.sort();
                for e in es {
                    walk(&e, out);
                }
            }
            Ok(m) => out.push_str(&format!("{} {} {} {}\n", p.display(), m.len(), m.mtime() as i128 * 1_000_000_000 + m.mtime_nsec() as i128, m.ino())),
            Err(e) => out.push_str(&format!("{} missing {e}\n", p.display())),
        }
    }
    let mut out = String::new();
    for p in paths.iter().filter(|p| !p.is_empty()) {
        walk(std::path::Path::new(p), &mut out);
    }
    out
}

impl GenCache {
    /// From GR8_GEN_CACHE plus the planned manifest (the key text). None when the env is unset or empty: cache off.
    /// Old key dirs of other builds are pruned to the newest two (each rebuild makes a new ~17 GB dir).
    pub fn from_env(manifest: &str) -> Option<GenCache> {
        let dir = gr8knobs::GEN_CACHE.get().filter(|s| !s.is_empty())?.to_string();
        let key = format!("{}-{:016x}", engine_id(), xxh64(&[manifest.as_bytes()], 0));
        let c = GenCache {
            dir: PathBuf::from(&dir),
            key,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            hit_bytes: AtomicU64::new(0),
            written: AtomicU64::new(0),
            written_bytes: AtomicU64::new(0),
        };
        c.prune_old(2);
        Some(c)
    }

    /// Keep this load's key dir and the `keep - 1` newest other key dirs under the cache dir (by mtime), so after
    /// this load's writes at most `keep` remain; errors are ignored (worst case: old dirs linger and a later write
    /// fails, which only costs a slow boot).
    fn prune_old(&self, keep: usize) {
        let Ok(rd) = std::fs::read_dir(&self.dir) else { return };
        let keep = keep.saturating_sub(1);
        let mut dirs: Vec<(std::time::SystemTime, PathBuf)> = rd
            .flatten()
            .filter(|e| e.file_name().to_string_lossy() != self.key)
            .filter(|e| e.path().is_dir() && is_key_name(&e.file_name().to_string_lossy()))
            .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path())))
            .collect();
        if dirs.len() <= keep {
            return;
        }
        dirs.sort();
        for (_, p) in &dirs[..dirs.len() - keep] {
            let _ = std::fs::remove_dir_all(p);
        }
    }

    /// <dir>/<key>/<what sanitized>.{bin,meta}; callers pass "@<image offset hex>-<tensor>" so every entry is unique
    fn paths(&self, what: &str) -> (PathBuf, PathBuf) {
        let name: String = what
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') { c } else { '_' })
            .take(160)
            .collect();
        let base = self.dir.join(&self.key).join(name);
        (PathBuf::from(format!("{}.bin", base.display())), PathBuf::from(format!("{}.meta", base.display())))
    }

    /// The cached bytes of `what`, or None on any miss: unreadable, key/size mismatch, or hash mismatch.
    /// Meta line: "<key>\t<what>\t<nbytes>\t<xxh64>".
    pub fn read(&self, what: &str, nbytes: u64) -> Option<Vec<u8>> {
        let (bin, meta) = self.paths(what);
        let m = std::fs::read_to_string(&meta).ok()?;
        let mut it = m.split('\t');
        let (k, w, n, x) = (it.next()?, it.next()?, it.next()?, it.next()?.trim());
        if k != self.key || w != what || n != nbytes.to_string() {
            return None;
        }
        let bytes = std::fs::read(&bin).ok()?;
        if bytes.len() as u64 != nbytes || format!("{:016x}", xxh64(&[&bytes], 0)) != x {
            return None;
        }
        Some(bytes)
    }

    /// Best-effort atomic write: data first, then the meta that validates it. A crash between the two (or a
    /// torn write) is a miss on the next read. Failures log one line and continue; the boot is unaffected.
    pub fn write(&self, what: &str, bytes: &[u8]) {
        let (bin, meta) = self.paths(what);
        let dir = bin.parent().unwrap();
        let tmp = PathBuf::from(format!("{}.tmp", bin.display()));
        let mtmp = PathBuf::from(format!("{}.tmp", meta.display()));
        let r = (|| -> std::io::Result<()> {
            std::fs::create_dir_all(dir)?;
            std::fs::write(&tmp, bytes)?;
            std::fs::rename(&tmp, &bin)?;
            let line = format!("{}\t{}\t{}\t{:016x}\n", self.key, what, bytes.len(), xxh64(&[bytes], 0));
            std::fs::write(&mtmp, line)?;
            std::fs::rename(&mtmp, &meta)?;
            Ok(())
        })();
        match r {
            Ok(()) => {
                self.written.fetch_add(1, Ordering::Relaxed);
                self.written_bytes.fetch_add(bytes.len() as u64, Ordering::Relaxed);
            }
            Err(e) => eprintln!("# gen-cache: write {what}: {e} (not cached, continuing)"),
        }
    }

    /// One stderr line for the boot log: the hit/miss counts of this load.
    pub fn summary(&self) -> String {
        let (h, m) = (self.hits.load(Ordering::Relaxed), self.misses.load(Ordering::Relaxed));
        format!(
            "key {}, {} hits ({:.2} GB), {} misses, {} written ({:.2} GB)",
            self.key,
            h,
            self.hit_bytes.load(Ordering::Relaxed) as f64 / 1e9,
            m,
            self.written.load(Ordering::Relaxed),
            self.written_bytes.load(Ordering::Relaxed) as f64 / 1e9
        )
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_source_replaced_at_the_same_path_changes_the_identity() {
        let d = std::env::temp_dir().join(format!("gencache-id-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("sub")).unwrap();
        std::fs::write(d.join("sub/w.bin"), b"old weights").unwrap();
        let p = d.to_string_lossy().to_string();
        let a = super::source_identity(&[&p]);
        assert!(a.contains("w.bin"), "{a}");
        std::fs::remove_file(d.join("sub/w.bin")).unwrap();
        std::fs::write(d.join("sub/w.bin"), b"new weights").unwrap();   // same path, same size
        assert_ne!(a, super::source_identity(&[&p]));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn three_keys_leave_two_dirs() {
        let d = std::env::temp_dir().join(format!("gencache-prune-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        for (i, k) in ["0000000000000000-000000000000000a", "0000000000000000-000000000000000b",
                       "0000000000000000-000000000000000c"].iter().enumerate() {
            let c = super::GenCache { dir: d.clone(), key: k.to_string(), hits: Default::default(),
                                      misses: Default::default(), hit_bytes: Default::default(),
                                      written: Default::default(), written_bytes: Default::default() };
            c.prune_old(2);
            c.write("@0-t", &[i as u8; 8]);
            std::thread::sleep(std::time::Duration::from_millis(20));   // distinct mtimes
        }
        let mut left: Vec<String> = std::fs::read_dir(&d).unwrap().flatten()
            .map(|e| e.file_name().to_string_lossy().to_string()).collect();
        left.sort();
        assert_eq!(left, ["0000000000000000-000000000000000b", "0000000000000000-000000000000000c"]);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn prune_touches_key_dirs_only() {
        assert!(super::is_key_name("0123456789abcdef-fedcba9876543210"));
        for n in ["models", "ds41gencache", "0123456789abcdef-fedcba987654321", "0123456789abcdefxfedcba9876543210",
                  "?-fedcba9876543210", "0123456789abcdeg-fedcba9876543210"] {
            assert!(!super::is_key_name(n), "{n}");
        }
    }
}
