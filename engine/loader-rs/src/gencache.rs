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

    /// Keep only the `keep` newest key dirs under the cache dir (by mtime); errors are ignored (worst case:
    /// old dirs linger and a later write fails, which only costs a slow boot).
    fn prune_old(&self, keep: usize) {
        let Ok(rd) = std::fs::read_dir(&self.dir) else { return };
        let mut dirs: Vec<(std::time::SystemTime, PathBuf)> = rd
            .flatten()
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
    fn prune_touches_key_dirs_only() {
        assert!(super::is_key_name("0123456789abcdef-fedcba9876543210"));
        for n in ["models", "ds41gencache", "0123456789abcdef-fedcba987654321", "0123456789abcdefxfedcba9876543210",
                  "?-fedcba9876543210", "0123456789abcdeg-fedcba9876543210"] {
            assert!(!super::is_key_name(n), "{n}");
        }
    }
}
