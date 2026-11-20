//! Prepare one rank's expert ranges as dense copies with the g6map-v1 maps the loader reads.
//! Originals stay in place: their headers, generated tensors and uncovered ranges are still needed.

use crate::fetch::{self, Unit};
use crate::plan::{Op, Plan, Seg};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

fn io(e: std::io::Error) -> String { e.to_string() }

/// Check every requested half through the same remapping and read path as the loader. A missing map or an
/// uncovered half is an error here, not a fallback: prepare promises that every requested half has a copy.
pub fn verify(src: &str, dir: &str, units: &[Unit]) -> Result<(), String> {
    let mut files = Vec::new();
    let mut ids = BTreeMap::new();
    let mut segs = Vec::new();
    let mut at = 0u64;
    for u in units {
        let name = fetch::width_file(u.layer, u.k);
        let file = *ids.entry(name.clone()).or_insert_with(|| {
            files.push(format!("{src}/{name}"));
            files.len() - 1
        });
        segs.push(Seg { pack_off: at, nbytes: u.nbytes, op: Op::Copy, file, file_off: u.offset });
        at = at.checked_add(u.nbytes).ok_or("compact image size overflows")?;
    }
    let mut plan = Plan::from_segs(files, segs, Vec::new(), Vec::new(), at)?;
    let (_, moved, left) = plan.remap_compact(dir)?;
    if moved != units.len() || left != 0 {
        return Err(format!("{dir}: compact maps cover {moved} of {} halves", units.len()));
    }
    let mut at = 0;
    let mut buf = Vec::new();
    for u in units {
        buf.resize(u.nbytes as usize, 0);
        crate::read_range(&plan, at, &mut buf)?;
        if !u.matches(&buf) {
            return Err(format!("{dir}/{}: half at {} fails its xxhash64", fetch::width_file(u.layer, u.k), u.offset));
        }
        at += u.nbytes;
    }
    Ok(())
}

/// Write a new directory, checking its copies before publishing it. An existing directory is verified and reused,
/// never overwritten: use another destination when preparing another release, rank, width or layout.
pub fn write(src: &str, out: &str, units: &[Unit]) -> Result<(), String> {
    if units.is_empty() { return Err("no expert halves to compact".into()); }
    let dest = Path::new(out);
    if dest.exists() { return verify(src, out, units); }
    let parent = dest.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    fs::create_dir_all(parent).map_err(io)?;
    let name = dest.file_name().ok_or("compact destination needs a directory name")?.to_string_lossy();
    // create_dir, not create_dir_all: an existing staging directory is never taken over.
    let temp = parent.join(format!(".{name}.partial-{}", std::process::id()));
    fs::create_dir(&temp).map_err(|e| format!("{}: {e}", temp.display()))?;
    struct Cleanup(PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); }
    }
    let _cleanup = Cleanup(temp.clone());
    let mut files: BTreeMap<String, Vec<(u64, u64)>> = BTreeMap::new();
    for u in units {
        files.entry(fetch::width_file(u.layer, u.k)).or_default().push((u.offset, u.nbytes));
    }
    for (name, mut ranges) in files {
        let source = Path::new(src).join(&name);
        let mut input = File::open(&source).map_err(|e| format!("{}: {e}", source.display()))?;
        let size = input.metadata().map_err(io)?.len();
        ranges.sort_unstable();
        let mut runs: Vec<(u64, u64)> = Vec::new();
        for (offset, len) in ranges {
            let end = offset.checked_add(len).filter(|end| *end <= size && len > 0)
                .ok_or_else(|| format!("{name}: range {offset} + {len} outside {size} bytes"))?;
            match runs.last_mut() {
                Some((_, last)) if offset <= *last => *last = (*last).max(end),
                _ => runs.push((offset, end)),
            }
        }
        let mut copy = File::create(temp.join(&name)).map_err(io)?;
        let mut map = File::create(temp.join(format!("{name}.g6map"))).map_err(io)?;
        writeln!(map, "g6map-v1 {size}").map_err(io)?;
        let mut packed = 0;
        for (start, end) in runs {
            input.seek(SeekFrom::Start(start)).map_err(io)?;
            let len = end - start;
            let got = std::io::copy(&mut (&mut input).take(len), &mut copy).map_err(io)?;
            if got != len { return Err(format!("{name}: short read at {start}: {got} of {len}")); }
            writeln!(map, "{start} {len} {packed}").map_err(io)?;
            packed += len;
        }
        copy.sync_all().map_err(io)?;
        map.sync_all().map_err(io)?;
    }
    verify(src, temp.to_str().ok_or("compact destination is not UTF-8")?, units)?;
    File::open(&temp).and_then(|f| f.sync_all()).map_err(io)?;
    fs::rename(&temp, dest).map_err(|e| format!("{out}: {e}"))?;
    File::open(parent).and_then(|f| f.sync_all()).map_err(io)?;
    Ok(())
}
