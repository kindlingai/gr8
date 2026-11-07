//! Prepare one rank's weight ranges as dense copies with the g6map-v1 maps the loader reads.
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

/// The release planner is the source of truth for dense/vocabulary ownership and load forms. Do not include Gen
/// segments: their inputs are still read by the existing generation/cache path, not by the compact reader.
fn dense_segments(plan: &Plan) -> Vec<Seg> {
    plan.sources(0, u64::MAX).iter().filter(|s| {
        if !matches!(s.op, Op::Copy | Op::Item128 { .. }) { return false; }
        let name = Path::new(&plan.files[s.file]).file_name().unwrap_or_default().to_string_lossy();
        name.starts_with("exlr8-dense-") || name == "exlr8-vocab.safetensors"
    }).map(|s| Seg { pack_off: s.pack_off, nbytes: s.nbytes, op: s.op, file: s.file, file_off: s.file_off }).collect()
}

fn verify_dense(plan: &Plan, dir: &str) -> Result<(), String> {
    let segs = dense_segments(plan);
    let end = segs.last().map_or(0, |s| s.pack_off + s.nbytes);
    let mut copy = Plan::from_segs(plan.files.clone(), segs, Vec::new(), Vec::new(), end)?;
    let (_, moved, _) = copy.remap_compact(dir)?;
    let ranges = dense_segments(plan);
    if moved != ranges.len() { return Err(format!("{dir}: compact maps cover {moved} of {} dense ranges", ranges.len())); }
    let mut original = vec![0; 4 << 20];
    let mut packed = vec![0; 4 << 20];
    for s in ranges {
        let mut offset = 0;
        while offset < s.nbytes {
            let n = (s.nbytes - offset).min(original.len() as u64) as usize;
            crate::read_range(plan, s.pack_off + offset, &mut original[..n])?;
            crate::read_range(&copy, s.pack_off + offset, &mut packed[..n])?;
            if original[..n] != packed[..n] {
                return Err(format!("{dir}: dense copy differs from its source at image byte {}", s.pack_off + offset));
            }
            offset += n as u64;
        }
    }
    Ok(())
}

/// Write a new directory, checking its copies before publishing it. An existing directory is verified and reused,
/// never overwritten: use another destination when preparing another release, rank, width or layout.
pub fn write(src: &str, out: &str, units: &[Unit]) -> Result<(), String> {
    write_with_dense(src, out, units, None)
}

/// Also compact the dense/vocabulary file ranges in the rank's release plan, using the same map format.
pub fn write_with_dense(src: &str, out: &str, units: &[Unit], dense: Option<&Plan>) -> Result<(), String> {
    if units.is_empty() && dense.is_none() { return Err("no weight ranges to compact".into()); }
    let check = |dir: &str| -> Result<(), String> {
        verify(src, dir, units)?;
        if let Some(plan) = dense { verify_dense(plan, dir)?; }
        Ok(())
    };
    let dest = Path::new(out);
    if dest.exists() { return check(out); }
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
    let mut files: BTreeMap<PathBuf, Vec<(u64, u64)>> = BTreeMap::new();
    for u in units {
        files.entry(Path::new(src).join(fetch::width_file(u.layer, u.k))).or_default().push((u.offset, u.nbytes));
    }
    if let Some(plan) = dense {
        for s in dense_segments(plan) {
            files.entry(PathBuf::from(&plan.files[s.file])).or_default().push((s.file_off, s.nbytes));
        }
    }
    let mut names = std::collections::BTreeSet::new();
    for (source, mut ranges) in files {
        let name = source.file_name().ok_or("source has no basename")?.to_string_lossy().to_string();
        if !names.insert(name.clone()) { return Err(format!("two sources have basename {name}")); }
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
    check(temp.to_str().ok_or("compact destination is not UTF-8")?)?;
    File::open(&temp).and_then(|f| f.sync_all()).map_err(io)?;
    fs::rename(&temp, dest).map_err(|e| format!("{out}: {e}"))?;
    File::open(parent).and_then(|f| f.sync_all()).map_err(io)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_and_vocab_round_trip_uses_the_planned_ranges() {
        let dir = std::env::temp_dir().join(format!("g6compact-dense-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let files: Vec<String> = ["exlr8-dense-000.safetensors", "exlr8-vocab.safetensors"].iter()
            .map(|name| dir.join(name).to_str().unwrap().to_string()).collect();
        let bytes: Vec<u8> = (0..8192).map(|i| (i % 251) as u8).collect();
        for f in &files { fs::write(f, &bytes).unwrap(); }
        let segs = vec![Seg { pack_off: 0, nbytes: 16, file: 0, file_off: 100, op: Op::Copy },
                        Seg { pack_off: 16, nbytes: 3072, file: 0, file_off: 1024, op: Op::Item128 { hidden: 128 } },
                        Seg { pack_off: 3088, nbytes: 32, file: 1, file_off: 400, op: Op::Copy }];
        let plan = Plan::from_segs(files, segs, Vec::new(), Vec::new(), 3120).unwrap();
        let out = dir.join("copies");
        write_with_dense("", out.to_str().unwrap(), &[], Some(&plan)).unwrap();
        write_with_dense("", out.to_str().unwrap(), &[], Some(&plan)).unwrap();
        assert_eq!(fs::read(out.join("exlr8-dense-000.safetensors")).unwrap(),
                   [&bytes[100..116], &bytes[1024..4096]].concat());
        assert_eq!(fs::read(out.join("exlr8-vocab.safetensors")).unwrap(), bytes[400..432]);
        fs::write(out.join("exlr8-vocab.safetensors"), [0; 32]).unwrap();
        assert!(write_with_dense("", out.to_str().unwrap(), &[], Some(&plan)).unwrap_err().contains("differs"));
        fs::remove_dir_all(dir).unwrap();
    }
}
