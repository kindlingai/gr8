//! Multi-group dense embedding release inspector and 64K-page-safe `pread` reader for `g6embed`.
//!
//! Wraps `exlr8::DenseRelease` (`qwen3_vl`, `embedding_gemma2`, `glm`) and exposes every matrix's stored
//! quantization components (`TRELLIS`, `INT4`, `MXFP8`, `FP8_BLOCK`, `FP8_ROW`, `BF16`, `FP32`) and every
//! copied non-matrix tensor (norms, biases, 5D `patch_embed.proj`) through a flat C ABI without memory mapping.

use crate::exlr8::{self, DenseFileEntry, DenseRelease, Matrix, StFile, BF16};
use std::collections::HashSet;
use std::ffi::CString;
use std::fs::File;
use std::os::unix::fs::FileExt;

pub const MAX_SHAPE_DIMS: usize = 6;
pub const MAX_COMPONENTS: usize = 3;

/// One component tensor (e.g. `trellis`, `suh`, `svh` or `fp8`, `scale` or a raw non-matrix tensor) inside a file.
#[derive(Clone, Debug)]
pub struct EmbedComponent {
    pub suffix: CString,
    pub dtype: CString,
    pub file_offset: u64,
    pub nbytes: u64,
    pub ndim: u32,
    pub shape: [u64; MAX_SHAPE_DIMS],
}

/// One HF weight or vector in a discovered multi-group exlr8 dense release.
#[derive(Clone, Debug)]
pub struct EmbedItem {
    pub hf_name: CString,
    pub stored_name: String,
    pub file_idx: usize,
    pub is_matrix: bool,
    /// `exlr8::{TRELLIS..FP32}` when `is_matrix`, else `0`.
    pub fmt: i32,
    pub side: i32,
    pub k: i32,
    pub unit: i32,
    /// Logical HF shape (`[out, inp]` for matrices; stored shape for non-matrix tensors).
    pub ndim: u32,
    pub shape: [u64; MAX_SHAPE_DIMS],
    pub dtype: CString,
    pub components: Vec<EmbedComponent>,
}

/// Open handle to a verified multi-group exlr8 dense release.
#[derive(Debug)]
pub struct EmbedRelease {
    pub profile: CString,
    pub release: DenseRelease,
    pub files: Vec<DenseFileEntry>,
    pub items: Vec<EmbedItem>,
}

fn fill_shape(src: &[usize]) -> Result<(u32, [u64; MAX_SHAPE_DIMS]), String> {
    if src.len() > MAX_SHAPE_DIMS {
        return Err(format!("tensor rank {} exceeds MAX_SHAPE_DIMS ({MAX_SHAPE_DIMS})", src.len()));
    }
    let mut arr = [0u64; MAX_SHAPE_DIMS];
    for (i, &d) in src.iter().enumerate() {
        arr[i] = d as u64;
    }
    Ok((src.len() as u32, arr))
}

impl EmbedRelease {
    /// Discovers all dense/vocab files in `dir`, optionally verifies every matrix's XXH64 block checksums
    /// in parallel across files, and builds the complete HF tensor inventory.
    pub fn open(dir: &str, verify_checksums: bool) -> Result<EmbedRelease, String> {
        let release = DenseRelease::discover(dir)?;
        let profile = CString::new(release.profile.name()).map_err(|e| e.to_string())?;
        let files = release.files.clone();

        // Inspect (and optionally checksum-verify) each file in parallel.
        let per_file: Vec<Result<Vec<EmbedItem>, String>> = std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(files.len());
            for (file_idx, entry) in files.iter().enumerate() {
                handles.push(s.spawn(move || Self::scan_file(file_idx, entry, verify_checksums)));
            }
            handles
                .into_iter()
                .map(|h| h.join().unwrap_or_else(|_| Err("embed loader worker panicked".into())))
                .collect()
        });

        let mut items = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for part in per_file {
            for item in part? {
                let key = item.hf_name.to_string_lossy().into_owned();
                if !seen.insert(key.clone()) {
                    return Err(format!("{dir}: duplicate HF tensor {key}"));
                }
                items.push(item);
            }
        }
        Ok(EmbedRelease { profile, release, files, items })
    }

    fn scan_file(file_idx: usize, entry: &DenseFileEntry, verify: bool) -> Result<Vec<EmbedItem>, String> {
        let f = StFile::open(&entry.path)?;
        exlr8::check_v2_file(&f)?;
        let mats = Matrix::names(&f);
        let mut owned: HashSet<String> = HashSet::new();
        let mut out = Vec::new();

        for m in &mats {
            let mx = Matrix::open(&f, m)?;
            if verify {
                mx.verify_checksums()?;
            }
            owned.insert(format!("{m}.record"));
            owned.insert(format!("{m}.checksum"));
            let suffixes = exlr8::data_tensors(mx.rec.fmt);
            let mut comps = Vec::with_capacity(suffixes.len());
            for &suf in suffixes {
                let tname = format!("{m}.{suf}");
                owned.insert(tname.clone());
                let info = f.info(&tname)?;
                let (ndim, shape) = fill_shape(&info.shape)?;
                let nbytes = info
                    .end
                    .checked_sub(info.start)
                    .ok_or_else(|| format!("{}: {tname} has inverted data_offsets", f.path))?;
                comps.push(EmbedComponent {
                    suffix: CString::new(suf).map_err(|e| e.to_string())?,
                    dtype: CString::new(info.dtype.as_str()).map_err(|e| e.to_string())?,
                    file_offset: info.start,
                    nbytes,
                    ndim,
                    shape,
                });
            }
            let hf = exlr8::hf_tensor_name(
                entry.profile,
                entry.group.as_deref(),
                entry.layer,
                entry.is_vocab,
                &f.meta,
                m,
                true,
            )?;
            let (ndim, shape) = fill_shape(&[mx.rec.out, mx.rec.inp])?;
            let dtype = if mx.rec.fmt == BF16 { "BF16" } else { "QUANT" };
            out.push(EmbedItem {
                hf_name: CString::new(hf).map_err(|e| e.to_string())?,
                stored_name: m.clone(),
                file_idx,
                is_matrix: true,
                fmt: mx.rec.fmt as i32,
                side: mx.rec.side as i32,
                k: mx.rec.k as i32,
                unit: mx.rec.unit as i32,
                ndim,
                shape,
                dtype: CString::new(dtype).map_err(|e| e.to_string())?,
                components: comps,
            });
        }

        for n in f.names() {
            if owned.contains(n) {
                continue;
            }
            let info = f.info(n)?;
            let hf = exlr8::hf_tensor_name(
                entry.profile,
                entry.group.as_deref(),
                entry.layer,
                entry.is_vocab,
                &f.meta,
                n,
                false,
            )?;
            let (ndim, shape) = fill_shape(&info.shape)?;
            let nbytes = info
                .end
                .checked_sub(info.start)
                .ok_or_else(|| format!("{}: {n} has inverted data_offsets", f.path))?;
            let comp = EmbedComponent {
                suffix: CString::new("raw").map_err(|e| e.to_string())?,
                dtype: CString::new(info.dtype.as_str()).map_err(|e| e.to_string())?,
                file_offset: info.start,
                nbytes,
                ndim,
                shape,
            };
            out.push(EmbedItem {
                hf_name: CString::new(hf).map_err(|e| e.to_string())?,
                stored_name: n.clone(),
                file_idx,
                is_matrix: false,
                fmt: 0,
                side: 0,
                k: 0,
                unit: 0,
                ndim,
                shape,
                dtype: CString::new(info.dtype.as_str()).map_err(|e| e.to_string())?,
                components: vec![comp],
            });
        }
        Ok(out)
    }

    /// Reads raw bytes of component `comp_idx` of item `item_idx` using `pread` into `dst` (zero mmap).
    pub fn read_component(&self, item_idx: usize, comp_idx: usize, dst: &mut [u8]) -> Result<(), String> {
        let item = self.items.get(item_idx).ok_or_else(|| format!("embed item index {item_idx} out of range"))?;
        let comp = item
            .components
            .get(comp_idx)
            .ok_or_else(|| format!("component index {comp_idx} out of range for {:?}", item.hf_name))?;
        if dst.len() as u64 != comp.nbytes {
            return Err(format!(
                "{:?}[{comp_idx}]: dst len {} != component bytes {}",
                item.hf_name,
                dst.len(),
                comp.nbytes
            ));
        }
        if dst.is_empty() {
            return Ok(());
        }
        let path = &self.files[item.file_idx].path;
        let file = File::open(path).map_err(|e| format!("cannot open {path}: {e}"))?;
        file.read_exact_at(dst, comp.file_offset)
            .map_err(|e| format!("{path}: read_exact_at({}): {e}", comp.file_offset))
    }

    /// Decodes a quantized matrix item `item_idx` directly into row-major BF16 bytes `[out, inp]`
    /// (bit-exact with `exlr8/loader/exlr8.py`). Used for exactness debug/reference mode.
    pub fn decode_bf16(&self, item_idx: usize, dst: &mut [u8], threads: usize) -> Result<(), String> {
        let item = self.items.get(item_idx).ok_or_else(|| format!("embed item index {item_idx} out of range"))?;
        if !item.is_matrix {
            return Err(format!("{:?}: decode_bf16 requires a matrix item", item.hf_name));
        }
        let path = &self.files[item.file_idx].path;
        let f = StFile::open(path)?;
        let mx = Matrix::open(&f, &item.stored_name)?;
        let want = mx.rec.out * mx.rec.inp * 2;
        if dst.len() != want {
            return Err(format!("{:?}: dst len {} != expected BF16 bytes {want}", item.hf_name, dst.len()));
        }
        if mx.rec.fmt == BF16 {
            let raw = f.read_all(&format!("{}.bf16", item.stored_name))?;
            if raw.len() != want {
                return Err(format!("{:?}: bf16 length {} != {want}", item.hf_name, raw.len()));
            }
            dst.copy_from_slice(&raw);
            return Ok(());
        }
        let v = exlr8::decode_par(&mx, 0..mx.rec.out, 0..mx.rec.inp, threads.max(1))?;
        for (i, &x) in v.iter().enumerate() {
            let b = exlr8::bf16_bits(x).to_le_bytes();
            dst[2 * i] = b[0];
            dst[2 * i + 1] = b[1];
        }
        Ok(())
    }
}
