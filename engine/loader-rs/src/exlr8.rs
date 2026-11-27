//! Decodes the exlr8 v2 release's dense and vocabulary matrices (`dense/exlr8-dense-LLL.safetensors`,
//! `dense/exlr8-vocab.safetensors`). The values equal the output of `loader/exlr8.py` (kindlingai/exlr8) bit for bit.
//!
//! A matrix M has `M.record`, int64 [8] = (format, out, in, K, side, unit, shared input rotation, solve method), and
//! data tensors by format (the model card's Loader section). `Matrix::decode` returns any block of the HF weight
//! (out, in) as fp32. For a trellis matrix it decodes only the 128-channel entries that cover the block on the sharded
//! side.
//!
//! Trellis reconstruction is W (in, out) = diag(suh) Hb T Hb diag(svh), Hb the block-diagonal 128-block Hadamard. The
//! reference computes the two Hadamard products as fp32 matmuls (torch on NVPL or cuBLAS). Both sum each output in
//! input order with one fused multiply-add per term, starting from zero, so `had_left` and `had_right` do the same,
//! and multiply by suh and svh as separate roundings. Any other order changes the last bit of some results.
//!
//! Written from the card's format description and checked against the reference (tests/exlr8_fixture.rs in CI,
//! tests/exlr8_compare.py on the release's files). The trellis codebook, the bit packing, the tile order and the
//! 128x128 Hadamard are exllamav3's primitives (MIT, Turboderp), as the card says.

use crate::codebook::{self, Codebook};
use crate::json::{self, Json};
use crate::xxh64::Xxh64;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs::File;
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::OnceLock;

pub const TRELLIS: i64 = 1;
pub const INT4: i64 = 2;
pub const MXFP8: i64 = 3;
pub const FP8_BLOCK: i64 = 4;
pub const FP8_ROW: i64 = 5;
pub const BF16: i64 = 6;
pub const FP32: i64 = 7;
/// FP8 E4M3 with one E8M0 scale per 32 x 32 block (DeepSeek-V4.1's dense matrices, stored as the source has them)
pub const FP8_B32: i64 = 8;
/// Row-trellis lookup table (exlr8 v4, FORMAT.md "Row trellis")
pub const ROWS: i64 = 9;
/// Signed 8-bit group-quantized dense matrix (`int8 gG`, G in {16, 32, 64, 128}); has no legacy code and requires `matrices` metadata.
pub const INT8: i64 = 10;
/// record field 4: the sharded side
pub const SIDE_REPLICATED: i64 = 0;
pub const SIDE_OUT: i64 = 1;
pub const SIDE_IN: i64 = 2;

/// The data tensors of each format, in the release's checksum order.
pub fn data_tensors(fmt: i64) -> &'static [&'static str] {
    match fmt {
        TRELLIS => &["trellis", "suh", "svh"],
        INT4 => &["q4", "s4"],
        INT8 => &["q8", "s8"],
        MXFP8 | FP8_B32 => &["fp8", "e8m0"],
        FP8_BLOCK | FP8_ROW => &["fp8", "scale"],
        BF16 => &["bf16"],
        FP32 => &["fp32"],
        ROWS => &["rows", "svh", "suh"],
        _ => &[],
    }
}

// ------------------------------------------------------------------------------------------------ files

/// The dense file of layer `l`, or the vocabulary file (`l` = None), under `dir/dense/` or `dir/`.
pub fn dense_path(dir: &str, l: Option<usize>) -> String {
    group_dense_path(dir, None, l).expect("dense_path with group=None cannot fail validation")
}

/// Validates a multi-group dense group name for safe filename construction (ASCII `[A-Za-z0-9_-]+`).
pub fn validate_group_name(group: &str) -> Result<(), String> {
    if group.is_empty() || !group.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-') {
        return Err(format!("invalid exlr8 dense group name {group:?}"));
    }
    Ok(())
}

/// The dense file of `(group, layer)`, or the vocabulary file (`layer` = None), under `dir/dense/` or `dir/`.
/// When `group` is `Some(g)` (multi-group dense profiles such as `embedding_gemma2` or `qwen3_vl`), selects
/// `exlr8-dense-<g>-<L:03>.safetensors`; when `None`, selects `exlr8-dense-<L:03>.safetensors`.
pub fn group_dense_path(dir: &str, group: Option<&str>, layer: Option<usize>) -> Result<String, String> {
    if let Some(g) = group {
        validate_group_name(g)?;
    }
    let name = match (group, layer) {
        (_, None) => "exlr8-vocab.safetensors".to_string(),
        (None, Some(l)) => format!("exlr8-dense-{l:03}.safetensors"),
        (Some(g), Some(l)) => format!("exlr8-dense-{g}-{l:03}.safetensors"),
    };
    let nested = format!("{dir}/dense/{name}");
    Ok(if Path::new(&nested).exists() { nested } else { format!("{dir}/{name}") })
}

/// Known exlr8 dense model profiles (`model_profile` safetensors metadata).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseProfile {
    Glm,
    EmbeddingGemma2,
    Qwen3Vl,
    FishOmni,
}

impl DenseProfile {
    /// Parses a safetensors `model_profile` metadata value (`None` or empty defaults to GLM).
    pub fn from_name(name: Option<&str>) -> Result<DenseProfile, String> {
        match name {
            None | Some("") | Some("glm") => Ok(DenseProfile::Glm),
            Some("embedding_gemma2") => Ok(DenseProfile::EmbeddingGemma2),
            Some("qwen3_vl") => Ok(DenseProfile::Qwen3Vl),
            Some("fish_omni") => Ok(DenseProfile::FishOmni),
            Some(other) => Err(format!("unknown exlr8 dense model_profile {other:?}")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            DenseProfile::Glm => "glm",
            DenseProfile::EmbeddingGemma2 => "embedding_gemma2",
            DenseProfile::Qwen3Vl => "qwen3_vl",
            DenseProfile::FishOmni => "fish_omni",
        }
    }

    /// Canonical group ordering for deterministic discovery.
    pub fn group_order(self) -> &'static [&'static str] {
        match self {
            DenseProfile::Glm => &["layers"],
            DenseProfile::EmbeddingGemma2 => &["text", "vision", "audio", "global"],
            DenseProfile::Qwen3Vl => &["text", "vision", "global"],
            DenseProfile::FishOmni => &["gdn", "attn", "mtp_shared", "mtp", "global", "experts"],
        }
    }

    /// HF tensor name prefix for `(group, layer)` within this profile.
    pub fn group_prefix(self, group: &str, layer: usize) -> Result<String, String> {
        match (self, group) {
            (DenseProfile::Glm, "layers") => Ok(format!("model.layers.{layer}.")),
            (DenseProfile::EmbeddingGemma2, "text") => Ok(format!("language_model.layers.{layer}.")),
            (DenseProfile::EmbeddingGemma2, "vision") => Ok(format!("vision_tower.encoder.layers.{layer}.")),
            (DenseProfile::EmbeddingGemma2, "audio") => Ok(format!("audio_tower.layers.{layer}.")),
            (DenseProfile::EmbeddingGemma2, "global") => Ok(String::new()),
            (DenseProfile::Qwen3Vl, "text") => Ok(format!("model.language_model.layers.{layer}.")),
            (DenseProfile::Qwen3Vl, "vision") => Ok(format!("model.visual.blocks.{layer}.")),
            (DenseProfile::Qwen3Vl, "global") => Ok(String::new()),
            (DenseProfile::FishOmni, "gdn" | "attn" | "experts") => Ok(format!("model.language_model.layers.{layer}.")),
            (DenseProfile::FishOmni, "mtp_shared") => Ok(format!("audio_decoder_mtp_shared.layers.{layer}.")),
            (DenseProfile::FishOmni, "mtp") => Ok(format!("audio_decoder_mtp.layers.{layer}.")),
            (DenseProfile::FishOmni, "global") => Ok(String::new()),
            _ => Err(format!("unknown dense group {group:?} for profile {:?}", self.name())),
        }
    }

    /// HF tensor name for a vocabulary file entry (`stored` matrix base or vector name).
    pub fn vocab_hf_name(self, stored: &str, is_matrix: bool) -> String {
        let mapped = match (self, stored) {
            (DenseProfile::Glm, "embed_tokens") => "model.embed_tokens",
            (DenseProfile::Glm, "lm_head") => "lm_head",
            (DenseProfile::Glm, "norm.weight") => "model.norm.weight",
            (DenseProfile::EmbeddingGemma2, "embed_tokens") => "language_model.embed_tokens",
            (DenseProfile::Qwen3Vl | DenseProfile::FishOmni, "embed_tokens") => "model.language_model.embed_tokens",
            (DenseProfile::Qwen3Vl | DenseProfile::FishOmni, "norm.weight") => "model.language_model.norm.weight",
            (DenseProfile::Qwen3Vl | DenseProfile::FishOmni, "lm_head") => "lm_head",
            (_, other) => other,
        };
        if is_matrix && !mapped.ends_with(".weight") {
            format!("{mapped}.weight")
        } else {
            mapped.to_string()
        }
    }
}

/// Maps a stored matrix base name or non-matrix tensor name inside an exlr8 dense/vocab file to its HF checkpoint key.
/// Honors explicit `hf_names` / `hf_prefix` / `tensor_prefix` metadata when present, falling back to `profile` rules
/// (which remain intact even after `strip_meta.py`).
///
/// For vocabulary files, when `hf_names` metadata is present it replaces the profile's default vocab table completely
/// (unmapped keys stay literal, matching `exlr8/loader/exlr8.py:_vocab_names`).
pub fn hf_tensor_name(
    profile: DenseProfile,
    group: Option<&str>,
    layer: Option<usize>,
    is_vocab: bool,
    meta: &Json,
    stored: &str,
    is_matrix: bool,
) -> Result<String, String> {
    if is_vocab {
        if let Some(raw) = meta.get("hf_names") {
            let parsed = match raw {
                Json::Str(s) => json::parse(s)?,
                Json::Obj(_) => raw.clone(),
                _ => return Err("hf_names metadata must be a JSON object or JSON-encoded object string".into()),
            };
            if parsed.obj().is_none() {
                return Err("hf_names metadata must decode to a JSON object".into());
            }
            let mapped = match parsed.get(stored) {
                Some(Json::Str(s)) => s.as_str(),
                Some(_) => return Err(format!("hf_names entry for {stored:?} must be a string")),
                None => stored,
            };
            return Ok(if is_matrix && !mapped.ends_with(".weight") {
                format!("{mapped}.weight")
            } else {
                mapped.to_string()
            });
        }
        return Ok(profile.vocab_hf_name(stored, is_matrix));
    }
    let l = layer.ok_or_else(|| "dense file missing layer number".to_string())?;
    let prefix = if let Some(p) = meta.get("hf_prefix").and_then(Json::str) {
        p.to_string()
    } else if profile == DenseProfile::Glm {
        let tp = meta.get("tensor_prefix").and_then(Json::str).unwrap_or("model.layers.");
        format!("{tp}{l}.")
    } else {
        let g = group.ok_or_else(|| format!("profile {:?} dense file missing group", profile.name()))?;
        profile.group_prefix(g, l)?
    };
    if is_matrix {
        Ok(format!("{prefix}{stored}.weight"))
    } else {
        Ok(format!("{prefix}{stored}"))
    }
}

/// Parses an exlr8 dense/vocab filename into `(group, layer, is_vocab)`.
pub fn parse_dense_filename(name: &str) -> Option<(Option<String>, Option<usize>, bool)> {
    if name == "exlr8-vocab.safetensors" {
        return Some((None, None, true));
    }
    let stem = name.strip_prefix("exlr8-dense-")?.strip_suffix(".safetensors")?;
    if !stem.is_empty() && stem.bytes().all(|b| b.is_ascii_digit()) {
        return Some((None, Some(stem.parse::<usize>().ok()?), false));
    }
    let (grp, num) = stem.rsplit_once('-')?;
    if grp.is_empty() || num.is_empty() || !num.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((Some(grp.to_string()), Some(num.parse::<usize>().ok()?), false))
}

fn is_width_suffix(w: &str) -> bool {
    let Some(rest) = w.strip_prefix('k') else { return false };
    let digits = rest.strip_suffix("p5").unwrap_or(rest);
    !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())
}

/// Rejects unsupported exlr8-v4 dense features (non-v2 `format`, `__metadata__.matrices`, multi-width `.kK.record`
/// tensors, or non-MCG `.trellis.<cb>` tensors) so v4 files fail closed instead of silently mis-decoding.
pub fn check_v2_file(f: &StFile) -> Result<(), String> {
    if let Some(fmt) = f.meta.get("format").and_then(Json::str) {
        if fmt != "exlr8-v2" {
            return Err(format!("{}: unsupported exlr8 format {fmt:?} (only \"exlr8-v2\" is supported)", f.path));
        }
    }
    if f.meta.get("matrices").is_some() {
        return Err(format!("{}: exlr8-v4 __metadata__.matrices is not supported by the v2 dense loader", f.path));
    }
    let names: HashSet<&str> = f.names().iter().map(String::as_str).collect();
    for &n in &names {
        if let Some(base) = n.strip_suffix(".record") {
            if let Some((m, w)) = base.rsplit_once('.') {
                if !m.is_empty() && is_width_suffix(w) && !names.contains(format!("{m}.record").as_str()) {
                    return Err(format!("{}: multi-width exlr8-v4 matrix {base:?} is not supported", f.path));
                }
            }
        }
        if n.contains(".trellis.") {
            return Err(format!("{}: non-MCG exlr8-v4 trellis tensor {n:?} is not supported", f.path));
        }
    }
    Ok(())
}

/// One discovered dense or vocabulary file in an exlr8 release directory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DenseFileEntry {
    pub path: String,
    pub profile: DenseProfile,
    pub group: Option<String>,
    pub layer: Option<usize>,
    pub is_vocab: bool,
}

/// A decoded HF tensor produced from an exlr8 dense/vocab safetensors file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedTensor {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

/// Discovered multi-group (or legacy single-group) exlr8 dense release.
#[derive(Clone, Debug)]
pub struct DenseRelease {
    pub root: String,
    pub profile: DenseProfile,
    pub files: Vec<DenseFileEntry>,
}

fn scan_dense_dir(dir: &str) -> Result<Vec<(String, Option<String>, Option<usize>, bool)>, String> {
    let mut out = Vec::new();
    for ent in std::fs::read_dir(dir).map_err(|e| format!("cannot read {dir}: {e}"))? {
        let ent = ent.map_err(|e| format!("cannot read entry in {dir}: {e}"))?;
        let fname = ent.file_name();
        let Some(s) = fname.to_str() else { continue };
        let Some((grp, layer, is_vocab)) = parse_dense_filename(s) else { continue };
        let path = ent.path().to_string_lossy().into_owned();
        out.push((path, grp, layer, is_vocab));
    }
    Ok(out)
}

impl DenseRelease {
    /// Discovers all `exlr8-dense-*.safetensors` and `exlr8-vocab.safetensors` files under `dir/dense/` (canonical,
    /// checked first) or `dir/`. Also supports split layouts where dense shards sit in `dir/dense/` and
    /// `exlr8-vocab.safetensors` sits at `dir/` (or vice versa).
    pub fn discover(dir: &str) -> Result<DenseRelease, String> {
        let nested = format!("{dir}/dense");
        let mut raw_paths = if Path::new(&nested).is_dir() { scan_dense_dir(&nested)? } else { Vec::new() };
        if raw_paths.is_empty() {
            raw_paths = scan_dense_dir(dir)?;
        } else {
            let has_vocab = raw_paths.iter().any(|(_, _, _, is_v)| *is_v);
            let has_dense = raw_paths.iter().any(|(_, _, _, is_v)| !*is_v);
            if !has_vocab {
                let root_vocab = format!("{dir}/exlr8-vocab.safetensors");
                if Path::new(&root_vocab).is_file() {
                    raw_paths.push((root_vocab, None, None, true));
                }
            }
            if !has_dense {
                for entry in scan_dense_dir(dir)? {
                    if !entry.3 {
                        raw_paths.push(entry);
                    }
                }
            }
        }
        if raw_paths.is_empty() {
            return Err(format!("{dir}: no exlr8 dense or vocab safetensors files found"));
        }
        raw_paths.sort_by(|a, b| a.0.cmp(&b.0));

        let mut profile: Option<DenseProfile> = None;
        let mut files: Vec<DenseFileEntry> = Vec::with_capacity(raw_paths.len());
        for (path, fname_grp, fname_layer, is_vocab) in raw_paths {
            let f = StFile::open(&path)?;
            check_v2_file(&f)?;
            if let Some(contents) = f.meta.get("contents").and_then(Json::str) {
                let want = if is_vocab { "vocab" } else { "dense" };
                if contents != want {
                    return Err(format!("{path}: metadata contents {contents:?}, expected {want:?}"));
                }
            }
            let file_profile = DenseProfile::from_name(f.meta.get("model_profile").and_then(Json::str))?;
            if let Some(p) = profile {
                if p != file_profile {
                    return Err(format!(
                        "{path}: model_profile {:?} conflicts with earlier {:?}",
                        file_profile.name(),
                        p.name()
                    ));
                }
            } else {
                profile = Some(file_profile);
            }
            let meta_grp = f.meta.get("group").and_then(Json::str).map(str::to_string);
            let meta_layer = match f.meta.get("layer") {
                Some(Json::Str(s)) => Some(
                    s.parse::<usize>().map_err(|_| format!("{path}: invalid layer metadata {s:?}"))?,
                ),
                Some(Json::Int(i)) if *i >= 0 => Some(*i as usize),
                Some(_) => return Err(format!("{path}: invalid layer metadata")),
                None => None,
            };
            let group = if is_vocab {
                None
            } else {
                match (fname_grp, meta_grp) {
                    (Some(a), Some(b)) if a != b => {
                        return Err(format!("{path}: filename group {a:?} disagrees with metadata group {b:?}"));
                    }
                    (Some(a), _) | (_, Some(a)) => Some(a),
                    (None, None) => {
                        if file_profile == DenseProfile::Glm {
                            Some("layers".to_string())
                        } else {
                            return Err(format!(
                                "{path}: profile {:?} dense file missing group in filename and metadata",
                                file_profile.name()
                            ));
                        }
                    }
                }
            };
            let layer = if is_vocab {
                None
            } else {
                match (fname_layer, meta_layer) {
                    (Some(a), Some(b)) if a != b => {
                        return Err(format!("{path}: filename layer {a} disagrees with metadata layer {b}"));
                    }
                    (Some(a), _) | (_, Some(a)) => Some(a),
                    (None, None) => return Err(format!("{path}: dense file missing layer number")),
                }
            };
            if let (Some(g), Some(l)) = (group.as_deref(), layer) {
                validate_group_name(g)?;
                if f.meta.get("hf_prefix").and_then(Json::str).is_none() {
                    let _ = file_profile.group_prefix(g, l)?;
                }
            }
            files.push(DenseFileEntry { path, profile: file_profile, group, layer, is_vocab });
        }

        let profile = profile.unwrap();
        let order = profile.group_order();
        fn sort_key<'a>(e: &'a DenseFileEntry, order: &[&str]) -> (usize, usize, &'a str, usize) {
            if e.is_vocab {
                (1usize, usize::MAX, "", usize::MAX)
            } else {
                let g = e.group.as_deref().unwrap_or("");
                let gi = order.iter().position(|&x| x == g).unwrap_or(order.len());
                (0usize, gi, g, e.layer.unwrap_or(0))
            }
        }
        files.sort_by(|a, b| sort_key(a, order).cmp(&sort_key(b, order)));

        Ok(DenseRelease { root: dir.to_string(), profile, files })
    }

    /// Looks up a discovered file by `(group, layer)` (pass `(None, None)` for the vocabulary file).
    /// Passing `(None, Some(l))` matches only legacy single-group files (`group` is `None` or `"layers"`).
    pub fn file(&self, group: Option<&str>, layer: Option<usize>) -> Option<&DenseFileEntry> {
        self.files.iter().find(|e| match (group, layer) {
            (None, None) => e.is_vocab,
            (None, Some(l)) => {
                !e.is_vocab && e.layer == Some(l) && (e.group.is_none() || e.group.as_deref() == Some("layers"))
            }
            (Some(g), Some(l)) => !e.is_vocab && e.layer == Some(l) && e.group.as_deref() == Some(g),
            (Some(_), None) => false,
        })
    }

    /// Decodes every discovered dense and vocabulary file into a flat HF tensor map.
    pub fn decode_all(&self, bf16: bool, verify: bool, threads: usize) -> Result<BTreeMap<String, DecodedTensor>, String> {
        let mut out = BTreeMap::new();
        for entry in &self.files {
            let f = StFile::open(&entry.path)?;
            let part = decode_dense_file(&f, entry, bf16, verify, threads)?;
            for (k, v) in part {
                if out.insert(k.clone(), v).is_some() {
                    return Err(format!("{}: duplicate HF tensor {k}", entry.path));
                }
            }
        }
        Ok(out)
    }
}

/// Decodes every matrix (to `BF16` if `bf16`, else `F32`) and copies every non-matrix tensor verbatim (preserving its
/// original dtype and shape, e.g. norms, biases, and 5D conv kernels) from one dense or vocabulary file.
pub fn decode_dense_file(
    f: &StFile,
    entry: &DenseFileEntry,
    bf16: bool,
    verify: bool,
    threads: usize,
) -> Result<BTreeMap<String, DecodedTensor>, String> {
    check_v2_file(f)?;
    let mats = Matrix::names(f);
    let mut owned: HashSet<String> = HashSet::new();
    let mut out = BTreeMap::new();
    for m in &mats {
        let mx = Matrix::open(f, m)?;
        if verify {
            mx.verify_checksums()?;
        }
        owned.insert(format!("{m}.record"));
        owned.insert(format!("{m}.checksum"));
        for s in data_tensors(mx.rec.fmt) {
            owned.insert(format!("{m}.{s}"));
        }
        let hf = hf_tensor_name(
            entry.profile,
            entry.group.as_deref(),
            entry.layer,
            entry.is_vocab,
            &f.meta,
            m,
            true,
        )?;
        let (dtype, bytes) = if bf16 && mx.rec.fmt == BF16 {
            mx.check_tensor("bf16", &["BF16"], &[mx.rec.out, mx.rec.inp])?;
            ("BF16".to_string(), f.read_all(&format!("{m}.bf16"))?)
        } else {
            let v = decode_par(&mx, 0..mx.rec.out, 0..mx.rec.inp, threads)?;
            if bf16 {
                ("BF16".to_string(), v.iter().flat_map(|&x| bf16_bits(x).to_le_bytes()).collect())
            } else {
                ("F32".to_string(), v.iter().flat_map(|x| x.to_le_bytes()).collect())
            }
        };
        let shape = vec![mx.rec.out, mx.rec.inp];
        if out.insert(hf.clone(), DecodedTensor { dtype, shape, bytes }).is_some() {
            return Err(format!("{}: duplicate HF tensor {hf}", f.path));
        }
    }
    for n in f.names() {
        if owned.contains(n) {
            continue;
        }
        let info = f.info(n)?;
        let hf = hf_tensor_name(
            entry.profile,
            entry.group.as_deref(),
            entry.layer,
            entry.is_vocab,
            &f.meta,
            n,
            false,
        )?;
        let bytes = f.read_all(n)?;
        if out
            .insert(hf.clone(), DecodedTensor { dtype: info.dtype.clone(), shape: info.shape.clone(), bytes })
            .is_some()
        {
            return Err(format!("{}: duplicate HF tensor {hf}", f.path));
        }
    }
    Ok(out)
}

/// A tensor's header entry.
#[derive(Clone, Debug)]
pub struct Info {
    pub dtype: String,
    pub shape: Vec<usize>,
    /// absolute byte range in the file
    pub start: u64,
    pub end: u64,
}

/// A safetensors file: header parsed at open, tensor bytes read on demand.
pub struct StFile {
    pub path: String,
    pub meta: Json,
    f: File,
    names: Vec<String>,
    infos: std::collections::HashMap<String, Info>,
}

impl StFile {
    pub fn open(path: &str) -> Result<StFile, String> {
        let f = File::open(path).map_err(|e| format!("cannot open {path}: {e}"))?;
        let mut n = [0u8; 8];
        f.read_exact_at(&mut n, 0).map_err(|e| format!("{path}: {e}"))?;
        let n = u64::from_le_bytes(n);
        let mut h = vec![0u8; n as usize];
        f.read_exact_at(&mut h, 8).map_err(|e| format!("{path}: {e}"))?;
        let head = json::parse(std::str::from_utf8(&h).map_err(|_| format!("{path}: bad header"))?)?;
        let obj = head.obj().ok_or(format!("{path}: header is not an object"))?;
        let mut names = Vec::new();
        let mut infos = std::collections::HashMap::new();
        for (k, v) in obj {
            if k == "__metadata__" || k.starts_with("__pad.") {
                continue;
            }
            let bad = || format!("{path}: bad header entry {k}");
            let off = v.get("data_offsets").and_then(Json::arr).ok_or_else(bad)?;
            let (a, b) = (off[0].int().ok_or_else(bad)? as u64, off[1].int().ok_or_else(bad)? as u64);
            let shape = v.get("shape").and_then(Json::arr).ok_or_else(bad)?.iter()
                .map(|d| d.int().map(|x| x as usize).ok_or_else(bad)).collect::<Result<Vec<_>, _>>()?;
            let dtype = v.get("dtype").and_then(Json::str).ok_or_else(bad)?.to_string();
            names.push(k.clone());
            infos.insert(k.clone(), Info { dtype, shape, start: 8 + n + a, end: 8 + n + b });
        }
        Ok(StFile { path: path.to_string(), meta: head.get("__metadata__").cloned().unwrap_or(Json::Null), f, names, infos })
    }

    /// Tensor names, sorted, without `__pad.*`.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    pub fn info(&self, name: &str) -> Result<&Info, String> {
        self.infos.get(name).ok_or_else(|| format!("{}: no tensor {name}", self.path))
    }

    /// `len` bytes of tensor `name` from byte `off` within it.
    pub fn read(&self, name: &str, off: u64, len: u64) -> Result<Vec<u8>, String> {
        let i = self.info(name)?;
        if off + len > i.end - i.start {
            return Err(format!("{}: read past the end of {name}", self.path));
        }
        let mut v = vec![0u8; len as usize];
        self.f.read_exact_at(&mut v, i.start + off).map_err(|e| format!("{}: {name}: {e}", self.path))?;
        Ok(v)
    }

    /// `len` bytes from the absolute file offset `at`.
    pub fn read_at(&self, at: u64, len: u64) -> Result<Vec<u8>, String> {
        let mut v = vec![0u8; len as usize];
        self.f.read_exact_at(&mut v, at).map_err(|e| format!("{}: at {at}: {e}", self.path))?;
        Ok(v)
    }

    pub fn read_all(&self, name: &str) -> Result<Vec<u8>, String> {
        let i = self.info(name)?;
        self.read(name, 0, i.end - i.start)
    }
}

// ------------------------------------------------------------------------------------------------ number formats

pub fn f16_to_f32(h: u16) -> f32 {
    let s = ((h as u32) & 0x8000) << 16;
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as u32;
    let v = if e == 0 {
        m as f32 * (1.0 / 16777216.0)          // subnormal: m * 2^-24, exact
    } else if e == 31 {
        return f32::from_bits(s | 0x7f80_0000 | (m << 13));
    } else {
        return f32::from_bits(s | (((e + 112) as u32) << 23) | (m << 13));
    };
    f32::from_bits(s | v.to_bits())
}

/// The unbiased exponent of a finite f32 (f32 subnormals give -127 and lower values never matter here).
fn exponent(x: f32) -> i32 {
    ((x.to_bits() >> 23) & 0xff) as i32 - 127
}

#[inline(always)]
pub fn pow2_f64(exp: i32) -> f64 {
    f64::from_bits(((exp + 1023) as u64) << 52)
}

/// |x| rounded to nearest even on a grid with `mant` fraction bits and exponents from `emin` up, sign kept.
fn round_to(x: f32, mant: i32, emin: i32) -> f32 {
    if x == 0.0 {
        return x;
    }
    let step = pow2_f64(exponent(x).max(emin) - mant);
    ((x.abs() as f64 / step).round_ties_even() * step).copysign(x as f64) as f32
}

/// f32 -> f16, round to nearest even, as f32. Finite values inside the f16 range only (the codebook sums).
pub(crate) fn f32_to_f16_value(x: f32) -> f32 {
    round_to(x, 10, -14)
}

/// f32 -> the nearest E4M3FN value (round to nearest even, saturating at 448), as f32.
pub fn e4m3_round(x: f32) -> f32 {
    round_to(x.clamp(-448.0, 448.0), 3, -6).clamp(-448.0, 448.0)
}

const fn e4m3_to_f32_const(b: u8) -> f32 {
    let e = ((b >> 3) & 15) as u32;
    let m = (b & 7) as u32;
    let v = if e == 0 {
        (m as f32) * (1.0 / 512.0)
    } else if e == 15 && m == 7 {
        f32::NAN
    } else {
        f32::from_bits(((e + 120) << 23) | (m << 20))
    };
    if b & 0x80 != 0 { -v } else { v }
}

const E4M3_LUT: [f32; 256] = {
    let mut t = [0.0f32; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = e4m3_to_f32_const(i as u8);
        i += 1;
    }
    t
};

/// E4M3FN byte -> f32 (exact).
#[inline(always)]
pub fn e4m3_to_f32(b: u8) -> f32 {
    E4M3_LUT[b as usize]
}

/// 2^(code - 127) as f32, the E8M0 scale. Code 0 gives an f32 subnormal, exactly.
fn e8m0_to_f32(code: u8) -> f32 {
    let e = code as i32 - 127;
    if e >= -126 { f32::from_bits(((e + 127) as u32) << 23) } else { f32::from_bits(1u32 << (e + 149)) }
}

/// f32 -> IEEE-754 binary16 (fp16) bits, round to nearest even, saturating finite values to +/-65504.
pub fn f16_bits(x: f32) -> u16 {
    if x.is_nan() {
        return 0x7e00;
    }
    let sign = ((x.to_bits() >> 16) & 0x8000) as u16;
    let ax = x.abs();
    if ax >= 65504.0 {
        return sign | 0x7bff;
    }
    if ax == 0.0 {
        return sign;
    }
    let b = ax.to_bits();
    let exp = ((b >> 23) & 0xff) as i32 - 127;
    let mant = b & 0x007f_ffff;
    if exp < -14 {
        // Subnormal or underflow to zero: values < 2^-25 always round to 0 in fp16.
        let shift = -14 - exp;
        if shift > 11 {
            return sign;
        }
        let full = 0x0080_0000u32 | mant;
        let round_shift = (shift + 13) as u32;
        let halfway = 1u32 << (round_shift - 1);
        let rem_mask = (1u32 << round_shift) - 1;
        let q = full >> round_shift;
        let rem = full & rem_mask;
        let rounded = if rem > halfway || (rem == halfway && (q & 1) != 0) { q + 1 } else { q };
        return sign | (rounded as u16);
    }
    // Normal fp16: round 23-bit mantissa down to 10 bits (shift 13), round to nearest even
    let q = mant >> 13;
    let rem = mant & 0x1fff;
    let rounded = if rem > 0x1000 || (rem == 0x1000 && (q & 1) != 0) { q + 1 } else { q };
    let mut he = (exp + 15) as u32;
    let mut hm = rounded;
    if hm == 0x0400 {
        hm = 0;
        he += 1;
        if he >= 31 {
            return sign | 0x7bff;
        }
    }
    sign | ((he as u16) << 10) | (hm as u16)
}

/// f32 -> BF16 bits, round to nearest even (torch's `.to(torch.bfloat16)`). Finite inputs.
pub fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

fn u16s(b: &[u8]) -> Vec<u16> {
    b.as_chunks::<2>().0.iter().map(|c| u16::from_le_bytes(*c)).collect()
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

// ------------------------------------------------------------------------------------------------ trellis codec

/// perm[j] = row * 16 + col of the tile element at stream position j (exllamav3's tensor-core order); rows are the
/// input side.
fn tile_perm() -> &'static [usize; 256] {
    static P: OnceLock<[usize; 256]> = OnceLock::new();
    P.get_or_init(|| {
        let mut p = [0usize; 256];
        for t in 0..32 {
            let (r0, c0) = (2 * (t % 4), t / 4);
            let rc = [(r0, c0), (r0 + 1, c0), (r0 + 8, c0), (r0 + 9, c0),
                      (r0, c0 + 8), (r0 + 1, c0 + 8), (r0 + 8, c0 + 8), (r0 + 9, c0 + 8)];
            for (n, (r, c)) in rc.iter().enumerate() {
                p[8 * t + n] = 16 * r + c;
            }
        }
        p
    })
}

/// One packed 16x16 tile at width k (32k bytes) -> 256 values, row-major (input, output). mcg: the dense matrices.
pub fn decode_tile(raw: &[u8], k: usize, out: &mut [f32; 256]) {
    decode_tile_cb(raw, k, Codebook::Mcg, out)
}

/// The same in any codebook (src/codebook.rs), the values rounded to E4M3 (the rounding is part of the format).
pub fn decode_tile_cb(raw: &[u8], k: usize, cb: Codebook, out: &mut [f32; 256]) {
    debug_assert_eq!(raw.len(), 32 * k);
    let perm = tile_perm();
    let nw = 16 * k;
    // the words in stream order: each adjacent pair swapped
    let word = |i: usize| -> u32 {
        let w = i ^ 1;
        u16::from_le_bytes([raw[2 * w], raw[2 * w + 1]]) as u32
    };
    let mut vals = [0u32; 256];
    let mut acc: u64 = 0;
    let mut have = 0usize;
    let mut wi = 0usize;
    for v in vals.iter_mut() {
        while have < k {
            acc = (acc << 16) | word(wi) as u64;
            wi += 1;
            have += 16;
        }
        *v = ((acc >> (have - k)) & ((1u64 << k) - 1)) as u32;
        have -= k;
    }
    debug_assert_eq!(wi, nw);
    // weight i's state: the 16 bits ending at weight i (pmcgx: the 20 ending at weight i | 1), tail-biting
    let tab = codebook::table(cb);
    let (sb, pair) = (cb.state_bits(), cb == Codebook::Pmcgx);
    let reach = sb.div_ceil(k);
    for i in 0..256 {
        let last = if pair { i | 1 } else { i };
        let mut s = 0u32;
        for j in 0..reach {
            s |= vals[(last + 256 - j) % 256] << (j * k);
        }
        let s = (s & ((1u32 << sb) - 1)) as usize;
        out[perm[i]] = if pair { tab[2 * s + (i & 1)] } else { tab[s] };
    }
}

/// Tiles stored [k16][c = 0..7], as in gate/up halves and dense entries on the output side -> T (rows, 128), row-major.
pub fn tiles_out_major(raw: &[u8], k: usize, rows: usize) -> Vec<f32> {
    tiles_out_major_cb(raw, k, rows, Codebook::Mcg)
}

/// Same as `tiles_out_major`, decoding tiles with codebook `cb`.
pub fn tiles_out_major_cb(raw: &[u8], k: usize, rows: usize, cb: Codebook) -> Vec<f32> {
    let tb = 32 * k;
    let mut t = vec![0f32; rows * 128];
    let mut tile = [0f32; 256];
    for k16 in 0..rows / 16 {
        for c in 0..8 {
            decode_tile_cb(&raw[(k16 * 8 + c) * tb..][..tb], k, cb, &mut tile);
            for r in 0..16 {
                t[(16 * k16 + r) * 128 + 16 * c..][..16].copy_from_slice(&tile[16 * r..16 * r + 16]);
            }
        }
    }
    t
}

/// Tiles stored [b][r = 0..7][c = 0..7], as in down halves and dense entries on the input side -> T (128, cols),
/// row-major.
pub fn tiles_in_major(raw: &[u8], k: usize, cols: usize) -> Vec<f32> {
    tiles_in_major_cb(raw, k, cols, Codebook::Mcg)
}

/// Same as `tiles_in_major`, decoding tiles with codebook `cb`.
pub fn tiles_in_major_cb(raw: &[u8], k: usize, cols: usize, cb: Codebook) -> Vec<f32> {
    let tb = 32 * k;
    let mut t = vec![0f32; 128 * cols];
    let mut tile = [0f32; 256];
    for b in 0..cols / 128 {
        for r in 0..8 {
            for c in 0..8 {
                decode_tile_cb(&raw[(64 * b + 8 * r + c) * tb..][..tb], k, cb, &mut tile);
                for i in 0..16 {
                    t[(16 * r + i) * cols + 128 * b + 16 * c..][..16].copy_from_slice(&tile[16 * i..16 * i + 16]);
                }
            }
        }
    }
    t
}

/// Sylvester Hadamard matrix of order 128 scaled by 1/sqrt(128) in f32, as the reference builds it.
fn hadamard() -> &'static [f32] {
    static H: OnceLock<Vec<f32>> = OnceLock::new();
    H.get_or_init(|| {
        let c = (1.0f64 / 128f64.sqrt()) as f32;
        (0..128 * 128u32).map(|i| if ((i / 128) & (i % 128)).count_ones() % 2 == 1 { -c } else { c }).collect()
    })
}

/// X = (Hb T) * suh[row]: T (rows, cols) row-major, rows a multiple of 128; suh one value per row.
pub(crate) fn had_left(t: &[f32], rows: usize, cols: usize, suh: &[f32]) -> Vec<f32> {
    let h = hadamard();
    let mut x = vec![0f32; rows * cols];
    let mut acc = vec![0f32; cols];
    for b in 0..rows / 128 {
        for r in 0..128 {
            acc.fill(0.0);
            for q in 0..128 {
                let a = h[r * 128 + q];
                let tr = &t[(b * 128 + q) * cols..][..cols];
                for (y, &v) in acc.iter_mut().zip(tr) {
                    *y = a.mul_add(v, *y);
                }
            }
            let s = suh[b * 128 + r];
            for (o, &y) in x[(b * 128 + r) * cols..][..cols].iter_mut().zip(&acc) {
                *o = y * s;
            }
        }
    }
    x
}

/// W = (X Hb) * svh[col]: X (rows, cols) row-major, cols a multiple of 128; svh one value per column.
pub(crate) fn had_right(x: &mut [f32], rows: usize, cols: usize, svh: &[f32]) {
    let h = hadamard();
    let mut acc = [0f32; 128];
    for r in 0..rows {
        let row = &mut x[r * cols..][..cols];
        for b in 0..cols / 128 {
            acc.fill(0.0);
            for q in 0..128 {
                let a = row[b * 128 + q];
                for (y, &v) in acc.iter_mut().zip(&h[q * 128..][..128]) {
                    *y = a.mul_add(v, *y);
                }
            }
            for (c, &y) in acc.iter().enumerate() {
                row[b * 128 + c] = y * svh[b * 128 + c];
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ matrices

/// The 8 fields of `M.record` (resolved through `matrices` metadata when present).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Record {
    pub fmt: i64,
    pub out: usize,
    pub inp: usize,
    /// Trellis width K (integer) or `int8 gG` group size G (16, 32, 64, 128); 0 for other formats.
    pub k: usize,
    pub side: i64,
    /// channels per shard on the sharded side
    pub unit: usize,
    pub rotation: i64,
    pub solve: i64,
}

/// One entry of a v4 dense/vocab/extra/split file's `matrices` metadata (FORMAT.md "Matrix descriptions").
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatrixDesc {
    pub format: String,
    pub kind: i64,
    /// Encoded K for trellis/row-trellis (negative `-2K` for half widths), G for `int8 gG`, else 0.
    pub field3: i64,
    pub side: i64,
    pub unit: usize,
    pub rotation: Option<String>,
    pub solve: i64,
}

/// Default legacy rotation names for record field 6 = 1, 2, 3 when metadata `rotations` is absent.
const DEFAULT_ROTATIONS: [&str; 3] = ["attn_in", "q_resid", "mlp_in"];

/// Parses a dense format string (FORMAT.md "Format strings") into `(kind, field3)`:
/// K (or `-2K` for half-integer `p5`) for `trellis`/`row-trellis`, G in `{16, 32, 64, 128}` for `int8 gG`, else 0.
pub fn parse_format(text: &str) -> Result<(i64, i64), String> {
    if let Some(rest) = text.strip_prefix("trellis k").or_else(|| text.strip_prefix("row-trellis k")) {
        let kind = if text.starts_with("trellis ") { TRELLIS } else { ROWS };
        let (digits, half) = match rest.strip_suffix("p5") {
            Some(d) => (d, true),
            None => (rest, false),
        };
        if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) {
            let k: i64 = digits.parse().map_err(|_| format!("dense format string {text:?} has invalid width"))?;
            let field3 = if half { -(2 * k + 1) } else { k };
            return Ok((kind, field3));
        }
        return Err(format!("dense format string {text:?} is not one this loader implements"));
    }
    match text {
        "int8 g16" => Ok((INT8, 16)),
        "int8 g32" => Ok((INT8, 32)),
        "int8 g64" => Ok((INT8, 64)),
        "int8 g128" => Ok((INT8, 128)),
        "int4 g32" => Ok((INT4, 0)),
        "mxfp8" => Ok((MXFP8, 0)),
        "fp8 b128" => Ok((FP8_BLOCK, 0)),
        "fp8 b32" => Ok((FP8_B32, 0)),
        "fp8 row" => Ok((FP8_ROW, 0)),
        "bf16" => Ok((BF16, 0)),
        "fp32" => Ok((FP32, 0)),
        _ => Err(format!("dense format string {text:?} is not one this loader implements")),
    }
}

/// Validates and parses a file's `matrices` metadata (FORMAT.md "Matrix descriptions"), or returns `Ok(None)` if absent.
/// Refuses `matrices` in an `exlr8-v2` file, unknown keys/values, invalid `unit`/`rotation`, and missing/extra matrices.
pub fn file_matrices(f: &StFile) -> Result<Option<BTreeMap<String, MatrixDesc>>, String> {
    let Some(meta_val) = f.meta.get("matrices") else {
        return Ok(None);
    };
    if f.meta.get("format").and_then(Json::str) == Some("exlr8-v2") {
        return Err(format!("{}: matrices metadata (a v4 feature) in an exlr8-v2 file", f.path));
    }
    let text = meta_val.str().ok_or_else(|| format!("{}: matrices metadata is not a JSON string", f.path))?;
    let parsed = json::parse(text).map_err(|e| format!("{}: matrices metadata: {e}", f.path))?;
    let obj = parsed.obj().ok_or_else(|| format!("{}: matrices is not a JSON object", f.path))?;
    let mut out = BTreeMap::new();
    for (base, val) in obj {
        let d = val.obj().ok_or_else(|| format!("{}: matrices[{base:?}] is not a JSON object", f.path))?;
        for k in d.keys() {
            if !matches!(k.as_str(), "format" | "side" | "unit" | "rotation" | "solve") {
                return Err(format!("{}: matrices[{base:?}]: unknown key {k:?}", f.path));
            }
        }
        let fmt_str = d.get("format").and_then(Json::str)
            .ok_or_else(|| format!("{}: matrices[{base:?}]: missing or non-string format", f.path))?;
        let (kind, field3) = parse_format(fmt_str)
            .map_err(|e| format!("{}: matrices[{base:?}]: {e}", f.path))?;
        let side_str = d.get("side").and_then(Json::str)
            .ok_or_else(|| format!("{}: matrices[{base:?}]: missing or non-string side", f.path))?;
        let side = match side_str {
            "replicated" => SIDE_REPLICATED,
            "out" => SIDE_OUT,
            "in" => SIDE_IN,
            _ => return Err(format!("{}: matrices[{base:?}]: unknown side {side_str:?}", f.path)),
        };
        let solve_str = d.get("solve").and_then(Json::str)
            .ok_or_else(|| format!("{}: matrices[{base:?}]: missing or non-string solve", f.path))?;
        let solve = match solve_str {
            "copy" | "rtn" => 0,
            "hessian" => 1,
            _ => return Err(format!("{}: matrices[{base:?}]: unknown solve {solve_str:?}", f.path)),
        };
        let unit = if side == SIDE_REPLICATED {
            if d.contains_key("unit") {
                return Err(format!("{}: matrices[{base:?}]: unit must be absent when side is replicated", f.path));
            }
            0usize
        } else {
            match d.get("unit") {
                Some(Json::Int(u)) if *u >= 1 => *u as usize,
                _ => return Err(format!("{}: matrices[{base:?}]: unit must be a positive integer", f.path)),
            }
        };
        let rotation = match d.get("rotation") {
            None => None,
            Some(Json::Str(rot)) => {
                if kind != TRELLIS && kind != ROWS {
                    return Err(format!("{}: matrices[{base:?}]: rotation is allowed for trellis and row-trellis only", f.path));
                }
                Some(rot.clone())
            }
            Some(_) => return Err(format!("{}: matrices[{base:?}]: rotation must be a string", f.path)),
        };
        out.insert(base.clone(), MatrixDesc { format: fmt_str.to_string(), kind, field3, side, unit, rotation, solve });
    }
    let stored: BTreeSet<&str> = f.names().iter().filter_map(|n| n.strip_suffix(".record")).collect();
    let extra: Vec<&str> = out.keys().map(String::as_str).filter(|k| !stored.contains(k)).collect();
    if !extra.is_empty() {
        return Err(format!("{}: matrices describes {extra:?}, which it does not hold", f.path));
    }
    let missing: Vec<&str> = stored.iter().copied().filter(|k| !out.contains_key(*k)).collect();
    if !missing.is_empty() {
        return Err(format!("{}: matrices does not describe {missing:?}", f.path));
    }
    Ok(Some(out))
}

/// Reads and resolves `name.record` against `f`'s `matrices` metadata (when present), enforcing every legacy-field
/// agreement rule of FORMAT.md "Records" and "Matrix descriptions".
pub fn matrix_record(f: &StFile, name: &str) -> Result<Record, String> {
    let info = f.info(&format!("{name}.record"))?;
    if info.dtype != "I64" || info.shape != [8] {
        return Err(format!("{}: {name}.record is not int64 [8]", f.path));
    }
    let r = f.read_all(&format!("{name}.record"))?;
    if r.len() != 64 {
        return Err(format!("{}: {name}.record is not int64 [8]", f.path));
    }
    let v: Vec<i64> = r.as_chunks::<8>().0.iter().map(|c| i64::from_le_bytes(*c)).collect();
    if v[1] < 0 || v[2] < 0 {
        return Err(format!("{}: {name}.record has negative shape [{}, {}]", f.path, v[1], v[2]));
    }
    let matrices = file_matrices(f)?;
    let Some(mats) = matrices else {
        if !(TRELLIS..=ROWS).contains(&v[0]) {
            return Err(format!("{}: {name} has legacy format {} and the file has no matrices", f.path, v[0]));
        }
        return Ok(Record {
            fmt: v[0],
            out: v[1] as usize,
            inp: v[2] as usize,
            k: if v[3] >= 0 { v[3] as usize } else { 0 },
            side: v[4],
            unit: v[5].max(0) as usize,
            rotation: v[6],
            solve: v[7],
        });
    };
    let d = mats.get(name).ok_or_else(|| format!("{}: matrices does not describe {name}", f.path))?;
    let rot_names: Vec<String> = match f.meta.get("rotations").and_then(Json::str) {
        Some(s) => s.split(',').map(|x| x.trim().to_string()).collect(),
        None => DEFAULT_ROTATIONS.iter().map(|s| s.to_string()).collect(),
    };
    let rot_str = d.rotation.as_deref().unwrap_or("own");
    let want_rot: Option<i64> = if rot_str == "own" {
        Some(0)
    } else {
        rot_names.iter().position(|x| x == rot_str).map(|idx| idx as i64 + 1)
    };
    let want_fmt = if d.kind == INT8 { 0 } else { d.kind };
    let want_f3 = if d.kind == TRELLIS || d.kind == ROWS { d.field3 } else { 0 };
    let checks: [(usize, Option<i64>); 6] = [
        (0, Some(want_fmt)),
        (3, Some(want_f3)),
        (4, Some(d.side)),
        (5, Some(d.unit as i64)),
        (6, want_rot),
        (7, Some(d.solve)),
    ];
    for (field, want) in checks {
        if v[field] != 0 && Some(v[field]) != want {
            return Err(format!("{}: {name} record field {field} is {}, disagrees with matrices description {:?}", f.path, v[field], d));
        }
    }
    Ok(Record {
        fmt: d.kind,
        out: v[1] as usize,
        inp: v[2] as usize,
        k: if d.field3 >= 0 { d.field3 as usize } else { 0 },
        side: d.side,
        unit: d.unit,
        rotation: want_rot.unwrap_or(0),
        solve: d.solve,
    })
}

/// One dense matrix of a dense or vocabulary file.
pub struct Matrix<'a> {
    pub f: &'a StFile,
    pub name: String,
    pub rec: Record,
}

impl<'a> Matrix<'a> {
    pub fn open(f: &'a StFile, name: &str) -> Result<Matrix<'a>, String> {
        let rec = matrix_record(f, name)?;
        if !(TRELLIS..=INT8).contains(&rec.fmt) {
            return Err(format!("{}: {name}: unknown format {}", f.path, rec.fmt));
        }
        if rec.fmt == TRELLIS {
            codebook::check_dense_mcg(f)?;
        }
        if rec.fmt == INT8 {
            let g = rec.k;
            let align = if rec.side == SIDE_IN { 128 } else { g };
            if !matches!(g, 16 | 32 | 64 | 128) || !rec.inp.is_multiple_of(align) {
                return Err(format!("{}: {name}: int8 group size {g} with {} inputs (side {})", f.path, rec.inp, rec.side));
            }
        }
        Ok(Matrix { f, name: name.to_string(), rec })
    }

    /// The names of a file's matrices (every name with a `.record`), in sorted order.
    pub fn names(f: &StFile) -> Vec<String> {
        f.names().iter().filter_map(|n| n.strip_suffix(".record").map(str::to_string)).collect()
    }

    fn t(&self, s: &str) -> String {
        format!("{}.{s}", self.name)
    }

    fn check_shape(&self, s: &str, shape: &[usize]) -> Result<&Info, String> {
        let i = self.f.info(&self.t(s))?;
        if i.shape != shape {
            return Err(format!("{}: {}.{s} has shape {:?}, expected {shape:?}", self.f.path, self.name, i.shape));
        }
        Ok(i)
    }

    fn check_tensor(&self, s: &str, dtypes: &[&str], shape: &[usize]) -> Result<&Info, String> {
        let i = self.check_shape(s, shape)?;
        if !dtypes.contains(&i.dtype.as_str()) {
            return Err(format!("{}: {}.{s} has dtype {:?}, expected one of {dtypes:?}", self.f.path, self.name, i.dtype));
        }
        let esz: u64 = match i.dtype.as_str() {
            "U8" | "I8" | "F8_E4M3" | "F8_E8M0" => 1,
            "F16" | "BF16" | "I16" | "U16" => 2,
            "F32" | "I32" | "U32" => 4,
            "F64" | "I64" | "U64" => 8,
            other => return Err(format!("{}: {}.{s} has unsupported dtype {other:?}", self.f.path, self.name)),
        };
        let want_bytes = shape
            .iter()
            .try_fold(esz, |acc, &d| acc.checked_mul(d as u64))
            .ok_or_else(|| format!("{}: {}.{s} shape {shape:?} overflows byte size", self.f.path, self.name))?;
        let got_bytes = i
            .end
            .checked_sub(i.start)
            .ok_or_else(|| format!("{}: {}.{s} has inverted data_offsets", self.f.path, self.name))?;
        if got_bytes != want_bytes {
            return Err(format!(
                "{}: {}.{s} has {got_bytes} bytes, expected {want_bytes} for dtype {} shape {shape:?}",
                self.f.path, self.name, i.dtype
            ));
        }
        Ok(i)
    }

    /// Rows `rows` of a row-major [out, w] tensor (w elements of `esz` bytes per row).
    fn rows_of(&self, s: &str, rows: &Range<usize>, w: usize, esz: usize) -> Result<Vec<u8>, String> {
        let rb = (w * esz) as u64;
        self.f.read(&self.t(s), rows.start as u64 * rb, rows.len() as u64 * rb)
    }

    /// Recomputes every 128-channel block's XXH64 (seed 0) checksum and verifies it against `{name}.checksum`,
    /// matching `dense_format.checksums()` and `exlr8/loader/exlr8.py:dense_checksums()` byte for byte.
    pub fn verify_checksums(&self) -> Result<(), String> {
        let r = self.rec;
        if r.side == SIDE_IN && matches!(r.fmt, MXFP8 | FP8_BLOCK | FP8_ROW | BF16 | FP32) {
            return Err(format!(
                "{}: {}: format {} is row-major and requires side=OUT or side=REPLICATED, not side=IN",
                self.f.path, self.name, r.fmt
            ));
        }
        let n_blocks = (if r.side == SIDE_IN { r.inp } else { r.out }).div_ceil(128);
        self.check_tensor("checksum", &["U64"], &[n_blocks])?;
        let raw_want = self.f.read_all(&self.t("checksum"))?;
        if raw_want.len() != n_blocks * 8 {
            return Err(format!(
                "{}: {}.checksum has {} bytes, expected {} ({} blocks)",
                self.f.path,
                self.name,
                raw_want.len(),
                n_blocks * 8,
                n_blocks
            ));
        }
        let want: Vec<u64> = raw_want.as_chunks::<8>().0.iter().map(|c| u64::from_le_bytes(*c)).collect();
        match r.fmt {
            TRELLIS => {
                if !(1..=8).contains(&r.k) {
                    return Err(format!("{}: {} trellis width {} out of range 1..=8", self.f.path, self.name, r.k));
                }
                if !r.out.is_multiple_of(128) || !r.inp.is_multiple_of(128) {
                    return Err(format!(
                        "{}: {} trellis shape ({}, {}) must be divisible by 128",
                        self.f.path, self.name, r.out, r.inp
                    ));
                }
                let side_in = r.side == SIDE_IN;
                let tb = 32 * r.k;
                let (n_ent, ent_bytes) = if side_in {
                    (r.inp / 128, r.out * 128 / 256 * tb)
                } else {
                    (r.out / 128, r.inp * 128 / 256 * tb)
                };
                self.check_tensor("trellis", &["U8"], &[n_ent, ent_bytes])?;
                self.check_tensor("suh", &["F16"], &[r.inp])?;
                self.check_tensor("svh", &["F16"], &[r.out])?;
                let suh = self.f.read_all(&self.t("suh"))?;
                let svh = self.f.read_all(&self.t("svh"))?;
                for (b, &w) in want.iter().enumerate() {
                    let lo = b * 128;
                    let ent = self.f.read(&self.t("trellis"), (b * ent_bytes) as u64, ent_bytes as u64)?;
                    let mut h = Xxh64::new(0);
                    h.update(&ent);
                    if side_in {
                        let hi = lo + 128;
                        h.update(&suh[lo * 2..hi * 2]);
                        h.update(&svh);
                    } else {
                        let hi = lo + 128;
                        h.update(&svh[lo * 2..hi * 2]);
                        h.update(&suh);
                    }
                    let got = h.digest();
                    if got != w {
                        return Err(format!("{}: {} block {b}: checksum {got:#x}, expected {w:#x}", self.f.path, self.name));
                    }
                }
            }
            INT4 if r.side == SIDE_IN => {
                if !r.inp.is_multiple_of(128) {
                    return Err(format!(
                        "{}: {} side=IN int4 inp {} must be divisible by 128",
                        self.f.path, self.name, r.inp
                    ));
                }
                self.check_tensor("q4", &["U8"], &[r.inp / 128, r.out, 64])?;
                self.check_tensor("s4", &["F16"], &[r.inp / 128, r.out, 4])?;
                for (b, &w) in want.iter().enumerate() {
                    let qb = self.f.read(&self.t("q4"), (b * r.out * 64) as u64, (r.out * 64) as u64)?;
                    let sb = self.f.read(&self.t("s4"), (b * r.out * 8) as u64, (r.out * 8) as u64)?;
                    let mut h = Xxh64::new(0);
                    h.update(&qb);
                    h.update(&sb);
                    let got = h.digest();
                    if got != w {
                        return Err(format!("{}: {} block {b}: checksum {got:#x}, expected {w:#x}", self.f.path, self.name));
                    }
                }
            }
            INT4 => {
                if !r.inp.is_multiple_of(32) {
                    return Err(format!(
                        "{}: {} int4 inp {} must be divisible by 32",
                        self.f.path, self.name, r.inp
                    ));
                }
                self.check_tensor("q4", &["U8"], &[r.out, r.inp / 2])?;
                self.check_tensor("s4", &["F16"], &[r.out, r.inp / 32])?;
                for (b, &w) in want.iter().enumerate() {
                    let rows = b * 128..((b + 1) * 128).min(r.out);
                    let mut h = Xxh64::new(0);
                    h.update(&self.rows_of("q4", &rows, r.inp / 2, 1)?);
                    h.update(&self.rows_of("s4", &rows, r.inp / 32, 2)?);
                    let got = h.digest();
                    if got != w {
                        return Err(format!("{}: {} block {b}: checksum {got:#x}, expected {w:#x}", self.f.path, self.name));
                    }
                }
            }
            INT8 if r.side == SIDE_IN => {
                let g = r.k;
                if !matches!(g, 16 | 32 | 64 | 128) || !r.inp.is_multiple_of(128) {
                    return Err(format!(
                        "{}: {} side=IN int8 g{g} inp {} must have g in {{16,32,64,128}} and inp divisible by 128",
                        self.f.path, self.name, r.inp
                    ));
                }
                let ng = 128 / g;
                self.check_tensor("q8", &["I8"], &[r.inp / 128, r.out, 128])?;
                self.check_tensor("s8", &["F16"], &[r.inp / 128, r.out, ng])?;
                for (b, &w) in want.iter().enumerate() {
                    let qb = self.f.read(&self.t("q8"), (b * r.out * 128) as u64, (r.out * 128) as u64)?;
                    let sb = self.f.read(&self.t("s8"), (b * r.out * ng * 2) as u64, (r.out * ng * 2) as u64)?;
                    let mut h = Xxh64::new(0);
                    h.update(&qb);
                    h.update(&sb);
                    let got = h.digest();
                    if got != w {
                        return Err(format!("{}: {} block {b}: checksum {got:#x}, expected {w:#x}", self.f.path, self.name));
                    }
                }
            }
            INT8 => {
                let g = r.k;
                if !matches!(g, 16 | 32 | 64 | 128) || !r.inp.is_multiple_of(g) {
                    return Err(format!(
                        "{}: {} int8 g{g} inp {} must have g in {{16,32,64,128}} and inp divisible by {g}",
                        self.f.path, self.name, r.inp
                    ));
                }
                self.check_tensor("q8", &["I8"], &[r.out, r.inp])?;
                self.check_tensor("s8", &["F16"], &[r.out, r.inp / g])?;
                for (b, &w) in want.iter().enumerate() {
                    let rows = b * 128..((b + 1) * 128).min(r.out);
                    let mut h = Xxh64::new(0);
                    h.update(&self.rows_of("q8", &rows, r.inp, 1)?);
                    h.update(&self.rows_of("s8", &rows, r.inp / g, 2)?);
                    let got = h.digest();
                    if got != w {
                        return Err(format!("{}: {} block {b}: checksum {got:#x}, expected {w:#x}", self.f.path, self.name));
                    }
                }
            }
            FP8_B32 if r.side == SIDE_IN => {
                if !r.inp.is_multiple_of(128) || !r.out.is_multiple_of(32) {
                    return Err(format!(
                        "{}: {} side=IN fp8_b32 shape ({}, {}) invalid",
                        self.f.path, self.name, r.out, r.inp
                    ));
                }
                self.check_tensor("fp8", &["U8", "F8_E4M3"], &[r.inp / 128, r.out, 128])?;
                self.check_tensor("e8m0", &["U8", "F8_E8M0"], &[r.inp / 128, r.out / 32, 4])?;
                for (b, &w) in want.iter().enumerate() {
                    let fb = self.f.read(&self.t("fp8"), (b * r.out * 128) as u64, (r.out * 128) as u64)?;
                    let eb = self.f.read(&self.t("e8m0"), (b * (r.out / 32) * 4) as u64, ((r.out / 32) * 4) as u64)?;
                    let mut h = Xxh64::new(0);
                    h.update(&fb);
                    h.update(&eb);
                    let got = h.digest();
                    if got != w {
                        return Err(format!("{}: {} block {b}: checksum {got:#x}, expected {w:#x}", self.f.path, self.name));
                    }
                }
            }
            FP8_B32 => {
                if !r.inp.is_multiple_of(32) || !r.out.is_multiple_of(32) {
                    return Err(format!(
                        "{}: {} fp8_b32 shape ({}, {}) must be divisible by 32",
                        self.f.path, self.name, r.out, r.inp
                    ));
                }
                self.check_tensor("fp8", &["U8", "F8_E4M3"], &[r.out, r.inp])?;
                self.check_tensor("e8m0", &["U8", "F8_E8M0"], &[r.out / 32, r.inp / 32])?;
                for (b, &w) in want.iter().enumerate() {
                    let rows = b * 128..((b + 1) * 128).min(r.out);
                    let srows = (4 * b)..(4 * b + 4).min(r.out / 32);
                    let mut h = Xxh64::new(0);
                    h.update(&self.rows_of("fp8", &rows, r.inp, 1)?);
                    h.update(&self.rows_of("e8m0", &srows, r.inp / 32, 1)?);
                    let got = h.digest();
                    if got != w {
                        return Err(format!("{}: {} block {b}: checksum {got:#x}, expected {w:#x}", self.f.path, self.name));
                    }
                }
            }
            MXFP8 => {
                if !r.inp.is_multiple_of(32) {
                    return Err(format!(
                        "{}: {} MXFP8 inp {} must be divisible by 32",
                        self.f.path, self.name, r.inp
                    ));
                }
                self.check_tensor("fp8", &["U8", "F8_E4M3"], &[r.out, r.inp])?;
                self.check_tensor("e8m0", &["U8", "F8_E8M0"], &[r.out, r.inp / 32])?;
                for (b, &w) in want.iter().enumerate() {
                    let rows = b * 128..((b + 1) * 128).min(r.out);
                    let mut h = Xxh64::new(0);
                    h.update(&self.rows_of("fp8", &rows, r.inp, 1)?);
                    h.update(&self.rows_of("e8m0", &rows, r.inp / 32, 1)?);
                    let got = h.digest();
                    if got != w {
                        return Err(format!("{}: {} block {b}: checksum {got:#x}, expected {w:#x}", self.f.path, self.name));
                    }
                }
            }
            FP8_BLOCK => {
                let (bo, bi) = (r.out.div_ceil(128), r.inp.div_ceil(128));
                self.check_tensor("fp8", &["U8", "F8_E4M3"], &[r.out, r.inp])?;
                self.check_tensor("scale", &["F32"], &[bo, bi])?;
                for (b, &w) in want.iter().enumerate() {
                    let rows = b * 128..((b + 1) * 128).min(r.out);
                    let mut h = Xxh64::new(0);
                    h.update(&self.rows_of("fp8", &rows, r.inp, 1)?);
                    h.update(&self.f.read(&self.t("scale"), (b * bi * 4) as u64, (bi * 4) as u64)?);
                    let got = h.digest();
                    if got != w {
                        return Err(format!("{}: {} block {b}: checksum {got:#x}, expected {w:#x}", self.f.path, self.name));
                    }
                }
            }
            FP8_ROW => {
                self.check_tensor("fp8", &["U8", "F8_E4M3"], &[r.out, r.inp])?;
                self.check_tensor("scale", &["F32"], &[r.out])?;
                for (b, &w) in want.iter().enumerate() {
                    let rows = b * 128..((b + 1) * 128).min(r.out);
                    let mut h = Xxh64::new(0);
                    h.update(&self.rows_of("fp8", &rows, r.inp, 1)?);
                    h.update(&self.rows_of("scale", &rows, 1, 4)?);
                    let got = h.digest();
                    if got != w {
                        return Err(format!("{}: {} block {b}: checksum {got:#x}, expected {w:#x}", self.f.path, self.name));
                    }
                }
            }
            BF16 => {
                self.check_tensor("bf16", &["BF16"], &[r.out, r.inp])?;
                for (b, &w) in want.iter().enumerate() {
                    let rows = b * 128..((b + 1) * 128).min(r.out);
                    let mut h = Xxh64::new(0);
                    h.update(&self.rows_of("bf16", &rows, r.inp, 2)?);
                    let got = h.digest();
                    if got != w {
                        return Err(format!("{}: {} block {b}: checksum {got:#x}, expected {w:#x}", self.f.path, self.name));
                    }
                }
            }
            FP32 => {
                self.check_tensor("fp32", &["F32"], &[r.out, r.inp])?;
                for (b, &w) in want.iter().enumerate() {
                    let rows = b * 128..((b + 1) * 128).min(r.out);
                    let mut h = Xxh64::new(0);
                    h.update(&self.rows_of("fp32", &rows, r.inp, 4)?);
                    let got = h.digest();
                    if got != w {
                        return Err(format!("{}: {} block {b}: checksum {got:#x}, expected {w:#x}", self.f.path, self.name));
                    }
                }
            }
            _ => return Err(format!("{}: {}: unsupported format {} for checksum verification", self.f.path, self.name, r.fmt)),
        }
        Ok(())
    }

    /// The block `rows` x `cols` of the HF weight (out, in) as fp32, row-major. Trellis and side-2 int4 matrices need
    /// the block's range on their sharded side on 128-channel boundaries.
    pub fn decode(&self, rows: Range<usize>, cols: Range<usize>) -> Result<Vec<f32>, String> {
        let r = self.rec;
        if rows.end > r.out || cols.end > r.inp || rows.is_empty() || cols.is_empty() {
            return Err(format!("{}: {} block {rows:?} x {cols:?} is outside ({}, {})", self.f.path, self.name, r.out, r.inp));
        }
        let (nr, nc) = (rows.len(), cols.len());
        let pick = |row_major: &dyn Fn(usize, usize) -> f32| -> Vec<f32> {
            let mut o = Vec::with_capacity(nr * nc);
            for i in 0..nr {
                for j in cols.clone() {
                    o.push(row_major(i, j));
                }
            }
            o
        };
        match r.fmt {
            TRELLIS => self.decode_trellis(rows, cols),
            INT4 => {
                let (q4, s4) = self.int4(&rows, &cols)?;
                let (hw, gw) = (nc / 2, nc / 32);
                let mut o = vec![0f32; nr * nc];
                for i in 0..nr {
                    for j in 0..nc {
                        let b = q4[i * hw + j / 2];
                        let n = if j % 2 == 0 { b & 15 } else { b >> 4 } as i32 - 8;
                        o[i * nc + j] = n as f32 * f16_to_f32(s4[i * gw + j / 32]);
                    }
                }
                Ok(o)
            }
            INT8 => {
                let g = r.k;
                let (c_aligned, c_off) = if r.side == SIDE_IN {
                    (cols.clone(), 0)
                } else {
                    let cs = cols.start / g * g;
                    let ce = cols.end.div_ceil(g) * g;
                    (cs..ce, cols.start - cs)
                };
                let (q8, s8) = self.int8(&rows, &c_aligned)?;
                let ac = c_aligned.len();
                let gw = ac / g;
                let mut o = vec![0f32; nr * nc];
                for i in 0..nr {
                    for j in 0..nc {
                        let aj = c_off + j;
                        o[i * nc + j] = q8[i * ac + aj] as f32 * f16_to_f32(s8[i * gw + aj / g]);
                    }
                }
                Ok(o)
            }
            MXFP8 => {
                self.check_shape("fp8", &[r.out, r.inp])?;
                self.check_shape("e8m0", &[r.out, r.inp / 32])?;
                let w = self.rows_of("fp8", &rows, r.inp, 1)?;
                let e = self.rows_of("e8m0", &rows, r.inp / 32, 1)?;
                Ok(pick(&|i, j| e4m3_to_f32(w[i * r.inp + j]) * e8m0_to_f32(e[i * (r.inp / 32) + j / 32])))
            }
            FP8_BLOCK => {
                let (bo, bi) = (r.out.div_ceil(128), r.inp.div_ceil(128));
                self.check_shape("fp8", &[r.out, r.inp])?;
                self.check_shape("scale", &[bo, bi])?;
                let w = self.rows_of("fp8", &rows, r.inp, 1)?;
                let s = f32s(&self.f.read_all(&self.t("scale"))?);
                Ok(pick(&|i, j| e4m3_to_f32(w[i * r.inp + j]) * s[(rows.start + i) / 128 * bi + j / 128]))
            }
            FP8_B32 => {
                let (so, si) = (r.out.div_ceil(32), r.inp.div_ceil(32));
                if r.side == SIDE_IN {
                    if !cols.start.is_multiple_of(128) || !nc.is_multiple_of(128) {
                        return Err(format!("{}: {} side-2 fp8 b32 columns {cols:?} are not on 128-input blocks", self.f.path, self.name));
                    }
                    self.check_shape("fp8", &[r.inp / 128, r.out, 128])?;
                    self.check_shape("e8m0", &[r.inp / 128, so, 4])?;
                    let mut o = vec![0f32; nr * nc];
                    for (bi, b) in (cols.start / 128..cols.end / 128).enumerate() {
                        let qb = self.f.read(&self.t("fp8"), (b * r.out * 128) as u64, (r.out * 128) as u64)?;
                        let sb = self.f.read(&self.t("e8m0"), (b * so * 4) as u64, (so * 4) as u64)?;
                        for i in 0..nr {
                            let row = rows.start + i;
                            for c in 0..128 {
                                o[i * nc + bi * 128 + c] = e4m3_to_f32(qb[row * 128 + c]) * e8m0_to_f32(sb[(row / 32) * 4 + c / 32]);
                            }
                        }
                    }
                    Ok(o)
                } else {
                    self.check_shape("fp8", &[r.out, r.inp])?;
                    self.check_shape("e8m0", &[so, si])?;
                    let w = self.rows_of("fp8", &rows, r.inp, 1)?;
                    let e = self.f.read_all(&self.t("e8m0"))?;
                    Ok(pick(&|i, j| e4m3_to_f32(w[i * r.inp + j]) * e8m0_to_f32(e[(rows.start + i) / 32 * si + j / 32])))
                }
            }
            FP8_ROW => {
                self.check_shape("fp8", &[r.out, r.inp])?;
                self.check_shape("scale", &[r.out])?;
                let w = self.rows_of("fp8", &rows, r.inp, 1)?;
                let s = f32s(&self.rows_of("scale", &rows, 1, 4)?);
                Ok(pick(&|i, j| e4m3_to_f32(w[i * r.inp + j]) * s[i]))
            }
            BF16 => {
                self.check_shape("bf16", &[r.out, r.inp])?;
                let w = u16s(&self.rows_of("bf16", &rows, r.inp, 2)?);
                Ok(pick(&|i, j| bf16_to_f32(w[i * r.inp + j])))
            }
            FP32 => {
                self.check_shape("fp32", &[r.out, r.inp])?;
                let w = f32s(&self.rows_of("fp32", &rows, r.inp, 4)?);
                Ok(pick(&|i, j| w[i * r.inp + j]))
            }
            other => Err(format!("{}: {}: decode not implemented for format {other}", self.f.path, self.name)),
        }
    }

    /// int8 gG signed codes ([rows][cols], i8 in [-127, 127]) and fp16 scale bits ([rows][cols / G]) of the block in
    /// the HF orientation. Side-2 storage (`[in/128, out, 128]` and `[in/128, out, 128/G]`) comes back transposed to
    /// this layout. Columns must fall on G-input groups (or 128-input blocks on side 2).
    /// Refuses any code == -128 and any scale that is non-finite or negative (FORMAT.md "int8 groups").
    pub fn int8(&self, rows: &Range<usize>, cols: &Range<usize>) -> Result<(Vec<i8>, Vec<u16>), String> {
        let r = self.rec;
        if r.fmt != INT8 {
            return Err(format!("{}: {} is not int8 gG", self.f.path, self.name));
        }
        let g = r.k;
        if !matches!(g, 16 | 32 | 64 | 128) || !r.inp.is_multiple_of(if r.side == SIDE_IN { 128 } else { g }) {
            return Err(format!("{}: {} int8 group size {g} with {} inputs (side {})", self.f.path, self.name, r.inp, r.side));
        }
        if rows.end > r.out || cols.end > r.inp || rows.is_empty() || cols.is_empty() {
            return Err(format!("{}: {} block {rows:?} x {cols:?} is outside ({}, {})", self.f.path, self.name, r.out, r.inp));
        }
        let (nr, nc) = (rows.len(), cols.len());
        if !cols.start.is_multiple_of(g) || !nc.is_multiple_of(g) {
            return Err(format!("{}: {} int8 g{g} columns {cols:?} are not on {g}-input groups", self.f.path, self.name));
        }
        let qi = self.f.info(&self.t("q8"))?;
        if qi.dtype != "I8" {
            return Err(format!("{}: {}.q8 has dtype {}, expected I8", self.f.path, self.name, qi.dtype));
        }
        let si = self.f.info(&self.t("s8"))?;
        if si.dtype != "F16" {
            return Err(format!("{}: {}.s8 has dtype {}, expected F16", self.f.path, self.name, si.dtype));
        }
        let mut q = vec![0i8; nr * nc];
        let mut s = vec![0u16; nr * (nc / g)];
        if r.side == SIDE_IN {
            if !cols.start.is_multiple_of(128) || !nc.is_multiple_of(128) {
                return Err(format!("{}: {} side-2 int8 columns {cols:?} are not on 128-input blocks", self.f.path, self.name));
            }
            let spg = 128 / g;
            self.check_shape("q8", &[r.inp / 128, r.out, 128])?;
            self.check_shape("s8", &[r.inp / 128, r.out, spg])?;
            for (bi, b) in (cols.start / 128..cols.end / 128).enumerate() {
                let qb = self.f.read(&self.t("q8"), (b * r.out * 128) as u64, (r.out * 128) as u64)?;
                let sb = u16s(&self.f.read(&self.t("s8"), (b * r.out * spg * 2) as u64, (r.out * spg * 2) as u64)?);
                for i in 0..nr {
                    let o = rows.start + i;
                    let dst_q = &mut q[i * nc + bi * 128..][..128];
                    for (d, &src_b) in dst_q.iter_mut().zip(&qb[o * 128..o * 128 + 128]) {
                        *d = src_b as i8;
                    }
                    s[i * (nc / g) + bi * spg..][..spg].copy_from_slice(&sb[o * spg..o * spg + spg]);
                }
            }
        } else {
            self.check_shape("q8", &[r.out, r.inp])?;
            self.check_shape("s8", &[r.out, r.inp / g])?;
            let qa = self.rows_of("q8", rows, r.inp, 1)?;
            let sa = u16s(&self.rows_of("s8", rows, r.inp / g, 2)?);
            for i in 0..nr {
                let dst_q = &mut q[i * nc..][..nc];
                for (d, &src_b) in dst_q.iter_mut().zip(&qa[i * r.inp + cols.start..][..nc]) {
                    *d = src_b as i8;
                }
                s[i * (nc / g)..][..nc / g].copy_from_slice(&sa[i * (r.inp / g) + cols.start / g..][..nc / g]);
            }
        }
        if q.contains(&-128) {
            return Err(format!("{}: {} int8 code -128 is not a code", self.f.path, self.name));
        }
        for &h in &s {
            let sf = f16_to_f32(h);
            if !sf.is_finite() || sf < 0.0 {
                return Err(format!("{}: {} int8 scales must be finite and >= 0", self.f.path, self.name));
            }
        }
        Ok((q, s))
    }


    /// int4 codes and fp16 scale bits of the block, as [rows][cols / 2] and [rows][cols / 32] in the HF orientation.
    /// Input 2j sits in the low nibble of byte j. Side-2 storage comes back transposed to this layout. The columns
    /// must fall on 32-input groups, or 128-input blocks on side 2.
    pub fn int4(&self, rows: &Range<usize>, cols: &Range<usize>) -> Result<(Vec<u8>, Vec<u16>), String> {
        let r = self.rec;
        if r.fmt != INT4 {
            return Err(format!("{}: {} is not int4", self.f.path, self.name));
        }
        let (nr, nc) = (rows.len(), cols.len());
        if !cols.start.is_multiple_of(32) || !nc.is_multiple_of(32) {
            return Err(format!("{}: {} int4 columns {cols:?} are not on 32-input groups", self.f.path, self.name));
        }
        let mut q = vec![0u8; nr * nc / 2];
        let mut s = vec![0u16; nr * nc / 32];
        if r.side == SIDE_IN {
            if !cols.start.is_multiple_of(128) || !nc.is_multiple_of(128) {
                return Err(format!("{}: {} side-2 int4 columns {cols:?} are not on 128-input blocks", self.f.path, self.name));
            }
            self.check_shape("q4", &[r.inp / 128, r.out, 64])?;
            self.check_shape("s4", &[r.inp / 128, r.out, 4])?;
            for (bi, b) in (cols.start / 128..cols.end / 128).enumerate() {
                let qb = self.f.read(&self.t("q4"), (b * r.out * 64) as u64, (r.out * 64) as u64)?;
                let sb = u16s(&self.f.read(&self.t("s4"), (b * r.out * 8) as u64, (r.out * 8) as u64)?);
                for i in 0..nr {
                    let o = rows.start + i;
                    q[i * nc / 2 + bi * 64..][..64].copy_from_slice(&qb[o * 64..o * 64 + 64]);
                    s[i * nc / 32 + bi * 4..][..4].copy_from_slice(&sb[o * 4..o * 4 + 4]);
                }
            }
        } else {
            self.check_shape("q4", &[r.out, r.inp / 2])?;
            self.check_shape("s4", &[r.out, r.inp / 32])?;
            let qa = self.rows_of("q4", rows, r.inp / 2, 1)?;
            let sa = u16s(&self.rows_of("s4", rows, r.inp / 32, 2)?);
            for i in 0..nr {
                q[i * nc / 2..][..nc / 2].copy_from_slice(&qa[i * r.inp / 2 + cols.start / 2..][..nc / 2]);
                s[i * nc / 32..][..nc / 32].copy_from_slice(&sa[i * r.inp / 32 + cols.start / 32..][..nc / 32]);
            }
        }
        Ok((q, s))
    }

    /// A block-FP8 matrix's E4M3 bytes of rows `rows` as stored ([rows][in]), and each row's scales: the scale of its
    /// 128-row block for every 128-input block ([rows][in / 128], f32). Weight (i, j) = e4m3(byte) x scale[i][j / 128],
    /// which is what decode() returns for it.
    pub fn fp8_rows(&self, rows: &Range<usize>) -> Result<(Vec<u8>, Vec<f32>), String> {
        let r = self.rec;
        if r.fmt != FP8_BLOCK {
            return Err(format!("{}: {} is not block FP8", self.f.path, self.name));
        }
        if rows.end > r.out || rows.is_empty() {
            return Err(format!("{}: {} rows {rows:?} are outside {}", self.f.path, self.name, r.out));
        }
        let (bo, bi) = (r.out.div_ceil(128), r.inp.div_ceil(128));
        self.check_shape("fp8", &[r.out, r.inp])?;
        self.check_shape("scale", &[bo, bi])?;
        let w = self.rows_of("fp8", rows, r.inp, 1)?;
        let s = f32s(&self.f.read_all(&self.t("scale"))?);
        let mut rs = Vec::with_capacity(rows.len() * bi);
        for i in rows.clone() {
            rs.extend_from_slice(&s[i / 128 * bi..][..bi]);
        }
        Ok((w, rs))
    }

    /// MXFP8 codes and E8M0 scales of rows `rows`, as stored ([rows][in], [rows][in / 32]).
    pub fn mxfp8(&self, rows: &Range<usize>) -> Result<(Vec<u8>, Vec<u8>), String> {
        let r = self.rec;
        if r.fmt != MXFP8 {
            return Err(format!("{}: {} is not MXFP8", self.f.path, self.name));
        }
        self.check_shape("fp8", &[r.out, r.inp])?;
        self.check_shape("e8m0", &[r.out, r.inp / 32])?;
        Ok((self.rows_of("fp8", rows, r.inp, 1)?, self.rows_of("e8m0", rows, r.inp / 32, 1)?))
    }

    /// FP8-row codes of rows `rows`, as stored ([rows][in]).
    pub fn fp8_row_codes(&self, rows: &Range<usize>) -> Result<Vec<u8>, String> {
        let r = self.rec;
        if r.fmt != FP8_ROW {
            return Err(format!("{}: {} is not FP8-row", self.f.path, self.name));
        }
        self.check_shape("fp8", &[r.out, r.inp])?;
        self.check_shape("scale", &[r.out])?;
        self.rows_of("fp8", rows, r.inp, 1)
    }

    /// FP8-row rows `rows` decoded directly to BF16 little-endian bytes ([rows][in] x 2 B) without intermediate f32/u16 Vecs.
    pub fn fp8_row_bf16_le(&self, rows: &Range<usize>) -> Result<Vec<u8>, String> {
        let r = self.rec;
        if r.fmt != FP8_ROW {
            return Err(format!("{}: {} is not FP8-row", self.f.path, self.name));
        }
        self.check_shape("fp8", &[r.out, r.inp])?;
        self.check_shape("scale", &[r.out])?;
        let w = self.rows_of("fp8", rows, r.inp, 1)?;
        let sb = self.rows_of("scale", rows, 1, 4)?;
        let mut out = vec![0u8; rows.len() * r.inp * 2];
        for (i, (row_w, sc_bytes)) in w.chunks_exact(r.inp).zip(sb.chunks_exact(4)).enumerate() {
            let s = f32::from_le_bytes(sc_bytes.try_into().unwrap());
            let dst = &mut out[i * r.inp * 2..(i + 1) * r.inp * 2];
            for (b, d) in row_w.iter().zip(dst.chunks_exact_mut(2)) {
                d.copy_from_slice(&bf16_bits(e4m3_to_f32(*b) * s).to_le_bytes());
            }
        }
        Ok(out)
    }

    /// FP8-block `kv_b` heads `h0..h0+nh` (each `head_rows` rows = `uk_rows` UK + `head_rows - uk_rows` UV) decoded
    /// directly to `(uk, uv)` BF16 little-endian bytes without intermediate f32/u16 Vecs.
    pub fn fp8_block_kv_b_bf16_le(&self, h0: usize, nh: usize, head_rows: usize, uk_rows: usize) -> Result<(Vec<u8>, Vec<u8>), String> {
        let r = self.rec;
        if r.fmt != FP8_BLOCK {
            return Err(format!("{}: {} is not FP8-block", self.f.path, self.name));
        }
        let (bo, bi) = (r.out.div_ceil(128), r.inp.div_ceil(128));
        self.check_shape("fp8", &[r.out, r.inp])?;
        self.check_shape("scale", &[bo, bi])?;
        let rows = h0 * head_rows..(h0 + nh) * head_rows;
        let w = self.rows_of("fp8", &rows, r.inp, 1)?;
        let s = f32s(&self.f.read_all(&self.t("scale"))?);
        let uv_rows = head_rows - uk_rows;
        let mut uk = vec![0u8; nh * uk_rows * r.inp * 2];
        let mut uv = vec![0u8; nh * uv_rows * r.inp * 2];
        for h in 0..nh {
            for hr in 0..head_rows {
                let i = h * head_rows + hr;
                let sb_row = (rows.start + i) / 128 * bi;
                let src = &w[i * r.inp..(i + 1) * r.inp];
                let dst = if hr < uk_rows {
                    let dr = h * uk_rows + hr;
                    &mut uk[dr * r.inp * 2..(dr + 1) * r.inp * 2]
                } else {
                    let dr = h * uv_rows + (hr - uk_rows);
                    &mut uv[dr * r.inp * 2..(dr + 1) * r.inp * 2]
                };
                for cb in 0..bi {
                    let c0 = cb * 128;
                    let c1 = (c0 + 128).min(r.inp);
                    let sc = s[sb_row + cb];
                    for (b, d) in src[c0..c1].iter().zip(dst[c0 * 2..c1 * 2].chunks_exact_mut(2)) {
                        d.copy_from_slice(&bf16_bits(e4m3_to_f32(*b) * sc).to_le_bytes());
                    }
                }
            }
        }
        Ok((uk, uv))
    }

    /// Byte range (in the file) of trellis entries `entries` and the entry size: raw tiles, no re-layout.
    pub fn trellis_entries(&self, entries: &Range<usize>) -> Result<(u64, u64), String> {
        let i = self.f.info(&self.t("trellis"))?;
        let eb = *i.shape.get(1).ok_or(format!("{}: {}.trellis is not 2-D", self.f.path, self.name))? as u64;
        if entries.end > i.shape[0] {
            return Err(format!("{}: {} has {} trellis entries, not {}", self.f.path, self.name, i.shape[0], entries.end));
        }
        Ok((i.start + entries.start as u64 * eb, eb))
    }

    fn decode_trellis(&self, rows: Range<usize>, cols: Range<usize>) -> Result<Vec<f32>, String> {
        let r = self.rec;
        let (inp, out, k) = (r.inp, r.out, r.k);
        if !(1..=8).contains(&k) {
            return Err(format!("{}: {} trellis width {k}", self.f.path, self.name));
        }
        let tb = 32 * k;
        let side_in = r.side == SIDE_IN;
        let (n_ent, ent_bytes) = if side_in { (inp / 128, out * 128 / 256 * tb) } else { (out / 128, inp * 128 / 256 * tb) };
        self.check_shape("trellis", &[n_ent, ent_bytes])?;
        self.check_shape("suh", &[inp])?;
        self.check_shape("svh", &[out])?;
        let suh: Vec<f32> = u16s(&self.f.read_all(&self.t("suh"))?).into_iter().map(f16_to_f32).collect();
        let svh: Vec<f32> = u16s(&self.f.read_all(&self.t("svh"))?).into_iter().map(f16_to_f32).collect();
        // the block's entries: HF columns on side 2 (input), HF rows otherwise (output)
        let sharded = if side_in { &cols } else { &rows };
        if !sharded.start.is_multiple_of(128) || !sharded.end.is_multiple_of(128) {
            return Err(format!("{}: {} trellis block {rows:?} x {cols:?} is not on 128-channel entries", self.f.path, self.name));
        }
        let ents = sharded.start / 128..sharded.end / 128;
        let (nr, nc) = (rows.len(), cols.len());
        let mut o = vec![0f32; nr * nc];
        let (off, eb) = self.trellis_entries(&ents)?;
        let i = self.f.info(&self.t("trellis"))?;
        let raw = self.f.read(&self.t("trellis"), off - i.start, ents.len() as u64 * eb)?;
        for (n, g) in ents.clone().enumerate() {
            let e = &raw[n * eb as usize..][..eb as usize];
            if side_in {
                // entry g: W rows (inputs) 128g.., all outputs; HF column c = W row
                let t = tiles_in_major(e, k, out);
                let mut x = had_left(&t, 128, out, &suh[128 * g..128 * g + 128]);
                had_right(&mut x, 128, out, &svh);
                for (ri, row) in rows.clone().enumerate() {
                    for c in 0..128 {
                        o[ri * nc + (128 * g + c - cols.start)] = x[c * out + row];
                    }
                }
            } else {
                // entry g: W columns (outputs) 128g.., all inputs; HF row = W column
                let t = tiles_out_major(e, k, inp);
                let mut x = had_left(&t, inp, 128, &suh);
                had_right(&mut x, inp, 128, &svh[128 * g..128 * g + 128]);
                for c in 0..128 {
                    let ri = 128 * g + c - rows.start;
                    for (ci, col) in cols.clone().enumerate() {
                        o[ri * nc + ci] = x[col * 128 + c];
                    }
                }
            }
        }
        Ok(o)
    }
}

/// `decode` split over `threads` threads along the sharded side (128-channel entries), for whole-matrix decodes.
pub fn decode_par(m: &Matrix, rows: Range<usize>, cols: Range<usize>, threads: usize) -> Result<Vec<f32>, String> {
    let along_cols = m.rec.side == SIDE_IN;
    let span = if along_cols { cols.clone() } else { rows.clone() };
    let step = if m.rec.fmt == TRELLIS || (matches!(m.rec.fmt, INT4 | INT8 | FP8_B32) && along_cols) { 128 } else { 1 };
    let units = span.len() / step;
    let threads = threads.max(1).min(units.max(1));
    if threads == 1 || !span.len().is_multiple_of(step) {
        return m.decode(rows, cols);
    }
    let per = units.div_ceil(threads) * step;
    let parts: Vec<Range<usize>> = (span.start..span.end).step_by(per).map(|a| a..(a + per).min(span.end)).collect();
    let out: Vec<Result<Vec<f32>, String>> = std::thread::scope(|s| {
        let hs: Vec<_> = parts.iter().map(|p| {
            let (r, c) = if along_cols { (rows.clone(), p.clone()) } else { (p.clone(), cols.clone()) };
            s.spawn(move || m.decode(r, c))
        }).collect();
        hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("a decode thread panicked".into()))).collect()
    });
    let (nr, nc) = (rows.len(), cols.len());
    let mut o = vec![0f32; nr * nc];
    for (p, part) in parts.iter().zip(out) {
        let part = part?;
        if along_cols {
            for i in 0..nr {
                o[i * nc + (p.start - cols.start)..][..p.len()].copy_from_slice(&part[i * p.len()..][..p.len()]);
            }
        } else {
            o[(p.start - rows.start) * nc..][..p.len() * nc].copy_from_slice(&part);
        }
    }
    Ok(o)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn number_formats() {
        for b in 0..=255u8 {
            let v = e4m3_to_f32(b);
            if !v.is_nan() {
                assert_eq!(e4m3_round(v).to_bits(), v.to_bits(), "code {b:#x}");
            }
        }
        assert_eq!(e4m3_round(500.0), 448.0);
        assert_eq!(e4m3_round(-0.0).to_bits(), (-0.0f32).to_bits());
        assert_eq!(e4m3_round(1.0625), 1.0);          // tie to even
        assert_eq!(e4m3_round(1.1875), 1.25);         // tie to even
        assert_eq!(e4m3_round(0.0009765625), 0.0);    // 2^-10: half of the smallest subnormal, ties to 0
        assert_eq!(e8m0_to_f32(127), 1.0);
        assert_eq!(e8m0_to_f32(0), f32::from_bits(1 << 22));
        assert_eq!(bf16_bits(1.0 + 1.0 / 256.0), 0x3f80);    // tie to even
        assert_eq!(bf16_bits(1.0 + 3.0 / 256.0), 0x3f82);
        for h in [0u16, 0x3c00, 0xbc00, 0x0001, 0x03ff, 0x7bff, 0x3555] {
            assert_eq!(f32_to_f16_value(f16_to_f32(h)).to_bits(), f16_to_f32(h).to_bits());
        }
    }

    #[test]
    fn perm_is_a_permutation() {
        let mut p = tile_perm().to_vec();
        p.sort_unstable();
        assert_eq!(p, (0..256).collect::<Vec<_>>());
    }
}
