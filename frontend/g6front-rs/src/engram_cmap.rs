//! DeepSeek-V4.1-Flash Engram compressed token map generator (`ds41.engram.cmap`).
//!
//! Port of `build_compressed_token_map` from the official DeepSeek-V4.1-Flash-1017 reference
//! (`inference/engram.py`), using the same Rust `tokenizers` backend and normalizer pipeline so
//! `gr8` can generate and verify `PREFIX.engram.cmap` on first boot from `<release>/tokenizer.json`
//! without PyTorch or Python dependencies.
//!
//! Binary layout (`engine/engram/engram_boot.h`):
//!   "G6ECMAP1" | u32 vocab | u32 cvocab | u32 pad_id | u32 reserved(0) | u32 map[vocab]   (LE)

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;

use sha2::{Digest, Sha256};
use tokenizers::normalizers::{Lowercase, NormalizerWrapper, Replace, Sequence, Strip, StripAccents, NFD, NFKC};
use tokenizers::{NormalizedString, Normalizer, Tokenizer};

/// Constants matching `engine/engram/hash_consts.h` (checked by unit test `constants_match_engine_header`).
pub const VOCAB: u32 = 129_280;
pub const CVOCAB: u32 = 99_092;
pub const PAD_TOK: u32 = 2;
pub const PAD_ID: u32 = 2;
pub const CMAP_SHA256: &str = "46ed86d2ee83d4b764e4390d38393a373886efdf2baeaa71a5370090d85b76de";

const MAGIC: &[u8; 8] = b"G6ECMAP1";
const USAGE: &str = "usage: g6front-rs engram-cmap <tokenizer.json> <out.cmap>";

/// Builds the exact normalizer sequence from DeepSeek-V4.1-Flash `inference/engram.py`.
fn build_normalizer() -> Result<Sequence, String> {
    let sentinel = "\u{e000}";
    let r_ws = Replace::new(tokenizers::normalizers::replace::ReplacePattern::Regex(r"[ \t\r\n]+".into()), " ")
        .map_err(|e| format!("engram normalizer ws regex: {e}"))?;
    let r_single = Replace::new(tokenizers::normalizers::replace::ReplacePattern::Regex(r"^ $".into()), sentinel)
        .map_err(|e| format!("engram normalizer space regex: {e}"))?;
    let r_restore = Replace::new(sentinel, " ").map_err(|e| format!("engram normalizer restore: {e}"))?;
    Ok(Sequence::new(vec![
        NormalizerWrapper::NFKC(NFKC),
        NormalizerWrapper::NFD(NFD),
        NormalizerWrapper::StripAccents(StripAccents),
        NormalizerWrapper::Lowercase(Lowercase),
        NormalizerWrapper::Replace(r_ws),
        NormalizerWrapper::Replace(r_single),
        NormalizerWrapper::StripNormalizer(Strip::new(true, true)),
        NormalizerWrapper::Replace(r_restore),
    ]))
}

/// Computes `(lookup, cvocab)` for a loaded `Tokenizer`, matching `build_compressed_token_map`.
pub fn build_compressed_token_map(tok: &Tokenizer) -> Result<(Vec<u32>, u32), String> {
    let normalizer = build_normalizer()?;
    let n_vocab = tok.get_vocab_size(true);
    let mut key_to_new: HashMap<String, u32> = HashMap::with_capacity(n_vocab);
    let mut lookup: Vec<u32> = Vec::with_capacity(n_vocab);

    for token_id in 0..n_vocab {
        let tid = token_id as u32;
        let text = tok.decode(&[tid], false).map_err(|e| format!("decode token {tid}: {e}"))?;
        let key = if text.contains('\u{fffd}') {
            // Partial UTF-8 byte token: key by raw token string, matching `backend.id_to_token(token_id)`.
            tok.id_to_token(tid).unwrap_or_default()
        } else {
            let mut ns = NormalizedString::from(text.as_str());
            normalizer.normalize(&mut ns).map_err(|e| format!("normalize token {tid}: {e}"))?;
            let normalized = ns.get();
            if normalized.is_empty() {
                text
            } else {
                normalized.to_owned()
            }
        };
        let next_id = key_to_new.len() as u32;
        let new_id = *key_to_new.entry(key).or_insert(next_id);
        lookup.push(new_id);
    }

    Ok((lookup, key_to_new.len() as u32))
}

/// Packs the `G6ECMAP1` binary body and returns `(bytes, hex_sha256)`.
pub fn pack_cmap(lookup: &[u32], cvocab: u32, pad_id: u32) -> (Vec<u8>, String) {
    let vocab = lookup.len() as u32;
    let mut body = Vec::with_capacity(24 + lookup.len() * 4);
    body.extend_from_slice(MAGIC);
    body.extend_from_slice(&vocab.to_le_bytes());
    body.extend_from_slice(&cvocab.to_le_bytes());
    body.extend_from_slice(&pad_id.to_le_bytes());
    body.extend_from_slice(&0u32.to_le_bytes());
    for &id in lookup {
        body.extend_from_slice(&id.to_le_bytes());
    }
    let digest = Sha256::digest(&body);
    let mut hex = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut hex, "{b:02x}");
    }
    (body, hex)
}

/// Generates `ds41.engram.cmap` from `tok_path`, verifies it against `hash_consts.h` (`CMAP_SHA256`),
/// and writes it atomically to `out_path`.
pub fn generate_verified(tok_path: &Path, out_path: &Path) -> Result<String, String> {
    let tok = Tokenizer::from_file(tok_path)
        .map_err(|e| format!("{}: cannot load tokenizer: {e}", tok_path.display()))?;
    let (lookup, cvocab) = build_compressed_token_map(&tok)?;
    let vocab = lookup.len() as u32;
    if vocab != VOCAB {
        return Err(format!(
            "{}: vocab size {vocab} != expected {VOCAB} (not the DeepSeek-V4.1-Flash-1017 tokenizer)",
            tok_path.display()
        ));
    }
    if cvocab != CVOCAB {
        return Err(format!(
            "{}: compressed vocab size {cvocab} != expected {CVOCAB}",
            tok_path.display()
        ));
    }
    let pad_id = lookup[PAD_TOK as usize];
    if pad_id != PAD_ID {
        return Err(format!(
            "{}: compressed pad id {pad_id} != expected {PAD_ID}",
            tok_path.display()
        ));
    }
    let (body, sha) = pack_cmap(&lookup, cvocab, pad_id);
    if sha != CMAP_SHA256 {
        return Err(format!(
            "{}: generated cmap sha256 {sha} != expected {CMAP_SHA256} (engine/engram/hash_consts.h)",
            tok_path.display()
        ));
    }
    write_atomic(out_path, &body)?;
    Ok(sha)
}

fn write_atomic(out_path: &Path, data: &[u8]) -> Result<(), String> {
    if let Some(parent) = out_path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .map_err(|e| format!("{}: cannot create directory: {e}", parent.display()))?;
    }
    let pid = std::process::id();
    let file_name = out_path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| format!("{}: invalid output file name", out_path.display()))?;
    let tmp_path = out_path.with_file_name(format!(".{file_name}.tmp.{pid}"));
    let write_res = (|| -> std::io::Result<()> {
        let mut f = fs::File::create(&tmp_path)?;
        f.write_all(data)?;
        f.sync_all()?;
        Ok(())
    })();
    if let Err(e) = write_res {
        let _ = fs::remove_file(&tmp_path);
        return Err(format!("{}: write failed: {e}", tmp_path.display()));
    }
    if let Err(e) = fs::rename(&tmp_path, out_path) {
        let _ = fs::remove_file(&tmp_path);
        return Err(format!("{}: rename failed: {e}", out_path.display()));
    }
    Ok(())
}

/// CLI entrypoint for `g6front-rs engram-cmap ...`. Returns process exit code.
pub fn run_cli(args: &[String]) -> i32 {
    if args.len() == 1 && (args[0] == "-h" || args[0] == "--help") {
        println!("{USAGE}");
        return 0;
    }
    if args.len() != 2 || args[0].starts_with('-') || args[1].starts_with('-') {
        eprintln!("{USAGE}");
        return 2;
    }
    let tok_path = Path::new(&args[0]);
    let out_path = Path::new(&args[1]);
    match generate_verified(tok_path, out_path) {
        Ok(sha) => {
            println!("cmap: {VOCAB} ids -> {CVOCAB} compressed, pad {PAD_ID}, sha256 {sha}");
            0
        }
        Err(e) => {
            eprintln!("engram-cmap: {e}");
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::models::bpe::BPE;

    #[test]
    fn constants_match_engine_header() {
        let header = include_str!("../../../engine/engram/hash_consts.h");
        assert!(
            header.contains(&format!(
                "constexpr uint32_t kVocab = {VOCAB}, kCVocab = {CVOCAB}, kPadId = {PAD_ID}, kPadTok = {PAD_TOK};"
            )),
            "vocab/cvocab/pad constants diverged from engine/engram/hash_consts.h"
        );
        assert!(
            header.contains(&format!("constexpr const char* kCmapSha256 = \"{CMAP_SHA256}\";")),
            "CMAP_SHA256 diverged from engine/engram/hash_consts.h"
        );
    }

    #[test]
    fn normalizer_collapses_case_accents_whitespace_and_preserves_single_space() {
        let n = build_normalizer().unwrap();
        let norm = |s: &str| {
            let mut ns = NormalizedString::from(s);
            n.normalize(&mut ns).unwrap();
            ns.get().to_owned()
        };
        assert_eq!(norm(" The"), "the");
        assert_eq!(norm("THE"), "the");
        assert_eq!(norm("caf\u{00e9}"), "cafe");
        assert_eq!(norm("  a\t\nb  "), "a b");
        // Single space must survive Strip() via the private-use sentinel.
        assert_eq!(norm(" "), " ");
        assert_eq!(norm("\t\n"), " ");
    }

    #[test]
    fn pack_cmap_header_and_layout() {
        let (body, sha) = pack_cmap(&[0, 1, 2, 1], 3, 2);
        assert_eq!(body.len(), 24 + 4 * 4);
        assert_eq!(&body[0..8], b"G6ECMAP1");
        assert_eq!(u32::from_le_bytes(body[8..12].try_into().unwrap()), 4);
        assert_eq!(u32::from_le_bytes(body[12..16].try_into().unwrap()), 3);
        assert_eq!(u32::from_le_bytes(body[16..20].try_into().unwrap()), 2);
        assert_eq!(u32::from_le_bytes(body[20..24].try_into().unwrap()), 0);
        assert_eq!(u32::from_le_bytes(body[36..40].try_into().unwrap()), 1);
        assert_eq!(sha.len(), 64);
    }

    #[test]
    fn small_bpe_tokenizer_collapses_equivalent_tokens_and_rejects_wrong_vocab() {
        let vocab = [
            ("<pad>".into(), 0u32),
            ("the".into(), 1u32),
            ("THE".into(), 2u32),
            (" The".into(), 3u32),
            ("caf\u{00e9}".into(), 4u32),
            ("cafe".into(), 5u32),
            (" ".into(), 6u32),
        ];
        let bpe = BPE::builder().vocab_and_merges(vocab, vec![]).build().unwrap();
        let tok = Tokenizer::new(bpe);
        let (lookup, cvocab) = build_compressed_token_map(&tok).unwrap();
        assert_eq!(lookup.len(), 7);
        assert_eq!(lookup[1], lookup[2]);
        assert_eq!(lookup[1], lookup[3]);
        assert_eq!(lookup[4], lookup[5]);
        assert_ne!(lookup[6], lookup[0]);
        assert_eq!(cvocab, 4);

        let dir = std::env::temp_dir().join(format!("gr8-engram-cmap-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let tok_path = dir.join("tokenizer.json");
        let out_path = dir.join("nested").join("ds41.engram.cmap");
        tok.save(&tok_path, false).unwrap();
        let err = generate_verified(&tok_path, &out_path).unwrap_err();
        assert!(err.contains("vocab size 7 != expected 129280"), "{err}");
        assert!(!out_path.exists(), "failed generation must not write output file");
        let _ = fs::remove_dir_all(&dir);
    }
}
