//! The checkpoint's own tokenizer.json (HF `tokenizers`, the library and version the release pins),
//! plus the per-token display strings logprob entries need, built once into one arena.
//!
//! Long prompts use parallel cuts after newline before an ASCII letter, or before space after an ASCII letter.
//! For GLM's GPT-style split these are pre-token boundaries. DeepSeek first isolates digits and CJK/Kana: neither
//! split matches space, newline or ASCII letters, so these cut regions remain in the same gap. In its final split,
//! punctuation+ASCII letters and the general letter match end before the space. At newline+letter, alternative E
//! (`\s+(?!\S)`) could see a new end-of-input in the left piece, but D (`\s*[\r\n]+`) precedes E in leftmost-first
//! alternation and already consumes the whitespace run ending in newline (D-before-E). Punctuation with trailing
//! newlines also ends there; the optional prefix of a letter match excludes CR/LF. Added tokens cannot cross either
//! cut: no stripping, single-word matching, space/CR/LF, or leading ASCII letter is allowed. Only these exact shapes
//! with no normalization, truncation or padding and a successful sequential/parallel startup check enable `par_ok`.
//!
//! Prefix reuse needs a stronger boundary: a previous end-of-input is NOT a safe cut for a continuation. Cache cuts
//! are before isolated AddedToken turn delimiters, never within ordinary text or the self-overlapping DSML marker.
//! Startup rejects overlapping/containing delimiters and proper-prefix added tokens. A reused checkpoint must include
//! the WHOLE delimiter in the common byte prefix, so edits, shortened previews and partial delimiters cannot change
//! tokenization to its left. The suffix includes the delimiter. Exact full hits additionally require equal lengths.
//! Entries are immutable under one bounded LRU mutex; encoding and Rayon work always happen outside that lock.

use std::collections::VecDeque;
use std::path::Path;
use std::str::FromStr;
use std::sync::Mutex;

use serde_json::Value;
use tokenizers::Tokenizer;

/// The pre-tokenizer split pattern the parallel-encode cut points are proven for.
const GLM_PRETOKENIZER_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

// These are the serialized Regex strings, including literal CR/LF in the last pattern.
const DS_DIGITS_REGEX: &str = "\\p{N}{1,3}";
const DS_CJK_REGEX: &str = "[一-龥぀-ゟ゠-ヿ]+";
const DS_TEXT_REGEX: &str = "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+";
/// Prompts at least this long (bytes) are encoded on the blocking pool so they do not stall the reactor.
const BLOCKING_BYTES: usize = 20000;
/// Prompts at least this long (bytes) are split and encoded in parallel when `par_ok`.
const PAR_MIN_BYTES: usize = 32 << 10;
/// Target piece size for the parallel encode.
const PAR_PIECE_BYTES: usize = 16 << 10;

const CACHE_MIN_BYTES: usize = 4096;
const CACHE_MAX_ENTRIES: usize = 32;
const CACHE_MAX_BYTES: usize = 32 << 20;
const TURN_DELIMITERS: [&str; 6] =
    ["<｜User｜>", "<｜Assistant｜>", "<｜tool▁outputs▁begin｜>", "<|user|>", "<|assistant|>", "<|observation|>"];
type Checkpoint = (u32, u32, u16); // byte offset, prefix id count, delimiter byte length

struct PrefixEntry {
    text: Box<str>,
    ids: Box<[u32]>,
    checkpoints: Box<[Checkpoint]>,
}
impl PrefixEntry {
    fn bytes(&self) -> usize {
        self.text.len() + self.ids.len() * 4 + self.checkpoints.len() * 12
    }
}

#[derive(Default)]
struct PrefixCache {
    entries: VecDeque<PrefixEntry>, // least recently used first
    bytes: usize,
    #[cfg(test)]
    hits: usize,
}

#[derive(Default)]
struct PrefixHit {
    byte_off: usize,
    ids: Vec<u32>,
    checkpoints: Vec<Checkpoint>,
    full: bool,
}

impl PrefixCache {
    fn lookup(&mut self, text: &str) -> PrefixHit {
        let mut best = None;
        let mut hit = PrefixHit::default();
        for (i, entry) in self.entries.iter().enumerate().rev() {
            let common = text.as_bytes().iter().zip(entry.text.as_bytes()).take_while(|(a, b)| a == b).count();
            if common == text.len() && text.len() == entry.text.len() {
                hit = PrefixHit { byte_off: common, ids: entry.ids.to_vec(), full: true, checkpoints: Vec::new() };
                best = Some(i);
                break;
            }
            if let Some(&(off, _, _)) = entry
                .checkpoints
                .iter()
                .rev()
                .find(|&&(off, _, len)| off as usize + len as usize <= common && off as usize > hit.byte_off)
            {
                hit.byte_off = off as usize;
                // Copy only the winning entry, after the scan.
                best = Some(i);
            }
        }
        if let Some(i) = best {
            let entry = self.entries.remove(i).unwrap();
            if !hit.full {
                let cut = entry.checkpoints.partition_point(|&(off, _, _)| (off as usize) < hit.byte_off);
                let count = entry.checkpoints[cut].1 as usize;
                hit.ids = entry.ids[..count].to_vec();
                hit.checkpoints = entry.checkpoints[..cut].to_vec();
            }
            self.entries.push_back(entry);
            #[cfg(test)]
            {
                self.hits += 1;
            }
        }
        hit
    }

    fn insert(&mut self, entry: PrefixEntry) {
        let bytes = entry.bytes();
        if bytes > CACHE_MAX_BYTES {
            return;
        }
        // Concurrent encodes of the same prompt may finish together; keep just one copy.
        if let Some(i) = self.entries.iter().position(|e| e.text == entry.text) {
            self.bytes -= self.entries.remove(i).unwrap().bytes();
        }
        while self.entries.len() >= CACHE_MAX_ENTRIES || self.bytes + bytes > CACHE_MAX_BYTES {
            self.bytes -= self.entries.pop_front().unwrap().bytes();
        }
        self.bytes += bytes;
        self.entries.push_back(entry);
    }
}

fn identity_normalizer(tk: &Tokenizer) -> bool {
    tk.get_normalizer().is_none_or(|n| {
        serde_json::to_value(n).ok() == Some(serde_json::json!({"type": "Sequence", "normalizers": []}))
    })
}

/// One sequential encode with the given tokenizer.
fn encode_one(tk: &Tokenizer, text: &str, add_special: bool) -> Result<Vec<u32>, String> {
    tk.encode_fast(text, add_special).map(|e| e.get_ids().to_vec()).map_err(|e| e.to_string())
}

/// `encode_one` in pieces of about `piece` bytes, cut where `split_pieces` allows, on the rayon pool.
fn encode_pieces(tk: &Tokenizer, text: &str, piece: usize) -> Result<Vec<u32>, String> {
    use rayon::prelude::*;
    let pieces = split_pieces(text, piece);
    if pieces.len() < 2 {
        return encode_one(tk, text, false);
    }
    let parts: Vec<Result<Vec<u32>, String>> = pieces.par_iter().map(|p| encode_one(tk, p, false)).collect();
    let mut ids = Vec::with_capacity(text.len() / 3);
    for p in parts {
        ids.extend_from_slice(&p?);
    }
    Ok(ids)
}

/// Proper suffix/prefix overlap in either direction; full equality is the same isolated token, not an overlap.
fn overlaps(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    (1..a.len().min(b.len())).any(|n| a[a.len() - n..] == b[..n] || b[b.len() - n..] == a[..n])
}

pub struct Tok {
    pub tk: &'static Tokenizer,
    pub eos_token: Value,
    pub eos_token_id: Option<u32>,
    pub eos_ids: Vec<i64>,
    pub pad_token: Value,
    /// sampling defaults from generation_config.json (temperature, top_p, top_k, min_p)
    pub gen_defaults: serde_json::Map<String, Value>,
    pub vocab_size: usize,
    /// the ids of the whole tokens `<think>` and `</think>` (None without them)
    pub think_start: Option<u32>,
    pub think_end: Option<u32>,
    /// what a thinking budget puts in place of the next token: "\n" + `</think>` + "\n\n" (TensorFold's close); empty
    /// without a `</think>` token
    pub think_close: Vec<u32>,
    par_ok: bool,
    cache_ok: bool,
    delimiters: Vec<&'static str>,
    cache: Mutex<PrefixCache>,
    strs: String,
    offs: Vec<u32>,
    /// the release keeps a control-token spelling in message content literal (`literal.rs`)
    literal: Option<Lit>,
}

/// What the literal-content encode needs: the spellings to split a rendered prompt at, and the same tokenizer with
/// its added tokens removed, which is how the run between two controls is encoded (`literal.rs`).
struct Lit {
    spellings: crate::literal::Spellings,
    tk: &'static Tokenizer,
    /// the parallel cuts hold for this copy too (it only has fewer added tokens), self-checked at load
    par_ok: bool,
}

fn read_json(p: &Path) -> Result<Value, String> {
    let s = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
    serde_json::from_slice(&s).map_err(|e| format!("{}: {e}", p.display()))
}

/// The text of a `tokenizer_config.json` special-token entry: a string (GLM), or an AddedToken object whose `content` is
/// the text (DeepSeek: `{"__type": "AddedToken", "content": "<｜end▁of▁sentence｜>", ...}`).
pub fn token_text(v: &Value) -> Option<&str> {
    v.as_str().or_else(|| v.get("content")?.as_str())
}

fn release_eos<'a>(generation: &'a Value, config: &'a Value) -> Option<&'a Value> {
    generation
        .get("eos_token_id")
        .filter(|v| !v.is_null())
        .or_else(|| config.get("eos_token_id").filter(|v| !v.is_null()))
        .or_else(|| config.pointer("/text_config/eos_token_id").filter(|v| !v.is_null()))
}

#[cfg(test)]
mod metadata_tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn eos_generation_overrides_config_and_nested_fallback() {
        let config = json!({"eos_token_id": 1, "text_config": {"eos_token_id": 2}});
        let generation = json!({"eos_token_id": [3, 4]});
        assert_eq!(release_eos(&generation, &config), Some(&json!([3, 4])));
        assert_eq!(release_eos(&Value::Null, &config), Some(&json!(1)));
        assert_eq!(release_eos(&Value::Null, &json!({"text_config": {"eos_token_id": 2}})), Some(&json!(2)));
        assert_eq!(release_eos(&Value::Null, &Value::Null), None);
    }
}

impl Tok {
    /// DeepSeek closes the think block with the single `</think>` token, not GLM's "\n" + `</think>` + "\n\n".
    pub fn use_single_think_close(&mut self) {
        self.think_close = self.think_end.into_iter().collect();
    }

    /// Inkling-2 has no `<think>` tags: reasoning is a `<|content_thinking|>` block that `<|end_message|>` ends, and
    /// the answer then opens with `<|message_model|><|content_text|>`. `think_close` is that end plus the opener of a
    /// text block, so a spent thinking budget moves the reply on to its answer. It stays empty when the tokenizer has
    /// no `<|end_message|>` token.
    pub fn use_inkling_thinking(&mut self) -> Result<(), String> {
        self.think_start = self.tk.token_to_id("<|content_thinking|>");
        self.think_end = self.tk.token_to_id("<|end_message|>");
        self.think_close = match self.think_end {
            Some(_) => self.encode("<|end_message|><|message_model|><|content_text|>", false)?,
            None => Vec::new(),
        };
        Ok(())
    }

    /// Stop on the tokenizer's own `eos_token` too, after the release's `eos_token_id`. Inkling-2's config names
    /// `<|content_model_end_sampling|>` and its tokenizer `<|endoftext|>`; a reply can end on either.
    pub fn add_tokenizer_eos(&mut self) {
        if let Some(id) = self.eos_token_id {
            if !self.eos_ids.contains(&(id as i64)) {
                self.eos_ids.push(id as i64);
            }
        }
    }

    /// A DeepSeek-V4.1-Flash request that names no top_p samples with top_p 0.95 (the exlr8 release carries no
    /// generation_config.json, so it was 1.0). DeepSeek's V4.1-Flash README recommends top_p 0.95 or 1.0 and evaluates at
    /// 0.95; the 1017 README recommends 1.0. At temperature 1.0, 0.95 decoded 12.9% (code) and 6.6% (prose) faster on
    /// 4 Sparks, as more drafts are accepted, with the same sampled quality.
    /// A request's own top_p, or one in the directory's generation_config.json, still wins.
    pub fn use_ds41_sampling_defaults(&mut self) {
        self.gen_defaults.entry("top_p".to_string()).or_insert(Value::from(0.95));
    }

    pub fn load(dir: &Path) -> Result<Tok, String> {
        let tk = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| format!("tokenizer.json: {e}"))?;
        let tk: &'static Tokenizer = Box::leak(Box::new(tk));
        let cfg = read_json(&dir.join("tokenizer_config.json"))?;
        let gp = dir.join("generation_config.json");
        let gen = if gp.exists() { read_json(&gp)? } else { Value::Object(Default::default()) };
        let eos_token = cfg.get("eos_token").cloned().unwrap_or_else(|| Value::from("<|endoftext|>"));
        let eos_token_id = token_text(&eos_token).and_then(|t| tk.token_to_id(t));
        // Standard HF releases may keep EOS only in config.json (including VLM text_config).
        let model_cfg = std::fs::read(dir.join("config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
            .unwrap_or(Value::Null);
        let eos =
            release_eos(&gen, &model_cfg).cloned().unwrap_or_else(|| eos_token_id.map_or(Value::Null, Value::from));
        let eos_ids = match &eos {
            Value::Array(a) => a.iter().filter_map(Value::as_i64).collect(),
            v => v.as_i64().into_iter().collect(),
        };
        let gen_defaults = ["temperature", "top_p", "top_k", "min_p"]
            .iter()
            .filter_map(|k| gen.get(*k).map(|v| (k.to_string(), v.clone())))
            .collect();
        let vocab_size = tk.get_vocab_size(true);
        let mut t = Tok {
            tk,
            eos_token,
            eos_token_id,
            eos_ids,
            pad_token: cfg.get("pad_token").cloned().unwrap_or(Value::Null),
            gen_defaults,
            vocab_size,
            think_start: tk.token_to_id("<think>"),
            think_end: tk.token_to_id("</think>"),
            think_close: Vec::new(),
            par_ok: false,
            cache_ok: false,
            delimiters: Vec::new(),
            cache: Mutex::new(PrefixCache::default()),
            strs: String::new(),
            offs: Vec::new(),
            literal: None,
        };
        t.build_token_strs();
        t.par_ok = t.check_parallel();
        t.delimiters = t.check_cache();
        t.cache_ok = !t.delimiters.is_empty();
        t.literal = t.check_literal(&cfg)?;
        if let Some(end) = t.think_end {
            let mut close = t.encode_seq("\n", false)?;
            close.push(end);
            close.extend(t.encode_seq("\n\n", false)?);
            t.think_close = close;
        }
        Ok(t)
    }

    /// Whether the cut points of `split_pieces` are provably safe for this tokenizer, confirmed on sample text.
    fn check_parallel(&self) -> bool {
        let tk = self.tk;
        let split = |v: &Value, regex: &str| {
            v.get("type") == Some(&Value::from("Split"))
                && v.pointer("/pattern/Regex") == Some(&Value::from(regex))
                && v.get("behavior") == Some(&Value::from("Isolated"))
                && v.get("invert") == Some(&Value::from(false))
        };
        let byte_level = |v: &Value| {
            v.get("type") == Some(&Value::from("ByteLevel"))
                && v.get("use_regex") == Some(&Value::from(false))
                && v.get("add_prefix_space") == Some(&Value::from(false))
        };
        let pre = tk.get_pre_tokenizer().and_then(|p| serde_json::to_value(p).ok());
        let pre_ok = pre.as_ref().is_some_and(|p| {
            if p.get("type") != Some(&Value::from("Sequence")) {
                return false;
            }
            match p.get("pretokenizers").and_then(Value::as_array).map(Vec::as_slice) {
                Some([glm, bl]) => split(glm, GLM_PRETOKENIZER_REGEX) && byte_level(bl),
                Some([digits, cjk, text, bl]) => {
                    split(digits, DS_DIGITS_REGEX)
                        && split(cjk, DS_CJK_REGEX)
                        && split(text, DS_TEXT_REGEX)
                        && byte_level(bl)
                }
                _ => false,
            }
        });
        let shape_ok = pre_ok
            && identity_normalizer(tk)
            && tk.get_truncation().is_none()
            && tk.get_padding().is_none()
            && tk.get_added_tokens_decoder().values().all(|a| {
                !a.lstrip
                    && !a.rstrip
                    && !a.single_word
                    && !a.content.contains(['\n', '\r', ' '])
                    && !a.content.starts_with(|c: char| c.is_ascii_alphabetic())
            });
        if !shape_ok {
            log::info!("parallel encode off: tokenizer shape not covered");
            return false;
        }
        let sample = parallel_sample();
        let seq = self.tk.encode_fast(sample.as_str(), false).map(|e| e.get_ids().to_vec());
        let par = self.encode_par(&sample, false, 64);
        let ok = matches!((&seq, &par), (Ok(a), Ok(b)) if a == b);
        if !ok {
            log::warn!("parallel encode off: self-check disagrees with the sequential encode");
        }
        ok
    }

    /// AddedToken extraction isolates every checkpoint from BPE. Restrict to the covered pre-tokenizers and a
    /// deterministic BPE with ID-preserving post-processing; arbitrary per-sequence processors need their own concatenation proof.
    fn check_cache(&self) -> Vec<&'static str> {
        let tk = self.tk;
        let model = serde_json::to_value(tk.get_model()).unwrap_or(Value::Null);
        // ByteLevel post-processing changes offsets/sequence IDs only, never token IDs.
        let processor_ok = tk.get_post_processor().is_none_or(|p| {
            serde_json::to_value(p).ok().is_some_and(|v| v.get("type") == Some(&Value::from("ByteLevel")))
        });
        let added = tk.get_added_tokens_decoder();
        if !self.par_ok
            || !identity_normalizer(tk)
            || tk.get_truncation().is_some()
            || tk.get_padding().is_some()
            || !processor_ok
            || model.get("type") != Some(&Value::from("BPE"))
            || model.get("dropout").is_some_and(|v| !v.is_null() && v.as_f64() != Some(0.0))
            || added.values().any(|a| a.lstrip || a.rstrip || a.single_word || a.content.is_empty())
        {
            return Vec::new();
        }
        let delimiters: Vec<_> =
            TURN_DELIMITERS.into_iter().filter(|d| added.values().any(|a| a.content == *d)).collect();
        let tokens: Vec<_> = added.values().map(|a| a.content.as_str()).collect();
        if delimiters.is_empty()
            || tokens.iter().any(|a| tokens.iter().any(|b| a.len() < b.len() && b.starts_with(a)))
            || delimiters
                .iter()
                .any(|d| tokens.iter().any(|a| overlaps(d, a) || (d != a && (d.contains(a) || a.contains(d)))))
        {
            return Vec::new();
        }
        let mut sample = String::from("system: tools and schema\n");
        for d in &delimiters {
            sample.push_str(d);
            sample.push_str("x = 1\n\nhello world 12345 你好 カナ <think>\n");
        }
        let mut parallel = false;
        let segmented = self.encode_segments(&sample, PrefixHit::default(), &delimiters, &mut parallel);
        if !matches!((segmented, self.encode_seq(&sample, false)), (Ok((a, _)), Ok(b)) if a == b) {
            log::warn!("prefix encode cache off: checkpoint self-check disagrees");
            return Vec::new();
        }
        delimiters
    }

    /// Whether this release keeps a control-token spelling in message content literal, and what that encode needs.
    /// `tokenizer_config.json` declares the rule by naming the tokenizer class (`literal::declared`). A run between
    /// two of the template's own controls is encoded with a copy of this tokenizer that has no added tokens: added
    /// token extraction already isolates every run from BPE, so that copy gives a run holding no whole spelling the
    /// ids the stock whole-string encode gives it, and a broken spelling the ordinary BPE of its run. Post-processing
    /// that adds tokens of its own cannot run over a prompt encoded in pieces, so it leaves the rule off.
    fn check_literal(&self, tok_cfg: &Value) -> Result<Option<Lit>, String> {
        if !crate::literal::declared(tok_cfg) {
            return Ok(None);
        }
        let spellings =
            crate::literal::Spellings::new(self.tk.get_added_tokens_decoder().into_values().map(|a| a.content));
        if spellings.is_empty() || self.adds_special() {
            log::warn!(
                "literal content off: the tokenizer has no added token to protect, or its post-processing adds tokens"
            );
            return Ok(None);
        }
        let mut spec: Value = serde_json::from_str(&self.tk.to_string(false).map_err(|e| e.to_string())?)
            .map_err(|e| format!("tokenizer.json: {e}"))?;
        spec["added_tokens"] = Value::Array(Vec::new());
        let lit = Tokenizer::from_str(&spec.to_string()).map_err(|e| format!("tokenizer.json: {e}"))?;
        let lit: &'static Tokenizer = Box::leak(Box::new(lit));
        let sample = parallel_sample();
        let par_ok = self.par_ok
            && matches!((encode_one(lit, &sample, false), encode_pieces(lit, &sample, 64)), (Ok(a), Ok(b)) if a == b);
        if self.par_ok && !par_ok {
            log::warn!("literal content: parallel encode off, the copy with no added tokens disagrees with itself");
        }
        log::info!(
            "literal content on: {} added-token spellings in message content stay text (parallel encode {})",
            spellings.len(),
            if par_ok { "on" } else { "off" }
        );
        Ok(Some(Lit { spellings, tk: lit, par_ok }))
    }

    /// The spellings to protect before a chat template renders, when this release keeps them literal in content.
    pub fn literal_spellings(&self) -> Option<crate::literal::Spellings> {
        self.literal.as_ref().map(|l| l.spellings.clone())
    }

    /// The literal-content encode of a rendered chat prompt (`literal.rs`): the ids of the spellings that are still
    /// whole, which are the template's own controls, and between them the runs the marks come out of, encoded with no
    /// added tokens. `hit` carries a reused prefix and collects the turn-delimiter checkpoints, as `encode_segments`
    /// does for the stock encode. Every cut is at a control, so a checkpoint is at least as safe as it is there.
    fn encode_literal(
        &self,
        text: &str,
        mut hit: PrefixHit,
        l: &Lit,
        parallel: &mut bool,
    ) -> Result<(Vec<u32>, Vec<Checkpoint>), String> {
        let tail = &text[hit.byte_off..];
        let mut at = 0;
        for (off, len) in l.spellings.matches(tail) {
            self.push_run(&mut hit.ids, &tail[at..off], l, parallel)?;
            let spelling = &tail[off..off + len];
            let id =
                self.tk.token_to_id(spelling).ok_or_else(|| format!("no id for the control token {spelling:?}"))?;
            if self.delimiters.iter().any(|d| *d == spelling) {
                hit.checkpoints.push(((hit.byte_off + off) as u32, hit.ids.len() as u32, len as u16));
            }
            hit.ids.push(id);
            at = off + len;
        }
        self.push_run(&mut hit.ids, &tail[at..], l, parallel)?;
        debug_assert!(hit.checkpoints.windows(2).all(|w| w[0].0 < w[1].0 && w[0].1 <= w[1].1));
        Ok((hit.ids, hit.checkpoints))
    }

    /// One run between two controls: the marks out, then BPE with no added tokens, in parallel cuts when it is long.
    fn push_run(&self, ids: &mut Vec<u32>, run: &str, l: &Lit, parallel: &mut bool) -> Result<(), String> {
        let run = crate::literal::strip(run);
        if run.is_empty() {
            return Ok(());
        }
        if l.par_ok && run.len() >= PAR_MIN_BYTES {
            *parallel = true;
            ids.extend_from_slice(&encode_pieces(l.tk, &run, PAR_PIECE_BYTES)?);
        } else {
            ids.extend_from_slice(&encode_one(l.tk, &run, false)?);
        }
        Ok(())
    }

    /// Encode after a reused checkpoint, including its delimiter; offsets always come from UTF-8 string matches,
    /// never from the possibly mid-codepoint common byte prefix. No tokenizer work runs while the LRU is locked.
    fn encode_segments(
        &self,
        text: &str,
        mut hit: PrefixHit,
        delimiters: &[&str],
        parallel: &mut bool,
    ) -> Result<(Vec<u32>, Vec<Checkpoint>), String> {
        use rayon::prelude::*;
        let tail = &text[hit.byte_off..];
        let mut cuts: Vec<(usize, usize)> = delimiters
            .iter()
            .flat_map(|d| tail.match_indices(d).map(move |(off, _)| (hit.byte_off + off, d.len())))
            .collect();
        cuts.sort_unstable();
        let mut starts = vec![hit.byte_off];
        starts.extend(cuts.iter().map(|&(off, _)| off).filter(|&off| off != hit.byte_off));
        let segments: Vec<_> = starts
            .iter()
            .enumerate()
            .map(|(i, &off)| &text[off..starts.get(i + 1).copied().unwrap_or(text.len())])
            .collect();
        let parts: Vec<_> = if self.par_ok && tail.len() >= PAR_MIN_BYTES && segments.len() > 1 {
            *parallel = true;
            segments.par_iter().map(|s| self.encode_uncached(s, false, &mut false)).collect()
        } else {
            segments.iter().map(|s| self.encode_uncached(s, false, parallel)).collect()
        };
        let mut cut = 0;
        for (&off, ids) in starts.iter().zip(parts) {
            if cuts.get(cut).is_some_and(|&(p, _)| p == off) {
                hit.checkpoints.push((off as u32, hit.ids.len() as u32, cuts[cut].1 as u16));
                cut += 1;
            }
            hit.ids.extend(ids?);
        }
        debug_assert!(hit.checkpoints.windows(2).all(|w| w[0].0 < w[1].0 && w[0].1 <= w[1].1));
        Ok((hit.ids, hit.checkpoints))
    }

    /// Encode `text` in pieces of about `piece` bytes, cut where `split_pieces` allows.
    fn encode_par(&self, text: &str, add_special: bool, piece: usize) -> Result<Vec<u32>, String> {
        self.encode_par_work(text, add_special, piece, &mut false)
    }

    fn encode_par_work(
        &self,
        text: &str,
        add_special: bool,
        piece: usize,
        parallel: &mut bool,
    ) -> Result<Vec<u32>, String> {
        use rayon::prelude::*;
        let pieces = split_pieces(text, piece);
        if pieces.len() < 2 {
            return self.encode_seq(text, add_special);
        }
        *parallel = true;
        let parts: Vec<Result<Vec<u32>, String>> = pieces.par_iter().map(|p| self.encode_seq(p, false)).collect();
        let mut ids = Vec::with_capacity(text.len() / 3);
        for p in parts {
            ids.extend_from_slice(&p?);
        }
        Ok(ids)
    }

    fn encode_seq(&self, text: &str, add_special: bool) -> Result<Vec<u32>, String> {
        encode_one(self.tk, text, add_special)
    }

    /// `decode([id], skip_special_tokens=False)` for every id, kept in one string arena.
    fn build_token_strs(&mut self) {
        let mut strs = String::with_capacity(self.vocab_size * 6);
        let mut offs = Vec::with_capacity(self.vocab_size + 1);
        offs.push(0u32);
        for id in 0..self.vocab_size as u32 {
            if let Ok(s) = self.tk.decode(&[id], false) {
                strs.push_str(&s);
            }
            offs.push(strs.len() as u32);
        }
        strs.shrink_to_fit();
        self.strs = strs;
        self.offs = offs;
    }

    /// The text of one token as vLLM shows it in logprob entries (decoded, special tokens kept).
    pub fn token_str(&self, id: u32) -> std::borrow::Cow<'_, str> {
        let i = id as usize;
        if i + 1 < self.offs.len() {
            std::borrow::Cow::Borrowed(&self.strs[self.offs[i] as usize..self.offs[i + 1] as usize])
        } else {
            std::borrow::Cow::Owned(self.tk.decode(&[id], false).unwrap_or_default())
        }
    }

    pub fn encode(&self, text: &str, add_special: bool) -> Result<Vec<u32>, String> {
        self.encode_work(text, add_special, &mut false)
    }

    fn encode_work(&self, text: &str, add_special: bool, parallel: &mut bool) -> Result<Vec<u32>, String> {
        // A rendered chat prompt whose content spellings carry the literal mark takes the literal-content encode.
        // Every other text keeps the release's stock encode, a `/v1/completions` prompt included, as the release's
        // own tokenizer does for a bare string (`literal.rs`). The two agree on a text with no mark in it.
        let literal = self.literal.as_ref().filter(|_| text.contains(crate::literal::RESERVED));
        if self.cache_ok
            && (CACHE_MIN_BYTES..=CACHE_MAX_BYTES).contains(&text.len())
            && !(add_special && self.adds_special())
        {
            let hit = self.cache.lock().unwrap().lookup(text);
            if hit.full {
                return Ok(hit.ids);
            }
            let (ids, checkpoints) = match literal {
                Some(l) => self.encode_literal(text, hit, l, parallel)?,
                None => self.encode_segments(text, hit, &self.delimiters, parallel)?,
            };
            self.cache.lock().unwrap().insert(PrefixEntry {
                text: text.into(),
                ids: ids.clone().into_boxed_slice(),
                checkpoints: checkpoints.into_boxed_slice(),
            });
            return Ok(ids);
        }
        match literal {
            Some(l) => Ok(self.encode_literal(text, PrefixHit::default(), l, parallel)?.0),
            None => self.encode_uncached(text, add_special, parallel),
        }
    }

    fn encode_uncached(&self, text: &str, add_special: bool, parallel: &mut bool) -> Result<Vec<u32>, String> {
        // A post-processor that adds tokens must run once over the whole sequence.
        if self.par_ok && text.len() >= PAR_MIN_BYTES && !(add_special && self.adds_special()) {
            return self.encode_par_work(text, add_special, PAR_PIECE_BYTES, parallel);
        }
        self.encode_seq(text, add_special)
    }

    fn adds_special(&self) -> bool {
        self.tk.encode_fast("a", true).map_or(true, |e| e.len() != 1)
    }

    pub async fn aencode(&'static self, text: String, add_special: bool) -> Result<Vec<u32>, String> {
        if text.len() < BLOCKING_BYTES {
            return self.encode(&text, add_special);
        }
        let (tx, rx) = tokio::sync::oneshot::channel();
        tokio::task::spawn_blocking(move || {
            let mut parallel = false;
            let r = self.encode_work(&text, add_special, &mut parallel);
            drop(text);
            let _ = tx.send(r);
            // after the ids are on their way: hand the encode's scratch memory back to the OS (the pieces of a
            // parallel encode allocated on the rayon threads)
            if parallel {
                rayon::broadcast(|_| crate::util::release_memory(true));
            }
            crate::util::release_memory(true);
        });
        rx.await.map_err(|e| e.to_string())?
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        self.tk.decode(ids, skip_special).unwrap_or_default()
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.tk.id_to_token(id)
    }
}

/// Cut points for the parallel encode: after '\n' + ASCII letter, or at ' ' after an ASCII letter (see module docs).
fn split_pieces(text: &str, piece: usize) -> Vec<&str> {
    let b = text.as_bytes();
    let mut out = Vec::with_capacity(b.len() / piece + 1);
    let mut start = 0;
    while b.len() - start > piece + piece / 2 {
        // first safe cut at or after start + piece (p >= 1 always)
        let cut = (start + piece..b.len()).find(|&p| {
            (b[p] == b' ' && b[p - 1].is_ascii_alphabetic()) || (b[p - 1] == b'\n' && b[p].is_ascii_alphabetic())
        });
        match cut {
            Some(p) => {
                out.push(&text[start..p]);
                start = p;
            }
            None => break,
        }
    }
    out.push(&text[start..]);
    out
}

/// Text for the start-up check: every kind of character class and whitespace next to the cut points.
fn parallel_sample() -> String {
    let lines = [
        "Hello world, it's 2026; we'll see.\n",
        "  indented code() {\n",
        "\tif (x >= 10) return y;\r\n",
        "Zebra: 123456789 and 3.14159!!\n\n\n",
        "naïve café — ünïcödé 你好 🙂\n",
        "<|user|>hi</think><tool_call>f<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>\n",
        "x = [1,2,3]   \n",
        "TRAILING spaces   \n   \n",
        "/nothink\n",
        "'s 't 're 've 'm 'll 'd\n",
        "a\u{a0}b\u{2028}c\u{3000}d\n",
        "<｜User｜>\"id\": \"abc\", /path/to/file.rs\r\n",
        "<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁sep｜>你好 カナ ひらがな 123456789\n",
        "<｜end▁of▁sentence｜>\r\n\n\nNext word\n",
        "end\n",
    ];
    let mut s = String::new();
    for i in 0..400 {
        s.push_str(lines[i % lines.len()]);
        s.push_str(lines[(i * 7 + 3) % lines.len()]);
    }
    s
}

/// Text deltas for a growing id stream; holds back incomplete UTF-8 sequences (tokenizers' DecodeStream).
pub struct Detok {
    ds: tokenizers::tokenizer::DecodeStream<
        'static,
        tokenizers::ModelWrapper,
        tokenizers::NormalizerWrapper,
        tokenizers::PreTokenizerWrapper,
        tokenizers::PostProcessorWrapper,
        tokenizers::DecoderWrapper,
    >,
}

impl Detok {
    pub fn new(t: &Tok, skip_special: bool) -> Detok {
        Detok { ds: t.tk.decode_stream(skip_special) }
    }

    pub fn step(&mut self, id: u32) -> Option<String> {
        self.ds.step(id).ok().flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tokenizer directory like DeepSeek's: the special tokens in tokenizer_config.json are AddedToken objects and
    /// there is no generation_config.json.
    fn dict_form_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("g6front-tok-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let added = |id: u32, c: &str, special: bool| {
            serde_json::json!({"id": id, "content": c, "single_word": false, "lstrip": false, "rstrip": false,
                               "normalized": false, "special": special})
        };
        let tk = serde_json::json!({
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": [added(10, "<think>", false), added(11, "</think>", false), added(12, "<|eos|>", true)],
            "normalizer": null,
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
            "post_processor": null,
            "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
            "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
                      "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                      "vocab": {"a": 0, "b": 1, "c": 2, "Ġ": 3, "Ċ": 4, "ĊĊ": 5, "x": 6, "y": 7, "z": 8, "!": 9},
                      "merges": ["Ċ Ċ"]}
        });
        std::fs::write(dir.join("tokenizer.json"), tk.to_string()).unwrap();
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"eos_token": {"__type": "AddedToken", "content": "<|eos|>", "lstrip": false, "normalized": true,
                              "rstrip": false, "single_word": false},
                "pad_token": {"__type": "AddedToken", "content": "<|eos|>"}}"#,
        )
        .unwrap();
        dir
    }

    #[test]
    fn an_added_token_object_names_the_eos_token() {
        let t = Tok::load(&dict_form_dir("eos")).unwrap();
        assert_eq!(t.eos_token_id, Some(12));
        assert_eq!(t.eos_ids, vec![12]);
        assert_eq!(token_text(&t.eos_token), Some("<|eos|>"));
        assert_eq!(token_text(&Value::from("x")), Some("x"));
        assert_eq!(token_text(&Value::Null), None);
    }

    #[test]
    fn deepseek_closes_thinking_with_the_one_token() {
        let mut t = Tok::load(&dict_form_dir("close")).unwrap();
        assert_eq!(t.think_close, vec![4, 11, 5]); // GLM's close: "\n" + </think> + "\n\n"
        t.use_single_think_close();
        assert_eq!(t.think_close, vec![11]);
    }

    #[test]
    fn deepseek_samples_with_top_p_095_unless_the_directory_says_otherwise() {
        let mut t = Tok::load(&dict_form_dir("topp")).unwrap();
        assert_eq!(t.gen_defaults.get("top_p"), None); // no generation_config.json: the request default would be 1.0
        t.use_ds41_sampling_defaults();
        assert_eq!(t.gen_defaults.get("top_p"), Some(&Value::from(0.95)));
        let dir = dict_form_dir("topp-gen");
        std::fs::write(dir.join("generation_config.json"), r#"{"top_p": 0.8, "temperature": 0.6}"#).unwrap();
        let mut t = Tok::load(&dir).unwrap();
        t.use_ds41_sampling_defaults();
        assert_eq!(t.gen_defaults.get("top_p"), Some(&Value::from(0.8)));
        assert_eq!(t.gen_defaults.get("temperature"), Some(&Value::from(0.6)));
    }

    /// Parallel == sequential on random text built from every character class around the cut points.
    /// Needs the real tokenizer: GR8_TEST_TOKENIZER_DIR=<dir with tokenizer.json ...> cargo test --release
    #[test]
    fn parallel_encode_matches_sequential() {
        let Some(dir) = gr8knobs::TEST_TOKENIZER_DIR.get() else {
            eprintln!("GR8_TEST_TOKENIZER_DIR not set; skipping");
            return;
        };
        let t = Tok::load(Path::new(&dir)).unwrap();
        assert!(t.par_ok, "parallel encode disabled for this tokenizer");
        let atoms = [
            "a",
            "Z",
            "word",
            " word",
            "  ",
            " ",
            "\t",
            "\n",
            "\r\n",
            "\n\n",
            "\r",
            "1",
            "12345",
            "3.5",
            "!",
            "?!",
            ".",
            ",",
            "'s",
            "'LL",
            "\"",
            "(",
            ")",
            "{",
            "}",
            "<",
            ">",
            "é",
            "ü",
            "你好",
            "🙂",
            "\u{a0}",
            "\u{2028}",
            "\u{3000}",
            "<|user|>",
            "<|assistant|>",
            "<think>",
            "</think>",
            "/nothink",
            "[gMASK]",
            "_",
            "-",
            "#",
            "\\",
            "word ",
            "a  ",
            "x\u{a0}",
            " 's",
            "'re ",
            "ab c",
            "q   \n",
        ];
        let mut seed = 0x2545F4914F6CDD1Du64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let rounds = gr8knobs::TEST_FUZZ_ROUNDS.get() as usize;
        for round in 0..rounds {
            let mut s = String::new();
            while s.len() < 60_000 + (round % 40) * 3000 {
                s.push_str(atoms[(next() % atoms.len() as u64) as usize]);
            }
            for piece in [64, 512, 4096, 16 << 10] {
                let par = t.encode_par(&s, false, piece).unwrap();
                let seq = t.encode_seq(&s, false).unwrap();
                assert_eq!(par, seq, "round {round} piece {piece}");
            }
        }
    }

    #[test]
    fn cut_points() {
        let text = "aaaa\nbbbb\n1111\ncccc\n\ndddd eeee 2222 ffff";
        let p = split_pieces(text, 2);
        assert_eq!(p.concat(), text);
        assert!(p.len() > 4);
        for w in p.windows(2) {
            let (l, r) = (w[0].as_bytes(), w[1].as_bytes());
            let (last, first) = (l[l.len() - 1], r[0]);
            assert!(
                (first == b' ' && last.is_ascii_alphabetic()) || (last == b'\n' && first.is_ascii_alphabetic()),
                "{:?} | {:?}",
                w[0],
                w[1]
            );
        }
    }
}

#[cfg(test)]
impl Tok {
    /// Test-only views for the HTTP-level checkpoint test in api.rs.
    pub(crate) fn encode_seq_for_test(&self, text: &str) -> Vec<u32> {
        self.encode_seq(text, false).unwrap()
    }
    pub(crate) fn cache_hits_for_test(&self) -> usize {
        self.cache.lock().unwrap().hits
    }
    pub(crate) fn cache_on_for_test(&self) -> bool {
        self.par_ok && self.cache_ok
    }
    /// The literal-content encode of a text the dispatch would not send there, to check the two agree.
    pub(crate) fn encode_literal_for_test(&self, text: &str) -> Vec<u32> {
        let l = self.literal.as_ref().expect("the literal-content rule is off");
        self.encode_literal(text, PrefixHit::default(), l, &mut false).unwrap().0
    }
    /// The ids of a run as the copy with no added tokens gives them (`literal.rs`).
    pub(crate) fn encode_run_for_test(&self, text: &str) -> Vec<u32> {
        encode_one(self.literal.as_ref().expect("the literal-content rule is off").tk, text, false).unwrap()
    }
    pub(crate) fn literal_par_off_for_test(&mut self) {
        self.literal.as_mut().expect("the literal-content rule is off").par_ok = false;
    }
}

#[cfg(test)]
pub(crate) mod prefix_tests {
    use super::*;
    use serde_json::json;

    fn tokenizer(ds: bool, extra: &[&str]) -> Tokenizer {
        let mut vocab = serde_json::Map::new();
        let mut alphabet: Vec<_> = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet().into_iter().collect();
        alphabet.sort_unstable();
        for c in alphabet {
            vocab.insert(c.to_string(), json!(vocab.len()));
        }
        // Actual merges make unsafe end-of-input reuse observably wrong, rather than testing only a byte tokenizer.
        let words = ["hello", "helloworld", "value", "123", "foo", "bar", "word", "words", "ĊĊ", "Ġbar", "Ġword"];
        let mut merges = Vec::new();
        for word in words {
            let mut left = String::new();
            for c in word.chars() {
                if left.is_empty() {
                    left.push(c);
                    continue;
                }
                let joined = format!("{left}{c}");
                if !vocab.contains_key(&joined) {
                    vocab.insert(joined.clone(), json!(vocab.len()));
                    merges.push(json!([left, c.to_string()]));
                }
                left = joined;
            }
        }
        let split = |regex| json!({"type":"Split", "pattern":{"Regex":regex}, "behavior":"Isolated", "invert":false});
        let bl = json!({"type":"ByteLevel", "add_prefix_space":false, "trim_offsets":true, "use_regex":false});
        let steps = if ds {
            vec![split(DS_DIGITS_REGEX), split(DS_CJK_REGEX), split(DS_TEXT_REGEX), bl]
        } else {
            vec![split(GLM_PRETOKENIZER_REGEX), bl]
        };
        let mut tk = Tokenizer::from_bytes(
            serde_json::to_vec(&json!({
                "version":"1.0", "truncation":null, "padding":null, "added_tokens":[],
                "normalizer": if ds { json!({"type":"Sequence", "normalizers":[]}) } else { Value::Null },
                "pre_tokenizer":{"type":"Sequence", "pretokenizers":steps},
            "post_processor":{"type":"ByteLevel", "add_prefix_space":true, "trim_offsets":false, "use_regex":true},
                "decoder":{"type":"ByteLevel", "add_prefix_space":false, "trim_offsets":true, "use_regex":false},
                "model":{"type":"BPE", "dropout":null, "unk_token":null, "continuing_subword_prefix":null,
                         "end_of_word_suffix":null, "fuse_unk":false, "byte_fallback":false, "ignore_merges":false,
                         "vocab":vocab, "merges":merges}
            }))
            .unwrap(),
        )
        .unwrap();
        let specials: Vec<_> = TURN_DELIMITERS
            .iter()
            .copied()
            .chain(["｜DSML｜", "<think>", "</think>"])
            .chain(extra.iter().copied())
            .map(|s| tokenizers::AddedToken::from(s, true))
            .collect();
        tk.add_special_tokens(specials).unwrap();
        tk
    }

    fn from_tk(tk: Tokenizer) -> Tok {
        let tk = Box::leak(Box::new(tk));
        let mut t = Tok {
            tk,
            eos_token: Value::Null,
            eos_token_id: None,
            eos_ids: Vec::new(),
            pad_token: Value::Null,
            gen_defaults: Default::default(),
            vocab_size: tk.get_vocab_size(true),
            think_start: None,
            think_end: None,
            think_close: Vec::new(),
            par_ok: false,
            cache_ok: false,
            delimiters: Vec::new(),
            cache: Mutex::new(PrefixCache::default()),
            strs: String::new(),
            offs: Vec::new(),
            literal: None,
        };
        t.par_ok = t.check_parallel();
        t.delimiters = t.check_cache();
        t.cache_ok = !t.delimiters.is_empty();
        t
    }

    pub(crate) fn fixture(ds: bool) -> Tok {
        from_tk(tokenizer(ds, &[]))
    }

    /// The GLM-shaped fixture as a release that declares the literal-content rule (`literal.rs`).
    fn literal_fixture() -> Tok {
        let mut t = from_tk(tokenizer(false, &[]));
        t.literal = t.check_literal(&json!({"auto_map": {"AutoTokenizer": ["tokenization_glm5.Glm5Tokenizer"]}})).unwrap();
        assert!(t.literal.is_some(), "the rule stayed off");
        t
    }
    fn exact(t: &Tok, text: &str) {
        assert_eq!(t.encode(text, false).unwrap(), t.encode_seq(text, false).unwrap(), "{} bytes", text.len());
        let cache = t.cache.lock().unwrap();
        assert!(cache.entries.len() <= CACHE_MAX_ENTRIES && cache.bytes <= CACHE_MAX_BYTES);
        assert_eq!(cache.bytes, cache.entries.iter().map(PrefixEntry::bytes).sum::<usize>());
    }

    #[test]
    fn covered_shapes_and_checkpoint_boot_checks() {
        for ds in [false, true] {
            let t = fixture(ds);
            assert!(t.par_ok && t.cache_ok);
            assert!(!t.delimiters.contains(&"｜DSML｜"));
            assert!(t.tk.token_to_id("｜DSML｜").is_some());
            for extra in ["<｜User｜>suffix", "<｜User", "｜>overlap", "<think>suffix"] {
                let t = from_tk(tokenizer(ds, &[extra]));
                assert!(!t.cache_ok, "must reject {extra}");
            }
        }
        assert!(overlaps("｜DSML｜", "｜DSML｜"));
        let mut tk = tokenizer(true, &[]);
        tk.with_normalizer(Some(tokenizers::normalizers::Lowercase)).unwrap();
        let t = from_tk(tk);
        assert!(!t.par_ok && !t.cache_ok);
        let mut tk = tokenizer(true, &[]);
        tk.add_special_tokens([tokenizers::AddedToken::from("<strip>", true).lstrip(true)]).unwrap();
        assert!(!from_tk(tk).cache_ok);
    }

    #[test]
    fn merge_continuations_shorter_previews_and_reasoning_rewrites() {
        for ds in [false, true] {
            let t = fixture(ds);
            let base = format!("{}<｜User｜>question<｜Assistant｜>", "system schema\n".repeat(400));
            for (tail, continuation) in [
                ("x = 1\n", "\nnext"),
                ("hello", "world"),
                ("value: 12", "3"),
                ("foo ", "bar"),
                ("…a word", "s and more"),
                ("<think>\n", "more reasoning</think>"),
            ] {
                let inserted = format!("{base}{tail}");
                exact(&t, &inserted);
                exact(&t, &format!("{inserted}{continuation}"));
            }
            let old = format!("{base}…a word<｜User｜>preview");
            exact(&t, &old);
            exact(&t, &format!("{base}…a words and more")); // delimiter disappears at the edit
            exact(&t, &format!("{base}<think>\n"));
            exact(&t, &format!("{base}</think>"));
            let real = format!("{base}hello<｜User｜>next<｜Assistant｜>");
            exact(&t, &format!("{real}synthetic preview<｜User｜>future"));
            exact(&t, &real);
            exact(&t, &real[..4200]); // shorter system prefix, no delimiter
            let earlier = format!("{base}<think>old thoughts</think>answer<｜User｜>next<｜Assistant｜>");
            exact(&t, &earlier);
            exact(&t, &earlier.replace("old thoughts", "changed reasoning"));
            // Common bytes end within a delimiter or a UTF-8 character, not at a usable checkpoint.
            exact(&t, &format!("{base}abc<｜User｜>old"));
            exact(&t, &format!("{base}abc<｜Us"));
            exact(&t, &format!("{base}é"));
            exact(&t, &format!("{base}ê"));
            assert!(t.cache.lock().unwrap().hits > 0);
        }
    }

    #[test]
    fn lookup_requires_equal_full_length_or_a_whole_delimiter() {
        let t = fixture(true);
        let system = "system schema\n".repeat(400);
        let text = format!("{system}<｜User｜>hello<｜Assistant｜>tail");
        exact(&t, &text);
        let mut cache = t.cache.lock().unwrap();
        let full = cache.lookup(&text);
        assert!(full.full && full.byte_off == text.len());
        let shorter = cache.lookup(&system);
        assert!(!shorter.full && shorter.byte_off == 0 && shorter.ids.is_empty());
        let partial = cache.lookup(&format!("{system}<｜User｜>hello<｜Ass"));
        assert!(!partial.full && partial.byte_off == system.len());
        let changed = cache.lookup(&format!("{system}<｜User｜>hello<｜Assistant｜>different"));
        assert!(!changed.full);
        assert_eq!(changed.byte_off, text.find("<｜Assistant｜>").unwrap());
        assert_eq!(changed.checkpoints.len(), 1);
    }

    #[test]
    fn release_hygiene_tracks_actual_parallel_work() {
        let t = fixture(true);
        let text = format!("{}<｜User｜>hello", "some words and code\n".repeat(3000));
        let mut parallel = false;
        t.encode_work(&text, false, &mut parallel).unwrap();
        assert!(parallel);
        parallel = false;
        t.encode_work(&text, false, &mut parallel).unwrap();
        assert!(!parallel, "full cache hit must not broadcast worker memory release");
        t.encode_work(&format!("{text} world"), false, &mut parallel).unwrap();
        assert!(!parallel, "short uncached tail must not broadcast");
        let mut unsupported = fixture(true);
        unsupported.par_ok = false;
        unsupported.cache_ok = false;
        unsupported.encode_work(&text, false, &mut parallel).unwrap();
        assert!(!parallel);
        let no_cut = "1".repeat(PAR_MIN_BYTES * 2);
        t.encode_uncached(&no_cut, false, &mut parallel).unwrap();
        assert!(!parallel, "eligible length without a safe split is sequential");
    }

    #[test]
    fn twenty_turns_concurrent_preview_check_system_and_chat() {
        let t = fixture(true);
        let mut text = "system tools schema\n".repeat(300);
        let mut turns = Vec::new();
        for i in 0..20 {
            text.push_str(&format!("<｜User｜>turn {i}: {}<｜Assistant｜><think>plan {i}</think>｜DSML｜tool_call\n{}<｜tool▁outputs▁begin｜>ok",
                "a b c d ".repeat(300), "x = 123\n".repeat(300)));
            turns.push(text.clone());
        }
        let tokens = t.encode_seq(turns.last().unwrap(), false).unwrap().len();
        assert!(tokens >= 80_000);
        eprintln!("TWENTY_TURN_STRESS turns=20 lanes=4 final_tokens={tokens}");
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        std::thread::scope(|scope| {
            for lane in 0..4 {
                let (t, turns, pool) = (&t, &turns, &pool);
                scope.spawn(move || {
                    use rayon::prelude::*;
                    for text in turns {
                        let requests = [
                            format!("{text}<｜User｜>preview {lane}<｜Assistant｜><think>\n"),
                            text.clone(),
                            text[..4200].to_string(),
                            format!("{text}<｜Assistant｜></think>"),
                        ];
                        pool.install(|| requests.par_iter().for_each(|q| exact(t, q)));
                        exact(t, text);
                    }
                });
            }
        });
        assert!(t.cache.lock().unwrap().hits > 0);
    }

    #[test]
    fn lru_count_bytes_and_duplicate_insertion_are_bounded() {
        fn entry(i: usize, n: usize) -> PrefixEntry {
            PrefixEntry {
                text: format!("{i:04}{}", "x".repeat(n)).into(),
                ids: vec![1; n / 4].into(),
                checkpoints: Box::new([]),
            }
        }
        let mut cache = PrefixCache::default();
        for i in 0..40 {
            cache.insert(entry(i, 4096));
        }
        assert_eq!(cache.entries.len(), CACHE_MAX_ENTRIES);
        assert!(cache.entries.front().unwrap().text.starts_with("0008"));
        let first = cache.entries.front().unwrap().text.to_string();
        assert!(cache.lookup(&first).full);
        cache.insert(entry(40, 4096));
        assert!(cache.entries.iter().any(|e| e.text.as_ref() == first));
        for i in 0..20 {
            cache.insert(entry(i, 1 << 20));
        }
        assert!(cache.entries.len() < CACHE_MAX_ENTRIES && cache.bytes <= CACHE_MAX_BYTES);
        let n = cache.entries.len();
        cache.insert(entry(19, 1 << 20));
        assert_eq!(cache.entries.len(), n);
        cache.insert(entry(99, CACHE_MAX_BYTES));
        assert_eq!(cache.entries.len(), n);
        assert_eq!(cache.bytes, cache.entries.iter().map(PrefixEntry::bytes).sum::<usize>());
    }

    // ---- literal content (literal.rs) ---------------------------------------------------------------------------

    #[test]
    fn a_release_names_its_literal_content_tokenizer_class_to_turn_the_rule_on() {
        let t = fixture(false);
        assert!(t.check_literal(&json!({"tokenizer_class": "TokenizersBackend"})).unwrap().is_none());
        assert!(t.check_literal(&Value::Null).unwrap().is_none());
        assert!(t.literal_spellings().is_none());
        let t = literal_fixture();
        let spellings = t.literal_spellings().expect("the rule is on");
        assert!(spellings.matches("x<|user|>y") == vec![(1, 8)], "the tokenizer's own added tokens are the spellings");
    }

    #[test]
    fn a_spelling_in_content_is_text_while_the_templates_own_controls_keep_their_ids() {
        let t = literal_fixture();
        let stock = fixture(false);
        let mark = crate::literal::MARK;
        let (user, assistant) = (t.tk.token_to_id("<|user|>").unwrap(), t.tk.token_to_id("<|assistant|>").unwrap());
        let close = t.tk.token_to_id("</think>").unwrap();
        // the prompt a template renders for a user turn whose content spells a control token
        let ids = t.encode(&format!("<|user|>hello </{mark}think> word<|assistant|>"), false).unwrap();
        let want = [vec![user], t.encode_run_for_test("hello </think> word"), vec![assistant]].concat();
        assert_eq!(ids, want);
        assert!(!ids.contains(&close), "the content's spelling must not be a control id");
        // the stock whole-string encode of the same conversation makes it one
        let today = stock.encode("<|user|>hello </think> word<|assistant|>", false).unwrap();
        assert!(today.contains(&close) && today.len() < ids.len());
    }

    #[test]
    fn a_whole_run_holding_several_broken_spellings_is_encoded_in_one_piece() {
        let t = literal_fixture();
        let mark = crate::literal::MARK;
        let (user, assistant) = (t.tk.token_to_id("<|user|>").unwrap(), t.tk.token_to_id("<|assistant|>").unwrap());
        let content = "say </think> then <|user|> and <think> words";
        let marked = content.replace("</think>", &format!("</{mark}think>"));
        let marked = marked.replace("<|user|>", &format!("<{mark}|user|>"));
        let marked = marked.replace("<think>", &format!("<{mark}think>"));
        let ids = t.encode(&format!("<|user|>{marked}<|assistant|>"), false).unwrap();
        // one encode of the whole run, not one per piece between its broken spellings
        assert_eq!(ids, [vec![user], t.encode_run_for_test(content), vec![assistant]].concat());
        for spelling in ["</think>", "<think>"] {
            assert!(!ids.contains(&t.tk.token_to_id(spelling).unwrap()), "{spelling} became a control id");
        }
        // only the template's own `<|user|>` is a control id, not the one in the content
        assert_eq!(ids.iter().filter(|&&id| id == user).count(), 1);
    }

    #[test]
    fn a_callers_own_reserved_character_survives_the_round_trip() {
        let t = literal_fixture();
        let (mark, escaped) = (crate::literal::MARK, crate::literal::ESCAPED);
        let user = t.tk.token_to_id("<|user|>").unwrap();
        // the content "\u{fdd0}<|user|> \u{fdd0}\u{fdd1}", as the protect step writes it
        let rendered = format!("<|user|>{escaped}<{mark}|user|> {escaped}\u{fdd1}");
        let ids = t.encode(&rendered, false).unwrap();
        assert_eq!(ids, [vec![user], t.encode_run_for_test("\u{fdd0}<|user|> \u{fdd0}\u{fdd1}")].concat());
    }

    #[test]
    fn the_literal_and_the_stock_encode_agree_on_a_text_with_no_mark() {
        let t = literal_fixture();
        let stock = fixture(false);
        for text in [
            "hello world",
            "<|user|>hello word<|assistant|>",
            "</think>",
            "<|user|>a</think>b<|observation|>words<|assistant|>",
            "",
        ] {
            assert_eq!(t.encode(text, false).unwrap(), stock.encode(text, false).unwrap(), "stock path: {text:?}");
            assert_eq!(t.encode_literal_for_test(text), stock.encode(text, false).unwrap(), "literal path: {text:?}");
        }
    }

    #[test]
    fn a_literal_prompt_reuses_a_turn_checkpoint_with_the_same_ids() {
        let t = literal_fixture();
        assert!(t.cache_on_for_test());
        let mark = crate::literal::MARK;
        let mut text = format!("<|user|>{}", "hello word value 123 ".repeat(300));
        for turn in 0..4 {
            text.push_str(&format!(
                "<|assistant|>reply </{mark}think> {turn}<|user|>{}",
                "foo bar words hello ".repeat(300)
            ));
            let hits = t.cache_hits_for_test();
            assert_eq!(t.encode(&text, false).unwrap(), t.encode_literal_for_test(&text), "turn {turn}");
            if turn > 0 {
                assert!(t.cache_hits_for_test() > hits, "turn {turn} reused no checkpoint");
            }
        }
    }

    #[test]
    fn the_parallel_cuts_of_a_long_literal_run_match_its_sequential_encode() {
        let mut t = literal_fixture();
        let mark = crate::literal::MARK;
        let text = format!(
            "<|user|>{}<{mark}tool_call>{}<|assistant|>",
            "hello word value 123 you bar\n".repeat(3000),
            "words foo\n".repeat(3000)
        );
        assert!(text.len() > PAR_MIN_BYTES);
        let mut parallel = false;
        let fast = t
            .encode_literal(&text, PrefixHit::default(), t.literal.as_ref().unwrap(), &mut parallel)
            .unwrap()
            .0;
        assert!(parallel, "the long run was not cut");
        t.literal_par_off_for_test();
        assert_eq!(fast, t.encode_literal_for_test(&text));
    }
}
