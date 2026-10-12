//! The checkpoint's own tokenizer.json (HF `tokenizers`, the same library and version the Python front end uses),
//! plus the per-token display strings logprob entries need, built once into one arena.
//!
//! Long prompts are encoded in parallel. The text is cut only where GLM's GPT-style pre-tokenizer regex always ends
//! a pre-token: (a) after a `\n` that is followed by an ASCII letter, (b) before a space that follows an ASCII
//! letter. No alternative of the regex can match across either cut (letter runs and contractions never take a
//! following space or a preceding newline; newline/space runs never take a letter), the one lookahead (`\s+(?!\S)`)
//! cannot end there, and the scan to the right restarts at the cut either way. So encoding the pieces separately and
//! concatenating the ids is the same as encoding the whole text. `par_ok` is only set when the tokenizer has the
//! shape that argument needs (no normalizer, no truncation/padding, no added token that strips whitespace, contains
//! a newline or space, or starts with a letter) and a start-up self-check on tricky text agrees; otherwise every
//! encode is sequential.

use std::path::Path;

use serde_json::Value;
use tokenizers::Tokenizer;

/// The pre-tokenizer split pattern the parallel-encode cut points are proven for.
const GLM_PRETOKENIZER_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Prompts at least this long (bytes) are encoded on the blocking pool so they do not stall the reactor.
const BLOCKING_BYTES: usize = 20000;
/// Prompts at least this long (bytes) are split and encoded in parallel when `par_ok`.
const PAR_MIN_BYTES: usize = 32 << 10;
/// Target piece size for the parallel encode.
const PAR_PIECE_BYTES: usize = 16 << 10;

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
    strs: String,
    offs: Vec<u32>,
}

fn read_json(p: &Path) -> Result<Value, String> {
    let s = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
    serde_json::from_slice(&s).map_err(|e| format!("{}: {e}", p.display()))
}

impl Tok {
    pub fn load(dir: &Path) -> Result<Tok, String> {
        let tk = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| format!("tokenizer.json: {e}"))?;
        let tk: &'static Tokenizer = Box::leak(Box::new(tk));
        let cfg = read_json(&dir.join("tokenizer_config.json"))?;
        let gp = dir.join("generation_config.json");
        let gen = if gp.exists() { read_json(&gp)? } else { Value::Object(Default::default()) };
        let eos_token = cfg.get("eos_token").cloned().unwrap_or_else(|| Value::from("<|endoftext|>"));
        let eos_token_id = eos_token.as_str().and_then(|t| tk.token_to_id(t));
        let eos = gen.get("eos_token_id").cloned().unwrap_or_else(|| eos_token_id.map_or(Value::Null, Value::from));
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
            strs: String::new(),
            offs: Vec::new(),
        };
        t.build_token_strs();
        t.par_ok = t.check_parallel();
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
        // the cut-point argument is about this exact pre-tokenizer: GPT-style regex split, then byte-level without regex
        let pre = tk.get_pre_tokenizer().and_then(|p| serde_json::to_value(p).ok());
        let pre_ok = pre.as_ref().is_some_and(|p| {
            let steps = p.get("pretokenizers").and_then(Value::as_array);
            matches!(steps.map(Vec::as_slice), Some([split, bl])
                if split.get("type") == Some(&Value::from("Split"))
                    && split.pointer("/pattern/Regex") == Some(&Value::from(GLM_PRETOKENIZER_REGEX))
                    && split.get("behavior") == Some(&Value::from("Isolated"))
                    && split.get("invert") == Some(&Value::from(false))
                    && bl.get("type") == Some(&Value::from("ByteLevel"))
                    && bl.get("use_regex") == Some(&Value::from(false))
                    && bl.get("add_prefix_space") == Some(&Value::from(false)))
        });
        let shape_ok = pre_ok
            && tk.get_normalizer().is_none()
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

    /// Encode `text` in pieces of about `piece` bytes, cut where `split_pieces` allows.
    fn encode_par(&self, text: &str, add_special: bool, piece: usize) -> Result<Vec<u32>, String> {
        use rayon::prelude::*;
        let pieces = split_pieces(text, piece);
        if pieces.len() < 2 {
            return self.encode_seq(text, add_special);
        }
        let parts: Vec<Result<Vec<u32>, String>> = pieces.par_iter().map(|p| self.encode_seq(p, false)).collect();
        let mut ids = Vec::with_capacity(text.len() / 3);
        for p in parts {
            ids.extend_from_slice(&p?);
        }
        Ok(ids)
    }

    fn encode_seq(&self, text: &str, add_special: bool) -> Result<Vec<u32>, String> {
        self.tk.encode_fast(text, add_special).map(|e| e.get_ids().to_vec()).map_err(|e| e.to_string())
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
        // a post-processor that adds tokens (none for GLM) would have to run once over the whole sequence
        if self.par_ok && text.len() >= PAR_MIN_BYTES && !(add_special && self.adds_special()) {
            return self.encode_par(text, add_special, PAR_PIECE_BYTES);
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
            let r = self.encode(&text, add_special);
            drop(text);
            let _ = tx.send(r);
            // after the ids are on their way: hand the encode's scratch memory back to the OS (the pieces of a
            // parallel encode allocated on the rayon threads)
            rayon::broadcast(|_| crate::util::release_memory(true));
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

    /// Parallel == sequential on random text built from every character class around the cut points.
    /// Needs the real tokenizer: G6_TOKENIZER_DIR=<dir with tokenizer.json ...> cargo test --release
    #[test]
    fn parallel_encode_matches_sequential() {
        let Ok(dir) = std::env::var("G6_TOKENIZER_DIR") else {
            eprintln!("G6_TOKENIZER_DIR not set; skipping");
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
        let rounds: usize = std::env::var("G6_FUZZ_ROUNDS").ok().and_then(|r| r.parse().ok()).unwrap_or(40);
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
