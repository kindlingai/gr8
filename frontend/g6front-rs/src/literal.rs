//! Message content that spells a control token, kept literal.
//!
//! An engine renders a conversation to one string and tokenizes that string afterwards, so a control-token spelling
//! inside message content (`</think>`, `<tool_call>`, `<|user|>` ...) becomes a real control id, indistinguishable
//! from the template's own structure. A release whose tokenizer encodes such spellings as ordinary text, and the run
//! between two genuine controls as one BPE run, declares it in `tokenizer_config.json` by naming that tokenizer class
//! in `auto_map.AutoTokenizer` (`declared()`). gr8 then serves it that way, in two steps that mirror the release's
//! own tokenizer:
//!
//! * before the template renders, every added-token spelling in the caller's strings (message fields, tools,
//!   documents, object keys included) is broken by `LITERAL_MARK` after its first character (`Protect`). The rendered
//!   structure is unchanged: the template still writes its own controls, and still reads content for its
//!   inline-reasoning rule, whose `</think>` (and the `<think>` before it) stay whole for exactly that reason.
//! * the encode (`tok.rs`) splits the rendered prompt at the spellings that are still whole -- the template's own
//!   controls -- emits their ids, and encodes each run between them with a copy of the tokenizer that has no added
//!   tokens, after `strip()` takes the marks out. A broken spelling is therefore encoded with the ordinary BPE of its
//!   surrounding run, and the result is token-identical to the release's own `apply_chat_template(tokenize=True)`.
//!
//! `LITERAL_MARK` is U+FDD0 U+FDD1 (Unicode noncharacters). A U+FDD0 in a caller's string is escaped as U+FDD0 U+FDD2
//! before marking and restored by `strip()`, so every caller string is encoded as written, and a text with no U+FDD0
//! in it is never on this path: a `/v1/completions` prompt, like the release's own tokenizer on a bare string, keeps
//! the stock encode.

use serde_json::{Map, Value};
use std::borrow::Cow;

/// U+FDD0, the first character of both markers; its presence is what puts a text on the literal path.
pub const RESERVED: char = '\u{fdd0}';
/// Inserted after the first character of a spelling that must stay text.
pub const MARK: &str = "\u{fdd0}\u{fdd1}";
/// A caller's own U+FDD0, as `strip()` restores it.
pub const ESCAPED: &str = "\u{fdd0}\u{fdd2}";

/// The `auto_map.AutoTokenizer` classes whose releases declare the rule.
const DECLARED_BY: [&str; 1] = ["tokenization_glm5.Glm5Tokenizer"];

/// The template rule that infers an assistant turn's reasoning from a `</think>` in its content.
const INLINE_REASONING_RULE: &str = "split('</think>')";
const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

/// What a template writes for one media part, by part type: the text its placeholders take up in the content a
/// template's inline-reasoning rule splits.
const TEMPLATE_MEDIA: [(&str, &str); 7] = [
    ("image", "<|begin_of_image|><|image|><|end_of_image|>"),
    ("image_url", "<|begin_of_image|><|image|><|end_of_image|>"),
    ("video", "<|begin_of_video|><|video|><|end_of_video|>"),
    ("video_url", "<|begin_of_video|><|video|><|end_of_video|>"),
    ("audio", "<|begin_of_audio|><|end_of_audio|>"),
    ("audio_url", "<|begin_of_audio|><|end_of_audio|>"),
    ("input_audio", "<|begin_of_audio|><|end_of_audio|>"),
];

/// Whether `tokenizer_config.json` names a tokenizer class that keeps control-token spellings in content literal.
pub fn declared(tok_cfg: &Value) -> bool {
    let named = |v: &Value| v.as_str().is_some_and(|s| DECLARED_BY.contains(&s));
    match tok_cfg.pointer("/auto_map/AutoTokenizer") {
        Some(Value::Array(a)) => a.iter().any(named),
        Some(v) => named(v),
        None => false,
    }
}

/// A caller's string, with its own U+FDD0 escaped.
fn escape(text: &str) -> Cow<'_, str> {
    match text.contains(RESERVED) {
        true => Cow::Owned(text.replace(RESERVED, ESCAPED)),
        false => Cow::Borrowed(text),
    }
}

/// The text of a run as it is encoded: the marks out, and the callers' own U+FDD0 back. The two replacements run in
/// this order, as the release's tokenizer normalizer does, so a caller's U+FDD0 U+FDD1 survives as itself.
pub fn strip(run: &str) -> Cow<'_, str> {
    match run.contains(RESERVED) {
        true => Cow::Owned(run.replace(MARK, "").replace(ESCAPED, &RESERVED.to_string())),
        false => Cow::Borrowed(run),
    }
}

/// The added-token spellings of a tokenizer, in the order its added vocabulary matches them.
#[derive(Clone)]
pub struct Spellings {
    /// longest first, so a spelling wins over the shorter ones it contains
    all: Vec<String>,
    /// the bytes a spelling can start with: every other position in the text is skipped without a compare
    first: [bool; 256],
}

impl Default for Spellings {
    fn default() -> Spellings {
        Spellings { all: Vec::new(), first: [false; 256] }
    }
}

impl Spellings {
    /// Every added token of more than one character. A one-character token cannot be broken after its first
    /// character, so it is not protected, as the release's own tokenizer does not protect it.
    pub fn new<I: IntoIterator<Item = String>>(contents: I) -> Spellings {
        let mut all: Vec<String> = contents.into_iter().filter(|c| c.chars().count() > 1).collect();
        all.sort_unstable();
        all.dedup();
        all.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        let mut first = [false; 256];
        for s in &all {
            first[s.as_bytes()[0] as usize] = true;
        }
        Spellings { all, first }
    }

    pub fn is_empty(&self) -> bool {
        self.all.is_empty()
    }

    pub fn len(&self) -> usize {
        self.all.len()
    }

    /// The spellings in `text` as `(offset, length)`, leftmost first, the longest at one offset, none overlapping an
    /// earlier one: the matches an added vocabulary extracts. One pass, so a spelling that can overlap itself or
    /// another one is still taken leftmost-longest (`match_indices` per spelling skips a self-overlap).
    /// Only offsets a spelling's own first byte can sit at are compared, and a spelling is valid UTF-8 whose first
    /// byte is never a continuation byte, so a match can only land on a character boundary.
    pub fn matches(&self, text: &str) -> Vec<(usize, usize)> {
        let text = text.as_bytes();
        let mut out: Vec<(usize, usize)> = Vec::new();
        let mut at = 0;
        while at < text.len() {
            if !self.first[text[at] as usize] {
                at += 1;
                continue;
            }
            match self.all.iter().find(|s| text[at..].starts_with(s.as_bytes())) {
                Some(s) => {
                    out.push((at, s.len()));
                    at += s.len();
                }
                None => at += 1,
            }
        }
        out
    }
}

/// Breaks the spellings in the arguments a chat template renders.
pub struct Protect {
    spellings: Spellings,
    /// the template infers an assistant turn's reasoning from a `</think>` in its content
    inline_reasoning: bool,
}

/// Whether a chat template's source holds the inline-reasoning rule.
pub fn has_inline_reasoning_rule(source: &str) -> bool {
    source.contains(INLINE_REASONING_RULE)
}

/// The text a part of this type contributes to the content a template's inline-reasoning rule splits.
fn media_text(part: &Value) -> &'static str {
    let ty = part.get("type").and_then(Value::as_str).unwrap_or("");
    TEMPLATE_MEDIA.iter().find(|(k, _)| *k == ty).map_or("", |(_, text)| *text)
}

/// The text of a content part a template renders as text: a bare string, or a text part.
fn text_of(item: &Value) -> Option<&str> {
    match item {
        Value::String(s) => Some(s),
        Value::Object(o) if o.get("type").and_then(Value::as_str) == Some("text") => {
            o.get("text").and_then(Value::as_str)
        }
        _ => None,
    }
}

impl Protect {
    pub fn new(spellings: Spellings, inline_reasoning: bool) -> Protect {
        Protect { spellings, inline_reasoning }
    }

    /// The messages, each with its spellings broken.
    pub fn messages(&self, messages: &[Value]) -> Vec<Value> {
        messages.iter().map(|m| self.message(m)).collect()
    }

    /// Any other argument the template reads (tools, documents).
    pub fn value(&self, v: &Value) -> Value {
        match v {
            Value::String(s) => Value::String(self.text(s)),
            Value::Array(a) => Value::Array(self.list(a, None)),
            Value::Object(o) => {
                Value::Object(o.iter().map(|(k, x)| (self.text(k), self.value(x))).collect::<Map<_, _>>())
            }
            other => other.clone(),
        }
    }

    /// One caller string: escaped, then every spelling in it broken.
    fn text(&self, text: &str) -> String {
        self.mark(&escape(text))
    }

    fn mark(&self, escaped: &str) -> String {
        let cuts = self.spellings.matches(escaped);
        if cuts.is_empty() {
            return escaped.to_string();
        }
        self.insert_marks(escaped, cuts.into_iter().map(|(off, _)| self.cut(escaped, off)))
    }

    /// Where a match at `off` is broken: after its first character.
    fn cut(&self, text: &str, off: usize) -> usize {
        off + text[off..].chars().next().map_or(0, char::len_utf8)
    }

    fn insert_marks(&self, text: &str, cuts: impl Iterator<Item = usize>) -> String {
        let mut out = String::with_capacity(text.len() + 8);
        let mut last = 0;
        for cut in cuts {
            out.push_str(&text[last..cut]);
            out.push_str(MARK);
            last = cut;
        }
        out.push_str(&text[last..]);
        out
    }

    /// Break the spellings in texts the template renders back to back, including a spelling that spans two of them.
    /// `keep(spelling, position)` leaves a match whole; `position` is its offset in the escaped text of the whole run,
    /// which starts at `offset` of the content the template's inline-reasoning rule reads.
    fn mark_run(&self, texts: &[&str], keep: Option<&dyn Fn(&str, usize) -> bool>, offset: usize) -> Vec<String> {
        if texts.len() == 1 && keep.is_none() {
            return vec![self.text(texts[0])];
        }
        let escaped: Vec<Cow<'_, str>> = texts.iter().map(|t| escape(t)).collect();
        let joined: String = escaped.concat();
        let cuts: Vec<usize> = self
            .spellings
            .matches(&joined)
            .into_iter()
            .filter(|&(off, len)| keep.is_none_or(|k| !k(&joined[off..off + len], offset + off)))
            .map(|(off, _)| self.cut(&joined, off))
            .collect();
        let mut out = Vec::with_capacity(texts.len());
        let (mut start, mut index) = (0usize, 0usize);
        for text in &escaped {
            let end = start + text.len();
            let mut piece = String::with_capacity(text.len() + 8);
            let mut last = start;
            while index < cuts.len() && cuts[index] < end {
                piece.push_str(&joined[last..cuts[index]]);
                piece.push_str(MARK);
                last = cuts[index];
                index += 1;
            }
            piece.push_str(&joined[last..end]);
            out.push(piece);
            start = end;
        }
        out
    }

    /// A content list: its text parts render concatenated, its media parts as placeholders.
    fn list(&self, items: &[Value], keep: Option<&dyn Fn(&str, usize) -> bool>) -> Vec<Value> {
        let mut out = Vec::with_capacity(items.len());
        let mut run: Vec<&Value> = Vec::new();
        let mut position = 0usize;
        for item in items {
            if text_of(item).is_some() {
                run.push(item);
                continue;
            }
            self.flush_run(&mut out, &mut run, &mut position, keep);
            out.push(self.value(item));
            position += media_text(item).len();
        }
        self.flush_run(&mut out, &mut run, &mut position, keep);
        out
    }

    fn flush_run(
        &self,
        out: &mut Vec<Value>,
        run: &mut Vec<&Value>,
        position: &mut usize,
        keep: Option<&dyn Fn(&str, usize) -> bool>,
    ) {
        if run.is_empty() {
            return;
        }
        let texts: Vec<&str> = run.iter().map(|i| text_of(i).unwrap_or("")).collect();
        let broken = self.mark_run(&texts, keep, *position);
        *position += texts.iter().map(|t| escape(t).len()).sum::<usize>();
        for (item, text) in run.iter().zip(broken) {
            out.push(match item {
                Value::String(_) => Value::String(text),
                Value::Object(o) => Value::Object(
                    o.iter()
                        .map(|(k, v)| match k.as_str() {
                            "text" => (self.text(k), Value::String(text.clone())),
                            _ => (self.text(k), self.value(v)),
                        })
                        .collect::<Map<_, _>>(),
                ),
                other => (*other).clone(),
            });
        }
        run.clear();
    }

    fn message(&self, m: &Value) -> Value {
        if self.inline_reasoning {
            if let Some(kept) = self.keep_inline_reasoning(m) {
                return kept;
            }
        }
        self.value(m)
    }

    /// An assistant message whose content the template splits on `</think>`: every `</think>`, and each `<think>`
    /// before the first of them, stays whole, so the template splits exactly as it does for an unprotected content,
    /// and every other spelling in that content is broken. `None` when the rule does not apply to this message.
    fn keep_inline_reasoning(&self, m: &Value) -> Option<Value> {
        let o = m.as_object()?;
        if o.get("role").and_then(Value::as_str) != Some("assistant") {
            return None;
        }
        if o.get("reasoning_content").is_some_and(Value::is_string) {
            return None;
        }
        let content = o.get("content")?;
        let visible = match content {
            Value::String(s) => escape(s).into_owned(),
            Value::Array(a) => a
                .iter()
                .map(|p| text_of(p).map_or_else(|| Cow::Borrowed(media_text(p)), escape))
                .collect::<Vec<_>>()
                .concat(),
            _ => return None,
        };
        let first_close = visible.find(THINK_CLOSE)?;
        let keep = move |spelling: &str, position: usize| {
            spelling == THINK_CLOSE || (spelling == THINK_OPEN && position < first_close)
        };
        let broken = match content {
            Value::String(s) => Value::String(self.mark_run(&[s.as_str()], Some(&keep), 0).swap_remove(0)),
            Value::Array(a) => Value::Array(self.list(a, Some(&keep))),
            _ => return None,
        };
        let mut out = Map::new();
        for (k, v) in o {
            match k.as_str() {
                "content" => out.insert(self.text(k), broken.clone()),
                _ => out.insert(self.text(k), self.value(v)),
            };
        }
        Some(Value::Object(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn spellings() -> Spellings {
        Spellings::new(
            ["<|user|>", "<|assistant|>", "<think>", "</think>", "<tool_call>", "</tool_call>", "[gMASK]", "<sop>", "a"]
                .iter()
                .map(|s| s.to_string()),
        )
    }

    fn protect(inline: bool) -> Protect {
        Protect::new(spellings(), inline)
    }

    #[test]
    fn only_the_declared_tokenizer_classes_turn_the_rule_on() {
        assert!(declared(&json!({"auto_map": {"AutoTokenizer": ["tokenization_glm5.Glm5Tokenizer", null]}})));
        assert!(declared(&json!({"auto_map": {"AutoTokenizer": "tokenization_glm5.Glm5Tokenizer"}})));
        assert!(!declared(&json!({"auto_map": {"AutoTokenizer": "tokenization_glm4.Glm4Tokenizer"}})));
        assert!(!declared(&json!({"auto_map": {"AutoProcessor": "tokenization_glm5.Glm5Tokenizer"}})));
        assert!(!declared(&json!({"tokenizer_class": "TokenizersBackend"})));
        assert!(!declared(&Value::Null));
    }

    #[test]
    fn matches_are_leftmost_longest_and_do_not_overlap() {
        let s = spellings();
        // the one-character token is not a spelling, and `</tool_call>` wins over `</think>`-length prefixes
        assert_eq!(s.matches("a<tool_call>b</tool_call>"), vec![(1, 11), (13, 12)]);
        // at one offset the longest wins: `</think>` over nothing, `<think>` not inside `</think>`
        assert_eq!(s.matches("x</think>y<think>"), vec![(1, 8), (10, 7)]);
        assert_eq!(s.matches("nothing here"), vec![]);
    }

    /// A spelling that can overlap itself, or whose tail starts another spelling, is still taken leftmost-longest.
    /// A per-spelling `match_indices` pass skips the second occurrence in `xababab` and misses `abab` at 3.
    #[test]
    fn a_spelling_that_overlaps_itself_is_still_found() {
        let s = Spellings::new(["xa".to_string(), "abab".to_string()]);
        assert_eq!(s.matches("xababab"), vec![(0, 2), (3, 4)]);
        // and the longest still wins where both can start
        let s = Spellings::new(["ab".to_string(), "abab".to_string()]);
        assert_eq!(s.matches("ababab"), vec![(0, 4), (4, 2)]);
        // a multi-byte spelling lands only on a character boundary
        let s = Spellings::new(["\u{4f60}\u{597d}".to_string()]);
        assert_eq!(s.matches("x\u{4f60}\u{597d}y"), vec![(1, 6)]);
    }

    #[test]
    fn a_spelling_in_content_is_broken_after_its_first_character() {
        let p = protect(false);
        assert_eq!(p.text("say </think> now"), format!("say <{MARK}/think> now"));
        assert_eq!(p.text("plain text"), "plain text");
        // a caller's own U+FDD0 is escaped, and strip() puts it back
        let marked = p.text("keep \u{fdd0} and </think>");
        assert_eq!(marked, format!("keep {ESCAPED} and <{MARK}/think>"));
        assert_eq!(strip(&marked), "keep \u{fdd0} and </think>");
        // a caller's own U+FDD0 U+FDD1 is not a mark
        assert_eq!(strip(&p.text("\u{fdd0}\u{fdd1}")), "\u{fdd0}\u{fdd1}");
    }

    #[test]
    fn strip_leaves_a_text_with_no_reserved_character_alone() {
        assert!(matches!(strip("plain </think>"), Cow::Borrowed("plain </think>")));
        assert_eq!(strip(&format!("<{MARK}|user|>")), "<|user|>");
    }

    #[test]
    fn a_spelling_that_spans_two_text_parts_is_broken() {
        let p = protect(false);
        let out = p.value(&json!([{"type": "text", "text": "ends in <too"}, {"type": "text", "text": "l_call> here"}]));
        let parts: Vec<&str> = out.as_array().unwrap().iter().map(|v| v["text"].as_str().unwrap()).collect();
        assert_eq!(parts, vec![format!("ends in <{MARK}too"), "l_call> here".to_string()]);
        // and the concatenation the template renders holds no whole spelling
        assert_eq!(spellings().matches(&parts.concat()), vec![]);
    }

    #[test]
    fn object_keys_and_nested_values_are_broken() {
        let p = protect(false);
        let out = p.value(&json!({"<tool_call>": {"name": "f", "args": ["<|user|>", 7]}}));
        let key = out.as_object().unwrap().keys().next().unwrap().clone();
        assert_eq!(key, format!("<{MARK}tool_call>"));
        assert_eq!(out[&key]["args"][0], Value::from(format!("<{MARK}|user|>")));
        assert_eq!(out[&key]["args"][1], Value::from(7));
        assert_eq!(out[&key]["name"], Value::from("f"));
    }

    #[test]
    fn the_inline_reasoning_rule_keeps_the_tags_it_splits_on() {
        let p = protect(true);
        let m = json!({"role": "assistant", "content": "<think>why <|user|> </think>answer <tool_call> </think>!"});
        let out = p.message(&m);
        assert_eq!(
            out["content"],
            Value::from(format!("<think>why <{MARK}|user|> </think>answer <{MARK}tool_call> </think>!"))
        );
        // a <think> after the first </think> is broken
        let m = json!({"role": "assistant", "content": "a </think> b <think> c"});
        assert_eq!(p.message(&m)["content"], Value::from(format!("a </think> b <{MARK}think> c")));
        // no </think> in the content, or a reasoning_content of its own: every spelling is broken
        let m = json!({"role": "assistant", "content": "<think> only"});
        assert_eq!(p.message(&m)["content"], Value::from(format!("<{MARK}think> only")));
        let m = json!({"role": "assistant", "reasoning_content": "r", "content": "a </think> b"});
        assert_eq!(p.message(&m)["content"], Value::from(format!("a <{MARK}/think> b")));
        // a user message is not an assistant message
        let m = json!({"role": "user", "content": "a </think> b"});
        assert_eq!(p.message(&m)["content"], Value::from(format!("a <{MARK}/think> b")));
        // the template without the rule breaks them all
        let m = json!({"role": "assistant", "content": "a </think> b"});
        assert_eq!(protect(false).message(&m)["content"], Value::from(format!("a <{MARK}/think> b")));
    }

    #[test]
    fn the_inline_reasoning_rule_counts_a_media_placeholder_in_the_content_it_reads() {
        let p = protect(true);
        // the image part renders as its placeholder, so the `<think>` after it is past the first `</think>`
        let m = json!({"role": "assistant", "content": [
            {"type": "text", "text": "a </think>"},
            {"type": "image_url", "image_url": {"url": "u"}},
            {"type": "text", "text": "<think> b"}]});
        let out = p.message(&m);
        assert_eq!(out["content"][0]["text"], Value::from("a </think>"));
        assert_eq!(out["content"][1]["type"], Value::from("image_url"));
        assert_eq!(out["content"][2]["text"], Value::from(format!("<{MARK}think> b")));
    }

    #[test]
    fn message_keys_other_than_content_keep_their_order_and_values() {
        let p = protect(true);
        let m = json!({"role": "assistant", "content": "x </think> y", "tool_calls": [{"function": {"name": "<|user|>"}}]});
        let out = p.message(&m);
        let keys: Vec<&String> = out.as_object().unwrap().keys().collect();
        assert_eq!(keys, vec!["role", "content", "tool_calls"]);
        assert_eq!(out["tool_calls"][0]["function"]["name"], Value::from(format!("<{MARK}|user|>")));
    }
}
