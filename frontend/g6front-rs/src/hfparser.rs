//! Generic HuggingFace reply parser (`--dialect hf`): optional `<think>...</think>` reasoning block + standard JSON
//! `<tool_call>...</tool_call>` tool invocations (Qwen, Hermes, Llama and compatible chat templates).
//!
//! Rules:
//! - Default mode (`HfParser::new`): an optional `<think>...</think>` block at the start of the reply (allowing
//!   leading whitespace before `<think>`) is routed to `reasoning`; everything after `</think>` — or from the start
//!   when the reply does not open `<think>` — is `content` (and tool calls).
//! - `content_first()`: the prompt already closed `<think>`, so the reply starts in `content`.
//! - `in_think()`: the prompt left `<think>` open, so the reply starts inside `reasoning` up to `</think>`.
//! - In `content`, when `tools` are offered and `tool_choice` is not `"none"`, each `<tool_call>...</tool_call>` block
//!   is parsed as a JSON object `{"name": "...", "arguments": ...}` (`"parameters"` is accepted as an alias for
//!   `"arguments"`, and object or string arguments are normalized to an OpenAI JSON string).
//! - Declared tool names (or plausible identifiers when `emit_undeclared` is true) become OpenAI tool calls; undeclared
//!   names when `emit_undeclared` is false, malformed JSON, or unclosed `<tool_call>` text at the end of the reply fall
//!   back to `content` without dropping bytes.

use std::collections::HashSet;

use serde_json::Value;

use crate::parser::{plausible_name, tool_call_id, Delta, FullParse, ToolDelta};
use crate::pyjson::{self, is_py_space, py_rstrip, py_strip};

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const TOOL_OPEN: &str = "<tool_call>";
const TOOL_CLOSE: &str = "</tool_call>";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    /// Start of reply: optional `<think>` opens reasoning; anything else enters `Content`.
    Start,
    /// Prompt opened `<think>`: inside reasoning, swallowing one optional redundant leading `<think>`.
    InThinkStart,
    /// Inside `<think>...</think>` up to `</think>`.
    Reasoning,
    /// Answer text and `<tool_call>...</tool_call>` blocks.
    Content,
    /// After `<tool_call>`, collecting JSON up to `</tool_call>`.
    ToolCall,
}

enum CallOutcome {
    Call { name: String, arguments: String },
    RawText,
}

pub struct HfParser {
    tools: HashSet<String>,
    parse_tools: bool,
    emit_undeclared: bool,
    pub include_reasoning: bool,
    mode: Mode,
    buf: String,
    reasoning_ended: bool,
    deferred_reasoning: String,
    deferred_content: String,
    post_call_ws: String,
    content_has_nonws: bool,
    after_valid_call: bool,
    calls: Vec<(String, String, String)>,
}

impl HfParser {
    pub fn new(tools: Option<&Value>, tool_choice: &Value, include_reasoning: bool) -> HfParser {
        let mut names = HashSet::new();
        let mut any = false;
        if let Some(Value::Array(ts)) = tools {
            any = !ts.is_empty();
            for t in ts {
                if let Some(n) = t.get("function").and_then(|f| f.get("name")).and_then(Value::as_str) {
                    names.insert(n.to_string());
                }
            }
        }
        HfParser {
            tools: names,
            parse_tools: any && tool_choice.as_str() != Some("none"),
            emit_undeclared: false,
            include_reasoning,
            mode: Mode::Start,
            buf: String::new(),
            reasoning_ended: false,
            deferred_reasoning: String::new(),
            deferred_content: String::new(),
            post_call_ws: String::new(),
            content_has_nonws: false,
            after_valid_call: false,
            calls: Vec::new(),
        }
    }

    /// The prompt closed the think block (or thinking was turned off): the reply starts directly on content.
    pub fn content_first(mut self) -> HfParser {
        self.mode = Mode::Content;
        self.reasoning_ended = true;
        self
    }

    /// The prompt left `<think>` open: the reply starts inside reasoning up to `</think>`.
    pub fn in_think(mut self) -> HfParser {
        self.mode = Mode::InThinkStart;
        self
    }

    /// `--undeclared-tool-calls emit`: a call to an undeclared tool whose name is a plausible identifier is emitted as
    /// a tool call rather than kept as content text.
    pub fn emit_undeclared(mut self, on: bool) -> HfParser {
        self.emit_undeclared = on;
        self
    }

    fn emit_content(&mut self, piece: &str, out: &mut String) {
        if piece.is_empty() {
            return;
        }
        if !self.content_has_nonws {
            if piece.chars().all(is_py_space) {
                if self.calls.is_empty() {
                    self.deferred_content.push_str(piece);
                }
                return;
            }
            self.content_has_nonws = true;
            self.after_valid_call = false;
            if self.calls.is_empty() {
                out.push_str(&std::mem::take(&mut self.deferred_content));
                out.push_str(piece);
            } else {
                self.deferred_content.clear();
                out.push_str(piece.trim_start_matches(is_py_space));
            }
            return;
        }
        if self.after_valid_call {
            if piece.chars().all(is_py_space) {
                self.post_call_ws.push_str(piece);
                return;
            }
            self.after_valid_call = false;
            out.push_str(&std::mem::take(&mut self.post_call_ws));
            out.push_str(piece);
            return;
        }
        out.push_str(piece);
    }

    fn feed(&mut self, text: &str, finished: bool) -> (String, String, Vec<ToolDelta>) {
        let mut raw_reasoning = String::new();
        let mut out_content = String::new();
        let mut out_tools = Vec::new();
        self.buf.push_str(text);

        loop {
            match self.mode {
                Mode::Start => {
                    let ws = self.buf.len() - self.buf.trim_start_matches(is_py_space).len();
                    let rest = &self.buf[ws..];
                    if rest.is_empty() {
                        if finished {
                            self.mode = Mode::Content;
                            continue;
                        }
                        break;
                    }
                    if rest.starts_with(THINK_OPEN) {
                        self.buf.drain(..ws + THINK_OPEN.len());
                        self.mode = Mode::Reasoning;
                        continue;
                    }
                    if !finished && THINK_OPEN.starts_with(rest) {
                        break;
                    }
                    self.mode = Mode::Content;
                }
                Mode::InThinkStart => {
                    let ws = self.buf.len() - self.buf.trim_start_matches(is_py_space).len();
                    let rest = &self.buf[ws..];
                    if rest.is_empty() {
                        if finished {
                            self.mode = Mode::Reasoning;
                            continue;
                        }
                        break;
                    }
                    if rest.starts_with(THINK_OPEN) {
                        self.buf.drain(..ws + THINK_OPEN.len());
                        self.mode = Mode::Reasoning;
                        continue;
                    }
                    if !finished && THINK_OPEN.starts_with(rest) {
                        break;
                    }
                    self.mode = Mode::Reasoning;
                }
                Mode::Reasoning => {
                    if let Some(p) = self.buf.find(THINK_CLOSE) {
                        raw_reasoning.push_str(&self.buf[..p]);
                        self.buf.drain(..p + THINK_CLOSE.len());
                        self.reasoning_ended = true;
                        self.mode = Mode::Content;
                        continue;
                    }
                    let keep = if finished { 0 } else { partial_tag(&self.buf, THINK_CLOSE) };
                    let n = self.buf.len() - keep;
                    raw_reasoning.push_str(&self.buf[..n]);
                    self.buf.drain(..n);
                    break;
                }
                Mode::Content => {
                    if !self.parse_tools {
                        if !self.buf.is_empty() {
                            let rest = std::mem::take(&mut self.buf);
                            self.emit_content(&rest, &mut out_content);
                        }
                        break;
                    }
                    if let Some(p) = self.buf.find(TOOL_OPEN) {
                        if p > 0 {
                            let pre: String = self.buf.drain(..p).collect();
                            self.emit_content(&pre, &mut out_content);
                        }
                        self.buf.drain(..TOOL_OPEN.len());
                        self.mode = Mode::ToolCall;
                        continue;
                    }
                    let keep = if finished { 0 } else { partial_tag(&self.buf, TOOL_OPEN) };
                    let n = self.buf.len() - keep;
                    if n > 0 {
                        let pre: String = self.buf.drain(..n).collect();
                        self.emit_content(&pre, &mut out_content);
                    }
                    break;
                }
                Mode::ToolCall => {
                    if let Some(end) = find_tool_end(&self.buf, finished) {
                        let raw_inner: String = self.buf.drain(..end).collect();
                        self.buf.drain(..TOOL_CLOSE.len());
                        self.mode = Mode::Content;
                        match parse_tool_json(&raw_inner, &self.tools, self.emit_undeclared) {
                            CallOutcome::Call { name, arguments } => {
                                self.deferred_content.clear();
                                self.post_call_ws.clear();
                                self.after_valid_call = true;
                                let index = self.calls.len();
                                let id = tool_call_id();
                                self.calls.push((id.clone(), name.clone(), arguments.clone()));
                                out_tools.push(ToolDelta {
                                    index,
                                    id: Some(id),
                                    name: Some(name),
                                    arguments: Some(arguments),
                                });
                            }
                            CallOutcome::RawText => {
                                let raw_block = format!("{TOOL_OPEN}{raw_inner}{TOOL_CLOSE}");
                                self.emit_content(&raw_block, &mut out_content);
                            }
                        }
                        continue;
                    }
                    if finished {
                        let unclosed = format!("{TOOL_OPEN}{}", std::mem::take(&mut self.buf));
                        self.mode = Mode::Content;
                        self.emit_content(&unclosed, &mut out_content);
                    }
                    break;
                }
            }
        }

        let mut out_reasoning = String::new();
        if !raw_reasoning.is_empty() {
            let comb = std::mem::take(&mut self.deferred_reasoning) + &raw_reasoning;
            let trimmed = py_rstrip(&comb).len();
            self.deferred_reasoning = comb[trimmed..].to_string();
            out_reasoning = comb[..trimmed].to_string();
        }
        if self.reasoning_ended || finished {
            self.deferred_reasoning.clear();
        }

        if finished {
            if !self.content_has_nonws && self.calls.is_empty() && !self.deferred_content.is_empty() {
                out_content.push_str(&std::mem::take(&mut self.deferred_content));
            } else {
                self.deferred_content.clear();
            }
            self.post_call_ws.clear();
        }

        (out_reasoning, out_content, out_tools)
    }

    // ---- streaming API ------------------------------------------------------------------------------------------

    pub fn delta(&mut self, text: &str, finished: bool) -> Option<Delta> {
        let (reasoning, content, tool_calls) = self.feed(text, finished);
        let d = Delta {
            content: (!content.is_empty()).then_some(content),
            reasoning: (!reasoning.is_empty() && self.include_reasoning).then_some(reasoning),
            tool_calls,
        };
        (d.content.is_some() || d.reasoning.is_some() || !d.tool_calls.is_empty()).then_some(d)
    }

    // ---- non-streaming API --------------------------------------------------------------------------------------

    pub fn parse_full(&mut self, text: &str) -> FullParse {
        let (r, c, _) = self.feed(text, true);
        let calls = std::mem::take(&mut self.calls);
        let content = c.as_str();
        FullParse {
            reasoning: (self.include_reasoning && !r.is_empty()).then_some(r),
            content: (!content.is_empty()).then(|| content.to_string()),
            calls,
        }
    }
}

/// The length of the longest proper prefix of `tag` that `text` ends with.
fn partial_tag(text: &str, tag: &str) -> usize {
    let (t, g) = (text.as_bytes(), tag.as_bytes());
    (1..g.len().min(t.len() + 1)).rev().find(|&n| t.ends_with(&g[..n])).unwrap_or(0)
}

/// Locate the closing `</tool_call>` for the open `<tool_call>` buffer. When the payload opens with `{`, `</tool_call>`
/// inside a valid JSON string literal is skipped so string arguments containing `</tool_call>` stay intact.
fn find_tool_end(buf: &str, finished: bool) -> Option<usize> {
    let ws = buf.len() - buf.trim_start_matches(is_py_space).len();
    let rest = &buf[ws..];
    if rest.starts_with('{') {
        let bytes = rest.as_bytes();
        let mut in_str = false;
        let mut esc = false;
        let mut broken_str = false;
        let mut i = 0;
        while i < bytes.len() {
            let b = bytes[i];
            if in_str && !broken_str {
                if esc {
                    esc = false;
                } else if b == b'\\' {
                    esc = true;
                } else if b == b'"' {
                    in_str = false;
                } else if b < 0x20 {
                    broken_str = true;
                }
            } else if !broken_str && b == b'"' {
                in_str = true;
            } else if bytes[i..].starts_with(TOOL_CLOSE.as_bytes()) {
                return Some(ws + i);
            }
            i += 1;
        }
        if finished {
            return buf.find(TOOL_CLOSE);
        }
        None
    } else if rest.is_empty() {
        None
    } else {
        buf.find(TOOL_CLOSE)
    }
}

fn parse_tool_json(inner: &str, known: &HashSet<String>, emit_undeclared: bool) -> CallOutcome {
    let trimmed = py_strip(inner);
    let Ok(Value::Object(obj)) = serde_json::from_str::<Value>(trimmed) else {
        return CallOutcome::RawText;
    };
    let Some(name) = obj.get("name").and_then(Value::as_str).filter(|n| !n.is_empty()) else {
        return CallOutcome::RawText;
    };
    let name_ok = known.contains(name) || (emit_undeclared && plausible_name(name));
    if !name_ok {
        return CallOutcome::RawText;
    }
    let raw_args = obj.get("arguments").filter(|v| !v.is_null()).or_else(|| obj.get("parameters"));
    let arguments = match raw_args {
        None | Some(Value::Null) => "{}".to_string(),
        Some(v @ Value::Object(_)) => pyjson::to_string(v),
        Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
            Ok(v @ Value::Object(_)) => pyjson::to_string(&v),
            _ if py_strip(s).is_empty() => "{}".to_string(),
            _ => s.clone(),
        },
        Some(_) => return CallOutcome::RawText,
    };
    CallOutcome::Call { name: name.to_string(), arguments }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tools() -> Value {
        json!([
            {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "fs::write", "parameters": {"type": "object"}}}
        ])
    }

    type Got = (Option<String>, Option<String>, Vec<(String, String)>);

    fn as_got(f: FullParse) -> Got {
        (f.reasoning, f.content, f.calls.into_iter().map(|(_, n, a)| (n, a)).collect())
    }

    fn stream_chars(mut p: HfParser, text: &str, chunk_chars: usize) -> Got {
        let chars: Vec<char> = text.chars().collect();
        let mut r = String::new();
        let mut c = String::new();
        let mut calls: Vec<(String, String)> = Vec::new();
        if chars.is_empty() {
            if let Some(d) = p.delta("", true) {
                r.push_str(d.reasoning.as_deref().unwrap_or(""));
                c.push_str(d.content.as_deref().unwrap_or(""));
            }
        }
        let mut i = 0;
        while i < chars.len() {
            let end = (i + chunk_chars).min(chars.len());
            let s: String = chars[i..end].iter().collect();
            i = end;
            if let Some(d) = p.delta(&s, i >= chars.len()) {
                if let Some(dr) = &d.reasoning {
                    r.push_str(dr);
                }
                if let Some(dc) = &d.content {
                    c.push_str(dc);
                    assert!(
                        !dc.contains("<tool_call>")
                            || text.contains("malformed")
                            || text.contains("secret_tool")
                            || text.contains("literal"),
                        "unexpected raw <tool_call> leaked in delta: {dc:?}"
                    );
                }
                for tc in d.tool_calls {
                    while calls.len() <= tc.index {
                        calls.push((String::new(), String::new()));
                    }
                    if let Some(n) = tc.name {
                        calls[tc.index].0 = n;
                    }
                    if let Some(a) = tc.arguments {
                        calls[tc.index].1.push_str(&a);
                    }
                }
            }
        }
        ((!r.is_empty()).then_some(r), (!c.is_empty()).then_some(c), calls)
    }

    /// Simulates raw 1-byte arrival across multi-byte UTF-8 boundaries (holding incomplete UTF-8 codepoints the way
    /// `Detok` does, then feeding each decoded slice to `HfParser::delta`).
    fn stream_raw_bytes(mut p: HfParser, text: &str, step_bytes: usize) -> Got {
        let bytes = text.as_bytes();
        let mut pending: Vec<u8> = Vec::new();
        let mut r = String::new();
        let mut c = String::new();
        let mut calls: Vec<(String, String)> = Vec::new();
        let mut at = 0;
        while at < bytes.len() {
            let end = (at + step_bytes).min(bytes.len());
            pending.extend_from_slice(&bytes[at..end]);
            at = end;
            let valid_len = match std::str::from_utf8(&pending) {
                Ok(_) => pending.len(),
                Err(e) => e.valid_up_to(),
            };
            let chunk = std::str::from_utf8(&pending[..valid_len]).unwrap().to_string();
            pending.drain(..valid_len);
            let done = at >= bytes.len() && pending.is_empty();
            if chunk.is_empty() && !done {
                continue;
            }
            if let Some(d) = p.delta(&chunk, done) {
                if let Some(dr) = d.reasoning {
                    r.push_str(&dr);
                }
                if let Some(dc) = d.content {
                    c.push_str(&dc);
                }
                for tc in d.tool_calls {
                    while calls.len() <= tc.index {
                        calls.push((String::new(), String::new()));
                    }
                    if let Some(n) = tc.name {
                        calls[tc.index].0 = n;
                    }
                    if let Some(a) = tc.arguments {
                        calls[tc.index].1.push_str(&a);
                    }
                }
            }
        }
        ((!r.is_empty()).then_some(r), (!c.is_empty()).then_some(c), calls)
    }

    #[test]
    fn think_and_content_full_and_streamed() {
        let text = "<think>\nLet me reason step by step.  \n</think>\n\nThe answer is 42.";
        let want = as_got(HfParser::new(None, &json!("none"), true).parse_full(text));
        assert_eq!(want.0.as_deref(), Some("\nLet me reason step by step."));
        assert_eq!(want.1.as_deref(), Some("\n\nThe answer is 42."));
        assert!(want.2.is_empty());
        for k in [1, 2, 3, 5, 7, 16] {
            assert_eq!(stream_chars(HfParser::new(None, &json!("none"), true), text, k), want, "chunk {k}");
            assert_eq!(stream_raw_bytes(HfParser::new(None, &json!("none"), true), text, k), want, "bytes {k}");
        }
    }

    #[test]
    fn plain_reply_without_think_is_content() {
        let t = tools();
        let text = "Hello! No thinking tags here.";
        let want = as_got(HfParser::new(Some(&t), &json!("auto"), true).parse_full(text));
        assert_eq!(want, (None, Some(text.to_string()), vec![]));
        for k in [1, 2, 4] {
            assert_eq!(stream_chars(HfParser::new(Some(&t), &json!("auto"), true), text, k), want);
        }
    }

    #[test]
    fn content_first_keeps_leading_think_tags_as_literal_content() {
        let t = tools();
        let text = "<think>literal example</think> Direct content.";
        let want = as_got(HfParser::new(Some(&t), &json!("auto"), true).content_first().parse_full(text));
        assert_eq!(want, (None, Some(text.to_string()), vec![]));
        assert_eq!(stream_chars(HfParser::new(Some(&t), &json!("auto"), true).content_first(), text, 1), want);
    }

    #[test]
    fn in_think_handles_prompt_opened_think() {
        let t = tools();
        let text = "already inside think</think>Visible reply.";
        let want = as_got(HfParser::new(Some(&t), &json!("auto"), true).in_think().parse_full(text));
        assert_eq!(want, (Some("already inside think".into()), Some("Visible reply.".into()), vec![]));
        assert_eq!(stream_chars(HfParser::new(Some(&t), &json!("auto"), true).in_think(), text, 1), want);
    }

    #[test]
    fn include_reasoning_false_suppresses_reasoning_without_leaking_to_content() {
        let t = tools();
        let text = "<think>internal thoughts</think>Public answer";
        let full = as_got(HfParser::new(Some(&t), &json!("auto"), false).parse_full(text));
        assert_eq!(full, (None, Some("Public answer".into()), vec![]));
        assert_eq!(stream_chars(HfParser::new(Some(&t), &json!("auto"), false), text, 1), full);
    }

    #[test]
    fn single_and_multiple_tool_calls_with_arguments_variants() {
        let t = tools();
        // Object arguments, stringified JSON arguments, and "parameters" alias in one multi-call reply
        let text = "<think>Need weather, time, and write.</think>\n\n\
            <tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"days\": 3}}\n</tool_call>\n\
            <tool_call>\n{\"name\": \"get_time\", \"arguments\": \"{\\\"tz\\\":\\\"UTC\\\"}\"}\n</tool_call>\n\
            <tool_call>{\"name\": \"fs::write\", \"parameters\": {\"path\": \"/tmp/a\", \"ok\": true}}</tool_call>\n";
        let full = HfParser::new(Some(&t), &json!("auto"), true).parse_full(text);
        assert_eq!(full.reasoning.as_deref(), Some("Need weather, time, and write."));
        assert_eq!(full.content, None);
        assert_eq!(full.calls.len(), 3);
        assert!(full.calls[0].0.starts_with("chatcmpl-tool-"));
        let want = as_got(full);
        assert_eq!(
            want.2,
            vec![
                ("get_weather".into(), "{\"city\": \"Paris\", \"days\": 3}".into()),
                ("get_time".into(), "{\"tz\": \"UTC\"}".into()),
                ("fs::write".into(), "{\"path\": \"/tmp/a\", \"ok\": true}".into()),
            ]
        );
        for k in [1, 2, 3, 5, 11] {
            assert_eq!(stream_chars(HfParser::new(Some(&t), &json!("auto"), true), text, k), want, "chunk {k}");
            assert_eq!(stream_raw_bytes(HfParser::new(Some(&t), &json!("auto"), true), text, k), want, "bytes {k}");
        }
    }

    #[test]
    fn partial_tag_holdback_never_leaks_tool_call_tags() {
        let t = tools();
        let text = "Checking weather now: <tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\"}}</tool_call>";
        let mut p = HfParser::new(Some(&t), &json!("auto"), true);
        let mut collected_content = String::new();
        let mut collected_calls = Vec::new();
        for (i, ch) in text.char_indices() {
            let next = i + ch.len_utf8();
            if let Some(d) = p.delta(&text[i..next], next == text.len()) {
                if let Some(c) = d.content {
                    assert!(!c.contains('<'), "partial tag leaked into content: {c:?}");
                    collected_content.push_str(&c);
                }
                for tc in d.tool_calls {
                    collected_calls.push((tc.name.unwrap(), tc.arguments.unwrap()));
                }
            }
        }
        assert_eq!(collected_content, "Checking weather now: ");
        assert_eq!(collected_calls, vec![("get_weather".into(), "{\"city\": \"Rome\"}".into())]);
    }

    #[test]
    fn declared_vs_undeclared_tools() {
        let t = tools();
        let text = "<tool_call>\n{\"name\": \"secret_tool\", \"arguments\": {\"k\": \"v\"}}\n</tool_call>";

        // emit_undeclared = false: stays in content
        let off = as_got(HfParser::new(Some(&t), &json!("auto"), true).emit_undeclared(false).parse_full(text));
        assert_eq!(off, (None, Some(text.to_string()), vec![]));
        assert_eq!(stream_chars(HfParser::new(Some(&t), &json!("auto"), true).emit_undeclared(false), text, 1), off);

        // emit_undeclared = true: emitted as a tool call
        let on = as_got(HfParser::new(Some(&t), &json!("auto"), true).emit_undeclared(true).parse_full(text));
        assert_eq!(on, (None, None, vec![("secret_tool".into(), "{\"k\": \"v\"}".into())]));
        assert_eq!(stream_chars(HfParser::new(Some(&t), &json!("auto"), true).emit_undeclared(true), text, 1), on);

        // implausible name stays in content even with emit_undeclared = true
        let bad_name = "<tool_call>{\"name\": \"not a tool!\", \"arguments\": {}}</tool_call>";
        let bad = as_got(HfParser::new(Some(&t), &json!("auto"), true).emit_undeclared(true).parse_full(bad_name));
        assert_eq!(bad, (None, Some(bad_name.to_string()), vec![]));
    }

    #[test]
    fn malformed_json_and_unclosed_tags_fall_back_to_content() {
        let t = tools();
        let cases = [
            "<tool_call>{malformed json: 123}</tool_call>",
            "<tool_call>malformed plain text</tool_call>",
            "<tool_call>{\"arguments\": {\"missing\": \"name\"}}</tool_call>",
            "Prefix <tool_call>{\"name\": \"get_weather\", \"arguments\": {\"unclosed\": ",
        ];
        for text in cases {
            let want = as_got(HfParser::new(Some(&t), &json!("auto"), true).parse_full(text));
            assert_eq!(want, (None, Some(text.to_string()), vec![]), "{text:?}");
            for k in [1, 2, 5] {
                let mut p = HfParser::new(Some(&t), &json!("auto"), true);
                let chars: Vec<char> = text.chars().collect();
                let mut c = String::new();
                let mut i = 0;
                while i < chars.len() {
                    let end = (i + k).min(chars.len());
                    let s: String = chars[i..end].iter().collect();
                    i = end;
                    if let Some(d) = p.delta(&s, i >= chars.len()) {
                        c.push_str(d.content.as_deref().unwrap_or(""));
                        assert!(d.tool_calls.is_empty());
                    }
                }
                assert_eq!(c, text, "chunk {k} on {text:?}");
            }
        }
    }

    #[test]
    fn utf8_multibyte_and_embedded_closer_in_json_string() {
        let t = tools();
        let text = "<think>考える 🙂 café</think>\n\
            <tool_call>\n{\"name\": \"fs::write\", \"arguments\": {\"city\": \"東京 🙂\", \"note\": \"keep </tool_call> literal\"}}\n</tool_call>";
        let want = as_got(HfParser::new(Some(&t), &json!("auto"), true).parse_full(text));
        assert_eq!(want.0.as_deref(), Some("考える 🙂 café"));
        assert_eq!(want.1, None);
        assert_eq!(
            want.2,
            vec![("fs::write".into(), "{\"city\": \"東京 🙂\", \"note\": \"keep </tool_call> literal\"}".into())]
        );
        for step in [1, 2, 3, 4, 7] {
            assert_eq!(
                stream_raw_bytes(HfParser::new(Some(&t), &json!("auto"), true), text, step),
                want,
                "step {step}"
            );
            assert_eq!(stream_chars(HfParser::new(Some(&t), &json!("auto"), true), text, step), want, "chars {step}");
        }
    }

    #[test]
    fn tool_choice_none_and_empty_tools_keep_tags_in_content() {
        let t = tools();
        let text = "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}</tool_call>";
        let none_choice = as_got(HfParser::new(Some(&t), &json!("none"), true).parse_full(text));
        assert_eq!(none_choice, (None, Some(text.to_string()), vec![]));
        let no_tools = as_got(HfParser::new(None, &json!("auto"), true).parse_full(text));
        assert_eq!(no_tools, (None, Some(text.to_string()), vec![]));
    }
    #[test]
    fn every_split_preserves_mixed_content_and_malformed_unicode() {
        let tools = tools();
        for text in [
            "  Prefix <tool_call>{\"name\":\"get_time\",\"arguments\":{}}</tool_call>  suffix  ",
            "<tool_call>{\"name\":\"get_time\",\"arguments\":{}}</tool_call>  suffix  ",
            "<tool_call>{🙂 malformed}</tool_call>",
            "<think>r </think> hi <tool_call>{\"name\":\"get_time\"}</tool_call>",
        ] {
            let want = as_got(HfParser::new(Some(&tools), &json!("auto"), true).parse_full(text));
            for step in 1..=text.len() {
                assert_eq!(
                    stream_raw_bytes(HfParser::new(Some(&tools), &json!("auto"), true), text, step),
                    want,
                    "step={step} {text}"
                );
            }
        }
    }
}
