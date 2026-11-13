//! Inkling reply parser (`--dialect inkling`): a reply is a run of channel-tagged blocks,
//! `<|message_model|>HEADER<|content_KIND|>BODY<|end_message|>`, closed by `<|content_model_end_sampling|>`.
//!
//! Rules:
//! - `<|content_thinking|>` bodies are `reasoning`; `<|content_text|>`, `<|content_tool_error|>`,
//!   `<|content_invoke_tool_text|>` and any other kind are `content`. Bodies are kept byte for byte (no stripping) and
//!   blocks of one kind join with nothing between them.
//! - `<|content_invoke_tool_json|>` bodies are tool calls when `tools` are offered and `tool_choice` is not `"none"`:
//!   a JSON object `{"name": ..., "args": {...}}` (`"arguments"` is read when `"args"` is absent; the header names the
//!   function when `"name"` is missing or empty). A declared name (or a plausible one with `emit_undeclared`) becomes
//!   an OpenAI tool call. Anything else -- an undeclared name, malformed JSON, a block the reply never closes, or a
//!   request without tools -- gives the body back as `content`, without its control tags.
//! - `<|message_model|>` and its header are optional before a block; `<|content_model_end_sampling|>`,
//!   `<|endoftext|>` and a stray `<|end_message|>` are dropped wherever they stand.
//! - `content_first()`: the prompt left an assistant text message open (`continue_final_message`), so the reply starts
//!   inside a text block.
//! - Streaming holds back any tail that may still grow into a control tag, so every chunking gives the same
//!   concatenated output as `parse_full`.

use std::collections::HashSet;

use serde_json::Value;

use crate::parser::{plausible_name, tool_call_id, Delta, FullParse, ToolDelta};
use crate::pyjson::{self, py_strip};

const MODEL: &str = "<|message_model|>";
const END: &str = "<|end_message|>";
const STOP: &str = "<|content_model_end_sampling|>";
const EOT: &str = "<|endoftext|>";
const CONTENT: &str = "<|content_";
/// The longest `<|content_KIND|>` tag read as one; a longer run is text.
const TAG_MAX: usize = 64;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Thinking,
    Text,
    Tool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Tag {
    Model,
    End,
    Stop,
    Content(Kind),
}

enum Scan {
    /// a whole tag and its length in bytes
    Full(Tag, usize),
    /// the text so far may still grow into a tag
    Partial,
    No,
}

/// The control tag `s` starts with. `s` starts with `<`.
fn tag_at(s: &str) -> Scan {
    for (text, tag) in [(MODEL, Tag::Model), (END, Tag::End), (STOP, Tag::Stop), (EOT, Tag::Stop)] {
        if s.starts_with(text) {
            return Scan::Full(tag, text.len());
        }
        if text.starts_with(s) {
            return Scan::Partial;
        }
    }
    let Some(rest) = s.strip_prefix(CONTENT) else {
        return if CONTENT.starts_with(s) { Scan::Partial } else { Scan::No };
    };
    let name = rest.bytes().take_while(|b| b.is_ascii_alphanumeric() || *b == b'_').count();
    let len = CONTENT.len() + name + 2;
    if len > TAG_MAX {
        return Scan::No;
    }
    let after = &rest[name..];
    if after.starts_with("|>") && name > 0 {
        let kind = match &rest[..name] {
            "thinking" => Kind::Thinking,
            "invoke_tool_json" => Kind::Tool,
            _ => Kind::Text,
        };
        return Scan::Full(Tag::Content(kind), len);
    }
    if after.is_empty() || (after == "|" && name > 0) {
        Scan::Partial
    } else {
        Scan::No
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    /// outside a block: a tag opens the next one
    Between,
    /// after `<|message_model|>`: the header runs to the next `<`
    Header,
    /// inside a reasoning or content block
    Body(Kind),
    /// inside a JSON tool call, kept until it closes
    Call,
}

pub struct InkParser {
    tools: HashSet<String>,
    parse_tools: bool,
    emit_undeclared: bool,
    pub include_reasoning: bool,
    mode: Mode,
    buf: String,
    header: String,
    calls: Vec<(String, String, String)>,
}

impl InkParser {
    pub fn new(tools: Option<&Value>, tool_choice: &Value, include_reasoning: bool) -> InkParser {
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
        InkParser {
            tools: names,
            parse_tools: any && tool_choice.as_str() != Some("none"),
            emit_undeclared: false,
            include_reasoning,
            mode: Mode::Between,
            buf: String::new(),
            header: String::new(),
            calls: Vec::new(),
        }
    }

    /// The prompt left an assistant text message open: the reply goes on inside it.
    pub fn content_first(mut self) -> InkParser {
        self.mode = Mode::Body(Kind::Text);
        self
    }

    /// `--undeclared-tool-calls emit`: a call to an undeclared tool whose name is a plausible identifier is a tool call.
    pub fn emit_undeclared(mut self, on: bool) -> InkParser {
        self.emit_undeclared = on;
        self
    }

    fn open(&mut self, kind: Kind) {
        self.mode = match kind {
            Kind::Tool if self.parse_tools => Mode::Call,
            Kind::Tool => Mode::Body(Kind::Text),
            k => Mode::Body(k),
        };
    }

    fn feed(&mut self, text: &str, finished: bool) -> (String, String, Vec<ToolDelta>) {
        let (mut reasoning, mut content, mut tools) = (String::new(), String::new(), Vec::new());
        self.buf.push_str(text);
        loop {
            match self.mode {
                Mode::Between => {
                    if self.buf.is_empty() {
                        break;
                    }
                    let scan = if self.buf.starts_with('<') { tag_at(&self.buf) } else { Scan::No };
                    match scan {
                        Scan::Full(tag, n) => {
                            self.buf.drain(..n);
                            match tag {
                                Tag::Model => self.mode = Mode::Header,
                                Tag::Content(kind) => {
                                    self.header.clear();
                                    self.open(kind);
                                }
                                Tag::End | Tag::Stop => {}
                            }
                        }
                        Scan::Partial if !finished => break,
                        // text outside any block: content, up to the next thing that may be a tag
                        _ => {
                            let n = self.buf[1..].find('<').map_or(self.buf.len(), |p| p + 1);
                            content.extend(self.buf.drain(..n));
                        }
                    }
                }
                Mode::Header => {
                    let Some(p) = self.buf.find('<') else {
                        if finished {
                            self.buf.clear(); // a header the reply never used
                            self.mode = Mode::Between;
                        }
                        break;
                    };
                    match tag_at(&self.buf[p..]) {
                        Scan::Full(Tag::Content(kind), n) => {
                            self.header = self.buf[..p].to_string();
                            self.buf.drain(..p + n);
                            self.open(kind);
                        }
                        Scan::Partial if !finished => break,
                        // no block follows: the header goes, and what follows is read on its own
                        _ => {
                            self.buf.drain(..p);
                            self.mode = Mode::Between;
                        }
                    }
                }
                Mode::Body(kind) => {
                    let out = if kind == Kind::Thinking { &mut reasoning } else { &mut content };
                    let Some(p) = self.buf.find('<') else {
                        out.push_str(&std::mem::take(&mut self.buf));
                        break;
                    };
                    out.extend(self.buf.drain(..p));
                    match tag_at(&self.buf) {
                        Scan::Full(tag, n) => {
                            // its own closer goes with the block; any other tag closes it and is read next
                            if tag == Tag::End {
                                self.buf.drain(..n);
                            }
                            self.mode = Mode::Between;
                        }
                        Scan::Partial if !finished => break,
                        _ => out.extend(self.buf.drain(..1)),
                    }
                }
                Mode::Call => {
                    let Some((end, n)) = find_call_end(&self.buf, finished) else {
                        if finished {
                            content.push_str(&std::mem::take(&mut self.buf));
                            self.mode = Mode::Between;
                        }
                        break;
                    };
                    let body: String = self.buf.drain(..end).collect();
                    if self.buf[..n] == *END {
                        self.buf.drain(..n);
                    }
                    self.mode = Mode::Between;
                    match parse_call(&body, &self.header, &self.tools, self.emit_undeclared) {
                        Some((name, arguments)) => {
                            let (index, id) = (self.calls.len(), tool_call_id());
                            self.calls.push((id.clone(), name.clone(), arguments.clone()));
                            tools.push(ToolDelta { index, id: Some(id), name: Some(name), arguments: Some(arguments) });
                        }
                        None => content.push_str(&body),
                    }
                }
            }
        }
        (reasoning, content, tools)
    }

    pub fn delta(&mut self, text: &str, finished: bool) -> Option<Delta> {
        let (reasoning, content, tool_calls) = self.feed(text, finished);
        let d = Delta {
            content: (!content.is_empty()).then_some(content),
            reasoning: (!reasoning.is_empty() && self.include_reasoning).then_some(reasoning),
            tool_calls,
        };
        (d.content.is_some() || d.reasoning.is_some() || !d.tool_calls.is_empty()).then_some(d)
    }

    pub fn parse_full(&mut self, text: &str) -> FullParse {
        let (r, c, _) = self.feed(text, true);
        FullParse {
            reasoning: (self.include_reasoning && !r.is_empty()).then_some(r),
            content: (!c.is_empty()).then_some(c),
            calls: std::mem::take(&mut self.calls),
        }
    }
}

/// Where the open call's body ends, and the length of the tag found there. A tag inside a JSON string literal does not
/// end it while the body opens with `{`; once the reply is over the first tag does.
fn find_call_end(buf: &str, finished: bool) -> Option<(usize, usize)> {
    let b = buf.as_bytes();
    let json = py_strip(buf).starts_with('{');
    let (mut in_str, mut esc, mut first) = (false, false, None);
    for i in 0..b.len() {
        if b[i] == b'<' {
            if let Scan::Full(tag, n) = tag_at(&buf[i..]) {
                if !matches!(tag, Tag::Content(_)) {
                    if !(json && in_str) {
                        return Some((i, n));
                    }
                    first = first.or(Some((i, n)));
                }
            }
        }
        if !json {
            continue;
        }
        if in_str {
            if esc {
                esc = false;
            } else if b[i] == b'\\' {
                esc = true;
            } else if b[i] == b'"' {
                in_str = false;
            }
        } else if b[i] == b'"' {
            in_str = true;
        }
    }
    if finished {
        first
    } else {
        None
    }
}

/// A call body as (name, arguments), or None when it is not a call the request can take.
fn parse_call(body: &str, header: &str, known: &HashSet<String>, emit_undeclared: bool) -> Option<(String, String)> {
    let Ok(Value::Object(obj)) = serde_json::from_str::<Value>(py_strip(body)) else { return None };
    let name = match obj.get("name").and_then(Value::as_str).filter(|n| !n.is_empty()) {
        Some(n) => n,
        None => Some(py_strip(header)).filter(|h| !h.is_empty())?,
    };
    if !(known.contains(name) || (emit_undeclared && plausible_name(name))) {
        return None;
    }
    let raw = if obj.contains_key("args") { obj.get("args") } else { obj.get("arguments") };
    let arguments = match raw {
        None | Some(Value::Null) => "{}".to_string(),
        Some(v @ Value::Object(_)) => pyjson::to_string(v),
        Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
            Ok(v @ Value::Object(_)) => pyjson::to_string(&v),
            _ if py_strip(s).is_empty() => "{}".to_string(),
            _ => s.clone(),
        },
        Some(_) => return None,
    };
    Some((name.to_string(), arguments))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tools() -> Value {
        json!([
            {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object"}}}
        ])
    }

    type Got = (Option<String>, Option<String>, Vec<(String, String)>);

    fn as_got(f: FullParse) -> Got {
        (f.reasoning, f.content, f.calls.into_iter().map(|(_, n, a)| (n, a)).collect())
    }

    /// The reply in `step`-byte pieces, each held to whole UTF-8 characters the way the detokenizer holds them. No
    /// delta may carry a control tag.
    fn stream(mut p: InkParser, text: &str, step: usize) -> Got {
        let bytes = text.as_bytes();
        let (mut r, mut c, mut calls) = (String::new(), String::new(), Vec::<(String, String)>::new());
        let mut pending: Vec<u8> = Vec::new();
        let mut at = 0;
        loop {
            let end = (at + step).min(bytes.len());
            pending.extend_from_slice(&bytes[at..end]);
            at = end;
            let valid = match std::str::from_utf8(&pending) {
                Ok(_) => pending.len(),
                Err(e) => e.valid_up_to(),
            };
            let chunk = String::from_utf8(pending.drain(..valid).collect()).unwrap();
            let done = at >= bytes.len();
            if let Some(d) = p.delta(&chunk, done) {
                for s in [&d.reasoning, &d.content].into_iter().flatten() {
                    assert!(!s.contains("<|") || text.contains("literal"), "a tag leaked: {s:?}");
                }
                r.push_str(d.reasoning.as_deref().unwrap_or(""));
                c.push_str(d.content.as_deref().unwrap_or(""));
                for tc in d.tool_calls {
                    assert_eq!(tc.index, calls.len());
                    assert!(tc.id.unwrap().starts_with("chatcmpl-tool-"));
                    calls.push((tc.name.unwrap(), tc.arguments.unwrap()));
                }
            }
            if done {
                break;
            }
        }
        ((!r.is_empty()).then_some(r), (!c.is_empty()).then_some(c), calls)
    }

    /// `parse_full`, checked against every chunk size from one byte up.
    fn both(make: impl Fn() -> InkParser, text: &str) -> Got {
        let want = as_got(make().parse_full(text));
        for step in 1..=text.len().max(1) {
            assert_eq!(stream(make(), text, step), want, "step {step} of {text:?}");
        }
        want
    }

    fn auto() -> InkParser {
        InkParser::new(Some(&tools()), &json!("auto"), true)
    }

    const CALL: &str = "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"SF\"}}<|end_message|>";

    fn weather() -> (String, String) {
        ("get_weather".into(), "{\"city\": \"SF\"}".into())
    }

    #[test]
    fn thinking_then_text() {
        let text = "<|message_model|><|content_thinking|> Let me think. \n<|end_message|><|message_model|><|content_text|>\nThe answer is 42. <|end_message|><|content_model_end_sampling|>";
        let want = (Some(" Let me think. \n".to_string()), Some("\nThe answer is 42. ".to_string()), vec![]);
        assert_eq!(both(auto, text), want);
        assert_eq!(both(|| InkParser::new(None, &json!("none"), true), text), want);
        // the end tags are optional, and <|endoftext|> goes the same way
        let cut = text.strip_suffix(STOP).unwrap();
        assert_eq!(both(auto, cut), want);
        assert_eq!(both(auto, &format!("{cut}<|endoftext|>")), want);
        // a reply cut inside a block keeps what it wrote
        let open = "<|message_model|><|content_thinking|>half a thou";
        assert_eq!(both(auto, open), (Some("half a thou".into()), None, vec![]));
    }

    #[test]
    fn text_alone_and_repeated_blocks_join() {
        let text = "<|message_model|><|content_text|>Hello.<|end_message|><|content_model_end_sampling|>";
        assert_eq!(both(auto, text), (None, Some("Hello.".into()), vec![]));
        let text =
            "<|message_model|><|content_thinking|>a<|end_message|><|message_model|><|content_text|>x<|end_message|>\
            <|message_model|><|content_thinking|>b<|end_message|><|content_tool_error|>y<|end_message|>\
            <|message_model|>f<|content_invoke_tool_text|>z<|end_message|><|content_model_end_sampling|>";
        assert_eq!(both(auto, text), (Some("ab".into()), Some("xyz".into()), vec![]));
    }

    #[test]
    fn include_reasoning_false_drops_the_reasoning() {
        let text = "<|message_model|><|content_thinking|>secret<|end_message|><|message_model|><|content_text|>Public<|end_message|>";
        let got = both(|| InkParser::new(Some(&tools()), &json!("auto"), false), text);
        assert_eq!(got, (None, Some("Public".into()), vec![]));
    }

    #[test]
    fn single_multi_and_headerless_calls() {
        let end = "<|content_model_end_sampling|>";
        assert_eq!(both(auto, &format!("{CALL}{end}")), (None, None, vec![weather()]));
        // a forced call has no header; the args may come as "arguments", as a JSON string, or not at all
        let bare = "<|message_model|><|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"SF\"}}<|end_message|>";
        assert_eq!(both(auto, bare), (None, None, vec![weather()]));
        assert_eq!(both(auto, &bare["<|message_model|>".len()..]), (None, None, vec![weather()]));
        let text = format!(
            "<|message_model|><|content_thinking|>two calls<|end_message|><|message_model|><|content_text|>On it.<|end_message|>{CALL}\
             <|message_model|>get_time<|content_invoke_tool_json|>{{\"name\":\"get_time\",\"arguments\":\"{{\\\"tz\\\":\\\"UTC\\\"}}\"}}<|end_message|>\
             <|message_model|>get_time<|content_invoke_tool_json|> {{\"args\":null}} <|end_message|>{end}"
        );
        let full = auto().parse_full(&text);
        assert_eq!(full.calls.len(), 3);
        assert!(full.calls.iter().all(|c| c.0.starts_with("chatcmpl-tool-")));
        assert_ne!(full.calls[0].0, full.calls[1].0);
        let want = (
            Some("two calls".to_string()),
            Some("On it.".to_string()),
            vec![weather(), ("get_time".into(), "{\"tz\": \"UTC\"}".into()), ("get_time".into(), "{}".into())],
        );
        assert_eq!(both(auto, &text), want);
    }

    #[test]
    fn a_closer_inside_a_json_string_and_multibyte_text_survive() {
        let text = "<|message_model|><|content_thinking|>考える 🙂<|end_message|><|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"東京 🙂\",\"note\":\"a literal <|end_message|> \\\" here\"}}<|end_message|>";
        let want = (
            Some("考える 🙂".to_string()),
            None,
            vec![(
                "get_weather".to_string(),
                "{\"city\": \"東京 🙂\", \"note\": \"a literal <|end_message|> \\\" here\"}".to_string(),
            )],
        );
        assert_eq!(both(auto, text), want);
    }

    #[test]
    fn tool_choice_none_and_no_tools_give_the_body_as_content() {
        let body = "{\"name\":\"get_weather\",\"args\":{\"city\":\"SF\"}}";
        let want = (None, Some(body.to_string()), vec![]);
        assert_eq!(both(|| InkParser::new(Some(&tools()), &json!("none"), true), CALL), want);
        assert_eq!(both(|| InkParser::new(None, &json!("auto"), true), CALL), want);
        assert_eq!(both(|| InkParser::new(Some(&json!([])), &json!("auto"), true), CALL), want);
    }

    #[test]
    fn undeclared_names_follow_emit_undeclared() {
        let body = "{\"name\":\"secret_tool\",\"args\":{\"k\":\"v\"}}";
        let text = format!("<|message_model|>secret_tool<|content_invoke_tool_json|>{body}<|end_message|>");
        assert_eq!(both(|| auto().emit_undeclared(false), &text), (None, Some(body.to_string()), vec![]));
        let call = ("secret_tool".to_string(), "{\"k\": \"v\"}".to_string());
        assert_eq!(both(|| auto().emit_undeclared(true), &text), (None, None, vec![call]));
        // a name that is no identifier stays text either way
        let body = "{\"name\":\"not a tool!\",\"args\":{}}";
        let text = format!("<|content_invoke_tool_json|>{body}<|end_message|>");
        assert_eq!(both(|| auto().emit_undeclared(true), &text), (None, Some(body.to_string()), vec![]));
    }

    #[test]
    fn malformed_and_unclosed_calls_fall_back_to_content() {
        let open = "<|message_model|>get_weather<|content_invoke_tool_json|>";
        for body in ["{malformed: 1}", "plain words", "[1, 2]", "{\"name\":\"get_weather\",\"args\":7}", "{\"args\":{}"]
        {
            let text = format!("{open}{body}<|end_message|><|content_model_end_sampling|>");
            assert_eq!(both(auto, &text), (None, Some(body.to_string()), vec![]), "{body}");
        }
        // the header names the function when the body does not
        let text = format!("{open}{{\"args\":{{\"city\":\"SF\"}}}}<|end_message|>");
        assert_eq!(both(auto, &text), (None, None, vec![weather()]));
        // cut by max_tokens inside the call, and inside one of its strings
        for body in ["{\"name\":\"get_weather\",\"args\":{\"ci", "{\"name\":\"get_weather\",\"args\":{\"city\":\"S"] {
            assert_eq!(both(auto, &format!("{open}{body}")), (None, Some(body.to_string()), vec![]));
        }
        // a call the reply ends without closing still ends at the stop tag
        let text = format!("{open}{{\"args\":{{}}}}<|content_model_end_sampling|>");
        assert_eq!(both(auto, &text), (None, None, vec![("get_weather".into(), "{}".into())]));
    }

    #[test]
    fn content_first_continues_an_open_text_message() {
        let text = format!(
            "lo there.<|end_message|><|message_model|><|content_thinking|>more<|end_message|>{CALL}<|content_model_end_sampling|>"
        );
        let got = both(|| auto().content_first(), &text);
        assert_eq!(got, (Some("more".into()), Some("lo there.".into()), vec![weather()]));
        assert_eq!(both(|| auto().content_first(), "plain"), (None, Some("plain".into()), vec![]));
    }

    #[test]
    fn stray_text_headers_and_lookalike_tags_do_not_drop_bytes() {
        // text outside a block is content; a header with no block after it goes
        assert_eq!(both(auto, "no tags at all"), (None, Some("no tags at all".into()), vec![]));
        assert_eq!(both(auto, "<|message_model|>get_weather"), (None, None, vec![]));
        let text = "<|message_model|>x<|message_model|><|content_text|>a literal < and <| and <|content_ and <|nope|> stay<|end_message|>";
        let want = "a literal < and <| and <|content_ and <|nope|> stay";
        assert_eq!(both(auto, text), (None, Some(want.into()), vec![]));
        assert_eq!(both(auto, ""), (None, None, vec![]));
    }
}
