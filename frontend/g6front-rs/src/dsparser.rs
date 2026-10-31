//! DeepSeek-V4.1 reply parser: reasoning split + DSML tool calls (the inverse of `dschat`).
//!
//! The reply, as the official `parse_message_from_completion_text` reads it:
//! - thinking on: reasoning up to `</think>` (one token, 128822), then the answer; thinking off: the answer from the start;
//! - the answer runs to the end of the reply, or to `\n\n<｜DSML｜ calls>`, a block of
//!   `<｜DSML｜ invoke name="f">\n<｜DSML｜ parameter name="k" string="true|false">v</｜DSML｜ parameter>…</｜DSML｜ invoke>`
//!   ending with `</｜DSML｜ calls>`. `string="true"` values are the raw text, the rest is JSON.
//!
//! A reply that WRITES ABOUT a call is the same bytes as one that MAKES a call, so a block is taken as a call only if
//! tools were offered and `tool_choice` is not none, it opens in the answer (not in the reasoning), outside a ``` fence,
//! right after a blank line (or at the start of the answer), and every invoke is well formed and names an offered tool.
//! Anything else stays answer text, byte for byte, including a block the model quotes from its own instructions
//! (`$TOOL_NAME`).
//!
//! Streaming: partial tags are held back, trailing newlines of the answer are held until the next text shows they were
//! not the blank line before a call, and an invoke is emitted whole (id, name, arguments) when its closer arrives.
//! After a live `</｜DSML｜ calls>` the parser is `closed`: the model is finished, whatever it writes after is dropped,
//! and the caller ends the generation (a model that goes on past its closer has run away).

use std::collections::HashSet;

use serde_json::Value;

use crate::dschat::{CALLS_CLOSE, CALLS_OPEN, INVOKE_CLOSE, PARAM_CLOSE, THINK_CLOSE};
use crate::parser::{tool_call_id, Delta, FullParse, ToolDelta};
use crate::pyjson::{self, is_py_space};

const INVOKE_NAME: &str = "<｜DSML｜ invoke name=\"";
const PARAM_NAME: &str = "<｜DSML｜ parameter name=\"";
const PARAM_STRING: &str = "\" string=\"";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    Reasoning,
    Content,
    /// Inside a candidate call block, undecided.
    Block,
    /// The block closed: nothing more is read.
    Done,
}

#[derive(Default)]
struct Out {
    reasoning: String,
    content: String,
    calls: Vec<ToolDelta>,
}

enum Parsed<T> {
    Need,
    Bad,
    Ok(T),
}

pub struct DsParser {
    tools: HashSet<String>,
    parse_tools: bool,
    pub include_reasoning: bool,
    mode: Mode,
    buf: String,
    /// Answer newlines held back: they may be the blank line in front of a call block.
    held: String,
    /// Only whitespace has been written to the answer so far.
    blank: bool,
    fence: bool,
    ticks: usize,
    /// The block in progress: the text after its opener that no accepted invoke has used up.
    blk: String,
    /// What the block replaced, given back as answer text if the block is not a call: the blank line and the opener.
    pre_raw: String,
    n_calls: usize,
}

impl DsParser {
    /// `tools` is the request's tool list (OpenAI format).
    pub fn new(tools: Option<&Value>, tool_choice: &Value, include_reasoning: bool) -> DsParser {
        let mut names = HashSet::new();
        let mut any = false;
        if let Some(Value::Array(ts)) = tools {
            any = !ts.is_empty();
            for t in ts {
                if let Some(n) = crate::dschat::tool_name(t) {
                    names.insert(n);
                }
            }
        }
        DsParser {
            tools: names,
            parse_tools: any && tool_choice.as_str() != Some("none"),
            include_reasoning,
            mode: Mode::Reasoning,
            buf: String::new(),
            held: String::new(),
            blank: true,
            fence: false,
            ticks: 0,
            blk: String::new(),
            pre_raw: String::new(),
            n_calls: 0,
        }
    }

    /// The reply starts as the answer: its prompt closed the think block (thinking off).
    pub fn content_first(mut self) -> DsParser {
        self.mode = Mode::Content;
        self
    }

    /// A live block has closed. The generation can stop.
    pub fn call_closed(&self) -> bool {
        self.mode == Mode::Done
    }

    // ---- core ---------------------------------------------------------------------------------------------------

    fn emit_content(&mut self, t: &str, out: &mut Out) {
        if t.is_empty() {
            return;
        }
        for c in t.chars() {
            self.ticks = if c == '`' { self.ticks + 1 } else { 0 };
            if self.ticks == 3 {
                self.fence = !self.fence;
            }
            if !is_py_space(c) {
                self.blank = false;
            }
        }
        if !self.parse_tools {
            out.content.push_str(t);
            return;
        }
        let all = std::mem::take(&mut self.held) + t;
        let body = all.trim_end_matches('\n').len();
        out.content.push_str(&all[..body]);
        self.held = all[body..].to_string();
    }

    fn flush_held(&mut self, out: &mut Out) {
        out.content.push_str(&std::mem::take(&mut self.held));
    }

    fn feed(&mut self, text: &str, finished: bool) -> Out {
        let mut out = Out::default();
        if self.mode == Mode::Done {
            return out;
        }
        if self.mode == Mode::Block {
            self.blk.push_str(text);
        } else {
            self.buf.push_str(text);
        }
        loop {
            match self.mode {
                Mode::Done => break,
                Mode::Reasoning => {
                    if let Some(p) = self.buf.find(THINK_CLOSE) {
                        out.reasoning.push_str(&self.buf[..p]);
                        self.buf.drain(..p + THINK_CLOSE.len());
                        self.mode = Mode::Content;
                        continue;
                    }
                    let keep = if finished { 0 } else { partial_tag(&self.buf, THINK_CLOSE) };
                    let n = self.buf.len() - keep;
                    out.reasoning.push_str(&self.buf[..n]);
                    self.buf.drain(..n);
                    break;
                }
                Mode::Content => {
                    if !self.scan_content(finished, &mut out) {
                        break;
                    }
                }
                Mode::Block => {
                    if !self.step_block(finished, &mut out) {
                        break;
                    }
                }
            }
        }
        if finished && self.mode != Mode::Done {
            let rest = std::mem::take(&mut self.buf);
            if self.mode == Mode::Reasoning {
                out.reasoning.push_str(&rest);
            } else {
                self.emit_content(&rest, &mut out);
                self.flush_held(&mut out);
            }
        }
        out
    }

    /// Answer text. Returns true if the mode changed and the loop should go on.
    fn scan_content(&mut self, finished: bool, out: &mut Out) -> bool {
        if !self.parse_tools {
            let t = std::mem::take(&mut self.buf);
            self.emit_content(&t, out);
            return false;
        }
        let Some(p) = self.buf.find(CALLS_OPEN) else {
            let keep = if finished { 0 } else { partial_tag(&self.buf, CALLS_OPEN) };
            let n = self.buf.len() - keep;
            let t: String = self.buf.drain(..n).collect();
            self.emit_content(&t, out);
            return false;
        };
        let pre: String = self.buf.drain(..p).collect();
        self.buf.drain(..CALLS_OPEN.len());
        let combined = std::mem::take(&mut self.held) + &pre;
        let newlines = combined.len() - combined.trim_end_matches('\n').len();
        let blank = self.blank && combined.chars().all(is_py_space);
        if newlines < 2 && !blank {
            // not preceded by a blank line: the opener is text
            self.emit_content(&combined, out);
            self.emit_content(CALLS_OPEN, out);
            return true;
        }
        let cut = if blank { 0 } else { combined.len() - 2 };
        self.emit_content(&combined[..cut], out);
        if self.fence {
            self.emit_content(&combined[cut..], out);
            self.emit_content(CALLS_OPEN, out);
            return true;
        }
        self.flush_held(out); // the answer's own trailing newlines, when more than the blank line preceded the opener
        self.pre_raw = format!("{}{CALLS_OPEN}", &combined[cut..]);
        self.blk = std::mem::take(&mut self.buf);
        self.mode = Mode::Block;
        true
    }

    /// The block in progress: accept invokes as they complete. Returns true if the mode changed.
    fn step_block(&mut self, finished: bool, out: &mut Out) -> bool {
        loop {
            let ws = self.blk.len() - self.blk.trim_start_matches(is_py_space).len();
            let r = &self.blk[ws..];
            if r.starts_with(CALLS_CLOSE) {
                if self.n_calls == 0 {
                    return self.abort_block(out);
                }
                self.blk.clear();
                self.mode = Mode::Done;
                return true;
            }
            if r.starts_with(INVOKE_NAME) {
                match parse_invoke(r, &self.tools) {
                    Parsed::Ok((used, name, args)) => {
                        out.calls.push(ToolDelta {
                            index: self.n_calls,
                            id: Some(tool_call_id()),
                            name: Some(name),
                            arguments: Some(args),
                        });
                        self.n_calls += 1;
                        self.pre_raw.clear();
                        self.blk.drain(..ws + used);
                        continue;
                    }
                    Parsed::Bad => return self.abort_block(out),
                    Parsed::Need => {}
                }
            } else if !(r.is_empty() || INVOKE_NAME.starts_with(r) || CALLS_CLOSE.starts_with(r)) {
                return self.abort_block(out);
            }
            // undecided: wait for more, unless the reply is over
            if finished {
                if self.n_calls > 0 && r.trim_matches(is_py_space).is_empty() {
                    self.blk.clear();
                    self.mode = Mode::Done;
                    return true;
                }
                return self.abort_block(out);
            }
            return false;
        }
    }

    /// The block is not a call: its text goes back to the answer and scanning resumes after the opener.
    fn abort_block(&mut self, out: &mut Out) -> bool {
        let raw = std::mem::take(&mut self.pre_raw);
        self.emit_content(&raw, out);
        self.buf = std::mem::take(&mut self.blk);
        self.mode = Mode::Content;
        true
    }

    // ---- streaming API ------------------------------------------------------------------------------------------

    /// Feed newly decoded text; the delta's fields, or None if there is nothing to send yet.
    pub fn delta(&mut self, text: &str, finished: bool) -> Option<Delta> {
        let o = self.feed(text, finished);
        let d = Delta {
            content: (!o.content.is_empty()).then_some(o.content),
            reasoning: (!o.reasoning.is_empty() && self.include_reasoning).then_some(o.reasoning),
            tool_calls: o.calls,
        };
        (d.content.is_some() || d.reasoning.is_some() || !d.tool_calls.is_empty()).then_some(d)
    }

    // ---- non-streaming API --------------------------------------------------------------------------------------

    /// On a fresh parser: the whole output at once.
    pub fn parse_full(&mut self, text: &str) -> FullParse {
        let o = self.feed(text, true);
        let calls: Vec<(String, String, String)> = o
            .calls
            .into_iter()
            .map(|c| (c.id.unwrap_or_else(tool_call_id), c.name.unwrap_or_default(), c.arguments.unwrap_or_default()))
            .collect();
        let content = if calls.is_empty() { o.content } else { o.content.trim_matches(is_py_space).to_string() };
        FullParse {
            reasoning: (!o.reasoning.is_empty()).then_some(o.reasoning),
            content: (!content.is_empty()).then_some(content),
            calls,
        }
    }
}

/// The length of the longest proper prefix of `tag` that `text` ends with: bytes that may be the start of the tag.
fn partial_tag(text: &str, tag: &str) -> usize {
    let (t, g) = (text.as_bytes(), tag.as_bytes());
    (1..g.len().min(t.len() + 1)).rev().find(|&n| t.ends_with(&g[..n])).unwrap_or(0)
}

/// One `<｜DSML｜ invoke name="f">…</｜DSML｜ invoke>` at the start of `s`: (bytes used, name, arguments as JSON text).
fn parse_invoke(s: &str, known: &HashSet<String>) -> Parsed<(usize, String, String)> {
    let rest = &s[INVOKE_NAME.len()..];
    let Some(q) = rest.find("\">") else {
        return if rest.contains(['\n', '<']) { Parsed::Bad } else { Parsed::Need };
    };
    let name = &rest[..q];
    if name.contains(['\n', '<']) || !known.contains(name) {
        return Parsed::Bad;
    }
    let mut at = INVOKE_NAME.len() + q + 2;
    let mut args: Vec<(String, String)> = Vec::new();
    loop {
        at += s[at..].len() - s[at..].trim_start_matches(is_py_space).len();
        let r = &s[at..];
        if r.starts_with(INVOKE_CLOSE) {
            at += INVOKE_CLOSE.len();
            break;
        }
        if !r.starts_with(PARAM_NAME) {
            return if r.is_empty() || PARAM_NAME.starts_with(r) || INVOKE_CLOSE.starts_with(r) {
                Parsed::Need
            } else {
                Parsed::Bad
            };
        }
        let body = &r[PARAM_NAME.len()..];
        let Some(k) = body.find(PARAM_STRING) else {
            return if body.contains(['\n', '<']) { Parsed::Bad } else { Parsed::Need };
        };
        let key = &body[..k];
        if key.contains(['\n', '<']) || args.iter().any(|(a, _)| a == key) {
            return Parsed::Bad;
        }
        let after = &body[k + PARAM_STRING.len()..];
        let (is_str, skip) = if after.starts_with("true\">") {
            (true, 6)
        } else if after.starts_with("false\">") {
            (false, 7)
        } else if "true\">".starts_with(after) || "false\">".starts_with(after) {
            return Parsed::Need;
        } else {
            return Parsed::Bad;
        };
        let val = &after[skip..];
        let Some(e) = val.find(PARAM_CLOSE) else { return Parsed::Need };
        let value = &val[..e];
        let json = if is_str || serde_json::from_str::<Value>(value).is_err() {
            // a value declared non-string that is not JSON still has to leave as valid JSON
            let mut b = Vec::new();
            pyjson::string(&mut b, value);
            String::from_utf8(b).unwrap_or_default()
        } else {
            value.to_string()
        };
        args.push((key.to_string(), json));
        at += (r.len() - body.len()) + k + PARAM_STRING.len() + skip + e + PARAM_CLOSE.len();
    }
    let mut j = Vec::from(b"{".as_slice());
    for (i, (k, v)) in args.iter().enumerate() {
        if i > 0 {
            j.extend_from_slice(b", ");
        }
        pyjson::string(&mut j, k);
        j.extend_from_slice(b": ");
        j.extend_from_slice(v.as_bytes());
    }
    j.push(b'}');
    Parsed::Ok((at, name.to_string(), String::from_utf8(j).unwrap_or_default()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tools() -> Value {
        json!([
            {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "fs::read", "parameters": {"type": "object"}}}
        ])
    }

    fn inv(name: &str, params: &[(&str, bool, &str)]) -> String {
        let mut s = format!("<｜DSML｜ invoke name=\"{name}\">\n");
        for (k, is_str, v) in params {
            s.push_str(&format!("<｜DSML｜ parameter name=\"{k}\" string=\"{is_str}\">{v}</｜DSML｜ parameter>\n"));
        }
        s.push_str("</｜DSML｜ invoke>");
        s
    }

    fn block(invokes: &[String]) -> String {
        format!("<｜DSML｜ calls>\n{}\n</｜DSML｜ calls>", invokes.join("\n"))
    }

    fn weather_call() -> String {
        block(&[inv("get_weather", &[("city", true, "Paris"), ("days", false, "3")])])
    }

    /// (reasoning, content, [(name, arguments)]) from one feed of the whole reply.
    type Got = (String, String, Vec<(String, String)>);

    fn whole(p: &mut DsParser, text: &str) -> Got {
        let o = p.feed(text, true);
        (o.reasoning, o.content, o.calls.into_iter().map(|c| (c.name.unwrap(), c.arguments.unwrap())).collect())
    }

    fn chunked(mut p: DsParser, chunks: &[&str]) -> Got {
        let (mut r, mut c, mut calls) = (String::new(), String::new(), Vec::new());
        for (i, ch) in chunks.iter().enumerate() {
            let o = p.feed(ch, i + 1 == chunks.len());
            r.push_str(&o.reasoning);
            c.push_str(&o.content);
            calls.extend(o.calls.into_iter().map(|c| (c.name.unwrap(), c.arguments.unwrap())));
        }
        (r, c, calls)
    }

    fn think(tools: Option<&Value>) -> DsParser {
        DsParser::new(tools, &json!("auto"), true)
    }

    fn chat(tools: Option<&Value>) -> DsParser {
        DsParser::new(tools, &json!("auto"), true).content_first()
    }

    #[test]
    fn reasoning_then_answer() {
        let (r, c, calls) = whole(&mut think(None), "step one</think>The answer is 42.");
        assert_eq!((r.as_str(), c.as_str(), calls.len()), ("step one", "The answer is 42.", 0));
    }

    #[test]
    fn chat_mode_is_all_answer_and_keeps_trailing_newlines() {
        let (r, c, _) = whole(&mut chat(None), "hello\n\n");
        assert_eq!((r.as_str(), c.as_str()), ("", "hello\n\n"));
        let t = tools();
        let (_, c, _) = whole(&mut chat(Some(&t)), "hello\n\n");
        assert_eq!(c, "hello\n\n");
    }

    #[test]
    fn call_after_thinking() {
        let t = tools();
        let text = format!("plan it</think>\n\n{}", weather_call());
        let mut p = think(Some(&t));
        let (r, c, calls) = whole(&mut p, &text);
        assert_eq!((r.as_str(), c.as_str()), ("plan it", ""));
        assert_eq!(calls, vec![("get_weather".to_string(), "{\"city\": \"Paris\", \"days\": 3}".to_string())]);
        assert!(p.call_closed());
    }

    #[test]
    fn answer_text_before_the_call_loses_the_blank_line_only() {
        let t = tools();
        let (_, c, calls) = whole(&mut chat(Some(&t)), &format!("Checking.\n\n{}", weather_call()));
        assert_eq!(c, "Checking.");
        assert_eq!(calls.len(), 1);
        let (_, c, calls) = whole(&mut chat(Some(&t)), &format!("Checking.\n\n\n{}", weather_call()));
        assert_eq!((c.as_str(), calls.len()), ("Checking.\n", 1));
    }

    #[test]
    fn two_invokes_and_a_namespaced_tool() {
        let t = tools();
        let text = block(&[
            inv("get_weather", &[("city", true, "Rome")]),
            inv("fs::read", &[("path", true, "/tmp/x"), ("opts", false, "{\"a\": [1, 2]}")]),
        ]);
        let (_, _, calls) = whole(&mut chat(Some(&t)), &format!("\n\n{text}"));
        assert_eq!(calls[0], ("get_weather".to_string(), "{\"city\": \"Rome\"}".to_string()));
        assert_eq!(calls[1], ("fs::read".to_string(), "{\"path\": \"/tmp/x\", \"opts\": {\"a\": [1, 2]}}".to_string()));
    }

    #[test]
    fn string_values_are_raw_and_a_closer_inside_a_value_is_text() {
        let t = tools();
        let v = "line1\nline \"q\" </｜DSML｜ invoke> <b>";
        let text = format!("\n\n{}", block(&[inv("get_weather", &[("city", true, v)])]));
        let (_, _, calls) = whole(&mut chat(Some(&t)), &text);
        let args: Value = serde_json::from_str(&calls[0].1).unwrap();
        assert_eq!(args["city"], v);
    }

    #[test]
    fn a_non_string_value_that_is_not_json_still_leaves_as_json() {
        let t = tools();
        let text = format!("\n\n{}", block(&[inv("get_weather", &[("days", false, "three")])]));
        let (_, _, calls) = whole(&mut chat(Some(&t)), &text);
        assert_eq!(calls[0].1, "{\"days\": \"three\"}");
    }

    /// Replies that write about a call stay answer text, byte for byte.
    #[test]
    fn writing_about_a_call_is_not_a_call() {
        let t = tools();
        let call = weather_call();
        let cases: Vec<(&str, String, Option<&Value>)> = vec![
            ("fenced", format!("Write:\n\n```\n{call}\n```\nDone."), Some(&t)),
            ("fenced after a blank line", format!("Write:\n\n```\n\n{call}\n```\nDone."), Some(&t)),
            ("inline prose", format!("Use the {call} block."), Some(&t)),
            ("unknown tool", format!("Try:\n\n{}", block(&[inv("launch_missiles", &[("x", true, "1")])])), Some(&t)),
            ("placeholder from the instructions", "The format is:\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"$TOOL_NAME\">\n<｜DSML｜ parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜ parameter>\n...\n</｜DSML｜ invoke>\n</｜DSML｜ calls>".to_string(), Some(&t)),
            ("no tools offered", format!("Sure:\n\n{call}"), None),
            ("unclosed at the end", "Sure:\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n<｜DSML｜ parameter name=\"city\" str".to_string(), Some(&t)),
            ("empty block", "Sure:\n\n<｜DSML｜ calls>\n</｜DSML｜ calls>".to_string(), Some(&t)),
            ("duplicate parameter", "Sure:\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n<｜DSML｜ parameter name=\"a\" string=\"true\">1</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"a\" string=\"true\">2</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>".to_string(), Some(&t)),
            ("garbage inside", "Sure:\n\n<｜DSML｜ calls>\nhello there".to_string(), Some(&t)),
        ];
        for (name, text, tl) in cases {
            let mut p = chat(tl);
            let (_, c, calls) = whole(&mut p, &text);
            assert_eq!(c, text, "{name}");
            assert!(calls.is_empty() && !p.call_closed(), "{name}");
        }
        // tool_choice none turns calls off even when tools are offered
        let mut p = DsParser::new(Some(&t), &json!("none"), true).content_first();
        let text = format!("Sure:\n\n{}", weather_call());
        assert_eq!(whole(&mut p, &text).1, text);
    }

    #[test]
    fn a_call_in_the_reasoning_is_reasoning() {
        let t = tools();
        let text = format!("maybe\n\n{}\nno</think>Final.", weather_call());
        let (r, c, calls) = whole(&mut think(Some(&t)), &text);
        assert_eq!(r, format!("maybe\n\n{}\nno", weather_call()));
        assert_eq!((c.as_str(), calls.len()), ("Final.", 0));
    }

    #[test]
    fn nothing_after_the_closer_is_read() {
        let t = tools();
        let mut p = chat(Some(&t));
        let o = p.feed(&format!("\n\n{}", weather_call()), false);
        assert_eq!(o.calls.len(), 1);
        assert!(p.call_closed());
        let junk = p.feed(&"</function>".repeat(100), false);
        assert!(junk.content.is_empty() && junk.calls.is_empty());
    }

    #[test]
    fn a_block_cut_off_after_a_whole_invoke_keeps_the_call() {
        let t = tools();
        let text = format!("\n\n<｜DSML｜ calls>\n{}\n", inv("get_weather", &[("city", true, "Oslo")]));
        let (_, c, calls) = whole(&mut chat(Some(&t)), &text);
        assert_eq!((c.as_str(), calls.len()), ("", 1));
    }

    /// Every way of cutting a reply in two, and one character at a time, gives the same totals as one feed.
    #[test]
    fn chunking_does_not_change_the_result() {
        let t = tools();
        let replies = [
            format!("Let me look.\n\nplan</think>Checking.\n\n{}", weather_call()),
            format!("plan</think>\n\n{}", block(&[inv("get_weather", &[("city", true, "日本語\n</think>")]), inv("fs::read", &[("p", true, "a")])])),
            "plan</think>The answer: ```\n\n<｜DSML｜ calls>\n```\n\nmore\n\n".to_string(),
            "short</think>Use <｜DSML｜ calls> to call.\n\nUnknown:\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"nope\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>\n".to_string(),
            "no thinking close at all\n\n".to_string(),
            "r</think>".to_string(),
            "r</think>\n\n\n\n<｜DSML｜ calls>\n<｜DSML｜ inv".to_string(),
        ];
        for text in &replies {
            let want = whole(&mut think(Some(&t)), text);
            let chars: Vec<String> = text.chars().map(String::from).collect();
            let singles: Vec<&str> = chars.iter().map(String::as_str).collect();
            assert_eq!(chunked(think(Some(&t)), &singles), want, "one char at a time: {text:?}");
            for (i, _) in text.char_indices().skip(1) {
                assert_eq!(chunked(think(Some(&t)), &[&text[..i], &text[i..]]), want, "split at {i}: {text:?}");
            }
            // chat mode too
            let want = whole(&mut chat(Some(&t)), text);
            assert_eq!(chunked(chat(Some(&t)), &singles), want, "chat, one char at a time: {text:?}");
        }
    }

    #[test]
    fn parse_full_matches_the_stream() {
        let t = tools();
        let text = format!("why</think>\n\n{}", weather_call());
        let f = think(Some(&t)).parse_full(&text);
        assert_eq!(f.reasoning.as_deref(), Some("why"));
        assert_eq!(f.content, None);
        assert_eq!(f.calls.len(), 1);
        assert_eq!(
            (f.calls[0].1.as_str(), f.calls[0].2.as_str()),
            ("get_weather", "{\"city\": \"Paris\", \"days\": 3}")
        );
        assert!(f.calls[0].0.starts_with("chatcmpl-tool-"));
    }

    #[test]
    fn reasoning_can_be_left_out() {
        let mut p = DsParser::new(None, &json!("auto"), false);
        let d = p.delta("hidden</think>shown", true).unwrap();
        assert_eq!((d.reasoning, d.content.as_deref()), (None, Some("shown")));
    }
}
