//! GLM 5.3 output parser: reasoning split + tool calls, a port of `g6front/glm_parser.py` (which matches vLLM 0.30
//! `--reasoning-parser glm45 --tool-call-parser glm47`). Same states, holdback and coercion rules:
//! - Output starts in REASONING (the prompt ends with `<think>`); `</think>` -> CONTENT; stray `<think>` in
//!   REASONING / `</think>` in CONTENT are swallowed; `<think>` in CONTENT re-enters REASONING.
//! - `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>` -> {name, arguments JSON}; values
//!   coerced by the tool's JSON schema (null > integer > number > boolean > object > array > string).
//! - Streaming: partial tags held back; trailing whitespace of reasoning held until more reasoning arrives;
//!   whitespace-only content held until non-whitespace content arrives.
//!
//! Tags are ASCII, so all scanning is on bytes; text is only ever cut at tag boundaries.

use serde_json::{Map, Value};

use crate::pyjson::{self, is_py_space, py_rstrip, py_strip};

const THINK_START: &str = "<think>";
const THINK_END: &str = "</think>";
const TOOL_START: &str = "<tool_call>";
const TOOL_END: &str = "</tool_call>";
const ARG_KEY: &str = "<arg_key>";

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Reasoning,
    Content,
    Tool,
}

#[derive(Default)]
struct Slot {
    body: String,
    name: Option<String>,
    id: Option<String>,
    done_pairs: usize,
    streamed: String,
    valid: bool,
}

/// One tool-call delta, fields in the order Python's coalesced dict has them.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolDelta {
    pub index: usize,
    pub id: Option<String>,
    pub name: Option<String>,
    pub arguments: Option<String>,
}

#[derive(Default, Debug, PartialEq)]
pub struct Delta {
    pub content: Option<String>,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ToolDelta>,
}

pub struct FullParse {
    pub reasoning: Option<String>,
    pub content: Option<String>,
    pub calls: Vec<(String, String, String)>, // id, name, arguments
}

pub fn tool_call_id() -> String {
    format!("chatcmpl-tool-{}", crate::util::hex_id())
}

pub struct GlmParser {
    tools: Map<String, Value>, // name -> function object
    parse_tools: bool,
    pub include_reasoning: bool,
    state: State,
    buf: String,
    slots: Vec<Slot>,
    reasoning_ended: bool,
    deferred_reasoning: String,
    deferred_content: String,
    content_has_nonws: bool,
}

impl GlmParser {
    pub fn new(tools: Option<&Value>, tool_choice: &Value, include_reasoning: bool) -> GlmParser {
        let mut map = Map::new();
        let mut any = false;
        if let Some(Value::Array(ts)) = tools {
            any = !ts.is_empty();
            for t in ts {
                if let Some(f) = t.get("function") {
                    if let Some(n) = f.get("name").and_then(Value::as_str) {
                        map.insert(n.to_string(), f.clone());
                    }
                }
            }
        }
        GlmParser {
            tools: map,
            parse_tools: any && tool_choice.as_str() != Some("none"),
            include_reasoning,
            state: State::Reasoning,
            buf: String::new(),
            slots: Vec::new(),
            reasoning_ended: false,
            deferred_reasoning: String::new(),
            deferred_content: String::new(),
            content_has_nonws: false,
        }
    }

    fn terms(&self) -> &'static [&'static str] {
        match (self.state, self.parse_tools) {
            (State::Tool, _) => &[TOOL_END],
            (_, true) => &[THINK_START, THINK_END, TOOL_START],
            (_, false) => &[THINK_START, THINK_END],
        }
    }

    // ---- core ---------------------------------------------------------------------------------------------------

    fn feed(&mut self, text: &str, final_: bool) -> (String, String, Vec<ToolDelta>) {
        let (mut reasoning, mut content, mut tools) = (String::new(), String::new(), Vec::new());
        self.buf.push_str(text);
        let mut pos = 0; // consumed prefix of buf
        loop {
            if pos >= self.buf.len() {
                break;
            }
            let terms = self.terms();
            let rest = &self.buf[pos..];
            let mut best: Option<(usize, &'static str)> = None;
            for &tag in terms {
                if let Some(i) = rest.find(tag) {
                    if best.is_none_or(|(b, _)| i < b) {
                        best = Some((i, tag));
                    }
                }
            }
            if let Some((i, tag)) = best {
                if i > 0 {
                    let t = self.buf[pos..pos + i].to_string();
                    self.on_text(&t, &mut reasoning, &mut content, &mut tools);
                }
                pos += i + tag.len();
                self.on_term(tag, &mut tools);
                continue;
            }
            let mut keep = 0;
            if !final_ {
                let rb = rest.as_bytes();
                for &tag in terms {
                    let tb = tag.as_bytes();
                    for n in (1..=(tb.len() - 1).min(rb.len())).rev() {
                        if rb.ends_with(&tb[..n]) {
                            keep = keep.max(n);
                            break;
                        }
                    }
                }
            }
            let end = self.buf.len() - keep;
            if end > pos {
                let t = self.buf[pos..end].to_string();
                self.on_text(&t, &mut reasoning, &mut content, &mut tools);
            }
            pos = end;
            break;
        }
        self.buf.drain(..pos);
        if final_ && self.state == State::Tool && !self.slots.is_empty() {
            self.tool_end(&mut tools); // unterminated call at end of output
        }
        (reasoning, content, tools)
    }

    fn on_text(&mut self, t: &str, reasoning: &mut String, content: &mut String, tools: &mut Vec<ToolDelta>) {
        match self.state {
            State::Reasoning => reasoning.push_str(t),
            State::Content => content.push_str(t),
            State::Tool => self.tool_text(t, tools),
        }
    }

    fn on_term(&mut self, tag: &str, tools: &mut Vec<ToolDelta>) {
        match self.state {
            State::Reasoning => {
                if tag == THINK_END {
                    self.state = State::Content;
                    self.reasoning_ended = true;
                } else if tag == TOOL_START {
                    self.state = State::Tool;
                    self.reasoning_ended = true;
                    self.slots.push(Slot::default());
                }
            }
            State::Content => {
                if tag == THINK_START {
                    self.state = State::Reasoning;
                } else if tag == TOOL_START {
                    self.state = State::Tool;
                    self.slots.push(Slot::default());
                }
            }
            State::Tool => {
                self.tool_end(tools);
                self.state = State::Content;
            }
        }
    }

    fn name_valid(&self, name: &str) -> bool {
        !name.is_empty() && (self.tools.is_empty() || self.tools.contains_key(name))
    }

    fn tool_text(&mut self, v: &str, deltas: &mut Vec<ToolDelta>) {
        let idx = self.slots.len() - 1;
        self.slots[idx].body.push_str(v);
        if self.slots[idx].name.is_none() {
            if let Some(at) = self.slots[idx].body.find(ARG_KEY) {
                let name = py_strip(&self.slots[idx].body[..at]).to_string();
                let valid = self.name_valid(&name);
                let s = &mut self.slots[idx];
                s.valid = valid;
                s.name = Some(name.clone());
                if valid {
                    let id = tool_call_id();
                    s.id = Some(id.clone());
                    deltas.push(ToolDelta { index: idx, id: Some(id), name: Some(name), arguments: None });
                }
            }
        }
        let s = &self.slots[idx];
        if s.name.is_some() && s.valid {
            let pairs = pairs(&s.body);
            if pairs.len() > s.done_pairs {
                let json = self.args_json(s.name.as_deref().unwrap(), &pairs);
                let prefix = &json[..json.len() - 1]; // without the closing brace
                let s = &mut self.slots[idx];
                s.done_pairs = pairs.len();
                if prefix.starts_with(s.streamed.as_str()) && prefix != s.streamed {
                    deltas.push(ToolDelta {
                        index: idx,
                        id: None,
                        name: None,
                        arguments: Some(prefix[s.streamed.len()..].to_string()),
                    });
                    s.streamed = prefix.to_string();
                }
            }
        }
    }

    fn tool_end(&mut self, deltas: &mut Vec<ToolDelta>) {
        let idx = self.slots.len() - 1;
        if self.slots[idx].name.is_none() {
            let name = py_strip(&self.slots[idx].body).to_string();
            let valid = self.name_valid(&name);
            let args = if valid { Some(self.args_json(&name, &[])) } else { None };
            let s = &mut self.slots[idx];
            s.valid = valid;
            s.name = Some(name.clone());
            if let Some(args) = args {
                let id = tool_call_id();
                s.id = Some(id.clone());
                deltas.push(ToolDelta { index: idx, id: Some(id), name: Some(name), arguments: Some(args.clone()) });
                s.streamed = args;
            }
            return;
        }
        if self.slots[idx].valid {
            let s = &self.slots[idx];
            let final_ = self.args_json(s.name.as_deref().unwrap(), &pairs(&s.body));
            let s = &mut self.slots[idx];
            if final_.starts_with(s.streamed.as_str()) && final_ != s.streamed {
                deltas.push(ToolDelta {
                    index: idx,
                    id: None,
                    name: None,
                    arguments: Some(final_[s.streamed.len()..].to_string()),
                });
            }
            s.streamed = final_;
        }
    }

    fn args_json(&self, name: &str, pairs: &[(String, String)]) -> String {
        let mut params: Map<String, Value> = Map::new();
        for (k, v) in pairs {
            params.insert(k.clone(), Value::from(v.as_str()));
        }
        let props = self
            .tools
            .get(name)
            .and_then(|f| f.get("parameters"))
            .and_then(Value::as_object)
            .and_then(|p| p.get("properties"))
            .and_then(Value::as_object);
        let mut out = Vec::with_capacity(64);
        out.push(b'{');
        for (i, (k, v)) in params.iter().enumerate() {
            if i > 0 {
                out.extend_from_slice(b", ");
            }
            pyjson::string(&mut out, k);
            out.extend_from_slice(b": ");
            match props.and_then(|p| p.get(k)).filter(|s| s.is_object()) {
                Some(schema) => coerce_str(&mut out, v.as_str().unwrap(), &schema_types(schema)),
                None => pyjson::value(&mut out, v),
            }
        }
        out.push(b'}');
        String::from_utf8(out).unwrap()
    }

    // ---- streaming API ------------------------------------------------------------------------------------------

    /// Feed newly decoded text; returns the DeltaMessage fields (None if empty).
    pub fn delta(&mut self, text: &str, finished: bool) -> Option<Delta> {
        let (mut reasoning, content, tools) = self.feed(text, finished);
        let valid_tools = self.slots.iter().any(|s| s.valid);
        if !reasoning.is_empty() {
            let comb = std::mem::take(&mut self.deferred_reasoning) + &reasoning;
            let trimmed = py_rstrip(&comb).len();
            self.deferred_reasoning = comb[trimmed..].to_string();
            reasoning = comb[..trimmed].to_string();
        } else if self.reasoning_ended {
            self.deferred_reasoning.clear();
        }
        let mut content = std::mem::take(&mut self.deferred_content) + &content;
        if !content.is_empty() && !self.content_has_nonws {
            if !content.chars().all(is_py_space) {
                self.content_has_nonws = true;
            } else if valid_tools {
                content.clear();
            } else if !finished {
                self.deferred_content = std::mem::take(&mut content);
            }
        }
        let d = Delta {
            content: (!content.is_empty()).then_some(content),
            reasoning: (!reasoning.is_empty() && self.include_reasoning).then_some(reasoning),
            tool_calls: coalesce(tools),
        };
        if d.content.is_none() && d.reasoning.is_none() && d.tool_calls.is_empty() {
            None
        } else {
            Some(d)
        }
    }

    // ---- non-streaming API --------------------------------------------------------------------------------------

    /// On a fresh parser: the whole output at once.
    pub fn parse_full(&mut self, text: &str) -> FullParse {
        let (r, c, _) = self.feed(text, true);
        let mut calls = Vec::new();
        for s in &self.slots {
            if let (true, Some(name)) = (s.valid, s.name.as_deref().filter(|n| !n.is_empty())) {
                let id = s.id.clone().unwrap_or_else(tool_call_id);
                calls.push((id, name.to_string(), self.args_json(name, &pairs(&s.body))));
            }
        }
        let reasoning = py_rstrip(&r);
        let content = if calls.is_empty() { c.as_str() } else { py_strip(&c) };
        FullParse {
            reasoning: (!reasoning.is_empty()).then(|| reasoning.to_string()),
            content: (!content.is_empty()).then(|| content.to_string()),
            calls,
        }
    }
}

fn coalesce(deltas: Vec<ToolDelta>) -> Vec<ToolDelta> {
    let mut merged: Vec<ToolDelta> = Vec::new();
    for d in deltas {
        match merged.iter_mut().find(|m| m.index == d.index) {
            None => merged.push(d),
            Some(m) => {
                if m.id.is_none() {
                    m.id = d.id;
                }
                if m.name.is_none() {
                    m.name = d.name;
                }
                if let Some(a) = d.arguments {
                    m.arguments.get_or_insert_with(String::new).push_str(&a);
                }
            }
        }
    }
    merged
}

/// `re.finditer(r'<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>', body, re.DOTALL)` -> (key.strip(), value)
fn pairs(body: &str) -> Vec<(String, String)> {
    const K0: &str = "<arg_key>";
    const K1: &str = "</arg_key>";
    const V0: &str = "<arg_value>";
    const V1: &str = "</arg_value>";
    let mut out = Vec::new();
    let mut at = 0;
    while let Some(i) = body[at..].find(K0) {
        let start = at + i;
        let kstart = start + K0.len();
        let mut found = None;
        // the lazy key group backtracks over every later </arg_key> until \s*<arg_value>...</arg_value> follows
        let mut search = kstart;
        while let Some(j) = body[search..].find(K1) {
            let kend = search + j;
            let after = &body[kend + K1.len()..];
            let ws = after.len() - after.trim_start_matches(is_py_space).len();
            let vstart = kend + K1.len() + ws;
            if body[vstart..].starts_with(V0) {
                if let Some(e) = body[vstart + V0.len()..].find(V1) {
                    let vend = vstart + V0.len() + e;
                    found = Some((kend, vstart + V0.len(), vend));
                    break;
                }
            }
            search = kend + 1;
        }
        match found {
            Some((kend, vs, ve)) => {
                out.push((py_strip(&body[kstart..kend]).to_string(), body[vs..ve].to_string()));
                at = ve + V1.len();
            }
            None => at = start + 1,
        }
    }
    out
}

// ---- schema coercion (vLLM tool_parsers/utils.py semantics) --------------------------------------------------------

fn schema_types(schema: &Value) -> Vec<String> {
    let mut types: Vec<String> = Vec::new();
    let add = |t: &str, types: &mut Vec<String>| {
        if !types.iter().any(|x| x == t) {
            types.push(t.to_string());
        }
    };
    let Some(o) = schema.as_object() else { return vec!["string".into()] };
    match o.get("type") {
        Some(Value::String(t)) => add(t, &mut types),
        Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).for_each(|t| add(t, &mut types)),
        _ => {}
    }
    if let Some(Value::Array(e)) = o.get("enum") {
        for v in e {
            let t = match v {
                Value::Null => "null",
                Value::Bool(_) => "boolean",
                Value::Number(n) if n.is_i64() || n.is_u64() => "integer",
                Value::Number(_) => "number",
                Value::String(_) => "string",
                Value::Array(_) => "array",
                Value::Object(_) => "object",
            };
            add(t, &mut types);
        }
    }
    for k in ["anyOf", "oneOf", "allOf"] {
        if let Some(Value::Array(subs)) = o.get(k) {
            for s in subs {
                for t in schema_types(s) {
                    add(&t, &mut types);
                }
            }
        }
    }
    if types.is_empty() {
        types.push("string".into());
    }
    types
}

fn alias(t: &str) -> String {
    let t = py_strip(t).to_lowercase();
    match t.as_str() {
        "int" => "integer".into(),
        "float" => "number".into(),
        "bool" => "boolean".into(),
        "dict" => "object".into(),
        "list" => "array".into(),
        "str" => "string".into(),
        _ => t,
    }
}

/// Unicode decimal digits (category Nd): the code point of each block's zero (Unicode 16, CPython 3.14's table).
const ND_ZEROS: [u32; 76] = [
    0x30, 0x660, 0x6F0, 0x7C0, 0x966, 0x9E6, 0xA66, 0xAE6, 0xB66, 0xBE6, 0xC66, 0xCE6, 0xD66, 0xDE6, 0xE50, 0xED0,
    0xF20, 0x1040, 0x1090, 0x17E0, 0x1810, 0x1946, 0x19D0, 0x1A80, 0x1A90, 0x1B50, 0x1BB0, 0x1C40, 0x1C50, 0xA620,
    0xA8D0, 0xA900, 0xA9D0, 0xA9F0, 0xAA50, 0xABF0, 0xFF10, 0x104A0, 0x10D30, 0x10D40, 0x11066, 0x110F0, 0x11136,
    0x111D0, 0x112F0, 0x11450, 0x114D0, 0x11650, 0x116C0, 0x116D0, 0x116DA, 0x11730, 0x118E0, 0x11950, 0x11BF0,
    0x11C50, 0x11D50, 0x11DA0, 0x11F50, 0x16130, 0x16A60, 0x16AC0, 0x16B50, 0x16D70, 0x1CCF0, 0x1D7CE, 0x1D7D8,
    0x1D7E2, 0x1D7EC, 0x1D7F6, 0x1E140, 0x1E2F0, 0x1E4F0, 0x1E5F1, 0x1E950, 0x1FBF0,
];

/// What CPython's int()/float() parse after `_PyUnicode_TransformDecimalAndSpaceToASCII`: Unicode digits become
/// ASCII digits, Unicode whitespace a space; any other non-ASCII character makes the text invalid.
fn py_ascii_numeric(s: &str) -> Option<String> {
    s.chars()
        .map(|c| {
            if c.is_ascii() {
                return Some(c);
            }
            if is_py_space(c) {
                return Some(' ');
            }
            let u = c as u32;
            let z = ND_ZEROS.iter().rev().find(|&&z| z <= u)?;
            (u - z < 10).then(|| char::from(b'0' + (u - z) as u8))
        })
        .collect()
}

/// Python `int(s)`: surrounding whitespace, a sign, digits with single underscores between digits.
/// Returns the canonical decimal text (arbitrary size, like a Python int).
fn py_int(s: &str) -> Option<String> {
    let s = py_ascii_numeric(s)?;
    let s = py_strip(&s);
    let (neg, digits) = match s.as_bytes().first()? {
        b'-' => (true, &s[1..]),
        b'+' => (false, &s[1..]),
        _ => (false, s),
    };
    let b = digits.as_bytes();
    if b.is_empty() || b[0] == b'_' || b[b.len() - 1] == b'_' {
        return None;
    }
    let mut clean = String::with_capacity(b.len());
    for (i, &c) in b.iter().enumerate() {
        match c {
            b'0'..=b'9' => clean.push(c as char),
            b'_' if b[i - 1] != b'_' => {}
            _ => return None,
        }
    }
    let trimmed = clean.trim_start_matches('0');
    if trimmed.is_empty() {
        return Some("0".into());
    }
    Some(if neg { format!("-{trimmed}") } else { trimmed.to_string() })
}

/// Python `float(s)`.
fn py_float(s: &str) -> Option<f64> {
    let s = py_ascii_numeric(s)?;
    let s = py_strip(&s);
    let s: std::borrow::Cow<str> = if s.contains('_') {
        let b = s.as_bytes();
        for (i, &c) in b.iter().enumerate() {
            if c == b'_' && !(i > 0 && b[i - 1].is_ascii_digit() && i + 1 < b.len() && b[i + 1].is_ascii_digit()) {
                return None;
            }
        }
        s.replace('_', "").into()
    } else {
        s.into()
    };
    let l = s.to_ascii_lowercase();
    let body = l.trim_start_matches(['+', '-']);
    if body.len() + 1 < l.len() {
        return None; // more than one sign
    }
    if matches!(body, "inf" | "infinity" | "nan") {
        let neg = l.starts_with('-');
        return Some(match (body, neg) {
            ("nan", _) => f64::NAN,
            (_, true) => f64::NEG_INFINITY,
            _ => f64::INFINITY,
        });
    }
    if body.is_empty() || !body.bytes().all(|c| c.is_ascii_digit() || matches!(c, b'.' | b'e' | b'+' | b'-')) {
        return None;
    }
    s.parse::<f64>().ok()
}

fn coerce_str(out: &mut Vec<u8>, value: &str, types: &[String]) {
    let types: Vec<String> = types.iter().map(|t| alias(t)).collect();
    let has = |t: &str| types.iter().any(|x| x == t);
    for cand in ["null", "integer", "number", "boolean", "object", "array", "string"] {
        if !has(cand) {
            continue;
        }
        match cand {
            "null" => {
                if value.to_lowercase() == "null" {
                    out.extend_from_slice(b"null");
                    return;
                }
            }
            "string" => {
                pyjson::string(out, value);
                return;
            }
            "integer" => {
                if let Some(i) = py_int(value) {
                    out.extend_from_slice(i.as_bytes());
                    return;
                }
            }
            "number" => {
                if let Some(f) = py_float(value) {
                    if f.is_finite() {
                        if f == f.trunc() {
                            out.extend_from_slice(pyjson::integral_float_text(f).as_bytes());
                        } else {
                            pyjson::float(out, f);
                        }
                        return;
                    }
                }
            }
            "boolean" => {
                let v = value.to_lowercase();
                match py_strip(&v) {
                    "true" | "1" => {
                        out.extend_from_slice(b"true");
                        return;
                    }
                    "false" | "0" => {
                        out.extend_from_slice(b"false");
                        return;
                    }
                    _ => {}
                }
            }
            _ => {
                if let Ok(p) = serde_json::from_str::<Value>(value) {
                    if (cand == "object" && p.is_object()) || (cand == "array" && p.is_array()) {
                        pyjson::value(out, &p);
                        return;
                    }
                }
            }
        }
    }
    pyjson::string(out, value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tools() -> Value {
        json!([{"type": "function", "function": {"name": "get_current_date", "description": null, "parameters": {}}},
               {"type": "function", "function": {"name": "get_weather", "description": null, "parameters": {
                   "type": "object", "properties": {"city": {"type": "string"}, "date": {"type": "string"},
                                                     "days": {"type": "integer"}, "metric": {"type": "boolean"}}}}}])
    }

    fn full(text: &str) -> FullParse {
        GlmParser::new(Some(&tools()), &json!("auto"), true).parse_full(text)
    }

    #[test]
    fn reasoning_and_content() {
        let r = full("Let me think.  \n</think>\n\nThe answer.");
        assert_eq!(r.reasoning.as_deref(), Some("Let me think."));
        assert_eq!(r.content.as_deref(), Some("\n\nThe answer."));
        assert!(r.calls.is_empty());
    }

    #[test]
    fn tool_call_coerced() {
        let r = full(
            "hmm</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>\
             <arg_key>days</arg_key>\n<arg_value>3</arg_value><arg_key>metric</arg_key><arg_value>True</arg_value></tool_call>",
        );
        assert_eq!(r.calls.len(), 1);
        assert_eq!(r.calls[0].1, "get_weather");
        assert_eq!(r.calls[0].2, r#"{"city": "Paris", "days": 3, "metric": true}"#);
        assert!(r.content.is_none());
    }

    #[test]
    fn stream_equals_full() {
        let text = "Think a bit.</think>Sure.<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>\
                    <arg_key>days</arg_key><arg_value>3</arg_value></tool_call>";
        let f = full(text);
        let mut p = GlmParser::new(Some(&tools()), &json!("auto"), true);
        let chars: Vec<char> = text.chars().collect();
        let (mut args, mut reasoning, mut content) = (String::new(), String::new(), String::new());
        for (i, ch) in chars.chunks(3).enumerate() {
            let s: String = ch.iter().collect();
            if let Some(d) = p.delta(&s, (i + 1) * 3 >= chars.len()) {
                reasoning += d.reasoning.as_deref().unwrap_or("");
                content += d.content.as_deref().unwrap_or("");
                for t in d.tool_calls {
                    args += t.arguments.as_deref().unwrap_or("");
                }
            }
        }
        assert_eq!(args, f.calls[0].2);
        assert_eq!(Some(reasoning), f.reasoning);
        assert_eq!(content, "Sure.");
    }

    #[test]
    fn python_int_and_float() {
        assert_eq!(py_int(" 1_000 ").as_deref(), Some("1000"));
        assert_eq!(py_int("-007").as_deref(), Some("-7"));
        assert_eq!(py_int("1__0"), None);
        assert_eq!(py_int("1.0"), None);
        assert_eq!(py_float("1e3"), Some(1000.0));
        assert_eq!(py_float(" -Infinity "), Some(f64::NEG_INFINITY));
        assert_eq!(py_float("1_0.5"), Some(10.5));
        assert_eq!(py_float("abc"), None);
        assert_eq!(py_float("--1"), None);
        assert_eq!(py_int("３").as_deref(), Some("3"));
        assert_eq!(py_int("\u{3000}-١٢\u{a0}").as_deref(), Some("-12"));
        assert_eq!(py_float("１.５"), Some(1.5));
        assert_eq!(py_int("3x"), None);
        assert_eq!(py_int("三"), None);
    }

    #[test]
    fn arg_regex_backtracks_like_python() {
        let p = pairs("<arg_key>a</arg_key>x</arg_key> <arg_value>1</arg_value>");
        assert_eq!(p, vec![("a</arg_key>x".to_string(), "1".to_string())]);
        assert!(pairs("<arg_key>a</arg_key><arg_value>1").is_empty());
    }
}
