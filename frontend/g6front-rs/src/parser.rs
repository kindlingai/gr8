//! GLM 5.3 output parser: reasoning split + tool calls, matching vLLM 0.30
//! `--reasoning-parser glm45 --tool-call-parser glm47`. Same states, holdback and coercion rules:
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
const ARG_VALUE: &str = "<arg_value>";
const ARG_VALUE_END: &str = "</arg_value>";

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Reasoning,
    Content,
}

/// How a pending run resolved; see `GlmParser::region_step`.
enum Step {
    Hold,
    Text,
    Calls,
}

/// A run of the reply that might be tool calls: from a `<tool_call>` on, held until it is decided either way.
///
/// A reply *writing about* a call and a reply *making* one are the same bytes, so nothing about the text itself tells
/// them apart -- only what follows. A call the model makes is the reply's last segment (the model stops right after
/// `</tool_call>`); a call it writes about is followed by prose, by a code fence's closing backticks, or by a `</think>`.
/// So the run is text unless the reply ends with it, and when it is text its raw bytes go back where they came from
/// instead of being dropped. That is also why a call's deltas are emitted when the run is decided rather than as it is
/// read: a call that turns out to be prose cannot be taken back from the client.
struct Region {
    state: State,  // the state at the opener: where the raw text goes if the run is not a call
    text: String,  // the raw reply text from the opener on
    at: usize,     // how much of `text` the scan has consumed
    slot0: usize,  // the first slot this run pushed
    open: bool,    // an opener whose closer has not been seen
}

#[derive(Default)]
struct Slot {
    body: String,
    name: Option<String>,
    id: Option<String>,
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
    region: Option<Region>,
    last: Option<char>,
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
            region: None,
            last: None,
            reasoning_ended: false,
            deferred_reasoning: String::new(),
            deferred_content: String::new(),
            content_has_nonws: false,
        }
    }

    fn terms(&self) -> &'static [&'static str] {
        match self.parse_tools {
            true => &[THINK_START, THINK_END, TOOL_START],
            false => &[THINK_START, THINK_END],
        }
    }

    // ---- core ---------------------------------------------------------------------------------------------------

    /// Route text to where the reply is: reasoning before `</think>`, content after. `last` is the last character the
    /// reply produced anywhere, which the backtick rule reads (an opener right after a backtick is inline code).
    fn emit_text(&mut self, t: &str, state: State, reasoning: &mut String, content: &mut String) {
        if t.is_empty() {
            return;
        }
        match state {
            State::Reasoning => reasoning.push_str(t),
            State::Content => content.push_str(t),
        }
        self.last = t.chars().next_back();
    }

    fn feed(&mut self, text: &str, final_: bool) -> (String, String, Vec<ToolDelta>) {
        let (mut reasoning, mut content, mut tools) = (String::new(), String::new(), Vec::new());
        self.buf.push_str(text);
        let mut pos = 0;
        loop {
            if let Some(mut reg) = self.region.take() {
                // a pending run absorbs what arrives; nothing about it is emitted until it is decided
                reg.text.push_str(&self.buf);
                self.buf.clear();
                match self.region_step(&mut reg, final_) {
                    Step::Hold => {
                        self.region = Some(reg);
                        break;
                    }
                    Step::Text => {
                        // the run was prose: its raw text goes back where it came from, and what follows it is read
                        // again from there -- a `</think>` inside it still closes the reasoning, a fence still counts
                        let rest = reg.text.split_off(reg.at);
                        self.emit_text(&reg.text, reg.state, &mut reasoning, &mut content);
                        self.slots.truncate(reg.slot0);
                        self.state = reg.state;
                        self.buf.clear();
                        self.buf.push_str(&rest);
                        pos = 0;
                        continue;
                    }
                    Step::Calls => {
                        // the reply ended with the run: the calls are real, and their deltas go out together
                        self.emit_calls(reg.slot0, &mut tools);
                        if reg.state == State::Reasoning {
                            self.reasoning_ended = true;
                        }
                        self.state = State::Content;
                        break;
                    }
                }
            }
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
                    let st = self.state;
                    self.emit_text(&t, st, &mut reasoning, &mut content);
                }
                pos += i + tag.len();
                if tag == TOOL_START {
                    // an opener right after a backtick is code the model is writing about, not a call it makes
                    if self.last == Some('`') {
                        let st = self.state;
                        self.emit_text(TOOL_START, st, &mut reasoning, &mut content);
                    } else {
                        self.buf.drain(..pos);
                        pos = 0;
                        self.slots.push(Slot::default());
                        self.region = Some(Region {
                            state: self.state,
                            text: TOOL_START.to_string(),
                            at: TOOL_START.len(),
                            slot0: self.slots.len() - 1,
                            open: true,
                        });
                        self.last = Some('>');
                    }
                } else {
                    self.on_think(tag);
                }
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
            let end = self.buf.len().saturating_sub(keep).max(pos);
            if end > pos {
                let t = self.buf[pos..end].to_string();
                let st = self.state;
                self.emit_text(&t, st, &mut reasoning, &mut content);
            }
            pos = end;
            break; // `buf[end..]` is a partial tag: it waits for the next delta
        }
        self.buf.drain(..pos); // what was emitted or fed to a region is done with
        (reasoning, content, tools)
    }

    /// Advance a pending run as far as its text allows: `Hold` (more input may still decide it), `Text` (it was never a
    /// call: prose the model wrote about one, or a call it left unfinished or named a non-tool in), or `Calls` (it was,
    /// decided only when the reply ends with it).
    fn region_step(&mut self, reg: &mut Region, final_: bool) -> Step {
        loop {
            if reg.open {
                let closer = find_tag(&reg.text, reg.at, TOOL_END);
                let think = {
                    let a = find_tag(&reg.text, reg.at, THINK_START);
                    let b = find_tag(&reg.text, reg.at, THINK_END);
                    match (a, b) {
                        (Some(a), Some(b)) => Some(a.min(b)),
                        (a, b) => a.or(b),
                    }
                };
                match (closer, think) {
                    (Some(c), Some(t)) if t < c => return Step::Text, // the reply moved on: never a call
                    (Some(c), _) => {
                        let body = reg.text[reg.at..c].to_string();
                        reg.at = c + TOOL_END.len();
                        reg.open = false;
                        self.last = Some('>');
                        if !self.close_call(body) {
                            return Step::Text; // a name no tool of this request has: the text stays prose
                        }
                    }
                    (None, Some(_)) => return Step::Text,
                    // an unterminated opener is prose, not a call, and its text is kept
                    (None, None) => return if final_ { Step::Text } else { Step::Hold },
                }
                continue;
            }
            // between calls only another call, or whitespace, may follow: anything else proves the run was prose
            let scan = &reg.text[reg.at..];
            match earliest_nonspace(scan) {
                None => return if final_ { Step::Calls } else { Step::Hold },
                Some(n) => {
                    let rest = &scan[n..];
                    if rest.starts_with(TOOL_START) {
                        if reg.text[..reg.at + n].ends_with('`') {
                            return Step::Text; // inline code again
                        }
                        self.slots.push(Slot::default());
                        reg.at += n + TOOL_START.len();
                        reg.open = true;
                        self.last = Some('>');
                        continue;
                    }
                    if !final_ && TOOL_START.starts_with(rest) {
                        return Step::Hold; // a partial opener: wait for the rest of it
                    }
                    return Step::Text;
                }
            }
        }
    }

    /// Take the body between a call's tags: its name (everything before the first `<arg_key>`, or the whole body for a
    /// call with no arguments), and whether that name is a tool this request offered.
    fn close_call(&mut self, body: String) -> bool {
        let idx = self.slots.len() - 1;
        let name = match body.find(ARG_KEY) {
            Some(at) => py_strip(&body[..at]).to_string(),
            None => py_strip(&body).to_string(),
        };
        let valid = self.name_valid(&name);
        let s = &mut self.slots[idx];
        s.body = body;
        s.name = Some(name);
        s.valid = valid;
        valid
    }

    /// The deltas for a run's calls, one each: nothing about them was streamed while the run was pending.
    fn emit_calls(&mut self, slot0: usize, deltas: &mut Vec<ToolDelta>) {
        for i in slot0..self.slots.len() {
            if !self.slots[i].valid {
                continue;
            }
            let name = self.slots[i].name.clone().unwrap_or_default();
            let args = self.args_json(&name, &pairs(&self.slots[i].body));
            let id = self.slots[i].id.clone().unwrap_or_else(tool_call_id);
            self.slots[i].id = Some(id.clone());
            self.slots[i].streamed = args.clone();
            deltas.push(ToolDelta { index: i, id: Some(id), name: Some(name), arguments: Some(args) });
        }
        self.last = Some('>');
    }

    fn on_think(&mut self, tag: &str) {
        // A think tag separates the next opener from any preceding backtick in the raw reply.
        self.last = Some('>');
        match self.state {
            State::Reasoning => {
                if tag == THINK_END {
                    self.state = State::Content;
                    self.reasoning_ended = true;
                }
            }
            State::Content => {
                if tag == THINK_START {
                    self.state = State::Reasoning;
                }
            }
        }
    }

    fn name_valid(&self, name: &str) -> bool {
        !name.is_empty() && (self.tools.is_empty() || self.tools.contains_key(name))
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
        // a run still pending is not a call yet: only a decided one trims the whitespace-only content before it
        let valid_tools = self.region.is_none() && self.slots.iter().any(|s| s.valid);
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

/// The first `tag` at or after `from` that does not sit inside an `<arg_value>...</arg_value>`.
///
/// The format has no escape, so an argument value may hold the tag as text; that text is part of the value and does not
/// end the call (or the reply's reasoning). `from` and the result are byte indices into `text`.
fn find_tag(text: &str, from: usize, tag: &str) -> Option<usize> {
    let mut at = from;
    while let Some(i) = text[at..].find(tag) {
        let abs = at + i;
        if !inside_value(&text[..abs]) {
            return Some(abs);
        }
        at = abs + tag.len();
    }
    None
}

/// Whether `before` ends inside an `<arg_value>` that no `</arg_value>` has closed yet.
fn inside_value(before: &str) -> bool {
    match (before.rfind(ARG_VALUE), before.rfind(ARG_VALUE_END)) {
        (Some(v), Some(e)) => v > e,
        (Some(_), None) => true,
        (None, _) => false,
    }
}

/// The byte index of the first character `str.strip()` would keep, if any.
fn earliest_nonspace(t: &str) -> Option<usize> {
    t.char_indices().find(|(_, c)| !is_py_space(*c)).map(|(i, _)| i)
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
    /// Feed `text` in `chunk`-sized pieces and accumulate what a client would (ids differ per run, so compare names and
    /// arguments, as `stream_equals_full` does).
    fn streamed(text: &str, chunk: usize) -> FullParse {
        let mut p = GlmParser::new(Some(&tools()), &json!("auto"), true);
        let chars: Vec<char> = text.chars().collect();
        let (mut reasoning, mut content) = (String::new(), String::new());
        let mut calls: Vec<(String, String, String)> = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            let end = (i + chunk).min(chars.len());
            let s: String = chars[i..end].iter().collect();
            i = end;
            if let Some(d) = p.delta(&s, i >= chars.len()) {
                reasoning += d.reasoning.as_deref().unwrap_or("");
                content += d.content.as_deref().unwrap_or("");
                for t in d.tool_calls {
                    while calls.len() <= t.index {
                        calls.push((String::new(), String::new(), String::new()));
                    }
                    if let Some(n) = t.name {
                        calls[t.index].1 = n;
                    }
                    if let Some(a) = t.arguments {
                        calls[t.index].2 += &a;
                    }
                }
            }
        }
        FullParse {
            reasoning: (!reasoning.is_empty()).then_some(reasoning),
            content: (!content.is_empty()).then_some(content),
            calls,
        }
    }

    // A reply the model *writes about* a call and one it *makes* are the same bytes: what tells them apart is that a
    // call it makes is the reply's last segment. Everything below is prose, and none of it may lose text.

    /// A documented example in the answer is content, not a call (it used to end the reasoning and be promoted).
    #[test]
    fn prose_example_is_not_a_call() {
        let text = "</think>Call it like <tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>\
                    </tool_call> in your code.";
        let r = full(text);
        assert!(r.calls.is_empty());
        assert_eq!(r.reasoning, None);
        assert_eq!(
            r.content.as_deref(),
            Some("Call it like <tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call> \
                  in your code.")
        );
    }

    /// The same inside reasoning: the raw text goes back to the reasoning, which the later `</think>` still closes.
    #[test]
    fn prose_example_in_reasoning_stays_reasoning() {
        let r = full("Think about <tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call> \
                      a bit.</think>Here.");
        assert!(r.calls.is_empty());
        assert_eq!(
            r.reasoning.as_deref(),
            Some("Think about <tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call> a bit.")
        );
        assert_eq!(r.content.as_deref(), Some("Here."));
    }

    /// A backtick before a think tag is not adjacent to the tool opener after that tag.
    #[test]
    fn backtick_before_think_end_does_not_quote_a_call() {
        let text = "Use `get_weather`</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>";
        let r = full(text);
        assert_eq!(r.calls.len(), 1);
        assert_eq!(r.reasoning.as_deref(), Some("Use `get_weather`"));
        assert_eq!(r.calls[0].2, r#"{"city": "Paris"}"#);
        for chunk in [1, 2, 7, 11, 4096] {
            let s = streamed(text, chunk);
            assert_eq!(s.calls.len(), 1, "chunk {chunk}");
            assert_eq!(s.calls[0].2, r.calls[0].2);
            assert_eq!(s.reasoning, r.reasoning);
            assert_eq!(s.content, r.content);
        }
    }

    /// An opener right after a backtick is inline code, even when the reply ends with it (the trailing rule cannot tell).
    #[test]
    fn inline_code_opener_is_not_a_call() {
        let quoted = "</think>Write `<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>\
                      </tool_call>";
        let r = full(quoted);
        assert!(r.calls.is_empty());
        assert_eq!(
            r.content.as_deref(),
            Some("Write `<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>")
        );
        // without the backtick the same text is the model's call
        let plain = quoted.replacen("`<tool_call>", "<tool_call>", 1);
        let r = full(&plain);
        assert_eq!(r.calls.len(), 1);
        assert_eq!(r.calls[0].2, r#"{"city": "Paris"}"#);
    }

    /// An opener the reply never closes is prose, and the text after it is kept (`used to be swallowed whole`).
    #[test]
    fn unterminated_opener_keeps_its_text() {
        let text = "</think>The format is <tool_call>get_weather and then the arguments.";
        let r = full(text);
        assert!(r.calls.is_empty());
        assert_eq!(r.content.as_deref(), Some("The format is <tool_call>get_weather and then the arguments."));
    }

    /// A call whose name is not one of the request's tools is prose, tags and all (the text used to be dropped).
    #[test]
    fn unknown_name_keeps_its_text() {
        let text = "</think>Use <tool_call>nope<arg_key>a</arg_key><arg_value>1</arg_value></tool_call> here.";
        let r = full(text);
        assert!(r.calls.is_empty());
        assert_eq!(
            r.content.as_deref(),
            Some("Use <tool_call>nope<arg_key>a</arg_key><arg_value>1</arg_value></tool_call> here.")
        );
    }

    /// A call followed by prose is prose too (the model writes about the call, then goes on).
    #[test]
    fn a_call_followed_by_prose_is_prose() {
        let r = full("</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call> \
                      and that is how you ask.");
        assert!(r.calls.is_empty());
        assert_eq!(
            r.content.as_deref(),
            Some("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call> and that is how \
                  you ask.")
        );
    }

    /// A fenced example: the closing fence is text after the call, so the run is prose.
    #[test]
    fn fenced_example_is_prose() {
        let text = "</think>Like this:\n```\n<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>\
                    </tool_call>\n```\n";
        let r = full(text);
        assert!(r.calls.is_empty());
        assert!(r.content.as_deref().unwrap().contains("```"));
    }

    /// The format has no escape, so an argument value may hold the closing tag: it is part of the value, not the end.
    #[test]
    fn closer_inside_an_argument_value() {
        let r = full("</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>a </tool_call> \
                      b</arg_value></tool_call>");
        assert_eq!(r.calls.len(), 1);
        assert_eq!(r.calls[0].2, r#"{"city": "a </tool_call> b"}"#);
        assert_eq!(r.content, None);
    }

    /// A think tag inside an open call means the reply moved on: the "call" was reasoning, and the answer follows.
    #[test]
    fn think_tag_in_an_open_call_is_prose() {
        let r = full("Think <tool_call>get_weather here</think>Done.");
        assert!(r.calls.is_empty());
        assert_eq!(r.reasoning.as_deref(), Some("Think <tool_call>get_weather here"));
        assert_eq!(r.content.as_deref(), Some("Done."));
    }

    /// Two real calls in a row are both calls (a zero-argument one included).
    #[test]
    fn trailing_calls_are_all_reported() {
        let r = full("</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>\
                      <tool_call>get_current_date</tool_call>");
        assert_eq!(r.calls.len(), 2);
        assert_eq!(r.calls[0].1, "get_weather");
        assert_eq!(r.calls[0].2, r#"{"city": "Paris"}"#);
        assert_eq!(r.calls[1].1, "get_current_date");
        assert_eq!(r.calls[1].2, "{}");
    }

    /// A held run gives the same answer streamed, whatever the chunk size.
    #[test]
    fn prose_streams_the_same_as_a_whole_reply() {
        let cases = [
            "</think>Call it like <tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call> in your code.",
            "Think <tool_call>get_weather here</think>Done.",
            "</think>The format is <tool_call>get_weather and then the arguments.",
            "</think>Use `<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
            "</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call> sure.",
            "Think <tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call> a bit.</think>Done.",
            "</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
            "</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call><tool_call>get_current_date</tool_call>",
            "</think>Like this:\n```\n<tool_call>get_current_date</tool_call>\n```\n",
        ];
        for text in cases {
            let f = full(text);
            for chunk in [1usize, 2, 3, 5, 11] {
                let s = streamed(text, chunk);
                assert_eq!(
                    (s.reasoning, s.content, s.calls.iter().map(|c| (c.1.clone(), c.2.clone())).collect::<Vec<_>>()),
                    (f.reasoning.clone(), f.content.clone(), f.calls.iter().map(|c| (c.1.clone(), c.2.clone())).collect()),
                    "chunk {chunk} of {text:?}"
                );
            }
        }
    }

}
