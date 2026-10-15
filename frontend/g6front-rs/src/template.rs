//! Chat template rendering, token-identical to the old vLLM stacks.
//!
//! Two layers:
//! - `normalize_messages()`: vLLM's message parsing for an "openai"-content-format template (string content -> text
//!   parts, tool results -> one string, assistant `reasoning` -> `reasoning_content`, tool-call `arguments` JSON
//!   strings -> objects, `developer` -> merged `system`).
//! - `ChatTemplate::render()`: minijinja configured like transformers' jinja environment (trim_blocks,
//!   lstrip_blocks, loop controls, Python string/dict methods, `tojson` as `json.dumps(ensure_ascii=False)`,
//!   `raise_exception`, `strftime_now`) + continue_final_message trimming.

use std::cell::RefCell;
use std::path::Path;

use minijinja::value::{Kwargs, ValueKind};
use minijinja::{Environment, Error, ErrorKind, Value as J};
use serde_json::{Map, Value};

use crate::pyjson::{self, py_strip};

#[derive(Debug)]
pub struct TemplateError(pub String);

impl From<&str> for TemplateError {
    fn from(s: &str) -> Self {
        TemplateError(s.to_string())
    }
}

thread_local! {
    /// The message of the last `raise_exception()` on this thread (minijinja wraps errors with location text).
    static RAISED: RefCell<Option<String>> = const { RefCell::new(None) };
}

pub struct ChatTemplate {
    env: Environment<'static>,
    special: Vec<(&'static str, Value)>,
    supports_developer: bool,
    /// the reasoning efforts the template names (`thinking::template_levels`), highest first
    pub levels: Vec<&'static str>,
}

/// Keyword args the renderer owns; a request cannot override them through chat_template_kwargs.
const RESERVED: [&str; 5] = ["messages", "tools", "documents", "chat_template", "tokenize"];

impl ChatTemplate {
    pub fn load(dir: &Path, eos_token: Value, pad_token: Value) -> Result<ChatTemplate, String> {
        let p = dir.join("chat_template.jinja");
        let source = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        Self::from_source(source, eos_token, pad_token)
    }

    pub fn from_source(source: String, eos_token: Value, pad_token: Value) -> Result<ChatTemplate, String> {
        let levels = crate::thinking::template_levels(&source);
        let supports_developer = source.contains("\"developer\"") || source.contains("'developer'");
        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_keep_trailing_newline(false);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.set_formatter(py_format);
        env.add_filter("tojson", tojson);
        env.add_function("raise_exception", |msg: String| -> Result<J, Error> {
            RAISED.with(|r| *r.borrow_mut() = Some(msg.clone()));
            Err(Error::new(ErrorKind::InvalidOperation, msg))
        });
        env.add_function("strftime_now", |fmt: String| -> Result<String, Error> { Ok(strftime_now(&fmt)) });
        env.add_template_owned("chat", source).map_err(|e| format!("chat_template.jinja: {e:#}"))?;
        Ok(ChatTemplate {
            env,
            special: vec![("eos_token", eos_token), ("pad_token", pad_token)],
            supports_developer,
            levels,
        })
    }

    /// `kw` is the output of `build_template_kwargs()`.
    pub fn render(
        &self,
        messages: Option<&Value>,
        tools: Option<&Value>,
        mut kw: Map<String, Value>,
    ) -> Result<String, TemplateError> {
        let add_generation_prompt = kw.shift_remove("add_generation_prompt").is_none_or(|v| pyjson::truthy(Some(&v)));
        let continue_final_message =
            kw.shift_remove("continue_final_message").is_some_and(|v| pyjson::truthy(Some(&v)));
        let documents = kw.shift_remove("documents").unwrap_or(Value::Null);
        if continue_final_message && add_generation_prompt {
            return Err("Cannot set both `continue_final_message` and `add_generation_prompt` to True.".into());
        }
        let conv = normalize_messages(messages, self.supports_developer)?;
        // render(messages=, tools=, documents=, add_generation_prompt=, continue_final_message=, **special, **kw)
        let mut ctx = Map::new();
        ctx.insert("messages".into(), Value::Array(conv));
        ctx.insert("tools".into(), tools.cloned().unwrap_or(Value::Null));
        ctx.insert("documents".into(), documents);
        ctx.insert("add_generation_prompt".into(), Value::Bool(add_generation_prompt));
        ctx.insert("continue_final_message".into(), Value::Bool(continue_final_message));
        for (k, v) in &self.special {
            ctx.insert((*k).into(), v.clone());
        }
        for (k, v) in kw {
            if !RESERVED.contains(&k.as_str()) {
                ctx.insert(k, v);
            }
        }
        let ctx = Value::Object(ctx);
        let tmpl = self.env.get_template("chat").map_err(|e| TemplateError(e.to_string()))?;
        RAISED.with(|r| *r.borrow_mut() = None);
        let text = match tmpl.render(J::from_serialize(&ctx)) {
            Ok(t) => t,
            Err(e) => {
                if let Some(msg) = RAISED.with(|r| r.borrow_mut().take()) {
                    return Err(TemplateError(msg));
                }
                return Err(TemplateError(format!("chat template error: {}", e.detail().unwrap_or(&e.to_string()))));
            }
        };
        if continue_final_message {
            let conv = ctx.get("messages").and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[]);
            return trim_to_final_message(text, conv);
        }
        Ok(text)
    }
}

fn strftime_now(fmt: &str) -> String {
    let Ok(cfmt) = std::ffi::CString::new(fmt) else { return String::new() };
    unsafe {
        let t = libc::time(std::ptr::null_mut());
        let mut tm: libc::tm = std::mem::zeroed();
        libc::localtime_r(&t, &mut tm);
        let mut buf = vec![0u8; 256 + fmt.len() * 8];
        let n = libc::strftime(buf.as_mut_ptr() as *mut libc::c_char, buf.len(), cfmt.as_ptr(), &tm);
        buf.truncate(n);
        String::from_utf8_lossy(&buf).into_owned()
    }
}

// ---- {{ value }}: what jinja2 prints, i.e. Python str() --------------------------------------------------------

fn py_format(out: &mut minijinja::Output, _: &minijinja::State, v: &J) -> Result<(), Error> {
    let s = match v.kind() {
        ValueKind::String => return out.write_str(v.as_str().unwrap_or("")).map_err(Error::from),
        ValueKind::Undefined => return Ok(()),
        _ => {
            let mut s = String::new();
            py_repr(&mut s, v)?;
            s
        }
    };
    out.write_str(&s).map_err(Error::from)
}

/// Python's printable test for repr(), close enough for prompt text: controls, separators other than the space,
/// format characters and private use are escaped.
fn py_printable(c: char) -> bool {
    !matches!(c as u32,
        0..=0x1f | 0x7f..=0xa0 | 0xad | 0x1680 | 0x180e | 0x2000..=0x200f | 0x2028..=0x202f | 0x205f..=0x2064
        | 0x3000 | 0xd800..=0xf8ff | 0xfeff | 0xfff9..=0xfffb | 0xf0000..)
}

fn py_repr_str(out: &mut String, s: &str) {
    let q = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    out.push(q);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == q => {
                out.push('\\');
                out.push(c);
            }
            c if c == ' ' || py_printable(c) => out.push(c),
            c if (c as u32) < 0x100 => out.push_str(&format!("\\x{:02x}", c as u32)),
            c if (c as u32) < 0x10000 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push_str(&format!("\\U{:08x}", c as u32)),
        }
    }
    out.push(q);
}

/// `str(x)` for non-strings and `repr(x)` inside containers, as Python prints values parsed from JSON.
fn py_repr(out: &mut String, v: &J) -> Result<(), Error> {
    match v.kind() {
        ValueKind::Undefined | ValueKind::None => out.push_str("None"),
        ValueKind::Bool => out.push_str(if v.is_true() { "True" } else { "False" }),
        ValueKind::Number if v.is_integer() => out.push_str(&v.to_string()),
        ValueKind::Number => out.push_str(&pyjson::float_string(f64::try_from(v.clone()).unwrap_or(f64::NAN))),
        ValueKind::String => py_repr_str(out, v.as_str().unwrap_or("")),
        ValueKind::Seq | ValueKind::Iterable => {
            out.push('[');
            for (i, x) in v.try_iter()?.enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_repr(out, &x)?;
            }
            out.push(']');
        }
        ValueKind::Map => {
            out.push('{');
            for (i, k) in v.try_iter()?.enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_repr(out, &k)?;
                out.push_str(": ");
                py_repr(out, &v.get_item(&k)?)?;
            }
            out.push('}');
        }
        _ => out.push_str(&v.to_string()),
    }
    Ok(())
}

// ---- tojson: json.dumps(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False) -------------------

struct Dump {
    ensure_ascii: bool,
    indent: Option<String>,
    item: String,
    kv: String,
    sort_keys: bool,
}

fn tojson(v: J, kwargs: Kwargs) -> Result<J, Error> {
    let ensure_ascii: Option<bool> = kwargs.get("ensure_ascii")?;
    let indent: Option<J> = kwargs.get("indent")?;
    let separators: Option<J> = kwargs.get("separators")?;
    let sort_keys: Option<bool> = kwargs.get("sort_keys")?;
    kwargs.assert_all_used()?;
    let indent = match indent {
        None => None,
        Some(i) if i.is_none() || i.is_undefined() => None,
        Some(i) => Some(match i.as_i64() {
            Some(n) => " ".repeat(n.max(0) as usize),
            None => i.as_str().unwrap_or("").to_string(),
        }),
    };
    let (mut item, mut kv) = (if indent.is_some() { "," } else { ", " }.to_string(), ": ".to_string());
    if let Some(s) = separators.filter(|s| !s.is_none() && !s.is_undefined()) {
        let parts: Vec<J> = s.try_iter()?.collect();
        if parts.len() == 2 {
            item = parts[0].as_str().unwrap_or(",").to_string();
            kv = parts[1].as_str().unwrap_or(":").to_string();
        }
    }
    let d =
        Dump { ensure_ascii: ensure_ascii.unwrap_or(false), indent, item, kv, sort_keys: sort_keys.unwrap_or(false) };
    let mut out = Vec::new();
    dump(&d, &mut out, &v, 0)?;
    Ok(J::from(String::from_utf8(out).unwrap()))
}

fn dump_str(d: &Dump, out: &mut Vec<u8>, s: &str) {
    if !d.ensure_ascii || s.is_ascii() {
        pyjson::string(out, s);
        return;
    }
    let mut tmp = Vec::new();
    pyjson::string(&mut tmp, s);
    for c in String::from_utf8(tmp).unwrap().chars() {
        if c.is_ascii() {
            out.push(c as u8);
        } else {
            let mut units = [0u16; 2];
            for u in c.encode_utf16(&mut units) {
                out.extend_from_slice(format!("\\u{:04x}", u).as_bytes());
            }
        }
    }
}

fn newline(d: &Dump, out: &mut Vec<u8>, depth: usize) {
    if let Some(ind) = &d.indent {
        out.push(b'\n');
        for _ in 0..depth {
            out.extend_from_slice(ind.as_bytes());
        }
    }
}

fn dump(d: &Dump, out: &mut Vec<u8>, v: &J, depth: usize) -> Result<(), Error> {
    match v.kind() {
        ValueKind::Undefined | ValueKind::None => out.extend_from_slice(b"null"),
        ValueKind::Bool => out.extend_from_slice(if v.is_true() { b"true" } else { b"false" }),
        ValueKind::Number => {
            if v.is_integer() {
                out.extend_from_slice(v.to_string().as_bytes());
            } else {
                pyjson::float(out, f64::try_from(v.clone()).unwrap_or(f64::NAN));
            }
        }
        ValueKind::String => dump_str(d, out, v.as_str().unwrap_or("")),
        ValueKind::Seq | ValueKind::Iterable => {
            let items: Vec<J> = v.try_iter()?.collect();
            if items.is_empty() {
                out.extend_from_slice(b"[]");
                return Ok(());
            }
            out.push(b'[');
            for (i, x) in items.iter().enumerate() {
                if i > 0 {
                    out.extend_from_slice(d.item.as_bytes());
                }
                newline(d, out, depth + 1);
                dump(d, out, x, depth + 1)?;
            }
            newline(d, out, depth);
            out.push(b']');
        }
        ValueKind::Map => {
            let mut keys: Vec<J> = v.try_iter()?.collect();
            if keys.is_empty() {
                out.extend_from_slice(b"{}");
                return Ok(());
            }
            if d.sort_keys {
                keys.sort_by_key(|a| a.to_string());
            }
            out.push(b'{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.extend_from_slice(d.item.as_bytes());
                }
                newline(d, out, depth + 1);
                match k.as_str() {
                    Some(s) => dump_str(d, out, s),
                    None => dump_str(d, out, &k.to_string()),
                }
                out.extend_from_slice(d.kv.as_bytes());
                dump(d, out, &v.get_item(k)?, depth + 1)?;
            }
            newline(d, out, depth);
            out.push(b'}');
        }
        _ => dump_str(d, out, &v.to_string()),
    }
    Ok(())
}

// ---- continue_final_message ----------------------------------------------------------------------------------

fn trim_to_final_message(text: String, conv: &[Value]) -> Result<String, TemplateError> {
    let mut final_ = conv.last().and_then(|c| c.get("content")).cloned().unwrap_or(Value::Null);
    if let Value::Array(blocks) = &final_ {
        match blocks.iter().rev().find_map(|b| b.as_object().and_then(|o| o.get("text")).cloned()) {
            Some(t) => final_ = t,
            None => return Err("continue_final_message needs a final message with text content".into()),
        }
    }
    let Some(final_) = final_.as_str() else {
        return Err("continue_final_message: final message not found in the rendered prompt".into());
    };
    let stripped = py_strip(final_);
    let Some(loc) = text.rfind(stripped) else {
        return Err("continue_final_message: final message not found in the rendered prompt".into());
    };
    // Python: text[loc:loc+len(final.lstrip())] == final, which can only hold when final has no leading whitespace
    let lstripped = final_.trim_start_matches(pyjson::is_py_space);
    if lstripped.len() == final_.len() && text[loc..].starts_with(final_) {
        return Ok(text[..loc + final_.len()].to_string());
    }
    Ok(text[..loc + stripped.len()].to_string())
}

// ---- message normalization (vLLM chat_utils behaviour for an "openai"-content-format template) -----------------

fn text_part(t: &Value) -> Value {
    let mut m = Map::new();
    m.insert("type".into(), Value::from("text"));
    m.insert("text".into(), t.clone());
    Value::Object(m)
}

fn parts(content: Option<&Value>) -> Result<Vec<Value>, TemplateError> {
    match content {
        None | Some(Value::Null) => Ok(vec![]),
        Some(s @ Value::String(_)) => Ok(vec![text_part(s)]),
        Some(Value::Array(a)) => {
            let mut out = Vec::with_capacity(a.len());
            for p in a {
                match p {
                    Value::String(_) => out.push(text_part(p)),
                    Value::Object(o) => match o.get("type").and_then(Value::as_str) {
                        Some("text" | "input_text" | "output_text") => {
                            out.push(text_part(o.get("text").unwrap_or(&Value::from(""))))
                        }
                        Some("refusal") => out.push(text_part(o.get("refusal").unwrap_or(&Value::from("")))),
                        _ => out.push(p.clone()),
                    },
                    _ => return Err("content parts must be objects or strings".into()),
                }
            }
            Ok(out)
        }
        Some(_) => Err("message content must be a string, a list of parts, or null".into()),
    }
}

fn parse_args(f: &Map<String, Value>) -> Value {
    let a = f.get("arguments");
    if !pyjson::truthy(a) {
        return Value::Object(Map::new());
    }
    match a.unwrap() {
        v @ Value::Object(_) => v.clone(),
        Value::String(s) => match serde_json::from_str::<Value>(s) {
            Ok(v @ Value::Object(_)) => v,
            _ => Value::Object(Map::new()),
        },
        _ => Value::Object(Map::new()),
    }
}

pub fn normalize_messages(messages: Option<&Value>, supports_developer: bool) -> Result<Vec<Value>, TemplateError> {
    let Some(Value::Array(messages)) = messages else { return Err("`messages` must be a list".into()) };
    let mut conv = Vec::with_capacity(messages.len());
    for m in messages {
        let Some(m) = m.as_object().filter(|m| m.contains_key("role")) else {
            return Err("each message needs a role".into());
        };
        let role = &m["role"];
        let mut c = Map::new();
        c.insert("role".into(), role.clone());
        c.insert("content".into(), Value::Array(parts(m.get("content"))?));
        match role.as_str() {
            Some("assistant") => {
                match m.get("tool_calls") {
                    None | Some(Value::Null) => {}
                    Some(tcs) => {
                        let tcs = match tcs {
                            Value::Array(a) => a.clone(),
                            Value::Object(o) if o.is_empty() => vec![],
                            Value::String(s) if s.is_empty() => vec![],
                            _ => return Err("assistant tool_calls entries must be objects.".into()),
                        };
                        let mut out = Vec::with_capacity(tcs.len());
                        for mut tc in tcs {
                            let Some(o) = tc.as_object_mut() else {
                                return Err("assistant tool_calls entries must be objects.".into());
                            };
                            let is_fn = o.get("type").is_none_or(|t| t.as_str() == Some("function"));
                            match o.get_mut("function") {
                                Some(Value::Object(f)) if is_fn => {
                                    let a = parse_args(f);
                                    f.insert("arguments".into(), a);
                                }
                                _ => {
                                    return Err(
                                        "chat completions only support assistant tool_calls of type 'function'.".into(),
                                    )
                                }
                            }
                            out.push(tc);
                        }
                        if !out.is_empty() {
                            c.insert("tool_calls".into(), Value::Array(out));
                        }
                    }
                }
                if let Some(r) = m.get("reasoning").filter(|r| !r.is_null()) {
                    c.insert("reasoning".into(), r.clone());
                    c.insert("reasoning_content".into(), r.clone());
                }
            }
            Some("tool") => {
                if let Some(id) = m.get("tool_call_id") {
                    c.insert("tool_call_id".into(), id.clone());
                }
                let ps = c["content"].as_array().unwrap();
                if !ps.iter().any(|p| p.get("type").and_then(Value::as_str) != Some("text")) {
                    let texts: Vec<String> = ps
                        .iter()
                        .map(|p| match p.get("text") {
                            Some(Value::String(s)) => s.clone(),
                            Some(v) => pyjson::to_string(v),
                            None => String::new(),
                        })
                        .collect();
                    c.insert("content".into(), Value::from(texts.join("\n")));
                }
            }
            _ => {}
        }
        if let Some(n @ Value::String(_)) = m.get("name") {
            c.insert("name".into(), n.clone());
        }
        if role.as_str() == Some("developer") {
            c.insert("tools".into(), m.get("tools").cloned().unwrap_or(Value::Null));
        }
        conv.push(Value::Object(c));
    }
    if !supports_developer && conv.iter().any(|c| c["role"].as_str() == Some("developer")) {
        conv = developer_to_system(conv);
    }
    Ok(conv)
}

fn developer_to_system(conv: Vec<Value>) -> Vec<Value> {
    let mut conv: Vec<Value> = conv
        .into_iter()
        .map(|mut c| {
            if c["role"].as_str() == Some("developer") {
                c["role"] = Value::from("system");
            }
            c
        })
        .collect();
    for c in conv.iter_mut() {
        if c["role"].as_str() == Some("system") {
            c.as_object_mut().unwrap().shift_remove("tools");
        }
    }
    let (mut sys_texts, mut rest, mut needs) = (Vec::<String>::new(), Vec::new(), false);
    for (i, c) in conv.iter().enumerate() {
        if c["role"].as_str() == Some("system") {
            if i > 0 || !sys_texts.is_empty() {
                needs = true;
            }
            let content = match c.get("content") {
                Some(Value::Array(ps)) => ps
                    .iter()
                    .filter_map(|p| match p {
                        Value::String(s) => Some(s.clone()),
                        Value::Object(o) => {
                            o.get("text").map(|t| t.as_str().map_or_else(|| t.to_string(), str::to_string))
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
                Some(Value::String(s)) => s.clone(),
                _ => String::new(),
            };
            if !content.is_empty() {
                sys_texts.push(content);
            }
        } else {
            rest.push(c.clone());
        }
    }
    if !needs {
        return conv;
    }
    let mut sys = Map::new();
    sys.insert("role".into(), Value::from("system"));
    sys.insert("content".into(), Value::from(sys_texts.join("\n\n")));
    let mut out = vec![Value::Object(sys)];
    out.extend(rest);
    out
}

/// vLLM's ChatCompletionToolsParam.model_dump(): {type, function:{name, description, parameters[, strict]
/// [, defer_loading]}[, defer_loading]} with null description/parameters kept (the template prints them).
pub fn normalize_tools(tools: Option<&Value>) -> Result<Option<Value>, TemplateError> {
    let tools = match tools {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Array(a)) => a,
        Some(_) => return Err("`tools` must be a list".into()),
    };
    let mut out = Vec::with_capacity(tools.len());
    for t in tools {
        let f = t.get("function").and_then(Value::as_object).filter(|f| f.contains_key("name"));
        let (Some(t), Some(f)) = (t.as_object(), f) else {
            return Err("each tool needs {\"type\": \"function\", \"function\": {\"name\": ...}}".into());
        };
        if t.get("type").is_some_and(|ty| ty.as_str() != Some("function")) {
            return Err("only tools of type 'function' are supported".into());
        }
        let mut fd = Map::new();
        fd.insert("name".into(), f["name"].clone());
        fd.insert("description".into(), f.get("description").cloned().unwrap_or(Value::Null));
        fd.insert("parameters".into(), f.get("parameters").cloned().unwrap_or(Value::Null));
        // f.get('defer_loading', t.get('defer_loading')): a null on the function wins over the outer value
        let dl = if f.contains_key("defer_loading") { f.get("defer_loading") } else { t.get("defer_loading") };
        let dl = dl.filter(|v| !v.is_null());
        if let Some(s) = f.get("strict").filter(|v| !v.is_null()) {
            fd.insert("strict".into(), s.clone());
        }
        if let Some(dl) = dl {
            fd.insert("defer_loading".into(), dl.clone());
        }
        let mut td = Map::new();
        td.insert("type".into(), Value::from("function"));
        td.insert("function".into(), Value::Object(fd));
        if let Some(dl) = t.get("defer_loading").filter(|v| !v.is_null()) {
            td.insert("defer_loading".into(), dl.clone());
        }
        out.push(Value::Object(td));
    }
    Ok(Some(Value::Array(out)))
}

/// vLLM ChatCompletionRequest.build_chat_params(): request chat_template_kwargs are the defaults; the top-level
/// fields override them unless null/"auto". `reasoning_effort` is the resolved one (`thinking::resolve`), already
/// mapped to the template's levels: it replaces whatever the kwargs held. No `enable_thinking` is added: GLM 5.3's
/// template does not read it.
pub fn build_template_kwargs(
    chat_template_kwargs: Option<&Value>,
    add_generation_prompt: Value,
    continue_final_message: Value,
    reasoning_effort: Option<&str>,
    documents: Option<&Value>,
) -> Result<Map<String, Value>, TemplateError> {
    let mut user = match chat_template_kwargs {
        Some(Value::Object(o)) => o.clone(),
        v if !pyjson::truthy(v) => Map::new(), // dict(x or {})
        _ => return Err("chat_template_kwargs must be an object".into()),
    };
    let extra: [(&str, Value); 3] = [
        ("add_generation_prompt", add_generation_prompt),
        ("continue_final_message", continue_final_message),
        ("documents", documents.cloned().unwrap_or(Value::Null)),
    ];
    let mut bad: Vec<&str> = ["chat_template", "tokenize"].into_iter().filter(|k| user.contains_key(*k)).collect();
    if !bad.is_empty() {
        bad.sort();
        let set = bad.iter().map(|k| format!("'{k}'")).collect::<Vec<_>>().join(", ");
        return Err(TemplateError(format!("Found unexpected chat template kwargs from request: {{{set}}}")));
    }
    for (k, v) in extra {
        if !v.is_null() && v.as_str() != Some("auto") {
            user.insert(k.to_string(), v);
        }
    }
    user.shift_remove("reasoning_effort");
    if let Some(e) = reasoning_effort {
        user.insert("reasoning_effort".into(), Value::from(e));
    }
    Ok(user)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::thinking::{resolve, Defaults};
    use serde_json::json;

    /// GLM 5.3's effort lines and generation prompt, with a minimal message loop.
    const GLM_HEAD: &str = r#"[gMASK]<sop>
{%- set effective_reasoning_effort = reasoning_effort if reasoning_effort is defined and reasoning_effort in ['low', 'high'] else 'max' -%}
{%- if effective_reasoning_effort is not none -%}<|system|>Reasoning Effort: {{ effective_reasoning_effort | capitalize }}{%- endif -%}
{%- for m in messages -%}{%- if m.role == 'user' -%}<|user|>{{ m.content[0].text }}{%- endif -%}{%- endfor -%}
{%- if add_generation_prompt -%}<|assistant|><think>{%- endif -%}"#;

    fn render(t: &ChatTemplate, body: Value) -> String {
        let body = body.as_object().unwrap();
        let th = resolve(body, &t.levels, Defaults::default()).unwrap();
        let kw = build_template_kwargs(
            body.get("chat_template_kwargs"),
            Value::Bool(true),
            Value::Bool(false),
            th.effort,
            None,
        )
        .unwrap();
        t.render(Some(&json!([{"role": "user", "content": "hi"}])), None, kw).unwrap()
    }

    fn effort_lines(t: &ChatTemplate) {
        let line = |b: Value| {
            let text = render(t, b);
            let i = text.find("Reasoning Effort: ").expect("an effort line");
            text[i + 18..].split('<').next().unwrap().to_string()
        };
        assert_eq!(line(json!({})), "Max"); // the template's own default, as before
        let want = [
            ("minimal", "Low"),
            ("low", "Low"),
            ("medium", "High"),
            ("high", "High"),
            ("xhigh", "Max"),
            ("max", "Max"),
        ];
        for (e, l) in want {
            assert_eq!(line(json!({ "reasoning_effort": e })), l, "{e}");
            assert_eq!(line(json!({"chat_template_kwargs": { "reasoning_effort": e }})), l, "kwargs {e}");
        }
        // an explicit switch on is accepted and changes nothing
        assert_eq!(line(json!({"chat_template_kwargs": {"enable_thinking": true}})), "Max");
        assert_eq!(render(t, json!({"chat_template_kwargs": {"enable_thinking": true}})), render(t, json!({})));
    }

    #[test]
    fn each_effort_renders_the_nearest_named_level() {
        let t = ChatTemplate::from_source(GLM_HEAD.into(), Value::Null, Value::Null).unwrap();
        assert_eq!(t.levels, vec!["max", "high", "low"]);
        effort_lines(&t);
    }

    /// The real template: G6_TOKENIZER_DIR=<dir with chat_template.jinja ...> cargo test --release
    #[test]
    fn each_effort_renders_the_nearest_named_level_in_the_model_template() {
        let Ok(dir) = std::env::var("G6_TOKENIZER_DIR") else {
            eprintln!("G6_TOKENIZER_DIR not set; skipping");
            return;
        };
        let t = ChatTemplate::load(Path::new(&dir), Value::Null, Value::Null).unwrap();
        assert_eq!(t.levels, vec!["max", "high", "low"]);
        effort_lines(&t);
    }
}
