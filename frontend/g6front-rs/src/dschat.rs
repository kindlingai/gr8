//! DeepSeek-V4.1 prompt renderer: OpenAI-format `messages` + `tools` -> the checkpoint's prompt text.
//!
//! A native port of the release's own `encoding.py` (DeepSeek-V4.1-Flash-1017), not a jinja template: the checkpoint
//! ships no chat_template.jinja. The V4.1 format differs from V4 in three ways the tests pin down: the DSML tags carry
//! a leading space (`<｜DSML｜ calls>`, `<｜DSML｜ invoke name="x">`, `<｜DSML｜ parameter ...>`), the reasoning effort is
//! a number in the system prefix, and a system message may appear mid-conversation.
//!
//! The text starts with `<｜begin▁of▁sentence｜>`; encode it with add_special=false (the bos is already in the text).
//! Rules kept from the encoder:
//! - thinking mode puts `<｜System｜>Reasoning Effort: N (...)` first; chat mode has no effort text;
//! - tool results become `<tool_result>…</tool_result>` blocks inside a user turn (tool messages merge into the
//!   neighbouring user message, ordered by the assistant's tool_calls order);
//! - tools are rendered on the first system message (an empty one is added when the conversation has none);
//! - `<｜Assistant｜>` is followed by `<think>` in thinking mode and `</think>` in chat mode; with thinking on, the
//!   reasoning of assistant turns before the last user turn is dropped, unless the request has tools;
//! - assistant turns end with the eos token.
//!
//! Out of scope (400): image parts, the `latest_reminder` role and `task`.

use std::collections::HashMap;

use serde_json::{Map, Value};

use crate::pyjson;

pub const BOS: &str = "<｜begin▁of▁sentence｜>";
pub const EOS: &str = "<｜end▁of▁sentence｜>";
const SYSTEM: &str = "<｜System｜>";
const USER: &str = "<｜User｜>";
pub const ASSISTANT: &str = "<｜Assistant｜>";
pub const THINK_OPEN: &str = "<think>";
pub const THINK_CLOSE: &str = "</think>";
const IMAGE_PLACEHOLDER: &str = "<｜deepseek_image｜>";

pub const CALLS_OPEN: &str = "<｜DSML｜ calls>";
pub const CALLS_CLOSE: &str = "</｜DSML｜ calls>";
pub const INVOKE_OPEN: &str = "<｜DSML｜ invoke";
pub const INVOKE_CLOSE: &str = "</｜DSML｜ invoke>";
pub const PARAM_OPEN: &str = "<｜DSML｜ parameter";
pub const PARAM_CLOSE: &str = "</｜DSML｜ parameter>";

const TOOLS_HEAD: &str = "## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<｜DSML｜ calls>\" block like the following:

<｜DSML｜ calls>
<｜DSML｜ invoke name=\"$TOOL_NAME\">
<｜DSML｜ parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜ parameter>
...
</｜DSML｜ invoke>
<｜DSML｜ invoke name=\"$TOOL_NAME2\">
...
</｜DSML｜ invoke>
</｜DSML｜ calls>

String parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.

If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas

";
const TOOLS_TAIL: &str = "

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
";

/// A request the renderer cannot honour; the API answers 400 with this text.
#[derive(Debug, PartialEq)]
pub struct RenderError(pub String);

fn err<T>(s: impl Into<String>) -> Result<T, RenderError> {
    Err(RenderError(s.into()))
}

#[derive(Clone, Copy, Debug)]
pub struct Opts {
    /// Thinking mode: the prompt ends `<think>` and the effort prefix is written. Off = chat mode, ends `</think>`.
    pub thinking: bool,
    /// 1..=100; written only in thinking mode.
    pub effort: u32,
}

// ---------------------------------------------------------------------------------------------------------------
// Message normalisation (the encoder's merge_tool_messages / sort_tool_results_by_call_order)
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
enum Role {
    System,
    User,
    Assistant,
}

enum Block {
    Text(String),
    Result { id: String, content: String },
    Other(String),
}

struct Call {
    id: String,
    name: String,
    args: Value,
}

struct Msg {
    role: Role,
    /// system / assistant text.
    content: String,
    /// user parts.
    blocks: Vec<Block>,
    calls: Vec<Call>,
    reasoning: String,
    /// The rendered `function` lines of the request's tools (first system message only).
    tools: Vec<String>,
    /// `find_last_user_index` skips a message made only of tool results.
    counts_as_user: bool,
}

impl Msg {
    fn new(role: Role) -> Msg {
        Msg {
            role,
            content: String::new(),
            blocks: Vec::new(),
            calls: Vec::new(),
            reasoning: String::new(),
            tools: Vec::new(),
            counts_as_user: true,
        }
    }
}

fn check_text(s: &str) -> Result<(), RenderError> {
    if s.contains(IMAGE_PLACEHOLDER) {
        return err(format!("message text contains the image token {IMAGE_PLACEHOLDER}; this model takes no images"));
    }
    Ok(())
}

fn part_text(v: &Value) -> String {
    v.as_str().unwrap_or("").to_string()
}

/// One element of a content list as a block. Images are refused.
fn list_block(b: &Value) -> Result<Block, RenderError> {
    if let Some(s) = b.as_str() {
        check_text(s)?;
        return Ok(Block::Text(s.to_string()));
    }
    let ty = b.get("type").and_then(Value::as_str);
    match ty {
        Some("text") => {
            let t = part_text(b.get("text").unwrap_or(&Value::Null));
            check_text(&t)?;
            Ok(Block::Text(t))
        }
        Some("image" | "image_url" | "input_image") => err("image content is not supported by this model"),
        Some("tool_result") => {
            let id = part_text(b.get("tool_use_id").unwrap_or(&Value::Null));
            Ok(Block::Result { id, content: tool_content(b.get("content"))? })
        }
        other => Ok(Block::Other(other.unwrap_or("None").to_string())),
    }
}

/// A tool message's content (or a tool_result block's): a string, or a list whose text parts join with a blank line.
fn tool_content(v: Option<&Value>) -> Result<String, RenderError> {
    match v {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(s)) => {
            check_text(s)?;
            Ok(s.clone())
        }
        Some(Value::Array(a)) => {
            let mut parts = Vec::new();
            for b in a {
                match list_block(b)? {
                    Block::Text(t) => parts.push(t),
                    Block::Other(t) => parts.push(format!("[Unsupported {t}]")),
                    Block::Result { .. } => parts.push("[Unsupported tool_result]".to_string()),
                }
            }
            Ok(parts.join("\n\n"))
        }
        Some(other) => Ok(pyjson::to_string(other)),
    }
}

/// System / assistant content: text, or the text parts of a list joined with a blank line.
fn plain_content(v: Option<&Value>) -> Result<(String, bool), RenderError> {
    match v {
        None | Some(Value::Null) => Ok((String::new(), true)),
        Some(Value::String(s)) => {
            check_text(s)?;
            Ok((s.clone(), true))
        }
        Some(Value::Array(a)) => {
            let mut texts = Vec::new();
            let mut has_text = false;
            for b in a {
                if let Block::Text(t) = list_block(b)? {
                    has_text = true;
                    texts.push(t)
                }
            }
            // a list with no text part does not count as a user turn when it is a mid-conversation system message
            Ok((texts.join("\n\n"), a.is_empty() || has_text))
        }
        Some(_) => err("message content must be a string or a list of parts"),
    }
}

fn user_blocks(v: Option<&Value>) -> Result<Vec<Block>, RenderError> {
    match v {
        None | Some(Value::Null) => Ok(vec![Block::Text(String::new())]),
        Some(Value::String(s)) => {
            check_text(s)?;
            Ok(vec![Block::Text(s.clone())])
        }
        Some(Value::Array(a)) => a.iter().map(list_block).collect(),
        Some(_) => err("message content must be a string or a list of parts"),
    }
}

/// `ns::name` as the encoder writes it. `ns` is the tool's / call's namespace (a string or `{"name": …}`).
fn encoded_name(name: &str, ns: Option<&Value>) -> Result<String, RenderError> {
    let ns = match ns {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(s.as_str()),
        Some(Value::Object(o)) => match o.get("name").and_then(Value::as_str) {
            Some(s) => Some(s),
            None => return err("tool namespace object needs a `name`"),
        },
        Some(_) => return err("tool namespace must be a string"),
    };
    let (ns, bare) = match name.split_once("::") {
        Some((prefix, bare)) => {
            if ns.is_some_and(|n| n != prefix) {
                return err(format!("conflicting tool namespaces: {} != {prefix}", ns.unwrap_or("")));
            }
            (Some(prefix), bare)
        }
        None => (ns, name),
    };
    if bare.contains("::") || ns.is_some_and(|n| n.contains("::")) {
        return err(format!("tool name must not contain '::' twice: {name}"));
    }
    Ok(match ns {
        None => bare.to_string(),
        Some(n) => format!("{n}::{bare}"),
    })
}

/// A tool's `function` object as the encoder prints it into the prompt (request key order, no added fields).
fn tool_line(tool: &Value) -> Result<String, RenderError> {
    let Some(f) = tool.get("function").and_then(Value::as_object) else {
        return err("each tool needs a `function` object");
    };
    let mut f: Map<String, Value> = f.clone();
    if let Some(ns) = tool.get("namespace").filter(|v| !v.is_null()) {
        f.insert("namespace".into(), ns.clone());
    }
    let Some(name) = f.get("name").and_then(Value::as_str) else {
        return err("each tool function needs a `name`");
    };
    let name = encoded_name(name, f.get("namespace"))?;
    f.insert("name".into(), Value::String(name));
    let ns = f.shift_remove("namespace");
    if let Some(Value::Object(n)) = ns {
        if let Some(d) = n.get("description").and_then(Value::as_str).filter(|d| !d.is_empty()) {
            let own = f.get("description").and_then(Value::as_str).unwrap_or("");
            let d = format!("{d}\n{own}");
            f.insert("description".into(), Value::String(d));
        }
    }
    Ok(pyjson::to_string(&Value::Object(f)))
}

/// A tool's name as the prompt and the model's calls write it (`ns::name` when it has a namespace).
pub fn tool_name(tool: &Value) -> Option<String> {
    let f = tool.get("function")?;
    let ns = tool.get("namespace").filter(|v| !v.is_null()).or_else(|| f.get("namespace"));
    encoded_name(f.get("name")?.as_str()?, ns).ok()
}

fn parse_calls(m: &Value) -> Result<Vec<Call>, RenderError> {
    let Some(calls) = m.get("tool_calls").and_then(Value::as_array) else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    for c in calls {
        let Some(f) = c.get("function").and_then(Value::as_object) else {
            return err("each tool call needs a `function` object");
        };
        let Some(name) = f.get("name").and_then(Value::as_str) else {
            return err("each tool call needs a function `name`");
        };
        let ns = c.get("namespace").filter(|v| !v.is_null()).or_else(|| f.get("namespace"));
        let id = c
            .get("id")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .or_else(|| f.get("id").and_then(Value::as_str))
            .unwrap_or("");
        out.push(Call {
            id: id.to_string(),
            name: encoded_name(name, ns)?,
            args: f.get("arguments").cloned().unwrap_or_else(|| Value::Object(Map::new())),
        });
    }
    Ok(out)
}

fn normalise(messages: &[Value], tools: &[Value]) -> Result<Vec<Msg>, RenderError> {
    let mut out: Vec<Msg> = Vec::new();
    let tool_lines: Vec<String> = tools.iter().map(tool_line).collect::<Result<_, _>>()?;
    let first_is_system = messages
        .first()
        .and_then(|m| m.get("role"))
        .and_then(Value::as_str)
        .is_some_and(|r| r == "system" || r == "developer");
    if !tool_lines.is_empty() && !first_is_system {
        let mut m = Msg::new(Role::System);
        m.tools = tool_lines.clone();
        out.push(m);
    }
    for m in messages {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("");
        match role {
            "system" | "developer" => {
                let (content, counts) = plain_content(m.get("content"))?;
                let mut msg = Msg::new(Role::System);
                msg.content = content;
                msg.counts_as_user = counts;
                out.push(msg);
            }
            "user" => {
                let blocks = user_blocks(m.get("content"))?;
                match out.last_mut() {
                    Some(last) if last.role == Role::User => last.blocks.extend(blocks),
                    _ => {
                        let mut msg = Msg::new(Role::User);
                        msg.blocks = blocks;
                        out.push(msg);
                    }
                }
            }
            "tool" => {
                let id = part_text(m.get("tool_call_id").unwrap_or(&Value::Null));
                let block = Block::Result { id, content: tool_content(m.get("content"))? };
                match out.last_mut() {
                    Some(last) if last.role == Role::User => last.blocks.push(block),
                    _ => {
                        let mut msg = Msg::new(Role::User);
                        msg.blocks = vec![block];
                        out.push(msg);
                    }
                }
            }
            "assistant" => {
                let mut msg = Msg::new(Role::Assistant);
                msg.content = plain_content(m.get("content"))?.0;
                msg.calls = parse_calls(m)?;
                let rc = m.get("reasoning_content").filter(|v| !v.is_null()).or_else(|| m.get("reasoning"));
                msg.reasoning = rc.and_then(Value::as_str).unwrap_or("").to_string();
                check_text(&msg.reasoning)?;
                out.push(msg);
            }
            "latest_reminder" => return err("the latest_reminder role is not supported"),
            other => return err(format!("unknown message role `{other}`")),
        }
    }
    if let Some(first) = out.first_mut() {
        if first.role == Role::System {
            first.tools = tool_lines;
        }
    }
    for m in &mut out {
        if m.role == Role::User {
            m.counts_as_user = m.blocks.is_empty() || m.blocks.iter().any(|b| matches!(b, Block::Text(_)));
        }
    }
    sort_results(&mut out);
    Ok(out)
}

/// Tool results inside one user message follow the order of the assistant's tool_calls (unknown ids sort first).
fn sort_results(msgs: &mut [Msg]) {
    let mut order: HashMap<String, usize> = HashMap::new();
    for m in msgs {
        match m.role {
            Role::Assistant if !m.calls.is_empty() => {
                order.clear();
                for (i, c) in m.calls.iter().enumerate() {
                    if !c.id.is_empty() {
                        order.insert(c.id.clone(), i);
                    }
                }
            }
            Role::User if !order.is_empty() => {
                let slots: Vec<usize> = m
                    .blocks
                    .iter()
                    .enumerate()
                    .filter(|(_, b)| matches!(b, Block::Result { .. }))
                    .map(|(i, _)| i)
                    .collect();
                if slots.len() > 1 {
                    let mut taken: Vec<Block> = slots
                        .iter()
                        .map(|&i| std::mem::replace(&mut m.blocks[i], Block::Other(String::new())))
                        .collect();
                    taken.sort_by_key(|b| match b {
                        Block::Result { id, .. } => order.get(id).copied().unwrap_or(0),
                        _ => 0,
                    });
                    for (slot, b) in slots.into_iter().zip(taken) {
                        m.blocks[slot] = b;
                    }
                }
            }
            _ => {}
        }
    }
}

fn last_user_index(msgs: &[Msg]) -> Option<usize> {
    (0..msgs.len())
        .rev()
        .find(|&i| (msgs[i].role == Role::User || (msgs[i].role == Role::System && i > 0)) && msgs[i].counts_as_user)
}

// ---------------------------------------------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------------------------------------------

/// The arguments of a call as `<｜DSML｜ parameter>` lines. A JSON string is parsed (twice at most, for double
/// encoding); anything that is not an object ends up as the single parameter `arguments`.
fn dsml_params(args: &Value) -> String {
    let mut cur = args.clone();
    if !cur.is_object() {
        for _ in 0..2 {
            let Value::String(s) = &cur else { break };
            match serde_json::from_str::<Value>(s) {
                Ok(v) => cur = v,
                Err(_) => break,
            }
        }
        if !cur.is_object() {
            let mut m = Map::new();
            m.insert("arguments".into(), args.clone());
            cur = Value::Object(m);
        }
    }
    let Value::Object(obj) = cur else { return String::new() };
    let mut lines = Vec::with_capacity(obj.len());
    for (k, v) in &obj {
        let (is_str, value) = match v {
            Value::String(s) => ("true", s.clone()),
            other => ("false", pyjson::to_string(other)),
        };
        lines.push(format!("{PARAM_OPEN} name=\"{k}\" string=\"{is_str}\">{value}{PARAM_CLOSE}"));
    }
    lines.join("\n")
}

/// A call as the model writes it: `<｜DSML｜ invoke name="x">\n<params>\n</｜DSML｜ invoke>`.
pub fn dsml_invoke(name: &str, args: &Value) -> String {
    format!("{INVOKE_OPEN} name=\"{name}\">\n{}\n{INVOKE_CLOSE}", dsml_params(args))
}

fn render_tools(lines: &[String]) -> String {
    format!("{TOOLS_HEAD}{}{TOOLS_TAIL}", lines.join("\n"))
}

fn effort_prefix(effort: u32) -> String {
    format!("Reasoning Effort: {effort} (range 1-100, the higher the value, the more thorough the reasoning)\n\n")
}

/// The prompt for `messages` (OpenAI wire format; `tools` is the request's tool list, may be empty).
pub fn render(messages: &[Value], tools: &[Value], o: &Opts) -> Result<String, RenderError> {
    if messages.is_empty() {
        return err("messages must not be empty");
    }
    if !(1..=100).contains(&o.effort) {
        return err(format!("reasoning effort must be 1..=100, got {}", o.effort));
    }
    let msgs = normalise(messages, tools)?;
    // With tools present the model's reasoning is never dropped from earlier turns.
    let drop_thinking = tools.is_empty();
    let last_user = last_user_index(&msgs).map(|i| i as isize).unwrap_or(-1);
    let mut p = String::from(BOS);
    for (i, m) in msgs.iter().enumerate() {
        let prefix = if i == 0 && o.thinking { effort_prefix(o.effort) } else { String::new() };
        if i == 0 && (!prefix.is_empty() || m.role == Role::System) {
            p.push_str(SYSTEM);
        }
        p.push_str(&prefix);
        match m.role {
            Role::System => {
                if i > 0 {
                    p.push_str(SYSTEM);
                }
                p.push_str(&m.content);
                if !m.tools.is_empty() {
                    p.push_str("\n\n");
                    p.push_str(&render_tools(&m.tools));
                }
            }
            Role::User => {
                p.push_str(USER);
                let parts: Vec<String> = m
                    .blocks
                    .iter()
                    .map(|b| match b {
                        Block::Text(t) => t.clone(),
                        Block::Result { content, .. } => format!("<tool_result>{content}</tool_result>"),
                        Block::Other(t) => format!("[Unsupported {t}]"),
                    })
                    .collect();
                p.push_str(&parts.join("\n\n"));
            }
            Role::Assistant => {
                if o.thinking && (!drop_thinking || i as isize > last_user) {
                    p.push_str(&m.reasoning);
                    p.push_str(THINK_CLOSE);
                }
                p.push_str(&m.content);
                if !m.calls.is_empty() {
                    let invokes: Vec<String> = m.calls.iter().map(|c| dsml_invoke(&c.name, &c.args)).collect();
                    p.push_str("\n\n");
                    p.push_str(CALLS_OPEN);
                    p.push('\n');
                    p.push_str(&invokes.join("\n"));
                    p.push('\n');
                    p.push_str(CALLS_CLOSE);
                }
                p.push_str(EOS);
            }
        }
        // The generation header follows the last message, or a message the assistant's reply follows.
        let next_is_assistant = msgs.get(i + 1).is_none_or(|n| n.role == Role::Assistant);
        if next_is_assistant && (m.role == Role::User || (m.role == Role::System && i > 0)) {
            p.push_str(ASSISTANT);
            let open = o.thinking && (!drop_thinking || i as isize >= last_user);
            p.push_str(if open { THINK_OPEN } else { THINK_CLOSE });
        }
    }
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const THINK: Opts = Opts { thinking: true, effort: 75 };
    const CHAT: Opts = Opts { thinking: false, effort: 75 };

    fn r(messages: Value, tools: Value, o: &Opts) -> String {
        render(messages.as_array().unwrap(), tools.as_array().unwrap(), o).unwrap()
    }

    #[test]
    fn chat_and_thinking_headers() {
        let m = json!([{"role": "user", "content": "hi"}]);
        assert_eq!(r(m.clone(), json!([]), &CHAT), "<｜begin▁of▁sentence｜><｜User｜>hi<｜Assistant｜></think>");
        assert_eq!(
            r(m, json!([]), &THINK),
            "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n<｜User｜>hi<｜Assistant｜><think>"
        );
    }

    #[test]
    fn system_first_in_chat_mode() {
        let m = json!([{"role": "developer", "content": "be brief"}, {"role": "user", "content": "hi"}]);
        assert_eq!(
            r(m, json!([]), &CHAT),
            "<｜begin▁of▁sentence｜><｜System｜>be brief<｜User｜>hi<｜Assistant｜></think>"
        );
    }

    #[test]
    fn tools_add_an_empty_system_message_and_keep_reasoning() {
        let tools = json!([{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}}]);
        let m = json!([
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "x", "reasoning_content": "why"},
            {"role": "user", "content": "b"}
        ]);
        let got = r(m, tools, &THINK);
        assert!(got.starts_with("<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n\n\n## Tools\n\n"));
        assert!(got.contains("{\"name\": \"f\", \"description\": \"d\", \"parameters\": {\"type\": \"object\"}}\n\nYou MUST strictly follow"));
        // tools present -> the earlier turn keeps its reasoning and every header opens <think>
        assert!(got.ends_with(
            "<｜User｜>a<｜Assistant｜><think>why</think>x<｜end▁of▁sentence｜><｜User｜>b<｜Assistant｜><think>"
        ));
    }

    #[test]
    fn earlier_reasoning_is_dropped_without_tools() {
        let m = json!([
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "x", "reasoning_content": "why"},
            {"role": "user", "content": "b"}
        ]);
        let got = r(m, json!([]), &THINK);
        assert!(
            got.ends_with("<｜User｜>a<｜Assistant｜></think>x<｜end▁of▁sentence｜><｜User｜>b<｜Assistant｜><think>")
        );
    }

    #[test]
    fn tool_loop_and_result_order() {
        let tools = json!([{"type": "function", "function": {"name": "f", "parameters": {}}}]);
        let m = json!([
            {"role": "system", "content": "S"},
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"a\": 1, \"s\": \"x\"}"}},
                {"id": "c2", "type": "function", "function": {"name": "f", "arguments": {"b": [1, 2]}}}
            ]},
            {"role": "tool", "tool_call_id": "c2", "content": "two"},
            {"role": "tool", "tool_call_id": "c1", "content": "one"}
        ]);
        let got = r(m, tools, &CHAT);
        let want_tail = "<｜User｜>go<｜Assistant｜></think>\n\n<｜DSML｜ calls>\n\
<｜DSML｜ invoke name=\"f\">\n<｜DSML｜ parameter name=\"a\" string=\"false\">1</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"s\" string=\"true\">x</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n\
<｜DSML｜ invoke name=\"f\">\n<｜DSML｜ parameter name=\"b\" string=\"false\">[1, 2]</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n\
</｜DSML｜ calls><｜end▁of▁sentence｜><｜User｜><tool_result>one</tool_result>\n\n<tool_result>two</tool_result><｜Assistant｜></think>";
        assert!(got.ends_with(want_tail), "{got}");
    }

    #[test]
    fn refuses_what_the_model_cannot_take() {
        for bad in [
            json!([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]),
            json!([{"role": "user", "content": "<｜deepseek_image｜>"}]),
            json!([{"role": "latest_reminder", "content": "x"}]),
            json!([{"role": "nobody", "content": "x"}]),
        ] {
            assert!(render(bad.as_array().unwrap(), &[], &CHAT).is_err(), "{bad}");
        }
        assert!(render(&[], &[], &CHAT).is_err());
        assert!(render(&[json!({"role": "user", "content": "x"})], &[], &Opts { thinking: true, effort: 0 }).is_err());
    }

    #[test]
    fn unsupported_part_and_null_content() {
        let m = json!([{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "audio"}, "b"]}, {"role": "assistant", "content": null}]);
        assert_eq!(r(m, json!([]), &CHAT), "<｜begin▁of▁sentence｜><｜User｜>a\n\n[Unsupported audio]\n\nb<｜Assistant｜></think><｜end▁of▁sentence｜>");
    }

    #[test]
    fn namespaces() {
        assert_eq!(encoded_name("a::b", None).unwrap(), "a::b");
        assert_eq!(encoded_name("b", Some(&json!("a"))).unwrap(), "a::b");
        assert_eq!(encoded_name("b", Some(&json!({"name": "a"}))).unwrap(), "a::b");
        assert!(encoded_name("a::b", Some(&json!("c"))).is_err());
        assert!(encoded_name("a::b::c", None).is_err());
    }

    #[test]
    fn non_object_arguments() {
        assert_eq!(
            dsml_params(&json!("[1]")),
            "<｜DSML｜ parameter name=\"arguments\" string=\"true\">[1]</｜DSML｜ parameter>"
        );
        assert_eq!(
            dsml_params(&json!("not json")),
            "<｜DSML｜ parameter name=\"arguments\" string=\"true\">not json</｜DSML｜ parameter>"
        );
        assert_eq!(
            dsml_params(&json!("\"{\\\"k\\\": 1}\"")),
            "<｜DSML｜ parameter name=\"k\" string=\"false\">1</｜DSML｜ parameter>"
        );
        assert_eq!(dsml_params(&json!({})), "");
    }

    /// Every case in ds41_golden.json was rendered by the release's own encoding.py (see tests/ds41_golden.py).
    #[test]
    fn matches_the_official_encoder() {
        let g: Value = serde_json::from_str(include_str!("ds41_golden.json")).unwrap();
        let cases = g.as_array().unwrap();
        assert!(cases.len() >= 20, "golden file looks truncated: {} cases", cases.len());
        for c in cases {
            let req = &c["request"];
            let o =
                Opts { thinking: req["thinking"].as_bool().unwrap(), effort: req["effort"].as_u64().unwrap() as u32 };
            let tools: Vec<Value> = req["tools"].as_array().cloned().unwrap_or_default();
            let got = render(req["messages"].as_array().unwrap(), &tools, &o).unwrap();
            assert_eq!(got, c["prompt"].as_str().unwrap(), "case {}", c["name"]);
        }
    }
}
