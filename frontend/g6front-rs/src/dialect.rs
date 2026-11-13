//! The front end speaks four model dialects. `glm` is GLM 5.3: a jinja chat template, `<tool_call>` calls, a thinking
//! switch that needs `--allow-thinking-off`. `ds41` is DeepSeek-V4.1 Flash: the prompt comes from `dschat.rs` (a
//! port of the release's `encoding.py`), calls are DSML blocks (`dsparser.rs`), thinking is a plain on/off switch and
//! `reasoning_effort` is a number from 1 to 100.
//! `hf` is generic HuggingFace (Qwen, Llama, Mistral, Hermes and compatible models): `chat_template.jinja` or
//! `tokenizer_config.json`'s `chat_template`, optional `<think>...</think>` reasoning, and JSON `<tool_call>` calls
//! (`hfparser.rs`).
//! `inkling` is Inkling-2: the model's own jinja template, channel-tagged reply blocks (`inkparser.rs`), and a
//! `reasoning_effort` that is a level name or a number from 0.0 to 0.99.
//!
//! This file holds what the dialects share at the seam: which dialect a model directory is, the parser enum the
//! handlers call, dialect-specific thinking and effort controls, and the text a forced `tool_choice` puts in the reply.

use std::path::Path;

use serde_json::{Map, Value};

use crate::api::ApiError;
use crate::dsparser::DsParser;
use crate::hfparser::HfParser;
use crate::inkparser::InkParser;
use crate::parser::{Delta, FullParse, GlmParser};
use crate::pyjson;
use crate::sampling::Limit;
use crate::thinking::{self, Thinking};
use crate::tool_choice::ToolChoice;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dialect {
    Glm,
    Ds41,
    Hf,
    Inkling,
}

impl Dialect {
    pub fn name(self) -> &'static str {
        match self {
            Dialect::Glm => "glm",
            Dialect::Ds41 => "ds41",
            Dialect::Hf => "hf",
            Dialect::Inkling => "inkling",
        }
    }

    /// `--dialect`: `glm`, `ds41`, `hf`, `inkling`, or `auto` (None: look at the model directory).
    pub fn parse_flag(s: &str) -> Result<Option<Dialect>, String> {
        match s {
            "glm" => Ok(Some(Dialect::Glm)),
            "ds41" => Ok(Some(Dialect::Ds41)),
            "hf" => Ok(Some(Dialect::Hf)),
            "inkling" => Ok(Some(Dialect::Inkling)),
            "auto" => Ok(None),
            other => Err(format!("--dialect {other}: want glm, ds41, hf, inkling or auto")),
        }
    }

    /// Auto-detect the dialect from `config.json` and the tokenizer:
    /// - `DeepseekV4` in `architectures[0]` or the `｜DSML｜` token -> `Ds41` (the 1017 release's `config.json` is the
    ///   inference layout and names no architecture);
    /// - an `architectures[0]` that starts with `Inkling` (any case) or `"model_type": "inkling2"` -> `Inkling`;
    /// - missing `config.json`, missing/empty `architectures[0]`, or a `Glm` / `ChatGLM` prefix -> `Glm`;
    /// - any other non-empty `architectures[0]` (e.g. `Qwen3ForCausalLM`, `LlamaForCausalLM`) -> `Hf`.
    pub fn detect(dir: &Path, tokenizer_has_dsml: bool) -> Dialect {
        let cfg = std::fs::read(dir.join("config.json")).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok());
        let arch = cfg.as_ref().and_then(|c| c.get("architectures")?.as_array()?.first()?.as_str().map(str::to_string));
        if tokenizer_has_dsml || arch.as_deref().is_some_and(|a| a.contains("DeepseekV4")) {
            return Dialect::Ds41;
        }
        let model_type = cfg.as_ref().and_then(|c| c.get("model_type")?.as_str());
        if model_type == Some("inkling2")
            || arch.as_deref().is_some_and(|a| a.to_ascii_lowercase().starts_with("inkling"))
        {
            return Dialect::Inkling;
        }
        match arch.as_deref() {
            Some(a) if !a.is_empty() && !is_glm_arch(a) => Dialect::Hf,
            _ => Dialect::Glm,
        }
    }
}

fn is_glm_arch(arch: &str) -> bool {
    let lower = arch.to_ascii_lowercase();
    lower.starts_with("glm") || lower.starts_with("chatglm")
}

/// Default `--served-model-name` for `--dialect hf`: the basename of `--model-dir`, or `"hf-model"` when empty.
pub fn hf_served_name(dir: &Path) -> String {
    dir.file_name().and_then(|n| n.to_str()).filter(|n| !n.is_empty()).unwrap_or("hf-model").to_string()
}

/// Default `--served-model-name` for `--dialect inkling`: `inkling-2-small`, `ink2`, and the basename of `--model-dir`
/// when it has one that is not already in the list.
pub fn inkling_served_names(dir: &Path) -> Vec<String> {
    let mut names = vec!["inkling-2-small".to_string(), "ink2".to_string()];
    if let Some(base) = dir.file_name().and_then(|n| n.to_str()).filter(|n| !n.is_empty()) {
        if !names.iter().any(|n| n == base) {
            names.push(base.to_string());
        }
    }
    names
}

// ---- the parser the handlers hold ------------------------------------------------------------------------------------

pub enum AnyParser {
    Glm(GlmParser),
    Ds(DsParser),
    Hf(HfParser),
    Ink(InkParser),
}

impl AnyParser {
    pub fn new(d: Dialect, tools: Option<&Value>, tool_choice: &Value, include_reasoning: bool) -> AnyParser {
        match d {
            Dialect::Glm => AnyParser::Glm(GlmParser::new(tools, tool_choice, include_reasoning)),
            Dialect::Ds41 => AnyParser::Ds(DsParser::new(tools, tool_choice, include_reasoning)),
            Dialect::Hf => AnyParser::Hf(HfParser::new(tools, tool_choice, include_reasoning)),
            Dialect::Inkling => AnyParser::Ink(InkParser::new(tools, tool_choice, include_reasoning)),
        }
    }

    /// `--undeclared-tool-calls emit` (the default): a call to a tool the request did not declare comes back as a
    /// tool call. GLM emits it at the end of the reply (so prose can still disprove it); the others emit a call once
    /// its closer arrives.
    pub fn emit_undeclared(self, on: bool) -> AnyParser {
        match self {
            AnyParser::Glm(p) => AnyParser::Glm(p.emit_undeclared(on)),
            AnyParser::Ds(p) => AnyParser::Ds(p.emit_undeclared(on)),
            AnyParser::Hf(p) => AnyParser::Hf(p.emit_undeclared(on)),
            AnyParser::Ink(p) => AnyParser::Ink(p.emit_undeclared(on)),
        }
    }

    /// The reply starts as the answer: the prompt closed the think block (or thinking was turned off).
    pub fn content_first(self) -> AnyParser {
        match self {
            AnyParser::Glm(p) => AnyParser::Glm(p.content_first()),
            AnyParser::Ds(p) => AnyParser::Ds(p.content_first()),
            AnyParser::Hf(p) => AnyParser::Hf(p.content_first()),
            AnyParser::Ink(p) => AnyParser::Ink(p.content_first()),
        }
    }

    /// The prompt left `<think>` open: for `Hf`, start inside reasoning up to `</think>` (`Glm` and `Ds` already start
    /// in reasoning by default).
    pub fn in_think(self) -> AnyParser {
        match self {
            AnyParser::Hf(p) => AnyParser::Hf(p.in_think()),
            p => p,
        }
    }

    pub fn delta(&mut self, text: &str, finished: bool) -> Option<Delta> {
        match self {
            AnyParser::Glm(p) => p.delta(text, finished),
            AnyParser::Ds(p) => p.delta(text, finished),
            AnyParser::Hf(p) => p.delta(text, finished),
            AnyParser::Ink(p) => p.delta(text, finished),
        }
    }

    /// The repeated-tool-call guard's held names (guard.rs). GLM only: the DSML and HF parsers stream closed calls and
    /// have no held run to take back.
    pub fn guard_names(self, names: Vec<String>) -> AnyParser {
        match self {
            AnyParser::Glm(p) => AnyParser::Glm(p.guard_names(names)),
            p => p,
        }
    }

    pub fn glm_mut(&mut self) -> Option<&mut GlmParser> {
        match self {
            AnyParser::Glm(p) => Some(p),
            AnyParser::Ds(_) | AnyParser::Hf(_) | AnyParser::Ink(_) => None,
        }
    }

    pub fn parse_full(&mut self, text: &str) -> FullParse {
        match self {
            AnyParser::Glm(p) => p.parse_full(text),
            AnyParser::Ds(p) => p.parse_full(text),
            AnyParser::Hf(p) => p.parse_full(text),
            AnyParser::Ink(p) => p.parse_full(text),
        }
    }

    pub fn include_reasoning(&self) -> bool {
        match self {
            AnyParser::Glm(p) => p.include_reasoning,
            AnyParser::Ds(p) => p.include_reasoning,
            AnyParser::Hf(p) => p.include_reasoning,
            AnyParser::Ink(p) => p.include_reasoning,
        }
    }

    /// A live call block has closed: the reply is complete and the generation should end. Only DeepSeek's format has a
    /// closer to stop on; the model sometimes goes on writing after it (the 10,559-line `</function>` runaway).
    pub fn call_closed(&self) -> bool {
        match self {
            AnyParser::Glm(_) | AnyParser::Hf(_) | AnyParser::Ink(_) => false,
            AnyParser::Ds(p) => p.call_closed(),
        }
    }
}

// ---- DeepSeek thinking and effort ------------------------------------------------------------------------------------

/// The effort the encoder uses when a request names none: its own default, `high`.
pub const DS_DEFAULT_EFFORT: u32 = 75;

const DS_EFFORT_ERROR: &str =
    "reasoning_effort must be an integer from 1 to 100, or none, minimal, low, medium, high, \
                               xhigh or max";

/// A named effort as a number: the release's low 50, high 75, max 100, plus SGLang's xhigh (75) and TensorFold's
/// minimal (25) and medium (62).
pub fn ds_effort_name(s: &str) -> Option<u32> {
    Some(match s {
        "minimal" => 25,
        "low" => 50,
        "medium" => 62,
        "high" | "xhigh" => 75,
        "max" => 100,
        _ => return None,
    })
}

/// `--default-reasoning-effort` for ds41: a name or an integer 1..=100. `none` is refused, as for GLM: it would turn
/// thinking off for every request.
pub fn parse_ds_default_effort(s: &str) -> Result<u32, String> {
    if let Some(n) = ds_effort_name(s) {
        return Ok(n);
    }
    match s.parse::<u32>() {
        Ok(n) if (1..=100).contains(&n) => Ok(n),
        _ => Err(format!("--default-reasoning-effort {s}: want an integer from 1 to 100, or minimal, low, medium, high, xhigh or max")),
    }
}

fn non_null<'a>(m: Option<&'a Map<String, Value>>, k: &str) -> Option<&'a Value> {
    m.and_then(|m| m.get(k)).filter(|v| !v.is_null())
}

/// A request's effort: Some(None) is `none` (thinking off), Some(Some(n)) a number, None names nothing.
fn read_effort(v: &Value) -> Result<Option<u32>, ApiError> {
    let bad = || ApiError::bad(DS_EFFORT_ERROR, Some("reasoning_effort"));
    match v {
        Value::String(s) if s == "none" => Ok(None),
        Value::String(s) => ds_effort_name(s).map(Some).ok_or_else(bad),
        Value::Number(n) => match n.as_i64() {
            Some(i) if (1..=100).contains(&i) => Ok(Some(i as u32)),
            _ => Err(bad()),
        },
        _ => Err(bad()),
    }
}

/// DeepSeek's thinking controls for one request, read the way the lane's `encoding_compat.py` does:
/// * `chat_template_kwargs.thinking` and `enable_thinking` are aliases; `thinking` wins when they disagree;
/// * `reasoning_effort` at the top level, else in `chat_template_kwargs`: an integer 1..=100 or a name; `none` turns
///   thinking off unless the kwargs switch says on;
/// * `thinking_budget` as for GLM (`thinking::parse_budget`).
///
/// Returns the common `Thinking` (`effort` unused: DeepSeek's is a number) and the number the prompt writes. Thinking is
/// on unless a switch or `none` turns it off; the server needs no `--allow-thinking-off` for DeepSeek.
pub fn resolve_ds(
    body: &Map<String, Value>,
    defaults: thinking::Defaults,
    default_effort: u32,
) -> Result<(Thinking, u32), ApiError> {
    let kw = body.get("chat_template_kwargs").and_then(Value::as_object);
    let t = non_null(kw, "thinking").map(|v| pyjson::truthy(Some(v)));
    let e = non_null(kw, "enable_thinking").map(|v| pyjson::truthy(Some(v)));
    let switch = t.or(e);
    let named = non_null(Some(body), "reasoning_effort").or_else(|| non_null(kw, "reasoning_effort"));
    let (effort, none) = match named {
        None => (None, false),
        Some(v) => match read_effort(v)? {
            Some(n) => (Some(n), false),
            None => (None, true),
        },
    };
    let off = match switch {
        Some(s) => !s,
        None => none,
    };
    if off {
        return Ok((
            Thinking { effort: None, budget: Limit::Tokens(0), budget_explicit: false, off: true },
            default_effort,
        ));
    }
    let (budget, budget_explicit) = match thinking::parse_budget(body.get("thinking_budget"))? {
        None | Some(0) => (defaults.budget, false),
        Some(n) if n < 0 => (Limit::Tokens(0), false),
        Some(n) => (Limit::Tokens(n as u64), true),
    };
    Ok((Thinking { effort: None, budget, budget_explicit, off: false }, effort.unwrap_or(default_effort)))
}

// ---- the prompt ------------------------------------------------------------------------------------------------------

/// The flags that shape the end of a chat prompt: `add_generation_prompt` (default true) and `continue_final_message`
/// (default false). Both explicitly true is a 400, as in vLLM; `continue_final_message` alone leaves the header off.
pub fn ds_gen_flags(body: &Map<String, Value>) -> Result<(bool, bool), ApiError> {
    let agp = body.get("add_generation_prompt");
    let cfm = pyjson::truthy(body.get("continue_final_message"));
    if cfm && agp.is_some_and(|v| pyjson::truthy(Some(v))) {
        return Err(ApiError::bad(
            "Cannot set both `continue_final_message` and `add_generation_prompt` to True.",
            None,
        ));
    }
    Ok((!cfm && agp.is_none_or(|v| pyjson::truthy(Some(v))), cfm))
}

fn render_err(e: crate::dschat::RenderError) -> ApiError {
    ApiError::bad(e.0, Some("messages"))
}

fn lists<'a>(messages: Option<&'a Value>, tools: Option<&'a Value>) -> Result<(&'a [Value], &'a [Value]), ApiError> {
    let Some(Value::Array(m)) = messages else {
        return Err(ApiError::bad("messages must be a list", Some("messages")));
    };
    Ok((m.as_slice(), tools.and_then(Value::as_array).map_or(&[][..], Vec::as_slice)))
}

/// The prompt text for a chat request: the release's encoder, then the generation header off for
/// `add_generation_prompt: false` and the closing end-of-sentence off for `continue_final_message`. It starts with the
/// BOS text, so it is encoded without special tokens added. Also returns the request's image blocks in prompt order,
/// one `<｜deepseek_image｜>` in the text for each.
pub fn ds_prompt(
    messages: Option<&Value>,
    tools: Option<&Value>,
    thinking: bool,
    effort: u32,
    add_generation_prompt: bool,
    continue_final: bool,
) -> Result<(String, Vec<Value>), ApiError> {
    use crate::dschat::{self, Opts};
    let (m, t) = lists(messages, tools)?;
    let (mut text, images) = dschat::render_mm(m, t, &Opts { thinking, effort }).map_err(render_err)?;
    if continue_final {
        match text.strip_suffix(dschat::EOS) {
            Some(open) if m.last().and_then(|x| x.get("role")).and_then(Value::as_str) == Some("assistant") => {
                text.truncate(open.len())
            }
            _ => {
                return Err(ApiError::bad(
                    "continue_final_message needs the last message to be from the assistant",
                    Some("messages"),
                ))
            }
        }
    } else if !add_generation_prompt {
        for tail in [dschat::THINK_OPEN, dschat::THINK_CLOSE] {
            if let Some(head) = text.strip_suffix(&format!("{}{tail}", dschat::ASSISTANT)) {
                text.truncate(head.len());
                break;
            }
        }
    }
    Ok((text, images))
}

/// The prompt's system span in the same render: the leading system or developer messages and the tools alone (an empty
/// system message stands in when there are none, as the renderer adds one for the tools). The caller counts the tokens
/// it shares with the whole prompt. None when that cannot render.
pub fn ds_system_text(messages: Option<&Value>, tools: Option<&Value>, thinking: bool, effort: u32) -> Option<String> {
    use crate::dschat::{self, Opts};
    let lead: Vec<Value> = messages?
        .as_array()?
        .iter()
        .take_while(|m| matches!(m.get("role").and_then(Value::as_str), Some("system" | "developer")))
        .cloned()
        .collect();
    let t = tools.and_then(Value::as_array).map_or(&[][..], Vec::as_slice);
    let lead = if lead.is_empty() { vec![serde_json::json!({"role": "system", "content": ""})] } else { lead };
    dschat::render(&lead, t, &Opts { thinking, effort }).ok()
}

// ---- forced tool_choice ----------------------------------------------------------------------------------------------

/// What a forced `tool_choice` puts in the reply: the start of a DSML call block, as the encoder writes one after an
/// empty answer. A named function carries its name and the start of its parameters; `required` stops after `invoke`, so
/// the model writes ` name="..."` itself and the parser checks the name against the offered tools.
pub fn ds_forced_text(name: Option<&str>) -> String {
    use crate::dschat::{CALLS_OPEN, INVOKE_OPEN};
    match name {
        Some(n) => format!("\n\n{CALLS_OPEN}\n{INVOKE_OPEN} name=\"{n}\">\n"),
        None => format!("\n\n{CALLS_OPEN}\n{INVOKE_OPEN}"),
    }
}

// ---- generic HuggingFace thinking and forced tool_choice ------------------------------------------------------------

/// Generic HuggingFace thinking controls (`--dialect hf`), with native thinking-off support (no `--allow-thinking-off`
/// required):
/// * `chat_template_kwargs.enable_thinking` and `thinking` are aliases (`enable_thinking` wins when both are set);
/// * `reasoning_effort` at top level, else in `chat_template_kwargs`: `none` turns thinking off unless the switch says
///   on; other valid levels map to the nearest level named in the template (`thinking::nearest`);
/// * `thinking_budget` follows `thinking::parse_budget`.
///
/// Returns `(Thinking, Option<bool>)`, where the second element is the resolved `enable_thinking` value to write into
/// template kwargs when a switch or `off` state was requested (`None` leaves the template's own default untouched).
pub fn resolve_hf(
    body: &Map<String, Value>,
    levels: &[&'static str],
    defaults: thinking::Defaults,
) -> Result<(Thinking, Option<bool>), ApiError> {
    let kw = body.get("chat_template_kwargs").and_then(Value::as_object);
    let e = non_null(kw, "enable_thinking").map(|v| pyjson::truthy(Some(v)));
    let t = non_null(kw, "thinking").map(|v| pyjson::truthy(Some(v)));
    let switch = e.or(t);
    let explicit = match non_null(Some(body), "reasoning_effort").or_else(|| non_null(kw, "reasoning_effort")) {
        None => None,
        Some(v) => match v.as_str().and_then(|s| thinking::EFFORTS.iter().find(|x| **x == s)) {
            Some(x) => Some(*x),
            None => {
                return Err(ApiError::bad(
                    "reasoning_effort must be none, minimal, low, medium, high, xhigh or max",
                    Some("reasoning_effort"),
                ))
            }
        },
    };
    let off = match switch {
        Some(s) => !s,
        None => explicit == Some("none"),
    };
    if off {
        return Ok((
            Thinking { effort: None, budget: Limit::Tokens(0), budget_explicit: false, off: true },
            Some(false),
        ));
    }
    let explicit = if explicit == Some("none") { None } else { explicit };
    let effort = explicit.or(defaults.effort).map(|x| thinking::nearest(x, levels));
    let (budget, budget_explicit) = match thinking::parse_budget(body.get("thinking_budget"))? {
        None | Some(0) => (defaults.budget, false),
        Some(n) if n < 0 => (Limit::Tokens(0), false),
        Some(n) => (Limit::Tokens(n as u64), true),
    };
    Ok((Thinking { effort, budget, budget_explicit, off: false }, switch))
}

/// Close an unclosed generation-prompt `<think>` block when thinking was turned off on `--dialect hf` and the template
/// itself did not close it.
pub fn hf_thinking_off(mut text: String, add_generation_prompt: bool) -> String {
    if add_generation_prompt && pyjson::py_rstrip(&text).ends_with("<think>") {
        text.push_str("</think>\n\n");
    }
    text
}

/// What a forced `tool_choice` puts at the start of the reply on `--dialect hf`: a standard JSON `<tool_call>` block.
pub fn hf_forced_text(name: Option<&str>) -> String {
    match name {
        Some(n) => {
            let mut q = Vec::with_capacity(n.len() + 2);
            pyjson::string(&mut q, n);
            let q = String::from_utf8(q).unwrap();
            format!("<tool_call>\n{{\"name\": {q}, \"arguments\": ")
        }
        None => "<tool_call>\n{\"name\": \"".to_string(),
    }
}

// ---- Inkling thinking, effort and forced calls -----------------------------------------------------------------------

const INK_EFFORT_ERROR: &str =
    "reasoning_effort must be none, minimal, low, medium, high, xhigh, max or a number from 0.0 to 0.99";

/// An Inkling effort value the template accepts: a level name or a JSON number from 0.0 to 0.99.
fn ink_effort_ok(v: &Value) -> bool {
    match v {
        Value::String(s) => thinking::EFFORTS.contains(&s.as_str()),
        Value::Number(n) => n.as_f64().is_some_and(|f| (0.0..=0.99).contains(&f)),
        _ => false,
    }
}

/// The effort is the template's "no thinking" level: `none` or the number 0.
fn ink_effort_zero(v: &Value) -> bool {
    v.as_str() == Some("none") || (v.is_number() && v.as_f64() == Some(0.0))
}

/// `--default-reasoning-effort` on `--dialect inkling`: a level name (`none` included) or a number from 0.0 to 0.99.
pub fn parse_inkling_default_effort(s: &str) -> Result<Value, String> {
    let t = s.trim();
    if thinking::EFFORTS.contains(&t) {
        return Ok(Value::from(t));
    }
    match t.parse::<f64>() {
        Ok(f) if (0.0..=0.99).contains(&f) => Ok(Value::from(f)),
        _ => Err(format!(
            "--default-reasoning-effort {s}: want none, minimal, low, medium, high, xhigh, max or a number from 0.0 \
             to 0.99"
        )),
    }
}

/// A request's thinking on `--dialect inkling`. The template takes `reasoning_effort` as a level name or a number
/// from 0.0 to 0.99, prints it as a system line, and uses 0.9 when none is given; effort 0 is thinking off.
///
/// * `chat_template_kwargs.enable_thinking` and `thinking` are aliases (`enable_thinking` wins when both are set);
/// * `reasoning_effort` at top level, else in `chat_template_kwargs`, else `default_effort` (the server's
///   `--default-reasoning-effort`);
/// * thinking is off when the switch says off, or when no switch is set and the effort is `none` or 0;
/// * `thinking_budget` follows `thinking::parse_budget`.
///
/// The second element is the `reasoning_effort` to hand the template (`None` leaves its own default). Thinking off
/// always hands it `none` (or the request's numeric 0), so the prompt carries effort level 0.
pub fn resolve_inkling(
    body: &Map<String, Value>,
    defaults: thinking::Defaults,
    default_effort: Option<&Value>,
) -> Result<(Thinking, Option<Value>), ApiError> {
    let kw = body.get("chat_template_kwargs").and_then(Value::as_object);
    let e = non_null(kw, "enable_thinking").map(|v| pyjson::truthy(Some(v)));
    let t = non_null(kw, "thinking").map(|v| pyjson::truthy(Some(v)));
    let switch = e.or(t);
    let named = non_null(Some(body), "reasoning_effort").or_else(|| non_null(kw, "reasoning_effort"));
    if named.is_some_and(|v| !ink_effort_ok(v)) {
        return Err(ApiError::bad(INK_EFFORT_ERROR, Some("reasoning_effort")));
    }
    let effective = named.or(default_effort);
    let zero = effective.is_some_and(ink_effort_zero);
    let off = match switch {
        Some(s) => !s,
        None => zero,
    };
    if off {
        let v = match effective {
            Some(v) if v.is_number() && zero => v.clone(),
            _ => Value::from("none"),
        };
        return Ok((Thinking { effort: None, budget: Limit::Tokens(0), budget_explicit: false, off: true }, Some(v)));
    }
    // The switch says on but the effort says none: fall back to a level that thinks.
    let effort = match effective {
        Some(v) if !zero => Some(v.clone()),
        _ => default_effort.filter(|v| !ink_effort_zero(v)).cloned(),
    };
    let (budget, budget_explicit) = match thinking::parse_budget(body.get("thinking_budget"))? {
        None | Some(0) => (defaults.budget, false),
        Some(n) if n < 0 => (Limit::Tokens(0), false),
        Some(n) => (Limit::Tokens(n as u64), true),
    };
    Ok((Thinking { effort: None, budget, budget_explicit, off: false }, effort))
}

/// What a forced `tool_choice` puts at the start of the reply on `--dialect inkling`: the start of a JSON tool-call
/// block. A named choice, or `required` with one declared tool, carries the name in the header and in the JSON;
/// `required` with several tools stops inside the name string so the model picks one.
pub fn ink_forced_text(tools: Option<&Value>, choice: &ToolChoice) -> Option<String> {
    let named = |name: &str| {
        let mut q = Vec::with_capacity(name.len() + 2);
        pyjson::string(&mut q, name);
        let q = String::from_utf8(q).unwrap();
        format!("<|message_model|>{name}<|content_invoke_tool_json|>{{\"name\":{q},\"args\":")
    };
    match choice {
        ToolChoice::Named(name) => Some(named(name)),
        ToolChoice::Required => {
            let names = crate::tool_choice::tool_names(tools);
            match names.as_slice() {
                [one] => Some(named(one)),
                _ => Some("<|message_model|><|content_invoke_tool_json|>{\"name\":\"".to_string()),
            }
        }
        ToolChoice::None | ToolChoice::Auto => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn body(v: Value) -> Map<String, Value> {
        v.as_object().unwrap().clone()
    }

    fn res(v: Value) -> Result<(Thinking, u32), ApiError> {
        resolve_ds(&body(v), thinking::Defaults::default(), DS_DEFAULT_EFFORT)
    }

    #[test]
    fn nothing_named_is_thinking_on_at_the_default_effort() {
        let (t, e) = res(json!({})).unwrap();
        assert!(!t.off);
        assert_eq!((t.budget, t.budget_explicit, e), (Limit::Tokens(0), false, 75));
    }

    #[test]
    fn thinking_and_enable_thinking_are_aliases_and_thinking_wins() {
        for (kw, off) in [
            (json!({"thinking": false}), true),
            (json!({"enable_thinking": false}), true),
            (json!({"thinking": true}), false),
            (json!({"enable_thinking": true}), false),
            (json!({"thinking": true, "enable_thinking": false}), false),
            (json!({"thinking": false, "enable_thinking": true}), true),
        ] {
            let (t, _) = res(json!({ "chat_template_kwargs": kw })).unwrap();
            assert_eq!(t.off, off, "{kw}");
        }
    }

    #[test]
    fn effort_is_a_number_or_a_name() {
        let want = [
            (json!(1), 1),
            (json!(100), 100),
            (json!(50), 50),
            (json!("low"), 50),
            (json!("high"), 75),
            (json!("xhigh"), 75),
            (json!("max"), 100),
            (json!("minimal"), 25),
            (json!("medium"), 62),
        ];
        for (v, n) in want {
            assert_eq!(res(json!({ "reasoning_effort": v })).unwrap().1, n, "{v}");
            assert_eq!(res(json!({"chat_template_kwargs": { "reasoning_effort": v }})).unwrap().1, n, "{v}");
        }
        // top level wins over the kwargs
        assert_eq!(
            res(json!({"reasoning_effort": "low", "chat_template_kwargs": {"reasoning_effort": 99}})).unwrap().1,
            50
        );
    }

    #[test]
    fn an_effort_out_of_range_or_of_the_wrong_kind_is_a_400() {
        for v in [json!(0), json!(101), json!(-5), json!(50.5), json!("auto"), json!("50"), json!(true), json!([1])] {
            let e = res(json!({ "reasoning_effort": v })).err().unwrap_or_else(|| panic!("{v} should be refused"));
            assert_eq!((e.status, e.param.as_deref()), (400, Some("reasoning_effort")), "{v}");
        }
    }

    #[test]
    fn effort_none_turns_thinking_off_unless_the_switch_says_on() {
        assert!(res(json!({"reasoning_effort": "none"})).unwrap().0.off);
        assert!(res(json!({"chat_template_kwargs": {"reasoning_effort": "none"}})).unwrap().0.off);
        let on = res(json!({"reasoning_effort": "none", "chat_template_kwargs": {"thinking": true}})).unwrap();
        assert!(!on.0.off);
        assert_eq!(on.1, 75);
        // an explicit off with an effort: still off
        assert!(res(json!({"reasoning_effort": "high", "chat_template_kwargs": {"thinking": false}})).unwrap().0.off);
    }

    #[test]
    fn the_budget_follows_the_glm_rules_and_the_server_default_effort_is_used() {
        let d = thinking::Defaults { effort: None, budget: Limit::Tokens(900), allow_off: false };
        let r = |v: Value| resolve_ds(&body(v), d, 50).unwrap();
        let (t, e) = r(json!({}));
        assert_eq!((t.budget, t.budget_explicit, e), (Limit::Tokens(900), false, 50));
        let (t, _) = r(json!({"thinking_budget": 100}));
        assert_eq!((t.budget, t.budget_explicit), (Limit::Tokens(100), true));
        let (t, _) = r(json!({"thinking_budget": -1}));
        assert_eq!((t.budget, t.budget_explicit), (Limit::Tokens(0), false));
        let e = resolve_ds(&body(json!({"thinking_budget": "x"})), d, 50).err().unwrap();
        assert_eq!(e.status, 400);
    }

    #[test]
    fn server_default_effort_accepts_names_and_integers() {
        assert_eq!(parse_ds_default_effort("low"), Ok(50));
        assert_eq!(parse_ds_default_effort("37"), Ok(37));
        assert!(parse_ds_default_effort("none").is_err());
        assert!(parse_ds_default_effort("0").is_err());
        assert!(parse_ds_default_effort("101").is_err());
    }

    #[test]
    fn forced_text_is_the_start_of_the_encoders_call_block() {
        assert_eq!(
            ds_forced_text(Some("get_weather")),
            "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n"
        );
        assert_eq!(ds_forced_text(None), "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke");
    }

    /// With the real DeepSeek tokenizer: GR8_TEST_DS41_TOKENIZER_DIR=<dir with tokenizer.json, tokenizer_config.json>.
    #[test]
    fn the_real_tokenizer_reads_the_prompt_the_way_the_engine_expects() {
        use crate::gen::think_open;
        use crate::tool_choice::ToolChoice;
        let Some(dir) = gr8knobs::TEST_DS41_TOKENIZER_DIR.get() else {
            eprintln!("GR8_TEST_DS41_TOKENIZER_DIR not set; skipping");
            return;
        };
        let mut t = crate::tok::Tok::load(std::path::Path::new(&dir)).unwrap();
        t.use_single_think_close();
        assert_eq!((t.eos_token_id, t.eos_ids.clone()), (Some(1), vec![1]));
        assert_eq!((t.think_start, t.think_end, t.think_close.clone()), (Some(128821), Some(128822), vec![128822]));
        assert!(t.tk.token_to_id("｜DSML｜").is_some());
        let msgs = json!([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}]);
        // thinking on: BOS first, the header and an open <think> last
        let on = ds_prompt(Some(&msgs), None, true, 75, true, false).unwrap().0;
        let ids = t.encode(&on, false).unwrap();
        assert_eq!(ids[0], 0);
        assert_eq!(&ids[ids.len() - 2..], &[128804, 128821]);
        assert!(think_open(&ids, 128821, 128822));
        // thinking off: the header and a closed block
        let off = ds_prompt(Some(&msgs), None, false, 75, true, false).unwrap().0;
        let ids = t.encode(&off, false).unwrap();
        assert_eq!(&ids[ids.len() - 2..], &[128804, 128822]);
        assert!(!think_open(&ids, 128821, 128822));
        // add_generation_prompt false drops the header; continue_final_message keeps an assistant turn open
        let bare = ds_prompt(Some(&msgs), None, true, 75, false, false).unwrap().0;
        assert!(bare.ends_with("hi") && !bare.contains("<｜Assistant｜>"));
        let cont = json!([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hel"}]);
        let c = ds_prompt(Some(&cont), None, false, 75, false, true).unwrap().0;
        assert!(c.ends_with("</think>Hel"), "{c:?}");
        assert!(ds_prompt(Some(&msgs), None, false, 75, false, true).is_err());
        // the system span: the leading system message and the tools, the same tokens at the front of the whole prompt
        let tools = json!([{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object", "properties": {}}}}]);
        let full = ds_prompt(Some(&msgs), Some(&tools), true, 75, true, false).unwrap().0;
        let sys = ds_system_text(Some(&msgs), Some(&tools), true, 75).unwrap();
        assert!(full.starts_with(&sys), "system text is a prefix of the prompt");
        let (full_ids, sys_ids) = (t.encode(&full, false).unwrap(), t.encode(&sys, false).unwrap());
        assert_eq!(full_ids[..sys_ids.len()], sys_ids[..]);
        // forced tool_choice: the one-token close, then the call opener, as the text says
        let named = ToolChoice::Named("f".into());
        let ids = named.forced_ids_ds(&t, &t.think_close).unwrap();
        assert_eq!(ids[0], 128822);
        let back = t.decode(&ids[1..], false);
        assert_eq!(back, ds_forced_text(Some("f")));
        assert!(ToolChoice::Required.forced_ids_ds(&t, &[]).unwrap().len() < 64);
        // images: each placeholder is the one image token, in a short prompt and in a long one the parallel encode cuts
        let img = json!({"type": "image", "data": "QUJD"});
        for pad in [0usize, 200_000] {
            let m = json!([{"role": "user", "content": [{"type": "text", "text": "x".repeat(pad)}, img, {"type": "text", "text": "y"}, img]}]);
            let (text, images) = ds_prompt(Some(&m), None, true, 75, true, false).unwrap();
            assert_eq!(images.len(), 2);
            let ids = t.encode(&text, false).unwrap();
            assert_eq!(ids.iter().filter(|&&i| i == crate::vision::IMAGE_TOKEN_ID).count(), 2);
            assert_eq!(t.tk.token_to_id(crate::dschat::IMAGE_PLACEHOLDER), Some(crate::vision::IMAGE_TOKEN_ID));
        }
    }

    #[test]
    fn the_flag_parses() {
        assert_eq!(Dialect::parse_flag("ds41"), Ok(Some(Dialect::Ds41)));
        assert_eq!(Dialect::parse_flag("glm"), Ok(Some(Dialect::Glm)));
        assert_eq!(Dialect::parse_flag("hf"), Ok(Some(Dialect::Hf)));
        assert_eq!(Dialect::parse_flag("auto"), Ok(None));
        assert!(Dialect::parse_flag("qwen").is_err());
        assert_eq!(Dialect::parse_flag("inkling"), Ok(Some(Dialect::Inkling)));
        assert_eq!(Dialect::parse_flag("qwen").unwrap_err(), "--dialect qwen: want glm, ds41, hf, inkling or auto");
        assert_eq!((Dialect::Glm.name(), Dialect::Ds41.name(), Dialect::Hf.name()), ("glm", "ds41", "hf"));
        assert_eq!(Dialect::Inkling.name(), "inkling");
    }

    #[test]
    fn detect_handles_glm_deepseek_missing_and_hf_architectures() {
        let dir = std::env::temp_dir().join(format!("g6front-dialect-detect-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // Missing config.json -> stays Glm
        assert_eq!(Dialect::detect(&dir, false), Dialect::Glm);
        // Missing config.json + DSML token -> Ds41
        assert_eq!(Dialect::detect(&dir, true), Dialect::Ds41);

        let check = |cfg: Value, has_dsml: bool, want: Dialect| {
            std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
            assert_eq!(Dialect::detect(&dir, has_dsml), want, "cfg={cfg}");
        };

        // Empty / missing / empty-array / empty-string architectures -> stays Glm
        check(json!({}), false, Dialect::Glm);
        check(json!({"architectures": []}), false, Dialect::Glm);
        check(json!({"architectures": [""]}), false, Dialect::Glm);

        // GLM / ChatGLM architectures (case-insensitive) -> Glm
        check(json!({"architectures": ["GlmMoeDsaForCausalLM"]}), false, Dialect::Glm);
        check(json!({"architectures": ["Glm4ForCausalLM"]}), false, Dialect::Glm);
        check(json!({"architectures": ["ChatGLMModel"]}), false, Dialect::Glm);
        check(json!({"architectures": ["glm5_next"]}), false, Dialect::Glm);

        // DeepSeekV4 -> Ds41
        check(json!({"architectures": ["DeepseekV41ForCausalLM"]}), false, Dialect::Ds41);

        // Generic HuggingFace architectures -> Hf
        check(json!({"architectures": ["Qwen3ForCausalLM"]}), false, Dialect::Hf);
        check(json!({"architectures": ["Qwen2ForCausalLM"]}), false, Dialect::Hf);
        check(json!({"architectures": ["LlamaForCausalLM"]}), false, Dialect::Hf);
        check(json!({"architectures": ["MistralForCausalLM"]}), false, Dialect::Hf);
        check(json!({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}), false, Dialect::Hf);

        // Inkling: by architecture (any case) or by model_type alone
        check(
            json!({"architectures": ["Inkling2ForConditionalGeneration"], "model_type": "inkling2"}),
            false,
            Dialect::Inkling,
        );
        check(json!({"architectures": ["inkling2ForConditionalGeneration"]}), false, Dialect::Inkling);
        check(json!({"model_type": "inkling2"}), false, Dialect::Inkling);
        check(json!({"architectures": [], "model_type": "inkling2"}), false, Dialect::Inkling);
        // the DSML token still wins
        check(json!({"model_type": "inkling2"}), true, Dialect::Ds41);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn hf_served_name_uses_directory_basename_or_fallback() {
        assert_eq!(hf_served_name(Path::new("/models/Qwen3-8B")), "Qwen3-8B");
        assert_eq!(hf_served_name(Path::new("/models/Llama-3.1-8B-Instruct/")), "Llama-3.1-8B-Instruct");
        assert_eq!(hf_served_name(Path::new("")), "hf-model");
        assert_eq!(hf_served_name(Path::new("/")), "hf-model");
        assert_eq!(hf_served_name(Path::new(".")), "hf-model");
    }

    #[test]
    fn inkling_served_names_add_the_directory_basename_once() {
        assert_eq!(
            inkling_served_names(Path::new("/models/Inkling-2-Small")),
            ["inkling-2-small", "ink2", "Inkling-2-Small"]
        );
        assert_eq!(inkling_served_names(Path::new("/models/ink2/")), ["inkling-2-small", "ink2"]);
        assert_eq!(inkling_served_names(Path::new("/")), ["inkling-2-small", "ink2"]);
        assert_eq!(inkling_served_names(Path::new("")), ["inkling-2-small", "ink2"]);
    }

    #[test]
    fn resolve_hf_supports_native_thinking_off_efforts_and_budgets() {
        let levels = ["high", "medium", "low"];
        let d = thinking::Defaults { effort: None, budget: Limit::Tokens(256), allow_off: true };
        let r = |v: Value| resolve_hf(&body(v), &levels, d);

        // Unset: leaves template's own default untouched (None switch)
        let (th, sw) = r(json!({})).unwrap();
        assert!(!th.off);
        assert_eq!((th.effort, th.budget, sw), (None, Limit::Tokens(256), None));

        // enable_thinking / thinking switches work without requiring --allow-thinking-off
        for kw in [json!({"enable_thinking": false}), json!({"thinking": false})] {
            let (th, sw) =
                resolve_hf(&body(json!({"chat_template_kwargs": kw})), &levels, thinking::Defaults::default()).unwrap();
            assert!(th.off);
            assert_eq!(sw, Some(false));
        }
        let (th, sw) = r(json!({"chat_template_kwargs": {"thinking": true}})).unwrap();
        assert!(!th.off);
        assert_eq!(sw, Some(true));

        // reasoning_effort: "none" turns thinking off unless switch is explicitly true
        let (th, sw) = r(json!({"reasoning_effort": "none"})).unwrap();
        assert!(th.off);
        assert_eq!(sw, Some(false));
        let (th, sw) =
            r(json!({"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": true}})).unwrap();
        assert!(!th.off);
        assert_eq!((th.effort, sw), (None, Some(true)));

        // Named effort maps to nearest template level; invalid effort returns 400
        let (th, _) = r(json!({"reasoning_effort": "max"})).unwrap();
        assert_eq!(th.effort, Some("high"));
        assert_eq!(r(json!({"reasoning_effort": "bogus"})).unwrap_err().status, 400);
    }

    #[test]
    fn hf_thinking_off_and_forced_text() {
        assert_eq!(
            hf_thinking_off("<|im_start|>assistant\n<think>".into(), true),
            "<|im_start|>assistant\n<think></think>\n\n"
        );
        assert_eq!(
            hf_thinking_off("<|im_start|>assistant\n<think>\n".into(), true),
            "<|im_start|>assistant\n<think>\n</think>\n\n"
        );
        assert_eq!(
            hf_thinking_off("<|im_start|>assistant\n<think>\n\n</think>\n\n".into(), true),
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        assert_eq!(hf_thinking_off("<|im_start|>assistant\n<think>".into(), false), "<|im_start|>assistant\n<think>");
        assert_eq!(hf_forced_text(Some("foo")), "<tool_call>\n{\"name\": \"foo\", \"arguments\": ");
        assert_eq!(hf_forced_text(None), "<tool_call>\n{\"name\": \"");
    }

    #[test]
    fn resolve_inkling_hands_the_template_its_effort_and_turns_thinking_off_at_zero() {
        let d = thinking::Defaults { effort: None, budget: Limit::Tokens(256), allow_off: true };
        let r = |v: Value| resolve_inkling(&body(v), d, None);

        // nothing named: thinking on, the template's own default (0.9) stays
        let (th, e) = r(json!({})).unwrap();
        assert!(!th.off);
        assert_eq!((th.effort, th.budget, th.budget_explicit, e), (None, Limit::Tokens(256), false, None));
        let (th, e) = r(json!({"reasoning_effort": null})).unwrap();
        assert_eq!((th.off, e), (false, None));

        // off: the template still gets the zero effort
        for v in [
            json!({"reasoning_effort": "none"}),
            json!({"chat_template_kwargs": {"reasoning_effort": "none"}}),
            json!({"chat_template_kwargs": {"enable_thinking": false}}),
            json!({"chat_template_kwargs": {"thinking": false}}),
            json!({"reasoning_effort": "high", "chat_template_kwargs": {"enable_thinking": false}}),
            json!({"chat_template_kwargs": {"enable_thinking": false, "thinking": true}}),
        ] {
            let (th, e) = r(v.clone()).unwrap();
            assert!(th.off, "{v}");
            assert_eq!((th.budget, e), (Limit::Tokens(0), Some(json!("none"))), "{v}");
        }
        for zero in [json!(0.0), json!(0)] {
            let (th, e) = r(json!({"reasoning_effort": zero})).unwrap();
            assert!(th.off);
            assert_eq!(e, Some(zero));
        }

        // names and numbers pass through; the top level wins over the kwargs
        for v in [json!("minimal"), json!("medium"), json!("max"), json!(0.35), json!(0.99), json!(0.9)] {
            let (th, e) = r(json!({"reasoning_effort": v})).unwrap();
            assert!(!th.off);
            assert_eq!((th.effort, e), (None, Some(v)));
        }
        let (_, e) =
            r(json!({"reasoning_effort": "low", "chat_template_kwargs": {"reasoning_effort": "high"}})).unwrap();
        assert_eq!(e, Some(json!("low")));
        let (_, e) = r(json!({"chat_template_kwargs": {"reasoning_effort": 0.5}})).unwrap();
        assert_eq!(e, Some(json!(0.5)));

        // out of range or the wrong kind is a 400 on reasoning_effort
        for bad in [json!(1.5), json!(1), json!(-0.1), json!("bogus"), json!(true), json!([0.5]), json!("0.5")] {
            let err = r(json!({"reasoning_effort": bad})).unwrap_err();
            assert_eq!((err.status, err.param.as_deref()), (400, Some("reasoning_effort")), "{bad}");
        }
        assert_eq!(r(json!({"chat_template_kwargs": {"reasoning_effort": 2}})).unwrap_err().status, 400);

        // effort none with the switch on: thinking on at the template default
        for zero in [json!("none"), json!(0.0)] {
            let (th, e) =
                r(json!({"reasoning_effort": zero, "chat_template_kwargs": {"enable_thinking": true}})).unwrap();
            assert_eq!((th.off, e), (false, None));
        }

        // thinking_budget follows the shared rules
        let (th, _) = r(json!({"thinking_budget": 64})).unwrap();
        assert_eq!((th.budget, th.budget_explicit), (Limit::Tokens(64), true));
        let (th, _) = r(json!({"thinking_budget": -1})).unwrap();
        assert_eq!((th.budget, th.budget_explicit), (Limit::Tokens(0), false));
        assert_eq!(r(json!({"thinking_budget": "x"})).unwrap_err().status, 400);
    }

    #[test]
    fn resolve_inkling_uses_the_server_default_effort() {
        let d = thinking::Defaults { effort: None, budget: Limit::Tokens(0), allow_off: true };
        let low = json!("low");
        let (th, e) = resolve_inkling(&body(json!({})), d, Some(&low)).unwrap();
        assert_eq!((th.off, e), (false, Some(json!("low"))));
        let (_, e) = resolve_inkling(&body(json!({"reasoning_effort": 0.4})), d, Some(&low)).unwrap();
        assert_eq!(e, Some(json!(0.4)));
        // none with the switch on falls back to a default that thinks
        let (th, e) = resolve_inkling(
            &body(json!({"reasoning_effort": "none", "chat_template_kwargs": {"thinking": true}})),
            d,
            Some(&low),
        )
        .unwrap();
        assert_eq!((th.off, e), (false, Some(json!("low"))));

        // a server default of none (or 0) is thinking off until a request names an effort or the switch
        for zero in [json!("none"), json!(0.0)] {
            let (th, e) = resolve_inkling(&body(json!({})), d, Some(&zero)).unwrap();
            assert!(th.off);
            assert_eq!(e, Some(zero.clone()));
            let (th, e) = resolve_inkling(&body(json!({"reasoning_effort": "high"})), d, Some(&zero)).unwrap();
            assert_eq!((th.off, e), (false, Some(json!("high"))));
            let (th, e) =
                resolve_inkling(&body(json!({"chat_template_kwargs": {"enable_thinking": true}})), d, Some(&zero))
                    .unwrap();
            assert_eq!((th.off, e), (false, None));
        }
    }

    #[test]
    fn inkling_default_effort_accepts_names_and_numbers_up_to_099() {
        assert_eq!(parse_inkling_default_effort("none"), Ok(json!("none")));
        assert_eq!(parse_inkling_default_effort("xhigh"), Ok(json!("xhigh")));
        assert_eq!(parse_inkling_default_effort("0.35"), Ok(json!(0.35)));
        assert_eq!(parse_inkling_default_effort("0"), Ok(json!(0.0)));
        assert_eq!(parse_inkling_default_effort("0.99"), Ok(json!(0.99)));
        for bad in ["1", "1.5", "-0.1", "bogus", "", "nan", "inf"] {
            assert!(parse_inkling_default_effort(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn inkling_forced_text_opens_a_json_tool_call_block() {
        let one = json!([{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]);
        let two = json!([
            {"type": "function", "function": {"name": "get_weather", "parameters": {}}},
            {"type": "function", "function": {"name": "get_time", "parameters": {}}}
        ]);
        let named = "<|message_model|>get_time<|content_invoke_tool_json|>{\"name\":\"get_time\",\"args\":";
        assert_eq!(ink_forced_text(Some(&two), &ToolChoice::Named("get_time".into())).as_deref(), Some(named));
        assert_eq!(
            ink_forced_text(Some(&one), &ToolChoice::Required).as_deref(),
            Some("<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":")
        );
        assert_eq!(
            ink_forced_text(Some(&two), &ToolChoice::Required).as_deref(),
            Some("<|message_model|><|content_invoke_tool_json|>{\"name\":\"")
        );
        assert_eq!(ink_forced_text(Some(&two), &ToolChoice::Auto), None);
        assert_eq!(ink_forced_text(Some(&two), &ToolChoice::None), None);
        // the name is a JSON string in the body
        assert_eq!(
            ink_forced_text(None, &ToolChoice::Named("a\"b".into())).as_deref(),
            Some("<|message_model|>a\"b<|content_invoke_tool_json|>{\"name\":\"a\\\"b\",\"args\":")
        );
    }
}
