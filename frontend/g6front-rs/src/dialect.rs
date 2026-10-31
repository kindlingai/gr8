//! The front end speaks two model dialects. `glm` is GLM 5.3: a jinja chat template, `<tool_call>` calls, a thinking
//! switch that needs `--allow-thinking-off`. `ds41` is DeepSeek-V4.1 Flash: the prompt comes from `dschat.rs` (a
//! port of the release's `encoding.py`), calls are DSML blocks (`dsparser.rs`), thinking is a plain on/off switch and
//! `reasoning_effort` is a number from 1 to 100.
//!
//! This file holds what the two have in common at the seam: which dialect a model directory is, the parser enum the
//! handlers call, DeepSeek's thinking and effort controls, and the text a forced `tool_choice` puts in the reply.

use std::path::Path;

use serde_json::{Map, Value};

use crate::api::ApiError;
use crate::dsparser::DsParser;
use crate::parser::{Delta, FullParse, GlmParser};
use crate::pyjson;
use crate::thinking::{self, Thinking};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dialect {
    Glm,
    Ds41,
}

impl Dialect {
    pub fn name(self) -> &'static str {
        match self {
            Dialect::Glm => "glm",
            Dialect::Ds41 => "ds41",
        }
    }

    /// `--dialect`: `glm`, `ds41`, or `auto` (None: look at the model directory).
    pub fn parse_flag(s: &str) -> Result<Option<Dialect>, String> {
        match s {
            "glm" => Ok(Some(Dialect::Glm)),
            "ds41" => Ok(Some(Dialect::Ds41)),
            "auto" => Ok(None),
            other => Err(format!("--dialect {other}: want glm, ds41 or auto")),
        }
    }

    /// A model directory whose `config.json` names a DeepseekV4 architecture, or whose tokenizer has the `｜DSML｜` token
    /// (the 1017 release's `config.json` is the inference layout and names no architecture), is DeepSeek; any other is GLM.
    pub fn detect(dir: &Path, tokenizer_has_dsml: bool) -> Dialect {
        let arch_is_ds = std::fs::read(dir.join("config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
            .and_then(|c| c.get("architectures")?.as_array()?.first()?.as_str().map(|a| a.contains("DeepseekV4")))
            .unwrap_or(false);
        if arch_is_ds || tokenizer_has_dsml {
            Dialect::Ds41
        } else {
            Dialect::Glm
        }
    }
}

// ---- the parser the handlers hold ------------------------------------------------------------------------------------

pub enum AnyParser {
    Glm(GlmParser),
    Ds(DsParser),
}

impl AnyParser {
    pub fn new(d: Dialect, tools: Option<&Value>, tool_choice: &Value, include_reasoning: bool) -> AnyParser {
        match d {
            Dialect::Glm => AnyParser::Glm(GlmParser::new(tools, tool_choice, include_reasoning)),
            Dialect::Ds41 => AnyParser::Ds(DsParser::new(tools, tool_choice, include_reasoning)),
        }
    }

    /// `--undeclared-tool-calls emit`: a call to a tool the request did not declare comes back as a tool call. GLM only;
    /// the DSML parser keeps such a call as text.
    pub fn emit_undeclared(self, on: bool) -> AnyParser {
        match self {
            AnyParser::Glm(p) => AnyParser::Glm(p.emit_undeclared(on)),
            p => p,
        }
    }

    /// The reply starts as the answer: the prompt closed the think block.
    pub fn content_first(self) -> AnyParser {
        match self {
            AnyParser::Glm(p) => AnyParser::Glm(p.content_first()),
            AnyParser::Ds(p) => AnyParser::Ds(p.content_first()),
        }
    }

    pub fn delta(&mut self, text: &str, finished: bool) -> Option<Delta> {
        match self {
            AnyParser::Glm(p) => p.delta(text, finished),
            AnyParser::Ds(p) => p.delta(text, finished),
        }
    }

    pub fn parse_full(&mut self, text: &str) -> FullParse {
        match self {
            AnyParser::Glm(p) => p.parse_full(text),
            AnyParser::Ds(p) => p.parse_full(text),
        }
    }

    pub fn include_reasoning(&self) -> bool {
        match self {
            AnyParser::Glm(p) => p.include_reasoning,
            AnyParser::Ds(p) => p.include_reasoning,
        }
    }

    /// A live call block has closed: the reply is complete and the generation should end. Only DeepSeek's format has a
    /// closer to stop on; the model sometimes goes on writing after it (the 10,559-line `</function>` runaway).
    pub fn call_closed(&self) -> bool {
        match self {
            AnyParser::Glm(_) => false,
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
        return Ok((Thinking { effort: None, budget: 0, budget_explicit: false, off: true }, default_effort));
    }
    let (budget, budget_explicit) = match thinking::parse_budget(body.get("thinking_budget"))? {
        None | Some(0) => (defaults.budget, false),
        Some(n) if n < 0 => (0, false),
        Some(n) => (n as u64, true),
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
/// BOS text, so it is encoded without special tokens added.
pub fn ds_prompt(
    messages: Option<&Value>,
    tools: Option<&Value>,
    thinking: bool,
    effort: u32,
    add_generation_prompt: bool,
    continue_final: bool,
) -> Result<String, ApiError> {
    use crate::dschat::{self, Opts};
    let (m, t) = lists(messages, tools)?;
    let mut text = dschat::render(m, t, &Opts { thinking, effort }).map_err(render_err)?;
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
    Ok(text)
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
        assert_eq!((t.budget, t.budget_explicit, e), (0, false, 75));
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
        let d = thinking::Defaults { effort: None, budget: 900, allow_off: false };
        let r = |v: Value| resolve_ds(&body(v), d, 50).unwrap();
        let (t, e) = r(json!({}));
        assert_eq!((t.budget, t.budget_explicit, e), (900, false, 50));
        let (t, _) = r(json!({"thinking_budget": 100}));
        assert_eq!((t.budget, t.budget_explicit), (100, true));
        let (t, _) = r(json!({"thinking_budget": -1}));
        assert_eq!((t.budget, t.budget_explicit), (0, false));
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
        let on = ds_prompt(Some(&msgs), None, true, 75, true, false).unwrap();
        let ids = t.encode(&on, false).unwrap();
        assert_eq!(ids[0], 0);
        assert_eq!(&ids[ids.len() - 2..], &[128804, 128821]);
        assert!(think_open(&ids, 128821, 128822));
        // thinking off: the header and a closed block
        let off = ds_prompt(Some(&msgs), None, false, 75, true, false).unwrap();
        let ids = t.encode(&off, false).unwrap();
        assert_eq!(&ids[ids.len() - 2..], &[128804, 128822]);
        assert!(!think_open(&ids, 128821, 128822));
        // add_generation_prompt false drops the header; continue_final_message keeps an assistant turn open
        let bare = ds_prompt(Some(&msgs), None, true, 75, false, false).unwrap();
        assert!(bare.ends_with("hi") && !bare.contains("<｜Assistant｜>"));
        let cont = json!([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hel"}]);
        let c = ds_prompt(Some(&cont), None, false, 75, false, true).unwrap();
        assert!(c.ends_with("</think>Hel"), "{c:?}");
        assert!(ds_prompt(Some(&msgs), None, false, 75, false, true).is_err());
        // the system span: the leading system message and the tools, the same tokens at the front of the whole prompt
        let tools = json!([{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object", "properties": {}}}}]);
        let full = ds_prompt(Some(&msgs), Some(&tools), true, 75, true, false).unwrap();
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
    }

    #[test]
    fn the_flag_parses() {
        assert_eq!(Dialect::parse_flag("ds41"), Ok(Some(Dialect::Ds41)));
        assert_eq!(Dialect::parse_flag("glm"), Ok(Some(Dialect::Glm)));
        assert_eq!(Dialect::parse_flag("auto"), Ok(None));
        assert!(Dialect::parse_flag("qwen").is_err());
    }
}
