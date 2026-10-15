//! `tool_choice` (OpenAI/vLLM): which tools a request lets the model call.
//!
//! `none` and `auto` need nothing from the front end -- `parser.rs` already reads a call only when the request carries
//! tools and the choice is not `none`. `required`, and an object naming one function, are the case where the reply must
//! be a call rather than text. The front end forces that with the engine's insert rule (PROTOCOL.md `insert`, the same
//! mechanism the thinking budget uses for `</think>`): the engine puts the tool-call opener in place of the reply's first
//! token, so the model can only go on inside a call.
//!
//! The opener goes where the reply's first token would be, and GLM's chat template leaves `<think>` open at the end of
//! the prompt, so that position is inside the reasoning span. A call written there is reasoning text, and the parser
//! does not read a call in reasoning (`parser.rs`), which left `required` returning prose for some prompts. So the rule
//! closes thinking first, the way the budget's rule does: the inserted tokens are `think_close` then the opener, and the
//! reply's first content token is the call.
//!
//! It is a strong nudge, not a grammar. gr8's engine samples, so the front end cannot mask logits, and a reply that
//! opens a call to a name the request did not offer is still possible -- the parser drops such a name, which leaves text
//! and `finish_reason: stop` -- as are arguments the tool's schema rejects (`strict` is not enforced; the parser coerces
//! values by the schema as before). The opener is also why a forced reply carries no reasoning: its first token is the
//! call, not `<think>`.

use serde_json::Value;

use crate::api::ApiError;
use crate::tok::Tok;

/// The opener of GLM's tool-call format (`parser.rs`, `TOOL_START`): what a forced reply starts with.
const TOOL_START: &str = "<tool_call>";
/// Tokens one insert rule may hold (engine/runtime/insert_rule.h, `INSERT_MAX`).
const INSERT_MAX: usize = 64;

/// What the request asked for. `Named` carries the function the reply must open with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolChoice {
    None,
    Auto,
    Required,
    Named(String),
}

impl ToolChoice {
    /// `none`, `auto`, `required`, or `{"type": "function", "function": {"name": ...}}`. Absent is `auto` with tools and
    /// `none` without, as before. A forced choice needs the function to exist: `required` names none, so it needs a
    /// non-empty `tools`, and a named one must be in the list -- the parser drops a name the request did not offer, so
    /// the forced call would silently come back as text.
    pub fn parse(tools: Option<&Value>, raw: Option<&Value>) -> Result<ToolChoice, ApiError> {
        let names = tool_names(tools);
        let choice = match raw {
            None | Some(Value::Null) => {
                return Ok(if names.is_empty() { ToolChoice::None } else { ToolChoice::Auto });
            }
            Some(Value::String(s)) => match s.as_str() {
                "none" => ToolChoice::None,
                "auto" => ToolChoice::Auto,
                "required" => {
                    if names.is_empty() {
                        return Err(ApiError::bad(
                            "tool_choice 'required' needs a non-empty 'tools'",
                            Some("tool_choice"),
                        ));
                    }
                    ToolChoice::Required
                }
                other => {
                    return Err(ApiError::bad(
                        format!("tool_choice '{other}' is not one of 'none', 'auto', 'required', or a function"),
                        Some("tool_choice"),
                    ));
                }
            },
            Some(Value::Object(o)) => {
                if o.get("type").and_then(Value::as_str) != Some("function") {
                    return Err(ApiError::bad(
                        "a tool_choice object needs {\"type\": \"function\", \"function\": {\"name\": ...}}",
                        Some("tool_choice"),
                    ));
                }
                let name = o
                    .get("function")
                    .and_then(Value::as_object)
                    .and_then(|f| f.get("name"))
                    .and_then(Value::as_str);
                let Some(name) = name.filter(|n| !n.is_empty()) else {
                    return Err(ApiError::bad("a forced tool_choice needs function.name", Some("tool_choice")));
                };
                if !names.contains(&name) {
                    return Err(ApiError::bad(
                        format!("tool_choice names '{name}', which is not in 'tools'"),
                        Some("tool_choice"),
                    ));
                }
                ToolChoice::Named(name.to_string())
            }
            Some(_) => {
                return Err(ApiError::bad(
                    "tool_choice must be a string or {\"type\": \"function\", \"function\": {\"name\": ...}}",
                    Some("tool_choice"),
                ));
            }
        };
        Ok(choice)
    }

    /// Whether the reply must be a call (the case the insert rule forces).
    pub fn is_forced(&self) -> bool {
        matches!(self, ToolChoice::Required | ToolChoice::Named(_))
    }

    /// The text the engine puts in place of the reply's first token, or `None` when the choice forces nothing. A named
    /// choice carries the name too, so the model cannot open a call to another function.
    pub fn forced_text(&self) -> Option<String> {
        match self {
            ToolChoice::Required => Some(TOOL_START.to_string()),
            ToolChoice::Named(name) => Some(format!("{TOOL_START}{name}")),
            ToolChoice::None | ToolChoice::Auto => None,
        }
    }

    /// `forced_text` as token ids, behind `think_close`: what goes in the request's insert rule
    /// (engine/runtime/insert_rule.h). `think_close` is the template's `</think>` when the prompt leaves `<think>` open
    /// (`gen::think_open`) and empty otherwise, so the opener always starts the reply's content rather than its
    /// reasoning. Empty when the choice forces nothing.
    pub fn forced_ids(&self, tok: &Tok, think_close: &[u32]) -> Result<Vec<u32>, ApiError> {
        let Some(text) = self.forced_text() else { return Ok(Vec::new()) };
        let opener = tok.encode(&text, false).map_err(|e| ApiError::new(500, &e, None))?;
        if opener.is_empty() {
            return Err(ApiError::bad(format!("'{text}' does not tokenize"), Some("tool_choice")));
        }
        let mut ids = think_close.to_vec();
        ids.extend_from_slice(&opener);
        if ids.len() > INSERT_MAX {
            return Err(ApiError::bad(
                format!("the forced tool_choice is {} tokens, more than the engine's {INSERT_MAX}", ids.len()),
                Some("tool_choice"),
            ));
        }
        Ok(ids)
    }

    /// The value `GlmParser::new` takes: it reads it only for `!= "none"` (`parse_tools`), and a forced choice always
    /// parses (it needs tools, since the name must be in the list).
    pub fn as_value(&self) -> Value {
        match self {
            ToolChoice::None => Value::from("none"),
            ToolChoice::Auto => Value::from("auto"),
            ToolChoice::Required => Value::from("required"),
            ToolChoice::Named(name) => {
                serde_json::json!({"type": "function", "function": {"name": name}})
            }
        }
    }
}

/// The names in a normalized `tools` list (`template::normalize_tools`), in order.
fn tool_names(tools: Option<&Value>) -> Vec<&str> {
    let Some(Value::Array(ts)) = tools else { return Vec::new() };
    ts.iter()
        .filter_map(|t| t.get("function").and_then(Value::as_object).and_then(|f| f.get("name")))
        .filter_map(Value::as_str)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tools() -> Value {
        json!([
            {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "get_time"}}
        ])
    }

    #[test]
    fn absent_is_auto_with_tools_and_none_without() {
        assert_eq!(ToolChoice::parse(Some(&tools()), None).unwrap(), ToolChoice::Auto);
        assert_eq!(ToolChoice::parse(None, None).unwrap(), ToolChoice::None);
        assert_eq!(ToolChoice::parse(None, Some(&Value::Null)).unwrap(), ToolChoice::None);
        assert_eq!(ToolChoice::parse(Some(&json!([])), Some(&Value::Null)).unwrap(), ToolChoice::None);
    }

    #[test]
    fn none_and_auto_are_unchanged() {
        for (raw, want) in [("none", ToolChoice::None), ("auto", ToolChoice::Auto)] {
            assert_eq!(ToolChoice::parse(Some(&tools()), Some(&json!(raw))).unwrap(), want);
            // both work without tools too (auto there parses nothing, as before)
            assert_eq!(ToolChoice::parse(None, Some(&json!(raw))).unwrap(), want);
        }
        assert!(!ToolChoice::Auto.is_forced());
        assert!(!ToolChoice::None.is_forced());
        assert_eq!(ToolChoice::Auto.forced_text(), None);
    }

    #[test]
    fn required_needs_tools() {
        assert_eq!(ToolChoice::parse(Some(&tools()), Some(&json!("required"))).unwrap(), ToolChoice::Required);
        assert!(ToolChoice::Required.is_forced());
        assert_eq!(ToolChoice::Required.forced_text().unwrap(), "<tool_call>");
        for none in [None, Some(&json!([]))] {
            let e = ToolChoice::parse(none, Some(&json!("required"))).unwrap_err();
            assert_eq!((e.status, e.param.as_deref()), (400, Some("tool_choice")));
        }
    }

    #[test]
    fn a_named_choice_must_be_one_of_the_tools() {
        let raw = json!({"type": "function", "function": {"name": "get_weather"}});
        let c = ToolChoice::parse(Some(&tools()), Some(&raw)).unwrap();
        assert_eq!(c, ToolChoice::Named("get_weather".into()));
        assert!(c.is_forced());
        assert_eq!(c.forced_text().unwrap(), "<tool_call>get_weather");
        // the parser only needs a value that is not "none"
        assert_ne!(c.as_value(), Value::from("none"));
        let unknown = json!({"type": "function", "function": {"name": "get_rain"}});
        assert_eq!(ToolChoice::parse(Some(&tools()), Some(&unknown)).unwrap_err().status, 400);
        // no tools at all
        assert_eq!(ToolChoice::parse(None, Some(&raw)).unwrap_err().status, 400);
    }

    #[test]
    fn a_malformed_choice_is_refused() {
        for raw in [
            json!("require"),
            json!(3),
            json!(["required"]),
            json!({"type": "tool", "function": {"name": "get_weather"}}),
            json!({"type": "function"}),
            json!({"type": "function", "function": {"name": ""}}),
            json!({"type": "function", "function": {"name": 7}}),
        ] {
            let e = ToolChoice::parse(Some(&tools()), Some(&raw)).unwrap_err();
            assert_eq!((e.status, e.param.as_deref()), (400, Some("tool_choice")), "{raw}");
        }
    }
}
