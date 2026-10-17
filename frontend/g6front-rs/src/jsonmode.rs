//! response_format enforcement: pull the first JSON value out of the reply and, for json_schema, validate it.
//!
//! The engine has no constrained decoding yet, so the front end enforces after generation: `json_object` replaces
//! the content with the first balanced JSON object or array found in it (strings and escapes honored; a candidate
//! that does not parse is skipped and the scan continues); `json_schema` also checks that value with the schema's
//! compiled grammar (XGrammar, `grammar.rs`). Nothing valid, or a schema violation: the content is returned unchanged with a
//! `warnings` entry — the client sees the miss, the request never fails.
//!
//! `warnings` is not an OpenAI response field: a client that asked for JSON and gets prose plus a warning is still
//! broken, just visibly. JSON mode has no incremental output (streamed content is withheld until the end so it can
//! be extracted); both facts belong in the caller's expectations.
//!
//! A reply that made tool calls skips enforcement: the structured answer is the tool call, not the content.

use serde_json::Value;
use xgrammar::CompiledGrammar;

use crate::api::ApiError;
use crate::grammar::{self, Grammars};

const JSON_OBJECT_HINT: &str = "Respond with only a JSON object.";
const NO_JSON: &str = "response_format {kind}: no JSON object or array found in the output";
const SCHEMA_BAD: &str = "response_format json_schema: output does not match the schema: {err}";
const MAX_OPENERS: usize = 64; // candidate starts tried before giving up; keeps pathological prose linear

/// The `(kind, schema)` a request's `response_format` resolves to; kind "text" enforces nothing. A `json_schema`
/// request carries the schema compiled against the model's vocabulary.
pub struct Rf {
    pub kind: String,
    pub schema: Option<Value>,
    pub grammar: Option<CompiledGrammar>,
}

pub fn parse_response_format(v: Option<&Value>, grammars: &Grammars) -> Result<Rf, ApiError> {
    let bad = |m: String| ApiError::bad(m, Some("response_format"));
    let Some(v) = v else {
        return Ok(Rf { kind: "text".into(), schema: None, grammar: None });
    };
    let Some(o) = v.as_object() else {
        return Err(bad("response_format must be an object".into()));
    };
    match o.get("type").and_then(Value::as_str) {
        Some("text") => Ok(Rf { kind: "text".into(), schema: None, grammar: None }),
        Some("json_object") => Ok(Rf { kind: "json_object".into(), schema: None, grammar: None }),
        Some("json_schema") => {
            let js = o.get("json_schema").and_then(Value::as_object);
            let ok = js.is_some_and(|js| js.get("name").is_some_and(Value::is_string))
                && js.is_some_and(|js| js.get("schema").is_some_and(Value::is_object));
            if !ok {
                return Err(bad("response_format json_schema needs {\"name\": str, \"schema\": object}".into()));
            }
            let schema = js.and_then(|js| js.get("schema")).cloned().unwrap_or(Value::Null);
            // a schema the compiler cannot take is a 400 before the engine sees the request
            let grammar = grammars
                .json_schema(&schema.to_string())
                .map_err(|e| bad(format!("response_format json_schema: the schema does not compile: {e}")))?;
            Ok(Rf { kind: "json_schema".into(), schema: Some(schema), grammar: Some(grammar) })
        }
        Some(t) => Err(bad(format!(
            "response_format \"{t}\" is not supported by the gr8 front end yet (only \"text\", \"json_object\", \
             \"json_schema\")"
        ))),
        None => Err(bad("response_format needs a string type".into())),
    }
}

/// The instruction appended after the last user or tool message when response_format asks for JSON.
pub fn response_format_hint(rf: &Rf) -> Option<String> {
    match rf.kind.as_str() {
        "json_object" => Some(JSON_OBJECT_HINT.to_string()),
        "json_schema" => Some(format!(
            "{JSON_OBJECT_HINT} It must validate against this JSON schema: {}",
            serde_json::to_string(rf.schema.as_ref().unwrap_or(&Value::Null)).unwrap_or_default()
        )),
        _ => None,
    }
}

/// The first balanced JSON object or array in `text` (a ```json fence counts), or None.
pub fn extract_json(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut starts = 0;
    for (start, &c) in bytes.iter().enumerate() {
        if c != b'{' && c != b'[' {
            continue;
        }
        starts += 1;
        if starts > MAX_OPENERS {
            return None;
        }
        let mut stack: Vec<u8> = Vec::new();
        let mut in_str = false;
        let mut esc = false;
        let mut closed_at = None;
        for (i, &c) in bytes[start..].iter().enumerate() {
            let c = c as char;
            if in_str {
                if esc {
                    esc = false;
                } else if c == '\\' {
                    esc = true;
                } else if c == '"' {
                    in_str = false;
                }
                continue;
            }
            if c == '"' {
                in_str = true;
            } else if c == '{' || c == '[' {
                stack.push(if c == '{' { b'}' } else { b']' });
            } else if c == '}' || c == ']' {
                if stack.last() != Some(&(c as u8)) {
                    break;
                }
                stack.pop();
                if stack.is_empty() {
                    closed_at = Some(start + i + 1);
                    break;
                }
            }
        }
        if let Some(end) = closed_at {
            let candidate = &text[start..end];
            if serde_json::from_str::<Value>(candidate).is_ok() {
                return Some(candidate.to_string());
            }
            // balanced but not JSON: try the next opener
        }
    }
    None
}

pub fn enforce(content: Option<&str>, rf: &Rf) -> (Option<String>, Vec<String>) {
    let mut warnings: Vec<String> = Vec::new();
    match rf.kind.as_str() {
        "json_object" => match extract_json(content.unwrap_or("")) {
            Some(found) => (Some(found), warnings),
            None => {
                warnings.push(NO_JSON.replace("{kind}", "json_object"));
                (content.map(str::to_string), warnings)
            }
        },
        "json_schema" => match (extract_json(content.unwrap_or("")), rf.grammar.as_ref()) {
            (Some(found), Some(g)) => {
                // the extracted span alone: the prose around it is not the schema's to accept
                if grammar::matches(g, &found) {
                    (Some(found), warnings)
                } else {
                    warnings.push(SCHEMA_BAD.replace("{err}", "the reply does not satisfy the schema"));
                    (content.map(str::to_string), warnings)
                }
            }
            (Some(found), None) => (Some(found), warnings), // no grammar available: extraction only
            (None, _) => {
                warnings.push(NO_JSON.replace("{kind}", "json_schema"));
                (content.map(str::to_string), warnings)
            }
        },
        _ => (content.map(str::to_string), warnings),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extracts_first_balanced_object() {
        assert_eq!(extract_json("noise {\"a\": 1} tail").as_deref(), Some("{\"a\": 1}"));
        assert_eq!(extract_json("```json\n{\"a\": [1, 2]}\n```").as_deref(), Some("{\"a\": [1, 2]}"));
        assert_eq!(extract_json("quote \"bracket { in string\" {\"ok\": 1}").as_deref(), Some("{\"ok\": 1}"));
        assert_eq!(extract_json("brace { not json [x] } then {\"a\": 2}"), Some("{\"a\": 2}".to_string()));
        assert_eq!(extract_json("no json here"), None);
        assert_eq!(extract_json("truncated {\"a\": 1"), None);
        assert_eq!(extract_json("nested {\"a\": {\"b\": 2}}"), Some("{\"a\": {\"b\": 2}}".to_string()));
    }

    /// The schema check through the real grammar: G6_TOKENIZER_DIR=<the model's tokenizer dir> cargo test -- --ignored
    #[test]
    #[ignore = "needs G6_TOKENIZER_DIR, a directory holding the model's tokenizer.json"]
    fn schema_grammar_accepts_and_rejects() {
        let dir = std::env::var("G6_TOKENIZER_DIR").expect("G6_TOKENIZER_DIR");
        let g = Grammars::load(std::path::Path::new(&dir), 154880).expect("grammars");
        let schema = json!({"type": "object", "required": ["a"], "properties": {"a": {"type": "integer"}}});
        let rf = parse_response_format(
            Some(&json!({"type": "json_schema", "json_schema": {"name": "t", "schema": schema}})),
            &g,
        )
        .unwrap_or_else(|e| panic!("{}", e.message));
        let (c, w) = enforce(Some("Here it is: {\"a\": 3} done"), &rf);
        assert_eq!((c.as_deref(), w.len()), (Some("{\"a\": 3}"), 0));
        for bad in ["{\"a\": \"x\"}", "{\"b\": 1}", "[1]"] {
            let (c, w) = enforce(Some(bad), &rf);
            assert_eq!((c.as_deref(), w.len()), (Some(bad), 1), "{bad}");
        }
        // a schema the compiler cannot take is a 400
        let e = parse_response_format(
            Some(&json!({"type": "json_schema", "json_schema": {"name": "t",
            "schema": {"$ref": "#/nowhere"}}})),
            &g,
        )
        .err()
        .expect("a 400");
        assert!(e.message.contains("does not compile"), "{}", e.message);
    }

    #[test]
    fn enforcement_policies() {
        let rf = Rf { kind: "json_object".into(), schema: None, grammar: None };
        let (c, w) = enforce(Some("Here: {\"a\": 1} done"), &rf);
        assert_eq!(c.as_deref(), Some("{\"a\": 1}"));
        assert!(w.is_empty());
        let (c, w) = enforce(Some("no json"), &rf);
        assert_eq!(c.as_deref(), Some("no json"));
        assert_eq!(w.len(), 1);
        let schema = json!({"type": "object", "required": ["a"], "properties": {"a": {"type": "integer"}}});
        let rf = Rf { kind: "json_schema".into(), schema: Some(schema), grammar: None };
        // no compiled grammar in this unit test: extraction only, no validation claim
        let (c, w) = enforce(Some("{\"a\": \"x\"}"), &rf);
        assert_eq!(c.as_deref(), Some("{\"a\": \"x\"}"));
        assert!(w.is_empty());
        let (c, w) = enforce(Some("{\"a\": 2}"), &rf);
        assert_eq!(c.as_deref(), Some("{\"a\": 2}"));
        assert!(w.is_empty());
    }
}
