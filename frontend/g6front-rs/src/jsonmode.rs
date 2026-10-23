//! response_format enforcement: pull the first JSON value out of the reply and, for json_schema, validate it.
//!
//! The engine has no constrained decoding yet, so the front end enforces after generation: `json_object` replaces
//! the content with the first balanced JSON object or array found in it (strings and escapes honored; a candidate
//! that does not parse is skipped and the scan continues); `json_schema` validates the reply's token ids against
//! the compiled schema (xgrammar). Nothing valid, or a schema violation: the content is returned unchanged with a
//! `warnings` entry — the client sees the miss, the request never fails.
//!
//! `warnings` is not an OpenAI response field: a client that asked for JSON and gets prose plus a warning is still
//! broken, just visibly. JSON mode has no incremental output (streamed content is withheld until the end so it can
//! be extracted); both facts belong in the caller's expectations.
//!
//! A reply that made tool calls skips enforcement: the structured answer is the tool call, not the content.

use serde_json::Value;
use xgrammar::{CompiledGrammar, GrammarMatcher};

use crate::api::ApiError;

const JSON_OBJECT_HINT: &str = "Respond with only a JSON object.";
const NO_JSON: &str = "response_format {kind}: no JSON object or array found in the output";
const SCHEMA_BAD: &str = "response_format json_schema: output does not match the schema: {err}";
const MAX_OPENERS: usize = 64; // candidate starts tried before giving up; keeps pathological prose linear

/// The `(kind, schema)` a request's `response_format` resolves to; kind "text" enforces nothing. A `json_schema`
/// request carries the schema compiled against the model's vocabulary (built once per front end).
pub struct Rf {
    pub kind: String,
    pub schema: Option<Value>,
    pub grammar: Option<CompiledGrammar>,
}

pub fn parse_response_format(v: Option<&Value>, compiler: Option<&xgrammar::GrammarCompiler>) -> Result<Rf, ApiError> {
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
                return Err(bad(
                    "response_format json_schema needs {\"name\": str, \"schema\": object}".into(),
                ));
            }
            let schema = js.and_then(|js| js.get("schema")).cloned().unwrap_or(Value::Null);
            // a schema the compiler cannot take is a 400 before the engine sees the request
            let grammar = match compiler {
                Some(c) => match c.compile_json_schema(&schema.to_string(), false, None, None, false, None) {
                    Ok(g) => Some(g),
                    Err(e) => {
                        return Err(bad(format!("response_format json_schema: the schema is not compilable: {e}")))
                    }
                },
                None => None,
            };
            Ok(Rf { kind: "json_schema".into(), schema: Some(schema), grammar })
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



/// Whether the reply's token ids are accepted by the compiled schema grammar. A fresh matcher per reply: the
/// grammar is compiled once per request, the matcher is per-sequence state.
fn schema_ok(grammar: &CompiledGrammar, ids: &[u32]) -> bool {
    if ids.is_empty() {
        return false;
    }
    let mut m = GrammarMatcher::from_compiled_grammar(grammar, false);
    ids.iter().all(|&id| m.accept_token(id as i32))
}

pub fn enforce(content: Option<&str>, tok: Option<&crate::tok::Tok>, rf: &Rf) -> (Option<String>, Vec<String>) {
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
            (Some(found), Some(grammar)) => {
                // replay the extracted span's own tokenization through the grammar: the ids of the whole
                // reply include the prose around the JSON, which the grammar would rightly reject
                let ids: Vec<u32> = tok.and_then(|t| t.encode(&found, false).ok()).unwrap_or_default();
                if schema_ok(grammar, &ids) {
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

    /// Schema validation through the real grammar: G6_TOKENIZER_DIR=<dir with tokenizer.json> cargo test
    #[test]
    #[ignore = "needs G6_TOKENIZER_DIR (a real tokenizer); runs in CI with the model dir set"]
    fn schema_grammar_accepts_and_rejects() {
        use xgrammar::{GrammarCompiler, TokenizerInfo, VocabType};
        let dir = std::env::var("G6_TOKENIZER_DIR").expect("G6_TOKENIZER_DIR");
        let tk = tokenizers::Tokenizer::from_file(format!("{dir}/tokenizer.json")).expect("tokenizer");
        let n = tk.get_vocab_size(true);
        let vocab: Vec<String> =
            (0..n as u32).map(|i| tk.decode(&[i], false).unwrap_or_default()).collect();
        let meta = format!(
            "{{\"vocab_type\": {}, \"vocab_size\": {n}}}",
            VocabType::ByteFallback as i64
        );
        let info = TokenizerInfo::from_vocab_and_metadata(&vocab, &meta).expect("tokenizer info");
        let compiler = GrammarCompiler::with_defaults(info);
        let schema = json!({"type": "object", "required": ["a"], "properties": {"a": {"type": "integer"}}});
        let compiled = compiler
            .compile_json_schema(&schema.to_string(), false, None, None, false, None)
            .expect("compile");
        let ok_ids: Vec<u32> = tk.encode(r#"{"a": 3}"#, false).expect("enc").get_ids().to_vec();
        assert!(schema_ok(&compiled, &ok_ids));
        let bad_ids: Vec<u32> = tk.encode(r#"{"a": "x"}"#, false).expect("enc").get_ids().to_vec();
        assert!(!schema_ok(&compiled, &bad_ids));
    }

    #[test]
    fn enforcement_policies() {
        let rf = Rf { kind: "json_object".into(), schema: None, grammar: None };
        let (c, w) = enforce(Some("Here: {\"a\": 1} done"), None, &rf);
        assert_eq!(c.as_deref(), Some("{\"a\": 1}"));
        assert!(w.is_empty());
        let (c, w) = enforce(Some("no json"), None, &rf);
        assert_eq!(c.as_deref(), Some("no json"));
        assert_eq!(w.len(), 1);
        let schema = json!({"type": "object", "required": ["a"], "properties": {"a": {"type": "integer"}}});
        let rf = Rf { kind: "json_schema".into(), schema: Some(schema), grammar: None };
        // no compiled grammar in this unit test: extraction only, no validation claim
        let (c, w) = enforce(Some("{\"a\": \"x\"}"), None, &rf);
        assert_eq!(c.as_deref(), Some("{\"a\": \"x\"}"));
        assert!(w.is_empty());
        let (c, w) = enforce(Some("{\"a\": 2}"), None, &rf);
        assert_eq!(c.as_deref(), Some("{\"a\": 2}"));
        assert!(w.is_empty());
    }
}
