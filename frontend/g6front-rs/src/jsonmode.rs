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
    let Some(v) = v.filter(|v| !v.is_null()) else {
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
            root_type_ok(&schema).map_err(bad)?;
            let strict = js.and_then(|js| js.get("strict")).is_some_and(|v| v.as_bool() == Some(true));
            // a schema the compiler cannot take is a 400 before the engine sees the request
            let grammar = grammars
                .json_schema(&schema.to_string(), strict)
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

/// The reply's value is an object or an array (`extract_json`), so a schema whose root type allows neither could
/// never pass.
fn root_type_ok(schema: &Value) -> Result<(), String> {
    let root = |t: &str| t == "object" || t == "array";
    match schema.get("type") {
        None => Ok(()),
        Some(Value::String(t)) if root(t) => Ok(()),
        Some(Value::Array(ts)) if ts.iter().filter_map(Value::as_str).any(root) => Ok(()),
        Some(t) => Err(format!("response_format json_schema: the root type must be object or array, not {t}")),
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

/// The reply's JSON object or array: inside a ```json fence when the reply has one, else the first top-level value.
/// A value cut off before its close (the text ends inside it) ends the search with None: a complete value nested in
/// it is a fragment, not the answer. A balanced span that does not parse is skipped whole, openers inside it included.
/// An unclosed opener that is not the start of a cut-off value (a brace in prose) is skipped alone.
pub fn extract_json(text: &str) -> Option<String> {
    if let Some(i) = text.find("```json") {
        let body = &text[i + 7..];
        if let Some(found) = body.find("```").and_then(|end| first_value(&body[..end])) {
            return Some(found);
        }
    }
    first_value(text)
}

fn first_value(text: &str) -> Option<String> {
    let b = text.as_bytes();
    let (mut i, mut tried) = (0, 0);
    while i < b.len() {
        if b[i] != b'{' && b[i] != b'[' {
            i += 1;
            continue;
        }
        tried += 1;
        if tried > MAX_OPENERS {
            return None;
        }
        let Some(end) = span_end(b, i) else {
            match serde_json::from_str::<Value>(&text[i..]) {
                Err(e) if e.classify() == serde_json::error::Category::Eof => return None,
                _ => {
                    i += 1;
                    continue;
                }
            }
        };
        if serde_json::from_str::<Value>(&text[i..end]).is_ok() {
            return Some(text[i..end].to_string());
        }
        i = end;
    }
    None
}

/// The end of the bracketed span that opens at `start`: past its matching close, or past the first close that does
/// not match. None when the text ends first. Strings and their escapes are honored.
fn span_end(b: &[u8], start: usize) -> Option<usize> {
    let (mut stack, mut in_str, mut esc) = (Vec::new(), false, false);
    for (i, &c) in b.iter().enumerate().skip(start) {
        if in_str {
            match (esc, c) {
                (true, _) => esc = false,
                (false, b'\\') => esc = true,
                (false, b'"') => in_str = false,
                _ => {}
            }
            continue;
        }
        match c {
            b'"' => in_str = true,
            b'{' => stack.push(b'}'),
            b'[' => stack.push(b']'),
            b'}' | b']' => {
                if stack.pop() != Some(c) || stack.is_empty() {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// What the grammar leaves unchecked with keys in any order: no object repeats a key, and each object has the keys its
/// schema's `required` lists, through `properties`, `items` and `additionalProperties` schemas. Err: the first miss.
fn check_value(schema: &Value, text: &str) -> Result<(), String> {
    let mut de = serde_json::Deserializer::from_str(text);
    serde::de::DeserializeSeed::deserialize(NoDup, &mut de).map_err(|e| e.to_string())?;
    let v: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
    required(schema, &v, "$")
}

fn required(schema: &Value, v: &Value, at: &str) -> Result<(), String> {
    match v {
        Value::Object(o) => {
            for k in schema.get("required").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str) {
                if !o.contains_key(k) {
                    return Err(format!("{at} has no required key {k:?}"));
                }
            }
            let props = schema.get("properties").and_then(Value::as_object);
            for (k, x) in o {
                let sub = props
                    .and_then(|p| p.get(k))
                    .or_else(|| schema.get("additionalProperties").filter(|a| a.is_object()));
                if let Some(sub) = sub {
                    required(sub, x, &format!("{at}.{k}"))?;
                }
            }
        }
        Value::Array(a) => {
            if let Some(items) = schema.get("items").filter(|i| i.is_object()) {
                for (n, x) in a.iter().enumerate() {
                    required(items, x, &format!("{at}[{n}]"))?;
                }
            }
        }
        _ => {}
    }
    Ok(())
}

/// A JSON value walked for its shape only: Err at an object that repeats a key, which serde_json's Value keeps once.
struct NoDup;

impl<'de> serde::de::DeserializeSeed<'de> for NoDup {
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, d: D) -> Result<(), D::Error> {
        d.deserialize_any(self)
    }
}

impl<'de> serde::de::Visitor<'de> for NoDup {
    type Value = ();
    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("a JSON value")
    }
    fn visit_map<A: serde::de::MapAccess<'de>>(self, mut m: A) -> Result<(), A::Error> {
        let mut seen = std::collections::HashSet::new();
        while let Some(k) = m.next_key::<String>()? {
            if !seen.insert(k.clone()) {
                return Err(serde::de::Error::custom(format!("duplicate key {k:?}")));
            }
            m.next_value_seed(NoDup)?;
        }
        Ok(())
    }
    fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut s: A) -> Result<(), A::Error> {
        while s.next_element_seed(NoDup)?.is_some() {}
        Ok(())
    }
    fn visit_bool<E>(self, _: bool) -> Result<(), E> {
        Ok(())
    }
    fn visit_i64<E>(self, _: i64) -> Result<(), E> {
        Ok(())
    }
    fn visit_u64<E>(self, _: u64) -> Result<(), E> {
        Ok(())
    }
    fn visit_f64<E>(self, _: f64) -> Result<(), E> {
        Ok(())
    }
    fn visit_str<E>(self, _: &str) -> Result<(), E> {
        Ok(())
    }
    fn visit_unit<E>(self) -> Result<(), E> {
        Ok(())
    }
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
                if let Err(why) = check_value(rf.schema.as_ref().unwrap_or(&Value::Null), &found) {
                    warnings.push(SCHEMA_BAD.replace("{err}", &why));
                    (content.map(str::to_string), warnings)
                } else if grammar::matches(g, &found) {
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
        // a cut-off or invalid outer value never yields the complete value nested in it
        assert_eq!(extract_json("{\"city\":\"Paris\",\"days\":{\"n\":3}"), None);
        assert_eq!(extract_json("{\"a\":{\"b\":1},}"), None);
        // a ```json fence wins over a value before it
        assert_eq!(extract_json("e.g. {\"x\": 0}\n```json\n{\"a\": 1}\n```").as_deref(), Some("{\"a\": 1}"));
    }

    #[test]
    fn check_value_finds_what_any_order_leaves_out() {
        let schema = json!({"type": "object", "required": ["city", "days"], "properties": {
            "city": {"type": "string"},
            "days": {"type": "object", "required": ["n"], "properties": {"n": {"type": "integer"}}},
            "stops": {"type": "array", "items": {"type": "object", "required": ["at"]}}}});
        assert_eq!(check_value(&schema, r#"{"days": {"n": 3}, "city": "Paris"}"#), Ok(()));
        let e = check_value(&schema, r#"{"city": "Paris"}"#).unwrap_err();
        assert!(e.contains("no required key \"days\""), "{e}");
        let e = check_value(&schema, r#"{"city": "Paris", "days": {}}"#).unwrap_err();
        assert!(e.contains("$.days has no required key \"n\""), "{e}");
        let e = check_value(&schema, r#"{"city": "a", "days": {"n": 1}, "stops": [{"at": 1}, {}]}"#).unwrap_err();
        assert!(e.contains("$.stops[1]"), "{e}");
        let e = check_value(&schema, r#"{"city": "a", "city": "b", "days": {"n": 1}}"#).unwrap_err();
        assert!(e.contains("duplicate key \"city\""), "{e}");
    }

    #[test]
    fn null_is_text_and_the_root_must_be_an_object_or_array() {
        assert!(matches!(root_type_ok(&json!({"type": "object"})), Ok(())));
        assert!(matches!(root_type_ok(&json!({"type": ["array", "null"]})), Ok(())));
        assert!(root_type_ok(&json!({"type": "string"})).unwrap_err().contains("not \"string\""));
        assert!(matches!(root_type_ok(&json!({"anyOf": []})), Ok(())));
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
        // keys in any order; an undeclared key passes unless the request says strict
        let two = json!({"type": "object", "required": ["city", "days"],
                         "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}});
        let rf_for = |strict: bool| {
            parse_response_format(
                Some(&json!({"type": "json_schema", "json_schema": {"name": "t", "strict": strict,
                "schema": two.clone()}})),
                &g,
            )
            .unwrap_or_else(|e| panic!("{}", e.message))
        };
        let (loose, strict) = (rf_for(false), rf_for(true));
        assert!(enforce(Some(r#"{"days": 3, "city": "Paris"}"#), &loose).1.is_empty());
        assert!(enforce(Some(r#"{"days": 3, "city": "Paris"}"#), &strict).1.is_empty());
        assert!(enforce(Some(r#"{"city": "Paris", "days": 3, "x": 1}"#), &loose).1.is_empty());
        assert_eq!(enforce(Some(r#"{"city": "Paris", "days": 3, "x": 1}"#), &strict).1.len(), 1);
        assert_eq!(enforce(Some(r#"{"days": 3}"#), &loose).1.len(), 1);
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
