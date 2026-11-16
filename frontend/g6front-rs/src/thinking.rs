//! Thinking controls: `reasoning_effort`, `thinking_budget` and the `enable_thinking` switch, read the way
//! TensorFold 0.6.1 reads them (`server/request_options.py`), and the `response_format` guard.
//!
//! GLM's chat template has no way to turn thinking off: it always opens `<think>` and names the efforts low, high and
//! max, rendering anything else as max. So a request that turns thinking off (`enable_thinking: false`,
//! `thinking: false`, `reasoning_effort: "none"`, or `--default-reasoning-effort none`) renders the template's lowest
//! effort with the think block left open, and sets `Thinking::off` so the reasoning is withheld from the response.

use serde_json::{Map, Value};

use crate::api::ApiError;
use crate::pyjson;
use crate::sampling::Limit;

/// The efforts a request may name.
pub const EFFORTS: [&str; 7] = ["none", "minimal", "low", "medium", "high", "xhigh", "max"];

/// `nearest`'s ladder, highest first. `none` is not on it.
const LADDER: [&str; 6] = ["max", "xhigh", "high", "medium", "low", "minimal"];

const EFFORT_ERROR: &str = "reasoning_effort must be none, minimal, low, medium, high, xhigh or max";
const BUDGET_ERROR: &str = "thinking_budget must be an integer or null";

/// The efforts named in a chat template: a quoted minimal, low, medium, high, xhigh or max anywhere in its text
/// (GLM 5.3: low, high and max). Highest first.
pub fn template_levels(source: &str) -> Vec<&'static str> {
    let quoted =
        |l: &str| ['\'', '"'].iter().any(|a| ['\'', '"'].iter().any(|b| source.contains(&format!("{a}{l}{b}"))));
    LADDER.into_iter().filter(|l| quoted(l)).collect()
}

/// The level of `levels` nearest to `effort` on the ladder, ties going higher. With empty `levels`, `effort` itself:
/// a template without levels hears what the request sent.
pub fn nearest(effort: &'static str, levels: &[&'static str]) -> &'static str {
    let rung = |l: &str| LADDER.iter().position(|x| *x == l);
    let Some(want) = rung(effort) else { return effort };
    levels.iter().copied().min_by_key(|l| rung(l).map(|i| (i.abs_diff(want), i))).unwrap_or(effort)
}

/// Parse a server default (`--default-reasoning-effort`). `none` turns thinking off for requests that name no effort
/// and no switch.
pub fn parse_default_effort(s: &str) -> Result<&'static str, String> {
    match EFFORTS.iter().find(|e| **e == s) {
        Some(e) => Ok(e),
        None => Err(format!("--default-reasoning-effort {s}: want none, minimal, low, medium, high, xhigh or max")),
    }
}

/// A request's resolved thinking controls.
#[derive(Debug, PartialEq)]
pub struct Thinking {
    /// The effort the template renders, already one of its names; None: the template's own default.
    pub effort: Option<&'static str>,
    /// The think block's limit in reply tokens, after which the front end closes it (`budget_tokens`). 0: no limit.
    pub budget: Limit,
    /// The budget is the request's own positive value, not the server default.
    pub budget_explicit: bool,
    /// The request (or the server default) turned thinking off. On GLM the prompt renders the lowest effort with the
    /// think block open, and the response withholds the reasoning.
    pub off: bool,
}

impl Thinking {
    /// The budget in reply tokens for a request whose effective max_tokens is `max_tokens`. 0: no limit. A server
    /// default that scales with max_tokens gives at least 1 token: a tiny max_tokens makes the budget tighter, never
    /// unlimited, so the close comes first.
    pub fn budget_tokens(&self, max_tokens: u64) -> u64 {
        match self.budget {
            Limit::Tokens(n) => n,
            limit => limit.resolve(max_tokens).max(1),
        }
    }
}

/// The server defaults (front-end flags), used where a request names nothing.
#[derive(Clone, Copy, Default)]
pub struct Defaults {
    pub effort: Option<&'static str>,
    /// `--default-thinking-budget`: tokens, a fraction of the request's max_tokens, or both (sampling::Limit)
    pub budget: Limit,
    /// `--allow-thinking-off`, kept for existing callers. GLM turns thinking off whatever its value; other dialects
    /// read it as before.
    pub allow_off: bool,
}

fn non_null<'a>(m: Option<&'a Map<String, Value>>, k: &str) -> Option<&'a Value> {
    m.and_then(|m| m.get(k)).filter(|v| !v.is_null())
}

/// Read `reasoning_effort` (top level, else `chat_template_kwargs.reasoning_effort`, where vLLM's clients put it),
/// the `chat_template_kwargs.enable_thinking` / `thinking` switch and `thinking_budget`.
pub fn resolve(body: &Map<String, Value>, levels: &[&'static str], defaults: Defaults) -> Result<Thinking, ApiError> {
    let kw = body.get("chat_template_kwargs").and_then(Value::as_object);
    // enable_thinking wins over thinking when both are set
    let switch = non_null(kw, "enable_thinking").or_else(|| non_null(kw, "thinking")).map(|v| pyjson::truthy(Some(v)));
    let explicit = match non_null(Some(body), "reasoning_effort").or_else(|| non_null(kw, "reasoning_effort")) {
        None => None,
        Some(v) => match v.as_str().and_then(|s| EFFORTS.iter().find(|e| **e == s)) {
            Some(e) => Some(*e),
            None => return Err(ApiError::bad(EFFORT_ERROR, Some("reasoning_effort"))),
        },
    };
    // TensorFold: int(budget or default), and a budget <= 0 is no limit. So 0 and null take the server default, and a
    // negative budget is no limit even when the server has a default. Checked even when thinking is off.
    let (budget, budget_explicit) = match parse_budget(body.get("thinking_budget"))? {
        None | Some(0) => (defaults.budget, false),
        Some(n) if n < 0 => (Limit::Tokens(0), false),
        Some(n) => (Limit::Tokens(n as u64), true),
    };
    // as TensorFold: an explicit switch wins over "none"; an explicit effort wins over the server default
    let off = match switch {
        Some(on) => !on,
        None if explicit == Some("none") => true,
        None if explicit.is_some() => false,
        None => defaults.effort == Some("none"),
    };
    let effort = if off {
        Some(nearest("low", levels))
    } else {
        let explicit = explicit.filter(|e| *e != "none");
        let default_effort = defaults.effort.filter(|e| *e != "none");
        explicit.or(default_effort).map(|e| nearest(e, levels))
    };
    Ok(Thinking { effort, budget, budget_explicit, off })
}

/// `thinking_budget`: None when absent or null. TensorFold parses it with int(): an integer, a float with no
/// fraction, or a string holding an integer. Values past 64 bits saturate.
pub fn parse_budget(v: Option<&Value>) -> Result<Option<i64>, ApiError> {
    let bad = || ApiError::bad(BUDGET_ERROR, Some("thinking_budget"));
    let n: i128 = match v {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Number(n)) => {
            if let Some(i) = n.as_i64() {
                i as i128
            } else if let Some(u) = n.as_u64() {
                u as i128
            } else {
                let x = n.as_f64().ok_or_else(bad)?;
                if !(x.is_finite() && x == x.trunc()) {
                    return Err(bad());
                }
                x as i128
            }
        }
        Some(Value::String(s)) => {
            let t = s.trim();
            let digits = t.strip_prefix(['-', '+']).unwrap_or(t);
            if digits.is_empty() || !digits.bytes().all(|c| c.is_ascii_digit()) {
                return Err(bad());
            }
            t.parse::<i128>().unwrap_or(if t.starts_with('-') { i128::MIN } else { i128::MAX })
        }
        Some(_) => return Err(bad()), // booleans, lists, objects
    };
    Ok(Some(n.clamp(i64::MIN as i128, i64::MAX as i128) as i64))
}

/// `response_format`: absent, null or `{"type": "text"}` is plain text, which is what the engine produces.
/// JSON mode and schemas need constrained decoding. The engine does not have it yet.
pub fn check_response_format(v: Option<&Value>) -> Result<(), ApiError> {
    let ty = match v {
        None | Some(Value::Null) => return Ok(()),
        Some(Value::Object(o)) => o.get("type"),
        Some(_) => return Err(ApiError::bad("response_format must be an object", Some("response_format"))),
    };
    match ty.and_then(Value::as_str) {
        Some("text") => Ok(()),
        Some(t) => Err(ApiError::bad(
            format!("response_format {t:?} is not supported by the gr8 front end yet (only \"text\")"),
            Some("response_format"),
        )),
        None => Err(ApiError::bad("response_format needs a string type", Some("response_format"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const GLM: [&str; 3] = ["max", "high", "low"];

    fn body(v: Value) -> Map<String, Value> {
        v.as_object().unwrap().clone()
    }

    fn resolve_v(v: Value) -> Result<Thinking, ApiError> {
        resolve(&body(v), &GLM, Defaults::default())
    }

    fn err(v: Value) -> (String, Option<String>) {
        let e = resolve_v(v).err().expect("a 400");
        assert_eq!(e.status, 400);
        (e.message, e.param)
    }

    #[test]
    fn levels_come_from_the_quoted_names_in_the_template() {
        let glm = "{%- set e = reasoning_effort if reasoning_effort in ['low', 'high'] else 'max' -%}";
        assert_eq!(template_levels(glm), vec!["max", "high", "low"]);
        assert_eq!(template_levels(r#"{{ "minimal" }} 'medium" xhigh"#), vec!["medium", "minimal"]);
        assert!(template_levels("{{ messages }} high max").is_empty());
    }

    #[test]
    fn every_effort_goes_to_the_nearest_named_level_ties_higher() {
        let want = [
            ("minimal", "low"),
            ("low", "low"),
            ("medium", "high"),
            ("high", "high"),
            ("xhigh", "max"),
            ("max", "max"),
        ];
        for (e, n) in want {
            assert_eq!(nearest(e, &GLM), n, "{e}");
        }
        assert_eq!(nearest("medium", &["low", "medium", "xhigh"]), "medium");
        assert_eq!(nearest("high", &["low", "medium", "xhigh"]), "xhigh");
        assert_eq!(nearest("max", &["low", "medium", "xhigh"]), "xhigh");
        assert_eq!(nearest("minimal", &["medium"]), "medium");
        assert_eq!(nearest("xhigh", &[]), "xhigh"); // a template with no levels hears the request's own name
    }

    #[test]
    fn effort_is_read_top_level_first_then_from_the_template_kwargs() {
        assert_eq!(resolve_v(json!({})).unwrap().effort, None);
        assert_eq!(resolve_v(json!({"reasoning_effort": "medium"})).unwrap().effort, Some("high"));
        let kw = json!({"chat_template_kwargs": {"reasoning_effort": "minimal"}});
        assert_eq!(resolve_v(kw).unwrap().effort, Some("low"));
        let both = json!({"reasoning_effort": "xhigh", "chat_template_kwargs": {"reasoning_effort": "low"}});
        assert_eq!(resolve_v(both).unwrap().effort, Some("max"));
        let null_top = json!({"reasoning_effort": null, "chat_template_kwargs": {"reasoning_effort": "low"}});
        assert_eq!(resolve_v(null_top).unwrap().effort, Some("low"));
    }

    #[test]
    fn a_bad_effort_is_a_400() {
        for v in [json!("auto"), json!("LOW"), json!(""), json!(3), json!(true), json!(["low"])] {
            let (m, p) = err(json!({ "reasoning_effort": v }));
            assert_eq!((m.as_str(), p.as_deref()), (EFFORT_ERROR, Some("reasoning_effort")));
            let (m, _) = err(json!({"chat_template_kwargs": { "reasoning_effort": v }}));
            assert_eq!(m, EFFORT_ERROR);
        }
    }

    #[test]
    fn thinking_off_renders_the_lowest_effort_even_without_allow_off() {
        let mut offs = vec![];
        for v in [json!(false), json!(0), json!("")] {
            offs.push(json!({"chat_template_kwargs": {"enable_thinking": v}}));
            offs.push(json!({"chat_template_kwargs": {"thinking": v}}));
        }
        offs.push(json!({"reasoning_effort": "none"}));
        offs.push(json!({"chat_template_kwargs": {"reasoning_effort": "none"}}));
        for d in [Defaults::default(), Defaults { allow_off: true, ..Defaults::default() }] {
            for b in &offs {
                let t = resolve(&body(b.clone()), &GLM, d).unwrap_or_else(|_| panic!("{b} refused"));
                assert_eq!((t.effort, t.off), (Some("low"), true), "{b}");
            }
        }
        // the server default none turns thinking off for a request that names nothing
        let none = Defaults { effort: Some("none"), budget: Limit::Tokens(512), allow_off: false };
        assert_eq!(
            resolve(&body(json!({})), &GLM, none).unwrap(),
            Thinking { effort: Some("low"), budget: Limit::Tokens(512), budget_explicit: false, off: true }
        );
        // the lowest level the template names: none of the low rungs gives the lowest one there is
        let t = resolve(&body(json!({"reasoning_effort": "none"})), &["max", "high"], Defaults::default()).unwrap();
        assert_eq!((t.effort, t.off), (Some("high"), true));
    }

    #[test]
    fn an_explicit_switch_or_effort_wins_over_none() {
        let none = Defaults { effort: Some("none"), budget: Limit::Tokens(0), allow_off: false };
        let r = |b: Value, d: Defaults| resolve(&body(b), &GLM, d).unwrap();
        // enable_thinking wins over thinking when both are set
        let both = json!({"chat_template_kwargs": {"enable_thinking": true, "thinking": false}});
        assert!(!r(both, Defaults::default()).off);
        let both = json!({"chat_template_kwargs": {"enable_thinking": false, "thinking": true}});
        assert!(r(both, Defaults::default()).off);
        let both = json!({"chat_template_kwargs": {"enable_thinking": null, "thinking": false}});
        assert!(r(both, Defaults::default()).off);
        for key in ["enable_thinking", "thinking"] {
            for d in [Defaults::default(), none] {
                let t = r(json!({"reasoning_effort": "none", "chat_template_kwargs": {key: true}}), d);
                assert_eq!((t.off, t.effort), (false, None), "{key}");
                let t = r(json!({"chat_template_kwargs": {key: true}}), d);
                assert_eq!((t.off, t.effort), (false, None), "{key}");
            }
        }
        // an explicit effort wins over the server default none
        let t = r(json!({"reasoning_effort": "high"}), none);
        assert_eq!((t.off, t.effort), (false, Some("high")));
        let t = r(json!({"chat_template_kwargs": {"reasoning_effort": "medium"}}), none);
        assert_eq!((t.off, t.effort), (false, Some("high")));
        // an explicit on with a real default keeps the default
        let high = Defaults { effort: Some("high"), budget: Limit::Tokens(0), allow_off: false };
        let t = r(json!({"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": true}}), high);
        assert_eq!((t.off, t.effort), (false, Some("high")));
    }

    #[test]
    fn a_bad_budget_is_a_400_even_when_thinking_is_off() {
        for v in [json!("abc"), json!(true), json!(1.5)] {
            for b in [
                json!({"chat_template_kwargs": {"enable_thinking": false}, "thinking_budget": v}),
                json!({"chat_template_kwargs": {"thinking": false}, "thinking_budget": v}),
                json!({"reasoning_effort": "none", "thinking_budget": v}),
            ] {
                let (m, p) = err(b);
                assert_eq!((m.as_str(), p.as_deref()), (BUDGET_ERROR, Some("thinking_budget")), "{v}");
            }
        }
        // a good budget is kept when thinking is off
        let t = resolve_v(json!({"reasoning_effort": "none", "thinking_budget": 64})).unwrap();
        assert_eq!((t.off, t.budget, t.budget_explicit), (true, Limit::Tokens(64), true));
    }

    #[test]
    fn the_server_defaults_fill_what_the_request_leaves_out() {
        let d = Defaults { effort: Some("medium"), budget: Limit::Tokens(512), allow_off: false };
        let r = resolve(&body(json!({})), &GLM, d).unwrap();
        assert_eq!(
            r,
            Thinking { effort: Some("high"), budget: Limit::Tokens(512), budget_explicit: false, off: false }
        );
        // 0 and null take the default, as in TensorFold; a negative budget is no limit; a positive one wins
        for b in [json!(0), json!(null), json!("0")] {
            let r = resolve(&body(json!({"reasoning_effort": "low", "thinking_budget": b})), &GLM, d).unwrap();
            assert_eq!(
                r,
                Thinking { effort: Some("low"), budget: Limit::Tokens(512), budget_explicit: false, off: false },
                "{b}"
            );
        }
        let r = resolve(&body(json!({"thinking_budget": -1})), &GLM, d).unwrap();
        assert_eq!((r.budget, r.budget_explicit), (Limit::Tokens(0), false));
        let r = resolve(&body(json!({"thinking_budget": 64})), &GLM, d).unwrap();
        assert_eq!((r.budget, r.budget_explicit), (Limit::Tokens(64), true));
        let r = resolve(&body(json!({"thinking_budget": -1})), &GLM, Defaults::default()).unwrap();
        assert_eq!(r.budget, Limit::Tokens(0));
        assert_eq!(parse_default_effort("xhigh"), Ok("xhigh"));
        assert_eq!(parse_default_effort("none"), Ok("none"));
        assert!(parse_default_effort("auto").is_err());
    }

    #[test]
    fn a_default_budget_resolves_against_the_request_s_max_tokens() {
        let limit = Limit::parse("32768|0.8x").unwrap();
        let d = Defaults { effort: None, budget: limit, allow_off: false };
        let r = resolve(&body(json!({})), &GLM, d).unwrap();
        assert_eq!((r.budget, r.budget_explicit), (limit, false));
        assert_eq!(r.budget_tokens(8192), 6554);
        assert_eq!(r.budget_tokens(40000), 32000);
        assert_eq!(r.budget_tokens(200000), 32768);
        // a request's own budget is a token count whatever max_tokens is
        let r = resolve(&body(json!({"thinking_budget": 64})), &GLM, d).unwrap();
        assert_eq!((r.budget_tokens(8192), r.budget_explicit), (64, true));
        assert_eq!(resolve(&body(json!({"thinking_budget": -1})), &GLM, d).unwrap().budget_tokens(8192), 0);
        // 0 alone is no limit; a fraction that rounds to 0 is 1 (the close comes first), not no limit
        let t = |s: &str, max_tokens: u64| {
            let d = Defaults { effort: None, budget: Limit::parse(s).unwrap(), allow_off: false };
            resolve(&body(json!({})), &GLM, d).unwrap().budget_tokens(max_tokens)
        };
        assert_eq!(t("0", 8192), 0);
        assert_eq!(t("4096", 100), 4096);
        assert_eq!(t("0.1x", 4), 1);
        assert_eq!(t("32768|0.8x", 1), 1);
        assert_eq!(t("0.5x", 8192), 4096);
    }

    #[test]
    fn budget_parses_like_int() {
        let p = |v: Value| parse_budget(Some(&v));
        assert_eq!(p(json!(null)).unwrap(), None);
        assert_eq!(parse_budget(None).unwrap(), None);
        assert_eq!(p(json!(256)).unwrap(), Some(256));
        assert_eq!(p(json!(0)).unwrap(), Some(0));
        assert_eq!(p(json!(-5)).unwrap(), Some(-5));
        assert_eq!(p(json!(64.0)).unwrap(), Some(64));
        assert_eq!(p(json!(" 128 ")).unwrap(), Some(128));
        assert_eq!(p(json!("-3")).unwrap(), Some(-3));
        assert_eq!(p(json!(u64::MAX)).unwrap(), Some(i64::MAX));
        assert_eq!(p(json!("99999999999999999999999")).unwrap(), Some(i64::MAX));
        assert_eq!(p(json!("-99999999999999999999999")).unwrap(), Some(i64::MIN));
        for v in [json!(true), json!(false), json!(1.5), json!("1.5"), json!("abc"), json!(""), json!([1]), json!({})] {
            let e = p(v.clone()).err().unwrap_or_else(|| panic!("{v} accepted"));
            assert_eq!(
                (e.status, e.message.as_str(), e.param.as_deref()),
                (400, BUDGET_ERROR, Some("thinking_budget"))
            );
        }
    }

    #[test]
    fn response_format_text_only() {
        assert!(check_response_format(None).is_ok());
        assert!(check_response_format(Some(&json!(null))).is_ok());
        assert!(check_response_format(Some(&json!({"type": "text"}))).is_ok());
        for v in [json!({"type": "json_object"}), json!({"type": "json_schema", "json_schema": {"name": "x"}})] {
            let e = check_response_format(Some(&v)).err().unwrap();
            assert!(e.message.contains("is not supported by the gr8 front end yet"), "{}", e.message);
            assert_eq!((e.status, e.param.as_deref()), (400, Some("response_format")));
        }
        for v in [json!({"type": "grammar"}), json!({}), json!("json_object"), json!([])] {
            assert_eq!(check_response_format(Some(&v)).err().unwrap().status, 400);
        }
    }
}
