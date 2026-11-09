//! The pseudo model's weights: a TOML file of rules. A rule is a regex over one part of the prompt and the reply it
//! gives. Every rule that matches is a candidate; the most specific wins (its `specificity`, then the longer match,
//! then the earlier rule), and the candidates' weights give the logprobs and, with a seed, the sampling.
//!
//! A rule's `set` writes values into the KV, one JSON object the engine keeps across requests, and `{{name}}` in a
//! reply reads one back (`{{*}}`: the whole object). `needs` names the keys that must be set for the rule to match.

use std::collections::BTreeMap;

use regex::{Captures, Regex};
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::prompt::{Prompt, THINK_CLOSE, TOOL_CLOSE, TOOL_OPEN};

pub const STOP_USER: &str = "<|user|>";
pub const STOP_OBSERVATION: &str = "<|observation|>";

/// The variables rules set.
pub type Kv = Map<String, Value>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FileSrc {
    #[serde(default)]
    engine: Engine,
    #[serde(default, rename = "rule")]
    rules: Vec<RuleSrc>,
    fallback: ReplySrc,
}

/// How the engine paces and labels itself (the `[engine]` table).
#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct Engine {
    /// The hello's model name.
    pub model: String,
    pub max_model_len: u64,
    /// Generated tokens a second (0 = as fast as the socket takes them).
    pub tokens_per_s: f64,
    /// Ids a `tokens` frame carries, as an engine's accepted drafts arrive together.
    pub chunk: usize,
    /// Prompt tokens a second before the first token (0 = no prefill delay).
    pub prefill_tps: f64,
    /// The reasoning a reply gives when its rule names none.
    pub reasoning: String,
    /// Draft tokens per verify round, for the speculative-decoding counters (0: none, and no `spec` is reported). A
    /// `tokens` frame is then one round: 1 + its accepted drafts, from a fixed cycle, in place of `chunk`.
    pub draft_k: usize,
}

impl Default for Engine {
    fn default() -> Engine {
        Engine {
            model: "glm-5.3".into(),
            max_model_len: 200_000,
            tokens_per_s: 2000.0,
            chunk: 4,
            prefill_tps: 0.0,
            reasoning: "Let me work out the answer.".into(),
            draft_k: 0,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RuleSrc {
    name: String,
    #[serde(rename = "match")]
    pattern: String,
    #[serde(default)]
    on: Scope,
    #[serde(default)]
    tools: Vec<String>,
    #[serde(default)]
    specificity: i64,
    reasoning: Option<String>,
    #[serde(default)]
    content: String,
    #[serde(default, rename = "call")]
    calls: Vec<CallSrc>,
    #[serde(default)]
    fill: bool,
    #[serde(default)]
    set: BTreeMap<String, String>,
    #[serde(default)]
    needs: Vec<String>,
}

#[derive(Deserialize, Default, Clone)]
#[serde(deny_unknown_fields)]
struct ReplySrc {
    reasoning: Option<String>,
    #[serde(default)]
    content: String,
    #[serde(default, rename = "call")]
    calls: Vec<CallSrc>,
    #[serde(default)]
    fill: bool,
    #[serde(default)]
    set: BTreeMap<String, String>,
}

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct CallSrc {
    name: String,
    #[serde(default)]
    arguments: BTreeMap<String, toml::Value>,
}

/// The part of the prompt a rule's regex reads.
#[derive(Deserialize, Default, Clone, Copy, Debug, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Scope {
    /// The last user message (and the rule applies only when it is the last turn, not a tool result).
    #[default]
    User,
    /// The last tool result (the rule applies only when the prompt ends with one).
    Tool,
    /// The system messages and the tool block.
    System,
    /// The whole decoded prompt.
    Prompt,
}

pub struct Rule {
    pub name: String,
    re: Regex,
    on: Scope,
    tools: Vec<String>,
    needs: Vec<String>,
    pub specificity: i64,
    reply: ReplySrc,
}

pub struct Rules {
    pub engine: Engine,
    pub rules: Vec<Rule>,
    fallback: ReplySrc,
}

/// A rule that matched a prompt.
pub struct Candidate<'a> {
    pub rule: Option<&'a Rule>,
    caps: Option<Captures<'a>>,
    /// The match's length in bytes (0 for the fallback).
    pub len: usize,
}

impl Candidate<'_> {
    pub fn name(&self) -> &str {
        self.rule.map_or("fallback", |r| r.name.as_str())
    }

    /// The ranking score and logit: specificity first, the longer match within it.
    pub fn score(&self) -> f64 {
        self.rule.map_or(-1e9, |r| r.specificity as f64 + self.len.min(999) as f64 / 1000.0)
    }
}

impl Rules {
    pub fn parse(src: &str) -> Result<Rules, String> {
        let f: FileSrc = toml::from_str(src).map_err(|e| e.to_string())?;
        let mut rules = Vec::with_capacity(f.rules.len());
        for r in f.rules {
            let re = Regex::new(&r.pattern).map_err(|e| format!("rule {}: match: {e}", r.name))?;
            if rules.iter().any(|x: &Rule| x.name == r.name) {
                return Err(format!("rule {}: the name is used twice", r.name));
            }
            let reply =
                ReplySrc { reasoning: r.reasoning, content: r.content, calls: r.calls, fill: r.fill, set: r.set };
            let (on, tools, needs, specificity) = (r.on, r.tools, r.needs, r.specificity);
            rules.push(Rule { name: r.name, re, on, tools, needs, specificity, reply });
        }
        if f.engine.chunk == 0 {
            return Err("engine.chunk must be at least 1".into());
        }
        Ok(Rules { engine: f.engine, rules, fallback: f.fallback })
    }

    pub fn load(path: &std::path::Path) -> Result<Rules, String> {
        let src = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Rules::parse(&src).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Every rule that matches `p` with the keys it needs set in `kv`, best first, then the fallback.
    pub fn candidates<'a>(&'a self, p: &'a Prompt, kv: &Kv) -> Vec<Candidate<'a>> {
        let mut out: Vec<Candidate> = Vec::new();
        for r in &self.rules {
            if !r.tools.iter().all(|t| t == "*" && !p.tools.is_empty() || p.has_tool(t)) {
                continue;
            }
            let text: &str = match r.on {
                Scope::User if p.tool_result.is_none() => &p.user,
                Scope::Tool => match &p.tool_result {
                    Some(t) => t,
                    None => continue,
                },
                Scope::System => &p.system,
                Scope::Prompt => &p.text,
                Scope::User => continue,
            };
            if let Some(c) = r.re.captures(text) {
                if !r.needs.iter().all(|k| kv.contains_key(&expand(&c, k))) {
                    continue;
                }
                let len = c.get(0).map_or(0, |m| m.len());
                out.push(Candidate { rule: Some(r), caps: Some(c), len });
            }
        }
        // stable: equal scores keep file order
        out.sort_by(|a, b| b.score().total_cmp(&a.score()));
        out.push(Candidate { rule: None, caps: None, len: 0 });
        out
    }

    /// The reply text a candidate gives `p` with the variables in `kv`, its stop marker last unless the reply fills to
    /// max_tokens, and what it sets.
    pub fn reply(&self, c: &Candidate, p: &Prompt, kv: &Kv) -> Reply {
        let src = c.rule.map_or(&self.fallback, |r| &r.reply);
        let caps = |s: &str| c.caps.as_ref().map_or_else(|| s.to_string(), |caps| expand(caps, s));
        let fill = |s: &str| recall(&caps(s), kv);
        let set = src.set.iter().map(|(k, v)| (caps(k), scalar(&caps(v)))).collect();
        let calls: Vec<(String, String)> = src
            .calls
            .iter()
            .map(|call| {
                let args: String = call
                    .arguments
                    .iter()
                    .map(|(k, v)| format!("<arg_key>{k}</arg_key><arg_value>{}</arg_value>", arg_text(v, &fill)))
                    .collect();
                (fill(&call.name), args)
            })
            .collect();
        // a forced tool_choice opened a call: go on inside it, the name only if the opener holds none
        if let Some(open) = &p.open_call {
            let (name, args) = calls
                .iter()
                .find(|(n, _)| open.trim() == n.as_str())
                .or(calls.first())
                .cloned()
                .unwrap_or_else(|| first_tool_call(p));
            let head = if open.trim().is_empty() { name } else { String::new() };
            return Reply { text: format!("{head}{args}{TOOL_CLOSE}{STOP_OBSERVATION}"), fill: None, set };
        }
        let content = fill(&src.content);
        if !p.chat {
            // a raw completion: the text goes on, no reasoning, no turn markers
            return Reply { text: format!("{content}{STOP_USER}"), fill: None, set };
        }
        let mut text = String::new();
        if !p.think_closed {
            text += &fill(src.reasoning.as_deref().unwrap_or(&self.engine.reasoning));
            text += THINK_CLOSE;
        }
        text += &content;
        for (name, args) in &calls {
            text += &format!("{TOOL_OPEN}{name}{args}{TOOL_CLOSE}");
        }
        if src.fill {
            return Reply { text, fill: Some(if content.is_empty() { " ".into() } else { content }), set };
        }
        text += if calls.is_empty() { STOP_USER } else { STOP_OBSERVATION };
        Reply { text, fill: None, set }
    }
}

/// A reply's text, with `fill` the text it repeats after it until max_tokens, and the variables it sets.
#[derive(Debug, PartialEq)]
pub struct Reply {
    pub text: String,
    pub fill: Option<String>,
    pub set: Vec<(String, Value)>,
}

fn expand(caps: &Captures, s: &str) -> String {
    let mut out = String::new();
    caps.expand(s, &mut out);
    out
}

/// `s` with each `{{name}}` replaced by that variable (a string as itself, anything else as JSON, nothing when it is
/// unset), and `{{*}}` by the whole KV.
fn recall(s: &str, kv: &Kv) -> String {
    let mut out = String::new();
    let mut rest = s;
    while let Some(i) = rest.find("{{") {
        let Some(j) = rest[i..].find("}}") else { break };
        out += &rest[..i];
        let name = &rest[i + 2..i + j];
        out += &match (name, kv.get(name)) {
            ("*", _) => Value::Object(kv.clone()).to_string(),
            (_, Some(Value::String(v))) => v.clone(),
            (_, Some(v)) => v.to_string(),
            (_, None) => String::new(),
        };
        rest = &rest[i + j + 2..];
    }
    out + rest
}

/// A set value: a JSON number, true, false or null as itself, anything else as a string.
fn scalar(s: &str) -> Value {
    match serde_json::from_str::<Value>(s.trim()) {
        Ok(v @ (Value::Number(_) | Value::Bool(_) | Value::Null)) => v,
        _ => Value::String(s.to_string()),
    }
}

/// An argument as GLM writes it: a string as itself (captures filled in), anything else as JSON.
fn arg_text(v: &toml::Value, fill: &dyn Fn(&str) -> String) -> String {
    match v {
        toml::Value::String(s) => fill(s),
        other => toml_json(other, fill).to_string(),
    }
}

fn toml_json(v: &toml::Value, fill: &dyn Fn(&str) -> String) -> Value {
    match v {
        toml::Value::String(s) => Value::String(fill(s)),
        toml::Value::Integer(i) => Value::from(*i),
        toml::Value::Float(f) => serde_json::Number::from_f64(*f).map_or(Value::Null, Value::Number),
        toml::Value::Boolean(b) => Value::Bool(*b),
        toml::Value::Datetime(d) => Value::String(d.to_string()),
        toml::Value::Array(a) => Value::Array(a.iter().map(|x| toml_json(x, fill)).collect()),
        toml::Value::Table(t) => Value::Object(t.iter().map(|(k, x)| (k.clone(), toml_json(x, fill))).collect()),
    }
}

/// A call to the prompt's first tool, for a forced tool_choice no rule describes: "example" for each parameter.
fn first_tool_call(p: &Prompt) -> (String, String) {
    let Some(t) = p.tools.first() else { return (String::new(), String::new()) };
    let args = t.params.iter().map(|k| format!("<arg_key>{k}</arg_key><arg_value>example</arg_value>")).collect();
    (t.name.clone(), args)
}

/// The candidates' weights at `temperature` (softmax of their scores; temperature <= 0 counts as 1).
pub fn weights(cands: &[Candidate], temperature: f64) -> Vec<f64> {
    let t = if temperature > 0.0 { temperature } else { 1.0 };
    let best = cands.first().map_or(0.0, |c| c.score());
    let w: Vec<f64> = cands.iter().map(|c| ((c.score() - best) / t).exp()).collect();
    let sum: f64 = w.iter().sum();
    w.into_iter().map(|x| x / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const RULES: &str = r#"
[engine]
reasoning = "Thinking."

[[rule]]
name = "weather"
match = '(?i)weather in (?P<city>[A-Z][a-z]+)'
tools = ["get_weather"]
specificity = 10
reasoning = "The user wants the weather in ${city}."
content = "Checking ${city}."
[[rule.call]]
name = "get_weather"
arguments = { city = "${city}", days = 3, metric = true }

[[rule]]
name = "weather-no-tool"
match = '(?i)weather'
content = "I can't check the weather."

[[rule]]
name = "after-weather"
match = 'temp_c\D*(\d+)'
on = "tool"
content = "It is $1 degrees."

[[rule]]
name = "short"
match = 'weather in'
content = "A shorter match."

[fallback]
content = "I don't know."
"#;

    fn prompt(user: &str, tools: bool) -> Prompt {
        let t = if tools {
            "<|system|><tools>\n{\"name\": \"get_weather\", \"parameters\": {\"properties\": {\"city\": {}}}}\n</tools>"
        } else {
            ""
        };
        Prompt::parse(&format!("[gMASK]<sop>{t}<|user|>{user}<|assistant|><think>"))
    }

    #[test]
    fn the_most_specific_rule_wins_then_the_longer_match() {
        let r = Rules::parse(RULES).unwrap();
        let p = prompt("What is the weather in Paris?", true);
        let names: Vec<_> = r.candidates(&p, &Kv::new()).iter().map(|c| c.name().to_string()).collect();
        assert_eq!(names, ["weather", "short", "weather-no-tool", "fallback"]);
        // without the tool the tool rule drops out
        let p = prompt("What is the weather in Paris?", false);
        assert_eq!(r.candidates(&p, &Kv::new())[0].name(), "short");
        let p = prompt("Tell me a joke", false);
        assert_eq!(r.candidates(&p, &Kv::new())[0].name(), "fallback");
    }

    #[test]
    fn captures_fill_the_reply_and_its_call() {
        let r = Rules::parse(RULES).unwrap();
        let p = prompt("What is the weather in Paris?", true);
        let c = r.candidates(&p, &Kv::new());
        assert_eq!(
            r.reply(&c[0], &p, &Kv::new()),
            Reply {
                text: "The user wants the weather in Paris.</think>Checking Paris.<tool_call>get_weather\
<arg_key>city</arg_key><arg_value>Paris</arg_value><arg_key>days</arg_key><arg_value>3</arg_value>\
<arg_key>metric</arg_key><arg_value>true</arg_value></tool_call><|observation|>"
                    .into(),
                fill: None,
                set: Vec::new(),
            }
        );
    }

    #[test]
    fn a_tool_result_a_closed_think_block_and_a_forced_call() {
        let r = Rules::parse(RULES).unwrap();
        let after = Prompt::parse(
            "<|user|>weather in Paris<|assistant|></think><tool_call>get_weather</tool_call>\
<|observation|><tool_response>{\"temp_c\": 21}</tool_response><|assistant|><think>",
        );
        let c = r.candidates(&after, &Kv::new());
        assert_eq!(c[0].name(), "after-weather");
        assert_eq!(r.reply(&c[0], &after, &Kv::new()).text, "Thinking.</think>It is 21 degrees.<|user|>");
        let mut p = prompt("What is the weather in Paris?", true);
        p.think_closed = true;
        let c = r.candidates(&p, &Kv::new());
        assert!(r.reply(&c[0], &p, &Kv::new()).text.starts_with("Checking Paris."));
        let mut forced = prompt("What is the weather in Paris?", true);
        forced.think_closed = true;
        forced.open_call = Some(String::new());
        let c = r.candidates(&forced, &Kv::new());
        assert_eq!(r.reply(&c[0], &forced, &Kv::new()).text, "get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>\
<arg_key>days</arg_key><arg_value>3</arg_value><arg_key>metric</arg_key><arg_value>true</arg_value></tool_call><|observation|>");
    }

    #[test]
    fn weights_follow_the_ranking() {
        let r = Rules::parse(RULES).unwrap();
        let p = prompt("What is the weather in Paris?", true);
        let c = r.candidates(&p, &Kv::new());
        let w = weights(&c, 0.0);
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(w.windows(2).all(|x| x[0] >= x[1]) && w[0] > 0.99);
        // a higher temperature spreads the mass
        assert!(weights(&c, 20.0)[1] > w[1]);
    }

    #[test]
    fn rules_set_and_recall_variables() {
        let r = Rules::parse(
            r#"
[[rule]]
name = "set"
match = '^set (?P<k>\w+) to (?P<v>.+)$'
content = "Done."
set = { "${k}" = "${v}" }

[[rule]]
name = "get"
match = '^what is (?P<k>\w+)$'
needs = ["${k}"]
content = "${k} is {{${k}}}; all: {{*}}"

[fallback]
content = "${k} is not set."
"#,
        )
        .unwrap();
        let mut kv = Kv::new();
        for (msg, v) in [("set x to 5", Value::from(5)), ("set name to Ada Lovelace", Value::from("Ada Lovelace"))] {
            let p = prompt(msg, false);
            let reply = r.reply(&r.candidates(&p, &kv)[0], &p, &kv);
            assert_eq!(reply.set, [(msg.split(' ').nth(1).unwrap().to_string(), v)]);
            kv.extend(reply.set);
        }
        let p = prompt("what is x", false);
        let c = r.candidates(&p, &kv);
        assert_eq!(c[0].name(), "get");
        assert!(r.reply(&c[0], &p, &kv).text.ends_with(r#"x is 5; all: {"x":5,"name":"Ada Lovelace"}<|user|>"#));
        let p = prompt("what is y", false);
        assert_eq!(r.candidates(&p, &kv)[0].name(), "fallback", "get needs y");
    }

    #[test]
    fn bad_files_are_refused() {
        assert!(Rules::parse("[fallback]\ncontent='x'\n[[rule]]\nname='a'\nmatch='('\n").is_err());
        assert!(Rules::parse(
            "[fallback]\ncontent='x'\n[[rule]]\nname='a'\nmatch='a'\n[[rule]]\nname='a'\nmatch='b'\n"
        )
        .is_err());
        assert!(Rules::parse("[fallback]\ncontent='x'\n[[rule]]\nname='a'\nmatch='a'\ntypo=1\n").is_err());
    }
}
