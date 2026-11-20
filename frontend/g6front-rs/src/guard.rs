//! The repeated-tool-call guard (`--tool-repeat-guard`): an agent loop in which the model makes the same tool call turn
//! after turn, and gets the same result each time, is taken back and resampled at a higher temperature before the
//! client sees it.
//!
//! The trigger reads the request's messages: the last assistant turn's tool calls (name plus canonical arguments, the
//! set of them when a turn has several), how many turns in a row ending the conversation made that same set, and
//! whether the tool results that answered those turns are identical. A changed result, or another call in between,
//! resets the count, so running the tests again after an edit is never blocked. The guard fires when the reply's calls
//! equal that set and the count has reached `repeats`.
//!
//! The action rewinds the reply to just before the call block's `<tool_call>` and continues in a new engine run at
//! temperature max(request, `temperature`) and top_p min(request, 0.95), the way a thinking budget continues after its
//! close (gen.rs). The parser held the call back (parser.rs `guard_names`), so the client never saw it. After
//! `retries` attempts the reply goes out as sampled, so the guard cannot loop itself.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::Value;

use crate::api::ApiError;
use crate::gen::Generation;
use crate::parser::GlmParser;

/// The server's settings (`--tool-repeat-guard*`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    pub on: bool,
    /// turns in a row the same call must already have made before a repeat is taken back
    pub repeats: u32,
    /// the resample's temperature floor
    pub temperature: f64,
    /// resamples per request before the reply goes out as sampled
    pub retries: u32,
}

impl Default for Config {
    fn default() -> Config {
        Config { on: false, repeats: 2, temperature: 0.7, retries: 2 }
    }
}

/// the resample's top_p ceiling
pub const TOP_P: f64 = 0.95;

/// `/metrics`: `gr8:tool_repeat_retries_total` and `gr8:tool_repeat_given_up_total`
pub static RETRIES: AtomicU64 = AtomicU64::new(0);
pub static GIVEN_UP: AtomicU64 = AtomicU64::new(0);

/// A tool call as the trigger compares it: the name and the arguments as canonical JSON.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Call {
    pub name: String,
    pub args: String,
}

/// `args` as compact JSON with sorted keys at every level. Text that is not JSON stays as it is.
pub fn canonical(args: &str) -> String {
    fn sort(v: &Value) -> Value {
        match v {
            Value::Object(m) => {
                let b: BTreeMap<&String, Value> = m.iter().map(|(k, v)| (k, sort(v))).collect();
                Value::Object(b.into_iter().map(|(k, v)| (k.clone(), v)).collect())
            }
            Value::Array(a) => Value::Array(a.iter().map(sort).collect()),
            v => v.clone(),
        }
    }
    match serde_json::from_str::<Value>(args) {
        Ok(v) => sort(&v).to_string(),
        Err(_) => args.to_string(),
    }
}

/// The calls of a reply or a message, in the order the trigger compares them.
pub fn call_set(calls: impl IntoIterator<Item = (String, String)>) -> Vec<Call> {
    let mut v: Vec<Call> = calls.into_iter().map(|(name, args)| Call { name, args: canonical(&args) }).collect();
    v.sort();
    v
}

/// What the end of the conversation repeats: the last assistant turn's calls, and how many turns in a row made them
/// and got the same results.
#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    pub calls: Vec<Call>,
    pub count: u32,
}

/// One assistant turn with tool calls and the results that answered it.
struct Turn {
    calls: Vec<Call>,
    results: Vec<String>,
}

fn message_calls(m: &Value) -> Option<Vec<(String, String, String)>> {
    let tc = m.get("tool_calls")?.as_array().filter(|a| !a.is_empty())?;
    let mut out = Vec::with_capacity(tc.len());
    for t in tc {
        let f = t.get("function")?;
        let name = f.get("name")?.as_str()?.to_string();
        let args = match f.get("arguments") {
            Some(Value::String(s)) => s.clone(),
            Some(v) if !v.is_null() => v.to_string(),
            _ => String::new(),
        };
        let id = t.get("id").and_then(Value::as_str).unwrap_or("").to_string();
        out.push((id, name, args));
    }
    Some(out)
}

fn result_text(m: &Value) -> String {
    match m.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(v) if !v.is_null() => v.to_string(),
        _ => String::new(),
    }
}

/// The trailing turns of `messages`, latest first: each an assistant message with tool calls followed by the tool
/// messages that answer it. Anything else ends the walk, and a conversation that does not end with tool results has no
/// turns.
fn trailing_turns(messages: &[Value]) -> Vec<Turn> {
    let role = |m: &Value| m.get("role").and_then(Value::as_str).unwrap_or("").to_string();
    let mut turns = Vec::new();
    let mut i = messages.len();
    loop {
        // the tool results, then the assistant turn they answer
        let mut results: Vec<(String, String)> = Vec::new();
        while i > 0 && role(&messages[i - 1]) == "tool" {
            i -= 1;
            let m = &messages[i];
            let id = m.get("tool_call_id").and_then(Value::as_str).unwrap_or("").to_string();
            results.push((id, result_text(m)));
        }
        if results.is_empty() || i == 0 || role(&messages[i - 1]) != "assistant" {
            return turns;
        }
        i -= 1;
        let Some(calls) = message_calls(&messages[i]) else { return turns };
        // each call with its result (by id, else the next result the ids do not place), in the set's order
        let by_id = |id: &str| results.iter().find(|(rid, _)| !id.is_empty() && rid == id).map(|(_, t)| t.clone());
        let mut unplaced = results.iter().filter(|(rid, _)| rid.is_empty() || !calls.iter().any(|(id, ..)| id == rid));
        let mut pairs: Vec<(Call, String)> = calls
            .iter()
            .map(|(id, name, args)| {
                let r = by_id(id).or_else(|| unplaced.next().map(|(_, t)| t.clone())).unwrap_or_default();
                (Call { name: name.clone(), args: canonical(args) }, r)
            })
            .collect();
        pairs.sort();
        let (calls, results) = pairs.into_iter().unzip();
        turns.push(Turn { calls, results });
    }
}

impl Candidate {
    /// The last assistant turn's calls and how many turns in a row ending `messages` made them with identical results;
    /// None when the conversation does not end with a tool turn.
    pub fn from_messages(messages: &Value) -> Option<Candidate> {
        let turns = trailing_turns(messages.as_array()?);
        let last = turns.first()?;
        let count = turns.iter().take_while(|t| t.calls == last.calls && t.results == last.results).count() as u32;
        Some(Candidate { calls: last.calls.clone(), count })
    }
}

/// One request's guard: armed when the server has it on (or the request asked) and the conversation ends with a tool
/// turn.
pub struct Guard {
    pub cfg: Config,
    pub candidate: Candidate,
    /// resamples made so far
    pub attempts: u32,
}

impl Guard {
    /// `on`: the server default or the request's `tool_repeat_guard` field.
    pub fn new(cfg: Config, on: bool, messages: Option<&Value>) -> Option<Guard> {
        if !on {
            return None;
        }
        let candidate = Candidate::from_messages(messages?)?;
        Some(Guard { cfg, candidate, attempts: 0 })
    }
    /// The names a call must not stream under: the parser holds such a call until the reply ends.
    pub fn names(&self) -> Vec<String> {
        self.candidate.calls.iter().map(|c| c.name.clone()).collect()
    }
    /// Whether the reply's calls are the repeat the trigger describes.
    pub fn fires(&self, calls: &[Call]) -> bool {
        !calls.is_empty() && self.candidate.count >= self.cfg.repeats && calls == self.candidate.calls.as_slice()
    }
    fn describe(&self) -> String {
        self.candidate.calls.iter().map(|c| format!("{}({})", c.name, c.args)).collect::<Vec<_>>().join(", ")
    }
}

/// At the reply's end, with `parser` holding the reply's final text short of `finished`: whether the guard took the
/// held call back. True: the parser dropped the pending call block, `gen` runs on from just before it, and the caller
/// goes on reading steps. False: nothing changed, the caller finishes the reply.
pub async fn take_back(gen: &mut Generation, parser: &mut GlmParser, guard: &mut Guard) -> Result<bool, ApiError> {
    let Some(calls) = parser.pending_calls() else { return Ok(false) };
    let calls = call_set(calls);
    if !guard.fires(&calls) {
        return Ok(false);
    }
    if guard.attempts >= guard.cfg.retries {
        GIVEN_UP.fetch_add(1, Ordering::Relaxed);
        log::info!(
            "{}: tool repeat guard: {} still repeats after {} resamples; sent as sampled",
            gen.id,
            guard.describe(),
            guard.attempts
        );
        return Ok(false);
    }
    let Some(block) = parser.pending_bytes() else { return Ok(false) };
    let cut = parser.fed_bytes() - block;
    let temperature = gen.params.gen.temperature.max(guard.cfg.temperature);
    let top_p = gen.params.gen.top_p.min(TOP_P);
    guard.attempts += 1;
    RETRIES.fetch_add(1, Ordering::Relaxed);
    log::info!(
        "{}: tool repeat guard: {} made {} turns in a row with the same result; resampling at temperature {temperature} top_p {top_p} (attempt {} of {})",
        gen.id,
        guard.describe(),
        guard.candidate.count,
        guard.attempts,
        guard.cfg.retries
    );
    if !gen.resample(cut, temperature, top_p, guard.attempts).await? {
        return Ok(false);
    }
    parser.drop_pending();
    Ok(true)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn call(name: &str, args: &str) -> Value {
        json!({"id": format!("call-{name}-{}", args.len()), "type": "function", "function": {"name": name, "arguments": args}})
    }
    fn assistant(calls: Vec<Value>) -> Value {
        json!({"role": "assistant", "content": null, "tool_calls": calls})
    }
    fn result(c: &Value, text: &str) -> Value {
        json!({"role": "tool", "tool_call_id": c["id"], "content": text})
    }
    /// n turns of `calls`, each answered by `results`
    fn turns(n: usize, calls: &[(&str, &str)], results: &[&str]) -> Vec<Value> {
        let mut m = vec![json!({"role": "user", "content": "fix it"})];
        for _ in 0..n {
            let cs: Vec<Value> = calls.iter().map(|(n, a)| call(n, a)).collect();
            m.push(assistant(cs.clone()));
            for (c, r) in cs.iter().zip(results) {
                m.push(result(c, r));
            }
        }
        m
    }
    fn candidate(m: Vec<Value>) -> Option<Candidate> {
        Candidate::from_messages(&Value::Array(m))
    }
    fn calls(v: &[(&str, &str)]) -> Vec<Call> {
        call_set(v.iter().map(|(n, a)| (n.to_string(), a.to_string())))
    }

    #[test]
    fn canonical_json_sorts_keys_at_every_level() {
        assert_eq!(
            canonical(r#"{"b": 1, "a": {"d": [2, {"z": 1, "y": 2}], "c": "x"}}"#),
            r#"{"a":{"c":"x","d":[2,{"y":2,"z":1}]},"b":1}"#
        );
        assert_eq!(canonical("{}"), "{}");
        assert_eq!(canonical("not json"), "not json");
    }

    #[test]
    fn the_count_is_the_trailing_turns_with_the_same_calls_and_results() {
        let same = turns(3, &[("run_tests", "{}")], &["Ran 5 tests OK"]);
        assert_eq!(candidate(same), Some(Candidate { calls: calls(&[("run_tests", "{}")]), count: 3 }));
        // a changed result resets the count to the turns since it
        let mut m = turns(2, &[("run_tests", "{}")], &["FAILED"]);
        m.extend(turns(1, &[("run_tests", "{}")], &["OK"]).into_iter().skip(1));
        assert_eq!(candidate(m).unwrap().count, 1);
        // another call in between
        let mut m = turns(2, &[("run_tests", "{}")], &["OK"]);
        m.extend(turns(1, &[("edit", r#"{"path": "a"}"#)], &["done"]).into_iter().skip(1));
        m.extend(turns(1, &[("run_tests", "{}")], &["OK"]).into_iter().skip(1));
        assert_eq!(candidate(m).unwrap().count, 1);
        // different arguments are another call
        let mut m = turns(2, &[("read", r#"{"path": "a"}"#)], &["x"]);
        m.extend(turns(1, &[("read", r#"{"path": "b"}"#)], &["x"]).into_iter().skip(1));
        assert_eq!(candidate(m).unwrap().count, 1);
    }

    #[test]
    fn argument_key_order_and_whitespace_do_not_matter() {
        let mut m = turns(1, &[("read", r#"{"path": "a", "n": 1}"#)], &["x"]);
        m.extend(turns(1, &[("read", r#"{"n":1,"path":"a"}"#)], &["x"]).into_iter().skip(1));
        assert_eq!(candidate(m).unwrap().count, 2);
    }

    #[test]
    fn several_calls_in_a_turn_compare_as_a_set() {
        let mut m = turns(1, &[("a", "{}"), ("b", "{}")], &["ra", "rb"]);
        m.extend(turns(1, &[("b", "{}"), ("a", "{}")], &["rb", "ra"]).into_iter().skip(1));
        let c = candidate(m).unwrap();
        assert_eq!((c.calls.len(), c.count), (2, 2));
        // one call fewer is another set
        let mut m = turns(1, &[("a", "{}"), ("b", "{}")], &["ra", "rb"]);
        m.extend(turns(1, &[("a", "{}")], &["ra"]).into_iter().skip(1));
        assert_eq!(candidate(m).unwrap().count, 1);
    }

    #[test]
    fn a_conversation_that_does_not_end_with_tool_results_has_no_candidate() {
        let mut m = turns(2, &[("run_tests", "{}")], &["OK"]);
        assert!(candidate(m.clone()).is_some());
        m.push(json!({"role": "user", "content": "now what"}));
        assert_eq!(candidate(m), None);
        assert_eq!(candidate(vec![json!({"role": "user", "content": "hi"})]), None);
        assert_eq!(Candidate::from_messages(&json!("x")), None);
    }

    #[test]
    fn the_guard_fires_on_the_same_set_once_the_count_is_reached() {
        let cfg = Config { on: true, ..Default::default() };
        let msgs = Value::Array(turns(2, &[("run_tests", "{}")], &["OK"]));
        let g = Guard::new(cfg, true, Some(&msgs)).unwrap();
        assert_eq!(g.names(), vec!["run_tests"]);
        assert!(g.fires(&calls(&[("run_tests", "{ }")])));
        assert!(!g.fires(&calls(&[("run_tests", r#"{"v": 1}"#)])));
        assert!(!g.fires(&calls(&[("edit", "{}")])));
        assert!(!g.fires(&[]));
        let once = Value::Array(turns(1, &[("run_tests", "{}")], &["OK"]));
        assert!(!Guard::new(cfg, true, Some(&once)).unwrap().fires(&calls(&[("run_tests", "{}")])));
        assert!(Guard::new(Config { repeats: 1, ..cfg }, true, Some(&once))
            .unwrap()
            .fires(&calls(&[("run_tests", "{}")])));
        assert!(Guard::new(cfg, false, Some(&msgs)).is_none());
        assert!(Guard::new(cfg, true, None).is_none());
    }
}
