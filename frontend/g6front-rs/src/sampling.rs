//! OpenAI/vLLM request fields -> PROTOCOL.md `generate` fields, with vLLM's defaults and validation
//! (vLLM's checks in the same order, with the same messages).

use std::collections::HashSet;

use serde_json::{Map, Value};

use crate::api::ApiError;
use crate::pyjson;
use crate::wire::GenParams;

/// The most ids `stop_token_ids` may name.
pub const MAX_STOP_TOKEN_IDS: usize = 256;

pub struct Params {
    pub gen: GenParams,
    pub stop: Vec<String>,
    pub include_stop: bool,
}

/// One part of a penalty spec: `[+]A[*M]`, an amount added and a multiplier (`base=0.2` adds 0.2; `thinking=*1.1`
/// adds nothing and boosts by 10%; `thinking=+0.1*1.1` both). `mul` is None when `*M` was not written: 1 for every
/// part but `temperature`, whose factor is then absent rather than 0.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Part {
    pub add: f64,
    pub mul: Option<f64>,
}

impl Part {
    fn parse(t: &str) -> Result<Part, String> {
        let t = t.trim();
        let (add, mul) = match t.split_once('*') {
            Some((a, m)) => (a.trim(), Some(m.trim())),
            None => (t, None),
        };
        let mut p = Part::default();
        if !add.is_empty() || mul.is_none() {
            p.add = add.parse().map_err(|_| format!("{add:?} is not a number"))?;
            if !p.add.is_finite() {
                return Err(format!("{add:?} is not a number"));
            }
        }
        if let Some(m) = mul {
            let v: f64 = m.parse().map_err(|_| format!("{m:?} is not a multiplier"))?;
            if !v.is_finite() || v < 0.0 {
                return Err(format!("multiplier {m:?}: want a finite number, 0 or more"));
            }
            p.mul = Some(v);
        }
        Ok(p)
    }
    fn factor(&self) -> f64 {
        self.mul.unwrap_or(1.0)
    }
}

/// A server-default penalty (`--default-presence-penalty SPEC`, `--default-frequency-penalty SPEC`): the parts that
/// make up a request's penalty. SPEC is one part (the base alone) or a comma list of `base=`, `thinking=`, `greedy=`,
/// `nongreedy=` and `temperature=` parts. For a token, the amounts of the parts whose condition matches add up, and the
/// product of their multipliers scales the sum: base always, greedy or nongreedy by the request's temperature, thinking
/// inside the think block. The temperature part is a last stage over that: (rest + A * T) * (B * T) with the request's
/// temperature T, the `* (B * T)` only when `*B` was written. A spec is refused when any combination, at any T in
/// [0, 2], resolves outside [-2, 2].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PenaltySpec {
    pub base: Part,
    pub thinking: Part,
    pub greedy: Part,
    pub nongreedy: Part,
    pub temperature: Part,
}

/// What a request's penalty resolves to: `out` for the whole reply, `inside` within the think block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Resolved {
    pub out: f64,
    pub inside: f64,
}

const RANGE: std::ops::RangeInclusive<f64> = -2.0..=2.0;

impl Resolved {
    fn ok(&self) -> bool {
        RANGE.contains(&self.out) && RANGE.contains(&self.inside) && RANGE.contains(&self.extra())
    }
    /// what the engine adds inside the block (its `thinking_*` field): the sum there equals `inside`
    pub fn extra(&self) -> f64 {
        self.inside - self.out
    }
}

impl PenaltySpec {
    pub fn parse(s: &str) -> Result<PenaltySpec, String> {
        let mut spec = PenaltySpec::default();
        if !s.contains('=') {
            spec.base = Part::parse(s)?;
        } else {
            let mut seen = Vec::new();
            for part in s.split(',') {
                let (key, value) = part.split_once('=').ok_or_else(|| format!("{part:?}: want key=value"))?;
                let key = key.trim();
                let slot = match key {
                    "base" => &mut spec.base,
                    "thinking" => &mut spec.thinking,
                    "greedy" => &mut spec.greedy,
                    "nongreedy" => &mut spec.nongreedy,
                    "temperature" => &mut spec.temperature,
                    _ => return Err(format!("{key:?}: want base, thinking, greedy, nongreedy or temperature")),
                };
                if seen.contains(&key) {
                    return Err(format!("{key} is given twice"));
                }
                seen.push(key);
                *slot = Part::parse(value)?;
            }
        }
        spec.check()?;
        Ok(spec)
    }

    /// Every combination within [-2, 2] at every temperature in [0, 2]. A total is at most quadratic in T, so its
    /// extremes on the interval are at its ends or its vertex.
    fn check(&self) -> Result<(), String> {
        for greedy in [true, false] {
            let (rest_out, rest_in) = self.rests(self.base.add, self.thinking.add, greedy);
            let mut ts = vec![0.0, 2.0];
            if let Some(b) = self.temperature.mul {
                // f(T) = (rest + A T) B T: the vertex at -rest / (2 A), for out, in and their difference
                for rest in [rest_out, rest_in, rest_in - rest_out] {
                    if b != 0.0 && self.temperature.add != 0.0 {
                        ts.push(-rest / (2.0 * self.temperature.add));
                    }
                }
            }
            for t in ts.into_iter().filter(|t| (0.0..=2.0).contains(t)) {
                let r = self.at(rest_out, rest_in, t);
                if !r.ok() {
                    return Err(format!(
                        "resolves to {} outside the think block and {} inside for a {} request at temperature {t}; \
                         every total and the difference must be in [-2, 2]",
                        r.out,
                        r.inside,
                        if greedy { "greedy" } else { "sampling" }
                    ));
                }
            }
        }
        Ok(())
    }

    /// The totals before the temperature stage, with `base_add` and `thinking_add` in place of the base's and the
    /// thinking part's amounts.
    fn rests(&self, base_add: f64, thinking_add: f64, greedy: bool) -> (f64, f64) {
        let gn = if greedy { self.greedy } else { self.nongreedy };
        let out = (base_add + gn.add) * (self.base.factor() * gn.factor());
        let inside = (base_add + gn.add + thinking_add) * (self.base.factor() * gn.factor() * self.thinking.factor());
        (out, inside)
    }

    /// The temperature stage at `t`.
    fn at(&self, rest_out: f64, rest_in: f64, t: f64) -> Resolved {
        let stage = |rest: f64| (rest + self.temperature.add * t) * self.temperature.mul.map_or(1.0, |b| b * t);
        Resolved { out: stage(rest_out), inside: stage(rest_in) }
    }

    /// A request's penalty at temperature `t`: its own `value` replaces the base's amount (the multiplier stays) and
    /// its own `thinking` the thinking part's amount. Err: whether the whole reply's total left [-2, 2] (else the
    /// difference inside the block did), with the totals.
    pub fn resolve(&self, value: Option<f64>, thinking: Option<f64>, t: f64) -> Result<Resolved, (bool, Resolved)> {
        let (rest_out, rest_in) =
            self.rests(value.unwrap_or(self.base.add), thinking.unwrap_or(self.thinking.add), t == 0.0);
        let r = self.at(rest_out, rest_in, t);
        if r.ok() {
            Ok(r)
        } else {
            Err((!RANGE.contains(&r.out), r))
        }
    }
}

/// The server's two penalty defaults.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Penalties {
    pub presence: PenaltySpec,
    pub frequency: PenaltySpec,
}

/// The think-end boost, a soft thinking budget (`--default-think-end-boost SPEC`, a request's `think_end_boost`):
/// inside the think block the `</think>` logit gains min(max, per1k * (n - start) / 1000) at the reply's n-th thinking
/// token. SPEC is `start=S,per1k=P,max=M`: S a token count or a fraction of the request's max_tokens written `0.5x`,
/// P logits per thousand tokens past S, M the cap (0 to 20). per1k or max 0 = off.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoostSpec {
    pub start: Start,
    pub per1k: f64,
    pub max: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Start {
    Tokens(u64),
    /// of the request's max_tokens
    Fraction(f64),
}

/// The boost as the engine takes it (PROTOCOL.md `think_end_boost`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Boost {
    pub start: u64,
    pub per1k: f64,
    pub max: f64,
}

impl BoostSpec {
    pub fn parse(s: &str) -> Result<BoostSpec, String> {
        let (mut start, mut per1k, mut max) = (None, None, None);
        for part in s.split(',') {
            let (key, value) = part.split_once('=').ok_or_else(|| format!("{part:?}: want key=value"))?;
            let (key, value) = (key.trim(), value.trim());
            let num = |t: &str, lo: f64, hi: f64| -> Result<f64, String> {
                let v: f64 = t.parse().map_err(|_| format!("{key}={t:?} is not a number"))?;
                if !v.is_finite() || !(lo..=hi).contains(&v) {
                    return Err(format!("{key}={t}: want {lo} to {hi}"));
                }
                Ok(v)
            };
            match key {
                "start" if start.is_some() => return Err("start is given twice".into()),
                "start" => {
                    start = Some(match value.strip_suffix('x') {
                        Some(f) => Start::Fraction(num(f, 0.0, 1.0)?),
                        None => Start::Tokens(
                            value
                                .parse()
                                .map_err(|_| format!("start={value:?}: want a token count or a fraction like 0.5x"))?,
                        ),
                    })
                }
                "per1k" if per1k.is_some() => return Err("per1k is given twice".into()),
                "per1k" => per1k = Some(num(value, 0.0, f64::MAX)?),
                "max" if max.is_some() => return Err("max is given twice".into()),
                "max" => max = Some(num(value, 0.0, 20.0)?),
                _ => return Err(format!("{key:?}: want start, per1k or max")),
            }
        }
        Ok(BoostSpec { start: start.unwrap_or(Start::Tokens(0)), per1k: per1k.unwrap_or(0.0), max: max.unwrap_or(0.0) })
    }

    /// The boost for a request of `max_tokens`, or None when off.
    pub fn resolve(&self, max_tokens: u64) -> Option<Boost> {
        if self.per1k <= 0.0 || self.max <= 0.0 {
            return None;
        }
        let start = match self.start {
            Start::Tokens(n) => n,
            Start::Fraction(f) => (f * max_tokens as f64).round() as u64,
        };
        Some(Boost { start, per1k: self.per1k, max: self.max })
    }
}

pub fn steering(
    body: &Map<String, Value>,
    capability: Option<&Value>,
) -> Result<(Option<String>, Option<f64>), ApiError> {
    let hp = body.get("steering_profile").is_some_and(|v| !v.is_null());
    let hs = body.get("steering_scale").is_some_and(|v| !v.is_null());
    if !hp && !hs {
        return Ok((None, None));
    }
    let profile = if hp {
        match body.get("steering_profile") {
            Some(Value::String(s))
                if !s.is_empty()
                    && s.len() <= 128
                    && s.as_bytes()[0].is_ascii_alphanumeric()
                    && s.bytes().all(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'_' | b'-')) =>
            {
                Some(s.clone())
            }
            _ => return Err(ApiError::bad("steering_profile must be a profile name", Some("steering_profile"))),
        }
    } else {
        None
    };
    let raw = if hs {
        match body.get("steering_scale") {
            Some(Value::Number(n)) => n.as_f64().unwrap_or(f64::NAN),
            _ => return Err(ApiError::bad("steering_scale must be a number", Some("steering_scale"))),
        }
    } else if hp {
        1.0
    } else {
        0.0
    };
    if !raw.is_finite() || !(0.0..=4.0).contains(&raw) {
        return Err(ApiError::bad("steering_scale must be finite and in [0, 4]", Some("steering_scale")));
    }
    let mut scale = (raw as f32) as f64;
    if scale == 0.0 {
        scale = 0.0;
    }
    if !scale.is_finite() || !(0.0..=4.0).contains(&scale) {
        return Err(ApiError::bad("steering_scale must be finite and in [0, 4]", Some("steering_scale")));
    }
    if profile.is_none() {
        if scale != 0.0 {
            return Err(ApiError::bad("steering_scale requires steering_profile", Some("steering_scale")));
        }
        if capability.and_then(|c| c.get("forced")).and_then(Value::as_bool) == Some(true) {
            return Err(ApiError::bad("request steering conflicts with forced mode", Some("steering_scale")));
        }
        return Ok((None, None));
    }
    let Some(cap) = capability.filter(|c| c.get("profile").and_then(Value::as_str).is_some()) else {
        return Err(ApiError::bad("steering is not enabled on this server", Some("steering_profile")));
    };
    // several loaded profiles (ready.s2 version 5) list every name; one profile has only "profile"
    let loaded = match cap.get("profiles").and_then(Value::as_array) {
        Some(ps) => ps.iter().any(|p| p.get("profile").and_then(Value::as_str) == profile.as_deref()),
        None => cap.get("profile").and_then(Value::as_str) == profile.as_deref(),
    };
    if !loaded {
        return Err(ApiError::bad("unknown steering_profile", Some("steering_profile")));
    }
    if capability.and_then(|c| c.get("forced")).and_then(Value::as_bool) == Some(true)
        && capability.and_then(|c| c.get("forced_scale")).and_then(Value::as_f64) != Some(scale)
    {
        return Err(ApiError::bad("request steering conflicts with forced mode", Some("steering_scale")));
    }
    Ok((profile, Some(scale)))
}

/// The request body, parsed and raw. The raw bytes are only read again for an integer that serde could not hold
/// exactly (beyond 64 bits); Python's json keeps those exact, so the range checks must too.
struct Fields<'a> {
    body: &'a Map<String, Value>,
    raw: &'a [u8],
}

/// The literal text of a top-level field, borrowed from the raw body.
fn raw_literal<'a>(raw: &'a [u8], key: &str) -> Option<&'a str> {
    let m: std::collections::HashMap<std::borrow::Cow<str>, &serde_json::value::RawValue> =
        serde_json::from_slice(raw).ok()?;
    m.get(key).map(|r| r.get())
}

/// An integer field: (value saturated to i128, Python's str() of it). None = not an integer.
fn json_int(f: &Fields, key: &str, n: &serde_json::Number) -> Option<(i128, String)> {
    if let Some(i) = n.as_i64() {
        return Some((i as i128, i.to_string()));
    }
    if let Some(u) = n.as_u64() {
        return Some((u as i128, u.to_string()));
    }
    let x = n.as_f64()?;
    if !(x.is_finite() && x == x.trunc()) {
        return None;
    }
    if let Some(lit) = raw_literal(f.raw, key).filter(|l| l.bytes().all(|c| c == b'-' || c.is_ascii_digit())) {
        // an integer literal beyond 64 bits: exact, like a Python int
        let neg = lit.starts_with('-');
        let v = lit.parse::<i128>().unwrap_or(if neg { i128::MIN } else { i128::MAX });
        return Some((v, lit.to_string()));
    }
    Some((x as i128, pyjson::integral_float_text(x))) // a float literal such as 1e20: int(float)
}

fn get<'a>(body: &'a Map<String, Value>, key: &str) -> Option<&'a Value> {
    body.get(key).filter(|v| !v.is_null())
}

/// `_num(body, key, lo, hi, int)`: exact integer bounds.
fn int_num(f: &Fields, key: &str, lo: Option<i128>, hi: Option<i128>) -> Result<Option<i128>, ApiError> {
    let Some(v) = get(f.body, key) else { return Ok(None) };
    let bad = || ApiError::bad(format!("`{key}` must be a integer"), Some(key));
    let Value::Number(n) = v else { return Err(bad()) };
    let (i, text) = json_int(f, key, n).ok_or_else(bad)?;
    if lo.is_some_and(|lo| i < lo) || hi.is_some_and(|hi| i > hi) {
        return Err(ApiError::bad(format!("`{key}` out of range: {text}"), Some(key)));
    }
    Ok(Some(i))
}

/// `_num(body, key, lo, hi, float)`.
fn float_num(body: &Map<String, Value>, key: &str, lo: f64, hi: f64) -> Result<Option<f64>, ApiError> {
    let Some(v) = get(body, key) else { return Ok(None) };
    let Value::Number(n) = v else { return Err(ApiError::bad(format!("`{key}` must be a number"), Some(key))) };
    let x = n.as_f64().unwrap_or(f64::NAN);
    if x < lo || x > hi {
        return Err(ApiError::bad(format!("`{key}` out of range: {}", pyjson::float_string(x)), Some(key)));
    }
    Ok(Some(x))
}

fn default_num(defaults: &Map<String, Value>, key: &str, neutral: f64) -> f64 {
    defaults.get(key).and_then(Value::as_f64).unwrap_or(neutral)
}

/// The least max_tokens a chat request without one gets under --share-tokens.
pub const SHARE_MIN_MAX_TOKENS: u64 = 8192;

/// max_tokens for a chat request of `prompt_len` tokens that names none, before build() caps it by the room the
/// context leaves. `default` (--default-max-tokens) wins when set. Otherwise, with `share` (--share-tokens), the most
/// that keeps prompt + max_tokens within the share limit, but at least SHARE_MIN_MAX_TOKENS: past the limit the request
/// runs alone, and a short answer budget would cost more than sharing gains. None: all the room.
pub fn chat_default_max_tokens(default: u64, share: u64, prompt_len: usize) -> Option<u64> {
    if default > 0 {
        return Some(default);
    }
    (share > 0).then(|| share.saturating_sub(prompt_len as u64).max(SHARE_MIN_MAX_TOKENS))
}

#[allow(clippy::too_many_arguments)]
pub fn build(
    body: &Map<String, Value>,
    raw: &[u8],
    prompt_len: usize,
    max_model_len: u64,
    eos_ids: &[i64],
    vocab_size: usize,
    gen_defaults: &Map<String, Value>,
    penalties: &Penalties,
    boost: Option<&BoostSpec>,
    default_max_tokens: Option<u64>,
    max_logprobs: i64,
    chat: bool,
) -> Result<Params, ApiError> {
    let f = Fields { body, raw };
    if let Some(n) = body.get("n").filter(|v| pyjson::truthy(Some(v))) {
        if !pyjson::eq_int(n, 1) {
            return Err(ApiError::bad("only n=1 is supported by the gr8 front end", Some("n")));
        }
    }
    if let Some(b) = body.get("best_of").filter(|v| !v.is_null()) {
        if !pyjson::eq_int(b, 1) {
            return Err(ApiError::bad("best_of is not supported", Some("best_of")));
        }
    }
    let mt_key = if chat && body.get("max_completion_tokens").is_some_and(|v| !v.is_null()) {
        "max_completion_tokens"
    } else {
        "max_tokens"
    };
    let max_tokens = int_num(&f, mt_key, Some(0), None)?;
    let room = max_model_len as i128 - prompt_len as i128;
    if room <= 0 || max_tokens.is_some_and(|m| m > room) {
        let mt = max_tokens.unwrap_or(0);
        let want = (prompt_len as i128).saturating_add(mt);
        return Err(ApiError::bad(
            format!(
                "This model's maximum context length is {max_model_len} tokens. However, you requested {want} tokens \
                 ({prompt_len} in the messages, {mt} in the completion). Please reduce the length of the messages or \
                 completion."
            ),
            Some("max_tokens"),
        ));
    }
    let max_tokens = match max_tokens {
        Some(m) => m,
        None => default_max_tokens.map_or(room, |d| (d as i128).min(room)),
    };
    let temperature = float_num(body, "temperature", 0.0, 2.0)?;
    let top_p = float_num(body, "top_p", 0.0, 1.0)?;
    let top_k = int_num(&f, "top_k", Some(-1), None)?;
    let min_p = float_num(body, "min_p", 0.0, 1.0)?;
    // vLLM: request value, else generation_config.json value, else neutral
    let temperature = temperature.unwrap_or_else(|| default_num(gen_defaults, "temperature", 1.0));
    let top_p = top_p.unwrap_or_else(|| default_num(gen_defaults, "top_p", 1.0));
    let top_k = top_k.unwrap_or_else(|| default_num(gen_defaults, "top_k", 0.0) as i128);
    let min_p = min_p.unwrap_or_else(|| default_num(gen_defaults, "min_p", 0.0));
    if top_p == 0.0 {
        return Err(ApiError::bad("top_p must be in (0, 1]", Some("top_p")));
    }
    let seed = int_num(&f, "seed", Some(i64::MIN as i128), Some(i64::MAX as i128))?.map(|s| s as i64);
    let min_tokens = int_num(&f, "min_tokens", Some(0), None)?.unwrap_or(0);
    if min_tokens > max_tokens {
        return Err(ApiError::bad("min_tokens must be <= max_tokens", Some("min_tokens")));
    }
    if max_tokens == 0 {
        return Err(ApiError::bad("max_tokens must be at least 1", Some("max_tokens")));
    }
    let stop: Vec<String> = match body.get("stop") {
        None | Some(Value::Null) => vec![],
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => {
            let mut v = Vec::with_capacity(a.len());
            for s in a {
                match s {
                    Value::String(s) => v.push(s.clone()),
                    _ => return Err(ApiError::bad("`stop` must be a string or a list of strings", Some("stop"))),
                }
            }
            v
        }
        Some(v) if !pyjson::truthy(Some(v)) => vec![],
        Some(_) => return Err(ApiError::bad("`stop` must be a string or a list of strings", Some("stop"))),
    };
    let stop: Vec<String> = stop.into_iter().filter(|s| !s.is_empty()).collect();
    let stop_token_ids: Vec<i64> = match body.get("stop_token_ids") {
        None | Some(Value::Null) => vec![],
        Some(v) if !pyjson::truthy(Some(v)) && !v.is_array() => vec![],
        Some(Value::Array(a)) => {
            if a.len() > MAX_STOP_TOKEN_IDS {
                return Err(ApiError::bad(
                    format!("`stop_token_ids` may name at most {MAX_STOP_TOKEN_IDS} ids"),
                    Some("stop_token_ids"),
                ));
            }
            let mut out = Vec::with_capacity(a.len());
            for x in a {
                match x {
                    // the engine bounds-checks neither stop ids nor what an overlong stop list spills into
                    Value::Number(n) if n.as_i64().is_some_and(|i| i >= 0 && (i as u64) < vocab_size as u64) => {
                        out.push(n.as_i64().unwrap())
                    }
                    Value::Number(n) if n.is_i64() => {
                        return Err(ApiError::bad(
                            "`stop_token_ids` contains a token id out of vocabulary range",
                            Some("stop_token_ids"),
                        ))
                    }
                    Value::Bool(b) => out.push(*b as i64),
                    _ => {
                        return Err(ApiError::bad(
                            "`stop_token_ids` must be a list of integers",
                            Some("stop_token_ids"),
                        ))
                    }
                }
            }
            out
        }
        Some(_) => return Err(ApiError::bad("`stop_token_ids` must be a list of integers", Some("stop_token_ids"))),
    };
    let ignore_eos = pyjson::truthy(body.get("ignore_eos"));
    let mut seen = HashSet::with_capacity(eos_ids.len() + stop_token_ids.len());
    let stop_ids: Vec<i64> = (if ignore_eos { &[][..] } else { eos_ids })
        .iter()
        .chain(stop_token_ids.iter())
        .copied()
        .filter(|i| seen.insert(*i))
        .collect();
    // OpenAI's penalties (PROTOCOL.md), and the thinking additions inside the think block the prompt leaves open
    // (api.rs sets think_end): the request's values and the server's spec resolved at its temperature
    // (PenaltySpec::resolve, where exactly 0 is greedy). The engine counts the tokens it generated, vLLM's way
    let resolve = |spec: &PenaltySpec, key: &str, tkey: &str| -> Result<Resolved, ApiError> {
        let (value, thinking) = (float_num(body, key, -2.0, 2.0)?, float_num(body, tkey, -2.0, 2.0)?);
        spec.resolve(value, thinking, temperature).map_err(|(whole, r)| {
            let k = if whole { key } else { tkey };
            ApiError::bad(
                format!(
                    "`{k}` with the server's default gives {} outside the think block and {} inside, outside [-2, 2]",
                    pyjson::float_string(r.out),
                    pyjson::float_string(r.inside)
                ),
                Some(k),
            )
        })
    };
    // the think-end boost: the request's spec, else the server's. `false` or "off" turns it off
    let think_end_boost = match body.get("think_end_boost").filter(|v| !v.is_null()) {
        None => boost.copied(),
        Some(Value::Bool(false)) => None,
        Some(Value::String(s)) if s == "off" => None,
        Some(Value::String(s)) => Some(
            BoostSpec::parse(s).map_err(|e| ApiError::bad(format!("think_end_boost: {e}"), Some("think_end_boost")))?,
        ),
        Some(_) => {
            return Err(ApiError::bad(
                "think_end_boost must be a string like \"start=0.5x,per1k=2,max=10\", \"off\" or false",
                Some("think_end_boost"),
            ))
        }
    }
    .and_then(|b| b.resolve(max_tokens as u64));
    let presence = resolve(&penalties.presence, "presence_penalty", "thinking_presence_penalty")?;
    let frequency = resolve(&penalties.frequency, "frequency_penalty", "thinking_frequency_penalty")?;
    let (presence_penalty, thinking_presence_penalty) = (presence.out, presence.extra());
    let (frequency_penalty, thinking_frequency_penalty) = (frequency.out, frequency.extra());
    if let Some(v) = body.get("repetition_penalty").filter(|v| !v.is_null()) {
        if !pyjson::eq_int(v, 1) {
            return Err(ApiError::bad(
                "repetition_penalty is not supported by the gr8 engine yet",
                Some("repetition_penalty"),
            ));
        }
    }

    // logprobs
    let lp = if chat {
        if pyjson::truthy(body.get("logprobs")) {
            let tl = int_num(&f, "top_logprobs", Some(0), None)?.unwrap_or(0);
            Some(tl)
        } else if body.get("top_logprobs").is_some_and(|v| !v.is_null()) {
            return Err(ApiError::bad(
                "when using `top_logprobs`, `logprobs` must be set to true.",
                Some("top_logprobs"),
            ));
        } else {
            None
        }
    } else {
        int_num(&f, "logprobs", Some(0), None)?
    };
    let mut plp = int_num(&f, "prompt_logprobs", Some(-1), None)?;
    if plp == Some(-1) {
        plp = Some(max_logprobs as i128);
    }
    for (name, v) in [("logprobs", lp), ("prompt_logprobs", plp)] {
        if let Some(v) = v {
            if v > max_logprobs as i128 {
                return Err(ApiError::bad(
                    format!("Requested {name} of {v}, which is greater than max allowed: {max_logprobs}"),
                    Some(name),
                ));
            }
        }
    }
    Ok(Params {
        gen: GenParams {
            max_tokens: max_tokens as u64,
            min_tokens: min_tokens as u64,
            temperature,
            top_p,
            // Python passes any size through; a top_k beyond i64 means "off" to every engine, as 0 does
            top_k: if top_k == -1 { 0 } else { i64::try_from(top_k).unwrap_or(0) },
            min_p,
            presence_penalty,
            frequency_penalty,
            thinking_presence_penalty,
            thinking_frequency_penalty,
            think_end: None,
            think_end_boost,
            seed,
            stop_ids,
            logprobs: lp.map(|v| v as i64),
            prompt_logprobs: plp.map(|v| v as i64),
            system_tokens: None,
            insert: None,
            grammar: None,
            steering_profile: None,
            steering_scale: None,
            images: Default::default(),
        },
        stop,
        include_stop: pyjson::truthy(body.get("include_stop_str_in_output")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const VOCAB: usize = 1000;

    fn stop_ids(stop_token_ids: Value) -> Result<Vec<i64>, ApiError> {
        let body = serde_json::json!({"stop_token_ids": stop_token_ids});
        let raw = serde_json::to_vec(&body).unwrap();
        let p = build(
            body.as_object().unwrap(),
            &raw,
            10,
            4096,
            &[999],
            VOCAB,
            &Map::new(),
            &Penalties::default(),
            None,
            Some(16),
            20,
            false,
        )?;
        Ok(p.gen.stop_ids)
    }

    fn rejected(stop_token_ids: Value) -> String {
        let e = stop_ids(stop_token_ids).err().expect("a 400");
        assert_eq!((e.status, e.param.as_deref()), (400, Some("stop_token_ids")));
        e.message
    }

    #[test]
    fn stop_ids_in_range_are_kept_once_after_eos() {
        assert_eq!(stop_ids(serde_json::json!([5, 999, 5, 0])).unwrap(), vec![999, 5, 0]);
    }

    #[test]
    fn a_stop_id_outside_the_vocabulary_is_refused() {
        assert!(rejected(serde_json::json!([VOCAB])).contains("out of vocabulary range"));
        assert!(rejected(serde_json::json!([-1])).contains("out of vocabulary range"));
    }

    #[test]
    fn too_many_stop_ids_are_refused() {
        assert!(stop_ids(Value::from(vec![1; MAX_STOP_TOKEN_IDS])).is_ok());
        assert!(rejected(Value::from(vec![1; MAX_STOP_TOKEN_IDS + 1])).contains("at most"));
    }

    fn max_tokens(body: &str, prompt_len: usize, max_model_len: u64, default: Option<u64>) -> u64 {
        let m: Map<String, Value> = serde_json::from_str(body).unwrap();
        let p = build(
            &m,
            body.as_bytes(),
            prompt_len,
            max_model_len,
            &[],
            VOCAB,
            &Map::new(),
            &Penalties::default(),
            None,
            default,
            20,
            true,
        );
        p.unwrap_or_else(|e| panic!("{}", e.message)).gen.max_tokens
    }

    #[test]
    fn a_chat_request_without_max_tokens_gets_the_default_capped_by_the_room() {
        assert_eq!(max_tokens("{}", 100, 393202, None), 393102); // no default: all the room
        assert_eq!(max_tokens("{}", 100, 393202, Some(16384)), 16384);
        assert_eq!(max_tokens("{}", 393000, 393202, Some(16384)), 202); // less room than the default
        assert_eq!(max_tokens(r#"{"max_tokens": 200000}"#, 100, 393202, Some(16384)), 200000);
        // named: kept
    }

    /// the four resolved penalties of a request under `spec` (presence and frequency alike), with gen_defaults `defaults`
    fn resolved(body: &str, spec: &str, defaults: &str) -> Result<(f64, f64, f64, f64), ApiError> {
        let m: Map<String, Value> = serde_json::from_str(body).unwrap();
        let d: Map<String, Value> = serde_json::from_str(defaults).unwrap();
        let one = PenaltySpec::parse(spec).unwrap();
        let pen = Penalties { presence: one, frequency: one };
        let p = build(&m, body.as_bytes(), 10, 4096, &[], VOCAB, &d, &pen, None, Some(16), 20, true)?;
        assert_eq!(p.gen.think_end, None); // api.rs sets it from the prompt
        Ok((
            p.gen.presence_penalty,
            p.gen.frequency_penalty,
            p.gen.thinking_presence_penalty,
            p.gen.thinking_frequency_penalty,
        ))
    }

    fn part(add: f64, mul: Option<f64>) -> Part {
        Part { add, mul }
    }
    fn near(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn a_penalty_spec_is_one_part_or_named_parts() {
        let p = |s| PenaltySpec::parse(s).unwrap();
        assert_eq!(p("1.5").base, part(1.5, None));
        assert_eq!(p(" -0.5 ").base, part(-0.5, None));
        assert_eq!(p("+0.5").base, part(0.5, None));
        assert_eq!(p("*1.1").base, part(0.0, Some(1.1)));
        assert_eq!(p("+0.1*1.1").base, part(0.1, Some(1.1)));
        assert_eq!(p("-0.1 * 0.5").base, part(-0.1, Some(0.5)));
        assert_eq!(p("0"), PenaltySpec::default());
        let s = p("thinking=0.5,greedy=0.2,nongreedy=0");
        assert_eq!(
            (s.base, s.thinking, s.greedy, s.nongreedy),
            (part(0.0, None), part(0.5, None), part(0.2, None), part(0.0, None))
        );
        let s = p("base=1, thinking = +0.1*1.1,greedy=*1.5 , nongreedy=-1*0, temperature = +0.1 * 0.25");
        assert_eq!((s.base, s.thinking, s.greedy), (part(1.0, None), part(0.1, Some(1.1)), part(0.0, Some(1.5))));
        assert_eq!((s.nongreedy, s.temperature), (part(-1.0, Some(0.0)), part(0.1, Some(0.25))));
        assert_eq!(p("temperature=*1").temperature, part(0.0, Some(1.0)));
        assert_eq!(p("temperature=+0.1").temperature, part(0.1, None));
        assert_eq!(p("temperature=0.1*0.5").temperature, part(0.1, Some(0.5)));
        for bad in [
            "x",
            "2.5",
            "base=3",
            "base=1,base=2",
            "warm=1",
            "base",
            "base=",
            "thinking=0.5,,greedy=1",
            "nan",
            "greedy=inf",
            "*",
            "+",
            "**1",
            "1**2",
            "base=*-1",
            "base=1*nan",
            "base=1*",
            "base=1*x",
            "temperature=",
            "temperature=*",
            // in range part by part, outside it in combination
            "base=1.5,greedy=1",
            "base=1,thinking=1.5",
            "base=1*1.5,greedy=0.5",
            "base=1,nongreedy=-3.5",
            "base=2,thinking=-2.5*1",
            "base=-1,thinking=*4",
            // outside it at some temperature in [0, 2]
            "base=1,temperature=*1.5",
            "temperature=+1.5",
            "base=1,temperature=+1",
            "base=-1,temperature=+1.5*1",
        ] {
            assert!(PenaltySpec::parse(bad).is_err(), "{bad}");
        }
        // the difference inside the block is what the engine takes, so it is bounded too
        assert!(PenaltySpec::parse("base=1,thinking=*3").is_err());
        assert!(PenaltySpec::parse("base=-1,thinking=1.5").is_ok());
        // temperature forms that stay within the range over 0..2
        for ok in [
            "base=1,temperature=*1",
            "temperature=+1",
            "base=1,temperature=-0.5*1",
            "temperature=-0.5*1",
            "base=0.5,temperature=+0.5*0.5",
        ] {
            assert!(PenaltySpec::parse(ok).is_ok(), "{ok}");
        }
        // a vertex inside the interval: (-1 + 0.75 T) T dips to -1/3 at T = 2/3 and reaches 1 at T = 2. With 1.5 the
        // top at T = 2 is 4
        assert!(PenaltySpec::parse("base=-1,temperature=+1.5*1").is_err()); // 4 at T = 2
        assert!(PenaltySpec::parse("base=-1,temperature=+0.75*1").is_ok());
    }

    #[test]
    fn parts_add_and_their_multipliers_compound() {
        let s = PenaltySpec::parse("base=0.2*1.5,thinking=+0.1*1.1,greedy=0.3*2,nongreedy=-0.1").unwrap();
        // greedy: out = (0.2 + 0.3) * (1.5 * 2) = 1.5; in = (0.2 + 0.3 + 0.1) * (1.5 * 2 * 1.1) = 1.98
        let r = s.resolve(None, None, 0.0).unwrap();
        assert!(near(r.out, 1.5) && near(r.inside, 1.98) && near(r.extra(), 0.48));
        // sampling: out = (0.2 - 0.1) * 1.5 = 0.15; in = (0.2 - 0.1 + 0.1) * (1.5 * 1.1) = 0.33
        let r = s.resolve(None, None, 1.0).unwrap();
        assert!(near(r.out, 0.15) && near(r.inside, 0.33));
        // the request's value replaces the base's amount, not its multiplier. The same for thinking
        let r = s.resolve(Some(0.0), None, 0.0).unwrap();
        assert!(near(r.out, 0.9) && near(r.inside, 1.32));
        let r = s.resolve(None, Some(0.0), 0.0).unwrap();
        assert!(near(r.out, 1.5) && near(r.inside, 1.65));
        // an override that leaves the range names the total that did
        assert!(s.resolve(Some(1.0), None, 0.0).unwrap_err().0); // out = 3.9
        let s = PenaltySpec::parse("thinking=*1.5").unwrap();
        assert!(!s.resolve(Some(2.0), None, 1.0).unwrap_err().0); // out 2, inside 3
        assert_eq!(
            PenaltySpec::default().resolve(Some(2.0), Some(-2.0), 1.0).unwrap(),
            Resolved { out: 2.0, inside: 0.0 }
        );
    }

    #[test]
    fn the_temperature_part_is_a_last_stage_scaled_by_the_request_s_temperature() {
        // rest: out = (0.5 + gn) * 1, in = rest_out + 0.25 (thinking); then (rest + 0.1 T) * (0.5 T)
        let s = PenaltySpec::parse("base=0.5,thinking=0.25,greedy=0.25,nongreedy=-0.25,temperature=+0.1*0.5").unwrap();
        let r = s.resolve(None, None, 0.0).unwrap();
        assert_eq!(r, Resolved { out: 0.0, inside: 0.0 }); // the factor B T is 0 under greedy
        for (t, out, inside) in
            [(0.7, (0.25 + 0.07) * 0.35, (0.5 + 0.07) * 0.35), (1.0, 0.35 * 0.5, 0.6 * 0.5), (2.0, 0.45, 0.7)]
        {
            let r = s.resolve(None, None, t).unwrap();
            assert!(near(r.out, out) && near(r.inside, inside), "T {t}: {r:?}");
        }
        // an amount alone adds A T with no factor, and a factor alone scales by B T
        let s = PenaltySpec::parse("base=0.5,temperature=+0.25").unwrap();
        assert_eq!(s.resolve(None, None, 0.0).unwrap().out, 0.5);
        assert_eq!(s.resolve(None, None, 2.0).unwrap().out, 1.0);
        let s = PenaltySpec::parse("base=0.5,thinking=0.5,temperature=*1").unwrap();
        assert_eq!(s.resolve(None, None, 0.0).unwrap(), Resolved { out: 0.0, inside: 0.0 });
        assert_eq!(s.resolve(None, None, 1.0).unwrap(), Resolved { out: 0.5, inside: 1.0 });
        assert_eq!(s.resolve(None, None, 2.0).unwrap(), Resolved { out: 1.0, inside: 2.0 });
        // the request's value goes through the stage too, and the stage's amount is not scaled by the other multipliers
        let s = PenaltySpec::parse("base=*2,temperature=+0.5*0.5").unwrap();
        let r = s.resolve(Some(0.5), None, 1.0).unwrap();
        assert_eq!(r.out, (1.0 + 0.5) * 0.5);
        assert!(s.resolve(Some(2.0), None, 2.0).unwrap_err().0); // (4 + 1) * 1
    }

    #[test]
    fn a_request_s_value_replaces_the_base_and_the_temperature_picks_the_rest() {
        let spec = "base=0.25,thinking=0.5,greedy=0.25,nongreedy=-0.125";
        // nothing named: greedy (temperature 0 after defaults) adds greedy, a sampling request adds nongreedy. The
        // engine's thinking field is what the block adds on top
        assert_eq!(resolved(r#"{"temperature": 0}"#, spec, "{}").unwrap(), (0.5, 0.5, 0.5, 0.5));
        assert_eq!(resolved("{}", spec, r#"{"temperature": 0.0}"#).unwrap(), (0.5, 0.5, 0.5, 0.5));
        assert_eq!(resolved("{}", spec, "{}").unwrap(), (0.125, 0.125, 0.5, 0.5)); // vLLM's default temperature 1.0
        assert_eq!(resolved(r#"{"temperature": 1e-6}"#, spec, "{}").unwrap().0, 0.125); // only exactly 0 is greedy
                                                                                        // the request's own value replaces the base, 0 included, and the condition's part still adds
        assert_eq!(resolved(r#"{"temperature": 0, "presence_penalty": 1.0}"#, spec, "{}").unwrap().0, 1.25);
        assert_eq!(resolved(r#"{"temperature": 0, "presence_penalty": 0}"#, spec, "{}").unwrap().0, 0.25);
        assert_eq!(resolved(r#"{"temperature": 1, "frequency_penalty": 0}"#, spec, "{}").unwrap().1, -0.125);
        // the request's thinking value replaces the thinking part alone
        let r = resolved(r#"{"temperature": 0, "thinking_presence_penalty": 1.5}"#, spec, "{}").unwrap();
        assert_eq!((r.0, r.2, r.3), (0.5, 1.5, 0.5));
        assert_eq!(resolved(r#"{"thinking_frequency_penalty": 0}"#, spec, "{}").unwrap().3, 0.0);
        // a multiplier scales the request's value too, so the engine's fields carry the resolved totals
        let r = resolved(r#"{"temperature": 0, "presence_penalty": 1}"#, "base=*1.5,thinking=*1.2", "{}").unwrap();
        assert!(near(r.0, 1.5) && near(r.2, 0.3));
        // no spec: the request's values alone, else off
        assert_eq!(resolved("{}", "0", "{}").unwrap(), (0.0, 0.0, 0.0, 0.0));
        assert_eq!(
            resolved(r#"{"presence_penalty": 1.5, "frequency_penalty": -0.5}"#, "0", "{}").unwrap(),
            (1.5, -0.5, 0.0, 0.0)
        );
        assert_eq!(resolved(r#"{"presence_penalty": 2, "frequency_penalty": null}"#, "0", "{}").unwrap().0, 2.0);
        for (body, spec, param) in [
            (r#"{"presence_penalty": 2.5}"#, "0", "presence_penalty"),
            (r#"{"frequency_penalty": -2.5}"#, "0", "frequency_penalty"),
            (r#"{"presence_penalty": "1"}"#, "0", "presence_penalty"),
            (r#"{"thinking_presence_penalty": 2.5}"#, "0", "thinking_presence_penalty"),
            (r#"{"thinking_frequency_penalty": "x"}"#, "0", "thinking_frequency_penalty"),
            (r#"{"repetition_penalty": 1.1}"#, "0", "repetition_penalty"),
            // in range alone, outside it with the server's parts
            (r#"{"temperature": 0, "presence_penalty": 2}"#, "greedy=0.5", "presence_penalty"),
            (r#"{"thinking_presence_penalty": 1.5}"#, "base=1", "thinking_presence_penalty"),
        ] {
            let e = resolved(body, spec, "{}").err().expect(body);
            assert_eq!((e.status, e.param.as_deref()), (400, Some(param)), "{body}");
        }
        assert!(resolved(r#"{"repetition_penalty": 1}"#, spec, "{}").is_ok());
    }

    fn boosted(body: &str, spec: Option<&str>, default_max: Option<u64>) -> Result<Option<Boost>, ApiError> {
        let m: Map<String, Value> = serde_json::from_str(body).unwrap();
        let spec = spec.map(|s| BoostSpec::parse(s).unwrap());
        let p = build(
            &m,
            body.as_bytes(),
            10,
            40000,
            &[],
            VOCAB,
            &Map::new(),
            &Penalties::default(),
            spec.as_ref(),
            default_max,
            20,
            true,
        )?;
        Ok(p.gen.think_end_boost)
    }

    #[test]
    fn a_boost_spec_names_start_per1k_and_max() {
        let s = BoostSpec::parse("start=0.5x,per1k=2,max=10").unwrap();
        assert_eq!(s, BoostSpec { start: Start::Fraction(0.5), per1k: 2.0, max: 10.0 });
        assert_eq!(
            BoostSpec::parse(" start = 4096 , per1k=0.5,max=1.5").unwrap(),
            BoostSpec { start: Start::Tokens(4096), per1k: 0.5, max: 1.5 }
        );
        assert_eq!(BoostSpec::parse("per1k=1,max=1").unwrap().start, Start::Tokens(0));
        assert_eq!(BoostSpec::parse("start=1x,per1k=1,max=20").unwrap().start, Start::Fraction(1.0));
        for bad in [
            "",
            "x",
            "start=-1,per1k=1,max=1",
            "start=1.5x,per1k=1,max=1",
            "start=1,per1k=-1,max=1",
            "start=1,per1k=1,max=21",
            "start=1,per1k=1,max=1,max=2",
            "start=1,ramp=1,max=1",
            "start=abc,per1k=1,max=1",
        ] {
            assert!(BoostSpec::parse(bad).is_err(), "{bad}");
        }
        // off when per1k or max is 0 or absent
        assert_eq!(BoostSpec::parse("start=1,per1k=0,max=10").unwrap().resolve(100), None);
        assert_eq!(BoostSpec::parse("start=1,per1k=1,max=0").unwrap().resolve(100), None);
        assert_eq!(BoostSpec::parse("start=0.5x").unwrap().resolve(100), None);
    }

    #[test]
    fn the_boost_s_start_resolves_against_the_request_s_max_tokens() {
        let spec = Some("start=0.5x,per1k=2,max=10");
        // the request's max_tokens, the default when it names none, else the room the context leaves
        assert_eq!(
            boosted(r#"{"max_tokens": 8192}"#, spec, None).unwrap(),
            Some(Boost { start: 4096, per1k: 2.0, max: 10.0 })
        );
        assert_eq!(boosted("{}", spec, Some(16384)).unwrap().unwrap().start, 8192);
        assert_eq!(boosted("{}", spec, None).unwrap().unwrap().start, 19995); // half of the room, 40000 - 10
        assert_eq!(
            boosted(r#"{"max_tokens": 100}"#, Some("start=4096,per1k=2,max=10"), None).unwrap().unwrap().start,
            4096
        );
        // no default: off. The request's own spec, or off, wins over the default
        assert_eq!(boosted(r#"{"max_tokens": 100}"#, None, None).unwrap(), None);
        assert_eq!(
            boosted(r#"{"max_tokens": 1000, "think_end_boost": "start=0.1x,per1k=1,max=2"}"#, None, None).unwrap(),
            Some(Boost { start: 100, per1k: 1.0, max: 2.0 })
        );
        assert_eq!(boosted(r#"{"max_tokens": 100, "think_end_boost": "off"}"#, spec, None).unwrap(), None);
        assert_eq!(boosted(r#"{"max_tokens": 100, "think_end_boost": false}"#, spec, None).unwrap(), None);
        assert_eq!(boosted(r#"{"max_tokens": 100, "think_end_boost": null}"#, spec, None).unwrap().unwrap().start, 50);
        for body in [
            r#"{"think_end_boost": "start=1,per1k=1,max=30"}"#,
            r#"{"think_end_boost": 1}"#,
            r#"{"think_end_boost": {"start": 1}}"#,
        ] {
            let e = boosted(body, spec, None).err().expect(body);
            assert_eq!((e.status, e.param.as_deref()), (400, Some("think_end_boost")));
        }
    }

    #[test]
    fn without_a_default_a_chat_request_fits_the_share_limit_when_it_can() {
        assert_eq!(chat_default_max_tokens(0, 65472, 49439), Some(16033));
        assert_eq!(chat_default_max_tokens(0, 65472, 60000), Some(SHARE_MIN_MAX_TOKENS)); // past the limit anyway
        assert_eq!(chat_default_max_tokens(0, 65472, 100000), Some(SHARE_MIN_MAX_TOKENS));
        assert_eq!(chat_default_max_tokens(4096, 65472, 100), Some(4096)); // --default-max-tokens wins
        assert_eq!(chat_default_max_tokens(0, 0, 100), None); // no share limit: all the room
    }

    #[test]
    fn steering_is_canonical_and_checked_against_hello() {
        let cap = serde_json::json!({"profile":"example","forced":false,"forced_scale":0.0});
        let body = serde_json::json!({"steering_profile":"example","steering_scale":1.1});
        let (p, s) = steering(body.as_object().unwrap(), Some(&cap)).unwrap();
        assert_eq!(p.as_deref(), Some("example"));
        assert_eq!(s, Some((1.1f32) as f64));
        let zero = serde_json::json!({"steering_scale":-0.0});
        assert_eq!(steering(zero.as_object().unwrap(), Some(&cap)).unwrap(), (None, None));
        let other = serde_json::json!({"steering_profile":"other"});
        assert_eq!(
            steering(other.as_object().unwrap(), Some(&cap)).unwrap_err().param.as_deref(),
            Some("steering_profile")
        );
        let malformed = serde_json::json!({"steering_profile":"bad/name"});
        assert_eq!(
            steering(malformed.as_object().unwrap(), Some(&cap)).unwrap_err().message,
            "steering_profile must be a profile name"
        );
        let disabled = steering(body.as_object().unwrap(), None).unwrap_err();
        assert_eq!(disabled.param.as_deref(), Some("steering_profile"));
        assert_eq!(disabled.message, "steering is not enabled on this server");
        let forced = serde_json::json!({"profile":"example","forced":true,"forced_scale":2.0});
        assert!(steering(body.as_object().unwrap(), Some(&forced)).is_err());
        let two = serde_json::json!({"profile":"example","profiles":[{"profile":"example"},{"profile":"zh-en"}]});
        let second = serde_json::json!({"steering_profile":"zh-en"});
        assert_eq!(steering(second.as_object().unwrap(), Some(&two)).unwrap(), (Some("zh-en".into()), Some(1.0)));
        assert!(steering(other.as_object().unwrap(), Some(&two)).is_err());
        let default = serde_json::json!({"steering_profile":"example"});
        assert_eq!(steering(default.as_object().unwrap(), Some(&cap)).unwrap(), (Some("example".into()), Some(1.0)));
        let nulls = serde_json::json!({"steering_profile":null,"steering_scale":null});
        assert_eq!(steering(nulls.as_object().unwrap(), Some(&cap)).unwrap(), (None, None));
        let null_scale = serde_json::json!({"steering_profile":"example","steering_scale":null});
        assert_eq!(steering(null_scale.as_object().unwrap(), Some(&cap)).unwrap().1, Some(1.0));
        let named_zero = serde_json::json!({"steering_profile":"example","steering_scale":-0.0});
        assert_eq!(steering(named_zero.as_object().unwrap(), Some(&cap)).unwrap().1, Some(0.0));
        assert!(steering(named_zero.as_object().unwrap(), None).is_err());
        assert!(steering(named_zero.as_object().unwrap(), Some(&forced)).is_err());
        let none = serde_json::json!({});
        assert_eq!(steering(none.as_object().unwrap(), Some(&forced)).unwrap(), (None, None));
        let matching = serde_json::json!({"steering_profile":"example","steering_scale":2});
        assert_eq!(steering(matching.as_object().unwrap(), Some(&forced)).unwrap().1, Some(2.0));
        for (body, param) in [
            (serde_json::json!({"steering_profile":7}), "steering_profile"),
            (serde_json::json!({"steering_scale":1}), "steering_scale"),
            (serde_json::json!({"steering_profile":"example","steering_scale":4.1}), "steering_scale"),
        ] {
            assert_eq!(steering(body.as_object().unwrap(), Some(&cap)).unwrap_err().param.as_deref(), Some(param));
        }
    }

    #[test]
    fn steering_matches_shared_host_contract() {
        let cases: Value =
            serde_json::from_str(include_str!("../../../engine/runtime/tests/fixtures/steering_contract.json"))
                .unwrap();
        for case in cases["profile_names"].as_array().unwrap() {
            let body = serde_json::json!({"steering_profile":case["input"]});
            let cap = serde_json::json!({"profile":case["input"]});
            assert_eq!(
                steering(body.as_object().unwrap(), Some(&cap)).is_ok(),
                case["valid"].as_bool().unwrap(),
                "{case}"
            );
        }
        let cap = serde_json::json!({"profile":"example"});
        for case in cases["scales"].as_array().unwrap() {
            let raw = case["input"].as_str().unwrap().parse::<f64>().unwrap();
            let value = serde_json::Number::from_f64(raw).map(Value::Number).unwrap_or(Value::Null);
            // Non-finite numbers are not JSON numbers; a string exercises the API's type rejection.
            let value = if value.is_null() { case["input"].clone() } else { value };
            let body = serde_json::json!({"steering_profile":"example","steering_scale":value});
            let result = steering(body.as_object().unwrap(), Some(&cap));
            assert_eq!(result.is_ok(), case["valid"].as_bool().unwrap(), "{case}");
            if let Ok((_, Some(scale))) = result {
                let bits = u32::from_str_radix(case["bits"].as_str().unwrap(), 16).unwrap();
                assert_eq!((scale as f32).to_bits(), bits, "{case}");
            }
        }
    }
}
