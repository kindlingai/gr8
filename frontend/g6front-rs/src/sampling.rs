//! OpenAI/vLLM request fields -> PROTOCOL.md `generate` fields, with vLLM's defaults and validation
//! (a port of `g6front/sampling.py`; same checks in the same order, same messages).

use serde_json::{Map, Value};

use crate::api::ApiError;
use crate::pyjson;
use crate::wire::GenParams;

pub struct Params {
    pub gen: GenParams,
    pub stop: Vec<String>,
    pub include_stop: bool,
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

#[allow(clippy::too_many_arguments)]
pub fn build(
    body: &Map<String, Value>,
    raw: &[u8],
    prompt_len: usize,
    max_model_len: u64,
    eos_ids: &[i64],
    gen_defaults: &Map<String, Value>,
    default_max_tokens: Option<u64>,
    max_logprobs: i64,
    chat: bool,
) -> Result<Params, ApiError> {
    let f = Fields { body, raw };
    if let Some(n) = body.get("n").filter(|v| pyjson::truthy(Some(v))) {
        if !pyjson::eq_int(n, 1) {
            return Err(ApiError::bad("only n=1 is supported by the g6 front end", Some("n")));
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
            let mut out = Vec::with_capacity(a.len());
            for x in a {
                match x {
                    Value::Number(n) if n.is_i64() => out.push(n.as_i64().unwrap()),
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
    let mut stop_ids: Vec<i64> = Vec::with_capacity(eos_ids.len() + stop_token_ids.len());
    for &i in (if ignore_eos { &[][..] } else { eos_ids }).iter().chain(stop_token_ids.iter()) {
        if !stop_ids.contains(&i) {
            stop_ids.push(i);
        }
    }
    for k in ["frequency_penalty", "presence_penalty"] {
        if let Some(v) = body.get(k).filter(|v| !v.is_null()) {
            if !pyjson::eq_int(v, 0) {
                return Err(ApiError::bad(format!("{k} is not supported by the g6 engine yet"), Some(k)));
            }
        }
    }
    if let Some(v) = body.get("repetition_penalty").filter(|v| !v.is_null()) {
        if !pyjson::eq_int(v, 1) {
            return Err(ApiError::bad(
                "repetition_penalty is not supported by the g6 engine yet",
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
            seed,
            stop_ids,
            logprobs: lp.map(|v| v as i64),
            prompt_logprobs: plp.map(|v| v as i64),
        },
        stop,
        include_stop: pyjson::truthy(body.get("include_stop_str_in_output")),
    })
}
