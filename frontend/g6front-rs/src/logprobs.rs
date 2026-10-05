//! Engine logprob entries ({lp, rank, top:[[id, lp], ...]}) -> vLLM's response shapes, written straight as JSON
//! (a port of `g6front/logprobs.py`). `prompt_logprobs` is transcoded from the raw engine frame one position at a
//! time, so a 200K-position result never exists as a parsed tree.

use std::fmt;

use serde::de::{self, DeserializeSeed, IgnoredAny, MapAccess, SeqAccess, Visitor};

use crate::pyjson;
use crate::tok::Tok;
use crate::wire::LpEntry;

const FLOOR: f64 = -9999.0;

/// Python `max(float(x), -9999.0)`.
#[inline]
fn lp(x: f64) -> f64 {
    if x.is_nan() || x >= FLOOR {
        x
    } else {
        FLOOR
    }
}

fn tokstr_owned(t: &Tok, id: u32, as_ids: bool) -> String {
    if as_ids {
        format!("token_id:{id}")
    } else {
        t.token_str(id).into_owned()
    }
}

/// vLLM's per-position dict: the actual/sampled token first (with its rank), then the top-k (ranks 1..k).
fn step_dict(tid: u32, e: &LpEntry) -> Vec<(u32, f64, Option<i64>)> {
    let top = e.top.as_deref().unwrap_or(&[]);
    let mut d = Vec::with_capacity(top.len() + 1);
    d.push((tid, lp(e.lp), e.rank));
    for (r, &(i, v)) in top.iter().enumerate() {
        if !d.iter().any(|x| x.0 == i) {
            d.push((i, lp(v), Some(r as i64 + 1)));
        }
    }
    d
}

fn opt_int(out: &mut Vec<u8>, v: Option<i64>) {
    match v {
        Some(i) => pyjson::int(out, i),
        None => out.extend_from_slice(b"null"),
    }
}

fn prompt_entry(out: &mut Vec<u8>, t: &Tok, tid: u32, e: &LpEntry) {
    out.push(b'{');
    for (n, (i, v, rk)) in step_dict(tid, e).into_iter().enumerate() {
        if n > 0 {
            out.push(b',');
        }
        out.push(b'"');
        pyjson::int(out, i);
        out.extend_from_slice(b"\":{\"logprob\":");
        pyjson::float(out, v);
        out.extend_from_slice(b",\"rank\":");
        opt_int(out, rk);
        out.extend_from_slice(b",\"decoded_token\":");
        pyjson::string(out, &t.token_str(i));
        out.push(b'}');
    }
    out.push(b'}');
}

/// `prompt_logprobs`: `[null, {"<id>": {"logprob", "rank", "decoded_token"}, ...}, ...]` from the raw frame body.
pub fn prompt_logprobs(out: &mut Vec<u8>, frame: Option<&[u8]>, prompt: &[u32], t: &Tok) -> Result<(), String> {
    let Some(frame) = frame else {
        out.extend_from_slice(b"null");
        return Ok(());
    };
    out.reserve(frame.len() * 2);
    let mut de = serde_json::Deserializer::from_slice(frame);
    FrameSeed { out, prompt, t }.deserialize(&mut de).map_err(|e| format!("bad prompt_logprobs frame: {e}"))?;
    Ok(())
}

struct FrameSeed<'a> {
    out: &'a mut Vec<u8>,
    prompt: &'a [u32],
    t: &'a Tok,
}

impl<'de> DeserializeSeed<'de> for FrameSeed<'_> {
    type Value = ();
    fn deserialize<D: de::Deserializer<'de>>(self, d: D) -> Result<(), D::Error> {
        d.deserialize_map(self)
    }
}

impl<'de> Visitor<'de> for FrameSeed<'_> {
    type Value = ();
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a prompt_logprobs frame")
    }
    fn visit_map<A: MapAccess<'de>>(self, mut m: A) -> Result<(), A::Error> {
        let mut seen = false;
        while let Some(k) = m.next_key::<std::borrow::Cow<'de, str>>()? {
            if k == "lps" && !seen {
                seen = true;
                m.next_value_seed(LpsSeed { out: &mut *self.out, prompt: self.prompt, t: self.t })?;
            } else {
                m.next_value::<IgnoredAny>()?;
            }
        }
        if !seen {
            self.out.extend_from_slice(b"null");
        }
        Ok(())
    }
}

struct LpsSeed<'a> {
    out: &'a mut Vec<u8>,
    prompt: &'a [u32],
    t: &'a Tok,
}

impl<'de> DeserializeSeed<'de> for LpsSeed<'_> {
    type Value = ();
    fn deserialize<D: de::Deserializer<'de>>(self, d: D) -> Result<(), D::Error> {
        d.deserialize_seq(self)
    }
}

impl<'de> Visitor<'de> for LpsSeed<'_> {
    type Value = ();
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a list of prompt logprob entries")
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut s: A) -> Result<(), A::Error> {
        self.out.push(b'[');
        let mut j = 0usize;
        while let Some(e) = s.next_element::<Option<LpEntry>>()? {
            if j > 0 {
                self.out.push(b',');
            }
            match e {
                None => self.out.extend_from_slice(b"null"),
                Some(e) => {
                    let Some(&tid) = self.prompt.get(j) else {
                        return Err(de::Error::custom("more prompt logprob entries than prompt tokens"));
                    };
                    prompt_entry(self.out, self.t, tid, &e);
                }
            }
            j += 1;
        }
        self.out.push(b']');
        Ok(())
    }
}

/// `/v1/completions` `logprobs`: {text_offset, token_logprobs, tokens, top_logprobs (sampled + top-k)}.
/// Returns the text offset after the last token (the next chunk's starting offset).
pub fn completion_logprobs(
    out: &mut Vec<u8>,
    ids: &[u32],
    entries: &[LpEntry],
    k: i64,
    t: &Tok,
    as_ids: bool,
    offset0: usize,
) -> usize {
    let n = ids.len().min(entries.len());
    let strs: Vec<String> = ids[..n].iter().map(|&i| tokstr_owned(t, i, as_ids)).collect();
    out.extend_from_slice(b"{\"text_offset\":[");
    let mut off = offset0;
    for (j, s) in strs.iter().enumerate() {
        if j > 0 {
            out.push(b',');
        }
        pyjson::int(out, off);
        off += s.chars().count();
    }
    out.extend_from_slice(b"],\"token_logprobs\":[");
    for (j, e) in entries[..n].iter().enumerate() {
        if j > 0 {
            out.push(b',');
        }
        pyjson::float(out, lp(e.lp));
    }
    out.extend_from_slice(b"],\"tokens\":[");
    for (j, s) in strs.iter().enumerate() {
        if j > 0 {
            out.push(b',');
        }
        pyjson::string(out, s);
    }
    out.extend_from_slice(b"],\"top_logprobs\":[");
    for j in 0..n {
        if j > 0 {
            out.push(b',');
        }
        // dict keyed by token string: a repeated string keeps its first position and takes the later value
        let mut top: Vec<(String, f64)> = Vec::with_capacity((k.max(0) as usize).min(64) + 1);
        for (m, (i, v, _)) in step_dict(ids[j], &entries[j]).into_iter().enumerate() {
            if m as i64 > k {
                break;
            }
            let s = tokstr_owned(t, i, as_ids);
            match top.iter_mut().find(|x| x.0 == s) {
                Some(x) => x.1 = v,
                None => top.push((s, v)),
            }
        }
        out.push(b'{');
        for (m, (s, v)) in top.iter().enumerate() {
            if m > 0 {
                out.push(b',');
            }
            pyjson::string(out, s);
            out.push(b':');
            pyjson::float(out, *v);
        }
        out.push(b'}');
    }
    out.extend_from_slice(b"]}");
    off
}

fn bytes_list(out: &mut Vec<u8>, s: &[u8]) {
    out.push(b'[');
    for (i, b) in s.iter().enumerate() {
        if i > 0 {
            out.push(b',');
        }
        pyjson::int(out, *b);
    }
    out.push(b']');
}

/// `/v1/chat/completions` `logprobs`: {content: [{token, logprob, bytes, top_logprobs: [...k]}]}.
pub fn chat_logprobs(out: &mut Vec<u8>, ids: &[u32], entries: &[LpEntry], k: i64, t: &Tok, as_ids: bool) {
    out.extend_from_slice(b"{\"content\":[");
    for (j, (&tid, e)) in ids.iter().zip(entries).enumerate() {
        if j > 0 {
            out.push(b',');
        }
        let s = tokstr_owned(t, tid, as_ids);
        out.extend_from_slice(b"{\"token\":");
        pyjson::string(out, &s);
        out.extend_from_slice(b",\"logprob\":");
        pyjson::float(out, lp(e.lp));
        out.extend_from_slice(b",\"bytes\":");
        bytes_list(out, s.as_bytes());
        out.extend_from_slice(b",\"top_logprobs\":[");
        let top = e.top.as_deref().unwrap_or(&[]);
        for (m, &(i, v)) in top.iter().take(k.max(0) as usize).enumerate() {
            if m > 0 {
                out.push(b',');
            }
            let ts = tokstr_owned(t, i, as_ids);
            out.extend_from_slice(b"{\"token\":");
            pyjson::string(out, &ts);
            out.extend_from_slice(b",\"logprob\":");
            pyjson::float(out, lp(v));
            out.extend_from_slice(b",\"bytes\":");
            bytes_list(out, ts.as_bytes());
            out.push(b'}');
        }
        out.extend_from_slice(b"]}");
    }
    out.extend_from_slice(b"]}");
}
