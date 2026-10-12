//! PROTOCOL.md framing: u32 big-endian length + one UTF-8 JSON object.
//!
//! Reading: frames are cut out of one growing `BytesMut` as refcounted `Bytes` slices (no copy). The reader only
//! looks at `type`/`id`/`seq` (borrowed, no allocation) to route a frame; the request that owns it parses the rest
//! straight from the same bytes.
//! Writing: a `generate` frame (thousands of prompt ids) is written with itoa into one buffer whose first four bytes
//! are patched with the length afterwards.

use std::borrow::Cow;

use bytes::{Bytes, BytesMut};
use serde::Deserialize;
use tokio::io::{AsyncRead, AsyncReadExt};

use crate::pyjson;

pub const MAX_FRAME: usize = 256 << 20;

#[derive(Debug)]
pub struct FrameError(pub String);

impl std::fmt::Display for FrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Reads frames off a stream; each returned `Bytes` is the JSON body (length prefix stripped).
pub struct FrameReader<R> {
    r: R,
    buf: BytesMut,
}

impl<R: AsyncRead + Unpin> FrameReader<R> {
    pub fn new(r: R) -> Self {
        FrameReader { r, buf: BytesMut::with_capacity(64 << 10) }
    }

    /// Next frame body, or None on a clean EOF at a frame boundary.
    pub async fn next(&mut self) -> Result<Option<Bytes>, FrameError> {
        loop {
            if self.buf.len() >= 4 {
                let n = u32::from_be_bytes(self.buf[..4].try_into().unwrap()) as usize;
                if n > MAX_FRAME {
                    return Err(FrameError(format!("frame too large: {n} bytes")));
                }
                if self.buf.len() >= 4 + n {
                    let mut f = self.buf.split_to(4 + n);
                    let _ = f.split_to(4);
                    return Ok(Some(f.freeze()));
                }
                // one reservation for a large frame (prompt_logprobs can be ~100 MB) instead of doubling into it
                self.buf.reserve(4 + n - self.buf.len());
            } else if self.buf.capacity() - self.buf.len() < 4096 {
                self.buf.reserve(64 << 10);
            }
            let got = self.r.read_buf(&mut self.buf).await.map_err(|e| FrameError(format!("read: {e}")))?;
            if got == 0 {
                return if self.buf.is_empty() { Ok(None) } else { Err(FrameError("EOF inside frame".into())) };
            }
        }
    }
}

/// The routing fields of any frame. Other fields are skipped without allocating; an `id` that is not a string or a
/// `seq` that is not an unsigned integer routes nowhere (the frame is dropped) instead of failing the connection.
#[derive(Deserialize)]
struct RawHead<'a> {
    #[serde(rename = "type", borrow)]
    kind: Cow<'a, str>,
    #[serde(default, borrow)]
    id: Option<Loose<'a>>,
    #[serde(default)]
    seq: Option<LooseU64>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Loose<'a> {
    S(#[serde(borrow)] Cow<'a, str>),
    Other(serde::de::IgnoredAny),
}

#[derive(Deserialize)]
#[serde(untagged)]
enum LooseU64 {
    U(u64),
    Other(serde::de::IgnoredAny),
}

pub struct Head<'a> {
    pub kind: Cow<'a, str>,
    pub id: Option<Cow<'a, str>>,
    pub seq: Option<u64>,
}

pub fn head(body: &[u8]) -> Result<Head<'_>, FrameError> {
    let h: RawHead = serde_json::from_slice(body).map_err(|e| FrameError(format!("bad frame: {e}")))?;
    Ok(Head {
        kind: h.kind,
        id: match h.id {
            Some(Loose::S(s)) => Some(s),
            _ => None,
        },
        seq: match h.seq {
            Some(LooseU64::U(u)) => Some(u),
            _ => None,
        },
    })
}

#[derive(Deserialize, Clone, Debug)]
pub struct LpEntry {
    pub lp: f64,
    #[serde(default)]
    pub rank: Option<i64>,
    #[serde(default)]
    pub top: Option<Vec<(u32, f64)>>,
}

#[derive(Deserialize)]
pub struct Tokens {
    pub ids: Vec<u32>,
    #[serde(default)]
    pub logprobs: Option<Vec<LpEntry>>,
}

#[derive(Deserialize)]
pub struct Finish {
    pub reason: String,
    #[serde(default)]
    pub stop_id: Option<i64>,
    #[serde(default)]
    pub cached_tokens: Option<u64>,
    #[serde(default)]
    pub message: Option<String>,
}

/// One frame with a fixed shape; the length prefix is patched in `finish()`.
pub struct FrameWriter {
    pub b: Vec<u8>,
}

impl FrameWriter {
    pub fn with_capacity(n: usize) -> Self {
        let mut b = Vec::with_capacity(n + 4);
        b.extend_from_slice(&[0, 0, 0, 0]);
        FrameWriter { b }
    }

    pub fn finish(mut self) -> Bytes {
        let n = (self.b.len() - 4) as u32;
        self.b[..4].copy_from_slice(&n.to_be_bytes());
        Bytes::from(self.b)
    }
}

/// `generate` parameters (PROTOCOL.md).
#[derive(Clone, Debug)]
pub struct GenParams {
    pub max_tokens: u64,
    pub min_tokens: u64,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: i64,
    pub min_p: f64,
    pub seed: Option<i64>,
    pub stop_ids: Vec<i64>,
    pub logprobs: Option<i64>,
    pub prompt_logprobs: Option<i64>,
    /// the prompt's system span in tokens (PROTOCOL.md): chat requests only
    pub system_tokens: Option<u64>,
    /// the engine's insert rule (PROTOCOL.md `insert`); only sent when set
    pub insert: Option<InsertRule>,
    /// Only sent after validation against hello.steering. None preserves stock frame bytes.
    pub steering_profile: Option<String>,
    /// Canonical FP32 widened to f64 for lossless JSON serialization.
    pub steering_scale: Option<f64>,
}

/// Once the reply holds `after_tokens` tokens and none of them is `unless_emitted`, the engine puts `tokens` in place of
/// the next one.
#[derive(Clone, Debug)]
pub struct InsertRule {
    pub after_tokens: u64,
    pub unless_emitted: Option<u32>,
    pub tokens: Vec<u32>,
}

fn opt_int(b: &mut Vec<u8>, v: Option<i64>) {
    match v {
        Some(i) => pyjson::int(b, i),
        None => b.extend_from_slice(b"null"),
    }
}

pub fn generate(id: &str, prompt: &[u32], p: &GenParams) -> Bytes {
    let mut w = FrameWriter::with_capacity(256 + prompt.len() * 7 + p.stop_ids.len() * 8);
    let b = &mut w.b;
    b.extend_from_slice(b"{\"id\":");
    pyjson::string(b, id);
    b.extend_from_slice(b",\"prompt\":[");
    for (i, t) in prompt.iter().enumerate() {
        if i > 0 {
            b.push(b',');
        }
        pyjson::int(b, *t);
    }
    b.extend_from_slice(b"],\"max_tokens\":");
    pyjson::int(b, p.max_tokens);
    b.extend_from_slice(b",\"min_tokens\":");
    pyjson::int(b, p.min_tokens);
    b.extend_from_slice(b",\"temperature\":");
    pyjson::float(b, p.temperature);
    b.extend_from_slice(b",\"top_p\":");
    pyjson::float(b, p.top_p);
    b.extend_from_slice(b",\"top_k\":");
    pyjson::int(b, p.top_k);
    b.extend_from_slice(b",\"min_p\":");
    pyjson::float(b, p.min_p);
    b.extend_from_slice(b",\"seed\":");
    opt_int(b, p.seed);
    b.extend_from_slice(b",\"stop_ids\":[");
    for (i, t) in p.stop_ids.iter().enumerate() {
        if i > 0 {
            b.push(b',');
        }
        pyjson::int(b, *t);
    }
    b.extend_from_slice(b"],\"logprobs\":");
    opt_int(b, p.logprobs);
    b.extend_from_slice(b",\"prompt_logprobs\":");
    opt_int(b, p.prompt_logprobs);
    if let Some(n) = p.system_tokens {
        b.extend_from_slice(b",\"system_tokens\":");
        pyjson::int(b, n);
    }
    if let Some(r) = &p.insert {
        b.extend_from_slice(b",\"insert\":[{\"after_tokens\":");
        pyjson::int(b, r.after_tokens);
        b.extend_from_slice(b",\"unless_emitted\":");
        opt_int(b, r.unless_emitted.map(i64::from));
        b.extend_from_slice(b",\"tokens\":[");
        for (i, t) in r.tokens.iter().enumerate() {
            if i > 0 {
                b.push(b',');
            }
            pyjson::int(b, *t);
        }
        b.extend_from_slice(b"]}]");
    }
    if let Some(profile) = &p.steering_profile {
        b.extend_from_slice(b",\"steering_profile\":");
        pyjson::string(b, profile);
        b.extend_from_slice(b",\"steering_scale\":");
        pyjson::float(b, p.steering_scale.unwrap_or(1.0));
    }
    b.extend_from_slice(b",\"type\":\"generate\"}");
    w.finish()
}

pub fn cancel(id: &str) -> Bytes {
    let mut w = FrameWriter::with_capacity(64);
    w.b.extend_from_slice(b"{\"type\":\"cancel\",\"id\":");
    pyjson::string(&mut w.b, id);
    w.b.push(b'}');
    w.finish()
}

pub fn metrics(seq: u64) -> Bytes {
    let mut w = FrameWriter::with_capacity(48);
    w.b.extend_from_slice(b"{\"type\":\"metrics\",\"seq\":");
    pyjson::int(&mut w.b, seq);
    w.b.push(b'}');
    w.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn frames_roundtrip_and_split() {
        let p = GenParams {
            max_tokens: 5,
            min_tokens: 0,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            seed: None,
            stop_ids: vec![1, 2],
            logprobs: Some(3),
            prompt_logprobs: None,
            system_tokens: Some(2),
            insert: None,
            steering_profile: None,
            steering_scale: None,
        };
        let mut all = Vec::new();
        all.extend_from_slice(&generate("cmpl-1", &[10, 20, 30], &p));
        all.extend_from_slice(&cancel("cmpl-1"));
        let mut r = FrameReader::new(&all[..]);
        let f = r.next().await.unwrap().unwrap();
        let v: serde_json::Value = serde_json::from_slice(&f).unwrap();
        assert_eq!(v["prompt"], serde_json::json!([10, 20, 30]));
        assert_eq!(v["temperature"], serde_json::json!(0.0));
        assert_eq!(v["type"], "generate");
        assert_eq!(v["system_tokens"], serde_json::json!(2));
        let h = r.next().await.unwrap().unwrap();
        let h = head(&h).unwrap();
        assert_eq!((h.kind.as_ref(), h.id.as_deref()), ("cancel", Some("cmpl-1")));
        assert!(r.next().await.unwrap().is_none());
        assert!(!String::from_utf8_lossy(&generate("x", &[1], &p)[4..]).contains("insert"));
        assert!(!v.as_object().unwrap().contains_key("steering_profile"));
        assert!(!v.as_object().unwrap().contains_key("steering_scale"));
        let selected =
            GenParams { steering_profile: Some("example".into()), steering_scale: Some(1.1f32 as f64), ..p.clone() };
        let v: serde_json::Value = serde_json::from_slice(&generate("x", &[1], &selected)[4..]).unwrap();
        assert_eq!(v["steering_profile"], "example");
        assert_eq!((v["steering_scale"].as_f64().unwrap() as f32).to_bits(), 0x3f8ccccd);
        let p = GenParams {
            logprobs: None,
            insert: Some(InsertRule { after_tokens: 31, unless_emitted: Some(154842), tokens: vec![198, 154842, 271] }),
            ..p
        };
        let g = generate("x", &[1], &p);
        let v: serde_json::Value = serde_json::from_slice(&g[4..]).unwrap();
        assert_eq!(
            v["insert"],
            serde_json::json!([{"after_tokens": 31, "unless_emitted": 154842, "tokens": [198, 154842, 271]}])
        );
        let p = GenParams { insert: Some(InsertRule { unless_emitted: None, ..p.insert.unwrap() }), ..p };
        let v: serde_json::Value = serde_json::from_slice(&generate("x", &[1], &p)[4..]).unwrap();
        assert_eq!(v["insert"][0]["unless_emitted"], serde_json::Value::Null);
    }

    #[test]
    fn odd_routing_fields_route_nowhere() {
        let h = head(br#"{"type":"tokens","id":[1,2],"seq":-1,"ids":[1]}"#).unwrap();
        assert_eq!((h.kind.as_ref(), h.id.is_none(), h.seq.is_none()), ("tokens", true, true));
        let h = head(br#"{"id":"a\u0062","type":"finish","seq":7}"#).unwrap();
        assert_eq!((h.id.as_deref(), h.seq), (Some("ab"), Some(7)));
        assert!(head(b"[1]").is_err());
    }

    #[tokio::test]
    async fn truncated_frame_is_an_error() {
        let g = cancel("x");
        let mut r = FrameReader::new(&g[..g.len() - 1]);
        assert!(r.next().await.is_err());
    }
}
