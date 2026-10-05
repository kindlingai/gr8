//! One request's generation: engine frames -> text steps with stop tokens / stop strings applied (vLLM semantics),
//! a port of `g6front/runner.py`. String positions that Python counts in characters (stop-string holdback and
//! search windows) are counted in characters here too, so stream chunk boundaries come out the same.

use std::sync::Arc;

use bytes::Bytes;

use crate::api::ApiError;
use crate::engine::{Engine, Kind, Msg, Stream};
use crate::sampling::Params;
use crate::tok::{Detok, Tok};
use crate::wire::{self, LpEntry};

#[derive(Clone, Debug, PartialEq)]
pub enum StopReason {
    None,
    Id(i64),
    Str(String),
}

#[derive(Default)]
pub struct Step {
    pub text: String,
    pub ids: Vec<u32>,
    pub logprobs: Vec<LpEntry>,
    pub finish_reason: Option<&'static str>,
    pub stop_reason: Option<StopReason>,
}

#[derive(Default)]
pub struct Res {
    pub prompt_tokens: usize,
    pub cached_tokens: u64,
    pub completion_tokens: usize,
    /// the raw `prompt_logprobs` frame; turned into the response shape only when the response is written
    pub prompt_logprobs: Option<Bytes>,
    pub ids: Vec<u32>,
    pub logprobs: Vec<LpEntry>,
    pub text: String,
    pub finish_reason: Option<&'static str>,
    pub stop_reason: Option<StopReason>,
}

pub struct Generation {
    pub id: String,
    pub prompt: Vec<u32>,
    pub params: Params,
    tok: &'static Tok,
    stream: Option<Stream>,
    detok: Detok,
    hold: usize, // chars
    max_stop_chars: usize,
    keep: bool, // accumulate ids/logprobs (non-streaming responses need them)
    pub res: Res,
    sent: usize, // bytes of res.text already released
    done: bool,
}

/// A finish reason PROTOCOL.md does not list, passed through as Python does; each distinct string is stored once.
fn intern(s: &str) -> &'static str {
    static SEEN: std::sync::Mutex<Vec<&'static str>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap();
    if let Some(x) = seen.iter().find(|x| **x == s) {
        return x;
    }
    let x: &'static str = Box::leak(s.to_string().into_boxed_str());
    seen.push(x);
    x
}

/// Byte index of the position `n` characters before byte index `end` (clamped to 0).
fn back_chars(s: &str, end: usize, n: usize) -> usize {
    if n == 0 {
        return end;
    }
    s[..end].char_indices().rev().nth(n - 1).map_or(0, |(i, _)| i)
}

impl Generation {
    pub fn new(
        tok: &'static Tok,
        id: String,
        prompt: Vec<u32>,
        params: Params,
        skip_special: bool,
        keep: bool,
    ) -> Generation {
        let max_stop_chars = params.stop.iter().map(|s| s.chars().count()).max().unwrap_or(0);
        let hold = if !params.stop.is_empty() && !params.include_stop { max_stop_chars - 1 } else { 0 };
        let res = Res { prompt_tokens: prompt.len(), ..Default::default() };
        Generation {
            id,
            prompt,
            params,
            tok,
            stream: None,
            detok: Detok::new(tok, skip_special),
            hold,
            max_stop_chars,
            keep,
            res,
            sent: 0,
            done: false,
        }
    }

    pub fn start(&mut self, engine: &Arc<Engine>) -> Result<(), ApiError> {
        if !engine.ready() {
            return Err(ApiError::new(503, "engine not connected", None));
        }
        let s = engine
            .generate(self.id.clone(), &self.prompt, &self.params.gen)
            .map_err(|e| ApiError::new(503, &e.0, None))?;
        self.stream = Some(s);
        Ok(())
    }

    /// The next step, or None after the final one (which has `finish_reason` set).
    pub async fn next(&mut self) -> Result<Option<Step>, ApiError> {
        loop {
            if self.done {
                return Ok(None);
            }
            let Some(stream) = self.stream.as_mut() else { return Ok(None) };
            let msg = stream.next().await;
            let (kind, f) = match msg {
                Some(Msg::Frame(k, f)) => (k, f),
                Some(Msg::Lost(m)) => {
                    self.done = true;
                    return Err(ApiError::new(500, &format!("engine error: {m}"), None));
                }
                None => {
                    self.done = true;
                    return Err(ApiError::new(500, "engine error: engine connection lost", None));
                }
            };
            match kind {
                Kind::PromptLogprobs => self.res.prompt_logprobs = Some(f),
                Kind::Tokens => {
                    let t: wire::Tokens = serde_json::from_slice(&f)
                        .map_err(|e| ApiError::new(500, &format!("bad tokens frame: {e}"), None))?;
                    let step = self.on_tokens(t);
                    if step.finish_reason.is_some() {
                        // stop string matched: dropping the stream cancels the engine request
                        self.done = true;
                        self.stream = None;
                        return Ok(Some(step));
                    }
                    if !step.text.is_empty() || !step.ids.is_empty() {
                        return Ok(Some(step));
                    }
                }
                Kind::Finish => {
                    self.done = true;
                    let fin: wire::Finish = serde_json::from_slice(&f)
                        .map_err(|e| ApiError::new(500, &format!("bad finish frame: {e}"), None))?;
                    self.res.cached_tokens = fin.cached_tokens.unwrap_or(0);
                    let reason: &'static str = match fin.reason.as_str() {
                        "error" => {
                            let m = fin.message.as_deref().unwrap_or("unknown");
                            return Err(ApiError::new(500, &format!("engine error: {m}"), None));
                        }
                        "abort" => return Err(ApiError::new(500, "request aborted by the engine", None)),
                        "stop" => "stop",
                        "length" => "length",
                        other => intern(other),
                    };
                    self.res.finish_reason = Some(reason);
                    if reason == "stop" {
                        if let Some(sid) = fin.stop_id {
                            self.res.stop_reason = Some(if Some(sid) == self.tok.eos_token_id.map(i64::from) {
                                StopReason::None
                            } else {
                                StopReason::Id(sid)
                            });
                        }
                    }
                    return Ok(Some(Step {
                        text: self.release(true),
                        finish_reason: Some(reason),
                        stop_reason: self.res.stop_reason.clone(),
                        ..Default::default()
                    }));
                }
            }
        }
    }

    fn on_tokens(&mut self, f: wire::Tokens) -> Step {
        let mut step = Step { ids: Vec::with_capacity(f.ids.len()), ..Default::default() };
        let mut lps = f.logprobs.unwrap_or_default().into_iter();
        let stop_ids = &self.params.gen.stop_ids;
        let min_tokens = self.params.gen.min_tokens as usize;
        for tid in f.ids {
            self.res.completion_tokens += 1;
            step.ids.push(tid);
            if self.keep {
                self.res.ids.push(tid);
            }
            if let Some(e) = lps.next() {
                if self.keep {
                    self.res.logprobs.push(e.clone());
                }
                step.logprobs.push(e);
            }
            if stop_ids.contains(&(tid as i64)) && self.res.completion_tokens > min_tokens {
                continue; // the engine finishes on it; its text is never shown
            }
            let before = self.res.text.len();
            if let Some(t) = self.detok.step(tid) {
                self.res.text.push_str(&t);
            }
            if !self.params.stop.is_empty() && self.res.completion_tokens > min_tokens {
                if let Some((i, s)) = self.find_stop(before) {
                    let cut = if self.params.include_stop { i + s.len() } else { i };
                    self.res.text.truncate(cut);
                    self.res.finish_reason = Some("stop");
                    self.res.stop_reason = Some(StopReason::Str(s.clone()));
                    step.text = self.release(true);
                    step.finish_reason = Some("stop");
                    step.stop_reason = Some(StopReason::Str(s));
                    return step;
                }
            }
        }
        step.text = self.release(false);
        step
    }

    fn find_stop(&self, before: usize) -> Option<(usize, String)> {
        let text = &self.res.text;
        let start = back_chars(text, before, self.max_stop_chars.saturating_sub(1));
        let mut best: Option<(usize, &String)> = None;
        for s in &self.params.stop {
            if let Some(i) = text[start..].find(s.as_str()) {
                let i = start + i;
                if best.is_none_or(|(b, _)| i < b) {
                    best = Some((i, s));
                }
            }
        }
        best.map(|(i, s)| (i, s.clone()))
    }

    fn release(&mut self, final_: bool) -> String {
        let len = self.res.text.len();
        let end = if final_ { len } else { self.sent.max(back_chars(&self.res.text, len, self.hold)) };
        // Python's text[sent:end] is '' when a stop string cut the text below what was already released
        let out = self.res.text[self.sent.min(end)..end].to_string();
        self.sent = end;
        out
    }
}
