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
    /// reply tokens inside the think block the prompt left open, through its `</think>` (all of them when the reply
    /// never closes it); 0 when the prompt opened none
    pub reasoning_tokens: usize,
    /// the raw `prompt_logprobs` frame; turned into the response shape only when the response is written
    pub prompt_logprobs: Option<Bytes>,
    pub ids: Vec<u32>,
    pub logprobs: Vec<LpEntry>,
    pub text: String,
    pub finish_reason: Option<&'static str>,
    pub stop_reason: Option<StopReason>,
}

/// A thinking budget, as TensorFold's ThinkBudget: while the think block is open, the reply's `limit`-th token is
/// replaced by `close` ("\n" + `</think>` + "\n\n"), even when it is `</think>` itself. So `limit` - 1 reasoning
/// tokens come before the close.
///
/// The engine cannot take a token it did not sample (PROTOCOL.md has no such command), so the cut ends the engine
/// run and the reply goes on in a new one whose prompt is the prompt, the reply so far and the close. Under S2 the
/// cancelled run stays in the pool, but prefix reuse keeps only its prompt's whole 4,096-row prefill cells. The
/// continuation prefills the prompt's last partial cell, the reasoning and the close again.
pub struct Budget {
    pub limit: usize,
    pub close: &'static [u32],
}

/// What one reply token did.
enum Push {
    Text,
    /// a member of `stop_ids` past `min_tokens`: the engine ends the run on it, and its text is never shown
    StopId,
    /// a stop string matched: the step carries the final text and finish fields
    StopStr,
}

pub struct Generation {
    pub id: String,
    pub prompt: Vec<u32>,
    pub params: Params,
    tok: &'static Tok,
    engine: Option<Arc<Engine>>,
    stream: Option<Stream>,
    detok: Detok,
    hold: usize, // chars
    max_stop_chars: usize,
    keep: bool, // accumulate ids/logprobs (non-streaming responses need them)
    pub res: Res,
    sent: usize, // bytes of res.text already released
    done: bool,
    /// `</think>` while the reply is inside the think block its prompt opened
    think_end: Option<u32>,
    budget: Option<Budget>,
    /// the reply's ids while a budget may still cut it: the continuation's prompt needs them
    reply: Vec<u32>,
    /// the last step closed the think block: the next call starts the continuation
    cut: bool,
    /// engine runs after the first (a budget cut starts one)
    runs: u32,
    /// the continuation's prompt length, for its log line
    cont_prompt: usize,
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

/// Whether `prompt` leaves a think block open: its last `<think>` comes after its last `</think>`.
pub fn think_open(prompt: &[u32], start: u32, end: u32) -> bool {
    match prompt.iter().rposition(|&t| t == start || t == end) {
        Some(i) => prompt[i] == start,
        None => false,
    }
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
            engine: None,
            stream: None,
            detok: Detok::new(tok, skip_special),
            hold,
            max_stop_chars,
            keep,
            res,
            sent: 0,
            done: false,
            think_end: None,
            budget: None,
            reply: Vec::new(),
            cut: false,
            runs: 0,
            cont_prompt: 0,
        }
    }

    /// The reply starts inside a think block (the chat prompt opened one): count its reasoning tokens, and with a
    /// budget close the block at the reply's `budget.limit`-th token.
    pub fn thinking(&mut self, think_end: u32, budget: Option<Budget>) {
        self.think_end = Some(think_end);
        self.budget = budget.filter(|b| b.limit > 0 && !b.close.is_empty());
    }

    pub fn start(&mut self, engine: &Arc<Engine>) -> Result<(), ApiError> {
        if !engine.ready() {
            return Err(ApiError::new(503, "engine not connected", None));
        }
        let s = engine
            .generate(self.id.clone(), &self.prompt, &self.params.gen)
            .map_err(|e| ApiError::new(503, &e.0, None))?;
        self.stream = Some(s);
        self.engine = Some(engine.clone());
        Ok(())
    }

    /// The next step, or None after the final one (which has `finish_reason` set).
    pub async fn next(&mut self) -> Result<Option<Step>, ApiError> {
        if std::mem::take(&mut self.cut) {
            self.continue_after_close().await?;
        }
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
                        // the reply ended here (a stop string, or a budget's close that reached max_tokens):
                        // dropping the stream cancels the engine request
                        self.done = true;
                        self.stream = None;
                        return Ok(Some(step));
                    }
                    if self.cut || !step.text.is_empty() || !step.ids.is_empty() {
                        return Ok(Some(step));
                    }
                }
                Kind::Finish => {
                    self.done = true;
                    let fin: wire::Finish = serde_json::from_slice(&f)
                        .map_err(|e| ApiError::new(500, &format!("bad finish frame: {e}"), None))?;
                    if self.runs == 0 {
                        self.res.cached_tokens = fin.cached_tokens.unwrap_or(0);
                    } else {
                        log::info!(
                            "{}: thinking budget: the continuation took {} of its {} prompt tokens from the cache",
                            self.id,
                            fin.cached_tokens.unwrap_or(0),
                            self.cont_prompt
                        );
                    }
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
                            self.res.stop_reason = Some(self.stop_id_reason(sid));
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

    fn stop_id_reason(&self, sid: i64) -> StopReason {
        if Some(sid) == self.tok.eos_token_id.map(i64::from) {
            StopReason::None
        } else {
            StopReason::Id(sid)
        }
    }

    /// After a budget cut: end the cut run, then go on from the prompt, the reply and the close.
    async fn continue_after_close(&mut self) -> Result<(), ApiError> {
        let Some(mut old) = self.stream.take() else { return Ok(()) };
        old.cancel();
        // Wait for the cut run's finish: usage reports its cached tokens, and the engine has let the run go (S2
        // keeps it for reuse) before the continuation is admitted. Its tokens past the cut are not the reply's.
        loop {
            match old.next().await {
                Some(Msg::Frame(Kind::Finish, f)) => {
                    if let Ok(fin) = serde_json::from_slice::<wire::Finish>(&f) {
                        self.res.cached_tokens = fin.cached_tokens.unwrap_or(0);
                    }
                    break;
                }
                Some(Msg::Frame(..)) => {}
                Some(Msg::Lost(m)) => {
                    self.done = true;
                    return Err(ApiError::new(500, &format!("engine error: {m}"), None));
                }
                None => {
                    self.done = true;
                    return Err(ApiError::new(500, "engine error: engine connection lost", None));
                }
            }
        }
        let engine = self.engine.clone().ok_or_else(|| ApiError::new(500, "generation not started", None))?;
        let reply = std::mem::take(&mut self.reply);
        let mut prompt = Vec::with_capacity(self.prompt.len() + reply.len());
        prompt.extend_from_slice(&self.prompt);
        prompt.extend_from_slice(&reply);
        let n = reply.len() as u64;
        let mut p = self.params.gen.clone();
        p.max_tokens -= n; // close_think() ends the reply when the close reaches max_tokens
        p.min_tokens = p.min_tokens.saturating_sub(n);
        p.prompt_logprobs = None;
        self.runs += 1;
        self.cont_prompt = prompt.len();
        let id = format!("{}-{}", self.id, self.runs);
        let s = engine.generate(id, &prompt, &p).map_err(|e| ApiError::new(503, &e.0, None))?;
        self.stream = Some(s);
        Ok(())
    }

    fn on_tokens(&mut self, f: wire::Tokens) -> Step {
        let mut step = Step { ids: Vec::with_capacity(f.ids.len()), ..Default::default() };
        let mut lps = f.logprobs.unwrap_or_default().into_iter();
        for tid in f.ids {
            if self.budget_due() {
                return self.close_think(step);
            }
            if let Push::StopStr = self.push(tid, lps.next(), &mut step) {
                return step;
            }
        }
        step.text = self.release(false);
        step
    }

    /// Whether the next token is the one the budget replaces: the block is open and it is the `limit`-th.
    fn budget_due(&self) -> bool {
        self.think_end.is_some() && self.budget.as_ref().is_some_and(|b| self.res.completion_tokens + 1 >= b.limit)
    }

    /// Put the budget's close in place of the next token. `max_tokens` may cut the close short.
    fn close_think(&mut self, mut step: Step) -> Step {
        // the close goes into `reply` (the continuation's prompt) while the budget is still set
        let close = self.budget.as_ref().map_or(&[][..], |b| b.close);
        let room = (self.params.gen.max_tokens as usize).saturating_sub(self.res.completion_tokens);
        for &tid in close.iter().take(room) {
            match self.push(tid, None, &mut step) {
                Push::Text => {}
                Push::StopStr => return step,
                Push::StopId => {
                    // the request named a token of the close as a stop id: the reply ends on it
                    let reason = self.stop_id_reason(tid as i64);
                    self.res.finish_reason = Some("stop");
                    self.res.stop_reason = Some(reason.clone());
                    step.text = self.release(true);
                    step.finish_reason = Some("stop");
                    step.stop_reason = Some(reason);
                    return step;
                }
            }
        }
        self.budget = None;
        if self.res.completion_tokens as u64 >= self.params.gen.max_tokens {
            self.res.finish_reason = Some("length");
            step.text = self.release(true);
            step.finish_reason = Some("length");
            return step;
        }
        self.cut = true;
        step.text = self.release(false);
        step
    }

    /// One reply token: counted, kept, detokenized and checked against the stop strings.
    fn push(&mut self, tid: u32, lp: Option<LpEntry>, step: &mut Step) -> Push {
        self.res.completion_tokens += 1;
        step.ids.push(tid);
        if self.keep {
            self.res.ids.push(tid);
        }
        if self.budget.is_some() {
            self.reply.push(tid);
        }
        if let Some(e) = lp {
            if self.keep {
                self.res.logprobs.push(e.clone());
            }
            step.logprobs.push(e);
        }
        if let Some(end) = self.think_end {
            self.res.reasoning_tokens += 1;
            if tid == end {
                self.think_end = None;
            }
        }
        let min_tokens = self.params.gen.min_tokens as usize;
        if self.params.gen.stop_ids.contains(&(tid as i64)) && self.res.completion_tokens > min_tokens {
            return Push::StopId;
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
                return Push::StopStr;
            }
        }
        Push::Text
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Mutex, OnceLock};
    use std::time::Duration;

    use serde_json::{json, Value};
    use tokio::io::AsyncWriteExt;
    use tokio::net::UnixListener;
    use tokio::sync::mpsc;

    use super::*;
    use crate::wire::{FrameReader, GenParams};

    // a byte-level tokenizer small enough to read: a b c x y z ! and space, "\n" (4), "\n\n" (5), <think> (10),
    // </think> (11) and <|endoftext|> (12)
    const A: u32 = 0;
    const Y: u32 = 7;
    const NL: u32 = 4;
    const NLNL: u32 = 5;
    const THINK: u32 = 10;
    const END: u32 = 11;
    const EOS: u32 = 12;

    fn tok() -> &'static Tok {
        static T: OnceLock<&'static Tok> = OnceLock::new();
        T.get_or_init(|| {
            let dir = std::env::temp_dir().join(format!("g6front-gen-test-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let added = |id: u32, c: &str, special: bool| {
                json!({"id": id, "content": c, "single_word": false, "lstrip": false, "rstrip": false,
                       "normalized": false, "special": special})
            };
            let tk = json!({
                "version": "1.0", "truncation": null, "padding": null,
                "added_tokens": [added(THINK, "<think>", false), added(END, "</think>", false),
                                 added(EOS, "<|endoftext|>", true)],
                "normalizer": null,
                "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
                "post_processor": null,
                "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
                "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
                          "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                          "vocab": {"a": 0, "b": 1, "c": 2, "Ġ": 3, "Ċ": 4, "ĊĊ": 5, "x": 6, "y": 7, "z": 8, "!": 9},
                          "merges": ["Ċ Ċ"]}
            });
            std::fs::write(dir.join("tokenizer.json"), tk.to_string()).unwrap();
            std::fs::write(dir.join("tokenizer_config.json"), r#"{"eos_token": "<|endoftext|>"}"#).unwrap();
            std::fs::write(dir.join("generation_config.json"), r#"{"eos_token_id": [12]}"#).unwrap();
            Box::leak(Box::new(Tok::load(&dir).unwrap()))
        })
    }

    /// A PROTOCOL.md engine: each `generate` gets `script(its frame)`'s ids, three per `tokens` frame, cut at
    /// max_tokens. It ends `stop` on a stop id, `length` at the end, `abort` on `cancel`. cached_tokens is the run's
    /// ordinal + 100 (abort: + 200), so a test can tell which run's finish it reports.
    struct FakeEngine {
        gens: Arc<Mutex<Vec<Value>>>,
        engine: Arc<Engine>,
    }

    fn frame(v: Value) -> Vec<u8> {
        let b = serde_json::to_vec(&v).unwrap();
        let mut out = (b.len() as u32).to_be_bytes().to_vec();
        out.extend(b);
        out
    }

    async fn fake_engine(script: fn(&Value) -> Vec<u32>) -> FakeEngine {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("g6front-gen-{}-{n}.sock", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path).unwrap();
        let gens = Arc::new(Mutex::new(Vec::new()));
        let g2 = gens.clone();
        tokio::spawn(async move {
            let (sock, _) = listener.accept().await.unwrap();
            let (rd, mut wr) = sock.into_split();
            let (tx, mut rx) = mpsc::unbounded_channel::<Vec<u8>>();
            tokio::spawn(async move {
                while let Some(b) = rx.recv().await {
                    if wr.write_all(&b).await.is_err() {
                        break;
                    }
                }
            });
            tx.send(frame(json!({"type": "hello", "version": 1, "max_model_len": 4096}))).unwrap();
            let cancels: Arc<Mutex<HashMap<String, Arc<AtomicBool>>>> = Arc::default();
            let mut r = FrameReader::new(rd);
            while let Ok(Some(f)) = r.next().await {
                let v: Value = serde_json::from_slice(&f).unwrap();
                let id = v["id"].as_str().unwrap_or("").to_string();
                match v["type"].as_str() {
                    Some("cancel") => {
                        if let Some(c) = cancels.lock().unwrap().get(&id) {
                            c.store(true, Ordering::Relaxed);
                        }
                    }
                    Some("generate") => {
                        let run = {
                            let mut g = g2.lock().unwrap();
                            g.push(v.clone());
                            g.len() - 1
                        };
                        let cancel = Arc::new(AtomicBool::new(false));
                        cancels.lock().unwrap().insert(id.clone(), cancel.clone());
                        let ids = script(&v);
                        let max = v["max_tokens"].as_u64().unwrap() as usize;
                        let stops: Vec<u32> =
                            v["stop_ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
                        let min = v["min_tokens"].as_u64().unwrap() as usize;
                        let tx = tx.clone();
                        tokio::spawn(async move {
                            let mut out: Vec<u32> = Vec::new();
                            let mut end = (json!("length"), json!(null));
                            for &t in ids.iter().take(max) {
                                out.push(t);
                                if stops.contains(&t) && out.len() > min {
                                    end = (json!("stop"), json!(t));
                                    break;
                                }
                            }
                            for chunk in out.chunks(3) {
                                tokio::time::sleep(Duration::from_millis(2)).await;
                                if cancel.load(Ordering::Relaxed) {
                                    let f = json!({"type": "finish", "id": id, "reason": "abort", "stop_id": null,
                                                   "cached_tokens": 200 + run});
                                    let _ = tx.send(frame(f));
                                    return;
                                }
                                let _ = tx.send(frame(json!({"type": "tokens", "id": id, "ids": chunk})));
                            }
                            // a cancel that comes after the last tokens frame finds the run finished
                            let f = json!({"type": "finish", "id": id, "reason": end.0, "stop_id": end.1,
                                           "cached_tokens": 100 + run});
                            let _ = tx.send(frame(f));
                        });
                    }
                    _ => {}
                }
            }
        });
        let engine = Engine::new(path.to_string_lossy().into_owned());
        engine.start();
        for _ in 0..500 {
            if engine.ready() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(engine.ready(), "the fake engine did not connect");
        FakeEngine { gens, engine }
    }

    fn params(max_tokens: u64, min_tokens: u64) -> Params {
        Params {
            gen: GenParams {
                max_tokens,
                min_tokens,
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                min_p: 0.0,
                seed: None,
                stop_ids: vec![EOS as i64],
                logprobs: None,
                prompt_logprobs: None,
                system_tokens: None,
                insert: None,
            },
            stop: vec![],
            include_stop: false,
        }
    }

    /// Reasoning that never closes on its own. After a budget's close (the prompt ends "\n\n"): two answer
    /// tokens and EOS.
    fn endless_thinking(g: &Value) -> Vec<u32> {
        let prompt = g["prompt"].as_array().unwrap();
        if prompt.last() == Some(&json!(NLNL)) {
            vec![Y, Y, EOS]
        } else {
            vec![A; 1000]
        }
    }

    /// Two reasoning tokens, </think>, two answer tokens, EOS. After a budget's close: two answer tokens and EOS.
    fn closes_early(g: &Value) -> Vec<u32> {
        if g["prompt"].as_array().unwrap().last() == Some(&json!(NLNL)) {
            vec![Y, Y, EOS]
        } else {
            vec![A, A, END, Y, Y, EOS]
        }
    }

    const PROMPT: [u32; 2] = [6, THINK]; // "x<think>"

    /// (steps, the generation) for one request; `budget` 0 = none.
    async fn run(fake: &FakeEngine, p: Params, budget: usize, keep: bool) -> (Vec<Step>, Generation) {
        let t = tok();
        let mut g = Generation::new(t, format!("chatcmpl-{}", crate::util::hex_id()), PROMPT.to_vec(), p, false, keep);
        assert!(think_open(&g.prompt, THINK, END));
        g.thinking(END, (budget > 0).then_some(Budget { limit: budget, close: &t.think_close }));
        g.start(&fake.engine).unwrap();
        let mut steps = Vec::new();
        while let Some(s) = g.next().await.unwrap() {
            steps.push(s);
        }
        (steps, g)
    }

    fn joined(steps: &[Step]) -> (String, Vec<u32>) {
        (steps.iter().map(|s| s.text.as_str()).collect(), steps.iter().flat_map(|s| s.ids.clone()).collect())
    }

    #[test]
    fn the_close_is_newline_think_end_blank_line() {
        assert_eq!(tok().think_close, vec![NL, END, NLNL]);
        assert_eq!((tok().think_start, tok().think_end), (Some(THINK), Some(END)));
    }

    #[test]
    fn a_prompt_leaves_a_think_block_open_when_its_last_think_tag_opens_one() {
        assert!(think_open(&[1, THINK], THINK, END));
        assert!(think_open(&[THINK, 1, END, 2, THINK, 3], THINK, END));
        assert!(!think_open(&[THINK, 1, END], THINK, END));
        assert!(!think_open(&[1, 2], THINK, END));
        assert!(!think_open(&[], THINK, END));
    }

    #[tokio::test]
    async fn a_budget_reached_mid_thinking_closes_the_block_and_the_reply_goes_on() {
        let fake = fake_engine(endless_thinking).await;
        let (steps, g) = run(&fake, params(50, 0), 5, true).await;
        // the 5th reply token becomes the close (TensorFold): 4 reasoning tokens before it
        let want = [A, A, A, A, NL, END, NLNL, Y, Y, EOS];
        assert_eq!(g.res.ids, want);
        assert_eq!(joined(&steps).1, want);
        assert_eq!(g.res.text, "aaaa\n</think>\n\nyy");
        assert_eq!(joined(&steps).0, g.res.text);
        assert_eq!((g.res.finish_reason, &g.res.stop_reason), (Some("stop"), &Some(StopReason::None)));
        // usage counts each reply token once: the close counts, the cut run's tokens past the cut do not. Reasoning
        // runs through </think>, and cached_tokens is the first run's
        assert_eq!(g.res.completion_tokens, want.len());
        assert_eq!(g.res.reasoning_tokens, 6);
        assert_eq!(g.res.cached_tokens, 200);
        let gens = fake.gens.lock().unwrap().clone();
        assert_eq!(gens.len(), 2);
        let mut cont = PROMPT.to_vec();
        cont.extend([A, A, A, A, NL, END, NLNL]);
        assert_eq!(gens[1]["prompt"], json!(cont));
        assert_eq!(gens[1]["max_tokens"], json!(50 - 7));
        assert_ne!(gens[0]["id"], gens[1]["id"]);
    }

    #[tokio::test]
    async fn streaming_gives_the_same_reply_and_the_close_arrives_in_one_step() {
        let fake = fake_engine(endless_thinking).await;
        let (full, g_full) = run(&fake, params(50, 0), 5, true).await;
        let (steps, g) = run(&fake, params(50, 0), 5, false).await;
        assert_eq!(joined(&steps), joined(&full));
        assert!(g.res.ids.is_empty()); // a stream keeps no ids
        assert_eq!((g.res.completion_tokens, g.res.reasoning_tokens), (10, 6));
        assert_eq!(g.res.completion_tokens, g_full.res.completion_tokens);
        let close = steps.iter().find(|s| s.ids.contains(&END)).unwrap();
        assert!(close.ids.ends_with(&[NL, END, NLNL]) && close.text.contains("</think>"));
        assert_eq!(steps.last().unwrap().finish_reason, Some("stop"));
    }

    #[tokio::test]
    async fn think_end_before_the_budget_needs_no_cut() {
        let fake = fake_engine(closes_early).await;
        let (_, g) = run(&fake, params(50, 0), 5, true).await;
        assert_eq!(g.res.ids, [A, A, END, Y, Y, EOS]);
        assert_eq!((g.res.reasoning_tokens, g.res.completion_tokens, g.res.cached_tokens), (3, 6, 100));
        assert_eq!(fake.gens.lock().unwrap().len(), 1);
        // the budget's token is </think> itself: TensorFold replaces it with the close all the same
        let (_, g) = run(&fake, params(50, 0), 3, true).await;
        assert_eq!(g.res.ids, [A, A, NL, END, NLNL, Y, Y, EOS]);
        assert_eq!(g.res.reasoning_tokens, 4);
        assert_eq!(fake.gens.lock().unwrap().len(), 3);
    }

    #[tokio::test]
    async fn budget_one_makes_the_first_reply_token_the_close() {
        let fake = fake_engine(endless_thinking).await;
        let (_, g) = run(&fake, params(50, 0), 1, true).await;
        assert_eq!(g.res.ids, [NL, END, NLNL, Y, Y, EOS]);
        assert_eq!((g.res.reasoning_tokens, g.res.completion_tokens), (2, 6));
    }

    #[tokio::test]
    async fn budget_zero_is_no_limit() {
        let fake = fake_engine(endless_thinking).await;
        let (_, g) = run(&fake, params(20, 0), 0, true).await;
        assert_eq!(g.res.ids, [A; 20]);
        assert_eq!((g.res.finish_reason, g.res.completion_tokens, g.res.reasoning_tokens), (Some("length"), 20, 20));
        assert_eq!(fake.gens.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn the_close_is_cut_to_max_tokens() {
        let fake = fake_engine(endless_thinking).await;
        let (steps, g) = run(&fake, params(6, 0), 5, true).await;
        assert_eq!(g.res.ids, [A, A, A, A, NL, END]);
        assert_eq!((g.res.finish_reason, g.res.completion_tokens), (Some("length"), 6));
        assert_eq!(steps.last().unwrap().finish_reason, Some("length"));
        let (_, g) = run(&fake, params(7, 0), 5, true).await; // the whole close fits, nothing after it
        assert_eq!(g.res.ids, [A, A, A, A, NL, END, NLNL]);
        assert_eq!(g.res.finish_reason, Some("length"));
        assert_eq!(fake.gens.lock().unwrap().len(), 2); // one run each: no continuation
    }

    #[tokio::test]
    async fn the_continuation_owes_what_is_left_of_min_tokens() {
        let fake = fake_engine(endless_thinking).await;
        let (_, g) = run(&fake, params(50, 9), 5, true).await;
        let gens = fake.gens.lock().unwrap().clone();
        assert_eq!((gens[0]["min_tokens"].clone(), gens[1]["min_tokens"].clone()), (json!(9), json!(2)));
        // the continuation's EOS comes at reply token 10, past min_tokens
        assert_eq!(g.res.finish_reason, Some("stop"));
    }
}
