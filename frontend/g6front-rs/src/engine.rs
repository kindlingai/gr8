//! One multiplexed Unix-socket connection to the engine (PROTOCOL.md), reconnecting on loss.
//!
//! The read loop routes each frame by id to the request that owns it as a `Bytes` slice of the read buffer;
//! frames for unknown (cancelled) ids are dropped. Writes go through one writer task so a request never waits on
//! another's write. Dropping a `Stream` before its `finish` frame sends `cancel`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bytes::Bytes;
use serde_json::Value;
use tokio::io::AsyncWriteExt;
use tokio::net::UnixStream;
use tokio::sync::{mpsc, oneshot, Notify};

use crate::wire::{self, EmbedParams, FrameReader, GenParams, SpeechParams};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Tokens,
    PromptLogprobs,
    Embedding,
    Audio,
    Codes,
    Finish,
}

/// A frame for one request, or the connection dying under it.
pub enum Msg {
    Frame(Kind, Bytes),
    Lost(&'static str),
}

#[derive(Debug)]
pub struct EngineError(pub String);

#[derive(Default)]
struct Conn {
    tx: Option<mpsc::UnboundedSender<Bytes>>,
    routes: HashMap<Box<str>, mpsc::UnboundedSender<Msg>>,
    metrics: HashMap<u64, oneshot::Sender<Value>>,
    hello: Option<Arc<Value>>,
}

pub struct Engine {
    path: String,
    conn: Mutex<Conn>,
    ready: AtomicBool,
    seq: AtomicU64,
    /// `/internal/hold`: until this instant a request waits for the engine instead of failing (a scale's cutover)
    hold_until: Mutex<Option<Instant>>,
    /// wakes the waiters when the hold or `ready` changes
    changed: Notify,
    /// requests blocked in `wait_ready`
    waiting: AtomicUsize,
    /// requests let through (`admit`) whose route is not registered yet: tokenizing, rendering
    admitted: AtomicUsize,
}

/// A request `Engine::admit` let through, counted as running until its route is registered or it ends.
pub struct Admit(Arc<Engine>);

impl Drop for Admit {
    fn drop(&mut self) {
        self.0.admitted.fetch_sub(1, Ordering::AcqRel);
    }
}

impl Engine {
    pub fn new(path: String) -> Arc<Engine> {
        Arc::new(Engine {
            path,
            conn: Mutex::default(),
            ready: AtomicBool::new(false),
            seq: AtomicU64::new(1),
            hold_until: Mutex::new(None),
            changed: Notify::new(),
            waiting: AtomicUsize::new(0),
            admitted: AtomicUsize::new(0),
        })
    }

    pub fn ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    /// The hold's deadline while one is set and not yet passed.
    fn held(&self) -> Option<Instant> {
        self.hold_until.lock().unwrap().filter(|t| Instant::now() < *t)
    }

    /// Hold new requests for up to `seconds` (0 clears the hold and releases the waiters).
    pub fn hold(&self, seconds: u64) {
        *self.hold_until.lock().unwrap() =
            (seconds > 0).then(|| Instant::now() + Duration::from_secs(seconds.min(86_400)));
        self.changed.notify_waiters();
    }

    /// (hold set, requests running on the engine or let through to it, requests waiting for it)
    pub fn counts(&self) -> (bool, usize, usize) {
        let running = self.conn.lock().unwrap().routes.len() + self.admitted.load(Ordering::Acquire);
        (self.held().is_some(), running, self.waiting.load(Ordering::Acquire))
    }

    /// `wait_ready` for a request, which is counted as running (`counts`) from before it reads the hold until its route
    /// is registered or the returned guard drops: a drain that sets the hold and then reads no running request cannot
    /// miss one that is still tokenizing. None: the engine is not connected.
    pub async fn admit(self: &Arc<Self>) -> Option<Admit> {
        loop {
            self.admitted.fetch_add(1, Ordering::AcqRel);
            let a = Admit(self.clone());
            if self.held().is_none() {
                return self.ready().then_some(a);
            }
            drop(a);
            if !self.wait_ready().await {
                return None;
            }
        }
    }

    /// Whether a new request may go to the engine. With a hold set it first waits until the hold is cleared or its
    /// deadline passes; with none it answers at once.
    pub async fn wait_ready(&self) -> bool {
        if self.held().is_none() {
            return self.ready();
        }
        self.waiting.fetch_add(1, Ordering::AcqRel);
        struct Waiting<'a>(&'a AtomicUsize);
        impl Drop for Waiting<'_> {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::AcqRel);
            }
        }
        let _w = Waiting(&self.waiting);
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable(); // registered before the check, so no change is missed
            let Some(deadline) = self.held() else { return self.ready() };
            tokio::select! {
                _ = changed => {}
                _ = tokio::time::sleep_until(deadline.into()) => {}
            }
        }
    }

    pub fn hello(&self) -> Option<Arc<Value>> {
        self.conn.lock().unwrap().hello.clone()
    }

    /// The engine masks a reply with `generate`'s grammar (its hello says `"grammar": true`).
    pub fn grammars(&self) -> bool {
        self.hello().is_some_and(|h| h.get("grammar") == Some(&Value::Bool(true)))
    }
    /// The engine takes `generate`'s images (its hello says `"vision": true`).
    pub fn vision(&self) -> bool {
        self.hello().is_some_and(|h| h.get("vision") == Some(&Value::Bool(true)))
    }
    /// The engine's image geometry name from `vision.s2` (e.g. `"hy4"`, `"qw38"`, `"glm5v"`, `"ink2"`, `"ds41"`).
    pub fn vision_model(&self) -> Option<String> {
        self.hello().and_then(|h| h.get("vision_model").and_then(|v| v.as_str()).map(str::to_string))
    }
    /// The engine applies `generate`'s insert rule (its hello says `"insert": true`).
    pub fn inserts(&self) -> bool {
        self.hello().is_some_and(|h| h.get("insert") == Some(&Value::Bool(true)))
    }
    /// The engine applies `generate`'s presence / frequency penalties (its hello says `"penalties": true`).
    pub fn penalties(&self) -> bool {
        self.hello().is_some_and(|h| h.get("penalties") == Some(&Value::Bool(true)))
    }
    /// The engine applies `generate`'s think-end boost (its hello says `"think_end_boost": true`).
    pub fn think_end_boost(&self) -> bool {
        self.hello().is_some_and(|h| h.get("think_end_boost") == Some(&Value::Bool(true)))
    }
    /// The engine applies `generate`'s sequence penalty (its hello says `"sequence_penalty": true`).
    pub fn sequence_penalty(&self) -> bool {
        self.hello().is_some_and(|h| h.get("sequence_penalty") == Some(&Value::Bool(true)))
    }
    /// The engine supports dense embeddings via `embed` / `embedding` frames (its hello says `"embeddings": true`).
    pub fn embeddings(&self) -> bool {
        self.hello().is_some_and(|h| h.get("embeddings") == Some(&Value::Bool(true)))
    }
    /// The engine's native embedding dimension, if advertised in `hello.embedding_dim`.
    pub fn embedding_dim(&self) -> Option<usize> {
        self.hello().and_then(|h| h.get("embedding_dim").and_then(Value::as_u64)).map(|d| d as usize)
    }
    /// The engine supports speech synthesis and codec encoding (its hello says `"speech": true`).
    pub fn speech(&self) -> bool {
        self.hello().is_some_and(|h| h.get("speech") == Some(&Value::Bool(true)))
    }
    /// The engine's audio sample rate in Hz (default 44100).
    pub fn sample_rate(&self) -> u32 {
        self.hello().and_then(|h| h.get("sample_rate").and_then(Value::as_u64)).map_or(44_100, |r| r as u32)
    }
    /// The engine's codec codebook count (default 10).
    pub fn codebooks(&self) -> usize {
        self.hello().and_then(|h| h.get("codebooks").and_then(Value::as_u64)).map_or(10, |c| c as usize)
    }
    /// The engine's PCM samples per audio frame (default 2048).
    pub fn frame_samples(&self) -> usize {
        self.hello().and_then(|h| h.get("frame_samples").and_then(Value::as_u64)).map_or(2048, |s| s as usize)
    }

    pub fn start(self: &Arc<Self>) {
        let me = self.clone();
        tokio::spawn(async move { me.run().await });
    }

    async fn run(self: Arc<Self>) {
        let mut backoff = 0.2f64;
        let mut ever = false;
        loop {
            let r = self.session(&mut backoff).await;
            let was = self.ready.swap(false, Ordering::AcqRel);
            ever |= was;
            if let Err(e) = r {
                if was || backoff >= 5.0 {
                    log::warn!("engine connection: {e}");
                }
            } else {
                log::warn!("engine closed the connection");
            }
            self.fail_all("engine connection lost");
            // a scale's cutover (hold set) brings the new engine up soon: retry often
            let wait = if self.held().is_some() { 0.25 } else { backoff };
            tokio::time::sleep(Duration::from_secs_f64(wait)).await;
            // until the first connection the shim is starting (the launchers start the front end early): retry often
            backoff = if ever { (backoff * 2.0).min(5.0) } else { 0.25 };
        }
    }

    async fn session(&self, backoff: &mut f64) -> Result<(), String> {
        let sock = UnixStream::connect(&self.path).await.map_err(|e| e.to_string())?;
        let (rd, mut wr) = sock.into_split();
        let mut reader = FrameReader::new(rd);
        let hello = reader.next().await.map_err(|e| e.0)?.ok_or("EOF before hello")?;
        let hv: Value = serde_json::from_slice(&hello).map_err(|e| format!("bad hello: {e}"))?;
        if hv.get("type").and_then(Value::as_str) != Some("hello") || hv.get("version") != Some(&Value::from(1)) {
            return Err(format!("bad hello: {}", String::from_utf8_lossy(&hello)));
        }
        let (tx, mut rx) = mpsc::unbounded_channel::<Bytes>();
        {
            let mut c = self.conn.lock().unwrap();
            c.tx = Some(tx);
            c.hello = Some(Arc::new(hv));
        }
        self.ready.store(true, Ordering::Release);
        log::info!("engine connected: {}", String::from_utf8_lossy(&hello));
        *backoff = 0.2;
        let writer = async move {
            // drain whatever is queued into one vectored write
            let mut batch: Vec<Bytes> = Vec::with_capacity(16);
            while rx.recv_many(&mut batch, 64).await > 0 {
                let mut slices: Vec<std::io::IoSlice> = batch.iter().map(|b| std::io::IoSlice::new(b)).collect();
                let mut bufs = &mut slices[..];
                while !bufs.is_empty() {
                    let n = wr.write_vectored(bufs).await.map_err(|e| format!("write: {e}"))?;
                    if n == 0 {
                        return Err("write: zero".to_string());
                    }
                    std::io::IoSlice::advance_slices(&mut bufs, n);
                }
                batch.clear();
            }
            Ok::<(), String>(())
        };
        let reading = async {
            while let Some(f) = reader.next().await.map_err(|e| e.0)? {
                self.dispatch(f)?;
            }
            Ok::<(), String>(())
        };
        tokio::select! {
            r = reading => r,
            r = writer => r,
        }
    }

    fn dispatch(&self, f: Bytes) -> Result<(), String> {
        let h = wire::head(&f).map_err(|e| e.0)?;
        let kind = match h.kind.as_ref() {
            "tokens" => Kind::Tokens,
            "prompt_logprobs" => Kind::PromptLogprobs,
            "embedding" => Kind::Embedding,
            "audio" => Kind::Audio,
            "codes" => Kind::Codes,
            "finish" => Kind::Finish,
            "metrics" => {
                if let Some(seq) = h.seq {
                    let w = self.conn.lock().unwrap().metrics.remove(&seq);
                    if let Some(w) = w {
                        let v: Value = serde_json::from_slice(&f).map_err(|e| format!("bad metrics: {e}"))?;
                        let _ = w.send(v);
                    }
                }
                return Ok(());
            }
            _ => return Ok(()), // unknown types are ignored
        };
        let Some(id) = h.id else { return Ok(()) };
        let mut c = self.conn.lock().unwrap();
        let route =
            if kind == Kind::Finish { c.routes.remove(id.as_ref()) } else { c.routes.get(id.as_ref()).cloned() };
        drop(c);
        if let Some(q) = route {
            let _ = q.send(Msg::Frame(kind, f));
        }
        Ok(())
    }

    fn fail_all(&self, msg: &'static str) {
        let mut c = self.conn.lock().unwrap();
        c.tx = None;
        for (_, q) in c.routes.drain() {
            let _ = q.send(Msg::Lost(msg));
        }
        c.metrics.clear(); // dropping the senders fails the waiters
    }

    fn send(&self, frame: Bytes) -> Result<(), EngineError> {
        let c = self.conn.lock().unwrap();
        match &c.tx {
            Some(tx) => tx.send(frame).map_err(|_| EngineError("engine not connected".into())),
            None => Err(EngineError("engine not connected".into())),
        }
    }

    /// Start a request. Frames arrive on the returned stream until `finish`.
    pub fn generate(self: &Arc<Self>, id: String, prompt: &[u32], p: &GenParams) -> Result<Stream, EngineError> {
        let frame = wire::generate(&id, prompt, p);
        self.start_stream(id, frame)
    }

    /// Start an embedding request. Expects one `embedding` frame followed by `finish`.
    pub fn embed(self: &Arc<Self>, id: String, prompt: &[u32], p: &EmbedParams) -> Result<Stream, EngineError> {
        let frame = wire::embed(&id, prompt, p);
        self.start_stream(id, frame)
    }

    /// Start a speech synthesis request. Expects zero or more `audio` frames followed by `finish`.
    pub fn speech_stream(self: &Arc<Self>, id: String, rows: &[i32], p: &SpeechParams) -> Result<Stream, EngineError> {
        let frame = wire::speech(&id, rows, p);
        self.start_stream(id, frame)
    }

    /// Start an audio encoding request. Expects one `codes` frame followed by `finish`.
    pub fn encode_audio(self: &Arc<Self>, id: String, pcm_b64: &str) -> Result<Stream, EngineError> {
        let frame = wire::encode_audio(&id, pcm_b64);
        self.start_stream(id, frame)
    }

    fn start_stream(self: &Arc<Self>, id: String, frame: Bytes) -> Result<Stream, EngineError> {
        let (qtx, qrx) = mpsc::unbounded_channel();
        {
            let mut c = self.conn.lock().unwrap();
            if c.routes.contains_key(id.as_str()) {
                return Err(EngineError(format!("duplicate request id {id}")));
            }
            let Some(tx) = &c.tx else { return Err(EngineError("engine not connected".into())) };
            tx.send(frame).map_err(|_| EngineError("engine not connected".into()))?;
            c.routes.insert(id.clone().into_boxed_str(), qtx);
        }
        Ok(Stream { id, rx: qrx, engine: self.clone(), finished: false })
    }

    pub async fn metrics(&self, timeout: Duration) -> Result<Value, EngineError> {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.conn.lock().unwrap().metrics.insert(seq, tx);
        if let Err(e) = self.send(wire::metrics(seq)) {
            self.conn.lock().unwrap().metrics.remove(&seq);
            return Err(e);
        }
        let r = tokio::time::timeout(timeout, rx).await;
        self.conn.lock().unwrap().metrics.remove(&seq);
        match r {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(_)) => Err(EngineError("engine connection lost".into())),
            Err(_) => Err(EngineError(String::new())), // Python: str(TimeoutError()) ==
        }
    }
}

/// One request's frames. Dropped before `finish` -> `cancel` goes to the engine and late frames are discarded.
pub struct Stream {
    id: String,
    rx: mpsc::UnboundedReceiver<Msg>,
    engine: Arc<Engine>,
    finished: bool,
}

impl Stream {
    pub async fn next(&mut self) -> Option<Msg> {
        let m = self.rx.recv().await;
        match &m {
            Some(Msg::Frame(Kind::Finish, _)) | Some(Msg::Lost(_)) | None => self.finished = true,
            _ => {}
        }
        m
    }

    /// Ask the engine to stop this request but keep reading it: its `finish` frame still arrives here.
    pub fn cancel(&self) {
        if !self.finished {
            let _ = self.engine.send(wire::cancel(&self.id));
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if !self.finished {
            self.engine.conn.lock().unwrap().routes.remove(self.id.as_str());
            let _ = self.engine.send(wire::cancel(&self.id));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn an_admitted_request_counts_as_running_until_its_route_or_its_end() {
        let e = Engine::new(String::new());
        e.ready.store(true, Ordering::Release);
        let a = e.admit().await.expect("ready");
        assert_eq!(e.counts(), (false, 1, 0), "admitted, still tokenizing: a drain must wait for it");
        e.hold(60); // the drain's hold: the admitted request still counts, so running is not 0
        assert_eq!(e.counts(), (true, 1, 0));
        drop(a);
        assert_eq!(e.counts(), (true, 0, 0));
        // a request during the hold waits, uncounted as running, and is admitted when the hold ends
        let w = tokio::spawn({
            let e = e.clone();
            async move { e.admit().await.is_some() }
        });
        while e.counts().2 == 0 {
            tokio::task::yield_now().await;
        }
        assert_eq!(e.counts(), (true, 0, 1));
        e.hold(0);
        assert!(w.await.unwrap());
        assert_eq!(e.counts(), (false, 0, 0));
        e.ready.store(false, Ordering::Release);
        assert!(e.admit().await.is_none());
        assert_eq!(e.counts(), (false, 0, 0));
    }
}
