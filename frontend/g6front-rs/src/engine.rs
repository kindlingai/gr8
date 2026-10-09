//! One multiplexed Unix-socket connection to the engine (PROTOCOL.md), reconnecting on loss.
//!
//! The read loop routes each frame by id to the request that owns it as a `Bytes` slice of the read buffer;
//! frames for unknown (cancelled) ids are dropped. Writes go through one writer task so a request never waits on
//! another's write. Dropping a `Stream` before its `finish` frame sends `cancel`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use serde_json::Value;
use tokio::io::AsyncWriteExt;
use tokio::net::UnixStream;
use tokio::sync::{mpsc, oneshot};

use crate::wire::{self, FrameReader, GenParams};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Tokens,
    PromptLogprobs,
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
}

impl Engine {
    pub fn new(path: String) -> Arc<Engine> {
        Arc::new(Engine { path, conn: Mutex::default(), ready: AtomicBool::new(false), seq: AtomicU64::new(1) })
    }

    pub fn ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    pub fn hello(&self) -> Option<Arc<Value>> {
        self.conn.lock().unwrap().hello.clone()
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
            tokio::time::sleep(Duration::from_secs_f64(backoff)).await;
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
}

impl Drop for Stream {
    fn drop(&mut self) {
        if !self.finished {
            self.engine.conn.lock().unwrap().routes.remove(self.id.as_str());
            let _ = self.engine.send(wire::cancel(&self.id));
        }
    }
}
