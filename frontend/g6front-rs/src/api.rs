//! HTTP routes, shaped like vLLM 0.30:
//! POST /v1/chat/completions, /v1/completions, /tokenize, /detokenize; GET /v1/models, /health, /metrics.
//!
//! Responses are written directly as JSON text into one buffer per response or per SSE event; a stream's events
//! go to hyper through a bounded channel (backpressure), and a client that goes away drops the channel, which
//! drops the engine stream, which sends `cancel`.

use std::convert::Infallible;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use bytes::Bytes;
use http_body_util::BodyExt;
use hyper::body::{Body, Frame, Incoming};
use hyper::{header, Method, Request, Response, StatusCode};
use serde_json::{Map, Value};
use tokio::sync::mpsc;

use crate::dialect::{self, AnyParser, Dialect};
use crate::engine::Admit;
use crate::gen::{think_open, Budget, Generation, Res, Step, StepStats, StopReason};
use crate::guard;
use crate::jsonmode;
use crate::parser::Delta;
use crate::pyjson::{self, truthy};
use crate::template::{build_template_kwargs, normalize_tools};
use crate::tool_choice::ToolChoice;
use crate::util::{new_id, now_secs};
use crate::wire::InsertRule;
use crate::{logprobs as L, metrics, sampling, thinking, State};

const MAX_BODY: usize = 512 << 20;
const BODY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

// ---- errors ------------------------------------------------------------------------------------------------------

#[derive(Debug)]
pub struct ApiError {
    pub status: u16,
    pub message: String,
    pub param: Option<String>,
    /// seconds for the Retry-After response header (the busy guard's 429; PROTOCOL.md `finish`), None elsewhere
    pub retry_after: Option<u32>,
}

impl ApiError {
    pub fn new(status: u16, message: &str, param: Option<&str>) -> ApiError {
        ApiError { status, message: message.to_string(), param: param.map(str::to_string), retry_after: None }
    }

    /// The busy guard's refusal: an OpenAI-shaped 429 body with Retry-After (docs/ENV.md GR8_BUSY_*).
    pub fn busy(message: &str, retry_after: u32) -> ApiError {
        ApiError { status: 429, message: message.to_string(), param: None, retry_after: Some(retry_after) }
    }

    pub fn bad(message: impl Into<String>, param: Option<&str>) -> ApiError {
        ApiError { status: 400, message: message.into(), param: param.map(str::to_string), retry_after: None }
    }

    fn body(&self) -> Vec<u8> {
        let ty = match self.status {
            404 => "NotFoundError",
            429 => "RateLimitError",
            500 => "InternalServerError",
            503 => "ServiceUnavailableError",
            _ => "BadRequestError",
        };
        let mut o = Vec::with_capacity(self.message.len() + 96);
        o.extend_from_slice(b"{\"error\":{\"message\":");
        pyjson::string(&mut o, &self.message);
        o.extend_from_slice(b",\"type\":\"");
        o.extend_from_slice(ty.as_bytes());
        o.extend_from_slice(b"\",\"param\":");
        match &self.param {
            Some(p) => pyjson::string(&mut o, p),
            None => o.extend_from_slice(b"null"),
        }
        o.extend_from_slice(b",\"code\":");
        pyjson::int(&mut o, self.status);
        o.extend_from_slice(b"}}");
        o
    }

    fn response(&self) -> Response<HttpBody> {
        let mut r = json_response(self.status, self.body());
        if let Some(secs) = self.retry_after {
            if let Ok(v) = header::HeaderValue::from_str(&secs.to_string()) {
                r.headers_mut().insert(header::RETRY_AFTER, v);
            }
        }
        r
    }

    fn sse(&self) -> Bytes {
        sse_bytes(self.body())
    }
}

impl From<crate::template::TemplateError> for ApiError {
    fn from(e: crate::template::TemplateError) -> Self {
        ApiError::bad(e.0, None)
    }
}

// ---- bodies ------------------------------------------------------------------------------------------------------

pub enum HttpBody {
    Full(Option<Bytes>),
    Chan(mpsc::Receiver<Bytes>),
}

impl Body for HttpBody {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_frame(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Result<Frame<Bytes>, Infallible>>> {
        match self.get_mut() {
            HttpBody::Full(b) => Poll::Ready(b.take().map(|b| Ok(Frame::data(b)))),
            HttpBody::Chan(rx) => rx.poll_recv(cx).map(|o| o.map(|b| Ok(Frame::data(b)))),
        }
    }

    fn is_end_stream(&self) -> bool {
        matches!(self, HttpBody::Full(None))
    }

    fn size_hint(&self) -> hyper::body::SizeHint {
        match self {
            HttpBody::Full(Some(b)) => hyper::body::SizeHint::with_exact(b.len() as u64),
            HttpBody::Full(None) => hyper::body::SizeHint::with_exact(0),
            HttpBody::Chan(_) => hyper::body::SizeHint::default(),
        }
    }
}

fn response(status: u16, ctype: &'static str, body: Vec<u8>) -> Response<HttpBody> {
    let mut r = Response::new(HttpBody::Full(Some(Bytes::from(body))));
    *r.status_mut() = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    r.headers_mut().insert(header::CONTENT_TYPE, header::HeaderValue::from_static(ctype));
    r
}

fn json_response(status: u16, body: Vec<u8>) -> Response<HttpBody> {
    response(status, "application/json; charset=utf-8", body)
}

fn json_value(status: u16, v: Value) -> Response<HttpBody> {
    json_response(status, serde_json::to_vec(&v).unwrap_or_default())
}

fn sse_bytes(json: Vec<u8>) -> Bytes {
    let mut b = Vec::with_capacity(json.len() + 8);
    b.extend_from_slice(b"data: ");
    b.extend_from_slice(&json);
    b.extend_from_slice(b"\n\n");
    Bytes::from(b)
}

/// The SSE side of a stream: events go out through a bounded channel; `Err` = the client is gone.
struct Sse(mpsc::Sender<Bytes>);

impl Sse {
    fn open() -> (Sse, Response<HttpBody>) {
        let (tx, rx) = mpsc::channel(64);
        let mut r = Response::new(HttpBody::Chan(rx));
        let h = r.headers_mut();
        h.insert(header::CONTENT_TYPE, header::HeaderValue::from_static("text/event-stream; charset=utf-8"));
        h.insert(header::CACHE_CONTROL, header::HeaderValue::from_static("no-cache"));
        (Sse(tx), r)
    }

    async fn send(&self, b: Bytes) -> Result<(), ()> {
        self.0.send(b).await.map_err(|_| ())
    }

    async fn json(&self, json: Vec<u8>) -> Result<(), ()> {
        self.send(sse_bytes(json)).await
    }
}

// ---- JSON writing helpers ----------------------------------------------------------------------------------------

/// Writes `"key":` (with a leading comma after the first key) and returns the buffer for the value.
struct Obj<'a> {
    o: &'a mut Vec<u8>,
    first: bool,
}

impl<'a> Obj<'a> {
    fn new(o: &'a mut Vec<u8>) -> Obj<'a> {
        o.push(b'{');
        Obj { o, first: true }
    }
    fn key(&mut self, k: &str) -> &mut Vec<u8> {
        if !self.first {
            self.o.push(b',');
        }
        self.first = false;
        self.o.push(b'"');
        self.o.extend_from_slice(k.as_bytes());
        self.o.extend_from_slice(b"\":");
        self.o
    }
    fn raw(&mut self, k: &str, v: &[u8]) {
        self.key(k).extend_from_slice(v);
    }
    fn str(&mut self, k: &str, v: &str) {
        pyjson::string(self.key(k), v);
    }
    fn opt_str(&mut self, k: &str, v: Option<&str>) {
        match v {
            Some(s) => self.str(k, s),
            None => self.raw(k, b"null"),
        }
    }
    fn int<I: itoa::Integer>(&mut self, k: &str, v: I) {
        pyjson::int(self.key(k), v);
    }
    fn end(self) {
        self.o.push(b'}');
    }
}

fn ids_json(o: &mut Vec<u8>, ids: &[u32]) {
    o.push(b'[');
    for (i, t) in ids.iter().enumerate() {
        if i > 0 {
            o.push(b',');
        }
        pyjson::int(o, *t);
    }
    o.push(b']');
}

fn stop_reason_json(o: &mut Vec<u8>, s: &Option<StopReason>) {
    match s {
        Some(StopReason::Id(i)) => pyjson::int(o, *i),
        Some(StopReason::Str(s)) => pyjson::string(o, s),
        Some(StopReason::None) | None => o.extend_from_slice(b"null"),
    }
}

/// `reasoning`: a chat reply's reasoning tokens, as `completion_tokens_details.reasoning_tokens` (OpenAI's field), so
/// a client can tell visible output from reasoning. `stats`: `step_stats`, for a request that asked for it.
#[allow(clippy::too_many_arguments)]
fn usage(
    o: &mut Vec<u8>,
    prompt: usize,
    completion: usize,
    cached: Option<u64>,
    details: bool,
    full: bool,
    reasoning: Option<usize>,
    stats: Option<&StepStats>,
) {
    let mut u = Obj::new(o);
    u.int("prompt_tokens", prompt);
    u.int("total_tokens", prompt + completion);
    u.int("completion_tokens", completion);
    if details {
        let d = u.key("prompt_tokens_details");
        d.extend_from_slice(b"{\"cached_tokens\":");
        pyjson::int(d, cached.unwrap_or(0));
        d.push(b'}');
    } else if full {
        u.raw("prompt_tokens_details", b"null");
    }
    if let Some(r) = reasoning {
        let d = u.key("completion_tokens_details");
        d.extend_from_slice(b"{\"reasoning_tokens\":");
        pyjson::int(d, r);
        d.push(b'}');
    }
    if let Some(s) = stats {
        step_stats_json(u.key("step_stats"), s);
    }
    u.end();
}

/// `usage.step_stats` (`return_step_stats`): the request's share of the `/metrics` speculative-decoding counters and
/// request times. The times are left out when the engine did not report them.
fn step_stats_json(o: &mut Vec<u8>, s: &StepStats) {
    let mut x = Obj::new(o);
    x.int("steps", s.steps);
    x.int("draft_tokens", s.draft_tokens);
    x.int("accepted_tokens", s.accepted_tokens);
    let p = x.key("accepted_per_pos");
    p.push(b'[');
    for (i, n) in s.accepted_per_pos.iter().enumerate() {
        if i > 0 {
            p.push(b',');
        }
        pyjson::int(p, *n);
    }
    p.push(b']');
    x.int("generated_tokens", s.generated_tokens);
    if let Some(t) = s.prefill_s {
        pyjson::float(x.key("prefill_s"), t);
    }
    if let Some(t) = s.decode_s {
        pyjson::float(x.key("decode_s"), t);
    }
    x.end();
}

/// `{"id", "object", "created", "model"` -- the shared head of every chunk/response (object left open).
fn head<'a>(o: &'a mut Vec<u8>, id: &str, object: &str, created: u64, model: &str) -> Obj<'a> {
    let mut c = Obj::new(o);
    c.str("id", id);
    c.str("object", object);
    c.int("created", created);
    c.str("model", model);
    c
}

// ---- request plumbing --------------------------------------------------------------------------------------------

async fn read_json(req: Request<Incoming>) -> Result<Map<String, Value>, ApiError> {
    read_json_raw(req).await.map(|(m, _)| m)
}

/// The body as a JSON object, plus its raw bytes (sampling re-reads integers too large for serde's numbers).
async fn read_json_raw(req: Request<Incoming>) -> Result<(Map<String, Value>, Bytes), ApiError> {
    // a deadline for the whole body: a client that trickles it would otherwise hold its connection forever
    let body = tokio::time::timeout(BODY_TIMEOUT, http_body_util::Limited::new(req.into_body(), MAX_BODY).collect())
        .await
        .map_err(|_| ApiError::new(408, "request body not received in time", None))?
        .map_err(|e| ApiError::new(413, &format!("request body too large or unreadable: {e}"), None))?
        .to_bytes();
    match serde_json::from_slice::<Value>(&body) {
        Ok(Value::Object(m)) => Ok((m, body)),
        Ok(_) => Err(ApiError::bad("request body must be a JSON object", None)),
        Err(e) => Err(ApiError::bad(format!("invalid JSON body: {e}"), None)),
    }
}

/// A nonzero penalty needs an engine whose hello says `"penalties": true`, a think-end boost one that says
/// `"think_end_boost": true` (PROTOCOL.md). The thinking penalties and the boost apply inside the think block the prompt
/// leaves open, and the sequence penalty's answer scope outside it, so the frame names its close (`think_end`) when
/// there is one. A boost or a thinking scope with no open block is dropped.
fn check_penalties(st: &State, prompt: &[u32], g: &mut crate::wire::GenParams) -> Result<(), ApiError> {
    if g.think_end_boost.is_some() && !st.engine.think_end_boost() {
        return Err(ApiError::bad("think_end_boost is not supported by this engine", Some("think_end_boost")));
    }
    if (g.seq_thinking.is_some() || g.seq_answer.is_some() || g.seq_guard.is_some()) && !st.engine.sequence_penalty() {
        return Err(ApiError::bad("sequence_penalty is not supported by this engine", Some("sequence_penalty")));
    }
    let named = [
        ("presence_penalty", g.presence_penalty),
        ("frequency_penalty", g.frequency_penalty),
        ("thinking_presence_penalty", g.thinking_presence_penalty),
        ("thinking_frequency_penalty", g.thinking_frequency_penalty),
    ];
    if let Some((k, _)) = named.iter().find(|(_, v)| *v != 0.0) {
        if !st.engine.penalties() {
            return Err(ApiError::bad(format!("{k} is not supported by this engine"), Some(k)));
        }
    }
    if g.thinking_presence_penalty != 0.0
        || g.thinking_frequency_penalty != 0.0
        || g.think_end_boost.is_some()
        || g.seq_thinking.is_some()
        || g.seq_answer.is_some()
    {
        g.think_end = match (st.tok.think_start, st.tok.think_end) {
            (Some(s), Some(e)) if think_open(prompt, s, e) => Some(e),
            _ => None,
        };
        if g.think_end.is_none() {
            g.think_end_boost = None;
            g.seq_thinking = None;
        }
    }
    Ok(())
}

impl State {
    /// The jinja chat template (used by the `glm` and `hf` dialects).
    fn tpl(&self) -> &crate::template::ChatTemplate {
        self.template.as_ref().expect("a template-backed State has a chat template")
    }

    fn check_model(&self, body: &Map<String, Value>) -> Result<(), ApiError> {
        match body.get("model") {
            None | Some(Value::Null) => Ok(()),
            Some(Value::String(m)) if self.served.iter().any(|s| s == m) => Ok(()),
            Some(m) => {
                let m = m.as_str().map_or_else(|| m.to_string(), str::to_string);
                Err(ApiError::new(404, &format!("The model `{m}` does not exist."), Some("model")))
            }
        }
    }

    /// Waits through a scale's hold (`/internal/hold`); then 503 with Retry-After when the engine is not connected.
    /// The guard counts the request as running for a drain until its route is registered (Generation::start).
    async fn require_engine(&self) -> Result<Admit, ApiError> {
        match self.engine.admit().await {
            Some(a) => Ok(a),
            None => {
                Err(ApiError { status: 503, message: "engine not connected".into(), param: None, retry_after: Some(5) })
            }
        }
    }

    pub fn max_model_len(&self) -> u64 {
        if self.max_model_len_override > 0 {
            return self.max_model_len_override;
        }
        self.engine.hello().and_then(|h| h.get("max_model_len").and_then(Value::as_u64)).unwrap_or(0)
    }

    pub fn chat_default_max_tokens(&self, prompt_len: usize) -> Option<u64> {
        sampling::chat_default_max_tokens(self.default_max_tokens, self.share_tokens, prompt_len)
    }

    fn max_logprobs(&self) -> i64 {
        self.engine.hello().and_then(|h| h.get("max_logprobs").and_then(Value::as_i64)).unwrap_or(20)
    }

    fn steering(&self) -> Option<Value> {
        self.engine.hello().and_then(|h| h.get("steering").cloned())
    }
}

pub async fn route(
    st: &'static State,
    req: Request<Incoming>,
    peer: std::net::SocketAddr,
) -> Result<Response<HttpBody>, Infallible> {
    let path = req.uri().path().to_owned();
    let deep = req.uri().query().is_some_and(|q| q.split('&').any(|x| x == "deep" || x.starts_with("deep=")));
    let m = req.method().clone();
    let r = match (path.as_str(), &m) {
        ("/v1/chat/completions", &Method::POST) => chat(st, req).await,
        (p, &Method::POST) if st.sessions.is_some() && p.starts_with("/v1/sessions/") && p.ends_with("/forks") => {
            let id = &p[13..p.len() - 6];
            session_forks(st, id, req).await
        }
        (p, &Method::DELETE) if st.sessions.is_some() && p.starts_with("/v1/sessions/") => {
            let id = &p[13..];
            match st.sessions.as_ref().unwrap().delete(id) {
                Ok(()) => json_value(200, serde_json::json!({"id": id, "deleted": true})),
                Err(e) => e.response(),
            }
        }
        ("/v1/completions", &Method::POST) => completions(st, req).await,
        ("/tokenize", &Method::POST) => tokenize(st, req).await,
        ("/detokenize", &Method::POST) => detokenize(st, req).await,
        ("/v1/models", &Method::GET | &Method::HEAD) => models(st),
        ("/health", &Method::GET | &Method::HEAD) if deep => health_deep(st, peer).await,
        ("/health", &Method::GET | &Method::HEAD) => health(st),
        ("/internal/hold", &Method::GET | &Method::POST) => hold(st, &req, peer),
        ("/metrics", &Method::GET | &Method::HEAD) => metrics_route(st).await,
        ("/v1/chat/completions" | "/v1/completions" | "/tokenize" | "/detokenize", _) => {
            let mut r = response(405, "text/plain; charset=utf-8", b"405: Method Not Allowed".to_vec());
            r.headers_mut().insert(header::ALLOW, header::HeaderValue::from_static("POST"));
            r
        }
        ("/v1/models" | "/health" | "/metrics", _) => {
            let mut r = response(405, "text/plain; charset=utf-8", b"405: Method Not Allowed".to_vec());
            r.headers_mut().insert(header::ALLOW, header::HeaderValue::from_static("GET,HEAD"));
            r
        }
        _ => response(404, "text/plain; charset=utf-8", b"404: Not Found".to_vec()),
    };
    Ok(r)
}

async fn session_forks(st: &'static State, id: &str, req: Request<Incoming>) -> Response<HttpBody> {
    let (body, _) = match read_json_raw(req).await {
        Ok(b) => b,
        Err(e) => return e.response(),
    };
    let count = match body.get("count").and_then(Value::as_u64).and_then(|n| usize::try_from(n).ok()) {
        Some(n) => n,
        None => return ApiError::bad("count must be between 1 and 16", Some("count")).response(),
    };
    if !(1..=16).contains(&count) {
        return ApiError::bad("count must be between 1 and 16", Some("count")).response();
    }
    let (messages, parent_prompt, mut render, rev) = match st.sessions.as_ref().unwrap().snapshot(id) {
        Ok(s) => s,
        Err(e) => return e.response(),
    };
    render.insert("messages".into(), Value::Array(messages));
    render.insert("max_tokens".into(), Value::from(1));
    let raw = serde_json::to_vec(&render).unwrap_or_default();
    let candidate = match chat_prepare(st, &render, &raw).await {
        Ok(c) => c,
        Err(e) => return e.response(),
    };
    let prefix = parent_prompt.iter().zip(&candidate.prompt).take_while(|(a, b)| a == b).count();
    match st.sessions.as_ref().unwrap().fork(id, count, prefix, rev) {
        Ok(v) => json_value(200, v),
        Err(e) => e.response(),
    }
}

fn session_render(body: &Map<String, Value>) -> Map<String, Value> {
    const KEYS: &[&str] = &[
        "model",
        "tools",
        "tool_choice",
        "chat_template_kwargs",
        "documents",
        "reasoning_effort",
        "thinking",
        "enable_thinking",
        "add_generation_prompt",
        "continue_final_message",
        "response_format",
    ];
    KEYS.iter().filter_map(|&k| body.get(k).map(|v| (k.to_owned(), v.clone()))).collect()
}

// ---- /v1/chat/completions ----------------------------------------------------------------------------------------

struct ChatReq {
    tools: Option<Value>,
    choice: ToolChoice,
    /// what the insert rule puts in place of the reply's first token (empty: nothing is forced)
    force: Vec<u32>,
    prompt: Vec<u32>,
    params: sampling::Params,
    /// the thinking budget in reply tokens; 0: none
    budget: u64,
    /// the validated response_format ('text' enforces nothing)
    rf: jsonmode::Rf,
    /// whether thinking was explicitly turned off for this request
    thinking_off: bool,
    /// whether the rendered prompt leaves `<think>` open for the assistant turn
    prompt_think_open: bool,
    /// `continue_final_message`: the reply goes on from the prompt's last assistant message
    cfm_on: bool,
    admit: Admit,
}

async fn chat_prepare(st: &'static State, body: &Map<String, Value>, raw: &[u8]) -> Result<ChatReq, ApiError> {
    st.check_model(body)?;
    let admit = st.require_engine().await?;
    if truthy(body.get("echo")) {
        return Err(ApiError::bad("echo is not supported by the gr8 front end", Some("echo")));
    }
    let tools = normalize_tools(body.get("tools"))?;
    let choice = ToolChoice::parse(tools.as_ref(), body.get("tool_choice"))?;
    let rf = jsonmode::parse_response_format(body.get("response_format"), &st.grammars)?;
    let ds = st.dialect == Dialect::Ds41;
    let (mut th, ds_effort, hf_enable, ink_effort) = match st.dialect {
        Dialect::Ds41 => {
            let (th, e) = dialect::resolve_ds(body, st.thinking, st.ds_effort, st.ds_thinking_default)?;
            (th, e, None, None)
        }
        Dialect::Hf | Dialect::Qwen => {
            let (th, en) = dialect::resolve_hf(body, &st.tpl().levels, st.thinking)?;
            (th, st.ds_effort, en, None)
        }
        Dialect::Inkling => {
            let (th, e) = dialect::resolve_inkling(body, st.thinking, st.ink_effort.as_ref())?;
            (th, st.ds_effort, None, e)
        }
        Dialect::Glm => (st.tpl().resolve_thinking(body, st.thinking)?, st.ds_effort, None, None),
    };
    let agp_val = body.get("add_generation_prompt").cloned().unwrap_or(Value::Bool(true));
    let cfm_val = body.get("continue_final_message").cloned().unwrap_or(Value::Bool(false));
    let agp_on = truthy(Some(&agp_val));
    let cfm_on = truthy(Some(&cfm_val));
    let mut kw = if ds {
        Map::new() // the prompt is rendered by dschat from the messages alone
    } else {
        build_template_kwargs(body.get("chat_template_kwargs"), agp_val, cfm_val, th.effort, body.get("documents"))?
    };
    if let Some(en) = hf_enable {
        kw.insert("enable_thinking".into(), Value::Bool(en));
    }
    // Inkling's template takes the effort as a name or a number, and effort 0 is its thinking off
    if let Some(e) = ink_effort {
        kw.insert("reasoning_effort".into(), e);
    }
    if truthy(body.get("chat_template")) {
        return Err(ApiError::bad("custom chat_template is not supported", Some("chat_template")));
    }
    // a JSON mode asks the model for JSON up front (post-hoc extraction stays the enforcement)
    let messages = match jsonmode::response_format_hint(&rf) {
        Some(h) => append_hint(body.get("messages"), &h)?,
        None => body.get("messages").cloned().unwrap_or(Value::Array(Vec::new())),
    };
    // A forced call cannot generate reasoning before its token-zero insert. Use native chat mode unless the
    // request or its history needs thinking; keep that history's prompt prefix unchanged.
    if ds
        && !st.forced_tool_thinking
        && choice.is_forced()
        && !cfm_on
        && !th.off
        && !dialect::ds_explicit_thinking_on(body)
        && !dialect::ds_has_prior_reasoning(Some(&messages))
    {
        th.off = true;
    }
    let t0 = std::time::Instant::now();
    let kw_system = kw.clone();
    let mut images = Vec::new();
    let text = if ds {
        let (agp, cfm) = dialect::ds_gen_flags(body)?;
        let (text, im) = dialect::ds_prompt(Some(&messages), tools.as_ref(), !th.off, ds_effort, agp, cfm)?;
        images = im;
        text
    } else {
        // image parts: the template's placeholders and the parts in their order (GLM-5.3-Flash image input)
        let mut text = if crate::template::has_images(&messages) {
            let (text, im) = st.tpl().render_images(Some(&messages), tools.as_ref(), kw)?;
            images = im;
            text
        } else {
            st.tpl().render(Some(&messages), tools.as_ref(), kw)?
        };
        // GLM renders its lowest effort with the think block open, and Inkling's effort line says it: only an hf
        // template's open think block is closed here
        if th.off && matches!(st.dialect, Dialect::Hf | Dialect::Qwen) {
            text = dialect::hf_thinking_off(text, agp_on);
        }
        text
    };
    let hf_prompt_ends_think = matches!(st.dialect, Dialect::Hf | Dialect::Qwen) && !cfm_on && pyjson::py_rstrip(&text).ends_with("<think>");
    let thinking_off = th.off || (matches!(st.dialect, Dialect::Hf | Dialect::Qwen) && pyjson::py_rstrip(&text).ends_with("</think>"));
    let (t1, chars) = (std::time::Instant::now(), text.len());
    // The same Tok checkpoint cache serves chat, /tokenize previews and the shorter system-span query.
    // The DeepSeek prompt text opens with its BOS, so no special tokens are added to it
    let add_special = !ds && truthy(body.get("add_special_tokens"));
    let prompt = st.tok.aencode(text, add_special).await.map_err(|e| ApiError::new(500, &e, None))?;
    let t2 = std::time::Instant::now();
    let system = if ds {
        ds_system_span(st, body.get("messages"), tools.as_ref(), !th.off, ds_effort, &prompt).await
    } else {
        system_span(st, body.get("messages"), tools.as_ref(), kw_system, add_special, &prompt).await
    };
    let (prompt, system, spans) = if images.is_empty() {
        (prompt, system, Vec::new())
    } else {
        let model = image_model(st);
        let (ids, spans, n_spans) = with_images(st, model, &images, &prompt).await?;
        let system = system.map(|n| crate::vision::expanded_index(&prompt, n as usize, &n_spans, model.token()) as u64);
        (ids, system, spans)
    };
    log::debug!(
        "chat prompt: {chars} bytes -> {} tokens; template {:.2} ms, encode {:.2} ms, system span {system:?} tokens {:.2} ms",
        prompt.len(),
        (t1 - t0).as_secs_f64() * 1e3,
        (t2 - t1).as_secs_f64() * 1e3,
        t2.elapsed().as_secs_f64() * 1e3
    );
    let mut params = sampling::build(
        body,
        raw,
        prompt.len(),
        st.max_model_len(),
        &st.tok.eos_ids,
        st.tok.vocab_size,
        &st.tok.gen_defaults,
        &st.penalties,
        st.boost.as_ref(),
        st.sequence.as_ref(),
        st.chat_default_max_tokens(prompt.len()),
        st.clamp_max_tokens,
        st.max_logprobs(),
        true,
    )?;
    // a forced choice rides the engine's insert rule, and an inserted token has no logprob (PROTOCOL.md `insert`)
    if choice.is_forced() && params.gen.logprobs.is_some() {
        return Err(ApiError::bad("a forced tool_choice cannot be combined with logprobs", Some("tool_choice")));
    }
    // the close a budget puts in is an insert rule too, and a request carries at most one: one that asks for both is
    // refused, while the server's default budget steps aside, as it does for logprobs
    let budget = match (th.budget_tokens(params.gen.max_tokens), params.gen.logprobs.is_some() || choice.is_forced()) {
        (b, true) if b > 0 && th.budget_explicit => {
            return Err(ApiError::bad(
                if choice.is_forced() {
                    "thinking_budget cannot be combined with a forced tool_choice"
                } else {
                    "thinking_budget cannot be combined with logprobs"
                },
                Some("thinking_budget"),
            ))
        }
        (_, true) => 0,
        (b, false) => b,
    };
    if choice.is_forced() && !st.engine.inserts() {
        return Err(ApiError::bad("the engine does not support a forced tool_choice", Some("tool_choice")));
    }
    check_penalties(st, &prompt, &mut params.gen)?;
    let tok_think_open = match (st.tok.think_start, st.tok.think_end) {
        (Some(s), Some(e)) => think_open(&prompt, s, e),
        _ => false,
    };
    // An Inkling prompt ends at `<|end_message|>`: the model opens its own thinking block in the reply.
    let prompt_think_open = match st.dialect {
        Dialect::Hf | Dialect::Qwen if !cfm_on => hf_prompt_ends_think,
        Dialect::Inkling => false,
        _ => tok_think_open,
    };
    // The template leaves `<think>` open at the end of a chat prompt, so the reply's first token sits in the reasoning
    // span: the forced opener has to follow the close, or the call it opens is reasoning and the parser reads prose.
    let close = if prompt_think_open { st.tok.think_close.as_slice() } else { &[] };
    let force = match st.dialect {
        Dialect::Ds41 => choice.forced_ids_ds(&st.tok, close)?,
        Dialect::Hf => choice.forced_ids_hf(&st.tok, close)?,
        Dialect::Qwen => choice.forced_ids_qwen(&st.tok, close)?,
        Dialect::Inkling => choice.forced_ids_ink(tools.as_ref(), &st.tok, close)?,
        Dialect::Glm => choice.forced_ids(&st.tok, close)?,
    };
    params.gen.system_tokens = system;
    // a JSON mode with no tools: the engine masks the reply to the grammar (after the think close while the prompt leaves
    // the think block open); the extraction after generation still runs. With tools the grammar would forbid a call
    if rf.kind != "text" && tools.is_none() && st.engine.grammars() {
        let after_token = if prompt_think_open { st.tok.think_end } else { None };
        params.gen.grammar = Some(crate::wire::GrammarReq {
            kind: if rf.kind == "json_schema" { "json_schema" } else { "json" },
            schema: rf.schema.as_ref().map(Value::to_string),
            strict: rf.strict,
            after_token,
        });
    }
    (params.gen.steering_profile, params.gen.steering_scale) = sampling::steering(body, st.steering().as_ref())?;
    if !spans.is_empty() {
        // the engine runs image rows only through its prefill chunks, which prompt logprobs and steering do not use
        if params.gen.prompt_logprobs.is_some() {
            return Err(ApiError::bad("images cannot be combined with prompt_logprobs", Some("prompt_logprobs")));
        }
        if params.gen.steering_profile.is_some() {
            return Err(ApiError::bad("images cannot be combined with steering", Some("messages")));
        }
        params.gen.images = std::sync::Arc::new(spans);
    }
    Ok(ChatReq { tools, choice, force, prompt, params, budget, rf, thinking_off, prompt_think_open, cfm_on, admit })
}

fn image_err(e: crate::vision::VisionError) -> ApiError {
    ApiError::bad(e.0, Some("messages"))
}

/// The image processor and image token of the served model: DeepSeek-V4.1's, Inkling-2-Small's, else GLM-5.3-Flash's
/// (an engine without a vision tower refuses images before any is used).
fn image_model(st: &State) -> crate::vision::Model {
    match st.dialect {
        Dialect::Ds41 => crate::vision::Model::Ds41,
        Dialect::Inkling => crate::vision::Model::Ink2,
        Dialect::Qwen => crate::vision::Model::Qw38,
        _ => crate::vision::Model::Glm5v,
    }
}

/// The request's images (blocks in prompt order) -> the prompt with each placeholder token grown to its span, the
/// spans for the engine, and each span's length. Decoding and resizing run off the async threads.
async fn with_images(
    st: &'static State,
    model: crate::vision::Model,
    images: &[Value],
    prompt: &[u32],
) -> Result<(Vec<u32>, Vec<crate::vision::Span>, Vec<usize>), ApiError> {
    use crate::vision;
    if !st.engine.vision() {
        return Err(ApiError::bad("this engine takes no images", Some("messages")));
    }
    let b64: Vec<String> =
        images.iter().map(|b| vision::block_b64(b).map(str::to_string)).collect::<Result<_, _>>().map_err(image_err)?;
    vision::check_budget(&b64).map_err(image_err)?;
    let prepared =
        tokio::task::spawn_blocking(move || b64.iter().map(|s| model.prepare_b64(s)).collect::<Result<Vec<_>, _>>())
            .await
            .map_err(|e| ApiError::new(500, &e.to_string(), None))?
            .map_err(image_err)?;
    let n_spans: Vec<usize> = prepared.iter().map(|p| p.n_span).collect();
    let (ids, pos) = vision::expand(prompt, &n_spans, model.token()).map_err(image_err)?;
    let spans: Vec<vision::Span> = prepared
        .into_iter()
        .zip(pos)
        .map(|(p, pos)| vision::Span {
            pos,
            vit_h: p.vit_h,
            vit_w: p.vit_w,
            rgb_b64: vision::b64_encode(&p.rgb),
            key: p.key,
        })
        .collect();
    // Match frontend/g6s2shim.py IMAGES_B64_MAX so oversized decoded RGB payloads return 400 instead of shim 500.
    const IMAGES_RGB_B64_MAX: usize = 96 << 20;
    let total_rgb_b64: usize = spans.iter().map(|s| s.rgb_b64.len()).sum();
    if total_rgb_b64 > IMAGES_RGB_B64_MAX {
        return Err(ApiError::bad(
            &format!("decoded images exceed {} MiB base64 budget", IMAGES_RGB_B64_MAX >> 20),
            Some("messages"),
        ));
    }
    Ok((ids, spans, n_spans))
}

/// Append a one-line instruction after the last user or tool message (vLLM's chat-ml injection shape).
fn append_hint(messages: Option<&Value>, hint: &str) -> Result<Value, ApiError> {
    let Some(Value::Array(items)) = messages else {
        return Err(ApiError::bad("response_format needs at least one message", Some("response_format")));
    };
    let mut out = items.clone();
    for m in out.iter_mut().rev() {
        let role = m.get("role").and_then(Value::as_str);
        if !matches!(role, Some("user" | "tool")) {
            continue;
        }
        let Some(o) = m.as_object_mut() else { break };
        match o.get_mut("content") {
            Some(Value::String(s)) => *s = format!("{s}\n\n{hint}"),
            Some(Value::Array(parts)) => {
                let mut p = Map::new();
                p.insert("type".into(), Value::from("text"));
                p.insert("text".into(), Value::from(hint));
                parts.push(Value::Object(p));
            }
            None | Some(Value::Null) => {
                o.insert("content".into(), Value::from(hint));
            }
            _ => break, // a non-text content form the hint cannot join; render unchanged
        }
        return Ok(Value::Array(out));
    }
    Ok(Value::Array(out))
}

/// The prompt's system span (PROTOCOL.md `system_tokens`): its tokens up to where the first message after the leading
/// system ones begins, found by rendering those messages and the tools alone. The engine's KV stores keep system
/// prefixes by it. None when the template cannot render them alone.
async fn system_span(
    st: &'static State,
    messages: Option<&Value>,
    tools: Option<&Value>,
    mut kw: Map<String, Value>,
    add_special: bool,
    prompt: &[u32],
) -> Option<u64> {
    let lead: Vec<Value> = messages?
        .as_array()?
        .iter()
        .take_while(|m| matches!(m.get("role").and_then(Value::as_str), Some("system" | "developer")))
        .cloned()
        .collect();
    if lead.is_empty() && tools.is_none() {
        return Some(0);
    }
    kw.insert("add_generation_prompt".into(), Value::Bool(false));
    kw.insert("continue_final_message".into(), Value::Bool(false));
    let text = st.tpl().render(Some(&Value::Array(lead)), tools, kw).ok()?;
    let ids = st.tok.aencode(text, add_special).await.ok()?;
    Some(ids.iter().zip(prompt).take_while(|(a, b)| a == b).count() as u64)
}

/// `system_span` for the ds41 dialect: the leading system messages and the tools rendered alone, and how many of their
/// tokens open the whole prompt.
async fn ds_system_span(
    st: &'static State,
    messages: Option<&Value>,
    tools: Option<&Value>,
    thinking: bool,
    effort: u32,
    prompt: &[u32],
) -> Option<u64> {
    let has_lead = messages?
        .as_array()?
        .first()
        .is_some_and(|m| matches!(m.get("role").and_then(Value::as_str), Some("system" | "developer")));
    if !has_lead && tools.is_none() {
        return Some(0);
    }
    let text = dialect::ds_system_text(messages, tools, thinking, effort)?;
    let ids = st.tok.aencode(text, false).await.ok()?;
    Some(ids.iter().zip(prompt).take_while(|(a, b)| a == b).count() as u64)
}

async fn chat(st: &'static State, req: Request<Incoming>) -> Response<HttpBody> {
    let (mut body, raw) = match read_json_raw(req).await {
        Ok(b) => b,
        Err(e) => return e.response(),
    };
    let mut session: Option<(String, bool, Vec<Value>)> = None;
    let create = st.sessions.is_some()
        && body.get("session").and_then(Value::as_object).and_then(|v| v.get("create")) == Some(&Value::Bool(true));
    if let Some(store) = &st.sessions {
        if create && body.contains_key("session_id") {
            return ApiError::bad("session.create and session_id are exclusive", Some("session")).response();
        }
        if let Some(id) = body.get("session_id").and_then(Value::as_str).map(str::to_owned) {
            let append = match body.get("append_messages").and_then(Value::as_array) {
                Some(a) if !a.is_empty() => a.clone(),
                _ => {
                    return ApiError::bad("append_messages must be a nonempty array", Some("append_messages"))
                        .response()
                }
            };
            let model = body.get("model").and_then(Value::as_str);
            let (messages, render) = match store.begin(&id, &append, model) {
                Ok(m) => m,
                Err(e) => return e.response(),
            };
            for (k, v) in render {
                body.entry(k).or_insert(v);
            }
            body.insert("messages".into(), Value::Array(messages));
            session = Some((id, false, append));
        } else if create {
            let messages = match body.get("messages").and_then(Value::as_array) {
                Some(a) => a.clone(),
                None => return ApiError::bad("messages must be an array", Some("messages")).response(),
            };
            session = Some((String::new(), true, messages));
        }
    }
    let cr = match chat_prepare(st, &body, &raw).await {
        Ok(c) => c,
        Err(e) => {
            if let Some((id, false, _)) = &session {
                st.sessions.as_ref().unwrap().abort(id, false);
            }
            return e.response();
        }
    };
    if let Some((id, true, messages)) = &mut session {
        *id = st.sessions.as_ref().unwrap().create(
            messages.clone(),
            cr.prompt.clone(),
            body.get("model").and_then(Value::as_str).unwrap_or(&st.model_name).to_owned(),
            session_render(&body),
        );
    }
    let rid = new_id("chatcmpl");
    // thinking off withholds the reasoning, whatever the model still writes in its think block
    let include_reasoning = !cr.thinking_off && body.get("include_reasoning").is_none_or(|v| truthy(Some(v)));
    let choice_value = cr.choice.as_value();
    let content_first = match (st.tok.think_start, st.tok.think_end) {
        (Some(start), Some(end)) => !think_open(&cr.prompt, start, end),
        _ => false,
    };
    let prompt_think_open = cr.prompt_think_open;
    let thinking_off = cr.thinking_off;
    // the repeated-tool-call guard (guard.rs): GLM's parser, tools offered, no forced call (that is the client's own)
    let guard_on = body.get("tool_repeat_guard").filter(|v| !v.is_null()).map_or(st.guard.on, |v| truthy(Some(v)));
    let mut guard = (st.dialect == Dialect::Glm && cr.tools.is_some() && cr.force.is_empty())
        .then(|| guard::Guard::new(st.guard, guard_on, body.get("messages")))
        .flatten();
    let guard_names = guard.as_ref().map(guard::Guard::names).unwrap_or_default();
    let new_parser = || {
        let p = AnyParser::new(st.dialect, cr.tools.as_ref(), &choice_value, include_reasoning)
            .emit_undeclared(st.emit_undeclared)
            .guard_names(guard_names.clone());
        if matches!(st.dialect, Dialect::Hf | Dialect::Qwen) {
            if prompt_think_open {
                p.in_think()
            } else if thinking_off {
                p.content_first()
            } else {
                p
            }
        } else if st.dialect == Dialect::Inkling {
            // the reply names its own blocks; only a continued message starts inside one (its text)
            if cr.cfm_on {
                p.content_first()
            } else {
                p
            }
        } else if content_first {
            p.content_first()
        } else {
            p
        }
    };
    let parser = new_parser();
    let stream = truthy(body.get("stream"));
    // the GLM reasoning parser forces skip_special_tokens=False (vLLM parser_engine.adjust_request)
    let mut gen = Generation::new(&st.tok, rid, cr.prompt, cr.params, false, !stream);
    gen.admit = Some(cr.admit);
    if truthy(body.get("return_step_stats")) {
        gen.want_step_stats();
    }
    if guard.is_some() {
        gen.keep_reply();
    }
    let rf = cr.rf;
    if st.dialect == Dialect::Inkling {
        // The model opens its thinking block itself (`<|message_model|><|content_thinking|>`, two tokens), so the
        // budget counts from the reply's start and is at least 3: the close then lands inside the block. A reply that
        // ended its thinking earlier has emitted `<|end_message|>`, and the rule does not fire.
        let limit = usize::try_from(cr.budget).unwrap_or(usize::MAX);
        let on = !thinking_off && !cr.cfm_on && cr.force.is_empty() && limit > 0 && !st.tok.think_close.is_empty();
        if let Some(end) = st.tok.think_end.filter(|_| on) {
            let limit = limit.max(3);
            if st.engine.inserts() {
                gen.params.gen.insert = Some(InsertRule {
                    after_tokens: limit as u64 - 1,
                    unless_emitted: Some(end),
                    tokens: st.tok.think_close.clone(),
                });
                gen.thinking(end, None);
            } else {
                gen.thinking(end, Some(Budget { limit, close: &st.tok.think_close }));
            }
        }
    } else if prompt_think_open {
        if let Some(end) = st.tok.think_end {
            let limit = usize::try_from(cr.budget).unwrap_or(usize::MAX);
            if limit > 0 && !st.tok.think_close.is_empty() && st.engine.inserts() {
                // the engine puts the close in place of the budget-th reply token itself: no second run
                gen.params.gen.insert = Some(InsertRule {
                    after_tokens: limit as u64 - 1,
                    unless_emitted: Some(end),
                    tokens: st.tok.think_close.clone(),
                });
                gen.thinking(end, None);
            } else {
                gen.thinking(end, (limit > 0).then_some(Budget { limit, close: &st.tok.think_close }));
            }
        }
    }
    if !cr.force.is_empty() {
        // `tool_choice: required` (or a named function): the engine puts the thinking close and the opener -- and the
        // name -- in place of the reply's first token, so the reply is a call and cannot be text. A budget cannot also
        // be in force here (the combination is refused above), so this is the request's one rule, and it replaces the
        // thinking state that block set: the close is in the insert now.
        gen.params.gen.insert = Some(InsertRule { after_tokens: 0, unless_emitted: None, tokens: cr.force.clone() });
    }
    if stream {
        // the engine's refusal (the busy guard's HTTP 429, PROTOCOL.md finish.status) and any error before the first
        // streamed event must be an HTTP status, not an SSE event after the 200 is out: the response starts only once
        // the engine answered something (PROTOCOL.md "the front end fails them with HTTP 500 / a stream error chunk")
        if let Err(e) = gen.start(&st.engine) {
            if let Some((id, created, _)) = &session {
                st.sessions.as_ref().unwrap().abort(id, *created);
            }
            return e.response();
        }
        let pre = match gen.next().await {
            Ok(p) => p,
            Err(e) => {
                if let Some((id, created, _)) = &session {
                    st.sessions.as_ref().unwrap().abort(id, *created);
                }
                return e.response();
            }
        };
        if pre.as_ref().is_some_and(|s| !s.ids.is_empty()) {
            if let Some((id, _, _)) = &session {
                st.sessions.as_ref().unwrap().ready(id);
            }
        }
        let session_parser = session.as_ref().map(|_| new_parser());
        let (sse, mut resp) = Sse::open();
        if let Some((id, _, _)) = &session {
            if let Ok(v) = header::HeaderValue::from_str(id) {
                resp.headers_mut().insert("x-session-id", v);
            }
        }
        tokio::spawn(
            async move { chat_stream(st, body, gen, parser, guard, rf, sse, pre, session, session_parser).await },
        );
        return resp;
    }
    let created = now_secs();
    // a second parser watches the reply as it arrives, so a closed call block ends the generation (chat_full parses the
    // whole text afterwards, which reads the same call), and the guard reads a held call before the reply ends
    let mut watch = (st.dialect == Dialect::Ds41 || guard.is_some()).then(new_parser);
    if let Err(e) = run_to_end(st, &mut gen, watch.as_mut(), guard.as_mut()).await {
        if let Some((id, created, _)) = &session {
            st.sessions.as_ref().unwrap().abort(id, *created);
        }
        return e.response();
    }
    let resp = chat_full(st, &body, &gen, parser, &rf, created, session.as_ref().map(|x| x.0.as_str()));
    if let Some((id, made, append)) = &session {
        if let Some(message) = response_message(&resp) {
            st.sessions.as_ref().unwrap().finish(
                id,
                if *made { &[] } else { append },
                message,
                gen.prompt.clone(),
                session_render(&body),
            );
        } else {
            st.sessions.as_ref().unwrap().abort(id, *made);
        }
    }
    resp
}

fn response_message(resp: &Response<HttpBody>) -> Option<Value> {
    let HttpBody::Full(Some(bytes)) = resp.body() else { return None };
    let v: Value = serde_json::from_slice(bytes).ok()?;
    v.get("choices")?.as_array()?.first()?.get("message").cloned()
}

async fn run_to_end(
    st: &State,
    gen: &mut Generation,
    mut watch: Option<&mut AnyParser>,
    mut guard: Option<&mut guard::Guard>,
) -> Result<(), ApiError> {
    gen.start(&st.engine)?;
    while let Some(step) = gen.next().await? {
        if let Some(w) = watch.as_deref_mut() {
            let finished = step.finish_reason.is_some();
            if finished {
                if let (Some(g), Some(p)) = (guard.as_deref_mut(), w.glm_mut()) {
                    p.delta(&step.text, false);
                    if guard::take_back(gen, p, g).await? {
                        continue; // the reply goes on from before the call
                    }
                    p.delta("", true);
                    continue;
                }
            }
            w.delta(&step.text, finished);
            if !finished && w.call_closed() {
                gen.end_now();
                break;
            }
        }
    }
    gen.finish_stats().await;
    Ok(())
}

fn reasoning_fields(st: &State, o: &mut Obj, r: Option<&str>) {
    for f in st.reasoning_fields {
        o.opt_str(f, r);
    }
}

fn chat_full(
    st: &State,
    body: &Map<String, Value>,
    gen: &Generation,
    mut parser: AnyParser,
    rf: &jsonmode::Rf,
    created: u64,
    session_id: Option<&str>,
) -> Response<HttpBody> {
    let res = &gen.res;
    let mut p = parser.parse_full(&res.text);
    let mut warnings: Vec<String> = Vec::new();
    if p.calls.is_empty() {
        let (content, warns) = jsonmode::enforce(p.content.as_deref(), rf);
        p.content = content;
        warnings = warns;
    }
    let reasoning = if parser.include_reasoning() { p.reasoning } else { None };
    let mut o = Vec::with_capacity(1024 + res.text.len() * 2);
    let mut c = head(&mut o, &gen.id, "chat.completion", created, &st.model_name);
    {
        let ch = c.key("choices");
        ch.push(b'[');
        let mut x = Obj::new(ch);
        x.int("index", 0);
        {
            let m = x.key("message");
            let mut msg = Obj::new(m);
            msg.str("role", "assistant");
            msg.opt_str("content", p.content.as_deref());
            for k in ["refusal", "annotations", "audio", "function_call"] {
                msg.raw(k, b"null");
            }
            if !p.calls.is_empty() {
                let tc = msg.key("tool_calls");
                tc.push(b'[');
                for (i, (id, name, args)) in p.calls.iter().enumerate() {
                    if i > 0 {
                        tc.push(b',');
                    }
                    let mut t = Obj::new(tc);
                    t.str("id", id);
                    t.str("type", "function");
                    let f = t.key("function");
                    let mut fo = Obj::new(f);
                    fo.str("name", name);
                    fo.str("arguments", args);
                    fo.end();
                    t.end();
                }
                tc.push(b']');
            }
            reasoning_fields(st, &mut msg, reasoning.as_deref());
            msg.end();
        }
        match gen.params.gen.logprobs {
            Some(k) => {
                let as_ids = truthy(body.get("return_tokens_as_token_ids"));
                L::chat_logprobs(x.key("logprobs"), &res.ids, &res.logprobs, k, &st.tok, as_ids);
            }
            None => x.raw("logprobs", b"null"),
        }
        x.str("finish_reason", finish_with_calls(!p.calls.is_empty(), res.finish_reason.unwrap_or("stop")));
        stop_reason_json(x.key("stop_reason"), &res.stop_reason);
        let ret_ids = truthy(body.get("return_token_ids"));
        if ret_ids {
            ids_json(x.key("token_ids"), &res.ids);
        } else {
            x.raw("token_ids", b"null");
        }
        x.end();
        ch.push(b']');
    }
    c.raw("service_tier", b"null");
    c.raw("system_fingerprint", b"null");
    usage(
        c.key("usage"),
        res.prompt_tokens,
        res.completion_tokens,
        Some(res.cached_tokens),
        st.prompt_tokens_details,
        true,
        Some(res.reasoning_tokens),
        res.step_stats.as_ref(),
    );
    if let Err(e) = L::prompt_logprobs(c.key("prompt_logprobs"), res.prompt_logprobs.as_deref(), &gen.prompt, &st.tok) {
        return ApiError::new(500, &e, None).response();
    }
    if truthy(body.get("return_token_ids")) {
        ids_json(c.key("prompt_token_ids"), &gen.prompt);
    } else {
        c.raw("prompt_token_ids", b"null");
    }
    c.raw("kv_transfer_params", b"null");
    if let Some(id) = session_id {
        c.str("session_id", id);
    }
    if !warnings.is_empty() {
        let w = c.key("warnings");
        w.push(b'[');
        for (i, s) in warnings.iter().enumerate() {
            if i > 0 {
                w.push(b',');
            }
            pyjson::string(w, s);
        }
        w.push(b']');
    }
    c.end();
    json_response(200, o)
}

fn delta_json(st: &State, o: &mut Vec<u8>, d: Option<&Delta>) {
    let mut x = Obj::new(o);
    if let Some(d) = d {
        if let Some(c) = &d.content {
            x.str("content", c);
        }
        if !d.tool_calls.is_empty() {
            let tc = x.key("tool_calls");
            tc.push(b'[');
            for (i, t) in d.tool_calls.iter().enumerate() {
                if i > 0 {
                    tc.push(b',');
                }
                let mut to = Obj::new(tc);
                to.int("index", t.index);
                if let Some(id) = &t.id {
                    to.str("id", id);
                    to.str("type", "function");
                }
                let f = to.key("function");
                let mut fo = Obj::new(f);
                if let Some(n) = &t.name {
                    fo.str("name", n);
                }
                if let Some(a) = &t.arguments {
                    fo.str("arguments", a);
                }
                fo.end();
                to.end();
            }
            tc.push(b']');
        }
        if let Some(r) = &d.reasoning {
            for f in st.reasoning_fields {
                x.str(f, r);
            }
        }
    }
    x.end();
}

async fn chat_stream(
    st: &'static State,
    body: Map<String, Value>,
    mut gen: Generation,
    mut parser: AnyParser,
    mut guard: Option<guard::Guard>,
    rf: jsonmode::Rf,
    sse: Sse,
    mut pre: Option<Step>,
    session: Option<(String, bool, Vec<Value>)>,
    mut session_parser: Option<AnyParser>,
) {
    let created = now_secs();
    let so = body.get("stream_options").and_then(Value::as_object);
    let include_usage = so.is_some_and(|s| truthy(s.get("include_usage")));
    let continuous = include_usage && so.is_some_and(|s| truthy(s.get("continuous_usage_stats")));
    let ret_ids = truthy(body.get("return_token_ids"));
    let as_ids = truthy(body.get("return_tokens_as_token_ids"));
    let lp_k = gen.params.gen.logprobs;
    let mut first = true;
    let mut tools_streamed = false;
    let json_mode = rf.kind != "text";
    let mut held_content = String::new(); // JSON mode: content is withheld until the end, then extracted
    let mut committed = false;
    let r: Result<(), ApiError> = async {
        // the engine request is already started and its first frame is `pre` (the handler held the HTTP response back
        // for it, so a refusal arrives as the status); None = the engine finished without a token
        loop {
            // a client that goes away while nothing is being sent (prefill, held-back text) still cancels at once
            let step = match pre.take() {
                Some(s) => Some(s),
                None => tokio::select! {
                    biased;
                    _ = sse.0.closed() => return Ok(()),
                    s = gen.next() => s?,
                },
            };
            let Some(step) = step else { break };
            if !step.ids.is_empty() {
                if let Some((id, _, _)) = &session { st.sessions.as_ref().unwrap().ready(id); }
            }
            if first {
                first = false;
                let mut o = Vec::with_capacity(256);
                let mut c = head(&mut o, &gen.id, "chat.completion.chunk", created, &st.model_name);
                c.raw(
                    "choices",
                    b"[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"logprobs\":null,\"finish_reason\":null}]",
                );
                if ret_ids {
                    ids_json(c.key("prompt_token_ids"), &gen.prompt);
                }
                if let Some((id, _, _)) = &session { c.str("session_id", id); }
                if continuous {
                    usage(c.key("usage"), gen.res.prompt_tokens, 0, None, false, false, None, None);
                }
                c.end();
                if sse.json(o).await.is_err() {
                    return Ok(());
                }
            }
            let last = step.finish_reason.is_some();
            // the guard: the final text goes to the parser short of `finished`, so a held call stays held while the
            // guard reads it. Taken back, the reply goes on. Otherwise an empty final step lets the call out
            let step = match (last, guard.as_mut(), parser.glm_mut()) {
                (true, Some(g), Some(_)) => {
                    let pre = Step { text: step.text.clone(), ids: step.ids.clone(), logprobs: step.logprobs.clone(), ..Default::default() };
                    if chat_step(st, &gen, &mut parser, &pre, &sse, created, lp_k, ret_ids, as_ids, continuous,
                                 &mut tools_streamed, json_mode, &mut held_content, &rf)
                        .await
                        .is_err()
                    {
                        return Ok(());
                    }
                    if guard::take_back(&mut gen, parser.glm_mut().unwrap(), g).await? {
                        continue;
                    }
                    Step { finish_reason: step.finish_reason, stop_reason: step.stop_reason, ..Default::default() }
                }
                _ => step,
            };
            if chat_step(st, &gen, &mut parser, &step, &sse, created, lp_k, ret_ids, as_ids, continuous,
                         &mut tools_streamed, json_mode, &mut held_content, &rf)
                .await
                .is_err()
            {
                return Ok(()); // client went away; dropping `gen` cancels the engine request
            }
            if !last && parser.call_closed() {
                // the call block is complete: end the generation here, whatever the model would write next
                let end = gen.end_now();
                if chat_step(st, &gen, &mut parser, &end, &sse, created, lp_k, ret_ids, as_ids, continuous,
                             &mut tools_streamed, json_mode, &mut held_content, &rf)
                    .await
                    .is_err()
                {
                    return Ok(());
                }
                break;
            }
        }
        gen.finish_stats().await;
        if let Some((id, made, append)) = &session {
            if let Some(p) = session_parser.as_mut() {
                let parsed = p.parse_full(&gen.res.text);
                let mut message = Map::new();
                message.insert("role".into(), Value::String("assistant".into()));
                message.insert("content".into(), parsed.content.map(Value::String).unwrap_or(Value::Null));
                if !parsed.calls.is_empty() {
                    message.insert("tool_calls".into(), Value::Array(parsed.calls.into_iter().map(|(id, name, arguments)|
                        serde_json::json!({"id": id, "type": "function", "function": {"name": name, "arguments": arguments}})).collect()));
                }
                st.sessions.as_ref().unwrap().finish(id, if *made { &[] } else { append }, Value::Object(message), gen.prompt.clone(), session_render(&body));
                committed = true;
            }
        }
        if include_usage {
            let res = &gen.res;
            let mut o = Vec::with_capacity(256);
            let mut c = head(&mut o, &gen.id, "chat.completion.chunk", created, &st.model_name);
            c.raw("choices", b"[]");
            usage(
                c.key("usage"),
                res.prompt_tokens,
                res.completion_tokens,
                Some(res.cached_tokens),
                st.prompt_tokens_details,
                false,
                Some(res.reasoning_tokens),
                res.step_stats.as_ref(),
            );
            c.end();
            if sse.json(o).await.is_err() {
                return Ok(());
            }
        }
        Ok(())
    }
    .await;
    if !committed {
        if let Some((id, made, _)) = &session {
            st.sessions.as_ref().unwrap().abort(id, *made);
        }
    }
    if let Err(e) = r {
        if sse.send(e.sse()).await.is_err() {
            return;
        }
    }
    if sse.0.is_closed() {
        log::info!("{}: client went away", gen.id);
        return;
    }
    let _ = sse.send(Bytes::from_static(b"data: [DONE]\n\n")).await;
}

#[allow(clippy::too_many_arguments)]
async fn chat_step(
    st: &State,
    gen: &Generation,
    parser: &mut AnyParser,
    step: &Step,
    sse: &Sse,
    created: u64,
    lp_k: Option<i64>,
    ret_ids: bool,
    as_ids: bool,
    continuous: bool,
    tools_streamed: &mut bool,
    json_mode: bool,
    held_content: &mut String,
    rf: &jsonmode::Rf,
) -> Result<(), ()> {
    let finished = step.finish_reason.is_some();
    let was_reasoning = parser.in_reasoning();
    let mut delta = parser.delta(&step.text, finished);
    let has_visible = delta.is_some();
    let first_call = !*tools_streamed && delta.as_ref().is_some_and(|d| !d.tool_calls.is_empty());
    if first_call {
        *tools_streamed = true;
        // a call ends JSON mode: the withheld text goes out ahead of it unextracted, as chat_full keeps it
        if let Some(d) = delta.as_mut().filter(|_| json_mode && !held_content.is_empty()) {
            d.content = Some(std::mem::take(held_content) + d.content.as_deref().unwrap_or(""));
        }
    }
    let mut warnings: Option<Vec<String>> = None;
    if json_mode && !*tools_streamed {
        // response_format JSON mode: content is withheld until the end, then extracted once
        if let Some(d) = &mut delta {
            if let Some(c) = d.content.take() {
                held_content.push_str(&c);
            }
            // an all-buffered step leaves nothing to stream; skip the chunk like the Python front end
            if d.reasoning.is_none() && d.tool_calls.is_empty() && !finished {
                delta = None;
            }
        }
        if finished {
            let (content, w): (Option<String>, Vec<String>) =
                jsonmode::enforce((!held_content.is_empty()).then_some(held_content.as_str()), rf);
            if let Some(c) = content {
                let d = delta.get_or_insert_with(crate::parser::Delta::default);
                d.content = Some(c);
            }
            if !w.is_empty() {
                warnings = Some(w);
            }
        }
    }
    let want_lp = lp_k.is_some()
        && !step.ids.is_empty()
        && (parser.include_reasoning() || has_visible || (!was_reasoning && !parser.in_reasoning()));
    if delta.is_none() && !finished && !want_lp && !ret_ids {
        return Ok(());
    }
    let mut o = Vec::with_capacity(256 + step.text.len() * 2);
    let mut c = head(&mut o, &gen.id, "chat.completion.chunk", created, &st.model_name);
    {
        let ch = c.key("choices");
        ch.push(b'[');
        let mut x = Obj::new(ch);
        x.int("index", 0);
        delta_json(st, x.key("delta"), delta.as_ref());
        if want_lp {
            L::chat_logprobs(x.key("logprobs"), &step.ids, &step.logprobs, lp_k.unwrap(), &st.tok, as_ids);
        } else {
            x.raw("logprobs", b"null");
        }
        match step.finish_reason {
            Some(fr) => x.str("finish_reason", finish_with_calls(*tools_streamed, fr)),
            None => x.raw("finish_reason", b"null"),
        }
        if ret_ids {
            ids_json(x.key("token_ids"), &step.ids);
        } else {
            x.raw("token_ids", b"null");
        }
        if finished {
            stop_reason_json(x.key("stop_reason"), &step.stop_reason);
        }
        if let Some(ws) = &warnings {
            let w = x.key("warnings");
            w.push(b'[');
            for (i, s) in ws.iter().enumerate() {
                if i > 0 {
                    w.push(b',');
                }
                pyjson::string(w, s);
            }
            w.push(b']');
        }
        x.end();
        ch.push(b']');
    }
    if continuous {
        usage(c.key("usage"), gen.res.prompt_tokens, gen.res.completion_tokens, None, false, false, None, None);
    }
    c.end();
    sse.json(o).await
}

// ---- /v1/completions ---------------------------------------------------------------------------------------------

async fn prompt_ids(st: &'static State, body: &Map<String, Value>) -> Result<Vec<u32>, ApiError> {
    let mut p = body.get("prompt");
    if let Some(Value::Array(a)) = p {
        if a.len() == 1 && (a[0].is_string() || a[0].is_array()) {
            p = Some(&a[0]); // a batch of one
        }
    }
    match p {
        Some(Value::String(s)) => {
            let add_special = body.get("add_special_tokens").is_none_or(|v| truthy(Some(v)));
            st.tok.aencode(s.clone(), add_special).await.map_err(|e| ApiError::new(500, &e, None))
        }
        Some(Value::Array(a)) if !a.is_empty() && a.iter().all(|i| i.is_i64() || i.is_u64()) => {
            let mut ids = Vec::with_capacity(a.len());
            for i in a {
                match i.as_i64() {
                    Some(i) if i >= 0 && (i as u64) < st.tok.vocab_size as u64 => ids.push(i as u32),
                    _ => {
                        return Err(ApiError::bad("prompt contains a token id out of vocabulary range", Some("prompt")))
                    }
                }
            }
            Ok(ids)
        }
        Some(Value::Array(a)) if a.len() > 1 => Err(ApiError::bad(
            "batched prompts are not supported by the gr8 front end; send one prompt per request",
            Some("prompt"),
        )),
        _ => Err(ApiError::bad("`prompt` must be a string or a non-empty list of token ids", Some("prompt"))),
    }
}

async fn completions(st: &'static State, req: Request<Incoming>) -> Response<HttpBody> {
    let (body, raw) = match read_json_raw(req).await {
        Ok(b) => b,
        Err(e) => return e.response(),
    };
    let prep = async {
        st.check_model(&body)?;
        let admit = st.require_engine().await?;
        for k in ["echo", "suffix"] {
            if truthy(body.get(k)) {
                return Err(ApiError::bad(format!("`{k}` is not supported by the gr8 front end"), Some(k)));
            }
        }
        thinking::check_response_format(body.get("response_format"))?;
        let prompt = prompt_ids(st, &body).await?;
        let mut params = sampling::build(
            &body,
            &raw,
            prompt.len(),
            st.max_model_len(),
            &st.tok.eos_ids,
            st.tok.vocab_size,
            &st.tok.gen_defaults,
            &st.penalties,
            st.boost.as_ref(),
            st.sequence.as_ref(),
            Some(16),
            st.clamp_max_tokens,
            st.max_logprobs(),
            false,
        )?;
        (params.gen.steering_profile, params.gen.steering_scale) = sampling::steering(&body, st.steering().as_ref())?;
        check_penalties(st, &prompt, &mut params.gen)?;
        Ok((prompt, params, admit))
    };
    let (prompt, params, admit) = match prep.await {
        Ok(x) => x,
        Err(e) => return e.response(),
    };
    let rid = new_id("cmpl");
    let skip_special = body.get("skip_special_tokens").is_none_or(|v| truthy(Some(v)));
    let stream = truthy(body.get("stream"));
    let mut gen = Generation::new(&st.tok, rid, prompt, params, skip_special, !stream);
    gen.admit = Some(admit);
    if truthy(body.get("return_step_stats")) {
        gen.want_step_stats();
    }
    if stream {
        // the busy guard's refusal and any error before the first streamed event must be an HTTP status (see chat)
        if let Err(e) = gen.start(&st.engine) {
            return e.response();
        }
        let pre = match gen.next().await {
            Ok(p) => p,
            Err(e) => return e.response(),
        };
        let (sse, resp) = Sse::open();
        tokio::spawn(async move { completions_stream(st, body, gen, sse, pre).await });
        return resp;
    }
    let created = now_secs();
    if let Err(e) = run_to_end(st, &mut gen, None, None).await {
        return e.response();
    }
    completions_full(st, &body, &gen, created)
}

fn completions_full(st: &State, body: &Map<String, Value>, gen: &Generation, created: u64) -> Response<HttpBody> {
    let res: &Res = &gen.res;
    let ret_ids = truthy(body.get("return_token_ids"));
    let mut o = Vec::with_capacity(1024 + res.text.len() * 2);
    let mut c = head(&mut o, &gen.id, "text_completion", created, &st.model_name);
    {
        let ch = c.key("choices");
        ch.push(b'[');
        let mut x = Obj::new(ch);
        x.int("index", 0);
        x.str("text", &res.text);
        match gen.params.gen.logprobs {
            Some(k) => {
                let as_ids = truthy(body.get("return_tokens_as_token_ids"));
                L::completion_logprobs(x.key("logprobs"), &res.ids, &res.logprobs, k, &st.tok, as_ids, 0);
            }
            None => x.raw("logprobs", b"null"),
        }
        x.opt_str("finish_reason", res.finish_reason);
        stop_reason_json(x.key("stop_reason"), &res.stop_reason);
        if ret_ids {
            ids_json(x.key("token_ids"), &res.ids);
        } else {
            x.raw("token_ids", b"null");
        }
        if let Err(e) =
            L::prompt_logprobs(x.key("prompt_logprobs"), res.prompt_logprobs.as_deref(), &gen.prompt, &st.tok)
        {
            return ApiError::new(500, &e, None).response();
        }
        if ret_ids {
            ids_json(x.key("prompt_token_ids"), &gen.prompt);
        } else {
            x.raw("prompt_token_ids", b"null");
        }
        x.end();
        ch.push(b']');
    }
    c.raw("service_tier", b"null");
    c.raw("system_fingerprint", b"null");
    usage(
        c.key("usage"),
        res.prompt_tokens,
        res.completion_tokens,
        Some(res.cached_tokens),
        st.prompt_tokens_details,
        true,
        None,
        res.step_stats.as_ref(),
    );
    c.raw("kv_transfer_params", b"null");
    c.end();
    json_response(200, o)
}

async fn completions_stream(
    st: &'static State,
    body: Map<String, Value>,
    mut gen: Generation,
    sse: Sse,
    mut pre: Option<Step>,
) {
    let created = now_secs();
    let so = body.get("stream_options").and_then(Value::as_object);
    let include_usage = so.is_some_and(|s| truthy(s.get("include_usage")));
    let continuous = include_usage && so.is_some_and(|s| truthy(s.get("continuous_usage_stats")));
    let as_ids = truthy(body.get("return_tokens_as_token_ids"));
    let ret_ids = truthy(body.get("return_token_ids"));
    let lp_k = gen.params.gen.logprobs;
    let (mut offset, mut first) = (0usize, true);
    let r: Result<(), ApiError> = async {
        // the engine request is already started and its first frame is `pre` (see chat_stream)
        loop {
            // a client that goes away while nothing is being sent (prefill, held-back text) still cancels at once
            let step = match pre.take() {
                Some(s) => Some(s),
                None => tokio::select! {
                    biased;
                    _ = sse.0.closed() => return Ok(()),
                    s = gen.next() => s?,
                },
            };
            let Some(step) = step else { break };
            let finished = step.finish_reason.is_some();
            let want_lp = lp_k.is_some() && !step.ids.is_empty();
            if step.text.is_empty() && !finished && !want_lp {
                continue;
            }
            let mut o = Vec::with_capacity(256 + step.text.len() * 2);
            let mut c = head(&mut o, &gen.id, "text_completion", created, &st.model_name);
            {
                let ch = c.key("choices");
                ch.push(b'[');
                let mut x = Obj::new(ch);
                x.int("index", 0);
                x.str("text", &step.text);
                if want_lp {
                    offset = L::completion_logprobs(
                        x.key("logprobs"),
                        &step.ids,
                        &step.logprobs,
                        lp_k.unwrap(),
                        &st.tok,
                        as_ids,
                        offset,
                    );
                } else {
                    x.raw("logprobs", b"null");
                }
                x.opt_str("finish_reason", step.finish_reason);
                stop_reason_json(x.key("stop_reason"), &step.stop_reason);
                if ret_ids {
                    ids_json(x.key("token_ids"), &step.ids);
                    if first {
                        ids_json(x.key("prompt_token_ids"), &gen.prompt);
                    }
                }
                if first && gen.res.prompt_logprobs.is_some() {
                    L::prompt_logprobs(
                        x.key("prompt_logprobs"),
                        gen.res.prompt_logprobs.as_deref(),
                        &gen.prompt,
                        &st.tok,
                    )
                    .map_err(|e| ApiError::new(500, &e, None))?;
                }
                x.end();
                ch.push(b']');
            }
            first = false;
            if continuous {
                usage(c.key("usage"), gen.res.prompt_tokens, gen.res.completion_tokens, None, false, false, None, None);
            }
            c.end();
            if sse.json(o).await.is_err() {
                return Ok(());
            }
        }
        gen.finish_stats().await;
        if include_usage {
            let res = &gen.res;
            let mut o = Vec::with_capacity(256);
            let mut c = head(&mut o, &gen.id, "text_completion", created, &st.model_name);
            c.raw("choices", b"[]");
            usage(
                c.key("usage"),
                res.prompt_tokens,
                res.completion_tokens,
                Some(res.cached_tokens),
                st.prompt_tokens_details,
                false,
                None,
                res.step_stats.as_ref(),
            );
            c.end();
            if sse.json(o).await.is_err() {
                return Ok(());
            }
        }
        Ok(())
    }
    .await;
    if let Err(e) = r {
        if sse.send(e.sse()).await.is_err() {
            return;
        }
    }
    if sse.0.is_closed() {
        log::info!("{}: client went away", gen.id);
        return;
    }
    let _ = sse.send(Bytes::from_static(b"data: [DONE]\n\n")).await;
}

// ---- /tokenize, /detokenize, /v1/models, /health, /metrics -------------------------------------------------------

async fn tokenize(st: &'static State, req: Request<Incoming>) -> Response<HttpBody> {
    let r = async {
        let body = read_json(req).await?;
        st.check_model(&body)?;
        let ids = if body.contains_key("messages") {
            let agp = body.get("add_generation_prompt").cloned().unwrap_or(Value::Bool(true));
            let cfm = body.get("continue_final_message").cloned().unwrap_or(Value::Bool(false));
            if truthy(Some(&cfm)) && truthy(Some(&agp)) {
                return Err(ApiError::bad(
                    "Cannot set both `continue_final_message` and `add_generation_prompt` to True.",
                    None,
                ));
            }
            if st.dialect == Dialect::Ds41 {
                // the same effort, thinking switch and prompt text as a chat request, so the counts agree
                let (mut th, effort) = dialect::resolve_ds(&body, st.thinking, st.ds_effort, st.ds_thinking_default)?;
                let tools = normalize_tools(body.get("tools"))?;
                let (agp, cfm) = dialect::ds_gen_flags(&body)?;
                if body.contains_key("tool_choice") {
                    let choice = ToolChoice::parse(tools.as_ref(), body.get("tool_choice"))?;
                    if !st.forced_tool_thinking
                        && choice.is_forced()
                        && !cfm
                        && !th.off
                        && !dialect::ds_explicit_thinking_on(&body)
                        && !dialect::ds_has_prior_reasoning(body.get("messages"))
                    {
                        th.off = true;
                    }
                }
                let (text, images) =
                    dialect::ds_prompt(body.get("messages"), tools.as_ref(), !th.off, effort, agp, cfm)?;
                let ids = st.tok.aencode(text, false).await.map_err(|e| ApiError::new(500, &e, None))?;
                if images.is_empty() {
                    ids
                } else {
                    // the span lengths come from the images' headers; the pixels are not built
                    use crate::vision;
                    let b64: Vec<&str> =
                        images.iter().map(vision::block_b64).collect::<Result<_, _>>().map_err(image_err)?;
                    vision::check_budget(&b64).map_err(image_err)?;
                    let n_spans: Vec<usize> = b64
                        .iter()
                        .map(|s| vision::Model::Ds41.n_span_b64(s))
                        .collect::<Result<_, _>>()
                        .map_err(image_err)?;
                    vision::expand(&ids, &n_spans, vision::IMAGE_TOKEN_ID).map_err(image_err)?.0
                }
            } else {
                // the same effort and the same refusals as a chat request, so the counts agree
                let (th, hf_enable, ink_effort) = match st.dialect {
                    Dialect::Hf | Dialect::Qwen => {
                        let (th, en) = dialect::resolve_hf(&body, &st.tpl().levels, st.thinking)?;
                        (th, en, None)
                    }
                    Dialect::Inkling => {
                        let (th, e) = dialect::resolve_inkling(&body, st.thinking, st.ink_effort.as_ref())?;
                        (th, None, e)
                    }
                    _ => (st.tpl().resolve_thinking(&body, st.thinking)?, None, None),
                };
                let agp_on = truthy(Some(&agp));
                let mut kw = build_template_kwargs(body.get("chat_template_kwargs"), agp, cfm, th.effort, None)?;
                if let Some(en) = hf_enable {
                    kw.insert("enable_thinking".into(), Value::Bool(en));
                }
                if let Some(e) = ink_effort {
                    kw.insert("reasoning_effort".into(), e);
                }
                let tools = normalize_tools(body.get("tools"))?;
                let msgs = body.get("messages").cloned().unwrap_or(Value::Null);
                let (mut text, images) = if crate::template::has_images(&msgs) {
                    st.tpl().render_images(Some(&msgs), tools.as_ref(), kw)?
                } else {
                    (st.tpl().render(Some(&msgs), tools.as_ref(), kw)?, Vec::new())
                };
                if th.off && matches!(st.dialect, Dialect::Hf | Dialect::Qwen) {
                    text = dialect::hf_thinking_off(text, agp_on);
                }
                let ids = st
                    .tok
                    .aencode(text, truthy(body.get("add_special_tokens")))
                    .await
                    .map_err(|e| ApiError::new(500, &e, None))?;
                if images.is_empty() {
                    ids
                } else {
                    // the span lengths come from the images' headers, without building the pixels
                    use crate::vision;
                    let b64: Vec<&str> =
                        images.iter().map(vision::block_b64).collect::<Result<_, _>>().map_err(image_err)?;
                    vision::check_budget(&b64).map_err(image_err)?;
                    let model = image_model(st);
                    let n_spans: Vec<usize> =
                        b64.iter().map(|s| model.n_span_b64(s)).collect::<Result<_, _>>().map_err(image_err)?;
                    vision::expand(&ids, &n_spans, model.token()).map_err(image_err)?.0
                }
            }
        } else if let Some(Value::String(p)) = body.get("prompt") {
            let add_special = body.get("add_special_tokens").is_none_or(|v| truthy(Some(v)));
            st.tok.aencode(p.clone(), add_special).await.map_err(|e| ApiError::new(500, &e, None))?
        } else {
            return Err(ApiError::bad("`prompt` (string) or `messages` is required", None));
        };
        Ok((body, ids))
    };
    let (body, ids) = match r.await {
        Ok(x) => x,
        Err(e) => return e.response(),
    };
    let mut o = Vec::with_capacity(64 + ids.len() * 8);
    let mut c = Obj::new(&mut o);
    c.int("count", ids.len());
    c.int("max_model_len", st.max_model_len());
    ids_json(c.key("tokens"), &ids);
    if truthy(body.get("return_token_strs")) {
        let s = c.key("token_strs");
        s.push(b'[');
        for (i, &t) in ids.iter().enumerate() {
            if i > 0 {
                s.push(b',');
            }
            match st.tok.id_to_token(t) {
                Some(x) => pyjson::string(s, &x),
                None => s.extend_from_slice(b"null"),
            }
        }
        s.push(b']');
    } else {
        c.raw("token_strs", b"null");
    }
    c.end();
    json_response(200, o)
}

async fn detokenize(st: &'static State, req: Request<Incoming>) -> Response<HttpBody> {
    let r = async {
        let body = read_json(req).await?;
        st.check_model(&body)?;
        let bad = || ApiError::bad("`tokens` must be a list of token ids", Some("tokens"));
        let Some(Value::Array(a)) = body.get("tokens") else { return Err(bad()) };
        let mut ids = Vec::with_capacity(a.len());
        for t in a {
            match t {
                Value::Number(n) if n.is_i64() || n.is_u64() => {
                    ids.push(n.as_u64().map_or(u32::MAX, |u| u.min(u32::MAX as u64) as u32))
                }
                Value::Bool(b) => ids.push(*b as u32),
                _ => return Err(bad()),
            }
        }
        Ok(ids)
    };
    let ids = match r.await {
        Ok(x) => x,
        Err(e) => return e.response(),
    };
    let mut o = Vec::new();
    let mut c = Obj::new(&mut o);
    c.str("prompt", &st.tok.decode(&ids, false));
    c.end();
    json_response(200, o)
}

fn models(st: &State) -> Response<HttpBody> {
    let mut o = Vec::new();
    let mut c = Obj::new(&mut o);
    c.str("object", "list");
    {
        let d = c.key("data");
        d.push(b'[');
        for (i, name) in st.served.iter().enumerate() {
            if i > 0 {
                d.push(b',');
            }
            let mut m = Obj::new(d);
            m.str("id", name);
            m.str("object", "model");
            m.int("created", st.started);
            m.str("owned_by", "gr8");
            m.str("root", &st.model_dir);
            m.raw("parent", b"null");
            m.int("max_model_len", st.max_model_len());
            m.raw("permission", b"[]");
            m.end();
        }
        d.push(b']');
    }
    c.end();
    json_response(200, o)
}

fn health(st: &State) -> Response<HttpBody> {
    health_response(st.engine.ready(), st.steering().as_ref())
}

fn health_response(ready: bool, capability: Option<&Value>) -> Response<HttpBody> {
    let digest = |c: &Value| c.get("digest").and_then(Value::as_str).unwrap_or("").chars().take(12).collect::<String>();
    let steering = match capability {
        Some(c) => {
            let mut s = serde_json::json!({
                "enabled": true,
                "forced": c.get("forced").and_then(Value::as_bool).unwrap_or(false),
                "profile": c.get("profile").and_then(Value::as_str).unwrap_or(""),
                "digest": digest(c),
                "scope": c.get("scope").and_then(Value::as_str).unwrap_or(""),
                "layers": c.get("layers").and_then(Value::as_u64).unwrap_or(0),
            });
            // several loaded profiles: the fields above are the first one's, and this lists all of them in load order
            if let Some(ps) = c.get("profiles").and_then(Value::as_array) {
                s["profiles"] = ps
                    .iter()
                    .map(|p| {
                        serde_json::json!({
                            "profile": p.get("profile").and_then(Value::as_str).unwrap_or(""),
                            "digest": digest(p),
                            "layers": p.get("layers").and_then(Value::as_u64).unwrap_or(0),
                        })
                    })
                    .collect();
            }
            s
        }
        None => serde_json::json!({"enabled": false, "forced": false}),
    };
    let body = serde_json::to_vec(&serde_json::json!({"ready": ready, "steering": steering})).unwrap();
    json_response(if ready { 200 } else { 503 }, body)
}

/// /health?deep: the head's supervisor checks every rank (weights against the engine's sealed sums, while it serves) and
/// answers with one JSON object; 200 when every rank passes, else 503. Each check reads every weight byte once, so it
/// answers localhost only.
async fn health_deep(st: &State, peer: std::net::SocketAddr) -> Response<HttpBody> {
    if !peer.ip().is_loopback() {
        return json_value(403, serde_json::json!({"ok": false, "error": "/health?deep answers localhost only"}));
    }
    let Some(addr) = st.supervisor.as_deref() else {
        return json_value(503, serde_json::json!({"ok": false, "error": "no supervisor (--supervisor HOST:PORT)"}));
    };
    match tokio::time::timeout(Duration::from_secs(90), ask_supervisor(addr, r#"{"t": "check_all"}"#)).await {
        Ok(Ok(v)) => {
            let ok = v.get("ok").and_then(Value::as_bool) == Some(true);
            json_value(if ok { 200 } else { 503 }, v)
        }
        Ok(Err(e)) => json_value(503, serde_json::json!({"ok": false, "error": format!("supervisor {addr}: {e}")})),
        Err(_) => {
            json_value(503, serde_json::json!({"ok": false, "error": format!("supervisor {addr}: no answer in 90 s")}))
        }
    }
}

/// /internal/hold (localhost only; the supervisor's `scale`): POST `?seconds=N` holds new requests up to N s while the
/// engine is replaced (0 releases them). Both methods answer the hold and the request counts.
fn hold(st: &State, req: &Request<Incoming>, peer: std::net::SocketAddr) -> Response<HttpBody> {
    if !peer.ip().is_loopback() {
        return json_value(403, serde_json::json!({"error": "/internal/hold answers localhost only"}));
    }
    if req.method() == Method::POST {
        let secs = req.uri().query().and_then(|q| q.split('&').find_map(|x| x.strip_prefix("seconds=")));
        match secs.and_then(|s| s.parse::<u64>().ok()) {
            Some(n) => st.engine.hold(n),
            None => return json_value(400, serde_json::json!({"error": "POST /internal/hold?seconds=N"})),
        }
    }
    let (hold, running, waiting) = st.engine.counts();
    json_value(200, serde_json::json!({"hold": hold, "running": running, "waiting": waiting}))
}

/// One JSON line to a supervisor's control port, one JSON line back.
async fn ask_supervisor(addr: &str, msg: &str) -> Result<Value, String> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    let mut s = tokio::net::TcpStream::connect(addr).await.map_err(|e| e.to_string())?;
    s.write_all(format!("{msg}\n").as_bytes()).await.map_err(|e| e.to_string())?;
    let mut line = String::new();
    BufReader::new(s).read_line(&mut line).await.map_err(|e| e.to_string())?;
    serde_json::from_str(&line).map_err(|e| format!("bad answer: {e}"))
}

async fn metrics_route(st: &State) -> Response<HttpBody> {
    match st.engine.metrics(Duration::from_secs(5)).await {
        Ok(snap) => {
            let front = metrics::Front {
                guard_retries: guard::RETRIES.load(std::sync::atomic::Ordering::Relaxed),
                guard_given_up: guard::GIVEN_UP.load(std::sync::atomic::Ordering::Relaxed),
            };
            response(200, "text/plain; charset=utf-8", metrics::render(&snap, &st.model_name, front).into_bytes())
        }
        Err(e) => {
            response(503, "text/plain; charset=utf-8", format!("engine metrics unavailable: {}\n", e.0).into_bytes())
        }
    }
}

/// A reply that ends normally with a tool call finishes as `tool_calls`. Any other reason stays: a call that
/// `max_tokens` cut off finishes as `length`, so a client does not run its truncated arguments.
fn finish_with_calls(has_calls: bool, reason: &'static str) -> &'static str {
    if has_calls && reason == "stop" {
        "tool_calls"
    } else {
        reason
    }
}

#[cfg(test)]
mod finish_tests {
    use super::finish_with_calls;

    #[test]
    fn a_cut_off_call_finishes_as_length() {
        assert_eq!(finish_with_calls(true, "stop"), "tool_calls");
        assert_eq!(finish_with_calls(true, "length"), "length");
        assert_eq!(finish_with_calls(false, "stop"), "stop");
        assert_eq!(finish_with_calls(false, "length"), "length");
    }
}

#[cfg(test)]
mod usage_tests {
    use super::*;

    #[test]
    fn step_stats_are_in_usage_only_when_given() {
        let mut o = Vec::new();
        usage(&mut o, 10, 5, Some(4), true, true, None, None);
        let v: Value = serde_json::from_slice(&o).unwrap();
        assert!(v.get("step_stats").is_none());
        let s = StepStats {
            steps: 3,
            draft_tokens: 9,
            accepted_tokens: 4,
            accepted_per_pos: vec![2, 1, 1],
            generated_tokens: 7,
            prefill_s: Some(0.125),
            decode_s: None,
        };
        let mut o = Vec::new();
        usage(&mut o, 10, 5, Some(4), true, true, None, Some(&s));
        let v: Value = serde_json::from_slice(&o).unwrap();
        assert_eq!(v["completion_tokens"], 5);
        assert_eq!(
            v["step_stats"],
            serde_json::json!({"steps": 3, "draft_tokens": 9, "accepted_tokens": 4, "accepted_per_pos": [2, 1, 1],
                               "generated_tokens": 7, "prefill_s": 0.125})
        );
    }
}

#[cfg(test)]
mod steering_health_tests {
    use super::*;

    #[tokio::test]
    async fn health_reports_disabled_or_the_loaded_receipt_with_readiness_status() {
        for (ready, status) in [(true, 200), (false, 503)] {
            let r = health_response(ready, None);
            assert_eq!(r.status().as_u16(), status);
            let body = r.into_body().collect().await.unwrap().to_bytes();
            let v: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(v, serde_json::json!({"ready":ready,"steering":{"enabled":false,"forced":false}}));
        }
        let cap = serde_json::json!({"profile":"example","digest":"a".repeat(64),"scope":"full",
                                    "layers":2,"forced":true,"forced_scale":2.0});
        let r = health_response(true, Some(&cap));
        let body = r.into_body().collect().await.unwrap().to_bytes();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            v["steering"],
            serde_json::json!({"enabled":true,"forced":true,"profile":"example",
                                                   "digest":"a".repeat(12),"scope":"full","layers":2})
        );
        let one = |name: &str, d: &str, layers: u64| serde_json::json!({"profile":name,"digest":d.repeat(64),"layers":layers});
        let mut two = one("en-zh", "a", 78);
        two["scope"] = "full".into();
        two["forced"] = false.into();
        two["profiles"] = serde_json::json!([one("en-zh", "a", 78), one("zh-en", "b", 77)]);
        let r = health_response(true, Some(&two));
        let body = r.into_body().collect().await.unwrap().to_bytes();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["steering"]["profile"], "en-zh");
        assert_eq!(
            v["steering"]["profiles"],
            serde_json::json!([{"profile":"en-zh","digest":"a".repeat(12),"layers":78},
                                                                 {"profile":"zh-en","digest":"b".repeat(12),"layers":77}])
        );
    }
}

#[cfg(test)]
mod http_tokenize_chat_tests {
    //! HTTP-level: a multi-turn DS41 prompt goes through /tokenize and then /v1/chat/completions on the real router.
    //! The ids each endpoint produces must equal the sequential encode, and the checkpoint cache must be reused.
    use std::sync::{Arc, Mutex};

    use hyper::service::service_fn;
    use hyper_util::rt::TokioIo;
    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream, UnixListener};

    use super::*;
    use crate::engine::Engine;
    use crate::wire::FrameReader;

    fn frame(v: Value) -> Vec<u8> {
        let b = serde_json::to_vec(&v).unwrap();
        let mut out = (b.len() as u32).to_be_bytes().to_vec();
        out.extend(b);
        out
    }

    /// A PROTOCOL.md engine that records each generate frame and answers it with `reply`, ending at `length`.
    async fn fake_engine(reply: Vec<u32>) -> (Arc<Engine>, Arc<Mutex<Vec<Value>>>) {
        // /tmp, not temp_dir(): macOS runners have a long temp_dir and a socket path must fit SUN_LEN (104 bytes)
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::path::PathBuf::from(format!("/tmp/g6f-{}-{n}.sock", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path).unwrap();
        let gens = Arc::new(Mutex::new(Vec::new()));
        let g2 = gens.clone();
        tokio::spawn(async move {
            let (sock, _) = listener.accept().await.unwrap();
            let (rd, mut wr) = sock.into_split();
            wr.write_all(&frame(json!({"type": "hello", "version": 1, "max_model_len": 1 << 20}))).await.unwrap();
            let mut r = FrameReader::new(rd);
            while let Ok(Some(f)) = r.next().await {
                let v: Value = serde_json::from_slice(&f).unwrap();
                if v["type"] != "generate" {
                    continue;
                }
                g2.lock().unwrap().push(v.clone());
                let n = (v["max_tokens"].as_u64().unwrap() as usize).min(reply.len());
                let id = v["id"].clone();
                if v["logprobs"].is_number() {
                    let lps: Vec<Value> = reply[..n]
                        .iter()
                        .map(|&tid| json!({"lp": -0.25, "rank": 1, "top": [[tid, -0.25]]}))
                        .collect();
                    wr.write_all(&frame(json!({"type": "tokens", "id": id, "ids": &reply[..n], "logprobs": lps})))
                        .await
                        .unwrap();
                } else {
                    wr.write_all(&frame(json!({"type": "tokens", "id": id, "ids": &reply[..n]}))).await.unwrap();
                }
                let fin = json!({"type": "finish", "id": id, "reason": "length", "stop_id": null,
                                 "cached_tokens": 0, "completion_tokens": n});
                wr.write_all(&frame(fin)).await.unwrap();
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
        (engine, gens)
    }

    /// One HTTP/1.1 POST with Connection: close; the status and the JSON body.
    async fn post(addr: std::net::SocketAddr, path: &str, body: &Value) -> (u16, Value) {
        let (status, payload) = post_raw(addr, path, body).await;
        (status, serde_json::from_slice(&payload).unwrap_or(Value::Null))
    }

    async fn post_raw(addr: std::net::SocketAddr, path: &str, body: &Value) -> (u16, Vec<u8>) {
        let b = serde_json::to_vec(body).unwrap();
        let mut s = TcpStream::connect(addr).await.unwrap();
        let mut req = format!(
            "POST {path} HTTP/1.1\r\nHost: test\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            b.len()
        )
        .into_bytes();
        req.extend_from_slice(&b);
        s.write_all(&req).await.unwrap();
        let mut raw = Vec::new();
        // On BSD/macOS, when the server sends an early error response (e.g. 404) without draining the request body
        // and closes the socket, the kernel follows the response bytes with a TCP RST (ECONNRESET).
        match s.read_to_end(&mut raw).await {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::ConnectionReset && !raw.is_empty() => {}
            Err(e) => panic!("read_to_end failed: {e}"),
        }
        let split = raw.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
        let head = String::from_utf8_lossy(&raw[..split]).to_ascii_lowercase();
        let status: u16 = head.split_whitespace().nth(1).unwrap().parse().unwrap();
        let mut payload = raw[split + 4..].to_vec();
        if head.contains("transfer-encoding: chunked") {
            let (mut out, mut rest) = (Vec::new(), &payload[..]);
            loop {
                let eol = rest.windows(2).position(|w| w == b"\r\n").unwrap();
                let n = usize::from_str_radix(std::str::from_utf8(&rest[..eol]).unwrap().trim(), 16).unwrap();
                if n == 0 {
                    break;
                }
                out.extend_from_slice(&rest[eol + 2..eol + 2 + n]);
                rest = &rest[eol + 4 + n..];
            }
            payload = out;
        }
        (status, payload)
    }

    /// The prompt text both handlers render for a DS41 request without tools (dialect::resolve_ds + ds_prompt).
    fn prompt_text(st: &State, body: &Value) -> String {
        let body = body.as_object().unwrap();
        let (th, effort) = dialect::resolve_ds(body, st.thinking, st.ds_effort, st.ds_thinking_default).unwrap();
        let (agp, cfm) = dialect::ds_gen_flags(body).unwrap();
        dialect::ds_prompt(body.get("messages"), None, !th.off, effort, agp, cfm).unwrap().0
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn tokenize_then_chat_reuses_turn_checkpoints_with_exact_ids() {
        let mut tok = crate::tok::prefix_tests::fixture(true);
        assert!(tok.cache_on_for_test());
        tok.use_single_think_close();
        let dir = std::env::temp_dir().join(format!("g6front-http-model-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        tok.tk.save(dir.join("tokenizer.json"), false).unwrap();
        let grammars = crate::grammar::Grammars::load(&dir, tok.vocab_size).unwrap();
        let reply: Vec<u32> = "answer".bytes().map(|b| tok.tk.token_to_id(&(b as char).to_string()).unwrap()).collect();
        let (engine, gens) = fake_engine(reply).await;
        let st: &'static State = Box::leak(Box::new(State {
            tok,
            template: None,
            dialect: Dialect::Ds41,
            ds_effort: dialect::DS_DEFAULT_EFFORT,
            ds_thinking_default: true,
            forced_tool_thinking: false,
            ink_effort: None,
            engine,
            served: vec!["deepseek-v4.1-flash".into()],
            model_name: "deepseek-v4.1-flash".into(),
            model_dir: dir.to_string_lossy().into_owned(),
            reasoning_fields: &["reasoning_content"],
            prompt_tokens_details: true,
            max_model_len_override: 0,
            default_max_tokens: 0,
            share_tokens: 0,
            clamp_max_tokens: false,
            thinking: thinking::Defaults { effort: None, budget: sampling::Limit::Tokens(0), allow_off: true },
            grammars,
            emit_undeclared: true,
            guard: guard::Config::default(),
            penalties: sampling::Penalties::default(),
            boost: None,
            sequence: None,
            started: now_secs(),
            supervisor: None,
            sessions: None,
        }));
        let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let (sock, peer) = listener.accept().await.unwrap();
                tokio::spawn(async move {
                    let svc = service_fn(move |req| route(st, req, peer));
                    let _ = hyper::server::conn::http1::Builder::new().serve_connection(TokioIo::new(sock), svc).await;
                });
            }
        });

        let (off_status, off_body) = post_raw(addr, "/v1/sessions/absent/forks", &json!({"count": 1})).await;
        assert_eq!((off_status, off_body.as_slice()), (404, b"404: Not Found".as_slice()));
        let base = json!({"model": st.model_name, "messages": [{"role": "user", "content": "same"}]});
        let mut with_session = base.clone();
        with_session["session"] = json!({"create": true});
        let a = post_raw(addr, "/tokenize", &base).await;
        let b = post_raw(addr, "/tokenize", &with_session).await;
        assert_eq!(a, b, "knob off: existing route bytes must not change");

        let system = "system tools schema: read files, run commands, edit code\n".repeat(120);
        let mut messages = vec![json!({"role": "system", "content": system})];
        let mut last_hits = st.tok.cache_hits_for_test();
        for turn in 0..4 {
            messages.push(json!({"role": "user", "content": format!("turn {turn}: {}", "a b c d word ".repeat(80))}));
            let req = json!({"model": "deepseek-v4.1-flash", "messages": messages.clone()});
            let want = st.tok.encode_seq_for_test(&prompt_text(st, &req));

            let (code, v) = post(addr, "/tokenize", &req).await;
            assert_eq!(code, 200, "/tokenize turn {turn}: {v}");
            let ids: Vec<u32> = v["tokens"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
            assert_eq!(ids, want, "/tokenize turn {turn} ids differ from the sequential encode");
            let hits = st.tok.cache_hits_for_test();
            if turn > 0 {
                assert!(hits > last_hits, "/tokenize turn {turn} did not reuse an earlier turn's checkpoint");
            }
            last_hits = hits;

            let mut chat = req.clone();
            chat["max_tokens"] = json!(6);
            let (code, v) = post(addr, "/v1/chat/completions", &chat).await;
            assert_eq!(code, 200, "chat turn {turn}: {v}");
            assert_eq!(v["usage"]["prompt_tokens"].as_u64().unwrap() as usize, want.len());
            let sent: Vec<u32> = gens.lock().unwrap().last().unwrap()["prompt"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect();
            assert_eq!(sent, want, "chat turn {turn}: engine prompt differs from the sequential encode");
            let hits = st.tok.cache_hits_for_test();
            assert!(hits > last_hits, "chat turn {turn} did not hit the prompt /tokenize cached");
            last_hits = hits;
            let content = v["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string();
            messages.push(json!({"role": "assistant", "content": content}));
        }
        assert_eq!(gens.lock().unwrap().len(), 4);
        eprintln!("HTTP_TOKENIZE_CHAT turns=4 cache_hits={last_hits}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn session_http_lifecycle() {
        let tok = crate::tok::prefix_tests::fixture(true);
        let dir = std::env::temp_dir().join(format!("g6front-session-model-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        tok.tk.save(dir.join("tokenizer.json"), false).unwrap();
        let grammars = crate::grammar::Grammars::load(&dir, tok.vocab_size).unwrap();
        let reply: Vec<u32> = "answer".bytes().map(|b| tok.tk.token_to_id(&(b as char).to_string()).unwrap()).collect();
        let (engine, gens) = fake_engine(reply).await;
        let st: &'static State = Box::leak(Box::new(State {
            tok,
            template: None,
            dialect: Dialect::Ds41,
            ds_effort: dialect::DS_DEFAULT_EFFORT,
            ds_thinking_default: true,
            forced_tool_thinking: false,
            ink_effort: None,
            engine,
            served: vec!["model".into()],
            model_name: "model".into(),
            model_dir: dir.to_string_lossy().into_owned(),
            reasoning_fields: &["reasoning_content"],
            prompt_tokens_details: true,
            max_model_len_override: 0,
            default_max_tokens: 0,
            share_tokens: 0,
            clamp_max_tokens: false,
            thinking: thinking::Defaults { effort: None, budget: sampling::Limit::Tokens(0), allow_off: true },
            grammars,
            emit_undeclared: true,
            guard: guard::Config::default(),
            penalties: sampling::Penalties::default(),
            boost: None,
            sequence: None,
            started: now_secs(),
            supervisor: None,
            sessions: Some(crate::sessions::Sessions::new(Duration::from_secs(60))),
        }));
        let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let (sock, peer) = listener.accept().await.unwrap();
                tokio::spawn(async move {
                    let svc = service_fn(move |req| route(st, req, peer));
                    let _ = hyper::server::conn::http1::Builder::new().serve_connection(TokioIo::new(sock), svc).await;
                });
            }
        });
        let (status, made) = post(addr, "/v1/chat/completions", &json!({
            "model":"model", "session":{"create":true}, "messages":[{"role":"user","content":"base"}], "max_tokens":6
        })).await;
        assert_eq!(status, 200, "{made}");
        let parent = made["session_id"].as_str().unwrap();
        let (status, forked) = post(addr, &format!("/v1/sessions/{parent}/forks"), &json!({"count":3})).await;
        assert_eq!(status, 200, "{forked}");
        assert!(forked["prefix_tokens"].as_u64().unwrap() > 0);
        let ids = forked["fork_ids"].as_array().unwrap();
        assert_eq!(ids.len(), 3);
        let (status, first) = post(addr, "/v1/chat/completions", &json!({
            "model":"model", "session_id":ids[0], "append_messages":[{"role":"user","content":"branch A"}], "max_tokens":6
        })).await;
        assert_eq!(status, 200, "{first}");
        assert_eq!(first["session_id"], ids[0]);
        let mut sock = TcpStream::connect(addr).await.unwrap();
        sock.write_all(
            format!("DELETE /v1/sessions/{parent} HTTP/1.1\r\nHost: test\r\nConnection: close\r\n\r\n").as_bytes(),
        )
        .await
        .unwrap();
        let mut deleted = Vec::new();
        sock.read_to_end(&mut deleted).await.unwrap();
        assert!(String::from_utf8_lossy(&deleted).contains("200 OK"), "{}", String::from_utf8_lossy(&deleted));
        let (status, second) = post(addr, "/v1/chat/completions", &json!({
            "model":"model", "session_id":ids[1], "append_messages":[{"role":"user","content":"branch B"}], "max_tokens":6
        })).await;
        assert_eq!(status, 200, "{second}");
        assert_eq!(gens.lock().unwrap().len(), 3);
        let (status, unknown) = post(addr, "/v1/sessions/missing/forks", &json!({"count":1})).await;
        assert_eq!(status, 404, "{unknown}");
        let (status, streamed) = post_raw(
            addr,
            "/v1/chat/completions",
            &json!({
                "model":"model", "session":{"create":true}, "messages":[{"role":"user","content":"stream"}],
                "max_tokens":6, "stream":true
            }),
        )
        .await;
        assert_eq!(status, 200);
        let stream_text = String::from_utf8(streamed).unwrap();
        assert!(stream_text.contains("\"session_id\":\"sess-"), "{stream_text}");
        assert!(stream_text.contains("data: [DONE]"), "{stream_text}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn stream_logprobs_kept_when_thinking_off_and_hidden_reasoning_not_leaked() {
        let tok = crate::tok::prefix_tests::fixture(true);
        let dir = std::env::temp_dir().join(format!("g6front-stream-lp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        tok.tk.save(dir.join("tokenizer.json"), false).unwrap();
        std::fs::write(
            dir.join("chat_template.jinja"),
            "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}{% if add_generation_prompt %}assistant:\n{% if enable_thinking is defined and not enable_thinking %}<think>\n\n</think>\n\n{% else %}<think>\n{% endif %}{% endif %}",
        )
        .unwrap();
        let template = crate::template::ChatTemplate::load(&dir, tok.eos_token.clone(), tok.pad_token.clone()).unwrap();
        let grammars = crate::grammar::Grammars::load(&dir, tok.vocab_size).unwrap();
        let reply: Vec<u32> = "answer".bytes().map(|b| tok.tk.token_to_id(&(b as char).to_string()).unwrap()).collect();
        let (engine, _gens) = fake_engine(reply).await;
        let st: &'static State = Box::leak(Box::new(State {
            tok,
            template: Some(template),
            dialect: Dialect::Qwen,
            ds_effort: dialect::DS_DEFAULT_EFFORT,
            ds_thinking_default: true,
            forced_tool_thinking: false,
            ink_effort: None,
            engine,
            served: vec!["model".into()],
            model_name: "model".into(),
            model_dir: dir.to_string_lossy().into_owned(),
            reasoning_fields: &["reasoning_content"],
            prompt_tokens_details: true,
            max_model_len_override: 0,
            default_max_tokens: 0,
            share_tokens: 0,
            clamp_max_tokens: false,
            thinking: thinking::Defaults { effort: None, budget: sampling::Limit::Tokens(0), allow_off: true },
            grammars,
            emit_undeclared: true,
            guard: guard::Config::default(),
            penalties: sampling::Penalties::default(),
            boost: None,
            sequence: None,
            started: now_secs(),
            supervisor: None,
            sessions: None,
        }));
        let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let (sock, peer) = listener.accept().await.unwrap();
                tokio::spawn(async move {
                    let svc = service_fn(move |req| route(st, req, peer));
                    let _ = hyper::server::conn::http1::Builder::new().serve_connection(TokioIo::new(sock), svc).await;
                });
            }
        });
        // Thinking off (enable_thinking: false) + stream: true + logprobs: true MUST emit content logprobs.
        let (status, raw) = post_raw(
            addr,
            "/v1/chat/completions",
            &json!({
                "model": "model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 6,
                "stream": true,
                "logprobs": true,
                "top_logprobs": 1,
                "chat_template_kwargs": {"enable_thinking": false}
            }),
        )
        .await;
        assert_eq!(status, 200);
        let sse = String::from_utf8(raw).unwrap();
        assert!(sse.contains("\"logprobs\":{\"content\":["), "thinking_off streaming dropped logprobs: {sse}");
        assert!(sse.contains("\"token\":\"a\""), "{sse}");

        // Thinking ON + include_reasoning: false MUST NOT leak withheld reasoning tokens via logprobs.
        let (status, raw_hidden) = post_raw(
            addr,
            "/v1/chat/completions",
            &json!({
                "model": "model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 6,
                "stream": true,
                "logprobs": true,
                "include_reasoning": false
            }),
        )
        .await;
        assert_eq!(status, 200);
        let sse_hidden = String::from_utf8(raw_hidden).unwrap();
        assert!(!sse_hidden.contains("\"logprobs\":{\"content\":["), "hidden reasoning leaked via logprobs: {sse_hidden}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The literal-content gate: every case in `literal_golden.json` was tokenized by the release's own tokenizer
    /// (`frontend/tests/literal_golden.py`), so these ids are what the model was trained to see. Each one goes
    /// through the real router on /v1/chat/completions, and the prompt the engine receives must equal them id for id
    /// -- rendering and encoding together, not just the encode. Skipped unless the release is at hand:
    /// GR8_TEST_TOKENIZER_DIR=<the release directory> cargo test --release
    #[tokio::test]
    async fn literal_content_golden_ids_through_the_http_path() {
        let Some(dir) = gr8knobs::TEST_TOKENIZER_DIR.get() else {
            eprintln!("GR8_TEST_TOKENIZER_DIR not set; skipping");
            return;
        };
        let dir = std::path::PathBuf::from(dir);
        let golden: Value = serde_json::from_str(include_str!("literal_golden.json")).unwrap();
        let cfg: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("tokenizer_config.json")).unwrap()).unwrap();
        let declared = &cfg["auto_map"]["AutoTokenizer"];
        let declared = declared.as_str().or_else(|| declared.get(0).and_then(|v| v.as_str()));
        if declared != golden["auto_map"].as_str() {
            eprintln!(
                "GR8_TEST_TOKENIZER_DIR declares {declared:?}, not the {:?} literal_golden.json was built from; skipping",
                golden["auto_map"].as_str()
            );
            return;
        }

        let tok = crate::tok::Tok::load(&dir).unwrap();
        let spellings = tok.literal_spellings();
        assert!(spellings.is_some(), "the release declares its own tokenizer but literal content stayed off");
        let template = crate::template::ChatTemplate::load(&dir, tok.eos_token.clone(), tok.pad_token.clone())
            .unwrap()
            .with_literal_content(spellings);
        let grammars = crate::grammar::Grammars::load(&dir, tok.vocab_size).unwrap();
        let reply = tok.tk.encode("ok", false).unwrap().get_ids().to_vec();
        let (engine, gens) = fake_engine(reply).await;
        let st: &'static State = Box::leak(Box::new(State {
            tok,
            template: Some(template),
            dialect: Dialect::Glm,
            ds_effort: dialect::DS_DEFAULT_EFFORT,
            ds_thinking_default: true,
            forced_tool_thinking: false,
            ink_effort: None,
            engine,
            served: vec!["literal-golden".into()],
            model_name: "literal-golden".into(),
            model_dir: dir.to_string_lossy().into_owned(),
            reasoning_fields: &["reasoning_content"],
            prompt_tokens_details: true,
            max_model_len_override: 0,
            default_max_tokens: 0,
            share_tokens: 0,
            clamp_max_tokens: false,
            thinking: thinking::Defaults { effort: None, budget: sampling::Limit::Tokens(0), allow_off: true },
            grammars,
            emit_undeclared: true,
            guard: guard::Config::default(),
            penalties: sampling::Penalties::default(),
            boost: None,
            sequence: None,
            started: now_secs(),
            supervisor: None,
            sessions: None,
        }));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let (sock, peer) = listener.accept().await.unwrap();
                tokio::spawn(async move {
                    let svc = service_fn(move |req| route(st, req, peer));
                    let _ = hyper::server::conn::http1::Builder::new().serve_connection(TokioIo::new(sock), svc).await;
                });
            }
        });

        let cases = golden["cases"].as_array().unwrap();
        assert!(!cases.is_empty());
        for c in cases {
            let name = c["name"].as_str().unwrap();
            let mut req = json!({"model": "literal-golden", "messages": c["request"]["messages"], "max_tokens": 1});
            if c["request"]["tools"].as_array().is_some_and(|t| !t.is_empty()) {
                req["tools"] = c["request"]["tools"].clone();
            }
            let (code, v) = post(addr, "/v1/chat/completions", &req).await;
            assert_eq!(code, 200, "{name}: {v}");
            let want: Vec<u32> =
                c["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
            let sent: Vec<u32> = gens.lock().unwrap().last().unwrap()["prompt"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect();
            let at = sent.iter().zip(&want).position(|(a, b)| a != b);
            assert_eq!(
                sent,
                want,
                "{name}: the prompt the engine received is not the release's own tokenization \
                 ({} ids against {}, first difference at {at:?})",
                sent.len(),
                want.len()
            );
            assert_eq!(v["usage"]["prompt_tokens"].as_u64().unwrap() as usize, want.len(), "{name}: usage");
            eprintln!("LITERAL_GOLDEN {name} ids={} exact", want.len());
        }
        assert_eq!(gens.lock().unwrap().len(), cases.len());
    }
}
