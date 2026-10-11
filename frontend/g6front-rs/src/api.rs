//! HTTP routes, shaped like vLLM 0.30 and byte-identical to the Python front end (`g6front/routes_*.py`):
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

use crate::gen::{Generation, Res, Step, StopReason};
use crate::parser::{Delta, GlmParser};
use crate::pyjson::{self, truthy};
use crate::template::{build_template_kwargs, normalize_tools};
use crate::util::{new_id, now_secs};
use crate::{logprobs as L, metrics, sampling, State};

const MAX_BODY: usize = 512 << 20;

// ---- errors ------------------------------------------------------------------------------------------------------

#[derive(Debug)]
pub struct ApiError {
    pub status: u16,
    pub message: String,
    pub param: Option<String>,
}

impl ApiError {
    pub fn new(status: u16, message: &str, param: Option<&str>) -> ApiError {
        ApiError { status, message: message.to_string(), param: param.map(str::to_string) }
    }

    pub fn bad(message: impl Into<String>, param: Option<&str>) -> ApiError {
        ApiError { status: 400, message: message.into(), param: param.map(str::to_string) }
    }

    fn body(&self) -> Vec<u8> {
        let ty = match self.status {
            404 => "NotFoundError",
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
        json_response(self.status, self.body())
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

fn usage(o: &mut Vec<u8>, prompt: usize, completion: usize, cached: Option<u64>, details: bool, full: bool) {
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
    u.end();
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
    let body = http_body_util::Limited::new(req.into_body(), MAX_BODY)
        .collect()
        .await
        .map_err(|e| ApiError::new(413, &format!("request body too large or unreadable: {e}"), None))?
        .to_bytes();
    match serde_json::from_slice::<Value>(&body) {
        Ok(Value::Object(m)) => Ok((m, body)),
        Ok(_) => Err(ApiError::bad("request body must be a JSON object", None)),
        Err(e) => Err(ApiError::bad(format!("invalid JSON body: {e}"), None)),
    }
}

impl State {
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

    fn require_engine(&self) -> Result<(), ApiError> {
        if self.engine.ready() {
            Ok(())
        } else {
            Err(ApiError::new(503, "engine not connected", None))
        }
    }

    pub fn max_model_len(&self) -> u64 {
        if self.max_model_len_override > 0 {
            return self.max_model_len_override;
        }
        self.engine.hello().and_then(|h| h.get("max_model_len").and_then(Value::as_u64)).unwrap_or(0)
    }

    fn max_logprobs(&self) -> i64 {
        self.engine.hello().and_then(|h| h.get("max_logprobs").and_then(Value::as_i64)).unwrap_or(20)
    }
}

pub async fn route(st: &'static State, req: Request<Incoming>) -> Result<Response<HttpBody>, Infallible> {
    let path = req.uri().path();
    let m = req.method().clone();
    let r = match (path, &m) {
        ("/v1/chat/completions", &Method::POST) => chat(st, req).await,
        ("/v1/completions", &Method::POST) => completions(st, req).await,
        ("/tokenize", &Method::POST) => tokenize(st, req).await,
        ("/detokenize", &Method::POST) => detokenize(st, req).await,
        ("/v1/models", &Method::GET | &Method::HEAD) => models(st),
        ("/health", &Method::GET | &Method::HEAD) => health(st),
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

// ---- /v1/chat/completions ----------------------------------------------------------------------------------------

struct ChatReq {
    tools: Option<Value>,
    tool_choice: Value,
    prompt: Vec<u32>,
    params: sampling::Params,
}

async fn chat_prepare(st: &'static State, body: &Map<String, Value>, raw: &[u8]) -> Result<ChatReq, ApiError> {
    st.check_model(body)?;
    st.require_engine()?;
    if truthy(body.get("echo")) {
        return Err(ApiError::bad("echo is not supported by the g6 front end", Some("echo")));
    }
    let tools = normalize_tools(body.get("tools"))?;
    let tool_choice = match body.get("tool_choice") {
        None | Some(Value::Null) => Value::from(if tools.is_some() { "auto" } else { "none" }),
        Some(v) => v.clone(),
    };
    if tool_choice.is_object() || tool_choice.as_str() == Some("required") {
        return Err(ApiError::bad(
            "named / required tool_choice is not supported by the g6 front end",
            Some("tool_choice"),
        ));
    }
    let kw = build_template_kwargs(
        body.get("chat_template_kwargs"),
        body.get("add_generation_prompt").cloned().unwrap_or(Value::Bool(true)),
        body.get("continue_final_message").cloned().unwrap_or(Value::Bool(false)),
        body.get("reasoning_effort"),
        body.get("documents"),
    )?;
    if truthy(body.get("chat_template")) {
        return Err(ApiError::bad("custom chat_template is not supported", Some("chat_template")));
    }
    let t0 = std::time::Instant::now();
    let text = st.template.render(body.get("messages"), tools.as_ref(), kw)?;
    let (t1, chars) = (std::time::Instant::now(), text.len());
    let add_special = truthy(body.get("add_special_tokens"));
    let prompt = st.tok.aencode(text, add_special).await.map_err(|e| ApiError::new(500, &e, None))?;
    log::debug!(
        "chat prompt: {chars} bytes -> {} tokens; template {:.2} ms, encode {:.2} ms",
        prompt.len(),
        (t1 - t0).as_secs_f64() * 1e3,
        t1.elapsed().as_secs_f64() * 1e3
    );
    let params = sampling::build(
        body,
        raw,
        prompt.len(),
        st.max_model_len(),
        &st.tok.eos_ids,
        st.tok.vocab_size,
        &st.tok.gen_defaults,
        None,
        st.max_logprobs(),
        true,
    )?;
    Ok(ChatReq { tools, tool_choice, prompt, params })
}

async fn chat(st: &'static State, req: Request<Incoming>) -> Response<HttpBody> {
    let (body, raw) = match read_json_raw(req).await {
        Ok(b) => b,
        Err(e) => return e.response(),
    };
    let cr = match chat_prepare(st, &body, &raw).await {
        Ok(c) => c,
        Err(e) => return e.response(),
    };
    let rid = new_id("chatcmpl");
    let include_reasoning = body.get("include_reasoning").is_none_or(|v| truthy(Some(v)));
    let parser = GlmParser::new(cr.tools.as_ref(), &cr.tool_choice, include_reasoning);
    let stream = truthy(body.get("stream"));
    // the GLM reasoning parser forces skip_special_tokens=False (vLLM parser_engine.adjust_request)
    let mut gen = Generation::new(&st.tok, rid, cr.prompt, cr.params, false, !stream);
    if stream {
        let (sse, resp) = Sse::open();
        tokio::spawn(async move { chat_stream(st, body, gen, parser, sse).await });
        return resp;
    }
    let created = now_secs();
    if let Err(e) = run_to_end(st, &mut gen).await {
        return e.response();
    }
    chat_full(st, &body, &gen, parser, created)
}

async fn run_to_end(st: &State, gen: &mut Generation) -> Result<(), ApiError> {
    gen.start(&st.engine)?;
    while gen.next().await?.is_some() {}
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
    mut parser: GlmParser,
    created: u64,
) -> Response<HttpBody> {
    let res = &gen.res;
    let p = parser.parse_full(&res.text);
    let reasoning = if parser.include_reasoning { p.reasoning } else { None };
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
        x.str("finish_reason", if p.calls.is_empty() { res.finish_reason.unwrap_or("stop") } else { "tool_calls" });
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
    mut parser: GlmParser,
    sse: Sse,
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
    let r: Result<(), ApiError> = async {
        gen.start(&st.engine)?;
        loop {
            // a client that goes away while nothing is being sent (prefill, held-back text) still cancels at once
            let step = tokio::select! {
                biased;
                _ = sse.0.closed() => return Ok(()),
                s = gen.next() => s?,
            };
            let Some(step) = step else { break };
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
                if continuous {
                    usage(c.key("usage"), gen.res.prompt_tokens, 0, None, false, false);
                }
                c.end();
                if sse.json(o).await.is_err() {
                    return Ok(());
                }
            }
            if chat_step(st, &gen, &mut parser, &step, &sse, created, lp_k, ret_ids, as_ids, continuous, &mut tools_streamed)
                .await
                .is_err()
            {
                return Ok(()); // client went away; dropping `gen` cancels the engine request
            }
        }
        if include_usage {
            let res = &gen.res;
            let mut o = Vec::with_capacity(256);
            let mut c = head(&mut o, &gen.id, "chat.completion.chunk", created, &st.model_name);
            c.raw("choices", b"[]");
            usage(c.key("usage"), res.prompt_tokens, res.completion_tokens, Some(res.cached_tokens), st.prompt_tokens_details, false);
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

#[allow(clippy::too_many_arguments)]
async fn chat_step(
    st: &State,
    gen: &Generation,
    parser: &mut GlmParser,
    step: &Step,
    sse: &Sse,
    created: u64,
    lp_k: Option<i64>,
    ret_ids: bool,
    as_ids: bool,
    continuous: bool,
    tools_streamed: &mut bool,
) -> Result<(), ()> {
    let finished = step.finish_reason.is_some();
    let delta = parser.delta(&step.text, finished);
    if delta.as_ref().is_some_and(|d| !d.tool_calls.is_empty()) {
        *tools_streamed = true;
    }
    let want_lp = lp_k.is_some() && !step.ids.is_empty() && parser.include_reasoning;
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
            Some(fr) => x.str("finish_reason", if *tools_streamed && fr == "stop" { "tool_calls" } else { fr }),
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
        x.end();
        ch.push(b']');
    }
    if continuous {
        usage(c.key("usage"), gen.res.prompt_tokens, gen.res.completion_tokens, None, false, false);
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
            "batched prompts are not supported by the g6 front end; send one prompt per request",
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
        st.require_engine()?;
        for k in ["echo", "suffix"] {
            if truthy(body.get(k)) {
                return Err(ApiError::bad(format!("`{k}` is not supported by the g6 front end"), Some(k)));
            }
        }
        let prompt = prompt_ids(st, &body).await?;
        let params = sampling::build(
            &body,
            &raw,
            prompt.len(),
            st.max_model_len(),
            &st.tok.eos_ids,
            st.tok.vocab_size,
            &st.tok.gen_defaults,
            Some(16),
            st.max_logprobs(),
            false,
        )?;
        Ok((prompt, params))
    };
    let (prompt, params) = match prep.await {
        Ok(x) => x,
        Err(e) => return e.response(),
    };
    let rid = new_id("cmpl");
    let skip_special = body.get("skip_special_tokens").is_none_or(|v| truthy(Some(v)));
    let stream = truthy(body.get("stream"));
    let mut gen = Generation::new(&st.tok, rid, prompt, params, skip_special, !stream);
    if stream {
        let (sse, resp) = Sse::open();
        tokio::spawn(async move { completions_stream(st, body, gen, sse).await });
        return resp;
    }
    let created = now_secs();
    if let Err(e) = run_to_end(st, &mut gen).await {
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
    );
    c.raw("kv_transfer_params", b"null");
    c.end();
    json_response(200, o)
}

async fn completions_stream(st: &'static State, body: Map<String, Value>, mut gen: Generation, sse: Sse) {
    let created = now_secs();
    let so = body.get("stream_options").and_then(Value::as_object);
    let include_usage = so.is_some_and(|s| truthy(s.get("include_usage")));
    let continuous = include_usage && so.is_some_and(|s| truthy(s.get("continuous_usage_stats")));
    let as_ids = truthy(body.get("return_tokens_as_token_ids"));
    let ret_ids = truthy(body.get("return_token_ids"));
    let lp_k = gen.params.gen.logprobs;
    let (mut offset, mut first) = (0usize, true);
    let r: Result<(), ApiError> = async {
        gen.start(&st.engine)?;
        loop {
            // a client that goes away while nothing is being sent (prefill, held-back text) still cancels at once
            let step = tokio::select! {
                biased;
                _ = sse.0.closed() => return Ok(()),
                s = gen.next() => s?,
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
                usage(c.key("usage"), gen.res.prompt_tokens, gen.res.completion_tokens, None, false, false);
            }
            c.end();
            if sse.json(o).await.is_err() {
                return Ok(());
            }
        }
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
            let kw = build_template_kwargs(body.get("chat_template_kwargs"), agp, cfm, None, None)?;
            let tools = normalize_tools(body.get("tools"))?;
            let text = st.template.render(body.get("messages"), tools.as_ref(), kw)?;
            st.tok
                .aencode(text, truthy(body.get("add_special_tokens")))
                .await
                .map_err(|e| ApiError::new(500, &e, None))?
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
            m.str("owned_by", "g6");
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
    let mut r = Response::new(HttpBody::Full(None));
    *r.status_mut() = if st.engine.ready() { StatusCode::OK } else { StatusCode::SERVICE_UNAVAILABLE };
    r
}

async fn metrics_route(st: &State) -> Response<HttpBody> {
    match st.engine.metrics(Duration::from_secs(5)).await {
        Ok(snap) => response(200, "text/plain; charset=utf-8", metrics::render(&snap, &st.model_name).into_bytes()),
        Err(e) => {
            response(503, "text/plain; charset=utf-8", format!("engine metrics unavailable: {}\n", e.0).into_bytes())
        }
    }
}
