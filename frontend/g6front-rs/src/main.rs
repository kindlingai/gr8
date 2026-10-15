//! g6front-rs: OpenAI/vLLM-compatible HTTP front end for the gr8 engine, speaking PROTOCOL.md over the engine's
//! Unix socket. This is the front end `./gr8 glm53 up` runs.
//!
//! g6front-rs --model-dir DIR [--engine-sock /run/g6/engine.sock] [--port 8100]

mod api;
mod engine;
mod gen;
mod logprobs;
mod metrics;
mod parser;
mod pyjson;
mod sampling;
mod template;
mod thinking;
mod tok;
mod util;
mod wire;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use hyper::service::service_fn;
use hyper_util::rt::{TokioIo, TokioTimer};
use tokio::net::TcpListener;

/// mimalloc: ~40% faster long-prompt encodes than glibc malloc on aarch64, and with `mi_collect` the tokenizer's
/// ~280 MB load-time peak goes back to the OS instead of staying resident.
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub struct State {
    pub tok: tok::Tok,
    pub template: template::ChatTemplate,
    pub engine: Arc<engine::Engine>,
    pub served: Vec<String>,
    pub model_name: String,
    pub model_dir: String,
    pub reasoning_fields: &'static [&'static str],
    pub prompt_tokens_details: bool,
    pub max_model_len_override: u64,
    /// max_tokens for a chat request that names none; 0: see State::chat_default_max_tokens
    pub default_max_tokens: u64,
    /// the tokens (prompt + max_tokens) a request may take and still share decode steps; 0: no such limit
    pub share_tokens: u64,
    /// reasoning_effort and thinking_budget for a request that names none
    pub thinking: thinking::Defaults,
    pub started: u64,
    /// HOST:PORT of the head's supervisor, which runs the cluster's self-checks for /health?deep
    pub supervisor: Option<String>,
}

struct Args {
    model_dir: String,
    engine_sock: String,
    host: String,
    port: u16,
    served: Vec<String>,
    max_model_len: u64,
    reasoning_fields: String,
    prompt_tokens_details: bool,
    log_level: String,
    workers: usize,
    encode_threads: usize,
    default_max_tokens: u64,
    share_tokens: u64,
    max_connections: usize,
    default_effort: Option<&'static str>,
    default_budget: u64,
    supervisor: Option<String>,
}

const USAGE: &str = "usage: g6front-rs --model-dir DIR [--engine-sock PATH] [--host H] [--port N]
                  [--served-model-name NAME ...] [--max-model-len N]
                  [--reasoning-fields both|reasoning|reasoning_content]
                  [--enable-prompt-tokens-details] [--log-level LEVEL]
                  [--workers N] [--encode-threads N] [--default-max-tokens N]
                  [--share-tokens N] [--max-connections N]
                  [--default-reasoning-effort minimal|low|medium|high|xhigh|max]
                  [--default-thinking-budget N] [--supervisor HOST:PORT]";

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        model_dir: String::new(),
        engine_sock: "/run/g6/engine.sock".into(),
        host: "0.0.0.0".into(),
        port: 8100,
        served: vec!["glm-5.3".into(), "glm53".into()],
        max_model_len: 0,
        reasoning_fields: "reasoning".into(),
        prompt_tokens_details: false,
        log_level: "INFO".into(),
        workers: 2,
        encode_threads: 0,
        default_max_tokens: 0,
        share_tokens: 0,
        max_connections: 512,
        default_effort: None,
        default_budget: 0,
        supervisor: None,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    // accepts `--flag value` and `--flag=value`
    let next = |i: &mut usize, flag: &str, inline: Option<String>| -> Result<String, String> {
        if let Some(v) = inline {
            return Ok(v);
        }
        *i += 1;
        argv.get(*i).cloned().ok_or(format!("{flag} needs a value"))
    };
    while i < argv.len() {
        let (flag, inline) = match argv[i].split_once('=') {
            Some((f, v)) if argv[i].starts_with("--") => (f.to_string(), Some(v.to_string())),
            _ => (argv[i].clone(), None),
        };
        match flag.as_str() {
            "--model-dir" => a.model_dir = next(&mut i, &flag, inline)?,
            "--engine-sock" => a.engine_sock = next(&mut i, &flag, inline)?,
            "--host" => a.host = next(&mut i, &flag, inline)?,
            "--port" => a.port = next(&mut i, &flag, inline)?.parse().map_err(|_| "--port: not a number")?,
            "--max-model-len" => {
                a.max_model_len = next(&mut i, &flag, inline)?.parse().map_err(|_| "--max-model-len: not a number")?
            }
            "--reasoning-fields" => a.reasoning_fields = next(&mut i, &flag, inline)?,
            "--enable-prompt-tokens-details" => a.prompt_tokens_details = true,
            "--log-level" => a.log_level = next(&mut i, &flag, inline)?,
            "--workers" => a.workers = next(&mut i, &flag, inline)?.parse().map_err(|_| "--workers: not a number")?,
            "--default-max-tokens" => {
                a.default_max_tokens =
                    next(&mut i, &flag, inline)?.parse().map_err(|_| "--default-max-tokens: not a number")?
            }
            "--share-tokens" => {
                a.share_tokens = next(&mut i, &flag, inline)?.parse().map_err(|_| "--share-tokens: not a number")?
            }
            "--max-connections" => {
                a.max_connections = next(&mut i, &flag, inline)?.parse().map_err(|_| "--max-connections: not a number")?
            }
            "--default-reasoning-effort" => {
                a.default_effort = Some(thinking::parse_default_effort(&next(&mut i, &flag, inline)?)?)
            }
            "--default-thinking-budget" => {
                a.default_budget =
                    next(&mut i, &flag, inline)?.parse().map_err(|_| "--default-thinking-budget: not a number")?
            }
            "--supervisor" => a.supervisor = Some(next(&mut i, &flag, inline)?),
            "--encode-threads" => {
                a.encode_threads = next(&mut i, &flag, inline)?.parse().map_err(|_| "--encode-threads: not a number")?
            }
            "--served-model-name" => {
                a.served = match inline {
                    Some(v) => vec![v],
                    None => {
                        let mut v = Vec::new();
                        while i + 1 < argv.len() && !argv[i + 1].starts_with("--") {
                            i += 1;
                            v.push(argv[i].clone());
                        }
                        v
                    }
                };
                if a.served.is_empty() {
                    return Err("--served-model-name needs at least one name".into());
                }
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }
    if a.model_dir.is_empty() {
        return Err("--model-dir is required".into());
    }
    Ok(a)
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}\n{USAGE}");
            std::process::exit(2);
        }
    };
    util::init_logging(&args.log_level);
    // The head node also runs engine rank 0: keep the front end's own threads few. Both pools size from the CPU
    // affinity mask the launch scripts set (taskset), never from the whole machine.
    let avail = std::thread::available_parallelism().map_or(1, |n| n.get());
    let encode_threads = if args.encode_threads > 0 { args.encode_threads } else { (avail / 2).clamp(1, 8) };
    rayon::ThreadPoolBuilder::new()
        .num_threads(encode_threads)
        .thread_name(|i| format!("g6-encode-{i}"))
        .build_global()
        .expect("rayon pool");
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(args.workers.max(1))
        .max_blocking_threads(16)
        .thread_name("g6-front")
        .enable_all()
        .build()
        .expect("tokio runtime");
    log::info!("{} workers, {encode_threads} encode threads ({avail} CPUs allowed)", args.workers.max(1));
    if let Err(e) = rt.block_on(serve(args)) {
        log::error!("{e}");
        std::process::exit(1);
    }
}

async fn serve(args: Args) -> Result<(), String> {
    let reasoning_fields: &'static [&'static str] = match args.reasoning_fields.as_str() {
        "both" => &["reasoning", "reasoning_content"],
        "reasoning" => &["reasoning"],
        "reasoning_content" => &["reasoning_content"],
        other => return Err(format!("--reasoning-fields: invalid choice {other:?}")),
    };
    let dir = PathBuf::from(&args.model_dir);
    let t0 = std::time::Instant::now();
    let tok = tokio::task::spawn_blocking(move || {
        let t = tok::Tok::load(&dir);
        util::release_memory(true); // tokenizer.json parsing peaks far above what the loaded tokenizer keeps
        t
    })
    .await
    .map_err(|e| e.to_string())??;
    let template =
        template::ChatTemplate::load(&PathBuf::from(&args.model_dir), tok.eos_token.clone(), tok.pad_token.clone())?;
    log::info!("tokenizer + template loaded in {:.0} ms (vocab {})", t0.elapsed().as_secs_f64() * 1e3, tok.vocab_size);
    log::info!(
        "reasoning efforts the template names: {:?}; default effort {:?}, default thinking budget {}",
        template.levels,
        args.default_effort,
        args.default_budget
    );
    let engine = engine::Engine::new(args.engine_sock.clone());
    let st: &'static State = Box::leak(Box::new(State {
        tok,
        template,
        engine: engine.clone(),
        model_name: args.served[0].clone(),
        served: args.served,
        model_dir: args.model_dir,
        reasoning_fields,
        prompt_tokens_details: args.prompt_tokens_details,
        max_model_len_override: args.max_model_len,
        default_max_tokens: args.default_max_tokens,
        share_tokens: args.share_tokens,
        thinking: thinking::Defaults { effort: args.default_effort, budget: args.default_budget },
        started: util::now_secs(),
        supervisor: args.supervisor,
    }));
    engine.start();
    let addr: SocketAddr = format!("{}:{}", args.host, args.port)
        .parse()
        .or_else(|_| format!("[{}]:{}", args.host, args.port).parse())
        .map_err(|e| format!("bad --host/--port: {e}"))?;
    let listener = TcpListener::bind(addr).await.map_err(|e| format!("bind {addr}: {e}"))?;
    log::info!("listening on http://{addr} (engine socket {})", args.engine_sock);
    let mut term =
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).map_err(|e| e.to_string())?;
    // At most --max-connections at once: past it, new connections wait in the listen backlog. Each connection's
    // headers must arrive within hyper's header read timeout (30 s), which only runs with a timer set.
    let slots = Arc::new(tokio::sync::Semaphore::new(args.max_connections.max(1)));
    loop {
        let slot = tokio::select! {
            s = slots.clone().acquire_owned() => s.expect("semaphore closed"),
            _ = tokio::signal::ctrl_c() => break,
            _ = term.recv() => break,
        };
        let (sock, peer) = tokio::select! {
            r = listener.accept() => match r {
                Ok(x) => x,
                Err(e) => {
                    // out of file descriptors (EMFILE) fails every accept at once: pause instead of spinning
                    log::warn!("accept: {e}");
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    continue;
                }
            },
            _ = tokio::signal::ctrl_c() => break,
            _ = term.recv() => break,
        };
        let _ = sock.set_nodelay(true);
        tokio::spawn(async move {
            let _slot = slot;
            let svc = service_fn(move |req| api::route(st, req, peer));
            let conn = hyper::server::conn::http1::Builder::new()
                .timer(TokioTimer::new())
                .keep_alive(true)
                .serve_connection(TokioIo::new(sock), svc);
            if let Err(e) = conn.await {
                log::debug!("connection: {e}");
            }
        });
    }
    log::info!("shutting down");
    Ok(())
}
