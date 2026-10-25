//! g6pseudo: a pseudo engine. It speaks PROTOCOL.md in GLM-5.3 tokens like g6m4, and answers from its weights, a TOML
//! file of rules (src/rules.rs), so the front end, the shim and the launcher run end to end on a CPU.
//!
//!   g6pseudo --model DIR [--sock PATH]      DIR holds responses.toml (the weights) and the GLM tokenizer files, at its
//!                                           top level or under tokenizer/
//!   g6pseudo --model DIR --rank R --serve S2DIR --s2-epoch E [--s2-sessions N] [--cap ROWS] [--ips A,B,..]
//!                                           the S2 file protocol in place of the socket, as g6m4 --serve (src/s2.rs).
//!                                           It takes the rest of g6m4's command line and ignores it
//!   g6pseudo --weights FILE --check         load and compile the rules, print how many
//!   g6pseudo --weights FILE --explain TEXT  the candidates for a user message TEXT, their weights and the winner's reply
//!
//! Deterministic: the reply is a function of the prompt, the request's temperature and seed, and the weights.

mod prompt;
mod rules;
mod s2;

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Value};
use tokenizers::Tokenizer;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixListener;
use tokio::sync::mpsc;

use prompt::Prompt;
use rules::{weights, Candidate, Rules};

const MAX_FRAME: usize = 256 << 20;
const MAX_LOGPROBS: usize = 20;
/// The probability mass the candidates leave to the rest of the vocabulary (so a top list never sums to 1).
const REST: f64 = 1e-3;
/// Candidates whose replies the logprobs follow.
const LP_CANDIDATES: usize = 8;

struct Engine {
    rules: Rules,
    tok: Tokenizer,
    eos: Vec<u32>,
    vocab: u32,
    c: Counters,
}

#[derive(Default)]
struct Counters {
    prompt: AtomicU64,
    gen: AtomicU64,
    running: AtomicU64,
    stop: AtomicU64,
    length: AtomicU64,
    abort: AtomicU64,
    error: AtomicU64,
}

/// What one request emits: its ids, the logprob entry of each (when asked), and how it ends.
struct Plan {
    ids: Vec<u32>,
    lps: Vec<Value>,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("g6pseudo: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut model: Option<PathBuf> = None;
    let mut weights_path: Option<PathBuf> = None;
    let mut sock = PathBuf::from("/run/g6/engine.sock");
    let (mut check, mut explain) = (false, None);
    let mut s2 = s2::Config { dir: PathBuf::new(), rank: 0, ranks: 1, epoch: 0, sessions: 8, cap: 0 };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut val = || it.next().ok_or(format!("{a} needs a value"));
        let num = |v: String| v.parse::<usize>().map_err(|_| format!("{a} {v}: want a whole number"));
        match a.as_str() {
            "--model" => model = Some(PathBuf::from(val()?)),
            "--weights" => weights_path = Some(PathBuf::from(val()?)),
            "--sock" => sock = PathBuf::from(val()?),
            "--check" => check = true,
            "--explain" => explain = Some(val()?),
            "--serve" => s2.dir = PathBuf::from(val()?),
            "--rank" => s2.rank = num(val()?)?,
            "--s2-epoch" => s2.epoch = num(val()?)? as u64,
            "--s2-sessions" => s2.sessions = num(val()?)?.max(1),
            "--cap" => s2.cap = num(val()?)?,
            "--ips" => s2.ranks = val()?.split(',').count(),
            // g6m4's: the fabric, the pinned CPU, the S2 port and the model set id
            "--ips2" | "--pf-ips2" | "--proxy-cpu" | "--s2-port" | "--set-id" => {
                val()?;
            }
            _ => return Err(format!("unknown argument {a} (see the header of src/main.rs)")),
        }
    }
    let wpath = weights_path
        .or_else(|| model.as_ref().map(|m| m.join("responses.toml")))
        .ok_or("--model DIR or --weights FILE")?;
    let rules = Rules::load(&wpath)?;
    if check {
        println!("{}: {} rules and a fallback", wpath.display(), rules.rules.len());
        return Ok(());
    }
    if let Some(text) = explain {
        print!("{}", explain_text(&rules, &text));
        return Ok(());
    }
    let model = model.ok_or("--model DIR: serving needs the tokenizer files")?;
    let dir = [model.join("tokenizer"), model.clone()]
        .into_iter()
        .find(|d| d.join("tokenizer.json").is_file())
        .ok_or(format!("no tokenizer.json in {0}/tokenizer or {0}", model.display()))?;
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| format!("tokenizer.json: {e}"))?;
    let eos = eos_ids(&dir)?;
    let vocab = tok.get_vocab_size(true) as u32;
    let e = Arc::new(Engine { rules, tok, eos, vocab, c: Counters::default() });
    if !s2.dir.as_os_str().is_empty() {
        if s2.cap == 0 {
            s2.cap = e.rules.engine.max_model_len as usize;
        }
        return s2::serve(&e, &s2);
    }
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?
        .block_on(serve(e, &sock))
}

/// generation_config.json's eos_token_id: one id or a list.
fn eos_ids(dir: &Path) -> Result<Vec<u32>, String> {
    let p = dir.join("generation_config.json");
    let v: Value = serde_json::from_str(&std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?)
        .map_err(|e| format!("{}: {e}", p.display()))?;
    Ok(match &v["eos_token_id"] {
        Value::Number(n) => vec![n.as_u64().unwrap_or(0) as u32],
        Value::Array(a) => a.iter().filter_map(Value::as_u64).map(|x| x as u32).collect(),
        _ => Vec::new(),
    })
}

fn explain_text(rules: &Rules, user: &str) -> String {
    let p = Prompt::parse(&format!("<|user|>{user}<|assistant|><think>"));
    let c = rules.candidates(&p);
    let w = weights(&c, 1.0);
    let mut out = String::new();
    for (cand, w) in c.iter().zip(&w) {
        out += &format!(
            "{:<28} specificity {:>4}  match {:>4} bytes  weight {w:.4}\n",
            cand.name(),
            cand.rule.map_or(0, |r| r.specificity),
            cand.len
        );
    }
    out += &format!("reply: {:?}\n", rules.reply(&c[0], &p).text);
    out
}

async fn serve(e: Arc<Engine>, sock: &Path) -> Result<(), String> {
    let _ = std::fs::remove_file(sock);
    let l = UnixListener::bind(sock).map_err(|err| format!("{}: {err}", sock.display()))?;
    eprintln!("g6pseudo: {} rules, listening on {}", e.rules.rules.len(), sock.display());
    loop {
        let (s, _) = l.accept().await.map_err(|err| err.to_string())?;
        let e = e.clone();
        tokio::spawn(async move {
            if let Err(err) = connection(e, s).await {
                eprintln!("g6pseudo: connection closed: {err}");
            }
        });
    }
}

async fn connection(e: Arc<Engine>, s: tokio::net::UnixStream) -> Result<(), String> {
    let (mut rd, mut wr) = s.into_split();
    let (tx, mut rx) = mpsc::unbounded_channel::<Value>();
    let writer = tokio::spawn(async move {
        while let Some(v) = rx.recv().await {
            let b = serde_json::to_vec(&v).unwrap_or_default();
            if wr.write_all(&(b.len() as u32).to_be_bytes()).await.is_err() || wr.write_all(&b).await.is_err() {
                break;
            }
        }
    });
    let mut hello = json!({"type": "hello", "version": 1, "model": e.rules.engine.model,
        "max_model_len": e.rules.engine.max_model_len, "vocab_size": e.vocab, "eos_ids": e.eos,
        "max_logprobs": MAX_LOGPROBS, "insert": true});
    hello["pseudo"] = Value::Bool(true);
    let _ = tx.send(hello);
    let live: Arc<Mutex<HashMap<String, Arc<AtomicBool>>>> = Arc::default();
    loop {
        let mut len = [0u8; 4];
        if rd.read_exact(&mut len).await.is_err() {
            break;
        }
        let n = u32::from_be_bytes(len) as usize;
        if n > MAX_FRAME {
            return Err(format!("frame of {n} bytes"));
        }
        let mut buf = vec![0u8; n];
        rd.read_exact(&mut buf).await.map_err(|err| err.to_string())?;
        let v: Value = serde_json::from_slice(&buf).map_err(|err| format!("bad frame: {err}"))?;
        match v["type"].as_str() {
            Some("generate") => {
                let id = v["id"].as_str().unwrap_or("").to_string();
                let cancel = Arc::new(AtomicBool::new(false));
                if live.lock().unwrap().insert(id.clone(), cancel.clone()).is_some() {
                    let _ = tx.send(json!({"type": "finish", "id": id, "reason": "error", "stop_id": null,
                        "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0, "message": "duplicate id"}));
                    continue;
                }
                let (e, tx, live) = (e.clone(), tx.clone(), live.clone());
                tokio::spawn(async move {
                    generate(&e, &v, &cancel, &tx).await;
                    live.lock().unwrap().remove(&id);
                });
            }
            Some("cancel") => {
                if let Some(c) = live.lock().unwrap().get(v["id"].as_str().unwrap_or("")) {
                    c.store(true, Ordering::Relaxed);
                }
            }
            Some("metrics") => {
                let _ = tx.send(metrics(&e, v["seq"].clone()));
            }
            _ => {}
        }
    }
    drop(tx);
    let _ = writer.await;
    Ok(())
}

fn metrics(e: &Engine, seq: Value) -> Value {
    let g = |a: &AtomicU64| a.load(Ordering::Relaxed);
    json!({"type": "metrics", "seq": seq,
        "requests": {"running": g(&e.c.running), "waiting": 0,
                     "finished": {"stop": g(&e.c.stop), "length": g(&e.c.length), "abort": g(&e.c.abort), "error": g(&e.c.error)}},
        "tokens": {"prompt": g(&e.c.prompt), "generation": g(&e.c.gen)}})
}

async fn generate(e: &Engine, v: &Value, cancel: &AtomicBool, tx: &mpsc::UnboundedSender<Value>) {
    let id = v["id"].as_str().unwrap_or("").to_string();
    let prompt: Vec<u32> =
        v["prompt"].as_array().map_or(Vec::new(), |a| a.iter().filter_map(Value::as_u64).map(|x| x as u32).collect());
    let max_tokens = v["max_tokens"].as_u64().unwrap_or(16).max(1) as usize;
    let min_tokens = v["min_tokens"].as_u64().unwrap_or(0) as usize;
    let stop_ids: HashSet<u32> = v["stop_ids"]
        .as_array()
        .map_or(HashSet::new(), |a| a.iter().filter_map(Value::as_u64).map(|x| x as u32).collect());
    let lp_k = v["logprobs"].as_u64().map(|k| (k as usize).min(MAX_LOGPROBS));
    let finish = |reason: &str, stop: Option<u32>, n: usize| {
        json!({"type": "finish", "id": id, "reason": reason, "stop_id": stop, "prompt_tokens": prompt.len(),
               "cached_tokens": 0, "completion_tokens": n})
    };
    if prompt.is_empty() || prompt.iter().any(|&t| t >= e.vocab) {
        e.c.error.fetch_add(1, Ordering::Relaxed);
        let mut f = finish("error", None, 0);
        f["message"] = Value::from("prompt is empty or holds an id outside the vocabulary");
        let _ = tx.send(f);
        return;
    }
    e.c.prompt.fetch_add(prompt.len() as u64, Ordering::Relaxed);
    e.c.running.fetch_add(1, Ordering::Relaxed);
    if let Some(k) = v["prompt_logprobs"].as_u64() {
        let k = (k as usize).min(MAX_LOGPROBS);
        let lps: Vec<Value> = (0..prompt.len())
            .map(|j| if j == 0 { Value::Null } else { fake_lp(prompt[j], k, e.vocab, -1.0) })
            .collect();
        let _ = tx.send(json!({"type": "prompt_logprobs", "id": id, "lps": lps}));
    }
    let plan = plan(e, &prompt, v, max_tokens, lp_k);
    let eng = &e.rules.engine;
    if eng.prefill_tps > 0.0 {
        tokio::time::sleep(Duration::from_secs_f64(prompt.len() as f64 / eng.prefill_tps)).await;
    }
    let pace =
        if eng.tokens_per_s > 0.0 { Some(Duration::from_secs_f64(eng.chunk as f64 / eng.tokens_per_s)) } else { None };
    let stop = stop_at(&plan.ids, &stop_ids, min_tokens);
    let last = stop.map_or(plan.ids.len(), |i| i + 1);
    let (mut reason, stop_id) = match stop {
        Some(i) => ("stop", Some(plan.ids[i])),
        None => ("length", None),
    };
    let mut sent = 0usize;
    while sent < last {
        if cancel.load(Ordering::Relaxed) {
            reason = "abort";
            break;
        }
        let end = (sent + eng.chunk).min(last);
        let mut f = json!({"type": "tokens", "id": id, "ids": &plan.ids[sent..end]});
        if lp_k.is_some() {
            f["logprobs"] = Value::Array(plan.lps[sent..end].to_vec());
        }
        let _ = tx.send(f);
        e.c.gen.fetch_add((end - sent) as u64, Ordering::Relaxed);
        sent = end;
        if let Some(d) = pace.filter(|_| sent < last) {
            tokio::time::sleep(d).await;
        }
    }
    let stop_id = if reason == "stop" { stop_id } else { None };
    let counter = match reason {
        "stop" => &e.c.stop,
        "abort" => &e.c.abort,
        _ => &e.c.length,
    };
    counter.fetch_add(1, Ordering::Relaxed);
    e.c.running.fetch_sub(1, Ordering::Relaxed);
    let _ = tx.send(finish(reason, stop_id, sent));
}

/// The index of the stop id that ends a reply: the first one past `min_tokens`. The request emits it last.
fn stop_at(ids: &[u32], stop: &HashSet<u32>, min_tokens: usize) -> Option<usize> {
    (min_tokens..ids.len()).find(|&i| stop.contains(&ids[i]))
}

/// The ids a request emits: the chosen candidate's reply, an insert rule applied, cut to max_tokens.
fn plan(e: &Engine, prompt: &[u32], v: &Value, max_tokens: usize, lp_k: Option<usize>) -> Plan {
    let text = e.tok.decode(prompt, false).unwrap_or_default();
    let p = Prompt::parse(&text);
    let cands = e.rules.candidates(&p);
    let temperature = v["temperature"].as_f64().unwrap_or(0.0);
    let pick = if temperature > 0.0 {
        sample(&weights(&cands, temperature), v["seed"].as_u64().unwrap_or(0) ^ hash(prompt))
    } else {
        0
    };
    let mut ids = reply_ids(e, &cands[pick], &p, max_tokens);
    if let Some(rule) = v["insert"].as_array().and_then(|a| a.first()) {
        let after = rule["after_tokens"].as_u64().unwrap_or(0) as usize;
        let unless = rule["unless_emitted"].as_u64().map(|x| x as u32);
        let ins: Vec<u32> = rule["tokens"]
            .as_array()
            .map_or(Vec::new(), |a| a.iter().filter_map(Value::as_u64).map(|x| x as u32).collect());
        if after < ids.len() && unless.is_none_or(|u| !ids[..after].contains(&u)) {
            // the reply goes on from the inserted tokens as from a prompt that ended with them
            let mut head = ids[..after].to_vec();
            head.extend(&ins);
            let mut cont = prompt.to_vec();
            cont.extend(&head);
            let cp = Prompt::parse(&e.tok.decode(&cont, false).unwrap_or_default());
            let cc = e.rules.candidates(&cp);
            head.extend(reply_ids(e, &cc[0], &cp, max_tokens));
            ids = head;
        }
    }
    ids.truncate(max_tokens);
    let lps = match lp_k {
        Some(k) => logprobs(e, &cands, &p, &ids, k, max_tokens),
        None => Vec::new(),
    };
    Plan { ids, lps }
}

fn reply_ids(e: &Engine, c: &Candidate, p: &Prompt, max_tokens: usize) -> Vec<u32> {
    let r = e.rules.reply(c, p);
    let mut ids = encode(e, &r.text);
    if let Some(f) = r.fill {
        let f = encode(e, &f);
        while ids.len() < max_tokens && !f.is_empty() {
            ids.extend(&f);
        }
    }
    ids
}

fn encode(e: &Engine, text: &str) -> Vec<u32> {
    e.tok.encode(text, false).map(|x| x.get_ids().to_vec()).unwrap_or_default()
}

/// Logprobs from the candidates: at each position the emitted id's probability is the weight of the candidates still
/// in agreement with the reply so far that emit it, over the weight of all of them still in agreement; the top list is
/// the ids those candidates emit there, best first, then made-up ids from the rest of the vocabulary.
fn logprobs(e: &Engine, cands: &[Candidate], p: &Prompt, ids: &[u32], k: usize, max_tokens: usize) -> Vec<Value> {
    let n = cands.len().min(LP_CANDIDATES);
    let w = weights(&cands[..n], 1.0);
    let seqs: Vec<Vec<u32>> = cands[..n].iter().map(|c| reply_ids(e, c, p, max_tokens)).collect();
    let mut alive: Vec<bool> = vec![true; n];
    let mut out = Vec::with_capacity(ids.len());
    for (i, &id) in ids.iter().enumerate() {
        let mut mass: Vec<(u32, f64)> = Vec::new();
        for j in 0..n {
            if !alive[j] || i >= seqs[j].len() || w[j] <= 0.0 {
                continue;
            }
            match mass.iter_mut().find(|(t, _)| *t == seqs[j][i]) {
                Some(m) => m.1 += w[j],
                None => mass.push((seqs[j][i], w[j])),
            }
        }
        let total: f64 = mass.iter().map(|m| m.1).sum();
        mass.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        // PROTOCOL.md: -inf goes on the wire as -9999.0
        let lp_of = |m: f64| if total > 0.0 { (m / total * (1.0 - REST)).ln().max(-9999.0) } else { REST.ln() };
        let mut top: Vec<(u32, f64)> = mass.iter().map(|&(t, m)| (t, lp_of(m))).collect();
        let mut filler = 1u32;
        while top.len() < k {
            let t = (id.wrapping_add(filler * 7919)) % e.vocab;
            filler += 1;
            if !top.iter().any(|x| x.0 == t) {
                top.push((t, (REST / (k as f64 + 1.0)).ln() - filler as f64 * 0.01));
            }
        }
        top.truncate(k);
        let (lp, rank) = match mass.iter().position(|m| m.0 == id) {
            Some(r) => (lp_of(mass[r].1), Value::from(r + 1)),
            None => ((REST / (k as f64 + 1.0)).ln(), Value::Null),
        };
        out.push(json!({"lp": lp, "rank": rank, "top": top.iter().map(|(t, l)| json!([t, l])).collect::<Vec<_>>()}));
        for j in 0..n {
            if alive[j] && seqs[j].get(i) != Some(&id) {
                alive[j] = false;
            }
        }
    }
    out
}

/// A made-up but proper logprob entry for `id` (prompt logprobs, which the rules say nothing about).
fn fake_lp(id: u32, k: usize, vocab: u32, lp: f64) -> Value {
    let top: Vec<Value> = (0..k).map(|j| json!([(id + 1 + j as u32 * 31) % vocab, lp - 1.0 - j as f64])).collect();
    json!({"lp": lp, "rank": Value::Null, "top": top})
}

fn hash(ids: &[u32]) -> u64 {
    ids.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x100_0000_01b3))
}

/// An index drawn from `w` by a splitmix64 of `seed`.
fn sample(w: &[f64], seed: u64) -> usize {
    let mut z = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    let u = ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64;
    let mut acc = 0.0;
    for (i, x) in w.iter().enumerate() {
        acc += x;
        if u < acc {
            return i;
        }
    }
    w.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_follows_the_weights_and_the_seed() {
        let w = [0.7, 0.2, 0.1];
        assert_eq!(sample(&w, 42), sample(&w, 42));
        let mut counts = [0usize; 3];
        for s in 0..10_000 {
            counts[sample(&w, s)] += 1;
        }
        assert!(counts[0] > 6500 && counts[0] < 7500 && counts[2] > 700 && counts[2] < 1300, "{counts:?}");
    }
}
