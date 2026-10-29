//! The file side of `g6m4 --serve DIR`, so the launcher boots g6pseudo where it boots g6m4 and the real shim drives it.
//! frontend/g6s2shim.py documents the protocol and frontend/tests/fake_s2_engine.py is the other implementation of it.
//! Rank 0 serves DIR. The other ranks wait for DIR/quit, since the real head passes them each request itself.
//! What the real engine refuses (a bad header, epoch, hash or id order, or a request over the pool) writes DIR/abort.s2
//! and exits 3. DIR/quit exits 0.

use std::collections::{HashSet, VecDeque};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::thread::sleep;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::{fake_lp, plan, stop_at, Engine, MAX_LOGPROBS};

const PAGE: usize = 64;
/// Rows the engine takes in one step: a session's chunk must fit.
const MAX_ROWS: usize = 32;
const POLL: Duration = Duration::from_millis(2);

pub struct Config {
    pub dir: PathBuf,
    pub rank: usize,
    pub ranks: usize,
    pub epoch: u64,
    pub sessions: usize,
    /// The KV pool in rows: a request's prompt plus its n_gen must fit.
    pub cap: usize,
}

/// Why the epoch ends: the engine writes DIR/abort.s2 with it and exits 3.
struct Fail(String);

struct Request {
    n_gen: usize,
    temp: f64,
    seed: u64,
    min: usize,
    stop: HashSet<u32>,
    lp: Option<usize>,
    plp: Option<usize>,
    /// after, unless (-1: none), ids
    ins: Option<(usize, i64, Vec<u32>)>,
    prompt: Vec<u32>,
}

struct Session {
    id: u64,
    slot: usize,
    pages: usize,
    out: File,
    prompt: Vec<u32>,
    plp: Option<usize>,
    lp: bool,
    ids: Vec<u32>,
    lps: Vec<Value>,
    /// Ids the request emits: through its stop id, or all of them.
    end: usize,
    stop_id: Option<u32>,
    sent: usize,
    rounds: usize,
    next: Instant,
}

/// Serves `c.dir` until DIR/quit (Ok) or a refused command (Err, after DIR/abort.s2).
pub fn serve(e: &Engine, c: &Config) -> Result<(), String> {
    let quit = c.dir.join("quit");
    if c.rank != 0 {
        while !quit.exists() && !c.dir.join("abort.s2").exists() {
            sleep(Duration::from_millis(200));
        }
        return Ok(());
    }
    let mut s = Server::new(e, c);
    let r = s.run(&quit);
    if let Err(Fail(why)) = &r {
        let _ = fs::write(c.dir.join("abort.s2"), format!("g6pseudo: {why}\n"));
        return Err(why.clone());
    }
    Ok(())
}

struct Server<'a> {
    e: &'a Engine,
    c: &'a Config,
    next_cmd: u64,
    last_id: Option<u64>,
    queue: VecDeque<(u64, Request)>,
    live: Vec<Session>,
    generation: Vec<u64>,
    pages_used: usize,
    dirty: bool,
}

impl<'a> Server<'a> {
    fn new(e: &'a Engine, c: &'a Config) -> Self {
        Server {
            e,
            c,
            next_cmd: 0,
            last_id: None,
            queue: VecDeque::new(),
            live: Vec::new(),
            generation: vec![0; c.sessions],
            pages_used: 0,
            dirty: true,
        }
    }

    fn run(&mut self, quit: &Path) -> Result<(), Fail> {
        let d = &self.c.dir;
        for f in ["ready", "ready.s2", "abort.s2", "quit", "cmd.0", "req.0"] {
            if d.join(f).exists() {
                return Err(Fail(format!("{} is not a fresh S2 directory ({f})", d.display())));
            }
        }
        // mtp 0, so the guard is 0 too
        let ready = format!("G6S2 3 {} {} {} 0 0 {MAX_ROWS} {MAX_LOGPROBS}\n", self.c.epoch, self.c.ranks, self.c.cap);
        publish(&d.join("ready.s2"), &ready).map_err(Fail)?;
        eprintln!("g6pseudo: serving {} (epoch {}, {} sessions)", d.display(), self.c.epoch, self.c.sessions);
        while !quit.exists() {
            self.commands()?;
            self.admit().map_err(Fail)?;
            self.step().map_err(Fail)?;
            if self.dirty {
                self.status().map_err(Fail)?;
                self.dirty = false;
            }
            sleep(POLL);
        }
        Ok(())
    }

    /// Reads cmd.<k> in order, with no gap.
    fn commands(&mut self) -> Result<(), Fail> {
        loop {
            let p = self.c.dir.join(format!("cmd.{}", self.next_cmd));
            let Ok(text) = fs::read_to_string(&p) else { return Ok(()) };
            let w: Vec<&str> = text.split_whitespace().collect();
            let num = |i: usize| w.get(i).and_then(|x| x.parse::<u64>().ok());
            let (Some(epoch), Some(id), Some(hash)) = (num(1), num(2), num(3)) else {
                return Err(Fail(format!("invalid ingress command {text:?}")));
            };
            if w.len() != 4 || epoch != self.c.epoch {
                return Err(Fail(format!("invalid ingress command {text:?} (epoch {})", self.c.epoch)));
            }
            self.next_cmd += 1;
            match w[0] {
                "ADMIT" => self.on_admit(id, hash)?,
                "CANCEL" if hash == 0 => self.on_cancel(id).map_err(Fail)?,
                _ => return Err(Fail(format!("invalid ingress command {text:?}"))),
            }
        }
    }

    fn on_admit(&mut self, id: u64, hash: u64) -> Result<(), Fail> {
        if self.last_id.is_some_and(|l| id <= l) {
            return Err(Fail("request ids must increase".into()));
        }
        self.last_id = Some(id);
        let data = fs::read(self.c.dir.join(format!("req.{id}")))
            .map_err(|err| Fail(format!("req.{id} when its ADMIT arrived: {err}")))?;
        if fnv1a(&data) != hash {
            return Err(Fail(format!("req.{id}: the payload hash differs from the ADMIT's")));
        }
        let r = parse_request(&data, self.e.vocab).map_err(|err| Fail(format!("req.{id}: {err}")))?;
        if r.prompt.len() + r.n_gen > self.c.cap {
            return Err(Fail(format!("req.{id}: {} rows exceed the pool of {}", r.prompt.len() + r.n_gen, self.c.cap)));
        }
        self.queue.push_back((id, r));
        self.dirty = true;
        Ok(())
    }

    fn on_cancel(&mut self, id: u64) -> Result<(), String> {
        if let Some(i) = self.queue.iter().position(|(q, _)| *q == id) {
            self.queue.remove(i);
            self.dirty = true;
            return publish(&self.c.dir.join(format!("resp.{id}")), "end abort -1 0\n");
        }
        if let Some(i) = self.live.iter().position(|s| s.id == id) {
            let s = self.live.swap_remove(i);
            return self.finish(s, "abort");
        }
        Ok(())
    }

    /// Moves queued requests into free session slots while their KV fits, in order.
    fn admit(&mut self) -> Result<(), String> {
        while self.live.len() < self.c.sessions {
            let Some((_, r)) = self.queue.front() else { return Ok(()) };
            let pages = (r.prompt.len() + r.n_gen).div_ceil(PAGE);
            if self.pages_used + pages > self.c.cap / PAGE {
                return Ok(());
            }
            let (id, r) = self.queue.pop_front().unwrap();
            let slot = (0..self.c.sessions).find(|i| self.live.iter().all(|s| s.slot != *i)).unwrap();
            self.generation[slot] += 1;
            let mut out = File::create(self.c.dir.join(format!("resp.{id}"))).map_err(|err| err.to_string())?;
            writeln!(out, "a {} {slot} {} {} {pages}", self.c.epoch, self.generation[slot], self.pages_used)
                .map_err(|err| err.to_string())?;
            self.pages_used += pages;
            self.live.push(self.start(id, slot, pages, out, r));
            self.dirty = true;
        }
        Ok(())
    }

    fn start(&self, id: u64, slot: usize, pages: usize, out: File, r: Request) -> Session {
        let mut frame = json!({"temperature": r.temp, "seed": r.seed});
        if let Some((after, unless, ids)) = &r.ins {
            let unless = if *unless < 0 { Value::Null } else { Value::from(*unless) };
            frame["insert"] = json!([{"after_tokens": after, "unless_emitted": unless, "tokens": ids}]);
        }
        let p = plan(self.e, &r.prompt, &frame, r.n_gen, r.lp);
        let stop = stop_at(&p.ids, &r.stop, r.min);
        let eng = &self.e.rules.engine;
        let prefill = if eng.prefill_tps > 0.0 {
            Duration::from_secs_f64(r.prompt.len() as f64 / eng.prefill_tps)
        } else {
            Duration::ZERO
        };
        Session {
            id,
            slot,
            pages,
            out,
            plp: r.plp,
            lp: r.lp.is_some(),
            end: stop.map_or(p.ids.len(), |i| i + 1),
            stop_id: stop.map(|i| p.ids[i]),
            ids: p.ids,
            lps: p.lps,
            prompt: r.prompt,
            sent: 0,
            rounds: 0,
            next: Instant::now() + prefill,
        }
    }

    /// Commits the next chunk of every session that is due, and ends those that are done.
    fn step(&mut self) -> Result<(), String> {
        let e = self.e;
        let eng = &e.rules.engine;
        let pace = if eng.tokens_per_s > 0.0 { eng.chunk as f64 / eng.tokens_per_s } else { 0.0 };
        let now = Instant::now();
        let mut i = 0;
        while i < self.live.len() {
            let s = &mut self.live[i];
            if s.next > now {
                i += 1;
                continue;
            }
            let mut text = String::new();
            if s.rounds == 0 {
                if let Some(k) = s.plp {
                    for j in 1..s.prompt.len() {
                        text += &format!("p {j} {}\n", lp_fields(&fake_lp(s.prompt[j], k, e.vocab, -1.0)));
                    }
                }
            }
            let to = (s.sent + eng.chunk.min(MAX_ROWS)).min(s.end);
            if s.lp {
                for lp in &s.lps[s.sent..to] {
                    text += &format!("l {}\n", lp_fields(lp));
                }
            }
            if to > s.sent {
                let ids: Vec<String> = s.ids[s.sent..to].iter().map(u32::to_string).collect();
                text += &format!("t {}\n", ids.join(" "));
            }
            s.out.write_all(text.as_bytes()).map_err(|err| err.to_string())?;
            s.sent = to;
            s.rounds += 1;
            s.next = now + Duration::from_secs_f64(pace);
            if s.sent == s.end {
                let s = self.live.swap_remove(i);
                let reason = if s.stop_id.is_some() { "stop" } else { "length" };
                self.finish(s, reason)?;
            } else {
                i += 1;
            }
        }
        Ok(())
    }

    fn finish(&mut self, mut s: Session, reason: &str) -> Result<(), String> {
        let stop = if reason == "stop" { s.stop_id.map_or(-1, i64::from) } else { -1 };
        write!(s.out, "m {} 0 0\nend {reason} {stop} {}\n", s.rounds, s.sent).map_err(|err| err.to_string())?;
        self.pages_used -= s.pages;
        self.dirty = true;
        Ok(())
    }

    fn status(&self) -> Result<(), String> {
        let free = self.c.cap / PAGE - self.pages_used;
        let held: usize = self.live.iter().map(|s| (s.prompt.len() + s.sent).div_ceil(PAGE)).sum();
        let body = json!({"epoch": self.c.epoch, "resident": self.live.len(), "queued": self.queue.len(),
            "retiring": 0, "reserved_pages": self.pages_used, "held_pages": held, "free_pages": free,
            "largest_free_run": free, "admission_stalls": 0, "fragmentation_stalls": 0});
        publish(&self.c.dir.join("status.s2"), &format!("{body}\n"))
    }
}

/// Writes `path` whole: a temporary file, then a rename.
fn publish(path: &Path, text: &str) -> Result<(), String> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, text).and_then(|_| fs::rename(&tmp, path)).map_err(|err| format!("{}: {err}", path.display()))
}

/// The engine's payload hash (engine/runtime/s2/ingress.h).
fn fnv1a(data: &[u8]) -> u64 {
    data.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &c| (h ^ c as u64).wrapping_mul(0x100_0000_01b3))
}

/// A logprob entry {lp, rank, top} as the resp line's `<lp> <rank> <n> <id>:<lp> ...` (rank 0: outside the top list).
fn lp_fields(v: &Value) -> String {
    let top = v["top"].as_array().map_or(&[][..], Vec::as_slice);
    let mut s = format!("{} {} {}", v["lp"].as_f64().unwrap_or(-9999.0), v["rank"].as_u64().unwrap_or(0), top.len());
    for t in top {
        s += &format!(" {}:{}", t[0], t[1].as_f64().unwrap_or(-9999.0));
    }
    s
}

/// req.<id>: `n_gen N temp F seed U min M stop ids|- [top_p F] [lp K] [plp K] [sys N] [ins A:U:ids]`, then the prompt.
fn parse_request(data: &[u8], vocab: u32) -> Result<Request, String> {
    let text = std::str::from_utf8(data).map_err(|_| "not UTF-8")?;
    let (head, prompt) = text.split_once('\n').ok_or("no prompt line")?;
    let w: Vec<&str> = head.split_whitespace().collect();
    if w.len() < 10
        || !w.len().is_multiple_of(2)
        || [w[0], w[2], w[4], w[6], w[8]] != ["n_gen", "temp", "seed", "min", "stop"]
    {
        return Err(format!("invalid request header {head:?}"));
    }
    let num = |s: &str| s.parse::<usize>().map_err(|_| format!("bad number {s:?}"));
    let mut r = Request {
        n_gen: num(w[1])?,
        temp: w[3].parse().map_err(|_| format!("bad temp {:?}", w[3]))?,
        seed: num(w[5])? as u64,
        min: num(w[7])?,
        stop: if w[9] == "-" { HashSet::new() } else { ids(w[9])?.into_iter().collect() },
        lp: None,
        plp: None,
        ins: None,
        prompt: ids(prompt.trim())?,
    };
    for kv in w[10..].chunks(2) {
        match kv[0] {
            "top_p" | "sys" => {}
            "lp" => r.lp = Some(num(kv[1])?.min(MAX_LOGPROBS)),
            "plp" => r.plp = Some(num(kv[1])?.min(MAX_LOGPROBS)),
            "ins" => {
                let mut f = kv[1].splitn(3, ':');
                let (a, u, t) = (f.next().unwrap_or(""), f.next().unwrap_or(""), f.next().unwrap_or(""));
                let unless = if u == "-1" { -1 } else { num(u)? as i64 };
                r.ins = Some((num(a)?, unless, ids(t)?));
            }
            k => return Err(format!("{k}: the pseudo engine has no such request field")),
        }
    }
    if r.n_gen == 0 || r.min > r.n_gen || r.prompt.is_empty() || r.prompt.iter().any(|&t| t >= vocab) {
        return Err("invalid sampling, length or prompt".into());
    }
    Ok(r)
}

fn ids(s: &str) -> Result<Vec<u32>, String> {
    s.split(',').map(|x| x.parse::<u32>().map_err(|_| format!("bad id {x:?}"))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_file() {
        let r =
            parse_request(b"n_gen 64 temp 0.7 seed 9 min 2 stop 5,6 top_p 1 lp 3 ins 4:-1:7,8\n1,2,3\n", 100).unwrap();
        assert_eq!((r.n_gen, r.seed, r.min, r.lp, r.plp), (64, 9, 2, Some(3), None));
        assert_eq!(r.stop, HashSet::from([5, 6]));
        assert_eq!(r.ins, Some((4, -1, vec![7, 8])));
        assert_eq!(r.prompt, vec![1, 2, 3]);
        assert!(parse_request(b"n_gen 64 temp 0 seed 0 min 0 stop - gram 1:0:-1:\n1\n", 100).is_err());
        assert!(parse_request(b"n_gen 64 temp 0 seed 0 min 0 stop -\n1,200\n", 100).is_err());
    }
}
