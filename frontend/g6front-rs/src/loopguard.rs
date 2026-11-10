//! The loop guard (`--default-loop-guard`, a request's `loop_guard`): a reply that falls into a verbatim loop inside
//! its think block gets a steering phrase, and if need be the think block's close, put into its reasoning. The reply
//! goes on in a new engine run, the way a front-end thinking budget continues after its close (gen.rs).
//!
//! The detector reads only the reply's own thinking tokens. It hashes every window of `min` tokens (a rolling hash, so
//! one multiply-add per token) and keeps, per window, where it last occurred and how many times it has occurred
//! without overlapping an earlier counted occurrence. It fires when one window reaches `copies` occurrences, or, with
//! `long` set, when the reply's tail has repeated one earlier stretch verbatim for `long` tokens. A window the prompt
//! also holds never fires, so quoting a file from the prompt, however often, is not a loop. The prompt's windows are
//! hashed only when a window first reaches the trigger, so a reply that never comes close costs nothing more.
//!
//! A loop inside the think block repeats the reply's own text: once in one, GLM-5.3-Flash gives the next copy p ~ 1.0.
//! Healthy reasoning does repeat itself (a function drafted twice, a line of code checked again), so the trigger wants
//! a third copy rather than a second, and the single-repeat trigger (`long`) is off by default: one healthy agent
//! step redrafted a 600-token stretch of code verbatim.

use std::collections::{HashMap, HashSet};

/// The default steering phrase.
pub const DEFAULT_TEXT: &str =
    "\n\nI'm going in circles. I already have enough to act. I'll decide now and stop re-checking.\n\n";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// log each trigger, change nothing
    Log,
    /// put the phrase in, still inside the think block
    Phrase,
    /// the phrase; a second trigger within `window` thinking tokens of it puts the phrase and the close in
    PhraseClose,
    /// the phrase and the close at once
    Close,
}

impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Mode::Log => "log",
            Mode::Phrase => "phrase",
            Mode::PhraseClose => "phrase-close",
            Mode::Close => "close",
        }
    }
}

/// `mode=M,min=N,copies=C,long=L,max=X,window=W,text=T`, any of the parts, each once; `text=` comes last.
#[derive(Clone, Debug, PartialEq)]
pub struct Spec {
    pub mode: Mode,
    /// the window length in tokens
    pub min: u32,
    /// occurrences of one window that fire the guard
    pub copies: u32,
    /// a verbatim repeat of this many tokens fires it too; 0 = off
    pub long: u32,
    /// triggers per request
    pub max: u32,
    /// phrase-close: a trigger this many thinking tokens or fewer after the phrase closes the block. log: the
    /// least distance between two logged triggers
    pub window: u32,
    /// the steering phrase
    pub text: String,
}

impl Default for Spec {
    fn default() -> Spec {
        Spec { mode: Mode::Phrase, min: 64, copies: 3, long: 0, max: 2, window: 512, text: DEFAULT_TEXT.into() }
    }
}

/// `text=`'s escapes: `\n`, `\t` and `\\`. Any other backslash is an error, so a later escape can be added.
fn unescape(s: &str) -> Result<String, String> {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('\\') => out.push('\\'),
            Some(x) => return Err(format!("text: unknown escape \\{x} (want \\n, \\t or \\\\)")),
            None => return Err("text: a lone \\ at the end".into()),
        }
    }
    Ok(out)
}

impl Spec {
    /// Parse a spec. `text=` takes the rest of the string, commas included, so it must be the last part.
    pub fn parse(s: &str) -> Result<Spec, String> {
        let mut spec = Spec::default();
        let (head, text) = match s.find("text=").filter(|&i| i == 0 || s[..i].ends_with(',')) {
            Some(i) => (s[..i].trim_end_matches(','), Some(&s[i + 5..])),
            None => (s, None),
        };
        let mut seen: Vec<&str> = Vec::new();
        for part in head.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            let (key, value) = part.split_once('=').ok_or_else(|| format!("{part:?}: want key=value"))?;
            let (key, value) = (key.trim(), value.trim());
            if seen.contains(&key) {
                return Err(format!("{key} is given twice"));
            }
            seen.push(key);
            let count = |lo: u32, hi: u32| -> Result<u32, String> {
                match value.parse::<u32>() {
                    Ok(v) if (lo..=hi).contains(&v) => Ok(v),
                    _ => Err(format!("{key}={value:?}: want a whole number from {lo} to {hi}")),
                }
            };
            match key {
                "mode" => {
                    spec.mode = match value {
                        "log" => Mode::Log,
                        "phrase" => Mode::Phrase,
                        "phrase-close" => Mode::PhraseClose,
                        "close" => Mode::Close,
                        _ => return Err(format!("mode={value:?}: want log, phrase, phrase-close or close")),
                    }
                }
                "min" => spec.min = count(4, 4096)?,
                "copies" => spec.copies = count(2, 1000)?,
                "long" => spec.long = count(0, 1 << 20)?,
                "max" => spec.max = count(1, 64)?,
                "window" => spec.window = count(0, 1 << 20)?,
                _ => return Err(format!("{key:?}: want mode, min, copies, long, max, window or text")),
            }
        }
        if let Some(t) = text {
            spec.text = unescape(t)?;
            if spec.text.is_empty() || spec.text.chars().count() > 2000 {
                return Err("text: want 1 to 2000 characters".into());
            }
        }
        if spec.long != 0 && spec.long < spec.min {
            return Err(format!("long={}: want 0 (off) or at least min ({})", spec.long, spec.min));
        }
        Ok(spec)
    }
}

const B: u64 = 0x9e37_79b9_7f4a_7c15;

/// One window's record: the end of its last occurrence, the end of its last counted (non-overlapping) one, and the
/// count. Token positions are u32: a reply is far shorter than 4G tokens.
struct Entry {
    seen: u32,
    counted: u32,
    count: u32,
}

/// What fired: at thinking token `at` (0-based, the last token of the window), the tail repeats a stretch `period`
/// tokens back for `run` tokens, and its last window has occurred `copies` times.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hit {
    pub at: usize,
    pub run: usize,
    pub period: usize,
    pub copies: u32,
}

pub struct Detector {
    min: usize,
    copies: u32,
    long: usize,
    toks: Vec<u32>,
    hash: u64,
    /// B^min, to take the token leaving the window out of the hash
    pow: u64,
    windows: HashMap<u64, Entry>,
    /// the current verbatim repeat: its offset back, and how many windows in a row matched at that offset
    period: usize,
    streak: usize,
    /// the prompt's window hashes, built the first time a window reaches the trigger
    prompt: Option<HashSet<u64>>,
}

impl Detector {
    pub fn new(spec: &Spec) -> Detector {
        let min = spec.min as usize;
        Detector {
            min,
            copies: spec.copies,
            long: spec.long as usize,
            toks: Vec::new(),
            hash: 0,
            pow: (0..min).fold(1u64, |p, _| p.wrapping_mul(B)),
            windows: HashMap::new(),
            period: 0,
            streak: 0,
            prompt: None,
        }
    }

    /// Thinking tokens seen so far.
    pub fn len(&self) -> usize {
        self.toks.len()
    }

    fn prompt_has(&mut self, prompt: &[u32], h: u64) -> bool {
        let (min, pow) = (self.min, self.pow);
        let set = self.prompt.get_or_insert_with(|| {
            let mut set = HashSet::with_capacity(prompt.len());
            let mut x = 0u64;
            for (i, &t) in prompt.iter().enumerate() {
                x = x.wrapping_mul(B).wrapping_add(u64::from(t) + 1);
                if i >= min {
                    x = x.wrapping_sub((u64::from(prompt[i - min]) + 1).wrapping_mul(pow));
                }
                if i + 1 >= min {
                    set.insert(x);
                }
            }
            set
        });
        set.contains(&h)
    }

    /// One more thinking token of the reply; `prompt` is the request's prompt.
    pub fn push(&mut self, t: u32, prompt: &[u32]) -> Option<Hit> {
        self.toks.push(t);
        let n = self.toks.len();
        let i = n - 1;
        self.hash = self.hash.wrapping_mul(B).wrapping_add(u64::from(t) + 1);
        if n > self.min {
            self.hash = self.hash.wrapping_sub((u64::from(self.toks[i - self.min]) + 1).wrapping_mul(self.pow));
        }
        // a repeat goes on while each new token equals the one `period` back
        if self.streak > 0 && self.toks[i] == self.toks[i - self.period] {
            self.streak += 1;
        } else {
            self.streak = 0;
        }
        if n < self.min {
            return None;
        }
        let (min, h) = (self.min, self.hash);
        let mut copies = 0;
        match self.windows.get_mut(&h) {
            None => {
                self.windows.insert(h, Entry { seen: i as u32, counted: i as u32, count: 1 });
            }
            Some(e) => {
                let prev = e.seen as usize;
                if self.toks[i + 1 - min..=i] != self.toks[prev + 1 - min..=prev] {
                    return None; // another window with the same hash
                }
                if self.streak == 0 {
                    self.period = i - prev;
                    self.streak = 1;
                }
                e.seen = i as u32;
                if i - e.counted as usize >= min {
                    e.count += 1;
                    e.counted = i as u32;
                    copies = e.count;
                }
            }
        }
        let run = if self.streak > 0 { min - 1 + self.streak } else { 0 };
        let fire = copies >= self.copies || (self.long > 0 && run >= self.long);
        if !fire {
            return None;
        }
        if self.prompt_has(prompt, h) {
            return None;
        }
        let copies = self.windows.get(&h).map_or(0, |e| e.count);
        Some(Hit { at: i, run, period: self.period, copies })
    }
}

/// What the guard wants put into the reply after a trigger.
#[derive(Debug, PartialEq)]
pub struct Action {
    pub tokens: Vec<u32>,
    /// the tokens end with the think block's close
    pub closes: bool,
}

/// One request's guard: the detector, the triggers so far, and the tokens it puts in.
pub struct LoopGuard {
    pub spec: Spec,
    det: Option<Detector>,
    phrase: Vec<u32>,
    close: Vec<u32>,
    pub triggers: u32,
    /// the thinking token after the last phrase put in
    phrase_end: Option<usize>,
    /// the last logged trigger (log mode)
    last_log: Option<usize>,
    /// feeding the guard's own tokens: no trigger
    quiet: bool,
}

impl LoopGuard {
    pub fn new(spec: Spec, phrase: Vec<u32>, close: Vec<u32>) -> LoopGuard {
        let det = Some(Detector::new(&spec));
        LoopGuard { spec, det, phrase, close, triggers: 0, phrase_end: None, last_log: None, quiet: false }
    }

    /// Feed one thinking token. Some((hit, action)) when it fires: in log mode the action is None.
    pub fn push(&mut self, t: u32, prompt: &[u32]) -> Option<(Hit, Option<Action>)> {
        let det = self.det.as_mut()?;
        let hit = det.push(t, prompt)?;
        if self.quiet {
            return None;
        }
        if self.spec.mode == Mode::Log {
            if self.last_log.is_some_and(|l| hit.at - l < self.spec.window as usize) {
                return None;
            }
            self.last_log = Some(hit.at);
        }
        self.triggers += 1;
        if self.triggers >= self.spec.max {
            self.det = None; // nothing more to do: free the windows
        }
        let close = match self.spec.mode {
            Mode::Log => return Some((hit, None)),
            Mode::Phrase => false,
            Mode::Close => true,
            Mode::PhraseClose => self.phrase_end.is_some_and(|e| hit.at + 1 - e <= self.spec.window as usize),
        };
        let mut tokens = self.phrase.clone();
        if close {
            tokens.extend_from_slice(&self.close);
        }
        Some((hit, Some(Action { tokens, closes: close })))
    }

    /// The guard's own tokens went into the reply: the detector sees them, and they fire nothing.
    pub fn inserted(&mut self, tokens: &[u32], prompt: &[u32]) {
        self.quiet = true;
        for &t in tokens {
            self.push(t, prompt);
        }
        self.quiet = false;
        self.phrase_end = self.det.as_ref().map(Detector::len);
    }

    /// The repeated-tool-call guard cut the reply back to `kept` (all thinking): read it again from the start.
    pub fn rewind(&mut self, kept: &[u32], prompt: &[u32]) {
        if self.det.is_none() {
            return;
        }
        self.det = Some(Detector::new(&self.spec));
        self.quiet = true;
        for &t in kept {
            self.push(t, prompt);
        }
        self.quiet = false;
        self.phrase_end = self.phrase_end.filter(|&e| e <= kept.len());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(s: &str) -> Spec {
        Spec::parse(s).unwrap()
    }

    /// Feed `toks` through a detector; the hits.
    fn hits(s: &Spec, prompt: &[u32], toks: &[u32]) -> Vec<Hit> {
        let mut d = Detector::new(s);
        toks.iter().filter_map(|&t| d.push(t, prompt)).collect()
    }

    /// `n` tokens of healthy-looking reasoning: no window repeats (a counter from `from`).
    fn fresh(from: u32, n: u32) -> Vec<u32> {
        (from..from + n).collect()
    }

    #[test]
    fn a_spec_names_its_parts_and_text_takes_the_rest() {
        assert_eq!(spec(""), Spec::default());
        let s = spec("mode=phrase-close,min=32,copies=4,max=3,window=256,long=128");
        assert_eq!((s.mode, s.min, s.copies, s.max, s.window, s.long), (Mode::PhraseClose, 32, 4, 3, 256, 128));
        assert_eq!(s.text, DEFAULT_TEXT);
        let s = spec("mode=close,text=Stop, decide.\\n\\nNow.\\t\\\\");
        assert_eq!((s.mode, s.text.as_str()), (Mode::Close, "Stop, decide.\n\nNow.\t\\"));
        assert_eq!(spec("text=a=b,c=d").text, "a=b,c=d");
        assert_eq!(spec("mode=log").mode, Mode::Log);
        for (bad, why) in [
            ("mode=loud", "want log, phrase"),
            ("min=2", "want a whole number from 4"),
            ("copies=1", "from 2"),
            ("max=0", "from 1"),
            ("min=64,min=32", "given twice"),
            ("speed=3", "want mode, min"),
            ("min", "want key=value"),
            ("text=", "1 to 2000"),
            ("text=a\\x", "unknown escape"),
            ("text=a\\", "lone"),
            ("long=10", "at least min"),
            ("mode=phrasetext=x", "want log, phrase"),
        ] {
            let e = Spec::parse(bad).expect_err(bad);
            assert!(e.contains(why), "{bad}: {e}");
        }
    }

    #[test]
    fn a_loop_fires_at_its_third_copy() {
        let s = spec("min=8,copies=3");
        let unit = fresh(1000, 20); // a 20-token loop body
        let mut toks = fresh(0, 50);
        for _ in 0..5 {
            toks.extend(&unit);
        }
        let h = hits(&s, &[], &toks);
        // the first window of the body (ending at its 8th token) occurs a third time in the third copy
        let third = 50 + 2 * 20 + 7;
        assert_eq!(h[0], Hit { at: third, run: 20 + 8, period: 20, copies: 3 });
        // and keeps firing while the loop runs: the caller decides what to do with later hits
        assert!(h.len() > 1 && h.iter().all(|x| x.at >= third));
    }

    #[test]
    fn a_short_period_loop_counts_only_occurrences_that_do_not_overlap() {
        let s = spec("min=8,copies=3");
        let mut toks = fresh(0, 30);
        toks.extend([7u32; 40]); // "aaaa...": every window equal, overlapping
        let h = hits(&s, &[], &toks);
        // counted at its 1st, 9th and 17th occurrence: tokens 37, 45, 53
        assert_eq!(h[0].at, 30 + 7 + 16);
        assert_eq!((h[0].copies, h[0].period), (3, 1));
    }

    #[test]
    fn two_copies_and_near_copies_do_not_fire() {
        let s = spec("min=8,copies=3");
        let unit = fresh(1000, 30);
        // two copies of a drafted function with healthy text between: no third copy
        let mut toks = unit.clone();
        toks.extend(fresh(0, 100));
        toks.extend(&unit);
        toks.extend(fresh(200, 100));
        assert!(hits(&s, &[], &toks).is_empty());
        // a near-loop: the same 30-token frame five times, a different token every 6 (a counter, a line number)
        let mut toks = Vec::new();
        for k in 0..5u32 {
            for (j, &t) in unit.iter().enumerate() {
                toks.push(if j % 6 == 5 { 5000 + k * 10 + j as u32 } else { t });
            }
        }
        assert!(hits(&s, &[], &toks).is_empty());
        // the same near-loop with a window shorter than the gap between the changes fires
        assert!(!hits(&spec("min=4,copies=3"), &[], &toks).is_empty());
    }

    #[test]
    fn quoting_the_prompt_never_fires() {
        let s = spec("min=8,copies=3");
        let file = fresh(3000, 40); // code from a tool result
        let mut prompt = fresh(0, 500);
        prompt.extend(&file);
        prompt.extend(fresh(600, 100));
        // the reply quotes the file four times, with its own reasoning between
        let mut toks = Vec::new();
        for k in 0..4 {
            toks.extend(&file);
            toks.extend(fresh(10_000 + 100 * k, 20));
        }
        assert!(hits(&s, &prompt, &toks).is_empty());
        // its own words looped do fire, though the same prompt is there
        let mut toks = fresh(20_000, 10);
        let own = fresh(30_000, 15);
        for _ in 0..4 {
            toks.extend(&own);
        }
        assert!(!hits(&s, &prompt, &toks).is_empty());
        // and the prompt's windows are built only once a window came that close
        let mut d = Detector::new(&s);
        for t in fresh(40_000, 200) {
            d.push(t, &prompt);
        }
        assert!(d.prompt.is_none());
    }

    #[test]
    fn a_long_repeat_fires_at_its_length_when_on() {
        let unit = fresh(1000, 100);
        let mut toks = unit.clone();
        toks.extend(&unit);
        assert!(hits(&spec("min=8,copies=3"), &[], &toks).is_empty());
        let h = hits(&spec("min=8,copies=3,long=40"), &[], &toks);
        // the repeat reaches 40 tokens at the second copy's 40th token
        assert_eq!(h[0], Hit { at: 100 + 39, run: 40, period: 100, copies: 2 });
    }

    #[test]
    fn modes_put_in_the_phrase_the_close_or_nothing() {
        let unit = fresh(1000, 20);
        let mut loop_toks = fresh(0, 30);
        for _ in 0..12 {
            loop_toks.extend(&unit);
        }
        let (phrase, close) = (vec![90u32, 91, 92], vec![4u32, 11, 5]);
        // drive a guard as gen.rs does: on an action the action's tokens go in and the reply goes on (here: more loop)
        let run = |s: &str| -> Vec<(Hit, Option<Action>)> {
            let mut g = LoopGuard::new(spec(s), phrase.clone(), close.clone());
            let mut out = Vec::new();
            for &t in &loop_toks {
                if let Some((h, a)) = g.push(t, &[]) {
                    if let Some(a) = &a {
                        g.inserted(&a.tokens, &[]);
                    }
                    let closes = a.as_ref().is_some_and(|a| a.closes);
                    out.push((h, a));
                    if closes {
                        break;
                    }
                }
            }
            out
        };
        let p = run("mode=phrase,min=8,max=2");
        assert_eq!(p.len(), 2);
        assert!(p.iter().all(|(_, a)| a.as_ref() == Some(&Action { tokens: phrase.clone(), closes: false })));
        let both: Vec<u32> = phrase.iter().chain(&close).copied().collect();
        let c = run("mode=close,min=8");
        assert_eq!((c.len(), c[0].1.as_ref()), (1, Some(&Action { tokens: both.clone(), closes: true })));
        // phrase-close: the loop comes back within the window, so the second trigger closes
        let pc = run("mode=phrase-close,min=8");
        assert_eq!(pc.len(), 2);
        assert_eq!(pc[0].1.as_ref().map(|a| a.closes), Some(false));
        assert_eq!(pc[1].1.as_ref(), Some(&Action { tokens: both, closes: true }));
        // ... and with a window too short to reach the second trigger, it is a second phrase
        let pc = run("mode=phrase-close,min=8,window=4");
        assert_eq!(pc.iter().map(|(_, a)| a.as_ref().unwrap().closes).collect::<Vec<_>>(), [false, false]);
        // log: no tokens, triggers spaced by the window, at most max
        let l = run("mode=log,min=8,max=5,window=50");
        assert!(l.iter().all(|(_, a)| a.is_none()));
        assert!(l.windows(2).all(|w| w[1].0.at - w[0].0.at >= 50));
        assert!(l.len() >= 2 && l.len() <= 5);
    }

    #[test]
    fn after_max_triggers_the_detector_is_dropped() {
        let mut g = LoopGuard::new(spec("min=8,max=1"), vec![90], vec![4]);
        let unit = fresh(1000, 10);
        let mut n = 0;
        for _ in 0..10 {
            for &t in &unit {
                n += g.push(t, &[]).is_some() as u32;
            }
        }
        assert_eq!((n, g.triggers), (1, 1));
        assert!(g.det.is_none());
    }
}
