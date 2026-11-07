//! The S2 request file's parser (was parse_request and its helpers in engine/runtime/s2/sessions.h). A request is a
//! header line, `n_gen <n> temp <f> seed <n> min <n> stop <ids|->` and then, in this order and each optional,
//! `top_p <f>`, `pen <presence> <frequency> <thinking presence> <thinking frequency> <think_end or -1>`,
//! `boost <start> <per1k> <max>`, `seqt <mult>:<base>:<min>:<cap>`, `seqa <mult>:<base>:<min>:<cap>`,
//! `seqg <mult>:<base>:<min>:<cap>:<ids>`, `lp <k>`, `plp <k>`, `sys <n>`, `ins <rule>`, `gram <grammar>` and `steer <profile> <hex>`, then any
//! number of `img <field>` (DS41 images, ds41_image.h), then a line of comma-separated prompt ids. The header is read the way libstdc++'s istringstream read it, so every request
//! the C++ took, and every refusal it gave, is the same here: a number ends at the first character it cannot take, and
//! the rest of that word is the next word.

use crate::scheduler::Identity;

pub const LP_MAX: i32 = 20;
pub const INSERT_MAX: usize = 64;

/// A refusal: one of the C++ messages, or one the steering callback returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    Msg(&'static str),
    Foreign(*const std::ffi::c_char),
}
use Refusal::Msg;
type R<T> = Result<T, Refusal>;

fn require(ok: bool, m: &'static str) -> R<()> {
    if ok {
        Ok(())
    } else {
        Err(Msg(m))
    }
}

/// ingress.h number: decimal digits only, no sign, refusing overflow.
pub fn number(s: &[u8]) -> R<u64> {
    require(!s.is_empty(), "empty number")?;
    let mut n: u64 = 0;
    for &c in s {
        require(c.is_ascii_digit() && n <= (u64::MAX - u64::from(c - b'0')) / 10, "bad/overflow number")?;
        n = n * 10 + u64::from(c - b'0');
    }
    Ok(n)
}

fn find(s: &[u8], c: u8, from: usize) -> Option<usize> {
    s.get(from..)?.iter().position(|&x| x == c).map(|i| i + from)
}

/// Comma-separated token ids.
pub fn request_ids(s: &[u8]) -> R<Vec<i32>> {
    let mut out = Vec::new();
    let mut a = 0;
    loop {
        let b = find(s, b',', a);
        let n = number(&s[a..b.unwrap_or(s.len())])?;
        require(n <= i32::MAX as u64, "request token integer overflow")?;
        out.push(n as i32);
        match b {
            Some(b) => a = b + 1,
            None => return Ok(out),
        }
    }
}

/// One scope of the sequence penalty (engine/runtime/penalty.h pen_seq: thinking, answer, guard):
/// `<mult>:<base>:<min>:<cap>`, the guard's followed by `:<ids>`, the call it must not repeat.
#[derive(Clone, Debug, PartialEq)]
pub struct Seq {
    pub mult: f32,
    pub base: f32,
    pub min: i32,
    pub cap: f32,
    pub avoid: Vec<i32>,
}

impl Default for Seq {
    fn default() -> Seq {
        Seq { mult: 0.0, base: 1.0, min: 1, cap: 0.0, avoid: vec![] }
    }
}

/// `seqt` / `seqa` / `seqg` fields: mult 0 or more, base 1 or more, min 1 or more, cap 0 to 20; the guard's ids 1 to 4096
pub fn parse_seq(s: &[u8], guard: bool) -> R<Seq> {
    let parts: Vec<&[u8]> = s.splitn(if guard { 5 } else { 4 }, |&c| c == b':').collect();
    require(parts.len() == if guard { 5 } else { 4 }, "invalid seq header")?;
    let f = |t: &[u8]| -> R<f32> {
        let v = strtof_all(t).ok_or(Msg("invalid seq header"))?;
        require(v.is_finite(), "invalid seq header")?;
        Ok(v)
    };
    let (mult, base, cap) = (f(parts[0])?, f(parts[1])?, f(parts[3])?);
    let min = number(parts[2])?;
    require(mult >= 0.0 && base >= 1.0 && (1..=i32::MAX as u64).contains(&min) && (0.0..=20.0).contains(&cap), "seq: want mult >= 0, base >= 1, min >= 1 and cap in [0, 20]")?;
    let avoid = if guard { request_ids(parts[4])? } else { vec![] };
    require(!guard || (1..=4096).contains(&avoid.len()), "seqg: 1 to 4096 ids")?;
    Ok(Seq { mult, base, min: min as i32, cap, avoid })
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Insert {
    pub after: i32,
    pub unless: i32,
    pub tokens: Vec<i32>,
}

/// `<after>:<unless or -1>:<id,id,..>` (insert_rule.h)
pub fn parse_insert(s: &[u8]) -> R<Insert> {
    let a = find(s, b':', 0);
    let b = a.and_then(|a| find(s, b':', a + 1));
    let (Some(a), Some(b)) = (a, b) else { return Err(Msg("invalid ins header")) };
    let after = number(&s[..a])?;
    require(after <= i32::MAX as u64, "invalid ins after")?;
    let unless = &s[a + 1..b];
    let unless = if unless == b"-1" { -1 } else { number(unless)?.min(i32::MAX as u64) as i32 };
    let tokens = request_ids(&s[b + 1..])?;
    require(!tokens.is_empty() && tokens.len() <= INSERT_MAX, "invalid ins token count")?;
    Ok(Insert { after: after as i32, unless, tokens })
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Grammar {
    pub kind: u32, // grammar.h GrammarSpec::Kind: 0 off, 1 JSON, 2 schema
    pub strict: bool,
    pub arm: i32,
    pub schema: Vec<u8>,
}

/// `<kind>:<strict>:<arm or -1>:<hex>` (grammar.h): kind 1 = JSON, 2 = the JSON schema whose UTF-8 bytes are <hex>
pub fn parse_grammar(s: &[u8]) -> R<Grammar> {
    let mut c = [0usize; 3];
    let mut at = 0;
    for x in c.iter_mut() {
        *x = find(s, b':', at).ok_or(Msg("invalid gram header"))?;
        at = *x + 1;
    }
    let (kind, strict, arm) = (&s[..c[0]], &s[c[0] + 1..c[1]], &s[c[1] + 1..c[2]]);
    require((kind == b"1" || kind == b"2") && (strict == b"0" || strict == b"1"), "invalid gram kind/strict")?;
    let kind = if kind == b"1" { 1 } else { 2 };
    let arm = if arm == b"-1" { -1 } else { number(arm)?.min(i32::MAX as u64) as i32 };
    let hex = &s[c[2] + 1..];
    require(hex.len().is_multiple_of(2) && (kind == 1) == hex.is_empty(), "invalid gram schema")?;
    let nib = |h: u8| -> R<u8> {
        require(h.is_ascii_hexdigit(), "invalid gram hex")?;
        Ok(if h <= b'9' { h - b'0' } else { (h | 0x20) - b'a' + 10 })
    };
    let schema = hex.chunks(2).map(|p| Ok(nib(p[0])? << 4 | nib(p[1])?)).collect::<R<Vec<u8>>>()?;
    Ok(Grammar { kind, strict: strict == b"1", arm, schema })
}

/// steering.h valid_profile_name
pub fn valid_profile_name(s: &[u8]) -> bool {
    let word = |c: u8| c.is_ascii_alphanumeric() || c == b'.' || c == b'_' || c == b'-';
    !s.is_empty() && s.len() <= 128 && s[0].is_ascii_alphanumeric() && s.iter().all(|&c| word(c))
}

/// steering.h parse_scale_bits: eight lowercase hex digits of a float in [0, 4] (0 for either zero).
pub fn parse_scale_bits(s: &[u8]) -> R<u32> {
    const BAD: &str = "steering scale bits must be eight lowercase hex digits";
    require(s.len() == 8, BAD)?;
    let mut bits: u32 = 0;
    for &c in s {
        require(c.is_ascii_digit() || (b'a'..=b'f').contains(&c), BAD)?;
        bits = bits.wrapping_mul(16).wrapping_add(u32::from(if c <= b'9' { c - b'0' } else { c - b'a' + 10 }));
    }
    let scale = f32::from_bits(bits);
    if scale == 0.0 {
        return Ok(0);
    }
    require(scale.is_finite() && (0.0..=4.0).contains(&scale), "steering scale bits are outside [0, 4]")?;
    Ok(bits)
}

/// An istringstream over one line: words split on C-locale whitespace, numbers read the libstdc++ way, and the
/// stream failed for good after one failed read.
struct Stream<'a> {
    s: &'a [u8],
    at: usize,
    failed: bool,
}

fn space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

impl<'a> Stream<'a> {
    fn new(s: &'a [u8]) -> Self {
        Stream { s, at: 0, failed: false }
    }
    fn skip(&mut self) -> bool {
        while self.at < self.s.len() && space(self.s[self.at]) {
            self.at += 1;
        }
        if self.failed || self.at == self.s.len() {
            self.failed = true;
        }
        !self.failed
    }
    fn word(&mut self) -> Option<&'a [u8]> {
        if !self.skip() {
            return None;
        }
        let a = self.at;
        while self.at < self.s.len() && !space(self.s[self.at]) {
            self.at += 1;
        }
        Some(&self.s[a..self.at])
    }
    fn peek(&self) -> Option<u8> {
        self.s.get(self.at).copied()
    }
    /// num_get for float: [+-] digits [. digits] [e [+-] digits], an exponent only after a mantissa digit, then
    /// strtof on what it took; no conversion, trailing text or overflow fails.
    fn float(&mut self) -> Option<f32> {
        if !self.skip() {
            return None;
        }
        let a = self.at;
        if matches!(self.peek(), Some(b'+' | b'-')) {
            self.at += 1;
        }
        let (mut mantissa, mut dec, mut sci) = (false, false, false);
        while let Some(c) = self.peek() {
            if c == b'.' {
                if dec || sci {
                    break;
                }
                dec = true;
            } else if c.is_ascii_digit() {
                mantissa = true;
            } else if (c == b'e' || c == b'E') && !sci && mantissa {
                sci = true;
                if matches!(self.s.get(self.at + 1), Some(b'+' | b'-')) {
                    self.at += 1;
                }
            } else {
                break;
            }
            self.at += 1;
        }
        match strtof_all(&self.s[a..self.at]) {
            Some(v) if v.is_finite() => Some(v),
            _ => {
                self.failed = true;
                None
            }
        }
    }
    /// num_get for a signed integer in base 10: [+-] digits; none, or out of range, fails.
    fn int(&mut self, lo: i64, hi: i64) -> Option<i64> {
        if !self.skip() {
            return None;
        }
        let neg = match self.peek() {
            Some(b'-') => {
                self.at += 1;
                true
            }
            Some(b'+') => {
                self.at += 1;
                false
            }
            _ => false,
        };
        let (mut v, mut any) = (0i128, false);
        while let Some(c) = self.peek().filter(u8::is_ascii_digit) {
            any = true;
            // capped past any range a caller asks for, so it cannot overflow
            v = (v * 10 + i128::from(c - b'0')).min(i128::from(u64::MAX) + 1);
            self.at += 1;
        }
        let v = if neg { -v } else { v };
        if !any || v < i128::from(lo) || v > i128::from(hi) {
            self.failed = true;
            return None;
        }
        Some(v as i64)
    }
}

extern "C" {
    fn strtof(s: *const std::ffi::c_char, end: *mut *mut std::ffi::c_char) -> f32;
}

/// C strtof over all of `t`, as libstdc++ converts what num_get took (the engine stays in the C locale): None when it
/// converts nothing or leaves text over. Rust's own float parser rounds differently on very long exponents.
fn strtof_all(t: &[u8]) -> Option<f32> {
    let mut z = Vec::with_capacity(t.len() + 1);
    z.extend_from_slice(t);
    z.push(0);
    let mut end: *mut std::ffi::c_char = std::ptr::null_mut();
    // SAFETY: z is NUL-terminated and outlives the call
    let v = unsafe { strtof(z.as_ptr().cast(), &mut end) };
    let used = end as usize - z.as_ptr() as usize;
    (used != 0 && used == t.len()).then_some(v)
}
/// The parsed request (serve.h ServeReq, but steering already resolved by the caller's callback).
#[derive(Clone, Debug, PartialEq)]
pub struct Request {
    pub prompt: Vec<i32>,
    pub stop: Vec<i32>,
    pub n_gen: i32,
    pub min_tokens: i32,
    pub temp: f32,
    pub top_p: f32,
    /// presence / frequency penalties (engine/runtime/penalty.h), each in [-2, 2]; 0 = off
    pub presence: f32,
    pub frequency: f32,
    /// the thinking penalties, inside the think block `think_end` closes (-1: the prompt left none open)
    pub tpresence: f32,
    pub tfrequency: f32,
    pub think_end: i32,
    /// the think-end boost (engine/runtime/penalty.h pen_boost): per1k 0 = off
    pub boost_start: i32,
    pub boost_per1k: f32,
    pub boost_max: f32,
    /// the sequence penalty's thinking, answer and guard scopes
    pub seq: [Seq; 3],
    pub seed: u32,
    pub lp: i32,
    pub plp: i32,
    pub sys: i64,
    pub steering: Identity,
    pub ins: Insert,
    pub gram: Grammar,
}

/// The request's steering fields, for the caller to resolve against the profiles it loaded (steering.h BootConfig).
pub struct SteerFields<'a> {
    pub profile: Option<&'a [u8]>,
    pub scale_bits: u32,
}

/// `grammar_ready`: the engine was started with --grammar-dir. `resolve` turns the steering fields into the request's
/// identity, or refuses. `image` takes each `img` field's value in order, or refuses it: the caller keeps the images
/// (ds41_image.h parse_field) and checks them against the prompt after the parse (img::check, img::digest).
pub fn parse(
    bytes: &[u8],
    grammar_ready: bool,
    resolve: &mut dyn FnMut(&SteerFields) -> R<Identity>,
    image: &mut dyn FnMut(&[u8]) -> R<()>,
) -> R<Request> {
    let line = find(bytes, b'\n', 0).ok_or(Msg("request missing prompt line"))?;
    let mut inp = Stream::new(&bytes[..line]);
    let mut temp = 0f32;
    let (mut gen, mut seed, mut min, mut stop): (&[u8], &[u8], &[u8], &[u8]) = (b"", b"", b"", b"");
    let kw = |inp: &mut Stream, want: &[u8]| inp.word() == Some(want);
    let ok = kw(&mut inp, b"n_gen")
        && inp.word().map(|w| gen = w).is_some()
        && kw(&mut inp, b"temp")
        && inp.float().map(|v| temp = v).is_some()
        && kw(&mut inp, b"seed")
        && inp.word().map(|w| seed = w).is_some()
        && kw(&mut inp, b"min")
        && inp.word().map(|w| min = w).is_some()
        && kw(&mut inp, b"stop")
        && inp.word().map(|w| stop = w).is_some();
    require(ok, "invalid request header")?;
    let mut k: &[u8] = inp.word().unwrap_or(b"");
    let mut r = Request {
        prompt: vec![],
        stop: vec![],
        n_gen: 0,
        min_tokens: 0,
        temp,
        top_p: 0.0,
        presence: 0.0,
        frequency: 0.0,
        tpresence: 0.0,
        tfrequency: 0.0,
        think_end: -1,
        boost_start: 0,
        boost_per1k: 0.0,
        boost_max: 0.0,
        seq: [Seq::default(), Seq::default(), Seq::default()],
        seed: 1234,
        lp: -1,
        plp: -1,
        sys: -1,
        steering: Identity {
            enabled: false,
            profile_digest: [0; 32],
            scope: 2,
            schedule_digest: [0; 32],
            scale_bits: 0,
            slot: 0,
            images: false,
            image_digest: [0; 32],
        },
        ins: Insert { after: -1, unless: -1, tokens: vec![] },
        gram: Grammar { kind: 0, strict: false, arm: -1, schema: vec![] },
    };
    macro_rules! next {
        () => {
            k = inp.word().unwrap_or(b"")
        };
    }
    if k == b"top_p" {
        r.top_p = inp.float().ok_or(Msg("invalid top_p header"))?;
        next!();
    }
    if k == b"pen" {
        r.presence = inp.float().ok_or(Msg("invalid pen header"))?;
        r.frequency = inp.float().ok_or(Msg("invalid pen header"))?;
        r.tpresence = inp.float().ok_or(Msg("invalid pen header"))?;
        r.tfrequency = inp.float().ok_or(Msg("invalid pen header"))?;
        r.think_end = inp.int(-1, i32::MAX.into()).ok_or(Msg("invalid pen header"))? as i32;
        next!();
        require(
            [r.presence, r.frequency, r.tpresence, r.tfrequency].iter().all(|v| (-2.0..=2.0).contains(v)),
            "pen: presence and frequency penalties must be in [-2, 2]",
        )?;
    }
    if k == b"boost" {
        r.boost_start = inp.int(0, i32::MAX.into()).ok_or(Msg("invalid boost header"))? as i32;
        r.boost_per1k = inp.float().ok_or(Msg("invalid boost header"))?;
        r.boost_max = inp.float().ok_or(Msg("invalid boost header"))?;
        next!();
        require(r.boost_per1k >= 0.0 && (0.0..=20.0).contains(&r.boost_max), "boost: per1k must be 0 or more and max in [0, 20]")?;
    }
    if k == b"seqt" {
        r.seq[0] = parse_seq(inp.word().ok_or(Msg("invalid seq header"))?, false)?;
        next!();
    }
    if k == b"seqa" {
        r.seq[1] = parse_seq(inp.word().ok_or(Msg("invalid seq header"))?, false)?;
        next!();
    }
    if k == b"seqg" {
        r.seq[2] = parse_seq(inp.word().ok_or(Msg("invalid seq header"))?, true)?;
        next!();
    }
    if k == b"lp" {
        r.lp = inp
            .int(i32::MIN.into(), i32::MAX.into())
            .filter(|v| (0..=LP_MAX.into()).contains(v))
            .ok_or(Msg("invalid lp header"))? as i32;
        next!();
    }
    if k == b"plp" {
        r.plp = inp
            .int(i32::MIN.into(), i32::MAX.into())
            .filter(|v| (0..=LP_MAX.into()).contains(v))
            .ok_or(Msg("invalid plp header"))? as i32;
        next!();
    }
    if k == b"sys" {
        r.sys = inp
            .int(i64::MIN, i64::MAX)
            .filter(|v| (0..=i32::MAX.into()).contains(v))
            .ok_or(Msg("invalid sys header"))?;
        next!();
    }
    if k == b"ins" {
        let v = inp.word().ok_or(Msg("invalid ins header"))?;
        r.ins = parse_insert(v)?;
        next!();
    }
    if k == b"gram" {
        let v = inp.word().ok_or(Msg("invalid gram header"))?;
        r.gram = parse_grammar(v)?;
        next!();
        require(grammar_ready, "gram: the engine was started without --grammar-dir")?;
    }
    let mut sf = SteerFields { profile: None, scale_bits: 0 };
    if k == b"steer" {
        let (name, bits) = match (inp.word(), inp.word()) {
            (Some(n), Some(b)) => (n, b),
            _ => return Err(Msg("invalid steering header")),
        };
        next!();
        require(valid_profile_name(name), "invalid steering profile name")?;
        sf = SteerFields { profile: Some(name), scale_bits: parse_scale_bits(bits)? };
    }
    while k == b"img" {
        let v = inp.word().ok_or(Msg("invalid img header"))?;
        image(v)?;
        next!();
    }
    require(k.is_empty(), "invalid request header tail (S2 accepts top_p, pen, boost, seqt, seqa, seqg, lp, plp, sys, ins, gram and steer)")?;
    require(r.ins.after < 0 || r.lp < 0, "ins with lp")?;
    r.steering = resolve(&sf)?;
    let g = number(gen)?;
    let sd = number(seed)?;
    let mn = number(min)?;
    require(
        g > 0
            && g <= i32::MAX as u64
            && sd <= u64::from(u32::MAX)
            && mn <= g
            && r.temp.is_finite()
            && r.temp >= 0.0
            && r.top_p.is_finite()
            && r.top_p >= 0.0
            && r.top_p <= 1.0,
        "invalid request sampling/length",
    )?;
    r.n_gen = g as i32;
    r.seed = sd as u32;
    r.min_tokens = mn as i32;
    if stop != b"-" {
        r.stop = request_ids(stop)?;
    }
    let mut ids = Stream::new(&bytes[line + 1..]);
    let prompt = ids.word();
    let extra = ids.word();
    let (Some(prompt), None) = (prompt, extra) else { return Err(Msg("invalid prompt line")) };
    r.prompt = request_ids(prompt)?;
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn off(_: &SteerFields) -> R<Identity> {
        Ok(Identity {
            enabled: false,
            profile_digest: [0; 32],
            scope: 2,
            schedule_digest: [0; 32],
            scale_bits: 0,
            slot: 0,
            images: false,
            image_digest: [0; 32],
        })
    }
    fn no_img(_: &[u8]) -> R<()> {
        Err(Msg("no img here"))
    }

    #[test]
    fn a_full_request() {
        let r = parse(
            b"n_gen 9 temp 0.5 seed 7 min 1 stop 3,4 top_p 0.9 plp 2 sys 5 ins 3:-1:8,9\n1,2,3\n",
            false,
            &mut off,
            &mut no_img,
        )
        .unwrap();
        assert_eq!((r.n_gen, r.seed, r.min_tokens, r.temp, r.top_p, r.plp, r.sys), (9, 7, 1, 0.5, 0.9, 2, 5));
        assert_eq!((r.stop, r.prompt, r.ins.tokens), (vec![3, 4], vec![1, 2, 3], vec![8, 9]));
        assert_eq!(
            parse(b"n_gen 9 temp 0.5 seed 7 min 1 stop - lp 2 ins 3:-1:8\n1\n", false, &mut off, &mut no_img).err(),
            Some(Msg("ins with lp"))
        );
    }

    #[test]
    fn numbers_end_where_libstdcxx_stops() {
        // "0.5seed" reads 0.5, then "seed" is the next word
        assert!(parse(b"n_gen 9 temp 0.5seed 7 min 1 stop -\n1\n", false, &mut off, &mut no_img).is_ok());
        assert_eq!(
            parse(b"n_gen 9 temp 1e seed 7 min 1 stop -\n1\n", false, &mut off, &mut no_img).err(),
            Some(Msg("invalid request header"))
        );
        assert_eq!(
            parse(b"n_gen 9 temp 1e99 seed 7 min 1 stop -\n1\n", false, &mut off, &mut no_img).err(),
            Some(Msg("invalid request header"))
        );
        assert_eq!(
            parse(b"n_gen 9 temp 0 seed 7 min 1 stop - lp 21\n1\n", false, &mut off, &mut no_img).err(),
            Some(Msg("invalid lp header"))
        );
        assert_eq!(
            parse(b"n_gen 9 temp 0 seed 7 min 1 stop - lp 2x\n1\n", false, &mut off, &mut no_img).err(),
            Some(Msg("invalid request header tail (S2 accepts top_p, pen, boost, seqt, seqa, seqg, lp, plp, sys, ins, gram and steer)"))
        );
    }

    #[test]
    fn penalties() {
        let r = parse(b"n_gen 9 temp 0 seed 7 min 1 stop - top_p 0.9 pen 1.5 -0.25 0 0 -1 lp 2\n1\n", false, &mut off, &mut no_img).unwrap();
        assert_eq!((r.top_p, r.presence, r.frequency, r.think_end, r.lp), (0.9, 1.5, -0.25, -1, 2));
        // without top_p, and absent: off
        let r = parse(b"n_gen 9 temp 0 seed 7 min 1 stop - pen 0 2 1.5 -1 154842\n1\n", false, &mut off, &mut no_img).unwrap();
        assert_eq!((r.frequency, r.tpresence, r.tfrequency, r.think_end), (2.0, 1.5, -1.0, 154842));
        let r = parse(b"n_gen 9 temp 0 seed 7 min 1 stop - lp 2\n1\n", false, &mut off, &mut no_img).unwrap();
        assert_eq!((r.presence, r.frequency, r.tpresence, r.tfrequency, r.think_end), (0.0, 0.0, 0.0, 0.0, -1));
        assert_eq!((r.boost_start, r.boost_per1k, r.boost_max), (0, 0.0, 0.0));
        // the boost, after pen or alone
        let r = parse(b"n_gen 9 temp 0 seed 7 min 1 stop - pen 0 0 0 0 50 boost 4096 2 10 lp 1\n1\n", false, &mut off, &mut no_img).unwrap();
        assert_eq!((r.think_end, r.boost_start, r.boost_per1k, r.boost_max, r.lp), (50, 4096, 2.0, 10.0, 1));
        assert_eq!(parse(b"n_gen 9 temp 0 seed 7 min 1 stop - boost 0 0.5 1.5\n1\n", false, &mut off, &mut no_img).unwrap().boost_max, 1.5);
        // the sequence penalty's scopes
        let r = parse(b"n_gen 9 temp 0 seed 7 min 1 stop - seqt 0.8:1.75:4:6 seqa 1:1.75:12:6 seqg 0.5:2:2:8:154822,3,4 lp 1\n1\n", false, &mut off, &mut no_img).unwrap();
        assert_eq!(r.seq[0], Seq { mult: 0.8, base: 1.75, min: 4, cap: 6.0, avoid: vec![] });
        assert_eq!(r.seq[1], Seq { mult: 1.0, base: 1.75, min: 12, cap: 6.0, avoid: vec![] });
        assert_eq!(r.seq[2], Seq { mult: 0.5, base: 2.0, min: 2, cap: 8.0, avoid: vec![154822, 3, 4] });
        assert_eq!(r.lp, 1);
        assert_eq!(parse(b"n_gen 9 temp 0 seed 7 min 1 stop - seqg 1:1:1:1:5\n1\n", false, &mut off, &mut no_img).unwrap().seq[2].avoid, vec![5]);
        assert_eq!(parse(b"n_gen 9 temp 0 seed 7 min 1 stop - seqa 1:1:1:1\n1\n", false, &mut off, &mut no_img).unwrap().seq[1].min, 1);
        for bad in [
            &b"n_gen 9 temp 0 seed 7 min 1 stop - seqt 0.8:1.75:4\n1\n"[..],
            b"n_gen 9 temp 0 seed 7 min 1 stop - seqt 0.8:0.5:4:6\n1\n",
            b"n_gen 9 temp 0 seed 7 min 1 stop - seqt 0.8:1.5:0:6\n1\n",
            b"n_gen 9 temp 0 seed 7 min 1 stop - seqt 0.8:1.5:4:21\n1\n",
            b"n_gen 9 temp 0 seed 7 min 1 stop - seqt -1:1.5:4:6\n1\n",
            b"n_gen 9 temp 0 seed 7 min 1 stop - seqg 0.8:1.5:4:6\n1\n",
            b"n_gen 9 temp 0 seed 7 min 1 stop - seqg 0.8:1.5:4:6:\n1\n",
            b"n_gen 9 temp 0 seed 7 min 1 stop - seqg 0.8:1.5:4:6:1 seqt 1:1:1:1\n1\n",
            b"n_gen 9 temp 0 seed 7 min 1 stop - seqa 0.8:1.5:4:6:1\n1\n",
        ] {
            assert!(parse(bad, false, &mut off, &mut no_img).is_err(), "{}", String::from_utf8_lossy(bad));
        }
        for (bad, why) in [
            (&b"n_gen 9 temp 0 seed 7 min 1 stop - boost 1 2\n1\n"[..], "invalid boost header"),
            (b"n_gen 9 temp 0 seed 7 min 1 stop - boost -1 2 10\n1\n", "invalid boost header"),
            (b"n_gen 9 temp 0 seed 7 min 1 stop - boost 1 -2 10\n1\n", "boost: per1k must be 0 or more and max in [0, 20]"),
            (b"n_gen 9 temp 0 seed 7 min 1 stop - boost 1 2 21\n1\n", "boost: per1k must be 0 or more and max in [0, 20]"),
            (b"n_gen 9 temp 0 seed 7 min 1 stop - lp 1 boost 1 2 10\n1\n", "invalid request header tail (S2 accepts top_p, pen, boost, seqt, seqa, seqg, lp, plp, sys, ins, gram and steer)"),
        ] {
            assert_eq!(parse(bad, false, &mut off, &mut no_img).err(), Some(Msg(why)), "{}", String::from_utf8_lossy(bad));
        }
        for (bad, why) in [
            (&b"n_gen 9 temp 0 seed 7 min 1 stop - pen 1.5 0 0 0\n1\n"[..], "invalid pen header"),
            (b"n_gen 9 temp 0 seed 7 min 1 stop - pen 1.5 x 0 0 -1\n1\n", "invalid pen header"),
            (b"n_gen 9 temp 0 seed 7 min 1 stop - pen 1.5 0 0 0 -2\n1\n", "invalid pen header"),
            (b"n_gen 9 temp 0 seed 7 min 1 stop - pen 2.5 0 0 0 -1\n1\n", "pen: presence and frequency penalties must be in [-2, 2]"),
            (b"n_gen 9 temp 0 seed 7 min 1 stop - pen 0 0 0 -2.5 -1\n1\n", "pen: presence and frequency penalties must be in [-2, 2]"),
            // after lp: the tail
            (b"n_gen 9 temp 0 seed 7 min 1 stop - lp 2 pen 1 1 0 0 -1\n1\n", "invalid request header tail (S2 accepts top_p, pen, boost, seqt, seqa, seqg, lp, plp, sys, ins, gram and steer)"),
        ] {
            assert_eq!(parse(bad, false, &mut off, &mut no_img).err(), Some(Msg(why)), "{}", String::from_utf8_lossy(bad));
        }
    }

    #[test]
    fn img_fields_go_to_the_caller_in_order() {
        let mut seen: Vec<Vec<u8>> = vec![];
        let mut keep = |v: &[u8]| -> R<()> {
            seen.push(v.to_vec());
            Ok(())
        };
        let r = parse(b"n_gen 9 temp 0 seed 7 min 1 stop - steer calm 3f800000 img a img b\n1\n", false, &mut off, &mut keep);
        assert!(r.is_ok());
        assert_eq!(seen, vec![b"a".to_vec(), b"b".to_vec()]);
        // the caller's refusal comes back where the C++ loop threw it: before the tail, ins/lp and steering checks
        assert_eq!(
            parse(b"n_gen 9 temp 0 seed 7 min 1 stop - lp 2 ins 3:-1:8 img a extra\n1\n", false, &mut off, &mut no_img).err(),
            Some(Msg("no img here"))
        );
        assert_eq!(
            parse(b"n_gen 9 temp 0 seed 7 min 1 stop - img\n1\n", false, &mut off, &mut no_img).err(),
            Some(Msg("invalid img header"))
        );
        // img comes last: a field after it is the tail
        let mut any = |_: &[u8]| -> R<()> { Ok(()) };
        assert_eq!(
            parse(b"n_gen 9 temp 0 seed 7 min 1 stop - img a top_p 0.5\n1\n", false, &mut off, &mut any).err(),
            Some(Msg("invalid request header tail (S2 accepts top_p, pen, boost, seqt, seqa, seqg, lp, plp, sys, ins, gram and steer)"))
        );
    }

    #[test]
    fn grammar() {
        assert_eq!(parse_grammar(b"2:1:5:7b7d").unwrap().schema, b"{}");
        assert_eq!(parse_grammar(b"1:0:-1:").unwrap().kind, 1);
        assert_eq!(parse_grammar(b"1:0:-1:7b").err(), Some(Msg("invalid gram schema")));
        assert_eq!(parse_grammar(b"2:0:-1:zz").err(), Some(Msg("invalid gram hex")));
    }
}
