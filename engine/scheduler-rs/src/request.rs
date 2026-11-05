//! The S2 request file's parser (was parse_request and its helpers in engine/runtime/s2/sessions.h). A request is a
//! header line, `n_gen <n> temp <f> seed <n> min <n> stop <ids|->` and then, in this order and each optional,
//! `top_p <f>`, `lp <k>`, `plp <k>`, `sys <n>`, `ins <rule>`, `gram <grammar>` and `steer <profile> <hex>`, then a
//! line of comma-separated prompt ids. The header is read the way libstdc++'s istringstream read it, so every request
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
        let v = std::str::from_utf8(&self.s[a..self.at]).ok().and_then(|t| t.parse::<f32>().ok());
        match v {
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

/// The parsed request (serve.h ServeReq, but steering already resolved by the caller's callback).
#[derive(Clone, Debug, PartialEq)]
pub struct Request {
    pub prompt: Vec<i32>,
    pub stop: Vec<i32>,
    pub n_gen: i32,
    pub min_tokens: i32,
    pub temp: f32,
    pub top_p: f32,
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
/// identity, or refuses.
pub fn parse(bytes: &[u8], grammar_ready: bool, resolve: &mut dyn FnMut(&SteerFields) -> R<Identity>) -> R<Request> {
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
    require(k.is_empty(), "invalid request header tail (S2 accepts top_p, lp, plp, sys, ins, gram and steer)")?;
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
        })
    }

    #[test]
    fn a_full_request() {
        let r = parse(
            b"n_gen 9 temp 0.5 seed 7 min 1 stop 3,4 top_p 0.9 plp 2 sys 5 ins 3:-1:8,9\n1,2,3\n",
            false,
            &mut off,
        )
        .unwrap();
        assert_eq!((r.n_gen, r.seed, r.min_tokens, r.temp, r.top_p, r.plp, r.sys), (9, 7, 1, 0.5, 0.9, 2, 5));
        assert_eq!((r.stop, r.prompt, r.ins.tokens), (vec![3, 4], vec![1, 2, 3], vec![8, 9]));
        assert_eq!(
            parse(b"n_gen 9 temp 0.5 seed 7 min 1 stop - lp 2 ins 3:-1:8\n1\n", false, &mut off).err(),
            Some(Msg("ins with lp"))
        );
    }

    #[test]
    fn numbers_end_where_libstdcxx_stops() {
        // "0.5seed" reads 0.5, then "seed" is the next word
        assert!(parse(b"n_gen 9 temp 0.5seed 7 min 1 stop -\n1\n", false, &mut off).is_ok());
        assert_eq!(
            parse(b"n_gen 9 temp 1e seed 7 min 1 stop -\n1\n", false, &mut off).err(),
            Some(Msg("invalid request header"))
        );
        assert_eq!(
            parse(b"n_gen 9 temp 1e99 seed 7 min 1 stop -\n1\n", false, &mut off).err(),
            Some(Msg("invalid request header"))
        );
        assert_eq!(
            parse(b"n_gen 9 temp 0 seed 7 min 1 stop - lp 21\n1\n", false, &mut off).err(),
            Some(Msg("invalid lp header"))
        );
        assert_eq!(
            parse(b"n_gen 9 temp 0 seed 7 min 1 stop - lp 2x\n1\n", false, &mut off).err(),
            Some(Msg("invalid request header tail (S2 accepts top_p, lp, plp, sys, ins, gram and steer)"))
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
