//! A small JSON reader for the width manifests and safetensors headers (no dependencies), and a writer that prints a
//! value exactly as Python's json.dump(v, indent=1, sort_keys=True) does, for manifests tools/convert writes too.

use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub enum Json {
    Null,
    Bool(bool),
    Int(i64),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(BTreeMap<String, Json>),
}

impl Json {
    pub fn get(&self, k: &str) -> Option<&Json> {
        match self {
            Json::Obj(m) => m.get(k),
            _ => None,
        }
    }
    pub fn arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }
    pub fn int(&self) -> Option<i64> {
        match self {
            Json::Int(n) => Some(*n),
            Json::Num(n) if n.fract() == 0.0 => Some(*n as i64),
            _ => None,
        }
    }
    pub fn num(&self) -> Option<f64> {
        match self {
            Json::Int(n) => Some(*n as f64),
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }
    /// Python's bool(value).
    pub fn truthy(&self) -> bool {
        match self {
            Json::Null => false,
            Json::Bool(b) => *b,
            Json::Int(n) => *n != 0,
            Json::Num(n) => *n != 0.0,
            Json::Str(s) => !s.is_empty(),
            Json::Arr(a) => !a.is_empty(),
            Json::Obj(m) => !m.is_empty(),
        }
    }
    pub fn str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn obj(&self) -> Option<&BTreeMap<String, Json>> {
        match self {
            Json::Obj(m) => Some(m),
            _ => None,
        }
    }
}

pub fn parse(text: &str) -> Result<Json, String> {
    let mut p = P { s: text.as_bytes(), i: 0 };
    let v = p.value()?;
    p.ws();
    if p.i != p.s.len() {
        return Err(format!("JSON: trailing bytes at {}", p.i));
    }
    Ok(v)
}

struct P<'a> {
    s: &'a [u8],
    i: usize,
}

impl P<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\n' | b'\r' | b'\t') {
            self.i += 1;
        }
    }
    fn err<T>(&self, what: &str) -> Result<T, String> {
        Err(format!("JSON: {what} at byte {}", self.i))
    }
    fn value(&mut self) -> Result<Json, String> {
        self.ws();
        match self.s.get(self.i) {
            None => self.err("unexpected end"),
            Some(b'{') => {
                self.i += 1;
                let mut m = BTreeMap::new();
                self.ws();
                if self.s.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    return Ok(Json::Obj(m));
                }
                loop {
                    self.ws();
                    let k = self.string()?;
                    self.ws();
                    if self.s.get(self.i) != Some(&b':') {
                        return self.err("expected :");
                    }
                    self.i += 1;
                    let v = self.value()?;
                    m.insert(k, v);
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(Json::Obj(m));
                        }
                        _ => return self.err("expected , or }"),
                    }
                }
            }
            Some(b'[') => {
                self.i += 1;
                let mut a = Vec::new();
                self.ws();
                if self.s.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return Ok(Json::Arr(a));
                }
                loop {
                    a.push(self.value()?);
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(Json::Arr(a));
                        }
                        _ => return self.err("expected , or ]"),
                    }
                }
            }
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') if self.s[self.i..].starts_with(b"true") => {
                self.i += 4;
                Ok(Json::Bool(true))
            }
            Some(b'f') if self.s[self.i..].starts_with(b"false") => {
                self.i += 5;
                Ok(Json::Bool(false))
            }
            Some(b'n') if self.s[self.i..].starts_with(b"null") => {
                self.i += 4;
                Ok(Json::Null)
            }
            Some(_) => {
                let a = self.i;
                while self.i < self.s.len() && matches!(self.s[self.i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
                    self.i += 1;
                }
                let t = std::str::from_utf8(&self.s[a..self.i]).unwrap_or("");
                if !t.contains(['.', 'e', 'E']) {
                    if let Ok(n) = t.parse::<i64>() {
                        return Ok(Json::Int(n));
                    }
                }
                t.parse::<f64>().map(Json::Num).or_else(|_| self.err("bad value"))
            }
        }
    }
    fn string(&mut self) -> Result<String, String> {
        if self.s.get(self.i) != Some(&b'"') {
            return self.err("expected string");
        }
        self.i += 1;
        let mut out = String::new();
        loop {
            let a = self.i;
            while self.i < self.s.len() && self.s[self.i] != b'"' && self.s[self.i] != b'\\' {
                self.i += 1;
            }
            out.push_str(std::str::from_utf8(&self.s[a..self.i]).map_err(|_| "JSON: bad UTF-8".to_string())?);
            match self.s.get(self.i) {
                None => return self.err("unterminated string"),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(_) => {
                    self.i += 1;
                    let c = *self.s.get(self.i).ok_or("JSON: bad escape")?;
                    self.i += 1;
                    match c {
                        b'n' => out.push('\n'),
                        b't' => out.push('\t'),
                        b'r' => out.push('\r'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'u' => {
                            let h = std::str::from_utf8(self.s.get(self.i..self.i + 4).ok_or("JSON: bad \\u")?)
                                .map_err(|_| "JSON: bad \\u".to_string())?;
                            let cp = u32::from_str_radix(h, 16).map_err(|_| "JSON: bad \\u".to_string())?;
                            out.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                            self.i += 4;
                        }
                        c => out.push(c as char),
                    }
                }
            }
        }
    }
}

/// The text Python's json.dump(v, f, indent=1, sort_keys=True) writes (keys sort the same: BTreeMap orders UTF-8
/// bytes, which is code point order).
pub fn dump_py(v: &Json) -> String {
    let mut out = String::new();
    dump(v, 0, &mut out);
    out
}

fn dump(v: &Json, depth: usize, out: &mut String) {
    let pad = |out: &mut String, d: usize| {
        out.push('\n');
        out.extend(std::iter::repeat_n(' ', d));
    };
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Int(n) => out.push_str(&n.to_string()),
        Json::Num(n) => out.push_str(&py_float(*n)),
        Json::Str(s) => py_str(s, out),
        Json::Arr(a) if a.is_empty() => out.push_str("[]"),
        Json::Obj(m) if m.is_empty() => out.push_str("{}"),
        Json::Arr(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                pad(out, depth + 1);
                dump(x, depth + 1, out);
            }
            pad(out, depth);
            out.push(']');
        }
        Json::Obj(m) => {
            out.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                pad(out, depth + 1);
                py_str(k, out);
                out.push_str(": ");
                dump(x, depth + 1, out);
            }
            pad(out, depth);
            out.push('}');
        }
    }
}

/// A string as json.dump writes it with ensure_ascii: anything outside printable ASCII as an escape.
fn py_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            _ => {
                let mut b = [0u16; 2];
                for u in c.encode_utf16(&mut b) {
                    out.push_str(&format!("\\u{u:04x}"));
                }
            }
        }
    }
    out.push('"');
}

/// Python's repr(float): the shortest digits that read back exactly, in fixed notation when the decimal exponent is
/// -4 to 15 (with ".0" when integral), else as d.ddde+XX.
pub fn py_float(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    let e = format!("{x:e}"); // shortest round trip, as "-1.5e-5"
    let (m, exp) = e.split_once('e').unwrap();
    let (sign, m) = m.strip_prefix('-').map_or(("", m), |m| ("-", m));
    let digits: String = m.chars().filter(|c| *c != '.').collect();
    let exp: i32 = exp.parse().unwrap();
    let decpt = exp + 1; // value = 0.<digits> x 10^decpt
    let n = digits.len() as i32;
    let body = if decpt <= -4 || decpt > 16 {
        let mant = if n > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits.clone() };
        format!("{mant}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    } else if decpt <= 0 {
        format!("0.{}{digits}", "0".repeat((-decpt) as usize))
    } else if decpt >= n {
        format!("{digits}{}.0", "0".repeat((decpt - n) as usize))
    } else {
        format!("{}.{}", &digits[..decpt as usize], &digits[decpt as usize..])
    };
    format!("{sign}{body}")
}

#[cfg(test)]
mod tests {
    #[test]
    fn python_float_repr() {
        for (x, want) in [(1e-5, "1e-05"), (0.0001, "0.0001"), (1e16, "1e+16"), (1e15, "1000000000000000.0"),
                          (1000000.0, "1000000.0"), (0.0, "0.0"), (-0.0, "-0.0"), (1.5e-7, "1.5e-07"),
                          (123.456, "123.456"), (-2.5, "-2.5"), (1e100, "1e+100"), (0.1, "0.1"),
                          (1.7976931348623157e308, "1.7976931348623157e+308"), (5e-324, "5e-324")] {
            assert_eq!(super::py_float(x), want, "{x}");
        }
    }

    #[test]
    fn dump_matches_python() {
        // json.dump({"b": [1, 2.0, None, True], "a": {}, "c": [], "d": {"x": "q\"\u00e9"}}, f, indent=1, sort_keys=True)
        let j = super::parse(r#"{"b": [1, 2.0, null, true], "a": {}, "c": [], "d": {"x": "q\"\u00e9"}}"#).unwrap();
        assert_eq!(super::dump_py(&j),
                   "{\n \"a\": {},\n \"b\": [\n  1,\n  2.0,\n  null,\n  true\n ],\n \"c\": [],\n \"d\": {\n  \"x\": \"q\\\"\\u00e9\"\n }\n}");
    }

    #[test]
    fn parses_manifest_shapes() {
        let j = super::parse(r#"{"a": [1, 2, [3, 4]], "b": {"c": "x\"y", "d": -1.5e2}, "e": null}"#).unwrap();
        assert_eq!(j.get("a").unwrap().arr().unwrap()[2].arr().unwrap()[1].int(), Some(4));
        assert_eq!(j.get("b").unwrap().get("c").unwrap().str(), Some("x\"y"));
        assert!(matches!(j.get("b").unwrap().get("d").unwrap(), super::Json::Num(n) if *n == -150.0));
    }
}
