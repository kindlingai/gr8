//! JSON text exactly as CPython's `json.dumps(..., ensure_ascii=False)` writes it, so responses diff clean against
//! the Python front end: `repr()` floats (`1e-05`, `1e+16`, `-9999.0`), only `"`, `\` and C0 controls escaped
//! (`\u001f` lowercase), keys in insertion order. Writes straight into a `Vec<u8>`; no intermediate tree.

use serde_json::Value;

/// `repr(float)`: shortest round-trip digits; fixed notation for 1e-4 <= |x| < 1e16, else `d.ddde+XX`.
pub fn float(out: &mut Vec<u8>, x: f64) {
    if x.is_nan() {
        out.extend_from_slice(b"NaN");
        return;
    }
    if x.is_infinite() {
        out.extend_from_slice(if x > 0.0 { b"Infinity" } else { b"-Infinity" });
        return;
    }
    if x == 0.0 {
        out.extend_from_slice(if x.is_sign_negative() { b"-0.0" } else { b"0.0" });
        return;
    }
    // Rust's `{:e}` is the shortest round-trip digit string, the same digits CPython's repr picks.
    let mut tmp = [0u8; 40];
    let s = {
        use std::io::Write;
        let mut cur = std::io::Cursor::new(&mut tmp[..]);
        write!(cur, "{:e}", x.abs()).unwrap();
        let n = cur.position() as usize;
        &tmp[..n]
    };
    let e_at = s.iter().position(|&c| c == b'e').unwrap();
    let (mant, exp) = (&s[..e_at], &s[e_at + 1..]);
    let exp: i32 = std::str::from_utf8(exp).unwrap().parse().unwrap();
    let mut digits = [0u8; 20];
    let mut nd = 0;
    for &c in mant {
        if c != b'.' {
            digits[nd] = c;
            nd += 1;
        }
    }
    let digits = &digits[..nd];
    let decpt = exp + 1; // value = 0.DIGITS * 10^decpt
    if x < 0.0 {
        out.push(b'-');
    }
    if -4 < decpt && decpt <= 16 {
        if decpt <= 0 {
            out.extend_from_slice(b"0.");
            out.extend(std::iter::repeat_n(b'0', (-decpt) as usize));
            out.extend_from_slice(digits);
        } else if decpt as usize >= nd {
            out.extend_from_slice(digits);
            out.extend(std::iter::repeat_n(b'0', decpt as usize - nd));
            out.extend_from_slice(b".0");
        } else {
            out.extend_from_slice(&digits[..decpt as usize]);
            out.push(b'.');
            out.extend_from_slice(&digits[decpt as usize..]);
        }
    } else {
        out.push(digits[0]);
        if nd > 1 {
            out.push(b'.');
            out.extend_from_slice(&digits[1..]);
        }
        out.push(b'e');
        out.push(if exp < 0 { b'-' } else { b'+' });
        let a = exp.unsigned_abs();
        if a < 10 {
            out.push(b'0');
        }
        out.extend_from_slice(itoa::Buffer::new().format(a).as_bytes());
    }
}

pub fn float_string(x: f64) -> String {
    let mut v = Vec::with_capacity(24);
    float(&mut v, x);
    String::from_utf8(v).unwrap()
}

#[inline]
pub fn int<I: itoa::Integer>(out: &mut Vec<u8>, i: I) {
    out.extend_from_slice(itoa::Buffer::new().format(i).as_bytes());
}

/// A JSON string literal, CPython `ensure_ascii=False` escaping.
pub fn string(out: &mut Vec<u8>, s: &str) {
    out.reserve(s.len() + 2);
    out.push(b'"');
    let b = s.as_bytes();
    let mut start = 0;
    for (i, &c) in b.iter().enumerate() {
        let esc: &[u8] = match c {
            b'"' => b"\\\"",
            b'\\' => b"\\\\",
            b'\n' => b"\\n",
            b'\r' => b"\\r",
            b'\t' => b"\\t",
            0x08 => b"\\b",
            0x0c => b"\\f",
            0..=0x1f => b"",
            _ => continue,
        };
        out.extend_from_slice(&b[start..i]);
        if esc.is_empty() {
            const HEX: &[u8; 16] = b"0123456789abcdef";
            out.extend_from_slice(b"\\u00");
            out.push(HEX[(c >> 4) as usize]);
            out.push(HEX[(c & 15) as usize]);
        } else {
            out.extend_from_slice(esc);
        }
        start = i + 1;
    }
    out.extend_from_slice(&b[start..]);
    out.push(b'"');
}

pub fn number(out: &mut Vec<u8>, n: &serde_json::Number) {
    if let Some(i) = n.as_i64() {
        int(out, i);
    } else if let Some(u) = n.as_u64() {
        int(out, u);
    } else {
        float(out, n.as_f64().unwrap_or(f64::NAN));
    }
}

/// `json.dumps(v, ensure_ascii=False)`: default separators `, ` and `: `.
pub fn value(out: &mut Vec<u8>, v: &Value) {
    match v {
        Value::Null => out.extend_from_slice(b"null"),
        Value::Bool(true) => out.extend_from_slice(b"true"),
        Value::Bool(false) => out.extend_from_slice(b"false"),
        Value::Number(n) => number(out, n),
        Value::String(s) => string(out, s),
        Value::Array(a) => {
            out.push(b'[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.extend_from_slice(b", ");
                }
                value(out, x);
            }
            out.push(b']');
        }
        Value::Object(m) => {
            out.push(b'{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.extend_from_slice(b", ");
                }
                string(out, k);
                out.extend_from_slice(b": ");
                value(out, x);
            }
            out.push(b'}');
        }
    }
}

pub fn to_string(v: &Value) -> String {
    let mut out = Vec::new();
    value(&mut out, v);
    String::from_utf8(out).unwrap()
}

/// Python truthiness of a JSON value (`bool(x)` after `json.loads`).
pub fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64() != Some(0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

/// Python `x == n` for a JSON value and an int (`True == 1`, `1.0 == 1`).
pub fn eq_int(v: &Value, n: i64) -> bool {
    match v {
        Value::Bool(b) => (*b as i64) == n,
        Value::Number(x) => x.as_f64() == Some(n as f64),
        _ => false,
    }
}

/// CPython `str.isspace()` for one char (unicode whitespace plus the \x1c-\x1f separators).
#[inline]
pub fn is_py_space(c: char) -> bool {
    c.is_whitespace() || ('\x1c'..='\x1f').contains(&c)
}

pub fn py_strip(s: &str) -> &str {
    s.trim_matches(is_py_space)
}

pub fn py_rstrip(s: &str) -> &str {
    s.trim_end_matches(is_py_space)
}

/// Exact decimal text of an integral f64 (`str(int(f))`), any magnitude.
pub fn integral_float_text(f: f64) -> String {
    if f.abs() < 9.0e15 {
        return (f as i64).to_string();
    }
    let bits = f.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i32 - 1075;
    let mant = (bits & ((1u64 << 52) - 1)) | (1u64 << 52);
    // |f| >= 9e15 > 2^52, so exp >= 0: value = mant * 2^exp. Little-endian base-1e9 limbs.
    let mut limbs: Vec<u64> = vec![mant % 1_000_000_000, mant / 1_000_000_000];
    for _ in 0..exp {
        let mut carry = 0u64;
        for l in limbs.iter_mut() {
            let v = *l * 2 + carry;
            *l = v % 1_000_000_000;
            carry = v / 1_000_000_000;
        }
        if carry > 0 {
            limbs.push(carry);
        }
    }
    while limbs.len() > 1 && *limbs.last().unwrap() == 0 {
        limbs.pop();
    }
    let mut s = String::new();
    if f < 0.0 {
        s.push('-');
    }
    s += &limbs.last().unwrap().to_string();
    for l in limbs.iter().rev().skip(1) {
        s += &format!("{:09}", l);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(x: f64) -> String {
        float_string(x)
    }

    #[test]
    fn floats_match_cpython_repr() {
        let cases: &[(f64, &str)] = &[
            (1.0, "1.0"),
            (-9999.0, "-9999.0"),
            (0.1, "0.1"),
            (-0.012, "-0.012"),
            (1e-4, "0.0001"),
            (1e-5, "1e-05"),
            (1.5e-7, "1.5e-07"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1.2345e16, "1.2345e+16"),
            (123456789.125, "123456789.125"),
            (1e100, "1e+100"),
            (-0.0, "-0.0"),
            (0.0, "0.0"),
            (2.5e-300, "2.5e-300"),
            (-0.5978370308876038, "-0.5978370308876038"),
            (1200.0, "1200.0"),
        ];
        for &(x, want) in cases {
            assert_eq!(f(x), want, "{x:e}");
        }
    }

    #[test]
    fn strings_escape_like_cpython() {
        let mut o = Vec::new();
        string(&mut o, "a\"b\\c\n\u{1}é\u{7f}\u{2028}");
        assert_eq!(String::from_utf8(o).unwrap(), "\"a\\\"b\\\\c\\n\\u0001é\u{7f}\u{2028}\"");
    }

    #[test]
    fn integral_floats() {
        assert_eq!(integral_float_text(1e3), "1000");
        assert_eq!(integral_float_text(1e20), "100000000000000000000");
        assert_eq!(integral_float_text(-1.5e300).len(), 302);
        assert_eq!(integral_float_text(2f64.powi(70)), "1180591620717411303424");
    }
}
