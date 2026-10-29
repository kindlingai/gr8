//! The engine's settings from the environment, all in one table (the knobs/ YAML files). Code reads a setting through its typed
//! knob: `knobs::ATTN_CTX2.get()` in Rust, `knob::attn_ctx2()` in C++ (gr8_knobs.h, which `knobs gen` writes from the
//! table). The knob NAME is the variable GR8_NAME. Its old name, G6_NAME, still works and prints a warning.
//!
//! The environment is read once, at the first read of any knob, so a value cannot change while the engine runs. A value
//! the table does not accept (a word for a number, a number out of range) stops the process at that first read. The
//! engine calls [`problems`] at boot (C: `gr8_knobs_init`) to print every problem at once, before it reads any knob.

use std::collections::BTreeMap;
use std::ffi::{CString, OsString};
use std::marker::PhantomData;
use std::sync::OnceLock;

#[cfg(feature = "capi")]
pub mod capi;
pub mod gen;
mod table;
pub use table::*;

/// Which ranks must agree on a knob's value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Scope {
    /// Every rank of one engine runs the same value: set values join the S2 runtime signature.
    Agree,
    /// May differ between ranks: paths, per-box resources, logging and timing.
    Local,
    /// Read by tests, audits and benchmarks only.
    Test,
    /// Read by the build or the launcher, never by the engine.
    Outside,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// 0 or 1.
    Flag,
    /// A decimal integer, or hex with 0x.
    Int,
    Float,
    Text,
    Path,
    /// Comma-separated integers.
    Ints,
    /// Comma-separated floats.
    Floats,
}

impl Kind {
    pub fn word(self) -> &'static str {
        match self {
            Kind::Flag => "flag",
            Kind::Int => "int",
            Kind::Float => "float",
            Kind::Text => "text",
            Kind::Path => "path",
            Kind::Ints => "ints",
            Kind::Floats => "floats",
        }
    }
}

/// One table row.
pub struct Spec {
    /// NAME in GR8_NAME.
    pub name: &'static str,
    /// The variables are NAME followed by any decimal number: GR8_MTP_KSTOPB_TAU_S1, ..S8.
    pub family: bool,
    pub scope: Scope,
    pub kind: Kind,
    /// The value an unset variable has, written as a variable's value would be.
    pub default: Option<&'static str>,
    /// With no default: what the code does when the variable is unset.
    pub unset: &'static str,
    /// The accepted values: "" any, "lo..hi" (either end may be left out) or "a|b|c". A list checks each element.
    pub values: &'static str,
    /// The variable's exact name, in place of GR8_NAME (a system variable such as TMPDIR).
    pub env: Option<&'static str>,
    /// The old names the variable is still read under, with a warning: G6_NAME unless knobs/ says otherwise.
    pub legacy: &'static [&'static str],
    pub doc: &'static str,
}

impl Spec {
    /// The variable's name, with `n` for a family.
    pub fn var(&self, n: Option<u32>) -> String {
        let base = self.env.map(str::to_string).unwrap_or_else(|| format!("GR8_{}", self.name));
        match n {
            Some(n) => format!("{base}{n}"),
            None => base,
        }
    }
}

/// A parsed value. Text keeps its NUL so the C API can hand out a pointer.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Flag(bool),
    Int(i64),
    Float(f64),
    Text(CString),
    Ints(Vec<i64>),
    Floats(Vec<f64>),
}

impl Value {
    /// The value written back as a variable's value, one spelling per value, for the runtime signature.
    pub fn canon(&self) -> String {
        let join = |v: Vec<String>| v.join(",");
        match self {
            Value::Flag(b) => (*b as u8).to_string(),
            Value::Int(i) => i.to_string(),
            Value::Float(f) => f.to_string(),
            Value::Text(s) => s.to_str().unwrap().replace('\n', "\\n"),
            Value::Ints(v) => join(v.iter().map(i64::to_string).collect()),
            Value::Floats(v) => join(v.iter().map(f64::to_string).collect()),
        }
    }
}

fn int(s: &str) -> Option<i64> {
    let (neg, digits) = match s.strip_prefix('-') {
        Some(d) => (true, d),
        None => (false, s.strip_prefix('+').unwrap_or(s)),
    };
    let v = match digits.strip_prefix("0x").or_else(|| digits.strip_prefix("0X")) {
        Some(h) => i64::from_str_radix(h, 16).ok()?,
        None if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) => digits.parse().ok()?,
        None => return None,
    };
    Some(if neg { -v } else { v })
}

fn float(s: &str) -> Option<f64> {
    s.parse::<f64>().ok().filter(|f| f.is_finite())
}

/// `raw` as a value of `spec`, or why the table refuses it.
pub fn parse(spec: &Spec, raw: &str) -> Result<Value, String> {
    let v = match spec.kind {
        Kind::Flag => match raw {
            "0" => Value::Flag(false),
            "1" => Value::Flag(true),
            _ => return Err("takes 0 or 1".into()),
        },
        Kind::Int => Value::Int(int(raw).ok_or("is not an integer")?),
        Kind::Float => Value::Float(float(raw).ok_or("is not a number")?),
        Kind::Text | Kind::Path => Value::Text(CString::new(raw).map_err(|_| "holds a NUL")?),
        Kind::Ints => Value::Ints(
            raw.split(',').map(|s| int(s.trim())).collect::<Option<_>>().ok_or("is not a comma list of integers")?,
        ),
        Kind::Floats => Value::Floats(
            raw.split(',').map(|s| float(s.trim())).collect::<Option<_>>().ok_or("is not a comma list of numbers")?,
        ),
    };
    allowed(spec, &v)?;
    Ok(v)
}

fn allowed(spec: &Spec, v: &Value) -> Result<(), String> {
    let values = spec.values;
    if values.is_empty() {
        return Ok(());
    }
    let nums: Vec<f64> = match v {
        Value::Flag(b) => vec![*b as u8 as f64],
        Value::Int(i) => vec![*i as f64],
        Value::Float(f) => vec![*f],
        Value::Ints(l) => l.iter().map(|&i| i as f64).collect(),
        Value::Floats(l) => l.clone(),
        Value::Text(s) => {
            let s = s.to_str().unwrap();
            return if values.split('|').any(|w| w == s) { Ok(()) } else { Err(format!("is not one of {values}")) };
        }
    };
    let ok = |x: f64| match values.split_once("..") {
        Some((lo, hi)) => {
            (lo.is_empty() || float(lo).is_some_and(|lo| x >= lo))
                && (hi.is_empty() || float(hi).is_some_and(|hi| x <= hi))
        }
        None => values.split('|').any(|w| int(w).map(|i| i as f64).or_else(|| float(w)) == Some(x)),
    };
    match nums.iter().all(|&x| ok(x)) {
        true => Ok(()),
        false if values.contains("..") => Err(format!("is outside {values}")),
        false => Err(format!("is not one of {values}")),
    }
}

/// Something wrong with the environment.
#[derive(Clone, Debug, PartialEq)]
pub struct Problem {
    /// An error stops the engine at boot. A warning does not.
    pub error: bool,
    pub text: String,
}

/// The knob values of one environment.
pub struct Env {
    /// (table index, family number or 0) -> the parsed value, or why the table refused it
    values: BTreeMap<(usize, u32), Result<Value, String>>,
    problems: Vec<Problem>,
}

/// A variable that is set: its name and value.
type Set = (String, String);

/// Which knob a variable sets: (table index, family number, whether it is the old name).
fn resolve(var: &str) -> Option<(usize, Option<u32>, bool)> {
    for (i, s) in TABLE.iter().enumerate() {
        let names = std::iter::once((s.var(None), false)).chain(s.legacy.iter().map(|l| (l.to_string(), true)));
        for (name, old) in names {
            if !s.family {
                if var == name {
                    return Some((i, None, old));
                }
            } else if let Some(d) = var.strip_prefix(name.as_str()) {
                if !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()) {
                    return Some((i, Some(d.parse().ok()?), old));
                }
            }
        }
    }
    None
}

impl Env {
    /// The knob values of `vars`, an environment's (name, value) pairs. An empty value counts as unset.
    pub fn read(vars: impl IntoIterator<Item = (OsString, OsString)>) -> Env {
        // (index, n) -> (the GR8_ variable and value, the old variable and value)
        let mut seen: BTreeMap<(usize, Option<u32>), [Option<Set>; 2]> = BTreeMap::new();
        let mut problems = Vec::new();
        for (k, v) in vars {
            let Some(k) = k.to_str() else { continue };
            let Some((i, n, old)) = resolve(k) else {
                if k.starts_with("GR8_") || k.starts_with("G6_") {
                    problems.push(Problem {
                        error: true,
                        text: format!(
                            "{k} is not an engine variable (docs/ENV.md): a typo, or a knob that was removed"
                        ),
                    });
                }
                continue;
            };
            let Some(v) = v.to_str() else {
                problems.push(Problem { error: true, text: format!("{k}: the value is not UTF-8") });
                continue;
            };
            seen.entry((i, n)).or_default()[old as usize] = Some((k.to_string(), v.to_string()));
        }
        let mut values = BTreeMap::new();
        for ((i, n), [new, old]) in seen {
            let spec = &TABLE[i];
            if let Some((ok, ov)) = &old {
                match &new {
                    Some((nk, nv)) if nv != ov => {
                        problems.push(Problem { error: true, text: format!("{ok}={ov} and {nk}={nv}: set only {nk}") })
                    }
                    _ => problems.push(Problem {
                        error: false,
                        text: format!("{ok} is the old name of {}: rename it, the old name will go away", spec.var(n)),
                    }),
                }
            }
            let Some((k, v)) = new.or(old) else { continue };
            if v.is_empty() {
                continue;
            }
            let parsed = parse(spec, &v).map_err(|e| format!("{k}={v} {e} (docs/ENV.md)"));
            if let Err(e) = &parsed {
                problems.push(Problem { error: true, text: e.clone() });
            }
            values.insert((i, n.unwrap_or(0)), parsed);
        }
        problems.sort_by(|a, b| b.error.cmp(&a.error).then(a.text.cmp(&b.text)));
        Env { values, problems }
    }

    /// Everything wrong with this environment, the errors first.
    pub fn problems(&self) -> &[Problem] {
        &self.problems
    }

    /// The value of knob `i` (number `n` of a family), or None when unset.
    pub fn get(&self, i: usize, n: u32) -> Option<&Value> {
        match self.values.get(&(i, n))? {
            Ok(v) => Some(v),
            Err(e) => {
                eprintln!("gr8: {e}");
                std::process::exit(2)
            }
        }
    }

    /// The set agree-scope knobs whose value is not the default, as " env NAME=value ..." in table order, or "" when
    /// there are none. The engine adds it to the S2 runtime signature, so a rank whose value differs refuses to start.
    pub fn signature(&self) -> String {
        let mut out = String::new();
        for (&(i, n), v) in &self.values {
            let (spec, Ok(v)) = (&TABLE[i], v) else { continue };
            if spec.scope != Scope::Agree || defaults()[i].as_ref() == Some(v) {
                continue;
            }
            let num = if spec.family { n.to_string() } else { String::new() };
            out += &format!("{} {}{num}={}", if out.is_empty() { " env" } else { "" }, spec.name, v.canon());
        }
        out
    }
}

/// This process's environment, read at the first call.
pub fn env() -> &'static Env {
    static ENV: OnceLock<Env> = OnceLock::new();
    ENV.get_or_init(|| Env::read(std::env::vars_os()))
}

/// Everything wrong with this process's environment, the errors first.
pub fn problems() -> &'static [Problem] {
    env().problems()
}

/// The parsed defaults, by table index.
pub fn defaults() -> &'static [Option<Value>] {
    static D: OnceLock<Vec<Option<Value>>> = OnceLock::new();
    D.get_or_init(|| {
        TABLE
            .iter()
            .map(|s| s.default.map(|d| parse(s, d).unwrap_or_else(|e| panic!("{}: default {e}", s.name))))
            .collect()
    })
}

/// The value of knob `i`, its default when unset.
pub fn value(i: usize, n: u32) -> Option<&'static Value> {
    env().get(i, n).or(defaults()[i].as_ref())
}

/// A Rust type a knob's value reads as.
pub trait Read: Sized {
    fn read(v: &'static Value) -> Self;
}

macro_rules! read {
    ($t:ty, $($pat:pat => $e:expr),+) => {
        impl Read for $t {
            fn read(v: &'static Value) -> $t {
                match v {
                    $($pat => $e,)+
                    _ => unreachable!("the table's kind and the knob's type disagree"),
                }
            }
        }
    };
}
read!(bool, Value::Flag(b) => *b);
read!(i64, Value::Int(i) => *i);
read!(f64, Value::Float(f) => *f);
read!(&'static str, Value::Text(s) => s.to_str().unwrap());
read!(&'static [i64], Value::Ints(v) => v);
read!(&'static [f64], Value::Floats(v) => v);

/// A knob with a default.
pub struct Knob<T>(pub usize, PhantomData<fn() -> T>);
/// A knob without one: None when unset.
pub struct OptKnob<T>(pub usize, PhantomData<fn() -> T>);
/// A family: NAME<n> for any n, None when unset.
pub struct Family<T>(pub usize, PhantomData<fn() -> T>);

impl<T: Read> Knob<T> {
    pub const fn new(i: usize) -> Self {
        Knob(i, PhantomData)
    }
    pub fn get(&self) -> T {
        T::read(value(self.0, 0).unwrap())
    }
    /// The variable's name, for a test that sets it for a child process.
    pub fn var(&self) -> String {
        TABLE[self.0].var(None)
    }
}

impl<T: Read> OptKnob<T> {
    pub const fn new(i: usize) -> Self {
        OptKnob(i, PhantomData)
    }
    pub fn get(&self) -> Option<T> {
        value(self.0, 0).map(T::read)
    }
    /// The variable's name, for a test that sets it for a child process.
    pub fn var(&self) -> String {
        TABLE[self.0].var(None)
    }
}

impl<T: Read> Family<T> {
    pub const fn new(i: usize) -> Self {
        Family(i, PhantomData)
    }
    pub fn get(&self, n: u32) -> Option<T> {
        value(self.0, n).map(T::read)
    }
    /// The variable's name for number `n`, for a test that sets it for a child process.
    pub fn var(&self, n: u32) -> String {
        TABLE[self.0].var(Some(n))
    }
}

/// An FNV-1a hash of the table's names, kinds and default presence. gr8_knobs.h carries the hash of the table it was
/// written from, and gr8_knobs_init refuses a header from another table.
pub fn table_hash() -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for s in TABLE {
        for b in s.name.bytes().chain([s.kind as u8, s.family as u8, s.default.is_some() as u8, 0]) {
            h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

#[cfg(test)]
mod tests;
