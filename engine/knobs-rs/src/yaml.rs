//! The knobs/ YAML files (knobs/README.md) as entries, with the rules a file must keep. build.rs uses it to write the
//! table, and the crate's tests use it to check the rules.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use yaml_rust2::{Yaml, YamlLoader};

const SCOPES: [&str; 4] = ["agree", "local", "test", "outside"];
const KINDS: [&str; 7] = ["flag", "int", "float", "text", "path", "ints", "floats"];
const FIELDS: [&str; 7] = ["scope", "kind", "default", "unset", "values", "legacy", "doc"];

/// One knob as its file gives it.
pub struct Entry {
    pub family: bool,
    pub scope: String,
    pub kind: String,
    pub default: Option<String>,
    pub unset: String,
    pub values: String,
    /// The whole name of a variable outside the GR8_ space.
    pub env: Option<String>,
    pub legacy: Vec<String>,
    pub doc: String,
    pub file: String,
}

fn text(y: &Yaml, what: &str) -> Result<String, String> {
    y.as_str().map(str::to_string).ok_or_else(|| format!("{what}: want a string"))
}

fn entry(file: &str, prefix: &str, fscope: Option<&str>, key: &str, y: &Yaml) -> Result<(String, Entry), String> {
    let at = |e: String| format!("{file}: {key}: {e}");
    let h = y.as_hash().ok_or_else(|| at("want a mapping".into()))?;
    let get = |f: &str| h.get(&Yaml::String(f.into()));
    for k in h.keys() {
        let k = k.as_str().unwrap_or("");
        if !FIELDS.contains(&k) {
            return Err(at(format!("unknown field {k:?}")));
        }
    }
    let (var, family) = match key.strip_suffix("<n>") {
        Some(v) => (v, true),
        None => (key, false),
    };
    if !var.starts_with(prefix) || var.len() == prefix.len() {
        return Err(at(format!("the names in this file start with {prefix:?}")));
    }
    if !var.bytes().all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_') {
        return Err(at("a name is upper-case letters, digits and _".into()));
    }
    let scope = match (fscope, get("scope")) {
        (Some(_), Some(_)) => return Err(at("this file sets the scope of every entry".into())),
        (Some(s), None) => s.to_string(),
        (None, Some(s)) => text(s, "scope").map_err(at)?,
        (None, None) => return Err(at("no scope".into())),
    };
    if !SCOPES.contains(&scope.as_str()) {
        return Err(at(format!("scope {scope:?}: want one of {SCOPES:?}")));
    }
    if scope == "test" && fscope != Some("test") {
        return Err(at("a test knob belongs in a file whose scope is test".into()));
    }
    let kind = text(get("kind").ok_or_else(|| at("no kind".into()))?, "kind").map_err(at)?;
    if !KINDS.contains(&kind.as_str()) {
        return Err(at(format!("kind {kind:?}: want one of {KINDS:?}")));
    }
    let (default, unset) = match (get("default"), get("unset")) {
        (Some(d), None) => (Some(text(d, "default").map_err(at)?), String::new()),
        (None, Some(u)) => (None, text(u, "unset").map_err(at)?),
        _ => return Err(at("give exactly one of default and unset".into())),
    };
    let opt = |f: &str| get(f).map(|v| text(v, f).map_err(at)).transpose().map(Option::unwrap_or_default);
    let (name, env) = match var.strip_prefix("GR8_") {
        Some(n) => (n.to_string(), None),
        None => (var.to_string(), Some(var.to_string())),
    };
    let legacy = match get("legacy") {
        None if env.is_none() => vec![format!("G6_{name}")],
        None => vec![],
        Some(l) => l
            .as_vec()
            .ok_or_else(|| at("legacy: want a list".into()))?
            .iter()
            .map(|v| text(v, "legacy").map_err(at))
            .collect::<Result<_, _>>()?,
    };
    let e = Entry {
        family,
        scope,
        kind,
        default,
        unset,
        values: opt("values")?,
        env,
        legacy,
        doc: opt("doc")?,
        file: file.to_string(),
    };
    Ok((name, e))
}

/// Adds the entries of `src`, the text of the YAML file `file`, to `out`, or returns the first rule it breaks.
pub fn add(out: &mut BTreeMap<String, Entry>, file: &str, src: &str) -> Result<(), String> {
    let docs = YamlLoader::load_from_str(src).map_err(|e| format!("{file}: {e}"))?;
    let [doc] = docs.as_slice() else { return Err(format!("{file}: want one YAML document")) };
    let prefix = text(&doc["prefix"], "prefix").map_err(|e| format!("{file}: {e}"))?;
    let fscope = doc["scope"].as_str();
    let knobs = doc["knobs"].as_hash().ok_or_else(|| format!("{file}: no knobs mapping"))?;
    for (k, v) in knobs {
        let key = k.as_str().ok_or_else(|| format!("{file}: a knob's name is not a string"))?;
        let (name, e) = entry(file, &prefix, fscope, key, v)?;
        if let Some(was) = out.get(&name) {
            return Err(format!("{file}: {key} is also in {}", was.file));
        }
        out.insert(name, e);
    }
    Ok(())
}

/// The YAML files in `dir`, in name order.
pub fn files(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "yaml"))
        .collect();
    paths.sort();
    Ok(paths)
}

/// Every entry of the YAML files in `dir`, by NAME: GR8_NAME's NAME, or the whole name of a variable outside the GR8_
/// space.
pub fn load(dir: &Path) -> Result<BTreeMap<String, Entry>, String> {
    let mut out = BTreeMap::new();
    for p in files(dir)? {
        let file = p.file_name().unwrap().to_string_lossy().into_owned();
        let src = std::fs::read_to_string(&p).map_err(|e| format!("{file}: {e}"))?;
        add(&mut out, &file, &src)?;
    }
    Ok(out)
}
