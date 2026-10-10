//! The exlr8 v2 fetch helper for ./gr8 glm53 up --model https://... (tools/launch/hfsource.py and
//! tools/tune/fetch_ranks.py keep the HTTP; this plans and checks). JSON on stdout.
//!
//!   g6load fetch-plan --slices DIR --tp N --rank R (--manifest FILE | --widths 2,3,4) [--layers 3-40,78]
//!       the byte ranges of each width file rank R loads: {"units", "bytes", "files": [{"name", "ranges": [[off, n]]}]}
//!   g6load verify --slices DIR --v2 DIR --tp N --rank R (--manifest FILE | --widths ...) [--layers ...]
//!       each of those halves' xxhash64 against the index slices: {"checked", "bad": [{"file", "offset", "nbytes",
//!       "layer", "expert", "half", "proj"}]}. Exit 0 when every half matches, 3 when one does not, 1 on an error.
//!
//! DIR for --slices holds tp<N>/k<K>/rank<R>.safetensors (the release's index-slices/).
//!
//!   g6load dflash2 --src DIR --out DIR [--nr 4] [--ranks 0,1,2,3] [--bf16 fc,attn,mlp,conv,sel]
//!       converts the DFlash2 drafter (config.json + model.safetensors in --src) into dflash.r<r>.pack and
//!       dflash.r<r>.manifest.json for each rank, byte for byte what tools/convert/dflash2.py writes. One JSON line
//!       per rank: {"rank", "tensors", "bytes", "seconds"}.

use g6load::dflash2;
use g6load::fetch::{self, Choice};
use std::collections::BTreeMap;

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("g6load: {e}");
            std::process::exit(1);
        }
    }
}

fn layers(spec: &str) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for part in spec.split(',').filter(|s| !s.is_empty()) {
        let (a, b) = part.split_once('-').unwrap_or((part, part));
        let n = |s: &str| s.parse::<usize>().map_err(|_| format!("--layers {spec}: not a number: {s}"));
        out.extend(n(a)?..=n(b)?);
    }
    Ok(out)
}

fn run() -> Result<i32, String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let cmd = argv.first().ok_or("usage: g6load fetch-plan|verify|dflash2 (see the source) --slices DIR --tp N --rank R (--manifest F | --widths K,..)")?;
    let mut opt: BTreeMap<&str, &str> = BTreeMap::new();
    let mut it = argv[1..].iter();
    while let Some(k) = it.next() {
        let v = it.next().ok_or(format!("{k} needs a value"))?;
        opt.insert(k.as_str(), v.as_str());
    }
    let need = |k: &str| opt.get(k).copied().ok_or(format!("{cmd}: {k} is required"));
    let num = |k: &str| need(k)?.parse::<usize>().map_err(|_| format!("{k}: not a number"));
    if cmd == "dflash2" {
        let nr = opt.get("--nr").map_or(Ok(4), |_| num("--nr"))?;
        dflash2::main(need("--src")?, need("--out")?, nr, opt.get("--ranks").copied(), opt.get("--bf16").copied().unwrap_or(""),
                      &dflash2::GLM53)?;
        return Ok(0);
    }
    let choice = match (opt.get("--manifest"), opt.get("--widths")) {
        (Some(m), None) => Choice::manifest(m)?,
        (None, Some(w)) => Choice::Widths(w.split(',').map(|k| k.parse::<u8>().map_err(|_| format!("--widths {w}")))
            .collect::<Result<_, _>>()?),
        _ => return Err("give one of --manifest and --widths".into()),
    };
    let ls = opt.get("--layers").map(|s| layers(s)).transpose()?;
    let units = fetch::units(need("--slices")?, num("--tp")?, num("--rank")?, &choice, ls.as_deref())?;
    match cmd.as_str() {
        "fetch-plan" => {
            let mut files: BTreeMap<String, Vec<(u64, u64)>> = BTreeMap::new();
            for u in &units {
                files.entry(fetch::width_file(u.layer, u.k)).or_default().push((u.offset, u.nbytes));
            }
            let bytes: u64 = units.iter().map(|u| u.nbytes).sum();
            let mut out = format!("{{\"units\": {}, \"bytes\": {bytes}, \"files\": [", units.len());
            for (i, (name, mut r)) in files.into_iter().enumerate() {
                r.sort();
                let rs: Vec<String> = r.iter().map(|(a, n)| format!("[{a}, {n}]")).collect();
                out += &format!("{}\n{{\"name\": \"{name}\", \"ranges\": [{}]}}", if i > 0 { "," } else { "" }, rs.join(", "));
            }
            println!("{out}\n]}}");
            Ok(0)
        }
        "verify" => {
            let bad = fetch::verify(need("--v2")?, &units)?;
            let rows: Vec<String> = bad.iter().map(|&i| {
                let u = &units[i];
                format!("{{\"file\": \"{}\", \"offset\": {}, \"nbytes\": {}, \"layer\": {}, \"expert\": {}, \"half\": {}, \"proj\": {}}}",
                        fetch::width_file(u.layer, u.k), u.offset, u.nbytes, u.layer, u.expert, u.half, u.proj)
            }).collect();
            println!("{{\"checked\": {}, \"bad\": [{}]}}", units.len(), rows.join(",\n"));
            Ok(if bad.is_empty() { 0 } else { 3 })
        }
        _ => Err(format!("unknown command {cmd}: fetch-plan, verify or dflash2")),
    }
}
