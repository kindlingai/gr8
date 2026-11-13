//! The exlr8 v2 fetch helper for ./gr8 glm53 up --model https://... (tools/launch/hfsource.py and
//! tools/launch/fetch_ranks.py keep the HTTP; this plans and checks). JSON on stdout.
//!
//!   g6load fetch-plan --slices DIR --tp N --rank R (--manifest FILE | --widths 2,3,4) [--layers 3-40,78] [--layout FILE]
//!       the byte ranges of each width file rank R loads: {"units", "bytes", "files": [{"name", "ranges": [[off, n]]}]}.
//!       With --layout (bin/g6layout, src/layout.rs) the halves the layout gives rank R, which reads every rank's
//!       index slices at TP N.
//!   g6load verify --slices DIR --v2 DIR --tp N --rank R (--manifest FILE | --widths ...) [--layers ...] [--layout FILE]
//!       each of those halves' xxhash64 against the index slices: {"checked", "bad": [{"file", "offset", "nbytes",
//!       "layer", "expert", "half", "proj"}]}. Exit 0 when every half matches, 3 when one does not, 1 on an error.
//!
//!   g6load compact --slices DIR --v2 DIR --out DIR --tp N --rank R (--manifest FILE | --widths ...)
//!       [--layers ...] [--layout FILE]: write dense expert copies and g6map-v1 maps for this rank, verify every
//!       half through the compact reader, then publish the directory. An existing directory is checked and reused.
//!
//! DIR for --slices holds tp<N>/k<K>/rank<R>.safetensors (the release's index-slices/).
//!
//!   g6load dflash2 --src DIR --out DIR [--nr 4] [--ranks 0,1,2,3] [--bf16 fc,attn,mlp,conv,sel]
//!       converts the DFlash2 drafter (config.json + model.safetensors in --src) into dflash.r<r>.pack and
//!       dflash.r<r>.manifest.json for each rank (src/dflash2.rs). One JSON line
//!       per rank: {"rank", "tensors", "bytes", "seconds"}.
//!
//!   g6load ink2-plan --src DIR --tp N --rank R [--out-manifest FILE] [--no-gv] [--no-mtp] [--no-vision]
//!       plans one rank's image of an Inkling-2-Small checkpoint directory (src/ink2.rs) without loading it. One JSON
//!       line: {"tp", "rank", "tensors", "image_bytes", "read_bytes", "files"}. --out-manifest writes the image
//!       manifest (sha256 fields zero).

use g6load::dflash2;
use g6load::fetch::{self, Choice};
use g6load::layout::Layout;
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

/// `g6load ink2-plan`: its three switches take no value, so it parses its own arguments.
fn ink2_plan(args: &[String]) -> Result<i32, String> {
    let mut opts = g6load::ink2::Opts::default();
    let (mut src, mut tp, mut rank, mut out) = (None, None, None, None);
    let mut it = args.iter();
    while let Some(k) = it.next() {
        let mut val = || it.next().ok_or(format!("{k} needs a value"));
        match k.as_str() {
            "--no-gv" => opts.bf16_gv = false,
            "--no-mtp" => opts.with_mtp = false,
            "--no-vision" => opts.with_vision = false,
            "--src" => src = Some(val()?.clone()),
            "--tp" => tp = Some(val()?.parse::<usize>().map_err(|_| "--tp: not a number")?),
            "--rank" => rank = Some(val()?.parse::<usize>().map_err(|_| "--rank: not a number")?),
            "--out-manifest" => out = Some(val()?.clone()),
            _ => return Err(format!("ink2-plan: unknown option {k}")),
        }
    }
    let (src, tp, rank) = (src.ok_or("ink2-plan: --src is required")?, tp.ok_or("ink2-plan: --tp is required")?,
                           rank.ok_or("ink2-plan: --rank is required")?);
    let p = g6load::ink2::plan(&src, tp, rank, &opts)?;
    if let Some(out) = out {
        let zeros = vec!["0".repeat(64); p.img.tensors.len()];
        std::fs::write(&out, p.manifest_json(&zeros, tp)).map_err(|e| format!("{out}: {e}"))?;
    }
    println!("{{\"tp\": {tp}, \"rank\": {rank}, \"tensors\": {}, \"image_bytes\": {}, \"read_bytes\": {}, \"files\": {}}}",
             p.img.tensors.len(), p.img.bytes, p.read_bytes(), p.reads.len());
    Ok(0)
}

fn run() -> Result<i32, String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let cmd = argv.first().ok_or("usage: g6load fetch-plan|verify|compact|dflash2|ink2-plan (see the source) --slices DIR --tp N --rank R (--manifest F | --widths K,..)")?;
    if cmd == "ink2-plan" {
        return ink2_plan(&argv[1..]);
    }
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
        dflash2::main(need("--src")?, need("--out")?, nr, opt.get("--ranks").copied(),
                      opt.get("--bf16").copied().unwrap_or(""))?;
        return Ok(0);
    }
    let choice = match (opt.get("--manifest"), opt.get("--widths")) {
        (Some(m), None) => Choice::manifest(m)?,
        (None, Some(w)) => Choice::Widths(w.split(',').map(|k| k.parse::<u8>().map_err(|_| format!("--widths {w}")))
            .collect::<Result<_, _>>()?),
        _ => return Err("give one of --manifest and --widths".into()),
    };
    let ls = opt.get("--layers").map(|s| layers(s)).transpose()?;
    let layout = opt.get("--layout").map(|p| Layout::read(p)).transpose()?;
    let units = fetch::units(need("--slices")?, num("--tp")?, num("--rank")?, &choice, ls.as_deref(), layout.as_ref())?;
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
        "compact" => {
            g6load::compact::write(need("--v2")?, need("--out")?, &units)?;
            println!("{{\"checked\": {}, \"bad\": []}}", units.len());
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
        _ => Err(format!("unknown command {cmd}: fetch-plan, verify, compact, dflash2 or ink2-plan")),
    }
}
