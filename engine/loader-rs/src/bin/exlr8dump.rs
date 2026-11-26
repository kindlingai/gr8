//! Decodes one exlr8 v2 dense or vocabulary file with gr8's decoder (src/exlr8.rs) and writes the tensors under the
//! HF names `loader/exlr8.py` gives them, for an exact comparison against it (engine/loader-rs/tests/exlr8_compare.py).
//!
//!   exlr8dump DIR LAYER|GROUP:LAYER|vocab OUT.safetensors [--dtype f32|bf16] [--shard N R] [--threads T] [--verify]
//!
//! DIR holds dense/ (or the dense files themselves). --shard N R writes only rank R's block of each sharded matrix.
//! The block is one of N runs of whole units (record field 5) on the sharded side, the larger runs first. The file's
//! metadata gives each block's range under the tensor's name. Norms and biases are written as stored.

use g6load::exlr8::{self, DenseProfile, Matrix, StFile, SIDE_IN, SIDE_REPLICATED};
use g6load::json::Json;
use std::collections::HashSet;
use std::io::Write;
use std::ops::Range;

fn main() {
    if let Err(e) = run() {
        eprintln!("exlr8dump: {e}");
        std::process::exit(1);
    }
}

/// Rank r's run of `n` units over `nr` ranks, the larger runs first (engine/g6nr.h).
fn run_of(n: usize, nr: usize, r: usize) -> Range<usize> {
    let w = |j: usize| n / nr + usize::from(j < n % nr);
    let a: usize = (0..r).map(w).sum();
    a..a + w(r)
}

enum PlanEntry<'a> {
    Matrix(Matrix<'a>, Range<usize>, Range<usize>),
    Raw(String),
}

fn run() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        return Err(
            "usage: exlr8dump DIR LAYER|GROUP:LAYER|vocab OUT.safetensors [--dtype f32|bf16] [--shard N R] [--threads T] [--verify]"
                .into(),
        );
    }
    let (dir, which, out) = (&a[1], &a[2], &a[3]);
    let (mut bf16, mut shard, mut verify, mut threads) =
        (false, None, false, std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));
    let mut i = 4;
    while i < a.len() {
        match a[i].as_str() {
            "--dtype" => {
                match a.get(i + 1).map(String::as_str) {
                    Some("bf16") => bf16 = true,
                    Some("f32") => bf16 = false,
                    _ => return Err("--dtype must be f32 or bf16".into()),
                }
                i += 2;
            }
            "--shard" => {
                let p = |k: usize| a.get(k).and_then(|v| v.parse::<usize>().ok()).ok_or("--shard N R");
                let (n, r) = (p(i + 1)?, p(i + 2)?);
                if n == 0 || r >= n {
                    return Err(format!("--shard N R requires N > 0 and R < N (got {n} {r})"));
                }
                shard = Some((n, r));
                i += 3;
            }
            "--threads" => {
                threads = a.get(i + 1).and_then(|v| v.parse().ok()).ok_or("--threads T")?;
                i += 2;
            }
            "--verify" => {
                verify = true;
                i += 1;
            }
            x => return Err(format!("unknown option {x}")),
        }
    }
    let (group, layer, is_vocab) = if which == "vocab" {
        (None, None, true)
    } else if let Some((g, l)) = which.split_once(':') {
        exlr8::validate_group_name(g)?;
        (Some(g.to_string()), Some(l.parse::<usize>().map_err(|_| "GROUP:LAYER: layer must be a number")?), false)
    } else {
        (None, Some(which.parse::<usize>().map_err(|_| "LAYER: a number, GROUP:LAYER, or vocab")?), false)
    };
    let path = exlr8::group_dense_path(dir, group.as_deref(), layer)?;
    let f = StFile::open(&path)?;
    exlr8::check_v2_file(&f)?;
    if let Some(contents) = f.meta.get("contents").and_then(Json::str) {
        let want = if is_vocab { "vocab" } else { "dense" };
        if contents != want {
            return Err(format!("{}: metadata contents {contents:?}, expected {want:?}", f.path));
        }
    }
    if let (Some(cli_g), Some(meta_g)) = (group.as_deref(), f.meta.get("group").and_then(Json::str)) {
        if cli_g != meta_g {
            return Err(format!("{}: CLI group {cli_g:?} disagrees with metadata group {meta_g:?}", f.path));
        }
    }
    if let Some(cli_l) = layer {
        let meta_l = match f.meta.get("layer") {
            Some(Json::Str(s)) => {
                Some(s.parse::<usize>().map_err(|_| format!("{}: invalid layer metadata {s:?}", f.path))?)
            }
            Some(Json::Int(v)) if *v >= 0 => Some(*v as usize),
            Some(_) => return Err(format!("{}: invalid layer metadata", f.path)),
            None => None,
        };
        if let Some(ml) = meta_l {
            if cli_l != ml {
                return Err(format!("{}: CLI layer {cli_l} disagrees with metadata layer {ml}", f.path));
            }
        }
    }
    let profile = DenseProfile::from_name(f.meta.get("model_profile").and_then(Json::str))?;
    let eff_group = group.or_else(|| f.meta.get("group").and_then(Json::str).map(str::to_string));
    let hf = |n: &str, is_matrix: bool| -> Result<String, String> {
        exlr8::hf_tensor_name(profile, eff_group.as_deref(), layer, is_vocab, &f.meta, n, is_matrix)
    };
    let mats = Matrix::names(&f);
    // (name, dtype, shape, entry) for the header, then the data in the same order
    let mut plan: Vec<(String, String, Vec<usize>, PlanEntry)> = Vec::new();
    let mut owned: HashSet<String> = HashSet::new();
    let mut meta = Vec::new();
    for m in &mats {
        let mx = Matrix::open(&f, m)?;
        if verify {
            mx.verify_checksums()?;
        }
        owned.insert(format!("{m}.record"));
        owned.insert(format!("{m}.checksum"));
        for s in exlr8::data_tensors(mx.rec.fmt) {
            owned.insert(format!("{m}.{s}"));
        }
        let r = mx.rec;
        let (mut rows, mut cols) = (0..r.out, 0..r.inp);
        let hf_w = hf(m, true)?;
        if let (Some((n, k)), true) = (shard, r.side != SIDE_REPLICATED) {
            let span = if r.side == SIDE_IN { r.inp } else { r.out };
            if r.unit == 0 || !span.is_multiple_of(r.unit) {
                return Err(format!("{}: {m}: cannot shard span {span} with unit {}", f.path, r.unit));
            }
            let u = run_of(span / r.unit, n, k);
            let b = u.start * r.unit..u.end * r.unit;
            if r.side == SIDE_IN { cols = b } else { rows = b }
            meta.push(format!("\"{hf_w}\": \"rows {}:{} cols {}:{}\"", rows.start, rows.end, cols.start, cols.end));
        }
        let shape = vec![rows.len(), cols.len()];
        plan.push((hf_w, if bf16 { "BF16" } else { "F32" }.into(), shape, PlanEntry::Matrix(mx, rows, cols)));
    }
    for n in f.names() {
        if !owned.contains(n) {
            let i = f.info(n)?;
            plan.push((hf(n, false)?, i.dtype.clone(), i.shape.clone(), PlanEntry::Raw(n.clone())));
        }
    }
    let esz = |d: &str| match d { "F32" | "I32" => 4, "BF16" | "F16" | "I16" => 2, "I64" | "U64" => 8, _ => 1 };
    let mut head = String::from("{");
    let mut off = 0usize;
    for (name, dtype, shape, _) in &plan {
        let n = shape.iter().product::<usize>() * esz(dtype);
        let s: Vec<String> = shape.iter().map(usize::to_string).collect();
        head += &format!("\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{}],\"data_offsets\":[{off},{}]}},", s.join(","), off + n);
        off += n;
    }
    head += &format!("\"__metadata__\":{{{}}}}}", meta.join(","));
    while head.len() % 8 != 0 {
        head.push(' ');
    }
    let mut w = std::io::BufWriter::with_capacity(1 << 24, std::fs::File::create(out).map_err(|e| format!("{out}: {e}"))?);
    let io = |e: std::io::Error| format!("{out}: {e}");
    w.write_all(&(head.len() as u64).to_le_bytes()).map_err(io)?;
    w.write_all(head.as_bytes()).map_err(io)?;
    let t0 = std::time::Instant::now();
    for (_, _, _, entry) in &plan {
        match entry {
            PlanEntry::Matrix(mx, rows, cols) => {
                let v = exlr8::decode_par(mx, rows.clone(), cols.clone(), threads)?;
                let bytes: Vec<u8> = if bf16 {
                    v.iter().flat_map(|&x| exlr8::bf16_bits(x).to_le_bytes()).collect()
                } else {
                    v.iter().flat_map(|x| x.to_le_bytes()).collect()
                };
                w.write_all(&bytes).map_err(io)?;
            }
            PlanEntry::Raw(stored) => {
                w.write_all(&f.read_all(stored)?).map_err(io)?;
            }
        }
    }
    w.flush().map_err(io)?;
    eprintln!("exlr8dump: {} tensors from {} in {:.1} s -> {out}", plan.len(), f.path, t0.elapsed().as_secs_f64());
    Ok(())
}
