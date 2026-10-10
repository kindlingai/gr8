//! Decodes one exlr8 v2 dense or vocabulary file with g6's decoder (src/exlr8.rs) and writes the tensors under the
//! HF names `loader/exlr8.py` gives them, for an exact comparison against it (engine/loader-rs/tests/exlr8_compare.py).
//!
//!   exlr8dump DIR LAYER|vocab OUT.safetensors [--dtype f32|bf16] [--shard N R] [--threads T]
//!
//! DIR holds dense/ (or the dense files themselves). --shard N R writes only rank R's block of each sharded matrix.
//! The block is one of N runs of whole units (record field 5) on the sharded side, the larger runs first. The file's
//! metadata gives each block's range under the tensor's name. Norms and biases are written as stored.

use g6load::exlr8::{self, Matrix, StFile, SIDE_IN, SIDE_REPLICATED};
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

fn run() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        return Err("usage: exlr8dump DIR LAYER|vocab OUT.safetensors [--dtype f32|bf16] [--shard N R] [--threads T]".into());
    }
    let (dir, which, out) = (&a[1], &a[2], &a[3]);
    let (mut bf16, mut shard, mut threads) = (false, None, std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));
    let mut i = 4;
    while i < a.len() {
        match a[i].as_str() {
            "--dtype" => { bf16 = a.get(i + 1).map(String::as_str) == Some("bf16"); i += 2; }
            "--shard" => {
                let p = |k: usize| a.get(k).and_then(|v| v.parse::<usize>().ok()).ok_or("--shard N R");
                shard = Some((p(i + 1)?, p(i + 2)?));
                i += 3;
            }
            "--threads" => { threads = a.get(i + 1).and_then(|v| v.parse().ok()).ok_or("--threads T")?; i += 2; }
            x => return Err(format!("unknown option {x}")),
        }
    }
    let layer = if which == "vocab" { None } else { Some(which.parse::<usize>().map_err(|_| "LAYER: a number or vocab")?) };
    let f = StFile::open(&exlr8::dense_path(dir, layer))?;
    let hf = |n: &str| -> String {
        match layer {
            Some(l) => format!("model.layers.{l}.{n}"),
            None => match n {
                "embed_tokens" => "model.embed_tokens".into(),
                "norm.weight" => "model.norm.weight".into(),
                x => x.into(),
            },
        }
    };
    let mats = Matrix::names(&f);
    // (name, dtype, shape, block) for the header, then the data in the same order
    type Block<'a> = Option<(Matrix<'a>, Range<usize>, Range<usize>)>;
    let mut plan: Vec<(String, String, Vec<usize>, Block)> = Vec::new();
    let mut meta = Vec::new();
    for m in &mats {
        let mx = Matrix::open(&f, m)?;
        let r = mx.rec;
        let (mut rows, mut cols) = (0..r.out, 0..r.inp);
        if let (Some((n, k)), true) = (shard, r.side != SIDE_REPLICATED) {
            let span = if r.side == SIDE_IN { r.inp } else { r.out };
            let u = run_of(span / r.unit, n, k);
            let b = u.start * r.unit..u.end * r.unit;
            if r.side == SIDE_IN { cols = b } else { rows = b }
            meta.push(format!("\"{}.weight\": \"rows {}:{} cols {}:{}\"", hf(m), rows.start, rows.end, cols.start, cols.end));
        }
        let shape = vec![rows.len(), cols.len()];
        plan.push((format!("{}.weight", hf(m)), if bf16 { "BF16" } else { "F32" }.into(), shape, Some((mx, rows, cols))));
    }
    let owned: Vec<String> = mats.iter().flat_map(|m| {
        let fmt = Matrix::open(&f, m).map(|x| x.rec.fmt).unwrap_or(0);
        ["record", "checksum"].iter().chain(exlr8::data_tensors(fmt)).map(move |s| format!("{m}.{s}")).collect::<Vec<_>>()
    }).collect();
    for n in f.names() {
        if !owned.contains(n) {
            let i = f.info(n)?;
            plan.push((hf(n), i.dtype.clone(), i.shape.clone(), None));
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
    for (name, _, _, blk) in &plan {
        match blk {
            Some((mx, rows, cols)) => {
                let v = exlr8::decode_par(mx, rows.clone(), cols.clone(), threads)?;
                let bytes: Vec<u8> = if bf16 {
                    v.iter().flat_map(|&x| exlr8::bf16_bits(x).to_le_bytes()).collect()
                } else {
                    v.iter().flat_map(|x| x.to_le_bytes()).collect()
                };
                w.write_all(&bytes).map_err(io)?;
            }
            None => {
                let stored = f.names().iter().find(|n| hf(n) == *name).ok_or(format!("lost {name}"))?;
                w.write_all(&f.read_all(stored)?).map_err(io)?;
            }
        }
    }
    w.flush().map_err(io)?;
    eprintln!("exlr8dump: {} tensors from {} in {:.1} s -> {out}", plan.len(), f.path, t0.elapsed().as_secs_f64());
    Ok(())
}
