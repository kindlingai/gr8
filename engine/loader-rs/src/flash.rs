//! Lossless Flash BF16/FP32 release path. Source names/shapes survive; split rows are validated before planning.
//! BF16 GEMV records are exact re-layouts, never a float round trip. Unknown tensors remain under their HF names.
use super::*;
use crate::json;

const KEYS: [&str; 7] = ["model", "hidden_size", "moe_intermediate_size", "n_routed_experts",
    "num_hidden_layers", "first_moe_layer", "tensor_prefix"];

#[derive(Clone)]
struct Part { file: Arc<StFile>, tensor: String, lo: usize, hi: usize }
#[derive(Clone)]
struct Source { shape: Vec<usize>, dtype: &'static str, parts: Vec<Part>, count: usize }

fn meta_json(f: &StFile, key: &str) -> Result<Json, String> {
    json::parse(f.meta.get(key).and_then(Json::str).ok_or(format!("{}: no {key}", f.path))?)
}
fn dtype(d: &str) -> Result<&'static str, String> {
    match d { "BF16" => Ok("bf16"), "F32" => Ok("f32"), "F16" => Ok("f16"), _ => Err(format!("unsupported lossless dtype {d}")) }
}
/// The release's model metadata (FORMAT.md "Model metadata"), from its vocabulary file: None for GLM-5.3, whose files
/// leave the keys out. Every dense file of the release must carry the same values (identity below).
pub(super) fn model_meta(dir: &str) -> Result<Option<Json>, String> {
    let f = StFile::open(&exlr8::dense_path(dir, None))?;
    if f.meta.get("hidden_size").is_none() { return Ok(None); }
    let mut m = std::collections::BTreeMap::new();
    for k in KEYS {
        let v = f.meta.get(k).and_then(Json::str).ok_or(format!("{}: no {k}", f.path))?;
        m.insert(k.to_string(), Json::Str(v.to_string()));
    }
    Ok(Some(Json::Obj(m)))
}
fn identity(f: &StFile, model: &Json) -> Result<(), String> {
    for k in KEYS {
        let a = f.meta.get(k).and_then(Json::str).ok_or(format!("{}: no {k}", f.path))?;
        let b = model.get(k).ok_or(format!("release model metadata: no {k}"))?;
        if b.str().map(str::to_string).or_else(|| b.int().map(|v| v.to_string())).as_deref() != Some(a) {
            return Err(format!("{}: model identity mismatch: {k}", f.path));
        }
    }
    Ok(())
}

/// group = layer number, or None for vocabulary. Extras are preserved separately by the release, not executed.
fn sources(dir: &str, group: Option<usize>, model: &Json) -> Result<BTreeMap<String, Source>, String> {
    let nested = std::path::Path::new(dir).join("dense");
    let root = if nested.is_dir() { nested } else { dir.into() };
    let mut paths = std::fs::read_dir(&root).map_err(|e| format!("{}: {e}", root.display()))?
        .map(|e| e.map(|x| x.path()).map_err(|e| e.to_string())).collect::<Result<Vec<_>, _>>()?;
    paths.sort();
    let regular = group.map(|l| format!("exlr8-dense-{l:03}.safetensors")).unwrap_or("exlr8-vocab.safetensors".into());
    let mut out: BTreeMap<String, Source> = BTreeMap::new();
    for path in paths {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name != regular && !(name.starts_with("exlr8-split-") && name.ends_with(".safetensors")) { continue; }
        let f = Arc::new(StFile::open(path.to_str().ok_or("non-UTF8 path")?)?);
        let layer = f.meta.get("layer").and_then(Json::str).and_then(|s| s.parse().ok());
        if layer != group { continue; }
        let names = meta_json(&f, "source_names")?;
        let names = names.obj().ok_or("source_names is not an object")?;
        if group.is_none() && !names.keys().all(|n| ["embed_tokens", "lm_head", "norm.weight"].contains(&n.as_str())) { continue; }
        identity(&f, model)?;
        let shapes = meta_json(&f, "source_shapes")?;
        let rows = f.meta.get("rows").and_then(Json::str);
        let count = if rows.is_some() { f.meta.get("parts").and_then(Json::str).and_then(|s| s.parse().ok()).ok_or("invalid split parts")? } else { 1 };
        for (local, full) in names {
            let full = full.str().ok_or("bad source name")?.to_string();
            let rec = format!("{local}.record");
            let (shape, tensor) = if f.info(&rec).is_ok() {
                let r = f.read_all(&rec)?;
                if r.len() != 64 { return Err(format!("{local}: bad record")); }
                let r: Vec<i64> = r.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap())).collect();
                if ![exlr8::BF16, exlr8::FP32].contains(&r[0]) || r[3..].iter().any(|&v| v != 0) { return Err(format!("{local}: expected lossless BF16/FP32")); }
                let shape = shapes.get(local).and_then(Json::arr).ok_or("missing source shape")?.iter()
                    .map(|v| v.int().filter(|&n| n > 0).map(|n| n as usize).ok_or("invalid shape"))
                    .collect::<Result<Vec<_>, _>>()?;
                if shape.len() < 2 || r[1] != shape[0] as i64 || r[2] != shape[1..].iter().product::<usize>() as i64 { return Err(format!("{local}: record/source shape mismatch")); }
                let tensor = format!("{local}.{}", if r[0] == exlr8::BF16 { "bf16" } else { "fp32" });
                if f.info(&tensor)?.dtype != if r[0] == exlr8::BF16 { "BF16" } else { "F32" } {
                    return Err(format!("{local}: record/payload dtype mismatch"));
                }
                (shape, tensor)
            } else { (f.info(local)?.shape.clone(), local.clone()) };
            let i = f.info(&tensor)?;
            let dt = dtype(&i.dtype)?;
            let total = shape.first().copied().unwrap_or(1);
            let (lo, hi) = match rows {
                Some(s) => {
                    let (a, b) = s.split_once('-').ok_or("invalid split rows")?;
                    (a.parse::<usize>().map_err(|_| "invalid split start")?, b.parse::<usize>().map_err(|_| "invalid split end")?)
                }
                None => (0, total),
            };
            if lo >= hi || hi > total || (rows.is_some() && (lo % 128 != 0 || (hi != total && hi % 128 != 0))) { return Err(format!("{local}: invalid split rows")); }
            let rowbytes = shape.iter().skip(1).product::<usize>() * if dt == "f32" { 4 } else { 2 };
            if i.end - i.start != ((hi - lo) * rowbytes) as u64 { return Err(format!("{local}: payload size mismatch")); }
            let s = out.entry(full.clone()).or_insert_with(|| Source { shape: shape.clone(), dtype: dt, parts: Vec::new(), count });
            if s.shape != shape || s.dtype != dt || s.count != count { return Err(format!("{full}: inconsistent split metadata")); }
            s.parts.push(Part { file: f.clone(), tensor, lo, hi });
        }
    }
    if out.is_empty() { return Err(format!("no Flash dense files for {group:?}")); }
    for (name, s) in &mut out {
        s.parts.sort_by_key(|p| p.lo);
        let mut end = 0;
        for p in &s.parts {
            if p.lo != end { return Err(format!("{name}: split gaps or overlaps")); }
            end = p.hi;
        }
        if s.parts.len() != s.count || end != s.shape.first().copied().unwrap_or(1) { return Err(format!("{name}: incomplete split tensor")); }
    }
    Ok(out)
}

impl Source {
    fn cols(&self) -> usize { self.shape.iter().skip(1).product() }
    fn elem(&self) -> usize { if self.dtype == "f32" { 4 } else { 2 } }
    fn read(&self, rows: Range<usize>, cols: Range<usize>) -> Result<Vec<u8>, String> {
        if rows.end > self.shape[0] || cols.end > self.cols() { return Err("source slice out of bounds".into()); }
        let mut out = Vec::with_capacity(rows.len() * cols.len() * self.elem());
        for p in &self.parts {
            let lo = rows.start.max(p.lo); let hi = rows.end.min(p.hi);
            if hi <= lo { continue; }
            if cols.start == 0 && cols.end == self.cols() {
                out.extend(p.file.read(&p.tensor, ((lo-p.lo)*self.cols()*self.elem()) as u64, ((hi-lo)*self.cols()*self.elem()) as u64)?);
            } else {
                for row in lo..hi {
                    out.extend(p.file.read(&p.tensor, (((row-p.lo)*self.cols()+cols.start)*self.elem()) as u64, (cols.len()*self.elem()) as u64)?);
                }
            }
        }
        Ok(out)
    }
}

fn emit(p: &mut Planner, name: String, s: &Source, rows: Range<usize>, cols: Range<usize>, shape: Vec<i64>, gv: bool) -> Result<(), String> {
    let fullcols = cols.start == 0 && cols.end == s.cols();
    if fullcols {
        let n = nbytes(s.dtype, &shape);
        let off = p.b.begin(name.clone(), s.dtype, shape.clone(), n, None);
        let mut written = 0;
        for part in &s.parts {
            let lo = rows.start.max(part.lo); let hi = rows.end.min(part.hi);
            if hi <= lo { continue; }
            let i = part.file.info(&part.tensor)?;
            let n = ((hi-lo)*s.cols()*s.elem()) as u64;
            let f = p.b.file(part.file.path.clone());
            p.b.seg(off + written, n, Op::Copy, f, i.start + ((lo-part.lo)*s.cols()*s.elem()) as u64);
            written += n;
        }
        if written != n { return Err(format!("{name}: copied size mismatch")); }
    } else {
        let (s, r, c) = (s.clone(), rows.clone(), cols.clone());
        p.gen(name.clone(), vec![(name.clone(), s.dtype, shape)], move || Ok(vec![s.read(r.clone(), c.clone())?]));
    }
    if gv && s.dtype == "bf16" && rows.len() % 16 == 0 && cols.len() % 16 == 0 {
        let (s, n, k) = (s.clone(), rows.len(), cols.len());
        let label = format!("{name}.gv");
        p.gen(label.clone(), vec![(label, "u8", vec![(n/16) as i64, (k/16) as i64, 512])], move || {
            let b = s.read(rows.clone(), cols.clone())?;
            let v = b.chunks_exact(2).map(|b| u16::from_le_bytes([b[0],b[1]])).collect::<Vec<_>>();
            Ok(vec![records::bf16_record(&v, k)])
        });
    }
    Ok(())
}

fn alias(local: &str) -> &str {
    match local {
        "input_layernorm.weight" => "attn_norm", "post_attention_layernorm.weight" => "mlp_norm",
        "self_attn.q_a_proj.weight" => "q_a", "self_attn.q_a_layernorm.weight" => "q_a_norm",
        "self_attn.q_b_proj.weight" => "q_b", "self_attn.kv_a_proj_with_mqa.weight" => "kv_a",
        "self_attn.kv_a_layernorm.weight" => "kv_a_norm", "self_attn.o_proj.weight" => "o_proj",
        "self_attn.indexer.wq_b.weight" => "idx.wq_b", "self_attn.indexer.wk.weight" => "idx.wk",
        "self_attn.indexer.weights_proj.weight" => "idx.weights_proj",
        "self_attn.indexer.k_norm.weight" => "idx.k_norm.w", "self_attn.indexer.k_norm.bias" => "idx.k_norm.b",
        "mlp.gate.weight" => "router.w", "mlp.gate.e_score_correction_bias" => "router.bias",
        "mlp.gate_proj.weight" => "dense.gate", "mlp.up_proj.weight" => "dense.up", "mlp.down_proj.weight" => "dense.down",
        "eh_proj.weight" => "eh", "enorm.weight" => "enorm", "hnorm.weight" => "hnorm", "shared_head.norm.weight" => "shared_head_norm",
        _ => local,
    }
}

pub(super) fn layer(p: &mut Planner, l: usize, dir: &str, model: &Json) -> Result<(), String> {
    if let Some(f) = quant_file(dir, Some(l), model)? { return quant_layer(p, l, f); }
    let all = sources(dir, Some(l), model)?;
    let prefix = format!("model.language_model.layers.{l}.");
    let (h0, nh) = p.shard.heads(l);
    for (full, s) in all {
        let local = full.strip_prefix(&prefix).ok_or(format!("layer {l}: unexpected source name {full}"))?;
        let name = alias(local);
        let (mut rows, mut cols) = (0..s.shape.first().copied().unwrap_or(1), 0..s.cols());
        let mut shape: Vec<i64> = s.shape.iter().map(|&n| n as i64).collect();
        if local == "self_attn.kv_b_proj.weight" {
            if s.shape != [64*512, 512] { return Err("Flash kv_b must be [32768,512]".into()); }
            for (part, offset) in [("uk",0), ("uv",256)] {
                let label = format!("L{l:03}.kv_b.{part}");
                let src = s.clone();
                p.gen(label.clone(), vec![(label, s.dtype, vec![nh as i64,256,512])], move || {
                    let mut bytes = Vec::new();
                    for h in h0..h0+nh { bytes.extend(src.read(h*512+offset..h*512+offset+256,0..512)?); }
                    Ok(vec![bytes])
                });
            }
            continue;
        }
        match name {
            "q_b" => rows = h0*256..(h0+nh)*256,
            "o_proj" => { let head = s.cols()/64; cols = h0*head..(h0+nh)*head; },
            "dense.gate" | "dense.up" => rows = p.shard.dense(l),
            "dense.down" => cols = p.shard.dense(l),
            _ => (),
        }
        if shape.len() == 2 { shape = vec![rows.len() as i64, cols.len() as i64]; }
        emit(p, format!("L{l:03}.{name}"), &s, rows, cols, shape, s.shape.len()==2)?;
    }
    Ok(())
}

pub(super) fn globals(p: &mut Planner, dir: &str, model: &Json) -> Result<(), String> {
    if let Some(f) = quant_file(dir, None, model)? { return quant_globals(p, f); }
    let all = sources(dir, None, model)?;
    let mut seen = std::collections::BTreeSet::new();
    for (full, s) in all {
        let name = match full.as_str() {
            "model.language_model.embed_tokens.weight" => "G.embed",
            "lm_head.weight" | "model.language_model.lm_head.weight" => "G.lm_head",
            "model.language_model.norm.weight" => "G.final_norm",
            _ => return Err(format!("unknown Flash vocabulary tensor {full}")),
        };
        seen.insert(name);
        let rows = if name == "G.lm_head" { p.shard.vocab() } else { 0..s.shape[0] };
        let mut shape: Vec<i64> = s.shape.iter().map(|&n| n as i64).collect();
        shape[0] = rows.len() as i64;
        emit(p, name.into(), &s, rows, 0..s.cols(), shape, name == "G.lm_head")?;
    }
    if seen.len() != 3 { return Err("incomplete Flash vocabulary".into()); }
    Ok(())
}

// Quantized records use the existing exlr8 decoder and record builders. The
// separate dispatch preserves the lossless split-file validation above.
fn quant_file(dir: &str, l: Option<usize>, model: &Json) -> Result<Option<Arc<StFile>>, String> {
    let path = exlr8::dense_path(dir, l);
    if !std::path::Path::new(&path).exists() { return Ok(None); }
    let f = Arc::new(StFile::open(&path)?);
    if f.meta.get("dense_model").and_then(Json::str) != Some("flash") && !quantized(&f)? { return Ok(None); }
    identity(&f, model)?;
    let layer = f.meta.get("layer").and_then(Json::str).and_then(|v| v.parse::<usize>().ok());
    if layer != l { return Err(format!("{path}: quantized layer mismatch")); }
    if f.meta.get("rows").is_some() { return Err("quantized Flash split records are unsupported".into()); }
    Ok(Some(f))
}

/// A dense file holds quantized records when any matrix record names a format other than BF16 / FP32 (the lossless
/// release's two).
fn quantized(f: &StFile) -> Result<bool, String> {
    for name in f.names().iter().filter(|n| n.ends_with(".record")) {
        let r = f.read_all(name)?;
        if r.len() != 64 { return Err(format!("{}: {name}: bad record", f.path)); }
        let fmt = i64::from_le_bytes(r[..8].try_into().unwrap());
        if fmt != exlr8::BF16 && fmt != exlr8::FP32 { return Ok(true); }
    }
    Ok(false)
}

fn copied(p: &mut Planner, f: &Arc<StFile>, local: &str, label: String, shape: &[usize]) -> Result<(), String> {
    let record = format!("{local}.record");
    let tensor = if f.info(&record).is_err() {
        if f.info(local)?.shape != shape { return Err(format!("{local}: source shape mismatch")); }
        if shape.len() != 2 { return p.vector(label, f, local); }
        local.to_string()
    } else {
        let m = Matrix::open(f, local)?;
        if ![exlr8::BF16, exlr8::FP32].contains(&m.rec.fmt) { return Err(format!("{local}: expected copied matrix")); }
        let t = format!("{local}.{}", exlr8::data_tensors(m.rec.fmt)[0]);
        let info = f.info(&t)?;
        let expected_dtype = if m.rec.fmt == exlr8::BF16 { "BF16" } else { "F32" };
        if info.dtype != expected_dtype || info.shape != shape {
            return Err(format!("{local}: copied payload dtype/shape mismatch"));
        }
        t
    };
    let info = f.info(&tensor)?;
    let dt = dtype(&info.dtype)?;
    let s = Source { shape: shape.to_vec(), dtype: dt, count: 1,
        parts: vec![Part { file: f.clone(), tensor, lo: 0, hi: shape[0] }] };
    emit(p, label, &s, 0..shape[0], 0..s.cols(), shape.iter().map(|&x| x as i64).collect(), shape.len() == 2)
}

fn quant_entries(f: &StFile) -> Result<Vec<(String, String, Vec<usize>)>, String> {
    let names = meta_json(f, "source_names")?;
    let shapes = meta_json(f, "source_shapes")?;
    let mut out = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    let mut owned = std::collections::BTreeSet::new();
    for (local, full) in names.obj().ok_or("source_names is not an object")? {
        let full = full.str().ok_or("invalid source name")?;
        if !seen.insert(full) { return Err(format!("duplicate source {full}")); }
        let shape = shapes.get(local).and_then(Json::arr).ok_or("missing source shape")?.iter()
            .map(|x| x.int().filter(|&n| n>0).map(|n| n as usize).ok_or("invalid source shape"))
            .collect::<Result<Vec<_>, _>>()?;
        if shape.is_empty() { return Err(format!("{local}: empty shape")); }
        if f.info(&format!("{local}.record")).is_ok() {
            let info = f.info(&format!("{local}.record"))?;
            if info.dtype != "I64" || info.shape != [8] { return Err(format!("{local}: invalid record dtype/shape")); }
            let m = Matrix::open(f, local)?;
            for suffix in ["record", "checksum"].iter().chain(exlr8::data_tensors(m.rec.fmt)) {
                owned.insert(format!("{local}.{suffix}"));
            }
            if shape.len()!=2 || (m.rec.out, m.rec.inp)!=(shape[0],shape[1]) {
                return Err(format!("{local}: record/source shape mismatch"));
            }
        } else {
            owned.insert(local.clone());
        }
        out.push((local.clone(), full.into(), shape));
    }
    for n in f.names() {
        if !owned.contains(n) { return Err(format!("{n}: tensor absent from source metadata")); }
    }
    Ok(out)
}

fn quant_layer(p: &mut Planner, l: usize, f: Arc<StFile>) -> Result<(), String> {
    let prefix = format!("model.language_model.layers.{l}.");
    let (h0, nh) = p.shard.heads(l);
    for (local, full, shape) in quant_entries(&f)? {
        let hf = full.strip_prefix(&prefix).ok_or(format!("unexpected source {full}"))?;
        let name = alias(hf);
        let label = format!("L{l:03}.{name}");
        if f.info(&format!("{local}.record")).is_err() {
            copied(p, &f, &local, label, &shape)?;
            continue;
        }
        let m = Matrix::open(&f, &local)?;
        match m.rec.fmt {
            TRELLIS => {
                if !p.trellis { return Err("Flash quantized dense requires --trellis".into()); }
                let (ch, side, unit) = match name {
                    "q_b" => (h0*256..(h0+nh)*256, exlr8::SIDE_OUT, 256),
                    "self_attn.q_proj.weight" | "self_attn.k_proj.weight" | "self_attn.v_proj.weight" =>
                        (h0*128..(h0+nh)*128, exlr8::SIDE_OUT, 128),
                    "o_proj" => {
                        let head = m.rec.inp/64;
                        if ![128,256].contains(&head) || m.rec.inp%64!=0 { return Err("invalid Flash o_proj head geometry".into()); }
                        (h0*head..(h0+nh)*head, exlr8::SIDE_IN, head)
                    },
                    "dense.gate" | "dense.up" => (p.shard.dense(l), exlr8::SIDE_OUT, 128),
                    "dense.down" => (p.shard.dense(l), exlr8::SIDE_IN, 128),
                    _ => return Err(format!("{local}: unsupported Flash trellis matrix")),
                };
                let rotation = match name {
                    "q_b" => 2, "dense.gate" | "dense.up" => 3,
                    "self_attn.q_proj.weight" | "self_attn.k_proj.weight" | "self_attn.v_proj.weight" => 1,
                    _ => 0,
                };
                if m.rec.k != 4 || m.rec.side != side || m.rec.unit != unit || m.rec.rotation != rotation {
                    return Err(format!("{local}: invalid Flash trellis K/side/unit/rotation"));
                }
                p.trellis(l, name, &format!("L{l:03}"), &m, ch)?;
            },
            MXFP8 => p.mxfp8(l, "Flash MXFP8", &label, &f, &local, false)?,
            FP8_BLOCK if hf == "self_attn.kv_b_proj.weight" => {
                if shape != [64*512,512] { return Err("Flash kv_b must be [32768,512]".into()); }
                let src = f.clone();
                p.note(l, Form { what: "kv_b", src: fmt_name(&m), form: "BF16 (UK, UV)", exact: BF16_OF });
                p.gen(label, vec![(format!("L{l:03}.kv_b.uk"), "bf16", vec![nh as i64,256,512]),
                                  (format!("L{l:03}.kv_b.uv"), "bf16", vec![nh as i64,256,512])], move || {
                    let (uk, uv) = Matrix::open(&src, &local)?.fp8_block_kv_b_bf16_le(h0, nh, 512, 256)?;
                    Ok(vec![uk, uv])
                });
            },
            exlr8::BF16 | exlr8::FP32 => copied(p, &f, &local, label, &shape)?,
            _ => return Err(format!("{local}: unsupported Flash dense format {}",m.rec.fmt)),
        }
    }
    Ok(())
}

fn quant_globals(p: &mut Planner, f: Arc<StFile>) -> Result<(), String> {
    let entries = quant_entries(&f)?;
    if entries.len()!=3 { return Err("incomplete Flash quantized vocabulary".into()); }
    let mut seen = std::collections::BTreeSet::new();
    for (local, full, shape) in entries {
        let name = match full.as_str() {
            "model.language_model.embed_tokens.weight" => "G.embed",
            "lm_head.weight" | "model.language_model.lm_head.weight" => "G.lm_head",
            "model.language_model.norm.weight" => "G.final_norm",
            _ => return Err(format!("unknown vocabulary source {full}")),
        };
        if !seen.insert(name) { return Err(format!("duplicate vocabulary tensor {name}")); }
        if name == "G.final_norm" { copied(p, &f, &local, name.into(), &shape)?; continue; }
        let m = Matrix::open(&f, &local)?;
        if m.rec.fmt != exlr8::FP8_ROW || shape != [VOCAB,4096] { return Err(format!("{local}: expected Flash FP8-row vocabulary")); }
        let src = f.clone();
        if name == "G.embed" {
            let mut stripes: Vec<(u64, Box<dyn Fn() -> Result<Vec<u8>, String> + Send + Sync>)> = Vec::new();
            for a in (0..VOCAB).step_by(4096) {
                let r = a..(a + 4096).min(VOCAB);
                let span = (r.len() * 4096 * 2) as u64;
                let (s2, loc) = (src.clone(), local.clone());
                stripes.push((span, Box::new(move || Matrix::open(&s2, &loc)?.fp8_row_bf16_le(&r))));
            }
            p.gen_striped(name, name.into(), "bf16", vec![VOCAB as i64, 4096], stripes);
        } else {
            let rows = p.shard.vocab();
            let n = rows.len();
            p.copy("G.lm_head.scale".into(), "f32", vec![n as i64], &f, &format!("{local}.scale"), (rows.start * 4) as u64)?;
            let mut stripes: Vec<(u64, Box<dyn Fn() -> Result<Vec<u8>, String> + Send + Sync>)> = Vec::new();
            for a in (rows.start..rows.end).step_by(8192) {
                let r = a..(a + 8192).min(rows.end);
                let span = (r.len() * 4096) as u64;
                let (s2, loc) = (src.clone(), local.clone());
                stripes.push((span, Box::new(move || {
                    let c = Matrix::open(&s2, &loc)?.fp8_row_codes(&r)?;
                    Ok(records::fp8_record(&c, 4096))
                })));
            }
            p.gen_striped(name, "G.lm_head.r8.gv".into(), "u8", vec![(n / 16) as i64, 128, 512], stripes);
        }
    }
    Ok(())
}
