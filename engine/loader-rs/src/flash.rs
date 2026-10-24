//! GLM-5.3-Flash's tensor map onto the shared release planner: the release's dense and vocabulary files hold the same
//! exlr8 records as GLM-5.3's (FORMAT.md), under Flash's source names, and load through the same Planner and loader.
use super::*;
use crate::json;

const KEYS: [&str; 7] = ["model", "hidden_size", "moe_intermediate_size", "n_routed_experts",
    "num_hidden_layers", "first_moe_layer", "tensor_prefix"];

fn meta_json(f: &StFile, key: &str) -> Result<Json, String> {
    json::parse(f.meta.get(key).and_then(Json::str).ok_or(format!("{}: no {key}", f.path))?)
}
fn dtype(d: &str) -> Result<&'static str, String> {
    match d { "BF16" => Ok("bf16"), "F32" => Ok("f32"), "F16" => Ok("f16"), _ => Err(format!("unsupported dtype {d}")) }
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
    quant_layer(p, l, release_file(dir, Some(l), model)?)
}

pub(super) fn globals(p: &mut Planner, dir: &str, model: &Json) -> Result<(), String> {
    quant_globals(p, release_file(dir, None, model)?)
}

fn release_file(dir: &str, l: Option<usize>, model: &Json) -> Result<Arc<StFile>, String> {
    let f = Arc::new(StFile::open(&exlr8::dense_path(dir, l))?);
    identity(&f, model)?;
    let layer = f.meta.get("layer").and_then(Json::str).and_then(|v| v.parse::<usize>().ok());
    if layer != l { return Err(format!("{}: layer {layer:?}, want {l:?}", f.path)); }
    Ok(f)
}

/// A BF16 / FP32 tensor as stored (a record's payload, or a plain tensor), plus a BF16 decode record (.gv) for a
/// 2-D BF16 matrix in 16 x 16 tiles.
fn copied(p: &mut Planner, f: &Arc<StFile>, local: &str, label: String, shape: &[usize]) -> Result<(), String> {
    let tensor = if f.info(&format!("{local}.record")).is_ok() {
        let m = Matrix::open(f, local)?;
        if ![exlr8::BF16, exlr8::FP32].contains(&m.rec.fmt) { return Err(format!("{local}: expected a BF16 or FP32 record")); }
        if shape.len() != 2 || (m.rec.out, m.rec.inp) != (shape[0], shape[1]) { return Err(format!("{local}: record/source shape mismatch")); }
        format!("{local}.{}", exlr8::data_tensors(m.rec.fmt)[0])
    } else {
        local.to_string()
    };
    let info = f.info(&tensor)?.clone();
    if info.shape.iter().product::<usize>() != shape.iter().product::<usize>() { return Err(format!("{local}: payload/source shape mismatch")); }
    let dt = dtype(&info.dtype)?;
    p.copy(label.clone(), dt, shape.iter().map(|&n| n as i64).collect(), f, &tensor, 0)?;
    if shape.len() == 2 && dt == "bf16" && shape[0] % 16 == 0 && shape[1] % 16 == 0 {
        let (f, k, n) = (f.clone(), shape[1], info.end - info.start);
        let gv = format!("{label}.gv");
        p.gen(gv.clone(), vec![(gv, "u8", vec![(shape[0] / 16) as i64, (k / 16) as i64, 512])], move || {
            let b = f.read(&tensor, 0, n)?;
            let v: Vec<u16> = b.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
            Ok(vec![records::bf16_record(&v, k)])
        });
    }
    Ok(())
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

/// KDA matrices the engine reads in BF16 only (kda/kda.cuh bind_kda): q/k/v and b as BF16 decode records (.gv), f_b /
/// g_b as the BF16 matrix itself. A quantized release stores them in MXFP8, whose values BF16 holds exactly (a 3-bit
/// mantissa and a power-of-two scale), so they are decoded at load, as kv_b's FP8 blocks are.
const KDA_BF16: [&str; 6] = ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
    "self_attn.b_proj.weight", "self_attn.f_b_proj.weight", "self_attn.g_b_proj.weight"];

fn kda_bf16(p: &mut Planner, l: usize, label: String, f: &Arc<StFile>, local: &str, name: &str) -> Result<(), String> {
    let m = Matrix::open(f, local)?;
    let (o, k, threads) = (m.rec.out, m.rec.inp, p.threads);
    let decode = {
        let (f, local) = (f.clone(), local.to_string());
        move || -> Result<Vec<u16>, String> {
            let m = Matrix::open(&f, &local)?;
            Ok(exlr8::decode_par(&m, 0..o, 0..k, threads)?.into_iter().map(exlr8::bf16_bits).collect())
        }
    };
    p.note(l, Form { what: "KDA projection", src: fmt_name(&m), form: "BF16", exact: BF16_OF });
    if name.ends_with("f_b_proj.weight") || name.ends_with("g_b_proj.weight") {
        p.gen(label.clone(), vec![(label, "bf16", vec![o as i64, k as i64])], move || {
            Ok(vec![decode()?.iter().flat_map(|v| v.to_le_bytes()).collect()])
        });
    } else {
        if o % 16 != 0 || k % 16 != 0 { return Err(format!("{local}: {o}x{k} is not in 16-row tiles")); }
        if !name.ends_with("b_proj.weight") {
            // q / k / v: also this rank's heads as MXFP8 as stored, for the prefill GEMM (flash_prefill.cuh pf_kda_proj_gemm)
            let (h0, nh) = p.chunk(Group::Heads(l))?;
            let rows = h0 * 128..(h0 + nh) * 128;
            let (n, f2, local2) = (rows.len() as i64, f.clone(), local.to_string());
            p.gen(format!("{label}.mx.codes"), vec![(format!("{label}.mx.codes"), "u8", vec![n, k as i64]),
                                                   (format!("{label}.mx.scales"), "u8", vec![n, (k / 32) as i64])], move || {
                let (c, s) = Matrix::open(&f2, &local2)?.mxfp8(&rows)?;
                Ok(vec![c, s])
            });
        }
        let gv = format!("{label}.gv");
        p.gen(gv.clone(), vec![(gv, "u8", vec![(o / 16) as i64, (k / 16) as i64, 512])], move || {
            Ok(vec![records::bf16_record(&decode()?, k)])
        });
    }
    Ok(())
}

fn quant_layer(p: &mut Planner, l: usize, f: Arc<StFile>) -> Result<(), String> {
    let prefix = format!("model.language_model.layers.{l}.");
    let (h0, nh) = p.chunk(Group::Heads(l))?;
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
                    "dense.gate" | "dense.up" | "dense.down" => {
                        let (d0, nd) = p.chunk(Group::Dense(l))?;
                        (d0 * 128..(d0 + nd) * 128, if name == "dense.down" { exlr8::SIDE_IN } else { exlr8::SIDE_OUT }, 128)
                    },
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
                p.trellis(l, name, &format!("L{l:03}"), &m, &[ch])?;
            },
            MXFP8 if KDA_BF16.contains(&name) => kda_bf16(p, l, label, &f, &local, name)?,
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
            let (t0, nt) = p.chunk(Group::Vocab)?;
            let rows = t0 * 128..(t0 + nt) * 128;
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
