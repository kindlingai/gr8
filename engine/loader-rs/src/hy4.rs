//! Tencent Hy4-flash (`HYV4FlashForCausalLM`): plans a rank's device image from our assembled exlr8 v2 release
//! (`dense/exlr8-{dense,vocab}.safetensors`, `experts/exlr8-layer-LLL.k{2,3,4}.safetensors`, `index-slices/`,
//! width manifest `manifests/exlr8-manifest-*.json`).
//!
//! Tensor-parallel split over `nr in {1, 2, 4}` ranks (rank r):
//!   DSA attention (12 layers: 0,1,5,...,41): 32 heads -> 32/nr heads per rank (q_b, kv_b, linear_gate, sinks, o_proj)
//!   SWA attention (30 layers):               64 heads -> 64/nr heads per rank
//!   Dense MLP (layer 0):                     I=13824 -> 13824/nr per rank
//!   MoE routed (layers 1..41):               512 experts, latent hidden=1536, 12 x 128-col pmcgx blocks via v2plan::v2_layer
//!   MoE shared (layers 1..41):               1 expert, hidden=3072, 12 x 128-col pmcgx blocks via v2plan::v2_shared_layer
//!   Engram tables (layers 1, 11, 21, 31):    stay on NVMe (engram/exlr8-engram-Lnnn.bin); small projections/norms live here
//!   lm_head:                                 944 tiles of 128 split evenly across ranks

use crate::exlr8::StFile;
use crate::plan::{Gen, Op, Plan};
use crate::v2plan::{self, Builder, Slices, V2Image};
use std::ops::Range;
use std::sync::Arc;

const ALIGN: u64 = 4096;
const H: usize = 3072;
const EH: usize = 1536;
const L_MAIN: usize = 42;
const NH: usize = 12;
const NEXP: usize = 512;
const HEADS_DSA: usize = 32;
const HEADS_SWA: usize = 64;
const QK_HEAD: usize = 256;
const KVB_HEAD: usize = 448;
const V_HEAD: usize = 256;
const QA: usize = 1024;
const KV_LORA: usize = 512;
const DENSE_I: usize = 13824;
const VOCAB: usize = 120832;
const VTILES: usize = VOCAB / 128;

pub fn is_dsa(l: usize) -> bool {
    l == 0 || (l >= 1 && l < L_MAIN && (l - 1) % 4 == 0)
}

pub fn is_engram(l: usize) -> bool {
    matches!(l, 1 | 11 | 21 | 31)
}

pub fn layer_heads(l: usize) -> usize {
    if is_dsa(l) { HEADS_DSA } else { HEADS_SWA }
}

pub fn check_nr(nr: usize, rank: usize) -> Result<(), String> {
    if !matches!(nr, 1 | 2 | 4) || rank >= nr {
        return Err(format!("hy4: bad TP {nr} rank {rank} (want nr in {{1,2,4}}, rank < nr)"));
    }
    Ok(())
}

pub fn attn_range(l: usize, nr: usize, r: usize) -> Range<usize> {
    let nh = layer_heads(l) / nr;
    (r * nh)..((r + 1) * nh)
}

pub fn dense_range(nr: usize, r: usize) -> Range<usize> {
    let n = DENSE_I / nr;
    (r * n)..((r + 1) * n)
}

pub fn vocab_range(nr: usize, r: usize) -> Range<usize> {
    let base = VTILES / nr;
    let rem = VTILES % nr;
    let t0 = r * base + r.min(rem);
    let nt = base + usize::from(r < rem);
    (t0 * 128)..((t0 + nt) * 128)
}

fn find_file(dir: &str, sub: &str, name: &str) -> Result<String, String> {
    for p in [format!("{dir}/{sub}/{name}"), format!("{dir}/{name}")] {
        if std::path::Path::new(&p).exists() {
            return Ok(p);
        }
    }
    Err(format!("hy4: cannot find {sub}/{name} under {dir}"))
}

fn copy_whole(b: &mut Builder, dst_name: String, dtype: &str, f: &StFile, fi: usize, src_name: &str) -> Result<(), String> {
    let info = f.info(src_name)?;
    let n = info.end - info.start;
    let shape: Vec<i64> = info.shape.iter().map(|&d| d as i64).collect();
    let off = b.begin(dst_name, dtype, shape, n, None);
    if n > 0 {
        b.seg(off, n, Op::Copy, fi, info.start);
    }
    Ok(())
}

fn copy_stack(b: &mut Builder, dst_name: String, dtype: &str, dst_shape: Vec<i64>, f: &StFile, fi: usize, src_names: &[String]) -> Result<(), String> {
    let mut total = 0u64;
    let mut spans = Vec::with_capacity(src_names.len());
    for sn in src_names {
        let info = f.info(sn)?;
        let n = info.end - info.start;
        spans.push((info.start, n));
        total += n;
    }
    let off = b.begin(dst_name, dtype, dst_shape, total, None);
    let mut cur = off;
    for (s, n) in spans {
        if n > 0 {
            b.seg(cur, n, Op::Copy, fi, s);
            cur += n;
        }
    }
    Ok(())
}

fn copy_row_range(
    b: &mut Builder,
    dst_name: String,
    dtype: &str,
    dst_shape: Vec<i64>,
    f: &StFile,
    fi: usize,
    src_name: &str,
    row_bytes: u64,
    run: Range<usize>,
) -> Result<(), String> {
    let info = f.info(src_name)?;
    let len = run.len() as u64 * row_bytes;
    let src_off = info.start + run.start as u64 * row_bytes;
    if src_off + len > info.end {
        return Err(format!("{src_name}: row range {run:?} exceeds tensor size"));
    }
    let off = b.begin(dst_name, dtype, dst_shape, len, None);
    if len > 0 {
        b.seg(off, len, Op::Copy, fi, src_off);
    }
    Ok(())
}

fn copy_cols_bf16(
    b: &mut Builder,
    dst_name: String,
    f: &Arc<StFile>,
    src_name: &str,
    rows: usize,
    cols: usize,
    col_range: Range<usize>,
) -> Result<(), String> {
    let info = f.info(src_name)?;
    let dst_cols = col_range.len();
    let nbytes = (rows * dst_cols * 2) as u64;
    let dst_label = dst_name.clone();
    let off = b.begin(dst_name, "bf16", vec![rows as i64, dst_cols as i64], nbytes, None);
    if col_range.start == 0 && dst_cols == cols {
        let fi = b.file(f.path.clone());
        b.seg(off, nbytes, Op::Copy, fi, info.start);
        return Ok(());
    }
    let f = f.clone();
    let start = info.start;
    let len = info.end - info.start;
    let c0 = col_range.start;
    let gi = b.gens.len();
    b.gens.push(Gen::new(
        dst_label,
        nbytes,
        Box::new(move || {
            let raw = f.read_at(start, len)?;
            let mut out = vec![0u8; rows * dst_cols * 2];
            for r in 0..rows {
                let s = (r * cols + c0) * 2;
                let d = r * dst_cols * 2;
                out[d..d + dst_cols * 2].copy_from_slice(&raw[s..s + dst_cols * 2]);
            }
            Ok(out)
        }),
    ));
    b.seg(off, nbytes, Op::Gen, gi, 0);
    Ok(())
}

const DFLY_LAYERS: usize = 5;
const DFLY_I: usize = 9216;
const DFLY_HEADS: usize = 32;

/// Plans one rank's Hy4-flash device image from an assembled exlr8 v2 release.
/// `part`:
///   0 = full text image (permanent + quantizable BF16 dense in one slab, backward compatible)
///   1 = permanent tensors only (embed, lm_head, norms, hc, router, idx_f32, sinks, engram norms/biases, routed+shared pmcgx)
///   2 = quantizable BF16 dense linears only (freed right after GPU MXFP8 quantization!)
///   3 = optional DFly2 0.85B drafter (dense/exlr8-dfly.safetensors, TP-sharded across `nr`)
pub fn plan_part(
    release_dir: &str,
    widths_path: &str,
    nr: usize,
    rank: usize,
    part: u32,
) -> Result<V2Image, String> {
    check_nr(nr, rank)?;
    if part == 3 {
        return plan_dfly(release_dir, nr, rank);
    }
    let inc_perm = part == 0 || part == 1;
    let inc_dense = part == 0 || part == 2;
    if !inc_perm && !inc_dense {
        return Err(format!("hy4: bad plan part {part} (want 0..3)"));
    }

    let man = if inc_perm { Some(v2plan::width_manifest(widths_path)?) } else { None };
    let dense_path = find_file(release_dir, "dense", "exlr8-dense.safetensors")?;
    let vocab_path = find_file(release_dir, "dense", "exlr8-vocab.safetensors")?;
    let dense = Arc::new(StFile::open(&dense_path)?);
    let vocab = Arc::new(StFile::open(&vocab_path)?);

    let slices = if inc_perm { Some(Slices::open(release_dir, nr, &[rank])?) } else { None };
    let wfile = |l: usize, k: u8| v2plan::width_file(release_dir, l, k);

    let mut b = Builder::new();
    let fi_dense = b.file(dense_path.clone());
    let fi_vocab = b.file(vocab_path.clone());

    if inc_perm {
        copy_whole(&mut b, "embed.weight".into(), "bf16", &vocab, fi_vocab, "model.embed_tokens.weight")?;
        let vr = vocab_range(nr, rank);
        copy_row_range(&mut b, "lm_head.weight".into(), "bf16", vec![vr.len() as i64, H as i64],
                       &vocab, fi_vocab, "lm_head.weight", (H * 2) as u64, vr)?;
        copy_whole(&mut b, "norm.weight".into(), "bf16", &vocab, fi_vocab, "model.norm.weight")?;
        copy_whole(&mut b, "hc_head.fn".into(), "f32", &vocab, fi_vocab, "model.hc_head.hc_head_fn")?;
        copy_whole(&mut b, "hc_head.base".into(), "f32", &vocab, fi_vocab, "model.hc_head.hc_head_base")?;
        copy_whole(&mut b, "hc_head.scale".into(), "f32", &vocab, fi_vocab, "model.hc_head.hc_head_scale")?;
    }

    for l in 0..L_MAIN {
        let lp = format!("L{l:03}");
        let sp = format!("model.layers.{l}.");

        // Layer norms & non-combined iHC (FP32)
        if inc_perm {
            copy_whole(&mut b, format!("{lp}.input_norm"), "bf16", &dense, fi_dense, &format!("{sp}input_layernorm.weight"))?;
            copy_whole(&mut b, format!("{lp}.post_attn_norm"), "bf16", &dense, fi_dense, &format!("{sp}post_attention_layernorm.weight"))?;
            for (dst, src) in [("attn_hc", "hc_attn_layer.hc_pre"), ("mlp_hc", "hc_mlp_layer.hc_pre")] {
                copy_whole(&mut b, format!("{lp}.{dst}.fn"), "f32", &dense, fi_dense, &format!("{sp}{src}.hc_fn"))?;
                copy_whole(&mut b, format!("{lp}.{dst}.base"), "f32", &dense, fi_dense, &format!("{sp}{src}.hc_base"))?;
                copy_whole(&mut b, format!("{lp}.{dst}.scale"), "f32", &dense, fi_dense, &format!("{sp}{src}.hc_scale"))?;
            }
        }

        // Engram small tensors (layers 1, 11, 21, 31)
        if is_engram(l) {
            let ep = format!("{sp}engram_layer.");
            if inc_perm {
                copy_whole(&mut b, format!("{lp}.eng.lookup_table"), "i32", &dense, fi_dense, &format!("{ep}lookup_table"))?;
                let hnorms: Vec<String> = (0..16).map(|h| format!("{ep}lookup_per_head_norms.{h}.weight")).collect();
                copy_stack(&mut b, format!("{lp}.eng.head_norms"), "bf16", vec![16, 96], &dense, fi_dense, &hnorms)?;
                copy_whole(&mut b, format!("{lp}.eng.val_b"), "bf16", &dense, fi_dense, &format!("{ep}value_proj.bias"))?;
                let kb: Vec<String> = (0..4).map(|c| format!("{ep}key_projs.{c}.bias")).collect();
                let nk: Vec<String> = (0..4).map(|c| format!("{ep}norm_key.{c}.weight")).collect();
                let nq: Vec<String> = (0..4).map(|c| format!("{ep}norm_query.{c}.weight")).collect();
                let hs: Vec<String> = (0..4).map(|c| format!("{ep}unified_per_hc_scales.{c}")).collect();
                copy_stack(&mut b, format!("{lp}.eng.key_b"), "bf16", vec![4, H as i64], &dense, fi_dense, &kb)?;
                copy_stack(&mut b, format!("{lp}.eng.norm_k"), "bf16", vec![4, H as i64], &dense, fi_dense, &nk)?;
                copy_stack(&mut b, format!("{lp}.eng.norm_q"), "bf16", vec![4, H as i64], &dense, fi_dense, &nq)?;
                copy_whole(&mut b, format!("{lp}.eng.pre_gamma"), "bf16", &dense, fi_dense, &format!("{ep}unified_pre_gamma"))?;
                copy_stack(&mut b, format!("{lp}.eng.hc_scales"), "bf16", vec![4, H as i64], &dense, fi_dense, &hs)?;
            }
            if inc_dense {
                copy_whole(&mut b, format!("{lp}.eng.val_w"), "bf16", &dense, fi_dense, &format!("{ep}value_proj.weight"))?;
                let kw: Vec<String> = (0..4).map(|c| format!("{ep}key_projs.{c}.weight")).collect();
                copy_stack(&mut b, format!("{lp}.eng.key_w"), "bf16", vec![4, H as i64, EH as i64], &dense, fi_dense, &kw)?;
            }
        }

        // Gated MLA attention (dual shape: 32 heads DSA / 64 heads SWA)
        let hr = attn_range(l, nr, rank);
        let nh_loc = hr.len();
        let nh_tot = layer_heads(l);
        if inc_perm {
            copy_whole(&mut b, format!("{lp}.attn.q_a_norm"), "bf16", &dense, fi_dense, &format!("{sp}self_attn.q_a_layernorm.weight"))?;
            copy_whole(&mut b, format!("{lp}.attn.kv_a_norm"), "bf16", &dense, fi_dense, &format!("{sp}self_attn.kv_a_layernorm.weight"))?;
            copy_whole(&mut b, format!("{lp}.attn.post_proj_norm"), "bf16", &dense, fi_dense, &format!("{sp}self_attn.post_proj_norm.weight"))?;
            copy_row_range(&mut b, format!("{lp}.attn.sinks"), "f32", vec![nh_loc as i64],
                           &dense, fi_dense, &format!("{sp}self_attn.learnable_sink_param"), 4, hr.clone())?;
        }
        if inc_dense {
            copy_whole(&mut b, format!("{lp}.attn.q_a_proj"), "bf16", &dense, fi_dense, &format!("{sp}self_attn.q_a_proj.weight"))?;
            copy_row_range(&mut b, format!("{lp}.attn.q_b_proj"), "bf16", vec![(nh_loc * QK_HEAD) as i64, QA as i64],
                           &dense, fi_dense, &format!("{sp}self_attn.q_b_proj.weight"), (QA * 2) as u64,
                           (hr.start * QK_HEAD)..(hr.end * QK_HEAD))?;
            copy_whole(&mut b, format!("{lp}.attn.kv_a_proj"), "bf16", &dense, fi_dense, &format!("{sp}self_attn.kv_a_proj_with_mqa.weight"))?;
            copy_row_range(&mut b, format!("{lp}.attn.kv_b_proj"), "bf16", vec![(nh_loc * KVB_HEAD) as i64, KV_LORA as i64],
                           &dense, fi_dense, &format!("{sp}self_attn.kv_b_proj.weight"), (KV_LORA * 2) as u64,
                           (hr.start * KVB_HEAD)..(hr.end * KVB_HEAD))?;
            copy_row_range(&mut b, format!("{lp}.attn.linear_gate"), "bf16", vec![(nh_loc * V_HEAD) as i64, H as i64],
                           &dense, fi_dense, &format!("{sp}self_attn.linear_gate.weight"), (H * 2) as u64,
                           (hr.start * V_HEAD)..(hr.end * V_HEAD))?;
            copy_cols_bf16(&mut b, format!("{lp}.attn.o_proj"), &dense, &format!("{sp}self_attn.o_proj.weight"),
                           H, nh_tot * V_HEAD, (hr.start * V_HEAD)..(hr.end * V_HEAD))?;
        }

        // Compressed indexer (DSA layers only)
        if is_dsa(l) {
            let ip = format!("{sp}self_attn.compressed_indexer.");
            if inc_perm {
                copy_whole(&mut b, format!("{lp}.idx.head_w"), "bf16", &dense, fi_dense, &format!("{ip}head_weight_projection.weight"))?;
                copy_whole(&mut b, format!("{lp}.idx.tok_k_norm"), "bf16", &dense, fi_dense, &format!("{ip}token_key_norm.weight"))?;
                copy_whole(&mut b, format!("{lp}.idx.k_val_proj"), "f32", &dense, fi_dense, &format!("{ip}key_value_projection.weight"))?;
                copy_whole(&mut b, format!("{lp}.idx.k_gate_proj"), "f32", &dense, fi_dense, &format!("{ip}key_gate_projection.weight"))?;
                copy_whole(&mut b, format!("{lp}.idx.k_norm"), "f32", &dense, fi_dense, &format!("{ip}key_norm.weight"))?;
                copy_whole(&mut b, format!("{lp}.idx.ape"), "f32", &dense, fi_dense, &format!("{ip}intra_group_ape"))?;
            }
            if inc_dense {
                copy_whole(&mut b, format!("{lp}.idx.q_proj"), "bf16", &dense, fi_dense, &format!("{ip}query_projection.weight"))?;
                copy_whole(&mut b, format!("{lp}.idx.tok_k_proj"), "bf16", &dense, fi_dense, &format!("{ip}token_key_projection.weight"))?;
            }
        }

        // MLP (layer 0 dense, layers 1..41 latent MoE)
        if l == 0 {
            let dr = dense_range(nr, rank);
            if inc_perm {
                copy_whole(&mut b, format!("{lp}.mlp.post_fc2_norm"), "bf16", &dense, fi_dense, &format!("{sp}mlp.post_fc2_norm.weight"))?;
            }
            if inc_dense {
                copy_row_range(&mut b, format!("{lp}.mlp.gate_proj"), "bf16", vec![dr.len() as i64, H as i64],
                               &dense, fi_dense, &format!("{sp}mlp.gate_proj.weight"), (H * 2) as u64, dr.clone())?;
                copy_row_range(&mut b, format!("{lp}.mlp.up_proj"), "bf16", vec![dr.len() as i64, H as i64],
                               &dense, fi_dense, &format!("{sp}mlp.up_proj.weight"), (H * 2) as u64, dr.clone())?;
                copy_cols_bf16(&mut b, format!("{lp}.mlp.down_proj"), &dense, &format!("{sp}mlp.down_proj.weight"), H, DENSE_I, dr)?;
            }
        } else {
            if inc_perm {
                copy_whole(&mut b, format!("{lp}.router.w"), "bf16", &dense, fi_dense, &format!("{sp}mlp.gate.weight"))?;
                copy_whole(&mut b, format!("{lp}.router.bias"), "f32", &dense, fi_dense, &format!("{sp}mlp.gate.e_score_correction_bias"))?;
                copy_whole(&mut b, format!("{lp}.moe.combine_norm"), "bf16", &dense, fi_dense, &format!("{sp}mlp.combine_output_norm.weight"))?;

                let w_full = v2plan::widths_nh(man.as_ref().unwrap(), l, NH)?
                    .ok_or_else(|| format!("hy4: manifest has no widths for layer {l}"))?;
                let w_routed = v2plan::Widths {
                    gate: w_full.gate.clone(),
                    up: w_full.up.clone(),
                    down: w_full.down.clone(),
                    sgate: [0; v2plan::NH_MAX],
                    sup: [0; v2plan::NH_MAX],
                    sdown: 0,
                    nh: NH,
                };
                let sl = slices.as_ref().unwrap();
                let own_r = |e: usize| Ok(v2plan::formula_halves_nh(e, l, nr, rank, NH));
                v2plan::v2_layer(&mut b, l, &w_routed, sl, &wfile, &own_r)?;

                let own_s = v2plan::formula_halves_nh(NEXP, l, nr, rank, NH);
                v2plan::v2_shared_layer(&mut b, l, &w_full, NEXP, H, sl, &wfile, &own_s)?;
            }
            if inc_dense {
                copy_whole(&mut b, format!("{lp}.moe.down_proj"), "bf16", &dense, fi_dense, &format!("{sp}mlp.routed_expert_down_proj.weight"))?;
                copy_whole(&mut b, format!("{lp}.moe.up_proj"), "bf16", &dense, fi_dense, &format!("{sp}mlp.routed_expert_up_proj.weight"))?;
            }
        }
    }

    let bytes = b.cur.div_ceil(ALIGN) * ALIGN;
    let plan = Plan::from_segs(b.files, b.segs, b.mems, b.gens, bytes)?;
    Ok(V2Image { rank, tensors: b.tensors, bytes, plan, cache: None })
}

fn plan_dfly(release_dir: &str, nr: usize, rank: usize) -> Result<V2Image, String> {
    let dfly_path = find_file(release_dir, "dense", "exlr8-dfly.safetensors")?;
    let dfly = Arc::new(StFile::open(&dfly_path)?);
    let mut b = Builder::new();
    let fi = b.file(dfly_path);
    let p = "model.dfly_layers.";

    copy_whole(&mut b, "dfly.mtp.enorm".into(), "bf16", &dfly, fi, &format!("{p}mtp_fusion.enorm.weight"))?;
    copy_whole(&mut b, "dfly.mtp.hnorm".into(), "bf16", &dfly, fi, &format!("{p}mtp_fusion.hnorm.weight"))?;
    copy_whole(&mut b, "dfly.mtp.eh_proj".into(), "bf16", &dfly, fi, &format!("{p}mtp_fusion.eh_proj.weight"))?;
    copy_whole(&mut b, "dfly.mtp.gate".into(), "bf16", &dfly, fi, &format!("{p}mtp_fusion.gate"))?;

    copy_whole(&mut b, "dfly.final_norm".into(), "bf16", &dfly, fi, &format!("{p}final_norm.weight"))?;

    copy_whole(&mut b, "dfly.hc.hidden_norm".into(), "bf16", &dfly, fi, &format!("{p}hidden_correction.hidden_norm.weight"))?;
    copy_whole(&mut b, "dfly.hc.embed_norm".into(), "bf16", &dfly, fi, &format!("{p}hidden_correction.embed_norm.weight"))?;
    copy_whole(&mut b, "dfly.hc.gate_proj".into(), "bf16", &dfly, fi, &format!("{p}hidden_correction.gate_proj.weight"))?;
    copy_whole(&mut b, "dfly.hc.up_proj".into(), "bf16", &dfly, fi, &format!("{p}hidden_correction.up_proj.weight"))?;
    copy_whole(&mut b, "dfly.hc.down_proj".into(), "bf16", &dfly, fi, &format!("{p}hidden_correction.down_proj.weight"))?;

    copy_whole(&mut b, "dfly.conf.w".into(), "bf16", &dfly, fi, &format!("{p}confidence_head.proj.weight"))?;
    copy_whole(&mut b, "dfly.conf.b".into(), "bf16", &dfly, fi, &format!("{p}confidence_head.proj.bias"))?;

    let nh_loc = DFLY_HEADS / nr;
    let hr = (rank * nh_loc)..((rank + 1) * nh_loc);
    let i_loc = DFLY_I / nr;
    let ir = (rank * i_loc)..((rank + 1) * i_loc);

    for l in 0..DFLY_LAYERS {
        let lp = format!("dfly.L{l}");
        let sp = format!("{p}{l}.");
        copy_whole(&mut b, format!("{lp}.input_norm"), "bf16", &dfly, fi, &format!("{sp}input_layernorm.weight"))?;
        copy_whole(&mut b, format!("{lp}.post_attn_norm"), "bf16", &dfly, fi, &format!("{sp}post_attention_layernorm.weight"))?;

        copy_whole(&mut b, format!("{lp}.attn.q_a_proj"), "bf16", &dfly, fi, &format!("{sp}self_attn.q_a_proj.weight"))?;
        copy_whole(&mut b, format!("{lp}.attn.q_a_norm"), "bf16", &dfly, fi, &format!("{sp}self_attn.q_a_layernorm.weight"))?;
        copy_row_range(&mut b, format!("{lp}.attn.q_b_proj"), "bf16", vec![(nh_loc * QK_HEAD) as i64, QA as i64],
                       &dfly, fi, &format!("{sp}self_attn.q_b_proj.weight"), (QA * 2) as u64,
                       (hr.start * QK_HEAD)..(hr.end * QK_HEAD))?;
        copy_whole(&mut b, format!("{lp}.attn.kv_a_proj"), "bf16", &dfly, fi, &format!("{sp}self_attn.kv_a_proj_with_mqa.weight"))?;
        copy_whole(&mut b, format!("{lp}.attn.kv_a_norm"), "bf16", &dfly, fi, &format!("{sp}self_attn.kv_a_layernorm.weight"))?;
        copy_row_range(&mut b, format!("{lp}.attn.kv_b_proj"), "bf16", vec![(nh_loc * KVB_HEAD) as i64, KV_LORA as i64],
                       &dfly, fi, &format!("{sp}self_attn.kv_b_proj.weight"), (KV_LORA * 2) as u64,
                       (hr.start * KVB_HEAD)..(hr.end * KVB_HEAD))?;
        copy_row_range(&mut b, format!("{lp}.attn.linear_gate"), "bf16", vec![(nh_loc * V_HEAD) as i64, H as i64],
                       &dfly, fi, &format!("{sp}self_attn.linear_gate.weight"), (H * 2) as u64,
                       (hr.start * V_HEAD)..(hr.end * V_HEAD))?;
        copy_cols_bf16(&mut b, format!("{lp}.attn.o_proj"), &dfly, &format!("{sp}self_attn.o_proj.weight"),
                       H, DFLY_HEADS * V_HEAD, (hr.start * V_HEAD)..(hr.end * V_HEAD))?;
        copy_whole(&mut b, format!("{lp}.attn.post_proj_norm"), "bf16", &dfly, fi, &format!("{sp}self_attn.post_proj_norm.weight"))?;
        copy_row_range(&mut b, format!("{lp}.attn.sinks"), "bf16", vec![nh_loc as i64],
                       &dfly, fi, &format!("{sp}self_attn.learnable_sink_param"), 2, hr.clone())?;

        copy_row_range(&mut b, format!("{lp}.mlp.gate_proj"), "bf16", vec![i_loc as i64, H as i64],
                       &dfly, fi, &format!("{sp}mlp.gate_proj.weight"), (H * 2) as u64, ir.clone())?;
        copy_row_range(&mut b, format!("{lp}.mlp.up_proj"), "bf16", vec![i_loc as i64, H as i64],
                       &dfly, fi, &format!("{sp}mlp.up_proj.weight"), (H * 2) as u64, ir.clone())?;
        copy_cols_bf16(&mut b, format!("{lp}.mlp.down_proj"), &dfly, &format!("{sp}mlp.down_proj.weight"),
                       H, DFLY_I, ir.clone())?;
    }

    let bytes = b.cur.div_ceil(ALIGN) * ALIGN;
    let plan = Plan::from_segs(b.files, b.segs, b.mems, b.gens, bytes)?;
    Ok(V2Image { rank, tensors: b.tensors, bytes, plan, cache: None })
}

pub fn plan(
    release_dir: &str,
    widths_path: &str,
    nr: usize,
    rank: usize,
) -> Result<V2Image, String> {
    plan_part(release_dir, widths_path, nr, rank, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::read_range;

    #[test]
    fn ranges_cover_full_geometry_tp1_tp2_tp4() {
        for nr in [1usize, 2, 4] {
            let mut voc = 0;
            let mut den = 0;
            for r in 0..nr {
                let vr = vocab_range(nr, r);
                assert_eq!(vr.start, voc);
                assert_eq!(vr.len() % 128, 0);
                voc += vr.len();

                let dr = dense_range(nr, r);
                assert_eq!(dr.start, den);
                den += dr.len();
            }
            assert_eq!(voc, VOCAB);
            assert_eq!(den, DENSE_I);
            for l in 0..L_MAIN {
                let mut h = 0;
                for r in 0..nr {
                    let hr = attn_range(l, nr, r);
                    assert_eq!(hr.start, h);
                    h += hr.len();
                }
                assert_eq!(h, layer_heads(l));
            }
        }
    }

    /// Synthetic round-trip test for GEOM-PLAN rev2 §4.2: verifies routed (hidden=1536, nh=12) and
    /// shared (hidden=3072, nh=12) experts coexist in one layer's width/slice files and survive
    /// v2_layer + v2_shared_layer planning & item128->g6 read bit-exactly at TP=1 and TP=2.
    #[test]
    fn shared_and_routed_dual_geometry_round_trip() {
        let dir = std::env::temp_dir().join(format!("hy4-rt-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("experts")).unwrap();
        std::fs::create_dir_all(dir.join("index-slices/tp1/k3")).unwrap();
        std::fs::create_dir_all(dir.join("index-slices/tp1/k4")).unwrap();
        std::fs::create_dir_all(dir.join("index-slices/tp2/k3")).unwrap();
        std::fs::create_dir_all(dir.join("index-slices/tp2/k4")).unwrap();

        // Build minimal width files for layer 1 at K=3 and K=4 with routed [512*12*B_r] (sparse size) and shared [12*B_s]
        // Use small pattern in expert 0 unit 0 and shared unit 0 to verify exact byte read & item128 transpose!
        let mut sh_down_pattern = vec![0u8; 3072 * 16 * 4]; // K4 shared unit: 196608 B
        for (i, b) in sh_down_pattern.iter_mut().enumerate() {
            *b = ((i.wrapping_mul(37) ^ (i >> 5)) & 0xFF) as u8;
        }
        let mut sh_gate_pattern = vec![0u8; 3072 * 16 * 3]; // K3 shared unit: 147456 B
        for (i, b) in sh_gate_pattern.iter_mut().enumerate() {
            *b = ((i.wrapping_mul(13) + 7) & 0xFF) as u8;
        }

        let write_wf = |k: u8| -> ([u64; 3], [u64; 3]) {
            let b_r = 1536u64 * 16 * k as u64;
            let b_s = 3072u64 * 16 * k as u64;
            let nr_tot = 512u64 * 12 * b_r;
            let ns_tot = 12u64 * b_s;
            let mut cur = 0u64;
            let mut roff = [0u64; 3];
            let mut soff = [0u64; 3];
            let mut hdr = format!(
                "{{\"__metadata__\":{{\"format\":\"exlr8-v2\",\"contents\":\"trellis\",\"codebook\":\"pmcgx V2 L20 0xCBAC1FED/0xBCA14C51, E4M3\"}}"
            );
            for (p, nm) in ["gate", "up", "down"].iter().enumerate() {
                roff[p] = 65536 + cur;
                hdr += &format!(",\"trellis.{nm}.pmcgx\":{{\"dtype\":\"U8\",\"shape\":[{nr_tot}],\"data_offsets\":[{cur},{}]}}", cur + nr_tot);
                cur += nr_tot;
            }
            for (p, nm) in ["gate", "up", "down"].iter().enumerate() {
                soff[p] = 65536 + cur;
                hdr += &format!(",\"shared.trellis.{nm}.pmcgx\":{{\"dtype\":\"U8\",\"shape\":[{ns_tot}],\"data_offsets\":[{cur},{}]}}", cur + ns_tot);
                cur += ns_tot;
            }
            hdr += "}";
            let mut hb = hdr.into_bytes();
            hb.resize(65536 - 8, b' ');
            let p = dir.join(format!("experts/exlr8-layer-001.k{k}.safetensors"));
            let f = std::fs::OpenOptions::new().create(true).truncate(true).write(true).open(&p).unwrap();
            f.set_len(65536 + cur).unwrap();
            use std::os::unix::fs::FileExt;
            f.write_all_at(&(hb.len() as u64).to_le_bytes(), 0).unwrap();
            f.write_all_at(&hb, 8).unwrap();
            if k == 3 {
                f.write_all_at(&sh_gate_pattern, soff[0]).unwrap();
            }
            if k == 4 {
                f.write_all_at(&sh_down_pattern, soff[2]).unwrap();
            }
            (roff, soff)
        };
        let (roff3, soff3) = write_wf(3);
        let (roff4, soff4) = write_wf(4);

        // Write TP=1 and TP=2 slices for K=3 and K=4
        let write_slice = |nr: usize, r: usize, k: u8, roff: [u64; 3], soff: [u64; 3]| {
            let b_r = 1536i64 * 16 * k as i64;
            let b_s = 3072i64 * 16 * k as i64;
            let mut runits = Vec::new();
            let mut roffs = Vec::new();
            for e in 0..512usize {
                for u in 0..12usize {
                    if (12 * e + u + 1) % nr == r {
                        runits.extend_from_slice(&(e as i16).to_le_bytes());
                        runits.extend_from_slice(&(u as i16).to_le_bytes());
                        for p in 0..3 {
                            roffs.extend_from_slice(&(roff[p] as i64 + (12 * e as i64 + u as i64) * b_r).to_le_bytes());
                        }
                    }
                }
            }
            let n = runits.len() / 4;
            let rchan = vec![0u8; n * 3 * 256];
            let rblk = vec![0u8; 512 * 3 * 12 * 2];
            let rsigns = vec![1u8; 2 * 1536];

            let mut sunits = Vec::new();
            let mut soffs = Vec::new();
            for u in 0..12usize {
                if (12 * 512 + u + 1) % nr == r {
                    sunits.extend_from_slice(&0i16.to_le_bytes());
                    sunits.extend_from_slice(&(u as i16).to_le_bytes());
                    for p in 0..3 {
                        soffs.extend_from_slice(&(soff[p] as i64 + u as i64 * b_s).to_le_bytes());
                    }
                }
            }
            let ns = sunits.len() / 4;
            let schan = vec![0x3Cu8; ns * 3 * 256];
            let sblk = vec![0x40u8; 3 * 24 * 2];
            let mut ssigns = vec![1u8; 2 * 3072];
            ssigns[3] = 0xFF; // -1 at index 3 to verify sign -> -1.0 f16 (0xBC00)

            let items: Vec<(&str, &str, Vec<usize>, Vec<u8>)> = vec![
                ("L001.offset", "I64", vec![n, 3], roffs),
                ("L001.units", "I16", vec![n, 2], runits),
                ("L001.chan", "F16", vec![n, 3, 128], rchan),
                ("L001.block", "F16", vec![512, 3, 12], rblk),
                ("L001.signs", "I8", vec![2, 1536], rsigns),
                ("L001.shared.offset", "I64", vec![ns, 3], soffs),
                ("L001.shared.units", "I16", vec![ns, 2], sunits),
                ("L001.shared.chan", "F16", vec![ns, 3, 128], schan),
                ("L001.shared.block", "F16", vec![1, 3, 24], sblk),
                ("L001.shared.signs", "I8", vec![2, 3072], ssigns),
            ];
            let mut cur = 0usize;
            let mut hdr = String::from("{\"__metadata__\":{\"format\":\"exlr8-v2-index-slice\"}");
            for (nm, dt, sh, data) in &items {
                let shs: Vec<String> = sh.iter().map(usize::to_string).collect();
                hdr += &format!(",\"{nm}\":{{\"dtype\":\"{dt}\",\"shape\":[{}],\"data_offsets\":[{cur},{}]}}", shs.join(","), cur + data.len());
                cur += data.len();
            }
            hdr += "}";
            while hdr.len() % 8 != 0 { hdr.push(' '); }
            let mut out = (hdr.len() as u64).to_le_bytes().to_vec();
            out.extend_from_slice(hdr.as_bytes());
            for (_, _, _, data) in items { out.extend_from_slice(&data); }
            let p = dir.join(format!("index-slices/tp{nr}/k{k}/rank{r}.safetensors"));
            std::fs::write(p, out).unwrap();
        };
        write_slice(1, 0, 3, roff3, soff3);
        write_slice(1, 0, 4, roff4, soff4);
        for r in 0..2 {
            write_slice(2, r, 3, roff3, soff3);
            write_slice(2, r, 4, roff4, soff4);
        }

        // Plan layer 1 at TP=1 with per-matrix mixed widths: routed gate=3, up=4, down=3; shared gate=3, up=4, down=4!
        let mut w_routed = v2plan::Widths {
            gate: vec![[3; v2plan::NH_MAX]; 512],
            up: vec![[4; v2plan::NH_MAX]; 512],
            down: vec![3; 512],
            sgate: [0; v2plan::NH_MAX],
            sup: [0; v2plan::NH_MAX],
            sdown: 0,
            nh: 12,
        };
        w_routed.gate[0][..12].fill(3);
        let w_shared = v2plan::Widths {
            gate: vec![[3; v2plan::NH_MAX]; 512],
            up: vec![[4; v2plan::NH_MAX]; 512],
            down: vec![3; 512],
            sgate: [3; v2plan::NH_MAX],
            sup: [4; v2plan::NH_MAX],
            sdown: 4,
            nh: 12,
        };

        let slices = Slices::open(dir.to_str().unwrap(), 1, &[0]).unwrap();
        let wfile = |l: usize, k: u8| v2plan::width_file(dir.to_str().unwrap(), l, k);
        let mut b = Builder::new();
        v2plan::v2_layer(&mut b, 1, &w_routed, &slices, &wfile, &|e| Ok(v2plan::formula_halves_nh(e, 1, 1, 0, 12))).unwrap();
        let own_s = v2plan::formula_halves_nh(512, 1, 1, 0, 12);
        v2plan::v2_shared_layer(&mut b, 1, &w_shared, 512, 3072, &slices, &wfile, &own_s).unwrap();

        // Check shapes: routed sg=[2,1536], v1b=[6144,2,12]; shared sg=[2,3072], v1b=[12,2,24]
        let find = |name: &str| b.tensors.iter().find(|t| t.name == name).unwrap().clone();
        assert_eq!(find("L001.tr3.sg").shape, vec![2, 1536]);
        assert_eq!(find("L001.tr3.v1b").shape, vec![6144, 2, 12]);
        assert_eq!(find("L001.sh_tr3.sg").shape, vec![2, 3072]);
        assert_eq!(find("L001.sh_tr3.v1b").shape, vec![12, 2, 24]);
        assert_eq!(find("L001.sh_tr3.v3b").shape, vec![12, 24]);

        let bytes = b.cur.div_ceil(ALIGN) * ALIGN;
        let plan = Plan::from_segs(b.files, b.segs, b.mems, b.gens, bytes).unwrap();

        // Verify shared gate unit 0 read back bit-exactly
        let sh_fc1 = find("L001.sh_tr3.fc1");
        let mut got_g = vec![0u8; sh_gate_pattern.len()];
        read_range(&plan, sh_fc1.offset, &mut got_g).unwrap();
        assert_eq!(got_g, sh_gate_pattern);

        // Verify shared down unit 0 underwent item128_to_g6_geometry with hidden=3072 (24 groups, NOT 6144 or 1536!)
        let sh_fc2 = find("L001.sh_tr3.fc2");
        let mut got_d = vec![0u8; sh_down_pattern.len()];
        read_range(&plan, sh_fc2.offset, &mut got_d).unwrap();
        let mut want_d = vec![0u8; sh_down_pattern.len()];
        crate::plan::item128_to_g6_geometry(&sh_down_pattern, &mut want_d, 3072);
        assert_eq!(got_d, want_d);

        // Verify shared sign[3] = -1 -> 0xBC00 f16
        let sh_sg = find("L001.sh_tr3.sg");
        let mut got_sg = vec![0u8; 16];
        read_range(&plan, sh_sg.offset, &mut got_sg).unwrap();
        assert_eq!(&got_sg[0..2], &[0x00, 0x3C]); // +1.0
        assert_eq!(&got_sg[6..8], &[0x00, 0xBC]); // -1.0

        // Also check TP=2 rank 0 & rank 1 shared split = 6 blocks each
        for r in 0..2 {
            let sl2 = Slices::open(dir.to_str().unwrap(), 2, &[r]).unwrap();
            let mut b2 = Builder::new();
            v2plan::v2_layer(&mut b2, 1, &w_routed, &sl2, &wfile, &|e| Ok(v2plan::formula_halves_nh(e, 1, 2, r, 12))).unwrap();
            let os2 = v2plan::formula_halves_nh(512, 1, 2, r, 12);
            assert_eq!(os2.len(), 6);
            v2plan::v2_shared_layer(&mut b2, 1, &w_shared, 512, 3072, &sl2, &wfile, &os2).unwrap();
            assert_eq!(b2.tensors.iter().find(|t| t.name == "L001.sh_tr3.v1b").unwrap().shape, vec![6, 2, 24]);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
