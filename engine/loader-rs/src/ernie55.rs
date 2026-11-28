//! ERNIE-5.5-Flash-Preview (`Ernie5_5_MoeForCausalLM`): plans a rank's device (or split VRAM / host) image
//! from the public exlr8 v2/v4 release (`dense/exlr8-vocab.safetensors`, `dense/exlr8-dense-{000..042}.safetensors`,
//! `experts/exlr8-layer-{001..042}.k{2,3,4}.safetensors`, `index-slices/`, `manifests/exlr8-manifest-*.json`).
//!
//! Geometry:
//!   - 43 layers (L0 dense SiTU MLP I=14336; L1..L42 Latent MoE: 512 routed 2048x2048 top-10 + 1 shared 4096x2048)
//!   - 37 HCA C128 layers + 6 sparse MLA-DSA layers (L7, 14, 21, 28, 35, 42)
//!   - 4-stream Manifold Hyper-Connections (mHC, Sinkhorn-20, F32 weights) + mhc_contract head
//!   - Vocab 201216 = 1572 tiles of 128 + multimax LM head parameters
//!
//! `plan_part(..., part)` modes:
//!   0 = full resident image (all tensors in one slab: single-Spark TP=1 or TP=2/4 BF16 dense)
//!   1 = permanent resident tensors + routed/shared experts + embed (excludes quantizable BF16 dense linears)
//!   2 = quantizable BF16 dense linears only (freed after GPU MXFP8 quantization)
//!   3 = hybrid VRAM permanent core (excludes dense linears, `embed.weight`, and routed `tr3.{fc1,fc2}`; keeps
//!       shared experts + routed metadata `sg/v1b/v2/v3b/table/kmix` + norms/mHC/router/sinks/`lm_head` in VRAM)
//!   4 = hybrid pageable host backing (`embed.weight` + routed `L001..L042.tr3.{fc1,fc2}` only)

use crate::exlr8::StFile;
use crate::plan::{Gen, Op, Plan};
use crate::v2plan::{self, Builder, RoutedEmit, Slices, V2Image};
use std::ops::Range;
use std::sync::Arc;

const ALIGN: u64 = 4096;
const H: usize = 4096;
const L_MAIN: usize = 43;
const NH: usize = 16;
const NEXP: usize = 512;
const HEADS: usize = 64;
const HCA_HD: usize = 512;
const HCA_QA: usize = 1024;
const HCA_OGROUPS: usize = 8;
const HCA_HPG: usize = 8;
const HCA_OLORA: usize = 1024;
const DSA_QA: usize = 1024;
const DSA_KV_LORA: usize = 512;
const DSA_QK_HEAD: usize = 256;
const DSA_KVB_HEAD: usize = 448;
const DSA_V_HEAD: usize = 256;
const DENSE_I: usize = 14336;
const VOCAB: usize = 201216;
const VTILES: usize = VOCAB / 128;

pub fn is_dsa(l: usize) -> bool {
    matches!(l, 7 | 14 | 21 | 28 | 35 | 42)
}

pub fn check_nr(nr: usize, rank: usize) -> Result<(), String> {
    if !matches!(nr, 1 | 2 | 4) || rank >= nr {
        return Err(format!("ernie55: bad TP {nr} rank {rank} (want nr in {{1,2,4}}, rank < nr)"));
    }
    Ok(())
}

pub fn head_range(nr: usize, r: usize) -> Range<usize> {
    let nh = HEADS / nr;
    (r * nh)..((r + 1) * nh)
}

pub fn hca_group_range(nr: usize, r: usize) -> Range<usize> {
    let ng = HCA_OGROUPS / nr;
    (r * ng)..((r + 1) * ng)
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
    Err(format!("ernie55: cannot find {sub}/{name} under {dir}"))
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

/// Concatenates the three F32 scalars (`alpha_pre`, `alpha_post`, `alpha_res`) into one `[3]` F32 tensor.
fn pack_hc_scales(b: &mut Builder, dst_name: String, f: &StFile, sp: &str, prefix: &str) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(12);
    for suf in ["alpha_pre", "alpha_post", "alpha_res"] {
        let nm = format!("{sp}{prefix}.{suf}");
        let info = f.info(&nm)?;
        let raw = f.read_at(info.start, info.end - info.start)?;
        if raw.len() != 4 {
            return Err(format!("{nm}: expected 4-byte F32 scalar, got {}", raw.len()));
        }
        bytes.extend_from_slice(&raw);
    }
    let off = b.begin(dst_name, "f32", vec![3], 12, None);
    let mi = b.mems.len();
    b.mems.push(bytes);
    b.segs.push(crate::plan::Seg { pack_off: off, nbytes: 12, op: Op::Mem, file_off: 0, file: mi });
    Ok(())
}

/// Plans one rank's ERNIE 5.5 Flash image (or split VRAM / host part) from an assembled exlr8 release.
pub fn plan_part(
    release_dir: &str,
    widths_path: &str,
    nr: usize,
    rank: usize,
    part: u32,
) -> Result<V2Image, String> {
    check_nr(nr, rank)?;
    if part > 4 {
        return Err(format!("ernie55: bad plan part {part} (want 0..4)"));
    }
    let inc_perm = matches!(part, 0 | 1 | 3);
    let inc_dense = matches!(part, 0 | 2);
    let inc_embed = matches!(part, 0 | 1 | 4);
    let inc_shared = matches!(part, 0 | 1 | 3);
    let routed_emit = match part {
        0 | 1 => Some(RoutedEmit::All),
        3 => Some(RoutedEmit::MetaOnly),
        4 => Some(RoutedEmit::TrellisOnly),
        _ => None,
    };

    let need_experts = inc_shared || routed_emit.is_some();
    let man = if need_experts { Some(v2plan::width_manifest(widths_path)?) } else { None };
    let slices = if need_experts { Some(Slices::open(release_dir, nr, &[rank])?) } else { None };
    let wfile = |l: usize, k: u8| v2plan::width_file(release_dir, l, k);

    let mut b = Builder::new();

    if inc_perm || inc_embed {
        let vocab_path = find_file(release_dir, "dense", "exlr8-vocab.safetensors")?;
        let vocab = Arc::new(StFile::open(&vocab_path)?);
        let fi_vocab = b.file(vocab_path);
        if inc_embed {
            copy_whole(&mut b, "embed.weight".into(), "bf16", &vocab, fi_vocab, "model.embed_tokens.weight")?;
        }
        if inc_perm {
            let vr = vocab_range(nr, rank);
            copy_row_range(
                &mut b,
                "lm_head.weight".into(),
                "bf16",
                vec![vr.len() as i64, H as i64],
                &vocab,
                fi_vocab,
                "lm_head.weight",
                (H * 2) as u64,
                vr,
            )?;
            copy_whole(&mut b, "lm_head.multimax_ranges".into(), "bf16", &vocab, fi_vocab, "model.lm_head.multimax_ranges")?;
            copy_whole(&mut b, "lm_head.multimax_ts".into(), "bf16", &vocab, fi_vocab, "model.lm_head.multimax_ts")?;
            copy_whole(&mut b, "norm.weight".into(), "bf16", &vocab, fi_vocab, "model.norm.weight")?;
            copy_whole(&mut b, "hc_head.fn".into(), "f32", &vocab, fi_vocab, "model.mhc_contract.hc_head_fn")?;
            copy_whole(&mut b, "hc_head.base".into(), "f32", &vocab, fi_vocab, "model.mhc_contract.hc_head_base")?;
            copy_whole(&mut b, "hc_head.scale".into(), "f32", &vocab, fi_vocab, "model.mhc_contract.hc_head_scale")?;
        }
    }

    let hr = head_range(nr, rank);
    let nh_loc = hr.len();
    let gr = hca_group_range(nr, rank);
    let ng_loc = gr.len();

    for l in 0..L_MAIN {
        let lp = format!("L{l:03}");
        let sp = format!("model.layers.{l}.");

        let need_dense_file = inc_perm || inc_dense;
        if need_dense_file {
            let dpath = find_file(release_dir, "dense", &format!("exlr8-dense-{l:03}.safetensors"))?;
            let dense = Arc::new(StFile::open(&dpath)?);
            let fi_dense = b.file(dpath);

            if inc_perm {
                copy_whole(&mut b, format!("{lp}.input_norm"), "bf16", &dense, fi_dense, &format!("{sp}input_layernorm.weight"))?;
                copy_whole(&mut b, format!("{lp}.post_attn_norm"), "bf16", &dense, fi_dense, &format!("{sp}post_attention_layernorm.weight"))?;

                for (dst, src) in [("attn_hc", "self_attention_hyper_connection"), ("mlp_hc", "mlp_hyper_connection")] {
                    copy_whole(&mut b, format!("{lp}.{dst}.fn"), "f32", &dense, fi_dense, &format!("{sp}{src}.mapping_proj.weight"))?;
                    copy_whole(&mut b, format!("{lp}.{dst}.bias"), "f32", &dense, fi_dense, &format!("{sp}{src}.bias"))?;
                    pack_hc_scales(&mut b, format!("{lp}.{dst}.scale"), &dense, &sp, src)?;
                }

                // VHA postmix U/V are small [64, 16] BF16 matrices replicated on all ranks (applied across full heads or TP-gathered)
                copy_whole(&mut b, format!("{lp}.attn.vha_u"), "bf16", &dense, fi_dense, &format!("{sp}self_attn.vha_postmix_U"))?;
                copy_whole(&mut b, format!("{lp}.attn.vha_v"), "bf16", &dense, fi_dense, &format!("{sp}self_attn.vha_postmix_V"))?;
            }

            if is_dsa(l) {
                if inc_perm {
                    copy_whole(&mut b, format!("{lp}.attn.q_a_norm"), "bf16", &dense, fi_dense, &format!("{sp}self_attn.q_a_layernorm.weight"))?;
                    copy_whole(&mut b, format!("{lp}.attn.kv_a_norm"), "bf16", &dense, fi_dense, &format!("{sp}self_attn.kv_a_layernorm.weight"))?;
                    copy_row_range(
                        &mut b,
                        format!("{lp}.attn.sinks"),
                        "bf16",
                        vec![nh_loc as i64],
                        &dense,
                        fi_dense,
                        &format!("{sp}self_attn.core_attention.softmax_offset"),
                        2,
                        hr.clone(),
                    )?;
                    copy_whole(&mut b, format!("{lp}.idx.k_norm_w"), "bf16", &dense, fi_dense, &format!("{sp}self_attn.core_attention.indexer.k_norm.weight"))?;
                    copy_whole(&mut b, format!("{lp}.idx.k_norm_b"), "bf16", &dense, fi_dense, &format!("{sp}self_attn.core_attention.indexer.k_norm.bias"))?;
                    copy_whole(&mut b, format!("{lp}.idx.weights_proj"), "bf16", &dense, fi_dense, &format!("{sp}self_attn.core_attention.indexer.weights_proj.weight"))?;
                }
                if inc_dense {
                    copy_whole(&mut b, format!("{lp}.attn.q_a_proj"), "bf16", &dense, fi_dense, &format!("{sp}self_attn.q_a_proj.weight"))?;
                    copy_row_range(
                        &mut b,
                        format!("{lp}.attn.q_b_proj"),
                        "bf16",
                        vec![(nh_loc * DSA_QK_HEAD) as i64, DSA_QA as i64],
                        &dense,
                        fi_dense,
                        &format!("{sp}self_attn.q_b_proj.weight"),
                        (DSA_QA * 2) as u64,
                        (hr.start * DSA_QK_HEAD)..(hr.end * DSA_QK_HEAD),
                    )?;
                    copy_whole(&mut b, format!("{lp}.attn.kv_a_proj"), "bf16", &dense, fi_dense, &format!("{sp}self_attn.kv_a_proj_with_mqa.weight"))?;
                    copy_row_range(
                        &mut b,
                        format!("{lp}.attn.kv_b_proj"),
                        "bf16",
                        vec![(nh_loc * DSA_KVB_HEAD) as i64, DSA_KV_LORA as i64],
                        &dense,
                        fi_dense,
                        &format!("{sp}self_attn.kv_b_proj.weight"),
                        (DSA_KV_LORA * 2) as u64,
                        (hr.start * DSA_KVB_HEAD)..(hr.end * DSA_KVB_HEAD),
                    )?;
                    copy_row_range(
                        &mut b,
                        format!("{lp}.attn.gate_proj"),
                        "bf16",
                        vec![(nh_loc * DSA_V_HEAD) as i64, DSA_QA as i64],
                        &dense,
                        fi_dense,
                        &format!("{sp}self_attn.gate_proj.weight"),
                        (DSA_QA * 2) as u64,
                        (hr.start * DSA_V_HEAD)..(hr.end * DSA_V_HEAD),
                    )?;
                    copy_cols_bf16(
                        &mut b,
                        format!("{lp}.attn.o_proj"),
                        &dense,
                        &format!("{sp}self_attn.o_proj.weight"),
                        H,
                        HEADS * DSA_V_HEAD,
                        (hr.start * DSA_V_HEAD)..(hr.end * DSA_V_HEAD),
                    )?;
                    copy_whole(&mut b, format!("{lp}.idx.wq_b"), "bf16", &dense, fi_dense, &format!("{sp}self_attn.core_attention.indexer.wq_b.weight"))?;
                    copy_whole(&mut b, format!("{lp}.idx.wk"), "bf16", &dense, fi_dense, &format!("{sp}self_attn.core_attention.indexer.wk.weight"))?;
                }
            } else {
                // HCA C128 layer
                if inc_perm {
                    copy_whole(&mut b, format!("{lp}.attn.q_norm"), "bf16", &dense, fi_dense, &format!("{sp}self_attn.q_layernorm.weight"))?;
                    copy_whole(&mut b, format!("{lp}.attn.kv_norm"), "bf16", &dense, fi_dense, &format!("{sp}self_attn.kv_layernorm.weight"))?;
                    copy_whole(&mut b, format!("{lp}.attn.comp_norm"), "bf16", &dense, fi_dense, &format!("{sp}self_attn.core_attention.compressor.norm.weight"))?;
                    copy_whole(&mut b, format!("{lp}.attn.comp_ape"), "f32", &dense, fi_dense, &format!("{sp}self_attn.core_attention.compressor.ape"))?;
                    copy_row_range(
                        &mut b,
                        format!("{lp}.attn.sinks"),
                        "f32",
                        vec![nh_loc as i64],
                        &dense,
                        fi_dense,
                        &format!("{sp}self_attn.core_attention.attn_sink"),
                        4,
                        hr.clone(),
                    )?;
                }
                if inc_dense {
                    copy_whole(&mut b, format!("{lp}.attn.q_down"), "bf16", &dense, fi_dense, &format!("{sp}self_attn.linear_q_down_proj.weight"))?;
                    copy_row_range(
                        &mut b,
                        format!("{lp}.attn.q_up"),
                        "bf16",
                        vec![(nh_loc * HCA_HD) as i64, HCA_QA as i64],
                        &dense,
                        fi_dense,
                        &format!("{sp}self_attn.linear_q_up_proj.weight"),
                        (HCA_QA * 2) as u64,
                        (hr.start * HCA_HD)..(hr.end * HCA_HD),
                    )?;
                    copy_whole(&mut b, format!("{lp}.attn.kv_proj"), "bf16", &dense, fi_dense, &format!("{sp}self_attn.linear_kv_proj.weight"))?;
                    copy_whole(&mut b, format!("{lp}.attn.comp_wkv"), "bf16", &dense, fi_dense, &format!("{sp}self_attn.core_attention.compressor.linear_wkv.weight"))?;
                    copy_whole(&mut b, format!("{lp}.attn.comp_wgate"), "bf16", &dense, fi_dense, &format!("{sp}self_attn.core_attention.compressor.linear_wgate.weight"))?;
                    // linear_o_group_proj has NO .weight suffix in the checkpoint: [8 * 1024, 8 * 512] = [8192, 4096]
                    copy_row_range(
                        &mut b,
                        format!("{lp}.attn.wo_a"),
                        "bf16",
                        vec![(ng_loc * HCA_OLORA) as i64, (HCA_HPG * HCA_HD) as i64],
                        &dense,
                        fi_dense,
                        &format!("{sp}self_attn.linear_o_group_proj"),
                        (HCA_HPG * HCA_HD * 2) as u64,
                        (gr.start * HCA_OLORA)..(gr.end * HCA_OLORA),
                    )?;
                    copy_row_range(
                        &mut b,
                        format!("{lp}.attn.gate_proj"),
                        "bf16",
                        vec![(ng_loc * HCA_OLORA) as i64, HCA_QA as i64],
                        &dense,
                        fi_dense,
                        &format!("{sp}self_attn.gate_proj.weight"),
                        (HCA_QA * 2) as u64,
                        (gr.start * HCA_OLORA)..(gr.end * HCA_OLORA),
                    )?;
                    copy_cols_bf16(
                        &mut b,
                        format!("{lp}.attn.o_proj"),
                        &dense,
                        &format!("{sp}self_attn.o_proj.weight"),
                        H,
                        HCA_OGROUPS * HCA_OLORA,
                        (gr.start * HCA_OLORA)..(gr.end * HCA_OLORA),
                    )?;
                }
            }

            // Layer 0 dense MLP vs Layers 1..42 Latent MoE dense projections/router
            if l == 0 {
                if inc_dense {
                    let dr = dense_range(nr, rank);
                    copy_row_range(
                        &mut b,
                        format!("{lp}.mlp.gate_proj"),
                        "bf16",
                        vec![dr.len() as i64, H as i64],
                        &dense,
                        fi_dense,
                        &format!("{sp}mlp.gate_proj.weight"),
                        (H * 2) as u64,
                        dr.clone(),
                    )?;
                    copy_row_range(
                        &mut b,
                        format!("{lp}.mlp.up_proj"),
                        "bf16",
                        vec![dr.len() as i64, H as i64],
                        &dense,
                        fi_dense,
                        &format!("{sp}mlp.up_proj.weight"),
                        (H * 2) as u64,
                        dr.clone(),
                    )?;
                    copy_cols_bf16(&mut b, format!("{lp}.mlp.down_proj"), &dense, &format!("{sp}mlp.down_proj.weight"), H, DENSE_I, dr)?;
                }
            } else {
                if inc_perm {
                    copy_whole(&mut b, format!("{lp}.router.w0"), "f32", &dense, fi_dense, &format!("{sp}block_sparse_moe.gate.weight"))?;
                    copy_whole(&mut b, format!("{lp}.router.w1"), "f32", &dense, fi_dense, &format!("{sp}block_sparse_moe.gate.weight_1"))?;
                    copy_whole(&mut b, format!("{lp}.router.bias"), "f32", &dense, fi_dense, &format!("{sp}block_sparse_moe.e_score_correction_bias"))?;
                    copy_whole(&mut b, format!("{lp}.router.scale"), "f32", &dense, fi_dense, &format!("{sp}block_sparse_moe.gate.routed_scaling_factor_param"))?;
                    copy_whole(&mut b, format!("{lp}.moe.latent_norm"), "bf16", &dense, fi_dense, &format!("{sp}block_sparse_moe.latent_norm.weight"))?;
                }
                if inc_dense {
                    copy_whole(&mut b, format!("{lp}.moe.fc1_lat"), "bf16", &dense, fi_dense, &format!("{sp}block_sparse_moe.fc1_latent_proj.weight"))?;
                    copy_whole(&mut b, format!("{lp}.moe.fc2_lat"), "bf16", &dense, fi_dense, &format!("{sp}block_sparse_moe.fc2_latent_proj.weight"))?;
                }
            }
        }

        // Experts (L1..L42)
        if l >= 1 && need_experts {
            let w_full = v2plan::widths_nh(man.as_ref().unwrap(), l, NH)?
                .ok_or_else(|| format!("ernie55: manifest has no widths for layer {l}"))?;
            let sl = slices.as_ref().unwrap();
            if let Some(emit) = routed_emit {
                let w_routed = v2plan::Widths {
                    gate: w_full.gate.clone(),
                    up: w_full.up.clone(),
                    down: w_full.down.clone(),
                    sgate: [0; v2plan::NH_MAX],
                    sup: [0; v2plan::NH_MAX],
                    sdown: 0,
                    nh: NH,
                };
                let own_r = |e: usize| Ok(v2plan::formula_halves_nh(e, l, nr, rank, NH));
                v2plan::v2_layer_split(&mut b, l, &w_routed, sl, &wfile, &own_r, emit)?;
            }
            if inc_shared {
                let own_s = v2plan::formula_halves_nh(NEXP, l, nr, rank, NH);
                v2plan::v2_shared_layer(&mut b, l, &w_full, NEXP, H, sl, &wfile, &own_s)?;
            }
        }
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
            let (mut voc, mut den, mut hd, mut grp) = (0, 0, 0, 0);
            for r in 0..nr {
                let vr = vocab_range(nr, r);
                assert_eq!(vr.start, voc);
                assert_eq!(vr.len() % 128, 0);
                voc += vr.len();

                let dr = dense_range(nr, r);
                assert_eq!(dr.start, den);
                den += dr.len();

                let hr = head_range(nr, r);
                assert_eq!(hr.start, hd);
                hd += hr.len();

                let gr = hca_group_range(nr, r);
                assert_eq!(gr.start, grp);
                grp += gr.len();
            }
            assert_eq!(voc, VOCAB);
            assert_eq!(den, DENSE_I);
            assert_eq!(hd, HEADS);
            assert_eq!(grp, HCA_OGROUPS);
        }
        assert_eq!((0..L_MAIN).filter(|&l| is_dsa(l)).count(), 6);
    }

    /// Synthetic round-trip verifying ERNIE 5.5 Flash's merged index-slice layout (`e=0..511` routed 2048x2048 +
    /// `e=512` shared 4096x2048 in `L001.{units,offset,chan}` with separate `L001.shared.{block,signs}`),
    /// `mul1` codebook, bit-63 per-matrix mixed widths (K2.5: gate=2, up=2, down=3), and `RoutedEmit` hybrid splits.
    #[test]
    fn merged_slice_dual_geometry_mul1_kmix_and_hybrid_split() {
        let dir = std::env::temp_dir().join(format!("ernie55-rt-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("experts")).unwrap();
        std::fs::create_dir_all(dir.join("index-slices/tp1/k2")).unwrap();
        std::fs::create_dir_all(dir.join("index-slices/tp1/k3")).unwrap();
        std::fs::create_dir_all(dir.join("index-slices/tp1/k4")).unwrap();
        std::fs::create_dir_all(dir.join("index-slices/tp2/k2")).unwrap();
        std::fs::create_dir_all(dir.join("index-slices/tp2/k3")).unwrap();
        std::fs::create_dir_all(dir.join("index-slices/tp2/k4")).unwrap();

        let mut rt_gate_pat = vec![0u8; 2048 * 16 * 2]; // routed K2 unit = 65536 B
        for (i, b) in rt_gate_pat.iter_mut().enumerate() {
            *b = ((i.wrapping_mul(17) + 3) & 0xFF) as u8;
        }
        let mut sh_down_pat = vec![0u8; 4096 * 16 * 4]; // shared K4 unit = 262144 B
        for (i, b) in sh_down_pat.iter_mut().enumerate() {
            *b = ((i.wrapping_mul(29) ^ (i >> 4)) & 0xFF) as u8;
        }

        let write_wf = |k: u8| -> ([u64; 3], [u64; 3]) {
            let b_r = 2048u64 * 16 * k as u64;
            let b_s = 4096u64 * 16 * k as u64;
            let nr_tot = 512u64 * 16 * b_r;
            let ns_tot = 16u64 * b_s;
            let mut cur = 0u64;
            let mut roff = [0u64; 3];
            let mut soff = [0u64; 3];
            let mut hdr = String::from(
                "{\"__metadata__\":{\"format\":\"exlr8-v2\",\"contents\":\"trellis\",\"codebook\":\"mul1 0x83DCD12D, every value rounded to E4M3\"}"
            );
            for (p, nm) in ["gate", "up", "down"].iter().enumerate() {
                roff[p] = 65536 + cur;
                hdr += &format!(",\"trellis.{nm}.mul1\":{{\"dtype\":\"U8\",\"shape\":[{nr_tot}],\"data_offsets\":[{cur},{}]}}", cur + nr_tot);
                cur += nr_tot;
            }
            for (p, nm) in ["gate", "up", "down"].iter().enumerate() {
                soff[p] = 65536 + cur;
                hdr += &format!(",\"shared.trellis.{nm}.mul1\":{{\"dtype\":\"U8\",\"shape\":[{ns_tot}],\"data_offsets\":[{cur},{}]}}", cur + ns_tot);
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
            if k == 2 {
                f.write_all_at(&rt_gate_pat, roff[0]).unwrap();
            }
            if k == 4 {
                f.write_all_at(&sh_down_pat, soff[2]).unwrap();
            }
            (roff, soff)
        };
        let (roff2, soff2) = write_wf(2);
        let (roff3, soff3) = write_wf(3);
        let (roff4, soff4) = write_wf(4);

        let write_slice = |nr: usize, r: usize, k: u8, roff: [u64; 3], soff: [u64; 3]| {
            let b_r = 2048i64 * 16 * k as i64;
            let b_s = 4096i64 * 16 * k as i64;
            let mut units = Vec::new();
            let mut offs = Vec::new();
            for e in 0..512usize {
                for u in 0..16usize {
                    if (16 * e + u + 1) % nr == r {
                        units.extend_from_slice(&(e as i16).to_le_bytes());
                        units.extend_from_slice(&(u as i16).to_le_bytes());
                        for p in 0..3 {
                            offs.extend_from_slice(&(roff[p] as i64 + (16 * e as i64 + u as i64) * b_r).to_le_bytes());
                        }
                    }
                }
            }
            // Merged shared rows at e = 512 (exact ERNIE 5.5 index-slice layout!)
            for u in 0..16usize {
                if (16 * 512 + u + 1) % nr == r {
                    units.extend_from_slice(&512i16.to_le_bytes());
                    units.extend_from_slice(&(u as i16).to_le_bytes());
                    for p in 0..3 {
                        offs.extend_from_slice(&(soff[p] as i64 + u as i64 * b_s).to_le_bytes());
                    }
                }
            }
            let n = units.len() / 4;
            let chan = vec![0x3Cu8; n * 3 * 256];
            let rblk = vec![0x3Cu8; 512 * 3 * 16 * 2];
            let mut rsigns = vec![1u8; 2 * 2048];
            rsigns[5] = 0xFF;
            let sblk = vec![0x40u8; 3 * 32 * 2];
            let mut ssigns = vec![1u8; 2 * 4096];
            ssigns[7] = 0xFF;

            let items: Vec<(&str, &str, Vec<usize>, Vec<u8>)> = vec![
                ("L001.offset", "I64", vec![n, 3], offs),
                ("L001.units", "I16", vec![n, 2], units),
                ("L001.chan", "F16", vec![n, 3, 128], chan),
                ("L001.block", "F16", vec![512, 3, 16], rblk),
                ("L001.signs", "I8", vec![2, 2048], rsigns),
                ("L001.shared.block", "F16", vec![1, 3, 32], sblk),
                ("L001.shared.signs", "I8", vec![2, 4096], ssigns),
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
            std::fs::write(dir.join(format!("index-slices/tp{nr}/k{k}/rank{r}.safetensors")), out).unwrap();
        };
        for (k, ro, so) in [(2u8, roff2, soff2), (3, roff3, soff3), (4, roff4, soff4)] {
            write_slice(1, 0, k, ro, so);
            write_slice(2, 0, k, ro, so);
            write_slice(2, 1, k, ro, so);
        }

        // K2.5 per-matrix mixed widths: routed gate=2, up=2, down=3; shared gate=4, up=4, down=4
        let w_routed = v2plan::Widths {
            gate: vec![[2; v2plan::NH_MAX]; 512],
            up: vec![[2; v2plan::NH_MAX]; 512],
            down: vec![3; 512],
            sgate: [0; v2plan::NH_MAX],
            sup: [0; v2plan::NH_MAX],
            sdown: 0,
            nh: 16,
        };
        let w_shared = v2plan::Widths {
            gate: vec![[2; v2plan::NH_MAX]; 512],
            up: vec![[2; v2plan::NH_MAX]; 512],
            down: vec![3; 512],
            sgate: [4; v2plan::NH_MAX],
            sup: [4; v2plan::NH_MAX],
            sdown: 4,
            nh: 16,
        };

        let sl1 = Slices::open(dir.to_str().unwrap(), 1, &[0]).unwrap();
        let wfile = |l: usize, k: u8| v2plan::width_file(dir.to_str().unwrap(), l, k);

        // 1. Full resident TP=1 plan
        let mut b = Builder::new();
        v2plan::v2_layer(&mut b, 1, &w_routed, &sl1, &wfile, &|e| Ok(v2plan::formula_halves_nh(e, 1, 1, 0, 16))).unwrap();
        let own_s = v2plan::formula_halves_nh(512, 1, 1, 0, 16);
        v2plan::v2_shared_layer(&mut b, 1, &w_shared, 512, 4096, &sl1, &wfile, &own_s).unwrap();

        let find = |tensors: &[v2plan::Tensor], name: &str| tensors.iter().find(|t| t.name == name).unwrap().clone();
        assert_eq!(find(&b.tensors, "L001.tr3.sg").shape, vec![2, 2048]);
        assert_eq!(find(&b.tensors, "L001.tr3.v1b").shape, vec![8192, 2, 16]);
        assert_eq!(find(&b.tensors, "L001.tr3.fc1").expert_format.as_deref(), Some("exlr8-e4m3-mul1"));
        assert_eq!(find(&b.tensors, "L001.sh_tr3.sg").shape, vec![2, 4096]);
        assert_eq!(find(&b.tensors, "L001.sh_tr3.v1b").shape, vec![16, 2, 32]);
        assert_eq!(find(&b.tensors, "L001.sh_tr3.v3b").shape, vec![16, 32]);

        let tensors = b.tensors;
        let bytes = b.cur.div_ceil(ALIGN) * ALIGN;
        let plan = Plan::from_segs(b.files, b.segs, b.mems, b.gens, bytes).unwrap();

        // Check routed gate unit 0 read-back & bit-63 kmix word (gate=2, up=2 -> (1<<63)|2|(2<<4))
        let mut got_rg = vec![0u8; rt_gate_pat.len()];
        read_range(&plan, find(&tensors, "L001.tr3.fc1").offset, &mut got_rg).unwrap();
        assert_eq!(got_rg, rt_gate_pat);

        let mut km_raw = [0u8; 8];
        read_range(&plan, find(&tensors, "L001.tr3.kmix").offset, &mut km_raw).unwrap();
        let km0 = i64::from_le_bytes(km_raw);
        assert_eq!(km0, (1i64 << 63) | 2 | (2 << 4));

        // Check shared down unit 0 underwent item128_to_g6_geometry with hidden=4096 (32 groups!)
        let mut got_sd = vec![0u8; sh_down_pat.len()];
        read_range(&plan, find(&tensors, "L001.sh_tr3.fc2").offset, &mut got_sd).unwrap();
        let mut want_sd = vec![0u8; sh_down_pat.len()];
        crate::plan::item128_to_g6_geometry(&sh_down_pat, &mut want_sd, 4096);
        assert_eq!(got_sd, want_sd);

        // 2. Hybrid split check: MetaOnly has NO fc1/fc2; TrellisOnly has ONLY fc1/fc2 with identical byte sizes!
        let mut b_meta = Builder::new();
        v2plan::v2_layer_split(&mut b_meta, 1, &w_routed, &sl1, &wfile,
            &|e| Ok(v2plan::formula_halves_nh(e, 1, 1, 0, 16)), RoutedEmit::MetaOnly).unwrap();
        assert!(b_meta.tensors.iter().all(|t| t.name != "L001.tr3.fc1" && t.name != "L001.tr3.fc2"));
        assert!(b_meta.tensors.iter().any(|t| t.name == "L001.tr3.table"));

        let mut b_tr = Builder::new();
        v2plan::v2_layer_split(&mut b_tr, 1, &w_routed, &sl1, &wfile,
            &|e| Ok(v2plan::formula_halves_nh(e, 1, 1, 0, 16)), RoutedEmit::TrellisOnly).unwrap();
        assert_eq!(b_tr.tensors.len(), 2);
        assert_eq!(find(&b_tr.tensors, "L001.tr3.fc1").nbytes, find(&tensors, "L001.tr3.fc1").nbytes);
        assert_eq!(find(&b_tr.tensors, "L001.tr3.fc2").nbytes, find(&tensors, "L001.tr3.fc2").nbytes);

        // 3. TP=2 split check
        for r in 0..2 {
            let sl2 = Slices::open(dir.to_str().unwrap(), 2, &[r]).unwrap();
            let mut b2 = Builder::new();
            v2plan::v2_layer(&mut b2, 1, &w_routed, &sl2, &wfile, &|e| Ok(v2plan::formula_halves_nh(e, 1, 2, r, 16))).unwrap();
            let os2 = v2plan::formula_halves_nh(512, 1, 2, r, 16);
            assert_eq!(os2.len(), 8);
            v2plan::v2_shared_layer(&mut b2, 1, &w_shared, 512, 4096, &sl2, &wfile, &os2).unwrap();
            assert_eq!(find(&b2.tensors, "L001.sh_tr3.v1b").shape, vec![8, 2, 32]);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
