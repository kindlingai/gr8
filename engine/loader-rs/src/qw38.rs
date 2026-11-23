//! Qwen3.8-Flash-Next (qwen4_exp_text): plans a rank's device image from an exlr8 v2 release directory
//! (`dense/exlr8-{dense,vocab,mtp}.safetensors`, `experts/exlr8-layer-LLL.k{2,3,4}.safetensors`, `index-slices/`,
//! width manifest `exlr8-manifest-*.json`, and optional `g6layout` file).
//!
//! Tensor-parallel split over `nr in {1, 2, 3, 4}` ranks (rank r):
//!   full attention   Q heads [r*24/nr .. (r+1)*24/nr]; KV heads [q0/12 .. (q1-1)/12 + 1]
//!   linear (GDN)     V heads [r*48/nr .. (r+1)*48/nr]; K/Q heads [v0/3 .. (v1-1)/3 + 1]
//!   MoE routed       5 x 128-col pmcgx blocks per expert via v2plan::v2_layer (follows --layout or formula)
//!   MoE shared / MTP intermediate 640 split across ranks in 32-aligned slices
//!   lm_head          1940 tiles of 128 split across ranks (larger chunks first)
//!   PLE table        stays on NVMe (ple/exlr8-ple-L001.bin) via RowStore; only its projections/conv/norms live here

use crate::exlr8::StFile;
use crate::layout::{Group, Layout, Model};
use crate::plan::{Gen, Op, Plan};
use crate::v2plan::{self, Builder, Slices, V2Image};
use std::ops::Range;
use std::sync::Arc;

const ALIGN: u64 = 4096;
const H: usize = 2560;
const L_MAIN: usize = 48;
const NH: usize = 5;
const HEADS: usize = 24;
const GQA: usize = 12;
const HEAD_DIM: usize = 256;
const GDN_KH: usize = 16;
const GDN_VH: usize = 48;
const GDN_V_PER_K: usize = 3;
const GDN_DK: usize = 128;
const VOCAB: usize = 248320;
const VTILES: usize = VOCAB / 128;
const SHARED_I: usize = 640;

pub fn check_nr(nr: usize, rank: usize) -> Result<(), String> {
    if !(1..=4).contains(&nr) || rank >= nr {
        return Err(format!("qw38: bad TP {nr} rank {rank} (want nr in 1..=4, rank < nr)"));
    }
    Ok(())
}

pub fn full_ranges(nr: usize, r: usize) -> (Range<usize>, Range<usize>) {
    let nq = HEADS / nr;
    let q0 = r * nq;
    let q1 = q0 + nq;
    let kv0 = q0 / GQA;
    let kv1 = (q1 - 1) / GQA + 1;
    (q0..q1, kv0..kv1)
}

pub fn gdn_ranges(nr: usize, r: usize) -> (Range<usize>, Range<usize>) {
    let nv = GDN_VH / nr;
    let v0 = r * nv;
    let v1 = v0 + nv;
    let k0 = v0 / GDN_V_PER_K;
    let k1 = (v1 - 1) / GDN_V_PER_K + 1;
    (v0..v1, k0..k1)
}

/// 32-aligned intermediate slice of 640 channels for rank r of nr:
/// TP1: [0..640], TP2: [0..320, 320..640], TP3: [0..224, 224..448, 448..640], TP4: [0..160, 160..320, 320..480, 480..640]
pub fn shared_range(nr: usize, r: usize) -> Range<usize> {
    let units = SHARED_I / 32; // 20
    let base = units / nr;
    let rem = units % nr;
    let start = r * base + r.min(rem);
    let len = base + usize::from(r < rem);
    (start * 32)..((start + len) * 32)
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
    Err(format!("qw38: cannot find {sub}/{name} under {dir}"))
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

/// Copy contiguous row ranges `[r0..r1]` from a `[R, ...]` tensor of element size `elem` bytes.
fn copy_row_runs(
    b: &mut Builder,
    dst_name: String,
    dtype: &str,
    dst_shape: Vec<i64>,
    f: &StFile,
    fi: usize,
    src_name: &str,
    row_bytes: u64,
    runs: &[Range<usize>],
) -> Result<(), String> {
    let info = f.info(src_name)?;
    let total_rows: usize = runs.iter().map(Range::len).sum();
    let total_bytes = total_rows as u64 * row_bytes;
    let off = b.begin(dst_name, dtype, dst_shape, total_bytes, None);
    let mut cur = off;
    for r in runs {
        let len = r.len() as u64 * row_bytes;
        let src_off = info.start + r.start as u64 * row_bytes;
        if src_off + len > info.end {
            return Err(format!("{src_name}: row range {r:?} exceeds tensor size"));
        }
        b.seg(cur, len, Op::Copy, fi, src_off);
        cur += len;
    }
    Ok(())
}

/// Copy a column slice `col_range` from a 2D `[rows, cols]` BF16 tensor using an `Op::Gen` closure.
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

/// Copy a 3D MTP expert tensor slice:
/// - `gate_up` (`[E, 1280, 2560]`): per expert take gate rows `[s0..s1]` and up rows `[640+s0..640+s1]` -> `[E, 2*sn, 2560]`
/// - `down` (`[E, 2560, 640]`): per expert take columns `[s0..s1]` -> `[E, 2560, sn]`
fn copy_mtp_experts_bf16(
    b: &mut Builder,
    f: &Arc<StFile>,
    prefix: &str,
    sh: Range<usize>,
) -> Result<(), String> {
    let ne = 512usize;
    let sn = sh.len();
    let s0 = sh.start;
    // gate_up_proj [512, 1280, 2560]
    {
        let src_name = format!("{prefix}mlp.experts.gate_up_proj");
        let info = f.info(&src_name)?;
        let nbytes = (ne * 2 * sn * H * 2) as u64;
        let off = b.begin("L048.mtp_exp.gate_up".into(), "bf16", vec![ne as i64, (2 * sn) as i64, H as i64], nbytes, None);
        if s0 == 0 && sn == SHARED_I {
            let fi = b.file(f.path.clone());
            b.seg(off, nbytes, Op::Copy, fi, info.start);
        } else {
            let f = f.clone();
            let start = info.start;
            let row_b = H * 2;
            let exp_b = (2 * SHARED_I) * row_b;
            let gi = b.gens.len();
            b.gens.push(Gen::new(
                "L048.mtp_exp.gate_up".into(),
                nbytes,
                Box::new(move || {
                    let mut out = vec![0u8; ne * 2 * sn * row_b];
                    let chunk_b = sn * row_b;
                    for e in 0..ne {
                        let ebase = start + (e * exp_b) as u64;
                        let g = f.read_at(ebase + (s0 * row_b) as u64, chunk_b as u64)?;
                        let u = f.read_at(ebase + ((SHARED_I + s0) * row_b) as u64, chunk_b as u64)?;
                        let d = e * 2 * chunk_b;
                        out[d..d + chunk_b].copy_from_slice(&g);
                        out[d + chunk_b..d + 2 * chunk_b].copy_from_slice(&u);
                    }
                    Ok(out)
                }),
            ));
            b.seg(off, nbytes, Op::Gen, gi, 0);
        }
    }
    // down_proj [512, 2560, 640]
    {
        let src_name = format!("{prefix}mlp.experts.down_proj");
        let info = f.info(&src_name)?;
        let nbytes = (ne * H * sn * 2) as u64;
        let off = b.begin("L048.mtp_exp.down".into(), "bf16", vec![ne as i64, H as i64, sn as i64], nbytes, None);
        if s0 == 0 && sn == SHARED_I {
            let fi = b.file(f.path.clone());
            b.seg(off, nbytes, Op::Copy, fi, info.start);
        } else {
            let f = f.clone();
            let start = info.start;
            let exp_b = H * SHARED_I * 2;
            let gi = b.gens.len();
            b.gens.push(Gen::new(
                "L048.mtp_exp.down".into(),
                nbytes,
                Box::new(move || {
                    let mut out = vec![0u8; ne * H * sn * 2];
                    for e in 0..ne {
                        let raw = f.read_at(start + (e * exp_b) as u64, exp_b as u64)?;
                        let dbase = e * H * sn * 2;
                        for r in 0..H {
                            let s = (r * SHARED_I + s0) * 2;
                            let d = dbase + r * sn * 2;
                            out[d..d + sn * 2].copy_from_slice(&raw[s..s + sn * 2]);
                        }
                    }
                    Ok(out)
                }),
            ));
            b.seg(off, nbytes, Op::Gen, gi, 0);
        }
    }
    Ok(())
}

/// Plans one rank's Qwen3.8-Flash-Next device image from an assembled exlr8 v2 release.
pub fn plan(
    release_dir: &str,
    widths_path: &str,
    nr: usize,
    rank: usize,
    layout_path: Option<&str>,
    with_mtp: bool,
) -> Result<V2Image, String> {
    check_nr(nr, rank)?;
    let man = v2plan::width_manifest(widths_path)?;
    let layout = match layout_path.filter(|s| !s.is_empty()) {
        Some(p) => {
            let l = Layout::read(p)?;
            let m = Model::release(&man)?;
            l.check(&m)?;
            if l.nr() != nr {
                return Err(format!("qw38: layout {p} has {} members, want TP {nr}", l.nr()));
            }
            Some(l)
        }
        None => None,
    };

    let dense_path = find_file(release_dir, "dense", "exlr8-dense.safetensors")?;
    let vocab_path = find_file(release_dir, "dense", "exlr8-vocab.safetensors")?;
    let dense = Arc::new(StFile::open(&dense_path)?);
    let vocab = Arc::new(StFile::open(&vocab_path)?);

    // Open index slices: if a planned layout is in use, read all ranks' slices so halves moved from another rank resolve
    let slice_ranks: Vec<usize> = if layout.is_some() { (0..nr).collect() } else { vec![rank] };
    let slices = Slices::open(release_dir, nr, &slice_ranks)?;
    let wfile = |l: usize, k: u8| v2plan::width_file(release_dir, l, k);

    let mut b = Builder::new();
    let fi_dense = b.file(dense_path.clone());
    let fi_vocab = b.file(vocab_path.clone());

    // Global vocab & final HC mixer
    copy_whole(&mut b, "embed.weight".into(), "bf16", &vocab, fi_vocab, "model.language_model.embed_tokens.weight")?;
    let vr = vocab_range(nr, rank);
    copy_row_runs(
        &mut b,
        "lm_head.weight".into(),
        "bf16",
        vec![vr.len() as i64, H as i64],
        &vocab,
        fi_vocab,
        "lm_head.weight",
        (H * 2) as u64,
        &[vr],
    )?;
    for (dst, src) in [
        ("hc_final.norm", "model.language_model.hyper_connection_mixer.hc_norm.weight"),
        ("hc_final.down", "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight"),
        ("hc_final.up", "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight"),
    ] {
        copy_whole(&mut b, dst.into(), "bf16", &dense, fi_dense, src)?;
    }

    let (qr, kvr) = full_ranges(nr, rank);
    let (vr_gdn, kr_gdn) = gdn_ranges(nr, rank);
    let sh = shared_range(nr, rank);

    for l in 0..L_MAIN {
        let lp = format!("L{l:03}");
        let sp = format!("model.language_model.layers.{l}.");

        // Hyper-connections
        for (hc_dst, hc_src) in [("attn_hc", "attn_hyper_connection"), ("mlp_hc", "mlp_hyper_connection")] {
            copy_whole(&mut b, format!("{lp}.{hc_dst}.norm"), "bf16", &dense, fi_dense, &format!("{sp}{hc_src}.hc_norm.weight"))?;
            copy_whole(&mut b, format!("{lp}.{hc_dst}.down"), "bf16", &dense, fi_dense, &format!("{sp}{hc_src}.input_mix_weight_down.weight"))?;
            copy_whole(&mut b, format!("{lp}.{hc_dst}.up"), "bf16", &dense, fi_dense, &format!("{sp}{hc_src}.input_mix_weight_up.weight"))?;
            copy_whole(&mut b, format!("{lp}.{hc_dst}.inject"), "bf16", &dense, fi_dense, &format!("{sp}{hc_src}.block_inject_weight.weight"))?;
        }

        // Layer 1 PLE projections, conv, norms, and hash tables
        if l == 1 {
            for (dst, src, dt) in [
                ("ple.key_proj", "ple.key_proj.weight", "bf16"),
                ("ple.value_proj", "ple.value_proj.weight", "bf16"),
                ("ple.conv1d", "ple.conv1d.weight", "bf16"),
                ("ple.norm_query", "ple.norm_query.weight", "bf16"),
                ("ple.norm_key", "ple.norm_key.weight", "bf16"),
                ("ple.norm_conv", "ple.norm_conv.weight", "bf16"),
                ("ple.layer_multipliers", "ple.ple_embedding.layer_multipliers", "i64"),
                ("ple.ngram_heads_offsets", "ple.ple_embedding.ngram_heads_offsets", "i64"),
                ("ple.ngram_heads_vocab_sizes", "ple.ple_embedding.ngram_heads_vocab_sizes", "i64"),
            ] {
                copy_whole(&mut b, format!("{lp}.{dst}"), dt, &dense, fi_dense, &format!("{sp}{src}"))?;
            }
        }

        if l % 4 != 3 {
            // Gated DeltaNet linear attention
            let nk = kr_gdn.len();
            let nv = vr_gdn.len();
            let qkv_ch = (2 * nk + nv) * GDN_DK;
            let q_run = (kr_gdn.start * GDN_DK)..(kr_gdn.end * GDN_DK);
            let k_run = (GDN_KH * GDN_DK + kr_gdn.start * GDN_DK)..(GDN_KH * GDN_DK + kr_gdn.end * GDN_DK);
            let v_run = (2 * GDN_KH * GDN_DK + vr_gdn.start * GDN_DK)..(2 * GDN_KH * GDN_DK + vr_gdn.end * GDN_DK);
            let qkv_runs = [q_run, k_run, v_run];
            copy_row_runs(
                &mut b,
                format!("{lp}.gdn.in_proj_qkv"),
                "bf16",
                vec![qkv_ch as i64, H as i64],
                &dense,
                fi_dense,
                &format!("{sp}linear_attn.in_proj_qkv.weight"),
                (H * 2) as u64,
                &qkv_runs,
            )?;
            copy_row_runs(
                &mut b,
                format!("{lp}.gdn.conv1d"),
                "bf16",
                vec![qkv_ch as i64, 4],
                &dense,
                fi_dense,
                &format!("{sp}linear_attn.conv1d.weight"),
                8,
                &qkv_runs,
            )?;
            let z_run = (vr_gdn.start * GDN_DK)..(vr_gdn.end * GDN_DK);
            copy_row_runs(
                &mut b,
                format!("{lp}.gdn.in_proj_z"),
                "bf16",
                vec![(nv * GDN_DK) as i64, H as i64],
                &dense,
                fi_dense,
                &format!("{sp}linear_attn.in_proj_z.weight"),
                (H * 2) as u64,
                &[z_run.clone()],
            )?;
            copy_row_runs(
                &mut b,
                format!("{lp}.gdn.in_proj_a"),
                "bf16",
                vec![nv as i64, H as i64],
                &dense,
                fi_dense,
                &format!("{sp}linear_attn.in_proj_a.weight"),
                (H * 2) as u64,
                &[vr_gdn.clone()],
            )?;
            copy_row_runs(
                &mut b,
                format!("{lp}.gdn.in_proj_b"),
                "bf16",
                vec![nv as i64, H as i64],
                &dense,
                fi_dense,
                &format!("{sp}linear_attn.in_proj_b.weight"),
                (H * 2) as u64,
                &[vr_gdn.clone()],
            )?;
            copy_row_runs(
                &mut b,
                format!("{lp}.gdn.A_log"),
                "bf16",
                vec![nv as i64],
                &dense,
                fi_dense,
                &format!("{sp}linear_attn.A_log"),
                2,
                &[vr_gdn.clone()],
            )?;
            copy_row_runs(
                &mut b,
                format!("{lp}.gdn.dt_bias"),
                "bf16",
                vec![nv as i64],
                &dense,
                fi_dense,
                &format!("{sp}linear_attn.dt_bias"),
                2,
                &[vr_gdn.clone()],
            )?;
            copy_whole(&mut b, format!("{lp}.gdn.norm"), "bf16", &dense, fi_dense, &format!("{sp}linear_attn.norm.weight"))?;
            copy_cols_bf16(&mut b, format!("{lp}.gdn.out_proj"), &dense, &format!("{sp}linear_attn.out_proj.weight"), H, GDN_VH * GDN_DK, z_run)?;
        } else {
            // Gated GQA full attention + QSA indexer
            let nq = qr.len();
            let nkv = kvr.len();
            copy_row_runs(
                &mut b,
                format!("{lp}.attn.q_proj"),
                "bf16",
                vec![(nq * HEAD_DIM * 2) as i64, H as i64],
                &dense,
                fi_dense,
                &format!("{sp}self_attn.q_proj.weight"),
                (H * 2) as u64,
                &[(qr.start * HEAD_DIM * 2)..(qr.end * HEAD_DIM * 2)],
            )?;
            copy_row_runs(
                &mut b,
                format!("{lp}.attn.k_proj"),
                "bf16",
                vec![(nkv * HEAD_DIM) as i64, H as i64],
                &dense,
                fi_dense,
                &format!("{sp}self_attn.k_proj.weight"),
                (H * 2) as u64,
                &[(kvr.start * HEAD_DIM)..(kvr.end * HEAD_DIM)],
            )?;
            copy_row_runs(
                &mut b,
                format!("{lp}.attn.v_proj"),
                "bf16",
                vec![(nkv * HEAD_DIM) as i64, H as i64],
                &dense,
                fi_dense,
                &format!("{sp}self_attn.v_proj.weight"),
                (H * 2) as u64,
                &[(kvr.start * HEAD_DIM)..(kvr.end * HEAD_DIM)],
            )?;
            copy_cols_bf16(
                &mut b,
                format!("{lp}.attn.o_proj"),
                &dense,
                &format!("{sp}self_attn.o_proj.weight"),
                H,
                HEADS * HEAD_DIM,
                (qr.start * HEAD_DIM)..(qr.end * HEAD_DIM),
            )?;
            copy_whole(&mut b, format!("{lp}.attn.q_norm"), "bf16", &dense, fi_dense, &format!("{sp}self_attn.q_norm.weight"))?;
            copy_whole(&mut b, format!("{lp}.attn.k_norm"), "bf16", &dense, fi_dense, &format!("{sp}self_attn.k_norm.weight"))?;
            copy_whole(&mut b, format!("{lp}.idx.qk_proj"), "bf16", &dense, fi_dense, &format!("{sp}self_attn.indexer.index_qk_proj.weight"))?;
            copy_whole(&mut b, format!("{lp}.idx.q_norm"), "bf16", &dense, fi_dense, &format!("{sp}self_attn.indexer.q_layernorm.weight"))?;
            copy_whole(&mut b, format!("{lp}.idx.k_norm"), "bf16", &dense, fi_dense, &format!("{sp}self_attn.indexer.k_layernorm.weight"))?;
        }

        // MoE router + BF16 shared expert + pmcgx routed experts
        copy_whole(&mut b, format!("{lp}.router"), "bf16", &dense, fi_dense, &format!("{sp}mlp.gate.weight"))?;
        copy_whole(&mut b, format!("{lp}.shared.gate_w"), "bf16", &dense, fi_dense, &format!("{sp}mlp.shared_expert_gate.weight"))?;
        copy_row_runs(
            &mut b,
            format!("{lp}.shared.gate_proj"),
            "bf16",
            vec![sh.len() as i64, H as i64],
            &dense,
            fi_dense,
            &format!("{sp}mlp.shared_expert.gate_proj.weight"),
            (H * 2) as u64,
            &[sh.clone()],
        )?;
        copy_row_runs(
            &mut b,
            format!("{lp}.shared.up_proj"),
            "bf16",
            vec![sh.len() as i64, H as i64],
            &dense,
            fi_dense,
            &format!("{sp}mlp.shared_expert.up_proj.weight"),
            (H * 2) as u64,
            &[sh.clone()],
        )?;
        copy_cols_bf16(
            &mut b,
            format!("{lp}.shared.down_proj"),
            &dense,
            &format!("{sp}mlp.shared_expert.down_proj.weight"),
            H,
            SHARED_I,
            sh.clone(),
        )?;

        let w = v2plan::widths_nh(&man, l, NH)?.ok_or_else(|| format!("qw38: manifest has no widths for layer {l}"))?;
        let own = |e: usize| -> Result<Vec<usize>, String> {
            match &layout {
                Some(ly) => ly.held(Group::Halves(l, e), rank),
                None => Ok(v2plan::formula_halves_nh(e, l, nr, rank, NH)),
            }
        };
        v2plan::v2_layer(&mut b, l, &w, &slices, &wfile, &own)?;
    }

    // Optional MTP layer (index 48)
    if with_mtp {
        if let Ok(mtp_path) = find_file(release_dir, "dense", "exlr8-mtp.safetensors") {
            let mtp = Arc::new(StFile::open(&mtp_path)?);
            let fi_mtp = b.file(mtp_path);
            for (dst, src) in [
                ("MTP.pre_fc_norm_embedding", "mtp.pre_fc_norm_embedding.weight"),
                ("MTP.pre_fc_norm_hidden", "mtp.pre_fc_norm_hidden.weight"),
                ("MTP.fc_embedding", "mtp.fc_embedding.weight"),
                ("MTP.fc_hidden", "mtp.fc_hidden.weight"),
                ("MTP.hc_final.norm", "mtp.hyper_connection_mixer.hc_norm.weight"),
                ("MTP.hc_final.down", "mtp.hyper_connection_mixer.input_mix_weight_down.weight"),
                ("MTP.hc_final.up", "mtp.hyper_connection_mixer.input_mix_weight_up.weight"),
            ] {
                copy_whole(&mut b, dst.into(), "bf16", &mtp, fi_mtp, src)?;
            }
            let sp = "mtp.layers.0.";
            let lp = "L048";
            for (hc_dst, hc_src) in [("attn_hc", "attn_hyper_connection"), ("mlp_hc", "mlp_hyper_connection")] {
                copy_whole(&mut b, format!("{lp}.{hc_dst}.norm"), "bf16", &mtp, fi_mtp, &format!("{sp}{hc_src}.hc_norm.weight"))?;
                copy_whole(&mut b, format!("{lp}.{hc_dst}.down"), "bf16", &mtp, fi_mtp, &format!("{sp}{hc_src}.input_mix_weight_down.weight"))?;
                copy_whole(&mut b, format!("{lp}.{hc_dst}.up"), "bf16", &mtp, fi_mtp, &format!("{sp}{hc_src}.input_mix_weight_up.weight"))?;
                copy_whole(&mut b, format!("{lp}.{hc_dst}.inject"), "bf16", &mtp, fi_mtp, &format!("{sp}{hc_src}.block_inject_weight.weight"))?;
            }
            let nq = qr.len();
            let nkv = kvr.len();
            copy_row_runs(&mut b, format!("{lp}.attn.q_proj"), "bf16", vec![(nq * HEAD_DIM * 2) as i64, H as i64], &mtp, fi_mtp, &format!("{sp}self_attn.q_proj.weight"), (H * 2) as u64, &[(qr.start * HEAD_DIM * 2)..(qr.end * HEAD_DIM * 2)])?;
            copy_row_runs(&mut b, format!("{lp}.attn.k_proj"), "bf16", vec![(nkv * HEAD_DIM) as i64, H as i64], &mtp, fi_mtp, &format!("{sp}self_attn.k_proj.weight"), (H * 2) as u64, &[(kvr.start * HEAD_DIM)..(kvr.end * HEAD_DIM)])?;
            copy_row_runs(&mut b, format!("{lp}.attn.v_proj"), "bf16", vec![(nkv * HEAD_DIM) as i64, H as i64], &mtp, fi_mtp, &format!("{sp}self_attn.v_proj.weight"), (H * 2) as u64, &[(kvr.start * HEAD_DIM)..(kvr.end * HEAD_DIM)])?;
            copy_cols_bf16(&mut b, format!("{lp}.attn.o_proj"), &mtp, &format!("{sp}self_attn.o_proj.weight"), H, HEADS * HEAD_DIM, (qr.start * HEAD_DIM)..(qr.end * HEAD_DIM))?;
            copy_whole(&mut b, format!("{lp}.attn.q_norm"), "bf16", &mtp, fi_mtp, &format!("{sp}self_attn.q_norm.weight"))?;
            copy_whole(&mut b, format!("{lp}.attn.k_norm"), "bf16", &mtp, fi_mtp, &format!("{sp}self_attn.k_norm.weight"))?;
            copy_whole(&mut b, format!("{lp}.idx.qk_proj"), "bf16", &mtp, fi_mtp, &format!("{sp}self_attn.indexer.index_qk_proj.weight"))?;
            copy_whole(&mut b, format!("{lp}.idx.q_norm"), "bf16", &mtp, fi_mtp, &format!("{sp}self_attn.indexer.q_layernorm.weight"))?;
            copy_whole(&mut b, format!("{lp}.idx.k_norm"), "bf16", &mtp, fi_mtp, &format!("{sp}self_attn.indexer.k_layernorm.weight"))?;
            copy_whole(&mut b, format!("{lp}.router"), "bf16", &mtp, fi_mtp, &format!("{sp}mlp.gate.weight"))?;
            copy_whole(&mut b, format!("{lp}.shared.gate_w"), "bf16", &mtp, fi_mtp, &format!("{sp}mlp.shared_expert_gate.weight"))?;
            copy_row_runs(&mut b, format!("{lp}.shared.gate_proj"), "bf16", vec![sh.len() as i64, H as i64], &mtp, fi_mtp, &format!("{sp}mlp.shared_expert.gate_proj.weight"), (H * 2) as u64, &[sh.clone()])?;
            copy_row_runs(&mut b, format!("{lp}.shared.up_proj"), "bf16", vec![sh.len() as i64, H as i64], &mtp, fi_mtp, &format!("{sp}mlp.shared_expert.up_proj.weight"), (H * 2) as u64, &[sh.clone()])?;
            copy_cols_bf16(&mut b, format!("{lp}.shared.down_proj"), &mtp, &format!("{sp}mlp.shared_expert.down_proj.weight"), H, SHARED_I, sh.clone())?;
            copy_mtp_experts_bf16(&mut b, &mtp, sp, sh)?;
        }
    }

    let bytes = b.cur.div_ceil(ALIGN) * ALIGN;
    let plan = Plan::from_segs(b.files, b.segs, b.mems, b.gens, bytes)?;
    Ok(V2Image { rank, tensors: b.tensors, bytes, plan, cache: None })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_cover_full_geometry_tp1_to_tp4() {
        for nr in 1..=4 {
            let mut q_total = 0;
            let mut v_total = 0;
            let mut sh_total = 0;
            let mut voc_total = 0;
            for r in 0..nr {
                let (qr, kvr) = full_ranges(nr, r);
                assert_eq!(qr.start, q_total);
                q_total += qr.len();
                assert!(kvr.start < kvr.end && kvr.end <= 2);

                let (vr, kr) = gdn_ranges(nr, r);
                assert_eq!(vr.start, v_total);
                v_total += vr.len();
                assert!(kr.start < kr.end && kr.end <= GDN_KH);

                let sr = shared_range(nr, r);
                assert_eq!(sr.start, sh_total);
                assert_eq!(sr.len() % 32, 0);
                sh_total += sr.len();

                let vr2 = vocab_range(nr, r);
                assert_eq!(vr2.start, voc_total);
                assert_eq!(vr2.len() % 128, 0);
                voc_total += vr2.len();
            }
            assert_eq!(q_total, HEADS);
            assert_eq!(v_total, GDN_VH);
            assert_eq!(sh_total, SHARED_I);
            assert_eq!(voc_total, VOCAB);
        }
    }
}
