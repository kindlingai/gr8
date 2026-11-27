//! Drama 3 (`fish_omni`) TTS release loader for `g6tts`.
//!
//! Delegates all dense, vocabulary, tower, and `dev-ref` expert records to `embed_plan::EmbedRelease`
//! (supporting the `fish_omni` groups `gdn`, `attn`, `mtp_shared`, `mtp`, `global`, and `experts`),
//! and adds per-layer expert store resolution (`0 = FP8_BLOCK dense records`, `1 = trellis`, `-1 = none`)
//! plus CPU decode of routed/shared experts to row-major FP16 `[out, inp]` from either store.

use crate::codebook;
use crate::embed_plan::EmbedRelease;
use crate::exlr8::{self, f16_bits, f16_to_f32, Matrix, StFile, FP8_BLOCK};
use crate::json::Json;
use crate::v2plan::{self, Widths};
use crate::xxh64::Xxh64;
use std::collections::HashMap;
use std::path::Path;

const PROJ_NAMES: [&str; 3] = ["gate", "up", "down"];

struct TrellisLayer {
    index_path: String,
    widths: Widths,
    hidden: usize,
    n_routed: usize,
    u_blocks: usize,
}

/// Open handle to a Drama 3 exlr8 release (dense + optional trellis experts).
pub struct TtsRelease {
    pub root: String,
    pub verify: bool,
    pub embed: EmbedRelease,
    /// Map HF tensor name -> index in `embed.items`.
    pub by_name: HashMap<String, usize>,
    /// Per-layer trellis metadata when `experts/exlr8-layer-LLL.index.safetensors` + manifest entry exist.
    trellis_layers: HashMap<usize, TrellisLayer>,
    /// Layers that have dense-record (`FP8_BLOCK` or `BF16`) routed experts in `embed`.
    dense_expert_layers: HashMap<usize, usize>, // layer -> n_routed
}

fn resolve_manifest_path(release_dir: &str, manifest_name: &str) -> Result<String, String> {
    if Path::new(manifest_name).is_file() {
        return Ok(manifest_name.to_string());
    }
    let candidates = [
        format!("{release_dir}/manifests/exlr8-manifest-{manifest_name}.json"),
        format!("{release_dir}/manifests/{manifest_name}"),
        format!("{release_dir}/exlr8-manifest-{manifest_name}.json"),
        format!("{release_dir}/{manifest_name}"),
    ];
    for p in &candidates {
        if Path::new(p).is_file() {
            return Ok(p.clone());
        }
    }
    Err(format!("{release_dir}: manifest {manifest_name:?} not found"))
}

fn find_index_file(release_dir: &str, layer: usize) -> Option<String> {
    for p in [
        format!("{release_dir}/experts/exlr8-layer-{layer:03}.index.safetensors"),
        format!("{release_dir}/exlr8-layer-{layer:03}.index.safetensors"),
    ] {
        if Path::new(&p).is_file() {
            return Some(p);
        }
    }
    None
}

fn parse_meta_usize(meta: &Json, key: &str) -> Option<usize> {
    match meta.get(key) {
        Some(Json::Int(i)) if *i >= 0 => Some(*i as usize),
        Some(Json::Str(s)) => s.parse::<usize>().ok(),
        _ => None,
    }
}

impl TtsRelease {
    /// Opens a Drama 3 release directory.
    /// `manifest_name` may be `None` or empty for `dev-ref` releases where experts live in `exlr8-dense-experts-LLL.safetensors`.
    pub fn open(release_dir: &str, manifest_name: Option<&str>, verify_checksums: bool) -> Result<TtsRelease, String> {
        let embed = EmbedRelease::open(release_dir, verify_checksums)?;
        let mut by_name = HashMap::with_capacity(embed.items.len());
        let mut dense_expert_max: HashMap<usize, usize> = HashMap::new();
        for (idx, it) in embed.items.iter().enumerate() {
            let name = it.hf_name.to_string_lossy().into_owned();
            if let Some(rest) = name.strip_prefix("model.language_model.layers.") {
                if let Some((l_str, tail)) = rest.split_once(".mlp.experts.") {
                    if let Ok(l) = l_str.parse::<usize>() {
                        if let Some((e_str, proj_tail)) = tail.split_once('.') {
                            if proj_tail == "gate_proj.weight" {
                                if let Ok(e) = e_str.parse::<usize>() {
                                    let ent = dense_expert_max.entry(l).or_insert(0);
                                    *ent = (*ent).max(e + 1);
                                }
                            }
                        }
                    }
                }
            }
            by_name.insert(name, idx);
        }

        let mut trellis_layers = HashMap::new();
        if let Some(mname) = manifest_name.filter(|s| !s.is_empty()) {
            let mpath = resolve_manifest_path(release_dir, mname)?;
            let man = v2plan::width_manifest(&mpath)?;
            if let Some(layers_obj) = man.get("layers").and_then(Json::obj) {
                let mut layer_ids: Vec<usize> = layers_obj
                    .keys()
                    .filter_map(|k| k.parse::<usize>().ok())
                    .collect();
                layer_ids.sort_unstable();
                for l in layer_ids {
                    if let Some(index_path) = find_index_file(release_dir, l) {
                        if let Some(w) = v2plan::widths_auto(&man, l)? {
                            let idx_f = StFile::open(&index_path)?;
                            let signs_info = idx_f.info("signs.in")?;
                            let hidden = signs_info
                                .shape
                                .first()
                                .copied()
                                .or_else(|| parse_meta_usize(&idx_f.meta, "hidden_size"))
                                .ok_or_else(|| format!("{index_path}: cannot infer hidden_size"))?;
                            let n_routed = parse_meta_usize(&idx_f.meta, "n_routed_experts")
                                .unwrap_or(w.down.len());
                            let u_blocks = w.nh;
                            trellis_layers.insert(
                                l,
                                TrellisLayer {
                                    index_path,
                                    widths: w,
                                    hidden,
                                    n_routed,
                                    u_blocks,
                                },
                            );
                        }
                    }
                }
            }
        }

        Ok(TtsRelease {
            root: release_dir.to_string(),
            verify: verify_checksums,
            embed,
            by_name,
            trellis_layers,
            dense_expert_layers: dense_expert_max,
        })
    }

    /// Returns expert store kind for `layer`: `1` = trellis, `0` = FP8_BLOCK/dense records, `-1` = none.
    pub fn expert_store(&self, layer: usize) -> i32 {
        if self.trellis_layers.contains_key(&layer) {
            1
        } else if self.dense_expert_layers.contains_key(&layer) {
            0
        } else {
            -1
        }
    }

    fn dense_expert_item(&self, layer: usize, expert: usize, proj: usize) -> Result<usize, String> {
        let p = *PROJ_NAMES.get(proj).ok_or_else(|| format!("invalid expert proj {proj} (expected 0..=2)"))?;
        let n_routed = self.dense_expert_layers.get(&layer).copied().unwrap_or(0);
        let name = if expert == n_routed {
            format!("model.language_model.layers.{layer}.mlp.shared_expert.{p}_proj.weight")
        } else {
            format!("model.language_model.layers.{layer}.mlp.experts.{expert}.{p}_proj.weight")
        };
        self.by_name
            .get(&name)
            .copied()
            .ok_or_else(|| format!("layer {layer} expert {expert} {p}: missing dense tensor {name}"))
    }

    /// Reads raw `FP8_BLOCK` `(fp8, scale)` bytes for a store-0 routed expert projection without decoding.
    pub fn expert_read_fp8(
        &self,
        layer: usize,
        expert: usize,
        proj: usize,
        dst_fp8: &mut [u8],
        dst_scale: &mut [u8],
    ) -> Result<(), String> {
        let idx = self.dense_expert_item(layer, expert, proj)?;
        let item = &self.embed.items[idx];
        if !item.is_matrix || item.fmt != FP8_BLOCK as i32 {
            return Err(format!("{:?}: expected FP8_BLOCK matrix (got fmt={})", item.hf_name, item.fmt));
        }
        self.embed.read_component(idx, 0, dst_fp8)?;
        self.embed.read_component(idx, 1, dst_scale)?;
        Ok(())
    }

    /// Decodes expert `expert` (`0..n_routed-1`, or `n_routed` for the shared expert) projection `proj`
    /// (`0=gate`, `1=up`, `2=down`) of `layer` into row-major FP16 `[out, inp]` bytes in `dst`.
    pub fn expert_decode_f16(&self, layer: usize, expert: usize, proj: usize, dst: &mut [u8]) -> Result<(), String> {
        if proj >= 3 {
            return Err(format!("invalid expert projection {proj} (expected 0=gate, 1=up, 2=down)"));
        }
        if let Some(tl) = self.trellis_layers.get(&layer) {
            return self.decode_trellis_expert_f16(layer, tl, expert, proj, dst);
        }
        if self.dense_expert_layers.contains_key(&layer) {
            let idx = self.dense_expert_item(layer, expert, proj)?;
            let item = &self.embed.items[idx];
            let path = &self.embed.files[item.file_idx].path;
            let f = StFile::open(path)?;
            let mx = Matrix::open(&f, &item.stored_name)?;
            let want = mx.rec.out * mx.rec.inp * 2;
            if dst.len() != want {
                return Err(format!(
                    "layer {layer} expert {expert} {}: dst len {} != expected FP16 bytes {want}",
                    PROJ_NAMES[proj],
                    dst.len()
                ));
            }
            let vals = exlr8::decode_par(&mx, 0..mx.rec.out, 0..mx.rec.inp, 1)?;
            for (i, &v) in vals.iter().enumerate() {
                let b = f16_bits(v).to_le_bytes();
                dst[2 * i] = b[0];
                dst[2 * i + 1] = b[1];
            }
            return Ok(());
        }
        Err(format!("layer {layer}: no expert store available"))
    }

    fn decode_trellis_expert_f16(
        &self,
        layer: usize,
        tl: &TrellisLayer,
        expert: usize,
        proj: usize,
        dst: &mut [u8],
    ) -> Result<(), String> {
        let is_shared = expert == tl.n_routed;
        if !is_shared && expert >= tl.n_routed {
            return Err(format!(
                "layer {layer}: expert {expert} out of range (n_routed={})",
                tl.n_routed
            ));
        }
        let h_dim = tl.hidden;
        let i_dim = tl.u_blocks * 128;
        let (out_dim, in_dim) = if proj == 2 { (h_dim, i_dim) } else { (i_dim, h_dim) };
        let want = out_dim * in_dim * 2;
        if dst.len() != want {
            return Err(format!(
                "layer {layer} expert {expert} {}: dst len {} != expected FP16 bytes {want}",
                PROJ_NAMES[proj],
                dst.len()
            ));
        }
        if is_shared && tl.widths.sdown == 0 {
            // Shared expert may live in dense files if not encoded in trellis
            if self.dense_expert_layers.contains_key(&layer) {
                let idx = self.dense_expert_item(layer, expert, proj)?;
                return self.embed.decode_f16(idx, dst, 1);
            }
            return Err(format!("layer {layer}: shared expert has no trellis width in manifest"));
        }

        let idx_f = StFile::open(&tl.index_path)?;
        let sign_name = if proj == 2 { "signs.out" } else { "signs.in" };
        let signs_raw = idx_f.read_all(sign_name)?;
        if signs_raw.len() != h_dim {
            return Err(format!("{}: {sign_name} len {} != hidden {h_dim}", tl.index_path, signs_raw.len()));
        }
        let signs: Vec<f32> = signs_raw.iter().map(|&b| (b as i8) as f32).collect();

        let (pre, row_j) = if is_shared { ("shared.", 0usize) } else { ("", expert) };
        let p_name = PROJ_NAMES[proj];
        let mut wf_cache: HashMap<u8, (StFile, codebook::Codebook, codebook::TrellisRanges)> = HashMap::new();
        let mut out_f32 = vec![0f32; out_dim * in_dim];

        for u in 0..tl.u_blocks {
            let k_u8 = match (is_shared, proj) {
                (false, 0) => tl.widths.gate[expert][u],
                (false, 1) => tl.widths.up[expert][u],
                (false, 2) => tl.widths.down[expert],
                (true, 0) => tl.widths.sgate[u],
                (true, 1) => tl.widths.sup[u],
                (true, 2) => tl.widths.sdown,
                _ => unreachable!(),
            };
            let k = k_u8 as usize;
            if !(2..=4).contains(&k) {
                return Err(format!("layer {layer} expert {expert} {p_name} unit {u}: unsupported width K{k}"));
            }
            let kp = format!("{pre}k{k}.");
            // index tensor: [E, 3, U, 2] i64 (offset, length)
            let idx_name = format!("{kp}index");
            let entry_off = ((row_j * 3 + proj) * tl.u_blocks + u) * 16;
            let off_len = idx_f.read(&idx_name, entry_off as u64, 16)?;
            let file_off = i64::from_le_bytes(off_len[0..8].try_into().unwrap()) as u64;
            let byte_len = i64::from_le_bytes(off_len[8..16].try_into().unwrap()) as u64;
            let expected_len = (16 * k * h_dim) as u64;
            if byte_len != expected_len {
                return Err(format!(
                    "{}: layer {layer} expert {expert} {p_name} unit {u} K{k}: length {byte_len} != {expected_len}",
                    tl.index_path
                ));
            }

            if !wf_cache.contains_key(&k_u8) {
                let wpath = v2plan::width_file(&self.root, layer, k_u8);
                let wf = StFile::open(&wpath)?;
                let (cb, ranges) = codebook::width_file_codebook(&wf)?;
                wf_cache.insert(k_u8, (wf, cb, ranges));
            }
            let (wf, cb, ranges) = wf_cache.get(&k_u8).unwrap();
            if !ranges.holds(is_shared, proj, file_off, byte_len) {
                return Err(format!(
                    "{}: layer {layer} expert {expert} {p_name} unit {u} K{k} at {file_off}+{byte_len} outside trellis tensor",
                    wf.path
                ));
            }
            let raw = wf.read_at(file_off, byte_len)?;

            // chan: [E, U*128] f16; block: [E, H/128] f16
            let (chan_name, block_name) = if proj == 2 {
                (format!("{kp}down.suh"), format!("{kp}down.sv_block"))
            } else {
                (format!("{kp}{p_name}.svh"), format!("{kp}{p_name}.su_block"))
            };
            let chan_off = (row_j * i_dim + u * 128) * 2;
            let chan_raw = idx_f.read(&chan_name, chan_off as u64, 256)?;
            let nblocks = h_dim / 128;
            let blk_off = row_j * nblocks * 2;
            let blk_raw = idx_f.read(&block_name, blk_off as u64, (nblocks * 2) as u64)?;

            if self.verify {
                let csum_name = format!("{kp}checksum");
                let csum_off = ((row_j * 3 + proj) * tl.u_blocks + u) * 8;
                let want = u64::from_le_bytes(idx_f.read(&csum_name, csum_off as u64, 8)?.try_into().unwrap());
                let mut h = Xxh64::new(0);
                h.update(&raw);
                h.update(&chan_raw);
                h.update(&blk_raw);
                let got = h.digest();
                if got != want {
                    return Err(format!(
                        "layer {layer} expert {expert} {p_name} unit {u} K{k}: checksum {got:#x}, expected {want:#x}"
                    ));
                }
            }

            let chan: Vec<f32> = chan_raw
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            let blk: Vec<f32> = blk_raw
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            let mut full_side = vec![0f32; h_dim];
            for c in 0..h_dim {
                full_side[c] = signs[c] * blk[c / 128];
            }

            if proj == 2 {
                // down: W_in_out (128, H) = diag(chan) Hb T Hb diag(full_side); HF weight is W^T [H, I] at cols 128u..
                let t = exlr8::tiles_in_major_cb(&raw, k, h_dim, *cb);
                let mut w = exlr8::had_left(&t, 128, h_dim, &chan);
                exlr8::had_right(&mut w, 128, h_dim, &full_side);
                for r in 0..h_dim {
                    for c in 0..128 {
                        out_f32[r * i_dim + u * 128 + c] = w[c * h_dim + r];
                    }
                }
            } else {
                // gate/up: W_in_out (H, 128) = diag(full_side) Hb T Hb diag(chan); HF weight is W^T [I, H] at rows 128u..
                let t = exlr8::tiles_out_major_cb(&raw, k, h_dim, *cb);
                let mut w = exlr8::had_left(&t, h_dim, 128, &full_side);
                exlr8::had_right(&mut w, h_dim, 128, &chan);
                for r in 0..128 {
                    let dst_row = &mut out_f32[(u * 128 + r) * h_dim..(u * 128 + r + 1) * h_dim];
                    for c in 0..h_dim {
                        dst_row[c] = w[c * 128 + r];
                    }
                }
            }
        }

        for (i, &v) in out_f32.iter().enumerate() {
            let b = f16_bits(v).to_le_bytes();
            dst[2 * i] = b[0];
            dst[2 * i + 1] = b[1];
        }
        Ok(())
    }
}
