//! The Inkling-2-Small planner (src/ink2.rs): ownership rules, a miniature synthetic checkpoint planned at TP 1, 4
//! and 6 and read back tensor by tensor against the checkpoint, the C interface, the reads map, and (when
//! GR8_TEST_INK2_METADATA_DIR names a directory with the real checkpoint's config.json, hf_quant_config.json,
//! model.safetensors.index.json and _tensor_shapes.json) the full-size model planned from header-only sparse files.

use g6load::ds41;
use g6load::ink2::{self, ExpertFormat, Ink2Plan, Opts};
use g6load::json::{self, Json};
use g6load::nvfp4::{self, Nvfp4ScaleRule};
use g6load::records;
use std::collections::BTreeMap;
use std::ffi::CString;
use std::ops::Range;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
    /// E4M3 scale bytes that are finite and positive.
    fn e4m3(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| 1 + (self.next() % 0x70) as u8).collect()
    }
    /// E8M0 scale bytes near 2^0.
    fn e8m0(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| 120 + (self.next() % 12) as u8).collect()
    }
    fn f32s(&mut self, n: usize) -> Vec<u8> {
        (0..n).flat_map(|_| (0.25 + (self.next() & 0xffff) as f32 / 4096.0).to_le_bytes()).collect()
    }
}

// the miniature model
const HEADS: usize = 48;
const KV: usize = 8;
const HD: usize = 16;
const HID: usize = 192;
const INTER: usize = 768;
const NEXP: usize = 16;
const VOCAB: usize = 500;
const REL: usize = 4;
const RELWIN: usize = 32;
const MTP_INTER: usize = 768;
const MARKOV: usize = 32;
/// per layer: global attention
const GLOBAL: [bool; 4] = [false, true, false, true];
/// per layer: 0 = NVFP4 with amax / denominator, 1 = NVFP4 with scale_2, 2 = MXFP8
const KIND: [u8; 4] = [0, 1, 2, 2];
const DENOM: f32 = 1536.0;

type Item = (String, &'static str, Vec<usize>, Vec<u8>);
/// (output, native, K, rows, columns, has a decode record)
type MatCase = (String, String, usize, Range<usize>, Range<usize>, bool);
/// (name, dtype, shape, bytes)
type Entry = (String, String, Vec<u64>, u64);

/// A checkpoint in memory: tensor name (as the planner keys it) -> (dtype, shape, bytes).
struct Mini {
    dir: String,
    /// model shards, then mtp.safetensors (names without the `mtp.` prefix)
    shards: Vec<Vec<Item>>,
    mtp: Vec<Item>,
    t: BTreeMap<String, (Vec<usize>, Vec<u8>)>,
}

impl Drop for Mini {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }
}

fn esize(dtype: &str) -> usize {
    match dtype { "BF16" => 2, "F32" => 4, _ => 1 }
}

fn mini(name: &str) -> Mini {
    let dir = std::env::temp_dir().join(format!("g6l-ink2-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut rng = Rng(0x1234_5678 ^ name.len() as u64);
    let mut shards: Vec<Vec<Item>> = vec![Vec::new(), Vec::new()];
    let mut mtp: Vec<Item> = Vec::new();
    let mut add = |to: &mut Vec<Item>, n: String, d: &'static str, sh: &[usize]| {
        let count = sh.iter().product::<usize>();
        let b = match d {
            "F32" => rng.f32s(count),
            "F8_E4M3" if n.ends_with("_scale") => rng.e4m3(count),
            "U8" if n.ends_with("_scale") => rng.e8m0(count),
            _ => rng.bytes(count * esize(d)),
        };
        to.push((n, d, sh.to_vec(), b));
    };
    let g = &mut shards[0];
    add(g, "model.llm.embed.weight".into(), "BF16", &[VOCAB, HID]);
    add(g, "model.llm.embed_norm.weight".into(), "BF16", &[HID]);
    add(g, "model.llm.norm.weight".into(), "BF16", &[HID]);
    add(g, "model.llm.attn_res.output_query".into(), "F32", &[HID]);
    add(g, "model.llm.unembed.weight".into(), "BF16", &[VOCAB, HID]);
    for l in 0..GLOBAL.len() {
        let s = &mut shards[l / 2];
        let t = |x: &str| format!("model.llm.layers.{l}.{x}");
        if l >= 1 {
            let sh: &[usize] = if GLOBAL[l] { &[3, HID] } else { &[HID] };
            add(s, format!("model.llm.attn_res.attn_queries.{}", l - 1), "F32", sh);
        }
        add(s, format!("model.llm.attn_res.mlp_queries.{l}"), "F32", &[HID]);
        if GLOBAL[l] {
            for n in ["q_input_norm", "k_input_norm", "v_input_norm"] {
                add(s, t(&format!("{n}.weight")), "BF16", &[HID]);
            }
        } else {
            add(s, t("attn_norm.weight"), "BF16", &[HID]);
        }
        add(s, t("attn.wq_du.weight"), "BF16", &[HEADS * HD, HID]);
        add(s, t("attn.wr_du.weight"), "BF16", &[HEADS * REL, HID]);
        add(s, t("attn.gate_attn.weight"), "BF16", &[HEADS * HD, HID]);
        add(s, t("attn.attn_sink_proj.weight"), "BF16", &[HEADS, HID]);
        add(s, t("attn.wk_dv.weight"), "BF16", &[KV * HD, HID]);
        add(s, t("attn.wv_dv.weight"), "BF16", &[KV * HD, HID]);
        add(s, t("attn.wo_ud.weight"), "BF16", &[HID, HEADS * HD]);
        add(s, t("attn.q_l2_clip.scale"), "BF16", &[HEADS, HD]);
        add(s, t("attn.rel_l2_clip.scale"), "BF16", &[HEADS, REL]);
        add(s, t("attn.rel_logits_proj.proj"), "BF16", &[REL, RELWIN]);
        for w in ["k_sconv", "v_sconv"] {
            if GLOBAL[l] {
                add(s, t(&format!("attn.{w}.static_w")), "BF16", &[4, KV * HD]);
                add(s, t(&format!("attn.{w}.weight_proj")), "BF16", &[KV * 4, HID]);
            } else {
                add(s, t(&format!("attn.{w}.weight")), "BF16", &[KV * HD, 1, 4]);
            }
        }
        add(s, t("attn_sconv.weight"), "BF16", &[HID, 1, 4]);
        add(s, t("mlp_sconv.weight"), "BF16", &[HID, 1, 4]);
        add(s, t("mlp_norm.weight"), "BF16", &[HID]);
        add(s, t("mlp.gate.weight"), "BF16", &[NEXP, HID]);
        add(s, t("mlp.gate.bias"), "F32", &[NEXP]);
        let x = |n: &str| t(&format!("mlp.experts.{n}"));
        if KIND[l] < 2 {
            add(s, x("w13_weight"), "U8", &[NEXP, 2 * INTER, HID / 2]);
            add(s, x("w13_weight_scale"), "F8_E4M3", &[NEXP, 2 * INTER, HID / 16]);
            add(s, x("w2_weight"), "U8", &[NEXP, HID, INTER / 2]);
            add(s, x("w2_weight_scale"), "F8_E4M3", &[NEXP, HID, INTER / 16]);
            for w in ["w13", "w2"] {
                if KIND[l] == 0 {
                    add(s, x(&format!("{w}_weight_amax")), "F32", &[NEXP]);
                    s.push((x(&format!("{w}_weight_outer_scale_denominator")), "F32", vec![1], DENOM.to_le_bytes().to_vec()));
                } else {
                    add(s, x(&format!("{w}_weight_scale_2")), "F32", &[NEXP]);
                }
            }
        } else {
            add(s, x("w13_weight"), "F8_E4M3", &[NEXP, 2 * INTER, HID]);
            add(s, x("w13_weight_scale"), "U8", &[NEXP, 2 * INTER, HID / 32]);
            add(s, x("w2_weight"), "F8_E4M3", &[NEXP, HID, INTER]);
            add(s, x("w2_weight_scale"), "U8", &[NEXP, HID, INTER / 32]);
        }
    }
    let s = &mut shards[1];
    add(s, "model.visual.proj_in.weight".into(), "BF16", &[8, 24]);
    add(s, "model.visual.backbone.layers.0.ln_1.weight".into(), "BF16", &[8]);
    add(s, "model.visual.backbone.posemb".into(), "F32", &[5, 8]);
    add(&mut mtp, "fuse_weight".into(), "BF16", &[6, 8]);
    add(&mut mtp, "mask_embed".into(), "BF16", &[HID]);
    add(&mut mtp, "norm.weight".into(), "BF16", &[HID]);
    add(&mut mtp, "markov_head.dflare_norm.weight".into(), "BF16", &[HID]);
    add(&mut mtp, "markov_head.rnn_proj.weight".into(), "BF16", &[MARKOV, HID]);
    add(&mut mtp, "markov_head.rnn_proj.bias".into(), "BF16", &[MARKOV]);
    add(&mut mtp, "markov_head.hidden_out.weight".into(), "BF16", &[MARKOV, MARKOV]);
    add(&mut mtp, "markov_head.hidden_out.bias".into(), "BF16", &[MARKOV]);
    add(&mut mtp, "markov_head.markov_w1.weight".into(), "BF16", &[VOCAB, MARKOV]);
    for m in 0..2 {
        let t = |x: &str| format!("layers.{m}.{x}");
        for n in ["dflare_norm", "input_layernorm", "post_attention_layernorm"] {
            add(&mut mtp, t(&format!("{n}.weight")), "BF16", &[HID]);
        }
        for n in ["q_norm", "k_norm", "k_norm_ctx"] {
            add(&mut mtp, t(&format!("self_attn.{n}.weight")), "BF16", &[HD]);
        }
        add(&mut mtp, t("self_attn.q_proj.weight"), "BF16", &[HEADS * HD, HID]);
        for n in ["k_proj", "k_proj_ctx", "v_proj", "v_proj_ctx"] {
            add(&mut mtp, t(&format!("self_attn.{n}.weight")), "BF16", &[KV * HD, HID]);
        }
        add(&mut mtp, t("self_attn.o_proj.weight"), "BF16", &[HID, HEADS * HD]);
        add(&mut mtp, t("mlp.gate_proj.weight"), "BF16", &[MTP_INTER, HID]);
        add(&mut mtp, t("mlp.up_proj.weight"), "BF16", &[MTP_INTER, HID]);
        add(&mut mtp, t("mlp.down_proj.weight"), "BF16", &[HID, MTP_INTER]);
    }
    let mut t = BTreeMap::new();
    for (n, _, sh, b) in shards.iter().flatten() {
        t.insert(n.clone(), (sh.clone(), b.clone()));
    }
    for (n, _, sh, b) in &mtp {
        t.insert(format!("mtp.{n}"), (sh.clone(), b.clone()));
    }
    Mini { dir: dir.to_string_lossy().into_owned(), shards, mtp, t }
}

fn st_bytes(items: &[Item]) -> Vec<u8> {
    let (mut parts, mut off) = (vec!["\"__metadata__\": {\"format\": \"pt\"}".to_string()], 0usize);
    for (n, d, sh, b) in items {
        let shape: Vec<String> = sh.iter().map(usize::to_string).collect();
        parts.push(format!("\"{n}\": {{\"dtype\": \"{d}\", \"shape\": [{}], \"data_offsets\": [{off}, {}]}}", shape.join(", "), off + b.len()));
        off += b.len();
    }
    let mut head = format!("{{{}}}", parts.join(", ")).into_bytes();
    while (8 + head.len()) % 8 != 0 {
        head.push(b' ');
    }
    let mut out = (head.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&head);
    for (_, _, _, b) in items {
        out.extend_from_slice(b);
    }
    out
}

fn shard_name(i: usize, n: usize) -> String {
    format!("model-{:05}-of-{n:05}.safetensors", i + 1)
}

impl Mini {
    fn config(&self) -> String {
        let local: Vec<String> = (0..GLOBAL.len()).filter(|&l| !GLOBAL[l]).map(|l| l.to_string()).collect();
        format!("{{\"model_type\": \"inkling2\", \"text_config\": {{\"hidden_size\": {HID}, \"num_hidden_layers\": {}, \"vocab_size\": {VOCAB}, \
                 \"num_attention_heads\": {HEADS}, \"num_key_value_heads\": {KV}, \"head_dim\": {HD}, \"local_layer_ids\": [{}], \
                 \"n_routed_experts\": {NEXP}, \"num_experts_per_tok\": 4, \"intermediate_size\": {INTER}}}}}", GLOBAL.len(), local.join(", "))
    }

    fn quant(&self) -> String {
        let names = |want: fn(u8) -> bool| -> String {
            (0..KIND.len()).filter(|&l| want(KIND[l]))
                .flat_map(|l| ["w13_weight", "w2_weight"].map(|w| format!("\"model.llm.layers.{l}.mlp.experts.{w}\""))).collect::<Vec<_>>().join(", ")
        };
        format!("{{\"model\": {{\"parameter_groups\": [\
                 {{\"compute_dtype\": \"bfloat16\", \"tensors\": [\"model.llm.embed.weight\"]}}, \
                 {{\"compute_dtype\": \"nvfp4\", \"required_activation_quantization\": {{\"scale_selection_rule\": \"fouroversix_mae\"}}, \"tensors\": [{}]}}, \
                 {{\"compute_dtype\": \"mxfp8\", \"tensors\": [{}]}}]}}, \"quant_method\": \"inkling2\"}}", names(|k| k < 2), names(|k| k == 2))
    }

    fn write(&self) {
        let d = &self.dir;
        std::fs::write(format!("{d}/config.json"), self.config()).unwrap();
        std::fs::write(format!("{d}/hf_quant_config.json"), self.quant()).unwrap();
        let mut map = Vec::new();
        for (i, s) in self.shards.iter().enumerate() {
            let f = shard_name(i, self.shards.len());
            std::fs::write(format!("{d}/{f}"), st_bytes(s)).unwrap();
            map.extend(s.iter().map(|(n, ..)| format!("\"{n}\": \"{f}\"")));
        }
        std::fs::write(format!("{d}/model.safetensors.index.json"), format!("{{\"metadata\": {{}}, \"weight_map\": {{{}}}}}", map.join(", "))).unwrap();
        std::fs::write(format!("{d}/mtp.safetensors"), st_bytes(&self.mtp)).unwrap();
    }

    fn raw(&self, n: &str) -> &[u8] {
        &self.t.get(n).unwrap_or_else(|| panic!("no source tensor {n}")).1
    }

    /// Rows `rows` and columns `cols` of the [N][k] matrix `n` with elements of `e` bytes.
    fn cut(&self, n: &str, k: usize, e: usize, rows: Range<usize>, cols: Range<usize>) -> Vec<u8> {
        let b = self.raw(n);
        assert_eq!(b.len() % (k * e), 0, "{n}");
        rows.flat_map(|r| b[(r * k + cols.start) * e..(r * k + cols.end) * e].to_vec()).collect()
    }
}

fn plan(m: &Mini, nr: usize, rank: usize) -> Ink2Plan {
    ink2::plan(&m.dir, nr, rank, &Opts::default()).unwrap_or_else(|e| panic!("plan nr {nr} rank {rank}: {e}"))
}

fn plan_err(dir: &str, nr: usize, rank: usize) -> String {
    match ink2::plan(dir, nr, rank, &Opts::default()) {
        Err(e) => e,
        Ok(_) => panic!("plan nr {nr} rank {rank} should fail"),
    }
}

fn tensor(p: &Ink2Plan, name: &str) -> Vec<u8> {
    ds41::read_tensor(&p.img, name).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn meta<'a>(p: &'a Ink2Plan, name: &str) -> &'a g6load::v2plan::Tensor {
    p.img.tensors.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("no tensor {name}"))
}

fn u16s(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

// ---------------------------------------------------------------------------------------------------- tests

#[test]
fn ownership_rules() {
    // TP 6 and TP 4 of the real model: every rank reads two contiguous KV heads
    let kv6: Vec<_> = (0..6).map(|r| ink2::head_ranges(48, 8, 6, r).unwrap()).collect();
    assert_eq!(kv6.iter().map(|x| x.1.clone()).collect::<Vec<_>>(), vec![0..2, 1..3, 2..4, 4..6, 5..7, 6..8]);
    assert_eq!(kv6[3].0, 24..32);
    let kv4: Vec<_> = (0..4).map(|r| ink2::head_ranges(48, 8, 4, r).unwrap().1).collect();
    assert_eq!(kv4, vec![0..2, 2..4, 4..6, 6..8]);
    assert_eq!(ink2::head_ranges(48, 8, 1, 0).unwrap(), (0..48, 0..8));
    // every q head finds its KV head inside the rank's range, for every valid TP
    for nr in [1usize, 2, 3, 4, 6, 12] {
        let mut seen = 0;
        for rank in 0..nr {
            let (q, kv) = ink2::head_ranges(48, 8, nr, rank).unwrap();
            assert_eq!(q.start, seen);
            seen = q.end;
            for lq in 0..q.len() {
                let g = (q.start + lq) / 6;
                assert!(kv.contains(&g), "nr {nr} rank {rank} q {lq}");
                assert!(g - kv.start < kv.len());
            }
            assert!(kv.clone().all(|k| q.clone().any(|h| h / 6 == k)), "no unused KV head");
        }
        assert_eq!(seen, 48);
    }
    assert!(ink2::head_ranges(48, 8, 5, 0).is_err());
    assert!(ink2::head_ranges(48, 8, 6, 6).is_err());
    assert!(ink2::head_ranges(48, 8, 0, 0).is_err());
    assert!(ink2::head_ranges(48, 7, 6, 0).is_err());
    // vocabulary padding
    assert_eq!(ink2::vocab_split(201024, 6), (201216, 33536));
    assert_eq!(ink2::vocab_split(201024, 4), (201216, 50304));
    assert_eq!(ink2::vocab_split(201024, 1), (201088, 201088));
    assert_eq!(ink2::vocab_split(512, 4), (512, 128));
}

/// Every tensor of every rank at TP 1, 4 and 6 equals the slice of the checkpoint the rules name.
#[test]
fn slices_match_the_checkpoint() {
    let m = mini("slices");
    m.write();
    for nr in [1usize, 4, 6] {
        for rank in 0..nr {
            let p = plan(&m, nr, rank);
            let what = format!("nr {nr} rank {rank}");
            let (q, kv) = ink2::head_ranges(HEADS, KV, nr, rank).unwrap();
            assert_eq!((&p.q_range, &p.kv_range), (&q, &kv));
            let (padded, per) = ink2::vocab_split(VOCAB, nr);
            assert_eq!((p.vocab_padded, p.v_range.clone()), (padded, rank * per..(rank + 1) * per));
            assert_eq!(p.experts, vec![ExpertFormat::Nvfp4(Nvfp4ScaleRule::FourOverSixMae), ExpertFormat::Nvfp4(Nvfp4ScaleRule::FourOverSixMae),
                                       ExpertFormat::Mxfp8, ExpertFormat::Mxfp8]);
            let isl = rank * (INTER / nr)..(rank + 1) * (INTER / nr);
            let iw = isl.len();
            let seen: std::cell::RefCell<Vec<String>> = std::cell::RefCell::new(Vec::new());
            let eq = |name: &str, shape: &[usize], want: Vec<u8>| {
                let t = meta(&p, name);
                assert_eq!(t.shape, shape.iter().map(|&d| d as i64).collect::<Vec<_>>(), "{what} {name}");
                assert!(tensor(&p, name) == want, "{what}: {name} differs from the checkpoint");
                seen.borrow_mut().push(name.to_string());
            };
            // vocabulary rows of this rank, zero past the vocabulary
            let vrows = |n: &str, k: usize| {
                let mut b = m.cut(n, k, 2, p.v_range.start.min(VOCAB)..p.v_range.end.min(VOCAB), 0..k);
                b.resize(per * k * 2, 0);
                b
            };
            eq("G.embed", &[VOCAB, HID], m.raw("model.llm.embed.weight").to_vec());
            eq("G.embed_norm", &[HID], m.raw("model.llm.embed_norm.weight").to_vec());
            eq("G.final_norm", &[HID], m.raw("model.llm.norm.weight").to_vec());
            eq("G.attn_res.output_query", &[HID], m.raw("model.llm.attn_res.output_query").to_vec());
            let head = vrows("model.llm.unembed.weight", HID);
            eq("G.lm_head.gv", &[per / 16, HID / 16, 512], records::bf16_record(&u16s(&head), HID));
            eq("G.lm_head", &[per, HID], head);
            for l in 0..GLOBAL.len() {
                let t = |x: &str| format!("model.llm.layers.{l}.{x}");
                let o = |x: &str| format!("L{l:03}.{x}");
                let qs = |mul: usize| q.start * mul..q.end * mul;
                let ks = |mul: usize| kv.start * mul..kv.end * mul;
                if l >= 1 {
                    let sh: &[usize] = if GLOBAL[l] { &[3, HID] } else { &[HID] };
                    eq(&o("attn_res.attn_query"), sh, m.raw(&format!("model.llm.attn_res.attn_queries.{}", l - 1)).to_vec());
                } else {
                    assert!(!p.img.tensors.iter().any(|t| t.name == "L000.attn_res.attn_query"));
                }
                eq(&o("attn_res.mlp_query"), &[HID], m.raw(&format!("model.llm.attn_res.mlp_queries.{l}")).to_vec());
                if GLOBAL[l] {
                    for n in ["q_input_norm", "k_input_norm", "v_input_norm"] {
                        eq(&o(n), &[HID], m.raw(&t(&format!("{n}.weight"))).to_vec());
                    }
                } else {
                    eq(&o("attn_norm"), &[HID], m.raw(&t("attn_norm.weight")).to_vec());
                }
                let mut mats: Vec<MatCase> = vec![
                    (o("wq"), t("attn.wq_du.weight"), HID, qs(HD), 0..HID, true),
                    (o("wr"), t("attn.wr_du.weight"), HID, qs(REL), 0..HID, true),
                    (o("gate_attn"), t("attn.gate_attn.weight"), HID, qs(HD), 0..HID, true),
                    (o("attn_sink_proj"), t("attn.attn_sink_proj.weight"), HID, qs(1), 0..HID, false),
                    (o("wk"), t("attn.wk_dv.weight"), HID, ks(HD), 0..HID, true),
                    (o("wv"), t("attn.wv_dv.weight"), HID, ks(HD), 0..HID, true),
                    (o("wo"), t("attn.wo_ud.weight"), HEADS * HD, 0..HID, qs(HD), true),
                    (o("q_l2_clip"), t("attn.q_l2_clip.scale"), HD, qs(1), 0..HD, false),
                    (o("rel_l2_clip"), t("attn.rel_l2_clip.scale"), REL, qs(1), 0..REL, false),
                    (o("rel_logits_proj"), t("attn.rel_logits_proj.proj"), RELWIN, 0..REL, 0..RELWIN, false),
                    (o("attn_sconv"), t("attn_sconv.weight"), 4, 0..HID, 0..4, false),
                    (o("mlp_sconv"), t("mlp_sconv.weight"), 4, 0..HID, 0..4, false),
                    (o("router.w"), t("mlp.gate.weight"), HID, 0..NEXP, 0..HID, true),
                ];
                for w in ["k_sconv", "v_sconv"] {
                    if GLOBAL[l] {
                        mats.push((o(&format!("{w}.static_w")), t(&format!("attn.{w}.static_w")), KV * HD, 0..4, ks(HD), false));
                        mats.push((o(&format!("{w}.weight_proj")), t(&format!("attn.{w}.weight_proj")), HID, ks(4), 0..HID, false));
                    } else {
                        mats.push((o(w), t(&format!("attn.{w}.weight")), 4, ks(HD), 0..4, false));
                    }
                }
                for (out, native, k, rows, cols, gv) in mats {
                    let want = m.cut(&native, k, 2, rows.clone(), cols.clone());
                    if gv && rows.len() % 16 == 0 && cols.len() % 16 == 0 {
                        eq(&format!("{out}.gv"), &[rows.len() / 16, cols.len() / 16, 512], records::bf16_record(&u16s(&want), cols.len()));
                    } else {
                        assert!(!p.img.tensors.iter().any(|t| t.name == format!("{out}.gv")), "{what} {out}.gv");
                    }
                    eq(&out, &[rows.len(), cols.len()], want);
                }
                eq(&o("mlp_norm"), &[HID], m.raw(&t("mlp_norm.weight")).to_vec());
                eq(&o("router.bias"), &[NEXP], m.raw(&t("mlp.gate.bias")).to_vec());

                // routed experts
                let x = |n: &str| t(&format!("mlp.experts.{n}"));
                let (fc1, fc2) = (tensor(&p, &o("x.fc1")), tensor(&p, &o("x.fc2")));
                seen.borrow_mut().extend([o("x.fc1"), o("x.fc2")]);
                if KIND[l] < 2 {
                    assert_eq!(meta(&p, &o("x.fc1")).shape, [NEXP as i64, (iw / 16) as i64, (HID / 64) as i64, 1152]);
                    assert_eq!(meta(&p, &o("x.fc2")).shape, [NEXP as i64, (HID / 16) as i64, (iw / 64) as i64, 576]);
                    assert_eq!(meta(&p, &o("x.fc1")).expert_format.as_deref(), Some("nv4g"));
                    assert_eq!(meta(&p, &o("x.fc2")).expert_format.as_deref(), Some("nv4g"));
                    let outer = |w: &str| -> Vec<f32> {
                        if KIND[l] == 0 {
                            f32s(m.raw(&x(&format!("{w}_weight_amax")))).iter().map(|a| a / DENOM).collect()
                        } else {
                            f32s(m.raw(&x(&format!("{w}_weight_scale_2"))))
                        }
                    };
                    let (s1, s2) = (outer("w13"), outer("w2"));
                    eq(&o("x.s1"), &[NEXP], s1.iter().flat_map(|v| v.to_le_bytes()).collect());
                    eq(&o("x.s2"), &[NEXP], s2.iter().flat_map(|v| v.to_le_bytes()).collect());
                    let (c13, sc13, c2, sc2) = (m.raw(&x("w13_weight")), m.raw(&x("w13_weight_scale")), m.raw(&x("w2_weight")), m.raw(&x("w2_weight_scale")));
                    let (n1, n2) = (2 * INTER * HID, HID * INTER);
                    let (per1, per2) = (fc1.len() / NEXP, fc2.len() / NEXP);
                    // two experts decoded value by value against the row-major dequantization of the whole expert
                    for e in [0usize, NEXP - 1 - rank % 3] {
                        let d1 = nvfp4::nvfp4_dequantize(&c13[e * n1 / 2..][..n1 / 2], &sc13[e * n1 / 16..][..n1 / 16], s1[e]);
                        let d2 = nvfp4::nvfp4_dequantize(&c2[e * n2 / 2..][..n2 / 2], &sc2[e * n2 / 16..][..n2 / 16], s2[e]);
                        let (ng1, ng2) = (HID / 64, iw / 64);
                        for (i, rec) in fc1[e * per1..][..per1].chunks_exact(576).enumerate() {
                            let (pair, half) = (i / 2, i % 2);
                            let (ch, g) = (pair / ng1, pair % ng1);
                            let d = nvfp4::nvfp4_decode_group(rec.try_into().unwrap(), s1[e]);
                            for (r, row) in d.iter().enumerate() {
                                let src = &d1[(half * INTER + isl.start + ch * 16 + r) * HID + g * 64..][..64];
                                assert!(row.iter().zip(src).all(|(a, b)| a.to_bits() == b.to_bits()), "{what} L{l} fc1 expert {e} chunk {ch} group {g}");
                            }
                        }
                        for (i, rec) in fc2[e * per2..][..per2].chunks_exact(576).enumerate() {
                            let (ot, g) = (i / ng2, i % ng2);
                            let d = nvfp4::nvfp4_decode_group(rec.try_into().unwrap(), s2[e]);
                            for (r, row) in d.iter().enumerate() {
                                let src = &d2[(ot * 16 + r) * INTER + isl.start + g * 64..][..64];
                                assert!(row.iter().zip(src).all(|(a, b)| a.to_bits() == b.to_bits()), "{what} L{l} fc2 expert {e} tile {ot} group {g}");
                            }
                        }
                    }
                    // every expert: the records of the copied slices
                    let (mut want1, mut want2) = (Vec::new(), Vec::new());
                    for e in 0..NEXP {
                        let part = |b: &[u8], row: usize, half: usize| b[(e * 2 * INTER + half * INTER + isl.start) * row..][..iw * row].to_vec();
                        want1.extend(nvfp4::nvfp4_fc1(&part(c13, HID / 2, 0), &part(sc13, HID / 16, 0), &part(c13, HID / 2, 1), &part(sc13, HID / 16, 1), iw, HID));
                        let cols = |b: &[u8], row: usize, c0: usize, cn: usize| -> Vec<u8> {
                            (0..HID).flat_map(|r| b[(e * HID + r) * row + c0..][..cn].to_vec()).collect()
                        };
                        want2.extend(nvfp4::nvfp4_record(&cols(c2, INTER / 2, isl.start / 2, iw / 2), &cols(sc2, INTER / 16, isl.start / 16, iw / 16), iw));
                    }
                    assert!(fc1 == want1 && fc2 == want2, "{what} L{l}: nv4g records");
                } else {
                    assert_eq!(meta(&p, &o("x.fc1")).shape, [NEXP as i64, (iw / 16) as i64, (HID / 32) as i64, 1056]);
                    assert_eq!(meta(&p, &o("x.fc2")).shape, [NEXP as i64, (HID / 16) as i64, (iw / 32) as i64, 528]);
                    assert_eq!(meta(&p, &o("x.fc2")).expert_format.as_deref(), Some("mxfp8"));
                    assert!(!p.img.tensors.iter().any(|t| t.name == o("x.s1") || t.name == o("x.s2")));
                    let (c13, sc13, c2, sc2) = (m.raw(&x("w13_weight")), m.raw(&x("w13_weight_scale")), m.raw(&x("w2_weight")), m.raw(&x("w2_weight_scale")));
                    let (mut want1, mut want2) = (Vec::new(), Vec::new());
                    for e in 0..NEXP {
                        let part = |b: &[u8], row: usize, half: usize| b[(e * 2 * INTER + half * INTER + isl.start) * row..][..iw * row].to_vec();
                        let g = records::mxfp8_record(&part(c13, HID, 0), &part(sc13, HID / 32, 0), HID);
                        let u = records::mxfp8_record(&part(c13, HID, 1), &part(sc13, HID / 32, 1), HID);
                        for (a, b) in g.chunks_exact(528).zip(u.chunks_exact(528)) {
                            want1.extend_from_slice(a);
                            want1.extend_from_slice(b);
                        }
                        let cols = |b: &[u8], row: usize, c0: usize, cn: usize| -> Vec<u8> {
                            (0..HID).flat_map(|r| b[(e * HID + r) * row + c0..][..cn].to_vec()).collect()
                        };
                        want2.extend(records::mxfp8_record(&cols(c2, INTER, isl.start, iw), &cols(sc2, INTER / 32, isl.start / 32, iw / 32), iw));
                    }
                    assert!(fc1 == want1 && fc2 == want2, "{what} L{l}: mxfp8 records");
                }
            }
            // vision: replicated as stored
            for (n, sh) in [("proj_in.weight", vec![8, 24]), ("backbone.layers.0.ln_1.weight", vec![8]), ("backbone.posemb", vec![5, 8])] {
                eq(&format!("V.{n}"), &sh, m.raw(&format!("model.visual.{n}")).to_vec());
            }
            assert_eq!(meta(&p, "V.backbone.posemb").dtype, "f32");
            // MTP
            for (n, sh) in [("fuse_weight", vec![6, 8]), ("mask_embed", vec![HID]), ("norm.weight", vec![HID]), ("markov_head.dflare_norm.weight", vec![HID]),
                            ("markov_head.rnn_proj.weight", vec![MARKOV, HID]), ("markov_head.rnn_proj.bias", vec![MARKOV]),
                            ("markov_head.hidden_out.weight", vec![MARKOV, MARKOV]), ("markov_head.hidden_out.bias", vec![MARKOV])] {
                eq(&format!("MTP.{n}"), &sh, m.raw(&format!("mtp.{n}")).to_vec());
            }
            eq("MTP.markov_head.markov_w1", &[per, MARKOV], vrows("mtp.markov_head.markov_w1.weight", MARKOV));
            let msl = rank * (MTP_INTER / nr)..(rank + 1) * (MTP_INTER / nr);
            for l in 0..2 {
                let t = |x: &str| format!("mtp.layers.{l}.{x}");
                let o = |x: &str| format!("MTP.L{l:02}.{x}");
                for n in ["dflare_norm", "input_layernorm", "post_attention_layernorm"] {
                    eq(&o(&format!("{n}.weight")), &[HID], m.raw(&t(&format!("{n}.weight"))).to_vec());
                }
                for n in ["q_norm", "k_norm", "k_norm_ctx"] {
                    eq(&o(&format!("self_attn.{n}.weight")), &[HD], m.raw(&t(&format!("self_attn.{n}.weight"))).to_vec());
                }
                let (qr, kr) = (q.start * HD..q.end * HD, kv.start * HD..kv.end * HD);
                let cutm = |n: &str, k: usize, rows: Range<usize>, cols: Range<usize>| {
                    eq(&o(n), &[rows.len(), cols.len()], m.cut(&t(n), k, 2, rows, cols));
                };
                cutm("self_attn.q_proj.weight", HID, qr.clone(), 0..HID);
                for n in ["k_proj", "k_proj_ctx", "v_proj", "v_proj_ctx"] {
                    cutm(&format!("self_attn.{n}.weight"), HID, kr.clone(), 0..HID);
                }
                cutm("self_attn.o_proj.weight", HEADS * HD, 0..HID, qr);
                cutm("mlp.gate_proj.weight", HID, msl.clone(), 0..HID);
                cutm("mlp.up_proj.weight", HID, msl.clone(), 0..HID);
                cutm("mlp.down_proj.weight", MTP_INTER, 0..HID, msl.clone());
            }
            // nothing else is in the image, tensors do not overlap, the image is page aligned
            let mut names: Vec<&str> = p.img.tensors.iter().map(|t| t.name.as_str()).collect();
            let seen = seen.into_inner();
            let mut seen: Vec<&str> = seen.iter().map(String::as_str).collect();
            names.sort();
            seen.sort();
            assert_eq!(names, seen, "{what}: the image holds exactly the checked tensors");
            let mut ext: Vec<(u64, u64)> = p.img.tensors.iter().map(|t| (t.offset, t.nbytes)).collect();
            ext.sort();
            assert!(ext.windows(2).all(|w| w[0].0 + w[0].1 <= w[1].0) && p.img.bytes.is_multiple_of(4096));
        }
    }
}

#[test]
fn options_and_errors() {
    let m = mini("opts");
    m.write();
    let full = plan(&m, 4, 1);
    let lean = ink2::plan(&m.dir, 4, 1, &Opts { bf16_gv: false, with_mtp: false, with_vision: false }).unwrap();
    assert!(!lean.img.tensors.iter().any(|t| t.name.ends_with(".gv") || t.name.starts_with("MTP.") || t.name.starts_with("V.")));
    assert!(lean.img.tensors.iter().all(|t| full.img.tensors.iter().any(|u| u.name == t.name && u.shape == t.shape)));
    assert!(!lean.reads.contains_key("mtp.safetensors") && full.reads.contains_key("mtp.safetensors"));
    assert!(lean.img.bytes < full.img.bytes && lean.read_bytes() < full.read_bytes());
    // the manifest names every tensor
    let shas = vec!["0".repeat(64); full.img.tensors.len()];
    let man = json::parse(&full.manifest_json(&shas, 4)).unwrap();
    assert_eq!(man.get("tensors").and_then(Json::arr).unwrap().len(), full.img.tensors.len());
    assert_eq!(man.get("format").and_then(Json::str), Some("g6pack-v1"));
    assert_eq!((man.get("tp").and_then(Json::int), man.get("rank").and_then(Json::int)), (Some(4), Some(1)));

    assert!(plan_err(&m.dir, 4, 4).contains("rank 4"));
    assert!(plan_err(&m.dir, 5, 0).contains("5 ranks"));
    // 768 / 8 = 96 is not a multiple of 64
    assert!(plan_err(&m.dir, 8, 0).contains("multiple of 64"));

    // a tensor the planner does not know is refused, not dropped
    let mut extra = mini("extra");
    extra.shards[1].push(("model.llm.layers.0.surprise".into(), "BF16", vec![4], vec![0; 8]));
    extra.write();
    let e = plan_err(&extra.dir, 4, 0);
    assert!(e.contains("not planned") && e.contains("surprise"), "{e}");
    // a missing tensor
    let mut short = mini("short");
    short.shards[0].retain(|i| i.0 != "model.llm.layers.1.attn.wv_dv.weight");
    short.write();
    assert!(plan_err(&short.dir, 4, 0).contains("wv_dv"));
    // hf_quant_config.json that disagrees with the stored dtypes
    let wrong = mini("wrong");
    wrong.write();
    std::fs::write(format!("{}/hf_quant_config.json", wrong.dir), wrong.quant().replace("\"mxfp8\"", "\"nvfp4\"")).unwrap();
    let e = plan_err(&wrong.dir, 4, 0);
    assert!(e.contains("hf_quant_config.json") && e.contains("layers.2"), "{e}");
    // without the file the dtypes decide and the rule is the default
    std::fs::remove_file(format!("{}/hf_quant_config.json", wrong.dir)).unwrap();
    let p = ink2::plan(&wrong.dir, 4, 0, &Opts::default()).unwrap();
    assert_eq!(p.experts[0], ExpertFormat::Nvfp4(Nvfp4ScaleRule::Static6));
    assert_eq!(p.experts[3], ExpertFormat::Mxfp8);
    // a single model.safetensors instead of an index
    let single = mini("single");
    single.write();
    let sharded = ds41::hash_image(&plan(&single, 6, 5).img, 2).unwrap();
    let all: Vec<Item> = single.shards.iter().flatten().cloned().collect();
    for i in 0..single.shards.len() {
        std::fs::remove_file(format!("{}/{}", single.dir, shard_name(i, single.shards.len()))).unwrap();
    }
    std::fs::remove_file(format!("{}/model.safetensors.index.json", single.dir)).unwrap();
    std::fs::write(format!("{}/model.safetensors", single.dir), st_bytes(&all)).unwrap();
    let a = ink2::plan(&single.dir, 6, 5, &Opts::default()).unwrap();
    assert!(ds41::hash_image(&a.img, 2).unwrap() == sharded, "one file and shards give the same image");
}

/// The reads map is enough: a copy of the checkpoint holding only the listed byte ranges plans and loads to the same
/// image. A rank reads less than the whole checkpoint.
#[test]
fn reads_cover_the_load() {
    let m = mini("reads");
    m.write();
    let whole = plan(&m, 1, 0).read_bytes();
    for (nr, rank) in [(6usize, 3usize), (4, 0)] {
        let p = plan(&m, nr, rank);
        assert!(p.read_bytes() < whole, "nr {nr}");
        let rj = json::parse(&p.reads_json()).unwrap();
        let sparse = format!("{}/sparse-{nr}-{rank}", m.dir);
        std::fs::create_dir_all(&sparse).unwrap();
        for e in std::fs::read_dir(&m.dir).unwrap() {
            let path = e.unwrap().path();
            if !path.is_file() {
                continue;
            }
            let rel = path.file_name().unwrap().to_string_lossy().into_owned();
            let full = std::fs::read(&path).unwrap();
            if !rel.ends_with(".safetensors") {
                std::fs::write(format!("{sparse}/{rel}"), &full).unwrap();
                continue;
            }
            let mut out = vec![0u8; full.len()];
            let runs = p.reads.get(&rel).unwrap_or_else(|| panic!("{rel} is not read"));
            assert_eq!(rj.get("files").and_then(|f| f.get(&rel)).and_then(Json::arr).map(<[Json]>::len), Some(runs.len()), "{rel}");
            assert!(runs.windows(2).all(|w| w[0].0 + w[0].1 < w[1].0), "{rel}: runs are coalesced");
            for &(a, n) in runs {
                out[a as usize..(a + n) as usize].copy_from_slice(&full[a as usize..(a + n) as usize]);
            }
            std::fs::write(format!("{sparse}/{rel}"), out).unwrap();
        }
        let q = ink2::plan(&sparse, nr, rank, &Opts::default()).unwrap();
        assert!(ds41::hash_image(&q.img, 2).unwrap() == ds41::hash_image(&p.img, 2).unwrap(), "nr {nr} rank {rank}");
        assert!(q.reads == p.reads);
    }
}

#[test]
fn c_interface() {
    let m = mini("ffi");
    m.write();
    let p = plan(&m, 6, 4);
    let dir = CString::new(m.dir.clone()).unwrap();
    let out = CString::new(format!("{}/ffi.manifest.json", m.dir)).unwrap();
    let mut err = vec![0 as std::ffi::c_char; 512];
    let h = unsafe { ink2::g6l_ink2_plan(dir.as_ptr(), 6, 4, out.as_ptr(), err.as_mut_ptr(), err.len()) };
    assert!(!h.is_null());
    for name in ["L000.x.fc1", "L003.x.fc2", "L001.wo", "G.lm_head.gv", "MTP.L01.mlp.down_proj.weight"] {
        let t = meta(&p, name);
        let mut buf = vec![0u8; t.nbytes as usize];
        let rc = unsafe { g6load::g6l_plan_read(h, t.offset, t.nbytes, buf.as_mut_ptr() as *mut _, err.as_mut_ptr(), err.len()) };
        assert_eq!(rc, 0);
        assert!(buf == tensor(&p, name), "{name}");
    }
    unsafe { g6load::g6l_plan_free(h) };
    let man = json::parse(&std::fs::read_to_string(out.to_str().unwrap()).unwrap()).unwrap();
    let ts = man.get("tensors").and_then(Json::arr).unwrap();
    assert_eq!(ts.len(), p.img.tensors.len());
    for (a, b) in ts.iter().zip(&p.img.tensors) {
        assert_eq!(a.get("name").and_then(Json::str), Some(b.name.as_str()));
        assert_eq!(a.get("offset").and_then(Json::int), Some(b.offset as i64));
        assert_eq!(a.get("nbytes").and_then(Json::int), Some(b.nbytes as i64));
    }
    assert_eq!(man.get("pack_bytes").and_then(Json::int), Some(p.img.bytes as i64));
    // a bad rank gives NULL and a message
    let h = unsafe { ink2::g6l_ink2_plan(dir.as_ptr(), 6, 9, out.as_ptr(), err.as_mut_ptr(), err.len()) };
    assert!(h.is_null());
    let msg = unsafe { std::ffi::CStr::from_ptr(err.as_ptr()) }.to_string_lossy().into_owned();
    assert!(msg.contains("rank 9"), "{msg}");
}

/// The full-size checkpoint planned from its metadata alone: header-only sparse safetensors files with the real names,
/// dtypes and shapes. Runs when GR8_TEST_INK2_METADATA_DIR names a directory holding config.json, hf_quant_config.json,
/// model.safetensors.index.json and _tensor_shapes.json (name -> {dtype, shape, file}); skipped otherwise.
#[test]
fn real_metadata_plans_every_tensor() {
    let Some(src) = gr8knobs::TEST_INK2_METADATA_DIR.get() else {
        eprintln!("real_metadata_plans_every_tensor: GR8_TEST_INK2_METADATA_DIR is not set, skipped");
        return;
    };
    let dir = std::env::temp_dir().join(format!("g6l-ink2-real-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let dir = dir.to_string_lossy().into_owned();
    for f in ["config.json", "hf_quant_config.json", "model.safetensors.index.json"] {
        std::fs::copy(format!("{src}/{f}"), format!("{dir}/{f}")).unwrap();
    }
    let shapes = json::parse(&std::fs::read_to_string(format!("{src}/_tensor_shapes.json")).unwrap()).unwrap();
    let shapes = shapes.obj().unwrap();
    let mut files: BTreeMap<String, Vec<Entry>> = BTreeMap::new();
    for (name, t) in shapes {
        let dtype = t.get("dtype").and_then(Json::str).unwrap().to_string();
        let shape: Vec<u64> = t.get("shape").and_then(Json::arr).unwrap().iter().map(|d| d.int().unwrap() as u64).collect();
        let e = match dtype.as_str() { "BF16" => 2, "F32" => 4, "U8" | "F8_E4M3" => 1, d => panic!("{name}: dtype {d}") };
        let bytes = shape.iter().product::<u64>() * e;
        files.entry(t.get("file").and_then(Json::str).unwrap().to_string()).or_default().push((name.clone(), dtype, shape, bytes));
    }
    let n_tensors = shapes.len();
    assert_eq!(n_tensors, 1854);
    let mut total = 0u64;
    for (file, ts) in &files {
        let (mut parts, mut off, mut ones) = (Vec::new(), 0u64, Vec::new());
        for (name, dtype, shape, bytes) in ts {
            let sh: Vec<String> = shape.iter().map(u64::to_string).collect();
            parts.push(format!("\"{name}\": {{\"dtype\": \"{dtype}\", \"shape\": [{}], \"data_offsets\": [{off}, {}]}}", sh.join(", "), off + bytes));
            // the outer-scale inputs are read at plan time: 1.0 everywhere
            if name.ends_with("_amax") || name.ends_with("_outer_scale_denominator") || name.ends_with("_scale_2") {
                ones.push((off, *bytes));
            }
            off += bytes;
        }
        let head = format!("{{{}}}", parts.join(", ")).into_bytes();
        let f = std::fs::File::create(format!("{dir}/{file}")).unwrap();
        let base = 8 + head.len() as u64;
        f.set_len(base + off).unwrap();
        use std::os::unix::fs::FileExt;
        f.write_all_at(&(head.len() as u64).to_le_bytes(), 0).unwrap();
        f.write_all_at(&head, 8).unwrap();
        for (a, n) in ones {
            f.write_all_at(&1f32.to_le_bytes().repeat(n as usize / 4), base + a).unwrap();
        }
        total += off;
    }
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    eprintln!("checkpoint: {n_tensors} tensors, {} files, {:.2} GiB of tensor data", files.len(), gib(total));
    for nr in [6usize, 4] {
        for rank in 0..nr {
            let p = ink2::plan(&dir, nr, rank, &Opts::default()).unwrap_or_else(|e| panic!("nr {nr} rank {rank}: {e}"));
            // plan() fails on any checkpoint tensor it did not consume, so success = all 1854 mapped. Count the
            // image the other way round too: the layers, formats and split are the real model's.
            assert_eq!(p.cfg.layers, 48);
            assert_eq!(p.experts.iter().filter(|x| matches!(x, ExpertFormat::Nvfp4(Nvfp4ScaleRule::FourOverSixMae))).count(), 40);
            assert_eq!(p.experts.iter().filter(|x| **x == ExpertFormat::Mxfp8).count(), 8);
            assert!(p.experts[..40].iter().all(|x| *x != ExpertFormat::Mxfp8));
            assert_eq!(p.kv_range.len(), 2);
            assert_eq!(p.vocab_padded, 201216);
            assert_eq!(p.reads.len(), files.len(), "every checkpoint file is read");
            let n = |pre: &str| p.img.tensors.iter().filter(|t| t.name.starts_with(pre)).count();
            assert!(n("V.") > 0 && n("MTP.") == 84 && n("L047.") > 0);
            let s1 = f32s(&ds41::read_tensor(&p.img, "L000.x.s1").unwrap());
            assert!(s1.len() == 384 && s1.iter().all(|&v| v == 1.0));
            let iw = 2304 / nr;
            assert_eq!(meta(&p, "L000.x.fc1").shape, [384, (iw / 16) as i64, 72, 1152]);
            assert_eq!(meta(&p, "L047.x.fc2").shape, [384, 288, (iw / 32) as i64, 528]);
            eprintln!("tp {nr} rank {rank}: {} image tensors, image_bytes {} ({:.2} GiB), read_bytes {} ({:.2} GiB)",
                      p.img.tensors.len(), p.img.bytes, gib(p.img.bytes), p.read_bytes(), gib(p.read_bytes()));
            if nr == 6 {
                assert!((56.0..=66.0).contains(&gib(p.img.bytes)), "TP 6 rank {rank}: {:.2} GiB", gib(p.img.bytes));
            }
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

// ------------------------------------------------------------------------------- execution over the rank images

/// f32 -> BF16 bits, round to nearest even (finite input).
fn bf16(x: f32) -> u16 {
    let u = x.to_bits();
    ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16
}

fn bf16s(b: &[u8]) -> Vec<f32> {
    u16s(b).iter().map(|&h| f32::from_bits((h as u32) << 16)).collect()
}

impl Mini {
    /// Replace the random bytes by small finite weights, so that sums over them mean something: BF16 values in
    /// (-0.15, 0.15), NVFP4 block scales in [0.5, 2) under outer scales near 0.005, MXFP8 codes below 1 under scales
    /// 2^-6 .. 2^-4.
    fn tame(&mut self) {
        let mut rng = Rng(99);
        for (n, d, _, b) in self.shards.iter_mut().flatten() {
            let scale = n.ends_with("_scale");
            match *d {
                "BF16" => {
                    for c in b.chunks_exact_mut(2) {
                        let v = ((rng.next() & 0xffff) as f32 / 32768.0 - 1.0) * 0.15;
                        c.copy_from_slice(&bf16(v).to_le_bytes());
                    }
                }
                "F8_E4M3" if scale => b.iter_mut().for_each(|x| *x = 0x30 + (rng.next() % 16) as u8),
                "F8_E4M3" => b.iter_mut().for_each(|x| *x = (*x & 0x80) | (rng.next() % 0x38) as u8),
                "U8" if scale => b.iter_mut().for_each(|x| *x = 121 + (rng.next() % 3) as u8),
                "F32" if n.ends_with("_amax") || n.ends_with("_scale_2") => {
                    let mul = if n.ends_with("_amax") { DENOM } else { 1.0 };
                    for c in b.chunks_exact_mut(4) {
                        let v = 0.005 * mul * (1.0 + (rng.next() & 0xff) as f32 / 256.0);
                        c.copy_from_slice(&v.to_le_bytes());
                    }
                }
                _ => {}
            }
        }
        self.t.clear();
        for (n, _, sh, b) in self.shards.iter().flatten() {
            self.t.insert(n.clone(), (sh.clone(), b.clone()));
        }
        for (n, _, sh, b) in &self.mtp {
            self.t.insert(format!("mtp.{n}"), (sh.clone(), b.clone()));
        }
    }
}

fn dot64(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum()
}

fn dot32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0f32, |s, (&x, &y)| s + x * y)
}

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// One MXFP8 record -> [16][32] values (the inverse of records::mxfp8_record's lane order).
fn mx_decode(rec: &[u8]) -> [[f32; 32]; 16] {
    let mut out = [[0f32; 32]; 16];
    for h in 0..2 {
        for i in 0..256 {
            let (lane, j) = (i / 8, i % 8);
            let (r, c) = (lane / 4 + ((j >> 1) & 1) * 8, 2 * (lane % 4) + (j & 1) + (j >> 2) * 8);
            let s = if r < 8 { rec[512 + 2 * r] } else { rec[512 + 2 * (r - 8) + 1] };
            out[r][16 * h + c] = g6load::exlr8::e4m3_to_f32(rec[h * 256 + i]) * 2f32.powi(s as i32 - 127);
        }
    }
    out
}

/// Expert `e` of layer `l` from a rank image's records, dequantized: (gate [iw][HID], up [iw][HID], down [HID][iw]).
fn expert_dense(p: &Ink2Plan, fc1: &[u8], fc2: &[u8], l: usize, e: usize, iw: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (per1, per2) = (fc1.len() / NEXP, fc2.len() / NEXP);
    let (fc1, fc2) = (&fc1[e * per1..][..per1], &fc2[e * per2..][..per2]);
    let (mut gate, mut up, mut down) = (vec![0f32; iw * HID], vec![0f32; iw * HID], vec![0f32; HID * iw]);
    if KIND[l] < 2 {
        let s1 = f32s(&tensor(p, &format!("L{l:03}.x.s1")))[e];
        let s2 = f32s(&tensor(p, &format!("L{l:03}.x.s2")))[e];
        for (i, rec) in fc1.chunks_exact(576).enumerate() {
            let (ch, g, dst) = (i / 2 / (HID / 64), i / 2 % (HID / 64), if i % 2 == 0 { &mut gate } else { &mut up });
            for (r, row) in nvfp4::nvfp4_decode_group(rec.try_into().unwrap(), s1).iter().enumerate() {
                dst[(ch * 16 + r) * HID + g * 64..][..64].copy_from_slice(row);
            }
        }
        for (i, rec) in fc2.chunks_exact(576).enumerate() {
            let (ot, g) = (i / (iw / 64), i % (iw / 64));
            for (r, row) in nvfp4::nvfp4_decode_group(rec.try_into().unwrap(), s2).iter().enumerate() {
                down[(ot * 16 + r) * iw + g * 64..][..64].copy_from_slice(row);
            }
        }
    } else {
        for (i, rec) in fc1.chunks_exact(528).enumerate() {
            let (ch, g, dst) = (i / 2 / (HID / 32), i / 2 % (HID / 32), if i % 2 == 0 { &mut gate } else { &mut up });
            for (r, row) in mx_decode(rec).iter().enumerate() {
                dst[(ch * 16 + r) * HID + g * 32..][..32].copy_from_slice(row);
            }
        }
        for (i, rec) in fc2.chunks_exact(528).enumerate() {
            let (ot, g) = (i / (iw / 32), i % (iw / 32));
            for (r, row) in mx_decode(rec).iter().enumerate() {
                down[(ot * 16 + r) * iw + g * 32..][..32].copy_from_slice(row);
            }
        }
    }
    (gate, up, down)
}

/// Quantize then dequantize an activation slice in the layer's format.
fn requantize(l: usize, a: &[f32]) -> Vec<f32> {
    if KIND[l] < 2 {
        let (c, s, outer) = nvfp4::nvfp4_quantize(a, a.len(), Nvfp4ScaleRule::FourOverSixMae, None);
        nvfp4::nvfp4_dequantize(&c, &s, outer)
    } else {
        let (c, s) = records::mxfp8_quantize(a, a.len());
        c.iter().enumerate().map(|(i, &b)| g6load::exlr8::e4m3_to_f32(b) * 2f32.powi(s[i / 32] as i32 - 127)).collect()
    }
}

/// The attention of one rank over `x` [T][HID] with the tensors of its image: its q heads over its K / V heads
/// (causal, with the relative logits, the sink and the output gate), then its columns of wo. The arithmetic of the
/// reference without the convolutions and the BF16 rounding, which do not depend on the split. Returns [T][HID].
fn rank_attention(p: &Ink2Plan, l: usize, x: &[f32]) -> Vec<f64> {
    let w = |n: &str| bf16s(&tensor(p, &format!("L{l:03}.{n}")));
    let (wq, wr, wg, ws, wk, wv, wo) = (w("wq"), w("wr"), w("gate_attn"), w("attn_sink_proj"), w("wk"), w("wv"), w("wo"));
    let (qc, rc, proj) = (w("q_l2_clip"), w("rel_l2_clip"), w("rel_logits_proj"));
    let (nq, nkv, t_n) = (p.q_range.len(), p.kv_range.len(), x.len() / HID);
    assert_eq!((wq.len(), wk.len(), wo.len(), ws.len()), (nq * HD * HID, nkv * HD * HID, HID * nq * HD, nq * HID));
    let clip = |v: &mut [f32], max: f32| {
        let f = (max / (dot32(v, v).sqrt() + 1e-6)).min(1.0);
        v.iter_mut().for_each(|y| *y *= f);
    };
    // K and V of every position and local K / V head
    let (mut k, mut v) = (vec![0f32; t_n * nkv * HD], vec![0f32; t_n * nkv * HD]);
    for t in 0..t_n {
        let xt = &x[t * HID..][..HID];
        for h in 0..nkv {
            let (kh, vh) = (&mut k[(t * nkv + h) * HD..][..HD], &mut v[(t * nkv + h) * HD..][..HD]);
            for d in 0..HD {
                kh[d] = dot32(&wk[(h * HD + d) * HID..][..HID], xt);
                vh[d] = dot32(&wv[(h * HD + d) * HID..][..HID], xt);
            }
            clip(kh, 2.0 * (HD as f32).sqrt());
            let r = 1.0 / (dot32(vh, vh) / HD as f32 + 1e-6).sqrt();
            vh.iter_mut().for_each(|y| *y *= r);
        }
    }
    let mut out = vec![0f64; t_n * HID];
    for t in 0..t_n {
        let xt = &x[t * HID..][..HID];
        let mut og = vec![0f32; nq * HD];
        for h in 0..nq {
            let kvh = (p.q_range.start + h) / (HEADS / KV) - p.kv_range.start;
            let mut q: Vec<f32> = (0..HD).map(|d| dot32(&wq[(h * HD + d) * HID..][..HID], xt) * qc[h * HD + d]).collect();
            clip(&mut q, 2.0 * (HD as f32).sqrt());
            let mut r: Vec<f32> = (0..REL).map(|d| dot32(&wr[(h * REL + d) * HID..][..HID], xt) * rc[h * REL + d]).collect();
            clip(&mut r, 2.0 * (REL as f32).sqrt());
            let sink = dot32(&ws[h * HID..][..HID], xt);
            let score: Vec<f32> = (0..=t).map(|kp| {
                let rel: f32 = (0..REL).map(|d| r[d] * proj[d * RELWIN + (t - kp)]).sum();
                dot32(&q, &k[(kp * nkv + kvh) * HD..][..HD]) / (HD as f32).sqrt() + rel
            }).collect();
            let m = score.iter().fold(sink, |a, &b| a.max(b));
            let den = (sink - m).exp() + score.iter().map(|s| (s - m).exp()).sum::<f32>();
            for d in 0..HD {
                let o: f32 = (0..=t).map(|kp| (score[kp] - m).exp() / den * v[(kp * nkv + kvh) * HD + d]).sum();
                og[h * HD + d] = o * sigmoid(dot32(&wg[(h * HD + d) * HID..][..HID], xt));
            }
        }
        for c in 0..HID {
            out[t * HID + c] = dot64(&wo[c * nq * HD..][..nq * HD], &og);
        }
    }
    out
}

/// The routed experts of one rank for one token: the router on the replicated tensors, fc1 and the activation on the
/// rank's intermediate slice, fc2 over that slice. `shared` = the whole activation of each picked expert, quantized
/// once by one rank. Returns (picked experts, the raw activation slices [picked][iw], the output [HID] with the rank's
/// own quantization of its slice, the output with its slice of `shared`).
type Moe = (Vec<usize>, Vec<Vec<f32>>, Vec<f64>, Vec<f64>);
fn rank_moe(p: &Ink2Plan, l: usize, nr: usize, rank: usize, x: &[f32], shared: Option<&[Vec<f32>]>) -> Moe {
    let (iw, at) = (INTER / nr, rank * (INTER / nr));
    let (rw, rb) = (bf16s(&tensor(p, &format!("L{l:03}.router.w"))), f32s(&tensor(p, &format!("L{l:03}.router.bias"))));
    let score: Vec<f32> = (0..NEXP).map(|e| sigmoid(dot32(&rw[e * HID..][..HID], x))).collect();
    let mut order: Vec<usize> = (0..NEXP).collect();
    order.sort_by(|&a, &b| (score[b] + rb[b]).total_cmp(&(score[a] + rb[a])).then(a.cmp(&b)));
    let picked = order[..4].to_vec();
    let sum: f32 = picked.iter().map(|&e| score[e]).sum();
    let xq = requantize(l, x);
    let (mut acts, mut out, mut out_shared) = (Vec::new(), vec![0f64; HID], vec![0f64; HID]);
    let (fc1, fc2) = (tensor(p, &format!("L{l:03}.x.fc1")), tensor(p, &format!("L{l:03}.x.fc2")));
    for (k, &e) in picked.iter().enumerate() {
        let (gate, up, down) = expert_dense(p, &fc1, &fc2, l, e, iw);
        let a: Vec<f32> = (0..iw).map(|i| {
            let (g, u) = (dot32(&gate[i * HID..][..HID], &xq), dot32(&up[i * HID..][..HID], &xq));
            (4.0 * (g / 4.0).tanh()) * sigmoid(g) * (25.0 * (u / 25.0).tanh())
        }).collect();
        let (aq, wt) = (requantize(l, &a), (score[e] / (sum + 1e-20)) as f64);
        for c in 0..HID {
            out[c] += wt * dot64(&down[c * iw..][..iw], &aq);
            if let Some(s) = shared {
                out_shared[c] += wt * dot64(&down[c * iw..][..iw], &s[k][at..at + iw]);
            }
        }
        acts.push(a);
    }
    (picked, acts, out, out_shared)
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).fold(0f64, |m, (x, y)| m.max((x - y).abs()))
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let d = |x: &[f64], y: &[f64]| x.iter().zip(y).map(|(p, q)| p * q).sum::<f64>();
    d(a, b) / (d(a, a) * d(b, b)).sqrt()
}

/// The checkpoint planned at TP 4 and TP 6, every rank's tensors read back from its image and run: the ranks' attention
/// and expert outputs add up to the TP 1 image's.
#[test]
fn rank_images_add_up_to_one_rank() {
    let mut m = mini("sum");
    m.tame();
    m.write();
    let mut rng = Rng(7);
    const T: usize = 4;
    let x: Vec<f32> = (0..T * HID).map(|_| f32::from_bits((bf16((rng.next() & 0xffff) as f32 / 32768.0 - 1.0) as u32) << 16)).collect();
    let one = plan(&m, 1, 0);
    let mut low = 1f64;
    for (l, &kind) in KIND.iter().enumerate() {
        let att1 = rank_attention(&one, l, &x);
        assert!(att1.iter().any(|v| v.abs() > 1e-3) && att1.iter().all(|v| v.is_finite()), "L{l}: attention output");
        // per token: (picked, activations, output with the rank's own quantization)
        let moe1: Vec<_> = (0..T).map(|t| rank_moe(&one, l, 1, 0, &x[t * HID..][..HID], None)).collect();
        for nr in [4usize, 6] {
            let plans: Vec<Ink2Plan> = (0..nr).map(|r| plan(&m, nr, r)).collect();
            // attention: the q heads of all ranks cover the heads once, and the wo partial sums add up
            let mut att = vec![0f64; T * HID];
            for p in &plans {
                att.iter_mut().zip(rank_attention(p, l, &x)).for_each(|(a, b)| *a += b);
            }
            assert!(max_diff(&att, &att1) < 1e-5, "L{l} TP{nr}: attention differs by {}", max_diff(&att, &att1));
            for t in 0..T {
                let xt = &x[t * HID..][..HID];
                let (picked1, acts1, out1, _) = &moe1[t];
                assert!(out1.iter().any(|v| v.abs() > 1e-3) && out1.iter().all(|v| v.is_finite()), "L{l}: expert output");
                // the activation of one rank, quantized once, shared by every rank: the same sum
                let shared: Vec<Vec<f32>> = acts1.iter().map(|a| requantize(l, a)).collect();
                let (mut sum_shared, mut sum_own) = (vec![0f64; HID], vec![0f64; HID]);
                let mut acts: Vec<Vec<f32>> = vec![Vec::new(); 4];
                for (r, p) in plans.iter().enumerate() {
                    let (picked, a, own, sh) = rank_moe(p, l, nr, r, xt, Some(&shared));
                    assert_eq!(&picked, picked1, "L{l} TP{nr} rank {r}: the router");
                    sum_shared.iter_mut().zip(sh).for_each(|(s, v)| *s += v);
                    sum_own.iter_mut().zip(own).for_each(|(s, v)| *s += v);
                    acts.iter_mut().zip(a).for_each(|(all, part)| all.extend(part));
                }
                // fc1 and the activation of the slices are those of the whole expert, bit for bit
                assert!(acts.iter().zip(acts1).all(|(a, b)| a.iter().zip(b).all(|(p, q)| p.to_bits() == q.to_bits())), "L{l} TP{nr}: fc1 slices");
                assert!(max_diff(&sum_shared, out1) < 1e-5, "L{l} TP{nr} token {t}: fc2 differs by {}", max_diff(&sum_shared, out1));
                // each rank quantizing its own slice: MXFP8 blocks do not cross a slice, so nothing changes; an NVFP4
                // slice has its own outer scale, so the block scales and codes move a little
                let cs = cosine(&sum_own, out1);
                if kind == 2 {
                    assert!(max_diff(&sum_own, out1) < 1e-5 && cs > 0.99999, "L{l} TP{nr} token {t}: MXFP8 cosine {cs}");
                } else {
                    assert!(cs > 0.99, "L{l} TP{nr} token {t}: NVFP4 cosine {cs}");
                    low = low.min(cs);
                }
            }
        }
    }
    println!("rank_images_add_up_to_one_rank: lowest NVFP4 cosine with per-rank activation scales {low:.6}");
}
