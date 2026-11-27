#!/usr/bin/env python3
"""Verify engine/model_geom_hy4.h against Hy4-flash config.json AND all 1,964 checkpoint tensor shapes."""
import json, os, re, struct, sys

BANK = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("HY4_MODEL_DIR", "")
if not BANK:
    print("usage: test_hy4_geom.py <hf_model_dir> (or set HY4_MODEL_DIR)", file=sys.stderr)
    sys.exit(2)
HDR = os.path.join(os.path.dirname(__file__), "..", "model_geom_hy4.h")

with open(HDR) as f:
    src = f.read()

def get_int(name):
    m = re.search(rf"constexpr\s+int\s+{name}\s*=\s*([0-9]+)\s*;", src)
    assert m, f"missing int {name}"
    return int(m.group(1))

def get_float(name):
    m = re.search(rf"constexpr\s+float\s+{name}\s*=\s*([0-9eE.+\-]+)f?\s*;", src)
    assert m, f"missing float {name}"
    return float(m.group(1))

with open(os.path.join(BANK, "config.json")) as f:
    cfg = json.load(f)
tc = cfg["text_config"]

assert get_int("HID") == tc["hidden_size"]
assert get_int("VOCAB") == tc["vocab_size"]
assert get_int("L_MAIN") == tc["num_hidden_layers"]
assert get_int("FIRST_MOE") == tc["first_k_dense_replace"]
assert get_int("DENSE_I") == tc["intermediate_size"]
assert get_int("MAX_POS") == tc["max_position_embeddings"]
assert get_int("HC") == tc["hc_mult"]
assert abs(get_float("HC_EPS") - tc["hc_eps"]) < 1e-12
assert abs(get_float("HC_MAGNITUDE") - tc["hc_magnitude"]) < 1e-9
assert abs(get_float("RMS_EPS") - tc["rms_norm_eps"]) < 1e-12
assert get_int("HEADS_DSA") == tc["num_attention_heads"]
assert get_int("HEADS_SWA") == tc["swa_num_attention_heads"]
assert get_int("SWA_WINDOW") == tc["sliding_window"]
assert get_int("QA") == tc["q_lora_rank"]
assert get_int("KV_LORA") == tc["kv_lora_rank"]
assert get_int("QK_NOPE") == tc["qk_nope_head_dim"] == tc["swa_qk_nope_head_dim"]
assert get_int("QK_ROPE") == tc["qk_rope_head_dim"] == tc["swa_qk_rope_head_dim"]
assert get_int("V_HEAD") == tc["v_head_dim"] == tc["swa_v_head_dim"]
assert get_int("IDX_HEADS") == tc["compressed_indexer_num_attention_heads"]
assert get_int("IDX_DIM") == tc["compressed_indexer_head_dim"]
assert get_int("IDX_RATIO") == tc["compressed_indexer_compress_ratio"]
assert get_int("IDX_GROUP_TOPK") == tc["compressed_indexer_group_topk"]
assert get_int("IDX_TOPK") == tc["compressed_indexer_candidate_topk"]
assert get_int("NEXP") == tc["n_routed_experts"]
assert get_int("TOPK") == tc["num_experts_per_tok"]
assert get_int("EH") == tc["routed_expert_hidden_size"]
assert get_int("EXP_I") == tc["moe_intermediate_size"]
assert abs(get_float("ROUTE_SCALE") - tc["routed_scaling_factor"]) < 1e-6
assert abs(get_float("SWIGLU_LIMIT") - tc["swiglu_limit"]) < 1e-6
assert get_int("ENGRAM_MAX_NGRAM") == tc["engram_max_ngram_size"]
assert get_int("ENGRAM_HEADS_PER_NGRAM") == tc["engram_n_head_per_ngram"]
assert get_int("ENGRAM_PAD_ID") == tc["engram_pad_id"]
assert get_int("ENGRAM_SEED") == tc["engram_seed"]
assert get_int("TOK_BOS") == tc["bos_token_id"]
assert get_int("TOK_PAD") == tc["pad_token_id"]
assert get_int("TOK_EOS") == tc["eos_token_id"]

# Verify layer_types matches is_dsa/is_swa
for l, lt in enumerate(tc["layer_types"]):
    is_d = (l == 0) or (l >= 1 and (l - 1) % 4 == 0)
    assert (lt == "deepseek_sparse_attention") == is_d, f"layer {l}: {lt} vs is_d={is_d}"
assert tc["engram_layer_ids"] == [2, 12, 22, 32]

# Check all 1,964 checkpoint tensor shapes against geometry!
with open(os.path.join(BANK, "model.safetensors.index.json")) as f:
    wmap = json.load(f)["weight_map"]
assert len(wmap) == 1964, f"expected 1964 tensors, got {len(wmap)}"

shapes = {}
dtypes = {}
for fn in sorted(set(wmap.values())):
    with open(os.path.join(BANK, fn), "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
    for k, v in hdr.items():
        if k == "__metadata__": continue
        shapes[k] = tuple(v["shape"])
        dtypes[k] = v["dtype"]

assert len(shapes) == 1964
# Verify every text tensor shape
assert shapes["model.embed_tokens.weight"] == (120832, 3072)
assert shapes["lm_head.weight"] == (120832, 3072)
assert shapes["model.norm.weight"] == (3072,)
assert shapes["model.hc_head.hc_head_fn"] == (4, 12288) and dtypes["model.hc_head.hc_head_fn"] == "F32"
assert shapes["model.hc_head.hc_head_base"] == (4,) and dtypes["model.hc_head.hc_head_base"] == "F32"
assert shapes["model.hc_head.hc_head_scale"] == (1,) and dtypes["model.hc_head.hc_head_scale"] == "F32"

checked = 6
for l in range(42):
    p = f"model.layers.{l}."
    is_d = (l == 0) or ((l - 1) % 4 == 0)
    nh = 32 if is_d else 64
    expect = {
        p + "input_layernorm.weight": ((3072,), "BF16"),
        p + "post_attention_layernorm.weight": ((3072,), "BF16"),
        p + "hc_attn_layer.hc_pre.hc_fn": ((8, 12288), "F32"),
        p + "hc_attn_layer.hc_pre.hc_base": ((8,), "F32"),
        p + "hc_attn_layer.hc_pre.hc_scale": ((2,), "F32"),
        p + "hc_mlp_layer.hc_pre.hc_fn": ((8, 12288), "F32"),
        p + "hc_mlp_layer.hc_pre.hc_base": ((8,), "F32"),
        p + "hc_mlp_layer.hc_pre.hc_scale": ((2,), "F32"),
        p + "self_attn.q_a_proj.weight": ((1024, 3072), "BF16"),
        p + "self_attn.q_a_layernorm.weight": ((1024,), "BF16"),
        p + "self_attn.q_b_proj.weight": ((nh * 256, 1024), "BF16"),
        p + "self_attn.kv_a_proj_with_mqa.weight": ((576, 3072), "BF16"),
        p + "self_attn.kv_a_layernorm.weight": ((512,), "BF16"),
        p + "self_attn.kv_b_proj.weight": ((nh * 448, 512), "BF16"),
        p + "self_attn.linear_gate.weight": ((nh * 256, 3072), "BF16"),
        p + "self_attn.o_proj.weight": ((3072, nh * 256), "BF16"),
        p + "self_attn.post_proj_norm.weight": ((3072,), "BF16"),
        p + "self_attn.learnable_sink_param": ((nh,), "F32"),
    }
    if is_d:
        ip = p + "self_attn.compressed_indexer."
        expect.update({
            ip + "query_projection.weight": ((2048, 1024), "BF16"),
            ip + "head_weight_projection.weight": ((16, 3072), "BF16"),
            ip + "token_key_projection.weight": ((128, 3072), "BF16"),
            ip + "token_key_norm.weight": ((128,), "BF16"),
            ip + "key_value_projection.weight": ((128, 128), "F32"),
            ip + "key_gate_projection.weight": ((128, 128), "F32"),
            ip + "key_norm.weight": ((128,), "F32"),
            ip + "intra_group_ape": ((4, 128), "F32"),
        })
    if l == 0:
        expect.update({
            p + "mlp.gate_proj.weight": ((13824, 3072), "BF16"),
            p + "mlp.up_proj.weight": ((13824, 3072), "BF16"),
            p + "mlp.down_proj.weight": ((3072, 13824), "BF16"),
            p + "mlp.post_fc2_norm.weight": ((3072,), "BF16"),
        })
    else:
        expect.update({
            p + "mlp.gate.weight": ((512, 3072), "BF16"),
            p + "mlp.gate.e_score_correction_bias": ((512,), "F32"),
            p + "mlp.routed_expert_down_proj.weight": ((1536, 3072), "BF16"),
            p + "mlp.routed_expert_up_proj.weight": ((3072, 1536), "BF16"),
            p + "mlp.combine_output_norm.weight": ((3072,), "BF16"),
            p + "mlp.experts.gate_up_proj": ((512, 3072, 1536), "BF16"),
            p + "mlp.experts.down_proj": ((512, 1536, 1536), "BF16"),
            p + "mlp.shared_experts.gate_proj.weight": ((1536, 3072), "BF16"),
            p + "mlp.shared_experts.up_proj.weight": ((1536, 3072), "BF16"),
            p + "mlp.shared_experts.down_proj.weight": ((3072, 1536), "BF16"),
        })
    if l in (1, 11, 21, 31):
        ep = p + "engram_layer."
        lid = str(l + 1)
        tot_rows = sum(sum(o) for o in tc["engram_layer_head_vocab_sizes"][lid])
        expect.update({
            ep + "lookup_table": ((120818,), "I32"),
            ep + "multi_head_embedding.embedding.weight": ((tot_rows, 96), "BF16"),
            ep + "value_proj.weight": ((3072, 1536), "BF16"),
            ep + "value_proj.bias": ((3072,), "BF16"),
            ep + "unified_pre_gamma": ((3072,), "BF16"),
        })
        for h in range(16):
            expect[ep + f"lookup_per_head_norms.{h}.weight"] = ((96,), "BF16")
        for hc in range(4):
            expect[ep + f"key_projs.{hc}.weight"] = ((3072, 1536), "BF16")
            expect[ep + f"key_projs.{hc}.bias"] = ((3072,), "BF16")
            expect[ep + f"norm_key.{hc}.weight"] = ((3072,), "BF16")
            expect[ep + f"norm_query.{hc}.weight"] = ((3072,), "BF16")
            expect[ep + f"unified_per_hc_scales.{hc}"] = ((3072,), "BF16")
    for k, (sh, dt) in expect.items():
        assert k in shapes, f"missing {k}"
        assert shapes[k] == sh, f"{k}: {shapes[k]} != {sh}"
        assert dtypes[k] == dt, f"{k}: {dtypes[k]} != {dt}"
        checked += 1

vit_n = sum(1 for k in shapes if k.startswith("vit."))
dfly_n = sum(1 for k in shapes if "dfly_layers" in k or "confidence_head" in k)
assert checked + vit_n + dfly_n == 1964, f"unaccounted tensors: {checked}+{vit_n}+{dfly_n} != 1964"
print(f"test_hy4_geom.py: OK ({checked} text tensors + {vit_n} vit + {dfly_n} dfly = 1964/1964 exact!)")
