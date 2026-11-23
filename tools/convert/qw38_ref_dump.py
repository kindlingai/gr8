#!/usr/bin/env python3
"""Dump exact BF16 reference activations from HuggingFace Qwen4ExpTextModel on real checkpoint weights
for per-block CUDA kernel verification on GB10:
  - Layer 0: Gated DeltaNet + Hyper-Connections + MoE (512 experts + gated shared)
  - Layer 1: Layer-1 PLE (real NVMe n-gram shards) + GDN + MoE
  - Layer 3: Full Attention + QSA Block-4 Indexer (both short seq=16 dense AND long seq=2200 sparse!) + MoE
  - Final HC mixer
"""
import argparse, json, os, struct, time
import numpy as np
import torch
from safetensors import safe_open
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextDecoderLayer,
    Qwen4ExpTextGatedResidual,
    Qwen4ExpTextRotaryEmbedding,
)


def load_tensors(bf16_dir, index_map, prefixes):
    by_file = {}
    for k, fn in index_map.items():
        if any(k.startswith(p) for p in prefixes):
            by_file.setdefault(fn, []).append(k)
    out = {}
    for fn, ks in sorted(by_file.items()):
        with safe_open(os.path.join(bf16_dir, fn), framework="pt", device="cpu") as f:
            for k in ks:
                out[k] = f.get_tensor(k)
    return out


def save_bin(path, t: torch.Tensor):
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        arr = t.view(torch.uint16).numpy()
    elif t.dtype == torch.float32:
        arr = t.numpy()
    elif t.dtype == torch.int32:
        arr = t.numpy()
    elif t.dtype == torch.int64:
        arr = t.numpy()
    else:
        raise ValueError(t.dtype)
    with open(path, "wb") as f:
        f.write(arr.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)

    raw_cfg = json.load(open(os.path.join(args.bf16, "config.json")))
    tc = raw_cfg["text_config"]
    cfg = Qwen4ExpTextConfig(**tc)
    cfg._attn_implementation = "eager"
    idx_map = json.load(open(os.path.join(args.bf16, "model.safetensors.index.json")))["weight_map"]

    # Deterministic 16-token prompt including EOS token to test _shift_right_ignore_eos!
    toks = [248044, 785, 6722, 315, 9625, 374, 12095, 13, 248044, 42, 100, 999, 5000, 12345, 77777, 200000]
    input_ids = torch.tensor([toks], dtype=torch.long, device=dev)
    save_bin(os.path.join(args.out, "toks.i32"), torch.tensor(toks, dtype=torch.int32))

    # Load embed_tokens and final HC mixer
    print("Loading embed + layers 0, 1, 3 + final mixer...")
    t0 = time.time()
    base_w = load_tensors(
        args.bf16,
        idx_map,
        [
            "model.language_model.embed_tokens.",
            "model.language_model.hyper_connection_mixer.",
            "model.language_model.layers.0.",
            "model.language_model.layers.3.",
        ],
    )
    # Layer 1 non-shard tensors (skip 128 huge PLE shards; we fetch only needed rows directly!)
    l1_keys = [k for k in idx_map if k.startswith("model.language_model.layers.1.") and ".shard_" not in k]
    by_f = {}
    for k in l1_keys:
        by_f.setdefault(idx_map[k], []).append(k)
    for fn, ks in by_f.items():
        with safe_open(os.path.join(args.bf16, fn), framework="pt", device="cpu") as f:
            for k in ks:
                base_w[k] = f.get_tensor(k)
    print(f"Loaded {len(base_w)} tensors in {time.time() - t0:.1f}s")

    # Build embed output
    emb_w = base_w["model.language_model.embed_tokens.weight"]
    x_emb = emb_w[torch.tensor(toks)].to(dev, dtype=torch.bfloat16).unsqueeze(0)  # [1, 16, 2560]
    hyper0 = x_emb.repeat(1, 1, 4)  # [1, 16, 10240]
    save_bin(os.path.join(args.out, "hyper_in.bf16"), hyper0[0])

    # Build rotary embeddings for seq=16
    rotary = Qwen4ExpTextRotaryEmbedding(cfg).to(dev)
    pos_ids = torch.arange(len(toks), device=dev).view(1, 1, -1).expand(3, 1, -1)
    pos_emb = rotary(hyper0, pos_ids)

    # Helper to load one decoder layer (avoid allocating 95 GiB PLE embedding table!)
    def make_layer(lidx):
        with torch.device("meta"):
            layer = Qwen4ExpTextDecoderLayer(cfg, lidx)
        pref = f"model.language_model.layers.{lidx}."
        sd = {}
        for k, v in base_w.items():
            if k.startswith(pref):
                sd[k[len(pref):]] = v
        # Keep ngram_embedding as tiny dummy so to_empty/load_state_dict never allocates 190 GiB
        if layer.ple is not None:
            layer.ple.ple_embedding.ngram_embedding = torch.nn.Embedding(1, 160)
        missing, unexpected = layer.load_state_dict(sd, strict=False, assign=True)
        missing = [m for m in missing if "ngram_embedding.weight" not in m]
        assert not missing and not unexpected, f"L{lidx}: missing={missing}, unexpected={unexpected}"
        return layer.to(dev, dtype=torch.bfloat16).eval()

    with torch.no_grad():
        # --- Layer 0: GDN + HC + MoE ---
        l0 = make_layer(0)
        m0_in, h0_orig, inj0 = l0.attn_hyper_connection(hyper0)
        save_bin(os.path.join(args.out, "l0_hc_mixed.bf16"), m0_in[0])
        save_bin(os.path.join(args.out, "l0_hc_inj.f32"), inj0[0].float())

        gdn0_out = l0.linear_attn(m0_in)
        save_bin(os.path.join(args.out, "l0_gdn_out.bf16"), gdn0_out[0])

        h0_mid = h0_orig + (gdn0_out.unsqueeze(-2) * inj0.unsqueeze(-1)).flatten(-2)
        m0_mlp_in, h0_mid2, inj0_m = l0.mlp_hyper_connection(h0_mid)
        moe0_out = l0.mlp(m0_mlp_in)
        save_bin(os.path.join(args.out, "l0_mlp_in.bf16"), m0_mlp_in[0])
        save_bin(os.path.join(args.out, "l0_moe_out.bf16"), moe0_out[0])
        h0_out = h0_mid2 + (moe0_out.unsqueeze(-2) * inj0_m.unsqueeze(-1)).flatten(-2)
        save_bin(os.path.join(args.out, "l0_out.bf16"), h0_out[0])
        del l0
        torch.cuda.empty_cache()

        # --- Layer 1: PLE (exact n-gram lookup from real shards!) ---
        l1 = make_layer(1)
        ple_emb_mod = l1.ple.ple_embedding
        # Compute exact row IDs using HF logic
        eos_id = ple_emb_mod.eos_token_id
        prev = input_ids.new_full((1, 2), eos_id)
        hist = torch.cat([prev, input_ids], dim=-1)
        shifted = [ple_emb_mod._shift_right_ignore_eos(hist, s) for s in range(3)]
        mult = ple_emb_mod.layer_multipliers.to(dev)
        vs = ple_emb_mod.ngram_heads_vocab_sizes.to(dev)
        offs = ple_emb_mod.ngram_heads_offsets.to(dev)
        blocks = []
        for ng in (2, 3):
            s0 = (ng - 2) * 8
            mixed = shifted[0] * mult[0]
            for p in range(1, ng):
                mixed = torch.bitwise_xor(mixed, shifted[p] * mult[p])
            nids = torch.remainder(mixed.unsqueeze(-1), vs[s0:s0 + 8].view(1, 1, -1)) + offs[s0:s0 + 8].view(1, 1, -1)
            blocks.append(nids)
        ngram_ids = torch.cat(blocks, dim=-1)[:, -len(toks):]  # [1, 16, 16]
        save_bin(os.path.join(args.out, "ple_ids.i64"), ngram_ids[0].to(torch.int64))

        # Fetch exact rows from the 128 BF16 shards (same concatenation order as qw38_release.py)
        shard_bounds = []
        cur = 0
        for s in range(128):
            sk = f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{s}.weight"
            fn = idx_map[sk]
            with open(os.path.join(args.bf16, fn), "rb") as f:
                hl = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(hl))
            rows = hdr[sk]["shape"][0]
            shard_bounds.append((cur, cur + rows, fn, sk))
            cur += rows

        flat_ids = ngram_ids.view(-1).cpu().tolist()
        fetched = torch.empty((len(flat_ids), 160), dtype=torch.bfloat16)
        by_shard = {}
        for pos_i, rid in enumerate(flat_ids):
            for s, (lo, hi, fn, sk) in enumerate(shard_bounds):
                if lo <= rid < hi:
                    by_shard.setdefault((fn, sk, lo), []).append((pos_i, rid - lo))
                    break
        for (fn, sk, lo), items in by_shard.items():
            with safe_open(os.path.join(args.bf16, fn), framework="pt", device="cpu") as f:
                sl = f.get_slice(sk)
                for pos_i, local_r in items:
                    fetched[pos_i] = sl[local_r : local_r + 1][0]
        ple_emb = fetched.view(1, len(toks), 2560).to(dev)
        save_bin(os.path.join(args.out, "ple_emb.bf16"), ple_emb[0])

        # Run PLE forward with the fetched embedding directly
        ple = l1.ple
        key_n = ple.norm_key(ple.key_proj(ple_emb)).unflatten(-1, (4, 2560))
        val = ple.value_proj(ple_emb)
        q_n = ple.norm_query(h0_out).unflatten(-1, (4, 2560))
        gate = (key_n * q_n).sum(dim=-1, keepdim=True) / (2560 ** 0.5)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gv = torch.sigmoid(gate) * val.unsqueeze(-2)
        gv_n = ple.norm_conv(gv.flatten(-2))
        gv_f = gv.flatten(-2)
        ple_out = gv_f + ple._short_conv(gv_n, None)
        h1_after_ple = h0_out + ple_out
        save_bin(os.path.join(args.out, "l1_after_ple.bf16"), h1_after_ple[0])
        del l1
        torch.cuda.empty_cache()

        # --- Layer 3: Full Attention + QSA (short seq=16 dense + long seq=2200 sparse!) ---
        l3 = make_layer(3)
        m3_in, _, _ = l3.attn_hyper_connection(h1_after_ple)
        save_bin(os.path.join(args.out, "l3_attn_in.bf16"), m3_in[0])
        S = len(toks)
        causal_mask = torch.zeros((1, 1, S, S), device=dev, dtype=torch.bfloat16)
        causal_mask.masked_fill_(torch.triu(torch.ones((S, S), device=dev, dtype=torch.bool), 1), torch.finfo(torch.bfloat16).min)
        attn3_out, _ = l3.self_attn(m3_in, pos_emb, causal_mask, cache=None)
        save_bin(os.path.join(args.out, "l3_attn_out.bf16"), attn3_out[0])

        # Long sequence QSA test (S_long = 2200 > 2048 budget!): verify sparse block-4 selection!
        torch.manual_seed(42)
        S_long = 2200
        x_long = (torch.randn((1, S_long, 2560), device=dev, dtype=torch.bfloat16) * 0.1)
        pos_long = torch.arange(S_long, device=dev).view(1, 1, -1).expand(3, 1, -1)
        pe_long = rotary(x_long, pos_long)
        cm_long = torch.zeros((1, 1, S_long, S_long), device=dev, dtype=torch.bfloat16)
        cm_long.masked_fill_(torch.triu(torch.ones((S_long, S_long), device=dev, dtype=torch.bool), 1), torch.finfo(torch.bfloat16).min)
        # Only need last 8 rows of attention output to verify sparse top-512 blocks + tail!
        # Run indexer & attention on full 2200 (takes <1s on GB10!)
        attn_long_out, _ = l3.self_attn(x_long, pe_long, cm_long, cache=None)
        save_bin(os.path.join(args.out, "l3_long_in.bf16"), x_long[0])
        save_bin(os.path.join(args.out, "l3_long_out_tail8.bf16"), attn_long_out[0, -8:])

        # Final HC mixer check
        fmix = Qwen4ExpTextGatedResidual(cfg, use_combine=False)
        fmix.load_state_dict({
            k.replace("model.language_model.hyper_connection_mixer.", ""): v
            for k, v in base_w.items()
            if k.startswith("model.language_model.hyper_connection_mixer.")
        })
        fmix = fmix.to(dev, dtype=torch.bfloat16).eval()
        final_mixed = fmix(h1_after_ple)
        save_bin(os.path.join(args.out, "final_mixed.bf16"), final_mixed[0])

    print(f"Reference fixtures dumped to {args.out} successfully!")


if __name__ == "__main__":
    main()
