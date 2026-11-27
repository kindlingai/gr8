#!/usr/bin/env python3
"""Layer-streaming BF16 Reference Oracle for Tencent Hy4-flash (HF PR 50903).
Runs directly from a local Hugging Face Hy4-flash checkpoint directory (--bank or HY4_MODEL_DIR)
using 1-layer-at-a-time streaming (< 14 GiB VRAM on GPU):
  Mode 1 (--dump DIR --tokens ...):
    Dumps embed.bf16, L{000..041}_out.bf16, L{001,011,021,031}_post_engram.bf16, L{000..041}_post_attn.bf16,
    final_norm.bf16, logits.f32 for exact per-layer parity verification against m4sim_hy4!
  Mode 2 (--eval-npy PATH --eval-start 977 --eval-count 64 --out-ref-dir DIR):
    Streams ALL 64 held-out rows [64, 2048] through all 42 native BF16 layers + lm_head, computing exact
    BF16 reference NLL/PPL AND saving per-position exact BF16 log-probabilities for exact full-vocab KLD!
"""
import argparse, json, math, os, struct, sys, time
import numpy as np
import torch
import torch.nn.functional as F


from transformers.models.hy_v4_flash.configuration_hy_v4_flash import HYV4FlashTextConfig
from transformers.models.hy_v4_flash.modeling_hy_v4_flash import (
    HYV4FlashDecoderLayer, HYV4FlashHyperHead, HYV4FlashRMSNorm, HYV4FlashRotaryEmbedding,
    _fp8_round_trip,
)
from hy4hess.keymap import map_key, draw_hash_multipliers
from safetensors import safe_open

BANK_DEFAULT = os.environ.get("HY4_MODEL_DIR", "")


class BankReader:
    def __init__(self, bank_dir: str):
        self.bank_dir = bank_dir
        with open(os.path.join(bank_dir, "model.safetensors.index.json")) as f:
            self.wmap = json.load(f)["weight_map"]
        self.by_mod = {}
        for bk, fn in self.wmap.items():
            mk = map_key(bk)
            if mk is not None:
                self.by_mod[mk] = (bk, os.path.join(bank_dir, fn))

    def load_tensor(self, mod_key: str, device: str, dtype=None) -> torch.Tensor:
        bk, fn = self.by_mod[mod_key]
        with safe_open(fn, framework="pt", device="cpu") as f:
            t = f.get_tensor(bk)
        if dtype is not None:
            t = t.to(dtype)
        return t.to(device)

    def load_engram_rows(self, layer_idx: int, unique_rows: np.ndarray) -> torch.Tensor:
        bk = f"model.layers.{layer_idx}.engram_layer.multi_head_embedding.embedding.weight"
        fn = os.path.join(self.bank_dir, self.wmap[bk])
        with open(fn, "rb") as f:
            hl = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hl))
            s0, _ = hdr[bk]["data_offsets"]
            base = 8 + hl + s0
            out = np.empty((len(unique_rows), 96), dtype=np.uint16)
            for i, r in enumerate(unique_rows):
                f.seek(base + int(r) * 192)
                out[i] = np.frombuffer(f.read(192), dtype=np.uint16)
        return torch.from_numpy(out).view(torch.bfloat16)


def make_text_config(bank_dir: str) -> HYV4FlashTextConfig:
    with open(os.path.join(bank_dir, "config.json")) as f:
        tc = json.load(f)["text_config"]
    tc = dict(tc)
    tc["layer_types"] = ["indexed_attention" if t == "deepseek_sparse_attention" else t for t in tc["layer_types"]]
    if "rope_parameters" in tc and "deepseek_sparse_attention" in tc["rope_parameters"]:
        rp = dict(tc["rope_parameters"])
        rp["indexed_attention"] = rp.pop("deepseek_sparse_attention")
        tc["rope_parameters"] = rp
    tc["engram_lookup_table_size"] = 120818
    cfg = HYV4FlashTextConfig(**tc)
    cfg._attn_implementation = "eager"
    return cfg


def build_ngram_tokens(input_ids: torch.Tensor, pad_id: int, max_ngram: int = 3) -> torch.Tensor:
    B, T = input_ids.shape
    pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
    shifted = [input_ids]
    for off in range(1, max_ngram):
        prev = F.pad(input_ids, (off, 0), value=pad_id)[:, :T]
        shifted.append(torch.where(pos < off, pad_id, prev))
    return torch.stack(shifted, dim=-1)


def make_causal_masks(B: int, T: int, swa_window: int, device: str, dtype: torch.dtype):
    q = torch.arange(T, device=device)[:, None]
    k = torch.arange(T, device=device)[None, :]
    allow_full = (k <= q)
    allow_swa = (k <= q) & (k > q - swa_window)
    min_v = torch.finfo(dtype).min
    m_full = torch.where(allow_full, torch.zeros((), dtype=dtype, device=device), torch.full((), min_v, dtype=dtype, device=device))
    m_swa  = torch.where(allow_swa,  torch.zeros((), dtype=dtype, device=device), torch.full((), min_v, dtype=dtype, device=device))
    return m_full[None, None, :, :].expand(B, 1, T, T), m_swa[None, None, :, :].expand(B, 1, T, T)


def load_decoder_layer_weights(bank: BankReader, layer: HYV4FlashDecoderLayer, layer_idx: int, device: str):
    prefix = f"layers.{layer_idx}."
    fp32_sub = ("attn_hc.", "ffn_hc.", "e_score_correction_bias",
                "key_value_projection", "key_gate_projection", "intra_group_ape", "compressed_indexer.key_norm")
    sd = {}
    for mk in bank.by_mod:
        if not mk.startswith(prefix):
            continue
        sub = mk[len(prefix):]
        if sub == "engram_layer.multi_head_embedding.embedding.weight":
            continue
        dt = torch.float32 if any(k in sub for k in fp32_sub) else None
        sd[sub] = bank.load_tensor(mk, device, dtype=dt)
    if layer.engram_layer is not None:
        layer.engram_layer.multi_head_embedding.embedding = None
    missing, unexpected = layer.load_state_dict(sd, strict=False, assign=True)
    bad_miss = [m for m in missing if not m.endswith(("embedding.weight", "offsets", "head_vocab_sizes", "hash_multipliers"))
                and "inv_freq" not in m]
    if bad_miss or unexpected:
        raise RuntimeError(f"L{layer_idx} load mismatch: missing={bad_miss} unexpected={unexpected}")
    if layer.engram_layer is not None:
        eng = layer.engram_layer
        eng.multi_head_embedding.offsets = eng.multi_head_embedding.head_offsets().to(device)
        eng.head_vocab_sizes = torch.tensor(eng.config_head_vocab_sizes, device=device)
        cv = int(eng.lookup_table.max().item()) + 1
        eng.hash_multipliers = draw_hash_multipliers(eng.seed, eng.layer_id, cv, eng.max_ngram_size).to(device)
    if layer.self_attn.compressed_indexer is not None:
        irot = layer.self_attn.compressed_indexer.rotary_emb
        for lt in irot.layer_types:
            inv_f, _ = irot.compute_default_rope_parameters(irot.config, device=device, layer_type=lt)
            setattr(irot, f"{lt}_inv_freq", inv_f)


def apply_engram_sparse(bank: BankReader, layer: HYV4FlashDecoderLayer, layer_idx: int,
                        hidden: torch.Tensor, ngram_tokens: torch.Tensor, device: str) -> torch.Tensor:
    eng = layer.engram_layer
    B, T, _ = ngram_tokens.shape
    hash_ids = eng.hash_ngrams(ngram_tokens)
    flat_rows = (hash_ids.clamp_min(0) + eng.multi_head_embedding.offsets).reshape(-1).cpu().numpy()
    uniq, inv = np.unique(flat_rows, return_inverse=True)
    emb_uniq = bank.load_engram_rows(layer_idx, uniq).to(device=device, dtype=hidden.dtype)
    inv_t = torch.from_numpy(inv).to(device=device, dtype=torch.long)
    embeddings = emb_uniq[inv_t].view(B, T, eng.num_hash_heads, 96)
    embeddings = embeddings.masked_fill(hash_ids.unsqueeze(-1) < 0, 0.0)
    if eng.fp8_embedding:
        embeddings = _fp8_round_trip(embeddings)
    embeddings = torch.stack([norm(embeddings[:, :, h]) for h, norm in enumerate(eng.lookup_per_head_norms)], dim=2)
    embeddings = embeddings.flatten(start_dim=-2)

    gates = []
    for hc in range(eng.hc_mult):
        key = eng.norm_key[hc](eng.key_projs[hc](embeddings))
        query = eng.norm_query[hc](hidden[:, :, hc])
        g = (key * query).sum(dim=-1) * (eng.hidden_size ** -0.5)
        g = g.abs().clamp_min(1e-6).sqrt() * g.sign()
        gates.append(g.sigmoid())
    gates = torch.stack(gates, dim=2).unsqueeze(-1)
    main_signal = eng.unified_pre_gamma * eng.unified_rmsnorm(eng.value_proj(embeddings))
    branch = torch.stack([sc * main_signal for sc in eng.unified_per_hc_scales], dim=2)
    return hidden + (gates * main_signal.unsqueeze(2) + branch)


def save_bf16(path: str, t: torch.Tensor):
    raw = t.detach().to(device="cpu", dtype=torch.bfloat16).contiguous().view(torch.uint16).numpy()
    raw.tofile(path)


@torch.no_grad()
def run_oracle(args):
    device = args.device
    bank = BankReader(args.bank)
    cfg = make_text_config(args.bank)
    rotary = HYV4FlashRotaryEmbedding(cfg).to(device)

    if args.eval_npy:
        arr = np.load(args.eval_npy)
        toks_np = arr[args.eval_start : args.eval_start + args.eval_count].astype(np.int64)
        input_ids = torch.from_numpy(toks_np).to(device)
    else:
        toks = [int(x) for x in args.tokens.split(",") if x.strip()]
        input_ids = torch.tensor([toks], dtype=torch.long, device=device)

    B, T = input_ids.shape
    print(f"# hy4_oracle: device={device} shape=[{B}, {T}]", flush=True)
    t0 = time.time()

    emb_w = bank.load_tensor("embed_tokens.weight", device, dtype=torch.bfloat16)
    inputs_embeds = F.embedding(input_ids, emb_w)
    del emb_w
    hidden = inputs_embeds.unsqueeze(2).expand(-1, -1, cfg.hc_mult, -1).contiguous()
    del inputs_embeds

    if args.dump:
        os.makedirs(args.dump, exist_ok=True)
        save_bf16(os.path.join(args.dump, "embed.bf16"), hidden)

    pos_ids = torch.arange(T, device=device, dtype=torch.long).view(1, 1, T).expand(3, B, T)
    pos_emb = {lt: rotary(hidden, position_ids=pos_ids, layer_type=lt) for lt in rotary.layer_types}
    m_full, m_swa = make_causal_masks(B, T, cfg.sliding_window, device, torch.bfloat16)
    ngram_tokens = build_ngram_tokens(input_ids, cfg.engram_pad_id, cfg.engram_max_ngram_size)

    mb = args.batch_size
    for l in range(cfg.num_hidden_layers):
        tl0 = time.time()
        with torch.device("meta"):
            layer = HYV4FlashDecoderLayer(cfg, l)
        load_decoder_layer_weights(bank, layer, l, device)
        layer.eval()

        for b0 in range(0, B, mb):
            b1 = min(B, b0 + mb)
            h_sub = hidden[b0:b1]
            if layer.engram_layer is not None:
                h_sub = apply_engram_sparse(bank, layer, l, h_sub, ngram_tokens[b0:b1], device)
                if args.dump and b0 == 0:
                    save_bf16(os.path.join(args.dump, f"L{l:03d}_post_engram.bf16"), h_sub)
            res = h_sub
            post, h_in = layer.attn_hc(h_sub)
            h_in = layer.input_layernorm(h_in)
            lt = layer.attention_type
            pe = (pos_emb[lt][0][b0:b1], pos_emb[lt][1][b0:b1])
            am = m_full[b0:b1] if lt == "indexed_attention" else m_swa[b0:b1]
            a_out, _ = layer.self_attn(
                hidden_states=h_in, position_embeddings=pe, attention_mask=am,
                position_ids=pos_ids[:, b0:b1, :], indexer_position_ids=pos_ids[:, b0:b1, :],
            )
            h_sub = (post.float().unsqueeze(-1) * a_out.float().unsqueeze(-2) + res.float()).to(h_sub.dtype)
            if args.dump and b0 == 0:
                save_bf16(os.path.join(args.dump, f"L{l:03d}_post_attn.bf16"), h_sub)

            res = h_sub
            post, h_in = layer.ffn_hc(h_sub)
            h_in = layer.post_attention_layernorm(h_in)
            m_out = layer.mlp(h_in)
            h_sub = (post.float().unsqueeze(-1) * m_out.float().unsqueeze(-2) + res.float()).to(h_sub.dtype)
            hidden[b0:b1] = h_sub

        if args.dump:
            save_bf16(os.path.join(args.dump, f"L{l:03d}_out.bf16"), hidden)
        del layer
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        if (l + 1) % 6 == 0 or l == 0 or l == cfg.num_hidden_layers - 1:
            print(f"# oracle L{l:03d}/{cfg.num_hidden_layers-1} done ({time.time()-tl0:.2f}s, elapsed {time.time()-t0:.1f}s)", flush=True)

    hc_head = HYV4FlashHyperHead(cfg).to(device=device, dtype=torch.float32)
    hc_head.hc_fn.data.copy_(bank.load_tensor("hc_head.hc_fn", device, dtype=torch.float32))
    hc_head.hc_base.data.copy_(bank.load_tensor("hc_head.hc_base", device, dtype=torch.float32))
    hc_head.hc_scale.data.copy_(bank.load_tensor("hc_head.hc_scale", device, dtype=torch.float32))
    norm = HYV4FlashRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(device=device, dtype=torch.bfloat16)
    norm.weight.data.copy_(bank.load_tensor("norm.weight", device, dtype=torch.bfloat16))
    lm_w = bank.load_tensor("lm_head.weight", device, dtype=torch.bfloat16)

    h_final = norm(hc_head(hidden))
    if args.dump:
        save_bf16(os.path.join(args.dump, "final_norm.bf16"), h_final)
        logits = F.linear(h_final.float(), lm_w.float())
        logits.cpu().numpy().astype(np.float32).tofile(os.path.join(args.dump, "logits.f32"))
        top1 = logits[0].argmax(dim=-1).tolist()
        print(f"# oracle dump complete -> {args.dump}; last token argmax = {top1[-1]}", flush=True)

    if args.out_ref_dir:
        os.makedirs(args.out_ref_dir, exist_ok=True)
        logp_path = os.path.join(args.out_ref_dir, "ref_logp_bf16.bin")
        sum_nll = 0.0
        n_tok = 0
        with open(logp_path, "wb") as fp:
            fp.write(struct.pack("<4i", 0x48593452, B, T - 1, cfg.vocab_size))
            for b in range(B):
                lg = F.linear(h_final[b, :-1].float(), lm_w.float())
                lp = F.log_softmax(lg, dim=-1)
                tgts = input_ids[b, 1:]
                nll = -lp[torch.arange(T - 1, device=device), tgts]
                sum_nll += float(nll.sum().item())
                n_tok += (T - 1)
                lp_bf16 = lp.to(torch.bfloat16).cpu().view(torch.uint16).numpy()
                fp.write(lp_bf16.tobytes())
        mean_nll = sum_nll / n_tok
        meta = {"rows": B, "seq": T, "tokens": n_tok, "vocab": cfg.vocab_size,
                "bf16_mean_nll": mean_nll, "bf16_ppl": math.exp(mean_nll)}
        with open(os.path.join(args.out_ref_dir, "ref_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"ORACLE_BF16_SUMMARY rows={B} tokens={n_tok} mean_nll={mean_nll:.6f} ppl={math.exp(mean_nll):.6f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=BANK_DEFAULT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tokens", default="120000,500,1234,5678")
    ap.add_argument("--dump", default="")
    ap.add_argument("--eval-npy", default="")
    ap.add_argument("--eval-start", type=int, default=977)
    ap.add_argument("--eval-count", type=int, default=64)
    ap.add_argument("--out-ref-dir", default="")
    ap.add_argument("--batch-size", type=int, default=4)
    run_oracle(ap.parse_args())
