#!/usr/bin/env python3
"""All-in-one GPU layer-streaming BF16 Reference Oracle for Tencent Hy4-flash (HF PR 50903).
Streams each BF16 MoE layer (L001..L041) ONCE from --stage-dir/L{001..041}.safetensors
(fed ahead over 10GbE LAN by stage_bf16_feeder.py and unlinked right after loading) and
simultaneously produces:
  1. --dump-4tok DIR : exact per-layer + logits BF16 reference for 4 tokens [120000,500,1234,5678]
  2. --dump-2200 DIR : exact per-layer + logits BF16 reference for slice [2048:2200] of a 2200-token
                       sequence (row 977 + row 978[:152]), exercising the sparse Compressed Group
                       Indexer (> 2048 tokens) across all 42 layers!
  3. --out-ref-dir DIR : exact full 120,832-way BF16 log-probabilities on all 64 held-out rows
                         calib/hy4-calib-2m-v2.npy[977:1041] (131,008 tokens) + ref_meta.json!
Peak GPU VRAM < 22 GiB on RTX PRO 6000.
"""
import argparse, json, math, os, struct, sys, time
import numpy as np
import torch
import torch.nn.functional as F

for p in os.environ.get("HY4_REF_PATHS", "").split(":"):
    if p and os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from transformers.models.hy_v4_flash.configuration_hy_v4_flash import HYV4FlashTextConfig
from transformers.models.hy_v4_flash.modeling_hy_v4_flash import (
    HYV4FlashDecoderLayer, HYV4FlashHyperHead, HYV4FlashRMSNorm, HYV4FlashRotaryEmbedding,
)
from hy4hess.keymap import map_key, draw_hash_multipliers
from safetensors import safe_open


class ReleaseReader:
    def __init__(self, release_dir: str, engram_dir: str, engram_cache_dir: str = ""):
        self.release_dir = release_dir
        self.engram_dir = engram_dir
        self.engram_cache_dir = engram_cache_dir
        self._eng_cache = {}
        self.dense_p = os.path.join(release_dir, "dense/exlr8-dense.safetensors")
        self.vocab_p = os.path.join(release_dir, "dense/exlr8-vocab.safetensors")
        self.by_mod = {}
        for fn in (self.dense_p, self.vocab_p):
            with safe_open(fn, framework="pt", device="cpu") as f:
                for bk in f.keys():
                    if bk.startswith("__pad."):
                        continue
                    mk = map_key(bk)
                    if mk is not None:
                        self.by_mod[mk] = (bk, fn)
                    elif bk == "lm_head.weight":
                        self.by_mod["lm_head.weight"] = (bk, fn)

    def load_tensor(self, mod_key: str, device: str, dtype=None) -> torch.Tensor:
        bk, fn = self.by_mod[mod_key]
        with safe_open(fn, framework="pt", device="cpu") as f:
            t = f.get_tensor(bk)
        if dtype is not None:
            t = t.to(dtype)
        return t.to(device)

    def load_engram_rows_fp8(self, layer_idx: int, unique_rows: np.ndarray, device: str) -> torch.Tensor:
        """Read exact 128B/row FP8+BF16-scale Engram rows (from pre-extracted local NVMe cache if present!)."""
        if self.engram_cache_dir:
            if layer_idx not in self._eng_cache:
                p_ids = os.path.join(self.engram_cache_dir, f"L{layer_idx:03d}_ids.npy")
                p_rows = os.path.join(self.engram_cache_dir, f"L{layer_idx:03d}_rows.bin")
                while not (os.path.exists(p_ids) and os.path.exists(p_rows)):
                    time.sleep(0.5)
                ids = np.load(p_ids)
                rows = np.fromfile(p_rows, dtype=np.uint8).reshape(-1, 128)
                self._eng_cache[layer_idx] = (ids, rows)
            ids, rows = self._eng_cache[layer_idx]
            pos = np.searchsorted(ids, unique_rows)
            assert np.all(pos < len(ids)) and np.all(ids[pos] == unique_rows), f"L{layer_idx} engram cache miss!"
            buf = rows[pos]
        else:
            fn = os.path.join(self.engram_dir, f"exlr8-engram-L{layer_idx:03d}.bin")
            buf = np.empty((len(unique_rows), 128), dtype=np.uint8)
            with open(fn, "rb") as f:
                for i, r in enumerate(unique_rows):
                    f.seek(int(r) * 128)
                    buf[i] = np.frombuffer(f.read(128), dtype=np.uint8)
        q_u8 = torch.from_numpy(buf[:, :96].copy()).to(device)
        sc_u16 = torch.from_numpy(buf[:, 96:98].copy()).view(torch.bfloat16).to(device)
        q_bf16 = q_u8.view(torch.float8_e4m3fn).to(torch.bfloat16)
        return q_bf16 * sc_u16


def make_text_config(release_dir: str) -> HYV4FlashTextConfig:
    with open(os.path.join(release_dir, "config.json")) as f:
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
    min_v = torch.finfo(dtype).min
    m_full = torch.where(k <= q, torch.zeros((), dtype=dtype, device=device), torch.full((), min_v, dtype=dtype, device=device))
    m_swa  = torch.where((k <= q) & (k > q - swa_window), torch.zeros((), dtype=dtype, device=device), torch.full((), min_v, dtype=dtype, device=device))
    return m_full[None, None, :, :].expand(B, 1, T, T), m_swa[None, None, :, :].expand(B, 1, T, T)


def save_bf16(path: str, t: torch.Tensor):
    raw = t.detach().to(device="cpu", dtype=torch.bfloat16).contiguous().view(torch.uint16).numpy()
    raw.tofile(path)


def load_layer(rel: ReleaseReader, layer: HYV4FlashDecoderLayer, l: int, stage_dir: str, device: str):
    prefix = f"layers.{l}."
    fp32_sub = ("attn_hc.", "ffn_hc.", "e_score_correction_bias",
                "key_value_projection", "key_gate_projection", "intra_group_ape", "compressed_indexer.key_norm")
    sd = {}
    for mk in rel.by_mod:
        if not mk.startswith(prefix):
            continue
        sub = mk[len(prefix):]
        dt = torch.float32 if any(k in sub for k in fp32_sub) else None
        sd[sub] = rel.load_tensor(mk, device, dtype=dt)

    if l >= 1:
        st_path = os.path.join(stage_dir, f"L{l:03d}.safetensors")
        t_wait = time.time()
        while not os.path.exists(st_path):
            time.sleep(0.1)
        if time.time() - t_wait > 1.0:
            print(f"# waited {time.time()-t_wait:.1f}s for {st_path}", flush=True)
        with safe_open(st_path, framework="pt", device="cpu") as f:
            for bk in f.keys():
                mk = map_key(bk)
                assert mk is not None and mk.startswith(prefix), f"unexpected {bk} -> {mk}"
                sd[mk[len(prefix):]] = f.get_tensor(bk).to(device)
        try:
            os.remove(st_path)
        except OSError:
            pass

    if layer.engram_layer is not None:
        layer.engram_layer.multi_head_embedding.embedding = None
    missing, unexpected = layer.load_state_dict(sd, strict=False, assign=True)
    bad_miss = [m for m in missing if not m.endswith(("embedding.weight", "offsets", "head_vocab_sizes", "hash_multipliers"))
                and "inv_freq" not in m]
    if bad_miss or unexpected:
        raise RuntimeError(f"L{l} load mismatch: missing={bad_miss} unexpected={unexpected}")
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


def apply_engram_release(rel: ReleaseReader, layer: HYV4FlashDecoderLayer, l: int,
                         hidden: torch.Tensor, ngram_tokens: torch.Tensor, device: str) -> torch.Tensor:
    eng = layer.engram_layer
    B, T, _ = ngram_tokens.shape
    hash_ids = eng.hash_ngrams(ngram_tokens)
    flat_rows = (hash_ids.clamp_min(0) + eng.multi_head_embedding.offsets).reshape(-1).cpu().numpy()
    uniq, inv = np.unique(flat_rows, return_inverse=True)
    emb_uniq = rel.load_engram_rows_fp8(l, uniq, device)
    inv_t = torch.from_numpy(inv).to(device=device, dtype=torch.long)
    embeddings = emb_uniq[inv_t].view(B, T, eng.num_hash_heads, 96)
    embeddings = embeddings.masked_fill(hash_ids.unsqueeze(-1) < 0, 0.0)
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


def run_track_layer(rel, layer, l, tr, device):
    B = tr["hidden"].shape[0]
    mb = tr["mb"]
    lt = layer.attention_type
    for b0 in range(0, B, mb):
        b1 = min(B, b0 + mb)
        h_sub = tr["hidden"][b0:b1].to(device)
        if layer.engram_layer is not None:
            h_sub = apply_engram_release(rel, layer, l, h_sub, tr["ngram"][b0:b1].to(device), device)
            if tr["dump"] and b0 == 0:
                sl = tr["dump_slice"]
                save_bf16(os.path.join(tr["dump"], f"L{l:03d}_post_engram.bf16"), h_sub[:, sl[0]:sl[1]])
        res = h_sub
        post, h_in = layer.attn_hc(h_sub)
        h_in = layer.input_layernorm(h_in)
        pe = (tr["pos_emb"][lt][0][b0:b1], tr["pos_emb"][lt][1][b0:b1])
        am = (tr["m_full"] if lt == "indexed_attention" else tr["m_swa"])[b0:b1]
        pids = tr["pos_ids"][:, b0:b1, :]
        a_out, _ = layer.self_attn(
            hidden_states=h_in, position_embeddings=pe, attention_mask=am,
            position_ids=pids, indexer_position_ids=pids,
        )
        h_sub = (post.float().unsqueeze(-1) * a_out.float().unsqueeze(-2) + res.float()).to(h_sub.dtype)
        if tr["dump"] and b0 == 0:
            sl = tr["dump_slice"]
            save_bf16(os.path.join(tr["dump"], f"L{l:03d}_post_attn.bf16"), h_sub[:, sl[0]:sl[1]])

        res = h_sub
        post, h_in = layer.ffn_hc(h_sub)
        h_in = layer.post_attention_layernorm(h_in)
        m_out = layer.mlp(h_in)
        h_sub = (post.float().unsqueeze(-1) * m_out.float().unsqueeze(-2) + res.float()).to(h_sub.dtype)
        tr["hidden"][b0:b1] = h_sub.to(tr["hidden"].device)

    if tr["dump"]:
        sl = tr["dump_slice"]
        save_bf16(os.path.join(tr["dump"], f"L{l:03d}_out.bf16"), tr["hidden"][:, sl[0]:sl[1]])


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="/var/tmp/hy4_pack")
    ap.add_argument("--engram-dir", default="")
    ap.add_argument("--engram-cache-dir", default="/var/tmp/hy4_engram_oracle")
    ap.add_argument("--stage-dir", default="/var/tmp/hy4_stage")
    ap.add_argument("--eval-npy", default="")
    ap.add_argument("--eval-start", type=int, default=977)
    ap.add_argument("--eval-count", type=int, default=64)
    ap.add_argument("--out-ref-dir", default="/var/tmp/hy4_ref")
    ap.add_argument("--meta-out", default="")
    ap.add_argument("--dump-4tok", default="")
    ap.add_argument("--dump-2200", default="")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device
    rel = ReleaseReader(args.release, args.engram_dir, args.engram_cache_dir)
    cfg = make_text_config(args.release)
    rotary = HYV4FlashRotaryEmbedding(cfg).to(device)
    emb_w = rel.load_tensor("embed_tokens.weight", device, dtype=torch.bfloat16)

    arr = np.load(args.eval_npy)
    tracks = {}

    def init_track(name, input_ids_cpu, mb, dump_dir, dump_slice, keep_on_cpu=False):
        B, T = input_ids_cpu.shape
        ids_dev = input_ids_cpu.to(device)
        hid = F.embedding(ids_dev, emb_w).unsqueeze(2).expand(-1, -1, cfg.hc_mult, -1).contiguous()
        if keep_on_cpu:
            hid = hid.cpu()
        pos_ids = torch.arange(T, device=device, dtype=torch.long).view(1, 1, T).expand(3, B, T)
        pos_emb = {lt: rotary(torch.empty((B, T, 1), dtype=torch.bfloat16, device=device), position_ids=pos_ids, layer_type=lt)
                   for lt in rotary.layer_types}
        m_full, m_swa = make_causal_masks(B, T, cfg.sliding_window, device, torch.bfloat16)
        ngram = build_ngram_tokens(ids_dev, cfg.engram_pad_id, cfg.engram_max_ngram_size)
        if keep_on_cpu:
            ngram = ngram.cpu()
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
            save_bf16(os.path.join(dump_dir, "embed.bf16"), hid[:, dump_slice[0]:dump_slice[1]])
        tracks[name] = dict(input_ids=ids_dev, hidden=hid, pos_ids=pos_ids, pos_emb=pos_emb,
                            m_full=m_full, m_swa=m_swa, ngram=ngram, mb=mb,
                            dump=dump_dir, dump_slice=dump_slice)
        print(f"# track {name}: shape=[{B}, {T}] dump={dump_dir} slice={dump_slice}", flush=True)

    if args.dump_4tok:
        init_track("4tok", torch.tensor([[120000, 500, 1234, 5678]], dtype=torch.long), 1, args.dump_4tok, (0, 4))
    if args.dump_2200:
        seq2200 = np.concatenate([arr[args.eval_start], arr[args.eval_start + 1][:152]]).astype(np.int64)
        os.makedirs(args.dump_2200, exist_ok=True)
        with open(os.path.join(args.dump_2200, "tokens.txt"), "w") as f:
            f.write(",".join(str(int(x)) for x in seq2200))
        init_track("2200", torch.from_numpy(seq2200).unsqueeze(0), 1, args.dump_2200, (2048, 2200))
    if args.out_ref_dir:
        toks64 = arr[args.eval_start : args.eval_start + args.eval_count].astype(np.int64)
        init_track("heldout64", torch.from_numpy(toks64), args.batch_size, "", (0, 0), keep_on_cpu=True)

    del emb_w
    torch.cuda.empty_cache()

    t0 = time.time()
    for l in range(cfg.num_hidden_layers):
        tl0 = time.time()
        with torch.device("meta"):
            layer = HYV4FlashDecoderLayer(cfg, l)
        load_layer(rel, layer, l, args.stage_dir, device)
        layer.eval()
        for name, tr in tracks.items():
            run_track_layer(rel, layer, l, tr, device)
        del layer
        torch.cuda.empty_cache()
        print(f"# oracle_gpu L{l:03d}/41 done ({time.time()-tl0:.2f}s, elapsed {time.time()-t0:.1f}s)", flush=True)

    hc_head = HYV4FlashHyperHead(cfg).to(device=device, dtype=torch.float32)
    hc_head.hc_fn.data.copy_(rel.load_tensor("hc_head.hc_fn", device, dtype=torch.float32))
    hc_head.hc_base.data.copy_(rel.load_tensor("hc_head.hc_base", device, dtype=torch.float32))
    hc_head.hc_scale.data.copy_(rel.load_tensor("hc_head.hc_scale", device, dtype=torch.float32))
    norm = HYV4FlashRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(device=device, dtype=torch.bfloat16)
    norm.weight.data.copy_(rel.load_tensor("norm.weight", device, dtype=torch.bfloat16))
    lm_w = rel.load_tensor("lm_head.weight", device, dtype=torch.bfloat16)

    for name in ("4tok", "2200"):
        if name not in tracks:
            continue
        tr = tracks[name]
        sl = tr["dump_slice"]
        hf = norm(hc_head(tr["hidden"].to(device)))[:, sl[0]:sl[1]]
        save_bf16(os.path.join(tr["dump"], "final_norm.bf16"), hf)
        lg = F.linear(hf.float(), lm_w.float())
        lg.cpu().numpy().astype(np.float32).tofile(os.path.join(tr["dump"], "logits.f32"))
        print(f"# dump {name} complete -> {tr['dump']}; last argmax={int(lg[0, -1].argmax().item())}", flush=True)

    if "heldout64" in tracks:
        tr = tracks["heldout64"]
        os.makedirs(args.out_ref_dir, exist_ok=True)
        # Checkpoint L041 hidden states on local NVMe (3.2 GB) before lm_head!
        tr["hidden"].cpu().contiguous().view(torch.uint16).numpy().tofile(os.path.join(args.out_ref_dir, "heldout64_L041_out.bf16"))
        B, T = tr["input_ids"].shape
        os.makedirs(args.out_ref_dir, exist_ok=True)
        logp_path = os.path.join(args.out_ref_dir, "ref_logp_bf16.bin")
        sum_nll = 0.0
        n_tok = 0
        with open(logp_path, "wb") as fp:
            fp.write(struct.pack("<4i", 0x48593452, B, T - 1, cfg.vocab_size))
            for b in range(B):
                hf = norm(hc_head(tr["hidden"][b:b+1].to(device)))
                lg = F.linear(hf[0, :-1].float(), lm_w.float())
                lp = F.log_softmax(lg, dim=-1)
                tgts = tr["input_ids"][b, 1:]
                nll = -lp[torch.arange(T - 1, device=device), tgts]
                sum_nll += float(nll.sum().item())
                n_tok += (T - 1)
                fp.write(lp.to(torch.bfloat16).cpu().view(torch.uint16).numpy().tobytes())
        mean_nll = sum_nll / n_tok
        meta = {"rows": B, "seq": T, "tokens": n_tok, "vocab": cfg.vocab_size,
                "bf16_mean_nll": mean_nll, "bf16_ppl": math.exp(mean_nll)}
        os.makedirs(os.path.dirname(args.meta_out), exist_ok=True)
        with open(args.meta_out, "w") as f:
            json.dump(meta, f, indent=2)
        with open(os.path.join(args.out_ref_dir, "ref_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"ORACLE_BF16_SUMMARY rows={B} tokens={n_tok} mean_nll={mean_nll:.6f} ppl={math.exp(mean_nll):.6f}", flush=True)


if __name__ == "__main__":
    main()
