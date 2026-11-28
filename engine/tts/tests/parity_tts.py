#!/usr/bin/env python3
"""Gate E1 parity verifier: compares `g6tts_bench --teacher` against reference PyTorch Drama 3
(`exlr8.load_drama3` + fish-speech `FishOmniBackbone` & `FishOmniAudioDecoder`) on the exact same
release weights and teacher-forced sequences.

Gate E1 thresholds (from DRAMA_3_EXLR8_PLAN.md):
  - Final-hidden cosine similarity: mean >= 0.999 and min >= 0.995
  - Main-token argmax agreement (over the 4097 legal candidates): >= 0.99
  - Each of the 9 residual codebook heads argmax agreement (over 0..1023): >= 0.98

Binary formats:
  ROWS.bin ("G6TR"): [4B magic][u32 n_seq] then per seq: [u32 T][int32 rows[T * 11]]
  DUMP.bin ("G6TD"): [4B magic][u32 n_seq][u32 H][u32 n_cand][u32 n_heads][u32 head_vocab]
                     then per seq: [u32 T][f32 hidden[T,H]][f32 main_logits[T,n_cand]][f32 head_logits[T-1,n_heads,head_vocab]]
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def write_rows_bin(path: Path, seqs: list[np.ndarray]) -> None:
    with open(path, "wb") as f:
        f.write(b"G6TR")
        f.write(struct.pack("<I", len(seqs)))
        for s in seqs:
            arr = np.ascontiguousarray(s, dtype=np.int32)
            assert arr.ndim == 2 and arr.shape[1] == 11, f"expected [T, 11], got {arr.shape}"
            f.write(struct.pack("<I", arr.shape[0]))
            f.write(arr.tobytes())


def read_rows_bin(path: Path) -> list[np.ndarray]:
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"G6TR":
            raise ValueError(f"{path}: bad magic {magic!r} (expected b'G6TR')")
        (n_seq,) = struct.unpack("<I", f.read(4))
        seqs: list[np.ndarray] = []
        for _ in range(n_seq):
            (t_len,) = struct.unpack("<I", f.read(4))
            arr = np.frombuffer(f.read(t_len * 11 * 4), dtype=np.int32).reshape(t_len, 11).copy()
            seqs.append(arr)
    return seqs


def read_dump_bin(path: Path) -> list[dict[str, np.ndarray]]:
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"G6TD":
            raise ValueError(f"{path}: bad magic {magic!r} (expected b'G6TD')")
        n_seq, hid, n_cand, n_heads, h_vocab = struct.unpack("<5I", f.read(20))
        outs: list[dict[str, np.ndarray]] = []
        for _ in range(n_seq):
            (t_len,) = struct.unpack("<I", f.read(4))
            h_arr = np.frombuffer(f.read(t_len * hid * 4), dtype=np.float32).reshape(t_len, hid).copy()
            m_arr = np.frombuffer(f.read(t_len * n_cand * 4), dtype=np.float32).reshape(t_len, n_cand).copy()
            r_arr = np.frombuffer(f.read((t_len - 1) * n_heads * h_vocab * 4), dtype=np.float32).reshape(
                t_len - 1, n_heads, h_vocab
            ).copy()
            outs.append({"hidden": h_arr, "main_logits": m_arr, "head_logits": r_arr})
    return outs


def row_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    dot = np.sum(a64 * b64, axis=-1)
    na = np.linalg.norm(a64, axis=-1)
    nb = np.linalg.norm(b64, axis=-1)
    denom = np.maximum(na * nb, 1e-12)
    return dot / denom


def evaluate_gate_e1(
    ref_outs: list[dict[str, np.ndarray]],
    got_outs: list[dict[str, np.ndarray]],
    seqs: list[np.ndarray] | None = None,
) -> tuple[bool, dict[str, Any]]:
    if len(ref_outs) != len(got_outs):
        raise ValueError(f"sequence count mismatch: ref={len(ref_outs)} vs got={len(got_outs)}")

    all_cos: list[np.ndarray] = []
    pos_lt128_cos: list[np.ndarray] = []
    pos_ge128_cos: list[np.ndarray] = []
    text_cos: list[np.ndarray] = []
    sem_cos: list[np.ndarray] = []
    per_seq_min: list[float] = []
    per_seq_mean: list[float] = []
    main_match = 0
    main_total = 0
    head_match = [0] * 9
    head_total = [0] * 9
    worst_cos = 2.0
    worst_seq = -1
    worst_row = -1
    worst_tok = -1

    for si, (ref, got) in enumerate(zip(ref_outs, got_outs)):
        rc = row_cosines(ref["hidden"], got["hidden"])
        all_cos.append(rc)
        if rc.size > 0:
            per_seq_min.append(round(float(np.min(rc)), 8))
            per_seq_mean.append(round(float(np.mean(rc)), 8))
            pos_lt128_cos.append(rc[:128])
            if rc.size > 128:
                pos_ge128_cos.append(rc[128:])
            if seqs is not None and si < len(seqs) and seqs[si].shape[0] == rc.size:
                toks = seqs[si][:, 0]
                is_sem = (toks >= 248099) & (toks <= 252194)
                if np.any(is_sem):
                    sem_cos.append(rc[is_sem])
                if np.any(~is_sem):
                    text_cos.append(rc[~is_sem])
            ri = int(np.argmin(rc))
            if float(rc[ri]) < worst_cos:
                worst_cos = float(rc[ri])
                worst_seq = si
                worst_row = ri
                if seqs is not None and si < len(seqs) and ri < seqs[si].shape[0]:
                    worst_tok = int(seqs[si][ri, 0])
        ref_m = np.argmax(ref["main_logits"], axis=-1)
        got_m = np.argmax(got["main_logits"], axis=-1)
        main_match += int(np.sum(ref_m == got_m))
        main_total += int(ref_m.size)
        for h in range(9):
            ref_h = np.argmax(ref["head_logits"][:, h, :1024], axis=-1)
            got_h = np.argmax(got["head_logits"][:, h, :1024], axis=-1)
            head_match[h] += int(np.sum(ref_h == got_h))
            head_total[h] += int(ref_h.size)

    def _slice_stats(chunks: list[np.ndarray]) -> dict[str, Any] | None:
        if not chunks:
            return None
        cat = np.concatenate(chunks)
        if cat.size == 0:
            return None
        return {
            "count": int(cat.size),
            "mean": round(float(np.mean(cat)), 8),
            "min": round(float(np.min(cat)), 8),
        }

    cos_cat = np.concatenate(all_cos) if all_cos else np.array([0.0])
    cos_mean = float(np.mean(cos_cat))
    cos_min = float(np.min(cos_cat))
    cos_p01 = float(np.percentile(cos_cat, 1.0)) if cos_cat.size else 0.0
    main_acc = float(main_match) / max(1, main_total)
    head_accs = [float(head_match[h]) / max(1, head_total[h]) for h in range(9)]

    passed = (
        cos_mean >= 0.999
        and cos_min >= 0.995
        and main_acc >= 0.99
        and all(a >= 0.98 for a in head_accs)
    )
    summary = {
        "gate": "E1",
        "pass": passed,
        "sequences": len(ref_outs),
        "rows_total": int(cos_cat.size),
        "hidden_cos_mean": round(cos_mean, 8),
        "hidden_cos_p01": round(cos_p01, 8),
        "hidden_cos_min": round(cos_min, 8),
        "hidden_cos_pos_lt128": _slice_stats(pos_lt128_cos),
        "hidden_cos_pos_ge128": _slice_stats(pos_ge128_cos),
        "hidden_cos_text": _slice_stats(text_cos),
        "hidden_cos_semantic": _slice_stats(sem_cos),
        "per_seq_min_cos": per_seq_min,
        "per_seq_mean_cos": per_seq_mean,
        "worst_row": {"seq": worst_seq, "row": worst_row, "main_tok": worst_tok},
        "main_argmax_agree": round(main_acc, 6),
        "head_argmax_agree": [round(a, 6) for a in head_accs],
        "head_argmax_min": round(min(head_accs), 6),
        "thresholds": {
            "hidden_cos_mean": 0.999,
            "hidden_cos_min": 0.995,
            "main_argmax_agree": 0.99,
            "head_argmax_agree": 0.98,
        },
    }
    return passed, summary


def _import_exlr8(exlr8_repo: Path | None) -> Any:
    """Import the standalone `exlr8` loader module from `<exlr8_repo>/loader/exlr8.py` (or `EXLR8_LOADER_DIR`)."""
    candidates: list[Path] = []
    if exlr8_repo is not None:
        candidates.append(exlr8_repo / "loader")
        candidates.append(exlr8_repo)
    env_dir = os.environ.get("EXLR8_LOADER_DIR", "").strip()
    if env_dir:
        p = Path(env_dir)
        candidates.append(p / "loader" if (p / "loader" / "exlr8.py").is_file() else p)
    for c in candidates:
        if (c / "exlr8.py").is_file():
            cs = str(c.resolve())
            if cs not in sys.path:
                sys.path.insert(0, cs)
            break
    import exlr8  # type: ignore[import-not-found]

    return exlr8


def _safe_to_device(t: Any, device: Any, dtype: Any | None = None) -> Any:
    """Clone CPU tensors before transferring to CUDA (64K-page kernel mmap safety)."""
    if t.device.type == "cpu":
        t = t.clone()
    return t.to(device=device, dtype=dtype) if dtype is not None else t.to(device=device)


def _load_release_vocab_and_tower(
    exlr8: Any, model_dir: Path, dtype: Any, verify: bool = False
) -> dict[str, Any]:
    """Load only vocab + global + mtp_shared + mtp dense files (~2.5 GB total)."""
    state: dict[str, Any] = {}
    for path in exlr8._discover_dense_files(str(model_dir)):
        f = exlr8.SafeFile(path)
        exlr8.check_format(f)
        contents = f.meta.get("contents")
        if contents == "vocab" or (contents is None and os.path.basename(path) == "exlr8-vocab.safetensors"):
            names = exlr8._vocab_names(f)
            state.update(exlr8.load_matrices(f, lambda n: names.get(n, n), dtype, "cpu", verify))
        elif f.meta.get("group") in ("global", "mtp_shared", "mtp"):
            pre = exlr8._dense_prefix(f)
            state.update(exlr8.load_matrices(f, lambda n: pre + n, dtype, "cpu", verify))
    return state


def _load_release_backbone_layer(
    exlr8: Any,
    model_dir: Path,
    manifest: dict[str, Any] | None,
    layer_idx: int,
    layer_type: str,
    dtype: Any,
    verify: bool = False,
) -> dict[str, Any]:
    """Decode only backbone layer L (dense + experts) to bound peak RAM/VRAM to < 8 GB."""
    grp = "attn" if layer_type == "full_attention" else "gdn"
    dense_path = model_dir / "dense" / f"exlr8-dense-{grp}-{layer_idx:03d}.safetensors"
    if not dense_path.is_file():
        dense_path = model_dir / f"exlr8-dense-{grp}-{layer_idx:03d}.safetensors"
    f_dense = exlr8.SafeFile(str(dense_path))
    exlr8.check_format(f_dense)
    pre = exlr8._dense_prefix(f_dense)
    state = exlr8.load_matrices(f_dense, lambda n: pre + n, dtype, "cpu", verify)

    idx_file = model_dir / "experts" / f"exlr8-layer-{layer_idx:03d}.index.safetensors"
    if idx_file.is_file():
        if manifest is None:
            raise ValueError(f"{model_dir}: trellis experts present for layer {layer_idx} but --manifest is empty")
        exp_state = exlr8._load_drama3_trellis_experts(str(model_dir), manifest, layer_idx, dtype, "cpu", verify)
    else:
        exp_path = model_dir / "dense" / f"exlr8-dense-experts-{layer_idx:03d}.safetensors"
        if not exp_path.is_file():
            exp_path = model_dir / f"exlr8-dense-experts-{layer_idx:03d}.safetensors"
        f_exp = exlr8.SafeFile(str(exp_path))
        exlr8.check_format(f_exp)
        epre = exlr8._dense_prefix(f_exp)
        raw_exp = exlr8.load_matrices(f_exp, lambda n: epre + n, dtype, "cpu", verify)
        exp_state = exlr8._stack_drama3_experts(raw_exp, epre, dtype)

    state.update(exp_state)
    return state


def read_layer_trace_bin(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic not in (b"G6LT", b"G6L2"):
            raise ValueError(f"{path}: bad magic {magic!r} (expected b'G6LT' or b'G6L2')")
        v2 = magic == b"G6L2"
        t_len, hid, n_layers, top_k, n_tw = struct.unpack("<5I", f.read(20))

        def rd_f32(r: int, c: int) -> np.ndarray:
            return np.frombuffer(f.read(r * c * 4), dtype=np.float32).reshape(r, c).copy()

        embed = rd_f32(t_len, hid)
        final_hidden = rd_f32(t_len, hid) if v2 else None
        l0_qkv_raw = rd_f32(t_len, 8192) if v2 else None
        l0_qkv_conv = rd_f32(t_len, 8192) if v2 else None
        l0_core_out = rd_f32(t_len, 4096) if v2 else None
        l3_q_ready = rd_f32(t_len, 4096) if v2 else None
        l3_attn_out = rd_f32(t_len, 4096) if v2 else None
        mixer_res = []
        layer_res = []
        topk_ids = []
        for _ in range(n_layers):
            mixer_res.append(rd_f32(t_len, hid))
            layer_res.append(rd_f32(t_len, hid))
            topk_ids.append(np.frombuffer(f.read(t_len * top_k * 4), dtype=np.int32).reshape(t_len, top_k).copy())
        tw_hids = [rd_f32(t_len - 1, hid) for _ in range(n_tw)]
        tw_sub = []
        if v2:
            for _ in range(2):
                tw_sub.append(
                    {
                        "eh_cat": rd_f32(t_len - 1, 2 * hid),
                        "eh_proj": rd_f32(t_len - 1, hid),
                        "wqkv": rd_f32(t_len - 1, 9216),
                        "q_ready": rd_f32(t_len - 1, 4096),
                        "attn_out": rd_f32(t_len - 1, 4096),
                        "wo_res": rd_f32(t_len - 1, hid),
                        "ffn_act": rd_f32(t_len - 1, 6144),
                        "w2_res": rd_f32(t_len - 1, hid),
                    }
                )
    return {
        "T": t_len,
        "H": hid,
        "L": n_layers,
        "K": top_k,
        "embed": embed,
        "final_hidden": final_hidden,
        "l0_qkv_raw": l0_qkv_raw,
        "l0_qkv_conv": l0_qkv_conv,
        "l0_core_out": l0_core_out,
        "l3_q_ready": l3_q_ready,
        "l3_attn_out": l3_attn_out,
        "mixer_res": mixer_res,
        "layer_res": layer_res,
        "topk_ids": topk_ids,
        "tower_hids": tw_hids,
        "tower_sub": tw_sub,
    }


def flat_cos(a: np.ndarray, b: np.ndarray) -> float:
    a64 = a.astype(np.float64).ravel()
    b64 = b.astype(np.float64).ravel()
    denom = float(np.linalg.norm(a64) * np.linalg.norm(b64))
    return float(np.dot(a64, b64) / denom) if denom > 1e-12 else 0.0


def run_torch_reference(
    model_dir: Path,
    manifest: str | None,
    fish_speech_dir: Path,
    exlr8_repo: Path | None,
    seqs: list[np.ndarray],
    device: str,
    got_trace: dict[str, Any] | None = None,
    got_seq0_hid: np.ndarray | None = None,
) -> tuple[list[dict[str, np.ndarray]], dict[str, Any] | None]:
    """Layer-streamed teacher-forced PyTorch Drama 3 reference (< 8 GB peak RAM/VRAM)."""
    import torch

    fs_str = str(fish_speech_dir.resolve())
    if fs_str not in sys.path:
        sys.path.insert(0, fs_str)
    exlr8 = _import_exlr8(exlr8_repo)

    from fish_speech.models.text2semantic.fish_omni.backbone import DecoderLayer
    from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig
    from fish_speech.models.text2semantic.fish_omni.layers import (
        LayerContext,
        ZeroCenteredRMSNorm,
        apply_neox_rope,
        chunk_gated_delta_rule,
        grouped_attention,
        recurrent_gated_delta_rule,
    )
    from fish_speech.models.text2semantic.fish_omni.tower import (
        FishOmniAudioDecoder,
        load_audio_decoder_weights,
    )

    cfg_dict = json.loads((model_dir / "config.json").read_text())
    cfg = FishOmniConfig.from_dict(cfg_dict)
    dtype = torch.bfloat16
    dev = torch.device(device)

    manifest_obj = exlr8.load_manifest(str(model_dir), manifest) if manifest else None
    vt_state = _load_release_vocab_and_tower(exlr8, model_dir, dtype=dtype, verify=False)

    # 1. Input embeddings (embed_tokens + semantic codebook c1 mix / sqrt(1 + vq_backbone_codebooks))
    embed_w = _safe_to_device(vt_state["model.language_model.embed_tokens.weight"], dev, dtype)
    cb_w = _safe_to_device(vt_state["audio_decoder.codebook_embeddings.weight"], dev, dtype)
    frame_tensors = [torch.from_numpy(arr.copy()).to(dtype=torch.long) for arr in seqs]

    hidden_seqs: list[torch.Tensor] = []
    with torch.inference_mode():
        vq_count = cfg.audio_decoder.vq_backbone_codebooks
        cb_vocab = cfg.audio_decoder.vocab_size
        offsets = torch.arange(1, 1 + vq_count, device=dev) * cb_vocab
        inv_sqrt = 1.0 / math.sqrt(1 + vq_count)
        for ft in frame_tensors:
            f_dev = ft.to(dev)
            text_emb = torch.nn.functional.embedding(f_dev[:, 0], embed_w)
            rows = f_dev[:, 2 : 2 + vq_count] + offsets
            extra = torch.nn.functional.embedding_bag(rows, cb_w, mode="sum")
            mixed = (text_emb + extra) * inv_sqrt
            sem = (f_dev[:, 0] >= cfg.semantic_start_token_id) & (f_dev[:, 0] <= cfg.semantic_end_token_id)
            emb = torch.where(sem[:, None], mixed, text_emb).to(dtype)
            hidden_seqs.append(emb.cpu())
    del embed_w, cb_w

    diag: dict[str, Any] | None = None
    trace_t = 0
    if got_trace is not None and len(seqs) > 0:
        trace_t = min(int(got_trace["T"]), int(seqs[0].shape[0]))
        diag = {
            "T": trace_t,
            "embed_cos": round(flat_cos(hidden_seqs[0][:trace_t].float().numpy(), got_trace["embed"][:trace_t]), 8),
            "mixer_cos": [],
            "layer_cos": [],
            "layer_row_min_cos": [],
            "layer_cos_f32res": [],
            "router_agree": [],
            "l0_subops": {},
            "l3_subops": {},
            "tower_cos": [],
            "tower_cos_isolated": [],
            "tower_row_min_cos_isolated": [],
            "tower_torch_bf16_vs_fp32": [],
            "tower_subops": {},
        }
    h0_f32 = hidden_seqs[0][:trace_t].float().clone() if diag is not None else None
    h0_t64 = hidden_seqs[0][:trace_t].clone() if diag is not None else None
    h0_sr = hidden_seqs[0][:trace_t].float().clone() if diag is not None else None

    # Precompute NeoX RoPE table matching `FishOmniBackbone.run_layers`
    max_len = max(int(arr.shape[0]) for arr in seqs)
    rotary_dim = cfg.text.rotary_dim
    exponents = torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=dev)
    frequencies = 1.0 / (cfg.text.rope_theta ** (exponents / rotary_dim))
    all_positions = torch.arange(max_len, dtype=torch.float32, device=dev)
    angles = all_positions[:, None] * frequencies[None, :]
    angles = torch.cat((angles, angles), dim=-1)
    cos_all, sin_all = angles.cos(), angles.sin()

    # 2. Layer-streamed backbone forward (only 1 layer resident at a time)
    for layer_idx in range(cfg.text.num_hidden_layers):
        layer_type = cfg.text.layer_types[layer_idx]
        lstate = _load_release_backbone_layer(
            exlr8, model_dir, manifest_obj, layer_idx, layer_type, dtype=dtype, verify=False
        )
        layer = DecoderLayer(cfg.text, layer_idx, fp8=False, dtype=dtype, device=dev).eval()
        prefix = f"model.language_model.layers.{layer_idx}."
        targets = dict(layer.named_parameters())
        targets.update(dict(layer.named_buffers()))
        for rel_name, tgt in targets.items():
            full_k = prefix + rel_name
            if full_k not in lstate:
                raise KeyError(f"Layer {layer_idx}: missing tensor {full_k}")
            tgt.data.copy_(_safe_to_device(lstate[full_k], dev, tgt.dtype))
        del lstate

        with torch.inference_mode():
            if diag is not None and got_trace is not None:
                pos_t = torch.arange(trace_t, dtype=torch.long, device=dev)
                ctx_t = LayerContext(
                    input_pos=pos_t,
                    cos=cos_all[:trace_t],
                    sin=sin_all[:trace_t],
                    kv_len=trace_t,
                    continue_state=False,
                    use_cache=False,
                )
                # Standard BF16 residual sub-block trace
                x_in = hidden_seqs[0][:trace_t].to(dev).unsqueeze(0)
                x_norm = layer.input_layernorm(x_in).to(dtype)
                mixer = layer.self_attn if layer.layer_type == "full_attention" else layer.linear_attn
                x_mix = x_in + mixer(x_norm, ctx_t)
                if layer_idx == 0 and got_trace.get("l0_qkv_raw") is not None:
                    gdn = layer.linear_attn
                    qkv_raw = gdn.in_proj_qkv(x_norm)
                    mixed = qkv_raw.transpose(1, 2)
                    padded = torch.cat((mixed.new_zeros(1, gdn.conv_dim, gdn.conv_kernel_size - 1), mixed), dim=-1)
                    qkv_conv = torch.nn.functional.silu(
                        torch.nn.functional.conv1d(padded, gdn.conv1d.weight, groups=gdn.conv_dim)
                    ).transpose(1, 2)
                    q_g, k_g, v_g = qkv_conv.split([gdn.key_dim, gdn.key_dim, gdn.value_dim], dim=-1)
                    q_g = q_g.reshape(1, trace_t, gdn.num_k_heads, gdn.head_k_dim).repeat_interleave(
                        gdn.num_v_heads // gdn.num_k_heads, dim=2
                    )
                    k_g = k_g.reshape(1, trace_t, gdn.num_k_heads, gdn.head_k_dim).repeat_interleave(
                        gdn.num_v_heads // gdn.num_k_heads, dim=2
                    )
                    v_g = v_g.reshape(1, trace_t, gdn.num_v_heads, gdn.head_v_dim)
                    z_g = gdn.in_proj_z(x_norm).reshape(1, trace_t, gdn.num_v_heads, gdn.head_v_dim)
                    beta_g = torch.sigmoid(gdn.in_proj_b(x_norm).float())
                    decay_g = -gdn.A_log.float().exp() * torch.nn.functional.softplus(
                        gdn.in_proj_a(x_norm).float() + gdn.dt_bias.float()
                    )
                    c_chk, _ = chunk_gated_delta_rule(q_g, k_g, v_g, decay_g, beta_g, None)
                    c_rec, _ = recurrent_gated_delta_rule(q_g, k_g, v_g, decay_g, beta_g, None)
                    core_chk = gdn.norm(c_chk.reshape(-1, gdn.head_v_dim), z_g.reshape(-1, gdn.head_v_dim)).reshape(
                        trace_t, gdn.value_dim
                    )
                    core_rec = gdn.norm(c_rec.reshape(-1, gdn.head_v_dim), z_g.reshape(-1, gdn.head_v_dim)).reshape(
                        trace_t, gdn.value_dim
                    )
                    mix_rec = x_in[0] + gdn.out_proj(core_rec)
                    diag["l0_subops"] = {
                        "qkv_raw": round(flat_cos(qkv_raw[0].float().cpu().numpy(), got_trace["l0_qkv_raw"][:trace_t]), 8),
                        "qkv_conv": round(flat_cos(qkv_conv[0].float().cpu().numpy(), got_trace["l0_qkv_conv"][:trace_t]), 8),
                        "core_vs_chunk": round(flat_cos(core_chk.float().cpu().numpy(), got_trace["l0_core_out"][:trace_t]), 8),
                        "core_vs_rec": round(flat_cos(core_rec.float().cpu().numpy(), got_trace["l0_core_out"][:trace_t]), 8),
                        "torch_chunk_vs_rec": round(flat_cos(core_chk.float().cpu().numpy(), core_rec.float().cpu().numpy()), 8),
                        "mixer_vs_rec": round(flat_cos(mix_rec.float().cpu().numpy(), got_trace["mixer_res"][0][:trace_t]), 8),
                    }
                if layer_idx == 3 and got_trace.get("l3_q_ready") is not None:
                    sa = layer.self_attn
                    q_a, g_a = sa.q_proj(x_norm).view(1, trace_t, sa.num_heads, 2 * sa.head_dim).chunk(2, dim=-1)
                    g_a = g_a.reshape(1, trace_t, sa.num_heads * sa.head_dim)
                    q_a = apply_neox_rope(sa.q_norm(q_a), ctx_t.cos, ctx_t.sin)
                    k_a = apply_neox_rope(
                        sa.k_norm(sa.k_proj(x_norm).view(1, trace_t, sa.num_kv_heads, sa.head_dim)), ctx_t.cos, ctx_t.sin
                    )
                    v_a = sa.v_proj(x_norm).view(1, trace_t, sa.num_kv_heads, sa.head_dim)
                    att = grouped_attention(
                        q_a.transpose(1, 2), k_a.transpose(1, 2), v_a.transpose(1, 2), None,
                        is_causal=trace_t > 1, scale=sa.head_dim**-0.5
                    ).transpose(1, 2).reshape(1, trace_t, sa.num_heads * sa.head_dim)
                    att_gated = (att * torch.sigmoid(g_a))[0]
                    diag["l3_subops"] = {
                        "q_ready": round(flat_cos(q_a[0].reshape(trace_t, -1).float().cpu().numpy(), got_trace["l3_q_ready"][:trace_t]), 8),
                        "attn_out": round(flat_cos(att_gated.float().cpu().numpy(), got_trace["l3_attn_out"][:trace_t]), 8),
                    }
                x_post = layer.post_attention_layernorm(x_mix).to(dtype)
                r_probs = torch.softmax(layer.mlp.gate(x_post[0]).float(), dim=-1)
                _, r_ids = r_probs.topk(layer.mlp.top_k, dim=-1)
                r_np = r_ids.cpu().numpy()
                g_np = got_trace["topk_ids"][layer_idx][:trace_t]
                hit = sum(int(len(set(r_np[m].tolist()) & set(g_np[m].tolist()))) for m in range(trace_t))
                diag["router_agree"].append(round(hit / max(1, trace_t * layer.mlp.top_k), 6))
                x_out = x_mix + layer.mlp(x_post)
                x_out_np = x_out[0].float().cpu().numpy()
                got_l_np = got_trace["layer_res"][layer_idx][:trace_t]
                diag["mixer_cos"].append(
                    round(flat_cos(x_mix[0].float().cpu().numpy(), got_trace["mixer_res"][layer_idx][:trace_t]), 8)
                )
                diag["layer_cos"].append(round(flat_cos(x_out_np, got_l_np), 8))
                diag["layer_row_min_cos"].append(round(float(np.min(row_cosines(x_out_np, got_l_np))), 8))
                # FP32 residual comparison (same as g6tts d_residual!)
                xf = h0_f32.to(dev).unsqueeze(0)
                xf = xf + mixer(layer.input_layernorm(xf).to(dtype), ctx_t).float()
                xf = xf + layer.mlp(layer.post_attention_layernorm(xf).to(dtype)).float()
                diag["layer_cos_f32res"].append(
                    round(flat_cos(xf[0].cpu().numpy(), got_l_np), 8)
                )
                h0_f32 = xf[0].cpu()
                # Independent PyTorch T=64 run (measures PyTorch T=261 vs PyTorch T=64 self-consistency!)
                xt = h0_t64.to(dev).unsqueeze(0)
                h0_t64 = layer(xt, ctx_t)[0].cpu()
                # Controlled same-router FP32-res PyTorch run replaying g6tts's exact topk_ids
                xsr = h0_sr.to(dev).unsqueeze(0)
                xsr = xsr + mixer(layer.input_layernorm(xsr).to(dtype), ctx_t).float()
                xsr_post = layer.post_attention_layernorm(xsr).to(dtype)
                flat_sr = xsr_post.reshape(-1, xsr_post.shape[-1])
                probs_sr = torch.softmax(layer.mlp.gate(flat_sr).float(), dim=-1)
                g_ids_t = torch.from_numpy(g_np.copy()).to(device=dev, dtype=torch.long)
                w_sr = probs_sr.gather(-1, g_ids_t)
                if layer.mlp.norm_topk_prob:
                    w_sr = w_sr / w_sr.sum(dim=-1, keepdim=True)
                exp_sr = layer.mlp.experts(flat_sr, g_ids_t, w_sr.to(dtype))
                sh_sr = torch.sigmoid(layer.mlp.shared_expert_gate(flat_sr)) * layer.mlp.shared_expert(flat_sr)
                xsr = xsr + (exp_sr + sh_sr).view_as(xsr_post).float()
                h0_sr = xsr[0].cpu()

            for si, h_cpu in enumerate(hidden_seqs):
                t_len = int(h_cpu.shape[0])
                pos = torch.arange(t_len, dtype=torch.long, device=dev)
                ctx = LayerContext(
                    input_pos=pos,
                    cos=cos_all[:t_len],
                    sin=sin_all[:t_len],
                    kv_len=t_len,
                    continue_state=False,
                    use_cache=False,
                )
                out = layer(h_cpu.to(dev).unsqueeze(0), ctx)[0]
                hidden_seqs[si] = out.cpu()

        del layer
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    del cos_all, sin_all, exponents, frequencies, all_positions, angles

    # 3. Final norm + sliced lm_head over the 4097 legal main candidates
    norm = ZeroCenteredRMSNorm(cfg.text.hidden_size, cfg.text.rms_norm_eps, dtype=dtype, device=dev).eval()
    norm.weight.data.copy_(_safe_to_device(vt_state["model.language_model.norm.weight"], dev, dtype))
    cand_ids = torch.cat(
        (
            torch.tensor([cfg.eos_token_id], dtype=torch.long, device="cpu"),
            torch.arange(cfg.semantic_start_token_id, cfg.semantic_end_token_id + 1, dtype=torch.long, device="cpu"),
        )
    ).sort().values
    lm_head_sliced = _safe_to_device(vt_state["lm_head.weight"][cand_ids], dev, dtype)

    normed_seqs: list[torch.Tensor] = []
    hid_np_list: list[np.ndarray] = []
    main_np_list: list[np.ndarray] = []
    with torch.inference_mode():
        for h_cpu in hidden_seqs:
            h_norm = norm(h_cpu.to(dev)).to(dtype)
            normed_seqs.append(h_norm.cpu())
            hid_np_list.append(h_norm.float().cpu().numpy().astype(np.float32))
            mlogits = torch.nn.functional.linear(h_norm, lm_head_sliced)
            main_np_list.append(mlogits.float().cpu().numpy().astype(np.float32))
        if diag is not None and got_trace is not None and got_trace.get("final_hidden") is not None:
            def _cmp_hid_pair(a_t: torch.Tensor, b_t: torch.Tensor) -> dict[str, Any]:
                a_n = a_t.float().cpu().numpy()
                b_n = b_t.float().cpu().numpy()
                rc = row_cosines(a_n, b_n)
                la = torch.nn.functional.linear(a_t.to(dtype), lm_head_sliced).float().argmax(dim=-1).cpu().numpy()
                lb = torch.nn.functional.linear(b_t.to(dtype), lm_head_sliced).float().argmax(dim=-1).cpu().numpy()
                return {
                    "hid_cos_mean": round(float(np.mean(rc)), 8),
                    "hid_cos_min": round(float(np.min(rc)), 8),
                    "worst_row": int(np.argmin(rc)),
                    "main_argmax_agree": round(float(np.mean(la == lb)), 6),
                }

            h_t261 = normed_seqs[0][:trace_t].to(dev)
            h_t64_n = norm(h0_t64.to(dev)).to(dtype)
            h_f32_n = norm(h0_f32.to(dev)).to(dtype)
            h_sr_n = norm(h0_sr.to(dev)).to(dtype)
            h_g6_t = torch.from_numpy(got_trace["final_hidden"][:trace_t].copy()).to(dev, dtype)
            diag["torch_t261_vs_t64"] = _cmp_hid_pair(h_t261, h_t64_n)
            diag["torch_bf16_vs_f32res"] = _cmp_hid_pair(h_t64_n, h_f32_n)
            diag["g6tts_vs_torch_f32res"] = _cmp_hid_pair(h_f32_n, h_g6_t)
            diag["g6tts_vs_torch_same_router"] = _cmp_hid_pair(h_sr_n, h_g6_t)
    del norm, lm_head_sliced, hidden_seqs

    # 4. Teacher-forced codebook tower (4 shared + 9 residual + 9 heads)
    tower = FishOmniAudioDecoder(cfg, dtype=dtype, device=dev).eval()
    tw_items = [
        (k, v.clone() if v.device.type == "cpu" else v)
        for k, v in vt_state.items()
        if k.startswith(("audio_decoder.", "audio_decoder_mtp_shared.", "audio_decoder_mtp."))
    ]
    load_audio_decoder_weights(tower, tw_items)
    del vt_state, tw_items

    outs: list[dict[str, np.ndarray]] = []
    with torch.inference_mode():
        for si, (h_norm_cpu, ft, hid_np, main_np) in enumerate(zip(normed_seqs, frame_tensors, hid_np_list, main_np_list)):
            t_len = int(ft.shape[0])
            frames = ft.to(dev)
            h_in = h_norm_cpu[:-1].to(dev)
            cur_f = frames[:-1]
            nxt_f = frames[1:]
            p_pos = torch.arange(t_len - 1, device=dev, dtype=torch.long)
            sem = tower.is_semantic(nxt_f[:, 0])
            if si == 0 and diag is not None and got_trace is not None:
                unseen = tower.embedding_sum(cur_f, 1 + tower.config.vq_backbone_codebooks, tower.config.num_codebooks)
                unseen = torch.where(tower.is_semantic(cur_f[:, 0])[:, None], unseen, 0.0)
                th = h_in
                for idx_s, slayer in enumerate(tower.shared_layers):
                    th = slayer(th, nxt_f[:, 1], sem, p_pos, unseen if idx_s == 0 else None)
                    diag["tower_cos"].append(
                        round(flat_cos(th[: trace_t - 1].float().cpu().numpy(), got_trace["tower_hids"][idx_s][: trace_t - 1]), 8)
                    )
            else:
                th = tower.run_shared(h_in, nxt_f[:, 1], sem, p_pos, cur_f)
            head_list = []
            for idx, (rlayer, head) in enumerate(zip(tower.residual_layers, tower.codebook_heads)):
                th = rlayer(th, nxt_f[:, idx + 1], sem, p_pos)
                if si == 0 and diag is not None and got_trace is not None:
                    tl = len(tower.shared_layers) + idx
                    diag["tower_cos"].append(
                        round(flat_cos(th[: trace_t - 1].float().cpu().numpy(), got_trace["tower_hids"][tl][: trace_t - 1]), 8)
                    )
                hl = head(th)[:, :1024].float().cpu().numpy()
                head_list.append(hl)
            if si == 0 and diag is not None and got_trace is not None:
                # Compare s0 final_hidden (M=64) vs PyTorch and vs M=128 full-seq dump
                if got_trace.get("final_hidden") is not None:
                    rc_lt64 = row_cosines(hid_np[:trace_t], got_trace["final_hidden"][:trace_t])
                    diag["seq0_lt64_hid_cos"] = {
                        "mean": round(float(np.mean(rc_lt64)), 8),
                        "min": round(float(np.min(rc_lt64)), 8),
                        "worst_row": int(np.argmin(rc_lt64)),
                    }
                    if got_seq0_hid is not None:
                        rc_m = row_cosines(got_seq0_hid[:trace_t], got_trace["final_hidden"][:trace_t])
                        diag["seq0_m128_vs_m64_cos"] = {
                            "mean": round(float(np.mean(rc_m)), 8),
                            "min": round(float(np.min(rc_m)), 8),
                        }
                # Isolated tower check: feed g6tts's EXACT s0 final_hidden (M=64) into PyTorch tower!
                iso_src = got_trace.get("final_hidden")
                if iso_src is None:
                    iso_src = got_seq0_hid
                if iso_src is not None:
                    t_iso = trace_t - 1
                    h_iso = torch.from_numpy(iso_src[:t_iso].copy()).to(dev, dtype)
                    cur_iso = frames[:t_iso]
                    nxt_iso = frames[1 : t_iso + 1]
                    pos_iso = torch.arange(t_iso, device=dev, dtype=torch.long)
                    sem_iso = tower.is_semantic(nxt_iso[:, 0])
                    unseen_iso = tower.embedding_sum(
                        cur_iso, 1 + tower.config.vq_backbone_codebooks, tower.config.num_codebooks
                    )
                    unseen_iso = torch.where(tower.is_semantic(cur_iso[:, 0])[:, None], unseen_iso, 0.0)

                    def _probe_tower_layer(
                        tlayer: Any,
                        h_in_l: torch.Tensor,
                        codes_l: torch.Tensor,
                        shared_in_l: torch.Tensor | None,
                        got_sub: dict[str, np.ndarray],
                        got_out_l: np.ndarray,
                    ) -> tuple[torch.Tensor, dict[str, Any]]:
                        emb = tlayer.semantic_embedding if tlayer.shared else tlayer.codebook_embedding
                        c_cl = codes_l.clamp(0, emb.num_embeddings - 1)
                        e_v = torch.where(sem_iso[:, None], emb(c_cl), 0.0)
                        if shared_in_l is not None:
                            e_v = e_v + shared_in_l
                        eh_cat = torch.cat((tlayer.enorm(e_v), tlayer.hnorm(h_in_l)), dim=-1)
                        res = tlayer.eh_proj(eh_cat)
                        tl_m = tlayer.transformer_layer
                        att_m = tl_m.attention
                        c_cfg = att_m.config
                        qw_l, kvw_l = c_cfg.n_head * c_cfg.head_dim, c_cfg.n_local_heads * c_cfg.head_dim
                        wqkv_v = att_m.wqkv(tl_m.attention_norm(res).to(dtype))
                        q_t, g_t, k_t, v_t = wqkv_v.split((qw_l, qw_l, kvw_l, kvw_l), -1)
                        q_t = att_m.rotate(att_m.q_norm(q_t.reshape(t_iso, c_cfg.n_head, c_cfg.head_dim)), pos_iso)
                        k_t = att_m.rotate(att_m.k_norm(k_t.reshape(t_iso, c_cfg.n_local_heads, c_cfg.head_dim)), pos_iso)
                        v_t = v_t.reshape(t_iso, c_cfg.n_local_heads, c_cfg.head_dim)
                        mask_l = pos_iso[None, :] <= pos_iso[:, None]
                        att_o = grouped_attention(
                            q_t.transpose(0, 1).unsqueeze(0),
                            k_t.transpose(0, 1).unsqueeze(0),
                            v_t.transpose(0, 1).unsqueeze(0),
                            mask_l,
                        ).squeeze(0).transpose(0, 1).reshape(t_iso, qw_l)
                        att_g = att_o * g_t.sigmoid()
                        wo_res = res + att_m.wo(att_g)
                        ff_in = tl_m.ffn_norm(wo_res).to(dtype)
                        ff_act = torch.nn.functional.silu(tl_m.feed_forward.w1(ff_in)) * tl_m.feed_forward.w3(ff_in)
                        w2_res = wo_res + tl_m.feed_forward.w2(ff_act)
                        out_l = tlayer.final_layernorm(w2_res).to(dtype)
                        sub_d = {
                            "eh_cat": round(flat_cos(eh_cat.float().cpu().numpy(), got_sub["eh_cat"][:t_iso]), 8),
                            "eh_proj": round(flat_cos(res.float().cpu().numpy(), got_sub["eh_proj"][:t_iso]), 8),
                            "wqkv": round(flat_cos(wqkv_v.float().cpu().numpy(), got_sub["wqkv"][:t_iso]), 8),
                            "q_ready": round(flat_cos(q_t.reshape(t_iso, qw_l).float().cpu().numpy(), got_sub["q_ready"][:t_iso]), 8),
                            "attn_out": round(flat_cos(att_g.float().cpu().numpy(), got_sub["attn_out"][:t_iso]), 8),
                            "wo_res": round(flat_cos(wo_res.float().cpu().numpy(), got_sub["wo_res"][:t_iso]), 8),
                            "ffn_act": round(flat_cos(ff_act.float().cpu().numpy(), got_sub["ffn_act"][:t_iso]), 8),
                            "w2_res": round(flat_cos(w2_res.float().cpu().numpy(), got_sub["w2_res"][:t_iso]), 8),
                            "final_ln": round(flat_cos(out_l.float().cpu().numpy(), got_out_l[:t_iso]), 8),
                            "ffn_act_max_abs": round(float(ff_act.float().abs().max().item()), 2),
                            "w2_res_max_abs": round(float(w2_res.float().abs().max().item()), 2),
                        }
                        return out_l, sub_d

                    th_iso = h_iso
                    th_iso_list: list[np.ndarray] = []
                    got_sub_list = got_trace.get("tower_sub") or []
                    for idx_s, slayer in enumerate(tower.shared_layers):
                        if idx_s == 0 and len(got_sub_list) >= 1:
                            th_iso, diag["tower_subops"]["shared0"] = _probe_tower_layer(
                                slayer, th_iso, nxt_iso[:, 1], unseen_iso, got_sub_list[0], got_trace["tower_hids"][0]
                            )
                        else:
                            th_iso = slayer(th_iso, nxt_iso[:, 1], sem_iso, pos_iso, unseen_iso if idx_s == 0 else None)
                        th_np = th_iso.float().cpu().numpy()
                        th_iso_list.append(th_np)
                        g_np_t = got_trace["tower_hids"][idx_s][:t_iso]
                        diag["tower_cos_isolated"].append(round(flat_cos(th_np, g_np_t), 8))
                        diag["tower_row_min_cos_isolated"].append(round(float(np.min(row_cosines(th_np, g_np_t))), 8))
                    for idx_r, rlayer in enumerate(tower.residual_layers):
                        tl = len(tower.shared_layers) + idx_r
                        if idx_r == 0 and len(got_sub_list) >= 2:
                            th_iso, diag["tower_subops"]["res0"] = _probe_tower_layer(
                                rlayer, th_iso, nxt_iso[:, 1], None, got_sub_list[1], got_trace["tower_hids"][tl]
                            )
                        else:
                            th_iso = rlayer(th_iso, nxt_iso[:, idx_r + 1], sem_iso, pos_iso)
                        th_np = th_iso.float().cpu().numpy()
                        th_iso_list.append(th_np)
                        g_np_t = got_trace["tower_hids"][tl][:t_iso]
                        diag["tower_cos_isolated"].append(round(flat_cos(th_np, g_np_t), 8))
                        diag["tower_row_min_cos_isolated"].append(round(float(np.min(row_cosines(th_np, g_np_t))), 8))

                    # Self-consistency baseline: cast tower to FP32 on the EXACT same h_iso (only on seq 0 after head_list done? Wait: tower is needed for seq 1..15!)
                    # Cast one layer at a time in a cloned FP32 copy would cost 0 permanent VRAM, or run after the seq loop!
                    diag["_iso_bf16_list"] = th_iso_list
                    diag["_iso_h_src"] = iso_src[:t_iso].copy()
                    diag["_iso_frames"] = seqs[0][:trace_t].copy()
            head_arr = np.stack(head_list, axis=1).astype(np.float32)  # [T-1, 9, 1024]
            outs.append(
                {
                    "hidden": hid_np,
                    "main_logits": main_np,
                    "head_logits": head_arr,
                }
            )

    if diag is not None and "_iso_bf16_list" in diag:
        iso_bf16_list = diag.pop("_iso_bf16_list")
        iso_h_src = diag.pop("_iso_h_src")
        iso_frames = diag.pop("_iso_frames")
        with torch.inference_mode():
            tower = tower.float()
            t_iso = int(iso_h_src.shape[0])
            fr_f = torch.from_numpy(iso_frames).to(device=dev, dtype=torch.long)
            cur_f = fr_f[:t_iso]
            nxt_f = fr_f[1 : t_iso + 1]
            pos_f = torch.arange(t_iso, device=dev, dtype=torch.long)
            sem_f = tower.is_semantic(nxt_f[:, 0])
            unseen_f = tower.embedding_sum(
                cur_f, 1 + tower.config.vq_backbone_codebooks, tower.config.num_codebooks
            )
            unseen_f = torch.where(tower.is_semantic(cur_f[:, 0])[:, None], unseen_f, 0.0)
            th_f = torch.from_numpy(iso_h_src).to(device=dev, dtype=torch.float32)
            g6_vs_f32: list[float] = []
            for idx_s, slayer in enumerate(tower.shared_layers):
                th_f = slayer(th_f, nxt_f[:, 1], sem_f, pos_f, unseen_f if idx_s == 0 else None)
                f_np = th_f.cpu().numpy()
                diag["tower_torch_bf16_vs_fp32"].append(round(flat_cos(iso_bf16_list[idx_s], f_np), 8))
                if got_trace is not None:
                    g6_vs_f32.append(round(flat_cos(f_np, got_trace["tower_hids"][idx_s][:t_iso]), 8))
            for idx_r, rlayer in enumerate(tower.residual_layers):
                tl = len(tower.shared_layers) + idx_r
                th_f = rlayer(th_f, nxt_f[:, idx_r + 1], sem_f, pos_f)
                f_np = th_f.cpu().numpy()
                diag["tower_torch_bf16_vs_fp32"].append(round(flat_cos(iso_bf16_list[tl], f_np), 8))
                if got_trace is not None:
                    g6_vs_f32.append(round(flat_cos(f_np, got_trace["tower_hids"][tl][:t_iso]), 8))
            diag["tower_g6tts_vs_torch_fp32"] = g6_vs_f32

    del tower, normed_seqs, frame_tensors
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return outs, diag


def main() -> int:
    ap = argparse.ArgumentParser(description="Gate E1 teacher-forced parity checker for g6tts")
    ap.add_argument("--model-dir", required=True, help="Release directory (e.g. /work-data/out/dev-ref)")
    ap.add_argument("--manifest", default="", help="Optional trellis manifest name (empty for dev-ref)")
    ap.add_argument("--fish-speech", required=True, help="Path to fish-speech @ 0f8a12e8 checkout")
    ap.add_argument("--exlr8-repo", default="", help="Optional path to exlr8 repo containing loader/exlr8.py")
    ap.add_argument("--rows", required=True, help="Input ROWS.bin (G6TR format)")
    ap.add_argument("--bench-bin", default="bin/g6tts_bench", help="Path to compiled g6tts_bench binary")
    ap.add_argument("--weight-mode", default="ref", choices=["ref", "ref-fp16", "native"])
    ap.add_argument("--dump-in", default="", help="Skip invoking g6tts_bench and compare an existing DUMP.bin")
    ap.add_argument("--ref-dump", default="", help="Optional cached reference DUMP.bin (skip PyTorch)")
    ap.add_argument("--no-hash", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--json-out", default="", help="Optional path to write Gate E1 JSON summary")
    args = ap.parse_args()

    seqs = read_rows_bin(Path(args.rows))
    if not seqs:
        print("parity_tts.py: empty ROWS.bin", file=sys.stderr)
        return 2

    got_trace: dict[str, Any] | None = None
    if args.dump_in:
        got_outs = read_dump_bin(Path(args.dump_in))
        tpath = Path(args.dump_in + ".trace0")
        if tpath.is_file():
            got_trace = read_layer_trace_bin(tpath)
    else:
        with tempfile.TemporaryDirectory() as td:
            dump_path = Path(td) / "g6tts_dump.bin"
            cmd = [
                args.bench_bin,
                "--model-dir",
                args.model_dir,
                "--weight-mode",
                args.weight_mode,
                "--teacher",
                args.rows,
                "--dump",
                str(dump_path),
            ]
            if args.manifest:
                cmd += ["--manifest", args.manifest]
            if args.no_hash:
                cmd.append("--no-hash")
            subprocess.run(cmd, check=True)
            got_outs = read_dump_bin(dump_path)
            tpath = Path(str(dump_path) + ".trace0")
            if tpath.is_file():
                got_trace = read_layer_trace_bin(tpath)

    diag: dict[str, Any] | None = None
    if args.ref_dump:
        ref_outs = read_dump_bin(Path(args.ref_dump))
    else:
        ref_outs, diag = run_torch_reference(
            Path(args.model_dir),
            args.manifest or None,
            Path(args.fish_speech),
            Path(args.exlr8_repo) if args.exlr8_repo else None,
            seqs,
            args.device,
            got_trace=got_trace,
            got_seq0_hid=got_outs[0]["hidden"] if got_outs else None,
        )

    passed, summary = evaluate_gate_e1(ref_outs, got_outs, seqs=seqs)
    if diag is not None:
        summary["diag_seq0"] = diag
    text = json.dumps(summary, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
