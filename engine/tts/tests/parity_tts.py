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
) -> tuple[bool, dict[str, Any]]:
    if len(ref_outs) != len(got_outs):
        raise ValueError(f"sequence count mismatch: ref={len(ref_outs)} vs got={len(got_outs)}")

    all_cos: list[np.ndarray] = []
    main_match = 0
    main_total = 0
    head_match = [0] * 9
    head_total = [0] * 9

    for ref, got in zip(ref_outs, got_outs):
        all_cos.append(row_cosines(ref["hidden"], got["hidden"]))
        ref_m = np.argmax(ref["main_logits"], axis=-1)
        got_m = np.argmax(got["main_logits"], axis=-1)
        main_match += int(np.sum(ref_m == got_m))
        main_total += int(ref_m.size)
        for h in range(9):
            ref_h = np.argmax(ref["head_logits"][:, h, :1024], axis=-1)
            got_h = np.argmax(got["head_logits"][:, h, :1024], axis=-1)
            head_match[h] += int(np.sum(ref_h == got_h))
            head_total[h] += int(ref_h.size)

    cos_cat = np.concatenate(all_cos) if all_cos else np.array([0.0])
    cos_mean = float(np.mean(cos_cat))
    cos_min = float(np.min(cos_cat))
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
        "hidden_cos_min": round(cos_min, 8),
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


def run_torch_reference(
    model_dir: Path,
    manifest: str | None,
    fish_speech_dir: Path,
    exlr8_repo: Path | None,
    seqs: list[np.ndarray],
    device: str,
) -> list[dict[str, np.ndarray]]:
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
        for h_norm_cpu, ft, hid_np, main_np in zip(normed_seqs, frame_tensors, hid_np_list, main_np_list):
            t_len = int(ft.shape[0])
            frames = ft.to(dev)
            h_in = h_norm_cpu[:-1].to(dev)
            cur_f = frames[:-1]
            nxt_f = frames[1:]
            p_pos = torch.arange(t_len - 1, device=dev, dtype=torch.long)
            sem = tower.is_semantic(nxt_f[:, 0])
            th = tower.run_shared(h_in, nxt_f[:, 1], sem, p_pos, cur_f)
            head_list = []
            for idx, (rlayer, head) in enumerate(zip(tower.residual_layers, tower.codebook_heads)):
                th = rlayer(th, nxt_f[:, idx + 1], sem, p_pos)
                hl = head(th)[:, :1024].float().cpu().numpy()
                head_list.append(hl)
            head_arr = np.stack(head_list, axis=1).astype(np.float32)  # [T-1, 9, 1024]
            outs.append(
                {
                    "hidden": hid_np,
                    "main_logits": main_np,
                    "head_logits": head_arr,
                }
            )

    del tower, normed_seqs, frame_tensors
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return outs


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

    if args.ref_dump:
        ref_outs = read_dump_bin(Path(args.ref_dump))
    else:
        ref_outs = run_torch_reference(
            Path(args.model_dir),
            args.manifest or None,
            Path(args.fish_speech),
            Path(args.exlr8_repo) if args.exlr8_repo else None,
            seqs,
            args.device,
        )

    if args.dump_in:
        got_outs = read_dump_bin(Path(args.dump_in))
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

    passed, summary = evaluate_gate_e1(ref_outs, got_outs)
    text = json.dumps(summary, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
