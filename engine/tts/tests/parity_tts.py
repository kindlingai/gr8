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
import json
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


def run_torch_reference(
    model_dir: Path,
    manifest: str | None,
    fish_speech_dir: Path,
    exlr8_repo: Path | None,
    seqs: list[np.ndarray],
    device: str,
) -> list[dict[str, np.ndarray]]:
    import torch

    sys.path.insert(0, str(fish_speech_dir))
    if exlr8_repo is not None:
        sys.path.insert(0, str(exlr8_repo))

    from exlr8.loader.exlr8 import load_drama3
    from fish_speech.models.text2semantic.fish_omni.backbone import (
        FishOmniBackbone,
        load_backbone_weights,
    )
    from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig
    from fish_speech.models.text2semantic.fish_omni.tower import (
        FishOmniAudioDecoder,
        load_audio_decoder_weights,
    )

    cfg_dict = json.loads((model_dir / "config.json").read_text())
    cfg = FishOmniConfig.from_dict(cfg_dict)

    # Load weights from the exact same release (keeps memory bounded via CPU state dict -> device).
    sd = load_drama3(str(model_dir), manifest_name=manifest, dtype=torch.bfloat16, device="cpu", verify=False)
    bb_items = []
    tw_items = []
    for k, v in sd.items():
        t = v.clone()
        if k.startswith(("audio_decoder.", "audio_decoder_mtp_shared.", "audio_decoder_mtp.")):
            tw_items.append((k, t))
        else:
            bb_items.append((k, t))

    backbone = FishOmniBackbone(cfg.text, fp8_experts=False, dtype=torch.bfloat16, device=device)
    load_backbone_weights(backbone, bb_items)
    backbone.eval()

    tower = FishOmniAudioDecoder(cfg, dtype=torch.bfloat16, device=device)
    load_audio_decoder_weights(tower, tw_items)
    tower.eval()

    cand_ids = torch.cat(
        (
            torch.tensor([cfg.eos_token_id], device=device),
            torch.arange(cfg.semantic_start_token_id, cfg.semantic_end_token_id + 1, device=device),
        )
    ).sort().values

    outs: list[dict[str, np.ndarray]] = []
    with torch.inference_mode():
        for arr in seqs:
            t_len = arr.shape[0]
            frames = torch.from_numpy(arr.copy()).to(device=device, dtype=torch.long)
            pos = torch.arange(t_len, device=device, dtype=torch.long)
            emb = tower.embed_backbone(frames, backbone.embed(frames[:, 0]))
            hidden = backbone(inputs_embeds=emb[None], input_pos=pos)[0]  # [T, H]
            main_logits = backbone.compute_logits(hidden)[:, cand_ids]   # [T, 4097]

            # Teacher-forced tower over rows 0..T-2 labelled with frames[1..T-1]
            cur_f = frames[:-1]
            nxt_f = frames[1:]
            p_pos = pos[:-1]
            sem = tower.is_semantic(nxt_f[:, 0])
            th = tower.run_shared(hidden[:-1], nxt_f[:, 1], sem, p_pos, cur_f)
            head_list = []
            for idx, (layer, head) in enumerate(zip(tower.residual_layers, tower.codebook_heads)):
                th = layer(th, nxt_f[:, idx + 1], sem, p_pos)
                hl = head(th)[:, :1024].float().cpu().numpy()
                head_list.append(hl)
            head_arr = np.stack(head_list, axis=1).astype(np.float32)  # [T-1, 9, 1024]

            outs.append(
                {
                    "hidden": hidden.float().cpu().numpy().astype(np.float32),
                    "main_logits": main_logits.float().cpu().numpy().astype(np.float32),
                    "head_logits": head_arr,
                }
            )
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
