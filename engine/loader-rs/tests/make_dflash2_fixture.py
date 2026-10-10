#!/usr/bin/env python3
"""Writes tests/data/dflash2/: a small synthetic DFlash2 drafter (src/config.json + src/model.safetensors, hidden 64,
2 layers) and what tools/convert/dflash2.py makes of it at TP=4: every rank's manifest and pack sha256 with the
default MXFP8 records, and rank 2's with the attn and sel classes kept in BF16. tests/dflash2.rs converts the same
drafter with src/dflash2.rs and compares. The weights mix scales from 2**-20 to 2**10 per row and carry zero groups,
-0.0, BF16 subnormals, values near the BF16 maximum and E4M3 rounding ties. Needs numpy. Run from the repo root:

    python3 engine/loader-rs/tests/make_dflash2_fixture.py
"""
import hashlib
import json
import os
import shutil
import struct
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from tools.convert import dflash2  # noqa: E402
from tools.convert.st import ST  # noqa: E402

OUT = os.path.join(ROOT, 'engine', 'loader-rs', 'tests', 'data', 'dflash2')
HID, HD, NH, NKV, INTER, LAYERS, VOCAB, SEL = 64, 16, 8, 4, 128, 2, 32, 16
NR = 4


def bf16(rng, shape):
    """Random BF16 bits: normal values at a per-row scale, plus the edge cases the quantizer must round alike."""
    rows = shape[0] if len(shape) > 1 else 1
    x = rng.standard_normal(shape).astype(np.float32).reshape(rows, -1)
    x *= np.exp2(rng.integers(-20, 11, size=(rows, 1))).astype(np.float32)
    b = (x.view(np.uint32) >> 16).astype(np.uint16)
    flat = b.reshape(-1)
    n = flat.size
    if n >= 256:
        flat[0:32] = 0                                        # a whole zero group: amax clamps to float32 tiny
        flat[32:40] = 0x8000                                  # -0.0
        flat[40:48] = rng.integers(1, 0x80, 8)                # BF16 subnormals
        flat[64:96] = rng.integers(1, 0x80, 32)               # a group of only subnormals
        flat[96] = 0x7f7f                                     # near the BF16 maximum
        flat[97] = 0xff7f
        flat[128:160] = [0x43e0] + [0x3f88, 0x3f98, 0x3fa8, 0x3fb8, 0xbf88, 0xbf98] * 5 + [0x3c90]  # 448 and ties
    return b.reshape(shape)


def write_st(path, tensors):
    off, header, blobs = 0, {}, []
    for name, a in tensors.items():
        raw = np.ascontiguousarray(a, '<u2').tobytes()
        header[name] = dict(dtype='BF16', shape=list(a.shape), data_offsets=[off, off + len(raw)])
        blobs.append(raw)
        off += len(raw)
    h = json.dumps(header, separators=(',', ':')).encode()
    h += b' ' * (-len(h) % 8)
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(h)) + h + b''.join(blobs))


def main():
    rng = np.random.default_rng(53)
    t = {'fc.weight': bf16(rng, (HID, 6 * HID)), 'hidden_norm.weight': bf16(rng, (HID,)), 'norm.weight': bf16(rng, (HID,))}
    for j in range(LAYERS):
        p = f'layers.{j}.'
        t |= {p + 'input_layernorm.weight': bf16(rng, (HID,)), p + 'post_attention_layernorm.weight': bf16(rng, (HID,)),
              p + 'self_attn.q_norm.weight': bf16(rng, (HD,)), p + 'self_attn.k_norm.weight': bf16(rng, (HD,)),
              p + 'self_attn.q_proj.weight': bf16(rng, (NH * HD, HID)),
              p + 'self_attn.k_proj.weight': bf16(rng, (NKV * HD, HID)),
              p + 'self_attn.v_proj.weight': bf16(rng, (NKV * HD, HID)),
              p + 'self_attn.o_proj.weight': bf16(rng, (HID, NH * HD)),
              p + 'mlp.gate_proj.weight': bf16(rng, (INTER, HID)), p + 'mlp.up_proj.weight': bf16(rng, (INTER, HID)),
              p + 'mlp.down_proj.weight': bf16(rng, (HID, INTER))}
        for s in ('attention_conv', 'mlp_conv'):
            t |= {p + f'{s}.kernel_projection.weight': bf16(rng, (HID // 4, HID)), p + f'{s}.base_kernel': bf16(rng, (2, 2, HID))}
    t |= {'candidate_selector.hidden_projection.weight': bf16(rng, (SEL, HID)),
          'candidate_selector.predecessor_codebook': bf16(rng, (VOCAB, SEL)),
          'candidate_selector.successor_codebook': bf16(rng, (VOCAB, SEL))}
    cfg = dict(attention_bias=False, head_dim=HD, hidden_size=HID, intermediate_size=INTER, is_causal=False,
               num_attention_heads=NH, num_hidden_layers=LAYERS, num_key_value_heads=NKV, rms_norm_eps=1e-05,
               rope_parameters=dict(rope_theta=1000000, rope_type='default'), sliding_window=2048,
               use_sliding_window=True, vocab_size=VOCAB,
               dflash_config=dict(block_size=8, conv_group_size=16, conv_kernel_size=2, mask_token_id=30,
                                  selector_rank=SEL, selector_top_k=4, target_layer_ids=[1, 3, 5, 7, 9, 11]))
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(os.path.join(OUT, 'src'))
    with open(os.path.join(OUT, 'src', 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    write_st(os.path.join(OUT, 'src', 'model.safetensors'), t)

    keep = dflash2.HID, dflash2.HD, dflash2.NH, dflash2.NKV, dflash2.INTER
    dflash2.HID, dflash2.HD, dflash2.NH, dflash2.NKV, dflash2.INTER = HID, HD, NH, NKV, INTER
    tmp = tempfile.mkdtemp()
    try:
        st = ST(os.path.join(OUT, 'src', 'model.safetensors'))
        for sub, ranks, classes in (('mxfp8', range(NR), set()), ('bf16-attn-sel', [2], {'attn', 'sel'})):
            os.makedirs(os.path.join(OUT, sub))
            sums = {}
            for r in ranks:
                dflash2.convert_rank(st, cfg, NR, r, tmp, classes, False)
                shutil.copy(os.path.join(tmp, f'dflash.r{r}.manifest.json'), os.path.join(OUT, sub))
                with open(os.path.join(tmp, f'dflash.r{r}.pack'), 'rb') as f:
                    sums[f'dflash.r{r}.pack'] = hashlib.sha256(f.read()).hexdigest()
            with open(os.path.join(OUT, sub, 'packs.json'), 'w') as f:
                json.dump(sums, f, indent=1, sort_keys=True)
    finally:
        dflash2.HID, dflash2.HD, dflash2.NH, dflash2.NKV, dflash2.INTER = keep
        shutil.rmtree(tmp)


if __name__ == '__main__':
    main()
