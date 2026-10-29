"""The embedding rows engine/runtime/dflreplay.cu needs: every bonus token of a GR8_DFL_DUMP's draft steps, and the mask.

python3 engine/ref/dflreplay_emb.py DUMP --src DRAFTER --glm GLM_DIR --out DIR/emb.bin

DRAFTER: the drafter dir (config.json: dflash_config.mask_token_id). GLM_DIR: model-embed.safetensors (as
dflash2_replay.py). Writes int32 n, int32 ids[n], float32 rows[n][6144].
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dflash2_ref as R  # noqa: E402
from dflash2_replay import load_dump  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dump')
    ap.add_argument('--src', required=True)
    ap.add_argument('--glm', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    steps, _, _ = load_dump(a.dump)
    mask = json.load(open(os.path.join(a.src, 'config.json')))['dflash_config']['mask_token_id']
    ids = sorted({s['tok0'] for s in steps} | {mask})
    st = R.ST(os.path.join(a.glm, 'model-embed.safetensors'))
    rows = np.stack([R.bf(st.get('model.embed_tokens.weight', rows=(t, t + 1))[0]) for t in ids]).astype(np.float32)
    with open(a.out, 'wb') as f:
        np.array([len(ids)], np.int32).tofile(f)
        np.array(ids, np.int32).tofile(f)
        rows.tofile(f)
    print(f'{a.out}: {len(ids)} rows ({len(steps)} draft steps, mask {mask})')


if __name__ == '__main__':
    main()
