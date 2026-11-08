# ostore

Output-only derivative of our `k_attn_cpa_t` at `4fb7f0bf`, starting from off.
All attention arithmetic and final division/BF16 rounding remain unchanged.
All off dispatch cases use the new epilogue, including window and floor cases.

The final BF16 fragment pairs go into the now-dead Q shared-memory tile.
Its existing 520-element row stride makes each warp's pair stores use distinct
32-bit banks. One CTA barrier publishes the tile; each thread then copies four
aligned `uint4` vectors to output. Each warp covers four contiguous 128-byte
spans per vector-store instruction, instead of partially occupied sectors.
There is no allocation, pre-pass, extra shared-memory capacity, or input write.
The extra shared traffic and barrier may outweigh the saved global transactions;
only measurement can establish the net effect.

Checks from repository root:

```sh
python3 engine/attn/sweep/variants/ostore/test_layout.py
python3 engine/attn/sweep/run.py --arch 120a --variants off ostore --suite smoke --out /tmp/ostore-smoke
python3 engine/attn/sweep/run.py --arch 120a --variants off ostore --suite all --out /tmp/ostore-all
```

The CPU check proves the coordinate mapping and checks the unchanged arithmetic
source, not runtime exactness. The CUDA suites provide the byte comparison.
