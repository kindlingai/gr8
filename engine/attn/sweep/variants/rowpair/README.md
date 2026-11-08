# rowpair

A two-query-row CTA, derived from our original CPA attention kernel. No
change to key order, 64-key block boundaries, QK/PV MMA order, softmax
reduction, or rounding. This is an isolated sweep candidate, not an engine
change or a claimed speed improvement.

The CTA caches both queries in distributed registers and stages each query
through the existing shared Q tile. Its two accumulator sets remain separate.
After processing row 0 for a key block, row 1 retains each shared K slot whose
pointer equals row 0's pointer, and loads/widens only changed slots. Null
slots retain their zero fill only when row 0 actually processed the block.
Entirely masked blocks are skipped independently, as in the reference.

The output mapping is `(2 * blockIdx.x, 2 * blockIdx.x + 1)`, with an odd
last row guarded. Query rows and session slabs can differ; equality compares
the complete tagged row pointer, not merely a compressed key ID.

`TK == 0`, non-packed inputs, and single-row calls use the independent off
kernel. Paired packed-CQ calls include floor-masked and mixed window/CQ
blocks. There is no pre-pass, global scratch, allocation, or synchronization
on the host.

Tradeoffs: the K tile stays 64 rows and shared memory stays 91,008 bytes.
Two output accumulators and two distributed Q caches raise registers. Q must
be restaged for every query/block, and consecutive-row K reuse requires the
same key at the same slot. Set overlap alone does not suffice. In particular,
the sweep fixture draws independent sorted random top-k lists beyond 512
visible keys, so adjacent rows need not share many same-slot keys.

Validation (from repository root, native 5090 only):

```
flock /tmp/attn-sweep-5090.lock python3 engine/attn/sweep/run.py --arch 120a --variants off rowpair --suite smoke --out /tmp/rowpair-smoke
flock /tmp/attn-sweep-5090.lock python3 engine/attn/sweep/run.py --arch 120a --variants off rowpair --suite all --out /tmp/rowpair-all
```
