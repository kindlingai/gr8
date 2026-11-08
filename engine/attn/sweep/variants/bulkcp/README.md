# bulkcp

Private attention-sweep variant. Our `ds41_attn_pipe.cuh` kernel at
`4fb7f0bf` supplies the arithmetic and pipeline; the packed-row transfer is
replaced by one 288-byte `cp.async.bulk` per live row, completed through a
reusable 64-arrival mbarrier. The barrier adds 16 bytes to the dynamic
allocation (8 bytes used, 16-byte layout alignment retained). Null rows are
explicitly zeroed and arrive with zero expected transaction bytes.

QK/PV accumulation, online-softmax reductions and bf16 conversion order are
unchanged. Q stays register resident. The ordinary bf16 K tile retains the
pipe layout (no XOR swizzle). Mixed window/CQ tiles retain the original
copy/dequant path; whole-CQ tiles use bulk transfers. Window-only and
bounded-replay floor cases dispatch the independent off kernel.

`launch` enqueues one kernel: no pre-pass, scratch allocation, host sync,
reference reads, or input writes. Only `p.out` is written in global memory.

## Reproduce

From the repository root with harness `72c5e43b99b782d3a3821e656af758ae93a2f5f7`:

```sh
python3 -m unittest discover -s engine/attn/sweep -p test_sweep.py -v
python3 engine/attn/sweep/run.py --arch 120a --variants off bulkcp \
  --suite smoke --out /tmp/bulkcp-smoke-UNIQUE
python3 engine/attn/sweep/run.py --arch 120a --variants off bulkcp \
  --suite all --out /tmp/bulkcp-all-UNIQUE
```

The revised sweep gate accepts local smoke plus zero stack/spills; the all
command is optional locally, with full testing owned by the GB10 bench lead.
Shared RTX 5090 timing is a hint, not a keep/drop decision. GB10 validation
is owned by the sweep bench lead; this variant's author did not access it.

## Provenance

All implementation code is ours. Source copied and modified:
`engine/attn/ds41_attn_pipe.cuh:52-266` at `4fb7f0bf`. Helpers come from our
existing `ds41a` namespace. NVIDIA PTX ISA documentation was consulted for
instruction syntax and memory-ordering semantics, not implementation code:
https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html
(sections `cp.async.bulk`, `mbarrier.arrive`, `mbarrier.try_wait`, and `fence`).
