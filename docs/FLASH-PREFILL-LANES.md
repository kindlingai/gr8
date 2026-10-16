# Flash prefill: two full-sized lanes

Flash prefill now defaults to two lanes for MP up to 8192. `--pf-lanes 1`
restores the old serial path. Larger MPs retain the one-lane default pending
shared-scratch memory gates; `--pf-lanes 2` remains an explicit opt-in there.
Non-Flash defaults do not change. This is the offline Flash path; the existing
served-path restriction is unchanged. `G6_PF_STREAMS=2` and `G6_RESID_DUMP` are not supported with two
Flash lanes. `G6_PF_HASH=1`, prefill/decode logit dumps, and XTRACE work on both
lanes. XTRACE prints the second lane after a `# PXT second lane` marker.

## Default-on evidence

Both-rank full-logit/step/hash gates passed at 9K (MP8192) and 32K (MP16384).
At 32K, the one-rail off/on/off first-token intervals were
27.709 / 25.331 / 27.776 seconds. The stripe comparison showed a smaller gain
but had a concurrent reference workload; it is not a clean-host measurement.

The clean one-rail 300K run used MP8192, two lanes, capacity 300544, 32 generated
tokens, no MTP, and synchronized profiling. Both engines exited successfully.
Rank peaks were 104,574,377,984 and 104,619,806,720 bytes, below the 110 GB gate.
NV_ERR counts did not increase. Total cold boot-to-first-token was
289.345 / 289.068 seconds; the post-boot-stage intervals were
274.669 / 274.673 seconds. These times are PROFILED, not clean speed rows.

## Schedule and correctness

Two independent FlashPrefill allocations use the existing fabp/fabp2 regions
and step cells. All kernels execute on ONE compute stream. The fabric proxies
and NICs run independently of that stream. A lane's publish precedes a compute
block from the other lane; its consume follows that block. The final B MLP
consume waits until after A's next-layer attention publish, except at drain.

A always processes the earlier sub-batch. Each layer's KDA state and DSA cache
see A before B. No kernel row shape or reduction order changes. Model scratch
remains serialized. B retains its previous step until its delayed MLP consume
completes. The last short sub-batch retains exactly its serial row count.
Post-prefill hashes, logits and draft cache work run in chronological A/B order.
The main-model no-MTP path is the measured gate; MTP remains unmeasured here.

## Initial TP2 screen (2026-10-15)

Base: `567e9c7`. Same binary for control and candidate, full-clock configuration,
45 layers, MP8192, one rail, no MTP, graph decode, identical fixed token files.
Hash/dump/profiler switches are off for speed rows. One candidate screen between
two controls; not a multi-boot acceptance study.

| Metric (rank 0) | One lane A | Two lanes | One lane B |
|---|---:|---:|---:|
| 32K prompt start to first token, seconds | 27.709 | 25.331 | 27.776 |
| Process start to first token, seconds | 41.405 | 39.210 | 41.517 |
| Decode step p50, milliseconds | 27.062 | 27.169 | 27.078 |
| Peak guarded memory, decimal GB | 92.56 | 99.44 | 92.54 |

Prompt latency improves 2.412 seconds (8.69%) against the control mean. Control
spread is 0.067 seconds. Peak guarded memory means RSS plus the GPU-memory value
reported to the existing guard; it is not a standalone CUDA allocation count.
Rank 1's two-lane peak is 99.50 GB. This does not establish 300K headroom.

Gates:
- TP2 Flash build: exit 0, Rust loader linked.
- 8-layer, 300-token canary, MP128: both-rank full logits and steps/hashes byte-equal.
- 45-layer, 9216-token prompt, MP8192, 32 generated tokens: both-rank full
  prefill/decode logits and steps/residual hashes byte-equal (`cmp`, exit 0).
- All three 32K generated step files also byte-equal on both ranks.
- All completed gate/speed jobs: both ranks exit 0, no guard kill, no new memory errors.
- Non-TP2 builds, MTP, dual-rail composition, 300K and concurrent serving: not verified.

Tested executable SHA256:
`4036b74c1fd228df962ca96f5b4906e8d98655ed5c56d0ef97e58a311ee7ef51`

Operational note: the first speed attempt did not run inference. The peer guard
refused a disk-space floor after the gate wrote large logits; the other rank
waited for a peer. That attempt is excluded. The rerun checked both disks before
launch and used a shorter timeout. Verified duplicate dumps were removed only
after the byte comparisons and hashes were recorded.

## Composition

No communication API, slot layout, tail sequence, or comms source changed.
Dual-rail striping can compose with both prefill regions. Its combined gain and
remaining exposed peer wait must be measured before selecting the next change.

## Combined dual-rail screen and direct receive-wait trace

The combined build includes dual-rail commits `2d72369` and `d594d9a`. With
`G6_PF_XTRACE=1`, `PFCOMM step c wait_begin wait_end copy_end` records three
local-GPU timestamps for each staged receive. The two-lane marker separates
repeated step IDs. The old put-to-mHC-entry gap includes useful work from the
other lane and must NOT be interpreted as exposed network wait under overlap.
`engine/runtime/tests/pf_overlap_trace.py` validates coverage and timestamp order
and sums the explicit wait and copy spans. It does not subtract remote clocks.

The first combined screen had a co-resident reference job on rank 0. Absolute
times are not clean-box results. MP8192, stripe on throughout, lanes 1/2/1:

| Metric | One lane A | Two lanes | One lane B |
|---|---:|---:|---:|
| 32K prompt to first token, seconds | 27.510 | 26.516 | 27.203 |
| Rank 0 new driver memory warnings | 0 | 4 | 0 |

The screen delta is -0.841 seconds (-3.07% versus the control mean). Both ranks
exited 0, and all three generated step files were byte-equal. The driver warnings
have no verified process attribution; the candidate is not a clean-memory row.

One separate two-lane XTRACE run captured all 360 boundaries per rank:

| Local-clock span, seconds | Rank 0 | Rank 1 |
|---|---:|---:|
| Exposed staged peer wait | 0.005163 | 1.568009 |
| Receive copy | 0.452865 | 0.435219 |
| mHC entry through end | 3.167039 | 3.105003 |

Rank 1's maximum wait is 325.604 ms at the first attention boundary of lane B.
Rank 0 almost never waits. This does not establish 1.568 seconds of removable
wire time: producer/host skew can account for the asymmetry. A clean-box repeat
is required before deciding on chunked publication. The trace row added one
rank-0 driver warning; rank 1 added none.

MP16384 combined gate: a 32768-token prompt actually exercises both lanes;
9216 tokens would not. With stripe on and `G6_PF_HASH=1`, lanes 1 versus 2 passed
both-rank byte comparisons of complete prefill/decode logits and steps/hashes.
Neither gate row added driver memory warnings. Two-lane peak engine-only guard
memory was 112.784 GB on rank 0 and 113.091 GB on rank 1. Rank 0's accounting
including the co-resident GPU context reached 114.325 GB. Even the engine-only
values exceed the planned 110 GB acceptance ceiling. This is NOT a 300K pass.

Diagnostic build SHA256:
`8e0854f72dcea10c02d60c2868969c4cd60fc61830e06c970b9f0f3d41bf47d2`

CPU parser check: `cd engine/runtime/tests && python3 -m unittest test_pf_overlap_trace -v`
(four tests pass: separate lanes, missing coverage, duplicate keys, reversed timestamps).
