# TP5 serve: T5X (GLM 5.3 full, five DGX Sparks)

T5X is the TP5 serve config: [recipes/tp5-t5x/T5X.env](../recipes/tp5-t5x/T5X.env), on this branch's engine. It is the TP4 FSM win set,
all on from the first TP5 boot, plus five knobs measured at TP5 (K1_TLBW, PF_GEMMF32, ATTN_CTX3, PF_ISGRT_RG, PF_ATTN_ROWS). Output is byte-identical to the
first checked TP5 build (T5S2).

## Speeds (2026-10-19, single run per row unless noted, same-env rerun noise about +-2%)
| | 2000 MHz cap (serve, 2-run mean) | full clock (1 run) |
|---|---:|---:|
| prefill 8K tok/s | 1,572 | 1,731 |
| prefill 32K tok/s | 1,569 | 1,652 |
| prose decode, 1 chat tok/s | 39.8 | 41.9 |
| code decode, 4 chats (total) tok/s | 77.4 | 80.1 |
| prose decode, 4 chats (total) tok/s | 64.6 | 66.8 |
| first token, 157K prompt | 115.8 s | 107.7 s |
| decode at 157K tok/s | 24.5 | 28.8 |

Context and load (2000 MHz): KV pool 617,536 tokens; 4 chats of 139K each stay resident together; one 90K chat
decodes at 25.8 tok/s, four 90K chats at 44.1 tok/s total; 15 minutes of mixed agent load: 107 requests, 0 errors,
at least 2.6 GB free memory on every box. Needle tests at 128K and 437K pass (on T5S2, same numerics).

## Boot and stop
On all five boxes: free page cache and compact memory, cap the clock (`nvidia-smi -lgc 0,2000`), then start the
rank detached: `cd $G6_HOME && ./gr8 glm53 up @<cluster.env> > out/window-up.log 2>&1 < /dev/null` (for example
under `systemd-run --collect -p RemainAfterExit=yes`). The ranks wait for each other. Ready when
`curl -s 127.0.0.1:8000/health` on the head returns 200. Stop: `./gr8 glm53 down @<cluster.env>` on the head
stops all ranks.

Build: `G6_WIDTHS="4 5" F12_WIDTHS="4 5"` (TP5 widths), Rust toolchain 1.94.1.

## Measured at TP5 and left off
- `L2PF`: 0.4 GB less free memory, no mixed-load gain.
- `PF_OWNPERM=1`: output changes (KL 0.0031 vs T5S2), prefill 8K -6.7%.
- `QA_SPLIT=2`: prefill 32K -5%.
- `SHARE_LIMIT=8192`: the 4th 128K chat is evicted, 90K solo decode -4%.

## Option: 4-bit KV
`KV_FORMAT=int4` (needs `bin/g6m4-int4` from the same build) with a larger `CAP` boots more context (731,776 tested: -2% prefill, -4.6% code with 4 chats).
Its output drifts more than our fixed gate allows (KL 0.0169 vs 0.01 limit, top-1 96.4% vs 97%), so it is a
boot option only, not the default.
