# Baselines

## tp6.json — `main` at afca111, TP6, GPU clock cap 2200 MHz

- Build: `main` at afca111, `scripts/build.sh` (K1H2 flags, `G6_NR=6`), the same binary on all six nodes.
- Launch: `scripts/launch_tp6.sh` defaults (`CAP=614400`, `MTP=2`, `NOHASH=1`, K1H2 engine env), rank health probe admitted
  on every boot. The rank ssh line was wrapped in `{ ...; }` so the launcher returns; nothing else changed.
- **GPU clock cap 2200 MHz on all six nodes** (`nvidia-smi -lgc 0,2200`), recorded in every run record. Compare only
  against numbers taken at the same cap.
- KL reference: `../refs/tp6.json`, `gate.py kl-ref` on boot b1 (16 sequences x 256 tokens, top 20).
- Three boots (b1, b2, b3), each stopped completely and started again; `gate.py run --repeat 2 --skip mixed4 --kl-ref`.
- KL: 0.0 on every run of every boot (4096 positions, top-1 agreement 1.0): gr8 at a fixed commit gives the same
  top-20 logprobs boot to boot on TP6. The KL threshold is therefore the floor, 1e-6 (2 x worst boot = 0).
- Not in the baseline: `work.mixed4.*` (needs `-DDEC_S2`), so `check` reports PARTIAL.
- The `workloads` numbers are reasoning text: `effort=nothink` cannot turn thinking off with the GLM 5.3 template.
  Since 2026-10-12 the front end refuses `enable_thinking` false, and `nothink` leaves out every thinking control. The
  prompt is the same as before (the template's default effort, Max).

## tp4.json — `main` at 34d0d05, TP4, GPU clock cap 2200 MHz

- Build: `main` at 34d0d05, `G6_NR=4 BUNDLE_LIBS=1 scripts/build.sh`, the same binary on all four nodes.
- Launch: `scripts/launch_tp4.sh` defaults (`CAP=393216`, `PF_OWN=4`, DFlash2 drafter with adaptive draft count,
  `NOHASH=1`). Drafter packs from `tools/convert/dflash2.py --nr 4` (0.86 GB per rank).
- `launch_tp4.sh` does not run the rank health probe (`launch_tp6.sh` does), so `scripts/rank_health.sh` was run before
  each boot; all four ranks were admitted every time.
- **GPU clock cap 2200 MHz on all four nodes**, recorded in every run record.
- KL reference: `../refs/tp4.json`, `gate.py kl-ref` on boot b1 (16 sequences x 256 tokens, top 20).
- Three boots, each `gate.py run --repeat 2 --skip mixed4 --needle-sizes 87000 --kl-ref`.
- KL: 0.0 on every run of every boot (4096 positions, top-1 1.0). Threshold: the floor, 1e-6.
- Not in the baseline: `needle.437k.pass` (437K tokens is over the TP4 pool of 393,216; the request is refused) and
  `work.mixed4.*`. `check` reports PARTIAL.
- During these boots the rank 0 node was also downloading a large file over its internet link (not the fabric). The
  rows with the widest boot spread, `pbench.prefill_8192` (7.3%) and `long.18k.prefill_tok_s` / `ttft_s` (5.4% / 4.9%),
  may include that load; the 32K and 87K prefill rows (1.5-1.9%) do not show it.
