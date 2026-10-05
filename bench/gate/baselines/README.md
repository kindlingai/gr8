# Baselines

## tp6.json — `main` at afca111, TP6, GPU clock cap 2200 MHz

- Build: `main` at afca111, `scripts/build.sh` (K1H2 flags, `G6_NR=6`), the same binary on all six nodes.
- Launch: `scripts/launch_tp6.sh` defaults (`CAP=614400`, `MTP=2`, `NOHASH=1`, K1H2 engine env), rank health probe admitted
  on every boot. The rank ssh line was wrapped in `{ ...; }` so the launcher returns (the fix in #40); nothing else changed.
- **GPU clock cap 2200 MHz on all six nodes** (`nvidia-smi -lgc 0,2200`), recorded in every run record. Compare only
  against numbers taken at the same cap.
- KL reference: `../refs/tp6.json`, `gate.py kl-ref` on boot b1 (16 sequences x 256 tokens, top 20).
- Three boots (b1, b2, b3), each stopped completely and started again; `gate.py run --repeat 2 --skip mixed4 --kl-ref`.
- KL: 0.0 on every run of every boot (4096 positions, top-1 agreement 1.0): g6 at a fixed commit gives the same
  top-20 logprobs boot to boot on TP6. The KL threshold is therefore the floor, 1e-6 (2 x worst boot = 0).
- Not in the baseline: `work.mixed4.*` (needs `-DDEC_S2`, #26/#28), so `check` reports PARTIAL.
- The `workloads` numbers are reasoning text: `effort=nothink` cannot turn thinking off with the GLM 5.3 template (#1).
