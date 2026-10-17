# HANDOFF — PR #71 validation: finishing the count-200 evidence (strict comparison)

Status (2026-10-17): DONE. The strict re-run was executed — 8/8 strict PASS, raw replies captured —
and the results are recorded in `PR71-VALIDATION-EVIDENCE.md`. Fleet restored to DS serving.
This document is retained as the record of the recipe and findings.

## Where the run stopped (2026-10-16 evening)

- Fleet: DS V4.1-Flash ring serving on l/r/m/z (r6n baseline via `~/spark-ds41f/config/cluster.json`;
  unit `dsv41-cluster-ring4` fixed to point there — it previously pinned the retired r6h overlay).
  gr8 lab fully down (zero `gr8` containers on all 4 nodes).
- Evidence file: `~/gr8/PR71-VALIDATION-EVIDENCE.md` (complete for Phases 0-3.2, 4.1, 5; raw logs in
  `~/gr8/evidence/` and lspark `~/gr8-lab/pr71-*`).
- The maintainer's review on #71 (the one open item):
  > The new count-to-200 gate has a false pass: scripts/count200_probe.sh strips non-digits and keeps
  > only the first 200 numbers, so a reply with 1 through 201, or 1 through 200 followed by prose,
  > reports PASS. Please compare the raw reply with exactly 200 digit-only lines and post the
  > eight-run gr8 result before using this as the thinking-off gate.

## Why the current 8/8 is not proof

`scripts/count200_probe.sh` (added by the PR, commit 7618dc5, which amended the original e0d7f68) compares:

```bash
got=$(printf '%s' "$out" | jq -r '.choices[0].message.content // ""' \
      | sed -e 's/[^0-9]/ /g' -e 's/  */\n/g' | sed '/^$/d' | head -200)
if [ "$got" = "$want" ]; then ...
```

`want=$(seq 1 200)`. This normalizes the reply by stripping ALL non-digits and truncating to the
first 200 numbers — so "1..201", "1..200 plus prose", or "1..200 with stray digits elsewhere" all
PASS. The maintainer wants the RAW reply compared with exactly 200 digit-only lines.

## What to obtain from the re-run (the proof)

For each of the 8 runs, all of the following must hold against the RAW `content` string:

1. **Byte-exact match**: `content` == the output of `seq 1 200` (200 lines, "1\n2\n...\n200\n",
   trailing newline handling: require the string to equal `$(seq 1 200)` + optional final newline —
   define it once and use the same rule for every run).
2. **Line count == 200**: `content` splits into exactly 200 lines (no more, no fewer).
3. **Every line is digits only**: `grep -qvE '^[0-9]+$'` over the split lines must find nothing —
   no prose, no bullets, no spaces, no "thinking" text.
4. **Order + values**: line k == k for k = 1..200 (implied by 1, but assert it for the log).
5. **Report all 8 runs either way** (the runbook's rule: numbers reported even if not 8/8).

Suggested output shape for the PR comment (make it mechanical to verify):

```
run 1: PASS  (lines=200, digits-only=200/200, byte-exact=yes)
...
8/8 clean (strict)
```

Plus: save the FULL raw `content` of all 8 runs (e.g. `evidence/count200-run<N>-raw.txt`) and attach
or inline one complete raw reply so the maintainer can eyeball it.

## Fixing the probe (do this in the PR branch, not just locally)

The probe is part of the PR (`scripts/count200_probe.sh`, added in 7618dc5, originally +32 lines in e0d7f68). Two options:

- **Preferred**: change the comparison to strict. E.g.:

```bash
want=$(seq 1 200)
content=$(printf '%s' "$out" | jq -r '.choices[0].message.content // ""')
got=$(printf '%s\n' "$content" | sed '${/^$/d}')          # strip one trailing newline only
if [ "$got" = "$want" ] && [ "$(printf '%s\n' "$content" | wc -l)" = 200 ] \
   && ! printf '%s\n' "$content" | grep -qvE '^[0-9]+$'; then
```

  (beware `$(...)` eating the final newline — the sed above removes exactly one; test both shapes.)
- **Alternative**: keep the loose matcher for convenience but make PASS require the strict check, and
  keep the loose normalized diff only for the FAIL diagnostic display.

Amend the commit (per user convention: amend + force-push after review) and re-run.

## How to run (docker lab path — see also skill `gr8-pr-validation`)

NOTE (2026-10-17): the re-run was executed per this recipe. Two lab-specific findings are now
encoded in the skill: (a) the flag wrapper `~/gr8-lab/fix-front/g6front-rs` reverts to the plain
`exec ... "$@"` form when re-staged — rewrite it with `--allow-thinking-off` on all 4 nodes and
verify the live cmdline before counting runs; (b) DS restore can hit transient exit-137 rollbacks
on worker nodes (startup race, not OOM) — `docker start` all 4 nodes, wait for /health, then
`reset-failed` + `start` the unit. The strict 8/8 result is recorded in
`PR71-VALIDATION-EVIDENCE.md` (maintainer round section) with raw replies in
`evidence/count200-raw-8runs.tar`.

1. Sync lspark repo: `cd ~/gr8-lab/gr8 && git fetch fork && git checkout -B missing-api-calls fork/missing-api-calls`
   (DONE: lspark now at `7618dc5`; the only change vs e0d7f68 is the probe script, so the front
   binary md5 fe9b5c00… was reused, verified in-container).
2. Build front end only is NOT enough this time unless the amendment touched only the probe script —
   if only scripts/count200_probe.sh changed, the front binary is unchanged: reuse
   `/tmp/g6front-branch` (md5 fe9b5c00f5a1e82bcc6c40d40daec7f9, already staged on all 4 nodes at
   `~/gr8-lab/fix-front/g6front-rs.real`). Verify md5 before trusting.
3. Bring the gr8 ring up (DS down first — see the runbook + skill):
   - stop DS: `~/spark-ds41f/bin/spark --cluster-config config/cluster.json cluster stop --apply`
   - memory tune x4: `sudo sysctl -w vm.compaction_proactiveness=0`; zspark `sudo swapoff /swap-64k.img`
   - wrapper must carry the flag: `~/gr8-lab/fix-front/g6front-rs` = `exec /opt/gr8/fix-front/g6front-rs.real --allow-thinking-off "$@"`
   - up: `cd ~/gr8-lab/gr8/docker && docker compose -f compose.yaml -f compose.override.yaml up -d` on
     lspark, then z/m/r (BOTH -f files every time; ~80 s to healthy)
   - verify: head `/health` → `{"ready":true,...}`; `docker exec gr8 md5sum /opt/gr8/fix-front/g6front-rs.real`
4. Run the strict probe: `bash scripts/count200_probe.sh http://127.0.0.1:8100 8 glm-5.3`
   (8 runs, a few seconds each; tee to `/tmp/pr71-count200-strict.log`; also save the 8 raw contents).
5. Restore the fleet: `docker compose -f compose.yaml -f compose.override.yaml down` on z/m/r then lspark;
   `sysctl -w vm.compaction_proactiveness=20` x4; zspark `sudo swapon /swap-64k.img`;
   `sudo systemctl start dsv41-cluster-ring4`; verify `/health` 200 + one real completion.

## Gotchas learned this run (already encoded in the gr8-pr-validation skill)

- serve_mock.sh leaves stale front processes; kill `g6front-rs|mock_engine` between suite runs.
- The lab is compose-based: always pass `-f compose.yaml -f compose.override.yaml` (the override does
  not auto-merge). Verify the running binary by md5 INSIDE the container, and confirm the wrapper
  actually carries `--allow-thinking-off` (a stale wrapper from a previous session bit us once).
- `~/gr8-lab/home/bin` must not exist as a directory (crash-loop) or a live symlink (cp goes into the
  dying container layer) before compose up.
- gate.py default `--needle-sizes 87000,437000` exceeds the lab's 310,514 cap → use `--needle-sizes 87000`.
- The rank-probe RDMA ring test cannot pass on f0→next/f1→prev cabling; run `--allow-unhealthy` and
  document (per-node GPU numbers were in-band: 2177-2190 MHz, HBM ~235 GB/s, H2D ~54 GB/s).
- Restore DS last; never leave the fleet on gr8.

## After the re-run

Update `PR71-VALIDATION-EVIDENCE.md`:
- Replace the Phase 4.2 section with the strict 8-run result + raw-reply evidence;
- Add a "Maintainer review round" note: the loose-matcher false-pass issue, the fix (commit sha), and
  the strict re-run numbers;
- Note in the PR comment that the previous 8/8 was measured with the loose matcher and should be
  disregarded in favor of the strict result.
