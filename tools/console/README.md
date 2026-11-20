# Kindling dash

Kindling dash was called the gr8 console. The code, paths, config file and commands
keep the old names (`tools/console`, `console.json`).

A small FastAPI + SvelteKit dashboard, separate from the engine so it stays up
while models restart. One compact dark-theme page: stats, three presets,
boxes, the head log, and the models table. No inference traffic is generated
by the dash.

## Run

From the repository root, using Python 3.10+ and Node 20+ with pnpm:

```sh
python3 -m venv tools/console/.venv
tools/console/.venv/bin/pip install -r tools/console/server/requirements.txt
(cd tools/console/web && pnpm install --frozen-lockfile && pnpm build)
GR8_CONSOLE_CONFIG=tools/console/console.json tools/console/.venv/bin/uvicorn tools.console.server.main:app --host localhost --port 8800
```

Open `http://localhost:8800`. Without a config it starts disconnected and
read-only; it does not discover machines or contact a default cluster.
Run **one worker**, with a persistent private state directory. Do not run with
`--reload` during a control operation. The process owns an advisory lock in its
state directory and refuses a second worker there.

For development, `pnpm dev` in `web` proxies `/api` to the local backend.
`GR8_CONSOLE_BACKEND` overrides the development proxy target.

### Deployment configuration

Copy `console.example.json` to the ignored `console.json`. Replace placeholders
with your deployment's configuration. Relative paths resolve from the server's
working directory. No credentials are accepted from the browser.

- `models`: read-only head URLs and supervisor TCP addresses. Per-model
  `control_port` changes the derived TCP port; explicit per-box `controls`
  entries can also set `port`. `served_id` must
  match `/v1/models` exactly (case-insensitive); built-in defaults are
  `deepseek-v4.1-flash`, `glm-5.3`, and `glm-5.3-flash`.
- `switch_file`: the existing `gr8 switch` configuration. Requires
  `tools/switch/switch.py` and the `./gr8 switch MODEL @FILE` command in this
  checkout. This dependency is not reimplemented here. URLs, membership and
  widths default to that config; explicit observation endpoints can override
  them for tunnels. Every model's `.env` must already exist.
- `allow_control`: false by default. Set true only when this dash owns the
  deployment and restarting is appropriate. SSH BatchMode and remote Python 3
  must work. The dash runs the existing switch phase for target stop and
  invokes the switch CLI for activation, including restart activation.
- `allowed_hosts`: browser-facing hostnames accepted by the HTTP server.
  Defaults to `localhost`; list the actual name when behind a proxy. Do not
  use a wildcard. This is DNS-rebinding protection, **not login/authentication**.
- `state_dir`: private operation status and exact dotenv backups. Keep it
  outside `web/build`. It is created mode 0700; backups are mode 0600.

The control port is **line-delimited JSON over TCP**, not HTTP. Supervisors
permit only cluster/local callers. Run the dash on a cluster member, or
provide appropriate SSH tunnels and explicit `controls` endpoints. An
unreachable/refused supervisor appears as unavailable, not as a healthy box.

The server is an administrative interface with the SSH authority of its user.
Keep it loopback-only and reach it through a private SSH tunnel, or put an
**authenticated** proxy in front of it. Do not expose it directly to an untrusted
network. Writes require a same-origin CSRF token and JSON; no permissive CORS
is enabled. These checks are not a substitute for authentication.

## Statistics

The backend samples `/health`, `/v1/models`, `/metrics`, and configured
supervisors' `status`/`logs`. It caches snapshots briefly and the browser polls
at most once every two seconds after the previous poll completes.

- Decode/prefill: aggregate token counter delta divided by elapsed wall time.
  These are **not** single-request decode or prefill benchmark speeds.
- TTFT: boot-lifetime sum/count. Prefix hits: hit tokens / queried tokens.
- Draft acceptance: accepted **tokens** / drafted **tokens**, not draft rounds.
- Requests finished: sums every exported finish reason, including errors/aborts.
- Free memory: supervisor `mem_gib`, system memory, not invented GPU memory.
- Missing fields, zero denominators, first samples and counter resets show `—`.
  On an unreachable head values are cleared. Browser disconnection marks its
  last snapshot stale and disables controls. A ready flag without the expected
  served model identity is not credited to that model.

## Presets

The page offers three presets and no other knob controls. A choice is reviewed
in a confirm dialog (previous setting, then the new one, plus each override
line the preset removes) and applied only by
**Restart now**. That writes the model’s `.env` on every target box, stops the
target, invokes `gr8 switch`, and waits for readiness plus the requested model
identity. Running chats end. The same compare-and-swap, backup, and rollback
path used for any dotenv edit applies. Unrelated lines stay as they are.
A DeepSeek or GLM-5.3-Flash preset rewrites `EXTRA_ENV` only to drop the
recipe words named below.

| Model | What a preset writes | Source of the one-line description |
|---|---|---|
| DeepSeek-V4.1-Flash | `TUNE=speed`, `quality`, or `context`. Removes explicit `SESS`, `MAX_MODEL_LEN`, `VARIANT`, and `EXACT`. Removes the recipe fast words from `EXTRA_ENV` (`GR8_DS_RELAXED_ACCEPT`, `GR8_DS_MOE_TOPK_DECODE`, `GR8_DS41_HEAD_MX`, `GR8_DS_ATTN_TC_MF`, `GR8_DS_ATTN_TC_TILE`, `GR8_DS_MHC_ROW_SHARD`, `GR8_DS_MHC_Y_MXFP8`, `GR8_DS_CACHE_MIB`, `GR8_DS_SESS_DRAFTS`). Keeps `GR8_DS_ENGRAM*`. | the recipe `knobs.sh` / “Pick what to prioritize” |
| GLM-5.3-Flash | the same three `TUNE` values and the same `SESS` / `MAX_MODEL_LEN` / `VARIANT` removals, and removes `GR8_CP` from `EXTRA_ENV` | the recipe `knobs.sh` / “Pick what to prioritize” |
| GLM-5.3 | Speed clears `TUNE`, `VARIANT`, `KV_FORMAT`, `SESS`, and `MAX_MODEL_LEN` (launcher home, 3.25 bpw). Quality sets `VARIANT=bit325`, the same 3.25 bpw size with the bits placed differently, and clears `KV_FORMAT`, `SESS`, and `MAX_MODEL_LEN`. Context sets `KV_FORMAT=int4` and clears `VARIANT`, `SESS`, and `MAX_MODEL_LEN`. | launcher options in [docs/GLM-5.3.md](../../docs/GLM-5.3.md) |

The current preset is read from the keys that preset writes. An explicit
`SESS` or `MAX_MODEL_LEN` on any of the three models shows
**custom**, and so does any other combination, including mixed boxes.
No preset removes `CAP`. `CAP` is the box's own limit on the KV pool, set to
fit that ring. A box with `CAP` set reads as its preset when the other keys
match. A preset that removed `CAP` let the pool grow past what the ring holds,
and the engine failed to boot.
On GLM-5.3 only, a preset that moves a box between the fp8 KV rows (Speed,
Quality) and the int4 KV row (Context) rescales an explicit `CAP` so the pool
keeps the same KV memory. The ratio is the `CAP` pair that
[docs/GLM-5.3.md](../../docs/GLM-5.3.md) quotes for TP=4 (fp8 408,832, int4
513,536, about 1.26x); the table in [docs/KV4.md](../../docs/KV4.md) gives a
larger ratio, so this one asks for less memory. The change moves in steps of 64 tokens times `TP`
(`TP` in the box's dotenv, else the number of boxes), counted from that box's
own `CAP`, and rounds down. A box with `CAP` 409,600 at TP4 gets 514,304 on
Context. Speed or Quality takes it back to 409,600: the dash picks the
largest whole-step fp8 `CAP` whose int4 image does not pass the current one.
The card, the confirm dialog (`CAP=409600`, then `CAP=514304`), and the step
list show the new value. Boxes with different `CAP` values each scale from
their own, and the dialog says so. Unset `CAP` stays unset. DeepSeek and
GLM-5.3-Flash presets do not set `KV_FORMAT` (the recipe picks it), so their
`CAP` stays as the dotenv has it, and an explicit `CAP` there overrides the
`TUNE` default.
On DeepSeek, `EXACT` or one of the fast words above also shows **custom**.
On GLM-5.3-Flash, `GR8_CP` in `EXTRA_ENV` shows **custom**. The confirm
dialog lists each removal (`KEY=old`, then `KEY removed`). A preset that
changes `VARIANT` (an unset `VARIANT` means home) first checks every box
for that variant’s verified weight marker, the `complete/` file written
after the rank’s weights are checked. A manifest alone does not count.
Unset `GR8_CACHE` and `GR8_HOME` are the Docker named volumes `gr8_cache`
and `gr8_home`; the check uses each volume’s mountpoint. If a mountpoint
cannot be resolved, a box cannot be checked, or the cache exists but the
SSH user cannot read it (a named volume under a root-only Docker directory),
the dialog says it cannot check that box. That is not a download, and the health wait stays at the
stock timeout. If a box is missing the marker, the dialog says this preset
downloads up to about N GB to each box. N is a rounded upper bound, because
expert halves already fetched for another variant are not fetched again.
When a fetch log has a measured rate, a second sentence gives the minutes.
The dialog also states the extended health wait, which can be hours when
no rate was measured. **Restart now** confirms a missing download. The
`gr8 switch` health wait becomes that extended wait, and the operation
log reports `downloading`.

While `gr8 switch` waits for health, on every switch or restart, the dash
reads the head supervisor log every 5 seconds. When the engine of this boot
exits (a line such as `rank 1 (...) failed: engine exited (2): ...`, or the
head's own engine, front end, or shim exit), the dash ends the wait at
once, stops the `gr8 switch` command, and fails the operation. The job error
and the progress steps carry the supervisor's reason. It does not wait for a
later retry: the supervisor stops the whole cluster on the first exit, and
the boxes may keep restarting the engine, so inspect them before you try
again. A failure line counts only for a boot whose `engine started (epoch E)` line
is newer than the last one in the log before the operation stopped anything,
so lines left by an earlier boot never stop a new one. The dash reads
that last epoch after the stop, because an old deployment that crash-loops can
start a boot before it is stopped. If no supervisor answered, a boot counts
when it began in the second the operation began or later. All models share one
control port, so until `gr8 switch` downs the old deployment, the log is its
log. The dash counts a failure only once the supervisor status shows a
`model_id` other than the old deployment's, and only for boots after the log
line `model set ... id <model_id>` of that deployment. If the old deployment
still answers with the same `model_id` (a restart whose stop left it up), no
failure ends the wait early and the health timeout applies.

Every preset removes the keys it lists. On GLM-5.3 the three presets are exclusive: Quality
removes `KV_FORMAT`, Context removes `VARIANT`, and Speed removes both.
Unset `TUNE` on DeepSeek or GLM-5.3-Flash is Speed, the recipe default.
On GLM-5.3, the watched keys unset is Speed, because that is the launcher
default. The dash does not guess other defaults and does not write
engine cache paths. Other `EXTRA_ENV` words, including `GR8_DS_ENGRAM*`,
stay as they are. Recipe `knobs.sh` applies `SESS`,
`MAX_MODEL_LEN`, and `VARIANT` (and `CAP`, which a preset never removes) only when they are unset, and it appends
the user's `EXTRA_ENV` after its own words.
No performance claim is made here.

The backend pins the switch configuration at startup and refuses operations
if that file changes. Restart the dash to review new deployment targets.
It compares all dotenv revisions with the reviewed snapshot before writing
and checks each again atomically on the remote host. A raw knob map in the
operation request is rejected.

## Failure and recovery

Every confirmation includes the active model observed during review. The server
rechecks a fresh snapshot and returns 409 if it changed; a non-active target’s
restart dialog explicitly names the model it will stop.

One operation at a time per dash; duplicate concurrent submissions get 409.
Use one controller for a deployment: **external CLI invocations or another
dash using a different state directory are not distributed-locked**.

Before editing, exact original dotenv contents are saved in the private state
directory under the operation ID. A partial write failure attempts a
compare-and-swap restore of the affected boxes and does **not** restart. A
concurrent external edit is never overwritten by rollback. Inspect any rollback
error before proceeding.

After stop/start has begun, failure remains visible with streamed command output;
the dash does **not** pretend success or automatically switch back. Saved
overrides stay on disk. **Restore previous settings** restores the last preset
operation’s exact original dotenv files, checking every current revision against
what that operation wrote. An outside edit makes it refuse; a partial restore
can be retried without overwriting other edits. Restore does **not** restart:
inspect every target, then restart or switch back explicitly. Older retained
backups can be restored manually, preserving each file’s owner/mode. Do not start another operation until the cluster state is
understood. An operation interrupted by a dash restart is marked interrupted.
Backups are retained; remove them deliberately when no longer needed.

## Off-ring tests

```sh
python3 -m unittest discover -s tools/console/tests -v
(cd tools/console/web && pnpm build)
# Optional browser tests: pip install playwright; playwright install chromium
python3 tools/console/tests/browser.py --output console-screenshots
# Interactive synthetic demo (never contacts a ring):
python3 -m tools.console.tests.serve_fixture --port 8899
```

Fixtures run an actual fake HTTP head and TCP supervisor and a fake control
adapter. They are **synthetic protocol-shaped replies**, not recorded live
measurements; their shapes follow gr8's own API/metrics/supervisor source.
The banner and screenshots say so. Tests cover missing/reset counters, scope
validation, injection, stale revisions, write rollback, rejected raw knob maps,
concurrent operations, failed boot, CSRF, and browser confirm/cancel,
model switch/stop, disconnected state, and desktop/mobile layout.

Live end-to-end proof is separate: only in a booked window, live stats, one
confirmed preset restart, one UI model switch, and exact stock restore. Off-ring
passing tests do not establish live fleet success.
