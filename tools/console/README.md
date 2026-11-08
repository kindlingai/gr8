# gr8 console

A small FastAPI + SvelteKit console, separate from the engine so it stays up
while models restart. One compact Pulsar Nightfall page: stats, three presets,
boxes, the head log, and the models table. No inference traffic is generated
by the console.

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
- `allow_control`: false by default. Set true only when this console owns the
  deployment and restarting is appropriate. SSH BatchMode and remote Python 3
  must work. The console runs the existing switch phase for target stop and
  invokes the switch CLI for activation, including restart activation.
- `allowed_hosts`: browser-facing hostnames accepted by the HTTP server.
  Defaults to `localhost`; list the actual name when behind a proxy. Do not
  use a wildcard. This is DNS-rebinding protection, **not login/authentication**.
- `state_dir`: private operation status and exact dotenv backups. Keep it
  outside `web/build`. It is created mode 0700; backups are mode 0600.

The control port is **line-delimited JSON over TCP**, not HTTP. Supervisors
permit only cluster/local callers. Run the console on a cluster member, or
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
| DeepSeek-V4.1-Flash | `TUNE=speed`, `quality`, or `context`. Removes explicit `SESS`, `CAP`, `MAX_MODEL_LEN`, `VARIANT`, and `EXACT`. Removes the recipe fast words from `EXTRA_ENV` (`GR8_DS_RELAXED_ACCEPT`, `GR8_DS_MOE_TOPK_DECODE`, `GR8_DS41_HEAD_MX`, `GR8_DS_ATTN_TC_MF`, `GR8_DS_ATTN_TC_TILE`, `GR8_DS_MHC_ROW_SHARD`, `GR8_DS_MHC_Y_MXFP8`, `GR8_DS_CACHE_MIB`, `GR8_DS_SESS_DRAFTS`). Keeps `GR8_DS_ENGRAM*`. | the recipe `knobs.sh` / “Pick what to prioritize” |
| GLM-5.3-Flash | the same three `TUNE` values and the same `SESS` / `CAP` / `MAX_MODEL_LEN` / `VARIANT` removals, and removes `GR8_CP` from `EXTRA_ENV` | the recipe `knobs.sh` / “Pick what to prioritize” |
| GLM-5.3 | Speed clears `TUNE`, `VARIANT`, `KV_FORMAT`, `SESS`, `CAP`, and `MAX_MODEL_LEN` (launcher home, 3.25 bpw). Quality sets `VARIANT=bit325`, the same 3.25 bpw size with the bits placed differently, and clears `KV_FORMAT`, `SESS`, `CAP`, and `MAX_MODEL_LEN`. Context sets `KV_FORMAT=int4` and clears `VARIANT`, `SESS`, `CAP`, and `MAX_MODEL_LEN`. | launcher options in [docs/GLM-5.3.md](../../docs/GLM-5.3.md) |

The current preset is read from the keys that preset writes. An explicit
`SESS`, `CAP`, or `MAX_MODEL_LEN` on any of the three models shows
**custom**, and so does any other combination, including mixed boxes.
On DeepSeek, `EXACT` or one of the fast words above also shows **custom**.
On GLM-5.3-Flash, `GR8_CP` in `EXTRA_ENV` shows **custom**. The confirm
dialog lists each removal (`KEY=old`, then `KEY removed`). Every preset
removes those keys. On GLM-5.3 the three presets are exclusive: Quality
removes `KV_FORMAT`, Context removes `VARIANT`, and Speed removes both.
Unset `TUNE` on DeepSeek or GLM-5.3-Flash is Speed, the recipe default.
On GLM-5.3, the watched keys unset is Speed, because that is the launcher
default. The console does not guess other defaults and does not write
engine cache paths. Other `EXTRA_ENV` words, including `GR8_DS_ENGRAM*`,
stay as they are. Recipe `knobs.sh` applies `SESS`, `CAP`,
`MAX_MODEL_LEN`, and `VARIANT` only when they are unset, and it appends
the user's `EXTRA_ENV` after its own words.
No performance claim is made here.

The backend pins the switch configuration at startup and refuses operations
if that file changes. Restart the console to review new deployment targets.
It compares all dotenv revisions with the reviewed snapshot before writing
and checks each again atomically on the remote host. A raw knob map in the
operation request is rejected.

## Failure and recovery

Every confirmation includes the active model observed during review. The server
rechecks a fresh snapshot and returns 409 if it changed; a non-active target’s
restart dialog explicitly names the model it will stop.

One operation at a time per console; duplicate concurrent submissions get 409.
Use one controller for a deployment: **external CLI invocations or another
console using a different state directory are not distributed-locked**.

Before editing, exact original dotenv contents are saved in the private state
directory under the operation ID. A partial write failure attempts a
compare-and-swap restore of the affected boxes and does **not** restart. A
concurrent external edit is never overwritten by rollback. Inspect any rollback
error before proceeding.

After stop/start has begun, failure remains visible with streamed command output;
the console does **not** pretend success or automatically switch back. Saved
overrides stay on disk. **Restore previous settings** restores the last preset
operation’s exact original dotenv files, checking every current revision against
what that operation wrote. An outside edit makes it refuse; a partial restore
can be retried without overwriting other edits. Restore does **not** restart:
inspect every target, then restart or switch back explicitly. Older retained
backups can be restored manually, preserving each file’s owner/mode. Do not start another operation until the cluster state is
understood. An operation interrupted by a console restart is marked interrupted.
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
