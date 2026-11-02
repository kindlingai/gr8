# bench

`bench.py` measures a serving cluster's speed and correctness the same way on every build, and gates a build against
an earlier report. It talks to the front end over HTTP and to the boxes over ssh, and needs only Python 3.

```bash
./gr8 glm53 bench                      # the default level, about 20 minutes; PEERS from ./gr8's options are the boxes
./gr8 glm53 bench --quick              # prefill, code and prose decode, concurrent streams
./gr8 glm53 bench --full               # everything, with GSM8K and HumanEval
python3 bench/bench.py --url http://HEAD:8100 --hosts 10.0.0.1,10.0.0.2 --cap 10.0.0.2=2200 --json main.json
python3 bench/bench.py --baseline main.json                  # this build against main's report: gates the speed
python3 bench/bench.py compare main.json pr.json             # two reports, after the fact
```

The report prints as ASCII tables. The JSON report holds everything: each run, the drafter counters, each box's
clocks per stage, and every single-stream request in send order. It goes to `--json`, or to
`/tmp/gr8bench-MODEL-LABEL-TIME.json`.

## Levels

| stage | `--quick` | default | `--full` |
|---|---|---|---|
| smoke: `smoketest/run.sh` (or `ds41.sh`), first, so a broken boot stops early | | yes | yes |
| T=0 replies: a hash of 7 fixed replies, to compare builds byte for byte | | yes | yes |
| count to 200, greedy; any wrong line fails | | 3 | 5 |
| prompt cache report: `cached_tokens` on a repeated prompt | | yes | yes |
| KL against a reference (`--kl-ref`) | | with `--kl-ref` | with `--kl-ref` |
| needle at 15% depth | | 128K | 32K, 128K, 256K, 480K |
| decode, one stream, 512 tokens: code and prose | 6 runs | 6 runs | 16 runs |
| decode: structured and counting | | 6 runs | 16 runs |
| concurrent streams of mixed prompts, 512 tokens each | 1, 2, 4, 8 | 1, 2, 4, 8 | 1, 2, 4, 8, 16 |
| cold prefill 8K, 32K, 128K, then the same prompt cached | once | once | twice |
| tool call at 42K context, 40 runs, 4 at once | | | yes |
| agent tools: 5 tasks on a sandboxed file tree, whole and streamed | | | yes |
| long reply: 2 replies of 16K tokens at T=1, decode against prefill logprobs | | | yes |
| GSM8K (first 250) and HumanEval (164), greedy | | | yes |

`--stages`, `--runs`, `--streams` and `--prefill-sizes` change a level's plan. Sizes over the server's
`max_model_len` are skipped and say so.

Requests use `reasoning_effort` low (`--effort`) and temperature 0. A decode run's prompt carries a run tag fixed by
`--series`, the workload and the run number, so run 5 of `code` sends the same bytes on every build and never hits the
prefix cache. Two reports pair only within one series. The decode prompts are the
[glm-5.3-flash-gx10](https://github.com/kindlingai/glm-5.3-flash-gx10) recipe's (`gate/prompts.py`), and so are the
statistics, the tool-call, agent and long-reply checks, and the GSM8K and HumanEval runs.

## Heat

A GB10 that has run for a few minutes is hotter and slower than a cool one: under a full bench, uncapped boxes drop
from about 2,450 MHz to 2,050-2,370 MHz on average at 80-84 C, and a run's numbers depend on what ran before it. So
the timed stages (decode, concurrency, prefill) run first, and each starts from the same state:

1. the server has no request in flight, and every box is locked at `--cool-mhz` (1,800);
2. every box's GPU temperature is sampled each second; the stage waits until every GPU is idle and every box is
   either at or under `--cool-temp` (55 C) or no longer cooling: its trend over the last 60 s is above -1 C per minute
   (`--settle-max`, 300 s, is the longest wait);
3. the clocks are unlocked; a box named with `--cap HOST=MHZ` is locked at its cap instead;
4. the stage runs with no clock limit, and its clocks are kept: each box's start and peak temperature, mean and
   lowest SM clock, the share of time under 2,000 MHz, power, and any throttle reason.

With the limiter stopped, uncapped GB10s reach 85-89 C in these stages. A box that reaches `--guard-temp` (90 C) is
locked at `--cool-mhz` for the rest of the stage, and the checks carry a WARN naming the stage and the box: its
numbers ran partly capped.

The other stages (smoke, correctness, the agentic checks, GSM8K and HumanEval) measure no time, so they run locked at
`--cool-mhz` with no wait before them.

A service that sets the clocks itself would undo all this: `sparkos-thermal` moves a cap every 15 s. Each service
named by `--thermal-service` (default `sparkos-thermal`) is stopped where it runs and started again at the end, and a
systemd timer on each box starts it again after 4 hours in case the bench dies first. A box without the service is
left as it is. `--hosts` names the boxes (ssh destinations; `local` is this box). Locking clocks and stopping the
service need passwordless `sudo` on every box; without it the clocks are left alone and the report says so. On exit,
by Ctrl-C or SIGTERM too, every box is unlocked, the service starts again, and the report covers what ran.

## Gates

`RESULT: PASS` needs every check to pass:

- smoke passes every case; every count is clean; no reply holds Hangul or U+FFFD (how corruption has shown on these
  models); the repeated prompt reports cached tokens; every needle in the context is found;
- no decode run or stream fails (an error, an empty reply, a wrong count);
- with `--kl-max`, KL against the reference is at or under it;
- the agent tools all pass and the long reply does not drift (`--full`);
- against `--baseline`: each decode workload, paired run by run, is not slower by more than 3% with a Holm-corrected
  Wilcoxon p under 0.05, and each prefill size is not slower by more than 3%.

Shown and not gated: the T=0 hash (equal or not to the baseline's), concurrency against the baseline (8 streams
spread 6-15% run to run), the tool call at 42K (gr8's batch numerics differ from one stream's), GSM8K and HumanEval.

The decode table splits a ratio into tokens per step, which depends on how easy the answers were to draft, and ms per
step, which depends on the build, so a faster build shows apart from easier answers.

## KL reference

```bash
python3 bench/bench.py kl-ref --out bench/refs/glm53-tp4.json --commit SHA --topology tp4   # on the reference build
python3 bench/bench.py --kl-ref bench/refs/glm53-tp4.json --kl-max 1e-6                    # a later build
```

The reference holds 16 greedy continuations of 256 tokens with the top 20 logprobs at each position. A build is
re-scored on the same token ids, and the report gives KL(reference, build) per position (mean, p95, max) and top-1
agreement. A reference is for one model: a run against another model skips it.

## Other tools here

- `route_bytes.py`: expert bytes per decode step from a `GR8_ROUTE_LOG` boot; `--drive` takes a bench report of the
  same boot, whose `requests` list names each request in send order (run `--stages decode`: concurrent streams
  have no order).
- `score_rows.py`: teacher-forced scoring of token rows, the student side of a KL against a teacher.
- `tests/`: `python3 -m unittest discover -s bench/tests`, against `tests/fake_server.py` (no GPU, no model).
