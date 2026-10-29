# S2 traces

`GR8_S2_TRACE=PATH` makes every rank write a Chrome trace of its S2 loop to `PATH.r<rank>.json`:

- the `steps` lane: a span per prefill quantum, prompt tail and decode step, from its launch to the next boundary,
  with the requests in it, the rows and the boundary;
- the `control` lane (the head): each request's arrival (prompt and `max_tokens`), its wait for a slot or for
  capacity, its admission (rows reserved, rows reused) and each notice;
- instants for every commit, each request's first token and its finish.

Timestamps are wall-clock microseconds, so the ranks and a client line up. Add it to `EXTRA_ENV` with the engine
variables the config already passes (`--extra-env` on the command line replaces the config file's):

    EXTRA_ENV="... GR8_S2_TRACE=/tmp/s2"

Then join the ranks, and optionally a client's request times (one JSON object per request with `send`, `first` and
`end` in seconds, plus `agent` and `turn`), and open the result in ui.perfetto.dev or chrome://tracing:

    tools/trace/merge.py /tmp/s2.r*.json --requests requests.jsonl -o trace.json

Each machine stamps with its own wall clock, so two machines differ by whatever NTP left. When that is not small
next to the steps being lined up, measure each rank's machine from the client's machine before and after the run and
pass the probes to merge.py:

    tools/trace/clock_probe.py HOST1 r1-before.json      # ... the run ...
    tools/trace/clock_probe.py HOST1 r1-after.json
    tools/trace/merge.py /tmp/s2.r*.json --requests requests.jsonl --clock 1=r1-before.json,r1-after.json -o trace.json

merge.py takes each probe's lowest-round-trip sample (offset good to half that round trip), subtracts the mean of the
two offsets from that rank's timestamps, and refuses a probe with a round trip or a before/after drift over 50 ms
(`--max-rtt-ms`, `--max-drift-ms`). Without `--clock` the timestamps are joined as written.

The trace reads the host clock only: no device syncs and no change to what the engine does.
