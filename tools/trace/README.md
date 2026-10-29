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

The trace reads the host clock only: no device syncs and no change to what the engine does.
