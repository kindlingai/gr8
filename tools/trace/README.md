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

# ConnectX counters

`tools/trace/nicprof.py` samples every RDMA port of each rank's machine over ssh during a run and writes
`PATH.nic.r<rank>.json`: per port, the send and receive rate in Gb/s, and the increase per interval of the counters that
show transport trouble (retransmits, sequence errors, CNPs, ECN marks, transmit waits). The samples carry each machine's
wall clock, so merge.py joins them with that rank's S2 trace and applies the same `--clock` offsets:

    tools/trace/nicprof.py --out /tmp/nic 0=HOST0 1=HOST1 2=HOST2 3=HOST3 --seconds 120 &
    # ... the run ...
    tools/trace/merge.py /tmp/s2.r*.json /tmp/nic.nic.r*.json -o trace.json

It also prints a summary per rank and port: mean and peak rates over the intervals the port moved data, and the
trouble counters' totals.

Once a second it also reads each machine's GPU through nvidia-smi (graphics clock, temperature, active throttle
reasons) into a `gpu` counter track, and the summary ends with the clock line per rank: mean, min and max MHz, the
temperature range and the throttle reasons seen. The boxes' clocks are not fixed (a thermal governor moves the cap
between 1500 and 2400 MHz by temperature), so a measurement is only comparable with its clock line next to it.

`GR8_DS_SPAN=1` adds the DS41 GPU side: the time of each plan unit (attention, consume, MoE, head, ...) summed over the
layers, per eager step and, for each captured step graph, as the mean over 16 sampled replays (every 16th). A captured
step cannot be timed with events (an event recorded as a graph node cannot be read back), so the capture stamps
`%globaltimer` around each unit with a one-thread kernel node; the stamps make the profiled step a little slower.
`GR8_DS_SPAN=2` adds one line per unit kind with the time of each layer, in layer order. A served prefill chunk runs
its ops outside the units, so it is one pass per chunk and lane with a mark after every op (and inside the MoE ops,
after each kernel group): an `ops:` line with the marks summed over the layers, and with `GR8_DS_SPAN=2` a `layer L
ops:` line per layer. `GR8_DS_SPAN=3` marks the ops of the decode units the same way. The lines go to the engine's
stderr (the launcher's `g6m4.r<rank>.err`; `./gr8 ds41 logs` shows only its tail):

    # span M=6 per step over 16 sampled replays, rank 0: step_in 0.012 ms embed 0.031 ms attn 9.8 ms ...
    # span M=6 per step over 16 sampled replays, rank 0 per layer mlp: 0.201 0.198 ...
    # span prefill pass 12 rank 0: prefill 481.2 ms (1024 rows)
    # span prefill pass 12 rank 0 ops: hc_attn 3.1 attn_proj 41.0 ... router 6.2 shared 24.0 ro_gate 1.1 ro_fc1 160.3 ro_fc2 110.2 experts 9.8 ...

`tools/trace/ds_span.py LOG... [--layers A-B]` sums the lines per print kind and rank, and splits the per-layer and
per-op times into the layers A..B and the rest (a release's trellis layers against its native ones).
