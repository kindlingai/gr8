# Is a cancel's delay the event lookahead or the scheduler?

[#61](https://github.com/kindlingai/gr8/issues/61) reports a client that disconnected 1.5 ms into a 31,984-token prompt on
TP4; the slot stayed occupied for 23.819 s and a short follow-up saw 23.817 s TTFT. Two mechanisms produce a delay of
that shape and the report alone cannot separate them — they need different fixes:

- **the scheduler**, which served one prefill at a time and always the lowest request id, so a short request admitted
  behind a long prompt waited out every one of its quanta. Fixed by fair prefill (#63): the shortest remaining prompt
  goes next.
- **the event lookahead**, which anchors a cancel at `boundary + LOOKAHEAD` with `LOOKAHEAD = 8` (`s2/service.h`). Eight
  boundaries is fine slack for decode rounds of tens of milliseconds; during a prefill it is up to eight 8,192-row
  quanta, about 50 s.

This is the run that tells them apart. It needs `G6_S2_EVENTLOG=1`, because nothing else prints the boundary a notice was
anchored to.

## What gets logged

Start the cluster with the diagnostic in the engine environment:

    G6ENV_X="G6_S2_EVENTLOG=1" ./gr8 glm53 up

`G6ENV_X` is the launch knob for extra engine environment on every rank. Do **not** export `G6_S2_EVENTLOG` directly:
the engine does not inherit `G6_*` from the environment, and `./gr8 glm53 up` refuses to start while one is set
([KNOBS.md](../KNOBS.md)). As a cluster-scope knob it goes into the pre-boot handshake, so either every box runs with it
or none does.

Three kinds of line then land on stderr, `out/g6m4.r<r>.err`:

    # s2_event notice rank 0 event 3 cancel request 2 anchored 41 -> boundary 49 (lookahead 8)
    # s2_event commit rank 0 event 3 cancel request 2 boundary 49
    # s2_event step rank 0 boundary 42 request 1 prefill first 0 rows 8192

- **notice** — rank 0's control thread, when it anchors the event: the boundary it read (`anchored`, call it `A`) and the
  future boundary it chose (`boundary`, `A + LOOKAHEAD`, call it `N`).
- **commit** — every rank, when the notice lands and the canonical changes apply. `boundary` is `N`; if a rank never
  prints this line for an event, that rank failed the epoch.
- **step** — every rank, one per prefill quantum and one per prompt tail (**not** per decode round, which would be far
  too much). `first` is the row the quantum starts at, `rows` how many it covers, so `first + rows` is the prompt
  position reached at that boundary.

The lines are off by default, add no state and change no behaviour, so this runs on a serving build.

## The workload

Per size: send one long prompt, abort the streaming client about 1.5 ms after the engine admits it, then send a short
follow-up (a few hundred tokens) and record its TTFT. The shim turns the disconnect into the CANCEL
(`frontend/g6s2shim.py`, `send_cancel`), so no engine change is needed to produce the event.

Run it at `--prefill 4096 --pf-lanes 2` (the cluster's line), for P at 16K, 32K and 64K, and with `SESS` at 2 and at 4.
A prefill quantum is `prefill_rows * 2` = 8,192 rows, the last one short, and then a one-row prompt tail:

| P | prefill quanta | boundaries spent prefilling (incl. the tail) |
|---|---|---|
| 16,384 | 2 (8,192 + 8,191) | 3 |
| 32,768 | 4 (8,192 x 3 + 8,191) | 5 |
| 65,536 | 8 (8,192 x 7 + 8,191) | 9 |

**64K is the discriminator.** `LOOKAHEAD` is 8, so a cancel anchored at the admission boundary `A` lands at `A + 8`: for
64K that is inside the eighth quantum, for 32K and 16K it is past the end of the prefill either way. Without a size that
outruns the lookahead, both mechanisms predict the same thing and the run proves nothing.

## Reading it

Take the long request's id (say `1`) and its step lines. Let `L` be the boundary of its **last** step line — the tail —
which is when its prefill finished. Take `N` from the notice line.

    grep -h "# s2_event" out/g6m4.r0.err

- **`N < L`** — the notice landed while the long prefill was still running. The lookahead is what held the abort, so
  step 3 (size the notice by the work in flight rather than a constant) is needed, and #63 alone will not close #61.
- **`N >= L`** — the prefill had already finished before the notice's boundary. The abort only ever waited for the
  prefill, which is the scheduler's admission; #63 is the whole story and step 3 can wait.

And the follow-up's TTFT should track `N`, not the prefill's end: a boundary costs about 6 s during a prefill and tens of
milliseconds during a decode round, so TTFT ≈ the number of boundaries from the follow-up's admission to `N`, priced by
the kind of work in each. If TTFT instead tracks `L`, the abort is waiting for the prefill and not for the notice.

## What this run does not settle

- It says nothing about the **3% no-lost-wins rule**. That is the usual gate (`bench/gate/gate.py`), run alongside.
- `SESS` at 2 and 4 changes who competes for prefill quanta; if the two sizes disagree about `N` vs `L`, say so rather
  than picking one — that means a third term is in play.
- The 1.5 ms disconnect is a reproduction of the report, not a specification. The mechanism is the same for any cancel
  that arrives while a long prefill is running.
