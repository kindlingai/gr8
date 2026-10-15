# Is a cancel's delay the event lookahead or the scheduler?

[#61](https://github.com/kindlingai/gr8/issues/61) reports a client that disconnected 1.5 ms into a 31,984-token prompt on
TP4; the slot stayed occupied for 23.819 s and a short follow-up saw 23.817 s TTFT. Two mechanisms produced a delay of
that shape, and both are now fixed:

- **the scheduler** served one prefill at a time and always the lowest request id, so a short request admitted behind a
  long prompt waited out every one of its quanta. Fixed by fair prefill (#63): the shortest remaining prompt goes next.
- **the event lookahead** anchored every notice at a constant `boundary + 8`, which is the right margin when each
  boundary is a decode round of tens of milliseconds — and during a prefill eight 8,192-row quanta, about 50 s, for a
  propagation that takes milliseconds. At the cluster's line this was structural: R is 65,536 = exactly eight 8,192-row
  quanta, so a below-R prompt can never span more than eight prefill boundaries and a notice anchored when it starts
  could not land inside it. Fixed by sizing the target by the work in flight: the first boundary whose work after the one
  in flight covers `NOTICE_ROWS` (2,048) prefill rows, within `boundary + NOTICE_MIN (2)` and `boundary + NOTICE_MAX
  (8)` (`s2/scheduler.h`, `Scheduler::horizon`; `s2/service.h`, `event`). During a prefill that is the floor; in pure
  decode it is the cap, which is exactly the old constant's behaviour.

This run attributes what is left, and verifies the sizing on the hardware. It needs `G6_S2_EVENTLOG=1`, because nothing
else prints the boundary a notice was anchored to.

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

- **notice** — rank 0's control thread, when it anchors the event: the boundary it read (`anchored`, call it `A`), the
  future boundary it chose (`boundary`, call it `N`), and the lookahead it used, `N - A`, which is 2 while a prefill
  quantum sits beyond the one in flight and 8 in pure decode.
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

Run it at `--prefill 4096 --pf-lanes 2` (the cluster's line), for P at 16K and 32K, and with `SESS` at 2 and at 4. A
prefill quantum is `prefill_rows * 2` = 8,192 rows, the last one short, and then a one-row prompt tail:

| P | prefill quanta | boundaries spent prefilling (incl. the tail) |
|---|---|---|
| 16,384 | 2 (8,192 + 8,191) | 3 |
| 32,768 | 4 (8,192 x 3 + 8,191) | 5 |

**Do not grow the prompt to discriminate — shrink the quantum.** At 8,192 rows a quantum, no below-R prompt outruns even
the old constant of eight (R is exactly eight quanta), so every size predicts the same thing; and at 64K the request
crosses R, where the run is exclusive and `RunPool::admissible` refuses the follow-up outright, so the abort timing and
the follow-up TTFT stop measuring the same thing. The discriminating configuration is `--prefill 2048 --pf-lanes 1`,
where a 32K prompt is sixteen boundaries and every notice lands mid-prefill — the configuration where #63's fair choice
also became visible (14.2 s against 28.5 s).

## Reading it

Take the long request's id (say `1`) and its step lines. Let `L` be the boundary of its **last** step line — the tail —
which is when its prefill finished. Take `N` and the lookahead from the notice line.

    grep -h "# s2_event" out/g6m4.r0.err

The lookahead field is the attribution: **2 during a prefill** says the target was sized by the work in flight and the
notice landed mid-prefill; **8** says the remaining work was cheap (decode rounds, or a remainder under `NOTICE_ROWS`
rows) and the target took the cap.

One correction worth writing down, because the first version of this document got it backwards: **`N >= L` does not
acquit the lookahead.** The measured run before the fix had `L` = 13 and `N` = 16, and the lookahead was exactly what
bound — the cancel anchored at boundary 8 with five boundaries of prefill left, so a lookahead of 8 could not land
inside it. `N >= L` is what a lookahead larger than the remaining prefill produces; had it been 1, the cancel would have
applied at boundary 10, about 6 s in rather than 20 s. With the sizing in, that case reads lookahead 2 and `N` inside
the prefill.

Also read the **serialization**: events apply one at a time, and the next notice cannot anchor until the previous one
commits, so each pays its own lookahead after the last. In the measured trace the follow-up's admit anchored at the
cancel's commit boundary (`anchored 16`), not at its own arrival — three requests are three serialized waits. The sizing
turns each of those waits from eight boundaries into two during a prefill; it does not remove the serialization itself.

And the follow-up's TTFT should track `N`, not the prefill's end: a boundary costs about 6 s during a prefill and tens of
milliseconds during a decode round, so TTFT ≈ the number of boundaries from the follow-up's admission to `N`, priced by
the kind of work in each. If TTFT instead tracks `L`, the abort is waiting for the prefill and not for the notice.

## What this run does not settle

- It says nothing about the **3% no-lost-wins rule**. That is the usual gate (`bench/gate/gate.py`), run alongside.
- `SESS` at 2 and 4 changes who competes for prefill quanta; if the two disagree about the lookahead, say so rather
  than picking one — that means a third term is in play.
- The 1.5 ms disconnect is a reproduction of the report, not a specification. The mechanism is the same for any cancel
  that arrives while a long prefill is running.
