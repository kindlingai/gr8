# Offline adaptive-policy reset

Flash `g6m4_flash --spec-reset-per-request` restores the initial, fully configured
`SpecLen` before each offline `--prompt`. The flag is OFF by default. It is not
a served option: `--serve` with this flag is rejected. S2's MTP admission already
creates fresh per-session estimates. With `--spec-fixed`, there is no adaptive
policy to learn, so the reset has no effect on decisions.

The snapshot is taken after levels, Flash verify costs, and drafting costs are
configured. It resets `pn`, `pk`, `bn`, `bk`, `rounds`, `chosen`, `pred_sum` and
`got_sum`. It preserves `on`, levels, costs, the table, seeds, decays and prior.
The flag does not change cache resets, transport sequence counters, sampling
seeds, cumulative `MtpStats`, or device code. The adaptive summary printed for
an ON request describes that request rather than all earlier requests.

This is an evaluation-isolation option. It does not establish row-count
invariance or prove that policy carry caused a particular token difference.
Both ranks must use the same flag and adaptive configuration.

## CPU checks

```
g++ -O2 -std=c++17 -DG6_MODEL_FLASH=1 engine/tests/test_spec_reset.cpp -o /tmp/test_spec_reset
/tmp/test_spec_reset
python3 engine/tests/test_spec_reset_source.py
g++ -O2 -std=c++17 -DG6_MODEL_FLASH=1 engine/tests/test_spec_len.cpp -o /tmp/test_spec_len
/tmp/test_spec_len
```

The first test checks every policy field, OFF carry against the old policy,
five identical synthetic ON requests, and custom gapped levels. The source
check verifies the default, parser, snapshot position, and offline reset site.
CPU checks are not GPU output proof.

## Pending GPU gate

On the same eight-layer fixture and rank pair, compare the parent build with
the candidate without the flag. Require byte-identical stdout per rank (not
between ranks, whose head output is sharded), generated tokens, and counters.
Then run five identical offline prompts in one boot with the flag. Compare
per-request token, LP-bit, residual-hash, step-shape and adaptive-decision
records, excluding request ordinal and elapsed-time labels. Check cross-rank
generated tokens and all guards. Preserve raw stdout and stderr. No GPU gate
or performance result is claimed here.
