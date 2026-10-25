# n-gram draft source: offline replay (#56)

Step 2 of #56: before any engine work, measure on held-out GLM-5.3 traces whether a table of the model's own
near-certain continuations would propose drafts that the target accepts. Python standard library only, CPU only.

| file | what |
|---|---|
| `capture.py` | step 1: greedy generations with the top-k log-probabilities of every generated token, through the front end |
| `ngram.py` | steps 2-4: the table, chain and tree proposals, prompt lookup (#34) for comparison, and the replay |
| `ab.py` | the engine gate: one boot with `NGRAM` unset, one with it set, the same held-out prompts at temperature 0; outputs byte for byte, decode tok/s and tokens per verify step by workload |
| `data/prompts.json` | 45 agentic coding prompts: code (16), edit (8), tool calls with a tool list and multi-turn transcripts (11), JSON (6), prose (4) |
| `data/prompts_tool2.json` | 24 more tool-call turns for the table (the first cluster replay's tool row rested on 11): first turns and transcripts after one or two tool results |
| `data/ab_prompts.json` | 17 prompts held out of both table sets for the A/B: tool (8), edit (3), JSON (3), code (2), prose (1). Six tool prompts share the table sets' system prompt and tool list with new tasks (the repeated scaffolding is what the table is for); two use a system prompt and tools the table never saw |
| `tests/` | `python3 -m unittest discover -s tools/ngram/tests` |

## Run

    # 1. traces, on a booted cluster (any TP): one line per prompt, both table sets into one file
    for p in prompts prompts_tool2; do
      python3 tools/ngram/capture.py --url http://127.0.0.1:8100 --prompts tools/ngram/data/$p.json \
          --out traces.jsonl --max-tokens 1024 --top 8 --effort low
    done
    # 2. replay: 5 folds by prompt (table from 80%, replay the other 20%), chain and tree, rows 3/5/8
    python3 tools/ngram/ngram.py replay traces.jsonl --folds 5 --json replay.json
    # 3. table size against hit rate
    python3 tools/ngram/ngram.py sizes traces.jsonl --floors 0.1,0.3,0.5,0.7,0.9

    # 4. the engine's table (G6_NGRAM / NGRAM=FILE, engine/runtime/ngram_lookup.h), calibrated on held-out folds
    python3 tools/ngram/ngram.py export traces.jsonl --out ngram.tab

    # 5. the engine gate on two boots of the same build (NGRAM unset, then NGRAM=ngram.tab), same settings
    python3 tools/ngram/ab.py run --url http://127.0.0.1:8100 --prompts tools/ngram/data/ab_prompts.json \
        --label off --out ab_off.json --effort low --traces traces.jsonl
    python3 tools/ngram/ab.py run ... --label on --out ab_on.json            # after the reboot with NGRAM set
    python3 tools/ngram/ab.py compare ab_off.json ab_on.json                 # exits 1 on any output difference

`export` writes each context's top token and GLM's mean probability for it, and a calibration: per probability bucket
(`spec_len.h`'s edges), the share of held-out chain tokens GLM produced after a right prefix, shrunk toward the bucket's
lower edge. The engine prices a chain with the calibrated values, so the 30-60% overestimate the raw means showed on
the first cluster traces does not reach its choice. With `NGRAM` set, a request decoding alone with DFlash2 verifies the
table's chain instead of the drafter's drafts in rounds where the chain promises more tokens per verify ms
(`docs/SETTINGS.md`, `G6_NGRAM`).

`import-kl-ref` turns a `bench/gate` KL reference (16 greedy continuations of 256 tokens, top 20) into traces, which
checks the tools but is too little and too varied to judge the table. Its top lists come from a separate scoring pass:
in `refs/tp4.json` 108 of 4,096 positions have the generated token outside first place.

`sizes` sweeps the confidence floor (`--floors`), and at `--floor` the width (`--widths 2,3,4`) and the context lengths
(`--ns-list "8,6,4,3;4,3;3"`).

## What the replay does

- **Table.** For each context length n (8, 6, 4, 3 by default, longest first with backoff), the last n tokens map to
  the top `--width` next tokens, each with how many positions were seen and GLM's mean probability for it there (a
  token outside a position's top-k counts as 0 at that position). Entries seen fewer than `--min-count` times or with a
  mean below `--floor` are dropped. A proposed token extends the context for the next lookup, so a hit can chain.
- **Chain.** The top candidate at each step, up to rows - 1 tokens.
- **Tree.** The rows - 1 nodes with the highest path probability (best first). Path probability falls along each path,
  so this is a valid tree and, for the stored probabilities, the one with the most expected accepted tokens. Trees
  need tree verification (#57).
- **Rows.** By default a table proposal is cut to the prefix with the most expected tokens per ms (`--fill` keeps every
  row). Prompt lookup has no probabilities, so its proposals are never cut: it always fills the budget. Cost per
  round is the verify time of rows = drafts + 1 from `engine/runtime/spec_len.h` (TP4, 2200 MHz); lookups cost no
  GPU time.
- **Acceptance.** Greedy: the longest proposed path equal to GLM's own tokens, plus the one token the verify step
  always commits.
- **Sources.** `prompt` (#34's prompt lookup alone: the tokens after the latest earlier match of the last 4..2
  tokens), `chain`, `tree`, and `chain+prompt` / `tree+prompt` (prompt lookup when the table proposes nothing).
- **Folds.** By prompt: a prompt's traces are all on one side, so a rerun of the same prompt cannot teach the table its
  own held-out answer. Exact repeats of a line are dropped on load.

## Columns

`hit %`: rounds where the source had a proposal (for the table: an entry for the context, before the row cut).
`used %`: rounds that verified drafts. `accepted / used`: draft tokens accepted per round that verified drafts.
`tokens / round`: committed per verify step. `rows / round`, `tokens / row`: verify rows and tokens per row.
`expected tokens / row` and `table acc / round: expected (actual)`: what the table's own probabilities predicted, over
the rounds not served by prompt lookup, against what happened.
`speedup vs 1 row`: tokens per ms against plain decoding with the same verify table. It ignores the MTP or DFlash2
drafts the engine would otherwise make, so compare a source's tokens per round with theirs from the engine's log, not
the speedup with 1.

## The A/B (`ab.py`)

`run` templates each prompt on the server as `capture.py` does, then sends it to `/v1/completions` at temperature 0
with `return_token_ids` and `return_step_stats`, one request at a time: the table serves only a request decoding alone.
It asks for no log-probabilities, so decoding runs as it does when serving. Decode tok/s is (tokens - 1) over the
request's `step_stats.decode_s` (the first token to the end, on the front end's clock); replies under 50 tokens are not
timed, and the table says how many prompts were. `--traces` refuses the run before it generates anything when a
prompt's ids are in the table's traces (ids depend on `--effort`: use the traces' effort). A failed request becomes an
error row. `compare` fails on any prompt whose ids differ, any error row, or runs with different settings, prompt
files or models, and prints, by workload, the median decode tok/s and the decode tokens per verify step on each side.
It cannot tell whether the table served: on the `on` boot, check rank 0's `# s2 ngram slot` lines for nonzero table
rounds (the table serves only a request decoding alone with DFlash2, so a TP3 or MTP-only boot gives ratio 1).
With `G6ENV_X="G6_NGRAM_LOG=1"` rank 0 also logs each table round.
