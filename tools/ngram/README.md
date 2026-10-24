# n-gram draft source: offline replay (#56)

Step 2 of #56: before any engine work, measure on held-out GLM-5.3 traces whether a table of the model's own
near-certain continuations would propose drafts that the target accepts. Python standard library only, CPU only.

| file | what |
|---|---|
| `capture.py` | step 1: greedy generations with the top-k log-probabilities of every generated token, through the front end |
| `ngram.py` | steps 2-4: the table, chain and tree proposals, prompt lookup (#34) for comparison, and the replay |
| `data/prompts.json` | 45 agentic coding prompts: code (16), edit (8), tool calls with a tool list and multi-turn transcripts (11), JSON (6), prose (4) |
| `tests/` | `python3 -m unittest discover -s tools/ngram/tests` |

## Run

    # 1. traces, on a booted cluster (any TP): one line per prompt
    python3 tools/ngram/capture.py --url http://127.0.0.1:8100 --prompts tools/ngram/data/prompts.json \
        --out traces.jsonl --max-tokens 1024 --top 8
    # 2. replay: 5 folds by trace (table from 80%, replay the other 20%), chain and tree, rows 3/5/8
    python3 tools/ngram/ngram.py replay traces.jsonl --folds 5 --json replay.json
    # 3. table size against hit rate
    python3 tools/ngram/ngram.py sizes traces.jsonl --floors 0.1,0.3,0.5,0.7,0.9

`import-kl-ref` turns a `bench/gate` KL reference (16 greedy continuations of 256 tokens, top 20) into traces, which
is enough to check the tools but too little and too varied to judge the table.

## What the replay does

- **Table.** For each context length n (8, 6, 4, 3 by default, longest first with backoff), the last n tokens map to
  the top `--width` next tokens, each with how many positions were seen and GLM's mean probability for it there (a
  token outside a position's top-k counts as 0 at that position). Entries seen fewer than `--min-count` times or with a
  mean below `--floor` are dropped. A proposed token extends the context for the next lookup, so a hit can chain.
- **Chain.** The top candidate at each step, up to rows - 1 tokens.
- **Tree.** The rows - 1 nodes with the highest path probability (best first). Path probability falls along each path,
  so this is a valid tree and, for the stored probabilities, the one with the most expected accepted tokens. Trees
  need tree verification (#57).
- **Rows.** By default the proposal is cut to the prefix with the most expected tokens per ms (`--fill` keeps every
  row). Cost per round is the verify time of rows = drafts + 1 from `engine/runtime/spec_len.h` (TP4, 2200 MHz);
  lookups cost no GPU time.
- **Acceptance.** Greedy: the longest proposed path equal to GLM's own tokens, plus the one token the verify step
  always commits.
- **Sources.** `prompt` (#34's prompt lookup alone: the tokens after the latest earlier match of the last 4..2
  tokens), `chain`, `tree`, and `chain+prompt` / `tree+prompt` (prompt lookup when the table has nothing).

## Columns

`hit %`: rounds with a proposal. `accepted / hit`: draft tokens accepted per round with a proposal. `tokens / round`:
committed per verify step. `tokens / row`: committed per verify row. `expected acc / round`: what the table's own
probabilities predicted (calibration check). `speedup vs 1 row`: tokens per ms against plain decoding with the same
verify table; it ignores the MTP or DFlash2 drafts the engine would otherwise make, so compare it with their own
tokens per round from the engine's log, not with 1.
