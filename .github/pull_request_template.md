## What and why


## Merge-queue category (tick exactly one)

- [ ] **Bit-exact / off by default**: with default settings, output is byte-identical to main (new flags default off)
- [ ] **Changes defaults or performance**: default behaviour, speed or memory changes (needs its own A/B gate)
- [ ] **Front end only**: `frontend/`, `tools/launch/`, `smoketest/`, `./gr8` and Markdown, nothing else (.github/ci/scope.js); CI skips the engine builds
- [ ] **Docs or tests only**: no change to shipped code

CI runs once a maintainer comments `/ci` (CI only) or `/ready` (CI, and ready for the merge queue; `/unready` to
undo). A new push clears both.
