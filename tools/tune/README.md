# tools/tune: v2 (exlr8 K2/K3/K4) fetch and manifest tools

- `fetch_ranks.py` : range-GET only the expert units a width manifest (or `all`, `k2,k3`, ...) needs for the chosen
  ranks and layers from the exlr8 HF repo into the v2 dir that `g6m4 --v2` (the launchers' `V2=`) reads. It plans the
  ranges from the ranks' index slices with `bin/g6load fetch-plan` (engine/loader-rs, built by `./gr8 glm53 build`), so
  it needs no numpy. Resumes, `--verify` checks every unit (`bin/g6load verify`). Non-LFS hub files are fetched from the direct URL with the token; LFS files via the CDN URL
  (no token sent to the CDN). `HF_TOKEN` is needed only for a private repo. The published one is public. Split layers across hosts (`--layers`) to use several uplinks.
  `fetch_layers()` is the same fetch as a function: `./gr8 glm53 up --model https://huggingface.co/...` calls it for
  one rank (tools/launch/hfsource.py).
- `run_fetch.sh RANKS WIDTHS LOG` : reruns fetch_ranks.py every 10 min until it succeeds (an upload still landing fails it).
- `route_manifest.py` : width manifest from a `G6_ROUTE_LOG` routing histogram x the per-unit error tables in the v2
  index, at a byte budget (greedy error-drop per byte), with an optional per-layer busiest-rank traffic cap
  (`--rank-cap base`). Code only: its manifests have not passed a quality gate yet.
