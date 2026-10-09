# tools/tune: v2 (exlr8 K2/K3/K4) fetch and manifest tools

- `fetch_ranks.py` : range-GET only the expert units a width manifest (or `all`, `k2,k3`, ...) needs for the chosen
  ranks and layers from the exlr8 HF repo into the v2 dir that `g6m4 --v2` (the launchers' `V2=`) reads. Resumes, `--verify`
  checks every unit. Non-LFS hub files are fetched from the direct URL with the token; LFS files via the CDN URL
  (no token sent to the CDN). `HF_TOKEN` is needed only for a private repo. The published one is public. Split layers across hosts (`--layers`) to use several uplinks.
- `run_fetch.sh RANKS WIDTHS LOG` : reruns fetch_ranks.py every 10 min until no layer is skipped (upload still landing).
- `route_manifest.py` : width manifest from a `G6_ROUTE_LOG` routing histogram x the per-unit error tables in the v2
  index, at a byte budget (greedy error-drop per byte), with an optional per-layer busiest-rank traffic cap
  (`--rank-cap base`). Code only: its manifests have not passed a quality gate yet.
