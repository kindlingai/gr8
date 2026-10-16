# Flash reference (W0 of the Flash port)

`flash_ref.py` is an fp32, op-by-op GLM-5.3-Flash forward for one sequence, written the way the engine computes:
token-by-token decode with fixed KDA state, absorbed MLA over the 512-d latent, kpool indexer on a growing cache,
mHC streams, lazy routed experts, vLLM-semantics MTP (layer 45). Every op output goes to `rec(name, tensor)`.

`weights.py` reads tensors by checkpoint name from memory, local safetensors shards, or the HF hub by range request.

`test_flash_ref.py` proves the forward equals transformers 5.19.0 `Glm5NextTextModel` on a small random config:
per layer from identical inputs (prefill and token-by-token decode) to 1e-5 rel-L2, whole model to 1e-4, KDA chunk ==
recurrent, the indexer actually drops keys, MTP batched == per row.

    python -m venv --system-site-packages /tmp/fe/venv && /tmp/fe/venv/bin/pip install transformers==5.19.0
    /tmp/fe/venv/bin/python engine/ref/flash/test_flash_ref.py

Rules the engine must share with this reference:
- Indexer ties (pools whose score is exactly equal, e.g. every head's relu is 0) go to the lower pool index.
  transformers' `topk` leaves tie order unspecified.
- Compare kernels op by op from identical inputs. A 1e-5 difference flips near-tied router choices, so long chains
  diverge without any bug.
