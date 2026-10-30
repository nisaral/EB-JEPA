# FAQ

**Do I need a GPU?**
No. `nisaralll/soro-kev-0.8b` runs on CPU; on the CI runner, loading takes about 15–45 s. The published
latency (124 ms for 0.8B, 649 ms for 4B per pass) was measured on an RTX 4050 Laptop GPU. 4B is practical
only on a GPU.

**What does `Router.heuristic()` do?**
It scores tools by hashed keyword overlap. It exists so the API, tests and examples run without
downloading weights. Do not use it for real routing.

**Which model should I use?**
Start with 0.8B on CPU or edge hardware. Use 4B when wrong calls are expensive or you need better refusal:
it refuses 25/33 BFCL irrelevance prompts, against 17/33 for 0.8B.

**It calls a tool when it shouldn't. What can I do?**
Lower `q_hat`, for example `Router.from_pretrained(..., q_hat=0.1)`. On the BFCL diagnostic this raised
0.8B refusals from 17 to 23 of 33 without losing correct calls. That was tuned on 100 rows, so check it
on your own traffic. See [RETRAIN_ANALYSIS](RETRAIN_ANALYSIS.md).

**Is the 90% coverage guaranteed?**
Only on data exchangeable with the calibration set. Refit q̂ on your own logs
(`scripts/recalibrate_qhat.py` shows how).

**How many tools can I pass?**
Any number. Up to 16 go in one pass; larger menus are chunked (see [concepts](concepts.md)). Accuracy figures
in the README come from menus of the benchmark's size; very large menus are less tested.

**Does it fill in arguments?**
No. It chooses the tool. `repair_args()` fixes argument keys and types that your LLM or user supplied;
it does not extract values from the query.

**Does it work with multi-turn conversations?**
It routes one text query. For chat, pass the latest user message (`POST /v1/route` does this). Multi-turn
τ²-bench routing is weak: 10.5% for 0.8B and 24.8% for 4B.

**Do the models need retraining?**
Not for v0.2.1. See [RETRAIN_ANALYSIS.md](RETRAIN_ANALYSIS.md) for what retraining would and would not fix.

**What licenses apply?**
The code is Apache-2.0. Base models are Qwen (see their cards). Training data sources and their licenses
are listed in [DATA_LICENSES.md](../DATA_LICENSES.md); several upstream licenses are not yet confirmed.
