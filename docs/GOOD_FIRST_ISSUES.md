# Good first issues

These are ready-to-file starter tasks. Each one is small, testable offline, and points at the code involved.
They are filed as GitHub issues #2–#11 with the `good first issue` and `help wanted` labels: [open list](https://github.com/nisaral/Soro-Boro/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

1. **`veyra/tests` collects cleanly on its own.** Running `pytest veyra/tests` alone gives 3 collection
   errors (`veyra.core` missing); the full root run is fine. Make those modules skip cleanly or fix the
   imports. *Files:* `veyra/tests/`, `repo_test_support.py`.
2. **Async router.** Add `Router.aroute()` and `aroute_batch()`, running the scorer in a thread pool, and
   make the server endpoints async. *Files:* `soro_boro/router.py`, `server.py`; tests with a dummy scorer.
3. **Abstain-bias knob.** Add an `abstain_bias` parameter that adds δ to the `__abstain__` logit before the
   gate. Sweep δ with the logit caches in `veyra/reports/logit_cache_*` and report the refusal-versus-accuracy
   trade-off (see `docs/RETRAIN_ANALYSIS.md`, cheap experiment 3).
4. **Permutation averaging.** Add `Router(..., permutations=k)`, which averages probabilities over k
   shuffled tool orders. Measure the order-flip rate on 100 external rows on CPU. This would back the
   README's 1.5–3.0% claim with a script.
5. **`soro-boro calibrate` CLI.** Wrap `scripts/recalibrate_qhat.py`: read a JSONL file of
   `{query, tools, expected}` and print or save a q̂ for a target α.
6. **Pydantic-v2 tool models.** Accept a bare `pydantic.BaseModel` subclass as a tool, with the class
   docstring as the description. *File:* `soro_boro/tool_schema.py`.
7. **Real LangChain/LlamaIndex test.** Add an optional CI job that installs `langchain-core` and
   `llama-index-core` and runs `examples/framework_adapters.py` against the real classes. They are
   stand-ins today.
8. **`head.safetensors` on the Hub.** Document and test a load path that prefers `head.safetensors`
   when both files exist, plus a `doctor` warning when only `head.pt` is present.
9. **Widen the lint scope.** Extend the ruff `files:` pattern in `.pre-commit-config.yaml` to one more
   module at a time (for example `soro/harness/`) and fix what it flags.
10. **Ruff/pytest config polish.** Set `asyncio_default_fixture_loop_scope = "function"` in
    `pyproject.toml` to silence the pytest-asyncio deprecation warning, and check that CI stays green.
