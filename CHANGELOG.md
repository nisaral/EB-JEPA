# Changelog

All notable changes to `soro-boro` are recorded here. The Veyra execution-layer history before 0.2.1 is in
[veyra/CHANGELOG.md](veyra/CHANGELOG.md).

## [0.2.1] - unreleased

The weights are unchanged. `nisaralll/soro-kev-0.8b` and `nisaralll/soro-kev-4b` are the same checkpoints
as in 0.2.0.

### Added
- **`soro_boro.Router`**, a high-level routing API: `Router.from_pretrained()`, `Router.heuristic()`,
  `route()` and `route_batch()`. It returns a typed `RoutingDecision` (`action`, `tool`, `candidates`,
  `scores`, `confidence`, `repair_args()`, `to_dict()`).
- **Tool schema normalisation** (`normalize_tools`, `ToolSpec`) for OpenAI, MCP, Anthropic, LangChain,
  LlamaIndex, dicts, tuples and callables.
- **`soro-boro` CLI** with `route`, `eval`, `serve`, `doctor` and `convert-head`. Every command takes
  `--heuristic` for offline use.
- **HTTP server** (`soro-boro[server]`) with `GET /healthz`, `POST /route`, and `POST /v1/route`, which
  accepts OpenAI messages and returns a `tool_choice`.
- `examples/`: an OpenAI pre-router, an MCP gateway and LangChain/LlamaIndex adapters. All run offline
  and are tested.
- Docs: `docs/quickstart.md`, `concepts.md`, `API.md`, `FAQ.md`, `RETRAIN_ANALYSIS.md` and
  `OPEN_QUESTIONS.md`, plus `DATA_LICENSES.md`.
- Analysis scripts: `scripts/per_query_ci.py` (cluster-bootstrap CIs over distinct queries) and
  `scripts/recalibrate_qhat.py` (refits q̂ from committed logs).
- CI: the full test suite on Python 3.10–3.13, a heuristic smoke test, and a cached real-weights smoke job
  for 0.8B on CPU.
- Extras: `server`, `grpc`, `demo`, `test`, `dev` and `all`. A root `Makefile`, `.pre-commit-config.yaml`,
  contributor docs and issue/PR templates.

### Changed
- `SoroHarness.route(candidates=...)` now emits a `DeprecationWarning` pointing to `Router`.
- `head.pt` loads with `weights_only=True` and warns. `soro-boro convert-head` writes `head.safetensors`.
- `conformal_gate` and `repair_arguments` are now reusable functions.

### Fixed
- Docs now state the single-pass menu size correctly (≤ 16 tools, not 8).
- Loss docs now state the JSD weight correctly (0.30; 0 only for `single_sequence`, which the 4B run used).
- Corrected a stale q̂ calibration note, the external q̂ statement and the release badge link.
- Removed dead Go-kernel targets from `veyra/Makefile`.
- Tests are isolated: per-test claim DB and WAL, fixed golden/report paths, and skips when optional
  benchmark data is absent.

### Documented limitations
- README cluster CIs: 90.6% (85.9–94.7) for 4B and 85.7% (80.3–90.6) for 0.8B on the external held-out set.
- τ² routing: 24.8% (4B) and 10.5% (0.8B).
- 0.8B no-tool refusal: 17/33 against 27/33 for base kev-0.8b.
- The "human" split is a byte copy of the external held-out split.
- Several upstream data licenses are unconfirmed.
