# Contributing to Soro-Boro

Thanks for helping! Bug reports, docs fixes, new tool-format adapters and benchmark rows are all welcome.
Look in [docs/GOOD_FIRST_ISSUES.md](docs/GOOD_FIRST_ISSUES.md) for starter tasks.

## Setup

```bash
git clone https://github.com/nisaral/Soro-Boro && cd Soro-Boro
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU wheels are enough
pip install -e ".[dev]"
pre-commit install
```

## Everyday commands

| Task | Command |
| :--- | :--- |
| Full test suite (no weights needed) | `pytest -q` or `make test` |
| Heuristic smoke test | `python scripts/ci_smoke.py` |
| Real-weights smoke test (downloads 0.8B, CPU) | `SORO_RUN_PRETRAINED=1 pytest tests/test_pretrained_smoke.py` |
| Lint | `pre-commit run --all-files` or `make lint` |
| Environment check | `soro-boro doctor` |

## Guidelines

- **Keep PRs small** and focused. Add a test for each behaviour change. Tests must run offline; use
  `Router.heuristic()` or a dummy `Scorer`.
- **Benchmarks must be reproducible.** Any number in the README or docs must come from a committed log
  or script under `benchmarks/` or `scripts/`. Report per-distinct-query CIs (`scripts/per_query_ci.py`),
  not row counts alone.
- **Don't train on eval data.** Keep `external_heldout`, BFCL and τ² telecom/retail out of training
  sets (see `docs/RETRAIN_ANALYSIS.md`).
- **New data sources** need a license entry in `DATA_LICENSES.md`.
- Public API changes (`soro_boro.*`) need an update to `docs/API.md` and `CHANGELOG.md`.

## Project layout

```
veyra/python/src/soro_boro/   Router, tool schemas, CLI, server (the public API)
veyra/python/src/soro/        harness, scorers, conformal gate, training code
examples/                     runnable integration examples (tested in tests/test_examples.py)
benchmarks/                   prediction logs and tables behind every published number
scripts/                      analysis and CI helpers
docs/                         user docs, analyses, draft model-card updates
```

By contributing, you agree that your contributions are licensed under Apache-2.0 and that you will follow
the [Code of Conduct](CODE_OF_CONDUCT.md).
