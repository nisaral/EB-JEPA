PY ?= python3

.PHONY: help install test smoke smoke-pretrained lint doctor

help:
	@echo "install           pip install -e '.[dev]'"
	@echo "test              full test suite (no weights, no network)"
	@echo "smoke             heuristic harness smoke test"
	@echo "smoke-pretrained  download nisaralll/soro-kev-0.8b and test it on CPU"
	@echo "lint              pre-commit on all files"
	@echo "doctor            soro-boro doctor"

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

smoke:
	$(PY) scripts/ci_smoke.py

smoke-pretrained:
	SORO_RUN_PRETRAINED=1 $(PY) -m pytest tests/test_pretrained_smoke.py -q -rs

lint:
	pre-commit run --all-files

doctor:
	$(PY) -m soro_boro.cli doctor
