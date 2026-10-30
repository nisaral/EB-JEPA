# Veyra Python harness / decision plane.
#
# The Go kernel and the `veyra.cli` bench driver that earlier versions of this
# Makefile built were removed in 1fd4d3f ("purge legacy go/veyra
# directories"); their targets (build, test-go, lint, fmt, bench-dev, train,
# bench-test) are gone too. Use the repo-root Makefile for day-to-day work.

SHELL := /bin/sh
PY    ?= python3

export PYTHONPATH := python/src:$(PYTHONPATH)

.PHONY: help proto proto-check test test-py doctor clean

help:
	@echo "targets:"
	@echo "  proto        regenerate the Python wire format from proto/ (needs grpcio-tools)"
	@echo "  proto-check  regenerate and fail if the committed output differs"
	@echo "  test         pytest (Python harness tests)"
	@echo "  doctor       environment check (soro-boro doctor)"
	@echo "  clean        remove run artifacts"

proto:
	$(PY) -m grpc_tools.protoc -I proto --python_out=python/src --pyi_out=python/src \
	  --grpc_python_out=python/src proto/veyra/v1/runtime.proto

proto-check: proto
	git diff --exit-code -- python/src/veyra/v1

test: test-py

test-py:
	cd .. && $(PY) -m pytest -q veyra/python/tests veyra/tests

doctor:
	cd .. && $(PY) -m soro_boro.cli doctor

clean:
	rm -rf out .pytest_cache
