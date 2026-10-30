"""Unit and regression tests for rigorous M4 real-tool fault injection safety evidence."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from experiments.run_m4_fault_injection import (
    run_sqlite_m4_suite,
    run_third_party_mcp_sqlite_suite,
    run_git_sandbox_m4_suite,
    run_http_ledger_m4_suite,
)


def test_m4_sqlite_suite_metrics():
    res = run_sqlite_m4_suite(n_trials=100)
    metrics = res["overall_metrics"]
    assert metrics["blind_retry"]["duplicate_rate"] > 0.0
    assert metrics["no_retry"]["lost_write_rate"] > 0.0
    assert metrics["naive_read_check"]["duplicate_rate"] > 0.0
    # Opaque regime defers when verifier crashes or race occurs
    assert metrics["soro_opaque_tool"]["duplicate_rate"] == 0.0
    assert metrics["soro_opaque_tool"]["deferred_rate"] > 0.20
    # Id-verified regime cuts deferral to under 10%
    assert metrics["soro_id_verified"]["duplicate_rate"] == 0.0
    assert metrics["soro_id_verified"]["deferred_rate"] <= 0.10
    assert metrics["soro_id_verified"]["exactly_once_rate"] >= 0.90


def test_m4_third_party_mcp_suite_metrics():
    res = run_third_party_mcp_sqlite_suite(n_trials=100)
    metrics = res["overall_metrics"]
    assert metrics["blind_retry"]["duplicate_rate"] > 0.0
    assert metrics["idempotency_key"]["exactly_once_rate"] == 1.0
    assert metrics["soro_id_verified"]["duplicate_rate"] == 0.0
    assert metrics["soro_id_verified"]["deferred_rate"] <= 0.10


def test_m4_git_sandbox_suite_metrics():
    res = run_git_sandbox_m4_suite(n_trials=100)
    metrics = res["overall_metrics"]
    assert metrics["blind_retry"]["duplicate_rate"] > 0.0
    assert metrics["soro_id_verified"]["duplicate_rate"] == 0.0
    assert metrics["soro_id_verified"]["deferred_rate"] <= 0.10


def test_m4_http_suite_metrics():
    res = run_http_ledger_m4_suite(n_trials=100)
    metrics = res["overall_metrics"]
    assert metrics["blind_retry"]["duplicate_rate"] > 0.0
    assert metrics["idempotency_key"]["exactly_once_rate"] == 1.0
    assert metrics["soro_id_verified"]["duplicate_rate"] == 0.0


def test_m4_report_artifact_validity():
    p = Path(__file__).resolve().parents[1] / "reports" / "results_m4_real_faults.json"
    assert p.exists()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert len(data["systems"]) == 4
    for sys_res in data["systems"]:
        assert sys_res["total_trials"] == 1000
        assert sys_res["at_risk_trials"] == 700
        assert sys_res["at_risk_95_upper_bound"] <= 0.0043
        assert sys_res["overall_metrics"]["soro_opaque_tool"]["duplicate_rate"] == 0.0
        assert sys_res["overall_metrics"]["soro_id_verified"]["duplicate_rate"] == 0.0
        assert sys_res["overall_metrics"]["soro_id_verified"]["deferred_rate"] <= 0.10
        assert "per_regime_raw_counts" in sys_res
