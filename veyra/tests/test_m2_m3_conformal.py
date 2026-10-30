"""Unit tests for M2/M3 rigorous Conformal and PASC cascade."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from experiments.run_m2_m3_conformal_pasc import (
    extract_query_slots,
    conformal_quantile,
)


def test_extract_query_slots():
    slots1 = extract_query_slots("cat /var/log/syslog.txt")
    assert "path" in slots1

    slots2 = extract_query_slots("send email to test@domain.com")
    assert "email" in slots2 or "recipient" in slots2

    slots3 = extract_query_slots("run sql select query on table users")
    assert "query" in slots3
    assert "table" in slots3


def test_conformal_quantile_finite_order_statistic():
    import numpy as np
    data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    q = conformal_quantile(data, alpha=0.10)
    assert q == 1.0


def test_m2_m3_benchmark_artifacts_validity():
    p = Path(__file__).resolve().parents[1] / "reports" / "results_m2_m3_conformal_pasc.json"
    assert p.exists()
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["n_splits"] == 1000
    # Lac conformal maintains nominal coverage
    assert d["mondrian_lac_conformal"]["write_empirical_coverage_mean"] >= 0.95
    assert d["mondrian_lac_conformal"]["read_empirical_coverage_mean"] >= 0.90
    assert d["cascade_ablation"]["finding"].startswith("Degenerate")
    assert "fixed_k_curve" in d
    assert "score_margin_heuristic" in d["baselines_comparison"]
