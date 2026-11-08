"""
Test Benchmark Reconciliation and Mathematical Consistency Invariants.

Enforces:
1. Total row count matches sum of subgroup row counts.
2. Total correct count matches sum of subgroup correct counts.
3. Weighted subgroup accuracy matches overall accuracy to 1e-9.
4. Clean vs Collision row counts partition the zero-shot split without remainder.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PRED_DIR = REPO_ROOT / "benchmarks" / "predictions"


def load_predictions(split: str, system: str) -> list[dict]:
    path = PRED_DIR / split / f"{system}.jsonl"
    if not path.exists():
        pytest.skip(f"Predictions file not found: {path}")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


@pytest.mark.parametrize("system", ["bm25", "dense_cosine", "cross_encoder", "option_logits", "soro_0.8b", "soro_4b"])
def test_external_heldout_reconciliation(system: str):
    pred_path = PRED_DIR / "external_heldout" / f"{system}.jsonl"
    if not pred_path.exists():
        pytest.skip(f"Predictions for {system} on external_heldout not generated yet.")

    records = load_predictions("external_heldout", system)
    total_n = len(records)
    assert total_n == 1200, f"Expected 1,200 rows in external heldout, got {total_n}"
    total_correct = sum(1 for r in records if r["is_correct"])

    # 1. Package partition check
    packages = ["mcp_benchmark/system_mcp", "mcp_benchmark/geo_mcp", "mcp_benchmark/media_mcp", "tau2/telecom", "tau2/retail"]
    pkg_totals = {pkg: sum(1 for r in records if r["package"] == pkg) for pkg in packages}
    pkg_correct = {pkg: sum(1 for r in records if r["package"] == pkg and r["is_correct"]) for pkg in packages}

    assert sum(pkg_totals.values()) == total_n
    assert sum(pkg_correct.values()) == total_correct
    weighted_pkg_acc = sum((pkg_correct[p] / pkg_totals[p]) * (pkg_totals[p] / total_n) for p in packages)
    assert abs(weighted_pkg_acc - (total_correct / total_n)) < 1e-9

    # 2. Abstain partition check
    non_abs_n = sum(1 for r in records if not r["is_abstain_row"])
    non_abs_c = sum(1 for r in records if not r["is_abstain_row"] and r["is_correct"])
    abs_n = sum(1 for r in records if r["is_abstain_row"])
    abs_c = sum(1 for r in records if r["is_abstain_row"] and r["is_correct"])

    assert non_abs_n + abs_n == total_n
    assert non_abs_c + abs_c == total_correct
    weighted_abs_acc = (non_abs_c / non_abs_n) * (non_abs_n / total_n) + (abs_c / abs_n) * (abs_n / total_n)
    assert abs(weighted_abs_acc - (total_correct / total_n)) < 1e-9

    # 3. Disjoint tools partition check
    disj_n = sum(1 for r in records if not r["is_shared_tool"])
    disj_c = sum(1 for r in records if not r["is_shared_tool"] and r["is_correct"])
    shared_n = sum(1 for r in records if r["is_shared_tool"])
    shared_c = sum(1 for r in records if r["is_shared_tool"] and r["is_correct"])

    assert disj_n == 1038
    assert shared_n == 162
    assert disj_n + shared_n == total_n
    assert disj_c + shared_c == total_correct
    weighted_disj_acc = (disj_c / disj_n) * (disj_n / total_n) + (shared_c / shared_n) * (shared_n / total_n)
    assert abs(weighted_disj_acc - (total_correct / total_n)) < 1e-9


@pytest.mark.parametrize("system", ["bm25", "dense_cosine", "cross_encoder", "option_logits", "soro_0.8b", "soro_4b"])
def test_zeroshot_reconciliation(system: str):
    pred_path = PRED_DIR / "zeroshot" / f"{system}.jsonl"
    if not pred_path.exists():
        pytest.skip(f"Predictions for {system} on zeroshot not generated yet.")

    records = load_predictions("zeroshot", system)
    total_n = len(records)
    if total_n < 2220:
        pytest.skip(f"Predictions for {system} on zeroshot still generating ({total_n}/2220 rows)...")
    assert total_n == 2220, f"Expected 2,220 rows in zeroshot, got {total_n}"
    total_correct = sum(1 for r in records if r["is_correct"])

    clean_n = sum(1 for r in records if not r["has_collision"])
    clean_c = sum(1 for r in records if not r["has_collision"] and r["is_correct"])
    col_n = sum(1 for r in records if r["has_collision"])
    col_c = sum(1 for r in records if r["has_collision"] and r["is_correct"])

    assert clean_n == 2168
    assert col_n == 52
    assert clean_n + col_n == total_n
    assert clean_c + col_c == total_correct

    weighted_acc = (clean_c / clean_n) * (clean_n / total_n) + (col_c / col_n) * (col_n / total_n)
    assert abs(weighted_acc - (total_correct / total_n)) < 1e-9


@pytest.mark.parametrize("system", ["bm25", "dense_cosine", "cross_encoder", "option_logits", "soro_0.8b", "soro_4b", "arch_router", "xlam_1b", "hammer_1.5b", "kev_27b"])
def test_bfcl_reconciliation(system: str):
    pred_path = PRED_DIR / "bfcl" / f"{system}.jsonl"
    if not pred_path.exists():
        pytest.skip(f"Predictions for {system} on bfcl not generated yet.")

    records = load_predictions("bfcl", system)
    total_n = len(records)
    assert total_n == 100, f"Expected 100 rows in BFCL diagnostic slice, got {total_n}"
    total_correct = sum(1 for r in records if r["is_correct"])

    files = ["BFCL_v3_multiple.json", "BFCL_v3_live_multiple.json", "BFCL_v3_irrelevance.json"]
    f_totals = {f: sum(1 for r in records if r["bfcl_file"] == f) for f in files}
    f_correct = {f: sum(1 for r in records if r["bfcl_file"] == f and r["is_correct"]) for f in files}

    assert f_totals["BFCL_v3_multiple.json"] == 34
    assert f_totals["BFCL_v3_live_multiple.json"] == 33
    assert f_totals["BFCL_v3_irrelevance.json"] == 33
    assert sum(f_totals.values()) == total_n
    assert sum(f_correct.values()) == total_correct

    weighted_acc = sum((f_correct[f] / f_totals[f]) * (f_totals[f] / total_n) for f in files)
    assert abs(weighted_acc - (total_correct / total_n)) < 1e-9
