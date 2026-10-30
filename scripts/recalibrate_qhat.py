"""Recompute split-conformal LAC thresholds from the committed prediction logs (CPU, seconds).

For each model and eval set in ``benchmarks/predictions/`` it reports:

* q̂(α=0.10) fitted on the full set, using the score s = 1 − p(true label) and the finite-sample quantile
  ceil((n+1)(1−α))/n;
* empirical coverage, mean |S| and the EXECUTE/ASK/ABSTAIN mix under the **deployed** q̂;
* for the external set, q̂ refit over 200 random 50/50 splits **grouped by base query** (no leakage
  between views of the same query), with test coverage;
* on BFCL irrelevance rows, the refusal rate under argmax, under the deployed gate, and as q̂ varies.

Usage: python scripts/recalibrate_qhat.py [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "benchmarks/predictions"
DEPLOYED = {"soro_0.8b": 0.9518, "soro_4b": 0.8066}
EXT = ROOT / "veyra/data/soro_kev_dataset/val_external_human.jsonl"
A = "__abstain__"


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def true_label(r: dict) -> str:
    return A if r.get("is_abstain_row") else r["label"]


def score(r: dict) -> float:
    return 1.0 - float(r["probabilities"].get(true_label(r), 0.0))


def qhat(scores: list[float], alpha: float = 0.10) -> float:
    n = len(scores)
    k = min(n, math.ceil((n + 1) * (1 - alpha)))
    return sorted(scores)[k - 1]


def gate(probs: dict, q: float) -> tuple[str, list[str]]:
    thr = 1 - q
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    S = [k for k, p in ranked if p >= thr]
    tools = [k for k in S if k != A]
    if not tools or ranked[0][0] == A:
        return "ABSTAIN", S
    return ("EXECUTE" if len(tools) == 1 else "ASK"), S


def evaluate(rows: list[dict], q: float) -> dict:
    cov = size = 0
    acts = Counter()
    exec_ok = 0
    for r in rows:
        act, S = gate(r["probabilities"], q)
        cov += true_label(r) in S
        size += len(S)
        acts[act] += 1
        if act == "EXECUTE":
            exec_ok += [k for k in S if k != A][0] == true_label(r)
    n = len(rows)
    return {
        "coverage": cov / n, "mean_set_size": size / n,
        "execute": acts["EXECUTE"] / n, "ask": acts["ASK"] / n, "abstain": acts["ABSTAIN"] / n,
        "execute_precision": exec_ok / acts["EXECUTE"] if acts["EXECUTE"] else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    out: dict = {}
    base_ids = [re.match(r"ext_human_(\d+)", json.loads(l)["id"]).group(1)
                for l in EXT.read_text().splitlines() if l.strip()]
    for model, q_dep in DEPLOYED.items():
        out[model] = {}
        for split in ("zeroshot", "external_heldout", "bfcl"):
            f = P / split / f"{model}.jsonl"
            if not f.exists():
                continue
            rows = load(f)
            s = [score(r) for r in rows]
            res = {"n": len(rows), "q_hat_full_alpha0.10": qhat(s), "deployed": evaluate(rows, q_dep),
                   "argmax_acc": sum(r["is_correct"] for r in rows) / len(rows)}
            res["refit"] = evaluate(rows, res["q_hat_full_alpha0.10"])
            if split == "external_heldout":
                groups = defaultdict(list)
                for i, r in enumerate(rows):
                    groups[base_ids[i]].append(r)
                keys = sorted(groups)
                rng = random.Random(42)
                qs, covs = [], []
                for _ in range(200):
                    rng.shuffle(keys)
                    cal = [r for k in keys[: len(keys) // 2] for r in groups[k]]
                    test = [r for k in keys[len(keys) // 2:] for r in groups[k]]
                    q = qhat([score(r) for r in cal])
                    qs.append(q)
                    covs.append(evaluate(test, q)["coverage"])
                qs.sort(); covs.sort()
                res["grouped_split_q_hat_median"] = qs[100]
                res["grouped_split_q_hat_90pct_range"] = (qs[10], qs[189])
                res["grouped_split_test_coverage_mean"] = sum(covs) / len(covs)
                res["grouped_split_test_coverage_min"] = covs[0]
            if split == "bfcl":
                irr = [r for r in rows if r.get("is_abstain_row")]
                res["irrelevance_n"] = len(irr)
                res["irrelevance_refusal_argmax"] = sum(r["top_tool"] == A for r in irr)
                res["irrelevance_refusal_by_q_hat"] = {
                    f"{q:.3f}": {"refused": sum(gate(r["probabilities"], q)[0] == "ABSTAIN" for r in irr),
                                 "tool_rows_execute_ok": sum(
                                     gate(r["probabilities"], q)[0] == "EXECUTE" and r["is_correct"]
                                     for r in rows if not r.get("is_abstain_row"))}
                    for q in (0.5, 0.7, 0.8066, 0.9, 0.9518, 0.99)}
            out[model][split] = res
    print(json.dumps(out, indent=1, default=lambda x: round(x, 4) if isinstance(x, float) else x))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
