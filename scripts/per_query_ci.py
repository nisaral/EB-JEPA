"""Per-distinct-query confidence intervals for the external held-out table.

The 1,200 external rows are ~10 format/distractor views of a much smaller set
of base queries, so row-level binomial CIs are too narrow. This script groups
rows by base query id (``ext_human_<NNNNNN>_...``) and reports:

* row-level accuracy with a Wilson 95% CI (what the tables imply), and
* a cluster bootstrap 95% CI that resamples base queries (the honest one).

Usage: python scripts/per_query_ci.py [--boot 10000] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "veyra/data/soro_kev_dataset/val_external_human.jsonl"
PRED = ROOT / "benchmarks/predictions/external_heldout"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--json")
    a = ap.parse_args()
    rows = [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]
    base = [re.match(r"ext_human_(\d+)", r["id"]).group(1) for r in rows]
    pkg_family = [r["package"].split("/")[0] for r in rows]
    out = {"n_rows": len(rows), "n_base_queries": len(set(base)), "systems": {}}
    rng = random.Random(0)
    for f in sorted(PRED.glob("*.jsonl")):
        preds = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        assert len(preds) == len(rows), f
        by_q: dict[str, list[int]] = defaultdict(list)
        fam: dict[str, list[int]] = defaultdict(list)
        for i, p in enumerate(preds):
            assert p["id"] == i and p["package"] == rows[i]["package"], (f, i)
            by_q[base[i]].append(int(p["is_correct"]))
            fam[pkg_family[i]].append(int(p["is_correct"]))
        k, n = sum(sum(v) for v in by_q.values()), len(preds)
        groups = list(by_q.values())
        stats = []
        for _ in range(a.boot):
            s = [groups[rng.randrange(len(groups))] for _ in groups]
            stats.append(sum(map(sum, s)) / sum(map(len, s)))
        stats.sort()
        lo, hi = stats[int(0.025 * a.boot)], stats[int(0.975 * a.boot) - 1]
        q_mean = sum(sum(g) / len(g) for g in groups) / len(groups)
        out["systems"][f.stem] = {
            "row_acc": k / n, "row_k": k, "row_n": n,
            "wilson95": wilson(k, n), "cluster_boot95": (lo, hi),
            "per_query_macro_acc": q_mean,
            "by_source": {s: {"k": sum(v), "n": len(v), "acc": sum(v) / len(v), "wilson95": wilson(sum(v), len(v))}
                          for s, v in sorted(fam.items())},
        }
    print(f"{out['n_rows']} rows, {out['n_base_queries']} distinct base queries\n")
    print(f"{'system':16} {'row acc':>16} {'Wilson 95% (rows)':>20} {'cluster 95% (queries)':>24} {'macro/query':>12}")
    for s, v in out["systems"].items():
        print(f"{s:16} {v['row_acc']:7.2%} ({v['row_k']}/{v['row_n']}) "
              f"[{v['wilson95'][0]:.1%}, {v['wilson95'][1]:.1%}]   "
              f"[{v['cluster_boot95'][0]:.1%}, {v['cluster_boot95'][1]:.1%}]       {v['per_query_macro_acc']:.1%}")
    print("\nby source family (soro_0.8b / soro_4b):")
    for s in ("soro_0.8b", "soro_4b"):
        if s in out["systems"]:
            for fam_name, v in out["systems"][s]["by_source"].items():
                print(f"  {s:10} {fam_name:32} {v['acc']:6.1%} ({v['k']}/{v['n']}) Wilson [{v['wilson95'][0]:.1%}, {v['wilson95'][1]:.1%}]")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
