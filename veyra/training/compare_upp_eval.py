"""Paired McNemar comparison across two full UPP evaluations on identical rows.

Reads ``scored_rows.json`` (or ``scored_shard_*.json`` inside an eval output directory)
and computes paired accuracy delta, exact two-sided McNemar binomial p-value,
continuity-corrected chi-square statistic, and 5-order flip rates overall and per split
(with explicit focus on ``collisions`` n=421).

Usage:
    python veyra/training/compare_upp_eval.py --a v2=path/to/v2_dir --b v3=path/to/v3_dir
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_scored_rows(spec: str) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    if "=" in spec:
        name, path_s = spec.split("=", 1)
    else:
        path_s = spec
        name = Path(path_s).name
    p = Path(path_s.strip())
    if p.is_dir():
        sr = p / "scored_rows.json"
        if sr.exists():
            blob = json.loads(sr.read_text(encoding="utf-8"))
            return name.strip(), blob.get("model", {}), list(blob["rows"])
        shards = sorted(p.glob("scored_shard_*.json"))
        if not shards:
            raise FileNotFoundError(f"No scored_rows.json or scored_shard_*.json under {p}")
        rows: list[dict[str, Any]] = []
        meta: dict[str, Any] = {}
        for sp in shards:
            blob = json.loads(sp.read_text(encoding="utf-8"))
            meta = blob.get("model", meta)
            rows.extend(blob["rows"])
        return name.strip(), meta, rows
    blob = json.loads(p.read_text(encoding="utf-8"))
    return name.strip(), blob.get("model", {}), list(blob["rows"])


def mcnemar_exact(n10: int, n01: int) -> dict[str, float]:
    """Exact two-sided binomial McNemar p-value + mid-p + Yates chi2.

    ``n10`` = A right, B wrong; ``n01`` = A wrong, B right.
    """
    m = n10 + n01
    if m == 0:
        return {"discordant": 0, "p_exact": 1.0, "p_mid": 1.0, "chi2_cc": 0.0}
    k = min(n10, n01)
    log_denom = m * math.log(2.0)
    tail = sum(math.exp(math.lgamma(m + 1) - math.lgamma(i + 1) - math.lgamma(m - i + 1) - log_denom) for i in range(k + 1))
    pk = math.exp(math.lgamma(m + 1) - math.lgamma(k + 1) - math.lgamma(m - k + 1) - log_denom)
    p_exact = min(1.0, 2.0 * tail)
    p_mid = min(1.0, max(0.0, 2.0 * (tail - 0.5 * pk)))
    chi2_cc = (max(0.0, abs(n10 - n01) - 1.0) ** 2) / m
    return {"discordant": m, "p_exact": p_exact, "p_mid": p_mid, "chi2_cc": chi2_cc}


def compare_pair(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    *,
    exclude_splits: tuple[str, ...] = (),
) -> dict[str, Any]:
    map_a = {(r["split"], r["orig_idx"], r["id"]): r for r in rows_a}
    map_b = {(r["split"], r["orig_idx"], r["id"]): r for r in rows_b}
    if set(map_a) != set(map_b):
        only_a = len(set(map_a) - set(map_b))
        only_b = len(set(map_b) - set(map_a))
        raise ValueError(f"Mismatched scored row keys between A and B (only_a={only_a}, only_b={only_b})")

    keys = sorted(map_a)
    by_split: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    all_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    tool_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for k in keys:
        ra, rb = map_a[k], map_b[k]
        if ra["split"] in exclude_splits:
            continue
        by_split.setdefault(ra["split"], []).append((ra, rb))
        all_pairs.append((ra, rb))
        if ra["split"] != "near_miss":
            tool_pairs.append((ra, rb))

    def _stats(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
        n = len(pairs)
        acc_a = sum(int(ra["ok"]) for ra, _ in pairs) / n
        acc_b = sum(int(rb["ok"]) for _, rb in pairs) / n
        flips_a = sum(int(ra["flipped"]) for ra, _ in pairs)
        flips_b = sum(int(rb["flipped"]) for _, rb in pairs)
        n11 = sum(1 for ra, rb in pairs if ra["ok"] and rb["ok"])
        n10 = sum(1 for ra, rb in pairs if ra["ok"] and not rb["ok"])
        n01 = sum(1 for ra, rb in pairs if not ra["ok"] and rb["ok"])
        n00 = sum(1 for ra, rb in pairs if not ra["ok"] and not rb["ok"])
        mc = mcnemar_exact(n10, n01)
        return {
            "n": n,
            "acc_a": acc_a,
            "acc_b": acc_b,
            "delta_pts": (acc_b - acc_a) * 100.0,
            "flips_a": flips_a,
            "flip_rate_a": flips_a / n,
            "flips_b": flips_b,
            "flip_rate_b": flips_b / n,
            "n11": n11,
            "n10_a_only": n10,
            "n01_b_only": n01,
            "n00": n00,
            **mc,
        }

    out: dict[str, Any] = {}
    if tool_pairs and len(tool_pairs) != len(all_pairs):
        out["overall_excl_nearmiss"] = _stats(tool_pairs)
    out["overall_all"] = _stats(all_pairs)
    for sp, plist in by_split.items():
        out[sp] = _stats(plist)
    return out


def format_compare_md(name_a: str, name_b: str, stats: dict[str, Any]) -> str:
    lines = [
        f"## Paired McNemar Comparison: `{name_a}` (A) vs `{name_b}` (B)",
        "",
        "| Split | n | Acc A | Acc B | Δ (B-A) pts | n10 (A only) | n01 (B only) | McNemar exact p | Flips A | Flips B |",
        "| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |",
    ]
    for split, d in stats.items():
        lines.append(
            f"| {split} | {d['n']} | {d['acc_a']:.4f} | {d['acc_b']:.4f} | {d['delta_pts']:+.2f} | "
            f"{d['n10_a_only']} | {d['n01_b_only']} | {d['p_exact']:.4g} | "
            f"{d['flip_rate_a']:.4f} ({d['flips_a']}) | {d['flip_rate_b']:.4f} ({d['flips_b']}) |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, help="name=path (dir, scored_rows.json)")
    ap.add_argument("--b", required=True, help="name=path (dir, scored_rows.json)")
    ap.add_argument("--out", default="", help="Optional output directory to write json/md report")
    args = ap.parse_args()

    name_a, meta_a, rows_a = load_scored_rows(args.a)
    name_b, meta_b, rows_b = load_scored_rows(args.b)
    stats = compare_pair(rows_a, rows_b)
    md = format_compare_md(name_a, name_b, stats)
    print(md)
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {"a": {"name": name_a, "model": meta_a}, "b": {"name": name_b, "model": meta_b}, "comparison": stats}
        (out_dir / f"compare_{name_a}_vs_{name_b}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (out_dir / f"compare_{name_a}_vs_{name_b}.md").write_text(md + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

