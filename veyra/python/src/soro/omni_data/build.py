"""Assemble the Omni dataset: gate, dedupe, mix to targets, split, check, write.

    python -m soro.omni_data.build --config veyra/omni_data/omni_mix.json --out runs/omni_data

The config lists input row files (OmniRow jsonl, produced by the stage scripts)
and target counts per bucket. Buckets are (modality, kind) where kind is one of
``positive``, ``abstain``, ``need_input``, ``collision``, ``near_miss``. Splits are
group-disjoint and package-aware: every row of a base request (and every derived
row) lands in one split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from soro.omni_data import licenses, quality
from soro.omni_data.schema import ABSTAIN, NEED_INPUT, OmniRow, read_rows, write_rows


def kind_of(r: OmniRow) -> str:
    if r.label == NEED_INPUT:
        return "need_input"
    if "near_miss" in r.tags:
        return "near_miss"
    if r.label == ABSTAIN:
        return "abstain"
    if "collision" in r.tags or "twin" in r.tags:
        return "collision"
    return "positive"


def root_group(r: OmniRow) -> str:
    """Derived rows inherit their parent's group so they never cross splits."""
    g = r.group
    return g[4:] if g.startswith("gen-") else g


def assign_splits(rows: list[OmniRow], selection_frac: float = 0.04, seed: int = 0) -> None:
    """Group-disjoint selection split for checkpoint choice and q_hat (test sets are separate files)."""
    for r in rows:
        if r.split != "train":
            continue
        h = int(hashlib.sha1(f"{seed}:{root_group(r)}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        r.split = "selection" if h < selection_frac else "train"


def mix(rows: list[OmniRow], targets: dict[str, int], seed: int = 0) -> tuple[list[OmniRow], dict[str, Any]]:
    """Sample rows per bucket ``"{modality}/{kind}"`` up to the target; report shortfalls."""
    rng = random.Random(seed)
    buckets: dict[str, list[OmniRow]] = defaultdict(list)
    for r in rows:
        buckets[f"{r.modality}/{kind_of(r)}"].append(r)
    out, report = [], {}
    for key, rs in sorted(buckets.items()):
        rng.shuffle(rs)
        want = targets.get(key, targets.get(f"{key.split('/')[0]}/*", len(rs)))
        take = rs[:want]
        out += take
        report[key] = {"available": len(rs), "target": want, "taken": len(take),
                       "short_by": max(0, want - len(rs))}
    for key, want in targets.items():
        if key not in report and "*" not in key:
            report[key] = {"available": 0, "target": want, "taken": 0, "short_by": want}
    return out, report


def build(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    rows: list[OmniRow] = []
    missing = []
    for spec in config["inputs"]:
        if not Path(spec["path"]).exists():
            missing.append(spec["path"])
            continue
        for r in read_rows(spec["path"]):
            if spec.get("split"):
                r.split = spec["split"]
            rows.append(r)
    report: dict[str, Any] = {"input_rows": len(rows), "missing_inputs": missing}

    rows, bad = quality.validate_rows(rows)
    report["invalid"] = {"count": len(bad), "examples": bad[:10]}
    rows, moved = licenses.gate(rows)
    report["license_moved_to_eval"] = moved

    train = [r for r in rows if r.split == "train"]
    evals: dict[str, list[OmniRow]] = defaultdict(list)
    for r in rows:
        if r.split != "train":
            evals[r.split].append(r)

    # Unverified rows go first: a verified copy of the same row must survive dedupe.
    pending = sum("needs_teacher" in r.tags for r in train)
    if pending and config.get("require_teacher_verification", True):
        train = [r for r in train if "needs_teacher" not in r.tags]
        report["dropped_unverified"] = pending

    train, dd = quality.dedupe(train, max_per_group_label=config.get("max_per_package_label"))
    report["dedupe"] = dd
    leak = quality.leakage(train, evals)
    report["leakage_before_filter"] = {k: {kk: v[kk] for kk in ("rows", "exact_query_overlap", "near_query_overlap",
                                                                "n_shared_packages")} for k, v in leak.items()}
    leaked = {i for v in leak.values() for i in v["leaked_ids"]}
    if config.get("drop_leaked_eval_rows", True) and leaked:
        for k in evals:
            evals[k] = [r for r in evals[k] if r.id not in leaked]
        report["eval_rows_dropped_for_leakage"] = len(leaked)

    train, mix_report = mix(train, config.get("targets", {}), seed=config.get("seed", 0))
    report["mix"] = mix_report
    assign_splits(train, config.get("selection_frac", 0.04), config.get("seed", 0))
    report["label_audit"] = quality.label_audit(train)

    out_dir.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[OmniRow]] = defaultdict(list)
    for r in train:
        by_split[r.split].append(r)
    for k, v in evals.items():
        by_split[k] += v
    report["written"] = {s: write_rows(v, out_dir / f"{s}.jsonl") for s, v in by_split.items()}
    report["balance"] = {s: quality.balance(v) for s, v in by_split.items()}
    (out_dir / "build_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    rep = build(cfg, Path(args.out))
    print(json.dumps({k: rep[k] for k in ("input_rows", "written", "dedupe", "mix")}, indent=2, default=str))


if __name__ == "__main__":
    main()
