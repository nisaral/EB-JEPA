"""Generate position-debiased teacher soft labels (teacher.probs) for S2 training rows.

Supports 2-GPU Kaggle sharding (--shard 0/2, 1/2), resumable incremental flushing,
batched prompt inference across rotations, and merging/validation via OmniRow.validate()
+ quality.label_audit().

Usage:
    python veyra/training/label_s2_teacher.py --in s2_train.jsonl --model Qwen/Qwen3-8B --four-bit --rotations 4 --shard 0/2 --out out_dir
    python veyra/training/label_s2_teacher.py --merge s2_teacher_shard_0.jsonl,s2_teacher_shard_1.jsonl --out s2_train_teacher.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "veyra" / "python" / "src"))
sys.path.insert(0, str(REPO / "veyra" / "training"))

from collections import Counter

from judge_s2_sample import S2TeacherJudge, score_rows_batched  # noqa: E402
from soro.omni_data import quality  # noqa: E402
from soro.omni_data.s2_actions import action_kind  # noqa: E402
from soro.omni_data.schema import OmniRow, read_rows, write_rows  # noqa: E402


def attach_teacher(row: OmniRow, probs: dict[str, float], model_id: str, rotations: int, temperature: float = 1.0) -> OmniRow:
    top = max(probs, key=probs.get)
    return replace(
        row,
        teacher={
            "model": model_id,
            "probs": {k: round(float(v), 5) for k, v in probs.items()},
            "top": top,
            "agrees": top == row.label,
            "gold_prob": round(float(probs.get(row.label, 0.0)), 5),
            "temperature": temperature,
            "rotations": rotations,
        },
    )


def merge_and_validate(
    shard_paths: list[Path],
    out_path: Path,
    report_path: Path | None = None,
    *,
    filter_mode: str = "none",
    strip_probs: bool = False,
) -> dict[str, Any]:
    by_id: dict[str, OmniRow] = {}
    ordered: list[OmniRow] = []
    for sp in shard_paths:
        for r in read_rows(sp):
            if r.id in by_id:
                continue
            by_id[r.id] = r
            ordered.append(r)
    valid, bad = quality.validate_rows(ordered)
    if bad:
        raise ValueError(f"{len(bad)} invalid rows during teacher merge; first={bad[0]}")

    kept: list[OmniRow] = []
    drop_by_act: Counter[str] = Counter()
    total_by_act: Counter[str] = Counter()
    drop_by_source: Counter[str] = Counter()
    exact_agree = 0
    kind_agree = 0
    for r in valid:
        gk = action_kind(r.label)
        total_by_act[gk] += 1
        t = r.teacher or {}
        top = str(t.get("top") or "")
        ex_ok = bool(top == r.label)
        kn_ok = bool(top and action_kind(top) == gk)
        exact_agree += int(ex_ok)
        kind_agree += int(kn_ok)
        keep = True
        if filter_mode == "exact":
            keep = ex_ok
        elif filter_mode == "kind":
            keep = kn_ok
        if keep:
            if strip_probs and r.teacher is not None:
                kept.append(replace(r, teacher={k: v for k, v in r.teacher.items() if k != "probs"}))
            else:
                kept.append(r)
        else:
            drop_by_act[gk] += 1
            drop_by_source[str(r.source)] += 1

    write_rows(kept, out_path)
    audit = quality.label_audit(kept)
    n_valid = max(len(valid), 1)
    summary = {
        "total": len(kept),
        "total_scored": len(valid),
        "kept": len(kept),
        "dropped": len(valid) - len(kept),
        "filter_mode": filter_mode,
        "strip_probs": bool(strip_probs),
        "teacher_coverage": sum(1 for r in valid if r.teacher and r.teacher.get("probs")) / n_valid,
        "exact_agreement": exact_agree / n_valid,
        "kind_agreement": kind_agree / n_valid,
        "total_by_action": dict(sorted(total_by_act.items())),
        "dropped_by_action": dict(sorted(drop_by_act.items())),
        "kept_by_action": {k: total_by_act[k] - drop_by_act.get(k, 0) for k in sorted(total_by_act)},
        "dropped_by_source": dict(sorted(drop_by_source.items())),
        "label_audit": audit,
    }
    if report_path is not None:
        report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", default="", help="Input s2_train.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--four-bit", action="store_true")
    ap.add_argument("--rotations", type=int, default=4)
    ap.add_argument("--batch-rows", type=int, default=4, help="Rows scored per forward pass (total batch = batch_rows * rotations)")
    ap.add_argument("--shard", default="0/1", help="Shard idx/total, e.g. 0/2 or 1/2")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--merge", default="", help="Comma-separated shard JSONLs to merge & validate")
    ap.add_argument("--filter-mode", choices=("none", "exact", "kind"), default="none", help="Filter disagreed rows during --merge")
    ap.add_argument("--strip-probs", action="store_true", help="Drop teacher.probs when soft-label +3pt gate did not pass")
    ap.add_argument("--out", required=True, help="Output dir (sharding) or output JSONL (--merge)")
    args = ap.parse_args()

    if args.merge:
        shards = [Path(p.strip()) for p in args.merge.split(",") if p.strip()]
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        rep = merge_and_validate(
            shards,
            out_p,
            out_p.with_suffix(".audit.json"),
            filter_mode=args.filter_mode,
            strip_probs=args.strip_probs,
        )
        print(json.dumps(rep, indent=2))
        return

    if not args.in_path:
        raise ValueError("--in is required unless --merge is set")

    s_idx_s, s_tot_s = args.shard.split("/")
    shard_idx, shard_total = int(s_idx_s), int(s_tot_s)
    all_rows = list(read_rows(args.in_path))
    if args.max_rows > 0:
        all_rows = all_rows[: args.max_rows]
    rows = [r for i, r in enumerate(all_rows) if i % shard_total == shard_idx]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"s2_teacher_shard_{shard_idx}.jsonl"

    done_ids: set[str] = set()
    if out_jsonl.exists():
        for r in read_rows(out_jsonl):
            if r.teacher and r.teacher.get("probs"):
                done_ids.add(r.id)
    todo = [r for r in rows if r.id not in done_ids]
    print(f"Shard {shard_idx}/{shard_total}: total={len(rows)} resumed={len(done_ids)} todo={len(todo)}", flush=True)
    if not todo:
        return

    scorer = S2TeacherJudge(args.model, load_in_4bit=args.four_bit, rotations=args.rotations)
    t0 = time.time()
    written = 0
    bs = max(1, args.batch_rows)
    with out_jsonl.open("a", encoding="utf-8") as f:
        for i in range(0, len(todo), bs):
            chunk = todo[i : i + bs]
            try:
                probs_list = score_rows_batched(scorer, chunk)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                probs_list = [scorer.probs(r) for r in chunk]
            for r, probs in zip(chunk, probs_list):
                labeled = attach_teacher(r, probs, args.model, args.rotations, scorer.temperature)
                f.write(json.dumps(labeled.to_json(), ensure_ascii=False) + "\n")
                written += 1
            if (i // bs + 1) % 10 == 0 or i + len(chunk) >= len(todo):
                f.flush()
                rate = written / max(time.time() - t0, 1e-6)
                print(f"  [{written}/{len(todo)}] ({rate:.2f} rows/s)", flush=True)


if __name__ == "__main__":
    main()

