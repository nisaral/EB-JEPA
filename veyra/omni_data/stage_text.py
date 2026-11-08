"""Stage 1 (text): convert Soro rows, mine look-alike distractors and near-miss negatives.

    python veyra/omni_data/stage_text.py --out runs/omni_data/stage_text

Writes OmniRow jsonl files:
  text_base.jsonl        existing train rows (converted)
  text_collisions.jsonl  label-preserving look-alike rows (ready to train)
  text_near_miss.jsonl   abstain rows that still need teacher verification
  eval_zeroshot.jsonl    val_zeroshot as a test split
  eval_external.jsonl    external held-out as a test split (if present)
  stage_text_report.json counts, balance and examples
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "veyra" / "python" / "src"))

from soro.omni_data import quality  # noqa: E402
from soro.omni_data.schema import from_soro_record, write_rows  # noqa: E402
from soro.omni_data.text import ToolPool, make_near_misses, mine_collisions  # noqa: E402

DATA = REPO / "releases" / "soro-universal-tool-harness"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA))
    ap.add_argument("--out", default=str(REPO / "runs" / "omni_data" / "stage_text"))
    ap.add_argument("--collision-rate", type=float, default=0.35)
    ap.add_argument("--near-miss-rate", type=float, default=0.15)
    args = ap.parse_args()
    data, out = Path(args.data_dir), Path(args.out)

    base = [from_soro_record(r, "soro_train", "mixed-see-DATA_LICENSES") for r in read_jsonl(data / "train.jsonl")]
    # Template requests reused across many unrelated packages carry suspect labels:
    # keep them only if the teacher later agrees (needs_teacher), and never derive rows from them.
    generic_train = quality.generic_requests(base)
    for r in base:
        if "generic_request" in r.tags:
            r.tags.append("needs_teacher")
    clean_base = [r for r in base if "generic_request" not in r.tags]
    pool = ToolPool(base)
    collisions = mine_collisions(clean_base, pool, rate=args.collision_rate)
    near = make_near_misses(clean_base, pool, rate=args.near_miss_rate)

    evals = {}
    report_eval: dict = {}
    for name, fname in (("zeroshot", "val_zeroshot.jsonl"), ("external", "val_external_heldout.jsonl")):
        if (data / fname).exists():
            rows = [from_soro_record(r, f"soro_{name}", "mixed-see-DATA_LICENSES") for r in read_jsonl(data / fname)]
            for r in rows:
                r.split = "test"
            gen = quality.generic_requests(rows, min_packages=2)
            suspect = [r for r in rows if "generic_request" in r.tags]
            rows = [r for r in rows if "generic_request" not in r.tags]
            write_rows(suspect, out / f"eval_{name}_suspect.jsonl")
            report_eval[name] = {"kept": len(rows), "suspect_removed": len(suspect), "generic": gen["generic_requests"]}
            evals[name] = rows
            write_rows(rows, out / f"eval_{name}.jsonl")

    report = {
        "base": len(base), "tool_pool": len(pool.keys),
        "collisions": len(collisions), "near_miss_candidates": len(near),
        "generic_train": generic_train, "eval_cleaning": report_eval,
        "written": {
            "text_base": write_rows(base, out / "text_base.jsonl"),
            "text_collisions": write_rows(collisions, out / "text_collisions.jsonl"),
            "text_near_miss": write_rows(near, out / "text_near_miss.jsonl"),
            "text_generic_needs_teacher": write_rows([r for r in base if "generic_request" in r.tags],
                                                     out / "text_generic.jsonl"),
        },
        "balance_base": quality.balance(base),
        "balance_after": quality.balance(base + collisions + near),
        "leakage_base_vs_eval": {k: {kk: v[kk] for kk in ("rows", "exact_query_overlap", "near_query_overlap",
                                                          "n_shared_packages")}
                                 for k, v in quality.leakage(base, evals).items()},
        "examples": {
            "collision": [{"text": c.text[:160], "label": c.label, "lookalikes": c.meta.get("lookalikes")}
                          for c in collisions[:5]],
            "near_miss": [{"text": n.text[:160], "removed_gold": n.meta.get("removed_gold"),
                           "lookalike": n.meta.get("lookalike")} for n in near[:5]],
        },
    }
    (out / "stage_text_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("base", "tool_pool", "collisions", "near_miss_candidates",
                                             "eval_cleaning", "leakage_base_vs_eval")}, indent=2))
    print("generic train rows needing teacher:", report["generic_train"]["rows_tagged"])
    print("balance after:", json.dumps({k: report["balance_after"][k] for k in
                                        ("rows", "abstain_share", "collision_share", "near_miss_share")}))


if __name__ == "__main__":
    main()
