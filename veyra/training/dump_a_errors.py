"""Dump pointer-A selection errors on Omni build_v2 eval files.

Uses the released Soro-Kev-4B checkpoint (query-first pointer). The 1k
Qwen3-0.6B A run did not save weights.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "veyra" / "python" / "src"))
sys.path.insert(0, str(ROOT / "veyra" / "training"))

from p0_upp_ablation import read_jsonl  # noqa: E402
from soro.routing.kev_local import LocalKevScorer  # noqa: E402

CKPT = ROOT / "veyra" / "runs" / "soro_kev_qwen3_4b_best"
DATA = ROOT / "veyra" / "runs" / "omni_data"
OUT = ROOT / "veyra" / "runs" / "s0_a_errors"


def bucket(gold: str, pred: str, criteria: dict, tags: list[str], p_gold: float, p_pred: float) -> str:
    if gold == "__abstain__" and pred != "__abstain__":
        return "false_call"
    if gold != "__abstain__" and pred == "__abstain__":
        return "over_abstain"
    if "near_miss" in tags or "near-miss" in tags:
        return "near_miss"
    if "collision" in tags:
        return "collision"
    gtoks = set(gold.lower().replace("-", "_").split("_"))
    ptoks = set(pred.lower().replace("-", "_").split("_"))
    if len(gtoks & ptoks) >= 1 and gold != pred:
        return "name_overlap_swap"
    if p_pred - p_gold >= 0.3:
        return "confident_wrong"
    if p_gold < 0.15:
        return "gold_buried"
    if len(criteria) >= 16:
        return "wide_menu"
    return "other_swap"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--data-root", default=str(DATA))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "test": data_root / "build_v2" / "test.jsonl" if (data_root / "build_v2" / "test.jsonl").exists() else data_root / "test.jsonl",
        "selection": data_root / "build_v2" / "selection.jsonl" if (data_root / "build_v2" / "selection.jsonl").exists() else data_root / "selection.jsonl",
        "eval_zeroshot": data_root / "stage_text" / "eval_zeroshot.jsonl" if (data_root / "stage_text" / "eval_zeroshot.jsonl").exists() else data_root / "eval_zeroshot.jsonl",
    }
    scorer = LocalKevScorer(run=args.ckpt, backend="neural", device="cuda")
    if not scorer.load():
        raise SystemExit(scorer.error)
    model = scorer._neural_model
    model.inference_max_seq_len = 768
    all_err = []
    summary = {}
    for split, path in files.items():
        if not path.exists():
            print("missing", path, flush=True)
            continue
        rows = read_jsonl(path)
        n = hit = 0
        buckets: Counter[str] = Counter()
        err_path = out_dir / f"errors_{split}.jsonl"
        with err_path.open("w", encoding="utf-8", buffering=1) as out:
            for i_row, r in enumerate(rows, start=1):
                q = r["questions"]["tool_route"]
                crit = q["criteria"]
                gold = q["label"]
                import torch
                if len(crit) > 16:
                    gold_txt = crit.get(gold, "")
                    others = [(k, v) for k, v in crit.items() if k != gold][:15]
                    crit = dict(others)
                    if gold in q["criteria"]:
                        crit[gold] = gold_txt
                try:
                    logits, keys = model.score_choice_sequence(
                        r["state"], q.get("instructions") or "Which tool?", crit, max_seq_len=768
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"{split} {i_row}/{len(rows)} OOM skip", flush=True)
                    continue
                if logits.numel() == 0 or not keys:
                    continue
                probs = torch.softmax(logits.float(), dim=-1)
                pred = keys[int(torch.argmax(probs).item())]
                pmap = {k: float(probs[i]) for i, k in enumerate(keys)}
                n += 1
                if i_row == 1 or i_row % 25 == 0:
                    print(f"{split} {i_row}/{len(rows)} acc={hit/max(n,1):.3f} errors={n-hit}", flush=True)
                if i_row % 50 == 0:
                    torch.cuda.empty_cache()
                ok = pred == gold
                hit += int(ok)
                if ok:
                    continue
                tags = r.get("tags") or []
                b = bucket(gold, pred, crit, tags, pmap.get(gold, 0.0), pmap.get(pred, 0.0))
                buckets[b] += 1
                rec = {
                    "id": r.get("id"),
                    "split": split,
                    "gold": gold,
                    "pred": pred,
                    "p_gold": round(pmap.get(gold, 0.0), 4),
                    "p_pred": round(pmap.get(pred, 0.0), 4),
                    "menu_size": len(crit),
                    "bucket": b,
                    "state": (r.get("state") or "")[:400],
                    "tags": tags,
                    "source": r.get("source"),
                    "package": r.get("package"),
                }
                out.write(json.dumps(rec) + "\n")
                out.flush()
                all_err.append(rec)
        summary[split] = {"n": n, "acc": round(hit / max(n, 1), 4), "errors": n - hit, "buckets": dict(buckets)}
        print(split, summary[split], flush=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "errors_all.jsonl").write_text("\n".join(json.dumps(r) for r in all_err) + ("\n" if all_err else ""), encoding="utf-8")
    print("total_errors", len(all_err), "wrote", out_dir, flush=True)


if __name__ == "__main__":
    main()
