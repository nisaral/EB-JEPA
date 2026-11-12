"""Judge sampled S2 action rows with an open teacher model and report per-action agreement.

Evaluates 50 rows per action class (tool / ask / abstain / escalate / answer) using
position-debiased TeacherScorer logits over the actual S2 menu (including the four
S2 action cards), then reports exact label agreement and action-kind agreement with
95% Wilson CIs and a confusion matrix.

Usage:
    python veyra/training/judge_s2_sample.py --in sample_250.jsonl --model Qwen/Qwen3-8B --four-bit --out out_dir
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "veyra" / "python" / "src"))

from soro.omni_data.s2_actions import action_kind  # noqa: E402
from soro.omni_data.schema import ABSTAIN, ANSWER, ASK, ESCALATE, OmniRow  # noqa: E402
from soro.omni_data.teacher import LETTERS, TeacherScorer  # noqa: E402

S2_SYSTEM = (
    "You judge the single best next router decision for a user request given the Context, "
    "available tool schemas, and four action cards:\n"
    "- __answer__: Reply directly without calling any tool (when tool output is already provided in Context so the request is resolved, OR it is a general-knowledge, math, writing, or coding question needing no listed API tool).\n"
    "- __ask__: Ask the user a clarifying question BEFORE calling any tool. Pick __ask__ when a matching tool IS listed AND either (a) a required parameter marked with * (like coord*, domain*, symbol*, is_id*, city*, date*) is missing or replaced by a vague placeholder (like 'the website', 'the topic', 'a certain word', 'the given image', 'the creator', 'the strain'), OR (b) two or more listed tools match equally and the user's intended operation is ambiguous. Never pick __ask__ if NO listed tool matches — pick __abstain__ instead.\n"
    "- __escalate__: Escalate to human/supervisor confirmation BEFORE acting (when the request asks for a high-risk/destructive mutation or approval-gate bypass).\n"
    "- none of these / __abstain__: No listed tool fits the requested external capability and none of the above apply.\n"
    "- Or pick the single matching tool key if one listed tool clearly fits the request and its required (*) parameters are provided without unapproved high-risk flags (for schemas without * markers, only choose __ask__ if an essential target like email/URL/file path/ID is blatantly missing).\n"
    "Answer with one option letter only."
)


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)
    return (max(0.0, center - half), min(1.0, center + half))


class S2TeacherJudge(TeacherScorer):
    def _prompt_text(self, row: OmniRow, keys: list[str]) -> str:
        menu = "\n".join(
            f"{LETTERS[i]}. {k}: {row.option_map[k][:400]}"
            for i, k in enumerate(keys)
        )
        req = row.text or (row.transcript or "")
        ctx = f"Context:\n{row.context}\n\n" if row.context else ""
        return f"{S2_SYSTEM}\n\n{ctx}Request: {req}\n\nOptions:\n{menu}\n\nAnswer:"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--four-bit", action="store_true")
    ap.add_argument("--rotations", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [OmniRow.from_json(json.loads(l)) for l in Path(args.in_path).open(encoding="utf-8") if l.strip()]
    scorer = S2TeacherJudge(args.model, load_in_4bit=args.four_bit, rotations=args.rotations)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    judged: list[dict[str, Any]] = []
    by_act: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "exact_ok": 0, "kind_ok": 0})
    conf = Counter()
    mismatches: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for idx, r in enumerate(rows):
        p = scorer.probs(r)
        top = max(p, key=p.get)
        gk = action_kind(r.label)
        pk = action_kind(top)
        exact_ok = int(top == r.label)
        kind_ok = int(pk == gk)

        by_act[gk]["n"] += 1
        by_act[gk]["exact_ok"] += exact_ok
        by_act[gk]["kind_ok"] += kind_ok
        conf[(gk, pk)] += 1

        rec = {
            "id": r.id,
            "source": r.source,
            "tags": r.tags,
            "gold": r.label,
            "gold_kind": gk,
            "teacher_top": top,
            "teacher_kind": pk,
            "exact_ok": exact_ok,
            "kind_ok": kind_ok,
            "gold_prob": round(p.get(r.label, 0.0), 4),
            "top_prob": round(p[top], 4),
            "text": r.text[:240],
        }
        judged.append(rec)
        if not exact_ok and len(mismatches[gk]) < 15:
            mismatches[gk].append(rec)
        if (idx + 1) % 25 == 0 or idx + 1 == len(rows):
            print(f"[{idx+1}/{len(rows)}] judged", flush=True)

    summary = {}
    for act in ("tool", "ask", "abstain", "escalate", "answer"):
        d = by_act.get(act, {"n": 0, "exact_ok": 0, "kind_ok": 0})
        n = d["n"]
        ex = d["exact_ok"]
        kk = d["kind_ok"]
        summary[act] = {
            "n": n,
            "exact_agreement": ex / n if n else 0.0,
            "exact_ok": ex,
            "exact_ci95": list(wilson_ci(ex, n)),
            "kind_agreement": kk / n if n else 0.0,
            "kind_ok": kk,
            "kind_ci95": list(wilson_ci(kk, n)),
            "confusion_pred_kinds": {pk: conf[(act, pk)] for (_, pk) in sorted(conf) if _ == act},
        }

    (out_dir / "s2_teacher_judged.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in judged) + "\n", encoding="utf-8"
    )
    report = {"model": args.model, "rotations": args.rotations, "n": len(judged), "by_action": summary, "mismatches": dict(mismatches)}
    (out_dir / "s2_teacher_agreement.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n## S2 Teacher Agreement (" + args.model + ")")
    print("| Action | n | Exact Agr (95% CI) | Action-Kind Agr (95% CI) | Pred Kind Breakdown |")
    print("| :-- | --: | --: | --: | :-- |")
    for act, s in summary.items():
        eci, kci = s["exact_ci95"], s["kind_ci95"]
        print(
            f"| {act} | {s['n']} | {s['exact_agreement']:.3f} [{eci[0]:.3f}, {eci[1]:.3f}] ({s['exact_ok']}/{s['n']}) | "
            f"{s['kind_agreement']:.3f} [{kci[0]:.3f}, {kci[1]:.3f}] ({s['kind_ok']}/{s['n']}) | {s['confusion_pred_kinds']} |"
        )


if __name__ == "__main__":
    main()

