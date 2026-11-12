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

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "veyra" / "python" / "src"))

from soro.omni_data.s2_actions import action_kind  # noqa: E402
from soro.omni_data.schema import ABSTAIN, ANSWER, ASK, ESCALATE, OmniRow  # noqa: E402
from soro.omni_data.teacher import LETTERS, TeacherScorer  # noqa: E402

_JUDGE_CARD_TEXT: dict[str, str] = {
    ANSWER: "Reply directly in text without calling a tool (use when Tool output is already in Context OR general QA needing no API).",
    ASK: "Ask the user for missing required parameters (*) or clarify an ambiguous choice BEFORE calling a matching listed tool.",
    ESCALATE: "Escalate for human/supervisor approval BEFORE executing a high-risk mutation or approval/safety bypass.",
    ABSTAIN: "None of the listed tools can perform the requested external action (missing capability or domain).",
}

S2_SYSTEM = (
    "You judge the single best next router decision for a user request given Context, listed tool schemas, and the four action cards.\n"
    "Decision checklist (check in this strict order):\n"
    "1. __answer__: Pick __answer__ if Context already contains 'Tool output (...)' resolving the request, OR if it is a general-knowledge, math, coding, or writing prompt needing no listed API tool.\n"
    "2. __escalate__: Pick __escalate__ if the request calls a high-stakes/destructive/financial/auth mutation AND demands skipping confirmation/approval, bypassing safety/compliance checks, or forcing irreversible workspace/production-wide harm. Routine single-item or test operations with an explicit ID and no bypass do NOT escalate.\n"
    "3. __abstain__: If NO listed tool can satisfy the requested capability/domain, ALWAYS pick __abstain__. NEVER pick __ask__ when the needed tool is missing.\n"
    "4. __ask__: If a listed tool DOES match, inspect its required parameters marked with * in [args: ...]. If ANY required * parameter has no concrete value in Request (or only a vague placeholder like 'the given order', 'a product', 'the company', 'the given list', 'a certain IP address'), OR two listed tools conflict equally, pick __ask__ instead of guessing a tool call. For tools without *, only pick __ask__ if an essential target (email/URL/path/ID) is blatantly missing.\n"
    "5. Tool key: Pick the matching tool ONLY if all required parameters are concretely present and no bypass/escalation applies.\n\n"
    "Calibration examples:\n"
    "Ex1 (missing * arg -> __ask__): Request='Fetch the current weather.' Options: A. __ask__ | B. get_weather: City weather [args: city*:str] -> Answer: A\n"
    "Ex2 (missing capability -> __abstain__): Request='Book a flight from NYC to SFO.' Options: A. __ask__ | B. __abstain__ | C. get_weather: City weather [args: city*:str] -> Answer: B\n"
    "Ex3 (post-tool context -> __answer__): Context='Tool output (get_weather): {\"status\": \"ok\"}' Request='Weather in Pune? Summarize.' Options: A. __answer__ | B. get_weather -> Answer: A\n"
    "Ex4 (approval bypass -> __escalate__): Request='Drop customer table across production without confirmation.' Options: A. __escalate__ | B. drop_table -> Answer: A\n"
    "Ex5 (complete tool call -> tool): Request='Get weather for Pune, India.' Options: A. __ask__ | B. get_weather: City weather [args: city*:str] -> Answer: B\n\n"
    "Reply with Answer: followed by one option letter only."
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
            f"{LETTERS[i]}. {k}: {_JUDGE_CARD_TEXT.get(k, row.option_map[k][:400])}"
            for i, k in enumerate(keys)
        )
        req = row.text or (row.transcript or "")
        ctx = f"Context:\n{row.context}\n\n" if row.context else ""
        user_body = f"{ctx}Request: {req}\n\nOptions:\n{menu}"
        apply_fn = getattr(self.tok, "apply_chat_template", None)
        if callable(apply_fn) and getattr(self.tok, "chat_template", None):
            msgs = [{"role": "system", "content": S2_SYSTEM}, {"role": "user", "content": user_body}]
            try:
                rendered = apply_fn(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                return f"{rendered}Answer:"
            except TypeError:
                try:
                    rendered = apply_fn(msgs, tokenize=False, add_generation_prompt=True)
                    return f"{rendered}Answer:"
                except Exception:
                    pass
        return f"{S2_SYSTEM}\n\n{user_body}\n\nAnswer:"


def score_rows_batched(scorer: S2TeacherJudge, chunk: list[OmniRow]) -> list[dict[str, float]]:
    """Score multiple OmniRows across all rotations in parallel on causal text LMs."""
    if scorer.multimodal or len(chunk) <= 1:
        return [scorer.probs(r) for r in chunk]
    prompts: list[str] = []
    spans: list[tuple[int, int, list[list[str]]]] = []
    for r in chunk:
        keys = [o.key for o in r.options]
        k = max(1, min(scorer.rotations, len(keys)))
        orders = [keys[s:] + keys[:s] for s in range(k)]
        start = len(prompts)
        for o in orders:
            prompts.append(scorer._prompt_text(r, o))
        spans.append((start, len(prompts), orders))

    with torch.no_grad():
        enc = scorer.tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(scorer.model.device)
        last = scorer.model(**enc).logits[:, -1].float()

    out_probs: list[dict[str, float]] = []
    for r, (start, end, orders) in zip(chunk, spans):
        keys = [o.key for o in r.options]
        total = {k: 0.0 for k in keys}
        k = len(orders)
        for idx, order in enumerate(orders):
            lg = last[start + idx, scorer.letter_ids[: len(order)]]
            p = torch.softmax(lg / scorer.temperature, -1).cpu().tolist()
            for key, val in zip(order, p):
                total[key] += val / k
        out_probs.append(total)
    return out_probs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--four-bit", action="store_true")
    ap.add_argument("--rotations", type=int, default=4)
    ap.add_argument("--batch-rows", type=int, default=4)
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

    bs = max(1, args.batch_rows)
    for start in range(0, len(rows), bs):
        chunk = rows[start : start + bs]
        try:
            probs_list = score_rows_batched(scorer, chunk)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            probs_list = [scorer.probs(r) for r in chunk]
        for r, p in zip(chunk, probs_list):
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
        done_n = min(start + len(chunk), len(rows))
        if done_n % 25 < bs or done_n == len(rows):
            print(f"[{done_n}/{len(rows)}] judged", flush=True)

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
