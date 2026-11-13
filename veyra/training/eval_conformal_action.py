"""G2 & G3 Evaluator: Option-Count Calibration (6-20 opts), Split/Mondrian Conformal Sets, and Set-as-Action.

Evaluates any scored UPP / S2 JSON (e.g., ``s2_baseline_scored_rows.json`` or ``eval_upp_full.py``
outputs) along two axes matching ``docs/NORTH_STAR.md``:

  G2. Coverage & Multi-Option Calibration (Drex Gap):
      - Accuracy, ECE (15-bin), NLL, Brier, and Flip Rate bucketed by option count
        ``K <= 5``, ``6 <= K <= 20`` (Drex Table 19 target), and ``K > 20``.
      - Split conformal sets (LAC, optional ``u``-normalized, and action-Mondrian) calibrated
        strictly on ``--cal-split`` (default ``s2_selection``) and evaluated out-of-sample, plus
        within-split 50/50 finite-sample split conformal (matching per-modality/domain guarantees).

  G3. Set-as-Action vs Learned Head:
      - Derives action purely from the plausible real-tool conformal set ``S_tool``:
            |S_tool| == 0  ->  __abstain__
            |S_tool| == 1  ->  act (the unique plausible tool)
            |S_tool| >= 2  ->  __ask__ (if available, else __abstain__)
      - Compares pure conformal quantile ``q_tool`` (no accuracy tuning, fixed ``alpha``) and
        ``s2_selection``-tuned ``q_tool`` against raw argmax (4-class head) via paired McNemar,
        including ``When2Call MCQ``, ``AgentAbstain`` (``missing_critical_parameter``),
        ``external_s2``, ``collisions_s2``, and ``near_miss_s2``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "veyra" / "python" / "src"))
sys.path.insert(0, str(REPO / "veyra" / "training"))

from compare_upp_eval import mcnemar_exact  # noqa: E402
from eval_upp_full import wilson_ci  # noqa: E402
from p0_upp_ablation import conformal as in_split_conformal, ece  # noqa: E402
from soro.omni_data.s2_actions import action_kind  # noqa: E402
from soro.omni_data.schema import ABSTAIN, ANSWER, ASK, SPECIAL  # noqa: E402


def k_bucket(k: int) -> str:
    if k <= 5:
        return "k_le_5"
    if k <= 20:
        return "k_6_20"
    return "k_gt_20"


def lac_quantile(scores: list[float], alpha: float) -> float:
    """Exact finite-sample split-conformal quantile ceil((n+1)*(1-alpha)) / n."""
    if not scores:
        return 1.0
    s = sorted(float(x) for x in scores)
    n = len(s)
    rank = min(n, max(1, math.ceil((n + 1) * (1.0 - alpha))))
    return float(s[rank - 1])


def summarize_bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    hits = [int(r["ok"]) for r in rows]
    acc_k = sum(hits)
    flips_k = sum(int(r.get("flipped", 0)) for r in rows)
    confs = [max(r["probs"].values()) for r in rows]
    ks = [len(r["probs"]) for r in rows]
    nll = sum(-math.log(max(float(r["probs"].get(r["gold"], 0.0)), 1e-9)) for r in rows) / n
    brier = (
        sum(
            sum((float(p) - (1.0 if opt == r["gold"] else 0.0)) ** 2 for opt, p in r["probs"].items()) / max(1, len(r["probs"]))
            for r in rows
        )
        / n
    )
    return {
        "n": n,
        "mean_k": sum(ks) / n,
        "acc": acc_k / n,
        "acc_k": acc_k,
        "acc_ci95": list(wilson_ci(acc_k, n)),
        "ece_15": ece(confs, hits, bins=15),
        "nll": nll,
        "brier": brier,
        "flip_rate": flips_k / n,
        "flips_k": flips_k,
    }


def _evaluate_set_thresholds(
    rows: list[dict[str, Any]],
    q_by_row: list[dict[str, float]],
    *,
    norm_u: bool = False,
) -> dict[str, Any]:
    """Evaluate conformal sets S = {k : (1 - p(k))/scale <= q_k} (allows |S|=0 without forced fallback)."""
    n = len(rows)
    if n == 0:
        return {"n": 0}
    cov_k = empty_k = single_k = multi_k = 0
    sizes: list[int] = []
    by_act: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for r, q_map in zip(rows, q_by_row):
        scale = math.sqrt(max(float(r.get("u", 1.0)), 1e-6)) if (norm_u and math.isfinite(float(r.get("u", float("nan"))))) else 1.0
        s = [k for k, p in r["probs"].items() if ((1.0 - float(p)) / scale) <= q_map[k]]
        covered = int(r["gold"] in s)
        sz = len(s)
        cov_k += covered
        sizes.append(sz)
        empty_k += int(sz == 0)
        single_k += int(sz == 1)
        multi_k += int(sz >= 2)
        gk = r.get("gold_kind") or action_kind(r["gold"])
        by_act[gk].append((covered, sz))

    return {
        "n": n,
        "coverage": cov_k / n,
        "cov_k": cov_k,
        "coverage_ci95": list(wilson_ci(cov_k, n)),
        "mean_set_size": sum(sizes) / n,
        "median_set_size": statistics.median(sizes),
        "empty_rate": empty_k / n,
        "singleton_rate": single_k / n,
        "multi_rate": multi_k / n,
        "by_action": {
            gk: {
                "n": len(vals),
                "coverage": sum(c for c, _ in vals) / len(vals),
                "mean_set_size": sum(sz for _, sz in vals) / len(vals),
            }
            for gk, vals in sorted(by_act.items())
        },
    }


def calibrate_and_eval_conformal(
    rows: list[dict[str, Any]],
    *,
    cal_split: str = "s2_selection",
    alphas: tuple[float, ...] = (0.05, 0.10),
    exclude_splits: tuple[str, ...] = ("escalate_probe",),
) -> dict[str, Any]:
    """Run G2 conformal analysis: fixed-split transfer from ``cal_split`` + in-split 50/50 split conformal."""
    cal_rows = [r for r in rows if r["split"] == cal_split]
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r["split"] in exclude_splits:
            continue
        by_split[r["split"]].append(r)

    eval_pooled = [r for sp, rs in by_split.items() if sp != cal_split for r in rs]
    eval_6_20 = [r for r in eval_pooled if 6 <= len(r["probs"]) <= 20]

    out: dict[str, Any] = {"cal_split": cal_split, "n_cal": len(cal_rows), "by_alpha": {}}
    if not cal_rows:
        return out

    has_u = all(math.isfinite(float(r.get("u", float("nan")))) for r in cal_rows)
    cal_scores = [1.0 - float(r["probs"].get(r["gold"], 0.0)) for r in cal_rows]
    cal_by_act: dict[str, list[float]] = defaultdict(list)
    for r, sc in zip(cal_rows, cal_scores):
        cal_by_act[r.get("gold_kind") or action_kind(r["gold"])].append(sc)

    for alpha in alphas:
        q_global = lac_quantile(cal_scores, alpha)
        q_mondrian = {
            gk: (lac_quantile(scs, alpha) if len(scs) >= 5 else q_global)
            for gk, scs in cal_by_act.items()
        }

        targets: dict[str, list[dict[str, Any]]] = {
            **{sp: rs for sp, rs in sorted(by_split.items()) if sp != cal_split},
            "ALL_EVAL_POOLED": eval_pooled,
            "ALL_EVAL_K_6_20": eval_6_20,
        }
        split_res: dict[str, Any] = {}
        for name, target_rows in targets.items():
            if not target_rows:
                continue
            q_glob_rows = [{k: q_global for k in r["probs"]} for r in target_rows]
            q_mond_rows = [
                {k: q_mondrian.get(action_kind(k), q_global) for k in r["probs"]}
                for r in target_rows
            ]
            entry: dict[str, Any] = {
                "transfer_lac": _evaluate_set_thresholds(target_rows, q_glob_rows),
                "transfer_mondrian_action": _evaluate_set_thresholds(target_rows, q_mond_rows),
            }
            if len(target_rows) >= 20:
                entry["in_split_50_50"] = in_split_conformal(target_rows, alpha=alpha, splits=50, normalized=False)
                if has_u and all(math.isfinite(float(r.get("u", float("nan")))) for r in target_rows):
                    entry["in_split_50_50_norm_u"] = in_split_conformal(target_rows, alpha=alpha, splits=50, normalized=True)
            split_res[name] = entry

        out["by_alpha"][f"{alpha:.2f}"] = {
            "q_global": q_global,
            "q_mondrian": q_mondrian,
            "splits": split_res,
        }
    return out


def set_as_action_predict(
    probs: dict[str, float],
    *,
    q_tool: float,
    answer_thresh: float | None = None,
) -> tuple[str, list[str]]:
    """Map real-tool plausible set cardinality directly to ABSTAIN / ACT / ASK (+ optional ANSWER gate)."""
    if answer_thresh is not None and ANSWER in probs and float(probs[ANSWER]) >= answer_thresh:
        # Defer to ANSWER only if it clears threshold and beats every real tool.
        max_tool_p = max((float(p) for k, p in probs.items() if k not in SPECIAL), default=0.0)
        if float(probs[ANSWER]) >= max_tool_p:
            return ANSWER, []

    tool_probs = [(k, float(p)) for k, p in probs.items() if k not in SPECIAL]
    tool_probs.sort(key=lambda kv: -kv[1])
    s_tool = [k for k, p in tool_probs if (1.0 - p) <= q_tool]

    if len(s_tool) == 0:
        if ABSTAIN in probs:
            return ABSTAIN, s_tool
        return (ASK if ASK in probs else max(probs, key=probs.get)), s_tool
    if len(s_tool) == 1:
        return s_tool[0], s_tool
    if ASK in probs:
        return ASK, s_tool
    if ABSTAIN in probs:
        return ABSTAIN, s_tool
    return s_tool[0], s_tool


def _score_set_as_action_rows(
    rows: list[dict[str, Any]],
    *,
    q_tool: float,
    answer_thresh: float | None = None,
) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    ok_k = 0
    n10_head_only = 0
    n01_set_only = 0
    by_act: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "ok": 0})
    by_cat: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "ok": 0})
    card_counts = {"s0_abstain": 0, "s1_act": 0, "s2p_ask": 0, "answer_gate": 0}

    for r in rows:
        dec, s_tool = set_as_action_predict(r["probs"], q_tool=q_tool, answer_thresh=answer_thresh)
        is_ok = int(dec == r["gold"])
        head_ok = int(r["ok"])
        ok_k += is_ok
        if head_ok and not is_ok:
            n10_head_only += 1
        elif is_ok and not head_ok:
            n01_set_only += 1

        if dec == ANSWER and answer_thresh is not None and len(s_tool) == 0 and ANSWER in r["probs"] and float(r["probs"][ANSWER]) >= answer_thresh:
            card_counts["answer_gate"] += 1
        elif len(s_tool) == 0:
            card_counts["s0_abstain"] += 1
        elif len(s_tool) == 1:
            card_counts["s1_act"] += 1
        else:
            card_counts["s2p_ask"] += 1

        gk = r.get("gold_kind") or action_kind(r["gold"])
        by_act[gk]["n"] += 1
        by_act[gk]["ok"] += is_ok
        for c in r.get("cats") or []:
            by_cat[c]["n"] += 1
            by_cat[c]["ok"] += is_ok

    head_k = sum(int(r["ok"]) for r in rows)
    return {
        "n": n,
        "acc": ok_k / n,
        "acc_k": ok_k,
        "acc_ci95": list(wilson_ci(ok_k, n)),
        "head_acc": head_k / n,
        "head_acc_k": head_k,
        "delta_vs_head_pts": ((ok_k - head_k) / n) * 100.0,
        "mcnemar_vs_head": mcnemar_exact(n10_head_only, n01_set_only),
        "n10_head_only": n10_head_only,
        "n01_set_only": n01_set_only,
        "cardinality_counts": card_counts,
        "by_action": {
            k: {"n": d["n"], "acc": d["ok"] / d["n"], "acc_k": d["ok"], "acc_ci95": list(wilson_ci(d["ok"], d["n"]))}
            for k, d in sorted(by_act.items())
        },
        **({"by_cat": {k: {"n": d["n"], "acc": d["ok"] / d["n"], "acc_k": d["ok"]} for k, d in sorted(by_cat.items())}} if by_cat else {}),
    }


def evaluate_set_as_action(
    rows: list[dict[str, Any]],
    *,
    cal_split: str = "s2_selection",
    exclude_splits: tuple[str, ...] = ("escalate_probe",),
) -> dict[str, Any]:
    """Run G3 comparison: pure conformal tool-LAC Set-as-Action & s2_selection-tuned Set-as-Action vs raw head."""
    cal_rows = [r for r in rows if r["split"] == cal_split]
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r["split"] in exclude_splits:
            continue
        by_split[r["split"]].append(r)

    cal_tool_scores = [
        1.0 - float(r["probs"].get(r["gold"], 0.0))
        for r in cal_rows
        if (r.get("gold_kind") or action_kind(r["gold"])) == "tool"
    ]
    policies: dict[str, dict[str, Any]] = {}
    if cal_tool_scores:
        for alpha in (0.10, 0.20):
            q_tool = lac_quantile(cal_tool_scores, alpha)
            policies[f"conformal_tool_lac_a{int(alpha * 100)}"] = {"q_tool": q_tool, "answer_thresh": None}

    if cal_rows:
        # Select (q_tool, answer_thresh) strictly on s2_selection.
        best_pure = (-1.0, 0.65)
        best_ans = (-1.0, 0.65, None)
        q_grid = [round(0.30 + 0.05 * i, 2) for i in range(13)]  # 0.30 .. 0.90 (p_tool >= 0.70 .. 0.10)
        for q_cand in q_grid:
            res_p = _score_set_as_action_rows(cal_rows, q_tool=q_cand, answer_thresh=None)
            if res_p["acc"] > best_pure[0] + 1e-12:
                best_pure = (res_p["acc"], q_cand)
            for a_th in (None, 0.25, 0.35, 0.45, 0.55):
                res_a = _score_set_as_action_rows(cal_rows, q_tool=q_cand, answer_thresh=a_th)
                if res_a["acc"] > best_ans[0] + 1e-12:
                    best_ans = (res_a["acc"], q_cand, a_th)
        policies["tuned_tool_set"] = {"q_tool": best_pure[1], "answer_thresh": None, "cal_acc": best_pure[0]}
        policies["tuned_tool_set_plus_answer"] = {
            "q_tool": best_ans[1],
            "answer_thresh": best_ans[2],
            "cal_acc": best_ans[0],
        }

    eval_splits = {sp: rs for sp, rs in sorted(by_split.items())}
    eval_pooled = [r for sp, rs in by_split.items() if sp != cal_split for r in rs]
    if eval_pooled:
        eval_splits["ALL_EVAL_POOLED"] = eval_pooled
        eval_k6_20 = [r for r in eval_pooled if 6 <= len(r["probs"]) <= 20]
        if eval_k6_20:
            eval_splits["ALL_EVAL_K_6_20"] = eval_k6_20

    results: dict[str, Any] = {}
    for pname, cfg in policies.items():
        results[pname] = {
            "params": cfg,
            "splits": {
                sp: _score_set_as_action_rows(rs, q_tool=cfg["q_tool"], answer_thresh=cfg["answer_thresh"])
                for sp, rs in eval_splits.items()
            },
        }
    return results


def build_report(rows: list[dict[str, Any]], *, cal_split: str = "s2_selection") -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r["split"] == "escalate_probe":
            continue
        by_split[r["split"]].append(r)

    eval_pooled = [r for sp, rs in by_split.items() if sp != cal_split for r in rs]
    bucket_targets = {**dict(sorted(by_split.items())), "ALL_EVAL_POOLED": eval_pooled}

    by_k_bucket: dict[str, dict[str, Any]] = {}
    for sp, rs in bucket_targets.items():
        if not rs:
            continue
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rs:
            grouped[k_bucket(len(r["probs"]))].append(r)
        entry = {"all": summarize_bucket(rs)}
        for bk in ("k_le_5", "k_6_20", "k_gt_20"):
            if grouped[bk]:
                entry[bk] = summarize_bucket(grouped[bk])
        by_k_bucket[sp] = entry

    return {
        "n_rows": len(rows),
        "cal_split": cal_split,
        "g2_k_buckets": by_k_bucket,
        "g2_conformal": calibrate_and_eval_conformal(rows, cal_split=cal_split),
        "g3_set_as_action": evaluate_set_as_action(rows, cal_split=cal_split),
    }


def format_markdown(rep: dict[str, Any]) -> str:
    lines = ["# G2 & G3 Evaluation: Conformal Coverage, K in [6, 20] Calibration, and Set-as-Action", ""]
    lines.append(f"- Total rows: `{rep['n_rows']}` | Calibration split: `{rep['cal_split']}`")
    lines.extend(["", "## 1. Option-Count (`K`) Buckets & Drex `K in [6, 20]` Calibration"])
    lines.append("| Split | Bucket | n | mean K | Acc (95% CI) | ECE-15 | NLL | Flips |")
    lines.append("| :-- | :-- | --: | --: | --: | --: | --: | --: |")
    for sp, bmap in rep["g2_k_buckets"].items():
        for bk in ("all", "k_le_5", "k_6_20", "k_gt_20"):
            d = bmap.get(bk)
            if not d:
                continue
            ci = d["acc_ci95"]
            lines.append(
                f"| {sp} | {bk} | {d['n']} | {d['mean_k']:.1f} | {d['acc']:.3f} [{ci[0]:.3f},{ci[1]:.3f}] | "
                f"{d['ece_15']:.4f} | {d['nll']:.3f} | {d['flips_k']}/{d['n']} |"
            )

    lines.extend(["", "## 2. G2 Finite-Sample Conformal Sets (`cal = " + rep["cal_split"] + "`)"])
    for alpha_s, adat in rep["g2_conformal"].get("by_alpha", {}).items():
        lines.append(f"### `alpha = {alpha_s}` (`q_global = {adat['q_global']:.4f}`)")
        lines.append("| Split | Transfer LAC Cov | Transfer Mean|S| | Transfer Empty | Mondrian Cov | In-Split 50/50 Cov (|S|) |")
        lines.append("| :-- | --: | --: | --: | --: | --: |")
        for sp, sdat in adat["splits"].items():
            tl = sdat["transfer_lac"]
            tm = sdat["transfer_mondrian_action"]
            isp = sdat.get("in_split_50_50")
            isp_s = f"{isp['coverage']:.3f} ({isp['set_size']:.2f})" if isp else "-"
            lines.append(
                f"| {sp} | {tl['coverage']:.3f} | {tl['mean_set_size']:.2f} | {tl['empty_rate']:.3f} | "
                f"{tm['coverage']:.3f} | {isp_s} |"
            )

    lines.extend(["", "## 3. G3 Set-as-Action (`|S|=0 -> abstain`, `|S|=1 -> act`, `|S|>=2 -> ask`) vs Head"])
    for pname, pdat in rep.get("g3_set_as_action", {}).items():
        lines.append(f"### Policy: `{pname}` (`{pdat['params']}`)")
        lines.append("| Split | n | Head Acc | Set-Action Acc | Δ (pts) | McNemar p | `tool` | `ask` | `abstain` |")
        lines.append("| :-- | --: | --: | --: | --: | --: | --: | --: | --: |")
        for sp, sd in pdat["splits"].items():
            ba = sd["by_action"]
            def _a(k: str) -> str:
                return f"{ba[k]['acc']:.3f}" if k in ba else "-"
            lines.append(
                f"| {sp} | {sd['n']} | {sd['head_acc']:.3f} | {sd['acc']:.3f} | {sd['delta_vs_head_pts']:+.2f} | "
                f"{sd['mcnemar_vs_head']['p_exact']:.4g} | {_a('tool')} | {_a('ask')} | {_a('abstain')} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scored-json", required=True, help="JSON file containing {'rows': [...]} from eval_s2_baseline_and_gate / eval_upp_full")
    ap.add_argument("--cal-split", default="s2_selection", help="Held-out calibration split")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()

    payload = json.loads(Path(args.scored_json).read_text(encoding="utf-8"))
    rows = payload["rows"] if isinstance(payload, dict) else payload
    rep = build_report(rows, cal_split=args.cal_split)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    md = format_markdown(rep)
    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(md, encoding="utf-8")
    print(md, flush=True)


if __name__ == "__main__":
    main()

