"""S2 Baseline (v3b + abstain threshold + escalate inference rule) & Teacher Gate Evaluator.

Runs three non-training checks in one reproducible script:
  1. S2 Baseline (no training, fp32): evaluates locked S1 checkpoint (v3b) on all S2 splits
     (s2_selection, eval_when2call_mcq, eval_agentabstain incl. missing_critical_parameter,
     eval_external_s2, eval_collisions_s2, eval_near_miss_s2, eval_escalate_probe),
     selects ``abstain_thresh`` strictly on ``s2_selection``, and evaluates the
     inference-time ``ESCALATE`` rule (``set_size * error_cost >= cost_thresh``).
  2. Gold-100 Judge Audit: runs frozen ``S2TeacherJudge`` on ``s2_gold_100.jsonl`` (100 rows)
     and reports per-action exact/kind agreement vs rule labels with 95% Wilson CIs.
  3. +3.0 pt Soft-Label Gate: compares ``S2TeacherJudge`` top-1 vs ``v3b`` top-1 on the exact
     same ``s2_selection`` tool-labeled rows (n=270); soft labels (``--teacher-kl > 0``) are
     permitted only if ``teacher_top1 - v3b_top1 >= +3.0 pts``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "veyra" / "python" / "src"))
sys.path.insert(0, str(REPO / "veyra" / "training"))

from compare_upp_eval import mcnemar_exact  # noqa: E402
from eval_upp_full import _score_split_shard, summarize_by_split, wilson_ci  # noqa: E402
from judge_s2_sample import S2TeacherJudge, score_rows_batched  # noqa: E402
from p0_upp_ablation import read_jsonl  # noqa: E402
from soro.omni_data.s2_actions import action_kind, predict_s2_decision  # noqa: E402
from soro.omni_data.schema import ABSTAIN, ESCALATE, OmniRow  # noqa: E402
from soro.routing.upp import UPPModel  # noqa: E402


DEFAULT_SPLITS = (
    ("s2_selection", "s2_selection.jsonl"),
    ("when2call_mcq", "eval_when2call_mcq.jsonl"),
    ("agentabstain", "eval_agentabstain.jsonl"),
    ("external_s2", "eval_external_s2.jsonl"),
    ("collisions_s2", "eval_collisions_s2.jsonl"),
    ("near_miss_s2", "eval_near_miss_s2.jsonl"),
    ("escalate_probe", "eval_escalate_probe.jsonl"),
)


def _load_option_texts(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            r = json.loads(line)
            rid = str(r.get("id") or idx)
            out[rid] = {o["key"]: o["text"] for o in (r.get("options") or []) if o.get("key")}
    return out


def _apply_policy(
    rows: list[dict[str, Any]],
    opt_texts_by_split: dict[str, dict[str, dict[str, str]]],
    *,
    abstain_thresh: float | None,
    q_hat: float,
    min_set_size: int,
    cost_thresh: float,
) -> dict[str, dict[str, Any]]:
    """Apply inference-time abstain threshold + ESCALATE rule over scored fp32 probabilities."""
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        sp = r["split"]
        om = opt_texts_by_split.get(sp, {}).get(str(r.get("id") or ""), {})
        dec, info = predict_s2_decision(
            r["probs"],
            om,
            q_hat=q_hat,
            min_set_size=min_set_size,
            cost_thresh=cost_thresh,
            abstain_thresh=abstain_thresh,
        )
        by_split[sp].append((r, dec, info))

    out: dict[str, dict[str, Any]] = {}
    for sp, items in by_split.items():
        n = len(items)
        if sp == "escalate_probe":
            esc_k = sum(1 for _, dec, _ in items if dec == ESCALATE)
            defer_k = sum(1 for _, dec, _ in items if dec in (ESCALATE, ABSTAIN) or dec == "__ask__")
            exec_k = sum(1 for _, dec, _ in items if action_kind(dec) == "tool")
            mean_score = sum(float(info["score"]) for _, _, info in items) / n
            out[sp] = {
                "n": n,
                "escalate_rate": esc_k / n,
                "escalate_k": esc_k,
                "escalate_ci95": list(wilson_ci(esc_k, n)),
                "defer_or_escalate_rate": defer_k / n,
                "defer_or_escalate_k": defer_k,
                "defer_ci95": list(wilson_ci(defer_k, n)),
                "tool_exec_rate": exec_k / n,
                "tool_exec_k": exec_k,
                "mean_rule_score": mean_score,
            }
            continue

        ok_k = 0
        esc_k = 0
        acts: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "ok": 0})
        cats: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "ok": 0})
        for r, dec, _ in items:
            is_ok = int(dec == r["gold"])
            ok_k += is_ok
            if dec == ESCALATE:
                esc_k += 1
            gk = r.get("gold_kind", action_kind(r["gold"]))
            acts[gk]["n"] += 1
            acts[gk]["ok"] += is_ok
            for c in r.get("cats") or []:
                cats[c]["n"] += 1
                cats[c]["ok"] += is_ok
        entry: dict[str, Any] = {
            "n": n,
            "acc": ok_k / n,
            "acc_k": ok_k,
            "acc_ci95": list(wilson_ci(ok_k, n)),
            "false_escalate_rate": esc_k / n,
            "false_escalate_k": esc_k,
            "by_action": {
                k: {"n": d["n"], "acc_k": d["ok"], "acc": d["ok"] / d["n"], "acc_ci95": list(wilson_ci(d["ok"], d["n"]))}
                for k, d in sorted(acts.items())
            },
        }
        if cats:
            entry["by_cat"] = {
                k: {"n": d["n"], "acc_k": d["ok"], "acc": d["ok"] / d["n"], "acc_ci95": list(wilson_ci(d["ok"], d["n"]))}
                for k, d in sorted(cats.items())
            }
        out[sp] = entry
    return out


def _select_thresholds(
    scored_rows: list[dict[str, Any]],
    opt_texts_by_split: dict[str, dict[str, dict[str, str]]],
) -> dict[str, Any]:
    """Select abstain_thresh strictly on s2_selection; inspect ESCALATE operating point."""
    sel_only = [r for r in scored_rows if r["split"] == "s2_selection"]
    probe_only = [r for r in scored_rows if r["split"] == "escalate_probe"]
    candidates = [None, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    sweep_abstain = []
    best_tau = None
    best_acc = -1.0
    for tau in candidates:
        res = _apply_policy(
            sel_only, opt_texts_by_split, abstain_thresh=tau, q_hat=0.85, min_set_size=2, cost_thresh=999.0
        )["s2_selection"]
        sweep_abstain.append({"abstain_thresh": tau, "acc": res["acc"], "acc_k": res["acc_k"], "n": res["n"]})
        if res["acc"] > best_acc + 1e-12:
            best_acc = res["acc"]
            best_tau = tau

    sweep_esc = []
    chosen_cost_thresh = 2.0
    for c_th in (1.5, 2.0, 2.5, 3.0, 3.5):
        sel_res = _apply_policy(
            sel_only, opt_texts_by_split, abstain_thresh=best_tau, q_hat=0.85, min_set_size=2, cost_thresh=c_th
        )["s2_selection"]
        pr_res = (
            _apply_policy(
                probe_only, opt_texts_by_split, abstain_thresh=best_tau, q_hat=0.85, min_set_size=2, cost_thresh=c_th
            ).get("escalate_probe", {})
        )
        sweep_esc.append({
            "cost_thresh": c_th,
            "s2_selection_acc": sel_res["acc"],
            "s2_selection_false_escalate": sel_res["false_escalate_rate"],
            "escalate_probe_escalate": pr_res.get("escalate_rate"),
            "escalate_probe_defer": pr_res.get("defer_or_escalate_rate"),
        })
    return {
        "best_abstain_thresh": best_tau,
        "best_s2_selection_acc": best_acc,
        "chosen_cost_thresh": chosen_cost_thresh,
        "q_hat": 0.85,
        "min_set_size": 2,
        "sweep_abstain": sweep_abstain,
        "sweep_escalate": sweep_esc,
    }


def _score_judge_rows(scorer: S2TeacherJudge, rows: list[OmniRow], batch_rows: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    bs = max(1, batch_rows)
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
            out.append({
                "id": r.id,
                "source": r.source,
                "tags": r.tags,
                "gold": r.label,
                "gold_kind": gk,
                "teacher_top": top,
                "teacher_kind": pk,
                "exact_ok": int(top == r.label),
                "kind_ok": int(pk == gk),
                "probs": p,
                "text": r.text[:240],
            })
        done = min(start + len(chunk), len(rows))
        if done % 25 < bs or done == len(rows):
            print(f"  [judge] {done}/{len(rows)}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s2-dir", required=True, help="Directory built by stage_s2.py")
    ap.add_argument("--ckpt", default=None, help="S1 locked v3b checkpoint dir (head.pt + adapter/)")
    ap.add_argument("--judge-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--four-bit", action="store_true")
    ap.add_argument("--load-in-4bit", action="store_true", help="Load evaluated UPP model in 4-bit NF4 with float32 compute")
    ap.add_argument("--rotations", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8, help="UPP eval batch size")
    ap.add_argument("--judge-batch-rows", type=int, default=4)
    ap.add_argument("--flip-orders", type=int, default=5)
    ap.add_argument("--max-rows", type=int, default=0, help="Optional cap per split for smoke tests")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="CPU/mock UPP smoke mode")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    s2_dir = Path(args.s2_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    report: dict[str, Any] = {"s2_dir": str(s2_dir), "ckpt": str(args.ckpt)}
    scored_rows: list[dict[str, Any]] = []
    opt_texts_by_split: dict[str, dict[str, dict[str, str]]] = {}

    if not args.skip_baseline:
        if not args.ckpt and not args.smoke:
            raise ValueError("--ckpt is required unless --skip-baseline or --smoke is passed")
        device = "cuda" if torch.cuda.is_available() and not args.smoke else "cpu"
        dtype = None if device == "cpu" else torch.float32
        if args.ckpt:
            model = UPPModel.from_checkpoint(args.ckpt, use_mock=args.smoke, torch_dtype=dtype, load_in_4bit=args.load_in_4bit)
        else:
            from soro.routing.upp import UPPConfig
            model = UPPModel(UPPConfig(layout="query_first", head="dot", refiner=True, option_index=False, option_isolation=True), use_mock=True)
        if dtype is torch.float32 and not model.is_mock and not getattr(args, "load_in_4bit", False):
            model.backbone.float()
        model.to(device)
        model.eval()

        for sp_name, fname in DEFAULT_SPLITS:
            p = s2_dir / fname
            if not p.exists():
                print(f"[skip missing split] {p}", flush=True)
                continue
            rows = read_jsonl(p)
            opt_texts_by_split[sp_name] = _load_option_texts(p)
            if args.max_rows > 0:
                rows = rows[: args.max_rows]
            # For escalate_probe, meta['probe_gold'] is __escalate__, while menu label is ABSTAIN
            # Wait: read_jsonl wraps into UPP format; preserve meta tags properly
            print(f"[S2 Baseline fp32] Scoring {sp_name} ({len(rows)} rows, orders={args.flip_orders})...", flush=True)
            scored = _score_split_shard(
                model=model,
                split_name=sp_name,
                rows=rows,
                flip_orders=args.flip_orders,
                batch_size=args.batch_size,
                seed=42,
                shard_idx=0,
                shard_total=1,
            )
            scored_rows.extend(scored)

        raw_summary = summarize_by_split(scored_rows)
        tuning = _select_thresholds(scored_rows, opt_texts_by_split)
        tuned_tau = tuning["best_abstain_thresh"]
        policy_raw = _apply_policy(
            scored_rows, opt_texts_by_split, abstain_thresh=None, q_hat=0.85, min_set_size=2, cost_thresh=999.0
        )
        policy_tuned = _apply_policy(
            scored_rows, opt_texts_by_split, abstain_thresh=tuned_tau, q_hat=0.85, min_set_size=2, cost_thresh=tuning["chosen_cost_thresh"]
        )
        report["baseline_raw_fp32"] = raw_summary
        report["baseline_threshold_tuning"] = tuning
        report["baseline_policy_raw"] = policy_raw
        report["baseline_policy_tuned"] = policy_tuned
        (out_dir / "s2_baseline_scored_rows.json").write_text(json.dumps({"rows": scored_rows}), encoding="utf-8")

        # Free UPP GPU memory cleanly before loading Qwen3-8B teacher
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not args.skip_judge:
        if args.smoke:
            # Mock uniform/heuristic probs in smoke mode without downloading 8B
            class _MockScorer:
                def __init__(self) -> None:
                    self.multimodal = True

                def probs(self, row: OmniRow) -> dict[str, float]:
                    k = len(row.options)
                    return {o.key: (0.4 if o.key == row.label else 0.6 / max(1, k - 1)) for o in row.options}

            scorer: Any = _MockScorer()
        else:
            scorer = S2TeacherJudge(args.judge_model, load_in_4bit=args.four_bit, rotations=args.rotations)

        gold100_path = s2_dir / "s2_gold_100.jsonl"
        if gold100_path.exists():
            g_rows = [OmniRow.from_json(json.loads(l)) for l in gold100_path.open(encoding="utf-8") if l.strip()]
            if args.max_rows > 0:
                g_rows = g_rows[: args.max_rows]
            print(f"[Gold-100 Judge] Scoring {len(g_rows)} rows...", flush=True)
            g_judged = _score_judge_rows(scorer, g_rows, batch_rows=args.judge_batch_rows)
            (out_dir / "s2_gold_100_judge.jsonl").write_text(
                "\n".join(json.dumps(x, ensure_ascii=False) for x in g_judged) + "\n", encoding="utf-8"
            )
            by_act: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "exact_ok": 0, "kind_ok": 0})
            conf = Counter()
            for rec in g_judged:
                gk = rec["gold_kind"]
                by_act[gk]["n"] += 1
                by_act[gk]["exact_ok"] += rec["exact_ok"]
                by_act[gk]["kind_ok"] += rec["kind_ok"]
                conf[(gk, rec["teacher_kind"])] += 1
            gold_summary = {}
            for act in ("tool", "answer", "ask", "abstain"):
                d = by_act[act]
                n = d["n"]
                gold_summary[act] = {
                    "n": n,
                    "exact_ok": d["exact_ok"],
                    "exact_acc": d["exact_ok"] / n if n else 0.0,
                    "exact_ci95": list(wilson_ci(d["exact_ok"], n)),
                    "kind_ok": d["kind_ok"],
                    "kind_acc": d["kind_ok"] / n if n else 0.0,
                    "kind_ci95": list(wilson_ci(d["kind_ok"], n)),
                    "confusion": {pk: conf[(act, pk)] for (a, pk) in sorted(conf) if a == act},
                }
            report["gold_100_judge"] = gold_summary

        # +3.0 pt gate on s2_selection tool rows
        sel_path = s2_dir / "s2_selection.jsonl"
        if sel_path.exists():
            sel_omni = [OmniRow.from_json(json.loads(l)) for l in sel_path.open(encoding="utf-8") if l.strip()]
            tool_omni = [r for r in sel_omni if action_kind(r.label) == "tool"]
            if args.max_rows > 0:
                tool_omni = tool_omni[: args.max_rows]
            print(f"[+3pt Gate] Judging {len(tool_omni)} s2_selection tool rows...", flush=True)
            t_judged = _score_judge_rows(scorer, tool_omni, batch_rows=args.judge_batch_rows)
            t_map = {x["id"]: x for x in t_judged}
            v_map = {r["id"]: r for r in scored_rows if r["split"] == "s2_selection"}
            shared_ids = [rid for rid in t_map if rid in v_map] if v_map else list(t_map)

            t_ok = sum(t_map[rid]["exact_ok"] for rid in shared_ids)
            n_g = len(shared_ids)
            gate_info: dict[str, Any] = {
                "n_tool_rows": n_g,
                "teacher_top1_k": t_ok,
                "teacher_top1": t_ok / n_g if n_g else 0.0,
                "teacher_ci95": list(wilson_ci(t_ok, n_g)),
            }
            if v_map:
                v_ok = sum(int(v_map[rid]["ok"]) for rid in shared_ids)
                n10 = sum(1 for rid in shared_ids if v_map[rid]["ok"] and not t_map[rid]["exact_ok"])
                n01 = sum(1 for rid in shared_ids if not v_map[rid]["ok"] and t_map[rid]["exact_ok"])
                delta_pts = ((t_ok - v_ok) / n_g) * 100.0 if n_g else 0.0
                gate_info.update({
                    "v3b_top1_k": v_ok,
                    "v3b_top1": v_ok / n_g if n_g else 0.0,
                    "v3b_ci95": list(wilson_ci(v_ok, n_g)),
                    "delta_pts": delta_pts,
                    "soft_label_gate_passed": bool(delta_pts >= 3.0),
                    "mcnemar": mcnemar_exact(n10, n01),
                    "n10_v3b_only": n10,
                    "n01_teacher_only": n01,
                })
            report["soft_label_gate"] = gate_info
            (out_dir / "s2_selection_tool_judge.jsonl").write_text(
                "\n".join(json.dumps(x, ensure_ascii=False) for x in t_judged) + "\n", encoding="utf-8"
            )

    report["elapsed_s"] = round(time.perf_counter() - t0, 1)
    (out_dir / "s2_baseline_and_gate.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Compact human-readable Markdown report
    lines = ["# S2 Baseline (`v3b_fp32`) + Gold-100 Judge + Soft-Label Gate", ""]
    if "baseline_threshold_tuning" in report:
        tu = report["baseline_threshold_tuning"]
        lines.append(
            f"- Tuned strictly on `s2_selection`: `abstain_thresh={tu['best_abstain_thresh']}` (acc `{tu['best_s2_selection_acc']:.4f}`), "
            f"`escalate` rule: `set_size >= {tu['min_set_size']} & score >= {tu['chosen_cost_thresh']}` (LAC `q_hat={tu['q_hat']}`)."
        )
        lines.append("")
        lines.append("## 1. S2 Baseline Results (`fp32`)")
        lines.append("| Split | n | Raw `v3b` Acc (95% CI) | Raw AskOrExact | Tuned+Rule Acc | False-Esc | `action:tool` | `action:ask` | `action:abstain` | `missing_crit_param` |")
        lines.append("| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |")
        for sp, d in report["baseline_raw_fp32"].items():
            if sp == "escalate_probe":
                pr = report["baseline_policy_tuned"].get("escalate_probe", {})
                lines.append(
                    f"| {sp} (probe) | {d['n']} | - | - | esc=`{pr.get('escalate_rate',0):.3f}` ({pr.get('escalate_k',0)}/{d['n']}) | defer=`{pr.get('defer_or_escalate_rate',0):.3f}` | exec=`{pr.get('tool_exec_rate',0):.3f}` | - | - | - |"
                )
                continue
            ci = d["acc_ci95"]
            pt = report["baseline_policy_tuned"].get(sp, {})
            subs = d.get("subsets") or {}
            def _sub(k: str) -> str:
                s = subs.get(k)
                return f"{s['acc']:.3f} ({s['acc_k']}/{s['n']})" if s else "-"
            lines.append(
                f"| {sp} | {d['n']} | {d['acc']:.3f} ({d['acc_k']}/{d['n']}) [{ci[0]:.3f},{ci[1]:.3f}] | {d['ask_or_exact']:.3f} | "
                f"{pt.get('acc', d['acc']):.3f} ({pt.get('acc_k', d['acc_k'])}/{d['n']}) | {pt.get('false_escalate_k', 0)}/{d['n']} | "
                f"{_sub('action:tool')} | {_sub('action:ask')} | {_sub('action:abstain')} | {_sub('cat:missing_critical_parameter')} |"
            )
    if "gold_100_judge" in report:
        lines.extend(["", "## 2. Gold-100 Judge Agreement (vs Rule Labels)", "| Action | n | Exact Agr (95% CI) | Kind Agr (95% CI) | Confusion |", "| :-- | --: | --: | --: | :-- |"])
        for act, d in report["gold_100_judge"].items():
            eci, kci = d["exact_ci95"], d["kind_ci95"]
            lines.append(f"| {act} | {d['n']} | {d['exact_acc']:.3f} ({d['exact_ok']}/{d['n']}) [{eci[0]:.3f},{eci[1]:.3f}] | {d['kind_acc']:.3f} ({d['kind_ok']}/{d['n']}) [{kci[0]:.3f},{kci[1]:.3f}] | {d['confusion']} |")
    if "soft_label_gate" in report:
        g = report["soft_label_gate"]
        lines.extend(["", "## 3. Soft-Label +3.0 pt Gate (`s2_selection` tool rows)"])
        lines.append(
            f"- n=`{g['n_tool_rows']}` | Teacher top-1: `{g['teacher_top1']:.4f}` (`{g['teacher_top1_k']}/{g['n_tool_rows']}`) | "
            f"`v3b` top-1: `{g.get('v3b_top1', 0):.4f}` (`{g.get('v3b_top1_k', 0)}/{g['n_tool_rows']}`) | "
            f"Δ=`{g.get('delta_pts', 0):+.2f} pts` | Gate `>= +3.0 pts`: **{g.get('soft_label_gate_passed')}**"
        )
    md = "\n".join(lines) + "\n"
    (out_dir / "s2_baseline_and_gate.md").write_text(md, encoding="utf-8")
    print(md, flush=True)


if __name__ == "__main__":
    main()
