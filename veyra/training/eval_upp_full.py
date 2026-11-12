"""Full held-out UPP evaluator with 95% Wilson CIs, 5-order permutation flips, and collision slices.

Evaluates a saved UPP checkpoint (head.pt + adapter/) across large held-out sets
(selection, eval_zeroshot, eval_external, collisions, near_miss) using batched inference
and deterministic multi-order menu permutations. Supports 2-GPU sharding (--shard 0/2, 1/2)
followed by --merge-shards so large suites complete in minutes on 2x T4.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "veyra" / "python" / "src"))
sys.path.insert(0, str(REPO / "veyra" / "training"))

from p0_upp_ablation import auroc, conformal, ece, gold_key, is_correct, read_jsonl  # noqa: E402
from soro.omni_data.s2_actions import action_kind  # noqa: E402
from soro.omni_data.schema import ASK  # noqa: E402
from soro.routing.upp import UPPModel  # noqa: E402


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Exact 95% Wilson score binomial confidence interval."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)
    return (max(0.0, center - half), min(1.0, center + half))


def _row_tags(r: dict[str, Any]) -> set[str]:
    tags = set(r.get("tags") or [])
    if r.get("has_collision_distractor"):
        tags.add("collision")
    return tags


def _shuffled_criteria(crit: dict[str, str], rng: random.Random) -> dict[str, str]:
    keys = list(crit)
    rng.shuffle(keys)
    return {k: crit[k] for k in keys}


def _score_split_shard(
    model: UPPModel,
    split_name: str,
    rows: list[dict[str, Any]],
    flip_orders: int,
    batch_size: int,
    seed: int,
    shard_idx: int,
    shard_total: int,
    max_seq_len: int | None = None,
    use_amp: bool = False,
) -> list[dict[str, Any]]:
    device = next(model.head.parameters()).device
    use_cuda = device.type == "cuda"
    use_autocast = use_cuda and use_amp and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_autocast else torch.float32

    sub_rows = [(i, r) for i, r in enumerate(rows) if i % shard_total == shard_idx]
    if not sub_rows:
        return []

    # Optionally check length cap only if specified (eval uses generous length cap).
    work_items: list[tuple[int, dict[str, Any], list[dict[str, str]]]] = []
    for i, r in sub_rows:
        crit = r["questions"]["tool_route"]["criteria"]
        rng = random.Random(seed * 100003 + i)
        orders = [_shuffled_criteria(crit, rng) for _ in range(max(1, flip_orders))]
        work_items.append((i, r, orders))

    # Flatten (row_pos, order_idx, state, crit, media) so batched inference handles everything uniformly.
    flat_inputs: list[tuple[str, dict[str, str], Any]] = []
    flat_meta: list[tuple[int, int]] = []
    for pos, (_, r, orders) in enumerate(work_items):
        for o_idx, crit_o in enumerate(orders):
            if max_seq_len is not None and len(model.build(r["state"], crit_o)["ids"]) > max_seq_len:
                continue
            flat_inputs.append((r["state"], crit_o, r.get("media")))
            flat_meta.append((pos, o_idx))

    per_row_orders: list[dict[int, tuple[dict[str, float], dict[str, Any]]]] = [{} for _ in work_items]
    t0 = time.perf_counter()
    with torch.autocast("cuda", dtype=amp_dtype, enabled=use_autocast):
        for start in range(0, len(flat_inputs), max(1, batch_size)):
            chunk = flat_inputs[start : start + max(1, batch_size)]
            res = model.batch_score(chunk, batch_size=len(chunk))
            for idx_in_chunk, (probs, info) in enumerate(res):
                pos, o_idx = flat_meta[start + idx_in_chunk]
                per_row_orders[pos][o_idx] = (probs, info)
            if (start // max(1, batch_size) + 1) % 50 == 0 or start + len(chunk) >= len(flat_inputs):
                elapsed = time.perf_counter() - t0
                done = start + len(chunk)
                print(
                    f"  [{split_name} shard {shard_idx}/{shard_total}] {done}/{len(flat_inputs)} passes "
                    f"({done / max(elapsed, 1e-6):.1f} pass/s)",
                    flush=True,
                )

    out_rows: list[dict[str, Any]] = []
    for pos, (orig_idx, r, _) in enumerate(work_items):
        ord_map = per_row_orders[pos]
        if 0 not in ord_map:
            continue
        probs0, info0 = ord_map[0]
        tops = [max(ord_map[k][0], key=ord_map[k][0].get) for k in sorted(ord_map)]
        flipped = len(set(tops)) > 1
        max_spread = 0.0
        if len(ord_map) > 1:
            keys = list(probs0)
            for k in sorted(ord_map)[1:]:
                pk = ord_map[k][0]
                spread = max(abs(probs0[key] - pk[key]) for key in keys)
                if spread > max_spread:
                    max_spread = spread
        top0 = max(probs0, key=probs0.get)
        ok = is_correct(r, top0)
        ask_or_ok = bool(ok or (ASK in probs0 and top0 == ASK))
        g = gold_key(r, probs0)
        g_kind = action_kind(g)
        p_kind = action_kind(top0)
        tags = sorted(_row_tags(r))
        out_rows.append(
            {
                "split": split_name,
                "orig_idx": orig_idx,
                "id": r.get("id"),
                "gold": g,
                "gold_kind": g_kind,
                "top": top0,
                "pred_kind": p_kind,
                "ok": int(ok),
                "ask_or_ok": int(ask_or_ok),
                "abstain": int(g == "__abstain__"),
                "collision": int("collision" in tags),
                "near_miss": int("near_miss" in tags),
                "cats": sorted(t for t in tags if t.startswith("cat:")),
                "flipped": int(flipped),
                "max_spread": float(max_spread),
                "probs": probs0,
                "u": float(info0.get("u", float("nan"))),
                "energy": float(info0.get("energy", float("nan"))),
            }
        )
    return out_rows


def summarize_scored_rows(rows: list[dict[str, Any]], include_conformal: bool = True) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    hits = [int(r["ok"]) for r in rows]
    acc_k = sum(hits)
    acc = acc_k / n
    acc_ci = wilson_ci(acc_k, n)
    ask_ok_k = sum(int(r.get("ask_or_ok", r["ok"])) for r in rows)
    ask_ok = ask_ok_k / n
    ask_ok_ci = wilson_ci(ask_ok_k, n)
    flips_k = sum(int(r["flipped"]) for r in rows)
    flip_rate = flips_k / n
    flip_ci = wilson_ci(flips_k, n)

    confs = [max(r["probs"].values()) for r in rows]
    nll = sum(-math.log(max(r["probs"].get(r["gold"], 0.0), 1e-9)) for r in rows) / n
    brier = (
        sum(
            sum((p - (1.0 if k == r["gold"] else 0.0)) ** 2 for k, p in r["probs"].items()) / len(r["probs"])
            for r in rows
        )
        / n
    )
    spreads = sorted(float(r["max_spread"]) for r in rows)
    p95_idx = min(len(spreads) - 1, int(0.95 * len(spreads)))

    abst_rows = [r for r in rows if r["abstain"]]
    abst_k = sum(r["ok"] for r in abst_rows)
    out: dict[str, Any] = {
        "n": n,
        "acc": acc,
        "acc_k": acc_k,
        "acc_ci95": list(acc_ci),
        "ask_or_exact": ask_ok,
        "ask_or_exact_k": ask_ok_k,
        "ask_or_exact_ci95": list(ask_ok_ci),
        "nll": nll,
        "brier": brier,
        "ece": ece(confs, hits),
        "flips_k": flips_k,
        "flip_rate": flip_rate,
        "flip_ci95": list(flip_ci),
        "max_prob_spread_p95": spreads[p95_idx],
        "max_prob_spread_max": spreads[-1],
        "abstain_rows": len(abst_rows),
        "refusal": (abst_k / len(abst_rows)) if abst_rows else None,
        "refusal_ci95": list(wilson_ci(abst_k, len(abst_rows))) if abst_rows else None,
    }
    if include_conformal and n >= 4:
        scored = [{"probs": r["probs"], "gold": r["gold"], "u": r["u"]} for r in rows]
        out["conformal"] = conformal(scored)
        if all(math.isfinite(r["u"]) for r in rows):
            out["conformal_normalized"] = conformal(scored, normalized=True)
            errors = [1 - h for h in hits]
            energies = [r["energy"] for r in rows]
            out["error_auroc"] = {
                "u": auroc([r["u"] for r in rows], errors),
                "neg_energy": auroc([-e for e in energies], errors) if all(math.isfinite(e) for e in energies) else None,
                "one_minus_maxprob": auroc([1.0 - c for c in confs], errors),
            }
    return out


def summarize_by_split(all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    splits: dict[str, list[dict[str, Any]]] = {}
    for r in all_rows:
        splits.setdefault(r["split"], []).append(r)
    summary: dict[str, Any] = {}
    for name, srows in splits.items():
        srows = sorted(srows, key=lambda x: x["orig_idx"])
        entry = summarize_scored_rows(srows, include_conformal=True)
        subsets: dict[str, Any] = {}
        col_rows = [r for r in srows if r["collision"]]
        if col_rows:
            subsets["collision"] = summarize_scored_rows(col_rows, include_conformal=False)
        nm_rows = [r for r in srows if r["near_miss"]]
        if nm_rows:
            subsets["near_miss"] = summarize_scored_rows(nm_rows, include_conformal=False)
        tool_rows = [r for r in srows if not r["abstain"]]
        if tool_rows:
            subsets["tool_call"] = summarize_scored_rows(tool_rows, include_conformal=False)
        for act in ("tool", "ask", "abstain", "answer"):
            arows = [r for r in srows if r.get("gold_kind", action_kind(r["gold"])) == act]
            if arows and len(arows) < len(srows):
                subsets[f"action:{act}"] = summarize_scored_rows(arows, include_conformal=False)
        all_cats = sorted({c for r in srows for c in (r.get("cats") or [])})
        for cat in all_cats:
            crows = [r for r in srows if cat in (r.get("cats") or [])]
            if crows and len(crows) < len(srows):
                subsets[cat] = summarize_scored_rows(crows, include_conformal=False)
        if subsets:
            entry["subsets"] = subsets
        summary[name] = entry
    return summary


def format_markdown(summary: dict[str, Any], title: str = "Full Evaluation") -> str:
    lines = [
        f"## {title}",
        "",
        "| Split (slice) | n | Acc (95% CI) | AskOrExact | NLL | Brier | ECE | Refusal (95% CI) | 5-Order Flips (95% CI) | LAC set @cov | Err-AUROC u |",
        "| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |",
    ]

    def _row_line(label: str, d: dict[str, Any]) -> str:
        ci = d.get("acc_ci95", [0.0, 0.0])
        acc_s = f"{d['acc']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
        aoe = d.get("ask_or_exact", d["acc"])
        aoe_s = f"{aoe:.3f}"
        if d.get("refusal") is None:
            ref_s = "-"
        else:
            rci = d.get("refusal_ci95", [0.0, 0.0])
            ref_s = f"{d['refusal']:.3f} [{rci[0]:.3f}, {rci[1]:.3f}] ({d['abstain_rows']})"
        fci = d.get("flip_ci95", [0.0, 0.0])
        flip_s = f"{d['flip_rate']:.4f} ({d['flips_k']}) [{fci[0]:.4f}, {fci[1]:.4f}]"
        lac = d.get("conformal")
        lac_s = f"{lac['set_size']:.2f} @ {lac['coverage']:.3f}" if lac else "-"
        au = (d.get("error_auroc") or {}).get("u")
        au_s = f"{au:.3f}" if au is not None else "-"
        return f"| {label} | {d['n']} | {acc_s} | {aoe_s} | {d['nll']:.3f} | {d['brier']:.4f} | {d['ece']:.3f} | {ref_s} | {flip_s} | {lac_s} | {au_s} |"

    for split, d in summary.items():
        if d.get("n", 0) == 0:
            continue
        lines.append(_row_line(split, d))
        for sub_name, sub_d in (d.get("subsets") or {}).items():
            lines.append(_row_line(f"{split}/{sub_name}", sub_d))
    return "\n".join(lines)


def _parse_splits(spec: str) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for item in [x.strip() for x in spec.split(",") if x.strip()]:
        if "=" not in item:
            raise ValueError(f"Expected name=path in --eval-files, got {item!r}")
        name, p = item.split("=", 1)
        out.append((name.strip(), Path(p.strip())))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=None, help="Directory holding head.pt (+ optional adapter/)")
    ap.add_argument("--eval-files", default="", help="Comma-separated name=path.jsonl splits")
    ap.add_argument("--flip-orders", type=int, default=5, help="Number of random menu orders per row")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard", default="0/1", help="Shard index/total, e.g. 0/2 or 1/2")
    ap.add_argument("--merge-shards", default="", help="Comma-separated scored_rows_*.json files to merge")
    ap.add_argument("--smoke", action="store_true", help="Mock/CPU tiny sanity mode")
    ap.add_argument("--amp", action="store_true", help="Allow bf16 autocast on GPUs with native bf16 (off by default to match fp32 scoring)")
    ap.add_argument("--max-rows", type=int, default=0, help="Optional cap per split (0 = all rows)")
    ap.add_argument("--out", default=str(REPO / "veyra" / "runs" / "full_eval"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_shards:
        merged_rows: list[dict[str, Any]] = []
        meta_cfg: dict[str, Any] = {}
        for p_str in [x.strip() for x in args.merge_shards.split(",") if x.strip()]:
            blob = json.loads(Path(p_str).read_text(encoding="utf-8"))
            meta_cfg = blob.get("model") or meta_cfg
            merged_rows.extend(blob["rows"])
        split_order = {s: idx for idx, s in enumerate(dict.fromkeys(r["split"] for r in merged_rows))}
        merged_rows.sort(key=lambda r: (split_order[r["split"]], r["orig_idx"]))
        summary = summarize_by_split(merged_rows)
        md = format_markdown(summary, title=meta_cfg.get("config", "Full Evaluation"))
        payload = {"model": meta_cfg, "summary": summary}
        (out_dir / "scored_rows.json").write_text(json.dumps({"model": meta_cfg, "rows": merged_rows}), encoding="utf-8")
        (out_dir / "full_eval.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (out_dir / "full_eval.md").write_text(md + "\n", encoding="utf-8")
        print(md)
        return

    if not args.ckpt:
        raise ValueError("--ckpt is required unless --merge-shards is used")

    shard_idx_s, shard_tot_s = args.shard.split("/")
    shard_idx, shard_total = int(shard_idx_s), int(shard_tot_s)
    device = "cuda" if torch.cuda.is_available() and not args.smoke else "cpu"
    dtype = None if (device == "cpu" or (args.amp and torch.cuda.is_bf16_supported())) else torch.float32
    model = UPPModel.from_checkpoint(args.ckpt, use_mock=args.smoke, torch_dtype=dtype)
    if dtype is torch.float32 and not model.is_mock:
        model.backbone.float()
    model.to(device)
    model.eval()

    eval_pairs = _parse_splits(args.eval_files)
    if not eval_pairs:
        raise ValueError("Provide --eval-files name=path,...")

    all_scored: list[dict[str, Any]] = []
    for name, path in eval_pairs:
        rows = read_jsonl(path)
        if args.max_rows > 0:
            rows = rows[: args.max_rows]
        print(f"Scoring {name} ({len(rows)} rows) | shard {shard_idx}/{shard_total} | orders={args.flip_orders}", flush=True)
        scored = _score_split_shard(
            model=model,
            split_name=name,
            rows=rows,
            flip_orders=args.flip_orders,
            batch_size=args.batch_size,
            seed=args.seed,
            shard_idx=shard_idx,
            shard_total=shard_total,
            max_seq_len=args.max_seq_len,
            use_amp=args.amp,
        )
        all_scored.extend(scored)

    model_meta = {
        "config": model.cfg.name(),
        "cfg": model.config_dict(),
        "ckpt": str(args.ckpt),
        "flip_orders": args.flip_orders,
        "seed": args.seed,
        "amp": bool(args.amp),
    }
    if shard_total > 1:
        shard_path = out_dir / f"scored_shard_{shard_idx}.json"
        shard_path.write_text(json.dumps({"model": model_meta, "rows": all_scored}), encoding="utf-8")
        print(f"Wrote {len(all_scored)} scored rows to {shard_path}", flush=True)
        return

    summary = summarize_by_split(all_scored)
    md = format_markdown(summary, title=model.cfg.name())
    payload = {"model": model_meta, "summary": summary}
    (out_dir / "scored_rows.json").write_text(json.dumps({"model": model_meta, "rows": all_scored}), encoding="utf-8")
    (out_dir / "full_eval.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "full_eval.md").write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()

