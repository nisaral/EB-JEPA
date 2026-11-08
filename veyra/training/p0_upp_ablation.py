"""P0 ablation for the Universal Probabilistic Pointer (UPP) on a single small GPU.

Trains several UPP variants with identical data, seed, steps and backbone, then
scores each on held-out rows. Each variant changes one part of the head or layout:

  A  query_first + dot                       (Soro-Kev v0.2 / Kev pointer)
  B  query_first + gauss                     (Gaussian pointer)
  C  catalog_first + gauss + refiner         (cached catalog, set refiner)
  D  catalog_first + gauss + refiner + slots (adds think slots)
  E  catalog_first + dot + refiner           (separates layout from the Gaussian)

Held-out sets (none is used for training or model selection):
  selection : whole train packages carved out with the training script's split
  zeroshot  : a stride sample of val_zeroshot.jsonl (55 unseen packages)
  bfcl      : the 100-row BFCL v3 diagnostic (34 multiple, 33 live, 33 irrelevance)

Metrics: top-1 accuracy, NLL, Brier, ECE, refusal on no-tool rows, order-flip
rate under reversed menus, split-conformal coverage and set size (plain LAC and,
for Gaussian heads, variance-normalized), and median latency (cold and with a
cached catalog for catalog_first).

Example (RTX 4050 6 GB):
  python veyra/training/p0_upp_ablation.py --configs A,B,C,D,E --train-rows 4000
CPU smoke test:
  python veyra/training/p0_upp_ablation.py --smoke
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "veyra" / "python" / "src"))
sys.path.insert(0, str(REPO / "veyra" / "training"))

from soro.routing.upp import UPPConfig, UPPModel  # noqa: E402
from train_soro_kev import get_cosine_lr, split_selection_packages  # noqa: E402

CONFIGS = {
    # round 1
    "A": UPPConfig(layout="query_first", head="dot"),
    "B": UPPConfig(layout="query_first", head="gauss"),
    "C": UPPConfig(layout="catalog_first", head="gauss", refiner=True),
    "D": UPPConfig(layout="catalog_first", head="gauss", refiner=True, think_slots=8),
    "E": UPPConfig(layout="catalog_first", head="dot", refiner=True),
    # round 2
    "F": UPPConfig(layout="block", head="dot", refiner=True),
    "G": UPPConfig(layout="query_first", head="rbf"),
    "H": UPPConfig(layout="query_first", head="dot", ensemble=4),
    "I": UPPConfig(layout="query_first", head="dot", exits=(0.5, 0.75)),
    "J": UPPConfig(layout="block", head="rbf", refiner=True, ensemble=4, exits=(0.5, 0.75)),
}

DATA = REPO / "releases" / "soro-universal-tool-harness"


def resolve_data_dir(arg: str | None) -> Path:
    """Local release dir, an explicit --data-dir, or the public HF dataset (Kaggle/Colab)."""
    for cand in [Path(arg)] if arg else [DATA]:
        if (cand / "train.jsonl").exists():
            return cand
    from huggingface_hub import snapshot_download

    return Path(snapshot_download("nisaralll/soro-universal-tool-harness", repo_type="dataset",
                                  allow_patterns=["train.jsonl", "val_zeroshot.jsonl"]))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_bfcl() -> list[dict[str, Any]]:
    """The 100-row BFCL diagnostic, or [] when the BFCL files are not present (e.g. on Kaggle)."""
    if not (REPO / "veyra" / "data" / "bfcl" / "soro" / "bfcl_routing.jsonl").exists():
        print("BFCL slice not found; skipping the bfcl split", flush=True)
        return []
    sys.path.insert(0, str(REPO / "benchmarks"))
    import generate_all_predictions as g  # noqa: E402

    return g.load_bfcl_slice()


def auroc(scores: list[float], positives: list[int]) -> float | None:
    """AUROC of ``scores`` for detecting positives (rank-based, ties averaged)."""
    pos = sum(positives)
    neg = len(positives) - pos
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    r_pos = sum(r for r, y in zip(ranks, positives) if y)
    return (r_pos - pos * (pos + 1) / 2) / (pos * neg)


def is_correct(r: dict[str, Any], top: str) -> bool:
    q = r["questions"]["tool_route"]
    golds = r.get("gold_functions") or []
    if golds and r.get("bfcl_file") != "BFCL_v3_irrelevance.json" and q["label"] != "__abstain__":
        return top in golds
    return top == q["label"]


def gold_key(r: dict[str, Any], probs: dict[str, float]) -> str:
    q = r["questions"]["tool_route"]
    golds = [g for g in (r.get("gold_functions") or []) if g in probs]
    if golds and q["label"] != "__abstain__" and r.get("bfcl_file") != "BFCL_v3_irrelevance.json":
        return max(golds, key=lambda g: probs[g])
    return q["label"]


def ece(conf: list[float], hit: list[int], bins: int = 15) -> float:
    conf_a, hit_a = np.array(conf), np.array(hit)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf_a > lo) & (conf_a <= hi)
        if m.any():
            total += m.mean() * abs(conf_a[m].mean() - hit_a[m].mean())
    return float(total)


def conformal(rows: list[dict[str, Any]], alpha: float = 0.10, splits: int = 50, normalized: bool = False) -> dict[str, float]:
    """Split conformal (LAC) over random 50/50 splits. normalized divides scores by sqrt(u)."""
    rng = random.Random(0)
    covs, sizes, singles = [], [], []
    for _ in range(splits):
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        half = len(idx) // 2
        cal, test = [rows[i] for i in idx[:half]], [rows[i] for i in idx[half:]]

        def u(row: dict[str, Any]) -> float:
            return math.sqrt(max(row["u"], 1e-6)) if normalized else 1.0

        scores = np.array([(1.0 - row["probs"].get(row["gold"], 0.0)) / u(row) for row in cal])
        n = len(scores)
        level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
        q_hat = float(np.quantile(scores, level, method="higher"))
        cov = size = single = 0
        for row in test:
            s = [k for k, p in row["probs"].items() if (1.0 - p) / u(row) <= q_hat]
            cov += int(row["gold"] in s)
            size += len(s)
            single += int(len(s) == 1)
        covs.append(cov / len(test))
        sizes.append(size / len(test))
        singles.append(single / len(test))
    return {"coverage": float(np.mean(covs)), "set_size": float(np.mean(sizes)), "singleton": float(np.mean(singles))}


def evaluate(model: UPPModel, name: str, rows: list[dict[str, Any]], timing_rows: int = 40) -> dict[str, Any]:
    model.eval()
    scored, conf, hit, flips, lat = [], [], [], 0, []
    exit_hits: list[list[int]] = []
    exit_depth: list[float] = []
    energies: list[float] = []
    abstain_hits = abstain_n = 0
    nll = brier = 0.0
    use_cuda = next(model.head.parameters()).is_cuda
    for i, r in enumerate(rows):
        crit = r["questions"]["tool_route"]["criteria"]
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        probs, info = model.score(r["state"], crit)
        if use_cuda:
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
        rev = dict(reversed(list(crit.items())))
        probs_rev, _ = model.score(r["state"], rev)
        top = max(probs, key=probs.get)
        flips += int(top != max(probs_rev, key=probs_rev.get))
        g = gold_key(r, probs)
        ok = is_correct(r, top)
        hit.append(int(ok))
        conf.append(probs[top])
        p_g = max(probs.get(g, 0.0), 1e-9)
        nll += -math.log(p_g)
        brier += sum((p - (1.0 if k == g else 0.0)) ** 2 for k, p in probs.items()) / len(probs)
        if g == "__abstain__":
            abstain_n += 1
            abstain_hits += int(ok)
        scored.append({"probs": probs, "gold": g, "u": info.get("u", float("nan"))})
        energies.append(info.get("energy", float("nan")))
        if info.get("exits"):
            hits_r = [int(is_correct(r, max(pe, key=pe.get))) for pe in info["exits"]] + [int(ok)]
            exit_hits.append(hits_r)
            # exit policy: first readout whose top probability >= 0.9, else the full depth
            depths = list(model.cfg.exits) + [1.0]
            chain = info["exits"] + [probs]
            for d, pe in zip(depths, chain):
                if max(pe.values()) >= 0.9 or d == 1.0:
                    exit_depth.append(d)
                    exit_hits[-1].append(int(is_correct(r, max(pe, key=pe.get))))
                    break
    n = len(rows)
    out: dict[str, Any] = {
        "n": n,
        "acc": sum(hit) / n,
        "nll": nll / n,
        "brier": brier / n,
        "ece": ece(conf, hit),
        "refusal": (abstain_hits / abstain_n) if abstain_n else None,
        "abstain_rows": abstain_n,
        "flip_rate": flips / n,
        "latency_ms": statistics.median(lat),
        "conformal": conformal(scored),
    }
    if all(math.isfinite(s["u"]) for s in scored):
        out["conformal_normalized"] = conformal(scored, normalized=True)
        errors = [1 - h for h in hit]
        out["error_auroc"] = {
            "u": auroc([s["u"] for s in scored], errors),
            "neg_energy": auroc([-e for e in energies], errors) if all(math.isfinite(e) for e in energies) else None,
            "one_minus_maxprob": auroc([1 - c for c in conf], errors),
        }
    if exit_hits:
        k = len(model.cfg.exits)
        out["exits"] = {
            "acc_per_readout": [sum(h[i] for h in exit_hits) / n for i in range(k + 1)],
            "policy_acc@0.9": sum(h[-1] for h in exit_hits) / n,
            "policy_mean_depth": sum(exit_depth) / len(exit_depth),
        }
    if model.cfg.layout in ("catalog_first", "block") and not model.is_mock:
        cached = []
        for r in rows[:timing_rows]:
            cache = model.encode_catalog(r["questions"]["tool_route"]["criteria"])
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.score_cached(r["state"], cache)
            if use_cuda:
                torch.cuda.synchronize()
            cached.append((time.perf_counter() - t0) * 1000)
        out["cached_latency_ms"] = statistics.median(cached)
        # Agreement check: cached and uncached scoring must pick the same tool.
        agree = 0
        for r in rows[:timing_rows]:
            crit = r["questions"]["tool_route"]["criteria"]
            a, _ = model.score(r["state"], crit)
            b, _ = model.score_cached(r["state"], model.encode_catalog(crit))
            agree += int(max(a, key=a.get) == max(b, key=b.get))
        out["cache_agreement"] = agree / min(timing_rows, n)
    print(f"  [{name}] acc {out['acc']:.3f} nll {out['nll']:.3f} ece {out['ece']:.3f} flips {out['flip_rate']:.3f} "
          f"set {out['conformal']['set_size']:.2f}@{out['conformal']['coverage']:.3f} lat {out['latency_ms']:.1f}ms",
          flush=True)
    return out


def train_one(key: str, cfg: UPPConfig, args: argparse.Namespace, train_rows: list[dict[str, Any]],
              evals: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.smoke else "cpu"
    four_bit = bool(getattr(args, "load_in_4bit", False))
    bf16_ok = device == "cuda" and torch.cuda.is_bf16_supported()
    # T4 has no bf16. fp16 weights + AdamW without a scaler underflow, so keep fp32
    # master weights and run the forward under fp16 autocast with a GradScaler.
    dtype = None if (four_bit or bf16_ok or device == "cpu") else torch.float32
    model = UPPModel(cfg, model_name=args.base_model, use_mock=args.smoke,
                     load_in_4bit=four_bit, torch_dtype=dtype)
    if dtype is torch.float32 and not model.is_mock:
        model.backbone.float()
    if not model.is_mock and hasattr(model.backbone, "enable_input_require_grads"):
        model.backbone.enable_input_require_grads()
        if args.no_grad_ckpt and hasattr(model.backbone, "gradient_checkpointing_disable"):
            # A 0.5B backbone fits in 6 GB without recomputation, and runs ~2x faster.
            model.backbone.gradient_checkpointing_disable()
    if getattr(args, "load_in_4bit", False) and device == "cuda":
        # A 4-bit backbone is already placed on the GPU; move only the new modules.
        model.head.to(device)
        if model.slots is not None:
            model.slots.data = model.slots.data.to(device)
    else:
        model.to(device)
    params = model.trainable_parameters()
    n_train = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps = max(1, len(train_rows) // args.batch)
    use_bf16 = bf16_ok
    use_fp16 = device == "cuda" and not bf16_ok and not four_bit
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    print(f"\n=== {key}: {cfg.name()} | trainable {n_train / 1e6:.2f}M | {steps} steps ===", flush=True)
    model.train()
    t0 = time.time()
    run_loss = run_acc = 0.0
    run_n = skipped = 0
    for step in range(steps):
        for pg in opt.param_groups:
            pg["lr"] = get_cosine_lr(step, steps, max(1, steps // 20), args.lr)
        batch = train_rows[step * args.batch:(step + 1) * args.batch]
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_bf16 or use_fp16):
            out = model.forward_batch(batch, max_seq_len=args.max_seq_len)
        if out["skipped"]:
            skipped += 1
            continue
        scaler.scale(out["loss"]).backward()
        if scaler.is_enabled():
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        run_loss += float(out["loss"].detach())
        run_acc += out["acc"]
        run_n += 1
        if run_n and (step + 1) % args.log_every == 0:
            print(f"  step {step + 1}/{steps} loss {run_loss / run_n:.4f} acc {run_acc / run_n:.3f} "
                  f"{(step + 1) * args.batch / (time.time() - t0):.1f} rows/s", flush=True)
            run_loss = run_acc = 0.0
            run_n = 0
    train_secs = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if device == "cuda" else 0.0
    result = {"config": cfg.name(), "trainable_params": n_train, "train_secs": train_secs,
              "steps": steps, "skipped_batches": skipped, "peak_vram_mb": peak, "eval": {}}
    for name, rows in evals.items():
        result["eval"][name] = evaluate(model, f"{key}/{name}", rows)
    del model, opt
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return result


def _fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.2f}"


def _auroc_s(e: dict[str, Any]) -> str:
    a = e.get("error_auroc")
    if not a:
        return "-"
    return f"{_fmt(a['u'])} / {_fmt(a['neg_energy'])} / {_fmt(a['one_minus_maxprob'])}"


def _exit_s(e: dict[str, Any]) -> str:
    x = e.get("exits")
    if not x:
        return "-"
    return f"{x['policy_acc@0.9']:.3f} @ {x['policy_mean_depth']:.2f}"


def markdown(results: dict[str, Any]) -> str:
    lines = []
    for split in next(iter(results.values()))["eval"]:
        lines.append(f"\n### {split}\n")
        lines.append("| Config | Acc | NLL | Brier | ECE | Refusal | Order flips | LAC set @cov | Norm. set @cov "
                     "| Err-AUROC u / energy / maxprob | Early exit acc @ depth | Latency ms | Cached ms |")
        lines.append("| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |")
        for key, res in results.items():
            e = res["eval"][split]
            norm = e.get("conformal_normalized")
            norm_s = f"{norm['set_size']:.2f} @ {norm['coverage']:.3f}" if norm else "-"
            ref = "-" if e["refusal"] is None else f"{e['refusal']:.2f} ({e['abstain_rows']})"
            lac = f"{e['conformal']['set_size']:.2f} @ {e['conformal']['coverage']:.3f}"
            cached = e.get("cached_latency_ms")
            cached_s = f"{cached:.1f}" if cached is not None else "-"
            lines.append(
                f"| {key} {res['config']} | {e['acc']:.3f} | {e['nll']:.3f} | {e['brier']:.4f} | {e['ece']:.3f} | "
                f"{ref} | {e['flip_rate']:.3f} | {lac} | {norm_s} | {_auroc_s(e)} | {_exit_s(e)} | "
                f"{e['latency_ms']:.1f} | {cached_s} |"
            )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default="A,B,C,D,E")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--train-rows", type=int, default=4000)
    ap.add_argument("--eval-rows", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq-len", type=int, default=768)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--out", default=str(REPO / "runs" / "p0_upp"))
    ap.add_argument("--data-dir", default=None, help="Folder with train.jsonl and val_zeroshot.jsonl")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--no-grad-ckpt", action="store_true", help="Disable gradient checkpointing (faster, more VRAM)")
    ap.add_argument("--smoke", action="store_true", help="Tiny CPU run with a mock backbone")
    args = ap.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    train_all = read_jsonl(data_dir / "train.jsonl")
    train_all, selection = split_selection_packages(train_all, frac=0.05, seed=args.seed)
    rng = random.Random(args.seed)
    rng.shuffle(train_all)
    zs = read_jsonl(data_dir / "val_zeroshot.jsonl")
    stride = max(1, len(zs) // args.eval_rows)
    evals = {
        "selection": selection[: args.eval_rows],
        "zeroshot": zs[::stride][: args.eval_rows],
        "bfcl": load_bfcl(),
    }
    evals = {k: v for k, v in evals.items() if v}
    train_rows = train_all[: args.train_rows]
    if args.smoke:
        args.base_model = "mock"
        train_rows = train_rows[:16]
        evals = {k: v[:12] for k, v in evals.items()}
        args.batch, args.log_every = 4, 2
    print(f"train {len(train_rows)} | " + " | ".join(f"{k} {len(v)}" for k, v in evals.items()), flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for key in [k.strip() for k in args.configs.split(",") if k.strip()]:
        results[key] = train_one(key, CONFIGS[key], args, train_rows, evals)
        (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    md = markdown(results)
    (out_dir / "results.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
