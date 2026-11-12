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
Spoken and screenshot requests (omni backbone, e.g. Gemma 4 E2B), rows from the omni data build:
  python veyra/training/p0_upp_ablation.py --base-model google/gemma-4-E2B --load-in-4bit --media \
      --media-rows runs/omni_data/audio/audio.jsonl,runs/omni_data/ui/ui.jsonl --media-root /kaggle/input
  Adds eval splits ``speech`` (TTS requests), ``speech_text`` (the same rows as their transcript:
  the text-vs-speech gap), ``ui`` (screenshot pointing) and mixes the train-split media rows into training.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

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
    # round 3: option compression and interaction
    "K": UPPConfig(layout="block", gists=4, refiner=True),
    "K8": UPPConfig(layout="block", gists=8, refiner=True),
    "Ksd": UPPConfig(layout="block", gists=4, refiner=True, gist_distill=1.0),
    "L": UPPConfig(layout="block", gists=4, interaction="late"),
    "P": UPPConfig(layout="block", refiner=True, option_pool="attn"),
    "J2": UPPConfig(layout="block", gists=4, gist_distill=1.0, head="rbf", refiner=True, ensemble=4),
    # round 4: Matryoshka gists (one cache for coarse retrieval + fine listwise rerank)
    "KM": UPPConfig(layout="block", gists=8, gist_budgets=(1, 2, 4, 8), gist_distill=1.0, refiner=True),
    # round 5: options read the request (cross-attention readout; fixes block configs at chance)
    # diagnostics: same as F / E without the refiner (A, which trains, has none)
    "F0": UPPConfig(layout="block", head="dot"),
    "E0": UPPConfig(layout="catalog_first", head="dot"),
    "FX": UPPConfig(layout="block", readout="xattn"),
    "KMX": UPPConfig(layout="block", gists=8, gist_budgets=(1, 2, 4, 8), gist_distill=1.0, readout="xattn"),
    # late MaxSim over 4 gists: request tokens see option summaries without a refiner
    "L0": UPPConfig(layout="block", gists=4, interaction="late"),
}

DATA = REPO / "releases" / "soro-universal-tool-harness"


def resolve_data_dir(arg: str | None) -> Path:
    """Local release dir, omni build dir, an explicit --data-dir, or the public HF dataset."""
    candidates = []
    if arg:
        candidates.append(Path(arg))
    else:
        candidates.extend([
            DATA,
            REPO / "veyra" / "runs" / "omni_data" / "build",
            REPO / "veyra" / "data" / "soro_kev_dataset",
        ])
    for cand in candidates:
        if (cand / "train.jsonl").exists():
            return cand
    from huggingface_hub import snapshot_download

    return Path(snapshot_download("nisaralll/soro-universal-tool-harness", repo_type="dataset",
                                  allow_patterns=["train.jsonl", "val_zeroshot.jsonl"]))


def to_kev_record(r: dict[str, Any]) -> dict[str, Any] | None:
    """Accept Soro-Kev rows or OmniRow jsonl from the omni build."""
    if "questions" in r and "tool_route" in r["questions"]:
        return r
    options = r.get("options") or []
    crit = {o["key"]: o["text"] for o in options if o.get("key") and o.get("text")}
    label = r.get("label")
    if not crit or label not in crit:
        return None
    ctx = r.get("context") or ""
    text = r.get("text") or ""
    state = (f"Context: {ctx}\n" if ctx else "") + f"User Request: {text}"
    return {
        "id": r.get("id"),
        "state": state,
        "questions": {"tool_route": {"criteria": crit, "label": label}},
        "source": r.get("source"),
        "package": r.get("package"),
        "tags": list(r.get("tags") or []),
        "meta": dict(r.get("meta") or {}),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = to_kev_record(json.loads(line))
            if rec is not None:
                rows.append(rec)
    return rows


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
        probs, info = model.score(r["state"], crit, media=r.get("media"))
        if use_cuda:
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
        top = max(probs, key=probs.get)
        if len(crit) <= 64:  # order invariance is exact for block layouts; skip the costly check on huge menus
            rev = dict(reversed(list(crit.items())))
            probs_rev, _ = model.score(r["state"], rev, media=r.get("media"))
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
    if model.cfg.gist_budgets:
        out["acc_by_budget"] = {}
        for b in model.cfg.gist_budgets:
            hit_b = 0
            for r in rows:
                pb, _ = model.score(r["state"], r["questions"]["tool_route"]["criteria"], budget=b, media=r.get("media"))
                hit_b += int(is_correct(r, max(pb, key=pb.get)))
            out["acc_by_budget"][b] = hit_b / n
        if not model.is_mock and len(rows[0]["questions"]["tool_route"]["criteria"]) > 24:
            hit2, lat2 = 0, []
            for r in rows:
                cat = model.encode_catalog(r["questions"]["tool_route"]["criteria"])
                if use_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                p2, _ = model.score_two_level(r["state"], cat, coarse=1, top_m=16, fine=model.cfg.gists, media=r.get("media"))
                if use_cuda:
                    torch.cuda.synchronize()
                lat2.append((time.perf_counter() - t0) * 1000)
                hit2 += int(is_correct(r, max(p2, key=p2.get)))
            out["two_level"] = {"acc": hit2 / n, "latency_ms": statistics.median(lat2), "coarse_budget": 1, "top_m": 16}
        print(f"  [{name}] acc by gist budget {out['acc_by_budget']} two-level {out.get('two_level')}", flush=True)
    if model.cfg.layout in ("catalog_first", "block") and not model.is_mock:
        cached = []
        for r in rows[:timing_rows]:
            cache = model.encode_catalog(r["questions"]["tool_route"]["criteria"])
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.score_cached(r["state"], cache, media=r.get("media"))
            if use_cuda:
                torch.cuda.synchronize()
            cached.append((time.perf_counter() - t0) * 1000)
        out["cached_latency_ms"] = statistics.median(cached)
        # Agreement check: cached and uncached scoring must pick the same tool.
        agree = 0
        for r in rows[:timing_rows]:
            crit = r["questions"]["tool_route"]["criteria"]
            a, _ = model.score(r["state"], crit, media=r.get("media"))
            b, _ = model.score_cached(r["state"], model.encode_catalog(crit), media=r.get("media"))
            agree += int(max(a, key=a.get) == max(b, key=b.get))
        out["cache_agreement"] = agree / min(timing_rows, n)
    print(f"  [{name}] acc {out['acc']:.3f} nll {out['nll']:.3f} ece {out['ece']:.3f} flips {out['flip_rate']:.3f} "
          f"set {out['conformal']['set_size']:.2f}@{out['conformal']['coverage']:.3f} lat {out['latency_ms']:.1f}ms",
          flush=True)
    return out


def _curriculum_k(step: int, steps: int, k_min: int, k_max: int) -> int:
    t = step / max(steps - 1, 1)
    return max(k_min, min(k_max, int(round(k_min + t * (k_max - k_min)))))


def _shrink_menu(row: dict[str, Any], k: int, rng: random.Random) -> dict[str, Any]:
    q = row["questions"]["tool_route"]
    crit = dict(q["criteria"])
    gold = q["label"]
    keep = [gold]
    if "__abstain__" in crit and gold != "__abstain__":
        keep.append("__abstain__")
    rest = [name for name in crit if name not in keep]
    gtoks = set(gold.lower().replace("-", "_").split("_"))
    rest.sort(key=lambda name: -len(gtoks & set(name.lower().replace("-", "_").split("_"))))
    keep.extend(rest[: max(0, k - len(keep))])
    rng.shuffle(keep)
    out = json.loads(json.dumps(row))
    out["questions"]["tool_route"]["criteria"] = {name: crit[name] for name in keep if name in crit}
    return out


def _shuffle_menu(row: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    crit = dict(row["questions"]["tool_route"]["criteria"])
    keys = list(crit)
    rng.shuffle(keys)
    out = json.loads(json.dumps(row))
    out["questions"]["tool_route"]["criteria"] = {name: crit[name] for name in keys}
    return out


def _reverse_menu(row: dict[str, Any]) -> dict[str, Any]:
    crit = dict(row["questions"]["tool_route"]["criteria"])
    out = json.loads(json.dumps(row))
    out["questions"]["tool_route"]["criteria"] = dict(reversed(list(crit.items())))
    return out


def _aligned_order_loss(
    logits_a: list[torch.Tensor],
    keys_a: list[list[str]],
    logits_b: list[torch.Tensor],
    keys_b: list[list[str]],
    logit_weight: float = 0.15,
) -> torch.Tensor:
    """Symmetric JSD + centered-logit MSE between two permutations of the same rows."""
    terms = []
    for lg_a, ka, lg_b, kb in zip(logits_a, keys_a, logits_b, keys_b):
        if len(ka) != len(kb) or set(ka) != set(kb) or len(ka) < 2:
            continue
        index = [kb.index(name) for name in ka]
        la = lg_a.float()
        lb = lg_b[index].float()
        pa = F.softmax(la, dim=-1)
        pb = F.softmax(lb, dim=-1)
        log_pm = (0.5 * (pa + pb)).clamp_min(1e-9).log()
        jsd = 0.5 * (F.kl_div(log_pm, pa, reduction="sum") + F.kl_div(log_pm, pb, reduction="sum"))
        za = la - la.mean()
        zb = lb - lb.mean()
        lmse = ((za - zb) ** 2).mean()
        terms.append(jsd + logit_weight * lmse)
    if not terms:
        return logits_a[0].sum() * 0.0
    return torch.stack(terms).mean()


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
                     load_in_4bit=four_bit, torch_dtype=dtype, media=bool(getattr(args, "media", False)))
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
    steps = args.max_steps if getattr(args, "max_steps", 0) else max(1, len(train_rows) // args.batch)
    use_bf16 = bf16_ok
    use_fp16 = device == "cuda" and not bf16_ok and not four_bit
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    print(f"\n=== {key}: {cfg.name()} | trainable {n_train / 1e6:.2f}M | {steps} steps ===", flush=True)
    model.train()
    t0 = time.time()
    run_loss = run_acc = 0.0
    run_n = skipped = 0
    s1 = bool(getattr(args, "s1", False))
    rng_s1 = random.Random(args.seed + 7)
    for step in range(steps):
        for pg in opt.param_groups:
            pg["lr"] = get_cosine_lr(step, steps, max(1, steps // 20), args.lr)
        start = (step * args.batch) % max(len(train_rows), 1)
        batch = [train_rows[(start + i) % len(train_rows)] for i in range(args.batch)]
        if s1:
            k = _curriculum_k(step, steps, args.k_min, args.k_max)
            batch = [_shrink_menu(r, k, rng_s1) for r in batch]
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        out2 = None
        need_perm_pass = s1 and not cfg.option_isolation
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_bf16 or use_fp16):
            out = model.forward_batch(batch, max_seq_len=args.max_seq_len)
            if need_perm_pass and not out["skipped"]:
                if rng_s1.random() < args.rev_prob:
                    shuf = [_reverse_menu(r) for r in batch]
                else:
                    shuf = [_shuffle_menu(r, rng_s1) for r in batch]
                out2 = model.forward_batch(shuf, max_seq_len=args.max_seq_len)
        if out["skipped"]:
            skipped += 1
            continue
        loss = out["loss"]
        if need_perm_pass and out2 is not None and not out2["skipped"]:
            loss = 0.5 * (out["loss"] + out2["loss"]) + args.order_kl * _aligned_order_loss(
                out["logits"], out["keys"], out2["logits"], out2["keys"], logit_weight=args.order_logit_w
            )
        scaler.scale(loss).backward()
        if scaler.is_enabled():
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if s1 and hasattr(model.head, "log_temp"):
            with torch.no_grad():
                model.head.log_temp.clamp_(min=math.log(max(args.temp_floor, 1e-4)))
        run_loss += float(loss.detach())
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
        if name.startswith("toolret_large") and not (cfg.gists or cfg.interaction == "late"):
            result["eval"][name] = {"skipped": "full-text layouts cannot read menus this large in one pass"}
            continue
        result["eval"][name] = evaluate(model, f"{key}/{name}", rows)
    if getattr(args, "save_ckpt", False):
        ckpt_dir = Path(args.out) / f"ckpt_{key}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        head_state = {
            "head": model.head.state_dict(),
            "cfg": model.config_dict(),
        }
        if model.pool is not None:
            head_state["pool"] = model.pool.state_dict()
        if model.slots is not None:
            head_state["slots"] = model.slots.detach().cpu()
        if model.gist_emb is not None:
            head_state["gist_emb"] = model.gist_emb.detach().cpu()
        torch.save(head_state, ckpt_dir / "head.pt")
        if not model.is_mock and hasattr(model.backbone, "save_pretrained"):
            model.backbone.save_pretrained(ckpt_dir / "adapter")
        print(f"  saved checkpoint to {ckpt_dir}", flush=True)
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
            e = res["eval"].get(split, {})
            if "skipped" in e or not e:
                lines.append(f"| {key} {res['config']} | skipped |" + " |" * 12)
                continue
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


def _media_path(path: str, root: Path | None, index: dict[str, Path]) -> str | None:
    """Rows store paths relative to the run that made them; find the file wherever it is now."""
    cand = Path(path)
    if cand.exists():
        return str(cand)
    if root is not None and (root / path).exists():
        return str(root / path)
    hit = index.get(cand.name)
    return str(hit) if hit else None


def load_media_rows(paths: str, root: str | None, seed: int = 0) -> tuple[list, dict[str, list]]:
    """Omni rows (soro.omni_data schema) with audio or an image -> P0 records.

    Returns (train records, eval splits). Eval rows are the build's held-out splits (group-disjoint)."""
    rootp = Path(root) if root else None
    index: dict[str, Path] = {}
    if rootp is not None and rootp.exists():
        for f in rootp.rglob("*"):
            if f.suffix.lower() in (".wav", ".flac", ".mp3", ".png", ".jpg", ".jpeg", ".webp"):
                index.setdefault(f.name, f)
    train, speech, speech_text, ui = [], [], [], []
    missing = 0
    for path in [x for x in paths.split(",") if x.strip()]:
        for r in read_jsonl(Path(path.strip())):
            kind = "audio" if r.get("audio") else ("image" if r.get("image") else None)
            if kind is None:
                continue
            if "derived" in r.get("tags", []) and r.get("meta", {}).get("prefix_fraction"):
                continue  # streaming prefixes are a separate experiment
            fp = _media_path(r[kind], rootp, index)
            if fp is None:
                missing += 1
                continue
            crit = {o["key"]: o["text"] for o in r["options"]}
            if r.get("label") not in crit:
                continue
            ctx = f"Context: {r['context']}\n" if r.get("context") else ""
            text = r.get("text") or ""
            state = ctx + (f"User Request: {text}" if text else "User Request: (spoken request in the input)"
                           if kind == "audio" else "User Request: (see the screenshot)")
            rec = {"id": r["id"], "state": state, "media": {kind: fp}, "source": r.get("source"),
                   "questions": {"tool_route": {"criteria": crit, "label": r["label"]}}}
            if r.get("split", "train") == "train":
                train.append(rec)
            elif kind == "audio":
                speech.append(rec)
                if r.get("transcript"):
                    speech_text.append(dict(rec, state=ctx + f"User Request: {r['transcript']}", media=None))
            else:
                ui.append(rec)
    rng = random.Random(seed)
    for xs in (train, speech, ui):
        rng.shuffle(xs)
    print(f"media rows: train {len(train)} | speech {len(speech)} | speech_text {len(speech_text)} | ui {len(ui)} "
          f"| missing files {missing}", flush=True)
    return train, {"speech": speech, "speech_text": speech_text, "ui": ui}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default="A,B,C,D,E")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--train-rows", type=int, default=4000)
    ap.add_argument("--max-steps", type=int, default=0, help="Cap optimizer steps (0 = one pass over train-rows).")
    ap.add_argument("--eval-rows", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq-len", type=int, default=832)  # audit: max prompt 791 tokens
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--out", default=str(REPO / "runs" / "p0_upp"))
    ap.add_argument("--data-dir", default=None, help="Folder with train.jsonl and val_zeroshot.jsonl")
    ap.add_argument("--toolret", type=int, default=0, help="Add N ToolRet queries as eval splits (0 = off)")
    ap.add_argument("--toolret-k", type=int, default=16, help="Menu size for the main ToolRet split")
    ap.add_argument("--toolret-k-large", type=int, default=0, help="Extra ToolRet split with a larger menu (0 = off)")
    ap.add_argument("--toolret-large-n", type=int, default=300, help="Queries in the large-menu ToolRet split")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--no-grad-ckpt", action="store_true", help="Disable gradient checkpointing (faster, more VRAM)")
    ap.add_argument("--s1", action="store_true",
                    help="Pointer-A S1: menu shuffle/reverse + symmetric order JSD/logit loss, curriculum 4-64, temp floor, no option index.")
    ap.add_argument("--temp-floor", type=float, default=0.05)
    ap.add_argument("--order-kl", type=float, default=1.0)
    ap.add_argument("--order-logit-w", type=float, default=0.15)
    ap.add_argument("--rev-prob", type=float, default=0.75)
    ap.add_argument("--keep-option-index", action="store_true", help="Keep [i] option prefix even under --s1.")
    ap.add_argument("--isolate-options", action="store_true",
                    help="S1 v3: mask option-to-option attention in query_first & share option start position (exact order invariance).")
    ap.add_argument("--save-ckpt", action="store_true", help="Save trained LoRA adapter and UPP head to --out/ckpt_<key>.")
    ap.add_argument("--k-min", type=int, default=4)
    ap.add_argument("--k-max", type=int, default=64)
    ap.add_argument("--smoke", action="store_true", help="Tiny CPU run with a mock backbone")
    ap.add_argument("--media", action="store_true", help="Keep the omni backbone's audio/vision towers")
    ap.add_argument("--media-rows", default="", help="Comma-separated omni jsonl files with audio/image rows")
    ap.add_argument("--media-root", default=None, help="Where the rows' wav/png files live now (searched by name)")
    ap.add_argument("--media-train", type=int, default=2000, help="Media rows mixed into training (0 = eval only)")
    args = ap.parse_args()
    if args.s1:
        args.configs = "A"

    data_dir = resolve_data_dir(args.data_dir)
    train_all = read_jsonl(data_dir / "train.jsonl")
    train_all, selection = split_selection_packages(train_all, frac=0.05, seed=args.seed)
    rng = random.Random(args.seed)
    rng.shuffle(train_all)
    zs_path = next((p for p in (
        data_dir / "val_zeroshot.jsonl",
        data_dir / "test.jsonl",
        data_dir.parent / "stage_text" / "eval_zeroshot.jsonl",
    ) if p.exists()), None)
    if zs_path is None:
        raise FileNotFoundError(f"No val_zeroshot.jsonl or test.jsonl under {data_dir}")
    zs = read_jsonl(zs_path)
    if args.eval_rows > 0:
        stride = max(1, len(zs) // args.eval_rows)
        sel_eval = selection[: args.eval_rows]
        zs_eval = zs[::stride][: args.eval_rows]
    else:
        sel_eval = selection
        zs_eval = zs
    evals = {
        "selection": sel_eval,
        "zeroshot": zs_eval,
        "bfcl": load_bfcl(),
    }
    toolret_stats = {}
    if args.toolret and not args.smoke:
        from toolret_data import build as build_toolret

        for name, k in (("toolret", args.toolret_k), ("toolret_large", args.toolret_k_large)):
            if not k:
                continue
            recs, st = build_toolret(n=args.toolret if name == "toolret" else args.toolret_large_n, k=k)
            toolret_stats[name] = st
            # pointer accuracy is measured where the gold tool made it into the menu
            evals[f"{name}_k{k}"] = [r for r in recs if r["gold_in_menu"]]
            print(f"{name}: k={k} recall@k={st['recall_at_k']:.3f} first-stage acc@1={st['first_stage_acc1']:.3f} "
                  f"train overlap={st['train_query_overlap']}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    media_train: list[dict[str, Any]] = []
    if args.media_rows and not args.smoke:
        media_train, media_evals = load_media_rows(args.media_rows, args.media_root, seed=args.seed)
        for name, rows_m in media_evals.items():
            evals[name] = rows_m[: args.eval_rows]
        if evals.get("speech_text"):  # keep the paired text rows aligned with the speech rows kept
            keep = {r["id"] for r in evals.get("speech", [])}
            evals["speech_text"] = [r for r in media_evals["speech_text"] if r["id"] in keep]
    evals = {k: v for k, v in evals.items() if v}
    train_rows = train_all[: args.train_rows]
    if evals.get("speech_text"):
        # the spoken eval requests are TTS of text requests; drop their text twins from training
        said = {r["state"].split("User Request: ", 1)[-1].strip().lower() for r in evals["speech_text"]}
        said = {t for t in said if len(t) > 8}
        before = len(train_rows)
        train_rows = [r for r in train_rows if not any(t in r["state"].lower() for t in said)]
        print(f"dropped {before - len(train_rows)} train rows that repeat a spoken eval request", flush=True)
    if media_train and args.media_train:
        train_rows = train_rows + media_train[: args.media_train]
        random.Random(args.seed).shuffle(train_rows)
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
        cfg = CONFIGS[key]
        if args.s1 and not getattr(args, "keep_option_index", False):
            cfg = replace(cfg, option_index=False)
        if getattr(args, "isolate_options", False):
            cfg = replace(cfg, option_index=(cfg.option_index and not args.s1 and getattr(args, "keep_option_index", False)),
                          option_isolation=True)
        results[key] = train_one(key, cfg, args, train_rows, evals)
        if toolret_stats:
            results[key]["toolret_first_stage"] = toolret_stats
        (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    md = markdown(results)
    (out_dir / "results.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
