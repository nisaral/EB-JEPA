"""Check candidate parent backbones for the UPP architecture before any training.

For each model it loads the backbone with LoRA, builds an untrained UPP head with
the full block layout (block + rbf + refiner + ensemble + early exit), then checks:

  * block isolation works: reversing the option order leaves probabilities unchanged
  * the catalog KV cache is exact: cached and uncached scores match
  * early-exit readouts are produced
  * peak VRAM and median latency (cold and cached) on 20 real rows

A model with linear-attention layers (e.g. Qwen3.5 Gated DeltaNet) is reported as
incompatible with the block layout and checked with query_first instead.

Usage (Kaggle / Colab GPU):
  python veyra/training/parent_check.py --models Qwen/Qwen3-0.6B-Base,Qwen/Qwen3-1.7B-Base,google/gemma-4-E2B
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "veyra" / "python" / "src"))
sys.path.insert(0, str(REPO / "veyra" / "training"))

from p0_upp_ablation import read_jsonl, resolve_data_dir  # noqa: E402
from soro.routing.upp import UPPConfig, UPPModel  # noqa: E402

FULL = UPPConfig(layout="block", head="rbf", refiner=True, ensemble=4, exits=(0.5, 0.75))


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def check(model_name: str, rows: list[dict], load_in_4bit: bool) -> dict:
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        model = UPPModel(FULL, model_name=model_name, load_in_4bit=load_in_4bit)
        layout = "block"
    except ValueError as exc:  # linear attention: block layout impossible
        print(f"  {exc}")
        model = UPPModel(UPPConfig(layout="query_first", head="rbf", ensemble=4, exits=(0.5, 0.75)),
                         model_name=model_name, load_in_4bit=load_in_4bit)
        layout = "query_first (block unsupported)"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if load_in_4bit and device == "cuda":
        model.head.to(device)
    else:
        model.to(device)
    model.eval()
    load_s = time.time() - t0
    res = {"model": model_name, "layout": layout, "load_secs": round(load_s, 1), "4bit": load_in_4bit}

    crit = rows[0]["questions"]["tool_route"]["criteria"]
    st = rows[0]["state"]
    a, info = model.score(st, crit)
    b, _ = model.score(st, dict(reversed(list(crit.items()))))
    res["order_max_abs_diff"] = max(abs(a[k] - b[k]) for k in a)
    res["exit_readouts"] = len(info.get("exits", []))
    if layout == "block":
        c, _ = model.score_cached(st, model.encode_catalog(crit))
        res["cache_max_abs_diff"] = max(abs(a[k] - c[k]) for k in a)

    cold, cached = [], []
    for r in rows[:20]:
        crit = r["questions"]["tool_route"]["criteria"]
        _sync()
        t = time.perf_counter()
        model.score(r["state"], crit)
        _sync()
        cold.append((time.perf_counter() - t) * 1000)
        if layout == "block":
            cache = model.encode_catalog(crit)
            _sync()
            t = time.perf_counter()
            model.score_cached(r["state"], cache)
            _sync()
            cached.append((time.perf_counter() - t) * 1000)
    res["latency_ms_cold"] = round(statistics.median(cold), 1)
    if cached:
        res["latency_ms_cached"] = round(statistics.median(cached), 1)
    if torch.cuda.is_available():
        res["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
    res["trainable_params_m"] = round(sum(p.numel() for p in model.trainable_parameters()) / 1e6, 2)
    del model
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="Qwen/Qwen3-0.6B-Base,Qwen/Qwen3-1.7B-Base,google/gemma-4-E2B")
    ap.add_argument("--four-bit", default="google/gemma-4-E2B,google/gemma-4-E4B",
                    help="Comma list of models to load in 4-bit")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default=str(REPO / "runs" / "parent_check.json"))
    args = ap.parse_args()
    rows = read_jsonl(resolve_data_dir(args.data_dir) / "val_zeroshot.jsonl")[::100]
    four = {m.strip() for m in args.four_bit.split(",") if m.strip()}
    results = []
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n=== {name} ===", flush=True)
        try:
            r = check(name, rows, name in four)
        except Exception as exc:  # report and continue with the next candidate
            r = {"model": name, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(r, indent=2), flush=True)
        results.append(r)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
