"""Architecture bench: exactness checks and latency/memory scaling (inference only, no training).

For each layout it measures, on a real backbone:
  * order invariance   max |p - p(reversed menu)|
  * cache exactness    max |p(full forward) - p(cached catalog)|
  * composition        max |p(full forward) - p(per-tool caches joined in shuffled order)|
  * latency per request at menu sizes N (cached where the layout allows it)
  * cache size per tool (MB) and one-time encode cost per tool

Heads are untrained: none of these quantities depend on training.

    python veyra/training/arch_bench.py --base-model Qwen/Qwen2.5-0.5B --sizes 6,16,64,256
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "veyra" / "python" / "src"))

from soro.routing.upp import UPPConfig, UPPModel  # noqa: E402

VARIANTS = {
    "A_kev_query_first": UPPConfig(layout="query_first"),
    "E_catalog_first": UPPConfig(layout="catalog_first"),
    "F_block_full": UPPConfig(layout="block"),
    "P_block_attnpool": UPPConfig(layout="block", option_pool="attn"),
    "K_block_gist4": UPPConfig(layout="block", gists=4),
    "K8_block_gist8": UPPConfig(layout="block", gists=8),
    "L_late_gist4": UPPConfig(layout="block", gists=4, interaction="late"),
    "KM_matryoshka_gist8": UPPConfig(layout="block", gists=8, gist_budgets=(1, 2, 4, 8)),
}


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(fn, reps=5):
    out = []
    for _ in range(reps):
        _sync()
        t = time.perf_counter()
        fn()
        _sync()
        out.append((time.perf_counter() - t) * 1000)
    return statistics.median(out)


def tool_pool(n: int) -> dict[str, str]:
    src = REPO / "runs" / "omni_data" / "stage_text" / "text_base.jsonl"
    tools: dict[str, str] = {}
    if src.exists():
        with src.open(encoding="utf-8") as f:
            for line in f:
                for o in json.loads(line)["options"]:
                    if o["kind"] == "tool" and o["key"] not in tools:
                        tools[o["key"]] = o["text"]
                if len(tools) >= n:
                    break
    i = 0
    while len(tools) < n:  # synthetic filler if the stage output is missing
        tools[f"tool_{i}"] = f"Performs operation number {i} on the user's data and returns a result."
        i += 1
    return dict(list(tools.items())[:n])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--sizes", default="6,16,64,256")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--max-full-tokens", type=int, default=12000, help="skip uncached layouts beyond this prompt size")
    ap.add_argument("--max-block-tokens", type=int, default=4000, help="skip full-text block layouts beyond this (4D mask)")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"],
                    help="fp32 for exactness checks; bf16/fp16 for latency")
    ap.add_argument("--out", default=str(REPO / "runs" / "arch_bench.json"))
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sizes = [int(s) for s in a.sizes.split(",")]
    pool = tool_pool(max(sizes))
    request = "User Request: will it rain in Pune tomorrow morning?"
    results = {}
    for name in [v.strip() for v in a.variants.split(",") if v.strip()]:
        cfg = VARIANTS[name]
        torch.manual_seed(0)
        dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.dtype]
        if dt is torch.bfloat16 and device == "cuda" and not torch.cuda.is_bf16_supported():
            dt = torch.float16
        m = UPPModel(cfg, model_name=a.base_model, torch_dtype=dt)
        m.to(device)
        m.backbone.to(dt)  # heads stay fp32, as in training
        m.eval()
        r: dict = {"config": cfg.name()}
        small = dict(list(pool.items())[:6])
        small["__abstain__"] = "None of the listed tools fit the request."
        p, _ = m.score(request, small)
        p_rev, _ = m.score(request, dict(reversed(list(small.items()))))
        r["order_max_abs_diff"] = max(abs(p[k] - p_rev[k]) for k in p)
        if cfg.layout in ("catalog_first", "block"):
            pc, _ = m.score_cached(request, m.encode_catalog(small))
            r["cache_max_abs_diff"] = max(abs(p[k] - pc[k]) for k in p)
        if cfg.layout == "block" and (cfg.gists or cfg.interaction == "late"):
            tools = [m.encode_tool(k, v) for k, v in small.items()]
            random.Random(1).shuffle(tools)
            pcomp, _ = m.score_composed(request, m.compose(tools))
            r["compose_max_abs_diff"] = max(abs(p[k] - pcomp[k]) for k in p)
            t1 = tools[0]
            if t1["past"] is not None:
                bytes_tool = sum((l.keys[:, :, t1["prefix_len"]:].numel() + l.values[:, :, t1["prefix_len"]:].numel())
                                 * l.keys.element_size() for l in t1["past"].layers)
            else:
                bytes_tool = t1["opt_state"].numel() * t1["opt_state"].element_size()
            r["cache_mb_per_tool"] = round(bytes_tool / 2 ** 20, 4)
            k0, v0 = next(iter(pool.items()))
            r["encode_ms_per_tool"] = round(_timed(lambda: m.encode_tool(k0, v0), 3), 1)
        r["latency_ms"] = {}
        for n in sizes:
            menu = dict(list(pool.items())[:n])
            menu["__abstain__"] = "None of the listed tools fit the request."
            if cfg.layout == "block" and (cfg.gists or cfg.interaction == "late"):
                # per-tool caches composed into one catalog: how a deployment would hold 1000+ tools
                t0 = time.perf_counter()
                cat = m.compose([m.encode_tool(k, v) for k, v in menu.items()])
                r.setdefault("catalog_build_s", {})[n] = round(time.perf_counter() - t0, 2)
                r.setdefault("cached_tokens", {})[n] = cat["cached_tokens"]
                r["latency_ms"][n] = round(_timed(lambda: m.score_composed(request, cat)), 1)
                if cfg.gist_budgets:
                    if n == sizes[0]:
                        full, _ = m.score_composed(request, cat)
                        two, _ = m.score_two_level(request, cat, coarse=cfg.gists, top_m=len(cat["keys"]))
                        r["two_level_exact_max_abs_diff"] = max(abs(full[k] - two[k]) for k in full)
                    r.setdefault("two_level_ms", {})[n] = round(
                        _timed(lambda: m.score_two_level(request, cat, coarse=1, top_m=16)), 1)
            elif cfg.layout in ("catalog_first", "block") and (
                    len(m.build(request, menu)["ids"]) > (a.max_block_tokens if cfg.layout == "block" else a.max_full_tokens)):
                r["latency_ms"][n] = f"skipped ({len(m.build(request, menu)['ids'])} tokens of full option text)"
                continue
            elif cfg.layout in ("catalog_first", "block"):
                cat = m.encode_catalog(menu)
                r["latency_ms"][n] = round(_timed(lambda: m.score_cached(request, cat)), 1)
            else:
                n_tok = len(m.build(request, menu)["ids"])
                if n_tok > a.max_full_tokens:
                    r["latency_ms"][n] = f"skipped ({n_tok} tokens)"
                    continue
                r["latency_ms"][n] = round(_timed(lambda: m.score(request, menu), 3), 1)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        results[name] = r
        print(name, json.dumps(r), flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=2), encoding="utf-8")  # keep partial results
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
