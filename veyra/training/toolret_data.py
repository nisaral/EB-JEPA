"""ToolRet as a decision benchmark (ACL Findings 2025; 35 task sets, 43k tools).

ToolRet is a retrieval benchmark: a query and a corpus of tools (web / code /
customized). A pointer model decides over a menu, so the harness:

1. retrieves the top-K tools per query from the query's corpus with a cheap
   first stage (all-MiniLM-L6-v2 embeddings, cached per corpus),
2. builds a menu of those K tools plus ``__abstain__``,
3. reports first-stage Recall@K separately. Pointer accuracy is measured only
   on queries whose gold tool is in the menu; end-to-end accuracy is
   Recall@K x conditional accuracy. No gold tool is ever injected.

The ``toolace`` subset is excluded by default because Soro training rows were
converted from ToolACE. Query overlap with the Soro train file is reported.

    python veyra/training/toolret_data.py --n 600 --k 16 --out runs/toolret/toolret_k16.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
QUERY_REPO = "mangopy/ToolRet-Queries"
TOOL_REPO = "mangopy/ToolRet-Tools"
CATEGORIES = ("web", "code", "customized")
DEFAULT_EXCLUDE = ("toolace",)


def _hf(repo: str, path: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, path, repo_type="dataset")


def _subsets() -> list[str]:
    from huggingface_hub import list_repo_files

    return sorted({f.split("/")[0] for f in list_repo_files(QUERY_REPO, repo_type="dataset") if f.endswith(".parquet")})


def render_tool(doc: Any) -> tuple[str, str]:
    """(name, text) from a ToolRet documentation field (JSON string or dict)."""
    if isinstance(doc, str):
        try:
            doc = json.loads(doc)
        except json.JSONDecodeError:
            return "tool", doc[:400]
    name = str(doc.get("name") or doc.get("api_name") or doc.get("tool_name") or "tool")
    desc = str(doc.get("description") or doc.get("desc") or "")
    params = doc.get("parameters") or doc.get("arguments") or {}
    if isinstance(params, dict):
        props = params.get("properties", params)
        pnames = ", ".join(list(props)[:8]) if isinstance(props, dict) else ""
    elif isinstance(params, list):
        pnames = ", ".join(str(p.get("name", p)) if isinstance(p, dict) else str(p) for p in params[:8])
    else:
        pnames = str(params)[:120]
    text = desc + (f" Parameters: {pnames}." if pnames else "")
    return name, re.sub(r"\s+", " ", text).strip()[:400]


def load_queries(n: int, exclude: tuple[str, ...], seed: int) -> list[dict[str, Any]]:
    import pandas as pd

    rows = []
    for sub in _subsets():
        if sub in exclude:
            continue
        df = pd.read_parquet(_hf(QUERY_REPO, f"{sub}/queries-00000-of-00001.parquet"))
        for r in df.to_dict("records"):
            labels = json.loads(r["labels"]) if isinstance(r["labels"], str) else r["labels"]
            rows.append({"id": r["id"], "subset": sub, "query": r["query"], "instruction": r.get("instruction", ""),
                         "category": r["category"], "gold_ids": [l["id"] for l in labels]})
    rng = random.Random(seed)
    rng.shuffle(rows)
    # stratify: round-robin over subsets so small task sets are represented
    by_sub: dict[str, list] = {}
    for r in rows:
        by_sub.setdefault(r["subset"], []).append(r)
    out, i = [], 0
    while len(out) < n and any(by_sub.values()):
        for sub in list(by_sub):
            if by_sub[sub] and len(out) < n:
                out.append(by_sub[sub].pop())
        i += 1
    return out


def load_corpus(category: str) -> dict[str, tuple[str, str]]:
    import pandas as pd

    df = pd.read_parquet(_hf(TOOL_REPO, f"{category}/tools-00000-of-00001.parquet"))
    return {r["id"]: render_tool(r["documentation"]) for r in df.to_dict("records")}


def build(n: int = 600, k: int = 16, exclude: tuple[str, ...] = DEFAULT_EXCLUDE, seed: int = 0,
          device: str | None = None, cache_dir: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    from sentence_transformers import SentenceTransformer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = cache_dir or (REPO / "runs" / "toolret")
    cache_dir.mkdir(parents=True, exist_ok=True)
    queries = load_queries(n, exclude, seed)
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    records, stats = [], {"queries": len(queries), "k": k, "gold_in_menu": 0, "first_stage_acc1": 0}
    for cat in CATEGORIES:
        qs = [q for q in queries if q["category"] == cat]
        if not qs:
            continue
        corpus = load_corpus(cat)
        ids = list(corpus)
        emb_path = cache_dir / f"minilm_{cat}.pt"
        if emb_path.exists():
            emb = torch.load(emb_path, weights_only=True).to(device)
        else:
            emb = enc.encode([f"{corpus[i][0]}: {corpus[i][1]}" for i in ids], convert_to_tensor=True,
                             normalize_embeddings=True, batch_size=256, show_progress_bar=False)
            torch.save(emb.cpu(), emb_path)
        qemb = enc.encode([q["query"] for q in qs], convert_to_tensor=True, normalize_embeddings=True,
                          batch_size=256, show_progress_bar=False)
        top = (qemb @ emb.T).topk(k, dim=1).indices.cpu().tolist()
        for q, idxs in zip(qs, top):
            cand_ids = [ids[i] for i in idxs]
            menu, key_of = {}, {}
            for tid in cand_ids:
                name, text = corpus[tid]
                key = name
                j = 2
                while key in menu:  # same display name for different tools: keep both, distinguishable
                    key = f"{name}_{j}"
                    j += 1
                menu[key] = text
                key_of[tid] = key
            gold_keys = [key_of[g] for g in q["gold_ids"] if g in key_of]
            stats["gold_in_menu"] += bool(gold_keys)
            stats["first_stage_acc1"] += bool(cand_ids and cand_ids[0] in q["gold_ids"])
            menu["__abstain__"] = "None of the listed tools fit the user request."
            records.append({
                "id": q["id"], "package": f"toolret/{q['subset']}", "state": f"User Request: {q['query']}",
                "questions": {"tool_route": {"type": "choice", "criteria": menu,
                                             "label": gold_keys[0] if gold_keys else "__missing__"}},
                "gold_functions": gold_keys, "toolret_subset": q["subset"], "toolret_category": cat,
                "gold_in_menu": bool(gold_keys),
            })
    nq = max(1, stats["queries"])
    stats["recall_at_k"] = stats["gold_in_menu"] / nq
    stats["first_stage_acc1"] = stats["first_stage_acc1"] / nq
    stats["excluded_subsets"] = list(exclude)
    stats["subsets"] = len({r["toolret_subset"] for r in records})
    stats["train_query_overlap"] = _overlap_with_train([q["query"] for q in queries])
    del enc
    return records, stats


def _overlap_with_train(queries: list[str]) -> int | None:
    path = REPO / "releases" / "soro-universal-tool-harness" / "train.jsonl"
    if not path.exists():
        return None
    norm = lambda s: " ".join(re.findall(r"[a-z0-9]+", s.lower()))  # noqa: E731
    qs = {norm(q) for q in queries}
    hit = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            st = json.loads(line).get("state", "")
            if "User Request:" in st and norm(st.split("User Request:", 1)[1].split("\n")[0]) in qs:
                hit += 1
    return hit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--exclude", default=",".join(DEFAULT_EXCLUDE))
    ap.add_argument("--out", default=str(REPO / "runs" / "toolret" / "toolret.jsonl"))
    a = ap.parse_args()
    recs, stats = build(a.n, a.k, tuple(s for s in a.exclude.split(",") if s))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
