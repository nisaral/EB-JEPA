"""Full data audit for Omni rows (any split, any modality).

    python -m soro.omni_data.audit runs/omni_data/stage_text/*.jsonl --out runs/omni_data/audit --tokenizer Qwen/Qwen3-0.6B-Base

Checks, per file and across files:
  schema        invalid rows with reasons
  labels        gold not in menu, abstain / need-input share, label-key oddities (delexicalized op_N names)
  menus         size distribution, duplicate option texts, options with empty or very short descriptions
  text          empty / very short / very long requests, non-ASCII share, prompt-injection-looking requests
  duplicates    exact duplicates within a file, exact request overlap across files
  noise         template requests reused across packages (suspect labels)
  leakage       train vs every test file (exact + near duplicate requests, shared packages)
  length        token length of the full model prompt; share that would be dropped at 512/768/1024/1536 tokens
  sources       rows per source / license / modality / package; files with eval-only licenses in train
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from soro.omni_data import licenses, quality
from soro.omni_data.schema import ABSTAIN, NEED_INPUT, OmniRow, read_rows

_INJECT = re.compile(r"ignore (all|previous|the above)|system prompt|you are now|disregard", re.I)
_DELEX = re.compile(r"^op_\d+$")


def _prompt_text(r: OmniRow) -> str:
    menu = "\n".join(f"[{i}] {o.key}: {o.text[:400]}" for i, o in enumerate(r.options))
    return f"State: {r.state()}\nQuestion [tool_route]: Which single tool?\nOptions:\n{menu}"


def audit_file(rows: list[OmniRow], tok=None) -> dict[str, Any]:
    n = len(rows) or 1
    ok, bad = quality.validate_rows(rows)
    menu_sizes = [len([o for o in r.options if o.kind != "special"]) for r in rows]
    dup_opt_text = sum(len({o.text.strip().lower() for o in r.options}) < len(r.options) for r in rows)
    short_desc = sum(any(o.kind == "tool" and len(o.text.strip()) < 8 for o in r.options) for r in rows)
    texts = [r.text or r.transcript or "" for r in rows]
    lens = [len(t) for t in texts]
    hashes = Counter(r.content_hash() for r in rows)
    if tok is not None:
        toks = [len(tok.encode(_prompt_text(r), add_special_tokens=False)) for r in rows]
    else:
        toks = [len(_prompt_text(r)) // 4 for r in rows]  # rough chars/4 estimate
    toks_sorted = sorted(toks)
    return {
        "rows": len(rows),
        "invalid": len(bad), "invalid_examples": bad[:5],
        "labels": {
            "abstain_share": sum(r.label == ABSTAIN for r in rows) / n,
            "need_input_share": sum(r.label == NEED_INPUT for r in rows) / n,
            "delexicalized_gold_share": sum(bool(_DELEX.match(r.label)) for r in rows) / n,
            "distinct_gold": len({r.label for r in rows}),
        },
        "menus": {
            "size_min": min(menu_sizes, default=0), "size_median": statistics.median(menu_sizes) if menu_sizes else 0,
            "size_max": max(menu_sizes, default=0),
            "rows_with_duplicate_option_texts": dup_opt_text,
            "rows_with_near_empty_descriptions": short_desc,
        },
        "text": {
            "empty": sum(l == 0 for l in lens) if any(r.modality == "text" for r in rows) else None,
            "very_short_lt10": sum(0 < l < 10 for l in lens), "very_long_gt1500": sum(l > 1500 for l in lens),
            "median_chars": statistics.median(lens) if lens else 0,
            "non_ascii_share": sum(any(ord(c) > 127 for c in t) for t in texts) / n,
            "injection_like": sum(bool(_INJECT.search(t)) for t in texts),
        },
        "duplicates": {"exact_rows": sum(c - 1 for c in hashes.values() if c > 1)},
        "length_tokens": {
            "estimated": tok is None,
            "p50": toks_sorted[len(toks) // 2] if toks else 0, "p95": toks_sorted[int(len(toks) * 0.95)] if toks else 0,
            "max": toks_sorted[-1] if toks else 0,
            **{f"over_{L}": sum(t > L for t in toks) / n for L in (512, 768, 1024, 1536)},
        },
        "sources": {
            "by_source": dict(Counter(r.source for r in rows).most_common(10)),
            "by_license": dict(Counter(r.license for r in rows)),
            "by_modality": dict(Counter(r.modality for r in rows)),
            "eval_only_license_in_train": sum(r.split in ("train", "selection") and not licenses.trainable(r.license)
                                              for r in rows),
            "packages": len({r.package for r in rows if r.package}),
        },
        "tags": dict(Counter(t for r in rows for t in r.tags if ":" not in t).most_common(15)),
    }


def audit(paths: list[str], tokenizer: str | None = None) -> dict[str, Any]:
    tok = None
    if tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer)
    missing = [p for p in paths if not Path(p).exists()]
    files = {Path(p).stem: list(read_rows(p)) for p in paths if Path(p).exists()}
    report: dict[str, Any] = {"missing_files": missing,
                              "files": {k: audit_file(v, tok) for k, v in files.items() if v}}
    train = [r for rows in files.values() for r in rows if r.split in ("train", "selection")]
    tests = defaultdict(list)
    for name, rows in files.items():
        for r in rows:
            if r.split not in ("train", "selection"):
                tests[name].append(r)
    if train and tests:
        lk = quality.leakage(train, dict(tests))
        report["leakage"] = {k: {kk: v[kk] for kk in ("rows", "exact_query_overlap", "near_query_overlap",
                                                      "n_shared_packages")} for k, v in lk.items()}
    all_rows = [r for rows in files.values() for r in rows]
    report["template_noise"] = quality.generic_requests([OmniRow.from_json(r.to_json()) for r in all_rows])
    # cross-file exact request overlap
    norm_files = {k: {quality.normalize(r.text) for r in v if r.text} for k, v in files.items()}
    names = list(norm_files)
    report["cross_file_request_overlap"] = {
        f"{a}~{b}": len(norm_files[a] & norm_files[b]) for i, a in enumerate(names) for b in names[i + 1:]
        if norm_files[a] & norm_files[b]
    }
    report["teacher_label_audit"] = quality.label_audit(all_rows)
    return report


def to_markdown(rep: dict[str, Any]) -> str:
    lines = ["# Omni data audit", "", "| File | Rows | Invalid | Abstain | Delex gold | Menu med | Exact dup | "
             "Tok p95 | >768 tok | >1024 tok | Inject-like |", "| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |"]
    for k, f in rep["files"].items():
        lt = f["length_tokens"]
        lines.append(f"| {k} | {f['rows']} | {f['invalid']} | {f['labels']['abstain_share']:.1%} | "
                     f"{f['labels']['delexicalized_gold_share']:.1%} | {f['menus']['size_median']} | "
                     f"{f['duplicates']['exact_rows']} | {lt['p95']} | {lt['over_768']:.1%} | {lt['over_1024']:.1%} | "
                     f"{f['text']['injection_like']} |")
    lines += ["", "## Leakage", "```", json.dumps(rep.get("leakage", {}), indent=1), "```",
              "", "## Template noise", "```", json.dumps(rep["template_noise"], indent=1)[:2500], "```",
              "", "## Cross-file request overlap", "```", json.dumps(rep["cross_file_request_overlap"], indent=1), "```"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default=None)
    a = ap.parse_args()
    rep = audit(a.paths, a.tokenizer)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit_report.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    (out / "AUDIT.md").write_text(to_markdown(rep), encoding="utf-8")
    print(to_markdown(rep))


if __name__ == "__main__":
    main()
