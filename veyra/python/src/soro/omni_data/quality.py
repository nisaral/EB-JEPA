"""Quality gates: validation, exact and near-duplicate removal, leakage, balance, label audit.

Near duplicates use MinHash over word 3-shingles with LSH banding (pure Python,
deterministic). Leakage checks compare train against every eval split on
request text and on packages.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from typing import Any, Iterable

from soro.omni_data.schema import ABSTAIN, NEED_INPUT, OmniRow

_WORD = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    return " ".join(_WORD.findall((text or "").lower()))


def shingles(text: str, n: int = 3) -> set[str]:
    words = normalize(text).split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


class MinHashLSH:
    """MinHash signatures + banding. ``threshold`` is the target Jaccard."""

    def __init__(self, num_perm: int = 64, bands: int = 16):
        assert num_perm % bands == 0
        self.num_perm, self.bands, self.rows = num_perm, bands, num_perm // bands
        self.seeds = [int(hashlib.md5(str(i).encode()).hexdigest()[:8], 16) for i in range(num_perm)]
        self.buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)

    def signature(self, sh: set[str]) -> tuple[int, ...]:
        if not sh:
            return tuple([0] * self.num_perm)
        hashes = [int(hashlib.md5(s.encode()).hexdigest()[:8], 16) for s in sh]
        return tuple(min((h ^ seed) for h in hashes) for seed in self.seeds)

    def candidates_and_add(self, idx: int, sig: tuple[int, ...]) -> set[int]:
        found: set[int] = set()
        for b in range(self.bands):
            key = (b, sig[b * self.rows:(b + 1) * self.rows])
            found.update(self.buckets[key])
            self.buckets[key].append(idx)
        return found


def validate_rows(rows: Iterable[OmniRow]) -> tuple[list[OmniRow], list[tuple[str, list[str]]]]:
    ok, bad = [], []
    for r in rows:
        errs = r.validate()
        (bad.append((r.id, errs)) if errs else ok.append(r))
    return ok, bad


def dedupe(rows: list[OmniRow], near_threshold: float = 0.85, max_per_group_label: int | None = None,
           ) -> tuple[list[OmniRow], dict[str, int]]:
    """Drop exact repeats, then near-duplicate requests that share the same menu and label.

    ``max_per_group_label`` caps paraphrase floods: at most N rows per (package, label).
    """
    stats = Counter()
    seen: set[str] = set()
    exact: list[OmniRow] = []
    for r in rows:
        h = r.content_hash()
        if h in seen:
            stats["exact_dupes"] += 1
            continue
        seen.add(h)
        exact.append(r)

    lsh = MinHashLSH()
    sh_cache: list[set[str]] = []
    kept_at: dict[int, OmniRow] = {}  # LSH index -> kept row
    keep: list[OmniRow] = []
    for r in exact:
        sh = shingles(r.text or r.transcript or "")
        idx = len(sh_cache)
        sh_cache.append(sh)
        menu = sorted(o.key for o in r.options)
        dup = False
        for c in lsh.candidates_and_add(idx, lsh.signature(sh)):
            other = kept_at.get(c)
            if (other is not None and other.label == r.label and other.image == r.image
                    and other.modality == r.modality
                    and sorted(o.key for o in other.options) == menu and jaccard(sh, sh_cache[c]) >= near_threshold):
                dup = True
                break
        if dup:
            stats["near_dupes"] += 1
            continue
        kept_at[idx] = r
        keep.append(r)

    if max_per_group_label:
        counts: Counter = Counter()
        capped = []
        for r in keep:
            key = (r.group, r.label) if r.task == "ui_point" else (r.package or r.source, r.label)
            if counts[key] >= max_per_group_label:
                stats["capped"] += 1
                continue
            counts[key] += 1
            capped.append(r)
        keep = capped
    stats["kept"] = len(keep)
    return keep, dict(stats)


def leakage(train: list[OmniRow], evals: dict[str, list[OmniRow]], threshold: float = 0.75) -> dict[str, Any]:
    """Near-duplicate requests and shared packages between train and each eval split."""
    lsh = MinHashLSH()
    tr_sh = []
    for i, r in enumerate(train):
        sh = shingles(r.text or r.transcript or "")
        tr_sh.append(sh)
        lsh.candidates_and_add(i, lsh.signature(sh))
    tr_packages = {r.package for r in train if r.package}
    report: dict[str, Any] = {}
    for name, rows in evals.items():
        near, exact, leaked_ids = 0, 0, []
        tr_norm = {normalize(r.text) for r in train if r.text}
        for r in rows:
            q = r.text or r.transcript or ""
            if normalize(q) in tr_norm and q:
                exact += 1
                leaked_ids.append(r.id)
                continue
            sh = shingles(q)
            sig = lsh.signature(sh)
            hit = False
            for b in range(lsh.bands):
                for c in lsh.buckets.get((b, sig[b * lsh.rows:(b + 1) * lsh.rows]), []):
                    if jaccard(sh, tr_sh[c]) >= threshold:
                        hit = True
                        break
                if hit:
                    break
            if hit:
                near += 1
                leaked_ids.append(r.id)
        pk = {r.package for r in rows if r.package}
        report[name] = {"rows": len(rows), "exact_query_overlap": exact, "near_query_overlap": near,
                        "shared_packages": sorted(pk & tr_packages)[:20], "n_shared_packages": len(pk & tr_packages),
                        "leaked_ids": leaked_ids[:50]}
    return report


def balance(rows: list[OmniRow]) -> dict[str, Any]:
    n = len(rows) or 1
    by = lambda f: dict(Counter(f(r) for r in rows).most_common())  # noqa: E731
    tags = Counter(t for r in rows for t in r.tags if ":" not in t)
    menu = [len([o for o in r.options if o.kind != "special"]) for r in rows]
    return {
        "rows": len(rows),
        "abstain_share": sum(r.label == ABSTAIN for r in rows) / n,
        "need_input_share": sum(r.label == NEED_INPUT for r in rows) / n,
        "collision_share": tags.get("collision", 0) / n,
        "near_miss_share": tags.get("near_miss", 0) / n,
        "teacher_labelled_share": sum(r.teacher is not None for r in rows) / n,
        "by_modality": by(lambda r: r.modality), "by_task": by(lambda r: r.task),
        "by_source": by(lambda r: r.source), "by_split": by(lambda r: r.split),
        "packages": len({r.package for r in rows if r.package}),
        "distinct_groups": len({r.group for r in rows}),
        "menu_size_mean": sum(menu) / n, "tags": dict(tags.most_common(20)),
    }


def generic_requests(rows: list[OmniRow], min_packages: int = 3) -> dict[str, Any]:
    """Requests reused verbatim across many unrelated packages.

    In the existing Soro data a generic request ("Book a table for me and my friend")
    is often attached to an unrelated package and labelled with whatever tool that
    menu has. Those labels are suspect (often the right answer is __abstain__).
    Rows are tagged ``generic_request`` so teacher audit checks them first.
    """
    pk: dict[str, set] = defaultdict(set)
    labels: dict[str, set] = defaultdict(set)
    for r in rows:
        n = normalize(r.text)
        if n:
            pk[n].add(r.package)
            labels[n].add(r.label)
    generic = {n for n, p in pk.items() if len(p) >= min_packages}
    tagged = 0
    for r in rows:
        if normalize(r.text) in generic and "generic_request" not in r.tags:
            r.tags.append("generic_request")
            tagged += 1
    top = sorted(generic, key=lambda n: -len(pk[n]))[:10]
    return {"generic_requests": len(generic), "rows_tagged": tagged,
            "examples": [{"request": n[:100], "packages": len(pk[n]), "distinct_gold": len(labels[n])} for n in top]}


def label_audit(rows: list[OmniRow], confident: float = 0.85) -> dict[str, Any]:
    """Rows where the teacher confidently picks a different answer than the gold label."""
    flagged, checked = [], 0
    for r in rows:
        if not r.teacher or not r.teacher.get("probs"):
            continue
        checked += 1
        probs = r.teacher["probs"]
        top = max(probs, key=probs.get)
        if top != r.label and probs[top] >= confident:
            flagged.append({"id": r.id, "gold": r.label, "teacher_top": top, "p": round(probs[top], 3)})
    return {"checked": checked, "flagged": len(flagged), "rate": len(flagged) / checked if checked else None,
            "examples": flagged[:25]}
