"""Text hardening: look-alike distractors, near-miss negatives, paraphrase caps.

Small routers lose to big ones on near-duplicate tools and on "nothing fits".
The current train set has 0.8% collision rows and 9.2% abstain rows, so this
stage manufactures both from existing rows:

* ``mine_collisions``  swap ordinary distractors for the tools most similar to
  the gold tool (from a global pool, never the gold itself). The label is
  unchanged, so it stays correct by construction.
* ``make_near_misses`` drop the gold tool and add its closest look-alike; the
  gold becomes ``__abstain__``. A look-alike can sometimes still do the job, so
  these rows are tagged ``needs_teacher`` and must pass ``filter_near_misses``
  (teacher agrees nothing fits) before training.

Similarity: name-token and description-token Jaccard by default; pass an
``embed`` function (e.g. MiniLM) for dense similarity when available.
"""

from __future__ import annotations

import random
import re
from dataclasses import replace
from typing import Callable, Iterable

from soro.omni_data.schema import ABSTAIN, OmniRow, Option, special_option

_TOK = re.compile(r"[a-z0-9]+")


def _name_tokens(name: str) -> set[str]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    return set(_TOK.findall(spaced.lower()))


_MOD_TOKENS = frozenset({
    "research", "academic", "scholar", "scientific", "web", "online",
    "api", "v1", "v2", "v3", "mcp",
})


def _desc_tokens(text: str) -> set[str]:
    return {t for t in _TOK.findall(text.lower()) if len(t) > 2}


class ToolPool:
    """All distinct tools seen across rows, with a look-alike search."""

    def __init__(self, rows: Iterable[OmniRow], embed: Callable[[list[str]], list[list[float]]] | None = None):
        self.tools: dict[str, Option] = {}
        self.package: dict[str, str | None] = {}
        for r in rows:
            for o in r.options:
                if o.kind == "tool" and o.key not in self.tools:
                    self.tools[o.key] = o
                    self.package[o.key] = r.package
        self.keys = list(self.tools)
        self._name = {k: _name_tokens(k) for k in self.keys}
        self._desc = {k: _desc_tokens(self.tools[k].text) for k in self.keys}
        self._vec = None
        if embed is not None:
            vecs = embed([f"{k}: {self.tools[k].text}" for k in self.keys])
            self._vec = dict(zip(self.keys, vecs))
        # Inverted index for candidate lookup. Tokens shared by many tools ("string",
        # "query", "the") are left out of the index (still used in scoring), so a
        # lookup touches tens of tools instead of thousands.
        df: dict[str, int] = {}
        for k in self.keys:
            for t in self._name[k] | self._desc[k]:
                df[t] = df.get(t, 0) + 1
        cap = max(50, int(0.01 * len(self.keys)))
        self._index: dict[str, set[str]] = {}
        for k in self.keys:
            for t in self._name[k] | self._desc[k]:
                if df[t] <= cap:
                    self._index.setdefault(t, set()).add(k)

    def similarity(self, a: str, b: str) -> float:
        if self._vec is not None and a in self._vec and b in self._vec:
            va, vb = self._vec[a], self._vec[b]
            return float(sum(x * y for x, y in zip(va, vb)))
        na, nb = self._name.get(a, set()), self._name.get(b, set())
        da, db = self._desc.get(a, set()), self._desc.get(b, set())
        jn = len(na & nb) / len(na | nb) if na | nb else 0.0
        jd = len(da & db) / len(da | db) if da | db else 0.0
        return 0.6 * jn + 0.4 * jd

    def same_tool(self, a: str, b: str) -> bool:
        """True when b is almost surely the same/synonym tool as a (send_email vs send-email, search_papers vs search_research_papers, copied descriptions)."""
        na, nb = self._name.get(a, set()), self._name.get(b, set())
        if na and na == nb:
            return True
        ca, cb = na - _MOD_TOKENS, nb - _MOD_TOKENS
        if len(ca) >= 2 and ca == cb:
            return True
        da, db = self._desc.get(a, set()), self._desc.get(b, set())
        return bool(da and db) and len(da & db) / len(da | db) >= 0.8

    def lookalikes(self, key: str, k: int = 5, exclude: set[str] | None = None, min_sim: float = 0.15) -> list[str]:
        """Similar but different tools. Same-tool duplicates are excluded: they would make
        a collision row ambiguous and a near-miss row wrongly labelled abstain."""
        exclude = (exclude or set()) | {key}
        toks = self._name.get(key, set()) | self._desc.get(key, set())
        cands: set[str] = set()
        for t in toks:
            cands |= self._index.get(t, set())
        scored = sorted(((self.similarity(key, c), c) for c in cands
                         if c not in exclude and not self.same_tool(key, c)), reverse=True)
        return [c for s, c in scored[:k] if s >= min_sim]


def mine_collisions(rows: list[OmniRow], pool: ToolPool, per_row: int = 2, rate: float = 0.15,
                    seed: int = 0) -> list[OmniRow]:
    """New rows where ``per_row`` distractors are replaced by look-alikes of the gold tool."""
    rng = random.Random(seed)
    out = []
    for r in rows:
        if r.label == ABSTAIN or r.task != "tool_route" or rng.random() > rate:
            continue
        menu = [o.key for o in r.options]
        looks = pool.lookalikes(r.label, k=per_row, exclude=set(menu))
        if not looks:
            continue
        replaceable = [i for i, o in enumerate(r.options) if o.key != r.label and o.kind == "tool"]
        if not replaceable:
            continue
        rng.shuffle(replaceable)
        opts = list(r.options)
        for i, lk in zip(replaceable, looks):
            opts[i] = pool.tools[lk]
        rng.shuffle(opts)
        out.append(replace(r, id=f"{r.id}~col", options=opts, tags=r.tags + ["collision", "derived"],
                           source="soro_hardened", license="own", teacher=None,
                           meta={**r.meta, "parent": r.id, "lookalikes": looks}))
    return out


def make_near_misses(rows: list[OmniRow], pool: ToolPool, rate: float = 0.08, seed: int = 1) -> list[OmniRow]:
    """New rows where the gold tool is replaced by its closest look-alike; gold becomes abstain."""
    rng = random.Random(seed)
    out = []
    for r in rows:
        if r.label == ABSTAIN or r.task != "tool_route" or rng.random() > rate:
            continue
        menu = {o.key for o in r.options}
        looks = pool.lookalikes(r.label, k=1, exclude=menu, min_sim=0.25)
        if not looks:
            continue
        opts = [pool.tools[looks[0]] if o.key == r.label else o for o in r.options]
        if not any(o.key == ABSTAIN for o in opts):
            opts.append(special_option(ABSTAIN))
        rng.shuffle(opts)
        out.append(replace(r, id=f"{r.id}~nm", options=opts, label=ABSTAIN,
                           tags=r.tags + ["near_miss", "needs_teacher", "derived"], source="soro_hardened",
                           license="own", teacher=None,
                           meta={**r.meta, "parent": r.id, "removed_gold": r.label, "lookalike": looks[0]}))
    return out


def filter_near_misses(rows: list[OmniRow], min_abstain: float = 0.6) -> tuple[list[OmniRow], int]:
    """Keep near-miss rows only when the teacher agrees nothing fits. Others pass through."""
    kept, dropped = [], 0
    for r in rows:
        if "needs_teacher" not in r.tags:
            kept.append(r)
            continue
        probs = (r.teacher or {}).get("probs")
        if probs and probs.get(ABSTAIN, 0.0) >= min_abstain:
            kept.append(replace(r, tags=[t for t in r.tags if t != "needs_teacher"] + ["teacher_verified"]))
        else:
            dropped += 1
    return kept, dropped
