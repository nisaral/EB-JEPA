"""S2 Action Cards (ANSWER / ASK / ESCALATE / ABSTAIN) & Asymmetric Expected-Cost Helpers.

Implements PLAN v4 S2 without GPU generation:
  * Every S2 row keeps all four action cards in its menu alongside real tool options.
  * Cards carry explicit cost descriptions, and ``row.meta["s2"]`` stores the exact
    per-option asymmetric cost vector ``costs: dict[str, float]`` for ``expected_cost_loss``.
  * Converters ingest ``nvidia/When2Call`` (CC BY 4.0) and ``antiquality/agentabstain``
    (CC BY 4.0, eval-only), plus deterministic ASK / ESCALATE / ANSWER construction over
    ``soro_train`` tool menus.
  * Includes per-action-group (Mondrian) split conformal calibration across action types.
"""

from __future__ import annotations

import ast
import hashlib
import json
import random
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from soro.omni_data.schema import (
    ABSTAIN,
    ACTION_CARDS,
    ANSWER,
    ASK,
    ESCALATE,
    NEED_INPUT,
    SPECIAL,
    OmniRow,
    Option,
)

S2_CARD_TEXT: dict[str, str] = {
    ANSWER: (
        "Answer directly without calling any tool (cautious deferral cost 0.45 if unnecessary; "
        "false direct-answer cost 1.1-1.5 when execution or escalation is needed)."
    ),
    ASK: (
        "Ask the user a brief clarifying question before acting (cautious ask cost 0.40-0.45; "
        "guessing a tool blindly costs 1.5)."
    ),
    ESCALATE: (
        "Escalate to a stronger supervisor or human for confirmation (cautious escalate cost 0.55-0.60; "
        "unsafe high-risk execution without approval costs 2.0)."
    ),
    ABSTAIN: (
        "Abstain because no listed tool or capability fits the request (safe abstain cost 0.50-0.80; "
        "hallucinated tool execution costs 1.5)."
    ),
}

# Asymmetric cost matrix C(gold_kind, pred_kind). Exact hit is 0.0.
COST_MATRIX: dict[tuple[str, str], float] = {
    ("tool", "tool"): 1.0,
    ("tool", "answer"): 1.2,
    ("tool", "ask"): 0.45,
    ("tool", "escalate"): 0.60,
    ("tool", "abstain"): 0.80,
    ("answer", "tool"): 1.2,
    ("answer", "ask"): 0.45,
    ("answer", "escalate"): 0.60,
    ("answer", "abstain"): 0.75,
    ("ask", "tool"): 1.5,
    ("ask", "answer"): 1.1,
    ("ask", "escalate"): 0.55,
    ("ask", "abstain"): 0.70,
    ("escalate", "tool"): 2.0,
    ("escalate", "answer"): 1.5,
    ("escalate", "ask"): 0.40,
    ("escalate", "abstain"): 0.50,
    ("abstain", "tool"): 1.5,
    ("abstain", "answer"): 1.1,
    ("abstain", "ask"): 0.45,
    ("abstain", "escalate"): 0.55,
}

_HIGH_RISK_WORDS = re.compile(
    r"\b(delete|remove|drop|wipe|purge|truncate|revoke|transfer|wire|pay|charge|"
    r"refund|publish|broadcast|terminate|destroy|erase|grant_admin|reset_password|"
    r"overwrite|cancel_all|shutdown)\b",
    re.I,
)

_ABSTAIN_KW = (
    "apolog",
    "sorry",
    "unable",
    "cannot",
    "can't",
    "do not have access",
    "don't have",
    "not possible",
    "not equipped",
    "not capable",
    "no tool",
    "i'm afraid",
    "unfortunately",
    "impossible",
    "beyond my",
    "don't possess",
    "do not possess",
    "lack the",
)
_ASK_PHRASE_KW = (
    "to proceed",
    "please provide",
    "could you",
    "can you",
    "would you",
    "specify",
    "let me know",
    "need to know",
    "need more",
    "need the",
    "clarify",
    "please tell",
    "which one",
    "i'll need",
    "i need the",
)

# Pre-execution categories in antiquality/agentabstain -> canonical S2 label.
AGENTABSTAIN_PREEXEC_MAP: dict[str, str] = {
    "missing_critical_parameter": ASK,
    "ambiguous_action_specification": ASK,
    "conflicting_constraints": ASK,
    "high_stakes_action": ESCALATE,
    "insufficient_tool_capability": ABSTAIN,
}

# Runtime categories in antiquality/agentabstain (require tool output observation, tracked for S4/eval).
AGENTABSTAIN_RUNTIME_MAP: dict[str, str] = {
    "conflicting_evidence": ASK,
    "emergent_risk_discovery": ESCALATE,
    "critical_tool_failure": ABSTAIN,
}


def action_kind(key: str) -> str:
    if key == ANSWER:
        return "answer"
    if key == ASK:
        return "ask"
    if key == ESCALATE:
        return "escalate"
    if key in (ABSTAIN, NEED_INPUT):
        return "abstain"
    return "tool"


def s2_card_option(key: str) -> Option:
    if key not in S2_CARD_TEXT:
        raise ValueError(f"Unknown S2 action card {key!r}; expected one of {ACTION_CARDS}")
    return Option(key=key, text=S2_CARD_TEXT[key], kind="special")


def infer_risk_tier(options: Iterable[Option], label: str = "") -> str:
    if label == ESCALATE:
        return "high"
    for o in options:
        if o.kind == "tool" and (_HIGH_RISK_WORDS.search(o.key) or _HIGH_RISK_WORDS.search(o.text[:200])):
            return "high"
    return "low"


def action_cost(gold: str, pred: str, risk_tier: str = "low") -> float:
    if pred == gold:
        return 0.0
    gk, pk = action_kind(gold), action_kind(pred)
    base = COST_MATRIX.get((gk, pk), 1.0)
    if risk_tier == "high" and pk == "tool" and gk != "escalate":
        base = min(2.5, round(base * 1.5, 4))
    return float(base)


def compute_cost_vector(option_keys: Iterable[str], gold: str, risk_tier: str = "low") -> dict[str, float]:
    return {k: action_cost(gold, k, risk_tier=risk_tier) for k in option_keys}


def wrap_s2_row(
    row: OmniRow,
    *,
    label: str | None = None,
    risk_tier: str | None = None,
    extra_tags: Iterable[str] = (),
    shuffle_seed: int | None = None,
) -> OmniRow:
    """Ensure all 4 S2 action cards are in ``row.options`` and attach ``meta['s2']`` costs."""
    target = label if label is not None else row.label
    tool_opts = [o for o in row.options if o.kind != "special"]
    extra_specials = [o for o in row.options if o.kind == "special" and o.key not in ACTION_CARDS and o.key == target]
    cards = [s2_card_option(k) for k in ACTION_CARDS]
    opts = tool_opts + extra_specials + cards
    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        rng.shuffle(opts)
    tier = risk_tier or infer_risk_tier(tool_opts, target)
    keys = [o.key for o in opts]
    if target not in keys:
        raise ValueError(f"S2 target label {target!r} missing from options {keys}")
    costs = compute_cost_vector(keys, target, risk_tier=tier)
    tags = list(dict.fromkeys([*row.tags, "s2", f"s2:{action_kind(target)}", *extra_tags]))
    meta = dict(row.meta)
    meta["s2"] = {
        "risk_tier": tier,
        "action_kind": action_kind(target),
        "costs": costs,
    }
    return replace(row, options=opts, label=target, tags=tags, meta=meta)


# ------------------------------------------------------------------ When2Call converters
def _parse_json_obj(item: Any) -> dict[str, Any] | None:
    if isinstance(item, dict):
        return item
    if isinstance(item, str):
        s = item.strip()
        if not s:
            return None
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _format_tool_schema(tool: dict[str, Any]) -> str:
    desc = str(tool.get("description") or "").strip()
    params = tool.get("parameters") or {}
    props = params.get("properties") if isinstance(params, dict) else None
    req = set(params.get("required") or []) if isinstance(params, dict) else set()
    arg_parts: list[str] = []
    if isinstance(props, dict):
        for pname, pspec in list(props.items())[:8]:
            ptype = pspec.get("type", "any") if isinstance(pspec, dict) else "any"
            star = "*" if pname in req else ""
            arg_parts.append(f"{pname}{star}:{ptype}")
    if arg_parts:
        sig = f" [args: {', '.join(arg_parts)}]"
        return f"{desc}{sig}".strip()
    return desc or str(tool.get("name") or "tool")


def _extract_when2call_options(tools_raw: Any) -> list[Option]:
    seen: set[str] = set()
    opts: list[Option] = []
    for item in tools_raw or []:
        t = _parse_json_obj(item)
        if not t or not t.get("name"):
            continue
        name = str(t["name"]).strip()
        if not name or name in seen or name in SPECIAL:
            continue
        seen.add(name)
        opts.append(Option(key=name, text=_format_tool_schema(t), kind="tool"))
    return opts


def _split_messages(messages: list[dict[str, Any]]) -> tuple[str, str]:
    user_turns = [m for m in messages if m.get("role") in ("user", "system", "assistant")]
    last_u = -1
    for i in range(len(user_turns) - 1, -1, -1):
        if user_turns[i].get("role") == "user":
            last_u = i
            break
    if last_u < 0:
        return "", ""
    text = str(user_turns[last_u].get("content") or "").strip()
    prior = []
    for m in user_turns[:last_u]:
        c = str(m.get("content") or "").strip()
        if c:
            prior.append(f"{m.get('role', 'user').capitalize()}: {c}")
    return "\n".join(prior), text


def _classify_when2call_response(content: str, tool_keys: set[str]) -> str | None:
    s = (content or "").strip()
    if "<TOOLCALL>" in s:
        m = re.search(r"<TOOLCALL>\s*(\[.*\])\s*(?:</TOOLCALL>)?", s, re.S)
        if not m:
            return None
        try:
            calls = json.loads(m.group(1))
        except json.JSONDecodeError:
            return None
        if not isinstance(calls, list) or not calls:
            return None
        name = str(calls[0].get("name") or "").strip()
        return name if name in tool_keys else None
    low = s.lower()
    if any(k in low for k in _ABSTAIN_KW):
        return ABSTAIN
    if "?" in s or any(k in low for k in _ASK_PHRASE_KW):
        return ASK
    return ANSWER


def _short_hash(text: str) -> str:
    return hashlib.sha1(" ".join((text or "").lower().split()).encode("utf-8")).hexdigest()[:16]


def from_when2call_pref(records: Iterable[dict[str, Any]], *, split: str = "train") -> list[OmniRow]:
    out: list[OmniRow] = []
    for idx, rec in enumerate(records):
        opts = _extract_when2call_options(rec.get("tools"))
        if not opts:
            continue
        msgs = rec.get("messages") or []
        context, text = _split_messages(msgs)
        if not text:
            continue
        chosen = rec.get("chosen_response")
        content = chosen.get("content", "") if isinstance(chosen, dict) else str(chosen or "")
        label = _classify_when2call_response(content, {o.key for o in opts})
        if label is None:
            continue
        gid = _short_hash(f"{context}\n{text}")
        row = OmniRow(
            id=f"w2c-pref-{idx}",
            task="tool_route",
            modality="text",
            options=opts,
            label=label,
            source="when2call",
            license="cc-by-4.0",
            group=f"w2c-{gid}",
            text=text,
            context=context,
            package=f"w2c:{opts[0].key}",
            split=split,
            tags=["when2call_pref"],
        )
        out.append(wrap_s2_row(row, shuffle_seed=idx))
    return out


def from_when2call_sft(
    records: Iterable[dict[str, Any]],
    *,
    split: str = "train",
    allowed_actions: tuple[str, ...] = (ASK, ABSTAIN),
    seed: int = 0,
) -> list[OmniRow]:
    out: list[OmniRow] = []
    allowed = set(allowed_actions)
    for idx, rec in enumerate(records):
        opts = _extract_when2call_options(rec.get("tools"))
        if not opts:
            continue
        msgs = list(rec.get("messages") or [])
        if not msgs or msgs[-1].get("role") != "assistant":
            continue
        label = _classify_when2call_response(str(msgs[-1].get("content") or ""), {o.key for o in opts})
        if label not in allowed:
            continue
        context, text = _split_messages(msgs[:-1])
        if not text:
            continue
        gid = _short_hash(f"{context}\n{text}")
        row = OmniRow(
            id=f"w2c-sft-{idx}",
            task="tool_route",
            modality="text",
            options=opts,
            label=label,
            source="when2call",
            license="cc-by-4.0",
            group=f"w2c-{gid}",
            text=text,
            context=context,
            package=f"w2c:{opts[0].key}",
            split=split,
            tags=["when2call_sft"],
        )
        out.append(wrap_s2_row(row, shuffle_seed=seed + idx))
    return out


def from_when2call_mcq(records: Iterable[dict[str, Any]], *, split: str = "test") -> list[OmniRow]:
    out: list[OmniRow] = []
    for idx, rec in enumerate(records):
        opts = _extract_when2call_options(rec.get("tools"))
        text = str(rec.get("question") or "").strip()
        if not opts or not text:
            continue
        ca = rec.get("correct_answer")
        keys = {o.key for o in opts}
        if ca == "tool_call":
            t = _parse_json_obj(rec.get("target_tool"))
            if not t or not t.get("name"):
                continue
            tname = str(t["name"]).strip()
            if tname not in keys:
                opts.append(Option(key=tname, text=_format_tool_schema(t), kind="tool"))
            label = tname
        elif ca == "request_for_info":
            label = ASK
        elif ca == "cannot_answer":
            label = ABSTAIN
        else:
            continue
        src_id = str(rec.get("uuid") or rec.get("source_id") or idx)
        row = OmniRow(
            id=f"w2c-mcq-{src_id}",
            task="tool_route",
            modality="text",
            options=opts,
            label=label,
            source="when2call",
            license="cc-by-4.0",
            group=f"w2c-mcq-{_short_hash(text)}",
            text=text,
            package=str(rec.get("source") or "when2call_mcq"),
            split=split,
            tags=["when2call_mcq"],
            meta={"source_id": rec.get("source_id")},
        )
        out.append(wrap_s2_row(row, shuffle_seed=idx))
    return out


# ------------------------------------------------------------------ AgentAbstain converter
def parse_agentabstain_env_source(source_code: str) -> dict[str, Option]:
    """Parse ``@self.mcp.tool()`` definitions in an AgentAbstain environment.py via AST."""
    tree = ast.parse(source_code)
    tools: dict[str, Option] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
            continue
        has_tool_dec = any(
            isinstance(d, ast.Call)
            and isinstance(d.func, ast.Attribute)
            and d.func.attr == "tool"
            for d in node.decorator_list
        )
        if not has_tool_dec:
            continue
        args = [a.arg for a in node.args.args if a.arg != "self"]
        doc = ast.get_docstring(node) or ""
        first_line = doc.splitlines()[0].strip() if doc else node.name.replace("_", " ")
        sig = f" [args: {', '.join(args[:8])}]" if args else ""
        tools[node.name] = Option(key=node.name, text=f"{first_line}{sig}".strip(), kind="tool")
    return tools


def _norm_tool_key(name: str) -> str:
    return name.split(".")[-1].strip()


def from_agentabstain_rows(
    tasks: Iterable[dict[str, Any]],
    env_tools: dict[str, dict[str, Option]],
    *,
    split: str = "test",
    include_runtime: bool = False,
) -> list[OmniRow]:
    task_list = list(tasks)
    # Map pair_id -> sibling 'act' DAG tools so 'insufficient_tool_capability' knows what to drop.
    act_first_tool: dict[str, str] = {}
    for r in task_list:
        if r.get("task_type") == "act":
            nodes = (r.get("execution_dag") or {}).get("nodes") or []
            if nodes and nodes[0].get("tool"):
                act_first_tool[str(r.get("pair_id"))] = _norm_tool_key(nodes[0]["tool"])

    out: list[OmniRow] = []
    for idx, r in enumerate(task_list):
        phase = r.get("phase") or "pre_execution"
        if phase != "pre_execution" and not include_runtime:
            continue
        envs = list(r.get("environments") or [])
        pool: dict[str, Option] = {}
        for env in envs:
            pool.update(env_tools.get(env, {}))
        if not pool:
            continue

        cat = str(r.get("category") or "")
        ttype = str(r.get("task_type") or "")
        pair_id = str(r.get("pair_id") or idx)
        instr = str(r.get("instruction") or "").strip()
        if not instr:
            continue

        if ttype == "act":
            nodes = (r.get("execution_dag") or {}).get("nodes") or []
            if not nodes or not nodes[0].get("tool"):
                continue
            gold = _norm_tool_key(nodes[0]["tool"])
            if gold not in pool:
                continue
        elif ttype == "abstain":
            gold = AGENTABSTAIN_PREEXEC_MAP.get(cat) or (
                AGENTABSTAIN_RUNTIME_MAP.get(cat) if include_runtime else None
            )
            if gold is None:
                continue
            if cat == "insufficient_tool_capability":
                drop_tool = act_first_tool.get(pair_id)
                trig = r.get("abstention_trigger") or {}
                drop_names = {drop_tool} if drop_tool else set()
                for k in ("missing_tool", "removed_tool", "unavailable_tool"):
                    if trig.get(k):
                        drop_names.add(_norm_tool_key(str(trig[k])))
                pool = {k: v for k, v in pool.items() if k not in drop_names}
                if not pool:
                    continue
        else:
            continue

        risk = "high" if cat == "high_stakes_action" or r.get("action_type") == "operational" else "low"
        sys_prompt = str(r.get("system_prompt") or "").strip()
        row = OmniRow(
            id=f"aa-{pair_id.replace('/', '_')}-{ttype}",
            task="tool_route",
            modality="text",
            options=list(pool.values()),
            label=gold,
            source="agentabstain",
            license="cc-by-4.0",
            group=f"aa-{pair_id}",
            text=instr,
            context=sys_prompt,
            package=",".join(envs) if envs else "agentabstain",
            split=split,
            tags=["agentabstain", f"phase:{phase}", f"cat:{cat}"],
            meta={"pair_id": pair_id, "category": cat, "task_type": ttype, "phase": phase},
        )
        out.append(wrap_s2_row(row, risk_tier=risk, shuffle_seed=idx))
    return out


def load_agentabstain_env_tools(
    envs: Iterable[str],
    *,
    repo_id: str = "antiquality/agentabstain",
) -> dict[str, dict[str, Option]]:
    """Download/read cached ``environments/<env>/environment.py`` files and extract tools via AST."""
    from huggingface_hub import hf_hub_download

    loaded: dict[str, dict[str, Option]] = {}
    for env in sorted({e for e in envs if e}):
        p = hf_hub_download(repo_id, f"environments/{env}/environment.py", repo_type="dataset")
        loaded[env] = parse_agentabstain_env_source(Path(p).read_text(encoding="utf-8"))
    return loaded


# ------------------------------------------------------------------ Deterministic ASK / ESCALATE / ANSWER over Soro
_EMAIL_RE = re.compile(r"\b(?:to|for|from|at|email|cc)?\s*['\"]?[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}['\"]?", re.I)
_URL_RE = re.compile(r"\b(?:at|from|on|url|link)?\s*https?://\S+", re.I)
_PATH_RE = re.compile(r"\b(?:at|from|in|to|path|file)?\s*(?:~?/(?:[\w.\-]+/)*[\w.\-]+)", re.I)
_QUOTED_ARG_RE = re.compile(r"(?:named|called|for|id|titled|query|matching)\s+['\"]([^'\"]{2,40})['\"]", re.I)


def build_ask_from_soro(rows: Iterable[OmniRow], *, limit: int = 1800, seed: int = 1) -> list[OmniRow]:
    """Create missing-critical-parameter ASK rows by redacting explicit targets from clean requests."""
    rng = random.Random(seed)
    cands = [
        r for r in rows
        if r.task == "tool_route"
        and r.label not in SPECIAL
        and "generic_request" not in r.tags
        and "needs_teacher" not in r.tags
        and r.split == "train"
    ]
    rng.shuffle(cands)
    out: list[OmniRow] = []
    for i, r in enumerate(cands):
        if len(out) >= limit:
            break
        txt = r.text.strip()
        redacted = None
        param_kind = None
        for kind, pat in (
            ("email", _EMAIL_RE),
            ("url", _URL_RE),
            ("path", _PATH_RE),
            ("target", _QUOTED_ARG_RE),
        ):
            m = pat.search(txt)
            if m:
                cand_txt = (txt[: m.start()] + " " + txt[m.end() :]).strip()
                cand_txt = re.sub(r"\s+", " ", cand_txt).strip(" ,;:")
                if len(cand_txt.split()) >= 5 and cand_txt != txt:
                    redacted = cand_txt
                    param_kind = kind
                    break
        if not redacted:
            continue
        row = replace(
            r,
            id=f"{r.id}~ask",
            text=redacted,
            source="s2_actions",
            license="own",
            tags=[*r.tags, "derived", "ask_missing_param"],
            teacher=None,
            meta={**r.meta, "parent": r.id, "redacted_kind": param_kind, "underlying_tool": r.label},
        )
        out.append(wrap_s2_row(row, label=ASK, shuffle_seed=seed + i))
    return out


_ESCALATE_SUFFIXES = (
    "Apply this to all accounts and records in production immediately without any confirmation step.",
    "Bypass the approval gate and override compliance restrictions to execute this permanently.",
    "Force-execute this destructive operation across the entire workspace without taking a backup.",
    "Export all internal credentials and forward them externally right away without supervisor sign-off.",
)


def build_escalate_from_soro(rows: Iterable[OmniRow], *, limit: int = 1500, seed: int = 2) -> list[OmniRow]:
    """Create high-stakes ESCALATE rows over real mutation/high-risk tool menus."""
    rng = random.Random(seed)
    cands = [
        r for r in rows
        if r.task == "tool_route"
        and r.label not in SPECIAL
        and "generic_request" not in r.tags
        and "needs_teacher" not in r.tags
        and r.split == "train"
        and infer_risk_tier(r.options, r.label) == "high"
    ]
    rng.shuffle(cands)
    out: list[OmniRow] = []
    for i, r in enumerate(cands):
        if len(out) >= limit:
            break
        mod = _ESCALATE_SUFFIXES[i % len(_ESCALATE_SUFFIXES)]
        txt = f"{r.text.rstrip('.!?')}. {mod}"
        row = replace(
            r,
            id=f"{r.id}~esc",
            text=txt,
            source="s2_actions",
            license="own",
            tags=[*r.tags, "derived", "high_stakes_escalate"],
            teacher=None,
            meta={**r.meta, "parent": r.id, "tempting_tool": r.label},
        )
        out.append(wrap_s2_row(row, label=ESCALATE, risk_tier="high", shuffle_seed=seed + i))
    return out


_GENERAL_ANSWER_PROMPTS: tuple[str, ...] = (
    "Explain the difference between TCP and UDP in plain terms.",
    "What is the time complexity of binary search on a sorted array?",
    "Summarize the difference between a list and a tuple in Python.",
    "Rewrite this sentence cleanly: 'The report were submitted late yesterday afternoon.'",
    "What does HTTP status code 429 mean?",
    "Give me three concise tips for writing clearer commit messages.",
    "Why is a balanced binary search tree faster than a linked list for lookups?",
    "Explain what idempotency means in REST APIs with a quick example.",
    "What is the difference between precision and recall?",
    "How do I reverse a string in Python without using a loop?",
    "Explain why Brier score is a proper scoring rule.",
    "What causes a race condition in concurrent code?",
    "Compare SQL INNER JOIN vs LEFT JOIN briefly.",
    "What does chmod 600 do on Unix-like systems?",
    "Explain the purpose of LayerNorm in a transformer block.",
    "How does cosine annealing affect the learning rate schedule?",
    "Convert the active sentence to passive voice: 'The router selected the tool.'",
    "What is the difference between UTF-8 and ASCII?",
    "Briefly explain why overfitting happens on small datasets.",
    "Hello, thanks for your help today!",
    "What is the difference between synchronous and asynchronous I/O?",
    "Explain what gradient accumulation accomplishes during neural network training.",
    "Why does LoRA freeze the pretrained backbone weights?",
    "What is the capital of France, and what river runs through it?",
    "Explain the difference between L1 and L2 regularization in two sentences.",
    "In Git, how does git rebase differ from git merge?",
    "What does a HTTP 503 Service Unavailable response usually indicate?",
    "Briefly explain what conformal prediction guarantees without distributional assumptions.",
    "How do you handle a KeyError safely when reading a Python dictionary?",
    "What is the difference between breadth-first search and depth-first search?",
)


def _diverse_direct_qa_prompts(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    prompts = list(_GENERAL_ANSWER_PROMPTS)
    # Deterministically synthesize varied conceptual/math prompts up to n so text rows do not repeat verbatim.
    concepts = [
        ("REST", "GraphQL"), ("stack", "queue"), ("mutex", "semaphore"),
        ("process", "thread"), ("compiler", "interpreter"), ("TCP", "QUIC"),
        ("B-tree", "hash table"), ("FP16", "BF16"), ("bias", "variance"),
        ("supervised learning", "reinforcement learning"), ("microbatch", "minibatch"),
        ("causal attention", "bidirectional attention"), ("L2 norm", "cosine similarity"),
    ]
    for a, b in concepts:
        prompts.append(f"Explain the practical trade-offs between {a} and {b} in two concise bullet points.")
    for k in range(12, 72):
        a = (k * 7) % 89 + 11
        b = (k * 13) % 47 + 5
        prompts.append(f"Compute {a} * {b} + {k} mentally and explain the steps briefly.")
    rng.shuffle(prompts)
    return prompts[:n]


def build_answer_from_soro(
    rows: Iterable[OmniRow],
    *,
    limit: int = 1800,
    seed: int = 3,
    include_direct_qa: bool = True,
) -> list[OmniRow]:
    """Create direct ANSWER rows: (1) post-observation synthesis where tool already ran, (2) non-tool QA."""
    rng = random.Random(seed)
    usable = [
        r for r in rows
        if r.task == "tool_route"
        and r.label not in SPECIAL
        and "generic_request" not in r.tags
        and "needs_teacher" not in r.tags
        and r.split == "train"
    ]
    rng.shuffle(usable)
    out: list[OmniRow] = []
    n_obs = min(limit if not include_direct_qa else int(limit * 0.8), len(usable))
    for i, r in enumerate(usable[:n_obs]):
        ctx_head = f"{r.context.strip()}\n" if r.context else ""
        ctx = (
            f"{ctx_head}Tool output ({r.label}): "
            '{"status": "ok", "summary": "Completed requested lookup and verified result."}'
        ).strip()
        txt = f"{r.text.rstrip('.!?')}. The tool result is already provided above; present the final answer directly."
        row = replace(
            r,
            id=f"{r.id}~ans",
            text=txt,
            context=ctx,
            source="s2_actions",
            license="own",
            tags=[*r.tags, "derived", "post_tool_answer"],
            teacher=None,
            meta={**r.meta, "parent": r.id, "completed_tool": r.label},
        )
        out.append(wrap_s2_row(row, label=ANSWER, shuffle_seed=seed + i))

    if not include_direct_qa:
        return out

    # Add unique non-tool conceptual/arithmetic QA paired with unrelated real menus.
    pool_menus = usable[n_obs:] or usable
    qa_prompts = _diverse_direct_qa_prompts(max(0, limit - len(out)), seed + 99)
    menu_cursor = 0
    for idx, prompt in enumerate(qa_prompts):
        chosen_r = None
        while menu_cursor < len(pool_menus) * 2 and pool_menus:
            cand = pool_menus[menu_cursor % len(pool_menus)]
            menu_cursor += 1
            menu_str = " ".join(f"{o.key} {o.text}" for o in cand.options).lower()
            if not any(w in menu_str for w in ("explain", "translate", "python", "code_exec", "chat", "calc", "math", "search")):
                chosen_r = cand
                break
        if chosen_r is None:
            break
        row = replace(
            chosen_r,
            id=f"s2-qa-{idx}",
            text=prompt,
            context="",
            group=f"s2-qa-{_short_hash(prompt)}",
            source="s2_actions",
            license="own",
            tags=["derived", "direct_qa_answer"],
            teacher=None,
            meta={"menu_source": chosen_r.id},
        )
        out.append(wrap_s2_row(row, label=ANSWER, shuffle_seed=seed + 10000 + idx))
    return out


# ------------------------------------------------------------------ Loss & Per-Action Conformal Calibration
def expected_cost_loss(logits, cost_vec):
    """Differentiable expected asymmetric decision cost: sum_a softmax(logits)_a * cost(gold, a)."""
    import torch
    import torch.nn.functional as F

    probs = F.softmax(logits, dim=-1)
    c = torch.as_tensor(cost_vec, dtype=probs.dtype, device=probs.device)
    return (probs * c).sum(dim=-1)


def action_group_conformal(
    items: list[tuple[dict[str, float], str]],
    *,
    alpha: float = 0.10,
    cal_frac: float = 0.5,
    seed: int = 0,
) -> dict[str, Any]:
    """Mondrian (per-action-group) split conformal calibration across tool/answer/ask/escalate/abstain."""
    if len(items) < 4:
        return {"n": len(items), "overall": None, "by_action": {}}
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    cut = max(2, min(len(shuffled) - 2, int(len(shuffled) * cal_frac)))
    cal, ev = shuffled[:cut], shuffled[cut:]

    # Calibrate nonconformity s = 1 - p(gold) per true action group (with global fallback for rare groups).
    global_scores = sorted(1.0 - probs.get(g, 0.0) for probs, g in cal)
    group_scores: dict[str, list[float]] = {}
    for probs, g in cal:
        group_scores.setdefault(action_kind(g), []).append(1.0 - probs.get(g, 0.0))

    def _quantile(scores: list[float]) -> float:
        s = sorted(scores)
        n = len(s)
        idx = min(n - 1, max(0, int(((n + 1) * (1.0 - alpha)) // 1) - 1))
        return float(s[idx])

    global_q = _quantile(global_scores)
    q_hat: dict[str, float] = {
        gk: (_quantile(sc) if len(sc) >= 5 else global_q) for gk, sc in group_scores.items()
    }

    by_action: dict[str, dict[str, Any]] = {}
    all_cov, all_sz, all_cost = [], [], []
    for probs, g in ev:
        gk = action_kind(g)
        thr = q_hat.get(gk, global_q)
        # Wait: at test time true group isn't known — use predicted-group / conservative max q_hat or
        # for per-group coverage report, evaluate both oracle-group threshold & unified max(p_k >= 1-q_k).
        pred_set = [k for k, p in probs.items() if (1.0 - p) <= q_hat.get(action_kind(k), global_q)]
        if not pred_set and probs:
            pred_set = [max(probs, key=probs.get)]
        cov = int(g in pred_set)
        top = max(probs, key=probs.get) if probs else g
        cost = action_cost(g, top)
        all_cov.append(cov)
        all_sz.append(len(pred_set))
        all_cost.append(cost)
        rec = by_action.setdefault(gk, {"n": 0, "cov": 0, "size": 0, "cost": 0.0, "q_hat": thr})
        rec["n"] += 1
        rec["cov"] += cov
        rec["size"] += len(pred_set)
        rec["cost"] += cost

    summary_groups = {
        gk: {
            "n": d["n"],
            "coverage": d["cov"] / d["n"],
            "set_size": d["size"] / d["n"],
            "mean_cost": d["cost"] / d["n"],
            "q_hat": d["q_hat"],
        }
        for gk, d in sorted(by_action.items())
    }
    ne = len(ev)
    return {
        "n_cal": len(cal),
        "n_eval": ne,
        "alpha": alpha,
        "global_q_hat": global_q,
        "overall": {
            "coverage": sum(all_cov) / ne,
            "set_size": sum(all_sz) / ne,
            "mean_cost": sum(all_cost) / ne,
        },
        "by_action": summary_groups,
    }
