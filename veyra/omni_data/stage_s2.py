"""Stage 2 data builder (CPU only): 4-action S2 rows (tool/ANSWER/ASK/ABSTAIN) + cost labels + escalate probe.

    python veyra/omni_data/stage_s2.py --smoke
    python veyra/omni_data/stage_s2.py --out veyra/runs/omni_data/s2

Reads local Soro release/eval files + HF Hub cached datasets (``nvidia/When2Call``,
``antiquality/agentabstain``) with NO GPU and writes validated, license-gated,
leakage-checked OmniRow jsonl files:
  s2_train.jsonl             Balanced 4-action S2 training rows (tool / ask / abstain / answer)
  s2_selection.jsonl         Group- and package-disjoint validation rows (4 actions)
  s2_gold_100.jsonl          Stratified 100-row owner audit set (25 each of tool/answer/ask/abstain)
  s2_gold_100_review.md      Human-readable review sheet for owner hand-checking
  eval_escalate_probe.jsonl  Eval-only high-stakes probe (Soro high-risk + AgentAbstain high_stakes_action)
  eval_when2call_mcq.jsonl   When2Call test MCQ benchmark (test split)
  eval_agentabstain.jsonl    AgentAbstain 4-action pre-execution benchmark (test split)
  eval_zeroshot_s2.jsonl     Held-out zeroshot wrapped with S2 action cards (test split)
  eval_external_s2.jsonl     Held-out external wrapped with S2 action cards (test split)
  stage_s2_report.json       Counts, action balance, cost summary, leakage audit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "veyra" / "python" / "src"))
sys.path.insert(0, str(REPO / "veyra" / "training"))

from soro.omni_data import licenses, quality  # noqa: E402
from soro.omni_data.s2_actions import (  # noqa: E402
    _ANAPHORIC_OR_VAGUE_RE,
    _ctoks,
    _distinctive_toks,
    action_kind,
    build_abstain_cross_domain,
    build_ambiguous_ask,
    build_answer_from_soro,
    build_ask_from_soro,
    build_escalate_from_soro,
    from_agentabstain_rows,
    from_when2call_mcq,
    from_when2call_pref,
    from_when2call_sft,
    is_clean_soro_row,
    is_pure_llm_prompt,
    load_agentabstain_env_tools,
    wrap_s2_row,
)
from soro.omni_data.schema import (  # noqa: E402
    ABSTAIN,
    ANSWER,
    ASK,
    ESCALATE,
    SPECIAL,
    OmniRow,
    Option,
    from_soro_record,
    special_option,
    write_rows,
)
from soro.omni_data.text import ToolPool, mine_collisions  # noqa: E402

DATA = REPO / "releases" / "soro-universal-tool-harness"
EVAL_DIR = REPO / "veyra" / "omni_data" / "eval_splits"

_NM_DELEX_TWINS = {"op_9504", "op_9506", "op_9935", "op_9992", "op_9996", "op_9997"}


def _is_valid_nearmiss_eval_row(row: OmniRow) -> bool:
    """Exclude the 29 verified wrong-gold delexicalized twin / synonym rows from near_miss_s2 eval."""
    if row.id.startswith("seq_"):
        return True
    rem = str((row.meta or {}).get("removed_gold") or "")
    look = str((row.meta or {}).get("lookalike") or "")
    if look in _NM_DELEX_TWINS:
        if (
            (look == "op_9504" and rem == "generate_uuid")
            or (look == "op_9506" and rem == "calculate_date_difference")
            or (look == "op_9935" and rem == "encode_base64")
            or (look in ("op_9992", "op_9996") and rem in ("decode_base64", "op_10906", "op_6644"))
            or (look == "op_9997" and "date" in (row.text or "").lower())
        ):
            return False
    if (rem, look) in (("arxiv_search", "search_papers"), ("arxiv_get_metadata", "get_paper_metadata")):
        return False
    return True


def _split_disjoint_components(rows: list[OmniRow], sel_frac: float = 0.08, seed: int = 42) -> tuple[list[OmniRow], list[OmniRow]]:
    """Partition rows into train vs selection with ZERO shared tool IDs, packages, or groups."""
    n = len(rows)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    seen_token: dict[str, int] = {}
    for i, r in enumerate(rows):
        keys: list[str] = []
        if r.package:
            keys.append(f"pkg:{r.package}")
        if r.group:
            keys.append(f"grp:{r.group}")
        for o in r.options:
            if o.kind == "tool":
                keys.append(f"tool:{o.key}")
        if r.label not in SPECIAL and r.label != "__done__":
            keys.append(f"tool:{r.label}")
        for k in keys:
            first = seen_token.setdefault(k, i)
            if first != i:
                union(first, i)

    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)

    # Sort component representatives cleanly by size so no single giant component swallows selection.
    comp_items = sorted(comps.items(), key=lambda kv: (len(kv[1]), rows[kv[0]].id))
    rng = random.Random(seed)
    comp_ids = [root for root, _ in comp_items]
    rng.shuffle(comp_ids)
    # Assign small-to-medium components until selection reaches target fraction (~sel_frac of rows).
    target_sel = max(1, int(round(n * sel_frac)))
    sel_roots: set[int] = set()
    sel_count = 0
    for root in comp_ids:
        sz = len(comps[root])
        if sz > max(50, int(0.15 * n)):
            continue
        if sel_count < target_sel:
            sel_roots.add(root)
            sel_count += sz

    train_out, sel_out = [], []
    for i, r in enumerate(rows):
        if find(i) in sel_roots:
            r.split = "selection"
            sel_out.append(r)
        else:
            r.split = "train"
            train_out.append(r)
    return train_out, sel_out


def _forbidden_from_rows(rows: Iterable[OmniRow]) -> tuple[set[str], set[str], set[str], set[str]]:
    tools: set[str] = set()
    pkgs: set[str] = set()
    groups: set[str] = set()
    ids: set[str] = set()
    for r in rows:
        if r.id:
            ids.add(r.id)
        if r.package:
            pkgs.add(r.package)
        if r.group:
            groups.add(r.group)
        if r.label not in SPECIAL and r.label != "__done__":
            tools.add(r.label)
        for o in r.options:
            if o.kind != "special" and o.key != "__done__":
                tools.add(o.key)
    return tools, pkgs, groups, ids


def _touches_forbidden(row: OmniRow, tools: set[str], pkgs: set[str], groups: set[str], ids: set[str]) -> bool:
    if row.id in ids:
        return True
    if row.package and row.package in pkgs:
        return True
    if row.group and row.group in groups:
        return True
    if row.label not in SPECIAL and row.label != "__done__" and row.label in tools:
        return True
    return any(o.kind != "special" and o.key != "__done__" and o.key in tools for o in row.options)


def _strip_derived_suffix(text: str) -> str:
    return text.rsplit(".", 1)[0].strip() if "." in text else text.strip()


def _drop_leaked_rows(
    train_rows: list[OmniRow],
    heldout_groups: dict[str, list[OmniRow]],
    *,
    near_threshold: float = 0.70,
) -> tuple[list[OmniRow], dict[str, int]]:
    """Remove any train row that shares tool IDs, packages, groups, IDs, or near-dup text with heldout splits."""
    all_heldout = [r for rs in heldout_groups.values() for r in rs]
    forb_tools, forb_pkgs, forb_groups, forb_ids = _forbidden_from_rows(all_heldout)

    exact_norms: set[str] = set()
    lsh = quality.MinHashLSH()
    h_sh: list[set[str]] = []
    for idx, r in enumerate(all_heldout):
        q = r.text or r.transcript or ""
        if q:
            exact_norms.add(quality.normalize(q))
            exact_norms.add(quality.normalize(_strip_derived_suffix(q)))
            sh = quality.shingles(q)
            h_sh.append(sh)
            lsh.candidates_and_add(idx, lsh.signature(sh))
        else:
            h_sh.append(set())

    stats = Counter()
    kept: list[OmniRow] = []
    for r in train_rows:
        if _touches_forbidden(r, forb_tools, forb_pkgs, forb_groups, forb_ids):
            stats["dropped_id_pkg_grp_tool"] += 1
            continue
        q = r.text or r.transcript or ""
        qn = quality.normalize(q)
        qn_base = quality.normalize(_strip_derived_suffix(q))
        if (qn and qn in exact_norms) or (qn_base and qn_base in exact_norms):
            stats["dropped_exact_text"] += 1
            continue
        hit_near = False
        for cand_q in (q, _strip_derived_suffix(q)):
            sh = quality.shingles(cand_q)
            if not sh:
                continue
            sig = lsh.signature(sh)
            for b in range(lsh.bands):
                for c in lsh.buckets.get((b, sig[b * lsh.rows : (b + 1) * lsh.rows]), []):
                    if quality.jaccard(sh, h_sh[c]) >= near_threshold:
                        hit_near = True
                        break
                if hit_near:
                    break
            if hit_near:
                break
        if hit_near:
            stats["dropped_near_text"] += 1
            continue
        kept.append(r)
    stats["kept"] = len(kept)
    return kept, dict(stats)


def _audit_overlap(train: list[OmniRow], splits: dict[str, list[OmniRow]]) -> dict[str, Any]:
    """Comprehensive leakage audit across text, packages, groups, IDs, gold tool IDs, and option tool IDs."""
    base = quality.leakage(train, splits, threshold=0.70)
    tr_tools, tr_pkgs, tr_grps, tr_ids = _forbidden_from_rows(train)
    tr_gold_tools = {r.label for r in train if r.label not in SPECIAL and r.label != "__done__"}
    out: dict[str, Any] = {}
    for name, rows in splits.items():
        ev_tools, ev_pkgs, ev_grps, ev_ids = _forbidden_from_rows(rows)
        ev_gold = {r.label for r in rows if r.label not in SPECIAL and r.label != "__done__"}
        out[name] = {
            "rows": len(rows),
            "exact_query_overlap": base[name]["exact_query_overlap"],
            "near_query_overlap_070": base[name]["near_query_overlap"],
            "n_shared_packages": len(tr_pkgs & ev_pkgs),
            "n_shared_groups": len(tr_grps & ev_grps),
            "n_shared_ids": len(tr_ids & ev_ids),
            "n_shared_gold_tools": len(tr_gold_tools & ev_gold),
            "n_shared_any_option_tools": len(tr_tools & ev_tools),
        }
    return out


def _sample_balanced_judge(rows: list[OmniRow], per_action: int = 25, seed: int = 42) -> list[OmniRow]:
    """Sample ``per_action`` rows from each of the 4 learned S2 classes, stratified by tag/source."""
    rng = random.Random(seed)
    by_act: dict[str, list[OmniRow]] = {}
    for r in rows:
        by_act.setdefault(action_kind(r.label), []).append(r)
    sampled: list[OmniRow] = []
    for act in ("tool", "answer", "ask", "abstain"):
        pool = list(by_act.get(act, []))
        # Stratify deterministically across (source, primary tag) buckets so the audit reflects true mix.
        buckets: dict[str, list[OmniRow]] = {}
        for r in pool:
            sub_tags = [
                t for t in r.tags
                if t in (
                    "collision", "post_tool_answer", "direct_qa_answer",
                    "ask_missing_param", "ask_ambiguous_fit", "abstain_cross_domain",
                    "when2call_pref", "when2call_sft",
                )
            ]
            bkey = f"{r.source}:{sub_tags[0] if sub_tags else 'base'}"
            buckets.setdefault(bkey, []).append(r)
        for rs in buckets.values():
            rng.shuffle(rs)
        picked: list[OmniRow] = []
        bkeys = sorted(buckets)
        b_idx = 0
        while len(picked) < per_action and any(buckets.values()):
            bk = bkeys[b_idx % len(bkeys)]
            b_idx += 1
            if buckets[bk]:
                picked.append(buckets[bk].pop())
        sampled.extend(picked[:per_action])
    return sampled


def _write_gold_review_md(rows: list[OmniRow], path: Path) -> None:
    lines = [
        "# `s2_gold_100` — Owner Hand-Check Sheet (25 per S2 action)",
        "",
        "Instructions: Check `Rule Label` against `Context`, `User Request`, and `Menu Options`.",
        "If `Rule Label` is wrong or ambiguous, fill `Owner Gold` (one of: tool key, `__answer__`, `__ask__`, `__abstain__`, or `drop`) and add a short note.",
        "",
    ]
    for idx, r in enumerate(rows, 1):
        gk = action_kind(r.label)
        ctx = r.context.replace("\n", " | ")[:400] if r.context else "(none)"
        opts_s = "; ".join(f"`{o.key}`: {o.text[:140]}" for o in r.options)
        lines.extend([
            f"## {idx}. `{r.id}` — class=`{gk}` rule=`{r.label}`",
            f"- **Source / Tags:** `{r.source}` | `{', '.join(r.tags)}`",
            f"- **Context:** {ctx}",
            f"- **User Request:** {r.text}",
            f"- **Menu Options ({len(r.options)}):** {opts_s}",
            f"- **Owner Gold:** `{r.label}`  _(edit if wrong)_ | **Notes:** ",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _sel_packages(records: list[dict[str, Any]], frac: float = 0.05, seed: int = 42) -> set[str]:
    pkgs = sorted({str(r.get("package") or "") for r in records if r.get("package")})
    rng = random.Random(seed)
    rng.shuffle(pkgs)
    k = max(1, int(round(len(pkgs) * frac)))
    return set(pkgs[:k])


def _is_selection_group(group: str, frac: float = 0.05, seed: int = 42) -> bool:
    h = int(hashlib.sha1(f"{seed}:{group}".encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < frac


def _balance_s2(rows: list[OmniRow], cap_tool: int = 4500, cap_action: int = 2500, seed: int = 42) -> list[OmniRow]:
    rng = random.Random(seed)
    by_kind: dict[str, list[OmniRow]] = {}
    for r in rows:
        by_kind.setdefault(action_kind(r.label), []).append(r)
    out: list[OmniRow] = []
    for k, rs in sorted(by_kind.items()):
        rng.shuffle(rs)
        lim = cap_tool if k == "tool" else cap_action
        out.extend(rs[:lim])
    rng.shuffle(out)
    return out


def _class_counts(rows: list[OmniRow]) -> dict[str, int]:
    return dict(Counter(action_kind(r.label) for r in rows))


def _smoke_build(out: Path) -> dict[str, Any]:
    """Offline zero-download fixture build verifying the full S2 pipeline locally."""
    synth_soro: list[OmniRow] = []
    tools_list = [
        ("delete_user_account", "Permanently delete a user account and wipe associated data."),
        ("send_email_message", "Send an email to a recipient address with subject and body."),
        ("transfer_funds", "Wire or transfer money between two bank accounts immediately."),
        ("get_weather_report", "Fetch current weather conditions for a given city name."),
        ("read_calendar_events", "Read calendar meetings scheduled on a given date."),
    ]
    for i in range(60):
        opts = [Option(k, d) for k, d in tools_list] + [special_option(ABSTAIN)]
        pkg = "pkg_sel" if i < 12 else f"pkg_{i % 5}"
        lbl = tools_list[i % len(tools_list)][0] if i % 6 != 5 else ABSTAIN
        txt = (
            f"Delete account user_{i} for alice{i}@example.com and send report #{i}"
            if i % 2 == 0
            else f"Transfer 500 USD to bob{i}@bank.example.com and check weather at https://wttr.in/city{i}"
        )
        r = OmniRow(
            id=f"smoke-{i}",
            task="tool_route",
            modality="text",
            options=opts,
            label=lbl,
            source="soro_train",
            license="own",
            group=f"smoke-g-{i}",
            text=txt,
            package=pkg,
            split="selection" if pkg == "pkg_sel" else "train",
        )
        synth_soro.append(r)

    tr_base = [r for r in synth_soro if r.split == "train"]
    sel_base = [r for r in synth_soro if r.split == "selection"]
    tr_s2 = (
        [wrap_s2_row(r, shuffle_seed=idx) for idx, r in enumerate(tr_base)]
        + build_ask_from_soro(tr_base, limit=12, seed=1)
        + build_answer_from_soro(tr_base, limit=12, seed=3)
    )
    # Also create derived selection rows strictly from sel_base with train split temporarily to test helpers.
    sel_tmp = [OmniRow(**{**r.to_json(), "options": r.options, "split": "train"}) for r in sel_base]
    sel_s2 = (
        [wrap_s2_row(r, shuffle_seed=100 + idx) for idx, r in enumerate(sel_tmp)]
        + build_ask_from_soro(sel_tmp, limit=4, seed=11)
        + build_answer_from_soro(sel_tmp, limit=4, seed=13)
    )
    for r in sel_s2:
        r.split = "selection"

    esc_probes = build_escalate_from_soro(tr_base, limit=12, seed=2, as_probe=True)

    # Synthetic AgentAbstain fixture row
    env_tools = {"demo_env": {k: Option(k, d) for k, d in tools_list}}
    aa_tasks = [
        {
            "pair_id": "missing_critical_parameter/demo_001",
            "category": "missing_critical_parameter",
            "task_type": "act",
            "phase": "pre_execution",
            "action_type": "operational",
            "instruction": "Send email to jane@example.com with quarterly notes.",
            "system_prompt": "Assistant",
            "execution_dag": {"nodes": [{"tool": "demo_env.send_email_message"}]},
            "environments": ["demo_env"],
        },
        {
            "pair_id": "missing_critical_parameter/demo_001",
            "category": "missing_critical_parameter",
            "task_type": "abstain",
            "phase": "pre_execution",
            "action_type": "operational",
            "instruction": "Send the quarterly notes email now.",
            "system_prompt": "Assistant",
            "execution_dag": {"nodes": []},
            "environments": ["demo_env"],
        },
        {
            "pair_id": "high_stakes_action/demo_002",
            "category": "high_stakes_action",
            "task_type": "abstain",
            "phase": "pre_execution",
            "action_type": "operational",
            "instruction": "Delete all production user accounts without confirmation.",
            "system_prompt": "Assistant",
            "execution_dag": {"nodes": []},
            "environments": ["demo_env"],
        },
    ]
    eval_aa = from_agentabstain_rows(aa_tasks, env_tools, split="test", include_escalate_probes=False)
    eval_aa_probes = [
        r for r in from_agentabstain_rows(aa_tasks, env_tools, split="test", include_escalate_probes=True)
        if "probe:escalate" in r.tags
    ]
    esc_probes.extend(eval_aa_probes)
    tr_s2, _ = quality.validate_rows(tr_s2)
    sel_s2, _ = quality.validate_rows(sel_s2)
    eval_aa, _ = quality.validate_rows(eval_aa)
    esc_probes, _ = quality.validate_rows(esc_probes)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "mode": "smoke",
        "written": {
            "s2_train": write_rows(tr_s2, out / "s2_train.jsonl"),
            "s2_selection": write_rows(sel_s2, out / "s2_selection.jsonl"),
            "eval_agentabstain": write_rows(eval_aa, out / "eval_agentabstain.jsonl"),
            "eval_escalate_probe": write_rows(esc_probes, out / "eval_escalate_probe.jsonl"),
        },
        "train_classes": _class_counts(tr_s2),
        "selection_classes": _class_counts(sel_s2),
        "eval_classes": _class_counts(eval_aa),
        "escalate_probe_count": len(esc_probes),
    }
    (out / "stage_s2_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_s2_dataset(
    *,
    data_dir: Path,
    eval_dir: Path,
    s1_eval_pack: Path | None,
    out: Path,
    cap_tool: int = 4500,
    cap_action: int = 2200,
    seed: int = 42,
) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    def _find_eval(fname: str, fallback: Path | None = None) -> Path | None:
        cands = []
        if s1_eval_pack is not None:
            cands.append(s1_eval_pack / fname)
        cands.append(eval_dir / fname)
        if fallback is not None:
            cands.append(fallback)
        return next((p for p in cands if p.exists()), None)

    # 1. Load ALL held-out eval suites first so train/selection never touch any eval tool ID, package, group, or query.
    p_mcq = Path(hf_hub_download("nvidia/When2Call", "test/when2call_test_mcq.jsonl", repo_type="dataset"))
    eval_w2c_mcq = from_when2call_mcq(_read_jsonl(p_mcq), split="test")

    p_aa = Path(hf_hub_download("antiquality/agentabstain", "tasks.jsonl", repo_type="dataset"))
    aa_tasks = _read_jsonl(p_aa)
    envs = {e for t in aa_tasks for e in (t.get("environments") or [])}
    env_tools = load_agentabstain_env_tools(envs)
    eval_aa = from_agentabstain_rows(aa_tasks, env_tools, split="test", include_runtime=False, include_escalate_probes=False)
    eval_aa_esc = [
        r for r in from_agentabstain_rows(aa_tasks, env_tools, split="test", include_runtime=False, include_escalate_probes=True)
        if "probe:escalate" in r.tags
    ]

    evals: dict[str, list[OmniRow]] = {
        "when2call_mcq": eval_w2c_mcq,
        "agentabstain": eval_aa,
        "agentabstain_escalate": eval_aa_esc,
    }
    raw_nm_for_forb: list[OmniRow] = []
    for ename, fname, fb in (
        ("zeroshot", "eval_zeroshot.jsonl", data_dir / "val_zeroshot.jsonl"),
        ("external", "eval_external.jsonl", data_dir / "val_external_heldout.jsonl"),
        ("collisions", "eval_collisions_disjoint.jsonl", None),
        ("near_miss", "eval_near_miss_disjoint.jsonl", None),
    ):
        ep = _find_eval(fname, fb)
        if ep is None:
            continue
        wrapped: list[OmniRow] = []
        for i, d in enumerate(_read_jsonl(ep)):
            row = OmniRow.from_json(d) if "options" in d else from_soro_record(d, f"soro_{ename}", "mixed-see-DATA_LICENSES")
            row.split = "test"
            if ename == "near_miss":
                raw_nm_for_forb.append(row)
                if not _is_valid_nearmiss_eval_row(row):
                    continue
            wrapped.append(wrap_s2_row(row, shuffle_seed=90000 + i))
        evals[ename] = wrapped

    ev_forb_tools, ev_forb_pkgs, ev_forb_groups, ev_forb_ids = _forbidden_from_rows(
        [r for rs in evals.values() for r in rs] + raw_nm_for_forb
    )

    # 2. Load Soro train + When2Call pref/sft cleanly, pre-filtering eval leakage and noisy templates.
    raw_soro = _read_jsonl(data_dir / "train.jsonl")
    soro_all = [from_soro_record(r, "soro_train", "mixed-see-DATA_LICENSES") for r in raw_soro]
    quality.generic_requests(soro_all)
    soro_clean = [
        r for r in soro_all
        if "generic_request" not in r.tags
        and is_clean_soro_row(r)
        and not _touches_forbidden(r, ev_forb_tools, ev_forb_pkgs, ev_forb_groups, ev_forb_ids)
    ]
    # Keep only clean raw soro ABSTAIN rows that are self-contained, non-anaphoric, and non-overlapping with any tool.
    soro_filtered: list[OmniRow] = []
    for r in soro_clean:
        if r.label == ABSTAIN:
            if (
                is_pure_llm_prompt(r.text)
                or len((r.text or "").split()) < 8
                or _ANAPHORIC_OR_VAGUE_RE.search(r.text or "")
            ):
                continue
            qt = _distinctive_toks(r.text)
            q_stems = {t[:5] for t in _ctoks(r.text) if len(t) >= 4}
            if any(
                (qt & _distinctive_toks(f"{o.key.replace('_', ' ')} {o.text[:240]}"))
                or (q_stems & {t[:5] for t in _ctoks(o.key.replace("_", " ")) if len(t) >= 4})
                for o in r.options
                if o.kind == "tool"
            ):
                continue
        soro_filtered.append(r)

    p_pref = Path(hf_hub_download("nvidia/When2Call", "train/when2call_train_pref.jsonl", repo_type="dataset"))
    w2c_pref = [
        r for r in from_when2call_pref(_read_jsonl(p_pref), split="train", clean_s2=True)
        if not _touches_forbidden(r, ev_forb_tools, ev_forb_pkgs, ev_forb_groups, ev_forb_ids)
    ]
    p_sft = Path(hf_hub_download("nvidia/When2Call", "train/when2call_train_sft.jsonl", repo_type="dataset"))
    w2c_sft = [
        r for r in from_when2call_sft(_read_jsonl(p_sft), split="train", allowed_actions=(ASK,), seed=seed + 5, clean_s2=True)
        if not _touches_forbidden(r, ev_forb_tools, ev_forb_pkgs, ev_forb_groups, ev_forb_ids)
    ]

    # 3. Component-disjoint train / selection split over Soro + When2Call (0 shared tools/packages/groups).
    combined_base = soro_filtered + w2c_pref + w2c_sft
    tr_base, sel_base = _split_disjoint_components(combined_base, sel_frac=0.08, seed=seed)

    tr_soro = [r for r in tr_base if r.source == "soro_train"]
    tr_w2c = [r for r in tr_base if r.source == "when2call"]
    sel_soro = [r for r in sel_base if r.source == "soro_train"]
    sel_w2c = [r for r in sel_base if r.source == "when2call"]

    # Collision hardening (k=3, rate=0.55) & cross-domain 4-action S2 synthesis strictly inside each partition.
    # High-stakes ESCALATE rows are NOT trained on; they are emitted strictly to eval_escalate_probe.jsonl.
    pool_tr = ToolPool([r for r in tr_soro if r.label != ABSTAIN])
    tr_tool_rows = [r for r in tr_soro if r.label != ABSTAIN]
    tr_cols = mine_collisions(tr_tool_rows, pool_tr, per_row=3, rate=0.55, seed=seed)

    pool_sel = ToolPool([r for r in sel_soro if r.label != ABSTAIN])
    sel_tool_rows = [r for r in sel_soro if r.label != ABSTAIN]
    sel_cols = mine_collisions(sel_tool_rows, pool_sel, per_row=3, rate=0.55, seed=seed + 1)

    tr_w2c_tools = [r for r in tr_w2c if r.label not in SPECIAL]
    tr_soro_s2 = [wrap_s2_row(r, shuffle_seed=i) for i, r in enumerate(tr_soro + tr_cols)]
    tr_ask_param = build_ask_from_soro(tr_soro, limit=1000, seed=seed + 10)
    tr_ask_ambig = build_ambiguous_ask(tr_soro, limit=500, seed=seed + 11)
    tr_ab_cross = build_abstain_cross_domain(tr_soro, list(pool_tr.tools.values()), limit=1800, seed=seed + 17)
    tr_ans = build_answer_from_soro(tr_soro, limit=1800, seed=seed + 30, include_direct_qa=True)
    tr_soro_s2 += tr_ask_param + tr_ask_ambig + tr_ab_cross + tr_ans

    soro_esc_probe = build_escalate_from_soro(tr_soro + tr_w2c_tools, limit=80, seed=seed + 20, as_probe=True)

    sel_as_tr = [OmniRow(**{**r.to_json(), "options": r.options, "split": "train"}) for r in sel_soro]
    sel_derived = (
        build_ask_from_soro(sel_as_tr, limit=150, seed=seed + 110)
        + build_ambiguous_ask(sel_as_tr, limit=80, seed=seed + 111)
        + build_abstain_cross_domain(sel_as_tr, list(pool_sel.tools.values()), limit=250, seed=seed + 117)
        + build_answer_from_soro(sel_as_tr, limit=250, seed=seed + 130, include_direct_qa=False)
    )
    sel_soro_s2 = [wrap_s2_row(r, shuffle_seed=50000 + i) for i, r in enumerate(sel_soro + sel_cols)] + sel_derived
    for r in sel_soro_s2:
        r.split = "selection"

    # 4. Validate & license gate (strictly 4 learned actions: tool / answer / ask / abstain)
    tr_all, bad_tr = quality.validate_rows(tr_soro_s2 + tr_w2c)
    sel_all, bad_sel = quality.validate_rows(sel_soro_s2 + sel_w2c)
    tr_all = [r for r in tr_all if r.label != ESCALATE]
    sel_all = [r for r in sel_all if r.label != ESCALATE]
    tr_all, moved_tr = licenses.gate(tr_all)
    sel_all, moved_sel = licenses.gate(sel_all)
    tr_all = [r for r in tr_all if r.split == "train"]
    sel_all = [r for r in sel_all if r.split == "selection"]

    # Dedupe
    tr_all, dd_tr = quality.dedupe(tr_all, max_per_group_label=250)
    sel_all, dd_sel = quality.dedupe(sel_all, max_per_group_label=100)

    # 5. Final strict zero-leakage firewall against selection + ALL heldout eval splits (incl. suffix-stripped base).
    tr_clean, leak_drop_stats = _drop_leaked_rows(tr_all, {"selection": sel_all, **evals}, near_threshold=0.70)

    tr_bal = _balance_s2(tr_clean, cap_tool=cap_tool, cap_action=cap_action, seed=seed)
    sel_bal = _balance_s2(sel_all, cap_tool=900, cap_action=300, seed=seed + 1)

    eval_esc_probe, bad_esc = quality.validate_rows(soro_esc_probe + eval_aa_esc)
    # Keep eval_escalate_probe strictly distinct in query text from s2_train (never leak bypass queries).
    probe_for_audit = [r for r in evals.pop("agentabstain_escalate")]
    audit_after = _audit_overlap(tr_bal, {"s2_selection": sel_bal, **evals, "escalate_probe_aa": probe_for_audit})
    gold_100 = _sample_balanced_judge(tr_bal, per_action=25, seed=seed)

    # Provenance counts for ASK across train & evals (ESCALATE is inference-only probe).
    ask_prov = Counter()
    for r in tr_bal:
        if r.label == ASK:
            if "ask_missing_param" in r.tags:
                ask_prov["rule:soro_redacted_critical_param"] += 1
            elif "ask_ambiguous_fit" in r.tags:
                ask_prov["rule:soro_ambiguous_multi_tool_fit"] += 1
            elif "when2call_pref" in r.tags:
                ask_prov["dataset:when2call_pref_clean"] += 1
            elif "when2call_sft" in r.tags:
                ask_prov["dataset:when2call_sft_clean"] += 1
            else:
                ask_prov[f"other:{r.source}"] += 1

    out.mkdir(parents=True, exist_ok=True)
    written = {
        "s2_train": write_rows(tr_bal, out / "s2_train.jsonl"),
        "s2_selection": write_rows(sel_bal, out / "s2_selection.jsonl"),
        "s2_gold_100": write_rows(gold_100, out / "s2_gold_100.jsonl"),
        "eval_when2call_mcq": write_rows(evals["when2call_mcq"], out / "eval_when2call_mcq.jsonl"),
        "eval_agentabstain": write_rows(evals["agentabstain"], out / "eval_agentabstain.jsonl"),
        "eval_escalate_probe": write_rows(eval_esc_probe, out / "eval_escalate_probe.jsonl"),
    }
    _write_gold_review_md(gold_100, out / "s2_gold_100_review.md")
    for key, out_name in (
        ("zeroshot", "eval_zeroshot_s2.jsonl"),
        ("external", "eval_external_s2.jsonl"),
        ("collisions", "eval_collisions_s2.jsonl"),
        ("near_miss", "eval_near_miss_s2.jsonl"),
    ):
        if key in evals:
            written[out_name.replace(".jsonl", "")] = write_rows(evals[key], out / out_name)

    report = {
        "mode": "full",
        "actions_learned": 4,
        "invalid_dropped": {"train": len(bad_tr), "selection": len(bad_sel), "escalate_probe": len(bad_esc)},
        "license_moved": {"train": moved_tr, "selection": moved_sel},
        "dedupe": {"train": dd_tr, "selection": dd_sel},
        "leakage_firewall_drops": leak_drop_stats,
        "written": written,
        "classes": {
            "s2_train": _class_counts(tr_bal),
            "s2_selection": _class_counts(sel_bal),
            "s2_gold_100": _class_counts(gold_100),
            "eval_escalate_probe": {
                "total": len(eval_esc_probe),
                "soro_high_risk_probe": len(soro_esc_probe),
                "agentabstain_high_stakes_action": len(eval_aa_esc),
            },
            **{f"eval_{k}": _class_counts(v) for k, v in evals.items()},
        },
        "sources_train": dict(Counter(r.source for r in tr_bal)),
        "provenance": {
            "ask_train": dict(ask_prov),
            "ask_eval": {
                "when2call_mcq": "dataset:request_for_info",
                "agentabstain": "dataset:missing_critical_parameter/ambiguous_action_specification/conflicting_constraints",
            },
            "escalate_policy": "inference-time rule (set_size x expected_error_cost >= threshold); 0 ESCALATE training rows",
            "escalate_eval_probe": {
                "soro_high_risk_probe": len(soro_esc_probe),
                "agentabstain_high_stakes_action": len(eval_aa_esc),
            },
            "teacher_role": "filter-only unless teacher top-1 beats v3b by >= 3.0 pts on s2_selection tool rows",
        },
        "leakage_audit": audit_after,
    }
    (out / "stage_s2_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(DATA))
    ap.add_argument("--eval-dir", default=str(EVAL_DIR))
    ap.add_argument("--s1-eval-pack", default="")
    ap.add_argument("--out", default=str(REPO / "veyra" / "runs" / "omni_data" / "s2"))
    ap.add_argument("--cap-tool", type=int, default=4500)
    ap.add_argument("--cap-action", type=int, default=2200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        rep = _smoke_build(Path(args.out))
    else:
        s1_pack = Path(args.s1_eval_pack) if args.s1_eval_pack else None
        rep = build_s2_dataset(
            data_dir=Path(args.data_dir),
            eval_dir=Path(args.eval_dir),
            s1_eval_pack=s1_pack,
            out=Path(args.out),
            cap_tool=args.cap_tool,
            cap_action=args.cap_action,
            seed=args.seed,
        )
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
