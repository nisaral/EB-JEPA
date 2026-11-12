"""Stage 2 data builder (CPU only): S2 action-card rows (ANSWER/ASK/ESCALATE/ABSTAIN) + exact cost labels.

    python veyra/omni_data/stage_s2.py --smoke
    python veyra/omni_data/stage_s2.py --out veyra/runs/omni_data/s2

Reads local Soro release/eval files + HF Hub cached datasets (``nvidia/When2Call``,
``antiquality/agentabstain``) with NO GPU and writes validated, license-gated,
leakage-checked OmniRow jsonl files:
  s2_train.jsonl             Balanced S2 training rows (tool / ask / abstain / answer / escalate)
  s2_selection.jsonl         Group- and package-disjoint validation rows
  eval_when2call_mcq.jsonl   When2Call test MCQ benchmark (test split)
  eval_agentabstain.jsonl    AgentAbstain pre-execution benchmark (test split)
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
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "veyra" / "python" / "src"))
sys.path.insert(0, str(REPO / "veyra" / "training"))

from soro.omni_data import licenses, quality  # noqa: E402
from soro.omni_data.s2_actions import (  # noqa: E402
    action_kind,
    build_answer_from_soro,
    build_ask_from_soro,
    build_escalate_from_soro,
    from_agentabstain_rows,
    from_when2call_mcq,
    from_when2call_pref,
    load_agentabstain_env_tools,
    wrap_s2_row,
)
from soro.omni_data.schema import (  # noqa: E402
    ABSTAIN,
    ANSWER,
    ASK,
    ESCALATE,
    OmniRow,
    Option,
    from_soro_record,
    special_option,
    write_rows,
)
from soro.omni_data.text import ToolPool, mine_collisions  # noqa: E402

DATA = REPO / "releases" / "soro-universal-tool-harness"
EVAL_DIR = REPO / "veyra" / "omni_data" / "eval_splits"


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
        + build_escalate_from_soro(tr_base, limit=12, seed=2)
        + build_answer_from_soro(tr_base, limit=12, seed=3)
    )
    # Also create derived selection rows strictly from sel_base with train split temporarily to test helpers.
    sel_tmp = [OmniRow(**{**r.to_json(), "options": r.options, "split": "train"}) for r in sel_base]
    sel_s2 = (
        [wrap_s2_row(r, shuffle_seed=100 + idx) for idx, r in enumerate(sel_tmp)]
        + build_ask_from_soro(sel_tmp, limit=4, seed=11)
        + build_escalate_from_soro(sel_tmp, limit=4, seed=12)
        + build_answer_from_soro(sel_tmp, limit=4, seed=13)
    )
    for r in sel_s2:
        r.split = "selection"

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
    ]
    eval_aa = from_agentabstain_rows(aa_tasks, env_tools, split="test")
    tr_s2, _ = quality.validate_rows(tr_s2)
    sel_s2, _ = quality.validate_rows(sel_s2)
    eval_aa, _ = quality.validate_rows(eval_aa)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "mode": "smoke",
        "written": {
            "s2_train": write_rows(tr_s2, out / "s2_train.jsonl"),
            "s2_selection": write_rows(sel_s2, out / "s2_selection.jsonl"),
            "eval_agentabstain": write_rows(eval_aa, out / "eval_agentabstain.jsonl"),
        },
        "train_classes": _class_counts(tr_s2),
        "selection_classes": _class_counts(sel_s2),
        "eval_classes": _class_counts(eval_aa),
    }
    (out / "stage_s2_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_s2_dataset(
    *,
    data_dir: Path,
    eval_dir: Path,
    out: Path,
    cap_tool: int = 4500,
    cap_action: int = 2500,
    seed: int = 42,
) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    raw_soro = _read_jsonl(data_dir / "train.jsonl")
    sel_pkgs = _sel_packages(raw_soro, frac=0.05, seed=seed)
    soro_all = [from_soro_record(r, "soro_train", "mixed-see-DATA_LICENSES") for r in raw_soro]
    quality.generic_requests(soro_all)
    soro_clean = [r for r in soro_all if "generic_request" not in r.tags]

    tr_soro = [r for r in soro_clean if (r.package or "") not in sel_pkgs]
    sel_soro = [r for r in soro_clean if (r.package or "") in sel_pkgs]

    # Collision hardening strictly inside each partition (zero package cross-over).
    pool_tr = ToolPool(tr_soro)
    tr_cols = mine_collisions(tr_soro, pool_tr, rate=0.25, seed=seed)
    pool_sel = ToolPool(sel_soro)
    sel_cols = mine_collisions(sel_soro, pool_sel, rate=0.25, seed=seed + 1)

    tr_soro_s2 = [wrap_s2_row(r, shuffle_seed=i) for i, r in enumerate(tr_soro + tr_cols)]
    tr_soro_s2 += build_ask_from_soro(tr_soro, limit=1800, seed=seed + 10)
    tr_soro_s2 += build_escalate_from_soro(tr_soro, limit=1800, seed=seed + 20)
    tr_soro_s2 += build_answer_from_soro(tr_soro, limit=1800, seed=seed + 30)

    # For selection derived S2 rows, temporarily mark train so builders accept them, then assign selection.
    sel_as_tr = [OmniRow(**{**r.to_json(), "options": r.options, "split": "train"}) for r in sel_soro]
    sel_derived = (
        build_ask_from_soro(sel_as_tr, limit=250, seed=seed + 110)
        + build_escalate_from_soro(sel_as_tr, limit=250, seed=seed + 120)
        + build_answer_from_soro(sel_as_tr, limit=250, seed=seed + 130, include_direct_qa=False)
    )
    sel_soro_s2 = [wrap_s2_row(r, shuffle_seed=50000 + i) for i, r in enumerate(sel_soro + sel_cols)] + sel_derived
    for r in sel_soro_s2:
        r.split = "selection"

    # HF datasets: When2Call train_pref + test_mcq, and AgentAbstain (eval only).
    p_pref = Path(hf_hub_download("nvidia/When2Call", "train/when2call_train_pref.jsonl", repo_type="dataset"))
    w2c_pref = from_when2call_pref(_read_jsonl(p_pref), split="train")
    tr_w2c, sel_w2c = [], []
    for r in w2c_pref:
        if _is_selection_group(r.group, frac=0.05, seed=seed):
            r.split = "selection"
            sel_w2c.append(r)
        else:
            tr_w2c.append(r)

    p_mcq = Path(hf_hub_download("nvidia/When2Call", "test/when2call_test_mcq.jsonl", repo_type="dataset"))
    eval_w2c_mcq = from_when2call_mcq(_read_jsonl(p_mcq), split="test")

    p_aa = Path(hf_hub_download("antiquality/agentabstain", "tasks.jsonl", repo_type="dataset"))
    aa_tasks = _read_jsonl(p_aa)
    envs = {e for t in aa_tasks for e in (t.get("environments") or [])}
    env_tools = load_agentabstain_env_tools(envs)
    eval_aa = from_agentabstain_rows(aa_tasks, env_tools, split="test", include_runtime=False)

    # Wrap existing held-out zeroshot & external evals with S2 action cards for unified evaluation.
    evals: dict[str, list[OmniRow]] = {
        "when2call_mcq": eval_w2c_mcq,
        "agentabstain": eval_aa,
    }
    for ename, candidates in (
        ("zeroshot", (eval_dir / "eval_zeroshot.jsonl", data_dir / "val_zeroshot.jsonl")),
        ("external", (eval_dir / "eval_external.jsonl", data_dir / "val_external_heldout.jsonl")),
    ):
        ep = next((p for p in candidates if p.exists()), None)
        if ep is not None:
            wrapped = []
            for i, d in enumerate(_read_jsonl(ep)):
                row = OmniRow.from_json(d) if "options" in d else from_soro_record(d, f"soro_{ename}", "mixed-see-DATA_LICENSES")
                row.split = "test"
                wrapped.append(wrap_s2_row(row, shuffle_seed=90000 + i))
            evals[ename] = wrapped

    # Validate & license gate
    tr_all, bad_tr = quality.validate_rows(tr_soro_s2 + tr_w2c)
    sel_all, bad_sel = quality.validate_rows(sel_soro_s2 + sel_w2c)
    tr_all, moved_tr = licenses.gate(tr_all)
    sel_all, moved_sel = licenses.gate(sel_all)
    tr_all = [r for r in tr_all if r.split == "train"]
    sel_all = [r for r in sel_all if r.split == "selection"]

    # Dedupe
    tr_all, dd_tr = quality.dedupe(tr_all, max_per_group_label=250)
    sel_all, dd_sel = quality.dedupe(sel_all, max_per_group_label=100)

    # Leakage guard against selection + all test splits: drop any overlapping query from train!
    leak_vs_splits = quality.leakage(tr_all, {"selection": sel_all, **evals})
    leaked_queries = set()
    for split_name, eval_rows in [("selection", sel_all), *evals.items()]:
        ev_texts = {quality.normalize(r.text) for r in eval_rows if r.text}
        leaked_queries.update(ev_texts)
    before_leak_drop = len(tr_all)
    tr_all = [r for r in tr_all if quality.normalize(r.text) not in leaked_queries]
    dropped_leak_tr = before_leak_drop - len(tr_all)

    tr_bal = _balance_s2(tr_all, cap_tool=cap_tool, cap_action=cap_action, seed=seed)
    sel_bal = _balance_s2(sel_all, cap_tool=900, cap_action=300, seed=seed + 1)

    leak_after = quality.leakage(tr_bal, {"selection": sel_bal, **evals})

    out.mkdir(parents=True, exist_ok=True)
    written = {
        "s2_train": write_rows(tr_bal, out / "s2_train.jsonl"),
        "s2_selection": write_rows(sel_bal, out / "s2_selection.jsonl"),
        "eval_when2call_mcq": write_rows(evals["when2call_mcq"], out / "eval_when2call_mcq.jsonl"),
        "eval_agentabstain": write_rows(evals["agentabstain"], out / "eval_agentabstain.jsonl"),
    }
    if "zeroshot" in evals:
        written["eval_zeroshot_s2"] = write_rows(evals["zeroshot"], out / "eval_zeroshot_s2.jsonl")
    if "external" in evals:
        written["eval_external_s2"] = write_rows(evals["external"], out / "eval_external_s2.jsonl")

    report = {
        "mode": "full",
        "invalid_dropped": {"train": len(bad_tr), "selection": len(bad_sel)},
        "license_moved": {"train": moved_tr, "selection": moved_sel},
        "dedupe": {"train": dd_tr, "selection": dd_sel},
        "dropped_for_exact_eval_overlap": dropped_leak_tr,
        "written": written,
        "classes": {
            "s2_train": _class_counts(tr_bal),
            "s2_selection": _class_counts(sel_bal),
            **{f"eval_{k}": _class_counts(v) for k, v in evals.items()},
        },
        "sources_train": dict(Counter(r.source for r in tr_bal)),
        "leakage_before_clean": {
            k: {kk: v[kk] for kk in ("rows", "exact_query_overlap", "near_query_overlap", "n_shared_packages")}
            for k, v in leak_vs_splits.items()
        },
        "leakage_after_clean": {
            k: {kk: v[kk] for kk in ("rows", "exact_query_overlap", "near_query_overlap", "n_shared_packages")}
            for k, v in leak_after.items()
        },
    }
    (out / "stage_s2_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(DATA))
    ap.add_argument("--eval-dir", default=str(EVAL_DIR))
    ap.add_argument("--out", default=str(REPO / "veyra" / "runs" / "omni_data" / "s2"))
    ap.add_argument("--cap-tool", type=int, default=4500)
    ap.add_argument("--cap-action", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        rep = _smoke_build(Path(args.out))
    else:
        rep = build_s2_dataset(
            data_dir=Path(args.data_dir),
            eval_dir=Path(args.eval_dir),
            out=Path(args.out),
            cap_tool=args.cap_tool,
            cap_action=args.cap_action,
            seed=args.seed,
        )
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
