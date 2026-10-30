"""Soro Unified Harness Pipeline: Route -> Repair -> Guard.

A lightweight, developer-first tool harness that chains:
1. Conformal Tool Routing: 2-stage pipeline (Fast shortlist -> Calibrated LocalKev scoring -> Conformal LAC filter).
2. Argument Repair: Auto-coerces LLM type mismatches and maps semantic misnomers to target schema properties.
3. Idempotent Execution Guard: Soro-Boro boundary protecting against runaway LLM duplicate loops and retry double-writes.
"""

from __future__ import annotations

import datetime
import functools
import inspect
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar

from soro_boro import SideEffectClass, TerminalOutcome, soro_wrap
from soro.routing.kev_local import LocalKevScorer, _text_to_vec, _cosine

F = TypeVar("F", bound=Callable[..., Any])

COMMON_SYNONYMS: dict[str, set[str]] = {
    "id": {"identifier", "uuid", "key", "number", "code", "no"},
    "amount": {"value", "val", "total", "cost", "price", "fee", "sum", "quantity"},
    "user": {"customer", "client", "account", "member", "person", "usr"},
    "query": {"search", "q", "term", "text", "prompt", "lookup"},
    "reason": {"explanation", "memo", "description", "details", "comment", "note"},
    "target": {"destination", "dest", "to", "output", "path"},
    "source": {"origin", "src", "from", "input"},
}


@dataclass
class ToolEntry:
    name: str
    description: str
    fn: Callable[..., Any]
    side_effect: SideEffectClass
    parameters: dict[str, Any]
    wrapped_fn: Callable[..., TerminalOutcome[Any]]


@dataclass
class RouteResult:
    """Conformal routing decision output with prediction set S."""

    action: str  # "EXECUTE", "ASK", "ABSTAIN"
    selected_tool: str | None
    candidates: list[str]  # Prediction set S
    probabilities: dict[str, float] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    alpha: float = 0.10
    q_hat: float | None = None
    reason: str = ""

    def __iter__(self):
        return iter(self.candidates)

    def __getitem__(self, idx: int | slice):
        return self.candidates[idx]

    def __len__(self) -> int:
        return len(self.candidates)

    def __bool__(self) -> bool:
        return bool(self.candidates)

    @property
    def tool_name(self) -> str | None:
        return self.selected_tool

    @property
    def candidate_set(self) -> list[str]:
        return self.candidates

    @property
    def is_safe(self) -> bool:
        return self.action == "EXECUTE"


def _split_tokens(name: str) -> set[str]:
    clean = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    return set(re.findall(r"[a-z0-9]+", clean))


def _semantic_param_similarity(input_key: str, target_param: str, target_type: Any) -> float:
    in_toks = _split_tokens(input_key)
    tar_toks = _split_tokens(target_param)
    if not in_toks or not tar_toks:
        return 0.0

    # 1. Exact match
    if input_key.lower() == target_param.lower():
        return 1.0

    # 2. Token overlap (Jaccard)
    overlap = in_toks.intersection(tar_toks)
    overlap_score = len(overlap) / float(len(in_toks | tar_toks)) if (in_toks | tar_toks) else 0.0

    # 3. Synonym matching
    has_synonym = False
    for tar in tar_toks:
        syns = COMMON_SYNONYMS.get(tar, set())
        if syns.intersection(in_toks):
            has_synonym = True
            break
    if not has_synonym:
        for in_t in in_toks:
            syns = COMMON_SYNONYMS.get(in_t, set())
            if syns.intersection(tar_toks):
                has_synonym = True
                break

    # 4. Dense cosine similarity
    vec_in = _text_to_vec(input_key)
    vec_tar = _text_to_vec(target_param)
    dense_sim = max(0.0, _cosine(vec_in, vec_tar))

    score = 0.0
    if overlap_score > 0:
        score = max(score, 0.5 + 0.5 * overlap_score)
    if has_synonym:
        score = max(score, 0.75)
    score = max(score, dense_sim)
    return score


_JSON_TYPES: dict[str, Any] = {
    "integer": int,
    "number": float,
    "boolean": bool,
    "string": str,
    "object": dict,
    "array": list,
}


def conformal_gate(probabilities: dict[str, float], q_hat: float) -> tuple[str, str | None, list[str], float, str]:
    """Split-conformal LAC gate used by both ``SoroHarness.route`` and ``soro_boro.Router``.

    Prediction set S = {i : p_i >= 1 - q_hat}. Returns
    ``(action, selected_tool, set_without_abstain, threshold, reason)`` where action is
    EXECUTE (exactly one tool in S), ASK (two or more), or ABSTAIN (no tool in S, or
    ``__abstain__`` is the top-1 option).
    """
    threshold = 1.0 - float(q_hat)
    ranked = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked:
        return "ABSTAIN", None, [], threshold, "No candidate probabilities available"
    top_tool, top_prob = ranked[0]
    S = [name for name, p in ranked if p >= threshold]
    s_tools = [name for name in S if name != "__abstain__"]
    if not s_tools or top_tool == "__abstain__":
        return (
            "ABSTAIN",
            None,
            [],
            threshold,
            f"Conformal abstention (top={top_tool}, p={top_prob:.3f}, |S|={len(S)}, q_hat={q_hat:.4f})",
        )
    if len(s_tools) == 1:
        sel = s_tools[0]
        return (
            "EXECUTE",
            sel,
            s_tools,
            threshold,
            f"Singleton conformal set ({sel}, p={probabilities.get(sel, 0.0):.3f} >= {threshold:.4f})",
        )
    return (
        "ASK",
        s_tools[0],
        s_tools,
        threshold,
        f"Ambiguous prediction set size {len(s_tools)} under q_hat={q_hat:.4f} (threshold={threshold:.4f})",
    )


def repair_arguments(
    parameters: dict[str, dict[str, Any]],
    arguments: dict[str, Any],
    scorer: Any = None,
    forward_extra: bool = False,
) -> dict[str, Any]:
    """Repair proposed arguments against a parameter table.

    ``parameters`` maps a parameter name to ``{"type": <python type or JSON-schema
    type name>, "default": <value or inspect.Parameter.empty>}``. Keys are matched
    exactly, then camelCase->snake_case, then by token/synonym/hashed-dense
    similarity (>= 0.40), then (if ``scorer`` has ``evaluate_turn``) by the
    model's arg_mapping head. Values are then type-coerced.
    """

    repaired: dict[str, Any] = {}
    raw = dict(arguments)

    # 1. Direct and camelCase -> snake_case matching
    raw_normalized: dict[str, Any] = {}
    matched_keys: set[str] = set()

    for k, v in raw.items():
        raw_normalized[k] = v
        snake = re.sub(r"(?<!^)(?=[A-Z])", "_", k).lower()
        if snake != k:
            raw_normalized.setdefault(snake, v)

    for p_name, p_info in parameters.items():
        if p_name in raw_normalized:
            repaired[p_name] = raw_normalized[p_name]
            matched_keys.add(p_name)
            # Also mark camelCase source key as matched
            for raw_k in raw:
                snake_k = re.sub(r"(?<!^)(?=[A-Z])", "_", raw_k).lower()
                if snake_k == p_name or raw_k == p_name:
                    matched_keys.add(raw_k)

    # 2. Semantic Key Mapping for unrecognized arguments
    unmatched_raw = {k: v for k, v in raw.items() if k not in matched_keys}
    unpopulated_params = {p: info for p, info in parameters.items() if p not in repaired}

    if unmatched_raw and unpopulated_params:
        for raw_k, raw_v in list(unmatched_raw.items()):
            best_param = None
            best_sim = 0.0

            for p_name, p_info in unpopulated_params.items():
                sim = _semantic_param_similarity(raw_k, p_name, p_info.get("type"))
                if sim > best_sim:
                    best_sim = sim
                    best_param = p_name

            # If confidence exceeds threshold, map the argument
            if best_param and best_sim >= 0.40:
                repaired[best_param] = raw_v
                matched_keys.add(raw_k)
                del unpopulated_params[best_param]
            elif unpopulated_params and scorer is not None and hasattr(scorer, "evaluate_turn"):
                criteria = {p: f"Schema parameter: {p}" for p in unpopulated_params}
                criteria["__drop__"] = "This argument does not belong to the tool."
                decision = scorer.evaluate_turn(
                    f"User Request: {raw_k}={raw_v}",
                    {"arg_mapping": {
                        "type": "choice",
                        "instructions": f"Which target parameter does the argument '{raw_k}' correspond to?",
                        "criteria": criteria,
                    }},
                )
                choice = (decision.get("answers") or {}).get("arg_mapping")
                if choice in unpopulated_params:
                    repaired[choice] = raw_v
                    matched_keys.add(raw_k)
                    del unpopulated_params[choice]

    # 3. Type Coercion & Normalization on populated fields
    for p_name, p_info in parameters.items():
        target_type = _JSON_TYPES.get(p_info.get("type"), p_info.get("type")) if isinstance(p_info.get("type"), str) else p_info.get("type")
        val = repaired.get(p_name)

        if val is None:
            if p_info.get("default", inspect.Parameter.empty) != inspect.Parameter.empty:
                repaired[p_name] = p_info.get("default")
            continue

        try:
            # Integer coercion
            if target_type in (int, "int") and isinstance(val, (str, float)) and not isinstance(val, bool):
                repaired[p_name] = int(float(val))
            # Float coercion
            elif target_type in (float, "float") and isinstance(val, (str, int)) and not isinstance(val, bool):
                repaired[p_name] = float(val)
            # Boolean coercion
            elif target_type in (bool, "bool") and isinstance(val, (str, int)):
                if isinstance(val, str):
                    repaired[p_name] = val.strip().lower() in ("true", "1", "yes", "on")
                elif isinstance(val, int):
                    repaired[p_name] = bool(val)
            # Serialized JSON string parsing
            elif target_type in (dict, list) and isinstance(val, str):
                try:
                    repaired[p_name] = json.loads(val)
                except Exception:
                    repaired[p_name] = val
            # ISO Date Normalization
            elif target_type in (str, "str") and isinstance(val, str):
                clean_val = val.strip()
                # Check YYYY/MM/DD or ISO with T
                if "/" in clean_val or "T" in clean_val:
                    for fmt in ("%Y/%m/%d", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%dT%H:%M:%SZ"):
                        try:
                            dt = datetime.datetime.strptime(clean_val.split("T")[0] if "T" in clean_val else clean_val, fmt.split("T")[0])
                            clean_val = dt.strftime("%Y-%m-%d")
                            break
                        except ValueError:
                            pass
                repaired[p_name] = clean_val
            elif target_type in (str, "str") and not isinstance(val, str):
                repaired[p_name] = str(val)
        except (ValueError, TypeError):
            repaired[p_name] = val

    # Forward extra keys only if target function accepts **kwargs
    if forward_extra:
        for k, v in raw.items():
            if k not in repaired:
                repaired[k] = v

    return repaired


class SoroHarness:
    """Type-safe MCP & Agent Tool Harness chaining Conformal Route -> Semantic Repair -> Soro-Boro Guard."""

    def __init__(
        self,
        dedupe_mode: str = "loop_guard",
        window_sec: float = 120.0,
        claims_db_path: str | None = None,
        wal_path: str | None = None,
        scorer: LocalKevScorer | None = None,
        default_alpha: float = 0.10,
        **kwargs: Any,
    ):
        self.dedupe_mode = dedupe_mode
        self.window_sec = window_sec
        self.claims_db_path = claims_db_path
        self.wal_path = wal_path
        self._tools: dict[str, ToolEntry] = {}
        self._scorer = scorer or LocalKevScorer()
        self.default_alpha = default_alpha
        self.kwargs = kwargs

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str = "nisaralll/soro-kev-4b",
        device: str = "auto",
        **kwargs: Any,
    ) -> "SoroHarness":
        """Load a local checkpoint directory or a Hugging Face repo id."""
        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        scorer = LocalKevScorer(run=model_id_or_path, backend="neural", device=device)
        if not scorer.load():
            raise RuntimeError(scorer.error or f"Could not load {model_id_or_path}")
        return cls(scorer=scorer, **kwargs)

    def tool(
        self,
        name: str | None = None,
        description: str | None = None,
        side_effect: str | SideEffectClass = "read_only",
        reconciler: Callable[..., Any] | None = None,
        window_sec: float | None = None,
    ) -> Callable[[F], F]:
        """Decorator to register a tool in the harness with boundary protection."""
        if isinstance(side_effect, str):
            s_clean = side_effect.strip().lower()
            if s_clean in ("mutation", "write", "non_idempotent", "non_idempotent_mutation"):
                sec = SideEffectClass.NON_IDEMPOTENT_MUTATION
            elif s_clean in ("read", "read_only", "readonly", "lookup"):
                sec = SideEffectClass.READ_ONLY
            elif s_clean in ("idempotent", "idempotent_mutation", "upsert"):
                sec = SideEffectClass.IDEMPOTENT_MUTATION
            elif s_clean in ("destructive", "delete", "drop"):
                sec = SideEffectClass.DESTRUCTIVE
            else:
                sec = SideEffectClass(s_clean)
        else:
            sec = side_effect
        eff_window = window_sec if window_sec is not None else self.window_sec

        def decorator(fn: F) -> F:
            t_name = name or getattr(fn, "__name__", "anonymous_tool")
            t_desc = description or getattr(fn, "__doc__", "") or t_name

            # Inspect parameters from signature
            sig = inspect.signature(fn)
            params = {}
            for p_name, param in sig.parameters.items():
                if p_name in ("op", "_sb_intent_id"):
                    continue
                params[p_name] = {
                    "type": param.annotation if param.annotation != inspect.Parameter.empty else Any,
                    "default": param.default if param.default != inspect.Parameter.empty else inspect.Parameter.empty,
                }

            # Wrap with soro_boro boundary
            wrapped = soro_wrap(
                side_effect_class=sec,
                reconciler=reconciler,
                dedupe_mode=self.dedupe_mode,
                window_sec=eff_window,
                claims_db_path=self.claims_db_path,
                wal_path=self.wal_path,
            )(fn)

            entry = ToolEntry(
                name=t_name,
                description=t_desc,
                fn=fn,
                side_effect=sec,
                parameters=params,
                wrapped_fn=wrapped,
            )
            self._tools[t_name] = entry
            return fn

        return decorator

    def register(
        self,
        fn_or_spec: Callable[..., Any] | dict[str, Any],
        name: str | None = None,
        description: str | None = None,
        side_effect: str | SideEffectClass = "read_only",
        reconciler: Callable[..., Any] | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """Register a Python callable, MCP tool definition, or API spec."""
        if isinstance(fn_or_spec, dict):
            spec = fn_or_spec
            t_name = name or spec.get("name") or spec.get("operationId") or "unnamed_action"
            t_desc = description or spec.get("description") or spec.get("summary") or t_name
            t_params = parameters or spec.get("parameters") or spec.get("inputSchema", {}).get("properties") or {}
            sec = side_effect
            if "side_effect" in spec:
                sec = spec["side_effect"]
            elif "readOnlyHint" in spec.get("annotations", {}):
                sec = "read_only" if spec["annotations"]["readOnlyHint"] else "mutation"

            def mock_dispatch(**kwargs):
                return {"action": t_name, "status": "executed", "inputs": kwargs}

            self.register_action(
                name=t_name,
                description=t_desc,
                parameters=t_params,
                fn=mock_dispatch,
                side_effect=sec,
                reconciler=reconciler,
            )
            return

        self.tool(
            name=name,
            description=description,
            side_effect=side_effect,
            reconciler=reconciler,
        )(fn_or_spec)

    def register_action(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        fn: Callable[..., Any],
        side_effect: str | SideEffectClass = "read_only",
        reconciler: Callable[..., Any] | None = None,
        window_sec: float | None = None,
    ) -> None:
        """Register an arbitrary action with explicit parameters dictionary."""
        if isinstance(side_effect, str):
            s_clean = side_effect.strip().lower()
            if s_clean in ("mutation", "write", "non_idempotent", "non_idempotent_mutation"):
                sec = SideEffectClass.NON_IDEMPOTENT_MUTATION
            elif s_clean in ("read", "read_only", "readonly", "lookup"):
                sec = SideEffectClass.READ_ONLY
            elif s_clean in ("idempotent", "idempotent_mutation", "upsert"):
                sec = SideEffectClass.IDEMPOTENT_MUTATION
            elif s_clean in ("destructive", "delete", "drop"):
                sec = SideEffectClass.DESTRUCTIVE
            else:
                sec = SideEffectClass(s_clean)
        else:
            sec = side_effect

        parsed_params = {}
        for p_name, p_val in parameters.items():
            if isinstance(p_val, dict):
                p_type = p_val.get("type", Any)
                p_def = p_val.get("default", inspect.Parameter.empty)
            else:
                p_type = p_val
                p_def = inspect.Parameter.empty
            parsed_params[p_name] = {"type": p_type, "default": p_def}

        eff_window = window_sec if window_sec is not None else self.window_sec
        wrapped = soro_wrap(
            side_effect_class=sec,
            reconciler=reconciler,
            dedupe_mode=self.dedupe_mode,
            window_sec=eff_window,
            claims_db_path=self.claims_db_path,
            wal_path=self.wal_path,
        )(fn)

        entry = ToolEntry(
            name=name,
            description=description,
            fn=fn,
            side_effect=sec,
            parameters=parsed_params,
            wrapped_fn=wrapped,
        )
        self._tools[name] = entry

    def register_rest(
        self,
        endpoint: str,
        method: str = "POST",
        description: str = "",
        parameters: dict[str, Any] | None = None,
        fn: Callable[..., Any] | None = None,
        side_effect: str | SideEffectClass | None = None,
        name: str | None = None,
    ) -> None:
        """Register a REST API endpoint (OpenAPI format)."""
        m_upper = method.upper()
        clean_ep = endpoint.strip("/")
        t_name = name or f"{m_upper}_{re.sub(r'[^a-zA-Z0-9]+', '_', clean_ep)}"
        full_desc = f"[REST: {m_upper} /{clean_ep}] {description}".strip()

        if side_effect is None:
            if m_upper == "GET":
                sec = "read_only"
            elif m_upper in ("PUT", "PATCH"):
                sec = "idempotent"
            elif m_upper == "DELETE":
                sec = "destructive"
            else:
                sec = "mutation"
        else:
            sec = side_effect

        def default_rest_handler(**kwargs):
            return {"endpoint": f"/{clean_ep}", "method": m_upper, "parameters": kwargs}

        self.register_action(
            name=t_name,
            description=full_desc,
            parameters=parameters or {},
            fn=fn or default_rest_handler,
            side_effect=sec,
        )

    def register_skill(
        self,
        skill_name: str,
        description: str = "",
        triggers: list[str] | None = None,
        command_format: str = "",
        parameters: dict[str, Any] | None = None,
        fn: Callable[..., Any] | None = None,
        side_effect: str | SideEffectClass = "read_only",
    ) -> None:
        """Register an Agent Skill (SKILL.md / plugin format)."""
        trig_str = ", ".join(triggers) if triggers else skill_name
        full_desc = f"[SKILL: {skill_name}] triggers: {trig_str} | format: {command_format} | {description}".strip()

        def default_skill_handler(**kwargs):
            return {"skill": skill_name, "executed": True, "arguments": kwargs}

        self.register_action(
            name=skill_name,
            description=full_desc,
            parameters=parameters or {},
            fn=fn or default_skill_handler,
            side_effect=side_effect,
        )

    # --------------------------------------------------------------------------
    # 1. ROUTE: 2-stage Conformal Router (Dense Shortlist -> Kev Scorer -> LAC)
    # --------------------------------------------------------------------------
    def route(
        self,
        query: str,
        candidates: list[dict[str, Any]] | dict[str, str] | None = None,
        top_k: int = 3,
        alpha: float | None = None,
        q_hat: float | None = None,
        **kwargs: Any,
    ) -> RouteResult:
        """2-stage Conformal Router returning calibrated prediction set S and action."""
        if candidates is not None:
            import warnings

            warnings.warn(
                "SoroHarness.route(candidates=...) permanently registers mock tools on the harness and is "
                "deprecated; use soro_boro.Router().route(query, tools=[...]) for stateless routing, or "
                "register real tools with @harness.tool.",
                DeprecationWarning,
                stacklevel=2,
            )
            if isinstance(candidates, list):
                for cand in candidates:
                    if isinstance(cand, dict):
                        c_name = cand.get("name", "")
                        c_desc = cand.get("description", "")
                        c_params = cand.get("parameters", {})
                        c_side = cand.get("side_effect", "read_only")
                        self.register_action(c_name, c_desc, c_params, lambda **kw: {"status": "mock", **kw}, side_effect=c_side)
            elif isinstance(candidates, dict):
                for c_name, c_desc in candidates.items():
                    self.register_action(c_name, str(c_desc), {}, lambda **kw: {"status": "mock", **kw})

        # Resolve q_hat and alpha: default to checkpoint's calibrated q_hat if present
        scorer_q_hat = getattr(self._scorer, "q_hat", None)
        if q_hat is not None:
            eff_q_hat = float(q_hat)
            eff_alpha = alpha if alpha is not None else max(0.0, 1.0 - eff_q_hat)
        elif scorer_q_hat is not None and alpha is None:
            eff_q_hat = float(scorer_q_hat)
            eff_alpha = max(0.0, 1.0 - eff_q_hat)
        else:
            eff_alpha = float(alpha if alpha is not None else self.default_alpha)
            eff_q_hat = float(scorer_q_hat) if scorer_q_hat is not None else (1.0 - eff_alpha)

        if not self._tools:
            return RouteResult(
                action="ABSTAIN",
                selected_tool=None,
                candidates=[],
                alpha=eff_alpha,
                q_hat=eff_q_hat,
                reason="No tools registered",
            )

        query_vec = _text_to_vec(query)
        query_terms = set(re.findall(r"\w+", query.lower()))

        # If a neural LocalKevScorer is active and len(self._tools) <= 16,
        # pass all candidate tools directly to the neural scorer without dropping candidates in a lexical hash filter!
        is_neural = getattr(self._scorer, "is_neural", False) or (
            hasattr(self._scorer, "_neural_model") and self._scorer._neural_model is not None
        )
        if is_neural and len(self._tools) <= 16:
            top_candidates = list(self._tools.keys())
            shortlist_scores = [(1.0, name) for name in top_candidates]
        else:
            # Stage 1: Fast Dense Shortlist (up to 16 candidates)
            shortlist_scores: list[tuple[float, str]] = []
            for name, entry in self._tools.items():
                text = f"{name} {entry.description}"
                tool_vec = _text_to_vec(text)
                sim = _cosine(query_vec, tool_vec)

                entry_terms = set(re.findall(r"\w+", text.lower()))
                overlap = query_terms.intersection(entry_terms)
                # If no term overlaps, dense similarity must be at least 0.28 to be considered relevant
                if not overlap and sim < 0.28:
                    sim = 0.0

                # Name match bonus
                for q in query_terms:
                    if q in name.lower() and len(q) >= 3:
                        sim += 0.35

                shortlist_scores.append((sim, name))

            shortlist_scores.sort(key=lambda x: x[0], reverse=True)
            max_raw_sim = shortlist_scores[0][0] if shortlist_scores else 0.0

            # Out-of-domain check for heuristic fallback
            if max_raw_sim < 0.15 and not is_neural:
                return RouteResult(
                    action="ABSTAIN",
                    selected_tool=None,
                    candidates=[],
                    probabilities={},
                    scores={name: s for s, name in shortlist_scores},
                    alpha=eff_alpha,
                    q_hat=eff_q_hat,
                    reason=f"Top candidate similarity {max_raw_sim:.3f} below minimum semantic threshold",
                )

            top_candidates = [name for _, name in shortlist_scores[:16]]

        # Stage 2: Calibrated System-One Scoring via LocalKevScorer
        options = {name: self._tools[name].description for name in top_candidates}
        options["__abstain__"] = "None of the listed tools fit the user request."
        probs = self._scorer.probabilities(query, options)
        scores_map = {name: s for s, name in shortlist_scores if name in probs}

        sorted_by_prob = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        if not sorted_by_prob:
            return RouteResult(
                action="ABSTAIN",
                selected_tool=None,
                candidates=[],
                probabilities=probs,
                scores=scores_map,
                alpha=eff_alpha,
                q_hat=eff_q_hat,
                reason="No candidate probabilities available",
            )

        top_tool, top_prob = sorted_by_prob[0]
        multi_plan = self._multi_tool_plan()
        if multi_plan:
            return RouteResult(
                action="ASK",
                selected_tool=multi_plan[0],
                candidates=multi_plan,
                probabilities=probs,
                scores=scores_map,
                alpha=eff_alpha,
                q_hat=eff_q_hat,
                reason=(
                    "Ordered multi-tool plan "
                    + ", ".join(multi_plan)
                    + f" (intent={getattr(self._scorer, 'last_intent', None)})"
                ),
            )

        # Form the prediction set strictly via Split Conformal LAC: S = { i | p_i >= 1 - q_hat }
        action, selected, s_tools, _threshold, reason = conformal_gate(probs, eff_q_hat)
        return RouteResult(
            action=action,
            selected_tool=selected,
            candidates=s_tools,
            probabilities=probs,
            scores=scores_map,
            alpha=eff_alpha,
            q_hat=eff_q_hat,
            reason=reason,
        )

    def _multi_tool_plan(self) -> list[str]:
        """Tools whose raw dot beats abstain by enough to clear sigmoid 0.75.

        p_include = sigmoid(z_tool - z_abstain). Softmax is unchanged and still
        used for single-tool conformal routing. A plan is not executed by route().
        """
        include = getattr(self._scorer, "last_include_probs", None) or {}
        intent = getattr(self._scorer, "last_intent", None)
        ranked = sorted(
            ((name, prob) for name, prob in include.items() if name != "__abstain__"),
            key=lambda item: item[1],
            reverse=True,
        )
        qualifying = [name for name, prob in ranked if prob >= 0.75]

        def category(name: str) -> str:
            entry = self._tools.get(name)
            parts = name.lower().replace("-", "_").split("_")
            verb = "_".join(parts[-2:]) if len(parts) >= 2 else name.lower()
            side = entry.side_effect if entry is not None else ""
            return f"{side}:{verb}"

        chosen: list[str] = []
        seen: set[str] = set()
        for name in qualifying:
            cat = category(name)
            if cat in seen:
                continue
            seen.add(cat)
            chosen.append(name)
        compound = intent == "multi" or len(chosen) >= 2
        if compound and len(chosen) >= 2:
            return chosen
        return []

    def run_workflow(
        self,
        query: str,
        raw_args: dict[str, Any] | None = None,
        max_steps: int = 5,
    ) -> list[dict[str, Any]]:
        """Execute one tool per request clause, then stop.

        Masking the winners and asking again makes the same clause pick the
        next issue tracker. A later pass runs only when a read result is
        required and some clause is still unmatched.
        """
        clauses = self._action_clauses(query)
        trace: list[dict[str, Any]] = []
        ambiguous = self._ambiguous_clause(clauses)
        if ambiguous is not None:
            trace.append({
                "action": "ASK",
                "status": "ASK",
                "tool": None,
                "candidates": ambiguous,
                "plan": ambiguous,
                "reason": "Two tools in the same category match this clause",
                "stopped": True,
            })
            return trace
        plan = self._plan_from_clauses(clauses)
        if not plan:
            trace.append({"action": "ABSTAIN", "status": "ABSTAIN", "plan": [], "stopped": True})
            return trace
        context: dict[str, Any] = {}
        for tool_name in sorted(plan, key=self._execution_rank)[:max_steps]:
            payload = self._bind_with_context(query, tool_name, context, raw_args)
            missing = self._missing_required(tool_name, payload)
            if missing:
                trace.append({
                    "action": "ASK",
                    "status": "ASK",
                    "tool": tool_name,
                    "missing": missing,
                    "missing_required_fields": missing,
                    "data": payload,
                    "stopped": True,
                })
                break
            outcome = self.execute(tool_name, payload)
            data = getattr(outcome, "data", outcome)
            status = getattr(outcome, "status", "ok")
            trace.append({
                "action": "EXECUTE",
                "status": status,
                "tool": tool_name,
                "data": data,
                "arguments": payload,
            })
            if status == "ASK":
                trace[-1]["missing"] = (data or {}).get("missing", []) if isinstance(data, dict) else []
                trace[-1]["stopped"] = True
                break
            if isinstance(data, dict):
                for key, value in data.items():
                    if value not in (None, ""):
                        context[key] = value
        return trace

    @staticmethod
    def _action_clauses(query: str) -> list[str]:
        parts = [
            part.strip(" .")
            for part in re.split(r",|\band\b|\bthen\b|;", query, flags=re.IGNORECASE)
            if len(part.strip(" .")) > 8
        ]
        return parts or [query.strip()]

    @staticmethod
    def _clause_negated(clause: str) -> bool:
        return re.search(r"\b(do not|don't|dont|never)\b", clause, flags=re.IGNORECASE) is not None

    def _include_for_text(self, text: str) -> dict[str, float]:
        options = {name: entry.description for name, entry in self._tools.items()}
        options["__abstain__"] = "None of the listed tools fit the user request."
        self._scorer.probabilities(text, options)
        return dict(getattr(self._scorer, "last_include_probs", {}) or {})

    def _span_coverage(self, clause: str, tool_name: str) -> int:
        entry = self._tools.get(tool_name)
        if entry is None:
            return 0
        spans = re.findall(r"\$\s*\d+(?:\.\d+)?|\bsev\d+\b|\b[A-Z]{2,}[-_][A-Z0-9-]+\b|\bv?\d+(?:\.\d+)+\b", clause, flags=re.I)
        titled = re.search(r"\btitled\s+(.+)$", clause, flags=re.I)
        if titled:
            spans.append(titled.group(1).strip(" ."))
        covered = 0
        for span in spans:
            kind = self._span_kind(span)
            if any(self._span_fits_parameter(span, kind, name) for name in entry.parameters):
                covered += 1
        return covered

    def _ambiguous_clause(self, clauses: list[str]) -> list[str] | None:
        for clause in clauses:
            if self._clause_negated(clause):
                continue
            grouped: dict[str, list[str]] = {}
            include = self._include_for_text(clause)
            raw = dict(getattr(self._scorer, "last_raw_logits", {}) or {})
            abstain_logit = float(raw.get("__abstain__", 0.0))
            clause_terms = set(re.findall(r"[a-z0-9]+", clause.lower()))
            for name, entry in self._tools.items():
                if float(raw.get(name, float("-inf"))) <= abstain_logit and include.get(name, 0.0) < 0.75:
                    continue
                tool_terms = set(re.findall(r"[a-z0-9]+", f"{name} {entry.description}".lower()))
                if not (clause_terms & tool_terms):
                    continue
                grouped.setdefault(self._functional_category(name), []).append(name)
            for names in grouped.values():
                if len(names) < 2:
                    continue
                coverage = {name: self._span_coverage(clause, name) for name in names}
                best = max(coverage.values())
                leaders = [name for name in names if coverage[name] == best]
                mentioned = [
                    name for name in leaders
                    if any(token in clause.lower() for token in name.lower().replace("-", "_").split("_") if len(token) >= 4)
                ]
                if len(mentioned) == 1 or len(leaders) == 1:
                    continue
                return leaders
        return None

    def _plan_from_clauses(self, clauses: list[str]) -> list[str]:
        chosen: list[str] = []
        used: set[str] = set()
        single = len([clause for clause in clauses if not self._clause_negated(clause)]) == 1
        for clause in clauses:
            if self._clause_negated(clause):
                continue
            include = self._include_for_text(clause)
            raw = dict(getattr(self._scorer, "last_raw_logits", {}) or {})
            abstain_logit = float(raw.get("__abstain__", 0.0))
            clause_terms = set(re.findall(r"[a-z0-9]+", clause.lower()))
            if single:
                ranked = sorted(
                    ((name, value) for name, value in raw.items() if name in self._tools and name != "__abstain__"),
                    key=lambda item: item[1],
                    reverse=True,
                )
                for name, value in ranked:
                    if value <= abstain_logit:
                        break
                    entry = self._tools[name]
                    tool_terms = set(re.findall(r"[a-z0-9]{4,}", f"{name} {entry.description}".lower()))
                    if clause_terms & tool_terms:
                        chosen.append(name)
                        used.add(self._functional_category(name))
                        break
                if chosen:
                    continue
            best_name = ""
            best_score = 0.0
            for name, prob in include.items():
                if name not in self._tools or prob < 0.75:
                    continue
                category = self._functional_category(name)
                if category in used:
                    continue
                entry = self._tools[name]
                tool_terms = set(re.findall(r"[a-z0-9]+", f"{name} {entry.description}".lower()))
                overlap = len(clause_terms & tool_terms)
                if overlap <= 0:
                    continue
                score = overlap + prob + self._span_coverage(clause, name)
                if score > best_score:
                    best_name = name
                    best_score = score
            if best_name:
                chosen.append(best_name)
                used.add(self._functional_category(best_name))
        return chosen

    def _bind_with_context(
        self,
        query: str,
        tool_name: str,
        context: dict[str, Any],
        raw_args: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload = dict(raw_args or {})
        if not payload:
            payload = self.extract_and_map_arguments(query, tool_name)
        entry = self._tools[tool_name]
        for name in entry.parameters:
            if payload.get(name) not in (None, ""):
                continue
            if context.get(name) not in (None, ""):
                payload[name] = context[name]
                continue
            for key, value in context.items():
                if value in (None, ""):
                    continue
                if self._context_key_fits(key, name):
                    payload[name] = value
                    break
            if payload.get(name) in (None, ""):
                mapped = self._map_context_with_pointer(name, context)
                if mapped is not None:
                    payload[name] = mapped
        return payload

    @staticmethod
    def _context_key_fits(context_key: str, parameter: str) -> bool:
        left = set(re.findall(r"[a-z0-9]{5,}", context_key.lower()))
        right = set(re.findall(r"[a-z0-9]{5,}", parameter.lower()))
        return bool(left & right)

    def _map_context_with_pointer(self, parameter: str, context: dict[str, Any]) -> Any:
        usable = {key: value for key, value in context.items() if value not in (None, "")}
        if not usable or not hasattr(self._scorer, "evaluate_turn"):
            return None
        criteria = {key: f"{key} = {value}" for key, value in usable.items()}
        criteria["__drop__"] = "None of these previous values fill this parameter."
        decision = self._scorer.evaluate_turn(
            f"Parameter to fill: {parameter}",
            {"arg_mapping": {
                "type": "choice",
                "instructions": f"Which previous value fills the parameter '{parameter}'?",
                "criteria": criteria,
            }},
        )
        choice = (decision.get("answers") or {}).get("arg_mapping")
        if choice in usable:
            return usable[choice]
        return None

    def _missing_required(self, tool_name: str, payload: dict[str, Any]) -> list[str]:
        entry = self._tools.get(tool_name)
        if entry is None:
            return []
        missing = []
        for name, info in entry.parameters.items():
            if info.get("default") != inspect.Parameter.empty:
                continue
            if payload.get(name) in (None, ""):
                missing.append(name)
        return missing

    @staticmethod
    def _functional_category(name: str) -> str:
        lowered = name.lower().replace("-", "_")
        groups = (
            ("issue", ("incident", "issue", "ticket")),
            ("deployment", ("rollback", "deploy")),
            ("notification", ("slack", "email", "sms", "message", "notify")),
            ("refund", ("refund",)),
            ("shipment", ("shipment",)),
            ("order_lookup", ("order_status", "get_order")),
            ("invoice", ("invoice",)),
            ("weather", ("weather",)),
        )
        group = lowered
        for category, keys in groups:
            if any(key in lowered for key in keys):
                group = category
                break
        verb = lowered.split("_", 1)[0]
        if verb in {"list", "get", "create", "query", "send", "cancel", "issue", "read", "delete", "update"}:
            return f"{verb}:{group}"
        return group

    def _plan_for_clauses(
        self,
        clauses: list[str],
        include: dict[str, float],
        blocked: set[str],
    ) -> list[str]:
        """At most one qualifying tool for each clause and each category."""
        qualifying = [
            name for name, prob in include.items()
            if name != "__abstain__" and prob >= 0.75 and name in self._tools
            and self._functional_category(name) not in blocked
        ]
        chosen: list[str] = []
        used = set(blocked)
        for clause in clauses:
            clause_terms = set(re.findall(r"[a-z0-9]+", clause.lower()))
            best_name = ""
            best_score = 0.0
            for name in qualifying:
                category = self._functional_category(name)
                if category in used:
                    continue
                entry = self._tools[name]
                tool_terms = set(re.findall(r"[a-z0-9]+", f"{name} {entry.description}".lower()))
                overlap = len(clause_terms & tool_terms)
                if overlap <= 0:
                    continue
                score = overlap + float(include.get(name, 0.0))
                if score > best_score:
                    best_name = name
                    best_score = score
            if best_name:
                chosen.append(best_name)
                used.add(self._functional_category(best_name))
        return chosen

    def _execution_rank(self, name: str) -> tuple[int, int]:
        entry = self._tools.get(name)
        side = entry.side_effect if entry is not None else None
        read_first = 0 if side == SideEffectClass.READ_ONLY else 1
        notify_last = 1 if self._functional_category(name) == "notification" else 0
        destructive = 1 if side == SideEffectClass.DESTRUCTIVE else 0
        return (notify_last, read_first, destructive)

    # --------------------------------------------------------------------------
    # 2. REPAIR: Rule-Based Coercions + Semantic Key Mapping
    # --------------------------------------------------------------------------
    def extract_and_map_arguments(self, query: str, tool_name: str) -> dict[str, Any]:
        """Pull values out of a raw sentence and bind them to schema parameters.

        Quoted strings, numbers, emails, paths, dates, and enum tokens are the
        candidates. The pointer head maps a candidate onto a parameter name.
        Type coercion then follows the schema.
        """
        entry = self._tools.get(tool_name)
        if entry is None:
            return {}
        values = re.findall(r"'([^']+)'|\"([^\"]+)\"", query)
        spans = [a or b for a, b in values]
        spans += re.findall(r"\b[\w.+-]+@[\w.-]+\.\w+\b", query)
        spans += re.findall(r"(?:/|\\)[\w./\\-]+", query)
        spans += re.findall(r"\b\d{4}-\d{2}-\d{2}\b", query)
        spans += re.findall(r"\$\s*\d+(?:\.\d+)?", query)
        spans += re.findall(r"\b\d+(?:\.\d+)+\b", query)
        spans += re.findall(r"\b[A-Z]{2,}[-_][A-Z0-9-]+\b", query)
        spans += re.findall(r"\bv\d+(?:\.\d+)+\b", query, flags=re.I)
        spans += re.findall(r"\b[a-z0-9]+(?:-[a-z0-9]+){1,}\b", query)
        spans += re.findall(r"\bsev\d+\b", query, flags=re.I)
        titled = re.search(r"\btitled\s+(.+)$", query, flags=re.I)
        if titled:
            spans.append(titled.group(1).strip(" ."))
        spans = [s for s in spans if not any(s != other and s in other for other in spans)]
        for info in entry.parameters.values():
            for enum in (info.get("enum") or []):
                if str(enum) in query and str(enum) not in spans:
                    spans.append(str(enum))
        bound: dict[str, Any] = {}
        names = list(entry.parameters)
        if not names or not spans:
            return bound
        for span in spans:
            kind = self._span_kind(span)
            allowed = [
                name for name in names
                if name not in bound and self._span_fits_parameter(span, kind, name)
            ]
            if not allowed:
                continue
            if kind == "currency":
                choice = next(
                    (name for name in allowed if any(key in name.lower() for key in ("amount", "price", "credit", "total", "usd", "cost"))),
                    None,
                )
            elif kind == "id":
                choice = next((name for name in allowed if name.lower() == "id" or name.lower().endswith("_id")), None)
            elif kind == "version":
                choice = next((name for name in allowed if "version" in name.lower()), None)
            elif kind == "severity":
                choice = next((name for name in allowed if "severity" in name.lower()), None)
            else:
                criteria = {name: f"Schema parameter: {name}" for name in allowed}
                criteria["__drop__"] = "This span is not a parameter value."
                decision = self._scorer.evaluate_turn(
                    f"User Request: {query}\nSpan: {span}",
                    {"arg_mapping": {
                        "type": "choice",
                        "instructions": f"Which parameter does the value '{span}' fill?",
                        "criteria": criteria,
                    }},
                )
                choice = (decision.get("answers") or {}).get("arg_mapping")
                if choice in ("__drop__", None):
                    choice = None
            if choice and choice in entry.parameters and choice not in bound:
                value: Any = span
                if kind == "currency":
                    value = float(re.sub(r"[^0-9.]", "", span))
                bound[choice] = value
        return self.repair(tool_name, bound)

    @staticmethod
    def _span_kind(span: str) -> str:
        if re.search(r"\$\s*\d", span) or re.fullmatch(r"\d+\.\d{2}", span):
            return "currency"
        if re.fullmatch(r"[A-Z]{2,}[-_][A-Z0-9-]+", span):
            return "id"
        if re.fullmatch(r"v?\d+(?:\.\d+)+", span, flags=re.IGNORECASE):
            return "version"
        if re.fullmatch(r"sev\d+", span, flags=re.IGNORECASE):
            return "severity"
        return "text"

    @staticmethod
    def _span_fits_parameter(span: str, kind: str, name: str) -> bool:
        lowered = name.lower()
        is_identifier = lowered == "id" or lowered.endswith("_id")
        if kind == "currency":
            return not is_identifier and any(
                key in lowered for key in ("amount", "price", "credit", "total", "usd", "cost", "fee")
            )
        if kind == "id":
            if not is_identifier:
                return False
            prefix = span.split("-", 1)[0].lower()
            if prefix.startswith("ord") and "shipment" in lowered:
                return False
            if prefix.startswith("shp") and "order" in lowered and "shipment" not in lowered:
                return False
            return True
        if kind in ("version", "severity"):
            return kind in lowered
        return not is_identifier or bool(re.search(r"[A-Za-z]", span))

    def repair(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Repair and validate proposed arguments against the tool's schema."""
        entry = self._tools.get(tool_name)
        if not entry:
            return dict(arguments)
        sig = inspect.signature(entry.fn)
        has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        return repair_arguments(entry.parameters, arguments, scorer=self._scorer, forward_extra=has_var_kwargs)

    # --------------------------------------------------------------------------
    # 3. GUARD: Execute with loop guard and idempotency boundary
    # --------------------------------------------------------------------------
    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        **extra_kwargs: Any,
    ) -> TerminalOutcome[Any]:
        """Repair arguments and safely execute the tool via Soro-Boro boundary."""
        entry = self._tools.get(tool_name)
        if not entry:
            raise KeyError(f"Tool '{tool_name}' is not registered in SoroHarness.")

        raw_args = dict(arguments or {})
        raw_args.update(extra_kwargs)
        clean_args = self.repair(tool_name, raw_args)
        missing = self._missing_required(tool_name, clean_args)
        if missing:
            from soro_boro.lifecycle import ProviderLifecycleState
            return TerminalOutcome(
                status="ASK",
                data={"missing": missing, "tool": tool_name, "arguments": clean_args},
                provider_state=ProviderLifecycleState.FAILED,
                operation_id="",
                idempotency_key="",
                audit_hash="",
                reason=f"Missing required parameters: {missing}",
            )

        return entry.wrapped_fn(**clean_args)

    # --------------------------------------------------------------------------
    # FULL PIPELINE: Route -> Repair -> Guard
    # --------------------------------------------------------------------------
    def run(
        self,
        query: str,
        arguments: dict[str, Any] | None = None,
        tool_name: str | None = None,
        alpha: float | None = None,
    ) -> TerminalOutcome[Any]:
        """Chain Route -> Repair -> Guard in a single call."""
        target_tool = tool_name
        route_res = None
        if not target_tool:
            route_res = self.route(query, top_k=8, alpha=alpha)
            if route_res.action == "ABSTAIN" or not route_res.candidates:
                from soro_boro.lifecycle import ProviderLifecycleState, TerminalOutcome
                return TerminalOutcome(
                    status="ABSTAIN",
                    data=None,
                    provider_state=ProviderLifecycleState.FAILED,
                    operation_id="",
                    idempotency_key="",
                    audit_hash="",
                    reason=route_res.reason,
                    metadata={"probabilities": route_res.probabilities, "candidates": route_res.candidates},
                )
            if route_res.action == "ASK":
                from soro_boro.lifecycle import ProviderLifecycleState, TerminalOutcome
                return TerminalOutcome(
                    status="ASK",
                    data={"candidates": route_res.candidates},
                    provider_state=ProviderLifecycleState.FAILED,
                    operation_id="",
                    idempotency_key="",
                    audit_hash="",
                    reason=route_res.reason,
                    metadata={"probabilities": route_res.probabilities},
                )
            target_tool = route_res.selected_tool or route_res.candidates[0]

        payload = dict(arguments or {})
        if not payload:
            payload = self.extract_and_map_arguments(query, target_tool)
        repaired = self.repair(target_tool, payload)
        entry = self._tools[target_tool]
        missing = [
            name for name, info in entry.parameters.items()
            if name not in repaired and info.get("default") == inspect.Parameter.empty
        ]
        if missing:
            from soro_boro.lifecycle import ProviderLifecycleState, TerminalOutcome
            return TerminalOutcome(
                status="ASK",
                data={"missing": missing, "tool": target_tool, "repaired": repaired},
                provider_state=ProviderLifecycleState.FAILED,
                operation_id="",
                idempotency_key="",
                audit_hash="",
                reason=f"Missing required parameters: {missing}",
                metadata={"route": None if route_res is None else route_res.probabilities},
            )
        return self.execute(target_tool, arguments=repaired)

    @property
    def tools(self) -> list[str]:
        return list(self._tools.keys())
