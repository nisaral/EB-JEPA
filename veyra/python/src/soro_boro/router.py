"""High-level tool router.

>>> from soro_boro import Router
>>> router = Router.from_pretrained("nisaralll/soro-kev-0.8b")   # doctest: +SKIP
>>> d = router.route("What plan is user 123 on?", tools=[...])  # doctest: +SKIP
>>> d.action, d.tool, d.confidence                               # doctest: +SKIP
('EXECUTE', 'check_plan', 0.97)

The router scores every candidate tool plus an ``__abstain__`` option and
applies the split-conformal LAC gate S = {tool : p >= 1 - q_hat}:

* ``EXECUTE`` - exactly one tool in S: call it.
* ``ASK``     - two or more tools in S: ask the user (or a bigger model) to disambiguate.
* ``ABSTAIN`` - no tool in S, or ``__abstain__`` ranks first: answer without a tool.

It does not execute anything. Use :class:`soro.SoroHarness` (or your own
agent loop) to run the chosen tool.
"""
from __future__ import annotations

import enum
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, runtime_checkable

from soro_boro.tool_schema import ToolSpec, normalize_tools

__all__ = ["Router", "RoutingDecision", "Action", "Scorer", "RouterLoadError"]

ABSTAIN_KEY = "__abstain__"
ABSTAIN_TEXT = "None of the listed tools fit the user request."
DEFAULT_MODEL = "nisaralll/soro-kev-0.8b"


class RouterLoadError(RuntimeError):
    """Raised when model weights cannot be loaded."""


class Action(str, enum.Enum):
    EXECUTE = "EXECUTE"
    ASK = "ASK"
    ABSTAIN = "ABSTAIN"

    def __str__(self) -> str:  # "EXECUTE", not "Action.EXECUTE"
        return self.value


@runtime_checkable
class Scorer(Protocol):
    """Anything that maps (query, {name: description}) to probabilities.

    ``options`` always contains ``__abstain__``. Probabilities should sum to ~1.
    """

    def probabilities(self, query: str, options: dict[str, str]) -> dict[str, float]: ...


@dataclass
class RoutingDecision:
    """Typed output of :meth:`Router.route`."""

    action: Action
    tool: str | None
    """The tool to run (EXECUTE), the top suggestion (ASK), or ``None`` (ABSTAIN)."""
    candidates: list[str]
    """Tools in the conformal prediction set, most likely first."""
    scores: dict[str, float]
    """Calibrated probability per tool, including ``__abstain__``."""
    confidence: float
    """Probability of ``tool`` (or of ``__abstain__`` when abstaining)."""
    q_hat: float
    threshold: float
    reason: str
    backend: str = ""
    query: str = ""
    _specs: dict[str, ToolSpec] = field(default_factory=dict, repr=False)

    @property
    def should_execute(self) -> bool:
        return self.action is Action.EXECUTE

    @property
    def tool_spec(self) -> ToolSpec | None:
        return self._specs.get(self.tool) if self.tool else None

    def repair_args(self, arguments: Mapping[str, Any], tool: str | None = None) -> dict[str, Any]:
        """Map an LLM's proposed arguments onto the chosen tool's JSON schema.

        Renames near-miss keys (``user_id`` -> ``customer_id``, camelCase -> snake_case),
        fills declared defaults and coerces JSON types (``"3"`` -> ``3``). Unknown keys
        that match nothing are dropped. Returns a new dict.
        """
        name = tool or self.tool
        spec = self._specs.get(name) if name else None
        if spec is None:
            raise ValueError("no tool selected; pass tool=<name> to repair arguments")
        return repair_args_for_schema(spec.parameters, arguments)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "tool": self.tool,
            "candidates": list(self.candidates),
            "scores": dict(self.scores),
            "confidence": self.confidence,
            "q_hat": self.q_hat,
            "threshold": self.threshold,
            "reason": self.reason,
            "backend": self.backend,
        }


def repair_args_for_schema(schema: Mapping[str, Any], arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Repair ``arguments`` against a JSON-schema ``object`` (see :meth:`RoutingDecision.repair_args`)."""
    import inspect

    from soro.harness.pipeline import repair_arguments

    props = dict((schema or {}).get("properties") or {})
    if not props:
        return dict(arguments)
    params = {
        k: {"type": (v or {}).get("type", "string"), "default": (v or {}).get("default", inspect.Parameter.empty)}
        for k, v in props.items()
    }
    return repair_arguments(params, dict(arguments))


class Router:
    """Calibrated tool router. Construct with :meth:`from_pretrained` or pass any :class:`Scorer`."""

    def __init__(
        self,
        scorer: Scorer | None = None,
        q_hat: float | None = None,
        tools: Iterable[Any] | Mapping[str, Any] | None = None,
    ) -> None:
        if scorer is None:
            from soro.routing.kev_local import LocalKevScorer

            scorer = LocalKevScorer(backend="heuristic")
        if not callable(getattr(scorer, "probabilities", None)):
            raise TypeError("scorer must have a probabilities(query, options) method")
        self.scorer = scorer
        scorer_q = getattr(scorer, "q_hat", None)
        self.q_hat = float(q_hat if q_hat is not None else (scorer_q if scorer_q is not None else 0.9))
        if not 0.0 < self.q_hat <= 1.0:
            raise ValueError(f"q_hat must be in (0, 1], got {self.q_hat}")
        self._default_tools = normalize_tools(tools) if tools is not None else []

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str = DEFAULT_MODEL,
        device: str = "auto",
        q_hat: float | None = None,
        fallback: str = "error",
        **kwargs: Any,
    ) -> "Router":
        """Load a Soro-Kev checkpoint from the Hugging Face Hub or a local directory.

        ``device``: ``"auto"`` (CUDA if available, else CPU), ``"cpu"``, ``"cuda"``, ...
        ``fallback``: ``"error"`` (raise :class:`RouterLoadError`) or ``"heuristic"``
        (warn and use the lexical fallback scorer; useful offline).
        """
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - torch is a hard dependency
            raise RouterLoadError("PyTorch is required: pip install torch") from exc
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            import warnings

            warnings.warn("CUDA requested but not available; falling back to CPU", RuntimeWarning, stacklevel=2)
            device = "cpu"

        from soro.routing.kev_local import LocalKevScorer

        try:
            scorer = LocalKevScorer(run=model_id_or_path, backend="neural", device=device)
            ok = scorer.load()
            err = scorer.error
        except Exception as exc:  # network errors, bad files, ...
            ok, err = False, f"{type(exc).__name__}: {exc}"
        if not ok:
            msg = (
                f"could not load Soro-Kev weights from {model_id_or_path!r}: {err or 'unknown error'}.\n"
                "  - Check the repo id or path (expected adapter_model.safetensors, head.safetensors|head.pt, config.json).\n"
                "  - Offline? Pre-download with `huggingface-cli download "
                f"{model_id_or_path}` or set HF_HUB_OFFLINE=1 with a warm cache.\n"
                "  - Run `soro-boro doctor` for an environment report."
            )
            if fallback == "heuristic":
                import warnings

                warnings.warn(msg + "\nUsing the heuristic fallback scorer.", RuntimeWarning, stacklevel=2)
                return cls(LocalKevScorer(backend="heuristic"), q_hat=q_hat, **kwargs)
            raise RouterLoadError(msg)
        return cls(scorer, q_hat=q_hat, **kwargs)

    @classmethod
    def heuristic(cls, q_hat: float | None = None, **kwargs: Any) -> "Router":
        """A weights-free router (hashed lexical similarity). For tests and demos, not production."""
        from soro.routing.kev_local import LocalKevScorer

        return cls(LocalKevScorer(backend="heuristic"), q_hat=q_hat, **kwargs)

    @property
    def backend(self) -> str:
        return str(getattr(self.scorer, "backend_name", type(self.scorer).__name__))

    # ------------------------------------------------------------------ routing
    def route(
        self,
        query: str,
        tools: Iterable[Any] | Mapping[str, Any] | None = None,
        q_hat: float | None = None,
    ) -> RoutingDecision:
        """Route ``query`` to one of ``tools`` (OpenAI / MCP / LangChain / dict / callable)."""
        from soro.harness.pipeline import conformal_gate

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        specs = normalize_tools(tools) if tools is not None else self._default_tools
        eff_q = float(q_hat if q_hat is not None else self.q_hat)
        by_name = {s.name: s for s in specs}
        if not specs:
            return RoutingDecision(Action.ABSTAIN, None, [], {}, 1.0, eff_q, 1.0 - eff_q,
                                   "No tools provided", self.backend, query, by_name)

        options = {s.name: (s.description or s.name) for s in specs}
        options[ABSTAIN_KEY] = ABSTAIN_TEXT
        raw = self.scorer.probabilities(query, options) or {}
        probs = {k: float(raw.get(k, 0.0)) for k in options}
        probs = dict(sorted(probs.items(), key=lambda kv: kv[1], reverse=True))

        action, selected, s_tools, threshold, reason = conformal_gate(probs, eff_q)
        if action == "ABSTAIN":
            conf = probs.get(ABSTAIN_KEY, 0.0) if next(iter(probs)) == ABSTAIN_KEY else 1.0 - max(
                (p for k, p in probs.items() if k != ABSTAIN_KEY), default=0.0
            )
            return RoutingDecision(Action.ABSTAIN, None, [], probs, conf, eff_q, threshold, reason,
                                   self.backend, query, by_name)
        top = selected or s_tools[0]
        return RoutingDecision(Action(action), top, s_tools, probs, probs.get(top, 0.0), eff_q, threshold,
                               reason, self.backend, query, by_name)

    def route_batch(self, queries: Iterable[str], tools: Iterable[Any] | Mapping[str, Any] | None = None,
                    q_hat: float | None = None) -> list[RoutingDecision]:
        specs = normalize_tools(tools) if tools is not None else None
        return [self.route(q, specs, q_hat=q_hat) for q in queries]

    def __repr__(self) -> str:
        return f"Router(backend={self.backend!r}, q_hat={self.q_hat:.4f})"


def _env_device() -> str:  # pragma: no cover - tiny helper used by CLI/server
    return os.environ.get("SORO_DEVICE", "auto")
