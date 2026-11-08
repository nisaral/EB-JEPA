"""In-process Kev (Apache-2.0). Uses local PyTorch checkpoints or fast deterministic fallback.

Supports single-turn multi-question System-One evaluation:
- tool_route: choice over candidate tools + "__abstain__"
- arg_mapping: choice mapping unrecognized LLM argument keys to target schema property names
- is_mutation: noul/boolean predicting if a tool mutates state when MCP annotations are missing
- missing_required: noul/boolean predicting if required arguments were omitted

Directly loads SoroKevModel checkpoints (head.pt + config.json + temperature.json)
without requiring external pip dependencies.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from soro.routing.kev_model import SoroKevModel, load_soro_kev_checkpoint

if sys.platform == "win32" and "fcntl" not in sys.modules:
    sys.modules["fcntl"] = types.ModuleType("fcntl")

DEFAULT_MODEL = os.environ.get("SORO_KEV_MODEL", "jaredpalmer/kev-0.8b")
DEFAULT_KEV_PATH = os.environ.get("SORO_KEV_PATH")

MUTATION_TERMS = {
    "create", "update", "delete", "write", "drop", "post", "patch", "remove",
    "insert", "upsert", "modify", "cancel", "refund", "charge", "transfer",
    "send", "publish", "deploy", "kill", "terminate", "set", "reset"
}


import zlib

# The exact instruction every tool_route training row carries.
TRAIN_ROUTE_INSTRUCTIONS = "Which single tool or skill should run for this request? Use only the listed candidates."
LEGACY_ROUTE_INSTRUCTIONS = "Which single tool should run for this request? Use only the listed tools."
_STATE_PREFIXES = ("User Request:", "User:", "Agent:", "Prior tool observation:")


def route_state(query: str) -> str:
    """Wrap a bare query the way training states are written ("User Request: ...")."""
    text = query.strip()
    if text.startswith(_STATE_PREFIXES):
        return text
    return f"User Request: {text}"


def _hash_token(token: str, num_dims: int = 256) -> list[int]:
    h = zlib.crc32(token.encode("utf-8"))
    dims = [h % num_dims]
    if len(token) >= 3:
        for i in range(len(token) - 2):
            sub_h = zlib.crc32(token[i : i + 3].encode("utf-8"))
            dims.append(sub_h % num_dims)
    return dims


def _text_to_vec(text: str, num_dims: int = 256) -> list[float]:
    vec = [0.0] * num_dims
    tokens = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+", text)
    for tok in tokens:
        w = tok.lower()
        for d in _hash_token(w, num_dims):
            vec[d] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm > 0.0 else vec


def _cosine(v1: list[float], v2: list[float]) -> float:
    return sum(a * b for a, b in zip(v1, v2))


def resolve_kev_dir(path: str | None = None) -> Path | str:
    if path:
        p = Path(path)
        if p.exists():
            return p
        if "/" in path and not path.startswith("models--"):
            try:
                from huggingface_hub import snapshot_download
                return snapshot_download(path)
            except Exception:
                return path
        return path

    if DEFAULT_KEV_PATH:
        p = Path(DEFAULT_KEV_PATH)
        if p.exists():
            return p

    hub_root = Path.home() / ".cache" / "huggingface" / "hub"
    for candidate_model in ("models--jaredpalmer--kev-0.8b", "models--jaredpalmer--kev-4b"):
        snap_dir = hub_root / candidate_model / "snapshots"
        if snap_dir.is_dir():
            snaps = sorted(snap_dir.glob("*/head.pt"))
            if snaps:
                return snaps[-1].parent

    return DEFAULT_MODEL


class LocalKevScorer:
    """Multi-question System-One decision scorer with direct checkpoint loading and deterministic fallback."""

    def __init__(
        self,
        run: str | None = None,
        device: str | None = None,
        backend: str = "auto",
    ) -> None:
        self.run = str(resolve_kev_dir(run))
        self.device = device or os.environ.get("SORO_KEV_DEVICE", "cpu")
        self.backend = backend  # "auto", "neural", "heuristic"
        self.error: str | None = None
        self._neural_model: SoroKevModel | None = None
        self._temperature: float = 1.0
        self._q_hat: float | None = None
        self._loaded = False
        self.last_intent: str | None = None
        self.last_raw_logits: dict[str, float] = {}
        self.last_intent_logits: dict[str, float] = {}
        self.last_include_probs: dict[str, float] = {}

    def _remember_route(self, opt_keys: list[str]) -> None:
        detail = getattr(self._neural_model, "_last_route", None) or {}
        raw = detail.get("raw_logits")
        self.last_intent = detail.get("intent")
        self.last_raw_logits = {}
        remembered_keys = list(detail.get("keys") or opt_keys)
        if raw is not None:
            values = raw.detach().float().cpu().tolist()
            self.last_raw_logits = {
                key: float(values[i]) for i, key in enumerate(remembered_keys) if i < len(values)
            }
        intent_logits = detail.get("intent_logits")
        self.last_intent_logits = {}
        if intent_logits is not None:
            names = ("single", "multi", "none")
            vals = intent_logits.detach().float().cpu().tolist()
            self.last_intent_logits = {names[i]: float(vals[i]) for i in range(min(len(names), len(vals)))}
        abstain = self.last_raw_logits.get("__abstain__")
        self.last_include_probs = {}
        if abstain is not None:
            for name, value in self.last_raw_logits.items():
                if name == "__abstain__":
                    continue
                gap = value - abstain
                self.last_include_probs[name] = 1.0 / (1.0 + math.exp(-gap))

    def load(self) -> bool:
        if self._loaded:
            return self._neural_model is not None

        self._loaded = True
        if self.backend == "heuristic":
            return False

        # Attempt to load SoroKevModel checkpoint directly
        run_path = Path(self.run)
        if run_path.exists() and (run_path / "head.pt").exists():
            try:
                model, temp = load_soro_kev_checkpoint(run_path, device=self.device)
                self._neural_model = model
                self._temperature = temp
                self._q_hat = getattr(model, "q_hat", None)
                if self._q_hat is None and (run_path / "temperature.json").exists():
                    try:
                        with (run_path / "temperature.json").open(encoding="utf-8") as f:
                            self._q_hat = json.load(f).get("q_hat")
                    except Exception:
                        pass
                if self._q_hat is None and (run_path / "config.json").exists():
                    try:
                        with (run_path / "config.json").open(encoding="utf-8") as f:
                            c_data = json.load(f)
                            self._q_hat = c_data.get("calibrated_q_hat") or c_data.get("calibration", {}).get("q_hat")
                    except Exception:
                        pass
                return True
            except Exception as exc:
                self.error = f"Failed to load neural checkpoint from {run_path}: {exc}"
                if self.backend == "neural":
                    raise RuntimeError(self.error) from exc
                return False

        # Attempt external kev library if available
        try:
            from kev.checkpoint import LoadOptions
            from kev.predictors import LocalPredictor

            self._external_pred = LocalPredictor(self.run, self.device, LoadOptions())
            return True
        except Exception:
            pass

        if self.backend == "neural":
            raise RuntimeError(f"Neural checkpoint requested but not found at '{self.run}' (head.pt missing).")

        return False

    @property
    def is_available(self) -> bool:
        return self.load()

    @property
    def is_neural(self) -> bool:
        return self.load() and self._neural_model is not None

    @property
    def q_hat(self) -> float | None:
        self.load()
        if self._q_hat is not None:
            return self._q_hat
        if self._neural_model is not None and hasattr(self._neural_model, "q_hat"):
            return getattr(self._neural_model, "q_hat", None)
        return None

    @property
    def backend_name(self) -> str:
        if self.load() and self._neural_model is not None:
            return f"Neural SoroKevModel ({self.run})"
        elif self.load() and hasattr(self, "_external_pred"):
            return f"External Kev ({self.run})"
        return "Heuristic Fallback"

    def evaluate_turn(self, state: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Evaluate one or multiple System-One questions in a single forward pass."""
        if not questions:
            return {"probabilities": {}, "answers": {}}

        # If native SoroKevModel is loaded, run neural forward pass
        if self.load() and self._neural_model is not None:
            return self._neural_evaluate_turn(state, questions)

        # If external Kev predictor is loaded
        if self.load() and hasattr(self, "_external_pred"):
            record = {"state": state, "questions": questions}
            try:
                out = self._external_pred(record)
                return {
                    "probabilities": out.get("probabilities", {}),
                    "answers": out.get("answers", {}),
                }
            except Exception as exc:
                self.error = f"Kev inference failed: {exc}"

        # Fast deterministic fallback (dense hashed embeddings + token overlap)
        return self._fallback_evaluate_turn(state, questions)

    def _neural_evaluate_turn(self, state: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        assert self._neural_model is not None
        probabilities: dict[str, dict[str, float]] = {}
        answers: dict[str, Any] = {}

        with torch.no_grad():
            for q_name, q_spec in questions.items():
                q_type = q_spec.get("type", "choice")
                criteria: dict[str, str] = q_spec.get("criteria", {})
                instructions = q_spec.get("instructions", "")

                if q_type == "choice":
                    if not criteria:
                        probabilities[q_name] = {}
                        answers[q_name] = None
                        continue

                    logits, opt_keys = self._neural_model.score_choice_sequence(state, instructions, criteria)
                    # score_choice_sequence already divides by the learned temperature.
                    probs = F.softmax(logits, dim=-1).cpu().tolist()
                    q_probs = {k: float(probs[i]) for i, k in enumerate(opt_keys)}
                    probabilities[q_name] = q_probs
                    answers[q_name] = max(q_probs.keys(), key=lambda k: q_probs[k])
                    self._remember_route(opt_keys)

                elif q_type in ("noul", "boolean"):
                    if criteria and "yes" in criteria and "no" in criteria:
                        logits, opt_keys = self._neural_model.score_choice_sequence(
                            state, instructions, criteria, max_seq_len=4096
                        )
                        probs = F.softmax(logits, dim=-1).cpu().tolist()
                        q_probs = {k: float(probs[i]) for i, k in enumerate(opt_keys)}
                        probabilities[q_name] = q_probs
                        answers[q_name] = max(q_probs, key=q_probs.get)
                    else:
                        logits = self._neural_model.score_binary(state)
                        probs = F.softmax(logits, dim=-1).cpu().tolist()
                        probabilities[q_name] = {"no": float(probs[0]), "yes": float(probs[1])}
                        answers[q_name] = "yes" if probs[1] > probs[0] else "no"

                else:
                    probabilities[q_name] = {}
                    answers[q_name] = None

        return {"probabilities": probabilities, "answers": answers}

    def _fallback_evaluate_turn(self, state: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        probabilities: dict[str, dict[str, float]] = {}
        answers: dict[str, Any] = {}
        state_vec = _text_to_vec(state)
        state_tokens = set(re.findall(r"\w+", state.lower()))

        for q_name, q_spec in questions.items():
            q_type = q_spec.get("type", "choice")
            criteria: dict[str, str] = q_spec.get("criteria", {})

            if q_type == "choice":
                if not criteria:
                    probabilities[q_name] = {}
                    answers[q_name] = None
                    continue

                scores: dict[str, float] = {}
                for opt_k, opt_desc in criteria.items():
                    if opt_k == "__abstain__":
                        continue
                    opt_text = f"{opt_k} {opt_desc}"
                    opt_vec = _text_to_vec(opt_text)
                    c_sim = _cosine(state_vec, opt_vec)
                    # Name match boost
                    for tok in re.findall(r"\w+", opt_k.lower()):
                        if len(tok) >= 3 and tok in state_tokens:
                            c_sim += 0.35
                    scores[opt_k] = max(c_sim, 0.001)

                if "__abstain__" in criteria:
                    max_score = max(scores.values()) if scores else 0.0
                    scores["__abstain__"] = max(0.20 - max_score, 0.01)

                # Softmax over scores with calibrated temperature (default ~0.075)
                temp = self._temperature if (self._temperature and self._temperature < 0.5) else 0.075
                mx = max(scores.values()) if scores else 0.0
                exps = {k: math.exp((v - mx) / temp) for k, v in scores.items()}
                total = sum(exps.values()) or 1.0
                q_probs = {k: v / total for k, v in exps.items()}
                probabilities[q_name] = q_probs
                answers[q_name] = max(q_probs.keys(), key=lambda k: q_probs[k])

            elif q_type in ("noul", "boolean"):
                # Check mutation indicators
                has_mutation = any(m in state_tokens for m in MUTATION_TERMS)
                if has_mutation:
                    probabilities[q_name] = {"yes": 0.95, "no": 0.05}
                    answers[q_name] = "yes"
                else:
                    probabilities[q_name] = {"yes": 0.05, "no": 0.95}
                    answers[q_name] = "no"

            else:
                probabilities[q_name] = {}
                answers[q_name] = None

        return {"probabilities": probabilities, "answers": answers}

    def predict_batch(
        self,
        records: list[dict[str, Any]],
        max_seq_len: int = 1024,
    ) -> list[dict[str, Any]]:
        """Score several dataset rows in one packed forward. Skipped rows fall back."""
        if not records:
            return []
        if not (self.load() and self._neural_model is not None):
            return [self.evaluate_turn(row["state"], row["questions"]) for row in records]
        with torch.inference_mode():
            out = self._neural_model.forward_batch(records, max_seq_len=max_seq_len)
        if out.get("skipped"):
            return [self.evaluate_turn(row["state"], row["questions"]) for row in records]
        predictions = list(out.get("row_predictions") or [])
        kept = list(out.get("kept_indices") or [])
        by_index = {index: predictions[pos] for pos, index in enumerate(kept) if pos < len(predictions)}
        turns = []
        for index, row in enumerate(records):
            if index in by_index:
                turns.append(by_index[index])
            else:
                turns.append(self.evaluate_turn(row["state"], row["questions"]))
        return turns

    def probabilities(self, query: str, options: dict[str, str]) -> dict[str, float]:
        """Convenience method for tool choice probabilities."""
        if not options:
            return {}
        model = self._neural_model if self.load() else None
        if getattr(model, "prompt_format", "train") == "legacy":
            instructions = LEGACY_ROUTE_INSTRUCTIONS
            state = query
        else:
            instructions = TRAIN_ROUTE_INSTRUCTIONS
            state = route_state(query)
        questions = {
            "tool": {
                "type": "choice",
                "instructions": instructions,
                "criteria": options,
            }
        }
        res = self.evaluate_turn(state, questions)
        probs = res.get("probabilities", {}).get("tool", {})
        return {k: float(probs.get(k, 0.0)) for k in options}
