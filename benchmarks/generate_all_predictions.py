"""
Generate Per-Row Predictions for All Systems and Splits.

Outputs per-row JSONL predictions to benchmarks/predictions/<split>/<model>.jsonl
Each line contains:
{
  "id": int,
  "package": str,
  "label": str,
  "top_tool": str,
  "is_correct": bool,
  "is_abstain_row": bool,
  "is_shared_tool": bool,
  "has_collision": bool,
  "bfcl_file": str,
  "latency_ms": float,
  "probabilities": dict[str, float]
}

Strictly single-source: all benchmark tables in README and model cards
are computed directly from these prediction files by reconcile_and_verify.py.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "veyra" / "python" / "src"))

from soro.routing.kev_local import LocalKevScorer

SHARED_TOOLS = {
    "check_service_status",
    "list_processes",
    "reverse_geocode",
    "transfer_to_human_agents",
}

BFCL_FILES = {
    "BFCL_v3_multiple.json",
    "BFCL_v3_live_multiple.json",
    "BFCL_v3_irrelevance.json",
}


def extract_query(state: str) -> str:
    match = re.search(r"User Request:\s*(.+?)(?:\nAgent:|\nPrior tool observation:|\nUser:|$)", state, re.DOTALL)
    if match:
        return match.group(1).strip()
    return state.strip()


def parse_candidate_text(name: str, desc: str) -> str:
    if name == "__abstain__":
        return "None of the listed tools or skills fit the user request."
    if desc.startswith("{") and desc.endswith("}"):
        try:
            data = json.loads(desc)
            t_name = data.get("name", name)
            t_desc = data.get("description", "")
            props = list(data.get("inputSchema", {}).get("properties", {}).keys()) if "inputSchema" in data else []
            return f"{t_name} {t_desc} {' '.join(props)}".strip()
        except Exception:
            pass
    clean = re.sub(r"[^a-zA-Z0-9_\s\.\-]", " ", desc)
    clean = re.sub(r"\s+", " ", clean).strip()
    return f"{name} {clean}"


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


# ---------------------------------------------------------
# Baseline Scorers
# ---------------------------------------------------------

class BM25Scorer:
    def __init__(self, k1: float = 1.5, b: float = 0.75, abstain_threshold: float = 0.5):
        self.k1 = k1
        self.b = b
        self.abstain_threshold = abstain_threshold

    def score(self, query: str, candidates: Dict[str, str]) -> Tuple[Dict[str, float], float]:
        t0 = time.perf_counter()
        q_tokens = tokenize(query)
        tool_candidates = {k: v for k, v in candidates.items() if k != "__abstain__"}
        if not tool_candidates:
            latency = (time.perf_counter() - t0) * 1000
            return {"__abstain__": 1.0}, latency

        doc_tokens = {name: tokenize(parse_candidate_text(name, desc)) for name, desc in tool_candidates.items()}
        N = len(tool_candidates)
        avgdl = sum(len(d) for d in doc_tokens.values()) / max(N, 1)

        df = Counter()
        for d in doc_tokens.values():
            df.update(set(d))

        raw_scores = {}
        for name, d in doc_tokens.items():
            doc_len = len(d)
            tf = Counter(d)
            score = 0.0
            for qt in q_tokens:
                if qt in tf:
                    n_qt = df[qt]
                    idf = math.log(1.0 + (N - n_qt + 0.5) / (n_qt + 0.5))
                    f = tf[qt]
                    denom = f + self.k1 * (1.0 - self.b + self.b * (doc_len / max(avgdl, 1e-6)))
                    score += idf * (f * (self.k1 + 1.0)) / max(denom, 1e-6)
            raw_scores[name] = score

        max_raw = max(raw_scores.values()) if raw_scores else 0.0
        exp_s = {k: math.exp(v / 1.5) for k, v in raw_scores.items()}
        sum_exp = sum(exp_s.values())
        tool_probs = {k: v / sum_exp for k, v in exp_s.items()}

        if max_raw < self.abstain_threshold:
            p_abstain = min(0.95, (self.abstain_threshold - max_raw) / self.abstain_threshold * 0.8 + 0.2)
            scale = 1.0 - p_abstain
            res = {k: v * scale for k, v in tool_probs.items()}
            res["__abstain__"] = p_abstain
        else:
            p_abstain = 0.02
            scale = 0.98
            res = {k: v * scale for k, v in tool_probs.items()}
            res["__abstain__"] = p_abstain
        latency = (time.perf_counter() - t0) * 1000
        return res, latency


def _with_abstain(tool_probs: Dict[str, float], top_score: float, threshold: float) -> Dict[str, float]:
    """Attach an __abstain__ mass from a fixed (not fitted) threshold on the top raw score."""
    if top_score < threshold:
        p_abstain = min(0.95, (threshold - top_score) / max(abs(threshold), 1e-6) * 0.8 + 0.2)
    else:
        p_abstain = 0.02
    scale = 1.0 - p_abstain
    res = {k: v * scale for k, v in tool_probs.items()}
    res["__abstain__"] = p_abstain
    return res


def _softmax(scores: Dict[str, float], temp: float) -> Dict[str, float]:
    mx = max(scores.values())
    exp_s = {k: math.exp((v - mx) / temp) for k, v in scores.items()}
    tot = sum(exp_s.values())
    return {k: v / tot for k, v in exp_s.items()}


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


class DenseCosineScorer:
    """Bi-encoder retrieval: sentence-transformers/all-MiniLM-L6-v2 cosine similarity.

    Predictions and latency both come from the same MiniLM forward pass.
    The abstain threshold is a fixed constant, not fitted on any split.
    """

    MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, abstain_threshold: float = 0.25, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.abstain_threshold = abstain_threshold
        self.device = device
        self.model = SentenceTransformer(self.MODEL_ID, device=device)

    def score(self, query: str, candidates: Dict[str, str]) -> Tuple[Dict[str, float], float]:
        tool_candidates = {k: v for k, v in candidates.items() if k != "__abstain__"}
        if not tool_candidates:
            return {"__abstain__": 1.0}, 0.0
        names = list(tool_candidates)
        texts = [query] + [parse_candidate_text(k, tool_candidates[k]) for k in names]
        _sync(self.device)
        t0 = time.perf_counter()
        emb = self.model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        sims = (emb[1:] @ emb[0]).float().cpu().tolist()
        _sync(self.device)
        latency = (time.perf_counter() - t0) * 1000
        raw = {k: float(s) for k, s in zip(names, sims)}
        return _with_abstain(_softmax(raw, 0.05), max(raw.values()), self.abstain_threshold), latency


class CrossEncoderScorer:
    """Zero-shot NLI cross-encoder: cross-encoder/nli-deberta-v3-small.

    Each (request, "The request needs the tool <name>: <description>") pair is
    scored jointly; the tool score is the entailment logit. The abstain
    threshold on the top entailment probability is fixed, not fitted.
    """

    MODEL_ID = "cross-encoder/nli-deberta-v3-small"
    ENTAILMENT = 1  # id2label: 0 contradiction, 1 entailment, 2 neutral

    def __init__(self, abstain_threshold: float = 0.5, device: str = "cpu"):
        from sentence_transformers import CrossEncoder

        self.abstain_threshold = abstain_threshold
        self.device = device
        self.model = CrossEncoder(self.MODEL_ID, device=device, max_length=256)

    def score(self, query: str, candidates: Dict[str, str]) -> Tuple[Dict[str, float], float]:
        tool_candidates = {k: v for k, v in candidates.items() if k != "__abstain__"}
        if not tool_candidates:
            return {"__abstain__": 1.0}, 0.0
        names = list(tool_candidates)
        pairs = [
            (query, f"The request needs the tool {parse_candidate_text(k, tool_candidates[k])}")
            for k in names
        ]
        _sync(self.device)
        t0 = time.perf_counter()
        logits = self.model.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
        _sync(self.device)
        latency = (time.perf_counter() - t0) * 1000
        ent_logit = {k: float(row[self.ENTAILMENT]) for k, row in zip(names, logits)}
        ent_prob = {
            k: float(math.exp(row[self.ENTAILMENT]) / sum(math.exp(x) for x in row))
            for k, row in zip(names, logits)
        }
        return _with_abstain(_softmax(ent_logit, 1.0), max(ent_prob.values()), self.abstain_threshold), latency


class UntrainedOptionLogitsScorer:
    """Untrained readout: the base Qwen3.5-0.8B LM picks an option letter.

    The menu (tools plus a "none of these" option) is listed as A, B, C, ...
    and the next-token logits over those letters are the scores. No LoRA,
    no pointer head, no fine-tuning: this is what the backbone knows alone.
    """

    MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
    LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def __init__(self, device: str = "cpu"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.tok = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(self.MODEL_ID, dtype=dtype).to(device)
        self.model.eval()
        self.letter_ids = [self.tok.encode(" " + c, add_special_tokens=False)[0] for c in self.LETTERS]

    def score(self, query: str, candidates: Dict[str, str]) -> Tuple[Dict[str, float], float]:
        names = [k for k in candidates if k != "__abstain__"] + ["__abstain__"]
        lines = []
        for i, k in enumerate(names):
            if k == "__abstain__":
                lines.append(f"{self.LETTERS[i]}. none: None of the listed tools fit the request.")
            else:
                lines.append(f"{self.LETTERS[i]}. {parse_candidate_text(k, candidates[k])[:400]}")
        prompt = (
            f"User request: {query}\n\nTools:\n" + "\n".join(lines)
            + "\n\nWhich option should handle the request? Answer with one letter.\nAnswer:"
        )
        ids = self.tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(self.device)
        _sync(self.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(**ids).logits[0, -1].float()
        _sync(self.device)
        latency = (time.perf_counter() - t0) * 1000
        sub = logits[self.letter_ids[: len(names)]]
        probs = torch.softmax(sub, dim=-1).cpu().tolist()
        return {k: float(p) for k, p in zip(names, probs)}, latency


# ---------------------------------------------------------
# Soro-Kev Wrapper
# ---------------------------------------------------------

class SoroKevEvaluator:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.scorer = LocalKevScorer(run=checkpoint_path, backend="neural", device=device)
        loaded = self.scorer.load()
        if not loaded:
            raise RuntimeError(f"Could not load neural SoroKev checkpoint from {checkpoint_path}")

    def score(self, state: str, candidates: Dict[str, str]) -> Tuple[Dict[str, float], float]:
        t0 = time.perf_counter()
        with torch.inference_mode():
            probs = self.scorer.probabilities(state, candidates)
        latency = (time.perf_counter() - t0) * 1000
        return probs, latency


# ---------------------------------------------------------
# Local Dedicated Routing & Function-Calling Evaluators
# ---------------------------------------------------------

class ArchRouterEvaluator:
    """
    Evaluator for katanemo/Arch-Router-1.5B
    Outputs JSON: {"route": "..."}
    Maps {"route": "other"} or no-match to "__abstain__".
    """
    def __init__(self, model_id: str = "katanemo/Arch-Router-1.5B", device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device
        print(f"Loading {model_id} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.float16,
            ).to(device)
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.float16,
                device_map=device,
            )

    def score(self, query: str, candidates: Dict[str, str]) -> Tuple[Dict[str, float], float]:
        t0 = time.perf_counter()
        tool_candidates = {k: v for k, v in candidates.items() if k != "__abstain__"}
        if not tool_candidates:
            latency = (time.perf_counter() - t0) * 1000
            return {"__abstain__": 1.0}, latency

        routes = [
            {"name": k, "description": parse_candidate_text(k, desc)[:250]}
            for k, desc in tool_candidates.items()
        ]

        prompt = (
            f"You are a helpful assistant designed to find the best suited route.\n"
            f"You are provided with route description within <routes></routes> XML tags:\n"
            f"<routes>\n{json.dumps(routes)}\n</routes>\n\n"
            f"<conversation>\n{json.dumps([{'role': 'user', 'content': query}])}\n</conversation>\n\n"
            f"Based on your analysis, provide your response in the following JSON formats if you find a matching route:\n"
            f'{{"route": "route_name"}}'
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=24,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
            )
        gen_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        resp = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        latency = (time.perf_counter() - t0) * 1000

        chosen_tool = "__abstain__"
        route_matches = re.findall(r"['\"]route['\"]\s*:\s*['\"]?([^'\"}\s,]+)['\"]?", resp)
        for r in route_matches:
            r = r.strip()
            if r.lower() in ("other", "__abstain__", "none", "null"):
                continue
            if r in tool_candidates:
                chosen_tool = r
                break
            for cand in tool_candidates:
                if cand.lower() == r.lower():
                    chosen_tool = cand
                    break
            if chosen_tool != "__abstain__":
                break

        if chosen_tool == "__abstain__":
            for cand in tool_candidates:
                if re.search(r'\b' + re.escape(cand) + r'\b', resp, re.IGNORECASE):
                    chosen_tool = cand
                    break

        if "__abstain__" not in candidates:
            candidates["__abstain__"] = "Refuse or no matching tool"
        probs = {k: (0.95 if k == chosen_tool else 0.05 / max(len(candidates) - 1, 1)) for k in candidates}
        return probs, latency


class XLAMEvaluator:
    """
    Evaluator for Salesforce/xLAM-2-1b-fc-r
    Extracts selected tool name from its JSON tool_calls output.
    Maps empty list [] or no tool call to "__abstain__".
    """
    def __init__(self, model_id: str = "Salesforce/xLAM-2-1b-fc-r", device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device
        print(f"Loading {model_id} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.float16,
            ).to(device)
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.float16,
                device_map=device,
            )

    def score(self, query: str, candidates: Dict[str, str]) -> Tuple[Dict[str, float], float]:
        t0 = time.perf_counter()
        tool_candidates = {k: v for k, v in candidates.items() if k != "__abstain__"}
        if not tool_candidates:
            latency = (time.perf_counter() - t0) * 1000
            return {"__abstain__": 1.0}, latency

        tools = [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": parse_candidate_text(name, desc)[:250],
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for name, desc in tool_candidates.items()
        ]

        messages = [{"role": "user", "content": query}]
        try:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=32,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=False,
                )
            gen_tokens = outputs[:, input_len:]
            resp = self.tokenizer.decode(gen_tokens[0], skip_special_tokens=True).strip()
        except Exception:
            prompt = (
                f"Available tools:\n{json.dumps(tools)}\n\n"
                f"User request: {query}\n"
                f"Select tool in JSON format or return [] if none applies:"
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=32,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=False,
                )
            gen_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
            resp = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        latency = (time.perf_counter() - t0) * 1000

        chosen_tool = "__abstain__"
        if resp.strip().startswith("[]") or "[]" in resp:
            chosen_tool = "__abstain__"
        else:
            name_matches = re.findall(r"['\"]name['\"]\s*:\s*['\"]([^'\"]+)['\"]", resp)
            for t_name in name_matches:
                t_name = t_name.strip()
                if t_name in tool_candidates:
                    chosen_tool = t_name
                    break
                for cand in tool_candidates:
                    if cand.lower() == t_name.lower():
                        chosen_tool = cand
                        break
                if chosen_tool != "__abstain__":
                    break

            if chosen_tool == "__abstain__":
                for cand in tool_candidates:
                    if re.search(r'\b' + re.escape(cand) + r'\b', resp, re.IGNORECASE):
                        chosen_tool = cand
                        break

        if "__abstain__" not in candidates:
            candidates["__abstain__"] = "Refuse or no matching tool"
        probs = {k: (0.95 if k == chosen_tool else 0.05 / max(len(candidates) - 1, 1)) for k in candidates}
        return probs, latency


class HammerEvaluator:
    """
    Evaluator for MadeAgents/Hammer2.1-1.5b
    Extracts tool name from output.
    Maps no-call output or refusal to "__abstain__".
    """
    def __init__(self, model_id: str = "MadeAgents/Hammer2.1-1.5b", device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device
        print(f"Loading {model_id} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.float16,
            ).to(device)
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.float16,
                device_map=device,
            )

    def score(self, query: str, candidates: Dict[str, str]) -> Tuple[Dict[str, float], float]:
        t0 = time.perf_counter()
        tool_candidates = {k: v for k, v in candidates.items() if k != "__abstain__"}
        if not tool_candidates:
            latency = (time.perf_counter() - t0) * 1000
            return {"__abstain__": 1.0}, latency

        tools = [
            {"name": k, "description": parse_candidate_text(k, desc)[:250], "parameters": {"type": "object", "properties": {}}}
            for k, desc in tool_candidates.items()
        ]

        prompt = (
            "[BEGIN OF TASK INSTRUCTION]\n"
            "You are a tool calling assistant. In order to complete the user's request, you need to select one or more appropriate tools from the following tools.\n"
            "If none of the tools can be used, point it out and refuse to answer.\n"
            "[END OF TASK INSTRUCTION]\n\n"
            f"[BEGIN OF AVAILABLE TOOLS]\n{json.dumps(tools)}\n[END OF AVAILABLE TOOLS]\n\n"
            "[BEGIN OF FORMAT INSTRUCTION]\n"
            "The output MUST strictly adhere to the following JSON format:\n"
            '{"name": "function_name", "arguments": {}}\n'
            "[END OF FORMAT INSTRUCTION]\n\n"
            f"[BEGIN OF QUERY]\n{query}\n[END OF QUERY]\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=32,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
            )
        gen_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        resp = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        latency = (time.perf_counter() - t0) * 1000

        chosen_tool = "__abstain__"
        name_matches = re.findall(r"['\"]name['\"]\s*:\s*['\"]([^'\"]+)['\"]", resp)
        for t_name in name_matches:
            t_name = t_name.strip()
            if t_name in tool_candidates:
                chosen_tool = t_name
                break
            for cand in tool_candidates:
                if cand.lower() == t_name.lower():
                    chosen_tool = cand
                    break
            if chosen_tool != "__abstain__":
                break

        if chosen_tool == "__abstain__":
            for cand in tool_candidates:
                if re.search(r'\b' + re.escape(cand) + r'\b', resp, re.IGNORECASE):
                    chosen_tool = cand
                    break

        if "__abstain__" not in candidates:
            candidates["__abstain__"] = "Refuse or no matching tool"
        probs = {k: (0.95 if k == chosen_tool else 0.05 / max(len(candidates) - 1, 1)) for k in candidates}
        return probs, latency


class KevRemoteEndpointEvaluator:
    """
    Evaluator for jaredpalmer/kev-27b via HTTP endpoint.
    Sends (state, question, options) payload to a /v1/systemone or HF Inference Endpoint.
    """
    def __init__(self, endpoint_url: str, timeout: float = 30.0):
        self.endpoint_url = endpoint_url.rstrip("/")
        self.timeout = timeout
        import httpx
        self.client = httpx.Client(timeout=timeout)

    def score(self, state: str, candidates: Dict[str, str]) -> Tuple[Dict[str, float], float]:
        t0 = time.perf_counter()
        url = self.endpoint_url if self.endpoint_url.endswith("/v1/systemone") else f"{self.endpoint_url}/v1/systemone"
        body = {
            "model": "jaredpalmer/kev-27b",
            "state": state,
            "questions": {
                "tool_route": {
                    "type": "choice",
                    "instructions": "Which single tool should be routed?",
                    "criteria": candidates,
                }
            },
        }
        resp = self.client.post(url, json=body)
        resp.raise_for_status()
        data = resp.json()
        latency = (time.perf_counter() - t0) * 1000

        answers = data.get("answers") or {}
        answer = answers.get("tool_route") or answers.get("next_action") or {}
        raw_probs = answer.get("probabilities")
        if isinstance(raw_probs, dict) and raw_probs:
            probs = {cid: float(raw_probs.get(cid, 0.0)) for cid in candidates}
            tot = sum(probs.values())
            if tot > 0:
                return {k: v / tot for k, v in probs.items()}, latency
        choice = answer.get("choice")
        if choice in candidates:
            return {cid: (0.95 if cid == choice else 0.05 / max(len(candidates) - 1, 1)) for cid in candidates}, latency
        return {"__abstain__": 1.0}, latency


# ---------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------

def load_external_dataset() -> List[Dict[str, Any]]:
    path = REPO_ROOT / "releases" / "soro-universal-tool-harness" / "val_external_heldout.jsonl"
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.strip():
                r = json.loads(line)
                r["_row_id"] = idx
                records.append(r)
    return records


def load_zeroshot_dataset(collision_only: bool = False) -> List[Dict[str, Any]]:
    path = REPO_ROOT / "releases" / "soro-universal-tool-harness" / "val_zeroshot.jsonl"
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.strip():
                r = json.loads(line)
                r["_row_id"] = idx
                if collision_only and not r.get("has_collision_distractor", False):
                    continue
                records.append(r)
    return records


def load_bfcl_slice() -> List[Dict[str, Any]]:
    path = REPO_ROOT / "veyra" / "data" / "bfcl" / "soro" / "bfcl_routing.jsonl"
    all_rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            name = r.get("bfcl_file") or ""
            if name not in BFCL_FILES or r.get("trivial_single_tool"):
                continue
            all_rows.append(r)

    rng = random.Random(42)
    by_file = defaultdict(list)
    for r in all_rows:
        by_file[r.get("bfcl_file")].append(r)

    sampled = []
    counts = {"BFCL_v3_multiple.json": 34, "BFCL_v3_live_multiple.json": 33, "BFCL_v3_irrelevance.json": 33}
    for fname, target_n in counts.items():
        sub = by_file[fname]
        rng.shuffle(sub)
        sampled.extend(sub[:target_n])

    for idx, r in enumerate(sampled):
        r["_row_id"] = idx
    return sampled


# ---------------------------------------------------------
# Prediction Evaluation Pipeline
# ---------------------------------------------------------

def evaluate_system_on_dataset(
    system_name: str,
    system_obj: Any,
    records: List[Dict[str, Any]],
    split_name: str,
    output_path: Path,
    overwrite: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_preds = []
    if output_path.exists() and not overwrite:
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        existing_preds.append(json.loads(line))
                    except Exception:
                        pass

    start_idx = len(existing_preds)
    n_correct = sum(1 for p in existing_preds if p.get("is_correct", False))
    predictions = list(existing_preds)

    if start_idx >= len(records):
        print(f"'{system_name}' on '{split_name}' already fully evaluated ({start_idx}/{len(records)} rows). Skipping.")
        return

    print(f"\nEvaluating '{system_name}' on '{split_name}' (resuming from {start_idx}/{len(records)} rows)...")

    t0 = time.time()
    latencies: List[float] = [p["latency_ms"] for p in existing_preds if "latency_ms" in p]
    mode = "a" if start_idx > 0 else "w"

    with open(output_path, mode, encoding="utf-8") as out_f:
        for idx in range(start_idx, len(records)):
            r = records[idx]
            q_data = r.get("questions", {}).get("tool_route", {})
            state = r.get("state", "")
            query = extract_query(state)
            candidates = q_data.get("criteria", {})
            label = q_data.get("label", "__abstain__")
            package = r.get("package", "default_pkg")
            bfcl_file = r.get("bfcl_file", "")
            has_collision = bool(r.get("has_collision_distractor", False))

            # Score candidates
            if isinstance(system_obj, (SoroKevEvaluator, KevRemoteEndpointEvaluator)):
                res = system_obj.score(state, candidates)
            else:
                res = system_obj.score(query, candidates)

            if isinstance(res, tuple):
                probs, lat_ms = res
            else:
                probs, lat_ms = res, 0.0

            latencies.append(lat_ms)

            # Normalize probs
            tot_p = sum(probs.values())
            if tot_p > 0:
                probs = {k: v / tot_p for k, v in probs.items()}

            top_tool = max(probs.items(), key=lambda x: x[1])[0]

            # For BFCL: gold can be in gold_functions
            golds = r.get("gold_functions") or []
            if golds and bfcl_file != "BFCL_v3_irrelevance.json" and label != "__abstain__":
                is_correct = (top_tool in golds)
            else:
                is_correct = (top_tool == label)

            if is_correct:
                n_correct += 1

            is_abstain_row = (label == "__abstain__" or bfcl_file == "BFCL_v3_irrelevance.json")
            is_shared_tool = (label in SHARED_TOOLS)

            row_pred = {
                "id": r.get("_row_id", idx),
                "package": package,
                "label": label,
                "top_tool": top_tool,
                "is_correct": is_correct,
                "is_abstain_row": is_abstain_row,
                "is_shared_tool": is_shared_tool,
                "has_collision": has_collision,
                "bfcl_file": bfcl_file,
                "latency_ms": round(lat_ms, 2),
                "probabilities": {k: round(v, 6) for k, v in probs.items()},
            }
            predictions.append(row_pred)
            out_f.write(json.dumps(row_pred) + "\n")
            out_f.flush()

            if (idx + 1) % 25 == 0 or (idx + 1) == len(records):
                elapsed = time.time() - t0
                rate = (idx + 1) / max(elapsed, 0.001)
                med_lat = (sorted(latencies)[len(latencies) // 2]) if latencies else 0.0
                print(
                    f"  [{idx+1}/{len(records)}] Correct: {n_correct}/{idx+1} ({n_correct/(idx+1)*100:.2f}%) "
                    f"| Median Latency: {med_lat:.1f}ms | Rate: {rate:.1f} rows/s",
                    flush=True,
                )

    med_lat = (sorted(latencies)[len(latencies) // 2]) if latencies else 0.0
    print(f"Saved {len(predictions)} predictions to: {output_path} (Median latency: {med_lat:.1f}ms)", flush=True)


def load_system_instance(sys_name: str, device: str, endpoint_url: Optional[str] = None) -> Any:
    """Instantiate a single system evaluator."""
    if sys_name == "bm25":
        return BM25Scorer()
    elif sys_name == "dense_cosine":
        return DenseCosineScorer(device=device)
    elif sys_name == "cross_encoder":
        return CrossEncoderScorer(device=device)
    elif sys_name == "option_logits":
        return UntrainedOptionLogitsScorer(device=device)
    elif sys_name == "soro_0.8b":
        ckpt = str(REPO_ROOT / "releases" / "soro-kev-0.8b")
        return SoroKevEvaluator(ckpt, device=device)
    elif sys_name == "soro_4b":
        ckpt = str(REPO_ROOT / "releases" / "soro-kev-4b")
        return SoroKevEvaluator(ckpt, device=device)
    elif sys_name == "arch_router":
        return ArchRouterEvaluator(device=device)
    elif sys_name == "xlam_1b":
        return XLAMEvaluator(device=device)
    elif sys_name == "hammer_1.5b":
        return HammerEvaluator(device=device)
    elif sys_name == "kev_27b":
        if not endpoint_url:
            raise ValueError("Must provide --endpoint-url or set KEV_ENDPOINT to evaluate kev_27b")
        return KevRemoteEndpointEvaluator(endpoint_url)
    else:
        raise ValueError(f"Unknown system name: {sys_name}")


def main():
    parser = argparse.ArgumentParser(description="Generate per-row prediction files for all systems")
    parser.add_argument(
        "--split",
        default="all",
        help="Split name or comma-separated list of splits (external_heldout, zeroshot, zeroshot_collision, bfcl, all)",
    )
    parser.add_argument(
        "--system",
        default="all",
        help="System name or comma-separated list of systems (e.g. arch_router,xlam_1b,hammer_1.5b)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--endpoint-url", default=os.environ.get("KEV_ENDPOINT", None), help="URL for remote Kev-27B /v1/systemone endpoint")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing prediction files instead of resuming")
    args = parser.parse_args()

    pred_dir = REPO_ROOT / "benchmarks" / "predictions"

    # Datasets
    splits_to_run = []
    split_tokens = [s.strip() for s in args.split.split(",") if s.strip()]
    if "all" in split_tokens:
        splits_to_run.append(("external_heldout", load_external_dataset()))
        splits_to_run.append(("zeroshot", load_zeroshot_dataset(collision_only=False)))
        splits_to_run.append(("zeroshot_collision", load_zeroshot_dataset(collision_only=True)))
        splits_to_run.append(("bfcl", load_bfcl_slice()))
    else:
        if "external_heldout" in split_tokens:
            splits_to_run.append(("external_heldout", load_external_dataset()))
        if "zeroshot" in split_tokens:
            splits_to_run.append(("zeroshot", load_zeroshot_dataset(collision_only=False)))
        if "zeroshot_collision" in split_tokens:
            splits_to_run.append(("zeroshot_collision", load_zeroshot_dataset(collision_only=True)))
        if "bfcl" in split_tokens:
            splits_to_run.append(("bfcl", load_bfcl_slice()))

    # Systems
    req_tokens = [s.strip() for s in args.system.split(",") if s.strip()]
    req_systems = []
    for tok in req_tokens:
        if tok == "all":
            req_systems.extend([
                "bm25", "dense_cosine", "cross_encoder", "option_logits",
                "soro_0.8b", "soro_4b", "arch_router", "xlam_1b", "hammer_1.5b"
            ])
        elif tok == "all_baselines":
            req_systems.extend(["bm25", "dense_cosine", "cross_encoder", "option_logits"])
        elif tok == "all_dedicated":
            req_systems.extend(["arch_router", "xlam_1b", "hammer_1.5b"])
        else:
            req_systems.append(tok)
    req_systems = list(dict.fromkeys(req_systems))

    # Run each system sequentially to maintain maximum GPU VRAM headroom
    for sys_name in req_systems:
        print(f"\n==================================================")
        print(f"Loading and Evaluating System: {sys_name} on {args.device}")
        print(f"==================================================")
        sys_obj = load_system_instance(sys_name, device=args.device, endpoint_url=args.endpoint_url)

        for split_name, dataset in splits_to_run:
            out_file = pred_dir / split_name / f"{sys_name}.jsonl"
            evaluate_system_on_dataset(sys_name, sys_obj, dataset, split_name, out_file, overwrite=args.overwrite)

        del sys_obj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
