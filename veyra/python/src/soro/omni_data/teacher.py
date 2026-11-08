"""Soft labels from an open teacher (text, image and audio).

The teacher sees the request and a lettered menu and we read its next-token
logits over the option letters (no generation). Large models favour certain
positions, so every row is scored under ``k`` rotations of the menu and the
probabilities are averaged per option. A temperature is then fitted on rows
with gold labels so the soft targets are calibrated.

Text teachers load with ``AutoModelForCausalLM``. Multimodal teachers
(Qwen3-Omni, Gemma 4) load with ``AutoModelForImageTextToText`` / ``AutoProcessor``
and receive the image and/or audio inside the chat message. The multimodal path
follows the processors' chat-template API and must be smoke-tested on the GPU
box for each teacher before a full labelling run.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import torch

from soro.omni_data.schema import ABSTAIN, NEED_INPUT, OmniRow

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SYSTEM = ("You route user requests. Read the request (and any attached image or audio) and pick the single "
          "best option. Answer with one letter only.")


def _option_line(key: str, text: str) -> str:
    if key == ABSTAIN:
        return "none of these: no listed option fits the request"
    if key == NEED_INPUT:
        return "need input: the request depends on an image, file or recording that is missing or unreadable"
    return f"{key}: {text[:400]}"


class TeacherScorer:
    def __init__(self, model_id: str, device: str = "cuda", load_in_4bit: bool = False, multimodal: bool = False,
                 rotations: int = 4, dtype: torch.dtype | None = None):
        from transformers import AutoTokenizer

        self.model_id, self.device, self.rotations, self.multimodal = model_id, device, rotations, multimodal
        # T4 (sm_75) has no bf16: use fp16 there, bf16 on Ampere+, fp32 on CPU.
        if dtype is None:
            if device.startswith("cuda"):
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                dtype = torch.float32
        self.dtype = dtype
        kw: dict[str, Any] = {"dtype": dtype}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=dtype if dtype != torch.float32 else torch.float16)
            kw["device_map"] = "auto"
        if multimodal:
            from transformers import AutoModelForImageTextToText, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(model_id)
            self.tok = self.processor.tokenizer
            self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kw)
        else:
            from transformers import AutoModelForCausalLM

            self.processor = None
            self.tok = AutoTokenizer.from_pretrained(model_id)
            self.tok.padding_side = "left"  # last position is the answer slot for every row in a batch
            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        if not load_in_4bit:
            self.model.to(device)
        self.model.eval()
        self.letter_ids = [self.tok.encode(" " + c, add_special_tokens=False)[-1] for c in LETTERS]
        self.temperature = 1.0

    # ------------------------------------------------------------------ prompts
    def _prompt_text(self, row: OmniRow, keys: list[str]) -> str:
        menu = "\n".join(f"{LETTERS[i]}. {_option_line(k, row.option_map[k])}" for i, k in enumerate(keys))
        req = row.text or (row.transcript or "")
        ctx = f"Context:\n{row.context}\n\n" if row.context else ""
        att = "".join(["(an image is attached)\n" if row.image else "", "(an audio recording is attached)\n" if row.audio else ""])
        return f"{SYSTEM}\n\n{ctx}Request: {req}\n{att}\nOptions:\n{menu}\n\nAnswer:"

    @torch.no_grad()
    def _letter_logits(self, row: OmniRow, keys: list[str]) -> torch.Tensor:
        prompt = self._prompt_text(row, keys)
        if self.multimodal:
            content: list[dict[str, Any]] = []
            if row.image:
                content.append({"type": "image", "image": row.image})
            if row.audio:
                content.append({"type": "audio", "audio": row.audio})
            content.append({"type": "text", "text": prompt})
            inputs = self.processor.apply_chat_template(
                [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            ).to(self.model.device)
        else:
            inputs = self.tok(prompt, return_tensors="pt", truncation=True, max_length=6144).to(self.model.device)
        logits = self.model(**inputs).logits[0, -1].float()
        return logits[self.letter_ids[: len(keys)]]

    @torch.no_grad()
    def _batched_text_logits(self, row: OmniRow, orders: list[list[str]]) -> list[torch.Tensor]:
        """All rotations of one row in a single padded forward (text teachers)."""
        prompts = [self._prompt_text(row, o) for o in orders]
        enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(self.model.device)
        last = self.model(**enc).logits[:, -1].float()
        return [last[i, self.letter_ids[: len(o)]] for i, o in enumerate(orders)]

    def probs(self, row: OmniRow) -> dict[str, float]:
        """Position-debiased probabilities: average over rotations of the menu."""
        keys = [o.key for o in row.options]
        total = {k: 0.0 for k in keys}
        k = max(1, min(self.rotations, len(keys)))
        orders = [keys[s:] + keys[:s] for s in range(k)]
        logits = (self._batched_text_logits(row, orders) if not self.multimodal
                  else [self._letter_logits(row, o) for o in orders])
        for order, lg in zip(orders, logits):
            p = torch.softmax(lg / self.temperature, -1).cpu().tolist()
            for key, v in zip(order, p):
                total[key] += v / k
        return total

    def label(self, rows: list[OmniRow], gold_agree_threshold: float = 0.5) -> list[OmniRow]:
        out = []
        for r in rows:
            p = self.probs(r)
            top = max(p, key=p.get)
            out.append(replace(r, teacher={"model": self.model_id, "probs": {k: round(v, 5) for k, v in p.items()},
                                           "top": top, "agrees": top == r.label,
                                           "gold_prob": round(p.get(r.label, 0.0), 5),
                                           "temperature": self.temperature, "rotations": self.rotations}))
        return out


def fit_temperature(rows: list[OmniRow], grid: tuple[float, ...] = tuple(0.25 * 1.1 ** i for i in range(40))) -> float:
    """Temperature minimizing gold NLL over rows that carry teacher probs (applied to log-probs)."""
    data = [(r.teacher["probs"], r.label) for r in rows if r.teacher and r.label in r.teacher.get("probs", {})]
    if not data:
        return 1.0
    best_t, best_nll = 1.0, float("inf")
    for t in grid:
        nll = 0.0
        for probs, gold in data:
            logs = {k: math.log(max(v, 1e-9)) / t for k, v in probs.items()}
            m = max(logs.values())
            z = m + math.log(sum(math.exp(v - m) for v in logs.values()))
            nll -= logs[gold] - z
        if nll < best_nll:
            best_t, best_nll = t, nll
    return best_t


def soft_target(row: OmniRow, lam: float = 0.5, temperature: float = 1.0) -> dict[str, float] | None:
    """lam * onehot(gold) + (1 - lam) * tempered teacher. None if the row has no teacher."""
    if not row.teacher:
        return None
    probs = row.teacher["probs"]
    logs = {k: math.log(max(v, 1e-9)) / temperature for k, v in probs.items()}
    m = max(logs.values())
    z = sum(math.exp(v - m) for v in logs.values())
    tp = {k: math.exp(v - m) / z for k, v in logs.items()}
    return {k: lam * (1.0 if k == row.label else 0.0) + (1 - lam) * tp.get(k, 0.0) for k in row.option_map}
