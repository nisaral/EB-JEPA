"""Teacher-written requests: RLCD-style contrastive pairs and image-dependent routing.

RLCD (Yang et al., ICLR 2024) gets cleaner training signal by prompting a model
two contrasting ways. Here, for one menu and one target tool, the teacher writes:

  positive  a request that clearly needs the target tool
  negative  a request that sounds close (same words, same domain) but that the
            target tool cannot do and no other listed tool covers -> __abstain__
  confuser  a request that needs a look-alike tool rather than the target

All generated rows go through teacher scoring afterwards (``teacher.py``); rows
whose scored answer disagrees with the intended label are dropped.

For images, ``necessity_filter`` keeps a row only when the image is what decides
the answer: the teacher is unsure from the text alone but confident with the
image. That is the property no public dataset has.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import replace
from typing import Any, Iterable

from soro.omni_data.schema import ABSTAIN, OmniRow, Option, special_option

KINDS = ("positive", "negative", "confuser")

_INSTRUCT = {
    "positive": ("Write one realistic user request that clearly requires the tool `{target}`. "
                 "Do not name the tool. Vary tone and length."),
    "negative": ("Write one realistic user request that uses similar words and domain as the tool `{target}`, "
                 "but that `{target}` cannot do and that none of the listed tools can do. Do not name any tool."),
    "confuser": ("Write one realistic user request that needs the tool `{other}` and NOT `{target}`, even though the "
                 "two look alike. Do not name either tool."),
}


def build_prompts(rows: Iterable[OmniRow], pool_lookalikes, per_row: int = 1, seed: int = 0) -> list[dict[str, Any]]:
    """Generation prompts for contrastive pairs. ``pool_lookalikes(key) -> list[str]``."""
    rng = random.Random(seed)
    prompts = []
    for r in rows:
        tools = [o for o in r.options if o.kind == "tool"]
        if not tools:
            continue
        for _ in range(per_row):
            target = rng.choice(tools)
            menu = "\n".join(f"- {o.key}: {o.text[:300]}" for o in tools)
            for kind in KINDS:
                other = None
                if kind == "confuser":
                    looks = [k for k in pool_lookalikes(target.key) if k in {o.key for o in tools}]
                    if not looks:
                        continue
                    other = looks[0]
                instr = _INSTRUCT[kind].format(target=target.key, other=other)
                prompts.append({
                    "parent": r.id, "kind": kind, "target": target.key, "other": other,
                    "intended_label": target.key if kind == "positive" else (ABSTAIN if kind == "negative" else other),
                    "prompt": f"Tools available:\n{menu}\n\n{instr}\nReply with the request only, in quotes.",
                })
    return prompts


def parse_request(text: str) -> str | None:
    m = re.search(r'"([^"]{8,400})"', text)
    req = (m.group(1) if m else text).strip().strip('"').strip()
    req = req.split("\n")[0].strip()
    return req if 8 <= len(req) <= 400 else None


def generate_requests(prompts: list[dict[str, Any]], model_id: str, device: str = "cuda", max_new_tokens: int = 80,
                      temperature: float = 0.9, batch_size: int = 16) -> list[dict[str, Any]]:
    """Run an instruct teacher over the prompts (transformers generate; use vLLM for big runs)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if device.startswith("cuda"):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16  # T4: fp16
    else:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    out = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        texts = [tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], tokenize=False,
                                         add_generation_prompt=True) if tok.chat_template else p["prompt"]
                 for p in batch]
        enc = tok(texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                                 top_p=0.95, pad_token_id=tok.pad_token_id)
        for p, g in zip(batch, gen[:, enc["input_ids"].shape[1]:]):
            out.append({**p, "raw": tok.decode(g, skip_special_tokens=True)})
    return out


def to_rows(generated: list[dict[str, Any]], parents: dict[str, OmniRow], model_id: str) -> list[OmniRow]:
    rows = []
    for g in generated:
        req = parse_request(g.get("raw", ""))
        parent = parents.get(g["parent"])
        if not req or parent is None:
            continue
        opts = list(parent.options)
        if g["intended_label"] == ABSTAIN:
            opts = [o for o in opts if o.key != g["target"]]
            if not any(o.key == ABSTAIN for o in opts):
                opts.append(special_option(ABSTAIN))
        tags = ["generated", f"gen:{g['kind']}"] + (["near_miss"] if g["kind"] == "negative" else []) + \
               (["collision"] if g["kind"] == "confuser" else [])
        rows.append(replace(parent, id=f"{parent.id}~gen-{g['kind']}-{len(rows)}", text=req, context="",
                            options=opts, label=g["intended_label"], source="teacher_generated", license="own",
                            group=f"gen-{parent.group}", tags=tags + ["needs_teacher"], teacher=None,
                            meta={"parent": parent.id, "generator": model_id, "kind": g["kind"]}))
    return rows


def agree_filter(rows: list[OmniRow], min_prob: float = 0.5) -> tuple[list[OmniRow], int]:
    """Keep generated rows whose teacher-scored answer matches the intended label."""
    kept, dropped = [], 0
    for r in rows:
        if "needs_teacher" not in r.tags:
            kept.append(r)
            continue
        p = (r.teacher or {}).get("probs", {})
        top = max(p, key=p.get) if p else None
        if top == r.label and p.get(r.label, 0.0) >= min_prob:
            kept.append(replace(r, tags=[t for t in r.tags if t != "needs_teacher"] + ["teacher_verified"]))
        else:
            dropped += 1
    return kept, dropped


# ---------------------------------------------------------------------- image rows
IMAGE_REQUEST_PROMPT = (
    "Look at the attached image. The assistant has these tools:\n{menu}\n\n"
    "Write a short, natural request a user would send together with this image, such that the right tool can "
    "only be chosen by looking at the image (for example 'can you deal with this?' or 'what should I do here?'). "
    "Do not describe the image in the request. Reply with the request only, in quotes."
)


def necessity_filter(text_only: dict[str, float], with_image: dict[str, float], label: str,
                     max_text_conf: float = 0.5, min_image_conf: float = 0.8) -> bool:
    """True when the image decides the answer: unsure from text, confident and correct with the image."""
    return max(text_only.values()) <= max_text_conf and with_image.get(label, 0.0) >= min_image_conf \
        and max(with_image, key=with_image.get) == label


def image_routing_row(image: str, request: str, catalog: list[Option], label: str, source_id: str,
                      image_license: str) -> OmniRow:
    opts = list(catalog) + [special_option(ABSTAIN)]
    return OmniRow(id=f"imgroute-{source_id}", task="tool_route", modality="image", options=opts, label=label,
                   source="image_routing_synth", license=image_license, group=f"img-{source_id}", text=request,
                   image=image, tags=["image_dependent", "needs_teacher"])


def dump_jsonl(items: list[dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
