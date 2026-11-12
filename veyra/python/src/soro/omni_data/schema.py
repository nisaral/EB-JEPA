"""One row format for every modality.

A row is a closed-set decision: given inputs (text and/or image and/or audio)
and a menu of options, the gold answer is one option, ``__abstain__`` (nothing
fits) or ``__need_input__`` (the decision needs a missing or unreadable input).

Options can be tools (text descriptions) or UI elements (text plus a bounding
box on the input image). Every row carries its source, license and a ``group``
key; rows that share a group (same base request) never cross a split.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

ABSTAIN = "__abstain__"
NEED_INPUT = "__need_input__"
ANSWER = "__answer__"
ASK = "__ask__"
ESCALATE = "__escalate__"
ACTION_CARDS = (ANSWER, ASK, ABSTAIN)
SPECIAL = (ABSTAIN, NEED_INPUT, ANSWER, ASK, ESCALATE)
ABSTAIN_TEXT = "None of the listed tools or skills fit the user request."
NEED_INPUT_TEXT = "The request depends on an image, file or recording that is missing or unreadable."
ANSWER_TEXT = "Answer directly without calling any tool (request is conversational or already answerable)."
ASK_TEXT = "Ask the user a brief clarifying question before acting (missing argument or ambiguous tool)."
ESCALATE_TEXT = "Escalate to a stronger supervisor or human (high-risk irreversible mutation, policy boundary, or critical failure)."
SPECIAL_TEXT = {
    ABSTAIN: ABSTAIN_TEXT,
    NEED_INPUT: NEED_INPUT_TEXT,
    ANSWER: ANSWER_TEXT,
    ASK: ASK_TEXT,
    ESCALATE: ESCALATE_TEXT,
}

MODALITIES = ("text", "image", "audio", "image+audio")
TASKS = ("tool_route", "ui_point", "typed_decision")
SPLITS = ("train", "selection", "test", "human_eval")


@dataclass
class Option:
    key: str
    text: str
    kind: str = "tool"  # tool | ui_element | special
    bbox: list[float] | None = None  # [left, top, right, bottom] in 0..1 (ui_element)

    def __post_init__(self) -> None:
        if self.bbox is not None:
            if len(self.bbox) != 4 or not all(0.0 <= v <= 1.0 for v in self.bbox):
                raise ValueError(f"bbox must be 4 values in [0, 1]: {self.bbox}")
            if self.bbox[0] > self.bbox[2] or self.bbox[1] > self.bbox[3]:
                raise ValueError(f"bbox must be [left, top, right, bottom]: {self.bbox}")


@dataclass
class OmniRow:
    id: str
    task: str
    modality: str
    options: list[Option]
    label: str
    source: str
    license: str
    group: str
    text: str = ""  # the user's request as text (may be empty for audio-only rows)
    context: str = ""  # prior turns or tool observations
    image: str | None = None  # path or URL
    audio: str | None = None  # path to a wav
    transcript: str | None = None  # exact words spoken (audio rows)
    ocr: str | None = None  # text read from the image, when known
    package: str | None = None  # tool package / app, for package-disjoint splits
    split: str = "train"
    tags: list[str] = field(default_factory=list)  # collision, near_miss, hard_negative, streaming, tts, ...
    teacher: dict[str, Any] | None = None  # {"model":..., "probs": {...}, "agrees": bool}
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ checks
    def validate(self) -> list[str]:
        errs: list[str] = []
        keys = [o.key for o in self.options]
        if self.task not in TASKS:
            errs.append(f"task {self.task!r} not in {TASKS}")
        if self.modality not in MODALITIES:
            errs.append(f"modality {self.modality!r} not in {MODALITIES}")
        if self.split not in SPLITS:
            errs.append(f"split {self.split!r} not in {SPLITS}")
        if len(keys) != len(set(keys)):
            errs.append("duplicate option keys")
        if len([k for k in keys if k not in SPECIAL]) < 1:
            errs.append("menu has no real options")
        if self.label not in keys:
            errs.append(f"label {self.label!r} not in options")
        if "image" in self.modality and not self.image:
            errs.append("image modality without an image")
        if "audio" in self.modality and not self.audio:
            errs.append("audio modality without audio")
        if self.modality == "text" and not self.text.strip():
            errs.append("text row with empty text")
        if self.task == "ui_point" and any(o.kind == "ui_element" and o.bbox is None for o in self.options):
            errs.append("ui_element option without bbox")
        if not self.license:
            errs.append("missing license")
        if self.teacher and not set(self.teacher.get("probs", {})) <= set(keys):
            errs.append("teacher probs mention unknown options")
        return errs

    @property
    def option_map(self) -> dict[str, str]:
        return {o.key: o.text for o in self.options}

    def content_hash(self) -> str:
        """Hash of what the model sees: inputs + sorted menu + label."""
        payload = {
            "text": " ".join(self.text.lower().split()),
            "context": " ".join(self.context.lower().split()),
            "image": self.image, "audio": self.audio,
            "menu": sorted(o.key for o in self.options), "label": self.label,
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------ export
    def state(self) -> str:
        """Text state in the Soro-Kev training layout."""
        parts = []
        if self.context:
            parts.append(self.context.strip())
        if self.text:
            parts.append(f"User Request: {self.text.strip()}")
        elif self.transcript is not None:
            parts.append("User Request: [spoken request]")
        if self.image:
            parts.append("[image attached]")
        return "\n".join(parts)

    def to_upp_record(self) -> dict[str, Any]:
        """The record format soro.routing.upp / train_soro_kev consume (text path)."""
        return {
            "id": self.id, "package": self.package or self.source, "state": self.state(),
            "questions": {"tool_route": {"type": "choice", "criteria": self.option_map, "label": self.label}},
            "modality": self.modality, "image": self.image, "audio": self.audio, "tags": self.tags,
        }

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "OmniRow":
        d = dict(d)
        d["options"] = [o if isinstance(o, Option) else Option(**o) for o in d["options"]]
        return cls(**d)


def special_option(key: str, text: str | None = None) -> Option:
    if key not in SPECIAL:
        raise ValueError(f"Unknown special option {key!r}")
    return Option(key=key, text=text if text is not None else SPECIAL_TEXT[key], kind="special")


def read_rows(path: str | Path) -> Iterator[OmniRow]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield OmniRow.from_json(json.loads(line))


def write_rows(rows: Iterable[OmniRow], path: str | Path) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n")
            n += 1
    return n


def from_soro_record(rec: dict[str, Any], source: str, license: str) -> OmniRow:
    """Convert an existing Soro-Kev text record into an OmniRow."""
    q = rec["questions"]["tool_route"]
    state = rec.get("state", "")
    text, context = state, ""
    if "User Request:" in state:
        head, _, tail = state.partition("User Request:")
        context, text = head.strip(), tail.strip()
    opts = [special_option(k) if k == ABSTAIN else Option(key=k, text=str(v)) for k, v in q["criteria"].items()]
    return OmniRow(
        id=str(rec.get("id")), task="tool_route", modality="text", options=opts, label=q["label"],
        source=source, license=license, group=_group_key(text), text=text, context=context,
        package=rec.get("package"),
        tags=(["collision"] if rec.get("has_collision_distractor") else []) + [f"format:{rec.get('format')}"],
        meta={k: rec.get(k) for k in ("provenance", "format", "delexicalized") if k in rec},
    )


def _group_key(text: str) -> str:
    norm = " ".join(text.lower().split())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]
