"""Source registry and license gate.

Only sources whose license allows training a model released under Apache-2.0 go
into train/selection splits. Non-commercial or unclear sources are eval-only.
Licenses below were read from each source's dataset or model card (Nov 2026);
re-check before a release.
"""

from __future__ import annotations

from dataclasses import dataclass

TRAIN_OK = {"apache-2.0", "mit", "cc-by-4.0", "cc0-1.0", "cc-by-sa-4.0", "own", "bsd-3-clause"}
EVAL_ONLY = {"cc-by-nc-4.0", "cc-by-nc-sa-4.0", "research-only", "unknown"}


@dataclass(frozen=True)
class Source:
    name: str
    license: str
    modality: str
    use: str  # "train" or "eval"
    note: str = ""


REGISTRY: dict[str, Source] = {s.name: s for s in [
    Source("soro_train", "mixed-see-DATA_LICENSES", "text", "train",
           "Existing Soro-Kev rows; upstream licenses listed in DATA_LICENSES.md"),
    Source("soro_hardened", "own", "text", "train", "Collision / near-miss rows derived from soro_train"),
    Source("teacher_generated", "own", "text", "train",
           "Requests written by Apache-2.0 open teachers (Qwen3-Omni, Gemma 4)"),
    Source("tts_kokoro", "own", "audio", "train", "Our text rendered with Kokoro-82M (Apache-2.0)"),
    Source("os_atlas", "apache-2.0", "image", "train",
           "OS-Atlas grounding data; upstream sets (AMEX, Rico, SeeClick, ...) keep their own terms"),
    Source("image_routing_synth", "own", "image", "train", "Image-dependent routing rows, teacher-labelled"),
    Source("audio2tool", "cc-by-nc-4.0", "audio", "eval", "Non-commercial: evaluation only"),
    Source("bfcl", "apache-2.0", "text", "eval", "BFCL v3 diagnostic slice; kept eval-only by policy"),
    Source("bfcl_audio", "unknown", "audio", "eval", "License not stated on the paper page"),
    Source("screenspot_pro", "unknown", "image", "eval", "Check card before any training use"),
    Source("m3_bench", "unknown", "image", "eval", ""),
    Source("human_eval", "own", "any", "eval", "Our own human-written requests and recordings"),
    Source("when2call", "cc-by-4.0", "text", "train", "NVIDIA When2Call benchmark (train SFT/pref and test MCQ/judge)"),
    Source("agentabstain", "cc-by-4.0", "text", "eval", "AgentAbstain paired act vs abstain tasks across 8 abstention triggers"),
    Source("s2_actions", "own", "text", "train", "S2 action-card (ANSWER/ASK/ESCALATE/ABSTAIN) + expected-cost training rows"),
]}

# Upstream license per OS-Atlas sub-source (fill as each is verified).
OS_ATLAS_SUBSOURCES = {
    "fineweb_3m": "apache-2.0",
    "seeclick_web": "unknown",
    "amex": "cc-by-4.0",
    "uibert": "cc-by-4.0",
    "widget_captioning": "cc-by-4.0",
    "ricosca": "cc-by-4.0",
    "aw_mobile": "apache-2.0",
}


def trainable(license: str) -> bool:
    lic = (license or "").lower()
    if lic in TRAIN_OK:
        return True
    if lic.startswith("mixed"):
        return True  # audited separately in DATA_LICENSES.md
    return False


def gate(rows, split_for_eval_only: str = "test"):
    """Yield rows; force eval-only licenses out of train/selection. Returns (kept, moved)."""
    moved = 0
    out = []
    for r in rows:
        if r.split in ("train", "selection") and not trainable(r.license):
            r.split = split_for_eval_only
            moved += 1
        out.append(r)
    return out, moved
