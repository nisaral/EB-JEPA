"""Shared helper: build a Router from CLI flags (heuristic by default so examples run without weights)."""
from __future__ import annotations

import argparse

from soro_boro import Router


def router_from_args(description: str) -> tuple[Router, argparse.Namespace]:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--model", default=None, help="HF id or local dir, e.g. nisaralll/soro-kev-0.8b (default: heuristic, no weights)")
    p.add_argument("--device", default="auto")
    p.add_argument("--q-hat", type=float, default=None)
    a = p.parse_args()
    if a.model:
        return Router.from_pretrained(a.model, device=a.device, q_hat=a.q_hat), a
    return Router.heuristic(q_hat=a.q_hat), a
