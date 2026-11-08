"""Image rows: UI-element pointing (OS-Atlas) and image-dependent tool routing.

OS-Atlas grounding records are ``{"img_filename", "instruction", "bbox": [l, t, r, b]
in 0..1, "data_type"?}``. Records that share a screenshot become one menu: the
options are elements on that screen (their referring text + bbox) and the request
asks for one of them. Two hard cases are built on purpose:

* ``absent``  the request names an element from a different screen -> __abstain__
* ``twin``    the menu holds several elements with near-identical text, so the
              model must use the image (position, context) to pick the right one

Image-dependent tool routing rows are assembled in ``generate.py`` (teacher
writes the request, ``necessity_filter`` keeps rows where only the image decides).
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from soro.omni_data.schema import ABSTAIN, OmniRow, Option, special_option

_TEMPLATES = ["Click {x}", "Tap {x}", "Open {x}", "Select {x}", "Go to {x}", "Press {x}", "{x}"]


def _clean(s: str) -> str:
    return " ".join(str(s).split())[:200]


def _valid_bbox(b: Any) -> bool:
    try:
        l, t, r, bt = (float(v) for v in b)
    except (TypeError, ValueError):
        return False
    return 0 <= l < r <= 1 and 0 <= t < bt <= 1 and (r - l) * (bt - t) > 1e-5


def os_atlas_rows(records: list[dict[str, Any]], image_root: str, subsource: str, license: str,
                  menu_size: int = 8, absent_rate: float = 0.12, max_per_image: int = 3, seed: int = 0) -> list[OmniRow]:
    rng = random.Random(seed)
    by_img: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        ins = _clean(rec.get("instruction", ""))
        if len(ins) < 2 or not _valid_bbox(rec.get("bbox")):
            continue
        by_img[rec["img_filename"]].append({"text": ins, "bbox": [round(float(v), 4) for v in rec["bbox"]],
                                            "type": rec.get("data_type")})
    images = [k for k, v in by_img.items() if len(v) >= 3]
    rows: list[OmniRow] = []
    for img in images:
        elems = by_img[img]
        # drop exact duplicate texts on one screen unless they are true twins (different boxes)
        uniq: dict[tuple[str, tuple], dict] = {}
        for e in elems:
            uniq[(e["text"].lower(), tuple(e["bbox"]))] = e
        elems = list(uniq.values())
        rng.shuffle(elems)
        for k, target in enumerate(elems[:max_per_image]):
            others = [e for e in elems if e is not target]
            twins = [e for e in others if e["text"].lower() == target["text"].lower()]
            menu = [target] + twins + [e for e in others if e not in twins][: max(0, menu_size - 1 - len(twins))]
            rng.shuffle(menu)
            opts = [Option(key=f"el{i}", text=e["text"] + (f" ({e['type']})" if e.get("type") else ""),
                           kind="ui_element", bbox=e["bbox"]) for i, e in enumerate(menu)]
            opts.append(special_option(ABSTAIN))
            label = f"el{menu.index(target)}"
            text = rng.choice(_TEMPLATES).format(x=target["text"])
            tags = ["ui", f"os_atlas:{subsource}"] + (["twin"] if twins else [])
            rid = f"osa-{subsource}-{Path(img).stem}-{k}"
            if rng.random() < absent_rate and len(images) > 1:
                other_img = rng.choice([i for i in images if i != img])
                foreign = rng.choice(by_img[other_img])["text"]
                if all(foreign.lower() != e["text"].lower() for e in elems):
                    text, label, tags = rng.choice(_TEMPLATES).format(x=foreign), ABSTAIN, tags + ["absent"]
            rows.append(OmniRow(id=rid, task="ui_point", modality="image", options=opts, label=label,
                                source="os_atlas", license=license, group=f"osa-{Path(img).stem}",
                                text=text, image=str(Path(image_root) / img), package=f"os_atlas/{subsource}",
                                tags=tags, meta={"n_elements_on_screen": len(elems)}))
    return rows


def load_os_atlas_json(path: str) -> list[dict[str, Any]]:
    import json

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):  # some files wrap records
        for v in data.values():
            if isinstance(v, list):
                return v
    return data


def missing_image_rows(rows: list[OmniRow], rate: float = 0.05, seed: int = 0) -> list[OmniRow]:
    """Image rows with the image removed: gold becomes NEED_INPUT (the request needs the screen)."""
    from dataclasses import replace

    from soro.omni_data.schema import NEED_INPUT

    rng = random.Random(seed)
    out = []
    for r in rows:
        if r.modality != "image" or rng.random() > rate:
            continue
        opts = [o for o in r.options if o.kind != "ui_element"] or list(r.options)
        if r.task == "ui_point":  # without the screen the elements cannot be shown; keep tool-style menu
            continue
        if not any(o.key == NEED_INPUT for o in opts):
            opts.append(special_option(NEED_INPUT))
        out.append(replace(r, id=f"{r.id}~noimg", modality="text", image=None, options=opts, label=NEED_INPUT,
                           tags=r.tags + ["need_input", "derived"], meta={**r.meta, "parent": r.id}))
    return out


def is_vague(request: str) -> bool:
    """Requests that cannot be routed without the attachment ('handle this', 'what do I do here?')."""
    return bool(re.search(r"\b(this|these|that|here|it)\b", request.lower())) and len(request.split()) <= 12
