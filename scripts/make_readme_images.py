"""Render docs/hero.png and docs/social_preview.png (1280x640) from docs/hero_decisions.json.

docs/hero_decisions.json holds real decisions from nisaralll/soro-kev-0.8b on CPU (default q_hat).
    python scripts/make_readme_images.py
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
SANS_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BG, FG, DIM = (13, 17, 23), (230, 237, 243), (139, 148, 158)
COL = {"EXECUTE": (63, 185, 80), "ASK": (210, 153, 34), "ABSTAIN": (248, 81, 73)}


def font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:  # pragma: no cover
        return ImageFont.load_default()


def decisions():
    return json.loads((ROOT / "docs/hero_decisions.json").read_text())


def hero(path):
    data = decisions()
    W, H = 1640, 600
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    f, fb, fs = font(MONO, 24), font(BOLD, 24), font(MONO, 20)
    d.rounded_rectangle([20, 20, W - 20, H - 20], 16, outline=(48, 54, 61), width=2)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([44 + i * 30, 40, 62 + i * 30, 58], fill=c)
    d.text((150, 36), f"router = Router.from_pretrained(\"{data['model']}\")   # {data['device']}", font=fs, fill=DIM)
    y = 100
    for row in data["decisions"]:
        d.text((50, y), ">>> router.route(", font=f, fill=DIM)
        d.text((50 + 17 * 14.5, y), json.dumps(row["q"]), font=f, fill=FG)
        d.text((50 + (17 + len(json.dumps(row["q"]))) * 14.5, y), ", tools)", font=f, fill=DIM)
        y += 40
        act = row["action"]
        d.text((90, y), f"{act:<8}", font=fb, fill=COL[act])
        detail = {"EXECUTE": f"{row['tool']}   p={row['conf']:.3f}",
                  "ASK": f"which one? {row['cands']}   top p={row['conf']:.2f}",
                  "ABSTAIN": f"no tool fits   p(__abstain__)={row['conf']:.3f}"}[act]
        d.text((90 + 10 * 14.5, y), detail, font=f, fill=FG)
        y += 64
    d.text((50, H - 70), "tools: get_weather, send_email, create_calendar_event, search_flights   "
           f"(q_hat={data['q_hat']}, real 0.8B output)", font=fs, fill=DIM)
    im.save(path, optimize=True)


def social(path):
    W, H = 1280, 640
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((80, 90), "Soro-Boro", font=font(SANS_B, 96), fill=FG)
    d.text((84, 215), "Calibrated tool routing for AI agents", font=font(SANS, 44), fill=(201, 209, 217))
    d.text((84, 285), "One forward pass  ·  no invented tool names  ·  OpenAI / MCP / LangChain",
           font=font(SANS, 28), fill=DIM)
    x = 84
    for act in ("EXECUTE", "ASK", "ABSTAIN"):
        fb = font(BOLD, 34)
        w = d.textlength(act, font=fb) + 48
        d.rounded_rectangle([x, 370, x + w, 430], 14, outline=COL[act], width=3)
        d.text((x + 24, 378), act, font=fb, fill=COL[act])
        x += w + 28
    d.text((84, 500), "pip install soro-boro", font=font(MONO, 34), fill=FG)
    d.text((84, 560), "github.com/nisaral/Soro-Boro  ·  Soro-Kev 0.8B / 4B open weights", font=font(SANS, 24), fill=DIM)
    im.save(path, optimize=True)


if __name__ == "__main__":
    hero(ROOT / "docs/hero.png")
    social(ROOT / "docs/social_preview.png")
    print("wrote docs/hero.png, docs/social_preview.png")
