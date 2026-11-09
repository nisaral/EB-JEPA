"""Exactness and plumbing check for spoken / screenshot requests on an omni backbone.

Checks, for a block-layout gist pointer (no training needed):
  * per-layer-embedding backbones (Gemma 4 E2B/E4B) run from embeddings + clean ids
  * a speech request and a screenshot request enter as the backbone's own soft tokens
  * order equivariance holds with a speech request
  * per-tool caches composed in reverse order match a full pass with a speech request
  * two-level scoring at full budget matches single-pass scoring
  * one training step on mixed text + speech rows

  python veyra/training/media_check.py --model google/gemma-4-E2B --four-bit      (Kaggle T4)
  python veyra/training/media_check.py --tiny                                     (CPU/laptop: random tiny Gemma 4)
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "veyra" / "python" / "src"))

from soro.routing.upp import UPPConfig, UPPModel  # noqa: E402

CRIT = {"get_weather": "Get the weather forecast for a city and date",
        "send_email": "Send an email to a contact with a subject and body",
        "set_timer": "Start a countdown timer for a number of minutes",
        "open_settings": "Open the device settings screen",
        "__abstain__": "None of the tools fit the request"}


def tiny_gemma4(out: Path) -> str:
    from transformers import AutoConfig, AutoProcessor, Gemma4Model

    if (out / "model.safetensors").exists():
        return str(out)
    cfg = AutoConfig.from_pretrained("google/gemma-4-E2B")
    t = cfg.text_config
    if getattr(t, "per_layer_config", None):
        t.per_layer_config = {}
    t.hidden_size, t.intermediate_size, t.num_hidden_layers = 64, 128, 5
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 2, 1, 32
    if hasattr(t, "global_head_dim"):
        t.global_head_dim = 32
    t.hidden_size_per_layer_input = 16
    t.layer_types = ["sliding_attention"] * 4 + ["full_attention"]
    if hasattr(t, "num_kv_shared_layers"):
        t.num_kv_shared_layers = 0
    for c in (cfg.audio_config, cfg.vision_config):
        for k, val in (("hidden_size", 64), ("intermediate_size", 128), ("num_hidden_layers", 1),
                       ("num_attention_heads", 2), ("num_key_value_heads", 1), ("head_dim", 32)):
            if hasattr(c, k):
                setattr(c, k, val)
    torch.manual_seed(0)
    Gemma4Model(cfg).save_pretrained(out)
    AutoProcessor.from_pretrained("google/gemma-4-E2B").save_pretrained(out)
    return str(out)


def request_state(m: UPPModel, state: str, crit: dict, media: dict | None = None) -> torch.Tensor:
    """The pointer's request readout state (before the head), for exactness checks."""
    pk = m.build(state, crit, media=m._resolve_media(media))
    e, mask, pos, ids = m._batch([pk], m._device, m._mode())
    with torch.no_grad():
        h, _, _ = m._hidden(e, mask, pos, ids=ids)
    return h[0, pk["q_idx"]].double()


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / a.abs().max().clamp_min(1e-12))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-4-E2B")
    ap.add_argument("--four-bit", action="store_true")
    ap.add_argument("--tiny", action="store_true", help="random tiny Gemma 4 with the real config layout")
    ap.add_argument("--wav", default=None, help="a speech wav (default: a synthetic tone)")
    ap.add_argument("--image", default=None, help="a screenshot (default: random pixels)")
    ap.add_argument("--out", default=str(REPO / "runs" / "media_check.json"))
    a = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="media_check_"))
    name = tiny_gemma4(work / "tiny") if a.tiny else a.model
    dtype = torch.float32 if (a.tiny or not torch.cuda.is_available()) else None
    torch.manual_seed(0)
    m = UPPModel(UPPConfig(layout="block", gists=8, gist_budgets=(1, 2, 4, 8)), model_name=name, media=True,
                 load_in_4bit=a.four_bit, torch_dtype=dtype).eval()
    if a.tiny:  # float64 backbone: exactness is checked without float32 rounding noise
        m.backbone.double()
        m.gist_emb.data = m.gist_emb.data.double()
    if torch.cuda.is_available() and not a.tiny:
        m.head.to("cuda")
        m.gist_emb.data = m.gist_emb.data.to("cuda")
        if not a.four_bit:
            m.to("cuda")
    wav = a.wav
    if wav is None:
        import soundfile as sf

        wav = str(work / "tone.wav")
        sf.write(wav, (0.1 * np.sin(2 * np.pi * 220 * np.arange(48000) / 24000)).astype(np.float32), 24000)
    img = a.image
    if img is None:
        from PIL import Image

        img = str(work / "screen.png")
        Image.fromarray((np.random.RandomState(0).rand(448, 224, 3) * 255).astype("uint8")).save(img)
    rev = dict(reversed(list(CRIT.items())))
    res: dict = {"model": name, "per_layer_inputs": m._ple_model() is not None}
    a_t, _ = m.score("User Request: will it rain in Paris tomorrow?", CRIT)
    b_t, _ = m.score("User Request: will it rain in Paris tomorrow?", rev)
    res["text_order_max_dp"] = max(abs(a_t[k] - b_t[k]) for k in a_t)
    q = "User Request: will it rain in Paris tomorrow?"
    res["text_order_rel_state"] = rel(request_state(m, q, CRIT), request_state(m, q, rev))
    sp = {"audio": wav}
    enc = m.media_encoder.encode(sp)
    res["speech_soft_tokens"] = int(enc["embeds"].shape[0])
    p_a, _ = m.score("", CRIT, media=sp)
    p_b, _ = m.score("", rev, media=sp)
    res["speech_order_max_dp"] = max(abs(p_a[k] - p_b[k]) for k in p_a)
    res["speech_order_rel_state"] = rel(request_state(m, "", CRIT, sp), request_state(m, "", rev, sp))
    cat = m.compose([m.encode_tool(k, v) for k, v in reversed(list(CRIT.items()))])
    p_c, _ = m.score_composed("", cat, media=sp)
    res["speech_compose_vs_full_max_dp"] = max(abs(p_c[k] - p_a[k]) for k in p_a)
    p_2, _ = m.score_two_level("", m.encode_catalog(CRIT), coarse=1, top_m=len(CRIT), media=sp)
    res["speech_two_level_vs_full_max_dp"] = max(abs(p_2[k] - p_a[k]) for k in p_a)
    im = {"image": img}
    res["image_soft_tokens"] = int(m.media_encoder.encode(im)["embeds"].shape[0])
    p_i, _ = m.score("User Request: open the settings", CRIT, media=im)
    res["image_probs_sum"] = round(sum(p_i.values()), 6)
    m.train()
    rec = {"state": "User Request: will it rain in Paris?",
           "questions": {"tool_route": {"criteria": CRIT, "label": "get_weather"}}}
    out = m.forward_batch([rec, dict(rec, state="", media=sp)])
    out["loss"].backward()
    res["train_step_loss"] = round(float(out["loss"]), 4)
    res["gist_grad"] = m.gist_emb.grad is not None
    # Exactness is judged on the request state (relative error): an untrained head can amplify
    # float rounding into visible probability differences. fp64 tiny: ~0; fp32: ~1e-6; 4-bit/fp16: ~1e-3.
    tol = 1e-9 if a.tiny else 1e-2
    res["pass"] = bool(res["text_order_rel_state"] < tol and res["speech_order_rel_state"] < tol and res["gist_grad"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
