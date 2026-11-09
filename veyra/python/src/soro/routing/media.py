"""Speech and image requests for UPP pointers on an omni backbone (Gemma 4 E2B/E4B).

The tool catalog stays text and is encoded once; only the *request* changes modality.
``MediaEncoder`` turns a wav or an image into the backbone's own soft tokens (its
frozen audio / vision tower + projector, exactly what the backbone sees in normal
use) and returns them with the token-id layout the request needs:

    {"tokens": [boa, MEDIA, MEDIA, ..., eoa], "embeds": Tensor(n_media, d)}

``MEDIA`` (-3) marks the positions whose embeddings come from ``embeds``. UPP places
these tokens inside the request block at the fixed request offset, so every
exactness property of the block layout (composition, order equivariance, budget
nesting) holds unchanged for spoken and visual requests.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch

MEDIA = -3
SR = 16_000


def load_wav_16k(path: str | Path) -> np.ndarray:
    """Mono float32 at 16 kHz (Kokoro writes 24 kHz; Audio2Tool varies)."""
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        try:
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(int(sr), SR)
            wav = resample_poly(wav, SR // g, int(sr) // g).astype(np.float32)
        except ImportError:  # linear interpolation is enough for speech features
            n = int(round(len(wav) * SR / sr))
            wav = np.interp(np.linspace(0, len(wav) - 1, n), np.arange(len(wav)), wav).astype(np.float32)
    return wav


class MediaEncoder:
    """Frozen media towers of a multimodal backbone, with a small LRU cache per file."""

    def __init__(self, mm_model: Any, model_name: str, cache_size: int = 2048):
        from transformers import AutoProcessor

        self.model = mm_model
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tok = self.processor.tokenizer
        cfg = mm_model.config
        self.audio_token_id = getattr(cfg, "audio_token_id", None)
        self.image_token_id = getattr(cfg, "image_token_id", None)
        self.cache_size = cache_size
        self._cache: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
        for p in self.model.parameters():
            p.requires_grad_(False)

    TOWERS = ("audio_tower", "vision_tower", "embed_audio", "embed_vision")

    def to(self, device: torch.device | str) -> "MediaEncoder":
        """Move only the media towers (the language model is placed by UPPModel)."""
        for name in self.TOWERS:
            mod = getattr(self.model, name, None)
            if mod is not None and not getattr(mod, "is_loaded_in_4bit", False):
                mod.to(device)
        return self

    def _device_dtype(self, tower: str) -> tuple[torch.device, torch.dtype]:
        mod = getattr(self.model, tower)
        p = next(mod.parameters())
        dtype = p.dtype if p.dtype.is_floating_point else torch.float16
        return p.device, dtype

    def _layout(self, input_ids: torch.Tensor, media_id: int) -> list[int]:
        bos = self.tok.bos_token_id
        ids = [int(i) for i in input_ids[0].tolist()]
        if ids and bos is not None and ids[0] == bos:
            ids = ids[1:]
        return [MEDIA if i == media_id else i for i in ids]

    @torch.no_grad()
    def audio(self, path: str | Path) -> dict[str, Any]:
        wav = load_wav_16k(path)
        inp = self.processor(text=self.processor.audio_token, audio=[wav], return_tensors="pt")
        dev, dt = self._device_dtype("audio_tower")
        out = self.model.get_audio_features(inp["input_features"].to(dev, dt),
                                            inp["input_features_mask"].to(dev), return_dict=True)
        emb = out.pooler_output[out.attention_mask.to(out.pooler_output.device)]
        tokens = self._layout(inp["input_ids"], self.audio_token_id)
        return self._check(tokens, emb, path)

    @torch.no_grad()
    def image(self, path: str | Path) -> dict[str, Any]:
        from PIL import Image

        img = Image.open(str(path)).convert("RGB")
        inp = self.processor(text=self.processor.image_token, images=[img], return_tensors="pt")
        dev, dt = self._device_dtype("vision_tower")
        out = self.model.get_image_features(inp["pixel_values"].to(dev, dt), inp["image_position_ids"].to(dev),
                                            return_dict=True)
        emb = torch.cat(list(out.pooler_output), dim=0)
        tokens = self._layout(inp["input_ids"], self.image_token_id)
        return self._check(tokens, emb, path)

    @staticmethod
    def _check(tokens: list[int], emb: torch.Tensor, path: Any) -> dict[str, Any]:
        n = sum(1 for t in tokens if t == MEDIA)
        if n != emb.shape[0]:
            raise ValueError(f"{path}: {n} media slots but {emb.shape[0]} media embeddings")
        return {"tokens": tokens, "embeds": emb.detach()}

    def encode(self, media: dict[str, Any] | None) -> dict[str, Any] | None:
        """``{"audio": path}`` / ``{"image": path}`` -> tokens + embeds. Already-encoded dicts pass through."""
        if not media:
            return None
        if "embeds" in media:
            return media
        kind = "audio" if media.get("audio") else ("image" if media.get("image") else None)
        if kind is None:
            return None
        key = (kind, str(media[kind]))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        enc = self.audio(media["audio"]) if kind == "audio" else self.image(media["image"])
        self._cache[key] = enc
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return enc
