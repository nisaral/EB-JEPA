"""Audio rows: TTS rendering, augmentation, streaming prefixes, Audio2Tool (eval only).

Training audio is our own text rendered with an Apache-2.0 TTS (Kokoro-82M) and
then degraded with realistic noise, speed, gain, reverb and telephone band-limit.
Audio2Tool is CC BY-NC 4.0, so it is converted for evaluation only.

TTS-only audio does not prepare a model for real voices; real recordings go in
the human eval set (see docs/OMNI_DATA.md), and the BLSP-style consistency loss
(audio vs transcript) is what keeps the audio path tied to the text path.

Waveforms are float32 numpy arrays at ``SR`` Hz; files are 16-bit PCM wav
written with the standard library, so no audio dependency is needed except TTS.
"""

from __future__ import annotations

import json
import math
import random
import wave
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from soro.omni_data.schema import ABSTAIN, OmniRow, Option, special_option

SR = 16000


# ---------------------------------------------------------------------- io
def write_wav(path: str | Path, audio: np.ndarray, sr: int = SR) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        data = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(1)
    return data, sr


def resample(audio: np.ndarray, sr_in: int, sr_out: int = SR) -> np.ndarray:
    if sr_in == sr_out:
        return audio.astype(np.float32)
    n_out = int(round(len(audio) * sr_out / sr_in))
    return np.interp(np.linspace(0, len(audio) - 1, n_out), np.arange(len(audio)), audio).astype(np.float32)


# ---------------------------------------------------------------------- augmentation
def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator, noise: np.ndarray | None = None,
              color: str = "pink") -> np.ndarray:
    if noise is None:
        white = rng.standard_normal(len(x)).astype(np.float32)
        if color == "pink":  # 1/f shaping in the frequency domain
            spec = np.fft.rfft(white)
            f = np.arange(len(spec), dtype=np.float32)
            f[0] = 1.0
            white = np.fft.irfft(spec / np.sqrt(f), n=len(x)).astype(np.float32)
        noise = white
    else:
        reps = int(math.ceil(len(x) / len(noise)))
        start = int(rng.integers(0, max(1, len(noise) * reps - len(x) + 1)))
        noise = np.tile(noise, reps)[start:start + len(x)]
    target = rms(x) / (10 ** (snr_db / 20))
    return (x + noise * (target / rms(noise))).astype(np.float32)


def speed(x: np.ndarray, factor: float) -> np.ndarray:
    """Tempo and pitch change together (resample), like Kaldi speed perturbation."""
    n = int(round(len(x) / factor))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def reverb(x: np.ndarray, rt60: float, rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    n = int(sr * rt60)
    t = np.arange(n) / sr
    ir = rng.standard_normal(n).astype(np.float32) * np.exp(-6.9 * t / rt60).astype(np.float32)
    ir[0] = 1.0
    ir /= np.abs(ir).sum() ** 0.5
    y = np.convolve(x, ir)[: len(x)]
    return (y * (rms(x) / rms(y))).astype(np.float32)


def telephone(x: np.ndarray, sr: int = SR, lo: float = 300.0, hi: float = 3400.0) -> np.ndarray:
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    spec[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(spec, n=len(x)).astype(np.float32)


def augment(x: np.ndarray, rng: np.random.Generator, noise_bank: list[np.ndarray] | None = None) -> tuple[np.ndarray, dict]:
    """One random degradation chain. Returns audio and a record of what was applied."""
    rec: dict[str, Any] = {}
    if rng.random() < 0.5:
        f = float(rng.choice([0.9, 0.95, 1.05, 1.1]))
        x, rec["speed"] = speed(x, f), f
    if rng.random() < 0.3:
        rt = float(rng.uniform(0.2, 0.7))
        x, rec["reverb_rt60"] = reverb(x, rt, rng), round(rt, 2)
    if rng.random() < 0.15:
        x, rec["telephone"] = telephone(x), True
    if rng.random() < 0.8:
        snr = float(rng.uniform(0, 25))
        nz = noise_bank[int(rng.integers(len(noise_bank)))] if noise_bank else None
        x, rec["snr_db"] = add_noise(x, snr, rng, nz), round(snr, 1)
    g = float(rng.uniform(0.3, 1.0))
    x = x / max(np.abs(x).max(), 1e-6) * g
    rec["gain"] = round(g, 2)
    return x.astype(np.float32), rec


def prefixes(x: np.ndarray, fractions: tuple[float, ...] = (0.4, 0.6, 0.8)) -> list[tuple[float, np.ndarray]]:
    """Cut points for streaming rows (gold stays the full-utterance decision)."""
    return [(f, x[: int(len(x) * f)]) for f in fractions]


# ---------------------------------------------------------------------- TTS
class KokoroTTS:
    """Kokoro-82M (Apache-2.0). ``pip install kokoro>=0.9.2 soundfile`` and espeak-ng."""

    VOICES_EN = ["af_heart", "af_bella", "af_nicole", "af_sarah", "am_adam", "am_michael", "am_fenrir",
                 "bf_emma", "bf_isabella", "bm_george", "bm_lewis"]

    def __init__(self, lang_code: str = "a"):
        from kokoro import KPipeline

        self.pipes = {"a": KPipeline(lang_code="a"), "b": KPipeline(lang_code="b")}

    def __call__(self, text: str, voice: str, speed_: float = 1.0) -> np.ndarray:
        pipe = self.pipes["b" if voice.startswith("b") else "a"]
        chunks = [np.asarray(a, dtype=np.float32) for _, _, a in pipe(text, voice=voice, speed=speed_)]
        return resample(np.concatenate(chunks) if chunks else np.zeros(1, np.float32), 24000, SR)


def render_rows(rows: list[OmniRow], out_dir: str | Path, tts, voices: list[str], per_row: int = 1,
                augment_prob: float = 0.85, streaming_rate: float = 0.1, seed: int = 0,
                noise_bank: list[np.ndarray] | None = None) -> list[OmniRow]:
    """Text rows -> audio rows. The spoken words are the request; ``transcript`` keeps them for consistency loss."""
    rng = np.random.default_rng(seed)
    prng = random.Random(seed)
    out_dir = Path(out_dir)
    out = []
    for r in rows:
        if not r.text or r.modality != "text":
            continue
        for j in range(per_row):
            voice = prng.choice(voices)
            wav = tts(r.text, voice)
            rec: dict[str, Any] = {"voice": voice}
            if prng.random() < augment_prob:
                wav, aug = augment(wav, rng, noise_bank)
                rec.update(aug)
            path = out_dir / f"{r.id}-{j}.wav"
            write_wav(path, wav)
            base = replace(r, id=f"{r.id}~tts{j}", modality="audio", audio=str(path), transcript=r.text, text="",
                           source="tts_kokoro", license="own", tags=r.tags + ["tts"], teacher=r.teacher,
                           meta={**r.meta, "parent": r.id, "audio": rec, "duration_s": round(len(wav) / SR, 2)})
            out.append(base)
            if prng.random() < streaming_rate:
                for frac, cut in prefixes(wav):
                    p = out_dir / f"{r.id}-{j}-p{int(frac * 100)}.wav"
                    write_wav(p, cut)
                    out.append(replace(base, id=f"{base.id}-p{int(frac * 100)}", audio=str(p),
                                       tags=base.tags + ["streaming", f"prefix:{frac}"],
                                       meta={**base.meta, "prefix_fraction": frac}))
    return out


# ---------------------------------------------------------------------- Audio2Tool (eval only)
def audio2tool_rows(records: list[dict[str, Any]], audio_root: str, distractors: int = 5, seed: int = 0) -> list[OmniRow]:
    """Convert Audio2Tool (CC BY-NC 4.0) records into eval rows.

    The published records list one function per query, and the gold ``tool_name``
    can differ from that function's name (e.g. setLighting vs setLightState), so
    the gold option is matched by ``tool_id``. Distractors come from other tools
    in the same domain; multi-intent tiers are skipped (single-decision task).
    """
    rng = random.Random(seed)
    by_domain: dict[str, dict[int, dict[str, Any]]] = {}
    for rec in records:
        for fn in rec.get("functions") or []:
            by_domain.setdefault(fn["domain"], {})[fn["tool_id"]] = fn
    rows = []
    for rec in records:
        if rec.get("additional_tool_calls") or rec.get("tier") in ("tier3_multi_intent", "tier8_intent_blending"):
            continue
        gold_fn = next((f for f in rec.get("functions") or [] if f["tool_id"] == rec["tool_id"]), None)
        if gold_fn is None:
            continue
        pool = [f for tid, f in by_domain.get(rec["domain"], {}).items() if tid != rec["tool_id"]]
        rng.shuffle(pool)
        fns = [gold_fn] + pool[:distractors]
        rng.shuffle(fns)
        opts = [Option(key=f["tool_name"], text=f"{f['signature']}. {f['description']}") for f in fns]
        opts.append(special_option(ABSTAIN))
        rows.append(OmniRow(
            id=f"a2t-{rec['id']}", task="tool_route", modality="audio", options=opts, label=gold_fn["tool_name"],
            source="audio2tool", license="cc-by-nc-4.0", group=f"a2t-q{rec['query_idx']}", split="test",
            audio=str(Path(audio_root) / rec["path"]), transcript=rec["query"], package=f"a2t/{rec['domain']}",
            tags=[rec["tier"], "eval_only"],
            meta={"speaker_source": rec.get("speaker_source"), "duration_s": rec.get("duration"),
                  "published_tool_name": rec.get("tool_name")},
        ))
    return rows


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
