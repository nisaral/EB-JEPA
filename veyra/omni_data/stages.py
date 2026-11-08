"""Omni data stages that need a GPU or downloads (run on Kaggle / Colab).

    python veyra/omni_data/stages.py teacher   --in X.jsonl --out Y.jsonl --model Qwen/Qwen3-Omni-30B-A3B-Instruct --multimodal --four-bit
    python veyra/omni_data/stages.py generate  --in text_base.jsonl --out gen.jsonl --model Qwen/Qwen3-4B-Instruct-2507 --limit 3000
    python veyra/omni_data/stages.py tts       --in verified_text.jsonl --out audio.jsonl --wav-dir wav/ --limit 20000
    python veyra/omni_data/stages.py a2t       --out eval_audio2tool.jsonl --root a2t/   (eval only, CC BY-NC)
    python veyra/omni_data/stages.py osatlas   --json mobile_domain/uibert_raw.json --images imgs/ --subsource uibert --out ui.jsonl
    python veyra/omni_data/stages.py filter    --in labelled.jsonl --out kept.jsonl   (near-miss + generated verification)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "veyra" / "python" / "src"))

from soro.omni_data import audio, image, licenses  # noqa: E402
from soro.omni_data.schema import read_rows, write_rows  # noqa: E402


def _limit(rows, n, seed=0, shard: str | None = None):
    """Deterministic subset, then this process's shard ("k/n", one process per GPU)."""
    rows = list(rows)
    if n and len(rows) > n:
        random.Random(seed).shuffle(rows)
        rows = rows[:n]
    if shard:
        k, n_sh = (int(x) for x in shard.split("/"))
        rows = rows[k::n_sh]
    return rows


def cmd_teacher(a):
    from soro.omni_data.teacher import TeacherScorer, fit_temperature

    rows = _limit(read_rows(a.inp), a.limit, shard=a.shard)
    t = TeacherScorer(a.model, device=a.device, load_in_4bit=a.four_bit, multimodal=a.multimodal,
                      rotations=a.rotations)
    labelled = []
    for i in range(0, len(rows), 50):
        labelled += t.label(rows[i:i + 50])
        print(f"  labelled {len(labelled)}/{len(rows)}", flush=True)
    gold_rows = [r for r in labelled if "needs_teacher" not in r.tags]
    temp = fit_temperature(gold_rows) if gold_rows else 1.0
    for r in labelled:
        r.teacher["fitted_temperature"] = temp
    n = write_rows(labelled, a.out)
    agree = (sum(r.teacher["agrees"] for r in gold_rows) / len(gold_rows)) if gold_rows else None
    print(json.dumps({"written": n, "teacher": a.model, "gold_agreement": None if agree is None else round(agree, 4), "gold_rows": len(gold_rows), "temperature": temp}))


def cmd_generate(a):
    from soro.omni_data.generate import build_prompts, generate_requests, to_rows
    from soro.omni_data.text import ToolPool

    base = list(read_rows(a.inp))
    pool = ToolPool(base)
    parents = _limit([r for r in base if r.task == "tool_route" and "generic_request" not in r.tags], a.limit, shard=a.shard)
    prompts = build_prompts(parents, lambda k: pool.lookalikes(k, k=3))
    gen = generate_requests(prompts, a.model, device=a.device)
    rows = to_rows(gen, {r.id: r for r in parents}, a.model)
    print(json.dumps({"prompts": len(prompts), "rows": write_rows(rows, a.out)}))


def cmd_tts(a):
    rows = _limit([r for r in read_rows(a.inp) if r.modality == "text" and r.text and "needs_teacher" not in r.tags], a.limit, shard=a.shard)
    tts = audio.KokoroTTS()
    noise = [audio.read_wav(p)[0] for p in Path(a.noise_dir).glob("*.wav")] if a.noise_dir else None
    out = audio.render_rows(rows, a.wav_dir, tts, audio.KokoroTTS.VOICES_EN, per_row=a.per_row, noise_bank=noise)
    print(json.dumps({"audio_rows": write_rows(out, a.out)}))


def cmd_a2t(a):
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download("RVtech/Audio2Tool", repo_type="dataset", local_dir=a.root,
                                  allow_patterns=["public/*_data/*.json"] + ([] if a.no_audio else ["public/*_audios/**"])))
    recs = []
    for f in sorted((root / "public").glob("*_data/*.json")):
        recs += audio.load_json(str(f))
    rows = audio.audio2tool_rows(recs, str(root / "public"))
    print(json.dumps({"records": len(recs), "eval_rows": write_rows(rows, a.out), "license": "cc-by-nc-4.0 (eval only)"}))


def cmd_osatlas(a):
    lic = licenses.OS_ATLAS_SUBSOURCES.get(a.subsource, "unknown")
    recs = image.load_os_atlas_json(a.json)
    rows = image.os_atlas_rows(recs, a.images, a.subsource, lic)
    if not licenses.trainable(lic):
        for r in rows:
            r.split = "test"
    print(json.dumps({"records": len(recs), "rows": write_rows(rows, a.out), "license": lic}))


def cmd_filter(a):
    from soro.omni_data.generate import agree_filter
    from soro.omni_data.text import filter_near_misses

    rows = list(read_rows(a.inp))
    rows, d1 = filter_near_misses(rows)
    rows, d2 = agree_filter(rows)
    print(json.dumps({"kept": write_rows(rows, a.out), "dropped_near_miss": d1, "dropped_generated": d2}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("teacher", "generate", "tts", "filter"):
        p = sub.add_parser(name)
        p.add_argument("--in", dest="inp", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--limit", type=int, default=0)
        p.add_argument("--device", default="cuda")
        p.add_argument("--shard", default=None, help='"k/n": process shard k of n (one per GPU)')
        if name in ("teacher", "generate"):
            p.add_argument("--model", required=True)
        if name == "teacher":
            p.add_argument("--multimodal", action="store_true")
            p.add_argument("--four-bit", action="store_true")
            p.add_argument("--rotations", type=int, default=4)
        if name == "tts":
            p.add_argument("--wav-dir", required=True)
            p.add_argument("--per-row", type=int, default=1)
            p.add_argument("--noise-dir", default=None)
    p = sub.add_parser("a2t")
    p.add_argument("--out", required=True)
    p.add_argument("--root", default="audio2tool")
    p.add_argument("--no-audio", action="store_true")
    p = sub.add_parser("osatlas")
    for k in ("--json", "--images", "--subsource", "--out"):
        p.add_argument(k, required=True)
    a = ap.parse_args()
    {"teacher": cmd_teacher, "generate": cmd_generate, "tts": cmd_tts, "a2t": cmd_a2t, "osatlas": cmd_osatlas,
     "filter": cmd_filter}[a.cmd](a)


if __name__ == "__main__":
    main()
