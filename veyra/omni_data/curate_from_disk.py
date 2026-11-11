"""Turn pass-2 files on disk into clean Omni inputs.

Filters teacher-disagreed generated rows, attaches existing wavs to TTS
source rows, writes the remaining TTS queue, then you run build.

    python veyra/omni_data/curate_from_disk.py --root veyra/runs/omni_data
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

from soro.omni_data.schema import OmniRow, read_rows, write_rows


def _parent_id(stem: str) -> tuple[str, str]:
    """wav stem -> (source_row_id, kind) where kind is full | streaming."""
    if "-p" in stem:
        base = stem.rsplit("-p", 1)[0]
        if base.endswith("-0") or base.endswith("-1"):
            base = base.rsplit("-", 1)[0]
        return base, "streaming"
    if stem.endswith("-0") or stem.endswith("-1"):
        return stem.rsplit("-", 1)[0], "full"
    return stem, "full"


def filter_generated(path: Path, out: Path) -> dict:
    rows = list(read_rows(path))
    kept, dropped = [], 0
    for r in rows:
        agrees = (r.teacher or {}).get("agrees")
        if agrees is False:
            dropped += 1
            continue
        kept.append(r)
    write_rows(kept, out)
    abstain = sum(r.label == "__abstain__" for r in kept)
    return {"in": len(rows), "kept": len(kept), "dropped_disagree": dropped, "abstain": abstain}


def attach_wavs(tts_source: Path, wav_dir: Path, out_jsonl: Path) -> dict:
    by_id = {r.id: r for r in read_rows(tts_source)}
    files = list(wav_dir.glob("*.wav"))
    groups: dict[str, list[Path]] = defaultdict(list)
    unmatched = 0
    for wav in files:
        pid, _kind = _parent_id(wav.stem)
        if pid not in by_id:
            unmatched += 1
            continue
        groups[pid].append(wav)
    out: list[OmniRow] = []
    for pid, wavs in groups.items():
        src = by_id[pid]
        full = [w for w in wavs if "-p" not in w.stem]
        stream = [w for w in wavs if "-p" in w.stem]
        if not full:
            full = wavs[:1]
            stream = [w for w in wavs if w not in full]
        for j, wav in enumerate(full):
            row = OmniRow(
                id=f"{pid}~tts{j}",
                task=src.task,
                modality="audio",
                options=src.options,
                label=src.label,
                source="tts_kokoro",
                license="own",
                group=src.group,
                text="",
                context=src.context,
                audio=str(wav),
                transcript=src.text,
                package=src.package,
                split=src.split,
                tags=list(src.tags) + ["tts"],
                teacher=src.teacher,
                meta={**src.meta, "parent": pid},
            )
            out.append(row)
        for wav in stream:
            frac = wav.stem.rsplit("-p", 1)[-1]
            out.append(OmniRow(
                id=f"{pid}~tts0-p{frac}",
                task=src.task,
                modality="audio",
                options=src.options,
                label=src.label,
                source="tts_kokoro",
                license="own",
                group=src.group,
                text="",
                context=src.context,
                audio=str(wav),
                transcript=src.text,
                package=src.package,
                split=src.split,
                tags=list(src.tags) + ["tts", "streaming", f"prefix:0.{frac}" if frac.isdigit() else f"prefix:{frac}"],
                teacher=src.teacher,
                meta={**src.meta, "parent": pid, "prefix_fraction": frac},
            ))
    write_rows(out, out_jsonl)
    remaining = [r for r in by_id.values() if r.id not in groups]
    return {
        "wav_files": len(files),
        "matched_parents": len(groups),
        "unmatched_wavs": unmatched,
        "audio_rows": len(out),
        "tts_remaining": len(remaining),
        "remaining_ids": remaining,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="veyra/runs/omni_data")
    args = ap.parse_args()
    root = Path(args.root)
    gen_in = root / "verified" / "text_generated.jsonl"
    gen_out = root / "verified" / "text_generated_agreed.jsonl"
    gen_stats = filter_generated(gen_in, gen_out)
    shutil.copy2(gen_out, gen_in)
    print("generated_filter", json.dumps({k: v for k, v in gen_stats.items()}), flush=True)

    audio_dir = root / "audio"
    audio_dir.mkdir(exist_ok=True)
    wav_dir = audio_dir / "wav"
    tts_stats = attach_wavs(root / "tts_source.jsonl", wav_dir, audio_dir / "audio_tts.jsonl")
    remaining = tts_stats.pop("remaining_ids")
    write_rows(remaining, audio_dir / "tts_remaining.jsonl")
    print("audio_attach", json.dumps(tts_stats | {"tts_remaining_written": len(remaining)}), flush=True)


if __name__ == "__main__":
    main()
