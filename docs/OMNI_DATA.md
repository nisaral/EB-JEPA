# Soro-Kev-Omni data pipeline

Code: `veyra/python/src/soro/omni_data/` (library), `veyra/omni_data/` (stage scripts, mixture config).
Tests: `tests/test_omni_data.py`.

## Principles

1. **Hard, distinct and real beats big.** We cap paraphrases, mine look-alikes, and keep only rows whose answer a teacher confirms.
2. **License first.** Only Apache/MIT/CC-BY/CC0/own data trains the Apache-released model. Non-commercial sets are evaluation only (`licenses.py`).
3. **Nothing leaks.** Splits are group-disjoint (derived rows inherit the parent's group); train is checked against every eval split for exact and near-duplicate requests and shared packages.
4. **Every row says where it came from.** `source`, `license`, `group`, `package`, `tags`, `teacher`, `meta.parent`.

## Row format (`schema.OmniRow`)

`text` / `image` / `audio` (+ `transcript`, `ocr`, `context`), `options` (tools or UI elements with bbox), `label`
(an option key, `__abstain__` or `__need_input__`), `task` (`tool_route`, `ui_point`), `modality`, `split`,
`tags`, `teacher` (permutation-averaged probabilities). `to_upp_record()` exports the training format.

## Sources

| Source | Modality | License | Use |
| :-- | :-- | :-- | :-- |
| Soro train (existing) | text | mixed, see DATA_LICENSES.md | train |
| Look-alike / near-miss rows (`text.py`) | text | own | train (near-miss only after teacher check) |
| Teacher-written contrastive rows (`generate.py`) | text | own (Apache teachers) | train after agreement filter |
| Kokoro-82M TTS of our rows (`audio.py`) | audio | Apache-2.0 model, own text | train |
| OS-Atlas grounding (`image.py`) | image | Apache-2.0 repo; per-upstream terms in `OS_ATLAS_SUBSOURCES` | train where upstream allows |
| Image-dependent routing (`generate.py` + necessity filter) | image | own requests; image license per source | train |
| Audio2Tool | audio | **CC BY-NC 4.0** (dataset card) | **eval only** |
| BFCL v3 slice, BFCL Audio, ScreenSpot-Pro, M³-Bench | mixed | various / unstated | eval only |
| Human eval (ours) | all | own | final eval only |

Found while building: Audio2Tool's paper page says CC BY 4.0 but the dataset card says CC BY-NC 4.0, and some
records' gold `tool_name` differs from the listed function's name (e.g. `setLighting` vs `setLightState`); the
converter matches gold by `tool_id`.

## Stages and run order

| # | Stage | Where | Command |
| :-- | :-- | :-- | :-- |
| 1 | Convert + harden text (collisions, near-miss candidates, eval files) | CPU | `python veyra/omni_data/stage_text.py` |
| 2 | Contrastive request generation (RLCD-style) | GPU | `stages.py generate --model <instruct teacher>` |
| 3 | Teacher labels on near-miss + generated + a base sample | GPU | `stages.py teacher --model <teacher> [--multimodal --four-bit]` |
| 4 | Verification filter | CPU | `stages.py filter` |
| 5 | TTS + augmentation (+ streaming prefixes) on verified text | GPU/CPU | `stages.py tts` |
| 6 | OS-Atlas UI rows per subsource | CPU + download | `stages.py osatlas` |
| 7 | Image-dependent routing (teacher writes request, necessity filter) | GPU | `generate.IMAGE_REQUEST_PROMPT` + `necessity_filter` |
| 8 | Audio2Tool eval conversion | download | `stages.py a2t` |
| 9 | Build: gate, dedupe, leakage, verify, mix, split, report | CPU | `python -m soro.omni_data.build --config veyra/omni_data/omni_mix.json --out runs/omni_data/build` |

`build_report.json` lists every shortfall against `omni_mix.json`; buckets are never padded with duplicates.

## Quality gates (all automatic)

- Schema validation (label in menu, bbox sane, modality has its input, license present).
- Exact dedupe on what the model sees; MinHash near-dupes (same menu and label, Jaccard ≥ 0.85).
- Paraphrase cap per (package, label).
- Leakage: exact and near-duplicate requests and shared packages between train and each eval split; leaked eval rows are dropped and reported.
- Teacher verification for derived abstain rows and generated rows; label audit lists gold rows where the teacher confidently disagrees (review queue, not auto-delete).
- Balance report per split: abstain, need-input, collision and near-miss shares, modality, source, menu size.

## Teachers

- Text: an Apache-2.0 instruct model for generation (e.g. Qwen3-4B/8B-Instruct) and scoring.
- Image + audio + text: `Qwen/Qwen3-Omni-30B-A3B-Instruct` (card license_name apache-2.0) via `--multimodal --four-bit` on an A100/L40S.
- Image + text, stronger: `google/gemma-4-31B` (Apache-2.0).
- Scoring reads option-letter logits under 4 menu rotations and averages per option; a temperature is fitted on gold rows.

## Human eval set (the most important missing piece)

Target: **500–1,000 distinct requests per modality**, never used for training, selection or q̂.

Protocol:
1. Pick 60–100 real tool menus (MCP servers, phone apps, smart home) covering the target use cases.
2. Writers who have not seen the training data write requests for each menu: about 70% that need one tool, 15% that need nothing listed, 10% look-alike traps, 5% that need an attachment that is not there.
3. Image: real screenshots and photos taken by the writers (no web scraping), with the request typed against them.
4. Audio: the same writers record their requests on their own phones and laptops (quiet and noisy rooms, accents as they come). Even 1,000 real clips beat any amount of TTS for measuring real-voice robustness.
5. Two people label each row independently; disagreements are discussed and either fixed or dropped. Report agreement.
6. Store as OmniRow with `split="human_eval"`, `source="human_eval"`, `license="own"`, and keep consent records for every voice.

## Current numbers (text only, before the GPU stages; 8 Nov 2026)

Stage 1 (`stage_text.py`, 76 s on CPU):

| | Before | After stage 1 |
| :-- | --: | --: |
| Train rows | 23,192 | 32,910 (incl. 2,706 near-miss candidates awaiting teacher check) |
| Distinct tools in pool | — | 21,123 |
| Collision (look-alike) share | 0.8% | 22.1% |
| Abstain share (incl. candidates) | 9.2% | 14.7% |

Problems found in the existing data and handled:
- **Template label noise.** 34 requests are reused across many unrelated packages; the worst three ("Show the status
  of order ORD-14.", "Search for salt and pepper grinders.", "Book a table for me and my friend.") appear in 59–68
  packages with ~120 different gold tools (e.g. "book a table" labelled with a tool of a flight MCP). 503 train rows
  now need teacher agreement to stay; 48 such rows were removed from `val_zeroshot` into `eval_zeroshot_suspect.jsonl`.
- **Train/test leakage.** 41 zero-shot test requests appeared verbatim in train (40 of them the template rows above);
  the remaining one is dropped by the build. The earlier disjointness audit compared full states and missed these.
- **Same-tool duplicates** (`send_email` vs `send-email`) are excluded from look-alike mining so collision rows stay
  unambiguous and near-miss rows are not mislabelled.

Build (text inputs only; image/audio/generated inputs not produced yet): 28,336 train / 1,209 selection / 3,371 test,
group-disjoint, 0 invalid rows, 40 exact duplicates and 117 paraphrase-capped rows removed, 502 unverified rows held
back. Shortfalls against `omni_mix.json` are filled by the GPU stages (generation, teacher verification, TTS, OS-Atlas,
image routing).

## Full audit (8 Nov 2026, `python -m soro.omni_data.audit`, Qwen tokenizer)

| File | Rows | Invalid | Abstain | Delexicalized gold | Exact dup | Prompt tokens p95 / max | > 512 tokens |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| text_base | 23,192 | 0 | 9.2% | 22.0% | 24 | 504 / 791 | 4.1% |
| text_collisions | 7,012 | 0 | 0% | 23.6% | 16 | 501 / 758 | 4.1% |
| text_near_miss (candidates) | 2,706 | 0 | 100% | 0% | 0 | 509 / 718 | 4.4% |
| eval_zeroshot (cleaned) | 2,172 | 0 | 9.6% | 26.8% | 0 | 481 / 644 | 1.8% |
| eval_external | 1,200 | 0 | 11.2% | 0% | 0 | 372 / 505 | 0% |

- No invalid rows, no empty menus, no duplicate option texts, no prompt-injection-like requests.
- Exact duplicates (40) are removed by the build.
- 22–27% of gold labels are delexicalized (`op_1234`): those rows test description-only routing, by design.
- **Sequence length:** the longest text prompt is 791 tokens, so text training uses `max_seq_len=832` (was 1536 for
  0.8B and 512 for 4B). The released 4B, trained at 512, silently skipped the longest ~4% of rows.

## 2x T4 settings

| Stage | Setting | Why |
| :-- | :-- | :-- |
| All GPU stages | `--shard k/2`, one process per T4 | both GPUs busy, no inter-GPU traffic |
| Generation | Qwen3-4B-Instruct-2507 in fp16, batch 16, left padding | T4 has no bf16; fits in 16 GB |
| Teacher | Qwen3-8B 4-bit NF4, fp16 compute, 4 rotations in one padded forward | ~4x fewer forwards than rotating one at a time |
| Gemma 4 on T4 | 4-bit with fp32 compute | Gemma activations overflow fp16 |
| P0 / training | fp32 master weights + fp16 autocast + GradScaler; `max_seq_len 832` | stable on T4, nothing dropped |
| Env | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | less fragmentation on 16 GB cards |
