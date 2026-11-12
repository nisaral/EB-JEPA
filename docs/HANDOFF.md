# Soro-Kev-Omni knowledge transfer (PLAN v4)

Written **2026-11-12** for the next agent. Local folder: `C:\Users\nisar\OneDrive\Desktop\EB-JEPA`. GitHub: `nisaral/Soro-Boro`, branch **`upp-p0`**. HF user: **`nisaralll`**. Do not push to `main` unless the owner asks.

**Secrets:** never put Kaggle/HF tokens in this file, git, or chat logs if you can avoid it. Use `KAGGLE_API_TOKEN` as an env var. The owner may also have `%USERPROFILE%\.kaggle\access_token`. Revoke tokens after runs (Kaggle Settings > API). Do not email the owner's address to third parties.

---

## 0. Read this first

**Winning architecture is pointer A (query-first Kev pointer).** Block / xattn / late-gist layouts lost a 100-step smoke and a 1000-step follow-up. Do not train Gemma, FX, L0, or F0 for accuracy.

**Current stage:** S0 is done (papers + 842 A-errors). **S1 GPU ran and mixed-passed:** accuracy up, **order-flip gate failed** (still 0.05, need &lt; 0.01). **Do not start S2** until S1 flips are fixed or the owner explicitly drops that gate.

PLAN v4 is the scope. Out of scope until the matching stage: official BFCL leaderboard submit, multi-turn GRPO, 32k TTS in one 12h session, OS-Atlas GPU, Hub dataset `keyushnisar/omni-da` (that listing is a git dump, not training data).

---

## 1. Goal

A*-level paper (target **NeurIPS 2027**): a *meta-learner for the task* (tool/option selection) across text, speech, screenshots. Not a reasoning paper. Not a generator. Release on Hugging Face.

Thesis: request (text / speech / screen) + listed tool cards + abstain; one prefill pointer; conformal EXECUTE / ASK / ABSTAIN; later action cards with costs; later result-check gate.

---

## 2. PLAN v4 (locked)

Backbone: **pointer A, query-first**. No block/xattn work.

| Stage | What | GPU? | Status |
|---|---|---|---|
| **S0** | Six papers + 100+ A selection errors bucketed | No (dump used T4) | **Done** |
| **S1** | Shuffle + order KL, curriculum 4→64, temp floor, distractors | Yes | **Ran; acc pass, flips fail** |
| **S2** | ANSWER / ASK / ESCALATE / ABSTAIN **always in the menu**, costs on the card. Loss = listwise CE + **exact expected cost** (no GRPO). Conformal per action group. Eval When2Call, ClarifyBench, AgentAbstain | Yes after S1 lock | **Not started** |
| **S3** | Cross-modal after TTS + screenshots. Twins, confident-modality teacher, latent alignment, fusion. Baselines Whisper→text-A, OCR→text-A. Gate: speech within 3 pts of text, agreement &gt; 95%, coverage ≥ 90% / modality | Yes after data | **Blocked on data** |
| **S4** | Result-check: same request, result swapped/faulted. Options = spans + RESULT_FAILED + USE_PRIOR. Gate: follows result &gt; 95% | Later | **Not started** |
| **S5** | Paper section: per-turn conformal \|S\| vs first fail on tau2 vs TRACER. Multi-turn GRPO optional and last | Optional | **Not started** |

Action cards (S2): always in the menu (owner confirmed).

---

## 3. Architecture decision (closed)

Code: `veyra/python/src/soro/routing/upp.py`, trainer `veyra/training/p0_upp_ablation.py`. Config **A** = `layout=query_first`, `head=dot`.

| Run | Kernel | Result |
|---|---|---|
| 100-step smoke | `keyushnisar/soro-upp-p0-smoke` | A sel **0.613** / zs **0.688**. F0 **0.188**. FX **0.237**. Chance ~0.22 |
| 1000-step | `keyushnisar/soro-upp-p0-1k` | A **0.858 / 0.900**, flips 0.05. L0 **0.342 / 0.408**. FX **0.250 / 0.367**. ~45 min each, ~2.2 GB |
| S1 1000-step | `jackobperalta/soro-s1-a` | A **0.875 / 0.933**, flips still **0.05**. 97.4 min, 2.4 GB |

**Lock:** train Omni on **A**. Block/xattn/late are cache/latency experiments only.

S1 added `--s1` (shuffle KL 0.3, curriculum 4–64, `log_temp` floor 0.05). Accuracy improved; **flips did not drop**. Next S1 attempt needs a stronger order term (higher KL, CE on both permutations, or permute every batch without a weak detach-KL).

Local results copies:

- `veyra/runs/p0_smoke_kaggle_out/`
- `veyra/runs/p0_1k_kaggle/Soro-Boro/runs/p0_1k.json`
- `veyra/runs/s1_a_kaggle/Soro-Boro/runs/s1_a/results.json`

---

## 4. S0 papers (cite / beat)

Full notes: `docs/S0_PLAN_V4.md`.

- **TRUST (2606.06976):** uncertainty in RL reward. We use exact expected cost (S2), not GRPO, plus conformal abstain.
- **Role-stratified CRC (2607.24343):** per-argument-role conformal. Cite; we do not claim per-field guarantees yet.
- **When2Call (2504.18851):** four-way call / ask / cannot / direct-answer. S2 is this as a pointer with those cards always present.
- **ClarifyBench / SAGE (2511.08798):** EVPI over parameters. ASK is a discrete card, not free-form questions.
- **LettuceDetect follow-on (2607.00895):** span hallucination on tool output. S4 baseline.
- **TRACER (2602.11409):** trajectory risk on tau2. S5 comparison.

---

## 5. S0 error dump (842 errors)

Kernel: `keyushnisar/soro-s0-dump-a` v5 (COMPLETE). Production **Soro-Kev-4B** (not the 0.6B 1k weights; those were not saved). Menus clipped to 16, seq 768, OOM rows skipped.

Files: `veyra/runs/s0_a_errors_v5/s0_a_errors/summary.json` and `errors_*.jsonl`.

| Split | n | Acc | Errors |
|---|---|---|---|
| test | 3357 | 0.900 | 336 |
| selection | 1450 | 0.807 | 280 |
| eval_zeroshot | 2168 | 0.896 | 226 |
| **total** | **6975** | | **842** |

Buckets (all splits): **name_overlap_swap 283**, false_call 192, over_abstain 162, confident_wrong 141, other_swap 45, gold_buried 19.

S1 should target name-overlap / collisions. S2 should target false_call + over_abstain.

---

## 6. Data

Permanent copy: `veyra/runs/omni_data/` (jsonl + 2771 wavs). Eval splits in git: `veyra/omni_data/eval_splits/{test,selection,eval_zeroshot}.jsonl`.

**Usable now:** `veyra/runs/omni_data/build_v2/` — 33,984 train / 1,456 selection / 3,361 test. Text routing is real. Audio twins ~2,089 after a dedupe fix (near-dupe must not collapse audio vs text of the same request: `soro/omni_data/quality.py`).

**Not usable:** `keyushnisar/omni-da` (git clone of Soro-Boro). Screenshots / OS-Atlas = 0. Full TTS = 2,771 / 35,057 (~8%). Remaining queue: `veyra/runs/omni_data/audio/tts_remaining.jsonl` (32,434). Teacher-disagreed generated positives were dropped; remaining generated agreed rows are almost all abstain.

**Do not** dump 32k Kokoro or OS-Atlas on a 12h GPU session. TTS = CPU/shards. OS-Atlas = download Apache subsets (`fineweb_3m`, `aw_mobile`). Then rebuild: `python -m soro.omni_data.build --config omni_data/omni_mix.json` from `veyra/`.

Pass-1 zip (`Downloads/omni_data_pass1.zip`) had the original `build/` before v2. Pass-2 kernel `keyushnisar/soro-omni-data-pass2` was 12h-cancelled.

---

## 7. Released models (HF)

- https://huggingface.co/nisaralll/soro-kev-4b — Qwen3-4B NF4 LoRA + pointer head
- https://huggingface.co/nisaralll/soro-kev-0.8b — Qwen3.5-0.8B-Base
- https://huggingface.co/datasets/nisaralll/soro-universal-tool-harness

Load: `SoroHarness.from_pretrained("nisaralll/soro-kev-4b")`. Hub id with `/` uses `snapshot_download`. `4b` in the name + CUDA + bitsandbytes → NF4; else bf16/fp16.

GitHub code (not weights): `soro-kev-kaggle` and `upp-p0` on `nisaral/Soro-Boro`. Weights are gitignored under `releases/**/*.safetensors`.

Fair tables (same seed-42 slices) live in the root `README.md`. Do not mix the unshuffled 1920-row 0.8B number (80.99%) with the 200-row 4B slice (89%) in one unlabeled row. BFCL numbers are **converted tool-name accuracy**, not an official Gorilla leaderboard submit.

---

## 8. Kaggle accounts and kernels

| Account | Use |
|---|---|
| **jackobperalta** | Current GPU. S1 kernel lives here. |
| **keyushnisar** | Older GPU (quota low). Smoke, 1k, S0 dump. |

Kernels:

- `keyushnisar/soro-upp-p0-smoke` COMPLETE
- `keyushnisar/soro-upp-p0-1k` COMPLETE
- `keyushnisar/soro-s0-dump-a` COMPLETE (v5)
- `jackobperalta/soro-s1-a` COMPLETE
- `keyushnisar/soro-omni-data-pass2` CANCELLED (12h)

Dataset `keyushnisar/soro-omni-eval-splits` exists but **did not attach** to kernels; eval jsonl is in the git clone instead.

**Hard rules (owner was angry twice):** (1) Local ~100-step mock/smoke with loss moving before a long GPU job. (2) Time steps; jobs ≪ 12h (aim ≤ 8h). (3) Only runs that decide something. (4) 2 concurrent GPU sessions / 12h / weekly quota.

HF Hub 429 killed dump v4. Retry Hub downloads. `!python` does not fail a Kaggle cell on CUDA abort — use `subprocess.run` + `raise SystemExit(rc)`.

---

## 9. What the next agent should do

1. **S1 follow-up (if owner wants the flip gate):** stronger order invariance on A. Do not rerun FX/L0. Keep 0.6B, ≤1000 steps, one T4. Gate: flips &lt; 0.01 **and** sel/zs not below 0.875 / 0.933.
2. **If owner drops the 0.01 flip gate:** lock S1 as “accuracy ok, flips unchanged,” then **S2** (four action cards always in the menu, exact expected cost).
3. **S3 data, no GPU:** shard TTS from `audio/tts_remaining.jsonl`; Apache OS-Atlas → `image/ui_os_atlas.jsonl`; rebuild mix.
4. **Do not** start Gemma 4 E2B Omni train, 12h grids, or GRPO.
5. Commit on `upp-p0` with a real message. Trailer from the earlier Claude sessions is optional: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

---

## 10. Key files

- Plan notes: `docs/S0_PLAN_V4.md`, this file, `docs/OMNI_PLAN.md`, `docs/OMNI_DATA.md`
- UPP: `veyra/python/src/soro/routing/upp.py`
- Ablation / S1: `veyra/training/p0_upp_ablation.py` (`--s1`)
- Error dump: `veyra/training/dump_a_errors.py`
- Notebooks: `veyra/training/kaggle_p0_smoke.ipynb`, `kaggle_p0_1k.ipynb`, `kaggle_s0_dump.ipynb`, `kaggle_s1_a.ipynb`
- Omni mix: `veyra/omni_data/omni_mix.json`, `veyra/omni_data/curate_from_disk.py`
- Tests: `tests/test_upp.py` (16), `tests/test_omni_data.py` (audio twins must survive dedupe)

---

## 11. Closed UPP block work (do not reopen for accuracy)

P0 v2 (`soro-upp-p0-v2`): KM/F at chance; refiner blew loss; tools never see the request. CompRank (2606.11700) motivated xattn; **FX still lost to A**. Probe tokens (plan B) were never implemented. Gemma 4 E2B media path remains in `upp.py` / `media.py` for S3 later, not for S1.
