# Results Ledger (`Soro-Kev-Omni` / UPP)

Single source of truth for all P0, S1, S2, and S3 training runs and matched-row baseline comparisons.

Rules:
- Every run gets one row recording exact `config`, `seed`, `commit`, `steps`, `eval_dtype`, and every evaluated split.
- All S1-lock and S2/S3 decisions use full-split **`fp32`** inference with **5 random menu orders** and paired McNemar tests on identical rows.
- Final paper ladder configs get **3 seeds** (`42, 43, 44`); single-seed numbers are marked `1 seed`.
- Every percentage includes exact counts (`k/N`) and 95% Wilson score intervals (`[lo, hi]`).

---

## 1. Training Runs Ledger

### P0 & S1 Exploration (`n=120` dev / legacy runs)

| Run ID | Stage | Config | Backbone | Seed | Commit | Steps | Eval dtype | `selection` (`n=120`) | `zeroshot` (`n=120`) | `flips` | Notes |
| :-- | :-- | :-- | :-- | --: | :-- | --: | :-- | --: | --: | --: | :-- |
| `p0_1k_A` | P0 | `query_first+dot` | `Qwen3-0.6B-Base` | `42` | `76e48f6` | `1000` | `fp16` | `0.858` (`103/120`, ref `0.750`) | `0.900` (`108/120`, ref `0.417`) | `5.0%` / `10.0%` | Query-first readout wins (`~2.7s/step` on T4) |
| `p0_1k_L0` | P0 | `block+dot+gist4+late` | `Qwen3-0.6B-Base` | `42` | `76e48f6` | `1000` | `fp16` | `0.342` (`41/120`) | `0.408` (`49/120`) | `1.7%` / `0.8%` | Rejected (options blocked from query) |
| `p0_1k_FX` | P0 | `block+dot+xattn` | `Qwen3-0.6B-Base` | `42` | `76e48f6` | `1000` | `fp16` | `0.250` (`30/120`) | `0.367` (`44/120`) | `0.8%` / `0.8%` | Rejected (cross-attn collapse on 1k steps) |
| `s1_v1` | S1 | `query_first+dot` + shuffle KL + curriculum 4–64 | `Qwen3-0.6B-Base` | `42` | `ac98314` | `1000` | `fp16` | `0.875` (`105/120`, ref `0.750`) | `0.933` (`112/120`, ref `0.583`) | `5.0%` / `5.0%` | No checkpoint saved (`n=120` only) |
| `s1_v2_n120` | S1 | `query_first+dot+noidx` (no opt idx + sym JSD) | `Qwen3-0.6B-Base` | `42` | `0b05f08` | `1000` | `fp16` | `0.858` (`103/120`, ref `0.813`) | `0.908` (`109/120`, ref `0.667`) | `3.3%` / `3.3%` | Saved `ckpt_A`; superseded by full `fp32` eval |
| `s1_v3b_n120` | S1 | `query_first+dot+refiner+noidx+iso` | `Qwen3-0.6B-Base` | `42` | `37c4bf3` | `1000` | `fp16` | `0.850` (`102/120`, `flip=1/120`) | `0.917` (`110/120`, `flip=0/120`) | `0.8%` / `0.0%` | Saved `ckpt_A`; `1` flip on `n=120` was `fp16` quantization noise (`0` in `fp32`) |

### S1 Full-Split `fp32` Lock Ladder (5 random orders per row, `n=5,207` total; `1` seed `42`)

| Run ID | Config | Commit | `selection` (`1,330`) | `zeroshot` (`2,172`) | `external` (`1,200`) | `collisions` (`421`) | `near_miss` (`84`) | `BFCL v3` (`100`) | 5-Ord Flips (All `5,207`) | Status |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| `s1_v2_fp32` | `query_first+dot+noidx` | `0b05f08` | `0.8902` (`1184/1330`) `[0.872, 0.906]` | `0.8936` (`1941/2172`) `[0.880, 0.906]` | `0.8383` (`1006/1200`) `[0.816, 0.858]` | `0.7340` (`309/421`) `[0.690, 0.774]` | `0.2857` (`24/84`) | — | `8.12%` (`423/5207`) | Superseded (`28.74%` flips on collisions) |
| `s1_v3_fp32` | `query_first+dot+noidx+iso` | `e684932` | `0.8774` (`1167/1330`) `[0.859, 0.894]` | `0.8794` (`1910/2172`) `[0.865, 0.892]` | `0.8525` (`1023/1200`) `[0.831, 0.871]` | `0.6865` (`289/421`) `[0.641, 0.729]` | `0.2976` (`25/84`) | — | **`0.00%` (`0/5207`)** | Failed `v3` collision gate (`-4.75` pts vs `v2`) |
| `s1_v3b_fp32` | `query_first+dot+refiner+noidx+iso` | `37c4bf3` | `0.8744` (`1163/1330`) `[0.856, 0.891]` | `0.8831` (`1918/2172`) `[0.869, 0.896]` | **`0.8600` (`1032/1200`) `[0.839, 0.878]`** | `0.7031` (`296/421`) `[0.658, 0.745]` | `0.2857` (`24/84`) | **`0.840` (`84/100`) `[0.756, 0.899]`** | **`0.00%` (`0/5207`)** | **LOCKED S1 checkpoint (`ckpt_A`)** |

Paired McNemar tests (`fp32`, identical rows):
- **`v2_fp32` -> `v3b_fp32`:** `external`: **`+2.17 pts`** (`n10=13, n01=39, p=0.00041`); `collisions`: `-3.09 pts` (`n10=48, n01=35, p=0.1875`, not significant); `overall_excl_nearmiss` (`5,123`): `-0.61 pts` (`n10=195, n01=164, p=0.1132`); **order flips drop from `423/5,207 (8.12%)` (`121/421 = 28.74%` on collisions) to `0/5,207 (0.00%)`**.
- **`v3_fp32` -> `v3b_fp32`:** `collisions`: `+1.66 pts` (`n10=22, n01=29, p=0.401`); `external`: `+0.75 pts` (`n10=25, n01=34, p=0.298`); `zeroshot`: `+0.37 pts` (`p=0.523`); `overall_excl_nearmiss`: `+0.39 pts` (`n10=141, n01=161, p=0.274`); `0` order flips on both.

Additional `fp32` metrics (`v2` / `v3` / `v3b`):
- Refusal accuracy: `selection` (`128` rows): `0.773 / 0.805 / 0.773`; `zeroshot` (`209` rows): `0.766 / 0.727 / 0.737`; `external` (`135` rows): `0.793 / 0.837 / 0.830`.
- Error-AUROC (`u = 1 - max_p`): `selection`: `0.922 / 0.930 / 0.948`; `zeroshot`: `0.913 / 0.907 / 0.924`; `external`: `0.934 / 0.924 / 0.930`; `collisions`: `0.838 / 0.878 / 0.914`.
- LAC conformal (`alpha=0.10`): `v3b_fp32` gives `1.08 @ 0.905` (`selection`), `1.05 @ 0.900` (`zeroshot`), `1.14 @ 0.905` (`external`), `1.83 @ 0.899` (`collisions`).

### S2 & S3 Ladder (4 learned actions: `tool / answer / ask / abstain`; `ESCALATE` = inference rule)

| Run ID | Stage | Config | Seed | Commit | Steps | `s2_selection` | `When2Call MCQ` (`3,394`) | `AgentAbstain` (`301`) | `missing_crit_param` | `external` (`1,200`) | `collisions` (`421`) | `escalate_probe` (`107`) | Status |
| :-- | :-- | :-- | --: | :-- | --: | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| `s2_baseline` | S2 | `v3b_fp32 + abstain τ + escalate rule` (no training) | `—` | pending | `0` | pending | pending | pending | pending | pending | pending | pending | Queued on Kaggle |
| `s2_run1` | S2 | `4 actions + filtered s2_train + hard negatives` (`from v3b`) | `42` | pending | `1000` | pending | pending | pending | pending | pending | pending | pending | After baseline & filter |

---

## 2. Matched-Row Baseline Comparison (`s1_v3b_fp32` vs `benchmarks/predictions/`)

Every comparison below is computed on **strictly identical row IDs**.

### A. `eval_external` (`n=1,200`: `1,065` tool calls + `135` abstain rows)

| System | Params / Kind | Overall (`n=1,200`) | 95% Wilson CI | Tool (`n=1,065`) | Abstain (`n=135`) | Conformal (`α=0.10`) Size @ Cov | Median Latency (ms) | Flips (5-ord) |
| :-- | :-- | --: | :-- | --: | --: | --: | --: | --: |
| `option_logits` | `Qwen3.5-0.8B-Base` (untrained) | `0.2075` (`249/1200`) | `[0.186, 0.231]` | `0.2338` (`249/1065`) | `0.0000` (`0/135`) | `5.06 @ 90.4%` | `781.6` | — |
| `cross_encoder` | `DeBERTa-v3-small` NLI | `0.2400` (`288/1200`) | `[0.217, 0.265]` | `0.1756` (`187/1065`) | `0.7481` (`101/135`) | `4.83 @ 90.3%` | `307.1` | — |
| `bm25` | Lexical + fixed abstain τ | `0.5867` (`704/1200`) | `[0.559, 0.614]` | `0.6272` (`668/1065`) | `0.2667` (`36/135`) | `4.32 @ 84.1%` | `0.1` | — |
| `dense_cosine` | MiniLM-L6 + fixed τ | `0.7917` (`950/1200`) | `[0.768, 0.814]` | `0.8451` (`900/1065`) | `0.3704` (`50/135`) | `1.54 @ 90.4%` | `33.0` | — |
| `v2_fp32` | S1-UPP `0.6B` (joint menu) | `0.8383` (`1006/1200`) | `[0.816, 0.858]` | `0.8441` (`899/1065`) | `0.7926` (`107/135`) | `1.26 @ 90.2%` | — | `85/1200` (`7.1%`) |
| `v3_fp32` | S1-UPP `0.6B` (isolated) | `0.8525` (`1023/1200`) | `[0.831, 0.871]` | `0.8545` (`910/1065`) | `0.8370` (`113/135`) | `1.20 @ 90.2%` | — | `0/1200` (`0.0%`) |
| `soro_0.8b` | Soro-Kev `0.8B` SFT | `0.8567` (`1028/1200`) | `[0.836, 0.875]` | `0.8695` (`926/1065`) | `0.7556` (`102/135`) | `1.14 @ 90.3%` | — | — |
| **`v3b_fp32` (Ours)** | **S1-UPP `0.6B` (iso + refiner)** | **`0.8600` (`1032/1200`)** | **`[0.839, 0.878]`** | `0.8638` (`920/1065`) | **`0.8296` (`112/135`)** | **`1.13 @ 90.2%`** | — | **`0/1200` (`0.0%`)** |
| `soro_4b` (ceiling) | Soro-Kev `4B` SFT | `0.9058` (`1087/1200`) | `[0.888, 0.921]` | `0.9052` (`964/1065`) | `0.9111` (`123/135`) | `1.01 @ 91.0%` | — | — |

### B. `BFCL v3` (`n=100`: `34` Multiple + `33` Live Multiple + `33` Irrelevance)

| System | Overall (`n=100`) | 95% Wilson CI | Multiple (`34`) | Live (`33`) | Irrel (`33`) | Median Latency (ms) |
| :-- | --: | :-- | --: | --: | --: | --: |
| `option_logits` | `0.260` (`26/100`) | `[0.184, 0.354]` | `15/34` | `11/33` | `0/33` | `456.8` |
| `cross_encoder` | `0.400` (`40/100`) | `[0.309, 0.498]` | `19/34` | `10/33` | `11/33` | `317.0` |
| `arch_router` (`1.5B`) | `0.410` (`41/100`) | `[0.319, 0.508]` | `4/34` | `5/33` | `32/33` | `8465.4` |
| `bm25` | `0.630` (`63/100`) | `[0.532, 0.718]` | `32/34` | `19/33` | `12/33` | `0.1` |
| `xlam_1b` | `0.650` (`65/100`) | `[0.553, 0.736]` | `27/34` | `28/33` | `10/33` | `11453.6` |
| `dense_cosine` | `0.690` (`69/100`) | `[0.594, 0.772]` | `33/34` | `28/33` | `8/33` | `19.6` |
| `hammer_1.5b` | `0.730` (`73/100`) | `[0.636, 0.807]` | `33/34` | `31/33` | `9/33` | `10564.5` |
| `soro_0.8b` | `0.830` (`83/100`) | `[0.745, 0.891]` | `34/34` | `32/33` | `17/33` | `124.2` |
| **`v3b_fp32` (Ours)** | **`0.840` (`84/100`)** | **`[0.756, 0.899]`** | **`34/34`** | **`33/33`** | **`17/33`** | `760.8` (T4, `fp32`, 1st call +5-ord) |
| `soro_4b` (ceiling) | `0.910` (`91/100`) | `[0.838, 0.952]` | `34/34` | `32/33` | `25/33` | `648.6` |

### C. `eval_zeroshot` (`n=2,172` aligned rows) & `zeroshot/collision` (`n=52` aligned rows)

| System | `zeroshot` (`n=2,172`) | 95% CI | `zeroshot/collision` (`n=52`) | 95% CI |
| :-- | --: | :-- | --: | :-- |
| `cross_encoder` | `0.1390` (`302/2172`) | `[0.125, 0.154]` | `0.0769` (`4/52`) | `[0.030, 0.182]` |
| `option_logits` | `0.1750` (`380/2172`) | `[0.160, 0.192]` | `0.1731` (`9/52`) | `[0.094, 0.297]` |
| `arch_router` (`1.5B`) | — | — | `0.2500` (`13/52`) | `[0.152, 0.382]` |
| `bm25` | `0.4936` (`1072/2172`) | `[0.473, 0.515]` | `0.1538` (`8/52`) | `[0.080, 0.275]` |
| `dense_cosine` | `0.6381` (`1386/2172`) | `[0.618, 0.658]` | `0.4808` (`25/52`) | `[0.351, 0.613]` |
| `hammer_1.5b` | — | — | `0.5962` (`31/52`) | `[0.461, 0.718]` |
| `xlam_1b` | — | — | `0.6923` (`36/52`) | `[0.557, 0.801]` |
| `soro_0.8b` | `0.8324` (`1808/2172`) | `[0.816, 0.848]` | `0.7500` (`39/52`) | `[0.618, 0.848]` |
| **`v3b_fp32` (Ours)** | **`0.8831` (`1918/2172`)** | **`[0.869, 0.896]`** | `0.7692` (`40/52`) | `[0.639, 0.863]` |
| `soro_4b` (ceiling) | `0.8964` (`1947/2172`) | `[0.883, 0.909]` | `0.7885` (`41/52`) | `[0.660, 0.878]` |

---

## 3. Owed Integrity & Audit Checks (No GPU)

### A. Gold-Position Check (`selection`, `zeroshot`, `external`, `collisions`)
- All 5 evaluation menu permutations in `eval_upp_full.py` (`seed + order_idx`) distribute the gold tool uniformly across valid menu slots `0..len(menu)-1` (trailing drop-off at positions `5..7` reflects menu lengths `4..8`, not position bias).
- `v3b_fp32` produces identical predictions under every permutation (`0/5,207` flips), confirming zero positional leakage:
  - `external` (`n=1,200`) accuracy by position `0..6`: `0.868`, `0.861`, `0.828`, `0.870`, `0.836`, `0.893`, `0.922`; predicted position counts `222, 211, 191, 197, 199, 129, 51` vs ground-truth `228, 202, 198, 192, 207, 122, 51`.
  - `zeroshot` (`n=2,172`) accuracy across `0..7`: `0.855`–`0.913` (`0.900` on `n=10` length-8 slot).
  - `collisions` (`n=421`) has zero flips in `v3b` (`0/421`) vs `121/421 (28.74%)` flips in `v2`.

### B. `near_miss` 84-Row Bucket Audit (Moved out of Tool-Choice Scoring)
Bucketed all `84` rows in the S1 `near_miss` split (`v2` & `v3b` on identical rows):
- **`should be ABSTAIN or ASK` (`55/84 = 65.5%`):**
  - `41/84` true abstain/ask cases where `v2` (`10/41`) and `v3b` (`11/41`) mostly picked the lookalike distractor (e.g., `encode_base64` removed and `decode_base64` kept, `validate_node` removed and `get_node` kept, `moveItem` removed and `Get_History_Item` kept).
  - `14/84` true abstain cases where models already abstain (`v2: 13/14`, `v3b: 12/14`).
- **`wrong gold label` (`29/84 = 34.5%`):**
  - `26/84`: removed gold was a delexicalized alias (`op_XXXX`) of an identical lookalike still in the menu.
  - `3/84`: removed gold had a valid semantic synonym still present (`list_issues` vs `search_issues`).
- **Action taken:** Purged the `29` wrong-gold twin rows in `stage_s2.py`; kept only the clean **`55`** `near_miss` rows in `eval_near_miss_s2.jsonl` as an **S2 abstain probe** (`__abstain__`), never scored as tool selection.

---

## 4. Drex (`arXiv:2611.07854`) Reference Numbers (For `G1`–`G3` Positioning)

Published baselines from Dhankhar et al. (Nov 2026) — full analysis in [`docs/DREX_DELTA.md`](DREX_DELTA.md). Pending live `nace-ai/drex-v1.5` (`9B`, 4-bit T4) run on our exact `eval_external_s2`, `When2Call MCQ`, and `6–20` option slices.

| Model | Params (Active/Total) | `When2Call` (DI `3,652` / AA) | Decision Bench v4 (`1,071`) | DecideBench Pair (`200`) | Perm Flips (All `1,591` / `K>=6` `173`) | ECE (`6–20` opt / Tools) | Speech / Conformal / Outcome Cards |
| :--- | :--- | --: | --: | --: | --: | --: | :--- |
| `Drex v1.5` | `9B / 9B` (Open) | `89.7` / `84.8` | `86.8%` | `80.5%` | `5.3%` / `15.6%` (`8.4%` rev) | `13.6%` / `10.1%` | None |
| `Drex v2.0` | `3B / 35B` (Open) | `86.5` / `86.2` | `87.9%` | `93.5%` | `5.9%` / `16.2%` (`8.0%` rev) | `2.0%` / `5.0%` | None |
| `Drex Max` | `27B / 27B` | `85.2` / `94.0` | `94.0%` | `97.0%` | `4.3%` / `13.3%` (`6.2%` rev) | `8.4%` / `9.2%` | Vision (`Max` only); no speech / CP / cards |
| `S1-UPP v3b` (Ours) | **`0.6B`** | pending S2 eval | planned | planned | **`0.00%` (`0/5207`) / `0.00%` (`K>=6`)** | Conformal (`G2`) | `G1`–`G5` in execution (`docs/NORTH_STAR.md`) |
