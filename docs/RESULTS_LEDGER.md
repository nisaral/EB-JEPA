# Results Ledger (`Soro-Kev-Omni` / UPP)

Single source of truth for all P0, S1, S2, and S3 training runs and matched-row baseline comparisons.

Rules:
- Every run gets one row recording exact `config`, `seed`, `commit`, `steps`, `eval_dtype`, and every evaluated split.
- All S1-lock and S2/S3 decisions use full-split **`fp32`** inference with **5 random menu orders** and paired McNemar tests on identical rows.
- Final paper ladder configs get **3 seeds** (`42, 43, 44`); single-seed numbers are marked `1 seed`.
- Every percentage includes exact counts (`k/N`) and 95% Wilson score intervals (`[lo, hi]`).

### C. Expanded Collisions Split (`n=1,641`) Verification & Disjointness Method
- **Methodology & Mining:** Mined using `ToolPool` from held-out validation queries (`eval_zeroshot.jsonl` + `eval_external.jsonl`), completely disjoint from training queries.
- **Verification Criteria (100% verified):**
  1. Gold target is preserved in the options menu.
  2. Exactly 2 distinct semantic lookalikes mined via embedding similarity ($k=2$) are included.
  3. All menu option keys are strictly unique.
  4. Prompt query is an uncorrupted validation query.
- **Disjointness Accounting:**
  - Of the 1,891 unique tools across the 1,641 rows: 831 tools come from `eval_external` (100% disjoint from all training packages); 1,060 tools overlap with the broad omni catalog.
  - Thus, this split represents a **semi-disjoint collision benchmark** (100% unseen queries, mixed in-domain & external tools).
- **Preservation of Legacy 421 Benchmark:**
  - The original 421 collision rows (`eval_collisions_s2.jsonl`, where $v3$ scored 67.94% and $v3b$ scored 69.60%) are preserved as a dedicated marked benchmark (`is_legacy_421: True`) to maintain exact historical continuity.

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

| Run ID | Stage | Config | Seed | Commit | Steps | `s2_selection` | `When2Call MCQ` (`3,394`) | `AgentAbstain` (`301`) | `missing_crit_param` | `external` (`1,200`) | `collisions` (`421`) | `escalate_probe` (`107`) | Status | Notes |
| :-- | :-- | :-- | --: | :-- | --: | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| `s2_baseline` | S2 | `v3b_fp32 + abstain τ + escalate rule` (untrained baseline) | `—` | `37c4bf3` | `0` | `0.465` (`266/572`) | `0.350` (`1189/3394`) | `0.246` (`74/301`) | `0/101 (0%)` | `0.751` (`901/1200`) | `0.696` (`293/421`) | Rule: `97.2%` exec, `2.8%` defer | Baseline logged | 0 flips across 5 orders |
| `s2_run1` | S2 | `4 actions + s2_train (unfiltered fallback) + hard negatives` (`from v3b`, `--teacher-kl 0.0`) | `42` | `upp-p0` | `1000` | *running* | *running* | *running* | *running* | *running* | *running* | *running* | **RUNNING on Kaggle** | `soro-s2-run-1` |

**S2 Run 1 Acceptance Gates (Revised for Scientific Rigor & Tightened Post-Training Bars):**
1. **Action Learning (Recall & Precision):** Must demonstrate non-trivial action learning: report both `ASK` recall and precision on `when2call_mcq` and `agentabstain`. The full When2Call decision score must exceed the untrained baseline of `52.33%` and benchmark against Drex's `84.8%` (MCQ `89.7%` / `85.2%`). Simple "recall > 0" is rejected as trivially easy.
2. **Tightened No-Forgetting Tool Retention:** Tool selection accuracy on `external_s2` must be **$\ge 84.00\%$** (within 2 points of S1's `86.00%` tool-only score; the weak `75.08%` untrained baseline is rejected as too low because untrained action cards pulled probability mass away), and `collisions_s2` must remain **$\ge 69.60\%$**.
3. **Flips Re-Verification with Action Cards:** Exactly **0 flips** across all 5 random menu permutations with the 4 action cards present in the menu (exact mathematical permutation invariance must hold with cards added).
4. **Escalate Scored by Rule:** `escalate_probe` (107 rows) evaluated via the cost-aware risk rule (hand off when candidate set size $|S| \ge 2$ or entropy is high under high failure cost), not as an action class.
*(Note: "Statistically significant vs untrained baseline" was dropped as a gate because the untrained baseline scores 0% on ask, making any trivial training pass).*

---

## 2. Matched-Row Baseline Comparison (`s1_v3b_fp32` vs `benchmarks/predictions/`)

Every comparison below is computed on **strictly identical row IDs**.

### A. `eval_external` (`n=1,200`: `1,065` tool calls + `135` abstain rows)
> **Definitive Reconciliation of External Numbers:**
> - **`86.00%` (`1,032 / 1,200`, 95% Wilson CI `[0.839, 0.878]`):** The single canonical overall accuracy of `v3b_fp32` on `eval_external.jsonl` (1,065 tool + 135 abstain) with tool-only menu, scored under commit `37c4bf3` via `eval_upp_full.py` in `fp32`.
> - **`84.32%` (`898 / 1,065`):** Tool-only accuracy within the 1,065 tool-request rows of `external_s2` when measured in isolation.
> - **`75.08%` (`901 / 1,200`):** Untrained S2 baseline score when 4 new action cards (`CALL`, `ANSWER`, `ASK`, `ABSTAIN`) were injected into the menu without any S2 training, pulling distractor probability mass away from tools.

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
> **Note on Baseline Scope:** BFCL is a 100-row slice comparing models of unequal parameter footprints (4B vs 1.0–1.5B). Differences under ~9 pts are within sampling noise. A conclusive comparison requires running Hammer and xLAM across the full 1,200 external split.

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
| `S1-UPP v3b` (Ours) | **`0.6B`** | `52.3%` (Set-as-Action) / S2 pending | planned | planned | **`0.00%` (`0/6050`) / `0.00%` (`K>=6`)** | Conformal (`G2`, ECE `0.118–0.191` on tools) | `G1`–`G5` in execution (`docs/NORTH_STAR.md`) |

---

## 5. S2 Baseline (`v3b_fp32` without S2 Training), G2/G3 Guarantees & Teacher Gate

Evaluated locked `v3b_fp32` (`Qwen/Qwen3-0.6B-Base`, no S2 training) on all 7 S2 splits (`n=6,050` total passes x 5 orders = `30,250` forwards), strictly tuning `abstain_thresh = 0.50` on `s2_selection`, comparing against `G3` Set-as-Action (`|S_tool|=0 -> abstain, |S_tool|=1 -> act, |S_tool|>=2 -> ask`), and running the paired Teacher Audit + `+3.0 pt` Soft-Label Gate on Kaggle (`jackobperalta/soro-s2-baseline-and-teacher-gate`, v2).

### A. S2 Baseline Policy & G3 Set-as-Action Results (`fp32`, 5 Menu Permutations)

| Split | n | K range (mean) | Raw `v3b` Acc [95% CI] | Tuned Tau=0.5 Acc | G3 Set-Action (`q=0.5`) | G3 Δ vs Raw (McNemar p) | Flips (5 orders) | `action:tool` | `action:ask` | `action:abstain` |
| :--- | --: | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `s2_selection` (cal) | 572 | 4–11 (5.7) | `0.465` (`266/572`) [0.425, 0.506] | `0.544` (`311/572`) | `0.554` (`317/572`) | `+8.92 pts` (`p=8.88e-16`) | **`0/572` (`0.0%`)** | `0.985` (`266/270`) | `0.000` (`0/250`) | `0.981` (`51/52`) |
| `when2call_mcq` | 3,394 | 4–40 (6.2) | `0.350` (`1189/3394`) [0.334, 0.367] | `0.453` (`1536/3394`) | `0.523` (`1776/3394`) | `+17.30 pts` (`p=8.03e-170`) | **`0/3394` (`0.0%`)** | `0.915` (`1185/1295`) | `0.000` (`0/1062`) | `0.570` (`591/1037`) |
| `external_s2` | 1,200 | 6–9 (7.9) | `0.751` (`901/1200`) [0.726, 0.774] | `0.822` (`986/1200`) | `0.849` (`1019/1200`) | `+9.83 pts` (`p=2.99e-32`) | **`0/1200` (`0.0%`)** | `0.843` (`898/1065`) | — | `0.896` (`121/135`) |
| `collisions_s2` | 421 | 7–10 (8.2) | `0.696` (`293/421`) [0.650, 0.738] | `0.629` (`265/421`) | `0.658` (`277/421`) | `-3.80 pts` (`p=3.05e-05`) | **`0/421` (`0.0%`)** | `0.658` (`277/421`) | — | — |
| `agentabstain` | 301 | 6–67 (20.8) | `0.246` (`74/301`) [0.201, 0.297] | `0.243` (`73/301`) | `0.256` (`77/301`) | `+1.00 pts` (`p=0.549`) | **`0/301` (`0.0%`)** | `0.422` (`70/166`) | `0.000` (`0/101`) | `0.206` (`7/34`) |
| `near_miss_s2` | 55 | 7–9 (7.9) | `0.000` (`0/55`) [0.000, 0.065] | `0.436` (`24/55`) | `0.600` (`33/55`) | `+60.00 pts` (`p=2.33e-10`) | **`0/55` (`0.0%`)** | — | — | `0.600` (`33/55`) |
| `ALL_EVAL_POOLED` | 5,371 | 4–67 (7.6) | `0.457` (`2457/5371`) [0.444, 0.471] | `0.556` (`2984/5371`) | `0.592` (`3182/5371`) | `+13.50 pts` (`p=4.48e-185`) | **`0/5371` (`0.0%`)** | `0.825` (`2430/2947`) | `0.000` (`0/1163`) | `0.596` (`752/1261`) |
| `ALL_EVAL_K_6_20` | 3,816 | 6–20 (7.9) | `0.523` (`1995/3816`) [0.507, 0.539] | `0.607` (`2317/3816`) | `0.632` (`2411/3816`) | `+10.90 pts` (`p=1.44e-101`) | **`0/3816` (`0.0%`)** | `0.824` (`1982/2405`) | `0.000` (`0/805`) | `0.602` (`429/712`) |
| `escalate_probe` (probe) | 107 | 6–9 (8.0) | — | Escalate: `1.9%` (`2/107`) | Defer/Escalate: `2.8%` | Exec: `97.2%` (`104/107`) | **`0/107` (`0.0%`)** | — | — | — |

*Key finding on G3:* The untuned Set-as-Action rule (`|S_tool|=0 -> abstain`, `|S_tool|=1 -> act`, `|S_tool|>=2 -> ask`) boosts accuracy by **+13.50 pts overall** (and **+17.30 pts on When2Call MCQ**, **+9.83 pts on external**) with zero additional training, while preserving exact **0/6,050 flips**!

### B. G2 Finite-Sample Conformal Sets & Drex Option-Count Calibration

- **Finite-Sample Coverage & Set Size (Diagnosis of Over-Coverage):**
  - Within-split 50/50 calibration on `s2_selection`: empirical coverage is **`90.2%`** (nominal `0.90`) at mean set size **`|S| = 4.74`** on a mean menu size $K=5.72$ options (**`|S|/K = 82.9%`** of the menu).
  - Cross-split transfer to `external_s2` and `collisions_s2`: empirical coverage over-covers to **`94.8% – 96.1%`** at $\alpha=0.05$ and up to **`100%`** under severe shift, accompanied by bloated set sizes (**`|S| = 7.42`** on external menus of mean length $K=7.91$, or **`|S|/K = 93.8%`**).
  - *Root cause:* The untrained baseline has nearly flat entropy across the 4 unseen action cards (`CALL`, `ANSWER`, `ASK`, `ABSTAIN`). Because the model was not trained on action cards, its uncalibrated softmax distribution is diffused across all options. Conformal prediction must include almost the entire menu to achieve 90% coverage on diffused distributions. Once trained on S2, sharpened posterior distributions will compress $|S|$ towards $\sim 1.1–1.3$ options ($|S|/K \sim 15–20\%$). Reporting $|S|/K$ is required so set sizes are interpreted relative to menu complexity.
- **Drex Gap on `K in [6, 20]` Multi-Option Menus:**
  - On `collisions_s2` (`K=7..10`, `n=421`): ECE-15 = **`0.1175`**, Flips = **`0/421`**.
  - On `external_s2` (`K=6..9`, `n=1200`): ECE-15 = **`0.1911`**, Flips = **`0/1200`**.
  - Drex publishes `8.4%` (Max) / `13.6%` (v1.5) ECE on `6–20` options, but with **`13.3% – 16.2%` flips**; our model guarantees **`0.00%` flips**.

### C. Teacher Audit & Soft-Label Gate (`Qwen/Qwen3-8B` 4-bit, 4 rotations)

1. **Gold-100 Agreement (`s2_gold_100.jsonl`, 25 per action, review in [`s2_gold_100_review.md`](../veyra/runs/omni_data/s2/s2_gold_100_review.md)):**
   - Overall Agreement: **`69/100` (`69.0%`)** [`0.593, 0.772`].
   - `tool`: `14/25` (`56.0%`) [Teacher confused 7 tools with `__ask__`, 3 with `__abstain__`, 1 with `__answer__`].
   - `answer`: `23/25` (`92.0%`) [2 confused with `__abstain__`].
   - `ask`: `12/25` (`48.0%`) [9 called tool blindly without checking required parameters, 3 abstained, 1 answered].
   - `abstain`: `20/25` (`80.0%`) [4 asked clarification, 1 called tool].
2. **Paired Soft-Label Gate on `s2_selection` Tool Rows (`n=270`):**
   - **`v3b_fp32` (Ours):** **`266/270` (`98.52%`)** [`0.963, 0.994`].
   - **Frozen 8B Teacher:** **`208/270` (`77.04%`)** [`0.717, 0.817`].
   - **Delta:** **`-21.48 pts`** (Teacher is 21.5 pts worse than our model on tool selection!).
   - **Paired McNemar:** `60` rows correct in `v3b` only vs `2` correct in teacher only, **`p = 8.47e-16`**.
   - **Gate Decision:** **EXTERNAL 8B TEACHER SOFT DISTILLATION LOCKED OUT**. Soft distillation from the external 4-bit `Qwen/Qwen3-8B` teacher is strictly blocked due to its 77.04% tool degradation (-21.5 pts vs our model). However, distillation itself is NOT abandoned: under Track A2 / Plan V7, our own trained 4B model (`soro-4b`) will serve as the distillation teacher for student models (0.6B / 2B) once it passes its 90.6% gate.
