# NORTH STAR: Decision Models with Guarantees (`G1`–`G5`)

**Status:** Active Execution Blueprint (read at the start of every session).
**Repo / Branch:** `nisaral/Soro-Boro` (`upp-p0`) · **Related Docs:** [`docs/DREX_DELTA.md`](DREX_DELTA.md), [`docs/RESULTS_LEDGER.md`](RESULTS_LEDGER.md).

---

## 0. Core Thesis & Competitive Framing (Move Fast)

Drex (`arXiv:2611.07854`, 9B–35B, 1.15M records) + Kev/Jev own **"plain decision models"**. Competing on general-knowledge scale with a small model is a losing battle. Simultaneously, October 2026 preprints (`2610.01601`, `2610.18233`) already published standalone option isolation / 2-pass runoff.

**Our distinct defensible claim:**
> **Decision Models with Guarantees (`0.6B–4B`, single forward pass, edge/production speed)** — Drex gives an empirical probability with 4.3–5.9% flips (13–16% at `K>=6`), 8–14% calibration error on multi-option/tool menus, and no speech or test-time learning. We give **formal guarantees** across five axes (`G1`–`G5`):

| Guarantee | Claim | What Drex / `2610.01601` Lack | Target Metrics & Benchmarks |
| :--- | :--- | :--- | :--- |
| **`G1` Exact Order Invariance + Interaction** | Single-pass **0.00% (`0/N`) flips** under any option permutation AND dense collision discrimination via a **1–2 layer permutation-equivariant head refiner (`v3b`)**. | Drex: `4.3–5.9%` flips (`13–16%` for `K>=6`). `2610.01601`: isolated-only (loses `-4.75 pts` on our 421 tool collisions). | `v2` vs `v3` vs `v3b` (`fp32`, 5 orders), **3 seeds (`42,43,44`)**, **menus of 6–20 options**, refiner depth `1` vs `2`, `0.6B` vs `1.7B/4B` scaling, DecideBench v1.1 (`400`), Decision Bench v4. |
| **`G2` Coverage Guarantee** | Finite-sample **split/Mondrian conformal prediction sets** per split and **per modality**, bounding miscoverage `<= alpha` even on hard multi-option tool menus. | Drex: global temp `T`; **`8.4%` ECE on 6–20 options** (Max) / **`13.6%`** (v1.5), `9.2% / 10.1%` Tools ECE, **no coverage guarantee**. | Report empirical **coverage**, **mean set size**, and **ECE on `K in [6, 20]`** across all splits (`alpha = 0.05, 0.10`) and modalities (`text, speech, screen`). |
| **`G3` Set-as-Action (`ABSTAIN / ACT / ASK`)** | Conformal set size defines action without extra ad-hoc heads: `|S| = 0 -> abstain`, `|S| = 1 -> act`, `|S| >= 2 -> ask` (or **expected-cost 4-class head** vs **set-as-action** comparison). | Drex: `"none"` option / confidence threshold only; zero `ASK` or cost-aware action training. | Compare `set-as-action` vs trained `4-class` expected-cost S2 head (`tool, answer, ask, abstain` + `ESCALATE` inference rule) on **`When2Call MCQ`** (Drex `v1.5 = 89.7` DI / `84.8` AA; `Max = 85.2` DI / `94.0` AA), **`AgentAbstain`** (`301`, incl. `missing_critical_parameter`), **`external_s2` (`1,200`)**, **`collisions_s2` (`421`)**, **`escalate_probe` (`107`)**. |
| **`G4` Speech & Bounded-Error Early Commit** | Raw-speech twins + streaming prefix training + **conformal early-commit** gating: commit mid-utterance when prefix set/margin meets risk guarantee (`<= eps` stale-execution rate vs full utterance). | Drex: **zero speech**. `RASSG (2610.14659)` & `Stream RAG (2510.02044)` rely on ASR heuristics with no coverage bound. | **Speech twins** (`s0..s7` TTS shards + real speech eval `SLURP` / `Speech-MASSIVE`), **Whisper->text cascade** baseline, **early-commit error vs latency Pareto** vs RASSG-style `LocalAgreement-k` gating. |
| **`G5` Stretch: Test-Time Outcome Cards** | Append **live execution outcome cards** (success/failure rates, latency, schema deprecation) to option text at inference time -> accuracy & abstention adapt **without retraining**. | Drex: static context, cannot learn from live tool failures or added/removed tools at test time. | Non-stationary **stream eval** where tools are **added, removed, and injected with faults**; plot accuracy vs # of outcome cards (`0, 5, 10, ...`, with vs without cards). Compare vs bandit/experience baselines (`BaRP`, CVPR 2026). |

---

## 1. Publication & Preprint Strategy (Speed vs Completeness)

1. **Fast Preprint (Jan 2027, `G1 + G2 + G3`):** Lock **"Order-Invariant Decision Models with Coverage & Action Guarantees"** first while `v3b` + conformal `6–20` options + `When2Call`/`AgentAbstain` results are fresh (citing and beating `2610.01601` single-pass isolation, `2610.18233` 2-pass Runoff, and `Drex 2611.07854`).
2. **Full Conference Paper (NeurIPS 2027, `G1 + G2 + G3 + G4 + G5`):** Full ladder (`v2 / v3 / v3b / refiner 1 vs 2 / 0.6B vs 4B`, `+ expected cost`, `+ speech twins + early-commit guarantee`, `+ outcome cards stream`, 3 seeds on all final rows, plus **Drex v1.5 (`9B`, 4-bit T4) live matched-row baseline**).

---

## 2. Immediate Execution Queue (Strictly Ordered — Update Every Run)

- [x] **S1 Lock & Owed Integrity Checks:** `v3b_fp32` locked (`0/5,207` flips, `0.8600` external, `84/100` BFCL v3); baseline table, gold-position check, and 84 `near_miss` buckets completed & pushed (`6b2b57f`).
- [x] **S2 Baseline + Gold-100 Judge + `+3pt` Soft-Label Gate:** Complete (`0/6,050` flips across 5 orders; G2 coverage 0.948–0.961; G3 Set-as-Action +13.5 pts; gate failed with teacher 77.0% vs v3b 98.5%, soft labels locked OUT).
- [x] **Owner Hand-Check ([`s2_gold_100_review.md`](file:///c:/Users/nisar/OneDrive/Desktop/EB-JEPA/veyra/runs/omni_data/s2/s2_gold_100_review.md)):** Delivered with 31 disagreement rows, rule vs teacher breakdown, and soft-label gate analysis.
- [ ] **S2 Train Filtering:** Run `soro-s2-filter-train` (`Qwen3-8B` 4-bit filter on `8,953` `s2_train` rows, zero leakage check).
- [ ] **S2 Run 1 (4 actions, expected cost, from `v3b`) + `G2`/`G3` Eval:** Train on Kaggle + `fp32` 5-order eval + **test `set-as-action` (`|S|=0/1/>=2`) vs 4-class head** + **Drex v1.5 4-bit baseline** on `eval_external_s2`, `When2Call MCQ`, `AgentAbstain`, `6–20` opt calibration & flips (`docs/RESULTS_LEDGER.md`).
- [ ] **G1 Scaling & Depth Ablations (before preprint):** 3 seeds (`42, 43, 44`), `refiner-layers 1 vs 2`, `K in [6, 20]` slice, and `Decision Bench v4` + `DecideBench v1.1`.
- [ ] **G4 Speech & Streaming Early-Commit:** Collect TTS shards `s0..s7` + download `SLURP` real-speech eval -> Whisper cascade baseline -> S3 multimodal + prefix early-commit conformal bound vs `RASSG`.
- [ ] **G5 Outcome Cards Stream Eval:** Build synthetic+real non-stationary tool stream and measure no-retrain accuracy recovery.

