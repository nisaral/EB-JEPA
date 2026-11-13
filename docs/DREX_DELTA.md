# Competitor & Literature Delta: Drex (`arXiv:2611.07854`) & Recent Order-Invariant Decision Literature

Last updated: **2026-11-13** · Tracked against repo branch `upp-p0`.

---

## 1. Full Analysis of Drex (*Towards Foundation Decision Models*, Nace.AI, `arXiv:2611.07854v1`, 05 Nov 2026)

Read in full (including Sections 1–11 and Appendices A–P). Code and open weights: [`github.com/nace-ai/drex-decision-models`](https://github.com/nace-ai/drex-decision-models), built on [`github.com/jaredpalmer/kev`](https://github.com/jaredpalmer/kev) and following TypeSafe's System One interface (`Jev`).

### 1.1 Model Family & Training Scale (Appendix E, Table 12; Appendix D, Table 11; Appendix F, Table 13)
| Model | Params (Active/Total) | Backbone Architecture | Head & Readout | Attention / Cache | Train Data / Stages | Open Weights |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Drex DLM** | 8B / 8B | `Efficient-DLM-8B` (block-bidir diffusion) | `256`-d pointer (`<decide>`, `</opt>`) | Block-causal + `state_bidir`; shared state cache | Text SFT + RLAF | [`nace-ai/drex-dlm`](https://huggingface.co/nace-ai/drex-dlm) (CC BY-NC 4.0) |
| **Drex v1.5** | 9B / 9B | `MiMo-V2.6-Distill-Qwen-9B` (`Qwen3_5`, 24 Gated DeltaNet + 8 full-attn layers) | `256`-d pointer (`2.1M` params) | Causal row-wise `[S; b_k]` (cannot mask); shared state | `1,150,460` records (`15.7%` tools = `181,004`); SFT + RLAF | [`nace-ai/drex-v1.5`](https://huggingface.co/nace-ai/drex-v1.5) (Open RAIL-M) |
| **Drex v2.0** | 3B / 35B | `Qwen3.6-35B-A3B` (`3:1` DeltaNet MoE, 8/256 experts) | `256`-d pointer (`1.0M` params) | Causal row-wise `[S; b_k]`; shared state | SFT + specialist LoRA stage + RLAF | [`nace-ai/drex-v2.0`](https://huggingface.co/nace-ai/drex-v2.0) (Open RAIL-M) |
| **Drex Max** | 27B / 27B | `Qwen3.8-27B` (`3:1` DeltaNet + ViT, 48 GDN + 16 full-attn layers) | `255`-way readout (`<ans>`, `1.3M` params, symbol-bound `A..Z, AA..`) | Non-causal full-attention layers (`pplx-decider` style); **no** shared state cache | `1.15M` SFT (9k steps) + `180k` RLAF + `95k` vision SFT | Gated / API (`drex-max`) |

### 1.2 Exact Published Numbers (For Direct Comparison)
- **Decision Index 0.4.1** (Tables 1, 24 — `37` public benchmarks, `n=140,178` requests):
  - Overall Public Skill / Private-Inclusive Full Skill: **Drex Max** `67.52 / 66.60`, **Drex v2.0** `61.25 / 52.76`, **Drex v1.5** `58.08 / 45.78`, **Jev 1.14.0** `60.70 / 60.24`.
  - **Tools domain skill (public):** Drex Max `81.9`, Drex v2.0 `76.5`, Drex v1.5 `75.0`, Jev 1.14 `73.8`.
  - **Per-benchmark Tool numbers (Table 24):**
    - **`When2Call MCQ` (`n=3,652`):** **Drex v1.5 = `89.7`**, Drex v2.0 = `86.5`, Drex Max = `85.2`, GPT-6 Luna = `70.2`. *(Note: v1.5 actually beats Max here!)*
    - **`BFCL` (`n=1,694`, case exact):** Drex Max = `98.2`, v2.0 = `96.6`, v1.5 = `94.9`.
    - **`API-Bank` (`n=508`):** Drex Max = `88.4`, v2.0 = `83.3`, v1.5 = `85.4`.
    - **`ToolRet` (`n=685`, nDCG@10):** Drex Max = `70.6`, v2.0 = `67.3`, v1.5 = `70.2`.
  - **Fixed-seed tool subset (Table 23, `n=300` When2Call / `n=150` BFCL):** `When2Call`: Drex v1.5 `90.7`, v2.0 `87.3`, Max `87.0`, Claude Opus 5.5 `86.7`.
- **Artificial Analysis Decision Index** (Table 3, Oct 29 2026):
  - **`When2Call`:** **Drex Max = `94.0`**, Decider v1.2 = `87.7`, **Drex v2.0 = `86.2`**, **Drex v1.5 = `84.8`**, Jev 1.14 = `80.8`, Claude Opus 5.5 = `90.5`.
  - Overall AA Index: Max `63.7`, v2.0 `57.9`, v1.5 `50.7`. *(Max's +13.0 jump over v1.5 is heavily driven by AA-Omniscience non-hallucination rate `0.6% -> 71.3%`.)*
- **Decision Bench v4** (Table 2, `n=1,071`: `949` text, `62` text+img, `60` image-only):
  - **Drex Max** `94.0% [92.4, 95.3]` (ECE `1.40%`), **Drex v2.0** `87.9%` (ECE `1.73%`), **Drex v1.5** `86.8%` (ECE `4.77%`), Jev 1.14 `92.3%`, Opus 5.5 `95.6%`.
- **DecideBench v1.1** (`400` items / `200` minimal pairs, Appendix H.3): Max `97.0%` pair acc (`1.1%` ECE), v2.0 `93.5%`, v1.5 `80.5%`, Jev 1.14 `96.0%`.
- **Order Robustness** (Section 8.5, Table 5, `1,591` decisions across 37 DI benchmarks — 3 orders: orig / rev / 1 perm):
  - Permuted flip rate / Reversed flip rate / **`K >= 6` perm flip rate (`n=173`):**
    - **Drex Max:** `4.3% [3.3, 5.3]` / `6.2%` / **`13.3%`** (`17.8%` without consistency loss `l_pi, l_s`, Table 7).
    - **Drex v2.0:** `5.9% [4.8, 7.1]` / `8.0%` / **`16.2%`**.
    - **Drex v1.5:** `5.3% [4.2, 6.5]` / `8.4%` / **`15.6%`**.
    - **Jev 1.14:** `4.3% [3.3, 5.4]` / `5.7%` / **`14.5%`**.
- **Calibration** (Section 8.4, Table 4 & Appendix H.4 Tables 18–19, `~115k–230k` decisions):
  - Deployed ECE: v2.0 `1.37%`, Max `3.61%`, v1.5 `4.54%`.
  - **Severe slice miscalibration:** In **Tools** (`n=5,816`, Table 18), deployed ECE = **`9.2%` (Max), `10.1%` (v1.5), `5.0%` (v2.0)**; even after oracle-half temperature `T*`, Tools ECE stays **`7.0%` / `6.8%` / `6.0%`**.
  - **6–20 options (`6 <= K <= 20`, `n=15,155`, Table 19):** **Drex Max ECE = `8.4%`** (conf `80.1%` vs acc `71.7%`); **Drex v1.5 ECE = `13.6%`** (conf `70.5%` vs acc `56.9%`); Drex v2.0 ECE = `2.0%`.

### 1.3 What Drex Explicitly States as Untested / Structural Limitations
1. **Appendix C.2 ("Option isolation") & Section 8.5 (lines 1214–1216):**
   - Explicitly defines option isolation (`<opt>..</opt>` attends only to state, instruction, itself; all options share start position; `<decide>` attends to all options), making the pointer head permutation-equivariant by construction.
   - **States verbatim:** *"The backbones of Drex v1.5, v2.0, and Max are hybrids with recurrent Gated DeltaNet layers, which cannot apply such a mask, so we have not tested option isolation on any of our released models."*
2. **Drex Max vs Pointer Trade-off (Section 10):** Max removes the causal mask and uses a fixed `255`-code readout head, losing shared state caching across questions (`O((|S|+|b|)^2)` per question) and requiring learned symbol binding (losing `-23.9 pts` on POP909-CL and trailing v2.0/v1.5 on `When2Call MCQ`).
3. **Zero Audio / Speech:** Only text + static/sampled images (`Max`).
4. **No Conformal Sets or Finite-Sample Guarantees:** Only scalar logits/confidence + single temperature `T` (`Section 6.6` & `8.4`). Thresholds must be refitted per model and domain (`Table 18`, `Appendix M`).
5. **No Cost-Aware Multi-Action Policy (`ACT / ASK / ABSTAIN`):** `none` option inserted in a random fraction of train rows (`Section 6.2`); abstention in `Section 8.5` is only post-hoc low-confidence cutoff; no `ASK` vs `ABSTAIN` vs expected-cost trade-off.
6. **No Test-Time Adaptation Without Retraining:** Static state only; no inference-time outcome cards.

---

## 2. Public Benchmark Availability & Drex v1.5 Local Baseline Feasibility

| Artifact | Location | Public? | Contents & Format | Use for Us |
| :--- | :--- | :--- | :--- | :--- |
| **Decision Bench v4** | [`github.com/atlanai/decision-bench`](https://github.com/atlanai/decision-bench) (`data/corpus/bench-v4`) | **Yes (Open)** | `1,071` decisions across `35` tasks / `11` categories (`949` text, `62` text+img, `60` image-only) | External general decision + G2 conformal + Drex v1.5/Max comparison |
| **DecideBench v1.1** | [`github.com/choyiny/decidebench`](https://github.com/choyiny/decidebench) (`data/v1/`) | **Yes (Open)** | `400` minimal-pair contrastive decisions (`8` `.jsonl` files, incl. `agent_routing.jsonl`, `action_review.jsonl`) | Minimal-pair flip/consistency & agent routing check (`G1`/`G2`) |
| **JevBench-231** | [`github.com/fstandhartinger/jevbench`](https://github.com/fstandhartinger/jevbench) (`datasets/public`) | **Yes (Open)** | `231` typed decisions (easy `48`, std `72`, hard `111`) | Standardized reference (`Drex Max 91.8`, `v2.0 86.1`, `v1.5 84.8`) |
| **Decision Index 0.4.1** | [`hf.co/spaces/multimodalart/jev-decision-index`](https://huggingface.co/spaces/multimodalart/jev-decision-index) | **Yes (Public index)** | Aggregate over `37` public benchmarks (`data/index.json`, `methodology.json`); subset has `When2Call`, `BFCL`, `ToolRet`, `API-Bank`, `CLINC150` | Public scoring scripts & benchmark subsets (run Tools subset `When2Call`, `BFCL`, `ToolRet`, `API-Bank`) |
| **`nace-ai/drex-v1.5` (9B)** | [`hf.co/nace-ai/drex-v1.5`](https://huggingface.co/nace-ai/drex-v1.5) | **Yes (Open RAIL-M)** | `Qwen3_5ForCausalLM` (`9B`), 4 shards + `head.pt` (`2.1M`), `code/kev/model.py` | **Fits on 16GB T4 in 4-bit (`~5.5 GB`)!** Run as live baseline on `eval_external_s2`, `BFCL`, `When2Call`, `6–20` opt calibration & flips |

---

## 3. Broader Late-2026 Literature Delta (Keep in Loop Every Turn)

Crucial check from November 2026 arXiv scan: **option isolation alone is already published** by two October 2026 preprints! Our novelty MUST rest on the **full "Decision Models with Guarantees" stack (`G1`–`G5`)** and our **`v3b` single-pass equivariant refiner**.

| Paper | Venue / ID | What They Already Claim | Exact Delta / What We Offer Beyond Them |
| :--- | :--- | :--- | :--- |
| **Candidate-Independent Block-Causal Attention** (Amit) | `arXiv:2610.01601` (Oct 2026) | Candidate isolation mask + position reset on Qwen3 `1.7B/4B` and Gemma 3 `1B` | Stops at **zero cross-candidate interaction**. We proved empirically (`v3`) this drops **`-4.75 pts` on dense tool collisions** (`73.40 -> 68.65`); **`v3b`** adds a single-pass permutation-equivariant refiner recovering collisions (`+1.66 pts`) & hitting `0.8600` on `external` (`p=0.00041` vs `v2`) while keeping `0/5,207` flips (`G1`). Plus `G2`–`G5`. |
| **Consider, Then Choose: Runoff** (Huang et al.) | `arXiv:2610.18233` (Oct 2026) | Training-free 2-pass method: score each option alone to shortlist, then run cyclic orderings over finalists | Needs **two sequential stages + cyclic enumeration** (`K'` passes). **`v3b`** gives exact order invariance in **one forward pass**, zero runtime overhead, trained end-to-end with interaction + conformal guarantees (`G1`–`G3`). |
| **InvariRank** (Bito et al.) | `arXiv:2604.27599` (Apr 2026) | Cross-candidate isolation mask + shared RoPE positions for listwise LLM recommendation reranking | Recommendation ranking only; no tool-collision refiner, no conformal coverage (`G2`), no action set (`G3`), no speech (`G4`), no outcome cards (`G5`). |
| **RASSG** (Hu & Wang) | `arXiv:2610.14659` (Oct 2026) | Repair-aware semantic survival gating + active cancellation for streaming ASR tool calls (reduces ReCAST stale rate `50% -> 3.15%`) | Relies on **ASR transcripts + heuristics** without a calibrated **finite-sample conformal bound on early-commit error rate** (`G4`) over raw audio. Compare against RASSG-style agreement gating. |
| **Stream RAG / Intent Stabilization** | `arXiv:2510.02044`, `arXiv:2606.20113` | Streaming query emission during speech; shows early stabilization bound `H` on CRAG | Issues retrieval/tools speculatively with **no distribution-free error bound** per early prefix (`G4`). |
| **CAELUM / Learnable Conformal Abstention / Agent UQ** | `arXiv:2611.10738`, `arXiv:2502.06884`, `arXiv:2609.07395` | Matchability-aware CP (aleatoric vs epistemic); RL threshold optimization over LAC/APS; step vs trajectory ECE (`TC-ECE`) | Conformalize open-ended text generation or VQA. None unify **order-invariant single-pass pointer (`G1`) + per-modality conformal set-as-action (`G2`/`G3`) + streaming audio commit (`G4`)**; cite and adapt **CAELUM** epistemic-vs-aleatoric separation (`abstain` vs `ask`) and **TC-ECE** for multi-step evals. |
| **R1-Ranker / IRanker / BaRP / TFMs / Visual Tools** | `2506.21638`, `2510.07429`, `2607.13314`, CVPR 2026 (Huang) | RL iterative ranking (`IRanker`), bandit routing with preference vector (`BaRP`), TFM heterogeneity encoding, test-time visual tool selection from experience | **`G5` stretch:** compare inference-time **outcome cards** (non-parametric reliability updates when tools are added/removed/degrade) against BaRP & CVPR-2026 experience selection without gradient updates. |

