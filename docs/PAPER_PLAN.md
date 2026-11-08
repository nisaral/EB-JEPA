# Paper plan: Matryoshka Gist Pointers

Working title: **One Cache, Any Catalog: Matryoshka Gist Pointers for Calibrated, Order-Invariant Tool Selection at Scale**

Status: architecture implemented and exactness-verified (Nov 2026); P0 accuracy grid queued on Kaggle.

## Problem

Agents choose among hundreds to thousands of tools and must know when none fits. Today's options trade off badly:

| Approach | Cost per request | Fine distinctions | Order-invariant | Calibrated abstain | New tools without retraining |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Bi-encoder (MiniLM, Tool-Embed) | query only | weak | yes | no | yes |
| Cross-encoder / pointwise LLM reranker | query x each tool | good | yes | no | yes |
| Listwise LLM reranker / Kev-style pointer | query + every tool's text, every request | best | **no** (we measure a 0.24 probability shift and 13-17% top-1 flips when the menu is reversed) | partly (Kev) | yes |
| Tools as vocabulary tokens (ToolGen) | query only | good | yes | no | **no** |
| **Matryoshka Gist Pointer (ours)** | query + attention over cached gists | listwise | **exact** | **yes (conformal, per modality)** | **yes** |

## Method

1. **Block-isolated tools.** Each tool's text is encoded as its own block (it never sees other tools) with
   catalog-independent positions; the request starts at a fixed offset.
2. **Gists.** g learned gist tokens are appended to each tool block; the request attends only to gists. A tool's
   cache is its g gist KV entries.
3. **Matryoshka (nested) gists.** Gist j attends to gists < j, and training samples a budget b per step so the first
   b gists are a usable summary on their own. One cache supports any budget.
4. **Two-level inference from one cache.** Coarse pass: the request attends to gist 1 of every tool (N tokens).
   Fine pass: all g gists of the top-M tools. Retrieval and listwise reranking in one model, one cache.
5. **Pointer + abstain + conformal.** A pointer head scores every tool and `__abstain__`; split-conformal sets
   turn scores into EXECUTE / ASK / ABSTAIN. Distance-aware (RBF) or ensemble heads are ablations.
6. **Training.** Listwise CE + Brier, gist self-distillation (gist path matches the same model reading the full
   text), nested-budget sampling; later teacher soft labels and contrastive near-miss pairs (see OMNI_DATA.md).

## Formal properties (all verified numerically, fp32, Qwen2.5-0.5B)

- **P1 Composition.** Per-tool caches encoded separately and joined in any order give the same scores as encoding
  the catalog jointly: max |Δp| = 1.2e-6 (gists), 1.8e-7 (late interaction).
- **P2 Equivariance.** Reversing the menu changes scores by ≤ 1.6e-6 (block/gist) vs 0.246 (Kev layout).
- **Corollary.** Split-conformal prediction sets are invariant to tool order; coverage statements do not depend
  on how a client lists its tools.
- **P3 Nesting.** Two-level scoring with full budget and M = N equals single-pass scoring (checked in `arch_bench.py`).
- **Complexity.** Kev: O((L_q + N·L_t)·d²) per request. Ours: O(L_q·d² + L_q·N·b·d) coarse + O(L_q·M·g·d) fine.
  Measured on our data: L_q ≈ 38 tokens vs a menu of ≈ 246 tokens (85% of every Kev prefill is static menu text).

## Measured so far (8-9 Nov 2026, untrained heads, RTX 4050 laptop GPU, Qwen2.5-0.5B)

**Why large menus matter (ToolRet, 400 queries, MiniLM first stage):** recall@1 17%, @16 47.7%, @64 67.5%,
@256 83.3%, @1024 91.0%. A reader limited to 16-64 tools caps end-to-end accuracy at 48-68%; one that reads
1,024 tools has a 91% ceiling.

**Latency per request (bf16, ms):**

| Tools | 6 | 16 | 64 | 256 | 1,024 |
| :-- | --: | --: | --: | --: | --: |
| Kev pointer (query first, full text) | 72 | 252 | 2,204 | cannot run (14.8k tokens) | cannot run (62k tokens) |
| Catalog-first, cached full text | 56 | 65 | 86 | cannot run | cannot run |
| Gist x4, per-tool caches composed | — | — | 102 | 86 | 204 |
| Matryoshka gist x8, per-tool caches | — | — | 87 | 76 | 110 |
| Two-level (1 gist x N -> top 16 x 8) | — | — | 524 | 164 | 361 |

Cache size: 0.047 MB per tool (4 gists, bf16) -> ~48 MB for 1,024 tools; late interaction 0.007 MB per tool.
Two-level is not faster at <= 1k tools on one GPU (single pass already is); its role is memory tiering
(1 gist per tool resident, the rest fetched for the top-M) at 10k+ tools, and its accuracy is measured in P0.
Small-menu timings are noisy on a laptop GPU; final numbers come from T4/A100 with more repetitions.

## Experiments (minimum for a main-track submission)

| ID | Question | Data | Baselines |
| :-- | :-- | :-- | :-- |
| E1 | Accuracy at scale | ToolRet (35 task sets, 43k tools; `toolace` excluded), menus of 16 / 64 / 256 / 1024 | MiniLM, BM25, Qwen3-Embedding-0.6B (+ Qwen3-Reranker-0.6B), jina-reranker-v3, Set-Encoder, PE-Rank, ToolGen (reported), Tool-Embed/Tool-Rank (reported), Kev-style pointer |
| E2 | Calibrated abstention | BFCL v3 irrelevance, When2Call, Soro held-out packages | same + KnowNo-style conformal on a generative LLM |
| E3 | Efficiency | latency and memory vs menu size (6 to 1024), cache MB per tool, build time | all |
| E4 | Ablations | g ∈ {1,2,4,8}, nested vs fixed, distillation on/off, block vs Kev, head type | — |
| E5 | Order robustness | reversed / shuffled menus | Kev, listwise rerankers |

Protocol: 3 seeds, query-clustered bootstrap CIs, fixed hardware (T4 and RTX 4050 for latency, plus CPU),
leakage audit (`soro.omni_data.audit`), code + weights + caches released.

## Compute

Main results on Qwen3-0.6B-Base (block-compatible) on Kaggle 2x T4; one scaling run on Qwen3-1.7B or Gemma 4 E2B.
ToolRet-train (~200k rows, `mangopy/ToolRet-Training-20w`, check license) for scale-up training.

## Venues and timeline (dates from aggregators unless noted; confirm on official sites)

| Venue | Deadline | Fit |
| :-- | :-- | :-- |
| **SIGIR 2027** (San Jose, Jul 18-24) | abstract **14 Jan 2027**, paper **21 Jan 2027** (AoE) | Primary: efficient listwise ranking over tool corpora, ToolRet |
| ICML 2027 | ~16 Jan abstract / ~22 Jan paper (estimate) | Alternative primary for the method + theory |
| ICLR 2027 workshops | typically early Feb | Non-archival feedback (agents, efficiency) |
| Interspeech 2027 | ~9 Feb (aggregator) | Streaming voice routing, only if audio is trained |
| UAI 2027 | ~12 Feb (projection; 2026 was 25 Feb) | If the conformal / anytime-valid angle leads |
| KDD 2027 cycle 2 | ~Feb (unconfirmed) | Applied data science framing |
| ICCV 2027 | ~1 Mar abstract / ~8 Mar paper (projection) | UI-element pointer (image options) |
| **COLM 2027** | ~late Mar (projection) | Fallback / omni extension |
| NeurIPS 2027 (+ Datasets & Benchmarks) | ~mid-May (past pattern) | Full omni model + human multimodal decision benchmark |

Timeline to SIGIR (today: 9 Nov 2026):
- Nov 9-15: Kaggle P0 grid (incl. KM Matryoshka); data pass 1.
- Nov 16-22: pick the design; ToolRet-train mix; baseline runs.
- Nov 23-Dec 13: main training (0.6B, 3 seeds; one 1.7B / E2B run); E1-E3.
- Dec 14-31: ablations (E4-E5), figures, draft.
- Jan 1-14: polish; abstract 14 Jan; paper 21 Jan.
Go/no-go on 22 Nov: if KM does not reach within ~2 points of the full-text block pointer at 16 tools and beat
retrieve-then-rerank on ToolRet at 64+ tools, switch the lead claim to calibration/order invariance and target COLM.

## Risks

- Gist compression can blur look-alike tools (ACL 2025 gist study: weak on exact recall). Mitigations: larger
  fine budget, two-level design, distillation.
- Reviewers will ask why not bi-encoder + cross-encoder cascade: E1/E3 must show the Pareto win.
- Prior-art overlap (PE-Rank, ToolGen, Set-Encoder, Block-Attention, KVLink, gist tokens, KnowNo) must be
  cited and contrasted explicitly; our claim is the combination: exact composable per-tool caches + nested
  two-level decisions + calibrated abstain, in one model.
