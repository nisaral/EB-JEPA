# Soro-Kev-Omni: plan for the third model

Status: draft, 8 Nov 2026. Owner: Keyush. P0 code is in the repo; P0 results are at the end.

## 1. What we are building, and what we are not

A **small, calibrated decision model** that reads text, images and audio and picks from a closed set:
a tool, a UI element, an image region, an audio segment, or "abstain / ask / send more input".

It does **not** generate text and does not compete with frontier models on open-ended reasoning. That
needs data, compute and time we do not have, and it is not the job. The job is the closed-set decision
that sits in front of an agent, where a small model can match a big one and beat it on latency,
calibration and cost. The evidence that this works: GUI-Actor-7B's pointer head beats UI-TARS-72B on
ScreenSpot-Pro, and Kev-9B lands within a few points of hosted Jev on typed decisions.

## 2. Gap in the field (Nov 2026)

| System | Inputs | Options | Calibrated abstain | Small |
| :-- | :-- | :-- | :-- | :-- |
| Jev / Kev (TypeSafe, Palmer) | text | text spans | yes (text only) | 0.8–9B |
| PixelJev (arXiv 2609.29283) | image + text | text candidates | no ("accuracy gains do not ensure calibrated probabilities") | 2–4B |
| GUI-Actor (NeurIPS 2025) | image + text | image patches | no | 2–7B |
| Arch-Router, xLAM, Hammer | text | generated names | no | 1–1.5B |
| Speech tool-calling (BFCL Audio, Audio2Tool, VoiceAgentBench) | audio | generated calls | no; evaluated on big SpeechLMs / ASR→LLM | large |
| **Soro-Kev-Omni (this plan)** | **text + image + audio** | **spans of any modality** | **yes, per modality** | **~2–4B** |

No published system combines a pointer over any-modality options with calibrated, per-modality abstention.

## 3. Architecture: Universal Probabilistic Pointer (UPP)

Code: `veyra/python/src/soro/routing/upp.py`. Each part is a flag, so it can be ablated alone.

```
[catalog / options] → [input: text | image | audio] → [think slots ×K] → [readout token]
 KV-cached once        spliced in as embeddings      learned, no decode       query
     │ option span ends (text desc | image region | audio segment | UI element)  │
     ▼                                                                            ▼
 (μᵢ, log σᵢ²) ─────────── set refiner: 2-layer transformer, no positions ───── (μ_q, log σ_q²)
     └── score = −½ Σ [ (μ_q−μᵢ)² / (σ_q²+σᵢ²) + log(σ_q²+σᵢ²) ]  /  (√d · T) ──┘
                                        ▼
       conformal set per modality (Mondrian), σ-normalized scores
       → EXECUTE · ASK · ABSTAIN · NEED_INPUT
```

| Component | Fixes | Precedent |
| :-- | :-- | :-- |
| **Gaussian pointer** (`head="gauss"`) | Point scores carry no uncertainty. σ grows with noisy audio, blurry images or missing input, giving abstain a learned signal and letting conformal sets adapt per request. A small KL to N(0, I) regularizes the latent (VAE-style). | PCME (CVPR 2021): Gaussian embeddings beat point embeddings for ambiguous cross-modal matches |
| **Catalog first + KV cache** (`layout="catalog_first"`) | Today every request re-reads the menu. Cached, a request pays only for its own tokens; this matters most for image/audio, where the input is the expensive part. Also enables streaming: append audio chunks and re-read the pointer. | Prefix caching in LLM serving |
| **Set refiner** (`refiner=True`) | Options before the query cannot see it; the refiner restores that after the backbone. It is order-equivariant, so it removes the causal order bias (the 4B was trained without the JSD term). | Set Transformer / ISAB |
| **Think slots** (`think_slots=K`) | Extra compute without decoding. Used adaptively: K=0 by default, K>0 only when σ is high or the set is not a singleton. | Pause tokens (Goyal et al.), adaptive pause (DeepMind) |
| **Universal options** | One head for tool routing, UI-element clicks, image regions, audio segments. An option is a span of any modality; its readout is the span-end (text) or pooled patch/frame states (image/audio). | Kev spans + GUI-Actor patches + PixelJev candidates |
| **Per-modality conformal + early commit** | One noisy modality must not break the coverage guarantee of another. For voice, commit as soon as the set is a singleton, with conformal risk control bounding premature commits. | Mondrian CP, conformal risk control |

## 3b. Research round 2: what can beat or replace each part

### Readout (accuracy)
- **Keep the "last-token" pointer.** jina-reranker-v3 (Qwen3-0.6B) reads one embedding at each document's
  last token after causal attention over query + documents and reaches SOTA-level BEIR. That is the Kev
  readout, so the core is sound.
- **Add attention-pooled option readout**: a learned query attends over all tokens of an option span,
  instead of only its last token. Cheap; helps long descriptions and image/audio spans.
- **Learning-to-rank weighting** (FIRST, EMNLP 2024): weight errors on the top candidates more than at the tail.

### Layout (speed, order invariance)
- **Block-isolated options** (Block-Attention, ICLR 2025): each option is its own block with its own
  positions, so options never attend to each other; only the request block attends to all of them.
  - Exact permutation invariance by construction, so no JSD term and no refiner workaround.
  - **Per-tool KV cache**: a tool's states are computed once and reused across every catalog it appears
    in. Adding or removing an MCP tool costs one tool, not the whole menu.
  - Block-Attention needed fine-tuning to recover accuracy, and after that matched full attention
    (68.4% vs 67.9% on Llama 3), with prefill time-to-first-token cut to ~1% at 32K.
  - Kev's own training config has an `option_isolation` flag (off in kev-0.8b), so the Jev line has
    looked at this.
  - **Constraint:** block masks need full-attention layers. Qwen3.5's Gated DeltaNet layers are
    recurrent and cannot isolate blocks. Use Qwen3 / Gemma 4 backbones for this variant.
- This replaces "catalog-first + refiner" as the main caching design; the refiner stays as an option.

### Uncertainty head (calibration, abstain): alternatives to the Gaussian embedding

| Method | Cost | Evidence | Verdict |
| :-- | :-- | :-- | :-- |
| Gaussian embeddings (current B) | 1 pass | PCME: helps ambiguous cross-modal matching | Keep as a candidate |
| **Distance-aware pointer (SNGP-style)**: spectral-norm projections + RBF/GP score instead of dot | 1 pass | SNGP beats other single-model methods on calibration and OOD for BERT-class models, close to deep ensembles | **Strong candidate.** "No tool fits" and unseen tools are OOD problems |
| **Head / LoRA ensembles** | K tiny heads (or K LoRAs batched) | LoRA ensembles improve LLM fine-tune calibration at near single-model cost | Strong; heads-only ensemble is nearly free |
| **Energy score** (logsumexp of pointer logits) | free | Energy beats softmax confidence for OOD (−18% FPR95 on CIFAR-10) | Use as an extra abstain signal in every variant |
| **Laplace-LoRA** (post-hoc) | no retraining | Large calibration gains on fine-tuned LLaMA2-7B | **Quick win: can be applied to the released 0.8B/4B as-is** |
| Evidential (Dirichlet) heads | 1 pass | Known calibration problems in the literature | Skip |

### Training
- **Cross-modal consistency distillation** (BLSP / BLSP-KD): the speech-conditioned model is trained to
  match the same model's distribution on the transcript. For us: KL(p(audio) ‖ p(transcript)) on TTS
  rows, and KL(p(screenshot) ‖ p(OCR/caption)) on image rows. This needs no audio teacher for most rows
  and ties the modalities into one decision space.
- **Early-exit training** (LayerSkip, ACL 2024): layer dropout + a shared early-exit loss so the pointer
  works from intermediate layers.
- **Block fine-tuning curriculum**: start with full attention, then switch to block-isolated options.
- Kept from round 1: listwise KD from teachers, InfoNCE mid-training, RLCD-style near-miss pairs,
  ConfTr set-size loss, Brier (bounded proper scoring, per RLCR).

### Inference
- **Calibrated early exit** (CALM, NeurIPS 2022): stop at the first layer whose conformal set is a single
  tool, with distribution-free risk control on the error. CALM reports up to 3× speedups on generation.
- **Cascade**: real MiniLM gets 79% on our external set in ~20 ms. Let it decide when it is confident
  under a conformal risk bound; send the rest to UPP.
- Per-tool KV cache (block layout), energy + per-modality conformal, think slots only on hard requests.

### Revised P0 grid (next run, on Kaggle)

| ID | Variant | Tests |
| :-- | :-- | :-- |
| A | Kev dot pointer | baseline |
| B | Gaussian pointer | round 1 |
| F | Block-isolated options + per-tool cache | layout v2 |
| G | Distance-aware (SN + RBF) pointer | uncertainty v2 |
| H | 4-head ensemble + energy | uncertainty v2 |
| I | Early-exit pointer at 3 depths | inference speed |
| J | best of F/G/H combined | candidate for Omni |

### What each component is for

| Component | What it buys | Where it matters |
| :-- | :-- | :-- |
| Block-isolated options + per-catalog cache | Exact order invariance (measured: max prob change 3.6e-7 on Qwen2.5 vs 0.355 for the Kev layout); request-only latency | MCP gateways with 50–500 tools, IDE agents re-routing every turn, phones (pay only for the request) |
| Distance-aware (RBF) pointer | Low scores when nothing is close: "no tool fits", unseen tools, off-topic chat | Safe abstain in customer-facing agents; new MCP servers the model never saw |
| Head ensemble + energy | Disagreement and energy flag shaky decisions | Where a wrong call costs money: payments, deletes, sending messages (route to ASK or a human) |
| Early-exit readouts | Easy requests stop at half depth | Edge devices, voice assistants, high-QPS routers |
| Gaussian head | Uncertainty that grows with noisy or missing inputs | Noisy audio, blurry screenshots; drives NEED_INPUT |
| Per-modality conformal | Guarantee holds separately for text, image and voice | Regulated settings that need a stated error rate |
| Think slots | Extra compute only on hard requests | Near-duplicate tools, long catalogs |

### Parent backbone decision (checked 8 Nov 2026 on the Hub)

| Candidate | Modalities | License (HF card) | Block layout | Verdict |
| :-- | :-- | :-- | :-- | :-- |
| **google/gemma-4-E2B** | text, image, audio | Apache-2.0, ungated | Full + 512-token sliding layers; to confirm in the notebook | **Primary parent ("Soro-Kev-Omni-2B")** |
| google/gemma-4-E4B | text, image, audio | Apache-2.0, ungated | same | Second tier ("Omni-4B") |
| Qwen/Qwen3-VL-2B / 4B | text, image | Apache-2.0 | full attention | Fallback; audio would need the Qwen3-ASR encoder + projector |
| Qwen/Qwen3-0.6B / 1.7B Base | text | Apache-2.0 | full attention | P0 test bed (fast, block-compatible) |
| Qwen3.5-0.8B / 2B | text, image | Apache-2.0 | **No** (Gated DeltaNet is recurrent) | Keep for the current text models only |
| Phi-4-multimodal (5.6B) | text, image, audio | MIT | full attention | Backup omni parent; heavier |
| Qwen2.5-Omni-3B | text, image, audio | Qwen Research (non-commercial) | — | Excluded |
| LFM2.5 VL / Audio | split models | LFM Open | — | Excluded |

Teacher: Qwen/Qwen3-Omni-30B-A3B-Instruct (card: license_name apache-2.0) for audio + image + text,
Gemma 4 31B for stronger image/text labels.

The notebook `veyra/training/kaggle_p0_upp.ipynb` runs `parent_check.py` (block exactness, cache
exactness, early exits, VRAM, latency) on Qwen3-0.6B, Qwen3-1.7B and Gemma 4 E2B, then the P0 grid on
Qwen3-0.6B, then baseline vs combined (A vs J) on Gemma 4 E2B in 4-bit. The parent is final when Gemma
passes the exactness checks and J ≥ A on held-out accuracy with better calibration.

## 4. Backbone (student) and teachers

### Student
- **Gemma 4 E2B** (Apache-2.0; ~2.3B effective, ~5B total with per-layer embeddings): text, image and
  audio (≤30 s) natively; ~150M vision and ~300M audio encoders; llama.cpp runs its audio and vision;
  `transformers 5.19` (installed) has `gemma4`. **Primary student, "Soro-Kev-Omni-2B".**
- **Gemma 4 E4B** (~4.5B effective, 8B total): the "Omni-4B" tier after E2B works.
- Fallback: Qwen3.5-2B (native vision, already the Soro family) + the Qwen3-ASR-0.6B AuT audio encoder
  (180M, Apache-2.0) with a trained projector. More work (alignment), but no Gemma dependency.
- Note: Gemma 4 26B/31B are vision-only; audio is only in E2B/E4B.

### Teachers (soft labels for distillation)
Soft labels need logits, so teachers must be open-weight; API models also have terms that bar training
competing models.

| Role | Teacher | License | Why |
| :-- | :-- | :-- | :-- |
| **Omni (audio + image + text)** | **Qwen3-Omni-30B-A3B-Instruct** (Thinking variant for hard rows) | Apache-2.0 | Only open model with strong audio + vision + text in one; 3B active, runs on one rented A100/L40S |
| Vision + text, stronger | Gemma 4 31B | Apache-2.0 | Stronger on images and UI; same family as the student |
| Text, strongest | Qwen3.5-122B-A10B (or 397B-A17B via a provider with logprobs) | Apache-2.0 (check card) | Best open text decisions for tool routing |
| Avoid for now | Nemotron 3 Nano Omni | NVIDIA Open Model Agreement | Read distillation terms first |
| Avoid | Gemini / GPT / Claude APIs | ToS | No logits; terms restrict training competing models |

How labels are made: score option letters with the teacher (same readout as our option-letter baseline),
average over k permutations to remove the teacher's position bias, temperature-scale it on a labeled
slice, then mix with gold: `target = λ·onehot(gold) + (1−λ)·teacher`. Rows where the teacher confidently
disagrees with gold go to a label-noise review queue, not into training.

## 5. Training recipe (pre-, mid- and post-training)

We do not pretrain a backbone. Our "pretraining" is decision mid-training on top of one.

| Stage | What | Data | Loss | Why it helps small-vs-big |
| :-- | :-- | :-- | :-- | :-- |
| **S1. Decision mid-training** | Contrastive pretraining of the pointer and LoRA on huge weak pairs | 1–2M (query, option) pairs: MCP registry tool descriptions, OpenAPI specs, ToolACE/Toucan, OS-Atlas UI elements (13M), teacher-written queries | InfoNCE over in-batch options + format randomization (JSON / REST / MCP / signature) + delexicalization | Broad coverage of tool/UI vocabulary cheaply, before expensive labels |
| **S2. Listwise distillation SFT** | Multi-task UPP training on curated rows, curriculum text → image → audio | ~200k rows (≈40% text, 30% image, 30% audio), 15–20% no-tool | CE(gold) + KL(teacher, T²) + 0.25 Brier + JSD permutation + β·KL(Gaussian) + modality dropout | Teacher distributions carry "dark knowledge" a closed-set head absorbs very efficiently (Kev used only ~12k rows) |
| **S3. Contrastive near-miss pairs (RLCD-style)** | RLCD (Yang et al., ICLR 2024) builds cleaner preference pairs by prompting a model two contrasting ways. We reuse the idea for decisions: same menu, two inputs differing minimally, one needs tool X, the other abstain or tool Y | ~30k pairs written by the teacher with positive/negative prompts | Pairwise margin on the pointer + abstain CE | Targets exactly where small models fail: near-duplicate tools and false calls (0.8B refusal 17/33) |
| **S4. Calibration-aware post-training** | ConfTr (Stutz et al., ICLR 2022): differentiable conformal set size on a split batch; proper scoring (Brier) as in RLCR (Damani et al., ICLR 2026) | Same as S2 | ConfTr size loss + Brier | Smaller ASK sets at the same coverage. A closed-set head can optimize Brier directly; RL (GRPO) only for multi-step plans |
| **S5. Test-time** | Adaptive think slots, permutation averaging (`Router(permutations=k)`, done), per-modality q̂, online recalibration from logs | Deployment traffic | — | Spend compute only on hard requests |

Augmentations in every stage: option shuffling, option dropout, schema-format randomization, audio noise /
speed / accent (TTS voices), image resize / crop / JPEG, missing-modality rows with gold `NEED_INPUT`, and
streaming prefixes (audio cut at random points; gold is the final decision, the early-commit rule decides).

## 6. Data inventory

| Modality | Source | Size | License | Use |
| :-- | :-- | :-- | :-- | :-- |
| Text | Current Soro train + external intent | 23k + 15.7k | mixed (see DATA_LICENSES.md) | S2 |
| Text | ToolACE, Toucan conversions (scripts exist) | 10k–100k | check | S1/S2 |
| Text | MCP registries / OpenAPI specs (scrape descriptions) | 10k+ tools | per server | S1 options |
| Image | OS-Atlas grounding data | 2.3M screenshots, 13M elements | Apache-2.0 | S1/S2 UI-element pointing |
| Image | Screenshot / doc / receipt / chart + tool menus (synthesized, teacher-labeled) | ~60k | depends on image source | S2 image tool routing |
| Image (eval) | ScreenSpot-Pro, M³-Bench (231 MCP tools), EgoBench | — | check | eval only |
| Audio | Audio2Tool | ~30k | CC BY 4.0 | S2 |
| Audio | TTS of the text rows, several voices + noise | ~60k | own | S2, S3 |
| Audio (eval) | BFCL Audio, VoiceAgentBench, SpeechFCEval | 6k+ | check | eval only |

Hold-out rule (fixed in `train_soro_kev.py` this release): checkpoints and q̂ are selected only on
packages carved out of training data. Reported test sets are never read during training.

## 7. Compute and budget

| Phase | Hardware | Time | Cost |
| :-- | :-- | :-- | :-- |
| P0 architecture ablation (text) | RTX 4050 6 GB (local) | hours | free |
| P1 teacher labeling (~200k rows) | rented A100 80 GB, vLLM, Qwen3-Omni-30B-A3B + Gemma 4 31B | ~10–20 h | ~$10–30 (Vast A100 ≈ $0.80/h) |
| TTS + augmentation | local 4050 | days, unattended | free |
| S1 mid-training (LoRA, E2B) | rented 4090 / A100 | ~10 h | ~$5–15 |
| S2–S4 (E2B, 4-bit LoRA) | rented A100 / H100 | ~15–30 h | ~$20–60 |
| Iterations / failures | — | — | ×2 |
| **Total** | | | **~$100–250**, or mostly free on Kaggle (30 GPU-h/week) at ~4–6 weeks |

For scale, Kev's whole Qwen3.5 port cost about $95 of H100 time.

## 8. Evaluation and go/no-go gates

Metrics on every run: accuracy, NLL, Brier, ECE, refusal on no-tool rows, order-flip rate, conformal
coverage and set size (plain and σ-normalized), latency cold and cached, peak VRAM.

| Gate | Pass condition |
| :-- | :-- |
| P0 | A UPP variant matches Kev-style A within noise on accuracy and beats it on NLL/ECE, flip rate or cached latency |
| P2 (text, E2B) | ≥ Soro-Kev-4B on held-out text at lower latency |
| P3 (omni) | Image: within 5 pts of Gemma 4 31B zero-shot on closed-set image tool routing; UI: competitive with GUI-Actor-2B class on ScreenSpot. Audio: ≥ ASR→LLM pipeline baseline on BFCL Audio decisions. Calibrated coverage per modality |

Baselines are real models only (the fabricated ones were removed in this release).

## 9. Use cases (what people would install it for)

1. **MCP pre-router** for Claude Desktop, Cursor and IDE agents: pick the server/tool or abstain before the big model spends tokens.
2. **On-device voice control**: car, home, wearables (Audio2Tool's domains), deciding while the user speaks.
3. **Computer-use / GUI agents**: which element to click, or "not on this screen"; a calibrated replacement for coordinate generation.
4. **Support triage with screenshots**: route a ticket plus image to the right workflow or human.
5. **LLM cascade router**: decide small model vs big model vs tool, with a coverage guarantee.
6. **Typed decisions with evidence**: moderation, policy and compliance checks over text, images or calls (Kev-style questions, now multimodal).
7. **Accessibility**: voice-driven app control with safe abstain.

Distribution: `pip install soro-boro` with `route(text=, image=, audio=)`; an MCP server (`soro-router`);
llama.cpp GGUF backbone + head (to verify: per-token hidden-state export for multimodal input); HF Space demo.

## 10. Risks

- Catalog-first can lose accuracy because options do not see the query inside the backbone (P0 measures this; the refiner is the mitigation).
- Gemma 4 E2B's audio encoder works on whole clips; streaming needs chunked training.
- Teacher bias leaks into the student; permutation-averaged teacher scores and gold mixing limit it.
- Licenses of image sources and some audio sets need checking before release.
- 6 GB local GPU limits P0 to ~0.5B backbones; bigger ablations need Kaggle or rented GPUs.

## 11. Timeline

| Week | Work |
| :-- | :-- |
| 1 | P0 ablation (done locally), pick the head; set up teacher scoring script and TTS pipeline |
| 2 | P1 teacher labels (text + image), Audio2Tool + TTS audio; S1 data scrape |
| 3 | S1 + S2 text on E2B; gate P2 |
| 4 | Add image, then audio (curriculum); S3 near-miss pairs |
| 5 | S4 calibration training; full eval vs big models; gate P3 |
| 6 | Release: weights, `route(image=, audio=)`, MCP server, Space |

## 12. P0 results

(filled in from `runs/p0_upp/results.md` below)
