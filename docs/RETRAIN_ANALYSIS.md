# Do the models need retraining?

**Short answer: no, not for v0.2.1.** Every fix in v0.2.1 is code, docs or packaging: the API, CLI, CI, the
doc corrections, honest limitations and data licenses. None of them changes what the weights compute.
The published checkpoints (`nisaralll/soro-kev-0.8b`, `nisaralll/soro-kev-4b`) load and route correctly
on CPU with the new `Router` (checked in CI by the `pretrained-smoke` job). Their q̂ values come from the
checkpoint unchanged.

Retraining makes sense only to fix **model-quality** limitations. Two of them are real:

| Limitation | Evidence (committed logs) | Can it be fixed without retraining? |
| :--- | :--- | :--- |
| 0.8B no-tool refusal regression | BFCL irrelevance: 17/33 refused, versus 27/33 for base `kev-0.8b` | **Partly.** Tightening q̂ to 0.1 gives 23/33 (see below). Fully closing the gap needs data. |
| τ² routing | 10.5% (0.8B) and 24.8% (4B) on 105 τ² rows | **No.** This is a domain gap. It needs τ²-style training rows. |
| Order sensitivity (causal attention) | README claims 1.5–3.0% flips (no committed script) | Inference-time permutation averaging (k passes). Retraining with JSD for 4B. |
| Small, correlated eval (115 queries) | Cluster CIs are about ±5 pp | Not a training problem. Needs more eval data. |
| No independent human split | HF file is a byte copy of held-out | Not a training problem. Collect about 300 human queries. |

## CPU experiments run on the box (real numbers)

All of these are reproducible in seconds with no GPU:

```bash
python scripts/per_query_ci.py        # cluster-bootstrap CIs over the 115 distinct queries
python scripts/recalibrate_qhat.py    # refit q̂, gate behaviour, refusal vs q̂
```

Outputs are committed as `benchmarks/per_query_ci.json` and `benchmarks/recalibration.json`.

### 1. Recomputed q̂ (LAC, α = 0.10, s = 1 − p_true, finite-sample quantile)

| Model | Set (n) | Deployed q̂ | Refit q̂ | Coverage at deployed q̂ | Mean \|S\| at deployed q̂ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| 0.8B | val_zeroshot (2,220) | 0.9518 | 0.9974 | **86.4%** (under the 90% target) | 1.15 |
| 0.8B | external held-out (1,200) | 0.9518 | 0.9546 | 89.9% | 1.13 |
| 0.8B | BFCL diag (100) | 0.9518 | 0.9690 | 90.0% | 1.11 |
| 4B | val_zeroshot (2,220) | 0.8066 | 0.7642 | 91.1% | 1.10 |
| 4B | external held-out (1,200) | 0.8066 | 0.4825 | 92.3% | 1.07 |
| 4B | BFCL diag (100) | 0.8066 | 0.2524 | 92.0% | 1.02 |

Readings:

- The 0.8B deployed q̂ **under-covers on val_zeroshot** (86.4% against a 90% target). That is the
  distribution it was fitted on, using only a 100-row slice. A refit on the full 2,220 rows gives
  0.9974, which is nearly "include everything above 0.26%" and would turn 14.5% of rows into ASK.
- Grouped by base query, 200 random 50/50 cal/test splits of the external set give a 4B q̂ range of
  **0.06–0.87** (5th to 95th percentile), with test coverage as low as 75.7%. For 0.8B the range is
  0.63–0.997, with a minimum coverage of 71.6%. **115 queries are not enough to calibrate on.**
  Calibrate on your own traffic: a few hundred distinct queries.
- These numbers do not justify retraining. They argue for **per-deployment recalibration**, which
  `Router(q_hat=...)` and `soro-boro eval` already support.

### 2. Can the gate fix the 0.8B refusal regression?

Refusal on the 33 BFCL irrelevance rows, and correct EXECUTEs on the 67 tool rows, as q̂ varies
(lower q̂ means a stricter threshold 1 − q̂):

| q̂ (threshold) | 0.8B refused /33 | 0.8B correct EXECUTE /67 | 4B refused /33 | 4B correct EXECUTE /67 |
| :--- | :---: | :---: | :---: | :---: |
| 0.9518 (0.05), 0.8B deployed | 17 | 64 | 25 | 65 |
| 0.8066 (0.19), 4B deployed | 17 | 65 | 25 | 66 |
| 0.3 (0.70) | 20 | 66 | 25 | 66 |
| 0.1 (0.90) | **23** | 65 | 25 | 65 |
| 0.01 (0.99) | 26 | 60 | 29 | 60 |

Of the 16 irrelevance rows the 0.8B model gets wrong, 10 put ≥ 0.94 probability on a tool, so no
threshold separates them. At q̂ = 0.1 on the larger sets, 0.8B abstains on 22.3% of val_zeroshot
(deployed: 14.1%) and 20.0% of external (14.4%). EXECUTE precision rises from 89.7% to 90.3% and from
92.1% to 93.8%.

**Recommendation:** for refusal-sensitive deployments, use `Router.from_pretrained(..., q_hat=0.1)`
or the 4B model. This is tuned on 100 BFCL rows, so validate it on your own traffic. The remaining
gap to base `kev-0.8b` (27/33) needs **data**, not gate tuning.

## If you do retrain (v0.3)

### What it would address

1. **No-tool refusal at 0.8B.** Train on more and harder no-tool rows (BFCL-irrelevance-style chit-chat,
   near-miss requests, "explain how X works" next to tool X). The current abstain share in train has not been measured here. Target 15–20% abstain gold, with hard negatives drawn from the
   same catalog.
2. **τ² domain.** Convert more τ² telecom and retail turns: multi-turn state, persona context, tools
   whose names collide with generic MCP tools. τ² train rows today are airline only (1,436). Telecom
   and retail appear only in eval. **Keep the eval domains out of train**, or carve a new held-out
   domain, so the comparison stays honest.
3. **4B JSD.** The 4B model was trained `single_sequence` (JSD weight 0) because of T4 memory. On a
   24 GB+ GPU, train with the permuted sequence (JSD 0.30) to reduce order flips.
4. **Calibration data.** Hold out ≥ 500 distinct queries (not views) for q̂ fitting, and report
   coverage on a separate distinct-query test set.

### Data plan

| Addition | Rows | Source | License note |
| :--- | :---: | :--- | :--- |
| Hard no-tool negatives | 3–5k | BFCL irrelevance style, generated against existing catalogs, human-checked sample | generator terms; BFCL Apache-2.0 for style only |
| τ² telecom/retail train turns | 2–3k | τ²-bench task splits not used in eval | MIT |
| Real human queries (eval only) | ~300 | written or commissioned | own |
| Distinct-query calibration set | ≥ 500 queries | new, disjoint packages | own |

### Recipe

Same as v0.2.1: Qwen backbone, LoRA r = 16 / α = 32, pointer heads, loss
`route_CE + 0.25 Brier + 0.30 JSD + 0.50 arg_CE + 0.25 missing + 0.25 mutation + 0.40 intent`.
Change only the data mix and, for 4B, turn off `single_sequence`. Gate the release on:

- external accuracy within its CI of v0.2.1;
- BFCL irrelevance ≥ 25/33 at 0.8B;
- τ² ≥ 25% at 0.8B;
- coverage ≥ 88% on the distinct-query test set.

### GPU budget (estimates, not measured)

| Job | Hardware | Est. time | Est. cost |
| :--- | :--- | :---: | :---: |
| 0.8B LoRA, about 30k rows, 2 epochs, two sequences | RTX 4050 Laptop (6 GB), owned | 6–12 h | about $0 (electricity) |
| 0.8B same | rented 24 GB (RTX 4090 / L4 / A10) at about $0.4–0.8/h | 2–4 h | about $1–3 |
| 4B LoRA, two sequences (JSD on) | 6 GB laptop | **does not fit** (needs single_sequence or a smaller batch with gradient checkpointing) | n/a |
| 4B LoRA, two sequences | rented A100 80 GB / H100 at about $1.5–3/h | 4–8 h | about $6–25 |
| Eval + calibration (both models, all sets) | laptop GPU | about 1 h | about $0 |
| **Total, one v0.3 iteration** | | | **about $10–30 rented**, plus 2–3 attempts, so **about $30–90** |

Rental prices are typical 2026 marketplace ranges (RunPod, Lambda, Vast) and were not checked
for this document. Times are scaled from the dataset size, not measured.

### Cheap experiments to run before retraining

1. ✅ *Done (CPU):* refit q̂ from logs; refusal versus q̂; cluster CIs (above).
2. *CPU, minutes:* permutation-averaged inference (k = 2–4 shuffles of the tool order) on 100 external
   rows with the 0.8B model on CPU. This measures the real order-flip rate, which the README states but
   the repo cannot reproduce.
3. *CPU, minutes:* abstain-prior bias. Add a constant δ to the `__abstain__` logit and sweep δ on
   val_zeroshot. This often recovers refusal better than a global threshold, because it moves only
   the abstain slot. It needs the logit caches (`veyra/reports/logit_cache_*_300.jsonl`).
4. *Laptop GPU, 1–2 h:* a 1-epoch 0.8B LoRA on train plus 3k hard negatives. Check whether BFCL
   irrelevance moves before spending on 4B.
