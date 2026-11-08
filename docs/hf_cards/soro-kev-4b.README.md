---
license: apache-2.0
library_name: peft
pipeline_tag: text-classification
base_model:
  - Qwen/Qwen3-4B
datasets:
  - nisaralll/soro-universal-tool-harness
tags:
  - mcp
  - tool-calling
  - agent-harness
  - decision-model
  - pointer-network
  - conformal-prediction
---

# Soro-Kev-4B

[![PyPI Version](https://img.shields.io/pypi/v/soro-boro.svg?color=blue)](https://pypi.org/project/soro-boro/)
[![Interactive Space Demo](https://img.shields.io/badge/Hugging%20Face-Interactive%20Space%20Demo-blue)](https://huggingface.co/spaces/nisaralll/soro-kev-router)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Soro--Boro-black)](https://github.com/nisaral/Soro-Boro)
[![Dataset](https://img.shields.io/badge/Dataset-Universal%20Tool%20Harness-green)](https://huggingface.co/datasets/nisaralll/soro-universal-tool-harness)

```bash
pip install soro-boro
```

## Model summary

Soro-Kev-4B is a **Single-Prefill Causal Listwise Tool Reranker & Multi-Task Pointer Classifier with an Explicit `__abstain__` Slot** (1 forward pass for menus $\le 16$ tools; 2-stage tournament chunking for larger menus). It reads the state (user prompt, conversational history, and registered tool schemas) and produces a closed-set probability distribution over the options supplied with a question. Each question is scored in a prefill-only forward pass; the model does not generate text.

It is built directly on top of Jared Palmer's Kev pointer architecture, addressing an explicit limitation documented in the original Kev model cards (*"Tool-call routing: its When2Call accuracy is below chance (0.133). Do not use it for tool-call routing"*). Soro-Kev-4B is fine-tuned specifically on the multi-protocol Universal Tool Harness, raising tool routing accuracy to 90.58% (1,087/1,200) on held-out MCP & agent benchmarks, and 91.00% (91/100) on BFCL v3 tool selection.

It consists of a LoRA adapter ($r=16, \alpha=32$) and a pointer head on `Qwen/Qwen3-4B`. In 4-bit NF4 quantization, it operates in ~4,048 MB VRAM with a median prefill latency of ~648.6 ms on an NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM, PCIe Gen 4, PyTorch 2.6.0, CUDA 12.4). It is intended for production AI agents, complex Model Context Protocol (MCP) tool catalogs (50+ tools), and backend API routing engines where generative LLM JSON hallucinations and latency overhead cannot be tolerated.

---

## Model details

| Field | Detail |
| :--- | :--- |
| **Developer** | Keyush Nisar ([github.com/nisaral/Soro-Boro](https://github.com/nisaral/Soro-Boro)) |
| **Model type** | Single-Prefill Causal Listwise Tool Reranker & Multi-Task Pointer Classifier with an Explicit `__abstain__` Slot |
| **Backbone** | `Qwen/Qwen3-4B`: 36 layers ($d_{\text{model}}=2560$), multi-head attention; frozen |
| **Adapter** | LoRA, rank 16, $\alpha=32$, on `q_proj`, `k_proj`, `v_proj`, `o_proj` across all 36 layers (144 adapted modules; no MLP adapters) |
| **Projection & Classification Heads** | `q_proj_head` and `opt_proj_head` (`Linear(2560, 256)`), scoring candidate options via $q \cdot k / (\sqrt{256} \cdot T)$. Auxiliary heads: `binary_head` (`Linear(256, 2)`) for `missing_required` and `is_mutation`, and `intent_head` (`Linear(256, 3)`) |
| **Loss** | Multi-Task Calibrated Pointer Loss: $\mathcal{L} = \mathcal{L}_{\text{route\_CE}} + 0.25 \mathcal{L}_{\text{Brier}} + 0.50 \mathcal{L}_{\text{arg\_CE}} + 0.25 \mathcal{L}_{\text{missing\_CE}} + 0.25 \mathcal{L}_{\text{mutation\_CE}} + 0.40 \mathcal{L}_{\text{intent\_CE}}$ + 0.30 $\mathcal{L}_{\text{JSD}}$ (permutation consistency; the JSD weight is 0.30 whenever a permuted second sequence is used and 0 only in `single_sequence` mode, which the 4B T4 run used, so this checkpoint was trained without the JSD term) |
| **Precision** | Trained in bf16/fp16; served in NF4 4-bit (~4,048 MB VRAM) or fp16 (~8,200 MB VRAM); evaluated in fp32 |
| **Context length** | Backbone supports up to 32,768 tokens; evaluated up to 8,192 tokens |
| **Calibration & Temperature** | Learned pointer temperature $T = 0.0709$ (`0.070878`), split-conformal quantile $\hat{q} = 0.8066$ at nominal coverage $1 - \alpha = 0.90$ |
| **Production Decision Policy** | Split Conformal LAC: $S = \{ i \mid p_i \ge 1 - \hat{q} \}$. If $S = \emptyset$ or $S = \{\text{"\_\_abstain\_\_"}\}$ (or abstain is top-1): `ABSTAIN`; if $|S \setminus \{\text{"\_\_abstain\_\_"}\}| == 1$: `EXECUTE`; if $|S \setminus \{\text{"\_\_abstain\_\_"}\}| \ge 2$: `ASK` |
| **Languages** | English (tool schemas, OpenAPI specs, MCP JSON-RPC, Python docstrings) |
| **License** | Apache-2.0 |
| **Version** | Soro-Kev 1.0 (release `v0.2.1`, 2026-10-19) |
| **Base Kev Ancestor** | Derived from pointer network methodology of [jaredpalmer/kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b) |

---

## Input and Question Types

Soro-Kev-4B evaluates a state and any number of named questions across three typed categories:

| Type | Options | Output | Intended Purpose |
| :--- | :--- | :--- | :--- |
| **choice (`tool_route`)** | Up to 16 tool schemas + `__abstain__` in a single forward pass; larger catalogs (> 16 tools) use 6-tool chunks and a final top-two Championship Pass | Probability per listed tool and `__abstain__`; evaluation reports calibration sets | Selects the single correct tool to execute, or abstains |
| **choice (`arg_mapping`)** | Schema property keys + `__drop__` | Probability distribution over candidate property names | Maps unrecognized LLM argument keys to target schema parameters |
| **noul (`missing_required`)** | `yes` / `no` with descriptions | Probability that required arguments are missing | Determines whether harness must ask user before calling tool |
| **noul (`is_mutation`)** | `yes` / `no` with descriptions | Probability that action mutates external state | Classifies side-effect tier when MCP tool annotations are absent |

Each question is scored by its own forward pass in the current inference code. Questions do not share a cached state prefix, and the model does not promise a fixed number of questions per forward.

---

## Intended Uses

- **Production AI Agent Tool Routing**: Selecting tools from catalogs of 50+ schemas through a shortlist/chunking path without token-by-token JSON generation.
- **Model Context Protocol (MCP) Server Gateways**: High-accuracy routing across heterogeneous tool protocols (MCP JSON, REST APIs, GraphQL, Python callables).
- **Calibration-Aware Action Boundary**: Production routing natively applies Split Conformal LAC prediction sets ($S = \{ i \mid p_i \ge 1 - \hat{q} \}$) loaded directly from the checkpoint's `temperature.json` ($\hat{q} = 0.8066$).
- **Out-of-Domain Rejection**: Calibrated abstention (75.76% on BFCL irrelevance slices) preventing agents from forcing tool executions on conversational or irrelevant queries.
- **Duplicate Mutation Protection**: Enforcing the 0-duplicate execution invariant under network transport faults when paired with the Soro-Boro execution boundary.

## Out-of-Scope Uses

- **Text generation, conversation, or creative writing**: The model runs in prefill-only mode and has no generative token decode loop.
- **Open-ended text synthesis**: The model selects and normalizes schemas; it does not author free-form documents.
- **Fully autonomous execution of high-risk actions**: Critical financial or destructive operations should always require operator confirmation (`ASK` / human-in-the-loop).
- **Languages other than English**: Non-English tool descriptions and prompts are not validated.

---

## How to Use

### 1. High-Level Python SDK (Soro Harness)

```python
from soro.harness.pipeline import SoroHarness

# Loads directly from Hugging Face in NF4 quantization (4 GB VRAM)
harness = SoroHarness.from_pretrained("nisaralll/soro-kev-4b")

@harness.tool(description="Look up customer order details", side_effect="read_only")
def get_order_status(order_id: str) -> dict:
    return {"order_id": order_id, "status": "shipped", "tracking": "TRK-9810"}

@harness.tool(description="Cancel a warehouse shipment", side_effect="mutation")
def cancel_shipment(order_id: str) -> dict:
    return {"order_id": order_id, "cancelled": True}

# Fast Calibrated Routing
decision = harness.route("Where is my package for order ORD-1092?")
print("Decision Action:", decision.action)          # "EXECUTE"
print("Selected Tool:", decision.selected_tool)     # "get_order_status"
print("Confidence:", decision.probabilities.get("get_order_status"))

# Execute with Automatic Argument Coercion
result = harness.execute("get_order_status", {"order_id": "ORD-1092"})
print("Execution Result:", result.data)
```

### 2. Low-Level In-Process Scorer

```python
from soro.routing.kev_local import LocalKevScorer

scorer = LocalKevScorer(run="nisaralll/soro-kev-4b", backend="neural")
scorer.load()

state = "User Request: Check available disk storage in /var/log"
questions = {
    "tool_route": {
        "type": "choice",
        "instructions": "Which tool should execute for this request?",
        "criteria": {
            "get_disk_usage": "Return total, used, and available disk storage for path.",
            "list_processes": "List active system processes and PIDs.",
            "__abstain__": "None of the listed tools fit."
        }
    }
}

result = scorer.evaluate_turn(state, questions)
print(result["answers"]["tool_route"])  # "get_disk_usage"
```

---

## Training Data & Target Reconciliation

Soro-Kev-4B was trained on the `nisaralll/soro-universal-tool-harness` dataset (`train.jsonl`), consisting of exactly **23,192 multi-task state rows**. Each row contains five question targets. During training, the non-permuted questions are packed into one sequence and the permutation probe is a second sequence in the same batch; this is a training batching strategy, not a guarantee that inference evaluates all questions in one pass. The run yields **115,960 machine-counted supervision targets**:

| Question Head | Question Type | Rows Present | Ground Truth Breakdown | Purpose & Learning Task |
| :--- | :--- | :--- | :--- | :--- |
| **`tool_route`** | `choice` | 23,192 | 23,192 valid closed-candidate selections + `__abstain__` | Primary tool routing decision across multi-protocol candidates |
| **`tool_route_perm`** | `choice` | 23,192 | 23,192 candidate order permutations | Enforces permutation invariance via Jensen-Shannon Divergence ($L_{JSD}$) |
| **`arg_mapping`** | `choice` | 23,192 | **16,457** active schema parameter mappings, **6,735** `__drop__` targets | Maps unrecognized/hallucinated LLM parameter keys to valid schema properties |
| **`is_mutation`** | `noul` (bool) | 23,192 | **5,700** `yes` (state-mutating), **17,492** `no` (read-only) | Classifies execution side-effect tier when MCP tool annotations are absent |
| **`missing_required`** | `noul` (bool) | 23,192 | **6,034** `yes` (missing required), **17,158** `no` (complete) | Guards against calling tools with incomplete required arguments |
| **Total** | — | **23,192** | **115,960 Supervision Targets** | Single-pass multi-task representation sharing |

Evaluation suites (`val_external_human.jsonl` with 1,200 held-out rows over 115 distinct base queries. **Note:** on the HF dataset this file is byte-identical to `val_external_heldout.jsonl`, so it is not an independent human-written split; and `val_zeroshot.jsonl` with 2,220 queries across 55 unseen packages) are strictly disjoint from training:
- **Package Overlap**: 0 packages (0 / 5 external, 0 / 55 zero-shot).
- **Exact Query Overlap**: 0 queries across all splits.
- **Near-Duplicate Queries**: 0 pairs at 3-gram Jaccard $\ge 0.75$, 1 pair at dense embedding cosine $\ge 0.88$.
- **Tool Name Overlap**: 4 generic tools (`check_service_status`, `list_processes`, `reverse_geocode`, `transfer_to_human_agents`) account for 162 rows (13.50%) in external holdout; 1,038 rows (86.50%) feature strictly disjoint tools whose names never appear in `train.jsonl`. Both subsets are reported separately below.
- **Audit Verification**: Committed in `scripts/verify_dataset_disjointness.py` and `benchmarks/disjointness_audit.json`.

---

## Evaluation Results

All models and baselines were evaluated on an **NVIDIA GeForce RTX 4050 Laptop GPU** (6GB VRAM, PCIe 4.0 x4, PyTorch 2.6.0, CUDA 12.4). The coverage and prediction-set metrics below come from the calibration/evaluation pipeline over **100 random splits** (`N_cal = 500`, nominal coverage `1 - alpha = 0.90`). They are distinct from the production top-mass gate.

### 1. Decision Model & Retrieval Baseline Comparison (Full 1,200 External Rows across 115 Unique Base Queries)

> **Note on Query Stratification**: The 1,200 evaluations comprise 115 unique base queries tested across ~10 format/distractor views
> (1,095 reference MCP rows + 105 $\tau^2$ persona-dialogue rows). Soro-Kev-4B achieves **96.89% (1,061/1,095)** on reference MCP tools
> and **24.76% (26/105)** on $\tau^2$, where 64.56% of misses are calibrated abstentions or preliminary status checks.

To investigate why classical retrievers fail on tool catalogs, all retrievers were evaluated with clean schema text parsing (unwrapping JSON/REST docstrings) and calibrated with an explicit abstention threshold $\tau_{abstain}$ tuned on calibration splits:

| Method / Architecture | Overall (1,200) | Non-Abstain (1,065) | Abstain Refusal (135)* | Strictly Disjoint Tools (1,038) | Shared Tool Names (162) | Conformal Coverage ($1 - \alpha = 0.90$) | Mean Set Size $\|S\|$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25 (fixed abstain τ)** | 58.67% (704/1200) | 62.72% (668/1065) | 26.67% (36/135) | 58.29% (605/1038) | 61.11% (99/162) | 84.1% ± 2.4% | 4.32 ± 0.21 |
| **Dense MiniLM-L6 cosine (fixed abstain τ)** | 79.17% (950/1200) | 84.51% (900/1065) | 37.04% (50/135) | 81.70% (848/1038) | 62.96% (102/162) | 90.4% ± 1.6% | 1.54 ± 0.10 |
| **NLI cross-encoder DeBERTa-v3-small (fixed abstain τ)** | 24.00% (288/1200) | 17.56% (187/1065) | 74.81% (101/135) | 26.88% (279/1038) | 5.56% (9/162) | 90.3% ± 1.7% | 4.83 ± 0.10 |
| **Qwen3.5-0.8B-Base option letters (untrained)** | 20.75% (249/1200) | 23.38% (249/1065) | 0.00% (0/135) | 20.13% (209/1038) | 24.69% (40/162) | 90.4% ± 1.7% | 5.06 ± 0.09 |
| **Soro-Kev-0.8B (Edge Checkpoint)** | **85.67% (1028/1200)** | **86.95% (926/1065)** | **75.56% (102/135)** | **88.63% (920/1038)** | **66.67% (108/162)** | **90.3% ± 1.6%** | **1.14 ± 0.04** |
| **Soro-Kev-4B (Flagship Checkpoint)** | **90.58% (1087/1200)** | **90.52% (964/1065)** | **91.11% (123/135)** | **92.77% (963/1038)** | **76.54% (124/162)** | **91.0% ± 1.1%** | **1.01 ± 0.02** |

*\*Note on Abstention: The 135 abstain rows represent 15 unique base queries evaluated across 9 format/distractor views each ($15 \times 9 = 135$). Denominators strictly reconcile: $1,065 + 135 = 1,200$. Numerators strictly reconcile: e.g. 0.8B gets $926 + 102 = 1,028$; 4B gets $964 + 123 = 1,087$. Dedicated adversarial refusal is evaluated on the 33-row BFCL v3 Irrelevance benchmark (Table 5).*

> **Pointer Network Guarantee**: The dual pointer head restricts selection strictly to candidate keys in context + `__abstain__`. It generates **0% invalid or invented tool names** by construction. True misclassification rate ($1 - \text{Top-1 Accuracy}$) is **9.42% (113/1200)** on external holdouts.

### 2. Why Classical Retrievers Fail on Tool Catalogs (Failure Analysis)

Hand-review of 20 failure cases reveals three systematic root causes for classical retriever weakness:
1. **Directional Semantic Inversion**: Queries like *"Give me the street address"* require `reverse_geocode` (takes coordinates $\to$ outputs address). `geocode_location` (takes address $\to$ outputs coordinates) contains the keyword "address" in its input parameters, causing BM25 and Dense retrievers to pick the wrong direction.
2. **Lexical Synonym Gaps**: User prompts like *"What is the current RAM consumption?"* or *"Change the file type of this image"* have 0 token overlap with tools named `get_memory_usage` ("Returns memory...") or `convert_image_format` ("Converts image format..."). Keyword matching scores 0.0 and falls back to incorrect tools or premature abstention.
3. **Prompt Instruction Noise**: Real user dialogue prompts (e.g. from Tau-2) include conversational instructions (*"If the agent suggests actions that don't fix it, express mild frustration..."*). Incidental words like "suggest", "read", or "plan" trigger false matches against unrelated tools like `read_environment_variable` or `change_plan`.

### 3. Per-Package Accuracy Table (External Holdout Breakdown)

| Package | Domain / Origin | Rows | BM25 | Dense MiniLM | NLI Cross-Encoder | Qwen Option Letters | Soro-Kev-0.8B | Soro-Kev-4B |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`mcp_benchmark/system_mcp`** | OS & Process Management | 370 | 64.59% (239/370) | 55.68% (206/370) | 63.24% (234/370) | 66.22% (245/370) | 87.57% (324/370) | **95.68% (354/370)** |
| **`mcp_benchmark/geo_mcp`** | Coordinates & Routing | 364 | 56.32% (205/364) | 51.92% (189/364) | 56.32% (205/364) | 55.77% (203/364) | 95.33% (347/364) | **98.35% (358/364)** |
| **`mcp_benchmark/media_mcp`** | Image Transform & Inspection | 361 | 56.51% (204/361) | 45.15% (163/361) | 57.62% (208/361) | 59.28% (214/361) | 95.84% (346/361) | **96.68% (349/361)** |
| **`tau2/telecom`** | Telecom Customer Service (MIT) | 85 | 52.94% (45/85) | 29.41% (25/85) | 30.59% (26/85) | 41.18% (35/85) | 9.41% (8/85) | **24.71% (21/85)** |
| **`tau2/retail`** | Retail Order Management (MIT) | 20 | 55.00% (11/20) | 60.00% (12/20) | 40.00% (8/20) | 50.00% (10/20) | 15.00% (3/20) | **25.00% (5/20)** |
| **Total External Held-Out** | All 5 Domains | 1,200 | 58.67% (704/1200) | 79.17% (950/1200) | 24.00% (288/1200) | 20.75% (249/1200) | 85.67% (1028/1200) | **90.58% (1087/1200)** |

*\*Note on Tau-2 persona simulation: Across the 1,095 reference MCP tools (system, geo, media), Soro-Kev-4B achieves 96.89% (1,061/1,095). On the 105 Tau-2 multi-turn dialogue instructions, Soro-Kev correctly refuses to blindly fire backend mutation tools without customer identifiers: 64.56% (51/79) of Soro-Kev-4B's misses were explicit abstentions (21 rows) or preliminary read/status lookups (30 rows, e.g. check_service_status, get_device_details) rather than premature account mutations.*

### 4. Unseen Tool Package Split (2,220 evaluations across 55 held-out packages)

> **Note on Queries & Schemas**: Queries share structural template distributions with training while all 55 tool schemas
> and packages are strictly disjoint (0 package overlap with the 319 training packages).

| System | Overall Zero-Shot (2,220 rows) | Clean Menus (2,168 rows) | Collision Distractors (52 rows) |
| :--- | :---: | :---: | :---: |
| **BM25 (fixed abstain τ)** | 48.60% (1079/2220) | 49.40% (1071/2168) | 15.38% (8/52) |
| **Dense MiniLM-L6 cosine (fixed abstain τ)** | 62.48% (1387/2220) | 62.82% (1362/2168) | 48.08% (25/52) |
| **NLI cross-encoder DeBERTa-v3-small (fixed abstain τ)** | 13.60% (302/2220) | 13.75% (298/2168) | 7.69% (4/52) |
| **Qwen3.5-0.8B-Base option letters (untrained)** | 17.57% (390/2220) | 17.57% (381/2168) | 17.31% (9/52) |
| **Soro-Kev-0.8B (Edge Checkpoint)** | **81.67% (1813/2220)** | **81.83% (1774/2168)** | **75.00% (39/52)** |
| **Soro-Kev-4B (Flagship Checkpoint)** | **88.06% (1955/2220)** | **88.28% (1914/2168)** | **78.85% (41/52)** |
| **Arch-Router-1.5B (Katanemo)** | - | - | 25.00% (13/52) |
| **xLAM-2-1b-fc-r (Salesforce)** | - | - | 69.23% (36/52) |
| **Hammer2.1-1.5b (MadeAgents)** | - | - | 59.62% (31/52) |

### 5. Berkeley Function-Calling Leaderboard (BFCL v3 Diagnostic, 100 rows)

| System | Multiple Tools (34) | Live Multiple (33) | Irrelevance / Abstention (33) | Overall Diagnostic (100) | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BM25 (fixed abstain τ)** | 94.12% (32/34) | 57.58% (19/33) | 36.36% (12/33) | 63.00% (63/100) | 0.1 ms |
| **Dense MiniLM-L6 cosine (fixed abstain τ)** | 97.06% (33/34) | 84.85% (28/33) | 24.24% (8/33) | 69.00% (69/100) | 19.6 ms |
| **NLI cross-encoder DeBERTa-v3-small (fixed abstain τ)** | 55.88% (19/34) | 30.30% (10/33) | 33.33% (11/33) | 40.00% (40/100) | 317.0 ms |
| **Qwen3.5-0.8B-Base option letters (untrained)** | 44.12% (15/34) | 33.33% (11/33) | 0.00% (0/33) | 26.00% (26/100) | 456.8 ms |
| **Soro-Kev-0.8B (Edge Checkpoint)** | **100.00% (34/34)** | **96.97% (32/33)** | **51.52% (17/33)** | **83.00% (83/100)** | **124.2 ms** |
| **Soro-Kev-4B (Flagship Checkpoint)** | **100.00% (34/34)** | **96.97% (32/33)** | **75.76% (25/33)** | **91.00% (91/100)** | **648.6 ms** |
| **Arch-Router-1.5B (Katanemo)** | 11.76% (4/34) | 15.15% (5/33) | 96.97% (32/33) | 41.00% (41/100) | 8465.4 ms |
| **xLAM-2-1b-fc-r (Salesforce)** | 79.41% (27/34) | 84.85% (28/33) | 30.30% (10/33) | 65.00% (65/100) | 11453.6 ms |
| **Hammer2.1-1.5b (MadeAgents)** | 97.06% (33/34) | 93.94% (31/33) | 27.27% (9/33) | 73.00% (73/100) | 10564.5 ms |

*\*Hardware & Latency Note (RTX 4050 6GB Laptop GPU, Windows WDDM):*
- **Architecture Difference (Prefill vs. Autoregressive Decode):** Soro-Kev outputs routing + key-mapping decisions scoring candidate tools and an explicit `__abstain__` anchor via a dual linear pointer head in **1 single prefill forward pass** (124.2 ms for 0.8B, 648.6 ms for 4B) with 0 generated decode tokens. In contrast, generative models (`xLAM-2-1b-fc-r`, `Hammer2.1-1.5b`) generate full JSON tool calls with argument values (~27–32 decode tokens), requiring 1 prefill pass plus 24-32 sequential autoregressive decode steps (one forward pass per token).
- **In-VRAM vs. Sustained / Paged Latency:** In clean isolated VRAM bursts, `Arch-Router-1.5B` achieves a median latency of **1,212.9 ms** (~24 tokens generated). `xLAM-2-1b-fc-r` (mean prompt 406 tokens, mean gen 27 tokens) achieves **7,106.9 ms** median in clean VRAM (initial queries: ~1,458-2,549 ms; laptop GPU thermal throttling at 85C settles to ~7.1s). `Hammer2.1-1.5b` (generating full markdown JSON arrays of ~32 tokens) settles to **10,564.5 ms** median. When multiple generative models were evaluated sequentially without intermediate process garbage collection, Windows WDDM silently paged KV caches into Shared GPU Memory across PCIe, increasing latencies to ~10.4s-11.5s.

### 6. Accuracy by Schema Format (Independently Sourced Holdout)

- **Agent Skills (`SKILL.md`)**: 100.0% (29/29)
- **GraphQL RPC**: 96.8% (30/31)
- **REST / OpenAPI**: 96.2% (25/26)
- **Mixed / Heterogeneous Catalogs**: 94.6% (35/37)
- **MCP JSON-RPC**: 94.4% (34/36)
- **Python Signatures**: 87.8% (36/41)

---

## Architectural Trade-Offs & v0.3 Roadmap

### 1. Left-to-Right Causal Attention Asymmetry
In autoregressive backbones (`Qwen/Qwen3-4B`), attention is strictly causal (unidirectional). When tools are serialized in a single sequence:
$$\text{[Query]} \dots \text{[Tool 1]} \dots \text{[Tool 2]} \dots \text{[Tool } N\text{]} \dots \text{[__abstain__]}$$
Tokens for Tool $N$ attend to the query and all preceding tools ($1 \dots N-1$), but Tool 1 cannot attend to subsequent tools. While training includes an auxiliary permutation consistency term (JSD) to enforce position invariance, causal asymmetry can flip the top choice when candidates are reordered, mostly on closely matched tools. The flip rate has not been measured with a committed script; the 4B model was trained without the JSD term.

### 2. Context & Latency Scaling
Causal prefill complexity scales with the total token length of serialized tool schemas. While compact schemas fit comfortably within context (16 tools $\approx$ 1,200 tokens), scaling to 100+ schemas requires two-stage tournament chunking or lexical pre-filtering.

### 3. v0.3 Architectural Direction: Bidirectional Distillation + Set Transformer
To eliminate causal order sensitivity and achieve true $O(1)$ permutation invariance, the `v0.3` development branch focuses on:
1. **Bidirectional Encoder Backbone**: Distilling `Soro-Kev-4B` knowledge into a compact 149M bidirectional cross-encoder (e.g. ModernBERT or DeBERTa-v3).
2. **Permutation-Invariant Set Transformer Head**: Replacing causal span projections with an Induced Set Attention Block (ISAB) operating over pooled candidate tool representations, guaranteeing mathematical permutation invariance ($f(\pi(T)) = \pi(f(T))$).
3. **Sub-15ms Edge Inference**: Combining INT8/ONNX runtime quantization with bidirectional pooling for single-digit millisecond routing on commodity CPU and edge hardware.

---

## Provenance and Reproducibility

- **Repository**: [github.com/nisaral/Soro-Boro](https://github.com/nisaral/Soro-Boro) (release tag `v0.2.1`)
- **Trained Checkpoint**: Adapter weights and `head.pt` with learned temperature `0.070878` and calibration quantile `q_hat = 0.8066`
- **Harness Dataset**: [huggingface.co/datasets/nisaralll/soro-universal-tool-harness](https://huggingface.co/datasets/nisaralll/soro-universal-tool-harness)
- **Disjointness Audit**: Committed in `benchmarks/disjointness_audit.json`

---

## Limitations (read before citing these numbers)

- **115 distinct queries behind 1,200 rows.** The external held-out set is 115 distinct base queries, each shown in about 10 format and distractor views. Rows within a query are correlated, so row-level CIs are too narrow. Cluster-bootstrap 95% CIs over the 115 queries (`scripts/per_query_ci.py` in the GitHub repo, 10,000 resamples):

  | System | Row accuracy | Wilson 95% (rows, too narrow) | Cluster bootstrap 95% (queries) |
  | :--- | :---: | :---: | :---: |
  | Soro-Kev-4B | 90.58% (1087/1200) | 88.8–92.1% | **85.9–94.7%** |
  | Soro-Kev-0.8B | 85.67% (1028/1200) | 83.6–87.5% | **80.3–90.6%** |
  | Dense MiniLM-L6 cosine | 79.17% (950/1200) | 76.8–81.4% | 74.3–83.7% |
  | BM25 | 58.67% (704/1200) | 55.9–61.4% | 51.7–65.4% |
  | NLI cross-encoder (DeBERTa-v3-small) | 24.00% (288/1200) | 21.7–26.5% | 19.7–28.4% |
  | Qwen3.5-0.8B-Base option letters | 20.75% (249/1200) | 18.6–23.1% | 18.5–23.0% |

- **τ² is weak.** On the 105 τ²-bench rows (telecom and retail), accuracy is **10.5% (11/105)** for 0.8B and **24.8% (26/105)** for 4B. The headline numbers come almost entirely from the 1,095 MCP-benchmark rows (92.9% / 96.9%). Do not expect τ²-style customer-service routing to work well yet.
- **No-tool refusal regressed at 0.8B.** On the 33 BFCL irrelevance rows, Soro-Kev-0.8B abstains correctly on **17/33**. Soro-Kev-0.8B is trained from base `Qwen3.5-0.8B-Base` (Kev's architecture, not Kev's weights), and its training data has too few hard no-tool rows. Soro-Kev-4B gets 25/33. If false tool calls are costly, use 4B or raise `q_hat`.
- **BFCL scope.** The BFCL table is a 100-row diagnostic (34 multiple, 33 live-multiple, 33 irrelevance, from BFCL v3). It is not a full BFCL run or a leaderboard submission.
- **Conformal threshold.** `Router.route()` and `SoroHarness.route()` apply the LAC set rule with the checkpoint's `q_hat`. That value was fitted on a 100-row slice of `val_zeroshot`, so coverage is **not guaranteed** on your traffic. Refitting q̂ (α = 0.10) on the committed external-set logs gives 0.9546 for 0.8B and 0.4825 for 4B. The model cards list `external_human_q_hat` as 0.9813 and 0.845, and the reason for the difference is unresolved. Over 200 grouped 50/50 splits, the refit q̂ for 4B ranges from 0.06 to 0.87, so a 115-query set is too small to pin it down. Recalibrate on your own logs (`scripts/recalibrate_qhat.py` in the GitHub repo).
- **No independent human split.** On the HF dataset, `val_external_human.jsonl` is byte-identical to `val_external_heldout.jsonl`. Read any "human" result as a held-out result. See the dataset notes in the GitHub repo.
- **Data licenses.** The data is not all Apache-2.0. See [DATA_LICENSES.md](https://github.com/nisaral/Soro-Boro/blob/main/DATA_LICENSES.md).
- **Head weights format.** `head.pt` is a pickle. It is loaded with `torch.load(weights_only=True)`. New checkpoints also write `head.safetensors`, which the loader prefers. Convert an old checkpoint with `soro-boro convert-head path/to/ckpt`.

## Citations

If you use Soro-Kev or Universal Tool Harness in your research or production systems, please cite the following publications:

```bibtex
@software{nisar2026sorokev,
  author = {Nisar, Keyush},
  title = {Soro-Kev: Calibration-Aware System-One Tool Router and Execution Boundary},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/nisaral/Soro-Boro}},
  note = {Release v0.2.1}
}

@misc{palmer2026kev08b,
  author = {Palmer, Jared},
  title = {Kev-0.8B: a calibrated decision model on Qwen3.5-0.8B},
  year = {2026},
  howpublished = {\url{https://huggingface.co/jaredpalmer/kev-0.8b}},
  note = {Kev 1.0}
}

@inproceedings{shi2025toolret,
  author = {Shi, Zhengliang and Gao, Shen and Chen, Xiuying and Wang, Lingzhi and Pei, Jiahuan and Guan, Xinyu and Ren, Zhaochun},
  title = {Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2025},
  pages = {11624--11639},
  year = {2025},
  url = {https://aclanthology.org/2025.findings-acl.647/}
}

@inproceedings{mishra2026mcpsemantic,
  author = {Mishra, Amritansh and Panditrao, Yash and Parab, Shlok and Sharma, Achintya and Mishra, Payal and Saha, Ishani},
  title = {Scalable Semantic Routing and QoS-Aware Provider Selection for MCP Agents},
  booktitle = {Procedia Computer Science, International Conference on Machine Learning and Data Engineering (ICMLDE 2026)},
  year = {2026},
  howpublished = {\url{https://github.com/amrittmishra/mcp-semantic-tool-routing-benchmark}}
}

@misc{mo2025livemcpbench,
  author = {Mo, Guozhao and others},
  title = {LiveMCPBench: Can Agents Navigate an Ocean of MCP Tools?},
  howpublished = {Preprint},
  year = {2025}
}

@inproceedings{ross2025when2call,
  author = {Ross, Hayley and Mahabaleshwarkar, Ameya Sunil and Suhara, Yoshi},
  title = {When2Call: When (not) to Call Tools},
  booktitle = {Proceedings of the 2025 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)},
  year = {2025}
}
```
