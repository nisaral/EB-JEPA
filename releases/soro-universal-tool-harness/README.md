---
license: apache-2.0
task_categories:
  - text-classification
tags:
  - tool-calling
  - function-calling
  - mcp
  - model-context-protocol
  - agent-harness
  - conformal-prediction
  - pointer-network
  - decision-model
size_categories:
  - 10K<n<100K
---

# Soro Universal Tool Harness Dataset

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Models](https://img.shields.io/badge/Trained_Models-Soro--Kev--4B_&_0.8B-purple.svg)](https://huggingface.co/nisaralll/soro-kev-4b)
[![GitHub](https://img.shields.io/badge/GitHub-Soro--Boro-black.svg)](https://github.com/nisaral/Soro-Boro)

The Soro Universal Tool Harness dataset contains high-quality, multi-format routing, argument repair, and safety classification records designed for training and calibrating System-One pointer routers.

Unlike standard function-calling datasets that train models to emit token-by-token JSON strings, this dataset formats agent tasks as closed-set pointer queries over explicit option lists. That closed output space prevents invented tool names. The dataset also supports calibration and split-conformal evaluation, but a coverage guarantee depends on the training and serving implementation; it is not created by the dataset alone.

---

## Files & Split Breakdown

| Split File | Rows | Supervision Targets | Package Count | Real-World & Held-Out Description |
| :--- | :--- | :--- | :--- | :--- |
| **`train.jsonl`** | 23,192 | 115,960 | 319 packages | Diverse real-world MCP, REST, GraphQL, Agent Skills, and Python signatures. 5,853 rows (25.2%) feature delexicalized option keys (`op_N`). |
| **`val_zeroshot.jsonl`** | 2,220 | 11,100 | 55 held-out pkgs | Strictly disjoint tool schemas and packages never seen during training (0 package overlap with train). Contains 52 collision distractors. |
| **`val_external_heldout.jsonl`** | 1,200 | 6,000 | 5 held-out pkgs | 115 unique base queries evaluated across ~10 format/distractor views: 1,095 reference MCP server tool rows + 105 $\tau^2$ multi-turn customer persona rows. |

> **Known issue: no independent human split.** The repo also contains `val_external_human.jsonl`. It is byte-identical to
> `val_external_heldout.jsonl` and is **not** a separate set of human-written queries. Treat it as a duplicate.
>
> **Licenses:** the `license: apache-2.0` tag covers the packaging. Upstream rows keep their own licenses
> (τ²-bench MIT, MCP Semantic Tool Routing Benchmark CC BY 4.0, others unverified). See
> [DATA_LICENSES.md](https://github.com/nisaral/Soro-Boro/blob/main/DATA_LICENSES.md).

---

## Supported Schema Formats

The dataset balances 6 diverse tool declaration formats to prevent syntactic overfitting:

1. **`mcp_json`**: Model Context Protocol JSON-RPC tool schemas.
2. **`agent_skill`**: Agent Skills and `SKILL.md` plugin definitions with command formats and triggers.
3. **`rest_openapi`**: OpenAPI v3 REST endpoints (HTTP methods + path templates).
4. **`python_sig`**: Standard Python function signatures and type annotations.
5. **`graphql_rpc`**: GraphQL query/mutation RPC schemas.
6. **`mixed_heterogeneous`**: Real-world catalogs combining multiple formats simultaneously.

---

## Task Questions in Each Row

Each row provides a user state and query alongside options for:

1. **`tool_route`**: Pointer choice selecting the correct candidate tool, or `__abstain__` when the request cannot be fulfilled.
2. **`tool_route_perm`**: Permuted order of the same tool candidates, provided for permutation-consistency training. Its effect depends on whether the training path actually consumes the permutation sequence.
3. **`arg_mapping`**: Maps misnamed or stray argument keys back to canonical schema fields.
4. **`missing_required`**: Binary prediction of whether mandatory arguments are missing from user context.
5. **`is_mutation`**: Side-effect classification (whether the tool performs non-idempotent state mutations).
6. **`intent`**: Multi-tool intent detection (`single`, `multi`, or `none`).

---

## Loading with Hugging Face Datasets

```python
from datasets import load_dataset

dataset = load_dataset("nisaralll/soro-universal-tool-harness")
print(dataset["train"][0])
```

Or read directly with `json` in Python:

```python
import json
from huggingface_hub import hf_hub_download

file_path = hf_hub_download(
    repo_id="nisaralll/soro-universal-tool-harness",
    filename="val_external_heldout.jsonl",
    repo_type="dataset"
)

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        print("Query:", row["state"])
        print("Gold Tool:", row["gold_tool"])
        break
```

---

## Citation & Upstream Licenses

Released under the **Apache-2.0 License**.

This dataset compiles, normalizes, and evaluates against tool schemas and interaction traces from several open benchmark initiatives. We gratefully acknowledge and credit upstream works under their respective open-source licenses:

1. **$\tau^2$ (Tau-2) Benchmark** (Sierra Research / Tao et al., 2024):
   - License: **MIT License**
   - Source: [github.com/sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)
   - Used for multi-turn customer persona simulations and service domain tool definitions (`tau2/telecom`, `tau2/retail`).
2. **MCPAgentBench** (Mo et al., 2025):
   - License: **MIT License**
   - Source: [github.com/guozhao-mo/MCPAgentBench](https://github.com/guozhao-mo/MCPAgentBench)
   - Used for live FastMCP reference server tool schemas.
3. **MCP Semantic Tool Routing Benchmark** (Mishra et al., 2026):
   - License: **Creative Commons Attribution 4.0 International (CC BY 4.0)**
   - Source: [github.com/amrittmishra/mcp-semantic-tool-routing-benchmark](https://github.com/amrittmishra/mcp-semantic-tool-routing-benchmark)
   - Used for reference Model Context Protocol tool catalogs (`system_mcp`, `geo_mcp`, `media_mcp`).
4. **Berkeley Function-Calling Leaderboard (BFCL v3)** (Gorilla / Patil et al., 2024):
   - License: **Apache-2.0 License**
   - Source: [github.com/ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla)
   - Diagnostic slices (`multiple`, `live_multiple`, `irrelevance`) evaluated for tool selection and refusal.

```bibtex
@software{nisar2026sorokev,
  author = {Nisar, Keyush},
  title = {Soro Universal Tool Harness: Multi-Protocol Pointer Benchmark & Dataset},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/nisaralll/soro-universal-tool-harness}},
  note = {Version 1.0}
}
```
