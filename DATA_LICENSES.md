# Data licenses

The Soro-Kev training and evaluation data (`veyra/data/soro_kev_dataset/`, mirrored at
`nisaralll/soro-universal-tool-harness` on Hugging Face) mixes rows converted from upstream
sources. **The Apache-2.0 tag on this repository covers the code. It does not automatically cover
every upstream row.** When you redistribute the data, follow each upstream license below.

Row counts are machine-counted from the committed JSONL files (`source` / `package` fields).

| Upstream source | Used in | Rows (train / val_zeroshot / external) | License | Status |
| :--- | :--- | :--- | :--- | :--- |
| τ²-bench (`tau2_airline`, `tau2_telecom`, `tau2_retail`) | train, external held-out | 1,436 / 0 / 105 | MIT | stated upstream |
| MCP Semantic Tool Routing Benchmark (`mcp-semantic-tool-routing-benchmark`) | train, val_zeroshot, external held-out | 8,792 / 1,313 / 1,095 | CC BY 4.0 (attribution required) | stated upstream |
| Berkeley Function-Calling Leaderboard (BFCL v3) | 100-row diagnostic eval only (`benchmarks/predictions/bfcl`) | 0 / 0 / 0 (eval only) | Apache-2.0 | stated upstream |
| MCPAgentBench (`third_party/MCPAgentBench` submodule) | train | 3,687 / 0 / 0 | **unknown** | TODO(Keyush): check the upstream repo's LICENSE |
| `catalog_paraphrase` (paraphrased queries over public MCP server tool catalogs, many packages) | train, val_zeroshot | remainder of train / val_zeroshot | **unknown / mixed**: generated queries, but the tool names and descriptions come from each MCP server, under that server's license | TODO(Keyush): record the generator model and its terms; spot-check the catalog licenses |
| Toucan (`external_intent.jsonl`) | intent head training | 8,000 (external_intent) | **unknown**, verify upstream | TODO(Keyush) |
| ToolACE (`external_intent.jsonl`) | intent head training | 7,770 (external_intent) | **unknown**, verify upstream | TODO(Keyush) |

## Attribution

- MCP Semantic Tool Routing Benchmark: CC BY 4.0 requires attribution and an indication of changes.
  Rows were reformatted into the Soro-Kev `{state, questions}` schema and mixed with distractor
  tools. Cite the upstream benchmark when you use these rows.
- τ²-bench: MIT. Keep the copyright notice when you redistribute the converted rows.
- BFCL: Apache-2.0. Only predictions on 100 rows are committed here, not the BFCL data.

## Known data issues

- On the HF dataset, `val_external_human.jsonl` is byte-identical to `val_external_heldout.jsonl`.
  There is no independent human-written split. See `veyra/data/soro_kev_dataset/DATA_NOTES.md`.
- The 1,200 external held-out rows cover **115 distinct base queries**, each shown in about 10
  format and distractor views. See README › Limitations.
