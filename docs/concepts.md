# Concepts

## Routing is classification, not generation

The Soro-Kev models read the query and the tool menu in **one prefill**. They score each tool, plus a
special `__abstain__` option, with a pointer head (`q · k / (√256 · T)`). Nothing is decoded, so the output
is always one of your tool names or `__abstain__`.

## Menus and chunking

- **≤ 16 tools:** one forward pass over the whole menu.
- **> 16 tools:** the menu is split into 6-tool chunks (each with `__abstain__`). The top 2 per chunk go to a
  championship pass. Cost grows with menu size, roughly one pass per chunk plus one.

Attention is causal, so tool order can matter a little. The 0.8B model was trained with a
permutation-consistency (JSD) loss; the 4B model was not (`single_sequence` on a T4).

## Calibration: the conformal gate

Given probabilities `p`, the router forms the LAC prediction set `S = {tool : p_tool ≥ 1 − q̂}`:

| S | Action |
| :--- | :--- |
| exactly one real tool | `EXECUTE` that tool |
| two or more real tools | `ASK`: `candidates` lists them, most likely first |
| empty, or `__abstain__` is top-1 | `ABSTAIN` |

If q̂ is fitted on exchangeable data at α = 0.10, S contains the right answer about 90% of the time.
**That guarantee holds only for data like the calibration data.** The repo's own refits
(`benchmarks/recalibration.json`) show q̂ moving a lot between datasets and between splits of the
115-query external set. Recalibrate on a few hundred distinct queries from your own traffic.

## Tool formats

`normalize_tools()` turns any supported format into `ToolSpec(name, description, parameters, source)`.
Supported formats are OpenAI (chat and flat), MCP, Anthropic, LangChain, LlamaIndex, `(name, desc)` tuples,
`{name: desc}` mappings and Python callables (type hints become a JSON Schema). Duplicate names and the
reserved `__abstain__` name raise `ToolSchemaError`.

## Argument repair

`decision.repair_args(args)` maps keys to the chosen tool's schema. It handles case and `snake_case`/`camelCase`
differences, coerces simple types (`"3"` → `3`), fills defaults, and drops keys the schema does not define. It does **not** invent
missing required values.

## Router vs SoroHarness

- `soro_boro.Router` decides. It is stateless, fast to construct, and takes any tool format.
- `soro.harness.pipeline.SoroHarness` executes. It registers decorated Python functions and adds
  execution-boundary checks, a write-ahead log, and a duplicate-mutation guard (`@soro_wrap`).
