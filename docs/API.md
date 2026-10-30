# API reference

Everything below is importable from `soro_boro`.

## `Router`

```python
from soro_boro import Router

router = Router.heuristic()                       # weights-free lexical scorer (demo/tests)
# Router.from_pretrained(model_id_or_path="nisaralll/soro-kev-0.8b", device="auto",
#                        q_hat=None, fallback="error")
# Router(scorer, q_hat=None, tools=None)          # any object with probabilities(query, options) -> dict
```

| Member | Description |
| :--- | :--- |
| `Router.from_pretrained(id_or_path, device="auto", q_hat=None, fallback="error")` | Load a Soro-Kev checkpoint (Hub id or local dir). `device` is `"auto"`, `"cpu"` or `"cuda"`. `fallback="heuristic"` warns and uses the lexical scorer instead of raising `RouterLoadError`. `q_hat=None` uses the checkpoint's calibrated value. |
| `Router.heuristic(q_hat=None)` | Weights-free scorer. For demos and tests only. |
| `Router(scorer, q_hat=None, tools=None)` | Bring your own scorer (see `Scorer`). `tools` sets a default menu. |
| `router.route(query, tools=None, q_hat=None) -> RoutingDecision` | Route one query. `tools` is any supported format; `q_hat` overrides the gate for this call. |
| `router.route_batch(queries, tools=None, q_hat=None) -> list[RoutingDecision]` | Route several queries against one menu. |
| `router.q_hat`, `router.backend` | Current gate value and scorer name. |

```python
tools = {"get_weather": "Get the current weather for a city", "send_email": "Send an email"}
decisions = router.route_batch(["weather in Rome", "email alice"], tools)
assert [d.tool for d in decisions] == ["get_weather", "send_email"]
```

## `RoutingDecision`

| Field / method | Type | Meaning |
| :--- | :--- | :--- |
| `action` | `Action` | `EXECUTE`, `ASK` or `ABSTAIN` (a `str` enum, so `d.action == "EXECUTE"` works) |
| `tool` | `str \| None` | Tool to run (EXECUTE), top suggestion (ASK), `None` (ABSTAIN) |
| `candidates` | `list[str]` | Tools in the conformal set, most likely first |
| `scores` | `dict[str, float]` | Probability per tool, including `__abstain__` |
| `confidence` | `float` | Probability of `tool` (or of `__abstain__`) |
| `q_hat`, `threshold` | `float` | Gate used; `threshold = 1 - q_hat` |
| `reason`, `backend` | `str` | Human-readable explanation and scorer name |
| `should_execute` | `bool` | `action == EXECUTE` |
| `repair_args(args, tool=None)` | `dict` | Arguments repaired against the chosen (or named) tool's schema |
| `to_dict()` | `dict` | JSON-serialisable form (what the server returns) |

```python
d = router.route("weather in Rome", {"get_weather": "Get the current weather for a city"})
print(d.to_dict()["action"])   # EXECUTE
```

## Tool schemas

```python
from soro_boro import ToolSpec, normalize_tools

specs = normalize_tools([
    {"type": "function", "function": {"name": "a", "description": "OpenAI chat format"}},
    {"name": "b", "inputSchema": {"type": "object"}},          # MCP
    {"name": "c", "input_schema": {"type": "object"}},         # Anthropic
])
assert [s.source for s in specs] == ["openai", "mcp", "anthropic"]
```

`normalize_tool(x)` converts one tool. `ToolSchemaError` is raised for unknown shapes, duplicate names,
or the reserved name `__abstain__`.

## `Scorer` protocol

```python
class MyScorer:
    q_hat = 0.9              # optional default gate
    backend_name = "mine"    # optional

    def probabilities(self, query: str, options: dict[str, str]) -> dict[str, float]:
        # options maps tool name -> description and includes "__abstain__"; return probabilities summing to 1
        return {k: 1 / len(options) for k in options}

uniform = Router(MyScorer())
```

## HTTP server (`pip install "soro-boro[server]"`)

`soro-boro serve [--model ID] [--host 127.0.0.1] [--port 8765]` or `soro_boro.server.create_app(router)`.

| Endpoint | Body | Returns |
| :--- | :--- | :--- |
| `GET /healthz` | none | `{"ok", "backend", "q_hat"}` |
| `POST /route` | `{"query", "tools", "q_hat"?}` | `RoutingDecision.to_dict()` |
| `POST /v1/route` | OpenAI-style `{"messages", "tools", "q_hat"?}` | `{"decision", "tool_choice"}`. `tool_choice` is a forced function on EXECUTE, `"auto"` on ASK, `"none"` on ABSTAIN. |

## CLI

`soro-boro route | eval | serve | doctor | convert-head`. Every command accepts `--model`, `--device`,
`--q-hat` and `--heuristic`. Run `soro-boro <cmd> --help` for details.

## Lower-level API

`soro.harness.pipeline.SoroHarness` handles registration, execution and the mutation guard. Calling
`SoroHarness.route(..., candidates=...)` is deprecated in favour of `Router`.
