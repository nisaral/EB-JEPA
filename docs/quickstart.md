# Quickstart

## Install

```bash
pip install soro-boro                 # router + CLI (needs PyTorch)
pip install "soro-boro[server]"       # + FastAPI server
```

From source: `git clone https://github.com/nisaral/Soro-Boro && cd Soro-Boro && pip install -e ".[dev]"`.

## 1. Route a query

```python
from soro_boro import Router

tools = [
    {"name": "get_weather", "description": "Get the current weather for a city",
     "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}},
    {"name": "send_email", "description": "Send an email to a recipient"},
]
router = Router.heuristic()     # offline demo scorer; use Router.from_pretrained(...) for real routing
decision = router.route("what's the weather in Paris?", tools)
assert decision.action == "EXECUTE" and decision.tool == "get_weather"
```

With the real model, which downloads from the Hugging Face Hub on first use and runs on CPU or CUDA:

```python
router = Router.from_pretrained("nisaralll/soro-kev-0.8b")   # or nisaralll/soro-kev-4b (GPU recommended)
```

## 2. Act on the decision

```python
if decision.action == "EXECUTE":
    args = decision.repair_args({"City": "Paris"})   # -> {"city": "Paris"}
elif decision.action == "ASK":
    print("Did you mean:", decision.candidates)
else:  # ABSTAIN
    print("No tool fits; answer directly.")
```

## 3. Tune how often it acts

`q_hat` controls the gate. The threshold is `1 - q_hat`. A **lower** `q_hat` means a stricter threshold:
more `ABSTAIN`, fewer wrong calls. The checkpoint ships a calibrated value (0.9518 for 0.8B, 0.8066 for 4B).

```python
strict = router.route("tell me a joke", tools, q_hat=0.1)
```

Calibrate on your own logs with `scripts/recalibrate_qhat.py`. See [concepts](concepts.md#calibration-the-conformal-gate).

## 4. CLI and server

```bash
soro-boro route "weather in Paris" --tools tools.json --args '{"City": "Paris"}'
soro-boro eval my_eval.jsonl --tools tools.json         # rows: {"query", "expected", "tools"?}
soro-boro serve --port 8765                             # GET /healthz, POST /route, POST /v1/route
curl -s localhost:8765/route -d '{"query": "weather in Paris", "tools": [{"name": "get_weather"}]}' \
     -H 'content-type: application/json'
```

Add `--heuristic` to any command to run without weights. `soro-boro doctor --check-model` verifies the
model loads.

Next: [examples/](../examples) · [API reference](API.md) · [FAQ](FAQ.md)
