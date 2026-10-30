"""Tests for the high-level soro_boro.Router API, CLI and HTTP server (no weights needed)."""
from __future__ import annotations

import json
import warnings
from types import SimpleNamespace

import pytest

from soro_boro import Action, Router, RouterLoadError, RoutingDecision, ToolSpec, normalize_tools
from soro_boro.tool_schema import ToolSchemaError, normalize_tool


class DummyScorer:
    """Deterministic scorer: fixed probability table keyed by query."""

    q_hat = 0.95  # threshold 0.05
    backend_name = "dummy"

    def __init__(self, table):
        self.table = table
        self.calls = []

    def probabilities(self, query, options):
        self.calls.append((query, dict(options)))
        probs = dict(self.table[query])
        return {k: probs.get(k, 0.0) for k in options}


TOOLS = [
    {"type": "function", "function": {"name": "check_plan", "description": "Check customer plan",
                                      "parameters": {"type": "object", "properties": {
                                          "customer_id": {"type": "string"},
                                          "limit": {"type": "integer", "default": 5}},
                                          "required": ["customer_id"]}}},
    {"name": "get_weather", "description": "Weather for a city", "inputSchema": {"type": "object"}},
    {"name": "refund", "description": "Refund a charge", "input_schema": {"type": "object"}},
]

TABLE = {
    "plan?": {"check_plan": 0.97, "get_weather": 0.01, "refund": 0.01, "__abstain__": 0.01},
    "ambiguous": {"check_plan": 0.5, "refund": 0.4, "get_weather": 0.06, "__abstain__": 0.04},
    "poem": {"__abstain__": 0.9, "check_plan": 0.06, "get_weather": 0.02, "refund": 0.02},
    "flat": {"check_plan": 0.03, "get_weather": 0.03, "refund": 0.03, "__abstain__": 0.03},
}


@pytest.fixture
def router():
    return Router(DummyScorer(TABLE))


def test_execute(router):
    d = router.route("plan?", tools=TOOLS)
    assert isinstance(d, RoutingDecision)
    assert d.action is Action.EXECUTE and d.should_execute
    assert d.tool == "check_plan" and d.candidates == ["check_plan"]
    assert d.confidence == pytest.approx(0.97)
    assert d.threshold == pytest.approx(0.05)
    assert "__abstain__" in d.scores
    assert d.to_dict()["action"] == "EXECUTE"
    assert str(d.action) == "EXECUTE"


def test_scorer_sees_all_tools_plus_abstain(router):
    router.route("plan?", tools=TOOLS)
    _, options = router.scorer.calls[-1]
    assert set(options) == {"check_plan", "get_weather", "refund", "__abstain__"}


def test_ask(router):
    d = router.route("ambiguous", tools=TOOLS)
    assert d.action is Action.ASK
    assert d.tool == "check_plan"
    assert d.candidates == ["check_plan", "refund", "get_weather"]


def test_abstain_top1(router):
    d = router.route("poem", tools=TOOLS)
    assert d.action is Action.ABSTAIN and d.tool is None and d.candidates == []
    assert d.confidence == pytest.approx(0.9)


def test_abstain_empty_set(router):
    d = router.route("flat", tools=TOOLS)
    assert d.action is Action.ABSTAIN


def test_q_hat_override_changes_gate(router):
    assert router.route("ambiguous", tools=TOOLS, q_hat=0.55).action is Action.EXECUTE  # threshold 0.45


def test_no_tools(router):
    assert router.route("plan?", tools=[]).action is Action.ABSTAIN


def test_bad_query(router):
    with pytest.raises(ValueError):
        router.route("   ", tools=TOOLS)


def test_bad_q_hat():
    with pytest.raises(ValueError):
        Router(DummyScorer(TABLE), q_hat=1.5)


def test_bad_scorer():
    with pytest.raises(TypeError):
        Router(scorer=object())


def test_default_tools():
    r = Router(DummyScorer(TABLE), tools=TOOLS)
    assert r.route("plan?").tool == "check_plan"


def test_repair_args(router):
    d = router.route("plan?", tools=TOOLS)
    assert d.repair_args({"customerId": 123, "limit": "3"}) == {"customer_id": "123", "limit": 3}
    assert d.repair_args({"customer_id": "9"})["limit"] == 5


def test_repair_args_requires_tool(router):
    with pytest.raises(ValueError):
        router.route("poem", tools=TOOLS).repair_args({"x": 1})


def test_route_batch(router):
    assert [d.action.value for d in router.route_batch(["plan?", "poem"], TOOLS)] == ["EXECUTE", "ABSTAIN"]


def test_heuristic_router_end_to_end():
    r = Router.heuristic()
    tools = {"check_plan": "Check customer current plan and data balance", "get_weather": "Get the weather for a city"}
    assert r.route("What plan is user 123 on?", tools=tools).tool == "check_plan"


def test_from_pretrained_bad_path_raises(tmp_path):
    with pytest.raises(RouterLoadError, match="soro-boro doctor"):
        Router.from_pretrained(str(tmp_path / "nope"), device="cpu")


def test_from_pretrained_fallback_heuristic(tmp_path):
    with pytest.warns(RuntimeWarning):
        r = Router.from_pretrained(str(tmp_path / "nope"), device="cpu", fallback="heuristic")
    assert "Heuristic" in r.backend


# ------------------------------------------------------------------ tool schemas
def test_normalize_formats():
    specs = normalize_tools(TOOLS)
    assert [s.source for s in specs] == ["openai", "mcp", "anthropic"]
    assert specs[0].required == ["customer_id"]
    assert specs[0].to_openai()["function"]["name"] == "check_plan"


def test_normalize_mapping_tuple_callable_and_objects():
    def lookup(order_id: int, verbose: bool = False):
        """Look up an order.

        Long text."""

    mcp_obj = SimpleNamespace(name="mcp_tool", description="d", inputSchema={"type": "object"})
    lc = SimpleNamespace(name="lc_tool", description="d", args_schema=None, func=lambda q: q)
    li = SimpleNamespace(metadata=SimpleNamespace(name="li_tool", description="d", fn_schema=None))
    specs = normalize_tools([("t", "desc"), lookup, mcp_obj, lc, li])
    assert [s.source for s in specs] == ["tuple", "callable", "mcp", "langchain", "llamaindex"]
    assert specs[1].description == "Look up an order."
    assert specs[1].properties["order_id"]["type"] == "integer"
    assert specs[1].required == ["order_id"]
    assert [s.name for s in normalize_tools({"a": "x", "b": {"description": "y"}})] == ["a", "b"]


def test_normalize_errors():
    with pytest.raises(ToolSchemaError):
        normalize_tools([{"name": "a"}, {"name": "a"}])
    with pytest.raises(ToolSchemaError):
        normalize_tools([{"name": "__abstain__"}])
    with pytest.raises(ToolSchemaError):
        normalize_tool({"description": "no name"})
    with pytest.raises(ToolSchemaError):
        normalize_tool(42)
    assert normalize_tool(ToolSpec("x")).name == "x"


# ------------------------------------------------------------------ CLI
@pytest.fixture
def tools_file(tmp_path):
    p = tmp_path / "tools.json"
    p.write_text(json.dumps({"tools": [
        {"name": "check_plan", "description": "Check customer current plan and data balance",
         "inputSchema": {"type": "object", "properties": {"customer_id": {"type": "string"}}}},
        {"name": "get_weather", "description": "Get the weather for a city"},
    ]}))
    return p


def test_cli_route(tools_file, capsys):
    from soro_boro.cli import main

    assert main(["route", "--heuristic", "--tools", str(tools_file), "What plan is user 123 on?",
                 "--args", '{"userId": 123}']) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["tool"] == "check_plan"
    assert out["repaired_args"] == {"customer_id": "123"}


def test_cli_eval(tools_file, tmp_path, capsys):
    from soro_boro.cli import main

    data = tmp_path / "eval.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in [
        {"query": "What plan is user 123 on?", "expected": "check_plan"},
        {"query": "weather in Paris city", "expected": "get_weather"},
    ]))
    assert main(["eval", "--heuristic", str(data), "--tools", str(tools_file)]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["rows"] == 2 and rep["top1_accuracy"] == 1.0


def test_cli_doctor(capsys):
    from soro_boro.cli import main

    assert main(["doctor"]) == 0
    assert "soro-boro" in capsys.readouterr().out


def test_cli_missing_file():
    from soro_boro.cli import main

    with pytest.raises(SystemExit):
        main(["route", "--heuristic", "--tools", "/nonexistent.json", "q"])


# ------------------------------------------------------------------ HTTP server
def test_server(router):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from soro_boro.server import create_app

    c = TestClient(create_app(router))
    assert c.get("/healthz").json()["ok"] is True
    r = c.post("/route", json={"query": "plan?", "tools": TOOLS}).json()
    assert r["action"] == "EXECUTE" and r["tool"] == "check_plan"
    r = c.post("/v1/route", json={"messages": [{"role": "user", "content": "poem"}], "tools": TOOLS}).json()
    assert r["tool_choice"] == "none"
    r = c.post("/v1/route", json={"messages": [{"role": "user", "content": [{"type": "text", "text": "plan?"}]}],
                                  "tools": TOOLS}).json()
    assert r["tool_choice"] == {"type": "function", "function": {"name": "check_plan"}}
    assert c.post("/route", json={"tools": TOOLS}).status_code == 400
    assert c.post("/route", json={"query": "plan?", "tools": [{"name": "a"}, {"name": "a"}]}).status_code == 400
    assert c.post("/v1/route", json={"messages": []}).status_code == 400


# ------------------------------------------------------------------ old API
def test_old_harness_candidates_warns():
    from soro import SoroHarness

    h = SoroHarness()
    with pytest.warns(DeprecationWarning, match="Router"):
        h.route("What plan is user 123 on?", candidates={"check_plan": "Check customer plan"})


def test_old_harness_registered_tools_no_warning():
    from soro import SoroHarness

    h = SoroHarness()

    @h.tool(description="Check customer current plan")
    def check_plan(customer_id: str):
        return {"id": customer_id}

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert h.route("What plan is user 123 on?").selected_tool == "check_plan"
