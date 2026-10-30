"""The examples/ scripts and the installed `soro-boro` CLI must run offline (heuristic scorer, no weights)."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples"


def _run(*cmd, cwd=ROOT):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr
    return [json.loads(line) for line in p.stdout.splitlines() if line.startswith("{")]


def test_openai_prerouter():
    out = _run(sys.executable, str(EX / "openai_prerouter.py"))
    assert out[0]["action"] == "EXECUTE" and out[0]["tool"] == "get_weather"
    assert out[0]["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}
    assert out[0]["n_tools_sent"] == 1
    assert out[1]["action"] in {"EXECUTE", "ASK"}


def test_mcp_gateway_repairs_args():
    out = _run(sys.executable, str(EX / "mcp_gateway.py"))
    call = out[0]["call"]
    assert call["server"] == "github" and call["params"]["name"] == "list_pull_requests"
    assert call["params"]["arguments"] == {"repo": "nisaral/Soro-Boro"}  # "Repo" -> "repo"


def test_framework_adapters():
    out = _run(sys.executable, str(EX / "framework_adapters.py"))
    assert [o["framework"] for o in out] == ["langchain_tools", "llamaindex_tools"]
    assert all(o["tool"] == "multiply" and o["result"] == 42 for o in out)


def test_gateway_importable():
    sys.path.insert(0, str(EX))
    try:
        import mcp_gateway
        from soro_boro import Router
        res = mcp_gateway.gateway(Router.heuristic(), "create a github issue titled bug in repo a/b", {"title": "bug", "repo": "a/b"})
        assert res["action"] in {"EXECUTE", "ASK"}
        assert "create_issue" in res["candidates"]
    finally:
        sys.path.remove(str(EX))


@pytest.fixture()
def tools_file(tmp_path):
    f = tmp_path / "tools.json"
    f.write_text(json.dumps([
        {"name": "get_weather", "description": "Get the current weather for a city",
         "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}}}},
        {"name": "send_email", "description": "Send an email to a recipient"}]))
    return f


def _cli():
    exe = shutil.which("soro-boro")
    return [exe] if exe else [sys.executable, "-m", "soro_boro.cli"]


def test_cli_route_heuristic(tools_file):
    p = subprocess.run([*_cli(), "route", "weather in Paris", "--tools", str(tools_file), "--heuristic",
                        "--args", '{"City": "Paris"}'], capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr
    assert '"get_weather"' in p.stdout and "EXECUTE" in p.stdout


def test_cli_eval_heuristic(tools_file, tmp_path):
    data = tmp_path / "eval.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in [
        {"query": "weather in Paris", "expected": "get_weather"},
        {"query": "email bob the report", "expected": "send_email"}]))
    p = subprocess.run([*_cli(), "eval", str(data), "--tools", str(tools_file), "--heuristic"],
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr
    assert "accuracy" in p.stdout.lower()


def test_cli_doctor():
    p = subprocess.run([*_cli(), "doctor"], capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr
