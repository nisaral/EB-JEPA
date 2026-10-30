"""Repo-wide pytest fixtures: per-test state isolation and optional-resource skips."""


# ---------------------------------------------------------------------------
# Test isolation and optional-resource skips
# ---------------------------------------------------------------------------
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent
MCPAGENTBENCH_PRESENT = (_REPO_ROOT / "third_party" / "MCPAgentBench" / "servers").is_dir()

# Tests that need the third_party/MCPAgentBench git submodule.
_NEEDS_MCPAGENTBENCH = (
    "test_mcpagent.py::test_official_checkout_is_present",
    "test_bench_mcp_real.py::",
    "test_bench_llm_agent.py::test_llm_agent_comparative_benchmark_all_seeds",
    "test_bench_llm_agent.py::test_llm_agent_harness_single_task",
)


def pytest_collection_modifyitems(config, items):
    if MCPAGENTBENCH_PRESENT:
        return
    skip = pytest.mark.skip(
        reason="third_party/MCPAgentBench submodule not checked out "
        "(git submodule update --init third_party/MCPAgentBench)"
    )
    for item in items:
        if any(key in item.nodeid for key in _NEEDS_MCPAGENTBENCH):
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _isolated_soro_boro_state(tmp_path, monkeypatch):
    """Give every test its own claim DB and WAL.

    The boundary defaults to ``runs/soro_boro_claims.db`` in the working
    directory with a 120 s dedupe window, so without this a second test run
    within two minutes sees the first run's claims and reports
    PREVENTED_DUPLICATE.
    """
    monkeypatch.setenv("SORO_BORO_CLAIMS_DB", str(tmp_path / "claims.db"))
    monkeypatch.setenv("SORO_BORO_WAL", str(tmp_path / "wal.jsonl"))
    yield
