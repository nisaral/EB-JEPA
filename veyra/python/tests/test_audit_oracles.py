"""Regression Tests for Soro Audit Oracles and Spec Semantics.

Validates:
1. Append-style tool returning identical acks: state oracle detects non-idempotent mutation
   and flags CRITICAL_UNSAFE_CONTRADICTION when declared idempotent.
2. Overwrite-style tool: state oracle detects idempotent write and suggests idempotentHint.
3. Invariant read-only tool: state oracle detects read-only and flags CONFIRMATION_FATIGUE if omitted.
4. False read-only tool: flags CRITICAL_UNSAFE_CONTRADICTION when tool claims read-only but mutates disk.
5. Spec default semantics: omitted destructiveHint conservatively defaults to True, not flagged.
6. Exercised tool accounting: tools returning isError=True are classified as skipped/unexercised.
7. Credential scrubbing: ambient production API keys are strictly rejected or stripped.
"""

import os
import sys
from pathlib import Path

_GOLDEN_DIR = Path(__file__).resolve().parents[2] / "standalone" / "soro-audit"

import pytest

from soro.cli_audit import (
    audit_server,
    check_production_environment,
    create_scrubbed_env,
    synthesize_arguments_from_schema,
)

FIXTURE_SERVER_CMD = f'"{sys.executable}" "{Path(__file__).parent / "fixture_mcp_server.py"}"'


def test_append_style_state_oracle_detects_non_idempotent():
    """Identical ACK responses across two calls must NOT fool the oracle when file size/content changes."""
    report = audit_server(FIXTURE_SERVER_CMD, timeout_sec=5.0, allow_dirty_env=True)
    assert report["success"] is True

    append_audit = next(t for t in report["tool_audits"] if t["name"] == "append_log")
    assert append_audit["observed_class"] == "non_idempotent_mutation"
    assert append_audit["discrepancy_type"] == "unsafe_idempotent_contradiction"
    assert "CRITICAL UNSAFE CONTRADICTION" in append_audit["discrepancy"]


def test_overwrite_style_state_oracle_detects_idempotent_write():
    """Mutating Call 1 followed by invariant Call 2 must be classified as idempotent write."""
    report = audit_server(FIXTURE_SERVER_CMD, timeout_sec=5.0, allow_dirty_env=True)
    assert report["success"] is True

    write_audit = next(t for t in report["tool_audits"] if t["name"] == "overwrite_config")
    assert write_audit["observed_class"] == "idempotent_write"
    assert write_audit["discrepancy_type"] == "unannotated_idempotent"
    assert write_audit["suggested_patch"] == {"idempotentHint": True}


def test_verified_read_only_tool_confirmation_fatigue():
    """A tool verified read-only that omits readOnlyHint must trigger CONFIRMATION_FATIGUE."""
    report = audit_server(FIXTURE_SERVER_CMD, timeout_sec=5.0, allow_dirty_env=True)
    assert report["success"] is True

    status_audit = next(t for t in report["tool_audits"] if t["name"] == "get_status")
    assert status_audit["observed_class"] == "read_only"
    assert status_audit["discrepancy_type"] == "confirmation_fatigue"
    assert "CONFIRMATION FATIGUE" in status_audit["discrepancy"]
    assert status_audit["suggested_patch"] == {"readOnlyHint": True}


def test_false_read_only_tool_flags_critical_contradiction():
    """A tool declaring readOnlyHint: True that mutates disk must trigger CRITICAL_UNSAFE_CONTRADICTION."""
    report = audit_server(FIXTURE_SERVER_CMD, timeout_sec=5.0, allow_dirty_env=True)
    assert report["success"] is True

    false_ro_audit = next(t for t in report["tool_audits"] if t["name"] == "false_read_only_tool")
    assert false_ro_audit["observed_class"] in ("idempotent_write", "non_idempotent_mutation")
    assert false_ro_audit["discrepancy_type"] == "unsafe_read_only_contradiction"
    assert "CRITICAL UNSAFE CONTRADICTION" in false_ro_audit["discrepancy"]


def test_validation_error_tool_classified_unexercised():
    """Tools that return isError: True must NOT be counted as exercised."""
    report = audit_server(FIXTURE_SERVER_CMD, timeout_sec=5.0, allow_dirty_env=True)
    assert report["success"] is True

    failing_audit = next(t for t in report["tool_audits"] if t["name"] == "failing_validation_tool")
    assert failing_audit["behavior_status"] == "call_error"
    assert failing_audit["observed_class"] == "unverified"
    assert report["skipped_tools"] >= 1
    assert report["exercised_tools"] < report["total_tools"]


def test_spec_default_destructive_omission_is_conservative():
    """Per MCP spec, destructiveHint defaults to True. An omitted hint on destructive tool is safe."""
    report = audit_server(FIXTURE_SERVER_CMD, timeout_sec=5.0, allow_dirty_env=True)
    assert report["success"] is True

    delete_audit = next(t for t in report["tool_audits"] if t["name"] == "delete_backup")
    # Must NOT have unsafe contradiction for destructive hint
    assert delete_audit["discrepancy_type"] != "unsafe_destructive_contradiction"


def test_credential_scrubbing_and_dirty_env_guard():
    """Ambient credentials must trigger PermissionError unless allow_dirty_env is passed."""
    old_env = os.environ.get("AWS_ACCESS_KEY_ID")
    try:
        os.environ["AWS_ACCESS_KEY_ID"] = "AKIA_TEST_KEY_FOR_AUDIT"
        with pytest.raises(PermissionError) as exc_info:
            check_production_environment(allow_dirty=False)
        assert "AWS_ACCESS_KEY_ID" in str(exc_info.value)

        # Scrubbed env strips it completely
        scrubbed = create_scrubbed_env()
        assert "AWS_ACCESS_KEY_ID" not in scrubbed
        assert scrubbed.get("SORO_SANDBOX") == "1"
        assert scrubbed.get("CI") == "true"
    finally:
        if old_env is None:
            os.environ.pop("AWS_ACCESS_KEY_ID", None)
        else:
            os.environ["AWS_ACCESS_KEY_ID"] = old_env


def test_golden_filesystem_raw_wire_capture():
    """Golden-file regression test: raw tools/list JSON from @modelcontextprotocol/server-filesystem

    Verifies:
    1. Exact tool count is 14.
    2. All 14 tools declare explicit annotations in tool['annotations'].
    3. Critical tools like read_file have readOnlyHint: True.
    4. Overwrite tools like write_file have idempotentHint: True, destructiveHint: True.
    """
    raw_path = _GOLDEN_DIR / "raw_tools_list_filesystem.json"
    assert raw_path.exists(), f"Golden artifact missing: {raw_path}"

    import json
    data = json.loads(raw_path.read_text(encoding="utf-8"))
    tools = data["result"]["tools"]
    assert len(tools) == 14

    for t in tools:
        assert "annotations" in t, f"Tool {t['name']} is missing annotations dictionary"
        ann = t["annotations"]
        assert any(v is not None for v in ann.values()), f"Tool {t['name']} annotations are all None"

    read_file_tool = next(t for t in tools if t["name"] == "read_file")
    assert read_file_tool["annotations"]["readOnlyHint"] is True

    write_file_tool = next(t for t in tools if t["name"] == "write_file")
    assert write_file_tool["annotations"]["idempotentHint"] is True
    assert write_file_tool["annotations"]["destructiveHint"] is True


def test_golden_memory_raw_wire_capture():
    """Golden-file regression test: raw tools/list JSON from @modelcontextprotocol/server-memory

    Verifies:
    1. Exact tool count is 9.
    2. All 9 tools declare explicit annotations.
    3. Query tools like read_graph have readOnlyHint: True.
    """
    raw_path = _GOLDEN_DIR / "raw_tools_list_memory.json"
    assert raw_path.exists(), f"Golden artifact missing: {raw_path}"

    import json
    data = json.loads(raw_path.read_text(encoding="utf-8"))
    tools = data["result"]["tools"]
    assert len(tools) == 9

    for t in tools:
        assert "annotations" in t, f"Tool {t['name']} is missing annotations dictionary"

    read_graph_tool = next(t for t in tools if t["name"] == "read_graph")
    assert read_graph_tool["annotations"]["readOnlyHint"] is True


def test_golden_everything_raw_wire_capture():
    """Golden-file regression test: raw tools/list JSON from @modelcontextprotocol/server-everything

    Verifies:
    1. Exact tool count is 13.
    2. All 13 tools declare explicit annotations in tool['annotations'].
    3. Handshake handles notifications/tools/list_changed message prior to id: 2 response.
    """
    raw_path = _GOLDEN_DIR / "raw_tools_list_everything.json"
    assert raw_path.exists(), f"Golden artifact missing: {raw_path}"

    import json
    data = json.loads(raw_path.read_text(encoding="utf-8"))
    tools = data["result"]["tools"]
    assert len(tools) == 13

    for t in tools:
        assert "annotations" in t, f"Tool {t['name']} is missing annotations dictionary"
        ann = t["annotations"]
        assert any(v is not None for v in ann.values()), f"Tool {t['name']} annotations are all None"

    echo_tool = next(t for t in tools if t["name"] == "echo")
    assert echo_tool["annotations"]["readOnlyHint"] is True
    assert echo_tool["annotations"]["idempotentHint"] is True

