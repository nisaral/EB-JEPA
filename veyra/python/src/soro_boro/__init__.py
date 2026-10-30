"""Soro-Boro: Execution Boundary for Consequential AI Agent Writes.

Prevents unsafe retries of consequential agent writes and gives teams
a clear, auditable outcome when the result is uncertain.
"""

from soro_boro.boundary import SoroBoundary, TimeoutException, soro_wrap
from soro_boro.identity import OperationIdentity
from soro_boro.lifecycle import (
    ProviderLifecycleState,
    ReconciliationResult,
    SideEffectClass,
    TerminalOutcome,
)
from soro_boro.wal import WALRecord, WriteAheadLog

__version__ = "0.2.1"
__all__ = [
    "SoroBoundary",
    "TimeoutException",
    "soro_wrap",
    "OperationIdentity",
    "ProviderLifecycleState",
    "ReconciliationResult",
    "SideEffectClass",
    "TerminalOutcome",
    "WriteAheadLog",
    "WALRecord",
    "SoroHarness",
    "Router",
    "RoutingDecision",
    "Action",
    "ToolSpec",
    "normalize_tools",
    "RouterLoadError",
]


def __getattr__(name: str):
    if name == "SoroHarness":
        from soro.harness.pipeline import SoroHarness
        return SoroHarness
    if name in ("Router", "RoutingDecision", "Action", "RouterLoadError", "Scorer"):
        from soro_boro import router

        return getattr(router, name)
    if name in ("ToolSpec", "normalize_tools", "normalize_tool"):
        from soro_boro import tool_schema

        return getattr(tool_schema, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
