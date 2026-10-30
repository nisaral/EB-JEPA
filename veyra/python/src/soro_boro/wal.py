"""Soro-Boro: Write-Ahead Logger (WAL) and Tamper-Evident Durability Boundary.

Guarantees:
1. Synchronous flush of INTENT before tool dispatch.
2. Synchronous flush of OUTCOME upon completion or deferral.
3. Startup reconciliation scanner for crash recovery (detecting unacknowledged in-flight writes).
4. SHA-256 hash chaining across events.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

GENESIS_HASH = "0000000000000000000000000000000000000000000000000000000000000000"


@dataclass
class WALRecord:
    event_type: str  # "INTENT" | "OUTCOME" | "RECONCILIATION" | "RECOVERED"
    operation_id: str
    idempotency_key: str
    tool_name: str
    payload: dict[str, Any]
    timestamp: float
    prev_hash: str
    record_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WriteAheadLog:
    """Thread-safe append-only Write-Ahead Log with hash chaining and crash recovery."""

    def __init__(self, log_path: str | Path | None = None):
        self.log_path = Path(log_path or os.environ.get("SORO_BORO_WAL") or "runs/soro_boro_wal.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last_hash = self._init_last_hash()

    def _init_last_hash(self) -> str:
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            return GENESIS_HASH

        last = GENESIS_HASH
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        last = data.get("record_hash", last)
        except Exception:
            pass
        return last

    def _compute_hash(self, prev_hash: str, payload_str: str) -> str:
        return hashlib.sha256(f"{prev_hash}:{payload_str}".encode()).hexdigest()

    def record_intent(
        self,
        operation_id: str,
        idempotency_key: str,
        tool_name: str,
        arguments: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> WALRecord:
        """Synchronously persist INTENT before any external side effect."""
        with self._lock:
            ts = time.time()
            data_to_hash = {
                "event": "INTENT",
                "op_id": operation_id,
                "key": idempotency_key,
                "tool": tool_name,
                "args": arguments,
                "ts": ts,
                "meta": metadata or {},
            }
            payload_str = json.dumps(data_to_hash, sort_keys=True, default=str)
            rec_hash = self._compute_hash(self._last_hash, payload_str)

            record = WALRecord(
                event_type="INTENT",
                operation_id=operation_id,
                idempotency_key=idempotency_key,
                tool_name=tool_name,
                payload={"arguments": arguments, "metadata": metadata or {}},
                timestamp=ts,
                prev_hash=self._last_hash,
                record_hash=rec_hash,
            )

            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict()) + "\n")
                f.flush()
                os.fsync(f.fileno())

            self._last_hash = rec_hash
            return record

    def record_outcome(
        self,
        operation_id: str,
        idempotency_key: str,
        tool_name: str,
        status: str,
        provider_state: str,
        reason: str,
        data: Any = None,
        attempts: int = 1,
    ) -> WALRecord:
        """Synchronously persist OUTCOME after tool execution or resolution."""
        with self._lock:
            ts = time.time()
            data_to_hash = {
                "event": "OUTCOME",
                "op_id": operation_id,
                "key": idempotency_key,
                "tool": tool_name,
                "status": status,
                "provider_state": provider_state,
                "reason": reason,
                "ts": ts,
                "attempts": attempts,
            }
            payload_str = json.dumps(data_to_hash, sort_keys=True, default=str)
            rec_hash = self._compute_hash(self._last_hash, payload_str)

            record = WALRecord(
                event_type="OUTCOME",
                operation_id=operation_id,
                idempotency_key=idempotency_key,
                tool_name=tool_name,
                payload={
                    "status": status,
                    "provider_state": provider_state,
                    "reason": reason,
                    "data": data,
                    "attempts": attempts,
                },
                timestamp=ts,
                prev_hash=self._last_hash,
                record_hash=rec_hash,
            )

            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict()) + "\n")
                f.flush()
                os.fsync(f.fileno())

            self._last_hash = rec_hash
            return record

    def find_orphaned_intents(
        self, max_retention_window_sec: float | None = None
    ) -> tuple[list[WALRecord], list[WALRecord]]:
        """Scan WAL on startup to detect any INTENT without a matching terminal OUTCOME.
        
        Returns:
            (recoverable_orphans, expired_orphans)
            If an orphan is older than max_retention_window_sec (e.g. 24h key pruning window),
            it CANNOT be safely replayed and must be routed to human triage / AMBIGUOUS_DEFERRED.
        """
        now = time.time()
        with self._lock:
            if not self.log_path.exists():
                return [], []

            intents: dict[str, WALRecord] = {}
            outcomes: set[str] = set()

            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        op_id = d["operation_id"]
                        if d["event_type"] == "INTENT":
                            rec = WALRecord(**d)
                            intents[op_id] = rec
                        elif d["event_type"] == "OUTCOME":
                            outcomes.add(op_id)
                    except Exception:
                        continue

            recoverable: list[WALRecord] = []
            expired: list[WALRecord] = []

            for op_id, rec in intents.items():
                if op_id not in outcomes:
                    if max_retention_window_sec and (now - rec.timestamp) > max_retention_window_sec:
                        expired.append(rec)
                    else:
                        recoverable.append(rec)

            return recoverable, expired
