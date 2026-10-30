"""Soro-Boro: Cross-Process Atomic Concurrency Claim Store.

Guarantees:
1. Prevents two parallel agent actions or racing retries from executing the same operation_id simultaneously.
2. Cross-process mutual exclusion via atomic insert-if-absent using SQLite with BEGIN IMMEDIATE.
3. Leases with TTL to allow crash takeover if a worker process dies mid-mutation.
"""

from __future__ import annotations

import enum
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


class ConcurrentExecutionBlocked(Exception):
    """Raised when an operation is already being executed by another process/worker."""
    pass


class ClaimStatus(str, enum.Enum):
    IN_FLIGHT = "IN_FLIGHT"
    COMMITTED = "COMMITTED"
    DEFERRED = "DEFERRED"
    FAILED = "FAILED"


class SQLiteClaimStore:
    """Cross-process, thread-safe atomic claim store backed by SQLite."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = str(db_path or os.environ.get("SORO_BORO_CLAIMS_DB") or "runs/soro_boro_claims.db")
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 10000")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS claims (
                    operation_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL,
                    owner_token TEXT NOT NULL,
                    claimed_at REAL NOT NULL,
                    lease_ttl_sec REAL NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT
                )
                """
            )
            conn.commit()

    def acquire_claim(
        self,
        operation_id: str,
        idempotency_key: str,
        owner_token: str,
        lease_ttl_sec: float = 30.0,
        window_sec: float | None = None,
    ) -> tuple[bool, str, dict[str, Any] | None]:
        """Atomically acquire claim for an operation.
        
        Returns:
            (acquired: bool, status: str, cached_result: dict | None)
        Raises:
            ConcurrentExecutionBlocked if an active unexpired lease is held by another owner.
        """
        now = time.time()
        with self._lock, self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                "SELECT owner_token, claimed_at, lease_ttl_sec, status, result_json FROM claims WHERE operation_id = ?",
                (operation_id,),
            )
            row = cursor.fetchone()

            if row is None:
                # Fresh operation -> Claim atomically
                conn.execute(
                    """
                    INSERT INTO claims (operation_id, idempotency_key, owner_token, claimed_at, lease_ttl_sec, status, result_json)
                    VALUES (?, ?, ?, ?, ?, ?, NULL)
                    """,
                    (operation_id, idempotency_key, owner_token, now, lease_ttl_sec, ClaimStatus.IN_FLIGHT.value),
                )
                conn.commit()
                return True, ClaimStatus.IN_FLIGHT.value, None

            existing_owner, claimed_at, existing_ttl, status, result_json = row
            cached_data = json.loads(result_json) if result_json else None

            if status == ClaimStatus.COMMITTED.value:
                # If window_sec is provided and the commit is older than window_sec, allow fresh execution
                if window_sec is not None and window_sec > 0 and (now - claimed_at) >= window_sec:
                    conn.execute(
                        """
                        UPDATE claims
                        SET owner_token = ?, claimed_at = ?, lease_ttl_sec = ?, status = ?, result_json = NULL
                        WHERE operation_id = ?
                        """,
                        (owner_token, now, lease_ttl_sec, ClaimStatus.IN_FLIGHT.value, operation_id),
                    )
                    conn.commit()
                    return True, ClaimStatus.IN_FLIGHT.value, None

                conn.commit()
                return False, ClaimStatus.COMMITTED.value, cached_data

            if status == ClaimStatus.IN_FLIGHT.value:
                # Check if lease has expired
                is_expired = (now - claimed_at) >= existing_ttl
                if not is_expired and existing_owner != owner_token:
                    conn.rollback()
                    raise ConcurrentExecutionBlocked(
                        f"Concurrency collision: Operation '{operation_id}' is currently IN_FLIGHT "
                        f"(claimed by '{existing_owner}' at {claimed_at}, TTL {existing_ttl}s). "
                        f"Parallel execution blocked to prevent race condition."
                    )

                # Lease expired or same owner -> Take over lease
                conn.execute(
                    """
                    UPDATE claims
                    SET owner_token = ?, claimed_at = ?, lease_ttl_sec = ?, status = ?
                    WHERE operation_id = ?
                    """,
                    (owner_token, now, lease_ttl_sec, ClaimStatus.IN_FLIGHT.value, operation_id),
                )
                conn.commit()
                return True, ClaimStatus.IN_FLIGHT.value, None

            # DEFERRED or FAILED -> Allowed to retry
            conn.execute(
                """
                UPDATE claims
                SET owner_token = ?, claimed_at = ?, lease_ttl_sec = ?, status = ?
                WHERE operation_id = ?
                """,
                (owner_token, now, lease_ttl_sec, ClaimStatus.IN_FLIGHT.value, operation_id),
            )
            conn.commit()
            return True, ClaimStatus.IN_FLIGHT.value, None

    def release_claim(
        self,
        operation_id: str,
        owner_token: str,
        status: ClaimStatus | str,
        result_data: Any = None,
    ) -> None:
        """Update terminal status and release claim."""
        status_val = status.value if isinstance(status, ClaimStatus) else str(status)
        result_json = json.dumps(result_data, default=str) if result_data is not None else None

        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                UPDATE claims
                SET status = ?, result_json = ?
                WHERE operation_id = ? AND owner_token = ?
                """,
                (status_val, result_json, operation_id, owner_token),
            )
            conn.commit()
