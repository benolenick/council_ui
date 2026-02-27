"""Event schema, SQLite logging, NDJSON writer, and agent event line parser."""

from __future__ import annotations

import json
import re
import threading
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class EventType(str, Enum):
    GOAL_CREATED = "GOAL_CREATED"
    GOAL_PARTIAL = "GOAL_PARTIAL"
    GOAL_EXECUTION_STARTED = "GOAL_EXECUTION_STARTED"
    TICKET_CREATED = "TICKET_CREATED"
    TICKET_ASSIGNED = "TICKET_ASSIGNED"
    TICKET_STATE_CHANGED = "TICKET_STATE_CHANGED"
    TICKET_BLOCKED = "TICKET_BLOCKED"
    TICKET_DONE = "TICKET_DONE"
    AGENT_STATUS_CHANGED = "AGENT_STATUS_CHANGED"
    AGENT_RATE_LIMITED = "AGENT_RATE_LIMITED"
    AGENT_OUTPUT = "AGENT_OUTPUT"
    EXECUTION_RESULT = "EXECUTION_RESULT"
    PROPOSAL_RECEIVED = "PROPOSAL_RECEIVED"
    SYNTHESIS_COMPLETE = "SYNTHESIS_COMPLETE"
    STORY_UPDATED = "STORY_UPDATED"
    HANDOFF_CREATED = "HANDOFF_CREATED"
    SAFETY_FLAGGED = "SAFETY_FLAGGED"
    SAFETY_APPROVED = "SAFETY_APPROVED"
    SAFETY_DENIED = "SAFETY_DENIED"
    ROUTER_LOG = "ROUTER_LOG"
    # Agent-emitted events
    CLAIM_TICKET = "CLAIM_TICKET"
    PROPOSE_TICKETS = "PROPOSE_TICKETS"
    BLOCKED = "BLOCKED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    DONE = "DONE"


# Regex for parsing EVENT lines from agent stdout
_EVENT_LINE_RE = re.compile(r"^EVENT:\s+(\w+)\s*(.*)?$", re.MULTILINE)
_KV_RE = re.compile(r'(\w+)=(?:"([^"]*?)"|(\S+))')


def parse_agent_event_line(line: str) -> Optional[dict[str, Any]]:
    """Parse a single EVENT: TYPE key=val line from agent output.

    Returns dict with 'type' and parsed key-value pairs, or None if not an event line.
    """
    m = _EVENT_LINE_RE.match(line.strip())
    if not m:
        return None
    event_type = m.group(1)
    rest = m.group(2) or ""
    payload: dict[str, Any] = {"type": event_type}
    for kv in _KV_RE.finditer(rest):
        key = kv.group(1)
        value = kv.group(2) if kv.group(2) is not None else kv.group(3)
        # Try numeric conversion
        try:
            value = int(value)
        except (ValueError, TypeError):
            pass
        payload[key] = value
    return payload


def extract_events_from_output(text: str) -> list[dict[str, Any]]:
    """Extract all EVENT: lines from agent output text."""
    events = []
    for line in text.splitlines():
        parsed = parse_agent_event_line(line)
        if parsed:
            events.append(parsed)
    return events


class EventStore:
    """Emit events to SQLite and an append-only NDJSON log file."""

    def __init__(self, db_conn, log_path: str | Path, lock: threading.Lock):
        """Initialize with a shared SQLite connection and lock from TicketStore.

        Args:
            db_conn: sqlite3.Connection (shared with TicketStore)
            log_path: Path to story_log.ndjson
            lock: threading.Lock (shared with TicketStore)
        """
        self._conn = db_conn
        self._lock = lock
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        event_type: EventType | str,
        ticket_id: Optional[int] = None,
        agent: Optional[str] = None,
        payload: Optional[dict[str, Any]] = None,
    ) -> int:
        """Record an event in SQLite and append to NDJSON log. Returns event id."""
        etype = event_type.value if isinstance(event_type, EventType) else event_type
        payload_json = json.dumps(payload or {})
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO events (type, ticket_id, agent, payload_json)
                   VALUES (?, ?, ?, ?)""",
                (etype, ticket_id, agent, payload_json),
            )
            self._conn.commit()
            event_id = cur.lastrowid

            # Fetch the timestamp assigned by SQLite
            row = self._conn.execute(
                "SELECT ts FROM events WHERE id = ?", (event_id,)
            ).fetchone()
            ts = row["ts"] if row else ""

        # Append to NDJSON log (best-effort, don't block on IO errors)
        try:
            ndjson_line = json.dumps({
                "id": event_id,
                "ts": ts,
                "type": etype,
                "ticket_id": ticket_id,
                "agent": agent,
                "payload": payload or {},
            })
            with open(self.log_path, "a") as f:
                f.write(ndjson_line + "\n")
        except OSError:
            pass

        return event_id

    def query(
        self,
        event_type: Optional[str] = None,
        ticket_id: Optional[int] = None,
        agent: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Query events with optional filters."""
        clauses: list[str] = []
        params: list[Any] = []
        if event_type:
            clauses.append("type = ?")
            params.append(event_type)
        if ticket_id is not None:
            clauses.append("ticket_id = ?")
            params.append(ticket_id)
        if agent:
            clauses.append("agent = ?")
            params.append(agent)
        where = " AND ".join(clauses)
        sql = "SELECT * FROM events"
        if where:
            sql += f" WHERE {where}"
        sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def tail(self, n: int = 50) -> list[dict[str, Any]]:
        """Get the N most recent events (newest first)."""
        rows = self._conn.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) as cnt FROM events").fetchone()
        return row["cnt"] if row else 0

    @staticmethod
    def _row_to_dict(row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "ts": row["ts"],
            "type": row["type"],
            "ticket_id": row["ticket_id"],
            "agent": row["agent"],
            "payload": json.loads(row["payload_json"] or "{}"),
        }
