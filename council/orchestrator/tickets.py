"""Ticket model + SQLite CRUD for goals, tickets, and agent status."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class TicketState(str, Enum):
    NEW = "NEW"
    READY = "READY"
    IN_PROGRESS = "IN_PROGRESS"
    BLOCKED = "BLOCKED"
    REVIEW = "REVIEW"
    DONE = "DONE"


class TicketPriority(str, Enum):
    BIG = "BIG"
    SMALL = "SMALL"


class GoalStatus(str, Enum):
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


@dataclass
class Ticket:
    id: int = 0
    goal_id: int = 0
    title: str = ""
    description: str = ""
    acceptance: str = ""
    state: TicketState = TicketState.NEW
    priority: TicketPriority = TicketPriority.SMALL
    owner: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""
    depends_on: list[int] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    blocked_reason: Optional[str] = None
    review_notes: Optional[str] = None
    attempts_total: int = 0
    attempts_by_strategy: dict[str, int] = field(default_factory=dict)
    last_strategy_used: Optional[str] = None
    last_failure_reason: Optional[str] = None
    ticket_cooldown_until: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "goal_id": self.goal_id,
            "title": self.title,
            "description": self.description,
            "acceptance": self.acceptance,
            "state": self.state.value if isinstance(self.state, TicketState) else self.state,
            "priority": self.priority.value if isinstance(self.priority, TicketPriority) else self.priority,
            "owner": self.owner,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "depends_on": self.depends_on,
            "artifacts": self.artifacts,
            "blocked_reason": self.blocked_reason,
            "review_notes": self.review_notes,
            "attempts_total": self.attempts_total,
            "attempts_by_strategy": self.attempts_by_strategy,
            "last_strategy_used": self.last_strategy_used,
            "last_failure_reason": self.last_failure_reason,
            "ticket_cooldown_until": self.ticket_cooldown_until,
        }


@dataclass
class Goal:
    id: int = 0
    text: str = ""
    created_at: str = ""
    status: GoalStatus = GoalStatus.ACTIVE


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS goals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    status TEXT DEFAULT 'ACTIVE'
);

CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    goal_id INTEGER NOT NULL REFERENCES goals(id),
    title TEXT NOT NULL,
    description TEXT DEFAULT '',
    acceptance TEXT DEFAULT '',
    state TEXT DEFAULT 'NEW',
    priority TEXT DEFAULT 'SMALL',
    owner TEXT,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    depends_on TEXT DEFAULT '[]',
    artifacts TEXT DEFAULT '[]',
    blocked_reason TEXT,
    review_notes TEXT,
    attempts_total INTEGER DEFAULT 0,
    attempts_by_strategy TEXT DEFAULT '{}',
    last_strategy_used TEXT,
    last_failure_reason TEXT,
    ticket_cooldown_until TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    type TEXT NOT NULL,
    ticket_id INTEGER,
    agent TEXT,
    payload_json TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS agents (
    name TEXT PRIMARY KEY,
    status TEXT DEFAULT 'IDLE',
    last_seen TEXT,
    cooldown_until TEXT,
    current_ticket INTEGER
);
"""


class TicketStore:
    """SQLite-backed store for goals, tickets, events, and agent status."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(SCHEMA_SQL)
            self._ensure_ticket_columns_locked()
            self._conn.commit()

    def _ensure_ticket_columns_locked(self) -> None:
        """Add new ticket columns in-place for existing DBs (non-destructive migration)."""
        cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(tickets)").fetchall()
        }
        alters = [
            ("attempts_total", "ALTER TABLE tickets ADD COLUMN attempts_total INTEGER DEFAULT 0"),
            ("attempts_by_strategy", "ALTER TABLE tickets ADD COLUMN attempts_by_strategy TEXT DEFAULT '{}'"),
            ("last_strategy_used", "ALTER TABLE tickets ADD COLUMN last_strategy_used TEXT"),
            ("last_failure_reason", "ALTER TABLE tickets ADD COLUMN last_failure_reason TEXT"),
            ("ticket_cooldown_until", "ALTER TABLE tickets ADD COLUMN ticket_cooldown_until TEXT"),
        ]
        for name, ddl in alters:
            if name not in cols:
                self._conn.execute(ddl)

    def close(self) -> None:
        self._conn.close()

    # ── Goal CRUD ──

    def create_goal(self, text: str) -> Goal:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO goals (text) VALUES (?)", (text,)
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT * FROM goals WHERE id = ?", (cur.lastrowid,)
            ).fetchone()
        return self._row_to_goal(row)

    def get_goal(self, goal_id: int) -> Optional[Goal]:
        row = self._conn.execute(
            "SELECT * FROM goals WHERE id = ?", (goal_id,)
        ).fetchone()
        return self._row_to_goal(row) if row else None

    def update_goal_status(self, goal_id: int, status: GoalStatus) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE goals SET status = ? WHERE id = ?",
                (status.value, goal_id),
            )
            self._conn.commit()

    def list_goals(self, status: Optional[GoalStatus] = None) -> list[Goal]:
        if status:
            rows = self._conn.execute(
                "SELECT * FROM goals WHERE status = ? ORDER BY id DESC",
                (status.value,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM goals ORDER BY id DESC"
            ).fetchall()
        return [self._row_to_goal(r) for r in rows]

    # ── Ticket CRUD ──

    def create_ticket(
        self,
        goal_id: int,
        title: str,
        description: str = "",
        acceptance: str = "",
        priority: TicketPriority = TicketPriority.SMALL,
        depends_on: Optional[list[int]] = None,
    ) -> Ticket:
        deps = json.dumps(depends_on or [])
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO tickets (goal_id, title, description, acceptance, priority, depends_on)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (goal_id, title, description, acceptance, priority.value, deps),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT * FROM tickets WHERE id = ?", (cur.lastrowid,)
            ).fetchone()
        return self._row_to_ticket(row)

    def get_ticket(self, ticket_id: int) -> Optional[Ticket]:
        row = self._conn.execute(
            "SELECT * FROM tickets WHERE id = ?", (ticket_id,)
        ).fetchone()
        return self._row_to_ticket(row) if row else None

    def list_tickets(
        self,
        goal_id: Optional[int] = None,
        state: Optional[TicketState] = None,
        owner: Optional[str] = None,
    ) -> list[Ticket]:
        clauses: list[str] = []
        params: list[Any] = []
        if goal_id is not None:
            clauses.append("goal_id = ?")
            params.append(goal_id)
        if state is not None:
            clauses.append("state = ?")
            params.append(state.value)
        if owner is not None:
            clauses.append("owner = ?")
            params.append(owner)
        where = " AND ".join(clauses)
        sql = "SELECT * FROM tickets"
        if where:
            sql += f" WHERE {where}"
        sql += " ORDER BY id"
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_ticket(r) for r in rows]

    def update_ticket_state(self, ticket_id: int, state: TicketState) -> None:
        with self._lock:
            self._conn.execute(
                """UPDATE tickets SET state = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
                   WHERE id = ?""",
                (state.value, ticket_id),
            )
            self._conn.commit()

    def assign_ticket(self, ticket_id: int, owner: str) -> None:
        with self._lock:
            self._conn.execute(
                """UPDATE tickets SET owner = ?, state = 'READY',
                   updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
                   WHERE id = ?""",
                (owner, ticket_id),
            )
            self._conn.commit()

    def block_ticket(self, ticket_id: int, reason: str) -> None:
        with self._lock:
            self._conn.execute(
                """UPDATE tickets SET state = 'BLOCKED', blocked_reason = ?,
                   updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
                   WHERE id = ?""",
                (reason, ticket_id),
            )
            self._conn.commit()

    def complete_ticket(self, ticket_id: int, artifacts: Optional[list[str]] = None) -> None:
        arts = json.dumps(artifacts or [])
        with self._lock:
            self._conn.execute(
                """UPDATE tickets SET state = 'DONE', artifacts = ?,
                   updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
                   WHERE id = ?""",
                (arts, ticket_id),
            )
            self._conn.commit()

    def set_ticket_review(self, ticket_id: int, notes: str) -> None:
        with self._lock:
            self._conn.execute(
                """UPDATE tickets SET state = 'REVIEW', review_notes = ?,
                   updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
                   WHERE id = ?""",
                (notes, ticket_id),
            )
            self._conn.commit()

    def unassign_ticket(self, ticket_id: int) -> None:
        """Remove owner and reset to NEW for reassignment."""
        with self._lock:
            self._conn.execute(
                """UPDATE tickets SET owner = NULL, state = 'NEW',
                   updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
                   WHERE id = ?""",
                (ticket_id,),
            )
            self._conn.commit()


    def record_attempt(
        self,
        ticket_id: int,
        strategy_used: str,
        success: bool,
        failure_reason: str = "",
    ) -> None:
        """Track attempt counters and last strategy/failure context."""
        with self._lock:
            row = self._conn.execute(
                "SELECT attempts_total, attempts_by_strategy FROM tickets WHERE id = ?",
                (ticket_id,),
            ).fetchone()
            if not row:
                return
            total = int(row["attempts_total"] or 0) + 1
            by_strategy = json.loads(row["attempts_by_strategy"] or "{}")
            by_strategy[strategy_used] = int(by_strategy.get(strategy_used, 0)) + 1
            self._conn.execute(
                """UPDATE tickets
                   SET attempts_total = ?,
                       attempts_by_strategy = ?,
                       last_strategy_used = ?,
                       last_failure_reason = ?,
                       updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
                   WHERE id = ?""",
                (
                    total,
                    json.dumps(by_strategy),
                    strategy_used,
                    "" if success else failure_reason,
                    ticket_id,
                ),
            )
            self._conn.commit()

    def set_ticket_cooldown(self, ticket_id: int, cooldown_until: str) -> None:
        with self._lock:
            self._conn.execute(
                """UPDATE tickets SET ticket_cooldown_until = ?,
                   updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
                   WHERE id = ?""",
                (cooldown_until, ticket_id),
            )
            self._conn.commit()

    def clear_ticket_cooldown(self, ticket_id: int) -> None:
        self.set_ticket_cooldown(ticket_id, "")

    def is_ticket_available(self, ticket: Ticket) -> bool:
        value = (ticket.ticket_cooldown_until or "").strip()
        if not value:
            return True
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return datetime.now(timezone.utc) >= dt
        except ValueError:
            return True

    def mark_needs_rethink(self, ticket_id: int, reason: str) -> None:
        self.block_ticket(ticket_id, f"NEEDS_RETHINK: {reason}")

    # ── Agent status CRUD ──

    def upsert_agent(self, name: str, status: str = "IDLE") -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO agents (name, status, last_seen)
                   VALUES (?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                   ON CONFLICT(name) DO UPDATE SET
                   status = excluded.status, last_seen = excluded.last_seen""",
                (name, status),
            )
            self._conn.commit()

    def set_agent_status(self, name: str, status: str, current_ticket: Optional[int] = None) -> None:
        with self._lock:
            self._conn.execute(
                """UPDATE agents SET status = ?, current_ticket = ?,
                   last_seen = strftime('%Y-%m-%dT%H:%M:%fZ','now')
                   WHERE name = ?""",
                (status, current_ticket, name),
            )
            self._conn.commit()

    def set_agent_cooldown(self, name: str, cooldown_until: str) -> None:
        with self._lock:
            self._conn.execute(
                """UPDATE agents SET status = 'COOLDOWN', cooldown_until = ?,
                   last_seen = strftime('%Y-%m-%dT%H:%M:%fZ','now')
                   WHERE name = ?""",
                (cooldown_until, name),
            )
            self._conn.commit()

    def get_agent(self, name: str) -> Optional[dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM agents WHERE name = ?", (name,)
        ).fetchone()
        return dict(row) if row else None

    def list_agents(self) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM agents ORDER BY name").fetchall()
        return [dict(r) for r in rows]

    def get_flow_metrics(self, goal_id: Optional[int]) -> dict[str, Any]:
        """Compute Kanban-style flow metrics for a goal."""
        if goal_id is None:
            return {
                "total": 0,
                "done": 0,
                "blocked": 0,
                "wip": 0,
                "avg_lead_seconds": 0.0,
                "avg_cycle_seconds": 0.0,
                "blocked_seconds_total": 0.0,
            }

        tickets = self.list_tickets(goal_id=goal_id)
        if not tickets:
            return {
                "total": 0,
                "done": 0,
                "blocked": 0,
                "wip": 0,
                "avg_lead_seconds": 0.0,
                "avg_cycle_seconds": 0.0,
                "blocked_seconds_total": 0.0,
            }

        total = len(tickets)
        done_tickets = [t for t in tickets if t.state == TicketState.DONE]
        blocked = sum(1 for t in tickets if t.state == TicketState.BLOCKED)
        wip_states = {TicketState.READY, TicketState.IN_PROGRESS, TicketState.REVIEW}
        wip = sum(1 for t in tickets if t.state in wip_states)

        with self._lock:
            rows = self._conn.execute(
                """SELECT e.ticket_id, e.type, e.ts, e.payload_json
                   FROM events e
                   JOIN tickets t ON t.id = e.ticket_id
                   WHERE t.goal_id = ? AND e.ticket_id IS NOT NULL
                   ORDER BY e.id ASC""",
                (goal_id,),
            ).fetchall()

        first_in_progress: dict[int, datetime] = {}
        done_ts: dict[int, datetime] = {}
        blocked_since: dict[int, datetime] = {}
        blocked_by_ticket_s: dict[int, float] = {}
        blocked_total_s = 0.0
        now = datetime.now(timezone.utc)

        for row in rows:
            tid = int(row["ticket_id"])
            etype = row["type"] or ""
            ts = self._parse_ts(row["ts"])
            if ts is None:
                continue
            payload = json.loads(row["payload_json"] or "{}")

            if etype == "TICKET_STATE_CHANGED" and str(payload.get("state")) == "IN_PROGRESS":
                first_in_progress.setdefault(tid, ts)
                if tid in blocked_since:
                    blocked_delta = max(0.0, (ts - blocked_since.pop(tid)).total_seconds())
                    blocked_total_s += blocked_delta
                    blocked_by_ticket_s[tid] = blocked_by_ticket_s.get(tid, 0.0) + blocked_delta
                continue

            if etype in {"TICKET_BLOCKED", "BLOCKED"}:
                if tid in blocked_since:
                    blocked_delta = max(0.0, (ts - blocked_since[tid]).total_seconds())
                    blocked_total_s += blocked_delta
                    blocked_by_ticket_s[tid] = blocked_by_ticket_s.get(tid, 0.0) + blocked_delta
                blocked_since[tid] = ts
                continue

            if etype in {"TICKET_DONE", "DONE"}:
                done_ts.setdefault(tid, ts)
                if tid in blocked_since:
                    blocked_delta = max(0.0, (ts - blocked_since.pop(tid)).total_seconds())
                    blocked_total_s += blocked_delta
                    blocked_by_ticket_s[tid] = blocked_by_ticket_s.get(tid, 0.0) + blocked_delta
                continue

            if etype in {"NEEDS_REVIEW", "TICKET_ASSIGNED"} and tid in blocked_since:
                blocked_delta = max(0.0, (ts - blocked_since.pop(tid)).total_seconds())
                blocked_total_s += blocked_delta
                blocked_by_ticket_s[tid] = blocked_by_ticket_s.get(tid, 0.0) + blocked_delta

        for tid, started in blocked_since.items():
            blocked_delta = max(0.0, (now - started).total_seconds())
            blocked_total_s += blocked_delta
            blocked_by_ticket_s[tid] = blocked_by_ticket_s.get(tid, 0.0) + blocked_delta

        lead_durations: list[float] = []
        for t in done_tickets:
            created = self._parse_ts(t.created_at)
            finished = done_ts.get(t.id)
            if created and finished:
                dt = (finished - created).total_seconds()
                if dt >= 0:
                    lead_durations.append(dt)

        avg_lead_seconds = sum(lead_durations) / len(lead_durations) if lead_durations else 0.0

        cycle_durations: list[float] = []
        for t in done_tickets:
            start = first_in_progress.get(t.id)
            end = done_ts.get(t.id)
            if start and end:
                dt = (end - start).total_seconds() - blocked_by_ticket_s.get(t.id, 0.0)
                if dt >= 0:
                    cycle_durations.append(dt)

        avg_cycle_seconds = sum(cycle_durations) / len(cycle_durations) if cycle_durations else 0.0

        return {
            "total": total,
            "done": len(done_tickets),
            "blocked": blocked,
            "wip": wip,
            "avg_lead_seconds": avg_lead_seconds,
            "avg_cycle_seconds": avg_cycle_seconds,
            "blocked_seconds_total": blocked_total_s,
        }

    # ── Helpers ──

    @staticmethod
    def _parse_ts(value: str) -> Optional[datetime]:
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            return None

    @staticmethod
    def _row_to_goal(row: sqlite3.Row) -> Goal:
        return Goal(
            id=row["id"],
            text=row["text"],
            created_at=row["created_at"] or "",
            status=GoalStatus(row["status"]),
        )

    @staticmethod
    def _row_to_ticket(row: sqlite3.Row) -> Ticket:
        return Ticket(
            id=row["id"],
            goal_id=row["goal_id"],
            title=row["title"],
            description=row["description"] or "",
            acceptance=row["acceptance"] or "",
            state=TicketState(row["state"]),
            priority=TicketPriority(row["priority"]),
            owner=row["owner"],
            created_at=row["created_at"] or "",
            updated_at=row["updated_at"] or "",
            depends_on=json.loads(row["depends_on"] or "[]"),
            artifacts=json.loads(row["artifacts"] or "[]"),
            blocked_reason=row["blocked_reason"],
            review_notes=row["review_notes"],
            attempts_total=int(row["attempts_total"] or 0),
            attempts_by_strategy=json.loads(row["attempts_by_strategy"] or "{}"),
            last_strategy_used=row["last_strategy_used"],
            last_failure_reason=row["last_failure_reason"],
            ticket_cooldown_until=row["ticket_cooldown_until"],
        )
