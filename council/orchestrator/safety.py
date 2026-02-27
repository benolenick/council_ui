"""Safety gate — destructive action detection + approval queue."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ApprovalState(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    DENIED = "DENIED"


@dataclass
class SafetyAlert:
    """A flagged destructive action awaiting approval."""
    id: int
    ticket_id: int
    agent: str
    matched_pattern: str
    matched_text: str
    full_context: str  # surrounding lines for user review
    state: ApprovalState = ApprovalState.PENDING


# Patterns that indicate destructive or dangerous operations
DESTRUCTIVE_PATTERNS = [
    # File system destruction
    re.compile(r'\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f|--force)\b.*', re.IGNORECASE),
    re.compile(r'\brm\s+-[a-zA-Z]*f[a-zA-Z]*r\b.*', re.IGNORECASE),
    re.compile(r'\brm\s+-rf\b', re.IGNORECASE),
    re.compile(r'\brmdir\b.*--ignore-fail', re.IGNORECASE),
    re.compile(r'\bsudo\s+rm\b', re.IGNORECASE),
    # Git destructive
    re.compile(r'\bgit\s+reset\s+--hard\b', re.IGNORECASE),
    re.compile(r'\bgit\s+push\s+--force\b', re.IGNORECASE),
    re.compile(r'\bgit\s+push\s+-f\b', re.IGNORECASE),
    re.compile(r'\bgit\s+clean\s+-[a-zA-Z]*f', re.IGNORECASE),
    # Database destruction
    re.compile(r'\bDROP\s+(TABLE|DATABASE|INDEX)\b', re.IGNORECASE),
    re.compile(r'\bTRUNCATE\s+TABLE\b', re.IGNORECASE),
    re.compile(r'\bDELETE\s+FROM\b.*\bWHERE\b.*=.*\b1\s*=\s*1\b', re.IGNORECASE),
    # System-level
    re.compile(r'\bsudo\b', re.IGNORECASE),
    re.compile(r'\bchmod\s+777\b', re.IGNORECASE),
    re.compile(r'\bchown\s+-R\b', re.IGNORECASE),
    re.compile(r'\bmkfs\b', re.IGNORECASE),
    re.compile(r'\bdd\s+if=', re.IGNORECASE),
    # Credentials / secrets
    re.compile(r'\b(password|secret|token|api_key)\s*=\s*["\'][^"\']+["\']', re.IGNORECASE),
    # Network
    re.compile(r'\bcurl\b.*\|\s*(bash|sh)\b', re.IGNORECASE),
    re.compile(r'\bwget\b.*\|\s*(bash|sh)\b', re.IGNORECASE),
]


class SafetyGate:
    """Scans agent output for destructive actions and manages an approval queue.

    When a destructive pattern is detected, the ticket is blocked until the user
    approves or denies the action via the GUI.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._alerts: list[SafetyAlert] = []
        self._next_id = 1

    def check_output(self, output: str, ticket_id: int, agent: str) -> list[SafetyAlert]:
        """Scan agent output for destructive patterns. Returns new alerts."""
        new_alerts: list[SafetyAlert] = []
        lines = output.splitlines()

        for i, line in enumerate(lines):
            for pattern in DESTRUCTIVE_PATTERNS:
                m = pattern.search(line)
                if m:
                    # Get surrounding context (2 lines before/after)
                    ctx_start = max(0, i - 2)
                    ctx_end = min(len(lines), i + 3)
                    context = "\n".join(lines[ctx_start:ctx_end])

                    with self._lock:
                        alert = SafetyAlert(
                            id=self._next_id,
                            ticket_id=ticket_id,
                            agent=agent,
                            matched_pattern=pattern.pattern,
                            matched_text=m.group(0),
                            full_context=context,
                        )
                        self._alerts.append(alert)
                        self._next_id += 1
                    new_alerts.append(alert)
                    break  # Only one alert per line

        return new_alerts

    def get_pending(self) -> list[SafetyAlert]:
        """Get all pending approval requests."""
        with self._lock:
            return [a for a in self._alerts if a.state == ApprovalState.PENDING]

    def approve(self, alert_id: int) -> bool:
        """Approve a safety alert. Returns True if found and approved."""
        with self._lock:
            for alert in self._alerts:
                if alert.id == alert_id:
                    alert.state = ApprovalState.APPROVED
                    return True
        return False

    def deny(self, alert_id: int) -> bool:
        """Deny a safety alert. Returns True if found and denied."""
        with self._lock:
            for alert in self._alerts:
                if alert.id == alert_id:
                    alert.state = ApprovalState.DENIED
                    return True
        return False

    def has_pending(self) -> bool:
        """Check if there are any pending approvals."""
        with self._lock:
            return any(a.state == ApprovalState.PENDING for a in self._alerts)

    def get_pending_for_ticket(self, ticket_id: int) -> list[SafetyAlert]:
        """Get pending alerts for a specific ticket."""
        with self._lock:
            return [
                a for a in self._alerts
                if a.state == ApprovalState.PENDING and a.ticket_id == ticket_id
            ]

    def clear_resolved(self) -> None:
        """Remove resolved (approved/denied) alerts from the list."""
        with self._lock:
            self._alerts = [a for a in self._alerts if a.state == ApprovalState.PENDING]
