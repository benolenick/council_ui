"""Exponential backoff + cooldown tracking for rate-limited agents."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone


# Backoff steps in seconds: 30s, 60s, 2min, 5min, 10min
BACKOFF_STEPS = [30, 60, 120, 300, 600]


@dataclass
class _AgentCooldown:
    """Tracks cooldown state for a single agent."""
    consecutive_failures: int = 0
    cooldown_until: float = 0.0  # monotonic time
    last_success: float = 0.0
    # Reliability tracking
    total_calls: int = 0
    total_successes: int = 0
    total_failures: int = 0
    recent_results: list = field(default_factory=list)  # last 20 bools (True=success)


class CooldownTracker:
    """Tracks rate-limit cooldowns with exponential backoff for all agents."""

    def __init__(self):
        self._agents: dict[str, _AgentCooldown] = {}

    def _ensure(self, name: str) -> _AgentCooldown:
        if name not in self._agents:
            self._agents[name] = _AgentCooldown()
        return self._agents[name]

    def record_rate_limit(self, name: str) -> float:
        """Record a rate-limit hit. Returns the cooldown duration in seconds."""
        cd = self._ensure(name)
        step = min(cd.consecutive_failures, len(BACKOFF_STEPS) - 1)
        duration = BACKOFF_STEPS[step]
        cd.consecutive_failures += 1
        cd.cooldown_until = time.monotonic() + duration
        cd.total_calls += 1
        cd.total_failures += 1
        cd.recent_results.append(False)
        if len(cd.recent_results) > 20:
            cd.recent_results = cd.recent_results[-20:]
        return duration

    def record_timeout(self, name: str) -> float:
        """Record a timeout (treated like rate-limit but milder). Returns cooldown seconds."""
        cd = self._ensure(name)
        # Timeouts use half the backoff
        step = min(cd.consecutive_failures, len(BACKOFF_STEPS) - 1)
        duration = BACKOFF_STEPS[step] // 2
        cd.consecutive_failures += 1
        cd.cooldown_until = time.monotonic() + max(duration, 15)
        cd.total_calls += 1
        cd.total_failures += 1
        cd.recent_results.append(False)
        if len(cd.recent_results) > 20:
            cd.recent_results = cd.recent_results[-20:]
        return duration

    def record_success(self, name: str) -> None:
        """Record a successful call — resets cooldown."""
        cd = self._ensure(name)
        cd.consecutive_failures = 0
        cd.cooldown_until = 0.0
        cd.last_success = time.monotonic()
        cd.total_calls += 1
        cd.total_successes += 1
        cd.recent_results.append(True)
        if len(cd.recent_results) > 20:
            cd.recent_results = cd.recent_results[-20:]

    def is_available(self, name: str) -> bool:
        """Check if an agent is available (not in cooldown)."""
        cd = self._agents.get(name)
        if cd is None:
            return True
        return time.monotonic() >= cd.cooldown_until

    def get_cooldown_remaining(self, name: str) -> float:
        """Get seconds remaining in cooldown, or 0 if available."""
        cd = self._agents.get(name)
        if cd is None:
            return 0.0
        remaining = cd.cooldown_until - time.monotonic()
        return max(0.0, remaining)

    def cooldown_until_iso(self, name: str) -> str:
        """Get the cooldown-until time as an ISO string (for DB storage)."""
        cd = self._agents.get(name)
        if cd is None or cd.cooldown_until <= time.monotonic():
            return ""
        # Convert monotonic offset to wall clock
        offset = cd.cooldown_until - time.monotonic()
        dt = datetime.now(timezone.utc).replace(microsecond=0)
        from datetime import timedelta
        target = dt + timedelta(seconds=offset)
        return target.isoformat()

    def get_next_available(self, exclude: Optional[set[str]] = None) -> Optional[str]:
        """Get the agent name that will be available soonest, excluding some.

        Returns None if no agents are tracked.
        """
        exclude = exclude or set()
        best_name = None
        best_time = float("inf")
        for name, cd in self._agents.items():
            if name in exclude:
                continue
            avail = cd.cooldown_until
            if avail < best_time:
                best_time = avail
                best_name = name
        return best_name

    def get_available_agents(self, exclude: Optional[set[str]] = None) -> list[str]:
        """Get all agents currently available (not in cooldown)."""
        exclude = exclude or set()
        now = time.monotonic()
        return [
            name for name, cd in self._agents.items()
            if name not in exclude and now >= cd.cooldown_until
        ]

    def get_reliability_scores(self) -> dict[str, float]:
        """Return 0.0–1.0 reliability score per agent from recent results window.

        Returns 0.5 for agents with no data.
        """
        scores: dict[str, float] = {}
        for name, cd in self._agents.items():
            scores[name] = self._compute_reliability(cd)
        return scores

    def get_reliability_score(self, name: str) -> float:
        """Return reliability score for a single agent. 0.5 if no data."""
        cd = self._agents.get(name)
        if cd is None:
            return 0.5
        return self._compute_reliability(cd)

    @staticmethod
    def _compute_reliability(cd: _AgentCooldown) -> float:
        if not cd.recent_results:
            return 0.5
        return sum(cd.recent_results) / len(cd.recent_results)


# Allow Optional import
from typing import Optional
