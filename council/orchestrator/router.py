"""OrchestratorRouter — Qwen-driven ticket routing, dedup, assignment, execution, synthesis.

This is the brain of Council v0.3. It uses Qwen (local LLM) for orchestration decisions
and dispatches work to Claude, Codex, and Gemini agents via a thread pool.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from .agents import AgentStatus, AgentWrapper, CallResult, create_agent
from .events import EventStore, EventType, extract_events_from_output
from .handoff import HandoffBuilder
from .rate_limit import CooldownTracker
from .safety import SafetyGate
from .story import StoryManager
from .tickets import Goal, Ticket, TicketPriority, TicketState, TicketStore


# ── Prompt Templates ──

PROPOSAL_PROMPT = """/no_think
You are a planning agent. Your ONLY job is to break a goal into tickets.
DO NOT write any code. DO NOT create any files. DO NOT execute anything.
ONLY output a JSON object with ticket proposals.

GOAL: {goal_text}

STORY CONTEXT:
{story_context}

Respond with a JSON object containing a "tickets" array. Each ticket has:
- "title": short actionable title (imperative, e.g. "Add user login")
- "description": what needs to be done (1-2 sentences)
- "acceptance": how to verify it's done (1 sentence)
- "priority": "BIG" (multi-file, architecture) or "SMALL" (single file, quick fix)
- "depends_on": list of ticket titles this depends on (empty list if none)

IMPORTANT: Output ONLY the JSON object below. No explanations, no code, no markdown.
{{"tickets": [{{"title": "...", "description": "...", "acceptance": "...", "priority": "SMALL", "depends_on": []}}]}}
"""

DEDUPE_PROMPT = """/no_think
You are merging ticket proposals from multiple agents into a single canonical set.
Remove duplicates, merge similar tickets, and ensure clear dependencies.

GOAL: {goal_text}

RAW PROPOSALS:
{raw_proposals}

Output a JSON object with a "tickets" array. Each ticket:
- "title": canonical title
- "description": merged description
- "acceptance": clear acceptance criteria
- "priority": "BIG" or "SMALL"
- "depends_on": list of other ticket titles this depends on (empty list if none)

Output ONLY valid JSON.
"""

ASSIGNMENT_PROMPT = """/no_think
You are assigning tickets to coding agents. Available agents and their strengths:

- codex: Best for BUILD tasks — writing new code, implementing features, file creation
- claude: Best for ARCHITECTURE — design, code review, complex refactoring, documentation
- gemini: Best for RESEARCH/TEST — testing, analysis, debugging, investigation

AGENT RELIABILITY (recent success rate):
{agent_reliability}
Prefer agents with higher reliability for critical tickets.

RULES:
- Each agent can handle up to {max_big_per_agent} BIG tickets OR up to {max_small_per_agent} SMALL tickets at a time
- Assign based on agent strengths AND reliability
- Distribute tickets across agents — do NOT stack all tickets on one agent
- Skip agents in the unavailable list

Available agents: {available_agents}
Unavailable agents: {unavailable_agents}

TICKETS TO ASSIGN:
{tickets_json}

Output a JSON object with an "assignments" array:
{{"assignments": [{{"ticket_id": 1, "agent": "codex"}}, ...]}}

Output ONLY valid JSON.
"""

EXECUTION_PROMPT = """You are {agent_name}, a coding agent. Execute this ticket.

TICKET #{ticket_id}: {ticket_title}
DESCRIPTION: {ticket_description}
ACCEPTANCE: {ticket_acceptance}

{handoff_context}

STORY CONTEXT:
{story_context}

LIVE FEEDBACK:
{feedback_context}

Work in the project directory. When done, output:
EVENT: DONE id={ticket_id} summary="brief summary of what you did"

If you get stuck, output:
EVENT: BLOCKED id={ticket_id} reason="what's blocking you"

If you need another agent to review, output:
EVENT: NEEDS_REVIEW id={ticket_id}
"""

SYNTHESIS_PROMPT = """/no_think
You are the Council storyteller. Rewrite the project story based on recent progress.

CURRENT STORY:
{current_story}

RECENT EVENTS:
{recent_events}

COMPLETED TICKETS:
{completed_tickets}

IN-PROGRESS TICKETS:
{in_progress_tickets}

FAILED/PENDING TICKETS:
{failed_tickets}

Rewrite the story as a concise Markdown document that captures:
1. What the current goal is
2. What has been accomplished
3. What is currently in progress
4. What remains to be done
5. Any blockers, failures, or issues

Keep it under 2000 characters. Use clear Markdown headings and bullet points.
Output the story directly (not in a JSON wrapper).
"""

REVIEW_PROMPT = """/no_think
You are a code reviewer. Check this ticket's output for correctness.

TICKET #{ticket_id}: {ticket_title}
DESCRIPTION: {ticket_description}
ACCEPTANCE: {ticket_acceptance}

AGENT OUTPUT:
{agent_output}

Respond with a JSON object:
{{"verdict": "approve" or "request_changes", "notes": "your review notes"}}

Output ONLY valid JSON.
"""

CONSULT_PROMPT = """/no_think
You are a peer reviewer helping another coding agent decide next steps.

TICKET #{ticket_id}: {ticket_title}
DESCRIPTION: {ticket_description}
ACCEPTANCE: {ticket_acceptance}
REQUESTING AGENT: {requester}
RISK LEVEL: {risk_level}
REVIEW REASON: {review_reason}
QUESTION: {review_question}

REQUESTING AGENT OUTPUT:
{agent_output}

Respond ONLY as valid JSON:
{{
  "verdict": "approve" | "caution" | "reject",
  "risk": "low" | "med" | "high",
  "notes": "short explanation",
  "suggested_actions": ["short action 1", "short action 2"]
}}
"""


def _extract_json(text: str) -> Optional[dict[str, Any]]:
    """Extract JSON object from text (copied from protocol.py)."""
    text = text.strip()

    # Direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    # Strip markdown fences
    stripped = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    stripped = re.sub(r"\n?```\s*$", "", stripped, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    # Find first balanced { ... }
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            if in_string:
                escape = True
            continue
        if c == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start: i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                except (json.JSONDecodeError, ValueError):
                    pass
                break

    return None


class FailureClass(str, Enum):
    NONE = "NONE"
    HARD_FAIL = "HARD_FAIL"
    SOFT_FAIL = "SOFT_FAIL"
    STYLE_FAIL = "STYLE_FAIL"


@dataclass
class ExecutionResult:
    success: bool
    failure_class: FailureClass
    confidence: float
    strategy_used: str
    verified_by: list[str]
    next_steps: list[str]
    reason: str = ""


class OrchestratorRouter:
    """The brain of Council v0.3 — routes work through ticket lifecycle."""

    def __init__(
        self,
        store: TicketStore,
        event_store: EventStore,
        story: StoryManager,
        safety: SafetyGate,
        cooldowns: CooldownTracker,
        handoffs: HandoffBuilder,
        agents: dict[str, AgentWrapper],
        qwen: AgentWrapper,
        max_big_per_agent: int = 1,
        max_small_per_agent: int = 2,
        stalled_ticket_timeout_s: int = 300,
        on_log: Optional[Callable[[str], None]] = None,
    ):
        self.store = store
        self.events = event_store
        self.story = story
        self.safety = safety
        self.cooldowns = cooldowns
        self.handoffs = handoffs
        self.agents = agents  # {name: AgentWrapper} — codex, claude, gemini
        self.qwen = qwen
        self._on_log = on_log

        # FV persistent memory (optional)
        self._fv_memory = None
        try:
            from council.fv.agent_pipeline.agent import MemoryHandle
            self._fv_memory = MemoryHandle()
            if on_log:
                on_log("[FV] Persistent memory initialized")
        except Exception:
            pass  # FV not available or deps missing — that's fine
        self.max_big_per_agent = max(1, int(max_big_per_agent))
        self.max_small_per_agent = max(1, int(max_small_per_agent))
        self.stalled_ticket_timeout_s = max(60, int(stalled_ticket_timeout_s))

        # Execution state
        self._pending_results: list[tuple[str, CallResult, list[int]]] = []
        self._lock = threading.Lock()
        self._active_tasks: dict[str, threading.Thread] = {}  # agent -> thread
        self._agent_outputs: dict[str, str] = {}  # agent -> latest output
        self._agent_phases: dict[str, str] = {}  # agent -> current phase (propose/dedupe/execute/idle)
        self._agent_start_times: dict[str, float] = {}  # agent -> monotonic start time
        self._paused = False
        self._feedback_lock = threading.Lock()
        self._feedback_by_agent: dict[str, list[str]] = {}
        self._feedback_broadcast: list[str] = []
        self._consult_rounds: dict[int, int] = {}

        # Auto-retry state
        self._auto_retry_enabled: bool = False
        self._auto_retry_goal_id: Optional[int] = None
        self._last_retry_check: float = 0.0  # monotonic
        self._retry_interval: float = 10.0  # seconds
        self._retry_in_progress: bool = False
        self._was_busy: bool = False

        # Wire up streaming output callbacks
        for name, agent in agents.items():
            agent.on_output = self._on_agent_output
        qwen.on_output = self._on_agent_output

        # Initialize agent status in DB
        for name in agents:
            store.upsert_agent(name, "IDLE")
        store.upsert_agent("qwen", "IDLE")

    def _on_agent_output(self, agent_name: str, partial_text: str) -> None:
        """Callback from agent wrappers — streams partial output to GUI."""
        with self._lock:
            self._agent_outputs[agent_name] = partial_text

    def log(self, msg: str) -> None:
        if self._on_log:
            self._on_log(msg)

    def pause(self) -> None:
        self._paused = True
        self.log("Orchestrator paused")

    def resume(self) -> None:
        self._paused = False
        self.log("Orchestrator resumed")

    @property
    def is_paused(self) -> bool:
        return self._paused

    def add_feedback(self, message: str, target_agent: Optional[str] = None, author: str = "user") -> None:
        """Queue feedback for agents. target_agent=None broadcasts to all coding agents."""
        msg = message.strip()
        if not msg:
            return
        entry = f"[{author}] {msg}"
        with self._feedback_lock:
            if target_agent:
                queue = self._feedback_by_agent.setdefault(target_agent, [])
                queue.append(entry)
                if len(queue) > 50:
                    del queue[:-50]
            else:
                self._feedback_broadcast.append(entry)
                if len(self._feedback_broadcast) > 200:
                    del self._feedback_broadcast[:-200]
        if target_agent:
            self.log(f"Feedback queued for {target_agent}: {msg[:120]}")
        else:
            self.log(f"Broadcast feedback queued: {msg[:120]}")

    def _consume_feedback_for_agent(self, agent_name: str, limit: int = 8) -> str:
        """Get and clear direct feedback for an agent, plus recent broadcast context."""
        with self._feedback_lock:
            direct = list(self._feedback_by_agent.get(agent_name, []))
            broadcast = list(self._feedback_broadcast[-limit:])
            self._feedback_by_agent[agent_name] = []
        combined = (broadcast + direct)[-limit:]
        if not combined:
            return "None"
        return "\n".join(f"- {line}" for line in combined)

    @staticmethod
    def _normalize_risk_level(value: Any) -> str:
        risk = str(value or "med").strip().lower()
        if risk in {"low", "med", "high"}:
            return risk
        if risk in {"medium", "moderate"}:
            return "med"
        return "med"

    @staticmethod
    def _detect_uncertainty_signal(text: str) -> Optional[str]:
        lower = text.lower()
        signals = [
            "not sure",
            "unsure",
            "uncertain",
            "risky",
            "might break",
            "could break",
            "needs review",
            "please review",
            "security risk",
            "data loss",
        ]
        for sig in signals:
            if sig in lower:
                return sig
        return None


    @staticmethod
    def _strategy_for(agent_name: str) -> str:
        return f"{agent_name}:default"

    @staticmethod
    def _ticket_backoff_seconds(attempts_total: int) -> int:
        steps = [30, 120, 300, 900]
        idx = min(max(attempts_total - 1, 0), len(steps) - 1)
        return steps[idx]

    @staticmethod
    def _cooldown_until_iso(seconds: int) -> str:
        dt = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(seconds=seconds)
        return dt.isoformat()

    def _select_agent_for_ticket(
        self,
        ticket: Ticket,
        preferred_agent: str,
        available: list[str],
        force_same_strategy: bool = False,
    ) -> Optional[str]:
        ordered = []
        if preferred_agent in available:
            ordered.append(preferred_agent)
        ordered.extend([a for a in available if a != preferred_agent])
        for agent in ordered:
            strategy = self._strategy_for(agent)
            if (
                not force_same_strategy
                and ticket.last_failure_reason
                and ticket.last_strategy_used == strategy
            ):
                continue
            return agent
        return None

    def _validate_and_rebalance_assignments(
        self,
        raw: dict[str, list[int]],
        available: list[str],
        ticket_map: dict[int, Ticket],
    ) -> dict[str, list[int]]:
        """Enforce per-agent capacity limits and redistribute overflow.

        Rules: respect configured BIG/SMALL WIP caps per agent.
        Excess tickets go to overflow → redistributed round-robin.
        """
        cleaned: dict[str, list[int]] = {}
        overflow: list[int] = []
        existing_load: dict[str, tuple[int, int]] = {}
        for agent in available:
            existing = self.store.list_tickets(owner=agent)
            existing_big = sum(
                1 for t in existing
                if t.state in {TicketState.READY, TicketState.IN_PROGRESS, TicketState.REVIEW}
                and t.priority == TicketPriority.BIG
            )
            existing_small = sum(
                1 for t in existing
                if t.state in {TicketState.READY, TicketState.IN_PROGRESS, TicketState.REVIEW}
                and t.priority == TicketPriority.SMALL
            )
            existing_load[agent] = (existing_big, existing_small)

        for agent, tids in raw.items():
            base_big, base_small = existing_load.get(agent, (0, 0))
            big_count = base_big
            small_count = base_small
            kept: list[int] = []
            for tid in tids:
                ticket = ticket_map.get(tid)
                if not ticket:
                    continue
                is_big = ticket.priority == TicketPriority.BIG
                if is_big:
                    if big_count >= self.max_big_per_agent or small_count > 0:
                        overflow.append(tid)
                    else:
                        big_count += 1
                        kept.append(tid)
                else:
                    if big_count > 0 or small_count >= self.max_small_per_agent:
                        overflow.append(tid)
                    else:
                        small_count += 1
                        kept.append(tid)
            if kept:
                cleaned[agent] = kept

        if not overflow:
            return cleaned

        # Build capacity map for redistribution
        def _agent_capacity(agent: str) -> tuple[int, int]:
            """Returns (big_slots, small_slots) remaining."""
            tids = cleaned.get(agent, [])
            existing_big, existing_small = existing_load.get(agent, (0, 0))
            bigs = existing_big + sum(
                1 for t in tids
                if ticket_map.get(t) and ticket_map[t].priority == TicketPriority.BIG
            )
            smalls = existing_small + sum(
                1 for t in tids
                if ticket_map.get(t) and ticket_map[t].priority == TicketPriority.SMALL
            )
            if bigs > 0:
                return (0, 0)
            return (
                max(0, self.max_big_per_agent - bigs),
                max(0, self.max_small_per_agent - smalls),
            )

        # Sort available agents by reliability for better distribution
        reliability = self.cooldowns.get_reliability_scores()
        sorted_agents = sorted(available, key=lambda a: reliability.get(a, 0.5), reverse=True)

        unassigned_overflow: list[int] = []
        for tid in overflow:
            ticket = ticket_map.get(tid)
            if not ticket:
                continue
            is_big = ticket.priority == TicketPriority.BIG
            placed = False
            for agent in sorted_agents:
                big_cap, small_cap = _agent_capacity(agent)
                if is_big and big_cap > 0:
                    chosen = self._select_agent_for_ticket(ticket, agent, available)
                    if chosen:
                        cleaned.setdefault(chosen, []).append(tid)
                        placed = True
                        break
                elif not is_big and small_cap > 0:
                    chosen = self._select_agent_for_ticket(ticket, agent, available)
                    if chosen:
                        cleaned.setdefault(chosen, []).append(tid)
                        placed = True
                        break
            if not placed:
                unassigned_overflow.append(tid)

        if unassigned_overflow:
            self.log(
                "Deferred tickets due to WIP limits: "
                + ", ".join(f"#{tid}" for tid in unassigned_overflow)
            )

        return cleaned

    def _emit_execution_result(
        self,
        ticket_id: int,
        agent: str,
        result: ExecutionResult,
    ) -> None:
        self.events.emit(
            EventType.EXECUTION_RESULT,
            ticket_id=ticket_id,
            agent=agent,
            payload={
                "success": result.success,
                "failure_class": result.failure_class.value,
                "confidence": max(0.0, min(1.0, float(result.confidence))),
                "strategy_used": result.strategy_used,
                "verified_by": result.verified_by,
                "next_steps": result.next_steps,
                "reason": result.reason,
            },
        )

    def _agent_is_available_for_consult(self, agent_name: str) -> bool:
        if not self.cooldowns.is_available(agent_name):
            return False
        info = self.store.get_agent(agent_name)
        if not info:
            return False
        return info.get("status") == "IDLE"

    def _parse_consult_response(self, raw_output: str) -> dict[str, Any]:
        data = _extract_json(raw_output) or {}
        verdict = str(data.get("verdict", "caution")).strip().lower()
        if verdict not in {"approve", "caution", "reject"}:
            verdict = "caution"
        risk = self._normalize_risk_level(data.get("risk", "med"))
        notes = str(data.get("notes", "")).strip()
        actions = data.get("suggested_actions", [])
        if not isinstance(actions, list):
            actions = []
        actions = [str(a).strip() for a in actions if str(a).strip()][:3]
        return {
            "verdict": verdict,
            "risk": risk,
            "notes": notes,
            "suggested_actions": actions,
        }

    def _start_peer_consult(
        self,
        ticket_id: int,
        requester: str,
        requester_output: str,
        reason: str,
        risk_level: str,
        question: str,
    ) -> None:
        rounds = self._consult_rounds.get(ticket_id, 0)
        if rounds >= 1:
            self.log(f"Ticket #{ticket_id}: consult already completed once; skipping duplicate consult")
            return
        self._consult_rounds[ticket_id] = rounds + 1

        def _do_consult() -> None:
            ticket = self.store.get_ticket(ticket_id)
            if not ticket:
                return

            peers = [
                name for name in self.agents
                if name != requester and self._agent_is_available_for_consult(name)
            ][:2]
            if not peers:
                note = (
                    f"Consult requested by {requester}, but no idle peers were available.\n"
                    f"Reason: {reason or 'unspecified'}\n"
                    f"Risk: {risk_level}"
                )
                self.store.set_ticket_review(ticket_id, note)
                self.log(f"Ticket #{ticket_id}: no peers available for consult")
                return

            self.log(f"Ticket #{ticket_id}: launching peer consult with {', '.join(peers)}")
            responses: list[tuple[str, dict[str, Any]]] = []

            for peer in peers:
                peer_agent = self.agents.get(peer)
                if not peer_agent:
                    continue

                prompt = CONSULT_PROMPT.format(
                    ticket_id=ticket.id,
                    ticket_title=ticket.title,
                    ticket_description=ticket.description,
                    ticket_acceptance=ticket.acceptance,
                    requester=requester,
                    risk_level=risk_level,
                    review_reason=reason or "unspecified",
                    review_question=question or "Please assess risks and recommend next steps.",
                    agent_output=requester_output[:4000] or "(no output provided)",
                )

                self.store.set_agent_status(peer, "BUSY", current_ticket=ticket.id)
                self._agent_phases[peer] = "consult"
                self._agent_start_times[peer] = time.monotonic()
                result = peer_agent.call(prompt, phase="consult")
                self.store.set_agent_status(peer, "IDLE")
                self._agent_phases[peer] = "idle"
                self._agent_start_times.pop(peer, None)

                if result.ok:
                    parsed = self._parse_consult_response(result.output)
                else:
                    parsed = {
                        "verdict": "caution",
                        "risk": "high",
                        "notes": f"Consult failed: {result.error or 'unknown error'}",
                        "suggested_actions": [],
                    }
                responses.append((peer, parsed))

            if not responses:
                self.store.set_ticket_review(
                    ticket_id,
                    f"Consult requested by {requester}, but no consult responses were received.",
                )
                return

            verdict_counts = {"approve": 0, "caution": 0, "reject": 0}
            max_risk = "low"
            risk_rank = {"low": 1, "med": 2, "high": 3}
            for _, r in responses:
                verdict_counts[r["verdict"]] += 1
                if risk_rank[r["risk"]] > risk_rank[max_risk]:
                    max_risk = r["risk"]

            final_verdict = "approve"
            if verdict_counts["reject"] > 0:
                final_verdict = "reject"
            elif verdict_counts["caution"] > 0:
                final_verdict = "caution"
            if risk_level == "high" and len(responses) < 2:
                final_verdict = "caution"

            lines = [
                f"Peer consult for ticket #{ticket_id}",
                f"Requester: {requester}",
                f"Reason: {reason or 'unspecified'}",
                f"Question: {question or 'n/a'}",
                f"Requested risk: {risk_level}",
                f"Observed risk: {max_risk}",
                f"Final verdict: {final_verdict}",
                "",
                "Peer responses:",
            ]
            for peer, parsed in responses:
                line = f"- {peer}: verdict={parsed['verdict']} risk={parsed['risk']}"
                if parsed["notes"]:
                    line += f" notes={parsed['notes']}"
                lines.append(line)
                for action in parsed["suggested_actions"]:
                    lines.append(f"  action: {action}")
            note = "\n".join(lines)

            self.store.set_ticket_review(ticket_id, note)
            self.events.emit(
                EventType.NEEDS_REVIEW,
                ticket_id=ticket_id,
                agent=requester,
                payload={"verdict": final_verdict, "risk": max_risk},
            )

            if final_verdict == "approve" and risk_level != "high":
                self.store.unassign_ticket(ticket_id)
                self.events.emit(
                    EventType.TICKET_STATE_CHANGED,
                    ticket_id=ticket_id,
                    payload={"state": "NEW", "reason": "consult_approved_retry"},
                )
                self.log(
                    f"Ticket #{ticket_id}: consult approved, returned to NEW for reassignment/retry"
                )
            else:
                self.log(
                    f"Ticket #{ticket_id}: consult outcome={final_verdict}, kept in REVIEW"
                )

        t = threading.Thread(target=_do_consult, daemon=True)
        t.start()
        self._active_tasks[f"consult_{ticket_id}"] = t

    # ── Phase: Request proposals from agents ──

    def request_proposals(self, goal_id: int, goal_text: str) -> None:
        """Ask all available agents for ticket proposals in parallel."""
        story_ctx = self.story.get_context_for_prompt(max_chars=4000)
        prompt = PROPOSAL_PROMPT.format(goal_text=goal_text, story_context=story_ctx)

        self.log(f"Requesting proposals for goal #{goal_id}")
        self.events.emit(EventType.GOAL_CREATED, payload={"goal_id": goal_id, "text": goal_text})

        # Ask each available agent
        for name, agent in self.agents.items():
            if not self.cooldowns.is_available(name):
                self.log(f"  Skipping {name} (cooldown)")
                continue

            def _do_propose(n=name, a=agent, p=prompt, gid=goal_id):
                self.store.set_agent_status(n, "BUSY")
                self._agent_phases[n] = "propose"
                self._agent_start_times[n] = time.monotonic()
                result = a.call(p, phase="propose")
                self._handle_proposal_result(n, result, gid)
                self.store.set_agent_status(n, "IDLE")
                self._agent_phases[n] = "idle"
                self._agent_start_times.pop(n, None)

            t = threading.Thread(target=_do_propose, daemon=True)
            t.start()
            self._active_tasks[name] = t

        # Also ask Qwen
        def _qwen_propose():
            self.store.set_agent_status("qwen", "BUSY")
            self._agent_phases["qwen"] = "propose"
            self._agent_start_times["qwen"] = time.monotonic()
            result = self.qwen.call(prompt, phase="propose")
            self._handle_proposal_result("qwen", result, goal_id)
            self.store.set_agent_status("qwen", "IDLE")
            self._agent_phases["qwen"] = "idle"
            self._agent_start_times.pop("qwen", None)

        t = threading.Thread(target=_qwen_propose, daemon=True)
        t.start()
        self._active_tasks["qwen"] = t

    def _handle_proposal_result(self, agent: str, result: CallResult, goal_id: int) -> None:
        """Handle a proposal result from an agent."""
        # Always store output for GUI display
        with self._lock:
            self._agent_outputs[agent] = result.output or result.error or ""

        if result.rate_limited:
            duration = self.cooldowns.record_rate_limit(agent)
            self.log(f"{agent} rate-limited, cooldown {duration}s")
            self.events.emit(
                EventType.AGENT_RATE_LIMITED,
                agent=agent,
                payload={
                    "duration": duration,
                    "match": result.rate_limit_pattern,
                    "source": result.rate_limit_source,
                    "stderr_excerpt": (result.raw_stderr or "")[:300],
                },
            )
            self.store.set_agent_cooldown(agent, self.cooldowns.cooldown_until_iso(agent))
            return

        if not result.ok:
            self.log(f"{agent} proposal error: {result.error}")
            return

        self.cooldowns.record_success(agent)
        self.events.emit(EventType.PROPOSAL_RECEIVED, agent=agent,
                        payload={"goal_id": goal_id, "length": len(result.output)})
        self.log(f"Proposal received from {agent} ({len(result.output)} chars)")

        with self._lock:
            self._pending_results.append((agent, result, []))

    # ── Phase: Dedupe and merge proposals ──

    def dedupe_and_merge(self, goal_id: int, goal_text: str) -> list[Ticket]:
        """Use Qwen to merge raw proposals into canonical tickets."""
        with self._lock:
            proposals = list(self._pending_results)
            self._pending_results.clear()

        if not proposals:
            self.log("No proposals to merge")
            return []

        # Build raw proposals text
        raw_parts = []
        for agent, result, _ in proposals:
            raw_parts.append(f"=== {agent} ===\n{result.output}\n")
        raw_text = "\n".join(raw_parts)

        prompt = DEDUPE_PROMPT.format(goal_text=goal_text, raw_proposals=raw_text)

        self.log("Deduping proposals via Qwen...")
        self.store.set_agent_status("qwen", "BUSY")
        self._agent_phases["qwen"] = "dedupe"
        self._agent_start_times["qwen"] = time.monotonic()
        result = self.qwen.call(prompt, phase="dedupe")
        self.store.set_agent_status("qwen", "IDLE")
        self._agent_phases["qwen"] = "idle"
        self._agent_start_times.pop("qwen", None)

        if not result.ok:
            self.log(f"Dedupe failed: {result.error}")
        else:
            parsed = self._parse_tickets_from_output(result.output, goal_id)
            if parsed:
                return parsed
            self.log("Dedupe output unparsable — trying raw proposals")

        # Fallback: try parsing each raw proposal directly
        for agent_name, proposal_result, _ in proposals:
            parsed = self._parse_tickets_from_output(proposal_result.output, goal_id)
            if parsed:
                self.log(f"Recovered tickets from raw {agent_name} proposal")
                return parsed

        # Last resort: create one bootstrap ticket so the goal does not dead-end.
        self.log("No parseable tickets from any proposal — creating bootstrap ticket")
        ticket = self.store.create_ticket(
            goal_id=goal_id,
            title="Implement goal baseline",
            description=goal_text.strip(),
            acceptance="Deliver a working baseline implementation for this goal.",
            priority=TicketPriority.BIG,
        )
        self.events.emit(
            EventType.TICKET_CREATED,
            ticket_id=ticket.id,
            payload={"title": ticket.title, "priority": ticket.priority.value, "fallback": "bootstrap"},
        )
        return [ticket]

    def _parse_tickets_from_output(self, output: str, goal_id: int) -> list[Ticket]:
        """Parse ticket JSON from Qwen output and create them in the store."""
        data = _extract_json(output)
        if not data or "tickets" not in data:
            self.log("Could not parse tickets from output")
            return []

        tickets: list[Ticket] = []
        title_to_id: dict[str, int] = {}

        # First pass: create tickets
        for t in data["tickets"]:
            title = t.get("title", "Untitled")
            priority = TicketPriority.BIG if t.get("priority", "").upper() == "BIG" else TicketPriority.SMALL
            ticket = self.store.create_ticket(
                goal_id=goal_id,
                title=title,
                description=t.get("description", ""),
                acceptance=t.get("acceptance", ""),
                priority=priority,
            )
            title_to_id[title] = ticket.id
            tickets.append(ticket)
            self.events.emit(EventType.TICKET_CREATED, ticket_id=ticket.id,
                           payload={"title": title, "priority": priority.value})
            self.log(f"  Ticket #{ticket.id}: {title} [{priority.value}]")

        # Second pass: resolve dependencies
        for t, ticket in zip(data["tickets"], tickets):
            deps = t.get("depends_on", [])
            if deps:
                dep_ids = [title_to_id[d] for d in deps if d in title_to_id]
                if dep_ids:
                    # Update depends_on in DB
                    with self.store._lock:
                        self.store._conn.execute(
                            "UPDATE tickets SET depends_on = ? WHERE id = ?",
                            (json.dumps(dep_ids), ticket.id),
                        )
                        self.store._conn.commit()
                    ticket.depends_on = dep_ids

        return tickets

    # ── Phase: Assign tickets to agents ──

    def assign_tickets(self, goal_id: int) -> dict[str, list[int]]:
        """Use Qwen to assign tickets to agents. Returns {agent: [ticket_ids]}."""
        unassigned = self.store.list_tickets(goal_id=goal_id, state=TicketState.NEW)
        if not unassigned:
            self.log("No tickets to assign")
            return {}

        ready_for_assignment: list[Ticket] = []
        cooling_down = 0
        for ticket in unassigned:
            if ticket.attempts_total >= 4:
                self.store.mark_needs_rethink(ticket.id, "Exceeded 4 attempts")
                self.events.emit(
                    EventType.TICKET_BLOCKED,
                    ticket_id=ticket.id,
                    payload={"reason": "NEEDS_RETHINK: exceeded 4 attempts"},
                )
                continue
            if not self.store.is_ticket_available(ticket):
                cooling_down += 1
                continue
            ready_for_assignment.append(ticket)

        if not ready_for_assignment:
            self.log(f"No tickets ready for assignment (cooldown={cooling_down})")
            return {}

        # Determine available agents
        available = []
        unavailable = []
        for name in self.agents:
            if self.cooldowns.is_available(name):
                # Check concurrency limits
                big_count, small_count = self._agent_wip_counts(name)
                if big_count < self.max_big_per_agent and small_count < self.max_small_per_agent:
                    available.append(name)
                else:
                    unavailable.append(f"{name} (busy: {big_count}B+{small_count}S)")
            else:
                remaining = self.cooldowns.get_cooldown_remaining(name)
                unavailable.append(f"{name} (cooldown: {remaining:.0f}s)")

        if not available:
            self.log("No agents available for assignment")
            return {}

        # Compute reliability scores for the prompt
        reliability = self.cooldowns.get_reliability_scores()
        reliability_lines = []
        for name in available:
            score = reliability.get(name, 0.5)
            reliability_lines.append(f"- {name}: {score:.0%}")
        agent_reliability_text = "\n".join(reliability_lines) or "No data yet"

        tickets_json = json.dumps([t.to_dict() for t in ready_for_assignment], indent=2)
        prompt = ASSIGNMENT_PROMPT.format(
            available_agents=", ".join(available),
            unavailable_agents=", ".join(unavailable) or "none",
            tickets_json=tickets_json,
            agent_reliability=agent_reliability_text,
            max_big_per_agent=self.max_big_per_agent,
            max_small_per_agent=self.max_small_per_agent,
        )

        self.log("Assigning tickets via Qwen...")
        self.store.set_agent_status("qwen", "BUSY")
        self._agent_phases["qwen"] = "assign"
        self._agent_start_times["qwen"] = time.monotonic()
        result = self.qwen.call(prompt, phase="assign")
        self.store.set_agent_status("qwen", "IDLE")
        self._agent_phases["qwen"] = "idle"
        self._agent_start_times.pop("qwen", None)

        if not result.ok:
            self.log(f"Assignment failed: {result.error}")
            return self._fallback_assignment(unassigned, available)

        data = _extract_json(result.output)
        if not data or "assignments" not in data:
            self.log("Could not parse assignments, using fallback")
            return self._fallback_assignment(unassigned, available)

        raw_assignments: dict[str, list[int]] = {}
        ticket_map = {t.id: t for t in ready_for_assignment}

        for a in data["assignments"]:
            tid = a.get("ticket_id")
            requested_agent = a.get("agent", "")
            if tid in ticket_map and requested_agent in self.agents:
                ticket = ticket_map[tid]
                force_same = bool(a.get("force_strategy", False))
                chosen = self._select_agent_for_ticket(
                    ticket=ticket,
                    preferred_agent=requested_agent,
                    available=available,
                    force_same_strategy=force_same,
                )
                if not chosen:
                    self.log(f"  Skipping ticket #{tid}: no non-repeating strategy available")
                    continue
                raw_assignments.setdefault(chosen, []).append(tid)

        # Validate capacity limits and redistribute overflow
        assignments = self._validate_and_rebalance_assignments(raw_assignments, available, ticket_map)

        # Apply assignments to store
        planned_counts: dict[str, tuple[int, int]] = {}
        for agent_name, tids in assignments.items():
            current_big, current_small = self._agent_wip_counts(agent_name)
            planned_counts[agent_name] = (current_big, current_small)
            for tid in tids:
                ticket = ticket_map.get(tid)
                if not ticket:
                    continue
                big_count, small_count = planned_counts.get(agent_name, (0, 0))
                is_big = ticket.priority == TicketPriority.BIG
                can_assign = (
                    is_big
                    and big_count < self.max_big_per_agent
                    and small_count == 0
                ) or (
                    (not is_big)
                    and small_count < self.max_small_per_agent
                    and big_count == 0
                )
                if not can_assign:
                    self.log(
                        f"  Deferring ticket #{tid}: {agent_name} at WIP cap "
                        f"({big_count}B+{small_count}S)"
                    )
                    continue
                self.store.assign_ticket(tid, agent_name)
                self.events.emit(EventType.TICKET_ASSIGNED, ticket_id=tid, agent=agent_name)
                self.log(f"  Ticket #{tid} → {agent_name}")
                if is_big:
                    planned_counts[agent_name] = (big_count + 1, small_count)
                else:
                    planned_counts[agent_name] = (big_count, small_count + 1)

        return assignments

    def _fallback_assignment(self, tickets: list[Ticket], available: list[str]) -> dict[str, list[int]]:
        """Capacity-aware round-robin assignment as fallback, sorted by reliability."""
        assignments: dict[str, list[int]] = {}
        capacity: dict[str, dict[str, int]] = {}
        for agent_name in available:
            big_count, small_count = self._agent_wip_counts(agent_name)
            capacity[agent_name] = {
                "big": max(0, self.max_big_per_agent - big_count),
                "small": max(0, self.max_small_per_agent - small_count),
            }

        # Sort preference lists by reliability score
        reliability = self.cooldowns.get_reliability_scores()
        preference_base = {"BIG": ["codex", "claude", "gemini"], "SMALL": ["gemini", "claude", "codex"]}
        preference: dict[str, list[str]] = {}
        for prio, agents_list in preference_base.items():
            avail_pref = [a for a in agents_list if a in available]
            avail_pref.sort(key=lambda a: reliability.get(a, 0.5), reverse=True)
            preference[prio] = avail_pref

        for ticket in tickets:
            pref = preference.get(ticket.priority.value, available)
            is_big = ticket.priority == TicketPriority.BIG
            assigned = False
            for requested in pref:
                cap = capacity.get(requested)
                if not cap:
                    continue
                if is_big and cap["big"] > 0 and cap["small"] == self.max_small_per_agent:
                    chosen = self._select_agent_for_ticket(ticket, requested, available)
                    if not chosen:
                        continue
                    self.store.assign_ticket(ticket.id, chosen)
                    assignments.setdefault(chosen, []).append(ticket.id)
                    self.events.emit(EventType.TICKET_ASSIGNED, ticket_id=ticket.id, agent=chosen)
                    self.log(f"  Ticket #{ticket.id} → {chosen} (fallback)")
                    cap["big"] = 0
                    cap["small"] = 0  # BIG takes full capacity
                    assigned = True
                    break
                elif not is_big and cap["small"] > 0 and cap["big"] > 0:
                    chosen = self._select_agent_for_ticket(ticket, requested, available)
                    if not chosen:
                        continue
                    self.store.assign_ticket(ticket.id, chosen)
                    assignments.setdefault(chosen, []).append(ticket.id)
                    self.events.emit(EventType.TICKET_ASSIGNED, ticket_id=ticket.id, agent=chosen)
                    self.log(f"  Ticket #{ticket.id} → {chosen} (fallback)")
                    cap["small"] -= 1
                    cap["big"] = 0  # SMALL prevents BIG
                    assigned = True
                    break
            if not assigned and available:
                chosen = self._select_agent_for_ticket(ticket, available[0], available, force_same_strategy=True)
                if chosen:
                    self.store.assign_ticket(ticket.id, chosen)
                    assignments.setdefault(chosen, []).append(ticket.id)
                    self.events.emit(EventType.TICKET_ASSIGNED, ticket_id=ticket.id, agent=chosen)

        return assignments

    # ── Phase: Execute assigned tickets ──

    def execute_assigned(self, assignments: dict[str, list[int]]) -> None:
        """Submit execution prompts to agents via background threads."""
        story_ctx = self.story.get_context_for_prompt(max_chars=3000)

        for agent_name, ticket_ids in assignments.items():
            agent = self.agents.get(agent_name)
            if not agent:
                continue

            for tid in ticket_ids:
                ticket = self.store.get_ticket(tid)
                if not ticket:
                    continue

                # Check for handoff context
                handoff_ctx = ""
                handoff_content = self.handoffs.get_latest_handoff(tid)
                if handoff_content:
                    handoff_ctx = f"\nHANDOFF FROM PREVIOUS AGENT:\n{handoff_content}\n"

                # Inject FV memory context if available
                memory_ctx = ""
                if self._fv_memory:
                    try:
                        facts = self._fv_memory.retrieve_facts(f"{ticket.title} {ticket.description}", final_k=3)
                        if facts:
                            memory_ctx = "\nMEMORY (past outcomes):\n" + "\n".join(
                                f"- {txt}" for _, txt, _, _ in facts
                            ) + "\n"
                    except Exception:
                        pass

                prompt = EXECUTION_PROMPT.format(
                    agent_name=agent_name,
                    ticket_id=tid,
                    ticket_title=ticket.title,
                    ticket_description=ticket.description,
                    ticket_acceptance=ticket.acceptance,
                    handoff_context=handoff_ctx + memory_ctx,
                    story_context=story_ctx,
                    feedback_context=self._consume_feedback_for_agent(agent_name),
                )

                self.store.update_ticket_state(tid, TicketState.IN_PROGRESS)
                self.store.set_agent_status(agent_name, "BUSY", current_ticket=tid)
                self.events.emit(EventType.TICKET_STATE_CHANGED, ticket_id=tid,
                               agent=agent_name, payload={"state": "IN_PROGRESS"})

                def _do_execute(n=agent_name, a=agent, p=prompt, tids=[tid]):
                    self._agent_phases[n] = "execute"
                    self._agent_start_times[n] = time.monotonic()
                    result = a.call(p, phase="execute")
                    with self._lock:
                        self._pending_results.append((n, result, tids))
                        self._agent_outputs[n] = result.output
                    self._agent_phases[n] = "idle"
                    self._agent_start_times.pop(n, None)

                t = threading.Thread(target=_do_execute, daemon=True)
                t.start()
                self._active_tasks[f"{agent_name}_{tid}"] = t

    # ── Phase: Handle agent results ──

    def handle_agent_result(self, agent: str, result: CallResult, ticket_ids: list[int]) -> None:
        """Process an agent's execution result with failure taxonomy and retry intelligence."""
        # Guard: skip if ticket was already failovered (owner changed or state not IN_PROGRESS)
        valid_tids = []
        for tid in ticket_ids:
            ticket = self.store.get_ticket(tid)
            if not ticket:
                continue
            if ticket.state != TicketState.IN_PROGRESS:
                self.log(f"Ticket #{tid} no longer IN_PROGRESS (state={ticket.state.value}), skipping result")
                continue
            if ticket.owner != agent:
                self.log(f"Ticket #{tid} owner changed to {ticket.owner}, skipping result from {agent}")
                continue
            valid_tids.append(tid)
        if not valid_tids:
            return
        ticket_ids = valid_tids

        strategy = self._strategy_for(agent)

        def _record_failure(
            tid: int,
            failure_class: FailureClass,
            reason: str,
            confidence: float,
            next_steps: list[str],
            verified_by: Optional[list[str]] = None,
            review_note: Optional[str] = None,
            force_block: bool = False,
        ) -> None:
            verified = verified_by or []
            self.store.record_attempt(tid, strategy_used=strategy, success=False, failure_reason=reason)
            ticket_now = self.store.get_ticket(tid)
            attempts = ticket_now.attempts_total if ticket_now else 1

            if attempts >= 4:
                self.store.mark_needs_rethink(tid, f"{reason} (attempts={attempts})")
                self.events.emit(
                    EventType.TICKET_BLOCKED,
                    ticket_id=tid,
                    agent=agent,
                    payload={"reason": f"NEEDS_RETHINK after {attempts} attempts: {reason}"},
                )
                next_steps = ["Escalate with new strategy design", "Require operator rethink"]
            elif force_block:
                self.store.block_ticket(tid, reason)
                self.events.emit(
                    EventType.TICKET_BLOCKED,
                    ticket_id=tid,
                    agent=agent,
                    payload={"reason": reason},
                )
            elif failure_class == FailureClass.STYLE_FAIL:
                note = review_note or f"Style mismatch detected: {reason}. Ask one clarifying question or provide 2 variants."
                self.store.set_ticket_review(tid, note)
            elif failure_class == FailureClass.SOFT_FAIL:
                note = review_note or f"Soft failure: {reason}"
                self.store.set_ticket_review(tid, note)
                self.store.unassign_ticket(tid)
                cooldown_s = self._ticket_backoff_seconds(attempts)
                self.store.set_ticket_cooldown(tid, self._cooldown_until_iso(cooldown_s))
            else:
                self.store.unassign_ticket(tid)
                cooldown_s = self._ticket_backoff_seconds(attempts)
                self.store.set_ticket_cooldown(tid, self._cooldown_until_iso(cooldown_s))

            self._emit_execution_result(
                tid,
                agent,
                ExecutionResult(
                    success=False,
                    failure_class=failure_class,
                    confidence=confidence,
                    strategy_used=strategy,
                    verified_by=verified,
                    next_steps=next_steps,
                    reason=reason,
                ),
            )

        if result.rate_limited:
            duration = self.cooldowns.record_rate_limit(agent)
            self.store.set_agent_cooldown(agent, self.cooldowns.cooldown_until_iso(agent))
            self.log(f"{agent} rate-limited during execution, cooldown {duration}s")
            self.events.emit(
                EventType.AGENT_RATE_LIMITED,
                agent=agent,
                payload={
                    "duration": duration,
                    "match": result.rate_limit_pattern,
                    "source": result.rate_limit_source,
                    "stderr_excerpt": (result.raw_stderr or "")[:300],
                    "phase": "execute",
                    "ticket_ids": ticket_ids,
                },
            )
            for tid in ticket_ids:
                _record_failure(
                    tid,
                    FailureClass.HARD_FAIL,
                    reason="rate_limited",
                    confidence=0.1,
                    next_steps=["Retry with different strategy", "Wait for ticket cooldown"],
                )
            # Immediate failover: reassign this agent's remaining tickets to others
            goal_id = None
            for tid in ticket_ids:
                ticket = self.store.get_ticket(tid)
                if ticket:
                    goal_id = ticket.goal_id
                    break
            if goal_id is not None:
                self.log(f"Triggering immediate failover for {agent}'s tickets")
                self._cancel_and_reassign_agent_tickets(agent, goal_id)
            return

        if result.status == AgentStatus.TIMEOUT:
            self.cooldowns.record_timeout(agent)
            self.store.set_agent_status(agent, "ERROR")
            for tid in ticket_ids:
                _record_failure(
                    tid,
                    FailureClass.HARD_FAIL,
                    reason=f"timeout:{agent}",
                    confidence=0.15,
                    next_steps=["Retry with reduced scope", "Use alternate agent strategy"],
                )
            return

        if not result.ok:
            self.log(f"{agent} execution error: {result.error}")
            self.store.set_agent_status(agent, "ERROR")
            for tid in ticket_ids:
                _record_failure(
                    tid,
                    FailureClass.HARD_FAIL,
                    reason=f"error:{result.error or 'unknown'}",
                    confidence=0.2,
                    next_steps=["Retry with different strategy", "Inspect stderr excerpts"],
                )
            return

        self.cooldowns.record_success(agent)
        self.store.set_agent_status(agent, "IDLE")

        alerts = self.safety.check_output(result.output, ticket_ids[0] if ticket_ids else 0, agent)
        if alerts:
            for alert in alerts:
                self.events.emit(EventType.SAFETY_FLAGGED, ticket_id=alert.ticket_id,
                               agent=agent, payload={"matched": alert.matched_text})
                self.log(f"SAFETY: {agent} flagged: {alert.matched_text}")
            for tid in ticket_ids:
                _record_failure(
                    tid,
                    FailureClass.HARD_FAIL,
                    reason=f"safety_blocked:{alerts[0].matched_text}",
                    confidence=0.0,
                    next_steps=["Require explicit safety approval"],
                    force_block=True,
                )
            return

        events = extract_events_from_output(result.output)
        uncertainty = self._detect_uncertainty_signal(result.output)
        lower_output = result.output.lower()

        for tid in ticket_ids:
            done_event = None
            blocked_event = None
            review_event = None
            style_event = None
            for ev in events:
                ev_tid = ev.get("id", tid)
                if ev_tid != tid:
                    continue
                etype = str(ev.get("type", ""))
                if etype == "DONE":
                    done_event = ev
                elif etype == "BLOCKED":
                    blocked_event = ev
                elif etype in {"NEEDS_REVIEW", "RISK"}:
                    review_event = ev
                elif etype == "STYLE_FAIL":
                    style_event = ev

            if done_event:
                summary = str(done_event.get("summary", "")).strip()
                artifacts_str = str(done_event.get("artifacts", "")).strip()
                artifacts = [a.strip() for a in artifacts_str.split(",") if a.strip()] if artifacts_str else []
                missing_artifacts = []
                for art in artifacts:
                    if not Path(art).expanduser().exists():
                        missing_artifacts.append(art)

                verified_by: list[str] = ["done_event"]
                if "pytest" in lower_output and "passed" in lower_output:
                    verified_by.append("tests")
                if artifacts and not missing_artifacts:
                    verified_by.append("artifacts_present")

                if missing_artifacts:
                    _record_failure(
                        tid,
                        FailureClass.HARD_FAIL,
                        reason=f"artifact_missing:{','.join(missing_artifacts[:3])}",
                        confidence=0.1,
                        next_steps=["Regenerate missing artifacts", "Re-run verification"],
                        verified_by=verified_by,
                    )
                    continue

                self.store.record_attempt(tid, strategy_used=strategy, success=True, failure_reason="")
                self.store.clear_ticket_cooldown(tid)
                self.store.complete_ticket(tid, artifacts)
                self.events.emit(EventType.TICKET_DONE, ticket_id=tid, agent=agent,
                               payload={"summary": summary})
                self.log(f"Ticket #{tid} DONE by {agent}: {summary}")

                # Store outcome in FV persistent memory
                if self._fv_memory:
                    try:
                        ticket = self.store.get_ticket(tid)
                        fact = f"Ticket #{tid} '{ticket.title if ticket else '?'}' completed by {agent}: {summary}"
                        self._fv_memory.store_fact(fact, importance=0.7, tags={"source": "council", "type": "ticket_done"})
                    except Exception:
                        pass
                self._emit_execution_result(
                    tid,
                    agent,
                    ExecutionResult(
                        success=True,
                        failure_class=FailureClass.NONE,
                        confidence=0.9 if verified_by else 0.75,
                        strategy_used=strategy,
                        verified_by=verified_by,
                        next_steps=["Proceed to dependent tickets"],
                        reason="completed",
                    ),
                )
                continue

            if blocked_event:
                reason = str(blocked_event.get("reason", "blocked")).strip()
                _record_failure(
                    tid,
                    FailureClass.HARD_FAIL,
                    reason=f"blocked:{reason}",
                    confidence=0.2,
                    next_steps=["Unblock objective failure before retry"],
                    force_block=True,
                )
                continue

            if style_event or ("style" in lower_output and "works" in lower_output):
                reason = str((style_event or {}).get("reason", "likely preference mismatch")).strip()
                _record_failure(
                    tid,
                    FailureClass.STYLE_FAIL,
                    reason=f"style:{reason}",
                    confidence=0.6,
                    next_steps=["Produce 2 style variants", "Ask one clarifying question"],
                    review_note=f"STYLE_FAIL: {reason}. Provide variants or ask one clarifying question.",
                )
                continue

            if review_event or uncertainty:
                reason = str((review_event or {}).get("reason", f"uncertainty:{uncertainty or 'unknown'}")).strip()
                _record_failure(
                    tid,
                    FailureClass.SOFT_FAIL,
                    reason=f"soft:{reason}",
                    confidence=0.4,
                    next_steps=["Retry with alternate strategy", "Add acceptance evidence"],
                    review_note=f"SOFT_FAIL: {reason}",
                )
                if review_event:
                    question = str((review_event or {}).get("ask", "Validate risks and propose next step")).strip()
                    risk_level = self._normalize_risk_level((review_event or {}).get("risk", "med"))
                    self._start_peer_consult(
                        ticket_id=tid,
                        requester=agent,
                        requester_output=result.output,
                        reason=reason,
                        risk_level=risk_level,
                        question=question,
                    )
                continue

            _record_failure(
                tid,
                FailureClass.SOFT_FAIL,
                reason="soft:missing_acceptance_evidence",
                confidence=0.35,
                next_steps=["Retry with concrete verification", "Emit DONE with evidence"],
                review_note="SOFT_FAIL: Missing acceptance evidence; retry with explicit verification output.",
            )

    # ── Immediate failover ──

    def _cancel_and_reassign_agent_tickets(self, agent: str, goal_id: int) -> None:
        """Cancel a rate-limited agent's pending work and reassign to other agents."""
        # 1. Remove that agent's entries from _pending_results
        with self._lock:
            self._pending_results = [
                (a, r, tids) for (a, r, tids) in self._pending_results
                if a != agent
            ]

        # 2. Find IN_PROGRESS tickets owned by that agent → reset to NEW
        in_progress = self.store.list_tickets(owner=agent, state=TicketState.IN_PROGRESS)
        if not in_progress:
            return

        reassign_tids = []
        for ticket in in_progress:
            if ticket.goal_id != goal_id:
                continue
            self.store.unassign_ticket(ticket.id)
            self.store.clear_ticket_cooldown(ticket.id)
            reassign_tids.append(ticket.id)
            self.log(f"  Failover: ticket #{ticket.id} unassigned from {agent}")

        if not reassign_tids:
            return

        # 3. Get available agents (excluding rate-limited one)
        other_available = [
            name for name in self.agents
            if name != agent and self.cooldowns.is_available(name)
        ]
        if not other_available:
            self.log(f"  Failover: no other agents available, tickets returned to NEW")
            return

        # 4. Round-robin assign overflow → execute
        reliability = self.cooldowns.get_reliability_scores()
        other_available.sort(key=lambda a: reliability.get(a, 0.5), reverse=True)

        assignments: dict[str, list[int]] = {}
        idx = 0
        for tid in reassign_tids:
            chosen = other_available[idx % len(other_available)]
            self.store.assign_ticket(tid, chosen)
            assignments.setdefault(chosen, []).append(tid)
            self.events.emit(EventType.TICKET_ASSIGNED, ticket_id=tid, agent=chosen)
            self.log(f"  Failover: ticket #{tid} → {chosen}")
            idx += 1

        # Execute in background
        def _do_failover():
            self.execute_assigned(assignments)
        threading.Thread(target=_do_failover, daemon=True).start()

    # ── Phase: Synthesize and update story ──

    def synthesize_and_update_story(self, goal_id: int) -> None:
        """Use Qwen to rewrite story_main.md based on progress."""
        current_story = self.story.read_main()
        recent_events = self.story.get_recent_events(20)
        events_text = "\n".join(
            f"- [{e['ts'][:19]}] {e['type']} {e.get('agent', '')} ticket#{e.get('ticket_id', '')}"
            for e in recent_events
        )

        done_tickets = self.store.list_tickets(goal_id=goal_id, state=TicketState.DONE)
        done_text = "\n".join(f"- #{t.id}: {t.title}" for t in done_tickets) or "None"

        ip_tickets = self.store.list_tickets(goal_id=goal_id, state=TicketState.IN_PROGRESS)
        ip_text = "\n".join(f"- #{t.id}: {t.title} (owner: {t.owner})" for t in ip_tickets) or "None"

        # Gather failed (NEW with failures), BLOCKED, and REVIEW tickets
        failed_parts = []
        new_tickets = self.store.list_tickets(goal_id=goal_id, state=TicketState.NEW)
        for t in new_tickets:
            if t.last_failure_reason:
                failed_parts.append(f"- #{t.id}: {t.title} (FAILED: {t.last_failure_reason})")
            else:
                failed_parts.append(f"- #{t.id}: {t.title} (NEW, unassigned)")
        blocked_tickets = self.store.list_tickets(goal_id=goal_id, state=TicketState.BLOCKED)
        for t in blocked_tickets:
            failed_parts.append(f"- #{t.id}: {t.title} (BLOCKED: {t.blocked_reason or 'unknown'})")
        review_tickets = self.store.list_tickets(goal_id=goal_id, state=TicketState.REVIEW)
        for t in review_tickets:
            failed_parts.append(f"- #{t.id}: {t.title} (REVIEW: {t.review_notes or 'pending'})")
        failed_text = "\n".join(failed_parts) or "None"

        prompt = SYNTHESIS_PROMPT.format(
            current_story=current_story,
            recent_events=events_text,
            completed_tickets=done_text,
            in_progress_tickets=ip_text,
            failed_tickets=failed_text,
        )

        self.log("Synthesizing story update...")
        self.store.set_agent_status("qwen", "BUSY")
        self._agent_phases["qwen"] = "synthesize"
        self._agent_start_times["qwen"] = time.monotonic()
        result = self.qwen.call(prompt, phase="synthesis")
        self.store.set_agent_status("qwen", "IDLE")
        self._agent_phases["qwen"] = "idle"
        self._agent_start_times.pop("qwen", None)

        if result.ok and result.output.strip():
            self.story.rewrite_main(result.output.strip())
            self.events.emit(EventType.SYNTHESIS_COMPLETE, payload={"goal_id": goal_id})
            self.log("Story updated")
        else:
            self.log(f"Synthesis failed: {result.error}")

    # ── Phase: Skeptic review ──

    def trigger_skeptic_review(self, ticket_id: int) -> None:
        """Optional sanity check by Qwen after ticket DONE."""
        ticket = self.store.get_ticket(ticket_id)
        if not ticket:
            return

        agent_output = self._agent_outputs.get(ticket.owner or "", "")
        prompt = REVIEW_PROMPT.format(
            ticket_id=ticket_id,
            ticket_title=ticket.title,
            ticket_description=ticket.description,
            ticket_acceptance=ticket.acceptance,
            agent_output=agent_output[:3000],
        )

        self.store.set_agent_status("qwen", "BUSY")
        result = self.qwen.call(prompt, phase="review")
        self.store.set_agent_status("qwen", "IDLE")

        if result.ok:
            data = _extract_json(result.output)
            if data:
                verdict = data.get("verdict", "approve")
                notes = data.get("notes", "")
                if verdict == "approve":
                    self.log(f"Review passed for ticket #{ticket_id}")
                else:
                    self.store.set_ticket_review(ticket_id, notes)
                    self.log(f"Review requested changes for ticket #{ticket_id}: {notes[:100]}")

    # ── Auto-retry ──

    def enable_auto_retry(self, goal_id: int) -> None:
        self._auto_retry_enabled = True
        self._auto_retry_goal_id = goal_id
        self._last_retry_check = 0.0

    def disable_auto_retry(self) -> None:
        self._auto_retry_enabled = False
        self._auto_retry_goal_id = None

    def _check_and_retry(self) -> None:
        """Called from tick() every retry_interval when enabled.

        Finds NEW tickets with expired cooldowns + idle agents → assigns & executes.
        Uses direct fallback assignment (bypasses Qwen) to avoid repeating bad choices.
        Prefers agents that haven't already failed each ticket.
        """
        now = time.monotonic()
        if now - self._last_retry_check < self._retry_interval:
            return
        self._last_retry_check = now

        if self._retry_in_progress:
            return
        goal_id = self._auto_retry_goal_id
        if goal_id is None:
            return

        # Find NEW tickets that are retriable
        new_tickets = self.store.list_tickets(goal_id=goal_id, state=TicketState.NEW)
        retriable = [t for t in new_tickets if self.store.is_ticket_available(t)]
        if not retriable:
            return

        # Find idle agents with available cooldowns
        idle_agents = []
        for name in self.agents:
            if not self.cooldowns.is_available(name):
                continue
            info = self.store.get_agent(name)
            if info and info.get("status") in ("IDLE", None):
                has_active = any(
                    k.startswith(f"{name}_") and t.is_alive()
                    for k, t in self._active_tasks.items()
                )
                if not has_active:
                    big_count, small_count = self._agent_wip_counts(name)
                    if big_count < self.max_big_per_agent and small_count < self.max_small_per_agent:
                        idle_agents.append(name)

        if not idle_agents:
            return

        self.log(f"Auto-retry: {len(retriable)} retriable tickets, {len(idle_agents)} idle agents")
        self._retry_in_progress = True

        # Direct assignment — bypass Qwen, sort agents by reliability,
        # and prefer agents that haven't already failed each ticket.
        reliability = self.cooldowns.get_reliability_scores()
        idle_agents.sort(key=lambda a: reliability.get(a, 0.5), reverse=True)

        assignments: dict[str, list[int]] = {}
        for ticket in retriable:
            # Get agents who already failed this ticket (from attempts_by_strategy)
            failed_strategies = set()
            if ticket.attempts_by_strategy:
                try:
                    strat_data = json.loads(ticket.attempts_by_strategy) if isinstance(ticket.attempts_by_strategy, str) else ticket.attempts_by_strategy
                    if isinstance(strat_data, dict):
                        failed_strategies = {k.split(":")[0] for k in strat_data}
                except (json.JSONDecodeError, TypeError):
                    pass

            # Prefer agents that haven't tried this ticket yet
            fresh = [a for a in idle_agents if a not in failed_strategies]
            candidates = fresh if fresh else idle_agents

            chosen = self._select_agent_for_ticket(
                ticket, candidates[0], candidates, force_same_strategy=bool(not fresh),
            )
            if chosen:
                self.store.assign_ticket(ticket.id, chosen)
                assignments.setdefault(chosen, []).append(ticket.id)
                self.events.emit(EventType.TICKET_ASSIGNED, ticket_id=ticket.id, agent=chosen)
                self.log(f"  Auto-retry: ticket #{ticket.id} → {chosen}")

        if not assignments:
            self._retry_in_progress = False
            return

        def _do_retry():
            try:
                self.execute_assigned(assignments)
            finally:
                self._retry_in_progress = False

        threading.Thread(target=_do_retry, daemon=True).start()

    # ── Tick: main poll loop entry point ──

    def tick(self) -> None:
        """Called every 50ms from GUI poll loop. Processes pending results and checks state."""
        if self._paused:
            return

        # Process completed EXECUTION results only (ticket_ids non-empty).
        # Proposal results (ticket_ids=[]) are consumed by dedupe_and_merge().
        results_to_process = []
        remaining = []
        with self._lock:
            for entry in self._pending_results:
                agent, result, ticket_ids = entry
                if ticket_ids:  # execution result
                    results_to_process.append(entry)
                else:  # proposal result — leave for dedupe_and_merge
                    remaining.append(entry)
            self._pending_results[:] = remaining

        for agent, result, ticket_ids in results_to_process:
            self.handle_agent_result(agent, result, ticket_ids)

        # Check for stuck tickets (IN_PROGRESS for too long without active thread)
        # Clean up finished threads
        finished = [k for k, t in self._active_tasks.items() if not t.is_alive()]
        for k in finished:
            del self._active_tasks[k]

        # Check cooldown expiration
        for name in list(self.agents.keys()):
            if self.cooldowns.is_available(name):
                agent_info = self.store.get_agent(name)
                if agent_info and agent_info.get("status") == "COOLDOWN":
                    self.store.set_agent_status(name, "IDLE")
                    self.log(f"{name} cooldown expired, now IDLE")

        # Escalate stalled in-progress tickets with no running worker.
        self._escalate_stalled_tickets()

        # Auto-retry check
        if self._auto_retry_enabled:
            self._check_and_retry()

        # Busy→idle transition: trigger synthesis in background
        currently_busy = self.is_busy()
        if self._was_busy and not currently_busy:
            goal_id = self._auto_retry_goal_id
            if goal_id is not None:
                def _bg_synth(gid=goal_id):
                    self.synthesize_and_update_story(gid)
                threading.Thread(target=_bg_synth, daemon=True).start()
        self._was_busy = currently_busy

    def _agent_wip_counts(self, agent_name: str) -> tuple[int, int]:
        owned = self.store.list_tickets(owner=agent_name)
        active_wip = [
            t for t in owned
            if t.state in {TicketState.READY, TicketState.IN_PROGRESS, TicketState.REVIEW}
        ]
        big_count = sum(1 for t in active_wip if t.priority == TicketPriority.BIG)
        small_count = sum(1 for t in active_wip if t.priority == TicketPriority.SMALL)
        return big_count, small_count

    def _escalate_stalled_tickets(self) -> None:
        now = datetime.now(timezone.utc)
        in_progress = self.store.list_tickets(state=TicketState.IN_PROGRESS)
        for ticket in in_progress:
            owner = ticket.owner or ""
            if not owner:
                continue
            task_key = f"{owner}_{ticket.id}"
            active_thread = self._active_tasks.get(task_key)
            if active_thread and active_thread.is_alive():
                continue
            updated = self.store._parse_ts(ticket.updated_at)
            if not updated:
                continue
            age_s = (now - updated).total_seconds()
            if age_s < self.stalled_ticket_timeout_s:
                continue
            reason = (
                f"NEEDS_ATTENTION: stalled {int(age_s)}s with no active worker "
                f"(threshold={self.stalled_ticket_timeout_s}s)"
            )
            self.store.block_ticket(ticket.id, reason)
            self.events.emit(
                EventType.TICKET_BLOCKED,
                ticket_id=ticket.id,
                agent=owner,
                payload={"reason": reason, "stalled_seconds": int(age_s)},
            )
            self.store.set_agent_status(owner, "IDLE")
            self.log(f"Escalated stalled ticket #{ticket.id} ({owner}) -> BLOCKED")

    # ── Full cycle: goal → propose → dedupe → assign → execute → synthesize ──

    def run_full_cycle(self, goal_text: str) -> dict[str, Any]:
        """Run the full orchestration cycle and return structured outcome metadata."""
        goal = self.store.create_goal(goal_text)
        self.log(f"Created goal #{goal.id}: {goal_text[:80]}")

        # Phase 1: Request proposals
        self.request_proposals(goal.id, goal_text)

        # Wait for proposals — proceed after 60s if at least one arrived,
        # or 180s absolute max. This prevents one slow agent (e.g. Claude)
        # from blocking the entire pipeline.
        min_wait = 60   # proceed after this if we have ≥1 proposal
        max_wait = 180  # absolute deadline
        start = time.monotonic()
        while True:
            elapsed = time.monotonic() - start
            active = [t for t in self._active_tasks.values() if t.is_alive()]
            if not active:
                break
            # If we've waited long enough and have at least one proposal, proceed
            with self._lock:
                have_proposals = len(self._pending_results) > 0
            if elapsed >= min_wait and have_proposals:
                still_running = [k for k, t in self._active_tasks.items() if t.is_alive()]
                self.log(f"Proceeding with {len(self._pending_results)} proposals "
                         f"(still waiting: {', '.join(still_running)})")
                break
            if elapsed >= max_wait:
                self.log(f"Proposal deadline reached after {max_wait}s")
                break
            time.sleep(0.5)

        # Phase 2: Dedupe
        tickets = self.dedupe_and_merge(goal.id, goal_text)
        if not tickets:
            self.log("FALLBACK CHAIN exhausted: no tickets created")
            self.events.emit(
                EventType.GOAL_PARTIAL,
                payload={
                    "goal_id": goal.id,
                    "stage": "dedupe",
                    "reason": "no_tickets",
                },
            )
            return {
                "goal_id": goal.id,
                "should_synthesize": False,
                "outcome": "partial",
                "stage": "dedupe",
                "reason": "no_tickets",
                "ticket_count": 0,
            }

        # Phase 3: Assign
        assignments = self.assign_tickets(goal.id)
        assigned_count = sum(len(v) for v in assignments.values())
        if not assignments or assigned_count == 0:
            self.log("FALLBACK CHAIN ended at assign: no assignments available")
            self.events.emit(
                EventType.GOAL_PARTIAL,
                payload={
                    "goal_id": goal.id,
                    "stage": "assign",
                    "reason": "no_assignments",
                    "ticket_count": len(tickets),
                },
            )
            return {
                "goal_id": goal.id,
                "should_synthesize": False,
                "outcome": "partial",
                "stage": "assign",
                "reason": "no_assignments",
                "ticket_count": len(tickets),
                "assignment_count": 0,
            }

        # Phase 4: Execute
        self.events.emit(
            EventType.GOAL_EXECUTION_STARTED,
            payload={
                "goal_id": goal.id,
                "ticket_count": len(tickets),
                "assignment_count": assigned_count,
            },
        )
        self.execute_assigned(assignments)

        return {
            "goal_id": goal.id,
            "should_synthesize": True,
            "outcome": "execution_started",
            "ticket_count": len(tickets),
            "assignment_count": assigned_count,
        }

    def get_agent_output(self, agent: str) -> str:
        """Get the latest output for an agent."""
        return self._agent_outputs.get(agent, "")

    def get_all_outputs(self) -> dict[str, str]:
        """Get all agent outputs."""
        return dict(self._agent_outputs)

    def get_agent_phase(self, agent: str) -> str:
        """Get the current phase for an agent."""
        return self._agent_phases.get(agent, "idle")

    def get_agent_elapsed(self, agent: str) -> float:
        """Get seconds elapsed since agent started current work, or 0 if idle."""
        start = self._agent_start_times.get(agent)
        if start is None:
            return 0.0
        return time.monotonic() - start

    def get_all_phases(self) -> dict[str, str]:
        """Get all agent phases."""
        return dict(self._agent_phases)

    def is_busy(self) -> bool:
        """Check if any agents are currently executing."""
        return any(t.is_alive() for t in self._active_tasks.values())
