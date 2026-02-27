"""CouncilApp v0.3 — Ticket-based orchestration GUI with Tkinter."""

from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from typing import Any, Optional

from .. import __version__
from ..config import Config
from ..orchestrator.agents import QwenAgent, create_agent, AgentError
from ..orchestrator.events import EventStore, EventType
from ..orchestrator.handoff import HandoffBuilder
from ..orchestrator.rate_limit import CooldownTracker
from ..orchestrator.router import OrchestratorRouter
from ..orchestrator.safety import SafetyGate
from ..orchestrator.story import StoryManager
from ..orchestrator.tickets import TicketStore, TicketState
from . import theme
from .dialogs import SafetyApprovalDialog
from .input_bar import InputBar
from .panels import (
    AgentOutputPanel,
    AgentStatusPanel,
    EventLogPanel,
    FlowMetricsPanel,
    ProjectSelector,
    SafetyApprovalBar,
    StoryPanel,
    TicketPanel,
)


class CouncilApp(tk.Tk):
    """Main Council v0.3 GUI application — ticket-based orchestration."""

    def __init__(self, workspace: Optional[str] = None, mock: bool = False):
        super().__init__()

        # Log Tkinter callback exceptions to stderr instead of silently swallowing
        self.report_callback_exception = self._report_callback_exception

        # --- Configuration ---
        self.workspace = workspace or os.getcwd()
        config_dir = Path(self.workspace) / ".council"
        config_dir.mkdir(parents=True, exist_ok=True)
        self.config = Config(base_dir=config_dir)
        self.config.set("workspace", self.workspace)

        self._mock = mock or self.config.get("mock_mode", False)

        # --- Data directories ---
        data_dir = Path(self.workspace) / ".council" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "handoffs").mkdir(exist_ok=True)

        # --- Window setup ---
        self.title(f"COUNCIL v{__version__}")
        gui_cfg = self.config.get("gui", {})
        w = gui_cfg.get("window_width", 1400) if isinstance(gui_cfg, dict) else 1400
        h = gui_cfg.get("window_height", 800) if isinstance(gui_cfg, dict) else 800
        self.geometry(f"{w}x{h}")
        self.configure(bg=theme.BG_DARK)
        self.minsize(900, 600)

        # --- Initialize orchestrator stack ---
        db_path = data_dir / "council.db"
        self.store = TicketStore(db_path)

        self.event_store = EventStore(
            db_conn=self.store._conn,
            log_path=data_dir / "story_log.ndjson",
            lock=self.store._lock,
        )

        self.story = StoryManager(data_dir, self.event_store)
        self.story.init()

        self.safety = SafetyGate()
        self.cooldowns = CooldownTracker()
        self.handoffs = HandoffBuilder(data_dir / "handoffs")

        # --- Create agents ---
        cwd = self.workspace
        agents: dict[str, Any] = {}
        for name in ["codex", "claude", "gemini"]:
            agent_cfg = self.config.get(f"agents.{name}", {}) or {}
            if not isinstance(agent_cfg, dict):
                agent_cfg = {}
            try:
                agents[name] = create_agent(name, agent_cfg, cwd=cwd, mock=self._mock)
            except AgentError:
                agents[name] = create_agent(name, agent_cfg, cwd=cwd, mock=True)

        qwen_cfg = self.config.get("agents.qwen", {}) or {}
        if not isinstance(qwen_cfg, dict):
            qwen_cfg = {}
        try:
            qwen = create_agent("qwen", qwen_cfg, cwd=cwd, mock=self._mock)
        except AgentError:
            qwen = create_agent("qwen", qwen_cfg, cwd=cwd, mock=True)

        # --- Orchestrator log queue (for GUI display) ---
        self._log_queue: queue.Queue[str] = queue.Queue()

        def _on_router_log(msg: str):
            self._log_queue.put(msg)

        orchestrator_cfg = self.config.get("orchestrator", {})
        if not isinstance(orchestrator_cfg, dict):
            orchestrator_cfg = {}

        # --- Router ---
        self.router = OrchestratorRouter(
            store=self.store,
            event_store=self.event_store,
            story=self.story,
            safety=self.safety,
            cooldowns=self.cooldowns,
            handoffs=self.handoffs,
            agents=agents,
            qwen=qwen,
            max_big_per_agent=int(orchestrator_cfg.get("max_big_per_agent", 1)),
            max_small_per_agent=int(orchestrator_cfg.get("max_small_per_agent", 2)),
            stalled_ticket_timeout_s=int(orchestrator_cfg.get("stalled_ticket_timeout_s", 300)),
            on_log=_on_router_log,
        )

        # --- State ---
        self._current_goal_id: Optional[int] = None
        self._phase: str = "Idle"
        self._cycle_thread: Optional[threading.Thread] = None
        # Thread-safe phase update: background thread sets this, poll loop reads it
        self._pending_phase: Optional[str] = None
        self._pending_refresh_projects: bool = False

        # --- Build UI ---
        self._build_ui()

        # --- Start poll loop ---
        self._poll()

        # --- Warmup Qwen (if not mock) ---
        if not self._mock and isinstance(qwen, QwenAgent):
            threading.Thread(target=lambda: qwen.warmup(), daemon=True).start()

        # --- Clean shutdown ---
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        """Construct the full GUI layout.

        Layout:
        +================================================================+
        | COUNCIL v0.3  | Project: [dropdown] [+New]  [Propose][Start]   |
        +================================================================+
        | Goal: [multiline input]                                [Send]  |
        +------------------+--------+------------------------------------+
        | TICKETS          | AGENTS | STORY / EVENT LOG                  |
        | [treeview]       | codex  | [story or events, stacked]         |
        |                  | claude |                                    |
        | [detail]         | gemini |                                    |
        +------------------+--------+------------------------------------+
        | CODEX            | CLAUDE          | GEMINI         | QWEN     |
        | [terminal]       | [terminal]      | [terminal]     | [term]   |
        +------------------+-----------------+----------------+----------+
        | Safety bar (when needed)                                       |
        +================================================================+
        """

        # ═══ Title bar ═══
        title_bar = tk.Frame(self, bg=theme.BG_STATUS)
        title_bar.pack(fill=tk.X)

        tk.Label(
            title_bar,
            text=f"  COUNCIL v{__version__}",
            font=theme.FONT_TITLE,
            fg=theme.FG_ACCENT,
            bg=theme.BG_STATUS,
        ).pack(side=tk.LEFT, padx=theme.PAD)

        if self._mock:
            tk.Label(
                title_bar,
                text=" [MOCK] ",
                font=theme.FONT_STATUS,
                fg=theme.FG_WARNING,
                bg=theme.BG_STATUS,
            ).pack(side=tk.LEFT)

        # Project selector (left of buttons)
        self.project_selector = ProjectSelector(
            title_bar,
            on_select=self._on_project_select,
            on_new=self._on_project_new,
        )
        self.project_selector.pack(side=tk.LEFT, padx=theme.PAD)

        # Phase label
        self._phase_label = tk.Label(
            title_bar,
            text="Idle",
            font=theme.FONT_STATUS,
            fg=theme.FG_DIM,
            bg=theme.BG_STATUS,
        )
        self._phase_label.pack(side=tk.LEFT, padx=theme.PAD)

        # Right-side buttons
        btn_frame = tk.Frame(title_bar, bg=theme.BG_STATUS)
        btn_frame.pack(side=tk.RIGHT)

        self._retry_btn = tk.Button(
            btn_frame, text="Retry", font=theme.FONT_MONO_SMALL,
            bg=theme.BG_BUTTON, fg=theme.FG_WARNING,
            activebackground=theme.BG_BUTTON_HOVER, relief=tk.FLAT,
            padx=8, pady=1, command=self._retry_failed,
        )
        self._retry_btn.pack(side=tk.LEFT, padx=theme.PAD_SMALL, pady=theme.PAD_SMALL)

        self._pause_btn = tk.Button(
            btn_frame, text="Pause", font=theme.FONT_MONO_SMALL,
            bg=theme.BG_BUTTON, fg=theme.FG_TEXT,
            activebackground=theme.BG_BUTTON_HOVER, relief=tk.FLAT,
            padx=8, pady=1, command=self._toggle_pause,
        )
        self._pause_btn.pack(side=tk.LEFT, padx=theme.PAD_SMALL, pady=theme.PAD_SMALL)

        self._start_btn = tk.Button(
            btn_frame, text="Start", font=theme.FONT_MONO_SMALL,
            bg=theme.BG_BUTTON, fg=theme.FG_SUCCESS,
            activebackground=theme.BG_BUTTON_HOVER, relief=tk.FLAT,
            padx=8, pady=1, command=self._start_execution,
        )
        self._start_btn.pack(side=tk.LEFT, padx=theme.PAD_SMALL, pady=theme.PAD_SMALL)

        self._propose_btn = tk.Button(
            btn_frame, text="Propose", font=theme.FONT_MONO_SMALL,
            bg=theme.BG_BUTTON_ACCENT, fg=theme.FG_BUTTON_ACCENT,
            activebackground=theme.FG_SUCCESS, relief=tk.FLAT,
            padx=8, pady=1, command=self._propose_from_button,
        )
        self._propose_btn.pack(side=tk.LEFT, padx=theme.PAD_SMALL, pady=theme.PAD_SMALL)

        # ═══ Input bar (packed at bottom) ═══
        self.input_bar = InputBar(self, on_submit=self._on_input)
        self.input_bar.pack(fill=tk.X, side=tk.BOTTOM)

        # ═══ Safety approval bar (hidden, packed above input bar) ═══
        self.safety_bar = SafetyApprovalBar(
            self,
            on_approve=self._on_safety_approve,
            on_deny=self._on_safety_deny,
        )

        # ═══ Main content: vertical split — top workspace / bottom terminals ═══
        main_pane = tk.PanedWindow(
            self, orient=tk.VERTICAL,
            bg=theme.BG_DARK, sashwidth=5, sashrelief=tk.FLAT,
        )
        main_pane.pack(fill=tk.BOTH, expand=True, padx=theme.PAD, pady=theme.PAD)

        # ─── Top half: Tickets | Agents | Story+Events ───
        top_area = tk.Frame(main_pane, bg=theme.BG_DARK)
        main_pane.add(top_area, minsize=200)

        top_pane = tk.PanedWindow(
            top_area, orient=tk.HORIZONTAL,
            bg=theme.BG_DARK, sashwidth=4, sashrelief=tk.FLAT,
        )
        top_pane.pack(fill=tk.BOTH, expand=True)

        # Ticket panel
        self.ticket_panel = TicketPanel(top_pane, on_select=self._on_ticket_select)
        top_pane.add(self.ticket_panel, width=340, minsize=180)

        # Agent status
        self.agent_status_panel = AgentStatusPanel(top_pane)
        top_pane.add(self.agent_status_panel, width=160, minsize=100)

        # Story + Event log (stacked)
        right_frame = tk.Frame(top_pane, bg=theme.BG_DARK)
        top_pane.add(right_frame, width=400, minsize=200)

        self.flow_panel = FlowMetricsPanel(right_frame)
        self.flow_panel.pack(fill=tk.X, pady=(0, theme.PAD_SMALL))

        self.story_panel = StoryPanel(right_frame)
        self.story_panel.pack(fill=tk.BOTH, expand=True, pady=(0, theme.PAD_SMALL))

        self.event_log_panel = EventLogPanel(right_frame)
        self.event_log_panel.pack(fill=tk.BOTH, expand=True)

        # ─── Bottom half: 4 terminal output panels across ───
        bottom_area = tk.Frame(main_pane, bg=theme.BG_DARK)
        main_pane.add(bottom_area, minsize=150)

        bottom_area.rowconfigure(0, weight=1)
        for col in range(4):
            bottom_area.columnconfigure(col, weight=1)

        self.output_panels: dict[str, AgentOutputPanel] = {}
        agents_order = ["codex", "claude", "gemini", "qwen"]
        for col, name in enumerate(agents_order):
            panel = AgentOutputPanel(bottom_area, name)
            panel.grid(row=0, column=col, sticky="nsew",
                       padx=theme.PAD_SMALL, pady=theme.PAD_SMALL)
            self.output_panels[name] = panel

        # Focus input on start
        self.after(100, self.input_bar.focus_entry)

        # Load existing goals into project selector
        self._refresh_project_selector()

    # ──────────────────────────────────────────
    # Project selector
    # ──────────────────────────────────────────

    def _refresh_project_selector(self) -> None:
        """Reload the project dropdown from the database."""
        goals = self.store.list_goals()
        goal_dicts = [{"id": g.id, "text": g.text, "status": g.status.value} for g in goals]
        self.project_selector.update_goals(goal_dicts)
        if self._current_goal_id:
            self.project_selector.select_goal(self._current_goal_id)

    def _on_project_select(self, goal_id: int) -> None:
        """Switch to a different project/goal."""
        self.router.disable_auto_retry()
        self._current_goal_id = goal_id
        # Clear output panels when switching projects
        for panel in self.output_panels.values():
            panel.clear()
        self._log(f"Switched to goal #{goal_id}")

    def _on_project_new(self) -> None:
        """Clear state for a new goal."""
        self._current_goal_id = None
        for panel in self.output_panels.values():
            panel.clear()
        self._set_phase("Idle")
        self.input_bar.focus_entry()
        self._log("Ready for new goal")

    # ──────────────────────────────────────────
    # Input handling
    # ──────────────────────────────────────────

    def _on_input(self, text: str) -> None:
        """Handle user input — slash commands or goals."""
        if text.startswith("/"):
            self._handle_command(text)
            return

        if self.router.is_busy():
            # While a cycle is running, plain input becomes live feedback.
            self.router.add_feedback(text, author="user")
            self._log(f"[feedback] broadcast: {text[:160]}")
            return

        # Goal submission → trigger full cycle
        self._submit_goal(text)

    def _submit_goal(self, goal_text: str) -> None:
        """Submit a goal and run the full orchestration cycle."""
        self.router.disable_auto_retry()
        self._set_phase("Proposing...")

        # Clear output panels
        for panel in self.output_panels.values():
            panel.clear()

        self._log("Goal submitted: " + goal_text[:100])

        # Run full cycle in background thread.
        # Thread-safety: NEVER call self.after/after_idle from non-main thread.
        # Instead set self._pending_phase / self._pending_refresh_projects,
        # which the 50ms poll loop picks up on the main thread.
        def _cycle():
            try:
                cycle = self.router.run_full_cycle(goal_text)
                goal_id = cycle["goal_id"]
                self._current_goal_id = goal_id
                self._pending_refresh_projects = True
                self.router.enable_auto_retry(goal_id)

                # Wait for execution to complete only when execution started.
                if cycle.get("outcome") == "execution_started":
                    self._pending_phase = "Executing..."
                    import time
                    deadline = time.monotonic() + 600  # 10 min max
                    while time.monotonic() < deadline:
                        if not self.router.is_busy():
                            break
                        time.sleep(1)

                # Always synthesize story — even on partial completion
                self._pending_phase = "Synthesizing..."
                self.router.synthesize_and_update_story(goal_id)

                if cycle.get("should_synthesize"):
                    self._pending_phase = "Done"
                else:
                    reason = cycle.get("reason", "unknown")
                    stage = cycle.get("stage", "unknown")
                    self.router.log(f"Cycle partial at {stage}: {reason}")
                    self._pending_phase = "Partial"
            except Exception as e:
                self.router.log(f"Cycle error: {e}")
                self._pending_phase = "Error"

        self._cycle_thread = threading.Thread(target=_cycle, daemon=True)
        self._cycle_thread.start()

    def _propose_from_button(self) -> None:
        """Handle Propose button click — submit goal from input bar."""
        text = self.input_bar.get_text()
        if text:
            self._submit_goal(text)

    def _start_execution(self) -> None:
        """Handle Start button — assign and execute current tickets."""
        if self._current_goal_id is None:
            self._log("No goal to start. Submit a goal first.")
            return

        self._set_phase("Executing")
        self.router.enable_auto_retry(self._current_goal_id)

        def _execute():
            assignments = self.router.assign_tickets(self._current_goal_id)
            if assignments:
                self.router.execute_assigned(assignments)
            else:
                self._log("No tickets to assign")

        threading.Thread(target=_execute, daemon=True).start()

    def _toggle_pause(self) -> None:
        """Toggle pause/resume."""
        if self.router.is_paused:
            self.router.resume()
            self._pause_btn.configure(text="Pause", fg=theme.FG_TEXT)
        else:
            self.router.pause()
            self._pause_btn.configure(text="Resume", fg=theme.FG_WARNING)

    def _retry_failed(self) -> None:
        """Retry failed and review tickets for the current goal."""
        if self._current_goal_id is None:
            self._log("No goal selected. Submit a goal first.")
            return

        goal_id = self._current_goal_id

        # Find NEW tickets with failures + REVIEW tickets
        new_tickets = self.store.list_tickets(goal_id=goal_id, state=TicketState.NEW)
        failed_new = [t for t in new_tickets if t.last_failure_reason]

        review_tickets = self.store.list_tickets(goal_id=goal_id, state=TicketState.REVIEW)

        if not failed_new and not review_tickets:
            self._log("No failed or review tickets to retry.")
            return

        # Reset REVIEW → NEW
        for t in review_tickets:
            self.store.unassign_ticket(t.id)
            self._log(f"  Reset ticket #{t.id} from REVIEW → NEW")

        # Clear all ticket cooldowns for retriable tickets
        for t in failed_new + review_tickets:
            self.store.clear_ticket_cooldown(t.id)

        count = len(failed_new) + len(review_tickets)
        self._log(f"Retrying {count} tickets...")
        self._set_phase("Retrying...")
        self.router.enable_auto_retry(goal_id)

        def _do_retry():
            assignments = self.router.assign_tickets(goal_id)
            if assignments:
                self.router.execute_assigned(assignments)
            else:
                self.router.log("Retry: no assignments possible right now (auto-retry will keep trying)")

        threading.Thread(target=_do_retry, daemon=True).start()

    # ──────────────────────────────────────────
    # Poll loop
    # ──────────────────────────────────────────

    def _poll(self) -> None:
        """Poll every 50ms: tick router, refresh UI."""
        try:
            self._poll_inner()
        except Exception as e:
            import traceback, sys
            print(f"[poll error] {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        finally:
            # Always reschedule — never let the poll loop die
            self.after(50, self._poll)

    def _poll_inner(self) -> None:
        """Actual poll logic (separated so _poll can catch errors)."""
        # Tick the router
        self.router.tick()

        # Consume thread-safe phase updates from _cycle thread
        if self._pending_phase is not None:
            self._set_phase(self._pending_phase)
            self._pending_phase = None
        if self._pending_refresh_projects:
            self._pending_refresh_projects = False
            self._refresh_project_selector()

        # Process router log messages → qwen panel (orchestrator brain)
        try:
            while True:
                msg = self._log_queue.get_nowait()
                self._log(msg)
        except queue.Empty:
            pass

        # Refresh ticket panel
        tickets = self.store.list_tickets(goal_id=self._current_goal_id)
        self.ticket_panel.update_tickets([t.to_dict() for t in tickets])
        self.flow_panel.update_metrics(self.store.get_flow_metrics(self._current_goal_id))

        # Refresh agent status (with phase + elapsed time)
        agents = self.store.list_agents()
        phases = self.router.get_all_phases()
        elapsed = {name: self.router.get_agent_elapsed(name)
                   for name in ["codex", "claude", "gemini", "qwen"]}
        self.agent_status_panel.update_all(agents, phases=phases, elapsed=elapsed)

        # Refresh story
        story_content = self.story.read_main()
        self.story_panel.update_content(story_content)

        # Refresh event log
        events = self.event_store.tail(50)
        self.event_log_panel.update_events(events)

        # Refresh agent outputs (only update when content changes to avoid flicker)
        all_outputs = self.router.get_all_outputs()
        for name in ["codex", "claude", "gemini", "qwen"]:
            output = all_outputs.get(name, "")
            panel = self.output_panels.get(name)
            if panel:
                if output:
                    current = panel.text.get("1.0", "end-1c")
                    truncated = output[:20000]
                    if current != truncated:
                        panel.set_text(truncated)
                # Update status indicator
                phase = phases.get(name, "idle")
                agent_elapsed = elapsed.get(name, 0.0)
                if phase and phase != "idle":
                    secs = int(agent_elapsed)
                    panel.set_status(f"{phase} {secs}s")
                else:
                    panel.set_status("")

        # Check safety alerts
        pending = self.safety.get_pending()
        if pending:
            alert = pending[0]
            self.safety_bar.show_alert(alert.id, alert.agent, alert.matched_text)
        elif not self.safety.has_pending():
            self.safety_bar.hide()

    # ──────────────────────────────────────────
    # Safety callbacks
    # ──────────────────────────────────────────

    def _on_safety_approve(self, alert_id: int) -> None:
        self.safety.approve(alert_id)
        self.event_store.emit(EventType.SAFETY_APPROVED, payload={"alert_id": alert_id})
        self._log(f"Safety alert #{alert_id} approved")
        # Unblock associated tickets
        for alert in self.safety._alerts:
            if alert.id == alert_id:
                ticket = self.store.get_ticket(alert.ticket_id)
                if ticket and ticket.state == TicketState.BLOCKED:
                    self.store.update_ticket_state(alert.ticket_id, TicketState.IN_PROGRESS)
                break

    def _on_safety_deny(self, alert_id: int) -> None:
        self.safety.deny(alert_id)
        self.event_store.emit(EventType.SAFETY_DENIED, payload={"alert_id": alert_id})
        self._log(f"Safety alert #{alert_id} denied")

    # ──────────────────────────────────────────
    # Ticket selection
    # ──────────────────────────────────────────

    def _on_ticket_select(self, ticket_id: int) -> None:
        """Show ticket details when clicked in the tree."""
        ticket = self.store.get_ticket(ticket_id)
        if ticket:
            detail = (
                f"Ticket #{ticket.id}: {ticket.title}\n"
                f"State: {ticket.state.value}  Priority: {ticket.priority.value}\n"
                f"Owner: {ticket.owner or 'unassigned'}\n"
                f"Description: {ticket.description}\n"
                f"Acceptance: {ticket.acceptance}\n"
            )
            if ticket.blocked_reason:
                detail += f"Blocked: {ticket.blocked_reason}\n"
            if ticket.review_notes:
                detail += f"Review: {ticket.review_notes}\n"
            if ticket.artifacts:
                detail += f"Artifacts: {', '.join(ticket.artifacts)}\n"
            self.ticket_panel.show_detail(detail)

    # ──────────────────────────────────────────
    # Slash commands
    # ──────────────────────────────────────────

    def _handle_command(self, text: str) -> None:
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()

        if cmd == "/clear":
            for panel in self.output_panels.values():
                panel.clear()
            self._set_phase("Idle")
        elif cmd in ("/feedback", "/fb") or cmd.startswith("/feedback ") or cmd.startswith("/fb "):
            self._handle_feedback_command(text)
        elif cmd == "/status":
            self._show_status()
        elif cmd == "/help":
            self._show_help()
        else:
            self._log(f"Unknown command: {cmd}. Type /help for commands.")

    def _show_help(self) -> None:
        self._log(
            "=== Council v0.3 Commands ===\n"
            "  /clear    — Clear all panels\n"
            "  /feedback [agent|all] <msg> — Send live feedback\n"
            "  /fb [agent|all] <msg>       — Alias for /feedback\n"
            "  /status   — Show agent status\n"
            "  /help     — Show this help\n"
            "\nType any text to submit a goal.\n"
            "While agents are running, plain text is treated as broadcast feedback.\n"
            "Use [Propose] to break down, [Start] to execute, [Pause] to pause."
        )

    def _handle_feedback_command(self, text: str) -> None:
        parts = text.split(maxsplit=2)
        if len(parts) < 2:
            self._log("Usage: /feedback [agent|all] <message>")
            return

        target = "all"
        message = ""
        if len(parts) == 2:
            message = parts[1]
        else:
            target = parts[1].lower()
            message = parts[2]

        valid_targets = {"all", "codex", "claude", "gemini", "qwen"}
        if target not in valid_targets:
            message = " ".join(parts[1:])
            target = "all"

        if target == "all":
            self.router.add_feedback(message, author="user")
            self._log(f"[feedback] broadcast: {message[:160]}")
        else:
            self.router.add_feedback(message, target_agent=target, author="user")
            self._log(f"[feedback] {target}: {message[:160]}")

    def _show_status(self) -> None:
        agents = self.store.list_agents()
        lines = [f"Phase: {self._phase}", f"Mock: {self._mock}", ""]
        for a in agents:
            lines.append(f"  {a['name']}: {a['status']}")
        if self._current_goal_id:
            tickets = self.store.list_tickets(goal_id=self._current_goal_id)
            lines.append(f"\nTickets: {len(tickets)}")
            for t in tickets:
                lines.append(f"  #{t.id} [{t.state.value}] {t.title}")
        self._log("\n".join(lines))

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────

    def _report_callback_exception(self, exc_type, exc_value, exc_tb) -> None:
        """Log Tkinter callback exceptions to stderr (default Tk swallows them)."""
        import sys, traceback
        print(f"[Tkinter callback error] {exc_type.__name__}: {exc_value}", file=sys.stderr)
        traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr)

    def _set_phase(self, phase: str) -> None:
        self._phase = phase
        self._phase_label.configure(text=phase)

    def _log(self, msg: str) -> None:
        """Log to qwen panel (the orchestrator brain)."""
        panel = self.output_panels.get("qwen")
        if panel:
            panel.append_text(msg + "\n")

    def _on_close(self) -> None:
        """Clean shutdown."""
        self.store.close()
        self.destroy()


def run_gui(workspace: Optional[str] = None, mock: bool = False) -> None:
    """Launch the Council v0.3 GUI."""
    import sys, traceback
    ws = workspace or os.getcwd()
    crash_path = os.path.join(ws, ".council", "crash.log")
    os.makedirs(os.path.dirname(crash_path), exist_ok=True)

    print(f"Council v{__version__} starting (workspace={ws}, mock={mock})", file=sys.stderr)

    try:
        app = CouncilApp(workspace=workspace, mock=mock)
        print("Council GUI ready", file=sys.stderr)
        app.mainloop()
        print("Council GUI exited normally", file=sys.stderr)
    except tk.TclError as exc:
        with open(crash_path, "a") as f:
            f.write(f"\n--- Crash at {__import__('datetime').datetime.now().isoformat()} ---\n")
            traceback.print_exc(file=f)
        traceback.print_exc()
        print(f"\nCrash log written to: {crash_path}", file=sys.stderr)
        if "display" in str(exc).lower():
            print("GUI display unavailable. Set DISPLAY for desktop session or run with: xvfb-run -a council gui", file=sys.stderr)
        raise
    except Exception:
        with open(crash_path, "a") as f:
            f.write(f"\n--- Crash at {__import__('datetime').datetime.now().isoformat()} ---\n")
            traceback.print_exc(file=f)
        traceback.print_exc()
        print(f"\nCrash log written to: {crash_path}", file=sys.stderr)
        raise
