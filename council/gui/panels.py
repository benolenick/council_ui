"""GUI panels — TicketPanel, AgentStatusPanel, StoryPanel, EventLogPanel,
AgentOutputPanel, SafetyApprovalBar."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Optional

from . import theme


def _block_text_edits(event: tk.Event) -> Optional[str]:
    """Allow navigation and copy/select shortcuts, block edits."""
    navigation_keys = {
        "Left", "Right", "Up", "Down",
        "Home", "End", "Prior", "Next",
        "Shift_L", "Shift_R", "Control_L", "Control_R",
    }
    ctrl_pressed = (event.state & 0x4) != 0
    if ctrl_pressed and event.keysym.lower() in {"c", "a"}:
        return None
    if event.keysym in navigation_keys:
        return None
    return "break"


def _make_copyable_readonly(text_widget: tk.Text) -> None:
    """Configure a Text widget as non-editable while preserving selection/copy."""
    text_widget.configure(state=tk.NORMAL, insertwidth=0, takefocus=True)
    text_widget.bind("<Key>", _block_text_edits, add="+")
    text_widget.bind("<<Paste>>", lambda _: "break", add="+")
    text_widget.bind("<<Cut>>", lambda _: "break", add="+")


class TicketPanel(tk.Frame):
    """Treeview showing tickets with detail pane below."""

    def __init__(self, parent: tk.Widget, on_select: Optional[Callable[[int], None]] = None, **kwargs):
        super().__init__(parent, bg=theme.BG_PANEL, **kwargs)
        self._on_select = on_select

        # Header
        tk.Label(
            self, text="  TICKETS", font=theme.FONT_TITLE,
            fg=theme.FG_ACCENT, bg=theme.BG_PANEL, anchor="w",
        ).pack(fill=tk.X, padx=theme.PAD, pady=(theme.PAD, 0))

        # Treeview
        tree_frame = tk.Frame(self, bg=theme.BG_PANEL)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=theme.PAD, pady=theme.PAD_SMALL)

        columns = ("id", "title", "state", "owner", "pri")
        self.tree = ttk.Treeview(
            tree_frame, columns=columns, show="headings", height=8,
            selectmode="browse",
        )
        self.tree.heading("id", text="ID")
        self.tree.heading("title", text="Title")
        self.tree.heading("state", text="State")
        self.tree.heading("owner", text="Owner")
        self.tree.heading("pri", text="Pri")

        self.tree.column("id", width=30, minwidth=30, stretch=False)
        self.tree.column("title", width=180, minwidth=100)
        self.tree.column("state", width=70, minwidth=60, stretch=False)
        self.tree.column("owner", width=60, minwidth=50, stretch=False)
        self.tree.column("pri", width=45, minwidth=40, stretch=False)

        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        # Detail pane (below tree)
        self.detail = tk.Text(
            self, height=4, wrap=tk.WORD,
            font=theme.FONT_MONO_SMALL,
            bg=theme.BG_DARK, fg=theme.FG_TEXT,
            borderwidth=1, relief=tk.SOLID,
            highlightthickness=0,
            padx=4, pady=4,
        )
        self.detail.pack(fill=tk.X, padx=theme.PAD, pady=(0, theme.PAD))
        _make_copyable_readonly(self.detail)

        # Style the treeview for dark theme
        self._style_treeview()

    def _style_treeview(self) -> None:
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview",
            background=theme.BG_DARK,
            foreground=theme.FG_TEXT,
            fieldbackground=theme.BG_DARK,
            font=theme.FONT_MONO_SMALL,
            rowheight=22,
        )
        style.configure("Treeview.Heading",
            background=theme.BG_PANEL,
            foreground=theme.FG_ACCENT,
            font=theme.FONT_MONO_SMALL,
        )
        style.map("Treeview",
            background=[("selected", theme.BG_BUTTON)],
            foreground=[("selected", theme.FG_TEXT)],
        )

    def _on_tree_select(self, event=None) -> None:
        selection = self.tree.selection()
        if selection:
            item = self.tree.item(selection[0])
            ticket_id = item["values"][0] if item["values"] else None
            if ticket_id and self._on_select:
                self._on_select(int(ticket_id))

    def update_tickets(self, tickets: list[dict[str, Any]]) -> None:
        """Refresh the ticket list. Each dict has: id, title, state, owner, priority.
        Only redraws if data has changed to avoid flicker."""
        # Build fingerprint of current data to detect changes
        new_fingerprint = tuple(
            (t.get("id"), t.get("title", "")[:30], t.get("state"), t.get("owner"), t.get("priority"))
            for t in tickets
        )
        if hasattr(self, "_last_fingerprint") and self._last_fingerprint == new_fingerprint:
            return
        self._last_fingerprint = new_fingerprint

        # Remember selection
        selection = self.tree.selection()
        selected_id = None
        if selection:
            vals = self.tree.item(selection[0])["values"]
            selected_id = vals[0] if vals else None

        # Clear and repopulate
        for item in self.tree.get_children():
            self.tree.delete(item)

        for t in tickets:
            tid = t.get("id", "")
            title = t.get("title", "")[:30]
            state = t.get("state", "")
            owner = t.get("owner", "") or ""
            pri = t.get("priority", "")
            iid = self.tree.insert("", tk.END, values=(tid, title, state, owner, pri))

            # Color-code by state
            if state == "DONE":
                self.tree.item(iid, tags=("done",))
            elif state == "IN_PROGRESS":
                self.tree.item(iid, tags=("inprogress",))
            elif state == "BLOCKED":
                self.tree.item(iid, tags=("blocked",))

        self.tree.tag_configure("done", foreground=theme.FG_SUCCESS)
        self.tree.tag_configure("inprogress", foreground=theme.FG_WARNING)
        self.tree.tag_configure("blocked", foreground=theme.FG_ERROR)

        # Restore selection
        if selected_id is not None:
            for item in self.tree.get_children():
                if self.tree.item(item)["values"][0] == selected_id:
                    self.tree.selection_set(item)
                    break

    def show_detail(self, text: str) -> None:
        """Show ticket details in the detail pane."""
        self.detail.delete("1.0", tk.END)
        self.detail.insert("1.0", text)


class AgentStatusPanel(tk.Frame):
    """Colored labels showing each agent's current status."""

    def __init__(self, parent: tk.Widget, **kwargs):
        super().__init__(parent, bg=theme.BG_PANEL, **kwargs)

        tk.Label(
            self, text="  AGENTS", font=theme.FONT_TITLE,
            fg=theme.FG_ACCENT, bg=theme.BG_PANEL, anchor="w",
        ).pack(fill=tk.X, padx=theme.PAD, pady=(theme.PAD, theme.PAD_SMALL))

        self._labels: dict[str, tk.Label] = {}
        self._status_labels: dict[str, tk.Label] = {}

        for name in ["codex", "claude", "gemini", "qwen"]:
            row = tk.Frame(self, bg=theme.BG_PANEL)
            row.pack(fill=tk.X, padx=theme.PAD, pady=1)

            color = theme.AGENT_COLORS.get(name, theme.FG_DIM)
            name_label = tk.Label(
                row, text=f"  {name}:", font=theme.FONT_MONO,
                fg=color, bg=theme.BG_PANEL, anchor="w", width=8,
            )
            name_label.pack(side=tk.LEFT)
            self._labels[name] = name_label

            status_label = tk.Label(
                row, text="IDLE", font=theme.FONT_MONO,
                fg=theme.FG_DIM, bg=theme.BG_PANEL, anchor="w",
            )
            status_label.pack(side=tk.LEFT, padx=theme.PAD_SMALL)
            self._status_labels[name] = status_label

    def set_status(self, agent: str, status: str, phase: str = "", elapsed: float = 0.0) -> None:
        """Update agent status display with optional phase and elapsed time."""
        label = self._status_labels.get(agent)
        if not label:
            return
        color_map = {
            "IDLE": theme.FG_DIM,
            "BUSY": theme.FG_WARNING,
            "COOLDOWN": theme.FG_ERROR,
            "ERROR": theme.FG_ERROR,
            "DONE": theme.FG_SUCCESS,
        }
        text = status.upper()
        if status.upper() == "BUSY" and elapsed > 0:
            secs = int(elapsed)
            if secs >= 60:
                text = f"BUSY {secs // 60}m{secs % 60:02d}s"
            else:
                text = f"BUSY {secs}s"
            if phase and phase != "idle":
                text += f" [{phase}]"
        label.configure(text=text, fg=color_map.get(status.upper(), theme.FG_DIM))

    def update_all(self, agents: list[dict[str, Any]], phases: Optional[dict[str, str]] = None,
                   elapsed: Optional[dict[str, float]] = None) -> None:
        """Update from list of agent dicts (from TicketStore.list_agents)."""
        phases = phases or {}
        elapsed = elapsed or {}
        for a in agents:
            name = a.get("name", "")
            status = a.get("status", "IDLE")
            self.set_status(name, status, phases.get(name, ""), elapsed.get(name, 0.0))


class StoryPanel(tk.Frame):
    """Read-only text display showing story_main.md content."""

    def __init__(self, parent: tk.Widget, **kwargs):
        super().__init__(parent, bg=theme.BG_PANEL, **kwargs)

        tk.Label(
            self, text="  STORY", font=theme.FONT_TITLE,
            fg=theme.FG_ACCENT, bg=theme.BG_PANEL, anchor="w",
        ).pack(fill=tk.X, padx=theme.PAD, pady=(theme.PAD, 0))

        text_frame = tk.Frame(self, bg=theme.BG_PANEL)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=theme.PAD, pady=theme.PAD_SMALL)

        self.text = tk.Text(
            text_frame, wrap=tk.WORD,
            font=theme.FONT_MONO_SMALL,
            bg=theme.BG_DARK, fg=theme.FG_TEXT,
            borderwidth=1, relief=tk.SOLID,
            highlightthickness=0,
            padx=6, pady=4,
        )
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.text.yview)
        self.text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        _make_copyable_readonly(self.text)

    def update_content(self, content: str) -> None:
        """Replace story content (only if changed to avoid flicker)."""
        current = self.text.get("1.0", "end-1c")
        if current == content:
            return
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", content)


class FlowMetricsPanel(tk.Frame):
    """Compact Kanban flow metrics summary."""

    def __init__(self, parent: tk.Widget, **kwargs):
        super().__init__(parent, bg=theme.BG_PANEL, **kwargs)
        tk.Label(
            self, text="  FLOW", font=theme.FONT_TITLE,
            fg=theme.FG_ACCENT, bg=theme.BG_PANEL, anchor="w",
        ).pack(fill=tk.X, padx=theme.PAD, pady=(theme.PAD, theme.PAD_SMALL))

        self._line = tk.Label(
            self,
            text="No goal selected",
            font=theme.FONT_MONO_SMALL,
            fg=theme.FG_TEXT,
            bg=theme.BG_DARK,
            anchor="w",
            justify=tk.LEFT,
            padx=6,
            pady=4,
            relief=tk.SOLID,
            borderwidth=1,
        )
        self._line.pack(fill=tk.X, padx=theme.PAD, pady=(0, theme.PAD_SMALL))

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        secs = int(max(0.0, seconds))
        if secs < 60:
            return f"{secs}s"
        mins, rem = divmod(secs, 60)
        if mins < 60:
            return f"{mins}m{rem:02d}s"
        hrs, m = divmod(mins, 60)
        return f"{hrs}h{m:02d}m"

    def update_metrics(self, metrics: Optional[dict[str, Any]]) -> None:
        if not metrics:
            text = "No goal selected"
        else:
            total = int(metrics.get("total", 0))
            done = int(metrics.get("done", 0))
            blocked = int(metrics.get("blocked", 0))
            wip = int(metrics.get("wip", 0))
            avg_lead = self._fmt_duration(float(metrics.get("avg_lead_seconds", 0.0)))
            avg_cycle = self._fmt_duration(float(metrics.get("avg_cycle_seconds", 0.0)))
            blocked_total = self._fmt_duration(float(metrics.get("blocked_seconds_total", 0.0)))
            pct = int((done / total) * 100) if total else 0
            text = (
                f"Done {done}/{total} ({pct}%)   "
                f"WIP {wip}   "
                f"Blocked {blocked}   "
                f"Lead(avg) {avg_lead}   "
                f"Cycle(avg) {avg_cycle}   "
                f"Blocked(total) {blocked_total}"
            )
        if self._line.cget("text") != text:
            self._line.configure(text=text)


class EventLogPanel(tk.Frame):
    """Read-only text display tailing recent events."""

    def __init__(self, parent: tk.Widget, **kwargs):
        super().__init__(parent, bg=theme.BG_PANEL, **kwargs)

        tk.Label(
            self, text="  EVENT LOG", font=theme.FONT_TITLE,
            fg=theme.FG_ACCENT, bg=theme.BG_PANEL, anchor="w",
        ).pack(fill=tk.X, padx=theme.PAD, pady=(theme.PAD, 0))

        text_frame = tk.Frame(self, bg=theme.BG_PANEL)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=theme.PAD, pady=theme.PAD_SMALL)

        self.text = tk.Text(
            text_frame, wrap=tk.WORD,
            font=theme.FONT_MONO_SMALL,
            bg=theme.BG_DARK, fg=theme.FG_TEXT,
            borderwidth=1, relief=tk.SOLID,
            highlightthickness=0,
            padx=6, pady=4,
        )
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.text.yview)
        self.text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        _make_copyable_readonly(self.text)

    def update_events(self, events: list[dict[str, Any]]) -> None:
        """Update with list of event dicts (newest first → show oldest first).
        Only redraws if content has changed to avoid flicker."""
        lines = []
        for ev in reversed(events):
            ts = ev.get("ts", "")[:19]
            etype = ev.get("type", "")
            agent = ev.get("agent", "")
            tid = ev.get("ticket_id", "")
            line = f"[{ts}] {etype}"
            if agent:
                line += f" ({agent})"
            if tid:
                line += f" #{tid}"
            payload = ev.get("payload", {})
            if payload:
                extras = " ".join(f"{k}={v}" for k, v in payload.items() if k not in ("goal_id",))
                if extras:
                    line += f" {extras}"
            lines.append(line)
        new_content = "\n".join(lines)
        current = self.text.get("1.0", "end-1c")
        if current == new_content:
            return
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", new_content + "\n" if new_content else "")
        self.text.see(tk.END)


class AgentOutputPanel(tk.Frame):
    """Terminal-like scrollable output panel for a single agent.

    Features: colored header, status indicator, Ctrl+F search bar,
    [Clear] button, selectable/copyable text.
    """

    def __init__(self, parent: tk.Widget, agent_name: str, **kwargs):
        super().__init__(parent, bg=theme.BG_PANEL, **kwargs)
        self.agent_name = agent_name
        color = theme.AGENT_COLORS.get(agent_name, theme.FG_ACCENT)
        self._search_visible = False

        # ── Header row ──
        header = tk.Frame(self, bg=theme.BG_PANEL)
        header.pack(fill=tk.X, padx=theme.PAD, pady=(theme.PAD_SMALL, 0))

        label_text = f" {agent_name.upper()}"
        tk.Label(
            header, text=label_text, font=theme.FONT_TITLE,
            fg=color, bg=theme.BG_PANEL, anchor="w",
        ).pack(side=tk.LEFT)

        self._status_label = tk.Label(
            header, text="", font=theme.FONT_STATUS,
            fg=theme.FG_DIM, bg=theme.BG_PANEL,
        )
        self._status_label.pack(side=tk.LEFT, padx=theme.PAD)

        # Clear button
        self._clear_btn = tk.Button(
            header, text="Clear", font=("monospace", 8),
            bg=theme.BG_BUTTON, fg=theme.FG_DIM,
            activebackground=theme.BG_BUTTON_HOVER, relief=tk.FLAT,
            padx=4, pady=0, command=self.clear,
        )
        self._clear_btn.pack(side=tk.RIGHT, padx=1)

        # Search button
        self._search_btn = tk.Button(
            header, text="Find", font=("monospace", 8),
            bg=theme.BG_BUTTON, fg=theme.FG_DIM,
            activebackground=theme.BG_BUTTON_HOVER, relief=tk.FLAT,
            padx=4, pady=0, command=self._toggle_search,
        )
        self._search_btn.pack(side=tk.RIGHT, padx=1)

        # ── Search bar (hidden by default) ──
        self._search_frame = tk.Frame(self, bg=theme.BG_INPUT)
        # Not packed until toggled

        self._search_entry = tk.Entry(
            self._search_frame, font=("monospace", 9),
            bg=theme.BG_INPUT, fg=theme.FG_TEXT,
            insertbackground=theme.FG_TEXT,
            borderwidth=0, highlightthickness=0,
        )
        self._search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4, pady=2)
        self._search_entry.bind("<Return>", lambda _: self._do_search())
        self._search_entry.bind("<Escape>", lambda _: self._toggle_search())

        self._search_count = tk.Label(
            self._search_frame, text="", font=("monospace", 8),
            fg=theme.FG_DIM, bg=theme.BG_INPUT,
        )
        self._search_count.pack(side=tk.RIGHT, padx=4)

        # ── Text area (the terminal) ──
        text_frame = tk.Frame(self, bg=theme.BG_PANEL)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=theme.PAD, pady=theme.PAD_SMALL)

        self.text = tk.Text(
            text_frame, wrap=tk.WORD,
            font=theme.FONT_MONO_SMALL,
            bg="#0e0e1a", fg=theme.FG_TEXT,
            borderwidth=0, relief=tk.FLAT,
            highlightthickness=1, highlightbackground=theme.BORDER_COLOR,
            padx=6, pady=4,
        )
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.text.yview)
        self.text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        _make_copyable_readonly(self.text)

        # Search highlight tag
        self.text.tag_configure("search_match", background="#665c00", foreground=theme.FG_WARNING)

        # Ctrl+F binding
        self.text.bind("<Control-f>", lambda _: self._toggle_search())

    def _toggle_search(self) -> None:
        """Show/hide the search bar."""
        if self._search_visible:
            self._search_frame.pack_forget()
            self.text.tag_remove("search_match", "1.0", tk.END)
            self._search_visible = False
        else:
            self._search_frame.pack(fill=tk.X, padx=theme.PAD, before=self.text.master)
            self._search_entry.focus_set()
            self._search_visible = True

    def _do_search(self) -> None:
        """Search for text and highlight all matches."""
        query = self._search_entry.get().strip()
        self.text.tag_remove("search_match", "1.0", tk.END)
        if not query:
            self._search_count.configure(text="")
            return
        count = 0
        start = "1.0"
        while True:
            pos = self.text.search(query, start, stopindex=tk.END, nocase=True)
            if not pos:
                break
            end = f"{pos}+{len(query)}c"
            self.text.tag_add("search_match", pos, end)
            start = end
            count += 1
        self._search_count.configure(text=f"{count} found" if count else "0 found")
        # Scroll to first match
        first = self.text.tag_nextrange("search_match", "1.0")
        if first:
            self.text.see(first[0])

    def append_text(self, text: str) -> None:
        self.text.insert(tk.END, text)
        self.text.see(tk.END)

    def set_text(self, text: str) -> None:
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", text)
        self.text.see(tk.END)

    def clear(self) -> None:
        self.text.delete("1.0", tk.END)

    def set_status(self, status: str) -> None:
        color_map = {
            "IDLE": theme.FG_DIM,
            "BUSY": theme.FG_WARNING,
            "DONE": theme.FG_SUCCESS,
            "ERROR": theme.FG_ERROR,
        }
        self._status_label.configure(
            text=status.upper(),
            fg=color_map.get(status.upper(), theme.FG_DIM),
        )


class ProjectSelector(tk.Frame):
    """Dropdown for switching between goals/projects."""

    def __init__(
        self,
        parent: tk.Widget,
        on_select: Optional[Callable[[int], None]] = None,
        on_new: Optional[Callable[[], None]] = None,
        **kwargs,
    ):
        super().__init__(parent, bg=theme.BG_STATUS, **kwargs)
        self._on_select = on_select
        self._on_new = on_new
        self._goals: list[dict[str, Any]] = []  # [{id, text, status}]

        tk.Label(
            self, text="Project:", font=theme.FONT_MONO_SMALL,
            fg=theme.FG_DIM, bg=theme.BG_STATUS,
        ).pack(side=tk.LEFT, padx=(theme.PAD, 2))

        self._combo_var = tk.StringVar()
        self._combo = ttk.Combobox(
            self, textvariable=self._combo_var,
            state="readonly", width=40,
            font=("monospace", 9),
        )
        self._combo.pack(side=tk.LEFT, padx=2)
        self._combo.bind("<<ComboboxSelected>>", self._on_combo_select)

        # Style the combobox for dark theme
        style = ttk.Style()
        style.configure("Dark.TCombobox",
            fieldbackground=theme.BG_INPUT,
            background=theme.BG_BUTTON,
            foreground=theme.FG_TEXT,
        )
        self._combo.configure(style="Dark.TCombobox")

        tk.Button(
            self, text="+New", font=("monospace", 8),
            bg=theme.BG_BUTTON, fg=theme.FG_SUCCESS,
            activebackground=theme.BG_BUTTON_HOVER, relief=tk.FLAT,
            padx=6, pady=0, command=self._handle_new,
        ).pack(side=tk.LEFT, padx=2)

    def update_goals(self, goals: list[dict[str, Any]]) -> None:
        """Refresh the goal list. Each dict: {id, text, status}."""
        self._goals = goals
        display = []
        for g in goals:
            status_tag = ""
            if g.get("status") == "COMPLETED":
                status_tag = " [done]"
            elif g.get("status") == "CANCELLED":
                status_tag = " [cancelled]"
            text = g.get("text", "")[:60]
            display.append(f"#{g['id']}: {text}{status_tag}")
        self._combo["values"] = display

    def select_goal(self, goal_id: int) -> None:
        """Programmatically select a goal by ID."""
        for i, g in enumerate(self._goals):
            if g["id"] == goal_id:
                self._combo.current(i)
                return

    def _on_combo_select(self, event=None) -> None:
        idx = self._combo.current()
        if 0 <= idx < len(self._goals) and self._on_select:
            self._on_select(self._goals[idx]["id"])

    def _handle_new(self) -> None:
        if self._on_new:
            self._on_new()


class SafetyApprovalBar(tk.Frame):
    """Warning bar shown when a destructive action needs approval."""

    def __init__(
        self,
        parent: tk.Widget,
        on_approve: Optional[Callable[[int], None]] = None,
        on_deny: Optional[Callable[[int], None]] = None,
        **kwargs,
    ):
        super().__init__(parent, bg=theme.FG_WARNING, height=40, **kwargs)
        self.pack_propagate(False)
        self._on_approve = on_approve
        self._on_deny = on_deny
        self._current_alert_id: Optional[int] = None

        inner = tk.Frame(self, bg=theme.FG_WARNING)
        inner.pack(fill=tk.BOTH, expand=True, padx=theme.PAD, pady=theme.PAD_SMALL)

        self._icon_label = tk.Label(
            inner, text="  \u26a0 ", font=theme.FONT_TITLE,
            fg=theme.BG_DARK, bg=theme.FG_WARNING,
        )
        self._icon_label.pack(side=tk.LEFT)

        self._message_label = tk.Label(
            inner, text="", font=theme.FONT_MONO,
            fg=theme.BG_DARK, bg=theme.FG_WARNING, anchor="w",
        )
        self._message_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._deny_btn = tk.Button(
            inner, text=" Deny ", font=theme.FONT_MONO,
            bg=theme.FG_ERROR, fg=theme.BG_DARK,
            activebackground="#ff6666", relief=tk.FLAT,
            padx=10, pady=2, command=self._handle_deny,
        )
        self._deny_btn.pack(side=tk.RIGHT, padx=theme.PAD_SMALL)

        self._approve_btn = tk.Button(
            inner, text=" Approve ", font=theme.FONT_MONO,
            bg=theme.FG_SUCCESS, fg=theme.BG_DARK,
            activebackground="#88dd88", relief=tk.FLAT,
            padx=10, pady=2, command=self._handle_approve,
        )
        self._approve_btn.pack(side=tk.RIGHT, padx=theme.PAD_SMALL)

        # Hidden by default
        self.pack_forget()

    def show_alert(self, alert_id: int, agent: str, matched_text: str) -> None:
        """Show the safety bar with an alert."""
        self._current_alert_id = alert_id
        self._message_label.configure(
            text=f"Agent {agent} wants to run: {matched_text}"
        )
        self.pack(fill=tk.X, side=tk.BOTTOM, before=self.master.winfo_children()[-1]
                  if self.master.winfo_children() else None)

    def hide(self) -> None:
        """Hide the safety bar."""
        self._current_alert_id = None
        self.pack_forget()

    def _handle_approve(self) -> None:
        if self._current_alert_id is not None and self._on_approve:
            self._on_approve(self._current_alert_id)
        self.hide()

    def _handle_deny(self) -> None:
        if self._current_alert_id is not None and self._on_deny:
            self._on_deny(self._current_alert_id)
        self.hide()
