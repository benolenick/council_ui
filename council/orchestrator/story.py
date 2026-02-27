"""Dual story system: rewritable main story + append-only event log."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .events import EventStore, EventType


class StoryManager:
    """Manages the dual story: story_main.md (rewritable) + event log (append-only).

    The main story is a Markdown file that Qwen rewrites after each synthesis cycle.
    The event log is handled by EventStore (NDJSON + SQLite).
    """

    def __init__(self, data_dir: str | Path, event_store: EventStore):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.main_path = self.data_dir / "story_main.md"
        self.events = event_store

    def init(self) -> None:
        """Initialize story files if they don't exist."""
        if not self.main_path.exists():
            self.main_path.write_text(
                "# Council Story\n\n"
                "_No activity yet. Submit a goal to begin._\n"
            )

    def read_main(self) -> str:
        """Read the current main story content."""
        if self.main_path.exists():
            return self.main_path.read_text()
        return ""

    def rewrite_main(self, content: str) -> None:
        """Replace the main story with new content from Qwen synthesis."""
        self.main_path.write_text(content)
        self.events.emit(
            EventType.STORY_UPDATED,
            payload={"length": len(content)},
        )

    def get_context_for_prompt(self, max_chars: int = 6000) -> str:
        """Get story context suitable for injection into agent prompts.

        Returns the main story truncated to max_chars, plus a summary of recent events.
        """
        main = self.read_main()
        if len(main) > max_chars:
            main = main[:max_chars] + "\n\n[... truncated ...]"

        # Add recent event summary
        recent = self.get_recent_events(10)
        if recent:
            lines = ["\n## Recent Events"]
            for ev in recent:
                ts = ev.get("ts", "")[:19]
                etype = ev.get("type", "")
                agent = ev.get("agent", "")
                tid = ev.get("ticket_id", "")
                summary = f"[{ts}] {etype}"
                if agent:
                    summary += f" ({agent})"
                if tid:
                    summary += f" ticket #{tid}"
                lines.append(f"- {summary}")
            event_section = "\n".join(lines)
            available = max_chars - len(main)
            if available > 200:
                main += event_section[:available]

        return main

    def get_recent_events(self, n: int = 50) -> list[dict[str, Any]]:
        """Get the N most recent events from the log."""
        return self.events.tail(n)

    def append_event(
        self,
        event_type: EventType | str,
        ticket_id: Optional[int] = None,
        agent: Optional[str] = None,
        payload: Optional[dict[str, Any]] = None,
    ) -> int:
        """Convenience: emit an event through the event store."""
        return self.events.emit(event_type, ticket_id, agent, payload)
