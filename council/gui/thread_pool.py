"""AgentThread + AgentThreadPool — background threading for agent calls."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class AgentTask:
    """A task to execute on an agent thread."""
    agent_name: str
    prompt: str
    phase: str = "unknown"
    on_chunk: Optional[Callable[[str], None]] = None
    ticket_ids: list[int] = field(default_factory=list)
    status: str = "pending"


@dataclass
class AgentResult:
    """Result from an agent execution."""
    agent_name: str
    output: str
    error: Optional[str] = None
    phase: str = "unknown"
    ticket_ids: list[int] = field(default_factory=list)
    status: str = "done"


class AgentThread:
    """Persistent daemon thread for a single agent."""

    def __init__(self, agent_name: str, agent, result_queue: queue.Queue):
        self.agent_name = agent_name
        self.agent = agent
        self.result_queue = result_queue
        self.task_queue: queue.Queue[Optional[AgentTask]] = queue.Queue()
        self._thread = threading.Thread(
            target=self._run,
            name=f"agent-{agent_name}",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        """Main loop — wait for tasks, execute, post results."""
        while True:
            task = self.task_queue.get()
            if task is None:
                break  # Shutdown signal

            try:
                # Wire up chunk callback for streaming agents (Qwen)
                if task.on_chunk and hasattr(self.agent, 'on_chunk'):
                    original_callback = self.agent.on_chunk
                    self.agent.on_chunk = task.on_chunk

                output = self.agent.call(task.prompt, phase=task.phase)

                # Handle both old-style (str) and new-style (CallResult) returns
                if hasattr(output, 'output'):
                    # CallResult from new agents
                    out_text = output.output
                    error = output.error if not output.ok else None
                    status = output.status.value if hasattr(output.status, 'value') else str(output.status)
                else:
                    out_text = str(output)
                    error = None
                    status = "done"

                # Restore original callback
                if task.on_chunk and hasattr(self.agent, 'on_chunk'):
                    self.agent.on_chunk = original_callback

                self.result_queue.put(AgentResult(
                    agent_name=self.agent_name,
                    output=out_text,
                    error=error,
                    phase=task.phase,
                    ticket_ids=task.ticket_ids,
                    status=status,
                ))
            except Exception as e:
                self.result_queue.put(AgentResult(
                    agent_name=self.agent_name,
                    output="",
                    error=str(e),
                    phase=task.phase,
                    ticket_ids=task.ticket_ids,
                    status="error",
                ))

    def submit(
        self,
        prompt: str,
        phase: str = "unknown",
        on_chunk: Optional[Callable] = None,
        ticket_ids: Optional[list[int]] = None,
    ) -> None:
        """Submit a task to this agent thread."""
        self.task_queue.put(AgentTask(
            agent_name=self.agent_name,
            prompt=prompt,
            phase=phase,
            on_chunk=on_chunk,
            ticket_ids=ticket_ids or [],
        ))

    def shutdown(self) -> None:
        """Signal the thread to stop."""
        self.task_queue.put(None)


class AgentThreadPool:
    """Manages persistent threads for all agents."""

    def __init__(self):
        self.result_queue: queue.Queue[AgentResult] = queue.Queue()
        self._threads: dict[str, AgentThread] = {}

    def register(self, agent_name: str, agent) -> None:
        """Register an agent and start its background thread."""
        thread = AgentThread(agent_name, agent, self.result_queue)
        self._threads[agent_name] = thread

    def submit(
        self,
        agent_name: str,
        prompt: str,
        phase: str = "unknown",
        on_chunk: Optional[Callable] = None,
        ticket_ids: Optional[list[int]] = None,
    ) -> None:
        """Submit a task to a specific agent."""
        thread = self._threads.get(agent_name)
        if not thread:
            raise ValueError(f"No agent thread registered for: {agent_name}")
        thread.submit(prompt, phase, on_chunk, ticket_ids)

    def submit_all(self, tasks: dict[str, str], phase: str = "unknown") -> None:
        """Submit tasks to multiple agents in parallel.

        tasks: {agent_name: prompt}
        """
        for agent_name, prompt in tasks.items():
            self.submit(agent_name, prompt, phase)

    def get_result(self, timeout: float = 0.05) -> Optional[AgentResult]:
        """Non-blocking check for a result."""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None

    def shutdown_all(self) -> None:
        """Shutdown all agent threads."""
        for thread in self._threads.values():
            thread.shutdown()

    @property
    def agent_names(self) -> list[str]:
        return list(self._threads.keys())
