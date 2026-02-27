"""Agent wrappers with CallResult, AgentStatus, MockAgent, and rate-limit detection.

Evolved from council/agents.py — keeps ClaudeAgent, CodexAgent, GeminiAgent, QwenAgent
subprocess patterns. Adds structured CallResult, rate-limit scanning, and MockAgent for testing.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class AgentStatus(str, Enum):
    OK = "OK"
    RATE_LIMITED = "RATE_LIMITED"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"


class AgentError(Exception):
    pass


@dataclass
class CallResult:
    """Structured result from an agent call."""
    agent: str
    output: str
    status: AgentStatus = AgentStatus.OK
    error: Optional[str] = None
    duration: float = 0.0
    raw_stderr: str = ""
    rate_limit_pattern: str = ""
    rate_limit_source: str = ""

    @property
    def ok(self) -> bool:
        return self.status == AgentStatus.OK

    @property
    def rate_limited(self) -> bool:
        return self.status == AgentStatus.RATE_LIMITED


# Strong patterns: only actual upstream API rate limiting.
# Ordered by confidence — highest first.
RATE_LIMIT_PATTERNS: list[tuple[re.Pattern[str], str, bool]] = [
    # (pattern, label, strong?)  — strong=True means safe even on exit 0
    (re.compile(r"status[\"': ]*429|HTTP[/ ]429|\b429\b.*Too Many", re.IGNORECASE), "http_429", True),
    (re.compile(r"too many requests", re.IGNORECASE), "too_many_requests", True),
    (re.compile(r"rate[ _-]?limit(?:ed|ing)?", re.IGNORECASE), "rate_limit", False),
    (re.compile(r"quota exceeded", re.IGNORECASE), "quota_exceeded", True),
    (re.compile(r"resource[_ ]exhausted", re.IGNORECASE), "resource_exhausted", True),
    # "retry_after" header from APIs (NOT pip's "Retrying (Retry(total=..." output)
    (re.compile(r"retry-after\s*:", re.IGNORECASE), "retry_after_header", True),
]

# False-positive exclusions: if the match line also contains these, skip it
_RATE_LIMIT_FP_PATTERNS = re.compile(
    r"Retrying \(Retry\(total=|pip[._]vendor|NewConnectionError|urllib3", re.IGNORECASE
)


def _find_rate_limit_pattern(text: str, require_strong: bool = False) -> str:
    """Return the matched rate-limit pattern label, else empty string.

    If require_strong is True, only return patterns marked as strong
    (safe to trust even on exit code 0).
    """
    for pattern, label, strong in RATE_LIMIT_PATTERNS:
        if require_strong and not strong:
            continue
        match = pattern.search(text)
        if match:
            # Check for false-positive context around the match
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            context = text[start:end]
            if _RATE_LIMIT_FP_PATTERNS.search(context):
                continue
            return label
    return ""


class AgentWrapper(ABC):
    """Base class for agent CLI wrappers."""

    name: str = "base"

    def __init__(
        self,
        command: Optional[str] = None,
        timeout: int = 120,
        cwd: Optional[str] = None,
    ):
        self.command = command or self.name
        self.timeout = timeout
        self.cwd = cwd
        # Callback for streaming partial output (set by router)
        self.on_output: Optional[Callable[[str, str], None]] = None  # (agent_name, partial_text)

    @abstractmethod
    def _build_command(self, prompt: str) -> list[str]:
        ...

    def _build_env(self) -> dict[str, str]:
        """Build subprocess environment."""
        return os.environ.copy()

    def call(self, prompt: str, phase: str = "unknown") -> CallResult:
        """Call the agent and return a CallResult. Streams output via on_output callback."""
        cmd = self._build_command(prompt)
        env = self._build_env()
        t0 = time.monotonic()

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.cwd,
                env=env,
            )
        except FileNotFoundError:
            return CallResult(
                agent=self.name,
                output="",
                status=AgentStatus.ERROR,
                error=f"Command not found: {cmd[0]}",
                duration=time.monotonic() - t0,
            )

        # Stream stdout line by line
        stdout_lines: list[str] = []
        try:
            deadline = t0 + self.timeout
            for line in proc.stdout:
                stdout_lines.append(line)
                if self.on_output:
                    self.on_output(self.name, "".join(stdout_lines))
                if time.monotonic() > deadline:
                    proc.kill()
                    proc.wait()
                    return CallResult(
                        agent=self.name,
                        output="".join(stdout_lines),
                        status=AgentStatus.TIMEOUT,
                        error=f"Timed out after {self.timeout}s",
                        duration=time.monotonic() - t0,
                    )
            proc.wait(timeout=max(1, self.timeout - (time.monotonic() - t0)))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return CallResult(
                agent=self.name,
                output="".join(stdout_lines),
                status=AgentStatus.TIMEOUT,
                error=f"Timed out after {self.timeout}s",
                duration=time.monotonic() - t0,
            )

        duration = time.monotonic() - t0
        stdout = "".join(stdout_lines)
        stderr = proc.stderr.read() if proc.stderr else ""

        # Rate-limit classification.
        # Non-zero exit: any pattern match counts.
        # Zero exit: only strong patterns (actual HTTP 429, quota exceeded) count,
        # to avoid false positives from pip retry output, agent banners, etc.
        rate_limit_match = ""
        rate_limit_source = ""

        if proc.returncode != 0:
            stderr_match = _find_rate_limit_pattern(stderr)
            stdout_match = _find_rate_limit_pattern(stdout)
            if stderr_match:
                rate_limit_match = stderr_match
                rate_limit_source = "stderr"
            elif stdout_match:
                rate_limit_match = stdout_match
                rate_limit_source = "stdout"
        else:
            # Exit 0: only trust strong patterns in stderr
            stderr_match = _find_rate_limit_pattern(stderr, require_strong=True)
            if stderr_match:
                rate_limit_match = stderr_match
                rate_limit_source = "stderr"

        if rate_limit_match:
            return CallResult(
                agent=self.name,
                output=stdout,
                status=AgentStatus.RATE_LIMITED,
                error=f"Rate limit detected ({rate_limit_source}:{rate_limit_match})",
                duration=duration,
                raw_stderr=stderr,
                rate_limit_pattern=rate_limit_match,
                rate_limit_source=rate_limit_source,
            )

        if proc.returncode != 0 and not stdout.strip():
            return CallResult(
                agent=self.name,
                output="",
                status=AgentStatus.ERROR,
                error=f"Exit code {proc.returncode}: {stderr[:500]}",
                duration=duration,
                raw_stderr=stderr,
            )

        return CallResult(
            agent=self.name,
            output=stdout,
            status=AgentStatus.OK,
            duration=duration,
            raw_stderr=stderr,
        )

    def is_available(self) -> bool:
        """Check if the agent binary is available on PATH."""
        return shutil.which(self.command) is not None


class ClaudeAgent(AgentWrapper):
    """Wrapper for Claude Code CLI."""

    name = "claude"

    def __init__(self, command: Optional[str] = None, timeout: int = 300, cwd: Optional[str] = None):
        super().__init__(command or "claude", timeout, cwd)

    def _build_command(self, prompt: str) -> list[str]:
        return [self.command, "-p", prompt, "--output-format", "json"]

    def _build_env(self) -> dict[str, str]:
        env = super()._build_env()
        # Avoid nested Claude Code session detection when orchestrated from another session.
        env.pop("CLAUDECODE", None)
        return env

    def call(self, prompt: str, phase: str = "unknown") -> CallResult:
        result = super().call(prompt, phase)
        if not result.ok:
            return result

        # Claude --output-format json wraps result in JSON
        try:
            obj = json.loads(result.output)
            if isinstance(obj, dict) and "result" in obj:
                result.output = obj["result"]
            elif isinstance(obj, list):
                for item in reversed(obj):
                    if isinstance(item, dict):
                        if item.get("type") == "result" and "result" in item:
                            result.output = item["result"]
                            break
                        if "content" in item:
                            contents = item["content"]
                            if isinstance(contents, list):
                                for block in reversed(contents):
                                    if isinstance(block, dict) and block.get("type") == "text":
                                        result.output = block["text"]
                                        break
                                break
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        return result


class CodexAgent(AgentWrapper):
    """Wrapper for Codex CLI."""

    name = "codex"

    def __init__(self, command: Optional[str] = None, timeout: int = 120, cwd: Optional[str] = None):
        super().__init__(command or "codex", timeout, cwd)

    def _build_command(self, prompt: str) -> list[str]:
        return [self.command, "exec", "--skip-git-repo-check", prompt]


class GeminiAgent(AgentWrapper):
    """Wrapper for Gemini CLI."""

    name = "gemini"

    def __init__(self, command: Optional[str] = None, timeout: int = 120, cwd: Optional[str] = None):
        super().__init__(command or "gemini", timeout, cwd)

    def _build_command(self, prompt: str) -> list[str]:
        return [self.command, "-p", prompt]


class QwenAgent(AgentWrapper):
    """Wrapper for Qwen via Ollama HTTP API with CLI fallback.

    When harness_enabled=True and phase is 'review' or 'synthesis',
    routes through FV's evidence-grounded harness pipeline for
    fact-checked, cited responses.
    """

    name = "qwen"

    def __init__(
        self,
        model: str = "qwen3:32b-q8_0",
        ollama_url: str = "http://localhost:11434",
        timeout: int = 120,
        cwd: Optional[str] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
        harness_enabled: bool = False,
    ):
        super().__init__(command="ollama", timeout=timeout, cwd=cwd)
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")
        self.on_chunk = on_chunk
        self.harness_enabled = harness_enabled

    def _build_command(self, prompt: str) -> list[str]:
        return ["ollama", "run", self.model, prompt]

    def call(self, prompt: str, phase: str = "unknown") -> CallResult:
        t0 = time.monotonic()

        # Route through FV harness for review/synthesis phases when enabled
        if self.harness_enabled and phase in ("review", "synthesis"):
            try:
                output = self._call_harness(prompt, phase)
                return CallResult(
                    agent=self.name,
                    output=output,
                    status=AgentStatus.OK,
                    duration=time.monotonic() - t0,
                )
            except Exception:
                pass  # fall through to normal path

        try:
            output = self._call_http(prompt)
            return CallResult(
                agent=self.name,
                output=output,
                status=AgentStatus.OK,
                duration=time.monotonic() - t0,
            )
        except Exception as e:
            # Fall back to CLI
            try:
                result = super().call(prompt, phase)
                return result
            except Exception as e2:
                return CallResult(
                    agent=self.name,
                    output="",
                    status=AgentStatus.ERROR,
                    error=str(e2),
                    duration=time.monotonic() - t0,
                )

    def _call_harness(self, prompt: str, phase: str) -> str:
        """Route through FV's evidence-grounded harness pipeline."""
        from council.fv.agent_pipeline.harness.harness import harnessed_answer
        result = harnessed_answer(prompt)
        if isinstance(result, dict):
            return result.get("answer", result.get("raw", str(result)))
        return str(result)

    def _call_http(self, prompt: str) -> str:
        url = f"{self.ollama_url}/api/generate"
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": True,
        }).encode()

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        chunks: list[str] = []
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            for line in resp:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    token = obj.get("response", "")
                    if token:
                        chunks.append(token)
                        if self.on_chunk:
                            self.on_chunk(token)
                    if obj.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue

        result = "".join(chunks)
        # Strip <think>...</think> tags
        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()
        return result

    def warmup(self) -> bool:
        try:
            url = f"{self.ollama_url}/api/generate"
            payload = json.dumps({
                "model": self.model,
                "prompt": "hi",
                "stream": False,
            }).encode()
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp.read()
            return True
        except Exception:
            return False


# ── Mock Agent for testing ──

# Simulated EVENT lines that mock agents emit
_MOCK_EVENTS = {
    "propose": [
        'EVENT: PROPOSE_TICKETS count=3',
    ],
    "execute": [
        'EVENT: CLAIM_TICKET id={ticket_id}',
        'EVENT: DONE id={ticket_id} summary="Mock implementation complete"',
    ],
    "review": [
        'EVENT: NEEDS_REVIEW id={ticket_id}',
    ],
}


class MockAgent(AgentWrapper):
    """Simulated agent for testing the GUI and orchestrator without real binaries."""

    name = "mock"

    def __init__(self, mock_name: str = "mock", delay_range: tuple[float, float] = (0.5, 2.0)):
        super().__init__(command="mock", timeout=999)
        self.name = mock_name
        self.delay_range = delay_range

    def _build_command(self, prompt: str) -> list[str]:
        return ["echo", "mock"]

    def _extract_ticket_id(self, prompt: str) -> int:
        """Extract ticket ID from prompt (e.g. 'TICKET #31:')."""
        m = re.search(r"TICKET\s+#(\d+)", prompt)
        return int(m.group(1)) if m else 1

    def call(self, prompt: str, phase: str = "unknown") -> CallResult:
        delay = random.uniform(*self.delay_range)
        time.sleep(delay)

        # Determine phase type from the prompt's first few lines (the instruction).
        # We only check the first ~200 chars to avoid false matches on embedded
        # story context or event history that may contain keywords like "assigned".
        prompt_lower = prompt.lower()
        prompt_head = prompt_lower[:200]
        tid = self._extract_ticket_id(prompt)
        if "dedupe" in prompt_head or "merging ticket proposals" in prompt_head:
            output = self._mock_dedupe()
            events = []
        elif "assign" in prompt_head and "ticket" in prompt_head:
            output = self._mock_assignment()
            events = []
        elif "synthe" in prompt_head or "storyteller" in prompt_head or "rewrite the" in prompt_head:
            output = self._mock_synthesis()
            events = []
        elif "review" in prompt_head and "verdict" in prompt_head:
            output = self._mock_review()
            events = [e.format(ticket_id=tid) for e in _MOCK_EVENTS.get("review", [])]
        elif "propose" in prompt_head or "break down" in prompt_head:
            output = self._mock_proposal()
            events = _MOCK_EVENTS.get("propose", [])
        elif "execute" in prompt_head or "implement" in prompt_head:
            output = self._mock_execution()
            events = [e.format(ticket_id=tid) for e in _MOCK_EVENTS.get("execute", [])]
        else:
            output = f"[{self.name}] Mock response to: {prompt[:100]}..."
            events = []

        event_lines = "\n".join(events)
        if event_lines:
            output = output + "\n" + event_lines

        return CallResult(
            agent=self.name,
            output=output,
            status=AgentStatus.OK,
            duration=delay,
        )

    def _mock_proposal(self) -> str:
        tickets = [
            {"title": "Set up project structure", "description": "Create directories and config files", "acceptance": "All dirs exist", "priority": "SMALL"},
            {"title": "Implement core logic", "description": "Build the main processing pipeline", "acceptance": "Tests pass", "priority": "BIG"},
            {"title": "Add error handling", "description": "Handle edge cases and failures", "acceptance": "No unhandled exceptions", "priority": "SMALL"},
        ]
        return json.dumps({"tickets": tickets}, indent=2)

    def _mock_execution(self) -> str:
        return (
            f"[{self.name}] Executing ticket...\n"
            f"Created mock_output.py\n"
            f"All checks pass.\n"
        )

    def _mock_review(self) -> str:
        return json.dumps({
            "verdict": "approve",
            "notes": "Implementation looks correct. Tests pass.",
        }, indent=2)

    def _mock_synthesis(self) -> str:
        return (
            "# Council Story\n\n"
            "## Current Sprint\n\n"
            "The team is making progress on the goal. Key tickets are in progress.\n\n"
            "### Completed\n- Project structure set up\n\n"
            "### In Progress\n- Core logic implementation\n\n"
            "### Remaining\n- Error handling\n"
        )

    def _mock_dedupe(self) -> str:
        return json.dumps({
            "tickets": [
                {"title": "Set up project structure", "description": "Create directories and config", "acceptance": "Dirs exist", "priority": "SMALL"},
                {"title": "Implement core logic", "description": "Build main pipeline", "acceptance": "Tests pass", "priority": "BIG"},
                {"title": "Add error handling", "description": "Handle edge cases", "acceptance": "No crashes", "priority": "SMALL"},
            ]
        }, indent=2)

    def _mock_assignment(self) -> str:
        return json.dumps({
            "assignments": [
                {"ticket_id": 1, "agent": "codex"},
                {"ticket_id": 2, "agent": "claude"},
                {"ticket_id": 3, "agent": "gemini"},
            ]
        }, indent=2)

    def is_available(self) -> bool:
        return True


def create_agent(
    name: str,
    config: dict[str, Any],
    cwd: Optional[str] = None,
    mock: bool = False,
) -> AgentWrapper:
    """Factory: create an agent wrapper from config.

    Args:
        name: Agent name (codex/claude/gemini/qwen)
        config: Agent-specific config dict
        cwd: Working directory for subprocess calls
        mock: If True, return a MockAgent instead of real agent
    """
    if mock:
        return MockAgent(mock_name=name)

    command = config.get("command", name)
    timeout = config.get("timeout", 120)

    if name == "qwen":
        return QwenAgent(
            model=config.get("model", "qwen3:32b-q8_0"),
            ollama_url=config.get("ollama_url", "http://localhost:11434"),
            timeout=timeout,
            cwd=cwd,
            harness_enabled=config.get("harness_enabled", False),
        )

    agent_classes: dict[str, type[AgentWrapper]] = {
        "claude": ClaudeAgent,
        "codex": CodexAgent,
        "gemini": GeminiAgent,
    }

    cls = agent_classes.get(name)
    if cls is None:
        raise AgentError(f"Unknown agent type: {name}")

    return cls(command=command, timeout=timeout, cwd=cwd)
