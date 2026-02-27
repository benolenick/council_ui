"""Configuration management: load/save .council/config.json, project profiles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


DEFAULT_CONFIG: dict[str, Any] = {
    "version": 2,
    "workspace": ".",
    "agents": {
        "codex": {
            "command": "codex",
            "timeout": 600,
        },
        "claude": {
            "command": "claude",
            "timeout": 600,
        },
        "gemini": {
            "command": "gemini",
            "timeout": 600,
        },
        "qwen": {
            "model": "qwen3:32b-q8_0",
            "ollama_url": "http://localhost:11434",
            "timeout": 120,
        },
    },
    "orchestrator": {
        "heartbeat_interval_s": 30,
        "max_big_concurrent": 1,
        "max_small_per_agent": 2,
        "max_big_per_agent": 1,
        "proposal_timeout_s": 120,
        "execution_timeout_s": 600,
        "stalled_ticket_timeout_s": 300,
    },
    "gui": {
        "theme": "dark",
        "poll_interval_ms": 50,
        "window_width": 1400,
        "window_height": 800,
    },
    "safety": {
        "enabled": True,
        "auto_deny_patterns": [],
        "auto_approve_patterns": [],
    },
    "fv": {
        "enabled": True,
        "harness_mode": False,
        "memory_enabled": True,
        "bridge_port": 8000,
        "ollama_url": "http://127.0.0.1:11434",
        "ollama_planner_url": "http://127.0.0.1:11435",
        "model": "qwen3:14b",
        "planner_model": "gpt-oss:pinned",
        "ctt_index_dir": "",
        "gemini_api_key": "",
    },
    "mock_mode": False,
    "policy": {
        "denylist_paths": [],
        "allowlist_commands": [],
    },
    "test_commands": [],
}


class Config:
    """Manages .council/config.json and provides access to settings."""

    def __init__(self, base_dir: str | Path = ".council"):
        self.base_dir = Path(base_dir)
        self.config_path = self.base_dir / "config.json"
        self._data: dict[str, Any] = {}
        self.load()

    def load(self) -> dict[str, Any]:
        """Load config from disk, merging with defaults."""
        self._data = dict(DEFAULT_CONFIG)
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    saved = json.load(f)
                self._deep_merge(self._data, saved)
            except (json.JSONDecodeError, OSError):
                pass
        return self._data

    def save(self) -> None:
        """Persist current config to disk."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(self._data, f, indent=2)
            f.write("\n")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value by dotted key path (e.g. 'agents.builder.timeout')."""
        keys = key.split(".")
        val = self._data
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def set(self, key: str, value: Any) -> None:
        """Set a config value by dotted key path."""
        keys = key.split(".")
        d = self._data
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value

    @property
    def workspace(self) -> str:
        return self.get("workspace", ".")

    @workspace.setter
    def workspace(self, value: str) -> None:
        self.set("workspace", value)

    @property
    def max_review_rounds(self) -> int:
        return self.get("max_review_rounds", 2)

    @property
    def test_commands(self) -> list[str]:
        return self.get("test_commands", [])

    def agent_config(self, role: str) -> dict[str, Any]:
        """Get agent config for a role (builder/architect/skeptic)."""
        return self.get(f"agents.{role}", {})

    def init_workspace(self, workspace: Optional[str] = None) -> Path:
        """Initialize .council/ structure in the given or current workspace."""
        if workspace:
            self.workspace = str(Path(workspace).resolve())
            self.base_dir = Path(workspace).resolve() / ".council"
            self.config_path = self.base_dir / "config.json"

        self.base_dir.mkdir(parents=True, exist_ok=True)
        (self.base_dir / "runs").mkdir(exist_ok=True)
        (self.base_dir / "data").mkdir(exist_ok=True)
        (self.base_dir / "data" / "handoffs").mkdir(exist_ok=True)
        self.save()
        return self.base_dir

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """Recursively merge override into base."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                Config._deep_merge(base[key], value)
            else:
                base[key] = value
        return base
