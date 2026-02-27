#!/usr/bin/env python3
"""Council CLI v0.3 — ticket-based AI agent orchestration."""

from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

from council import __version__
from council.config import Config


def _setup_signal_handlers() -> None:
    """Install clean Ctrl+C handling."""
    def handler(sig, frame):
        print("\n[*] Interrupted — exiting cleanly.", file=sys.stderr)
        sys.exit(130)
    signal.signal(signal.SIGINT, handler)


def cmd_init(args: argparse.Namespace) -> int:
    """Initialize .council/ structure in a workspace."""
    import subprocess as _sp

    workspace = args.workspace or os.getcwd()
    workspace = str(Path(workspace).resolve())

    Path(workspace).mkdir(parents=True, exist_ok=True)

    # Initialize git repo if not already one
    git_dir = Path(workspace) / ".git"
    if not git_dir.exists():
        _sp.run(["git", "init", workspace], capture_output=True)
        print(f"  Git repo initialized in {workspace}")

    config = Config(base_dir=Path(workspace) / ".council")
    council_dir = config.init_workspace(workspace)

    print(f"Initialized council workspace at {council_dir}")
    print(f"  Config: {config.config_path}")
    print(f"  Data:   {council_dir / 'data'}")
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    """Launch the Council GUI."""
    workspace = _resolve_workspace(args)
    mock = getattr(args, 'mock', False)

    # Always ensure config/directory structure exists and defaults are backfilled.
    config_dir = Path(workspace) / ".council"
    config = Config(base_dir=config_dir)
    config.init_workspace(workspace)

    from council.gui.app import run_gui
    run_gui(workspace=workspace, mock=mock)
    return 0


def _check_binary(name: str, command: str) -> tuple[bool, str]:
    path = shutil.which(command)
    if path:
        return True, f"{name}: OK ({command} -> {path})"
    return False, f"{name}: MISSING ({command} not found on PATH)"


def _check_ollama_model(ollama_url: str, model: str) -> tuple[bool, str]:
    tags_url = f"{ollama_url.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return False, f"qwen: UNREACHABLE ({tags_url}: {exc})"
    except (TimeoutError, json.JSONDecodeError, ValueError) as exc:
        return False, f"qwen: ERROR reading models ({exc})"

    models = payload.get("models", [])
    names = {
        m.get("name", "")
        for m in models
        if isinstance(m, dict)
    }
    if model in names:
        return True, f"qwen: OK (model '{model}' available via {tags_url})"
    if names:
        sample = ", ".join(sorted(n for n in names if n)[:5])
        return False, f"qwen: MODEL MISSING ('{model}' not found; available: {sample})"
    return False, f"qwen: MODEL MISSING ('{model}' not found; Ollama returned no models)"


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check runtime prerequisites for a full 4-agent council run."""
    workspace = _resolve_workspace(args)
    config = Config(base_dir=Path(workspace) / ".council")

    print(f"Council doctor for workspace: {workspace}")

    failures = 0
    warnings = 0

    orchestrator_cfg = config.get("orchestrator", {})
    if not isinstance(orchestrator_cfg, dict):
        orchestrator_cfg = {}

    def _int_cfg(key: str, default: int) -> tuple[int, bool]:
        value = orchestrator_cfg.get(key, default)
        try:
            return int(value), True
        except (TypeError, ValueError):
            return default, False

    max_big, ok_big = _int_cfg("max_big_per_agent", 1)
    max_small, ok_small = _int_cfg("max_small_per_agent", 2)
    stalled_s, ok_stalled = _int_cfg("stalled_ticket_timeout_s", 300)

    print("Policy:")
    if ok_big and ok_small and ok_stalled and max_big >= 1 and max_small >= 1 and stalled_s >= 60:
        print(
            f"PASS  orchestrator: max_big_per_agent={max_big}, "
            f"max_small_per_agent={max_small}, stalled_ticket_timeout_s={stalled_s}"
        )
    else:
        warnings += 1
        print(
            f"WARN  orchestrator: invalid policy values detected; "
            f"using runtime clamps/defaults (big={max_big}, small={max_small}, stalled={stalled_s})"
        )

    display = os.environ.get("DISPLAY", "")
    if display:
        print(f"PASS  gui-display: DISPLAY={display}")
    else:
        warnings += 1
        print("WARN  gui-display: DISPLAY not set (GUI requires desktop/X11; headless launch can use xvfb-run)")

    for name in ["codex", "claude", "gemini"]:
        agent_cfg = config.get(f"agents.{name}", {}) or {}
        command = agent_cfg.get("command", name) if isinstance(agent_cfg, dict) else name
        ok, msg = _check_binary(name, command)
        if ok:
            print(f"PASS  {msg}")
        else:
            failures += 1
            print(f"FAIL  {msg}")

    qwen_cfg = config.get("agents.qwen", {}) or {}
    model = qwen_cfg.get("model", "qwen3:32b-q8_0") if isinstance(qwen_cfg, dict) else "qwen3:32b-q8_0"
    ollama_url = qwen_cfg.get("ollama_url", "http://localhost:11434") if isinstance(qwen_cfg, dict) else "http://localhost:11434"

    ok, msg = _check_binary("ollama", "ollama")
    if ok:
        print(f"PASS  {msg}")
    else:
        failures += 1
        print(f"FAIL  {msg}")

    ok, msg = _check_ollama_model(ollama_url, model)
    if ok:
        print(f"PASS  {msg}")
    else:
        failures += 1
        print(f"FAIL  {msg}")

    print(f"Summary: {failures} failed, {warnings} warnings")
    return 1 if failures else 0


def cmd_bridge(args: argparse.Namespace) -> int:
    """Start the FV Bridge API server."""
    try:
        import uvicorn
        from council.fv.bridge import app
    except ImportError as e:
        print(f"Error: FV Bridge dependencies not installed: {e}", file=sys.stderr)
        print("Install with: pip install -r requirements.txt", file=sys.stderr)
        return 1

    host = args.host
    port = args.port
    print(f"Starting FV Bridge API on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
    return 0


def _resolve_workspace(args: argparse.Namespace) -> str:
    """Resolve workspace path from args or cwd."""
    if hasattr(args, "workspace") and args.workspace:
        return str(Path(args.workspace).resolve())
    return os.getcwd()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="council",
        description="Ticket-based AI agent orchestration with GUI.",
    )
    parser.add_argument("--version", action="version", version=f"council {__version__}")
    parser.add_argument("--workspace", "-w", help="Workspace directory (default: cwd)")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init
    sp_init = subparsers.add_parser("init", help="Initialize council workspace")
    sp_init.add_argument("--workspace", "-w", help="Workspace directory")

    # gui
    sp_gui = subparsers.add_parser("gui", help="Launch Council GUI")
    sp_gui.add_argument("--workspace", "-w", help="Workspace directory")
    sp_gui.add_argument("--mock", action="store_true", help="Run with mock agents (no real binaries needed)")

    # doctor
    sp_doctor = subparsers.add_parser("doctor", help="Check environment readiness for 4-agent runs")
    sp_doctor.add_argument("--workspace", "-w", help="Workspace directory")

    # bridge
    sp_bridge = subparsers.add_parser("bridge", help="Start FV Bridge API server")
    sp_bridge.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    sp_bridge.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")

    return parser


def main() -> int:
    _setup_signal_handlers()
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        # Default to GUI
        args.command = "gui"
        if not hasattr(args, 'mock'):
            args.mock = False

    commands = {
        "init": cmd_init,
        "gui": cmd_gui,
        "doctor": cmd_doctor,
        "bridge": cmd_bridge,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
