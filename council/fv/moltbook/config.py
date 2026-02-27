"""
moltbook.config
───────────────
Loads .env from project root, exposes all settings.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# ── Load .env (no python-dotenv dependency) ──

def _load_dotenv():
    for candidate in [
        Path(__file__).resolve().parent.parent.parent.parent / ".env",  # council_ui/.env
        Path(__file__).resolve().parent.parent / ".env",                # council/fv/.env
        Path.cwd() / ".env",
    ]:
        if candidate.exists():
            with open(candidate) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    # inline comment strip
                    if "#" in val:
                        val = val[:val.index("#")].strip()
                    if key not in os.environ:
                        os.environ[key] = val
            return

_load_dotenv()

# ── API ──

API_BASE = "https://www.moltbook.com/api/v1"
CRED_DIR = Path.home() / ".config" / "moltbook"
CRED_FILE = CRED_DIR / "credentials.json"


def load_api_key() -> str | None:
    key = os.environ.get("MOLTBOOK_API_KEY")
    if key:
        return key
    if CRED_FILE.exists():
        try:
            return json.loads(CRED_FILE.read_text()).get("api_key")
        except (json.JSONDecodeError, OSError):
            return None
    return None


def save_credentials(api_key: str, agent_name: str = ""):
    CRED_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(CRED_DIR, 0o700)
    CRED_FILE.write_text(json.dumps({"api_key": api_key, "agent_name": agent_name}, indent=2))
    os.chmod(CRED_FILE, 0o600)
    print(f"[moltbook] Credentials saved → {CRED_FILE}")


def load_agent_name() -> str:
    if CRED_FILE.exists():
        return json.loads(CRED_FILE.read_text()).get("agent_name", "")
    return ""


def require_api_key() -> str:
    key = load_api_key()
    if not key:
        raise RuntimeError("No MOLTBOOK_API_KEY. Set it in .env or run: python -m council.fv.moltbook register")
    return key


# ── Behavior settings ──

def posting_enabled() -> bool:
    return os.environ.get("POSTING_ENABLED", "0") == "1"

def moltbook_mode() -> str:
    return os.environ.get("MOLTBOOK_MODE", "both")

def default_submolt() -> str:
    return os.environ.get("MOLTBOOK_SUBMOLT", "general")

def max_actions_per_run() -> int:
    try:
        return int(os.environ.get("MAX_ACTIONS_PER_RUN", "10"))
    except (ValueError, TypeError):
        return 10

def min_action_gap_seconds() -> float:
    try:
        return float(os.environ.get("MIN_POST_GAP_SECONDS", "25"))
    except (ValueError, TypeError):
        return 25.0

def ignore_seen() -> bool:
    return os.environ.get("IGNORE_SEEN", "0") == "1"

def max_thread_depth() -> int:
    try:
        return int(os.environ.get("MAX_THREAD_DEPTH", "10"))
    except (ValueError, TypeError):
        return 10

def comment_interval_seconds() -> float:
    try:
        return float(os.environ.get("COMMENT_INTERVAL_SECONDS", "25"))
    except (ValueError, TypeError):
        return 25.0
