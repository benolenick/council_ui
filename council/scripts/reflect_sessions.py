#!/usr/bin/env python3
"""
reflect_sessions.py -- Cron script (every 30 min)

Reads OpenClaw session .jsonl files, extracts user/assistant turns,
calls POST /reflect on the FV bridge to extract and store memorable facts,
and writes a digest to the daily memory file so Qwen reads it next session.
"""

import json
import os
import sys
import time
import requests
from datetime import datetime, timezone
from pathlib import Path

# --- Config ---
BRIDGE_URL = "http://127.0.0.1:8000"
SESSIONS_DIR = Path.home() / ".openclaw" / "agents" / "main" / "sessions"
STATE_FILE = Path.home() / ".openclaw" / "memory" / "reflect_state.json"
MEMORY_DIR = Path.home() / ".openclaw" / "workspace" / "memory"
MIN_TURNS = 4          # skip trivial exchanges
COOLDOWN_SEC = 300     # 5 min -- don't reflect on active conversations
MAX_TURNS_PER_CALL = 60  # cap transcript size sent to /reflect


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def bridge_healthy() -> bool:
    try:
        r = requests.get(f"{BRIDGE_URL}/health", timeout=5)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except Exception:
        return False


def parse_session_jsonl(path: Path, start_offset: int = 0):
    """
    Parse a session .jsonl file from a byte offset.
    Returns (turns, last_timestamp, new_offset).
    turns: list of {"role": str, "text": str}
    """
    file_size = path.stat().st_size
    if file_size <= start_offset:
        return [], 0, start_offset

    turns = []
    last_ts = 0

    with open(path, "r", encoding="utf-8") as f:
        f.seek(start_offset)
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            if event.get("type") != "message":
                continue

            msg = event.get("message", {})
            role = msg.get("role", "")
            if role not in ("user", "assistant"):
                continue

            # Extract text from content blocks
            content = msg.get("content", [])
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                text = "\n".join(text_parts)
            else:
                continue

            if not text.strip():
                continue

            # Strip system metadata prefixes from user messages
            cleaned = _clean_turn_text(text, role)
            if not cleaned:
                continue

            # Parse timestamp
            ts_str = msg.get("timestamp") or event.get("timestamp", "")
            try:
                if isinstance(ts_str, (int, float)):
                    ts = float(ts_str) / 1000.0
                else:
                    ts = float(ts_str) / 1000.0 if ts_str.isdigit() else 0
            except (ValueError, AttributeError):
                ts = 0
            if ts > last_ts:
                last_ts = ts

            turns.append({"role": role, "text": cleaned})

    new_offset = file_size
    return turns, last_ts, new_offset


def _clean_turn_text(text: str, role: str) -> str:
    """Strip system prefixes and metadata from turn text."""
    if role == "user":
        lines = text.split("\n")
        cleaned_lines = []
        skip_block = False
        for line in lines:
            # Skip system notifications
            if line.startswith("System: ["):
                continue
            # Skip conversation metadata blocks
            if line.strip() == "Conversation info (untrusted metadata):":
                skip_block = True
                continue
            if skip_block:
                if line.strip().startswith("```"):
                    if skip_block and line.strip() == "```":
                        skip_block = False
                    continue
                if line.strip().startswith("{") or line.strip().startswith("}") or line.strip().startswith('"'):
                    continue
                skip_block = False
            # Skip queued message headers
            if line.startswith("[Queued messages"):
                continue
            if line.strip() == "---":
                continue
            if line.startswith("Queued #"):
                continue
            cleaned_lines.append(line)

        result = "\n".join(cleaned_lines).strip()
        return result

    # For assistant role: skip tool call messages (they have no useful text)
    if role == "assistant":
        # If the text looks like an error or is very short system stuff, skip
        if text.startswith("[LLM ERROR") or text.startswith("[LLM_ERROR"):
            return ""
        return text.strip()

    return text.strip()


def call_reflect(turns: list, session_label: str) -> dict:
    """POST /reflect with turns, return response dict."""
    payload = {
        "turns": turns[-MAX_TURNS_PER_CALL:],
        "session_label": session_label,
    }
    r = requests.post(
        f"{BRIDGE_URL}/reflect",
        json=payload,
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def write_digest(facts: list):
    """Append reflection digest to today's memory file."""
    if not facts:
        return

    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    path = MEMORY_DIR / f"{today}.md"

    lines = []
    if path.exists():
        lines = path.read_text().splitlines()

    # Check if Reflections header already exists
    has_header = any("## Reflections" in line for line in lines)

    new_entries = []
    ts = datetime.now().strftime("%H:%M")
    for fact in facts:
        text = fact.get("text", "")
        cat = fact.get("category", "")
        imp = fact.get("importance", 0.5)
        tag = f" [{cat}]" if cat else ""
        new_entries.append(f"- {text}{tag} (importance: {imp:.1f}) -- reflected at {ts}")

    if not has_header:
        lines.append("")
        lines.append("## Reflections")
        lines.append("")

    lines.extend(new_entries)
    path.write_text("\n".join(lines) + "\n")
    log(f"  Wrote {len(new_entries)} reflection(s) to {path.name}")


def main():
    log("reflect_sessions starting")

    # Safety: bridge must be healthy
    if not bridge_healthy():
        log("Bridge not healthy, skipping")
        return

    state = load_state()

    # Find session .jsonl files (skip backups)
    if not SESSIONS_DIR.exists():
        log(f"Sessions dir not found: {SESSIONS_DIR}")
        return

    session_files = sorted(SESSIONS_DIR.glob("*.jsonl"))
    session_files = [f for f in session_files if ".bak" not in f.name]

    if not session_files:
        log("No session files found")
        return

    all_stored_facts = []

    for sf in session_files:
        fname = sf.name
        prev_offset = state.get(fname, {}).get("offset", 0)
        file_size = sf.stat().st_size

        if file_size <= prev_offset:
            continue  # no new data

        log(f"Processing {fname} (offset {prev_offset} -> {file_size})")

        turns, last_ts, new_offset = parse_session_jsonl(sf, prev_offset)

        if len(turns) < MIN_TURNS:
            log(f"  Only {len(turns)} turns, skipping (need {MIN_TURNS}+)")
            # Still update offset so we don't re-parse these lines
            state[fname] = {"offset": new_offset, "last_reflect": time.time()}
            continue

        # Cooldown: don't reflect on active conversations
        now = time.time()
        if last_ts > 0 and (now - last_ts) < COOLDOWN_SEC:
            log(f"  Last message {int(now - last_ts)}s ago, conversation still active, skipping")
            continue

        # Call /reflect
        try:
            label = fname.replace(".jsonl", "")
            result = call_reflect(turns, label)
            stored = result.get("stored", [])
            extracted = result.get("facts_extracted", 0)
            log(f"  Extracted {extracted} facts, stored {len(stored)}")
            all_stored_facts.extend(stored)
        except Exception as e:
            log(f"  /reflect call failed: {e}")
            continue

        # Update state
        state[fname] = {"offset": new_offset, "last_reflect": time.time()}

    save_state(state)

    # Write digest to daily memory file
    if all_stored_facts:
        write_digest(all_stored_facts)

    log(f"Done. {len(all_stored_facts)} total facts stored.")


if __name__ == "__main__":
    main()
