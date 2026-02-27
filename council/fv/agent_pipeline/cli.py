"""
agent_pipeline.cli
------------------
CLI entry point for FV v3.0.

    python -m council.fv.agent_pipeline.cli

Interactive chat with Rich evidence panels.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path

# -- Load .env before anything reads env vars --------------------------

def _load_env():
    """Load .env from project root (no dependency on python-dotenv)."""
    for candidate in [
        Path(__file__).resolve().parent.parent.parent.parent / ".env",
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
                    if "#" in val:
                        val = val[: val.index("#")].strip()
                    if key not in os.environ:
                        os.environ[key] = val
            return

_load_env()

# -- GPU pin + logger silencing ----------------------------------------

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

# Silence chatty libraries
for name in [
    "sentence_transformers",
    "transformers",
    "torch",
    "faiss",
    "httpx",
    "urllib3",
]:
    logging.getLogger(name).setLevel(logging.WARNING)

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger("fv.cli")


# -- Health check ------------------------------------------------------

def health_check() -> dict:
    """
    Quick check that the LLM backend, CUDA, and embeddings are reachable.
    Returns a status dict.
    """
    import requests
    import torch

    status = {"llm": False, "cuda": False, "embeddings": False}

    # LLM backend
    base = os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
    try:
        root = base.rsplit("/v1", 1)[0]
        r = requests.get(root, timeout=3)
        status["llm"] = r.status_code < 500
        status["llm_model"] = os.getenv("LLM_MODEL", "unknown")
    except Exception:
        pass

    # CUDA
    try:
        status["cuda"] = torch.cuda.is_available()
        if status["cuda"]:
            status["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass

    # Embeddings (try loading the model)
    try:
        from .mem import _embed
        _embed(["health check"])
        status["embeddings"] = True
    except Exception:
        pass

    return status


# -- Evidence panel rendering ------------------------------------------

def _render_evidence(result: dict):
    """Print evidence and answer using Rich panels."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        console = Console()
    except ImportError:
        _render_plain(result)
        return

    evidence = result.get("evidence", [])
    if evidence:
        table = Table(title="Evidence", show_lines=True, expand=True)
        table.add_column("ID", width=8)
        table.add_column("Source", width=8)
        table.add_column("Text", ratio=1)
        table.add_column("Score", width=6, justify="right")

        for e in evidence:
            if not isinstance(e, dict):
                continue
            eid = str(e.get("id", "?"))
            src = str(e.get("source", "?"))
            txt = (e.get("text") or "")[:200]
            score = e.get("score")
            score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "?"
            table.add_row(eid, src, txt, score_str)

        console.print(table)

    answer = result.get("answer", "(no answer)")
    conf = result.get("confidence", 0.0)
    at = result.get("answer_type", "answer")
    ku = result.get("known_unknown", "unsure")
    citations = result.get("citations", [])

    color = "green" if conf >= 0.7 else "yellow" if conf >= 0.4 else "red"

    header = f"[{at}] conf={conf:.2f} ku={ku}"
    if citations:
        header += f" cites={','.join(str(c) for c in citations)}"

    console.print(Panel(answer, title=header, border_style=color))

    risks = result.get("risks", [])
    notes = result.get("notes", "")
    if risks:
        console.print(f"[dim]Risks: {', '.join(str(r) for r in risks)}[/dim]")
    if notes:
        console.print(f"[dim]Notes: {notes}[/dim]")

    reviews = result.get("claim_reviews", [])
    if reviews:
        console.print()
        for r in reviews:
            if not isinstance(r, dict):
                continue
            status_icon = {"supported": "+", "contradicted": "X", "not_in_evidence": "?"}
            s = r.get("status", "?")
            icon = status_icon.get(s, "~")
            claim = r.get("claim", "")[:120]
            console.print(f"  [{icon}] {claim} ({s})")


def _render_plain(result: dict):
    """Fallback rendering without Rich."""
    evidence = result.get("evidence", [])
    if evidence:
        print("\n--- Evidence ---")
        for e in evidence:
            if isinstance(e, dict):
                print(f"  [{e.get('id','?')}] {(e.get('text',''))[:150]}")

    answer = result.get("answer", "(no answer)")
    conf = result.get("confidence", 0.0)
    at = result.get("answer_type", "answer")
    print(f"\n[{at}] (conf={conf:.2f})")
    print(answer)


# -- Chat loop ---------------------------------------------------------

def run_chat():
    """Interactive chat loop."""
    from .agent import ChatAgent

    print("FV v3.0 -- initializing...")
    agent = ChatAgent()
    print(f"Session: {agent._sess}")
    print(f"Facts in memory: {agent._fi.idx.ntotal}")
    print(f"Type 'quit' or Ctrl-C to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break

        try:
            result = agent.chat(user_input)
            _render_evidence(result)
        except Exception as e:
            logger.error(f"Chat error: {e}")
            print(f"Error: {e}")

    try:
        agent._fi.flush()
    except Exception:
        pass


# -- Main --------------------------------------------------------------

def main():
    """Entry point for `python -m council.fv.agent_pipeline.cli`."""
    import argparse

    ap = argparse.ArgumentParser(description="FV v3.0 CLI")
    ap.add_argument("--health", action="store_true", help="Run health check and exit")
    args = ap.parse_args()

    if args.health:
        import json
        status = health_check()
        print(json.dumps(status, indent=2))
        sys.exit(0 if status.get("llm") else 1)

    run_chat()


if __name__ == "__main__":
    main()
