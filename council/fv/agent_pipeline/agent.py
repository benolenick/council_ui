"""
agent_pipeline.agent
--------------------
ChatAgent orchestrator for FV v3.0.

Replaces PipelineAgent + UnifiedMemorySystem with a clean, minimal
implementation that wires together mem.py and the harness.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .mem import (
    FactIndex,
    fetch_relevant,
    get_ctt,
    log_turn,
    open_db,
    recent_dialogue,
)

logger = logging.getLogger(__name__)


# -- MemoryHandle -------------------------------------------------------

class MemoryHandle:
    """
    Wraps FactIndex + SQLite conn, providing the interface that
    bridge.py, harness/grounding.py, and moltbook/brain.py expect.

    Key attributes / methods consumed by user code:
        .stm_store            -> the FactIndex  (bridge reads .stm_store.index.ntotal)
        .retrieve_facts(q, …) -> returns (score, text, tags_dict, uid) tuples
        .save_turn(role, text)
        .add_fact(text, tags, importance)
        .get_recent_turns(n)
    """

    def __init__(self, conn, fact_index: FactIndex, sess: str):
        self._conn = conn
        self._fi = fact_index
        self._sess = sess

    # -- bridge compat: agent.memory.stm_store.index.ntotal --

    @property
    def stm_store(self) -> FactIndex:
        return self._fi

    # -- retrieve facts (tuple format for grounding.py + bridge) --

    def retrieve_facts(
        self,
        query: str,
        top_k: int = 20,
        final_k: int = 6,
    ) -> list[tuple[float, str, dict, str]]:
        """Returns (score, text, tags_dict, uid) tuples."""
        return fetch_relevant(self._fi, query, top_k=top_k, final_k=final_k, sess=None)

    # -- dialogue --

    def save_turn(self, role: str, text: str):
        log_turn(self._conn, role, text, self._sess)

    def get_recent_turns(self, n: int = 20) -> list[dict]:
        return recent_dialogue(self._conn, n, self._sess)

    # -- facts --

    def add_fact(
        self,
        text: str,
        tags: dict | None = None,
        importance: float = 0.5,
        fact_type: str = "unknown",
    ) -> str | None:
        return self._fi.put(text, tags=tags, weight=importance, sess=self._sess, memtype=fact_type)


# -- ChatAgent ----------------------------------------------------------

# Regex for extracting facts from LLM output
_FACT_RE = re.compile(
    r"(?:^|\n)\s*[-*]\s*(.+?)(?=\n\s*[-*]|\n\n|\Z)", re.DOTALL
)

_EXTRACT_SYSTEM = """Extract key facts from this exchange that are worth remembering.
Return a bullet list of concise facts (one per line, prefixed with -).
Only include substantive claims -- skip greetings, hedging, and meta-commentary.
For each fact, classify it with a tag: [preference], [episodic], [semantic], or [entity].
Example: - User prefers short answers [preference]
Example: - Discussed consciousness on Feb 20 [episodic]
Example: - Qualia resist reductionist explanation [semantic]
Example: - Robert Lawrence Kuhn hosts Closer to Truth [entity]
If nothing is worth remembering, return: - (none)"""

_MEMTYPE_RE = re.compile(r"\[(preference|episodic|semantic|entity)\]\s*$", re.IGNORECASE)


class ChatAgent:
    """
    Top-level agent replacing PipelineAgent.

    Usage:
        agent = ChatAgent()
        result = agent.chat("What is consciousness?")
    """

    def __init__(self, sess: str | None = None):
        self._sess = sess or os.getenv("SESSION_ID", "default-session")
        self._conn = open_db()
        self._fi = FactIndex(self._conn)
        self._memory = MemoryHandle(self._conn, self._fi, self._sess)

    @property
    def memory(self) -> MemoryHandle:
        return self._memory

    def chat(self, message: str) -> dict:
        """
        Full pipeline: save turn -> harnessed_answer -> save response -> extract facts.
        Returns the harness result dict.
        """
        # 1. Save user turn
        self._memory.save_turn("user", message)

        # 2. Run the harness
        from .harness.harness import harnessed_answer

        result = harnessed_answer(message, memory=self._memory)

        # 3. Save assistant response
        answer = result.get("answer", "")
        if answer:
            self._memory.save_turn("assistant", answer)

        # 4. Extract and store facts (best-effort, non-blocking)
        try:
            self._extract_facts(message, answer)
        except Exception as e:
            logger.debug(f"Fact extraction skipped: {e}")

        # 5. Persist FAISS
        try:
            self._fi.flush()
        except Exception:
            pass

        return result

    def _extract_facts(self, question: str, answer: str):
        """Use the LLM to extract memorable facts from the exchange."""
        if not answer or len(answer) < 30:
            return

        from .harness.client import chat_completion

        prompt = (
            f"User asked: {question[:500]}\n\n"
            f"Agent answered: {answer[:1000]}\n\n"
            f"Extract key facts:"
        )

        raw = chat_completion(
            [
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=300,
        )

        if not raw or "(none)" in raw.lower():
            return

        for m in _FACT_RE.finditer(raw):
            fact = m.group(1).strip()
            if fact and len(fact) > 10 and fact.lower() != "(none)":
                # Parse [type] bracket if present
                type_match = _MEMTYPE_RE.search(fact)
                if type_match:
                    fact_type = type_match.group(1).lower()
                    fact = fact[:type_match.start()].strip()
                else:
                    fact_type = "unknown"
                self._memory.add_fact(
                    fact,
                    tags={"source": "extraction", "type": "auto_fact"},
                    importance=0.5,
                    fact_type=fact_type,
                )
