"""
agent_pipeline.agent_short_memory
---------------------------------
Compatibility shim -- re-exports old names so moltbook/brain.py,
harness/grounding.py, and bridge.py work without import changes.

Maps v2.1 API -> v3.0 mem.py internals.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from .mem import FactIndex, fetch_relevant, get_ctt, log_turn, open_db

# Re-export get_ctt directly (used by grounding.py and bridge.py)
get_ctt = get_ctt

# Session ID constant (used by moltbook/brain.py)
SESSION_ID = os.getenv("SESSION_ID", "default-session")

# -- init_db ------------------------------------------------------------

_default_conn = None


def init_db():
    """Open (or return cached) the default database connection."""
    global _default_conn
    if _default_conn is None:
        _default_conn = open_db()
    return _default_conn


# -- VectorStore compat wrapper -----------------------------------------

class VectorStore:
    """
    Wraps FactIndex with the old VectorStore interface expected by
    moltbook/brain.py and bridge.py.

    Provides:
        .index          -> the underlying FAISS index (.ntotal works)
        .query_topk()   -> (uid, body, meta_dict, score) tuples
        .persist()      -> flush FAISS to disk
    """

    def __init__(self, conn=None):
        if conn is None:
            conn = init_db()
        self._fi = FactIndex(conn)

    @property
    def index(self):
        return self._fi.idx

    def query_topk(self, query: str, k: int = 10):
        """Returns list of (uid, body, meta_dict, score) tuples."""
        return self._fi.query_raw(query, k=k)

    def persist(self):
        self._fi.flush()


# -- save_turn ----------------------------------------------------------

def save_turn(conn, role: str, text: str, session_id: str | None = None):
    """Log a dialogue turn."""
    sess = session_id or SESSION_ID
    log_turn(conn, role, text, sess)


# -- upsert_fact --------------------------------------------------------

def upsert_fact(
    store,
    text: str,
    importance: float = 0.5,
    tags: dict | None = None,
    session_id: str | None = None,
    fact_type: str = "unknown",
) -> str | None:
    """
    Insert a fact into the vector store.

    `store` can be a FactIndex or a VectorStore wrapper.
    Returns uid on success, None if rejected.
    """
    sess = session_id or SESSION_ID

    if isinstance(store, VectorStore):
        fi = store._fi
    elif isinstance(store, FactIndex):
        fi = store
    else:
        # Fallback: assume it has a put() method
        return store.put(text, tags=tags, weight=importance, sess=sess, memtype=fact_type)

    return fi.put(text, tags=tags, weight=importance, sess=sess, memtype=fact_type)


# -- retrieve_rerank ----------------------------------------------------

def retrieve_rerank(
    store,
    query: str,
    top_k: int = 20,
    final_k: int = 6,
    session_id: str | None = None,
) -> list[tuple[float, str, dict, str]]:
    """
    Retrieve and rank facts. Returns (score, text, tags_dict, uid) tuples.
    Compat wrapper over fetch_relevant().
    """
    sess = session_id or SESSION_ID

    if isinstance(store, VectorStore):
        fi = store._fi
    elif isinstance(store, FactIndex):
        fi = store
    else:
        fi = store

    return fetch_relevant(fi, query, top_k=top_k, final_k=final_k, sess=sess)
