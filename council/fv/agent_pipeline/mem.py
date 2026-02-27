"""
agent_pipeline.mem
------------------
Core memory layer for FV v3.0.

SQLite for structured storage (dialogue turns, knowledge facts).
FAISS IndexFlatIP + IndexIDMap2 for vector search.
SentenceTransformer (BAAI/bge-small-en-v1.5, 384-dim) for embeddings.
CTTRetriever singleton for Closer-to-Truth RAG.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

# -- Embedding singleton ------------------------------------------------

_EMBED_MODEL: Any = None
_EMBED_DIM = 384
_EMBED_NAME = "BAAI/bge-small-en-v1.5"


def _embed(texts: list[str]) -> np.ndarray:
    """
    Encode texts into normalized 384-dim vectors.
    Lazily loads BAAI/bge-small-en-v1.5 on first call.
    """
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        device = os.getenv("EMBED_DEVICE", "cpu")
        _EMBED_MODEL = SentenceTransformer(_EMBED_NAME, device=device)
    vecs = _EMBED_MODEL.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype="float32")


# -- CTT Retriever singleton --------------------------------------------

_CTT: Any = None


def get_ctt():
    """Return a singleton CTTRetriever (loaded once)."""
    global _CTT
    if _CTT is not None:
        return _CTT

    from council.fv.ctt_rag.ctt_retriever import CTTRetriever

    # Search for the index in common locations
    candidates = [
        Path("data/ctt_index"),
        Path(__file__).resolve().parent.parent / "data" / "ctt_index",
        Path("fv_ctt_rag/index_ctt"),
    ]
    for p in candidates:
        if p.is_dir() and (p / "ctt.faiss").exists():
            _CTT = CTTRetriever(str(p))
            return _CTT

    raise FileNotFoundError(
        f"CTT index not found. Searched: {[str(c) for c in candidates]}"
    )


# -- SQLite helpers -----------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS dialogue (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    ts    REAL    NOT NULL,
    role  TEXT    NOT NULL,
    body  TEXT    NOT NULL,
    sess  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS knowledge (
    rowid   INTEGER PRIMARY KEY AUTOINCREMENT,
    uid     TEXT    UNIQUE NOT NULL,
    body    TEXT    NOT NULL,
    tags    TEXT    NOT NULL DEFAULT '{}',
    weight  REAL    NOT NULL DEFAULT 0.5,
    sess    TEXT    NOT NULL,
    created REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_dialogue_sess ON dialogue(sess);
CREATE INDEX IF NOT EXISTS ix_knowledge_sess ON knowledge(sess);
CREATE INDEX IF NOT EXISTS ix_knowledge_uid  ON knowledge(uid);
"""


def open_db(path: str | Path | None = None) -> sqlite3.Connection:
    """Create / open the FV SQLite database and ensure schema exists."""
    if path is None:
        path = os.getenv("AGENT_DB", "runtime/fv.db")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_SCHEMA)
    # Migration: add memtype column if missing (idempotent)
    try:
        conn.execute("ALTER TABLE knowledge ADD COLUMN memtype TEXT DEFAULT 'unknown'")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists
    conn.commit()
    return conn


def log_turn(conn: sqlite3.Connection, role: str, body: str, sess: str) -> int:
    """Insert a dialogue row. Returns the rowid."""
    cur = conn.execute(
        "INSERT INTO dialogue (ts, role, body, sess) VALUES (?, ?, ?, ?)",
        (time.time(), role, body, sess),
    )
    conn.commit()
    return cur.lastrowid


def recent_dialogue(
    conn: sqlite3.Connection, n: int = 20, sess: str | None = None
) -> list[dict]:
    """Return the last *n* dialogue turns as dicts, oldest first."""
    if sess:
        rows = conn.execute(
            "SELECT rowid, ts, role, body FROM dialogue WHERE sess=? ORDER BY rowid DESC LIMIT ?",
            (sess, n),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT rowid, ts, role, body FROM dialogue ORDER BY rowid DESC LIMIT ?",
            (n,),
        ).fetchall()
    return [
        {"rowid": r[0], "ts": r[1], "role": r[2], "body": r[3]}
        for r in reversed(rows)
    ]


# -- Recency decay ------------------------------------------------------

_RECENCY_HALF_LIFE = float(os.getenv("RECENCY_HALF_LIFE_DAYS", "30"))


def recency_weight(created_ts: float, half_life_days: float = _RECENCY_HALF_LIFE) -> float:
    """Exponential decay: returns 1.0 for brand-new facts, 0.5 after half_life_days."""
    age_days = (time.time() - created_ts) / 86400.0
    if age_days <= 0:
        return 1.0
    return 2.0 ** (-age_days / half_life_days)


# -- FactIndex (FAISS + knowledge table) --------------------------------

def _make_uid(text: str) -> str:
    """Deterministic UID from text content."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class FactIndex:
    """
    FAISS IndexFlatIP wrapped in IndexIDMap2, backed by the knowledge table.

    - put() inserts a fact into both SQLite and FAISS.
    - search() does vector search with optional session filter.
    - flush() persists the FAISS index to disk.
    - query_raw() returns (uid, body, meta_dict, score) tuples for bridge compat.
    """

    def __init__(self, conn: sqlite3.Connection, faiss_path: str | Path | None = None):
        self.conn = conn
        self.faiss_path = Path(faiss_path) if faiss_path else self._default_faiss_path()
        self.faiss_path.parent.mkdir(parents=True, exist_ok=True)

        # Build FAISS index
        inner = faiss.IndexFlatIP(_EMBED_DIM)
        self.idx = faiss.IndexIDMap2(inner)
        self._next_id = 1

        # uid -> faiss numeric id mapping
        self._uid_to_fid: dict[str, int] = {}
        self._fid_to_uid: dict[int, str] = {}

        # Load persisted index if it exists
        if self.faiss_path.exists():
            try:
                loaded = faiss.read_index(str(self.faiss_path))
                if loaded.ntotal > 0:
                    self.idx = loaded
                    self._next_id = loaded.ntotal + 1
                    self._rebuild_uid_map()
            except Exception:
                pass  # start fresh

        # Sync: if DB has more facts than FAISS, rebuild entirely from DB
        db_count = self.conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0]
        if db_count > self.idx.ntotal:
            self._rebuild_from_db()
        elif self.idx.ntotal == 0 and db_count > 0:
            self._rebuild_from_db()

    @property
    def index(self):
        """Alias so bridge code can do `.index.ntotal`."""
        return self.idx

    @staticmethod
    def _default_faiss_path() -> Path:
        return Path(os.getenv("FAISS_PATH", "runtime/fv.faiss"))

    def _rebuild_uid_map(self):
        """Rebuild uid<->fid maps from the knowledge table after loading a persisted index."""
        rows = self.conn.execute(
            "SELECT rowid, uid FROM knowledge ORDER BY rowid"
        ).fetchall()
        for rowid, uid in rows:
            self._uid_to_fid[uid] = rowid
            self._fid_to_uid[rowid] = uid
        if rows:
            self._next_id = max(r[0] for r in rows) + 1

    def _rebuild_from_db(self):
        """Rebuild the FAISS index from all knowledge rows."""
        rows = self.conn.execute(
            "SELECT rowid, uid, body FROM knowledge ORDER BY rowid"
        ).fetchall()
        if not rows:
            return

        # Reset FAISS index
        inner = faiss.IndexFlatIP(_EMBED_DIM)
        self.idx = faiss.IndexIDMap2(inner)
        self._uid_to_fid.clear()
        self._fid_to_uid.clear()

        texts = [r[2] for r in rows]
        ids = np.array([r[0] for r in rows], dtype="int64")
        vecs = _embed(texts)
        self.idx.add_with_ids(vecs, ids)

        for rowid, uid, _ in rows:
            self._uid_to_fid[uid] = rowid
            self._fid_to_uid[rowid] = uid
        self._next_id = max(ids) + 1

    def put(
        self,
        text: str,
        tags: dict | str | None = None,
        weight: float = 0.5,
        sess: str | None = None,
        memtype: str = "unknown",
    ) -> str | None:
        """
        Insert a fact. Returns uid on success, None if duplicate/empty.
        memtype: one of 'preference', 'episodic', 'semantic', 'entity', 'unknown'.
        """
        text = (text or "").strip()
        if not text or len(text) < 5:
            return None

        uid = _make_uid(text)
        if uid in self._uid_to_fid:
            return uid  # dedup

        if sess is None:
            sess = os.getenv("SESSION_ID", "default-session")

        # Normalize tags to JSON string
        import json as _json
        if tags is None:
            tags_str = "{}"
        elif isinstance(tags, str):
            tags_str = tags
        else:
            tags_str = _json.dumps(tags)

        try:
            cur = self.conn.execute(
                "INSERT INTO knowledge (uid, body, tags, weight, sess, created, memtype) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (uid, text, tags_str, weight, sess, time.time(), memtype),
            )
            self.conn.commit()
            fid = cur.lastrowid
        except sqlite3.IntegrityError:
            # uid uniqueness violation -- already exists
            return uid

        # Add to FAISS
        vec = _embed([text])
        fid_arr = np.array([fid], dtype="int64")
        self.idx.add_with_ids(vec, fid_arr)

        self._uid_to_fid[uid] = fid
        self._fid_to_uid[fid] = uid
        self._next_id = fid + 1

        return uid

    def flush(self):
        """Persist the FAISS index to disk."""
        faiss.write_index(self.idx, str(self.faiss_path))

    def search(
        self,
        query: str,
        top_k: int = 20,
        final_k: int = 6,
        sess: str | None = None,
        memtype: str | None = None,
    ) -> list[dict]:
        """
        Vector search returning list of dicts:
        {uid, body, tags, weight, score, sess, memtype, created}
        Score is recency-adjusted.
        """
        if self.idx.ntotal == 0:
            return []

        q_vec = _embed([query])
        k = min(top_k, self.idx.ntotal)
        scores, ids = self.idx.search(q_vec, k)
        scores, ids = scores[0], ids[0]

        import json as _json
        results = []
        for score, fid in zip(scores, ids):
            if int(fid) < 0:
                continue
            uid = self._fid_to_uid.get(int(fid))
            if uid is None:
                continue

            row = self.conn.execute(
                "SELECT body, tags, weight, sess, created, memtype FROM knowledge WHERE rowid=?",
                (int(fid),),
            ).fetchone()
            if row is None:
                continue

            body, tags_str, weight, row_sess, created_ts, row_memtype = row

            # Session filter
            if sess and row_sess != sess:
                continue

            # Memory-type filter
            if memtype and row_memtype != memtype:
                continue

            try:
                tags_dict = _json.loads(tags_str) if tags_str else {}
            except (ValueError, TypeError):
                tags_dict = {}

            adjusted_score = float(score) * recency_weight(created_ts)

            results.append({
                "uid": uid,
                "body": body,
                "tags": tags_dict,
                "weight": weight,
                "score": adjusted_score,
                "sess": row_sess,
                "memtype": row_memtype or "unknown",
                "created": created_ts,
            })

            if len(results) >= final_k:
                break

        return results

    def query_topk(
        self, query: str, k: int = 10
    ) -> list[tuple[str, str, dict, float]]:
        """Alias for query_raw -- bridge compat."""
        return self.query_raw(query, k=k)

    def query_raw(
        self, query: str, k: int = 10
    ) -> list[tuple[str, str, dict, float]]:
        """
        Raw FAISS search returning (uid, body, meta_dict, score) tuples.
        Used by bridge.py /context endpoint.
        meta_dict contains: {tags: {...}, weight: float, sess: str}
        """
        if self.idx.ntotal == 0:
            return []

        q_vec = _embed([query])
        actual_k = min(k, self.idx.ntotal)
        scores, ids = self.idx.search(q_vec, actual_k)
        scores, ids = scores[0], ids[0]

        import json as _json
        results = []
        for score, fid in zip(scores, ids):
            if int(fid) < 0:
                continue
            uid = self._fid_to_uid.get(int(fid))
            if uid is None:
                continue

            row = self.conn.execute(
                "SELECT body, tags, weight, sess, created, memtype FROM knowledge WHERE rowid=?",
                (int(fid),),
            ).fetchone()
            if row is None:
                continue

            body, tags_str, weight, row_sess, created_ts, row_memtype = row
            try:
                tags_dict = _json.loads(tags_str) if tags_str else {}
            except (ValueError, TypeError):
                tags_dict = {}

            adjusted_score = float(score) * recency_weight(created_ts)
            meta = {
                "tags": tags_dict,
                "weight": weight,
                "sess": row_sess,
                "memtype": row_memtype or "unknown",
                "created": created_ts,
            }
            results.append((uid, body, meta, adjusted_score))

        return results


def fetch_relevant(
    index: FactIndex,
    query: str,
    top_k: int = 20,
    final_k: int = 6,
    sess: str | None = None,
) -> list[tuple[float, str, dict, str]]:
    """
    Returns (score, text, tags_dict, uid) tuples -- compat format for
    moltbook/brain.py retrieve_rerank() and grounding.py.
    """
    results = index.search(query, top_k=top_k, final_k=final_k, sess=sess)
    return [
        (r["score"], r["body"], r["tags"], r["uid"])
        for r in results
    ]
