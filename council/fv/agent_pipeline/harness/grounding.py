# agent_pipeline/harness/grounding.py
from __future__ import annotations
from typing import List, Dict, Any

from council.fv.agent_pipeline.agent_short_memory import get_ctt


def format_evidence(chunks: List[Dict[str, Any]]) -> str:
    # Each chunk should carry an id and text (and optionally source)
    lines = []
    for c in chunks:
        cid = c.get("id") or c.get("source_id") or "unknown"
        txt = (c.get("text") or "").strip()
        if not txt:
            continue
        lines.append(f"[{cid}] {txt}")
    return "\n".join(lines)

def _fact_to_dict(f, i: int):
    """
    Normalizes fact items into a dict so downstream code can rely on .get().
    Supports dict facts or tuple/list facts returned by different memory backends.
    """
    if isinstance(f, dict):
        return f

    if isinstance(f, (tuple, list)):
        # Common tuple layouts we see:
        # (id, text, score)
        # (id, text, score, meta)
        # (text, score)
        _id = None
        text = None
        score = None
        meta = {}

        if len(f) == 4:
            _id, text, score, meta = f
        elif len(f) == 3:
            _id, text, score = f
        elif len(f) == 2:
            text, score = f
        elif len(f) == 1:
            (text,) = f

        d = {
            "id": _id or f"mem_{i+1}",
            "text": text,
            "score": score,
        }
        # meta might be None or not a dict
        if isinstance(meta, dict):
            d.update(meta)
        return d

    # Fallback: stringify unknown shapes
    return {"id": f"mem_{i+1}", "text": str(f), "score": None}


def retrieve_evidence(memory, query: str, top_k: int = 20, final_k: int = 6) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    # --- 1) CTT RAG FIRST ---
    try:
        ctt = get_ctt()
        ctt_hits = ctt.retrieve(query, final_k=final_k, rerank=True)

        for j, h in enumerate(ctt_hits or []):
            if isinstance(h, dict):
                txt = (h.get("text") or "").strip()
                score = h.get("score", None)
                cid = h.get("id", None)
            else:
                txt = str(h).strip()
                score = None
                cid = None

            if not txt:
                continue

            out.append({
                "id": cid or f"CTT-{j+1}",
                "text": txt,
                "source": "ctt",
                "score": score,
            })
    except Exception:
        pass

    # --- 2) STM / Memory facts (FAISS/SQLite) ---
    facts = memory.retrieve_facts(query, top_k=top_k, final_k=final_k) if memory else []
    for i, f in enumerate(facts or []):
        f = _fact_to_dict(f, i)
        out.append({
            "id": f.get("id") or f.get("source_id") or f"mem_{i+1}",
            "text": f.get("text", ""),
            "source": f.get("source", "memory"),
            "score": f.get("score", None),
        })

    # --- dedupe evidence (same text often appears twice) ---
    seen = set()
    deduped = []
    for e in out:
        key = (e.get("source"), (e.get("text") or "").strip())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(e)

    return deduped
