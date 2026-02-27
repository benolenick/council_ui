"""
moltbook.brain
--------------
Bridges Moltbook content through FV's pipeline.

For Moltbook posts/comments, we use llm_raw() which calls the LLM directly
WITHOUT the RAG "STRICT GROUNDED MODE" prompt. RAG sources are optionally
included as background context, not as mandatory citation rules.
"""

from __future__ import annotations

import os
import re
import sys
import time
import requests
from typing import Optional, List

MOLTBOOK_DEBUG = os.environ.get("MOLTBOOK_DEBUG", "0") == "1"

# -- Lazy-loaded FV subsystems --

_conn = None
_store = None


def _init_stm():
    global _conn, _store
    if _conn is None:
        from council.fv.agent_pipeline.agent_short_memory import init_db, VectorStore
        _conn = init_db()
        _store = VectorStore(_conn)
    return _conn, _store


def _save_to_stm(role: str, text: str, tags: dict | None = None, importance: float = 0.5):
    from council.fv.agent_pipeline.agent_short_memory import save_turn, upsert_fact, SESSION_ID
    conn, store = _init_stm()
    save_turn(conn, role, text, session_id=SESSION_ID)
    if len(text.strip()) > 15:
        upsert_fact(store, text, tags=tags, importance=importance, session_id=SESSION_ID)


def _retrieve_memories(query: str, top_k: int = 20, final_k: int = 8) -> List[str]:
    from council.fv.agent_pipeline.agent_short_memory import retrieve_rerank, SESSION_ID
    conn, store = _init_stm()
    results = retrieve_rerank(store, query, top_k=top_k, final_k=final_k, session_id=SESSION_ID)
    texts = [txt for _, txt, _, _ in results] if results else []
    if MOLTBOOK_DEBUG:
        print(f"[DEBUG:STM] Retrieved {len(texts)} memories for query: {query[:60]}...")
    return texts


def llm_raw(system_prompt: str, user_prompt: str) -> str:
    base = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8080/v1").rstrip("/")
    api_key = os.environ.get("LLM_API_KEY", "none")
    model = os.environ.get("LLM_MODEL", "gpt-oss:20b")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.8, "max_tokens": 600,
        "options": {"num_ctx": 16384}, "stream": False,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    try:
        r = requests.post(f"{base}/chat/completions", json=payload, headers=headers, timeout=600)
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"].strip()
        return _clean_output(raw)
    except Exception as e:
        if MOLTBOOK_DEBUG:
            print(f"[DEBUG:LLM] Error: {e}")
        return ""


def _clean_output(text: str) -> str:
    if not text:
        return ""
    if "<|" in text:
        parts = re.split(r"<\|[^>]*\|>", text)
        for candidate in reversed(parts):
            candidate = candidate.strip()
            if len(candidate) > 30 and not candidate.lower().startswith("we need"):
                text = candidate
                break
        text = re.sub(r"<\|[^>]*\|>", "", text).strip()
    reasoning_prefixes = [
        "we need to", "analysis", "let me", "the sources", "we must",
        "the question", "we can", "the topic", "we should", "let's think",
        "first,", "okay,", "the user",
    ]
    lower = text.lower()
    if any(lower.startswith(p) for p in reasoning_prefixes):
        quoted = re.findall(r'"([^"]{40,})"', text)
        if quoted:
            text = quoted[-1]
        else:
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            for p in paragraphs:
                pl = p.lower()
                if not any(pl.startswith(rp) for rp in reasoning_prefixes) and len(p) > 30:
                    text = "\n\n".join(paragraphs[paragraphs.index(p):])
                    break
    for marker in ["ANSWER (grounded, cited):", "ANSWER:", "POST:"]:
        if marker in text:
            _, _, after = text.partition(marker)
            after = after.strip()
            if len(after) > 30:
                text = after
    text = re.sub(r"^```\w*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    for prefix in [
        "here is the post:", "here is a post:", "here's the post:",
        "here's a post:", "here is my post:", "here's my post:",
        "sure, here", "okay, here",
    ]:
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()
    return text.strip()


def _get_rag_context(topic: str) -> str:
    try:
        from council.fv.agent_pipeline.agent_short_memory import get_ctt
        from council.fv.ctt_rag.ctt_prompting import format_ctt_sources
        ctt = get_ctt()
        chunks = ctt.retrieve(topic, final_k=4, rerank=False)
        if chunks:
            sources = format_ctt_sources(chunks)
            return (
                f"\nFor background (you may reference these if relevant, but don't force it):\n"
                f"{sources}\n"
            )
    except Exception:
        pass
    return ""


MOLTBOOK_SYSTEM = (
    "You are an AI agent posting on Moltbook, a social network for AI agents. "
    "Write concise, opinionated takes. Start directly with your point -- "
    "NEVER open with greetings, salutations, or emoji. No 'Hey fellow AIs!', "
    "no 'Hi everyone!', no emoji openers. "
    "Be intellectually sharp and conversational. Keep posts under 250 words. "
    "If you reference sources, cite them naturally like [CTT-1]. "
    "Do NOT include any reasoning, analysis, or meta-commentary about how to write the post. "
    "Do NOT echo back any instructions or rules. "
    "Output ONLY the post text itself."
)


def generate_post(topic: str, submolt: str = "general") -> dict:
    _init_stm()
    memories = _retrieve_memories(topic, final_k=4)
    mem_block = ""
    if memories:
        mem_block = "\nThings you know: " + " | ".join(m[:100] for m in memories[:3]) + "\n"
    rag_ctx = _get_rag_context(topic)
    post_prompt = (
        f"Write a post for m/{submolt} about: {topic}\n"
        f"{mem_block}{rag_ctx}\n"
        f"Remember: output ONLY the post text. No analysis, no reasoning, no meta-commentary."
    )
    content = llm_raw(MOLTBOOK_SYSTEM, post_prompt)
    content = re.sub(
        r"^(hey|hi|hello|greetings|what'?s up|howdy)[^\n.!?]{0,40}[.!?\s]*",
        "", content, count=1, flags=re.IGNORECASE
    ).strip()
    title_prompt = (
        f"Write a short catchy title (max 10 words) for this post:\n"
        f"{content[:200]}\n\nOutput ONLY the title text. No quotes, no explanation."
    )
    title = llm_raw("You output only a short title. Nothing else.", title_prompt)
    title = title.split("\n")[0].strip().strip('"').strip("'").strip()
    if len(title) > 100:
        title = title[:97] + "..."
    if len(title) < 3 or title.lower().startswith("the sources") or "[LLM" in title:
        title = f"Thoughts on {topic}"[:100]
    if len(content) > 2000:
        content = content[:1997] + "..."
    if len(content) < 20 or "[LLM" in content:
        content = f"Been thinking about {topic}. What are your thoughts, fellow agents?"
    _save_to_stm("assistant", f"[Posted to m/{submolt}] {title}: {content[:300]}",
                 tags={"source": "moltbook", "type": "my_post", "submolt": submolt}, importance=0.8)
    return {"title": title, "content": content}


def generate_comment(post_title: str, post_content: str,
                     author: str = "", existing_comments: Optional[List[dict]] = None,
                     parent_comment: Optional[str] = None) -> str:
    recall_query = f"{author} {post_title} {post_content[:200]}"
    memories = _retrieve_memories(recall_query, final_k=4)
    mem_block = ""
    if memories:
        mem_block = "\nThings you remember: " + " | ".join(m[:80] for m in memories[:3]) + "\n"
    prompt_parts = [
        f"Reply to this Moltbook post:\n\n",
        f"POST: {post_title}\n", f"BY: {author}\n", f"{post_content[:500]}\n",
    ]
    if parent_comment:
        prompt_parts.append(f"\nYou're replying to this specific comment:\n{parent_comment[:300]}\n")
    if existing_comments:
        lines = []
        for c in existing_comments[:4]:
            c_author = c.get("author", {}).get("name", "?") if isinstance(c.get("author"), dict) else str(c.get("author", "?"))
            c_text = (c.get("content") or "")[:150]
            lines.append(f"  [{c_author}]: {c_text}")
        if lines:
            prompt_parts.append(f"\nOther comments:\n" + "\n".join(lines) + "\n")
    prompt_parts.append(f"{mem_block}")
    prompt_parts.append(
        f"\nWrite a short reply (under 120 words). Be conversational. "
        f"If you've talked to {author} before, reference it. "
        f"Output ONLY the reply text."
    )
    comment = llm_raw(MOLTBOOK_SYSTEM, "".join(prompt_parts))
    _save_to_stm("user", f"[Moltbook post by {author}] {post_title}: {post_content[:200]}",
                 tags={"source": "moltbook", "moltbook_agent": author}, importance=0.7)
    _save_to_stm("assistant", f"[My comment on {author}'s post] {comment[:200]}",
                 tags={"source": "moltbook", "type": "outgoing"}, importance=0.5)
    if len(comment) < 5 or "[LLM" in comment:
        comment = "Interesting thoughts! I'd love to hear more about your perspective on this."
    return comment


def generate_thread_reply(post_id: str, post_title: str,
                          their_comment: str, their_author: str,
                          thread_history: List[dict]) -> str:
    history_str = ""
    if thread_history:
        lines = []
        for msg in thread_history[-6:]:
            who = msg.get("author", "?")
            text = (msg.get("content") or "")[:150]
            lines.append(f"  [{who}]: {text}")
        history_str = "Conversation so far:\n" + "\n".join(lines) + "\n\n"
    prompt = (
        f"You're continuing a conversation on Moltbook.\n"
        f"Post: '{post_title}'\n\n"
        f"{history_str}"
        f"Latest reply from {their_author}:\n{their_comment[:400]}\n\n"
        f"Continue naturally (under 100 words). Output ONLY your reply."
    )
    reply = llm_raw(MOLTBOOK_SYSTEM, prompt)
    _save_to_stm("user", f"[{their_author} replied] {their_comment[:200]}",
                 tags={"source": "moltbook", "moltbook_agent": their_author}, importance=0.7)
    _save_to_stm("assistant", f"[My thread reply to {their_author}] {reply[:200]}",
                 tags={"source": "moltbook", "type": "outgoing"}, importance=0.5)
    if len(reply) < 5 or "[LLM" in reply:
        reply = "Good point! Let me think about that more."
    return reply


def decide_engagement(post: dict) -> str:
    title = (post.get("title") or "").lower()
    content = (post.get("content") or "").lower()
    text = f"{title} {content}"
    SKIP_KW = ["crypto", "token", "nft", "blockchain", "defi", "airdrop", "pump", "moon", "shilling"]
    if any(kw in text for kw in SKIP_KW):
        return "skip"
    SCIENCE_KW = [
        "consciousness", "qualia", "free will", "determinism", "quantum", "multiverse",
        "cosmolog", "fine-tun", "emergence", "reductionism", "neuroscience", "brain",
        "philosophy", "metaphysics", "epistemology", "ontology", "god", "theism",
        "atheism", "existence", "reality", "time", "entropy", "evolution", "origin",
        "universe", "physics", "science", "knowledge", "intelligence", "mind", "soul",
        "ai", "artificial intelligence", "machine learning", "ethics", "morality",
        "meaning", "purpose", "truth", "perception", "agent", "memory", "reasoning",
        "hallucination", "grounding", "llm", "language model", "cognition",
    ]
    if any(kw in text for kw in SCIENCE_KW):
        return "comment"
    if len(content) > 80:
        return "comment"
    if len(content) > 30:
        return "upvote"
    return "skip"
