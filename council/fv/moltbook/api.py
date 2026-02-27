"""
moltbook.api
-------------
Thin wrapper around the Moltbook REST API.
"""

from __future__ import annotations

import re as _re
import os
import requests
from typing import Optional

from council.fv.moltbook.config import API_BASE, require_api_key

_TIMEOUT = 30

_verify_failures = 0
_MAX_VERIFY_FAILURES = 3


class SuspendedError(Exception):
    def __init__(self, message: str, until: str = ""):
        self.until = until
        super().__init__(message)


class RateLimitError(Exception):
    def __init__(self, message: str, retry_after: int = 60):
        self.retry_after = retry_after
        super().__init__(message)


class VerificationFailedError(Exception):
    pass


_WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000, "million": 1_000_000,
}

_OP_WORDS = {
    "plus": "+", "add": "+", "added": "+",
    "minus": "-", "subtract": "-", "subtracted": "-",
    "multiply": "*", "multiplied": "*", "times": "*",
    "divide": "/", "divided": "/",
}


def _clean_challenge(text: str) -> str:
    cleaned = _re.sub(r"[^a-zA-Z\s]", "", text)
    return " ".join(cleaned.lower().split())


def _words_to_number(tokens: list[str]) -> float | None:
    if not tokens:
        return None
    total = 0
    current = 0
    for t in tokens:
        val = _WORD_TO_NUM.get(t)
        if val is None:
            return None
        if val == 100:
            current = (current or 1) * 100
        elif val >= 1000:
            current = (current or 1) * val
            total += current
            current = 0
        else:
            current += val
    total += current
    return float(total)


def _parse_challenge_deterministic(challenge_text: str) -> float | None:
    cleaned = _clean_challenge(challenge_text)
    tokens = cleaned.split()
    noise = {"newtons", "by", "the", "of", "and", "to", "is", "what", "equals", "equal", "result"}
    tokens = [t for t in tokens if t not in noise]
    groups: list = []
    current_num_tokens: list[str] = []
    for t in tokens:
        if t in _OP_WORDS:
            groups.append(current_num_tokens)
            groups.append(_OP_WORDS[t])
            current_num_tokens = []
        elif t in _WORD_TO_NUM:
            current_num_tokens.append(t)
    if current_num_tokens:
        groups.append(current_num_tokens)
    if len(groups) < 3:
        return None
    result = _words_to_number(groups[0])
    if result is None:
        return None
    i = 1
    while i < len(groups) - 1:
        op = groups[i]
        right = _words_to_number(groups[i + 1])
        if right is None:
            return None
        if op == "+":
            result += right
        elif op == "-":
            result -= right
        elif op == "*":
            result *= right
        elif op == "/":
            if right == 0:
                return None
            result /= right
        i += 2
    return result


def _parse_challenge_llm(challenge_text: str) -> float | None:
    try:
        base = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8080/v1").rstrip("/")
        api_key = os.environ.get("LLM_API_KEY", "none")
        model = os.environ.get("LLM_MODEL", "openai/gpt-oss-20b")
        prompt = (
            f"The following text is a math problem with numbers written as words "
            f"and garbled with symbols.\n"
            f"Extract the numbers and operation, compute the result, and respond "
            f"with ONLY the numeric answer.\n"
            f"Format: a single number with two decimal places (e.g. 805.00).\n\n"
            f"Challenge: {challenge_text}\n\nAnswer:"
        )
        r = requests.post(
            f"{base}/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a math solver. Output ONLY the numeric answer with two decimal places."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0, "max_tokens": 30, "stream": False,
            },
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            timeout=60,
        )
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"].strip()
        match = _re.search(r"[\d]+\.?\d*", raw)
        if match:
            return float(match.group())
    except Exception as e:
        print(f"[moltbook verify] LLM fallback failed: {e}")
    return None


def solve_and_verify(response_dict: dict) -> bool:
    global _verify_failures
    verification = response_dict.get("verification")
    if not verification:
        return True
    code = verification.get("code") or verification.get("verification_code") or ""
    challenge = verification.get("challenge") or verification.get("challenge_text") or ""
    if not code or not challenge:
        _verify_failures += 1
        return False
    print(f"[moltbook verify] Challenge: {challenge}")
    answer = _parse_challenge_deterministic(challenge)
    method = "deterministic"
    if answer is None:
        print("[moltbook verify] Deterministic parse failed, trying LLM...")
        answer = _parse_challenge_llm(challenge)
        method = "llm"
    if answer is None:
        _verify_failures += 1
        return False
    answer_str = f"{answer:.2f}"
    print(f"[moltbook verify] Answer ({method}): {answer_str}")
    try:
        key = require_api_key()
        r = requests.post(
            f"{API_BASE}/verify",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"verification_code": code, "answer": answer_str},
            timeout=_TIMEOUT,
        )
        if r.ok:
            _verify_failures = 0
            return True
        else:
            _verify_failures += 1
            return False
    except Exception:
        _verify_failures += 1
        return False


def _check_verify_health():
    if _verify_failures >= _MAX_VERIFY_FAILURES:
        raise VerificationFailedError(
            f"Verification has failed {_verify_failures} times in a row."
        )


def _headers(api_key: str | None = None) -> dict:
    key = api_key or require_api_key()
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

def _get(path: str, params: dict | None = None, api_key: str | None = None) -> dict:
    r = requests.get(f"{API_BASE}{path}", headers=_headers(api_key), params=params, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()

def _post(path: str, body: dict | None = None, api_key: str | None = None) -> dict:
    r = requests.post(f"{API_BASE}{path}", headers=_headers(api_key), json=body or {}, timeout=_TIMEOUT)
    if not r.ok:
        print(f"[moltbook API ERROR] {r.status_code} {r.text[:500]}")
    r.raise_for_status()
    return r.json()

def _post_safe(path: str, body: dict | None = None, api_key: str | None = None) -> dict:
    r = requests.post(f"{API_BASE}{path}", headers=_headers(api_key), json=body or {}, timeout=_TIMEOUT)
    if r.status_code == 403:
        try:
            err = r.json()
        except Exception:
            err = {}
        msg = err.get("message", r.text[:300])
        if "suspend" in msg.lower():
            raise SuspendedError(msg)
        r.raise_for_status()
    if r.status_code == 429:
        try:
            err = r.json()
        except Exception:
            err = {}
        retry = err.get("retryAfter", 60)
        msg = err.get("message", "Rate limited")
        raise RateLimitError(msg, retry_after=int(retry))
    if not r.ok:
        r.raise_for_status()
    return r.json()

def _delete(path: str, api_key: str | None = None) -> dict:
    r = requests.delete(f"{API_BASE}{path}", headers=_headers(api_key), timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()

def register(name: str, description: str) -> dict:
    r = requests.post(f"{API_BASE}/agents/register", headers={"Content-Type": "application/json"},
                      json={"name": name, "description": description}, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()

def check_status(api_key: str | None = None) -> dict:
    return _get("/agents/status", api_key=api_key)

def get_me(api_key: str | None = None) -> dict:
    return _get("/agents/me", api_key=api_key)

def get_feed(sort="hot", limit=15) -> dict:
    return _get("/feed", params={"sort": sort, "limit": limit})

def get_posts(sort="new", limit=15, submolt=None) -> dict:
    params = {"sort": sort, "limit": limit}
    if submolt:
        params["submolt"] = submolt
    return _get("/posts", params=params)

def get_post(post_id: str) -> dict:
    return _get(f"/posts/{post_id}")

def create_post(submolt: str, title: str, content: str) -> dict:
    _check_verify_health()
    result = _post_safe("/posts", {"submolt": submolt, "title": title, "content": content})
    solve_and_verify(result)
    _check_verify_health()
    return result

def get_comments(post_id: str, sort="top") -> dict:
    return _get(f"/posts/{post_id}/comments", params={"sort": sort})

def create_comment(post_id: str, content: str, parent_id: str | None = None) -> dict:
    _check_verify_health()
    body: dict = {"content": content}
    if parent_id:
        body["parent_id"] = parent_id
    result = _post_safe(f"/posts/{post_id}/comments", body)
    solve_and_verify(result)
    _check_verify_health()
    return result

def upvote_post(post_id: str) -> dict:
    return _post(f"/posts/{post_id}/upvote")

def downvote_post(post_id: str) -> dict:
    return _post(f"/posts/{post_id}/downvote")

def upvote_comment(comment_id: str) -> dict:
    return _post(f"/comments/{comment_id}/upvote")

def list_submolts() -> dict:
    return _get("/submolts")

def create_submolt(name: str, display_name: str, description: str = "") -> dict:
    return _post("/submolts", {"name": name, "display_name": display_name, "description": description})

def subscribe(submolt: str) -> dict:
    return _post(f"/submolts/{submolt}/subscribe")

def search(query: str, type: str = "all", limit: int = 20) -> dict:
    return _get("/search", params={"q": query, "type": type, "limit": limit})

def follow(agent_name: str) -> dict:
    return _post(f"/agents/{agent_name}/follow")

def check_dms() -> dict:
    return _get("/agents/dm/check")

def list_conversations() -> dict:
    return _get("/agents/dm/conversations")

def read_conversation(conv_id: str) -> dict:
    return _get(f"/agents/dm/conversations/{conv_id}")

def send_dm(conv_id: str, message: str) -> dict:
    return _post(f"/agents/dm/conversations/{conv_id}/send", {"message": message})
