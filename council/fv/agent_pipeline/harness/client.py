import os, requests, json, time
from typing import List, Dict

BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
API_KEY  = os.environ.get("LLM_API_KEY", "none")
MODEL    = os.environ.get("LLM_MODEL", "qwen3:14b")

# Retry config
MAX_RETRIES = 3
RETRY_BACKOFF = [2, 5, 10]  # seconds between retries
CONNECT_TIMEOUT = 10   # seconds to establish connection
READ_TIMEOUT = 120     # seconds to wait for response


def _is_ollama_alive() -> bool:
    """Quick health check -- does the LLM backend respond?"""
    try:
        base = BASE_URL.rsplit("/v1", 1)[0]
        r = requests.get(base, timeout=3)
        return r.status_code < 500
    except Exception:
        return False


def chat_completion(messages: List[Dict], temperature=0.2, max_tokens=800) -> str:
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(
                f"{BASE_URL}/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            )
            if r.status_code >= 500:
                last_err = f"LLM HTTP {r.status_code}: {r.text[:500]}"
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF)-1)])
                    continue
                break

            if r.status_code >= 400:
                # Client error (bad model name, etc.) -- don't retry
                return json.dumps({"error": f"LLM HTTP {r.status_code}: {r.text[:200]}"})

            try:
                return r.json()["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as e:
                return json.dumps({"error": f"Unexpected LLM response format: {e}"})

        except requests.exceptions.ConnectionError as e:
            last_err = f"Connection refused ({e})"
            if attempt < MAX_RETRIES - 1:
                print(f"[LLM] Connection failed (attempt {attempt+1}/{MAX_RETRIES}), retrying in {RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF)-1)]}s...")
                time.sleep(RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF)-1)])
            continue

        except requests.exceptions.Timeout as e:
            last_err = f"Timeout ({e})"
            if attempt < MAX_RETRIES - 1:
                print(f"[LLM] Timeout (attempt {attempt+1}/{MAX_RETRIES}), retrying...")
                time.sleep(RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF)-1)])
            continue

        except Exception as e:
            last_err = str(e)
            break

    # All retries exhausted -- return a parseable error instead of crashing
    print(f"[LLM] All {MAX_RETRIES} attempts failed: {last_err}")
    return json.dumps({"error": str(last_err), "answer_type": "refusal", "answer": "LLM backend unavailable.", "confidence": 0.0})
