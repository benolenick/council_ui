"""
planner.py -- GPT-OSS plans, Qwen executes.

Two-model orchestration: GPT-OSS 20B decomposes complex tasks into steps,
Qwen 14B executes each step in parallel, then Qwen synthesizes the results.
"""

import re
import time
import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from council.fv.agent_pipeline.harness.client import chat_completion

logger = logging.getLogger(__name__)

OLLAMA_BASE = "http://127.0.0.1:11435"  # GPU 1 dedicated instance
GPTOSS_MODEL = "gpt-oss:pinned"

# -- Keywords that suggest a task needs multi-step planning --
PLANNING_TRIGGERS = [
    "analyze", "compare", "explain how", "step by step", "plan",
    "design", "investigate", "break down", "evaluate", "what are the",
    "differences between", "pros and cons", "trade-offs", "recommend",
    "walk me through", "outline", "summarize the",
]
SIMPLE_PREFIXES = ["what is", "who is", "when", "where is", "define"]

_BOLD_CHAR_RE = re.compile(r"\*\*(.)\*\*")
_JUNK_RE = re.compile(r"[*?]{2,}|\.{4,}")


def _sanitize_step(text: str) -> str:
    text = _BOLD_CHAR_RE.sub(r"\1", text)
    text = _JUNK_RE.sub("", text)
    text = re.sub(r"\s{2,}", " ", text).strip().rstrip(".")
    return text


def warmup_gptoss():
    import time as _time
    for attempt in range(6):
        try:
            r = requests.post(
                f"{OLLAMA_BASE}/api/generate",
                json={
                    "model": GPTOSS_MODEL,
                    "prompt": "hi",
                    "stream": False,
                    "keep_alive": -1,
                    "options": {"num_predict": 1},
                },
                timeout=180,
            )
            r.raise_for_status()
            logger.info("[planner] GPT-OSS warmed up and pinned in VRAM")
            return
        except Exception as e:
            logger.warning(f"[planner] GPT-OSS warmup attempt {attempt+1}/6 failed: {e}")
            _time.sleep(10)
    logger.error("[planner] GPT-OSS warmup failed after 6 attempts")


def _call_gptoss(prompt: str, max_tokens: int = 1024) -> str:
    payload = {
        "model": GPTOSS_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": -1,
        "options": {
            "temperature": 0.3,
            "num_predict": max_tokens,
        },
    }
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        logger.error(f"GPT-OSS call failed: {e}")
        raise


def needs_planning(message: str) -> bool:
    msg = message.strip()
    low = msg.lower()
    if len(msg) < 50:
        return False
    for prefix in SIMPLE_PREFIXES:
        if low.startswith(prefix):
            return False
    if len(msg) > 150:
        return True
    return any(trigger in low for trigger in PLANNING_TRIGGERS)


def create_plan(task: str, memory_context: str) -> list[str]:
    prompt = (
        f"Task: {task}\n\n"
        f"Relevant context:\n{memory_context}\n\n"
        "Break this task into 2-3 concrete steps. Each step should be a clear, "
        "self-contained instruction. Return ONLY a numbered list, no explanation.\n\n"
        "Steps:\n1."
    )
    raw = _call_gptoss(prompt, max_tokens=512)
    raw = "1." + raw

    all_steps = []
    for line in raw.split("\n"):
        line = line.strip()
        m = re.match(r"^\d+[.)]\s*(.+)", line)
        if m:
            step_text = _sanitize_step(m.group(1))
            if len(step_text) > 25:
                all_steps.append(step_text)

    if len(all_steps) <= 3:
        return all_steps
    return all_steps[-3:]


def execute_step(step: str, task: str, memory_context: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are executing one step of a plan. Complete this step "
                "thoroughly. Be concise and factual."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original task: {task}\n\n"
                f"Your step: {step}\n\n"
                "Relevant memory:\n"
                f"{memory_context if memory_context else '(none)'}"
            ),
        },
    ]
    return chat_completion(messages, temperature=0.2, max_tokens=400)


def synthesize_results(task: str, steps: list[str], results: list[str]) -> str:
    steps_block = "\n".join(
        f"Step {i+1}: {s}\nResult: {r}" for i, (s, r) in enumerate(zip(steps, results))
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Synthesize these step results into a clear, direct answer. "
                "Do not repeat the steps -- combine insights into a coherent response."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original task: {task}\n\n"
                f"Completed steps and results:\n{steps_block}"
            ),
        },
    ]
    return chat_completion(messages, temperature=0.2, max_tokens=600)


def plan_and_execute(task: str, memory) -> dict:
    if not needs_planning(task):
        return {"planning_used": False}

    t0 = time.time()

    memory_context = ""
    try:
        facts = memory.retrieve_facts(task, final_k=5)
        if facts:
            memory_context = "\n".join(f"- {txt}" for _, txt, _, _ in facts)
    except Exception as e:
        logger.warning(f"Memory retrieval failed: {e}")

    logger.info(f"[planner] Creating plan for: {task[:80]}...")
    t_plan_start = time.time()
    steps = create_plan(task, memory_context)
    t_plan_sec = round(time.time() - t_plan_start, 2)

    if len(steps) < 2:
        logger.info("[planner] <2 steps parsed, falling back to single execution")
        return {"planning_used": False}

    logger.info(f"[planner] Plan has {len(steps)} steps ({t_plan_sec}s)")

    t_exec_start = time.time()
    results = [None] * len(steps)

    def _run_step(index: int, step: str) -> tuple[int, str]:
        logger.info(f"[planner] Executing step {index+1}/{len(steps)}: {step[:60]}...")
        return index, execute_step(step, task, memory_context)

    with ThreadPoolExecutor(max_workers=len(steps)) as pool:
        futures = {
            pool.submit(_run_step, i, step): i for i, step in enumerate(steps)
        }
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    t_exec_sec = round(time.time() - t_exec_start, 2)
    logger.info(f"[planner] All steps done ({t_exec_sec}s)")

    logger.info("[planner] Synthesizing results...")
    t_synth_start = time.time()
    answer = synthesize_results(task, steps, results)
    t_synth_sec = round(time.time() - t_synth_start, 2)

    elapsed = round(time.time() - t0, 1)
    logger.info(f"[planner] Done in {elapsed}s (plan={t_plan_sec}s exec={t_exec_sec}s synth={t_synth_sec}s)")

    return {
        "answer": answer,
        "plan_steps": steps,
        "step_results": results,
        "planning_used": True,
        "elapsed_sec": elapsed,
        "timing": {
            "plan_sec": t_plan_sec,
            "execute_sec": t_exec_sec,
            "synthesize_sec": t_synth_sec,
        },
    }
