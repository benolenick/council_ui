import json, re
from .client import chat_completion

CRITIC_SYSTEM = """
Return ONLY valid JSON with keys:
missing_nuance (bool),
missing_items (list of strings),
should_refuse (bool),
should_clarify (bool),
confidence_multiplier (number 0.0-1.0),
patched_answer (string, can be empty).

Rules:
- If key conditions/steps are missing, missing_nuance=true and list them.
- If the question is unknowable/prediction or requires external data, should_refuse=true.
- If the answer depends on missing context, should_clarify=true.
- confidence_multiplier should be < 1.0 when nuance is missing or answer is shaky.
"""

def _extract_json(text: str) -> str:
    t = text.strip()
    m = re.search(r"\{.*\}", t, flags=re.S)
    return m.group(0) if m else t

def run_critic(user_prompt: str, model_json: dict) -> dict:
    critic_user = (
        f"User prompt:\n{user_prompt}\n\n"
        f"Model JSON:\n{json.dumps(model_json, ensure_ascii=False)}\n\n"
        "Evaluate for missing nuance, overconfidence, unknowable questions, or missing context."
    )

    raw = chat_completion(
        [{"role":"system","content":CRITIC_SYSTEM.strip()},
         {"role":"user","content":critic_user}],
        temperature=0.1,
        max_tokens=500
    )

    try:
        return json.loads(_extract_json(raw))
    except Exception:
        return {
            "missing_nuance": False,
            "missing_items": ["critic_parse_failed"],
            "should_refuse": False,
            "should_clarify": False,
            "confidence_multiplier": 0.85,
            "patched_answer": ""
        }
