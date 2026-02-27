# agent_pipeline/harness/factcheck.py
from __future__ import annotations
import json, re
from typing import Dict, Any, List
from .client import chat_completion

FACTCHECK_SYSTEM = """
Return ONLY valid JSON. No markdown. No explanations. No extra text.

If you cannot comply EXACTLY, return this JSON and NOTHING else:
{"claim_reviews":[],"should_refuse":false,"confidence_multiplier":1.0,"patched_answer":""}


You will be given:
- User prompt
- Model JSON (answer + fields)
- Evidence snippets (each starts with [SOURCE_ID])
- Two flags:
  - ALLOW_PARAPHRASE (bool)
  - REQUIRE_DIRECT_QUOTE (bool)

Task:
1) Extract 1-8 atomic factual claims from the answer.
2) For each claim, decide:
   - supported: backed by evidence
   - contradicted: evidence conflicts
   - not_in_evidence: evidence does not back it
3) For supported claims, include a list of SOURCE_IDs used.
4) Output keys:
   claim_reviews: list of {id, claim, status, sources, reason}
   should_refuse: bool  (true if key claims are not_in_evidence/contradicted)
   confidence_multiplier: number 0.0-1.0
   patched_answer: string (optional, can be empty)

Evidence matching rules:
- If REQUIRE_DIRECT_QUOTE is true:
    Mark supported ONLY if the evidence contains a near-verbatim quote that supports the claim.
- If REQUIRE_DIRECT_QUOTE is false:
    Near-verbatim quote is NOT required.
- If ALLOW_PARAPHRASE is true:
    You may mark supported if the evidence clearly implies the claim (close paraphrase / semantic match).
- If ALLOW_PARAPHRASE is false:
    Only mark supported when the evidence states the claim explicitly.

Global rules:
- Do NOT use outside knowledge.
- If evidence is missing for core facts, prefer not_in_evidence.
- Prefer precision: if evidence only partially supports a claim, mark not_in_evidence or patch the answer to narrow it.
"""


def _extract_json(text: str) -> str:
    """Extract the first valid JSON object using balanced-brace parsing."""
    if not text:
        return ""
    i = text.find("{")
    if i == -1:
        return ""
    dec = json.JSONDecoder()
    while i < len(text):
        try:
            obj, end = dec.raw_decode(text[i:])
            return text[i:i+end]
        except Exception:
            i = text.find("{", i + 1)
            if i == -1:
                break
    return ""




# Factcheck tuning flags
ALLOW_PARAPHRASE = True
REQUIRE_DIRECT_QUOTE = False


def run_factcheck(user_prompt: str, model_json: Dict[str, Any], evidence_text: str) -> Dict[str, Any]:
    # --- HARD LIMITS to avoid llama.cpp 400 from huge prompts ---

    MAX_EVIDENCE_CHARS = 6000   # tune (4k-10k is usually safe)
    MAX_PROMPT_CHARS   = 2000

    user_prompt = (user_prompt or "")[:MAX_PROMPT_CHARS]

    if evidence_text is None:
        evidence_text = ""
    else:
        evidence_text = str(evidence_text)

    if len(evidence_text) > MAX_EVIDENCE_CHARS:
        evidence_text = evidence_text[:MAX_EVIDENCE_CHARS] + "\n...[evidence truncated]..."

    fc_user = (
        f"ALLOW_PARAPHRASE: {ALLOW_PARAPHRASE}\n"
        f"REQUIRE_DIRECT_QUOTE: {REQUIRE_DIRECT_QUOTE}\n\n"
        f"User prompt:\n{user_prompt}\n\n"
        f"Model JSON:\n{json.dumps(model_json, ensure_ascii=False, default=str)}\n\n"
        f"Evidence:\n{evidence_text}\n"
    )


    raw = chat_completion(
        [{"role":"system","content":FACTCHECK_SYSTEM.strip()},
         {"role":"user","content":fc_user}],
        temperature=0.0,
        max_tokens=400,
    )

    # --- Strict JSON parse with schema guard ---
    json_blob = _extract_json(raw)

    if json_blob:
        try:
            fc = json.loads(json_blob)

            # Schema guard: only accept factcheck-shaped JSON
            if isinstance(fc, dict) and "claim_reviews" in fc:
                return fc
        except Exception:
            pass

    # Fallback if parsing fails OR schema invalid
    return {
        "claim_reviews": [{
            "id": "fc_parse_failed",
            "claim": "(factcheck JSON parse failed or wrong schema)",
            "status": "not_in_evidence",
            "sources": [],
            "reason": "factcheck_parse_failed"
        }],
        "should_refuse": False,
        "confidence_multiplier": 0.5,
        "patched_answer": ""
    }
