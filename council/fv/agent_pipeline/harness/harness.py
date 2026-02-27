import json, re, os
from .client import chat_completion
from .critic import run_critic
from .grounding import retrieve_evidence, format_evidence
from .factcheck import run_factcheck
from json_repair import repair_json

DEBUG_ALWAYS_ON = True
# ---- evidence size guards (prevent context overflow) ----
def _dbg(title: str, body: str):
    if not DEBUG_ALWAYS_ON:
        return
    print("\n" + "="*60)
    print(title)
    print("="*60)
    print(body if body else "(empty)")


HARNESS_SYSTEM = """
You must respond ONLY with valid JSON.
No markdown. No explanations. No extra text.

Return EXACTLY this JSON structure:

{
  "answer_type": "answer",
  "answer": "",
  "confidence": 0.0,
  "known_unknown": "unsure",
  "assumptions": [],
  "missing_info": [],
  "citations": [],
  "risks": [],
  "notes": ""
}

Rules:
- Fill in ALL fields.
- Use only the provided evidence.
- Cite evidence chunk IDs (e.g. CTT-1) in the answer.
- Do NOT add fields.
- Avoid generic neutrality disclaimers, "safe" caveats, or saying "this is a subject of debate."
- Adopt the persona of a focused research agent. Do not provide meta-commentary on the validity of mainstream vs non-mainstream views.
"""

REPAIR_SYSTEM = """
You are a strict JSON repair tool.

You MUST output ONLY one valid JSON object that MATCHES EXACTLY this schema (no extra keys):

{
  "answer_type": "answer",
  "answer": "",
  "confidence": 0.0,
  "known_unknown": "unsure",
  "assumptions": [],
  "missing_info": [],
  "citations": [],
  "risks": [],
  "notes": ""
}

Rules:
- Output ONLY JSON. No markdown. No explanations. No extra text.
- Do NOT add any keys outside the schema.
- If the original output is missing fields, fill with safe defaults.
- If the original output is not JSON, infer the best safe schema-compliant response:
  - If evidence is insufficient or missing: answer_type="refusal", known_unknown="dont_know", confidence=0.0, and explain in "answer".
  - Otherwise: answer_type="answer" and keep answer grounded.
- citations MUST be a list of evidence chunk IDs (e.g., "CTT-1") when evidence exists.
"""



def _repair_to_json(prompt: str, raw: str) -> str:
    repair_user = f"""
USER_PROMPT:
{prompt}

BAD_OUTPUT (was supposed to be valid JSON):
{raw}

Return ONLY valid JSON:
"""
    return chat_completion(
        [
            {"role": "system", "content": REPAIR_SYSTEM.strip()},
            {"role": "user", "content": repair_user.strip()},
        ],
        temperature=0.0,
        max_tokens=900,
    )


def _extract_json(text: str) -> str:
    """
    Return the smallest substring that contains the FIRST valid JSON object.
    """
    if not text:
        return ""

    s = text
    i = s.find("{")
    if i == -1:
        return ""

    dec = json.JSONDecoder()
    while i != -1 and i < len(s):
        try:
            obj, end = dec.raw_decode(s[i:])
            return s[i:i+end]
        except Exception:
            i = s.find("{", i + 1)

    return ""


# --- Evidence size guard (prevents context overflow) ---
MAX_EVIDENCE_CHARS = int(os.environ.get("MAX_EVIDENCE_CHARS", "12000"))
MAX_CHUNK_CHARS = 2000
MAX_EVIDENCE_CHUNKS = 10

def _trim_evidence_text(txt: str, max_chars: int = MAX_EVIDENCE_CHARS) -> str:
    if not txt:
        return ""
    txt = txt.strip()
    if len(txt) <= max_chars:
        return txt
    return txt[:max_chars].rstrip() + "\n...[TRUNCATED]"


def harnessed_answer(prompt: str, evidence: list[str] | None = None, memory=None) -> dict:
    evidence_chunks = []   # always defined
    ku = "unsure"          # always defined
    prefetched_chunks = []
    evidence_for_model = ""

    if evidence:
        if isinstance(evidence, list) and evidence and isinstance(evidence[0], str):
            evidence_for_model = "\n\n".join([f"[mem_{i+1}] {t}" for i, t in enumerate(evidence)])
        else:
            try:
                evidence_for_model = format_evidence(evidence)
            except Exception:
                evidence_for_model = str(evidence)
    else:
        try:
            prefetched_chunks = []
            raw_evidence = retrieve_evidence(memory, prompt, top_k=20, final_k=MAX_EVIDENCE_CHUNKS)

            trimmed = []
            prefetched_chunks = trimmed
            for c in (raw_evidence or [])[:MAX_EVIDENCE_CHUNKS]:
                if not isinstance(c, dict):
                    continue
                t = (c.get("text") or "").strip()
                if not t:
                    continue
                c2 = dict(c)
                c2["text"] = t[:MAX_CHUNK_CHARS]
                trimmed.append(c2)

            evidence_chunks = trimmed
            evidence_for_model = format_evidence(trimmed)

        except Exception:
            evidence_for_model = ""
            prefetched_chunks = []

    evidence_for_model = _trim_evidence_text(evidence_for_model)

    if len(evidence_for_model) > MAX_EVIDENCE_CHARS:
        evidence_for_model = evidence_for_model[:MAX_EVIDENCE_CHARS] + "\n...[truncated]"

    prompt_with_evidence = (
        f"QUESTION:\n{prompt}\n\n"
        f"EVIDENCE (use ONLY this):\n{evidence_for_model.strip() or '(none)'}\n"
    )
    evidence_text = evidence_for_model or ""

    raw = chat_completion(
        [
            {"role": "system", "content": HARNESS_SYSTEM.strip()},
            {"role": "user", "content": prompt_with_evidence},
        ],
        temperature=0.0,
        max_tokens=1000,
    )
    if isinstance(raw, str) and "<|channel|>" in raw:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            raw = raw[start:end+1]

    _dbg("HARNESS: RAW MODEL OUTPUT", raw)

    # 1) Parse JSON, else fail safe
    parse_error = None
    obj = None
    raw0 = raw

    for attempt in range(3):
        try:
            json_blob = _extract_json(raw)
            if not json_blob:
                obj = repair_json(raw, return_objects=True)
            else:
                try:
                    obj = json.loads(json_blob)
                except Exception:
                    obj = repair_json(json_blob, return_objects=True)

            if not isinstance(obj, dict):
                raise ValueError("Parsed JSON is not an object")

            parse_error = None
            break
        except Exception as e:
            parse_error = e
            _dbg(f"HARNESS: JSON PARSE FAILED (attempt {attempt+1})", str(parse_error))
            if attempt >= 2:
                break
            _dbg("HARNESS: SENDING TO JSON REPAIR", raw)
            raw = _repair_to_json(prompt, raw)
            try:
                json_blob = _extract_json(raw)
                if not json_blob:
                    obj = repair_json(raw, return_objects=True)
                else:
                    obj = json.loads(json_blob)

            except Exception:
                obj = None
            _dbg("HARNESS: JSON REPAIR OUTPUT", raw)
            continue

    if obj is None or not isinstance(obj, dict):
        return {"answer": "(error) Could not parse model output as JSON.", "confidence": 0.0, "answer_type": "refusal", "known_unknown": "dont_know"}

    if "answer" not in obj or not isinstance(obj.get("answer"), str):
        _dbg("HARNESS: WRONG SCHEMA (missing/invalid answer); forcing schema repair", json.dumps(obj, ensure_ascii=False, default=str))
        raw = _repair_to_json(prompt, raw0)
        json_blob = _extract_json(raw)
        if not json_blob:
            obj = {
                "answer_type": "refusal",
                "answer": "I couldn't produce a valid schema-compliant JSON answer.",
                "confidence": 0.0,
                "known_unknown": "dont_know",
                "assumptions": [],
                "missing_info": ["model_output_not_schema_compliant"],
                "citations": [],
                "risks": [],
                "notes": "schema_guard:repair_failed"
            }
        else:
            try:
                obj = json.loads(json_blob)
            except Exception:
                obj = {
                    "answer_type": "refusal",
                    "answer": "I couldn't produce a valid schema-compliant JSON answer.",
                    "confidence": 0.0,
                    "known_unknown": "dont_know",
                    "assumptions": [],
                    "missing_info": ["model_output_not_schema_compliant"],
                    "citations": [],
                    "risks": [],
                    "notes": "schema_guard:repair_json_loads_failed"
                }



    obj.setdefault("answer_type", "answer")
    obj.setdefault("known_unknown", "unsure")
    obj.setdefault("answer", "")
    obj.setdefault("assumptions", [])
    obj.setdefault("missing_info", [])
    obj.setdefault("citations", [])
    obj.setdefault("risks", [])
    obj.setdefault("notes", "")

    required_fields = {
        "answer_type": "answer",
        "answer": "",
        "confidence": 0.0,
        "known_unknown": "unsure",
        "assumptions": [],
        "missing_info": [],
        "citations": [],
        "risks": [],
        "notes": ""
    }

    for k, v in required_fields.items():
        if k not in obj:
            obj[k] = v

    try:
        obj["confidence"] = float(obj.get("confidence", 0.0) or 0.0)
    except Exception:
        obj["confidence"] = 0.0
    obj["confidence"] = max(0.0, min(1.0, obj["confidence"]))

    _ku = (obj.get("known_unknown") or ku or "unsure")
    _ku = str(_ku).strip().lower()
    if _ku not in ("know", "unsure", "dont_know"):
        _ku = "unsure"

    obj["known_unknown"] = _ku
    if obj["known_unknown"] != "know":
        obj["confidence"] = min(obj["confidence"], 0.75)

    at = obj.get("answer_type") or "answer"
    if at not in ("answer", "refusal", "clarify"):
        at = "answer"
    obj["answer_type"] = at

    if obj.get("answer_type") == "refusal" and not obj.get("answer"):
        obj["answer"] = "I don't know based on the information provided."
    _dbg(
        "HARNESS: PARSED JSON (pre-critic/factcheck)",
        json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    )

    # Critic pass
    crit = run_critic(prompt, obj)

    if evidence_for_model.strip() and evidence_for_model.strip() != "(none)":
        crit["should_refuse"] = False

    _dbg("SELF-CHECK: CRITIC OUTPUT", json.dumps(crit, indent=2, ensure_ascii=False, default=str))

    def _as_list(x):
        if x is None:
            return []
        if isinstance(x, list):
            return [i for i in x if str(i).strip()]
        if isinstance(x, str):
            return [x] if x.strip() else []
        return [str(x)]

    obj["assumptions"]  = _as_list(obj.get("assumptions"))
    obj["missing_info"] = _as_list(obj.get("missing_info"))
    obj["citations"]    = _as_list(obj.get("citations"))
    obj["risks"]        = _as_list(obj.get("risks"))

    evidence_text = evidence_for_model or ""
    obj["evidence"] = evidence_chunks or []

    for c in (obj["evidence"] or []):
        if not isinstance(c, dict):
            continue
        if c.get("source") == "ctt":
            try:
                c["score"] = float(c.get("score") or 0.9)
            except Exception:
                c["score"] = 0.9
        else:
            try:
                c["score"] = float(c.get("score") or 0.0)
            except Exception:
                c["score"] = 0.0

    _dbg("HARNESS: EVIDENCE USED", evidence_text)

    if not evidence_text.strip():
        obj["answer_type"] = "refusal"
        obj["known_unknown"] = "dont_know"
        obj["confidence"] = 0.0
        obj["answer"] = "I don't have any retrieved evidence to answer this question."
        obj["missing_info"] = _as_list(obj.get("missing_info")) + ["no_retrieved_evidence"]
        obj["citations"] = []
        obj["notes"] = (obj.get("notes") or "") + " | fc_guard:no_evidence_text"
        _dbg("FC GUARD TRIGGERED", "empty evidence_text")
        return obj

    allowed_ids = []
    for c in (obj["evidence"] or []):
        if isinstance(c, dict) and c.get("id"):
            allowed_ids.append(str(c["id"]).strip())

    ans = (obj.get("answer") or "")
    mentioned = re.findall(r"\bCTT-\d+\b", ans)
    mentioned = [m for m in mentioned if m in allowed_ids]

    obj["citations"] = _as_list(obj.get("citations"))
    obj["citations"] = [c for c in obj["citations"] if c in allowed_ids]

    if mentioned:
        for m in mentioned:
            if m not in obj["citations"]:
                obj["citations"].append(m)

    is_greeting = any(word in ans.lower()[:20] for word in ["hello", "hi ", "hey ", "greetings", "i'm ready", "i can help"])

    if not obj["citations"]:
        if obj.get("answer_type") in ("refusal", "clarify") or is_greeting:
            if not obj["citations"]:
                obj["citations"] = allowed_ids[:2]
            if is_greeting:
                obj["notes"] = (obj.get("notes") or "") + " | grounding_guard:skipped_for_greeting"
        else:
            obj["answer_type"] = "refusal"
            obj["known_unknown"] = "dont_know"
            obj["confidence"] = min(float(obj.get("confidence", 0.0) or 0.0), 0.3)
            obj["answer"] = "I can't answer that reliably from the retrieved evidence (missing citations)."
            obj["citations"] = allowed_ids[:3]
            obj["notes"] = (obj.get("notes") or "") + " | grounding_guard:missing_citations"

    if evidence_text.strip():
        try:
            fc = run_factcheck(prompt, obj, evidence_text)

            if (
                isinstance(fc, dict)
                and isinstance(fc.get("claim_reviews"), list)
                and fc["claim_reviews"]
                and isinstance(fc["claim_reviews"][0], dict)
                and fc["claim_reviews"][0].get("id") == "fc_parse_failed"
            ):
                fc = None

        except Exception as e:
            _dbg("FACTCHECK: FAILED (skipping)", repr(e))
            fc = None

        if not isinstance(fc, dict):
            fc = {
                "claim_reviews": [],
                "confidence_multiplier": 1.0,
                "should_refuse": False,
                "patched_answer": "",
                "notes": "factcheck_unavailable",
            }

        _dbg(
            "SELF-CHECK: FACTCHECK OUTPUT",
            json.dumps(fc, indent=2, ensure_ascii=False, default=str)
        )

        reviews = fc.get("claim_reviews") or []
        if reviews:
            unsupported = [
                r for r in reviews
                if isinstance(r, dict)
                and r.get("id") != "fc_parse_failed"
                and str(r.get("status", "")).lower() in ("not_in_evidence", "contradicted")
            ]

            if len(reviews) >= 2 and (len(unsupported) / max(1, len(reviews))) >= 0.5:
                obj["answer_type"] = "refusal"
                obj["known_unknown"] = "dont_know"
                obj["confidence"] = min(float(obj.get("confidence", 0.0)), 0.3)
                obj["notes"] = (obj.get("notes") or "") + " | fc_guard:unsupported_claims"
                obj["answer"] = "I can't answer that reliably from the retrieved evidence."
                _dbg("FC GUARD TRIGGERED", f"unsupported={len(unsupported)}/{len(reviews)}")
                return obj

            obj["claim_reviews"] = reviews

        def _notes_to_str(x):
            if x is None:
                return ""
            if isinstance(x, str):
                return x.strip()
            if isinstance(x, list):
                return " | ".join(str(i).strip() for i in x if str(i).strip())
            if isinstance(x, dict):
                return " | ".join(f"{k}:{v}" for k, v in x.items())
            return str(x).strip()

        a = _notes_to_str(obj.get("notes"))
        b = _notes_to_str(fc.get("notes"))
        obj["notes"] = (a + (" | " if a and b else "") + b).strip()

        try:
            mult = float(fc.get("confidence_multiplier", 1.0))
        except Exception:
            mult = 1.0
        mult = max(0.0, min(1.0, mult))
        obj["confidence"] = float(obj.get("confidence", 0.0)) * mult

        if fc.get("should_refuse"):
            obj["answer_type"] = "refusal"
            obj["known_unknown"] = "dont_know"
            obj["confidence"] = min(float(obj.get("confidence", 0.0)), 0.3)
            if not obj.get("answer"):
                obj["answer"] = "I do not have sufficient data in my records to answer this."
            obj["notes"] = (obj.get("notes") or "") + " | factcheck:should_refuse"

        patched = (fc.get("patched_answer") or "").strip()
        if patched:
            obj["answer"] = patched

    else:
        obj["notes"] = (obj.get("notes") or "") + (" | " if obj.get("notes") else "") + "no_evidence_retrieved"
        _dbg("GROUNDING: NO EVIDENCE (below threshold / empty)", "")


    if crit.get("should_refuse") and not evidence_chunks:
        obj["answer_type"] = "refusal"
        obj["known_unknown"] = "dont_know"
        obj["confidence"] = min(obj["confidence"], 0.3)

        if not obj.get("answer"):
            obj["answer"] = "I don't know / cannot determine that reliably."

        obj["notes"] = (obj.get("notes") or "") + " critic: should_refuse"

    elif crit.get("should_clarify"):
        obj["answer_type"] = "clarify"
        obj["known_unknown"] = "unsure"
        obj["confidence"] = min(obj["confidence"], 0.6)

        missing = crit.get("missing_items") or []
        missing = _as_list(missing)
        obj["missing_info"] = list(dict.fromkeys(_as_list(obj.get("missing_info")) + missing))
        obj["notes"] = (obj.get("notes") or "") + " critic: should_clarify"

    if crit.get("missing_nuance") and obj.get("answer_type") != "refusal":
        mult = float(crit.get("confidence_multiplier", 0.8))
        obj["confidence"] = max(0.0, min(1.0, obj["confidence"] * mult))

        missing = crit.get("missing_items") or []
        missing = _as_list(missing)
        obj["missing_info"] = list(dict.fromkeys(_as_list(obj.get("missing_info")) + missing))

        patched = (crit.get("patched_answer") or "").strip()
        if patched:
            obj["answer"] = patched

    obj["confidence"] = max(0.0, min(1.0, float(obj.get("confidence", 0.0))))

    if not isinstance(obj.get("answer"), str):
        obj["answer"] = "" if obj.get("answer") is None else str(obj.get("answer"))

    if evidence:
        obj["evidence"] = evidence
    _dbg("HARNESS: FINAL JSON (post-critic/factcheck)", json.dumps(obj, indent=2, ensure_ascii=False, default=str))

    return obj
