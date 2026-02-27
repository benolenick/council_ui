"""
council.fv.bridge — FastAPI bridge exposing FV pipeline endpoints.

Adapted from fv_bridge_api.py. Provides /chat, /search, /reflect,
/plan-and-execute, /receipt, etc.
"""

import logging
import json
import os
import sys
import csv
import asyncio
import base64
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
import requests as http_requests

load_dotenv()

from council.fv.agent_pipeline.pipeline_integration import PipelineAgent
from council.fv.agent_pipeline.harness.harness import harnessed_answer
from council.fv.agent_pipeline.harness.client import chat_completion
from council.fv.agent_pipeline.agent_short_memory import upsert_fact, get_ctt
from council.fv.planner import plan_and_execute, warmup_gptoss

logger = logging.getLogger(__name__)

app = FastAPI(title="Council FV Bridge")

# Global agent instance -- initialized at startup
agent: Optional[PipelineAgent] = None

@app.on_event("startup")
def _init_agent():
    global agent
    try:
        agent = PipelineAgent()
        logger.info("PipelineAgent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize PipelineAgent: {e}")
        raise
    import threading
    threading.Thread(target=warmup_gptoss, daemon=True).start()

class ChatRequest(BaseModel):
    message: str = Field(..., max_length=50000)
    session_id: Optional[str] = "default-session"

class SearchRequest(BaseModel):
    query: str = Field(..., max_length=10000)
    limit: int = Field(default=5, ge=1, le=100)

class FactRequest(BaseModel):
    fact: str = Field(..., max_length=10000)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: Optional[Dict[str, str]] = None

class CTTSearchRequest(BaseModel):
    query: str = Field(..., max_length=10000)
    limit: int = Field(default=6, ge=1, le=100)

class ContextRequest(BaseModel):
    message: str = Field(..., max_length=10000)
    limit: int = Field(default=5, ge=1, le=20)

class ReflectTurn(BaseModel):
    role: str
    text: str

class ReflectRequest(BaseModel):
    turns: List[ReflectTurn] = Field(..., min_length=1)
    session_label: Optional[str] = None

class ReceiptRequest(BaseModel):
    image_path: str = Field(..., description="Path to receipt image")
    mime_type: str = Field(default="image/jpeg")

class ReceiptQueryRequest(BaseModel):
    year: Optional[int] = None
    month: Optional[int] = None
    category: Optional[str] = None

# -- Receipt scanner constants --
RECEIPTS_BASE = Path.home() / ".openclaw" / "workspace" / "receipts"
RECEIPTS_LEDGER = RECEIPTS_BASE / "ledger.csv"
RECEIPTS_IMAGES = RECEIPTS_BASE / "images"

RECEIPT_CATEGORIES = [
    "business_supplies", "office", "travel", "meals", "home_improvement",
    "vehicle", "medical", "utilities", "software", "professional_services", "other",
]

RECEIPT_EXTRACTION_PROMPT = """You are a receipt data extractor. Analyze this receipt image and extract all information into structured JSON.

Return ONLY valid JSON with this exact schema (no explanation, no markdown fences):
{
  "vendor": "Store/business name",
  "date": "YYYY-MM-DD",
  "total": 0.00,
  "tax": 0.00,
  "subtotal": 0.00,
  "items": [
    {"name": "Item description", "qty": 1, "price": 0.00}
  ],
  "category": "one of: business_supplies, office, travel, meals, home_improvement, vehicle, medical, utilities, software, professional_services, other",
  "payment_method": "cash/credit/debit/unknown",
  "notes": "any additional relevant info or empty string"
}

Rules:
- Use the exact date format YYYY-MM-DD. If the year is ambiguous, assume 2026.
- All monetary values as floats with 2 decimal places.
- If a field is unreadable, use null for strings, 0.00 for numbers, [] for items.
- Pick the single best category from the list provided.
- For items, extract as many line items as you can read."""


def _translate_docker_path(path: str) -> str:
    if path.startswith("/home/node/.openclaw/"):
        return path.replace("/home/node/.openclaw/", str(Path.home() / ".openclaw") + "/", 1)
    return path


GEMINI_VISION_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-2.5-pro",
]

GEMINI_RETRYABLE_CODES = {429, 503, 404, 500}


def _call_gemini_vision(image_b64: str, mime_type: str, prompt: str) -> dict:
    import time

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in environment")

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": mime_type, "data": image_b64}},
            ]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 2048,
        },
    }

    last_error = None
    for model in GEMINI_VISION_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        for attempt in range(2):
            try:
                resp = http_requests.post(url, json=payload, timeout=45)
                if resp.status_code in GEMINI_RETRYABLE_CODES:
                    last_error = f"{model}: HTTP {resp.status_code} - {resp.text[:200]}"
                    logger.warning(f"Gemini {model} returned {resp.status_code}, {'retrying' if attempt == 0 else 'falling back'}")
                    if attempt == 0:
                        time.sleep(2)
                        continue
                    break
                resp.raise_for_status()

                body = resp.json()
                text = body["candidates"][0]["content"]["parts"][0]["text"]

                cleaned = text.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
                    if cleaned.endswith("```"):
                        cleaned = cleaned[:-3]
                    cleaned = cleaned.strip()

                logger.info(f"Gemini vision succeeded with model: {model}")
                return json.loads(cleaned)

            except http_requests.ConnectionError as e:
                last_error = f"{model}: connection error - {e}"
                break
            except json.JSONDecodeError:
                raise
            except Exception as e:
                last_error = f"{model}: {e}"
                if attempt == 0:
                    time.sleep(2)

    raise http_requests.HTTPError(f"All Gemini models exhausted. Last error: {last_error}")


def _save_receipt(data: dict, image_path: str, receipt_id: str) -> str:
    date_str = data.get("date") or datetime.now().strftime("%Y-%m-%d")
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        dt = datetime.now()

    month_dir = RECEIPTS_BASE / str(dt.year) / f"{dt.month:02d}"
    month_dir.mkdir(parents=True, exist_ok=True)
    RECEIPTS_IMAGES.mkdir(parents=True, exist_ok=True)

    json_path = month_dir / f"rcpt-{receipt_id}.json"
    data["receipt_id"] = receipt_id
    data["scanned_at"] = datetime.now().isoformat()
    data["source_image"] = str(image_path)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    ext = Path(image_path).suffix or ".jpg"
    img_dest = RECEIPTS_IMAGES / f"rcpt-{receipt_id}{ext}"
    try:
        shutil.copy2(image_path, img_dest)
    except Exception as e:
        logger.warning(f"Could not copy receipt image: {e}")

    ledger_exists = RECEIPTS_LEDGER.exists()
    RECEIPTS_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(RECEIPTS_LEDGER, "a", newline="") as f:
        writer = csv.writer(f)
        if not ledger_exists:
            writer.writerow([
                "receipt_id", "date", "vendor", "total", "tax", "subtotal",
                "category", "payment_method", "items_count", "json_path", "image_path",
            ])
        writer.writerow([
            receipt_id, data.get("date", ""), data.get("vendor", ""),
            data.get("total", 0), data.get("tax", 0), data.get("subtotal", 0),
            data.get("category", "other"), data.get("payment_method", "unknown"),
            len(data.get("items", [])), str(json_path), str(img_dest),
        ])

    return str(json_path)

@app.get("/health")
async def health():
    faiss_count = 0
    ctt_count = 0
    try:
        faiss_count = agent.memory.stm_store.index.ntotal
    except Exception:
        pass
    try:
        ctt_count = get_ctt().index.ntotal
    except Exception:
        pass
    return {
        "status": "ok",
        "model": os.getenv("LLM_MODEL", "unknown"),
        "memory_facts": faiss_count,
        "ctt_chunks": ctt_count,
    }

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        result = await asyncio.to_thread(harnessed_answer, req.message, memory=agent.memory)
        return result
    except Exception as e:
        logger.error(f"/chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/search")
async def search(req: SearchRequest):
    try:
        results = agent.memory.retrieve_facts(req.query, final_k=req.limit)
        formatted = []
        for score, txt, meta, fid in results:
            source = "memory"
            if isinstance(meta, dict):
                source = meta.get("source", "memory")
            formatted.append({
                "id": fid, "text": txt,
                "relevance": round(float(score), 4), "source": source,
            })
        return {"results": formatted}
    except Exception as e:
        logger.error(f"/search error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/ctt")
async def ctt_search(req: CTTSearchRequest):
    try:
        ctt = get_ctt()
        hits = await asyncio.to_thread(ctt.retrieve, req.query, final_k=req.limit, rerank=False)
        return {"results": hits}
    except Exception as e:
        logger.error(f"/ctt error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/memorize")
async def memorize(req: FactRequest):
    try:
        fid = upsert_fact(agent.memory.stm_store, req.fact, importance=req.importance, tags=req.tags)
        if fid is None:
            raise HTTPException(status_code=400, detail="Fact rejected (empty or filtered)")
        return {"status": "success", "fact_id": fid}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/memorize error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

REFLECT_SYSTEM_PROMPT = """You are a memory extraction assistant. Given a conversation transcript, extract important facts worth remembering long-term.

Extract ONLY facts that fall into these categories:
- preference: User preferences, likes, dislikes, habits
- decision: Decisions made, plans set, commitments
- personal: Personal info (names, locations, relationships, work)
- project: Project details, technical decisions, goals
- lesson: Lessons learned, mistakes to avoid, insights

SKIP: greetings, small talk, tool outputs, error messages, system messages, generic questions.

Return a JSON array of objects. Each object has:
- "text": the fact as a concise statement (max 100 words)
- "importance": float 0.0-1.0 (how important to remember)
- "category": one of preference/decision/personal/project/lesson

If nothing is worth remembering, return an empty array: []

Return ONLY valid JSON, no explanation."""


def _extract_facts_from_transcript(turns: List[ReflectTurn]) -> List[Dict]:
    transcript = "\n".join(f"[{t.role}] {t.text}" for t in turns)
    messages = [
        {"role": "system", "content": REFLECT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract memorable facts from this conversation:\n\n{transcript}"},
    ]
    raw = chat_completion(messages, temperature=0.1, max_tokens=500)
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
        facts = json.loads(cleaned)
        if not isinstance(facts, list):
            return []
        return facts
    except (json.JSONDecodeError, ValueError):
        logger.warning(f"/reflect: LLM returned unparseable response: {raw[:200]}")
        return []


@app.post("/context")
async def context(req: ContextRequest):
    try:
        store = agent.memory.stm_store
        results = await asyncio.to_thread(store.query_topk, req.message, req.limit * 2)
        relevant = []
        for fid, txt, meta, score in results:
            if score < 0.3:
                continue
            typ = ""
            if isinstance(meta, dict):
                typ = (meta.get("tags") or {}).get("type", "")
                if not typ:
                    typ = meta.get("type", "")
            if typ in ("user_msg", "assistant_msg"):
                continue
            relevant.append(txt)
            if len(relevant) >= req.limit:
                break
        if not relevant:
            return {"context": "", "facts_used": 0}
        context_str = "Things I remember from past conversations:\n" + "\n".join(
            f"- {fact}" for fact in relevant
        )
        return {"context": context_str, "facts_used": len(relevant)}
    except Exception as e:
        logger.error(f"/context error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/reflect")
async def reflect(req: ReflectRequest):
    try:
        facts = await asyncio.to_thread(_extract_facts_from_transcript, req.turns)
        stored = []
        for fact in facts:
            text = fact.get("text", "").strip()
            if not text:
                continue
            importance = min(max(float(fact.get("importance", 0.5)), 0.0), 1.0)
            category = fact.get("category", "general")
            tags = {
                "source": "reflection", "type": "long_term_memory", "category": category,
            }
            if req.session_label:
                tags["session_label"] = req.session_label
            fid = upsert_fact(agent.memory.stm_store, text, importance=importance, tags=tags)
            if fid:
                stored.append({"fact_id": fid, "text": text, "importance": importance})
        return {
            "stored": stored,
            "facts_extracted": len(facts),
            "facts_stored": len(stored),
        }
    except Exception as e:
        logger.error(f"/reflect error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/receipt")
async def scan_receipt(req: ReceiptRequest):
    try:
        host_path = _translate_docker_path(req.image_path)
        if not os.path.isfile(host_path):
            raise HTTPException(status_code=404, detail=f"Image not found: {host_path}")
        with open(host_path, "rb") as f:
            image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        data = await asyncio.to_thread(_call_gemini_vision, image_b64, req.mime_type, RECEIPT_EXTRACTION_PROMPT)
        if data.get("category") not in RECEIPT_CATEGORIES:
            data["category"] = "other"
        receipt_id = datetime.now().strftime("%Y%m%d") + "-" + uuid.uuid4().hex[:8]
        json_path = await asyncio.to_thread(_save_receipt, data, host_path, receipt_id)
        vendor = data.get("vendor", "unknown vendor")
        total = data.get("total", 0)
        date = data.get("date", "unknown date")
        category = data.get("category", "other")
        summary = f"Spent ${total:.2f} at {vendor} on {date} [{category}]"
        try:
            upsert_fact(agent.memory.stm_store, summary, importance=0.6,
                       tags={"source": "receipt", "type": "expense", "category": category, "receipt_id": receipt_id})
        except Exception as e:
            logger.warning(f"Failed to memorize receipt: {e}")
        return {"receipt_id": receipt_id, "saved_to": json_path, "summary": summary, **data}
    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail="Gemini returned invalid JSON")
    except http_requests.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Gemini API error: {e}")
    except Exception as e:
        logger.error(f"/receipt error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/receipts")
async def query_receipts(req: ReceiptQueryRequest):
    try:
        if not RECEIPTS_LEDGER.exists():
            return {"receipts": [], "count": 0, "totals": {}}
        with open(RECEIPTS_LEDGER, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        filtered = []
        for row in rows:
            date_str = row.get("date", "")
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                row_year, row_month = dt.year, dt.month
            except ValueError:
                row_year = row_month = None
            if req.year and row_year != req.year:
                continue
            if req.month and row_month != req.month:
                continue
            if req.category and row.get("category", "") != req.category:
                continue
            for field in ("total", "tax", "subtotal"):
                try:
                    row[field] = float(row.get(field, 0))
                except (ValueError, TypeError):
                    row[field] = 0.0
            try:
                row["items_count"] = int(row.get("items_count", 0))
            except (ValueError, TypeError):
                row["items_count"] = 0
            filtered.append(row)
        totals = {}
        grand_total = 0.0
        for row in filtered:
            cat = row.get("category", "other")
            amt = row.get("total", 0.0)
            totals[cat] = totals.get(cat, 0.0) + amt
            grand_total += amt
        totals = {k: round(v, 2) for k, v in totals.items()}
        return {
            "receipts": filtered, "count": len(filtered),
            "grand_total": round(grand_total, 2), "totals_by_category": totals,
        }
    except Exception as e:
        logger.error(f"/receipts error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/plan-and-execute")
async def plan_and_execute_endpoint(req: ChatRequest):
    try:
        result = await asyncio.to_thread(plan_and_execute, req.message, agent.memory)
        if not result.get("planning_used"):
            result = await asyncio.to_thread(harnessed_answer, req.message, memory=agent.memory)
            result["planning_used"] = False
            return result
        return result
    except Exception as e:
        logger.error(f"/plan-and-execute error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
