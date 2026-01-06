# app.py
from __future__ import annotations

import os
import re
import time
import json
import sqlite3
import base64
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import (
    FastAPI,
    Header,
    HTTPException,
    Request,
    UploadFile,
    File,
)
from fastapi.responses import HTMLResponse, Response, JSONResponse
from pydantic import BaseModel

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from groq import Groq
from openai import OpenAI


# =============================================================================
# App + Config
# =============================================================================

app = FastAPI(title="AERYNX API", version="0.2.0")

limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

API_KEY = os.getenv("AERYNX_API_KEY")
DB_PATH = os.getenv("AERYNX_DB_PATH", "/home/ubuntu/aerynx-api/aerynx_memory.sqlite3")

TGI_URL = os.getenv("TGI_URL", "http://127.0.0.1:3000/generate")
TGI_HEALTH_URL = os.getenv("TGI_HEALTH_URL", "http://127.0.0.1:3000/health")
TGI_TIMEOUT = float(os.getenv("TGI_TIMEOUT", "45"))

# Base temps (these become inputs to adaptive tuning)
AERYNX_TEMP_BASE = float(os.getenv("AERYNX_TEMP", "0.55"))
GROQ_TEMP_BASE = float(os.getenv("GROQ_TEMP", "0.75"))
TOP_P = float(os.getenv("TOP_P", "0.9"))

# Memory tuning
MEMORY_EVERY_TURNS = int(os.getenv("AERYNX_MEMORY_EVERY", "5"))     # periodic check interval
RECENT_MAX_MESSAGES = int(os.getenv("AERYNX_RECENT_MAX", "8"))      # short-term window size
RECENT_MAX_CHARS = int(os.getenv("AERYNX_RECENT_MAX_CHARS", "900")) # keep recent light

# Voice tuning
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
AERYNX_TTS_VOICE_DEFAULT = os.getenv("AERYNX_TTS_VOICE_DEFAULT", "alloy")
AERYNX_TTS_VOICE_WARM = os.getenv("AERYNX_TTS_VOICE_WARM", "alloy")
AERYNX_TTS_VOICE_SOOTHING = os.getenv("AERYNX_TTS_VOICE_SOOTHING", "alloy")
AERYNX_TTS_VOICE_COACH = os.getenv("AERYNX_TTS_VOICE_COACH", "alloy")
AERYNX_TTS_VOICE_SERIOUS = os.getenv("AERYNX_TTS_VOICE_SERIOUS", "alloy")

# Optional: if your TTS supports a "speed" parameter in your SDK/version, enable it.
# (Left off by default to avoid relying on a parameter that may not exist in your installed version.)
TTS_SPEED_ENABLE = os.getenv("AERYNX_TTS_SPEED_ENABLE", "0") == "1"
TTS_SPEED_DEFAULT = float(os.getenv("AERYNX_TTS_SPEED_DEFAULT", "1.05"))  # slightly faster than "calm narrator"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# =============================================================================
# DB (SQLite) + MIGRATION
# =============================================================================

_DB: Optional[sqlite3.Connection] = None

def _connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def _ensure_schema(conn: sqlite3.Connection) -> None:
    """
    sessions:
      - summary: durable bullet memory (selective)
      - recent: lightweight short-term window (JSON list of role/content dicts)
      - turns: chat turns count
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            summary TEXT DEFAULT '',
            recent TEXT DEFAULT '[]',
            turns INTEGER DEFAULT 0,
            updated_at INTEGER DEFAULT 0
        )
        """
    )
    conn.commit()

    cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}

    if "summary" not in cols:
        conn.execute("ALTER TABLE sessions ADD COLUMN summary TEXT DEFAULT ''")
    if "recent" not in cols:
        conn.execute("ALTER TABLE sessions ADD COLUMN recent TEXT DEFAULT '[]'")
    if "turns" not in cols:
        conn.execute("ALTER TABLE sessions ADD COLUMN turns INTEGER DEFAULT 0")
    if "updated_at" not in cols:
        conn.execute("ALTER TABLE sessions ADD COLUMN updated_at INTEGER DEFAULT 0")

    conn.commit()

@app.on_event("startup")
def startup() -> None:
    global _DB
    _DB = _connect_db()
    _ensure_schema(_DB)

@app.on_event("shutdown")
def shutdown() -> None:
    global _DB
    try:
        if _DB:
            _DB.close()
    finally:
        _DB = None


def get_session(session_id: str) -> Tuple[str, List[Dict[str, str]], int]:
    if not _DB:
        raise RuntimeError("DB not initialized")

    cur = _DB.execute(
        "SELECT summary, recent, turns FROM sessions WHERE session_id=?",
        (session_id,),
    )
    row = cur.fetchone()

    if not row:
        _DB.execute(
            "INSERT INTO sessions(session_id, summary, recent, turns, updated_at) VALUES(?,?,?,?,?)",
            (session_id, "", "[]", 0, int(time.time())),
        )
        _DB.commit()
        return "", [], 0

    summary = row[0] or ""
    recent_raw = row[1] or "[]"
    turns = int(row[2] or 0)

    try:
        recent = json.loads(recent_raw)
        if not isinstance(recent, list):
            recent = []
    except Exception:
        recent = []

    # sanitize shape
    cleaned: List[Dict[str, str]] = []
    for m in recent[-RECENT_MAX_MESSAGES:]:
        if isinstance(m, dict) and "role" in m and "content" in m:
            cleaned.append({"role": str(m["role"]), "content": str(m["content"])})
    return summary, cleaned, turns


def set_session(session_id: str, summary: str, recent: List[Dict[str, str]], turns: int) -> None:
    if not _DB:
        raise RuntimeError("DB not initialized")

    # clamp
    recent = recent[-RECENT_MAX_MESSAGES:]

    _DB.execute(
        """
        INSERT INTO sessions(session_id, summary, recent, turns, updated_at)
        VALUES(?,?,?,?,?)
        ON CONFLICT(session_id) DO UPDATE SET
            summary=excluded.summary,
            recent=excluded.recent,
            turns=excluded.turns,
            updated_at=excluded.updated_at
        """,
        (session_id, summary, json.dumps(recent), int(turns), int(time.time())),
    )
    _DB.commit()


# =============================================================================
# Auth
# =============================================================================

def require_api_key(authorization: Optional[str]) -> None:
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server misconfigured: AERYNX_API_KEY missing")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid token")


# =============================================================================
# Models
# =============================================================================

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: str
    messages: List[ChatMessage]

class TTSRequest(BaseModel):
    text: str


# =============================================================================
# Text cleaning / spoken-only
# =============================================================================

_ROLE_TAG_RE = re.compile(r"(?i)\b(USER|ASSISTANT|SYSTEM|AERYNX)\s*:\s*")

# remove stage directions (light)
_STAGE_RE_1 = re.compile(r"\*[^*]{1,200}\*")      # *pauses*
_STAGE_RE_2 = re.compile(r"\([^)]{1,200}\)")      # (pauses)
_BRACKET_RE = re.compile(r"\[[^\]]{1,200}\]")     # [pauses]

# If a model sometimes leaks a "style prefix" like "Speak slowly..." you can strip it.
# (This matches common variants without being overly destructive.)
_STYLE_PREFIX_RE = re.compile(
    r"(?is)^\s*(speak|voice|tone)\s+(slowly|warmly|softly|gently|calmly)[^.\n]{0,120}[\.\n]\s*"
)

def sanitize_output(text: str) -> str:
    if not text:
        return ""

    # Stop if the model starts printing a new dialogue turn.
    cut_markers = ["\nUSER:", "\nASSISTANT:", "\nSYSTEM:", "\nAERYNX:"]
    positions = [text.find(m) for m in cut_markers if text.find(m) != -1]
    if positions:
        text = text[: min(positions)]

    # Remove role tags + stage directions
    text = _ROLE_TAG_RE.sub("", text)
    text = _STYLE_PREFIX_RE.sub("", text)
    text = _STAGE_RE_1.sub("", text)
    text = _STAGE_RE_2.sub("", text)
    text = _BRACKET_RE.sub("", text)

    # Normalize whitespace
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()

    return text.strip()


SPOKEN_BAN_PATTERNS = [
    r"\b(smiles|leans|pauses|waits|sighs|laughs|nods|shrugs)\b",
    r"\b(I\s+(pause|smile|lean|listen|wait|nod|breathe|think)\b.*?)([.!?]|$)",
]

def enforce_spoken_only(text: str) -> str:
    if not text:
        return ""

    for pat in SPOKEN_BAN_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    # De-dup consecutive sentences (keeps tone, avoids repeated reassurance)
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    out: List[str] = []
    prev = ""
    for s in sentences:
        s2 = s.strip()
        if not s2:
            continue
        key = re.sub(r"\W+", "", s2).lower()
        if key and key != prev:
            out.append(s2)
        prev = key

    # Clamp length (voice UX)
    out = out[:4]
    return " ".join(out).strip()


def stop_at_first_turn(text: str) -> str:
    for tok in ["User:", "USER:", "You:", "AERYNX:", "ASSISTANT:", "SYSTEM:"]:
        if tok in text:
            text = text.split(tok)[0]
    return text.strip()


def clamp_recent_message(m: Dict[str, str]) -> Dict[str, str]:
    role = (m.get("role") or "").strip().lower()
    if role not in ("user", "assistant", "system"):
        role = "user"
    content = (m.get("content") or "").strip()
    if len(content) > RECENT_MAX_CHARS:
        content = content[:RECENT_MAX_CHARS] + "…"
    return {"role": role, "content": content}


# =============================================================================
# Emotion + context detection (lightweight)
# =============================================================================

# Simple keyword heuristics (fast, zero extra API calls).
_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF]")

def detect_context(user_text: str) -> str:
    t = (user_text or "").lower()

    # urgency / frustration
    if any(k in t for k in ["asap", "right now", "immediately", "urgent", "hurry"]):
        return "serious"
    if any(k in t for k in ["this is annoying", "wasted", "stupid", "broken", "frustrated", "pissed", "mad", "angry"]):
        return "serious"

    # comfort / emotional
    if any(k in t for k in ["i feel", "anxious", "panic", "sad", "depressed", "lonely", "overwhelmed", "stressed"]):
        return "soothing"

    # coaching / planning
    if any(k in t for k in ["plan", "strategy", "roadmap", "steps", "next steps", "how do i", "help me", "design", "architecture"]):
        return "coach"

    # playful / light
    if _EMOJI_RE.search(user_text or "") or any(k in t for k in ["lol", "haha", "lmao", "😂", "😅"]):
        return "warm"

    return "default"


def adaptive_temperature(context: str, base: float) -> float:
    """
    Lower temp for serious/soothing, slightly higher for warm/coach creativity.
    """
    if context == "serious":
        return max(0.2, base - 0.15)
    if context == "soothing":
        return max(0.2, base - 0.10)
    if context == "coach":
        return min(1.2, base + 0.05)
    if context == "warm":
        return min(1.2, base + 0.08)
    return base


def select_tts_voice(context: str) -> str:
    if context == "serious":
        return AERYNX_TTS_VOICE_SERIOUS or AERYNX_TTS_VOICE_DEFAULT
    if context == "soothing":
        return AERYNX_TTS_VOICE_SOOTHING or AERYNX_TTS_VOICE_WARM or AERYNX_TTS_VOICE_DEFAULT
    if context == "coach":
        return AERYNX_TTS_VOICE_COACH or AERYNX_TTS_VOICE_DEFAULT
    if context == "warm":
        return AERYNX_TTS_VOICE_WARM or AERYNX_TTS_VOICE_DEFAULT
    return AERYNX_TTS_VOICE_DEFAULT


def style_system_prompt(context: str) -> str:
    """
    Persona tuning. Keep it simple + voice-safe.
    """
    base = (
        "You are AERYNX.\n"
        "You speak naturally: warm, present, conversational.\n"
        "Output ONLY what the user should hear (no meta, no stage directions, no role tags).\n"
        "Be concise and clear. Avoid filler.\n"
        "Ask at most one gentle follow-up question only if it truly helps.\n"
    )

    if context == "serious":
        return (
            base
            + "Mode: crisp, grounded, efficient. Focus on the solution.\n"
            + "Tone: calm authority, minimal reassurance.\n"
        )
    if context == "soothing":
        return (
            base
            + "Mode: supportive, steady, reassuring without being slow.\n"
            + "Tone: warm, gentle, but keep sentences short and clear.\n"
        )
    if context == "coach":
        return (
            base
            + "Mode: strategic coach. Give an actionable plan in 2–4 short sentences.\n"
            + "Tone: encouraging, practical.\n"
        )
    if context == "warm":
        return (
            base
            + "Mode: friendly warmth. Natural, light, human.\n"
            + "Tone: warm and intimate, not overly formal.\n"
        )
    return base


# =============================================================================
# Memory summarization (selective + gated)
# =============================================================================

# Only update durable summary if the user message suggests stable memory.
MEMORY_TRIGGER_RE = re.compile(
    r"\b("
    r"i prefer|i like|i don't like|i do not like|remember|"
    r"my goal|i want|going forward|from now on|always|never|"
    r"keep in mind|note that|important|constraint|decision|"
    r"we decided|let's do|let’s do"
    r")\b",
    re.IGNORECASE,
)

def should_update_memory(turns: int, last_user: str, prev_summary: str) -> bool:
    # Periodic check + trigger, or if we have no memory yet and a trigger appears
    periodic = (turns % MEMORY_EVERY_TURNS == 0)
    trigger = bool(MEMORY_TRIGGER_RE.search(last_user or ""))
    if trigger and (periodic or not prev_summary.strip()):
        return True
    return False


def update_summary_with_groq(prev_summary: str, last_user: str, assistant_out: str) -> str:
    if not groq_client:
        return prev_summary

    summ_msgs = [
        {
            "role": "system",
            "content": (
                "You write durable conversation memory for a personal assistant.\n"
                "Return 3–8 bullet points.\n"
                "Only stable facts: user preferences, goals, constraints, decisions, project facts.\n"
                "Avoid fluff, avoid transcript, avoid time-sensitive details unless explicitly important.\n"
                "Do not include role tags.\n"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Existing memory:\n{prev_summary.strip()}\n\n"
                f"New exchange:\nUser: {last_user.strip()}\nAssistant: {assistant_out.strip()}\n\n"
                "Updated memory bullets:"
            ),
        },
    ]
    try:
        raw = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=summ_msgs,
            temperature=0.2,
        ).choices[0].message.content or ""
        return sanitize_output(raw)
    except Exception:
        return prev_summary


# =============================================================================
# Providers: TGI, Groq + Voice: Groq STT, OpenAI TTS
# =============================================================================

def call_tgi(prompt: str, temperature: float) -> str:
    r = requests.post(
        TGI_URL,
        json={
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 220,
                "temperature": temperature,
                "top_p": TOP_P,
                "repetition_penalty": 1.18,
                "stop": ["\nUSER:", "\nASSISTANT:", "\nSYSTEM:", "\nAERYNX:", "\n\n"],
            },
        },
        timeout=TGI_TIMEOUT,
    )
    r.raise_for_status()
    j = r.json()
    return (j.get("generated_text") or "").strip()

def call_groq_chat(messages: List[Dict[str, str]], temperature: float) -> str:
    if not groq_client:
        raise RuntimeError("Groq not configured")
    comp = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=temperature,
    )
    return (comp.choices[0].message.content or "").strip()

def groq_stt(filename: str, audio_bytes: bytes) -> str:
    if not groq_client:
        raise RuntimeError("Groq not configured for STT")
    tr = groq_client.audio.transcriptions.create(
        file=(filename, audio_bytes),
        model="whisper-large-v3",
    )
    return (tr.text or "").strip()


def openai_tts(text: str, voice: str) -> bytes:
    if not openai_client:
        raise RuntimeError("OpenAI not configured for TTS")

    # Keep punctuation natural; do not inject line breaks (line breaks => pauses).
    clean = re.sub(r"\s{2,}", " ", (text or "").strip())

    kwargs: Dict[str, Any] = {
        "model": OPENAI_TTS_MODEL,
        "voice": voice,
        "input": clean,
    }

    # Optional speed support (only if your SDK supports it)
    if TTS_SPEED_ENABLE:
        kwargs["speed"] = TTS_SPEED_DEFAULT

    resp = openai_client.audio.speech.create(**kwargs)
    return resp.read()


# =============================================================================
# Health (stable / no flapping)
# =============================================================================

_LAST_TGI_OK_AT = 0.0
_LAST_TGI_ERR = ""
_TGI_GRACE_SECONDS = 45

def check_tgi_health() -> Dict[str, Any]:
    global _LAST_TGI_OK_AT, _LAST_TGI_ERR

    try:
        r = requests.get(TGI_HEALTH_URL, timeout=2)
        if r.status_code == 200:
            _LAST_TGI_OK_AT = time.time()
            _LAST_TGI_ERR = ""
            return {"tgi_ok": True, "tgi_state": "ready", "tgi_error": ""}
        _LAST_TGI_ERR = f"health_status={r.status_code}"
    except Exception as e:
        _LAST_TGI_ERR = str(e)

    recently_ok = (time.time() - _LAST_TGI_OK_AT) <= _TGI_GRACE_SECONDS
    return {
        "tgi_ok": False,
        "tgi_state": "warming_or_restarting" if recently_ok else "down",
        "tgi_error": _LAST_TGI_ERR,
    }


# =============================================================================
# Prompt builder (now persona-aware + short-term window)
# =============================================================================

def build_aerynx_prompt(
    system_prompt: str,
    session_summary: str,
    merged_messages: List[Dict[str, str]],
) -> str:
    memory = ""
    if session_summary.strip():
        memory = f"\nConversation memory (summary):\n{session_summary.strip()}\n"

    dialogue = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in merged_messages)
    return f"{system_prompt}{memory}\n{dialogue}\nAERYNX:"


def merge_recent_with_incoming(
    recent: List[Dict[str, str]],
    incoming: List[ChatMessage],
) -> List[Dict[str, str]]:
    inc = [clamp_recent_message({"role": m.role, "content": m.content}) for m in incoming]

    # If the client already sends a full history, prefer it.
    if len(inc) >= 3:
        return inc[-RECENT_MAX_MESSAGES:]

    # For /voice (typically 1 msg), merge recent + incoming, avoid exact duplicate.
    merged = list(recent)
    if merged and inc:
        if merged[-1]["role"] == inc[0]["role"] and merged[-1]["content"] == inc[0]["content"]:
            inc = inc[1:]

    merged.extend(inc)
    return merged[-RECENT_MAX_MESSAGES:]


# =============================================================================
# Core chat runner (shared by /chat and /voice)
# =============================================================================

def run_chat(session_id: str, incoming: List[ChatMessage]) -> Dict[str, Any]:
    prev_summary, prev_recent, prev_turns = get_session(session_id)

    # Determine context from the most recent user message we have (incoming preferred).
    last_user_text = ""
    for m in reversed(incoming):
        if m.role.lower() == "user":
            last_user_text = m.content
            break
    if not last_user_text:
        for m in reversed(prev_recent):
            if m.get("role") == "user":
                last_user_text = m.get("content", "")
                break

    context = detect_context(last_user_text)
    system_prompt = style_system_prompt(context)

    merged = merge_recent_with_incoming(prev_recent, incoming)

    # Adaptive temperature
    tgi_temp = adaptive_temperature(context, AERYNX_TEMP_BASE)
    groq_temp = adaptive_temperature(context, GROQ_TEMP_BASE)

    prompt = build_aerynx_prompt(system_prompt, prev_summary, merged)

    provider = "aerynx"
    fallback_reason = ""

    try:
        raw = call_tgi(prompt, temperature=tgi_temp)
        out = enforce_spoken_only(sanitize_output(raw))
        if not out:
            raise RuntimeError("Empty TGI output")
    except Exception as e:
        provider = "groq"
        fallback_reason = str(e)

        groq_msgs: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        if prev_summary.strip():
            groq_msgs.append({"role": "system", "content": f"Conversation memory:\n{prev_summary.strip()}"})
        groq_msgs.extend(merged)

        raw = call_groq_chat(groq_msgs, temperature=groq_temp)
        out = enforce_spoken_only(sanitize_output(raw))

    # Update short-term window: append assistant output
    new_recent = merged + [{"role": "assistant", "content": out}]
    new_recent = [clamp_recent_message(m) for m in new_recent][-RECENT_MAX_MESSAGES:]

    # Selective durable memory update
    turns = prev_turns + 1
    summarized = False
    summary = prev_summary

    if should_update_memory(turns, last_user_text, prev_summary):
        summarized = True
        summary = update_summary_with_groq(prev_summary, last_user_text, out) or prev_summary

    set_session(session_id, summary, new_recent, turns)

    resp: Dict[str, Any] = {
        "response": out,
        "provider": provider,
        "session_id": session_id,
        "turns": turns,
        "summarized": summarized,
        "context": context,
        "temps": {"tgi": tgi_temp, "groq": groq_temp},
    }
    if provider == "groq":
        resp["fallback_reason"] = fallback_reason
    return resp


# =============================================================================
# Routes
# =============================================================================

@app.get("/health")
def health():
    tgi = check_tgi_health()
    return {
        "ok": True,
        **tgi,
        "groq_configured": bool(groq_client),
        "openai_configured": bool(openai_client),
        "time_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/chat")
@limiter.limit("10/minute")
def chat(
    request: Request,
    req: ChatRequest,
    authorization: Optional[str] = Header(default=None),
):
    require_api_key(authorization)
    data = run_chat(req.session_id, req.messages)
    return JSONResponse(data)


@app.post("/stt")
@limiter.limit("20/minute")
async def stt(
    request: Request,
    audio: UploadFile = File(...),
    authorization: Optional[str] = Header(default=None),
):
    require_api_key(authorization)
    audio_bytes = await audio.read()
    text = groq_stt(audio.filename or "audio.bin", audio_bytes)
    return {"text": text}


@app.post("/tts")
@limiter.limit("20/minute")
def tts(
    request: Request,
    req: TTSRequest,
    authorization: Optional[str] = Header(default=None),
):
    require_api_key(authorization)
    # default voice
    audio_bytes = openai_tts(req.text, voice=AERYNX_TTS_VOICE_DEFAULT)
    return Response(content=audio_bytes, media_type="audio/mpeg")


@app.post("/voice")
@limiter.limit("10/minute")
async def voice(
    request: Request,
    audio: UploadFile = File(...),
    session_id: str = "voice",
    authorization: Optional[str] = Header(default=None),
):
    require_api_key(authorization)

    # --- STT ---
    audio_bytes = await audio.read()
    transcript = groq_stt(audio.filename or "audio.bin", audio_bytes)

    # --- CHAT ---
    data = run_chat(
        session_id=session_id,
        incoming=[ChatMessage(role="user", content=transcript)],
    )

    response_text = stop_at_first_turn(data.get("response", "") or "")
    response_text = enforce_spoken_only(response_text)

    # Context-driven voice persona
    context = data.get("context") or "default"
    voice_name = select_tts_voice(context)

    # --- TTS ---
    audio_out = openai_tts(response_text, voice=voice_name)
    audio_b64 = base64.b64encode(audio_out).decode("utf-8")

    return JSONResponse({
        "transcript": transcript,
        "response": response_text,
        "context": context,
        "voice": voice_name,
        "audio_base64": audio_b64,
    })


# =============================================================================
# Demo UI (single page)
# =============================================================================

DEMO_HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>AERYNX Voice Demo</title>
<style>
  body {
    font-family: system-ui, sans-serif;
    max-width: 900px;
    margin: 40px auto;
    padding: 0 16px;
    background: #0f1115;
    color: #eaeaf0;
  }

  h2 { margin-bottom: 8px; }

  .panel {
    background: #161a22;
    border-radius: 14px;
    padding: 16px;
    margin-top: 16px;
  }

  #mic {
    width: 120px;
    height: 120px;
    border-radius: 60px;
    background: #2a2f3a;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 42px;
    cursor: pointer;
    margin: 24px auto;
    transition: all 0.2s ease;
    user-select: none;
  }

  #mic.recording {
    background: #c0392b;
    box-shadow: 0 0 0 12px rgba(192,57,43,0.25);
    animation: pulse 1.2s infinite;
  }

  @keyframes pulse {
    0% { box-shadow: 0 0 0 8px rgba(192,57,43,0.3); }
    70% { box-shadow: 0 0 0 18px rgba(192,57,43,0); }
    100% { box-shadow: 0 0 0 8px rgba(192,57,43,0); }
  }

  .label { font-size: 13px; color: #aaa; margin-top: 8px; }
  .text { white-space: pre-wrap; margin-top: 6px; }

  input {
    width: 100%;
    padding: 10px;
    border-radius: 10px;
    border: none;
    margin-top: 8px;
  }

  .status {
    text-align: center;
    margin-top: 10px;
    color: #9aa4ff;
    min-height: 20px;
  }

  .chips {
    display:flex; gap:10px; flex-wrap:wrap;
    font-size: 12px; color:#cfd5ff;
    justify-content:center;
    margin-top: 6px;
  }
  .chip {
    background:#2a2f3a; border-radius:999px; padding:6px 10px;
  }
</style>
</head>

<body>
<h2>AERYNX Voice MVP</h2>
<div class="label">Paste API key once (stored locally)</div>
<input id="key" placeholder="AERYNX_API_KEY" />

<div id="mic">🎙️</div>
<div class="status" id="status">Hold to speak</div>
<div class="chips">
  <div class="chip" id="ctx">context: —</div>
  <div class="chip" id="vce">voice: —</div>
</div>

<div class="panel">
  <div class="label">Transcript</div>
  <div class="text" id="transcript">—</div>
</div>

<div class="panel">
  <div class="label">AERYNX</div>
  <div class="text" id="response">—</div>
</div>

<script>
let mediaRecorder;
let chunks = [];
let recording = false;
let stream = null;

const mic = document.getElementById("mic");
const status = document.getElementById("status");
const transcriptEl = document.getElementById("transcript");
const responseEl = document.getElementById("response");
const keyInput = document.getElementById("key");
const ctxEl = document.getElementById("ctx");
const vceEl = document.getElementById("vce");

keyInput.value = localStorage.getItem("aerynx_api_key") || "";
keyInput.onchange = () =>
  localStorage.setItem("aerynx_api_key", keyInput.value.trim());

async function startRecording() {
  if (recording) return;
  recording = true;

  if (!stream) {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  }

  chunks = [];
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = e => chunks.push(e.data);
  mediaRecorder.start();

  mic.classList.add("recording");
  status.textContent = "Listening…";
}

async function stopRecording() {
  if (!recording) return;
  recording = false;

  await new Promise(resolve => {
    mediaRecorder.onstop = resolve;
    mediaRecorder.stop();
  });

  mic.classList.remove("recording");
  status.textContent = "Thinking…";

  const blob = new Blob(chunks, { type: mediaRecorder.mimeType });
  const form = new FormData();
  form.append("audio", blob, "speech.webm");

  const apiKey = localStorage.getItem("aerynx_api_key");
  if (!apiKey) {
    status.textContent = "Missing API key";
    return;
  }

  try {
    const res = await fetch("/voice", {
      method: "POST",
      headers: {
        "Authorization": "Bearer " + apiKey
      },
      body: form
    });

    const data = await res.json();

    transcriptEl.textContent = data.transcript || "—";
    responseEl.textContent = data.response || "—";
    ctxEl.textContent = "context: " + (data.context || "—");
    vceEl.textContent = "voice: " + (data.voice || "—");

    if (!res.ok) {
      status.textContent = "Server error";
      return;
    }

    // 🔊 play audio
    const audioBytes = Uint8Array.from(atob(data.audio_base64), c => c.charCodeAt(0));
    const audioBlob = new Blob([audioBytes], { type: "audio/mpeg" });
    const audioUrl = URL.createObjectURL(audioBlob);

    const audio = new Audio(audioUrl);
    audio.play();

    status.textContent = "Speaking…";
    audio.onended = () => status.textContent = "Hold to speak";

  } catch (e) {
    console.error(e);
    status.textContent = "Network error";
  }
}

mic.addEventListener("pointerdown", startRecording);
mic.addEventListener("pointerup", stopRecording);
mic.addEventListener("pointerleave", stopRecording);
</script>

</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def demo():
    return DEMO_HTML

