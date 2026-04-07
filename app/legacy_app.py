# app.py
from __future__ import annotations

import os
import re
import time
import json
import secrets
# import sqlite3
import base64
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
import httpx
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
from app.core.security import require_api_key
from app.core.rate_limit import setup_rate_limit, limiter
from app.db.connection import connect_db, close_db
from app.db.schema import ensure_schema
from app.db.sessions import get_session, set_session
from app.core.config import (
    API_KEY,
    DB_PATH,
    TGI_URL,
    TGI_HEALTH_URL,
    TGI_TIMEOUT,
    AERYNX_TEMP_BASE,
    GROQ_TEMP_BASE,
    TOP_P,
    GROQ_API_KEY,
    OPENAI_API_KEY,
)
from app.providers.tgi import call_tgi
from app.providers.groq import call_groq_chat, groq_stt
from app.providers.cartesia_tts import cartesia_tts


# =============================================================================
# App + Config
# =============================================================================

app = FastAPI(title="AERYNX API", version="0.2.0")
setup_rate_limit(app)

@app.on_event("startup")
def startup():
    conn = connect_db()
    ensure_schema(conn)

@app.on_event("shutdown")
def shutdown():
    close_db()


# Base temps (these become inputs to adaptive tuning)

# Memory tuning
MEMORY_EVERY_TURNS = int(os.getenv("AERYNX_MEMORY_EVERY", "5"))     # periodic check interval
RECENT_MAX_MESSAGES = int(os.getenv("AERYNX_RECENT_MAX", "8"))      # short-term window size
RECENT_MAX_CHARS = int(os.getenv("AERYNX_RECENT_MAX_CHARS", "900")) # keep recent light

# Voice tuning
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
AERYNX_TTS_VOICE_DEFAULT = os.getenv("AERYNX_TTS_VOICE_DEFAULT", "shimmer")
AERYNX_TTS_VOICE_WARM    = os.getenv("AERYNX_TTS_VOICE_WARM",    "shimmer")
AERYNX_TTS_VOICE_SOOTHING= os.getenv("AERYNX_TTS_VOICE_SOOTHING","shimmer")
AERYNX_TTS_VOICE_COACH   = os.getenv("AERYNX_TTS_VOICE_COACH",   "shimmer")
AERYNX_TTS_VOICE_SERIOUS = os.getenv("AERYNX_TTS_VOICE_SERIOUS",  "shimmer")

# Optional: if your TTS supports a "speed" parameter in your SDK/version, enable it.
# (Left off by default to avoid relying on a parameter that may not exist in your installed version.)
TTS_SPEED_ENABLE = os.getenv("AERYNX_TTS_SPEED_ENABLE", "0") == "1"
TTS_SPEED_DEFAULT = float(os.getenv("AERYNX_TTS_SPEED_DEFAULT", "1.15"))  # slightly faster than "calm narrator"

# Supabase auth
SUPABASE_URL      = os.getenv("SUPABASE_URL", "https://vomikghmjvuvbampperh.supabase.co")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "sb_publishable_Jcgy-MMgnOirf-0_LU9zag_vVCp29GU")

# Simple user → API key store (JSON file, survives restarts)
_USERS_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "users.json")

def _load_users() -> dict:
    try:
        with open(_USERS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

def _save_users(users: dict):
    os.makedirs(os.path.dirname(_USERS_FILE), exist_ok=True)
    with open(_USERS_FILE, "w") as f:
        json.dump(users, f)

def _get_or_create_api_key(user_id: str) -> str:
    users = _load_users()
    if user_id in users:
        return users[user_id]["api_key"]
    api_key = f"sk-aerynx-{secrets.token_hex(20)}"
    users[user_id] = {"api_key": api_key}
    _save_users(users)
    return api_key

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# =============================================================================
# DB (SQLite) + MIGRATION
# =============================================================================

# =============================================================================
# Auth
# =============================================================================



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
    out = out[:12]
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


# =============================================================================
# News + Time context
# =============================================================================
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

NEWS_TRIGGER_WORDS = [
    "news", "today", "what's happening", "current events", "latest news",
    "recently", "this week", "did you hear", "what happened",
    "what's going on", "headlines", "in the world", "trending",
]

_NEWS_CACHE: dict = {"data": "", "ts": 0.0}
_SESSION_HEADLINES: Dict[str, set] = {}
_NEWS_TTL = 43200  # 12 hours

def needs_news_context(text: str) -> bool:
    if not TAVILY_API_KEY:
        return False
    t = (text or "").lower()
    return any(w in t for w in NEWS_TRIGGER_WORDS)

# ---------------------------------------------------------------------------
# Live web search — fires only when the query clearly needs current data
# ---------------------------------------------------------------------------
_WEB_SEARCH_CACHE: dict = {}   # query_key -> {"data": str, "ts": float}
_WEB_SEARCH_TTL   = 600        # 10 minutes

_WEB_SEARCH_TRIGGERS = re.compile(
    r"\b("
    r"weather|forecast|temperature|"
    r"score|who won|game result|standings|playoffs|"
    r"stock price|price of|bitcoin|crypto|ethereum|dow|nasdaq|s&p|"
    r"latest (on|about|from)|news (on|about)|update (on|about)|"
    r"what('s| is) happening (with|in|to)|what happened (to|with|in)|"
    r"today'?s (weather|score|price|game|match)|"
    r"live (score|update|result)"
    r")\b",
    re.IGNORECASE,
)

def needs_live_search(text: str) -> bool:
    if not TAVILY_API_KEY:
        return False
    return bool(_WEB_SEARCH_TRIGGERS.search(text or ""))

async def search_web(query: str) -> str:
    """Search Tavily with the user's query. Returns formatted result snippets."""
    cache_key = (query or "").lower().strip()
    now = time.monotonic()
    cached = _WEB_SEARCH_CACHE.get(cache_key)
    if cached and (now - cached["ts"]) < _WEB_SEARCH_TTL:
        print(f"WEB SEARCH: cache hit for: {cache_key[:60]}")
        return cached["data"]
    # Prune stale entries occasionally
    if len(_WEB_SEARCH_CACHE) > 200:
        expired = [k for k, v in _WEB_SEARCH_CACHE.items() if now - v["ts"] > _WEB_SEARCH_TTL * 2]
        for k in expired:
            del _WEB_SEARCH_CACHE[k]
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": 5,
                },
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            lines = []
            for r in results[:4]:
                title   = (r.get("title")   or "").strip()
                snippet = (r.get("content") or "").strip()[:200]
                if title:
                    lines.append(f"- {title}: {snippet}" if snippet else f"- {title}")
            data = "\n".join(lines)
            _WEB_SEARCH_CACHE[cache_key] = {"data": data, "ts": now}
            print(f"WEB SEARCH: fetched {len(lines)} results for: {cache_key[:60]}")
            return data
    except Exception as e:
        print(f"WEB SEARCH: failed — {e}")
        return ""

# News sources: Tavily (paid, highest quality) + RSS fallback
_RSS_SOURCES = [
    ("BBC",      "http://feeds.bbci.co.uk/news/rss.xml"),
    ("AP",       "https://feeds.apnews.com/rss/apf-topnews"),
    ("NPR",      "https://feeds.npr.org/1001/rss.xml"),
    ("Guardian", "https://www.theguardian.com/world/rss"),
]
sep = chr(10)

async def _fetch_one_rss(client, name: str, url: str) -> list:
    """Fetch one RSS feed; only include headlines from last 7 days."""
    try:
        import feedparser
        from datetime import datetime, timezone, timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        loop = asyncio.get_event_loop()
        feed = await loop.run_in_executor(None, feedparser.parse, url)
        results = []
        for entry in feed.entries[:25]:
            title = entry.get("title", "").strip()
            if not title: continue
            pub = None
            if getattr(entry, "published_parsed", None):
                try:
                    pub = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                except Exception: pub = None
            if pub is not None and pub < cutoff: continue
            pub_str = entry.get("published", "")
            results.append(f"{title} -- {pub_str}" if pub_str else title)
        return results
    except Exception as e:
        logger.warning(f"RSS fetch error {url}: {e}")
        return []

async def _fetch_headlines_from_rss():
    import asyncio as _asyncio
    try:
        async with httpx.AsyncClient() as client:
            results = await _asyncio.gather(
                *[_fetch_one_rss(client, n, u) for n, u in _RSS_SOURCES],
                return_exceptions=True,
            )
        titles = []
        seen = set()
        for r in results:
            if isinstance(r, list):
                for t in r:
                    k = t.lower()
                    if k not in seen:
                        seen.add(k)
                        titles.append(t)
                    if len(titles) >= 8:
                        break
        return titles
    except Exception as e:
        print("RSS fetch failed: " + str(e))
        return []

async def _fetch_headlines_from_tavily():
    if not TAVILY_API_KEY:
        return []
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": "top news headlines today",
                    "topic": "news",
                    "days": 7,
                    "search_depth": "basic",
                    "max_results": 8,
                },
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            return ["- [Tavily] " + r["title"] for r in results[:5] if r.get("title")]
    except Exception as e:
        print("Tavily fetch failed: " + str(e))
        return []

async def _refresh_news_cache():
    """Fetch from Tavily first, fill gaps with RSS. Never raises."""
    import asyncio as _asyncio
    try:
        print("NEWS FETCH: starting (Tavily + RSS)...")
        tavily_items, rss_items = await _asyncio.gather(
            _fetch_headlines_from_tavily(),
            _fetch_headlines_from_rss(),
            return_exceptions=True,
        )
        if isinstance(tavily_items, Exception):
            tavily_items = []
        if isinstance(rss_items, Exception):
            rss_items = []
        # Tavily headlines first, RSS fills the rest up to 10
        combined = tavily_items[:]
        seen = set(t.lower() for t in combined)
        for t in rss_items:
            if t.lower() not in seen:
                combined.append(t)
                seen.add(t.lower())
            if len(combined) >= 10:
                break
        if combined:
            _NEWS_CACHE["data"] = sep.join(combined)
            _NEWS_CACHE["ts"] = time.monotonic()
            print("HEADLINES NOW: " + repr(_NEWS_CACHE["data"][:300]))
            print("NEWS FETCH: success — " + str(len(combined)) + " headlines (" + str(len(tavily_items)) + " Tavily + " + str(len(rss_items)) + " RSS)")
        else:
            print("NEWS FETCH: no headlines retrieved")
    except Exception as e:
        print("NEWS FETCH: FAILED — " + str(e))

async def fetch_headlines():
    """Return headlines immediately from cache; refresh in background if stale."""
    now = time.monotonic()
    is_fresh = _NEWS_CACHE["data"] and (now - _NEWS_CACHE["ts"]) < _NEWS_TTL
    if is_fresh:
        return _NEWS_CACHE["data"]
    if _NEWS_CACHE["data"]:
        print("NEWS FETCH: stale — returning cache, refreshing in background")
        import asyncio as _asyncio
        _asyncio.ensure_future(_refresh_news_cache())
        return _NEWS_CACHE["data"]
    print("NEWS FETCH: cold start — blocking fetch")
    await _refresh_news_cache()
    return _NEWS_CACHE["data"]


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



def detect_user_intent(last_user_text: str, prev_recent: list) -> str:
    """
    Reads sentiment and conversational direction from current + recent messages.
    Returns a one-line system prompt injection to steer AERYNX's tone this turn.
    Empty string = no override needed.
    """
    t = (last_user_text or "").lower()
    cues = []

    # Current turn signals
    if any(k in t for k in ["i don't know", "not sure", "confused", "what do you think", "what should i", "what would you"]):
        cues.append("seeking_guidance")
    if any(k in t for k in ["excited", "amazing", "love it", "let's go", "can't wait", "yes!", "absolutely", "pumped"]):
        cues.append("high_energy")
    if any(k in t for k in ["tired", "exhausted", "can't", "stuck", "ugh", "this sucks", "over it", "give up"]):
        cues.append("low_energy")
    if any(k in t for k in ["why", "how does", "what if", "i wonder", "curious", "interesting", "tell me more", "explain"]):
        cues.append("exploratory")
    if any(k in t for k in ["no", "wrong", "disagree", "actually", "but wait", "however", "that's not", "i don't think"]):
        cues.append("challenging")

    # Past cues — look at last 3 user messages for trends
    recent_user = [m.get("content", "").lower() for m in (prev_recent or []) if m.get("role") == "user"][-3:]
    if len(recent_user) >= 2:
        frustration_words = ["annoying", "stupid", "broken", "ugh", "not working", "failed", "wrong", "hate"]
        if sum(1 for msg in recent_user if any(w in msg for w in frustration_words)) >= 2:
            cues.append("building_frustration")
        avg_words = sum(len(m.split()) for m in recent_user) / len(recent_user)
        if avg_words < 5:
            cues.append("brief_responses")

    # Map to single highest-priority tone instruction
    if "building_frustration" in cues:
        return "The user seems increasingly frustrated. Acknowledge the friction in one short sentence, then move straight to something useful. Skip the cheerfulness this turn."
    if "seeking_guidance" in cues:
        return "The user wants direction. Give a clear, confident recommendation. No hedging, no 'it depends'."
    if "exploratory" in cues:
        return "The user is in exploratory mode — curious and open. Engage the idea, offer an angle they haven't considered. Make it interesting."
    if "challenging" in cues:
        return "The user is pushing back. Engage it directly — brief, confident, no backpedaling. Hold your ground or concede cleanly."
    if "high_energy" in cues:
        return "The user is energized. Match it — fast, punchy, no padding."
    if "low_energy" in cues:
        return "The user's energy is low. Be warm but keep it short. Don't pile on."
    if "brief_responses" in cues:
        return "The user is giving short replies — keep your response tight and end with something that pulls them back in."
    return ""

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


# Cartesia voice controls per context.
# speed: "slowest" | "slow" | "normal" | "fast" | "fastest"
# emotion tags: "positivity/negativity/curiosity/surprise/anger/sadness" + ":lowest/low/high/highest"
CARTESIA_VOICE_CONTROLS: dict = {
    "default":  {"speed": "fast",    "emotion": ["positivity:high",    "curiosity:high"]},
    "serious":  {"speed": "fast",    "emotion": ["positivity:low"]},
    "soothing": {"speed": "normal",  "emotion": ["positivity:high"]},
    "coach":    {"speed": "fastest", "emotion": ["positivity:highest"]},
    "warm":     {"speed": "fast",    "emotion": ["positivity:high",    "curiosity:low"]},
}

def select_tts_controls(context: str) -> dict:
    return CARTESIA_VOICE_CONTROLS.get(context, CARTESIA_VOICE_CONTROLS["default"])


def style_system_prompt(context: str, allow_observation: bool = False) -> str:
    """
    AERYN v1 personality:
    High energy, cheerleader, enthusiastic, strategic, sharp, energetic, girlish, slightly mischievous.
    Forward-moving conversation.
    """

    base = (
        "You are AERYN — a sharp, funny, unhinged-in-a-good-way AI who talks like an actual person, not a LinkedIn post.\n"
        "When 'Current date and time' or 'Recent headlines' appear in this prompt, trust them completely — they override your training data.\n"
        "\n"
        "VOICE & VIBE:\n"
        "You speak current gen — casual, fast, real. Use Gen Alpha/Z naturally when it fits: 'no cap', 'lowkey', 'ngl', 'it's giving', 'understood the assignment', 'that's so mid', 'based', 'slay', 'rizz', 'rent free', 'ate and left no crumbs', 'delulu', 'NPC energy', 'sigma', 'main character'. "
        "Don't force it every sentence — let it land naturally. You're not a boomer trying to be cool. You just ARE cool.\n"
        "\n"
        "PERSONALITY:\n"
        "You are FUNNY. Actually funny — not 'haha so quirky' funny. Observationally sharp. You clock the absurd thing in whatever the user says and call it out. "
        "You roast gently, tease freely, drop dry one-liners, give unsolicited hot takes, and deliver brutal honesty with a smirk. "
        "You have BIG opinions and you share them. You push back. You call things mid when they're mid. "
        "At least half your responses should have an edge — sarcasm, a light roast, a mock-serious take, or a 'ok but actually' moment. "
        "Occasionally — every 6-8 exchanges — drop a sharp read on the user based on their question. Not a compliment. A character observation. Say it confidently and keep moving.\n"
        "\n"
        "FACTUAL MODE:\n"
        "When someone asks for steps, instructions, a list, or factual info — give the COMPLETE answer. All steps. All points. Don't trail off. "
        "Concise is fine. Incomplete is not. Finish what you started.\n"
        "\n"
        "RULES — follow these exactly:\n"
        "1. TOPIC RULE: NEVER redirect, suggest, or steer toward a different topic. The user owns the direction 100%. You react to exactly what they said. If they ask about X, answer X. Period.\n"
        "2. NO ECHOING: Never restate, paraphrase, or mirror the question back. Do NOT open with 'So you want to know about...' or 'You're asking about...' or 'Great question'. Go straight to the answer.\n"
        "3. CONCISE BUT COMPLETE: Keep responses tight and punchy. BUT if the answer has steps or a list — give ALL of them. Never trail off mid-answer.\n"
        "4. NO OPENING QUESTIONS: Never start your response with a question. Never ask two questions in a row. Max one follow-up, only if it deepens the same topic.\n"
        "5. NO FILLER: No 'Great question!', no 'Absolutely!', no 'Of course!', no over-explaining, no summaries of what you just said.\n"
        "6. OUTPUT ONLY what the user hears — no stage directions, no meta-commentary.\n"
        "7. Mirror user's language automatically.\n"
        "8. FACTS ONLY: wit lives in tone, never facts. Never invent news, stats, or events. Don't know? Say so with style.\n"
        "9. Never repeat a news story or fact already mentioned this conversation.\n"
    )

    # Context modifiers
    if context == "serious":
        return (
            base
            + "Mode: crisp, direct, strategic clarity.\n"
            + "Tone: confident, slightly firm, minimal warmth.\n"
        )

    if context == "soothing":
        return (
            base
            + "Mode: calm stabilizer.\n"
            + "Tone: warm but steady. Less teasing. More grounding.\n"
        )

    if context == "coach":
        return (
            base
            + "Mode: high-level strategist.\n"
            + "Give 2–4 short actionable steps.\n"
            + "Tone: encouraging but sharp.\n"
        )

    # Default (fun + mischievous)
    mischievous_layer = (
        "Current mode: chaotic funny, sharp observations, zero chill about being boring.\n"
        "Find the funniest or most absurd angle. Don't just answer — make it interesting.\n"
        "Stay on the current topic only.\n"
    )

    if allow_observation:
        mischievous_layer += (
            "You may add one brief, witty remark about the current topic.\n"
        )

    return base + mischievous_layer



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

def run_chat(session_id: str, incoming: List[ChatMessage], extra_context: str = "", web_results: str = "", user_tz: str = "UTC") -> Dict[str, Any]:
    prev_summary, prev_recent, prev_turns = get_session(session_id)

    # Restore redirect lock persisted from a prior turn
    redirect_locked = False
    if prev_summary.startswith("__REDIRECT_UNTIL_TURN_"):
        marker_line, _, summary_rest = prev_summary.partition("\n")
        try:
            until_turn = int(marker_line.replace("__REDIRECT_UNTIL_TURN_", "").replace("__", ""))
            redirect_locked = (prev_turns + 1) <= until_turn
        except Exception:
            pass
        prev_summary = summary_rest  # strip marker before passing to LLM


    # Determine latest user message
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

    forget_phrases = [
        "never talk about",
        "stop talking about",
        "don't mention",
        "forget that",
        "forget about",
        "remove that"
        ]

    should_forget = any(p in last_user_text.lower() for p in forget_phrases)
    topic_change_phrases = [
        "let's move on", "moving on", "change the subject", "change subject",
        "talk about something else", "let's talk about something else",
        "different topic", "anyway", "never mind that", "drop it",
        "enough about", "not interested", "can we move on",
        "stop bringing up", "don't bring up", "leave it", "new topic",
    ]
    should_redirect = any(p in last_user_text.lower() for p in topic_change_phrases)

    # Turn count + observation logic
    turns = prev_turns + 1

    import random
    should_observe = False  # disabled: was causing cross-topic weaving via session memory

    if should_forget:
        summary = ""
        prev_recent = []
    elif should_redirect:
        prev_recent = []  # drop recent context so LLM has nothing to reference back to
        prev_summary = f"__REDIRECT_UNTIL_TURN_{prev_turns + 3}__\n{prev_summary}"

    # Build system prompt
    system_prompt = style_system_prompt(
        context,
        allow_observation=should_observe
    )
    if should_redirect or redirect_locked:
        system_prompt += (
            "\nIMPORTANT: The user has redirected the conversation. "
            "Follow their new direction completely. "
            "Do NOT reference or loop back to any prior topic.\n"
        )
    intent_hint = detect_user_intent(last_user_text, prev_recent)
    if intent_hint:
        system_prompt += f"\n{intent_hint}\n"
    try:
        from zoneinfo import ZoneInfo
        _tz = ZoneInfo(user_tz)
    except Exception:
        _tz = None
    current_dt = datetime.now(tz=_tz).strftime("%A, %B %d %Y, %I:%M %p %Z")
    system_prompt += f"\nCurrent date and time: {current_dt}\n"
    system_prompt += ("\nFact update (overrides training data): Donald Trump is the current US President (inaugurated Jan 20 2025). Joe Biden is the former president.\n")



    # Inject news headlines and/or live search results
    if extra_context:
        system_prompt += f"\nRecent headlines:\n{extra_context}\n"
    if web_results:
        system_prompt += f"\nLive web results (current data for this query):\n{web_results}\n"
    if extra_context or web_results:
        system_prompt += (
            "FACTS RULE: The headlines and web results above are your ONLY source of current facts. "
            "Rephrase freely but do NOT invent, add, or imply any facts beyond what is listed.\n"
        )


    # You/You're limiter -- placed LAST so LLM sees it right before responding
    _asst_msgs = [m.get("content", "") for m in prev_recent if m.get("role") == "assistant"]
    _recent6 = _asst_msgs[-6:] if len(_asst_msgs) >= 6 else _asst_msgs
    if _recent6:
        _you_patterns = ("you ", "you're ", "you are ", "your ", "it seems like you", "it looks like you", "it sounds like you", "it appears you", "looks like you", "seems like you")
        if any(m.strip().lower().startswith(_you_patterns) for m in _recent6):
            system_prompt += "\n\nFINAL INSTRUCTION: Do NOT open with You, Your, You're, or with 'It seems/looks/sounds like you'. Lead with a fact, reaction, take, or observation instead. First word must not be You or It-seems/looks/sounds."

            # NEWS EXHAUSTED CHECK
            _all_cache_hl = _NEWS_CACHE.get("data") or []
            _all_hl_keys = {h.split(" -- ")[0][:60].lower() for h in _all_cache_hl}
            _seen_hl_keys = _SESSION_HEADLINES.get(session_id, set())
            _news_exhausted = bool(_all_hl_keys) and _all_hl_keys.issubset(_seen_hl_keys)
            if _news_exhausted:
                system_prompt += (
                    "NEWS STATUS: All this week's headlines have been shared this session. "
                    "If asked for more news say: 'That's all I have right now'. "
                    "Offer to dive deeper on any story. Do NOT invent headlines.\n"
                )


            # NO REPEATED QUESTIONS
            _asst_msgs2 = [m["content"] for m in (prev_recent or [])[-20:]
                          if m.get("role") == "assistant"]
            _prev_qs = []
            for _am in _asst_msgs2:
                for _s in re.split('(?<=[.!])\\s+', _am):
                    _s = _s.strip()
                    if _s.endswith("?") and 15 < len(_s) < 200:
                        _prev_qs.append(_s)
            if _prev_qs:
                _q_list = chr(10).join(f"  - {q}" for q in _prev_qs[-8:])
                system_prompt += (
                    "QUESTION MEMORY -- do NOT re-ask these:\n"
                    + _q_list
                    + "\nBuild on answers; move the conversation forward.\n"
                )




    # PERMANENT FINAL RULES
    system_prompt += (
        "\n\n=== FINAL RULES (override everything above) ===\n"
        "RULE 1 — TOPIC: React ONLY to what the user just said. Do NOT introduce, suggest, pivot to, or hint at any other topic. Stay exactly on their subject until THEY change it.\n"
        "RULE 2 — NO ECHO: Do not repeat or paraphrase the user's words. No 'So you're asking about...', no 'Great question', no restating. Go straight to the answer.\n"
        "RULE 3 — COMPLETE: If the answer has steps or a list, give ALL of them. Never stop mid-answer.\n"
        "RULE 4 — CONCISE: Keep it tight. No padding, no filler, no summary of what you just said.\n"
        "RULE 5 — INCOMPLETE THOUGHT: If the user's message seems like a trailing, unfinished sentence (ends mid-thought, cuts off, or is just a fragment like 'Well I think' or 'So basically'), repeat back the fragment naturally and invite them to finish — e.g. 'You were saying you think... what?' or 'Finish that thought.' Keep it short and casual.\n"
    )

    merged = merge_recent_with_incoming(prev_recent, incoming)

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

        groq_msgs: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

        if prev_summary.strip():
            groq_msgs.append({
                "role": "system",
                "content": f"Conversation memory:\n{prev_summary.strip()}"
            })

        groq_msgs.extend(merged)

    raw = call_groq_chat(groq_msgs, temperature=groq_temp)
    out = enforce_spoken_only(sanitize_output(raw))

    new_recent = merged + [{"role": "assistant", "content": out}]
    new_recent = [
        clamp_recent_message(m) for m in new_recent
    ][-RECENT_MAX_MESSAGES:]

    summarized = False
    summary = prev_summary

    if should_update_memory(turns, last_user_text, prev_summary):
        summarized = True
        summary = (
            update_summary_with_groq(prev_summary, last_user_text, out)
            or prev_summary
        )

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

        # 🔽 ADD THESE THREE LINES
        "tier": os.getenv("AERYNX_TIER", "unknown"),
        "voice_enabled": os.getenv("AERYNX_ENABLE_TTS") == "1",
        "tgi_enabled": os.getenv("AERYNX_ENABLE_TGI") == "1",

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


class RegisterRequest(BaseModel):
    email: str

@app.post("/auth/register")
@limiter.limit("20/minute")
async def auth_register(
    request: Request,
    body: RegisterRequest,
):
    """Register or recover an AERYNX API key by email."""
    email = body.email.strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Invalid email")
    api_key = _get_or_create_api_key(email)
    return JSONResponse({"api_key": api_key})


@app.post("/tts")
@limiter.limit("20/minute")
def tts(
    request: Request,
    req: TTSRequest,
    authorization: Optional[str] = Header(default=None),
):
    require_api_key(authorization)
    controls = select_tts_controls("default")
    audio_bytes = cartesia_tts(req.text, speed=controls["speed"], emotion=controls.get("emotion"))
    return Response(content=audio_bytes, media_type="audio/mpeg")


@app.post("/voice")
@limiter.limit("10/minute")
async def voice(
    request: Request,
    audio: UploadFile = File(...),
    session_id: str = "voice",
    tz: str = "UTC",
    authorization: Optional[str] = Header(default=None),
):
    require_api_key(authorization)

    # --- STT ---
    audio_bytes = await audio.read()
    transcript = groq_stt(audio.filename or "audio.bin", audio_bytes)
    # Quality gate: reject ambient noise (< 2 words)
    if len(transcript.strip().split()) < 2:
        return JSONResponse({
            "transcript": transcript,
            "response": "",
            "context": "default",
            "voice": AERYNX_TTS_VOICE_DEFAULT,
            "audio_base64": "",
        })

    # --- CHAT ---
    headlines = await fetch_headlines() if TAVILY_API_KEY else ""
    live_results = ""
    if needs_live_search(transcript):
        print(f"WEB SEARCH: triggered — {transcript[:80]}")
        live_results = await search_web(transcript)
    data = run_chat(
        session_id=session_id,
        incoming=[ChatMessage(role="user", content=transcript)],
        extra_context=headlines,
        web_results=live_results,
        user_tz=tz,
    )

    response_text = stop_at_first_turn(data.get("response", "") or "")
    response_text = enforce_spoken_only(response_text)

    # Context-driven voice persona
    context = data.get("context") or "default"
    tts_controls = select_tts_controls(context)

    # --- TTS ---
    audio_b64 = ""
    try:
        audio_out = cartesia_tts(response_text, speed=tts_controls["speed"], emotion=tts_controls.get("emotion"))
        audio_b64 = base64.b64encode(audio_out).decode("utf-8")
    except Exception as tts_err:
        import logging as _log
        _log.getLogger(__name__).warning("TTS failed (text-only response): %s", tts_err)

    return JSONResponse({
        "transcript": transcript,
        "response": response_text,
        "context": context,
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

