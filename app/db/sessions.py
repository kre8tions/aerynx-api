import time
import json
import sqlite3
from typing import Tuple, List, Dict

from app.db.connection import connect_db

Message = Dict[str, str]

def get_session(session_id: str) -> Tuple[str, List[Message], int]:
    """
    Returns (summary, recent_messages, turns).
    recent_messages is a list of {"role": "...", "content": "..."} dicts.
    """
    conn = connect_db()
    cur = conn.execute(
        "SELECT summary, recent, turns FROM sessions WHERE session_id=?",
        (session_id,),
    )
    row = cur.fetchone()

    if not row:
        conn.execute(
            """
            INSERT INTO sessions(session_id, summary, recent, turns, updated_at)
            VALUES(?,?,?,?,?)
            """,
            (session_id, "", "[]", 0, int(time.time())),
        )
        conn.commit()
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

    # normalize shape defensively
    cleaned: List[Message] = []
    for m in recent:
        if isinstance(m, dict) and "role" in m and "content" in m:
            cleaned.append({"role": str(m["role"]), "content": str(m["content"])})
    return summary, cleaned, turns


def set_session(session_id: str, summary: str, recent: List[Message], turns: int) -> None:
    conn = connect_db()
    conn.execute(
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
    conn.commit()

