import sqlite3

conn = sqlite3.connect("memory.db", check_same_thread=False)
conn.execute("""
CREATE TABLE IF NOT EXISTS memory (
  session_id TEXT PRIMARY KEY,
  summary TEXT
)
""")

def get_memory(session_id):
    row = conn.execute(
        "SELECT summary FROM memory WHERE session_id=?",
        (session_id,)
    ).fetchone()
    return row[0] if row else ""

def save_memory(session_id, summary):
    conn.execute(
        "REPLACE INTO memory (session_id, summary) VALUES (?,?)",
        (session_id, summary)
    )
    conn.commit()
