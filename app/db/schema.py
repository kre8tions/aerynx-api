import sqlite3

def ensure_schema(conn: sqlite3.Connection) -> None:
    """
    Ensures the sessions table exists with the columns AERYNX expects:
      - summary (durable memory)
      - recent (short-term rolling messages as JSON string)
      - turns (turn counter)
      - updated_at (epoch seconds)
    Safe to run on every startup.
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

