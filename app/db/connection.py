import sqlite3
from app.core.config import DB_PATH

_DB = None

def connect_db() -> sqlite3.Connection:
    global _DB
    if _DB is None:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        _DB = conn
    return _DB

def close_db() -> None:
    global _DB
    if _DB:
        _DB.close()
    _DB = None

