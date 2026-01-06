import os

API_KEY = os.getenv("AERYNX_API_KEY")

DB_PATH = os.getenv(
    "AERYNX_DB_PATH",
    "/home/ubuntu/aerynx-api/aerynx_memory.sqlite3",
)

TGI_URL = os.getenv("TGI_URL", "http://127.0.0.1:3000/generate")
TGI_HEALTH_URL = os.getenv("TGI_HEALTH_URL", "http://127.0.0.1:3000/health")
TGI_TIMEOUT = float(os.getenv("TGI_TIMEOUT", "45"))

AERYNX_TEMP = float(os.getenv("AERYNX_TEMP", "0.55"))
GROQ_TEMP = float(os.getenv("GROQ_TEMP", "0.75"))
TOP_P = float(os.getenv("TOP_P", "0.9"))

# Backward-compatible aliases
AERYNX_TEMP_BASE = AERYNX_TEMP
GROQ_TEMP_BASE = GROQ_TEMP

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
