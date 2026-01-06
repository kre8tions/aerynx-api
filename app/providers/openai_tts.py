from openai import OpenAI
from app.core.config import OPENAI_API_KEY

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def openai_tts(text: str, voice: str = "alloy") -> bytes:
    if not openai_client:
        raise RuntimeError("OpenAI not configured for TTS")

    resp = openai_client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text,
    )
    return resp.read()
