from typing import List, Dict
from groq import Groq
from app.core.config import GROQ_API_KEY

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

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
