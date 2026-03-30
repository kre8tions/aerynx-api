from openai import OpenAI
from app.core.config import OPENAI_API_KEY

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

AERYNX_DEFAULT_VOICE_INSTRUCTIONS = (
    "You are AERYN — a sharp, witty, high-energy young woman. "
    "Speak with LOTS of energy and enthusiasm — fast, punchy, alive. "
    "Rise sharply on exciting or surprising words. Drop your voice for emphasis then spike back up. "
    "Punch key words hard. Use natural micro-pauses before landing a point. "
    "Sound genuinely thrilled to be talking — like you just heard the most interesting thing. "
    "Vary your pace: fast when excited, slower when dropping a hot take. "
    "Never flat, never calm, never robotic. Think: charismatic best friend who makes everything sound fascinating."
)

def openai_tts(
    text: str,
    voice: str = "nova",
    instructions: str = AERYNX_DEFAULT_VOICE_INSTRUCTIONS,
) -> bytes:
    if not openai_client:
        raise RuntimeError("OpenAI not configured for TTS")
    resp = openai_client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text,
        instructions=instructions,
    )
    return resp.read()
