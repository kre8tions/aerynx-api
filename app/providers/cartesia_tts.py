# app/providers/cartesia_tts.py
from __future__ import annotations

import os
import httpx

CARTESIA_API_KEY  = os.getenv("CARTESIA_API_KEY", "")
# Browse voices at https://play.cartesia.ai — set CARTESIA_VOICE_ID in env to override.
# Default: "California Girl" — energetic, young American female, suits AERYN well.
CARTESIA_VOICE_ID = os.getenv("CARTESIA_VOICE_ID", "b7d50908-b17c-442d-ad8d-810c63997ed9")
CARTESIA_MODEL    = os.getenv("CARTESIA_MODEL", "sonic-2")


def cartesia_tts(
    text: str,
    speed: str = "fast",
    emotion: list[str] | None = None,
) -> bytes:
    """
    Call Cartesia Sonic TTS and return raw MP3 bytes.

    speed:   "slowest" | "slow" | "normal" | "fast" | "fastest"
    emotion: list of "emotion:level" tags, e.g. ["positivity:high", "curiosity:high"]
             Valid emotions: positivity, negativity, curiosity, surprise, anger, sadness
             Valid levels:   lowest, low, high, highest
    """
    if not CARTESIA_API_KEY:
        raise RuntimeError("CARTESIA_API_KEY not set")

    voice_payload: dict = {"mode": "id", "id": CARTESIA_VOICE_ID}
    controls: dict = {"speed": speed}
    if emotion:
        controls["emotion"] = emotion
    voice_payload["experimental_controls"] = controls

    resp = httpx.post(
        "https://api.cartesia.ai/tts/bytes",
        headers={
            "X-API-Key": CARTESIA_API_KEY,
            "Cartesia-Version": "2024-06-10",
            "Content-Type": "application/json",
        },
        json={
            "model_id": CARTESIA_MODEL,
            "transcript": text,
            "voice": voice_payload,
            "output_format": {
                "container": "mp3",
                "encoding": "mp3",
                "sample_rate": 44100,
            },
            "language": "en",
        },
        timeout=15.0,
    )
    resp.raise_for_status()
    return resp.content
