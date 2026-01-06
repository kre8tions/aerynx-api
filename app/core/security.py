from typing import Optional
from fastapi import HTTPException
from app.core.config import API_KEY

def require_api_key(authorization: Optional[str]) -> None:
    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Server misconfigured: AERYNX_API_KEY missing",
        )

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    token = authorization.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid token")
