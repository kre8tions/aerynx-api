import requests
from app.core.config import TGI_URL, TGI_TIMEOUT, TOP_P

def call_tgi(prompt: str, temperature: float) -> str:
    r = requests.post(
        TGI_URL,
        json={
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 220,
                "temperature": temperature,
                "top_p": TOP_P,
                "repetition_penalty": 1.25,
                "stop": [
                    "\nUSER:",
                    "\nASSISTANT:",
                    "\nSYSTEM:",
                    "\nAERYNX:",
                    "\n\n",
                ],
            },
        },
        timeout=TGI_TIMEOUT,
    )
    r.raise_for_status()
    j = r.json()
    return (j.get("generated_text") or "").strip()
