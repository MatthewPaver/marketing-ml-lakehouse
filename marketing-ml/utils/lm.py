from __future__ import annotations

import os
import json
import requests
from typing import Any, Dict, Tuple

DEFAULT_ENDPOINT = os.getenv("LM_ENDPOINT", "http://localhost:1234/v1/chat/completions")
DEFAULT_MODEL = os.getenv("LM_MODEL", "local")
LM_API_KEY = os.getenv("LM_API_KEY", "")

SYSTEM_PROMPT = (
    """
    You are a marketing performance copilot. Use ONLY the numbers in the JSON provided. Output EXACTLY 3 bullet actions (Decision, Rationale ≤ 20 words, Expected impact, Confidence). If data insufficient, say so.
    """
).strip()


def health_check(host: str = DEFAULT_ENDPOINT) -> Tuple[bool, str]:
    """Lightweight connectivity check to LM endpoint (OpenAI-compatible)."""
    try:
        r = requests.get(host.replace("/v1/chat/completions", "/v1/models"), timeout=3)
        if r.ok:
            return True, "online"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)


def _post(host: str, payload: Dict[str, Any]) -> Tuple[str, str]:
    headers = {"Content-Type": "application/json"}
    if LM_API_KEY:
        headers["Authorization"] = f"Bearer {LM_API_KEY}"
    resp = requests.post(host, headers=headers, data=json.dumps(payload), timeout=10)
    if not resp.ok:
        return ("", f"HTTP {resp.status_code}")
    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    return (content, f"OK len={len(content)}")


def call_lm_with_status(system_prompt: str, user_json: Dict[str, Any], host: str = DEFAULT_ENDPOINT, model: str = DEFAULT_MODEL) -> Tuple[str, str]:
    try:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": (system_prompt or SYSTEM_PROMPT)},
                {"role": "user", "content": json.dumps(user_json)},
            ],
            "temperature": 0.2,
            "max_tokens": 300,
        }
        # Try provided host
        content, status = _post(host, payload)
        if content:
            return (content, status)
        # Fallback: if localhost, try Docker host bridge
        if "localhost" in host or "127.0.0.1" in host:
            alt = host.replace("localhost", "host.docker.internal").replace("127.0.0.1", "host.docker.internal")
            content2, status2 = _post(alt, payload)
            if content2:
                return (content2, status2)
        return ("- No suggestions (LM offline) -", status)
    except Exception:
        return ("- No suggestions (LM offline) -", "error")
