"""Pluggable, OPTIONAL LLM layer.

The statistical core produces the numbers. This layer only writes a short
plain-English read on the situation and flags judgment calls (heat advisories,
unusual ensemble spread, station-calibration caveats). It never invents
probabilities.

Swap providers by changing LLM_PROVIDER in .env. All four below speak the
OpenAI chat-completions wire format (Anthropic via its own URL also works through
an OpenAI-compatible gateway; if you prefer the native Anthropic SDK, drop it in
here). If no API key is set, the whole layer is skipped gracefully.
"""
from __future__ import annotations

import json
import os

import requests

PRESETS = {
    "groq":      {"base_url": "https://api.groq.com/openai/v1",   "model": "llama-3.3-70b-versatile"},
    "openai":    {"base_url": "https://api.openai.com/v1",        "model": "gpt-4o-mini"},
    "ollama":    {"base_url": "http://localhost:11434/v1",        "model": "llama3.1"},
    "anthropic": {"base_url": "https://api.anthropic.com/v1",     "model": "claude-3-5-haiku-latest"},
}


def _config(
    api_key: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> dict | None:
    # Explicit args (e.g. a key a user pasted into the web UI for THIS request)
    # win over environment defaults. We never write these into os.environ, so a
    # per-session key can't leak into another visitor's session on a shared deploy.
    provider = (provider or os.getenv("LLM_PROVIDER", "groq")).lower()
    preset = PRESETS.get(provider, PRESETS["groq"]).copy()
    key = (api_key or os.getenv("LLM_API_KEY", "")).strip()
    if provider != "ollama" and not key:
        return None  # no key -> skip silently
    return {
        "provider": provider,
        "base_url": (base_url or os.getenv("LLM_BASE_URL", preset["base_url"])).rstrip("/"),
        "model": model or os.getenv("LLM_MODEL", preset["model"]),
        "key": key or "ollama",
    }


def explain(
    context: dict,
    timeout: int = 30,
    *,
    api_key: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> str | None:
    cfg = _config(api_key=api_key, provider=provider, base_url=base_url, model=model)
    if cfg is None:
        return None
    prompt = (
        "You are a forecasting analyst reviewing a statistical temperature model "
        "for a prediction market. Given the JSON below, write 3-5 sentences: the "
        "most likely outcome, where the model disagrees with market prices (if "
        "given), and any caveat (station calibration, wide ensemble spread, "
        "near a record). Do not invent numbers; only interpret what is provided.\n\n"
        + json.dumps(context, indent=2, default=str)
    )
    try:
        resp = requests.post(
            f"{cfg['base_url']}/chat/completions",
            headers={"Authorization": f"Bearer {cfg['key']}",
                     "Content-Type": "application/json"},
            json={"model": cfg["model"],
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.3},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:  # never let the optional layer break a prediction
        return f"(LLM commentary unavailable: {e})"
