"""Thin wrapper around Gemini (free tier) for answer generation.

Uses the current `google-genai` SDK (the older `google-generativeai`
package is deprecated).
"""
from google import genai
from app import config

_client = None


def _get_client():
    global _client
    if _client is None:
        if not config.GOOGLE_API_KEY or "dummy" in config.GOOGLE_API_KEY:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Copy .env.example to .env and add a free "
                "key from https://aistudio.google.com/apikey"
            )
        _client = genai.Client(api_key=config.GOOGLE_API_KEY)
    return _client


def generate(prompt: str) -> str:
    client = _get_client()
    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=prompt,
    )
    return (response.text or "").strip()
