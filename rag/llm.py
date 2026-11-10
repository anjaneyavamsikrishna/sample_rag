"""Thin wrapper around an OpenAI-compatible chat API.
Works with OpenAI, Ollama (local, free), and any provider that offers an OpenAI-compatible endpoint:
just change OPENAI_BASE_URL / OPENAI_API_KEY / LLM_MODEL in .env."""
from functools import lru_cache

import config


def is_configured() -> bool:
    """True if .env points at a usable LLM (a real key, or a custom base URL such as Ollama)."""
    return bool(config.OPENAI_BASE_URL) or config.OPENAI_API_KEY not in ("", "your_api_key_here")


@lru_cache(maxsize=1)
def _client():
    from openai import OpenAI          # imported lazily so the rest of the project works without it
    return OpenAI(api_key=config.OPENAI_API_KEY or "none", base_url=config.OPENAI_BASE_URL,
                  timeout=60, max_retries=2)


def chat(system: str, user: str, temperature: float = 0.0) -> str:
    """One question in, one text answer out. Temperature 0 = as deterministic as possible."""
    response = _client().chat.completions.create(
        model=config.LLM_MODEL, temperature=temperature,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return (response.choices[0].message.content or "").strip()
