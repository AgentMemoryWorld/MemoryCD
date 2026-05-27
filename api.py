"""
LLM client configuration: OpenAI (official API) or OpenRouter.

Environment:
  LLM_PROVIDER       — "openai" (default) or "openrouter". If unset, uses
                       openrouter when OPENROUTER_API_KEY is set, else OpenAI.

  OpenAI:
    OPENAI_API_KEY     — required for provider openai
    OPENAI_BASE_URL    — optional, default https://api.openai.com/v1

  OpenRouter:
    OPENROUTER_API_KEY — required for provider openrouter
    OPENROUTER_BASE_URL — optional, default https://openrouter.ai/api/v1

  LLM_MODEL — optional; defaults depend on provider (see below).
"""

from __future__ import annotations

import os
from typing import Literal

from openai import OpenAI

Provider = Literal["openai", "openrouter"]

DEFAULT_OPENAI_BASE = "https://api.openai.com/v1"
DEFAULT_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
DEFAULT_MODEL_OPENAI = "gpt-5-nano"
DEFAULT_MODEL_OPENROUTER = "openai/gpt-5-nano"


def _resolve_provider() -> Provider:
    explicit = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if explicit in ("openai", "openrouter"):
        return explicit  # type: ignore[return-value]
    if os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        return "openrouter"
    return "openai"


def create_client(
    provider: Provider | None = None,
) -> tuple[OpenAI, Provider]:
    """Build an OpenAI SDK client for the chosen API (OpenAI or OpenRouter-compatible)."""
    p = provider or _resolve_provider()

    if p == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE)
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required when LLM_PROVIDER=openrouter "
                "(or when only OpenRouter credentials are intended)."
            )
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE)
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when using the OpenAI provider. "
                "Set LLM_PROVIDER=openrouter and OPENROUTER_API_KEY to use OpenRouter instead."
            )

    return OpenAI(base_url=base_url, api_key=api_key), p


def get_default_model(provider: Provider) -> str:
    env_model = (os.getenv("LLM_MODEL") or "").strip()
    if env_model:
        return env_model
    return DEFAULT_MODEL_OPENROUTER if provider == "openrouter" else DEFAULT_MODEL_OPENAI


_provider: Provider | None = None
_gateway_client: OpenAI | None = None


def get_gateway_client() -> OpenAI:
    """Lazy singleton used by scripts that expect a shared client."""
    global _provider, _gateway_client
    if _gateway_client is None:
        _gateway_client, _provider = create_client()
    return _gateway_client


def get_active_provider() -> Provider:
    """Provider used by get_gateway_client() after first call."""
    get_gateway_client()
    assert _provider is not None
    return _provider


# Lazy proxy so `gateway_client.chat.completions.create(...)` works without eager init.
class _LazyClient:
    def __getattr__(self, name: str):
        return getattr(get_gateway_client(), name)

    def __call__(self, *args, **kwargs):
        raise TypeError("Use get_gateway_client() or attribute access on the lazy proxy")


gateway_client = _LazyClient()  # type: ignore[assignment]


if __name__ == "__main__":
    client, prov = create_client()
    model = get_default_model(prov)
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        model=model,
    )
    print(chat_completion.choices[0].message.content)
