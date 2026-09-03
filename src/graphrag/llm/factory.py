"""Backend selection.

The single place that knows which provider is in use. Everything else asks for
an ``LLMBackend`` and stays provider-agnostic, so switching costs one setting.
"""

from __future__ import annotations

from functools import lru_cache

from graphrag.config import settings
from graphrag.llm.base import LLMBackend, LLMError
from graphrag.llm.meter import METER


@lru_cache(maxsize=2)
def get_backend(name: str | None = None) -> LLMBackend:
    """Return the configured backend. Cached -- one client per process."""
    name = (name or settings.llm_backend).lower()

    # The guard is only meaningful for a paid backend; on a free tier a dollar
    # ceiling would fire at zero spend and block a run for no reason.
    METER.spend_limit_usd = (
        settings.max_extraction_spend_usd if name == "anthropic" else None
    )

    if name == "mistral":
        from graphrag.llm.mistral_backend import MistralBackend

        return MistralBackend(
            api_key=settings.mistral_api_key or "",
            rps=settings.llm_rps,
            max_attempts=settings.llm_max_attempts,
        )

    if name == "anthropic":
        try:
            from graphrag.llm.anthropic_backend import AnthropicBackend
        except ImportError as exc:  # pragma: no cover - optional extra
            raise LLMError(
                "The anthropic backend needs its optional dependency: "
                'pip install -e ".[anthropic]"'
            ) from exc

        return AnthropicBackend(
            api_key=settings.anthropic_api_key or "",
            rps=settings.llm_rps,
            max_attempts=settings.llm_max_attempts,
        )

    raise LLMError(f"Unknown LLM_BACKEND {name!r}. Expected 'mistral' or 'anthropic'.")
