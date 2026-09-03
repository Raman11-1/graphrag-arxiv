"""Mistral backend.

Verified against mistralai 2.9.4 by introspection, not from memory:

* ``from mistralai.client import Mistral`` -- 2.x is a namespace package, so
  the older ``from mistralai import Mistral`` raises ImportError.
* ``client.chat.parse(response_format=<PydanticModel>, **kwargs)`` returns a
  ``ParsedChatCompletionResponse``; the validated object is on
  ``response.choices[0].message.parsed``.
* ``response.usage`` exposes ``prompt_tokens`` / ``completion_tokens`` only --
  there are no cache-token fields, so cache accounting stays zero here.
"""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

# Imported as a module, not as `from ... import METER`. A direct import
# binds the object at import time, so callers that swap in a scoped meter
# (the benchmark, per-request API metering) would rebind the module
# attribute while this file kept writing to the original -- reporting zero
# tokens for every run with no error to show for it.
from graphrag.llm import meter as meter_module
from graphrag.llm.base import (
    LLMError,
    LLMResponse,
    SchemaValidationError,
    Usage,
)
from graphrag.llm.rate_limit import TokenBucket, with_retry
from graphrag.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


def _usage_from(response: Any) -> Usage:
    u = getattr(response, "usage", None)
    if u is None:
        return Usage()
    return Usage(
        input_tokens=getattr(u, "prompt_tokens", 0) or 0,
        output_tokens=getattr(u, "completion_tokens", 0) or 0,
    )


class MistralBackend:
    """LLMBackend over Mistral's La Plateforme.

    The free Experiment tier does not publish its rate limits and they vary per
    account, so ``rps`` is configurable and every call goes through a token
    bucket plus 429-aware retry. Combined with checkpointed extraction, an
    unknown limit costs wall-clock time rather than lost work.
    """

    name = "mistral"

    def __init__(
        self,
        api_key: str,
        *,
        rps: float = 1.0,
        max_attempts: int = 6,
        timeout_ms: int = 120_000,
    ) -> None:
        if not api_key:
            raise LLMError(
                "MISTRAL_API_KEY is not set. Create a key at console.mistral.ai "
                "and add it to your .env file."
            )
        from mistralai.client import Mistral  # namespace package, see module docstring

        self._client = Mistral(api_key=api_key)
        self._bucket = TokenBucket(rate=rps)
        self._max_attempts = max_attempts
        self._timeout_ms = timeout_ms

    # -- internals ------------------------------------------------------

    def _messages(self, system: str, user: str) -> list[dict[str, str]]:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        return msgs

    def _call(self, fn, *, stage: str, model: str):
        """Throttle, budget-check, execute with retry, then record usage."""
        meter_module.METER.check_budget()
        self._bucket.acquire()
        response = with_retry(fn, max_attempts=self._max_attempts, label=f"mistral {stage}")
        usage = _usage_from(response)
        meter_module.METER.record(stage=stage, model=model, usage=usage)
        return response, usage

    # -- LLMBackend -----------------------------------------------------

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float | None = None,
        stage: str = "complete",
        cache_key: str | None = None,
    ) -> LLMResponse[BaseModel]:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._messages(system, user),
            "max_tokens": max_tokens,
            "timeout_ms": self._timeout_ms,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if cache_key:
            kwargs["prompt_cache_key"] = cache_key

        response, usage = self._call(
            lambda: self._client.chat.complete(**kwargs), stage=stage, model=model
        )

        content = response.choices[0].message.content if response.choices else ""
        if not isinstance(content, str):
            # Multi-part content comes back as a list of chunks.
            content = "".join(getattr(part, "text", "") for part in (content or []))

        return LLMResponse(text=content or "", usage=usage, model=model, raw=response)

    def parse(
        self,
        *,
        system: str,
        user: str,
        model: str,
        schema: type[T],
        max_tokens: int = 4096,
        stage: str = "parse",
        cache_key: str | None = None,
    ) -> LLMResponse[T]:
        """Structured call validated against ``schema``.

        Uses Mistral's native schema enforcement. If the model still returns
        something unparseable, we retry once with the validation error fed back
        before giving up -- extraction over hundreds of windows will hit this
        occasionally and must degrade to a recorded failure, not a crash.
        """
        # We deliberately do NOT use client.chat.parse(). That helper is a thin
        # wrapper -- build schema, call complete, json.loads the content -- and
        # its json.loads raises before any of our recovery can run. When the
        # model hits max_tokens the JSON is simply cut off mid-string, and the
        # resulting JSONDecodeError says "Expecting value: line 1 column 24817"
        # rather than "your output was truncated", which is actively misleading.
        # Owning the parse lets us detect truncation and salvage what we can.
        from mistralai.extra.utils import response_format_from_pydantic_model

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._messages(system, user),
            "response_format": response_format_from_pydantic_model(schema),
            "max_tokens": max_tokens,
            "timeout_ms": self._timeout_ms,
        }
        if cache_key:
            kwargs["prompt_cache_key"] = cache_key

        response, usage = self._call(
            lambda: self._client.chat.complete(**kwargs), stage=stage, model=model
        )

        choice = response.choices[0] if response.choices else None
        message = getattr(choice, "message", None)
        raw_text = getattr(message, "content", "") or ""
        if not isinstance(raw_text, str):
            raw_text = ""

        if getattr(choice, "finish_reason", None) == "length":
            raise SchemaValidationError(
                f"{model} hit the {max_tokens}-token output limit and returned "
                f"truncated JSON ({len(raw_text)} chars). Raise max_tokens, or "
                f"reduce how much this window is asked to produce."
            )

        recovered = self._recover(raw_text, schema)
        if recovered is not None:
            return LLMResponse(
                text=raw_text, usage=usage, model=model, parsed=recovered, raw=response
            )

        raise SchemaValidationError(
            f"{model} did not return valid {schema.__name__}. "
            f"First 300 chars of response: {raw_text[:300]!r}"
        )

    @staticmethod
    def _recover(text: str, schema: type[T]) -> T | None:
        """Best-effort salvage of a JSON object from a text response."""
        if not text.strip():
            return None
        candidates = [text]
        # Strip a markdown fence if the model wrapped its JSON in one.
        if "```" in text:
            inner = text.split("```")
            candidates.extend(part.removeprefix("json").strip() for part in inner[1::2])
        # Fall back to the outermost brace pair.
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])

        for candidate in candidates:
            try:
                return schema.model_validate(json.loads(candidate))
            except (json.JSONDecodeError, ValidationError, TypeError):
                continue
        return None
