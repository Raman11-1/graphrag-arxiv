"""Backend-agnostic LLM interface.

Every stage of the pipeline talks to an ``LLMBackend``. Concrete backends live
alongside this file (``mistral_backend.py``, ``anthropic_backend.py``). Nothing
outside ``graphrag.llm`` may import a provider SDK directly -- that is what
keeps the backend swappable via one setting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class Usage:
    """Token counts for a single call.

    ``input_tokens`` excludes cached tokens; providers report those separately
    and adding them would double-count against the cost estimate.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )

    @property
    def total(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )


@dataclass
class LLMResponse(Generic[T]):
    """A completed call: the text, the parsed object (if any), and what it cost."""

    text: str
    usage: Usage
    model: str
    parsed: T | None = None
    raw: object | None = field(default=None, repr=False)


class LLMError(RuntimeError):
    """Any backend failure the pipeline is expected to handle."""


class SchemaValidationError(LLMError):
    """The model returned output that did not satisfy the requested schema.

    Raised only after the backend has exhausted its own retries, so callers can
    treat it as a genuine extraction failure and record it rather than crash.
    """


class LLMBackend(Protocol):
    """What every stage of the pipeline is allowed to assume about an LLM."""

    name: str

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float | None = None,
    ) -> LLMResponse[BaseModel]:
        """Free-form text completion."""
        ...

    def parse(
        self,
        *,
        system: str,
        user: str,
        model: str,
        schema: type[T],
        max_tokens: int = 4096,
    ) -> LLMResponse[T]:
        """Structured completion validated against ``schema``.

        Backends with native schema enforcement (Mistral's ``chat.parse``,
        Anthropic's ``messages.parse``) use it. Backends without it prompt for
        JSON and validate locally, retrying once on a validation failure.
        """
        ...
