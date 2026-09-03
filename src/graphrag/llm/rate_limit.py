"""Rate limiting and retry.

Mistral's free tier does not publish its limits, and they differ per account.
Rather than hard-code a guess, we throttle to a configurable rate and treat 429
as normal operating feedback instead of an error. Combined with checkpointed
extraction, this makes an unknown or low rate limit cost wall-clock time only,
never lost work.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from typing import TypeVar

from graphrag.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T")


class TokenBucket:
    """Thread-safe token bucket limiting calls to ``rate`` per second.

    ``burst`` lets a short idle period bank a few calls, which smooths out the
    stop-start pattern of a pipeline that interleaves API calls with local work.
    """

    def __init__(self, rate: float, burst: int | None = None) -> None:
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        self.rate = rate
        self.capacity = float(burst if burst is not None else max(1, int(rate)))
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """Block until a call is permitted. Returns seconds spent waiting."""
        with self._lock:
            now = time.monotonic()
            self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
            self._last = now

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return 0.0

            wait = (1.0 - self._tokens) / self.rate
            self._tokens = 0.0
            self._last = now + wait

        time.sleep(wait)
        return wait


class RateLimitError(RuntimeError):
    """A 429 that survived every retry."""


def _retry_after_seconds(exc: Exception) -> float | None:
    """Pull a Retry-After hint off a provider exception, if it carries one.

    Provider SDKs expose headers inconsistently, so we probe a few shapes and
    fall back to exponential backoff when none of them match.
    """
    headers = (
        getattr(exc, "headers", None)
        or getattr(getattr(exc, "response", None), "headers", None)
        or {}
    )
    try:
        raw = headers.get("Retry-After") or headers.get("retry-after")
    except AttributeError:
        return None
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _is_rate_limit(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    if status == 429:
        return True
    # Some SDKs surface it only in the message.
    return "429" in str(exc) or "rate limit" in str(exc).lower()


def _is_retryable(exc: Exception) -> bool:
    if _is_rate_limit(exc):
        return True
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    if isinstance(status, int) and status >= 500:
        return True
    return isinstance(exc, (ConnectionError, TimeoutError))


def with_retry(
    fn: Callable[[], T],
    *,
    max_attempts: int = 6,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    label: str = "llm call",
) -> T:
    """Run ``fn``, retrying rate limits and transient failures.

    Honours ``Retry-After`` when the provider sends one; otherwise uses
    exponential backoff with full jitter, which avoids a thundering herd when
    several workers are throttled at the same moment.
    """
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - re-raised below if not retryable
            last_exc = exc
            if not _is_retryable(exc) or attempt == max_attempts:
                raise

            hinted = _retry_after_seconds(exc)
            if hinted is not None:
                delay = min(hinted, max_delay)
            else:
                delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                delay = random.uniform(0, delay)  # full jitter

            log.warning(
                "retrying",
                label=label,
                attempt=attempt,
                max_attempts=max_attempts,
                sleep_s=round(delay, 2),
                reason=type(exc).__name__,
            )
            time.sleep(delay)

    assert last_exc is not None
    raise RateLimitError(f"{label} failed after {max_attempts} attempts") from last_exc
