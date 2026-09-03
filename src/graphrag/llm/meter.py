"""Usage metering and the spend guard.

Every LLM call is recorded here. Two reasons this exists rather than being an
afterthought:

1. On a free tier we need to know how close we are to a quota we cannot see.
2. On a paid tier an unattended extraction run over hundreds of windows is
   exactly the shape of job that produces a surprise bill.

``--dry-run`` on the pipeline uses the same pricing table to project a cost
before anything is spent.
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from graphrag.llm.base import Usage
from graphrag.logging import get_logger

log = get_logger(__name__)

# USD per million tokens. Free-tier models are 0.0 but still metered, because
# token *volume* is what a free quota limits.
PRICING: dict[str, tuple[float, float]] = {
    # --- Mistral (free Experiment tier -> no charge, still counted) ---
    "mistral-large-latest": (0.0, 0.0),
    "mistral-medium-latest": (0.0, 0.0),
    "mistral-small-latest": (0.0, 0.0),
    # --- Anthropic (optional ablation backend), USD/MTok ---
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10


class SpendLimitExceeded(RuntimeError):
    """Raised before a call that would push spend past the configured ceiling."""


def cost_usd(model: str, usage: Usage) -> float:
    """Cost of one call. Unknown models cost 0 but emit a warning.

    We deliberately do not raise on an unknown model: a missing price should
    degrade the report, not halt a long extraction run.
    """
    price = PRICING.get(model)
    if price is None:
        log.warning("no_pricing_entry", model=model)
        return 0.0

    in_rate, out_rate = (p / 1_000_000 for p in price)
    return (
        usage.input_tokens * in_rate
        + usage.cache_write_tokens * in_rate * CACHE_WRITE_MULTIPLIER
        + usage.cache_read_tokens * in_rate * CACHE_READ_MULTIPLIER
        + usage.output_tokens * out_rate
    )


@dataclass
class StageStats:
    calls: int = 0
    usage: Usage = field(default_factory=Usage)
    cost: float = 0.0


class UsageMeter:
    """Accumulates usage across a run, sliced by pipeline stage."""

    def __init__(self, spend_limit_usd: float | None = None) -> None:
        self.spend_limit_usd = spend_limit_usd
        self._stages: dict[str, StageStats] = defaultdict(StageStats)
        self._lock = threading.Lock()

    def record(self, *, stage: str, model: str, usage: Usage) -> float:
        """Record a completed call. Returns its cost in USD."""
        c = cost_usd(model, usage)
        with self._lock:
            s = self._stages[stage]
            s.calls += 1
            s.usage = s.usage + usage
            s.cost += c
        return c

    def check_budget(self) -> None:
        """Raise if we have already exceeded the ceiling.

        Called *before* each request, so the limit stops the next call rather
        than being noticed after the money is gone.
        """
        if self.spend_limit_usd is None:
            return
        if self.total_cost > self.spend_limit_usd:
            raise SpendLimitExceeded(
                f"Spend ${self.total_cost:.2f} exceeded the limit of "
                f"${self.spend_limit_usd:.2f}. Raise MAX_EXTRACTION_SPEND_USD or "
                f"switch to a cheaper model to continue."
            )

    @property
    def total_cost(self) -> float:
        with self._lock:
            return sum(s.cost for s in self._stages.values())

    @property
    def total_tokens(self) -> int:
        with self._lock:
            return sum(s.usage.total for s in self._stages.values())

    @property
    def total_calls(self) -> int:
        with self._lock:
            return sum(s.calls for s in self._stages.values())

    def summary(self) -> dict[str, dict[str, float | int]]:
        with self._lock:
            return {
                stage: {
                    "calls": s.calls,
                    "input_tokens": s.usage.input_tokens,
                    "output_tokens": s.usage.output_tokens,
                    "cache_read_tokens": s.usage.cache_read_tokens,
                    "total_tokens": s.usage.total,
                    "cost_usd": round(s.cost, 4),
                }
                for stage, s in sorted(self._stages.items())
            }

    def log_summary(self) -> None:
        for stage, stats in self.summary().items():
            log.info("usage", stage=stage, **stats)
        log.info(
            "usage_total",
            calls=self.total_calls,
            tokens=self.total_tokens,
            cost_usd=round(self.total_cost, 4),
        )

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "stages": self.summary(),
                    "total_calls": self.total_calls,
                    "total_tokens": self.total_tokens,
                    "total_cost_usd": round(self.total_cost, 4),
                },
                indent=2,
            ),
            encoding="utf-8",
        )


# Process-wide meter. Stages read this through the module (`meter.METER`), never
# via `from ... import METER` -- a direct import binds the object at import time
# and would keep writing to it after `scoped()` swaps in a different one.
METER = UsageMeter()


@contextmanager
def scoped() -> Iterator[UsageMeter]:
    """Temporarily install a fresh meter, then restore the previous one.

    Used wherever usage must be attributed to one unit of work rather than the
    whole process: a benchmark run, an API request, a UI query.

    The new meter **inherits the spend limit** from the one it replaces.
    Without that, a scoped run would silently have no spend guard -- the
    ceiling is configured on the global meter at backend construction, and a
    bare ``UsageMeter()`` defaults to no limit.
    """
    global METER
    previous = METER
    METER = UsageMeter(spend_limit_usd=previous.spend_limit_usd)
    try:
        yield METER
    finally:
        METER = previous
