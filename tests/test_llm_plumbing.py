"""Tests for rate limiting, metering, and JSON recovery.

None of these touch the network. They cover the machinery that has to behave
correctly during a long unattended extraction run, where a bug means either a
silent overspend or an hour of lost work.
"""

from __future__ import annotations

import time

import pytest
from pydantic import BaseModel

from graphrag.llm.base import Usage
from graphrag.llm.meter import SpendLimitExceeded, UsageMeter, cost_usd
from graphrag.llm.mistral_backend import MistralBackend
from graphrag.llm.rate_limit import RateLimitError, TokenBucket, with_retry

# --- TokenBucket ------------------------------------------------------


def test_bucket_allows_burst_then_throttles():
    bucket = TokenBucket(rate=50.0, burst=3)
    # The first `burst` acquisitions come from the banked capacity.
    assert all(bucket.acquire() == 0.0 for _ in range(3))
    # The next one must wait for a refill.
    assert bucket.acquire() > 0.0


def test_bucket_enforces_average_rate():
    bucket = TokenBucket(rate=20.0, burst=1)
    start = time.monotonic()
    for _ in range(5):
        bucket.acquire()
    elapsed = time.monotonic() - start
    # 5 calls at 20/s needs ~0.2s; allow generous slack for a loaded machine.
    assert elapsed >= 0.15


def test_bucket_rejects_nonpositive_rate():
    with pytest.raises(ValueError):
        TokenBucket(rate=0)


# --- retry ------------------------------------------------------------


class FakeRateLimit(Exception):
    status_code = 429


class FakeServerError(Exception):
    status_code = 503


def test_retry_recovers_after_transient_failures():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise FakeRateLimit("429 too many requests")
        return "done"

    assert with_retry(flaky, max_attempts=5, base_delay=0.01) == "done"
    assert calls["n"] == 3


def test_retry_gives_up_and_reports():
    def always_throttled():
        raise FakeRateLimit("429")

    with pytest.raises((RateLimitError, FakeRateLimit)):
        with_retry(always_throttled, max_attempts=3, base_delay=0.01)


def test_retry_does_not_mask_real_errors():
    """A 400 is a bug in our request -- retrying it just wastes quota."""

    class BadRequest(Exception):
        status_code = 400

    def bad():
        raise BadRequest("invalid schema")

    with pytest.raises(BadRequest):
        with_retry(bad, max_attempts=4, base_delay=0.01)


def test_retry_handles_5xx():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise FakeServerError("503")
        return "ok"

    assert with_retry(flaky, max_attempts=3, base_delay=0.01) == "ok"


# --- metering ---------------------------------------------------------


def test_free_tier_costs_nothing_but_still_counts_tokens():
    meter = UsageMeter()
    meter.record(
        stage="extract",
        model="mistral-large-latest",
        usage=Usage(input_tokens=4000, output_tokens=2500),
    )
    assert meter.total_cost == 0.0
    assert meter.total_tokens == 6500  # quota is about volume, not money
    assert meter.total_calls == 1


def test_paid_model_pricing_is_applied():
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost_usd("claude-haiku-4-5", usage) == pytest.approx(6.0)  # $1 in + $5 out


def test_cached_tokens_are_discounted():
    cached = cost_usd("claude-haiku-4-5", Usage(cache_read_tokens=1_000_000))
    fresh = cost_usd("claude-haiku-4-5", Usage(input_tokens=1_000_000))
    assert cached == pytest.approx(fresh * 0.10)


def test_unknown_model_does_not_halt_a_run():
    """A missing price should degrade the report, never kill an extraction."""
    assert cost_usd("some-future-model", Usage(input_tokens=1000)) == 0.0


def test_spend_guard_trips_before_the_next_call():
    meter = UsageMeter(spend_limit_usd=0.50)
    meter.record(
        stage="extract",
        model="claude-opus-5",
        usage=Usage(input_tokens=200_000),  # $1.00, over the ceiling
    )
    with pytest.raises(SpendLimitExceeded, match="exceeded the limit"):
        meter.check_budget()


def test_spend_guard_is_inert_when_unset():
    meter = UsageMeter(spend_limit_usd=None)
    meter.record(stage="x", model="claude-opus-5", usage=Usage(input_tokens=10_000_000))
    meter.check_budget()  # must not raise


def test_usage_addition_accumulates_every_field():
    total = Usage(input_tokens=1, output_tokens=2) + Usage(
        input_tokens=10, output_tokens=20, cache_read_tokens=5
    )
    assert (total.input_tokens, total.output_tokens, total.cache_read_tokens) == (11, 22, 5)
    assert total.total == 38


# --- JSON recovery ----------------------------------------------------


class Triple(BaseModel):
    subject: str
    predicate: str
    obj: str


@pytest.mark.parametrize(
    "raw",
    [
        '{"subject": "DPR", "predicate": "EVALUATES_ON", "obj": "NQ"}',
        '```json\n{"subject": "DPR", "predicate": "EVALUATES_ON", "obj": "NQ"}\n```',
        'Here is the result:\n{"subject": "DPR", "predicate": "EVALUATES_ON", "obj": "NQ"}\nDone.',
    ],
    ids=["bare", "fenced", "prose-wrapped"],
)
def test_recovers_json_from_common_model_wrappers(raw):
    got = MistralBackend._recover(raw, Triple)
    assert got is not None
    assert got.subject == "DPR"


@pytest.mark.parametrize("raw", ["", "no json here at all", '{"subject": "only"}'])
def test_recovery_returns_none_when_genuinely_unusable(raw):
    assert MistralBackend._recover(raw, Triple) is None


# --- retry detection against the real SDK exception --------------------
#
# The retry logic reads `status_code` off the provider's exception. That is an
# assumption about a third-party class, and if it were wrong every 503 would
# fail immediately instead of retrying -- with no signal except a higher
# failure count that looks like a flaky network.


def _sdk_error(status: int, body: str = '{"message":"boom"}'):
    """Construct a genuine mistralai SDKError with the given HTTP status."""
    import httpx
    from mistralai.client.errors.sdkerror import SDKError

    response = httpx.Response(
        status_code=status,
        headers={"content-type": "application/json"},
        text=body,
        request=httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions"),
    )
    return SDKError("API error occurred", response, body)


def test_the_real_sdk_error_exposes_status_code():
    """If this ever stops being true, retries silently stop working."""
    assert _sdk_error(503).status_code == 503
    assert _sdk_error(429).status_code == 429


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_transient_provider_errors_are_retried(status):
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise _sdk_error(status)
        return "recovered"

    assert with_retry(flaky, max_attempts=3, base_delay=0.01) == "recovered"
    assert calls["n"] == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_errors_are_not_retried(status):
    """A 4xx other than 429 is our bug; retrying just burns quota."""
    calls = {"n": 0}

    def bad():
        calls["n"] += 1
        raise _sdk_error(status)

    from mistralai.client.errors.sdkerror import SDKError

    with pytest.raises(SDKError):
        with_retry(bad, max_attempts=4, base_delay=0.01)
    assert calls["n"] == 1, "a non-retryable error must not be attempted twice"


def test_retry_after_header_is_honoured():
    """Backing off longer than the server asked wastes time; shorter is rude."""
    import httpx
    from mistralai.client.errors.sdkerror import SDKError

    from graphrag.llm.rate_limit import _retry_after_seconds

    response = httpx.Response(
        status_code=429,
        headers={"content-type": "application/json", "Retry-After": "7"},
        text="{}",
        request=httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions"),
    )
    assert _retry_after_seconds(SDKError("", response, "{}")) == 7.0


# --- scoped metering ---------------------------------------------------
#
# The benchmark and the API both swap in a scoped meter so usage is attributed
# to one run or one request. That only works if the backend reads METER through
# its module. A `from ... import METER` binds the object at import time, and the
# swap would silently write to the original -- reporting zero tokens for every
# run, with no error and a whole column of zeros in the cost table.


def test_backend_reads_the_meter_through_its_module():
    """Guards the import style, not just the behaviour."""
    import inspect

    from graphrag.llm import mistral_backend

    source = inspect.getsource(mistral_backend)
    assert "from graphrag.llm.meter import METER" not in source, (
        "a direct METER import breaks scoped metering"
    )
    assert "meter_module.METER" in source


def test_swapping_the_meter_captures_usage_in_the_new_one():
    from graphrag.llm import meter as meter_module

    scoped = UsageMeter()
    previous, meter_module.METER = meter_module.METER, scoped
    try:
        # Simulate what the backend's _call does.
        meter_module.METER.record(
            stage="answer",
            model="mistral-medium-latest",
            usage=Usage(input_tokens=100, output_tokens=50),
        )
    finally:
        meter_module.METER = previous

    assert scoped.total_tokens == 150
    assert scoped.total_calls == 1
    # The process-wide meter must not have absorbed the scoped run.
    assert previous is meter_module.METER


# --- scoped() ----------------------------------------------------------


def test_scoped_isolates_usage_from_the_global_meter():
    from graphrag.llm import meter as meter_module
    from graphrag.llm.meter import scoped

    meter_module.METER.record(
        stage="x", model="mistral-medium-latest", usage=Usage(input_tokens=10)
    )
    before = meter_module.METER.total_tokens

    with scoped() as inner:
        meter_module.METER.record(
            stage="y", model="mistral-medium-latest", usage=Usage(input_tokens=500)
        )
        assert inner.total_tokens == 500

    assert meter_module.METER.total_tokens == before, "scoped usage leaked into the global meter"


def test_scoped_restores_the_previous_meter_even_on_error():
    from graphrag.llm import meter as meter_module
    from graphrag.llm.meter import scoped

    original = meter_module.METER
    with pytest.raises(RuntimeError):
        with scoped():
            raise RuntimeError("boom")
    assert meter_module.METER is original


def test_scoped_inherits_the_spend_limit():
    """A scoped run with no ceiling would silently disable the spend guard.

    The limit is configured on the global meter when the backend is built, so a
    bare UsageMeter() inside the scope would default to unlimited -- exactly
    during a benchmark, which is the longest unattended run there is.
    """
    from graphrag.llm import meter as meter_module
    from graphrag.llm.meter import scoped

    original = meter_module.METER
    meter_module.METER = UsageMeter(spend_limit_usd=5.0)
    try:
        with scoped() as inner:
            assert inner.spend_limit_usd == 5.0
    finally:
        meter_module.METER = original


def test_scoped_meters_nest():
    from graphrag.llm import meter as meter_module
    from graphrag.llm.meter import scoped

    with scoped() as outer:
        meter_module.METER.record(stage="a", model="m", usage=Usage(input_tokens=1))
        with scoped() as inner:
            meter_module.METER.record(stage="b", model="m", usage=Usage(input_tokens=2))
            assert inner.total_tokens == 2
        assert outer.total_tokens == 1
