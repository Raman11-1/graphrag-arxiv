"""Measure the LLM backend's real throughput before committing a pipeline to it.

Mistral stopped publishing free-tier rate limits and they vary per account, so
guessing is pointless. This sends a handful of deliberately tiny requests, times
them, and reports what your key actually sustains -- then tells you what to put
in LLM_RPS.

Usage:
    python scripts/probe_limits.py            # 8 calls, default model
    python scripts/probe_limits.py -n 20      # more samples
    python scripts/probe_limits.py --model mistral-small-latest
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pydantic import BaseModel  # noqa: E402

from graphrag.config import settings  # noqa: E402
from graphrag.llm.base import LLMError  # noqa: E402
from graphrag.llm.meter import METER  # noqa: E402


class Ping(BaseModel):
    """Smallest useful structured response -- also proves schema mode works."""

    answer: int


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", "--calls", type=int, default=8)
    ap.add_argument("--model", default=None)
    ap.add_argument(
        "--rps",
        type=float,
        default=100.0,
        help="Client-side throttle during the probe. Default is effectively off, "
        "so we measure the server's limit rather than our own.",
    )
    args = ap.parse_args()

    model = args.model or settings.router_model

    print(f"backend : {settings.llm_backend}")
    print(f"model   : {model}")
    print(f"calls   : {args.calls}")
    print()

    # Import after arg parsing so --help works without a key configured.
    from graphrag.llm.mistral_backend import MistralBackend

    try:
        backend = MistralBackend(
            api_key=settings.mistral_api_key or "",
            rps=args.rps,
            max_attempts=1,  # no retry: we want to SEE the 429s, not hide them
        )
    except LLMError as exc:
        print(f"FAILED: {exc}")
        return 1

    latencies: list[float] = []
    throttled = 0
    failures = 0
    started = time.monotonic()

    for i in range(1, args.calls + 1):
        t0 = time.monotonic()
        try:
            resp = backend.parse(
                system="Reply with the requested number and nothing else.",
                user=f"What is {i} + 1? Respond as JSON with an 'answer' field.",
                model=model,
                schema=Ping,
                max_tokens=64,
                stage="probe",
            )
            dt = time.monotonic() - t0
            latencies.append(dt)
            ok = resp.parsed is not None and resp.parsed.answer == i + 1
            print(
                f"  {i:3d}  {dt:6.2f}s  "
                f"in={resp.usage.input_tokens:4d} out={resp.usage.output_tokens:3d}  "
                f"{'ok' if ok else 'parsed but wrong value'}"
            )
        except Exception as exc:  # noqa: BLE001 - probing, report everything
            dt = time.monotonic() - t0
            text = str(exc)
            if "429" in text or "rate limit" in text.lower():
                throttled += 1
                print(f"  {i:3d}  {dt:6.2f}s  429 RATE LIMITED")
            else:
                failures += 1
                print(f"  {i:3d}  {dt:6.2f}s  ERROR: {type(exc).__name__}: {text[:120]}")

    elapsed = time.monotonic() - started
    ok_calls = len(latencies)

    print()
    print("-" * 58)
    print(f"succeeded    : {ok_calls}/{args.calls}")
    print(f"rate limited : {throttled}")
    print(f"other errors : {failures}")
    if latencies:
        print(f"latency      : median {statistics.median(latencies):.2f}s  "
              f"min {min(latencies):.2f}s  max {max(latencies):.2f}s")
    print(f"wall clock   : {elapsed:.1f}s  ({ok_calls / elapsed:.2f} successful calls/sec)")
    print(f"tokens used  : {METER.total_tokens}")
    print("-" * 58)
    print()

    if ok_calls == 0:
        print("VERDICT: no calls succeeded. Check MISTRAL_API_KEY and that the")
        print("         Experiment plan is activated in the Mistral console.")
        return 1

    if throttled:
        safe = max(0.2, round(ok_calls / elapsed, 1))
        print(f"VERDICT: throttling observed. Set LLM_RPS={safe} in .env")
    else:
        observed = ok_calls / elapsed
        print(f"VERDICT: no throttling at {observed:.2f} calls/sec on SMALL requests.")
        print()
        print("  CAUTION: this probe sends ~40-token requests. The free tier's")
        print("  binding limit is tokens per minute, not requests per second, so")
        print("  this number does NOT transfer to extraction, whose calls carry")
        print("  ~17k tokens each and hit 429 almost immediately at this rate.")
        print("  Measured guidance: LLM_RPS=0.25 sustains extraction;")
        print(f"  up to {max(1.0, round(observed, 1))} is fine for routing and answering.")

    print()
    print("Projected wall-clock for this project at that rate:")
    rate = max(0.2, ok_calls / elapsed)
    for label, n in (("extraction (60 windows)", 60), ("one eval run (160 calls)", 160)):
        secs = n / rate
        print(f"  {label:28} {secs / 60:5.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
