"""Checkpointed extraction over windows.

Every completed window is appended to ``triples.jsonl`` immediately. A rerun
reads that file first and skips what is already done, so an interrupt, a crash,
or a rate limit costs seconds rather than the whole run. This is the single
most important property of this stage -- extraction is the only expensive,
slow, non-deterministic step in the pipeline.

Failures are recorded too, with the error, so a window that never succeeds is
visible in the data rather than silently missing from the graph.
"""

from __future__ import annotations

import json
from pathlib import Path

from graphrag.config import settings
from graphrag.extract.prompts import EXTRACTION_SYSTEM, build_user_prompt
from graphrag.extract.schemas import WindowExtraction
from graphrag.llm.base import LLMBackend, SchemaValidationError
from graphrag.llm.meter import METER
from graphrag.logging import get_logger
from graphrag.models import ExtractionWindow

log = get_logger(__name__)

# Stable cache key: the system prompt is identical for every window, so the
# provider can serve the prefix from cache instead of re-reading it each time.
CACHE_KEY = "graphrag-extraction-v1"


def triples_path() -> Path:
    return settings.processed_dir / "triples.jsonl"


def load_records(path: Path | None = None) -> list[dict]:
    """Every checkpoint record, successes and failures alike, in write order."""
    path = Path(path or triples_path())
    if not path.exists():
        return []

    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A partial final line is normal after an interrupt mid-write.
                log.warning("skipping_bad_checkpoint_line", line=line_no)
    return records


def load_done(path: Path | None = None) -> dict[str, dict]:
    """Window ID -> record, for windows that **succeeded**.

    Only successes count as done. A failed window -- usually a transient rate
    limit -- must be retried on the next run, otherwise a temporary 429 turns
    into permanently missing graph data that nothing ever reports.

    The file is append-only, so a window that failed and later succeeded has two
    records; the successful one wins regardless of order.
    """
    done: dict[str, dict] = {}
    for record in load_records(path):
        window_id = record.get("window_id")
        if window_id and record.get("status") == "ok":
            done[window_id] = record
    return done


def _append(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()  # survive a hard kill, not just a clean exit


def extract_window(
    window: ExtractionWindow,
    *,
    paper_title: str,
    backend: LLMBackend,
    model: str | None = None,
) -> WindowExtraction:
    """Extract from one window. Raises SchemaValidationError on unusable output."""
    response = backend.parse(
        system=EXTRACTION_SYSTEM,
        user=build_user_prompt(
            paper_title=paper_title,
            sections=window.sections,
            text=window.text,
        ),
        model=model or settings.extraction_model,
        schema=WindowExtraction,
        max_tokens=32000,
        stage="extraction",
        cache_key=CACHE_KEY,
    )
    if response.parsed is None:  # pragma: no cover - backend guarantees this
        raise SchemaValidationError("backend returned no parsed object")
    return response.parsed


def run_extraction(
    windows: list[ExtractionWindow],
    *,
    titles: dict[str, str],
    backend: LLMBackend,
    model: str | None = None,
    force: bool = False,
    limit: int | None = None,
) -> dict[str, int]:
    """Extract over many windows, checkpointing after each."""
    path = triples_path()
    done = {} if force else load_done(path)

    pending = [w for w in windows if w.window_id not in done]
    if limit is not None:
        pending = pending[:limit]

    log.info(
        "extraction_starting",
        total=len(windows),
        already_done=len(windows) - len([w for w in windows if w.window_id not in done]),
        pending=len(pending),
        model=model or settings.extraction_model,
    )

    ok = failed = 0
    for i, window in enumerate(pending, start=1):
        try:
            result = extract_window(
                window,
                paper_title=titles.get(window.paper_id, window.paper_id),
                backend=backend,
                model=model,
            )
            _append(
                {
                    "window_id": window.window_id,
                    "paper_id": window.paper_id,
                    "char_start": window.char_start,
                    "char_end": window.char_end,
                    "status": "ok",
                    "extraction": result.model_dump(mode="json"),
                },
                path,
            )
            ok += 1
            log.info(
                "window_extracted",
                progress=f"{i}/{len(pending)}",
                paper=window.paper_id,
                entities=len(result.entities),
                relations=len(result.relations),
                results=len(result.results),
            )

        except Exception as exc:  # noqa: BLE001 - one window must not stop the run
            _append(
                {
                    "window_id": window.window_id,
                    "paper_id": window.paper_id,
                    "char_start": window.char_start,
                    "char_end": window.char_end,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                },
                path,
            )
            failed += 1
            log.warning(
                "window_extraction_failed",
                progress=f"{i}/{len(pending)}",
                paper=window.paper_id,
                error=str(exc)[:200],
            )

    summary = {
        "windows_total": len(windows),
        "attempted": len(pending),
        "succeeded": ok,
        "failed": failed,
        "skipped_already_done": len(windows) - len(pending),
        "tokens": METER.total_tokens,
    }
    log.info("extraction_complete", **summary)
    return summary
