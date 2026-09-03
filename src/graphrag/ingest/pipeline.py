"""Ingestion: arXiv -> parsed -> chunked -> indexed.

Checkpointed per paper. Each parsed paper is written to
``data/processed/papers/<id>.json`` before indexing, so a rerun skips work
already done and an interrupted run costs seconds rather than the whole batch.

This stage makes **zero LLM calls**. Everything here -- fetching, parsing,
chunking, embedding, BM25 -- runs locally and free.
"""

from __future__ import annotations

from pathlib import Path

from graphrag.config import settings
from graphrag.index.bm25 import BM25Index
from graphrag.index.vector_store import VectorStore
from graphrag.ingest.chunk import build_chunks, build_windows
from graphrag.ingest.fetch_arxiv import search_arxiv
from graphrag.ingest.parse import parse_pdf
from graphrag.logging import get_logger
from graphrag.models import ParsedPaper

log = get_logger(__name__)


def papers_dir() -> Path:
    return settings.processed_dir / "papers"


def parsed_path(paper_id: str) -> Path:
    return papers_dir() / f"{paper_id.replace('/', '_')}.json"


def load_parsed(paper_id: str) -> ParsedPaper | None:
    path = parsed_path(paper_id)
    if not path.exists():
        return None
    try:
        return ParsedPaper.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - a corrupt checkpoint should not be fatal
        log.warning("corrupt_checkpoint_reparsing", paper_id=paper_id, error=str(exc)[:120])
        return None


def load_all_parsed() -> list[ParsedPaper]:
    """Every paper ingested so far. Used to rebuild BM25 and by extraction."""
    if not papers_dir().exists():
        return []
    out: list[ParsedPaper] = []
    for path in sorted(papers_dir().glob("*.json")):
        try:
            out.append(ParsedPaper.model_validate_json(path.read_text(encoding="utf-8")))
        except Exception as exc:  # noqa: BLE001
            log.warning("skipping_unreadable_checkpoint", path=path.name, error=str(exc)[:120])
    return out


def process_paper(paper, *, force: bool = False) -> ParsedPaper | None:
    """Parse and chunk one paper, reusing an existing checkpoint when present."""
    if not force:
        cached = load_parsed(paper.paper_id)
        if cached is not None:
            log.debug("paper_cached", paper_id=paper.paper_id)
            return cached

    if not paper.pdf_path or not Path(paper.pdf_path).exists():
        log.warning("pdf_missing", paper_id=paper.paper_id)
        return None

    try:
        full_text, sections = parse_pdf(paper.pdf_path)
    except Exception as exc:  # noqa: BLE001 - one bad PDF must not stop the batch
        log.warning("parse_failed", paper_id=paper.paper_id, error=str(exc)[:150])
        return None

    if len(full_text) < 500:
        log.warning("parsed_text_too_short", paper_id=paper.paper_id, chars=len(full_text))
        return None

    chunks = build_chunks(
        paper_id=paper.paper_id,
        full_text=full_text,
        sections=sections,
        target_tokens=settings.chunk_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    windows = build_windows(
        paper_id=paper.paper_id,
        full_text=full_text,
        sections=sections,
        target_tokens=settings.extraction_window_tokens,
        overlap_tokens=settings.extraction_window_overlap_tokens,
    )

    parsed = ParsedPaper(
        paper=paper,
        full_text=full_text,
        sections=sections,
        chunks=chunks,
        windows=windows,
    )

    papers_dir().mkdir(parents=True, exist_ok=True)
    parsed_path(paper.paper_id).write_text(parsed.model_dump_json(indent=2), encoding="utf-8")

    log.info(
        "paper_processed",
        paper_id=paper.paper_id,
        chunks=len(chunks),
        windows=len(windows),
        sections=len(sections),
    )
    return parsed


def ingest(
    query: str,
    *,
    limit: int = 20,
    force: bool = False,
) -> dict[str, int]:
    """Full ingestion run. Returns a summary of what happened."""
    settings.ensure_dirs()

    papers = search_arxiv(query, limit=limit)
    parsed_papers = [p for p in (process_paper(pp, force=force) for pp in papers) if p]

    all_chunks = [c for p in parsed_papers for c in p.chunks]
    total_windows = sum(len(p.windows) for p in parsed_papers)

    if all_chunks:
        VectorStore().add(all_chunks)

        # BM25 has no incremental mode, so it is rebuilt from every paper on
        # disk -- not just this batch -- or earlier papers would vanish from it.
        bm25 = BM25Index()
        bm25.build([c for p in load_all_parsed() for c in p.chunks])
        bm25.save()

    summary = {
        "papers_found": len(papers),
        "papers_processed": len(parsed_papers),
        "chunks": len(all_chunks),
        "extraction_windows": total_windows,
    }
    log.info("ingest_complete", **summary)
    return summary


def reindex() -> dict[str, int]:
    """Rebuild the vector and BM25 indexes from papers already on disk.

    Needed when indexing was interrupted: parsing is checkpointed per paper but
    embedding happens in one pass at the end, so a kill mid-embedding leaves
    `vectors_indexed` behind `chunks` with no other symptom. Re-running `ingest`
    would work but re-fetches every PDF from arXiv for nothing.
    """
    settings.ensure_dirs()
    papers = load_all_parsed()
    if not papers:
        log.warning("nothing_to_reindex")
        return {"papers": 0, "chunks": 0, "vectors_indexed": 0}

    chunks = [c for p in papers for c in p.chunks]
    store = VectorStore()
    store.add(chunks)

    bm25 = BM25Index()
    bm25.build(chunks)
    bm25.save()

    summary = {
        "papers": len(papers),
        "chunks": len(chunks),
        "vectors_indexed": store.count(),
    }
    log.info("reindex_complete", **summary)
    return summary


def stats() -> dict[str, int]:
    """Current index state, for the CLI and the API's /stats endpoint."""
    parsed = load_all_parsed()
    return {
        "papers": len(parsed),
        "chunks": sum(len(p.chunks) for p in parsed),
        "extraction_windows": sum(len(p.windows) for p in parsed),
        "vectors_indexed": VectorStore().count(),
    }


def dry_run(query: str, limit: int = 20) -> dict[str, int | float]:
    """Project extraction cost without making a single LLM call.

    Uses papers already on disk when available so the estimate reflects real
    documents rather than an assumed page count.
    """
    parsed = load_all_parsed()
    if not parsed:
        log.info("dry_run_needs_ingestion", hint="run ingest first for a real estimate")
        return {"windows": 0, "estimated_input_tokens": 0, "note": 0}

    windows = [w for p in parsed for w in p.windows]
    input_tokens = sum(w.token_estimate for w in windows)
    # Extraction emits roughly 0.5-0.7 output tokens per input token of dense
    # academic text; 0.6 is the midpoint and is only ever an estimate.
    output_tokens = int(input_tokens * 0.6)

    return {
        "papers": len(parsed),
        "windows": len(windows),
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "estimated_total_tokens": input_tokens + output_tokens,
    }
