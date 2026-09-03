"""Fetch papers from arXiv.

Note on the SDK: arxiv 4.x removed ``Result.download_pdf`` -- results now only
expose ``pdf_url`` and downloading is the caller's job. Doing it ourselves is
no loss; it means the cache check, timeout and failure handling are explicit.

Downloads are cached on disk by arXiv ID, so re-running a query never re-fetches
a paper we already have. arXiv asks API clients to pace themselves, so there is
a deliberate delay between requests.
"""

from __future__ import annotations

import time
from pathlib import Path

from graphrag.config import settings
from graphrag.logging import get_logger
from graphrag.models import Paper

log = get_logger(__name__)

# arXiv's API guidance asks for roughly one request every three seconds.
_MIN_REQUEST_INTERVAL = 3.0
_DOWNLOAD_TIMEOUT = 60
# A PDF smaller than this is almost certainly an error page, not a paper.
_MIN_PDF_BYTES = 10_000


def _safe_filename(paper_id: str) -> str:
    """Old-style IDs contain '/', e.g. 'cs/0501001'."""
    return f"{paper_id.replace('/', '_')}.pdf"


def _download(url: str, dest: Path) -> bool:
    """Fetch a PDF to ``dest``. Returns False on any failure, never raises.

    Writes to a temporary path first so an interrupted download can never leave
    a truncated file that a later run would mistake for a valid cache hit.
    """
    import requests

    tmp = dest.with_suffix(".part")
    try:
        with requests.get(url, stream=True, timeout=_DOWNLOAD_TIMEOUT) as response:
            response.raise_for_status()
            with tmp.open("wb") as fh:
                for block in response.iter_content(chunk_size=65536):
                    fh.write(block)

        size = tmp.stat().st_size
        if size < _MIN_PDF_BYTES:
            log.warning("pdf_suspiciously_small", url=url, bytes=size)
            tmp.unlink(missing_ok=True)
            return False

        tmp.replace(dest)
        return True

    except Exception as exc:  # noqa: BLE001 - one failure must not stop the batch
        log.warning("pdf_download_failed", url=url, error=str(exc)[:150])
        tmp.unlink(missing_ok=True)
        return False


def search_arxiv(
    query: str,
    *,
    limit: int = 20,
    download_dir: Path | None = None,
    skip_existing: bool = True,
) -> list[Paper]:
    """Search arXiv and download PDFs, returning metadata for each paper.

    Papers whose PDF fails to download are logged and skipped rather than
    aborting the batch -- one bad paper should not cost the other nineteen.
    """
    import arxiv

    download_dir = Path(download_dir or settings.raw_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    client = arxiv.Client(page_size=min(limit, 100), delay_seconds=_MIN_REQUEST_INTERVAL)
    search = arxiv.Search(
        query=query,
        max_results=limit,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers: list[Paper] = []
    for result in client.results(search):
        paper_id = result.get_short_id()
        pdf_path = download_dir / _safe_filename(paper_id)

        if pdf_path.exists() and skip_existing:
            log.debug("pdf_cached", paper_id=paper_id)
        else:
            pdf_url = getattr(result, "pdf_url", None)
            if not pdf_url:
                log.warning("no_pdf_url", paper_id=paper_id)
                continue
            if not _download(pdf_url, pdf_path):
                continue
            log.info("pdf_downloaded", paper_id=paper_id, title=result.title[:60])
            time.sleep(_MIN_REQUEST_INTERVAL)

        papers.append(
            Paper(
                paper_id=paper_id,
                title=" ".join(result.title.split()),
                authors=[a.name for a in result.authors],
                abstract=" ".join((result.summary or "").split()),
                published=result.published.date() if result.published else None,
                categories=list(result.categories or []),
                pdf_path=str(pdf_path),
                source_url=result.entry_id,
            )
        )

    log.info("arxiv_search_complete", query=query, requested=limit, retrieved=len(papers))
    return papers
