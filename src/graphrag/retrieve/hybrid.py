"""Vector + BM25 retrieval fused by RRF.

This is the M1 baseline and, deliberately, the control arm of the final
evaluation. Every mode added later (graph, global) is measured against exactly
this, so it is worth being a genuinely good plain-RAG system rather than a
strawman.
"""

from __future__ import annotations

from functools import lru_cache

from graphrag.config import settings
from graphrag.index.bm25 import BM25Index
from graphrag.index.fuse import reciprocal_rank_fusion
from graphrag.index.vector_store import Hit, VectorStore
from graphrag.logging import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def _stores() -> tuple[VectorStore, BM25Index]:
    """Load both indexes once per process. Chroma and BM25 are slow to open."""
    vector = VectorStore()
    bm25 = BM25Index()
    if not bm25.load():
        log.warning("bm25_index_missing", hint="run ingestion to build it")
    return vector, bm25


def search(
    query: str,
    *,
    k: int | None = None,
    mode: str = "hybrid",
) -> list[Hit]:
    """Retrieve chunks for ``query``.

    ``mode`` is one of ``hybrid`` (default), ``vector``, or ``bm25``. The
    single-retriever modes exist so the evaluation can measure each in
    isolation, not just the combination.
    """
    k = k or settings.top_k
    vector, bm25 = _stores()

    if mode == "vector":
        return vector.search(query, k=k)
    if mode == "bm25":
        return bm25.search(query, k=k)
    if mode != "hybrid":
        raise ValueError(f"Unknown retrieval mode {mode!r}: expected hybrid, vector, or bm25")

    # Over-fetch from each retriever so fusion has room to reorder; a document
    # ranked 9th by one and 2nd by the other should still be able to surface.
    over = max(k * 2, 20)
    return reciprocal_rank_fusion(
        [vector.search(query, k=over), bm25.search(query, k=over)],
        limit=k,
    )


def reset_cache() -> None:
    """Drop cached stores. Needed after ingestion adds new documents."""
    _stores.cache_clear()
