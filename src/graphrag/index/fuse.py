"""Reciprocal Rank Fusion.

Vector similarity and BM25 produce scores on incomparable scales, so blending
the raw numbers is meaningless. RRF ignores magnitudes and uses rank position
only, which is why it combines heterogeneous retrievers robustly without any
per-retriever weight tuning.

    score(d) = sum over retrievers of 1 / (k + rank(d))
"""

from __future__ import annotations

from collections.abc import Iterable

from graphrag.index.vector_store import Hit

# 60 is the value from the original RRF paper; it damps the influence of the
# very top ranks so one confident retriever cannot dominate the other.
RRF_K = 60


def reciprocal_rank_fusion(
    result_lists: Iterable[list[Hit]],
    *,
    k: int = RRF_K,
    limit: int | None = None,
) -> list[Hit]:
    """Fuse ranked lists into one, deduplicating by ``chunk_id``."""
    scores: dict[str, float] = {}
    best: dict[str, Hit] = {}
    sources: dict[str, set[str]] = {}

    for hits in result_lists:
        for rank, hit in enumerate(hits, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            sources.setdefault(hit.chunk_id, set()).add(hit.source)
            # Keep the first occurrence; all copies carry the same text/offsets.
            best.setdefault(hit.chunk_id, hit)

    fused = []
    for chunk_id, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        hit = best[chunk_id]
        fused.append(
            Hit(
                chunk_id=hit.chunk_id,
                paper_id=hit.paper_id,
                text=hit.text,
                score=score,
                char_start=hit.char_start,
                char_end=hit.char_end,
                section=hit.section,
                source="+".join(sorted(sources[chunk_id])),
            )
        )

    return fused[:limit] if limit else fused
