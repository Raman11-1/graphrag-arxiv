"""Global retrieval over community summaries.

Answers questions about the corpus as a whole. Instead of retrieving passages,
it retrieves pre-computed summaries of the graph's densely-connected entity
clusters -- so "what are the main themes?" reads a handful of theme
descriptions rather than eight arbitrary paragraphs.

Communities are ranked by embedding similarity to the question rather than
always returning all of them, so this still behaves sensibly once a corpus has
dozens of clusters. With only a few, everything is returned anyway.
"""

from __future__ import annotations

import numpy as np

from graphrag.graph.communities import Community, load_communities
from graphrag.graph.store import GraphStore
from graphrag.index.vector_store import Hit
from graphrag.logging import get_logger

log = get_logger(__name__)

# Below this many communities, ranking adds nothing -- return them all.
_RANK_THRESHOLD = 6


def _as_text(community: Community) -> str:
    title = community.title or community.community_id
    return f"{title}\n{community.summary}".strip()


def rank_communities(question: str, communities: list[Community], k: int) -> list[Community]:
    """Most relevant communities first, by embedding similarity."""
    if len(communities) <= _RANK_THRESHOLD:
        return communities

    from graphrag.index.embedder import get_embedder

    embedder = get_embedder()
    doc_vectors = embedder.embed_documents([_as_text(c) for c in communities])
    query_vector = embedder.embed_query(question)

    norms = np.linalg.norm(doc_vectors, axis=1)
    norms[norms == 0] = 1.0
    scores = (doc_vectors @ query_vector) / (norms * (np.linalg.norm(query_vector) or 1.0))

    order = np.argsort(-scores)[:k]
    return [communities[i] for i in order]


def global_search(
    question: str,
    *,
    store: GraphStore | None = None,
    k: int = 6,
) -> list[Hit]:
    """Return community summaries as citable sources.

    An empty list means communities have not been built yet; the caller falls
    back to passage retrieval rather than reporting an error.
    """
    store = store or GraphStore(read_only=True)
    communities = [c for c in load_communities(store) if c.summary or c.title]

    if not communities:
        log.warning(
            "no_communities",
            hint="run community detection before asking corpus-wide questions",
        )
        return []

    selected = rank_communities(question, communities, k)
    log.info("global_search", available=len(communities), selected=len(selected))

    return [
        Hit(
            chunk_id=f"community:{c.community_id}",
            paper_id="knowledge-graph",
            text=f"Research theme: {c.title}\n\n{c.summary}",
            score=1.0 - (rank / max(len(selected), 1)),
            char_start=0,
            char_end=0,
            section="community summary",
            source="global",
        )
        for rank, c in enumerate(selected)
    ]
