"""The full query pipeline: route -> retrieve -> answer.

Every mode degrades to vector retrieval rather than failing. A graph query that
returns nothing, a router that errors, an empty Cypher result -- all of them
fall through to the passage retriever, because an answer grounded in text beats
an error message.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from graphrag.answer.synthesize import Answer, synthesize
from graphrag.index.vector_store import Hit
from graphrag.llm.base import LLMBackend
from graphrag.logging import get_logger
from graphrag.retrieve.cypher import run_graph_query
from graphrag.retrieve.global_search import global_search
from graphrag.retrieve.hybrid import search
from graphrag.retrieve.router import MODES, RouteDecision, route

log = get_logger(__name__)


@dataclass
class QueryResult:
    question: str
    answer: Answer
    mode: str
    route_reason: str = ""
    graph_rows: list[dict] = field(default_factory=list)
    cypher: str | None = None
    fell_back: bool = False
    # Everything retrieved, not just what the answer ended up citing. The
    # evaluation needs this: measuring GraphRAG's recall over cited chunks while
    # measuring baselines over retrieved chunks compares different denominators
    # and understates GraphRAG.
    hits: list[Hit] = field(default_factory=list)


def _rows_as_context(rows: list[dict], cypher: str | None) -> list[Hit]:
    """Wrap graph rows as a synthetic source so the answerer can cite them.

    Graph facts have no chunk to point at, so the citation resolves to the query
    that produced them -- which is the honest provenance for a computed answer.
    """
    if not rows:
        return []

    lines = []
    for i, row in enumerate(rows[:60], start=1):
        parts = [f"{k}={v}" for k, v in row.items() if v is not None]
        lines.append(f"{i}. " + ", ".join(parts))

    body = (
        "Structured facts retrieved from the knowledge graph"
        + (f" via:\n{cypher}\n\n" if cypher else ":\n\n")
        + "\n".join(lines)
    )
    return [
        Hit(
            chunk_id="graph-result",
            paper_id="knowledge-graph",
            text=body,
            score=1.0,
            char_start=0,
            char_end=0,
            section="graph query",
            source="graph",
        )
    ]


def ask(
    question: str,
    *,
    backend: LLMBackend,
    k: int = 8,
    force_mode: str | None = None,
) -> QueryResult:
    """Answer a question end to end.

    ``force_mode`` bypasses routing, which the evaluation and the UI both use to
    compare modes on the same question. It is validated rather than trusted: an
    unrecognised mode matches no dispatch branch below, which would silently
    produce an empty answer instead of an error.
    """
    if force_mode is not None:
        if force_mode not in MODES:
            raise ValueError(f"Unknown mode {force_mode!r}. Expected one of {MODES}.")
        decision = RouteDecision(mode=force_mode, reason="forced")
    else:
        decision = route(question, backend=backend)

    mode = decision.mode
    log.info("routed", question=question[:70], mode=mode, reason=decision.reason)

    graph_rows: list[dict] = []
    cypher: str | None = None
    hits: list[Hit] = []
    fell_back = False

    if mode == "global":
        hits.extend(global_search(question))
        if not hits:
            # Communities have not been built. Passages are a weak substitute
            # for a corpus-wide answer, but better than refusing.
            fell_back = True
            log.info("falling_back_to_vector", reason="no communities built")
            hits.extend(search(question, k=k))

    if mode in ("graph", "hybrid"):
        graph_rows, cypher = run_graph_query(question, backend=backend)
        hits.extend(_rows_as_context(graph_rows, cypher))

    if mode in ("vector", "hybrid") or (mode == "graph" and not graph_rows):
        if mode == "graph" and not graph_rows:
            # The graph had nothing -- passages are better than nothing.
            fell_back = True
            log.info("falling_back_to_vector", reason="empty graph result")
        hits.extend(search(question, k=k))

    answer = synthesize(question, hits, backend=backend, mode=mode)
    return QueryResult(
        question=question,
        answer=answer,
        mode=mode,
        route_reason=decision.reason,
        graph_rows=graph_rows,
        cypher=cypher,
        fell_back=fell_back,
        hits=hits,
    )
