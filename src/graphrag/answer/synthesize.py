"""Answer synthesis with enforced citations.

Two rules make the citations real rather than decorative:

1. The prompt numbers each retrieved chunk and requires [n] markers.
2. Every marker in the returned text is validated against the chunks actually
   supplied. A citation the model invented is dropped, and the drop is logged.

Without step 2 a model can cite [7] when six chunks were provided, and the
answer still *looks* well-sourced.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from graphrag.config import settings
from graphrag.index.vector_store import Hit
from graphrag.llm.base import LLMBackend
from graphrag.logging import get_logger

log = get_logger(__name__)

_CITATION = re.compile(r"\[(\d+)\]")

SYSTEM = """You answer questions about research papers using only the numbered \
sources provided.

Rules:
- Use ONLY information present in the sources. Do not add outside knowledge.
- Cite every factual claim with the source number in square brackets, like [2].
- A sentence drawing on several sources cites each one: [1][3].
- If the sources do not contain the answer, say so plainly. Do not guess.
- Be concise and specific. Prefer concrete names, datasets and numbers over \
general description."""

# Appended when the context includes rows from the knowledge graph.
#
# Graph results arrive as structured rows, and a model handed a table tends to
# echo it back as a bare list. That is a correct answer that reads as an
# incomplete one -- in the benchmark it cost the multi-hop category, where
# GraphRAG returned the right entities and was still marked down against a
# prose reference. The facts are the graph's; the sentence around them is not.
GRAPH_CONTEXT_RULES = """

The sources include structured facts retrieved from a knowledge graph.
- Answer in prose. State the finding in a sentence, then list the specifics.
- Do not simply reproduce the rows as a bare list with no framing.
- Say what the result means: how many there are, what they have in common, or
  what the relationship is -- but only what the rows actually support.
- Never add entities that are absent from the rows to round out a sentence."""


@dataclass
class Answer:
    text: str
    citations: list[Hit] = field(default_factory=list)
    mode: str = "vector"
    dropped_citations: list[int] = field(default_factory=list)


def _format_sources(hits: list[Hit]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        header = f"[{i}] paper {hit.paper_id}"
        if hit.section:
            header += f", section: {hit.section}"
        blocks.append(f"{header}\n{hit.text}")
    return "\n\n---\n\n".join(blocks)


def _validate_citations(text: str, hits: list[Hit]) -> tuple[list[Hit], list[int]]:
    """Keep only citations that point at a real source. Report the rest."""
    cited, invalid = [], []
    for raw in _CITATION.findall(text):
        idx = int(raw)
        if 1 <= idx <= len(hits):
            cited.append(idx)
        else:
            invalid.append(idx)

    if invalid:
        log.warning("hallucinated_citations_dropped", indices=sorted(set(invalid)))

    seen: set[int] = set()
    ordered = [i for i in cited if not (i in seen or seen.add(i))]
    return [hits[i - 1] for i in ordered], sorted(set(invalid))


def synthesize(
    question: str,
    hits: list[Hit],
    *,
    backend: LLMBackend,
    model: str | None = None,
    mode: str = "vector",
    max_tokens: int = 1500,
) -> Answer:
    """Produce a grounded answer from retrieved chunks."""
    if not hits:
        return Answer(
            text="No relevant passages were found in the indexed papers for this question.",
            mode=mode,
        )

    user = f"Sources:\n\n{_format_sources(hits)}\n\nQuestion: {question}"

    # Structured sources need different writing guidance from prose passages.
    has_structured = any(h.source in ("graph", "global") for h in hits)
    system = SYSTEM + (GRAPH_CONTEXT_RULES if has_structured else "")

    response = backend.complete(
        system=system,
        user=user,
        model=model or settings.answer_model,
        max_tokens=max_tokens,
        stage="answer",
    )

    citations, dropped = _validate_citations(response.text, hits)
    return Answer(text=response.text, citations=citations, mode=mode, dropped_citations=dropped)
