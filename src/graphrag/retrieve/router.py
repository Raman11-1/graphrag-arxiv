"""Query routing.

Picks the retriever whose shape matches the question:

* ``vector`` -- definitional/local. "What is dense retrieval?"
* ``graph``  -- relational, multi-hop, counting. "Which authors did both?"
* ``global`` -- corpus-wide themes. "What are the main research directions?"
* ``hybrid`` -- needs both a fact and its surrounding explanation.

A cheap model is used deliberately: this is a 4-way classification on a short
string, and it runs on every single query. The routing decision is also
recoverable -- every mode falls back to ``vector`` -- so a wrong route degrades
the answer rather than breaking it.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from graphrag.config import settings
from graphrag.llm.base import LLMBackend
from graphrag.logging import get_logger

log = get_logger(__name__)

MODES = ("vector", "graph", "global", "hybrid")

# Cheap pre-filter. These patterns are unambiguous enough that spending an LLM
# call on them is waste; anything not matched falls through to the model.
_COUNTING = re.compile(
    r"\b(how many|count|number of|most|fewest|least|largest number|top \d+|"
    r"which .* (both|and))\b",
    re.IGNORECASE,
)

# Corpus-wide questions ask for *synthesis*, and they signal it two ways: an
# overview verb ("themes", "overview", "landscape"), or a phrase scoping the
# question to the whole collection ("across these papers", "in this corpus").
#
# The original pattern only matched a handful of exact phrasings -- "across the"
# but not "across this", nothing at all for "in this corpus" -- so two of three
# global questions in the benchmark fell through to the LLM router and were
# answered as graph queries. Broadened to the intent rather than the wording.
_SYNTHESIS = (
    r"main (themes|topics|ideas|directions|areas)|"
    r"(overview|survey|landscape|breadth|range) of|"
    r"what (application |research )?(domains|areas|fields)|"
    r"which (research |application )?(areas|domains|fields)|"
    r"summar(y|ise|ize)|"
    r"relate to each other|"
    r"trends"
)
_CORPUS_SCOPE = (
    r"across (the|this|these|all)|"
    r"in (this|these) (corpus|collection|papers|body|work)|"
    r"(these|the) papers|this collection|this corpus|body of work"
)
_GLOBAL = re.compile(rf"\b({_SYNTHESIS}|{_CORPUS_SCOPE})\b", re.IGNORECASE)


class RouteDecision(BaseModel):
    mode: str = Field(description="One of: vector, graph, global, hybrid")
    reason: str = Field(default="", description="One short clause explaining the choice")


ROUTER_SYSTEM = """You classify questions about a corpus of research papers by \
which retrieval strategy can answer them.

graph   - the answer is a LIST, a COUNT, a NUMBER, or a SET OF ENTITIES.
          Any question of the form "which X <relation> Y", "who wrote X",
          "what does X use / evaluate on / build on / compare against",
          "how many", "which is best". Choose graph even when the question
          needs only one hop -- a stored relationship gives a complete and
          exact answer, whereas passages give whichever ones happen to rank.
vector  - the answer is an EXPLANATION in prose: definitions, "how does X
          work", "why does X do Y", descriptions of a mechanism.
global  - the answer requires the WHOLE corpus: overall themes, trends,
          what topics are covered, how the field has developed.
hybrid  - genuinely needs a structured fact AND the prose around it, e.g.
          "which datasets does X use and why were they chosen".

The distinguishing test: could the answer be written as a table of entities?
If yes, choose graph. If it must be written as paragraphs, choose vector.

Examples:
  "Which datasets does DPR evaluate on?"        -> graph
  "Who are the authors of the DPR paper?"       -> graph
  "What methods outperform BM25?"               -> graph
  "How does dense retrieval encode passages?"   -> vector
  "Why is negative sampling important?"         -> vector
  "What are the main themes in this corpus?"    -> global

Answer with the mode and a short reason. Nothing else."""


def route(
    question: str,
    *,
    backend: LLMBackend | None = None,
    model: str | None = None,
) -> RouteDecision:
    """Choose a retrieval mode. Never raises -- defaults to hybrid."""
    # Counting is checked first, and the order matters. A counting question
    # often also names the corpus -- "how many papers here work on VQA" -- and
    # with global first, the corpus-scope phrase would win and send a question
    # with an exact numeric answer to the summariser.
    if _COUNTING.search(question):
        return RouteDecision(mode="graph", reason="matched counting/relational pattern")
    if _GLOBAL.search(question):
        return RouteDecision(mode="global", reason="matched corpus-wide pattern")

    if backend is None:
        return RouteDecision(mode="hybrid", reason="no backend available")

    try:
        response = backend.parse(
            system=ROUTER_SYSTEM,
            user=question,
            model=model or settings.router_model,
            schema=RouteDecision,
            max_tokens=200,
            stage="router",
        )
        decision = response.parsed
        if decision and decision.mode in MODES:
            return decision
        log.warning("router_returned_unknown_mode", got=getattr(decision, "mode", None))
    except Exception as exc:  # noqa: BLE001 - routing must never break a query
        log.warning("router_failed", error=str(exc)[:200])

    # Hybrid is the safe default: it runs both retrievers, so a failed routing
    # decision costs latency rather than the answer.
    return RouteDecision(mode="hybrid", reason="router unavailable, defaulting to hybrid")
