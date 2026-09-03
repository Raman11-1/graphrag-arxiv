"""Text-to-Cypher with a mutation guard.

An LLM writes queries that run against a live database, and the question text
comes from a user. Two independent protections:

**Token-level keyword rejection.** Substring matching is not enough -- it both
misses ``MATCH(n) DETACH DELETE n`` written without spaces and falsely rejects
a legitimate ``m.name = 'CREATE-Bench'``. We tokenise and check whole words.

**A mandatory LIMIT.** An unbounded MATCH over a dense graph returns a
combinatorial explosion of paths that will exhaust memory long before it
returns.

The guard is deliberately a denylist *and* a structural requirement (the query
must start with MATCH/OPTIONAL MATCH/WITH/RETURN), so a keyword we failed to
anticipate still cannot begin a mutating statement.
"""

from __future__ import annotations

import re

from graphrag.config import settings
from graphrag.graph.schema import schema_description
from graphrag.graph.store import GraphStore
from graphrag.llm.base import LLMBackend
from graphrag.logging import get_logger

log = get_logger(__name__)

# Anything that can write, delete, or redefine. Kùzu-specific verbs included.
FORBIDDEN = frozenset(
    {
        "CREATE", "DELETE", "DETACH", "SET", "MERGE", "REMOVE", "DROP",
        "ALTER", "COPY", "INSTALL", "LOAD", "ATTACH", "EXPORT", "IMPORT",
        "CALL", "BEGIN", "COMMIT", "ROLLBACK", "CHECKPOINT", "USE",
    }
)

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_STRING_LITERAL = re.compile(r"'[^']*'|\"[^\"]*\"")
_LIMIT = re.compile(r"\bLIMIT\s+\d+", re.IGNORECASE)
_ALLOWED_START = ("MATCH", "OPTIONAL", "WITH", "RETURN", "UNWIND")


class UnsafeCypherError(ValueError):
    """The generated query was rejected before execution."""


def strip_fences(text: str) -> str:
    """Models wrap Cypher in markdown fences regardless of instructions."""
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]
            for prefix in ("cypher", "sql", "json"):
                if text.lower().startswith(prefix):
                    text = text[len(prefix) :]
    return text.strip().rstrip(";").strip()


def validate(cypher: str, *, row_limit: int | None = None) -> str:
    """Return a safe, LIMIT-bounded query, or raise UnsafeCypherError."""
    row_limit = row_limit or settings.cypher_row_limit
    query = strip_fences(cypher)

    if not query:
        raise UnsafeCypherError("empty query")

    # Ignore string literals: a dataset legitimately named 'CREATE-Bench' must
    # not trip the guard, while DETACH DELETE outside quotes must.
    scrubbed = _STRING_LITERAL.sub("''", query)
    tokens = {t.upper() for t in _TOKEN.findall(scrubbed)}

    banned = tokens & FORBIDDEN
    if banned:
        raise UnsafeCypherError(f"query contains forbidden keyword(s): {sorted(banned)}")

    if not scrubbed.lstrip().upper().startswith(_ALLOWED_START):
        raise UnsafeCypherError(
            f"query must begin with one of {_ALLOWED_START}, got: {query[:60]!r}"
        )

    if not _LIMIT.search(scrubbed):
        query = f"{query}\nLIMIT {row_limit}"

    return query


CYPHER_SYSTEM = """You translate questions into Kùzu Cypher queries over a \
knowledge graph of research papers.

{schema}

RULES
- Return ONLY the Cypher query. No explanation, no markdown fences.
- READ-ONLY. Never use CREATE, MERGE, SET, DELETE, DROP or CALL.
- Always end with a LIMIT clause.
- Entity names are stored as written in the papers. Match them loosely with
  CONTAINS or a lowercase comparison, never assume exact capitalisation.
- Aliases are stored in the `aliases` property as a comma-separated lowercase
  string. To find an entity by an acronym, check both:
    WHERE toLower(m.name) CONTAINS 'dpr' OR m.aliases CONTAINS 'dpr'
- **When counting, always use count(DISTINCT x), never count(x).** Several
  passages can state the same fact, so counting rows over-counts entities.
- Prefer returning names and values over whole nodes.

EXAMPLES

Q: Which datasets does DPR evaluate on?
MATCH (m:Method)-[:EVALUATES_ON]->(d:Dataset)
WHERE toLower(m.name) CONTAINS 'dense passage' OR m.aliases CONTAINS 'dpr'
RETURN DISTINCT d.name AS dataset
LIMIT 50

Q: How many methods are evaluated on Natural Questions?
MATCH (m:Method)-[:EVALUATES_ON]->(d:Dataset)
WHERE toLower(d.name) CONTAINS 'natural questions'
RETURN count(DISTINCT m) AS method_count
LIMIT 1

Q: Which authors wrote about retrieval?
MATCH (p:Paper)-[:AUTHORED_BY]->(a:Author)
WHERE toLower(p.title) CONTAINS 'retrieval'
RETURN DISTINCT a.name AS author
LIMIT 50"""


def generate_cypher(
    question: str,
    *,
    backend: LLMBackend,
    model: str | None = None,
    error_feedback: str | None = None,
) -> str:
    """Ask the model for Cypher. ``error_feedback`` drives the single retry."""
    user = question
    if error_feedback:
        user = (
            f"{question}\n\n"
            f"Your previous query failed with:\n{error_feedback}\n"
            f"Return a corrected query."
        )

    response = backend.complete(
        system=CYPHER_SYSTEM.format(schema=schema_description()),
        user=user,
        model=model or settings.cypher_model,
        max_tokens=800,
        stage="cypher",
    )
    return response.text


def run_graph_query(
    question: str,
    *,
    backend: LLMBackend,
    store: GraphStore | None = None,
    model: str | None = None,
) -> tuple[list[dict], str | None]:
    """Generate, validate and execute Cypher. Returns (rows, query_used).

    On failure the query is regenerated once with the error text appended --
    invalid Cypher is usually a small schema mistake the model can fix when
    shown the message. A second failure returns no rows so the caller can fall
    back to vector retrieval rather than surfacing an error to the user.
    """
    store = store or GraphStore(read_only=True)
    feedback: str | None = None

    for attempt in (1, 2):
        raw = generate_cypher(
            question, backend=backend, model=model, error_feedback=feedback
        )
        try:
            query = validate(raw)
            rows = store.query(query)
            log.info("graph_query_ok", attempt=attempt, rows=len(rows))
            return rows, query
        except UnsafeCypherError as exc:
            log.warning("cypher_rejected", attempt=attempt, reason=str(exc)[:200])
            feedback = f"The query was rejected: {exc}"
        except Exception as exc:  # noqa: BLE001 - execution error, retry once
            log.warning("cypher_execution_failed", attempt=attempt, error=str(exc)[:200])
            feedback = str(exc)[:400]

    log.warning("graph_query_gave_up", question=question[:80])
    return [], None
