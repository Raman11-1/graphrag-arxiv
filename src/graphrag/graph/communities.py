"""Community detection and summarisation -- the 'global' retrieval mode.

A question like "what are the main themes?" cannot be answered by retrieving
the 8 most similar chunks, because it is about the corpus rather than any part
of it. Reading everything at query time is too slow and too expensive.

So we do the reading once, in advance:

1. Project the knowledge graph to an undirected entity graph, weighting edges
   by how often each relationship was observed.
2. Partition it with Leiden, which finds groups of entities densely connected
   to each other and sparsely connected to everything else. Those groups *are*
   the themes -- they emerge from the usage and comparison structure of the
   literature rather than from a topic list we imposed.
3. Summarise each community once with an LLM and store the result.

At query time a global question reads the summaries. Cost is O(communities)
once, not O(corpus) per question.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from pydantic import BaseModel

from graphrag.config import settings
from graphrag.graph.store import GraphStore
from graphrag.llm.base import LLMBackend
from graphrag.logging import get_logger

log = get_logger(__name__)

# Relationships indicating two entities belong to the same research theme.
# MENTIONED_IN and AUTHORED_BY are excluded deliberately: co-mention in a chunk
# is weak evidence, and authorship would cluster by lab rather than by topic.
THEMATIC_RELATIONS = (
    "EVALUATES_ON",
    "USES",
    "PROPOSES",
    "BUILDS_ON",
    "COMPARES_TO",
    "ADDRESSES",
)

# A group below this size is noise, not a theme.
MIN_COMMUNITY_SIZE = 3


@dataclass
class Community:
    community_id: str
    members: list[str] = field(default_factory=list)
    title: str = ""
    summary: str = ""

    @property
    def size(self) -> int:
        return len(self.members)


class CommunitySummary(BaseModel):
    title: str
    summary: str


def _edge_weights(store: GraphStore) -> dict[tuple[str, str], float]:
    """Undirected entity co-occurrence weights from thematic relations."""
    weights: dict[tuple[str, str], float] = defaultdict(float)

    for rel in THEMATIC_RELATIONS:
        try:
            rows = store.query(
                f"MATCH (a)-[r:{rel}]->(b) "
                f"RETURN a.name AS a, b.name AS b, r.confidence AS c"
            )
        except Exception as exc:  # noqa: BLE001 - a rel table may be empty
            log.debug("relation_scan_skipped", rel=rel, error=str(exc)[:120])
            continue

        for row in rows:
            a, b = row.get("a"), row.get("b")
            if not a or not b or a == b:
                continue
            key = (a, b) if a < b else (b, a)
            weights[key] += float(row.get("c") or 1.0)

    return dict(weights)


def detect_communities(
    store: GraphStore | None = None,
    *,
    resolution: float = 1.0,
    seed: int = 42,
) -> list[Community]:
    """Partition the entity graph with Leiden.

    ``seed`` is fixed so the same graph always yields the same partition. An
    evaluation whose communities silently change between runs is not
    reproducible, and the difference would be invisible in the results.
    """
    import igraph as ig
    import leidenalg as la

    store = store or GraphStore(read_only=True)
    weights = _edge_weights(store)

    if not weights:
        log.warning("no_thematic_edges", hint="build the graph first")
        return []

    nodes = sorted({n for pair in weights for n in pair})
    index = {name: i for i, name in enumerate(nodes)}

    graph = ig.Graph(n=len(nodes), edges=[(index[a], index[b]) for a, b in weights])
    graph.es["weight"] = list(weights.values())

    partition = la.find_partition(
        graph,
        la.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=seed,
    )

    communities = []
    for i, member_ids in enumerate(partition):
        if len(member_ids) < MIN_COMMUNITY_SIZE:
            continue
        members = sorted(nodes[j] for j in member_ids)
        communities.append(Community(community_id=f"c{i}", members=members))

    communities.sort(key=lambda c: c.size, reverse=True)
    log.info(
        "communities_detected",
        total_partitions=len(partition),
        kept=len(communities),
        entities=len(nodes),
        modularity=round(partition.modularity, 3),
    )
    return communities


SUMMARY_SYSTEM = """You summarise a cluster of related entities from a corpus \
of research papers.

You are given entity names and the relationships connecting them. They were
grouped together because they are densely interconnected in the literature.

Return:
- title: a specific 3-8 word name for this research theme. Name the actual
  subject matter, not a generic label like "Machine Learning Methods".
- summary: 2-4 sentences on what this cluster is about, how the entities relate
  to each other, and what problem the work addresses.

Describe only what the given entities and relations support. Do not add
background knowledge about these systems from elsewhere."""


def _community_context(store: GraphStore, members: list[str], cap: int = 60) -> str:
    """Entity list plus the relations among them, rendered as prompt text."""
    shown = members[:cap]
    header = f"Entities ({len(members)}): " + ", ".join(shown)
    if len(members) > cap:
        header += f", ... and {len(members) - cap} more"

    lines = [header, "", "Relationships:"]
    member_set = set(shown)
    seen = 0

    for rel in THEMATIC_RELATIONS:
        if seen >= 80:
            break
        try:
            rows = store.query(
                f"MATCH (a)-[:{rel}]->(b) RETURN a.name AS a, b.name AS b LIMIT 200"
            )
        except Exception:  # noqa: BLE001 - empty rel table
            continue
        for row in rows:
            if row.get("a") in member_set and row.get("b") in member_set:
                lines.append(f"  {row['a']} -{rel}-> {row['b']}")
                seen += 1
                if seen >= 80:
                    break

    return "\n".join(lines)


def summarise_communities(
    communities: list[Community],
    *,
    backend: LLMBackend,
    store: GraphStore | None = None,
    model: str | None = None,
) -> list[Community]:
    """Attach an LLM-written title and summary to each community."""
    store = store or GraphStore(read_only=True)

    for i, community in enumerate(communities, start=1):
        try:
            response = backend.parse(
                system=SUMMARY_SYSTEM,
                user=_community_context(store, community.members),
                model=model or settings.summary_model,
                schema=CommunitySummary,
                max_tokens=1200,
                stage="community_summary",
            )
            if response.parsed:
                community.title = response.parsed.title
                community.summary = response.parsed.summary
                log.info(
                    "community_summarised",
                    progress=f"{i}/{len(communities)}",
                    size=community.size,
                    title=community.title,
                )
        except Exception as exc:  # noqa: BLE001 - one failure must not stop the rest
            log.warning(
                "community_summary_failed",
                community=community.community_id,
                error=str(exc)[:200],
            )
            # A titled-but-unsummarised community is still usable as a theme.
            community.title = community.title or f"Cluster of {community.size} entities"

    return communities


def persist(communities: list[Community], store: GraphStore) -> int:
    """Write communities into the graph and link their members."""
    written = 0
    for community in communities:
        store.upsert_entity(
            "Community",
            community.community_id,
            {
                "level": 0,
                "title": community.title,
                "summary": community.summary,
                "size": community.size,
            },
        )
        for member in community.members:
            for table in ("Method", "Dataset", "Task", "Author", "Paper"):
                if store.has_node(table, member):
                    store.add_relation(
                        rel="IN_COMMUNITY",
                        from_table=table,
                        from_key=member,
                        to_table="Community",
                        to_key=community.community_id,
                        props={},
                    )
                    break
        written += 1
    log.info("communities_persisted", count=written)
    return written


def load_communities(store: GraphStore) -> list[Community]:
    """Read stored communities back out of the graph, largest first."""
    rows = store.query(
        "MATCH (c:Community) RETURN c.community_id AS id, c.title AS title, "
        "c.summary AS summary, c.size AS size ORDER BY c.size DESC"
    )
    return [
        Community(
            community_id=r["id"],
            title=r.get("title") or "",
            summary=r.get("summary") or "",
            members=[],
        )
        for r in rows
    ]
