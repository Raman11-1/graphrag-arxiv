"""Kùzu DDL for the knowledge graph.

Two design rules run through this schema:

**Provenance is not optional.** Every relationship carries ``window_id``,
``char_start``, ``char_end``, ``evidence`` and ``confidence``. A claim in an
answer must be traceable to the sentence that produced it, otherwise citations
are decoration.

**Entity types are closed.** Kùzu requires declared node and rel tables, which
is a feature here: it makes it impossible to accidentally create `PROPOSES`,
`proposes` and `introduces` as three different edges, which would silently
break every multi-hop query.
"""

from __future__ import annotations

import re

# Entity node tables. `name` is the primary key after resolution has merged
# aliases, so "DPR" and "Dense Passage Retrieval" become one node.
NODE_TABLES: dict[str, str] = {
    "Paper": """
        CREATE NODE TABLE IF NOT EXISTS Paper(
            id STRING,
            title STRING,
            abstract STRING,
            published DATE,
            categories STRING,
            PRIMARY KEY(id)
        )
    """,
    "Author": """
        CREATE NODE TABLE IF NOT EXISTS Author(
            name STRING,
            aliases STRING,
            PRIMARY KEY(name)
        )
    """,
    "Institution": """
        CREATE NODE TABLE IF NOT EXISTS Institution(
            name STRING,
            aliases STRING,
            PRIMARY KEY(name)
        )
    """,
    "Task": """
        CREATE NODE TABLE IF NOT EXISTS Task(
            name STRING,
            description STRING,
            aliases STRING,
            PRIMARY KEY(name)
        )
    """,
    "Method": """
        CREATE NODE TABLE IF NOT EXISTS Method(
            name STRING,
            description STRING,
            aliases STRING,
            PRIMARY KEY(name)
        )
    """,
    "Dataset": """
        CREATE NODE TABLE IF NOT EXISTS Dataset(
            name STRING,
            description STRING,
            aliases STRING,
            PRIMARY KEY(name)
        )
    """,
    "Metric": """
        CREATE NODE TABLE IF NOT EXISTS Metric(
            name STRING,
            aliases STRING,
            PRIMARY KEY(name)
        )
    """,
    # Chunks live in the graph purely as provenance anchors, so a graph result
    # can be hydrated back into the source text that a citation points at.
    "Chunk": """
        CREATE NODE TABLE IF NOT EXISTS Chunk(
            chunk_id STRING,
            paper_id STRING,
            char_start INT64,
            char_end INT64,
            section STRING,
            PRIMARY KEY(chunk_id)
        )
    """,
    "Community": """
        CREATE NODE TABLE IF NOT EXISTS Community(
            community_id STRING,
            level INT64,
            title STRING,
            summary STRING,
            size INT64,
            PRIMARY KEY(community_id)
        )
    """,
}

# Shared provenance columns on every extracted relationship.
_PROV = """
    window_id STRING,
    char_start INT64,
    char_end INT64,
    evidence STRING,
    confidence DOUBLE
"""

REL_TABLES: dict[str, str] = {
    "AUTHORED_BY": f"""
        CREATE REL TABLE IF NOT EXISTS AUTHORED_BY(
            FROM Paper TO Author, ordinal INT64, {_PROV}
        )
    """,
    "AFFILIATED_WITH": f"""
        CREATE REL TABLE IF NOT EXISTS AFFILIATED_WITH(
            FROM Author TO Institution, {_PROV}
        )
    """,
    "ADDRESSES": f"""
        CREATE REL TABLE IF NOT EXISTS ADDRESSES(
            FROM Paper TO Task, {_PROV}
        )
    """,
    "PROPOSES": f"""
        CREATE REL TABLE IF NOT EXISTS PROPOSES(
            FROM Paper TO Method, {_PROV}
        )
    """,
    # Multiple FROM/TO pairs in one table: Kùzu supports this, and it keeps
    # "uses" a single relation name rather than USES_METHOD / USES_DATASET.
    "USES": f"""
        CREATE REL TABLE IF NOT EXISTS USES(
            FROM Paper TO Method,
            FROM Paper TO Dataset,
            FROM Method TO Dataset,
            {_PROV}
        )
    """,
    "EVALUATES_ON": f"""
        CREATE REL TABLE IF NOT EXISTS EVALUATES_ON(
            FROM Paper TO Dataset,
            FROM Method TO Dataset,
            {_PROV}
        )
    """,
    # The quantitative edge -- this is what makes "best F1 on NQ" answerable.
    "REPORTS": f"""
        CREATE REL TABLE IF NOT EXISTS REPORTS(
            FROM Paper TO Metric,
            value DOUBLE,
            dataset STRING,
            method STRING,
            {_PROV}
        )
    """,
    "BUILDS_ON": f"""
        CREATE REL TABLE IF NOT EXISTS BUILDS_ON(
            FROM Method TO Method, {_PROV}
        )
    """,
    "COMPARES_TO": f"""
        CREATE REL TABLE IF NOT EXISTS COMPARES_TO(
            FROM Method TO Method, {_PROV}
        )
    """,
    # Provenance edge: which chunk mentions which entity.
    "MENTIONED_IN": """
        CREATE REL TABLE IF NOT EXISTS MENTIONED_IN(
            FROM Method TO Chunk,
            FROM Dataset TO Chunk,
            FROM Task TO Chunk,
            FROM Author TO Chunk,
            FROM Metric TO Chunk,
            FROM Institution TO Chunk,
            FROM Paper TO Chunk
        )
    """,
    "IN_COMMUNITY": """
        CREATE REL TABLE IF NOT EXISTS IN_COMMUNITY(
            FROM Method TO Community,
            FROM Dataset TO Community,
            FROM Task TO Community,
            FROM Author TO Community,
            FROM Paper TO Community
        )
    """,
}

_FROM_TO = re.compile(r"FROM\s+(\w+)\s+TO\s+(\w+)")
# Trailing comma is optional: all but the last property line ends with one.
_PROP = re.compile(r"^\s*(\w+)\s+(?:STRING|INT64|DOUBLE|DATE)\s*,?\s*$", re.MULTILINE)


# Table -> primary key column. Used to build MATCH and MERGE clauses.
#
# Every node table must appear here. Chunk and Community are keyed differently
# from the entity tables, and omitting them made upsert_entity fall back to
# "name", which raises "Cannot find property name" at write time rather than
# anywhere near the mistake.
ENTITY_KEY: dict[str, str] = {
    "Paper": "id",
    "Author": "name",
    "Institution": "name",
    "Task": "name",
    "Method": "name",
    "Dataset": "name",
    "Metric": "name",
    "Chunk": "chunk_id",
    "Community": "community_id",
}

# The subset that extraction may produce. Chunk and Community are structural,
# never extracted, so they are excluded from entity-writing paths.
EXTRACTABLE_TYPES: frozenset[str] = frozenset(
    {"Author", "Institution", "Task", "Method", "Dataset", "Metric"}
)


def ddl_statements() -> list[str]:
    """All DDL, node tables before rel tables (rels reference node tables)."""
    return [*NODE_TABLES.values(), *REL_TABLES.values()]


def schema_description() -> str:
    """A compact schema summary for the text-to-Cypher prompt.

    The generator needs the live schema, not a hard-coded copy -- otherwise a
    schema change silently starts producing invalid Cypher.
    """
    lines = ["Node tables:"]
    for name in NODE_TABLES:
        key = ENTITY_KEY.get(name, "chunk_id" if name == "Chunk" else "community_id")
        lines.append(f"  ({name}) primary key: {key}")

    lines.append("")
    lines.append("Relationship tables:")
    for name, ddl in REL_TABLES.items():
        # Regex, not a comma split: the DDL is multi-line and FROM/TO pairs do
        # not align with commas, which silently dropped the first pair of every
        # table and leaked a stray ')' into the prompt.
        pairs = [f"{src} -> {dst}" for src, dst in _FROM_TO.findall(ddl)]
        props = [p for p in _PROP.findall(ddl)]
        line = f"  -[:{name}]-> {'; '.join(pairs)}"
        if props:
            line += f"   [props: {', '.join(props)}]"
        lines.append(line)

    return "\n".join(lines)
