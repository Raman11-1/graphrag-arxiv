"""Kùzu-backed knowledge graph.

Every write is parameterised. Entity names come from LLM output and land inside
Cypher, so string interpolation here would be a query-injection hole reachable
from paper text -- a paper containing a crafted method name could rewrite the
graph. Parameters also spare us quote-escaping entirely.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from graphrag.config import settings
from graphrag.graph.schema import ENTITY_KEY, ddl_statements
from graphrag.logging import get_logger

log = get_logger(__name__)


class GraphStore:
    """Thin wrapper over a Kùzu database with the project's schema applied."""

    def __init__(self, path: str | Path | None = None, *, read_only: bool = False) -> None:
        import kuzu

        self.path = Path(path or settings.kuzu_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = kuzu.Database(str(self.path), read_only=read_only)
        self._conn = kuzu.Connection(self._db)
        if not read_only:
            self._apply_schema()

    def _apply_schema(self) -> None:
        for statement in ddl_statements():
            self._conn.execute(statement)

    # -- querying -------------------------------------------------------

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict]:
        """Run Cypher and materialise rows as dicts keyed by column name."""
        result = self._conn.execute(cypher, parameters=params or {})
        columns = result.get_column_names()
        rows = []
        while result.has_next():
            rows.append(dict(zip(columns, result.get_next(), strict=False)))
        return rows

    # -- writing --------------------------------------------------------

    def upsert_entity(self, table: str, key_value: str, props: dict[str, Any]) -> None:
        """Create the node if absent, then set its properties.

        Kùzu has no single MERGE-with-ON-CREATE form here, so this is a MERGE
        followed by SET. Properties are only overwritten when the incoming value
        is non-empty, so a later window with a sparser mention cannot blank out
        a description an earlier window captured.
        """
        key = ENTITY_KEY.get(table, "name")
        self._conn.execute(
            f"MERGE (n:{table} {{{key}: $key}})",
            parameters={"key": key_value},
        )
        for prop, value in props.items():
            if value in (None, "", []):
                continue
            self._conn.execute(
                f"MATCH (n:{table}) WHERE n.{key} = $key SET n.{prop} = $value",
                parameters={"key": key_value, "value": value},
            )

    def add_relation(
        self,
        *,
        rel: str,
        from_table: str,
        from_key: str,
        to_table: str,
        to_key: str,
        props: dict[str, Any],
    ) -> bool:
        """Create one relationship. Returns False if either endpoint is missing.

        Endpoints are checked rather than created: extraction sometimes names a
        relation object that never appeared in its own entity list, and silently
        conjuring a node for it produces a graph full of half-defined entities.
        """
        fkey = ENTITY_KEY.get(from_table, "name")
        tkey = ENTITY_KEY.get(to_table, "name")

        assignments = ", ".join(f"r.{p} = ${p}" for p in props)
        set_clause = f" SET {assignments}" if assignments else ""

        try:
            self._conn.execute(
                f"MATCH (a:{from_table}), (b:{to_table}) "
                f"WHERE a.{fkey} = $from_key AND b.{tkey} = $to_key "
                f"CREATE (a)-[r:{rel}]->(b){set_clause}",
                parameters={"from_key": from_key, "to_key": to_key, **props},
            )
            return True
        except Exception as exc:  # noqa: BLE001 - a bad triple must not stop the load
            log.debug(
                "relation_rejected",
                rel=rel,
                subject=from_key,
                object=to_key,
                error=str(exc)[:120],
            )
            return False

    def has_node(self, table: str, key_value: str) -> bool:
        key = ENTITY_KEY.get(table, "name")
        rows = self.query(
            f"MATCH (n:{table}) WHERE n.{key} = $key RETURN count(n) AS c",
            {"key": key_value},
        )
        return bool(rows and rows[0]["c"] > 0)

    # -- introspection ---------------------------------------------------

    def counts(self) -> dict[str, int]:
        """Node and relationship counts per table, for /stats and sanity checks."""
        from graphrag.graph.schema import NODE_TABLES, REL_TABLES

        out: dict[str, int] = {}
        for table in NODE_TABLES:
            rows = self.query(f"MATCH (n:{table}) RETURN count(n) AS c")
            if rows and rows[0]["c"]:
                out[table] = rows[0]["c"]
        for rel in REL_TABLES:
            try:
                rows = self.query(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c")
                if rows and rows[0]["c"]:
                    out[f"-[:{rel}]->"] = rows[0]["c"]
            except Exception:  # noqa: BLE001 - empty rel tables can raise
                continue
        return out

    def close(self) -> None:
        self._conn.close()
