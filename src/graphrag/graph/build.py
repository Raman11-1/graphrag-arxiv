"""Load extracted triples into the graph.

Two rules keep the graph honest:

**Alias resolution before insertion.** "DPR" and "Dense Passage Retriever" must
become one node or every multi-hop query silently misses half its paths. We
build an alias -> canonical map from the extraction output itself, preferring
the longest surface form as canonical since that is the spelled-out name.

**Dangling relations are dropped, not invented.** Extraction sometimes names a
relation endpoint that never appeared in its own entity list. Auto-creating
those nodes fills the graph with untyped, undescribed entities, so they are
rejected and counted instead.
"""

from __future__ import annotations

import re
from collections import defaultdict

from graphrag.extract.extractor import load_records
from graphrag.extract.resolver import resolve_entities, strip_config_suffix
from graphrag.graph.schema import EXTRACTABLE_TYPES
from graphrag.graph.store import GraphStore
from graphrag.logging import get_logger
from graphrag.models import ParsedPaper

log = get_logger(__name__)


# An entity name must begin with a letter or digit. Extraction occasionally
# emits a table-row fragment as an entity -- "+ (ft, T-REx)", "+ DAiD (ours)" --
# which is a modifier column, not a thing. They survive into communities and
# make the graph look noisier than it is.
_VALID_NAME = re.compile(r"^[A-Za-z0-9]")


def is_valid_entity_name(name: str) -> bool:
    """Reject fragments that are not entity names."""
    name = (name or "").strip()
    if len(name) < 2 or len(name) > 200:
        return False
    if not _VALID_NAME.match(name):
        return False
    # A name that is only digits and punctuation is a table cell, not an entity.
    return any(c.isalpha() for c in name)


def build_alias_map(records: list[dict]) -> tuple[dict[str, str], dict[str, str]]:
    """Return (surface form -> canonical name, canonical name -> entity type).

    Matching is case-insensitive on the surface form, because papers write
    "BERT", "Bert" and "bert" interchangeably.
    """
    # canonical candidates per lowercase key, plus the types seen for each.
    surfaces: dict[str, set[str]] = defaultdict(set)
    types: dict[str, list[str]] = defaultdict(list)

    for record in records:
        if record.get("status") != "ok":
            continue
        for entity in record.get("extraction", {}).get("entities", []):
            name = (entity.get("name") or "").strip()
            if not is_valid_entity_name(name):
                continue
            etype = entity.get("type") or "Method"
            # Strip the configuration suffix *here*, while building the alias
            # map -- not later during resolution. Stripping afterwards produced
            # a canonical "DPR" from the ablation variants that never met the
            # existing "dpr" alias of "Dense Passage Retriever", leaving the
            # entity split in two exactly as before.
            group = strip_config_suffix(name).lower()
            surfaces[group].add(name)
            types[group].append(etype)
            for alias in entity.get("aliases", []):
                alias = (alias or "").strip()
                if is_valid_entity_name(alias):
                    alias_key = strip_config_suffix(alias).lower()
                    surfaces[alias_key].add(name)
                    types[alias_key].append(etype)

    alias_to_canonical: dict[str, str] = {}
    canonical_type: dict[str, str] = {}

    for group, names in surfaces.items():
        # Longest surface form is usually the spelled-out name -- prefer
        # "Dense Passage Retriever" over "DPR" as the node identity.
        #
        # But only among names that carry no configuration suffix. Once
        # stripping puts "DPR", "Dense Passage Retriever" and
        # "DPR (rt, PAQ, 1enc, stopG) + qsft" in one group, plain longest-wins
        # elects the most heavily qualified ablation row as the entity's name.
        plain = [n for n in names if strip_config_suffix(n) == n]
        canonical = max(plain or names, key=len)
        alias_to_canonical[group] = canonical
        # Most frequently assigned type wins over a one-off mislabel.
        seen = types[group]
        canonical_type[canonical] = max(set(seen), key=seen.count)

    return alias_to_canonical, canonical_type


def _alias_key(name: str) -> str:
    """The form used as a key in the alias map.

    Must match how ``build_alias_map`` registers surface forms, or a relation
    naming "DPR (rt, NQ)" would fail to resolve and its edge would be dropped.
    """
    return strip_config_suffix((name or "").strip()).lower()


def _resolve(name: str, aliases: dict[str, str]) -> str | None:
    if not name:
        return None
    return aliases.get(_alias_key(name))


def _resolve_or_raw(name: str, aliases: dict[str, str]) -> str:
    """Canonical form if known, otherwise the original string.

    Used for edge *properties* (a result's method and dataset), where an
    unresolvable name is still worth storing -- unlike a relation endpoint,
    which must reference a real node.
    """
    name = (name or "").strip()
    if not name:
        return ""
    return aliases.get(_alias_key(name), name)


def build_graph(
    papers: list[ParsedPaper],
    *,
    store: GraphStore | None = None,
) -> dict[str, int]:
    """Populate the graph from checkpointed extractions."""
    store = store or GraphStore()
    records = load_records()
    ok_records = [r for r in records if r.get("status") == "ok"]

    aliases, canonical_type = build_alias_map(ok_records)

    # Merge surface forms that name the same thing ("Dense Passage Retriever" /
    # "Dense Passage Retrieval"). Without this a central entity splits in two
    # and multi-hop queries silently lose half their paths.
    merged = resolve_entities(canonical_type)
    canonical_type = {
        merged[name]: etype for name, etype in canonical_type.items() if name in merged
    }
    aliases = {surface: merged.get(canon, canon) for surface, canon in aliases.items()}

    # --- papers -------------------------------------------------------
    for parsed in papers:
        p = parsed.paper
        store.upsert_entity(
            "Paper",
            p.paper_id,
            {
                "title": p.title,
                "abstract": p.abstract[:4000],
                "categories": ", ".join(p.categories),
            },
        )
        for ordinal, author in enumerate(p.authors):
            store.upsert_entity("Author", author, {})
            store.add_relation(
                rel="AUTHORED_BY",
                from_table="Paper",
                from_key=p.paper_id,
                to_table="Author",
                to_key=author,
                props={"ordinal": ordinal, "confidence": 1.0, "evidence": "arXiv metadata"},
            )

    # --- entities -----------------------------------------------------
    entities_written = 0
    for canonical, etype in canonical_type.items():
        if etype == "Paper":
            continue  # papers come from arXiv metadata, keyed by ID not title
        if etype not in EXTRACTABLE_TYPES:
            continue
        variants = sorted({k for k, v in aliases.items() if v == canonical})
        store.upsert_entity(etype, canonical, {"aliases": ", ".join(variants)})
        entities_written += 1

    # --- relations ----------------------------------------------------
    written = dropped = 0
    # (rel, subject, object) already inserted. Several windows mention the same
    # fact, and one edge per mention inflates every COUNT -- "how many methods
    # evaluate on NQ" returned 7 for 5 real methods. Provenance for the first
    # mention is kept; the duplicates add no information.
    seen_edges: set[tuple[str, str, str]] = set()

    for record in ok_records:
        paper_id = record["paper_id"]
        prov = {
            "window_id": record["window_id"],
            "char_start": record["char_start"],
            "char_end": record["char_end"],
        }

        for rel in record.get("extraction", {}).get("relations", []):
            subject = _resolve(rel.get("subject", ""), aliases)
            obj = _resolve(rel.get("object", ""), aliases)
            predicate = rel.get("predicate")

            if not subject or not obj or not predicate:
                dropped += 1
                continue

            from_type = canonical_type.get(subject)
            to_type = canonical_type.get(obj)
            if from_type == "Paper":
                from_type, subject = "Paper", paper_id
            if to_type == "Paper":
                to_type, obj = "Paper", paper_id
            if not from_type or not to_type:
                dropped += 1
                continue

            edge = (predicate, subject, obj)
            if edge in seen_edges:
                continue
            seen_edges.add(edge)

            props = {
                **prov,
                "evidence": (rel.get("evidence") or "")[:2000],
                "confidence": float(rel.get("confidence", 0.8)),
            }
            if store.add_relation(
                rel=predicate,
                from_table=from_type,
                from_key=subject,
                to_table=to_type,
                to_key=obj,
                props=props,
            ):
                written += 1
            else:
                dropped += 1

        # --- reported results -> Paper -REPORTS-> Metric --------------
        for claim in record.get("extraction", {}).get("results", []):
            metric = (claim.get("metric") or "").strip()
            if not metric:
                dropped += 1
                continue

            # The method and dataset on this edge are stored as *properties*,
            # so they bypass node resolution entirely. Left raw, one claim says
            # "NQ" and another "Natural Questions", and a query filtering
            # r.dataset silently misses half its rows -- the same alias problem
            # the resolver fixes for nodes, reappearing on edge properties.
            method = _resolve_or_raw(claim.get("method", ""), aliases)
            dataset = _resolve_or_raw(claim.get("dataset", ""), aliases)

            store.upsert_entity("Metric", metric, {})
            if store.add_relation(
                rel="REPORTS",
                from_table="Paper",
                from_key=paper_id,
                to_table="Metric",
                to_key=metric,
                props={
                    **prov,
                    "value": float(claim.get("value", 0.0)),
                    "dataset": dataset[:200],
                    "method": method[:200],
                    "evidence": (claim.get("evidence") or "")[:2000],
                    "confidence": 1.0,
                },
            ):
                written += 1
            else:
                dropped += 1

    summary = {
        "canonical_entities": entities_written,
        "relations_written": written,
        "relations_dropped": dropped,
    }
    log.info("graph_built", **summary)
    return summary
