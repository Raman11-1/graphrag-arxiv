"""Graph construction: alias resolution and edge property consistency.

The subtle case is edge *properties*. A relation endpoint must reference a real
node, so it goes through resolution. A result's `method` and `dataset` are
stored as properties on the edge instead, which means they bypass node
resolution entirely -- and a query filtering on them silently misses rows
written under a different surface form.
"""

from __future__ import annotations

import pytest

from graphrag.graph.build import (
    _resolve,
    _resolve_or_raw,
    build_alias_map,
    is_valid_entity_name,
)


def record(entities, relations=None, results=None, wid="w1", pid="p1"):
    return {
        "window_id": wid,
        "paper_id": pid,
        "char_start": 0,
        "char_end": 100,
        "status": "ok",
        "extraction": {
            "entities": entities,
            "relations": relations or [],
            "results": results or [],
        },
    }


def entity(name, etype="Method", aliases=None):
    return {"name": name, "type": etype, "aliases": aliases or []}


# --- alias map --------------------------------------------------------


def test_declared_aliases_map_to_the_canonical_name():
    aliases, types = build_alias_map(
        [record([entity("Dense Passage Retriever", aliases=["DPR"])])]
    )
    assert aliases["dpr"] == "Dense Passage Retriever"
    assert aliases["dense passage retriever"] == "Dense Passage Retriever"
    assert types["Dense Passage Retriever"] == "Method"


def test_the_longest_surface_form_becomes_canonical():
    """The spelled-out name is the useful node identity, not the acronym."""
    aliases, _ = build_alias_map(
        [record([entity("BM25", aliases=["Best Match 25 ranking function"])])]
    )
    assert aliases["bm25"] == "BM25"


def test_lookup_is_case_insensitive():
    aliases, _ = build_alias_map([record([entity("BERT")])])
    assert _resolve("bert", aliases) == "BERT"
    assert _resolve("BERT", aliases) == "BERT"
    assert _resolve("  BeRt  ", aliases) == "BERT"


def test_the_most_frequent_type_wins_over_a_one_off_mislabel():
    records = [
        record([entity("FAISS", "Method")]),
        record([entity("FAISS", "Method")], wid="w2"),
        record([entity("FAISS", "Dataset")], wid="w3"),
    ]
    _, types = build_alias_map(records)
    assert types["FAISS"] == "Method"


def test_failed_records_contribute_nothing():
    bad = record([entity("Ghost")])
    bad["status"] = "failed"
    aliases, types = build_alias_map([bad])
    assert aliases == {} and types == {}


# --- edge properties --------------------------------------------------


def test_edge_properties_are_resolved_through_aliases():
    """Otherwise 'NQ' and 'Natural Questions' become different filter values."""
    aliases = {"nq": "Natural Questions", "natural questions": "Natural Questions"}
    assert _resolve_or_raw("NQ", aliases) == "Natural Questions"
    assert _resolve_or_raw("natural questions", aliases) == "Natural Questions"


def test_an_unknown_edge_property_is_kept_rather_than_dropped():
    """Unlike an endpoint, an unresolvable property is still worth storing."""
    assert _resolve_or_raw("Some Unseen Benchmark", {}) == "Some Unseen Benchmark"


def test_blank_edge_properties_become_empty_strings():
    assert _resolve_or_raw("", {}) == ""
    assert _resolve_or_raw(None, {}) == ""


def test_unresolvable_endpoints_return_none_so_the_edge_is_dropped():
    """Endpoints are stricter: a missing node must not be silently invented."""
    assert _resolve("Never Mentioned", {}) is None
    assert _resolve("", {}) is None


# --- entity name validity ----------------------------------------------
#
# Found by running Leiden on the real graph: communities contained members
# like "+ (ft, T-REx)" and "+ DAiD (ours)". Those are modifier columns from an
# ablation table, not entities, and they made the clustering look noisier than
# it was.


@pytest.mark.parametrize(
    "name",
    ["+ (ft, T-REx)", "+ DAiD (ours)", "(only)", "---", "42", "3.5", "", " ", "x"],
)
def test_table_fragments_are_rejected(name):
    assert is_valid_entity_name(name) is False


@pytest.mark.parametrize(
    "name",
    ["BM25", "DPR", "T5", "2Wiki-MultiHopQA", "GPT-4", "Dense Passage Retrieval"],
)
def test_real_entity_names_are_accepted(name):
    """Names starting with a digit are fine -- '2Wiki-MultiHopQA' is a dataset."""
    assert is_valid_entity_name(name) is True


def test_invalid_names_never_reach_the_alias_map():
    aliases, types = build_alias_map(
        [record([entity("+ (ft, T-REx)"), entity("BM25")])]
    )
    assert "bm25" in aliases
    assert not any(k.startswith("+") for k in aliases)
    assert list(types) == ["BM25"]


def test_invalid_aliases_are_dropped_but_the_entity_survives():
    aliases, _ = build_alias_map(
        [record([entity("Dense Passage Retrieval", aliases=["DPR", "+ variant"])])]
    )
    assert aliases["dpr"] == "Dense Passage Retrieval"
    assert "+ variant" not in aliases


# --- config stripping happens at alias-map time --------------------------
#
# A regression I introduced. Stripping "DPR (rt, NQ)" -> "DPR" during
# *resolution* produced a canonical named "DPR" that never met the existing
# "dpr" alias of "Dense Passage Retriever" -- leaving the entity split in two,
# which is the exact bug the stripping was meant to help with. The strip has to
# happen while the alias map is built, so every surface form lands on one key.


def test_ablation_variants_fold_into_the_spelled_out_name():
    aliases, types = build_alias_map(
        [
            record([entity("Dense Passage Retriever", aliases=["DPR"])]),
            record([entity("DPR (rt, NQ)"), entity("DPR (pt, Multi)")], wid="w2"),
        ]
    )
    assert list(types) == ["Dense Passage Retriever"], "DPR must not be a second node"
    assert aliases["dpr"] == "Dense Passage Retriever"


def test_a_relation_naming_a_variant_still_resolves():
    """Lookup must use the same key form the alias map was built with,
    otherwise the edge is silently dropped as dangling."""
    aliases, _ = build_alias_map(
        [record([entity("Dense Passage Retriever", aliases=["DPR"])])]
    )
    assert _resolve("DPR (rt, NQ, 1enc)", aliases) == "Dense Passage Retriever"
    assert _resolve("DPR", aliases) == "Dense Passage Retriever"


def test_edge_properties_resolve_variants_too():
    aliases, _ = build_alias_map(
        [record([entity("Dense Passage Retriever", aliases=["DPR"])])]
    )
    assert _resolve_or_raw("DPR (rt, PAQ)", aliases) == "Dense Passage Retriever"


def test_canonical_name_is_never_an_ablation_row():
    """The second half of the same regression.

    Once config-stripping groups "DPR", "Dense Passage Retriever" and
    "DPR (rt, PAQ, 1enc, stopG) + qsft" under one key, plain longest-wins
    elects the ablation row as the entity's name -- and every answer then
    reports that string as the method.
    """
    aliases, types = build_alias_map(
        [
            record([entity("DPR"), entity("DPR (rt, PAQ, 1enc, stopG) + qsft")]),
            record([entity("Dense Passage Retriever", aliases=["DPR"])], wid="w2"),
        ]
    )
    assert aliases["dpr"] == "Dense Passage Retriever"
    assert not any("(" in n for n in types), "an ablation row became an entity name"


def test_longest_still_wins_among_unqualified_names():
    """The original rule must survive: spell-out beats acronym."""
    aliases, _ = build_alias_map(
        [record([entity("Dense Passage Retriever", aliases=["DPR"])])]
    )
    assert aliases["dpr"] == "Dense Passage Retriever"


def test_a_group_of_only_qualified_names_still_gets_one():
    """If every surface form carries a suffix, fall back rather than crash."""
    _, types = build_alias_map([record([entity("Foo (a)"), entity("Foo (b)")])])
    assert len(types) == 1
