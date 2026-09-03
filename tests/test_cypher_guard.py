"""The Cypher mutation guard.

An LLM writes these queries and they run against a live database, so this is
the security boundary of the whole system. The tests below are the actual
attacks worth defending against, plus the false positives a naive substring
check would produce.
"""

from __future__ import annotations

import pytest

from graphrag.retrieve.cypher import (
    FORBIDDEN,
    UnsafeCypherError,
    strip_fences,
    validate,
)

DESTRUCTIVE = [
    "MATCH (n) DETACH DELETE n",
    "MATCH (n) DELETE n",
    "CREATE (m:Method {name: 'pwned'})",
    "MATCH (m:Method) SET m.name = 'x' RETURN m",
    "MERGE (m:Method {name: 'x'})",
    "DROP TABLE Method",
    "MATCH (m:Method) REMOVE m.name RETURN m",
    "CALL db.schema() RETURN *",
    "COPY Method FROM 'evil.csv'",
    "MATCH (m:Method) RETURN m; DROP TABLE Method",
]


@pytest.mark.parametrize("query", DESTRUCTIVE, ids=lambda q: q[:28])
def test_destructive_queries_are_rejected(query):
    with pytest.raises(UnsafeCypherError):
        validate(query)


def test_every_forbidden_keyword_is_actually_caught():
    """Guards against someone adding a keyword to the set but not the matcher."""
    for keyword in FORBIDDEN:
        with pytest.raises(UnsafeCypherError):
            validate(f"MATCH (n) {keyword} n RETURN n LIMIT 5")


def test_case_and_spacing_do_not_evade_the_guard():
    for variant in [
        "match (n) detach delete n",
        "MaTcH (n) DeTaCh DeLeTe n",
        "MATCH (n)\n\tDETACH\n\tDELETE n",
    ]:
        with pytest.raises(UnsafeCypherError):
            validate(variant)


def test_a_dataset_named_like_a_keyword_is_not_a_false_positive():
    """Substring matching would reject this legitimate query."""
    query = validate(
        "MATCH (d:Dataset) WHERE d.name = 'CREATE-Bench' RETURN d.name LIMIT 10"
    )
    assert "CREATE-Bench" in query


def test_keyword_inside_a_double_quoted_literal_is_allowed():
    query = validate('MATCH (m:Method) WHERE m.name = "DELETE-2020" RETURN m.name LIMIT 5')
    assert "DELETE-2020" in query


def test_query_must_start_with_a_read_clause():
    with pytest.raises(UnsafeCypherError):
        validate("EXPLAIN MATCH (n) RETURN n LIMIT 1")


def test_limit_is_added_when_missing():
    out = validate("MATCH (m:Method) RETURN m.name")
    assert "LIMIT" in out.upper()


def test_existing_limit_is_preserved():
    out = validate("MATCH (m:Method) RETURN m.name LIMIT 7")
    assert out.upper().count("LIMIT") == 1
    assert "LIMIT 7" in out


def test_empty_query_is_rejected():
    with pytest.raises(UnsafeCypherError):
        validate("   ")


# --- fence stripping --------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "```cypher\nMATCH (m:Method) RETURN m.name LIMIT 5\n```",
        "```\nMATCH (m:Method) RETURN m.name LIMIT 5\n```",
        "MATCH (m:Method) RETURN m.name LIMIT 5;",
        "  MATCH (m:Method) RETURN m.name LIMIT 5  ",
    ],
)
def test_fences_and_semicolons_are_stripped(raw):
    assert strip_fences(raw) == "MATCH (m:Method) RETURN m.name LIMIT 5"


def test_valid_read_queries_pass_through():
    for query in [
        "MATCH (m:Method)-[:EVALUATES_ON]->(d:Dataset) RETURN d.name LIMIT 20",
        "OPTIONAL MATCH (p:Paper) RETURN p.title LIMIT 5",
        "MATCH (m:Method) RETURN count(DISTINCT m) AS c LIMIT 1",
    ]:
        assert validate(query)
