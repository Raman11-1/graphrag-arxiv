"""API contract tests.

Read-only endpoints only -- nothing here spends money or mutates the corpus.
The validation tests matter most: `mode` and `k` come straight from a client,
and an out-of-range k would be handed to the retriever.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from graphrag.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_reports_the_backend(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["backend"] in ("mistral", "anthropic")


def test_stats_reports_index_and_graph(client):
    body = client.get("/stats").json()
    assert "index" in body
    assert {"papers", "chunks", "extraction_windows"} <= set(body["index"])
    # The graph key must exist even before a graph is built.
    assert "graph" in body


def test_search_needs_a_query(client):
    assert client.get("/search").status_code == 422


def test_search_returns_the_documented_shape(client):
    body = client.get("/search", params={"q": "retrieval", "k": 2}).json()
    assert body["query"] == "retrieval"
    assert isinstance(body["hits"], list)
    for hit in body["hits"]:
        assert {"chunk_id", "paper_id", "score", "preview"} <= set(hit)


def test_communities_endpoint_is_safe_before_communities_exist(client):
    body = client.get("/communities").json()
    assert "count" in body and isinstance(body["communities"], list)


# --- input validation --------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "", "k": 8},          # empty question
        {"question": "x", "k": 0},         # k below range
        {"question": "x", "k": 999},       # k above range
        {"question": "x", "mode": "sql"},  # not a retrieval mode
        {"k": 8},                          # missing question
    ],
)
def test_ask_rejects_bad_input(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_subgraph_rejects_an_unsupported_depth(client):
    r = client.get("/graph/subgraph", params={"entity": "BERT", "depth": 7})
    assert r.status_code == 400


# --- jobs --------------------------------------------------------------


def test_unknown_job_id_is_a_404(client):
    assert client.get("/jobs/does-not-exist").status_code == 404


def test_ingest_rejects_an_out_of_range_limit(client):
    assert client.post("/ingest", json={"query": "x", "limit": 500}).status_code == 422
