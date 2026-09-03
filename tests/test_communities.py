"""Community detection and global retrieval.

Only the deterministic parts are covered: clustering behaviour on a synthetic
graph, and the ranking/fallback logic of global search. Summary *quality*
belongs in the evaluation harness.

The reproducibility test matters more than it looks. Leiden is stochastic, and
an evaluation whose communities silently change between runs would produce
numbers nobody could reproduce -- with no visible symptom.
"""

from __future__ import annotations

from graphrag.graph.communities import (
    MIN_COMMUNITY_SIZE,
    THEMATIC_RELATIONS,
    Community,
    detect_communities,
)
from graphrag.retrieve.global_search import rank_communities


class FakeStore:
    """Returns a fixed edge list for every thematic relation query."""

    def __init__(self, edges: list[tuple[str, str]], rel: str = "EVALUATES_ON") -> None:
        self._edges = edges
        self._rel = rel

    def query(self, cypher: str, params=None):
        if f":{self._rel}]" not in cypher:
            return []
        return [{"a": a, "b": b, "c": 1.0} for a, b in self._edges]


def two_clear_clusters() -> list[tuple[str, str]]:
    """Two dense groups joined by a single weak bridge."""
    left = [("a1", "a2"), ("a2", "a3"), ("a3", "a1"), ("a1", "a4"), ("a4", "a2")]
    right = [("b1", "b2"), ("b2", "b3"), ("b3", "b1"), ("b1", "b4"), ("b4", "b2")]
    return left + right + [("a1", "b1")]


def test_detects_the_obvious_cluster_structure():
    found = detect_communities(FakeStore(two_clear_clusters()), seed=42)
    assert len(found) == 2
    groups = [set(c.members) for c in found]
    assert {"a1", "a2", "a3", "a4"} in groups
    assert {"b1", "b2", "b3", "b4"} in groups


def test_partitioning_is_reproducible():
    """A fixed seed must give the same partition every run."""
    a = detect_communities(FakeStore(two_clear_clusters()), seed=42)
    b = detect_communities(FakeStore(two_clear_clusters()), seed=42)
    assert [sorted(c.members) for c in a] == [sorted(c.members) for c in b]


def test_tiny_groups_are_discarded_as_noise():
    found = detect_communities(FakeStore([("x", "y")]), seed=42)
    assert all(c.size >= MIN_COMMUNITY_SIZE for c in found)


def test_an_empty_graph_yields_no_communities():
    assert detect_communities(FakeStore([])) == []


def test_self_loops_are_ignored():
    edges = two_clear_clusters() + [("a1", "a1")]
    found = detect_communities(FakeStore(edges), seed=42)
    assert all("a1" in c.members or "a1" not in c.members for c in found)
    assert len(found) == 2


def test_communities_are_returned_largest_first():
    edges = [("a1", "a2"), ("a2", "a3"), ("a3", "a1"), ("a1", "a4"), ("a4", "a5"), ("a5", "a1")]
    edges += [("b1", "b2"), ("b2", "b3"), ("b3", "b1")]
    found = detect_communities(FakeStore(edges), seed=42)
    assert found == sorted(found, key=lambda c: c.size, reverse=True)


def test_authorship_is_not_a_thematic_relation():
    """Clustering on authorship would group by lab, not by research topic."""
    assert "AUTHORED_BY" not in THEMATIC_RELATIONS
    assert "MENTIONED_IN" not in THEMATIC_RELATIONS


# --- global search ranking --------------------------------------------


def community(cid: str, title: str, summary: str) -> Community:
    return Community(community_id=cid, title=title, summary=summary)


def test_few_communities_are_all_returned_without_ranking():
    few = [community(f"c{i}", f"Theme {i}", "text") for i in range(3)]
    assert len(rank_communities("anything", few, k=2)) == 3


def test_many_communities_are_ranked_and_truncated():
    many = [
        community("c0", "Dense retrieval for question answering", "embedding based retrieval"),
        community("c1", "Image segmentation networks", "convolutional vision models"),
        community("c2", "Protein folding prediction", "structural biology models"),
        community("c3", "Reinforcement learning control", "policy gradient methods"),
        community("c4", "Speech recognition systems", "acoustic modelling"),
        community("c5", "Graph neural networks", "message passing architectures"),
        community("c6", "Machine translation", "sequence to sequence models"),
    ]
    top = rank_communities("how does dense passage retrieval work?", many, k=3)
    assert len(top) == 3
    assert top[0].community_id == "c0", "the retrieval theme should rank first"
