"""End-to-end dispatch and fallback logic, with every dependency stubbed.

No network, no model, no index. What is under test is the control flow: which
retrievers run for which mode, and whether the fallbacks actually fire.

The fallback behaviour is the reason this file exists. The design promise is
that a routing mistake or an empty graph result degrades the answer rather than
breaking the query -- and that promise lives entirely in branching that no
other test exercises.
"""

from __future__ import annotations

import pytest

from graphrag.answer.synthesize import Answer
from graphrag.index.vector_store import Hit
from graphrag.retrieve import pipeline as pipe


def make_hit(chunk_id: str, source: str = "vector") -> Hit:
    return Hit(
        chunk_id=chunk_id,
        paper_id="p1",
        text=f"passage {chunk_id}",
        score=1.0,
        char_start=0,
        char_end=10,
        source=source,
    )


@pytest.fixture
def stubs(monkeypatch):
    """Record which retrievers ran, and let each test control what they return."""
    calls: dict[str, int] = {"search": 0, "graph": 0, "global": 0}
    state = {"graph_rows": [], "global_hits": [], "vector_hits": [make_hit("c1")]}

    def fake_search(question, k=8, mode="hybrid"):
        calls["search"] += 1
        return list(state["vector_hits"])

    def fake_graph(question, *, backend, store=None, model=None):
        calls["graph"] += 1
        rows = state["graph_rows"]
        return rows, ("MATCH (n) RETURN n LIMIT 1" if rows else None)

    def fake_global(question, *, store=None, k=6):
        calls["global"] += 1
        return list(state["global_hits"])

    def fake_synthesize(question, hits, *, backend, model=None, mode="vector", max_tokens=1500):
        return Answer(text=f"answer from {len(hits)} sources", citations=list(hits), mode=mode)

    monkeypatch.setattr(pipe, "search", fake_search)
    monkeypatch.setattr(pipe, "run_graph_query", fake_graph)
    monkeypatch.setattr(pipe, "global_search", fake_global)
    monkeypatch.setattr(pipe, "synthesize", fake_synthesize)
    return calls, state


# --- dispatch ---------------------------------------------------------


def test_vector_mode_uses_only_the_passage_retriever(stubs):
    calls, _ = stubs
    result = pipe.ask("q", backend=object(), force_mode="vector")

    assert calls == {"search": 1, "graph": 0, "global": 0}
    assert result.mode == "vector"
    assert not result.fell_back


def test_graph_mode_skips_passage_retrieval_when_the_graph_answers(stubs):
    calls, state = stubs
    state["graph_rows"] = [{"dataset": "Natural Questions"}]

    result = pipe.ask("q", backend=object(), force_mode="graph")

    assert calls["graph"] == 1
    assert calls["search"] == 0, "a successful graph query should not also retrieve passages"
    assert result.cypher
    assert not result.fell_back


def test_hybrid_mode_runs_both_retrievers(stubs):
    calls, state = stubs
    state["graph_rows"] = [{"x": 1}]

    pipe.ask("q", backend=object(), force_mode="hybrid")

    assert calls["graph"] == 1
    assert calls["search"] == 1


def test_global_mode_uses_community_summaries(stubs):
    calls, state = stubs
    state["global_hits"] = [make_hit("community:c0", source="global")]

    result = pipe.ask("q", backend=object(), force_mode="global")

    assert calls["global"] == 1
    assert calls["search"] == 0
    assert not result.fell_back


# --- fallbacks --------------------------------------------------------


def test_an_empty_graph_result_falls_back_to_passages(stubs):
    calls, state = stubs
    state["graph_rows"] = []

    result = pipe.ask("q", backend=object(), force_mode="graph")

    assert calls["search"] == 1
    assert result.fell_back is True
    assert result.answer.text != ""


def test_missing_communities_fall_back_to_passages(stubs):
    calls, state = stubs
    state["global_hits"] = []

    result = pipe.ask("q", backend=object(), force_mode="global")

    assert calls["global"] == 1
    assert calls["search"] == 1
    assert result.fell_back is True


def test_a_fallback_still_produces_an_answer_not_an_error(stubs):
    _, state = stubs
    state["graph_rows"] = []
    state["vector_hits"] = []

    result = pipe.ask("q", backend=object(), force_mode="graph")
    assert isinstance(result.answer.text, str)


# --- result shape -----------------------------------------------------


def test_all_retrieved_hits_are_exposed_for_evaluation(stubs):
    """The benchmark measures over `hits`, not over the cited subset."""
    _, state = stubs
    state["vector_hits"] = [make_hit("c1"), make_hit("c2"), make_hit("c3")]

    result = pipe.ask("q", backend=object(), force_mode="vector")
    assert [h.chunk_id for h in result.hits] == ["c1", "c2", "c3"]


def test_graph_rows_are_carried_through_to_the_caller(stubs):
    _, state = stubs
    rows = [{"dataset": "NQ"}, {"dataset": "TriviaQA"}]
    state["graph_rows"] = rows

    result = pipe.ask("q", backend=object(), force_mode="graph")
    assert result.graph_rows == rows


def test_graph_rows_are_rendered_as_a_citable_source(stubs):
    _, state = stubs
    state["graph_rows"] = [{"dataset": "Natural Questions"}]

    result = pipe.ask("q", backend=object(), force_mode="graph")
    graph_hits = [h for h in result.hits if h.source == "graph"]
    assert len(graph_hits) == 1
    assert "Natural Questions" in graph_hits[0].text


# --- structured-source writing guidance --------------------------------


def test_graph_sources_get_extra_writing_guidance():
    """Handed a table, a model echoes it back as a bare list.

    That is a correct answer that reads as an incomplete one -- in the
    benchmark it cost the multi-hop category outright.
    """
    # NB: `from graphrag.answer import synthesize` gets the re-exported
    # *function*, not the module -- import the function directly.
    from graphrag.answer.synthesize import synthesize as run_synthesize

    captured = {}

    class Backend:
        name = "capture"

        def complete(self, *, system, user, model, max_tokens=4096, **kw):
            from graphrag.llm.base import LLMResponse, Usage

            captured["system"] = system
            return LLMResponse(text="an answer [1]", usage=Usage(), model=model)

        def parse(self, **kw):
            raise NotImplementedError

    graph_hit = Hit(
        chunk_id="graph-result", paper_id="knowledge-graph", text="1. dataset=NQ",
        score=1.0, char_start=0, char_end=0, source="graph",
    )
    run_synthesize("q", [graph_hit], backend=Backend(), mode="graph")
    assert "Answer in prose" in captured["system"]

    captured.clear()
    run_synthesize("q", [make_hit("c1")], backend=Backend(), mode="vector")
    assert "Answer in prose" not in captured["system"], (
        "passage-only answers should not get the structured-source rules"
    )
