"""Query routing.

Only the deterministic parts are tested here: the regex pre-filters and the
failure behaviour. LLM classification accuracy belongs in the evaluation
harness, not in a unit test that would cost money and flake.

The property that matters most is the last one: routing must never be able to
break a query. A wrong route should degrade the answer, not raise.
"""

from __future__ import annotations

import pytest

from graphrag.retrieve.router import MODES, RouteDecision, route


class ExplodingBackend:
    name = "exploding"

    def complete(self, **_):
        raise RuntimeError("backend down")

    def parse(self, **_):
        raise RuntimeError("backend down")


class FixedBackend:
    """Returns whatever mode it was constructed with."""

    name = "fixed"

    def __init__(self, mode: str) -> None:
        self._mode = mode

    def complete(self, **_):
        raise NotImplementedError

    def parse(self, **_):
        from graphrag.llm.base import LLMResponse, Usage

        return LLMResponse(
            text="",
            usage=Usage(),
            model="test",
            parsed=RouteDecision(mode=self._mode, reason="fixed"),
        )


@pytest.mark.parametrize(
    "question",
    [
        "How many papers evaluate on MS MARCO?",
        "What is the number of methods using BERT?",
        "Count the datasets in this corpus",
        "Which authors worked on both retrieval and generation?",
    ],
)
def test_counting_questions_shortcut_to_graph(question):
    """These are unambiguous -- spending an LLM call on them is waste."""
    assert route(question, backend=None).mode == "graph"


@pytest.mark.parametrize(
    "question",
    [
        "What are the main themes across the papers?",
        "What are the main topics in this collection?",
        "Summarise the corpus",
    ],
)
def test_corpus_wide_questions_shortcut_to_global(question):
    assert route(question, backend=None).mode == "global"


def test_no_backend_falls_back_to_hybrid():
    """Hybrid runs both retrievers, so it is the safe default."""
    decision = route("How does dense retrieval work?", backend=None)
    assert decision.mode == "hybrid"


def test_a_broken_backend_never_raises():
    """Routing must not be able to break a query."""
    decision = route("How does dense retrieval work?", backend=ExplodingBackend())
    assert decision.mode == "hybrid"
    assert "unavailable" in decision.reason


def test_an_invalid_mode_from_the_model_is_rejected():
    decision = route("How does X work?", backend=FixedBackend("nonsense-mode"))
    assert decision.mode == "hybrid"


@pytest.mark.parametrize("mode", MODES)
def test_every_valid_mode_is_accepted(mode):
    assert route("some question", backend=FixedBackend(mode)).mode == mode


def test_prefilter_beats_the_model():
    """A counting question routes to graph even if the model says otherwise."""
    assert route("How many papers use BERT?", backend=FixedBackend("vector")).mode == "graph"


# --- pipeline mode validation ------------------------------------------


def test_forced_mode_must_be_a_real_mode():
    """An unknown forced mode matches no dispatch branch, so it would silently
    return an empty answer rather than an error."""
    import pytest

    from graphrag.retrieve.pipeline import ask

    with pytest.raises(ValueError, match="Unknown mode"):
        ask("anything", backend=None, force_mode="sql")


@pytest.mark.parametrize("mode", MODES)
def test_every_documented_mode_is_accepted_as_forced(mode):
    """Guards against MODES and the pipeline's branches drifting apart."""
    from graphrag.retrieve.pipeline import ask

    try:
        ask("anything", backend=None, force_mode=mode)
    except ValueError as exc:
        if "Unknown mode" in str(exc):
            raise AssertionError(f"{mode} is in MODES but rejected by the pipeline") from exc
    except Exception:
        pass  # any other failure is a missing index, not a validation problem


# --- corpus-wide routing (benchmark regression) ------------------------
#
# Two of three global questions in the first benchmark fell through to the LLM
# router and were answered as graph queries, costing the category 2.00 vs 3.00.
# The pre-filter only matched a few exact phrasings: "across the" but not
# "across this", and nothing at all for "in this corpus".


@pytest.mark.parametrize(
    "question",
    [
        "What benchmark datasets are used across this collection of papers?",
        "How do the retrieval approaches in this corpus relate to each other?",
        "What application domains do these papers cover?",
        "Give me an overview of the retrieval techniques represented here.",
        "What are the main research themes across these papers?",
    ],
)
def test_corpus_wide_phrasings_route_to_global(question):
    assert route(question, backend=None).mode == "global"


@pytest.mark.parametrize(
    "question",
    [
        "How many papers here work on visual question answering?",
        "Count the distinct evaluation datasets across all of these papers.",
        "Which task is addressed by the largest number of methods in this collection?",
    ],
)
def test_counting_beats_corpus_scope(question):
    """Order matters: these name the corpus *and* ask for a number.

    With the global check first, the corpus-scope phrase wins and a question
    with an exact answer gets sent to the summariser.
    """
    assert route(question, backend=None).mode == "graph"
