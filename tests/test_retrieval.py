"""Fusion and citation-validation tests. No network, no model, no API key."""

from __future__ import annotations

import pytest

from graphrag.answer.synthesize import _validate_citations
from graphrag.index.bm25 import tokenize
from graphrag.index.fuse import reciprocal_rank_fusion
from graphrag.index.vector_store import Hit


def hit(chunk_id: str, score: float = 1.0, source: str = "vector") -> Hit:
    return Hit(
        chunk_id=chunk_id,
        paper_id="p1",
        text=f"text of {chunk_id}",
        score=score,
        char_start=0,
        char_end=10,
        source=source,
    )


# --- RRF --------------------------------------------------------------


def test_rrf_rewards_agreement_between_retrievers():
    """A doc both retrievers like must beat one that only tops a single list."""
    vector = [hit("a"), hit("b"), hit("c")]
    lexical = [hit("d", source="bm25"), hit("a", source="bm25"), hit("b", source="bm25")]

    fused = reciprocal_rank_fusion([vector, lexical])
    assert fused[0].chunk_id == "a"  # ranked 1st and 2nd
    assert {h.chunk_id for h in fused} == {"a", "b", "c", "d"}


def test_rrf_deduplicates_and_records_both_sources():
    fused = reciprocal_rank_fusion([[hit("a")], [hit("a", source="bm25")]])
    assert len(fused) == 1
    assert fused[0].source == "bm25+vector"


def test_rrf_ignores_raw_score_scale():
    """The point of RRF: BM25's unbounded scores must not swamp cosine's 0-1."""
    vector = [hit("a", score=0.9), hit("b", score=0.8)]
    lexical = [hit("b", score=250.0, source="bm25"), hit("a", score=240.0, source="bm25")]

    fused = reciprocal_rank_fusion([vector, lexical])
    # a: ranks 1,2 -> b: ranks 2,1. Symmetric, so scores must tie exactly.
    assert fused[0].score == pytest.approx(fused[1].score)


def test_rrf_respects_limit_and_handles_empties():
    fused = reciprocal_rank_fusion([[hit("a"), hit("b"), hit("c")], []], limit=2)
    assert len(fused) == 2
    assert reciprocal_rank_fusion([[], []]) == []


# --- tokenizer --------------------------------------------------------


def test_tokenizer_keeps_digits():
    """Benchmark names and scores are exactly what lexical search should catch."""
    assert tokenize("MS MARCO scored 78.4 on NQ") == [
        "ms", "marco", "scored", "78", "4", "on", "nq",
    ]


# --- citation validation ----------------------------------------------


def test_valid_citations_are_kept_in_order_without_duplicates():
    hits = [hit("a"), hit("b"), hit("c")]
    cited, dropped = _validate_citations("Claim one [2]. Claim two [1][2].", hits)

    assert [h.chunk_id for h in cited] == ["b", "a"]
    assert dropped == []


def test_out_of_range_citation_is_dropped():
    """A model citing [7] when 3 sources exist must not produce a fake source."""
    hits = [hit("a"), hit("b"), hit("c")]
    cited, dropped = _validate_citations("Grounded [1]. Invented [7].", hits)

    assert [h.chunk_id for h in cited] == ["a"]
    assert dropped == [7]


def test_zero_index_is_rejected():
    hits = [hit("a")]
    cited, dropped = _validate_citations("Bad [0].", hits)
    assert cited == []
    assert dropped == [0]


def test_uncited_answer_yields_no_sources():
    cited, dropped = _validate_citations("A confident but unsourced claim.", [hit("a")])
    assert cited == []
    assert dropped == []
