"""Entity resolution.

Two opposite failure modes, both silent:

* **Under-merging** splits an entity ("Dense Passage Retriever" vs "...Retrieval")
  so multi-hop queries lose paths through it.
* **Over-merging** fuses distinct entities. This one is worse: it does not just
  hide facts, it fabricates them. Embedding similarity rates "Yang et al. (2018)"
  and "Yang et al. (2019)" at ~0.98, and merging "SQuAD v1.1" with "SQuAD 2.0"
  rewrote which benchmark a paper was evaluated on.
"""

from __future__ import annotations

import pytest

from graphrag.extract.resolver import (
    MIN_TOKEN_OVERLAP,
    digits_conflict,
    normalise,
    resolve_entities,
    strip_config_suffix,
    token_overlap,
)

# --- normalisation ----------------------------------------------------


@pytest.mark.parametrize(
    "a,b",
    [
        ("Dense Passage Retriever", "Dense Passage Retrieval"),
        ("BERT model", "BERT"),
        ("Natural Questions", "Natural Question"),
        ("Retrieval-Augmented Generation", "retrieval augmented generation"),
    ],
)
def test_variants_share_a_normal_form(a, b):
    assert normalise(a) == normalise(b)


@pytest.mark.parametrize(
    "a,b",
    [
        ("Natural Questions", "TriviaQA"),
        ("BERT", "RoBERTa"),
        ("retrieval", "generation"),
    ],
)
def test_distinct_names_do_not_collide(a, b):
    assert normalise(a) != normalise(b)


# --- the digit guard --------------------------------------------------


@pytest.mark.parametrize(
    "a,b",
    [
        ("SQuAD v1.1", "SQuAD 2.0"),
        ("Yang et al. (2018)", "Yang et al. (2019)"),
        ("GPT-3", "GPT-4"),
        ("BERT-base", "BERT-large"),  # no digits in one -> no conflict
    ],
)
def test_differing_numbers_block_a_merge(a, b):
    if any(c.isdigit() for c in a) and any(c.isdigit() for c in b):
        assert digits_conflict(a, b) is True


def test_absent_digits_do_not_block_a_merge():
    """"BERT" must still merge with "BERT model"."""
    assert digits_conflict("BERT", "BERT model") is False
    assert digits_conflict("BM25", "BM25") is False


# --- end-to-end resolution -------------------------------------------


def test_dpr_variants_collapse_to_one_entity():
    mapping = resolve_entities(
        {
            "Dense Passage Retriever": "Method",
            "Dense Passage Retrieval": "Method",
            "BM25": "Method",
        },
        log_path=None,
    )
    assert mapping["Dense Passage Retriever"] == mapping["Dense Passage Retrieval"]
    assert mapping["BM25"] != mapping["Dense Passage Retriever"]


def test_dataset_versions_stay_separate():
    mapping = resolve_entities(
        {"SQuAD v1.1": "Dataset", "SQuAD 2.0": "Dataset"},
        log_path=None,
    )
    assert mapping["SQuAD v1.1"] != mapping["SQuAD 2.0"]


def test_citations_are_never_similarity_merged():
    """Paper-typed names differ only by year; embeddings cannot see that."""
    mapping = resolve_entities(
        {
            "Yang et al. (2018)": "Paper",
            "Yang et al. (2019)": "Paper",
            "Yang et al. (2019b)": "Paper",
        },
        log_path=None,
    )
    assert len(set(mapping.values())) == 3


def test_identical_names_of_different_types_stay_separate():
    """A Method and a Dataset sharing a name are different things.

    Merging across types would also produce edges Kuzu cannot store, since
    every rel table declares specific FROM/TO node tables.
    """
    mapping = resolve_entities(
        {"SQuAD": "Dataset", "SQuAD reader": "Method", "SQuAD readers": "Method"},
        log_path=None,
    )
    # The two Method surface forms merge with each other...
    assert mapping["SQuAD reader"] == mapping["SQuAD readers"]
    # ...but never with the Dataset of a similar name.
    assert mapping["SQuAD"] != mapping["SQuAD reader"]


def test_canonical_is_the_longest_surface_form():
    mapping = resolve_entities(
        {"DPR retriever": "Method", "DPR retrieval model": "Method"},
        log_path=None,
    )
    assert set(mapping.values()) == {"DPR retrieval model"}


def test_every_input_appears_in_the_mapping():
    names = {"A thing": "Method", "Another thing": "Dataset", "Third": "Task"}
    mapping = resolve_entities(names, log_path=None)
    assert set(mapping) >= set(names)


# --- token overlap guard -----------------------------------------------
#
# Found by pre-flighting the resolver against real extracted entities before
# building the graph. Embedding similarity alone merged "Dense Passage
# Retrieval" with "Dense Hierarchical Retrieval" (~0.95 cosine, two shared
# words out of three) and "BERT large baseline" with "BERT base baseline".
# Both would have attributed one method's benchmark scores to another.


@pytest.mark.parametrize(
    "a,b",
    [
        ("Dense Passage Retrieval", "Dense Hierarchical Retrieval"),
        ("BERT large baseline", "BERT base baseline"),
        ("In-Doc negative sampling", "negative sampling"),
        ("sparse retrieval", "dense retrieval"),
    ],
)
def test_names_differing_in_the_identifying_word_do_not_merge(a, b):
    """The differing token is the one carrying the identity, and cosine
    similarity cannot see that."""
    assert token_overlap(a, b) < MIN_TOKEN_OVERLAP


@pytest.mark.parametrize(
    "a,b",
    [
        ("Dense Passage Retriever", "Dense Passage Retrieval"),
        ("BERT model", "BERT"),
        ("dual-encoder framework", "dual encoder model"),
        ("Natural Questions dataset", "Natural Questions"),
    ],
)
def test_true_surface_variants_still_merge(a, b):
    """The guard must not block the variants it was built to catch."""
    assert token_overlap(a, b) >= MIN_TOKEN_OVERLAP


def test_token_overlap_of_empty_names_is_zero_not_a_crash():
    assert token_overlap("", "BERT") == 0.0
    assert token_overlap("", "") == 0.0


def test_similar_methods_survive_full_resolution_as_separate_entities():
    """End-to-end: the exact pair that was being wrongly fused."""
    mapping = resolve_entities(
        {
            "Dense Passage Retrieval": "Method",
            "Dense Hierarchical Retrieval": "Method",
            "Dense Passage Retriever": "Method",
        },
        log_path=None,
    )
    # The two genuine variants merge...
    assert mapping["Dense Passage Retrieval"] == mapping["Dense Passage Retriever"]
    # ...but the different method stays its own entity.
    assert mapping["Dense Hierarchical Retrieval"] != mapping["Dense Passage Retrieval"]


# --- configuration suffixes --------------------------------------------
#
# Extraction emits one entity per ablation row: "DPR (rt, NQ, 1enc)",
# "DHR-D(Abs)+T", "D-NET (Baidu)". 47 of these appeared in a 23-paper corpus,
# all variants of a handful of real methods, and each one inflated the answer
# to "how many distinct methods evaluate on X".


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("DPR (rt, NQ, 1enc)", "DPR"),
        ("DPR (rt, NQ, 1enc, stopG) + qsft", "DPR"),
        ("DHR-D(Abs)+T", "DHR-D"),
        ("D-NET (Baidu)", "D-NET"),
        ("BERT base baseline (MRQA Organizers)", "BERT base baseline"),
    ],
)
def test_configuration_labels_are_stripped(raw, expected):
    assert strip_config_suffix(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "Yang et al. (2018)",
        "Karpukhin et al. (2020)",
        "Lewis et al. (2020)",
    ],
)
def test_citation_years_are_never_stripped(raw):
    """Stripping the year would fuse every paper by the same author."""
    assert strip_config_suffix(raw) == raw


@pytest.mark.parametrize("raw", ["BM25 + DPR", "RAG + reranker", "BM25", "DPR"])
def test_genuine_names_are_left_alone(raw):
    """A hybrid method's '+' is part of its identity, not a config modifier.

    The distinguishing rule: a trailing '+ x' is only stripped when a config
    parenthesis remains behind it.
    """
    assert strip_config_suffix(raw) == raw


def test_a_name_is_never_stripped_to_a_fragment():
    assert strip_config_suffix("(only)") == "(only)"
    assert strip_config_suffix("AB (x)") == "AB (x)"


def test_ablation_variants_resolve_to_the_base_method():
    """End-to-end: the canonical name must be the base, not the longest variant."""
    mapping = resolve_entities(
        {
            "DPR": "Method",
            "DPR (rt, NQ)": "Method",
            "DPR (rt, PAQ, 1enc)": "Method",
            "DPR (rt, NQ, stopG) + qsft": "Method",
        },
        log_path=None,
    )
    assert set(mapping.values()) == {"DPR"}


# --- hyphen boundaries -------------------------------------------------
#
# Surfaced by reading a community's member list: "2Wiki-MultiHopQA" and
# "2WikiMultihopQA" were separate nodes. A hyphen creates a token boundary that
# plain normalisation preserves, so the two spellings of one dataset never met.


@pytest.mark.parametrize(
    "a,b",
    [
        ("2Wiki-MultiHopQA", "2WikiMultihopQA"),
        ("Multi-Hop QA", "MultiHop QA"),
        ("HotPot-QA", "HotPotQA"),
    ],
)
def test_hyphenation_variants_merge(a, b):
    mapping = resolve_entities({a: "Dataset", b: "Dataset"}, log_path=None)
    assert mapping[a] == mapping[b]


def test_despacing_does_not_survive_a_different_stem_split():
    """A known and accepted limitation.

    Stemming runs per token, so a hyphen that splits a word changes what gets
    stemmed: "Retrieval-Augmented" -> "retriev augmented" (the "al" comes off),
    while "RetrievalAugmented" stays whole. De-spacing afterwards cannot undo
    that. Nobody writes the run-together form, so the fix would cost real
    complexity for a spelling that does not occur -- documented rather than
    engineered around.
    """
    mapping = resolve_entities(
        {
            "Retrieval-Augmented Generation": "Method",
            "RetrievalAugmented Generation": "Method",
        },
        log_path=None,
    )
    assert mapping["Retrieval-Augmented Generation"] != mapping["RetrievalAugmented Generation"]


@pytest.mark.parametrize(
    "a,b,etype",
    [
        ("Dense Passage Retrieval", "Dense Hierarchical Retrieval", "Method"),
        ("SQuAD v1.1", "SQuAD 2.0", "Dataset"),
        ("Yang et al. (2018)", "Yang et al. (2019)", "Paper"),
        ("BERT large baseline", "BERT base baseline", "Method"),
    ],
)
def test_despacing_does_not_loosen_the_over_merge_guards(a, b, etype):
    """De-spacing must catch hyphen variants without reopening the bugs the
    digit and token-overlap guards were added to close."""
    mapping = resolve_entities({a: etype, b: etype}, log_path=None)
    assert mapping[a] != mapping[b]
