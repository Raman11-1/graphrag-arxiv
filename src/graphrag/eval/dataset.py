"""The gold question set.

Questions are grouped by the retrieval capability they test, because a single
average would hide the entire finding. The claim is *not* "GraphRAG is better";
it is "GraphRAG wins specifically on relational, aggregate and corpus-wide
questions, and ties on local ones". Only a per-category breakdown can show that.

Honesty note for the report: this set is authored by the system's own builder,
which is a real source of bias. Two mitigations are built in -- questions are
written against the corpus rather than against the implementation, and every
`relevant_papers` entry is verifiable by reading the cited paper. Neither
removes the bias, and the report should say so.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path


class Category(StrEnum):
    """The four question shapes the system claims to handle."""

    LOCAL = "local"            # answerable from one or two passages
    MULTI_HOP = "multi_hop"    # requires connecting entities
    AGGREGATE = "aggregate"    # counting, ranking, "how many"
    GLOBAL = "global"          # about the corpus as a whole


@dataclass
class GoldQuestion:
    id: str
    question: str
    category: Category
    reference: str
    # Papers whose text should be retrieved to answer this. Used for the
    # deterministic retrieval metrics, so it must be checkable by hand.
    relevant_papers: list[str] = field(default_factory=list)
    notes: str = ""


def default_questions() -> list[GoldQuestion]:
    """A starter set covering all four categories.

    Kept deliberately small and specific. A larger set of vague questions is
    worse than a smaller set of checkable ones, because vague questions cannot
    be graded reliably by any judge.
    """
    return [
        # --- LOCAL: plain RAG should do well here -----------------------
        GoldQuestion(
            id="L1",
            question=(
                "What is dense passage retrieval and how does it encode "
                "questions and passages?"
            ),
            category=Category.LOCAL,
            reference=(
                "Dense Passage Retrieval (DPR) encodes questions and passages into "
                "dense vectors using two separate BERT-based encoders (a dual-encoder). "
                "Relevance is the dot product of the question and passage vectors, and "
                "retrieval is done by maximum inner product search over a FAISS index."
            ),
            relevant_papers=["2004.04906v3"],
        ),
        GoldQuestion(
            id="L2",
            question="Why are in-batch negatives used when training dense retrievers?",
            category=Category.LOCAL,
            reference=(
                "In-batch negatives reuse the other passages in a training batch as "
                "negative examples for each question, which makes training far more "
                "efficient by avoiding separate negative sampling while still giving "
                "many negatives per question."
            ),
            relevant_papers=["2004.04906v3"],
        ),
        GoldQuestion(
            id="L3",
            question="What problem does retrieval-augmented generation address?",
            category=Category.LOCAL,
            reference=(
                "Retrieval-augmented generation addresses the limitation that a language "
                "model's knowledge is fixed in its parameters: it cannot easily be updated, "
                "its provenance cannot be inspected, and it hallucinates. Retrieving "
                "external passages at generation time grounds the output in sources."
            ),
            relevant_papers=[],
        ),
        # --- MULTI_HOP: needs connected facts ---------------------------
        GoldQuestion(
            id="M1",
            question="Which datasets does DPR evaluate on?",
            category=Category.MULTI_HOP,
            reference=(
                "DPR is evaluated on Natural Questions, TriviaQA, WebQuestions, "
                "CuratedTREC and SQuAD v1.1."
            ),
            relevant_papers=["2004.04906v3"],
            notes="Tests alias resolution: the graph must connect 'DPR' to the full name.",
        ),
        GoldQuestion(
            id="M2",
            question="Who are the authors of the dense passage retrieval paper?",
            category=Category.MULTI_HOP,
            reference=(
                "Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, "
                "Sergey Edunov, Danqi Chen and Wen-tau Yih."
            ),
            relevant_papers=["2004.04906v3"],
            notes="Author lists are metadata, poorly served by passage retrieval.",
        ),
        GoldQuestion(
            id="M3",
            question="Which methods does DPR compare itself against?",
            category=Category.MULTI_HOP,
            reference=(
                "DPR compares against BM25 as the sparse retrieval baseline, against a "
                "BM25+DPR hybrid, and against ORQA and REALM as prior dense/pretrained "
                "retrieval systems."
            ),
            relevant_papers=["2004.04906v3"],
        ),
        # --- AGGREGATE: counting, impossible for top-k retrieval --------
        GoldQuestion(
            id="A1",
            question="How many distinct methods in this corpus are evaluated on Natural Questions?",
            category=Category.AGGREGATE,
            reference=(
                "A specific whole number, stated plainly, derived from counting the "
                "distinct methods linked to Natural Questions across the corpus. "
                "A good answer commits to a figure and says what it counted. "
                "A poor answer hedges, lists a few examples without totalling them, "
                "or says the sources do not permit a count."
            ),
            relevant_papers=[],
            notes=(
                "The true value depends on the corpus and cannot be fixed in advance, "
                "and using the system's own graph as ground truth would be circular. "
                "So this question grades the capability -- does the system compute a "
                "count, or does it hedge -- rather than a specific value. Stated in "
                "the reference so the judge grades that and nothing else."
            ),
        ),
        GoldQuestion(
            id="A2",
            question="Which benchmark dataset is used by the most methods in this corpus?",
            category=Category.AGGREGATE,
            reference=(
                "Natural Questions is the most widely used benchmark across the corpus, "
                "used by more distinct methods than any other dataset."
            ),
            relevant_papers=[],
        ),
        GoldQuestion(
            id="A3",
            question="What is the highest Top-20 retrieval accuracy reported on Natural Questions?",
            category=Category.AGGREGATE,
            reference=(
                "DPR reports 78.4 Top-20 retrieval accuracy on Natural Questions, "
                "compared with 59.1 for BM25."
            ),
            relevant_papers=["2004.04906v3"],
            notes="Requires comparing numeric values across extracted results.",
        ),
        # --- GLOBAL: about the corpus, not any part of it ---------------
        GoldQuestion(
            id="G1",
            question="What are the main research themes across these papers?",
            category=Category.GLOBAL,
            reference=(
                "The corpus centres on retrieval-augmented question answering: dense "
                "retrieval methods and their training, benchmark datasets for open-domain "
                "QA, and the integration of retrieval with language model generation."
            ),
            relevant_papers=[],
        ),
        GoldQuestion(
            id="G2",
            question="What benchmark datasets are used across this collection of papers?",
            category=Category.GLOBAL,
            reference=(
                "The corpus uses open-domain QA benchmarks including Natural Questions, "
                "TriviaQA, WebQuestions, CuratedTREC and SQuAD."
            ),
            relevant_papers=[],
        ),
        GoldQuestion(
            id="G3",
            question="How do the retrieval approaches in this corpus relate to each other?",
            category=Category.GLOBAL,
            reference=(
                "The corpus contains sparse lexical retrieval (BM25) as the traditional "
                "baseline, dense neural retrieval that outperforms it, and hybrid "
                "combinations; later work builds retrieval into generation pipelines."
            ),
            relevant_papers=[],
        ),
    ]


def dataset_path() -> Path:
    return Path(__file__).parent / "datasets" / "gold.jsonl"


def save(questions: list[GoldQuestion], path: Path | None = None) -> Path:
    path = Path(path or dataset_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(json.dumps(asdict(q), ensure_ascii=False) + "\n")
    return path


def load(path: Path | None = None) -> list[GoldQuestion]:
    """Load the gold set, writing the default one out on first use."""
    path = Path(path or dataset_path())
    if not path.exists():
        save(default_questions(), path)

    questions = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                raw = json.loads(line)
                raw["category"] = Category(raw["category"])
                questions.append(GoldQuestion(**raw))
    return questions


def build_chunk_index() -> dict[str, set[str]]:
    """paper_id -> its chunk ids, read from disk once.

    The benchmark asks this for every question, and re-reading every parsed
    paper each time turns a cheap lookup into O(questions x papers) disk reads.
    """
    from graphrag.ingest.pipeline import load_all_parsed

    return {
        parsed.paper.paper_id: {c.chunk_id for c in parsed.chunks}
        for parsed in load_all_parsed()
    }


def chunk_ids_for_papers(
    paper_ids: list[str], index: dict[str, set[str]] | None = None
) -> set[str]:
    """Every chunk belonging to the given papers.

    Relevance is defined at paper level in the gold set because judging which
    individual chunk answers a question is subjective; which paper does is not.

    Pass ``index`` from :func:`build_chunk_index` to avoid re-reading the corpus.
    """
    index = build_chunk_index() if index is None else index
    return {cid for pid in paper_ids for cid in index.get(pid, ())}


__all__ = [
    "Category",
    "GoldQuestion",
    "build_chunk_index",
    "chunk_ids_for_papers",
    "dataset_path",
    "default_questions",
    "load",
    "save",

]
