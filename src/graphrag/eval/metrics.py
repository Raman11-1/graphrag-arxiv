"""Retrieval metrics.

Pure arithmetic against a gold set -- no LLM, no API calls, no cost, and fully
deterministic. These are the metrics that most directly test the project's
central claim, because they measure whether the right evidence was *found*,
independently of how well it was later written up.

All three answer different questions:

* **recall@k**  -- did we find the relevant items at all?
* **MRR**       -- how near the top was the first relevant one?
* **nDCG@k**    -- were the relevant ones ranked above the irrelevant ones,
                   discounted by position?

A system can have high recall and poor nDCG (finds everything, ranks it badly),
which is exactly the failure a single number would hide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items appearing in the top k."""
    if not relevant:
        return 0.0
    hits = len(set(retrieved[:k]) & relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top k that are relevant."""
    if k <= 0 or not retrieved:
        return 0.0
    top = retrieved[:k]
    return len(set(top) & relevant) / len(top)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1/rank of the first relevant item, or 0 if none was retrieved."""
    for i, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance.

    Positions are discounted by log2(rank+1), so a relevant item at rank 1 is
    worth much more than the same item at rank 8.
    """
    if not relevant:
        return 0.0

    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, item in enumerate(retrieved[:k], start=1)
        if item in relevant
    )
    # Ideal ranking: every relevant item packed into the top positions.
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


@dataclass
class RetrievalScores:
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    k: int

    def as_dict(self) -> dict[str, float]:
        return {
            f"recall@{self.k}": round(self.recall_at_k, 4),
            f"precision@{self.k}": round(self.precision_at_k, 4),
            "mrr": round(self.mrr, 4),
            f"ndcg@{self.k}": round(self.ndcg_at_k, 4),
        }


def score_retrieval(retrieved: list[str], relevant: set[str], k: int = 8) -> RetrievalScores:
    """All retrieval metrics for one question."""
    return RetrievalScores(
        recall_at_k=recall_at_k(retrieved, relevant, k),
        precision_at_k=precision_at_k(retrieved, relevant, k),
        mrr=reciprocal_rank(retrieved, relevant),
        ndcg_at_k=ndcg_at_k(retrieved, relevant, k),
        k=k,
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(scores: list[RetrievalScores]) -> dict[str, float]:
    """Average each metric across questions."""
    if not scores:
        return {}
    k = scores[0].k
    return {
        f"recall@{k}": round(mean([s.recall_at_k for s in scores]), 4),
        f"precision@{k}": round(mean([s.precision_at_k for s in scores]), 4),
        "mrr": round(mean([s.mrr for s in scores]), 4),
        f"ndcg@{k}": round(mean([s.ndcg_at_k for s in scores]), 4),
    }
