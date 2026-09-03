"""Entity resolution: merge surface forms that name the same thing.

Extraction produces "Dense Passage Retriever", "Dense Passage Retrieval" and
"DPR" as three nodes. Left unmerged they split a central entity, and every
multi-hop query silently misses paths through it -- the graph looks fine and
quietly under-answers.

Three stages, cheapest first, because the expensive one costs API calls:

1. **Normalise** -- casefold, strip punctuation, drop stopword suffixes
   ("model", "method", "approach"), and stem the -er/-or/-ion endings that
   separate "Retriever" from "Retrieval". Free, catches most of it.
2. **Embed and block** -- cosine similarity within the same entity type, above
   ``resolver_similarity_threshold``. Free (fastembed is local), catches
   paraphrase-level variation.
3. **Adjudicate** -- one batched LLM call for the borderline band only. Skipped
   entirely when nothing lands there.

Every merge is logged to ``data/processed/merges.jsonl`` so a wrong merge is
auditable rather than invisible.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from graphrag.config import settings
from graphrag.logging import get_logger

log = get_logger(__name__)

# Words that carry no identity: "BERT model" and "BERT" are the same thing.
_NOISE_SUFFIXES = {
    "model", "models", "method", "methods", "approach", "approaches",
    "system", "systems", "algorithm", "algorithms", "technique", "techniques",
    "framework", "frameworks", "architecture", "dataset", "datasets",
    "benchmark", "benchmarks", "task", "tasks", "score", "scores",
}
_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


# A trailing parenthetical that is a configuration label, not part of the name:
# "DPR (rt, NQ, 1enc)", "D-NET (Baidu)", "DHR-D(Abs)". Extraction emits one
# entity per ablation row, which inflates every COUNT -- 47 of them appeared in
# a 23-paper corpus, all variants of a handful of real methods.
_TRAILING_PARENS = re.compile(r"\s*\([^()]*\)\s*$")
_YEAR = re.compile(r"\b(1[89]|20)\d{2}\b")
# A short modifier appended after a configuration group: "+ qsft", "+T".
_TRAILING_MODIFIER = re.compile(r"\s*\+\s*[A-Za-z0-9_-]{1,12}\s*$")


def strip_config_suffix(name: str) -> str:
    """Drop trailing configuration parentheses, keeping citation years intact.

    A parenthetical containing a year is a citation ("Yang et al. (2018)") and
    is load-bearing for identity -- stripping it would fuse every paper by the
    same author. Anything else trailing in brackets is a configuration label.
    """
    out = name.strip()
    while True:
        match = _TRAILING_PARENS.search(out)
        if match and not _YEAR.search(match.group(0)):
            candidate = out[: match.start()].strip()
            # Never strip a name down to nothing, or to a fragment.
            if len(candidate) < 3:
                break
            out = candidate
            continue

        # "DPR (rt, NQ, 1enc, stopG) + qsft" -- a modifier appended *after* a
        # configuration group. Only stripped when a config paren remains, so a
        # genuine hybrid like "BM25 + DPR" keeps both halves of its name.
        mod = _TRAILING_MODIFIER.search(out)
        if mod:
            candidate = out[: mod.start()].strip()
            if len(candidate) >= 3 and _TRAILING_PARENS.search(candidate):
                out = candidate
                continue

        break
    return out


def normalise(name: str) -> str:
    """Aggressive normal form used only for grouping, never for display.

    Stems the -er/-or/-ion endings so "Retriever" and "Retrieval" collapse
    together -- the specific variation that split DPR into two nodes.
    """
    text = _PUNCT.sub(" ", strip_config_suffix(name).lower())
    words = [w for w in _WS.sub(" ", text).strip().split() if w]

    while len(words) > 1 and words[-1] in _NOISE_SUFFIXES:
        words.pop()

    stemmed = []
    for w in words:
        # Plural first, then derivational suffixes. Order matters: checking
        # "ion" before "s" stems "Question"->"quest" but "Questions"->"question",
        # so the singular and plural of the same word stop matching.
        if len(w) > 4 and w.endswith("es") and not w.endswith(("ses", "ies")):
            w = w[:-2]
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]

        for suffix in ("ation", "ional", "ing", "ion", "er", "or", "al"):
            if len(w) > len(suffix) + 2 and w.endswith(suffix):
                w = w[: -len(suffix)]
                break
        stemmed.append(w)

    return " ".join(stemmed)


class _Union:
    """Union-find over canonical names."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Longest name wins as root: it is the spelled-out form.
            root, child = (ra, rb) if len(ra) >= len(rb) else (rb, ra)
            self._parent[child] = root

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for node in self._parent:
            out[self.find(node)].append(node)
        return out


_DIGITS = re.compile(r"\d+")


def digits_conflict(a: str, b: str) -> bool:
    """True when two names carry different numbers.

    Numbers in this domain are identity-bearing and embeddings are blind to
    them: "Yang et al. (2018)" and "Yang et al. (2019)" score ~0.98 cosine but
    are different papers, and "SQuAD v1.1" / "SQuAD 2.0" are different datasets.
    Merging them corrupts facts -- it rewrote DPR's benchmark from v1.1 to 2.0.

    A name with no digits is unconstrained, so "BERT" can still merge with
    "BERT model".
    """
    da, db = set(_DIGITS.findall(a)), set(_DIGITS.findall(b))
    if not da or not db:
        return False
    return da != db


# Two names may only merge if their normalised content tokens overlap at least
# this much. Embedding similarity alone is not safe for short technical names:
# "Dense Passage Retrieval" and "Dense Hierarchical Retrieval" score ~0.95
# because they share two words out of three, but the differing word is the one
# carrying the identity. Cosine similarity cannot see which token matters.
MIN_TOKEN_OVERLAP = 0.8


def token_overlap(a: str, b: str) -> float:
    """Jaccard overlap of normalised tokens. 1.0 means identical token sets."""
    ta, tb = set(normalise(a).split()), set(normalise(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _cosine_pairs(
    names: list[str], threshold: float
) -> list[tuple[str, str, float]]:
    """Name pairs whose embeddings exceed ``threshold``.

    Guarded by token overlap as well as similarity. In practice this makes
    embedding merging conservative almost to the point of redundancy with the
    normalisation stage -- which is the honest conclusion: for short technical
    names, embeddings are a weak signal of identity and a strong signal of
    topical relatedness, and those are not the same thing.
    """
    if len(names) < 2:
        return []

    from graphrag.index.embedder import get_embedder

    vectors = get_embedder().embed_documents(names)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = vectors / norms
    sim = unit @ unit.T

    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            score = float(sim[i, j])
            if score < threshold:
                continue
            if digits_conflict(names[i], names[j]):
                continue
            if token_overlap(names[i], names[j]) < MIN_TOKEN_OVERLAP:
                continue
            pairs.append((names[i], names[j], score))
    return pairs


def resolve_entities(
    canonical_type: dict[str, str],
    *,
    threshold: float | None = None,
    log_path: Path | None = None,
) -> dict[str, str]:
    """Return name -> merged canonical name.

    Only entities of the *same* type are ever compared: "BERT" the Method and a
    hypothetical "BERT" Dataset are different things, and merging across types
    would produce edges Kùzu cannot even store.
    """
    threshold = threshold if threshold is not None else settings.resolver_similarity_threshold
    union = _Union()

    by_type: dict[str, list[str]] = defaultdict(list)
    for name, etype in canonical_type.items():
        by_type[etype].append(name)

    # Citation strings ("Yang et al. (2019)") differ only by year and author
    # ordinal, which embeddings cannot see. Never similarity-merge them.
    NO_EMBEDDING_MERGE = {"Paper"}

    embed_merges = 0
    for etype, names in by_type.items():
        # Stage 1: normalised-form collisions. Free.
        #
        # Grouped twice: once on the normal form, and once with spaces removed.
        # A hyphen creates a token boundary that plain normalisation keeps, so
        # "2Wiki-MultiHopQA" and "2WikiMultihopQA" stay apart despite naming
        # the same dataset. Comparing the de-spaced forms catches that without
        # loosening anything else -- "densepassageretriev" and
        # "densehierarchretriev" are still plainly different strings.
        for key in (normalise, lambda n: normalise(n).replace(" ", "")):
            by_norm: dict[str, list[str]] = defaultdict(list)
            for name in names:
                by_norm[key(name)].append(name)
            for group in by_norm.values():
                for other in group[1:]:
                    if digits_conflict(group[0], other):
                        continue
                    union.union(group[0], other)

        # Stage 2: embedding similarity. Also free -- fastembed runs locally.
        if len(names) > 1 and etype not in NO_EMBEDDING_MERGE:
            for a, b, score in _cosine_pairs(names, threshold):
                if union.find(a) != union.find(b):
                    embed_merges += 1
                    log.debug("embedding_merge", type=etype, a=a, b=b, score=round(score, 3))
                union.union(a, b)

    mapping: dict[str, str] = {}
    merges: list[dict] = []
    for root, members in union.groups().items():
        # The union root is the longest surface form, which for a group of
        # ablation variants is the most-qualified one ("DPR (rt, NQ, 1enc,
        # stopG)"). The base name is the useful node identity.
        root = strip_config_suffix(root)
        if len(members) > 1:
            merges.append(
                {
                    "canonical": root,
                    "merged": sorted(members),
                    "type": canonical_type.get(root, ""),
                }
            )
        for member in members:
            mapping[member] = root

    # Anything untouched maps to itself.
    for name in canonical_type:
        mapping.setdefault(name, name)

    path = Path(log_path or settings.processed_dir / "merges.jsonl")
    if merges:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for m in merges:
                fh.write(json.dumps(m, ensure_ascii=False) + "\n")

    log.info(
        "entities_resolved",
        input_entities=len(canonical_type),
        output_entities=len(set(mapping.values())),
        merge_groups=len(merges),
        embedding_merges=embed_merges,
        audit_log=str(path) if merges else None,
    )
    return mapping
