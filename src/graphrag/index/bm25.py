"""BM25 lexical index.

Serves two purposes: it is a fusion partner for vector search (lexical matching
catches exact identifiers like "MS MARCO" or "BM25" that embeddings blur), and
it is a standalone baseline arm in the evaluation.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

from graphrag.config import settings
from graphrag.index.vector_store import Hit
from graphrag.logging import get_logger
from graphrag.models import Chunk

log = get_logger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens.

    Deliberately keeps digits: benchmark names and scores ("MS MARCO", "78.4")
    are exactly what lexical search should catch and embeddings tend to miss.
    """
    return _TOKEN.findall(text.lower())


class BM25Index:
    def __init__(self) -> None:
        self._bm25 = None
        self._chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        from rank_bm25 import BM25Okapi

        self._chunks = list(chunks)
        if not chunks:
            self._bm25 = None
            return
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
        log.info("bm25_built", chunks=len(chunks))

    def search(self, query: str, k: int | None = None) -> list[Hit]:
        k = k or settings.top_k
        if self._bm25 is None or not self._chunks:
            return []

        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda kv: kv[1], reverse=True)[:k]

        return [
            Hit(
                chunk_id=self._chunks[i].chunk_id,
                paper_id=self._chunks[i].paper_id,
                text=self._chunks[i].text,
                score=float(score),
                char_start=self._chunks[i].char_start,
                char_end=self._chunks[i].char_end,
                section=self._chunks[i].section,
                source="bm25",
            )
            for i, score in ranked
            if score > 0  # a zero BM25 score means no term overlap at all
        ]

    # -- persistence ----------------------------------------------------

    def save(self, path: Path | None = None) -> None:
        path = Path(path or settings.bm25_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump([c.model_dump() for c in self._chunks], fh)
        log.info("bm25_saved", path=str(path), chunks=len(self._chunks))

    def load(self, path: Path | None = None) -> bool:
        """Rebuild from saved chunks. Returns False if there is nothing saved.

        We persist the chunks and re-fit BM25 rather than pickling the model:
        the index is cheap to rebuild and a pickled third-party object is a
        version-fragile thing to keep on disk.
        """
        path = Path(path or settings.bm25_path)
        if not path.exists():
            return False
        with path.open("rb") as fh:
            raw = pickle.load(fh)
        self.build([Chunk.model_validate(r) for r in raw])
        return True

    def __len__(self) -> int:
        return len(self._chunks)
