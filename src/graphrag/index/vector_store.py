"""Chroma-backed vector store over retrieval chunks.

We compute embeddings ourselves via fastembed and hand them to Chroma, rather
than letting Chroma pick an embedding function. That keeps one model in one
place -- indexing and querying provably use the same vectors, and swapping the
model is a single config change.
"""

from __future__ import annotations

from dataclasses import dataclass

from graphrag.config import settings
from graphrag.index.embedder import get_embedder
from graphrag.logging import get_logger
from graphrag.models import Chunk

log = get_logger(__name__)

COLLECTION = "chunks"
_BATCH = 256  # Chroma degrades on very large single adds.


@dataclass
class Hit:
    """One retrieved chunk with its score and provenance."""

    chunk_id: str
    paper_id: str
    text: str
    score: float
    char_start: int
    char_end: int
    section: str = ""
    source: str = "vector"


class VectorStore:
    def __init__(self, path: str | None = None) -> None:
        import chromadb

        self._path = str(path or settings.chroma_path)
        self._client = chromadb.PersistentClient(path=self._path)
        # Cosine, because BGE embeddings are trained for cosine similarity.
        # Chroma's default is L2, which would quietly rank slightly wrong.
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def delete_paper(self, paper_id: str) -> None:
        """Remove every vector belonging to one paper.

        Required for correct re-ingestion: re-parsing a PDF shifts character
        offsets, which changes chunk IDs, so a plain upsert leaves the previous
        run's vectors orphaned in the index -- still retrievable, still carrying
        whatever text the old parser produced.
        """
        self._collection.delete(where={"paper_id": paper_id})

    def add(self, chunks: list[Chunk]) -> int:
        """Index chunks, replacing any existing vectors for the same papers."""
        if not chunks:
            return 0

        for paper_id in {c.paper_id for c in chunks}:
            self.delete_paper(paper_id)

        embedder = get_embedder()
        added = 0
        for start in range(0, len(chunks), _BATCH):
            batch = chunks[start : start + _BATCH]
            vectors = embedder.embed_documents([c.text for c in batch])
            self._collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=vectors.tolist(),
                documents=[c.text for c in batch],
                metadatas=[
                    {
                        "paper_id": c.paper_id,
                        "char_start": c.char_start,
                        "char_end": c.char_end,
                        "section": c.section,
                    }
                    for c in batch
                ],
            )
            added += len(batch)
            log.debug("vector_batch_indexed", count=added, total=len(chunks))

        log.info("vector_index_updated", added=added, collection_size=self.count())
        return added

    def search(self, query: str, k: int | None = None) -> list[Hit]:
        k = k or settings.top_k
        if self.count() == 0:
            return []

        vector = get_embedder().embed_query(query)
        result = self._collection.query(
            query_embeddings=[vector.tolist()],
            n_results=min(k, self.count()),
        )

        hits: list[Hit] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        for cid, doc, meta, dist in zip(ids, docs, metas, dists, strict=False):
            meta = meta or {}
            hits.append(
                Hit(
                    chunk_id=cid,
                    paper_id=str(meta.get("paper_id", "")),
                    text=doc or "",
                    # Chroma returns cosine *distance*; similarity reads better
                    # in a report and keeps every scorer "higher is better".
                    score=1.0 - float(dist),
                    char_start=int(meta.get("char_start", 0)),
                    char_end=int(meta.get("char_end", 0)),
                    section=str(meta.get("section", "")),
                    source="vector",
                )
            )
        return hits

    def count(self) -> int:
        return int(self._collection.count())

    def reset(self) -> None:
        """Drop and recreate the collection. Used by tests and re-ingestion."""
        self._client.delete_collection(COLLECTION)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION, metadata={"hnsw:space": "cosine"}
        )
