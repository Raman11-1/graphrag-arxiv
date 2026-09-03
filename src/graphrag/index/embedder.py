"""Local embeddings via fastembed (ONNX).

Chosen over sentence-transformers deliberately: fastembed runs on onnxruntime
and pulls no PyTorch, which is a ~50 MB install instead of ~2.5 GB. Embedding
happens entirely on your machine, so it is free, offline, and unaffected by any
API rate limit.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from graphrag.config import settings
from graphrag.logging import get_logger

log = get_logger(__name__)

# BGE retrieval models are trained asymmetrically: queries get an instruction
# prefix, documents do not. Omitting this measurably degrades recall, and it is
# a silent failure -- results merely get slightly worse.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    """Wraps a fastembed model with the document/query asymmetry handled."""

    def __init__(self, model_name: str | None = None) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name or settings.embed_model
        log.info("loading_embedding_model", model=self.model_name)
        self._model = TextEmbedding(model_name=self.model_name)
        self._uses_prefix = "bge" in self.model_name.lower()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed passages for indexing. Returns (n, dim) float32."""
        if not texts:
            return np.zeros((0, settings.embed_dim), dtype=np.float32)
        vectors = list(self._model.embed(texts))
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a search query. Returns (dim,) float32."""
        payload = f"{_BGE_QUERY_PREFIX}{text}" if self._uses_prefix else text
        vector = next(iter(self._model.query_embed([payload])))
        return np.asarray(vector, dtype=np.float32)

    @property
    def dim(self) -> int:
        return int(self.embed_documents(["probe"]).shape[1])


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Process-wide embedder. Loading the ONNX model is slow; do it once."""
    return Embedder()
