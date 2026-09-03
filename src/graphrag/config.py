"""Central configuration, loaded from environment / .env.

Everything that varies between runs lives here so no module reads os.environ
directly. Import the singleton: ``from graphrag.config import settings``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root is three levels up from this file: src/graphrag/config.py -> repo/
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- backend ---------------------------------------------------------
    llm_backend: str = "mistral"  # "mistral" | "anthropic"

    # Left optional so offline work (parsing, chunking, BM25, embeddings) runs
    # without any key. The backend raises a clear error only when actually used.
    mistral_api_key: str | None = None
    anthropic_api_key: str | None = None

    # --- per-stage models ------------------------------------------------
    extraction_model: str = "mistral-medium-latest"
    router_model: str = "mistral-small-latest"
    cypher_model: str = "mistral-medium-latest"
    summary_model: str = "mistral-medium-latest"
    answer_model: str = "mistral-medium-latest"
    judge_model: str = "mistral-medium-latest"

    # --- throughput ------------------------------------------------------
    # Mistral's free tier does not publish its limits and they differ per
    # account. The binding constraint turned out to be tokens per minute, not
    # requests per second: a probe of small requests shows no throttling at
    # 1.6 req/s, while 17k-token extraction calls hit 429 immediately at the
    # same rate. 0.25 is what sustains extraction; raise it for light work.
    llm_rps: float = 0.25
    llm_max_attempts: int = 6

    # --- cost guard -------------------------------------------------------
    # Zero on the free tier, but the guard still protects the anthropic backend.
    max_extraction_spend_usd: float = 15.0

    # --- storage --------------------------------------------------------
    data_dir: Path = Path("data")
    storage_dir: Path = Path("storage")

    # --- retrieval tuning -----------------------------------------------
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    chunk_tokens: int = 800
    chunk_overlap_tokens: int = 100
    # Extraction reads far wider spans than retrieval, so relations that cross
    # paragraph breaks survive. See graphrag/ingest/chunk.py for why.
    extraction_window_tokens: int = 4000
    extraction_window_overlap_tokens: int = 200
    top_k: int = 8
    resolver_similarity_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    cypher_row_limit: int = 200

    log_level: str = "INFO"

    @field_validator("data_dir", "storage_dir", mode="after")
    @classmethod
    def _absolutize(cls, v: Path) -> Path:
        """Resolve relative paths against the repo root, not the cwd.

        Without this, running `streamlit run ...` from a different directory
        would silently create a second, empty set of stores.
        """
        return v if v.is_absolute() else (REPO_ROOT / v)

    # --- derived paths ---------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def kuzu_path(self) -> Path:
        return self.storage_dir / "kuzu"

    @property
    def chroma_path(self) -> Path:
        return self.storage_dir / "chroma"

    @property
    def bm25_path(self) -> Path:
        return self.storage_dir / "bm25.pkl"

    def ensure_dirs(self) -> None:
        """Create every directory the pipeline writes to. Idempotent."""
        for p in (self.raw_dir, self.processed_dir, self.storage_dir, self.chroma_path):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
