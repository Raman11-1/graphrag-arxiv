"""Core data types shared across the pipeline.

These are the contract between stages. Anything that crosses a stage boundary
or gets written to disk is defined here, so the shape of the data is in one
place rather than implied by whatever the previous stage happened to return.
"""

from __future__ import annotations

import hashlib
from datetime import date

from pydantic import BaseModel, Field


class Paper(BaseModel):
    """One source document."""

    paper_id: str = Field(description="Stable ID, e.g. the arXiv ID '2005.11401v4'")
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    published: date | None = None
    categories: list[str] = Field(default_factory=list)
    pdf_path: str | None = None
    source_url: str | None = None


class Section(BaseModel):
    """A titled span of a paper's body text.

    Sections drive extraction targeting: 'References' and 'Acknowledgements'
    are text we index for retrieval but never spend extraction calls on.
    """

    title: str
    text: str
    char_start: int
    char_end: int


class Chunk(BaseModel):
    """A retrieval unit -- small, for precise vector and BM25 matching.

    ``char_start``/``char_end`` index into the paper's full cleaned text, which
    is what makes a citation resolvable back to the exact source span.
    """

    chunk_id: str
    paper_id: str
    text: str
    char_start: int
    char_end: int
    section: str = ""
    token_estimate: int = 0

    @staticmethod
    def make_id(paper_id: str, char_start: int, char_end: int) -> str:
        raw = f"{paper_id}:{char_start}:{char_end}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


class ExtractionWindow(BaseModel):
    """A knowledge-graph extraction unit -- large, for cross-paragraph relations.

    Deliberately *not* the same as a Chunk. Retrieval wants small spans so the
    answer context stays precise; extraction wants wide spans because relations
    like "we evaluate X on Y, reaching Z" routinely straddle paragraph breaks.
    One parse, two chunkings.
    """

    window_id: str
    paper_id: str
    text: str
    char_start: int
    char_end: int
    sections: list[str] = Field(default_factory=list)
    token_estimate: int = 0

    @staticmethod
    def make_id(paper_id: str, char_start: int, char_end: int) -> str:
        raw = f"win:{paper_id}:{char_start}:{char_end}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


class ParsedPaper(BaseModel):
    """Everything one ingestion pass produces for a single paper."""

    paper: Paper
    full_text: str
    sections: list[Section] = Field(default_factory=list)
    chunks: list[Chunk] = Field(default_factory=list)
    windows: list[ExtractionWindow] = Field(default_factory=list)
