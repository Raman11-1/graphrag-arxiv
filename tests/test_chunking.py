"""Chunking invariants.

The load-bearing property is that every span resolves back into the source
text. If offsets drift, citations silently point at the wrong sentence -- which
looks like a working system while being wrong, the worst failure mode here.
"""

from __future__ import annotations

import pytest

from graphrag.ingest.chunk import build_chunks, build_windows, estimate_tokens
from graphrag.models import Section

PARA = (
    "Dense retrieval maps queries and passages into a shared vector space. "
    "This allows semantic matching beyond lexical overlap. "
)


def make_text(n_paragraphs: int) -> str:
    return "\n\n".join(f"Paragraph {i}. {PARA}" for i in range(n_paragraphs))


@pytest.fixture
def sample() -> tuple[str, list[Section]]:
    text = (
        "Introduction\n\n"
        + make_text(6)
        + "\n\nExperiments\n\n"
        + make_text(6)
        + "\n\nReferences\n\n"
        + make_text(4)
    )
    intro_end = text.index("Experiments")
    exp_end = text.index("References")
    sections = [
        Section(title="Introduction", text=text[:intro_end], char_start=0, char_end=intro_end),
        Section(
            title="Experiments",
            text=text[intro_end:exp_end],
            char_start=intro_end,
            char_end=exp_end,
        ),
        Section(
            title="References", text=text[exp_end:], char_start=exp_end, char_end=len(text)
        ),
    ]
    return text, sections


def test_chunk_offsets_resolve_to_source(sample):
    text, sections = sample
    chunks = build_chunks(paper_id="p1", full_text=text, sections=sections, target_tokens=60)

    assert chunks
    for c in chunks:
        assert 0 <= c.char_start < c.char_end <= len(text)
        # The stored text must be findable at the recorded offsets.
        assert c.text in text[c.char_start : c.char_end]


def test_chunks_cover_the_document(sample):
    text, sections = sample
    chunks = build_chunks(paper_id="p1", full_text=text, sections=sections, target_tokens=60)

    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(text)
    # Consecutive chunks must not leave a gap (overlap is fine).
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.char_start <= prev.char_end


def test_chunk_ids_are_stable_and_unique(sample):
    text, sections = sample
    first = build_chunks(paper_id="p1", full_text=text, sections=sections, target_tokens=60)
    second = build_chunks(paper_id="p1", full_text=text, sections=sections, target_tokens=60)

    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first)


def test_windows_are_fewer_and_wider_than_chunks(sample):
    """The whole point of the two-chunking design."""
    text, sections = sample
    chunks = build_chunks(paper_id="p1", full_text=text, sections=sections, target_tokens=60)
    windows = build_windows(paper_id="p1", full_text=text, sections=sections, target_tokens=400)

    assert len(windows) < len(chunks)
    assert max(w.token_estimate for w in windows) > max(c.token_estimate for c in chunks)


def test_windows_skip_references(sample):
    """References are indexed for retrieval but must never cost an extraction call."""
    text, sections = sample
    windows = build_windows(paper_id="p1", full_text=text, sections=sections, target_tokens=400)

    ref_start = sections[-1].char_start
    assert all(w.char_start < ref_start for w in windows)
    assert not any("References" in w.sections for w in windows)


def test_oversized_paragraph_is_split():
    """A single paragraph larger than the target must not produce one giant chunk."""
    giant = " ".join(f"Sentence number {i} about retrieval systems." for i in range(300))
    chunks = build_chunks(paper_id="p1", full_text=giant, sections=[], target_tokens=100)

    assert len(chunks) > 1
    assert all(c.token_estimate < 400 for c in chunks)


def test_empty_text_yields_nothing():
    assert build_chunks(paper_id="p1", full_text="", sections=[]) == []
    assert build_windows(paper_id="p1", full_text="", sections=[]) == []


def test_estimate_tokens_is_monotonic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("one") < estimate_tokens("one two three")
