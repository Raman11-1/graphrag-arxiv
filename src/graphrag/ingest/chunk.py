"""Dual chunking: small chunks for retrieval, wide windows for extraction.

The same parsed text is split twice, because the two consumers want opposite
things:

* **Retrieval** wants small spans. A precise chunk keeps irrelevant text out of
  the answer context and makes vector similarity discriminative.
* **Extraction** wants wide spans. A relation like "we evaluate RETRO on the
  Pile, reaching 3.2 perplexity" is routinely split across a paragraph break;
  an 800-token window would hand the model half of it and get a broken triple.

Both use the same packer, so offsets stay consistent and every span remains
resolvable back into the paper's full text for citations.
"""

from __future__ import annotations

import re

from graphrag.models import Chunk, ExtractionWindow, Section

# Sections that carry text worth indexing but almost never yield useful triples.
# Skipping them for extraction is the single biggest call-count saving available.
NON_EXTRACTIVE_SECTIONS = {
    "references",
    "bibliography",
    "acknowledgements",
    "acknowledgments",
    "appendix",
    "supplementary material",
    "author contributions",
    "funding",
    "conflicts of interest",
    "ethics statement",
}

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def estimate_tokens(text: str) -> int:
    """Approximate token count for English academic prose.

    A word-count heuristic, not a tokenizer. This drives chunk *sizing* only --
    anything cost- or quota-related uses the provider's reported usage, never
    this. Avoiding a real tokenizer here keeps chunking free of a model
    dependency and keeps the two backends comparable.
    """
    if not text:
        return 0
    words = len(text.split())
    return max(1, int(words * 1.3))


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of paragraphs, in order, excluding the separators."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in _PARAGRAPH_BREAK.finditer(text):
        if match.start() > cursor:
            spans.append((cursor, match.start()))
        cursor = match.end()
    if cursor < len(text):
        spans.append((cursor, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def _split_oversized(text: str, start: int, end: int, target: int) -> list[tuple[int, int]]:
    """Break one over-long paragraph on sentence boundaries.

    Only invoked when a single paragraph exceeds the target on its own, which
    happens with tables and unbroken derivations.
    """
    body = text[start:end]
    if estimate_tokens(body) <= target:
        return [(start, end)]

    pieces: list[tuple[int, int]] = []
    cursor = start
    budget = 0
    for part in _SENTENCE_END.split(body):
        if not part:
            continue
        part_start = text.find(part, cursor, end)
        if part_start == -1:
            continue
        part_tokens = estimate_tokens(part)

        if budget and budget + part_tokens > target:
            pieces.append((cursor, part_start))
            cursor = part_start
            budget = 0
        budget += part_tokens

    if cursor < end:
        pieces.append((cursor, end))
    return pieces or [(start, end)]


def _pack(
    text: str,
    spans: list[tuple[int, int]],
    target_tokens: int,
    overlap_tokens: int,
) -> list[tuple[int, int]]:
    """Greedily pack paragraph spans into units of ~``target_tokens``.

    Overlap is applied by replaying trailing paragraphs into the next unit,
    so a fact sitting on a boundary appears whole in at least one unit.
    """
    if not spans:
        return []

    # Explode any paragraph that is too big to ever fit.
    atoms: list[tuple[int, int]] = []
    for s, e in spans:
        atoms.extend(_split_oversized(text, s, e, target_tokens))

    units: list[tuple[int, int]] = []
    current: list[tuple[int, int]] = []
    current_tokens = 0

    for span in atoms:
        span_tokens = estimate_tokens(text[span[0] : span[1]])

        if current and current_tokens + span_tokens > target_tokens:
            units.append((current[0][0], current[-1][1]))

            # Carry back trailing paragraphs until the overlap budget is met.
            carry: list[tuple[int, int]] = []
            carry_tokens = 0
            for prev in reversed(current):
                prev_tokens = estimate_tokens(text[prev[0] : prev[1]])
                if carry_tokens + prev_tokens > overlap_tokens:
                    break
                carry.insert(0, prev)
                carry_tokens += prev_tokens

            current = carry
            current_tokens = carry_tokens

        current.append(span)
        current_tokens += span_tokens

    if current:
        units.append((current[0][0], current[-1][1]))

    return units


def _section_at(sections: list[Section], pos: int) -> str:
    for sec in sections:
        if sec.char_start <= pos < sec.char_end:
            return sec.title
    return ""


def build_chunks(
    *,
    paper_id: str,
    full_text: str,
    sections: list[Section],
    target_tokens: int = 800,
    overlap_tokens: int = 100,
) -> list[Chunk]:
    """Small, overlapping retrieval units covering the entire paper."""
    spans = _pack(full_text, _paragraph_spans(full_text), target_tokens, overlap_tokens)
    chunks: list[Chunk] = []
    for start, end in spans:
        body = full_text[start:end].strip()
        if not body:
            continue
        chunks.append(
            Chunk(
                chunk_id=Chunk.make_id(paper_id, start, end),
                paper_id=paper_id,
                text=body,
                char_start=start,
                char_end=end,
                section=_section_at(sections, start),
                token_estimate=estimate_tokens(body),
            )
        )
    return chunks


def build_windows(
    *,
    paper_id: str,
    full_text: str,
    sections: list[Section],
    target_tokens: int = 4000,
    overlap_tokens: int = 200,
    skip_non_extractive: bool = True,
) -> list[ExtractionWindow]:
    """Wide extraction units, restricted to sections that yield real facts.

    With the defaults this produces roughly 3 windows for a typical paper
    instead of ~14 retrieval chunks -- fewer calls, and relations that span
    paragraphs survive intact.
    """
    if skip_non_extractive and sections:
        keep = [s for s in sections if s.title.strip().lower() not in NON_EXTRACTIVE_SECTIONS]
        candidate_spans: list[tuple[int, int]] = []
        for sec in keep:
            candidate_spans.extend(
                (s, e)
                for s, e in _paragraph_spans(full_text)
                if s >= sec.char_start and e <= sec.char_end
            )
        candidate_spans.sort()
    else:
        candidate_spans = _paragraph_spans(full_text)

    spans = _pack(full_text, candidate_spans, target_tokens, overlap_tokens)

    windows: list[ExtractionWindow] = []
    for start, end in spans:
        body = full_text[start:end].strip()
        if not body:
            continue
        covered = sorted(
            {
                sec.title
                for sec in sections
                if sec.char_start < end and sec.char_end > start and sec.title
            }
        )
        windows.append(
            ExtractionWindow(
                window_id=ExtractionWindow.make_id(paper_id, start, end),
                paper_id=paper_id,
                text=body,
                char_start=start,
                char_end=end,
                sections=covered,
                token_estimate=estimate_tokens(body),
            )
        )
    return windows
