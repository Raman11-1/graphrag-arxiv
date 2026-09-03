"""PDF -> clean text + sections.

Academic PDFs are hostile to naive text extraction: words are hyphenated across
line breaks, every page repeats a running header, and column layout interleaves
lines if you read raw. This module produces one clean string per paper plus the
section boundaries, and every downstream character offset indexes into that
string -- so a citation can always be resolved back to real source text.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from graphrag.logging import get_logger
from graphrag.models import Section

log = get_logger(__name__)

# "1 Introduction", "2. Related Work", "III. METHOD", "A.1 Proofs"
_NUMBERED_HEADING = re.compile(
    r"^\s*((?:\d+|[IVXLC]+|[A-Z])(?:\.\d+)*)\.?\s+([A-Z][A-Za-z0-9 \-:,'&/]{2,60})\s*$"
)
# Unnumbered but conventional headings.
_BARE_HEADINGS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "method",
    "methods",
    "methodology",
    "approach",
    "model",
    "experiments",
    "experimental setup",
    "results",
    "evaluation",
    "analysis",
    "discussion",
    "ablation study",
    "limitations",
    "conclusion",
    "conclusions",
    "future work",
    "references",
    "bibliography",
    "acknowledgements",
    "acknowledgments",
    "appendix",
    "broader impact",
    "ethics statement",
}

# Hyphen at end of line followed by a lowercase continuation -> rejoin the word.
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
# A single newline inside a sentence is a layout artefact, not a paragraph break.
_SOFT_WRAP = re.compile(r"(?<![\n.\?!:;])\n(?!\n)")
_MULTI_BLANK = re.compile(r"\n{3,}")
_LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "\xa0": " "}


def _strip_repeated_lines(pages: list[str], threshold: float = 0.6) -> list[str]:
    """Drop running headers/footers.

    A line appearing near-identically on most pages is furniture, not content.
    Papers shorter than 4 pages are left alone -- with few pages, a legitimately
    repeated line is more likely than a header.
    """
    if len(pages) < 4:
        return pages

    counts: Counter[str] = Counter()
    for page in pages:
        for line in {ln.strip() for ln in page.splitlines() if ln.strip()}:
            counts[line] += 1

    cutoff = max(2, int(len(pages) * threshold))
    furniture = {
        line
        for line, n in counts.items()
        if n >= cutoff and len(line) < 120 and not line.endswith(".")
    }
    if furniture:
        log.debug("dropping_repeated_lines", count=len(furniture))

    return [
        "\n".join(ln for ln in page.splitlines() if ln.strip() not in furniture) for page in pages
    ]


def clean_text(raw: str) -> str:
    """Normalise ligatures, rejoin hyphenated words, unwrap soft line breaks."""
    for bad, good in _LIGATURES.items():
        raw = raw.replace(bad, good)
    raw = _HYPHEN_BREAK.sub(r"\1\2", raw)
    raw = _SOFT_WRAP.sub(" ", raw)
    raw = _MULTI_BLANK.sub("\n\n", raw)
    # Collapse runs of spaces without touching newlines.
    raw = re.sub(r"[ \t]{2,}", " ", raw)
    return raw.strip()


def _is_heading(line: str) -> str | None:
    """Return the normalised heading title, or None if the line isn't a heading."""
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return None

    if stripped.lower().rstrip(".:") in _BARE_HEADINGS:
        return stripped.rstrip(".:").title()

    match = _NUMBERED_HEADING.match(stripped)
    if match:
        return match.group(2).strip().rstrip(".:")

    return None


def detect_sections(text: str) -> list[Section]:
    """Find section spans over ``text``.

    Falls back to a single 'Body' section when a paper's headings don't survive
    extraction -- downstream code must never assume sections were found.
    """
    marks: list[tuple[int, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        title = _is_heading(line)
        if title:
            marks.append((offset, title))
        offset += len(line)

    if not marks:
        return [Section(title="Body", text=text, char_start=0, char_end=len(text))]

    sections: list[Section] = []
    if marks[0][0] > 0:
        sections.append(
            Section(
                title="Front Matter",
                text=text[: marks[0][0]],
                char_start=0,
                char_end=marks[0][0],
            )
        )

    for i, (start, title) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        sections.append(
            Section(title=title, text=text[start:end], char_start=start, char_end=end)
        )

    return sections


# A block covering more than this fraction of the page width spans both
# columns (title, abstract, wide tables) rather than belonging to one.
_FULL_WIDTH_RATIO = 0.6


def _page_text_in_reading_order(page) -> str:
    """Extract one page, respecting two-column layout.

    PyMuPDF's ``get_text("text", sort=True)`` sorts by y-then-x, which on a
    two-column paper reads straight *across* both columns and interleaves them
    mid-sentence -- producing text like "stronger empirical per|to various QA
    tasks". That corrupts retrieval and extraction alike, so we order blocks
    ourselves: full-width header first, then the left column top-to-bottom,
    then the right column.
    """
    blocks = [b for b in page.get_text("blocks") if len(b) > 6 and b[6] == 0]
    if not blocks:
        return ""

    width = page.rect.width or 1.0
    midline = page.rect.x0 + width / 2

    full, left, right = [], [], []
    for b in blocks:
        x0, y0, x1, _y1, text = b[0], b[1], b[2], b[3], b[4]
        if (x1 - x0) / width > _FULL_WIDTH_RATIO:
            full.append((y0, text))
        elif (x0 + x1) / 2 < midline:
            left.append((y0, text))
        else:
            right.append((y0, text))

    # Single-column page: no column split is meaningful, keep natural order.
    if not left or not right:
        return "\n".join(text for _, text in sorted(full + left + right))

    # Full-width blocks above the columns are the header; any below (wide
    # tables, footnotes) are appended rather than cut into a column.
    first_column_y = min(y for y, _ in left + right)
    header = [(y, t) for y, t in full if y < first_column_y]
    trailer = [(y, t) for y, t in full if y >= first_column_y]

    ordered = sorted(header) + sorted(left) + sorted(right) + sorted(trailer)
    return "\n".join(text for _, text in ordered)


def parse_pdf(path: str | Path) -> tuple[str, list[Section]]:
    """Extract cleaned full text and sections from a PDF."""
    # Imported here, not at module scope: clean_text() and detect_sections()
    # are pure string functions and must stay usable without PyMuPDF installed.
    # Note: `pymupdf`, not the legacy `fitz` alias, which is deprecated.
    import pymupdf

    path = Path(path)
    with pymupdf.open(path) as doc:
        pages = [_page_text_in_reading_order(page) for page in doc]

    pages = _strip_repeated_lines(pages)
    full_text = clean_text("\n\n".join(pages))
    sections = detect_sections(full_text)

    log.info(
        "parsed_pdf",
        path=path.name,
        pages=len(pages),
        chars=len(full_text),
        sections=len(sections),
    )
    return full_text, sections
