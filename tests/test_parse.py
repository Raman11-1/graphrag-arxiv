"""PDF parsing: column ordering, cleaning, section detection.

The column test matters most. PyMuPDF's built-in sort reads straight across a
two-column page and interleaves the columns mid-sentence, which silently
corrupts every downstream stage. A regression here would be invisible in the
logs and visible only as vaguely bad answers.
"""

from __future__ import annotations

from graphrag.ingest.parse import (
    _page_text_in_reading_order,
    clean_text,
    detect_sections,
)


class FakeRect:
    def __init__(self, width: float) -> None:
        self.width = width
        self.x0 = 0.0


class FakePage:
    """Duck-types just enough of a PyMuPDF page for the ordering logic."""

    def __init__(self, blocks, width: float = 600.0) -> None:
        self._blocks = blocks
        self.rect = FakeRect(width)

    def get_text(self, kind):
        assert kind == "blocks"
        return self._blocks


def block(x0, y0, x1, y1, text):
    # PyMuPDF block tuple: (x0, y0, x1, y1, text, block_no, block_type)
    return (x0, y0, x1, y1, text, 0, 0)


def test_two_column_page_reads_down_each_column():
    """The bug this guards: left/right blocks at the same y must NOT interleave."""
    page = FakePage(
        [
            block(30, 100, 280, 120, "left one"),
            block(320, 100, 570, 120, "right one"),
            block(30, 140, 280, 160, "left two"),
            block(320, 140, 570, 160, "right two"),
        ]
    )
    assert _page_text_in_reading_order(page).split("\n") == [
        "left one",
        "left two",
        "right one",
        "right two",
    ]


def test_full_width_header_precedes_the_columns():
    page = FakePage(
        [
            block(30, 40, 570, 70, "TITLE SPANNING PAGE"),
            block(30, 100, 280, 120, "left body"),
            block(320, 100, 570, 120, "right body"),
        ]
    )
    assert _page_text_in_reading_order(page).split("\n")[0] == "TITLE SPANNING PAGE"


def test_single_column_page_keeps_vertical_order():
    page = FakePage(
        [
            block(30, 200, 570, 220, "second"),
            block(30, 100, 570, 120, "first"),
        ]
    )
    assert _page_text_in_reading_order(page).split("\n") == ["first", "second"]


def test_empty_page_is_safe():
    assert _page_text_in_reading_order(FakePage([])) == ""


def test_non_text_blocks_are_ignored():
    """Block type 1 is an image; including it would inject binary noise."""
    image = (30, 100, 280, 200, "<image>", 0, 1)
    page = FakePage([image, block(30, 220, 570, 240, "real text")])
    assert _page_text_in_reading_order(page) == "real text"


# --- cleaning ---------------------------------------------------------


def test_hyphenated_words_are_rejoined():
    assert "retrieval" in clean_text("retrie-\nval works")


def test_ligatures_are_normalised():
    # U+FB00 is the "ff" ligature, U+FB01 is "fi" -- PDF text is full of them,
    # and leaving them in breaks both lexical matching and citation display.
    assert clean_text("e\ufb00icient \ufb01nding") == "efficient finding"


def test_soft_wraps_join_but_paragraphs_survive():
    out = clean_text("one line\ncontinues here\n\nnew paragraph")
    assert "one line continues here" in out
    assert "\n\n" in out


# --- sections ---------------------------------------------------------


def test_numbered_and_bare_headings_are_found():
    text = "Abstract\n\nsome text\n\n1 Introduction\n\nmore\n\n2. Related Work\n\nend"
    titles = [s.title for s in detect_sections(text)]
    assert "Abstract" in titles
    assert "Introduction" in titles
    assert "Related Work" in titles


def test_sections_tile_the_document_without_gaps():
    text = "Abstract\n\nbody\n\n1 Introduction\n\nmore body"
    sections = detect_sections(text)
    assert sections[0].char_start == 0
    assert sections[-1].char_end == len(text)
    for prev, nxt in zip(sections, sections[1:], strict=False):
        assert prev.char_end == nxt.char_start


def test_headingless_text_falls_back_to_one_body_section():
    sections = detect_sections("just prose with no headings at all")
    assert len(sections) == 1
    assert sections[0].title == "Body"
