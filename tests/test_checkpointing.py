"""Extraction checkpoint semantics.

Checkpointing is what makes an unknown rate limit cost time instead of work.
The subtle rule: a FAILED window must be retried on the next run. Treating a
failure as "done" turns a transient 429 into permanently missing graph data
that nothing ever reports -- the graph is simply smaller than it should be.
"""

from __future__ import annotations

import json

from graphrag.extract.extractor import load_done, load_records


def write(path, records):
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )


def test_missing_file_is_not_an_error(tmp_path):
    assert load_done(tmp_path / "nope.jsonl") == {}
    assert load_records(tmp_path / "nope.jsonl") == []


def test_only_successes_count_as_done(tmp_path):
    p = tmp_path / "t.jsonl"
    write(p, [
        {"window_id": "w1", "status": "ok", "extraction": {}},
        {"window_id": "w2", "status": "failed", "error": "429"},
    ])

    done = load_done(p)
    assert "w1" in done
    assert "w2" not in done, "a failed window must be retried, not skipped forever"


def test_later_success_supersedes_earlier_failure(tmp_path):
    """The file is append-only, so a retried window has two records."""
    p = tmp_path / "t.jsonl"
    write(p, [
        {"window_id": "w1", "status": "failed", "error": "429"},
        {"window_id": "w1", "status": "ok", "extraction": {"entities": []}},
    ])
    assert "w1" in load_done(p)
    assert len(load_records(p)) == 2


def test_success_is_not_undone_by_a_later_failure(tmp_path):
    """Order must not matter -- success wins either way."""
    p = tmp_path / "t.jsonl"
    write(p, [
        {"window_id": "w1", "status": "ok", "extraction": {}},
        {"window_id": "w1", "status": "failed", "error": "timeout"},
    ])
    assert "w1" in load_done(p)


def test_truncated_final_line_is_survivable(tmp_path):
    """A hard kill mid-write leaves a partial line; earlier work must survive."""
    p = tmp_path / "t.jsonl"
    p.write_text(
        json.dumps({"window_id": "w1", "status": "ok"}) + "\n" + '{"window_id": "w2", "sta',
        encoding="utf-8",
    )
    done = load_done(p)
    assert list(done) == ["w1"]


def test_records_without_a_window_id_are_ignored(tmp_path):
    p = tmp_path / "t.jsonl"
    write(p, [{"status": "ok"}, {"window_id": "w1", "status": "ok"}])
    assert list(load_done(p)) == ["w1"]
