"""Benchmark aggregation and report rendering.

These run on synthetic results, so they are deterministic and free. The point
is that the report must be able to express an *unfavourable* outcome: a
benchmark whose renderer only handles "GraphRAG wins" is not a benchmark.
"""

from __future__ import annotations

from graphrag.eval.report import build_report
from graphrag.eval.runner import QuestionResult, summarise


def result(
    qid: str,
    system: str,
    category: str,
    correctness: int,
    *,
    mode: str = "vector",
    tokens: int = 1000,
    judged: bool = True,
) -> QuestionResult:
    return QuestionResult(
        question_id=qid,
        question=f"question {qid}",
        category=category,
        system=system,
        answer=f"answer from {system}",
        mode=mode,
        retrieved_chunk_ids=["c1", "c2"],
        retrieval={"recall@8": 0.5, "mrr": 0.5},
        judgement=(
            {
                "correctness": correctness,
                "faithfulness": 4,
                "completeness": 3,
                "reason": "test",
            }
            if judged
            else None
        ),
        tokens=tokens,
        calls=2,
        latency_s=1.0,
    )


def test_summarise_splits_by_system_and_category():
    results = [
        result("A1", "graphrag", "aggregate", 5),
        result("A1", "vector", "aggregate", 2),
        result("L1", "graphrag", "local", 4),
        result("L1", "vector", "local", 4),
    ]
    summary = summarise(results)

    assert summary["overall"]["graphrag"]["correctness"] == 4.5
    assert summary["overall"]["vector"]["correctness"] == 3.0
    # The finding lives in the per-category split, not the average.
    assert summary["by_category"]["aggregate"]["graphrag"]["correctness"] == 5.0
    assert summary["by_category"]["aggregate"]["vector"]["correctness"] == 2.0
    assert summary["by_category"]["local"]["graphrag"]["correctness"] == 4.0


def test_summarise_ignores_unjudged_runs_rather_than_scoring_them_zero():
    """A judge outage must not be recorded as a system failure."""
    results = [
        result("A1", "graphrag", "aggregate", 5),
        result("A2", "graphrag", "aggregate", 0, judged=False),
    ]
    summary = summarise(results)
    assert summary["overall"]["graphrag"]["correctness"] == 5.0
    assert summary["overall"]["graphrag"]["n"] == 2


def test_summarise_counts_errors():
    bad = result("A1", "vector", "aggregate", 1)
    bad.error = "boom"
    assert summarise([bad])["overall"]["vector"]["errors"] == 1


def test_summarise_of_nothing_does_not_crash():
    assert summarise([]) == {"overall": {}, "by_category": {}}


# --- report -----------------------------------------------------------


def test_report_contains_the_category_breakdown():
    results = [
        result("A1", "graphrag", "aggregate", 5, mode="graph"),
        result("A1", "vector", "aggregate", 2),
    ]
    report = build_report(results)

    assert "Aggregate (counting)" in report
    assert "graphrag" in report and "vector" in report
    assert "Limitations" in report, "the report must state its own weaknesses"


def test_report_names_the_winner_per_category():
    results = [
        result("A1", "graphrag", "aggregate", 5),
        result("A1", "vector", "aggregate", 2),
    ]
    assert "**Best: graphrag**" in build_report(results)


def test_report_can_express_a_baseline_win():
    """The renderer must not be able to produce only one conclusion."""
    results = [
        result("L1", "graphrag", "local", 2),
        result("L1", "vector", "local", 5),
    ]
    assert "**Best: vector**" in build_report(results)


def test_report_marks_a_tie_as_a_tie():
    results = [
        result("L1", "graphrag", "local", 4),
        result("L1", "vector", "local", 4),
    ]
    assert "**Best: graphrag / vector**" in build_report(results)


def test_report_records_which_routes_were_taken():
    results = [
        result("A1", "graphrag", "aggregate", 5, mode="graph"),
        result("L1", "graphrag", "local", 4, mode="vector"),
    ]
    report = build_report(results)
    assert "How GraphRAG routed each category" in report
    assert "graph x1" in report


def test_report_includes_a_per_question_audit_trail():
    report = build_report([result("A1", "graphrag", "aggregate", 5)])
    assert "A1" in report
    assert "answer from graphrag" in report


def test_empty_results_produce_a_report_not_a_crash():
    assert "No results" in build_report([])


# --- synthetic source handling ----------------------------------------


def test_graph_and_community_ids_are_recognised_as_synthetic():
    """These are not passages and can never match a gold chunk id.

    Counting them would dilute precision for exactly the retrieval modes this
    project is about.
    """
    from graphrag.eval.runner import _is_synthetic

    assert _is_synthetic("graph-result")
    assert _is_synthetic("community:c0")
    assert not _is_synthetic("a3f9b2c1d4e5")


def test_report_states_the_recall_caveat():
    """A graph-mode answer scores 0 passage recall by construction; a reader
    must not mistake that for a retrieval failure."""
    report = build_report([result("A1", "graphrag", "aggregate", 5, mode="graph")])
    assert "recall@8 measures passage retrieval only" in report


# --- benchmark resume --------------------------------------------------
#
# A full benchmark is dozens of rate-limited LLM calls. Losing completed runs
# to one 429 near the end is the same failure that checkpointing fixed for
# extraction, so the same rule applies: successes are skipped, failures retry.


def test_completed_runs_are_skipped_on_resume(tmp_path, monkeypatch):
    import json
    from dataclasses import asdict

    from graphrag.eval import runner as run_mod
    from graphrag.eval.dataset import Category, GoldQuestion

    out = tmp_path / "eval.jsonl"
    prior = result("L1", "vector", "local", 4)
    out.write_text(json.dumps(asdict(prior)) + "\n", encoding="utf-8")

    ran: list[tuple[str, str]] = []

    def fake_run_one(question, system, *, backend, k):
        ran.append((question.id, system))
        return "answer", system, [], []

    monkeypatch.setattr(run_mod, "_run_one", fake_run_one)
    monkeypatch.setattr(run_mod, "build_chunk_index", lambda: {})
    monkeypatch.setattr(run_mod, "chunk_ids_for_papers", lambda ids, idx=None: set())

    questions = [
        GoldQuestion(id="L1", question="q1", category=Category.LOCAL, reference="r"),
        GoldQuestion(id="L2", question="q2", category=Category.LOCAL, reference="r"),
    ]
    results = run_mod.run_benchmark(
        backend=object(),
        systems=("vector",),
        questions=questions,
        judge=False,
        out_path=out,
    )

    assert ("L1", "vector") not in ran, "an already-completed run must be skipped"
    assert ("L2", "vector") in ran
    assert len(results) == 2, "prior results must still appear in the summary"


def test_failed_runs_are_retried_on_resume(tmp_path, monkeypatch):
    import json
    from dataclasses import asdict

    from graphrag.eval import runner as run_mod
    from graphrag.eval.dataset import Category, GoldQuestion

    out = tmp_path / "eval.jsonl"
    failed = result("L1", "vector", "local", 1)
    failed.error = "SDKError: 429"
    out.write_text(json.dumps(asdict(failed)) + "\n", encoding="utf-8")

    ran: list[tuple[str, str]] = []

    def fake_run_one(question, system, *, backend, k):
        ran.append((question.id, system))
        return "answer", system, [], []

    monkeypatch.setattr(run_mod, "_run_one", fake_run_one)
    monkeypatch.setattr(run_mod, "build_chunk_index", lambda: {})
    monkeypatch.setattr(run_mod, "chunk_ids_for_papers", lambda ids, idx=None: set())

    run_mod.run_benchmark(
        backend=object(),
        systems=("vector",),
        questions=[
            GoldQuestion(id="L1", question="q1", category=Category.LOCAL, reference="r")
        ],
        judge=False,
        out_path=out,
    )
    assert ("L1", "vector") in ran, "a failed run must be retried, not treated as done"


def test_an_ungraded_run_is_retried_when_judging_is_on(tmp_path, monkeypatch):
    """The judge can fail while the answer succeeds.

    Such a run carries no `error`, so resuming on `error` alone would treat it
    as complete and leave it permanently ungraded -- silently shrinking the
    sample the report averages over, with nothing to show for it.
    """
    import json
    from dataclasses import asdict

    from graphrag.eval import runner as run_mod
    from graphrag.eval.dataset import Category, GoldQuestion

    out = tmp_path / "eval.jsonl"
    ungraded = result("L1", "vector", "local", 4)
    ungraded.judgement = None  # answer fine, judge failed
    out.write_text(json.dumps(asdict(ungraded)) + "\n", encoding="utf-8")

    ran: list[tuple[str, str]] = []

    def fake_run_one(question, system, *, backend, k):
        ran.append((question.id, system))
        return "answer", system, [], []

    monkeypatch.setattr(run_mod, "_run_one", fake_run_one)
    monkeypatch.setattr(run_mod, "build_chunk_index", lambda: {})
    monkeypatch.setattr(run_mod, "chunk_ids_for_papers", lambda ids, idx=None: set())
    monkeypatch.setattr(run_mod, "judge_answer", lambda **kw: None)

    questions = [
        GoldQuestion(id="L1", question="q1", category=Category.LOCAL, reference="r")
    ]
    run_mod.run_benchmark(
        backend=object(), systems=("vector",), questions=questions, judge=True, out_path=out
    )
    assert ("L1", "vector") in ran, "an ungraded run must be retried when judging is on"


def test_an_ungraded_run_is_kept_when_judging_is_off(tmp_path, monkeypatch):
    """With --no-judge, a missing judgement is expected, not a gap."""
    import json
    from dataclasses import asdict

    from graphrag.eval import runner as run_mod
    from graphrag.eval.dataset import Category, GoldQuestion

    out = tmp_path / "eval.jsonl"
    ungraded = result("L1", "vector", "local", 4)
    ungraded.judgement = None
    out.write_text(json.dumps(asdict(ungraded)) + "\n", encoding="utf-8")

    ran: list[tuple[str, str]] = []
    monkeypatch.setattr(
        run_mod, "_run_one", lambda q, s, **kw: (ran.append((q.id, s)), ("a", s, [], []))[1]
    )
    monkeypatch.setattr(run_mod, "build_chunk_index", lambda: {})
    monkeypatch.setattr(run_mod, "chunk_ids_for_papers", lambda ids, idx=None: set())

    run_mod.run_benchmark(
        backend=object(),
        systems=("vector",),
        questions=[
            GoldQuestion(id="L1", question="q1", category=Category.LOCAL, reference="r")
        ],
        judge=False,
        out_path=out,
    )
    assert ran == [], "without judging, an ungraded run is already complete"


def test_report_warns_that_latency_is_contaminated_by_rate_limiting():
    """A 422s mean latency for bm25 is 429 backoff, not system speed.

    Without this warning the cost table invites a comparison the data cannot
    support -- and the numbers look precise enough to be believed.
    """
    report = build_report([result("L1", "bm25", "local", 4)])
    assert "Do not read the latency column as system performance" in report
