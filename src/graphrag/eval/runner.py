"""Run the benchmark: every system against every gold question.

Four systems, chosen so the comparison isolates one variable at a time:

* ``bm25``     -- lexical only. The classical baseline.
* ``vector``   -- dense retrieval only. Standard RAG.
* ``hybrid``   -- vector + BM25 fused by RRF. A *strong* plain-RAG baseline,
                  deliberately so: beating a weak baseline proves nothing.
* ``graphrag`` -- the full system with routing, graph queries and global mode.

Results are written per question so a surprising number can be traced back to
the actual answer that produced it. An aggregate table nobody can audit is not
evidence.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from graphrag.config import settings
from graphrag.eval.dataset import (
    Category,
    GoldQuestion,
    build_chunk_index,
    chunk_ids_for_papers,
    load,
)
from graphrag.eval.judge import Judgement, judge_answer
from graphrag.eval.metrics import RetrievalScores, score_retrieval
from graphrag.llm.base import LLMBackend
from graphrag.llm.meter import scoped as scoped_meter
from graphrag.logging import get_logger

log = get_logger(__name__)

SYSTEMS = ("bm25", "vector", "hybrid", "graphrag")

# Sources produced by the graph and community retrievers rather than the
# passage index. They have no chunk to point at.
_SYNTHETIC_PREFIXES = ("graph-result", "community:")


def _is_synthetic(chunk_id: str) -> bool:
    return chunk_id.startswith(_SYNTHETIC_PREFIXES)


@dataclass
class QuestionResult:
    question_id: str
    question: str
    category: str
    system: str
    # Defaulted because the record is constructed *before* the run, so that a
    # crash mid-run still produces a result row carrying the error rather than
    # vanishing from the report.
    answer: str = ""
    mode: str = ""
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    retrieval: dict[str, float] = field(default_factory=dict)
    judgement: dict[str, object] | None = None
    latency_s: float = 0.0
    tokens: int = 0
    calls: int = 0
    error: str = ""


def _run_one(
    question: GoldQuestion,
    system: str,
    *,
    backend: LLMBackend,
    k: int,
) -> tuple[str, str, list, list[str]]:
    """Return (answer_text, mode, hits, retrieved_chunk_ids) for one system."""
    from graphrag.answer.synthesize import synthesize
    from graphrag.retrieve import ask as run_ask
    from graphrag.retrieve import search as run_search

    if system == "graphrag":
        result = run_ask(question.question, backend=backend, k=k)
        # result.hits, NOT result.answer.citations. Citations are the subset the
        # model chose to cite; baselines are measured over everything retrieved,
        # and mixing the two would compare different denominators.
        return (
            result.answer.text,
            result.mode,
            result.hits,
            [h.chunk_id for h in result.hits],
        )

    # Baselines: retrieval mode fixed, no routing, no graph.
    mode = {"bm25": "bm25", "vector": "vector", "hybrid": "hybrid"}[system]
    hits = run_search(question.question, k=k, mode=mode)
    answer = synthesize(question.question, hits, backend=backend, mode=mode)
    return answer.text, mode, hits, [h.chunk_id for h in hits]


def run_benchmark(
    *,
    backend: LLMBackend,
    systems: tuple[str, ...] = SYSTEMS,
    questions: list[GoldQuestion] | None = None,
    k: int = 8,
    judge: bool = True,
    out_path: Path | None = None,
    resume: bool = True,
) -> list[QuestionResult]:
    """Run every system on every question and write per-question results.

    Resumable by default. A full run is dozens of LLM calls against a rate
    limit, and losing an hour of completed runs to one 429 near the end is the
    same failure mode that checkpointing fixed for extraction. Successful runs
    are skipped on a re-run; failed ones are retried.
    """
    questions = questions or load()
    out_path = Path(out_path or settings.processed_dir / "eval_results.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # (question_id, system) pairs already completed without error.
    completed: dict[tuple[str, str], QuestionResult] = {}
    if resume:
        for prior in load_results(out_path):
            # A run only counts as complete if it produced an answer *and*, when
            # judging is on, a judgement. The judge can fail on its own while
            # the answer succeeds -- and that run carries no `error`, so
            # skipping on `error` alone would leave it permanently ungraded,
            # silently shrinking the sample the report averages over.
            if prior.error:
                continue
            if judge and prior.judgement is None:
                continue
            completed[(prior.question_id, prior.system)] = prior
        if completed:
            log.info("resuming_benchmark", already_done=len(completed))

    # Read the corpus once, not once per question.
    chunk_index = build_chunk_index()

    results: list[QuestionResult] = list(completed.values())
    total = len(questions) * len(systems)
    done = len(results)

    mode = "a" if completed else "w"
    with out_path.open(mode, encoding="utf-8") as fh:
        for question in questions:
            relevant = chunk_ids_for_papers(question.relevant_papers, chunk_index)

            for system in systems:
                if (question.id, system) in completed:
                    continue
                done += 1
                started = time.monotonic()
                record = QuestionResult(
                    question_id=question.id,
                    question=question.question,
                    category=question.category.value,
                    system=system,
                )

                try:
                    # Scoped to _run_one only, deliberately. The judge call
                    # below is evaluation overhead, not part of what the system
                    # costs to answer -- charging it back would penalise a
                    # system twice for producing longer answers.
                    with scoped_meter() as meter:
                        answer, mode, hits, chunk_ids = _run_one(
                            question, system, backend=backend, k=k
                        )

                    record.answer = answer
                    record.mode = mode
                    record.retrieved_chunk_ids = chunk_ids

                    # Retrieval metrics only make sense where the gold set names
                    # relevant papers. Aggregate and global questions are graded
                    # on the answer alone.
                    if relevant:
                        # Synthetic sources -- a graph result set or a community
                        # summary -- are not passages and can never match a gold
                        # chunk id. Counting them would dilute precision for
                        # exactly the modes this project is about. Passage
                        # recall for a graph-mode answer is legitimately 0: it
                        # answered from a different source, which the report says.
                        passages = [c for c in chunk_ids if not _is_synthetic(c)]
                        scores: RetrievalScores = score_retrieval(passages, relevant, k=k)
                        record.retrieval = scores.as_dict()

                    if judge:
                        context = "\n\n".join(h.text for h in hits)[:12000]
                        verdict: Judgement | None = judge_answer(
                            question=question.question,
                            reference=question.reference,
                            context=context,
                            answer=answer,
                            backend=backend,
                        )
                        if verdict:
                            record.judgement = verdict.model_dump()

                except Exception as exc:  # noqa: BLE001 - record and continue
                    record.error = f"{type(exc).__name__}: {str(exc)[:300]}"
                    log.warning(
                        "eval_run_failed",
                        question=question.id,
                        system=system,
                        error=record.error,
                    )

                record.latency_s = round(time.monotonic() - started, 2)
                record.tokens = meter.total_tokens
                record.calls = meter.total_calls

                fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
                fh.flush()
                results.append(record)

                log.info(
                    "eval_progress",
                    progress=f"{done}/{total}",
                    question=question.id,
                    system=system,
                    mode=record.mode,
                    correctness=(record.judgement or {}).get("correctness"),
                    tokens=record.tokens,
                )

    log.info("benchmark_complete", runs=len(results), output=str(out_path))
    return results


def load_results(path: Path | None = None) -> list[QuestionResult]:
    path = Path(path or settings.processed_dir / "eval_results.jsonl")
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(QuestionResult(**json.loads(line)))
    return out


def summarise(results: list[QuestionResult]) -> dict:
    """Aggregate by system, and by system x category.

    The per-category breakdown is the actual finding. A single overall average
    would blend the categories where the systems tie with the ones where they
    diverge, and hide the result entirely.
    """
    from graphrag.eval.metrics import mean

    def stats(rows: list[QuestionResult]) -> dict[str, float]:
        graded = [r for r in rows if r.judgement]
        return {
            "n": len(rows),
            "correctness": round(
                mean([float(r.judgement["correctness"]) for r in graded]), 2
            )
            if graded
            else 0.0,
            "faithfulness": round(
                mean([float(r.judgement["faithfulness"]) for r in graded]), 2
            )
            if graded
            else 0.0,
            "completeness": round(
                mean([float(r.judgement["completeness"]) for r in graded]), 2
            )
            if graded
            else 0.0,
            "recall@8": round(
                mean([r.retrieval.get("recall@8", 0.0) for r in rows if r.retrieval]), 3
            ),
            "mrr": round(mean([r.retrieval.get("mrr", 0.0) for r in rows if r.retrieval]), 3),
            "tokens": int(mean([float(r.tokens) for r in rows])),
            "latency_s": round(mean([r.latency_s for r in rows]), 2),
            "errors": sum(1 for r in rows if r.error),
        }

    systems = sorted({r.system for r in results})
    categories = [c.value for c in Category]

    return {
        "overall": {s: stats([r for r in results if r.system == s]) for s in systems},
        "by_category": {
            cat: {
                s: stats([r for r in results if r.system == s and r.category == cat])
                for s in systems
            }
            for cat in categories
            if any(r.category == cat for r in results)
        },
    }
