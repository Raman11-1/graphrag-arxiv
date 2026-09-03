"""Turn benchmark results into a readable report.

The report is the deliverable. Two rules shape it:

**Lead with the per-category table**, because that is the actual finding. If
GraphRAG wins overall but the win comes entirely from one category, saying so
is more useful and more honest than a single headline average.

**Report what happened, not what was hoped for.** If the baselines tie or win,
the report says so. A benchmark that can only produce one conclusion is not a
benchmark.
"""

from __future__ import annotations

from pathlib import Path

from graphrag.config import settings
from graphrag.eval.runner import QuestionResult, summarise
from graphrag.logging import get_logger

log = get_logger(__name__)

CATEGORY_LABELS = {
    "local": "Local (single-passage)",
    "multi_hop": "Multi-hop (relational)",
    "aggregate": "Aggregate (counting)",
    "global": "Global (corpus-wide)",
}

# What each category is meant to demonstrate. Stated up front so a reader can
# check the result against the claim rather than taking the numbers on trust.
CATEGORY_EXPECTATION = {
    "local": "Baselines should be competitive here -- this is what plain RAG is for.",
    "multi_hop": "GraphRAG should win: the answer is a set of connected entities.",
    "aggregate": "GraphRAG should win decisively: top-k retrieval cannot count.",
    "global": "GraphRAG should win via community summaries; baselines see 8 chunks.",
}


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def _winner(per_system: dict[str, dict], metric: str = "correctness") -> str:
    scored = [(s, v.get(metric, 0.0)) for s, v in per_system.items() if v.get("n")]
    if not scored:
        return "n/a"
    best = max(scored, key=lambda kv: kv[1])
    ties = [s for s, v in scored if abs(v - best[1]) < 1e-9]
    return " / ".join(sorted(ties)) if len(ties) > 1 else best[0]


def build_report(results: list[QuestionResult]) -> str:
    """Render the full markdown report."""
    if not results:
        return "# Evaluation\n\nNo results. Run the benchmark first.\n"

    summary = summarise(results)
    systems = sorted(summary["overall"])
    n_questions = len({r.question_id for r in results})

    out: list[str] = []
    out.append("# GraphRAG evaluation\n")
    out.append(
        f"{n_questions} gold questions x {len(systems)} systems = {len(results)} runs. "
        f"Judged by `{settings.judge_model}`; retrieval metrics are deterministic.\n"
    )

    # --- headline ------------------------------------------------------
    out.append("## Overall\n")
    out.append(
        _table(
            [
                "system", "correctness", "faithfulness", "completeness",
                "recall@8", "tokens", "errors",
            ],
            [
                [
                    s,
                    summary["overall"][s]["correctness"],
                    summary["overall"][s]["faithfulness"],
                    summary["overall"][s]["completeness"],
                    summary["overall"][s]["recall@8"],
                    f"{summary['overall'][s]['tokens']:,}",
                    summary["overall"][s]["errors"],
                ]
                for s in systems
            ],
        )
    )
    out.append("\nScores are 1-5. Higher is better throughout.\n")

    # --- the actual finding --------------------------------------------
    out.append("## By question category\n")
    out.append(
        "This breakdown is the finding. The claim under test is not that "
        "GraphRAG is uniformly better, but that it wins on questions whose "
        "answers are structured, and ties where plain retrieval already works.\n"
    )

    for category, per_system in summary["by_category"].items():
        label = CATEGORY_LABELS.get(category, category)
        out.append(f"### {label}\n")
        out.append(f"*Expectation: {CATEGORY_EXPECTATION.get(category, '')}*\n")
        out.append(
            _table(
                ["system", "correctness", "faithfulness", "completeness", "tokens"],
                [
                    [
                        s,
                        per_system[s]["correctness"],
                        per_system[s]["faithfulness"],
                        per_system[s]["completeness"],
                        f"{per_system[s]['tokens']:,}",
                    ]
                    for s in systems
                ],
            )
        )
        out.append(f"\n**Best: {_winner(per_system)}**\n")

    # --- cost ----------------------------------------------------------
    out.append("## Cost and latency\n")
    out.append(
        "Token counts cover answering only. Grading is evaluation overhead and is "
        "excluded — a system producing longer answers is more expensive to judge, "
        "and charging that back would penalise it twice.\n"
    )
    out.append(
        "**Do not read the latency column as system performance.** These runs were "
        "made against a rate-limited free tier, so a figure largely reflects how "
        "much 429 backoff a run happened to absorb, not how fast the system is. "
        "Token counts are unaffected and are the meaningful cost signal here.\n"
    )
    out.append(
        _table(
            ["system", "mean tokens/question", "mean latency (s)", "LLM calls/question"],
            [
                [
                    s,
                    f"{summary['overall'][s]['tokens']:,}",
                    summary["overall"][s]["latency_s"],
                    round(
                        sum(r.calls for r in results if r.system == s)
                        / max(sum(1 for r in results if r.system == s), 1),
                        1,
                    ),
                ]
                for s in systems
            ],
        )
    )

    # --- routing behaviour ---------------------------------------------
    graphrag_rows = [r for r in results if r.system == "graphrag" and r.mode]
    if graphrag_rows:
        out.append("\n## How GraphRAG routed each category\n")
        modes: dict[str, dict[str, int]] = {}
        for r in graphrag_rows:
            modes.setdefault(r.category, {}).setdefault(r.mode, 0)
            modes[r.category][r.mode] += 1
        out.append(
            _table(
                ["category", "routes chosen"],
                [
                    [
                        CATEGORY_LABELS.get(cat, cat),
                        ", ".join(f"{m} x{n}" for m, n in sorted(d.items())),
                    ]
                    for cat, d in modes.items()
                ],
            )
        )

    # --- per-question audit trail ---------------------------------------
    out.append("\n## Per-question results\n")
    out.append("Included so any surprising number can be traced to the answer behind it.\n")
    by_question: dict[str, list[QuestionResult]] = {}
    for r in results:
        by_question.setdefault(r.question_id, []).append(r)

    for qid in sorted(by_question):
        rows = by_question[qid]
        out.append(f"### {qid} — {rows[0].question}\n")
        out.append(f"*Category: {CATEGORY_LABELS.get(rows[0].category, rows[0].category)}*\n")
        out.append(
            _table(
                ["system", "mode", "correct", "faithful", "answer (truncated)"],
                [
                    [
                        r.system,
                        r.mode or "-",
                        (r.judgement or {}).get("correctness", "-"),
                        (r.judgement or {}).get("faithfulness", "-"),
                        " ".join(r.answer.split())[:110] + ("..." if len(r.answer) > 110 else ""),
                    ]
                    for r in sorted(rows, key=lambda x: x.system)
                ],
            )
        )
        out.append("")

    # --- limitations -----------------------------------------------------
    out.append("## Limitations\n")
    out.append(
        "- The gold set was written by the system's author, which is a real source "
        "of bias. Questions were written against the corpus rather than the "
        "implementation, but that does not eliminate it.\n"
        "- The judge is the same model family used to generate answers, which can "
        "favour its own outputs.\n"
        f"- {n_questions} questions is small. Differences of a few tenths of a point "
        "should not be treated as significant.\n"
        "- Retrieval metrics are only computed for questions with paper-level "
        "relevance labels; aggregate and global questions are graded on the answer.\n"
        "- **recall@8 measures passage retrieval only.** A graph-mode answer scores "
        "0 there by construction: it answered from the knowledge graph rather than "
        "from retrieved passages. Read it alongside correctness, not instead of it.\n"
        "- Question A1 has no fixed numeric answer. The corpus determines the true "
        "count, and using the system's own graph as ground truth would be circular, "
        "so A1 grades whether a system *computes* a count or hedges — not whether a "
        "particular number is correct.\n"
    )

    return "\n".join(out)


def write_report(results: list[QuestionResult], path: Path | None = None) -> Path:
    path = Path(path or Path(__file__).parent / "reports" / "report.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_report(results), encoding="utf-8")
    log.info("report_written", path=str(path))
    return path
