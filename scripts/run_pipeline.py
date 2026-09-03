"""Run the whole pipeline end to end.

    python scripts/run_pipeline.py --query "retrieval augmented generation" --limit 20

Every stage is resumable, so re-running after an interruption picks up where it
stopped rather than starting over. Stages can also be skipped individually when
you only want to redo part of the run.

Order matters and is enforced: extraction needs chunks, the graph needs
extractions, communities need the graph, and the benchmark needs all of it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graphrag.logging import configure_logging, get_logger  # noqa: E402

log = get_logger(__name__)

STAGES = ("ingest", "extract", "graph", "communities", "evaluate")


def _banner(name: str, step: int, total: int) -> float:
    print()
    print("=" * 70)
    print(f"  [{step}/{total}]  {name.upper()}")
    print("=" * 70)
    return time.monotonic()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--query", default="retrieval augmented generation")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument(
        "--skip",
        default="",
        help=f"Comma-separated stages to skip. One of: {', '.join(STAGES)}",
    )
    ap.add_argument("--no-judge", action="store_true", help="Benchmark without LLM grading")
    args = ap.parse_args()

    configure_logging()
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    unknown = skip - set(STAGES)
    if unknown:
        print(f"Unknown stage(s) to skip: {sorted(unknown)}. Valid: {STAGES}")
        return 2

    todo = [s for s in STAGES if s not in skip]
    total = len(todo)
    step = 0
    started = time.monotonic()

    # --- ingest -------------------------------------------------------
    if "ingest" in todo:
        step += 1
        t = _banner("ingest", step, total)
        from graphrag.ingest.pipeline import ingest

        summary = ingest(args.query, limit=args.limit)
        if summary["papers_processed"] == 0:
            print("No papers ingested. Stopping -- later stages have nothing to work on.")
            return 1
        print(f"  {summary}   ({time.monotonic() - t:.0f}s)")

    # --- extract ------------------------------------------------------
    if "extract" in todo:
        step += 1
        t = _banner("extract", step, total)
        from graphrag.extract.extractor import run_extraction
        from graphrag.ingest.pipeline import load_all_parsed
        from graphrag.llm.factory import get_backend

        papers = load_all_parsed()
        if not papers:
            print("Nothing ingested. Run without --skip ingest.")
            return 1

        summary = run_extraction(
            [w for p in papers for w in p.windows],
            titles={p.paper.paper_id: p.paper.title for p in papers},
            backend=get_backend(),
        )
        print(f"  {summary}   ({time.monotonic() - t:.0f}s)")
        if summary["succeeded"] == 0 and summary["attempted"] > 0:
            print("Every window failed. Stopping before building an empty graph.")
            return 1

    # --- graph --------------------------------------------------------
    if "graph" in todo:
        step += 1
        t = _banner("graph", step, total)
        from graphrag.graph.build import build_graph
        from graphrag.graph.store import GraphStore
        from graphrag.ingest.pipeline import load_all_parsed

        store = GraphStore()
        summary = build_graph(load_all_parsed(), store=store)
        print(f"  {summary}   ({time.monotonic() - t:.0f}s)")
        for table, n in sorted(store.counts().items(), key=lambda kv: -kv[1])[:8]:
            print(f"    {table:26} {n}")

    # --- communities ---------------------------------------------------
    if "communities" in todo:
        step += 1
        t = _banner("communities", step, total)
        from graphrag.graph.communities import (
            detect_communities,
            persist,
            summarise_communities,
        )
        from graphrag.graph.store import GraphStore
        from graphrag.llm.factory import get_backend

        store = GraphStore()
        found = detect_communities(store)
        if not found:
            print("  No communities found -- the graph may be too sparse. Continuing.")
        else:
            summarise_communities(found, backend=get_backend(), store=store)
            persist(found, store)
            print(f"  {len(found)} communities   ({time.monotonic() - t:.0f}s)")
            for c in found[:6]:
                print(f"    [{c.size:3d}] {c.title}")

    # --- evaluate ------------------------------------------------------
    if "evaluate" in todo:
        step += 1
        t = _banner("evaluate", step, total)
        from graphrag.eval.report import write_report
        from graphrag.eval.runner import run_benchmark, summarise
        from graphrag.llm.factory import get_backend

        results = run_benchmark(backend=get_backend(), judge=not args.no_judge)
        path = write_report(results)
        print(f"  report -> {path}   ({time.monotonic() - t:.0f}s)")

        summary = summarise(results)
        print()
        print(f"  {'system':12} {'correct':>8} {'faithful':>9} {'tokens':>9}")
        for system, stats in summary["overall"].items():
            print(
                f"  {system:12} {stats['correctness']:>8} "
                f"{stats['faithfulness']:>9} {stats['tokens']:>9,}"
            )

    print()
    print("=" * 70)
    print(f"  pipeline finished in {(time.monotonic() - started) / 60:.1f} min")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
