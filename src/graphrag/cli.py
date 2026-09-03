"""Command-line interface.

    graphrag ingest "retrieval augmented generation" --limit 20
    graphrag stats
    graphrag search "what is dense retrieval?"      # retrieval only, no LLM
    graphrag query  "what is dense retrieval?"      # retrieval + answer
    graphrag dry-run                                 # project extraction cost
"""

from __future__ import annotations

import sys
import textwrap

import typer

from graphrag.logging import configure_logging


def _force_utf8_output() -> None:
    """Make stdout/stderr UTF-8 safe.

    The Windows console defaults to cp1252, which cannot encode the maths
    symbols, ligatures and accented author names that fill academic PDFs --
    printing a retrieved chunk raises UnicodeEncodeError and kills the command.
    'replace' is deliberate: a mangled glyph in a terminal preview is far better
    than a crash, and the stored text is untouched.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

app = typer.Typer(
    add_completion=False,
    help="GraphRAG: hybrid knowledge-graph + vector RAG over research papers.",
)


@app.callback()
def _root(log_level: str = typer.Option("INFO", help="DEBUG, INFO, WARNING, ERROR")) -> None:
    _force_utf8_output()
    configure_logging(log_level)


@app.command()
def ingest(
    query: str = typer.Argument(..., help="arXiv search query"),
    limit: int = typer.Option(20, help="Maximum papers to fetch"),
    force: bool = typer.Option(False, help="Re-parse papers even if checkpointed"),
) -> None:
    """Fetch, parse, chunk and index papers. Makes no LLM calls."""
    from graphrag.ingest.pipeline import ingest as run_ingest

    summary = run_ingest(query, limit=limit, force=force)
    typer.echo("")
    for key, value in summary.items():
        typer.echo(f"  {key.replace('_', ' '):22} {value}")

    # Exit non-zero when nothing landed. A run that downloads no papers and
    # still reports success is how a broken fetcher stays invisible.
    if summary["papers_processed"] == 0:
        typer.echo("")
        typer.echo(
            "No papers were processed. Check the warnings above -- the query "
            "may have returned nothing, or PDF downloads may be failing."
        )
        raise typer.Exit(1)


@app.command()
def reindex() -> None:
    """Rebuild vector and BM25 indexes from papers already on disk.

    Use when `stats` shows fewer vectors than chunks -- indexing was
    interrupted. Makes no LLM calls and re-fetches nothing.
    """
    from graphrag.ingest.pipeline import reindex as run_reindex

    summary = run_reindex()
    for key, value in summary.items():
        typer.echo(f"  {key.replace('_', ' '):22} {value}")

    if summary["chunks"] and summary["vectors_indexed"] != summary["chunks"]:
        typer.echo("\nWarning: vector count still does not match chunk count.")
        raise typer.Exit(1)


@app.command()
def stats() -> None:
    """Show what is currently indexed."""
    from graphrag.ingest.pipeline import stats as get_stats

    for key, value in get_stats().items():
        typer.echo(f"  {key.replace('_', ' '):22} {value}")


@app.command()
def search(
    question: str = typer.Argument(...),
    k: int = typer.Option(8, help="Number of chunks to retrieve"),
    mode: str = typer.Option("hybrid", help="hybrid | vector | bm25"),
) -> None:
    """Retrieve chunks without calling an LLM. Free, and useful for debugging."""
    from graphrag.retrieve import search as run_search

    hits = run_search(question, k=k, mode=mode)
    if not hits:
        typer.echo("No results. Has anything been ingested? Try: graphrag stats")
        raise typer.Exit(1)

    for i, hit in enumerate(hits, start=1):
        preview = " ".join(hit.text.split())[:200]
        typer.echo(f"\n[{i}] {hit.paper_id}  score={hit.score:.4f}  via={hit.source}")
        if hit.section:
            typer.echo(f"    section: {hit.section}")
        typer.echo(f"    {preview}...")


@app.command()
def query(
    question: str = typer.Argument(...),
    k: int = typer.Option(8),
    mode: str = typer.Option("hybrid", help="hybrid | vector | bm25"),
) -> None:
    """Retrieve and answer with citations. Uses one LLM call."""
    from graphrag.answer import synthesize
    from graphrag.llm.factory import get_backend
    from graphrag.llm.meter import METER
    from graphrag.retrieve import search as run_search

    hits = run_search(question, k=k, mode=mode)
    answer = synthesize(question, hits, backend=get_backend(), mode=mode)

    typer.echo("")
    typer.echo(answer.text)

    if answer.citations:
        typer.echo("\nSources:")
        for i, hit in enumerate(answer.citations, start=1):
            typer.echo(f"  [{i}] {hit.paper_id}  chars {hit.char_start}-{hit.char_end}")

    if answer.dropped_citations:
        typer.echo(f"\n  (dropped invalid citations: {answer.dropped_citations})")

    typer.echo(
        f"\n  {METER.total_calls} call(s), {METER.total_tokens} tokens, "
        f"${METER.total_cost:.4f}"
    )


@app.command()
def ask(
    question: str = typer.Argument(...),
    k: int = typer.Option(8, help="Passages to retrieve"),
    mode: str | None = typer.Option(None, help="Force vector|graph|global|hybrid"),
    show_cypher: bool = typer.Option(False, help="Print the generated Cypher"),
) -> None:
    """Route, retrieve and answer. This is the full GraphRAG pipeline."""
    from graphrag.llm.factory import get_backend
    from graphrag.llm.meter import METER
    from graphrag.retrieve import ask as run_ask

    result = run_ask(question, backend=get_backend(), k=k, force_mode=mode)

    typer.echo("")
    typer.echo(f"  mode: {result.mode}  ({result.route_reason})")
    if result.fell_back:
        typer.echo("  note: graph returned nothing, fell back to passage retrieval")
    if show_cypher and result.cypher:
        typer.echo("")
        for line in result.cypher.splitlines():
            typer.echo(f"    {line}")
    if result.graph_rows:
        typer.echo(f"\n  graph rows: {len(result.graph_rows)}")
        for row in result.graph_rows[:6]:
            typer.echo(f"    {row}")

    typer.echo("")
    typer.echo(result.answer.text)

    if result.answer.citations:
        typer.echo("\nSources:")
        for i, hit in enumerate(result.answer.citations, start=1):
            where = "knowledge graph" if hit.source == "graph" else (
                f"{hit.paper_id} chars {hit.char_start}-{hit.char_end}"
            )
            typer.echo(f"  [{i}] {where}")
    if result.answer.dropped_citations:
        typer.echo(f"  (dropped invalid citations: {result.answer.dropped_citations})")

    typer.echo(
        f"\n  {METER.total_calls} call(s), {METER.total_tokens} tokens, "
        f"${METER.total_cost:.4f}"
    )


@app.command()
def graph(
    rebuild: bool = typer.Option(False, help="Rebuild the graph from checkpointed triples"),
) -> None:
    """Build the knowledge graph, or show what is in it."""
    from graphrag.graph.build import build_graph
    from graphrag.graph.store import GraphStore
    from graphrag.ingest.pipeline import load_all_parsed

    store = GraphStore()
    if rebuild:
        for key, value in build_graph(load_all_parsed(), store=store).items():
            typer.echo(f"  {key.replace('_', ' '):24} {value}")
        typer.echo("")

    counts = store.counts()
    if not counts:
        typer.echo("Graph is empty. Run `graphrag extract` then `graphrag graph --rebuild`.")
        raise typer.Exit(1)
    for table, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        typer.echo(f"  {table:26} {n}")


@app.command()
def communities(
    build: bool = typer.Option(False, help="Detect and summarise communities"),
    resolution: float = typer.Option(1.0, help="Higher = more, smaller communities"),
) -> None:
    """Cluster the graph into research themes and summarise each one."""
    from graphrag.graph.communities import (
        detect_communities,
        load_communities,
        persist,
        summarise_communities,
    )
    from graphrag.graph.store import GraphStore
    from graphrag.llm.factory import get_backend
    from graphrag.llm.meter import METER

    store = GraphStore()

    if build:
        found = detect_communities(store, resolution=resolution)
        if not found:
            typer.echo("No communities found. Is the graph built? `graphrag graph --rebuild`")
            raise typer.Exit(1)
        summarise_communities(found, backend=get_backend(), store=store)
        persist(found, store)
        typer.echo(f"\n  built {len(found)} communities, ${METER.total_cost:.4f}\n")

    existing = load_communities(store)
    if not existing:
        typer.echo("No communities stored. Run `graphrag communities --build`.")
        raise typer.Exit(1)

    for c in existing:
        typer.echo(f"  [{c.community_id}] {c.title}")
        if c.summary:
            for line in textwrap.wrap(c.summary, width=88):
                typer.echo(f"        {line}")
        typer.echo("")


@app.command()
def extract(
    force: bool = typer.Option(False, help="Re-extract windows already done"),
    limit: int | None = typer.Option(None, help="Only process this many windows"),
    model: str | None = typer.Option(None, help="Override the extraction model"),
) -> None:
    """Extract entities and relations from every window. Checkpointed and resumable."""
    from graphrag.extract.extractor import run_extraction
    from graphrag.ingest.pipeline import load_all_parsed
    from graphrag.llm.factory import get_backend
    from graphrag.llm.meter import METER

    papers = load_all_parsed()
    if not papers:
        typer.echo("Nothing ingested. Run `graphrag ingest <query>` first.")
        raise typer.Exit(1)

    windows = [w for p in papers for w in p.windows]
    titles = {p.paper.paper_id: p.paper.title for p in papers}

    summary = run_extraction(
        windows,
        titles=titles,
        backend=get_backend(),
        model=model,
        force=force,
        limit=limit,
    )

    typer.echo("")
    for key, value in summary.items():
        typer.echo(f"  {key.replace('_', ' '):24} {value:,}")
    typer.echo(f"  {'cost usd':24} ${METER.total_cost:.4f}")

    if summary["succeeded"] == 0 and summary["attempted"] > 0:
        typer.echo("\nEvery window failed. See the warnings above.")
        raise typer.Exit(1)


@app.command()
def evaluate(
    systems: str = typer.Option(
        "bm25,vector,hybrid,graphrag", help="Comma-separated systems to compare"
    ),
    k: int = typer.Option(8, help="Passages retrieved per question"),
    no_judge: bool = typer.Option(False, help="Skip LLM grading (retrieval metrics only)"),
    report_only: bool = typer.Option(False, help="Rebuild the report from saved results"),
) -> None:
    """Benchmark GraphRAG against the baselines and write the report."""
    from graphrag.eval.report import write_report
    from graphrag.eval.runner import load_results, run_benchmark
    from graphrag.llm.factory import get_backend

    if report_only:
        results = load_results()
        if not results:
            typer.echo("No saved results. Run `graphrag evaluate` first.")
            raise typer.Exit(1)
    else:
        chosen = tuple(s.strip() for s in systems.split(",") if s.strip())
        results = run_benchmark(
            backend=get_backend(), systems=chosen, k=k, judge=not no_judge
        )

    path = write_report(results)
    typer.echo(f"\n  {len(results)} runs -> {path}")

    from graphrag.eval.runner import summarise

    summary = summarise(results)
    typer.echo("")
    typer.echo(f"  {'system':12} {'correct':>8} {'faithful':>9} {'tokens':>9}")
    for system, stats in summary["overall"].items():
        typer.echo(
            f"  {system:12} {stats['correctness']:>8} "
            f"{stats['faithfulness']:>9} {stats['tokens']:>9,}"
        )


@app.command("dry-run")
def dry_run() -> None:
    """Project extraction token usage without spending anything."""
    from graphrag.ingest.pipeline import dry_run as run_dry

    result = run_dry("")
    if not result.get("windows"):
        typer.echo("Nothing ingested yet. Run `graphrag ingest <query>` first.")
        raise typer.Exit(1)

    for key, value in result.items():
        typer.echo(f"  {key.replace('_', ' '):26} {value:,}")


if __name__ == "__main__":
    app()
