"""FastAPI service.

Ingestion and extraction are slow and rate-limited, so they run as background
jobs rather than blocking a request. Everything else is fast enough to answer
inline.

The heavy singletons -- the ONNX embedder, the Chroma client, the Kuzu
connection -- are created once at startup rather than per request. Loading the
embedding model takes seconds, so doing it per request would dominate latency.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from graphrag.config import settings
from graphrag.logging import configure_logging, get_logger

log = get_logger(__name__)

# job_id -> status record. In-process and therefore lost on restart, which is
# the right trade for a single-user research tool; a queue would be overkill.
JOBS: dict[str, dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings.ensure_dirs()
    log.info("api_starting", backend=settings.llm_backend)
    yield
    log.info("api_stopping")


app = FastAPI(
    title="GraphRAG",
    description="Hybrid knowledge-graph + vector RAG over research papers.",
    version="0.1.0",
    lifespan=lifespan,
)


# --- schemas ----------------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=8, ge=1, le=50)
    mode: Literal["vector", "graph", "global", "hybrid"] | None = None


class Citation(BaseModel):
    index: int
    paper_id: str
    chunk_id: str
    char_start: int
    char_end: int
    section: str
    source: str
    preview: str


class AskResponse(BaseModel):
    question: str
    answer: str
    mode: str
    route_reason: str
    fell_back: bool
    cypher: str | None
    graph_rows: list[dict[str, Any]]
    citations: list[Citation]
    dropped_citations: list[int]
    tokens: int
    calls: int


class IngestRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=10, ge=1, le=100)


# --- endpoints --------------------------------------------------------


STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """The single-page UI, served from the same process as the API."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "backend": settings.llm_backend}


@app.get("/stats")
def stats() -> dict[str, Any]:
    """Index and graph size."""
    from graphrag.graph.store import GraphStore
    from graphrag.ingest.pipeline import stats as index_stats

    out: dict[str, Any] = {"index": index_stats()}
    try:
        out["graph"] = GraphStore(read_only=True).counts()
    except Exception as exc:  # noqa: BLE001 - graph may not exist yet
        out["graph"] = {}
        out["graph_error"] = str(exc)[:200]
    return out


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Route, retrieve and answer."""
    from graphrag.llm.factory import get_backend
    from graphrag.llm.meter import scoped as scoped_meter
    from graphrag.retrieve import ask as run_ask

    # Per-request meter so reported usage is this question's, not the process's.
    try:
        with scoped_meter() as request_meter:
            result = run_ask(
                request.question,
                backend=get_backend(),
                k=request.k,
                force_mode=request.mode,
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("ask_failed", error=str(exc)[:300])
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    return AskResponse(
        question=result.question,
        answer=result.answer.text,
        mode=result.mode,
        route_reason=result.route_reason,
        fell_back=result.fell_back,
        cypher=result.cypher,
        graph_rows=result.graph_rows,
        citations=[
            Citation(
                index=i,
                paper_id=hit.paper_id,
                chunk_id=hit.chunk_id,
                char_start=hit.char_start,
                char_end=hit.char_end,
                section=hit.section,
                source=hit.source,
                preview=" ".join(hit.text.split())[:400],
            )
            for i, hit in enumerate(result.answer.citations, start=1)
        ],
        dropped_citations=result.answer.dropped_citations,
        tokens=request_meter.total_tokens,
        calls=request_meter.total_calls,
    )


@app.get("/search")
def search(q: str, k: int = 8, mode: str = "hybrid") -> dict[str, Any]:
    """Retrieval only -- no LLM call, so this is free and fast."""
    from graphrag.retrieve import search as run_search

    hits = run_search(q, k=k, mode=mode)
    return {
        "query": q,
        "mode": mode,
        "hits": [
            {
                "chunk_id": h.chunk_id,
                "paper_id": h.paper_id,
                "score": round(h.score, 4),
                "section": h.section,
                "source": h.source,
                "preview": " ".join(h.text.split())[:400],
            }
            for h in hits
        ],
    }


@app.get("/graph/subgraph")
def subgraph(entity: str, depth: int = 1, limit: int = 60) -> dict[str, Any]:
    """Neighbourhood of one entity, for visualisation."""
    from graphrag.graph.store import GraphStore

    if depth not in (1, 2):
        raise HTTPException(status_code=400, detail="depth must be 1 or 2")

    store = GraphStore(read_only=True)
    hops = "1..2" if depth == 2 else "1..1"
    try:
        rows = store.query(
            f"MATCH (a)-[r*{hops}]-(b) "
            f"WHERE toLower(a.name) CONTAINS toLower($entity) "
            f"RETURN DISTINCT a.name AS source, b.name AS target LIMIT $limit",
            {"entity": entity, "limit": limit},
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)[:300]) from exc

    nodes = sorted({r[key] for r in rows for key in ("source", "target") if r.get(key)})
    return {
        "entity": entity,
        "nodes": [{"id": n} for n in nodes],
        "edges": [
            {"source": r["source"], "target": r["target"]}
            for r in rows
            if r.get("source") and r.get("target")
        ],
    }


@app.get("/communities")
def communities() -> dict[str, Any]:
    from graphrag.graph.communities import load_communities
    from graphrag.graph.store import GraphStore

    found = load_communities(GraphStore(read_only=True))
    return {
        "count": len(found),
        "communities": [
            {"id": c.community_id, "title": c.title, "summary": c.summary} for c in found
        ],
    }


def _run_ingest_job(job_id: str, query: str, limit: int) -> None:
    from graphrag.ingest.pipeline import ingest as run_ingest
    from graphrag.retrieve import reset_cache

    JOBS[job_id]["status"] = "running"
    try:
        JOBS[job_id]["result"] = run_ingest(query, limit=limit)
        JOBS[job_id]["status"] = "done"
        # New documents are invisible until the cached stores are dropped.
        reset_cache()
    except Exception as exc:  # noqa: BLE001
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        log.warning("ingest_job_failed", job=job_id, error=JOBS[job_id]["error"])


@app.post("/ingest")
def ingest(request: IngestRequest, background: BackgroundTasks) -> dict[str, str]:
    """Start an ingestion job. Returns immediately with a job id."""
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "queued", "query": request.query, "limit": request.limit}
    background.add_task(_run_ingest_job, job_id, request.query, request.limit)
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job id")
    return {"job_id": job_id, **job}
