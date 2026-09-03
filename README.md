# GraphRAG

A question-answering system over research papers that combines **vector retrieval** with a
**knowledge graph**, routes each question to whichever can actually answer it, and measures whether
the combination beats plain RAG.

---

## The problem

Standard RAG splits documents into passages, embeds them, and returns the passages most similar to
your question. That works for *"what is dense passage retrieval?"* and fails for three whole classes
of question:

| Question | Why plain RAG fails |
|---|---|
| *"Which authors worked on both retrieval and generation?"* | The answer is a **relationship between papers**. No single passage contains it. |
| *"How many papers evaluate on Natural Questions?"* | **Counting requires seeing everything.** Eight retrieved chunks cannot tell you "eleven". |
| *"What are the main themes in this corpus?"* | It is a question about the **whole corpus**, not any part of it. |

These are not edge cases — they are what someone reading a literature actually asks.

## The approach

Alongside the vector index, an LLM reads each paper and extracts structured facts:

```
(DPR paper) ──PROPOSES─────→ (Dense Passage Retrieval)
(DPR paper) ──EVALUATES_ON─→ (Natural Questions)
(DPR paper) ──REPORTS──────→ (Top-20 accuracy: 78.4 on Natural Questions)
```

Those become a Cypher-queryable graph. A router then sends each question to the retriever whose
shape matches it, and **every mode falls back to passage retrieval**, so a routing mistake degrades
an answer rather than breaking it.

```
                    ┌──────────────────────────────────────┐
   arXiv PDF ──────►│ PARSE  (column-aware, section-aware) │
                    └───────┬──────────────────────┬───────┘
                            │                      │
                  chunks (~800 tok)      windows (~4000 tok)
                            │                      │
                    fastembed + BM25        LLM extraction
                            │                      │
                     Chroma index          triples + provenance
                            │                      │
                            │              entity resolution
                            │                      │
                            │                    Kùzu
                            │                      │
                            │            Leiden communities
                            │                      │
                            └──────────┬───────────┘
                                       ▼
   question ──► ROUTER ──► vector │ graph │ global │ hybrid ──► answer + citations
```

### Two chunkings, one parse

Retrieval and extraction want opposite things. Retrieval wants **small** spans so the answer context
stays precise. Extraction wants **wide** spans, because a relation like *"we evaluate RETRO on the
Pile, reaching 3.2 perplexity"* routinely straddles a paragraph break — an 800-token window hands
the model half of it and gets back a broken triple.

Measured on this corpus: **5.1× fewer extraction calls, and better triples.**

---

## Setup

Requires **Python 3.13** — Kùzu publishes no Windows wheel for 3.14.

```bash
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows
pip install -e ".[dev]"

cp .env.example .env                   # then add MISTRAL_API_KEY
python scripts/probe_limits.py         # checks the key and measures latency
```

The default backend is Mistral's free tier, so **running this costs nothing**. Embeddings run
locally via `fastembed` (ONNX, no PyTorch), as do both databases.

> **On rate limits:** the free tier's binding constraint is **tokens per minute**, not requests per
> second. The probe sends small requests and will report ~1.6 req/s with no throttling — that number
> does *not* transfer to extraction, whose calls carry ~17k tokens each and hit 429 immediately at
> that rate. `LLM_RPS=0.25` is what actually sustains extraction. Every stage retries 429s and
> checkpoints its progress, so a throttled run costs wall-clock time, never completed work.

> The free tier requires opting in to data training. Fine for public arXiv papers; **do not point
> it at private documents.** Set `LLM_BACKEND=anthropic` to use the Claude API instead.

## Usage

Everything at once:

```bash
python scripts/run_pipeline.py --query "retrieval augmented generation" --limit 20
```

Every stage is resumable — re-running after an interruption picks up where it stopped. Use
`--skip ingest,extract` to redo only part of a run.

Or stage by stage:

```bash
graphrag ingest "retrieval augmented generation" --limit 20   # no LLM calls, free
graphrag reindex                                              # rebuild indexes, no re-fetch
graphrag extract                                              # checkpointed, resumable
graphrag graph --rebuild
graphrag communities --build

graphrag ask "Which datasets does DPR evaluate on?" --show-cypher
graphrag search "how does dense retrieval work?"              # retrieval only, free
graphrag evaluate                                             # resumable; writes the report

uvicorn graphrag.api.main:app --reload                        # http://localhost:8000/docs
streamlit run src/graphrag/ui/app.py
```

## What it looks like working

```
$ graphrag ask "Which datasets does DPR evaluate on?" --show-cypher

  mode: graph  (the answer is a set of entities)

    MATCH (m:Method)-[:EVALUATES_ON]->(d:Dataset)
    WHERE toLower(m.name) CONTAINS 'dense passage' OR m.aliases CONTAINS 'dpr'
    RETURN DISTINCT d.name AS dataset
    LIMIT 50

  Natural Questions · TriviaQA · WebQuestions · CuratedTREC · SQuAD v1.1
  1,218 tokens        (the vector path used 8,986 and lost the SQuAD version)
```

---

## Design decisions worth knowing

**Provenance is not optional.** Every extracted triple carries the window it came from, the
character span, the verbatim sentence, and a confidence. A claim you cannot trace is a claim you
cannot trust.

**Citations are validated, not assumed.** If the model writes `[7]` when six sources were supplied,
that citation is dropped and logged. Otherwise an answer can look well-sourced while cited to
nothing.

**Generated Cypher is guarded at the token level.** Substring matching is not enough: it misses
`MATCH(n)DETACH DELETE n` and falsely rejects a dataset legitimately named `'CREATE-Bench'`. Queries
must also start with a read clause and carry a `LIMIT`. 23 tests cover real attacks.

**Extraction is checkpointed, and only successes count as done.** A failed window is retried on the
next run — otherwise a transient rate limit becomes permanently missing graph data that nothing ever
reports.

**Entity resolution refuses to merge across differing numbers.** Embeddings rate `Yang et al. (2018)`
and `Yang et al. (2019)` at ~0.98 similarity. Merging them fused distinct papers and rewrote
`SQuAD v1.1` into `SQuAD 2.0` — over-merging does not just hide facts, it fabricates them. Every
merge is written to an audit log.

## Evaluation

`graphrag evaluate` runs four systems — `bm25`, `vector`, `hybrid`, `graphrag` — across a gold set
split into four question categories, and writes a report.

`hybrid` is deliberately a **strong** baseline (vector + BM25 fused by reciprocal rank fusion).
Beating a weak baseline would prove nothing.

Retrieval metrics (recall@k, MRR, nDCG) are pure arithmetic against gold labels — no LLM, no cost,
fully deterministic. Answer quality is graded by an LLM judge that never learns which system
produced which answer.

The report leads with the **per-category** breakdown, because that is the actual claim: not that
GraphRAG is uniformly better, but that it wins where answers are structured and ties where plain
retrieval already works. The report also states its own limitations, including that the gold set was
written by this project's author.

## Project layout

```
src/graphrag/
├── ingest/     PDF parsing (column-aware), dual chunking, arXiv fetching
├── index/      fastembed embeddings, Chroma, BM25, RRF fusion
├── llm/        backend protocol, rate limiting, usage metering, Mistral client
├── extract/    extraction schemas and prompts, checkpointed runner, entity resolver
├── graph/      Kùzu schema and store, graph builder, Leiden communities
├── retrieve/   hybrid search, text-to-Cypher, router, global search, pipeline
├── answer/     answer synthesis with citation validation
├── eval/       gold set, metrics, judge, benchmark runner, report
├── api/        FastAPI service
└── ui/         Streamlit interface
```

240 tests. `pytest tests/ -q`
