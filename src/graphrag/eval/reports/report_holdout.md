# GraphRAG evaluation

12 gold questions x 4 systems = 48 runs. Judged by `mistral-medium-latest`; retrieval metrics are deterministic.

## Overall

| system | correctness | faithfulness | completeness | recall@8 | tokens | errors |
|---|---|---|---|---|---|---|
| bm25 | 3.0 | 2.5 | 2.92 | 0.346 | 8,045 | 0 |
| graphrag | 4.67 | 4.92 | 4.58 | 0.17 | 3,131 | 0 |
| hybrid | 3.1 | 3.4 | 3.0 | 0.378 | 7,838 | 0 |
| vector | 4.27 | 4.18 | 4.18 | 0.28 | 7,273 | 1 |

Scores are 1-5. Higher is better throughout.

## By question category

This breakdown is the finding. The claim under test is not that GraphRAG is uniformly better, but that it wins on questions whose answers are structured, and ties where plain retrieval already works.

### Local (single-passage)

*Expectation: Baselines should be competitive here -- this is what plain RAG is for.*

| system | correctness | faithfulness | completeness | tokens |
|---|---|---|---|---|
| bm25 | 5.0 | 3.67 | 5.0 | 8,006 |
| graphrag | 4.67 | 4.67 | 4.67 | 8,268 |
| hybrid | 5.0 | 5.0 | 5.0 | 7,870 |
| vector | 5.0 | 5.0 | 5.0 | 8,019 |

**Best: bm25 / hybrid / vector**

### Multi-hop (relational)

*Expectation: GraphRAG should win: the answer is a set of connected entities.*

| system | correctness | faithfulness | completeness | tokens |
|---|---|---|---|---|
| bm25 | 3.33 | 3.67 | 3.33 | 7,850 |
| graphrag | 5.0 | 5.0 | 5.0 | 1,741 |
| hybrid | 3.5 | 3.0 | 3.5 | 7,812 |
| vector | 3.5 | 4.0 | 3.5 | 5,499 |

**Best: graphrag**

### Aggregate (counting)

*Expectation: GraphRAG should win decisively: top-k retrieval cannot count.*

| system | correctness | faithfulness | completeness | tokens |
|---|---|---|---|---|
| bm25 | 2.0 | 1.67 | 1.67 | 8,503 |
| graphrag | 5.0 | 5.0 | 5.0 | 1,239 |
| hybrid | 1.5 | 1.0 | 1.5 | 7,976 |
| vector | 4.0 | 3.67 | 4.0 | 7,758 |

**Best: graphrag**

### Global (corpus-wide)

*Expectation: GraphRAG should win via community summaries; baselines see 8 chunks.*

| system | correctness | faithfulness | completeness | tokens |
|---|---|---|---|---|
| bm25 | 1.67 | 1.0 | 1.67 | 7,820 |
| graphrag | 4.0 | 5.0 | 3.67 | 1,277 |
| hybrid | 2.0 | 3.67 | 1.67 | 7,692 |
| vector | 4.33 | 4.0 | 4.0 | 7,816 |

**Best: vector**

## Cost and latency

Token counts cover answering only. Grading is evaluation overhead and is excluded — a system producing longer answers is more expensive to judge, and charging that back would penalise it twice.

**Do not read the latency column as system performance.** These runs were made against a rate-limited free tier, so a figure largely reflects how much 429 backoff a run happened to absorb, not how fast the system is. Token counts are unaffected and are the meaningful cost signal here.

| system | mean tokens/question | mean latency (s) | LLM calls/question |
|---|---|---|---|
| bm25 | 8,045 | 59.54 | 1.0 |
| graphrag | 3,131 | 28.33 | 2.1 |
| hybrid | 7,838 | 76.99 | 1.0 |
| vector | 7,273 | 62.33 | 0.9 |

## How GraphRAG routed each category

| category | routes chosen |
|---|---|
| Local (single-passage) | vector x3 |
| Multi-hop (relational) | graph x3 |
| Aggregate (counting) | graph x3 |
| Global (corpus-wide) | global x2, graph x1 |

## Per-question results

Included so any surprising number can be traced to the answer behind it.

### HA1 — How many papers here work on visual question answering?

*Category: Aggregate (counting)*

| system | mode | correct | faithful | answer (truncated) |
|---|---|---|---|---|
| bm25 | bm25 | 1 | 1 | Selected Label: REASONER |
| graphrag | graph | 5 | 5 | There are no papers in the provided sources that work on visual question answering [1]. |
| hybrid | hybrid | 2 | 1 | The sources provided contain the following papers that work on visual question answering: 1. Paper 2012.07192v... |
| vector | vector | 5 | 5 | The following papers work on visual question answering: - [1] Hengel, “Visual question answering: A survey of ... |

### HA2 — Which task is addressed by the largest number of methods in this collection?

*Category: Aggregate (counting)*

| system | mode | correct | faithful | answer (truncated) |
|---|---|---|---|---|
| bm25 | bm25 | 2 | 1 | The task addressed by the largest number of methods in the collection is **Named Entity Disambiguation (NED)**... |
| graphrag | graph | 5 | 5 | The task addressed by the largest number of methods in this collection is **Question Profiling**, with 6 disti... |
| hybrid | hybrid | 1 | 1 | The largest number of methods in the collection address the **Named Entity Disambiguation (NED)** task. This i... |
| vector | vector | 5 | 5 | The sources do not contain the answer to the question: "Which task is addressed by the largest number of metho... |

### HA3 — Count the distinct evaluation datasets that appear across all of these papers.

*Category: Aggregate (counting)*

| system | mode | correct | faithful | answer (truncated) |
|---|---|---|---|---|
| bm25 | bm25 | 3 | 3 | The distinct evaluation datasets mentioned across the provided sources are: 1. KRVQA [1] 2. LC-QuAD [3] 3. ISL... |
| graphrag | graph | 5 | 5 | The papers collectively evaluate on 26 distinct datasets [1]. |
| hybrid | hybrid | - | - | The distinct evaluation datasets mentioned across the provided sources are: 1. **SciTLDR** [1][2] 2. **T-REx Q... |
| vector | vector | 2 | 1 | The distinct evaluation datasets mentioned across the provided sources are: 1. T-REx QA [1] 2. PAQ [1] 3. NQ (... |

### HG1 — What application domains do these papers cover?

*Category: Global (corpus-wide)*

| system | mode | correct | faithful | answer (truncated) |
|---|---|---|---|---|
| bm25 | bm25 | 2 | 1 | The papers cover the following application domains: - Knowledge graph reasoning, representation, and integrati... |
| graphrag | global | 3 | 5 | The papers cover the following application domains: - Open-domain question answering, using datasets like Natu... |
| hybrid | hybrid | 1 | 5 | The provided sources do not contain the answer to your question. Please provide the specific user question, su... |
| vector | vector | 5 | 5 | The provided sources do not contain information about the application domains of the papers mentioned in the u... |

### HG2 — Give me an overview of the retrieval techniques represented here.

*Category: Global (corpus-wide)*

| system | mode | correct | faithful | answer (truncated) |
|---|---|---|---|---|
| bm25 | bm25 | 2 | 1 | The sources provided describe the following retrieval techniques: 1. **Frequency-based approach (spaCy)**: Use... |
| graphrag | global | 4 | 5 | The sources describe several distinct retrieval techniques and their evaluation contexts. - Dense retrieval me... |
| hybrid | hybrid | 3 | 5 | Based on the provided sources, here are the retrieval techniques represented: 1. **Dense Passage Retrieval (DP... |
| vector | vector | 3 | 2 | Here is an overview of the retrieval techniques represented in the sources: 1. **Sparse Retrieval (TF-IDF/BM25... |

### HG3 — Which research areas show up most often in this body of work?

*Category: Global (corpus-wide)*

| system | mode | correct | faithful | answer (truncated) |
|---|---|---|---|---|
| bm25 | bm25 | 1 | 1 | What is the name of the wearable fact-checking system described in source [2]? The wearable fact-checking syst... |
| graphrag | graph | 5 | 5 | The knowledge graph query returns eight research areas, each addressed by exactly one paper, so no area appear... |
| hybrid | hybrid | 2 | 1 | The most frequently referenced research areas in these sources are: - **Knowledge Graph (KG) reasoning and emb... |
| vector | vector | 5 | 5 | The sources do not contain the answer to the question "Which research areas show up most often in this body of... |

### HL1 — What is EVOR and how does it approach code generation?

*Category: Local (single-passage)*

| system | mode | correct | faithful | answer (truncated) |
|---|---|---|---|---|
| bm25 | bm25 | 5 | 1 | EVOR is a novel retrieval-augmented code generation (RACG) pipeline that employs **synchronous evolution of bo... |
| graphrag | vector | 5 | 5 | EVOR is a novel pipeline for **retrieval-augmented code generation (RACG)** that introduces **synchronous evol... |
| hybrid | hybrid | 5 | 5 | EVOR is a novel pipeline for **retrieval-augmented code generation (RACG)** that introduces **synchronous evol... |
| vector | vector | 5 | 5 | EVOR is a novel pipeline for **retrieval-augmented code generation (RACG)** that introduces **synchronous evol... |

### HL2 — How does Tree of Reviews structure its retrieval process?

*Category: Local (single-passage)*

| system | mode | correct | faithful | answer (truncated) |
|---|---|---|---|---|
| bm25 | bm25 | 5 | 5 | The **Tree of Reviews (TOR)** framework structures its retrieval process as a **dynamic, tree-based iterative ... |
| graphrag | vector | 5 | 5 | Tree of Reviews (TOR) structures its retrieval process as a **tree-based dynamic framework** where: - The **ro... |
| hybrid | hybrid | 5 | 5 | The Tree of Reviews (TOR) framework structures its retrieval process as a **tree-based dynamic iterative retri... |
| vector | vector | 5 | 5 | The Tree of Reviews (TOR) framework structures its retrieval process as a **tree-based dynamic iterative retri... |

### HL3 — Why does FAIR-RAG use iterative refinement?

*Category: Local (single-passage)*

| system | mode | correct | faithful | answer (truncated) |
|---|---|---|---|---|
| bm25 | bm25 | 5 | 5 | FAIR-RAG uses iterative refinement to **systematically deconstruct complex queries, gather comprehensive evide... |
| graphrag | vector | 4 | 4 | FAIR-RAG uses iterative refinement to **systematically address complex, multi-hop queries** by progressively b... |
| hybrid | hybrid | 5 | 5 | FAIR-RAG uses iterative refinement to **systematically build and validate a comprehensive context** for comple... |
| vector | vector | 5 | 5 | FAIR-RAG uses iterative refinement to handle complex, multi-hop queries that require synthesizing information ... |

### HM1 — Which methods does Dense Hierarchical Retrieval compare itself against?

*Category: Multi-hop (relational)*

| system | mode | correct | faithful | answer (truncated) |
|---|---|---|---|---|
| bm25 | bm25 | 4 | 5 | Dense Hierarchical Retrieval (DHR) compares itself against **Dense Passage Retrieval (DPR)** and **BM25** [1][... |
| graphrag | graph | 5 | 5 | Dense Hierarchical Retrieval compares itself against BM25*, ORQA, DHR w/o rerank, Dense Passage Retriever, Den... |
| hybrid | hybrid | 2 | 1 | Dense Hierarchical Retrieval (DHR) compares itself against the following methods: - BM25 [4] - DPR (Dense Pass... |
| vector | - | - | - |  |

### HM2 — Who are the authors of the EVOR paper on code generation?

*Category: Multi-hop (relational)*

| system | mode | correct | faithful | answer (truncated) |
|---|---|---|---|---|
| bm25 | bm25 | 1 | 1 | The authors of the EVOR paper on code generation are Hongjin Su, Shuyang Jiang, Yuhang Lai, Haoyuan Wu, Boao S... |
| graphrag | graph | 5 | 5 | The authors of the EVOR paper on code generation are Qian Liu, Che Liu, Hongjin Su, Tao Yu, Haoyuan Wu, Shuyan... |
| hybrid | hybrid | 5 | 5 | The authors of the EVOR paper are Hongjin Su, Shuyang Jiang, Yuhang Lai, Haoyuan Wu, Boao Shi, Che Liu, Qian L... |
| vector | vector | 5 | 5 | The authors of the EVOR paper on code generation are Hongjin Su, Shuyang Jiang, Yuhang Lai, Haoyuan Wu, Boao S... |

### HM3 — What benchmarks does the speech-based open-domain question answering work evaluate on?

*Category: Multi-hop (relational)*

| system | mode | correct | faithful | answer (truncated) |
|---|---|---|---|---|
| bm25 | bm25 | 5 | 5 | The speech-based open-domain question answering work evaluates on **Spoken-NQ** and **Spoken-MSMARCO** benchma... |
| graphrag | graph | 5 | 5 | The speech-based open-domain question answering work evaluates on two benchmarks: Spoken-MSMARCO and Spoken-NQ... |
| hybrid | hybrid | - | - | The speech-based open-domain question answering work evaluates on the **Spoken-MSMARCO** and **Spoken-NQ** dat... |
| vector | vector | 2 | 3 | The speech-based open-domain question answering work evaluates on shorter questions where the ASR would have m... |

## Limitations

- The gold set was written by the system's author, which is a real source of bias. Questions were written against the corpus rather than the implementation, but that does not eliminate it.
- The judge is the same model family used to generate answers, which can favour its own outputs.
- 12 questions is small. Differences of a few tenths of a point should not be treated as significant.
- Retrieval metrics are only computed for questions with paper-level relevance labels; aggregate and global questions are graded on the answer.
- **recall@8 measures passage retrieval only.** A graph-mode answer scores 0 there by construction: it answered from the knowledge graph rather than from retrieved passages. Read it alongside correctness, not instead of it.
- Question A1 has no fixed numeric answer. The corpus determines the true count, and using the system's own graph as ground truth would be circular, so A1 grades whether a system *computes* a count or hedges — not whether a particular number is correct.
