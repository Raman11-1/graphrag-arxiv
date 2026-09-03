# GraphRAG evaluation

12 gold questions x 4 systems = 48 runs. Judged by `mistral-medium-latest`; retrieval metrics are deterministic.

## Overall

| system | correctness | faithfulness | completeness | recall@8 | tokens | errors |
|---|---|---|---|---|---|---|
| bm25 | 2.83 | 2.83 | 2.75 | 0.146 | 8,031 | 0 |
| graphrag | 3.25 | 3.67 | 3.0 | 0.115 | 5,466 | 0 |
| hybrid | 3.17 | 3.17 | 3.08 | 0.188 | 8,034 | 0 |
| vector | 3.33 | 3.17 | 3.25 | 0.177 | 8,119 | 0 |

Scores are 1-5. Higher is better throughout.

## By question category

This breakdown is the finding. The claim under test is not that GraphRAG is uniformly better, but that it wins on questions whose answers are structured, and ties where plain retrieval already works.

### Local (single-passage)

*Expectation: Baselines should be competitive here -- this is what plain RAG is for.*

| system | correctness | faithfulness | completeness | tokens |
|---|---|---|---|---|
| bm25 | 3.67 | 4.33 | 3.33 | 7,842 |
| graphrag | 3.67 | 3.67 | 3.0 | 8,557 |
| hybrid | 3.67 | 4.33 | 3.33 | 8,162 |
| vector | 4.33 | 4.33 | 4.0 | 8,643 |

**Best: vector**

### Multi-hop (relational)

*Expectation: GraphRAG should win: the answer is a set of connected entities.*

| system | correctness | faithfulness | completeness | tokens |
|---|---|---|---|---|
| bm25 | 4.0 | 3.67 | 4.0 | 8,146 |
| graphrag | 3.67 | 5.0 | 3.67 | 1,930 |
| hybrid | 3.67 | 3.67 | 3.67 | 7,889 |
| vector | 3.67 | 3.67 | 3.67 | 7,856 |

**Best: bm25**

### Aggregate (counting)

*Expectation: GraphRAG should win decisively: top-k retrieval cannot count.*

| system | correctness | faithfulness | completeness | tokens |
|---|---|---|---|---|
| bm25 | 2.33 | 2.33 | 2.33 | 7,969 |
| graphrag | 3.67 | 3.67 | 3.67 | 4,466 |
| hybrid | 3.67 | 3.67 | 3.67 | 8,370 |
| vector | 2.33 | 2.33 | 2.33 | 8,361 |

**Best: graphrag / hybrid**

### Global (corpus-wide)

*Expectation: GraphRAG should win via community summaries; baselines see 8 chunks.*

| system | correctness | faithfulness | completeness | tokens |
|---|---|---|---|---|
| bm25 | 1.33 | 1.0 | 1.33 | 8,167 |
| graphrag | 2.0 | 2.33 | 1.67 | 6,912 |
| hybrid | 1.67 | 1.0 | 1.67 | 7,716 |
| vector | 3.0 | 2.33 | 3.0 | 7,615 |

**Best: vector**

## Cost and latency

Token counts cover answering only. Grading is evaluation overhead and is excluded — a system producing longer answers is more expensive to judge, and charging that back would penalise it twice.

**Do not read the latency column as system performance.** These runs were made against a rate-limited free tier, so a figure largely reflects how much 429 backoff a run happened to absorb, not how fast the system is. Token counts are unaffected and are the meaningful cost signal here.

| system | mean tokens/question | mean latency (s) | LLM calls/question |
|---|---|---|---|
| bm25 | 8,031 | 127.06 | 1.0 |
| graphrag | 5,466 | 42.2 | 2.8 |
| hybrid | 8,034 | 28.87 | 1.0 |
| vector | 8,119 | 39.33 | 1.0 |

## How GraphRAG routed each category

| category | routes chosen |
|---|---|
| Local (single-passage) | vector x3 |
| Multi-hop (relational) | graph x3 |
| Aggregate (counting) | graph x3 |
| Global (corpus-wide) | global x1, graph x2 |

## Per-question results

Included so any surprising number can be traced to the answer behind it.

### A1 — How many distinct methods in this corpus are evaluated on Natural Questions?

*Category: Aggregate (counting)*

| system | mode | correct | faithful | answer (truncated) |
|---|---|---|---|---|
| bm25 | bm25 | 5 | 5 | The sources do not contain the answer to the question "How many distinct methods in this corpus are evaluated ... |
| graphrag | graph | 5 | 5 | 9 [1] |
| hybrid | hybrid | 5 | 5 | The sources do not contain the answer to the question "How many distinct methods in this corpus are evaluated ... |
| vector | vector | 5 | 5 | The corpus evaluates the following distinct methods on Natural Questions: 1. BM25 [2] 2. DPR (Dense Passage Re... |

### A2 — Which benchmark dataset is used by the most methods in this corpus?

*Category: Aggregate (counting)*

| system | mode | correct | faithful | answer (truncated) |
|---|---|---|---|---|
| bm25 | bm25 | 1 | 1 | The benchmark dataset used by the most methods in this corpus is **Wikipedia Corpus** [2][4][5][8]. |
| graphrag | graph | 5 | 5 | The benchmark dataset used by the most methods in this corpus is the **Natural Questions dataset** [1]. |
| hybrid | hybrid | 5 | 5 | The benchmark dataset used by the most methods in this corpus is **NQ (Natural Questions)**, which appears in ... |
| vector | vector | 1 | 1 | The benchmark dataset used by the most methods in this corpus is **Natural Questions (NQ)**, as it is evaluate... |

### A3 — What is the highest Top-20 retrieval accuracy reported on Natural Questions?

*Category: Aggregate (counting)*

| system | mode | correct | faithful | answer (truncated) |
|---|---|---|---|---|
| bm25 | bm25 | 1 | 1 | The highest Top-20 retrieval accuracy reported on Natural Questions is **86.0%** from the multi-dataset DPR mo... |
| graphrag | graph | 1 | 1 | The highest Top-20 retrieval accuracy reported on Natural Questions is **94.1%** [1]. |
| hybrid | hybrid | 1 | 1 | The highest Top-20 retrieval accuracy reported on Natural Questions is **94.1%** from the multi-dataset DPR + ... |
| vector | vector | 1 | 1 | The highest Top-20 retrieval accuracy reported on Natural Questions is **90.64%** from the 2-iter DHR model [4... |

### G1 — What are the main research themes across these papers?

*Category: Global (corpus-wide)*

| system | mode | correct | faithful | answer (truncated) |
|---|---|---|---|---|
| bm25 | bm25 | 2 | 1 | The main research themes across these papers are: 1. **Evaluation of LLMs in Quranic Studies**: - Use of Retri... |
| graphrag | global | 3 | 5 | The main research themes across these papers are: 1. Advanced Retrieval-Augmented Generation Techniques [1] 2.... |
| hybrid | hybrid | 2 | 1 | **1. Mission Deconstruction:** - **Main Goal:** Identify the main research themes across the provided papers. ... |
| vector | vector | 2 | 1 | The main research themes across these papers are: - **Cognitive augmentation and AI support in information pro... |

### G2 — What benchmark datasets are used across this collection of papers?

*Category: Global (corpus-wide)*

| system | mode | correct | faithful | answer (truncated) |
|---|---|---|---|---|
| bm25 | bm25 | 1 | 1 | Here are the benchmark datasets mentioned across the provided sources: - **EVOR-BENCH** (with four datasets: S... |
| graphrag | graph | 1 | 1 | The benchmark datasets used across the provided papers are: - **EVOR-BENCH** (from [2][3]): Includes **Scipy-M... |
| hybrid | hybrid | 1 | 1 | The benchmark datasets used across the provided papers are: - **EVOR-BENCH** (from [2][3]): Includes **Scipy-M... |
| vector | vector | 5 | 5 | The benchmark datasets used across this collection of papers are: - **EVOR (paper [1])**: Scipy-M, Tensorflow-... |

### G3 — How do the retrieval approaches in this corpus relate to each other?

*Category: Global (corpus-wide)*

| system | mode | correct | faithful | answer (truncated) |
|---|---|---|---|---|
| bm25 | bm25 | 1 | 1 | The sources describe several retrieval approaches and their relationships as follows: 1. **Standard RAG vs. It... |
| graphrag | graph | 2 | 1 | The retrieval approaches in the corpus can be categorized and related as follows: 1. **Sparse Retrieval vs. De... |
| hybrid | hybrid | 2 | 1 | The retrieval approaches in the corpus can be categorized and related as follows: 1. **Dual Encoder Models (De... |
| vector | vector | 2 | 1 | The sources describe several retrieval approaches for open-domain QA, which can be categorized as follows: 1. ... |

### L1 — What is dense passage retrieval and how does it encode questions and passages?

*Category: Local (single-passage)*

| system | mode | correct | faithful | answer (truncated) |
|---|---|---|---|---|
| bm25 | bm25 | 4 | 5 | Dense Passage Retrieval (DPR) is a dual-encoder framework that encodes questions and passages separately into ... |
| graphrag | vector | 5 | 5 | Dense Passage Retrieval (DPR) is a dual-encoder framework that encodes questions and passages separately into ... |
| hybrid | hybrid | 5 | 5 | Dense passage retrieval (DPR) is a dual-encoder framework that encodes questions and passages separately into ... |
| vector | vector | 5 | 5 | Dense Passage Retrieval (DPR) is a method for open-domain QA that uses dense vector representations to index a... |

### L2 — Why are in-batch negatives used when training dense retrievers?

*Category: Local (single-passage)*

| system | mode | correct | faithful | answer (truncated) |
|---|---|---|---|---|
| bm25 | bm25 | 4 | 5 | In-batch negatives are used when training dense retrievers because they allow the model to reuse the negative ... |
| graphrag | vector | 3 | 1 | In-batch negatives are used when training dense retrievers because they are an **effective and memory-efficien... |
| hybrid | hybrid | 3 | 3 | In-batch negatives are used when training dense retrievers because they are an effective and memory-efficient ... |
| vector | vector | 5 | 5 | In-batch negatives are used when training dense retrievers because they allow the model to treat all positive ... |

### L3 — What problem does retrieval-augmented generation address?

*Category: Local (single-passage)*

| system | mode | correct | faithful | answer (truncated) |
|---|---|---|---|---|
| bm25 | bm25 | 3 | 3 | Retrieval-Augmented Generation (RAG) addresses the problem of hallucinations in large language models (LLMs), ... |
| graphrag | vector | 3 | 5 | Retrieval-augmented generation (RAG) addresses the hallucination problem of Large Language Models (LLMs) by re... |
| hybrid | hybrid | 3 | 5 | Retrieval-augmented generation (RAG) addresses the hallucination problem of Large Language Models (LLMs) by re... |
| vector | vector | 3 | 3 | Retrieval-augmented generation (RAG) addresses the problem of enhancing generative models by incorporating ext... |

### M1 — Which datasets does DPR evaluate on?

*Category: Multi-hop (relational)*

| system | mode | correct | faithful | answer (truncated) |
|---|---|---|---|---|
| bm25 | bm25 | 5 | 5 | DPR evaluates on the following datasets: Natural Questions (NQ), TriviaQA, Web Questions (WQ), CuratedTREC (TR... |
| graphrag | graph | 3 | 5 | DPR evaluates on the following datasets: 1. SQuAD v1.1 2. Natural Questions dataset 3. TriviaQA 4. CuratedTREC... |
| hybrid | hybrid | 5 | 5 | DPR evaluates on Natural Questions (NQ), TriviaQA, Web Questions (WQ), CuratedTREC (TREC), and SQuAD [2][3]. |
| vector | vector | 5 | 5 | DPR evaluates on the following datasets: Natural Questions (NQ), TriviaQA, Web Questions (WQ), TREC, and SQuAD... |

### M2 — Who are the authors of the dense passage retrieval paper?

*Category: Multi-hop (relational)*

| system | mode | correct | faithful | answer (truncated) |
|---|---|---|---|---|
| bm25 | bm25 | 5 | 5 | The authors of the Dense Passage Retrieval (DPR) paper are Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick... |
| graphrag | graph | 5 | 5 | The authors of the "Dense Passage Retrieval" paper are: 1. Ledell Wu 2. Danqi Chen 3. Barlas Oğuz 4. Vladimir ... |
| hybrid | hybrid | 4 | 5 | The authors of the Dense Passage Retrieval (DPR) paper are Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick... |
| vector | vector | 5 | 5 | The authors of the Dense Passage Retrieval paper are Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis... |

### M3 — Which methods does DPR compare itself against?

*Category: Multi-hop (relational)*

| system | mode | correct | faithful | answer (truncated) |
|---|---|---|---|---|
| bm25 | bm25 | 2 | 1 | DPR compares itself against the following methods: - Single BM25+BERT [2] - Single ORQA [2] - Single HardEM [2... |
| graphrag | graph | 3 | 5 | DPR compares itself against BM25+DPR, BM25*, and ORQA [1]. |
| hybrid | hybrid | 2 | 1 | DPR compares itself against the following methods: - Single BM25+BERT [1] - Single ORQA [1] - Single HardEM [1... |
| vector | vector | 1 | 1 | DPR compares itself against the following methods: - BM25+BERT [5] - ORQA [5] - HardEM [5] - GraphRetriever [5... |

## Limitations

- The gold set was written by the system's author, which is a real source of bias. Questions were written against the corpus rather than the implementation, but that does not eliminate it.
- The judge is the same model family used to generate answers, which can favour its own outputs.
- 12 questions is small. Differences of a few tenths of a point should not be treated as significant.
- Retrieval metrics are only computed for questions with paper-level relevance labels; aggregate and global questions are graded on the answer.
- **recall@8 measures passage retrieval only.** A graph-mode answer scores 0 there by construction: it answered from the knowledge graph rather than from retrieved passages. Read it alongside correctness, not instead of it.
- Question A1 has no fixed numeric answer. The corpus determines the true count, and using the system's own graph as ground truth would be circular, so A1 grades whether a system *computes* a count or hedges — not whether a particular number is correct.
