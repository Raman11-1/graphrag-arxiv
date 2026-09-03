"""Extraction prompt.

This is a fixed prefix reused across every window, which makes it a good
prompt-cache key. Keep it stable: any byte change invalidates the cache.

The instructions fight three specific failure modes seen in relation extraction
from academic text:

1. **Inventing relations from citations.** A paper mentioning BM25 in related
   work does not "use" BM25. Without an explicit rule the model wires up the
   entire bibliography.
2. **Paraphrasing entity names.** "DPR" and "the dense retriever" become two
   nodes unless names are copied verbatim and aliases are declared.
3. **Attributing baseline numbers to this paper.** A comparison table lists
   other systems' scores; those must not become this paper's REPORTS edges.
"""

from __future__ import annotations

EXTRACTION_SYSTEM = """You extract a knowledge graph from research paper text.

You are given one excerpt from a single paper. Return entities, relations and
reported results found in THIS excerpt only.

ENTITY RULES
- Copy names exactly as written. Do not paraphrase, expand or normalise them.
- Put every other surface form used in the excerpt into `aliases`, including
  acronyms the text defines, e.g. name "Dense Passage Retrieval", aliases ["DPR"].
- Only include entities the excerpt actually discusses. Do not add well-known
  entities from your own knowledge.

RELATION RULES
- `subject` and `object` must exactly match a `name` you returned in `entities`.
- Mentioning prior work is NOT using it. Only assert USES / EVALUATES_ON when
  the excerpt states that THIS paper's work uses or evaluates on that thing.
- Use COMPARES_TO for baselines the paper measures itself against.
- Use BUILDS_ON when the paper's method extends or is based on another method.
- `evidence` must be a sentence copied verbatim from the excerpt. Do not
  summarise it. If you cannot quote a supporting sentence, omit the relation.
- Set `confidence` honestly: 1.0 when stated outright, around 0.6 when implied.
  Omit anything you would score below 0.5.

RESULT RULES
- Only record a number if the excerpt states the metric, the value, and what it
  was measured on.
- Record ONLY accuracy-like evaluation scores: accuracy, F1, EM, recall,
  precision, BLEU, ROUGE, perplexity, nDCG, MRR and similar. Throughput,
  latency, parameter counts, runtimes, dataset sizes and questions-per-second
  are NOT results -- never record them.
- Comparison tables list other systems' scores. Attribute each number to the
  method it belongs to, which is often NOT this paper's method.
- **Record at most 12 results, and choose the headline ones**: the paper's main
  reported numbers and the baselines it directly compares against. Do not
  transcribe entire ablation tables. If the same method, dataset and metric
  appear several times under different configurations, record only the best.

Return nothing rather than guessing. An empty list is a correct answer for an
excerpt that contains no extractable facts."""


def build_user_prompt(*, paper_title: str, sections: list[str], text: str) -> str:
    """Per-window content. Everything volatile lives here, after the cached prefix."""
    where = ", ".join(sections) if sections else "unknown"
    return (
        f"Paper title: {paper_title}\n"
        f"Excerpt from section(s): {where}\n\n"
        f"--- BEGIN EXCERPT ---\n{text}\n--- END EXCERPT ---"
    )
