"""Schemas the extraction model must fill.

These are handed to the LLM as a JSON schema via structured output, so the
field names and descriptions are prompt surface, not just types -- they are
what tells the model what an "entity" means here. Keep descriptions concrete.

The closed vocabularies below matter: an open-ended relation set produces a
graph where PROPOSES, proposes, introduces and presents are four different
edges, which destroys every multi-hop query.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class EntityType(StrEnum):
    PAPER = "Paper"
    AUTHOR = "Author"
    INSTITUTION = "Institution"
    TASK = "Task"
    METHOD = "Method"
    DATASET = "Dataset"
    METRIC = "Metric"


class RelationType(StrEnum):
    AUTHORED_BY = "AUTHORED_BY"
    AFFILIATED_WITH = "AFFILIATED_WITH"
    ADDRESSES = "ADDRESSES"
    PROPOSES = "PROPOSES"
    USES = "USES"
    EVALUATES_ON = "EVALUATES_ON"
    REPORTS = "REPORTS"
    BUILDS_ON = "BUILDS_ON"
    COMPARES_TO = "COMPARES_TO"


class Entity(BaseModel):
    name: str = Field(
        description="The entity's name exactly as written in the text, e.g. "
        "'Dense Passage Retrieval' or 'Natural Questions'. Do not paraphrase."
    )
    type: EntityType = Field(description="What kind of thing this is.")
    aliases: list[str] = Field(
        default_factory=list,
        description="Other surface forms used in this text, e.g. ['DPR'] for "
        "'Dense Passage Retrieval'. Include acronyms defined in the text.",
    )
    description: str = Field(
        default="",
        description="One short clause describing it, drawn only from this text.",
    )


class Relation(BaseModel):
    subject: str = Field(description="Name of the source entity, matching an Entity name.")
    predicate: RelationType
    object: str = Field(description="Name of the target entity, matching an Entity name.")
    evidence: str = Field(
        description="The exact sentence from the text supporting this relation. "
        "Copy it verbatim -- it is checked against the source."
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="How clearly the text states this. 1.0 = stated explicitly, "
        "0.5 = strongly implied, below 0.5 = do not include it at all.",
    )


class ResultClaim(BaseModel):
    """A reported number. Split out because these carry the quantitative facts
    that make aggregation queries ('best F1 on NQ') answerable."""

    method: str = Field(description="The method or system the number belongs to.")
    dataset: str = Field(description="What it was evaluated on.")
    metric: str = Field(description="Metric name, e.g. 'F1', 'Top-20 accuracy', 'BLEU'.")
    value: float = Field(description="The numeric value as reported.")
    evidence: str = Field(description="The verbatim sentence or table caption stating it.")


class WindowExtraction(BaseModel):
    """Everything pulled from one extraction window.

    The caps are load-bearing, not cosmetic. Uncapped, extraction transcribed
    entire ablation tables -- up to 132 result rows from a single window -- and
    output tokens are what a free-tier rate limit actually measures. Capping
    cut the output roughly in half and removed a class of junk numbers
    (throughput, runtimes) that were being stored as accuracy scores.
    """

    entities: list[Entity] = Field(default_factory=list, max_length=60)
    relations: list[Relation] = Field(default_factory=list, max_length=40)
    results: list[ResultClaim] = Field(default_factory=list, max_length=12)
