from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class EntityType(StrEnum):
    METHOD = "Method"
    DATASET = "Dataset"
    METRIC = "Metric"
    TASK = "Task"
    MODEL = "Model"
    CONCEPT = "PaperConcept"


class ParsedChunk(BaseModel):
    chunk_id: str
    text: str
    section_title: str | None = None
    page_number: int | None = None
    order: int
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Chunk text cannot be blank.")
        return value.strip()


class ParsedSection(BaseModel):
    title: str
    order: int
    chunks: list[ParsedChunk] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    paper_id: str
    source_path: Path
    title: str | None = None
    sections: list[ParsedSection] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def chunks(self) -> list[ParsedChunk]:
        return [chunk for section in self.sections for chunk in section.chunks]


class Evidence(BaseModel):
    text: str = Field(description="Original evidence text copied or tightly paraphrased from the source chunk.")
    chunk_id: str = Field(description="Source chunk id where the evidence appears.")
    page_number: int | None = None

    @field_validator("text", "chunk_id")
    @classmethod
    def required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Evidence fields cannot be blank.")
        return value.strip()


class Entity(BaseModel):
    name: str
    type: EntityType = EntityType.CONCEPT
    canonical_name: str | None = None
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Entity name cannot be blank.")
        return value.strip()


class Objective(BaseModel):
    description: str
    evidence: list[Evidence] = Field(default_factory=list)


class Approach(BaseModel):
    description: str
    method_names: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)


class Result(BaseModel):
    description: str
    dataset_names: list[str] = Field(default_factory=list)
    metric_names: list[str] = Field(default_factory=list)
    task_names: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)


class Constraint(BaseModel):
    description: str
    evidence: list[Evidence] = Field(default_factory=list)


class Claim(BaseModel):
    statement: str
    entity_names: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)


class ChunkExtraction(BaseModel):
    chunk_id: str
    objectives: list[Objective] = Field(default_factory=list)
    approaches: list[Approach] = Field(default_factory=list)
    results: list[Result] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)


class PaperExtraction(BaseModel):
    paper_id: str
    title: str | None = None
    chunks: list[ChunkExtraction] = Field(default_factory=list)


class RetrievalHit(BaseModel):
    id: str
    text: str
    score: float
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryAnswer(BaseModel):
    question: str
    answer: str
    evidence: list[RetrievalHit] = Field(default_factory=list)
