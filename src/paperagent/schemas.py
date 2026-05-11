from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class EntityType(StrEnum):
    """科研图谱里当前支持的实体类型枚举。"""

    METHOD = "Method"
    DATASET = "Dataset"
    METRIC = "Metric"
    TASK = "Task"
    MODEL = "Model"
    CONCEPT = "PaperConcept"

# ============ 文档结构层对象 ================

class ParsedChunk(BaseModel):
    """解析阶段产出的最小文本单元。

    Chunk 是整个系统的枢纽：
    - 导入时，LLM 按 chunk 抽取语义；
    - 写图时，Chunk 会保存 embedding；
    - 查询时，向量检索首先召回 Chunk。
    """

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
    """解析后的章节对象。

    当前 MVP 里 Section 还比较轻量，但保留这一层有利于后续做更细的章节级分析。
    """

    title: str
    order: int
    chunks: list[ParsedChunk] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    """解析后的整篇论文对象。"""

    paper_id: str
    source_path: Path
    title: str | None = None
    sections: list[ParsedSection] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def chunks(self) -> list[ParsedChunk]:
        # 查询和抽取几乎都以 chunk 为中心，因此这里提供扁平访问方式。
        return [chunk for section in self.sections for chunk in section.chunks]


# ============ 科研语义层对象 ================

class Evidence(BaseModel):
    """支撑某个科研主张的证据片段。"""

    text: str = Field(description="直接从原文 chunk 中复制的证据原文，不要改写")
    chunk_id: str = Field(default="", description="该证据所属的 chunk 的 ID，如果不确定可留空")
    page_number: int | None = Field(default=None, description="证据所在的页码（如果知道的话）")

    @field_validator("text")
    @classmethod
    def required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Evidence text cannot be blank.")
        return value.strip()


class Entity(BaseModel):
    """跨论文复用的实体对象，如方法、数据集、指标、任务等。"""

    name: str = Field(description="实体名称，如 BERT、SQuAD、F1-score")
    type: EntityType = Field(default=EntityType.CONCEPT, description="实体类型：Method / Dataset / Metric / Task / Model / PaperConcept")
    canonical_name: str | None = Field(default=None, description="规范化的实体名称，用于跨论文去重")
    aliases: list[str] = Field(default_factory=list, description="实体的其他别名或缩写")
    description: str | None = Field(default=None, description="对该实体的简要说明，基于原文内容")

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Entity name cannot be blank.")
        return value.strip()


class Objective(BaseModel):
    """论文研究目标。"""

    description: str = Field(description="用原文中明确提到的词描述研究目标，不要编造")
    evidence: list[Evidence] = Field(default_factory=list, description="支撑该目标的原文证据片段，必须从 chunk 中复制")

    @model_validator(mode="before")
    @classmethod
    def coerce_string(cls, data: str | dict) -> dict:
        if isinstance(data, str):
            return {"description": data}
        return data


class Approach(BaseModel):
    """论文采用或提出的方法路线。"""

    description: str = Field(description="方法/技术路线的详细描述，基于原文")
    method_names: list[str] = Field(default_factory=list, description="方法的具体名称列表，如 ['RAG', 'BM25']")
    evidence: list[Evidence] = Field(default_factory=list, description="支撑该方法的原文证据片段")

    @model_validator(mode="before")
    @classmethod
    def coerce_string(cls, data: str | dict) -> dict:
        if isinstance(data, str):
            return {"description": data}
        return data


class Result(BaseModel):
    """实验结果或结论性发现。"""

    description: str = Field(description="实验结果或发现的描述，直接基于原文")
    dataset_names: list[str] = Field(default_factory=list, description="实验使用的数据集名称，如 ['SQuAD', 'ImageNet']")
    metric_names: list[str] = Field(default_factory=list, description="使用的评估指标名称，如 ['F1', 'Accuracy']")
    task_names: list[str] = Field(default_factory=list, description="实验涉及的任务名称，如 ['问答', '分类']")
    evidence: list[Evidence] = Field(default_factory=list, description="支撑该结果的原文证据片段")

    @model_validator(mode="before")
    @classmethod
    def coerce_string(cls, data: str | dict) -> dict:
        if isinstance(data, str):
            return {"description": data}
        return data


class Constraint(BaseModel):
    """论文中的限制、边界条件或性能瓶颈。"""

    description: str = Field(description="限制或瓶颈的详细描述，基于原文")
    evidence: list[Evidence] = Field(default_factory=list, description="支撑该约束的原文证据片段")

    @model_validator(mode="before")
    @classmethod
    def coerce_string(cls, data: str | dict) -> dict:
        if isinstance(data, str):
            return {"description": data}
        return data


class Claim(BaseModel):
    """可被证据支撑的主张。"""

    statement: str = Field(description="从原文提取的主张/断言，必须是文中有明确支撑的")
    entity_names: list[str] = Field(default_factory=list, description="该主张涉及的实体名称列表")
    evidence: list[Evidence] = Field(default_factory=list, description="支撑该主张的原文证据片段")

    @model_validator(mode="before")
    @classmethod
    def coerce_string(cls, data: str | dict) -> dict:
        if isinstance(data, str):
            return {"statement": data}
        return data


class ChunkExtraction(BaseModel):
    """单个 Chunk 的结构化抽取结果。"""

    chunk_id: str
    objectives: list[Objective] = Field(default_factory=list)
    approaches: list[Approach] = Field(default_factory=list)
    results: list[Result] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)


class PaperExtraction(BaseModel):
    """整篇论文的抽取结果聚合。"""

    paper_id: str
    title: str | None = None
    chunks: list[ChunkExtraction] = Field(default_factory=list)


class RetrievalHit(BaseModel):
    """一次检索命中的标准化表示。"""

    id: str
    text: str
    score: float
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryAnswer(BaseModel):
    """查询接口最终返回给上层的对象。"""

    question: str
    answer: str
    evidence: list[RetrievalHit] = Field(default_factory=list)
