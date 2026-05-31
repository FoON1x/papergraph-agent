from __future__ import annotations

from typing import Any, TYPE_CHECKING, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from paperagent.extraction.prompts import EXTRACTION_HUMAN_PROMPT, EXTRACTION_SYSTEM_PROMPT
from paperagent.schemas import ChunkExtraction

if TYPE_CHECKING:
    from paperagent.extraction.service import ExtractionService


class ChunkExtractionState(TypedDict, total=False):
    """正式版 chunk 抽取工作流使用的状态。"""

    payload: dict[str, Any]
    messages: Any
    structured_result: Any
    raw_text_result: str
    normalized_result: dict[str, Any]
    extraction: ChunkExtraction
    used_fallback: bool
    structured_error: str
    validation_error: str


def build_chunk_extraction_workflow(service: ExtractionService):
    """构建生产使用的 chunk 抽取工作流。

    这条工作流和 service.py 里保留的实验代码分开：
    - 这里负责正式运行路径
    - service.py 里的 _build_chain_by_functional / _build_chain_by_graph 继续作为学习和实验入口
    """

    prompt = ChatPromptTemplate(
        [
            ("system", EXTRACTION_SYSTEM_PROMPT),
            ("human", EXTRACTION_HUMAN_PROMPT),
        ]
    )
    fallback_chain = prompt | service.raw_model | StrOutputParser()

    async def prepare_payload(state: ChunkExtractionState) -> dict[str, Any]:
        """补齐抽取节点依赖的基础字段。"""
        payload = dict(state["payload"])
        payload["page_number"] = payload.get("page_number") or "unknown"
        return {"payload": payload, "used_fallback": False}

    async def format_prompt(state: ChunkExtractionState) -> dict[str, Any]:
        """把标准 payload 格式化成模型可直接消费的消息列表。"""
        return {"messages": await prompt.ainvoke(state["payload"])}

    async def extract_structured(state: ChunkExtractionState) -> dict[str, Any]:
        """优先走 structured output 主路径。"""
        try:
            result = await service.structured_model.ainvoke(state["messages"])
            return {"structured_result": result, "structured_error": ""}
        except Exception as exc:  # noqa: BLE001
            return {"structured_error": f"{type(exc).__name__}: {exc}"}

    def route_after_structured(state: ChunkExtractionState) -> str:
        """structured output 成功则继续校验，失败则走 fallback。"""
        if state.get("structured_error"):
            return "fallback_extract_json"
        return "normalize_structured_result"

    def normalize_structured_result(state: ChunkExtractionState) -> dict[str, Any]:
        """把 structured output 结果统一转成 schema 校验前的标准 dict。"""
        structured_result = state["structured_result"]
        if isinstance(structured_result, ChunkExtraction):
            normalized = structured_result.model_dump()
        elif hasattr(structured_result, "model_dump"):
            normalized = structured_result.model_dump()
        elif isinstance(structured_result, dict):
            normalized = structured_result
        else:
            raise TypeError(
                f"Unsupported structured result type: {type(structured_result).__name__}"
            )

        chunk_id = state["payload"]["chunk_id"]
        return {
            "normalized_result": service._normalize_extraction_payload(normalized, chunk_id),
            "used_fallback": False,
        }

    async def fallback_extract_json(state: ChunkExtractionState) -> dict[str, Any]:
        """structured output 失败时，退回普通文本 JSON 路径。"""
        raw_text = await fallback_chain.ainvoke(state["payload"])
        return {"raw_text_result": raw_text, "used_fallback": True}

    def parse_fallback_json(state: ChunkExtractionState) -> dict[str, Any]:
        """解析 fallback 路径返回的原始 JSON 文本。"""
        chunk_id = state["payload"]["chunk_id"]
        parsed = service._parse_json_object(state["raw_text_result"])
        return {
            "normalized_result": service._normalize_extraction_payload(parsed, chunk_id),
        }

    def validate_result(state: ChunkExtractionState) -> dict[str, Any]:
        """统一做最终 schema 校验。"""
        chunk_id = state["payload"]["chunk_id"]
        try:
            extraction = ChunkExtraction.model_validate(state["normalized_result"])
            if extraction.chunk_id != chunk_id:
                extraction.chunk_id = chunk_id
            return {"extraction": extraction, "validation_error": ""}
        except Exception as exc:  # noqa: BLE001
            return {"validation_error": f"{type(exc).__name__}: {exc}"}

    def route_after_validation(state: ChunkExtractionState) -> str:
        """如果主路径校验失败且尚未 fallback，则再给原始 JSON 一次机会。"""
        if not state.get("validation_error"):
            return "finalize_extraction"
        if state.get("used_fallback"):
            return "finalize_extraction"
        return "fallback_extract_json"

    def finalize_extraction(state: ChunkExtractionState) -> dict[str, Any]:
        """统一收口输出；若最终仍失败，则抛出包含上下文的异常。"""
        extraction = state.get("extraction")
        if extraction is not None:
            return {"extraction": extraction}

        chunk_id = state["payload"].get("chunk_id", "unknown")
        errors = []
        if state.get("structured_error"):
            errors.append(f"structured={state['structured_error']}")
        if state.get("validation_error"):
            errors.append(f"validation={state['validation_error']}")
        raise ValueError(
            f"Failed to build ChunkExtraction for chunk {chunk_id}. " + "; ".join(errors)
        )

    workflow = StateGraph(ChunkExtractionState)
    workflow.add_node("prepare_payload", prepare_payload)
    workflow.add_node("format_prompt", format_prompt)
    workflow.add_node("extract_structured", extract_structured)
    workflow.add_node("normalize_structured_result", normalize_structured_result)
    workflow.add_node("fallback_extract_json", fallback_extract_json)
    workflow.add_node("parse_fallback_json", parse_fallback_json)
    workflow.add_node("validate_result", validate_result)
    workflow.add_node("finalize_extraction", finalize_extraction)

    workflow.add_edge(START, "prepare_payload")
    workflow.add_edge("prepare_payload", "format_prompt")
    workflow.add_edge("format_prompt", "extract_structured")
    workflow.add_conditional_edges(
        "extract_structured",
        route_after_structured,
        {
            "normalize_structured_result": "normalize_structured_result",
            "fallback_extract_json": "fallback_extract_json",
        },
    )
    workflow.add_edge("normalize_structured_result", "validate_result")
    workflow.add_edge("fallback_extract_json", "parse_fallback_json")
    workflow.add_edge("parse_fallback_json", "validate_result")
    workflow.add_conditional_edges(
        "validate_result",
        route_after_validation,
        {
            "fallback_extract_json": "fallback_extract_json",
            "finalize_extraction": "finalize_extraction",
        },
    )
    workflow.add_edge("finalize_extraction", END)
    return workflow.compile()
