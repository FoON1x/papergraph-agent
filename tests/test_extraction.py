"""测试 ExtractionService 的解析和规范化逻辑。

所有测试不依赖外部 API，只验证容错和字段映射的正确性。
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from paperagent.extraction.service import ExtractionService
from paperagent.schemas import ChunkExtraction, Evidence, Objective


@pytest.fixture
def mock_chat_provider() -> MagicMock:
    provider = MagicMock()
    provider.get_chat_model.return_value = MagicMock()
    return provider


@pytest.fixture
def service(mock_chat_provider: MagicMock) -> ExtractionService:
    with patch(
        "paperagent.extraction.service.get_settings", return_value=MagicMock()
    ), patch(
        "paperagent.extraction.service.get_chat_provider",
        return_value=mock_chat_provider,
    ):
        return ExtractionService(chat_provider=mock_chat_provider)


# ==================== _parse_json_object ====================


def test_parse_json_plain_object(service: ExtractionService) -> None:
    result = service._parse_json_object('{"chunk_id": "c1", "objectives": []}')
    assert result == {"chunk_id": "c1", "objectives": []}


def test_parse_json_with_markdown_fence(service: ExtractionService) -> None:
    raw = '```json\n{"chunk_id": "c1", "objectives": []}\n```'
    result = service._parse_json_object(raw)
    assert result == {"chunk_id": "c1", "objectives": []}


def test_parse_json_with_untyped_fence(service: ExtractionService) -> None:
    raw = '```\n{"chunk_id": "c1"}\n```'
    result = service._parse_json_object(raw)
    assert result == {"chunk_id": "c1"}


# ==================== _normalize_entity_type ====================


def test_entity_type_standard_values(service: ExtractionService) -> None:
    assert service._normalize_entity_type("Method") == "Method"
    assert service._normalize_entity_type("Dataset") == "Dataset"
    assert service._normalize_entity_type("Metric") == "Metric"
    assert service._normalize_entity_type("Task") == "Task"
    assert service._normalize_entity_type("Model") == "Model"
    assert service._normalize_entity_type("PaperConcept") == "PaperConcept"


def test_entity_type_case_insensitive_alias(service: ExtractionService) -> None:
    assert service._normalize_entity_type("method") == "Method"
    assert service._normalize_entity_type("DATASET") == "Dataset"


def test_entity_type_filters_citation_reference_author(service: ExtractionService) -> None:
    assert service._normalize_entity_type("citation") is None
    assert service._normalize_entity_type("reference") is None
    assert service._normalize_entity_type("author") is None


def test_entity_type_unknown_falls_back_to_concept(service: ExtractionService) -> None:
    assert service._normalize_entity_type("UnknownType") == "PaperConcept"


# ==================== _normalize_entity ====================


def test_normalize_entity_from_string(service: ExtractionService) -> None:
    assert service._normalize_entity("GraphRAG") == {
        "name": "GraphRAG",
        "type": "PaperConcept",
    }


def test_normalize_entity_blank_string(service: ExtractionService) -> None:
    assert service._normalize_entity("   ") is None


def test_normalize_entity_dict_with_alias_fields(service: ExtractionService) -> None:
    result = service._normalize_entity({"entity": "ResNet", "entity_type": "model"})
    assert result is not None
    assert result["name"] == "ResNet"
    assert result["type"] == "Model"


def test_normalize_entity_filters_citation_type(service: ExtractionService) -> None:
    assert service._normalize_entity({"name": "Paper", "type": "citation"}) is None


# ==================== _normalize_objective ====================


def test_normalize_objective_from_string(service: ExtractionService) -> None:
    result = service._normalize_objective("Improve accuracy", "c1")
    assert result["description"] == "Improve accuracy"
    assert result["evidence"] == []


def test_normalize_objective_evidence_from_string(service: ExtractionService) -> None:
    result = service._normalize_objective(
        {"objective": "x", "evidence": ["supports this"]}, "c1"
    )
    assert len(result["evidence"]) == 1
    assert result["evidence"][0]["text"] == "supports this"


# ==================== _normalize_approach ====================


def test_normalize_approach_from_string(service: ExtractionService) -> None:
    result = service._normalize_approach("Use RAG", "c1")
    assert result["description"] == "Use RAG"
    assert result["method_names"] == []
    assert result["evidence"] == []


def test_normalize_approach_method_names_from_string(service: ExtractionService) -> None:
    result = service._normalize_approach(
        {"approach": "x", "methods": "BM25"}, "c1"
    )
    assert result["method_names"] == ["BM25"]


# ==================== _normalize_result ====================


def test_normalize_result_from_string(service: ExtractionService) -> None:
    result = service._normalize_result("Accuracy +5%", "c1")
    assert result["description"] == "Accuracy +5%"
    for key in ["dataset_names", "metric_names", "task_names"]:
        assert result[key] == []


def test_normalize_result_alias_fields(service: ExtractionService) -> None:
    result = service._normalize_result(
        {"result": "x", "datasets": "SQuAD", "metrics": "F1", "tasks": "QA"}, "c1"
    )
    assert result["dataset_names"] == ["SQuAD"]
    assert result["metric_names"] == ["F1"]
    assert result["task_names"] == ["QA"]


# ==================== _normalize_constraint ====================


def test_normalize_constraint_from_string(service: ExtractionService) -> None:
    result = service._normalize_constraint("Limited data", "c1")
    assert result["description"] == "Limited data"


# ==================== _normalize_claim ====================


def test_normalize_claim_entities_as_string(service: ExtractionService) -> None:
    result = service._normalize_claim(
        {"statement": "x", "entities": "GraphRAG"}, "c1"
    )
    assert result["entity_names"] == ["GraphRAG"]


# ==================== _normalize_evidence_list ====================


def test_normalize_evidence_from_string(service: ExtractionService) -> None:
    result = service._normalize_evidence_list("evidence text", "c1")
    assert result == [{"text": "evidence text", "chunk_id": "c1"}]


def test_normalize_evidence_from_list_of_strings(service: ExtractionService) -> None:
    result = service._normalize_evidence_list(["ev1", "ev2"], "c1")
    assert len(result) == 2
    assert result[0]["text"] == "ev1"


def test_normalize_evidence_from_single_dict(service: ExtractionService) -> None:
    result = service._normalize_evidence_list(
        {"text": "ev", "chunk_id": "c2", "page_number": 3}, "c1"
    )
    assert result == [{"text": "ev", "chunk_id": "c2", "page_number": 3}]


def test_normalize_evidence_alias_field(service: ExtractionService) -> None:
    result = service._normalize_evidence_list([{"evidence": "text"}], "c1")
    assert result[0]["text"] == "text"


# ==================== _coerce_extraction 整合 ====================


def test_coerce_extraction_standard_format(service: ExtractionService) -> None:
    raw = (
        '{"chunk_id": "c1", "objectives": ["Improve recall"], '
        '"approaches": [], "results": [], "constraints": [], '
        '"claims": [], "entities": []}'
    )
    result = service._coerce_extraction(raw, "c1")
    assert result.chunk_id == "c1"
    assert len(result.objectives) == 1
    assert result.objectives[0].description == "Improve recall"


def test_coerce_extraction_fixes_wrong_chunk_id(service: ExtractionService) -> None:
    raw = (
        '{"chunk_id": "wrong", "objectives": [], "approaches": [], '
        '"results": [], "constraints": [], "claims": [], "entities": []}'
    )
    result = service._coerce_extraction(raw, "correct")
    assert result.chunk_id == "correct"


def test_coerce_extraction_statements_format(service: ExtractionService) -> None:
    raw = (
        '{"source_chunk_id": "c1", "statements": ['
        '{"claim": "GraphRAG is effective", "evidence": "Fig 2 shows", '
        '"entities": ["GraphRAG"]}'
        "]}"
    )
    result = service._coerce_extraction(raw, "c1")
    assert len(result.claims) == 1
    assert result.claims[0].statement == "GraphRAG is effective"
    assert result.claims[0].evidence[0].text == "Fig 2 shows"
    assert result.claims[0].entity_names == ["GraphRAG"]
    assert len(result.entities) == 1
    assert result.entities[0].name == "GraphRAG"


# ==================== _build_chain_by_functional 集成测试 ====================


@pytest.fixture
def expected_extraction() -> ChunkExtraction:
    """预制的假抽取结果，作为模型的 fake 返回。"""
    return ChunkExtraction(
        chunk_id="c1",
        objectives=[Objective(description="Improve recall")],
    )


@pytest.fixture
def fake_model(expected_extraction: ChunkExtraction) -> MagicMock:
    """模拟 with_structured_output 返回的模型，ainvoke 是 async 方法。"""
    model = MagicMock()
    model.ainvoke = AsyncMock(return_value=expected_extraction)
    return model


@pytest.fixture
def service_with_fake_model(
    mock_chat_provider: MagicMock, fake_model: MagicMock
) -> ExtractionService:
    """创建 service 后，把 structured_model 替换为假模型。"""
    with patch(
        "paperagent.extraction.service.get_settings", return_value=MagicMock()
    ), patch(
        "paperagent.extraction.service.get_chat_provider",
        return_value=mock_chat_provider,
    ):
        svc = ExtractionService(chat_provider=mock_chat_provider)
        svc.structured_model = fake_model
        return svc


@pytest.mark.asyncio
async def test_functional_chain_invoke_returns_chunk_extraction(
    service_with_fake_model: ExtractionService,
    expected_extraction: ChunkExtraction,
    fake_model: MagicMock,
) -> None:
    """functional chain 调用：payload → format → extract → ChunkExtraction。"""
    chain = service_with_fake_model._build_chain_by_functional()

    result = await chain.ainvoke({
        "paper_id": "p1",
        "chunk_id": "c1",
        "page_number": 1,
        "chunk_text": "Test content",
    })

    assert result == expected_extraction
    fake_model.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_functional_chain_ainvoke_is_async(
    service_with_fake_model: ExtractionService,
    fake_model: MagicMock,
) -> None:
    """functional chain 异步调用：ainvoke 应正常返回。"""
    chain = service_with_fake_model._build_chain_by_functional()

    result = await chain.ainvoke({
        "paper_id": "p1",
        "chunk_id": "c1",
        "page_number": 1,
        "chunk_text": "Test content",
    })

    assert isinstance(result, ChunkExtraction)
    fake_model.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_functional_chain_passes_formatted_messages_to_model(
    service_with_fake_model: ExtractionService,
    fake_model: MagicMock,
) -> None:
    """验证传给模型的消息包含了 chunk_text 内容。"""
    chain = service_with_fake_model._build_chain_by_functional()

    await chain.ainvoke({
        "paper_id": "p1",
        "chunk_id": "c1",
        "page_number": 3,
        "chunk_text": "GraphRAG improves retrieval performance.",
    })

    call_args = fake_model.ainvoke.call_args[0][0]
    messages_text = str(call_args)
    assert "GraphRAG improves retrieval performance" in messages_text
    assert "c1" in messages_text


# ==================== _build_chain_by_graph 集成测试 ====================


@pytest.mark.asyncio
async def test_graph_chain_ainvoke_returns_state(
    service_with_fake_model: ExtractionService,
    expected_extraction: ChunkExtraction,
    fake_model: MagicMock,
) -> None:
    """graph chain 异步调用：返回 ExtractionState dict，内含 result。"""
    chain = service_with_fake_model._build_chain_by_graph()

    state = await chain.ainvoke({
        "payload": {
            "paper_id": "p1",
            "chunk_id": "c1",
            "page_number": 1,
            "chunk_text": "Test content",
        }
    })

    assert state["result"] == expected_extraction
    fake_model.ainvoke.assert_called_once()
