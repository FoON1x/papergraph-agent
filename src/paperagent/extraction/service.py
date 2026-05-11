import asyncio
import json

from typing import Any, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.func import entrypoint, task
from langgraph.graph import StateGraph, START, END

from paperagent.config import Settings, get_settings
from paperagent.extraction.prompts import EXTRACTION_HUMAN_PROMPT, EXTRACTION_SYSTEM_PROMPT
from paperagent.providers import ChatProvider, get_chat_provider
from paperagent.schemas import ChunkExtraction, Claim, Entity, EntityType, Evidence, ParsedDocument, PaperExtraction


class ExtractionState(TypedDict, total=False):
    payload: dict
    messages: Any
    result: ChunkExtraction


class ExtractionService:
    """把 ParsedDocument 转成结构化科研语义对象。"""

    def __init__(
        self,
        chat_provider: ChatProvider | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.chat_provider = chat_provider or get_chat_provider(self.settings)
        self.structured_model = (
            self.chat_provider.get_chat_model()
            .with_structured_output(ChunkExtraction, method="json_mode")
        )
        self.chain = self._build_chain_by_functional()

    async def extract_document(self, document: ParsedDocument) -> PaperExtraction:
        """并发抽取整篇论文的所有 Chunk。"""
        inputs: list[dict] = [
            {
                "paper_id": document.paper_id,
                "chunk_id": chunk.chunk_id,
                "page_number": chunk.page_number or "unknown",
                "chunk_text": chunk.text,
            }
            for chunk in document.chunks
        ]
        semaphore = asyncio.Semaphore(self.settings.max_concurrency)

        async def extract_one(payload: dict) -> ChunkExtraction:
            async with semaphore:
                # 每个 chunk 独立抽取；一旦模型输出不稳定，后续会进入 _coerce_extraction 做兜底。
                return await self.chain.ainvoke(payload)
                # raw_output = await self.chain.ainvoke(payload)
                # return self._coerce_extraction(raw_output, payload["chunk_id"])

        extractions = await asyncio.gather(*(extract_one(payload) for payload in inputs))
        for chunk, extraction in zip(document.chunks, extractions, strict=False):
            if extraction.chunk_id != chunk.chunk_id:
                extraction.chunk_id = chunk.chunk_id
        # 返回所有抽取出的chunk的集合
        return PaperExtraction(paper_id=document.paper_id, title=document.title, chunks=list(extractions))

    def extract_chunk(
        self,
        paper_id: str,
        chunk_id: str,
        chunk_text: str,
        page_number: int | None = None,
    ) -> ChunkExtraction:
        """同步抽取单个 Chunk。

        这个函数更适合调试或单元测试；正常导入流程会走 extract_document。
        """
        return asyncio.run(
            self.chain.ainvoke(
                {
                    "paper_id": paper_id,
                    "chunk_id": chunk_id,
                    "page_number": page_number or "unknown",
                    "chunk_text": chunk_text,
                }
            )
        )

    
    
    def _build_chain_by_functional(self):
        """使用langgraph functional api构建知识抽取流水线"""

        prompt = ChatPromptTemplate(
            [
                ("system", EXTRACTION_SYSTEM_PROMPT),
                ("human", EXTRACTION_HUMAN_PROMPT),
            ]
        )

        @task
        async def _format_prompt(payload: dict):
            """将payload格式化为消息列表"""
            return await prompt.ainvoke(payload)

        @task
        async def _extract_knowledge(messages):
            """调用LLM抽取结构化科研知识"""
            return await self.structured_model.ainvoke(messages)
    
        @entrypoint()
        async def extraction_workflow(payload: dict) -> ChunkExtraction:
            messages = await _format_prompt(payload)
            return await _extract_knowledge(messages)
        
        return extraction_workflow
    

    def _build_chain_by_graph(self):
        """使用langgraph graph api构建知识抽取流水线"""
        prompt = ChatPromptTemplate(
            [
                ("system", EXTRACTION_SYSTEM_PROMPT),
                ("human", EXTRACTION_HUMAN_PROMPT),
            ]
        )

        async def format_prompt(state: ExtractionState) -> dict:
            """将payload格式化为消息列表"""
            return {"messages": await prompt.ainvoke(state['payload'])}
        
        async def extract_knowledge(state: ExtractionState) -> dict:
            """调用LLM抽取结构化科研知识"""
            return {"result": await self.structured_model.ainvoke(state["messages"])}
        
        workflow = StateGraph(ExtractionState)
        workflow.add_node("format", format_prompt)
        workflow.add_node("extract", extract_knowledge)

        workflow.add_edge(START, "format")
        workflow.add_edge("format", "extract")
        workflow.add_edge("extract", END)

        return workflow.compile()


    def _build_chain(self):
        """构建 LangChain Runnable 抽取链。"""
        # 这里刻意使用 LangChain Runnable 风格：Prompt -> Model -> Parser。
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", EXTRACTION_SYSTEM_PROMPT),
                ("human", EXTRACTION_HUMAN_PROMPT),
            ]
        )
        return prompt | self.chat_provider.get_chat_model() | StrOutputParser()

    def _coerce_extraction(self, raw_output: str, chunk_id: str) -> ChunkExtraction:
        """把模型原始输出清洗成 ChunkExtraction。"""
        # 模型输出先转成普通 dict，再统一做字段规范化，最后交给 Pydantic 校验。
        payload = self._parse_json_object(raw_output)
        normalized = self._normalize_extraction_payload(payload, chunk_id)
        extraction = ChunkExtraction.model_validate(normalized)
        if extraction.chunk_id != chunk_id:
            extraction.chunk_id = chunk_id
        return extraction

    def _parse_json_object(self, raw_output: str) -> dict:
        """把模型返回的 JSON 文本解析成 Python 字典。"""
        text = raw_output.strip()
        if text.startswith("```"):
            # 兼容模型偶尔返回 ```json ... ``` 代码块的情况。
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return json.loads(text)

    def _normalize_extraction_payload(self, payload: dict, chunk_id: str) -> dict:
        """兼容多种模型输出形态，并统一转换成标准抽取结构。"""
        if "chunk_id" in payload:
            payload.setdefault("objectives", [])
            payload.setdefault("approaches", [])
            payload.setdefault("results", [])
            payload.setdefault("constraints", [])
            payload.setdefault("claims", [])
            payload.setdefault("entities", [])
            normalized_entities = []
            for item in payload.get("entities", []):
                normalized = self._normalize_entity(item)
                if normalized is not None:
                    normalized_entities.append(normalized)
            return {
                "chunk_id": payload.get("chunk_id") or chunk_id,
                "objectives": [self._normalize_objective(item, chunk_id) for item in payload.get("objectives", [])],
                "approaches": [self._normalize_approach(item, chunk_id) for item in payload.get("approaches", [])],
                "results": [self._normalize_result(item, chunk_id) for item in payload.get("results", [])],
                "constraints": [self._normalize_constraint(item, chunk_id) for item in payload.get("constraints", [])],
                "claims": [self._normalize_claim(item, chunk_id) for item in payload.get("claims", [])],
                "entities": normalized_entities,
            }

        # 兼容另一类常见输出：模型用 statements/source_chunk_id 包了一个近似结构。
        statements = payload.get("statements", [])
        claims: list[dict] = []
        entities: dict[str, dict] = {}

        for statement in statements:
            claim_text = statement.get("claim") or statement.get("statement") or statement.get("description")
            if not claim_text:
                continue
            source_chunk_id = statement.get("source_chunk_id") or chunk_id
            evidence_text = statement.get("evidence") or statement.get("source_text") or claim_text
            entity_names = statement.get("entity_names") or statement.get("entities") or []
            if isinstance(entity_names, str):
                entity_names = [entity_names]

            claims.append(
                Claim(
                    statement=claim_text,
                    entity_names=entity_names,
                    evidence=[
                        Evidence(
                            text=evidence_text,
                            chunk_id=source_chunk_id,
                        )
                    ],
                ).model_dump()
            )

            for entity_name in entity_names:
                normalized_entity = self._normalize_entity(entity_name)
                if normalized_entity is not None and entity_name not in entities:
                    entities[entity_name] = normalized_entity

        return {
            "chunk_id": payload.get("source_chunk_id") or chunk_id,
            "objectives": payload.get("objectives", []),
            "approaches": payload.get("approaches", []),
            "results": payload.get("results", []),
            "constraints": payload.get("constraints", []),
            "claims": claims or payload.get("claims", []),
            "entities": list(entities.values()) or payload.get("entities", []),
        }

    def _normalize_objective(self, item: str | dict, chunk_id: str) -> dict:
        """把单个 objective 规范化成 Objective 所需字段。"""
        # 允许模型偷懒只返回字符串；代码层再把它补成标准对象。
        if isinstance(item, str):
            return {"description": item, "evidence": []}
        item.setdefault("description", item.get("objective", ""))
        item["evidence"] = self._normalize_evidence_list(item.get("evidence", []), chunk_id)
        return item

    def _normalize_approach(self, item: str | dict, chunk_id: str) -> dict:
        """把单个 approach 规范化成 Approach 所需字段。"""
        if isinstance(item, str):
            return {"description": item, "method_names": [], "evidence": []}
        item.setdefault("description", item.get("approach", ""))
        method_names = item.get("method_names") or item.get("methods") or []
        if isinstance(method_names, str):
            method_names = [method_names]
        item["method_names"] = method_names
        item["evidence"] = self._normalize_evidence_list(item.get("evidence", []), chunk_id)
        return item

    def _normalize_result(self, item: str | dict, chunk_id: str) -> dict:
        """把单个 result 规范化成 Result 所需字段。"""
        if isinstance(item, str):
            return {
                "description": item,
                "dataset_names": [],
                "metric_names": [],
                "task_names": [],
                "evidence": [],
            }
        item.setdefault("description", item.get("result", ""))
        for source_key, target_key in [
            ("dataset_names", "dataset_names"),
            ("datasets", "dataset_names"),
            ("metric_names", "metric_names"),
            ("metrics", "metric_names"),
            ("task_names", "task_names"),
            ("tasks", "task_names"),
        ]:
            # 兼容模型返回的别名字段，避免 schema 轻微漂移就导致整条链路失败。
            if source_key in item and target_key not in item:
                item[target_key] = item[source_key]
        for key in ["dataset_names", "metric_names", "task_names"]:
            value = item.get(key, [])
            if isinstance(value, str):
                value = [value]
            item[key] = value
        item["evidence"] = self._normalize_evidence_list(item.get("evidence", []), chunk_id)
        return item

    def _normalize_constraint(self, item: str | dict, chunk_id: str) -> dict:
        """把单个 constraint 规范化成 Constraint 所需字段。"""
        if isinstance(item, str):
            return {"description": item, "evidence": []}
        item.setdefault("description", item.get("constraint", ""))
        item["evidence"] = self._normalize_evidence_list(item.get("evidence", []), chunk_id)
        return item

    def _normalize_claim(self, item: str | dict, chunk_id: str) -> dict:
        """把单个 claim 规范化成 Claim 所需字段。"""
        if isinstance(item, str):
            return {"statement": item, "entity_names": [], "evidence": []}
        item.setdefault("statement", item.get("claim", item.get("description", "")))
        entity_names = item.get("entity_names") or item.get("entities") or []
        if isinstance(entity_names, str):
            entity_names = [entity_names]
        item["entity_names"] = entity_names
        item["evidence"] = self._normalize_evidence_list(item.get("evidence", []), chunk_id)
        return item

    def _normalize_entity(self, item: str | dict) -> dict | None:
        """把实体输出统一成 Entity 可接受的字典。

        返回 None 表示该实体应该被丢弃。
        """
        if isinstance(item, str):
            cleaned = item.strip()
            if not cleaned:
                return None
            return {"name": cleaned, "type": EntityType.CONCEPT.value}

        name = (item.get("name") or item.get("entity") or item.get("text") or "").strip()
        if not name:
            return None

        raw_type = str(item.get("type") or item.get("entity_type") or EntityType.CONCEPT.value).strip()
        normalized_type = self._normalize_entity_type(raw_type)
        if normalized_type is None:
            # 像 Citation/Author 这类对当前 MVP 价值不高的实体，直接过滤掉。
            return None

        return {
            **item,
            "name": name,
            "type": normalized_type,
        }

    def _normalize_entity_type(self, raw_type: str) -> str | None:
        """把模型返回的实体类型映射到当前项目支持的实体类型集合。"""
        aliases = {
            "method": EntityType.METHOD.value,
            "dataset": EntityType.DATASET.value,
            "metric": EntityType.METRIC.value,
            "task": EntityType.TASK.value,
            "model": EntityType.MODEL.value,
            "paperconcept": EntityType.CONCEPT.value,
            "concept": EntityType.CONCEPT.value,
            "citation": None,
            "reference": None,
            "author": None,
        }
        lowered = raw_type.lower().replace(" ", "")
        if lowered in aliases:
            return aliases[lowered]
        allowed_values = {member.value for member in EntityType}
        if raw_type in allowed_values:
            return raw_type
        # 未知类型统一降级成 PaperConcept，尽量保住信息而不是直接报错。
        return EntityType.CONCEPT.value

    def _normalize_evidence_list(self, evidence_items: list | str | dict, chunk_id: str) -> list[dict]:
        """把各种 evidence 输入形态统一成 Evidence 字典列表。"""
        if isinstance(evidence_items, str):
            evidence_items = [evidence_items]
        if isinstance(evidence_items, dict):
            evidence_items = [evidence_items]
        normalized: list[dict] = []
        for item in evidence_items or []:
            if isinstance(item, str):
                normalized.append({"text": item, "chunk_id": chunk_id})
            elif isinstance(item, dict):
                normalized.append(
                    {
                        "text": item.get("text") or item.get("evidence") or "",
                        "chunk_id": item.get("chunk_id") or item.get("source_chunk_id") or chunk_id,
                        "page_number": item.get("page_number"),
                    }
                )
        return normalized
