import asyncio
import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from paperagent.config import Settings, get_settings
from paperagent.extraction.prompts import EXTRACTION_HUMAN_PROMPT, EXTRACTION_SYSTEM_PROMPT
from paperagent.providers import ChatProvider, get_chat_provider
from paperagent.schemas import ChunkExtraction, Claim, Entity, EntityType, Evidence, ParsedDocument, PaperExtraction


class ExtractionService:
    """Extract scientific schema objects from parsed chunks."""

    def __init__(
        self,
        chat_provider: ChatProvider | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.chat_provider = chat_provider or get_chat_provider(self.settings)
        self.chain = self._build_chain()

    async def extract_document(self, document: ParsedDocument) -> PaperExtraction:
        inputs = [
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
                raw_output = await self.chain.ainvoke(payload)
                return self._coerce_extraction(raw_output, payload["chunk_id"])

        extractions = await asyncio.gather(*(extract_one(payload) for payload in inputs))
        for chunk, extraction in zip(document.chunks, extractions, strict=False):
            if extraction.chunk_id != chunk.chunk_id:
                extraction.chunk_id = chunk.chunk_id
        return PaperExtraction(paper_id=document.paper_id, title=document.title, chunks=list(extractions))

    def extract_chunk(
        self,
        paper_id: str,
        chunk_id: str,
        chunk_text: str,
        page_number: int | None = None,
    ) -> ChunkExtraction:
        raw_output = self.chain.invoke(
            {
                "paper_id": paper_id,
                "chunk_id": chunk_id,
                "page_number": page_number or "unknown",
                "chunk_text": chunk_text,
            }
        )
        return self._coerce_extraction(raw_output, chunk_id)

    def _build_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", EXTRACTION_SYSTEM_PROMPT),
                ("human", EXTRACTION_HUMAN_PROMPT),
            ]
        )
        return prompt | self.chat_provider.get_chat_model() | StrOutputParser()

    def _coerce_extraction(self, raw_output: str, chunk_id: str) -> ChunkExtraction:
        payload = self._parse_json_object(raw_output)
        normalized = self._normalize_extraction_payload(payload, chunk_id)
        extraction = ChunkExtraction.model_validate(normalized)
        if extraction.chunk_id != chunk_id:
            extraction.chunk_id = chunk_id
        return extraction

    def _parse_json_object(self, raw_output: str) -> dict:
        text = raw_output.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return json.loads(text)

    def _normalize_extraction_payload(self, payload: dict, chunk_id: str) -> dict:
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
        if isinstance(item, str):
            return {"description": item, "evidence": []}
        item.setdefault("description", item.get("objective", ""))
        item["evidence"] = self._normalize_evidence_list(item.get("evidence", []), chunk_id)
        return item

    def _normalize_approach(self, item: str | dict, chunk_id: str) -> dict:
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
        if isinstance(item, str):
            return {"description": item, "evidence": []}
        item.setdefault("description", item.get("constraint", ""))
        item["evidence"] = self._normalize_evidence_list(item.get("evidence", []), chunk_id)
        return item

    def _normalize_claim(self, item: str | dict, chunk_id: str) -> dict:
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
            return None

        return {
            **item,
            "name": name,
            "type": normalized_type,
        }

    def _normalize_entity_type(self, raw_type: str) -> str | None:
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
        return EntityType.CONCEPT.value

    def _normalize_evidence_list(self, evidence_items: list | str | dict, chunk_id: str) -> list[dict]:
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
