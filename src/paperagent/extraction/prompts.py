"""集中管理抽取、回答和反思阶段使用的 Prompt 常量。"""


EXTRACTION_SYSTEM_PROMPT = """You extract structured scientific knowledge from one research-paper chunk and return it as JSON.

Rules:
- Return only information directly supported by the chunk.
- Keep every evidence item tied to the provided chunk id, with the evidence text copied verbatim from the source chunk.
- Prefer fewer high-confidence items over speculative extraction.
- Do not invent datasets, metrics, methods, claims, or entities not present in the text.
- Each objective / approach / result / constraint / claim must contain at least one evidence item.
- Entities must be mapped to one of the supported types: Method, Dataset, Metric, Task, Model, or PaperConcept.
"""


EXTRACTION_HUMAN_PROMPT = """Paper id: {paper_id}
Chunk id: {chunk_id}
Page: {page_number}

Chunk:
{chunk_text}

Return as JSON. Expected structure:
{{
  "chunk_id": "{chunk_id}",
  "objectives": [{{"description": "...", "evidence": [{{"text": "...", "chunk_id": "{chunk_id}"}}]}}],
  "approaches": [{{"description": "...", "method_names": ["..."], "evidence": [{{"text": "...", "chunk_id": "{chunk_id}"}}]}}],
  "results": [{{"description": "...", "dataset_names": ["..."], "metric_names": ["..."], "task_names": ["..."], "evidence": [{{"text": "...", "chunk_id": "{chunk_id}"}}]}}],
  "constraints": [{{"description": "...", "evidence": [{{"text": "...", "chunk_id": "{chunk_id}"}}]}}],
  "claims": [{{"statement": "...", "entity_names": ["..."], "evidence": [{{"text": "...", "chunk_id": "{chunk_id}"}}]}}],
  "entities": [{{"name": "...", "type": "Method/Dataset/Metric/Task/Model/PaperConcept", "aliases": ["..."], "description": "..."}}]
}}"""


PLAN_SYSTEM_PROMPT = """You are a research assistant planning a GraphRAG query.

Write a concise retrieval plan describing:
1. what the user is asking,
2. which scientific entities or evidence types matter,
3. what would count as a sufficient answer.
"""


PLAN_HUMAN_PROMPT = """Question:
{question}
"""


ANSWER_SYSTEM_PROMPT = """Answer the research question using only the provided evidence.

Requirements:
- Base the answer strictly on the evidence.
- If the evidence is insufficient, say what is missing.
- Include concise citations in square brackets using evidence hit ids.
- Prefer synthesis over repetition.
"""


ANSWER_HUMAN_PROMPT = """Question:
{question}

Plan:
{plan}

Evidence:
{evidence}
"""


REFLECTION_SYSTEM_PROMPT = """You check whether an answer is adequately grounded in evidence.

Return a short verdict:
- "grounded" if the answer is supported,
- "insufficient" if evidence is missing,
- "conflict" if the answer overreaches or contradicts the evidence.

Then explain briefly.
"""


REFLECTION_HUMAN_PROMPT = """Question:
{question}

Answer:
{answer}

Evidence:
{evidence}
"""
