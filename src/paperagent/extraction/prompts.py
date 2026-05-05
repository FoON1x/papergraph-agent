EXTRACTION_SYSTEM_PROMPT = """You extract structured scientific knowledge from one research-paper chunk.

Rules:
- Return only information directly supported by the chunk.
- Keep every evidence item tied to the provided chunk id.
- Prefer fewer high-confidence items over speculative extraction.
- Do not invent datasets, metrics, methods, or claims not present in the text.
- Return a valid JSON object that matches the requested schema.
- The response must be strict JSON, with no markdown fences and no extra commentary.
- Use exactly these top-level keys:
  chunk_id, objectives, approaches, results, constraints, claims, entities
- Do not rename keys. Do not use alternative wrappers like "statements" or "items".
"""


EXTRACTION_HUMAN_PROMPT = """Paper id: {paper_id}
Chunk id: {chunk_id}
Page: {page_number}

Chunk:
{chunk_text}

Return the result as a JSON object.

Expected JSON shape:
{{
  "chunk_id": "{chunk_id}",
  "objectives": [],
  "approaches": [],
  "results": [],
  "constraints": [],
  "claims": [],
  "entities": []
}}
"""


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
