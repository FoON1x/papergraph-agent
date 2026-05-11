# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

PaperGraph-Agent is a GraphRAG research-paper management system. It converts academic PDFs into a Neo4j knowledge graph and answers questions via a LangChain/LangGraph agent with three tools: `vector_match` (semantic search), `query_graph` (Cypher), and `cross_ref` (cross-paper entity lookup).

## Commands

| Purpose | Command |
|---|---|
| Install dependencies | `uv sync` |
| Check environment | `uv run paperagent doctor` |
| Init Neo4j schema | `uv run paperagent schema init --embedding-dimensions 1024` |
| Ingest PDF(s) | `uv run paperagent ingest --input <path> [--collection default]` |
| Query knowledge base | `uv run paperagent query "<question>" [--collection default]` |
| Inspect a paper node | `uv run paperagent inspect --paper paper:<id>` |
| Run tests | `pytest` |

## Architecture

Two data flows share the same Neo4j backend:

**Ingestion flow** (`ingestion/`) — orchestrated by a LangGraph `StateGraph` with nodes `parse → extract → write`:
1. `DocumentParser` loads PDF via `UnstructuredPDFLoader`, splits with `RecursiveCharacterTextSplitter` → `ParsedDocument` / `ParsedChunk` lists
2. `ExtractionService` sends each chunk to the LLM (DashScope/Qwen via OpenAI-compatible API), which returns structured `ChunkExtraction` (objectives, approaches, results, constraints, claims, entities)
3. `GraphRepository.write_document()` generates embeddings, then writes `Paper → Section → Chunk` document nodes plus semantic nodes (`Evidence`, `Claim`, `Entity`, etc.) and their relationships. Entity dedup uses canonicalized names + fuzzy matching via RapidFuzz.

**Query flow** (`agent/`) — `ResearchAgent` wraps a LangChain agent with three tools:
- `vector_match`: `LocalGraphRetriever` (extends `BaseRetriever`) → `GraphRepository.local_search()` → Neo4j vector index
- `query_graph`: raw Cypher through `GraphRepository.run_cypher()` — guarded by read-only validation
- `cross_ref`: entity lookup across papers via `GraphRepository.cross_reference()`

Both flows share the LLM abstraction layer in `providers/`: a `ChatProvider` protocol (with `DashScopeChatProvider` implementation) and an `EmbeddingProvider` protocol. The factory (`factory.py`) currently only returns DashScope providers but is structured for extension.

## Key design points

- **Entity dedup**: `graph/utils.py` `canonicalize_entity()` normalizes names; `_link_possible_same_as()` in the repository uses RapidFuzz to find fuzzy `SAME_AS` candidates. Entity IDs are deterministic SHA-256 hashes from `semantic_id(label, scope, text)`.
- **Incremental ingestion**: `IngestionPipeline` normalizes paper titles and checks Neo4j for existing matches before processing.
- **Concurrency**: `ExtractionService` uses `asyncio.gather` + semaphore (`PAPERAGENT_MAX_CONCURRENCY`) to extract multiple chunks in parallel.
- **Tool middleware**: `guard_tool_call()` in `agent/workflow.py` validates readonly Cypher, parses JSON args, tracks per-tool failures, and limits total calls via `ToolCallLimitMiddleware`.
- **Config**: all settings come from `Settings` (Pydantic `BaseSettings`) in `config.py`, loaded from `.env` via `python-dotenv`. Use `get_settings()` for the cached singleton.
- **Schema initialization**: `GraphSchemaManager.init_schema()` is idempotent — it creates constraints, indexes, and vector indexes only if they don't exist.
