# PaperGraph-Agent

[中文](#中文说明) | [English](#english)

PaperGraph-Agent is a research-paper GraphRAG MVP built with **LangChain**, **LangGraph**, and **Neo4j**. It parses academic PDFs, extracts structured scientific knowledge, stores both graph data and embeddings in Neo4j, and answers questions through an agent-based query workflow.

The project already supports an end-to-end loop:

- initialize Neo4j schema and vector indexes
- ingest a single PDF or a directory of PDFs
- build a traceable graph over papers, chunks, entities, claims, and evidence
- run hybrid retrieval over vector and graph signals
- answer research questions through a guarded LangChain/LangGraph agent

---

## 中文说明

### 1. 当前项目能做什么

PaperGraph-Agent 目前已经具备一条可运行的 GraphRAG 主链路：

1. 解析科研论文 PDF
2. 将论文切分为可检索的 chunk
3. 对每个 chunk 做结构化科研语义抽取
4. 将文档结构、语义节点、实体关系和 embedding 一起写入 Neo4j
5. 通过 Agent 组合向量检索、图查询和跨论文对照来回答问题

它现在更像一个**研究型 MVP / 可演示原型**，而不是生产系统，但导入与查询闭环已经打通。

### 2. 核心功能

- **PDF 导入与解析**
  - 支持导入单篇 PDF 或整个目录
  - 基于 LangChain 文档加载与文本切分，将论文拆为 `Section` / `Chunk`

- **LangGraph 编排的抽取流程**
  - 外层导入工作流：`parse -> extract -> write`
  - 内层 chunk 抽取工作流：
    `prepare_payload -> format_prompt -> extract_structured -> normalize -> validate`
  - 当 structured output 失败时，会退回原始 JSON 文本路径再做解析与校验

- **科研语义抽取**
  - 抽取对象包括：
    - `Objective`
    - `Approach`
    - `Result`
    - `Constraint`
    - `Claim`
    - `Evidence`
    - `Entity`
  - 对模型脏输出做字段规范化、别名兼容、实体过滤和 schema 校验

- **Neo4j 图谱入库**
  - 文档结构层：`Paper`, `Section`, `Chunk`
  - 语义层：`Evidence`, `Claim`, `Objective`, `Approach`, `Result`, `Constraint`
  - 实体层：`Entity`, `Method`, `Dataset`, `Metric`, `Task`, `Model`, `PaperConcept`
  - 写入时同时生成 chunk embedding，供后续向量检索使用

- **图谱 + 向量混合检索**
  - 基于 `Neo4jVector` 对 `Chunk.embedding` 做相似度搜索
  - 检索结果可带回 chunk 关联的 `Evidence` / `Claim`
  - 支持图查询和跨论文实体对照

- **Agent 问答**
  - 使用 LangChain `create_agent(...)`
  - 工具包括：
    - `vector_match`
    - `query_graph`
    - `cross_ref`
  - 已加入工具调用上限、只读 Cypher 校验和工具错误兜底

- **增量导入**
  - 当前通过归一化标题判断是否跳过已存在论文
  - 适合维护一个本地文献库

- **实体相似对齐**
  - 支持建立 `SAME_AS` 候选关系
  - 若 Neo4j 已启用 APOC，则优先使用 `apoc.text.sorensenDiceSimilarity(...)`
  - 若 APOC 不可用，会退回 Python 侧 `rapidfuzz` fallback

### 3. 代码结构

```text
src/paperagent/
├── agent/         # LangChain agent and tool-guarded query workflow
├── extraction/    # prompts, chunk workflow, normalization, extraction service
├── graph/         # Neo4j schema, repository, graph utilities
├── ingestion/     # PDF parsing, workflow, pipeline
├── providers/     # DashScope/Qwen chat and embedding provider abstraction
├── retrieval/     # retriever and local GraphRAG answer path
├── cli.py         # Typer CLI entrypoint
├── config.py      # environment-driven settings
└── schemas.py     # core Pydantic schemas
```

### 4. 技术栈

- **编排**：LangChain, LangGraph
- **图数据库 / 向量检索**：Neo4j, langchain-neo4j
- **模型**：DashScope-compatible Qwen chat + embedding models
- **PDF 解析**：Unstructured
- **配置**：Pydantic Settings
- **CLI**：Typer + Rich

### 5. 当前数据流

```text
PDF
-> DocumentParser
-> ParsedDocument / Chunk
-> chunk-level extraction workflow
-> normalized ChunkExtraction / PaperExtraction
-> GraphRepository write
-> Neo4j graph + vector index
-> retriever + graph tools
-> LangChain/LangGraph agent answer
```

### 6. 图谱模型

#### 文档结构层

- `Paper`
- `Section`
- `Chunk`

#### 科研语义层

- `Evidence`
- `Claim`
- `Objective`
- `Approach`
- `Result`
- `Constraint`

#### 实体层

- `Entity`
- `Method`
- `Dataset`
- `Metric`
- `Task`
- `Model`
- `PaperConcept`

#### 常见关系

- `HAS_SECTION`
- `HAS_CHUNK`
- `HAS_EVIDENCE`
- `SUPPORTS`
- `MENTIONS`
- `USES_METHOD`
- `EVALUATED_ON`
- `REPORTS_METRIC`
- `FOR_TASK`
- `ABOUT`
- `SAME_AS`

### 7. CLI 用法

#### 7.1 检查环境

```bash
uv run paperagent doctor
```

#### 7.2 初始化 Neo4j schema

```bash
uv run paperagent schema init --embedding-dimensions 1024
```

这一步会创建：

- 唯一约束
- 普通索引
- 向量索引

#### 7.3 导入论文

导入单篇 PDF：

```bash
uv run paperagent ingest --input papers/example.pdf
```

导入目录中的全部 PDF：

```bash
uv run paperagent ingest --input papers --collection default
```

当前命令会统计：

- `Ingested N paper(s)`
- `Skipped M existing paper(s)`

#### 7.4 查询知识库

```bash
uv run paperagent query "总结这篇论文的核心方法和主要结果"
```

指定 collection：

```bash
uv run paperagent query "GraphRAG 在这批论文中的主要应用场景是什么？" --collection default
```

#### 7.5 查看单篇论文摘要

```bash
uv run paperagent inspect --paper paper:xxxxxxxxxxxxxxxx
```

### 8. 环境变量

最常见的配置项包括：

```bash
DASHSCOPE_API_KEY=your_api_key
LLM_PROVIDER=dashscope
CHAT_MODEL=qwen3.5-flash
EMBEDDING_MODEL=text-embedding-v4
PAPERAGENT_ENABLE_THINKING=false

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

PAPERAGENT_CHUNK_SIZE=1200
PAPERAGENT_CHUNK_OVERLAP=160
PAPERAGENT_MAX_CONCURRENCY=1
PAPERAGENT_MAX_TOOL_CALLS=4
PAPERAGENT_MAX_TOOL_FAILURES=1
PAPERAGENT_ENTITY_MATCH_THRESHOLD=90
```

首次跑通建议：

```bash
PAPERAGENT_ENABLE_THINKING=false
PAPERAGENT_MAX_CONCURRENCY=1
```

### 9. 当前状态与边界

当前仓库已经实现：

- 论文导入闭环
- chunk 级结构化抽取
- Neo4j 图谱与向量存储
- Agent 问答闭环
- 基础增量导入
- 基础实体相似对齐

但它仍然属于 MVP，后续仍值得继续打磨：

- 更稳的论文级去重策略
- 更细的图谱扩展与 rerank
- 更完善的多轮对话状态管理
- 更强的评测、测试与回归验证
- 更统一的仓储层查询风格

### 10. FAQ

#### 导入为什么比较慢？

主要瓶颈通常在：

- PDF 解析
- chunk 数量
- 每个 chunk 的 LLM 抽取
- embedding 生成

#### 为什么 Neo4j 里节点很多？

因为系统不是“1 篇论文 = 1 个节点”，而是会拆成：

- 文档结构节点
- chunk 节点
- 语义节点
- 实体节点
- 它们之间的关系

这正是 GraphRAG 建模的一部分。

#### 必须安装 APOC 吗？

不是必须。

- 若安装并启用 APOC，`SAME_AS` 对齐会优先走库内 Sorensen-Dice 相似度
- 若没有 APOC，代码会退回到 Python 侧 `rapidfuzz`

### 11. 开源提醒

公开发布前请确认：

- 不要提交真实 `.env`
- 不要提交论文 PDF、日志、临时输出
- 不要泄露 Neo4j 凭据和 API key

### 12. License

仓库目前还没有正式许可证文件；若准备公开发布，建议补充 `LICENSE`（如 MIT 或 Apache-2.0）。

---

## English

### 1. What the project currently does

PaperGraph-Agent currently supports a working GraphRAG loop for research papers:

1. parse academic PDFs
2. split them into retrievable chunks
3. extract structured scientific knowledge per chunk
4. write graph data and embeddings into Neo4j
5. answer questions through an agent that combines vector retrieval and graph tools

This repository should still be viewed as an **MVP / research prototype**, but the ingest-and-query loop is already functional.

### 2. Core capabilities

- **PDF ingestion and parsing**
  - ingest a single PDF or an entire directory
  - build `Section` / `Chunk` structure from papers

- **LangGraph-driven extraction**
  - outer ingestion workflow: `parse -> extract -> write`
  - inner chunk workflow:
    `prepare_payload -> format_prompt -> extract_structured -> normalize -> validate`
  - fallback from structured output to raw JSON parsing when needed

- **Scientific knowledge extraction**
  - extracts:
    - `Objective`
    - `Approach`
    - `Result`
    - `Constraint`
    - `Claim`
    - `Evidence`
    - `Entity`
  - includes normalization, alias handling, entity filtering, and schema validation

- **Neo4j graph persistence**
  - document layer: `Paper`, `Section`, `Chunk`
  - semantic layer: `Evidence`, `Claim`, `Objective`, `Approach`, `Result`, `Constraint`
  - entity layer: `Entity`, `Method`, `Dataset`, `Metric`, `Task`, `Model`, `PaperConcept`
  - chunk embeddings are written together with graph nodes

- **Hybrid graph + vector retrieval**
  - vector search over `Chunk.embedding` via `Neo4jVector`
  - retrieval can return chunk-linked evidence and claim context
  - supports graph querying and cross-paper entity lookup

- **Agent-based QA**
  - built on LangChain `create_agent(...)`
  - tools:
    - `vector_match`
    - `query_graph`
    - `cross_ref`
  - includes tool-call limits, read-only Cypher validation, and tool error guards

- **Incremental ingestion**
  - skips already indexed papers by normalized title matching

- **Entity similarity linking**
  - builds `SAME_AS` candidate edges
  - prefers APOC `apoc.text.sorensenDiceSimilarity(...)` when available
  - falls back to Python-side `rapidfuzz` otherwise

### 3. Repository layout

```text
src/paperagent/
├── agent/         # LangChain agent and guarded query workflow
├── extraction/    # prompts, chunk workflow, normalization, extraction service
├── graph/         # Neo4j schema, repository, graph utilities
├── ingestion/     # PDF parsing, workflow, pipeline
├── providers/     # DashScope/Qwen model provider abstraction
├── retrieval/     # retriever and local GraphRAG answer path
├── cli.py         # Typer CLI entrypoint
├── config.py      # environment-driven settings
└── schemas.py     # core Pydantic schemas
```

### 4. Tech stack

- **Orchestration**: LangChain, LangGraph
- **Graph / vector storage**: Neo4j, langchain-neo4j
- **Models**: DashScope-compatible Qwen chat + embedding models
- **PDF parsing**: Unstructured
- **Configuration**: Pydantic Settings
- **CLI**: Typer + Rich

### 5. CLI usage

Check prerequisites:

```bash
uv run paperagent doctor
```

Initialize Neo4j schema:

```bash
uv run paperagent schema init --embedding-dimensions 1024
```

Ingest one paper:

```bash
uv run paperagent ingest --input papers/example.pdf
```

Ingest a directory:

```bash
uv run paperagent ingest --input papers --collection default
```

Query the knowledge base:

```bash
uv run paperagent query "Summarize the core method and main results of this paper."
```

Inspect one paper:

```bash
uv run paperagent inspect --paper paper:xxxxxxxxxxxxxxxx
```

### 6. Data flow

```text
PDF
-> DocumentParser
-> ParsedDocument / Chunk
-> chunk-level extraction workflow
-> normalized ChunkExtraction / PaperExtraction
-> GraphRepository write
-> Neo4j graph + vector index
-> retriever + graph tools
-> LangChain/LangGraph agent answer
```

### 7. Current status

The project already includes:

- an end-to-end ingestion loop
- chunk-level structured extraction
- Neo4j graph and vector storage
- an agent-based QA loop
- basic incremental ingestion
- basic entity similarity linking

Areas still worth improving:

- stronger paper-level deduplication
- richer graph expansion and reranking
- cleaner multi-turn state handling
- broader testing and evaluation
- more consistent repository-layer query style

### 8. FAQ

#### Why is ingestion slow?

Typical bottlenecks are:

- PDF parsing
- chunk count
- one LLM extraction call per chunk
- embedding generation

#### Why are there so many nodes in Neo4j?

Because the graph is not modeled as “one paper = one node”. A single paper expands into:

- document structure nodes
- chunk nodes
- semantic nodes
- entity nodes
- and the relationships among them

#### Is APOC required?

Not strictly.

- If APOC is installed and enabled, `SAME_AS` linking runs inside Neo4j with Sorensen-Dice similarity
- Otherwise the code falls back to Python-side `rapidfuzz`

### 9. Open-source note

Before publishing, make sure:

- no real `.env` is committed
- paper PDFs, local logs, and temporary outputs are ignored
- no Neo4j credentials or API keys remain in tracked files

### 10. License

The repository does not yet include a formal license file. If you plan to open-source it, consider adding MIT or Apache-2.0.
