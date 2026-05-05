# PaperGraph-Agent

[中文](#中文说明) | [English](#english)

PaperGraph-Agent is a GraphRAG-oriented research literature ingestion and reasoning project built around **LangChain**, **LangGraph**, and **Neo4j**. It turns PDF papers into a traceable knowledge graph, supports hybrid retrieval over graph and vector signals, and answers research questions through an agent-style workflow.

The current repository is an **MVP that already supports end-to-end execution**:

- initialize Neo4j schema
- ingest one PDF or a directory of PDFs
- extract paper structure and scientific knowledge
- write graph data and embeddings into Neo4j
- query the knowledge base through a LangChain/LangGraph agent

---

## 中文说明

### 1. 项目简介

PaperGraph-Agent 是一个面向科研文献场景的 GraphRAG 项目，目标是把论文从“纯文本 PDF”变成“可追溯、可连接、可检索、可推理”的知识图谱。

当前版本聚焦于一条可运行的 MVP 主链路：

1. 解析 PDF 文献
2. 抽取科研语义信息
3. 写入 Neo4j 图数据库与向量索引
4. 用 LangChain / LangGraph Agent 对图谱进行查询与回答

### 2. 核心能力

- **基于 LangChain 的文档处理与结构化抽取**
  - 使用 LangChain 文档加载器和文本切分器处理 PDF
  - 使用 LLM 对 chunk 做科研语义抽取

- **基于 LangGraph 的流程编排**
  - Ingestion 工作流采用 LangGraph 组织 `parse -> extract -> write`
  - Query 工作流使用 LangChain Agent，并运行在 LangGraph 支撑的 agent runtime 上

- **基于 Neo4j 的图谱与向量混合检索**
  - 使用 `langchain-neo4j` 对接 `Neo4jGraph` 和 `Neo4jVector`
  - 支持图谱查询、向量相似检索和面向 Agent 的工具调用

- **面向科研文献的知识建模**
  - 文档层：`Paper`, `Section`, `Chunk`
  - 语义层：`Evidence`, `Claim`, `Objective`, `Approach`, `Result`, `Constraint`
  - 实体层：`Entity`, `Method`, `Dataset`, `Metric`, `Task`, `Model`, `PaperConcept`

- **增量入库**
  - 当前版本支持按论文标题归一化后进行跳过判断，避免同一篇论文反复导入

### 3. 项目结构

```text
src/paperagent/
├── agent/         # LangChain Agent / LangGraph query workflow
├── extraction/    # LLM extraction prompts and normalization
├── graph/         # Neo4j schema, repository, graph utilities
├── ingestion/     # PDF parsing and LangGraph ingestion workflow
├── providers/     # DashScope/Qwen model provider abstraction
├── retrieval/     # LangChain retriever and local GraphRAG logic
├── cli.py         # Typer CLI entrypoint
├── config.py      # Environment-driven settings
└── schemas.py     # Core Pydantic schemas
```

### 4. 技术栈

- **编排层**：LangChain, LangGraph
- **图数据库 / 向量检索**：Neo4j, langchain-neo4j
- **解析层**：Unstructured PDF Loader
- **模型层**：DashScope-compatible Qwen chat and embedding models
- **配置与验证**：Pydantic Settings
- **CLI**：Typer + Rich

### 5. 运行前准备

#### 5.1 Python 版本

- Python `>= 3.12`

#### 5.2 安装依赖

推荐使用 `uv`：

```bash
uv sync
```

如果你已经有虚拟环境，也可以使用：

```bash
uv pip install -e .
```

#### 5.3 配置环境变量

将 `.env.example` 复制为 `.env`，并填写实际配置：

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
```

建议首次跑通时使用：

```bash
PAPERAGENT_ENABLE_THINKING=false
PAPERAGENT_MAX_CONCURRENCY=1
```

这样通常更稳。

### 6. 全流程使用方法

#### 第一步：检查本地依赖与连接

```bash
uv run paperagent doctor
```

这个命令会帮助检查：

- Python 依赖是否齐全
- DashScope key 是否可读取
- Neo4j 是否可连接

#### 第二步：初始化 Neo4j schema

```bash
uv run paperagent schema init --embedding-dimensions 1024
```

这个命令会在 Neo4j 中创建：

- 唯一约束
- 普通索引
- 向量索引

`1024` 需要与你使用的 embedding 模型输出维度一致。

#### 第三步：导入论文

导入单篇 PDF：

```bash
uv run paperagent ingest --input papers/example.pdf
```

导入一个目录下的全部 PDF：

```bash
uv run paperagent ingest --input papers --collection default
```

#### 第四步：发起查询

```bash
uv run paperagent query "总结这篇论文的核心方法和主要结果"
```

指定 collection 查询：

```bash
uv run paperagent query "GraphRAG 在这批论文中的主要应用场景是什么？" --collection default
```

#### 第五步：查看某篇论文节点摘要

```bash
uv run paperagent inspect --paper paper:xxxxxxxxxxxxxxxx
```

### 7. 当前系统的数据流

```text
PDF
-> LangChain loader / splitter
-> chunk 级科研语义抽取
-> Pydantic schema 规范化
-> Neo4j graph + vector write
-> LangChain retriever / tools
-> LangGraph-backed agent query
```

### 8. 当前实现的图谱模型

#### 文档层

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
- `SAME_AS`

### 9. 当前状态与边界

当前仓库已经可以完成导入与查询闭环，但它仍然是一个偏研究型的 MVP。

目前已经具备：

- 从 PDF 到图谱的基本链路
- Neo4j 中的图/向量混合存储
- 基于 LangChain / LangGraph 的查询路径
- 基础增量导入能力

目前仍值得继续打磨的地方包括：

- 抽取速度与稳定性
- 更稳的论文级去重策略
- 更精细的实体对齐
- 更强的全局社区检索与反思环路
- 更完善的测试与评测

### 10. 常见问题

#### 导入速度较慢

这通常来自以下几部分：

- PDF 解析较重
- chunk 数较多
- 每个 chunk 都需要一次 LLM 抽取
- Neo4j 写入前需要生成 embeddings

如果你只是想先跑通系统，建议：

- 关闭 thinking 模式
- 将 `PAPERAGENT_MAX_CONCURRENCY` 设为 `1`
- 先用少量论文验证流程

#### 为什么 Neo4j 中节点很多

因为系统不是“1 篇论文 = 1 个节点”，而是会拆成多层：

- 论文结构节点
- chunk 节点
- claim / evidence 节点
- entity 节点
- 各类语义关系

这是 GraphRAG 建模的一部分。

### 11. 开源建议

如果你准备公开发布这个仓库，建议：

- 使用 `.env.example` 提供模板，不要提交真实 `.env`
- 不要提交论文原文 PDF、实验输出、临时日志
- 在提交前确认 Neo4j 连接信息、密钥和本地测试数据没有泄露

### 12. License

当前仓库尚未包含许可证文件。若准备正式开源，建议补充 `LICENSE`（如 MIT、Apache-2.0 等）。

---

## English

### 1. Overview

PaperGraph-Agent is a GraphRAG-oriented project for research-paper ingestion, knowledge extraction, and question answering. It converts academic PDFs into a traceable knowledge graph that can be searched through both graph and vector signals.

The current repository focuses on a practical MVP pipeline:

1. parse papers from PDF
2. extract scientific knowledge from chunks
3. write graph data and embeddings into Neo4j
4. answer research questions with a LangChain/LangGraph agent

### 2. Features

- **LangChain-based document handling and extraction**
  - PDF loading and chunking through LangChain components
  - LLM-powered scientific knowledge extraction per chunk

- **LangGraph-based orchestration**
  - ingestion workflow organized as `parse -> extract -> write`
  - query path driven by a LangChain agent running on LangGraph-backed runtime

- **Neo4j graph + vector hybrid retrieval**
  - integration through `langchain-neo4j`
  - support for graph querying, vector similarity search, and agent tools

- **Research-aware knowledge model**
  - document layer: `Paper`, `Section`, `Chunk`
  - semantic layer: `Evidence`, `Claim`, `Objective`, `Approach`, `Result`, `Constraint`
  - entity layer: `Entity`, `Method`, `Dataset`, `Metric`, `Task`, `Model`, `PaperConcept`

- **Incremental ingestion**
  - the current version can skip already indexed papers based on normalized title matching

### 3. Repository Layout

```text
src/paperagent/
├── agent/         # LangChain agent and query workflow
├── extraction/    # prompts and extraction normalization
├── graph/         # Neo4j schema and repository logic
├── ingestion/     # PDF parsing and LangGraph ingestion workflow
├── providers/     # model provider abstraction
├── retrieval/     # retriever and local GraphRAG logic
├── cli.py         # Typer CLI entrypoint
├── config.py      # environment-based settings
└── schemas.py     # Pydantic schemas
```

### 4. Tech Stack

- **Orchestration**: LangChain, LangGraph
- **Graph / vector storage**: Neo4j, langchain-neo4j
- **PDF parsing**: Unstructured PDF loader
- **LLM / embeddings**: DashScope-compatible Qwen models
- **Configuration**: Pydantic Settings
- **CLI**: Typer + Rich

### 5. Prerequisites

#### 5.1 Python

- Python `>= 3.12`

#### 5.2 Install dependencies

Using `uv` is recommended:

```bash
uv sync
```

Or install the project into an existing environment:

```bash
uv pip install -e .
```

#### 5.3 Configure environment variables

Copy `.env.example` to `.env` and fill in real values:

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
```

For the first successful run, the safest setup is usually:

```bash
PAPERAGENT_ENABLE_THINKING=false
PAPERAGENT_MAX_CONCURRENCY=1
```

### 6. End-to-End Usage

#### Step 1: Check local prerequisites

```bash
uv run paperagent doctor
```

This checks:

- Python dependencies
- DashScope key availability
- Neo4j connectivity

#### Step 2: Initialize Neo4j schema

```bash
uv run paperagent schema init --embedding-dimensions 1024
```

This creates the required:

- uniqueness constraints
- indexes
- vector indexes

The dimension value must match the output size of your configured embedding model.

#### Step 3: Ingest papers

Ingest a single PDF:

```bash
uv run paperagent ingest --input papers/example.pdf
```

Ingest all PDFs in a directory:

```bash
uv run paperagent ingest --input papers --collection default
```

#### Step 4: Query the knowledge base

```bash
uv run paperagent query "Summarize the core method and main results of this paper."
```

Query a specific collection:

```bash
uv run paperagent query "What are the main GraphRAG application patterns in this collection?" --collection default
```

#### Step 5: Inspect one paper node

```bash
uv run paperagent inspect --paper paper:xxxxxxxxxxxxxxxx
```

### 7. Data Flow

```text
PDF
-> LangChain loader / splitter
-> chunk-level scientific extraction
-> Pydantic normalization
-> Neo4j graph + vector write
-> LangChain retriever / tools
-> LangGraph-backed agent query
```

### 8. Graph Model

#### Document layer

- `Paper`
- `Section`
- `Chunk`

#### Scientific semantic layer

- `Evidence`
- `Claim`
- `Objective`
- `Approach`
- `Result`
- `Constraint`

#### Entity layer

- `Entity`
- `Method`
- `Dataset`
- `Metric`
- `Task`
- `Model`
- `PaperConcept`

#### Common relationships

- `HAS_SECTION`
- `HAS_CHUNK`
- `HAS_EVIDENCE`
- `SUPPORTS`
- `MENTIONS`
- `USES_METHOD`
- `EVALUATED_ON`
- `REPORTS_METRIC`
- `FOR_TASK`
- `SAME_AS`

### 9. Current Status

This repository already supports a working ingest-and-query loop, but it should still be viewed as an MVP rather than a production system.

It already includes:

- a working PDF-to-graph pipeline
- graph and vector storage in Neo4j
- a LangChain / LangGraph-based query path
- basic incremental ingestion

Areas still worth improving:

- extraction speed and robustness
- stronger paper-level deduplication
- more precise entity alignment
- richer global retrieval and reflection loops
- broader testing and evaluation

### 10. FAQ

#### Why is ingestion slow?

The heavy parts are usually:

- PDF parsing
- large chunk counts
- one LLM extraction call per chunk
- embedding generation before writing to Neo4j

If your main goal is to get the system running first, start with:

- thinking mode disabled
- `PAPERAGENT_MAX_CONCURRENCY=1`
- a small paper set

#### Why are there so many nodes in Neo4j?

Because the graph is not modeled as “one paper equals one node.” A single paper can produce:

- document structure nodes
- chunk nodes
- claim and evidence nodes
- entity nodes
- many typed relationships

That is expected in a GraphRAG-style design.

### 11. Open-Source Checklist

Before publishing this repository, make sure:

- only `.env.example` is committed, not a real `.env`
- paper PDFs, temporary logs, and local outputs are ignored
- no API keys or local Neo4j credentials remain in tracked files

### 12. License

This repository does not yet include a license file. If you plan to open-source it, consider adding a `LICENSE` file such as MIT or Apache-2.0.
