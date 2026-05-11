# PaperGraph-Agent 中文教学文档

> 这是一份“带你从 0 到 1 读懂项目”的中文教程。目标不是只告诉你这个项目“做了什么”，而是带你逐步看清：**它为什么这么设计、每个模块如何衔接、关键代码在什么位置、运行时到底发生了什么。**

---

## 目录

1. [先建立整体心智模型](#1-先建立整体心智模型)
2. [项目解决了什么问题](#2-项目解决了什么问题)
3. [项目目录结构](#3-项目目录结构)
4. [从命令行入口开始：程序如何启动](#4-从命令行入口开始程序如何启动)
5. [配置系统：`.env` 是如何驱动整个项目的](#5-配置系统env-是如何驱动整个项目的)
6. [核心数据模型：项目内部到底在传什么对象](#6-核心数据模型项目内部到底在传什么对象)
7. [第一阶段：PDF 解析与 Chunk 构建](#7-第一阶段pdf-解析与-chunk-构建)
8. [第二阶段：LLM 结构化抽取](#8-第二阶段llm-结构化抽取)
9. [第三阶段：LangGraph 编排导入流程](#9-第三阶段langgraph-编排导入流程)
10. [第四阶段：Neo4j 节点与关系写入](#10-第四阶段neo4j-节点与关系写入)
11. [图谱 Schema 初始化：为什么要先 `schema init`](#11-图谱-schema-初始化为什么要先-schema-init)
12. [第五阶段：向量检索与 Local GraphRAG](#12-第五阶段向量检索与-local-graphrag)
13. [第六阶段：LangChain Agent / LangGraph 查询链路](#13-第六阶段langchain-agent--langgraph-查询链路)
14. [为什么说这个项目是“图谱 + 向量混合检索”](#14-为什么说这个项目是图谱--向量混合检索)
15. [一次完整运行到底发生了什么](#15-一次完整运行到底发生了什么)
16. [关键设计选择与原因](#16-关键设计选择与原因)
17. [当前实现的边界与可改进方向](#17-当前实现的边界与可改进方向)
18. [建议的阅读顺序](#18-建议的阅读顺序)

---

## 1. 先建立整体心智模型

先不要急着扎进代码。这个项目最核心的一句话是：

> **它把 PDF 论文变成 Neo4j 知识图谱，再通过 LangChain/LangGraph 驱动的 Agent 在图谱上做问答。**

完整链路可以先记成下面这张“脑内流程图”：

```text
PDF
-> 文本解析
-> Chunk 切分
-> LLM 抽取科研语义
-> 生成结构化对象
-> 写入 Neo4j 图谱和向量索引
-> 用户提问
-> 向量召回相关 Chunk
-> 图谱扩展实体/证据
-> Agent 组织答案
```

这个项目不是普通的“上传 PDF 然后让模型总结一下”。它真正想做的是：

- **结构化**：把论文里的内容拆成节点和关系
- **可追溯**：每个结论能回到原始 Chunk / Evidence
- **可连接**：不同论文中的同类实体可以连起来
- **可检索**：支持向量检索，也支持图谱查询
- **可推理**：让 Agent 在检索结果之上做回答

---

## 2. 项目解决了什么问题

传统 RAG 在科研文献场景会遇到几个典型问题：

1. **只会做文本相似度召回**  
   它能找到“看起来像”的段落，但不擅长跨论文结构化关联。

2. **证据追溯不够清楚**  
   回答里常常只有一段文本，没有清楚的科研语义结构。

3. **跨论文对比能力弱**  
   比如你问“哪些方法都用过 HotpotQA”，纯向量库不容易直接做。

4. **科研结论需要更强的结构表达**  
   比如方法、任务、指标、数据集、结果，这些都需要单独建模。

PaperGraph-Agent 的思路是：

- 用 `Chunk` 作为向量检索入口
- 用 `Claim / Evidence / Entity / Result` 等节点表达科研语义
- 用 Neo4j 把这些语义连接起来
- 用 LangChain / LangGraph 把检索和问答组织成工作流

---

## 3. 项目目录结构

核心代码都在 `src/paperagent/` 下：

```text
src/paperagent/
├── agent/
│   └── workflow.py
├── extraction/
│   ├── prompts.py
│   └── service.py
├── graph/
│   ├── repository.py
│   ├── schema.py
│   └── utils.py
├── ingestion/
│   ├── parser.py
│   ├── pipeline.py
│   └── workflow.py
├── providers/
│   ├── base.py
│   ├── dashscope.py
│   └── factory.py
├── retrieval/
│   ├── local.py
│   └── retriever.py
├── cli.py
├── config.py
└── schemas.py
```

你可以把这些目录按职责记住：

- `config.py`：配置中心
- `schemas.py`：数据结构中心
- `ingestion/`：导入链路
- `extraction/`：LLM 抽取链路
- `graph/`：Neo4j 图谱持久化与查询
- `retrieval/`：检索层
- `agent/`：问答 Agent 层
- `cli.py`：命令行入口

---

## 4. 从命令行入口开始：程序如何启动

这个项目的用户入口在 [cli.py](/E:/Study/python/PaperAgent/src/paperagent/cli.py)。

核心命令有四个：

```python
@app.command()
def doctor() -> None:
    ...

@schema_app.command("init")
def init_schema(...):
    ...

@app.command()
def ingest(...):
    ...

@app.command()
def query(...):
    ...
```

它们分别对应：

- `doctor`：检查依赖、API key、Neo4j 连通性
- `schema init`：初始化 Neo4j 约束和索引
- `ingest`：导入 PDF
- `query`：发起问答

所以从用户视角，项目的主流程其实就是三条命令：

```bash
uv run paperagent schema init --embedding-dimensions 1024
uv run paperagent ingest --input papers
uv run paperagent query "总结这篇论文的核心方法和结果"
```

后面整个项目的内部流程，本质上都是在支撑这三件事。

---

## 5. 配置系统：`.env` 是如何驱动整个项目的

配置入口在 [config.py](/E:/Study/python/PaperAgent/src/paperagent/config.py)。

核心代码：

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: Literal["dashscope"] = "dashscope"
    dashscope_api_key: str | None = None
    chat_model: str = "qwen3.5-flash"
    embedding_model: str = "text-embedding-v4"
    paperagent_enable_thinking: bool = False

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"
```

这里用了 `pydantic-settings`，好处是：

1. 自动从 `.env` 读取配置
2. 有类型检查
3. 有默认值
4. 代码里拿到的是一个统一的 `Settings` 对象

后面所有模块都会通过：

```python
settings = get_settings()
```

拿到同一份配置。

### 重点配置项

#### 模型相关

- `DASHSCOPE_API_KEY`
- `CHAT_MODEL`
- `EMBEDDING_MODEL`
- `PAPERAGENT_ENABLE_THINKING`

#### Neo4j 相关

- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE`

#### 运行策略相关

- `PAPERAGENT_CHUNK_SIZE`
- `PAPERAGENT_CHUNK_OVERLAP`
- `PAPERAGENT_MAX_CONCURRENCY`
- `PAPERAGENT_ENTITY_MATCH_THRESHOLD`

### `PAPERAGENT_MAX_CONCURRENCY` 是什么

它控制的是：

> 一次最多同时有多少个 Chunk 并发调用 LLM 做抽取。

这个变量直接影响：

- 速度
- 模型接口稳定性
- 并发连接压力

---

## 6. 核心数据模型：项目内部到底在传什么对象

如果你要真正读懂这个项目，最应该先吃透的文件是 [schemas.py](/E:/Study/python/PaperAgent/src/paperagent/schemas.py)。

这个文件定义了整个项目的“内部语言”。

### 6.1 文档结构对象

```python
class ParsedChunk(BaseModel):
    chunk_id: str
    text: str
    section_title: str | None = None
    page_number: int | None = None
    order: int
    metadata: dict[str, Any] = Field(default_factory=dict)
```

```python
class ParsedSection(BaseModel):
    title: str
    order: int
    chunks: list[ParsedChunk] = Field(default_factory=list)
```

```python
class ParsedDocument(BaseModel):
    paper_id: str
    source_path: Path
    title: str | None = None
    sections: list[ParsedSection] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
```

这三者分别表示：

- `ParsedDocument`：一篇解析后的论文
- `ParsedSection`：论文中的一个章节
- `ParsedChunk`：一个可用于抽取和检索的文本块

### 6.2 科研语义对象

```python
class Evidence(BaseModel):
    text: str
    chunk_id: str
    page_number: int | None = None
```

```python
class Claim(BaseModel):
    statement: str
    entity_names: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
```

```python
class Objective(BaseModel): ...
class Approach(BaseModel): ...
class Result(BaseModel): ...
class Constraint(BaseModel): ...
```

这些对象对应的是论文中真正有科研意义的内容：

- `Objective`：研究目标
- `Approach`：方法路线
- `Result`：实验结果
- `Constraint`：限制条件
- `Claim`：一个可被支持或检验的主张
- `Evidence`：支撑主张的证据

### 6.3 实体对象

```python
class EntityType(StrEnum):
    METHOD = "Method"
    DATASET = "Dataset"
    METRIC = "Metric"
    TASK = "Task"
    MODEL = "Model"
    CONCEPT = "PaperConcept"
```

```python
class Entity(BaseModel):
    name: str
    type: EntityType = EntityType.CONCEPT
    canonical_name: str | None = None
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None
```

为什么要把实体单独建出来？

因为实体是**跨论文连接点**。

例如：

- `HotpotQA`：数据集
- `F1`：指标
- `GraphRAG`：方法
- `multi-hop QA`：任务

如果没有实体层，论文之间就只能靠文本相似度勉强连接。

### 6.4 抽取结果对象

```python
class ChunkExtraction(BaseModel):
    chunk_id: str
    objectives: list[Objective] = Field(default_factory=list)
    approaches: list[Approach] = Field(default_factory=list)
    results: list[Result] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
```

```python
class PaperExtraction(BaseModel):
    paper_id: str
    title: str | None = None
    chunks: list[ChunkExtraction] = Field(default_factory=list)
```

可以把它们理解成：

- `ChunkExtraction`：某个 Chunk 的抽取结果
- `PaperExtraction`：整篇论文的抽取结果总和

---

## 7. 第一阶段：PDF 解析与 Chunk 构建

这一层的代码主要在 [parser.py](/E:/Study/python/PaperAgent/src/paperagent/ingestion/parser.py)。

### 7.1 `DocumentParser` 在做什么

```python
class DocumentParser:
    def parse_pdf(self, path: Path) -> ParsedDocument:
        ...
        paper_id = self._paper_id(path)
        section = ParsedSection(title="Document", order=0)
        documents = self.load_pdf_documents(path)
        section.chunks = self._documents_to_chunks(documents, paper_id, section.title, str(path))
        return ParsedDocument(...)
```

这一步做了三件事：

1. 生成 `paper_id`
2. 加载 PDF 文本
3. 切成多个 `ParsedChunk`

### 7.2 为什么 `paper_id` 用文件内容哈希

```python
@staticmethod
def _paper_id(path: Path) -> str:
    digest = sha256(path.read_bytes()).hexdigest()[:16]
    return f"paper:{digest}"
```

它不是简单用文件名，而是用 PDF 文件内容的哈希值。

好处：

- 同一个文件重跑时 ID 稳定
- 不依赖文件名是否规范
- 更适合去重和内部引用

### 7.3 PDF 是怎么被加载的

```python
loader = UnstructuredPDFLoader(str(path), mode="elements")
loaded_documents = loader.load()
split_documents = self.splitter.split_documents(loaded_documents)
```

这里用的是：

- `UnstructuredPDFLoader`
- `RecursiveCharacterTextSplitter`

也就是说，它不是自己手写 PDF 解析，而是：

1. 先借助 LangChain 社区 loader 把 PDF 读出来
2. 再用 LangChain 的 splitter 按规则切块

### 7.4 为什么要切 Chunk

因为一篇论文太长，不能一口气喂给 LLM，也不适合直接作为向量检索的最小单元。

所以项目把论文拆成很多 `Chunk`，每个 chunk 都有：

- `chunk_id`
- `text`
- `page_number`
- `order`

这个设计非常关键，因为后面：

- 抽取是按 chunk 做的
- 向量检索也是按 chunk 做的
- 证据追溯仍然回到 chunk

可以说，`Chunk` 是整个系统最重要的枢纽节点之一。

---

## 8. 第二阶段：LLM 结构化抽取

这一层在 [service.py](/E:/Study/python/PaperAgent/src/paperagent/extraction/service.py)。

### 8.1 抽取的输入是什么

```python
inputs = [
    {
        "paper_id": document.paper_id,
        "chunk_id": chunk.chunk_id,
        "page_number": chunk.page_number or "unknown",
        "chunk_text": chunk.text,
    }
    for chunk in document.chunks
]
```

也就是说：

- 每个 chunk 单独做一次抽取
- 抽取时带上 `paper_id`、`chunk_id`、`page_number`

### 8.2 它是怎么调用模型的

```python
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", EXTRACTION_SYSTEM_PROMPT),
        ("human", EXTRACTION_HUMAN_PROMPT),
    ]
)
return prompt | self.chat_provider.get_chat_model() | StrOutputParser()
```

这里已经是标准的 LangChain Runnable 风格：

```text
Prompt -> Model -> Output Parser
```

这比简单 `model.invoke("一大段字符串")` 更清晰，因为：

- Prompt 是显式对象
- Model 是可替换节点
- 输出解析是独立环节

### 8.3 为什么不用 `with_structured_output`

现在这版实现没有直接把模型输出绑定到 `ChunkExtraction`，而是：

1. 先拿到 JSON 字符串
2. `json.loads(...)`
3. 自己做一层规范化
4. 最后再 `ChunkExtraction.model_validate(...)`

原因很现实：

> 模型输出经常会“半结构化正确、细节字段不稳”。

项目之前遇到过很多典型问题：

- 缺少 `chunk_id`
- 用了 `statements` 而不是 `claims`
- `objectives` 返回字符串列表，而不是对象列表
- `entities` 里有空名字或不合法类型

所以现在采用的是**更稳的双层策略**：

- 第一层：Prompt 尽量约束模型输出
- 第二层：代码里自己做规范化和兜底

### 8.4 并发抽取是怎么做的

```python
semaphore = asyncio.Semaphore(self.settings.max_concurrency)

async def extract_one(payload: dict) -> ChunkExtraction:
    async with semaphore:
        raw_output = await self.chain.ainvoke(payload)
        return self._coerce_extraction(raw_output, payload["chunk_id"])

extractions = await asyncio.gather(*(extract_one(payload) for payload in inputs))
```

这段代码很值得理解。

它说明：

- 系统不是顺序一个个 chunk 抽取
- 而是并发调用模型
- 并发数由 `PAPERAGENT_MAX_CONCURRENCY` 控制

这既带来速度提升，也会带来连接稳定性问题。所以前面调试时我们常建议先把并发调到 `1`。

### 8.5 规范化函数为什么很重要

最关键的兜底逻辑在这里：

```python
def _normalize_extraction_payload(self, payload: dict, chunk_id: str) -> dict:
    ...
```

它负责把模型输出“翻译”成项目内部真正需要的结构。

比如：

- 字符串列表 -> 对象列表
- `claim` 字段 -> `statement`
- `methods` -> `method_names`
- `datasets` -> `dataset_names`
- 不认识的实体类型 -> `PaperConcept`
- `Citation` / `Author` -> 直接丢弃

这也是这个项目比“随手调用一下大模型”更工程化的地方：  
**模型输出不是直接相信，而是必须经过结构清洗。**

---

## 9. 第三阶段：LangGraph 编排导入流程

这一层在 [ingestion/workflow.py](/E:/Study/python/PaperAgent/src/paperagent/ingestion/workflow.py)。

### 9.1 为什么要用 LangGraph

因为导入不是一个单函数，而是一条状态流：

```text
parse -> extract -> write
```

这天生适合用图来描述。

### 9.2 实际代码

```python
graph = StateGraph(IngestionState)

graph.add_node("parse", parse)
graph.add_node("extract", extract)
graph.add_node("write", write)
graph.add_edge(START, "parse")
graph.add_edge("parse", "extract")
graph.add_edge("extract", "write")
graph.add_edge("write", END)
return graph.compile()
```

这几行就是导入工作流的“主骨架”。

### 9.3 每个节点的职责

#### `parse`

```python
def parse(state: IngestionState) -> IngestionState:
    document = self.parser.parse_pdf(state["path"])
    document.metadata["collection"] = state.get("collection", "default")
    return {"document": document}
```

负责把 PDF 路径变成 `ParsedDocument`。

#### `extract`

```python
async def extract(state: IngestionState) -> IngestionState:
    extraction = await self.extractor.extract_document(state["document"])
    return {"extraction": extraction}
```

负责把 `ParsedDocument` 变成 `PaperExtraction`。

#### `write`

```python
def write(state: IngestionState) -> IngestionState:
    self.graph.write_document(state["document"], state["extraction"])
    return {"paper_id": state["document"].paper_id}
```

负责把前两步得到的对象真正写进 Neo4j。

### 9.4 这层的意义

它的意义不只是“形式上用了 LangGraph”，而是：

- 让导入流程有明确节点边界
- 后续容易插入新节点
- 更适合未来做重试、统计、观察和扩展

比如以后你想加：

- `dedupe`
- `post_validate`
- `community_summary`

都可以自然接在这条图上。

---

## 10. 第四阶段：Neo4j 节点与关系写入

这一层是项目真正落图谱的地方，在 [repository.py](/E:/Study/python/PaperAgent/src/paperagent/graph/repository.py)。

你可以把这个文件理解成：

> **“Python 对象世界”和“图数据库世界”之间的翻译器。**

### 10.1 `write_document` 的职责

```python
def write_document(self, document: ParsedDocument, extraction: PaperExtraction) -> None:
    chunk_embeddings = self.embedding_provider.embed_documents([chunk.text for chunk in document.chunks])
    ...
    with self.driver.session(...) as session:
        session.execute_write(self._write_document_tx, document, chunk_embedding_by_id)
        session.execute_write(self._write_extraction_tx, extraction)
```

这一步做了两件大事：

1. 先给所有 `Chunk` 生成 embedding
2. 再分别写：
   - 文档结构层
   - 抽取语义层

### 10.2 文档结构层是怎么写入的

#### `Paper`

```python
MERGE (paper:Paper {paper_id: $paper_id})
SET paper.title = $title,
    paper.normalized_title = $normalized_title,
    paper.source_path = $source_path,
    paper.collection = coalesce($collection, 'default')
```

这里说明 `Paper` 节点的核心主键是 `paper_id`。

同时保存：

- `title`
- `normalized_title`
- `source_path`
- `collection`

#### `Section`

```python
MERGE (section:Section {section_id: $section_id})
SET section.title = $title, section.order = $order
MERGE (paper)-[:HAS_SECTION]->(section)
```

#### `Chunk`

```python
MERGE (chunk:Chunk {chunk_id: $chunk_id})
SET chunk.text = $text,
    chunk.order = $order,
    chunk.page_number = $page_number,
    chunk.paper_id = $paper_id,
    chunk.paper_title = $paper_title,
    chunk.collection = $collection,
    chunk.embedding = $embedding
MERGE (section)-[:HAS_CHUNK]->(chunk)
```

这一步非常重要，因为 `Chunk` 节点不仅保存文本，还保存：

- `embedding`
- `paper_id`
- `collection`

后面的向量检索就是直接打在这里。

### 10.3 语义层是怎么写入的

抽取结果写入在 `_write_extraction_tx(...)`。

它会遍历每个 `ChunkExtraction`：

```python
for chunk_result in extraction.chunks:
    for entity in chunk_result.entities:
        self._merge_entity(tx, entity, chunk_result.chunk_id)
```

然后依次处理：

- `Entity`
- `Objective`
- `Approach`
- `Result`
- `Constraint`
- `Claim`

#### 语义节点 ID 为什么是哈希

```python
def semantic_id(label: str, scope: str, text: str) -> str:
    digest = sha256(f"{label}|{scope}|{text}".encode("utf-8")).hexdigest()[:16]
    return f"{label.lower()}:{digest}"
```

这意味着：

- `Claim` 节点不是随机 ID
- 而是 `label + paper_id + text` 的哈希

好处是：

- 同样的语义描述重跑时尽量稳定
- 有一定去重能力

#### `Evidence` 为什么单独建节点

```python
MERGE (evidence:Evidence {evidence_id: $evidence_id})
SET evidence.text = $text, evidence.page_number = $page_number
MERGE (chunk)-[:HAS_EVIDENCE]->(evidence)
MERGE (evidence)-[:SUPPORTS]->(supported)
```

这一步表达的是：

- 证据来自某个 chunk
- 证据支撑某个语义节点

也就是说，系统不是只存“结论”，而是同时保存“结论背后的证据”。

这对科研场景非常关键。

### 10.4 实体为什么要先按基础 `Entity` 合并

```python
MERGE (entity:Entity {canonical_name: $canonical_name})
...
SET entity:{entity.type.value}
```

这里是之前踩坑之后改出来的关键逻辑。

不能直接：

```cypher
MERGE (entity:Entity:Dataset {canonical_name: "sst-2"})
```

因为如果库里已经有：

```cypher
(:Entity {canonical_name: "sst-2"})
```

但还没挂 `:Dataset` 标签，就可能触发唯一约束冲突。

所以现在采取的是：

1. 先按唯一键合并基础 `Entity`
2. 再补 `Method` / `Dataset` 等具体标签

这个设计很稳，也很值得你记住。

### 10.5 `SAME_AS` 是怎么来的

```python
if fuzz.ratio(canonical_name, other) >= self.settings.entity_match_threshold:
    MERGE (a)-[:SAME_AS {method: 'rapidfuzz'}]-(b)
```

项目当前的实体对齐是一个简化版：

- 用 `canonicalize_entity(...)` 做标准化
- 再用 `rapidfuzz` 做模糊匹配
- 相似度过阈值就建 `SAME_AS`

注意，这里目前不是直接自动合并实体，而是**保守地先建 `SAME_AS` 关系**。

---

## 11. 图谱 Schema 初始化：为什么要先 `schema init`

相关代码在 [graph/schema.py](/E:/Study/python/PaperAgent/src/paperagent/graph/schema.py)。

### 11.1 普通约束和索引

```python
statements = [
    "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT evidence_id IF NOT EXISTS FOR (e:Evidence) REQUIRE e.evidence_id IS UNIQUE",
    "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.canonical_name IS UNIQUE",
    "CREATE INDEX paper_title IF NOT EXISTS FOR (p:Paper) ON (p.title)",
    "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
]
```

这些是图谱的“地基”：

- 唯一约束防重复
- 普通索引加快匹配

### 11.2 向量索引

```python
CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
FOR (c:Chunk) ON (c.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 1024,
  `vector.similarity_function`: 'cosine'
}}
```

这是为什么我们运行前通常要执行：

```bash
paperagent schema init --embedding-dimensions 1024
```

因为 Neo4j 的向量索引必须提前知道向量维度。

如果你的 embedding 模型输出是 1024 维，这里就必须写 1024。

---

## 12. 第五阶段：向量检索与 Local GraphRAG

这一层主要在两个文件：

- [retrieval/retriever.py](/E:/Study/python/PaperAgent/src/paperagent/retrieval/retriever.py)
- [retrieval/local.py](/E:/Study/python/PaperAgent/src/paperagent/retrieval/local.py)

### 12.1 `LocalGraphRetriever`

```python
class LocalGraphRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str) -> list[Document]:
        hits = self.graph.local_search(query, collection=self.collection, top_k=self.top_k)
        self.last_hits = hits
        return [
            Document(
                page_content=hit.text,
                metadata={
                    "id": hit.id,
                    "source": hit.source,
                    "score": hit.score,
                    **hit.metadata,
                },
            )
            for hit in hits
        ]
```

这个类的意义是：

> 把 Neo4j 的本地向量搜索包装成一个标准 LangChain `Retriever`。

这样后面它就可以被：

- `create_retriever_tool(...)`
- LangChain Agent
- 其他 Runnable

直接消费。

### 12.2 `GraphRepository.local_search`

```python
rows = self.get_vector_store().similarity_search_with_score(
    question,
    k=top_k,
    filter={"collection": collection},
)
```

这里真正调的是 `langchain-neo4j` 的 `Neo4jVector`。

这说明：

- 检索不是自己手写 embedding 相似度逻辑
- 而是正式接到了 LangChain 生态里的向量存储抽象

### 12.3 `LocalGraphRAG`

```python
hits = self.graph.local_search(question, collection=collection, top_k=top_k)
evidence = "\n\n".join(
    f"[{hit.id}] score={hit.score:.3f}\n{hit.text}" for hit in hits
)
answer = self.answer_chain.invoke(...)
```

`LocalGraphRAG` 是一个更朴素的问答服务：

1. 先向量召回 chunk
2. 再把召回结果拼成 evidence
3. 然后喂给回答链生成答案

你可以把它理解成“最小 GraphRAG 问答服务”。

---

## 13. 第六阶段：LangChain Agent / LangGraph 查询链路

真正的 Agent 查询逻辑在 [agent/workflow.py](/E:/Study/python/PaperAgent/src/paperagent/agent/workflow.py)。

### 13.1 Agent 是怎么创建的

```python
return create_agent(
    model=self.rag.chat_provider.get_chat_model(),
    tools=[retriever_tool, query_graph, cross_ref],
    system_prompt=system_prompt,
)
```

这里很关键，它说明：

- 现在不是我们手搓一个“伪 Agent”
- 而是走 LangChain 官方推荐的 `create_agent`
- 它底层运行在 LangGraph-backed runtime 上

### 13.2 三个核心工具

#### 1. `vector_match`

```python
retriever_tool = create_retriever_tool(
    retriever,
    name="vector_match",
    description="Retrieve semantically relevant paper chunks ..."
)
```

这个工具让 Agent 可以调用向量检索。

#### 2. `query_graph`

```python
@tool("query_graph")
def query_graph(cypher: str, params_json: str = "{}") -> str:
    rows = self.rag.graph.run_cypher(cypher, params=params)
```

这个工具让 Agent 可以直接跑 Cypher。

#### 3. `cross_ref`

```python
@tool("cross_ref")
def cross_ref(entity_name: str, collection: str = "default", limit: int = 5) -> str:
    rows = self.rag.graph.cross_reference(entity_name, collection=collection, limit=limit)
```

这个工具让 Agent 可以做跨论文实体验证。

### 13.3 查询时的实际路径

当用户执行：

```bash
paperagent query "总结这篇论文的核心方法和结果"
```

内部大致发生的是：

```text
CLI
-> ResearchAgent.invoke()
-> LangChain agent
-> vector_match 检索相关 chunk
-> 必要时 query_graph / cross_ref
-> 模型整合结果
-> 输出答案 + evidence
```

所以，这个项目的查询已经不再是“单函数问答”，而是：

> **Agent 驱动的工具式问答。**

---

## 14. 为什么说这个项目是“图谱 + 向量混合检索”

这点很容易在简历里写，但不容易真正讲明白。

它的“混合”体现在：

### 向量部分

先通过 `Neo4jVector` 在 `Chunk.embedding` 上做相似度搜索，召回相关文本块。

### 图谱部分

被召回的 chunk 不是终点。系统还会利用图谱中的：

- `MENTIONS`
- `HAS_EVIDENCE`
- `SUPPORTS`
- `ABOUT`
- `USES_METHOD`
- `EVALUATED_ON`

这些结构化关系，继续扩展语义上下文。

### Agent 部分

Agent 还能主动调用：

- `query_graph`
- `cross_ref`

进行更明确的图谱推理。

所以它不是：

```text
只有向量检索
```

也不是：

```text
只有图查询
```

而是：

```text
向量召回入口 + 图谱关系扩展 + Agent 工具整合
```

这才是“混合检索”的真正含义。

---

## 15. 一次完整运行到底发生了什么

现在我们把整个流程串起来，看一次 `ingest + query` 的全过程。

### 15.1 执行导入

```bash
uv run paperagent ingest --input papers
```

内部步骤：

1. CLI 调用 `IngestionPipeline`
2. `IngestionPipeline` 遍历 PDF
3. 先根据标题做增量跳过判断
4. 对新论文启动 `IngestionWorkflow`
5. `parse`：PDF -> `ParsedDocument`
6. `extract`：`ParsedDocument` -> `PaperExtraction`
7. `write`：写入 `Paper / Section / Chunk / Claim / Evidence / Entity ...`

### 15.2 执行查询

```bash
uv run paperagent query "总结这篇论文的核心方法和结果"
```

内部步骤：

1. CLI 创建 `GraphRepository`
2. 创建 `LocalGraphRAG`
3. 创建 `ResearchAgent`
4. Agent 接收问题
5. 调用 `vector_match`
6. 如有需要调用 `query_graph` 或 `cross_ref`
7. 返回答案与证据列表

### 15.3 为什么输出里会有 evidence 表格

因为 CLI 里明确把 `answer.evidence` 渲染成 `Rich Table`：

```python
for hit in answer.evidence:
    table.add_row(hit.id, f"{hit.score:.3f}", hit.source)
```

所以你看到的蓝色数字，大概率就是分数列的终端高亮，而不是模型输出的奇怪标记。

---

## 16. 关键设计选择与原因

这一节很重要，因为它解释的是“为什么这么写”，不是“它写了什么”。

### 16.1 为什么把 Chunk 作为最小检索单元

因为 Chunk 同时具备：

- 向量表示
- 原文追溯能力
- 后续抽取语义的上下文边界

它是最自然的桥梁。

### 16.2 为什么 `Evidence` 要单独成节点

因为科研场景里，“主张”和“证据”不能混成一团。

- `Claim`：表达“说了什么”
- `Evidence`：表达“根据什么说”

这有助于未来做：

- 证据核验
- 反思式 RAG
- 冲突检测

### 16.3 为什么实体不直接全部自动合并

因为自动 `MERGE` 风险很高。

像：

- `sst2`
- `sst-2`
- `Stanford Sentiment Treebank 2`

有时是同一个，有时上下文不同。

所以当前项目采取的是保守策略：

- 先规范化
- 再模糊匹配
- 先建 `SAME_AS`
- 不做激进合并

### 16.4 为什么导入流程用 LangGraph

因为导入天生是一个阶段化状态流，比单个 service 函数更适合图来表达。

### 16.5 为什么查询流程要用 Agent

因为科研问答不是固定模板。

有时候用户问题只需要向量检索；
有时候需要图谱聚合；
有时候需要跨论文对照。

Agent + tools 的结构比单链式问答更灵活。

---

## 17. 当前实现的边界与可改进方向

虽然项目已经能全流程跑通，但你要清楚它现在还是 MVP。

### 17.1 当前已经完成的

- PDF -> Chunk -> 抽取 -> 图谱写入
- Neo4j 向量索引与图谱查询
- LangChain Retriever
- LangChain Agent + LangGraph-backed runtime
- 增量导入
- 端到端 CLI

### 17.2 当前仍有边界

#### 1. PDF 解析偏重

`UnstructuredPDFLoader(mode="elements")` 对高保真解析有帮助，但速度较慢。

#### 2. 抽取鲁棒性还在工程化中

模型输出不稳定时，需要大量 normalization 兜底。

#### 3. 实体对齐还是 MVP 级别

目前主要是：

- 标准化
- RapidFuzz
- `SAME_AS`

还没有做更强的语义向量对齐。

#### 4. 全局社区检索还没真正做起来

比如 Leiden / GDS 全局图社区检索，目前只是后续方向。

#### 5. 评估体系还不完整

还没有把 RAGAS、LangSmith 等完整接上。

---

## 18. 建议的阅读顺序

如果你接下来想“真正学会整个项目”，我建议按下面顺序读，而不是随机翻。

### 第一遍：建立整体感觉

1. [README.md](/E:/Study/python/PaperAgent/README.md)
2. [cli.py](/E:/Study/python/PaperAgent/src/paperagent/cli.py)
3. [config.py](/E:/Study/python/PaperAgent/src/paperagent/config.py)

目标：知道项目能做什么、入口在哪里。

### 第二遍：吃透数据结构

1. [schemas.py](/E:/Study/python/PaperAgent/src/paperagent/schemas.py)
2. [graph/utils.py](/E:/Study/python/PaperAgent/src/paperagent/graph/utils.py)

目标：知道系统内部到底在传哪些对象、ID 怎么生成。

### 第三遍：看导入链路

1. [parser.py](/E:/Study/python/PaperAgent/src/paperagent/ingestion/parser.py)
2. [service.py](/E:/Study/python/PaperAgent/src/paperagent/extraction/service.py)
3. [workflow.py](/E:/Study/python/PaperAgent/src/paperagent/ingestion/workflow.py)
4. [repository.py](/E:/Study/python/PaperAgent/src/paperagent/graph/repository.py)

目标：吃透从 PDF 到 Neo4j 的全过程。

### 第四遍：看查询链路

1. [retriever.py](/E:/Study/python/PaperAgent/src/paperagent/retrieval/retriever.py)
2. [local.py](/E:/Study/python/PaperAgent/src/paperagent/retrieval/local.py)
3. [agent/workflow.py](/E:/Study/python/PaperAgent/src/paperagent/agent/workflow.py)

目标：看懂向量检索、图谱查询和 Agent 是如何组合的。

### 第五遍：边跑边看

建议你一边执行命令，一边对照代码：

```bash
uv run paperagent doctor
uv run paperagent schema init --embedding-dimensions 1024
uv run paperagent ingest --input papers
uv run paperagent query "总结这篇论文的核心方法和结果"
```

这时候你会对“哪段代码在什么时候起作用”有非常清晰的感受。

---

## 最后一句

如果把这份文档压成一句话来收尾，我会这样说：

> **PaperGraph-Agent 的本质，是一个以 LangChain/LangGraph 为流程骨架、以 Neo4j 为知识底座、以 Chunk/Claim/Evidence/Entity 为核心建模对象的科研文献 GraphRAG 系统。**

它的重点不在“让模型总结论文”，而在于：

- 把论文变成结构化知识
- 把知识组织成图谱
- 把图谱接上检索和 Agent
- 让问答具备证据和可追溯性

这就是它真正的工程价值。

