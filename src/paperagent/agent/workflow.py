import json
from collections import defaultdict

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware, wrap_tool_call
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.tools.retriever import create_retriever_tool

from paperagent.retrieval import LocalGraphRAG
from paperagent.retrieval import LocalGraphRetriever
from paperagent.schemas import QueryAnswer


class ResearchAgent:
    """面向最终问答的研究助理 Agent。"""

    def __init__(self, rag: LocalGraphRAG) -> None:
        self.rag = rag
        self._latest_hits = []
        self._retriever: LocalGraphRetriever | None = None
        self._tool_failures: dict[str, int] = defaultdict(int)
        self.agent = self._build_agent()

    def invoke(self, question: str, collection: str = "default") -> QueryAnswer:
        """对外执行一次问答，并返回答案与证据列表。"""
        self._latest_hits = []
        self._tool_failures = defaultdict(int)
        if self._retriever is not None:
            self._retriever.collection = collection
        result = self.agent.invoke(
            {
                "messages": [
                    (
                        "user",
                        f"Collection: {collection}\nQuestion: {question}",
                    )
                ]
            }
        )
        messages = result["messages"]
        if self._retriever is not None:
            self._latest_hits = list(self._retriever.last_hits)
        final_text = str(messages[-1].content)
        return QueryAnswer(question=question, answer=final_text, evidence=self._latest_hits)

    def _build_agent(self):
        """构建 LangChain Agent，并注册检索与图查询工具。"""
        retriever = LocalGraphRetriever(
            graph=self.rag.graph,
            collection="default",
            top_k=6,
        )
        self._retriever = retriever

        retriever_tool = create_retriever_tool(
            retriever,
            name="vector_match",
            description=(
                "Retrieve semantically relevant paper chunks from the Neo4j vector index. "
                "Use this first when you need source-grounded evidence."
            ),
            document_prompt=None,
            document_separator="\n\n",
        )

        @tool("query_graph")
        def query_graph(cypher: str, params_json: str = "{}") -> str:
            """Run a read-only Cypher query against Neo4j. params_json must be a JSON object string."""

            params = json.loads(params_json)
            rows = self.rag.graph.run_cypher(cypher, params=params)
            if not rows:
                return "No graph rows returned."
            return json.dumps(rows[:10], ensure_ascii=False, indent=2, default=str)

        @tool("cross_ref")
        def cross_ref(entity_name: str, collection: str = "default", limit: int = 5) -> str:
            """Find how one entity appears across papers, chunks, and evidence."""

            rows = self.rag.graph.cross_reference(entity_name, collection=collection, limit=limit)
            if not rows:
                return f"No cross references found for {entity_name}."
            return json.dumps(rows, ensure_ascii=False, indent=2, default=str)

        system_prompt = """You are PaperGraph-Agent, a research assistant built with LangChain agents on top of LangGraph.

Use tools before answering.
- Use vector_match to retrieve evidence chunks.
- Use query_graph to inspect graph structure or aggregate facts.
- Use cross_ref to compare an entity across papers.

Rules:
- Ground every important claim in tool output.
- Cite chunk ids like [paper:...:chunk:3] when available.
- If evidence is insufficient, say so explicitly.
- Prefer concise synthesis over generic explanation.
"""

        def validate_read_only_cypher(cypher: str) -> None:
            """限制模型只能执行只读 Cypher，避免写操作进入数据库。"""

            banned_clauses = [
                "CREATE",
                "MERGE",
                "DELETE",
                "DETACH DELETE",
                "SET ",
                "DROP ",
                "REMOVE ",
                "LOAD CSV",
                "CALL DBMS",
            ]
            upper = cypher.upper()
            for clause in banned_clauses:
                if clause in upper:
                    raise ValueError(f"Disallowed Cypher clause: {clause}")

        @wrap_tool_call
        def guard_tool_call(request, handler):
            """统一处理工具调用前后的校验、限次和异常兜底。

            这是一个贴合本项目的 wrap_tool_call 示例：
            1. 对 query_graph 做只读 Cypher 校验；
            2. 校验 params_json 是否是合法 JSON；
            3. 统计单个工具失败次数；
            4. 把工具异常转成 ToolMessage 返回给模型，而不是直接炸掉整个 Agent。
            """

            tool_name = request.tool_call.get("name", "unknown")
            tool_call_id = request.tool_call["id"]
            tool_args = request.tool_call.get("args", {})

            # 在真正执行前，先针对高风险工具做额外校验。
            if tool_name == "query_graph":
                cypher = str(tool_args.get("cypher", "")).strip()
                params_json = str(tool_args.get("params_json", "{}")).strip()
                try:
                    validate_read_only_cypher(cypher)
                    parsed_params = json.loads(params_json)
                    if not isinstance(parsed_params, dict):
                        raise ValueError("params_json must decode to a JSON object.")
                except Exception as exc:  # noqa: BLE001
                    self._tool_failures[tool_name] += 1
                    return ToolMessage(
                        content=(
                            f"Tool {tool_name} validation failed: {type(exc).__name__}: {exc}. "
                            "Use a simpler read-only MATCH/RETURN query, or fall back to vector_match / cross_ref."
                        ),
                        tool_call_id=tool_call_id,
                        status="error",
                    )

            # 同一个工具如果已经失败过太多次，就不再允许继续试错。
            if self._tool_failures[tool_name] >= self.rag.graph.settings.max_tool_failures:
                return ToolMessage(
                    content=(
                        f"Tool {tool_name} is disabled for this run because it has already failed "
                        f"{self._tool_failures[tool_name]} time(s). Use the evidence you already have "
                        "or switch to another tool."
                    ),
                    tool_call_id=tool_call_id,
                    status="error",
                )

            try:
                return handler(request)
            except Exception as exc:  # noqa: BLE001
                self._tool_failures[tool_name] += 1
                return ToolMessage(
                    content=(
                        f"Tool {tool_name} failed: {type(exc).__name__}: {str(exc)[:300]}. "
                        "Try a narrower query, choose another tool, or answer conservatively with current evidence."
                    ),
                    tool_call_id=tool_call_id,
                    status="error",
                )

        # create_agent 是 LangChain 当前推荐入口，底层运行时由 LangGraph 支撑。
        return create_agent(
            model=self.rag.chat_provider.get_chat_model(),
            tools=[retriever_tool, query_graph, cross_ref],
            system_prompt=system_prompt,
            middleware=[
                # 官方现成 middleware：限制单次运行里的总工具调用次数。
                ToolCallLimitMiddleware(run_limit=self.rag.graph.settings.max_tool_calls),
                # 项目级 middleware：做只读校验、错误捕获和失败次数控制。
                guard_tool_call,
            ],
        )
