import json

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.tools.retriever import create_retriever_tool

from paperagent.retrieval import LocalGraphRAG
from paperagent.retrieval import LocalGraphRetriever
from paperagent.schemas import QueryAnswer


class ResearchAgent:
    """LangChain agent built on LangGraph with retriever and graph tools."""

    def __init__(self, rag: LocalGraphRAG) -> None:
        self.rag = rag
        self._latest_hits = []
        self._retriever: LocalGraphRetriever | None = None
        self.agent = self._build_agent()

    def invoke(self, question: str, collection: str = "default") -> QueryAnswer:
        self._latest_hits = []
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

        return create_agent(
            model=self.rag.chat_provider.get_chat_model(),
            tools=[retriever_tool, query_graph, cross_ref],
            system_prompt=system_prompt,
        )
