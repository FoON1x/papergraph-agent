from pathlib import Path
from typing import TypedDict

from paperagent.extraction import ExtractionService
from paperagent.graph import GraphRepository
from paperagent.ingestion.parser import DocumentParser
from paperagent.schemas import PaperExtraction, ParsedDocument


class IngestionState(TypedDict, total=False):
    """LangGraph 导入工作流在各节点之间传递的状态。"""

    path: Path
    collection: str
    document: ParsedDocument
    extraction: PaperExtraction
    paper_id: str


class IngestionWorkflow:
    """单篇论文的 LangGraph 工作流：parse -> extract -> write。"""

    def __init__(
        self,
        parser: DocumentParser | None = None,
        extractor: ExtractionService | None = None,
        graph: GraphRepository | None = None,
    ) -> None:
        self.parser = parser or DocumentParser()
        self.extractor = extractor or ExtractionService()
        self.graph = graph or GraphRepository()
        self.workflow = self._build_workflow()

    async def ainvoke(self, path: Path, collection: str = "default") -> IngestionState:
        return await self.workflow.ainvoke({"path": path, "collection": collection})

    def invoke(self, path: Path, collection: str = "default") -> IngestionState:
        return self.workflow.invoke({"path": path, "collection": collection})

    def _build_workflow(self):
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(IngestionState)

        def parse(state: IngestionState) -> IngestionState:
            # parse 节点只负责把文件路径变成 ParsedDocument，不做模型调用。
            document = self.parser.parse_pdf(state["path"])
            document.metadata["collection"] = state.get("collection", "default")
            return {"document": document}

        async def extract(state: IngestionState) -> IngestionState:
            # extract 节点是最耗时的一步：会按 chunk 并发调用 LLM。
            extraction = await self.extractor.extract_document(state["document"])
            return {"extraction": extraction}

        def write(state: IngestionState) -> IngestionState:
            # write 节点把文档结构和抽取语义一起写入 Neo4j。
            self.graph.write_document(state["document"], state["extraction"])
            return {"paper_id": state["document"].paper_id}

        graph.add_node("parse", parse)
        graph.add_node("extract", extract)
        graph.add_node("write", write)
        graph.add_edge(START, "parse")
        graph.add_edge("parse", "extract")
        graph.add_edge("extract", "write")
        graph.add_edge("write", END)
        return graph.compile()
