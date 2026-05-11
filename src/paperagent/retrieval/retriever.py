from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
from pydantic import ConfigDict

from paperagent.graph import GraphRepository


class LocalGraphRetriever(BaseRetriever):
    """基于 Neo4j 向量检索的 LangChain Retriever。"""

    graph: GraphRepository
    collection: str = "default"
    top_k: int = 6
    last_hits: list = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str) -> list[Document]:
        """执行一次检索，并把结果转换成 LangChain Document 列表。"""
        hits = self.graph.local_search(query, collection=self.collection, top_k=self.top_k)
        # last_hits 会被 Agent 层取走，用于在最终答案里附带 evidence 列表。
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
