from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from paperagent.extraction.prompts import ANSWER_HUMAN_PROMPT, ANSWER_SYSTEM_PROMPT
from paperagent.graph import GraphRepository
from paperagent.providers import ChatProvider, get_chat_provider
from paperagent.schemas import QueryAnswer


class LocalGraphRAG:
    """Local GraphRAG query service backed by Neo4j vector search."""

    def __init__(
        self,
        graph: GraphRepository,
        chat_provider: ChatProvider | None = None,
    ) -> None:
        self.graph = graph
        self.chat_provider = chat_provider or get_chat_provider(graph.settings)
        self.answer_chain = self._build_answer_chain()

    def answer(
        self,
        question: str,
        collection: str = "default",
        top_k: int = 6,
        plan: str | None = None,
    ) -> QueryAnswer:
        hits = self.graph.local_search(question, collection=collection, top_k=top_k)
        evidence = "\n\n".join(
            f"[{hit.id}] score={hit.score:.3f}\n{hit.text}" for hit in hits
        )
        answer = self.answer_chain.invoke(
            {
                "question": question,
                "plan": plan or "No explicit plan provided.",
                "evidence": evidence,
            }
        )
        return QueryAnswer(question=question, answer=answer, evidence=hits)

    def _build_answer_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ANSWER_SYSTEM_PROMPT),
                ("human", ANSWER_HUMAN_PROMPT),
            ]
        )
        return prompt | self.chat_provider.get_chat_model() | StrOutputParser()
