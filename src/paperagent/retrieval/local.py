from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from paperagent.extraction.prompts import ANSWER_HUMAN_PROMPT, ANSWER_SYSTEM_PROMPT
from paperagent.graph import GraphRepository
from paperagent.providers import ChatProvider, get_chat_provider
from paperagent.schemas import QueryAnswer


class LocalGraphRAG:
    """本地 GraphRAG 查询服务。

    它是一个较轻的问答层：先做向量召回，再把证据拼给回答链。
    更灵活的工具调用式问答则由上层 ResearchAgent 负责。
    """

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
        """执行一次本地 GraphRAG 问答。"""
        hits = self.graph.local_search(question, collection=collection, top_k=top_k)
        # 把召回证据拼成一个明确的文本块，方便回答链直接引用 chunk id。
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
        """构建回答阶段使用的 LangChain Runnable。"""
        # 这里仍然采用标准 Runnable 组合，方便以后替换 Prompt 或模型。
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ANSWER_SYSTEM_PROMPT),
                ("human", ANSWER_HUMAN_PROMPT),
            ]
        )
        return prompt | self.chat_provider.get_chat_model() | StrOutputParser()
