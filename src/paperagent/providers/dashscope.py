from pydantic import BaseModel

from paperagent.config import Settings
from paperagent.providers.base import SchemaT


class DashScopeChatProvider:
    """DashScope 聊天模型适配器。

    DashScope 提供了 OpenAI-compatible 接口，因此这里直接复用 LangChain 的
    ChatOpenAI 客户端，而不是自己重新封装 HTTP 调用。
    """

    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(self, settings: Settings) -> None:
        """初始化 DashScope 聊天模型客户端。"""
        if not settings.dashscope_api_key:
            raise ValueError("DASHSCOPE_API_KEY is required for the dashscope provider.")

        from langchain_openai import ChatOpenAI

        self._model = ChatOpenAI(
            api_key=settings.dashscope_api_key,
            base_url=self.base_url,
            model=settings.chat_model,
            # thinking 模式会显著影响延迟；这里通过配置显式控制。
            extra_body={"enable_thinking": settings.enable_thinking},
        )

    def extract_structured(self, prompt: str, schema: type[SchemaT]) -> SchemaT:
        """调用模型并要求其直接返回结构化对象。"""
        structured_model = self._model.with_structured_output(schema)
        result = structured_model.invoke(prompt)
        if isinstance(result, BaseModel):
            return result
        return schema.model_validate(result)

    def generate(self, prompt: str) -> str:
        """调用模型生成纯文本回答。"""
        result = self._model.invoke(prompt)
        return str(result.content)

    def get_chat_model(self):
        """返回底层 LangChain ChatModel，供 Runnable / Agent 直接复用。"""
        return self._model


class DashScopeEmbeddingProvider:
    """DashScope 向量模型适配器。"""

    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    max_batch_size = 10

    def __init__(self, settings: Settings) -> None:
        """初始化 DashScope embedding 客户端。"""
        if not settings.dashscope_api_key:
            raise ValueError("DASHSCOPE_API_KEY is required for the dashscope provider.")

        from langchain_openai import OpenAIEmbeddings

        self._model = OpenAIEmbeddings(
            api_key=settings.dashscope_api_key,
            base_url=self.base_url,
            model=settings.embedding_model,
            # DashScope 与 OpenAI 默认 token 预处理不完全兼容，这里关闭长度安全分词逻辑。
            check_embedding_ctx_length=False,
            tiktoken_enabled=False,
        )

    def embed_query(self, text: str) -> list[float]:
        """为单条查询文本生成向量。"""
        return self._model.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """为多条文档文本生成向量，并自动按 DashScope 限制分批。"""
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.max_batch_size):
            # DashScope embeddings 每批最多 10 条，因此必须手动分批。
            batch = texts[start : start + self.max_batch_size]
            embeddings.extend(self._model.embed_documents(batch))
        return embeddings

    def get_embeddings_model(self):
        """返回底层 LangChain Embeddings 对象。"""
        return self._model
