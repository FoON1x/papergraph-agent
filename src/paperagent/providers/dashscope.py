from pydantic import BaseModel

from paperagent.config import Settings
from paperagent.providers.base import SchemaT


class DashScopeChatProvider:
    """DashScope provider through OpenAI-compatible LangChain clients."""

    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(self, settings: Settings) -> None:
        if not settings.dashscope_api_key:
            raise ValueError("DASHSCOPE_API_KEY is required for the dashscope provider.")

        from langchain_openai import ChatOpenAI

        self._model = ChatOpenAI(
            api_key=settings.dashscope_api_key,
            base_url=self.base_url,
            model=settings.chat_model,
            extra_body={"enable_thinking": settings.enable_thinking},
        )

    def extract_structured(self, prompt: str, schema: type[SchemaT]) -> SchemaT:
        structured_model = self._model.with_structured_output(schema)
        result = structured_model.invoke(prompt)
        if isinstance(result, BaseModel):
            return result
        return schema.model_validate(result)

    def generate(self, prompt: str) -> str:
        result = self._model.invoke(prompt)
        return str(result.content)

    def get_chat_model(self):
        return self._model


class DashScopeEmbeddingProvider:
    """DashScope embeddings through OpenAI-compatible LangChain clients."""

    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    max_batch_size = 10

    def __init__(self, settings: Settings) -> None:
        if not settings.dashscope_api_key:
            raise ValueError("DASHSCOPE_API_KEY is required for the dashscope provider.")

        from langchain_openai import OpenAIEmbeddings

        self._model = OpenAIEmbeddings(
            api_key=settings.dashscope_api_key,
            base_url=self.base_url,
            model=settings.embedding_model,
            check_embedding_ctx_length=False,
            tiktoken_enabled=False,
        )

    def embed_query(self, text: str) -> list[float]:
        return self._model.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.max_batch_size):
            batch = texts[start : start + self.max_batch_size]
            embeddings.extend(self._model.embed_documents(batch))
        return embeddings

    def get_embeddings_model(self):
        return self._model
