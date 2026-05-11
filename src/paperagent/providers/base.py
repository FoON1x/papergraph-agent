from typing import Any, Protocol, TypeVar

from pydantic import BaseModel


SchemaT = TypeVar("SchemaT", bound=BaseModel)


class ChatProvider(Protocol):
    """面向业务层的统一聊天模型接口。"""

    def extract_structured(self, prompt: str, schema: type[SchemaT]) -> SchemaT:
        """Return a Pydantic object extracted from the prompt."""

    def generate(self, prompt: str) -> str:
        """Return plain text from the chat model."""

    def get_chat_model(self) -> Any:
        """Return the underlying chat model for LangChain runnables."""


class EmbeddingProvider(Protocol):
    """面向业务层的统一向量模型接口。"""

    def embed_query(self, text: str) -> list[float]:
        """Embed one query string."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple document strings."""

    def get_embeddings_model(self) -> Any:
        """Return the underlying embeddings model for LangChain vector stores."""


ModelConfig = dict[str, Any]
