from paperagent.config import Settings, get_settings
from paperagent.providers.base import ChatProvider, EmbeddingProvider
from paperagent.providers.dashscope import DashScopeChatProvider, DashScopeEmbeddingProvider


def get_chat_provider(settings: Settings | None = None) -> ChatProvider:
    settings = settings or get_settings()
    if settings.llm_provider == "dashscope":
        return DashScopeChatProvider(settings)
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    settings = settings or get_settings()
    if settings.llm_provider == "dashscope":
        return DashScopeEmbeddingProvider(settings)
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
