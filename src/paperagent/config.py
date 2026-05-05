from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: Literal["dashscope"] = "dashscope"
    dashscope_api_key: str | None = None
    chat_model: str = "qwen3.5-flash"
    embedding_model: str = "text-embedding-v4"
    paperagent_enable_thinking: bool = False

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"

    paperagent_chunk_size: int = Field(default=1200, ge=200)
    paperagent_chunk_overlap: int = Field(default=160, ge=0)
    paperagent_max_concurrency: int = Field(default=4, ge=1, le=32)
    paperagent_entity_match_threshold: int = Field(default=90, ge=0, le=100)
    paperagent_vector_match_threshold: float = Field(default=0.9, ge=0.0, le=1.0)

    @property
    def chunk_size(self) -> int:
        return self.paperagent_chunk_size

    @property
    def chunk_overlap(self) -> int:
        return self.paperagent_chunk_overlap

    @property
    def max_concurrency(self) -> int:
        return self.paperagent_max_concurrency

    @property
    def enable_thinking(self) -> bool:
        return self.paperagent_enable_thinking

    @property
    def entity_match_threshold(self) -> int:
        return self.paperagent_entity_match_threshold

    @property
    def vector_match_threshold(self) -> float:
        return self.paperagent_vector_match_threshold


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()
    return Settings()
