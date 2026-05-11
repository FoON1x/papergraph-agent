from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """运行时统一配置。

    这里刻意把所有环境变量都收口到一个对象里，后续模块只依赖 Settings，
    不直接到处读取 os.environ，这样更容易维护、测试和排查问题。
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # 当前使用的模型提供商；MVP 阶段只实现了 dashscope。
    llm_provider: Literal["dashscope"] = "dashscope"
    # DashScope API Key，聊天模型和 embedding 都会用到。
    dashscope_api_key: str | None = None
    # 查询与抽取阶段使用的聊天模型名。
    chat_model: str = "qwen3.5-flash"
    # 写入向量索引时使用的 embedding 模型名。
    embedding_model: str = "text-embedding-v4"
    # 是否开启模型的 thinking / reasoning 模式；关闭通常更快更稳。
    paperagent_enable_thinking: bool = False

    # Neo4j 连接地址。
    neo4j_uri: str = "bolt://localhost:7687"
    # Neo4j 用户名。
    neo4j_user: str = "neo4j"
    # Neo4j 密码。
    neo4j_password: str = "password"
    # 目标数据库名。
    neo4j_database: str = "neo4j"

    # 文本切块大小；越大越省请求次数，但每个 chunk 的语义会更混杂。
    paperagent_chunk_size: int = Field(default=1200, ge=200)
    # 相邻 chunk 的重叠字符数，用于减少切块造成的上下文断裂。
    paperagent_chunk_overlap: int = Field(default=160, ge=0)
    # 抽取阶段最大并发数；越高越快，但更容易触发模型接口不稳定。
    paperagent_max_concurrency: int = Field(default=4, ge=1, le=32)
    # 单次 Agent 查询允许的最大工具调用次数。
    paperagent_max_tool_calls: int = Field(default=4, ge=1, le=20)
    # 单次 Agent 查询中，同一个工具允许失败的最大次数。
    paperagent_max_tool_failures: int = Field(default=1, ge=0, le=10)
    # 实体模糊匹配阈值，供 SAME_AS 候选关系构建使用。
    paperagent_entity_match_threshold: int = Field(default=90, ge=0, le=100)
    # 预留给后续语义对齐 / 向量匹配的相似度阈值。
    paperagent_vector_match_threshold: float = Field(default=0.9, ge=0.0, le=1.0)

    @property
    def chunk_size(self) -> int:
        """返回文本切块大小。"""
        # 对外暴露更简洁的属性名，避免业务代码里到处出现 paperagent_ 前缀。
        return self.paperagent_chunk_size

    @property
    def chunk_overlap(self) -> int:
        """返回文本切块重叠长度。"""
        return self.paperagent_chunk_overlap

    @property
    def max_concurrency(self) -> int:
        """返回抽取阶段的最大并发数。"""
        return self.paperagent_max_concurrency

    @property
    def max_tool_calls(self) -> int:
        """返回单次查询允许的最大工具调用次数。"""
        return self.paperagent_max_tool_calls

    @property
    def max_tool_failures(self) -> int:
        """返回单次查询中同一工具允许失败的最大次数。"""
        return self.paperagent_max_tool_failures

    @property
    def enable_thinking(self) -> bool:
        """返回模型是否启用 thinking 模式。"""
        return self.paperagent_enable_thinking

    @property
    def entity_match_threshold(self) -> int:
        """返回实体模糊匹配阈值。"""
        return self.paperagent_entity_match_threshold

    @property
    def vector_match_threshold(self) -> float:
        """返回向量匹配阈值。"""
        return self.paperagent_vector_match_threshold


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回全局唯一的 Settings 实例。"""
    # 整个进程内通常只需要一份配置，因此这里做缓存，避免重复解析 .env。
    load_dotenv()
    return Settings()
