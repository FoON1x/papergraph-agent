__all__ = ["GraphRepository", "GraphSchemaManager"]


def __getattr__(name: str):
    # 延迟导入，避免在只用到部分模块时提前触发 Neo4j 相关依赖加载。
    if name == "GraphRepository":
        from paperagent.graph.repository import GraphRepository

        return GraphRepository
    if name == "GraphSchemaManager":
        from paperagent.graph.schema import GraphSchemaManager

        return GraphSchemaManager
    raise AttributeError(name)
