__all__ = ["GraphRepository", "GraphSchemaManager"]


def __getattr__(name: str):
    if name == "GraphRepository":
        from paperagent.graph.repository import GraphRepository

        return GraphRepository
    if name == "GraphSchemaManager":
        from paperagent.graph.schema import GraphSchemaManager

        return GraphSchemaManager
    raise AttributeError(name)
