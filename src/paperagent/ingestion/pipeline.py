import asyncio
from pathlib import Path

from paperagent.graph import GraphRepository
from paperagent.graph.utils import normalize_title
from paperagent.ingestion.workflow import IngestionWorkflow


class IngestionPipeline:
    """Path-level orchestrator over the LangGraph ingestion workflow."""

    def __init__(
        self,
        graph: GraphRepository | None = None,
    ) -> None:
        self.graph = graph or GraphRepository()
        self.workflow = IngestionWorkflow(graph=self.graph)

    async def ingest_path(self, input_path: Path, collection: str = "default") -> tuple[int, int]:
        paths = [input_path] if input_path.is_file() else sorted(input_path.glob("*.pdf"))
        ingested = 0
        skipped = 0
        for path in paths:
            candidate_title = normalize_title(path.stem)
            if self.graph.paper_exists(candidate_title, collection=collection):
                skipped += 1
                continue
            await self.workflow.ainvoke(path, collection=collection)
            ingested += 1
        return ingested, skipped

    def ingest_path_sync(self, input_path: Path, collection: str = "default") -> tuple[int, int]:
        return asyncio.run(self.ingest_path(input_path, collection=collection))
