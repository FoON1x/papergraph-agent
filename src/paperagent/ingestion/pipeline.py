import asyncio
from pathlib import Path

from paperagent.graph import GraphRepository
from paperagent.graph.utils import normalize_title
from paperagent.ingestion.workflow import IngestionWorkflow


class IngestionPipeline:
    """目录级导入器。

    IngestionWorkflow 只关心“单篇论文怎么导入”，Pipeline 这一层负责：
    - 遍历目录；
    - 做增量跳过判断；
    - 统计成功导入和跳过数量。
    """

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
            # 当前版本先用文件名做论文级去重判断，适合增量维护一个本地文献库。
            candidate_title = normalize_title(path.stem)
            if self.graph.paper_exists(candidate_title, collection=collection):
                skipped += 1
                continue
            await self.workflow.ainvoke(path, collection=collection)
            ingested += 1
        return ingested, skipped

    def ingest_path_sync(self, input_path: Path, collection: str = "default") -> tuple[int, int]:
        # CLI 入口仍是同步风格，因此这里统一包一层 asyncio.run。
        return asyncio.run(self.ingest_path(input_path, collection=collection))
