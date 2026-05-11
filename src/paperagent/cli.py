from pathlib import Path
import importlib.util

import typer
from rich.console import Console
from rich.table import Table

from paperagent.config import get_settings
from paperagent.agent import ResearchAgent
from paperagent.graph import GraphRepository, GraphSchemaManager
from paperagent.ingestion import IngestionPipeline
from paperagent.retrieval import LocalGraphRAG


app = typer.Typer(help="PaperGraph-Agent GraphRAG MVP CLI.")
schema_app = typer.Typer(help="Neo4j schema commands.")
app.add_typer(schema_app, name="schema")
console = Console()


@app.command()
def doctor() -> None:
    """Check local runtime prerequisites for PaperGraph-Agent."""

    settings = get_settings()
    table = Table(title="PaperGraph-Agent Doctor")
    table.add_column("check")
    table.add_column("status")
    table.add_column("details")

    required_modules = [
        "langchain",
        "langchain_community",
        "langchain_openai",
        "langchain_neo4j",
        "langgraph",
        "neo4j",
        "unstructured",
    ]
    missing_modules = [name for name in required_modules if importlib.util.find_spec(name) is None]
    table.add_row(
        "python deps",
        "ok" if not missing_modules else "missing",
        "all installed" if not missing_modules else ", ".join(missing_modules),
    )
    table.add_row(
        "dashscope key",
        "ok" if bool(settings.dashscope_api_key) else "missing",
        "configured in .env" if bool(settings.dashscope_api_key) else "set DASHSCOPE_API_KEY",
    )

    try:
        # connectivity 检查放在这里做，能尽早发现 URI、用户名或密码问题。
        graph = GraphRepository(settings=settings)
        try:
            graph.driver.verify_connectivity()
        finally:
            graph.close()
    except Exception as exc:  # noqa: BLE001
        table.add_row("neo4j", "unreachable", str(exc).splitlines()[0][:120])
    else:
        table.add_row("neo4j", "ok", f"{settings.neo4j_uri} / {settings.neo4j_database}")

    console.print(table)


@schema_app.command("init")
def init_schema(
    embedding_dimensions: int | None = typer.Option(
        None,
        help="Optional embedding dimension. If omitted, vector indexes are skipped.",
    ),
) -> None:
    """Create Neo4j constraints, indexes, and optional vector indexes."""

    settings = get_settings()
    graph = GraphRepository(settings=settings)
    try:
        manager = GraphSchemaManager(graph.driver, settings.neo4j_database)
        manager.init_schema()
        if embedding_dimensions:
            # 只有显式给出维度时才创建向量索引，避免模型维度不明确时误建索引。
            manager.init_vector_indexes(embedding_dimensions)
        console.print("[green]Neo4j schema initialized.[/green]")
    finally:
        graph.close()


@app.command()
def ingest(
    input: Path = typer.Option(..., "--input", "-i", help="PDF file or directory containing PDFs."),
    collection: str = typer.Option("default", help="Collection name written to Paper nodes."),
) -> None:
    """Parse PDFs, extract scientific knowledge, and write graph data."""

    pipeline = IngestionPipeline()
    try:
        ingested, skipped = pipeline.ingest_path_sync(input, collection=collection)
        console.print(
            f"[green]Ingested {ingested} paper(s) into collection '{collection}'. "
            f"Skipped {skipped} existing paper(s).[/green]"
        )
    finally:
        pipeline.graph.close()


@app.command()
def query(
    question: str = typer.Argument(..., help="Research question to answer."),
    collection: str = typer.Option("default", help="Collection to query."),
) -> None:
    """Answer a question with a LangGraph agent over Neo4j tools."""

    graph = GraphRepository()
    try:
        # query 命令本身很薄，真正的查询逻辑会下沉到 RAG 层和 Agent 层。
        rag = LocalGraphRAG(graph)
        agent = ResearchAgent(rag)
        answer = agent.invoke(question, collection=collection)
        console.print("\n[bold]Answer[/bold]")
        console.print(answer.answer)

        table = Table(title="Evidence")
        table.add_column("id")
        table.add_column("score")
        table.add_column("source")
        for hit in answer.evidence:
            table.add_row(hit.id, f"{hit.score:.3f}", hit.source)
        console.print(table)
    finally:
        graph.close()


@app.command()
def inspect(
    paper_id: str = typer.Option(..., "--paper", help="Paper id to inspect."),
) -> None:
    """Show a small summary for one paper node."""

    graph = GraphRepository()
    try:
        with graph.driver.session(database=graph.settings.neo4j_database) as session:
            row = session.run(
                """
                MATCH (paper:Paper {paper_id: $paper_id})
                OPTIONAL MATCH (paper)-[:HAS_SECTION]->(:Section)-[:HAS_CHUNK]->(chunk:Chunk)
                OPTIONAL MATCH (chunk)-[:MENTIONS]->(entity:Entity)
                RETURN paper.title AS title,
                       count(DISTINCT chunk) AS chunks,
                       count(DISTINCT entity) AS entities
                """,
                paper_id=paper_id,
            ).single()
        if not row:
            console.print(f"[red]Paper not found: {paper_id}[/red]")
            raise typer.Exit(code=1)
        console.print(f"[bold]{row['title']}[/bold]")
        console.print(f"Chunks: {row['chunks']}")
        console.print(f"Entities: {row['entities']}")
    finally:
        graph.close()


if __name__ == "__main__":
    app()
