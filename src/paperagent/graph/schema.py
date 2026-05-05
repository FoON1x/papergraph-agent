from neo4j import Driver


class GraphSchemaManager:
    """Create constraints and indexes required by the MVP graph model."""

    def __init__(self, driver: Driver, database: str) -> None:
        self.driver = driver
        self.database = database

    def init_schema(self) -> None:
        statements = [
            "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT evidence_id IF NOT EXISTS FOR (e:Evidence) REQUIRE e.evidence_id IS UNIQUE",
            "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.canonical_name IS UNIQUE",
            "CREATE INDEX paper_title IF NOT EXISTS FOR (p:Paper) ON (p.title)",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
        ]
        with self.driver.session(database=self.database) as session:
            for statement in statements:
                session.run(statement)

    def init_vector_indexes(self, embedding_dimensions: int) -> None:
        statements = [
            f"""
            CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {{indexConfig: {{
              `vector.dimensions`: {embedding_dimensions},
              `vector.similarity_function`: 'cosine'
            }}}}
            """,
            f"""
            CREATE VECTOR INDEX entity_embedding IF NOT EXISTS
            FOR (e:Entity) ON (e.embedding)
            OPTIONS {{indexConfig: {{
              `vector.dimensions`: {embedding_dimensions},
              `vector.similarity_function`: 'cosine'
            }}}}
            """,
        ]
        with self.driver.session(database=self.database) as session:
            for statement in statements:
                session.run(statement)
