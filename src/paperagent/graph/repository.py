from neo4j import Driver, GraphDatabase
from rapidfuzz import fuzz

from paperagent.config import Settings, get_settings
from paperagent.graph.utils import canonicalize_entity, normalize_title, semantic_id
from paperagent.providers import EmbeddingProvider, get_embedding_provider
from paperagent.schemas import Entity, Evidence, ParsedDocument, PaperExtraction, RetrievalHit


class GraphRepository:
    """Neo4j persistence and local GraphRAG retrieval."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        driver: Driver | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedding_provider = embedding_provider or get_embedding_provider(self.settings)
        self.driver = driver or GraphDatabase.driver(
            self.settings.neo4j_uri,
            auth=(self.settings.neo4j_user, self.settings.neo4j_password),
        )
        self._lc_graph = None
        self._vector_store = None

    def close(self) -> None:
        self.driver.close()

    def paper_exists(self, title: str, collection: str = "default") -> bool:
        normalized_title = normalize_title(title)
        query = """
        MATCH (paper:Paper)
        WHERE paper.collection = $collection
          AND paper.normalized_title = $normalized_title
        RETURN count(paper) > 0 AS exists
        """
        rows = self.run_cypher(
            query,
            {
                "collection": collection,
                "normalized_title": normalized_title,
            },
        )
        return bool(rows and rows[0].get("exists"))

    def write_document(self, document: ParsedDocument, extraction: PaperExtraction) -> None:
        chunk_embeddings = self.embedding_provider.embed_documents([chunk.text for chunk in document.chunks])
        chunk_embedding_by_id = {
            chunk.chunk_id: chunk_embeddings[index] for index, chunk in enumerate(document.chunks)
        }

        with self.driver.session(database=self.settings.neo4j_database) as session:
            session.execute_write(self._write_document_tx, document, chunk_embedding_by_id)
            session.execute_write(self._write_extraction_tx, extraction)

    def local_search(self, question: str, collection: str = "default", top_k: int = 6) -> list[RetrievalHit]:
        rows = self.get_vector_store().similarity_search_with_score(
            question,
            k=top_k,
            filter={"collection": collection},
        )
        return [
            RetrievalHit(
                id=str(document.metadata.get("id", "")),
                text=document.page_content,
                score=float(score),
                source=str(document.metadata.get("source", "")),
                metadata={k: v for k, v in document.metadata.items() if k not in {"id", "source"}},
            )
            for document, score in rows
        ]

    def run_cypher(self, query: str, params: dict | None = None) -> list[dict]:
        return self.get_langchain_graph().query(query, params=params or {})

    def cross_reference(self, entity_name: str, collection: str = "default", limit: int = 5) -> list[dict]:
        query = """
        MATCH (paper:Paper {collection: $collection})-[:HAS_SECTION]->(:Section)-[:HAS_CHUNK]->(chunk:Chunk)-[:MENTIONS]->(entity:Entity)
        WHERE entity.canonical_name = $canonical_name OR toLower(entity.name) = toLower($entity_name)
        OPTIONAL MATCH (chunk)-[:HAS_EVIDENCE]->(evidence:Evidence)
        RETURN paper.paper_id AS paper_id,
               paper.title AS title,
               chunk.chunk_id AS chunk_id,
               left(chunk.text, 500) AS snippet,
               collect(DISTINCT evidence.text)[0..2] AS evidence
        LIMIT $limit
        """
        return self.run_cypher(
            query,
            {
                "collection": collection,
                "canonical_name": canonicalize_entity(entity_name),
                "entity_name": entity_name,
                "limit": limit,
            },
        )

    def _write_document_tx(self, tx, document: ParsedDocument, chunk_embedding_by_id: dict[str, list[float]]) -> None:
        tx.run(
            """
            MERGE (paper:Paper {paper_id: $paper_id})
            SET paper.title = $title,
                paper.normalized_title = $normalized_title,
                paper.source_path = $source_path,
                paper.collection = coalesce($collection, 'default')
            """,
            paper_id=document.paper_id,
            title=document.title,
            normalized_title=normalize_title(document.title or ""),
            source_path=str(document.source_path),
            collection=document.metadata.get("collection", "default"),
        )

        for section in document.sections:
            section_id = f"{document.paper_id}:section:{section.order}"
            tx.run(
                """
                MATCH (paper:Paper {paper_id: $paper_id})
                MERGE (section:Section {section_id: $section_id})
                SET section.title = $title, section.order = $order
                MERGE (paper)-[:HAS_SECTION]->(section)
                """,
                paper_id=document.paper_id,
                section_id=section_id,
                title=section.title,
                order=section.order,
            )
            for chunk in section.chunks:
                tx.run(
                    """
                    MATCH (section:Section {section_id: $section_id})
                    MERGE (chunk:Chunk {chunk_id: $chunk_id})
                    SET chunk.text = $text,
                        chunk.order = $order,
                        chunk.page_number = $page_number,
                        chunk.paper_id = $paper_id,
                        chunk.paper_title = $paper_title,
                        chunk.collection = $collection,
                        chunk.embedding = $embedding
                    MERGE (section)-[:HAS_CHUNK]->(chunk)
                    """,
                    section_id=section_id,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    order=chunk.order,
                    page_number=chunk.page_number,
                    paper_id=document.paper_id,
                    paper_title=document.title,
                    collection=document.metadata.get("collection", "default"),
                    embedding=chunk_embedding_by_id[chunk.chunk_id],
                )

    def _write_extraction_tx(self, tx, extraction: PaperExtraction) -> None:
        tx.run(
            "MATCH (paper:Paper {paper_id: $paper_id}) SET paper.extracted_title = $title",
            paper_id=extraction.paper_id,
            title=extraction.title,
        )
        for chunk_result in extraction.chunks:
            for entity in chunk_result.entities:
                self._merge_entity(tx, entity, chunk_result.chunk_id)

            for objective in chunk_result.objectives:
                node_id = self._merge_semantic_node(
                    tx, extraction.paper_id, "Objective", "HAS_OBJECTIVE", objective.description
                )
                self._write_evidence_list(tx, chunk_result.chunk_id, objective.evidence, node_id)

            for approach in chunk_result.approaches:
                node_id = self._merge_semantic_node(
                    tx, extraction.paper_id, "Approach", "PROPOSES", approach.description
                )
                self._write_evidence_list(tx, chunk_result.chunk_id, approach.evidence, node_id)
                for method_name in approach.method_names:
                    self._connect_named_entity(tx, node_id, method_name, "USES_METHOD")

            for result in chunk_result.results:
                node_id = self._merge_semantic_node(
                    tx, extraction.paper_id, "Result", "REPORTS", result.description
                )
                self._write_evidence_list(tx, chunk_result.chunk_id, result.evidence, node_id)
                for dataset_name in result.dataset_names:
                    self._connect_named_entity(tx, node_id, dataset_name, "EVALUATED_ON")
                for metric_name in result.metric_names:
                    self._connect_named_entity(tx, node_id, metric_name, "REPORTS_METRIC")
                for task_name in result.task_names:
                    self._connect_named_entity(tx, node_id, task_name, "FOR_TASK")

            for constraint in chunk_result.constraints:
                node_id = self._merge_semantic_node(
                    tx, extraction.paper_id, "Constraint", "HAS_CONSTRAINT", constraint.description
                )
                self._write_evidence_list(tx, chunk_result.chunk_id, constraint.evidence, node_id)

            for claim in chunk_result.claims:
                node_id = self._merge_semantic_node(tx, extraction.paper_id, "Claim", "HAS_CLAIM", claim.statement)
                self._write_evidence_list(tx, chunk_result.chunk_id, claim.evidence, node_id)
                for entity_name in claim.entity_names:
                    self._connect_named_entity(tx, node_id, entity_name, "ABOUT")

    def _merge_entity(self, tx, entity: Entity, chunk_id: str) -> None:
        canonical_name = canonicalize_entity(entity.canonical_name or entity.name)
        tx.run(
            """
            MATCH (chunk:Chunk {chunk_id: $chunk_id})
            MERGE (entity:Entity {canonical_name: $canonical_name})
            ON CREATE SET entity.name = $name,
                          entity.aliases = $aliases,
                          entity.description = $description
            ON MATCH SET entity.name = coalesce(entity.name, $name),
                         entity.aliases = CASE
                             WHEN entity.aliases IS NULL OR size(entity.aliases) = 0 THEN $aliases
                             ELSE entity.aliases
                         END,
                         entity.description = coalesce(entity.description, $description)
            MERGE (chunk)-[:MENTIONS]->(entity)
            """,
            chunk_id=chunk_id,
            canonical_name=canonical_name,
            name=entity.name,
            aliases=entity.aliases,
            description=entity.description,
        )
        tx.run(
            f"""
            MATCH (entity:Entity {{canonical_name: $canonical_name}})
            SET entity:{entity.type.value}
            """,
            canonical_name=canonical_name,
        )
        self._link_possible_same_as(tx, canonical_name)

    def _merge_semantic_node(self, tx, paper_id: str, label: str, relation: str, description: str) -> str:
        node_id = semantic_id(label, paper_id, description)
        tx.run(
            f"""
            MATCH (paper:Paper {{paper_id: $paper_id}})
            MERGE (node:{label} {{id: $node_id}})
            SET node.description = $description
            MERGE (paper)-[:{relation}]->(node)
            """,
            paper_id=paper_id,
            node_id=node_id,
            description=description,
        )
        return node_id

    def _write_evidence_list(
        self,
        tx,
        chunk_id: str,
        evidence_items: list[Evidence],
        supported_node_id: str,
    ) -> None:
        for evidence in evidence_items:
            evidence_id = semantic_id("Evidence", evidence.chunk_id, evidence.text)
            tx.run(
                """
                MATCH (chunk:Chunk {chunk_id: $chunk_id})
                MATCH (supported {id: $supported_node_id})
                MERGE (evidence:Evidence {evidence_id: $evidence_id})
                SET evidence.text = $text, evidence.page_number = $page_number
                MERGE (chunk)-[:HAS_EVIDENCE]->(evidence)
                MERGE (evidence)-[:SUPPORTS]->(supported)
                """,
                chunk_id=evidence.chunk_id or chunk_id,
                supported_node_id=supported_node_id,
                evidence_id=evidence_id,
                text=evidence.text,
                page_number=evidence.page_number,
            )

    def _connect_named_entity(self, tx, source_node_id: str, entity_name: str, relation: str) -> None:
        canonical_name = canonicalize_entity(entity_name)
        tx.run(
            f"""
            MATCH (source {{id: $source_node_id}})
            MERGE (entity:Entity {{canonical_name: $canonical_name}})
            SET entity.name = coalesce(entity.name, $entity_name)
            MERGE (source)-[:{relation}]->(entity)
            """,
            source_node_id=source_node_id,
            canonical_name=canonical_name,
            entity_name=entity_name,
        )

    def _link_possible_same_as(self, tx, canonical_name: str) -> None:
        rows = tx.run(
            """
            MATCH (candidate:Entity)
            WHERE candidate.canonical_name <> $canonical_name
            RETURN candidate.canonical_name AS canonical_name
            LIMIT 1000
            """,
            canonical_name=canonical_name,
        )
        for row in rows:
            other = row["canonical_name"]
            if fuzz.ratio(canonical_name, other) >= self.settings.entity_match_threshold:
                tx.run(
                    """
                    MATCH (a:Entity {canonical_name: $a})
                    MATCH (b:Entity {canonical_name: $b})
                    MERGE (a)-[:SAME_AS {method: 'rapidfuzz'}]-(b)
                    """,
                    a=canonical_name,
                    b=other,
                )

    def get_langchain_graph(self):
        if self._lc_graph is None:
            from langchain_neo4j import Neo4jGraph

            self._lc_graph = Neo4jGraph(
                url=self.settings.neo4j_uri,
                username=self.settings.neo4j_user,
                password=self.settings.neo4j_password,
                database=self.settings.neo4j_database,
                refresh_schema=False,
                sanitize=True,
            )
        return self._lc_graph

    def get_vector_store(self):
        if self._vector_store is None:
            from langchain_neo4j import Neo4jVector

            retrieval_query = """
            OPTIONAL MATCH (node)-[:HAS_EVIDENCE]->(evidence:Evidence)-[:SUPPORTS]->(claim:Claim)
            WITH node, score,
                 collect(DISTINCT evidence.text)[0..3] AS evidence,
                 collect(DISTINCT claim.statement)[0..3] AS claims
            RETURN node.text AS text,
                   score,
                   {
                     id: node.chunk_id,
                     source: node.paper_id,
                     title: node.paper_title,
                     collection: node.collection,
                     page_number: node.page_number,
                     evidence: evidence,
                     claims: claims
                   } AS metadata
            """
            self._vector_store = Neo4jVector.from_existing_graph(
                embedding=self.embedding_provider.get_embeddings_model(),
                node_label="Chunk",
                embedding_node_property="embedding",
                text_node_properties=["text"],
                index_name="chunk_embedding",
                keyword_index_name="chunk_keyword",
                retrieval_query=retrieval_query,
                url=self.settings.neo4j_uri,
                username=self.settings.neo4j_user,
                password=self.settings.neo4j_password,
                database=self.settings.neo4j_database,
            )
        return self._vector_store
