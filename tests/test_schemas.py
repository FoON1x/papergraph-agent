from paperagent.graph.utils import canonicalize_entity, semantic_id
from paperagent.schemas import Entity, EntityType, Evidence, ParsedChunk


def test_parsed_chunk_rejects_blank_text() -> None:
    try:
        ParsedChunk(chunk_id="c1", text=" ", order=0)
    except ValueError as exc:
        assert "Chunk text cannot be blank" in str(exc)
    else:
        raise AssertionError("Blank chunk text should fail validation.")


def test_entity_defaults_to_concept() -> None:
    entity = Entity(name="GraphRAG")
    assert entity.type == EntityType.CONCEPT


def test_evidence_requires_chunk_id() -> None:
    evidence = Evidence(text="Supported by the source.", chunk_id="chunk-1")
    assert evidence.chunk_id == "chunk-1"


def test_canonicalize_entity() -> None:
    assert canonicalize_entity("  SST_2  ") == "sst-2"


def test_semantic_id_is_stable() -> None:
    assert semantic_id("Claim", "paper-1", "hello") == semantic_id("Claim", "paper-1", "hello")
