"""Unit and integration tests for Neural Discourse Transducer (Phase 3).

Verifies:
3.1 Intermediate schemas, JSON round-tripping, foreign-key validation, and JSON schema export.
3.2 LMStudioTransducer connection handling, retry logic, markdown stripping, and error paths.
3.3 LocalGGUFTransducer initialization, lazy loading, and error handling.
3.4 MockTransducer deterministic fixture handling and heuristic fallback.
3.5 System prompt construction with 1-shot in-context demonstration.
3.6 Integration testing: Parsing the canonical Dr. Eleanor Vance narrative into the validated
    5-entity, 6-event schema with zero token debris and 100% foreign-key integrity.
End-to-End: Chunker (Phase 1) -> Entity Paging (Phase 2) -> Transducer (Phase 3).
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import requests

from parser.chunker import DiscourseChunker
from parser.entity_manifest import EntityPagingEngine
from parser.schema import (
    DiscourseExtractionResult,
    ExtractedEntity,
    ExtractedEvent,
    ExtractedProposition,
    ExtractedRelation,
)
from parser.transducer import (
    CANONICAL_ELEANOR_VANCE_FIXTURE,
    CANONICAL_ELEANOR_VANCE_TEXT,
    DEFAULT_SYSTEM_PROMPT,
    LMStudioTransducer,
    LocalGGUFTransducer,
    MockTransducer,
    create_transducer,
)


# ---------------------------------------------------------------------------
# 3.1: Typed Intermediate Schema Tests
# ---------------------------------------------------------------------------

def test_extracted_entity_creation_and_normalization():
    """Verify ExtractedEntity creation, dictionary conversion, and alias normalization."""
    data = {
        "id": "E1",
        "canonical_name": "Dr. Eleanor Vance",
        "category": "person",
        "surface_aliases": ["Eleanor", "Vance", "she"],
        "properties": {"role": "lead researcher"},
    }
    ent = ExtractedEntity.from_dict(data)
    assert ent.id == "E1"
    assert ent.canonical_name == "Dr. Eleanor Vance"
    assert ent.category == "PERSON"
    assert ent.surface_aliases == ["Eleanor", "Vance", "she"]
    assert ent.properties["role"] == "lead researcher"

    # Test key normalization with fallback field names
    alt_data = {
        "canonical_id": "E2",
        "name": "synthetic compound",
        "aliases": ["polymer"],
    }
    ent2 = ExtractedEntity.from_dict(alt_data)
    assert ent2.id == "E2"
    assert ent2.canonical_name == "synthetic compound"
    assert ent2.surface_aliases == ["polymer"]
    assert ent2.category == "OBJECT"


def test_extracted_event_creation_and_arguments():
    """Verify ExtractedEvent creation, foreign keys, and dictionary normalization."""
    ev = ExtractedEvent(
        id="Ev1",
        predicate="isolate",
        agent_id="E1",
        patient_id="E2",
        location_id="E3",
        temporal_anchor="at dawn",
        tense="PAST",
        polarity=True,
        raw_text="Eleanor isolated the compound.",
        arguments={"manner": "carefully"},
    )
    d = ev.to_dict()
    assert d["id"] == "Ev1"
    assert d["predicate"] == "isolate"
    assert d["agent_id"] == "E1"
    assert d["patient_id"] == "E2"
    assert d["location_id"] == "E3"
    assert d["temporal_anchor"] == "at dawn"
    assert d["arguments"]["manner"] == "carefully"

    restored = ExtractedEvent.from_dict(d)
    assert restored.id == ev.id
    assert restored.predicate == "isolate"
    assert restored.agent_id == "E1"
    assert restored.tense == "PAST"


def test_extracted_relation_and_proposition():
    """Verify ExtractedRelation and ExtractedProposition models."""
    rel = ExtractedRelation(
        relation_type="TEMP_ALLEN_MEETS",
        source_id="Ev1",
        target_id="Ev2",
        mechanism="immediately after",
    )
    rel_dict = rel.to_dict()
    restored_rel = ExtractedRelation.from_dict(rel_dict)
    assert restored_rel.relation_type == "TEMP_ALLEN_MEETS"
    assert restored_rel.source_id == "Ev1"
    assert restored_rel.target_id == "Ev2"

    prop = ExtractedProposition(
        id="P1",
        claim_text="specimen exhibited anomalous lattice expansion",
        epistemic_status="OBSERVATION",
        subject_id="E2",
        source_agent_id="E1",
        event_id="Ev2",
    )
    prop_dict = prop.to_dict()
    restored_prop = ExtractedProposition.from_dict(prop_dict)
    assert restored_prop.id == "P1"
    assert restored_prop.epistemic_status == "OBSERVATION"
    assert restored_prop.source_agent_id == "E1"


def test_discourse_extraction_result_json_roundtrip():
    """Verify complete JSON serialization and deserialization of DiscourseExtractionResult."""
    result = DiscourseExtractionResult(
        chunk_id="chunk_001",
        entities=[
            ExtractedEntity(id="E1", canonical_name="Alice", category="PERSON"),
            ExtractedEntity(id="E2", canonical_name="Sample", category="OBJECT"),
        ],
        events=[
            ExtractedEvent(id="Ev1", predicate="test", agent_id="E1", patient_id="E2"),
        ],
        relations=[
            ExtractedRelation(relation_type="TEMP_ALLEN_BEFORE", source_id="Ev1", target_id="Ev1"),
        ],
        propositions=[
            ExtractedProposition(id="P1", claim_text="Sample is pure", epistemic_status="FACT"),
        ],
        metadata={"model": "test_runner"},
    )

    json_str = result.to_json()
    restored = DiscourseExtractionResult.from_json(json_str)

    assert restored.chunk_id == "chunk_001"
    assert len(restored.entities) == 2
    assert len(restored.events) == 1
    assert restored.get_entity("E1").canonical_name == "Alice"
    assert restored.get_event("Ev1").predicate == "test"
    assert restored.metadata["model"] == "test_runner"


def test_foreign_key_validation():
    """Verify that foreign-key cross reference validation detects invalid IDs."""
    # 1. Valid graph
    valid_res = DiscourseExtractionResult(
        entities=[ExtractedEntity(id="E1", canonical_name="Alice")],
        events=[ExtractedEvent(id="Ev1", predicate="walk", agent_id="E1")],
        relations=[ExtractedRelation(relation_type="TEMP_ALLEN_BEFORE", source_id="Ev1", target_id="Ev1")],
        propositions=[ExtractedProposition(id="P1", claim_text="test", source_agent_id="E1", event_id="Ev1")],
    )
    assert len(valid_res.validate_foreign_keys()) == 0

    # 2. Invalid graph with dangling agent, location, relation, and proposition IDs
    invalid_res = DiscourseExtractionResult(
        entities=[ExtractedEntity(id="E1", canonical_name="Alice")],
        events=[
            ExtractedEvent(
                id="Ev1",
                predicate="walk",
                agent_id="E999",  # Dangling entity ID
                location_id="E888",  # Dangling location ID
            )
        ],
        relations=[
            ExtractedRelation(relation_type="CAUSAL_MECHANISM_LINK", source_id="Ev1", target_id="Ev999")  # Dangling
        ],
        propositions=[
            ExtractedProposition(id="P1", claim_text="bad", source_agent_id="E777", event_id="Ev555")  # Dangling
        ],
    )
    errors = invalid_res.validate_foreign_keys()
    assert len(errors) == 5
    assert any("E999" in e for e in errors)
    assert any("E888" in e for e in errors)
    assert any("Ev999" in e for e in errors)
    assert any("E777" in e for e in errors)
    assert any("Ev555" in e for e in errors)


def test_json_schema_export():
    """Verify that get_json_schema returns valid schema dictionary."""
    schema = DiscourseExtractionResult.get_json_schema()
    assert schema["type"] == "object"
    assert "entities" in schema["properties"]
    assert "events" in schema["properties"]
    assert "relations" in schema["properties"]
    assert "propositions" in schema["properties"]


# ---------------------------------------------------------------------------
# 3.4 & 3.6: MockTransducer & Canonical Eleanor Vance Extraction Tests
# ---------------------------------------------------------------------------

def test_mock_transducer_canonical_eleanor_vance():
    """Verify that parsing the Eleanor Vance narrative produces the validated
    5-entity, 6-event schema with zero token debris and 100% foreign-key integrity.
    """
    transducer = MockTransducer()
    result = transducer.transduce(
        chunk_text=CANONICAL_ELEANOR_VANCE_TEXT,
        chunk_id="chunk_01",
    )

    assert result.chunk_id == "chunk_01"

    # 1. Exactly 5 canonical entities
    assert len(result.entities) == 5, f"Expected exactly 5 entities, found {len(result.entities)}"
    entity_map = {e.id: e for e in result.entities}
    assert set(entity_map.keys()) == {"E1", "E2", "E3", "E4", "E5"}

    e1 = entity_map["E1"]
    assert e1.canonical_name == "Dr. Eleanor Vance"
    assert e1.category == "PERSON"
    assert "Eleanor" in e1.surface_aliases
    assert "Vance" in e1.surface_aliases

    e2 = entity_map["E2"]
    assert e2.canonical_name == "synthetic compound"
    assert e2.category == "SUBSTANCE"
    assert "specimen" in e2.surface_aliases
    assert "polymer" in e2.surface_aliases

    e3 = entity_map["E3"]
    assert e3.canonical_name == "cryogenic containment cell"
    assert "vessel" in e3.surface_aliases or "containment cell" in e3.surface_aliases

    e4 = entity_map["E4"]
    assert e4.canonical_name == "supervisor"
    assert e4.category == "PERSON"

    e5 = entity_map["E5"]
    assert e5.canonical_name == "laboratory director"
    assert e5.category == "PERSON"

    # 2. Exactly 6 event predicates
    assert len(result.events) == 6, f"Expected exactly 6 events, found {len(result.events)}"
    event_map = {ev.id: ev for ev in result.events}
    assert set(event_map.keys()) == {"Ev1", "Ev2", "Ev3", "Ev4", "Ev5", "Ev6"}

    # Ev1: isolate (E1, E2, E3)
    ev1 = event_map["Ev1"]
    assert ev1.predicate == "isolate"
    assert ev1.agent_id == "E1"
    assert ev1.patient_id == "E2"
    assert ev1.location_id == "E3"
    assert ev1.temporal_anchor == "at dawn"

    # Ev2: note (E1)
    ev2 = event_map["Ev2"]
    assert ev2.predicate == "note"
    assert ev2.agent_id == "E1"

    # Ev3: doubt (E4 supervisor)
    ev3 = event_map["Ev3"]
    assert ev3.predicate == "doubt"
    assert ev3.agent_id == "E4"

    # Ev4: verify (E1 Eleanor, location E3 vessel)
    ev4 = event_map["Ev4"]
    assert ev4.predicate == "verify"
    assert ev4.agent_id == "E1"
    assert ev4.location_id == "E3"

    # Ev5: retain (E2 polymer)
    ev5 = event_map["Ev5"]
    assert ev5.predicate == "retain"
    assert ev5.agent_id == "E2"

    # Ev6: prohibit (E5 laboratory director)
    ev6 = event_map["Ev6"]
    assert ev6.predicate == "prohibit"
    assert ev6.agent_id == "E5"

    # 3. Inter-Event Spatio-Temporal and Causal Relations
    assert len(result.relations) >= 4
    rel_types = {r.relation_type for r in result.relations}
    assert "TEMP_ALLEN_MEETS" in rel_types
    assert "TEMP_ALLEN_BEFORE" in rel_types
    assert "CAUSAL_MECHANISM_LINK" in rel_types

    causal = [r for r in result.relations if r.relation_type == "CAUSAL_MECHANISM_LINK"]
    assert any(r.source_id == "Ev5" and r.target_id == "Ev6" for r in causal)

    # 4. Epistemic Propositions
    assert len(result.propositions) >= 4
    prop_statuses = {p.epistemic_status for p in result.propositions}
    assert "OBSERVATION" in prop_statuses
    assert "HYPOTHESIS" in prop_statuses
    assert "DOUBTED" in prop_statuses
    assert "PROHIBITED" in prop_statuses

    # 5. Foreign Key Integrity Check
    fk_errors = result.validate_foreign_keys()
    assert fk_errors == [], f"Foreign key validation errors: {fk_errors}"


def test_mock_transducer_custom_fixture_and_fallback():
    """Verify registering custom fixtures and heuristic fallback on unmatched chunks."""
    custom_res = DiscourseExtractionResult(
        chunk_id="custom_01",
        entities=[ExtractedEntity(id="E1", canonical_name="Bob")],
        events=[ExtractedEvent(id="Ev1", predicate="run", agent_id="E1")],
    )
    transducer = MockTransducer()
    transducer.register_fixture("Custom Experiment Story", custom_res)

    # Matched custom fixture
    res = transducer.transduce("This is the Custom Experiment Story from yesterday.")
    assert res.entities[0].canonical_name == "Bob"

    # Unmatched fallback
    res_fallback = transducer.transduce("Galileo dropped two spheres from the tower.")
    assert len(res_fallback.entities) >= 1
    assert len(res_fallback.events) >= 1
    assert res_fallback.validate_foreign_keys() == []


# ---------------------------------------------------------------------------
# 3.2: LMStudioTransducer Tests
# ---------------------------------------------------------------------------

def test_lm_studio_health_check_offline():
    """Verify health check returns False gracefully when server is offline."""
    client = LMStudioTransducer(base_url="http://127.0.0.1:59999/v1", timeout=0.5)
    assert client.check_health() is False
    assert client.is_available() is False


@patch("requests.Session.post")
def test_lm_studio_successful_transduction(mock_post):
    """Verify LMStudioTransducer formats request properly and parses JSON completion."""
    client = LMStudioTransducer(base_url="http://localhost:1234/v1")

    # Mock server response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "entities": [
                            {"id": "E1", "canonical_name": "Eleanor", "category": "PERSON", "surface_aliases": ["she"]}
                        ],
                        "events": [
                            {"id": "Ev1", "predicate": "synthesize", "agent_id": "E1", "tense": "PAST", "polarity": True}
                        ],
                        "relations": [],
                        "propositions": [],
                    })
                }
            }
        ]
    }
    mock_response.json.return_value = mock_payload
    mock_post.return_value = mock_response

    manifest_prompt = "ACTIVE ENTITIES:\n- E1: Eleanor"
    result = client.transduce(
        chunk_text="Eleanor synthesized the specimen.",
        chunk_id="chunk_test",
        active_manifest_prompt=manifest_prompt,
    )

    assert result.chunk_id == "chunk_test"
    assert len(result.entities) == 1
    assert result.entities[0].canonical_name == "Eleanor"
    assert result.metadata["backend"] == "lm_studio"

    # Verify request payload
    call_args = mock_post.call_args
    sent_payload = call_args.kwargs["json"]
    assert sent_payload["temperature"] == 0.0
    assert sent_payload["response_format"] == {"type": "json_object"}
    assert any("ACTIVE ENTITIES:" in m["content"] for m in sent_payload["messages"])


@patch("requests.Session.post")
def test_lm_studio_markdown_code_fence_stripping(mock_post):
    """Verify parser strips ```json code fences generated by some LLMs."""
    client = LMStudioTransducer()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "```json\n{\n  \"entities\": [{\"id\": \"E1\", \"canonical_name\": \"Dr. Vance\", \"category\": \"PERSON\", \"surface_aliases\": []}],\n  \"events\": [],\n  \"relations\": [],\n  \"propositions\": []\n}\n```"
                }
            }
        ]
    }
    mock_post.return_value = mock_resp

    result = client.transduce("Dr. Vance waited.")
    assert len(result.entities) == 1
    assert result.entities[0].canonical_name == "Dr. Vance"


@patch("requests.Session.post")
def test_lm_studio_retry_on_server_error(mock_post):
    """Verify retry logic on transient HTTP 500 error followed by 200 OK."""
    client = LMStudioTransducer(max_retries=2, retry_backoff=0.01)

    fail_resp = MagicMock()
    fail_resp.status_code = 500
    fail_resp.text = "Internal Server Error"

    succ_resp = MagicMock()
    succ_resp.status_code = 200
    succ_resp.json.return_value = {
        "choices": [
            {"message": {"content": "{\"entities\": [], \"events\": [], \"relations\": [], \"propositions\": []}"}}
        ]
    }

    mock_post.side_effect = [fail_resp, succ_resp]

    result = client.transduce("Testing retries.")
    assert mock_post.call_count == 2
    assert result is not None


# ---------------------------------------------------------------------------
# 3.3: LocalGGUFTransducer Tests
# ---------------------------------------------------------------------------

def test_local_gguf_transducer_missing_file():
    """Verify LocalGGUFTransducer raises FileNotFoundError if model file does not exist."""
    transducer = LocalGGUFTransducer(model_path="/nonexistent/model.gguf")
    assert transducer.is_available() is False
    with pytest.raises(FileNotFoundError):
        transducer.load_model()


def test_local_gguf_transducer_no_model_path():
    """Verify ValueError is raised if no model path was provided."""
    transducer = LocalGGUFTransducer()
    with pytest.raises(ValueError):
        transducer.load_model()


# ---------------------------------------------------------------------------
# Factory Function Tests
# ---------------------------------------------------------------------------

def test_create_transducer_factory():
    """Verify create_transducer instantiates appropriate backend."""
    mock_t = create_transducer("mock")
    assert isinstance(mock_t, MockTransducer)

    lm_t = create_transducer("lmstudio", base_url="http://localhost:1234/v1")
    assert isinstance(lm_t, LMStudioTransducer)

    gguf_t = create_transducer("gguf", model_path="dummy.gguf")
    assert isinstance(gguf_t, LocalGGUFTransducer)

    auto_t = create_transducer("auto")
    assert isinstance(auto_t, (MockTransducer, LMStudioTransducer))

    with pytest.raises(ValueError):
        create_transducer("invalid_backend")


# ---------------------------------------------------------------------------
# 3.5: System Prompt & 1-Shot Demonstration Verification
# ---------------------------------------------------------------------------

def test_system_prompt_structure():
    """Verify system prompt contains required rules, foreign key guidelines, and 1-shot demo."""
    prompt = DEFAULT_SYSTEM_PROMPT
    assert "ACTIVE ENTITIES:" in prompt
    assert "ONE-SHOT DEMONSTRATION:" in prompt
    assert "VAL_X1_AGENT" in prompt
    assert "TEMP_ALLEN_MEETS" in prompt
    assert "CAUSAL_MECHANISM_LINK" in prompt
    assert "DO NOT create entities for abstract propositions" in prompt


# ---------------------------------------------------------------------------
# End-to-End Pipeline Integration: Chunker -> Entity Manifest -> Transducer
# ---------------------------------------------------------------------------

def test_end_to_end_chunker_manifest_transducer_flow():
    """Verify full neuro-symbolic pipeline flow across Phase 1, Phase 2, and Phase 3:
    1. DiscourseChunker segments text into episodic chunks.
    2. EntityPagingEngine pre-scans chunk and formats active manifest prompt block.
    3. Transducer processes chunk text + active manifest.
    4. Transducer output is synced back into EntityPagingEngine working memory.
    """
    chunker = DiscourseChunker(min_words=20, max_words=300)
    chunks = chunker.chunk(CANONICAL_ELEANOR_VANCE_TEXT)
    assert len(chunks) >= 1

    paging_engine = EntityPagingEngine(target_size=5)

    for idx, chunk in enumerate(chunks, 1):
        # 1. Pre-scan chunk for dormant entities
        paged_in, scan_ms = paging_engine.pre_scan_and_page(chunk.text, chunk_idx=idx)
        assert scan_ms < 1.0

        # 2. Render active entity manifest prompt block
        manifest_prompt = paging_engine.format_prompt_block()

        # 3. Transduce chunk
        transducer = MockTransducer()
        result = transducer.transduce(
            chunk_text=chunk.text,
            chunk_id=chunk.chunk_id,
            active_manifest_prompt=manifest_prompt,
        )

        # 4. Sync extracted entities into EntityPagingEngine working memory
        extracted_entities_dicts = [e.to_dict() for e in result.entities]
        updated_records = paging_engine.update_from_extraction(extracted_entities_dicts, chunk_idx=idx)

        assert len(updated_records) == 5
        assert paging_engine.manifest.is_active("E1")
        assert paging_engine.manifest.is_active("E2")
        assert paging_engine.manifest.get("E1").canonical_name == "Dr. Eleanor Vance"

    paging_engine.close()
