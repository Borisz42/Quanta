"""Unit and integration tests for QUANTA GraphStitcher (Phase 3).

Verifies:
1. Single-chunk passthrough and identity normalization.
2. Global entity resolution across chunks with diverse surface forms ("Dr. Vance", "Eleanor").
3. Surface aliases aggregation and property merging.
4. Category refinement (upgrading generic OBJECT to specific types).
5. Category incompatibility preservation (no accidental PERSON <-> LOCATION merges).
6. Event ID re-indexing across chunks (Ev1, Ev2, Ev3, ...).
7. Event foreign-key re-mapping to resolved global entity IDs (agent, patient, theme, location, instrument).
8. Cross-chunk Allen temporal interval synthesis (TEMP_ALLEN_MEETS and TEMP_ALLEN_BEFORE).
9. Intra-chunk relation endpoint re-mapping.
10. Proposition re-mapping and foreign key resolution.
11. Incremental vs. batch stitching equivalence.
12. QuantaGraph compilation bridge with zero orphan nodes and valid Clingo ASP verification.
13. Three-chunk canonical Eleanor Vance benchmark simulation.
"""

from __future__ import annotations

import pytest

from core.asg import QuantaGraph
from parser.asg_compiler import ASGCompiler
from parser.entity_manifest import ActiveEntityManifest, EntityRecord, EntityStorage
from parser.graph_stitcher import (
    GraphStitcher,
    stitch,
    stitch_to_graph,
    _categories_compatible,
    _detect_temporal_gap,
    _extract_name_tokens,
    _normalize_name,
)
from parser.schema import (
    DiscourseExtractionResult,
    ExtractedEntity,
    ExtractedEvent,
    ExtractedProposition,
    ExtractedRelation,
)


# ---------------------------------------------------------------------------
# Helper Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_chunk_1() -> DiscourseExtractionResult:
    """Chunk 1: Dr. Eleanor Vance isolates synthetic compound in cryogenic containment cell."""
    return DiscourseExtractionResult(
        chunk_id="chunk_0001",
        entities=[
            ExtractedEntity(
                id="E1",
                canonical_name="Dr. Eleanor Vance",
                category="PERSON",
                surface_aliases=["Dr. Vance", "Vance"],
                properties={"role": "researcher"},
            ),
            ExtractedEntity(
                id="E2",
                canonical_name="synthetic compound",
                category="SUBSTANCE",
                surface_aliases=["specimen"],
                properties={"state": "volatile"},
            ),
            ExtractedEntity(
                id="E3",
                canonical_name="cryogenic containment cell",
                category="LOCATION",
                surface_aliases=["cell"],
                properties={"type": "containment"},
            ),
        ],
        events=[
            ExtractedEvent(
                id="Ev1",
                predicate="isolate",
                agent_id="E1",
                patient_id="E2",
                location_id="E3",
                temporal_anchor="at dawn",
                tense="PAST",
                polarity=True,
                raw_text="Dr. Eleanor Vance isolated a volatile synthetic compound inside the cryogenic containment cell at dawn.",
            ),
        ],
        relations=[],
        propositions=[],
    )


@pytest.fixture
def sample_chunk_2() -> DiscourseExtractionResult:
    """Chunk 2: Eleanor notes lattice expansion; supervisor doubts validity."""
    return DiscourseExtractionResult(
        chunk_id="chunk_0002",
        entities=[
            ExtractedEntity(
                id="E1",
                canonical_name="Eleanor",
                category="PERSON",
                surface_aliases=["she"],
                properties={"title": "Dr."},
            ),
            ExtractedEntity(
                id="E2",
                canonical_name="specimen",
                category="SUBSTANCE",
                surface_aliases=["compound"],
                properties={"expansion": "anomalous"},
            ),
            ExtractedEntity(
                id="E3",
                canonical_name="supervisor",
                category="PERSON",
                surface_aliases=["her supervisor"],
                properties={"role": "supervisor"},
            ),
        ],
        events=[
            ExtractedEvent(
                id="Ev1",
                predicate="note",
                agent_id="E1",
                patient_id="E2",
                location_id=None,
                temporal_anchor="immediately",
                tense="PAST",
                polarity=True,
                raw_text="She immediately noted that this specimen exhibited anomalous lattice expansion.",
            ),
            ExtractedEvent(
                id="Ev2",
                predicate="doubt",
                agent_id="E3",
                theme_id="E2",
                temporal_anchor="initially",
                tense="PAST",
                polarity=True,
                raw_text="Her supervisor initially doubted the discovery.",
            ),
        ],
        relations=[
            ExtractedRelation(
                relation_type="TEMP_ALLEN_BEFORE",
                source_id="Ev1",
                target_id="Ev2",
                mechanism="intra_chunk_order",
            )
        ],
        propositions=[
            ExtractedProposition(
                id="P1",
                claim_text="specimen exhibited anomalous lattice expansion",
                predicate="exhibit",
                subject_id="E2",
                epistemic_status="OBSERVATION",
                source_agent_id="E1",
                event_id="Ev1",
            ),
        ],
    )


@pytest.fixture
def sample_chunk_3() -> DiscourseExtractionResult:
    """Chunk 3: Eleanor verifies hypothesis three hours later; polymer retains integrity; director prohibits tests."""
    return DiscourseExtractionResult(
        chunk_id="chunk_0003",
        entities=[
            ExtractedEntity(
                id="E1",
                canonical_name="Dr. Vance",
                category="PERSON",
                surface_aliases=["Eleanor"],
                properties={},
            ),
            ExtractedEntity(
                id="E2",
                canonical_name="polymer",
                category="SUBSTANCE",
                surface_aliases=["synthetic compound"],
                properties={"integrity": "retained"},
            ),
            ExtractedEntity(
                id="E3",
                canonical_name="containment cell",
                category="LOCATION",
                surface_aliases=["vessel"],
                properties={},
            ),
            ExtractedEntity(
                id="E4",
                canonical_name="laboratory director",
                category="PERSON",
                surface_aliases=["director"],
                properties={"role": "director"},
            ),
        ],
        events=[
            ExtractedEvent(
                id="Ev1",
                predicate="verify",
                agent_id="E1",
                location_id="E3",
                temporal_anchor="three hours later",
                tense="PAST",
                polarity=True,
                raw_text="Eleanor verified the hypothesis three hours later within the same vessel.",
            ),
            ExtractedEvent(
                id="Ev2",
                predicate="retain",
                agent_id="E2",
                temporal_anchor="throughout the afternoon",
                tense="PAST",
                polarity=True,
                raw_text="The resulting polymer retained its structural integrity.",
            ),
            ExtractedEvent(
                id="Ev3",
                predicate="prohibit",
                agent_id="E4",
                temporal_anchor=None,
                tense="PAST",
                polarity=True,
                raw_text="The laboratory director prohibited competing tests.",
            ),
        ],
        relations=[
            ExtractedRelation(
                relation_type="CAUSAL_MECHANISM_LINK",
                source_id="Ev2",
                target_id="Ev3",
                mechanism="prompted by structural retention",
            )
        ],
        propositions=[
            ExtractedProposition(
                id="P1",
                claim_text="competing tests are prohibited",
                predicate="prohibit",
                epistemic_status="PROHIBITED",
                source_agent_id="E4",
                event_id="Ev3",
            )
        ],
    )


# ---------------------------------------------------------------------------
# 1. Normalization and Matching Unit Tests
# ---------------------------------------------------------------------------

def test_normalize_name_and_extract_tokens():
    """Verify name normalization strips honorifics and articles, extracting clean tokens."""
    assert _normalize_name("Dr. Eleanor Vance") == "eleanor vance"
    assert _normalize_name("Doctor Eleanor Vance") == "eleanor vance"
    assert _normalize_name("The Laboratory Director") == "laboratory director"
    assert _normalize_name("Her Supervisor") == "supervisor"

    tokens1 = _extract_name_tokens("Dr. Eleanor Vance")
    assert tokens1 == {"eleanor", "vance"}

    tokens2 = _extract_name_tokens("Eleanor")
    assert tokens2 == {"eleanor"}

    tokens3 = _extract_name_tokens("containment cell")
    assert tokens3 == {"containment", "cell"}


def test_categories_compatible():
    """Verify category compatibility logic."""
    assert _categories_compatible("PERSON", "PERSON") is True
    assert _categories_compatible("OBJECT", "PERSON") is True
    assert _categories_compatible("SUBSTANCE", "OBJECT") is True
    assert _categories_compatible("PERSON", "ORGANIZATION") is True
    assert _categories_compatible("SUBSTANCE", "ARTIFACT") is True

    # Mutually exclusive domains
    assert _categories_compatible("PERSON", "LOCATION") is False
    assert _categories_compatible("PERSON", "SUBSTANCE") is False
    assert _categories_compatible("LOCATION", "SUBSTANCE") is False


def test_detect_temporal_gap():
    """Verify gap detection for Allen interval synthesis."""
    ev_gap = ExtractedEvent(id="Ev1", predicate="verify", temporal_anchor="three hours later")
    assert _detect_temporal_gap(ev_gap) is True

    ev_no_gap = ExtractedEvent(id="Ev1", predicate="verify", temporal_anchor="at dawn")
    assert _detect_temporal_gap(ev_no_gap) is False

    ev_next_day = ExtractedEvent(id="Ev1", predicate="verify", raw_text="The next morning they met.")
    assert _detect_temporal_gap(ev_next_day) is True


# ---------------------------------------------------------------------------
# 2. Single Chunk Passthrough Tests
# ---------------------------------------------------------------------------

def test_single_chunk_passthrough(sample_chunk_1):
    """Verify single chunk stitching preserves entities, events, and foreign-key integrity."""
    stitcher = GraphStitcher()
    result = stitcher.stitch([sample_chunk_1])

    assert len(result.entities) == 3
    assert len(result.events) == 1
    assert len(result.relations) == 0
    assert len(result.propositions) == 0

    fk_errors = result.validate_foreign_keys()
    assert fk_errors == []

    # Entity IDs canonicalized
    e1 = result.get_entity("E1")
    assert e1 is not None
    assert e1.canonical_name == "Dr. Eleanor Vance"
    assert "Dr. Vance" in e1.surface_aliases

    # Event references E1, E2, E3
    ev1 = result.get_event("Ev1")
    assert ev1 is not None
    assert ev1.agent_id == "E1"
    assert ev1.patient_id == "E2"
    assert ev1.location_id == "E3"


# ---------------------------------------------------------------------------
# 3. Multi-Chunk Global Entity Resolution Tests
# ---------------------------------------------------------------------------

def test_entity_resolution_across_diverse_surface_forms(sample_chunk_1, sample_chunk_2, sample_chunk_3):
    """Acceptance Criteria 1:
    Entities appearing across multiple chunks with different surface forms
    (e.g. 'Dr. Vance' and 'Eleanor') are unified into a single canonical entity with aggregated aliases.
    """
    stitcher = GraphStitcher()
    result = stitcher.stitch([sample_chunk_1, sample_chunk_2, sample_chunk_3])

    # Expect exactly 5 canonical global entities:
    # E1: Dr. Eleanor Vance (Eleanor Vance, Eleanor, Vance, Dr. Vance, she)
    # E2: synthetic compound (specimen, compound, polymer)
    # E3: cryogenic containment cell (cell, vessel, containment cell)
    # E4: supervisor
    # E5: laboratory director
    assert len(result.entities) == 5

    # 1. Eleanor Vance resolution
    vance = result.get_entity("E1")
    assert vance is not None
    assert "Eleanor" in vance.canonical_name
    # Verify aggregated surface aliases
    aliases_lower = {a.lower() for a in vance.surface_aliases}
    assert "dr. vance" in aliases_lower
    assert "eleanor" in aliases_lower
    assert "she" in aliases_lower
    assert "vance" in aliases_lower

    # 2. Synthetic compound resolution
    compound = result.get_entity("E2")
    assert compound is not None
    compound_aliases = {a.lower() for a in compound.surface_aliases}
    assert "specimen" in compound_aliases
    assert "polymer" in compound_aliases

    # 3. Containment cell resolution
    cell = result.get_entity("E3")
    assert cell is not None
    cell_aliases = {a.lower() for a in cell.surface_aliases}
    assert "cell" in cell_aliases or "vessel" in cell_aliases

    # 4. Supervisor
    supervisor = next((e for e in result.entities if "supervisor" in e.canonical_name.lower()), None)
    assert supervisor is not None
    assert supervisor.category == "PERSON"

    # 5. Laboratory Director
    director = next((e for e in result.entities if "director" in e.canonical_name.lower()), None)
    assert director is not None
    assert director.category == "PERSON"


def test_distinct_entities_not_merged():
    """Verify distinct entities with same category are not incorrectly merged."""
    c1 = DiscourseExtractionResult(
        chunk_id="c1",
        entities=[
            ExtractedEntity(id="E1", canonical_name="Alice Smith", category="PERSON"),
        ],
        events=[
            ExtractedEvent(id="Ev1", predicate="walk", agent_id="E1"),
        ],
    )
    c2 = DiscourseExtractionResult(
        chunk_id="c2",
        entities=[
            ExtractedEntity(id="E1", canonical_name="Bob Jones", category="PERSON"),
        ],
        events=[
            ExtractedEvent(id="Ev1", predicate="run", agent_id="E1"),
        ],
    )

    result = stitch([c1, c2])
    assert len(result.entities) == 2
    names = {e.canonical_name for e in result.entities}
    assert "Alice Smith" in names
    assert "Bob Jones" in names


def test_category_refinement_from_generic_object():
    """Verify generic OBJECT is refined when a later chunk specifies PERSON or SUBSTANCE."""
    c1 = DiscourseExtractionResult(
        chunk_id="c1",
        entities=[
            ExtractedEntity(id="E1", canonical_name="Eleanor", category="OBJECT"),
        ],
        events=[ExtractedEvent(id="Ev1", predicate="speak", agent_id="E1")],
    )
    c2 = DiscourseExtractionResult(
        chunk_id="c2",
        entities=[
            ExtractedEntity(id="E1", canonical_name="Dr. Eleanor Vance", category="PERSON"),
        ],
        events=[ExtractedEvent(id="Ev1", predicate="write", agent_id="E1")],
    )

    result = stitch([c1, c2])
    assert len(result.entities) == 1
    assert result.entities[0].category == "PERSON"
    assert result.entities[0].canonical_name == "Dr. Eleanor Vance"


# ---------------------------------------------------------------------------
# 4. Event Foreign-Key and ID Re-Mapping Tests
# ---------------------------------------------------------------------------

def test_event_foreign_key_remapping(sample_chunk_1, sample_chunk_2, sample_chunk_3):
    """Acceptance Criteria 2:
    All event arguments correctly reference the unified global entity IDs.
    """
    stitcher = GraphStitcher()
    result = stitcher.stitch([sample_chunk_1, sample_chunk_2, sample_chunk_3])

    # 6 total events across the 3 chunks (1 in c1, 2 in c2, 3 in c3)
    assert len(result.events) == 6

    # Verify event ID sequence
    event_ids = [ev.id for ev in result.events]
    assert event_ids == ["Ev1", "Ev2", "Ev3", "Ev4", "Ev5", "Ev6"]

    # Foreign key validation
    fk_errors = result.validate_foreign_keys()
    assert fk_errors == [], f"Found foreign-key errors: {fk_errors}"

    # Verify specific event bindings
    ev1 = result.get_event("Ev1")  # isolate: Eleanor, compound, cell
    assert ev1.agent_id == "E1"
    assert ev1.patient_id == "E2"
    assert ev1.location_id == "E3"

    ev2 = result.get_event("Ev2")  # note: Eleanor, specimen
    assert ev2.agent_id == "E1"
    assert ev2.patient_id == "E2"

    ev3 = result.get_event("Ev3")  # doubt: supervisor, theme=compound
    supervisor_id = [e.id for e in result.entities if "supervisor" in e.canonical_name.lower()][0]
    assert ev3.agent_id == supervisor_id
    assert ev3.theme_id == "E2"

    ev4 = result.get_event("Ev4")  # verify: Eleanor, cell
    assert ev4.agent_id == "E1"
    assert ev4.location_id == "E3"

    ev5 = result.get_event("Ev5")  # retain: polymer (E2)
    assert ev5.agent_id == "E2"

    ev6 = result.get_event("Ev6")  # prohibit: director
    director_id = [e.id for e in result.entities if "director" in e.canonical_name.lower()][0]
    assert ev6.agent_id == director_id


# ---------------------------------------------------------------------------
# 5. Cross-Chunk Allen Temporal Interval Synthesis Tests
# ---------------------------------------------------------------------------

def test_cross_chunk_temporal_interval_synthesis(sample_chunk_1, sample_chunk_2, sample_chunk_3):
    """Acceptance Criteria 3:
    Inter-chunk Allen temporal relations are synthesized between adjacent chunks.
    """
    stitcher = GraphStitcher(auto_detect_temporal_gap=True)
    result = stitcher.stitch([sample_chunk_1, sample_chunk_2, sample_chunk_3])

    # Inter-chunk relations:
    # Between chunk 1 and chunk 2: terminal Ev1 -> initial Ev2 (TEMP_ALLEN_MEETS)
    # Between chunk 2 and chunk 3: terminal Ev3 -> initial Ev4 (TEMP_ALLEN_BEFORE due to 'three hours later')
    relations = result.relations
    assert len(relations) >= 4  # 2 intra-chunk + 2 inter-chunk synthesized

    # Find relation connecting chunk 1 terminal (Ev1) to chunk 2 initial (Ev2)
    c1_to_c2 = [r for r in relations if r.source_id == "Ev1" and r.target_id == "Ev2"]
    assert len(c1_to_c2) == 1
    assert c1_to_c2[0].relation_type == "TEMP_ALLEN_MEETS"

    # Find relation connecting chunk 2 terminal (Ev3) to chunk 3 initial (Ev4)
    c2_to_c3 = [r for r in relations if r.source_id == "Ev3" and r.target_id == "Ev4"]
    assert len(c2_to_c3) == 1
    assert c2_to_c3[0].relation_type == "TEMP_ALLEN_BEFORE"

    # Verify intra-chunk relations preserved and re-mapped
    # Ev2 -> Ev3 (TEMP_ALLEN_BEFORE from chunk 2)
    c2_intra = [r for r in relations if r.source_id == "Ev2" and r.target_id == "Ev3"]
    assert len(c2_intra) == 1
    assert c2_intra[0].relation_type == "TEMP_ALLEN_BEFORE"

    # Ev5 -> Ev6 (CAUSAL_MECHANISM_LINK from chunk 3)
    c3_intra = [r for r in relations if r.source_id == "Ev5" and r.target_id == "Ev6"]
    assert len(c3_intra) == 1
    assert c3_intra[0].relation_type == "CAUSAL_MECHANISM_LINK"


# ---------------------------------------------------------------------------
# 6. Proposition Re-Mapping Tests
# ---------------------------------------------------------------------------

def test_proposition_remapping(sample_chunk_1, sample_chunk_2, sample_chunk_3):
    """Verify propositions across chunks are re-indexed with foreign keys updated."""
    result = stitch([sample_chunk_1, sample_chunk_2, sample_chunk_3])

    assert len(result.propositions) == 2
    p1 = result.get_proposition("P1")
    assert p1 is not None
    assert p1.source_agent_id == "E1"
    assert p1.event_id == "Ev2"  # Note event in chunk 2

    p2 = result.get_proposition("P2")
    assert p2 is not None
    assert p2.epistemic_status == "PROHIBITED"
    assert p2.event_id == "Ev6"  # Prohibit event in chunk 3


# ---------------------------------------------------------------------------
# 7. Incremental Stitching Equivalence
# ---------------------------------------------------------------------------

def test_incremental_and_batch_equivalence(sample_chunk_1, sample_chunk_2, sample_chunk_3):
    """Verify incremental add_chunk produces identical results to batch stitch."""
    # Batch
    batch_stitcher = GraphStitcher()
    batch_res = batch_stitcher.stitch([sample_chunk_1, sample_chunk_2, sample_chunk_3])

    # Incremental
    inc_stitcher = GraphStitcher()
    inc_stitcher.add_chunk(sample_chunk_1)
    inc_stitcher.add_chunk(sample_chunk_2)
    inc_stitcher.add_chunk(sample_chunk_3)
    inc_res = inc_stitcher.get_result()

    assert len(batch_res.entities) == len(inc_res.entities)
    assert len(batch_res.events) == len(inc_res.events)
    assert len(batch_res.relations) == len(inc_res.relations)
    assert len(batch_res.propositions) == len(inc_res.propositions)

    for be, ie in zip(batch_res.entities, inc_res.entities):
        assert be.id == ie.id
        assert be.canonical_name == ie.canonical_name
        assert set(be.surface_aliases) == set(ie.surface_aliases)


# ---------------------------------------------------------------------------
# 8. QuantaGraph Compilation Bridge & Zero Orphan Nodes
# ---------------------------------------------------------------------------

def test_stitch_to_graph_compilation_and_zero_orphans(sample_chunk_1, sample_chunk_2, sample_chunk_3):
    """Acceptance Criteria 4:
    Stitched result compiles into a unified QuantaGraph with valid foreign keys and zero orphan nodes.
    """
    stitcher = GraphStitcher()
    graph = stitcher.stitch_to_graph([sample_chunk_1, sample_chunk_2, sample_chunk_3], validate=True)

    assert isinstance(graph, QuantaGraph)
    assert graph.root_cid is not None
    assert graph.root is not None

    # Graph integrity verification (CIDs match and no dangling edges)
    valid, errors = graph.validate_integrity()
    assert valid is True
    assert errors == []

    # Verify all 5 entities and 6 events exist as nodes
    total_nodes = len(graph.nodes)
    assert total_nodes >= 11  # 5 entities + 6 events

    # Topological reachability: all events are reachable from root event node
    reachable_order = graph.topological_order()
    assert len(reachable_order) >= 11

    # Verify Clingo ASP validation passed
    assert hasattr(graph, "validation")
    assert graph.validation.is_valid is True
    assert graph.validation.errors == []


# ---------------------------------------------------------------------------
# 9. ActiveEntityManifest Integration Test
# ---------------------------------------------------------------------------

def test_stitcher_with_external_active_manifest(sample_chunk_1, sample_chunk_2):
    """Verify GraphStitcher updates ActiveEntityManifest in working memory."""
    storage = EntityStorage(":memory:")
    manifest = ActiveEntityManifest(storage=storage)
    stitcher = GraphStitcher(manifest=manifest, storage=storage)

    result = stitcher.stitch([sample_chunk_1, sample_chunk_2])
    assert len(result.entities) >= 3

    # Manifest should track active entities
    active_records = manifest.all_active()
    assert len(active_records) >= 3

    active_ids = {r.canonical_id for r in active_records}
    assert "E1" in active_ids
    vance_record = manifest.get("E1")
    assert "Eleanor" in vance_record.canonical_name
    assert "Dr. Vance" in vance_record.surface_aliases
