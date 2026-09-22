"""Unit and integration tests for Section 5: Dynamic World-State Tracking & Non-Monotonic Belief Revision.

Verifies:
- Task 5.1: Fluent State Representation (EntityStateRecord) with temporal validity intervals [t_start, t_end).
- Task 5.2: Non-Monotonic Transition Resolver (WorldStateManager.update_from_event) with TEMP_ALLEN_FINISHES wiring.
- Task 5.3: Temporal Point-in-Time State Queries (get_entity_state_at) answering past vs. current locations.
- Task 5.4: End-to-end integration via GraphStitcher, SQLite persistence, and GraphQueryAnswerer.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import pytest

from core.asg import QuantaGraph, QuantaNode
from memory.world_state import (
    EntityStateRecord,
    WorldStateManager,
    parse_timestamp,
)
from parser.asg_compiler import ASGCompiler
from parser.graph_stitcher import GraphStitcher, stitch_to_graph
from parser.schema import (
    DiscourseExtractionResult,
    ExtractedEntity,
    ExtractedEvent,
    ExtractedRelation,
)
from realizer.english_nlg import EnglishRealizer, GraphQueryAnswerer


# =============================================================================
# Task 5.1: Fluent State Representation Tests
# =============================================================================

def test_parse_timestamp():
    """Verify timestamp parsing across 12-hour, 24-hour, numeric, and milestone strings."""
    assert parse_timestamp(10.0) == 10.0
    assert parse_timestamp(14) == 14.0
    assert parse_timestamp("10:00 AM") == 10.0
    assert parse_timestamp("10:30 AM") == 10.5
    assert parse_timestamp("12:00 PM") == 12.0
    assert parse_timestamp("12:00 AM") == 0.0
    assert parse_timestamp("02:00 PM") == 14.0
    assert parse_timestamp("2:00 pm") == 14.0
    assert parse_timestamp("14:00") == 14.0
    assert parse_timestamp("14:30") == 14.5
    assert parse_timestamp("at dawn") == 6.0
    assert parse_timestamp("at noon") == 12.0
    assert parse_timestamp("midnight") == 0.0
    assert parse_timestamp("") is None
    assert parse_timestamp(None) is None


def test_entity_state_record_lifecycle():
    """Verify EntityStateRecord interval validity, open-ended status, and serialization."""
    rec = EntityStateRecord(
        entity_cid="E_SPECIMEN",
        property_slot="VAL_LOCATION_SLOT",
        value_cid="E_CELL_4",
        t_start=10.0,
        t_end=None,
        epistemic_status="FACT",
        event_cid="Ev1",
    )

    assert rec.is_current is True
    assert rec.is_active_at(9.9) is False
    assert rec.is_active_at(10.0) is True
    assert rec.is_active_at(13.9) is True
    assert rec.is_active_at(99.0) is True  # open-ended

    # Close the interval
    rec.t_end = 14.0
    rec.terminating_event_cid = "Ev2"
    assert rec.is_current is False
    assert rec.is_active_at(13.9) is True
    assert rec.is_active_at(14.0) is False
    assert rec.is_active_at(15.0) is False

    # Serialization roundtrip
    d = rec.to_dict()
    assert d["entity_cid"] == "E_SPECIMEN"
    assert d["value_cid"] == "E_CELL_4"
    assert d["t_start"] == 10.0
    assert d["t_end"] == 14.0
    assert d["terminating_event_cid"] == "Ev2"

    rec_copy = EntityStateRecord.from_dict(d)
    assert rec_copy.entity_cid == rec.entity_cid
    assert rec_copy.t_start == rec.t_start
    assert rec_copy.t_end == rec.t_end


# =============================================================================
# Task 5.2: Non-Monotonic Transition Resolver Tests
# =============================================================================

def test_non_monotonic_relocation_transition():
    """Verify that moving an entity closes prior interval and asserts new state."""
    ws = WorldStateManager()

    cell4 = QuantaNode(anchor="entity:cell_4", literal="Containment Cell 4")
    chamberB = QuantaNode(anchor="entity:chamber_b", literal="Analysis Chamber B")
    specimen = QuantaNode(anchor="entity:specimen", literal="synthetic compound specimen")
    eleanor = QuantaNode(anchor="entity:eleanor", literal="Dr. Eleanor Vance")

    ws.register_node(cell4)
    ws.register_node(chamberB)
    ws.register_node(specimen)
    ws.register_node(eleanor)

    # Chunk 1 (10:00 AM): "The specimen is in Containment Cell 4."
    ev1 = ExtractedEvent(
        id="Ev1",
        predicate="is",
        patient_id=specimen.cid,
        location_id=cell4.cid,
        temporal_anchor="10:00 AM",
        raw_text="The specimen was located in Containment Cell 4 at 10:00 AM.",
    )
    ws.update_from_event(ev1)

    rec1 = ws.get_entity_state_record_at(specimen.cid, "VAL_LOCATION_SLOT")
    assert rec1 is not None
    assert rec1.value_cid == cell4.cid
    assert rec1.t_start == 10.0
    assert rec1.t_end is None
    assert rec1.is_current is True

    # Chunk 5 (02:00 PM): "Eleanor moved the specimen into Analysis Chamber B."
    ev2 = ExtractedEvent(
        id="Ev2",
        predicate="move",
        agent_id=eleanor.cid,
        patient_id=specimen.cid,
        location_id=chamberB.cid,
        temporal_anchor="02:00 PM",
        raw_text="Eleanor moved the specimen into Analysis Chamber B at 02:00 PM.",
    )
    ws.update_from_event(ev2)

    # Historical record 1 must be closed at 14.0 (02:00 PM)
    history = ws.get_state_history(specimen.cid, "VAL_LOCATION_SLOT")
    assert len(history) == 2
    assert history[0].value_cid == cell4.cid
    assert history[0].t_start == 10.0
    assert history[0].t_end == 14.0
    assert history[0].terminating_event_cid == "Ev2"

    # Current record must be Chamber B starting at 14.0
    assert history[1].value_cid == chamberB.cid
    assert history[1].t_start == 14.0
    assert history[1].t_end is None
    assert history[1].is_current is True


def test_temp_allen_finishes_edge_wiring():
    """Verify TEMP_ALLEN_FINISHES relation is wired when prior state closes in QuantaGraph."""
    ws = WorldStateManager()
    graph = QuantaGraph()

    ev1_node = QuantaNode(literal="located in Containment Cell 4", anchor="ev:locate")
    ev2_node = QuantaNode(literal="moved into Chamber B", anchor="ev:move")
    graph.add_node(ev1_node)
    graph.add_node(ev2_node)

    # Initial state
    ws.assert_state(
        entity_cid="E_SPECIMEN",
        property_slot="VAL_LOCATION_SLOT",
        value_cid="E_CELL_4",
        t_start=10.0,
        event_cid=ev1_node.cid,
        graph=graph,
    )

    # Transition state
    ws.assert_state(
        entity_cid="E_SPECIMEN",
        property_slot="VAL_LOCATION_SLOT",
        value_cid="E_CHAMBER_B",
        t_start=14.0,
        event_cid=ev2_node.cid,
        graph=graph,
    )

    # Assert TEMP_ALLEN_FINISHES edge was added between ev1_node and ev2_node
    assert "TEMP_ALLEN_FINISHES" in ev1_node.edges
    assert ev2_node.cid in ev1_node.edges["TEMP_ALLEN_FINISHES"]


# =============================================================================
# Task 5.3: Temporal Point-in-Time State Queries Tests
# =============================================================================

def test_temporal_point_in_time_queries():
    """Verify querying state at 11:00 AM returns Cell 4, while querying current/15:00 returns Chamber B."""
    ws = WorldStateManager()

    cell4 = QuantaNode(anchor="entity:cell_4", literal="Containment Cell 4")
    chamberB = QuantaNode(anchor="entity:chamber_b", literal="Analysis Chamber B")
    vault7 = QuantaNode(anchor="entity:vault_7", literal="Vault 7")
    specimen = QuantaNode(anchor="entity:specimen", literal="specimen")

    ws.register_node(cell4)
    ws.register_node(chamberB)
    ws.register_node(vault7)
    ws.register_node(specimen)

    # Timeline:
    # 10:00 - 14:00: Cell 4
    # 14:00 - 17:00: Chamber B
    # 17:00 - onwards: Vault 7
    ws.assert_state("specimen", "VAL_LOCATION_SLOT", cell4.cid, t_start=10.0)
    ws.assert_state("specimen", "VAL_LOCATION_SLOT", chamberB.cid, t_start=14.0)
    ws.assert_state("specimen", "VAL_LOCATION_SLOT", vault7.cid, t_start=17.0)

    # Query 1: Before initial placement (< 10:00 AM)
    assert ws.get_entity_state_at("specimen", "VAL_LOCATION_SLOT", timestamp=8.0) is None

    # Query 2: Past state at 11:00 AM -> Containment Cell 4
    node_11am = ws.get_entity_state_at("specimen", "VAL_LOCATION_SLOT", timestamp=11.0)
    assert node_11am is not None
    assert node_11am.literal == "Containment Cell 4"

    # Query 3: Past state at 13:59 -> Containment Cell 4
    node_1359 = ws.get_entity_state_at("specimen", "VAL_LOCATION_SLOT", timestamp=13.99)
    assert node_1359 is not None
    assert node_1359.literal == "Containment Cell 4"

    # Query 4: Past state at 15:00 (3:00 PM) -> Analysis Chamber B
    node_1500 = ws.get_entity_state_at("specimen", "VAL_LOCATION_SLOT", timestamp=15.0)
    assert node_1500 is not None
    assert node_1500.literal == "Analysis Chamber B"

    # Query 5: Current state (timestamp=None) -> Vault 7
    node_current = ws.get_entity_state_at("specimen", "VAL_LOCATION_SLOT", timestamp=None)
    assert node_current is not None
    assert node_current.literal == "Vault 7"

    # Timeline explanation
    timeline = ws.explain_state_timeline("specimen", "VAL_LOCATION_SLOT")
    assert len(timeline) == 3
    assert "From 10 to 14: specimen was in Containment Cell 4" in timeline[0]
    assert "From 14 to 17: specimen was in Analysis Chamber B" in timeline[1]
    assert "From 17 onwards: specimen is in Vault 7 (current)" in timeline[2]


# =============================================================================
# Task 5.4: Multi-Chunk GraphStitcher & End-to-End Integration Tests
# =============================================================================

def test_graph_stitcher_world_state_integration():
    """Verify GraphStitcher aggregates world-state across chunks and syncs with QuantaGraph."""
    stitcher = GraphStitcher()

    chunk1 = DiscourseExtractionResult(
        chunk_id="chunk_001",
        entities=[
            ExtractedEntity(id="E1", canonical_name="specimen", category="SUBSTANCE", surface_aliases=["compound"]),
            ExtractedEntity(id="E2", canonical_name="Containment Cell 4", category="LOCATION", surface_aliases=["cell 4"]),
        ],
        events=[
            ExtractedEvent(
                id="Ev1",
                predicate="isolate",
                patient_id="E1",
                location_id="E2",
                temporal_anchor="10:00 AM",
                raw_text="The specimen was isolated inside Containment Cell 4 at 10:00 AM.",
            )
        ],
        relations=[],
        propositions=[],
    )

    chunk2 = DiscourseExtractionResult(
        chunk_id="chunk_002",
        entities=[
            ExtractedEntity(id="E1", canonical_name="Dr. Eleanor Vance", category="PERSON", surface_aliases=["Eleanor"]),
            ExtractedEntity(id="E2", canonical_name="specimen", category="SUBSTANCE", surface_aliases=["the compound"]),
            ExtractedEntity(id="E3", canonical_name="Analysis Chamber B", category="LOCATION", surface_aliases=["chamber B"]),
        ],
        events=[
            ExtractedEvent(
                id="Ev2",
                predicate="move",
                agent_id="E1",
                patient_id="E2",
                location_id="E3",
                temporal_anchor="02:00 PM",
                raw_text="Eleanor moved the specimen into Analysis Chamber B at 02:00 PM.",
            )
        ],
        relations=[],
        propositions=[],
    )

    graph = stitcher.stitch_to_graph([chunk1, chunk2], validate=True)
    assert hasattr(graph, "world_state")
    assert graph.world_state is stitcher.world_state

    # Verify stitcher point-in-time state lookup
    state_11am = stitcher.get_entity_state_at("specimen", "VAL_LOCATION_SLOT", timestamp=11.0)
    assert state_11am is not None
    assert "cell 4" in (state_11am.literal or "").lower()

    state_now = stitcher.get_entity_state_at("specimen", "VAL_LOCATION_SLOT", timestamp=None)
    assert state_now is not None
    assert "chamber b" in (state_now.literal or "").lower()

    # Check that TEMP_ALLEN_FINISHES edge was wired in graph
    has_finishes = any(
        "TEMP_ALLEN_FINISHES" in n.edges
        for n in graph.nodes.values()
    )
    assert has_finishes is True


def test_sqlite_persistence_and_reload():
    """Verify world-state records persist to SQLite and can be re-queried across instances."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "world_state.db"

        # Instance 1: Write state transitions
        ws1 = WorldStateManager(db_path=db_path)
        ws1.assert_state("specimen", "VAL_LOCATION_SLOT", "cell_4", t_start=10.0, event_cid="Ev1")
        ws1.assert_state("specimen", "VAL_LOCATION_SLOT", "chamber_b", t_start=14.0, event_cid="Ev2")
        ws1.close()

        # Instance 2: Connect to existing SQLite database
        ws2 = WorldStateManager(db_path=db_path)
        # Load records from SQLite
        cur = ws2._conn.cursor()
        cur.execute("SELECT entity_cid, property_slot, value_cid, t_start, t_end FROM entity_state_records ORDER BY t_start ASC")
        rows = cur.fetchall()

        assert len(rows) == 2
        # Record 1: cell_4, [10.0, 14.0)
        assert rows[0][0] == "specimen"
        assert rows[0][2] == "cell_4"
        assert rows[0][3] == 10.0
        assert rows[0][4] == 14.0

        # Record 2: chamber_b, [14.0, None)
        assert rows[1][0] == "specimen"
        assert rows[1][2] == "chamber_b"
        assert rows[1][3] == 14.0
        assert rows[1][4] is None
        ws2.close()



def test_english_realizer_temporal_where_query():
    """Verify GraphQueryAnswerer accurately resolves temporal point-in-time WHERE queries."""
    stitcher = GraphStitcher()

    chunk1 = DiscourseExtractionResult(
        chunk_id="chunk_001",
        entities=[
            ExtractedEntity(id="E1", canonical_name="specimen", category="SUBSTANCE"),
            ExtractedEntity(id="E2", canonical_name="containment cell", category="LOCATION"),
        ],
        events=[
            ExtractedEvent(
                id="Ev1",
                predicate="isolate",
                patient_id="E1",
                location_id="E2",
                temporal_anchor="10:00 AM",
            )
        ],
        relations=[],
        propositions=[],
    )

    chunk2 = DiscourseExtractionResult(
        chunk_id="chunk_002",
        entities=[
            ExtractedEntity(id="E1", canonical_name="Dr. Eleanor Vance", category="PERSON"),
            ExtractedEntity(id="E2", canonical_name="specimen", category="SUBSTANCE"),
            ExtractedEntity(id="E3", canonical_name="analysis chamber B", category="LOCATION"),
        ],
        events=[
            ExtractedEvent(
                id="Ev2",
                predicate="move",
                agent_id="E1",
                patient_id="E2",
                location_id="E3",
                temporal_anchor="02:00 PM",
            )
        ],
        relations=[],
        propositions=[],
    )

    graph = stitcher.stitch_to_graph([chunk1, chunk2], validate=True)
    realizer = EnglishRealizer()
    answerer = GraphQueryAnswerer(realizer=realizer)

    # 1. Historical query: "Where was the specimen at 11:00 AM?"
    ans_11am = answerer.answer(graph, "Where was the specimen at 11:00 AM?")
    assert "containment cell" in ans_11am.lower()

    # 2. Current query: "Where is the specimen now?"
    ans_now = answerer.answer(graph, "Where is the specimen now?")
    assert "analysis chamber b" in ans_now.lower()


def test_concurrency_thread_safety():
    """Verify thread-safety of WorldStateManager under concurrent assertions and queries."""
    ws = WorldStateManager()

    def worker(worker_id: int):
        for i in range(20):
            t = float(worker_id * 100 + i)
            ws.assert_state(f"entity_{worker_id}", "VAL_LOCATION_SLOT", f"loc_{worker_id}_{i}", t_start=t)
            res = ws.get_entity_state_at(f"entity_{worker_id}", "VAL_LOCATION_SLOT", timestamp=t)
            assert res is not None
            current = ws.get_entity_state_at(f"entity_{worker_id}", "VAL_LOCATION_SLOT", timestamp=None)
            assert current is not None

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, w) for w in range(4)]
        for f in futures:
            f.result()

    # Check that each worker's entity has exactly 20 records
    for w in range(4):
        history = ws.get_state_history(f"entity_{w}", "VAL_LOCATION_SLOT")
        assert len(history) == 20
        # The last record should be current
        assert history[-1].is_current is True
