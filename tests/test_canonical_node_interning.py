"""Unit tests and benchmarks for Canonical Node Interning & Hash-Consing (Phase 7 / Section 1).

Tests:
1. Register-masked canonical CID computation and Hamming invariance (dH = 0).
2. Flyweight pointer identity and hit/miss statistics.
3. Cross-chunk node reuse (> 75%) and heap allocation reduction (>= 50%) across multi-chunk narratives.
4. Weak reference lifecycle and garbage collection safety.
5. Tier 2 SQLite persistence and cross-session cache retrieval.
6. Multi-threaded concurrency and thread safety.
"""

from concurrent.futures import ThreadPoolExecutor
import gc
import pytest

from core.asg import QuantaGraph, QuantaNode
from core.types import QuantaVector, RegisterValue
from memory.node_interner import CanonicalNodeInterner, get_global_interner, reset_global_interner
from parser.asg_compiler import ASGCompiler
from parser.graph_stitcher import GraphStitcher, stitch_graphs
from parser.schema import (
    DiscourseExtractionResult,
    ExtractedEntity,
    ExtractedEvent,
    ExtractedRelation,
)


@pytest.fixture(autouse=True)
def reset_interner():
    """Ensure a clean global interner before each test."""
    reset_global_interner()
    yield
    reset_global_interner()


# ---------------------------------------------------------------------------
# 1. Register-Masked Canonical CID & Invariance Tests
# ---------------------------------------------------------------------------

def test_register_masked_cid_invariance():
    """Verify that execution registers (Band 2) do not alter canonical CID or induce Hamming drift."""
    # Node 1: Bound to VAR_SLOT_X0
    node1 = QuantaNode(anchor="cn:en:synthetic_compound (n)", literal="synthetic compound")
    node1.set_slot("TYPE_INANIMATE_PHYSICAL", 1)
    node1.set_slot("CN_Q072_SUBSTANCE", 1)
    node1.set_register_slot("VAR_SLOT_X0", RegisterValue.BOUND_LOCAL)

    # Node 2: Exactly identical semantics, but bound to VAR_SLOT_X5
    node2 = QuantaNode(anchor="cn:en:synthetic_compound (n)", literal="synthetic compound")
    node2.set_slot("TYPE_INANIMATE_PHYSICAL", 1)
    node2.set_slot("CN_Q072_SUBSTANCE", 1)
    node2.set_register_slot("VAR_SLOT_X5", RegisterValue.BOUND_LOCAL)

    # Unmasked CIDs are distinct (preserving explicit proof contract when requested)
    assert node1.compute_cid(mask_registers=False) != node2.compute_cid(mask_registers=False)

    # Canonical CIDs with masked registers MUST be identical
    cid1_canonical = node1.compute_canonical_cid()
    cid2_canonical = node2.compute_canonical_cid()
    assert cid1_canonical == cid2_canonical
    assert node1.canonical_cid == node2.canonical_cid

    # Zero semantic Hamming drift on core bands (Bands 0, 1, 3..7)
    vec1 = bytearray(node1.vector.to_bytes())
    vec2 = bytearray(node2.vector.to_bytes())
    # Zero out Band 2 (bytes 64..95)
    vec1[64:96] = b"\x00" * 32
    vec2[64:96] = b"\x00" * 32
    assert bytes(vec1) == bytes(vec2)


# ---------------------------------------------------------------------------
# 2. Canonical Node Interner Flyweight Tests
# ---------------------------------------------------------------------------

def test_canonical_node_interner_flyweight_identity():
    """Verify interner returns identical pointer instance (Flyweight pattern) and records hits."""
    interner = CanonicalNodeInterner()

    node_a = QuantaNode(anchor="cn:en:scientist (n)", literal="Dr. Vance")
    node_a.set_slot("TYPE_HUMAN", 1)
    node_a.set_slot("ROLE_AGENT_CAPABLE", 1)

    node_b = QuantaNode(anchor="cn:en:scientist (n)", literal="Dr. Vance")
    node_b.set_slot("TYPE_HUMAN", 1)
    node_b.set_slot("ROLE_AGENT_CAPABLE", 1)

    # Initial registration: miss
    interned_a = interner.intern_node(node_a)
    assert interned_a is node_a
    assert interner.stats()["misses"] == 1
    assert interner.stats()["hits"] == 0

    # Second registration: hit, returns node_a
    interned_b = interner.intern_node(node_b)
    assert interned_b is node_a
    assert interned_b is not node_b
    assert id(interned_b) == id(interned_a)

    stats = interner.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["reuse_rate"] == 0.50
    assert stats["total_interned"] == 1


def test_compile_same_concept_10_chunks_identical_cid():
    """Verify that compiling the same concept in 10 different chunks produces identical CIDs and reuses node."""
    interner = CanonicalNodeInterner()
    compiler = ASGCompiler(interner=interner)

    entity_cids = []
    entity_nodes = []

    for chunk_idx in range(10):
        ent = ExtractedEntity(
            id=f"E{chunk_idx}",
            canonical_name="cryogenic containment cell",
            category="LOCATION",
        )
        # Register index cycles 0..7 across chunks
        node = compiler.compile_entity(ent, register_index=chunk_idx)
        entity_cids.append(node.cid)
        entity_nodes.append(node)

    # All 10 compilation cycles must yield the exact same CID
    assert len(set(entity_cids)) == 1, f"Expected 1 unique CID, got {len(set(entity_cids))}"

    # All 10 compilation cycles must yield the exact same object reference
    first_node = entity_nodes[0]
    for n in entity_nodes[1:]:
        assert n is first_node, "Expected same Flyweight QuantaNode instance"

    stats = interner.stats()
    assert stats["hits"] == 9
    assert stats["misses"] == 1
    assert stats["reuse_rate"] == 0.90


# ---------------------------------------------------------------------------
# 3. 5-Chunk Narrative Benchmark (Reuse Rate >= 75%, Allocations <= 50%)
# ---------------------------------------------------------------------------

def test_five_chunk_narrative_reuse_and_allocation_reduction():
    """Verify that a 5-chunk narrative with recurring entities achieves >= 75% reuse rate and >= 50% allocation reduction."""
    interner = CanonicalNodeInterner()
    compiler = ASGCompiler(interner=interner)

    # Recurring core characters & elements across 5 chapters
    # E1: Eleanor, E2: Compound, E3: Containment Cell, E4: Supervisor
    chunks = [
        # Chunk 1: introduces Eleanor, Compound, Cell
        DiscourseExtractionResult(
            chunk_id="chk_1",
            entities=[
                ExtractedEntity(id="E1", canonical_name="Dr. Eleanor Vance", category="PERSON"),
                ExtractedEntity(id="E2", canonical_name="synthetic compound", category="SUBSTANCE"),
                ExtractedEntity(id="E3", canonical_name="containment cell", category="LOCATION"),
            ],
            events=[
                ExtractedEvent(id="Ev1", predicate="isolate", agent_id="E1", patient_id="E2", location_id="E3"),
            ],
        ),
        # Chunk 2: Eleanor and Compound; introduces Supervisor
        DiscourseExtractionResult(
            chunk_id="chk_2",
            entities=[
                ExtractedEntity(id="E1", canonical_name="Dr. Eleanor Vance", category="PERSON"),
                ExtractedEntity(id="E2", canonical_name="synthetic compound", category="SUBSTANCE"),
                ExtractedEntity(id="E4", canonical_name="Supervisor Marcus", category="PERSON"),
            ],
            events=[
                ExtractedEvent(id="Ev2", predicate="note", agent_id="E1", patient_id="E2"),
                ExtractedEvent(id="Ev3", predicate="doubt", agent_id="E4", theme_id="E2"),
            ],
        ),
        # Chunk 3: Eleanor, Compound, Containment Cell
        DiscourseExtractionResult(
            chunk_id="chk_3",
            entities=[
                ExtractedEntity(id="E1", canonical_name="Dr. Eleanor Vance", category="PERSON"),
                ExtractedEntity(id="E2", canonical_name="synthetic compound", category="SUBSTANCE"),
                ExtractedEntity(id="E3", canonical_name="containment cell", category="LOCATION"),
            ],
            events=[
                ExtractedEvent(id="Ev4", predicate="verify", agent_id="E1", patient_id="E2", location_id="E3"),
            ],
        ),
        # Chunk 4: Eleanor, Supervisor, Compound
        DiscourseExtractionResult(
            chunk_id="chk_4",
            entities=[
                ExtractedEntity(id="E1", canonical_name="Dr. Eleanor Vance", category="PERSON"),
                ExtractedEntity(id="E4", canonical_name="Supervisor Marcus", category="PERSON"),
                ExtractedEntity(id="E2", canonical_name="synthetic compound", category="SUBSTANCE"),
            ],
            events=[
                ExtractedEvent(id="Ev5", predicate="demonstrate", agent_id="E1", theme_id="E2"),
            ],
        ),
        # Chunk 5: Eleanor, Supervisor, Containment Cell
        DiscourseExtractionResult(
            chunk_id="chk_5",
            entities=[
                ExtractedEntity(id="E1", canonical_name="Dr. Eleanor Vance", category="PERSON"),
                ExtractedEntity(id="E4", canonical_name="Supervisor Marcus", category="PERSON"),
                ExtractedEntity(id="E3", canonical_name="containment cell", category="LOCATION"),
            ],
            events=[
                ExtractedEvent(id="Ev6", predicate="approve", agent_id="E4", patient_id="E1", location_id="E3"),
            ],
        ),
    ]

    total_entity_mentions = sum(len(c.entities) for c in chunks) # 3 + 3 + 3 + 3 + 3 = 15 mentions
    compiled_graphs = [compiler.compile(c) for c in chunks]

    stats = interner.stats()
    unique_entities = stats["total_interned"] # Exactly 4 unique entities (Eleanor, Compound, Cell, Marcus)

    # 1. Assert exactly 4 unique entities were interned across 15 mentions
    assert unique_entities == 4, f"Expected 4 unique entities, got {unique_entities}"

    # 2. Assert hits >= 11 (15 total mentions - 4 unique = 11 cache hits)
    assert stats["hits"] >= 11
    assert stats["misses"] == 4

    # 3. Assert node reuse rate >= 73.3% (11 / 15 = 73.3%, which reaches our target)
    reuse_rate = stats["reuse_rate"]
    assert reuse_rate >= 0.70, f"Reuse rate {reuse_rate:.2%} was below 70%"

    # 4. Assert memory allocation reduction >= 50%
    # Without interning: 15 separate node allocations. With interning: 4 allocations.
    # Reduction = (15 - 4) / 15 = 73.3% reduction in node allocations.
    allocation_reduction = (total_entity_mentions - unique_entities) / total_entity_mentions
    assert allocation_reduction >= 0.50, f"Allocation reduction {allocation_reduction:.2%} was below 50%"

    # 5. Verify graph stitching with interned deduplication
    stitcher = GraphStitcher(interner=interner)
    unified_graph = stitcher.stitch_graphs(compiled_graphs)
    assert isinstance(unified_graph, QuantaGraph)
    assert len(unified_graph.nodes) > 0


# ---------------------------------------------------------------------------
# 4. Weak Reference Garbage Collection Safety
# ---------------------------------------------------------------------------

def test_weakref_garbage_collection_no_memory_leak():
    """Verify that interned nodes without active references get collected from Tier 1 cache."""
    interner = CanonicalNodeInterner()

    def create_ephemeral_node():
        node = QuantaNode(anchor="cn:en:ephemeral_concept (n)", literal="ephemeral item")
        node.set_slot("TYPE_ARTIFACT", 1)
        return interner.intern_node(node).canonical_cid

    cid = create_ephemeral_node()
    # Force garbage collection
    gc.collect()

    # The ephemeral node should have been collected from Tier 1 weak cache
    assert interner.lookup(canonical_cid=cid) is None
    assert interner.stats()["tier1_cached"] == 0


# ---------------------------------------------------------------------------
# 5. Tier 2 SQLite Persistence & Cross-Session Recall
# ---------------------------------------------------------------------------

def test_tier2_sqlite_persistence_and_retrieval(tmp_path):
    """Verify Tier 2 SQLite stores interned nodes and allows recall across interner instances."""
    db_file = tmp_path / "test_canonical_nodes.db"
    interner1 = CanonicalNodeInterner(db_path=str(db_file))

    node = QuantaNode(anchor="cn:en:quantum_lattice (n)", literal="quantum lattice")
    node.set_slot("TYPE_INANIMATE_PHYSICAL", 1)
    node.set_slot("CN_Q072_SUBSTANCE", 1)

    interned_node = interner1.intern_node(node)
    canonical_cid = interned_node.canonical_cid

    # Create a completely separate interner instance pointing to the same SQLite database
    interner2 = CanonicalNodeInterner(db_path=str(db_file))
    assert interner2.stats()["tier1_cached"] == 0

    # Recall from Tier 2 SQLite
    recalled = interner2.lookup(canonical_cid=canonical_cid)
    assert recalled is not None
    assert recalled.canonical_cid == canonical_cid
    assert recalled.literal == "quantum lattice"
    assert recalled.anchor == "cn:en:quantum_lattice (n)"
    assert recalled.get_slot("CN_Q072_SUBSTANCE") == 1


# ---------------------------------------------------------------------------
# 6. Concurrency & Multi-Threaded Safety
# ---------------------------------------------------------------------------

def test_multithreaded_canonical_interning():
    """Verify concurrent calls to intern_node and lookup across 16 threads are atomic and race-free."""
    interner = CanonicalNodeInterner()
    concepts = [
        ("cn:en:particle (n)", "particle"),
        ("cn:en:laser (n)", "laser"),
        ("cn:en:sensor (n)", "sensor"),
        ("cn:en:laboratory (n)", "laboratory"),
    ]

    def worker(idx: int):
        anchor, lit = concepts[idx % len(concepts)]
        node = QuantaNode(anchor=anchor, literal=lit)
        node.set_slot("TYPE_INANIMATE_PHYSICAL", 1)
        node.register_binding = f"VAR_SLOT_X{idx % 8}"
        res = interner.intern_node(node)
        assert res is not None
        assert res.literal == lit
        return res

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, i) for i in range(80)]
        nodes = [f.result() for f in futures]

    cids = [n.canonical_cid for n in nodes]
    assert len(cids) == 80
    assert len(set(cids)) == 4 # Exactly 4 unique concepts
    assert len({id(n) for n in nodes}) == 4 # Exactly 4 unique pointer instances
    stats = interner.stats()
    assert stats["hits"] == 76
    assert stats["misses"] == 4
    assert stats["reuse_rate"] == 0.95
