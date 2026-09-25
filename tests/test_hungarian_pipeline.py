"""Hungarian-Specific Autonomous NLP & Coreference Pipeline Tests (Section 8 / exp-012a).

Verifies Hungarian linguistic and neuro-symbolic processing:
1. Language-Agnostic Agglutinative Entity Matching: Suffix-chain tolerance in EntityMatcher
   (e.g., kutya -> kutyát, kutyának, kutyával; Kovács -> Kovácsnak, Kováccsal).
2. Dormant Entity Resurfacing via Paging Engine on Hungarian inflected surface mentions.
3. Multi-Turn Narrative Coreference: Persistent canonical identity and register binding across turns.
4. Hungarian ASG Structural Integrity & Thematic Valency Wiring.
5. Zero-Copy ConceptNet / MMap Lexical Grounding with English Pivot Labels.
6. Closed-Loop Round-Trip Lattice Meet Gate Soundness.
"""

import pytest

from core.asg import QuantaGraph, QuantaNode
from parser.asg_compiler import ASGCompiler
from parser.entity_manifest import EntityMatcher, EntityPagingEngine, EntityRecord
from parser.sexpr_parser import parse_sexpr
from parser.unsloth_transducer import MockUnslothTransducer
from pipeline.translator_pipeline import TwoWayTranslationPipeline
from verification.lattice_gate import LatticeInvarianceGate


# ---------------------------------------------------------------------------
# Hungarian Multi-Turn Narrative Fixtures (Task 8.4)
# ---------------------------------------------------------------------------

HU_TURN_1_TEXT = "Dr. Kovács János szintetizálta az új polimert a laboratóriumban."
HU_TURN_1_SEXPR = """(graph :chunk-id "hu_turn_1"
  (entity :id E1 :type PERSON :label "Dr. János Kovács" :surface ("Dr. Kovács János" "Kovács"))
  (entity :id E2 :type SUBSTANCE :label "synthetic polymer" :surface ("új polimert" "polimer"))
  (entity :id E3 :type LOCATION :label "laboratory" :surface "laboratóriumban")
  (event :id Ev1 :pred synthesize :agent E1 :patient E2 :location E3 :time "in the past" :tense PAST :polarity TRUE :raw-text "Dr. Kovács János szintetizálta az új polimert a laboratóriumban.")
  (relation :type TEMP_ALLEN_DURING :source Ev1 :target Ev1)
)"""

HU_TURN_2_TEXT = "Kovács úr ellenőrizte a mintát a mikroszkóp alatt."
HU_TURN_2_SEXPR = """(graph :chunk-id "hu_turn_2"
  (entity :id E1 :type PERSON :label "Dr. János Kovács" :surface "Kovács úr")
  (entity :id E4 :type OBJECT :label "specimen sample" :surface "mintát")
  (entity :id E5 :type INSTRUMENT :label "microscope" :surface "mikroszkóp")
  (event :id Ev2 :pred verify :agent E1 :patient E4 :instrument E5 :time "subsequently" :tense PAST :polarity TRUE :raw-text "Kovács úr ellenőrizte a mintát a mikroszkóp alatt.")
  (relation :type TEMP_ALLEN_MEETS :source Ev2 :target Ev2)
)"""

HU_TURN_3_TEXT = "A polimer nem bomlott le a magas hőmérsékleten."
HU_TURN_3_SEXPR = """(graph :chunk-id "hu_turn_3"
  (entity :id E2 :type SUBSTANCE :label "synthetic polymer" :surface "A polimer")
  (event :id Ev3 :pred decompose :patient E2 :time "under heat" :tense PAST :polarity FALSE :raw-text "A polimer nem bomlott le a magas hőmérsékleten.")
  (relation :type TEMP_ALLEN_MEETS :source Ev3 :target Ev3)
)"""


@pytest.fixture
def mock_hungarian_transducer():
    """MockUnslothTransducer configured with multi-turn Hungarian fixtures."""
    mock = MockUnslothTransducer()
    mock.register_fixture(HU_TURN_1_TEXT, HU_TURN_1_SEXPR)
    mock.register_fixture(HU_TURN_2_TEXT, HU_TURN_2_SEXPR)
    mock.register_fixture(HU_TURN_3_TEXT, HU_TURN_3_SEXPR)

    mock.register_realization_fixture(HU_TURN_1_SEXPR, HU_TURN_1_TEXT)
    mock.register_realization_fixture(HU_TURN_2_SEXPR, HU_TURN_2_TEXT)
    mock.register_realization_fixture(HU_TURN_3_SEXPR, HU_TURN_3_TEXT)
    return mock


@pytest.fixture
def compiler():
    return ASGCompiler()


# ---------------------------------------------------------------------------
# Task 8.3: Agglutinative Entity Matching Tests
# ---------------------------------------------------------------------------

def test_hungarian_fuzzy_entity_matching():
    """Verify EntityMatcher tolerates Hungarian agglutinative suffix chains without static tables."""
    matcher = EntityMatcher(enable_fuzzy=True, fuzzy_threshold=0.70)
    matcher.build_index({
        "kutya": ["E1"],
        "postás": ["E2"],
        "Kovács": ["E3"],
        "kert": ["E4"],
    })

    # Test agglutinative inflections:
    # 1. Accusative (-t), Dative (-nak/-nek), Instrumental (-val/-vel)
    text = "A kutyát és a kutyának szánt ételt a kertben hagyták."
    matches = matcher.match_chunk(text)
    assert "E1" in matches, "Expected 'kutyát' or 'kutyának' to match 'kutya'"
    assert "E4" in matches, "Expected 'kertben' to match 'kert'"

    # 2. Hungarian name inflections (Kovácsnak, Kováccsal)
    text_kovacs = "A mintát átadták Kovácsnak a kísérlet után."
    matches_kovacs = matcher.match_chunk(text_kovacs)
    assert "E3" in matches_kovacs, "Expected 'Kovácsnak' to match 'Kovács'"

    # 3. Postman inflections
    text_postas = "A kutya dühösen ugatott a postást látva."
    matches_postas = matcher.match_chunk(text_postas)
    assert "E1" in matches_postas
    assert "E2" in matches_postas, "Expected 'postást' to match 'postás'"


def test_hungarian_paging_engine_resurrection(tmp_path):
    """Verify EntityPagingEngine resurrects dormant entities when Hungarian inflected form appears."""
    db_file = tmp_path / "hungarian_entities.db"
    engine = EntityPagingEngine(db_path=db_file, max_size=5)

    # Register initial entity
    e1 = engine.register_new_entity(
        name="Kovács János",
        category="PERSON",
        aliases=["Kovács", "János"],
        chunk_idx=0,
    )
    assert engine.manifest.is_active(e1.canonical_id)

    # Fill manifest beyond max_size to force LRU eviction of e1
    for i in range(10):
        engine.register_new_entity(name=f"Entity_{i}", category="OBJECT", chunk_idx=1)

    assert not engine.manifest.is_active(e1.canonical_id), "E1 should be evicted to SQLite storage"

    # Pre-scan text with Hungarian inflected dative form 'Kovácsnak'
    test_chunk = "A laboratóriumi jegyzőkönyvet tegnap átadták Kovácsnak."
    resurrected, scan_time = engine.pre_scan_and_page(test_chunk, chunk_idx=12)

    assert any(r.canonical_id == e1.canonical_id for r in resurrected), (
        "Dormant entity 'Kovács' should be paged in via 'Kovácsnak'"
    )
    assert scan_time < 5.0
    assert engine.manifest.is_active(e1.canonical_id)
    assert engine.manifest.get(e1.canonical_id).register_binding is not None


# ---------------------------------------------------------------------------
# Task 8.5: Multi-Turn Narrative Coreference & ASG Integrity
# ---------------------------------------------------------------------------

def test_hungarian_multi_turn_coreference(mock_hungarian_transducer, compiler):
    """Verify multi-turn Hungarian discourse maintains entity identity and register bindings."""
    # Turn 1
    ext1 = mock_hungarian_transducer.transduce(text=HU_TURN_1_TEXT)
    g1 = compiler.compile(ext1)
    assert len(g1.nodes) >= 4
    # E1: Dr. János Kovács, E2: synthetic polymer, E3: laboratory
    e1_cid = g1.extraction_result.entities[0].id
    assert e1_cid == "E1"

    # Turn 2
    ext2 = mock_hungarian_transducer.transduce(text=HU_TURN_2_TEXT)
    g2 = compiler.compile(ext2)
    # E1 is reused for 'Kovács úr'
    assert ext2.entities[0].id == "E1"
    assert ext2.entities[0].canonical_name == "Dr. János Kovács"

    # Turn 3
    ext3 = mock_hungarian_transducer.transduce(text=HU_TURN_3_TEXT)
    g3 = compiler.compile(ext3)
    # E2 is reused for 'A polimer'
    assert ext3.entities[0].id == "E2"
    assert ext3.entities[0].canonical_name == "synthetic polymer"
    assert ext3.events[0].polarity is False  # 'nem bomlott le'


def test_hungarian_asg_structural_integrity(mock_hungarian_transducer, compiler):
    """Verify thematic valency edges in compiled Hungarian ASG."""
    ext = mock_hungarian_transducer.transduce(text=HU_TURN_1_TEXT)
    graph = compiler.compile(ext)

    # Find event node
    root_node = graph.get_node(graph.root_cid)
    assert root_node is not None

    # Check thematic edges
    assert "VAL_X1_AGENT" in root_node.edges
    assert "VAL_X2_PATIENT" in root_node.edges
    assert "VAL_LOCATION_SLOT" in root_node.edges

    # Validate structural integrity
    valid, errors = graph.validate_integrity()
    assert valid is True, f"Integrity errors: {errors}"


def test_hungarian_conceptnet_mmap_grounding(mock_hungarian_transducer, compiler):
    """Verify Hungarian concepts with English pivot labels ground into ConceptNet vectors."""
    ext = mock_hungarian_transducer.transduce(text=HU_TURN_1_TEXT)
    graph = compiler.compile(ext)

    for ent in ext.entities:
        node = graph.get_node(graph.entity_nodes[ent.id].cid)
        assert node is not None
        # Must have grounded anchor
        assert node.anchor != ""
        # Must have deterministic 256-bit BLAKE3 Merkle CID
        assert len(node.cid) == 64
        # Vector must be populated with active slots
        assert len(node.vector.active_slots()) > 0


def test_hungarian_lattice_invariance_roundtrip(mock_hungarian_transducer):
    """Verify closed-loop round-trip invariance on Hungarian narrative sentence."""
    pipeline = TwoWayTranslationPipeline(transducer=mock_hungarian_transducer)
    result = pipeline.round_trip(HU_TURN_1_TEXT, modality="hungarian")

    assert result.validation_pass is True
    assert result.hamming_distance == 0
    assert result.slot_preservation_rate >= 0.95
    assert result.is_meet_sound is True
    assert len(result.lattice_meet_errors) == 0
    assert result.is_invariant(max_hamming=0) is True
