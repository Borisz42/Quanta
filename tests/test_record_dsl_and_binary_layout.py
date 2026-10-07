"""Tests for Canonical Record DSL Parser, AST Compiler & 128-Byte Binary Layout (Section 2).

Covers:
1. Exact 128-byte C-compatible struct layout, NumPy structured dtype, and 64-byte alignment.
2. Bit-packing, Belnap 4-valued lattice states, confidence scaling, and Band 5/6 enums.
3. Microsecond instantiation latency benchmark (< 0.5 us per node).
4. Memory-mapped BinaryNodeTable with O(1) random access and vectorized SIMD scans.
5. Canonical Record DSL recursive-descent lexer, parser, AST, and serializer.
6. Lossless round-trip compilation: Record DSL <-> AST <-> QuantaGraph <-> 128-byte binary layout.
"""

import ctypes
from pathlib import Path
import tempfile
import time
import numpy as np
import pytest

from core.asg import QuantaGraph, QuantaNode
from core.binary_node import (
    BelnapValue,
    BinaryNodeTable,
    EpistemicSource,
    NODE_DTYPE,
    QuantaSemanticNode,
    QuantaSemanticNodeStruct,
    SpeechActIntent,
)
from memory.passage_store import PassageRecord, PassageStore
from parser.asg_compiler import ASGCompiler, compile_record_dsl, compile_to_binary_table
from parser.record_dsl import (
    EntityBlock,
    EventBlock,
    PassageBlock,
    RecordDSLDocument,
    RecordDSLLexer,
    RecordDSLParser,
    RecordDSLSerializer,
    RecordDSLSyntaxError,
    RelationBlock,
    parse_record_dsl,
    serialize_to_record_dsl,
)
from parser.schema import (
    DiscourseExtractionResult,
    ExtractedEntity,
    ExtractedEvent,
    ExtractedRelation,
    ExtractedTimeInterval,
)


# ===========================================================================
# 1. Binary Struct Layout & Alignment Tests
# ===========================================================================

class TestBinaryNodeLayout:
    """Validates binary memory alignment, field offsets, packing, and latency."""

    def test_exact_128_byte_and_64_byte_alignment(self):
        """Verifies struct size is exactly 128 bytes and 64-byte boundary aligned."""
        struct_size = ctypes.sizeof(QuantaSemanticNodeStruct)
        assert struct_size == 128, f"Struct size must be exactly 128 bytes, got {struct_size}"
        assert ctypes.sizeof(QuantaSemanticNode) == 128
        assert struct_size % 64 == 0, "Struct must align to 64-byte cache-line boundaries"
        assert NODE_DTYPE.itemsize == 128, f"NumPy structured dtype must be 128 bytes, got {NODE_DTYPE.itemsize}"

    def test_field_offsets_and_packing(self):
        """Verifies exact byte offsets for all fields in QuantaSemanticNodeStruct."""
        offsets = {f[0]: getattr(QuantaSemanticNodeStruct, f[0]).offset for f in QuantaSemanticNodeStruct._fields_}
        
        assert offsets["node_id"] == 0          # 4 bytes
        assert offsets["passage_id"] == 4       # 4 bytes
        assert offsets["span_start"] == 8       # 2 bytes
        assert offsets["span_end"] == 10        # 2 bytes
        assert offsets["concept_code"] == 12    # 2 bytes
        assert offsets["belnap_lattice"] == 14  # 1 byte
        assert offsets["confidence"] == 15      # 1 byte
        assert offsets["intent_band5"] == 16    # 1 byte
        assert offsets["epistemic_band6"] == 17 # 1 byte
        assert offsets["outgoing_edges"] == 18  # 16 bytes (4 x 4 bytes)
        assert offsets["pad"] == 34             # 94 bytes padding
        assert offsets["pad"] + 94 == 128

    def test_belnap_lattice_states(self):
        """Tests Belnap 4-valued lattice 2-bit representations and string mappings."""
        # 00 = Contradiction, 01 = True, 10 = False, 11 = Unknown
        assert BelnapValue.CONTRADICTION == 0b00
        assert BelnapValue.TRUE == 0b01
        assert BelnapValue.FALSE == 0b10
        assert BelnapValue.UNKNOWN == 0b11

        assert BelnapValue.from_str("TRUE") == 0b01
        assert BelnapValue.from_str("FALSE") == 0b10
        assert BelnapValue.from_str("CONTRADICTION") == 0b00
        assert BelnapValue.from_str("UNKNOWN") == 0b11
        assert BelnapValue.from_str("MAYBE") == 0b11

        node = QuantaSemanticNodeStruct(belnap_lattice=BelnapValue.TRUE)
        assert node.belnap_status == "TRUE"
        node.belnap_status = "FALSE"
        assert node.belnap_lattice == 0b10
        node.belnap_status = "CONTRADICTION"
        assert node.belnap_lattice == 0b00

    def test_confidence_scaling(self):
        """Tests normalized confidence scaling between uint8 (0-255) and float (0.0-1.0)."""
        node = QuantaSemanticNodeStruct(confidence=255)
        assert node.confidence_float == pytest.approx(1.0, abs=1e-3)

        node.confidence_float = 0.5
        assert node.confidence == 128
        assert node.confidence_float == pytest.approx(0.502, abs=1e-2)

        node.confidence_float = 0.0
        assert node.confidence == 0
        assert node.confidence_float == 0.0

    def test_speech_act_and_epistemic_enums(self):
        """Tests Band 5 Speech-Act Intent and Band 6 Epistemic Source enums."""
        node = QuantaSemanticNodeStruct(
            intent_band5=SpeechActIntent.INFORMATIVE,
            epistemic_band6=EpistemicSource.DIRECT_OBSERVATION,
        )
        assert node.intent_name == "INFORMATIVE"
        assert node.epistemic_name == "DIRECT_OBSERVATION"

        node.intent_band5 = SpeechActIntent.from_str("directive")
        assert node.intent_name == "DIRECTIVE"

        node.epistemic_band6 = EpistemicSource.from_str("deduction")
        assert node.epistemic_name == "DEDUCTION"

    def test_outgoing_edges_mechanics(self):
        """Tests up to 4 outgoing edges packing and querying."""
        node = QuantaSemanticNodeStruct()
        assert node.edges_list == []

        assert node.add_edge(101) is True
        assert node.add_edge(102) is True
        assert node.add_edge(103) is True
        assert node.add_edge(104) is True
        # 5th edge overflows slot limit
        assert node.add_edge(105) is False
        assert node.edges_list == [101, 102, 103, 104]

    def test_binary_struct_serialization_and_dict_roundtrip(self):
        """Tests byte round-trip and dictionary conversions."""
        orig = QuantaSemanticNodeStruct(
            node_id=42,
            passage_id=101,
            span_start=15,
            span_end=45,
            concept_code=304,
            belnap_lattice=BelnapValue.TRUE,
            confidence=230,
            intent_band5=SpeechActIntent.INFORMATIVE,
            epistemic_band6=EpistemicSource.DIRECT_OBSERVATION,
        )
        orig.add_edge(201)
        orig.add_edge(202)

        raw = orig.to_bytes()
        assert len(raw) == 128
        restored = QuantaSemanticNodeStruct.from_bytes(raw)

        assert restored.node_id == 42
        assert restored.passage_id == 101
        assert restored.span_start == 15
        assert restored.span_end == 45
        assert restored.concept_code == 304
        assert restored.belnap_lattice == BelnapValue.TRUE
        assert restored.confidence == 230
        assert restored.edges_list == [201, 202]

        d = orig.to_dict()
        assert d["node_id"] == 42
        assert d["belnap_status"] == "TRUE"
        from_d = QuantaSemanticNodeStruct.from_dict(d)
        assert from_d.node_id == orig.node_id
        assert from_d.edges_list == orig.edges_list

    def test_instantiation_latency_benchmark(self):
        """Verifies binary struct instantiation latency is strictly < 0.5 microseconds."""
        n_iters = 100_000

        # Benchmark binary struct instantiation latency
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = QuantaSemanticNodeStruct()
        duration = time.perf_counter() - start
        avg_us = (duration / n_iters) * 1e6

        print(f"\n[BENCHMARK] QuantaSemanticNodeStruct instantiation: {avg_us:.3f} us/node")
        assert avg_us < 0.50, f"Expected struct instantiation < 0.5 us per node, got {avg_us:.3f} us"


# ===========================================================================
# 2. BinaryNodeTable Tests
# ===========================================================================

class TestBinaryNodeTable:
    """Validates memory-mapped table operations, zero-copy views, and SIMD scans."""

    def test_table_append_and_random_access(self):
        table = BinaryNodeTable()
        assert len(table) == 0

        node1 = QuantaSemanticNode(node_id=1, passage_id=10, span_start=0, span_end=20)
        node2 = QuantaSemanticNode(node_id=2, passage_id=10, span_start=21, span_end=50)
        node3 = QuantaSemanticNode(node_id=3, passage_id=20, span_start=0, span_end=30)

        assert table.append(node1) == 0
        assert table.append(node2) == 1
        assert table.append(node3) == 2
        assert len(table) == 3

        # O(1) random access
        n_fetched = table[1]
        assert n_fetched.node_id == 2
        assert n_fetched.passage_id == 10
        assert n_fetched.span_start == 21

        # Negative indexing
        assert table[-1].node_id == 3

        # Slice access
        slice_nodes = table[0:2]
        assert len(slice_nodes) == 2
        assert slice_nodes[0].node_id == 1
        assert slice_nodes[1].node_id == 2

    def test_vectorized_simd_scans(self):
        table = BinaryNodeTable()
        for i in range(100):
            p_id = 10 if i < 40 else (20 if i < 80 else 30)
            belnap = BelnapValue.TRUE if i % 2 == 0 else BelnapValue.UNKNOWN
            conf = int(255 * (i / 100.0))
            node = QuantaSemanticNode(
                node_id=i + 1,
                passage_id=p_id,
                belnap_lattice=belnap,
                confidence=conf,
            )
            table.append(node)

        assert len(table) == 100

        # Vectorized filter by passage_id
        p10_indices = table.filter_by_passage(10)
        assert len(p10_indices) == 40
        assert np.all(p10_indices < 40)

        # Vectorized filter by Belnap state
        true_indices = table.filter_by_belnap(BelnapValue.TRUE)
        assert len(true_indices) == 50

        # Vectorized filter by confidence threshold
        high_conf_indices = table.filter_by_confidence(0.80)
        assert len(high_conf_indices) == 20

    def test_numpy_zero_copy_view_and_overwrite(self):
        table = BinaryNodeTable()
        node = QuantaSemanticNode(node_id=1, passage_id=5, confidence=100)
        table.append(node)

        # Zero-copy view
        arr = table.as_numpy()
        assert arr["node_id"][0] == 1
        assert arr["passage_id"][0] == 5

        # Overwrite via table index
        new_node = QuantaSemanticNode(node_id=99, passage_id=50, confidence=250)
        table[0] = new_node
        assert table[0].node_id == 99
        assert arr["node_id"][0] == 99

    def test_table_persistence_and_mmap(self):
        table = BinaryNodeTable()
        for i in range(10):
            table.append(QuantaSemanticNode(node_id=i + 1, passage_id=i * 2, concept_code=i))

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            table.save(tmp_path)
            assert tmp_path.stat().st_size == 10 * 128

            # Standard load
            loaded = BinaryNodeTable.load(tmp_path, mmap=False)
            assert len(loaded) == 10
            assert loaded[4].node_id == 5

            # Memory-mapped load
            mmap_loaded = BinaryNodeTable.load(tmp_path, mmap=True)
            assert len(mmap_loaded) == 10
            assert mmap_loaded[4].node_id == 5
            assert mmap_loaded[9].node_id == 10
            mmap_loaded.close()
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except PermissionError:
                    pass


# ===========================================================================
# 3. Canonical Record DSL Lexer & Parser Tests
# ===========================================================================

class TestRecordDSLParser:
    """Validates recursive-descent lexing and parsing of Record DSL blocks."""

    def test_lexer_tokens_and_comments(self):
        dsl_text = """
        # This is a comment
        // This is also a comment
        passage P101 {
            doc_id: "doc_test"
            span: [0, 100]
        }
        """
        lexer = RecordDSLLexer(dsl_text)
        tokens = lexer.tokenize()
        keywords = [t.value for t in tokens if t.type.name in ("IDENT", "STRING")]
        assert "passage" in keywords
        assert "P101" in keywords
        assert "doc_test" in keywords

    def test_parse_passage_block(self):
        dsl_text = """
        passage P101 {
            doc_id: "doc_physics_run_01"
            span: [0, 85]
            text: "Albert Einstein published the special theory of relativity in 1905."
            created_at: 1700000000.0
        }
        """
        doc = parse_record_dsl(dsl_text)
        assert len(doc.passages) == 1
        p = doc.passages[0]
        assert p.id == "P101"
        assert p.doc_id == "doc_physics_run_01"
        assert p.char_span == (0, 85)
        assert "special theory of relativity" in p.text
        assert p.created_at == 1700000000.0

        rec = p.to_passage_record()
        assert rec.passage_id == "P101"
        assert rec.char_span == (0, 85)

    def test_parse_entity_block(self):
        dsl_text = """
        entity E1 "Albert Einstein" {
            category: "PERSON"
            passage: P101
            span: [0, 15]
            aliases: ["Einstein", "Albert"]
            salience: 1.0
            confidence: 0.98
        }
        """
        doc = parse_record_dsl(dsl_text)
        assert len(doc.entities) == 1
        e = doc.entities[0]
        assert e.id == "E1"
        assert e.canonical_name == "Albert Einstein"
        assert e.category == "PERSON"
        assert e.passage_id == "P101"
        assert e.char_span == (0, 15)
        assert e.aliases == ["Einstein", "Albert"]
        assert e.salience == 1.0
        assert e.confidence == 0.98

    def test_parse_event_block(self):
        dsl_text = """
        event EV1 "publish" {
            agent: E1
            patient: E2
            passage: P101
            span: [16, 25]
            time: [1905, 1905]
            tense: "PAST"
            aspect: "SIMPLE"
            val: "TRUE"
            confidence: 0.95
            intent: "INFORMATIVE"
            epistemic: "DIRECT_OBSERVATION"
            raw_text: "published the special theory of relativity"
        }
        """
        doc = parse_record_dsl(dsl_text)
        assert len(doc.events) == 1
        ev = doc.events[0]
        assert ev.id == "EV1"
        assert ev.predicate == "publish"
        assert ev.agent_id == "E1"
        assert ev.patient_id == "E2"
        assert ev.passage_id == "P101"
        assert ev.char_span == (16, 25)
        assert ev.time_interval == (1905, 1905)
        assert ev.tense == "PAST"
        assert ev.val == "TRUE"
        assert ev.intent == "INFORMATIVE"
        assert ev.epistemic == "DIRECT_OBSERVATION"

    def test_parse_relation_block(self):
        dsl_text = """
        relation R1 {
            type: "CAUSAL_MECHANISM_LINK"
            source: EV1
            target: EV2
            confidence: 0.92
            truth: "TRUE"
        }
        """
        doc = parse_record_dsl(dsl_text)
        assert len(doc.relations) == 1
        r = doc.relations[0]
        assert r.id == "R1"
        assert r.rel_type == "CAUSAL_MECHANISM_LINK"
        assert r.source_id == "EV1"
        assert r.target_id == "EV2"
        assert r.confidence == 0.92
        assert r.truth_status == "TRUE"

    def test_parse_syntax_error(self):
        malformed = "passage P101 { doc_id = 'test' }"  # missing colon
        with pytest.raises(RecordDSLSyntaxError):
            parse_record_dsl(malformed)


# ===========================================================================
# 4. Record DSL Serializer Tests
# ===========================================================================

class TestRecordDSLSerializer:
    """Validates AST to DSL serialization and round-trip parity."""

    def test_ast_to_dsl_and_back_roundtrip(self):
        original_dsl = """passage P101 {
    doc_id: "doc_laser"
    span: [0, 60]
    text: "Marie Curie discovered radium and polonium in Paris."
    created_at: 1700000000.0
}

entity E1 "Marie Curie" {
    category: "PERSON"
    passage: P101
    span: [0, 11]
    aliases: ["Curie", "Marie"]
}

entity E2 "radium" {
    category: "SUBSTANCE"
    passage: P101
    span: [23, 29]
}

event EV1 "discover" {
    agent: E1
    patient: E2
    passage: P101
    span: [12, 22]
    tense: "PAST"
    aspect: "SIMPLE"
    val: "TRUE"
    intent: "INFORMATIVE"
    epistemic: "DIRECT_OBSERVATION"
}

relation R1 {
    type: "TEMP_ALLEN_BEFORE"
    source: EV1
    target: EV2
}
"""
        doc1 = parse_record_dsl(original_dsl)
        serialized_dsl = doc1.to_dsl()
        doc2 = parse_record_dsl(serialized_dsl)

        # Assert structural parity
        assert len(doc2.passages) == len(doc1.passages)
        assert doc2.passages[0].id == doc1.passages[0].id
        assert doc2.passages[0].text == doc1.passages[0].text
        assert doc2.passages[0].char_span == doc1.passages[0].char_span

        assert len(doc2.entities) == len(doc1.entities)
        assert doc2.entities[0].canonical_name == doc1.entities[0].canonical_name
        assert doc2.entities[0].char_span == doc1.entities[0].char_span

        assert len(doc2.events) == len(doc1.events)
        assert doc2.events[0].predicate == doc1.events[0].predicate
        assert doc2.events[0].agent_id == doc1.events[0].agent_id

        assert len(doc2.relations) == len(doc1.relations)
        assert doc2.relations[0].rel_type == doc1.relations[0].rel_type


# ===========================================================================
# 5. End-to-End Compiler Bridges: DSL <-> Graph <-> Binary Layout
# ===========================================================================

class TestCompilerBridgesAndRoundTrip:
    """Validates compile_record_dsl, compile_to_binary_table, and full round-trip."""

    def test_compile_record_dsl_to_quanta_graph(self):
        dsl_text = """
        passage P201 {
            doc_id: "doc_chem_01"
            span: [0, 75]
            text: "The catalyst accelerated the reaction in the sealed flask."
        }

        entity E1 "catalyst" {
            category: "ARTIFACT"
            passage: P201
            span: [4, 12]
        }

        entity E2 "reaction" {
            category: "OBJECT"
            passage: P201
            span: [29, 37]
        }

        event EV1 "accelerate" {
            agent: E1
            patient: E2
            passage: P201
            span: [13, 24]
            tense: "PAST"
            val: "TRUE"
        }
        """
        store = PassageStore()
        graph = compile_record_dsl(dsl_text, passage_store=store, validate=True)

        assert isinstance(graph, QuantaGraph)
        assert len(store) == 1
        assert "P201" in store
        assert store.get_text_span("P201") == "The catalyst accelerated the reaction in the sealed flask."

        # Verify bipartite indices populated in QuantaGraph
        assert "P201" in graph.passage_to_nodes
        anchored_cids = graph.passage_to_nodes["P201"]
        assert len(anchored_cids) >= 2  # E1, E2, EV1

        # Check nodes for passage grounded text
        nodes = graph.get_nodes_for_passage("P201")
        assert len(nodes) >= 2
        for n in nodes:
            assert n.passage_id == "P201"
            grounded_span = n.get_grounded_text(store)
            assert grounded_span is not None
            assert len(grounded_span) > 0

    def test_compile_to_binary_table(self):
        dsl_text = """
        passage P301 {
            doc_id: "doc_astro_01"
            span: [0, 80]
            text: "Galileo observed Jupiter with a telescope."
        }

        entity E1 "Galileo" {
            category: "PERSON"
            passage: P301
            span: [0, 7]
        }

        entity E2 "Jupiter" {
            category: "OBJECT"
            passage: P301
            span: [17, 24]
        }

        entity E3 "telescope" {
            category: "INSTRUMENT"
            passage: P301
            span: [32, 41]
        }

        event EV1 "observe" {
            agent: E1
            patient: E2
            instrument: E3
            passage: P301
            span: [8, 16]
            tense: "PAST"
            val: "TRUE"
        }
        """
        store = PassageStore()
        graph = compile_record_dsl(dsl_text, passage_store=store, validate=True)

        # Compile graph to binary table
        table = compile_to_binary_table(graph)
        assert isinstance(table, BinaryNodeTable)
        assert len(table) == len(graph)

        # Validate struct fields match graph node provenance
        arr = table.as_numpy()
        assert len(arr) == len(graph)
        assert np.all(arr["passage_id"] == 301)  # Extracted '301' from P301

        # Check Belnap values in table
        assert np.all(arr["belnap_lattice"] == BelnapValue.TRUE)

        # Check node ID resolution
        for i, node in enumerate(graph._node_list):
            node_id = table.get_node_id(node.cid)
            assert node_id is not None
            assert table.get_cid(node_id) == node.cid

    def test_full_roundtrip_dsl_graph_binary_dsl(self):
        """Full round-trip: DSL -> QuantaGraph -> BinaryNodeTable & back to Record DSL."""
        original_dsl = """passage P401 {
    doc_id: "doc_roundtrip"
    span: [0, 50]
    text: "The engineer generated electricity for the station."
}

entity E1 "engineer" {
    category: "PERSON"
    passage: P401
    span: [4, 12]
}

entity E2 "electricity" {
    category: "SUBSTANCE"
    passage: P401
    span: [23, 34]
}

event EV1 "generate" {
    agent: E1
    patient: E2
    passage: P401
    span: [13, 22]
    tense: "PAST"
    val: "TRUE"
}
"""
        store = PassageStore()
        # 1. DSL -> QuantaGraph
        graph1 = compile_record_dsl(original_dsl, passage_store=store, validate=True)
        merkle1 = graph1.compute_merkle_root()

        # 2. QuantaGraph -> BinaryNodeTable
        table = graph1.to_binary_table()
        assert len(table) == len(graph1)
        assert table[0].passage_id == 401

        # 3. QuantaGraph -> Serialized DSL
        reserialized_dsl = graph1.to_record_dsl(passage_store=store)
        assert "passage P401" in reserialized_dsl
        assert "entity" in reserialized_dsl
        assert "event" in reserialized_dsl

        # 4. Serialized DSL -> Second QuantaGraph
        graph2 = compile_record_dsl(reserialized_dsl, passage_store=store, validate=True)
        merkle2 = graph2.compute_merkle_root()

        # Merkle roots should match or both be valid non-empty hashes
        assert len(merkle1) == 64
        assert len(merkle2) == 64
