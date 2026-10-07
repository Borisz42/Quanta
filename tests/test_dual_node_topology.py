"""Tests for Dual-Node Bipartite Graph Schema & Passage Storage Engine (Section 1).

Covers:
1. PassageRecord dataclass and span mechanics.
2. PassageStore in-memory and persistent SQLite operations.
3. Bipartite link integrity between QuantaNode instances and PassageRecord.
4. Sub-graph cloning and Merkle hashing invariance when passage pointers are attached.
"""

import pytest
import tempfile
from pathlib import Path

from core.asg import QuantaGraph, QuantaNode, fold_subgraph, unfold_subgraph
from core.types import QuantaVector
from memory.passage_store import PassageRecord, PassageStore


class TestPassageRecord:
    """Tests for PassageRecord dataclass."""

    def test_passage_record_creation(self):
        rec = PassageRecord(
            passage_id="P101",
            doc_id="doc_physics_run_01",
            char_span=(0, 120),
            text="Albert Einstein discovered the photoelectric effect in 1905, establishing quantum theory.",
        )
        assert rec.passage_id == "P101"
        assert rec.doc_id == "doc_physics_run_01"
        assert rec.char_span == (0, 120)
        assert rec.span_start == 0
        assert rec.span_end == 120
        assert len(rec) == len(rec.text)
        assert rec.created_at > 0

    def test_passage_record_validation(self):
        with pytest.raises(ValueError, match="cannot exceed end"):
            PassageRecord(
                passage_id="P_ERR",
                doc_id="doc_err",
                char_span=(50, 10),
                text="Invalid span",
            )

        with pytest.raises(ValueError, match="must be a 2-tuple"):
            PassageRecord(
                passage_id="P_ERR2",
                doc_id="doc_err",
                char_span=(10,),  # type: ignore
                text="Invalid span length",
            )

    def test_passage_record_dict_roundtrip(self):
        rec = PassageRecord(
            passage_id="P102",
            doc_id="doc_corpus_alpha",
            char_span=(150, 280),
            text="Quantum entanglement occurs when a pair or group of particles interact.",
            created_at=1700000000.0,
        )
        data = rec.to_dict()
        restored = PassageRecord.from_dict(data)
        assert restored.passage_id == rec.passage_id
        assert restored.doc_id == rec.doc_id
        assert restored.char_span == rec.char_span
        assert restored.text == rec.text
        assert restored.created_at == rec.created_at


class TestPassageStore:
    """Tests for PassageStore operations (in-memory and SQLite)."""

    def test_in_memory_crud_and_span_retrieval(self):
        store = PassageStore()
        text_p1 = "In 1687, Isaac Newton published Philosophiæ Naturalis Principia Mathematica."
        rec1 = store.add_passage(
            PassageRecord(
                passage_id="P201",
                doc_id="doc_principia",
                char_span=(0, len(text_p1)),
                text=text_p1,
            )
        )
        assert len(store) == 1
        assert "P201" in store
        assert store.get_passage("P201") == rec1

        # Full text
        assert store.get_text_span("P201") == text_p1

        # Relative slice
        span = store.get_text_span("P201", 9, 21)
        assert span == "Isaac Newton"

        # Missing passage returns None
        assert store.get_text_span("P_NONEXISTENT", 0, 10) is None

    def test_absolute_document_offset_mapping(self):
        store = PassageStore()
        doc_text = "Before text. The quick brown fox jumps over the lazy dog. After text."
        # Passage covers characters 13 to 57 in document
        p_text = "The quick brown fox jumps over the lazy dog."
        p_start = 13
        p_end = p_start + len(p_text)
        store.add_passage(
            PassageRecord(
                passage_id="P202",
                doc_id="doc_animals",
                char_span=(p_start, p_end),
                text=p_text,
            )
        )

        # Query using absolute document offsets (e.g., offsets for 'quick brown fox' in doc are 17 to 32)
        doc_fox_start = 13 + p_text.index("quick brown fox")
        doc_fox_end = doc_fox_start + len("quick brown fox")
        result = store.get_text_span("P202", doc_fox_start, doc_fox_end)
        assert result == "quick brown fox"

        # Query using relative passage offsets (index 4 to 19 within p_text)
        rel_result = store.get_text_span("P202", 4, 19)
        assert rel_result == "quick brown fox"

    def test_batch_insertion_and_ordering(self):
        store = PassageStore()
        passages = [
            PassageRecord(
                passage_id=f"P_{i}",
                doc_id="doc_history",
                char_span=(i * 100, (i + 1) * 100),
                text=f"Paragraph {i} content text.",
            )
            for i in [3, 1, 4, 0, 2]
        ]
        stored = store.add_passages(passages)
        assert len(stored) == 5
        assert len(store) == 5

        # Order by span_start
        ordered = store.get_passages_for_doc("doc_history")
        assert [p.passage_id for p in ordered] == ["P_0", "P_1", "P_2", "P_3", "P_4"]

        # Offset lookup
        found = store.find_passage_at_offset("doc_history", 250)
        assert found is not None
        assert found.passage_id == "P_2"

        # Missing offset
        assert store.find_passage_at_offset("doc_history", 9999) is None

    def test_sqlite_persistence_reload(self, tmp_path):
        db_path = tmp_path / "test_passages.db"
        store1 = PassageStore(db_path=db_path)
        store1.add_passage(
            PassageRecord(
                passage_id="P_PERSIST_1",
                doc_id="doc_persistent",
                char_span=(0, 50),
                text="First persistent passage stored on disk.",
            )
        )
        store1.add_passage(
            PassageRecord(
                passage_id="P_PERSIST_2",
                doc_id="doc_persistent",
                char_span=(50, 110),
                text="Second persistent passage stored in SQLite database.",
            )
        )
        store1.close()

        # Reload in a fresh instance
        store2 = PassageStore(db_path=db_path)
        assert len(store2) == 2
        assert "P_PERSIST_1" in store2
        assert "P_PERSIST_2" in store2
        assert (
            store2.get_text_span("P_PERSIST_1", 0, 16)
            == "First persistent"
        )
        assert (
            store2.get_text_span("P_PERSIST_2", 0, 17)
            == "Second persistent"
        )
        store2.close()

    def test_deletion_and_clear(self):
        store = PassageStore()
        store.add_passage(
            PassageRecord("P_DEL1", "doc1", (0, 10), "Sample 1")
        )
        store.add_passage(
            PassageRecord("P_DEL2", "doc1", (10, 20), "Sample 2")
        )
        assert len(store) == 2

        assert store.delete_passage("P_DEL1") is True
        assert len(store) == 1
        assert "P_DEL1" not in store

        store.clear()
        assert len(store) == 0


class TestBipartiteLinkIntegrity:
    """Tests for bipartite grounding links connecting semantic ASG nodes and passages."""

    def test_node_passage_binding(self):
        store = PassageStore()
        text = "Marie Curie received two Nobel Prizes in Physics and Chemistry."
        store.add_passage(
            PassageRecord(
                passage_id="P301",
                doc_id="doc_nobel",
                char_span=(0, len(text)),
                text=text,
            )
        )

        node = QuantaNode(anchor="cn:en:curie(n)", literal="Marie Curie")
        assert node.passage_id is None
        assert node.get_grounded_text(store) is None

        # Bind to passage span "Marie Curie" [0, 11]
        node.bind_passage("P301", 0, 11, confidence=0.98)
        assert node.passage_id == "P301"
        assert node.span_start == 0
        assert node.span_end == 11
        assert node.confidence == 0.98
        assert node.get_grounded_text(store) == "Marie Curie"

    def test_graph_bipartite_indexing(self):
        store = PassageStore()
        passage_text = "Niels Bohr proposed the atomic model with electron orbits in Copenhagen."
        store.add_passage(
            PassageRecord(
                passage_id="P302",
                doc_id="doc_physics",
                char_span=(0, len(passage_text)),
                text=passage_text,
            )
        )

        graph = QuantaGraph()
        # Semantic nodes
        node_bohr = QuantaNode(anchor="cn:en:bohr(n)", literal="Niels Bohr")
        node_atom = QuantaNode(anchor="cn:en:atom_model(n)", literal="atomic model")
        node_city = QuantaNode(anchor="cn:en:copenhagen(n)", literal="Copenhagen")

        bohr_cid = graph.add_node(node_bohr)
        atom_cid = graph.add_node(node_atom)
        city_cid = graph.add_node(node_city)

        # Ground nodes using add_passage_anchor
        graph.add_passage_anchor(bohr_cid, "P302", 0, 10)  # "Niels Bohr"
        graph.add_passage_anchor(atom_cid, "P302", 24, 36)  # "atomic model"
        graph.add_passage_anchor(city_cid, "P302", 61, 71)  # "Copenhagen"

        assert graph.node_to_passage[bohr_cid] == ("P302", 0, 10)
        assert graph.node_to_passage[atom_cid] == ("P302", 24, 36)
        assert graph.node_to_passage[city_cid] == ("P302", 61, 71)

        grounded_nodes = graph.get_nodes_for_passage("P302")
        grounded_cids = {n.cid for n in grounded_nodes}
        assert grounded_cids == {bohr_cid, atom_cid, city_cid}

        # Verify ground text via store
        assert node_bohr.get_grounded_text(store) == "Niels Bohr"
        assert node_atom.get_grounded_text(store) == "atomic model"
        assert node_city.get_grounded_text(store) == "Copenhagen"

    def test_bipartite_serialization_roundtrip(self):
        graph = QuantaGraph()
        node = QuantaNode(
            anchor="cn:en:photon(n)",
            literal="photon",
            passage_id="P400",
            span_start=15,
            span_end=21,
            salience=0.85,
            truth_status="TRUE",
            confidence=0.92,
            evidence_source="direct_observation",
        )
        cid = graph.add_node(node)
        assert "P400" in graph.passage_to_nodes
        assert cid in graph.passage_to_nodes["P400"]

        serialized = graph.to_dict()
        restored = QuantaGraph.from_dict(serialized)

        restored_node = restored.get_node(cid)
        assert restored_node is not None
        assert restored_node.passage_id == "P400"
        assert restored_node.span_start == 15
        assert restored_node.span_end == 21
        assert restored_node.salience == 0.85
        assert restored_node.truth_status == "TRUE"
        assert restored_node.confidence == 0.92
        assert restored_node.evidence_source == "direct_observation"

        assert "P400" in restored.passage_to_nodes
        assert cid in restored.passage_to_nodes["P400"]
        assert restored.node_to_passage[cid] == ("P400", 15, 21)


class TestMerkleInvarianceAndCloning:
    """Tests that attaching passage provenance preserves Merkle hashing invariance and cloning."""

    def test_merkle_hashing_invariance_on_node(self):
        # 1. Base semantic node without passage grounding
        vec = QuantaVector.zeros()
        vec["TYPE_EVENT"] = 1
        vec["WN_ACT_ACTION"] = 1

        node = QuantaNode(
            vector=vec,
            anchor="cn:en:synthesize(v)",
            literal="synthesized",
        )
        base_cid = node.compute_cid()
        base_canonical_cid = node.compute_canonical_cid()

        # 2. Attach passage grounding metadata
        node.bind_passage("P500", 12, 23, confidence=0.75)
        node.salience = 0.6
        node.truth_status = "TRUE"
        node.evidence_source = "retrieved_passage"

        # Provenance must NOT alter the node's cryptographic Content Identifier
        new_cid = node.compute_cid()
        new_canonical_cid = node.compute_canonical_cid()

        assert new_cid == base_cid
        assert new_canonical_cid == base_canonical_cid

    def test_graph_merkle_root_invariance(self):
        graph = QuantaGraph()
        n1 = QuantaNode(anchor="cn:en:electron(n)", literal="electron")
        n2 = QuantaNode(anchor="cn:en:orbit(v)", literal="orbits")
        c1 = graph.add_node(n1)
        c2 = graph.add_node(n2)
        graph.add_edge(c2, "ARG1", c1)

        base_merkle = graph.compute_merkle_root()

        # Add passage anchors to nodes in graph
        graph.add_passage_anchor(c1, "P601", 0, 8)
        graph.add_passage_anchor(c2, "P601", 9, 15)

        post_anchor_merkle = graph.compute_merkle_root()
        assert post_anchor_merkle == base_merkle

    def test_subgraph_cloning_integrity(self):
        graph = QuantaGraph()
        n1 = QuantaNode(
            anchor="cn:en:laser(n)",
            literal="laser",
            passage_id="P700",
            span_start=5,
            span_end=10,
            confidence=0.95,
        )
        n2 = QuantaNode(
            anchor="cn:en:beam(n)",
            literal="beam",
            passage_id="P700",
            span_start=11,
            span_end=15,
            confidence=0.88,
        )
        c1 = graph.add_node(n1)
        c2 = graph.add_node(n2)
        graph.add_edge(c1, "EMITS", c2)

        # Clone
        cloned = graph.copy()
        assert cloned.compute_merkle_root() == graph.compute_merkle_root()
        assert len(cloned) == len(graph)
        assert cloned.passage_to_nodes["P700"] == graph.passage_to_nodes["P700"]
        assert cloned.node_to_passage[c1] == ("P700", 5, 10)
        assert cloned.node_to_passage[c2] == ("P700", 11, 15)

        # Mutation of clone does not mutate original
        cloned_n1 = cloned.get_node(c1)
        assert cloned_n1 is not None
        cloned_n1.span_start = 999
        assert graph.get_node(c1).span_start == 5

    def test_merkle_folding_and_unfolding_with_passage_anchors(self):
        graph = QuantaGraph()
        n_root = QuantaNode(anchor="cn:en:experiment(n)", literal="experiment")
        n_sub1 = QuantaNode(
            anchor="cn:en:sample(n)",
            literal="sample A",
            passage_id="P800",
            span_start=10,
            span_end=18,
        )
        n_sub2 = QuantaNode(
            anchor="cn:en:temperature(n)",
            literal="300K",
            passage_id="P800",
            span_start=25,
            span_end=29,
        )

        root_cid = graph.add_node(n_root, set_as_root=True)
        sub1_cid = graph.add_node(n_sub1)
        sub2_cid = graph.add_node(n_sub2)

        graph.add_edge(sub1_cid, "HAS_PROP", sub2_cid)
        graph.add_edge(root_cid, "USES", sub1_cid)
        current_sub1_cid = n_sub1.cid

        # Fold subtree starting at sub1
        storage = {}
        ptr_cid, sub_merkle = fold_subgraph(graph, sub1_cid, storage=storage)

        assert ptr_cid in graph.nodes
        assert sub_merkle in storage

        # Unfold subtree back
        restored_sub_cid = unfold_subgraph(ptr_cid, storage=storage, target_graph=graph)
        assert restored_sub_cid == current_sub1_cid

        restored_node = graph.get_node(restored_sub_cid)
        assert restored_node is not None
        assert restored_node.passage_id == "P800"
        assert restored_node.span_start == 10
        assert restored_node.span_end == 18

        nodes_in_p800 = graph.get_nodes_for_passage("P800")
        assert len(nodes_in_p800) >= 2
