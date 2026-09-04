"""Tests for Phase 5: Hierarchical Merkle Sub-Graph Folding (Book-Scale Memory).

Verifies:
- 5.1: fold_discourse_episode produces a 32-byte Merkle fold node with aggregate quaternary vector
       and intact external entity references.
- 5.2: Hierarchical folding: combining chunk fold nodes into Chapter Merkle roots and Book Merkle roots.
- 5.3: Dynamic unfolding: unfold_subgraph verifying exact BLAKE3/SHA-256 cryptographic integrity.
- 5.4: 10-chunk narrative tamper detection: any tamper in Chunk 3 invalidates the Book Merkle root.
- PageTableStorage in-memory and SQLite-backed storage persistence.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import pytest

from core.asg import (
    QuantaGraph,
    QuantaNode,
    HierarchicalMerkleBook,
    fold_book,
    fold_chapter,
    fold_discourse_episode,
    unfold_subgraph,
)
from core.page_table import PageTableStorage
from core.types import QuantaVector, RegisterValue, StructuralValue


def _build_test_episode_graph(chunk_id: str = "chunk_0001") -> QuantaGraph:
    """Helper to build a representative discourse episode graph with events and entities."""
    graph = QuantaGraph()

    # External entities (Working memory characters / objects)
    e1_eleanor = QuantaNode(
        vector={
            "TYPE_HUMAN": 1,
            "TYPE_ANIMATE": 1,
            "ROLE_AGENT_CAPABLE": 1,
            "ROLE_SENTIENT": 1,
            "GRAPH_VARIABLE_BIND": 1,
            "GRAPH_LEAF": 1,
            "VAR_SLOT_X0": int(RegisterValue.BOUND_LOCAL),
        },
        anchor="cn:en:eleanor_vance (n)",
        literal={"id": "E1", "name": "Dr. Eleanor Vance"},
    )
    e2_compound = QuantaNode(
        vector={
            "CN_Q072_SUBSTANCE": 1,
            "TYPE_INANIMATE_PHYSICAL": 1,
            "GRAPH_VARIABLE_BIND": 1,
            "GRAPH_LEAF": 1,
            "VAR_SLOT_X1": int(RegisterValue.BOUND_LOCAL),
        },
        anchor="cn:en:polymer (n)",
        literal={"id": "E2", "name": "synthetic polymer compound"},
    )
    e1_cid = graph.add_node(e1_eleanor)
    e2_cid = graph.add_node(e2_compound)

    # Internal episode events (Ev1: synthesize, Ev2: observe, Ev3: verify)
    ev1 = QuantaNode(
        vector={
            "TYPE_EVENT": 1,
            "WN_ACT_ACTION": 1,
            "NSM_DO": 1,
            "LJB_PU_PAST_TENSE": 1,
            "GRAPH_ROOT_NODE": 1,
        },
        anchor="cn:en:synthesize (v)",
        literal={"id": "Ev1", "predicate": "synthesize", "chunk_id": chunk_id},
    )
    ev2 = QuantaNode(
        vector={
            "TYPE_EVENT": 1,
            "WN_ACT_ACTION": 1,
            "NSM_SEE": 1,
            "EPIST_DIRECT_OBSERVATION": 1,
            "LJB_PU_PAST_TENSE": 1,
        },
        anchor="cn:en:observe (v)",
        literal={"id": "Ev2", "predicate": "observe", "chunk_id": chunk_id},
    )
    ev3 = QuantaNode(
        vector={
            "TYPE_EVENT": 1,
            "WN_ACT_ACTION": 1,
            "NSM_THINK": 1,
            "NSM_TRUE": 1,
            "SOLVER_PROOF_VALIDATED": 1,
            "LJB_PU_PAST_TENSE": 1,
        },
        anchor="cn:en:verify (v)",
        literal={"id": "Ev3", "predicate": "verify", "chunk_id": chunk_id},
    )

    ev1_cid = graph.add_node(ev1, set_as_root=True)
    ev2_cid = graph.add_node(ev2)
    ev3_cid = graph.add_node(ev3)

    # Valency edges from events to external entities
    graph.add_edge(ev1_cid, "VAL_X1_AGENT", e1_cid)
    graph.add_edge(ev1_cid, "VAL_X2_PATIENT", e2_cid)
    graph.add_edge(ev2_cid, "VAL_X1_AGENT", e1_cid)
    graph.add_edge(ev2_cid, "VAL_X2_PATIENT", e2_cid)
    graph.add_edge(ev3_cid, "VAL_X1_AGENT", e1_cid)

    # Temporal sequencing between events
    graph.add_edge(ev1_cid, "TEMP_ALLEN_MEETS", ev2_cid)
    graph.add_edge(ev2_cid, "TEMP_ALLEN_MEETS", ev3_cid)

    return graph


def test_fold_discourse_episode_collapses_events_and_preserves_entities():
    """5.1: Verify fold_discourse_episode produces a 32-byte Merkle fold node with aggregate vector

    and preserves external entity references while collapsing internal events.
    """
    graph = _build_test_episode_graph("chunk_0001")
    initial_node_count = len(graph)
    assert initial_node_count == 5  # 2 entities + 3 events

    e1_node = [n for n in graph.nodes.values() if n.literal.get("id") == "E1"][0]
    e2_node = [n for n in graph.nodes.values() if n.literal.get("id") == "E2"][0]
    e1_cid = e1_node.cid
    e2_cid = e2_node.cid

    storage: dict = {}
    fold_node = fold_discourse_episode(graph, chunk_id="chunk_0001", storage=storage, keep_entities=True)

    # 1. 32-byte Merkle CID node verification
    assert isinstance(fold_node, QuantaNode)
    assert len(fold_node.cid) == 64  # 64-char hex = 256 bits = 32 bytes
    assert len(fold_node.node_cid_bytes) == 32
    assert fold_node.anchor is not None
    assert fold_node.anchor.startswith("merkle:")
    sub_merkle = fold_node.anchor.split("merkle:")[1]
    assert sub_merkle in storage

    # 2. Aggregate quaternary vector
    assert fold_node.get_slot("GRAPH_MERKLE_FOLD_POINT") == StructuralValue.ACTIVE_MERKLE
    assert fold_node.get_slot("GRAPH_EXT_REFERENCE") == StructuralValue.ACTIVE_MERKLE
    assert fold_node.get_slot("TYPE_PROPOSITION") == 1
    # Aggregate slots from events (synthesize + observe + verify)
    assert fold_node.get_slot("NSM_DO") == 1
    assert fold_node.get_slot("NSM_SEE") == 1
    assert fold_node.get_slot("NSM_THINK") == 1
    assert fold_node.get_slot("NSM_TRUE") == 1
    assert fold_node.get_slot("SOLVER_PROOF_VALIDATED") == 1

    # 3. Active graph compaction: events collapsed, entities preserved
    # Graph now contains only: fold_node + E1 + E2 (3 nodes down from 5)
    assert len(graph) == 3
    assert graph.get_node(e1_cid) is not None
    assert graph.get_node(e2_cid) is not None
    assert graph.get_node(fold_node.cid) is not None
    assert graph.root_cid == fold_node.cid

    # 4. External entity references attached to fold_node
    assert "VAL_X1_AGENT" in fold_node.edges
    assert e1_cid in fold_node.edges["VAL_X1_AGENT"]
    assert "VAL_X2_PATIENT" in fold_node.edges
    assert e2_cid in fold_node.edges["VAL_X2_PATIENT"]


def test_dynamic_unfold_discourse_episode_roundtrip():
    """5.3: Verify dynamic unfolding restores full episode sub-graph and verifies BLAKE3 integrity."""
    graph = _build_test_episode_graph("chunk_0001")
    storage: dict = {}
    fold_node = fold_discourse_episode(graph, chunk_id="chunk_0001", storage=storage, keep_entities=True)

    sub_merkle = fold_node.anchor.split("merkle:")[1]

    # Dynamic unfolding into isolated QuantaGraph
    unfolded_graph = unfold_subgraph(sub_merkle, storage)
    assert isinstance(unfolded_graph, QuantaGraph)
    assert len(unfolded_graph) == 3  # Exactly the 3 internal events
    assert unfolded_graph.compute_merkle_root() == sub_merkle

    # Dynamic unfolding with alias lookup
    unfolded_by_alias = unfold_subgraph("chunk_0001", storage)
    assert isinstance(unfolded_by_alias, QuantaGraph)
    assert unfolded_by_alias.compute_merkle_root() == sub_merkle

    # Dynamic in-place unfolding into target_graph
    restored_root = unfold_subgraph(fold_node.cid, storage, target_graph=graph)
    assert restored_root is not None
    assert len(graph) == 5  # Restored to 5 nodes (2 entities + 3 events)


def test_dynamic_unfold_tamper_rejection():
    """5.3: Verify that tampering with storage payload raises ValueError on unfold."""
    graph = _build_test_episode_graph("chunk_0001")
    storage: dict = {}
    fold_node = fold_discourse_episode(graph, chunk_id="chunk_0001", storage=storage)
    sub_merkle = fold_node.anchor.split("merkle:")[1]

    # Corrupt storage payload by modifying an anchor
    corrupted_storage = copy.deepcopy(storage)
    payload = corrupted_storage[sub_merkle]
    first_node_cid = next(iter(payload["nodes"].keys()))
    payload["nodes"][first_node_cid]["anchor"] = "cn:en:tampered_predicate (v)"

    with pytest.raises(ValueError, match="Merkle CID mismatch during unfold"):
        unfold_subgraph(sub_merkle, corrupted_storage)


def test_hierarchical_folding_chunk_to_chapter_to_book():
    """5.2: Verify hierarchical folding: combining chunk fold nodes into Chapter Merkle roots

    and Book Merkle roots.
    """
    storage: dict = {}

    # Build 4 chunks across 2 chapters
    # Chapter 1: chunk_00, chunk_01
    g0 = _build_test_episode_graph("chunk_00")
    g1 = _build_test_episode_graph("chunk_01")
    c0 = fold_discourse_episode(g0, "chunk_00", storage=storage)
    c1 = fold_discourse_episode(g1, "chunk_01", storage=storage)

    # Chapter 2: chunk_02, chunk_03
    g2 = _build_test_episode_graph("chunk_02")
    g3 = _build_test_episode_graph("chunk_03")
    c2 = fold_discourse_episode(g2, "chunk_02", storage=storage)
    c3 = fold_discourse_episode(g3, "chunk_03", storage=storage)

    # Fold into Chapter Merkle roots
    ch1_node = fold_chapter([c0, c1], chapter_id="ch_01", chapter_title="Chapter 1", storage=storage)
    ch2_node = fold_chapter([c2, c3], chapter_id="ch_02", chapter_title="Chapter 2", storage=storage)

    assert len(ch1_node.node_cid_bytes) == 32
    assert len(ch2_node.node_cid_bytes) == 32
    assert "GRAPH_MERKLE_FOLD_POINT" in ch1_node.edges
    assert c0.cid in ch1_node.edges["GRAPH_MERKLE_FOLD_POINT"]
    assert c1.cid in ch1_node.edges["GRAPH_MERKLE_FOLD_POINT"]

    # Fold into Book Merkle root
    book_node = fold_book([ch1_node, ch2_node], book_id="book_alpha", book_title="Alpha Book", storage=storage)

    assert len(book_node.node_cid_bytes) == 32
    assert "GRAPH_MERKLE_FOLD_POINT" in book_node.edges
    assert ch1_node.cid in book_node.edges["GRAPH_MERKLE_FOLD_POINT"]
    assert ch2_node.cid in book_node.edges["GRAPH_MERKLE_FOLD_POINT"]

    book_merkle_cid = book_node.anchor.split("merkle:")[1]
    assert book_merkle_cid in storage


def test_hierarchical_merkle_book_coordinator():
    """5.2 & 5.3: Verify HierarchicalMerkleBook coordinator class managing chunks, chapters, and book."""
    storage: dict = {}
    book = HierarchicalMerkleBook(book_id="book_01", title="Dr. Vance Chronicles", storage=storage)

    # Ingest 3 chunks into Chapter 1 and 2 chunks into Chapter 2
    for i in range(3):
        g = _build_test_episode_graph(f"chunk_ch1_{i}")
        book.add_chunk_episode(f"chunk_ch1_{i}", g, chapter_id="ch1")

    for i in range(2):
        g = _build_test_episode_graph(f"chunk_ch2_{i}")
        book.add_chunk_episode(f"chunk_ch2_{i}", g, chapter_id="ch2")

    # Fold Chapters and Book
    book.fold_chapter("ch1")
    book.fold_chapter("ch2")
    book_root_node = book.fold_book()

    assert book_root_node is not None
    assert book.book_root_cid == book_root_node.cid
    assert len(book.book_root_cid) == 64
    assert book.book_merkle_cid is not None

    # Unfold chapter and chunks
    unfolded_ch1 = book.unfold_chapter("ch1")
    assert isinstance(unfolded_ch1, QuantaGraph)
    assert len(unfolded_ch1) == 3  # 3 chunk fold nodes

    # Verify whole-book integrity
    is_valid, errors = book.verify_integrity()
    assert is_valid, f"Book integrity verification failed: {errors}"


def test_ten_chunk_narrative_tamper_detection():
    """5.4: Master Phase 5 Test: Fold a 10-chunk narrative into a single Book CID;

    verify any tamper in Chunk 3 strictly invalidates the Book Merkle root.
    """
    def build_narrative(tamper_chunk_3: bool = False):
        storage: dict = {}
        book = HierarchicalMerkleBook(book_id="novel_01", title="The Long Horizon", storage=storage)

        # 10 chunks partitioned across 2 chapters:
        # Chapter 1: chunks 0..4
        # Chapter 2: chunks 5..9
        for i in range(10):
            ch_id = "chapter_1" if i < 5 else "chapter_2"
            chunk_id = f"chunk_{i:02d}"
            graph = _build_test_episode_graph(chunk_id)

            if tamper_chunk_3 and i == 3:
                # Tamper with Chunk 3: modify an event predicate / slot
                ev_nodes = [n for n in graph.nodes.values() if n.get_slot("TYPE_EVENT") == 1]
                ev_nodes[0].set_slot("NSM_MOVE", 1)  # Mutate vector
                ev_nodes[0].anchor = "cn:en:tampered_action (v)"
                ev_nodes[0].invalidate_cache()

            book.add_chunk_episode(chunk_id, graph, chapter_id=ch_id)

        book.fold_chapter("chapter_1")
        book.fold_chapter("chapter_2")
        book.fold_book()
        return book, storage

    # 1. Build original untampered 10-chunk book
    book_orig, storage_orig = build_narrative(tamper_chunk_3=False)
    original_book_root_cid = book_orig.book_root_cid
    original_book_merkle_cid = book_orig.book_merkle_cid

    assert original_book_root_cid is not None
    assert len(original_book_root_cid) == 64
    assert original_book_merkle_cid is not None

    # Verify untampered book integrity
    is_valid, errors = book_orig.verify_integrity()
    assert is_valid
    assert len(errors) == 0

    # 2. Build narrative with Chunk 3 tampered
    book_tampered, storage_tampered = build_narrative(tamper_chunk_3=True)
    tampered_book_root_cid = book_tampered.book_root_cid
    tampered_book_merkle_cid = book_tampered.book_merkle_cid

    # 3. Assert strict invalidation of the Book Merkle root
    # Changing Chunk 3 alters Chunk 3's Merkle CID -> Chapter 1 CID -> Book Root CID!
    assert original_book_root_cid != tampered_book_root_cid, (
        f"Tampering Chunk 3 MUST change Book Root CID! "
        f"Original: {original_book_root_cid}, Tampered: {tampered_book_root_cid}"
    )
    assert original_book_merkle_cid != tampered_book_merkle_cid, (
        f"Tampering Chunk 3 MUST change Book Merkle CID! "
        f"Original: {original_book_merkle_cid}, Tampered: {tampered_book_merkle_cid}"
    )

    # Chapter 1 CID must be different, but Chapter 2 CID must be identical!
    ch1_orig = book_orig.chapter_nodes["chapter_1"].cid
    ch1_tamp = book_tampered.chapter_nodes["chapter_1"].cid
    assert ch1_orig != ch1_tamp, "Chapter 1 CID must change due to Chunk 3 tamper"

    ch2_orig = book_orig.chapter_nodes["chapter_2"].cid
    ch2_tamp = book_tampered.chapter_nodes["chapter_2"].cid
    assert ch2_orig == ch2_tamp, "Chapter 2 CID must remain completely unaffected"

    # 4. Storage tamper detection test:
    # If storage payload for Chunk 3 in the original book is altered, unfolding Chunk 3 fails
    c3_fold_node = book_orig.chapters["chapter_1"][3]
    c3_merkle = c3_fold_node.anchor.split("merkle:")[1]

    # Tamper payload in storage
    tampered_payload_storage = copy.deepcopy(storage_orig)
    stored_c3_dict = tampered_payload_storage[c3_merkle]
    first_node_key = next(iter(stored_c3_dict["nodes"].keys()))
    stored_c3_dict["nodes"][first_node_key]["anchor"] = "cn:en:tampered_predicate (v)"

    with pytest.raises(ValueError, match="Merkle CID mismatch during unfold"):
        unfold_subgraph(c3_merkle, tampered_payload_storage)


def test_pagetable_storage_sqlite_persistence():
    """Verify PageTableStorage supports SQLite disk persistence and tamper detection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_pagetable.db"
        storage = PageTableStorage(db_path=db_path)

        graph = _build_test_episode_graph("chunk_persisted")
        fold_node = fold_discourse_episode(graph, chunk_id="chunk_persisted", storage=storage)
        merkle_cid = fold_node.anchor.split("merkle:")[1]

        # Verify entry exists in storage
        assert merkle_cid in storage
        assert f"chunk:chunk_persisted" in storage

        # Retrieve and verify sub-graph
        retrieved_graph = storage.retrieve_subgraph(merkle_cid)
        assert isinstance(retrieved_graph, QuantaGraph)
        assert retrieved_graph.compute_merkle_root() == merkle_cid

        # Tamper directly in SQLite table
        with storage._sqlite_conn:
            storage._sqlite_conn.execute(
                "UPDATE subgraphs SET payload = replace(payload, 'synthesize', 'corrupted_verb') WHERE cid = ?",
                (merkle_cid,),
            )

        # Retrieval must fail with ValueError
        with pytest.raises(ValueError, match="Merkle CID mismatch during unfold"):
            storage.retrieve_subgraph(merkle_cid)

        storage.close()
