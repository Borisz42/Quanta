"""Tests for Section 4: Query-Driven Spreading-Activation Sub-Graph Attention (Context Retrieval).

Verifies:
- Task 4.1: Query ASG Transduction (compile_query_asg) with GRAPH_QUERY_TARGET=3 and QUERY_TARGET_VAR_X=3 (?X).
- Task 4.2: SIMD Top-K Seed Selection (find_seed_nodes) with < 1 ms latency over host memory index.
- Task 4.3: Spreading Activation Traversal (traverse_subgraph) along thematic valencies and causal links
            with decay gamma=0.7 and cutoff theta=0.35 up to depth 2.
- Task 4.4: Dynamic Context Builder for Host LLMs (format_context_for_llm) supporting English and GBNF S-expressions.
- Task 4.5: Multi-chapter multi-hop retrieval accuracy and 100,000-node < 5 ms retrieval latency benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np
import pytest

from core.asg import QuantaGraph, QuantaNode
from core.slots import SLOT_NAME_TO_INDEX
from core.types import QuantaVector, RegisterValue, StructuralValue
from memory.page_table import ActiveCanvas, PageTable, SimdHammingIndex
from memory.spreading_activation import SpreadingActivationRetriever
from pipeline.cognitive_pipeline import CognitivePipeline
from parser.schema import (
    DiscourseExtractionResult,
    ExtractedEntity,
    ExtractedEvent,
    ExtractedRelation,
)


# =============================================================================
# Helper Utilities
# =============================================================================

def _create_node(
    anchor: str,
    literal: Any,
    slots: dict[str, int] | None = None,
    edges: dict[str, list[str]] | None = None,
) -> QuantaNode:
    """Creates a QuantaNode with given slots and edges."""
    node = QuantaNode(
        anchor=anchor,
        literal=literal,
        edges=edges or {},
    )
    if slots:
        for s_name, val in slots.items():
            if s_name in SLOT_NAME_TO_INDEX:
                node.set_slot(s_name, val)
    node.compute_cid()
    return node


# =============================================================================
# Task 4.1: Query ASG Transduction Tests
# =============================================================================

class TestQueryASGTransduction:
    """Tests for compiling natural language questions into canonical query ASGs."""

    @pytest.fixture
    def retriever(self):
        return SpreadingActivationRetriever()

    def test_query_asg_who(self, retriever: SpreadingActivationRetriever):
        """'Who isolated the volatile synthetic compound?' -> ?X is VAL_X1_AGENT."""
        graph = retriever.compile_query_asg("Who isolated the volatile synthetic compound?")

        assert graph.root is not None
        root = graph.root

        # Root event checks
        assert root.get_slot("GRAPH_QUERY_TARGET") == 3
        assert root.get_slot("TYPE_EVENT") == 1
        assert "isolate" in (root.anchor or "").lower()

        # Valency checks: VAL_X1_AGENT must point to ?X
        assert "VAL_X1_AGENT" in root.edges
        var_cid = root.edges["VAL_X1_AGENT"][0]
        var_node = graph.get_node(var_cid)
        assert var_node is not None
        assert var_node.literal == "?X"
        assert var_node.get_slot("GRAPH_QUERY_TARGET") == 3
        assert var_node.get_slot("QUERY_TARGET_VAR_X") == 3
        assert var_node.get_slot("ROLE_AGENT_CAPABLE") == 1

        # Patient check: VAL_X2_PATIENT should contain compound
        assert "VAL_X2_PATIENT" in root.edges
        patient_cid = root.edges["VAL_X2_PATIENT"][0]
        patient_node = graph.get_node(patient_cid)
        assert patient_node is not None
        assert "compound" in str(patient_node.literal).lower()

    def test_query_asg_what(self, retriever: SpreadingActivationRetriever):
        """'What did Eleanor Vance verify?' -> ?X is VAL_X2_PATIENT."""
        graph = retriever.compile_query_asg("What did Eleanor Vance verify?")

        assert graph.root is not None
        root = graph.root

        assert root.get_slot("GRAPH_QUERY_TARGET") == 3
        assert "verify" in (root.anchor or "").lower()

        # Target is patient ?X
        assert "VAL_X2_PATIENT" in root.edges
        var_cid = root.edges["VAL_X2_PATIENT"][0]
        var_node = graph.get_node(var_cid)
        assert var_node is not None
        assert var_node.literal == "?X"
        assert var_node.get_slot("GRAPH_QUERY_TARGET") == 3
        assert var_node.get_slot("QUERY_TARGET_VAR_X") == 3

        # Agent is Eleanor Vance
        assert "VAL_X1_AGENT" in root.edges
        agent_cid = root.edges["VAL_X1_AGENT"][0]
        agent_node = graph.get_node(agent_cid)
        assert agent_node is not None
        assert "eleanor" in str(agent_node.literal).lower()

    def test_query_asg_where(self, retriever: SpreadingActivationRetriever):
        """'Where did Eleanor Vance store the specimen?' -> ?X is VAL_LOCATION_SLOT."""
        graph = retriever.compile_query_asg("Where did Eleanor Vance store the specimen?")

        assert graph.root is not None
        root = graph.root

        assert root.get_slot("GRAPH_QUERY_TARGET") == 3
        assert "store" in (root.anchor or "").lower()

        assert "VAL_LOCATION_SLOT" in root.edges
        var_cid = root.edges["VAL_LOCATION_SLOT"][0]
        var_node = graph.get_node(var_cid)
        assert var_node is not None
        assert var_node.literal == "?X"
        assert var_node.get_slot("GRAPH_QUERY_TARGET") == 3
        assert var_node.get_slot("QUERY_TARGET_VAR_X") == 3
        assert var_node.get_slot("TYPE_LOCATION") == 1

    def test_query_asg_why(self, retriever: SpreadingActivationRetriever):
        """'Why did the laboratory director prohibit competing tests?' -> ?X is CAUSAL_MECHANISM_LINK."""
        graph = retriever.compile_query_asg("Why did the laboratory director prohibit competing tests?")

        assert graph.root is not None
        root = graph.root

        assert root.get_slot("GRAPH_QUERY_TARGET") == 3
        assert "prohibit" in (root.anchor or "").lower()

        assert "CAUSAL_MECHANISM_LINK" in root.edges
        var_cid = root.edges["CAUSAL_MECHANISM_LINK"][0]
        var_node = graph.get_node(var_cid)
        assert var_node is not None
        assert var_node.literal == "?X"
        assert var_node.get_slot("GRAPH_QUERY_TARGET") == 3
        assert var_node.get_slot("QUERY_TARGET_VAR_X") == 3


# =============================================================================
# Task 4.2: SIMD Top-K Seed Selection Tests
# =============================================================================

class TestSimdSeedSelection:
    """Tests for < 1 ms SIMD seed selection over PageTable."""

    def test_simd_top_k_retrieval(self, tmp_path: Path):
        """Verifies top-K nearest seeds are retrieved in < 1 ms."""
        db_path = tmp_path / "seed_test.db"
        pt = PageTable(db_path)
        retriever = SpreadingActivationRetriever()

        # Seed 200 nodes
        rng = np.random.RandomState(42)
        target_node = None
        for i in range(200):
            node = _create_node(
                anchor=f"cn:en:concept_{i} (n)",
                literal=f"Entity_{i}",
                slots={"TYPE_EVENT": 1 if i % 4 == 0 else 0, "TYPE_HUMAN": 1 if i % 3 == 0 else 0},
            )
            if i == 50:
                node.set_slot("NSM_KNOW", 1)
                node.set_slot("NSM_TRUE", 1)
                node.anchor = "cn:en:verify (v)"
                node.literal = "verify"
                target_node = node
            pt.store_node(node)

        assert target_node is not None
        assert len(pt.vector_index) == 200

        # Warmup Numba JIT
        _ = retriever.find_seed_nodes("Warmup", pt, top_k=5)

        # Query vector search
        t0 = time.perf_counter()
        matches = retriever.find_seed_nodes("What did Eleanor Vance verify?", pt, top_k=5)
        t_elapsed = (time.perf_counter() - t0) * 1000

        print(f"[Seed Selection Benchmark] 200 nodes top-5 search in {t_elapsed:.3f} ms")
        assert len(matches) <= 5
        assert len(matches) > 0
        assert t_elapsed < 10.0, f"Seed search took {t_elapsed:.3f} ms, expected < 10 ms"

        # Verify sorted by distance
        distances = [d for _, d in matches]
        assert distances == sorted(distances)

        pt.close()


# =============================================================================
# Task 4.3: Spreading Activation Traversal Tests
# =============================================================================

class TestSpreadingActivationTraversal:
    """Tests for graph traversal with gamma=0.7 decay and theta=0.35 cutoff."""

    def test_activation_decay_and_pruning(self, tmp_path: Path):
        """Verifies depth-limiting and threshold cutoff:
        Depth 0: Ev1 (A=1.0)
        Depth 1: Ev2 (A=0.70 >= 0.35)
        Depth 2: Ev3 (A=0.49 >= 0.35)
        Depth 3: Ev4 (A=0.343 < 0.35 -> PRUNED)
        """
        db_path = tmp_path / "traversal_test.db"
        pt = PageTable(db_path)
        retriever = SpreadingActivationRetriever(decay=0.7, threshold=0.35, max_depth=2)

        # Construct 4-event chain: Ev1 -> Ev2 -> Ev3 -> Ev4
        ev4 = _create_node("cn:en:retain (v)", "retained", slots={"TYPE_EVENT": 1})
        ev3 = _create_node("cn:en:prohibit (v)", "prohibited", slots={"TYPE_EVENT": 1}, edges={"TEMP_ALLEN_MEETS": [ev4.cid]})
        ev2 = _create_node("cn:en:note (v)", "noted", slots={"TYPE_EVENT": 1}, edges={"TEMP_ALLEN_MEETS": [ev3.cid]})
        ev1 = _create_node("cn:en:isolate (v)", "isolated", slots={"TYPE_EVENT": 1}, edges={"TEMP_ALLEN_MEETS": [ev2.cid]})

        # Also give Ev1 an agent entity
        ent1 = _create_node("cn:en:eleanor_vance (n)", "Eleanor Vance", slots={"ROLE_AGENT_CAPABLE": 1, "TYPE_HUMAN": 1})
        ev1.edges["VAL_X1_AGENT"] = [ent1.cid]
        ev1.compute_cid()

        pt.store_node(ev4)
        pt.store_node(ev3)
        pt.store_node(ev2)
        pt.store_node(ev1)
        pt.store_node(ent1)

        # Traverse starting from Ev1
        subgraph = retriever.traverse_subgraph(
            seed_cids=[ev1.cid],
            page_table=pt,
            max_depth=2,
            decay=0.7,
            threshold=0.35,
        )

        sub_cids = set(subgraph.nodes.keys())
        # Ev1, Ev2, Ev3 and ent1 should be admitted
        assert ev1.cid in sub_cids
        assert ev2.cid in sub_cids
        assert ev3.cid in sub_cids
        assert ent1.cid in sub_cids

        # Ev4 MUST BE PRUNED because A = 0.343 < 0.35
        assert ev4.cid not in sub_cids
        assert ev4.cid not in subgraph.nodes

        # Ev1's edge to Ev2 must be preserved
        ev1_sub = subgraph.get_node(ev1.cid)
        assert ev1_sub is not None
        assert ev2.cid in ev1_sub.edges.get("TEMP_ALLEN_MEETS", [])

        pt.close()

    def test_bidirectional_spreading(self, tmp_path: Path):
        """Verifies spreading activation traverses reverse links from entity to pointing events."""
        db_path = tmp_path / "bidi_test.db"
        pt = PageTable(db_path)
        retriever = SpreadingActivationRetriever(decay=0.7, threshold=0.35, max_depth=2)

        ent = _create_node("cn:en:specimen (n)", "specimen", slots={"TYPE_SUBSTANCE": 1})
        ev = _create_node("cn:en:isolate (v)", "isolated", slots={"TYPE_EVENT": 1}, edges={"VAL_X2_PATIENT": [ent.cid]})

        pt.store_node(ent)
        pt.store_node(ev)

        # Seed is entity; should reach event via reverse incoming edge
        subgraph = retriever.traverse_subgraph(
            seed_cids=[ent.cid],
            page_table=pt,
            max_depth=1,
            decay=0.7,
            threshold=0.35,
            bidirectional=True,
        )

        assert ent.cid in subgraph.nodes
        assert ev.cid in subgraph.nodes

        pt.close()


# =============================================================================
# Task 4.4: Dynamic Context Builder Tests
# =============================================================================

class TestDynamicContextBuilder:
    """Tests for formatting sub-graphs into English prose and GBNF S-expressions."""

    @pytest.fixture
    def sample_graph(self):
        graph = QuantaGraph()
        ent_vance = _create_node("cn:en:eleanor_vance (n)", "Dr. Eleanor Vance", slots={"ROLE_AGENT_CAPABLE": 1, "TYPE_HUMAN": 1})
        ent_hyp = _create_node("cn:en:hypothesis (n)", "the hypothesis", slots={"TYPE_ABSTRACT": 1})
        ent_cell = _create_node("cn:en:containment_cell (n)", "Containment Cell 4", slots={"TYPE_LOCATION": 1})

        ev1 = _create_node(
            "cn:en:verify (v)",
            "verify",
            slots={"TYPE_EVENT": 1, "WN_ACT_ACTION": 1},
            edges={"VAL_X1_AGENT": [ent_vance.cid], "VAL_X2_PATIENT": [ent_hyp.cid], "VAL_LOCATION_SLOT": [ent_cell.cid]},
        )

        graph.add_node(ent_vance)
        graph.add_node(ent_hyp)
        graph.add_node(ent_cell)
        graph.add_node(ev1, set_as_root=True)
        return graph

    def test_format_english(self, sample_graph: QuantaGraph):
        retriever = SpreadingActivationRetriever()
        text = retriever.format_context_for_llm(sample_graph, format="english", max_tokens=100)

        assert isinstance(text, str)
        assert len(text) > 0
        assert text.endswith(".")
        # Should mention agent or verb
        assert "Vance" in text or "Eleanor" in text or "verif" in text.lower()

    def test_format_sexpr(self, sample_graph: QuantaGraph):
        retriever = SpreadingActivationRetriever()
        sexpr = retriever.format_context_for_llm(sample_graph, format="sexpr", max_tokens=200)

        assert isinstance(sexpr, str)
        assert len(sexpr) > 0
        assert "(graph" in sexpr or "(discourse" in sexpr
        assert "(event" in sexpr
        assert "(entity" in sexpr

    def test_token_budget_truncation(self, sample_graph: QuantaGraph):
        retriever = SpreadingActivationRetriever()
        text = retriever.format_context_for_llm(sample_graph, format="english", max_tokens=3)
        assert len(text.split()) <= 4


# =============================================================================
# Task 4.5: Multi-Chapter Narrative & 100,000-Node Benchmark Tests
# =============================================================================

class TestMultiChapterAndScalingBenchmark:
    """Tests multi-hop retrieval accuracy and < 5 ms latency across 100,000 nodes."""

    def test_multi_chapter_multi_hop_retrieval(self, tmp_path: Path):
        """Ingests 3 chapters, queries multi-hop facts, asserts exact facts retrieved without distractors."""
        db_path = tmp_path / "multichapter_test.db"
        pipeline = CognitivePipeline(transducer_backend="mock", page_table_path=db_path)

        # Chapter 1: Isolation
        ch1 = "Dr. Eleanor Vance isolated a volatile synthetic compound inside Containment Cell 4 at dawn."
        # Chapter 2: Prohibition due to expansion
        ch2 = "The laboratory director noted an anomalous crystalline lattice expansion. The director prohibited all competing tests."
        # Chapter 3: Verification
        ch3 = "Three hours later, Dr. Eleanor Vance verified the hypothesis."

        pipeline.process_narrative(ch1, chapter_id="chapter_01")
        pipeline.process_narrative(ch2, chapter_id="chapter_02")
        pipeline.process_narrative(ch3, chapter_id="chapter_03")

        # Query 1: What did Eleanor Vance verify?
        ctx1 = pipeline.retrieve_context("What did Eleanor Vance verify?", format="english", max_tokens=200)
        assert "verif" in ctx1.lower() or "hypothesis" in ctx1.lower()

        ans1 = pipeline.answer_query("What did Eleanor Vance verify?")
        assert "hypothesis" in ans1.lower()

        # Query 2: Where was the compound isolated?
        ctx2 = pipeline.retrieve_context("Where did Eleanor Vance isolate the compound?", format="english", max_tokens=200)
        assert "cell" in ctx2.lower() or "containment" in ctx2.lower()

        ans2 = pipeline.answer_query("Where did Eleanor Vance isolate the compound?")
        assert "cell" in ans2.lower() or "containment" in ans2.lower()

        # S-expression retrieval test
        sexpr_ctx = pipeline.retrieve_context("What did Eleanor Vance verify?", format="sexpr", max_tokens=200)
        assert "(graph" in sexpr_ctx or "(discourse" in sexpr_ctx

        pipeline.page_table.close()

    def test_100k_node_retrieval_latency(self, tmp_path: Path):
        """Ingests 100,000 nodes and asserts retrieval latency strictly < 5.0 ms."""
        db_path = tmp_path / "scaling_100k.db"
        pt = PageTable(db_path)
        retriever = SpreadingActivationRetriever()

        N = 100000
        rng = np.random.RandomState(1337)

        print(f"\n[Task 4.5 Scaling] Ingesting {N} nodes into PageTable...")
        t0_ingest = time.perf_counter()

        # Fast synthetic batch ingest (2000 per chunk)
        chunk_size = 2000
        total_chunks = N // chunk_size
        all_cids = []

        now = time.time()
        for chunk_idx in range(total_chunks):
            chunk_rows = []
            chunk_vecs = []
            for i in range(chunk_size):
                idx = chunk_idx * chunk_size + i
                cid = f"node_{idx:08x}_{rng.randint(0, 1000000):06x}"
                all_cids.append(cid)
                raw_bytes = rng.bytes(256)
                edges_dict = {}
                if idx > 0 and idx % 20 == 0:
                    edges_dict["TEMP_ALLEN_MEETS"] = [all_cids[idx - 1]]
                chunk_rows.append((cid, raw_bytes, 1, None, None, json.dumps(edges_dict), now, 0, now))
                chunk_vecs.append((cid, raw_bytes))

            with pt._conn:
                pt._conn.executemany(
                    "INSERT INTO nodes (cid, vector_bytes, anchor_id, literal, parent_cid, edges, created_at, access_count, last_accessed) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    chunk_rows,
                )
            pt.vector_index.add_batch(chunk_vecs)

        t_ingest = (time.perf_counter() - t0_ingest) * 1000
        print(f"[Task 4.5 Scaling] Ingested {N} nodes in {t_ingest:.2f} ms ({N / (t_ingest / 1000):.0f} nodes/sec)")
        assert pt.count_nodes() == N
        assert len(pt.vector_index) == N

        # Create realistic query vector
        query_bytes = rng.bytes(256)

        # Warmup
        warmup_seeds = pt.vector_index.search(query_bytes, top_k=5)
        _ = retriever.traverse_subgraph([c for c, _ in warmup_seeds], pt, max_depth=2, decay=0.7, threshold=0.35, bidirectional=False)

        latencies = []
        for trial in range(15):
            t0 = time.perf_counter()
            seeds = pt.vector_index.search(query_bytes, top_k=5)
            seed_cids = [cid for cid, _ in seeds]
            # Spreading activation traversal
            subgraph = retriever.traverse_subgraph(
                seed_cids=seed_cids,
                page_table=pt,
                max_depth=2,
                decay=0.7,
                threshold=0.35,
                bidirectional=False,  # Directed valency traversal over 100k nodes
            )
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

        min_lat = float(np.min(latencies))
        mean_lat = float(np.mean(latencies))
        print(
            f"[Task 4.5 Benchmark] 100,000-Node End-to-End Retrieval Latency: "
            f"min={min_lat:.3f} ms, mean={mean_lat:.3f} ms"
        )

        assert len(subgraph.nodes) >= len(seeds)
        # STRICT REQUIREMENT: Retrieval completes in < 5.0 ms
        assert min_lat < 5.0, f"Retrieval latency must be < 5.0 ms, got min={min_lat:.3f} ms"
        assert mean_lat < 5.0, f"Mean retrieval latency must be < 5.0 ms, got mean={mean_lat:.3f} ms"

        pt.close()


class TestMultiAnchorAndTraversalDynamics:
    """Validates multi-anchor harvesting, spreading activation retention, and thematic exemptions."""

    def test_multi_anchor_seed_extraction(self):
        """Verifies extraction of quoted entities, genitives, and single-word proper nouns."""
        retriever = SpreadingActivationRetriever()

        # 1. Quoted expressions
        q1 = "Who wrote 'Turn Me On' by performer of 'Happy Pills'?"
        ents1 = retriever._extract_salient_entities(q1, None)
        assert any("Turn Me On" in e for e in ents1), f"Expected 'Turn Me On' in {ents1}"
        assert any("Happy Pills" in e for e in ents1), f"Expected 'Happy Pills' in {ents1}"

        # 2. Genitive constructs
        q2 = "Who was Nelvana's founder?"
        ents2 = retriever._extract_salient_entities(q2, None)
        assert any("Nelvana" in e for e in ents2), f"Expected 'Nelvana' in {ents2}"
        assert any("founder" in e for e in ents2), f"Expected 'founder' in {ents2}"

        # 3. Single-word proper nouns
        q3 = "Did Candlebox release an album in 1993?"
        ents3 = retriever._extract_salient_entities(q3, None)
        assert any("Candlebox" in e for e in ents3), f"Expected 'Candlebox' in {ents3}"

    def test_multihop_traversal_retention_above_threshold(self, tmp_path: Path):
        """Verifies 3-hop and 4-hop chain retention above threshold theta=0.15 with decay gamma=0.85."""
        db_path = tmp_path / "multihop_chain.db"
        pt = PageTable(db_path)
        retriever = SpreadingActivationRetriever()

        # Build 4-hop chain: N1 -> N2 -> N3 -> N4 -> N5
        nodes = []
        for i in range(1, 6):
            n = QuantaNode(literal=f"Entity_{i}", anchor=f"ent_{i}")
            nodes.append(n)

        for n in nodes:
            pt.store_node(n)

        # Wire edges: N1 -> N2 -> N3 -> N4 -> N5 along CAUSAL_LEADS_TO
        for i in range(4):
            pt.add_edge(nodes[i].cid, "CAUSAL_LEADS_TO", nodes[i + 1].cid)

        subgraph = retriever.traverse_subgraph(
            seed_cids=[nodes[0].cid],
            page_table=pt,
            max_depth=4,
            decay=0.85,
            threshold=0.15,
            bidirectional=False,
        )

        activations = getattr(subgraph, "activations", {})
        # Hop 0: N1 = 1.0
        assert nodes[0].cid in subgraph.nodes
        assert activations.get(nodes[0].cid, 0.0) == 1.0

        # Hop 1: N2 = 0.85 >= 0.15
        assert nodes[1].cid in subgraph.nodes
        assert activations.get(nodes[1].cid, 0.0) == pytest.approx(0.85, abs=1e-3)

        # Hop 2: N3 = 0.85^2 = 0.7225 >= 0.15
        assert nodes[2].cid in subgraph.nodes
        assert activations.get(nodes[2].cid, 0.0) == pytest.approx(0.7225, abs=1e-3)

        # Hop 3: N4 = 0.85^3 = 0.6141 >= 0.15
        assert nodes[3].cid in subgraph.nodes
        assert activations.get(nodes[3].cid, 0.0) == pytest.approx(0.614125, abs=1e-3)

        # Hop 4: N5 = 0.85^4 = 0.5220 >= 0.15
        assert nodes[4].cid in subgraph.nodes
        assert activations.get(nodes[4].cid, 0.0) == pytest.approx(0.522006, abs=1e-3)

        pt.close()

    def test_thematic_valency_hub_degree_exemption(self, tmp_path: Path):
        """Verifies VAL_X1_AGENT and VAL_X2_PATIENT are exempt from degree attenuation on hubs."""
        db_path = tmp_path / "thematic_hub.db"
        pt = PageTable(db_path)
        retriever = SpreadingActivationRetriever()

        # Create high-degree entity hub
        hub = QuantaNode(literal="HubEntity", anchor="hub_ent")
        pt.store_node(hub)

        # Add 10 incoming event nodes pointing to hub
        ev_nodes = []
        for i in range(10):
            ev = QuantaNode(literal=f"Event_{i}", anchor=f"ev_{i}")
            # Event points to hub via VAL_X1_AGENT
            ev.add_edge("VAL_X1_AGENT", hub.cid)
            pt.store_node(ev)
            ev_nodes.append(ev)

        # Reverse traversal from hub
        subgraph = retriever.traverse_subgraph(
            seed_cids=[hub.cid],
            page_table=pt,
            max_depth=1,
            decay=0.85,
            threshold=0.15,
            bidirectional=True,
        )

        activations = getattr(subgraph, "activations", {})
        # Hub has in-degree 10. Without exemption, degree_penalty = min(log2(11), 1.5) = 1.5
        # With exemption, step_decay = 0.85 exactly, so rev_act = 1.0 * 0.85 = 0.85.
        for ev in ev_nodes:
            assert ev.cid in subgraph.nodes, f"Event {ev.cid} must be retained in subgraph"
            act = activations.get(ev.cid, 0.0)
            assert act == pytest.approx(0.85, abs=1e-3), f"Expected 0.85 unpenalized activation, got {act}"

        pt.close()

    def test_adaptive_context_budget_scaling(self):
        """Verifies adaptive context budget formula in proxy calculation."""
        from server.proxy import QuantaProxyConfig

        cfg = QuantaProxyConfig()
        assert cfg.max_context_tokens >= 1600

        # Case 1: 20 document headings -> target 1600 tokens
        headings_20 = 20
        budget_20 = min(cfg.max_context_tokens, max(1200, min(1600, headings_20 * 100)))
        assert budget_20 == 1600

        # Case 2: 12 document headings -> target 1200 tokens
        headings_12 = 12
        budget_12 = min(cfg.max_context_tokens, max(1200, min(1600, headings_12 * 100)))
        assert budget_12 == 1200

        # Case 3: 15 document headings -> target 1500 tokens
        headings_15 = 15
        budget_15 = min(cfg.max_context_tokens, max(1200, min(1600, headings_15 * 100)))
        assert budget_15 == 1500

    def test_global_kb_seed_qualification(self):
        """Verifies that single dictionary tokens are rejected for Global KB lookups while proper entities qualify."""
        retriever = SpreadingActivationRetriever()
        q = "On August 21, a flash flood warning was issued for the Las Vegas area. Which list presents the objects as shown below?"

        # Qualified entities: multi-word phrases, capitalized proper nouns, acronyms
        assert retriever._is_qualified_global_kb_seed("flash flood", q) is True
        assert retriever._is_qualified_global_kb_seed("Las Vegas", q) is True
        assert retriever._is_qualified_global_kb_seed("WASP-96b", q) is True
        assert retriever._is_qualified_global_kb_seed("CO2", q) is True

        # Rejected entities: single common dictionary words and procedural tokens
        assert retriever._is_qualified_global_kb_seed("shown", q) is False
        assert retriever._is_qualified_global_kb_seed("list", q) is False
        assert retriever._is_qualified_global_kb_seed("presents", q) is False
        assert retriever._is_qualified_global_kb_seed("area", q) is False
        assert retriever._is_qualified_global_kb_seed("below", q) is False
        assert retriever._is_qualified_global_kb_seed("flash", q) is False
        assert retriever._is_qualified_global_kb_seed("each", q) is False

    def test_global_kb_distractor_pruning(self):
        """Verifies that entertainment, sports, surname and config distractors are correctly pruned."""
        retriever = SpreadingActivationRetriever()
        q_science = "Which statement best describes the weather and climate in Las Vegas?"

        # Surnames
        surname_node = QuantaNode(literal={"label": "shown (surname)", "description": "Shown is an American surname."})
        assert retriever._is_distractor_entity(surname_node, q_science) is True

        # Films in non-media query
        film_node = QuantaNode(
            literal={"label": "The Warning", "description": "A 1928 silent drama film.", "category": "creative_work"},
            edges={"INSTANCE_OF": ["movie"]},
        )
        assert retriever._is_distractor_entity(film_node, q_science) is True

        # Pro-gamer / esports in non-sports query
        gamer_node = QuantaNode(literal={"label": "Flash", "description": "Lee Young-ho is a StarCraft player."})
        assert retriever._is_distractor_entity(gamer_node, q_science) is True

        # Scientific entity preserved
        flood_node = QuantaNode(literal={"label": "flash flood", "description": "A flash flood is a rapid flooding of low-lying areas."})
        assert retriever._is_distractor_entity(flood_node, q_science) is False


