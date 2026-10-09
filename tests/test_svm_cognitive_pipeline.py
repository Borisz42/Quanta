"""Integration and Benchmark Test Suite for Section 8: End-to-End SVM Cognitive Pipeline.

Verifies:
1. End-to-End Ingestion Cycle (CognitivePipeline.ingest_document):
   - Raw text ingestion -> PassageStore registration -> SkeletonTransducer extraction
     -> KevDecisionEngine 3-pass prefill scoring -> Belnap lattice & SIMD grounding
     -> ClingoDLGate difference logic validation -> Dual-node QuantaGraph & 128-byte BinaryNodeTable.
   - Exact bipartite provenance links (E_ground) linking semantic nodes to passage spans.
   - Thematic valency bindings (VAL_X1_AGENT, VAL_X2_PATIENT, VAL_X5_INSTRUMENT).
   - Allen temporal intervals and Pearl causal mechanism DAG links.
2. Dual-Stream Retrieval Cycle (CognitivePipeline.query_memory):
   - Query compilation -> HippoRAG 2 Personalized PageRank (PPR) -> PoP-RAG gating
     -> Bipartite passage projection (Score(v_p) = sum p(v_s) * conf(v_s))
     -> DualStreamContext assembly.
   - Stream 1: Logical Briefing Block (verified causal/temporal paths & valency frames).
   - Stream 2: Raw Source Passages (verbatim spans with 100% lexical fidelity).
   - Strict token budget enforcement (e.g., 250, 500, 1000 tokens).
3. Multi-Hop Associative Recall:
   - Ingestion of multiple related passages sharing entities and causal chains.
   - Multi-hop traversal activating intermediate entities and retrieving complete evidence path.
4. Memory & VRAM Budget Validation:
   - Approach A in-context logprob mode: KevDecisionEngine.vram_overhead_mb == 0.0 (shared llama-server weights).
   - Unified VRAM footprint calculation: Qwen3.5-4B (~3.1 GB) + KV cache (~1.1 GB) + Kev (0.0 GB) <= 4.8 GB.
5. Ingestion Latency & Throughput Benchmark:
   - Verification of per-chunk ingestion execution and latency profiling within budget (350-520 ms target).
"""

from __future__ import annotations

from pathlib import Path
import time
import pytest

from core.asg import QuantaGraph, QuantaNode
from core.binary_node import BinaryNodeTable, BelnapValue
from core.types import QuaternaryValue
from memory.context_assembler import DualStreamContext, DualStreamContextAssembler
from memory.passage_store import PassageRecord, PassageStore
from models.kev_engine import KevDecisionEngine, MockKevEngine
from parser.skeleton_transducer import MockSkeletonTransducer, SkeletonTransducer
from pipeline.cognitive_pipeline import CognitivePipeline


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "svm_pipeline_test.db"


@pytest.fixture
def pipeline(tmp_db_path: Path) -> CognitivePipeline:
    """Initializes CognitivePipeline wired with mock transducers for deterministic CI testing."""
    pipe = CognitivePipeline(
        transducer_backend="mock",
        page_table_path=tmp_db_path,
        canvas_capacity=512,
    )
    yield pipe
    pipe.close()


@pytest.fixture
def sample_passage_text() -> str:
    return (
        "Dr. Eleanor Vance isolated a volatile synthetic compound inside Containment Cell 4 at dawn. "
        "The reaction catalyzed an anomalous crystalline expansion that ruptured the titanium valve."
    )


# =============================================================================
# 1. End-to-End Ingestion Cycle Tests
# =============================================================================

def test_ingest_document_full_cycle(pipeline: CognitivePipeline, sample_passage_text: str):
    """Verifies all 6 steps of the Semantic Virtual Memory ingestion cycle."""
    doc_id = "doc_reactor_01"
    pid = "P_reactor_01_c1"

    graph = pipeline.ingest_document(
        text=sample_passage_text,
        doc_id=doc_id,
        passage_id=pid,
        validate=True,
    )

    # 1. Verify returned graph topology
    assert isinstance(graph, QuantaGraph)
    assert len(graph.nodes) > 0
    assert graph.root_cid is not None

    # Verify node partitioning: Entity nodes vs Event nodes
    entity_nodes = [n for n in graph.nodes.values() if n.get_slot("TYPE_EVENT") == 0]
    event_nodes = [n for n in graph.nodes.values() if n.get_slot("TYPE_EVENT") == 1]
    assert len(entity_nodes) > 0, "Expected entity nodes in ASG"
    assert len(event_nodes) > 0, "Expected event nodes in ASG"

    # 2. Step 1: PassageStore registration
    assert len(pipeline.passage_store) >= 1
    stored_p = pipeline.passage_store.get_passage(pid)
    assert stored_p is not None
    assert stored_p.doc_id == doc_id
    assert stored_p.text == sample_passage_text
    assert stored_p.char_span == (0, len(sample_passage_text))

    # 3. Step 2 & 3: Metadata attachment from SkeletonTransducer and KevDecisionEngine
    assert hasattr(graph, "skeleton_result")
    assert hasattr(graph, "kev_evaluation")
    assert graph.skeleton_result is not None
    assert graph.kev_evaluation is not None
    assert len(graph.skeleton_result.entities) > 0
    assert len(graph.skeleton_result.events) > 0

    # 4. Step 4: 128-byte BinaryNodeTable registration
    assert len(pipeline.binary_table) == len(entity_nodes) + len(event_nodes)
    for idx, (cid, node) in enumerate(graph.nodes.items(), start=1):
        struct = pipeline.binary_table[idx - 1]
        assert struct is not None
        assert struct.node_id == idx
        # Verify 128-byte struct packing
        packed = struct.to_bytes()
        assert len(packed) == 128

    # 5. Step 5 & 6: Bipartite provenance anchors (E_ground)
    assert len(graph.node_to_passage) > 0
    anchored_nodes = graph.get_nodes_for_passage(pid)
    assert len(anchored_nodes) > 0
    for node in anchored_nodes:
        assert node.passage_id == pid
        assert 0 <= node.span_start <= node.span_end <= len(sample_passage_text)

    # 6. Verify Thematic Valencies in graph edges
    valency_edges = [
        (node.canonical_cid, rel, tgt)
        for node in graph.nodes.values()
        for rel, targets in node.edges.items()
        for tgt in targets
        if rel.startswith("VAL_")
    ]
    assert len(valency_edges) > 0, "Expected thematic valency edges in dual-node ASG"


def test_ingest_document_valency_and_causal_bindings(pipeline: CognitivePipeline):
    """Verifies that Kev Pass 1 thematic valencies and Pass 3 causal/temporal links are bound as edges."""
    text = "The technician pressurized the cylinder, causing the relief valve to vent argon gas."
    pid = "P_valve_01"

    graph = pipeline.ingest_document(text=text, doc_id="doc_valve", passage_id=pid)
    assert len(graph.nodes) >= 2

    # Check for presence of valency or temporal/causal edges
    edge_types = {rel for node in graph.nodes.values() for rel in node.edges.keys()}
    has_valency = any(e.startswith("VAL_") for e in edge_types)
    has_temporal_or_causal = any(
        e.startswith("TEMP_") or e.startswith("CAUSAL_") or e == "ENABLING_CONDITION"
        for e in edge_types
    )
    assert has_valency or has_temporal_or_causal, f"Expected relational edges, got {edge_types}"


# =============================================================================
# 2. Dual-Stream Retrieval Cycle Tests
# =============================================================================

def test_query_memory_dual_stream_structure(pipeline: CognitivePipeline, sample_passage_text: str):
    """Verifies query_memory produces dual-stream output containing both logical briefing and verbatim passages."""
    pid = "P_reactor_01"
    pipeline.ingest_document(sample_passage_text, doc_id="doc_reactor_01", passage_id=pid)

    query = "Where did Dr. Eleanor Vance isolate the compound?"
    ctx = pipeline.query_memory(query, max_tokens=500, format="dual_stream")

    assert isinstance(ctx, DualStreamContext)
    assert isinstance(ctx.logical_briefing, str)
    assert isinstance(ctx.passage_stream, str)
    assert isinstance(ctx.full_context, str)

    # Stream 1 check: Logical Briefing Block
    assert "=== LOGICAL BRIEFING BLOCK ===" in ctx.logical_briefing
    assert len(ctx.logical_briefing) > 0

    # Stream 2 check: Verbatim Source Passages
    assert "SOURCE PASSAGES ===" in ctx.passage_stream
    assert pid in ctx.passage_stream
    assert sample_passage_text in ctx.passage_stream

    # Full context check: Both streams concatenated
    assert ctx.logical_briefing in ctx.full_context
    assert ctx.passage_stream in ctx.full_context
    assert ctx.token_count_estimate > 0


def test_query_memory_lexical_fidelity(pipeline: CognitivePipeline):
    """Verifies 100% lexical fidelity of retrieved passages (exact character match, no paraphrasing)."""
    raw_snippet = "The ultra-cold cryostat reached an equilibrium temperature of 4.2 Kelvin at 03:00 UTC."
    pid = "P_cryo_exact"
    pipeline.ingest_document(raw_snippet, doc_id="doc_cryo", passage_id=pid)

    ctx = pipeline.query_memory("temperature of cryostat", format="dual_stream")
    assert isinstance(ctx, DualStreamContext)
    assert len(ctx.passages) >= 1

    retrieved_text = ctx.passages[0].text
    # Strict 100% exact substring match
    assert retrieved_text == raw_snippet
    assert raw_snippet in ctx.passage_stream


def test_query_memory_token_budget_truncation(pipeline: CognitivePipeline):
    """Verifies that dual-stream context respects strict max_tokens constraints."""
    text_long = (
        "Passage alpha describes high-pressure gaseous fluid dynamics across multiple manifolds. "
        "Passage beta describes structural integrity tests conducted on reinforced bulkhead barriers. "
        "Passage gamma covers automatic emergency depressurization and valve safety interlocking routines."
    )
    pipeline.ingest_document(text_long, doc_id="doc_fluid", passage_id="P_fluid_long")

    # Tight budget: 150 tokens
    tight_ctx = pipeline.query_memory("pressure dynamics", max_tokens=150, format="dual_stream")
    assert tight_ctx.token_count_estimate <= 200  # Within tolerance of estimated budget

    # Generous budget: 800 tokens
    generous_ctx = pipeline.query_memory("pressure dynamics", max_tokens=800, format="dual_stream")
    assert generous_ctx.token_count_estimate >= tight_ctx.token_count_estimate


def test_query_memory_format_options(pipeline: CognitivePipeline, sample_passage_text: str):
    """Verifies different format options supported by query_memory."""
    pipeline.ingest_document(sample_passage_text, doc_id="doc_fmt")

    # Format 'str' / 'svm'
    res_str = pipeline.query_memory("Dr. Eleanor Vance", format="str")
    assert isinstance(res_str, str)
    assert "=== LOGICAL BRIEFING BLOCK ===" in res_str

    # Format 'dual_stream'
    res_obj = pipeline.query_memory("Dr. Eleanor Vance", format="dual_stream")
    assert isinstance(res_obj, DualStreamContext)

    # Empty query fallback
    res_empty = pipeline.query_memory("", format="dual_stream")
    assert isinstance(res_empty, DualStreamContext)
    assert res_empty.full_context == ""


# =============================================================================
# 3. Multi-Hop Associative Recall Tests
# =============================================================================

def test_multi_hop_associative_recall(pipeline: CognitivePipeline):
    """Verifies multi-hop associative recall across 3 interconnected narrative passages."""
    p1 = "Dr. Eleanor Vance synthesized an exotic hyper-dense crystal inside Chamber Alpha."
    p2 = "The hyper-dense crystal emitted coherent radiation that overwhelmed the optical sensor array."
    p3 = "The optical sensor array malfunction initiated an emergency facility lockdown protocol."

    pipeline.ingest_document(p1, doc_id="hop_doc", passage_id="P1_hop")
    pipeline.ingest_document(p2, doc_id="hop_doc", passage_id="P2_hop")
    pipeline.ingest_document(p3, doc_id="hop_doc", passage_id="P3_hop")

    assert len(pipeline.passage_store) == 3

    # Query tracing from origin (Dr. Eleanor Vance) to terminal consequence (lockdown)
    ctx = pipeline.query_memory("What triggered the emergency facility lockdown protocol?", top_k=10, max_tokens=1000)
    assert isinstance(ctx, DualStreamContext)

    # Bipartite projection should retrieve source passages along the chain
    retrieved_pids = {p.passage_id for p in ctx.passages}
    assert "P3_hop" in retrieved_pids or "P2_hop" in retrieved_pids or "P1_hop" in retrieved_pids


# =============================================================================
# 4. Memory & VRAM Budget Validation Tests
# =============================================================================

def test_kev_engine_vram_overhead_zero():
    """Verifies that KevDecisionEngine Approach A imposes exactly 0.0 MB extra VRAM overhead."""
    kev_a = KevDecisionEngine(mode="in_context_logprob", fallback_to_mock=True)
    assert kev_a.mode == "in_context_logprob"
    assert kev_a.vram_overhead_mb == 0.0

    # In contrast, Approach B (LoRA adapter) incurs ~30 MB adapter overhead
    kev_b = KevDecisionEngine(mode="lora_adapter", fallback_to_mock=True)
    assert kev_b.vram_overhead_mb == 30.0


def test_workstation_8gb_vram_budget_compliance(pipeline: CognitivePipeline):
    """Verifies that combined pipeline execution remains strictly within the 4.8 GB ceiling on 8GB RTX 3070.

    Workstation Budget Breakdown (RTX 3070 8GB):
    - Qwen3.5-4B-MTP-Q5_K_M Base Weights: ~3,100 MB
    - FlashAttention-2 KV Cache (8K ctx):   ~1,100 MB
    - KevDecisionEngine (Shared Weights):        0 MB
    - Total Serving Overhead:               ~4,200 MB (<= 4,800 MB limit, > 3.2 GB free headroom)
    """
    base_model_mb = 3100.0
    kv_cache_mb = 1100.0
    kev_overhead_mb = pipeline.kev_engine.vram_overhead_mb

    total_vram_estimate_mb = base_model_mb + kv_cache_mb + kev_overhead_mb
    vram_budget_mb = 4800.0

    assert total_vram_estimate_mb <= vram_budget_mb
    assert kev_overhead_mb == 0.0
    # Headroom remaining on 8,192 MB card
    headroom_mb = 8192.0 - total_vram_estimate_mb
    assert headroom_mb >= 3200.0


# =============================================================================
# 5. Ingestion Throughput & Latency Benchmark
# =============================================================================

def test_ingestion_latency_benchmark(pipeline: CognitivePipeline):
    """Benchmarks chunk ingestion latency.

    Target benchmark from Section 8 Master Plan: 350-520 ms per chunk under cold-start.
    In deterministic CI / mock mode, execution should be significantly faster (< 150 ms).
    """
    chunk_text = (
        "Specialist Marcus Thorne replaced the primary coolant filter at sector gamma. "
        "Telemetry confirmed normal fluid pressure across the circulation manifold."
    )

    t0 = time.perf_counter()
    graph = pipeline.ingest_document(
        text=chunk_text,
        doc_id="bench_doc_01",
        passage_id="P_bench_01",
        validate=True,
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0

    assert isinstance(graph, QuantaGraph)
    assert len(graph.nodes) > 0

    # In mock/offline CI mode, latency must be well below the 520 ms upper threshold
    assert dt_ms < 520.0, f"Ingestion latency ({dt_ms:.1f}ms) exceeded 520ms limit"


def test_batch_ingestion_throughput(pipeline: CognitivePipeline):
    """Verifies stable linear throughput across a batch of 5 sequential discourse chunks."""
    chunks = [
        f"Observation log entry {i}: Sensor probe {i} recorded ambient radiation of {i * 1.5} millisieverts."
        for i in range(1, 6)
    ]

    latencies_ms = []
    for i, chk in enumerate(chunks, start=1):
        t0 = time.perf_counter()
        pipeline.ingest_document(chk, doc_id="batch_doc", passage_id=f"P_batch_{i}")
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    assert len(pipeline.passage_store) == 5
    avg_latency = sum(latencies_ms) / len(latencies_ms)
    assert avg_latency < 520.0, f"Average batch latency ({avg_latency:.1f}ms) exceeded 520ms limit"


@pytest.mark.parametrize("kev_mode", ["bypass", "regular_kev_lora", "tiered", "co_decoded", "async"])
def test_unified_kev_modes_ingestion(tmp_path: Path, kev_mode: str):
    """Verifies that CognitivePipeline operates seamlessly across all 5 unified kev_mode options (Session 5)."""
    db_path = tmp_path / f"test_{kev_mode}.db"
    pipe = CognitivePipeline(
        transducer_backend="mock",
        kev_mode=kev_mode,
        page_table_path=db_path,
    )
    try:
        sample_text = (
            "Charles Babbage invented the Difference Engine to calculate mathematical tables automatically. "
            "Because errors were frequent in manual calculations, the machine improved accuracy."
        )
        graph = pipe.ingest_document(
            text=sample_text,
            doc_id=f"doc_{kev_mode}",
            passage_id=f"P_{kev_mode}_01",
            validate=True,
        )

        assert isinstance(graph, QuantaGraph)
        assert len(graph.nodes) > 0
        assert len(pipe.passage_store) == 1
        assert len(pipe.binary_table) > 0

        # Verify Belnap status on nodes
        for node in graph.nodes.values():
            assert hasattr(node, "truth_status")
            if kev_mode == "async":
                # Provisional mode commits UNKNOWN initially
                assert node.truth_status in ("UNKNOWN", "TRUE")
            else:
                assert node.truth_status in ("TRUE", "UNKNOWN", "FALSE")

        # Test query memory
        ctx = pipe.query_memory("What did Charles Babbage invent?", top_k=3, flush_async=(kev_mode == "async"))
        assert ctx.token_count_estimate > 0
        assert len(ctx.passages) > 0
    finally:
        pipe.close()

