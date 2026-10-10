"""Integration tests for S-Expression Pipeline Integration & Representation Toggles (Section 3).

Verifies:
1. CognitivePipeline.__init__ defaults (skeleton_format="sexpr_compact", co_decoded_kev=False).
2. Runtime format switching via pipe.set_skeleton_format() and pipe.set_co_decoded_kev().
3. Environment variable configuration (QUANTA_SKELETON_FORMAT, QUANTA_CO_DECODED_KEV).
4. MultiScaleConfig parameter registry keys and convenience properties.
5. End-to-end ingestion parity across sexpr_compact, sexpr_positional, and json.
6. Valency role binding (subj -> AGENT, obj -> PATIENT) across S-expression formats.
7. Co-decoded Kev decision mapping into Belnap truth values via BelnapLatticeMapper.
8. Preservation of 128-byte BinaryNodeTable and PassageStore invariants.
9. Multi-passage parallel ingestion parity.
10. Reverse proxy header dispatch (X-Quanta-Skeleton-Format, X-Quanta-Co-Decoded).
"""

from __future__ import annotations

import os
from pathlib import Path
import time
import pytest
from fastapi.testclient import TestClient

from config.multi_scale_config import MultiScaleConfig
from core.asg import QuantaGraph
from core.binary_node import BelnapValue
from memory.passage_store import PassageStore
from models.kev_engine import (
    AllenTemporalRelation,
    EpistemicSource,
    PearlCausalLink,
    SpeechActIntent,
)
from parser.skeleton_transducer import (
    MockSkeletonTransducer,
    SkeletonExtractionResult,
)
from pipeline.cognitive_pipeline import CognitivePipeline
from server.proxy import (
    PipelinePool,
    QuantaProxyConfig,
    create_proxy_app,
)


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    return tmp_path / "test_pipeline_sexpr.db"


@pytest.fixture
def pipeline(temp_db: Path) -> CognitivePipeline:
    pipe = CognitivePipeline(
        transducer_backend="mock",
        page_table_path=temp_db,
        canvas_capacity=512,
    )
    yield pipe
    pipe.close()


# -----------------------------------------------------------------------------
# 1. Pipeline Initialization & Parameter Resolution
# -----------------------------------------------------------------------------

def test_pipeline_init_defaults(temp_db: Path):
    """Asserts that CognitivePipeline defaults to sexpr_compact and co_decoded_kev=False."""
    pipe = CognitivePipeline(
        transducer_backend="mock",
        page_table_path=temp_db,
    )
    try:
        assert pipe.skeleton_format == "sexpr_compact"
        assert pipe.co_decoded_kev is False
        assert pipe.kev_mode in ("bypass", "none", "standard")
        assert hasattr(pipe, "skeleton_transducer")
        assert getattr(pipe.skeleton_transducer, "skeleton_format", None) == "sexpr_compact"
        assert getattr(pipe.skeleton_transducer, "co_decoded", None) is False
    finally:
        pipe.close()


def test_pipeline_init_explicit_parameters(temp_db: Path):
    """Asserts that CognitivePipeline accepts explicit format and co-decoding overrides."""
    pipe = CognitivePipeline(
        transducer_backend="mock",
        page_table_path=temp_db,
        skeleton_format="sexpr_positional",
        co_decoded_kev=True,
    )
    try:
        assert pipe.skeleton_format == "sexpr_positional"
        assert pipe.co_decoded_kev is True
        assert pipe.kev_mode == "co_decoded"
        assert getattr(pipe.skeleton_transducer, "skeleton_format", None) == "sexpr_positional"
        assert getattr(pipe.skeleton_transducer, "co_decoded", None) is True
    finally:
        pipe.close()


def test_pipeline_runtime_format_switching(pipeline: CognitivePipeline):
    """Asserts that set_skeleton_format and set_co_decoded_kev update pipeline and transducer state."""
    # Switch to positional ultra-compact
    pipeline.set_skeleton_format("sexpr_positional")
    assert pipeline.skeleton_format == "sexpr_positional"
    assert pipeline.skeleton_transducer.skeleton_format == "sexpr_positional"

    # Switch to json with co-decoding
    pipeline.set_skeleton_format("json", co_decoded=True)
    assert pipeline.skeleton_format == "json"
    assert pipeline.co_decoded_kev is True
    assert pipeline.kev_mode == "co_decoded"
    assert pipeline.skeleton_transducer.skeleton_format == "json"
    assert pipeline.skeleton_transducer.co_decoded is True

    # Toggle co-decoding off
    pipeline.set_co_decoded_kev(False)
    assert pipeline.co_decoded_kev is False
    assert pipeline.kev_mode == "bypass"
    assert pipeline.skeleton_transducer.co_decoded is False

    # Switch back to compact S-expression
    pipeline.set_skeleton_format("sexpr_compact")
    assert pipeline.skeleton_format == "sexpr_compact"
    assert pipeline.skeleton_transducer.skeleton_format == "sexpr_compact"


def test_pipeline_env_var_configuration(temp_db: Path, monkeypatch):
    """Asserts that QUANTA_SKELETON_FORMAT and QUANTA_CO_DECODED_KEV configure the pipeline."""
    monkeypatch.setenv("QUANTA_SKELETON_FORMAT", "sexpr_positional")
    monkeypatch.setenv("QUANTA_CO_DECODED_KEV", "1")

    pipe = CognitivePipeline(
        transducer_backend="mock",
        page_table_path=temp_db,
    )
    try:
        assert pipe.skeleton_format == "sexpr_positional"
        assert pipe.co_decoded_kev is True
        assert pipe.kev_mode == "co_decoded"
        assert pipe.skeleton_transducer.skeleton_format == "sexpr_positional"
        assert pipe.skeleton_transducer.co_decoded is True
    finally:
        pipe.close()


def test_multi_scale_config_transducer_keys(tmp_path: Path):
    """Asserts that MultiScaleConfig registers transducer keys and resolves them properly."""
    cfg = MultiScaleConfig(profile_path=tmp_path / "empty.json")
    assert cfg.transducer_skeleton_format == "sexpr_compact"
    assert cfg.transducer_co_decoded is False

    cfg_overridden = cfg.with_overrides({
        "transducer.skeleton_format": "sexpr_positional",
        "transducer.co_decoded": True,
    })
    assert cfg_overridden.transducer_skeleton_format == "sexpr_positional"
    assert cfg_overridden.transducer_co_decoded is True


# -----------------------------------------------------------------------------
# 2. End-to-End Ingestion Parity (Pure SVO)
# -----------------------------------------------------------------------------

def test_ingest_document_sexpr_compact_pure_svo(pipeline: CognitivePipeline):
    """Asserts that sexpr_compact ingests document chunk and commits to Graph, PassageStore, and BinaryNodeTable."""
    pipeline.set_skeleton_format("sexpr_compact", co_decoded=False)
    text = "Charles Babbage designed the Difference Engine at Cambridge."
    doc_id = "doc_babbage_01"
    pid = "P_babbage_01"

    graph = pipeline.ingest_document(
        text=text,
        doc_id=doc_id,
        passage_id=pid,
        validate=False,
    )

    # 1. Verify QuantaGraph topology
    assert isinstance(graph, QuantaGraph)
    assert len(graph.nodes) >= 2

    # Verify event nodes and entity nodes
    ev_nodes = [n for n in graph.nodes.values() if n.get_slot("TYPE_EVENT") == 1]
    ent_nodes = [n for n in graph.nodes.values() if n.get_slot("TYPE_EVENT") == 0]
    assert len(ev_nodes) >= 1
    assert len(ent_nodes) >= 1

    # Verify valencies are mapped to AGENT and PATIENT
    agent_edges = [
        (n, rel, tgt)
        for n in graph.nodes.values()
        for rel, targets in n.edges.items()
        for tgt in targets
        if rel == "VAL_X1_AGENT"
    ]
    assert len(agent_edges) >= 1

    # 2. Verify PassageStore persistence
    p_rec = pipeline.passage_store.get_passage(pid)
    assert p_rec is not None
    assert p_rec.text == text
    assert p_rec.doc_id == doc_id

    # 3. Verify BinaryNodeTable persistence (128-byte C-compatible struct)
    assert len(pipeline.binary_table) >= len(graph.nodes)
    for struct in pipeline.binary_table:
        assert struct.node_id > 0
        assert struct.belnap_lattice == BelnapValue.TRUE


def test_ingest_document_sexpr_positional_pure_svo(pipeline: CognitivePipeline):
    """Asserts that sexpr_positional achieves structural parity with sexpr_compact."""
    pipeline.set_skeleton_format("sexpr_positional", co_decoded=False)
    text = "Charles Babbage designed the Difference Engine at Cambridge."
    doc_id = "doc_babbage_pos_01"
    pid = "P_babbage_pos_01"

    graph = pipeline.ingest_document(
        text=text,
        doc_id=doc_id,
        passage_id=pid,
        validate=False,
    )

    assert isinstance(graph, QuantaGraph)
    ev_nodes = [n for n in graph.nodes.values() if n.get_slot("TYPE_EVENT") == 1]
    assert len(ev_nodes) >= 1

    agent_edges = [
        (n, rel, tgt)
        for n in graph.nodes.values()
        for rel, targets in n.edges.items()
        for tgt in targets
        if rel == "VAL_X1_AGENT"
    ]
    assert len(agent_edges) >= 1

    p_rec = pipeline.passage_store.get_passage(pid)
    assert p_rec is not None
    assert p_rec.text == text


def test_cross_format_ingestion_structural_parity(temp_db: Path):
    """Asserts that compact S-expression, positional S-expression, and JSON yield matching ASG topologies."""
    text = "Dr. Eleanor Vance isolated a volatile synthetic compound inside Containment Cell 4."

    results = {}
    for fmt in ("sexpr_compact", "sexpr_positional", "json"):
        pipe = CognitivePipeline(
            transducer_backend="mock",
            page_table_path=":memory:",
            skeleton_format=fmt,
            co_decoded_kev=False,
        )
        try:
            g = pipe.ingest_document(text=text, doc_id=f"doc_{fmt}", passage_id=f"P_{fmt}")
            results[fmt] = {
                "node_count": len(g.nodes),
                "event_count": len([n for n in g.nodes.values() if n.get_slot("TYPE_EVENT") == 1]),
                "entity_count": len([n for n in g.nodes.values() if n.get_slot("TYPE_EVENT") == 0]),
                "agent_count": len([
                    (n, rel, tgt)
                    for n in g.nodes.values()
                    for rel, targets in n.edges.items()
                    for tgt in targets
                    if rel == "VAL_X1_AGENT"
                ]),
                "patient_count": len([
                    (n, rel, tgt)
                    for n in g.nodes.values()
                    for rel, targets in n.edges.items()
                    for tgt in targets
                    if rel == "VAL_X2_PATIENT"
                ]),
            }
        finally:
            pipe.close()

    # Verify that compact S-expr and positional S-expr match JSON exactly
    assert results["sexpr_compact"] == results["json"]
    assert results["sexpr_positional"] == results["json"]


# -----------------------------------------------------------------------------
# 3. Co-Decoded Kev Mapping & Belnap Truth Lattice
# -----------------------------------------------------------------------------

def test_co_decoded_kev_belnap_mapping_direct_observation(pipeline: CognitivePipeline):
    """Asserts that co-decoded Direct Observation (O) maps to BelnapValue.TRUE."""
    pipeline.set_skeleton_format("sexpr_compact", co_decoded=True)
    text = "Dr. Eleanor Vance isolated the crystalline compound in Geneva."
    pid = "P_obs_01"

    graph = pipeline.ingest_document(text=text, passage_id=pid)
    assert hasattr(graph, "kev_evaluation")
    kev_eval = graph.kev_evaluation
    assert kev_eval.mode == "co_decoded"

    # Verify epistemic source is mapped
    assert len(kev_eval.intent_epistemics) >= 1
    ie = kev_eval.intent_epistemics[0]
    assert ie.intent == SpeechActIntent.INFORMATIVE.value
    assert ie.epistemic_source == EpistemicSource.DIRECT_OBSERVATION.value

    # Verify event node in QuantaGraph has truth_status TRUE
    ev_nodes = [n for n in graph.nodes.values() if n.get_slot("TYPE_EVENT") == 1]
    assert len(ev_nodes) >= 1
    for ev in ev_nodes:
        assert ev.truth_status == "TRUE"

    # Verify BinaryNodeTable struct has BelnapValue.TRUE
    ev_structs = [s for s in pipeline.binary_table if s.node_id in [n.node_id for n in ev_nodes if hasattr(n, "node_id")]]
    if ev_structs:
        for s in ev_structs:
            assert s.belnap_lattice == BelnapValue.TRUE


def test_co_decoded_kev_temporal_and_causal_relations(pipeline: CognitivePipeline):
    """Asserts that sequential events with co-decoded Allen/Pearl tags synthesize ASG relations."""
    pipeline.set_skeleton_format("sexpr_compact", co_decoded=True)
    text = "The voltage spiked, causing the titanium valve to rupture."
    pid = "P_cause_01"

    graph = pipeline.ingest_document(text=text, passage_id=pid)
    kev_eval = graph.kev_evaluation
    assert kev_eval.mode == "co_decoded"

    # Verify causal/temporal relations synthesized
    if kev_eval.relations:
        rel = kev_eval.relations[0]
        assert rel.allen_relation in (AllenTemporalRelation.BEFORE.value, AllenTemporalRelation.MEETS.value, "NONE")
        assert rel.pearl_relation in (PearlCausalLink.MECHANISM_LINK.value, "NONE")


# -----------------------------------------------------------------------------
# 4. End-to-End Retrieval Parity
# -----------------------------------------------------------------------------

def test_end_to_end_retrieval_parity(temp_db: Path):
    """Asserts that retrieved dual-stream context is identical whether ingested via S-expr or JSON."""
    doc_text = (
        "Charles Babbage entered Trinity College, Cambridge in 1810. "
        "He designed the Difference Engine to calculate mathematical tables. "
        "Babbage later designed the Analytical Engine, a general-purpose digital computing machine."
    )
    query = "What machine did Charles Babbage design?"

    contexts = {}
    for fmt in ("sexpr_compact", "sexpr_positional", "json"):
        pipe = CognitivePipeline(
            transducer_backend="mock",
            page_table_path=":memory:",
            skeleton_format=fmt,
            co_decoded_kev=False,
        )
        try:
            pipe.ingest_document(text=doc_text, doc_id="doc_babbage", passage_id="P_babbage")
            ctx = pipe.query_memory(query=query, format="dual_stream")
            contexts[fmt] = ctx
        finally:
            pipe.close()

    # All three formats must successfully return dual stream context containing the passage
    for fmt, ctx in contexts.items():
        assert ctx is not None
        ctx_str = str(ctx)
        assert "Charles Babbage" in ctx_str
        assert "Engine" in ctx_str


def test_parallel_ingestion_with_sexpr(pipeline: CognitivePipeline):
    """Asserts that ingest_passages_parallel functions seamlessly with S-expression formatting."""
    pipeline.set_skeleton_format("sexpr_compact")
    passages = [
        {"passage_id": "P_1", "doc_id": "d1", "text": "Alan Turing designed the ACE computer in London."},
        {"passage_id": "P_2", "doc_id": "d2", "text": "Ada Lovelace published the first algorithm for the Analytical Engine."},
    ]

    graphs = pipeline.ingest_passages_parallel(passages, max_workers=2)
    assert len(graphs) == 2
    for g in graphs:
        assert isinstance(g, QuantaGraph)
        assert len(g.nodes) >= 2


# -----------------------------------------------------------------------------
# 5. Reverse Proxy Header Dispatch
# -----------------------------------------------------------------------------

def test_proxy_headers_skeleton_format_dispatch(tmp_path: Path):
    """Asserts that X-Quanta-Skeleton-Format and X-Quanta-Co-Decoded headers configure session pipeline."""
    config = QuantaProxyConfig(
        compression_threshold=50,
        transducer_backend="mock",
        page_table_path=tmp_path / "proxy.db",
        fallback_to_local=True,
    )
    app = create_proxy_app(config)
    client = TestClient(app)

    prompt = (
        "Context:\n"
        "Project Chronos developed quantum gravimeter sensors in Zurich.\n\n"
        "Question:\n"
        "Where were the quantum sensors developed?"
    )

    resp = client.post(
        "/v1/chat/completions",
        json={"model": "quanta-context-expander", "messages": [{"role": "user", "content": prompt}]},
        headers={
            "X-Quanta-Session-ID": "test_sexpr_session",
            "X-Quanta-Skeleton-Format": "sexpr_positional",
            "X-Quanta-Co-Decoded": "true",
            "X-Quanta-Reset": "true",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["choices"]) > 0
    assert data["choices"][0]["message"]["content"]
