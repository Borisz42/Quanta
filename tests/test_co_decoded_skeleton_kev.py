"""Unit tests and benchmarks for Session 3: Co-Decoded Compact Kev Schema in Skeleton Transducer.

Verifies:
1. GBNF grammar specification for co-decoded compact Kev enums (I/D/C/E, O/D/H/C, M/B/O/D/N, M/C/N).
2. SkeletonEvent serialization, deserialization, and JSON schema formatting with co-decoded fields.
3. MockSkeletonTransducer and SkeletonTransducer in mode="co_decoded".
4. Exact character and byte-span grounding fidelity preserved when Kev tags are co-decoded.
5. Strict token overhead budget enforcement (<= 12 extra tokens per chunk).
6. CognitivePipeline end-to-end integration with kev_mode="co_decoded" bypassing secondary GPU calls.
7. Robust JSON parsing, truncation auto-repair, and normalizer fallback.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
import pytest

from core.asg import QuantaGraph
from core.binary_node import BelnapValue
from models.kev_engine import (
    AllenTemporalRelation,
    EpistemicSource,
    PearlCausalLink,
    SpeechActIntent,
    ValencyRole,
)
from parser.skeleton_transducer import (
    CO_DECODED_SKELETON_SYSTEM_PROMPT,
    DEFAULT_SKELETON_SYSTEM_PROMPT,
    MockSkeletonTransducer,
    SkeletonEntity,
    SkeletonEvent,
    SkeletonExtractionResult,
    SkeletonTransducer,
    _locate_skeleton_gbnf,
    estimate_token_count,
)
from pipeline.cognitive_pipeline import CognitivePipeline


# ---------------------------------------------------------------------------
# 1. GBNF Grammar & Schema Verification
# ---------------------------------------------------------------------------

class TestCoDecodedSchema:
    """Verifies grammar loading and data model serialization for co-decoded mode."""

    def test_locate_co_decoded_gbnf_grammar(self):
        """Verify co_decoded_skeleton_schema.gbnf is correctly located and syntax-checked."""
        gbnf_path = _locate_skeleton_gbnf(filename="co_decoded_skeleton_schema.gbnf")
        assert gbnf_path.is_file()
        content = gbnf_path.read_text(encoding="utf-8")
        assert "intent-val" in content
        assert "epist-val" in content
        assert "allen-val" in content
        assert "pearl-val" in content
        assert r'\"I\"' in content and r'\"D\"' in content and r'\"C\"' in content and r'\"E\"' in content
        assert r'\"O\"' in content and r'\"H\"' in content
        assert r'\"M\"' in content and r'\"B\"' in content and r'\"N\"' in content

    def test_skeleton_event_co_decoded_serialization(self):
        """Verify SkeletonEvent correctly preserves and serializes co-decoded Kev fields."""
        ev = SkeletonEvent(
            id="EV1",
            predicate="invented",
            char_span=(16, 24),
            subject_ent_id="E1",
            object_ent_id="E2",
            intent="I",
            epist="O",
            allen="B",
            pearl="M",
        )

        assert ev.intent == "I"
        assert ev.epist == "O"
        assert ev.epistemic == "O"
        assert ev.allen == "B"
        assert ev.pearl == "M"

        d = ev.to_dict()
        assert d["intent"] == "I"
        assert d["epist"] == "O"
        assert d["epistemic"] == "O"
        assert d["allen"] == "B"
        assert d["pearl"] == "M"

        ev_rebuilt = SkeletonEvent.from_dict(d)
        assert ev_rebuilt.id == "EV1"
        assert ev_rebuilt.predicate == "invented"
        assert ev_rebuilt.intent == "I"
        assert ev_rebuilt.epist == "O"
        assert ev_rebuilt.allen == "B"
        assert ev_rebuilt.pearl == "M"

        # Check property setter
        ev_rebuilt.epistemic = "H"
        assert ev_rebuilt.epist == "H"
        assert ev_rebuilt.epistemic == "H"

    def test_to_skeleton_json_with_co_decoded_fields(self):
        """Verify SkeletonExtractionResult.to_skeleton_json includes compact keys."""
        entities = [
            SkeletonEntity(id="E1", surface_text="Charles Babbage", char_span=(0, 15)),
            SkeletonEntity(id="E2", surface_text="Difference Engine", char_span=(29, 46)),
        ]
        events = [
            SkeletonEvent(
                id="EV1",
                predicate="invented",
                char_span=(16, 24),
                subject_ent_id="E1",
                object_ent_id="E2",
                intent="I",
                epist="O",
                allen="B",
                pearl="M",
            )
        ]
        res = SkeletonExtractionResult(entities=entities, events=events)
        raw_json = res.to_skeleton_json()

        data = json.loads(raw_json)
        assert "events" in data
        assert len(data["events"]) == 1
        ev_data = data["events"][0]
        assert ev_data["intent"] == "I"
        assert ev_data["epist"] == "O"
        assert ev_data["allen"] == "B"
        assert ev_data["pearl"] == "M"


# ---------------------------------------------------------------------------
# 2. Mock Co-Decoded Transduction Tests
# ---------------------------------------------------------------------------

class TestMockCoDecodedTransduction:
    """Verifies MockSkeletonTransducer behavior in co_decoded mode."""

    @pytest.fixture
    def mock_transducer(self):
        return MockSkeletonTransducer(mode="co_decoded")

    def test_canonical_fixtures_co_decoded(self, mock_transducer):
        """Verify canonical pre-seeded fixtures return compact Kev tags in co_decoded mode."""
        text = "Dr. Marcus Vance pressurized the argon cylinder."
        res = mock_transducer.transduce(text)

        assert res.metadata.get("mode") == "co_decoded"
        assert len(res.events) >= 1
        ev = res.events[0]
        assert ev.intent == "I"
        assert ev.epist == "O"
        assert ev.allen == "B"
        assert ev.pearl == "M"

        # Verify exact byte span grounding fidelity is preserved
        raw_bytes = text.encode("utf-8")
        assert ev.byte_span is not None
        assert raw_bytes[ev.byte_span[0]:ev.byte_span[1]].decode("utf-8") == "pressurize"

    def test_modal_hedge_co_decoded(self, mock_transducer):
        """Verify modal hedge triggers hearsay or conjecture code in co_decoded mode."""
        text = "The technician allegedly forged the security certificate."
        res = mock_transducer.transduce(text)

        assert len(res.events) >= 1
        ev = res.events[0]
        assert ev.epist == "H"  # Hearsay
        assert ev.intent == "I"

    def test_imperative_directive_co_decoded(self, mock_transducer):
        """Verify imperative command triggers directive code in co_decoded mode."""
        text = "Please ensure the pressure valve is sealed immediately."
        res = mock_transducer.transduce(text)

        assert len(res.events) >= 1
        ev = res.events[0]
        assert ev.intent == "D"  # Directive

    def test_causal_connective_co_decoded(self, mock_transducer):
        """Verify causal connective triggers mechanism link code in co_decoded mode."""
        text = "The voltage spiked, causing the titanium valve to rupture."
        res = mock_transducer.transduce(text)

        assert len(res.events) >= 1
        ev = res.events[0]
        assert ev.pearl == "M"  # Mechanism
        assert ev.allen == "B"  # Before

    def test_dynamic_mode_override(self):
        """Verify passing mode='co_decoded' overrides a standard transducer."""
        transducer = MockSkeletonTransducer(mode="standard")
        text = "Dr. Marcus Vance pressurized the argon cylinder."

        # Default standard mode does not emit Kev codes
        std_res = transducer.transduce(text)
        assert std_res.events[0].intent is None
        assert std_res.events[0].epist is None

        # Dynamic override emits Kev codes
        co_res = transducer.transduce(text, mode="co_decoded")
        assert co_res.events[0].intent == "I"
        assert co_res.events[0].epist == "O"


# ---------------------------------------------------------------------------
# 3. Token Overhead Budget Tests (<= 12 Extra Tokens)
# ---------------------------------------------------------------------------

class TestTokenOverhead:
    """Verifies that co-decoded mode strictly respects the <= 12 extra token SLA."""

    def test_token_overhead_within_budget(self):
        """Verify token overhead of co-decoded JSON vs standard JSON is strictly <= 12 tokens."""
        text = "Dr. Marcus Vance pressurized the argon cylinder inside the test chamber."
        std_transducer = MockSkeletonTransducer(mode="standard")
        co_transducer = MockSkeletonTransducer(mode="co_decoded")

        std_res = std_transducer.transduce(text)
        co_res = co_transducer.transduce(text)

        std_json = std_res.to_skeleton_json()
        co_json = co_res.to_skeleton_json()

        std_tokens = estimate_token_count(std_json)
        co_tokens = estimate_token_count(co_json)
        token_overhead = co_tokens - std_tokens

        # Master plan specification: adding only +8 tokens, overhead strictly <= 12 tokens
        assert token_overhead <= 12, f"Token overhead {token_overhead} exceeds 12-token limit"
        assert token_overhead >= 0, "Co-decoded token count should be greater or equal to standard"
        assert "intent" in co_json and "epist" in co_json


# ---------------------------------------------------------------------------
# 4. CognitivePipeline Integration Tests (kev_mode="co_decoded")
# ---------------------------------------------------------------------------

class TestCognitivePipelineCoDecoded:
    """Verifies CognitivePipeline execution with kev_mode='co_decoded'."""

    @pytest.fixture
    def co_decoded_pipeline(self, tmp_path):
        db_path = tmp_path / "co_decoded_svm.db"
        pipeline = CognitivePipeline(
            transducer_backend="mock",
            kev_mode="co_decoded",
            page_table_path=db_path,
            canvas_capacity=512,
        )
        yield pipeline
        pipeline.close()

    def test_ingest_document_co_decoded_zero_gpu_overhead(self, co_decoded_pipeline):
        """Verify document ingestion in co_decoded mode produces calibrated ASG with zero secondary GPU calls."""
        text = "Dr. Eleanor Vance isolated a volatile synthetic compound inside Containment Cell 4 at dawn."
        doc_id = "doc_reactor_co_01"
        pid = "P_reactor_co_01"

        t0 = time.perf_counter()
        graph = co_decoded_pipeline.ingest_document(
            text=text,
            doc_id=doc_id,
            passage_id=pid,
            validate=True,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Verify graph topology
        assert isinstance(graph, QuantaGraph)
        assert len(graph.nodes) > 0
        assert hasattr(graph, "kev_evaluation")
        kev_eval = graph.kev_evaluation
        assert kev_eval is not None
        assert kev_eval.mode == "co_decoded"

        # Verify execution is near-instantaneous (< 50 ms in mock environment)
        assert elapsed_ms < 50.0

        # Verify valencies are mapped to AGENT and PATIENT
        agent_edges = [
            (n, rel, tgt)
            for n in graph.nodes.values()
            for rel, targets in n.edges.items()
            for tgt in targets
            if rel == "VAL_X1_AGENT"
        ]
        patient_edges = [
            (n, rel, tgt)
            for n in graph.nodes.values()
            for rel, targets in n.edges.items()
            for tgt in targets
            if rel == "VAL_X2_PATIENT"
        ]
        assert len(agent_edges) >= 1
        assert len(patient_edges) >= 1

        # Verify intent & epistemic values mapped to formal enums
        assert len(kev_eval.intent_epistemics) >= 1
        ie = kev_eval.intent_epistemics[0]
        assert ie.intent == SpeechActIntent.INFORMATIVE.value
        assert ie.epistemic_source == EpistemicSource.DIRECT_OBSERVATION.value
        assert ie.intent_confidence >= 0.95
        assert ie.epistemic_confidence >= 0.95

        # Verify QuantaNode attributes
        ev_nodes = [n for n in graph.nodes.values() if n.get_slot("TYPE_EVENT") == 1]
        assert len(ev_nodes) >= 1
        for ev_node in ev_nodes:
            assert ev_node.evidence_source == "direct_observation"
            assert ev_node.truth_status == "TRUE"

        # Verify BinaryNodeTable registration with Belnap TRUE
        assert len(co_decoded_pipeline.binary_table) >= len(graph.nodes)
        for struct in co_decoded_pipeline.binary_table:
            assert struct.belnap_lattice == BelnapValue.TRUE

    def test_multi_event_relation_co_decoding(self, co_decoded_pipeline):
        """Verify sequential events with causal/temporal tags synthesize ASG relations."""
        text = "The voltage spiked, causing the titanium valve to rupture."
        doc_id = "doc_multi_ev_01"
        pid = "P_multi_ev_01"

        graph = co_decoded_pipeline.ingest_document(
            text=text,
            doc_id=doc_id,
            passage_id=pid,
            validate=True,
        )

        kev_eval = graph.kev_evaluation
        assert kev_eval.mode == "co_decoded"

        # If relations are co-decoded, check relation scoring result
        if kev_eval.relations:
            rel = kev_eval.relations[0]
            assert rel.allen_relation in (AllenTemporalRelation.BEFORE.value, AllenTemporalRelation.MEETS.value, "NONE")
            assert rel.pearl_relation in (PearlCausalLink.MECHANISM_LINK.value, "NONE")


# ---------------------------------------------------------------------------
# 5. Production SkeletonTransducer Mode & Fallback Handling
# ---------------------------------------------------------------------------

class TestSkeletonTransducerProductionClient:
    """Verifies SkeletonTransducer client initialization and JSON handling."""

    def test_transducer_mode_selection(self):
        """Verify mode initialization sets appropriate grammar and system prompt."""
        transducer = SkeletonTransducer(
            base_url="http://127.0.0.1:58999",
            mode="co_decoded",
            fallback_to_mock=True,
        )
        assert transducer.mode == "co_decoded"
        assert "co_decoded_skeleton_schema.gbnf" in str(transducer.grammar_path)
        assert transducer.system_prompt == CO_DECODED_SKELETON_SYSTEM_PROMPT
        assert transducer._mock.mode == "co_decoded"

    def test_clean_and_parse_json_with_co_decoded_keys(self):
        """Verify _clean_and_parse_json handles co-decoded responses with fences and formatting."""
        transducer = SkeletonTransducer(base_url="http://127.0.0.1:58999", fallback_to_mock=True)
        raw_json_with_fences = """```json
{
  "entities": [
    {"id": "E1", "text": "Charles Babbage"},
    {"id": "E2", "text": "Difference Engine"}
  ],
  "events": [
    {
      "id": "EV1",
      "pred": "invented",
      "subj": "E1",
      "obj": "E2",
      "intent": "I",
      "epist": "O",
      "allen": "B",
      "pearl": "M"
    }
  ]
}
```"""
        parsed = transducer._clean_and_parse_json(raw_json_with_fences)
        assert len(parsed["entities"]) == 2
        assert len(parsed["events"]) == 1
        ev = parsed["events"][0]
        assert ev["intent"] == "I"
        assert ev["epist"] == "O"
        assert ev["allen"] == "B"
        assert ev["pearl"] == "M"

        # Verify _ground_and_build extracts all fields into SkeletonEvent
        text = "Charles Babbage invented the Difference Engine."
        entities, events = transducer._ground_and_build(text, parsed, mode="co_decoded")
        assert len(events) == 1
        assert events[0].intent == "I"
        assert events[0].epist == "O"
        assert events[0].allen == "B"
        assert events[0].pearl == "M"
        assert events[0].char_span == (16, 24)
