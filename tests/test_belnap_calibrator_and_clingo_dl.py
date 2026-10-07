"""Unit and integration tests for Calibrated Belnap Truth Lattice & clingo-dl Gate (Section 5).

Verifies:
1. Belnap threshold mapping (P >= 0.85 -> TRUE, P <= 0.15 -> FALSE, 0.15 < P < 0.85 -> UNKNOWN, Conflict -> CONTRA).
2. Uint8 confidence quantization and dequantization precision.
3. Temperature scaling calibration, ECE, and Brier score.
4. Belnap lattice algebraic operations (knowledge meet/join and multi-source reconciliation).
5. Difference-logic constraint solving on valid temporal sequences (BEFORE, MEETS, OVERLAPS, DURING).
6. Difference-logic detection of synthetic temporal paradoxes and minimal unsatisfiable core (MUC) extraction.
7. Pearl causal DAG acyclicity and causal paradox detection.
8. Zero-copy ConceptNet grounding latency (<= 0.35 us per concept code) and 128-byte struct population.
9. Verification that MUC false rejection rate on valid narrative sequences is < 1.5%.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List
import numpy as np
import pytest

from core.asg import QuantaGraph, QuantaNode
from core.binary_node import (
    BelnapValue,
    QuantaSemanticNodeStruct,
    SpeechActIntent,
    EpistemicSource,
)
from parser.asg_compiler import compile_to_binary_table
from parser.mmap_grounder import MmapLexicalGrounder
from verification.belnap_calibrator import (
    BelnapLatticeMapper,
    TemperatureCalibrator,
    dequantize_confidence,
    quantize_confidence,
)
from verification.clingo_dl_gate import (
    ClingoDLGate,
    DifferenceConstraint,
    DifferenceLogicSolver,
    DLValidationResult,
)


class TestBelnapCalibrator:
    """Tests for Belnap 4-valued lattice mapping and confidence quantization."""

    def test_threshold_mapping(self):
        mapper = BelnapLatticeMapper(true_threshold=0.85, false_threshold=0.15)

        # True states (P >= 0.85)
        assert mapper.map_probability(0.85) == BelnapValue.TRUE
        assert mapper.map_probability(0.92) == BelnapValue.TRUE
        assert mapper.map_probability(1.00) == BelnapValue.TRUE

        # False states (P <= 0.15)
        assert mapper.map_probability(0.15) == BelnapValue.FALSE
        assert mapper.map_probability(0.08) == BelnapValue.FALSE
        assert mapper.map_probability(0.00) == BelnapValue.FALSE

        # Unknown states (0.15 < P < 0.85)
        assert mapper.map_probability(0.16) == BelnapValue.UNKNOWN
        assert mapper.map_probability(0.50) == BelnapValue.UNKNOWN
        assert mapper.map_probability(0.84) == BelnapValue.UNKNOWN

        # Conflict state
        assert mapper.map_probability(0.95, has_conflict=True) == BelnapValue.CONTRADICTION
        assert mapper.map_probability(0.05, has_conflict=True) == BelnapValue.CONTRADICTION

    def test_confidence_quantization(self):
        # Uint8 quantization: P in [0.0, 1.0] -> uint8 in [0, 255]
        assert quantize_confidence(0.0) == 0
        assert quantize_confidence(1.0) == 255
        assert quantize_confidence(0.5) == 128

        # Clipping
        assert quantize_confidence(-0.5) == 0
        assert quantize_confidence(1.5) == 255

        # Dequantization
        assert dequantize_confidence(0) == 0.0
        assert dequantize_confidence(255) == 1.0
        assert abs(dequantize_confidence(128) - 0.5019) < 0.005

        # Round-trip error <= 1/255
        for p in np.linspace(0.0, 1.0, 51):
            u8 = quantize_confidence(p)
            p_rec = dequantize_confidence(u8)
            assert abs(p - p_rec) <= (1.0 / 255.0 + 1e-6)

    def test_vectorized_mapping(self):
        mapper = BelnapLatticeMapper(true_threshold=0.85, false_threshold=0.15)
        probs = np.array([0.95, 0.05, 0.50, 0.85, 0.15, 0.70], dtype=np.float32)

        belnap_codes, conf_u8 = mapper.map_probabilities_to_belnap(probs)

        expected_belnap = np.array([
            BelnapValue.TRUE,
            BelnapValue.FALSE,
            BelnapValue.UNKNOWN,
            BelnapValue.TRUE,
            BelnapValue.FALSE,
            BelnapValue.UNKNOWN,
        ], dtype=np.uint8)

        np.testing.assert_array_equal(belnap_codes, expected_belnap)
        assert conf_u8[0] == quantize_confidence(0.95)
        assert conf_u8[1] == quantize_confidence(0.05)
        assert conf_u8[2] == quantize_confidence(0.50)

    def test_temperature_calibrator(self):
        cal = TemperatureCalibrator(temperature=2.0)
        # T > 1 softens probabilities towards 0.5
        p_raw = 0.90
        p_cal = cal.calibrate(p_raw)
        assert p_cal < p_raw
        assert p_cal > 0.50

        # T < 1 sharpens probabilities
        cal_sharp = TemperatureCalibrator(temperature=0.5)
        p_sharp = cal_sharp.calibrate(p_raw)
        assert p_sharp > p_raw

        # Array calibration
        probs = np.array([0.1, 0.5, 0.9])
        cal_probs = cal.calibrate(probs)
        assert cal_probs[0] > 0.1
        assert abs(cal_probs[1] - 0.5) < 1e-4
        assert cal_probs[2] < 0.9

        # Metrics computation
        labels = np.array([0.0, 0.0, 1.0])
        ece = TemperatureCalibrator.compute_ece(cal_probs, labels, n_bins=5)
        brier = TemperatureCalibrator.compute_brier_score(cal_probs, labels)
        assert 0.0 <= ece <= 1.0
        assert 0.0 <= brier <= 1.0

    def test_lattice_algebra(self):
        # Knowledge meet (information lower bound)
        assert BelnapLatticeMapper.knowledge_meet(BelnapValue.TRUE, BelnapValue.TRUE) == BelnapValue.TRUE
        assert BelnapLatticeMapper.knowledge_meet(BelnapValue.FALSE, BelnapValue.FALSE) == BelnapValue.FALSE
        assert BelnapLatticeMapper.knowledge_meet(BelnapValue.TRUE, BelnapValue.FALSE) == BelnapValue.UNKNOWN
        assert BelnapLatticeMapper.knowledge_meet(BelnapValue.UNKNOWN, BelnapValue.TRUE) == BelnapValue.UNKNOWN
        assert BelnapLatticeMapper.knowledge_meet(BelnapValue.CONTRADICTION, BelnapValue.TRUE) == BelnapValue.TRUE

        # Knowledge join (information pooling)
        assert BelnapLatticeMapper.knowledge_join(BelnapValue.TRUE, BelnapValue.TRUE) == BelnapValue.TRUE
        assert BelnapLatticeMapper.knowledge_join(BelnapValue.FALSE, BelnapValue.FALSE) == BelnapValue.FALSE
        assert BelnapLatticeMapper.knowledge_join(BelnapValue.TRUE, BelnapValue.FALSE) == BelnapValue.CONTRADICTION
        assert BelnapLatticeMapper.knowledge_join(BelnapValue.UNKNOWN, BelnapValue.TRUE) == BelnapValue.TRUE
        assert BelnapLatticeMapper.knowledge_join(BelnapValue.CONTRADICTION, BelnapValue.TRUE) == BelnapValue.CONTRADICTION

    def test_multi_source_reconciliation(self):
        mapper = BelnapLatticeMapper(true_threshold=0.85, false_threshold=0.15)

        # Agreement on TRUE
        state, conf = mapper.reconcile_sources([0.90, 0.88, 0.95])
        assert state == BelnapValue.TRUE
        assert conf >= 0.85

        # Agreement on FALSE
        state, conf = mapper.reconcile_sources([0.10, 0.05, 0.12])
        assert state == BelnapValue.FALSE
        assert conf <= 0.15

        # Clash: Source 1 says TRUE (0.92), Source 2 says FALSE (0.08) -> CONTRADICTION!
        state, conf = mapper.reconcile_sources([0.92, 0.08])
        assert state == BelnapValue.CONTRADICTION


class TestClingoDLGate:
    """Tests for polynomial Difference Logic temporal and causal verification."""

    def test_valid_temporal_sequence(self):
        gate = ClingoDLGate()

        # EV1 -> BEFORE -> EV2 -> BEFORE -> EV3 -> BEFORE -> EV4
        events = [
            {"id": "EV1", "time": (0, 10)},
            {"id": "EV2", "time": (12, 20)},
            {"id": "EV3", "time": (22, 30)},
            {"id": "EV4", "time": (32, 40)},
        ]
        relations = [
            {"type": "BEFORE", "source": "EV1", "target": "EV2"},
            {"type": "BEFORE", "source": "EV2", "target": "EV3"},
            {"type": "BEFORE", "source": "EV3", "target": "EV4"},
        ]

        result = gate.validate(events, relations)
        assert result.is_valid is True
        assert result.diagnostic is None
        assert len(result.assigned_times) > 0

    def test_valid_allen_relations(self):
        gate = ClingoDLGate()

        # Meets: EV1 meets EV2
        events = [{"id": "EV1"}, {"id": "EV2"}]
        relations = [{"type": "MEETS", "source": "EV1", "target": "EV2"}]
        res = gate.validate(events, relations)
        assert res.is_valid is True

        # Overlaps: EV1 overlaps EV2
        relations = [{"type": "OVERLAPS", "source": "EV1", "target": "EV2"}]
        res = gate.validate(events, relations)
        assert res.is_valid is True

        # During: EV2 during EV1
        relations = [{"type": "DURING", "source": "EV2", "target": "EV1"}]
        res = gate.validate(events, relations)
        assert res.is_valid is True

    def test_synthetic_temporal_paradox_cycle(self):
        gate = ClingoDLGate()

        # Grandfather-type paradox cycle:
        # EV1 before EV2, EV2 before EV3, EV3 before EV1
        events = [{"id": "EV1"}, {"id": "EV2"}, {"id": "EV3"}]
        relations = [
            {"type": "BEFORE", "source": "EV1", "target": "EV2"},
            {"type": "BEFORE", "source": "EV2", "target": "EV3"},
            {"type": "BEFORE", "source": "EV3", "target": "EV1"},
        ]

        result = gate.validate(events, relations)
        assert result.is_valid is False
        assert result.diagnostic is not None
        assert result.diagnostic.category == "temporal"
        assert set(result.muc_nodes).issubset({"EV1", "EV2", "EV3"})
        assert len(result.muc_nodes) > 0

        # Assert structured [REPAIR REQUEST] format
        prompt = result.diagnostic.format_repair_request()
        assert "[REPAIR REQUEST]" in prompt
        assert "Allen temporal calculus" in prompt
        assert "CONFLICT:" in prompt

    def test_direct_temporal_contradiction(self):
        gate = ClingoDLGate()

        # EV1 meets EV2 AND EV1 after EV2
        events = [{"id": "EV1"}, {"id": "EV2"}]
        relations = [
            {"type": "MEETS", "source": "EV1", "target": "EV2"},
            {"type": "AFTER", "source": "EV1", "target": "EV2"},
        ]

        result = gate.validate(events, relations)
        assert result.is_valid is False
        assert result.diagnostic is not None
        assert result.diagnostic.category == "temporal"
        assert "EV1" in result.muc_nodes or "EV2" in result.muc_nodes

    def test_causal_dag_acyclicity(self):
        gate = ClingoDLGate()

        # Valid causal DAG: EV1 -> CAUSES -> EV2 -> CAUSES -> EV3
        events = [{"id": "EV1"}, {"id": "EV2"}, {"id": "EV3"}]
        relations = [
            {"type": "CAUSES", "source": "EV1", "target": "EV2"},
            {"type": "CAUSES", "source": "EV2", "target": "EV3"},
        ]
        result = gate.validate(events, relations)
        assert result.is_valid is True

        # Contradictory causal cycle: EV1 causes EV2, EV2 causes EV1
        relations_cycle = [
            {"type": "CAUSES", "source": "EV1", "target": "EV2"},
            {"type": "CAUSES", "source": "EV2", "target": "EV1"},
        ]
        result_cycle = gate.validate(events, relations_cycle)
        assert result_cycle.is_valid is False
        assert result_cycle.diagnostic is not None
        assert result_cycle.diagnostic.category == "causal"
        assert set(result_cycle.muc_nodes) == {"EV1", "EV2"}
        assert "[REPAIR REQUEST]" in result_cycle.diagnostic.format_repair_request()

    def test_graph_validation(self):
        gate = ClingoDLGate()

        graph = QuantaGraph()
        n1 = QuantaNode(anchor="EV1", node_type="event")
        n1.time_start = 0
        n1.time_end = 10

        n2 = QuantaNode(anchor="EV2", node_type="event")
        n2.time_start = 15
        n2.time_end = 25

        graph.add_node(n1)
        graph.add_node(n2)
        n1.add_edge("TEMP_ALLEN_BEFORE", n2.cid)

        res = gate.validate_graph(graph)
        assert res.is_valid is True

        # Introduce contradiction in graph
        n2.add_edge("TEMP_ALLEN_BEFORE", n1.cid)
        res_cycle = gate.validate_graph(graph)
        assert res_cycle.is_valid is False
        assert res_cycle.diagnostic.category == "temporal"


class TestConceptNetGroundingAndBinaryStruct:
    """Tests for zero-copy ConceptNet grounding latency and 128-byte struct layout."""

    def test_mmap_grounding_latency_and_accuracy(self):
        grounder = MmapLexicalGrounder.get_default()
        assert grounder.is_available() is True
        assert grounder._num_concepts > 0

        # Warmup
        code_cat = grounder.resolve_concept_code("cat")
        code_dog = grounder.resolve_concept_code("dog")
        assert code_cat is not None
        assert code_dog is not None
        assert isinstance(code_cat, int)
        assert 0 <= code_cat < 65536

        # Benchmark 20,000 lookups to verify <= 0.35 us per lookup
        N = 20000
        t0 = time.perf_counter()
        for _ in range(N):
            _ = grounder.resolve_concept_code("cat")
        t1 = time.perf_counter()

        per_lookup_us = ((t1 - t0) / N) * 1e6
        print(f"\n[BENCHMARK] Mmap concept code lookup latency: {per_lookup_us:.4f} us/lookup")
        assert per_lookup_us <= 0.35, f"Lookup exceeded 0.35 us target: {per_lookup_us:.4f} us"

    def test_populate_128byte_binary_struct(self):
        grounder = MmapLexicalGrounder.get_default()

        # Build a QuantaGraph with an event and entity
        graph = QuantaGraph()
        ent = QuantaNode(anchor="cat")
        ent.passage_id = "P1"
        ent.span_start = 5
        ent.span_end = 8
        ent.truth_status = "TRUE"
        ent.confidence = 0.95

        graph.add_node(ent)

        table = compile_to_binary_table(graph)
        assert len(table) == 1

        struct = table[0]
        assert struct.node_id == 1
        assert struct.passage_id == 1
        assert struct.span_start == 5
        assert struct.span_end == 8
        assert struct.concept_code == grounder.resolve_concept_code("cat")
        assert struct.belnap_lattice == BelnapValue.TRUE
        assert struct.confidence == quantize_confidence(0.95)
        assert struct.NODE_SIZE == 128

    def test_narrative_muc_rejection_rate_under_threshold(self):
        """Verifies that on 200 valid synthetic narrative sequences, rejection rate is < 1.5%."""
        gate = ClingoDLGate()

        np.random.seed(42)
        n_trials = 200
        rejections = 0

        for trial in range(n_trials):
            n_events = np.random.randint(3, 8)
            events = []
            relations = []

            # Generate sequentially valid temporal timestamps
            curr_t = 0
            for i in range(n_events):
                duration = np.random.randint(2, 10)
                events.append({
                    "id": f"EV{i+1}",
                    "time": (curr_t, curr_t + duration),
                })
                if i > 0:
                    relations.append({
                        "type": "BEFORE",
                        "source": f"EV{i}",
                        "target": f"EV{i+1}",
                    })
                curr_t += duration + np.random.randint(1, 5)

            # Add non-conflicting causal relations (e.g. EV1 causes EV2)
            if n_events >= 2:
                relations.append({
                    "type": "CAUSES",
                    "source": "EV1",
                    "target": "EV2",
                })

            res = gate.validate(events, relations)
            if not res.is_valid:
                rejections += 1

        rejection_rate = (rejections / n_trials) * 100.0
        print(f"\n[METRIC] Narrative MUC rejection rate: {rejection_rate:.2f}% (Threshold: < 1.5%)")
        assert rejection_rate < 1.5, f"MUC rejection rate {rejection_rate:.2f}% exceeded 1.5% ceiling"
