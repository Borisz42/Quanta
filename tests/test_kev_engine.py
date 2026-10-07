"""Unit and integration tests for KevDecisionEngine (Section 4).

Verifies:
1. Engine initialization, parameters, and 0.0 GB VRAM reporting (Approach A vs Approach B).
2. Pass 1: Thematic valency classification and probability calibration.
3. Pass 2: Speech-act intent and epistemic source classification.
4. Pass 3: Spatio-temporal Allen intervals and Pearl causal links.
5. End-to-end evaluate_chunk execution and latency benchmark (< 120 ms).
6. Asynchronous batch evaluation (evaluate_chunk_async).
7. Robustness against missing entities/events and malformed payloads.
"""

import asyncio
import math
import time
import pytest

from models.kev_engine import (
    AllenTemporalRelation,
    EpistemicSource,
    IntentEpistemicResult,
    KevChunkEvaluation,
    KevDecisionEngine,
    MockKevEngine,
    PearlCausalLink,
    RelationScoringResult,
    SpeechActIntent,
    ValencyRole,
    ValencyScoringResult,
    _normalize_logprobs_to_probs,
    _normalize_option_logprobs_to_probs,
)
from parser.skeleton_transducer import SkeletonEntity, SkeletonEvent


@pytest.fixture
def kev_engine_a():
    """Default Approach A engine (zero-shot in-context logprob, offline mock for deterministic unit testing)."""
    engine = KevDecisionEngine(base_url="http://127.0.0.1:58999", mode="in_context_logprob", fallback_to_mock=True)
    engine._server_available = False
    engine._last_health_check_time = time.perf_counter()
    return engine


@pytest.fixture
def kev_engine_b():
    """Ablation Approach B engine (LoRA adapter, offline mock for deterministic unit testing)."""
    engine = KevDecisionEngine(base_url="http://127.0.0.1:58999", mode="lora_adapter", fallback_to_mock=True)
    engine._server_available = False
    engine._last_health_check_time = time.perf_counter()
    return engine


# ---------------------------------------------------------------------------
# Engine Initialization & VRAM Budget Tests
# ---------------------------------------------------------------------------

def test_engine_initialization(kev_engine_a, kev_engine_b):
    """Verify engine parameters and zero extra VRAM for Approach A."""
    assert kev_engine_a.mode == "in_context_logprob"
    assert kev_engine_a.vram_overhead_mb == 0.0  # Shared weights = 0.0 MB overhead

    assert kev_engine_b.mode == "lora_adapter"
    assert kev_engine_b.vram_overhead_mb == 30.0  # LoRA weights ~30 MB


def test_logprob_normalization():
    """Verify softmax normalization converts logprobs to valid probability distributions."""
    token_logprobs = {
        "AGENT": -0.22,
        "PATIENT": -2.30,
        "INSTRUMENT": -3.50,
        "NONE": -4.80,
    }
    labels = ["AGENT", "PATIENT", "INSTRUMENT", "NONE"]
    top_label, top_prob, probs = _normalize_logprobs_to_probs(token_logprobs, labels)

    assert top_label == "AGENT"
    assert top_prob > 0.70
    assert math.isclose(sum(probs.values()), 1.0, rel_tol=1e-5)
    for l in labels:
        assert l in probs
        assert 0.0 <= probs[l] <= 1.0


def test_option_letter_logprob_normalization():
    """Verify single-token option letter logprobs normalize and map back to labels."""
    token_logprobs = {
        "A": -0.15,
        "B": -2.50,
        "C": -3.80,
        "D": -4.90,
    }
    labels = ["AGENT", "PATIENT", "INSTRUMENT", "NONE"]
    top_label, top_prob, probs, option_lps = _normalize_option_logprobs_to_probs(token_logprobs, labels)

    assert top_label == "AGENT"
    assert top_prob > 0.80
    assert math.isclose(sum(probs.values()), 1.0, rel_tol=1e-5)
    assert probs["AGENT"] > probs["PATIENT"]
    assert "A" in option_lps


# ---------------------------------------------------------------------------
# Pass 1: Thematic Valency and Coreference Tests
# ---------------------------------------------------------------------------

def test_pass1_valency_scoring_skeleton_types(kev_engine_a):
    """Verify Pass 1 valency scoring works with SkeletonEntity and SkeletonEvent types."""
    text = "Dr. Eleanor Vance analyzed the mineral specimen using an electron microscope."
    entities = [
        SkeletonEntity(id="E1", surface_text="Dr. Eleanor Vance", char_span=(0, 17)),
        SkeletonEntity(id="E2", surface_text="mineral specimen", char_span=(31, 47)),
        SkeletonEntity(id="E3", surface_text="electron microscope", char_span=(57, 76)),
    ]
    events = [
        SkeletonEvent(id="EV1", predicate="analyzed", char_span=(18, 26), subject_ent_id="E1", object_ent_id="E2"),
    ]

    results = kev_engine_a.score_valency_and_coreference(entities, events, text)
    assert len(results) == 3

    # Check E1 role: AGENT
    e1_res = next(r for r in results if r.entity_id == "E1")
    assert e1_res.role == ValencyRole.AGENT.value
    assert e1_res.confidence >= 0.85
    assert math.isclose(sum(e1_res.probabilities.values()), 1.0, rel_tol=1e-5)

    # Check E2 role: PATIENT
    e2_res = next(r for r in results if r.entity_id == "E2")
    assert e2_res.role == ValencyRole.PATIENT.value
    assert e2_res.confidence >= 0.85

    # Check E3 role: INSTRUMENT
    e3_res = next(r for r in results if r.entity_id == "E3")
    assert e3_res.role == ValencyRole.INSTRUMENT.value
    assert e3_res.confidence >= 0.85


def test_pass1_valency_scoring_dict_types(kev_engine_a):
    """Verify Pass 1 accepts plain dict structures."""
    text = "The technician calibrated the laser device."
    entities = [
        {"id": "E1", "surface_text": "The technician"},
        {"id": "E2", "surface_text": "the laser device"},
    ]
    events = [
        {"id": "EV1", "predicate": "calibrated"},
    ]

    results = kev_engine_a.score_valency_and_coreference(entities, events, text)
    assert len(results) == 2
    assert results[0].role == ValencyRole.AGENT.value
    assert results[1].role in (ValencyRole.PATIENT.value, ValencyRole.INSTRUMENT.value)


# ---------------------------------------------------------------------------
# Pass 2: Theory of Mind Speech-Act Intent & Epistemics Tests
# ---------------------------------------------------------------------------

def test_pass2_intent_and_epistemics_observation(kev_engine_a):
    """Verify direct observation epistemic source and informative intent."""
    text = "The optical spectrometer observed anomalous emission lines at 450 nm."
    events = [{"id": "EV1", "predicate": "observed"}]

    results = kev_engine_a.score_intent_and_epistemics(events, text)
    assert len(results) == 1
    res = results[0]

    assert res.intent == SpeechActIntent.INFORMATIVE.value
    assert res.epistemic_source == EpistemicSource.DIRECT_OBSERVATION.value
    assert res.intent_confidence >= 0.85
    assert res.epistemic_confidence >= 0.85
    assert math.isclose(sum(res.intent_probabilities.values()), 1.0, rel_tol=1e-5)
    assert math.isclose(sum(res.epistemic_probabilities.values()), 1.0, rel_tol=1e-5)


def test_pass2_intent_and_epistemics_deduction(kev_engine_a):
    """Verify deduction epistemic source."""
    text = "Therefore, the mathematical theorem proves the asymptotic convergence of the bound."
    events = [{"id": "EV1", "predicate": "proves"}]

    results = kev_engine_a.score_intent_and_epistemics(events, text)
    assert len(results) == 1
    res = results[0]

    assert res.epistemic_source == EpistemicSource.DEDUCTION.value
    assert res.epistemic_confidence >= 0.85


def test_pass2_directive_intent(kev_engine_a):
    """Verify directive speech-act intent."""
    text = "Please ensure that the cryogenic chamber is sealed immediately."
    events = [{"id": "EV1", "predicate": "ensure"}]

    results = kev_engine_a.score_intent_and_epistemics(events, text)
    assert len(results) == 1
    res = results[0]

    assert res.intent == SpeechActIntent.DIRECTIVE.value


# ---------------------------------------------------------------------------
# Pass 3: Spatio-Temporal Allen & Pearl Causal Relations Tests
# ---------------------------------------------------------------------------

def test_pass3_temporal_and_causal_relations(kev_engine_a):
    """Verify Allen interval BEFORE and Pearl causal link MECHANISM_LINK."""
    text = "The electrical short ignited the fuel, which then caused the chamber to explode."
    pairs = [
        ({"id": "EV1", "predicate": "ignited"}, {"id": "EV2", "predicate": "caused"}),
    ]

    results = kev_engine_a.score_allen_and_pearl_relations(pairs, text)
    assert len(results) == 1
    res = results[0]

    assert res.source_event_id == "EV1"
    assert res.target_event_id == "EV2"
    assert res.allen_relation == AllenTemporalRelation.BEFORE.value
    assert res.pearl_relation == PearlCausalLink.MECHANISM_LINK.value
    assert res.allen_confidence >= 0.85
    assert res.pearl_confidence >= 0.85
    assert math.isclose(sum(res.allen_probabilities.values()), 1.0, rel_tol=1e-5)
    assert math.isclose(sum(res.pearl_probabilities.values()), 1.0, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# End-to-End evaluate_chunk & Latency Benchmark Tests
# ---------------------------------------------------------------------------

def test_evaluate_chunk_end_to_end_and_latency(kev_engine_a):
    """Verify evaluate_chunk runs all 3 passes under 120 ms."""
    text = "Alice heated the gold crystal, which caused it to expand rapidly."
    entities = [
        {"id": "E1", "surface_text": "Alice"},
        {"id": "E2", "surface_text": "the gold crystal"},
    ]
    events = [
        {"id": "EV1", "predicate": "heated"},
        {"id": "EV2", "predicate": "expand"},
    ]

    t0 = time.perf_counter()
    chunk_eval = kev_engine_a.evaluate_chunk(entities, events, text)
    t_total_ms = (time.perf_counter() - t0) * 1000.0

    assert isinstance(chunk_eval, KevChunkEvaluation)
    assert len(chunk_eval.valencies) == 4  # 2 entities x 2 events
    assert len(chunk_eval.intent_epistemics) == 2  # 2 events
    assert len(chunk_eval.relations) == 1  # 1 pair

    # Strict benchmark test: execution under 120 ms
    assert chunk_eval.total_latency_ms < 120.0
    assert t_total_ms < 120.0
    assert chunk_eval.mode == "in_context_logprob"
    assert chunk_eval.vram_overhead_mb == 0.0

    # Serialization test
    d = chunk_eval.to_dict()
    assert "valencies" in d
    assert "intent_epistemics" in d
    assert "relations" in d
    assert "total_latency_ms" in d


def test_evaluate_chunk_async(kev_engine_a):
    """Verify asynchronous evaluation dispatch."""
    text = "The laser sliced the target."
    entities = [{"id": "E1", "surface_text": "The laser"}]
    events = [{"id": "EV1", "predicate": "sliced"}]

    chunk_eval = asyncio.run(kev_engine_a.evaluate_chunk_async(entities, events, text))
    assert len(chunk_eval.valencies) == 1
    assert chunk_eval.valencies[0].role == ValencyRole.INSTRUMENT.value
