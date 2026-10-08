"""Unit tests and benchmarks for KevAmbiguityGater and Tier 1 Decision Fast-Path (Session 1).

Verifies:
1. Canonical SVO frames bypass GPU calls and achieve 100% role accuracy (Agent/Patient/Informative/DirectObs).
2. Modal hedges ("allegedly forged", "might indicate", "ensure") correctly route to Kev.
3. Causal & temporal connectives ("causing", "before", "because") correctly trigger Allen/Pearl evaluation.
4. Prepositional ("using a microscope") and passive voice ambiguity route to Kev.
5. Gating execution latency strictly meets the sub-0.1 ms per chunk SLA (< 100 microseconds).
6. KevDecisionEngine seamless integration and mode toggling ('tiered', use_gating=True/False).
"""

import math
import time
import pytest

from models.kev_engine import (
    AllenTemporalRelation,
    EpistemicSource,
    IntentEpistemicResult,
    KevChunkEvaluation,
    KevDecisionEngine,
    PearlCausalLink,
    RelationScoringResult,
    SpeechActIntent,
    ValencyRole,
    ValencyScoringResult,
)
from models.kev_gating import (
    GatedDecisionPlan,
    KevAmbiguityGater,
)
from parser.skeleton_transducer import SkeletonEntity, SkeletonEvent


@pytest.fixture
def gater():
    """Default ambiguity gater instance."""
    return KevAmbiguityGater()


@pytest.fixture
def tiered_kev_engine():
    """KevDecisionEngine with Tier 1 gating enabled."""
    engine = KevDecisionEngine(base_url="http://127.0.0.1:58999", mode="tiered", fallback_to_mock=True)
    engine._server_available = False
    engine._last_health_check_time = time.perf_counter()
    return engine


# ---------------------------------------------------------------------------
# 1. Canonical SVO Fast-Path Tests
# ---------------------------------------------------------------------------

def test_canonical_svo_fast_path_bypasses_gpu(gater, tiered_kev_engine):
    """Verify canonical active declarative SVO sentence is fully fast-pathed without Kev calls."""
    text = "Charles Babbage invented the Difference Engine."
    entities = [
        SkeletonEntity(id="E1", surface_text="Charles Babbage", char_span=(0, 15)),
        SkeletonEntity(id="E2", surface_text="Difference Engine", char_span=(29, 46)),
    ]
    events = [
        SkeletonEvent(id="EV1", predicate="invented", char_span=(16, 24), subject_ent_id="E1", object_ent_id="E2"),
    ]

    # Analyze with gater directly
    plan = gater.analyze(entities, events, text)
    assert plan.is_fully_fast_path is True
    assert len(plan.ambiguous_valency_candidates) == 0
    assert len(plan.ambiguous_events) == 0
    assert len(plan.candidate_relation_pairs) == 0
    assert plan.analysis_latency_ms < 0.5  # Sub-millisecond analysis

    # Check fast-path valencies
    assert len(plan.fast_path_valencies) == 2
    e1_val = next(v for v in plan.fast_path_valencies if v.entity_id == "E1")
    e2_val = next(v for v in plan.fast_path_valencies if v.entity_id == "E2")
    assert e1_val.role == ValencyRole.AGENT.value
    assert e1_val.confidence >= 0.95
    assert e2_val.role == ValencyRole.PATIENT.value
    assert e2_val.confidence >= 0.95

    # Check fast-path intent & epistemic source
    assert len(plan.fast_path_intents) == 1
    ev_int = plan.fast_path_intents[0]
    assert ev_int.intent == SpeechActIntent.INFORMATIVE.value
    assert ev_int.intent_confidence >= 0.95
    assert ev_int.epistemic_source == EpistemicSource.DIRECT_OBSERVATION.value
    assert ev_int.epistemic_confidence >= 0.95

    # Evaluate end-to-end via KevDecisionEngine with use_gating=True
    t0 = time.perf_counter()
    chunk_eval = tiered_kev_engine.evaluate_chunk(entities, events, text, use_gating=True)
    t_eval_ms = (time.perf_counter() - t0) * 1000.0

    assert isinstance(chunk_eval, KevChunkEvaluation)
    assert "gated_fastpath" in chunk_eval.mode
    assert chunk_eval.pass1_latency_ms == 0.0
    assert chunk_eval.pass2_latency_ms == 0.0
    assert chunk_eval.pass3_latency_ms == 0.0
    assert t_eval_ms < 5.0  # Instantaneous execution (< 5 ms)
    assert len(chunk_eval.valencies) == 2
    assert len(chunk_eval.intent_epistemics) == 1


# ---------------------------------------------------------------------------
# 2. Epistemic Hedges & Modal Markers Routing Tests
# ---------------------------------------------------------------------------

def test_modal_hedge_allegedly_routes_to_kev(gater, tiered_kev_engine):
    """Verify hearsay hedge ('allegedly forged') routes event to Kev for epistemic classification."""
    text = "The suspect allegedly forged the financial certificate."
    entities = [
        {"id": "E1", "surface_text": "The suspect"},
        {"id": "E2", "surface_text": "financial certificate"},
    ]
    events = [
        {"id": "EV1", "predicate": "forged", "subject_ent_id": "E1", "object_ent_id": "E2"},
    ]

    plan = gater.analyze(entities, events, text)
    assert plan.is_fully_fast_path is False
    assert len(plan.ambiguous_events) == 1
    assert plan.ambiguous_events[0]["id"] == "EV1"

    # Evaluate chunk end-to-end
    chunk_eval = tiered_kev_engine.evaluate_chunk(entities, events, text, use_gating=True)
    assert len(chunk_eval.intent_epistemics) == 1
    ev_res = chunk_eval.intent_epistemics[0]
    assert ev_res.epistemic_source in (EpistemicSource.HEARSAY.value, EpistemicSource.CONJECTURE.value)


def test_modal_hedge_might_indicate_routes_to_kev(gater, tiered_kev_engine):
    """Verify conjecture hedge ('might indicate') routes event to Kev."""
    text = "The experimental measurements might indicate a subtle phase transition."
    entities = [
        {"id": "E1", "surface_text": "experimental measurements"},
        {"id": "E2", "surface_text": "phase transition"},
    ]
    events = [
        {"id": "EV1", "predicate": "indicate", "subject_ent_id": "E1", "object_ent_id": "E2"},
    ]

    plan = gater.analyze(entities, events, text)
    assert plan.is_fully_fast_path is False
    assert len(plan.ambiguous_events) == 1

    chunk_eval = tiered_kev_engine.evaluate_chunk(entities, events, text, use_gating=True)
    assert len(chunk_eval.intent_epistemics) == 1
    ev_res = chunk_eval.intent_epistemics[0]
    assert ev_res.epistemic_source == EpistemicSource.CONJECTURE.value


def test_directive_speech_act_routes_to_kev(gater, tiered_kev_engine):
    """Verify directive marker ('please ensure') routes event to Kev."""
    text = "Please ensure that the cryogenic chamber is sealed."
    entities = [
        {"id": "E1", "surface_text": "cryogenic chamber"},
    ]
    events = [
        {"id": "EV1", "predicate": "ensure", "object_ent_id": "E1"},
    ]

    plan = gater.analyze(entities, events, text)
    assert len(plan.ambiguous_events) == 1

    chunk_eval = tiered_kev_engine.evaluate_chunk(entities, events, text, use_gating=True)
    assert chunk_eval.intent_epistemics[0].intent == SpeechActIntent.DIRECTIVE.value


# ---------------------------------------------------------------------------
# 3. Causal & Temporal Discourse Connectives Tests
# ---------------------------------------------------------------------------

def test_causal_connective_triggers_allen_pearl_evaluation(gater, tiered_kev_engine):
    """Verify causal connective ('causing the laser to overheat') routes pair to Kev."""
    text = "The electrical short ignited the fuel, causing the laser to overheat."
    entities = [
        {"id": "E1", "surface_text": "electrical short"},
        {"id": "E2", "surface_text": "fuel"},
        {"id": "E3", "surface_text": "laser"},
    ]
    events = [
        {"id": "EV1", "predicate": "ignited", "subject_ent_id": "E1", "object_ent_id": "E2"},
        {"id": "EV2", "predicate": "overheat", "object_ent_id": "E3"},
    ]

    plan = gater.analyze(entities, events, text)
    assert len(plan.candidate_relation_pairs) == 1

    chunk_eval = tiered_kev_engine.evaluate_chunk(entities, events, text, use_gating=True)
    assert len(chunk_eval.relations) == 1
    rel = chunk_eval.relations[0]
    assert rel.pearl_relation == PearlCausalLink.MECHANISM_LINK.value
    assert rel.allen_relation in (AllenTemporalRelation.BEFORE.value, AllenTemporalRelation.MEETS.value)


def test_unlinked_declarative_sentences_fast_path_none_relation(gater, tiered_kev_engine):
    """Verify sentences without connectives assign fast-path NONE relations without Kev query."""
    text = "Charles Babbage invented the Difference Engine. Ada Lovelace wrote the algorithm."
    entities = [
        {"id": "E1", "surface_text": "Charles Babbage"},
        {"id": "E2", "surface_text": "Difference Engine"},
        {"id": "E3", "surface_text": "Ada Lovelace"},
        {"id": "E4", "surface_text": "algorithm"},
    ]
    events = [
        {"id": "EV1", "predicate": "invented", "subject_ent_id": "E1", "object_ent_id": "E2"},
        {"id": "EV2", "predicate": "wrote", "subject_ent_id": "E3", "object_ent_id": "E4"},
    ]

    plan = gater.analyze(entities, events, text)
    assert len(plan.candidate_relation_pairs) == 0
    assert len(plan.fast_path_relations) == 1
    fast_rel = plan.fast_path_relations[0]
    assert fast_rel.allen_relation == AllenTemporalRelation.NONE.value
    assert fast_rel.pearl_relation == PearlCausalLink.NONE.value


# ---------------------------------------------------------------------------
# 4. Prepositional & Passive Valency Ambiguity Tests
# ---------------------------------------------------------------------------

def test_prepositional_instrument_routes_to_kev(gater, tiered_kev_engine):
    """Verify 'using an electron microscope' routes the tool entity to Kev."""
    text = "Dr. Eleanor Vance analyzed the mineral specimen using an electron microscope."
    entities = [
        {"id": "E1", "surface_text": "Dr. Eleanor Vance"},
        {"id": "E2", "surface_text": "mineral specimen"},
        {"id": "E3", "surface_text": "electron microscope"},
    ]
    events = [
        {"id": "EV1", "predicate": "analyzed", "subject_ent_id": "E1", "object_ent_id": "E2"},
    ]

    plan = gater.analyze(entities, events, text)
    assert plan.is_fully_fast_path is False
    # E3 should be marked ambiguous (prepositional instrument)
    ambig_eids = [e["id"] if isinstance(e, dict) else e.id for e, ev in plan.ambiguous_valency_candidates]
    assert "E3" in ambig_eids

    chunk_eval = tiered_kev_engine.evaluate_chunk(entities, events, text, use_gating=True)
    assert len(chunk_eval.valencies) == 3

    e1_v = next(v for v in chunk_eval.valencies if v.entity_id == "E1")
    e2_v = next(v for v in chunk_eval.valencies if v.entity_id == "E2")
    e3_v = next(v for v in chunk_eval.valencies if v.entity_id == "E3")

    assert e1_v.role == ValencyRole.AGENT.value
    assert e2_v.role == ValencyRole.PATIENT.value
    assert e3_v.role == ValencyRole.INSTRUMENT.value


def test_passive_voice_routes_to_kev(gater, tiered_kev_engine):
    """Verify passive voice ('was analyzed by') routes entity-event candidates to Kev."""
    text = "The mineral specimen was analyzed by Dr. Eleanor Vance."
    entities = [
        {"id": "E1", "surface_text": "The mineral specimen"},
        {"id": "E2", "surface_text": "Dr. Eleanor Vance"},
    ]
    events = [
        {"id": "EV1", "predicate": "analyzed"},
    ]

    plan = gater.analyze(entities, events, text)
    assert len(plan.ambiguous_valency_candidates) >= 1

    chunk_eval = tiered_kev_engine.evaluate_chunk(entities, events, text, use_gating=True)
    assert len(chunk_eval.valencies) == 2


# ---------------------------------------------------------------------------
# 5. Gating Execution Latency Benchmark
# ---------------------------------------------------------------------------

def test_gating_latency_benchmark_under_100_microseconds(gater):
    """Verify gater.analyze executes in strictly < 0.1 ms (100 microseconds) per chunk."""
    text = "The technician calibrated the laser device and recorded the energy output."
    entities = [
        {"id": "E1", "surface_text": "The technician"},
        {"id": "E2", "surface_text": "laser device"},
        {"id": "E3", "surface_text": "energy output"},
    ]
    events = [
        {"id": "EV1", "predicate": "calibrated", "subject_ent_id": "E1", "object_ent_id": "E2"},
        {"id": "EV2", "predicate": "recorded", "subject_ent_id": "E1", "object_ent_id": "E3"},
    ]

    # Warmup
    for _ in range(50):
        gater.analyze(entities, events, text)

    # Benchmark 1,000 iterations
    iterations = 1000
    t0 = time.perf_counter()
    for _ in range(iterations):
        plan = gater.analyze(entities, events, text)
    elapsed_total_ms = (time.perf_counter() - t0) * 1000.0
    mean_latency_ms = elapsed_total_ms / iterations

    print(f"\n[BENCHMARK] Ambiguity Gating Latency: {mean_latency_ms * 1000.0:.2f} microseconds / chunk ({mean_latency_ms:.4f} ms)")
    assert mean_latency_ms < 0.10, f"Gater mean latency {mean_latency_ms:.4f} ms exceeds 0.10 ms SLA"


# ---------------------------------------------------------------------------
# 6. Serialization and Mode Toggling Tests
# ---------------------------------------------------------------------------

def test_gated_decision_plan_serialization(gater):
    """Verify GatedDecisionPlan to_dict serialization."""
    text = "Alice heated the sample."
    entities = [{"id": "E1", "surface_text": "Alice"}, {"id": "E2", "surface_text": "sample"}]
    events = [{"id": "EV1", "predicate": "heated", "subject_ent_id": "E1", "object_ent_id": "E2"}]

    plan = gater.analyze(entities, events, text)
    data = plan.to_dict()

    assert data["is_fully_fast_path"] is True
    assert len(data["fast_path_valencies"]) == 2
    assert len(data["fast_path_intents"]) == 1
    assert "analysis_latency_ms" in data


def test_kev_engine_mode_toggles():
    """Verify KevDecisionEngine mode and use_gating configuration flags."""
    eng_default = KevDecisionEngine(mode="in_context_logprob")
    assert eng_default.use_gating is False

    eng_explicit = KevDecisionEngine(mode="in_context_logprob", use_gating=True)
    assert eng_explicit.use_gating is True

    eng_tiered = KevDecisionEngine(mode="tiered")
    assert eng_tiered.use_gating is True
