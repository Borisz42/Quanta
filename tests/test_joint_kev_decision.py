"""Unit tests and benchmarks for Session 2: Joint Consolidated Multi-Slot Decision Transducer.

Verifies:
1. Joint prompt formatting consolidates ambiguous valencies, events, and relations into a compact single-pass prompt.
2. Structured output parsing handles full enum strings, abbreviations, and single-letter codes.
3. Fallback normalizer gracefully handles partial truncation, missing slots, and empty model completions.
4. End-to-end multi-event narrative evaluation with modal hedges, instruments, and causal links.
5. Single-pass mock execution latency strictly meets the sub-5 ms SLA (< 5.0 ms).
6. KevDecisionEngine mode switching ('tiered', 'joint_multi_slot', 'joint_multi_slot_lora').
7. VRAM overhead reporting (0.0 MB for zero-shot vs 30.0 MB for dynamic LoRA).
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
def joint_engine():
    """Default KevDecisionEngine in joint_multi_slot mode (offline mock)."""
    engine = KevDecisionEngine(base_url="http://127.0.0.1:58999", mode="joint_multi_slot", fallback_to_mock=True)
    engine._server_available = False
    engine._last_health_check_time = time.perf_counter()
    return engine


@pytest.fixture
def tiered_engine():
    """KevDecisionEngine in tiered mode combining gating + joint multi-slot."""
    engine = KevDecisionEngine(base_url="http://127.0.0.1:58999", mode="tiered", fallback_to_mock=True)
    engine._server_available = False
    engine._last_health_check_time = time.perf_counter()
    return engine


@pytest.fixture
def lora_joint_engine():
    """KevDecisionEngine in joint_multi_slot_lora mode."""
    engine = KevDecisionEngine(base_url="http://127.0.0.1:58999", mode="joint_multi_slot_lora", fallback_to_mock=True)
    engine._server_available = False
    engine._last_health_check_time = time.perf_counter()
    return engine


# ---------------------------------------------------------------------------
# 1. Joint Prompt Formatting Tests
# ---------------------------------------------------------------------------

def test_joint_prompt_formatting(joint_engine, gater):
    """Verify consolidated multi-slot prompt structure."""
    text = "The technician allegedly cleaned the substrate using an ultrasonic bath, causing the laser to align."
    entities = [
        {"id": "E1", "surface_text": "The technician"},
        {"id": "E2", "surface_text": "substrate"},
        {"id": "E3", "surface_text": "ultrasonic bath"},
        {"id": "E4", "surface_text": "laser"},
    ]
    events = [
        {"id": "EV1", "predicate": "cleaned", "subject_ent_id": "E1", "object_ent_id": "E2"},
        {"id": "EV2", "predicate": "align", "object_ent_id": "E4"},
    ]

    plan = gater.analyze(entities, events, text)
    assert not plan.is_fully_fast_path

    prompt, slot_count = joint_engine._format_joint_prompt(plan, text)

    assert slot_count >= 2
    assert "Task: Classify ambiguous linguistic slots" in prompt
    assert "Context: " in prompt
    assert "Properties to classify:" in prompt
    assert "Decisions:" in prompt
    assert "EV_EV1:" in prompt
    assert "VAL_E3_EV1:" in prompt
    assert "REL_EV1_EV2:" in prompt


# ---------------------------------------------------------------------------
# 2. Parsing Robustness: Full Enums, Abbreviations & Single-Letter Codes
# ---------------------------------------------------------------------------

def test_joint_decision_full_enums_parsing(joint_engine):
    """Verify parser extracts full canonical enum strings."""
    plan = GatedDecisionPlan(
        fast_path_valencies=[],
        fast_path_intents=[],
        ambiguous_valency_candidates=[({"id": "E3", "surface_text": "probe"}, {"id": "EV1", "predicate": "scanned"})],
        ambiguous_events=[{"id": "EV1", "predicate": "scanned"}],
        candidate_relation_pairs=[({"id": "EV1", "predicate": "scanned"}, {"id": "EV2", "predicate": "detected"})],
    )
    raw_completion = (
        "EV_EV1: [intent=INFORMATIVE, epist=DIRECT_OBSERVATION]\n"
        "VAL_E3_EV1: [role=INSTRUMENT]\n"
        "REL_EV1_EV2: [allen=BEFORE, pearl=MECHANISM_LINK]\n"
    )

    valencies, intents, relations = joint_engine._parse_joint_decisions(raw_completion, plan, "sample text")

    assert len(valencies) == 1
    assert valencies[0].role == ValencyRole.INSTRUMENT.value
    assert valencies[0].confidence >= 0.95

    assert len(intents) == 1
    assert intents[0].intent == SpeechActIntent.INFORMATIVE.value
    assert intents[0].epistemic_source == EpistemicSource.DIRECT_OBSERVATION.value

    assert len(relations) == 1
    assert relations[0].allen_relation == AllenTemporalRelation.BEFORE.value
    assert relations[0].pearl_relation == PearlCausalLink.MECHANISM_LINK.value


def test_joint_decision_abbreviated_parsing(joint_engine):
    """Verify parser extracts abbreviated codes (INFO, OBS, INST, BEF, MECH)."""
    plan = GatedDecisionPlan(
        fast_path_valencies=[],
        fast_path_intents=[],
        ambiguous_valency_candidates=[({"id": "E3", "surface_text": "probe"}, {"id": "EV1", "predicate": "scanned"})],
        ambiguous_events=[{"id": "EV1", "predicate": "scanned"}],
        candidate_relation_pairs=[({"id": "EV1", "predicate": "scanned"}, {"id": "EV2", "predicate": "detected"})],
    )
    raw_completion = (
        "EV_EV1: [intent=INFO, epist=OBS]\n"
        "VAL_E3_EV1: [role=INST]\n"
        "REL_EV1_EV2: [allen=BEF, pearl=MECH]\n"
    )

    valencies, intents, relations = joint_engine._parse_joint_decisions(raw_completion, plan, "sample text")

    assert valencies[0].role == ValencyRole.INSTRUMENT.value
    assert intents[0].intent == SpeechActIntent.INFORMATIVE.value
    assert intents[0].epistemic_source == EpistemicSource.DIRECT_OBSERVATION.value
    assert relations[0].allen_relation == AllenTemporalRelation.BEFORE.value
    assert relations[0].pearl_relation == PearlCausalLink.MECHANISM_LINK.value


def test_joint_decision_single_letter_parsing(joint_engine):
    """Verify parser extracts compact 1-letter codes (I, O, I, B, M)."""
    plan = GatedDecisionPlan(
        fast_path_valencies=[],
        fast_path_intents=[],
        ambiguous_valency_candidates=[({"id": "E3", "surface_text": "probe"}, {"id": "EV1", "predicate": "scanned"})],
        ambiguous_events=[{"id": "EV1", "predicate": "scanned"}],
        candidate_relation_pairs=[({"id": "EV1", "predicate": "scanned"}, {"id": "EV2", "predicate": "detected"})],
    )
    raw_completion = (
        "EV_EV1: intent=I, epist=O\n"
        "VAL_E3_EV1: role=I\n"
        "REL_EV1_EV2: allen=B, pearl=M\n"
    )

    valencies, intents, relations = joint_engine._parse_joint_decisions(raw_completion, plan, "sample text")

    assert valencies[0].role == ValencyRole.INSTRUMENT.value
    assert intents[0].intent == SpeechActIntent.INFORMATIVE.value
    assert intents[0].epistemic_source == EpistemicSource.DIRECT_OBSERVATION.value
    assert relations[0].allen_relation == AllenTemporalRelation.BEFORE.value
    assert relations[0].pearl_relation == PearlCausalLink.MECHANISM_LINK.value


def test_joint_decision_colon_space_format(joint_engine):
    """Verify parser extracts colon-space format without brackets."""
    plan = GatedDecisionPlan(
        fast_path_valencies=[],
        fast_path_intents=[],
        ambiguous_valency_candidates=[({"id": "E3", "surface_text": "probe"}, {"id": "EV1", "predicate": "scanned"})],
        ambiguous_events=[{"id": "EV1", "predicate": "scanned"}],
        candidate_relation_pairs=[({"id": "EV1", "predicate": "scanned"}, {"id": "EV2", "predicate": "detected"})],
    )
    raw_completion = (
        "EV_EV1: intent: INFORMATIVE, epist: DIRECT_OBSERVATION\n"
        "VAL_E3_EV1: role: INSTRUMENT\n"
        "REL_EV1_EV2: allen: BEFORE, pearl: MECHANISM_LINK\n"
    )

    valencies, intents, relations = joint_engine._parse_joint_decisions(raw_completion, plan, "sample text")

    assert valencies[0].role == ValencyRole.INSTRUMENT.value
    assert intents[0].intent == SpeechActIntent.INFORMATIVE.value
    assert intents[0].epistemic_source == EpistemicSource.DIRECT_OBSERVATION.value
    assert relations[0].allen_relation == AllenTemporalRelation.BEFORE.value
    assert relations[0].pearl_relation == PearlCausalLink.MECHANISM_LINK.value


# ---------------------------------------------------------------------------
# 3. Truncation & Fallback Handling
# ---------------------------------------------------------------------------

def test_joint_decision_truncated_fallback(joint_engine):
    """Verify graceful recovery when model completion is cut off mid-sentence."""
    plan = GatedDecisionPlan(
        fast_path_valencies=[],
        fast_path_intents=[],
        ambiguous_valency_candidates=[({"id": "E3", "surface_text": "probe"}, {"id": "EV1", "predicate": "scanned"})],
        ambiguous_events=[{"id": "EV1", "predicate": "scanned"}],
        candidate_relation_pairs=[({"id": "EV1", "predicate": "scanned"}, {"id": "EV2", "predicate": "detected"})],
    )
    # Output truncated after intent, missing epistemic, valency, and relation
    truncated_completion = "EV_EV1: [intent=DIRECTIVE, "

    valencies, intents, relations = joint_engine._parse_joint_decisions(
        truncated_completion,
        plan,
        "Please ensure the probe is calibrated.",
    )

    # Intent extracted from partial text
    assert len(intents) == 1
    assert intents[0].intent == SpeechActIntent.DIRECTIVE.value
    # Epistemic fell back safely without crashing
    assert intents[0].epistemic_source is not None

    # Missing slots fell back to deterministic mock
    assert len(valencies) == 1
    assert valencies[0].role in (ValencyRole.INSTRUMENT.value, ValencyRole.AGENT.value, ValencyRole.PATIENT.value)

    assert len(relations) == 1
    assert relations[0].allen_relation is not None
    assert relations[0].pearl_relation is not None


def test_joint_decision_empty_completion_recovery(joint_engine):
    """Verify empty string completion recovers cleanly via fallback."""
    plan = GatedDecisionPlan(
        fast_path_valencies=[],
        fast_path_intents=[],
        ambiguous_valency_candidates=[({"id": "E1", "surface_text": "Alice"}, {"id": "EV1", "predicate": "measured"})],
        ambiguous_events=[{"id": "EV1", "predicate": "measured"}],
        candidate_relation_pairs=[],
    )

    valencies, intents, relations = joint_engine._parse_joint_decisions("", plan, "Alice measured the temperature.")

    assert len(valencies) == 1
    assert valencies[0].role == ValencyRole.AGENT.value
    assert len(intents) == 1
    assert intents[0].intent == SpeechActIntent.INFORMATIVE.value


# ---------------------------------------------------------------------------
# 4. End-to-End Multi-Event Narrative Evaluation
# ---------------------------------------------------------------------------

def test_joint_decision_end_to_end_mock_narrative(joint_engine, gater):
    """Verify end-to-end evaluation of complex scientific narrative."""
    text = "The suspect allegedly forged the documentation using a compromised terminal, causing the audit to trigger."
    entities = [
        {"id": "E1", "surface_text": "The suspect"},
        {"id": "E2", "surface_text": "documentation"},
        {"id": "E3", "surface_text": "compromised terminal"},
        {"id": "E4", "surface_text": "audit"},
    ]
    events = [
        {"id": "EV1", "predicate": "forged", "subject_ent_id": "E1", "object_ent_id": "E2"},
        {"id": "EV2", "predicate": "trigger", "object_ent_id": "E4"},
    ]

    plan = gater.analyze(entities, events, text)
    assert not plan.is_fully_fast_path

    chunk_eval = joint_engine.evaluate_joint_decision(plan, text)

    assert isinstance(chunk_eval, KevChunkEvaluation)
    assert chunk_eval.mode == "joint_multi_slot_joint"
    assert len(chunk_eval.valencies) >= 3
    assert len(chunk_eval.intent_epistemics) == 2
    assert len(chunk_eval.relations) == 1

    # Check that hearsay hedge was resolved
    ev1_ie = next(ie for ie in chunk_eval.intent_epistemics if ie.event_id == "EV1")
    assert ev1_ie.epistemic_source in (EpistemicSource.HEARSAY.value, EpistemicSource.CONJECTURE.value)

    # Check that causal connective was resolved
    rel = chunk_eval.relations[0]
    assert rel.pearl_relation == PearlCausalLink.MECHANISM_LINK.value


# ---------------------------------------------------------------------------
# 5. Latency Benchmark (< 5.0 ms in Mock)
# ---------------------------------------------------------------------------

def test_joint_decision_mock_latency_benchmark(joint_engine, gater):
    """Verify single-pass joint decision executes in strictly < 5.0 ms in mock engine."""
    text = "Dr. Eleanor Vance analyzed the mineral specimen using an electron microscope, causing the laser to overheat."
    entities = [
        {"id": "E1", "surface_text": "Dr. Eleanor Vance"},
        {"id": "E2", "surface_text": "mineral specimen"},
        {"id": "E3", "surface_text": "electron microscope"},
    ]
    events = [
        {"id": "EV1", "predicate": "analyzed", "subject_ent_id": "E1", "object_ent_id": "E2"},
        {"id": "EV2", "predicate": "overheat", "object_ent_id": "E3"},
    ]

    plan = gater.analyze(entities, events, text)

    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = joint_engine.evaluate_joint_decision(plan, text)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_latency = sum(latencies) / len(latencies)
    assert mean_latency < 5.0, f"Mean latency {mean_latency:.3f} ms exceeds 5.0 ms SLA"


# ---------------------------------------------------------------------------
# 6. Tiered Mode Integration
# ---------------------------------------------------------------------------

def test_tiered_mode_integration(tiered_engine):
    """Verify KevDecisionEngine in mode='tiered' routes via gating and joint decision."""
    # Canonical SVO: should take fast path
    text_fast = "Charles Babbage invented the Difference Engine."
    entities_fast = [
        {"id": "E1", "surface_text": "Charles Babbage"},
        {"id": "E2", "surface_text": "Difference Engine"},
    ]
    events_fast = [
        {"id": "EV1", "predicate": "invented", "subject_ent_id": "E1", "object_ent_id": "E2"},
    ]

    eval_fast = tiered_engine.evaluate_chunk(entities_fast, events_fast, text_fast)
    assert "gated_fastpath" in eval_fast.mode
    assert eval_fast.total_latency_ms < 5.0

    # Ambiguous passage: should execute joint multi-slot pass
    text_ambig = "The suspect allegedly forged the financial certificate."
    entities_ambig = [
        {"id": "E1", "surface_text": "The suspect"},
        {"id": "E2", "surface_text": "financial certificate"},
    ]
    events_ambig = [
        {"id": "EV1", "predicate": "forged", "subject_ent_id": "E1", "object_ent_id": "E2"},
    ]

    eval_ambig = tiered_engine.evaluate_chunk(entities_ambig, events_ambig, text_ambig)
    assert "joint" in eval_ambig.mode
    assert len(eval_ambig.intent_epistemics) == 1
    assert eval_ambig.intent_epistemics[0].epistemic_source in (
        EpistemicSource.HEARSAY.value,
        EpistemicSource.CONJECTURE.value,
    )


# ---------------------------------------------------------------------------
# 7. Dynamic LoRA Mode & VRAM Overhead
# ---------------------------------------------------------------------------

def test_dynamic_lora_joint_mode(joint_engine, lora_joint_engine):
    """Verify VRAM overhead reporting and LoRA calibration."""
    assert joint_engine.vram_overhead_mb == 0.0
    assert lora_joint_engine.vram_overhead_mb == 30.0

    plan = GatedDecisionPlan(
        fast_path_valencies=[],
        fast_path_intents=[],
        ambiguous_valency_candidates=[({"id": "E1", "surface_text": "laser"}, {"id": "EV1", "predicate": "ignited"})],
        ambiguous_events=[{"id": "EV1", "predicate": "ignited"}],
        candidate_relation_pairs=[],
    )

    eval_lora = lora_joint_engine.evaluate_joint_decision(plan, "The laser ignited the fuel.")
    assert eval_lora.vram_overhead_mb == 30.0
    assert "joint_multi_slot_lora_joint" in eval_lora.mode
    # LoRA confidences are calibrated sharper (>= 0.98)
    assert eval_lora.valencies[0].confidence >= 0.98


# ---------------------------------------------------------------------------
# 8. Direct Invocation with Raw Entities & Events
# ---------------------------------------------------------------------------

def test_direct_call_with_raw_entities_and_events(joint_engine):
    """Verify evaluate_joint_decision can be invoked directly without pre-computed plan."""
    text = "Please ensure the cryogenic chamber is sealed."
    entities = [{"id": "E1", "surface_text": "cryogenic chamber"}]
    events = [{"id": "EV1", "predicate": "ensure", "object_ent_id": "E1"}]

    eval_res = joint_engine.evaluate_joint_decision(entities, text, events=events)
    assert isinstance(eval_res, KevChunkEvaluation)
    assert len(eval_res.intent_epistemics) == 1
    assert eval_res.intent_epistemics[0].intent == SpeechActIntent.DIRECTIVE.value
