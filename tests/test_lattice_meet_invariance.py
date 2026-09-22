"""Comprehensive Test Suite for Section 3: Closed-Loop Round-Trip Lattice Meet Gate & Deep NSM Explication (exp-005a).

Verifies:
1. Belnap 4-valued lattice meet algebra and epistemic contradiction detection (Task 3.1).
2. Multi-hop NSM explication script expansion in ASGCompiler for buy, prohibit, transform, replicate (Task 3.2).
3. Discourse-tracking referring expression generation with anaphoric recency, topic-shift handling, and determiner preservation (Task 3.3).
4. Full round-trip lattice meet invariance across stress narratives (Task 3.4).
"""

import pytest
from core.asg import QuantaGraph, QuantaNode
from core.slots import get_slot_by_name, get_slot_by_index
from core.types import QuantaVector, QuaternaryValue
from parser.asg_compiler import ASGCompiler
from parser.schema import ExtractedEntity, ExtractedEvent
from pipeline.translator_pipeline import TwoWayTranslationPipeline
from realizer.english_nlg import EnglishRealizer, ReferringExpressionGenerator
from verification.lattice_gate import (
    ContradictionDetail,
    LatticeInvarianceGate,
    LatticeMeetResult,
)


@pytest.fixture(scope="module")
def pipeline():
    return TwoWayTranslationPipeline()


@pytest.fixture(scope="module")
def compiler():
    return ASGCompiler()


@pytest.fixture(scope="module")
def lattice_gate():
    return LatticeInvarianceGate(default_min_preservation=0.40)


# ==============================================================================
# 1. Unit Tests: Belnap Lattice Meet & Epistemic Contradictions (Task 3.1)
# ==============================================================================

def test_belnap_lattice_meet_algebra(lattice_gate):
    """Verify Belnap four-valued knowledge meet operations:

    1 ⊓ 1 = 1 (TRUE ⊓ TRUE = TRUE)
    2 ⊓ 2 = 2 (FALSE ⊓ FALSE = FALSE)
    1 ⊓ 2 = 0 (TRUE ⊓ FALSE = IRRELEVANT / information destroyed by conflict)
    3 ⊓ 1 = 1 (UNKNOWN ⊓ TRUE = TRUE)
    3 ⊓ 2 = 2 (UNKNOWN ⊓ FALSE = FALSE)
    0 ⊓ x = 0 (IRRELEVANT ⊓ x = IRRELEVANT)
    """
    q0 = QuaternaryValue.IRRELEVANT  # 0
    q1 = QuaternaryValue.TRUE        # 1
    q2 = QuaternaryValue.FALSE       # 2
    q3 = QuaternaryValue.UNKNOWN     # 3

    # Direct meet checks
    assert q1.meet(q1) == q1
    assert q2.meet(q2) == q2
    assert q1.meet(q2) == q0  # Conflict wipes out common information
    assert q2.meet(q1) == q0
    assert q3.meet(q1) == q1
    assert q3.meet(q2) == q2
    assert q0.meet(q1) == q0
    assert q0.meet(q2) == q0

    # QuantaVector meet checks
    v1 = QuantaVector([1] * 1024)
    v2 = QuantaVector([1] * 1024)
    v_meet = lattice_gate.compute_lattice_meet(v1, v2)
    assert v_meet == v1

    # Conflicting vectors
    v_true = QuantaVector([1] * 1024)
    v_false = QuantaVector([2] * 1024)
    v_clash = lattice_gate.compute_lattice_meet(v_true, v_false)
    # Band 0 and Bands 3..7 are Epistemic: 1 ⊓ 2 = 0
    assert int(v_clash[0]) == 0
    assert int(v_clash[500]) == 0


def test_epistemic_contradiction_detection(lattice_gate):
    """Verify LatticeInvarianceGate flags direct TRUE vs FALSE polarity clashes."""
    v1 = QuantaVector.zeros()
    v2 = QuantaVector.zeros()

    # Set slot 23 (NSM_TRUE) = 1 in v1, but 2 (FALSE) in v2
    v1["NSM_TRUE"] = 1
    v2["NSM_TRUE"] = 2

    # Set slot 802 (EPIST_DEONTIC_PROHIBITION) = 1 in both (agreement)
    v1["EPIST_DEONTIC_PROHIBITION"] = 1
    v2["EPIST_DEONTIC_PROHIBITION"] = 1

    contradictions = lattice_gate.detect_epistemic_contradictions(v1, v2)
    assert len(contradictions) == 1
    c = contradictions[0]
    assert c.slot_name == "NSM_TRUE"
    assert c.orig_value == 1
    assert c.reparsed_value == 2
    assert "Epistemic contradiction" in c.description

    # Audit should report failure due to contradiction
    audit = lattice_gate.audit_round_trip(v1, v2)
    assert not audit.is_sound
    assert audit.contradiction_count == 1
    assert len(audit.errors) >= 1


def test_slot_preservation_rate_under_meet(lattice_gate):
    """Verify calculation of slot preservation rate under lattice meet."""
    v1 = QuantaVector.zeros()
    v2 = QuantaVector.zeros()

    # Activate 10 slots in v1
    slots = ["NSM_DO", "NSM_HAVE", "NSM_PART", "NSM_SAY", "NSM_SEE",
             "NSM_THINK", "NSM_KNOW", "NSM_TRUE", "NSM_GOOD", "NSM_BIG"]
    for s in slots:
        v1[s] = 1

    # v2 agrees on 8 of 10 slots, drops 2
    for s in slots[:8]:
        v2[s] = 1

    rate = lattice_gate.compute_slot_preservation_rate(v1, v2, use_meet=True)
    assert rate == 0.80

    # If v2 agrees on all 10, rate = 1.0
    for s in slots:
        v2[s] = 1
    rate_full = lattice_gate.compute_slot_preservation_rate(v1, v2, use_meet=True)
    assert rate_full == 1.0


# ==============================================================================
# 2. Unit Tests: Multi-Hop NSM Explication Script Expansion (Task 3.2)
# ==============================================================================

def test_nsm_explication_buy_purchase(compiler):
    """Verify 'buy' / 'purchase' expands into: NSM_DO ⊓ NSM_HAVE ⊓ NSM_PART."""
    ev_buy = compiler.compile_event(
        ExtractedEvent(id="ev_buy", predicate="buy", arguments={"agent": "Alice", "patient": "book"})
    )
    assert ev_buy.get_slot("NSM_DO") == 1
    assert ev_buy.get_slot("NSM_HAVE") == 1
    assert ev_buy.get_slot("NSM_PART") == 1

    ev_purchase = compiler.compile_event(
        ExtractedEvent(id="ev_purch", predicate="purchase", arguments={"agent": "Bob", "patient": "car"})
    )
    assert ev_purchase.get_slot("NSM_DO") == 1
    assert ev_purchase.get_slot("NSM_HAVE") == 1
    assert ev_purchase.get_slot("NSM_PART") == 1


def test_nsm_explication_prohibit_forbid(compiler):
    """Verify 'prohibit' / 'forbid' expands into: NSM_SAY ⊓ DEONTIC_MUSTNOT_PROHIBITED ⊓ CAUSAL_PREVENTIVE_BLOCK."""
    ev_proh = compiler.compile_event(
        ExtractedEvent(id="ev_proh", predicate="prohibit", arguments={"agent": "director"})
    )
    assert ev_proh.get_slot("NSM_SAY") == 1
    assert ev_proh.get_slot("NSM_DO") == 1
    assert ev_proh.get_slot("EPIST_DEONTIC_PROHIBITION") == 1
    assert ev_proh.get_slot("DEONTIC_MUSTNOT_PROHIBITED") == 1  # Alias verification
    assert ev_proh.get_slot("CAUSAL_PREVENTIVE_BLOCK") == 1

    ev_forbid = compiler.compile_event(
        ExtractedEvent(id="ev_forb", predicate="forbid", arguments={"agent": "council"})
    )
    assert ev_forbid.get_slot("NSM_SAY") == 1
    assert ev_forbid.get_slot("EPIST_DEONTIC_PROHIBITION") == 1
    assert ev_forbid.get_slot("CAUSAL_PREVENTIVE_BLOCK") == 1


def test_nsm_explication_transform_transmute(compiler):
    """Verify 'transform' / 'transmute' expands into: NSM_DO ⊓ NSM_HAPPEN ⊓ PHYS_ENTROPY_DELTA_S."""
    ev_trans = compiler.compile_event(
        ExtractedEvent(id="ev_trans", predicate="transform", arguments={"patient": "specimen"})
    )
    assert ev_trans.get_slot("NSM_DO") == 1
    assert ev_trans.get_slot("NSM_HAPPEN") == 1
    assert ev_trans.get_slot("PHYS_ENTROPY_THERMAL") == 1
    assert ev_trans.get_slot("PHYS_ENTROPY_DELTA_S") == 1  # Alias verification

    ev_transmute = compiler.compile_event(
        ExtractedEvent(id="ev_tm", predicate="transmute", arguments={"patient": "compound"})
    )
    assert ev_transmute.get_slot("NSM_DO") == 1
    assert ev_transmute.get_slot("NSM_HAPPEN") == 1
    assert ev_transmute.get_slot("PHYS_ENTROPY_THERMAL") == 1


def test_nsm_explication_replicate(compiler):
    """Verify 'replicate' expands into: NSM_DO ⊓ NSM_SAME ⊓ SOLVER_PROOF_VALIDATED."""
    ev_rep = compiler.compile_event(
        ExtractedEvent(id="ev_rep", predicate="replicate", arguments={"agent": "scientist", "patient": "experiment"})
    )
    assert ev_rep.get_slot("NSM_DO") == 1
    assert ev_rep.get_slot("NSM_SAME") == 1
    assert ev_rep.get_slot("SOLVER_PROOF_VALIDATED") == 1


# ==============================================================================
# 3. Unit Tests: Enhanced Referring Expression Generation (Task 3.3)
# ==============================================================================

def test_referring_expression_anaphora_and_topic_shift():
    """Verify ReferringExpressionGenerator discourse recency, topic shifts, and ambiguity."""
    ref = ReferringExpressionGenerator()

    # Entity 1: Dr. Eleanor Vance (female)
    node_vance = QuantaNode(literal="Dr. Eleanor Vance")
    node_vance.set_slot("TYPE_HUMAN", 1)

    # Entity 2: Dr. Marcus Vance (male)
    node_marcus = QuantaNode(literal="Dr. Marcus Vance")
    node_marcus.set_slot("TYPE_HUMAN", 1)

    # 1. First mention: Full proper noun
    s1 = ref.realize_reference(node_vance, role="subject")
    assert s1 == "Dr. Eleanor Vance"

    # 2. Immediate subsequent mention (count == 1, same subject): pronoun 'she'
    s2 = ref.realize_reference(node_vance, role="subject")
    assert s2 == "she"

    # 3. Topic shift: another entity becomes subject
    s_other = ref.realize_reference(node_marcus, role="subject")
    assert s_other == "Dr. Marcus Vance"

    # 4. Return to Eleanor after intervening subject: re-introduce proper noun ("Eleanor")
    s3 = ref.realize_reference(node_vance, role="subject")
    assert s3 == "Eleanor"


def test_referring_expression_competing_gender_ambiguity():
    """Verify ReferringExpressionGenerator resolves ambiguous pronouns between two females."""
    ref = ReferringExpressionGenerator()

    alice = QuantaNode(literal="Alice Smith")
    alice.set_slot("TYPE_HUMAN", 1)

    carol = QuantaNode(literal="Carol White")
    carol.set_slot("TYPE_HUMAN", 1)

    # Alice first mention
    assert ref.realize_reference(alice, role="subject") == "Alice Smith"

    # Carol first mention
    assert ref.realize_reference(carol, role="subject") == "Carol White"

    # Referring to Alice again: another female has intervened, so 'she' is ambiguous -> emits "Alice"
    assert ref.realize_reference(alice, role="subject") == "Alice"


# ==============================================================================
# 4. Full Round-Trip Invariance Tests Across 5 Stress Narratives (Task 3.4)
# ==============================================================================

def test_stress_narratives_zero_epistemic_contradictions(pipeline, lattice_gate):
    """Verify all 5 stress narratives achieve zero epistemic contradictions under lattice meet."""
    narratives = [
        # Narrative 1: Counterfactual ToM & Sarcasm
        "Had Alice not falsely pretended to know that Bob believed her investment was secure, the auditor wouldn't have sarcastically remarked that her due diligence was a stroke of genius.",
        # Narrative 2: Kinematics & Mereotopology
        "While the drone was accelerating into the restricted airspace before dusk, the operator plausibly suspected, but could not deduce with certainty, that the left wingtip was tangentially touching the perimeter wire.",
        # Narrative 3: Quantifier Scope & Modal Logic
        "Every investigator who doubted that any suspect had necessarily committed every crime secretly wanted someone to prove the absolute impossibility of an accomplice's alibi.",
        # Narrative 4: Metalogical Self-Reference
        "By declaring this very decree to be legally void, the council obligated the commissioner to prevent its future enforcement unless the clause could recursively validate its own origin.",
        # Narrative 5: Multi-sentence Scientific Discourse Paragraph
        (
            "Dr. Eleanor Vance isolated a volatile synthetic compound inside the cryogenic containment cell at dawn. "
            "She immediately noted that this specimen exhibited anomalous crystalline lattice expansion, which strongly suggested an unobserved phase transition. "
            "Although her supervisor initially doubted the validity of the discovery, Eleanor verified the hypothesis three hours later by replicating the transformation within the same vessel. "
            "The resulting polymer retained its structural integrity throughout the afternoon, prompting the laboratory director to prohibit all competing tests until her synthesis protocol could be formally audited."
        ),
    ]

    for idx, text in enumerate(narratives, start=1):
        rt = pipeline.round_trip(text, modality="english")
        assert rt.validation_pass, f"Narrative {idx} validation failed: {rt.muc_errors}"

        # Audit with LatticeInvarianceGate
        audit = lattice_gate.audit_round_trip(rt.original_vector, rt.reparsed_vector)
        assert audit.contradiction_count == 0, (
            f"Narrative {idx} produced epistemic contradictions: {audit.errors}"
        )
        assert rt.is_meet_sound, f"Narrative {idx} is_meet_sound should be True"


def test_scientific_narrative_slot_preservation_threshold(pipeline, lattice_gate):
    """Verify that Narrative 5 (Scientific Narrative Paragraph) meets the >= 95% slot preservation rate."""
    scientific_paragraph = (
        "Dr. Eleanor Vance isolated a volatile synthetic compound inside the cryogenic containment cell at dawn. "
        "She immediately noted that this specimen exhibited anomalous crystalline lattice expansion, which strongly suggested an unobserved phase transition. "
        "Although her supervisor initially doubted the validity of the discovery, Eleanor verified the hypothesis three hours later by replicating the transformation within the same vessel. "
        "The resulting polymer retained its structural integrity throughout the afternoon, prompting the laboratory director to prohibit all competing tests until her synthesis protocol could be formally audited."
    )

    rt = pipeline.round_trip(scientific_paragraph, modality="english")
    assert rt.validation_pass

    # Audit meet preservation
    preservation = lattice_gate.compute_slot_preservation_rate(rt.original_vector, rt.reparsed_vector, use_meet=True)
    assert preservation >= 0.95, f"Expected preservation >= 0.95, got {preservation:.4f}"
    assert rt.slot_preservation_rate >= 0.95, f"Expected rt.slot_preservation_rate >= 0.95, got {rt.slot_preservation_rate:.4f}"

    # Verify key proper nouns and determiners
    out = rt.realized_output
    assert "Dr. Vance" in out or "Vance" in out or "Eleanor" in out
    assert "compound" in out.lower()
    assert "specimen" in out.lower()
    assert "polymer" in out.lower()
