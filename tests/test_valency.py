"""Tests for strongly-typed valency signatures and TypeConstraintRegistry (Phase 2B).

Verifies:
- Type constraint registry initialization and retrieval for all 11 Band 1 valency slots
- Agent valency constraints: cognitive predicates require sentience; rejection of inanimate/abstract agents
- Experiencer valency constraints: sentient beings required
- Instrument, Location, Temporal, and Patient constraints
- Figurative modality bypass (metaphorical agency)
- Variable binding bypass for formal logic variables
- Custom constraint registration and validation
"""

import pytest
from core.asg import QuantaNode
from core.types import QuantaVector, QuaternaryValue
from core.valency import TypeConstraintRegistry, ValencyConstraint, validate_valency
import core.slots as slots


@pytest.fixture
def registry():
    return TypeConstraintRegistry.get_default()


def test_registry_contains_all_band1_valencies(registry):
    """Verify registry has registered constraints for all Band 1 valency slots."""
    valency_slot_names = [
        "VAL_X1_AGENT",
        "VAL_X2_PATIENT",
        "VAL_X3_DESTINATION",
        "VAL_X4_SOURCE",
        "VAL_X5_INSTRUMENT",
        "VAL_EXPERIENCER",
        "VAL_LOCATION_SLOT",
        "VAL_TIME_SLOT",
        "VAL_MANNER_SLOT",
        "VAL_PURPOSE_SLOT",
        "VAL_RESULT_SLOT",
    ]
    for slot_name in valency_slot_names:
        constraint = registry.get_constraint(slot_name)
        assert constraint is not None, f"Missing constraint for {slot_name}"
        assert constraint.slot_name == slot_name
        assert constraint.slot_index == getattr(slots, slot_name)


def test_agent_valency_cognitive_predicate():
    """Verify cognitive verbs (THINK, KNOW, WANT, FEEL) require sentient agents."""
    think_node = QuantaNode(
        vector={"NSM_THINK": 1, "MODALITY_LITERAL": 1},
        anchor="wn:think.v.01",
    )
    human_node = QuantaNode(
        vector={"TYPE_HUMAN": 1, "TYPE_ANIMATE": 1, "ROLE_SENTIENT": 1, "ROLE_AGENT_CAPABLE": 1},
        anchor="wn:human.n.01",
    )
    dog_node = QuantaNode(
        vector={"TYPE_ANIMATE": 1, "WN_ANIMAL_FAUNA": 1, "ROLE_SENTIENT": 1, "ROLE_AGENT_CAPABLE": 1},
        anchor="wn:dog.n.01",
    )
    rock_node = QuantaNode(
        vector={"TYPE_INANIMATE_PHYSICAL": 1, "WN_OBJECT_NATURAL": 1},
        anchor="wn:rock.n.01",
    )
    concept_node = QuantaNode(
        vector={"TYPE_ABSTRACT_CONCEPT": 1, "WN_COGNITION_THOUGHT": 1},
        anchor="wn:democracy.n.01",
    )

    # Valid agents: human, dog
    assert validate_valency(think_node, "VAL_X1_AGENT", human_node) is True
    assert validate_valency(think_node, "VAL_X1_AGENT", dog_node) is True

    # Invalid agents: rock (inanimate), concept (abstract)
    assert validate_valency(think_node, "VAL_X1_AGENT", rock_node) is False
    assert validate_valency(think_node, "VAL_X1_AGENT", concept_node) is False


def test_agent_valency_speech_act():
    """Verify speech acts (SAY) require communicative agents."""
    say_node = QuantaNode(
        vector={"NSM_SAY": 1, "MODALITY_LITERAL": 1},
        anchor="wn:say.v.01",
    )
    human_node = QuantaNode(
        vector={"TYPE_HUMAN": 1, "ROLE_COMMUNICATOR": 1},
        anchor="wn:human.n.01",
    )
    org_node = QuantaNode(
        vector={"TYPE_ORGANIZATION": 1, "ROLE_AGENT_CAPABLE": 1},
        anchor="wn:organization.n.01",
    )
    tree_node = QuantaNode(
        vector={"TYPE_ANIMATE": 1, "WN_PLANT_FLORA": 1},
        anchor="wn:tree.n.01",
    )

    assert validate_valency(say_node, "VAL_X1_AGENT", human_node) is True
    assert validate_valency(say_node, "VAL_X1_AGENT", org_node) is True
    assert validate_valency(say_node, "VAL_X1_AGENT", tree_node) is False


def test_figurative_modality_bypasses_constraints():
    """Verify figurative modality allows metaphorical agency."""
    think_node = QuantaNode(
        vector={"NSM_THINK": 1, "MODALITY_FIGURATIVE": 1},
        anchor="wn:think.v.01",
    )
    rock_node = QuantaNode(
        vector={"TYPE_INANIMATE_PHYSICAL": 1},
        anchor="wn:rock.n.01",
    )
    # Metaphorical context allows "The rock thinks"
    assert validate_valency(think_node, "VAL_X1_AGENT", rock_node) is True


def test_variable_bindings_bypass_constraints():
    """Verify formal logic variables bypass static compile-time valency."""
    think_node = QuantaNode(
        vector={"NSM_THINK": 1},
        anchor="wn:think.v.01",
    )
    var_x = QuantaNode(
        vector={"GRAPH_VARIABLE_BIND": 1},
        literal="x",
    )
    assert validate_valency(think_node, "VAL_X1_AGENT", var_x) is True


def test_experiencer_valency():
    """Verify experiencer slot strictly requires sentient entities."""
    fear_node = QuantaNode(
        vector={"NSM_FEEL": 1, "WN_FEELING_EMOTION": 1},
        anchor="wn:fear.v.01",
    )
    sentient_node = QuantaNode(
        vector={"ROLE_SENTIENT": 1, "TYPE_HUMAN": 1},
        anchor="wn:person.n.01",
    )
    table_node = QuantaNode(
        vector={"TYPE_ARTIFACT": 1, "TYPE_INANIMATE_PHYSICAL": 1},
        anchor="wn:table.n.01",
    )

    assert validate_valency(fear_node, "VAL_EXPERIENCER", sentient_node) is True
    assert validate_valency(fear_node, "VAL_EXPERIENCER", table_node) is False


def test_instrument_valency():
    """Verify instrument slot accepts usable tools/artifacts and rejects abstract concepts."""
    cut_node = QuantaNode(
        vector={"NSM_DO": 1, "NSM_TOUCH": 1},
        anchor="wn:cut.v.01",
    )
    knife_node = QuantaNode(
        vector={"ROLE_INSTRUMENT_USABLE": 1, "TYPE_ARTIFACT": 1},
        anchor="wn:knife.n.01",
    )
    hand_node = QuantaNode(
        vector={"WN_BODY_PART": 1, "TYPE_ANIMATE": 1},
        anchor="wn:hand.n.01",
    )
    concept_node = QuantaNode(
        vector={"TYPE_ABSTRACT_CONCEPT": 1},
        anchor="wn:justice.n.01",
    )

    assert validate_valency(cut_node, "VAL_X5_INSTRUMENT", knife_node) is True
    assert validate_valency(cut_node, "VAL_X5_INSTRUMENT", hand_node) is True
    assert validate_valency(cut_node, "VAL_X5_INSTRUMENT", concept_node) is False


def test_location_and_temporal_valency():
    """Verify location and time slot valencies."""
    event_node = QuantaNode(
        vector={"TYPE_EVENT": 1, "NSM_DO": 1},
        anchor="wn:party.n.01",
    )
    garden_node = QuantaNode(
        vector={"TYPE_SPATIAL_REGION": 1, "WN_LOCATION_PLACE": 1},
        anchor="wn:garden.n.01",
    )
    noon_node = QuantaNode(
        vector={"TYPE_TEMPORAL_INTERVAL": 1, "NSM_MOMENT": 1},
        anchor="wn:noon.n.01",
    )
    person_node = QuantaNode(
        vector={"TYPE_HUMAN": 1},
        anchor="wn:person.n.01",
    )

    # Location
    assert validate_valency(event_node, "VAL_LOCATION_SLOT", garden_node) is True
    assert validate_valency(event_node, "VAL_LOCATION_SLOT", person_node) is False

    # Time
    assert validate_valency(event_node, "VAL_TIME_SLOT", noon_node) is True
    assert validate_valency(event_node, "VAL_TIME_SLOT", person_node) is False


def test_patient_physical_contact_valency():
    """Verify physical touch requires physical/animate patient."""
    touch_node = QuantaNode(
        vector={"NSM_TOUCH": 1, "MODALITY_LITERAL": 1},
        anchor="wn:touch.v.01",
    )
    apple_node = QuantaNode(
        vector={"TYPE_INANIMATE_PHYSICAL": 1, "WN_FOOD_NUTRITION": 1},
        anchor="wn:apple.n.01",
    )
    idea_node = QuantaNode(
        vector={"TYPE_ABSTRACT_CONCEPT": 1},
        anchor="wn:idea.n.01",
    )

    assert validate_valency(touch_node, "VAL_X2_PATIENT", apple_node) is True
    assert validate_valency(touch_node, "VAL_X2_PATIENT", idea_node) is False
