"""Strongly-typed valency signatures and ontological type constraint registry for QUANTA.

Enforces compile-time semantic typing on predicate argument slots, rendering category errors
(e.g., "The rock thinks") syntactically illegal before neural processing.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np

from core.types import (
    QuantaVector,
    QuaternaryValue,
    EpistemicValue,
    StructuralValue,
    RoutingValue,
    RegisterValue,
    BandContract,
)
from core.asg import QuantaNode
from core.slots import (
    # Band 0 Primes
    NSM_THINK,
    NSM_KNOW,
    NSM_WANT,
    NSM_FEEL,
    NSM_SEE,
    NSM_HEAR,
    NSM_SAY,
    NSM_DO,
    NSM_MOVE,
    NSM_TOUCH,
    NSM_LIVE,
    NSM_DIE,
    NSM_NOW,
    NSM_BEFORE,
    NSM_AFTER,
    NSM_MOMENT,
    NSM_FOR_SOME_TIME,
    # Band 1 Valencies
    VAL_X1_AGENT,
    VAL_X2_PATIENT,
    VAL_X3_DESTINATION,
    VAL_X4_SOURCE,
    VAL_X5_INSTRUMENT,
    VAL_EXPERIENCER,
    VAL_LOCATION_SLOT,
    VAL_TIME_SLOT,
    VAL_MANNER_SLOT,
    VAL_PURPOSE_SLOT,
    VAL_RESULT_SLOT,
    # Band 1 Topology
    GRAPH_VARIABLE_BIND,
    # Band 2 Entity Types
    TYPE_ANIMATE,
    TYPE_HUMAN,
    TYPE_INANIMATE_PHYSICAL,
    TYPE_NATURAL_OBJECT,
    TYPE_ARTIFACT,
    TYPE_SUBSTANCE_MASS,
    TYPE_COLLECTION_SET,
    TYPE_ABSTRACT_CONCEPT,
    TYPE_PROPOSITION,
    TYPE_EVENT,
    TYPE_STATE,
    TYPE_PROCESS,
    TYPE_TEMPORAL_INTERVAL,
    TYPE_SPATIAL_REGION,
    TYPE_MEASURE_SCALAR,
    TYPE_NUMERIC_VALUE,
    TYPE_ORGANIZATION,
    TYPE_COMMUNICATION_MSG,
    TYPE_ATTRIBUTE_PROPERTY,
    TYPE_RELATION_ROLE,
    # Band 2 Capabilities & Roles
    ROLE_AGENT_CAPABLE,
    ROLE_SENTIENT,
    ROLE_MOVEABLE,
    ROLE_COMMUNICATOR,
    ROLE_CONSUMABLE,
    ROLE_CONTAINER,
    ROLE_INSTRUMENT_USABLE,
    ROLE_VOLITIONAL_SOURCE,
    ROLE_COGNITIVE_SUBJECT,
    ROLE_AFFECTIVE_TARGET,
    ROLE_EPISTEMIC_AUTHORITY,
    ROLE_PATIENT_TARGET,
    # Band 2 Modalities
    MODALITY_LITERAL,
    MODALITY_FIGURATIVE,
    MODALITY_HYPOTHETICAL,
    MODALITY_COUNTERFACTUAL,
    # Band 2 WordNet Roots
    WN_ACT_ACTION,
    WN_ANIMAL_FAUNA,
    WN_ARTIFACT_OBJECT,
    WN_ATTRIBUTE_PROP,
    WN_BODY_PART,
    WN_COGNITION_THOUGHT,
    WN_COMMUNICATION_INFO,
    WN_EVENT_OCCURRENCE,
    WN_FEELING_EMOTION,
    WN_FOOD_NUTRITION,
    WN_GROUP_SOCIAL,
    WN_LOCATION_PLACE,
    WN_MOTIVE_REASON,
    WN_OBJECT_NATURAL,
    WN_PERSON_HUMAN,
    WN_PHENOMENON_NATURE,
    WN_PLANT_FLORA,
    WN_POSSESSION_ASSET,
    WN_PROCESS_SERIES,
    WN_QUANTITY_NUMBER,
    WN_RELATION_LINK,
    get_slot_by_name,
    get_slot_by_index,
)


def _to_vector(item: Union[QuantaNode, QuantaVector, Dict[str, Any]]) -> QuantaVector:
    """Normalizes QuantaNode, QuantaVector, or dict into a QuantaVector."""
    if isinstance(item, QuantaNode):
        return item.vector
    elif isinstance(item, QuantaVector):
        return item
    elif isinstance(item, dict):
        return QuantaVector(item)
    raise TypeError(f"Cannot convert {type(item)} to QuantaVector")


def _to_slot_idx(slot: Union[str, int]) -> int:
    """Normalizes slot name or index to an integer index."""
    if isinstance(slot, str):
        s_def = get_slot_by_name(slot)
        if s_def is None:
            raise KeyError(f"Unknown slot name: {slot}")
        return s_def.index
    return int(slot)


@dataclass
class ValencyConstraint:
    """Type constraint specification for a predicate argument slot."""
    slot_index: int
    slot_name: str
    allowed_types: List[int] = field(default_factory=list)
    prohibited_types: List[int] = field(default_factory=list)
    description: str = ""
    custom_checker: Optional[Callable[[QuantaVector, QuantaVector], bool]] = None


class TypeConstraintRegistry:
    """Declarative registry mapping predicate argument valencies to ontological requirements."""

    _default_instance: Optional[TypeConstraintRegistry] = None

    def __init__(self):
        self._constraints: Dict[int, ValencyConstraint] = {}
        self._init_default_constraints()

    @classmethod
    def get_default(cls) -> TypeConstraintRegistry:
        if cls._default_instance is None:
            cls._default_instance = cls()
        return cls._default_instance

    def register_constraint(self, constraint: ValencyConstraint):
        self._constraints[constraint.slot_index] = constraint

    def get_constraint(self, slot: Union[str, int]) -> Optional[ValencyConstraint]:
        idx = _to_slot_idx(slot)
        return self._constraints.get(idx)

    def _init_default_constraints(self):
        # 1. VAL_X1_AGENT (Actor / Initiator)
        def _check_agent(p_vec: QuantaVector, c_vec: QuantaVector) -> bool:
            # If predicate is a mental/cognitive action (THINK, KNOW, WANT, FEEL), agent MUST be sentient
            is_mental = (
                p_vec[NSM_THINK] == QuaternaryValue.TRUE
                or p_vec[NSM_KNOW] == QuaternaryValue.TRUE
                or p_vec[NSM_WANT] == QuaternaryValue.TRUE
                or p_vec[NSM_FEEL] == QuaternaryValue.TRUE
            )
            if is_mental:
                is_sentient = (
                    c_vec[ROLE_SENTIENT] == QuaternaryValue.TRUE
                    or c_vec[TYPE_HUMAN] == QuaternaryValue.TRUE
                    or c_vec[TYPE_ANIMATE] == QuaternaryValue.TRUE
                    or c_vec[WN_ANIMAL_FAUNA] == QuaternaryValue.TRUE
                    or c_vec[WN_PERSON_HUMAN] == QuaternaryValue.TRUE
                )
                if not is_sentient:
                    return False

            # If speech act (SAY), agent must be communicator/sentient
            if p_vec[NSM_SAY] == QuaternaryValue.TRUE:
                is_speaker = (
                    c_vec[ROLE_COMMUNICATOR] == QuaternaryValue.TRUE
                    or c_vec[TYPE_HUMAN] == QuaternaryValue.TRUE
                    or c_vec[TYPE_ORGANIZATION] == QuaternaryValue.TRUE
                    or c_vec[WN_PERSON_HUMAN] == QuaternaryValue.TRUE
                )
                if not is_speaker:
                    return False

            # General Agent check: Must be agent-capable, animate, human, organization, or person
            is_agent_capable = (
                c_vec[ROLE_AGENT_CAPABLE] == QuaternaryValue.TRUE
                or c_vec[TYPE_HUMAN] == QuaternaryValue.TRUE
                or c_vec[TYPE_ANIMATE] == QuaternaryValue.TRUE
                or c_vec[TYPE_ORGANIZATION] == QuaternaryValue.TRUE
                or c_vec[WN_ANIMAL_FAUNA] == QuaternaryValue.TRUE
                or c_vec[WN_PERSON_HUMAN] == QuaternaryValue.TRUE
                or c_vec[ROLE_VOLITIONAL_SOURCE] == QuaternaryValue.TRUE
            )
            if not is_agent_capable:
                if (
                    c_vec[TYPE_INANIMATE_PHYSICAL] == QuaternaryValue.TRUE
                    or c_vec[TYPE_ABSTRACT_CONCEPT] == QuaternaryValue.TRUE
                    or c_vec[TYPE_NATURAL_OBJECT] == QuaternaryValue.TRUE
                ):
                    return False
            return True

        self.register_constraint(
            ValencyConstraint(
                slot_index=VAL_X1_AGENT,
                slot_name="VAL_X1_AGENT",
                allowed_types=[
                    ROLE_AGENT_CAPABLE,
                    TYPE_HUMAN,
                    TYPE_ANIMATE,
                    TYPE_ORGANIZATION,
                    WN_PERSON_HUMAN,
                    WN_ANIMAL_FAUNA,
                    ROLE_VOLITIONAL_SOURCE,
                ],
                prohibited_types=[
                    TYPE_INANIMATE_PHYSICAL,
                    TYPE_ABSTRACT_CONCEPT,
                    TYPE_NATURAL_OBJECT,
                ],
                description="Agent initiator requires agent capability or animacy; rejects inanimate physical/abstract concepts",
                custom_checker=_check_agent,
            )
        )

        # 2. VAL_EXPERIENCER (Sentient experiencer)
        def _check_experiencer(p_vec: QuantaVector, c_vec: QuantaVector) -> bool:
            return bool(
                c_vec[ROLE_SENTIENT] == QuaternaryValue.TRUE
                or c_vec[TYPE_HUMAN] == QuaternaryValue.TRUE
                or c_vec[TYPE_ANIMATE] == QuaternaryValue.TRUE
                or c_vec[WN_PERSON_HUMAN] == QuaternaryValue.TRUE
                or c_vec[WN_ANIMAL_FAUNA] == QuaternaryValue.TRUE
                or c_vec[ROLE_COGNITIVE_SUBJECT] == QuaternaryValue.TRUE
                or c_vec[ROLE_AFFECTIVE_TARGET] == QuaternaryValue.TRUE
            )

        self.register_constraint(
            ValencyConstraint(
                slot_index=VAL_EXPERIENCER,
                slot_name="VAL_EXPERIENCER",
                allowed_types=[
                    ROLE_SENTIENT,
                    TYPE_HUMAN,
                    TYPE_ANIMATE,
                    WN_PERSON_HUMAN,
                    WN_ANIMAL_FAUNA,
                ],
                prohibited_types=[
                    TYPE_INANIMATE_PHYSICAL,
                    TYPE_ABSTRACT_CONCEPT,
                    TYPE_ARTIFACT,
                    TYPE_NATURAL_OBJECT,
                ],
                description="Experiencer requires sentient entity (human, animal, or sentient subject)",
                custom_checker=_check_experiencer,
            )
        )

        # 3. VAL_TIME_SLOT (Temporal anchor)
        def _check_time(p_vec: QuantaVector, c_vec: QuantaVector) -> bool:
            is_time = (
                c_vec[TYPE_TEMPORAL_INTERVAL] == QuaternaryValue.TRUE
                or c_vec[TYPE_EVENT] == QuaternaryValue.TRUE
                or c_vec[NSM_NOW] == QuaternaryValue.TRUE
                or c_vec[NSM_BEFORE] == QuaternaryValue.TRUE
                or c_vec[NSM_AFTER] == QuaternaryValue.TRUE
                or c_vec[NSM_MOMENT] == QuaternaryValue.TRUE
                or c_vec[NSM_FOR_SOME_TIME] == QuaternaryValue.TRUE
            )
            if not is_time and (
                c_vec[TYPE_ANIMATE] == QuaternaryValue.TRUE
                or c_vec[TYPE_HUMAN] == QuaternaryValue.TRUE
                or c_vec[TYPE_INANIMATE_PHYSICAL] == QuaternaryValue.TRUE
            ):
                return False
            return True

        self.register_constraint(
            ValencyConstraint(
                slot_index=VAL_TIME_SLOT,
                slot_name="VAL_TIME_SLOT",
                allowed_types=[
                    TYPE_TEMPORAL_INTERVAL,
                    TYPE_EVENT,
                    NSM_NOW,
                    NSM_BEFORE,
                    NSM_AFTER,
                    NSM_MOMENT,
                    NSM_FOR_SOME_TIME,
                ],
                prohibited_types=[
                    TYPE_ANIMATE,
                    TYPE_HUMAN,
                ],
                description="Time slot requires temporal interval, point moment, or event anchor",
                custom_checker=_check_time,
            )
        )

        # 4. VAL_LOCATION_SLOT (Spatial location)
        def _check_location(p_vec: QuantaVector, c_vec: QuantaVector) -> bool:
            is_loc = (
                c_vec[TYPE_SPATIAL_REGION] == QuaternaryValue.TRUE
                or c_vec[WN_LOCATION_PLACE] == QuaternaryValue.TRUE
                or c_vec[ROLE_CONTAINER] == QuaternaryValue.TRUE
                or c_vec[TYPE_ARTIFACT] == QuaternaryValue.TRUE
                or c_vec[TYPE_NATURAL_OBJECT] == QuaternaryValue.TRUE
            )
            if not is_loc and c_vec[TYPE_HUMAN] == QuaternaryValue.TRUE:
                return False
            return True

        self.register_constraint(
            ValencyConstraint(
                slot_index=VAL_LOCATION_SLOT,
                slot_name="VAL_LOCATION_SLOT",
                allowed_types=[
                    TYPE_SPATIAL_REGION,
                    WN_LOCATION_PLACE,
                    ROLE_CONTAINER,
                ],
                prohibited_types=[
                    TYPE_HUMAN,
                ],
                description="Location slot requires spatial region, place, container, or geographic entity",
                custom_checker=_check_location,
            )
        )

        # 5. VAL_X5_INSTRUMENT (Instrument / Medium)
        def _check_instrument(p_vec: QuantaVector, c_vec: QuantaVector) -> bool:
            is_instrument = (
                c_vec[ROLE_INSTRUMENT_USABLE] == QuaternaryValue.TRUE
                or c_vec[TYPE_ARTIFACT] == QuaternaryValue.TRUE
                or c_vec[WN_ARTIFACT_OBJECT] == QuaternaryValue.TRUE
                or c_vec[WN_BODY_PART] == QuaternaryValue.TRUE
                or c_vec[TYPE_INANIMATE_PHYSICAL] == QuaternaryValue.TRUE
                or c_vec[TYPE_NATURAL_OBJECT] == QuaternaryValue.TRUE
                or c_vec[TYPE_SUBSTANCE_MASS] == QuaternaryValue.TRUE
            )
            if not is_instrument and c_vec[TYPE_ABSTRACT_CONCEPT] == QuaternaryValue.TRUE:
                return False
            return True

        self.register_constraint(
            ValencyConstraint(
                slot_index=VAL_X5_INSTRUMENT,
                slot_name="VAL_X5_INSTRUMENT",
                allowed_types=[
                    ROLE_INSTRUMENT_USABLE,
                    TYPE_ARTIFACT,
                    WN_ARTIFACT_OBJECT,
                    WN_BODY_PART,
                    TYPE_INANIMATE_PHYSICAL,
                ],
                prohibited_types=[
                    TYPE_ABSTRACT_CONCEPT,
                ],
                description="Instrument requires usable tool, artifact, body part, or physical medium",
                custom_checker=_check_instrument,
            )
        )

        # 6. VAL_X3_DESTINATION (Goal / Recipient / Destination)
        self.register_constraint(
            ValencyConstraint(
                slot_index=VAL_X3_DESTINATION,
                slot_name="VAL_X3_DESTINATION",
                allowed_types=[
                    TYPE_SPATIAL_REGION,
                    WN_LOCATION_PLACE,
                    TYPE_HUMAN,
                    TYPE_ANIMATE,
                    TYPE_ORGANIZATION,
                    ROLE_SENTIENT,
                ],
                description="Destination requires spatial region, location, or recipient agent",
            )
        )

        # 7. VAL_X4_SOURCE (Origin / Source)
        self.register_constraint(
            ValencyConstraint(
                slot_index=VAL_X4_SOURCE,
                slot_name="VAL_X4_SOURCE",
                allowed_types=[
                    TYPE_SPATIAL_REGION,
                    WN_LOCATION_PLACE,
                    TYPE_HUMAN,
                    TYPE_ANIMATE,
                    TYPE_ORGANIZATION,
                    TYPE_INANIMATE_PHYSICAL,
                ],
                description="Source requires origin place, physical source, or origin agent",
            )
        )

        # 8. VAL_X2_PATIENT (Patient / Theme)
        def _check_patient(p_vec: QuantaVector, c_vec: QuantaVector) -> bool:
            # Physical touch / impact verbs require physical entity or animate target
            if p_vec[NSM_TOUCH] == QuaternaryValue.TRUE:
                is_phys = (
                    c_vec[TYPE_ANIMATE] == QuaternaryValue.TRUE
                    or c_vec[TYPE_HUMAN] == QuaternaryValue.TRUE
                    or c_vec[TYPE_INANIMATE_PHYSICAL] == QuaternaryValue.TRUE
                    or c_vec[TYPE_ARTIFACT] == QuaternaryValue.TRUE
                    or c_vec[TYPE_NATURAL_OBJECT] == QuaternaryValue.TRUE
                    or c_vec[WN_ANIMAL_FAUNA] == QuaternaryValue.TRUE
                    or c_vec[WN_BODY_PART] == QuaternaryValue.TRUE
                )
                if not is_phys and c_vec[TYPE_ABSTRACT_CONCEPT] == QuaternaryValue.TRUE:
                    return False
            return True

        self.register_constraint(
            ValencyConstraint(
                slot_index=VAL_X2_PATIENT,
                slot_name="VAL_X2_PATIENT",
                allowed_types=[
                    TYPE_ANIMATE,
                    TYPE_HUMAN,
                    TYPE_INANIMATE_PHYSICAL,
                    TYPE_ARTIFACT,
                    TYPE_NATURAL_OBJECT,
                    TYPE_SUBSTANCE_MASS,
                    TYPE_ABSTRACT_CONCEPT,
                    TYPE_PROPOSITION,
                    TYPE_EVENT,
                ],
                description="Patient/theme undergoing event or cognitive focus",
                custom_checker=_check_patient,
            )
        )

        # 9. VAL_MANNER_SLOT (Manner / Qualitative modifier)
        self.register_constraint(
            ValencyConstraint(
                slot_index=VAL_MANNER_SLOT,
                slot_name="VAL_MANNER_SLOT",
                allowed_types=[
                    TYPE_ATTRIBUTE_PROPERTY,
                    WN_ATTRIBUTE_PROP,
                    TYPE_ABSTRACT_CONCEPT,
                ],
                description="Manner modifier requires qualitative attribute or property",
            )
        )

        # 10. VAL_PURPOSE_SLOT (Teleological purpose)
        self.register_constraint(
            ValencyConstraint(
                slot_index=VAL_PURPOSE_SLOT,
                slot_name="VAL_PURPOSE_SLOT",
                allowed_types=[
                    TYPE_EVENT,
                    TYPE_STATE,
                    TYPE_PROCESS,
                    TYPE_PROPOSITION,
                    TYPE_ABSTRACT_CONCEPT,
                    WN_MOTIVE_REASON,
                ],
                description="Purpose slot requires event, motive, or propositional goal",
            )
        )

        # 11. VAL_RESULT_SLOT (Output / Result state)
        self.register_constraint(
            ValencyConstraint(
                slot_index=VAL_RESULT_SLOT,
                slot_name="VAL_RESULT_SLOT",
                allowed_types=[
                    TYPE_EVENT,
                    TYPE_STATE,
                    TYPE_PROCESS,
                    TYPE_PROPOSITION,
                    TYPE_ATTRIBUTE_PROPERTY,
                ],
                description="Result slot requires resulting state, event, or attribute",
            )
        )

    def validate(
        self,
        parent_node_or_vector: Union[QuantaNode, QuantaVector, Dict[str, Any]],
        relation_slot: Union[str, int],
        child_node_or_vector: Union[QuantaNode, QuantaVector, Dict[str, Any]],
    ) -> bool:
        """Validates whether child satisfies the ontological valency constraints of the slot."""
        p_vec = _to_vector(parent_node_or_vector)
        c_vec = _to_vector(child_node_or_vector)

        # 1. Figurative modality bypass: Metaphors suspend physical domain constraints
        if (
            p_vec[MODALITY_FIGURATIVE] != 0
            or c_vec[MODALITY_FIGURATIVE] != 0
        ):
            return True

        # 2. Variable binding bypass: FOL variables (x, y) are typed at quantification/runtime
        if c_vec[GRAPH_VARIABLE_BIND] != 0:
            return True

        # 3. Merkle fold pointer bypass: Cryptographic folded sub-trees represent valid composite propositions
        if c_vec["GRAPH_MERKLE_FOLD_POINT"] != 0:
            return True

        slot_idx = _to_slot_idx(relation_slot)
        constraint = self._constraints.get(slot_idx)

        # If no constraint registered, default to True
        if constraint is None:
            return True

        # Run custom checker if registered
        if constraint.custom_checker is not None:
            if not constraint.custom_checker(p_vec, c_vec):
                return False

        # Check prohibited types
        if constraint.prohibited_types:
            for prohibited_slot in constraint.prohibited_types:
                if c_vec[prohibited_slot] == QuaternaryValue.TRUE:
                    # Check if there is an explicit allowed capability that overrides prohibited type
                    has_override = False
                    if constraint.allowed_types:
                        for allowed_slot in constraint.allowed_types:
                            if c_vec[allowed_slot] == QuaternaryValue.TRUE:
                                has_override = True
                                break
                    if not has_override:
                        return False

        return True


def validate_valency(
    parent_node_or_vector: Union[QuantaNode, QuantaVector, Dict[str, Any]],
    relation_slot: Union[str, int],
    child_node_or_vector: Union[QuantaNode, QuantaVector, Dict[str, Any]],
    registry: Optional[TypeConstraintRegistry] = None,
) -> bool:
    """Validates compile-time type constraints and valency signatures.

    Checks whether a child argument satisfies the ontological requirements of the predicate slot.
    Returns True if valid, False if rejected by constraint violation.
    """
    reg = registry or TypeConstraintRegistry.get_default()
    return reg.validate(parent_node_or_vector, relation_slot, child_node_or_vector)


__all__ = [
    "ValencyConstraint",
    "TypeConstraintRegistry",
    "validate_valency",
]
